#!/usr/bin/env python3
"""
exp_p5_ccdpsgd.py — Phase 5: Coherence-Channeled DP-SGD
==========================================================
Tests whether decomposing gradients into coherent (public-subspace-aligned)
and incoherent components—and clipping them asymmetrically—improves DP-SGD
utility at matched (ε, δ).

Algorithm (CC-DPSGD, phase5_spex.tex):
  1. Every SUBSPACE_REFRESH steps: compute top-k subspace V_k from 500
     public anchor examples (zero privacy cost — public data only).
  2. For each private example (averaged over aug_mult augmented views):
       g_par  = V_k V_k^T g   (coherent component, in the k-dim subspace)
       g_perp = g - g_par      (incoherent residual)
  3. Asymmetric clip: clip g_par at C_par, g_perp at C_perp.
  4. Sum clipped gradients + N(0, σ² C_max² I)  where C_max = sqrt(C_par²+C_perp²)

Privacy: identical to standard DP-SGD at noise multiplier σ (same σ, same C_max ≈ 1.0).
All configs share the same σ so formal ε is identical—differences in accuracy
come purely from better gradient signal allocation.

Run (Phase 1 — identify best config at ε=4, 1 seed each):
  python experiments/exp_p5_ccdpsgd.py --mode phase1

Run (Phase 2 — vanilla vs best CC at all ε, 3 seeds):
  python experiments/exp_p5_ccdpsgd.py --mode phase2 --best_config cc_k50_r3

Run (everything):
  python experiments/exp_p5_ccdpsgd.py --mode all

Analysis only (re-generate tables/figures from saved history):
  python experiments/exp_p5_ccdpsgd.py --analysis_only
"""

import os
import sys
import gc
import math
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import make_model
from src.dp_training import set_seed, evaluate
from src.privacy_accounting import compute_sigma

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_PUB            = 500       # public anchor set size (from CIFAR-10 test set)
EPOCHS           = 100
BATCH_SIZE       = 256
AUG_MULT         = 8
LR_START         = 0.5
WD               = 5e-4
DELTA            = 1e-5
EPS_VALUES       = [2.0, 4.0, 8.0]
N_SEEDS          = 3
SUBSPACE_REFRESH = 100       # steps between public subspace recomputations
GRAD_CHUNK       = 16        # per-example vmap chunk size
DATA_ROOT        = "./data"
RESULTS_DIR      = "./results/exp_p5"

# Six configurations from the spec table.
# C_max = sqrt(C_par² + C_perp²); all ≈ 1.0 → same σ → same formal ε.
CONFIGS = [
    {"name": "vanilla",      "k": None, "C_par": 1.0,  "C_perp": None},
    {"name": "cc_k10_r3",    "k": 10,  "C_par": 0.95, "C_perp": 0.32},
    {"name": "cc_k50_r3",    "k": 50,  "C_par": 0.95, "C_perp": 0.32},
    {"name": "cc_k10_r10",   "k": 10,  "C_par": 1.0,  "C_perp": 0.10},
    {"name": "cc_k50_r10",   "k": 50,  "C_par": 1.0,  "C_perp": 0.10},
    {"name": "cc_k50_proj",  "k": 50,  "C_par": 1.0,  "C_perp": 0.0},
]
for _c in CONFIGS:
    if _c["C_perp"] is None:
        _c["C_max"] = _c["C_par"]
    else:
        _c["C_max"] = math.sqrt(_c["C_par"] ** 2 + _c["C_perp"] ** 2)

CONFIG_BY_NAME = {c["name"]: c for c in CONFIGS}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(data_root=DATA_ROOT, pub_seed=42):
    """
    Load balanced CIFAR-10.
      train_ds : all 50K training examples (no augmentation; aug applied in-step)
      pub_ds   : 500 stratified test examples as public anchor (50 per class)
      eval_ds  : full 10K test set for accuracy evaluation
    """
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as T

    norm    = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    noaug   = T.Compose([T.ToTensor(), norm])

    train_ds = CIFAR10(data_root, train=True,  download=True, transform=noaug)
    test_ds  = CIFAR10(data_root, train=False, download=True, transform=noaug)

    # 500 stratified public examples from test set (50 per class, fixed seed)
    rng = np.random.RandomState(pub_seed)
    test_targets = np.array(test_ds.targets)
    pub_idx = []
    for c in range(10):
        cls_idx = np.where(test_targets == c)[0]
        pub_idx.extend(rng.choice(cls_idx, 50, replace=False).tolist())
    pub_idx = sorted(pub_idx)

    return {
        "train_ds":    train_ds,
        "pub_ds":      Subset(test_ds, pub_idx),
        "eval_ds":     test_ds,
        "num_classes": 10,
        "n_train":     len(train_ds),
    }


# ---------------------------------------------------------------------------
# Augmentation helper (same pattern as other experiments)
# ---------------------------------------------------------------------------

def _augment_batch(x):
    """Random crop (4 px reflect padding) + random horizontal flip."""
    B, _, H, W = x.shape
    xp   = F.pad(x, (4, 4, 4, 4), mode="reflect")
    oi   = torch.randint(0, 8, (B,), device=x.device)
    oj   = torch.randint(0, 8, (B,), device=x.device)
    rows = oi[:, None] + torch.arange(H, device=x.device)           # (B, H)
    cols = oj[:, None] + torch.arange(W, device=x.device)           # (B, W)
    bidx = torch.arange(B, device=x.device)[:, None, None]
    crops = (xp.permute(0, 2, 3, 1)[bidx, rows[:, :, None], cols[:, None, :]]
               .permute(0, 3, 1, 2).contiguous())
    flip = torch.rand(B, device=x.device) > 0.5
    crops[flip] = crops[flip].flip(-1)
    return crops


# ---------------------------------------------------------------------------
# Public subspace computation (zero privacy cost)
# ---------------------------------------------------------------------------

def _compute_public_subspace(model, pub_ds, k, device):
    """
    Compute top-k right singular vectors of the public gradient matrix.
    G_pub : (N_pub, d) — one gradient per public example.
    Returns V_k : (d, k) orthonormal tensor on device.
    """
    from torch.func import vmap, grad, functional_call

    params      = {n: p.detach() for n, p in model.named_parameters()}
    buffers     = {n: b.detach() for n, b in model.named_buffers()}
    param_names = list(params.keys())
    d           = sum(params[n].numel() for n in param_names)

    def loss_fn(p, xi, yi):
        out = functional_call(model, {**p, **buffers}, (xi.unsqueeze(0),))
        return F.cross_entropy(out, yi.unsqueeze(0))

    per_grad_fn = vmap(grad(loss_fn), in_dims=(None, 0, 0))
    use_pin     = device.type == "cuda"
    loader      = DataLoader(pub_ds, batch_size=50, shuffle=False,
                             num_workers=2, pin_memory=use_pin)

    G      = torch.zeros(len(pub_ds), d, device=device)
    offset = 0
    was_training = model.training
    model.eval()

    for batch in loader:
        x, y  = batch[0].to(device), batch[1].to(device)
        Bc    = x.shape[0]
        gd    = per_grad_fn(params, x, y)
        flat  = torch.cat([gd[n].reshape(Bc, -1) for n in param_names], dim=1)
        G[offset:offset + Bc] = flat.detach()
        del gd, flat
        offset += Bc

    if was_training:
        model.train()

    # G : (N_pub, d) → top-k right singular vectors via randomized SVD
    # torch.svd_lowrank(A, q=k) returns U (m,k), S (k,), V (n,k)  for A (m,n)
    _, _, V_k = torch.svd_lowrank(G, q=k, niter=4)   # V_k : (d, k)
    del G
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return V_k.contiguous()


# ---------------------------------------------------------------------------
# Core DP step
# ---------------------------------------------------------------------------

def _dp_step(model, x, y, V_k, cfg, sigma, n_train, device):
    """
    One CC-DPSGD or vanilla DP-SGD mini-batch step.

    Per-example gradients are averaged over AUG_MULT augmented views before
    clipping (augmentation multiplicity, matching Phase 3 approach).

    Returns
    -------
    agg   : (d,) tensor — sum of clipped per-example gradients + Gaussian noise.
            Caller divides by BATCH_SIZE to get the normalized gradient.
    stats : dict of clipping diagnostics.
    """
    from torch.func import vmap, grad, functional_call

    params      = {n: p.detach() for n, p in model.named_parameters()}
    buffers     = {n: b.detach() for n, b in model.named_buffers()}
    param_names = list(params.keys())
    d           = sum(params[n].numel() for n in param_names)

    def loss_fn(p, xi, yi):
        out = functional_call(model, {**p, **buffers}, (xi.unsqueeze(0),))
        return F.cross_entropy(out, yi.unsqueeze(0))

    per_grad_fn = vmap(grad(loss_fn), in_dims=(None, 0, 0))

    B     = x.shape[0]
    C_par = cfg["C_par"]
    C_perp = cfg["C_perp"]
    C_max = cfg["C_max"]
    k     = cfg["k"]

    agg = torch.zeros(d, device=device)
    stats = {"n": 0, "n_clipped_par": 0, "n_clipped_perp": 0,
             "norm_par": [], "norm_perp": []}

    for start in range(0, B, GRAD_CHUNK):
        x_c = x[start:start + GRAD_CHUNK]
        y_c = y[start:start + GRAD_CHUNK]
        Bc  = x_c.shape[0]

        # Accumulate AUG_MULT augmented views per example, then average
        g_c = torch.zeros(Bc, d, device=device)
        for _ in range(AUG_MULT):
            x_aug = _augment_batch(x_c)
            gd    = per_grad_fn(params, x_aug, y_c)
            flat  = torch.cat([gd[n].reshape(Bc, -1) for n in param_names], dim=1)
            g_c.add_(flat.detach())
            del gd, flat, x_aug
        g_c.div_(AUG_MULT)   # (Bc, d) — averaged per-example gradient

        if k is not None and V_k is not None:
            # ------- CC-DPSGD: asymmetric projection + clipping -------
            coeffs = g_c @ V_k          # (Bc, k)  — projection coefficients
            g_par  = coeffs @ V_k.T     # (Bc, d)  — coherent component
            g_perp = g_c - g_par        # (Bc, d)  — incoherent residual

            # Clip coherent at C_par
            norms_par  = g_par.norm(dim=1, keepdim=True).clamp(min=1e-8)
            scale_par  = (C_par / norms_par).clamp(max=1.0)
            g_par_clip = g_par * scale_par

            # Clip incoherent at C_perp (or zero-out if C_perp=0)
            norms_perp = g_perp.norm(dim=1, keepdim=True).clamp(min=1e-8)
            if C_perp > 0:
                scale_perp  = (C_perp / norms_perp).clamp(max=1.0)
                g_perp_clip = g_perp * scale_perp
            else:
                scale_perp  = torch.zeros(Bc, 1, device=device)
                g_perp_clip = torch.zeros_like(g_perp)

            agg.add_((g_par_clip + g_perp_clip).sum(dim=0))

            stats["n"]             += Bc
            stats["n_clipped_par"] += (scale_par.squeeze(1) < 1.0 - 1e-6).sum().item()
            stats["n_clipped_perp"]+= (scale_perp.squeeze(1) < 1.0 - 1e-6).sum().item()
            stats["norm_par"].append(norms_par.squeeze(1).detach().cpu())
            stats["norm_perp"].append(norms_perp.squeeze(1).detach().cpu())

            del coeffs, g_par, g_perp, g_par_clip, g_perp_clip, norms_par, norms_perp
        else:
            # ------- Vanilla DP-SGD: single clip at C_max -------
            norms = g_c.norm(dim=1, keepdim=True).clamp(min=1e-8)
            scale = (C_max / norms).clamp(max=1.0)
            agg.add_((g_c * scale).sum(dim=0))

            stats["n"]             += Bc
            stats["n_clipped_par"] += (scale.squeeze(1) < 1.0 - 1e-6).sum().item()
            stats["norm_par"].append(norms.squeeze(1).detach().cpu())

        del g_c
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Gaussian noise: N(0, σ² C_max² I_d)
    agg.add_(torch.randn_like(agg).mul_(sigma * C_max))

    return agg, stats


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def _train_run(cfg, eps, seed, data, device, out_dir):
    """
    Train one (config, ε, seed) combination and return history {epoch: acc}.
    Skips if a cached history file already exists.
    """
    name = cfg["name"]
    tag  = f"{name}_eps{eps:.0f}_seed{seed}"
    hpath = os.path.join(out_dir, f"{tag}_history.pkl")

    if os.path.exists(hpath):
        with open(hpath, "rb") as f:
            return pickle.load(f)

    print(f"\n[P5] === {tag} ===")
    set_seed(seed * 2000 + int(eps * 10))

    n_train = data["n_train"]
    q       = BATCH_SIZE / n_train
    steps_per_epoch = n_train // BATCH_SIZE          # drop_last=True
    T       = EPOCHS * steps_per_epoch

    # Noise multiplier: use C_max=1.0 reference so σ is identical for all configs
    # (C_max ≈ 1.0 for all; formal ε is identical by Sampled Gaussian Mechanism)
    sigma = compute_sigma(eps, DELTA, q, T)
    print(f"  n={n_train}, q={q:.5f}, T={T}, σ={sigma:.3f}, C_max={cfg['C_max']:.4f}")

    model = make_model("wrn28-2", num_classes=data["num_classes"]).to(device)
    opt   = torch.optim.SGD(model.parameters(), lr=LR_START,
                             momentum=0.0, weight_decay=WD)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    use_pin     = device.type == "cuda"
    train_loader = DataLoader(data["train_ds"], batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4,
                              pin_memory=use_pin, drop_last=True)
    eval_loader  = DataLoader(data["eval_ds"], batch_size=512,
                              shuffle=False, num_workers=4, pin_memory=use_pin)

    V_k   = None
    step  = 0
    history = {}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_stats = {"n": 0, "n_clipped_par": 0, "n_clipped_perp": 0}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Refresh public subspace every SUBSPACE_REFRESH steps
            if cfg["k"] is not None and step % SUBSPACE_REFRESH == 0:
                V_k = _compute_public_subspace(model, data["pub_ds"],
                                               cfg["k"], device)

            # Compute DP gradient: sum of clipped per-example grads + noise
            agg, stats = _dp_step(model, x, y, V_k, cfg, sigma, n_train, device)

            # Apply gradient update via optimizer (handles LR + weight decay)
            opt.zero_grad()
            with torch.no_grad():
                offset = 0
                for p in model.parameters():
                    numel = p.numel()
                    # grad = agg / BATCH_SIZE  (normalised gradient estimate)
                    p.grad = (agg[offset:offset + numel] / BATCH_SIZE
                              ).reshape(p.shape).clone()
                    offset += numel
            opt.step()

            # Accumulate epoch-level stats
            epoch_stats["n"]              += stats["n"]
            epoch_stats["n_clipped_par"]  += stats["n_clipped_par"]
            epoch_stats["n_clipped_perp"] += stats["n_clipped_perp"]

            del agg
            step += 1

        sch.step()

        if epoch % 10 == 0 or epoch == EPOCHS:
            acc = evaluate(model, eval_loader, device)
            history[epoch] = acc
            clip_frac_par = (epoch_stats["n_clipped_par"] /
                             max(epoch_stats["n"], 1))
            print(f"  epoch {epoch:>3}/{EPOCHS}: acc={acc:.4f}  "
                  f"clip_par={clip_frac_par:.3f}")

    # Save history and model
    with open(hpath, "wb") as f:
        pickle.dump(history, f)
    torch.save(model.state_dict(),
               os.path.join(out_dir, f"{tag}_model.pt"))
    print(f"  [P5] saved {tag}")
    return history


# ---------------------------------------------------------------------------
# Diagnostics: clipping stats + gradient norm distributions
# ---------------------------------------------------------------------------

def _compute_diagnostics(cfg, eps, seed, data, device, out_dir):
    """
    Run one forward pass over a subset of training data with the trained model
    to collect clipping statistics and ||g_par|| / ||g_perp|| distributions.
    Saves to {tag}_diag.pkl.  Skipped if not a CC config.
    """
    if cfg["k"] is None:
        return None

    name  = cfg["name"]
    tag   = f"{name}_eps{eps:.0f}_seed{seed}"
    dpath = os.path.join(out_dir, f"{tag}_diag.pkl")
    if os.path.exists(dpath):
        with open(dpath, "rb") as f:
            return pickle.load(f)

    model_path = os.path.join(out_dir, f"{tag}_model.pt")
    if not os.path.exists(model_path):
        print(f"  [P5-diag] model not found: {model_path}, skipping")
        return None

    print(f"  [P5-diag] computing diagnostics for {tag}")
    model = make_model("wrn28-2", num_classes=data["num_classes"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    V_k = _compute_public_subspace(model, data["pub_ds"], cfg["k"], device)

    # Run diagnostic on a 2000-example subset
    n_diag = 2000
    indices = np.random.choice(len(data["train_ds"]), n_diag, replace=False)
    diag_ds = Subset(data["train_ds"], indices.tolist())
    use_pin = device.type == "cuda"
    loader  = DataLoader(diag_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=use_pin, drop_last=False)

    from torch.func import vmap, grad, functional_call
    params      = {n: p.detach() for n, p in model.named_parameters()}
    buffers     = {n: b.detach() for n, b in model.named_buffers()}
    param_names = list(params.keys())
    d           = sum(params[n].numel() for n in param_names)

    def loss_fn(p, xi, yi):
        out = functional_call(model, {**p, **buffers}, (xi.unsqueeze(0),))
        return F.cross_entropy(out, yi.unsqueeze(0))

    per_grad_fn = vmap(grad(loss_fn), in_dims=(None, 0, 0))

    all_norm_par, all_norm_perp = [], []

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        B = x.shape[0]
        g_b = torch.zeros(B, d, device=device)
        for _ in range(1):   # single aug view for diagnostics (speed)
            gd   = per_grad_fn(params, x, y)
            flat = torch.cat([gd[n].reshape(B, -1) for n in param_names], dim=1)
            g_b.add_(flat.detach())
            del gd, flat

        coeffs     = g_b @ V_k
        g_par      = coeffs @ V_k.T
        g_perp     = g_b - g_par
        all_norm_par.append(g_par.norm(dim=1).cpu())
        all_norm_perp.append(g_perp.norm(dim=1).cpu())
        del g_b, g_par, g_perp

    diag = {
        "norm_par":  torch.cat(all_norm_par).numpy(),
        "norm_perp": torch.cat(all_norm_perp).numpy(),
        "C_par":  cfg["C_par"],
        "C_perp": cfg["C_perp"],
        "C_max":  cfg["C_max"],
    }
    with open(dpath, "wb") as f:
        pickle.dump(diag, f)
    return diag


# ---------------------------------------------------------------------------
# Analysis: tables and figures
# ---------------------------------------------------------------------------

def _load_history(name, eps, seed, out_dir):
    path = os.path.join(out_dir, f"{name}_eps{eps:.0f}_seed{seed}_history.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _final_acc(history):
    if history is None:
        return None
    return history[max(history.keys())]


def _make_tab1(names, eps_list, seeds, out_dir):
    """Tab 1: test accuracy (mean ± std) for all configs at each ε."""
    lines = ["=== Tab 1: Test Accuracy (mean ± std over seeds) ===", ""]
    header = f"{'Config':<18}" + "".join(f"  ε={e:.0f}:mean±std" for e in eps_list)
    lines.append(header)
    lines.append("-" * (18 + len(eps_list) * 22))

    all_results = {}
    for name in names:
        row = f"{name:<18}"
        all_results[name] = {}
        for eps in eps_list:
            accs = [_final_acc(_load_history(name, eps, s, out_dir))
                    for s in range(seeds)]
            accs = [a for a in accs if a is not None]
            if accs:
                mu  = np.mean(accs) * 100
                std = np.std(accs)  * 100
                row += f"  {mu:>6.2f}±{std:.2f}    "
                all_results[name][eps] = (float(mu), float(std))
            else:
                row += f"  {'N/A':>13}    "
                all_results[name][eps] = None
        lines.append(row)

    tab = "\n".join(lines)
    print(tab)
    with open(os.path.join(out_dir, "tab1.txt"), "w") as f:
        f.write(tab)
    return all_results


def _make_fig1(names, eps_list, seeds, out_dir):
    """Fig 1: bar chart of accuracy per ε, configs side-by-side."""
    n_eps = len(eps_list)
    fig, axes = plt.subplots(1, n_eps, figsize=(5 * n_eps, 5), sharey=True)
    if n_eps == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(names)))

    for ax, eps in zip(axes, eps_list):
        xs = np.arange(len(names))
        for i, name in enumerate(names):
            accs = [_final_acc(_load_history(name, eps, s, out_dir))
                    for s in range(seeds)]
            accs = [a for a in accs if a is not None]
            if not accs:
                continue
            mu  = np.mean(accs) * 100
            std = np.std(accs)  * 100
            ax.bar(xs[i], mu, yerr=std, color=colors[i], capsize=4, alpha=0.85,
                   label=name if ax is axes[0] else "")
        ax.set_title(f"ε = {eps:.0f}", fontsize=13)
        ax.set_xticks(xs)
        ax.set_xticklabels([n.replace("_", "\n") for n in names],
                           fontsize=7, rotation=0)
        ax.set_ylabel("Test accuracy (%)" if ax is axes[0] else "")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("CC-DPSGD vs Vanilla DP-SGD\n(all configs at same ε, δ, σ)",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_bar_chart.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[P5] Fig 1 saved to {path}")


def _make_fig2(vanilla_name, cc_name, eps_list, out_dir, seed=0):
    """Fig 2: learning curves — vanilla vs best CC at selected ε."""
    fig, axes = plt.subplots(1, len(eps_list), figsize=(5 * len(eps_list), 4),
                             sharey=True)
    if len(eps_list) == 1:
        axes = [axes]

    for ax, eps in zip(axes, eps_list):
        for name, label, ls in [(vanilla_name, "Vanilla DP-SGD", "-"),
                                 (cc_name, f"CC ({cc_name})", "--")]:
            h = _load_history(name, eps, seed, out_dir)
            if h is None:
                continue
            epochs = sorted(h.keys())
            accs   = [h[e] * 100 for e in epochs]
            ax.plot(epochs, accs, ls=ls, label=label, linewidth=1.8)
        ax.set_title(f"ε = {eps:.0f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy (%)" if ax is axes[0] else "")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Learning Curves: Vanilla vs Best CC-DPSGD", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_learning_curves.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[P5] Fig 2 saved to {path}")


def _make_tab2(names, eps, seeds, out_dir):
    """Tab 2 (P1): clipping fraction statistics for CC configs."""
    lines = ["=== Tab 2: Clipping Statistics ===",
             f"At ε={eps:.0f}, seed=0 (diagnostic run)",
             f"{'Config':<18}  frac_par_clipped  frac_perp_clipped  "
             f"mean_norm_par  mean_norm_perp  C_par  C_perp"]
    lines.append("-" * 95)

    for name in names:
        if name == "vanilla":
            continue
        dpath = os.path.join(out_dir, f"{name}_eps{eps:.0f}_seed0_diag.pkl")
        if not os.path.exists(dpath):
            lines.append(f"{name:<18}  (no diagnostic data)")
            continue
        with open(dpath, "rb") as f:
            d = pickle.load(f)
        C_par, C_perp = d["C_par"], d["C_perp"]
        np_ = d["norm_par"]
        npe = d["norm_perp"]
        frac_par  = (np_ > C_par).mean()  if C_par  > 0 else 0.0
        frac_perp = (npe > C_perp).mean() if C_perp > 0 else 1.0
        lines.append(
            f"{name:<18}  {frac_par:>16.4f}  {frac_perp:>17.4f}  "
            f"{np_.mean():>13.4f}  {npe.mean():>14.4f}  "
            f"{C_par:>5.2f}  {C_perp:>6.2f}"
        )

    tab = "\n".join(lines)
    print(tab)
    with open(os.path.join(out_dir, "tab2.txt"), "w") as f:
        f.write(tab)


def _make_fig3(names, eps, out_dir):
    """Fig 3 (P1): histogram of ||g_par|| and ||g_perp|| for CC configs."""
    n_cc = sum(1 for n in names if n != "vanilla")
    if n_cc == 0:
        return
    fig, axes = plt.subplots(1, n_cc, figsize=(5 * n_cc, 4))
    if n_cc == 1:
        axes = [axes]

    ax_idx = 0
    for name in names:
        if name == "vanilla":
            continue
        dpath = os.path.join(out_dir, f"{name}_eps{eps:.0f}_seed0_diag.pkl")
        if not os.path.exists(dpath):
            ax_idx += 1
            continue
        with open(dpath, "rb") as f:
            d = pickle.load(f)
        ax = axes[ax_idx]
        ax.hist(d["norm_par"],  bins=50, alpha=0.6, label="||g_par||",  density=True)
        ax.hist(d["norm_perp"], bins=50, alpha=0.6, label="||g_perp||", density=True)
        ax.axvline(d["C_par"],  color="C0", ls="--", lw=1.5,
                   label=f"C_par={d['C_par']:.2f}")
        if d["C_perp"] > 0:
            ax.axvline(d["C_perp"], color="C1", ls="--", lw=1.5,
                       label=f"C_perp={d['C_perp']:.2f}")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("gradient norm")
        ax.set_ylabel("density" if ax_idx == 0 else "")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax_idx += 1

    fig.suptitle(f"Gradient Norm Distributions at Convergence (ε={eps:.0f})",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_norm_histograms.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[P5] Fig 3 saved to {path}")


def _make_success_table(names, eps_list, seeds, out_dir):
    """Print S1/S2/S3 success criteria against vanilla."""
    vanilla_accs = {}
    for eps in eps_list:
        accs = [_final_acc(_load_history("vanilla", eps, s, out_dir))
                for s in range(seeds)]
        accs = [a for a in accs if a is not None]
        vanilla_accs[eps] = np.mean(accs) * 100 if accs else None

    print("\n=== Success Criteria ===")
    print(f"{'Config':<18}  " + "  ".join(f"Δacc@ε={e:.0f}" for e in eps_list)
          + "  S1(≥2pp)  S2(pos≥2/3ε)  S3(vs sym)")
    print("-" * 95)

    for name in names:
        if name == "vanilla":
            continue
        gaps = []
        for eps in eps_list:
            accs = [_final_acc(_load_history(name, eps, s, out_dir))
                    for s in range(seeds)]
            accs = [a for a in accs if a is not None]
            if not accs or vanilla_accs.get(eps) is None:
                gaps.append(None)
            else:
                gaps.append(np.mean(accs) * 100 - vanilla_accs[eps])

        valid_gaps = [g for g in gaps if g is not None]
        s1 = any(g >= 2.0 for g in valid_gaps)
        s2 = sum(1 for g in valid_gaps if g and g > 0) >= 2
        gap_str = "  ".join(f"{g:+.2f}" if g is not None else " N/A " for g in gaps)
        print(f"{name:<18}  {gap_str}  {'PASS' if s1 else 'FAIL':>8}  "
              f"{'PASS' if s2 else 'FAIL':>11}")


def _run_analysis(names, eps_list, seeds, out_dir, best_cc=None):
    """Generate all outputs from existing history files."""
    print("\n[P5] === Analysis ===")
    _make_tab1(names, eps_list, seeds, out_dir)
    _make_fig1(names, eps_list, seeds, out_dir)

    if best_cc and "vanilla" in names and best_cc in names:
        _make_fig2("vanilla", best_cc, eps_list, out_dir)
    elif len(names) >= 2:
        cc_names = [n for n in names if n != "vanilla"]
        if cc_names:
            _make_fig2("vanilla", cc_names[0], eps_list, out_dir)

    # P1 diagnostics (if available)
    _make_tab2(names, eps_list[0], seeds, out_dir)
    _make_fig3(names, eps_list[0], out_dir)
    _make_success_table(names, eps_list, seeds, out_dir)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_exp_p5(
    mode          = "phase1",
    best_config   = None,
    eps_override  = None,
    n_seeds       = 1,
    gpu           = None,
    results_dir   = RESULTS_DIR,
    analysis_only = False,
):
    if gpu is not None:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P5] device={device}, mode={mode}")

    out_dir = os.path.join(results_dir, "cifar10_balanced")
    os.makedirs(out_dir, exist_ok=True)

    # Determine which configs/ε/seeds to run
    if mode == "phase1":
        run_configs = CONFIGS                    # all 6 configs
        run_eps     = [4.0]
        run_seeds   = 1
    elif mode == "phase2":
        bc = best_config or "cc_k50_r3"
        run_configs = [CONFIG_BY_NAME["vanilla"], CONFIG_BY_NAME[bc]]
        run_eps     = EPS_VALUES
        run_seeds   = n_seeds
    else:  # "all"
        run_configs = CONFIGS
        run_eps     = EPS_VALUES
        run_seeds   = n_seeds

    if eps_override:
        run_eps = eps_override

    names = [c["name"] for c in run_configs]
    print(f"[P5] configs={names}, ε={run_eps}, seeds=range({run_seeds})")

    if not analysis_only:
        data = _load_data()
        print(f"[P5] n_train={data['n_train']}, n_pub={len(data['pub_ds'])}, "
              f"n_eval={len(data['eval_ds'])}")

        for cfg in run_configs:
            for eps in run_eps:
                for seed in range(run_seeds):
                    _train_run(cfg, eps, seed, data, device, out_dir)

        # P1 diagnostics for CC configs at first ε value
        if mode in ("phase1", "all"):
            print("\n[P5] Computing diagnostics...")
            for cfg in run_configs:
                _compute_diagnostics(cfg, run_eps[0], 0, data, device, out_dir)

    _run_analysis(names, run_eps, run_seeds, out_dir, best_config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5: Coherence-Channeled DP-SGD"
    )
    parser.add_argument("--mode", choices=["phase1", "phase2", "all"],
                        default="phase1",
                        help="phase1=all configs@eps4 1-seed; "
                             "phase2=vanilla+best@all-eps N-seeds; "
                             "all=everything")
    parser.add_argument("--best_config", type=str, default=None,
                        help="Name of best CC config to use in phase2 "
                             "(e.g. cc_k50_r3). Identified from phase1 results.")
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS,
                        help="Number of seeds per config (default 3)")
    parser.add_argument("--eps", type=float, nargs="+", default=None,
                        help="Override ε values (e.g. --eps 4)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index (default: auto-detect)")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--analysis_only", action="store_true",
                        help="Skip training, regenerate outputs from saved histories")
    args = parser.parse_args()

    run_exp_p5(
        mode          = args.mode,
        best_config   = args.best_config,
        eps_override  = args.eps,
        n_seeds       = args.n_seeds,
        gpu           = args.gpu,
        results_dir   = args.results_dir,
        analysis_only = args.analysis_only,
    )
