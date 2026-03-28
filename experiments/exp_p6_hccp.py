#!/usr/bin/env python3
"""
exp_p6_hccp.py — Phase 6: HCCP-SGD End-to-End Evaluation
===========================================================
History-Conditioned Coherence Preprocessing DP-SGD (HCCP-SGD).

Algorithm:
  At each step, maintain an anchor direction a_t (unit vector).
  For each per-sample gradient g_j:
    1. Decompose:  g_par = (g_j · a_t) * a_t
                  g_perp = g_j - g_par
    2. Suppress:   g_perp_clip = clip(g_perp, C_perp)   where C_perp = C / r
    3. Recombine:  g_processed = g_par + g_perp_clip
    4. Standard clip:  g_final = clip(g_processed, C)
  Sum all g_final, add N(0, σ²C²I).
  Update EMA:  v_{t+1} = γ·v_t + (1-γ)·noised_agg
               a_{t+1} = v_{t+1} / ||v_{t+1}||

For oracle arms: a_t = mean(raw per-sample grads) / ||mean|| (not formally private).
For vanilla: no decomposition, standard clip only.

Seven arms × 3 ε × 3 seeds = 63 runs (≈47–73 GPU-hours on A100).

Run single arm:
  python experiments/exp_p6_hccp.py --arm vanilla --eps 4.0 --seed 0

Run all arms for one ε (parallel across 7 terminals):
  for ARM in vanilla hccp_r2_g9 hccp_r2_g0 hccp_r5_g9 hccp_proj_g9 oracle_r2 oracle_proj; do
      python experiments/exp_p6_hccp.py --arm $ARM --eps 4.0 --seed 0 &
  done

Analysis only (generate tables/figures from existing CSVs):
  python experiments/exp_p6_hccp.py --analysis_only
"""

import os
import sys
import csv
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

from src.models import WideResNet
from src.dp_training import set_seed, evaluate
from src.privacy_accounting import compute_sigma

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS       = 60
BATCH_SIZE   = 4096           # logical batch size
LR           = 2.0
WD           = 0.0            # no weight decay for DP-SGD (per spec)
WARMUP_EPOCHS= 5
C            = 1.0            # clipping norm
DELTA        = 1e-5
EPS_VALUES   = [8.0, 4.0, 2.0]
N_SEEDS      = 3
N_GROUPS     = 16             # GroupNorm groups for WRN-16-4
GRAD_CHUNK   = 32             # vmap chunk size for per-sample grads
N_PROBE      = 1000           # examples in train-loss probe set
DATA_ROOT    = "./data"
RESULTS_DIR  = "./results/exp_p6"

# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------
#   r=None  → vanilla (no decomposition)
#   r=inf   → full projection (discard incoherent)
#   gamma   → EMA momentum (None for oracle arms)
#   oracle  → use batch aggregate as anchor (not formally private)

ARMS = {
    "vanilla":      {"r": None,       "gamma": None,  "oracle": False},
    "hccp_r2_g9":   {"r": 2,          "gamma": 0.9,   "oracle": False},
    "hccp_r2_g0":   {"r": 2,          "gamma": 0.0,   "oracle": False},
    "hccp_r5_g9":   {"r": 5,          "gamma": 0.9,   "oracle": False},
    "hccp_proj_g9": {"r": float("inf"),"gamma": 0.9,  "oracle": False},
    "oracle_r2":    {"r": 2,          "gamma": None,  "oracle": True},
    "oracle_proj":  {"r": float("inf"),"gamma": None, "oracle": True},
}


# ---------------------------------------------------------------------------
# Model factory: WRN-16-4 with GroupNorm
# ---------------------------------------------------------------------------

def _make_model(num_classes=10):
    """WideResNet-16-4 with GroupNorm(16). Canonical DP-SGD architecture."""
    return WideResNet(depth=16, widen_factor=4,
                      num_classes=num_classes, n_groups=N_GROUPS)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_data(data_root=DATA_ROOT, probe_seed=99):
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as T

    norm    = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    aug_tf  = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
                         T.ToTensor(), norm])
    noaug   = T.Compose([T.ToTensor(), norm])

    train_aug   = CIFAR10(data_root, train=True,  download=True, transform=aug_tf)
    train_noaug = CIFAR10(data_root, train=True,  download=True, transform=noaug)
    test_ds     = CIFAR10(data_root, train=False, download=True, transform=noaug)

    # Fixed 1000-example probe set for train-loss diagnostics
    rng = np.random.RandomState(probe_seed)
    probe_idx = rng.choice(len(train_noaug), N_PROBE, replace=False).tolist()

    return {
        "train_aug":   train_aug,
        "train_noaug": train_noaug,
        "probe_ds":    Subset(train_noaug, probe_idx),
        "test_ds":     test_ds,
        "num_classes": 10,
        "n_train":     len(train_aug),
    }


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay (epoch-level)
# ---------------------------------------------------------------------------

def _make_scheduler(opt, warmup_epochs, total_epochs):
    def lr_lambda(epoch):                          # epoch is 0-indexed
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ---------------------------------------------------------------------------
# Per-sample gradient helper
# ---------------------------------------------------------------------------

def _per_sample_grad_fn_factory(model):
    """Return (params, buffers, param_names, d, per_grad_fn)."""
    from torch.func import vmap, grad, functional_call
    params      = {n: p.detach() for n, p in model.named_parameters()}
    buffers     = {n: b.detach() for n, b in model.named_buffers()}
    param_names = list(params.keys())
    d           = sum(params[n].numel() for n in param_names)

    def loss_fn(p, xi, yi):
        out = functional_call(model, {**p, **buffers}, (xi.unsqueeze(0),))
        return F.cross_entropy(out, yi.unsqueeze(0))

    per_grad_fn = vmap(grad(loss_fn), in_dims=(None, 0, 0))
    return params, buffers, param_names, d, per_grad_fn


# ---------------------------------------------------------------------------
# Core HCCP-SGD step
# ---------------------------------------------------------------------------

def _hccp_step(model, x, y, a_t, arm_cfg, sigma, device):
    """
    One HCCP-SGD (or vanilla) DP step.

    Parameters
    ----------
    x, y    : current batch (B, C, H, W) and labels, on device
    a_t     : current anchor direction (d,) on device — ignored for vanilla
    arm_cfg : dict with keys 'r', 'gamma', 'oracle'
    sigma   : noise multiplier

    Returns
    -------
    noised_agg : (d,) — clipped sum + noise (NOT divided by B)
    diag       : dict of per-step diagnostics
    raw_mean   : (d,) — mean of raw per-sample grads (for oracle anchor update)
    """
    from torch.func import vmap, grad, functional_call

    B  = x.shape[0]
    r  = arm_cfg["r"]
    is_vanilla = (r is None)
    C_perp = 0.0 if (r is not None and math.isinf(r)) else (C / r if r is not None else C)

    params, buffers, param_names, d, per_grad_fn = _per_sample_grad_fn_factory(model)

    # Running accumulators (on device)
    sum_raw     = torch.zeros(d, device=device)   # Σ g_j (unclipped)
    sum_clipped = torch.zeros(d, device=device)   # Σ g_final_j (after HCCP + clip)

    for start in range(0, B, GRAD_CHUNK):
        xc = x[start:start + GRAD_CHUNK]
        yc = y[start:start + GRAD_CHUNK]
        Bc = xc.shape[0]

        # Per-sample gradients
        gd   = per_grad_fn(params, xc, yc)
        flat = torch.cat([gd[n].reshape(Bc, -1) for n in param_names], dim=1)
        # flat: (Bc, d) — raw per-sample gradients
        del gd

        sum_raw.add_(flat.sum(dim=0))

        if is_vanilla:
            # Vanilla: standard clip at C
            norms    = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
            g_final  = flat * (C / norms).clamp(max=1.0)
        else:
            # HCCP decomposition + suppression
            proj    = (flat * a_t.unsqueeze(0)).sum(dim=1, keepdim=True)   # (Bc, 1)
            g_par   = proj * a_t.unsqueeze(0)                               # (Bc, d)
            g_perp  = flat - g_par                                          # (Bc, d)

            if C_perp > 0:
                perp_norms   = g_perp.norm(dim=1, keepdim=True).clamp(min=1e-8)
                g_perp_clip  = g_perp * (C_perp / perp_norms).clamp(max=1.0)
            else:
                g_perp_clip  = torch.zeros_like(g_perp)   # C_perp=0: discard

            g_processed = g_par + g_perp_clip

            # Standard clip at C (preserves formal DP guarantee)
            proc_norms = g_processed.norm(dim=1, keepdim=True).clamp(min=1e-8)
            g_final    = g_processed * (C / proc_norms).clamp(max=1.0)

        sum_clipped.add_(g_final.sum(dim=0))
        del flat, g_final

    # Gaussian noise: N(0, σ² C² I_d)
    noise      = torch.randn_like(sum_clipped).mul_(sigma * C)
    noised_agg = sum_clipped + noise

    # Diagnostics (cheap dot products on already-computed tensors)
    raw_mean    = sum_raw / B                                  # true batch aggregate
    raw_norm    = raw_mean.norm().clamp(min=1e-8)

    agg_norm    = sum_clipped.norm().item()                    # signal strength

    cos_m4      = F.cosine_similarity(                         # M4: noiseless cosine
        sum_clipped.unsqueeze(0), sum_raw.unsqueeze(0)
    ).item()
    cos_noised  = F.cosine_similarity(                         # Cosim: noised cosine
        noised_agg.unsqueeze(0), sum_raw.unsqueeze(0)
    ).item()

    anc_cosim = 0.0
    if a_t is not None and not is_vanilla:
        raw_dir   = raw_mean / raw_norm
        anc_cosim = F.cosine_similarity(
            a_t.unsqueeze(0), raw_dir.unsqueeze(0)
        ).item()

    diag = {
        "anchor_batch_cosim": anc_cosim,
        "clip_agg_norm":      agg_norm,
        "m4":                 cos_m4,
        "cosim":              cos_noised,
    }

    return noised_agg, diag, raw_mean.detach()


# ---------------------------------------------------------------------------
# Anchor management
# ---------------------------------------------------------------------------

def _init_anchor(d, device, seed=0):
    """Random unit vector, same seed across all HCCP arms."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    v = torch.randn(d, device=device, generator=g)
    return v, v / v.norm().clamp(min=1e-8)   # (v_t, a_t)


def _update_anchor(v_t, noised_agg, gamma):
    """EMA update: v ← γv + (1-γ)·noised; a = v/||v||."""
    v_new = gamma * v_t + (1.0 - gamma) * noised_agg
    a_new = v_new / v_new.norm().clamp(min=1e-8)
    return v_new, a_new


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------

def _train_run(arm_name, eps, seed, data, device, out_dir):
    """
    Train one (arm, ε, seed) combination.
    Saves CSV log and model checkpoint. Returns final test accuracy.
    Skips silently if already done.
    """
    tag      = f"{arm_name}_eps{eps:.0f}_seed{seed}"
    csv_path = os.path.join(out_dir, f"{tag}.csv")
    if os.path.exists(csv_path):
        print(f"[P6] {tag}: already done, loading CSV")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return float(rows[-1]["test_acc"])

    print(f"\n[P6] === {tag} ===")
    arm_cfg  = ARMS[arm_name]
    set_seed(seed * 3000 + int(eps * 10))

    n_train         = data["n_train"]
    steps_per_epoch = n_train // BATCH_SIZE
    T               = EPOCHS * steps_per_epoch
    q               = BATCH_SIZE / n_train
    sigma           = compute_sigma(eps, DELTA, q, T)
    print(f"  n={n_train}, B={BATCH_SIZE}, q={q:.5f}, T={T}, σ={sigma:.4f}")

    # Model + optimizer + scheduler
    model = _make_model(data["num_classes"]).to(device)
    d     = sum(p.numel() for p in model.parameters())
    print(f"  model: WRN-16-4, d={d:,}")

    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.0, weight_decay=WD)
    sch = _make_scheduler(opt, WARMUP_EPOCHS, EPOCHS)

    use_pin = device.type == "cuda"
    train_loader = DataLoader(data["train_aug"], batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4,
                              pin_memory=use_pin, drop_last=True)
    eval_loader  = DataLoader(data["test_ds"], batch_size=512,
                              shuffle=False, num_workers=4, pin_memory=use_pin)
    probe_loader = DataLoader(data["probe_ds"], batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=use_pin)

    # Anchor initialisation
    is_hccp  = (arm_cfg["r"] is not None and not arm_cfg["oracle"])
    is_oracle= arm_cfg["oracle"]
    v_t, a_t = _init_anchor(d, device, seed=0) if (is_hccp or is_oracle) else (None, None)

    # CSV writer
    fieldnames = ["epoch", "train_loss", "test_acc",
                  "anchor_batch_cosim", "clip_agg_norm", "m4", "cosim"]
    csv_file   = open(csv_path, "w", newline="")
    writer     = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Epoch-level diagnostic accumulators
        ep_diag = {k: [] for k in ["anchor_batch_cosim", "clip_agg_norm", "m4", "cosim"]}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # For oracle arm: compute batch aggregate direction (not private)
            if is_oracle:
                # Use the raw gradients of this batch as the oracle anchor
                # We get this from _hccp_step's raw_mean return value
                pass   # a_t is set below from raw_mean

            noised_agg, step_diag, raw_mean = _hccp_step(
                model, x, y, a_t, arm_cfg, sigma, device
            )

            # For oracle arms, update anchor to current batch aggregate (not private)
            if is_oracle:
                raw_norm = raw_mean.norm().clamp(min=1e-8)
                a_t      = raw_mean / raw_norm
                # For diagnostics, anchor-batch cosine is 1.0 by construction
                step_diag["anchor_batch_cosim"] = 1.0

            # Apply gradient update: grad = noised_agg / B
            opt.zero_grad()
            with torch.no_grad():
                offset = 0
                for p in model.parameters():
                    numel = p.numel()
                    p.grad = (noised_agg[offset:offset + numel] / BATCH_SIZE
                              ).reshape(p.shape).clone()
                    offset += numel
            opt.step()

            # EMA anchor update (HCCP arms only)
            if is_hccp:
                v_t, a_t = _update_anchor(v_t, noised_agg.detach(), arm_cfg["gamma"])

            # Accumulate diagnostics
            for k in ep_diag:
                ep_diag[k].append(step_diag[k])

            del noised_agg, raw_mean
            if device.type == "cuda":
                torch.cuda.empty_cache()

        sch.step()

        # Epoch-end evaluation
        model.eval()
        test_acc = evaluate(model, eval_loader, device)

        # Train loss on probe set (no noise, no clipping)
        probe_loss = 0.0
        n_probe    = 0
        with torch.no_grad():
            for xp, yp in probe_loader:
                xp, yp = xp.to(device), yp.to(device)
                loss    = F.cross_entropy(model(xp), yp, reduction="sum")
                probe_loss += loss.item()
                n_probe    += xp.shape[0]
        train_loss = probe_loss / max(n_probe, 1)

        # Mean diagnostics over steps
        row = {
            "epoch":              epoch,
            "train_loss":         f"{train_loss:.4f}",
            "test_acc":           f"{test_acc:.4f}",
            "anchor_batch_cosim": f"{np.mean(ep_diag['anchor_batch_cosim']):.4f}",
            "clip_agg_norm":      f"{np.mean(ep_diag['clip_agg_norm']):.4f}",
            "m4":                 f"{np.mean(ep_diag['m4']):.4f}",
            "cosim":              f"{np.mean(ep_diag['cosim']):.4f}",
        }
        writer.writerow(row)
        csv_file.flush()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}_best.pt"))

        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"  ep {epoch:>3}/{EPOCHS}: test={test_acc:.4f}  "
                  f"best={best_acc:.4f}  "
                  f"anchor_cos={row['anchor_batch_cosim']}  "
                  f"m4={row['m4']}  cosim={row['cosim']}")

    csv_file.close()
    torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}_final.pt"))
    print(f"  [P6] done — final={test_acc:.4f}  best={best_acc:.4f}")
    return test_acc


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _load_csv(arm, eps, seed, out_dir):
    path = os.path.join(out_dir, f"{arm}_eps{eps:.0f}_seed{seed}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def _final_acc(rows):
    if rows is None: return None
    return float(rows[-1]["test_acc"])


def _best_acc(rows):
    if rows is None: return None
    return max(float(r["test_acc"]) for r in rows)


def _epoch_series(rows, field):
    if rows is None: return None, None
    epochs = [int(r["epoch"]) for r in rows]
    vals   = [float(r[field]) for r in rows]
    return epochs, vals


# ---------------------------------------------------------------------------
# Tab 1: final accuracy table
# ---------------------------------------------------------------------------

def _make_tab1(arm_names, eps_list, n_seeds, out_dir):
    lines = ["=== Tab 1: Test Accuracy at Convergence (epoch 60) ===",
             "Mean ± std over seeds. Oracle arms marked (*).", ""]
    header = f"{'Arm':<18}" + "".join(f"  ε={e:.0f}:mean±std" for e in eps_list)
    lines.append(header); lines.append("-" * (18 + 22 * len(eps_list)))

    all_accs = {}
    for arm in arm_names:
        all_accs[arm] = {}
        row = f"{arm:<18}"
        marker = " (*)" if ARMS[arm]["oracle"] else ""
        for eps in eps_list:
            accs = [_final_acc(_load_csv(arm, eps, s, out_dir))
                    for s in range(n_seeds)]
            accs = [a for a in accs if a is not None]
            if accs:
                mu  = np.mean(accs) * 100
                std = np.std(accs)  * 100
                row += f"  {mu:>6.2f}±{std:.2f}    "
                all_accs[arm][eps] = (float(mu), float(std), accs)
            else:
                row += f"  {'N/A':>13}    "
                all_accs[arm][eps] = None
        lines.append(row + marker)

    # Best accuracy block
    lines.append(""); lines.append("=== Best Test Accuracy (any epoch) ===")
    lines.append(header); lines.append("-" * (18 + 22 * len(eps_list)))
    for arm in arm_names:
        marker = " (*)" if ARMS[arm]["oracle"] else ""
        row = f"{arm:<18}"
        for eps in eps_list:
            accs = [_best_acc(_load_csv(arm, eps, s, out_dir))
                    for s in range(n_seeds)]
            accs = [a for a in accs if a is not None]
            if accs:
                mu = np.mean(accs) * 100; std = np.std(accs) * 100
                row += f"  {mu:>6.2f}±{std:.2f}    "
            else:
                row += f"  {'N/A':>13}    "
        lines.append(row + marker)

    tab = "\n".join(lines)
    print(tab)
    with open(os.path.join(out_dir, "tab1.txt"), "w") as f:
        f.write(tab)
    return all_accs


# ---------------------------------------------------------------------------
# Fig 1: test accuracy learning curves at ε=4
# ---------------------------------------------------------------------------

def _make_fig1(arm_names, eps, n_seeds, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(arm_names)))
    styles = {arm: ("-" if arm == "vanilla" else
                    (":" if ARMS[arm]["oracle"] else "--"))
              for arm in arm_names}

    for arm, color in zip(arm_names, colors):
        epoch_arrs, acc_arrs = [], []
        for s in range(n_seeds):
            rows = _load_csv(arm, eps, s, out_dir)
            ep, accs = _epoch_series(rows, "test_acc")
            if ep is not None:
                epoch_arrs.append(ep); acc_arrs.append(accs)
        if not epoch_arrs: continue
        epochs = epoch_arrs[0]
        mu  = np.mean([[float(v) * 100 for v in a] for a in acc_arrs], axis=0)
        std = np.std( [[float(v) * 100 for v in a] for a in acc_arrs], axis=0)
        lbl = arm + (" [oracle*]" if ARMS[arm]["oracle"] else "")
        ax.plot(epochs, mu, ls=styles[arm], color=color, label=lbl, lw=1.8)
        if len(acc_arrs) > 1:
            ax.fill_between(epochs, mu - std, mu + std, color=color, alpha=0.12)

    ax.set_xlabel("Epoch"); ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Fig 1: HCCP-SGD vs Vanilla DP-SGD  (ε={eps:.0f})\n"
                 "*oracle arms have no formal DP guarantee")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"fig1_acc_eps{eps:.0f}.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"[P6] Fig 1 saved: {path}")


# ---------------------------------------------------------------------------
# Fig 2: anchor-batch cosine vs epoch (HCCP arms)
# ---------------------------------------------------------------------------

def _make_fig2(arm_names, eps, n_seeds, out_dir):
    hccp_arms = [a for a in arm_names
                 if ARMS[a]["r"] is not None and not ARMS[a]["oracle"]]
    if not hccp_arms: return
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(hccp_arms)))

    for arm, color in zip(hccp_arms, colors):
        for s in range(n_seeds):
            rows = _load_csv(arm, eps, s, out_dir)
            ep, vals = _epoch_series(rows, "anchor_batch_cosim")
            if ep is None: continue
            alpha = 0.9 if s == 0 else 0.4
            lw    = 1.8 if s == 0 else 0.8
            ax.plot(ep, vals, color=color, alpha=alpha, lw=lw,
                    label=arm if s == 0 else None)

    ax.axhline(0.5, color="gray", ls="--", lw=1, label="threshold=0.5")
    ax.set_xlabel("Epoch"); ax.set_ylabel("cos(a_t, true batch direction)")
    ax.set_title(f"Fig 2: Anchor–Batch Cosine (ε={eps:.0f})")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"fig2_anchor_cos_eps{eps:.0f}.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"[P6] Fig 2 saved: {path}")


# ---------------------------------------------------------------------------
# Fig 3: diagnostic trajectories (cosim, M4, clip_agg_norm) at ε=4
# ---------------------------------------------------------------------------

def _make_fig3(arm_names, eps, n_seeds, out_dir):
    fields = [("cosim", "Cosim  cos(G̃, G_true)"),
              ("m4",    "M4  cos(G_clipped, G_true)"),
              ("clip_agg_norm", "Clipped agg. norm")]
    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 4),
                             sharey=False)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(arm_names)))

    for ax, (field, ylabel) in zip(axes, fields):
        for arm, color in zip(arm_names, colors):
            rows = _load_csv(arm, eps, 0, out_dir)
            ep, vals = _epoch_series(rows, field)
            if ep is None: continue
            ls = "-" if arm == "vanilla" else (
                 ":" if ARMS[arm]["oracle"] else "--")
            ax.plot(ep, vals, ls=ls, color=color, label=arm, lw=1.5)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=7)
    fig.suptitle(f"Fig 3: Training Diagnostics (ε={eps:.0f}, seed=0)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, f"fig3_diag_eps{eps:.0f}.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"[P6] Fig 3 saved: {path}")


# ---------------------------------------------------------------------------
# Fig 4: bar chart — final accuracy across ε levels
# ---------------------------------------------------------------------------

def _make_fig4(arm_names, eps_list, n_seeds, out_dir):
    n_eps = len(eps_list)
    fig, axes = plt.subplots(1, n_eps, figsize=(5 * n_eps, 5), sharey=True)
    if n_eps == 1: axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(arm_names)))

    for ax, eps in zip(axes, eps_list):
        xs = np.arange(len(arm_names))
        for i, arm in enumerate(arm_names):
            accs = [_final_acc(_load_csv(arm, eps, s, out_dir))
                    for s in range(n_seeds)]
            accs = [a for a in accs if a is not None]
            if not accs: continue
            mu  = np.mean(accs) * 100
            std = np.std(accs)  * 100
            hatch = "//" if ARMS[arm]["oracle"] else ""
            ax.bar(xs[i], mu, yerr=std, color=colors[i], capsize=4,
                   alpha=0.85, hatch=hatch, label=arm if ax is axes[0] else "")
        ax.set_title(f"ε = {eps:.0f}", fontsize=12)
        ax.set_xticks(xs)
        ax.set_xticklabels([n.replace("_", "\n") for n in arm_names],
                           fontsize=6)
        ax.set_ylabel("Final test accuracy (%)" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(fontsize=7)
    fig.suptitle("Fig 4: HCCP-SGD vs Vanilla — Final Accuracy\n"
                 "(hatched bars = oracle, no formal DP guarantee)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_final_acc_bar.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"[P6] Fig 4 saved: {path}")


# ---------------------------------------------------------------------------
# Success criteria
# ---------------------------------------------------------------------------

def _print_success(arm_names, eps_list, n_seeds, out_dir):
    print("\n=== Success Criteria ===")

    for eps in eps_list:
        print(f"\nε = {eps:.0f}:")
        van_accs = [_final_acc(_load_csv("vanilla", eps, s, out_dir))
                    for s in range(n_seeds)]
        van_accs = [a for a in van_accs if a is not None]
        van_mu   = np.mean(van_accs) * 100 if van_accs else None

        for arm in arm_names:
            if arm == "vanilla": continue
            accs = [_final_acc(_load_csv(arm, eps, s, out_dir))
                    for s in range(n_seeds)]
            accs = [a for a in accs if a is not None]
            if not accs or van_mu is None:
                print(f"  {arm:<18}: N/A"); continue
            mu  = np.mean(accs) * 100
            std = np.std(accs)  * 100
            delta = mu - van_mu
            marker = "[oracle*]" if ARMS[arm]["oracle"] else ""
            print(f"  {arm:<18}: {mu:.2f}±{std:.2f}%  Δ={delta:+.2f}pp  {marker}")

    # Q1: does any HCCP arm beat vanilla at ≥1 ε?
    hccp_arms = [a for a in arm_names if ARMS[a]["r"] is not None and not ARMS[a]["oracle"]]
    q1 = False
    for arm in hccp_arms:
        for eps in eps_list:
            van_accs = [_final_acc(_load_csv("vanilla", eps, s, out_dir)) for s in range(n_seeds)]
            arm_accs = [_final_acc(_load_csv(arm,      eps, s, out_dir)) for s in range(n_seeds)]
            van_accs = [a for a in van_accs if a is not None]
            arm_accs = [a for a in arm_accs if a is not None]
            if van_accs and arm_accs and np.mean(arm_accs) > np.mean(van_accs):
                q1 = True

    print(f"\nQ1 (HCCP beats vanilla at ≥1 ε): {'PASS' if q1 else 'FAIL'}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def _run_analysis(arm_names, eps_list, n_seeds, out_dir):
    print("\n[P6] === Analysis ===")
    _make_tab1(arm_names, eps_list, n_seeds, out_dir)
    for eps in eps_list:
        _make_fig1(arm_names, eps, n_seeds, out_dir)
        _make_fig2(arm_names, eps, n_seeds, out_dir)
        _make_fig3(arm_names, eps, n_seeds, out_dir)
    _make_fig4(arm_names, eps_list, n_seeds, out_dir)
    _print_success(arm_names, eps_list, n_seeds, out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_exp_p6(
    arms          = None,    # None → all 7 arms
    eps_list      = None,    # None → all 3 ε levels
    n_seeds       = N_SEEDS,
    gpu           = None,
    results_dir   = RESULTS_DIR,
    analysis_only = False,
):
    if gpu is not None:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P6] device={device}")

    out_dir = os.path.join(results_dir, "cifar10_balanced")
    os.makedirs(out_dir, exist_ok=True)

    arm_names = arms     or list(ARMS.keys())
    run_eps   = eps_list or EPS_VALUES

    if not analysis_only:
        data = _load_data()
        print(f"[P6] n_train={data['n_train']}, n_probe={N_PROBE}")
        print(f"[P6] arms={arm_names}, ε={run_eps}, seeds=range({n_seeds})")

        for arm in arm_names:
            for eps in run_eps:
                for seed in range(n_seeds):
                    _train_run(arm, eps, seed, data, device, out_dir)

    _run_analysis(arm_names, run_eps, n_seeds, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 6: HCCP-SGD End-to-End Evaluation"
    )
    parser.add_argument("--arm", type=str, default=None,
                        choices=list(ARMS.keys()),
                        help="Single arm to run. Default: all arms.")
    parser.add_argument("--eps", type=float, default=None,
                        help="Single ε value. Default: all (2, 4, 8).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed. Default: 0..N_SEEDS-1.")
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--analysis_only", action="store_true")
    args = parser.parse_args()

    # Resolve single-arm / single-ε / single-seed shortcuts
    arms     = [args.arm] if args.arm else None
    eps_list = [args.eps] if args.eps else None
    n_seeds  = 1 if args.seed is not None else args.n_seeds

    if args.seed is not None and not args.analysis_only:
        # Run exactly one (arm, eps, seed) combination
        if not arms or not eps_list:
            parser.error("--seed requires --arm and --eps")
        data   = _load_data()
        if args.gpu is not None:
            device = torch.device(f"cuda:{args.gpu}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        out_dir = os.path.join(args.results_dir, "cifar10_balanced")
        os.makedirs(out_dir, exist_ok=True)
        _train_run(arms[0], eps_list[0], args.seed, data, device, out_dir)
    else:
        run_exp_p6(
            arms=arms, eps_list=eps_list, n_seeds=n_seeds,
            gpu=args.gpu, results_dir=args.results_dir,
            analysis_only=args.analysis_only,
        )
