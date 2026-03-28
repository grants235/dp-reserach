#!/usr/bin/env python3
"""
exp_p5b_probe.py — Phase 5b: Fast Gradient Quality Probe for CC-DPSGD
=======================================================================
Tests the CC-DPSGD MECHANISM (not full training) in ~4 GPU-hours:

  1. Train a non-private WRN-28-2 on CIFAR-10 for 30 epochs, saving
     checkpoints at {1, 5, 10, 20, 30}.  (~10 min)
  2. At each checkpoint:
       a. Compute V_k from 500 public test-set examples (vmap, once).
       b. Compute per-sample gradients for ONE fixed 256-example batch (vmap, once).
       c. For each config: apply clipping to get G_clipped (once, cheap).
       d. Simulate N=200 noise draws (just add Gaussian vectors—trivially cheap).
       e. Measure cosine similarity, relMSE, signal capture vs true gradient G*.
  3. Generate Fig 1-4, Tab 1, success criteria.

Key insight: the expensive vmap computation happens ONCE per checkpoint.
The 200-draw noise simulation is pure matrix ops (milliseconds).

Run:
  python experiments/exp_p5b_probe.py
  python experiments/exp_p5b_probe.py --skip_train  # if checkpoints exist
  python experiments/exp_p5b_probe.py --analysis_only
"""

import os
import sys
import gc
import math
import pickle
import argparse

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

CHECKPOINT_EPOCHS = [1, 5, 10, 20, 30]
EPOCHS_TRAIN      = 30
BATCH_SIZE_TRAIN  = 256
LR_TRAIN          = 0.1
WD                = 5e-4

N_PUB    = 500    # public anchor examples (from test set)
B_PROBE  = 256    # fixed batch size for gradient quality measurement
N_DRAWS  = 200    # noise draws per (config, checkpoint)
K_VALUES = [10, 50]

# Privacy scenario for σ computation (balanced CIFAR-10, 100-epoch training)
EPS_PROBE = 4.0
DELTA     = 1e-5
N_TRAIN   = 50000
# σ will be computed once and reused across all noise draws

GRAD_CHUNK  = 16     # vmap chunk size for per-sample gradients
DATA_ROOT   = "./data"
RESULTS_DIR = "./results/exp_p5b"

# ---------------------------------------------------------------------------
# Configurations (same as phase5_spex.tex)
# ---------------------------------------------------------------------------

import math as _math

CONFIGS = [
    {"name": "vanilla",      "k": None, "C_par": 1.0,  "C_perp": None},
    {"name": "cc_k10_r3",    "k": 10,  "C_par": 0.95, "C_perp": 0.32},
    {"name": "cc_k50_r3",    "k": 50,  "C_par": 0.95, "C_perp": 0.32},
    {"name": "cc_k50_r10",   "k": 50,  "C_par": 1.0,  "C_perp": 0.10},
    {"name": "cc_k50_proj",  "k": 50,  "C_par": 1.0,  "C_perp": 0.0},
]
for _c in CONFIGS:
    _c["C_max"] = (
        _c["C_par"] if _c["C_perp"] is None
        else _math.sqrt(_c["C_par"] ** 2 + _c["C_perp"] ** 2)
    )

CONFIG_BY_NAME = {c["name"]: c for c in CONFIGS}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_data(data_root=DATA_ROOT, pub_seed=42, probe_batch_seed=0):
    """
    Load balanced CIFAR-10.
      train_ds  : full 50K training set (no aug; aug applied if desired later)
      pub_ds    : 500 stratified test-set examples as public anchor
      eval_ds   : full 10K test set
      probe_batch : fixed (B_PROBE, C, H, W) + labels for gradient probe
    """
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as T

    norm   = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    noaug  = T.Compose([T.ToTensor(), norm])
    aug_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                        T.ToTensor(), norm])

    train_ds_aug  = CIFAR10(data_root, train=True,  download=True, transform=aug_tf)
    train_ds_noaug= CIFAR10(data_root, train=True,  download=True, transform=noaug)
    test_ds       = CIFAR10(data_root, train=False,  download=True, transform=noaug)

    # Fixed 500 public examples from test set (50 per class)
    rng = np.random.RandomState(pub_seed)
    test_tgts = np.array(test_ds.targets)
    pub_idx = []
    for c in range(10):
        cls_idx = np.where(test_tgts == c)[0]
        pub_idx.extend(rng.choice(cls_idx, 50, replace=False).tolist())
    pub_ds = Subset(test_ds, sorted(pub_idx))

    # Fixed probe batch: B_PROBE examples from training set (no aug)
    rng2 = np.random.RandomState(probe_batch_seed)
    probe_idx = rng2.choice(N_TRAIN, B_PROBE, replace=False).tolist()
    probe_ds  = Subset(train_ds_noaug, probe_idx)

    return {
        "train_ds_aug":   train_ds_aug,
        "train_ds_noaug": train_ds_noaug,
        "pub_ds":  pub_ds,
        "eval_ds": test_ds,
        "probe_ds": probe_ds,
        "num_classes": 10,
        "n_train": N_TRAIN,
    }


# ---------------------------------------------------------------------------
# Step 1: Non-private training trajectory
# ---------------------------------------------------------------------------

def _train_checkpoints(data, device, out_dir):
    """
    Train WRN-28-2 (GroupNorm) on CIFAR-10 for EPOCHS_TRAIN epochs (non-private).
    Save model state at each epoch in CHECKPOINT_EPOCHS.
    Skips epochs whose checkpoint already exists.
    """
    ckpt_paths = {
        e: os.path.join(out_dir, f"checkpoint_epoch{e:02d}.pt")
        for e in CHECKPOINT_EPOCHS
    }
    remaining = [e for e in CHECKPOINT_EPOCHS if not os.path.exists(ckpt_paths[e])]
    if not remaining:
        print("[P5b] all checkpoints already exist, skipping training")
        return ckpt_paths

    print(f"[P5b] training non-private WRN-28-2 for {EPOCHS_TRAIN} epochs...")
    set_seed(42)

    use_pin  = device.type == "cuda"
    loader   = DataLoader(data["train_ds_aug"], batch_size=BATCH_SIZE_TRAIN,
                          shuffle=True, num_workers=4, pin_memory=use_pin,
                          drop_last=True)
    eval_loader = DataLoader(data["eval_ds"], batch_size=512, shuffle=False,
                             num_workers=4, pin_memory=use_pin)

    model = make_model("wrn28-2", num_classes=data["num_classes"]).to(device)
    opt   = torch.optim.SGD(model.parameters(), lr=LR_TRAIN,
                             momentum=0.9, weight_decay=WD)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_TRAIN)

    for epoch in range(1, EPOCHS_TRAIN + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        sch.step()

        if epoch in CHECKPOINT_EPOCHS:
            torch.save(model.state_dict(), ckpt_paths[epoch])
            acc = evaluate(model, eval_loader, device)
            print(f"  epoch {epoch:>2}/{EPOCHS_TRAIN}: acc={acc:.4f}  "
                  f"→ saved {ckpt_paths[epoch]}")

    del model, opt, sch
    if device.type == "cuda": torch.cuda.empty_cache()
    return ckpt_paths


# ---------------------------------------------------------------------------
# Step 2a: Public subspace
# ---------------------------------------------------------------------------

def _compute_public_subspace(model, pub_ds, k, device):
    """
    Compute top-k right singular vectors of (N_pub, d) gradient matrix.
    Returns V_k : (d, k) on device.
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
    use_pin = device.type == "cuda"
    loader  = DataLoader(pub_ds, batch_size=50, shuffle=False,
                         num_workers=2, pin_memory=use_pin)

    G = torch.zeros(len(pub_ds), d, device=device)
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

    if was_training: model.train()

    _, _, V_k = torch.svd_lowrank(G, q=k, niter=4)   # V_k: (d, k)
    del G
    if device.type == "cuda": torch.cuda.empty_cache()
    return V_k.contiguous()


# ---------------------------------------------------------------------------
# Step 2b: Per-sample gradients for the fixed probe batch
# ---------------------------------------------------------------------------

def _compute_per_sample_grads(model, probe_ds, device):
    """
    Compute per-sample gradients for the fixed probe batch (no augmentation).
    Returns grads : (B, d) float32 on CPU (stored on CPU to free GPU memory).
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
    use_pin = device.type == "cuda"
    loader  = DataLoader(probe_ds, batch_size=GRAD_CHUNK, shuffle=False,
                         num_workers=2, pin_memory=use_pin)

    B   = len(probe_ds)
    out = torch.zeros(B, d)   # stored on CPU
    offset = 0
    was_training = model.training
    model.eval()

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        Bc   = x.shape[0]
        gd   = per_grad_fn(params, x, y)
        flat = torch.cat([gd[n].reshape(Bc, -1) for n in param_names], dim=1)
        out[offset:offset + Bc] = flat.detach().cpu()
        del gd, flat
        if device.type == "cuda": torch.cuda.empty_cache()
        offset += Bc

    if was_training: model.train()
    return out   # (B, d) on CPU


# ---------------------------------------------------------------------------
# Step 2c: Apply clipping for one config
# ---------------------------------------------------------------------------

def _apply_clipping(grads_gpu, V_k, cfg):
    """
    Clip per-sample gradients according to cfg.
    grads_gpu : (B, d) on GPU.
    V_k       : (d, k) on GPU, or None for vanilla.
    Returns G_clipped : (d,) sum of clipped gradients (no noise).
    """
    B = grads_gpu.shape[0]
    k     = cfg["k"]
    C_par = cfg["C_par"]
    C_perp = cfg["C_perp"]
    C_max = cfg["C_max"]

    if k is not None and V_k is not None:
        # CC: asymmetric decomposition + clipping
        coeffs = grads_gpu @ V_k        # (B, k)
        g_par  = coeffs @ V_k.T         # (B, d)
        g_perp = grads_gpu - g_par

        norms_par  = g_par.norm(dim=1, keepdim=True).clamp(min=1e-8)
        g_par_clip = g_par * (C_par / norms_par).clamp(max=1.0)

        if C_perp > 0:
            norms_perp  = g_perp.norm(dim=1, keepdim=True).clamp(min=1e-8)
            g_perp_clip = g_perp * (C_perp / norms_perp).clamp(max=1.0)
        else:
            g_perp_clip = torch.zeros_like(g_perp)

        G_clipped = (g_par_clip + g_perp_clip).sum(dim=0)
    else:
        # Vanilla: uniform clip at C_max
        norms     = grads_gpu.norm(dim=1, keepdim=True).clamp(min=1e-8)
        G_clipped = (grads_gpu * (C_max / norms).clamp(max=1.0)).sum(dim=0)

    return G_clipped   # (d,) on GPU


# ---------------------------------------------------------------------------
# Step 2d-e: Noise simulation + metrics
# ---------------------------------------------------------------------------

def _cosine_sim(a, b):
    """Cosine similarity between two 1-D tensors."""
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def _simulate_and_measure(G_clipped, G_star, V_k_dict, sigma, cfg, n_draws=N_DRAWS):
    """
    Simulate N_DRAWS noise draws and compute metrics.
    G_clipped : (d,) on GPU — clipped aggregate (no noise).
    G_star    : (d,) on GPU — true aggregate gradient.
    V_k_dict  : {k: V_k tensor}  for signal capture.
    Returns dict with arrays of shape (n_draws,).
    """
    C_max = cfg["C_max"]
    d = G_clipped.shape[0]

    # Pre-compute noiseless metrics (M4)
    m4_cosim   = _cosine_sim(G_clipped, G_star)
    m4_rel_mse = ((G_clipped - G_star).norm() ** 2 / (G_star.norm() ** 2 + 1e-12)).item()

    # Noiseless signal capture for each k
    m4_sig_cap = {}
    for k, Vk in V_k_dict.items():
        proj_clipped = Vk @ (Vk.T @ G_clipped)   # (d,)
        proj_star    = Vk @ (Vk.T @ G_star)
        m4_sig_cap[k] = (proj_clipped.norm() / (proj_star.norm() + 1e-12)).item()

    # Noisy draws
    cosims    = np.zeros(n_draws)
    rel_mses  = np.zeros(n_draws)
    sig_caps  = {k: np.zeros(n_draws) for k in V_k_dict}

    noise_std = sigma * C_max
    for i in range(n_draws):
        noise   = torch.randn_like(G_clipped) * noise_std
        G_noisy = G_clipped + noise

        cosims[i]   = _cosine_sim(G_noisy, G_star)
        rel_mses[i] = ((G_noisy - G_star).norm() ** 2
                       / (G_star.norm() ** 2 + 1e-12)).item()

        for k, Vk in V_k_dict.items():
            proj_noisy = Vk @ (Vk.T @ G_noisy)
            proj_star  = Vk @ (Vk.T @ G_star)
            sig_caps[k][i] = (proj_noisy.norm() / (proj_star.norm() + 1e-12)).item()

    return {
        "cosim":     cosims,
        "rel_mse":   rel_mses,
        "sig_cap":   sig_caps,
        "m4_cosim":  m4_cosim,
        "m4_relmse": m4_rel_mse,
        "m4_sig_cap":m4_sig_cap,
    }


# ---------------------------------------------------------------------------
# Probe at one checkpoint
# ---------------------------------------------------------------------------

def _probe_one_checkpoint(epoch, model_path, data, device, sigma, out_dir):
    """
    Full probe pipeline at one checkpoint. Results saved to disk and returned.
    """
    rpath = os.path.join(out_dir, f"probe_epoch{epoch:02d}.pkl")
    if os.path.exists(rpath):
        print(f"[P5b] epoch {epoch}: loading cached probe results")
        with open(rpath, "rb") as f:
            return pickle.load(f)

    print(f"\n[P5b] === Probe at epoch {epoch} ===")
    model = make_model("wrn28-2", num_classes=data["num_classes"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 2a: public subspace for each k ---
    print(f"  computing public subspace for k={K_VALUES}...")
    V_k_dict = {}
    for k in K_VALUES:
        V_k_dict[k] = _compute_public_subspace(model, data["pub_ds"], k, device)
    print(f"  done. V_k shapes: { {k: V_k_dict[k].shape for k in K_VALUES} }")

    # --- 2b: per-sample gradients (once) ---
    print(f"  computing per-sample grads for {B_PROBE} examples...")
    t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if t0: t0.record()
    grads_cpu = _compute_per_sample_grads(model, data["probe_ds"], device)
    if t1:
        t1.record(); torch.cuda.synchronize()
        print(f"  grads computed in {t0.elapsed_time(t1)/1000:.1f}s, shape={grads_cpu.shape}")
    grads_gpu = grads_cpu.to(device)

    # --- true aggregate G* ---
    G_star = grads_gpu.mean(dim=0)

    # --- cosine between public top-dir and G* (subspace alignment diagnostic) ---
    pub_alignment = {}
    for k, Vk in V_k_dict.items():
        top_dir  = Vk[:, 0]   # top public gradient direction
        pub_alignment[k] = _cosine_sim(top_dir, G_star / (G_star.norm() + 1e-12))

    print(f"  public-subspace / G* alignment: {pub_alignment}")

    # --- 2c-d-e: for each config, apply clipping + simulate noise ---
    results = {
        "epoch":          epoch,
        "pub_alignment":  pub_alignment,
        "G_star_norm":    G_star.norm().item(),
        "configs":        {},
    }

    for cfg in CONFIGS:
        name = cfg["name"]
        k    = cfg["k"]
        Vk   = V_k_dict.get(k)   # None for vanilla

        print(f"  config={name}  (k={k}, C_par={cfg['C_par']}, "
              f"C_perp={cfg['C_perp']}, C_max={cfg['C_max']:.4f})")

        G_clipped = _apply_clipping(grads_gpu, Vk, cfg)
        metrics   = _simulate_and_measure(G_clipped, G_star, V_k_dict,
                                          sigma, cfg, n_draws=N_DRAWS)
        results["configs"][name] = metrics

        print(f"    cosim mean={metrics['cosim'].mean():.4f} ± "
              f"{metrics['cosim'].std():.4f}  "
              f"m4_cosim={metrics['m4_cosim']:.4f}  "
              f"relMSE={metrics['rel_mse'].mean():.2f}")

    # Also record true gradient as "oracle" reference (cosim=1 by construction)
    results["configs"]["oracle"] = {
        "cosim":    np.ones(N_DRAWS),
        "rel_mse":  np.zeros(N_DRAWS),
        "m4_cosim": 1.0,
        "sig_cap":  {k: np.ones(N_DRAWS) for k in K_VALUES},
        "m4_sig_cap": {k: 1.0 for k in K_VALUES},
    }

    with open(rpath, "wb") as f:
        pickle.dump(results, f)
    print(f"  saved to {rpath}")

    del model, grads_gpu, G_star
    for Vk in V_k_dict.values(): del Vk
    if device.type == "cuda": torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Analysis and plotting
# ---------------------------------------------------------------------------

def _load_all_results(out_dir):
    results = {}
    for e in CHECKPOINT_EPOCHS:
        path = os.path.join(out_dir, f"probe_epoch{e:02d}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                results[e] = pickle.load(f)
    return results


def _make_tab1(all_results, out_dir):
    """Tab 1: all metrics at each checkpoint for each config."""
    cfg_names = [c["name"] for c in CONFIGS] + ["oracle"]
    epochs    = sorted(all_results.keys())

    lines = ["=== Tab 1: Gradient Quality Metrics (mean ± std, N=200 draws) ===", ""]
    header = (f"{'Config':<18}  {'Epoch':<6}  "
              f"{'cosim_mean':>10}  {'cosim_std':>9}  "
              f"{'relMSE_mean':>11}  {'M4_cosim':>9}  "
              f"{'sig_cap_k50':>11}")
    lines.append(header)
    lines.append("-" * 85)

    for e in epochs:
        if e not in all_results:
            continue
        res = all_results[e]
        for name in cfg_names:
            m = res["configs"].get(name)
            if m is None:
                continue
            sig_k50 = (m["sig_cap"].get(50, np.array([float("nan")])).mean()
                       if "sig_cap" in m else float("nan"))
            lines.append(
                f"{name:<18}  {e:<6}  "
                f"{m['cosim'].mean():>10.4f}  {m['cosim'].std():>9.4f}  "
                f"{m['rel_mse'].mean():>11.4f}  {m.get('m4_cosim', float('nan')):>9.4f}  "
                f"{sig_k50:>11.4f}"
            )
        lines.append("")

    tab = "\n".join(lines)
    print(tab)
    with open(os.path.join(out_dir, "tab1.txt"), "w") as f:
        f.write(tab)


def _make_fig1(all_results, out_dir):
    """Fig 1: mean cosine similarity vs epoch for each config (error bands)."""
    epochs    = sorted(all_results.keys())
    cfg_names = [c["name"] for c in CONFIGS]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(cfg_names)))

    for cfg, color in zip(CONFIGS, colors):
        name = cfg["name"]
        means, stds = [], []
        for e in epochs:
            if e not in all_results or name not in all_results[e]["configs"]:
                means.append(np.nan); stds.append(np.nan)
                continue
            m = all_results[e]["configs"][name]["cosim"]
            means.append(m.mean()); stds.append(m.std())
        means = np.array(means); stds = np.array(stds)
        ls = "-" if name == "vanilla" else "--"
        ax.plot(epochs, means, ls=ls, color=color, label=name, linewidth=1.8)
        ax.fill_between(epochs, means - stds, means + stds,
                        color=color, alpha=0.15)

    ax.set_xlabel("Training epoch (checkpoint)")
    ax.set_ylabel("Cosine similarity with true gradient")
    ax.set_title("Fig 1: Gradient Direction Quality — CC-DPSGD vs Vanilla\n"
                 f"(N={N_DRAWS} noise draws, ε={EPS_PROBE}, C_max≈1.0)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_cosim_vs_epoch.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[P5b] Fig 1 saved to {path}")


def _make_fig2(all_results, out_dir):
    """Fig 2: mean relMSE vs epoch."""
    epochs    = sorted(all_results.keys())
    cfg_names = [c["name"] for c in CONFIGS]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(cfg_names)))

    for cfg, color in zip(CONFIGS, colors):
        name = cfg["name"]
        means = []
        for e in epochs:
            if e not in all_results or name not in all_results[e]["configs"]:
                means.append(np.nan)
                continue
            means.append(all_results[e]["configs"][name]["rel_mse"].mean())
        ls = "-" if name == "vanilla" else "--"
        ax.plot(epochs, means, ls=ls, color=color, label=name, linewidth=1.8)

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Relative MSE  ||G̃ - G*||² / ||G*||²")
    ax.set_title("Fig 2: Gradient MSE — CC-DPSGD vs Vanilla")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_relmse_vs_epoch.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[P5b] Fig 2 saved to {path}")


def _make_fig3(all_results, out_dir):
    """Fig 3: M4 noiseless clipping cosine at each checkpoint, grouped by config."""
    epochs    = sorted(all_results.keys())
    cfg_names = [c["name"] for c in CONFIGS]

    fig, ax = plt.subplots(figsize=(9, 5))
    x   = np.arange(len(epochs))
    w   = 0.8 / max(len(cfg_names), 1)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(cfg_names)))

    for i, (cfg, color) in enumerate(zip(CONFIGS, colors)):
        name = cfg["name"]
        vals = [all_results[e]["configs"].get(name, {}).get("m4_cosim", np.nan)
                if e in all_results else np.nan for e in epochs]
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w * 0.9,
               label=name, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"ep {e}" for e in epochs])
    ax.set_ylabel("Noiseless cosine similarity (M4)")
    ax.set_title("Fig 3: Clipping Geometry Effect (no noise)\n"
                 "Higher = clipping preserves more of true gradient direction")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_m4_noiseless.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[P5b] Fig 3 saved to {path}")


def _make_fig4(all_results, out_dir, k=50):
    """Fig 4 (P1): signal capture (M3) vs epoch for k=50."""
    epochs    = sorted(all_results.keys())
    cfg_names = [c["name"] for c in CONFIGS if c["k"] is not None]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(cfg_names) + 1))

    # Vanilla baseline
    means = []
    for e in epochs:
        if e not in all_results or "vanilla" not in all_results[e]["configs"]:
            means.append(np.nan); continue
        sc = all_results[e]["configs"]["vanilla"].get("sig_cap", {}).get(k)
        means.append(sc.mean() if sc is not None else np.nan)
    ax.plot(epochs, means, "-", color=colors[0], label="vanilla", linewidth=1.8)

    for i, name in enumerate(cfg_names):
        means = []
        for e in epochs:
            if e not in all_results or name not in all_results[e]["configs"]:
                means.append(np.nan); continue
            sc = all_results[e]["configs"][name].get("sig_cap", {}).get(k)
            means.append(sc.mean() if sc is not None else np.nan)
        ax.plot(epochs, means, "--", color=colors[i + 1], label=name, linewidth=1.8)

    ax.axhline(1.0, color="gray", ls=":", lw=1, label="oracle (=1)")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(f"Signal capture  ||P_V{k} G̃|| / ||P_V{k} G*||")
    ax.set_title(f"Fig 4: Coherent Subspace Signal Capture (k={k})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"fig4_signal_capture_k{k}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[P5b] Fig 4 saved to {path}")


def _make_pub_alignment_fig(all_results, out_dir):
    """Extra: public subspace / G* alignment diagnostic per epoch."""
    epochs = sorted(all_results.keys())
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in K_VALUES:
        aligns = [all_results[e]["pub_alignment"].get(k, np.nan)
                  if e in all_results else np.nan for e in epochs]
        ax.plot(epochs, aligns, marker="o", label=f"k={k}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|cosine(top public dir, G*)|")
    ax.set_title("Public Subspace Alignment with True Gradient")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="threshold=0.5")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_pub_alignment.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[P5b] Public alignment figure saved to {path}")


def _print_success_criteria(all_results):
    """Print S1-S4 pass/fail against vanilla baseline."""
    epochs = sorted(all_results.keys())
    print("\n=== Success Criteria ===")

    for cfg in CONFIGS:
        name = cfg["name"]
        if name == "vanilla":
            continue

        improvements = []
        for e in epochs:
            if e not in all_results:
                continue
            van = all_results[e]["configs"]["vanilla"]["cosim"].mean()
            cc  = all_results[e]["configs"].get(name, {}).get("cosim",
                                                              np.array([van]))
            improvements.append(cc.mean() - van)

        n_positive = sum(1 for d in improvements if d > 0)
        max_imp    = max(improvements) if improvements else 0.0
        s1 = n_positive >= 3
        s2 = max_imp > 0.01

        # S3: any checkpoint where m4_cosim(CC) > m4_cosim(vanilla)?
        m4_better = sum(
            1 for e in epochs if e in all_results
            and all_results[e]["configs"].get(name, {}).get("m4_cosim", -1)
            > all_results[e]["configs"]["vanilla"].get("m4_cosim", 1)
        )
        s3 = m4_better >= 1

        print(f"{name:<18}  improvements={[f'{d:+.4f}' for d in improvements]}")
        print(f"  S1 (≥3 of 5 ckpts better): {'PASS' if s1 else 'FAIL'}")
        print(f"  S2 (max Δcosim > 0.01):    {'PASS' if s2 else 'FAIL'} (max={max_imp:+.4f})")
        print(f"  S3 (m4 clipping better):   {'PASS' if s3 else 'FAIL'}")
        print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_exp_p5b(
    skip_train    = False,
    analysis_only = False,
    gpu           = None,
    results_dir   = RESULTS_DIR,
):
    if gpu is not None:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P5b] device={device}")

    out_dir = os.path.join(results_dir, "cifar10_balanced")
    os.makedirs(out_dir, exist_ok=True)

    if not analysis_only:
        data = _load_data()
        print(f"[P5b] n_train={data['n_train']}, n_pub={len(data['pub_ds'])}, "
              f"n_probe={len(data['probe_ds'])}")

        # --- Step 1: training trajectory ---
        if not skip_train:
            ckpt_paths = _train_checkpoints(data, device, out_dir)
        else:
            ckpt_paths = {
                e: os.path.join(out_dir, f"checkpoint_epoch{e:02d}.pt")
                for e in CHECKPOINT_EPOCHS
            }
            missing = [e for e in CHECKPOINT_EPOCHS
                       if not os.path.exists(ckpt_paths[e])]
            if missing:
                print(f"[P5b] WARNING: missing checkpoints for epochs {missing}. "
                      f"Re-running training.")
                ckpt_paths = _train_checkpoints(data, device, out_dir)

        # --- Compute σ once (balanced CIFAR-10, 100-epoch DP-SGD at ε=4) ---
        q     = BATCH_SIZE_TRAIN / N_TRAIN
        T     = 100 * (N_TRAIN // BATCH_SIZE_TRAIN)   # 100 epochs
        sigma = compute_sigma(EPS_PROBE, DELTA, q, T)
        print(f"[P5b] σ={sigma:.4f} for ε={EPS_PROBE}, q={q:.5f}, T={T}")

        # --- Step 2: probe at each checkpoint ---
        for e in CHECKPOINT_EPOCHS:
            path = ckpt_paths[e]
            if not os.path.exists(path):
                print(f"[P5b] checkpoint for epoch {e} not found, skipping")
                continue
            _probe_one_checkpoint(e, path, data, device, sigma, out_dir)

    # --- Analysis ---
    all_results = _load_all_results(out_dir)
    if not all_results:
        print("[P5b] no probe results found — run without --analysis_only first")
        return

    print(f"\n[P5b] loaded results for epochs: {sorted(all_results.keys())}")
    _make_tab1(all_results, out_dir)
    _make_fig1(all_results, out_dir)
    _make_fig2(all_results, out_dir)
    _make_fig3(all_results, out_dir)
    _make_fig4(all_results, out_dir, k=50)
    _make_pub_alignment_fig(all_results, out_dir)
    _print_success_criteria(all_results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5b: Fast CC-DPSGD Gradient Quality Probe"
    )
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip checkpoint training (use existing .pt files)")
    parser.add_argument("--analysis_only", action="store_true",
                        help="Skip all computation, regenerate outputs only")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    run_exp_p5b(
        skip_train    = args.skip_train,
        analysis_only = args.analysis_only,
        gpu           = args.gpu,
        results_dir   = args.results_dir,
    )
