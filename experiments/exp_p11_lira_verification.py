#!/usr/bin/env python3
"""
Phase 11 — Experiment 3: LiRA Certificate Verification
========================================================

Verifies that the distribution-dependent privacy certificate (Arm B: dist_aware)
is not vacuous: no training example has empirical per-instance privacy worse
than the claimed eps=2.

Trains 8 DP shadow models under the dist_aware configuration, then runs
offline LiRA (Carlini et al. 2022) on 100 target examples stratified by
coherence (beta_i from Exp 1).

Plots LiRA score vs beta_i: if the direction-aware bound is correct,
higher beta should correlate with higher LiRA score (more vulnerable),
and no example should exceed the eps=2 certificate.

Prerequisite:
  - exp_p11_beta_measurement.py must have run (provides beta_i per example)
  - exp_p11_noise_reduction.py must show dist_aware >= vanilla_warm + 2pp

Usage
-----
  # Full run (trains 8 shadow models):
  python experiments/exp_p11_lira_verification.py --gpu 0

  # Train a single shadow:
  python experiments/exp_p11_lira_verification.py --shadow 3 --gpu 0

  # Analysis only (requires all 8 shadows trained):
  python experiments/exp_p11_lira_verification.py --analysis_only

  # Provide beta95 and rank directly:
  python experiments/exp_p11_lira_verification.py --beta95 0.35 --rank 50 --gpu 0
"""

import os
import sys
import csv
import math
import gc
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import WideResNet
from src.datasets import make_public_private_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DELTA         = 1e-5
EPS           = 2.0
CLIP_C        = 1.0
N_PUB         = 2000
EPOCHS        = 60
BATCH_SIZE    = 1000
LR            = 0.1
MOMENTUM      = 0.9
WEIGHT_DECAY  = 5e-4
PRETRAIN_EPOCHS = 50
PRETRAIN_LR   = 0.01
PUB_BATCH     = 256
GRAD_CHUNK    = 64
N_SHADOWS     = 8
N_TARGETS     = 100
TARGET_SEED   = 1234
DATA_ROOT     = "./data"
RESULTS_DIR   = "./results/exp_p11"

EXP1_DIR = os.path.join(RESULTS_DIR, "exp1")
EXP2_DIR = os.path.join(RESULTS_DIR, "exp2")
EXP3_DIR = os.path.join(RESULTS_DIR, "exp3")


# ---------------------------------------------------------------------------
# Data helpers (same as exp 1 and 2)
# ---------------------------------------------------------------------------

def _cifar10(data_root, train=True, augment=False):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    if augment:
        tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                        T.ToTensor(), T.Normalize(mean, std)])
    else:
        tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return torchvision.datasets.CIFAR10(
        root=data_root, train=train, download=True, transform=tf)


def _build_datasets(data_root, seed=42):
    full_train   = _cifar10(data_root, train=True, augment=False)
    full_targets = np.array(full_train.targets)
    all_idx      = np.arange(len(full_train))
    pub_idx, priv_idx = make_public_private_split(
        all_idx, full_targets, public_frac=0.1, seed=seed)
    rng         = np.random.default_rng(seed)
    pub_idx_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]

    pub_ds   = Subset(full_train, pub_idx_use.tolist())
    priv_ds_aug   = Subset(_cifar10(data_root, train=True, augment=True),
                           priv_idx.tolist())
    priv_ds_noaug = Subset(_cifar10(data_root, train=True, augment=False),
                           priv_idx.tolist())
    test_ds  = _cifar10(data_root, train=False, augment=False)

    pub_x = torch.stack([pub_ds[i][0] for i in range(len(pub_ds))])
    pub_y = torch.tensor([pub_ds[i][1] for i in range(len(pub_ds))], dtype=torch.long)

    return pub_ds, priv_ds_aug, priv_ds_noaug, test_ds, pub_x, pub_y, priv_idx


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _make_model():
    return WideResNet(depth=28, widen_factor=2, num_classes=10, n_groups=16)


def _set_grads(model, flat_grad):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat_grad[offset:offset+n].view(p.shape).clone()
        offset += n


# ---------------------------------------------------------------------------
# Per-sample gradients
# ---------------------------------------------------------------------------

def _loss_fn(params, buffers, x, y, model):
    pred = torch.func.functional_call(model, (params, buffers), x.unsqueeze(0))
    return F.cross_entropy(pred, y.unsqueeze(0))


def _per_sample_grads_chunk(model, x_chunk, y_chunk, device):
    params  = {k: v for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}
    grad_fn = torch.func.grad(
        lambda p, b, xi, yi: _loss_fn(p, b, xi, yi, model))
    vmapped = torch.func.vmap(grad_fn, in_dims=(None, None, 0, 0))
    with torch.no_grad():
        g_dict = vmapped(params, buffers,
                         x_chunk.to(device), y_chunk.to(device))
    flat = torch.cat(
        [g_dict[k].view(x_chunk.shape[0], -1) for k in model.state_dict()
         if k in g_dict], dim=1)
    return flat


def _per_sample_grads_all(model, x, y, device):
    model.eval()
    parts = []
    for i in range(0, x.shape[0], GRAD_CHUNK):
        g = _per_sample_grads_chunk(model, x[i:i+GRAD_CHUNK], y[i:i+GRAD_CHUNK], device)
        parts.append(g.cpu()); del g; torch.cuda.empty_cache()
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Privacy accounting
# ---------------------------------------------------------------------------

def _calibrate_sigma(eps, delta, q, T, sigma_scale=1.0):
    """Calibrate sigma, then scale by sigma_scale (for dist_aware: 1/sqrt(beta95))."""
    from opacus.accountants.utils import get_noise_multiplier
    sigma_van = get_noise_multiplier(
        target_epsilon=eps, target_delta=delta,
        sample_rate=q, steps=T, accountant="rdp")
    return sigma_van * sigma_scale


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _pretrain_on_public(model, pub_x, pub_y, device):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=PRETRAIN_LR,
                          momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)
    N = pub_x.shape[0]
    for ep in range(1, PRETRAIN_EPOCHS + 1):
        perm = torch.randperm(N)
        for i in range(0, N, PUB_BATCH):
            idx = perm[i:i+PUB_BATCH]
            opt.zero_grad()
            F.cross_entropy(model(pub_x[idx].to(device)),
                            pub_y[idx].to(device)).backward()
            opt.step()
        sch.step()


@torch.no_grad()
def _evaluate(model, test_ds, device):
    loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                        num_workers=4, pin_memory=True)
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.shape[0]
    return correct / total


# ---------------------------------------------------------------------------
# Read Exp 1 results
# ---------------------------------------------------------------------------

def _read_exp1_beta95(exp1_dir, target_epoch=60):
    """Return (beta95, r_star) from Exp 1 beta_spectrum.csv."""
    csv_path = os.path.join(exp1_dir, "beta_spectrum.csv")
    if not os.path.exists(csv_path):
        return None, None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    ep_rows = [r for r in rows if int(r["epoch"]) == target_epoch]
    if not ep_rows:
        max_ep  = max(int(r["epoch"]) for r in rows)
        ep_rows = [r for r in rows if int(r["epoch"]) == max_ep]
    best_b95, r_star = 1.0, None
    for row in sorted(ep_rows, key=lambda x: int(x["r"])):
        b95 = float(row["beta_95"])
        if b95 <= 0.5 and r_star is None:
            r_star = int(row["r"]); best_b95 = b95
    if r_star is None:
        best_row = min(ep_rows, key=lambda x: float(x["beta_95"]))
        r_star   = int(best_row["r"]); best_b95 = float(best_row["beta_95"])
    return best_b95, r_star


# ---------------------------------------------------------------------------
# Target selection: stratified by beta_i from Exp 1
# ---------------------------------------------------------------------------

def _load_beta_values(exp1_dir, n_priv, r_star=1, epoch=60):
    """
    Load per-example beta values from Exp 1.
    Returns array of shape [n_priv] aligned to private dataset indices.
    If Exp 1 results not found, returns uniform betas.
    """
    beta_path = os.path.join(exp1_dir, f"beta_r{r_star}_ep{epoch:03d}.npy")
    if not os.path.exists(beta_path):
        # Try rank-1 fallback
        for r in [1, 10, 50, 100]:
            beta_path = os.path.join(exp1_dir, f"beta_r{r}_ep{epoch:03d}.npy")
            if os.path.exists(beta_path):
                print(f"[P11-E3] Loaded beta from {beta_path}")
                betas_sub = np.load(beta_path)
                # betas_sub is for the 5000-example subsample
                return betas_sub, r
        print("[P11-E3] No beta arrays found from Exp 1. "
              "Using uniform betas for target selection.")
        return None, None

    betas_sub = np.load(beta_path)
    print(f"[P11-E3] Loaded beta from {beta_path}: shape {betas_sub.shape}")
    return betas_sub, r_star


def _select_targets(exp1_dir, n_priv, out_dir, r_star=1):
    """
    Select N_TARGETS examples stratified by beta_i.
    Strata: low-beta (coherent), mid-beta, high-beta (incoherent).
    Reuses target_indices.npy if present.
    """
    tgt_path = os.path.join(out_dir, "target_indices.npy")
    beta_path_out = os.path.join(out_dir, "target_betas.npy")
    if os.path.exists(tgt_path):
        target_idx  = np.load(tgt_path)
        target_betas = np.load(beta_path_out) if os.path.exists(beta_path_out) \
            else np.full(len(target_idx), 0.5)
        print(f"[P11-E3] Loaded {len(target_idx)} target indices from {tgt_path}")
        return target_idx, target_betas

    betas_sub, r_used = _load_beta_values(exp1_dir, n_priv, r_star=r_star)
    rng = np.random.default_rng(TARGET_SEED)

    if betas_sub is None:
        # No beta data — random selection
        target_idx = rng.choice(n_priv, size=N_TARGETS, replace=False).astype(np.int64)
        target_betas = np.full(N_TARGETS, 0.5)
    else:
        n_sub   = len(betas_sub)
        thirds  = N_TARGETS // 3
        r_extra = N_TARGETS - 3 * thirds

        # Sort by beta to get strata
        order       = np.argsort(betas_sub)
        low_pool    = order[:n_sub // 3]
        mid_pool    = order[n_sub // 3: 2 * n_sub // 3]
        high_pool   = order[2 * n_sub // 3:]

        sel_low  = rng.choice(low_pool,  size=thirds,          replace=False)
        sel_mid  = rng.choice(mid_pool,  size=thirds,          replace=False)
        sel_high = rng.choice(high_pool, size=thirds + r_extra, replace=False)

        sub_sel  = np.concatenate([sel_low, sel_mid, sel_high])
        target_idx   = sub_sel.astype(np.int64)  # indices into priv subsample
        target_betas = betas_sub[sub_sel]

    np.save(tgt_path, target_idx)
    np.save(beta_path_out, target_betas)
    print(f"[P11-E3] Selected {len(target_idx)} targets "
          f"(beta range: [{target_betas.min():.3f}, {target_betas.max():.3f}])")
    return target_idx, target_betas


# ---------------------------------------------------------------------------
# Shadow model training (DP, dist_aware config)
# ---------------------------------------------------------------------------

def _train_shadow(m, priv_ds_aug, priv_ds_noaug, test_ds, pub_x, pub_y,
                  target_idx, device, shadow_dir, beta95, eps=EPS, seed_base=1000):
    model_path = os.path.join(shadow_dir, f"shadow_{m:02d}.pt")
    mem_path   = os.path.join(shadow_dir, f"membership_{m:02d}.npy")
    if os.path.exists(model_path) and os.path.exists(mem_path):
        print(f"[P11-E3] shadow {m:02d}: already done, skipping.")
        return

    # Offline LiRA: random subset membership
    rng        = np.random.RandomState(seed=seed_base + m)
    membership = rng.binomial(1, 0.5, size=len(target_idx)).astype(np.int8)

    n_priv     = len(priv_ds_aug)
    target_set = set(int(i) for i in target_idx)
    non_target = [i for i in range(n_priv) if i not in target_set]
    included   = [int(target_idx[j]) for j in range(len(target_idx))
                  if membership[j] == 1]
    train_idx  = np.array(non_target + included, dtype=np.int64)

    print(f"[P11-E3] shadow {m:02d}: n_train={len(train_idx)}, "
          f"mem_in={membership.sum()}")

    # Privacy accounting
    tmp = DataLoader(Subset(priv_ds_aug, train_idx.tolist()),
                     batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    steps_per_epoch = len(tmp)
    T_steps = EPOCHS * steps_per_epoch
    q       = BATCH_SIZE / len(train_idx)
    del tmp

    sigma_van  = _calibrate_sigma(eps, DELTA, q, T_steps, sigma_scale=1.0)
    sigma_use  = sigma_van / math.sqrt(max(beta95, 1e-6))
    print(f"[P11-E3] shadow {m:02d}: sigma_van={sigma_van:.4f} "
          f"sigma_use={sigma_use:.4f} (beta95={beta95:.4f})")

    torch.manual_seed(seed_base * 10 + m)
    np.random.seed(seed_base + m)
    random.seed(seed_base + m)

    model = _make_model().to(device)
    print(f"[P11-E3] shadow {m:02d}: pretraining...")
    _pretrain_on_public(model, pub_x, pub_y, device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    loader = DataLoader(Subset(priv_ds_aug, train_idx.tolist()),
                        batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            B      = x.shape[0]
            grads  = _per_sample_grads_all(model, x, y, device)
            norms  = grads.norm(dim=1, keepdim=True).clamp(min=1e-8)
            clipped = grads * (CLIP_C / norms).clamp(max=1.0)
            sum_g  = clipped.sum(0)
            noise  = torch.randn_like(sum_g) * (sigma_use * CLIP_C)
            flat_g = (sum_g + noise) / B
            _set_grads(model, flat_g.to(device))
            optimizer.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == EPOCHS:
            acc = _evaluate(model, test_ds, device)
            print(f"  shadow {m:02d}  ep {epoch:3d}/{EPOCHS}  acc={acc:.4f}")

    torch.save(model.state_dict(), model_path)
    np.save(mem_path, membership)
    del model, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[P11-E3] shadow {m:02d}: done.")


# ---------------------------------------------------------------------------
# Logit collection
# ---------------------------------------------------------------------------

def _collect_logits(target_idx, target_betas, priv_ds_noaug, shadow_dir, device):
    logit_path  = os.path.join(shadow_dir, "logit_matrix.npy")
    mem_path    = os.path.join(shadow_dir, "membership_matrix.npy")
    if os.path.exists(logit_path) and os.path.exists(mem_path):
        print("[P11-E3] Logit matrix already collected.")
        return np.load(logit_path), np.load(mem_path)

    n_tgt = len(target_idx)
    logit_matrix = np.full((n_tgt, N_SHADOWS), np.nan, dtype=np.float32)
    mem_matrix   = np.zeros((n_tgt, N_SHADOWS), dtype=np.int8)

    # Pre-load target images (no augmentation)
    tgt_imgs, tgt_lbls = [], []
    for idx in target_idx:
        x, y = priv_ds_noaug[int(idx)]
        tgt_imgs.append(x); tgt_lbls.append(int(y))
    tgt_x = torch.stack(tgt_imgs).to(device)
    tgt_y = torch.tensor(tgt_lbls, device=device)

    for m in range(N_SHADOWS):
        model_path = os.path.join(shadow_dir, f"shadow_{m:02d}.pt")
        mem_f      = os.path.join(shadow_dir, f"membership_{m:02d}.npy")
        if not os.path.exists(model_path) or not os.path.exists(mem_f):
            print(f"[P11-E3] shadow {m:02d} not found, skipping logits.")
            continue

        state = torch.load(model_path, map_location=device)
        model = _make_model(); model.load_state_dict(state); model.eval().to(device)

        with torch.no_grad():
            logits      = model(tgt_x)                           # [n_tgt, C]
            true_logits = logits[torch.arange(n_tgt, device=device), tgt_y]
            logit_matrix[:, m] = true_logits.cpu().numpy()
        mem_matrix[:, m] = np.load(mem_f).astype(np.int8)

        del model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[P11-E3] shadow {m:02d}: logits collected.")

    np.save(logit_path, logit_matrix)
    np.save(mem_path, mem_matrix)
    return logit_matrix, mem_matrix


# ---------------------------------------------------------------------------
# LiRA scoring (offline variant)
# ---------------------------------------------------------------------------

def _compute_lira_scores(logit_matrix, mem_matrix, out_dir):
    """
    Offline LiRA: score_i = (mu_in_i - mu_out_i) / sigma_out_i.
    Higher score = more distinguishable = more vulnerable.
    """
    score_path = os.path.join(out_dir, "lira_scores.npy")
    if os.path.exists(score_path):
        return np.load(score_path)

    n_tgt, n_sh = logit_matrix.shape
    scores = np.full(n_tgt, np.nan)

    for i in range(n_tgt):
        in_idx  = np.where(mem_matrix[i] == 1)[0]
        out_idx = np.where(mem_matrix[i] == 0)[0]

        valid_in  = in_idx[np.isfinite(logit_matrix[i, in_idx])]
        valid_out = out_idx[np.isfinite(logit_matrix[i, out_idx])]

        if len(valid_in) < 2 or len(valid_out) < 2:
            continue

        mu_in   = logit_matrix[i, valid_in].mean()
        mu_out  = logit_matrix[i, valid_out].mean()
        std_out = logit_matrix[i, valid_out].std() + 1e-8
        scores[i] = (mu_in - mu_out) / std_out

    np.save(score_path, scores)
    print(f"[P11-E3] LiRA scores: mean={np.nanmean(scores):.3f}  "
          f"max={np.nanmax(scores):.3f}  "
          f"valid={np.isfinite(scores).sum()}/{n_tgt}")
    return scores


# ---------------------------------------------------------------------------
# Analysis and plotting
# ---------------------------------------------------------------------------

def _run_analysis(out_dir):
    shadow_dir  = os.path.join(out_dir, "shadows")
    logit_path  = os.path.join(shadow_dir, "logit_matrix.npy")
    mem_path    = os.path.join(shadow_dir, "membership_matrix.npy")
    score_path  = os.path.join(shadow_dir, "lira_scores.npy")
    tgt_path    = os.path.join(out_dir, "target_indices.npy")
    beta_path   = os.path.join(out_dir, "target_betas.npy")

    if not os.path.exists(logit_path):
        print("[P11-E3] Logit matrix not found. Run shadows first.")
        return
    if not os.path.exists(score_path):
        print("[P11-E3] LiRA scores not found.")
        return

    scores = np.load(score_path)
    betas  = np.load(beta_path) if os.path.exists(beta_path) \
        else np.full(len(scores), 0.5)

    valid  = np.isfinite(scores) & np.isfinite(betas)

    print(f"\n{'='*60}")
    print(" Phase 11 Exp 3 — LiRA Certificate Check")
    print(f"{'='*60}")
    print(f"  Valid targets: {valid.sum()}/{len(scores)}")
    print(f"  LiRA scores: mean={np.nanmean(scores):.3f}  "
          f"max={np.nanmax(scores):.3f}  "
          f"p95={np.nanpercentile(scores, 95):.3f}")

    # Spearman correlation: LiRA vs beta
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(betas[valid], scores[valid])
    except ImportError:
        rho, pval = float("nan"), float("nan")
    print(f"  Spearman(beta_i, LiRA_i): rho={rho:.4f}  p={pval:.4f}")

    # Certificate check: rough comparison with eps=2
    # Under (eps, delta)-DP, the max LiRA distinguishability should be O(eps)
    # We report whether max(scores) is consistent with eps=2
    max_score = np.nanmax(scores)
    print(f"  Max LiRA score: {max_score:.3f}  (eps=2 reference)")
    if max_score <= EPS * 2:
        print("  CERTIFICATE: max LiRA score within 2x of eps → certificate plausible.")
    else:
        print("  WARNING: max LiRA score large → investigate.")

    # Save summary
    summary = {"n_valid": int(valid.sum()), "lira_mean": float(np.nanmean(scores)),
               "lira_max": float(np.nanmax(scores)),
               "lira_p95": float(np.nanpercentile(scores, 95)),
               "spearman_rho": float(rho), "spearman_pval": float(pval)}
    with open(os.path.join(out_dir, "lira_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader(); w.writerow({k: f"{v:.6f}" for k, v in summary.items()})

    _plot_lira_vs_beta(scores, betas, valid, out_dir)


def _plot_lira_vs_beta(scores, betas, valid, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    try:
        from scipy.stats import spearmanr
    except ImportError:
        def spearmanr(x, y):
            return float("nan"), float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter LiRA vs beta_i
    ax = axes[0]
    sc = ax.scatter(betas[valid], scores[valid], c=betas[valid],
                    cmap="RdYlGn_r", alpha=0.7, s=40, edgecolors="none")
    plt.colorbar(sc, ax=ax, label=r"$\beta_i$")
    rho, _ = spearmanr(betas[valid], scores[valid])
    ax.set_xlabel(r"$\beta_i$ (gradient incoherence)")
    ax.set_ylabel("LiRA score")
    ax.set_title(f"LiRA vs. coherence  (Spearman ρ = {rho:.3f})")
    ax.axhline(EPS, color="red", ls="--", lw=1, label=f"ε={EPS:.0f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: histogram of LiRA scores by beta stratum
    ax = axes[1]
    n   = len(scores[valid])
    thr = [np.percentile(betas[valid], 33), np.percentile(betas[valid], 67)]
    s_low  = scores[valid & (betas <= thr[0])]
    s_mid  = scores[valid & (betas > thr[0]) & (betas <= thr[1])]
    s_high = scores[valid & (betas > thr[1])]
    bins   = np.linspace(np.nanmin(scores), np.nanmax(scores) + 0.1, 25)
    ax.hist(s_low,  bins=bins, alpha=0.6, label=f"low β (coherent, n={len(s_low)})",  color="green")
    ax.hist(s_mid,  bins=bins, alpha=0.6, label=f"mid β (n={len(s_mid)})",             color="orange")
    ax.hist(s_high, bins=bins, alpha=0.6, label=f"high β (incoherent, n={len(s_high)})", color="red")
    ax.axvline(EPS, color="black", ls="--", lw=1.5, label=f"ε={EPS:.0f}")
    ax.set_xlabel("LiRA score")
    ax.set_ylabel("Count")
    ax.set_title("LiRA score distribution by coherence stratum")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "lira_vs_beta.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P11-E3] Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--shadow", type=int, default=None,
                        help="Single shadow index. If not set, runs all N_SHADOWS.")
    parser.add_argument("--beta95", type=float, default=None)
    parser.add_argument("--rank",   type=int, default=None,
                        help="r* for loading beta arrays from Exp 1.")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--out_dir",  type=str, default=EXP3_DIR)
    parser.add_argument("--exp1_dir", type=str, default=EXP1_DIR)
    parser.add_argument("--exp2_dir", type=str, default=EXP2_DIR)
    parser.add_argument("--n_shadows", type=int, default=N_SHADOWS)
    parser.add_argument("--n_targets", type=int, default=N_TARGETS)
    parser.add_argument("--analysis_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    shadow_dir = os.path.join(args.out_dir, "shadows")
    os.makedirs(shadow_dir, exist_ok=True)

    if args.analysis_only:
        _run_analysis(args.out_dir)
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[P11-E3] Device: {device}")

    # Read beta95 from Exp 1 CSV or CLI
    beta95 = args.beta95
    if beta95 is None:
        beta95_e1, r_star_e1 = _read_exp1_beta95(args.exp1_dir)
        if beta95_e1 is not None:
            beta95 = beta95_e1
            print(f"[P11-E3] Read beta95={beta95:.4f} from Exp 1")
        else:
            print("[P11-E3] WARNING: beta95 not found. Using beta95=0.5.")
            beta95 = 0.5

    r_star = args.rank
    if r_star is None:
        _, r_star_from_e1 = _read_exp1_beta95(args.exp1_dir)
        r_star = r_star_from_e1 if r_star_from_e1 is not None else 1

    print(f"[P11-E3] Using beta95={beta95:.4f}, r*={r_star}")

    # Load data
    pub_ds, priv_ds_aug, priv_ds_noaug, test_ds, pub_x, pub_y, priv_idx = \
        _build_datasets(args.data_root)

    # Select targets
    target_idx, target_betas = _select_targets(
        args.exp1_dir, len(priv_ds_aug), args.out_dir, r_star=r_star)

    # Train shadows
    shadows_to_run = ([args.shadow] if args.shadow is not None
                      else list(range(args.n_shadows)))

    for m in shadows_to_run:
        _train_shadow(m, priv_ds_aug, priv_ds_noaug, test_ds,
                      pub_x, pub_y, target_idx, device, shadow_dir,
                      beta95=beta95)

    # Check if all shadows done before collecting logits
    all_done = all(
        os.path.exists(os.path.join(shadow_dir, f"shadow_{m:02d}.pt"))
        for m in range(args.n_shadows))

    if not all_done:
        n_done = sum(
            os.path.exists(os.path.join(shadow_dir, f"shadow_{m:02d}.pt"))
            for m in range(args.n_shadows))
        print(f"[P11-E3] {n_done}/{args.n_shadows} shadows complete. "
              f"Run remaining shadows before collecting logits.")
        return

    # Collect logits and score
    logit_matrix, mem_matrix = _collect_logits(
        target_idx, target_betas, priv_ds_noaug, shadow_dir, device)
    _compute_lira_scores(logit_matrix, mem_matrix, shadow_dir)
    _run_analysis(args.out_dir)


if __name__ == "__main__":
    main()
