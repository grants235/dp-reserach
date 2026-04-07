#!/usr/bin/env python3
"""
Phase 11 — Experiment 1: Measure beta*(r)
==========================================

Computes beta_i^(r) = 1 - ||P_r g_clip_i||^2 / ||g_clip_i||^2
for private examples i, where P_r is the top-r PCA projector of the
public gradient covariance.

Ranks r in {1, 10, 50, 100, 500, 1000}.
Checkpoints at epochs {1, 10, 30, 60} of a vanilla_warm model (eps=2, seed=0).
Uses 5000 private examples (stratified by class) for efficiency (~2 GPU-hours).

Gate: does there exist r s.t. beta_95^(r) <= 0.5?
If yes: noise reduction >= sqrt(0.5)=0.71x for 95% of examples → proceed to Exp 2.
If no: noise reduction is negligible → reframe paper as understanding contribution.

Usage
-----
  # Train + measure (will save checkpoints during training):
  python experiments/exp_p11_beta_measurement.py --gpu 0

  # Load existing checkpoint at a single epoch (skip training):
  python experiments/exp_p11_beta_measurement.py \\
      --checkpoint results/exp_p9/vanilla_warm_ir1_eps2_seed0_final.pt \\
      --epoch 60 --gpu 0

  # Analysis only (print table and plot from existing CSV):
  python experiments/exp_p11_beta_measurement.py --analysis_only

Output
------
  results/exp_p11/exp1/beta_spectrum.csv   — per-epoch per-rank statistics
  results/exp_p11/exp1/beta_rank1_<ep>.csv — per-example beta at rank 1
  results/exp_p11/exp1/beta_spectra.png    — Figure: beta_95 vs rank
"""

import os
import sys
import csv
import math
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
# Constants — match P9 exactly for valid comparison
# ---------------------------------------------------------------------------

DELTA           = 1e-5
EPS             = 2.0
CLIP_C          = 1.0
N_PUB           = 2000
N_PRIV_SAMPLE   = 5000     # subsample for efficiency
EPOCHS          = 60
BATCH_SIZE      = 1000
LR              = 0.1
MOMENTUM        = 0.9
WEIGHT_DECAY    = 5e-4
PRETRAIN_EPOCHS = 50
PRETRAIN_LR     = 0.01
PUB_BATCH       = 256
GRAD_CHUNK      = 64
DATA_ROOT       = "./data"
RESULTS_DIR     = "./results/exp_p11"

CHECKPOINT_EPOCHS = [1, 10, 30, 60]
RANKS             = [1, 10, 50, 100, 500, 1000]
MAX_RANK          = max(RANKS)

P9_DIR = "./results/exp_p9"   # check for reusable checkpoint


# ---------------------------------------------------------------------------
# Data helpers
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
    """Balanced CIFAR-10 (IR=1), N_PUB=2000, same split as P9."""
    full_train   = _cifar10(data_root, train=True, augment=False)
    full_targets = np.array(full_train.targets)
    all_idx      = np.arange(len(full_train))

    pub_idx, priv_idx = make_public_private_split(
        all_idx, full_targets, public_frac=0.1, seed=seed)

    rng         = np.random.default_rng(seed)
    pub_idx_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]

    pub_ds   = Subset(full_train, pub_idx_use.tolist())
    priv_ds  = Subset(_cifar10(data_root, train=True, augment=True), priv_idx.tolist())
    test_ds  = _cifar10(data_root, train=False, augment=False)

    pub_x = torch.stack([pub_ds[i][0] for i in range(len(pub_ds))])
    pub_y = torch.tensor([pub_ds[i][1] for i in range(len(pub_ds))], dtype=torch.long)

    # Preload private images without augmentation for beta measurement
    priv_noaug = Subset(_cifar10(data_root, train=True, augment=False), priv_idx.tolist())

    return pub_ds, priv_ds, priv_noaug, test_ds, pub_x, pub_y, priv_idx


def _subsample_private(priv_noaug, priv_idx, n_total, seed=42):
    """Stratified subsample of n_total private examples."""
    targets = np.array([priv_noaug[i][1] for i in range(len(priv_noaug))])
    rng     = np.random.default_rng(seed + 1)
    selected = []
    n_classes = 10
    per_class = n_total // n_classes
    for k in range(n_classes):
        pool = np.where(targets == k)[0]
        n    = min(per_class, len(pool))
        selected.extend(rng.choice(pool, size=n, replace=False).tolist())
    # fill remainder
    remaining = n_total - len(selected)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(len(targets)), selected)
        selected.extend(rng.choice(pool, size=remaining, replace=False).tolist())
    sub_ds = Subset(priv_noaug, selected)
    sub_targets = targets[selected]
    return sub_ds, sub_targets


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _make_model():
    return WideResNet(depth=28, widen_factor=2, num_classes=10, n_groups=16)


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Per-sample gradients (vmap, identical to P9)
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
    return flat  # [B_c, d] on device


def _per_sample_grads_all(model, x, y, device):
    model.eval()
    parts = []
    for i in range(0, x.shape[0], GRAD_CHUNK):
        g = _per_sample_grads_chunk(model, x[i:i+GRAD_CHUNK], y[i:i+GRAD_CHUNK], device)
        parts.append(g.cpu())
        del g; torch.cuda.empty_cache()
    return torch.cat(parts, dim=0)  # [N, d] on CPU


# ---------------------------------------------------------------------------
# Privacy accounting
# ---------------------------------------------------------------------------

def _calibrate_sigma(eps, delta, q, T):
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(
        target_epsilon=eps, target_delta=delta,
        sample_rate=q, steps=T, accountant="rdp")


# ---------------------------------------------------------------------------
# Public pretraining
# ---------------------------------------------------------------------------

def _pretrain_on_public(model, pub_x, pub_y, device, epochs=PRETRAIN_EPOCHS):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=PRETRAIN_LR,
                          momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = pub_x.shape[0]
    for ep in range(1, epochs + 1):
        perm = torch.randperm(N)
        for i in range(0, N, PUB_BATCH):
            idx = perm[i:i+PUB_BATCH]
            opt.zero_grad()
            F.cross_entropy(model(pub_x[idx].to(device)),
                            pub_y[idx].to(device)).backward()
            opt.step()
        sch.step()
    print(f"[P11-E1] Pretraining done ({epochs} ep)")


# ---------------------------------------------------------------------------
# Gradient-based utilities
# ---------------------------------------------------------------------------

def _set_grads(model, flat_grad):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat_grad[offset:offset+n].view(p.shape).clone()
        offset += n


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
# Training: vanilla_warm with checkpoint saving
# ---------------------------------------------------------------------------

def _train_vanilla_warm(priv_ds, test_ds, pub_x, pub_y, device, out_dir,
                        eps=EPS, seed=0):
    """
    Train vanilla_warm model (DP-SGD from public-pretrained init).
    Saves checkpoints at CHECKPOINT_EPOCHS.
    Reuses P9 final checkpoint if available.
    """
    ckpt_paths = {ep: os.path.join(out_dir, f"vanilla_warm_ep{ep:03d}.pt")
                  for ep in CHECKPOINT_EPOCHS}

    # Check if final checkpoint already exists (from P9 or previous run)
    final_path = ckpt_paths[60]
    p9_final   = os.path.join(P9_DIR, f"vanilla_warm_ir1_eps{eps:.0f}_seed{seed}_final.pt")
    p9_best    = os.path.join(P9_DIR, f"vanilla_warm_ir1_eps{eps:.0f}_seed{seed}_best.pt")

    if os.path.exists(final_path):
        print(f"[P11-E1] All checkpoints present, skipping training.")
        return ckpt_paths

    if all(os.path.exists(p) for p in ckpt_paths.values()):
        return ckpt_paths

    print(f"[P11-E1] Training vanilla_warm (eps={eps}, seed={seed})...")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    model = _make_model().to(device)
    _pretrain_on_public(model, pub_x, pub_y, device)

    tmp_loader = DataLoader(priv_ds, batch_size=BATCH_SIZE,
                            shuffle=True, drop_last=True)
    steps_per_epoch = len(tmp_loader)
    T_steps = EPOCHS * steps_per_epoch
    q       = BATCH_SIZE / len(priv_ds)
    del tmp_loader

    sigma = _calibrate_sigma(eps, DELTA, q, T_steps)
    print(f"[P11-E1] sigma={sigma:.4f}, T={T_steps}, q={q:.4f}")

    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    priv_loader = DataLoader(priv_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in priv_loader:
            optimizer.zero_grad(set_to_none=True)
            B      = x.shape[0]
            grads  = _per_sample_grads_all(model, x, y, device)
            norms  = grads.norm(dim=1, keepdim=True).clamp(min=1e-8)
            clipped = grads * (CLIP_C / norms).clamp(max=1.0)
            sum_g  = clipped.sum(0)
            noise  = torch.randn_like(sum_g) * (sigma * CLIP_C)
            flat_g = (sum_g + noise) / B
            _set_grads(model, flat_g.to(device))
            optimizer.step()
        scheduler.step()

        if epoch in CHECKPOINT_EPOCHS:
            p = ckpt_paths[epoch]
            torch.save(model.state_dict(), p)
            acc = _evaluate(model, test_ds, device)
            print(f"  ep {epoch:3d}/{EPOCHS}  acc={acc:.4f}  saved {p}")
        elif epoch % 10 == 0:
            acc = _evaluate(model, test_ds, device)
            print(f"  ep {epoch:3d}/{EPOCHS}  acc={acc:.4f}")

    return ckpt_paths


# ---------------------------------------------------------------------------
# PCA subspace from public gradients
# ---------------------------------------------------------------------------

def _compute_public_pca(model, pub_x, pub_y, device, max_rank=MAX_RANK):
    """
    Compute top-max_rank right singular vectors of the (clipped) public
    gradient matrix G ∈ R^{N_pub × d}.  Returns V ∈ R^{d × max_rank} on CPU.
    """
    print(f"[P11-E1] Computing public PCA (max_rank={max_rank})...")
    G = _per_sample_grads_all(model, pub_x, pub_y, device)  # [N, d] CPU
    # Clip each gradient
    norms  = G.norm(dim=1, keepdim=True).clamp(min=1e-8)
    G_clip = G * (CLIP_C / norms).clamp(max=1.0)            # [N, d]
    del G; torch.cuda.empty_cache()

    # Truncated SVD: top-max_rank right singular vectors
    k = min(max_rank, G_clip.shape[0] - 1, G_clip.shape[1])
    _, _, V = torch.svd_lowrank(G_clip.float(), q=k, niter=6)
    # V: [d, k]
    V = V[:, :k].cpu()
    print(f"[P11-E1] PCA done: V shape {V.shape}")
    return V   # [d, min(max_rank, k)]


# ---------------------------------------------------------------------------
# Beta computation for subsampled private examples
# ---------------------------------------------------------------------------

def _compute_beta_spectrum(model, sub_ds, device, V_pub, ranks):
    """
    For each private example in sub_ds, compute per-sample clipped gradient
    and beta_i^(r) for each rank r.

    Returns:
      betas: dict {r: np.array([beta_i for each example])}
    """
    n  = len(sub_ds)
    xs = torch.stack([sub_ds[i][0] for i in range(n)])
    ys = torch.tensor([sub_ds[i][1] for i in range(n)], dtype=torch.long)

    print(f"[P11-E1] Computing per-sample gradients for {n} examples...")
    G = _per_sample_grads_all(model, xs, ys, device)  # [n, d] CPU
    norms   = G.norm(dim=1, keepdim=True).clamp(min=1e-8)
    G_clip  = G * (CLIP_C / norms).clamp(max=1.0)    # [n, d]
    g_norms2 = G_clip.norm(dim=1).pow(2)              # [n]

    betas = {}
    V = V_pub.float()  # [d, max_rank]
    for r in ranks:
        r_actual = min(r, V.shape[1])
        Vr = V[:, :r_actual]           # [d, r]
        # Projection: proj_i = Vr @ (Vr.T @ g_i)
        # ||P_r g_i||^2 = ||Vr.T g_i||^2
        coords = G_clip.float() @ Vr   # [n, r]
        par_norm2 = coords.pow(2).sum(dim=1)   # [n]
        safe_denom = g_norms2.clamp(min=1e-12)
        beta_r = 1.0 - (par_norm2 / safe_denom)
        beta_r = beta_r.clamp(0.0, 1.0)
        betas[r] = beta_r.numpy()
        print(f"    r={r_actual:4d}  beta_95={np.percentile(betas[r], 95):.4f}"
              f"  beta_mean={betas[r].mean():.4f}  beta_max={betas[r].max():.4f}")

    return betas


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp1(data_root, out_dir, gpu, checkpoint=None, single_epoch=None):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[P11-E1] Device: {device}")

    pub_ds, priv_ds, priv_noaug, test_ds, pub_x, pub_y, priv_idx = \
        _build_datasets(data_root)

    # Subsample private examples (same subsample for all checkpoints)
    sub_ds, sub_targets = _subsample_private(priv_noaug, priv_idx, N_PRIV_SAMPLE)
    print(f"[P11-E1] Private subsample: {len(sub_ds)} examples "
          f"(class dist: {np.bincount(sub_targets, minlength=10).tolist()})")

    # Training / checkpoint discovery
    if checkpoint is not None:
        # Single external checkpoint
        epochs_to_run = [single_epoch] if single_epoch else [60]
        ckpt_paths = {epochs_to_run[0]: checkpoint}
    else:
        ckpt_paths = _train_vanilla_warm(
            priv_ds, test_ds, pub_x, pub_y, device, out_dir)
        epochs_to_run = CHECKPOINT_EPOCHS

    # Output CSV
    csv_path = os.path.join(out_dir, "beta_spectrum.csv")
    rows_exist = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                rows_exist.add((int(row["epoch"]), int(row["r"])))

    fieldnames = ["epoch", "r", "beta_max", "beta_99", "beta_95", "beta_90",
                  "beta_mean", "beta_median"]
    csv_file = open(csv_path, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not rows_exist:
        writer.writeheader()

    model = _make_model().to(device)

    for epoch in epochs_to_run:
        ckpt = ckpt_paths.get(epoch)
        if ckpt is None or not os.path.exists(ckpt):
            print(f"[P11-E1] Checkpoint for epoch {epoch} not found, skipping.")
            continue

        print(f"\n[P11-E1] === Epoch {epoch} checkpoint ===")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        # Public PCA
        V_pub = _compute_public_pca(model, pub_x, pub_y, device)

        # Beta spectrum over private subsample
        betas = _compute_beta_spectrum(model, sub_ds, device, V_pub, RANKS)

        # Write to CSV
        for r in RANKS:
            if (epoch, r) in rows_exist:
                continue
            b = betas[r]
            writer.writerow({
                "epoch":       epoch,
                "r":           r,
                "beta_max":    f"{b.max():.6f}",
                "beta_99":     f"{np.percentile(b, 99):.6f}",
                "beta_95":     f"{np.percentile(b, 95):.6f}",
                "beta_90":     f"{np.percentile(b, 90):.6f}",
                "beta_mean":   f"{b.mean():.6f}",
                "beta_median": f"{np.median(b):.6f}",
            })
        csv_file.flush()

        # Also save per-example beta at rank-1 and rank r* (where beta_95 first <= 0.5)
        b1 = betas[1]
        np.save(os.path.join(out_dir, f"beta_rank1_ep{epoch:03d}.npy"), b1)
        # Save all ranks for this epoch
        for r in RANKS:
            np.save(os.path.join(out_dir, f"beta_r{r}_ep{epoch:03d}.npy"), betas[r])

        del V_pub; torch.cuda.empty_cache()

    csv_file.close()
    print(f"\n[P11-E1] Results saved to {csv_path}")

    # Print gate decision
    _print_gate_decision(csv_path)


def _print_gate_decision(csv_path):
    if not os.path.exists(csv_path):
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    print(f"\n{'='*70}")
    print(" Beta Spectrum Summary (epoch 60)")
    print(f"{'='*70}")
    print(f"  {'r':>6}  {'beta_95':>10}  {'beta_mean':>10}  "
          f"{'noise_red_95':>14}  {'gate':>6}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*14}  {'-'*6}")

    gate_passed = False
    for row in rows:
        if int(row["epoch"]) != 60:
            continue
        r   = int(row["r"])
        b95 = float(row["beta_95"])
        bmu = float(row["beta_mean"])
        nr  = math.sqrt(b95)
        gate = "PASS" if b95 <= 0.5 else ("WARN" if b95 <= 0.85 else "FAIL")
        if b95 <= 0.5:
            gate_passed = True
        print(f"  {r:>6}  {b95:>10.4f}  {bmu:>10.4f}  {nr:>14.4f}  {gate:>6}")

    print()
    if gate_passed:
        print("  GATE PASSED: beta_95 <= 0.5 for some r* → proceed to Exp 2.")
    else:
        print("  GATE FAILED: beta_95 > 0.5 for all r → reframe as understanding contribution.")


# ---------------------------------------------------------------------------
# Analysis / plotting
# ---------------------------------------------------------------------------

def _run_analysis(out_dir):
    csv_path = os.path.join(out_dir, "beta_spectrum.csv")
    if not os.path.exists(csv_path):
        print("[P11-E1] No results found.")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    print(f"\n{'='*70}")
    print(" Beta Spectrum by Epoch and Rank")
    print(f"{'='*70}")
    epochs_seen = sorted({int(r["epoch"]) for r in rows})
    ranks_seen  = sorted({int(r["r"]) for r in rows})
    print(f"  Epochs: {epochs_seen}")
    print(f"  Ranks:  {ranks_seen}")

    for ep in epochs_seen:
        print(f"\n  Epoch {ep}:")
        print(f"    {'r':>6}  {'beta_95':>10}  {'beta_mean':>10}  {'noise_red':>10}")
        for row in rows:
            if int(row["epoch"]) != ep:
                continue
            r   = int(row["r"])
            b95 = float(row["beta_95"])
            bmu = float(row["beta_mean"])
            print(f"    {r:>6}  {b95:>10.4f}  {bmu:>10.4f}  {math.sqrt(b95):>10.4f}")

    _print_gate_decision(csv_path)
    _plot_beta_spectra(rows, out_dir)


def _plot_beta_spectra(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs_seen = sorted({int(r["epoch"]) for r in rows})
    ranks_seen  = sorted({int(r["r"]) for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(epochs_seen)))

    # Left: beta_95 vs rank (by epoch)
    ax = axes[0]
    for ep, color in zip(epochs_seen, colors):
        r_vals  = []
        b95_vals = []
        for row in rows:
            if int(row["epoch"]) == ep:
                r_vals.append(int(row["r"]))
                b95_vals.append(float(row["beta_95"]))
        r_vals, b95_vals = zip(*sorted(zip(r_vals, b95_vals)))
        ax.semilogx(r_vals, b95_vals, "o-", label=f"ep {ep}", color=color, lw=1.5)
    ax.axhline(0.5, color="red", ls="--", lw=1, label="gate (0.5)")
    ax.axhline(0.85, color="orange", ls="--", lw=1, label="warn (0.85)")
    ax.set_xlabel("Rank r")
    ax.set_ylabel(r"$\hat{\beta}^{95,(r)}$")
    ax.set_title(r"$\beta_{95}^{(r)}$ vs rank (lower = more coherent)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: noise reduction factor sqrt(beta_95) vs rank at epoch 60
    ax = axes[1]
    ep60_rows = [row for row in rows if int(row["epoch"]) == max(epochs_seen)]
    if ep60_rows:
        r_vals   = sorted([int(row["r"]) for row in ep60_rows])
        nr_vals  = [math.sqrt(float(next(r for r in ep60_rows
                                        if int(r["r"]) == rv)["beta_95"]))
                    for rv in r_vals]
        ax.semilogx(r_vals, nr_vals, "s-", color="steelblue", lw=2)
        ax.axhline(0.71, color="red", ls="--", lw=1, label=r"$\sqrt{0.5}=0.71$")
        ax.axhline(0.92, color="orange", ls="--", lw=1, label=r"$\sqrt{0.85}=0.92$")
        ax.set_xlabel("Rank r")
        ax.set_ylabel(r"$\sqrt{\hat{\beta}^{95,(r)}}$ (noise multiplier reduction)")
        ax.set_title("Effective noise reduction at epoch 60")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "beta_spectra.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P11-E1] Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",          type=int,   default=0)
    parser.add_argument("--data_root",    type=str,   default=DATA_ROOT)
    parser.add_argument("--out_dir",      type=str,
                        default=os.path.join(RESULTS_DIR, "exp1"))
    parser.add_argument("--checkpoint",   type=str,   default=None,
                        help="Path to existing checkpoint (.pt). Skips training.")
    parser.add_argument("--epoch",        type=int,   default=60,
                        help="Epoch label for external checkpoint (default 60).")
    parser.add_argument("--analysis_only", action="store_true")
    args = parser.parse_args()

    if args.analysis_only:
        _run_analysis(args.out_dir)
        return

    run_exp1(data_root=args.data_root, out_dir=args.out_dir, gpu=args.gpu,
             checkpoint=args.checkpoint,
             single_epoch=args.epoch if args.checkpoint else None)
    _run_analysis(args.out_dir)


if __name__ == "__main__":
    main()
