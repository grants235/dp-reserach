"""
Phase 7: CW-GEP Validation
==========================
Compares Coherence-Weighted GEP (CW-GEP) against standard GEP and vanilla DP-SGD
on balanced CIFAR-10 and long-tailed CIFAR-10 (IR=50, IR=100).

Arms:
  vanilla          – standard DP-SGD (clip=1, no subspace)
  gep              – GEP reproduced (clip0=5, clip1=2, variance PCA)
  cw_gep           – CW-GEP: unit-norm normalization before PCA
  gep_opt_split    – GEP + analytically optimal noise split
  cw_gep_opt_split – CW-GEP + optimal noise split
  cw_gep_half_pub  – CW-GEP with half public data (ablation)
  gep_random_sub   – GEP with random subspace (ablation)

Usage:
  # Single run (enables parallel dispatch)
  python experiments/exp_p7_cwgep.py --arm cw_gep --ir 50 --eps 4.0 --seed 0

  # Analysis only (all CSVs already present)
  python experiments/exp_p7_cwgep.py --analysis_only

  # Full sweep (sequential, slow)
  python experiments/exp_p7_cwgep.py --all
"""

import os
import sys
import csv
import gc
import math
import argparse
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import ResNet20
from src.datasets import (
    get_transforms,
    make_cifar10_lt_indices,
    make_public_private_split,
    IndexedDataset,
)

# ---------------------------------------------------------------------------
# Hyperparameters (fixed)
# ---------------------------------------------------------------------------

EPOCHS        = 60
BATCH_SIZE    = 1000          # match GEP paper
LR            = 0.1           # SGD, cosine decay
MOMENTUM      = 0.9
WEIGHT_DECAY  = 5e-4
CLIP_VAN      = 1.0           # vanilla clipping norm
CLIP0         = 5.0           # GEP embedding clip
CLIP1         = 2.0           # GEP residual clip
R_DIM         = 1000          # subspace rank
N_PUB         = 2000          # public anchor examples
DELTA         = 1e-5
EPS_LIST      = [2.0, 4.0, 8.0]
IR_LIST       = [1.0, 50.0, 100.0]
N_SEEDS       = 3
GRAD_CHUNK    = 64            # per-sample grad chunk size (memory)
SUB_REFRESH   = 1             # rebuild subspace every N epochs
DATA_ROOT     = "./data"
RESULTS_DIR   = "./results/exp_p7"

C_MAX_SQ      = CLIP0 ** 2 + CLIP1 ** 2   # = 29.0
C_MAX         = math.sqrt(C_MAX_SQ)         # ≈ 5.385

# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------

ARMS = {
    "vanilla":           {"gep": False, "cw": False, "opt_split": False,
                          "random_sub": False, "n_pub": N_PUB},
    "gep":               {"gep": True,  "cw": False, "opt_split": False,
                          "random_sub": False, "n_pub": N_PUB},
    "cw_gep":            {"gep": True,  "cw": True,  "opt_split": False,
                          "random_sub": False, "n_pub": N_PUB},
    "gep_opt_split":     {"gep": True,  "cw": False, "opt_split": True,
                          "random_sub": False, "n_pub": N_PUB},
    "cw_gep_opt_split":  {"gep": True,  "cw": True,  "opt_split": True,
                          "random_sub": False, "n_pub": N_PUB},
    "cw_gep_half_pub":   {"gep": True,  "cw": True,  "opt_split": False,
                          "random_sub": False, "n_pub": N_PUB // 2},
    "gep_random_sub":    {"gep": True,  "cw": False, "opt_split": False,
                          "random_sub": True,  "n_pub": N_PUB},
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _cifar10_full(data_root, train=True, augment=False):
    transforms = get_transforms(augment=augment)
    return torchvision.datasets.CIFAR10(
        root=data_root, train=train, download=True, transform=transforms
    )


def _build_datasets(ir, data_root, seed=42):
    """
    Returns (pub_dataset, priv_dataset, test_dataset, class_sizes).
    pub  : N_PUB examples (no augmentation)
    priv : 90 % of the LT-subsampled training set (with augmentation)
    test : full CIFAR-10 test set (no augmentation)
    """
    full_train = _cifar10_full(data_root, train=True, augment=False)
    full_targets = np.array(full_train.targets)

    if ir > 1.0:
        lt_idx = make_cifar10_lt_indices(full_targets, imbalance_ratio=ir, seed=seed)
    else:
        lt_idx = np.arange(len(full_train))

    lt_targets = full_targets[lt_idx]
    pub_idx, priv_idx = make_public_private_split(
        lt_idx, lt_targets, public_frac=0.1, seed=seed
    )

    # Public: no augmentation, first N_PUB examples
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pub_idx))
    pub_idx_use = pub_idx[perm[:N_PUB]]

    pub_dataset  = Subset(full_train, pub_idx_use.tolist())
    priv_dataset = Subset(_cifar10_full(data_root, train=True, augment=True),
                          priv_idx.tolist())
    test_dataset = _cifar10_full(data_root, train=False, augment=False)

    class_sizes = np.bincount(lt_targets, minlength=10)
    return pub_dataset, priv_dataset, test_dataset, class_sizes


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _make_model():
    return ResNet20(num_classes=10, n_groups=16)


def _flatten_params(model):
    """Return all params as a single 1-D tensor (no grad)."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def _param_shapes(model):
    return [p.shape for p in model.parameters()]


def _set_grads(model, flat_grad):
    """Scatter a flat gradient vector back into model.param.grad."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat_grad[offset: offset + n].view(p.shape).clone()
        offset += n


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Per-sample gradient computation  (vmap + grad)
# ---------------------------------------------------------------------------

def _loss_fn(params, buffers, x, y, model):
    """Functional single-example forward for vmap."""
    pred = torch.func.functional_call(model, (params, buffers), x.unsqueeze(0))
    return F.cross_entropy(pred, y.unsqueeze(0))


def _per_sample_grads_chunk(model, x_chunk, y_chunk, device):
    """
    Compute per-sample flat gradients for a small chunk [B_c, C, H, W].
    Returns tensor of shape [B_c, d].
    """
    params  = {k: v for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}

    grad_fn = torch.func.grad(
        lambda p, b, xi, yi: _loss_fn(p, b, xi, yi, model)
    )
    vmapped = torch.func.vmap(grad_fn, in_dims=(None, None, 0, 0))

    x_chunk = x_chunk.to(device)
    y_chunk = y_chunk.to(device)
    with torch.no_grad():
        g_dict = vmapped(params, buffers, x_chunk, y_chunk)

    # Flatten and concatenate all parameter grads
    flat = torch.cat(
        [g_dict[k].view(x_chunk.shape[0], -1) for k in model.state_dict()
         if k in g_dict],
        dim=1,
    )
    return flat  # [B_c, d]


def _per_sample_grads_all(model, x, y, device, chunk=GRAD_CHUNK):
    """
    Compute per-sample gradients for a full batch [B, C, H, W].
    Returns [B, d] on CPU to save GPU memory.
    """
    model.eval()
    parts = []
    for i in range(0, x.shape[0], chunk):
        g = _per_sample_grads_chunk(model, x[i:i+chunk], y[i:i+chunk], device)
        parts.append(g.cpu())
        del g
        torch.cuda.empty_cache()
    return torch.cat(parts, dim=0)  # [B, d] on CPU


# ---------------------------------------------------------------------------
# Subspace construction
# ---------------------------------------------------------------------------

def _build_subspace(model, pub_dataset, r, cw, random_sub, device, seed=0):
    """
    Compute the r-dimensional gradient subspace from public data.

    Returns V: FloatTensor [d, r] on CPU.
    """
    loader = DataLoader(pub_dataset, batch_size=min(256, len(pub_dataset)),
                        shuffle=False, num_workers=2, pin_memory=True)

    model.eval()
    grads_list = []
    for x, y in loader:
        g = _per_sample_grads_all(model, x, y, device)  # [B_pub, d]
        grads_list.append(g)

    G = torch.cat(grads_list, dim=0).float()  # [N_pub, d]
    d = G.shape[1]

    if random_sub:
        # Johnson-Lindenstrauss random subspace (baseline)
        rng_t = torch.Generator()
        rng_t.manual_seed(seed)
        V = torch.randn(d, r, generator=rng_t)
        V, _ = torch.linalg.qr(V)  # orthonormal [d, r]
        return V.cpu()

    if cw:
        # Coherence-weighted: normalize each gradient to unit norm
        norms = G.norm(dim=1, keepdim=True).clamp(min=1e-8)
        G = G / norms

    # Low-rank SVD via torch.svd_lowrank (randomized, fast)
    k = min(r, G.shape[0] - 1)
    _, _, V = torch.svd_lowrank(G, q=k, niter=4)  # V: [d, k]
    return V[:, :r].cpu()


# ---------------------------------------------------------------------------
# Privacy calibration
# ---------------------------------------------------------------------------

def _calibrate_sigma(eps, delta, q, T):
    """
    Returns σ (noise multiplier for sensitivity=1) achieving (eps, delta)-DP
    for Gaussian mechanism with Poisson subsampling via RDP accountant.
    """
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(
        target_epsilon=eps,
        target_delta=delta,
        sample_rate=q,
        steps=T,
        accountant="rdp",
    )


def _gep_sigmas(sigma_van, r, d, opt_split):
    """
    Compute GEP noise multipliers (σ_par, σ_perp) for the two channels.

    Default GEP: σ_par = σ_perp = σ_gep where σ_gep is scaled so that
    total RDP of both channels equals vanilla RDP:
        α*(clip0²+clip1²)/(2σ_gep²) = α*1/(2σ_van²)
        → σ_gep = σ_van * C_MAX

    For optimal split: reallocate the per-channel ε budget according to
        frac_par = clip0*sqrt(r) / (clip0*sqrt(r) + clip1*sqrt(d-r))
    while keeping total ε identical, yielding independent σ_par, σ_perp.
    """
    sigma_gep = sigma_van * C_MAX   # scalar multiplier for both channels

    if not opt_split:
        return sigma_gep, sigma_gep

    # Fractions from the spec's "noise power" formula
    w_par  = CLIP0 * math.sqrt(r)
    w_perp = CLIP1 * math.sqrt(d - r)
    frac_par  = w_par  / (w_par + w_perp)
    frac_perp = w_perp / (w_par + w_perp)

    # From: eps_ch = alpha*clip_ch^2 / (2*sigma_ch^2) = frac_ch * eps_total
    # and   eps_total = alpha*C_MAX^2 / (2*sigma_gep^2)
    # → sigma_ch^2 = sigma_gep^2 * clip_ch^2 / (frac_ch * C_MAX^2)
    sigma_par  = sigma_gep * math.sqrt(CLIP0 ** 2 / (frac_par  * C_MAX_SQ))
    sigma_perp = sigma_gep * math.sqrt(CLIP1 ** 2 / (frac_perp * C_MAX_SQ))
    return sigma_par, sigma_perp


# ---------------------------------------------------------------------------
# Training step: vanilla
# ---------------------------------------------------------------------------

def _vanilla_step(model, x, y, sigma_van, device):
    """
    One DP-SGD step.  Returns flat noised gradient (before dividing by B).
    """
    B = x.shape[0]
    grads = _per_sample_grads_all(model, x, y, device)   # [B, d] on CPU

    # Clip
    norms  = grads.norm(dim=1, keepdim=True).clamp(min=1e-8)
    scale  = (CLIP_VAN / norms).clamp(max=1.0)
    clipped = grads * scale                               # [B, d]

    sum_clipped = clipped.sum(dim=0).to(device)           # [d]
    d = sum_clipped.shape[0]
    noise = torch.randn(d, device=device) * (sigma_van * CLIP_VAN)
    return (sum_clipped + noise) / B                       # [d]


# ---------------------------------------------------------------------------
# Training step: GEP / CW-GEP
# ---------------------------------------------------------------------------

def _gep_step(model, x, y, V, sigma_par, sigma_perp, device):
    """
    One GEP step with subspace V [d, r].
    Returns flat noised gradient (before dividing by B).

    Algorithm per step:
      1. Per-sample grad g_i [d]
      2. Projection coeffs: c_i = V^T g_i  [r]
      3. Residual: g_perp_i = g_i - V c_i  [d]
      4. Clip c_i at CLIP0, g_perp_i at CLIP1
      5. Sum: sum_c [r], sum_perp [d]
      6. Add noise: N(0, (σ_par*CLIP0)^2 I_r), N(0, (σ_perp*CLIP1)^2 I_d)
      7. Reconstruct: V @ sum_c_noisy + sum_perp_noisy
    """
    B = x.shape[0]
    V_dev = V.to(device)          # [d, r]
    r     = V.shape[1]
    d     = V.shape[0]

    grads = _per_sample_grads_all(model, x, y, device)   # [B, d] CPU

    # Move to device in chunks to avoid OOM
    sum_c    = torch.zeros(r, device=device)
    sum_perp = torch.zeros(d, device=device)

    chunk = min(GRAD_CHUNK * 4, B)   # larger chunks for the projection step
    for i in range(0, B, chunk):
        g = grads[i:i+chunk].to(device)                  # [c, d]

        # Project
        c      = g @ V_dev                               # [c, r]
        g_par  = c @ V_dev.T                             # [c, d]
        g_perp = g - g_par                               # [c, d]

        # Clip coefficients in r-dim
        c_norms = c.norm(dim=1, keepdim=True).clamp(min=1e-8)
        c       = c * (CLIP0 / c_norms).clamp(max=1.0)

        # Clip residual in d-dim
        perp_norms = g_perp.norm(dim=1, keepdim=True).clamp(min=1e-8)
        g_perp     = g_perp * (CLIP1 / perp_norms).clamp(max=1.0)

        sum_c    += c.sum(dim=0)
        sum_perp += g_perp.sum(dim=0)
        del g, c, g_par, g_perp
        torch.cuda.empty_cache()

    # Add Gaussian noise to each channel
    sum_c    += torch.randn(r, device=device) * (sigma_par  * CLIP0)
    sum_perp += torch.randn(d, device=device) * (sigma_perp * CLIP1)

    # Reconstruct full gradient
    return (V_dev @ sum_c + sum_perp) / B               # [d]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.shape[0]
    return correct / total


@torch.no_grad()
def _train_loss_on_probe(model, probe_x, probe_y, device):
    model.eval()
    x, y = probe_x.to(device), probe_y.to(device)
    loss = F.cross_entropy(model(x), y)
    return loss.item()


# ---------------------------------------------------------------------------
# Subspace diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def _subspace_diagnostics(V_coh, V_var, model, pub_dataset, device):
    """
    V_coh: [d, r] CW-GEP subspace (coherence-weighted PCA)
    V_var: [d, r] GEP subspace (variance PCA)
    Returns dict with: overlap, coherence_capture_coh, coherence_capture_var,
                       variance_capture_coh, variance_capture_var
    """
    r = V_coh.shape[1]
    V_coh = V_coh.to(device)
    V_var = V_var.to(device)

    # Subspace overlap: ||P_coh P_var||_F^2 / r
    inner = V_coh.T @ V_var        # [r, r]
    overlap = (inner ** 2).sum().item() / r

    # Public gradients (un-normalized) for coherence/variance capture
    loader = DataLoader(pub_dataset, batch_size=256, shuffle=False,
                        num_workers=2, pin_memory=True)
    grads_list = []
    for xb, yb in loader:
        g = _per_sample_grads_all(model, xb, yb, device)
        grads_list.append(g)
    G = torch.cat(grads_list, dim=0).float().to(device)  # [N, d]

    G_mean = G.mean(dim=0)  # [d]  = mean public gradient (coherent direction)
    G_mean_norm2 = (G_mean ** 2).sum().item()

    def _coherence_capture(V):
        # ||P_V G_mean||^2 / ||G_mean||^2
        proj = V @ (V.T @ G_mean)
        return (proj ** 2).sum().item() / max(G_mean_norm2, 1e-12)

    def _variance_capture(V):
        # Fraction of total gradient variance in subspace
        G_c = G - G_mean.unsqueeze(0)
        proj = G_c @ V             # [N, r]
        var_in  = (proj ** 2).sum().item()
        var_tot = (G_c ** 2).sum().item()
        return var_in / max(var_tot, 1e-12)

    diag = {
        "overlap":               overlap,
        "coherence_capture_coh": _coherence_capture(V_coh),
        "coherence_capture_var": _coherence_capture(V_var),
        "variance_capture_coh":  _variance_capture(V_coh),
        "variance_capture_var":  _variance_capture(V_var),
    }
    return diag


# ---------------------------------------------------------------------------
# Per-instance certificates
# ---------------------------------------------------------------------------

@torch.no_grad()
def _per_instance_certs(model, V, eps_par, eps_perp, priv_dataset, device):
    """
    Compute per-instance ε_i for all private training examples at the
    final checkpoint.

    eps_i ≈ (Δ_i_par / CLIP0)^2 * eps_par + (Δ_i_perp / CLIP1)^2 * eps_perp

    Returns (eps_i, cos_theta_i) arrays of shape [N_priv].
    """
    V_dev = V.to(device)
    loader = DataLoader(priv_dataset, batch_size=256, shuffle=False,
                        num_workers=2, pin_memory=True)
    eps_i_list = []
    cos_i_list = []

    for xb, yb in loader:
        grads = _per_sample_grads_all(model, xb, yb, device)  # [B, d] cpu
        g     = grads.to(device)

        c     = g @ V_dev                                       # [B, r]
        g_par = c @ V_dev.T                                     # [B, d]
        g_per = g - g_par                                       # [B, d]

        delta_par  = c.norm(dim=1).clamp(max=CLIP0)             # [B]
        delta_perp = g_per.norm(dim=1).clamp(max=CLIP1)         # [B]

        eps_i = (delta_par  / CLIP0) ** 2 * eps_par + \
                (delta_perp / CLIP1) ** 2 * eps_perp             # [B]

        # Coherence angle: cos θ = ||g_par|| / ||g||
        g_norm = g.norm(dim=1).clamp(min=1e-8)
        g_par_norm = g_par.norm(dim=1)
        cos_i = (g_par_norm / g_norm)                            # [B]

        eps_i_list.append(eps_i.cpu())
        cos_i_list.append(cos_i.cpu())
        del g, c, g_par, g_per
        torch.cuda.empty_cache()

    return torch.cat(eps_i_list).numpy(), torch.cat(cos_i_list).numpy()


# ---------------------------------------------------------------------------
# Main training run
# ---------------------------------------------------------------------------

def _train_run(arm_name, ir, eps, seed, pub_dataset, priv_dataset,
               test_dataset, device, out_dir):
    """
    Full training run for one (arm, IR, eps, seed) combination.
    Saves a CSV with per-epoch metrics and the final model checkpoint.
    Returns final test accuracy.
    """
    arm_cfg = ARMS[arm_name]
    tag     = f"{arm_name}_ir{ir:.0f}_eps{eps:.0f}_seed{seed}"
    csv_path = os.path.join(out_dir, f"{tag}.csv")

    if os.path.exists(csv_path):
        print(f"[P7] {tag}: already done, loading CSV")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return float(rows[-1]["test_acc"])

    print(f"\n[P7] === {tag} ===")

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Privacy accounting
    priv_loader_tmp = DataLoader(priv_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, drop_last=True)
    steps_per_epoch = len(priv_loader_tmp)
    T_steps         = EPOCHS * steps_per_epoch
    q               = BATCH_SIZE / len(priv_dataset)
    del priv_loader_tmp

    sigma_van = _calibrate_sigma(eps, DELTA, q, T_steps)
    d         = _num_params(_make_model())

    if arm_cfg["gep"]:
        n_pub_use = arm_cfg["n_pub"]
        # Use a subset of pub_dataset if half-pub ablation
        if n_pub_use < len(pub_dataset):
            rng = np.random.default_rng(seed)
            sub_idx = rng.choice(len(pub_dataset), size=n_pub_use, replace=False)
            pub_use = Subset(pub_dataset, sub_idx.tolist())
        else:
            pub_use = pub_dataset

        sigma_par, sigma_perp = _gep_sigmas(sigma_van, R_DIM, d,
                                             arm_cfg["opt_split"])
    else:
        pub_use = pub_dataset

    # Model, optimizer, scheduler
    model = _make_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Data loaders
    priv_loader = DataLoader(priv_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4, pin_memory=True,
                             drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Probe set for train loss (256 random private examples, fixed)
    probe_idx = np.random.choice(len(priv_dataset), size=256, replace=False)
    probe_dl  = DataLoader(Subset(priv_dataset, probe_idx.tolist()),
                           batch_size=256, shuffle=False)
    probe_x, probe_y = next(iter(probe_dl))
    probe_x, probe_y = probe_x.to(device), probe_y.to(device)

    # CSV writer
    fieldnames = ["epoch", "train_loss", "test_acc", "lr"]
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    best_acc = 0.0
    V = None  # subspace, built lazily

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Rebuild subspace for GEP arms every SUB_REFRESH epochs
        if arm_cfg["gep"] and (V is None or (epoch - 1) % SUB_REFRESH == 0):
            V = _build_subspace(model, pub_use, R_DIM,
                                cw=arm_cfg["cw"],
                                random_sub=arm_cfg["random_sub"],
                                device=device, seed=seed)
            # V is [d, r] on CPU

        # Training steps
        for x, y in priv_loader:
            optimizer.zero_grad(set_to_none=True)

            if arm_cfg["gep"]:
                flat_grad = _gep_step(model, x, y, V,
                                      sigma_par, sigma_perp, device)
            else:
                flat_grad = _vanilla_step(model, x, y, sigma_van, device)

            _set_grads(model, flat_grad)
            optimizer.step()

        scheduler.step()

        # Metrics
        train_loss = _train_loss_on_probe(model, probe_x, probe_y, device)
        test_acc   = _evaluate(model, test_loader, device)
        cur_lr     = scheduler.get_last_lr()[0]

        row = {
            "epoch":      epoch,
            "train_loss": f"{train_loss:.4f}",
            "test_acc":   f"{test_acc:.4f}",
            "lr":         f"{cur_lr:.6f}",
        }
        writer.writerow(row)
        csv_file.flush()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f"{tag}_best.pt"))

        print(f"  epoch {epoch:3d}/{EPOCHS}  "
              f"loss={train_loss:.4f}  acc={test_acc:.4f}  "
              f"best={best_acc:.4f}")

    csv_file.close()
    torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}_final.pt"))
    print(f"  [P7] done — final={test_acc:.4f}  best={best_acc:.4f}")
    return test_acc


# ---------------------------------------------------------------------------
# Subspace diagnostics runner (once per ir/eps)
# ---------------------------------------------------------------------------

def _run_subspace_diagnostics(ir, eps, pub_dataset, priv_dataset, device, out_dir):
    """
    Compute subspace diagnostics comparing GEP and CW-GEP subspaces.
    Saves results as pkl and prints summary.
    """
    tag  = f"subspace_ir{ir:.0f}_eps{eps:.0f}"
    path = os.path.join(out_dir, f"{tag}.pkl")
    if os.path.exists(path):
        print(f"[P7] {tag}: subspace diagnostics already done")
        with open(path, "rb") as f:
            return pickle.load(f)

    print(f"\n[P7] Computing subspace diagnostics for IR={ir:.0f} eps={eps:.0f}")

    # Use a freshly initialized model (epoch 0 reference)
    model = _make_model().to(device)

    V_var = _build_subspace(model, pub_dataset, R_DIM, cw=False,
                            random_sub=False, device=device, seed=42)
    V_coh = _build_subspace(model, pub_dataset, R_DIM, cw=True,
                            random_sub=False, device=device, seed=42)

    diag = _subspace_diagnostics(V_coh, V_var, model, pub_dataset, device)
    diag["ir"]  = ir
    diag["eps"] = eps

    with open(path, "wb") as f:
        pickle.dump(diag, f)

    print(f"  overlap={diag['overlap']:.4f}  "
          f"coh_cap_coh={diag['coherence_capture_coh']:.4f}  "
          f"coh_cap_var={diag['coherence_capture_var']:.4f}")
    return diag


# ---------------------------------------------------------------------------
# Per-instance certificate runner
# ---------------------------------------------------------------------------

def _run_certificates(arm_name, ir, eps, seed, pub_dataset, priv_dataset,
                      device, out_dir):
    """
    Load the final checkpoint and compute per-instance privacy certificates.
    Only meaningful for GEP-based arms.
    """
    tag      = f"{arm_name}_ir{ir:.0f}_eps{eps:.0f}_seed{seed}"
    cert_path = os.path.join(out_dir, f"{tag}_certs.pkl")
    ckpt_path = os.path.join(out_dir, f"{tag}_final.pt")

    if not os.path.exists(ckpt_path):
        print(f"[P7] Certificate: no checkpoint for {tag}")
        return None
    if os.path.exists(cert_path):
        with open(cert_path, "rb") as f:
            return pickle.load(f)

    print(f"[P7] Computing certificates for {tag}")

    model = _make_model().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    V = _build_subspace(model, pub_dataset, R_DIM,
                        cw=ARMS[arm_name]["cw"],
                        random_sub=ARMS[arm_name]["random_sub"],
                        device=device, seed=seed)

    d = _num_params(model)
    priv_loader_tmp = DataLoader(priv_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, drop_last=False)
    steps_per_epoch = math.ceil(len(priv_dataset) / BATCH_SIZE)
    T_steps = EPOCHS * steps_per_epoch
    q = BATCH_SIZE / len(priv_dataset)
    del priv_loader_tmp

    sigma_van = _calibrate_sigma(eps, DELTA, q, T_steps)
    sigma_par, sigma_perp = _gep_sigmas(sigma_van, R_DIM, d,
                                         ARMS[arm_name]["opt_split"])

    # Approximate per-channel total ε (shared across all steps, proportional)
    # Total ε ≈ eps; split by RDP fraction
    eps_par  = CLIP0 ** 2 / C_MAX_SQ * eps
    eps_perp = CLIP1 ** 2 / C_MAX_SQ * eps

    eps_i, cos_i = _per_instance_certs(model, V, eps_par, eps_perp,
                                        priv_dataset, device)
    result = {"eps_i": eps_i, "cos_i": cos_i, "tag": tag}
    with open(cert_path, "wb") as f:
        pickle.dump(result, f)
    return result


# ---------------------------------------------------------------------------
# Analysis and reporting
# ---------------------------------------------------------------------------

def _load_csv(arm, ir, eps, seed, out_dir):
    path = os.path.join(out_dir, f"{arm}_ir{ir:.0f}_eps{eps:.0f}_seed{seed}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _final_acc(rows):
    if rows is None: return None
    return float(rows[-1]["test_acc"])


def _best_acc(rows):
    if rows is None: return None
    return max(float(r["test_acc"]) for r in rows)


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None, None
    return np.mean(vals), np.std(vals)


def _print_table(title, acc_fn, arm_names, ir_list, eps_list, n_seeds, out_dir):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")
    header = f"{'arm':<20}" + "".join(
        f"  IR={ir:.0f}/eps={e:.0f}  " for ir in ir_list for e in eps_list
    )
    print(header)
    print("-" * len(header))
    for arm in arm_names:
        row = f"{arm:<20}"
        for ir in ir_list:
            for eps in eps_list:
                accs = [acc_fn(_load_csv(arm, ir, eps, s, out_dir))
                        for s in range(n_seeds)]
                mu, sd = _mean_std(accs)
                if mu is None:
                    row += f"  {'N/A':>14}  "
                else:
                    row += f"  {mu*100:5.2f}±{sd*100:.2f}%  "
        print(row)


def _print_gap_table(arm_names, ir_list, eps_list, n_seeds, out_dir):
    """Print (arm - GEP) gap and (arm - vanilla) gap."""
    print(f"\n{'='*80}")
    print(" Accuracy gaps vs GEP and vs vanilla (final acc, pp)")
    print(f"{'='*80}")
    header = f"{'arm':<20}" + "".join(
        f"  IR={ir:.0f}/eps={e:.0f}  " for ir in ir_list for e in eps_list
    )
    print(header + "  [vs GEP / vs vanilla]")
    print("-" * len(header))

    for arm in arm_names:
        if arm in ("vanilla", "gep"): continue
        row = f"{arm:<20}"
        for ir in ir_list:
            for eps in eps_list:
                arm_accs = [_final_acc(_load_csv(arm,      ir, eps, s, out_dir)) for s in range(n_seeds)]
                gep_accs = [_final_acc(_load_csv("gep",    ir, eps, s, out_dir)) for s in range(n_seeds)]
                van_accs = [_final_acc(_load_csv("vanilla",ir, eps, s, out_dir)) for s in range(n_seeds)]
                arm_mu, _ = _mean_std(arm_accs)
                gep_mu, _ = _mean_std(gep_accs)
                van_mu, _ = _mean_std(van_accs)
                if arm_mu is None:
                    row += f"  {'N/A':>14}  "
                else:
                    vs_gep = (arm_mu - gep_mu) * 100 if gep_mu else float("nan")
                    vs_van = (arm_mu - van_mu) * 100 if van_mu else float("nan")
                    row += f"  {vs_gep:+5.2f}/{vs_van:+5.2f}pp "
        print(row)


def _plot_curves(arm_names, ir, eps, n_seeds, out_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[P7] matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(arm_names)))

    for ax, (metric, ylabel, acc_fn) in zip(
        axes,
        [("test_acc", "Test Accuracy", lambda r: float(r["test_acc"])),
         ("train_loss", "Train Loss", lambda r: float(r["train_loss"]))],
    ):
        for arm, color in zip(arm_names, colors):
            acc_runs = []
            for s in range(n_seeds):
                rows = _load_csv(arm, ir, eps, s, out_dir)
                if rows is None: continue
                acc_runs.append([acc_fn(r) for r in rows])
            if not acc_runs: continue
            arr  = np.array(acc_runs)
            ep   = np.arange(1, arr.shape[1] + 1)
            mu   = arr.mean(axis=0)
            sd   = arr.std(axis=0)
            ax.plot(ep, mu, label=arm, color=color, lw=1.5)
            ax.fill_between(ep, mu - sd, mu + sd, alpha=0.15, color=color)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"IR={ir:.0f}, ε={eps:.0f}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, f"curves_ir{ir:.0f}_eps{eps:.0f}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P7] Saved {path}")


def _plot_certificate(arm_names, ir, eps, seed, out_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    gep_arms = [a for a in arm_names if ARMS[a]["gep"]]
    n = len(gep_arms)
    if n == 0: return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1: axes = [axes]

    for ax, arm in zip(axes, gep_arms):
        path = os.path.join(out_dir,
               f"{arm}_ir{ir:.0f}_eps{eps:.0f}_seed{seed}_certs.pkl")
        if not os.path.exists(path): continue
        with open(path, "rb") as f:
            res = pickle.load(f)
        ax.scatter(res["cos_i"], res["eps_i"], alpha=0.3, s=2, rasterized=True)
        ax.set_xlabel("Coherence cos θ")
        ax.set_ylabel("Per-instance ε_i")
        ax.set_title(arm)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Per-instance certificates: IR={ir:.0f}, ε={eps:.0f}, seed={seed}")
    fig.tight_layout()
    path = os.path.join(out_dir, f"certs_ir{ir:.0f}_eps{eps:.0f}_seed{seed}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P7] Saved {path}")


def _plot_subspace_overlap(ir_list, eps_list, out_dir):
    """Bar chart of subspace overlap (GEP vs CW-GEP) by IR."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x  = np.arange(len(ir_list))
    w  = 0.25
    colors = plt.cm.tab10(np.linspace(0, 1, len(eps_list)))

    for i, eps in enumerate(eps_list):
        overlaps = []
        for ir in ir_list:
            path = os.path.join(out_dir, f"subspace_ir{ir:.0f}_eps{eps:.0f}.pkl")
            if not os.path.exists(path):
                overlaps.append(float("nan"))
                continue
            with open(path, "rb") as f:
                d = pickle.load(f)
            overlaps.append(d["overlap"])
        ax.bar(x + i * w, overlaps, w, label=f"ε={eps:.0f}", color=colors[i])

    ax.set_xticks(x + w)
    ax.set_xticklabels([f"IR={ir:.0f}" for ir in ir_list])
    ax.set_ylabel("Subspace overlap (GEP ∩ CW-GEP) / r")
    ax.set_title("Subspace overlap: variance PCA vs coherence-weighted PCA")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(out_dir, "subspace_overlap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P7] Saved {path}")


def _run_analysis(arm_names, ir_list, eps_list, n_seeds, out_dir):
    _print_table("Final test accuracy (epoch 60)", _final_acc,
                 arm_names, ir_list, eps_list, n_seeds, out_dir)
    _print_table("Best test accuracy (any epoch)", _best_acc,
                 arm_names, ir_list, eps_list, n_seeds, out_dir)
    _print_gap_table(arm_names, ir_list, eps_list, n_seeds, out_dir)

    # Subspace diagnostics summary
    print(f"\n{'='*80}")
    print(" Subspace diagnostics")
    print(f"{'='*80}")
    for ir in ir_list:
        for eps in eps_list[:1]:  # just eps=4 for brevity
            path = os.path.join(out_dir, f"subspace_ir{ir:.0f}_eps{eps:.0f}.pkl")
            if not os.path.exists(path): continue
            with open(path, "rb") as f:
                d = pickle.load(f)
            print(f"  IR={ir:.0f} ε={eps:.0f}: "
                  f"overlap={d['overlap']:.3f}  "
                  f"coh_cap[coh]={d['coherence_capture_coh']:.3f}  "
                  f"coh_cap[var]={d['coherence_capture_var']:.3f}  "
                  f"var_cap[coh]={d['variance_capture_coh']:.3f}  "
                  f"var_cap[var]={d['variance_capture_var']:.3f}")

    # Figures
    for ir in ir_list:
        for eps in eps_list:
            _plot_curves(arm_names, ir, eps, n_seeds, out_dir)
    _plot_subspace_overlap(ir_list, eps_list, out_dir)
    for ir in ir_list:
        _plot_certificate(
            [a for a in arm_names if ARMS[a]["gep"]],
            ir, eps_list[1], 0, out_dir
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 7: CW-GEP validation")
    ap.add_argument("--arm",   choices=list(ARMS.keys()),
                    help="Run a single arm")
    ap.add_argument("--ir",   type=float, choices=IR_LIST,
                    help="Imbalance ratio (1, 50, 100)")
    ap.add_argument("--eps",  type=float, choices=EPS_LIST,
                    help="Privacy budget ε")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed (0, 1, 2)")
    ap.add_argument("--all",  action="store_true",
                    help="Run all arms × IRs × ε × seeds sequentially")
    ap.add_argument("--analysis_only", action="store_true",
                    help="Skip training, just print tables and generate figures")
    ap.add_argument("--subspace_diag", action="store_true",
                    help="Compute subspace diagnostics for all IR × ε")
    ap.add_argument("--certificates",  action="store_true",
                    help="Compute per-instance certificates for GEP arms")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--out_dir",   default=RESULTS_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[P7] Device: {device}")

    arm_names = list(ARMS.keys())

    if args.analysis_only:
        _run_analysis(arm_names, IR_LIST, EPS_LIST, N_SEEDS, args.out_dir)
        return

    if args.all:
        # Sequential sweep over all combinations
        for ir in IR_LIST:
            pub_ds, priv_ds, test_ds, _ = _build_datasets(ir, args.data_root)
            # Subspace diagnostics (once per IR × ε)
            for eps in EPS_LIST:
                _run_subspace_diagnostics(ir, eps, pub_ds, priv_ds,
                                          device, args.out_dir)
            # Training
            for arm in arm_names:
                for eps in EPS_LIST:
                    for seed in range(N_SEEDS):
                        _train_run(arm, ir, eps, seed,
                                   pub_ds, priv_ds, test_ds,
                                   device, args.out_dir)

        # Certificates for GEP arms
        for ir in IR_LIST:
            pub_ds, priv_ds, test_ds, _ = _build_datasets(ir, args.data_root)
            for arm in arm_names:
                if not ARMS[arm]["gep"]: continue
                for eps in EPS_LIST:
                    _run_certificates(arm, ir, eps, 0,
                                      pub_ds, priv_ds, device, args.out_dir)

        _run_analysis(arm_names, IR_LIST, EPS_LIST, N_SEEDS, args.out_dir)
        return

    # Single run mode (useful for parallel dispatch)
    if args.arm is None or args.ir is None or args.eps is None:
        # Default: analysis only
        _run_analysis(arm_names, IR_LIST, EPS_LIST, N_SEEDS, args.out_dir)
        return

    pub_ds, priv_ds, test_ds, class_sizes = _build_datasets(
        args.ir, args.data_root
    )
    print(f"[P7] Class sizes: {class_sizes}")
    print(f"[P7] Public: {len(pub_ds)}  Private: {len(priv_ds)}  "
          f"Test: {len(test_ds)}")

    if args.subspace_diag:
        _run_subspace_diagnostics(args.ir, args.eps, pub_ds, priv_ds,
                                   device, args.out_dir)
        return

    if args.certificates:
        _run_certificates(args.arm, args.ir, args.eps, args.seed,
                          pub_ds, priv_ds, device, args.out_dir)
        return

    _train_run(args.arm, args.ir, args.eps, args.seed,
               pub_ds, priv_ds, test_ds, device, args.out_dir)


if __name__ == "__main__":
    main()
