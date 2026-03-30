"""
Phase 8: Coherence-Informed Wiener Denoising for DP-SGD Post-Processing
=========================================================================

Applies Wiener shrinkage to the noised aggregate produced by vanilla DP-SGD
or GEP.  Because post-processing preserves DP, all Wiener arms have the same
(ε,δ) guarantee as their base method.

Arms
----
  vanilla       – standard DP-SGD (reuse P7 if results/exp_p7/ present)
  gep           – GEP, variance-PCA subspace (reuse P7 if present)
  vanilla_W     – vanilla + adaptive Wiener (variance-PCA subspace)
  vanilla_CW    – vanilla + adaptive Wiener (coherence-PCA subspace)
  gep_W         – GEP + adaptive Wiener (variance-PCA)
  gep_CW        – GEP + adaptive Wiener (coherence-PCA)
  vanilla_proj  – vanilla + hard projection (w_perp=0, w_par=1)
  vanilla_W50   – vanilla + fixed half shrinkage (w_par=1, w_perp=0.5)

Usage
-----
  # Single arm run (preferred for parallel dispatch)
  python experiments/exp_p8_wiener.py --arm vanilla_W --ir 1 --eps 4 --seed 0

  # Analysis only (tables + figures from existing CSVs)
  python experiments/exp_p8_wiener.py --analysis_only

  # Full sweep (sequential)
  python experiments/exp_p8_wiener.py --all

  # Transfer P7 baselines (vanilla + gep) from a results/exp_p7 directory
  python experiments/exp_p8_wiener.py --transfer_baselines

Reuse logic
-----------
For the 'vanilla' and 'gep' arms (identical to P7 arms), the script checks
results/exp_p7/{tag}.csv first.  If found it is copied into results/exp_p8/
and no training is run.  The check validates that P7 used the same
hyperparameters by reading a sentinel header comment baked into the CSV.
"""

import os
import sys
import csv
import gc
import math
import shutil
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
# Hyperparameters — must match P7 exactly for valid A/B comparison
# ---------------------------------------------------------------------------

EPOCHS       = 60
BATCH_SIZE   = 1000           # match GEP paper / P7
LR           = 0.1
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4
CLIP_VAN     = 1.0            # vanilla clipping norm
CLIP0        = 5.0            # GEP embedding clip
CLIP1        = 2.0            # GEP residual clip
R_DIM        = 1000           # subspace rank
N_PUB        = 2000           # public examples used to build subspace
DELTA        = 1e-5
EPS_LIST     = [2.0, 4.0, 8.0]
IR_LIST      = [1.0, 50.0]   # IR=100 dropped per spec (uninformative in P7)
N_SEEDS      = 3
GRAD_CHUNK   = 64             # per-sample grad vmap chunk
SUB_REFRESH  = 1              # rebuild subspace every N epochs
DATA_ROOT    = "./data"
RESULTS_DIR  = "./results/exp_p8"
P7_DIR       = "./results/exp_p7"   # source of reusable baselines

C_MAX_SQ     = CLIP0 ** 2 + CLIP1 ** 2   # = 29.0
C_MAX        = math.sqrt(C_MAX_SQ)        # ≈ 5.385

# Sentinel string written into P7 CSVs to confirm matching hyperparams
_P7_SENTINEL = (
    f"#p7 EPOCHS={EPOCHS} BATCH={BATCH_SIZE} LR={LR} "
    f"CLIP_VAN={CLIP_VAN} CLIP0={CLIP0} CLIP1={CLIP1} "
    f"R={R_DIM} NPUB={N_PUB} DELTA={DELTA}"
)

# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------
#   gep        : use GEP decomposition (False → vanilla DP-SGD)
#   cw         : use coherence-weighted PCA for subspace (False → variance PCA)
#   wiener     : apply Wiener post-processing
#   wmode      : "adaptive" | "proj" (w_perp=0) | "fixed50" (w_perp=0.5)
#   baseline   : True → can try to reuse P7 result; arm name in P7 same as here

ARMS = {
    "vanilla":      {"gep": False, "cw": False, "wiener": False, "wmode": None,        "baseline": True},
    "gep":          {"gep": True,  "cw": False, "wiener": False, "wmode": None,        "baseline": True},
    "vanilla_W":    {"gep": False, "cw": False, "wiener": True,  "wmode": "adaptive",  "baseline": False},
    "vanilla_CW":   {"gep": False, "cw": True,  "wiener": True,  "wmode": "adaptive",  "baseline": False},
    "gep_W":        {"gep": True,  "cw": False, "wiener": True,  "wmode": "adaptive",  "baseline": False},
    "gep_CW":       {"gep": True,  "cw": True,  "wiener": True,  "wmode": "adaptive",  "baseline": False},
    "vanilla_proj": {"gep": False, "cw": False, "wiener": True,  "wmode": "proj",      "baseline": False},
    "vanilla_W50":  {"gep": False, "cw": False, "wiener": True,  "wmode": "fixed50",   "baseline": False},
}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _cifar10_full(data_root, train=True, augment=False):
    train_tf, test_tf = get_transforms(augment=augment)
    tf = train_tf if train else test_tf
    return torchvision.datasets.CIFAR10(
        root=data_root, train=train, download=True, transform=tf
    )


def _build_datasets(ir, data_root, seed=42):
    """
    Returns (pub_dataset, priv_dataset, test_dataset, class_sizes).
    Identical to P7: 10% public per class, 90% private, seed=42.
    """
    full_train   = _cifar10_full(data_root, train=True, augment=False)
    full_targets = np.array(full_train.targets)

    if ir > 1.0:
        lt_idx = make_cifar10_lt_indices(full_targets, imbalance_ratio=ir, seed=seed)
    else:
        lt_idx = np.arange(len(full_train))

    lt_targets = full_targets[lt_idx]
    pub_idx, priv_idx = make_public_private_split(
        lt_idx, lt_targets, public_frac=0.1, seed=seed
    )

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
# Model helpers (identical to P7)
# ---------------------------------------------------------------------------

def _make_model():
    return ResNet20(num_classes=10, n_groups=16)


def _flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def _set_grads(model, flat_grad):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat_grad[offset: offset + n].view(p.shape).clone()
        offset += n


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Per-sample gradient computation (vmap + grad, identical to P7)
# ---------------------------------------------------------------------------

def _loss_fn(params, buffers, x, y, model):
    pred = torch.func.functional_call(model, (params, buffers), x.unsqueeze(0))
    return F.cross_entropy(pred, y.unsqueeze(0))


def _per_sample_grads_chunk(model, x_chunk, y_chunk, device):
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

    flat = torch.cat(
        [g_dict[k].view(x_chunk.shape[0], -1) for k in model.state_dict()
         if k in g_dict],
        dim=1,
    )
    return flat  # [B_c, d]


def _per_sample_grads_all(model, x, y, device, chunk=GRAD_CHUNK):
    model.eval()
    parts = []
    for i in range(0, x.shape[0], chunk):
        g = _per_sample_grads_chunk(model, x[i:i+chunk], y[i:i+chunk], device)
        parts.append(g.cpu())
        del g
        torch.cuda.empty_cache()
    return torch.cat(parts, dim=0)  # [B, d] on CPU


# ---------------------------------------------------------------------------
# Subspace construction (identical to P7)
# ---------------------------------------------------------------------------

def _build_subspace(model, pub_dataset, r, cw, device, seed=0):
    """
    Returns V: FloatTensor [d, r] orthonormal columns.
    cw=True  → coherence-weighted PCA (unit-norm rows before SVD)
    cw=False → standard variance PCA
    """
    loader = DataLoader(pub_dataset, batch_size=min(256, len(pub_dataset)),
                        shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    grads_list = []
    for x, y in loader:
        g = _per_sample_grads_all(model, x, y, device)
        grads_list.append(g)

    G = torch.cat(grads_list, dim=0).float()  # [N_pub, d]
    d = G.shape[1]

    if cw:
        norms = G.norm(dim=1, keepdim=True).clamp(min=1e-8)
        G = G / norms

    k = min(r, G.shape[0] - 1)
    _, _, V = torch.svd_lowrank(G, q=k, niter=4)  # V: [d, k]
    return V[:, :r].cpu()


# ---------------------------------------------------------------------------
# Privacy calibration (identical to P7)
# ---------------------------------------------------------------------------

def _calibrate_sigma(eps, delta, q, T):
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(
        target_epsilon=eps,
        target_delta=delta,
        sample_rate=q,
        steps=T,
        accountant="rdp",
    )


def _gep_sigmas(sigma_van, r, d, opt_split=False):
    """
    Identical to P7 default (opt_split=False): σ_gep = σ_van * sqrt(2).
    P8 does not use opt_split (that was a P7 ablation arm).
    """
    sigma_gep = sigma_van * math.sqrt(2)
    if not opt_split:
        return sigma_gep, sigma_gep

    w_par  = CLIP0 * math.sqrt(r)
    w_perp = CLIP1 * math.sqrt(d - r)
    frac_par  = w_par  / (w_par + w_perp)
    frac_perp = w_perp / (w_par + w_perp)
    sigma_par  = sigma_van / math.sqrt(frac_par)
    sigma_perp = sigma_van / math.sqrt(frac_perp)
    return sigma_par, sigma_perp


# ---------------------------------------------------------------------------
# Wiener denoising
# ---------------------------------------------------------------------------

def wiener_denoise(noised_grad, basis, N_par, N_perp, wmode="adaptive"):
    """
    Apply Wiener shrinkage to a flat noised gradient (unnormalised sum).

    Args:
        noised_grad : [d] tensor — noised aggregate (before dividing by B)
        basis       : [d, r] tensor — orthonormal subspace columns
        N_par       : float — expected noise power in parallel channel
                      (= σ² C² r  for vanilla;  σ_par² CLIP0² r  for GEP)
        N_perp      : float — expected noise power in perp channel
                      (= σ² C² (d-r)  for vanilla;  σ_perp² CLIP1² (d-r) for GEP)
        wmode       : "adaptive" | "proj" | "fixed50"

    Returns:
        denoised    : [d] tensor
        w_par       : float — Wiener weight applied to coherent channel
        w_perp      : float — Wiener weight applied to incoherent channel
        diagnostics : dict  — S_par, S_perp, N_par, N_perp
    """
    basis = basis.to(noised_grad.device)

    # Decompose
    coeffs = basis.T @ noised_grad   # [r]
    g_par  = basis @ coeffs           # [d] — parallel component
    g_perp = noised_grad - g_par      # [d] — perpendicular component

    # Coherent channel: always keep as-is (w_par = 1 always)
    w_par = 1.0

    # Incoherent channel: Wiener shrinkage on the perp component only
    S_perp = max(0.0, g_perp.norm().item() ** 2 - N_perp)

    if wmode == "proj":
        w_perp = 0.0
    elif wmode == "fixed50":
        w_perp = 0.5
    else:  # adaptive Wiener
        denom_perp = S_perp + N_perp
        w_perp = S_perp / denom_perp if denom_perp > 0 else 0.0

    denoised = g_par + w_perp * g_perp

    # S_par kept for diagnostic logging (not used for weight)
    S_par = max(0.0, coeffs.norm().item() ** 2 - N_par)
    diag = {"S_par": S_par, "S_perp": S_perp,
            "N_par": N_par, "N_perp": N_perp}
    return denoised, w_par, w_perp, diag


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------

def _vanilla_step(model, x, y, sigma_van, device):
    """Standard DP-SGD step.  Returns (flat_noised_grad / B, empty_diag)."""
    B = x.shape[0]
    grads = _per_sample_grads_all(model, x, y, device)  # [B, d] CPU

    norms   = grads.norm(dim=1, keepdim=True).clamp(min=1e-8)
    scale   = (CLIP_VAN / norms).clamp(max=1.0)
    clipped = grads * scale                              # [B, d]

    sum_g = clipped.sum(dim=0).to(device)               # [d]
    d = sum_g.shape[0]
    noise = torch.randn(d, device=device) * (sigma_van * CLIP_VAN)
    noised_sum = sum_g + noise                           # [d] unnormalised
    return noised_sum / B, {}


def _vanilla_wiener_step(model, x, y, sigma_van, V, wmode, device):
    """
    Vanilla DP-SGD + Wiener denoising.
    Wiener operates on the unnormalised noised sum before dividing by B.
    Both channels use the same noise std = sigma_van * CLIP_VAN (isotropic).
    """
    B = x.shape[0]
    grads = _per_sample_grads_all(model, x, y, device)

    norms   = grads.norm(dim=1, keepdim=True).clamp(min=1e-8)
    scale   = (CLIP_VAN / norms).clamp(max=1.0)
    clipped = grads * scale

    sum_g = clipped.sum(dim=0).to(device)
    d = sum_g.shape[0]
    r = V.shape[1]

    noise = torch.randn(d, device=device) * (sigma_van * CLIP_VAN)
    noised_sum = sum_g + noise  # [d] unnormalised

    # Known noise power per channel (in terms of the unnormalised sum)
    N_par  = (sigma_van * CLIP_VAN) ** 2 * r
    N_perp = (sigma_van * CLIP_VAN) ** 2 * (d - r)

    denoised_sum, w_par, w_perp, diag = wiener_denoise(
        noised_sum, V, N_par, N_perp, wmode
    )
    # Shrinkage ratio
    noised_norm = noised_sum.norm().item()
    denoised_norm = denoised_sum.norm().item()
    shrink = denoised_norm / max(noised_norm, 1e-12)

    diag["w_par"]   = w_par
    diag["w_perp"]  = w_perp
    diag["shrink"]  = shrink

    return denoised_sum / B, diag


def _gep_step(model, x, y, V, sigma_par, sigma_perp, device, wiener=False,
              wmode="adaptive"):
    """
    GEP step (optionally + Wiener).
    Returns (flat_noised_grad / B, diag).
    """
    B = x.shape[0]
    V_dev = V.to(device)
    r, d  = V.shape[1], V.shape[0]

    grads = _per_sample_grads_all(model, x, y, device)  # [B, d] CPU

    sum_c    = torch.zeros(r, device=device)
    sum_perp = torch.zeros(d, device=device)

    chunk = min(GRAD_CHUNK * 4, B)
    for i in range(0, B, chunk):
        g = grads[i:i+chunk].to(device)

        c      = g @ V_dev
        g_par  = c @ V_dev.T
        g_perp = g - g_par

        c_norms    = c.norm(dim=1, keepdim=True).clamp(min=1e-8)
        c          = c * (CLIP0 / c_norms).clamp(max=1.0)

        perp_norms = g_perp.norm(dim=1, keepdim=True).clamp(min=1e-8)
        g_perp     = g_perp * (CLIP1 / perp_norms).clamp(max=1.0)

        sum_c    += c.sum(dim=0)
        sum_perp += g_perp.sum(dim=0)
        del g, c, g_par, g_perp
        torch.cuda.empty_cache()

    # Add noise — same structure as P7
    sum_c_noisy = sum_c + torch.randn(r, device=device) * (sigma_par * CLIP0)

    noise_perp = torch.randn(d, device=device) * (sigma_perp * CLIP1)
    noise_perp = noise_perp - V_dev @ (V_dev.T @ noise_perp)
    sum_perp_noisy = sum_perp + noise_perp

    if not wiener:
        result = (V_dev @ sum_c_noisy + sum_perp_noisy) / B
        return result, {}

    # GEP + Wiener: coherent channel always kept (w_par=1),
    # Wiener shrinkage applied to incoherent channel only.
    # Parallel channel: the noised coefficients live in R^r
    N_par_gep  = (sigma_par  * CLIP0) ** 2 * r
    # Perpendicular channel: noise projected onto (d-r)-dim complement
    N_perp_gep = (sigma_perp * CLIP1) ** 2 * (d - r)

    w_par = 1.0  # always keep coherent channel
    S_perp = max(0.0, sum_perp_noisy.norm().item() ** 2 - N_perp_gep)

    if wmode == "proj":
        w_perp = 0.0
    elif wmode == "fixed50":
        w_perp = 0.5
    else:
        denom_perp = S_perp + N_perp_gep
        w_perp = S_perp / denom_perp if denom_perp > 0 else 0.0

    g_par_noised  = V_dev @ sum_c_noisy    # [d]
    g_perp_noised = sum_perp_noisy         # [d]
    denoised_sum  = g_par_noised + w_perp * g_perp_noised

    # S_par kept for diagnostics only
    S_par = max(0.0, sum_c_noisy.norm().item() ** 2 - N_par_gep)

    noised_full = V_dev @ sum_c_noisy + sum_perp_noisy
    shrink = denoised_sum.norm().item() / max(noised_full.norm().item(), 1e-12)

    diag = {
        "w_par":  w_par,
        "w_perp": w_perp,
        "shrink": shrink,
        "S_par":  S_par,
        "S_perp": S_perp,
        "N_par":  N_par_gep,
        "N_perp": N_perp_gep,
    }
    return denoised_sum / B, diag


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total   += y.shape[0]
    return correct / total


@torch.no_grad()
def _train_loss_on_probe(model, probe_x, probe_y, device):
    model.eval()
    return F.cross_entropy(model(probe_x.to(device)), probe_y.to(device)).item()


# ---------------------------------------------------------------------------
# Baseline reuse helpers
# ---------------------------------------------------------------------------

def _try_copy_p7_baseline(arm_name, ir, eps, seed, out_dir, p7_dir=P7_DIR):
    """
    If P7 produced this arm's CSV (and optionally checkpoints), copy them
    into out_dir under the same filename.
    Returns True if the copy succeeded, False otherwise.
    """
    tag     = f"{arm_name}_ir{ir:.0f}_eps{eps:.0f}_seed{seed}"
    src_csv = os.path.join(p7_dir, f"{tag}.csv")
    dst_csv = os.path.join(out_dir, f"{tag}.csv")

    if os.path.exists(dst_csv):
        # Already copied in a previous run
        return True

    if not os.path.exists(src_csv):
        return False

    shutil.copy2(src_csv, dst_csv)
    print(f"[P8] Reused P7 baseline: {src_csv} → {dst_csv}")

    # Optionally copy checkpoints (not required for analysis, nice to have)
    for suffix in ("_best.pt", "_final.pt"):
        src_ckpt = os.path.join(p7_dir,  f"{tag}{suffix}")
        dst_ckpt = os.path.join(out_dir, f"{tag}{suffix}")
        if os.path.exists(src_ckpt) and not os.path.exists(dst_ckpt):
            shutil.copy2(src_ckpt, dst_ckpt)

    return True


# ---------------------------------------------------------------------------
# Main training run
# ---------------------------------------------------------------------------

def _train_run(arm_name, ir, eps, seed, pub_dataset, priv_dataset,
               test_dataset, device, out_dir, p7_dir=P7_DIR):
    """
    Full training run for (arm, IR, eps, seed).
    Returns final test accuracy.
    """
    arm_cfg  = ARMS[arm_name]
    tag      = f"{arm_name}_ir{ir:.0f}_eps{eps:.0f}_seed{seed}"
    csv_path = os.path.join(out_dir, f"{tag}.csv")

    # ---- Attempt P7 reuse for baseline arms ---------------------------------
    if arm_cfg["baseline"]:
        if _try_copy_p7_baseline(arm_name, ir, eps, seed, out_dir, p7_dir):
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            return float(rows[-1]["test_acc"])

    # ---- Already trained in a previous P8 run --------------------------------
    if os.path.exists(csv_path):
        print(f"[P8] {tag}: already done, loading CSV")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return float(rows[-1]["test_acc"])

    print(f"\n[P8] === {tag} ===")

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Privacy accounting (identical to P7)
    priv_loader_tmp = DataLoader(priv_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, drop_last=True)
    steps_per_epoch = len(priv_loader_tmp)
    T_steps         = EPOCHS * steps_per_epoch
    q               = BATCH_SIZE / len(priv_dataset)
    del priv_loader_tmp

    sigma_van = _calibrate_sigma(eps, DELTA, q, T_steps)
    d         = _num_params(_make_model())

    if arm_cfg["gep"]:
        sigma_par, sigma_perp = _gep_sigmas(sigma_van, R_DIM, d)
    else:
        sigma_par = sigma_perp = None

    # Model, optimiser, scheduler (identical to P7)
    model     = _make_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    priv_loader = DataLoader(priv_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4, pin_memory=True,
                             drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Fixed probe set for train-loss tracking
    rng_np = np.random.default_rng(seed)
    probe_idx = rng_np.choice(len(priv_dataset), size=256, replace=False)
    probe_dl  = DataLoader(Subset(priv_dataset, probe_idx.tolist()),
                           batch_size=256, shuffle=False)
    probe_x, probe_y = next(iter(probe_dl))

    # CSV — extra Wiener diagnostic columns (NaN for non-Wiener arms)
    has_wiener = arm_cfg["wiener"]
    fieldnames = ["epoch", "train_loss", "test_acc", "lr"]
    if has_wiener:
        fieldnames += ["w_par", "w_perp", "shrink", "S_par", "S_perp",
                       "N_par", "N_perp"]

    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    best_acc = 0.0
    V = None  # subspace, built lazily; rebuilt every SUB_REFRESH epochs

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Build / refresh subspace for arms that need it
        need_sub = arm_cfg["wiener"] or arm_cfg["gep"]
        if need_sub and (V is None or (epoch - 1) % SUB_REFRESH == 0):
            V = _build_subspace(model, pub_dataset, R_DIM,
                                cw=arm_cfg["cw"], device=device, seed=seed)

        # Accumulate per-epoch Wiener diagnostics
        epoch_diag_accum = {k: [] for k in
                            ["w_par", "w_perp", "shrink", "S_par", "S_perp",
                             "N_par", "N_perp"]}

        for x, y in priv_loader:
            optimizer.zero_grad(set_to_none=True)

            if arm_cfg["gep"]:
                flat_grad, step_diag = _gep_step(
                    model, x, y, V, sigma_par, sigma_perp, device,
                    wiener=arm_cfg["wiener"], wmode=arm_cfg["wmode"]
                )
            elif arm_cfg["wiener"]:
                flat_grad, step_diag = _vanilla_wiener_step(
                    model, x, y, sigma_van, V, arm_cfg["wmode"], device
                )
            else:
                flat_grad, step_diag = _vanilla_step(model, x, y, sigma_van, device)

            _set_grads(model, flat_grad)
            optimizer.step()

            for k, vals in epoch_diag_accum.items():
                if k in step_diag:
                    vals.append(step_diag[k])

        scheduler.step()

        train_loss = _train_loss_on_probe(model, probe_x, probe_y, device)
        test_acc   = _evaluate(model, test_loader, device)
        cur_lr     = scheduler.get_last_lr()[0]

        row = {
            "epoch":      epoch,
            "train_loss": f"{train_loss:.4f}",
            "test_acc":   f"{test_acc:.4f}",
            "lr":         f"{cur_lr:.6f}",
        }
        if has_wiener:
            for k in ["w_par", "w_perp", "shrink", "S_par", "S_perp",
                      "N_par", "N_perp"]:
                vals = epoch_diag_accum[k]
                row[k] = f"{np.mean(vals):.6g}" if vals else "nan"

        writer.writerow(row)
        csv_file.flush()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f"{tag}_best.pt"))

        # Progress
        diag_str = ""
        if has_wiener and epoch_diag_accum["w_par"]:
            wp  = np.mean(epoch_diag_accum["w_par"])
            wpe = np.mean(epoch_diag_accum["w_perp"])
            sh  = np.mean(epoch_diag_accum["shrink"])
            diag_str = f"  w_par={wp:.3f} w_perp={wpe:.3f} shrink={sh:.3f}"
        print(f"  epoch {epoch:3d}/{EPOCHS}  loss={train_loss:.4f}  "
              f"acc={test_acc:.4f}  best={best_acc:.4f}{diag_str}")

    csv_file.close()
    torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}_final.pt"))
    print(f"[P8] done — final={test_acc:.4f}  best={best_acc:.4f}")
    return test_acc


# ---------------------------------------------------------------------------
# Analysis helpers
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
        f"  IR={ir:.0f}/ε={e:.0f}  " for ir in ir_list for e in eps_list
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
    """Gap vs vanilla and vs gep (percentage points, final accuracy)."""
    print(f"\n{'='*80}")
    print(" Δ vs vanilla / Δ vs gep  (pp, final acc, mean over seeds)")
    print(f"{'='*80}")
    for ir in ir_list:
        print(f"\n  IR={ir:.0f}")
        header = f"  {'arm':<20}" + "".join(f"  ε={e:.0f}  " for e in eps_list)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for arm in arm_names:
            if arm in ("vanilla", "gep"): continue
            row = f"  {arm:<20}"
            for eps in eps_list:
                arm_accs = [_final_acc(_load_csv(arm,      ir, eps, s, out_dir)) for s in range(n_seeds)]
                van_accs = [_final_acc(_load_csv("vanilla",ir, eps, s, out_dir)) for s in range(n_seeds)]
                gep_accs = [_final_acc(_load_csv("gep",    ir, eps, s, out_dir)) for s in range(n_seeds)]
                a_mu, _ = _mean_std(arm_accs)
                v_mu, _ = _mean_std(van_accs)
                g_mu, _ = _mean_std(gep_accs)
                if a_mu is None:
                    row += f"  {'N/A':>12}  "
                else:
                    dv = (a_mu - v_mu) * 100 if v_mu is not None else float("nan")
                    dg = (a_mu - g_mu) * 100 if g_mu is not None else float("nan")
                    row += f"  {dv:+5.2f}/{dg:+5.2f}pp "
            print(row)


def _plot_curves(arm_names, ir, eps, n_seeds, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[P8] matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(arm_names)))

    for ax, (metric, ylabel) in zip(
        axes,
        [("test_acc", "Test Accuracy"), ("train_loss", "Train Loss")],
    ):
        for arm, color in zip(arm_names, colors):
            runs = []
            for s in range(n_seeds):
                rows = _load_csv(arm, ir, eps, s, out_dir)
                if rows is None: continue
                try:
                    runs.append([float(r[metric]) for r in rows])
                except (KeyError, ValueError):
                    continue
            if not runs: continue
            arr = np.array(runs)
            ep  = np.arange(1, arr.shape[1] + 1)
            mu  = arr.mean(axis=0)
            sd  = arr.std(axis=0)
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
    print(f"[P8] Saved {path}")


def _plot_wiener_weights(arm_names, ir, eps, n_seeds, out_dir):
    """Plot evolution of w_par and w_perp over training epochs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    wiener_arms = [a for a in arm_names if ARMS[a]["wiener"] and ARMS[a]["wmode"] == "adaptive"]
    if not wiener_arms:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(wiener_arms)))

    for ax, weight_key, ylabel in zip(
        axes,
        ["w_par", "w_perp"],
        ["w_∥ (coherent Wiener weight)", "w_⊥ (incoherent Wiener weight)"],
    ):
        for arm, color in zip(wiener_arms, colors):
            runs = []
            for s in range(n_seeds):
                rows = _load_csv(arm, ir, eps, s, out_dir)
                if rows is None: continue
                if weight_key not in rows[0]: continue
                try:
                    runs.append([float(r[weight_key]) for r in rows])
                except ValueError:
                    continue
            if not runs: continue
            arr = np.array(runs)
            ep  = np.arange(1, arr.shape[1] + 1)
            mu  = arr.mean(axis=0)
            sd  = arr.std(axis=0)
            ax.plot(ep, mu, label=arm, color=color, lw=1.5)
            ax.fill_between(ep, mu - sd, mu + sd, alpha=0.15, color=color)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Wiener weights — IR={ir:.0f}, ε={eps:.0f}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.axhline(0.0, color="gray", ls="--", lw=0.8)

    fig.tight_layout()
    path = os.path.join(out_dir, f"wiener_weights_ir{ir:.0f}_eps{eps:.0f}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P8] Saved {path}")


def _plot_shrinkage(arm_names, ir, eps, n_seeds, out_dir):
    """Plot effective shrinkage ratio over training."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    wiener_arms = [a for a in arm_names if ARMS[a]["wiener"]]
    if not wiener_arms:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(wiener_arms)))

    for arm, color in zip(wiener_arms, colors):
        runs = []
        for s in range(n_seeds):
            rows = _load_csv(arm, ir, eps, s, out_dir)
            if rows is None: continue
            if "shrink" not in rows[0]: continue
            try:
                runs.append([float(r["shrink"]) for r in rows])
            except ValueError:
                continue
        if not runs: continue
        arr = np.array(runs)
        ep  = np.arange(1, arr.shape[1] + 1)
        mu  = arr.mean(axis=0)
        sd  = arr.std(axis=0)
        ax.plot(ep, mu, label=arm, color=color, lw=1.5)
        ax.fill_between(ep, mu - sd, mu + sd, alpha=0.15, color=color)

    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="no shrinkage")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("‖Ĝ‖ / ‖G̃‖  (shrinkage ratio)")
    ax.set_title(f"Effective gradient shrinkage — IR={ir:.0f}, ε={eps:.0f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, f"shrinkage_ir{ir:.0f}_eps{eps:.0f}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P8] Saved {path}")


def _run_analysis(arm_names, ir_list, eps_list, n_seeds, out_dir):
    _print_table("Final test accuracy (epoch 60)", _final_acc,
                 arm_names, ir_list, eps_list, n_seeds, out_dir)
    _print_table("Best test accuracy (any epoch)", _best_acc,
                 arm_names, ir_list, eps_list, n_seeds, out_dir)
    _print_gap_table(arm_names, ir_list, eps_list, n_seeds, out_dir)

    for ir in ir_list:
        for eps in eps_list:
            _plot_curves(arm_names, ir, eps, n_seeds, out_dir)
            _plot_wiener_weights(arm_names, ir, eps, n_seeds, out_dir)
            _plot_shrinkage(arm_names, ir, eps, n_seeds, out_dir)


# ---------------------------------------------------------------------------
# Baseline transfer utility
# ---------------------------------------------------------------------------

def _transfer_baselines(out_dir, p7_dir=P7_DIR):
    """
    Copy all vanilla and gep P7 CSVs (and checkpoints) into out_dir.
    Safe to run multiple times (skips already-copied files).
    """
    copied = 0
    for arm in ("vanilla", "gep"):
        for ir in IR_LIST:
            for eps in EPS_LIST:
                for seed in range(N_SEEDS):
                    if _try_copy_p7_baseline(arm, ir, eps, seed, out_dir, p7_dir):
                        copied += 1
    print(f"[P8] Baseline transfer complete: {copied} file sets copied.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 8: Wiener-denoised DP-SGD")
    ap.add_argument("--arm",  choices=list(ARMS.keys()),
                    help="Run a single arm")
    ap.add_argument("--ir",   type=float, choices=IR_LIST,
                    help="Imbalance ratio (1 or 50)")
    ap.add_argument("--eps",  type=float, choices=EPS_LIST,
                    help="Privacy budget ε")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed (0, 1, 2)")
    ap.add_argument("--all",  action="store_true",
                    help="Run all arms × IRs × ε × seeds sequentially")
    ap.add_argument("--analysis_only", action="store_true",
                    help="Skip training, print tables and generate figures")
    ap.add_argument("--transfer_baselines", action="store_true",
                    help="Copy vanilla/gep P7 results into results/exp_p8/")
    ap.add_argument("--gpu",      type=int, default=0)
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--out_dir",   default=RESULTS_DIR)
    ap.add_argument("--p7_dir",    default=P7_DIR,
                    help="Directory containing P7 baseline results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p7_dir = args.p7_dir

    device    = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    arm_names = list(ARMS.keys())
    print(f"[P8] Device: {device}")
    print(f"[P8] Looking for P7 baselines in: {p7_dir}")

    if args.transfer_baselines:
        _transfer_baselines(args.out_dir, p7_dir)
        return

    if args.analysis_only:
        _run_analysis(arm_names, IR_LIST, EPS_LIST, N_SEEDS, args.out_dir)
        return

    if args.all:
        _transfer_baselines(args.out_dir, p7_dir)
        for ir in IR_LIST:
            pub_ds, priv_ds, test_ds, class_sizes = _build_datasets(ir, args.data_root)
            print(f"[P8] IR={ir:.0f}  class_sizes={class_sizes}")
            print(f"[P8]   pub={len(pub_ds)}  priv={len(priv_ds)}  test={len(test_ds)}")
            for arm in arm_names:
                for eps in EPS_LIST:
                    for seed in range(N_SEEDS):
                        _train_run(arm, ir, eps, seed,
                                   pub_ds, priv_ds, test_ds, device,
                                   args.out_dir, p7_dir)
        _run_analysis(arm_names, IR_LIST, EPS_LIST, N_SEEDS, args.out_dir)
        return

    # Single-run mode
    if args.arm is None or args.ir is None or args.eps is None:
        _run_analysis(arm_names, IR_LIST, EPS_LIST, N_SEEDS, args.out_dir)
        return

    pub_ds, priv_ds, test_ds, class_sizes = _build_datasets(
        args.ir, args.data_root
    )
    print(f"[P8] IR={args.ir:.0f}  class_sizes={class_sizes}")
    print(f"[P8] pub={len(pub_ds)}  priv={len(priv_ds)}  test={len(test_ds)}")
    _train_run(args.arm, args.ir, args.eps, args.seed,
               pub_ds, priv_ds, test_ds, device, args.out_dir, p7_dir)


if __name__ == "__main__":
    main()
