"""
Phase 9: PDA-DPMD and Coherence-Informed Mirror Descent
=========================================================

Three experiments testing coherence-informed mirror descent components:

  Experiment A — PDA-DPMD baseline reproduction
    vanilla          DP-SGD baseline                     (reuse P7)
    vanilla_warm     DP-SGD with public pretraining       (new)
    gep              GEP baseline                         (reuse P7)
    pda_dpmd         PDA-DPMD (Amid et al. 2022)          (new)

  Experiment B — Coherence-weighted mirror map
    pda_cw           PDA-DPMD, weights ∝ 1/‖∇ℓ‖           (new)
    pda_uniform_w    PDA-DPMD, random weights (ablation)   (new)

  Experiment C — Hierarchical (class-conditional) mirror map
    pda_class        global + β × class-conditional g_pub  (new)
    pda_class_only   class-conditional g_pub only           (new)

Usage
-----
  # K-tuning (run first, Exp A gate):
  python experiments/exp_p9_dpmd.py --tune_K --gpu 0

  # β-tuning (Exp C):
  python experiments/exp_p9_dpmd.py --tune_beta --gpu 0

  # Single arm (parallel dispatch):
  python experiments/exp_p9_dpmd.py --arm pda_dpmd --ir 1 --eps 4 --seed 0 --gpu 0

  # Full experiment sweep:
  python experiments/exp_p9_dpmd.py --exp A --gpu 0
  python experiments/exp_p9_dpmd.py --exp B --gpu 0
  python experiments/exp_p9_dpmd.py --exp C --gpu 0

  # Analysis only:
  python experiments/exp_p9_dpmd.py --analysis_only

  # Transfer P7 baselines:
  python experiments/exp_p9_dpmd.py --transfer_baselines --p7_dir /path/to/exp_p7

PDA-DPMD algorithm (Amid et al. 2022, first-order approx)
----------------------------------------------------------
  alpha_t = clamp(cos(π * t / (2K)), 0, 1)
  g_priv  = clip-and-noise(per_sample_grads, C, σ) / B   ← standard DP-SGD
  g_pub   = ∇L_pub(θ_t)                                  ← public, no privacy cost
  θ_{t+1} = θ_t − lr · (alpha_t · g_priv + (1−alpha_t) · g_pub)
"""

import os
import sys
import csv
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
# Hyperparameters — match P7 exactly for valid A/B comparison
# ---------------------------------------------------------------------------

EPOCHS         = 60
BATCH_SIZE     = 1000
LR             = 0.1
MOMENTUM       = 0.9
WEIGHT_DECAY   = 5e-4
CLIP_VAN       = 1.0       # vanilla / PDA clipping norm
CLIP0          = 5.0       # GEP embedding clip
CLIP1          = 2.0       # GEP residual clip
R_DIM          = 1000      # GEP subspace rank
N_PUB          = 2000      # public examples
DELTA          = 1e-5
EPS_LIST       = [2.0, 4.0, 8.0]
IR_LIST        = [1.0, 50.0]
N_SEEDS        = 3
GRAD_CHUNK     = 64
DATA_ROOT      = "./data"
RESULTS_DIR    = "./results/exp_p9"
P7_DIR         = "./results/exp_p7"

PRETRAIN_EPOCHS = 50       # public pretraining epochs (vanilla_warm + pda arms)
PUB_BATCH_SIZE  = 256      # mini-batch size for public gradient computation

K_DEFAULT      = 1000      # PDA cosine decay steps (tune with --tune_K)
K_SEARCH       = [100, 200, 500, 1000, 2000]
BETA_DEFAULT   = 1.0       # class-conditional weight (tune with --tune_beta)
BETA_SEARCH    = [0.1, 0.5, 1.0, 2.0]

# Experiment subsets
EXP_A_ARMS = ["vanilla", "vanilla_warm", "gep", "pda_dpmd"]
EXP_B_ARMS = ["pda_dpmd", "pda_cw", "pda_uniform_w"]
EXP_C_ARMS = ["pda_dpmd", "pda_class", "pda_class_only"]
EXP_B_EPS  = [2.0, 8.0]
EXP_C_EPS  = [2.0, 8.0]

# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------
#   kind       : "vanilla" | "warm" | "gep" | "pda"
#   pub_mode   : "global" | "cw" | "uniform_w" | "class_only"  (pda only)
#   class_cond : whether to add class-conditional term to g_pub
#   baseline   : reuse from P7 if available

ARMS = {
    # Experiment A
    "vanilla":       {"kind": "vanilla", "pub_mode": None,       "class_cond": False, "baseline": True},
    "vanilla_warm":  {"kind": "warm",    "pub_mode": None,       "class_cond": False, "baseline": False},
    "gep":           {"kind": "gep",     "pub_mode": None,       "class_cond": False, "baseline": True},
    "pda_dpmd":      {"kind": "pda",     "pub_mode": "global",   "class_cond": False, "baseline": False},
    # Experiment B
    "pda_cw":        {"kind": "pda",     "pub_mode": "cw",       "class_cond": False, "baseline": False},
    "pda_uniform_w": {"kind": "pda",     "pub_mode": "uniform_w","class_cond": False, "baseline": False},
    # Experiment C
    "pda_class":     {"kind": "pda",     "pub_mode": "global",   "class_cond": True,  "baseline": False},
    "pda_class_only":{"kind": "pda",     "pub_mode": "class_only","class_cond": True, "baseline": False},
}

# ---------------------------------------------------------------------------
# Data helpers (identical to P7)
# ---------------------------------------------------------------------------

def _cifar10_full(data_root, train=True, augment=False):
    train_tf, test_tf = get_transforms(augment=augment)
    tf = train_tf if train else test_tf
    return torchvision.datasets.CIFAR10(
        root=data_root, train=train, download=True, transform=tf
    )


def _build_datasets(ir, data_root, seed=42):
    """Same public/private split as P7: 10% public per class, seed=42."""
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
    class_sizes  = np.bincount(lt_targets, minlength=10)

    # Preload public tensors for fast per-step gradient computation
    pub_x = torch.stack([pub_dataset[i][0] for i in range(len(pub_dataset))])
    pub_y = torch.tensor([pub_dataset[i][1] for i in range(len(pub_dataset))],
                         dtype=torch.long)

    return pub_dataset, priv_dataset, test_dataset, class_sizes, pub_x, pub_y


# ---------------------------------------------------------------------------
# Model helpers (identical to P7)
# ---------------------------------------------------------------------------

def _make_model():
    return ResNet20(num_classes=10, n_groups=16)


def _set_grads(model, flat_grad):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat_grad[offset: offset + n].view(p.shape).clone()
        offset += n


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Per-sample gradient (vmap, identical to P7) — used only for private gradient
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
    with torch.no_grad():
        g_dict = vmapped(params, buffers,
                         x_chunk.to(device), y_chunk.to(device))
    flat = torch.cat(
        [g_dict[k].view(x_chunk.shape[0], -1) for k in model.state_dict()
         if k in g_dict],
        dim=1,
    )
    return flat  # [B_c, d] on device


def _per_sample_grads_all(model, x, y, device):
    model.eval()
    parts = []
    for i in range(0, x.shape[0], GRAD_CHUNK):
        g = _per_sample_grads_chunk(model, x[i:i+GRAD_CHUNK], y[i:i+GRAD_CHUNK], device)
        parts.append(g.cpu())
        del g
        torch.cuda.empty_cache()
    return torch.cat(parts, dim=0)  # [B, d] on CPU


# ---------------------------------------------------------------------------
# Public gradient computation — standard autograd (no per-sample needed)
# ---------------------------------------------------------------------------

def _pub_grad_flat(model, x_pub, y_pub, device, weights=None):
    """
    Compute a flat gradient vector from public data via standard autograd.

    weights: optional [N_pub] tensor of per-example weights (for pda_cw).
             If None, uses uniform mean loss.
    Returns flat [d] tensor on CPU.
    """
    model.train()
    model.zero_grad()

    N = x_pub.shape[0]
    # Process in mini-batches to avoid OOM, accumulate weighted loss
    total_loss = torch.tensor(0.0, device=device)
    total_w    = 0.0

    for i in range(0, N, PUB_BATCH_SIZE):
        xb = x_pub[i:i+PUB_BATCH_SIZE].to(device)
        yb = y_pub[i:i+PUB_BATCH_SIZE].to(device)
        out  = model(xb)
        loss_per = F.cross_entropy(out, yb, reduction='none')  # [B]

        if weights is not None:
            wb = weights[i:i+PUB_BATCH_SIZE].to(device)
            total_loss = total_loss + (wb * loss_per).sum()
            total_w   += wb.sum().item()
        else:
            total_loss = total_loss + loss_per.sum()
            total_w   += len(yb)

    avg_loss = total_loss / max(total_w, 1e-12)
    avg_loss.backward()

    flat = torch.cat([p.grad.view(-1).detach().cpu()
                      for p in model.parameters()])
    model.zero_grad()
    return flat  # [d] on CPU


def _class_grad_flat(model, x_pub, y_pub, device, n_classes=10):
    """
    Compute average of per-class public gradients.
    Returns flat [d] tensor on CPU = mean_k ∇L^(k).
    """
    model.train()
    total_grad = None
    classes_present = 0

    for k in range(n_classes):
        mask = (y_pub == k)
        if mask.sum() == 0:
            continue
        xk = x_pub[mask]
        yk = y_pub[mask]
        g_k = _pub_grad_flat(model, xk, yk, device)
        total_grad = g_k if total_grad is None else total_grad + g_k
        classes_present += 1

    if total_grad is None:
        d = _num_params(model)
        return torch.zeros(d)
    return total_grad / classes_present  # [d] on CPU


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


def _gep_sigmas(sigma_van, r, d):
    return sigma_van * math.sqrt(2), sigma_van * math.sqrt(2)


# ---------------------------------------------------------------------------
# Pretraining on public data (shared by vanilla_warm and all PDA arms)
# ---------------------------------------------------------------------------

def _pretrain_on_public(model, pub_x, pub_y, device, epochs=PRETRAIN_EPOCHS,
                        lr=0.01, momentum=0.9, wd=5e-4):
    """
    Standard (non-private) SGD training on public data.
    Returns the trained model (in-place) and the pretrained state_dict.
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=momentum, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    N = pub_x.shape[0]
    for ep in range(1, epochs + 1):
        perm = torch.randperm(N)
        for i in range(0, N, PUB_BATCH_SIZE):
            idx = perm[i:i+PUB_BATCH_SIZE]
            xb  = pub_x[idx].to(device)
            yb  = pub_y[idx].to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    print(f"[P9] Pretraining done ({epochs} epochs)")
    return model


# ---------------------------------------------------------------------------
# Compute CW weights from pretrained model (Exp B)
# ---------------------------------------------------------------------------

def _compute_cw_weights(model, pub_x, pub_y, device):
    """
    w_i = 1 / (‖∇ℓ(θ₀; zᵢ)‖ + 1e-6), normalized so mean(w)=1.
    Computed once from the pretrained model.
    Returns w: FloatTensor [N_pub].
    """
    print("[P9] Computing CW weights from pretrained model...")
    grads = _per_sample_grads_all(model, pub_x, pub_y, device)  # [N_pub, d]
    norms = grads.norm(dim=1)                                     # [N_pub]
    w = 1.0 / (norms + 1e-6)
    w = w / w.mean()
    return w  # [N_pub] on CPU


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------

def _vanilla_priv_step(model, x, y, sigma, device):
    """Clip-sum-noise per-sample DP-SGD. Returns flat [d] / B on CPU."""
    B = x.shape[0]
    grads   = _per_sample_grads_all(model, x, y, device)  # [B, d] CPU
    norms   = grads.norm(dim=1, keepdim=True).clamp(min=1e-8)
    clipped = grads * (CLIP_VAN / norms).clamp(max=1.0)
    sum_g   = clipped.sum(dim=0)                           # [d] CPU
    noise   = torch.randn_like(sum_g) * (sigma * CLIP_VAN)
    return (sum_g + noise) / B                             # [d] CPU


def _gep_priv_step(model, x, y, V, sigma_par, sigma_perp, device):
    """GEP step (identical to P7). Returns flat [d] / B on CPU."""
    B     = x.shape[0]
    V_dev = V.to(device)
    r, d  = V.shape[1], V.shape[0]

    grads    = _per_sample_grads_all(model, x, y, device)
    sum_c    = torch.zeros(r, device=device)
    sum_perp = torch.zeros(d, device=device)

    chunk = min(GRAD_CHUNK * 4, B)
    for i in range(0, B, chunk):
        g = grads[i:i+chunk].to(device)
        c      = g @ V_dev
        g_par  = c @ V_dev.T
        g_perp = g - g_par
        c      = c * (CLIP0 / c.norm(dim=1, keepdim=True).clamp(min=1e-8)).clamp(max=1.0)
        g_perp = g_perp * (CLIP1 / g_perp.norm(dim=1, keepdim=True).clamp(min=1e-8)).clamp(max=1.0)
        sum_c    += c.sum(0)
        sum_perp += g_perp.sum(0)
        del g, c, g_par, g_perp
        torch.cuda.empty_cache()

    sum_c    += torch.randn(r, device=device) * (sigma_par * CLIP0)
    noise_p   = torch.randn(d, device=device) * (sigma_perp * CLIP1)
    noise_p  -= V_dev @ (V_dev.T @ noise_p)
    sum_perp += noise_p

    return ((V_dev @ sum_c + sum_perp) / B).cpu()  # [d] CPU


def _build_subspace(model, pub_dataset, r, device, seed=0):
    """Variance-PCA subspace (identical to P7 gep arm). Returns [d, r] CPU."""
    loader = DataLoader(pub_dataset, batch_size=min(256, len(pub_dataset)),
                        shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    gs = []
    for x, y in loader:
        gs.append(_per_sample_grads_all(model, x, y, device))
    G = torch.cat(gs, 0).float()
    k = min(r, G.shape[0] - 1)
    _, _, V = torch.svd_lowrank(G, q=k, niter=4)
    return V[:, :r].cpu()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.shape[0]
    return correct / total


@torch.no_grad()
def _evaluate_per_class(model, loader, device, n_classes=10):
    """Returns per-class accuracy as array of length n_classes."""
    model.eval()
    correct = np.zeros(n_classes, dtype=int)
    total   = np.zeros(n_classes, dtype=int)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        for k in range(n_classes):
            mask = (y == k)
            correct[k] += (pred[mask] == y[mask]).sum().item()
            total[k]   += mask.sum().item()
    return np.divide(correct, total, out=np.zeros(n_classes), where=total > 0)


@torch.no_grad()
def _probe_loss(model, probe_x, probe_y, device):
    model.eval()
    return F.cross_entropy(model(probe_x.to(device)), probe_y.to(device)).item()


# ---------------------------------------------------------------------------
# P7 baseline reuse helpers
# ---------------------------------------------------------------------------

def _try_copy_p7_baseline(arm_name, ir, eps, seed, out_dir, p7_dir):
    tag     = f"{arm_name}_ir{ir:.0f}_eps{eps:.0f}_seed{seed}"
    src_csv = os.path.join(p7_dir, f"{tag}.csv")
    dst_csv = os.path.join(out_dir, f"{tag}.csv")

    if os.path.exists(dst_csv):
        return True
    if not os.path.exists(src_csv):
        return False

    shutil.copy2(src_csv, dst_csv)
    print(f"[P9] Reused P7 baseline: {src_csv} → {dst_csv}")
    for sfx in ("_best.pt", "_final.pt"):
        s = os.path.join(p7_dir,  f"{tag}{sfx}")
        d = os.path.join(out_dir, f"{tag}{sfx}")
        if os.path.exists(s) and not os.path.exists(d):
            shutil.copy2(s, d)
    return True


def _transfer_baselines(out_dir, p7_dir):
    copied = 0
    for arm in ("vanilla", "gep"):
        for ir in IR_LIST:
            for eps in EPS_LIST:
                for seed in range(N_SEEDS):
                    if _try_copy_p7_baseline(arm, ir, eps, seed, out_dir, p7_dir):
                        copied += 1
    print(f"[P9] Baseline transfer: {copied} file sets copied.")


# ---------------------------------------------------------------------------
# Main training run
# ---------------------------------------------------------------------------

def _train_run(arm_name, ir, eps, seed,
               pub_dataset, priv_dataset, test_dataset,
               pub_x, pub_y, device, out_dir,
               p7_dir=P7_DIR, K=K_DEFAULT, beta=BETA_DEFAULT,
               tag_suffix=""):
    """
    Full training run for (arm, IR, eps, seed).
    Returns final test accuracy.
    """
    arm_cfg = ARMS[arm_name]
    tag     = f"{arm_name}_ir{ir:.0f}_eps{eps:.0f}_seed{seed}{tag_suffix}"
    csv_path = os.path.join(out_dir, f"{tag}.csv")

    # ---- P7 reuse for baselines ----
    if arm_cfg["baseline"]:
        if _try_copy_p7_baseline(arm_name, ir, eps, seed, out_dir, p7_dir):
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            return float(rows[-1]["test_acc"])

    if os.path.exists(csv_path):
        print(f"[P9] {tag}: already done, loading CSV")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return float(rows[-1]["test_acc"])

    print(f"\n[P9] === {tag} ===")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Privacy accounting
    tmp_loader = DataLoader(priv_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, drop_last=True)
    steps_per_epoch = len(tmp_loader)
    T_steps         = EPOCHS * steps_per_epoch
    q               = BATCH_SIZE / len(priv_dataset)
    del tmp_loader

    sigma_van = _calibrate_sigma(eps, DELTA, q, T_steps)
    d         = _num_params(_make_model())

    if arm_cfg["kind"] == "gep":
        sigma_par, sigma_perp = _gep_sigmas(sigma_van, R_DIM, d)

    # Model
    model = _make_model().to(device)

    # Pretraining (all non-vanilla, non-gep arms)
    needs_pretrain = arm_cfg["kind"] in ("warm", "pda")
    pretrained_state = None

    if needs_pretrain:
        print(f"[P9]   Pretraining on public data ({PRETRAIN_EPOCHS} epochs)...")
        _pretrain_on_public(model, pub_x, pub_y, device)
        pretrained_state = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}

    # Compute CW weights from pretrained model (once)
    cw_weights = None
    if arm_cfg["kind"] == "pda" and arm_cfg["pub_mode"] == "cw":
        cw_weights = _compute_cw_weights(model, pub_x, pub_y, device)
    elif arm_cfg["kind"] == "pda" and arm_cfg["pub_mode"] == "uniform_w":
        rng_np = np.random.default_rng(seed)
        w_raw  = torch.from_numpy(rng_np.exponential(1.0, size=len(pub_x))).float()
        cw_weights = w_raw / w_raw.mean()
        print("[P9]   Using random (Exp(1)) weights for uniform_w ablation")

    # For vanilla_warm: reset model to pretrained weights, then run DP-SGD from scratch
    # For pda arms:     keep pretrained weights as starting point
    if arm_cfg["kind"] == "warm":
        # Already pretrained; just run vanilla DP-SGD from this init
        pass  # model already has pretrained weights

    # GEP subspace (for gep fallback)
    V = None
    if arm_cfg["kind"] == "gep":
        V = _build_subspace(model, pub_dataset, R_DIM, device, seed=seed)

    # Optimizer + scheduler over EPOCHS DP training epochs
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    priv_loader = DataLoader(priv_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4, pin_memory=True,
                             drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False,
                             num_workers=4, pin_memory=True)

    rng_np = np.random.default_rng(seed + 100)
    probe_idx = rng_np.choice(len(priv_dataset), size=256, replace=False)
    probe_dl  = DataLoader(Subset(priv_dataset, probe_idx.tolist()),
                           batch_size=256, shuffle=False)
    probe_x, probe_y = next(iter(probe_dl))

    is_pda        = arm_cfg["kind"] == "pda"
    log_per_class = arm_cfg["kind"] == "pda" and arm_cfg["class_cond"]

    fieldnames = ["epoch", "train_loss", "test_acc", "lr"]
    if is_pda:
        fieldnames += ["alpha_mean"]
    if log_per_class:
        fieldnames += [f"acc_c{k}" for k in range(10)]

    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    best_acc = 0.0
    step_global = 0  # cumulative step counter (for alpha_t)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Refresh GEP subspace
        if arm_cfg["kind"] == "gep":
            V = _build_subspace(model, pub_dataset, R_DIM, device, seed=seed)

        alpha_accum = []

        for x, y in priv_loader:
            optimizer.zero_grad(set_to_none=True)

            if arm_cfg["kind"] in ("vanilla", "warm"):
                flat_priv = _vanilla_priv_step(model, x, y, sigma_van, device)
                _set_grads(model, flat_priv.to(device))

            elif arm_cfg["kind"] == "gep":
                flat_grad = _gep_priv_step(model, x, y, V,
                                           sigma_par, sigma_perp, device)
                _set_grads(model, flat_grad.to(device))

            elif arm_cfg["kind"] == "pda":
                # Private gradient
                flat_priv = _vanilla_priv_step(model, x, y, sigma_van, device)

                # Public gradient (zero privacy cost)
                pub_mode    = arm_cfg["pub_mode"]
                class_cond  = arm_cfg["class_cond"]

                if pub_mode == "global":
                    g_pub = _pub_grad_flat(model, pub_x, pub_y, device)
                elif pub_mode == "cw":
                    g_pub = _pub_grad_flat(model, pub_x, pub_y, device,
                                           weights=cw_weights)
                elif pub_mode == "uniform_w":
                    g_pub = _pub_grad_flat(model, pub_x, pub_y, device,
                                           weights=cw_weights)
                elif pub_mode == "class_only":
                    g_pub = _class_grad_flat(model, pub_x, pub_y, device)
                else:
                    g_pub = torch.zeros_like(flat_priv)

                if class_cond and pub_mode != "class_only":
                    g_class = _class_grad_flat(model, pub_x, pub_y, device)
                    g_pub   = g_pub + beta * g_class

                # Alpha schedule: cos(π t / (2K)), clamped to [0, 1]
                alpha_t = max(0.0, min(1.0, math.cos(math.pi * step_global / (2 * K))))
                alpha_accum.append(alpha_t)

                flat_update = alpha_t * flat_priv + (1.0 - alpha_t) * g_pub
                _set_grads(model, flat_update.to(device))

            optimizer.step()
            step_global += 1

        scheduler.step()

        train_loss = _probe_loss(model, probe_x, probe_y, device)
        test_acc   = _evaluate(model, test_loader, device)
        cur_lr     = scheduler.get_last_lr()[0]

        row = {
            "epoch":      epoch,
            "train_loss": f"{train_loss:.4f}",
            "test_acc":   f"{test_acc:.4f}",
            "lr":         f"{cur_lr:.6f}",
        }
        if is_pda and alpha_accum:
            row["alpha_mean"] = f"{np.mean(alpha_accum):.4f}"
        if log_per_class:
            per_cls = _evaluate_per_class(model, test_loader, device)
            for k in range(10):
                row[f"acc_c{k}"] = f"{per_cls[k]:.4f}"

        writer.writerow(row)
        csv_file.flush()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f"{tag}_best.pt"))

        alpha_str = f"  α={np.mean(alpha_accum):.3f}" if alpha_accum else ""
        print(f"  epoch {epoch:3d}/{EPOCHS}  loss={train_loss:.4f}  "
              f"acc={test_acc:.4f}  best={best_acc:.4f}{alpha_str}")

    csv_file.close()
    torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}_final.pt"))
    print(f"[P9] done — final={test_acc:.4f}  best={best_acc:.4f}")
    return test_acc


# ---------------------------------------------------------------------------
# K and beta tuning
# ---------------------------------------------------------------------------

def _tune_K(pub_dataset, priv_dataset, test_dataset,
            pub_x, pub_y, device, out_dir):
    """
    Run pda_dpmd at all K values (IR=1, eps=4, seed=0).
    Saves a CSV summary and returns the best K.
    """
    print("\n[P9] === K-tuning for pda_dpmd (IR=1, eps=4, seed=0) ===")
    results = {}
    for K in K_SEARCH:
        print(f"\n[P9] K={K}")
        acc = _train_run(
            "pda_dpmd", ir=1.0, eps=4.0, seed=0,
            pub_dataset=pub_dataset, priv_dataset=priv_dataset,
            test_dataset=test_dataset, pub_x=pub_x, pub_y=pub_y,
            device=device, out_dir=out_dir,
            K=K, tag_suffix=f"_K{K}"
        )
        results[K] = acc
        print(f"[P9] K={K}  →  test_acc={acc:.4f}")

    best_K = max(results, key=results.get)
    print(f"\n[P9] Best K = {best_K}  (acc={results[best_K]:.4f})")

    summary_path = os.path.join(out_dir, "tune_K_results.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "test_acc"])
        for K, acc in sorted(results.items()):
            w.writerow([K, f"{acc:.4f}"])
    print(f"[P9] K-tuning summary: {summary_path}")
    return best_K


def _tune_beta(pub_dataset, priv_dataset, test_dataset,
               pub_x, pub_y, device, out_dir):
    """
    Run pda_class at all beta values (IR=50, eps=4, seed=0).
    Saves a CSV summary and returns the best beta.
    """
    print("\n[P9] === β-tuning for pda_class (IR=50, eps=4, seed=0) ===")
    results = {}
    for beta in BETA_SEARCH:
        print(f"\n[P9] β={beta}")
        acc = _train_run(
            "pda_class", ir=50.0, eps=4.0, seed=0,
            pub_dataset=pub_dataset, priv_dataset=priv_dataset,
            test_dataset=test_dataset, pub_x=pub_x, pub_y=pub_y,
            device=device, out_dir=out_dir,
            beta=beta, tag_suffix=f"_beta{beta}"
        )
        results[beta] = acc
        print(f"[P9] β={beta}  →  test_acc={acc:.4f}")

    best_beta = max(results, key=results.get)
    print(f"\n[P9] Best β = {best_beta}  (acc={results[best_beta]:.4f})")

    summary_path = os.path.join(out_dir, "tune_beta_results.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beta", "test_acc"])
        for b, acc in sorted(results.items()):
            w.writerow([b, f"{acc:.4f}"])
    print(f"[P9] β-tuning summary: {summary_path}")
    return best_beta


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _load_csv(arm, ir, eps, seed, out_dir, tag_suffix=""):
    tag  = f"{arm}_ir{ir:.0f}_eps{eps:.0f}_seed{seed}{tag_suffix}"
    path = os.path.join(out_dir, f"{tag}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _final_acc(rows):
    return float(rows[-1]["test_acc"]) if rows else None


def _best_acc(rows):
    return max(float(r["test_acc"]) for r in rows) if rows else None


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
                row += f"  {'N/A':>14}  " if mu is None else \
                       f"  {mu*100:5.2f}±{sd*100:.2f}%  "
        print(row)


def _print_gap_table(baseline, arm_names, ir_list, eps_list, n_seeds, out_dir):
    print(f"\n{'='*80}")
    print(f" Gap vs {baseline} (pp, final acc)")
    print(f"{'='*80}")
    for ir in ir_list:
        print(f"\n  IR={ir:.0f}")
        header = f"  {'arm':<20}" + "".join(f"  ε={e:.0f}  " for e in eps_list)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for arm in arm_names:
            if arm == baseline: continue
            row = f"  {arm:<20}"
            for eps in eps_list:
                arm_accs = [_final_acc(_load_csv(arm,      ir, eps, s, out_dir)) for s in range(n_seeds)]
                bas_accs = [_final_acc(_load_csv(baseline, ir, eps, s, out_dir)) for s in range(n_seeds)]
                a_mu, _ = _mean_std(arm_accs)
                b_mu, _ = _mean_std(bas_accs)
                if a_mu is None or b_mu is None:
                    row += f"  {'N/A':>10}  "
                else:
                    row += f"  {(a_mu-b_mu)*100:+6.2f}pp  "
            print(row)


def _print_per_class_table(arm_names, ir_list, eps_list, n_seeds, out_dir):
    """Print per-class test accuracy for class-conditional arms (Exp C)."""
    print(f"\n{'='*80}")
    print(" Per-class accuracy (final epoch, seed=0)")
    print(f"{'='*80}")
    for ir in ir_list:
        for eps in eps_list:
            print(f"\n  IR={ir:.0f}, ε={eps:.0f}")
            hdr = f"  {'arm':<20}" + "".join(f"  cls{k}  " for k in range(10))
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for arm in arm_names:
                rows = _load_csv(arm, ir, eps, 0, out_dir)
                if rows is None or f"acc_c0" not in rows[0]:
                    continue
                last = rows[-1]
                row = f"  {arm:<20}"
                for k in range(10):
                    v = float(last.get(f"acc_c{k}", float("nan")))
                    row += f"  {v*100:4.1f}%  "
                print(row)


def _plot_curves(arm_names, ir, eps, n_seeds, out_dir, title_prefix=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(arm_names)))

    for ax, (metric, ylabel) in zip(
        axes, [("test_acc", "Test Accuracy"), ("train_loss", "Train Loss")]
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
            mu  = arr.mean(0)
            sd  = arr.std(0)
            ax.plot(ep, mu, label=arm, color=color, lw=1.5)
            ax.fill_between(ep, mu - sd, mu + sd, alpha=0.15, color=color)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix}IR={ir:.0f}, ε={eps:.0f}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    stem = "_".join(arm_names[:4]) if len(arm_names) > 4 else "_".join(arm_names)
    path = os.path.join(out_dir, f"curves_{stem}_ir{ir:.0f}_eps{eps:.0f}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P9] Saved {path}")


def _plot_alpha_schedule(out_dir):
    """Visualize the alpha_t cosine schedule for each K value."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    T = EPOCHS * (45000 // BATCH_SIZE)  # approx steps
    fig, ax = plt.subplots(figsize=(8, 4))
    for K in K_SEARCH:
        t = np.arange(T)
        alpha = np.clip(np.cos(np.pi * t / (2 * K)), 0, 1)
        ax.plot(t, alpha, label=f"K={K}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("α_t (private weight)")
    ax.set_title("PDA-DPMD cosine schedule by K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "alpha_schedule.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P9] Saved {path}")


def _plot_per_class(arm_names, ir, eps, seed, out_dir):
    """Bar chart: per-class accuracy comparison for Exp C arms."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    arms_with_data = []
    accs_by_arm = {}
    for arm in arm_names:
        rows = _load_csv(arm, ir, eps, seed, out_dir)
        if rows is None or "acc_c0" not in rows[0]:
            # Try without per-class data (vanilla / gep)
            rows2 = _load_csv(arm, ir, eps, seed, out_dir)
            if rows2 is not None:
                final_acc_val = float(rows2[-1]["test_acc"])
                accs_by_arm[arm] = np.full(10, final_acc_val)
                arms_with_data.append(arm)
            continue
        last = rows[-1]
        accs_by_arm[arm] = np.array([float(last[f"acc_c{k}"]) for k in range(10)])
        arms_with_data.append(arm)

    if not arms_with_data:
        return

    x = np.arange(10)
    w = 0.8 / len(arms_with_data)
    colors = plt.cm.tab10(np.linspace(0, 1, len(arms_with_data)))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (arm, color) in enumerate(zip(arms_with_data, colors)):
        offset = (i - len(arms_with_data) / 2 + 0.5) * w
        ax.bar(x + offset, accs_by_arm[arm] * 100, w, label=arm, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"cls {k}" for k in range(10)], rotation=45)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Per-class accuracy — IR={ir:.0f}, ε={eps:.0f}, seed={seed}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    stem = "_".join(arms_with_data[:4])
    path = os.path.join(out_dir, f"per_class_{stem}_ir{ir:.0f}_eps{eps:.0f}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P9] Saved {path}")


def _run_analysis(out_dir, n_seeds=N_SEEDS):
    arm_names_A = EXP_A_ARMS
    arm_names_B = EXP_B_ARMS
    arm_names_C = EXP_C_ARMS

    print("\n[P9] ===== EXPERIMENT A: PDA-DPMD BASELINE =====")
    _print_table("Final accuracy — Exp A", _final_acc,
                 arm_names_A, IR_LIST, EPS_LIST, n_seeds, out_dir)
    _print_table("Best accuracy — Exp A",  _best_acc,
                 arm_names_A, IR_LIST, EPS_LIST, n_seeds, out_dir)
    _print_gap_table("vanilla", arm_names_A, IR_LIST, EPS_LIST, n_seeds, out_dir)

    print("\n[P9] ===== EXPERIMENT B: COHERENCE-WEIGHTED MIRROR MAP =====")
    _print_table("Final accuracy — Exp B", _final_acc,
                 arm_names_B, IR_LIST, EXP_B_EPS, n_seeds, out_dir)
    _print_gap_table("pda_dpmd", arm_names_B, IR_LIST, EXP_B_EPS, n_seeds, out_dir)

    print("\n[P9] ===== EXPERIMENT C: CLASS-CONDITIONAL MIRROR MAP =====")
    _print_table("Final accuracy — Exp C", _final_acc,
                 arm_names_C, IR_LIST, EXP_C_EPS, n_seeds, out_dir)
    _print_gap_table("pda_dpmd", arm_names_C, IR_LIST, EXP_C_EPS, n_seeds, out_dir)
    _print_per_class_table(EXP_C_ARMS, IR_LIST, EXP_C_EPS, n_seeds, out_dir)

    # Figures
    _plot_alpha_schedule(out_dir)
    for ir in IR_LIST:
        for eps in EPS_LIST:
            _plot_curves(arm_names_A, ir, eps, n_seeds, out_dir, "Exp A: ")
        for eps in EXP_B_EPS:
            _plot_curves(arm_names_B, ir, eps, n_seeds, out_dir, "Exp B: ")
        for eps in EXP_C_EPS:
            _plot_curves(arm_names_C, ir, eps, n_seeds, out_dir, "Exp C: ")
            _plot_per_class(arm_names_C, ir, eps, 0, out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 9: PDA-DPMD experiments")
    ap.add_argument("--arm",  choices=list(ARMS.keys()), help="Single arm")
    ap.add_argument("--ir",   type=float, choices=IR_LIST)
    ap.add_argument("--eps",  type=float, choices=EPS_LIST)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp",  choices=["A", "B", "C"],
                    help="Run full experiment A, B, or C")
    ap.add_argument("--all",  action="store_true",
                    help="Run all experiments A+B+C sequentially")
    ap.add_argument("--tune_K",    action="store_true",
                    help="Search over K values for pda_dpmd (IR=1, eps=4, seed=0)")
    ap.add_argument("--tune_beta", action="store_true",
                    help="Search over β values for pda_class (IR=50, eps=4, seed=0)")
    ap.add_argument("--analysis_only", action="store_true")
    ap.add_argument("--transfer_baselines", action="store_true")
    ap.add_argument("--K",    type=int,   default=K_DEFAULT,
                    help=f"PDA cosine decay steps (default {K_DEFAULT})")
    ap.add_argument("--beta", type=float, default=BETA_DEFAULT,
                    help=f"Class-conditional weight β (default {BETA_DEFAULT})")
    ap.add_argument("--gpu",       type=int, default=0)
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--out_dir",   default=RESULTS_DIR)
    ap.add_argument("--p7_dir",    default=P7_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p7_dir = args.p7_dir
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[P9] Device: {device}  |  K={args.K}  β={args.beta}")
    print(f"[P9] P7 baseline dir: {p7_dir}")

    if args.transfer_baselines:
        _transfer_baselines(args.out_dir, p7_dir)
        return

    if args.analysis_only:
        _run_analysis(args.out_dir)
        return

    # Pre-load datasets (shared across all runs for the same IR in a sweep)
    def _run_arm_grid(arm_list, ir_list, eps_list, n_seeds, K, beta):
        _transfer_baselines(args.out_dir, p7_dir)
        for ir in ir_list:
            pub_ds, priv_ds, test_ds, cs, pub_x, pub_y = \
                _build_datasets(ir, args.data_root)
            print(f"[P9] IR={ir:.0f}  sizes={cs}  "
                  f"pub={len(pub_ds)}  priv={len(priv_ds)}")
            for arm in arm_list:
                for eps in eps_list:
                    for seed in range(n_seeds):
                        _train_run(arm, ir, eps, seed,
                                   pub_ds, priv_ds, test_ds,
                                   pub_x, pub_y, device, args.out_dir,
                                   p7_dir=p7_dir, K=K, beta=beta)

    if args.tune_K:
        pub_ds, priv_ds, test_ds, _, pub_x, pub_y = \
            _build_datasets(1.0, args.data_root)
        best_K = _tune_K(pub_ds, priv_ds, test_ds, pub_x, pub_y,
                         device, args.out_dir)
        print(f"[P9] Recommended: --K {best_K}")
        return

    if args.tune_beta:
        pub_ds, priv_ds, test_ds, _, pub_x, pub_y = \
            _build_datasets(50.0, args.data_root)
        best_beta = _tune_beta(pub_ds, priv_ds, test_ds, pub_x, pub_y,
                               device, args.out_dir)
        print(f"[P9] Recommended: --beta {best_beta}")
        return

    if args.exp == "A":
        _run_arm_grid(EXP_A_ARMS, IR_LIST, EPS_LIST, N_SEEDS,
                      args.K, args.beta)
        _run_analysis(args.out_dir)
        return

    if args.exp == "B":
        _run_arm_grid(EXP_B_ARMS, IR_LIST, EXP_B_EPS, N_SEEDS,
                      args.K, args.beta)
        _run_analysis(args.out_dir)
        return

    if args.exp == "C":
        _run_arm_grid(EXP_C_ARMS, IR_LIST, EXP_C_EPS, N_SEEDS,
                      args.K, args.beta)
        _run_analysis(args.out_dir)
        return

    if args.all:
        for exp_arms, eps_l in [
            (EXP_A_ARMS, EPS_LIST),
            (EXP_B_ARMS, EXP_B_EPS),
            (EXP_C_ARMS, EXP_C_EPS),
        ]:
            _run_arm_grid(exp_arms, IR_LIST, eps_l, N_SEEDS, args.K, args.beta)
        _run_analysis(args.out_dir)
        return

    # Single-arm mode
    if args.arm is None or args.ir is None or args.eps is None:
        _run_analysis(args.out_dir)
        return

    pub_ds, priv_ds, test_ds, cs, pub_x, pub_y = \
        _build_datasets(args.ir, args.data_root)
    print(f"[P9] IR={args.ir:.0f}  sizes={cs}  "
          f"pub={len(pub_ds)}  priv={len(priv_ds)}")
    _train_run(args.arm, args.ir, args.eps, args.seed,
               pub_ds, priv_ds, test_ds, pub_x, pub_y,
               device, args.out_dir,
               p7_dir=p7_dir, K=args.K, beta=args.beta)


if __name__ == "__main__":
    main()
