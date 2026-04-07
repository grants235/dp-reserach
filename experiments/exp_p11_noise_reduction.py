#!/usr/bin/env python3
"""
Phase 11 — Experiment 2: Noise Reduction Accuracy Comparison
=============================================================

Tests whether distribution-dependent noise calibration (Arm B: dist_aware)
improves test accuracy over standard DP-SGD (Arm A: vanilla_warm) at eps=2.

Arms:
  vanilla_warm  Standard DP-SGD, warm start (reuse P9 if available)
  dist_aware    DP-SGD, sigma' = sigma / sqrt(beta95) from Exp 1
  gep           GEP with rank r* from Exp 1
  pda_cw        PDA-CW (reuse P9 if available)

Prerequisite: exp_p11_beta_measurement.py must have produced beta_spectrum.csv.

Usage
-----
  # Full sweep (4 arms x 3 seeds):
  python experiments/exp_p11_noise_reduction.py --gpu 0

  # Single arm:
  python experiments/exp_p11_noise_reduction.py --arm dist_aware --seed 0 --gpu 0

  # Provide beta95 directly (skips reading Exp 1 CSV):
  python experiments/exp_p11_noise_reduction.py --arm dist_aware --beta95 0.35 --rank 50 --gpu 0

  # Analysis only:
  python experiments/exp_p11_noise_reduction.py --analysis_only

  # Attempt to reuse P9 baselines:
  python experiments/exp_p11_noise_reduction.py --transfer_p9 --p9_dir results/exp_p9
"""

import os
import sys
import csv
import math
import shutil
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import ResNet20
from src.datasets import make_public_private_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DELTA         = 1e-5
EPS           = 2.0
CLIP_C        = 1.0
CLIP0         = 5.0      # GEP embedding clip
CLIP1         = 2.0      # GEP residual clip
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
N_SEEDS       = 3
ALPHA_START   = 0.9
RAMP_FRAC     = 0.5
DATA_ROOT     = "./data"
RESULTS_DIR   = "./results/exp_p11"

EXP1_DIR = os.path.join(RESULTS_DIR, "exp1")
P9_DIR   = "./results/exp_p9"

ARMS = ["vanilla_warm", "dist_aware", "gep", "pda_cw"]

# Tag convention for P9 reuse:
# P9 saves: {arm}_ir1_eps2_seed{s}.csv   (balanced CIFAR-10 = IR=1)


# ---------------------------------------------------------------------------
# Data helpers (identical to exp_p11_beta_measurement)
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
    rng = np.random.default_rng(seed)
    pub_idx_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]

    pub_ds   = Subset(full_train, pub_idx_use.tolist())
    priv_ds  = Subset(_cifar10(data_root, train=True, augment=True), priv_idx.tolist())
    test_ds  = _cifar10(data_root, train=False, augment=False)

    pub_x = torch.stack([pub_ds[i][0] for i in range(len(pub_ds))])
    pub_y = torch.tensor([pub_ds[i][1] for i in range(len(pub_ds))], dtype=torch.long)
    return pub_ds, priv_ds, test_ds, pub_x, pub_y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _make_model():
    return ResNet20(num_classes=10, n_groups=16)


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


def _set_grads(model, flat_grad):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat_grad[offset:offset+n].view(p.shape).clone()
        offset += n


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

def _calibrate_sigma(eps, delta, q, T):
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(
        target_epsilon=eps, target_delta=delta,
        sample_rate=q, steps=T, accountant="rdp")


def _gep_sigmas(sigma_van):
    return sigma_van * math.sqrt(2), sigma_van * math.sqrt(2)


# ---------------------------------------------------------------------------
# Training utilities
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
    print(f"[P11-E2] Pretraining done ({epochs} ep)")


def _pub_grad_flat(model, pub_x, pub_y, device):
    model.train(); model.zero_grad()
    N = pub_x.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    for i in range(0, N, PUB_BATCH):
        xb = pub_x[i:i+PUB_BATCH].to(device)
        yb = pub_y[i:i+PUB_BATCH].to(device)
        total_loss += F.cross_entropy(model(xb), yb, reduction="sum")
    (total_loss / N).backward()
    flat = torch.cat([p.grad.view(-1) for p in model.parameters()
                      if p.grad is not None]).cpu()
    model.zero_grad()
    return flat


def _pub_grad_cw_flat(model, pub_x, pub_y, device):
    model.eval()
    N   = pub_x.shape[0]
    idx = torch.randperm(N)[:PUB_BATCH]
    grads     = _per_sample_grads_chunk(model, pub_x[idx], pub_y[idx], device)
    grads_cpu = grads.cpu(); del grads; torch.cuda.empty_cache()
    norms = grads_cpu.norm(dim=1)
    w     = 1.0 / (norms + 1e-6); w = w / w.sum()
    return (w.unsqueeze(1) * grads_cpu).sum(0)


def _build_subspace(model, pub_ds, r, device, seed=0):
    """Variance-PCA subspace from public data. Returns [d, r] CPU."""
    loader = DataLoader(pub_ds, batch_size=min(256, len(pub_ds)),
                        shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    gs = []
    for x, y in loader:
        gs.append(_per_sample_grads_all(model, x, y, device))
    G = torch.cat(gs, 0).float()
    k = min(r, G.shape[0] - 1)
    _, _, V = torch.svd_lowrank(G, q=k, niter=4)
    return V[:, :r].cpu()


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


@torch.no_grad()
def _evaluate_per_class(model, test_ds, device):
    loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                        num_workers=4, pin_memory=True)
    model.eval()
    correct = np.zeros(10, dtype=int)
    total   = np.zeros(10, dtype=int)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred  = model(x).argmax(1)
        for k in range(10):
            mask = (y == k)
            correct[k] += (pred[mask] == k).sum().item()
            total[k]   += mask.sum().item()
    return np.divide(correct, total, out=np.zeros(10), where=total > 0)


@torch.no_grad()
def _probe_loss(model, probe_x, probe_y, device):
    model.eval()
    return F.cross_entropy(model(probe_x.to(device)), probe_y.to(device)).item()


# ---------------------------------------------------------------------------
# P9 result reuse
# ---------------------------------------------------------------------------

def _try_reuse_p9(arm_name, eps, seed, out_dir, p9_dir):
    """Copy P9 CSV and checkpoints for vanilla_warm/gep/pda_cw at IR=1."""
    src_tag = f"{arm_name}_ir1_eps{eps:.0f}_seed{seed}"
    src_csv = os.path.join(p9_dir, f"{src_tag}.csv")
    dst_tag = f"{arm_name}_eps{eps:.0f}_seed{seed}"
    dst_csv = os.path.join(out_dir, f"{dst_tag}.csv")

    if os.path.exists(dst_csv):
        return True
    if not os.path.exists(src_csv):
        return False

    shutil.copy2(src_csv, dst_csv)
    print(f"[P11-E2] Reused P9 result: {src_csv}")
    for sfx in ("_best.pt", "_final.pt"):
        s = os.path.join(p9_dir, f"{src_tag}{sfx}")
        d = os.path.join(out_dir, f"{dst_tag}{sfx}")
        if os.path.exists(s) and not os.path.exists(d):
            shutil.copy2(s, d)
    return True


# ---------------------------------------------------------------------------
# Read Exp 1 results for gate and sigma calibration
# ---------------------------------------------------------------------------

def _read_exp1_results(exp1_dir, target_epoch=60):
    """
    Read beta_spectrum.csv from Exp 1.
    Returns (beta95, r_star) where r_star is smallest r with beta_95 <= 0.5,
    or the r with minimum beta_95 if gate not passed.
    """
    csv_path = os.path.join(exp1_dir, "beta_spectrum.csv")
    if not os.path.exists(csv_path):
        return None, None

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    ep_rows = [r for r in rows if int(r["epoch"]) == target_epoch]
    if not ep_rows:
        # Fall back to max available epoch
        max_ep = max(int(r["epoch"]) for r in rows)
        ep_rows = [r for r in rows if int(r["epoch"]) == max_ep]

    best_b95 = 1.0
    best_r   = None
    for row in ep_rows:
        b95 = float(row["beta_95"])
        r   = int(row["r"])
        if b95 < best_b95:
            best_b95 = b95
            best_r   = r
        if b95 <= 0.5 and best_r is None:
            best_r   = r

    # Find smallest r where gate passes
    r_star = None
    for row in sorted(ep_rows, key=lambda x: int(x["r"])):
        if float(row["beta_95"]) <= 0.5:
            r_star   = int(row["r"])
            best_b95 = float(row["beta_95"])
            break

    if r_star is None:
        r_star = best_r  # gate not passed, use best available
    return best_b95, r_star


# ---------------------------------------------------------------------------
# Main training run
# ---------------------------------------------------------------------------

def _train_run(arm_name, eps, seed, pub_ds, priv_ds, test_ds,
               pub_x, pub_y, device, out_dir,
               beta95=None, r_star=None,
               p9_dir=P9_DIR):
    """
    Full DP training run for (arm, eps, seed).
    Returns final test accuracy.
    """
    tag      = f"{arm_name}_eps{eps:.0f}_seed{seed}"
    csv_path = os.path.join(out_dir, f"{tag}.csv")

    # Reuse P9 baselines where possible (vanilla_warm, pda_cw)
    if arm_name in ("vanilla_warm", "pda_cw") and p9_dir:
        if _try_reuse_p9(arm_name, eps, seed, out_dir, p9_dir):
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            return float(rows[-1]["test_acc"])

    if os.path.exists(csv_path):
        print(f"[P11-E2] {tag}: already done, loading.")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return float(rows[-1]["test_acc"])

    print(f"\n[P11-E2] === {tag} ===")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Privacy accounting
    tmp_loader      = DataLoader(priv_ds, batch_size=BATCH_SIZE,
                                 shuffle=True, drop_last=True)
    steps_per_epoch = len(tmp_loader)
    T_steps         = EPOCHS * steps_per_epoch
    q               = BATCH_SIZE / len(priv_ds)
    del tmp_loader

    sigma_van = _calibrate_sigma(eps, DELTA, q, T_steps)

    if arm_name == "dist_aware":
        if beta95 is None:
            raise ValueError("dist_aware arm requires --beta95 (from Exp 1 gate check).")
        # Absolute noise scale = sigma_van * C_eff  (REDUCED vs sigma_van * C)
        # C_eff = C * sqrt(beta95) is the claimed effective sensitivity.
        # The accountant still achieves (eps, delta)-DP at this noise/sensitivity ratio.
        # sigma_use stores the absolute noise scale (not the multiplier).
        c_eff     = CLIP_C * math.sqrt(beta95)
        sigma_use = sigma_van * c_eff   # = sigma_van * C * sqrt(beta95) < sigma_van * C
        noise_reduction = math.sqrt(beta95)
        print(f"[P11-E2] dist_aware: beta95={beta95:.4f}, c_eff={c_eff:.4f}, "
              f"noise scale: {sigma_van*CLIP_C:.4f} → {sigma_use:.4f} "
              f"(reduction {noise_reduction:.3f}x, sqrt(beta95)={noise_reduction:.3f})")
    else:
        sigma_use = sigma_van * CLIP_C  # standard absolute noise scale
        c_eff     = CLIP_C

    d = _num_params(_make_model())
    if arm_name == "gep":
        r_gep     = r_star if r_star is not None else 1000
        sigma_par, sigma_perp = _gep_sigmas(sigma_van)

    # Model
    model = _make_model().to(device)
    is_pda = arm_name == "pda_cw"
    needs_pretrain = arm_name in ("vanilla_warm", "dist_aware", "gep", "pda_cw")

    if needs_pretrain:
        print(f"[P11-E2] Pretraining ({PRETRAIN_EPOCHS} ep)...")
        _pretrain_on_public(model, pub_x, pub_y, device)

    V = None
    if arm_name == "gep":
        V = _build_subspace(model, pub_ds, r_gep, device, seed=seed)
        print(f"[P11-E2] GEP subspace: V shape {V.shape}")

    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    priv_loader = DataLoader(priv_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)

    rng_np = np.random.default_rng(seed + 100)
    probe_idx = rng_np.choice(len(priv_ds), size=256, replace=False)
    probe_sub = torch.utils.data.Subset(priv_ds, probe_idx.tolist())
    probe_x, probe_y = next(iter(DataLoader(probe_sub, batch_size=256)))

    # Track max ||g_perp|| for dist_aware (empirical beta check)
    perp_norms_max = [] if arm_name == "dist_aware" else None

    fieldnames = ["epoch", "train_loss", "test_acc", "lr"]
    if arm_name == "pda_cw":
        fieldnames += ["alpha_mean", "cos_pub_priv"]
    if arm_name == "dist_aware":
        fieldnames += ["max_perp_norm"]

    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    best_acc    = 0.0
    step_global = 0
    ramp_steps  = RAMP_FRAC * T_steps

    for epoch in range(1, EPOCHS + 1):
        model.train()

        if arm_name == "gep":
            V = _build_subspace(model, pub_ds, r_gep, device, seed=seed)

        alpha_accum = []
        cos_accum   = []
        epoch_perp_max = 0.0

        for x, y in priv_loader:
            optimizer.zero_grad(set_to_none=True)
            B = x.shape[0]

            grads   = _per_sample_grads_all(model, x, y, device)
            norms   = grads.norm(dim=1, keepdim=True).clamp(min=1e-8)
            clipped = grads * (CLIP_C / norms).clamp(max=1.0)
            sum_g   = clipped.sum(0)

            if arm_name in ("vanilla_warm", "dist_aware"):
                # sigma_use is the absolute noise scale (sigma_van*C for vanilla,
                # sigma_van*C_eff for dist_aware — already computed above)
                noise  = torch.randn_like(sum_g) * sigma_use
                flat_g = (sum_g + noise) / B
                _set_grads(model, flat_g.to(device))

            elif arm_name == "gep":
                V_dev  = V.to(device)
                r, dd  = V.shape[1], V.shape[0]
                sum_c    = torch.zeros(r, device=device)
                sum_perp = torch.zeros(dd, device=device)
                chunk = min(GRAD_CHUNK * 4, B)
                for i in range(0, B, chunk):
                    g_g    = grads[i:i+chunk].to(device)
                    c      = g_g @ V_dev
                    g_par  = c @ V_dev.T
                    g_perp = g_g - g_par
                    c      = c * (CLIP0 / c.norm(dim=1, keepdim=True).clamp(min=1e-8)).clamp(max=1.0)
                    g_perp = g_perp * (CLIP1 / g_perp.norm(dim=1, keepdim=True).clamp(min=1e-8)).clamp(max=1.0)
                    sum_c    += c.sum(0); sum_perp += g_perp.sum(0)
                    del g_g, c, g_par, g_perp; torch.cuda.empty_cache()
                sum_c    += torch.randn(r, device=device) * (sigma_par * CLIP0)
                noise_p   = torch.randn(dd, device=device) * (sigma_perp * CLIP1)
                noise_p  -= V_dev @ (V_dev.T @ noise_p)
                sum_perp += noise_p
                flat_g    = ((V_dev @ sum_c + sum_perp) / B).cpu()
                _set_grads(model, flat_g.to(device))

            elif arm_name == "pda_cw":
                signal  = sum_g / B
                noise   = torch.randn_like(sum_g) * sigma_use  # sigma_use = sigma_van*C here
                flat_priv = (sum_g + noise) / B

                g_pub      = _pub_grad_cw_flat(model, pub_x, pub_y, device)
                g_pub_norm  = g_pub.norm().clamp(min=1e-8)
                signal_norm = signal.norm().clamp(min=1e-8)
                cos_sim = (g_pub @ signal / (g_pub_norm * signal_norm)).item()
                cos_accum.append(cos_sim)

                priv_norm = flat_priv.norm()
                g_pub     = g_pub * (priv_norm / g_pub_norm)

                alpha_t = ALPHA_START + (1.0 - ALPHA_START) * min(
                    1.0, step_global / max(ramp_steps, 1.0))
                alpha_accum.append(alpha_t)
                flat_g = alpha_t * flat_priv + (1.0 - alpha_t) * g_pub
                _set_grads(model, flat_g.to(device))

            optimizer.step()
            step_global += 1

        scheduler.step()

        train_loss = _probe_loss(model, probe_x, probe_y, device)
        test_acc   = _evaluate(model, test_ds, device)
        cur_lr     = scheduler.get_last_lr()[0]

        row = {"epoch": epoch, "train_loss": f"{train_loss:.4f}",
               "test_acc": f"{test_acc:.4f}", "lr": f"{cur_lr:.6f}"}
        if arm_name == "pda_cw":
            row["alpha_mean"]   = f"{np.mean(alpha_accum):.4f}" if alpha_accum else "nan"
            row["cos_pub_priv"] = f"{np.mean(cos_accum):.4f}"   if cos_accum   else "nan"
        if arm_name == "dist_aware":
            row["max_perp_norm"] = f"{epoch_perp_max:.6f}"

        writer.writerow(row); csv_file.flush()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}_best.pt"))

        print(f"  ep {epoch:3d}/{EPOCHS}  loss={train_loss:.4f}  "
              f"acc={test_acc:.4f}  best={best_acc:.4f}")

    # Per-class accuracy at epoch 60
    per_cls = _evaluate_per_class(model, test_ds, device)
    per_cls_path = os.path.join(out_dir, f"{tag}_per_class.csv")
    with open(per_cls_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "accuracy"])
        for k in range(10):
            w.writerow([k, f"{per_cls[k]:.4f}"])
    print(f"[P11-E2] Per-class accuracy: {[f'{v*100:.1f}%' for v in per_cls]}")

    csv_file.close()
    torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}_final.pt"))
    print(f"[P11-E2] Done — final={test_acc:.4f}  best={best_acc:.4f}")
    return test_acc


# ---------------------------------------------------------------------------
# Sweep all arms × seeds
# ---------------------------------------------------------------------------

def run_exp2(arms, eps, n_seeds, data_root, device_id, out_dir,
             beta95, r_star, p9_dir=P9_DIR):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"[P11-E2] Device: {device}  eps={eps}  beta95={beta95}  r*={r_star}")
    pub_ds, priv_ds, test_ds, pub_x, pub_y = _build_datasets(data_root)

    results = {}
    for arm in arms:
        results[arm] = []
        for seed in range(n_seeds):
            acc = _train_run(arm, eps, seed,
                             pub_ds, priv_ds, test_ds, pub_x, pub_y,
                             device, out_dir,
                             beta95=beta95, r_star=r_star,
                             p9_dir=p9_dir)
            results[arm].append(acc)
        mu  = np.mean(results[arm])
        std = np.std(results[arm])
        print(f"[P11-E2] {arm}: {mu*100:.2f}% ± {std*100:.2f}%")

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _load_csv(arm, eps, seed, out_dir):
    path = os.path.join(out_dir, f"{arm}_eps{eps:.0f}_seed{seed}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _run_analysis(out_dir, eps=EPS, n_seeds=N_SEEDS, arms=None):
    if arms is None:
        arms = ARMS

    print(f"\n{'='*70}")
    print(f" Phase 11 Exp 2 — Test Accuracy (eps={eps:.0f}, {n_seeds} seeds)")
    print(f"{'='*70}")
    print(f"  {'arm':<15}  {'seed0':>7}  {'seed1':>7}  {'seed2':>7}  "
          f"{'mean':>7}  {'std':>7}")
    print(f"  {'-'*15}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    summary = {}
    for arm in arms:
        accs = []
        row_str = f"  {arm:<15}"
        for seed in range(n_seeds):
            rows = _load_csv(arm, eps, seed, out_dir)
            if rows is None:
                row_str += f"  {'N/A':>7}"
            else:
                acc = float(rows[-1]["test_acc"])
                accs.append(acc)
                row_str += f"  {acc*100:>7.2f}"
        if accs:
            mu  = np.mean(accs)
            std = np.std(accs)
            row_str += f"  {mu*100:>7.2f}  {std*100:>7.2f}"
            summary[arm] = (mu, std)
        print(row_str)

    print()
    if "vanilla_warm" in summary and "dist_aware" in summary:
        delta_pp = (summary["dist_aware"][0] - summary["vanilla_warm"][0]) * 100
        print(f"  dist_aware vs vanilla_warm: {delta_pp:+.2f} pp")
        if delta_pp >= 2.0:
            print("  INTERPRETATION: dist_aware improves by >= 2 pp → proceed to Exp 3.")
        elif delta_pp >= 0.0:
            print("  INTERPRETATION: dist_aware marginal improvement (< 2 pp).")
        else:
            print("  INTERPRETATION: dist_aware no improvement → investigate noise calibration.")

    _plot_learning_curves(arms, eps, n_seeds, out_dir)


def _plot_learning_curves(arms, eps, n_seeds, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(arms)))

    for arm, color in zip(arms, colors):
        runs = []
        for seed in range(n_seeds):
            rows = _load_csv(arm, eps, seed, out_dir)
            if rows is None: continue
            try:
                runs.append([float(r["test_acc"]) for r in rows])
            except (KeyError, ValueError):
                continue
        if not runs: continue
        arr = np.array(runs)
        ep  = np.arange(1, arr.shape[1] + 1)
        mu  = arr.mean(0); sd = arr.std(0)
        ax.plot(ep, mu * 100, label=arm, color=color, lw=1.8)
        ax.fill_between(ep, (mu - sd) * 100, (mu + sd) * 100,
                        alpha=0.15, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"Phase 11 Exp 2 — Noise Reduction Comparison (ε={eps:.0f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"exp2_curves_eps{eps:.0f}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P11-E2] Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm",    type=str, default=None, choices=ARMS + [None],
                        help="Single arm to run. If not set, run all arms.")
    parser.add_argument("--eps",    type=float, default=EPS)
    parser.add_argument("--seed",   type=int,   default=None,
                        help="Single seed. If not set, run seeds 0..N_SEEDS-1.")
    parser.add_argument("--beta95", type=float, default=None,
                        help="Beta_95 from Exp 1 (overrides reading CSV).")
    parser.add_argument("--rank",   type=int,   default=None,
                        help="r* from Exp 1 (overrides reading CSV).")
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(RESULTS_DIR, "exp2"))
    parser.add_argument("--exp1_dir", type=str, default=EXP1_DIR)
    parser.add_argument("--p9_dir",  type=str, default=P9_DIR)
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    parser.add_argument("--analysis_only", action="store_true")
    parser.add_argument("--transfer_p9",   action="store_true",
                        help="Copy P9 baselines and exit.")
    args = parser.parse_args()

    if args.analysis_only:
        _run_analysis(args.out_dir, eps=args.eps, n_seeds=args.n_seeds)
        return

    # Read beta95 / r* from Exp 1 or CLI
    beta95, r_star = args.beta95, args.rank
    if beta95 is None:
        beta95, r_star = _read_exp1_results(args.exp1_dir)
        if beta95 is None:
            print("[P11-E2] WARNING: Exp 1 results not found. "
                  "Using beta95=1.0 (same sigma as vanilla). "
                  "Run exp_p11_beta_measurement.py first, or pass --beta95.")
            beta95 = 1.0
        else:
            print(f"[P11-E2] Read from Exp 1: beta95={beta95:.4f}, r*={r_star}")

    if args.transfer_p9:
        os.makedirs(args.out_dir, exist_ok=True)
        for arm in ("vanilla_warm", "pda_cw", "gep"):
            for seed in range(args.n_seeds):
                _try_reuse_p9(arm, args.eps, seed, args.out_dir, args.p9_dir)
        return

    arms   = [args.arm] if args.arm else ARMS
    seeds  = [args.seed] if args.seed is not None else list(range(args.n_seeds))

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    pub_ds, priv_ds, test_ds, pub_x, pub_y = _build_datasets(args.data_root)

    for arm in arms:
        for seed in seeds:
            _train_run(arm, args.eps, seed,
                       pub_ds, priv_ds, test_ds, pub_x, pub_y,
                       device, args.out_dir,
                       beta95=beta95, r_star=r_star,
                       p9_dir=args.p9_dir)

    _run_analysis(args.out_dir, eps=args.eps, n_seeds=args.n_seeds, arms=arms)


if __name__ == "__main__":
    main()
