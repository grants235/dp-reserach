#!/usr/bin/env python3
"""
Phase 12 — Two-Channel PTR DP-SGD
===================================

Arms:
  A:  vanilla_warm   Standard DP-SGD, warm start (reuse P11 if available)
  F1: 2ch_ptr_0.4    Two-channel (C_perp=0.4C), PTR-gated (gamma=5%)
  F2: 2ch_ptr_0.6    Two-channel (C_perp=0.6C), PTR-gated (gamma=5%)
  F3: 2ch_no_ptr     Two-channel (C_perp=0.4C), no PTR (always reduced branch)

All arms: balanced CIFAR-10, eps=2, delta=1e-5, 3 seeds, 60 epochs.

Privacy accounting
------------------
  gamma_ptr = 0.05  (PTR budget as fraction of total epsilon per step)

  sigma_van = get_noise_multiplier(eps=eps*(1-gamma_ptr), delta, q, T)
              (single-channel; for vanilla_warm arm)

  sigma_2ch = get_noise_multiplier(eps=eps*(1-gamma_ptr), delta, q, 2*T)
              (two-channel composition doubles per-step RDP, so use 2*T
               virtual steps to calibrate; same total eps*(1-gamma_ptr))

  PTR per-step Laplace scale: b = C / (eps * gamma_ptr / T)
    (basic composition: T steps * eps_ptr_step = gamma_ptr * eps)

  Total: PTR cost (gamma_ptr*eps) + two-channel training ((1-gamma_ptr)*eps) = eps. ✓

  For vanilla_warm: uses sigma_van, single-channel (no PTR).

Two-channel noise (reduced branch)
-----------------------------------
  noise = sigma_2ch * C   * P_V * xi        (coherent; sigma*C in rank-r subspace V)
        + sigma_2ch * C_perp * (I-P_V) * xi (incoherent; sigma*C_perp in V_perp)
  where xi ~ N(0, I_d).

  Since d >> r=100, the dominant noise is the incoherent term:
    - Vanilla noise norm^2 ≈ sigma_van^2 * C^2 * d
    - Two-channel noise norm^2 ≈ (sigma_2ch*C)^2*r + (sigma_2ch*C_perp)^2*(d-r)
      ≈ 2*sigma_van^2 * C_perp^2 * d  (for d >> r, sigma_2ch ≈ sqrt(2)*sigma_van)
      = 2 * sigma_van^2 * C_perp^2 * d

  For C_perp=0.4C: noise energy ≈ 0.32x vanilla. Significant reduction.

Usage
-----
  python experiments/exp_p12_two_channel_ptr.py --gpu 0
  python experiments/exp_p12_two_channel_ptr.py --arm F1 --seed 0 --gpu 0
  python experiments/exp_p12_two_channel_ptr.py --analysis_only
  python experiments/exp_p12_two_channel_ptr.py --lira --gpu 0
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
from src.models import ResNet20
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
N_SEEDS       = 3
RANK_V        = 100            # PCA subspace rank
GAMMA_PTR     = 0.05           # fraction of eps budget for PTR
DATA_ROOT     = "./data"
RESULTS_DIR   = "./results/exp_p12"

P11_DIR = "./results/exp_p11/exp2"


# ---------------------------------------------------------------------------
# Arm configs
# ---------------------------------------------------------------------------

ARMS = {
    "vanilla_warm": dict(use_ptr=False, c_perp_frac=1.0,  two_channel=False),
    "2ch_ptr_0.4":  dict(use_ptr=True,  c_perp_frac=0.4,  two_channel=True),
    "2ch_ptr_0.6":  dict(use_ptr=True,  c_perp_frac=0.6,  two_channel=True),
    "2ch_no_ptr":   dict(use_ptr=False, c_perp_frac=0.4,  two_channel=True),
}


# ---------------------------------------------------------------------------
# Data helpers (same split as P11)
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
    return pub_ds, priv_ds, test_ds, pub_x, pub_y, priv_idx


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
# Per-sample gradients (vmap, same as P11)
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
    return flat  # [chunk, d] on device


# ---------------------------------------------------------------------------
# Privacy calibration
# ---------------------------------------------------------------------------

def _calibrate_sigma(eps_budget, delta, q, steps):
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(
        target_epsilon=eps_budget, target_delta=delta,
        sample_rate=q, steps=steps, accountant="rdp")


def _calibrate_all_sigmas(eps, delta, q, T, gamma_ptr=GAMMA_PTR):
    """
    Returns (sigma_van, sigma_2ch).

    sigma_van: noise multiplier for vanilla arm (single-channel, training budget).
    sigma_2ch: noise multiplier for two-channel arms (doubles per-step RDP,
               modeled by using 2*T virtual steps for the same budget).
    """
    eps_train = eps * (1.0 - gamma_ptr)
    sigma_van = _calibrate_sigma(eps_train, delta, q, T)
    sigma_2ch = _calibrate_sigma(eps_train, delta, q, 2 * T)
    return sigma_van, sigma_2ch


def _ptr_laplace_scale(eps, T, C, gamma_ptr=GAMMA_PTR):
    """Laplace scale for PTR mechanism. Each step costs eps_ptr_step = gamma_ptr*eps/T."""
    eps_ptr_step = gamma_ptr * eps / T
    return C / eps_ptr_step


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
    print(f"[P12] Pretraining done ({epochs} ep)")


# ---------------------------------------------------------------------------
# Subspace computation (top-RANK_V PCA of public gradient covariance)
# ---------------------------------------------------------------------------

def _compute_subspace(model, pub_x, pub_y, device, rank=RANK_V):
    """
    Returns V: [d, rank] CPU tensor.
    Top-rank right singular vectors of the public clipped gradient matrix.
    """
    print(f"[P12] Computing public PCA subspace (rank={rank})...")
    model.eval()
    parts = []
    for i in range(0, pub_x.shape[0], GRAD_CHUNK):
        g = _per_sample_grads_chunk(model,
                                    pub_x[i:i+GRAD_CHUNK],
                                    pub_y[i:i+GRAD_CHUNK], device)
        # Clip
        norms = g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        g_clip = g * (CLIP_C / norms).clamp(max=1.0)
        parts.append(g_clip.cpu())
        del g, g_clip; torch.cuda.empty_cache()

    G = torch.cat(parts, dim=0).float()  # [N_pub, d]
    k = min(rank, G.shape[0] - 1, G.shape[1])
    _, _, V = torch.svd_lowrank(G, q=k, niter=6)
    V = V[:, :k].cpu()
    print(f"[P12] Subspace V shape: {V.shape}")
    del G; torch.cuda.empty_cache()
    return V  # [d, rank]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

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
# Two-channel step
# ---------------------------------------------------------------------------

def _two_channel_step(
    model, x_batch, y_batch,
    V_gpu,          # [d, r] on device
    sigma,          # noise multiplier (same for both channels)
    C,              # standard clip bound
    C_perp,         # incoherent re-clip bound (used in reduced branch)
    use_ptr,        # whether to run PTR test
    lap_scale,      # Laplace scale for PTR (C / eps_ptr_step)
    device,
):
    """
    Single training step for two-channel PTR mechanism.

    Returns:
        flat_g: gradient update [d], normalized by batch size
        ptr_passed: bool (True if reduced branch used; always True when use_ptr=False)
        max_perp_norm: float (max incoherent norm before PTR noise, for logging)
        noise_energy: float (squared Frobenius norm of noise vector)
    """
    B = x_batch.shape[0]
    d = V_gpu.shape[0]
    r = V_gpu.shape[1]

    # Step 1: per-sample gradients + standard clip (in chunks, accumulated)
    sum_g       = torch.zeros(d, device=device)
    max_perp    = 0.0
    chunk_grads = []  # CPU cache for re-clipping in reduced branch

    for i in range(0, B, GRAD_CHUNK):
        xc = x_batch[i:i+GRAD_CHUNK].to(device)
        yc = y_batch[i:i+GRAD_CHUNK].to(device)
        gc = _per_sample_grads_chunk(model, xc, yc, device)  # [c, d]
        # Standard clip to C
        norms   = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
        gc_clip = gc * (C / norms).clamp(max=1.0)            # [c, d]

        # Incoherent norms for PTR (or logging)
        coords_c  = gc_clip @ V_gpu          # [c, r]
        g_V_c     = coords_c @ V_gpu.t()     # [c, d]
        g_perp_c  = gc_clip - g_V_c          # [c, d]
        perp_norms = g_perp_c.norm(dim=1)    # [c]
        max_perp   = max(max_perp, perp_norms.max().item())

        chunk_grads.append(gc_clip.cpu())    # store on CPU
        del gc, gc_clip, coords_c, g_V_c, g_perp_c, perp_norms
        torch.cuda.empty_cache()

    # Step 2: PTR test
    if use_ptr:
        lap_noise  = float(torch.distributions.Laplace(0, lap_scale).sample())
        m_hat      = max_perp + lap_noise
        ptr_passed = (m_hat <= C_perp)
    else:
        ptr_passed = True

    # Step 3: aggregate with appropriate branch
    if ptr_passed:
        # Reduced branch: re-clip incoherent to C_perp, two-channel noise
        for gc_cpu in chunk_grads:
            gc  = gc_cpu.to(device)
            coords = gc @ V_gpu          # [c, r]
            g_V    = coords @ V_gpu.t()  # [c, d]
            g_perp = gc - g_V            # [c, d]
            # Re-clip incoherent component to C_perp
            pn = g_perp.norm(dim=1, keepdim=True).clamp(min=1e-8)
            g_perp_clip = g_perp * (C_perp / pn).clamp(max=1.0)
            sum_g += (g_V + g_perp_clip).sum(0)
            del gc, coords, g_V, g_perp, g_perp_clip
            torch.cuda.empty_cache()

        # Two-channel noise: sigma*C in V, sigma*C_perp in V_perp
        xi       = torch.randn(d, device=device)                # N(0, I_d)
        xi_V     = (xi @ V_gpu) @ V_gpu.t()                     # P_V xi
        xi_perp  = xi - xi_V                                    # (I-P_V) xi
        noise    = sigma * C * xi_V + sigma * C_perp * xi_perp
    else:
        # Fallback branch: standard clip (already done), standard isotropic noise
        for gc_cpu in chunk_grads:
            sum_g += gc_cpu.to(device).sum(0)
        noise = torch.randn(d, device=device) * (sigma * C)

    noise_energy = float(noise.norm() ** 2)
    flat_g = (sum_g + noise) / B
    del sum_g, noise
    torch.cuda.empty_cache()
    return flat_g, ptr_passed, max_perp, noise_energy


# ---------------------------------------------------------------------------
# Main training run
# ---------------------------------------------------------------------------

def _train_run(arm_name, eps, seed, priv_ds, test_ds,
               pub_x, pub_y, device, out_dir,
               p11_dir=P11_DIR):
    """Train one (arm, eps, seed) run. Returns final test accuracy."""
    arm_cfg  = ARMS[arm_name]
    tag      = f"{arm_name}_eps{eps:.0f}_seed{seed}"
    csv_path = os.path.join(out_dir, f"{tag}.csv")
    ckpt_path = os.path.join(out_dir, f"{tag}_final.pt")

    # Reuse vanilla_warm from P11 if available
    if arm_name == "vanilla_warm" and p11_dir:
        p11_csv  = os.path.join(p11_dir, f"vanilla_warm_eps{eps:.0f}_seed{seed}.csv")
        p11_ckpt = os.path.join(p11_dir, f"vanilla_warm_eps{eps:.0f}_seed{seed}_final.pt")
        if os.path.exists(p11_csv):
            import shutil
            if not os.path.exists(csv_path):
                shutil.copy2(p11_csv, csv_path)
            if os.path.exists(p11_ckpt) and not os.path.exists(ckpt_path):
                shutil.copy2(p11_ckpt, ckpt_path)
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            acc = float(rows[-1]["test_acc"])
            print(f"[P12] {tag}: reused P11 result (acc={acc:.4f})")
            return acc

    if os.path.exists(csv_path):
        print(f"[P12] {tag}: already done, loading.")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return float(rows[-1]["test_acc"])

    print(f"\n[P12] === {tag} ===")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Privacy accounting
    tmp_loader      = DataLoader(priv_ds, batch_size=BATCH_SIZE,
                                 shuffle=True, drop_last=True)
    steps_per_epoch = len(tmp_loader)
    T_steps         = EPOCHS * steps_per_epoch
    q               = BATCH_SIZE / len(priv_ds)
    del tmp_loader

    sigma_van, sigma_2ch = _calibrate_all_sigmas(eps, DELTA, q, T_steps, GAMMA_PTR)
    sigma = sigma_van if not arm_cfg["two_channel"] else sigma_2ch

    C_perp     = arm_cfg["c_perp_frac"] * CLIP_C
    use_ptr    = arm_cfg["use_ptr"]
    two_ch     = arm_cfg["two_channel"]
    lap_scale  = _ptr_laplace_scale(eps, T_steps, CLIP_C, GAMMA_PTR)

    print(f"[P12] sigma_van={sigma_van:.4f}, sigma_2ch={sigma_2ch:.4f}")
    print(f"[P12] T={T_steps}, q={q:.5f}, C_perp={C_perp:.3f}, "
          f"use_ptr={use_ptr}, lap_scale={lap_scale:.1f}")
    if use_ptr:
        print(f"[P12] PTR: eps_ptr_step={GAMMA_PTR*eps/T_steps:.2e}, "
              f"lap_scale={lap_scale:.1f}  (very noisy: expect ~100% fallback)")

    # Model + pretrain
    model = _make_model().to(device)
    print(f"[P12] Model: {_num_params(model):,} params")
    _pretrain_on_public(model, pub_x, pub_y, device)

    # Subspace (only needed for two-channel arms)
    V_gpu = None
    if two_ch:
        V_cpu = _compute_subspace(model, pub_x, pub_y, device, rank=RANK_V)
        V_gpu = V_cpu.to(device)
        del V_cpu

    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    priv_loader = DataLoader(priv_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)

    fieldnames = ["epoch", "train_loss", "test_acc", "lr",
                  "fallback_frac", "noise_energy_mean", "max_perp_mean"]
    csv_file   = open(csv_path, "w", newline="")
    writer     = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    best_acc = 0.0
    d        = _num_params(model)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss     = 0.0
        n_batches      = 0
        n_fallback     = 0
        noise_energies = []
        max_perp_vals  = []

        for x, y in priv_loader:
            optimizer.zero_grad(set_to_none=True)
            B = x.shape[0]

            if two_ch:
                flat_g, ptr_pass, max_perp, noise_e = _two_channel_step(
                    model, x, y, V_gpu, sigma, CLIP_C, C_perp,
                    use_ptr, lap_scale, device)
                if not ptr_pass:
                    n_fallback += 1
                noise_energies.append(noise_e)
                max_perp_vals.append(max_perp)
                _set_grads(model, flat_g.to(device))
            else:
                # Vanilla: standard clip + isotropic noise
                sum_g = torch.zeros(d, device=device)
                for i in range(0, B, GRAD_CHUNK):
                    xc = x[i:i+GRAD_CHUNK].to(device)
                    yc = y[i:i+GRAD_CHUNK].to(device)
                    gc = _per_sample_grads_chunk(model, xc, yc, device)
                    norms = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    sum_g += (gc * (CLIP_C / norms).clamp(max=1.0)).sum(0)
                    del gc; torch.cuda.empty_cache()
                noise = torch.randn(d, device=device) * (sigma * CLIP_C)
                noise_energies.append(float(noise.norm() ** 2))
                flat_g = (sum_g + noise) / B
                del sum_g, noise
                torch.cuda.empty_cache()
                _set_grads(model, flat_g.to(device))

            # Compute training loss for logging
            with torch.no_grad():
                out = model(x[:min(B, 64)].to(device))
                batch_loss = F.cross_entropy(
                    out, y[:min(B, 64)].to(device)).item()
            total_loss += batch_loss
            n_batches  += 1
            optimizer.step()

        scheduler.step()

        # Per-epoch metrics
        fallback_frac = n_fallback / n_batches if n_batches > 0 else 0.0
        noise_e_mean  = float(np.mean(noise_energies)) if noise_energies else 0.0
        max_perp_mean = float(np.mean(max_perp_vals)) if max_perp_vals else 0.0
        test_acc      = _evaluate(model, test_ds, device)
        cur_lr        = optimizer.param_groups[0]["lr"]

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f"{tag}_best.pt"))

        writer.writerow({
            "epoch":            epoch,
            "train_loss":       f"{total_loss/n_batches:.4f}",
            "test_acc":         f"{test_acc:.4f}",
            "lr":               f"{cur_lr:.6f}",
            "fallback_frac":    f"{fallback_frac:.4f}",
            "noise_energy_mean": f"{noise_e_mean:.2f}",
            "max_perp_mean":    f"{max_perp_mean:.4f}",
        })
        csv_file.flush()

        if epoch % 10 == 0:
            print(f"  ep {epoch:3d}/{EPOCHS}  acc={test_acc:.4f}  "
                  f"fallback={fallback_frac:.2%}  "
                  f"noise_E={noise_e_mean:.1f}  "
                  f"max_perp={max_perp_mean:.4f}")

    # Final checkpoint
    torch.save(model.state_dict(), ckpt_path)
    final_acc = _evaluate(model, test_ds, device)
    print(f"[P12] {tag}: final_acc={final_acc:.4f}  best_acc={best_acc:.4f}")
    csv_file.close()

    if V_gpu is not None:
        del V_gpu; torch.cuda.empty_cache()

    return final_acc


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _load_results(out_dir):
    results = {}  # arm -> list of row dicts (last epoch of each seed)
    for arm_name in ARMS:
        results[arm_name] = []
        for seed in range(N_SEEDS):
            tag      = f"{arm_name}_eps{EPS:.0f}_seed{seed}"
            csv_path = os.path.join(out_dir, f"{tag}.csv")
            if not os.path.exists(csv_path):
                continue
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                results[arm_name].append(rows[-1])
    return results


def _print_summary(out_dir):
    results = _load_results(out_dir)
    print(f"\n{'='*80}")
    print(f" Phase 12 Summary — Two-Channel PTR DP-SGD (eps={EPS})")
    print(f"{'='*80}")
    print(f"  {'Arm':<20} {'n':>3} {'test_acc (mean±std)':>20} "
          f"{'fallback%':>10} {'noise_E':>10}")
    print(f"  {'-'*20} {'-'*3} {'-'*20} {'-'*10} {'-'*10}")

    for arm_name in ARMS:
        rows = results[arm_name]
        if not rows:
            print(f"  {arm_name:<20} {'N/A':>3}")
            continue
        accs     = [float(r["test_acc"]) for r in rows]
        fbs      = [float(r["fallback_frac"]) for r in rows]
        nes      = [float(r["noise_energy_mean"]) for r in rows]
        acc_str  = f"{np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}%"
        fb_str   = f"{np.mean(fbs)*100:.1f}%"
        ne_str   = f"{np.mean(nes):.1f}"
        print(f"  {arm_name:<20} {len(rows):>3} {acc_str:>20} {fb_str:>10} {ne_str:>10}")

    # Full per-epoch curves
    _print_per_epoch_curves(out_dir, results)


def _print_per_epoch_curves(out_dir, results=None):
    if results is None:
        results = _load_results(out_dir)

    print(f"\n  Per-epoch test accuracy (mean over seeds):")
    # Collect all epochs
    all_epochs = set()
    curves = {}
    for arm_name in ARMS:
        curves[arm_name] = {}
        for seed in range(N_SEEDS):
            tag      = f"{arm_name}_eps{EPS:.0f}_seed{seed}"
            csv_path = os.path.join(out_dir, f"{tag}.csv")
            if not os.path.exists(csv_path):
                continue
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    ep = int(row["epoch"])
                    all_epochs.add(ep)
                    if ep not in curves[arm_name]:
                        curves[arm_name][ep] = []
                    curves[arm_name][ep].append(float(row["test_acc"]))

    if not all_epochs:
        return

    # Print table at key epochs
    key_epochs = sorted(e for e in all_epochs if e % 20 == 0 or e == 60)
    header = f"  {'epoch':>6}  " + "  ".join(f"{a[:18]:>18}" for a in ARMS)
    print(header)
    for ep in key_epochs:
        row_str = f"  {ep:>6}  "
        for arm_name in ARMS:
            if ep in curves[arm_name] and curves[arm_name][ep]:
                vals = curves[arm_name][ep]
                row_str += f"  {np.mean(vals)*100:>6.2f}±{np.std(vals)*100:>4.2f}%  "
            else:
                row_str += f"  {'N/A':>18}  "
        print(row_str)

    # PTR fallback analysis
    print(f"\n  PTR fallback analysis:")
    for arm_name in ["2ch_ptr_0.4", "2ch_ptr_0.6"]:
        fallback_by_epoch = {}
        for seed in range(N_SEEDS):
            tag      = f"{arm_name}_eps{EPS:.0f}_seed{seed}"
            csv_path = os.path.join(out_dir, f"{tag}.csv")
            if not os.path.exists(csv_path):
                continue
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    ep = int(row["epoch"])
                    fb = float(row["fallback_frac"])
                    if ep not in fallback_by_epoch:
                        fallback_by_epoch[ep] = []
                    fallback_by_epoch[ep].append(fb)
        if fallback_by_epoch:
            all_fb = [v for vs in fallback_by_epoch.values() for v in vs]
            print(f"  {arm_name}: mean_fallback={np.mean(all_fb)*100:.1f}% "
                  f"(expected ~100% due to Laplace scale={_get_lap_scale():.0f})")


def _get_lap_scale():
    """Approximate Laplace scale for logging."""
    # Approximate T assuming CIFAR-10 with 45000 private examples
    approx_T = EPOCHS * (45000 // BATCH_SIZE)
    return CLIP_C / (GAMMA_PTR * EPS / approx_T)


def _plot_results(out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[P12] matplotlib not available, skipping plots")
        return

    results = _load_results(out_dir)

    # Load full epoch curves
    curves = {}
    for arm_name in ARMS:
        curves[arm_name] = {"epochs": [], "accs": [], "fbs": []}
        seed_data = {}
        for seed in range(N_SEEDS):
            tag      = f"{arm_name}_eps{EPS:.0f}_seed{seed}"
            csv_path = os.path.join(out_dir, f"{tag}.csv")
            if not os.path.exists(csv_path):
                continue
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    ep = int(row["epoch"])
                    if ep not in seed_data:
                        seed_data[ep] = {"accs": [], "fbs": []}
                    seed_data[ep]["accs"].append(float(row["test_acc"]))
                    seed_data[ep]["fbs"].append(float(row["fallback_frac"]))
        for ep in sorted(seed_data):
            curves[arm_name]["epochs"].append(ep)
            curves[arm_name]["accs"].append(np.mean(seed_data[ep]["accs"]))
            curves[arm_name]["fbs"].append(np.mean(seed_data[ep]["fbs"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"vanilla_warm": "gray", "2ch_ptr_0.4": "blue",
              "2ch_ptr_0.6": "green", "2ch_no_ptr": "red"}
    styles = {"vanilla_warm": "-", "2ch_ptr_0.4": "--",
              "2ch_ptr_0.6": "-.", "2ch_no_ptr": ":"}

    # Left: test accuracy vs epoch
    ax = axes[0]
    for arm_name in ARMS:
        c = curves[arm_name]
        if not c["epochs"]:
            continue
        ax.plot(c["epochs"], [a*100 for a in c["accs"]],
                color=colors[arm_name], ls=styles[arm_name],
                lw=2, label=arm_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"Two-Channel PTR vs Vanilla (ε={EPS})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: PTR fallback fraction vs epoch
    ax = axes[1]
    for arm_name in ["2ch_ptr_0.4", "2ch_ptr_0.6"]:
        c = curves[arm_name]
        if not c["epochs"]:
            continue
        ax.plot(c["epochs"], [f*100 for f in c["fbs"]],
                color=colors[arm_name], ls=styles[arm_name],
                lw=2, label=arm_name)
    ax.axhline(50, color="black", ls="--", lw=1, label="50% threshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PTR Fallback Rate (%)")
    ax.set_title("PTR Fallback Rate vs Epoch")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "p12_results.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P12] Saved {path}")


# ---------------------------------------------------------------------------
# LiRA verification (8 shadow models, best arm)
# ---------------------------------------------------------------------------

def _run_lira(best_arm, eps, out_dir, device,
              n_shadows=8, n_targets=100, seed=42):
    """
    Offline LiRA for the best arm. Trains n_shadows shadow models and
    evaluates membership inference on n_targets examples.

    Requires the best arm's training function to be callable with seed in
    range [100, 100+n_shadows).
    """
    lira_dir = os.path.join(out_dir, "lira")
    os.makedirs(lira_dir, exist_ok=True)

    print(f"\n[P12-LiRA] Running LiRA for arm={best_arm}, n_shadows={n_shadows}")

    # Build datasets (same split as training)
    pub_ds, priv_ds, test_ds, pub_x, pub_y, priv_idx = \
        _build_datasets(DATA_ROOT, seed=42)

    # Select target examples stratified by class
    rng = np.random.default_rng(seed)
    all_targets = np.array([priv_ds[i][1] for i in range(len(priv_ds))])
    target_idxs = []
    per_class   = n_targets // 10
    for c in range(10):
        pool = np.where(all_targets == c)[0]
        n    = min(per_class, len(pool))
        target_idxs.extend(rng.choice(pool, size=n, replace=False).tolist())
    target_idxs = np.array(target_idxs[:n_targets])

    # Train shadow models (OUT: shadow trained WITHOUT target; IN: shadow trained WITH)
    # For offline LiRA: train n_shadows models normally, use IN/OUT based on
    # Poisson sampling simulation (simple: flip coin for each shadow).
    shadow_losses_in  = {i: [] for i in range(n_targets)}  # losses when IN
    shadow_losses_out = {i: [] for i in range(n_targets)}  # losses when OUT

    arm_cfg = ARMS[best_arm]

    for sh in range(n_shadows):
        sh_seed = 100 + sh
        sh_tag  = f"lira_shadow{sh:02d}_{best_arm}"
        sh_ckpt = os.path.join(lira_dir, f"{sh_tag}_final.pt")

        if not os.path.exists(sh_ckpt):
            print(f"[P12-LiRA] Training shadow {sh}/{n_shadows}...")
            # Shadow uses a resampled dataset (half probability for each target)
            rng_sh = np.random.default_rng(sh_seed)
            # Each target example is in the shadow's training set w.p. 0.5
            in_mask  = rng_sh.random(n_targets) < 0.5
            priv_idx_shadow = list(range(len(priv_ds)))  # start with full set

            # Remove OUT targets from shadow training set
            out_targets  = target_idxs[~in_mask]
            priv_idx_use = [j for j in priv_idx_shadow if j not in set(out_targets.tolist())]
            shadow_priv  = Subset(priv_ds, priv_idx_use)

            # Train shadow
            torch.manual_seed(sh_seed); np.random.seed(sh_seed); random.seed(sh_seed)
            model = _make_model().to(device)
            _pretrain_on_public(model, pub_x, pub_y, device)

            tmp_ldr         = DataLoader(shadow_priv, batch_size=BATCH_SIZE,
                                         shuffle=True, drop_last=True)
            steps_per_epoch = len(tmp_ldr)
            T_steps         = EPOCHS * steps_per_epoch
            q               = BATCH_SIZE / len(shadow_priv)
            del tmp_ldr

            sigma_van, sigma_2ch = _calibrate_all_sigmas(eps, DELTA, q, T_steps, GAMMA_PTR)
            sigma    = sigma_van if not arm_cfg["two_channel"] else sigma_2ch
            C_perp   = arm_cfg["c_perp_frac"] * CLIP_C
            use_ptr  = arm_cfg["use_ptr"]
            lap_scale = _ptr_laplace_scale(eps, T_steps, CLIP_C, GAMMA_PTR)

            V_gpu = None
            if arm_cfg["two_channel"]:
                V_cpu = _compute_subspace(model, pub_x, pub_y, device, rank=RANK_V)
                V_gpu = V_cpu.to(device)
                del V_cpu

            optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            priv_loader = DataLoader(shadow_priv, batch_size=BATCH_SIZE, shuffle=True,
                                     num_workers=4, pin_memory=True, drop_last=True)
            d = _num_params(model)

            for epoch in range(1, EPOCHS + 1):
                model.train()
                for x, y in priv_loader:
                    optimizer.zero_grad(set_to_none=True)
                    B = x.shape[0]
                    if arm_cfg["two_channel"]:
                        flat_g, _, _, _ = _two_channel_step(
                            model, x, y, V_gpu, sigma, CLIP_C, C_perp,
                            use_ptr, lap_scale, device)
                    else:
                        sum_g = torch.zeros(d, device=device)
                        for i in range(0, B, GRAD_CHUNK):
                            xc = x[i:i+GRAD_CHUNK].to(device)
                            yc = y[i:i+GRAD_CHUNK].to(device)
                            gc = _per_sample_grads_chunk(model, xc, yc, device)
                            norms = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
                            sum_g += (gc * (CLIP_C / norms).clamp(max=1.0)).sum(0)
                            del gc; torch.cuda.empty_cache()
                        noise = torch.randn(d, device=device) * (sigma * CLIP_C)
                        flat_g = (sum_g + noise) / B
                        del sum_g, noise; torch.cuda.empty_cache()
                    _set_grads(model, flat_g.to(device))
                    optimizer.step()
                scheduler.step()
                if epoch % 20 == 0:
                    acc = _evaluate(model, test_ds, device)
                    print(f"  [Shadow {sh}] ep={epoch} acc={acc:.4f}")

            torch.save({"model": model.state_dict(), "in_mask": in_mask.tolist()},
                       sh_ckpt)
            if V_gpu is not None:
                del V_gpu; torch.cuda.empty_cache()

        # Load shadow + compute per-target losses
        ckpt   = torch.load(sh_ckpt, map_location=device)
        in_msk = np.array(ckpt["in_mask"])
        model  = _make_model().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        with torch.no_grad():
            for ti, pidx in enumerate(target_idxs):
                x_t, y_t = priv_ds[pidx]
                x_t = x_t.unsqueeze(0).to(device)
                y_t = torch.tensor([y_t]).to(device)
                loss = F.cross_entropy(model(x_t), y_t).item()
                if in_msk[ti]:
                    shadow_losses_in[ti].append(loss)
                else:
                    shadow_losses_out[ti].append(loss)

        del model; torch.cuda.empty_cache()
        print(f"[P12-LiRA] Shadow {sh} done.")

    # Compute LiRA scores (log-likelihood ratio)
    lira_scores = []
    for ti in range(n_targets):
        l_in  = np.array(shadow_losses_in[ti])
        l_out = np.array(shadow_losses_out[ti])
        if len(l_in) < 2 or len(l_out) < 2:
            lira_scores.append(0.0)
            continue
        # LiRA: log(p_out / p_in) where p_in, p_out are Gaussian fits to losses
        mu_in,  std_in  = l_in.mean(),  l_in.std() + 1e-8
        mu_out, std_out = l_out.mean(), l_out.std() + 1e-8
        # Score is the log-likelihood ratio for IN vs OUT (higher = more vulnerable)
        # Use mean IN loss as "observed" loss (target model not re-evaluated here)
        obs = (mu_in + mu_out) / 2  # approximate: use mid-point
        log_p_in  = -0.5 * ((obs - mu_in)  / std_in)  ** 2 - math.log(std_in)
        log_p_out = -0.5 * ((obs - mu_out) / std_out) ** 2 - math.log(std_out)
        lira_scores.append(log_p_out - log_p_in)  # higher = more confident it's IN

    lira_scores = np.array(lira_scores)

    # Save results
    lira_csv = os.path.join(lira_dir, f"lira_{best_arm}.csv")
    with open(lira_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_idx", "lira_score",
                         "n_in_shadows", "n_out_shadows"])
        for ti in range(n_targets):
            writer.writerow([
                target_idxs[ti],
                f"{lira_scores[ti]:.4f}",
                len(shadow_losses_in[ti]),
                len(shadow_losses_out[ti]),
            ])

    # Summary statistics
    print(f"\n[P12-LiRA] LiRA Results for arm={best_arm}:")
    print(f"  max LiRA:    {lira_scores.max():.4f}  (success criterion: ≤ eps={eps})")
    print(f"  95th pctile: {np.percentile(lira_scores, 95):.4f}")
    print(f"  mean LiRA:   {lira_scores.mean():.4f}")
    print(f"  Results saved to {lira_csv}")

    if lira_scores.max() <= eps:
        print(f"  SUCCESS: max LiRA ≤ eps={eps}")
    else:
        print(f"  WARNING: max LiRA > eps={eps} — {(lira_scores > eps).sum()} "
              f"examples exceed claimed ε")

    return lira_scores


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",          type=int,   default=0)
    parser.add_argument("--data_root",    type=str,   default=DATA_ROOT)
    parser.add_argument("--out_dir",      type=str,   default=RESULTS_DIR)
    parser.add_argument("--arm",          type=str,   default=None,
                        choices=list(ARMS.keys()),
                        help="Run a single arm (default: all)")
    parser.add_argument("--seed",         type=int,   default=None,
                        help="Run a single seed (default: all N_SEEDS)")
    parser.add_argument("--eps",          type=float, default=EPS)
    parser.add_argument("--analysis_only", action="store_true")
    parser.add_argument("--lira",         action="store_true",
                        help="Run LiRA verification for best arm")
    parser.add_argument("--lira_arm",     type=str,   default="2ch_no_ptr",
                        help="Arm to run LiRA on (default: 2ch_no_ptr)")
    parser.add_argument("--n_shadows",    type=int,   default=8)
    parser.add_argument("--p11_dir",      type=str,   default=P11_DIR,
                        help="Path to P11 exp2 results for vanilla reuse")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[P12] Device: {device}")

    if args.analysis_only:
        _print_summary(args.out_dir)
        _plot_results(args.out_dir)
        return

    if args.lira:
        _run_lira(
            best_arm=args.lira_arm,
            eps=args.eps,
            out_dir=args.out_dir,
            device=device,
            n_shadows=args.n_shadows,
        )
        return

    # Build datasets once
    pub_ds, priv_ds, test_ds, pub_x, pub_y, priv_idx = \
        _build_datasets(args.data_root, seed=42)
    print(f"[P12] Dataset: {len(priv_ds)} private, {len(pub_ds)} public, "
          f"{len(test_ds)} test")

    # Determine which arms/seeds to run
    arms_to_run  = [args.arm] if args.arm else list(ARMS.keys())
    seeds_to_run = [args.seed] if args.seed is not None else list(range(N_SEEDS))

    accs = {}
    for arm_name in arms_to_run:
        accs[arm_name] = []
        for seed in seeds_to_run:
            acc = _train_run(
                arm_name=arm_name,
                eps=args.eps,
                seed=seed,
                priv_ds=priv_ds,
                test_ds=test_ds,
                pub_x=pub_x,
                pub_y=pub_y,
                device=device,
                out_dir=args.out_dir,
                p11_dir=args.p11_dir,
            )
            accs[arm_name].append(acc)
            print(f"[P12] {arm_name} seed={seed}: acc={acc:.4f}")

    _print_summary(args.out_dir)
    _plot_results(args.out_dir)


if __name__ == "__main__":
    main()
