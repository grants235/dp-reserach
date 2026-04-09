#!/usr/bin/env python3
"""
Phase 14 — Experiment A: Train Standard DP-SGD Arms with Per-Step Logging
=========================================================================

Trains three arms using STANDARD DP-SGD noise (sigma calibrated for eps=2
unconditionally), logging per-sample gradient norms and incoherent norms
each step.  The mechanisms satisfy (eps=2, delta)-DP by construction.

The question: does the direction-aware post-hoc certificate give tighter
per-instance guarantees than the norm-based one, even for standard DP-SGD?

Arms:
  vanilla_warm_log — Standard DP-SGD + warm start + logging
  gep_log          — GEP-style (V-projected gradient + V-subspace noise) + logging
  pda_dpmd_log     — PDA-DPMD (public grad blending, standard sigma) + logging

For each step t and each sampled example j, logs:
  step, example_idx, grad_norm, incoherent_norm

Usage
-----
  python experiments/exp_posthoc_vanilla.py --arm vanilla_warm_log --gpu 0
  python experiments/exp_posthoc_vanilla.py --arm gep_log --gpu 0
  python experiments/exp_posthoc_vanilla.py --arm pda_dpmd_log --gpu 0
  python experiments/exp_posthoc_vanilla.py --gpu 0   # all arms
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
# Constants — match P11/P13 exactly so splits and calibration are identical
# ---------------------------------------------------------------------------

DELTA           = 1e-5
EPS             = 2.0
CLIP_C          = 1.0
RANK_V          = 100
C_PERP_FRAC     = 0.4        # GEP: clip norm in V_perp = C_PERP_FRAC * CLIP_C
N_PUB           = 2000
EPOCHS          = 60
BATCH_SIZE      = 1000
LR              = 0.1
MOMENTUM        = 0.9
WEIGHT_DECAY    = 5e-4
PRETRAIN_EPOCHS = 50
PRETRAIN_LR     = 0.01
PUB_BATCH       = 256
GRAD_CHUNK      = 64
I_FACTOR        = 8          # PDA-DPMD Amid cosine alpha schedule
DATA_ROOT       = "./data"
RESULTS_DIR     = "./results/exp_p14"
LOG_DIR         = os.path.join(RESULTS_DIR, "logs")
EXP_DIR         = os.path.join(RESULTS_DIR, "exp1")

ARMS = ["vanilla_warm_log", "gep_log", "pda_dpmd_log"]

# ---------------------------------------------------------------------------
# Data helpers (seed=42 everywhere — identical split to P11/P13)
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


class _IndexedSubset(torch.utils.data.Dataset):
    """Returns (x, y, global_dataset_index) for logging batch membership."""
    def __init__(self, base_ds, indices):
        self.base    = base_ds
        self.indices = np.asarray(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, y, int(self.indices[i])


def _build_datasets(data_root, seed=42):
    full_train   = _cifar10(data_root, train=True, augment=False)
    full_targets = np.array(full_train.targets)
    all_idx      = np.arange(len(full_train))
    pub_idx, priv_idx = make_public_private_split(
        all_idx, full_targets, public_frac=0.1, seed=seed)
    rng          = np.random.default_rng(seed)
    pub_idx_use  = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]

    pub_ds      = Subset(full_train, pub_idx_use.tolist())
    priv_ds_aug = _cifar10(data_root, train=True, augment=True)
    priv_ds     = _IndexedSubset(priv_ds_aug, priv_idx)
    test_ds     = _cifar10(data_root, train=False, augment=False)

    pub_x = torch.stack([pub_ds[i][0] for i in range(len(pub_ds))])
    pub_y = torch.tensor([pub_ds[i][1] for i in range(len(pub_ds))],
                         dtype=torch.long)
    return pub_ds, priv_ds, test_ds, pub_x, pub_y


# ---------------------------------------------------------------------------
# Model helpers
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
# Per-sample gradients via vmap (no Opacus GradSampleModule)
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
    return flat   # [chunk, d] on device


# ---------------------------------------------------------------------------
# Privacy calibration
# ---------------------------------------------------------------------------

def _calibrate_sigma(eps, delta, q, steps):
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(
        target_epsilon=eps, target_delta=delta,
        sample_rate=q, steps=steps, accountant="rdp")


# ---------------------------------------------------------------------------
# Public pretraining (warm start)
# ---------------------------------------------------------------------------

def _pretrain_on_public(model, pub_x, pub_y, device, tag):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=PRETRAIN_LR,
                          momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)
    N   = pub_x.shape[0]
    for ep in range(1, PRETRAIN_EPOCHS + 1):
        perm = torch.randperm(N)
        for i in range(0, N, PUB_BATCH):
            idx = perm[i:i+PUB_BATCH]
            opt.zero_grad()
            F.cross_entropy(model(pub_x[idx].to(device)),
                            pub_y[idx].to(device)).backward()
            opt.step()
        sch.step()
    print(f"[P14-A] [{tag}] Pretraining done ({PRETRAIN_EPOCHS} ep)")


# ---------------------------------------------------------------------------
# Coherent subspace V (fixed rank-100 PCA of public clipped gradients)
# ---------------------------------------------------------------------------

def _compute_subspace(model, pub_x, pub_y, device, rank=RANK_V):
    print(f"[P14-A] Computing public PCA subspace (rank={rank})...")
    model.eval()
    parts = []
    for i in range(0, pub_x.shape[0], GRAD_CHUNK):
        g = _per_sample_grads_chunk(model,
                                    pub_x[i:i+GRAD_CHUNK],
                                    pub_y[i:i+GRAD_CHUNK], device)
        norms = g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        parts.append((g * (CLIP_C / norms).clamp(max=1.0)).cpu())
        del g; torch.cuda.empty_cache()
    G = torch.cat(parts, dim=0).float()
    k = min(rank, G.shape[0] - 1)
    _, _, Vt = torch.svd_lowrank(G, q=k, niter=6)
    V = Vt[:, :k].cpu()
    print(f"[P14-A] Subspace V: {V.shape}")
    del G; torch.cuda.empty_cache()
    return V   # [d, rank]


# ---------------------------------------------------------------------------
# Public gradient (for pda_dpmd arm)
# ---------------------------------------------------------------------------

def _pub_grad_flat(model, pub_x, pub_y, device):
    model.eval()
    model.zero_grad()
    N          = pub_x.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    for i in range(0, N, PUB_BATCH):
        xb = pub_x[i:i+PUB_BATCH].to(device)
        yb = pub_y[i:i+PUB_BATCH].to(device)
        total_loss = total_loss + F.cross_entropy(model(xb), yb,
                                                   reduction="sum")
    (total_loss / N).backward()
    flat = torch.cat([p.grad.view(-1) for p in model.parameters()
                      if p.grad is not None]).cpu()
    model.zero_grad()
    return flat


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
        x, y  = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.shape[0]
    return correct / total


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _save_log(log_steps, log_idx, log_gnorm, log_inorm, out_path):
    """Flush accumulated log scalars to parquet (npz fallback)."""
    try:
        import pandas as pd
        df = pd.DataFrame({
            "step":            np.asarray(log_steps, dtype=np.int32),
            "example_idx":     np.asarray(log_idx,   dtype=np.int32),
            "grad_norm":       np.asarray(log_gnorm,  dtype=np.float32),
            "incoherent_norm": np.asarray(log_inorm,  dtype=np.float32),
        })
        df.to_parquet(out_path)
        print(f"[P14-A] Log saved: {out_path} "
              f"({len(df):,} rows, {os.path.getsize(out_path)//1024}KB)")
    except ImportError:
        npz_path = out_path.replace(".parquet", ".npz")
        np.savez_compressed(
            npz_path,
            step=np.asarray(log_steps, dtype=np.int32),
            example_idx=np.asarray(log_idx,   dtype=np.int32),
            grad_norm=np.asarray(log_gnorm,  dtype=np.float32),
            incoherent_norm=np.asarray(log_inorm,  dtype=np.float32),
        )
        print(f"[P14-A] Log saved (npz): {npz_path}")


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------

def _train_run(arm_name, eps, seed, priv_ds, test_ds,
               pub_x, pub_y, device, exp_dir, log_dir):
    tag       = f"{arm_name}_eps{eps:.0f}_seed{seed}"
    csv_path  = os.path.join(exp_dir, f"{tag}.csv")
    ckpt_path = os.path.join(exp_dir, f"{tag}_final.pt")
    log_path  = os.path.join(log_dir, f"{tag}.parquet")

    if os.path.exists(csv_path) and os.path.exists(log_path):
        print(f"[P14-A] {tag}: already done.")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return float(rows[-1]["test_acc"])

    print(f"\n[P14-A] === {tag} ===")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Privacy calibration
    tmp_loader      = DataLoader(priv_ds, batch_size=BATCH_SIZE,
                                 shuffle=True, drop_last=True)
    steps_per_epoch = len(tmp_loader)
    T_steps         = EPOCHS * steps_per_epoch
    q               = BATCH_SIZE / len(priv_ds)
    del tmp_loader

    is_gep = (arm_name == "gep_log")
    is_pda = (arm_name == "pda_dpmd_log")

    # GEP uses sigma calibrated for 2*T to account for two-channel RDP composition
    if is_gep:
        sigma_std = _calibrate_sigma(eps, DELTA, q, 2 * T_steps)
        print(f"[P14-A] GEP sigma_2ch={sigma_std:.4f} "
              f"(calibrated for 2*T={2*T_steps} steps)")
    else:
        sigma_std = _calibrate_sigma(eps, DELTA, q, T_steps)
        print(f"[P14-A] sigma_std={sigma_std:.4f}")

    print(f"[P14-A] T={T_steps}, q={q:.5f}, "
          f"noise_std={sigma_std * CLIP_C:.4f}")

    # Model
    model = _make_model().to(device)
    d     = _num_params(model)
    print(f"[P14-A] Model: {d:,} params")

    # Warm start
    _pretrain_on_public(model, pub_x, pub_y, device, tag)

    # Subspace V
    V_cpu = _compute_subspace(model, pub_x, pub_y, device, rank=RANK_V)
    V_gpu = V_cpu.to(device)   # [d, RANK_V]

    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    priv_loader = DataLoader(priv_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)

    fieldnames = ["epoch", "train_loss", "test_acc", "lr"]
    if is_pda:
        fieldnames += ["alpha_mean", "cos_pub_priv"]
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Log accumulators
    log_steps = []
    log_idx   = []
    log_gnorm = []
    log_inorm = []

    best_acc    = 0.0
    step_global = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        alpha_acc  = []
        cos_acc    = []

        for x, y, ex_idx in priv_loader:
            optimizer.zero_grad(set_to_none=True)
            B = x.shape[0]

            # DP per-sample gradients — accumulated in chunks
            sum_g = torch.zeros(d, device=device)

            for ci in range(0, B, GRAD_CHUNK):
                xc    = x[ci:ci+GRAD_CHUNK].to(device)
                yc    = y[ci:ci+GRAD_CHUNK].to(device)
                idx_c = ex_idx[ci:ci+GRAD_CHUNK]

                gc      = _per_sample_grads_chunk(model, xc, yc, device)  # [c, d]
                norms   = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
                gc_clip = gc * (CLIP_C / norms).clamp(max=1.0)             # [c, d]
                sum_g  += gc_clip.sum(0)

                # Log: incoherent norm = ||g^V⊥|| = ||g - g_V||
                with torch.no_grad():
                    coords  = gc_clip @ V_gpu           # [c, RANK_V]
                    g_V     = coords @ V_gpu.t()        # [c, d]
                    g_perp  = gc_clip - g_V             # [c, d]
                    gnorms  = gc_clip.norm(dim=1)       # [c]
                    inorms  = g_perp.norm(dim=1)        # [c]

                gn_cpu = gnorms.cpu().numpy().astype(np.float32)
                in_cpu = inorms.cpu().numpy().astype(np.float32)
                log_steps.extend([step_global] * len(idx_c))
                log_idx.extend(idx_c.tolist())
                log_gnorm.extend(gn_cpu.tolist())
                log_inorm.extend(in_cpu.tolist())

                del gc, gc_clip, coords, g_V, g_perp, gnorms, inorms
                torch.cuda.empty_cache()

            # --- Compute DP gradient update ---

            if is_gep:
                # GEP: use coherent component only; noise added in V subspace only
                # Privacy: same sigma_std (calibrated with 2*T for two-channel RDP)
                sum_g_V = (sum_g @ V_gpu) @ V_gpu.t()      # [d], V component of sum
                xi_r    = torch.randn(RANK_V, device=device) * sigma_std * CLIP_C
                noise_V = xi_r @ V_gpu.t()                  # [d] noise in V
                # Also add V_perp noise (for the V_perp "channel")
                xi_full  = torch.randn(d, device=device) * sigma_std * CLIP_C * C_PERP_FRAC
                xi_V_proj  = (xi_full @ V_gpu) @ V_gpu.t()
                xi_perp    = xi_full - xi_V_proj
                flat_g = (sum_g_V + noise_V + xi_perp) / B

            elif is_pda:
                # PDA-DPMD: blend private noisy grad with public grad
                noise  = torch.randn(d, device=device) * sigma_std * CLIP_C
                flat_priv = (sum_g + noise) / B
                g_pub  = _pub_grad_flat(model, pub_x, pub_y, device).to(device)
                g_pub_norm  = g_pub.norm().clamp(min=1e-8)
                signal      = sum_g / B
                signal_norm = signal.norm().clamp(min=1e-8)
                cos_acc.append((g_pub @ signal /
                                (g_pub_norm * signal_norm)).item())
                g_pub = g_pub * (flat_priv.norm() / g_pub_norm)
                alpha_t = max(0.0, min(1.0,
                    math.cos(math.pi * step_global / (2.0 * I_FACTOR * T_steps))))
                alpha_acc.append(alpha_t)
                flat_g = alpha_t * flat_priv + (1.0 - alpha_t) * g_pub

            else:
                # Vanilla: standard DP-SGD
                noise  = torch.randn(d, device=device) * sigma_std * CLIP_C
                flat_g = (sum_g + noise) / B

            _set_grads(model, flat_g)
            optimizer.step()

            with torch.no_grad():
                out = model(x[:min(B, 64)].to(device))
                total_loss += F.cross_entropy(
                    out, y[:min(B, 64)].to(device)).item()
            n_batches   += 1
            step_global += 1

        scheduler.step()

        test_acc = _evaluate(model, test_ds, device)
        cur_lr   = optimizer.param_groups[0]["lr"]

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(exp_dir, f"{tag}_best.pt"))

        row = {"epoch": epoch,
               "train_loss": f"{total_loss/n_batches:.4f}",
               "test_acc": f"{test_acc:.4f}",
               "lr": f"{cur_lr:.6f}"}
        if is_pda:
            row["alpha_mean"]   = f"{np.mean(alpha_acc):.4f}" if alpha_acc else "nan"
            row["cos_pub_priv"] = f"{np.mean(cos_acc):.4f}"   if cos_acc   else "nan"
        writer.writerow(row); csv_file.flush()

        print(f"  ep {epoch:3d}/{EPOCHS}  acc={test_acc:.4f}  "
              f"best={best_acc:.4f}  log_rows={len(log_steps):,}")

        # Flush log every 10 epochs to avoid OOM
        if epoch % 10 == 0:
            _save_log(log_steps, log_idx, log_gnorm, log_inorm, log_path)

    csv_file.close()
    torch.save(model.state_dict(), ckpt_path)
    _save_log(log_steps, log_idx, log_gnorm, log_inorm, log_path)

    del V_gpu; torch.cuda.empty_cache()
    print(f"[P14-A] Done — final={test_acc:.4f}  best={best_acc:.4f}  "
          f"total_log_rows={len(log_steps):,}")
    return test_acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm",       type=str,   default=None, choices=ARMS)
    parser.add_argument("--eps",       type=float, default=EPS)
    parser.add_argument("--seed",      type=int,   default=0)
    parser.add_argument("--gpu",       type=int,   default=0)
    parser.add_argument("--data_root", type=str,   default=DATA_ROOT)
    parser.add_argument("--exp_dir",   type=str,   default=EXP_DIR)
    parser.add_argument("--log_dir",   type=str,   default=LOG_DIR)
    args = parser.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[P14-A] Device: {device}")

    pub_ds, priv_ds, test_ds, pub_x, pub_y = _build_datasets(args.data_root)
    print(f"[P14-A] Dataset: {len(priv_ds)} private, {len(pub_ds)} public, "
          f"{len(test_ds)} test")

    arms = [args.arm] if args.arm else ARMS
    for arm in arms:
        acc = _train_run(arm, args.eps, args.seed, priv_ds, test_ds,
                         pub_x, pub_y, device, args.exp_dir, args.log_dir)
        print(f"[P14-A] {arm}: final test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
