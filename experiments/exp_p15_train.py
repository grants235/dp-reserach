#!/usr/bin/env python3
"""
Phase 15 — Training: Direction-Aware Certificate Sweep
=======================================================

Trains 8 arms with per-step gradient logging. Each arm logs per-example
(grad_norm, incoherent_norm) at every step where the example is sampled.
Logs are used by exp_p15_certify.py for post-hoc certificate computation.

Arms:
  A1  CIFAR-10 balanced,     batch=1000,  q≈0.02,  60 ep, ε=8
  A2  CIFAR-10 balanced,     batch=5000,  q≈0.10,  60 ep, ε=8
  A3  CIFAR-10 balanced,     batch=10000, q≈0.20,  60 ep, ε=8
  B1  CIFAR-10-LT (IR=50),   batch=1000,  q≈0.02,  60 ep, ε=8
  B2  CIFAR-10-LT (IR=50),   batch=5000,  q≈0.10,  60 ep, ε=8
  C1  EMNIST-Letters,        batch=2000,  q≈0.014, 30 ep, ε=8
  C2  EMNIST-Letters,        batch=10000, q≈0.069, 30 ep, ε=8
  C3  EMNIST-Letters,        batch=25000, q≈0.172, 30 ep, ε=8

Priority order (highest absolute tightening): A3, B2, C3, then the rest.

Usage
-----
  # High-priority arms only:
  python experiments/exp_p15_train.py --arm A3 --gpu 0
  python experiments/exp_p15_train.py --arm B2 --gpu 0
  python experiments/exp_p15_train.py --arm C3 --gpu 0

  # All arms:
  python experiments/exp_p15_train.py --gpu 0
"""

import os
import sys
import csv
import math
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets import make_public_private_split, make_cifar10_lt_indices

# ---------------------------------------------------------------------------
# Arm configurations
# ---------------------------------------------------------------------------

ARM_CONFIGS = {
    "A1": dict(dataset="cifar10",    lt_ir=1,   batch_size=1000,  epochs=60,  eps=8.0),
    "A2": dict(dataset="cifar10",    lt_ir=1,   batch_size=5000,  epochs=60,  eps=8.0),
    "A3": dict(dataset="cifar10",    lt_ir=1,   batch_size=10000, epochs=60,  eps=8.0),
    "B1": dict(dataset="cifar10_lt", lt_ir=50,  batch_size=1000,  epochs=60,  eps=8.0),
    "B2": dict(dataset="cifar10_lt", lt_ir=50,  batch_size=5000,  epochs=60,  eps=8.0),
    "C1": dict(dataset="emnist",     lt_ir=1,   batch_size=2000,  epochs=30,  eps=8.0),
    "C2": dict(dataset="emnist",     lt_ir=1,   batch_size=10000, epochs=30,  eps=8.0),
    "C3": dict(dataset="emnist",     lt_ir=1,   batch_size=25000, epochs=30,  eps=8.0),
}

ALL_ARMS = list(ARM_CONFIGS.keys())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_C          = 1.0
RANK_V          = 100
N_PUB           = 2000
LR              = 0.1
MOMENTUM        = 0.9
WEIGHT_DECAY    = 5e-4
PRETRAIN_EPOCHS = 50
PRETRAIN_LR     = 0.01
PUB_BATCH       = 256
GRAD_CHUNK      = 64
DATA_ROOT       = "./data"
RESULTS_DIR     = "./results/exp_p15"

# LT tier boundaries (by class index for 10-class CIFAR-10-LT IR=50)
# Head: classes 0-2, Mid: 3-6, Tail: 7-9
LT_HEAD_CLASSES = {0, 1, 2}
LT_MID_CLASSES  = {3, 4, 5, 6}
LT_TAIL_CLASSES = {7, 8, 9}


def class_to_tier(c):
    if c in LT_HEAD_CLASSES:
        return 0   # head
    elif c in LT_MID_CLASSES:
        return 1   # mid
    else:
        return 2   # tail


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

import torchvision
import torchvision.transforms as T


class _IndexedSubset(torch.utils.data.Dataset):
    """Returns (x, y, global_dataset_index) for per-step membership logging."""
    def __init__(self, base_ds, indices):
        self.base    = base_ds
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, y, int(self.indices[i])


def _cifar10_ds(data_root, augment):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    if augment:
        tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                        T.ToTensor(), T.Normalize(mean, std)])
    else:
        tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return torchvision.datasets.CIFAR10(root=data_root, train=True,
                                        download=True, transform=tf)


def _emnist_ds(data_root, augment):
    """EMNIST-Letters: 26 classes, 124800 train examples, 28x28 grayscale."""
    mean = (0.1722,)
    std  = (0.3309,)
    if augment:
        tf = T.Compose([T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                        T.ToTensor(), T.Normalize(mean, std)])
    else:
        tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    # EMNIST-Letters labels are 1-indexed; we remap to 0-25 below
    return torchvision.datasets.EMNIST(root=data_root, split="letters",
                                       train=True, download=True, transform=tf)


def _emnist_test_ds(data_root):
    mean = (0.1722,)
    std  = (0.3309,)
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return torchvision.datasets.EMNIST(root=data_root, split="letters",
                                       train=False, download=True, transform=tf)


def _build_cifar10(data_root, lt_ir, seed=42):
    full_train_aug   = _cifar10_ds(data_root, augment=True)
    full_train_noaug = _cifar10_ds(data_root, augment=False)
    test_ds          = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True,
        transform=T.Compose([T.ToTensor(),
                             T.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))]))

    full_targets = np.array(full_train_aug.targets)
    all_idx      = np.arange(len(full_train_aug))

    if lt_ir > 1:
        lt_idx = make_cifar10_lt_indices(full_targets, lt_ir, seed=seed)
    else:
        lt_idx = all_idx

    lt_targets = full_targets[lt_idx]
    pub_idx, priv_idx = make_public_private_split(
        lt_idx, lt_targets, public_frac=0.1, seed=seed)

    rng = np.random.default_rng(seed)
    pub_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]

    pub_ds    = Subset(full_train_noaug, pub_use.tolist())
    priv_ds   = _IndexedSubset(full_train_aug, priv_idx)
    num_classes = 10

    # Build class-to-tier mapping for each private example (only for LT)
    if lt_ir > 1:
        priv_targets = full_targets[priv_idx]
        tier_labels  = np.array([class_to_tier(c) for c in priv_targets], dtype=np.int32)
    else:
        tier_labels = None

    pub_x = torch.stack([pub_ds[i][0] for i in range(len(pub_ds))])
    pub_y = torch.tensor([pub_ds[i][1] for i in range(len(pub_ds))], dtype=torch.long)

    return pub_ds, priv_ds, test_ds, pub_x, pub_y, num_classes, priv_idx, tier_labels


def _build_emnist(data_root, seed=42):
    full_train_aug   = _emnist_ds(data_root, augment=True)
    full_train_noaug = _emnist_ds(data_root, augment=False)
    test_ds          = _emnist_test_ds(data_root)

    # EMNIST-Letters labels are 1-indexed (1..26), remap to 0..25
    raw_targets = np.array(full_train_aug.targets) - 1  # now 0-25
    all_idx     = np.arange(len(full_train_aug))
    num_classes = 26

    pub_idx, priv_idx = make_public_private_split(
        all_idx, raw_targets, public_frac=0.1, seed=seed)

    rng = np.random.default_rng(seed)
    pub_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]

    pub_ds  = _EMNISTIndexedSubset(full_train_noaug, pub_use)
    priv_ds = _EMNISTIndexedSubset(full_train_aug, priv_idx)

    pub_x = torch.stack([pub_ds[i][0] for i in range(len(pub_ds))])
    pub_y = torch.tensor([pub_ds[i][1] for i in range(len(pub_ds))], dtype=torch.long)

    # Compute delta = 1 / n^1.1
    n = len(priv_ds)
    delta = 1.0 / (n ** 1.1)

    return pub_ds, priv_ds, test_ds, pub_x, pub_y, num_classes, priv_idx, delta


class _EMNISTIndexedSubset(torch.utils.data.Dataset):
    """Returns (x, y_remapped, global_idx) for EMNIST (remap 1..26 → 0..25)."""
    def __init__(self, base_ds, indices):
        self.base    = base_ds
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, int(y) - 1, int(self.indices[i])  # remap label


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

from src.models import ResNet20


class SimpleCNN(nn.Module):
    """3-layer CNN for EMNIST (28×28 grayscale), GroupNorm compatible."""

    def __init__(self, num_classes=26, n_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1,   32,  3, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(n_groups, 32)
        self.conv2 = nn.Conv2d(32,  64,  3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(n_groups, 64)
        self.conv3 = nn.Conv2d(64,  128, 3, padding=1, bias=False)
        self.gn3   = nn.GroupNorm(n_groups, 128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc    = nn.Linear(128, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))  # 28→14
        x = self.pool(F.relu(self.gn2(self.conv2(x))))  # 14→7
        x = F.relu(self.gn3(self.conv3(x)))              # 7×7
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _make_model(dataset, num_classes):
    if dataset == "emnist":
        return SimpleCNN(num_classes=num_classes, n_groups=8)
    else:
        return ResNet20(num_classes=num_classes, n_groups=16)


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


def _set_grads(model, flat_grad):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat_grad[offset:offset+n].view(p.shape).clone()
        offset += n


# ---------------------------------------------------------------------------
# Per-sample gradients via vmap
# ---------------------------------------------------------------------------

def _loss_fn(params, buffers, x, y, model):
    pred = torch.func.functional_call(model, (params, buffers), x.unsqueeze(0))
    return F.cross_entropy(pred, y.unsqueeze(0))


def _per_sample_grads_chunk(model, x_chunk, y_chunk, device):
    params  = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    grad_fn = torch.func.grad(
        lambda p, b, xi, yi: _loss_fn(p, b, xi, yi, model))
    vmapped = torch.func.vmap(grad_fn, in_dims=(None, None, 0, 0))
    with torch.no_grad():
        g_dict = vmapped(params, buffers,
                         x_chunk.to(device),
                         y_chunk.to(device))
    # Flatten in param order (same as state_dict order)
    flat = torch.cat(
        [g_dict[k].reshape(x_chunk.shape[0], -1)
         for k in params.keys()], dim=1)
    return flat  # [chunk, d]


# ---------------------------------------------------------------------------
# Privacy calibration
# ---------------------------------------------------------------------------

def _calibrate_sigma(eps, delta, q, steps):
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(
        target_epsilon=eps, target_delta=delta,
        sample_rate=q, steps=steps, accountant="rdp")


# ---------------------------------------------------------------------------
# Public pretraining
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
    print(f"  [pretrain] done ({PRETRAIN_EPOCHS} ep)")


# ---------------------------------------------------------------------------
# Subspace (top-RANK_V PCA of public clipped gradients)
# ---------------------------------------------------------------------------

def _compute_subspace(model, pub_x, pub_y, device, rank=RANK_V):
    print(f"  [subspace] Computing rank-{rank} PCA subspace ...")
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
    _, _, V = torch.svd_lowrank(G, q=k, niter=6)
    V = V[:, :k].cpu()
    print(f"  [subspace] V shape: {V.shape}")
    del G; torch.cuda.empty_cache()
    return V  # [d, rank]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(model, test_ds, device, dataset_name):
    loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                        num_workers=4, pin_memory=True)
    model.eval()
    correct = total = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)
        if dataset_name == "emnist":
            y = y - 1  # remap 1-indexed to 0-indexed
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.shape[0]
    return correct / total


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _save_log(log_steps, log_idx, log_gnorm, log_inorm, out_path):
    try:
        import pandas as pd
        df = pd.DataFrame({
            "step":            np.asarray(log_steps,  dtype=np.int32),
            "example_idx":     np.asarray(log_idx,    dtype=np.int32),
            "grad_norm":       np.asarray(log_gnorm,  dtype=np.float32),
            "incoherent_norm": np.asarray(log_inorm,  dtype=np.float32),
            "beta":            (np.asarray(log_inorm, dtype=np.float32)**2 /
                                np.maximum(np.asarray(log_gnorm, dtype=np.float32)**2,
                                           1e-12)),
        })
        df.to_parquet(out_path)
        print(f"  [log] Saved {out_path} ({len(df):,} rows, "
              f"{os.path.getsize(out_path)//1024}KB)")
    except (ImportError, Exception) as e:
        npz_path = out_path.replace(".parquet", ".npz")
        np.savez_compressed(
            npz_path,
            step=np.asarray(log_steps,  dtype=np.int32),
            example_idx=np.asarray(log_idx,   dtype=np.int32),
            grad_norm=np.asarray(log_gnorm, dtype=np.float32),
            incoherent_norm=np.asarray(log_inorm, dtype=np.float32),
        )
        print(f"  [log] Saved (npz fallback): {npz_path}")


# ---------------------------------------------------------------------------
# Training run (single arm)
# ---------------------------------------------------------------------------

def _train_arm(arm_name, cfg, seed, device, data_root, out_dir, log_dir):
    tag       = f"p15_{arm_name}_eps{cfg['eps']:.0f}_seed{seed}"
    csv_path  = os.path.join(out_dir, f"{tag}.csv")
    ckpt_path = os.path.join(out_dir, f"{tag}_final.pt")
    log_path  = os.path.join(log_dir, f"{tag}.parquet")
    meta_path = os.path.join(log_dir, f"{tag}_meta.npz")

    log_npz = log_path.replace(".parquet", ".npz")
    log_done = os.path.exists(log_path) or os.path.exists(log_npz)
    if os.path.exists(ckpt_path) and os.path.exists(csv_path) and log_done:
        print(f"[P15] {tag}: already done, skipping.")
        return

    print(f"\n[P15] === ARM {arm_name} ===")
    print(f"  dataset={cfg['dataset']} lt_ir={cfg['lt_ir']} "
          f"batch={cfg['batch_size']} epochs={cfg['epochs']} eps={cfg['eps']}")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Build dataset
    dataset_type = cfg["dataset"]
    tier_labels  = None
    delta        = 1e-5

    if dataset_type in ("cifar10", "cifar10_lt"):
        pub_ds, priv_ds, test_ds, pub_x, pub_y, num_classes, priv_idx, tier_labels = \
            _build_cifar10(data_root, cfg["lt_ir"], seed=42)
    else:  # emnist
        pub_ds, priv_ds, test_ds, pub_x, pub_y, num_classes, priv_idx, delta = \
            _build_emnist(data_root, seed=42)

    n_priv = len(priv_ds)
    batch_size = cfg["batch_size"]
    epochs     = cfg["epochs"]
    eps        = cfg["eps"]

    print(f"  n_private={n_priv}, num_classes={num_classes}, delta={delta:.2e}")

    # Privacy calibration
    steps_per_epoch = n_priv // batch_size
    T_steps         = epochs * steps_per_epoch
    q               = batch_size / n_priv
    sigma           = _calibrate_sigma(eps, delta, q, T_steps)
    sigma_use       = sigma * CLIP_C  # absolute noise std = sigma * C

    print(f"  T={T_steps}, q={q:.5f}, sigma_mult={sigma:.4f}, "
          f"sigma_use={sigma_use:.4f} (absolute std)")

    # Save meta for certify script
    os.makedirs(log_dir, exist_ok=True)
    np.savez(meta_path,
             n_priv=np.int32(n_priv),
             batch_size=np.int32(batch_size),
             epochs=np.int32(epochs),
             eps=np.float64(eps),
             delta=np.float64(delta),
             q=np.float64(q),
             T_steps=np.int32(T_steps),
             sigma_mult=np.float64(sigma),
             sigma_use=np.float64(sigma_use),
             tier_labels=(tier_labels if tier_labels is not None
                          else np.array([], dtype=np.int32)),
             priv_idx=priv_idx.astype(np.int32))
    print(f"  Meta saved: {meta_path}")

    # Model
    model = _make_model(dataset_type, num_classes).to(device)
    d     = _num_params(model)
    print(f"  Model: {d:,} params")

    # Pretrain on public data
    pub_x = pub_x.to(device)
    pub_y = pub_y.to(device)
    _pretrain_on_public(model, pub_x, pub_y, device)
    pub_x = pub_x.cpu()
    pub_y = pub_y.cpu()

    # Subspace
    V_cpu = _compute_subspace(model, pub_x, pub_y, device, rank=RANK_V)
    V_gpu = V_cpu.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    priv_loader = DataLoader(priv_ds, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)

    os.makedirs(out_dir, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "test_acc", "lr"])
    writer.writeheader()

    log_steps = []
    log_idx   = []
    log_gnorm = []
    log_inorm = []

    best_acc    = 0.0
    step_global = 0

    pub_x_gpu = pub_x.to(device)
    pub_y_gpu = pub_y.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in priv_loader:
            x, y, ex_idx = batch[0], batch[1], batch[2]
            optimizer.zero_grad(set_to_none=True)
            B = x.shape[0]

            sum_g = torch.zeros(d, device=device)

            for ci in range(0, B, GRAD_CHUNK):
                xc    = x[ci:ci+GRAD_CHUNK].to(device)
                yc    = y[ci:ci+GRAD_CHUNK].to(device)
                idx_c = ex_idx[ci:ci+GRAD_CHUNK]

                gc    = _per_sample_grads_chunk(model, xc, yc, device)
                norms = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
                gc_clip = gc * (CLIP_C / norms).clamp(max=1.0)
                sum_g  += gc_clip.sum(0)

                with torch.no_grad():
                    coords   = gc_clip @ V_gpu          # [c, RANK_V]
                    g_V      = coords @ V_gpu.t()       # [c, d]
                    g_perp   = gc_clip - g_V            # [c, d]
                    gnorms_c = gc_clip.norm(dim=1)      # [c]
                    inorms_c = g_perp.norm(dim=1)       # [c]

                gn_cpu = gnorms_c.cpu().numpy().astype(np.float32)
                in_cpu = inorms_c.cpu().numpy().astype(np.float32)
                log_steps.extend([step_global] * len(idx_c))
                log_idx.extend(idx_c.tolist())
                log_gnorm.extend(gn_cpu.tolist())
                log_inorm.extend(in_cpu.tolist())

                del gc, gc_clip, coords, g_V, g_perp, gnorms_c, inorms_c
                torch.cuda.empty_cache()

            noise  = torch.randn(d, device=device) * sigma_use
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

        test_acc = _evaluate(model, test_ds, device, dataset_type)
        cur_lr   = optimizer.param_groups[0]["lr"]

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f"{tag}_best.pt"))

        row = {"epoch": epoch,
               "train_loss": f"{total_loss/max(n_batches,1):.4f}",
               "test_acc":   f"{test_acc:.4f}",
               "lr":         f"{cur_lr:.6f}"}
        writer.writerow(row); csv_file.flush()
        print(f"  ep {epoch:3d}/{epochs}  acc={test_acc:.4f}  "
              f"best={best_acc:.4f}  log_rows={len(log_steps):,}")

        if epoch % 5 == 0:
            _save_log(log_steps, log_idx, log_gnorm, log_inorm, log_path)

    csv_file.close()
    torch.save(model.state_dict(), ckpt_path)
    _save_log(log_steps, log_idx, log_gnorm, log_inorm, log_path)

    del V_gpu, pub_x_gpu, pub_y_gpu
    torch.cuda.empty_cache()
    print(f"[P15] ARM {arm_name} done — final={test_acc:.4f}  best={best_acc:.4f}  "
          f"total_log_rows={len(log_steps):,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm",       type=str, default=None, choices=ALL_ARMS,
                        help="Single arm to run. Default: all arms in priority order.")
    parser.add_argument("--seed",      type=int,   default=0)
    parser.add_argument("--gpu",       type=int,   default=0)
    parser.add_argument("--data_root", type=str,   default=DATA_ROOT)
    parser.add_argument("--out_dir",   type=str,   default=os.path.join(RESULTS_DIR, "train"))
    parser.add_argument("--log_dir",   type=str,   default=os.path.join(RESULTS_DIR, "logs"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[P15] Device: {device}")

    # Priority order: A3, B2, C3, then the rest
    priority_order = ["A3", "B2", "C3", "A2", "A1", "B1", "C2", "C1"]
    arms = [args.arm] if args.arm else priority_order

    for arm in arms:
        cfg = ARM_CONFIGS[arm]
        _train_arm(arm, cfg, args.seed, device, args.data_root,
                   args.out_dir, args.log_dir)


if __name__ == "__main__":
    main()
