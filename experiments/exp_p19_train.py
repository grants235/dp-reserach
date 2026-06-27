#!/usr/bin/env python3
"""
Phase 19 Training: Fixed-Grid Rényi-Filter Direction-Aware DP-SGD
=================================================================

Spec: phase19_spex.md

Key changes from Phase 18:
  1. True Poisson sampling (independent Bernoulli per example, NOT fixed-size batches)
  2. Pre-update timing: Nyström stats computed at θ_t BEFORE the DP update
  3. Every-step accounting (K=1 always)
  4. WRN (F5): no random augmentation for deterministic gradients
  5. Update denominator = B_expected = q*n (fixed, not realized batch size)

Run matrix:
  F1: CLIP ViT-B/32 frozen + linear head, CIFAR-10-LT(50), ε=8, seeds 0,1,2
  F2: CLIP ViT-B/32 frozen + linear head, CIFAR-10,         ε=8, seeds 0,1,2
  F5: WRN-28-2 + GroupNorm, warm-started, CIFAR-10-LT(50), ε=8, seeds 0,1,2
  F6: ResNet-20 + GroupNorm, warm-started, CIFAR-10-LT(10), ε=8, seeds 0,1,2
      [~270k params — still too large for DP at ε=8 to converge]
  F7: MNISTConvNet (~26k params, Tramer-Boneh ICLR2021 SampleConvNet), from-scratch,
      MNIST-LT(10), ε=8, seeds 0,1,2  [Tanh CNN, no GroupNorm needed, B=512, LR=0.6]
  F8: MNISTConvNet, from-scratch, MNIST (balanced), ε=8, seeds 0,1,2  [control for F7]

Usage:
  python experiments/exp_p19_train.py --run F1 --seed 0 --gpu 0
  python experiments/exp_p19_train.py --run F2 --seed 0 --gpu 0
  python experiments/exp_p19_train.py --run F5 --seed 0 --gpu 0
  python experiments/exp_p19_train.py --run F6 --seed 0 --gpu 0   # ResNet-20 LT(10)
  python experiments/exp_p19_train.py --run F7 --seed 0 --gpu 0   # Purchase-100 LT(50)
  python experiments/exp_p19_train.py --run F1 --all_seeds --gpu 0
"""

import os, sys, json, math, argparse, random, time, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets import make_public_private_split, make_cifar10_lt_indices, make_lt_indices, load_purchase100
from src.models import WideResNet, ResNet20, TinyCNN, PurchaseFC, MNISTConvNet

import torchvision
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Run matrix (Section 2 of spec)
# ---------------------------------------------------------------------------

RUNS = {
    "F1": dict(dataset="cifar10_lt50", regime="R3", arch="clip_linear", eps=8.0, B_expected=1400, n_seeds=3, epochs=40),
    "F2": dict(dataset="cifar10",      regime="R3", arch="clip_linear", eps=8.0, B_expected=5000, n_seeds=3, epochs=40),
    "F5": dict(dataset="cifar10_lt50", regime="R2", arch="wrn28-2",     eps=8.0, B_expected=1400, n_seeds=3, epochs=60),
    "F6": dict(dataset="cifar10_lt10", regime="R2", arch="resnet20",    eps=8.0, B_expected=2000, n_seeds=3, epochs=60),
    "F7": dict(dataset="mnist_lt10", regime="R2", arch="dpconv", eps=8.0,
               B_expected=512, n_seeds=3, epochs=40, lr=0.6, r_max=100),
    "F8": dict(dataset="mnist_balanced_lt10", regime="R2", arch="dpconv", eps=8.0,
               B_expected=512, n_seeds=3, epochs=40, lr=0.6, r_max=100),
}

# LiRA targets per run
LIRA_N_TARGETS = {"F1": 1500, "F2": 1000, "F5": 300, "F6": 600, "F7": 600, "F8": 600}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_C      = 1.0        # clipping norm
R_MAX       = 200        # Nyström rank
CHUNK_R3       = 512     # CLIP per-sample grad chunk size
CHUNK_WRN      = 32      # WRN-28-2 per-sample grad chunk size (memory-limited)
CHUNK_RN20     = 128     # ResNet-20 per-sample grad chunk size (~270k params, lighter)
CHUNK_TINY     = 512     # TinyCNN per-sample grad chunk size (~24k params, very light)
CHUNK_PURCHASE = 1024    # PurchaseFC per-sample grad chunk size (linear layers, very light)
CHUNK_DPCONV   = 512     # MNISTConvNet per-sample grad chunk size (~26k params, conv+Tanh)
DATA_ROOT   = "./data"
CACHE_DIR   = "./data/clip_features"
RUNS_DIR    = "./runs/p19"
N_PUB_WRN   = 2000       # public examples for WRN warm-start
PRETRAIN_EP = 50
PRETRAIN_LR = 0.01
LR          = 0.1
N_GROUPS    = 16         # GroupNorm groups for WRN

LT_HEAD = {0, 1, 2}
LT_MID  = {3, 4, 5, 6}
LT_TAIL = {7, 8, 9}

def class_to_tier(c):
    if c in LT_HEAD: return 0
    if c in LT_MID:  return 1
    return 2


def _step_seed(run_id: str, seed: int, t: int) -> int:
    """Stable cross-process seed for Poisson mask at step t."""
    h = hashlib.sha256(f"{run_id}::{seed}::{t}".encode()).hexdigest()[:16]
    return int(h, 16) % (2 ** 32)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class _IndexedSubset(Dataset):
    """Wraps a dataset, yields (x, y, global_idx). No random augmentation."""
    def __init__(self, base, indices):
        self.base = base
        self.indices = np.asarray(indices)

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, y, int(self.indices[i])


class _FeatureDataset(Dataset):
    def __init__(self, feats, labels, global_idx):
        self.feats = feats; self.labels = labels; self.idx = global_idx

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return self.feats[i], int(self.labels[i]), int(self.idx[i])


class _NumpyDataset(Dataset):
    """Wraps numpy (X, y) arrays; yields (x_tensor, y_int, global_idx)."""
    def __init__(self, X, y, global_idx=None):
        self.X   = np.asarray(X, dtype=np.float32)
        self.y   = np.asarray(y, dtype=np.int64)
        self.gidx = (np.arange(len(y), dtype=np.int64)
                     if global_idx is None else np.asarray(global_idx, dtype=np.int64))

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]).float(),
                int(self.y[i]),
                int(self.gidx[i]))


def _cifar10_noaug(data_root, train=True):
    """CIFAR-10 WITHOUT random augmentation (required for F5 deterministic grads)."""
    m, s = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tf = T.Compose([T.ToTensor(), T.Normalize(m, s)])
    return torchvision.datasets.CIFAR10(root=data_root, train=train,
                                        download=True, transform=tf)


def _load_clip_features(data_root, cache_dir, device):
    os.makedirs(cache_dir, exist_ok=True)
    paths = {k: os.path.join(cache_dir, f"cifar10_clip_{k}.pt")
             for k in ["train", "train_labels", "test", "test_labels"]}
    if all(os.path.exists(p) for p in paths.values()):
        print("  [CLIP] Loading cached features")
        return tuple(torch.load(paths[k], map_location="cpu", weights_only=False)
                     for k in ["train", "train_labels", "test", "test_labels"])
    print("  [CLIP] Extracting features...")
    try:
        import open_clip
        cm, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    except ImportError:
        import clip as openai_clip
        cm, _ = openai_clip.load("ViT-B/32", device=device)
    cm = cm.to(device).eval()
    clip_tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                         T.CenterCrop(224), T.ToTensor(),
                         T.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))])
    def _extract(ds):
        fs, ls = [], []
        for b in DataLoader(ds, batch_size=256, shuffle=False, num_workers=4):
            with torch.no_grad(): fs.append(cm.encode_image(b[0].to(device)).cpu().float())
            ls.append(b[1])
        return torch.cat(fs), torch.cat(ls)
    tr_raw = torchvision.datasets.CIFAR10(root=data_root, train=True,  download=True, transform=clip_tf)
    te_raw = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=clip_tf)
    tf_f, tf_l = _extract(tr_raw); te_f, te_l = _extract(te_raw)
    torch.save(tf_f, paths["train"]); torch.save(tf_l, paths["train_labels"])
    torch.save(te_f, paths["test"]);  torch.save(te_l, paths["test_labels"])
    del cm; torch.cuda.empty_cache()
    return tf_f, tf_l, te_f, te_l


def _parse_lt_ir(dataset_name):
    """Extract imbalance ratio from dataset name, e.g. 'cifar10_lt10' → 10."""
    import re
    m = re.search(r'lt(\d+)', dataset_name)
    return int(m.group(1)) if m else 1


def build_dataset_clip(dataset_name, data_root, cache_dir, device):
    """Returns dataset for CLIP linear-probe runs (F1, F2)."""
    is_lt  = "lt" in dataset_name
    lt_ir  = _parse_lt_ir(dataset_name)
    tf_f, tf_l, te_f, te_l = _load_clip_features(data_root, cache_dir, device)
    full_targets = tf_l.numpy(); all_idx = np.arange(len(full_targets))
    lt_idx   = make_cifar10_lt_indices(full_targets, lt_ir, seed=42) if is_lt else all_idx
    lt_targets = full_targets[lt_idx]
    pub_idx, priv_idx = make_public_private_split(lt_idx, lt_targets, public_frac=0.1, seed=42)
    priv_feats  = tf_f[priv_idx]
    priv_labels = torch.tensor(full_targets[priv_idx], dtype=torch.long)
    tier_labels = (np.array([class_to_tier(c) for c in full_targets[priv_idx]], dtype=np.int32)
                   if is_lt else None)
    test_ds = _FeatureDataset(te_f, te_l, torch.arange(len(te_l)))
    class_counts = np.bincount(full_targets[priv_idx], minlength=10)
    return (priv_feats, priv_labels, priv_idx, tier_labels, 10, 1e-5,
            class_counts, te_f, te_l, pub_idx, full_targets)


def build_dataset_wrn(dataset_name, data_root):
    """Returns dataset for WRN runs (F5, F6). NO random augmentation."""
    is_lt = "lt" in dataset_name
    lt_ir = _parse_lt_ir(dataset_name)
    # Use no-augmentation dataset for both private training and accounting
    train_noaug = _cifar10_noaug(data_root, train=True)
    test_noaug  = _cifar10_noaug(data_root, train=False)
    full_targets = np.array(train_noaug.targets)
    lt_idx = make_cifar10_lt_indices(full_targets, lt_ir, seed=42) if is_lt else np.arange(len(train_noaug))
    lt_targets = full_targets[lt_idx]
    pub_idx, priv_idx = make_public_private_split(lt_idx, lt_targets, public_frac=0.1, seed=42)
    # Warm-start: use at most N_PUB_WRN public examples
    rng_pub = np.random.default_rng(42)
    pub_use = pub_idx[rng_pub.permutation(len(pub_idx))[:N_PUB_WRN]]
    priv_ds   = _IndexedSubset(train_noaug, priv_idx)  # no augmentation
    pub_x = torch.stack([train_noaug[int(i)][0] for i in pub_use])
    pub_y = torch.tensor([int(full_targets[i]) for i in pub_use], dtype=torch.long)
    tier_labels = (np.array([class_to_tier(c) for c in full_targets[priv_idx]], dtype=np.int32)
                   if is_lt else None)
    class_counts = np.bincount(full_targets[priv_idx], minlength=10)
    priv_labels_np = full_targets[priv_idx].astype(np.int32)
    test_labels_np = np.array(test_noaug.targets)
    return (priv_ds, priv_idx, priv_labels_np, tier_labels, 10, 1e-5,
            class_counts, pub_x, pub_y, pub_idx, test_noaug, test_labels_np)


def _fmnist_noaug(data_root, train=True):
    """FashionMNIST without augmentation (deterministic grads required)."""
    tf = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
    return torchvision.datasets.FashionMNIST(root=data_root, train=train,
                                              download=True, transform=tf)


def build_dataset_fmnist(dataset_name, data_root):
    """Returns dataset for TinyCNN FashionMNIST runs (F7). No augmentation."""
    is_lt = "lt" in dataset_name
    lt_ir = _parse_lt_ir(dataset_name)
    train_ds    = _fmnist_noaug(data_root, train=True)
    test_ds     = _fmnist_noaug(data_root, train=False)
    full_targets = np.array(train_ds.targets)
    lt_idx = make_lt_indices(full_targets, lt_ir, seed=42) if is_lt else np.arange(len(train_ds))
    lt_targets = full_targets[lt_idx]
    pub_idx, priv_idx = make_public_private_split(lt_idx, lt_targets, public_frac=0.1, seed=42)
    rng_pub = np.random.default_rng(42)
    pub_use = pub_idx[rng_pub.permutation(len(pub_idx))[:N_PUB_WRN]]
    priv_ds = _IndexedSubset(train_ds, priv_idx)
    pub_x = torch.stack([train_ds[int(i)][0] for i in pub_use])
    pub_y = torch.tensor([int(full_targets[i]) for i in pub_use], dtype=torch.long)
    tier_labels = (np.array([class_to_tier(c) for c in full_targets[priv_idx]], dtype=np.int32)
                   if is_lt else None)
    class_counts = np.bincount(full_targets[priv_idx], minlength=10)
    priv_labels_np = full_targets[priv_idx].astype(np.int32)
    test_labels_np = np.array(test_ds.targets)
    return (priv_ds, priv_idx, priv_labels_np, tier_labels, 10, 1e-5,
            class_counts, pub_x, pub_y, pub_idx, test_ds, test_labels_np)


def _mnist_noaug(data_root, train=True):
    """MNIST without augmentation (deterministic grads required for accounting)."""
    tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    return torchvision.datasets.MNIST(root=data_root, train=train,
                                      download=True, transform=tf)


def build_dataset_mnist(dataset_name, data_root):
    """Returns dataset for MNISTConvNet runs (F7/F8).

    dataset_name variants:
      "mnist_lt<IR>"           — long-tailed MNIST with imbalance ratio IR  (F7)
      "mnist_balanced_lt<IR>"  — balanced MNIST subsampled to the same total
                                  size as the LT(IR) variant (F8)
      "mnist"                  — full balanced MNIST (~54k private)

    No augmentation (Tanh CNN, from-scratch). pub_x=None → warm-start skipped.
    Returns same 12-tuple format as build_dataset_wrn.
    """
    is_balanced_lt = "balanced_lt" in dataset_name
    is_lt          = ("lt" in dataset_name) and not is_balanced_lt
    lt_ir          = _parse_lt_ir(dataset_name) if (is_lt or is_balanced_lt) else 1

    train_ds     = _mnist_noaug(data_root, train=True)
    test_ds      = _mnist_noaug(data_root, train=False)
    full_targets = np.array(train_ds.targets)

    if is_lt:
        lt_idx = make_lt_indices(full_targets, lt_ir, num_classes=10, seed=42)
    elif is_balanced_lt:
        # Subsample each class equally to match the total size of LT(lt_ir).
        lt_idx_ref  = make_lt_indices(full_targets, lt_ir, num_classes=10, seed=42)
        n_per_class = len(lt_idx_ref) // 10
        rng = np.random.default_rng(42)
        parts = []
        for c in range(10):
            cls_idx = np.where(full_targets == c)[0]
            chosen  = rng.choice(cls_idx, size=min(n_per_class, len(cls_idx)), replace=False)
            parts.append(chosen)
        lt_idx = np.sort(np.concatenate(parts))
    else:
        lt_idx = np.arange(len(train_ds))

    lt_targets = full_targets[lt_idx]
    pub_idx, priv_idx = make_public_private_split(lt_idx, lt_targets, public_frac=0.1, seed=42)
    priv_ds = _IndexedSubset(train_ds, priv_idx)

    # Always assign tier labels so per-tier analysis works for both LT and balanced runs.
    tier_labels = np.array([class_to_tier(c) for c in full_targets[priv_idx]], dtype=np.int32)

    class_counts   = np.bincount(full_targets[priv_idx], minlength=10)
    priv_labels_np = full_targets[priv_idx].astype(np.int32)
    test_labels_np = np.array(test_ds.targets)
    print(f"  [MNIST] n_priv={len(priv_idx)}  n_pub={len(pub_idx)}  "
          f"n_test={len(test_ds)}  lt_ir={lt_ir}  balanced_lt={is_balanced_lt}")
    # pub_x=None → warm-start skipped automatically; pub_idx retained for metadata
    return (priv_ds, priv_idx, priv_labels_np, tier_labels, 10, 1e-5,
            class_counts, None, None, pub_idx, test_ds, test_labels_np)


def build_dataset_purchase(dataset_name, data_root):
    """
    Returns dataset for PurchaseFC runs (F7). No augmentation (tabular data).

    Workflow:
      1. Load Purchase-100 (197k × 600 binary features, 100 classes)
      2. Stratified 80/20 train/test split (seed=42)
      3. LT subsampling on train split (using make_lt_indices, generic 100-class version)
      4. 10% public / 90% private split
      5. Tier labels: head/mid/tail by class-count rank (top/mid/bottom third)

    Returns (priv_ds, priv_idx, priv_labels_np, tier_labels, num_classes, delta,
             class_counts, test_ds, test_labels_np).  No pub_x/pub_y — no warm-start.
    """
    is_lt = "lt" in dataset_name
    lt_ir = _parse_lt_ir(dataset_name) if is_lt else 1
    num_classes = 100
    delta = 1e-5

    X_all, y_all = load_purchase100(data_root)
    N = len(y_all)

    # Stratified 80/20 train/test split
    rng_split = np.random.default_rng(42)
    train_mask = np.zeros(N, dtype=bool)
    for c in range(num_classes):
        cls_idx = np.where(y_all == c)[0]
        n_train_c = max(1, int(len(cls_idx) * 0.8))
        chosen = rng_split.choice(cls_idx, size=n_train_c, replace=False)
        train_mask[chosen] = True
    train_idx = np.where(train_mask)[0]
    test_idx  = np.where(~train_mask)[0]

    # LT subsampling on training split
    y_train = y_all[train_idx]
    if is_lt and lt_ir > 1:
        lt_local = make_lt_indices(y_train, float(lt_ir), num_classes=num_classes, seed=42)
        lt_train_idx = train_idx[lt_local]
    else:
        lt_train_idx = train_idx

    # Pub/priv split (10% public, 90% private)
    y_lt = y_all[lt_train_idx]
    pub_local, priv_local = make_public_private_split(
        np.arange(len(lt_train_idx)), y_lt, public_frac=0.1, seed=42)
    priv_global = lt_train_idx[priv_local]
    pub_global  = lt_train_idx[pub_local]

    priv_labels_np = y_all[priv_global].astype(np.int32)
    class_counts   = np.bincount(priv_labels_np, minlength=num_classes)

    # Tier labels: head / mid / tail by class frequency rank
    sorted_by_count = np.argsort(class_counts)[::-1]  # most frequent first
    tier_map = np.zeros(num_classes, dtype=np.int32)
    n3 = num_classes // 3
    for rank, c in enumerate(sorted_by_count):
        tier_map[c] = 0 if rank < n3 else (1 if rank < 2 * n3 else 2)
    tier_labels = tier_map[priv_labels_np]

    priv_ds    = _NumpyDataset(X_all[priv_global], y_all[priv_global], global_idx=priv_global)
    test_ds    = _NumpyDataset(X_all[test_idx],    y_all[test_idx],    global_idx=test_idx)
    test_labels_np = y_all[test_idx].astype(np.int32)

    print(f"  [Purchase-100] n_priv={len(priv_global)}  n_pub={len(pub_global)}  "
          f"n_test={len(test_idx)}  lt_ir={lt_ir}")
    return (priv_ds, priv_global.astype(np.int32), priv_labels_np,
            tier_labels, num_classes, delta, class_counts,
            test_ds, test_labels_np)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, num_classes=10, feat_dim=512):
        super().__init__(); self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x): return self.fc(x.float())


def make_model(regime, num_classes, arch="wrn28-2"):
    if regime == "R3":
        return LinearHead(num_classes, feat_dim=512)
    if arch == "resnet20":
        return ResNet20(num_classes=num_classes, n_groups=N_GROUPS)
    if arch == "tinycnn":
        return TinyCNN(num_classes=num_classes)
    if arch == "purchase_fc":
        return PurchaseFC(num_classes=num_classes)
    if arch == "dpconv":
        return MNISTConvNet(num_classes=num_classes)
    return WideResNet(depth=28, widen_factor=2, num_classes=num_classes, n_groups=N_GROUPS)


# ---------------------------------------------------------------------------
# Per-sample gradient computation
# ---------------------------------------------------------------------------

def _loss_fn(params, buffers, x, y, model):
    pred = torch.func.functional_call(model, (params, buffers), x.unsqueeze(0))
    return F.cross_entropy(pred, y.unsqueeze(0))


def _per_sample_grads_vmap(model, x_chunk, y_chunk, device):
    params  = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    grad_fn = torch.func.grad(lambda p, b, xi, yi: _loss_fn(p, b, xi, yi, model))
    vmapped = torch.func.vmap(grad_fn, in_dims=(None, None, 0, 0))
    with torch.no_grad():
        g_dict = vmapped(params, buffers, x_chunk.to(device), y_chunk.to(device))
    return torch.cat([g_dict[k].reshape(x_chunk.shape[0], -1) for k in params], dim=1)


def _per_sample_grads_linear(model, h_chunk, y_chunk, device):
    """Analytic per-sample gradient for linear head: O(n*d*C) time."""
    h = h_chunk.to(device).float(); y = y_chunk.to(device).long()
    W = model.fc.weight; b = model.fc.bias
    with torch.no_grad():
        logits = h @ W.t() + b
        p = torch.softmax(logits, dim=1)
        ey = torch.zeros_like(p); ey.scatter_(1, y.unsqueeze(1), 1.0)
        delta = p - ey          # (bs, C)
        g_W = (delta.unsqueeze(2) * h.unsqueeze(1)).reshape(h.shape[0], -1)
    return torch.cat([g_W, delta], dim=1)   # (bs, d)


def _set_grads(model, flat_g):
    """Write flat gradient vector into model.grad."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat_g[offset:offset + n].view(p.shape).clone()
        offset += n


# ---------------------------------------------------------------------------
# Nyström sufficient statistics (CLIP, R3)
# ---------------------------------------------------------------------------

def nystrom_stats_r3(model, priv_feats, priv_labels, rho, r_max, device):
    """
    Compute Nyström stats at current θ_t for CLIP linear head.

    Returns (G_cpu, norms_t, losses_t, B_t, M_t, Y_proj_t) where:
      G_cpu    : (n, d) clipped gradients on CPU float32
      norms_t  : (n,)   clipped norms
      losses_t : (n,)   cross-entropy losses
      B_t      : (r_max, r_max) = Q_t^T Σ_t Q_t
      M_t      : (r_max, r_max) = Y_t^T Y_t
      Y_proj_t : (n, r_max)    = Y_t^T ḡ_{i,t}  (spec §7)
    """
    n = len(priv_feats)
    model.eval()
    G_parts, norm_parts, loss_parts = [], [], []

    for i in range(0, n, CHUNK_R3):
        h = priv_feats[i:i + CHUNK_R3].to(device).float()
        y = priv_labels[i:i + CHUNK_R3].to(device).long()
        with torch.no_grad():
            gc   = _per_sample_grads_linear(model, h, y, device)
            nms  = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
            gc_c = gc * (CLIP_C / nms).clamp(max=1.0)
            G_parts.append(gc_c.cpu().float())
            norm_parts.append(gc_c.norm(dim=1).cpu())
            loss_parts.append(F.cross_entropy(model(h), y, reduction='none').cpu())
        del gc, gc_c, h, y; torch.cuda.empty_cache()

    G        = torch.cat(G_parts, 0)           # (n, d) CPU
    norms_t  = torch.cat(norm_parts, 0)        # (n,)
    losses_t = torch.cat(loss_parts, 0)        # (n,)
    del G_parts, norm_parts, loss_parts

    # Use float64 for the eigensolve: CLIP d is small (~5130), so cost is trivial
    G_dev = G.to(device=device, dtype=torch.float64)
    # Σ_t = ρ G^T G   (d × d)
    Sigma = rho * (G_dev.T @ G_dev)
    # Top-r_max eigenvectors (exact since d is small for CLIP)
    eigvals, eigvecs = torch.linalg.eigh(Sigma)
    Q_t = eigvecs[:, -r_max:].flip(1)          # (d, r_max) float64
    # Y_t = Σ_t Q_t   (d, r_max)
    Y_t = Sigma @ Q_t
    B_t      = (Q_t.T @ Y_t).cpu().double()    # (r_max, r_max) float64
    M_t      = (Y_t.T @ Y_t).cpu().double()    # (r_max, r_max) float64
    Y_proj_t = (G_dev @ Y_t).cpu().float()     # (n, r_max) float32

    del Sigma, eigvals, eigvecs, Q_t, Y_t, G_dev; torch.cuda.empty_cache()
    return G.cpu(), norms_t, losses_t, B_t, M_t, Y_proj_t


# ---------------------------------------------------------------------------
# Nyström sufficient statistics (WRN, R2) — 4-pass rSVD + streaming
# ---------------------------------------------------------------------------

def nystrom_stats_wrn(model, priv_ds, rho, r_max, device, chunk_size=CHUNK_WRN):
    """
    Compute Nyström stats for WRN/ResNet-20 via 4-pass randomized SVD.

    No random augmentation — priv_ds must use deterministic transforms.

    Returns (norms_t, losses_t, B_t, M_t, Y_proj_t, top_eigvals, cond_num).
    Does NOT return full G (too large). DP update is computed in a separate pass.
    top_eigvals: (top_r,) descending eigenvalues of ρ G^T G (for §12.4 PSD check).
    cond_num: condition number λ_max / λ_min of the sketch, for diagnostics.
    """
    model.eval()
    N = len(priv_ds)
    d = sum(p.numel() for p in model.parameters())
    loader = DataLoader(priv_ds, batch_size=chunk_size, shuffle=False,
                        num_workers=2, pin_memory=True, drop_last=False)
    k = r_max + 20

    # --- Pass 1: rSVD sketch + norms + losses ---
    rng_sk = torch.Generator(device=device); rng_sk.manual_seed(99999)
    Omega = torch.randn(d, k, generator=rng_sk, device=device)
    Y_sketch_parts, norm_parts, loss_parts = [], [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        with torch.no_grad():
            gc  = _per_sample_grads_vmap(model, x, y, device)
            nms = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
            gc_c = gc * (CLIP_C / nms).clamp(max=1.0)
            Y_sketch_parts.append((gc_c @ Omega).cpu())
            norm_parts.append(gc_c.norm(dim=1).cpu())
            loss_parts.append(
                F.cross_entropy(model(x.to(device).float()), y.to(device).long(),
                                reduction='none').cpu())
        del gc, gc_c; torch.cuda.empty_cache()
    Y_sketch = torch.cat(Y_sketch_parts, 0)     # (N, k)
    norms_t  = torch.cat(norm_parts, 0)
    losses_t = torch.cat(loss_parts, 0)
    Q_svd, _ = torch.linalg.qr(Y_sketch)        # (N, k)
    del Y_sketch, Y_sketch_parts, Omega; torch.cuda.empty_cache()

    # --- Pass 2: B = Q_svd^T G → rSVD eigenpairs ---
    Bmat = torch.zeros(k, d, device=device, dtype=torch.float32)
    row = 0
    for batch in loader:
        x, y = batch[0], batch[1]; bs = x.shape[0]
        with torch.no_grad():
            gc   = _per_sample_grads_vmap(model, x, y, device)
            nms  = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
            gc_c = gc * (CLIP_C / nms).clamp(max=1.0)
            Q_row = Q_svd[row:row + bs].to(device)
            Bmat.addmm_(Q_row.T, gc_c)
        row += bs; del gc, gc_c, Q_row; torch.cuda.empty_cache()
    del Q_svd

    with torch.no_grad():
        BBT = Bmat @ Bmat.T
        eigvals_b, U_B = torch.linalg.eigh(BBT)
        top_r = min(r_max, int((eigvals_b > 0).sum().item()))
        eigvals_r = eigvals_b[-top_r:].flip(0).clamp(min=0.0)   # descending
        # Diagnostic eigenvalues (§12.4 PSD check)
        top_eigvals = (rho * eigvals_r).cpu().numpy().astype(np.float32)
        lam_max = float(eigvals_r[0].item()) if top_r > 0 else 0.0
        lam_min = float(eigvals_r[top_r - 1].item()) if top_r > 0 else 0.0
        cond_num = float(lam_max / lam_min) if lam_min > 1e-30 else float('inf')
        U_B_r = U_B[:, -top_r:].flip(1)
        V_r = (Bmat.T @ U_B_r) / eigvals_r.sqrt().clamp(min=1e-12)  # (d, top_r)
    del Bmat, BBT, U_B; torch.cuda.empty_cache()
    Q_t = V_r; actual_r = Q_t.shape[1]

    # --- Pass 3: Y_t = ρ G^T (G Q_t), P_buf = G Q_t ---
    P_buf = torch.zeros(N, actual_r, dtype=torch.float32)  # CPU
    Y_t   = torch.zeros(d, actual_r, device=device, dtype=torch.float32)
    row = 0
    for batch in loader:
        x, y = batch[0], batch[1]; bs = x.shape[0]
        with torch.no_grad():
            gc   = _per_sample_grads_vmap(model, x, y, device)
            nms  = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
            gc_c = gc * (CLIP_C / nms).clamp(max=1.0)
            P_chunk = gc_c @ Q_t                          # (bs, actual_r)
            P_buf[row:row + bs] = P_chunk.cpu()
            Y_t.addmm_(gc_c.T, P_chunk, alpha=rho)       # Y_t += ρ gc_c^T P_chunk
        row += bs; del gc, gc_c, P_chunk; torch.cuda.empty_cache()
    P_buf_dev = P_buf.to(device)
    # B_t = ρ P^T P = ρ (G Q_t)^T (G Q_t) = Q_t^T (ρ G^T G) Q_t = Q_t^T Σ_t Q_t
    B_t = (rho * P_buf_dev.T @ P_buf_dev).cpu()          # (actual_r, actual_r)
    M_t = (Y_t.T @ Y_t).cpu()                            # (actual_r, actual_r)
    del P_buf_dev

    # Pad to r_max × r_max if needed
    if actual_r < r_max:
        B_pad = torch.zeros(r_max, r_max); B_pad[:actual_r, :actual_r] = B_t; B_t = B_pad
        M_pad = torch.zeros(r_max, r_max); M_pad[:actual_r, :actual_r] = M_t; M_t = M_pad

    # --- Pass 4: Y_proj[i] = Y_t^T ḡ_i ---
    Y_proj_parts = []
    for batch in loader:
        x, y = batch[0], batch[1]
        with torch.no_grad():
            gc   = _per_sample_grads_vmap(model, x, y, device)
            nms  = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
            gc_c = gc * (CLIP_C / nms).clamp(max=1.0)
            yp   = gc_c @ Y_t                            # (bs, actual_r)
        Y_proj_parts.append(yp.cpu()); del gc, gc_c, yp; torch.cuda.empty_cache()
    Y_proj_t = torch.cat(Y_proj_parts, 0)               # (N, actual_r)
    if actual_r < r_max:
        tmp = torch.zeros(N, r_max); tmp[:, :actual_r] = Y_proj_t; Y_proj_t = tmp

    del Q_t, V_r, Y_t; torch.cuda.empty_cache()
    return norms_t, losses_t, B_t, M_t, Y_proj_t, top_eigvals, cond_num


# ---------------------------------------------------------------------------
# Compute G_sum for Poisson-selected examples (WRN pass 5)
# ---------------------------------------------------------------------------

def compute_poisson_G_sum_wrn(model, priv_ds, inclusion_mask, device, chunk_size=CHUNK_WRN):
    """
    Compute G_sum = Σ_{i: I_i=1} ḡ_{i,t}.

    Uses the FULL dataset loader (batch_size=chunk_size, shuffle=False) — the
    same configuration as nystrom_stats_wrn passes 1–4.  Per-sample gradients
    are computed over full batches and only the included rows are summed.
    Because per-sample vmap has no cross-example state, the clipped gradient
    for example i is bitwise-identical to the one computed in the accounting
    pass regardless of which other examples share the same batch, satisfying
    G_t^{update} ≡ G_t^{accounting} exactly (spec §8.2 soundness requirement).
    """
    loader = DataLoader(priv_ds, batch_size=chunk_size, shuffle=False,
                        num_workers=2, pin_memory=True, drop_last=False)
    d     = sum(p.numel() for p in model.parameters())
    G_sum = torch.zeros(d, device=device)
    model.eval()  # GroupNorm: eval == train for forward; no running-stat drift
    row = 0
    for batch in loader:
        x, y = batch[0], batch[1]; bs = x.shape[0]
        incl_batch = inclusion_mask[row:row + bs]
        if incl_batch.any():
            with torch.no_grad():
                gc   = _per_sample_grads_vmap(model, x, y, device)  # full batch
                nms  = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
                gc_c = gc * (CLIP_C / nms).clamp(max=1.0)
                G_sum += gc_c[incl_batch].sum(0)
            del gc, gc_c; torch.cuda.empty_cache()
        row += bs
    return G_sum


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------

def calibrate_sigma(eps, delta, q, T_steps, accountant="prv"):
    from opacus.accountants.utils import get_noise_multiplier
    try:
        return get_noise_multiplier(target_epsilon=float(eps), target_delta=float(delta),
                                    sample_rate=float(q), steps=int(T_steps),
                                    accountant=accountant)
    except Exception:
        return get_noise_multiplier(target_epsilon=float(eps), target_delta=float(delta),
                                    sample_rate=float(q), steps=int(T_steps),
                                    accountant="rdp")


def pretrain_wrn(model, pub_x, pub_y, device):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=PRETRAIN_LR, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EP)
    N = pub_x.shape[0]
    for _ep in range(PRETRAIN_EP):
        perm = torch.randperm(N)
        for i in range(0, N, 256):
            idx = perm[i:i + 256]; opt.zero_grad()
            F.cross_entropy(model(pub_x[idx].to(device)), pub_y[idx].to(device)).backward()
            opt.step()
        sch.step()
    print(f"  [pretrain] {PRETRAIN_EP} epochs done")


@torch.no_grad()
def evaluate_clip(model, te_feats, te_labels, device):
    model.eval()
    logits = model(te_feats.to(device).float())
    acc = (logits.argmax(1) == te_labels.to(device)).float().mean().item()
    return acc


@torch.no_grad()
def evaluate_wrn(model, test_ds, device):
    model.eval()
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)
    correct = total = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        correct += (model(x.to(device)).argmax(1) == y.to(device)).sum().item()
        total   += y.shape[0]
    return correct / total


# ---------------------------------------------------------------------------
# LiRA target selection
# ---------------------------------------------------------------------------

def select_lira_targets(priv_idx, priv_labels_np, tier_labels,
                        test_labels_np, n_targets, seed=1000):
    rng = np.random.default_rng(seed)
    if tier_labels is not None:
        n_per_tier = n_targets // 3
        member_idx = []
        for tier in [0, 1, 2]:
            mask = (tier_labels == tier)
            avail = np.where(mask)[0]
            n_sel = min(n_per_tier, len(avail))
            chosen = rng.choice(avail, size=n_sel, replace=False)
            member_idx.append(chosen)
        total = sum(len(m) for m in member_idx)
        if total < n_targets:
            remaining = n_targets - total
            all_used = np.concatenate(member_idx)
            all_avail = np.setdiff1d(np.arange(len(priv_idx)), all_used)
            extra = rng.choice(all_avail, size=min(remaining, len(all_avail)), replace=False)
            member_idx.append(extra)
        member_local = np.concatenate(member_idx)
    else:
        unique_cls = np.unique(priv_labels_np)
        n_per_class = n_targets // max(1, len(unique_cls))
        member_idx = []
        for c in unique_cls:
            mask  = (priv_labels_np == c)
            avail = np.where(mask)[0]
            n_sel = min(n_per_class, len(avail))
            chosen = rng.choice(avail, size=n_sel, replace=False)
            member_idx.append(chosen)
        member_local = np.concatenate(member_idx)

    n_nm = len(member_local)
    test_labels_np = np.asarray(test_labels_np)
    unique_test_cls = np.unique(test_labels_np)
    n_per_cls_nm    = max(1, n_nm // len(unique_test_cls))
    nm_idx = []
    for c in unique_test_cls:
        mask  = (test_labels_np == c)
        avail = np.where(mask)[0]
        n_sel = min(n_per_cls_nm, len(avail))
        chosen = rng.choice(avail, size=n_sel, replace=False)
        nm_idx.append(chosen)
    nonmember_test = np.concatenate(nm_idx)
    return member_local.astype(np.int32), nonmember_test.astype(np.int32)


# ---------------------------------------------------------------------------
# Memmap helper
# ---------------------------------------------------------------------------

def _open_mm(path, dtype, shape):
    """Open a numpy memmap in r+ (resume) or w+ (fresh) mode."""
    mode = 'r+' if os.path.exists(path) else 'w+'
    return np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_run(run_id, cfg, seed, device, data_root, cache_dir, runs_dir, max_steps=None):
    regime  = cfg["regime"]
    dataset = cfg["dataset"]
    arch    = cfg.get("arch", "wrn28-2")
    eps     = cfg["eps"]
    B_exp   = cfg["B_expected"]
    epochs  = cfg["epochs"]
    is_clip    = (regime == "R3")
    is_purchase = "purchase" in dataset
    is_mnist   = "mnist" in dataset and "fmnist" not in dataset
    chunk_size = (CHUNK_PURCHASE if arch == "purchase_fc" else
                  CHUNK_DPCONV  if arch == "dpconv"      else
                  CHUNK_TINY    if arch == "tinycnn"     else
                  CHUNK_RN20    if arch == "resnet20"    else CHUNK_WRN)
    lr = cfg.get("lr", LR)
    r_max = cfg.get("r_max", R_MAX)

    run_dir = os.path.join(runs_dir, run_id, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    final_path = os.path.join(run_dir, "model_final.pt")
    meta_path  = os.path.join(run_dir, "metadata.json")
    ckpt_path  = os.path.join(run_dir, "checkpoint_latest.pt")

    if os.path.exists(final_path) and os.path.exists(meta_path):
        print(f"[P19] {run_id}/seed_{seed}: already done, skipping.")
        return

    resuming = os.path.exists(ckpt_path)

    print(f"\n[P19] === {run_id} seed={seed} {'(RESUMING)' if resuming else ''} ===")
    print(f"  dataset={dataset}  regime={regime}  eps={eps}  B_expected={B_exp}")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # --- Build dataset ---
    tier_labels = None; test_labels_np = None
    if is_clip:
        (priv_feats, priv_labels, priv_idx, tier_labels, num_classes, delta,
         class_counts, te_feats, te_labels, pub_idx, _) = \
            build_dataset_clip(dataset, data_root, cache_dir, device)
        test_labels_np = te_labels.numpy()
        priv_labels_np = priv_labels.numpy().astype(np.int32)
        n_priv = len(priv_idx)
    elif is_purchase:
        (priv_ds, priv_idx, priv_labels_np, tier_labels, num_classes, delta,
         class_counts, test_ds, test_labels_np) = \
            build_dataset_purchase(dataset, data_root)
        pub_x = pub_y = None
        pub_idx = np.array([], dtype=np.int32)
        n_priv = len(priv_idx)
    elif is_mnist:
        (priv_ds, priv_idx, priv_labels_np, tier_labels, num_classes, delta,
         class_counts, pub_x, pub_y, pub_idx_wrn, test_ds, test_labels_np) = \
            build_dataset_mnist(dataset, data_root)
        n_priv = len(priv_idx)
        pub_idx = pub_idx_wrn
    elif "fmnist" in dataset:
        (priv_ds, priv_idx, priv_labels_np, tier_labels, num_classes, delta,
         class_counts, pub_x, pub_y, pub_idx_wrn, test_ds, test_labels_np) = \
            build_dataset_fmnist(dataset, data_root)
        n_priv = len(priv_idx)
        pub_idx = pub_idx_wrn
    else:
        (priv_ds, priv_idx, priv_labels_np, tier_labels, num_classes, delta,
         class_counts, pub_x, pub_y, pub_idx_wrn, test_ds, test_labels_np) = \
            build_dataset_wrn(dataset, data_root)
        n_priv = len(priv_idx)
        pub_idx = pub_idx_wrn

    # Compute q, T, σ
    q            = B_exp / n_priv
    steps_per_ep = max(1, round(n_priv / B_exp))
    T_train      = epochs * steps_per_ep
    rho          = q * (1.0 - q)
    sigma        = calibrate_sigma(eps, delta, q, T_train)
    a            = (sigma * CLIP_C) ** 2

    print(f"  n={n_priv}  q={q:.5f}  T_train={T_train}  σ={sigma:.4f}  a={a:.6f}")
    _q_spec = {"F1": 1/9, "F2": 1/9, "F5": 1/9, "F6": 1/9}
    if run_id in _q_spec and abs(q - _q_spec[run_id]) > 0.02:
        print(f"  [WARN] q={q:.5f} deviates from spec value {_q_spec[run_id]:.5f} "
              f"for {run_id} — n_priv={n_priv} vs spec n≈{round(B_exp/_q_spec[run_id])}. "
              f"σ calibrated for actual q; downstream filter ceiling uses T_sigma_calibration.")
    print(f"  steps_per_ep={steps_per_ep}  T_sigma_calibration={T_train}  "
          f"(σ={sigma:.4f} calibrated for exactly {T_train} steps)")

    # --- Model + optimizer (always constructed before checkpoint load) ---
    model    = make_model(regime, num_classes, arch).to(device)
    d_params = sum(p.numel() for p in model.parameters())
    print(f"  d_params={d_params:,}")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Pre-allocate accounting arrays (always memmapped for resume safety) ---
    Yproj_path    = os.path.join(run_dir, "Y_projections.npy")
    Y_proj_mm     = _open_mm(Yproj_path,
                             np.float32, (n_priv, T_train, r_max))
    cn_mm         = _open_mm(os.path.join(run_dir, "_clipped_norms_mm.npy"),
                             np.float32, (n_priv, T_train))
    losses_mm     = _open_mm(os.path.join(run_dir, "_losses_mm.npy"),
                             np.float32, (n_priv, T_train))
    B_mm          = _open_mm(os.path.join(run_dir, "_B_matrices_mm.npy"),
                             np.float64, (T_train, r_max, r_max))
    YTY_mm        = _open_mm(os.path.join(run_dir, "_YTY_matrices_mm.npy"),
                             np.float64, (T_train, r_max, r_max))
    gnorm_mm      = _open_mm(os.path.join(run_dir, "_grad_sum_norms_mm.npy"),
                             np.float32, (T_train,))
    rbs_mm        = _open_mm(os.path.join(run_dir, "_realized_bs_mm.npy"),
                             np.int32,   (T_train,))
    if not is_clip:
        eig_mm  = _open_mm(os.path.join(run_dir, "_sketch_eigs_mm.npy"),
                            np.float32, (T_train, r_max))
        cond_mm = _open_mm(os.path.join(run_dir, "_sketch_cond_mm.npy"),
                            np.float32, (T_train,))

    # Poisson inclusions: reconstructed from deterministic seeds on resume — keep in RAM
    all_inclusions = np.zeros((T_train, n_priv), dtype=bool)

    step_global  = 0
    best_acc     = 0.0
    resume_epoch = 0

    # B2: per-step timing accumulators (wall-clock overhead measurement)
    _nystrom_times = []   # Nyström block seconds per step
    _dp_times      = []   # bare DP-update block seconds per step

    if resuming:
        ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data["model_state"])
        optimizer.load_state_dict(ckpt_data["optimizer_state"])
        scheduler.load_state_dict(ckpt_data["scheduler_state"])
        step_global  = ckpt_data["step_global"]
        best_acc     = ckpt_data["best_acc"]
        resume_epoch = ckpt_data["epoch"]
        # Fail fast if dataset dimensions changed between launches
        if "n_priv" in ckpt_data:
            assert ckpt_data["n_priv"] == n_priv, (
                f"Resume mismatch: checkpoint n_priv={ckpt_data['n_priv']} "
                f"vs current {n_priv} — delete checkpoint and restart")
            assert ckpt_data["T_train"] == T_train, (
                f"Resume mismatch: checkpoint T_train={ckpt_data['T_train']} "
                f"vs current {T_train} — delete checkpoint and restart")
            assert abs(ckpt_data["q"] - q) < 1e-8, (
                f"Resume mismatch: checkpoint q={ckpt_data['q']:.8f} "
                f"vs current {q:.8f} — delete checkpoint and restart")
        print(f"  [P19] Resumed from epoch {resume_epoch}, step {step_global}")
        # Recompute Poisson masks for completed steps from deterministic seeds
        for t_past in range(step_global):
            rng_p = np.random.default_rng(_step_seed(run_id, seed, t_past))
            all_inclusions[t_past] = rng_p.random(n_priv) < q
    else:
        # WRN/FashionMNIST warm-start only on fresh run; Purchase-100/MNIST train from scratch
        if not is_clip and not is_purchase and not is_mnist and pub_x is not None:
            pretrain_wrn(model, pub_x.to(device), pub_y.to(device), device)

    # For CLIP, keep features on device
    if is_clip:
        priv_feats_dev  = priv_feats.to(device)
        priv_labels_dev = priv_labels.to(device)

    # --- Training loop ---
    for epoch in range(resume_epoch + 1, epochs + 1):
        model.train()
        epoch_loss = 0.0; n_steps_ep = 0

        for _ep_step in range(steps_per_ep):
            if step_global >= T_train:
                break
            if max_steps is not None and step_global >= max_steps:
                break
            t       = step_global
            t_start = time.time()

            # ===========================================================
            # PRE-UPDATE: Nyström stats at θ_t (spec §5.1 steps 1-2)
            # ===========================================================
            model.eval()
            if is_clip:
                G_cpu, norms_t, losses_t, B_t, M_t, Yp_t = nystrom_stats_r3(
                    model, priv_feats_dev, priv_labels_dev, rho, r_max, device)
            else:
                norms_t, losses_t, B_t, M_t, Yp_t, top_eigvals_t, cond_num_t = \
                    nystrom_stats_wrn(model, priv_ds, rho, r_max, device, chunk_size)

            # Persist accounting statistics
            cn_mm[:, t]        = norms_t.numpy().astype(np.float32)
            losses_mm[:, t]    = losses_t.numpy().astype(np.float32)
            B_mm[t]            = B_t.numpy().astype(np.float64)
            YTY_mm[t]          = M_t.numpy().astype(np.float64)
            Y_proj_mm[:, t, :] = Yp_t.numpy().astype(np.float32)
            if not is_clip:
                ne = min(len(top_eigvals_t), r_max)
                eig_mm[t, :ne] = top_eigvals_t[:ne]
                cond_mm[t]     = cond_num_t

            t_nystrom_end = time.time()   # B2: Nyström block ends here

            # ===========================================================
            # True Poisson sampling (spec §5.2)
            # ===========================================================
            rng_step   = np.random.default_rng(_step_seed(run_id, seed, t))
            inclusion  = rng_step.random(n_priv) < q
            n_included = int(inclusion.sum())
            rbs_mm[t]          = n_included
            all_inclusions[t]  = inclusion

            # ===========================================================
            # Gradient sum G_sum = Σ_{i: I_i=1} ḡ_{i,t}
            # ===========================================================
            model.train()
            if is_clip:
                included_idx = np.where(inclusion)[0]
                if len(included_idx) > 0:
                    G_sum = G_cpu[included_idx].to(device).sum(0)
                else:
                    G_sum = torch.zeros(d_params, device=device)
                del G_cpu
            else:
                # Full-dataset pass with same batch_size/ordering as accounting passes —
                # per-sample grads are bitwise-identical (spec §8.2 soundness).
                G_sum = compute_poisson_G_sum_wrn(model, priv_ds, inclusion, device, chunk_size)

            gnorm_mm[t] = G_sum.norm().item()

            # ===========================================================
            # Noise + update: θ_{t+1} = θ_t - η (G_sum + ξ) / B_expected
            # (spec §5.1 steps 5-6)
            # ===========================================================
            # Per-step deterministic noise — reproducible across preemption boundaries.
            # Use a separate seed domain from Poisson masks (XOR with distinct constant).
            _xi_gen = torch.Generator(device=device)
            _xi_gen.manual_seed(_step_seed(run_id, seed, t) ^ 0x5DEECE66D)
            xi      = torch.randn(d_params, device=device, generator=_xi_gen) * (sigma * CLIP_C)
            G_noisy = (G_sum + xi) / B_exp

            optimizer.zero_grad(set_to_none=True)
            _set_grads(model, G_noisy)
            optimizer.step()

            t_elapsed         = time.time() - t_start
            t_nystrom_elapsed = t_nystrom_end - t_start   # B2: Nyström block
            t_dp_elapsed      = t_elapsed - t_nystrom_elapsed  # B2: bare DP update
            _nystrom_times.append(t_nystrom_elapsed)
            _dp_times.append(t_dp_elapsed)

            epoch_loss += losses_t.mean().item()
            n_steps_ep += 1
            step_global += 1

            if t % 10 == 0 or t < 5:
                print(f"    [step t={t:4d}] n_batch={n_included}  "
                      f"norm_med={norms_t.median():.4f}  "
                      f"loss_mean={losses_t.mean():.4f}  "
                      f"B_diag_max={float(B_t.diagonal().max()):.4g}  "
                      f"elapsed={t_elapsed:.2f}s  "
                      f"(nystrom={t_nystrom_elapsed:.2f}s  dp={t_dp_elapsed:.2f}s)")

            del G_sum, xi, G_noisy, norms_t, losses_t, B_t, M_t, Yp_t
            torch.cuda.empty_cache()

        scheduler.step()

        if is_clip:
            acc = evaluate_clip(model, te_feats, te_labels, device)
        else:
            acc = evaluate_wrn(model, test_ds, device)
        if acc > best_acc: best_acc = acc
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  ep {epoch:3d}/{epochs}  acc={acc:.4f}  best={best_acc:.4f}  "
              f"lr={cur_lr:.5f}  avg_loss={epoch_loss/max(n_steps_ep,1):.4f}")

        # Checkpoint + flush all memmaps
        torch.save({"epoch": epoch, "step_global": step_global, "best_acc": best_acc,
                    "n_priv": n_priv, "T_train": T_train, "q": q,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict()},
                   ckpt_path)
        Y_proj_mm.flush(); cn_mm.flush(); losses_mm.flush()
        B_mm.flush(); YTY_mm.flush(); gnorm_mm.flush(); rbs_mm.flush()
        if not is_clip:
            eig_mm.flush(); cond_mm.flush()

        if max_steps is not None and step_global >= max_steps:
            print(f"  [P19] max_steps={max_steps} reached — stopping early.")
            break

    T_actual = step_global

    # B2: Print timing summary after training
    if _nystrom_times:
        import numpy as _np
        nyst_arr = _np.array(_nystrom_times)
        dp_arr   = _np.array(_dp_times)
        nyst_pct = 100.0 * nyst_arr.mean() / max(nyst_arr.mean() + dp_arr.mean(), 1e-12)
        print(f"\n  [B2 timing] over {len(nyst_arr)} steps:")
        print(f"    Nyström block:   mean={nyst_arr.mean():.3f}s  p50={_np.median(nyst_arr):.3f}s  "
              f"p95={_np.percentile(nyst_arr,95):.3f}s")
        print(f"    Bare DP update:  mean={dp_arr.mean():.3f}s  p50={_np.median(dp_arr):.3f}s  "
              f"p95={_np.percentile(dp_arr,95):.3f}s")
        print(f"    Nyström overhead: {nyst_pct:.1f}% of per-step time "
              f"({nyst_arr.mean():.3f}s / {nyst_arr.mean()+dp_arr.mean():.3f}s per step)")
        timing_path = os.path.join(run_dir, "step_timing.json")
        with open(timing_path, "w") as _f:
            json.dump({
                "n_steps": len(nyst_arr),
                "nystrom_mean_s":   float(nyst_arr.mean()),
                "nystrom_median_s": float(_np.median(nyst_arr)),
                "nystrom_p95_s":    float(_np.percentile(nyst_arr, 95)),
                "dp_update_mean_s": float(dp_arr.mean()),
                "dp_update_p95_s":  float(_np.percentile(dp_arr, 95)),
                "nystrom_overhead_pct": float(nyst_pct),
            }, _f, indent=2)
        print(f"    [saved] {timing_path}")

    # --- Export canonical arrays (sliced to T_actual) ---
    def _mm_save(out_path, arr, slc):
        np.save(out_path, np.asarray(arr[slc]))

    _mm_save(os.path.join(run_dir, "clipped_norms.npy"),
             cn_mm,    (slice(None), slice(None, T_actual)))
    _mm_save(os.path.join(run_dir, "losses.npy"),
             losses_mm,(slice(None), slice(None, T_actual)))
    _mm_save(os.path.join(run_dir, "B_matrices.npy"),
             B_mm,     (slice(None, T_actual),))
    _mm_save(os.path.join(run_dir, "YTY_matrices.npy"),
             YTY_mm,   (slice(None, T_actual),))
    _mm_save(os.path.join(run_dir, "gradient_sum_norms.npy"),
             gnorm_mm, (slice(None, T_actual),))
    _mm_save(os.path.join(run_dir, "realized_batch_sizes.npy"),
             rbs_mm,   (slice(None, T_actual),))
    if not is_clip:
        _mm_save(os.path.join(run_dir, "sketch_eigenvalues.npy"),
                 eig_mm,  (slice(None, T_actual),))
        _mm_save(os.path.join(run_dir, "sketch_condition_numbers.npy"),
                 cond_mm, (slice(None, T_actual),))

    # Y_projections: memmap IS the canonical file; truncate only if early-stopped
    Y_proj_mm.flush()
    if T_actual < T_train:
        tmp_path = Yproj_path + ".tmp"
        np.save(tmp_path, np.asarray(Y_proj_mm[:, :T_actual, :r_max]))
        os.rename(tmp_path, Yproj_path)

    # Poisson inclusions compressed
    np.savez_compressed(os.path.join(run_dir, "poisson_inclusions.npz"),
                        inclusions=all_inclusions[:T_actual])

    # Clean up intermediate memmap files
    for _tmp in ["_clipped_norms_mm.npy", "_losses_mm.npy", "_B_matrices_mm.npy",
                 "_YTY_matrices_mm.npy", "_grad_sum_norms_mm.npy", "_realized_bs_mm.npy",
                 "_sketch_eigs_mm.npy", "_sketch_cond_mm.npy"]:
        _p = os.path.join(run_dir, _tmp)
        if os.path.exists(_p): os.remove(_p)

    # Example metadata
    np.save(os.path.join(run_dir, "example_indices.npy"), priv_idx.astype(np.int32))
    np.save(os.path.join(run_dir, "labels.npy"),          priv_labels_np)
    np.save(os.path.join(run_dir, "class_counts.npy"),    class_counts)
    if tier_labels is not None:
        np.save(os.path.join(run_dir, "tier_labels.npy"), tier_labels)

    # LiRA target selection
    n_lira = LIRA_N_TARGETS[run_id]
    member_local, nonmember_test = select_lira_targets(
        priv_idx, priv_labels_np, tier_labels, test_labels_np, n_lira, seed=1000)
    np.save(os.path.join(run_dir, "lira_member_local_idx.npy"),   member_local)
    np.save(os.path.join(run_dir, "lira_nonmember_test_idx.npy"), nonmember_test)

    # Final model + public/private split
    torch.save(model.state_dict(), final_path)
    np.save(os.path.join(run_dir, "private_indices.npy"), priv_idx.astype(np.int32))
    np.save(os.path.join(run_dir, "public_indices.npy"),  pub_idx.astype(np.int32))

    # DP-model logits for LiRA targets
    @torch.no_grad()
    def _logits_clip(idx_local):
        model.eval(); parts = []
        for s in range(0, len(idx_local), 512):
            b = idx_local[s:s + 512]
            parts.append(model(priv_feats[b].to(device).float()).cpu())
        return torch.cat(parts).numpy().astype(np.float32)

    @torch.no_grad()
    def _logits_clip_test(idx_global):
        model.eval(); parts = []
        for s in range(0, len(idx_global), 512):
            b = idx_global[s:s + 512]
            parts.append(model(te_feats[b].to(device).float()).cpu())
        return torch.cat(parts).numpy().astype(np.float32)

    @torch.no_grad()
    def _logits_wrn_priv(idx_local):
        model.eval(); parts = []
        for s in range(0, len(idx_local), 256):
            b = idx_local[s:s + 256]
            x = torch.stack([priv_ds[int(j)][0] for j in b]).to(device)
            parts.append(model(x).cpu())
        return torch.cat(parts).numpy().astype(np.float32)

    @torch.no_grad()
    def _logits_wrn_test(idx_global):
        model.eval(); parts = []
        for s in range(0, len(idx_global), 256):
            b = idx_global[s:s + 256]
            x = torch.stack([test_ds[int(j)][0] for j in b]).to(device)
            parts.append(model(x).cpu())
        return torch.cat(parts).numpy().astype(np.float32)

    if is_clip:
        mem_logits  = _logits_clip(member_local)
        nmem_logits = _logits_clip_test(nonmember_test)
    else:
        mem_logits  = _logits_wrn_priv(member_local)
        nmem_logits = _logits_wrn_test(nonmember_test)

    np.save(os.path.join(run_dir, "target_logits_dp_members.npy"),    mem_logits)
    np.save(os.path.join(run_dir, "target_logits_dp_nonmembers.npy"), nmem_logits)
    np.save(os.path.join(run_dir, "target_logits_dp.npy"),            mem_logits)

    # Sanity check: realized batch sizes
    rbs = np.asarray(rbs_mm[:T_actual])
    print(f"\n  [sanity] Poisson batch: E[|B|]={rbs.mean():.1f}  "
          f"E_theory={q*n_priv:.1f}  "
          f"std[|B|]={rbs.std():.1f}  "
          f"std_theory={math.sqrt(n_priv*q*(1-q)):.1f}")

    # Run metadata
    metadata = {
        "run_id":           run_id,
        "seed":             seed,
        "dataset":          dataset,
        "regime":           regime,
        "architecture":     ("CLIP_ViT_B_32_linear"    if is_clip
                             else ("PurchaseFC_256"       if arch == "purchase_fc"
                                   else ("MNISTConvNet_dpconv" if arch == "dpconv"
                                         else ("TinyCNN_GroupNorm"    if arch == "tinycnn"
                                               else ("ResNet-20_GroupNorm16" if arch == "resnet20"
                                                     else "WRN-28-2_GroupNorm16"))))),
        "n_train":          int(n_priv),
        "B_expected":       int(B_exp),
        "q":                float(q),
        "rho":              float(rho),
        "C":                float(CLIP_C),
        "sigma":            float(sigma),
        "a":                float(a),
        "delta":            float(delta),
        "epsilon_target":   float(eps),
        "epochs":           int(epochs),
        "T_train":          int(T_actual),
        "T_sigma_calibration": int(T_train),
        "steps_per_epoch":  int(steps_per_ep),
        "r_max":            r_max,
        "accountant":       "prv",
        "sampling":         "poisson_bernoulli",
        "augmentation":     "none" if not is_clip else "none_frozen_clip",
        "pre_update_timing": True,
        "best_test_acc":    float(best_acc),
        "n_lira_members":   int(len(member_local)),
        "n_lira_nonmembers": int(len(nonmember_test)),
        "realized_batch_mean": float(rbs.mean()),
        "realized_batch_std":  float(rbs.std()),
        "phase":            "p19_training_logs",
        "per_instance_accounting": "posthoc_fixed_grid_renyi_filter_required",
        "certified_epsilons_computed": False,
        "gradient_convention": "per_sample_clipped_sum",
        "dp_update_convention": "noisy_sum_divided_by_B_expected",
        "accounting_timing": "pre_update",
        "Q_method":         "exact_top_eigenvectors_full_covariance" if is_clip
                            else "deterministic_randomized_range_finder",
        "nystrom_dtype":    "float64" if is_clip else "float32",
        "poisson_seed_scheme": "sha256_run_seed_step",
        "max_steps_used":   max_steps,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Remove checkpoint now that run is complete
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"\n[P19] {run_id}/seed_{seed} DONE  best_acc={best_acc:.4f}  T={T_actual}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 19 training")
    parser.add_argument("--run",       type=str, default=None, choices=list(RUNS.keys()))
    parser.add_argument("--seed",      type=int, default=None)
    parser.add_argument("--all_seeds", action="store_true")
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--gpu",       type=int, default=0)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--runs_dir",  type=str, default=RUNS_DIR)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after this many gradient steps (dry-run mode)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[P19] Device: {device}")

    if args.run and not args.all:
        runs_to_run = {args.run: RUNS[args.run]}
    elif args.all:
        runs_to_run = RUNS
    else:
        print("[P19] No run specified. Use --run F1/F2/F5 or --all.")
        return

    for rid, cfg in runs_to_run.items():
        if args.all_seeds:
            seeds = list(range(cfg["n_seeds"]))
        elif args.seed is not None:
            seeds = [args.seed]
        else:
            seeds = [0]
        for s in seeds:
            train_run(rid, cfg, s, device, args.data_root, args.cache_dir, args.runs_dir,
                      max_steps=args.max_steps)

    print("[P19] All training done.")


if __name__ == "__main__":
    main()
