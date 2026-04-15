#!/usr/bin/env python3
"""
Phase 16 — Training: Direction-Aware Certificates (Full Run Matrix)
====================================================================

Implements the full run matrix from phase16_spex.tex:

  Tier 1 (HIGH, 5 seeds):  H1-H10  — headline CLIP + from-scratch + EMNIST
  Tier 2 (MED,  3 seeds):  S1-S8   — ε-sweep and batch-size sweep
  Tier 3 (MED,  1 seed):   M3, M7  — cross-mechanism (auto-clip only)

Three regimes:
  R1: WRN-28-2 + GroupNorm, cold start (random init)
  R2: WRN-28-2 + GroupNorm, warm start (pretrained on public data)
  R3: CLIP ViT-B/32 frozen features + linear head only

Two mechanisms:
  vanilla  standard clip + Gaussian noise
  auto     smooth automatic clipping: g_scaled = g / (||g||/C + 1)

Per-step logs:
  Scalar columns (parquet/npz): example_idx, grad_norm, incoherent_norm
  PCA projections (companion _gproj.npy, shape (N_rows, rank) float32):
    g_proj_k = (gc_clip @ V)_k  for k=0..rank-1

  These projections are required for the Woodbury rigorous certificate in
  exp_p16_certify.py.  The PCA eigenvalues λ_k are saved in _meta.npz["lambdas"]
  so the certify script can compute d_eff^(r)² = Σ_k g_proj_k²/(σ²+λ_k) + ||g_perp||²/σ².

  g_proj is NOT stored in the rolling torch checkpoint (too large at rank=100).
  Instead a companion _gproj_ckpt.npy is written alongside each checkpoint and
  reloaded on resume.

Checkpoints saved every epoch — fully resumable.

Usage
-----
  # Block A: CLIP headline (fastest, run first)
  python experiments/exp_p16_train.py --block A --gpu 0

  # Single run, specific seed
  python experiments/exp_p16_train.py --run H1 --seed 0 --gpu 0

  # All seeds for a run
  python experiments/exp_p16_train.py --run H1 --gpu 0

  # All runs in order
  python experiments/exp_p16_train.py --all --gpu 0

  # By tier
  python experiments/exp_p16_train.py --tier 1 --gpu 0
"""

import os
import sys
import csv
import math
import argparse
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets import make_public_private_split, make_cifar10_lt_indices
from src.models import WideResNet, ResNet20

import torchvision
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Run matrix
# ---------------------------------------------------------------------------

RUN_MATRIX = {
    # --- Tier 1: Headline results (5 seeds) ---
    "H1":  dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5, tier=1),
    "H2":  dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=2.0, batch=5000,  n_seeds=5, tier=1),
    "H3":  dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=1.0, batch=5000,  n_seeds=5, tier=1),
    "H4":  dict(dataset="cifar10_lt50",  regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5, tier=1),
    "H5":  dict(dataset="cifar10_lt100", regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5, tier=1),
    "H6":  dict(dataset="cifar10",       regime="R1", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5, tier=1),
    "H7":  dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5, tier=1),
    "H8":  dict(dataset="cifar10_lt50",  regime="R2", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5, tier=1),
    "H9":  dict(dataset="emnist",        regime="R2", mech="vanilla", eps=8.0, batch=10000, n_seeds=5, tier=1),
    "H10": dict(dataset="emnist",        regime="R2", mech="vanilla", eps=8.0, batch=25000, n_seeds=5, tier=1),
    # --- Tier 2: ε-sweep and batch-size sweep (3 seeds) ---
    "S1":  dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=1.0, batch=5000,  n_seeds=3, tier=2),
    "S2":  dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=2.0, batch=5000,  n_seeds=3, tier=2),
    "S3":  dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=4.0, batch=5000,  n_seeds=3, tier=2),
    "S4":  dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=8.0, batch=1000,  n_seeds=3, tier=2),
    "S5":  dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=8.0, batch=10000, n_seeds=3, tier=2),
    "S6":  dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=1000,  n_seeds=3, tier=2),
    "S7":  dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=10000, n_seeds=3, tier=2),
    "S8":  dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=25000, n_seeds=3, tier=2),
    # --- Tier 3: Cross-mechanism, auto-clip only (1 seed) ---
    "M3":  dict(dataset="cifar10",       regime="R2", mech="auto",    eps=8.0, batch=5000,  n_seeds=1, tier=3),
    "M7":  dict(dataset="cifar10",       regime="R3", mech="auto",    eps=8.0, batch=5000,  n_seeds=1, tier=3),
}

# Execution order: Block A first (CLIP, fastest), then from-scratch, EMNIST, sweeps, mech
BLOCK_ORDER = {
    "A": ["H1", "H2", "H3", "H4", "H5", "S6", "S7", "S8"],
    "B": ["H6", "H7", "H8"],
    "C": ["H9", "H10"],
    "D": ["S1", "S2", "S3", "S4", "S5"],
    "E": ["M3", "M7"],
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_C          = 1.0
RANK_V          = 100          # subspace rank for R1/R2/EMNIST
RANK_V_CLIP     = 100          # subspace rank for R3
N_PUB           = 2000         # public examples for subspace
LR              = 0.1
MOMENTUM        = 0.9
WEIGHT_DECAY    = 5e-4
PRETRAIN_EPOCHS = 50
PRETRAIN_LR     = 0.01
PUB_BATCH       = 256
GRAD_CHUNK      = 64           # vmap chunk size (R1/R2)
CLIP_CHUNK      = 256          # analytical chunk size (R3)
DATA_ROOT       = "./data"
CACHE_DIR       = "./data/clip_features"
RESULTS_DIR     = "./results/exp_p16"

# Long-tailed tier boundaries (CIFAR-10)
LT_HEAD_CLASSES = {0, 1, 2}
LT_MID_CLASSES  = {3, 4, 5, 6}
LT_TAIL_CLASSES = {7, 8, 9}


def class_to_tier(c):
    if c in LT_HEAD_CLASSES:
        return 0
    elif c in LT_MID_CLASSES:
        return 1
    return 2


def get_epochs(regime, dataset):
    """Training epochs per regime/dataset."""
    if regime == "R3":
        return 100   # CLIP linear probe: fast, use more epochs
    if "emnist" in dataset:
        return 30
    return 60


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class _IndexedSubset(Dataset):
    """Returns (x, y, global_dataset_index)."""
    def __init__(self, base_ds, indices):
        self.base    = base_ds
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, y, int(self.indices[i])


class _EMNISTIndexedSubset(Dataset):
    """Returns (x, y_remapped, global_idx) for EMNIST (remap 1..26 → 0..25)."""
    def __init__(self, base_ds, indices):
        self.base    = base_ds
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, int(y) - 1, int(self.indices[i])


class _FeatureDataset(Dataset):
    """Dataset of pre-extracted (feature, label, global_idx) triples."""
    def __init__(self, features, labels, global_indices):
        self.features = features           # [N, d]
        self.labels   = labels             # [N] long
        self.indices  = global_indices     # [N] int

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.features[i], int(self.labels[i]), int(self.indices[i])


def _cifar10_raw(data_root, augment):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    tf   = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(mean, std)]) if augment else \
           T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return torchvision.datasets.CIFAR10(root=data_root, train=True,
                                        download=True, transform=tf)


def _cifar10_test(data_root):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    return torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True,
        transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]))


def _emnist_raw(data_root, augment):
    tf = T.Compose([T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                    T.ToTensor(), T.Normalize((0.1722,), (0.3309,))]) if augment else \
         T.Compose([T.ToTensor(), T.Normalize((0.1722,), (0.3309,))])
    return torchvision.datasets.EMNIST(root=data_root, split="letters",
                                       train=True, download=True, transform=tf)


def _emnist_test(data_root):
    return torchvision.datasets.EMNIST(
        root=data_root, split="letters", train=False, download=True,
        transform=T.Compose([T.ToTensor(), T.Normalize((0.1722,), (0.3309,))]))


def build_dataset_wrn(dataset_name, data_root, seed=42):
    """Build dataset for R1/R2 (WRN training on raw images)."""
    is_lt    = "lt" in dataset_name
    lt_ir    = 50 if "lt50" in dataset_name else (100 if "lt100" in dataset_name else 1)
    is_emnist = "emnist" in dataset_name

    if is_emnist:
        train_aug   = _emnist_raw(data_root, augment=True)
        train_noaug = _emnist_raw(data_root, augment=False)
        test_ds     = _emnist_test(data_root)
        raw_targets = np.array(train_aug.targets) - 1   # 0-indexed
        all_idx     = np.arange(len(train_aug))
        num_classes = 26
        pub_idx, priv_idx = make_public_private_split(
            all_idx, raw_targets, public_frac=0.1, seed=seed)
        rng     = np.random.default_rng(seed)
        pub_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]
        pub_ds  = Subset(train_noaug, pub_use.tolist())
        priv_ds = _EMNISTIndexedSubset(train_aug, priv_idx)
        tier_labels = None
        delta   = 1.0 / (len(priv_ds) ** 1.1)
        pub_x   = torch.stack([pub_ds[i][0] for i in range(len(pub_ds))])
        pub_y   = torch.tensor([int(train_noaug.targets[pub_use[i]]) - 1
                                 for i in range(len(pub_use))], dtype=torch.long)
        return pub_ds, priv_ds, test_ds, pub_x, pub_y, num_classes, priv_idx, tier_labels, delta
    else:
        train_aug   = _cifar10_raw(data_root, augment=True)
        train_noaug = _cifar10_raw(data_root, augment=False)
        test_ds     = _cifar10_test(data_root)
        full_targets = np.array(train_aug.targets)
        all_idx      = np.arange(len(train_aug))
        if is_lt:
            lt_idx = make_cifar10_lt_indices(full_targets, lt_ir, seed=seed)
        else:
            lt_idx = all_idx
        lt_targets = full_targets[lt_idx]
        pub_idx, priv_idx = make_public_private_split(
            lt_idx, lt_targets, public_frac=0.1, seed=seed)
        rng     = np.random.default_rng(seed)
        pub_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]
        pub_ds  = Subset(train_noaug, pub_use.tolist())
        priv_ds = _IndexedSubset(train_aug, priv_idx)
        if is_lt:
            priv_targets = full_targets[priv_idx]
            tier_labels  = np.array([class_to_tier(c) for c in priv_targets], dtype=np.int32)
        else:
            tier_labels = None
        pub_x = torch.stack([pub_ds[i][0] for i in range(len(pub_ds))])
        pub_y = torch.tensor([int(full_targets[pub_use[i]])
                               for i in range(len(pub_use))], dtype=torch.long)
        delta = 1e-5
        return pub_ds, priv_ds, test_ds, pub_x, pub_y, 10, priv_idx, tier_labels, delta


# ---------------------------------------------------------------------------
# CLIP feature extraction
# ---------------------------------------------------------------------------

def _load_or_extract_clip_features(dataset_name, data_root, cache_dir, device):
    """
    Extract CLIP ViT-B/32 features for CIFAR-10 (train + test).
    Caches to cache_dir. Returns (train_features [N,512], train_labels [N],
                                   test_features [M,512], test_labels [M]).
    """
    os.makedirs(cache_dir, exist_ok=True)
    tag = dataset_name.replace("_lt50", "").replace("_lt100", "")
    if tag == "cifar10":
        feat_path = os.path.join(cache_dir, "cifar10_clip_train.pt")
        lab_path  = os.path.join(cache_dir, "cifar10_clip_train_labels.pt")
        tfeat_path = os.path.join(cache_dir, "cifar10_clip_test.pt")
        tlab_path  = os.path.join(cache_dir, "cifar10_clip_test_labels.pt")
    else:
        raise ValueError(f"CLIP features only for cifar10 datasets, got {dataset_name}")

    if (os.path.exists(feat_path) and os.path.exists(lab_path) and
            os.path.exists(tfeat_path) and os.path.exists(tlab_path)):
        print(f"  [CLIP] Loading cached features from {feat_path}")
        train_feats  = torch.load(feat_path,  map_location="cpu")
        train_labels = torch.load(lab_path,   map_location="cpu")
        test_feats   = torch.load(tfeat_path, map_location="cpu")
        test_labels  = torch.load(tlab_path,  map_location="cpu")
        return train_feats, train_labels, test_feats, test_labels

    print("  [CLIP] Extracting features (requires open_clip). Downloading if needed...")
    try:
        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        clip_model = clip_model.to(device).eval()
    except ImportError:
        try:
            import clip as openai_clip
            clip_model, preprocess = openai_clip.load("ViT-B/32", device=device)
            clip_model.eval()
        except ImportError:
            raise RuntimeError(
                "Neither open_clip nor clip package found. "
                "Install with: pip install open_clip_torch  or  "
                "pip install git+https://github.com/openai/CLIP.git"
            )

    def _extract(dataset_obj, desc):
        all_feats  = []
        all_labels = []
        loader = DataLoader(dataset_obj, batch_size=256, shuffle=False, num_workers=4)
        for batch in loader:
            imgs, labs = batch[0], batch[1]
            imgs = imgs.to(device)
            with torch.no_grad():
                feats = clip_model.encode_image(imgs)
            all_feats.append(feats.cpu().float())
            all_labels.append(labs)
        return torch.cat(all_feats), torch.cat(all_labels)

    clip_mean = (0.48145466, 0.4578275,  0.40821073)
    clip_std  = (0.26862954, 0.26130258, 0.27577711)
    clip_tf   = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(clip_mean, clip_std),
    ])

    train_raw = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                              download=True, transform=clip_tf)
    test_raw  = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                              download=True, transform=clip_tf)

    print("  [CLIP] Extracting train features...")
    train_feats, train_labels = _extract(train_raw, "train")
    print(f"  [CLIP] train: {train_feats.shape}")
    print("  [CLIP] Extracting test features...")
    test_feats, test_labels = _extract(test_raw, "test")
    print(f"  [CLIP] test: {test_feats.shape}")

    torch.save(train_feats,  feat_path)
    torch.save(train_labels, lab_path)
    torch.save(test_feats,   tfeat_path)
    torch.save(test_labels,  tlab_path)
    print(f"  [CLIP] Saved to {cache_dir}")

    del clip_model
    torch.cuda.empty_cache()
    return train_feats, train_labels, test_feats, test_labels


def build_dataset_clip(dataset_name, data_root, cache_dir, device, seed=42):
    """Build dataset for R3 (CLIP linear probe)."""
    is_lt = "lt" in dataset_name
    lt_ir = 50 if "lt50" in dataset_name else (100 if "lt100" in dataset_name else 1)

    train_feats, train_labels, test_feats, test_labels = \
        _load_or_extract_clip_features("cifar10", data_root, cache_dir, device)

    full_targets = train_labels.numpy()
    all_idx      = np.arange(len(full_targets))

    if is_lt:
        lt_idx = make_cifar10_lt_indices(full_targets, lt_ir, seed=seed)
    else:
        lt_idx = all_idx

    lt_targets = full_targets[lt_idx]
    pub_idx, priv_idx = make_public_private_split(
        lt_idx, lt_targets, public_frac=0.1, seed=seed)

    rng     = np.random.default_rng(seed)
    pub_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]

    pub_feats  = train_feats[pub_use]
    pub_labels = torch.tensor(full_targets[pub_use], dtype=torch.long)

    priv_feats  = train_feats[priv_idx]
    priv_labels = torch.tensor(full_targets[priv_idx], dtype=torch.long)
    priv_global = torch.tensor(priv_idx, dtype=torch.long)

    priv_ds = _FeatureDataset(priv_feats, priv_labels, priv_global)

    if is_lt:
        tier_labels = np.array([class_to_tier(c) for c in full_targets[priv_idx]],
                                dtype=np.int32)
    else:
        tier_labels = None

    test_ds = _FeatureDataset(test_feats, test_labels, torch.arange(len(test_labels)))

    return (pub_feats, pub_labels, priv_ds, test_ds,
            pub_feats, pub_labels, 10, priv_idx, tier_labels, 1e-5)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    """CLIP linear probe: K×d weight + bias."""
    def __init__(self, num_classes, feature_dim=512):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def make_model(regime, num_classes, dataset_name):
    if regime == "R3":
        return LinearHead(num_classes, feature_dim=512)
    elif "emnist" in dataset_name:
        # SimpleCNN for EMNIST (28×28 grayscale)
        class SimpleCNN(nn.Module):
            def __init__(self, nc, n_groups=8):
                super().__init__()
                self.conv1 = nn.Conv2d(1,   32,  3, padding=1, bias=False)
                self.gn1   = nn.GroupNorm(n_groups, 32)
                self.conv2 = nn.Conv2d(32,  64,  3, padding=1, bias=False)
                self.gn2   = nn.GroupNorm(n_groups, 64)
                self.conv3 = nn.Conv2d(64,  128, 3, padding=1, bias=False)
                self.gn3   = nn.GroupNorm(n_groups, 128)
                self.pool  = nn.MaxPool2d(2, 2)
                self.fc    = nn.Linear(128, nc)
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    elif isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
            def forward(self, x):
                x = self.pool(F.relu(self.gn1(self.conv1(x))))
                x = self.pool(F.relu(self.gn2(self.conv2(x))))
                x = F.relu(self.gn3(self.conv3(x)))
                x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
                return self.fc(x)
        return SimpleCNN(num_classes)
    else:
        return ResNet20(num_classes=num_classes, n_groups=16)


# ---------------------------------------------------------------------------
# Per-sample gradients
# ---------------------------------------------------------------------------

def _loss_fn(params, buffers, x, y, model):
    pred = torch.func.functional_call(model, (params, buffers), x.unsqueeze(0))
    return F.cross_entropy(pred, y.unsqueeze(0))


def _per_sample_grads_vmap(model, x_chunk, y_chunk, device):
    """Per-sample gradients via vmap (for R1/R2 WRN)."""
    params  = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    grad_fn = torch.func.grad(lambda p, b, xi, yi: _loss_fn(p, b, xi, yi, model))
    vmapped = torch.func.vmap(grad_fn, in_dims=(None, None, 0, 0))
    with torch.no_grad():
        g_dict = vmapped(params, buffers, x_chunk.to(device), y_chunk.to(device))
    flat = torch.cat([g_dict[k].reshape(x_chunk.shape[0], -1) for k in params], dim=1)
    return flat  # [B, d]


def _per_sample_grads_linear(model, h_chunk, y_chunk, device):
    """
    Analytical per-sample gradients for linear head.
    g_j = [(p_j - e_{y_j}) ⊗ h_j, p_j - e_{y_j}]   (W grad then b grad)
    Returns [B, K*d + K].
    """
    h = h_chunk.to(device).float()   # [B, d]
    y = y_chunk.to(device).long()    # [B]
    W = model.fc.weight               # [K, d]
    b = model.fc.bias                 # [K]

    with torch.no_grad():
        logits = h @ W.t() + b        # [B, K]
        p      = torch.softmax(logits, dim=1)
        ey     = torch.zeros_like(p)
        ey.scatter_(1, y.unsqueeze(1), 1.0)
        delta  = p - ey               # [B, K]
        g_W    = (delta.unsqueeze(2) * h.unsqueeze(1)).reshape(h.shape[0], -1)   # [B, K*d]
        g_b    = delta                                                             # [B, K]
    return torch.cat([g_W, g_b], dim=1)   # [B, K*d+K]


def _set_grads(model, flat_grad):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat_grad[offset:offset+n].view(p.shape).clone()
        offset += n


# ---------------------------------------------------------------------------
# Subspace (PCA of public clipped gradients) + eigenvalues
# ---------------------------------------------------------------------------

def _compute_subspace(model, pub_x, pub_y, device, rank=RANK_V, regime="R2", chunk=GRAD_CHUNK):
    """
    Compute rank-r PCA subspace V from public clipped gradients.

    Returns
    -------
    V       : torch.Tensor [d, rank]  — right singular vectors (subspace basis)
    lambdas : np.ndarray  (rank,)     — eigenvalues of (1/N) G^T G = S²/N
                                        needed for Woodbury certificate formula
    """
    print(f"  [subspace] Computing rank-{rank} PCA subspace (regime={regime})...")
    model.eval()
    parts = []
    N = pub_x.shape[0]
    c = CLIP_CHUNK if regime == "R3" else chunk
    for i in range(0, N, c):
        xc = pub_x[i:i+c]
        yc = pub_y[i:i+c]
        if regime == "R3":
            g = _per_sample_grads_linear(model, xc, yc, device)
        else:
            g = _per_sample_grads_vmap(model, xc, yc, device)
        norms = g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        parts.append((g * (CLIP_C / norms).clamp(max=1.0)).cpu())
        del g; torch.cuda.empty_cache()
    G = torch.cat(parts, dim=0).float()
    N_pub = G.shape[0]
    k = min(rank, G.shape[0] - 1, G.shape[1] - 1)
    _, S, V = torch.svd_lowrank(G, q=k, niter=6)
    V       = V[:, :k].cpu()                                # [d, k]
    S       = S[:k].cpu()
    lambdas = (S ** 2 / N_pub).numpy().astype(np.float64)   # eigenvalues of G^T G / N
    print(f"  [subspace] V={V.shape}  λ_max={lambdas[0]:.4g}  λ_min={lambdas[-1]:.4g}")
    del G; torch.cuda.empty_cache()
    return V, lambdas


# ---------------------------------------------------------------------------
# Public pretraining (R2 regime)
# ---------------------------------------------------------------------------

def _pretrain(model, pub_x, pub_y, device):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=PRETRAIN_LR, momentum=0.9, weight_decay=5e-4)
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
# Privacy calibration
# ---------------------------------------------------------------------------

def calibrate_sigma(eps, delta, q, T_steps):
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(
        target_epsilon=float(eps), target_delta=float(delta),
        sample_rate=float(q), steps=int(T_steps), accountant="rdp")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, test_ds, device, is_emnist=False, is_clip=False):
    loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                        num_workers=4, pin_memory=True)
    model.eval()
    correct = total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        if is_emnist:
            y = y - 1  # remap
        correct += (model(x.float()).argmax(1) == y).sum().item()
        total   += y.shape[0]
    return correct / total


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def save_log(log_steps, log_idx, log_gnorm, log_inorm, out_path):
    """Save scalar per-step log (grad_norm, incoherent_norm) to parquet or npz."""
    try:
        import pandas as pd
        df = pd.DataFrame({
            "step":            np.asarray(log_steps,  dtype=np.int32),
            "example_idx":     np.asarray(log_idx,    dtype=np.int32),
            "grad_norm":       np.asarray(log_gnorm,  dtype=np.float32),
            "incoherent_norm": np.asarray(log_inorm,  dtype=np.float32),
        })
        df.to_parquet(out_path)
        print(f"  [log] {out_path} ({len(df):,} rows, {os.path.getsize(out_path)//1024}KB)")
    except Exception as e:
        npz = out_path.replace(".parquet", ".npz")
        np.savez_compressed(npz,
            step=np.asarray(log_steps,  dtype=np.int32),
            example_idx=np.asarray(log_idx,   dtype=np.int32),
            grad_norm=np.asarray(log_gnorm, dtype=np.float32),
            incoherent_norm=np.asarray(log_inorm, dtype=np.float32))
        print(f"  [log] fallback npz: {npz}")


def _save_gproj(log_gproj, rank, out_path):
    """
    Save accumulated g_proj rows to a .npy file.

    log_gproj : list of (rank,) float32 arrays, one per logged example-step
    Saves as float32 array of shape (N_rows, rank).
    """
    if log_gproj:
        arr = np.vstack(log_gproj).astype(np.float32)
    else:
        arr = np.empty((0, rank), dtype=np.float32)
    np.save(out_path, arr)
    print(f"  [log] gproj {out_path} ({arr.shape[0]:,} rows × {arr.shape[1]} cols, "
          f"{arr.nbytes // (1024*1024)}MB)")


def load_log_from_ckpt(ckpt):
    return (list(ckpt.get("log_steps", [])),
            list(ckpt.get("log_idx",   [])),
            list(ckpt.get("log_gnorm", [])),
            list(ckpt.get("log_inorm", [])))


# ---------------------------------------------------------------------------
# Core training step (mechanism-agnostic gradient collection)
# ---------------------------------------------------------------------------

def _collect_grads(model, x, y, ex_idx, V_gpu, device, regime,
                   log_steps, log_idx, log_gnorm, log_inorm, log_gproj, step):
    """
    Collect per-sample gradients, clip, compute per-example statistics, return sum_g.

    Logs per example-step:
      - grad_norm     : ||g_clip||
      - incoherent_norm: ||g_clip - V V^T g_clip||  (out-of-subspace component)
      - g_proj        : V^T g_clip  (rank-d projection coefficients)
                        appended to log_gproj as (rank,) float32 arrays
    """
    B     = x.shape[0]
    d     = sum(p.numel() for p in model.parameters())
    sum_g = torch.zeros(d, device=device)
    c     = CLIP_CHUNK if regime == "R3" else GRAD_CHUNK

    for ci in range(0, B, c):
        xc    = x[ci:ci+c]
        yc    = y[ci:ci+c]
        idx_c = ex_idx[ci:ci+c]

        if regime == "R3":
            gc = _per_sample_grads_linear(model, xc, yc, device)
        else:
            gc = _per_sample_grads_vmap(model, xc, yc, device)

        norms   = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
        gc_clip = gc * (CLIP_C / norms).clamp(max=1.0)
        sum_g  += gc_clip.sum(0)

        with torch.no_grad():
            coords   = gc_clip @ V_gpu       # (chunk, rank) — PCA projections
            g_V      = coords @ V_gpu.t()    # back-projection into subspace
            g_perp   = gc_clip - g_V
            gnorms_c = gc_clip.norm(dim=1)
            inorms_c = g_perp.norm(dim=1)

        log_steps.extend([step] * len(idx_c))
        log_idx.extend(idx_c.tolist())
        log_gnorm.extend(gnorms_c.cpu().numpy().astype(np.float32).tolist())
        log_inorm.extend(inorms_c.cpu().numpy().astype(np.float32).tolist())
        log_gproj.extend(list(coords.cpu().numpy().astype(np.float32)))

        del gc, gc_clip, coords, g_V, g_perp, gnorms_c, inorms_c
        torch.cuda.empty_cache()

    return sum_g


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_run(run_id, cfg, seed, device, data_root, cache_dir,
              out_dir, log_dir, ckpt_dir):
    tag       = f"p16_{run_id}_{cfg['mech']}_{cfg['dataset']}_{cfg['regime']}" \
                f"_eps{cfg['eps']:.0f}_seed{seed}"
    csv_path       = os.path.join(out_dir,  f"{tag}.csv")
    ckpt_path      = os.path.join(ckpt_dir, f"{tag}_ckpt.pt")
    final_path     = os.path.join(out_dir,  f"{tag}_final.pt")
    log_path       = os.path.join(log_dir,  f"{tag}.parquet")
    meta_path      = os.path.join(log_dir,  f"{tag}_meta.npz")
    # g_proj companion files (separate from torch checkpoint due to size)
    gproj_ckpt_path  = os.path.join(ckpt_dir, f"{tag}_gproj_ckpt.npy")
    gproj_final_path = os.path.join(log_dir,  f"{tag}_gproj.npy")

    log_npz  = log_path.replace(".parquet", ".npz")
    log_done = (os.path.exists(log_path) or os.path.exists(log_npz)) \
               and os.path.exists(gproj_final_path)
    if os.path.exists(final_path) and os.path.exists(csv_path) and log_done:
        print(f"[P16] {tag}: already done, skipping.")
        return

    print(f"\n[P16] === {run_id} (seed={seed}) ===")
    print(f"  dataset={cfg['dataset']} regime={cfg['regime']} mech={cfg['mech']} "
          f"eps={cfg['eps']} batch={cfg['batch']} seed={seed}")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    regime  = cfg["regime"]
    dataset = cfg["dataset"]
    mech    = cfg["mech"]
    eps     = cfg["eps"]
    batch   = cfg["batch"]
    is_clip  = regime == "R3"
    is_emnist= "emnist" in dataset

    # Build dataset
    if is_clip:
        (pub_feats, pub_labels, priv_ds, test_ds,
         pub_x, pub_y, num_classes, priv_idx, tier_labels, delta) = \
            build_dataset_clip(dataset, data_root, cache_dir, device, seed=42)
    else:
        (pub_ds, priv_ds, test_ds, pub_x, pub_y, num_classes,
         priv_idx, tier_labels, delta) = build_dataset_wrn(dataset, data_root, seed=42)

    n_priv = len(priv_ds)
    epochs  = get_epochs(regime, dataset)

    steps_per_epoch = max(1, n_priv // batch)
    T_steps = epochs * steps_per_epoch
    q       = batch / n_priv
    sigma   = calibrate_sigma(eps, delta, q, T_steps)
    sigma_use = sigma * CLIP_C

    print(f"  n_priv={n_priv}, num_classes={num_classes}, delta={delta:.2e}")
    print(f"  T={T_steps}, q={q:.5f}, sigma_mult={sigma:.4f}, sigma_use={sigma_use:.4f}")

    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Model
    model = make_model(regime, num_classes, dataset).to(device)
    d     = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {d:,}")

    # Pretrain for R2 (warm start)
    pub_x_dev = pub_x.to(device).float()
    pub_y_dev = pub_y.to(device).long()

    if regime == "R2":
        _pretrain(model, pub_x_dev, pub_y_dev, device)

    pub_x_cpu = pub_x.cpu()
    pub_y_cpu = pub_y.cpu()

    # Subspace + eigenvalues (needed before saving meta so lambdas are available)
    rank_v = RANK_V_CLIP if is_clip else RANK_V
    V_cpu, lambdas = _compute_subspace(model, pub_x_cpu, pub_y_cpu, device,
                                        rank=rank_v, regime=regime)
    V_gpu = V_cpu.to(device)

    # Save meta (includes lambdas for Woodbury certificate computation)
    np.savez(meta_path,
             run_id=np.str_(run_id), mech=np.str_(mech),
             dataset=np.str_(dataset), regime=np.str_(regime),
             n_priv=np.int32(n_priv), batch_size=np.int32(batch),
             epochs=np.int32(epochs), eps=np.float64(eps),
             delta=np.float64(delta), q=np.float64(q),
             T_steps=np.int32(T_steps), sigma_mult=np.float64(sigma),
             sigma_use=np.float64(sigma_use),
             lambdas=lambdas,
             tier_labels=(tier_labels if tier_labels is not None
                          else np.array([], dtype=np.int32)),
             priv_idx=priv_idx.astype(np.int32))

    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Resume from checkpoint
    start_epoch  = 1
    log_steps    = []
    log_idx      = []
    log_gnorm    = []
    log_inorm    = []
    log_gproj    = []   # list of (rank_v,) float32 arrays — NOT in torch checkpoint
    step_global  = 0
    best_acc     = 0.0

    if os.path.exists(ckpt_path):
        print(f"  [resume] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        torch.set_rng_state(ckpt["rng_state"])
        np.random.set_state(ckpt["np_rng_state"])
        random.setstate(ckpt["py_rng_state"])
        start_epoch = ckpt["epoch"] + 1
        step_global = ckpt["step_global"]
        best_acc    = ckpt["best_acc"]
        log_steps, log_idx, log_gnorm, log_inorm = load_log_from_ckpt(ckpt)
        # Load g_proj from companion file (not stored in torch checkpoint)
        expected_gproj_rows = ckpt.get("log_gproj_nrows", 0)
        if os.path.exists(gproj_ckpt_path):
            try:
                arr = np.load(gproj_ckpt_path)
                log_gproj = list(arr[:expected_gproj_rows])
                print(f"  [resume] Loaded {len(log_gproj):,} gproj rows from {gproj_ckpt_path}")
            except Exception as e:
                print(f"  [resume] gproj load failed ({e}); gproj will be incomplete")
                log_gproj = []
        else:
            print(f"  [resume] gproj companion not found; gproj will be incomplete")
        print(f"  [resume] Resumed from epoch {ckpt['epoch']}, step {step_global}, "
              f"log_rows={len(log_steps):,}")

    # CSV writer (append mode if resuming)
    csv_mode  = "a" if start_epoch > 1 and os.path.exists(csv_path) else "w"
    csv_file  = open(csv_path, csv_mode, newline="")
    writer    = csv.DictWriter(csv_file, fieldnames=["epoch","train_loss","test_acc","lr"])
    if csv_mode == "w":
        writer.writeheader()

    priv_loader = DataLoader(priv_ds, batch_size=batch, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch_data in priv_loader:
            x, y, ex_idx = batch_data[0], batch_data[1], batch_data[2]
            optimizer.zero_grad(set_to_none=True)

            if mech == "auto":
                # Smooth auto-clip: g_scaled = g / (||g||/C + 1)
                B     = x.shape[0]
                sum_g = torch.zeros(d, device=device)
                c     = CLIP_CHUNK if is_clip else GRAD_CHUNK
                for ci in range(0, B, c):
                    xc    = x[ci:ci+c]
                    yc    = y[ci:ci+c]
                    idx_c = ex_idx[ci:ci+c]
                    if is_clip:
                        gc = _per_sample_grads_linear(model, xc, yc, device)
                    else:
                        gc = _per_sample_grads_vmap(model, xc, yc, device)
                    norms   = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    scale   = CLIP_C / (norms + CLIP_C)
                    gc_clip = gc * scale
                    sum_g  += gc_clip.sum(0)
                    with torch.no_grad():
                        coords   = gc_clip @ V_gpu       # (chunk, rank)
                        g_V      = coords @ V_gpu.t()
                        g_perp   = gc_clip - g_V
                        gnorms_c = gc_clip.norm(dim=1)
                        inorms_c = g_perp.norm(dim=1)
                    log_steps.extend([step_global] * len(idx_c))
                    log_idx.extend(idx_c.tolist())
                    log_gnorm.extend(gnorms_c.cpu().numpy().astype(np.float32).tolist())
                    log_inorm.extend(inorms_c.cpu().numpy().astype(np.float32).tolist())
                    log_gproj.extend(list(coords.cpu().numpy().astype(np.float32)))
                    del gc, gc_clip, coords, g_V, g_perp, gnorms_c, inorms_c
                    torch.cuda.empty_cache()
            else:
                # vanilla: standard clip + Gaussian noise
                sum_g = _collect_grads(
                    model, x, y, ex_idx, V_gpu, device, regime,
                    log_steps, log_idx, log_gnorm, log_inorm, log_gproj, step_global)

            B = x.shape[0]
            noise  = torch.randn(d, device=device) * sigma_use
            flat_g = (sum_g + noise) / B

            _set_grads(model, flat_g)
            optimizer.step()

            with torch.no_grad():
                b_sub = min(B, 64)
                if is_clip:
                    out = model(x[:b_sub].to(device).float())
                else:
                    out = model(x[:b_sub].to(device))
                total_loss += F.cross_entropy(out, y[:b_sub].to(device)).item()
            n_batches   += 1
            step_global += 1

        scheduler.step()

        test_acc = evaluate(model, test_ds, device, is_emnist=is_emnist, is_clip=is_clip)
        cur_lr   = optimizer.param_groups[0]["lr"]

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f"{tag}_best.pt"))

        row = {"epoch": epoch, "train_loss": f"{total_loss/max(n_batches,1):.4f}",
               "test_acc": f"{test_acc:.4f}", "lr": f"{cur_lr:.6f}"}
        writer.writerow(row); csv_file.flush()
        print(f"  ep {epoch:3d}/{epochs}  acc={test_acc:.4f}  best={best_acc:.4f}  "
              f"log_rows={len(log_steps):,}  gproj_rows={len(log_gproj):,}")

        # Save checkpoint + companion g_proj file every epoch
        _save_gproj(log_gproj, rank_v, gproj_ckpt_path)
        torch.save({
            "epoch":             epoch,
            "model_state":       model.state_dict(),
            "optimizer_state":   optimizer.state_dict(),
            "scheduler_state":   scheduler.state_dict(),
            "rng_state":         torch.get_rng_state(),
            "np_rng_state":      np.random.get_state(),
            "py_rng_state":      random.getstate(),
            "step_global":       step_global,
            "best_acc":          best_acc,
            "log_steps":         log_steps,
            "log_idx":           log_idx,
            "log_gnorm":         log_gnorm,
            "log_inorm":         log_inorm,
            "log_gproj_nrows":   len(log_gproj),   # pointer into companion .npy
        }, ckpt_path)

        # Flush scalar log every 5 epochs (every epoch for CLIP)
        if is_clip or epoch % 5 == 0:
            save_log(log_steps, log_idx, log_gnorm, log_inorm, log_path)

    csv_file.close()
    torch.save(model.state_dict(), final_path)
    save_log(log_steps, log_idx, log_gnorm, log_inorm, log_path)
    _save_gproj(log_gproj, rank_v, gproj_final_path)

    # Remove rolling checkpoint (keep final only)
    for p in [ckpt_path, gproj_ckpt_path]:
        if os.path.exists(p):
            os.remove(p)

    del V_gpu; torch.cuda.empty_cache()
    print(f"[P16] {run_id} seed={seed} done — final={test_acc:.4f}  best={best_acc:.4f}  "
          f"total_log_rows={len(log_steps):,}  gproj_rows={len(log_gproj):,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 16 training")
    parser.add_argument("--run",       type=str, default=None,
                        help="Single run ID (e.g. H1, S3, M3)")
    parser.add_argument("--block",     type=str, default=None,
                        choices=list(BLOCK_ORDER.keys()),
                        help="Run all entries in a block (A=CLIP, B=scratch, C=EMNIST, D=sweep, E=mech)")
    parser.add_argument("--tier",      type=int, default=None, choices=[1, 2, 3],
                        help="Run all entries in a tier")
    parser.add_argument("--all",       action="store_true",
                        help="Run all entries in priority order")
    parser.add_argument("--seed",      type=int, default=None,
                        help="Override seed (default: run all n_seeds for the run)")
    parser.add_argument("--gpu",       type=int, default=0)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--out_dir",   type=str, default=os.path.join(RESULTS_DIR, "train"))
    parser.add_argument("--log_dir",   type=str, default=os.path.join(RESULTS_DIR, "logs"))
    parser.add_argument("--ckpt_dir",  type=str, default=os.path.join(RESULTS_DIR, "checkpoints"))
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[P16] Device: {device}")

    os.makedirs(args.out_dir,  exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Determine which runs to execute
    if args.run:
        run_ids = [args.run]
    elif args.block:
        run_ids = BLOCK_ORDER[args.block]
    elif args.tier:
        run_ids = [rid for rid, c in RUN_MATRIX.items() if c["tier"] == args.tier]
    elif args.all:
        run_ids = []
        for blk in "ABCDE":
            run_ids.extend(BLOCK_ORDER[blk])
    else:
        print("[P16] No run specified. Running Block A (CLIP headline). Use --all for full matrix.")
        run_ids = BLOCK_ORDER["A"]

    for run_id in run_ids:
        if run_id not in RUN_MATRIX:
            print(f"[P16] Unknown run ID: {run_id}, skipping.")
            continue
        cfg = RUN_MATRIX[run_id]

        if args.seed is not None:
            seeds = [args.seed]
        else:
            seeds = list(range(cfg["n_seeds"]))

        for seed in seeds:
            train_run(run_id, cfg, seed, device, args.data_root, args.cache_dir,
                      args.out_dir, args.log_dir, args.ckpt_dir)


if __name__ == "__main__":
    main()
