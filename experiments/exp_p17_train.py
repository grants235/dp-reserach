#!/usr/bin/env python3
"""
Phase 17 v4 — Training: Private-Side Eigenvectors + Corrected Composition
==========================================================================

V4 changes from V3:
  Public-side PCA + Loewner verification is dropped (failed in practice at κ=0.1).
  Instead: at each accounting step, compute the top-r eigenpairs of Σ_t directly
  from the all-example clipped gradients (private side).  The Loewner condition
  holds by construction, no shrinkage or verification is needed.

Per-step pipeline (at every K-th training step):
  1. DP-SGD update (unchanged).
  2. Compute per-sample clipped gradients for ALL n examples at θ_t.
  3. For R3 (CLIP, d=5130): materialize G_t ∈ ℝ^{n×d}, compute
       Σ_t = q(1-q) G_t.T G_t, eigenpairs via torch.linalg.eigh.
     For R1/R2 (ResNet/CNN, d~270k): chunked Lanczos on the implicit operator
       Σ_t v = q(1-q) Σ_j ḡ_j (ḡ_j^T v), recomputing gradients per iteration.
  4. Project all gradients onto top-r eigenvectors U_r^(t).
  5. Accumulate per example j:
       sum_norm2[j]      += ||ḡ_{j,t}||² / σ_use²
       sum_reduction_k[j,k] += λ_k·(ḡ_j^T u_k)² / (σ_use²·(σ_use²+λ_k))
  At non-accounting steps: nothing extra logged; certify uses data-independent fallback.

Accumulated statistics saved per example:
  sum_norm2       : [n_priv]        — Σ_t ||ḡ||²/σ_use² at accounting steps
  sum_reduction_k : [n_priv, rank]  — per-direction reduction at accounting steps
  eigval_history  : [n_accounted, rank] — eigenvalues at each accounting step (for plots)
  n_accounted     : int             — number of accounting steps completed

Usage
-----
  python experiments/exp_p17_train.py --block A --gpu 0
  python experiments/exp_p17_train.py --run C3 --seed 0 --gpu 0
  python experiments/exp_p17_train.py --all --gpu 0
  python experiments/exp_p17_train.py --run B4 --seed 0 --K 8 --gpu 0
"""

import os, sys, csv, math, argparse, random, json
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
# Run matrix  (v4: reduced seeds per budget)
# ---------------------------------------------------------------------------

RUN_MATRIX = {
    # Tier 1: R1 cold start (2 seeds, K=8)
    "A1": dict(dataset="cifar10",       regime="R1", mech="vanilla", eps=8.0, batch=5000,  n_seeds=2, tier=1, K=8),
    # Tier 1: R2 warm start ε-sweep + LT + EMNIST (2 seeds, K=8)
    "B1": dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=1.0, batch=5000,  n_seeds=2, tier=1, K=8),
    "B2": dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=2.0, batch=5000,  n_seeds=2, tier=1, K=8),
    "B3": dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=4.0, batch=5000,  n_seeds=2, tier=1, K=8),
    "B4": dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=8.0, batch=5000,  n_seeds=2, tier=1, K=8),
    "B5": dict(dataset="cifar10_lt50",  regime="R2", mech="vanilla", eps=8.0, batch=5000,  n_seeds=2, tier=1, K=8),
    "B6": dict(dataset="cifar10_lt100", regime="R2", mech="vanilla", eps=8.0, batch=5000,  n_seeds=2, tier=1, K=8),
    "B7": dict(dataset="emnist",        regime="R2", mech="vanilla", eps=8.0, batch=10000, n_seeds=2, tier=1, K=8),
    # Tier 1: R3 CLIP ε-sweep + LT (3 seeds, K=1)
    "C1": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=1.0, batch=5000,  n_seeds=3, tier=1, K=1),
    "C2": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=2.0, batch=5000,  n_seeds=3, tier=1, K=1),
    "C3": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=3, tier=1, K=1),
    "C4": dict(dataset="cifar10_lt50",  regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=3, tier=1, K=1),
    "C5": dict(dataset="cifar10_lt100", regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=3, tier=1, K=1),
    # Tier 2: CLIP batch sweep (2 seeds, K=1)
    "D1": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=1000,  n_seeds=2, tier=2, K=1),
    "D2": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=2, tier=2, K=1),
    "D3": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=10000, n_seeds=2, tier=2, K=1),
    "D4": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=25000, n_seeds=2, tier=2, K=1),
}

BLOCK_ORDER = {
    "A": ["C1", "C2", "C3", "C4", "C5"],       # CLIP headline first (fastest)
    "B": ["D1", "D2", "D3", "D4"],              # CLIP batch sweep
    "C": ["A1", "B4", "B5", "B6"],              # from-scratch
    "D": ["B1", "B2", "B3", "B7"],              # ε-sweep + EMNIST
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_C          = 1.0
RANK            = 100              # eigenvectors to compute per step
LANCZOS_ITERS   = 60              # for chunked Lanczos (WRN/CNN)
ALL_EX_CHUNK    = 256             # chunk size for all-example sweep (R3)
WRN_CHUNK       = 64              # chunk size for vmap per-sample grads (R1/R2)
N_PUB           = 2000
LR              = 0.1
MOMENTUM        = 0.9
WEIGHT_DECAY    = 5e-4
PRETRAIN_EPOCHS = 50
PRETRAIN_LR     = 0.01
PUB_BATCH       = 256
DATA_ROOT       = "./data"
CACHE_DIR       = "./data/clip_features"
RESULTS_DIR     = "./results/exp_p17"

LT_HEAD_CLASSES = {0, 1, 2}
LT_MID_CLASSES  = {3, 4, 5, 6}
LT_TAIL_CLASSES = {7, 8, 9}


def class_to_tier(c):
    if c in LT_HEAD_CLASSES: return 0
    if c in LT_MID_CLASSES:  return 1
    return 2


def get_epochs(regime, dataset):
    if regime == "R3": return 100
    if "emnist" in dataset: return 30
    return 60


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class _IndexedSubset(Dataset):
    def __init__(self, base_ds, indices):
        self.base = base_ds; self.indices = np.asarray(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, y, int(self.indices[i])


class _EMNISTIndexedSubset(Dataset):
    def __init__(self, base_ds, indices):
        self.base = base_ds; self.indices = np.asarray(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, int(y) - 1, int(self.indices[i])


class _FeatureDataset(Dataset):
    def __init__(self, features, labels, global_indices):
        self.features = features; self.labels = labels; self.indices = global_indices
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.features[i], int(self.labels[i]), int(self.indices[i])


def _cifar10_raw(data_root, augment):
    m, s = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize(m, s)]) if augment else \
         T.Compose([T.ToTensor(), T.Normalize(m, s)])
    return torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf)


def _cifar10_test(data_root):
    m, s = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    return torchvision.datasets.CIFAR10(root=data_root, train=False, download=True,
        transform=T.Compose([T.ToTensor(), T.Normalize(m, s)]))


def _emnist_raw(data_root, augment):
    tf = T.Compose([T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                    T.ToTensor(), T.Normalize((0.1722,), (0.3309,))]) if augment else \
         T.Compose([T.ToTensor(), T.Normalize((0.1722,), (0.3309,))])
    return torchvision.datasets.EMNIST(root=data_root, split="letters",
                                       train=True, download=True, transform=tf)


def _emnist_test(data_root):
    return torchvision.datasets.EMNIST(root=data_root, split="letters", train=False,
        download=True, transform=T.Compose([T.ToTensor(), T.Normalize((0.1722,), (0.3309,))]))


def build_dataset_wrn(dataset_name, data_root, seed=42):
    is_lt = "lt" in dataset_name; lt_ir = 50 if "lt50" in dataset_name else (100 if "lt100" in dataset_name else 1)
    is_emnist = "emnist" in dataset_name
    if is_emnist:
        train_aug = _emnist_raw(data_root, augment=True)
        train_noaug = _emnist_raw(data_root, augment=False)
        test_ds = _emnist_test(data_root)
        raw_targets = np.array(train_aug.targets) - 1
        all_idx = np.arange(len(train_aug)); num_classes = 26
        pub_idx, priv_idx = make_public_private_split(all_idx, raw_targets, public_frac=0.1, seed=seed)
        rng = np.random.default_rng(seed)
        pub_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]
        pub_ds = Subset(train_noaug, pub_use.tolist())
        priv_ds = _EMNISTIndexedSubset(train_aug, priv_idx)
        tier_labels = None; delta = 1.0 / (len(priv_ds) ** 1.1)
        pub_x = torch.stack([pub_ds[i][0] for i in range(len(pub_ds))])
        pub_y = torch.tensor([int(train_noaug.targets[pub_use[i]]) - 1 for i in range(len(pub_use))], dtype=torch.long)
        return pub_ds, priv_ds, test_ds, pub_x, pub_y, num_classes, priv_idx, tier_labels, delta
    else:
        train_aug = _cifar10_raw(data_root, augment=True)
        train_noaug = _cifar10_raw(data_root, augment=False)
        test_ds = _cifar10_test(data_root)
        full_targets = np.array(train_aug.targets); all_idx = np.arange(len(train_aug))
        lt_idx = make_cifar10_lt_indices(full_targets, lt_ir, seed=seed) if is_lt else all_idx
        lt_targets = full_targets[lt_idx]
        pub_idx, priv_idx = make_public_private_split(lt_idx, lt_targets, public_frac=0.1, seed=seed)
        rng = np.random.default_rng(seed)
        pub_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]
        pub_ds = Subset(train_noaug, pub_use.tolist())
        priv_ds = _IndexedSubset(train_aug, priv_idx)
        tier_labels = np.array([class_to_tier(c) for c in full_targets[priv_idx]], dtype=np.int32) if is_lt else None
        pub_x = torch.stack([pub_ds[i][0] for i in range(len(pub_ds))])
        pub_y = torch.tensor([int(full_targets[pub_use[i]]) for i in range(len(pub_use))], dtype=torch.long)
        return pub_ds, priv_ds, test_ds, pub_x, pub_y, 10, priv_idx, tier_labels, 1e-5


# ---------------------------------------------------------------------------
# CLIP feature extraction
# ---------------------------------------------------------------------------

def _load_or_extract_clip_features(data_root, cache_dir, device):
    os.makedirs(cache_dir, exist_ok=True)
    paths = {k: os.path.join(cache_dir, f"cifar10_clip_{k}.pt")
             for k in ["train", "train_labels", "test", "test_labels"]}
    if all(os.path.exists(p) for p in paths.values()):
        print("  [CLIP] Loading cached features")
        return (torch.load(paths["train"],        map_location="cpu", weights_only=False),
                torch.load(paths["train_labels"], map_location="cpu", weights_only=False),
                torch.load(paths["test"],         map_location="cpu", weights_only=False),
                torch.load(paths["test_labels"],  map_location="cpu", weights_only=False))
    print("  [CLIP] Extracting features (requires open_clip)...")
    try:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    except ImportError:
        import clip as openai_clip
        clip_model, _ = openai_clip.load("ViT-B/32", device=device)
    clip_model = clip_model.to(device).eval()
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std  = (0.26862954, 0.26130258, 0.27577711)
    clip_tf   = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                            T.CenterCrop(224), T.ToTensor(), T.Normalize(clip_mean, clip_std)])
    def _extract(ds_obj):
        feats, labs = [], []
        for batch in DataLoader(ds_obj, batch_size=256, shuffle=False, num_workers=4):
            with torch.no_grad():
                f = clip_model.encode_image(batch[0].to(device))
            feats.append(f.cpu().float()); labs.append(batch[1])
        return torch.cat(feats), torch.cat(labs)
    train_raw = torchvision.datasets.CIFAR10(root=data_root, train=True,  download=True, transform=clip_tf)
    test_raw  = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=clip_tf)
    tf, tl = _extract(train_raw); ef, el = _extract(test_raw)
    torch.save(tf, paths["train"]); torch.save(tl, paths["train_labels"])
    torch.save(ef, paths["test"]);  torch.save(el, paths["test_labels"])
    del clip_model; torch.cuda.empty_cache()
    return tf, tl, ef, el


def build_dataset_clip(dataset_name, data_root, cache_dir, device, seed=42):
    is_lt = "lt" in dataset_name
    lt_ir = 50 if "lt50" in dataset_name else (100 if "lt100" in dataset_name else 1)
    train_feats, train_labels, test_feats, test_labels = \
        _load_or_extract_clip_features(data_root, cache_dir, device)
    full_targets = train_labels.numpy(); all_idx = np.arange(len(full_targets))
    lt_idx = make_cifar10_lt_indices(full_targets, lt_ir, seed=seed) if is_lt else all_idx
    lt_targets = full_targets[lt_idx]
    pub_idx, priv_idx = make_public_private_split(lt_idx, lt_targets, public_frac=0.1, seed=seed)
    rng = np.random.default_rng(seed)
    pub_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]
    pub_feats  = train_feats[pub_use]
    pub_labels = torch.tensor(full_targets[pub_use], dtype=torch.long)
    priv_feats  = train_feats[priv_idx]
    priv_labels = torch.tensor(full_targets[priv_idx], dtype=torch.long)
    priv_ds = _FeatureDataset(priv_feats, priv_labels, torch.tensor(priv_idx, dtype=torch.long))
    tier_labels = np.array([class_to_tier(c) for c in full_targets[priv_idx]], dtype=np.int32) if is_lt else None
    test_ds = _FeatureDataset(test_feats, test_labels, torch.arange(len(test_labels)))
    return (pub_feats, pub_labels, priv_ds, test_ds,
            pub_feats, pub_labels, 10, priv_idx, tier_labels, 1e-5,
            priv_feats, priv_labels)   # last two: all private features for sweep


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super().__init__(); self.fc = nn.Linear(feature_dim, num_classes)
    def forward(self, x): return self.fc(x)


def make_model(regime, num_classes, dataset_name):
    if regime == "R3": return LinearHead(num_classes, feature_dim=512)
    if "emnist" in dataset_name:
        class SimpleCNN(nn.Module):
            def __init__(self, nc, n_groups=8):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False); self.gn1 = nn.GroupNorm(n_groups, 32)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False); self.gn2 = nn.GroupNorm(n_groups, 64)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False); self.gn3 = nn.GroupNorm(n_groups, 128)
                self.pool = nn.MaxPool2d(2, 2); self.fc = nn.Linear(128, nc)
                for m in self.modules():
                    if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    elif isinstance(m, nn.Linear): nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
            def forward(self, x):
                x = self.pool(F.relu(self.gn1(self.conv1(x))))
                x = self.pool(F.relu(self.gn2(self.conv2(x))))
                return self.fc(F.adaptive_avg_pool2d(F.relu(self.gn3(self.conv3(x))), 1).view(x.size(0), -1))
        return SimpleCNN(num_classes)
    return ResNet20(num_classes=num_classes, n_groups=16)


# ---------------------------------------------------------------------------
# Per-sample gradients
# ---------------------------------------------------------------------------

def _loss_fn(params, buffers, x, y, model):
    pred = torch.func.functional_call(model, (params, buffers), x.unsqueeze(0))
    return F.cross_entropy(pred, y.unsqueeze(0))


def _per_sample_grads_vmap(model, x_chunk, y_chunk, device):
    params = dict(model.named_parameters()); buffers = dict(model.named_buffers())
    grad_fn = torch.func.grad(lambda p, b, xi, yi: _loss_fn(p, b, xi, yi, model))
    vmapped = torch.func.vmap(grad_fn, in_dims=(None, None, 0, 0))
    with torch.no_grad():
        g_dict = vmapped(params, buffers, x_chunk.to(device), y_chunk.to(device))
    return torch.cat([g_dict[k].reshape(x_chunk.shape[0], -1) for k in params], dim=1)


def _per_sample_grads_linear(model, h_chunk, y_chunk, device):
    h = h_chunk.to(device).float(); y = y_chunk.to(device).long()
    W = model.fc.weight; b = model.fc.bias
    with torch.no_grad():
        logits = h @ W.t() + b; p = torch.softmax(logits, dim=1)
        ey = torch.zeros_like(p); ey.scatter_(1, y.unsqueeze(1), 1.0); delta = p - ey
        g_W = (delta.unsqueeze(2) * h.unsqueeze(1)).reshape(h.shape[0], -1)
    return torch.cat([g_W, delta], dim=1)


def _set_grads(model, flat_grad):
    offset = 0
    for p in model.parameters():
        n = p.numel(); p.grad = flat_grad[offset:offset+n].view(p.shape).clone(); offset += n


# ---------------------------------------------------------------------------
# Standard DP-SGD step (collect gradients for the training update)
# ---------------------------------------------------------------------------

def _collect_and_update(model, x, y, sigma_use, d_params, device, regime):
    """Standard DP-SGD: clip per-sample, sum, add noise, return flat noisy gradient."""
    B = x.shape[0]; sum_g = torch.zeros(d_params, device=device)
    c = ALL_EX_CHUNK if regime == "R3" else WRN_CHUNK
    for ci in range(0, B, c):
        xc, yc = x[ci:ci+c], y[ci:ci+c]
        gc = (_per_sample_grads_linear(model, xc, yc, device) if regime == "R3"
              else _per_sample_grads_vmap(model, xc, yc, device))
        norms = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sum_g += (gc * (CLIP_C / norms).clamp(max=1.0)).sum(0)
        del gc; torch.cuda.empty_cache()
    noise = torch.randn(d_params, device=device) * sigma_use
    return (sum_g + noise) / B


# ---------------------------------------------------------------------------
# Private-side eigenpairs of Σ_t (v4 core)
# ---------------------------------------------------------------------------

def _eigenpairs_r3(model, priv_feats, priv_labels, q, rank, device):
    """
    R3 (CLIP, d=5130): materialize G_t ∈ ℝ^{n×d} on GPU, compute eigenpairs
    of Σ_t = q(1-q) G_t.T G_t via torch.linalg.eigh on the [d×d] covariance.

    Returns (U_r [d, rank], eigvals [rank]) — descending order.
    """
    n = len(priv_feats)
    G_parts = []
    for i in range(0, n, ALL_EX_CHUNK):
        h = priv_feats[i:i+ALL_EX_CHUNK].to(device).float()
        y = priv_labels[i:i+ALL_EX_CHUNK].to(device).long()
        with torch.no_grad():
            gc = _per_sample_grads_linear(model, h, y, device)
            norms = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
            G_parts.append((gc * (CLIP_C / norms).clamp(max=1.0)).half())  # float16 to save VRAM
        del gc; torch.cuda.empty_cache()
    G_t = torch.cat(G_parts, dim=0).float()  # [n_priv, d]

    with torch.no_grad():
        cov = q * (1.0 - q) * (G_t.T @ G_t)      # [d, d]
        eigvals_all, eigvecs_all = torch.linalg.eigh(cov)  # ascending
        U_r = eigvecs_all[:, -rank:].flip(1)        # [d, rank], descending
        lam  = eigvals_all[-rank:].flip(0)           # [rank], descending

    projections = G_t @ U_r  # [n_priv, rank]
    norms_sq    = (G_t ** 2).sum(dim=1)  # [n_priv]

    del G_t, cov, eigvecs_all; torch.cuda.empty_cache()
    return U_r, lam, projections, norms_sq


def _sigma_t_matvec_chunked(v, model, priv_ds, q, device, regime):
    """
    Compute Σ_t v = q(1-q) Σ_j ḡ_j (ḡ_j^T v) by sweeping the dataset in chunks.
    v: [d] GPU tensor.  Returns [d] GPU tensor.
    """
    result = torch.zeros_like(v)
    N = len(priv_ds)
    for i in range(0, N, WRN_CHUNK):
        batch = [priv_ds[j] for j in range(i, min(i + WRN_CHUNK, N))]
        xb = torch.stack([b[0] for b in batch])
        yb = torch.tensor([b[1] for b in batch], dtype=torch.long)
        with torch.no_grad():
            gc   = _per_sample_grads_vmap(model, xb, yb, device)
            nms  = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
            gc_c = gc * (CLIP_C / nms).clamp(max=1.0)
            dots = gc_c @ v                  # [chunk]
            result += gc_c.T @ dots          # [d]
        del gc, gc_c, dots; torch.cuda.empty_cache()
    return q * (1.0 - q) * result


def _eigenpairs_chunked(model, priv_ds, q, rank, device, n_iter=LANCZOS_ITERS):
    """
    R1/R2: Lanczos on Σ_t using chunked matvec (gradients recomputed per iteration).
    Returns (U_r [d, rank], eigvals [rank], projections [n_priv, rank], norms_sq [n_priv]).
    """
    d = sum(p.numel() for p in model.parameters())

    # ----- Lanczos (custom, on GPU) ----------------------------------------
    v = F.normalize(torch.randn(d, device=device), dim=0)
    V_mat = torch.zeros(d, n_iter + 1, device=device)   # Krylov basis
    alpha = torch.zeros(n_iter, device=device)
    beta  = torch.zeros(n_iter + 1, device=device)

    V_mat[:, 0] = v
    for j in range(n_iter):
        Av = _sigma_t_matvec_chunked(V_mat[:, j], model, priv_ds, q, device, regime="R1")
        if j > 0:
            Av -= beta[j] * V_mat[:, j - 1]
        alpha[j] = V_mat[:, j] @ Av
        Av -= alpha[j] * V_mat[:, j]
        beta[j + 1] = Av.norm()
        if beta[j + 1].item() < 1e-10:
            n_iter = j + 1
            break
        V_mat[:, j + 1] = Av / beta[j + 1]

    # Tridiagonal eigenproblem
    n_conv = n_iter
    T_diag = alpha[:n_conv]
    T_off  = beta[1:n_conv]
    T_mat  = torch.diag(T_diag) + torch.diag(T_off, 1) + torch.diag(T_off, -1)
    evals_T, evecs_T = torch.linalg.eigh(T_mat)   # ascending

    # Top-rank Ritz vectors
    k_use   = min(rank, n_conv)
    U_r     = (V_mat[:, :n_conv] @ evecs_T[:, -k_use:]).flip(1)   # [d, k_use] descending
    lam     = evals_T[-k_use:].flip(0)                              # [k_use] descending
    del V_mat, T_mat, evecs_T; torch.cuda.empty_cache()

    # ----- Compute projections + norms (one more sweep) --------------------
    N = len(priv_ds)
    proj_parts = []; norms_sq_parts = []
    U_r_T = U_r.T   # [rank, d]
    for i in range(0, N, WRN_CHUNK):
        batch = [priv_ds[j] for j in range(i, min(i + WRN_CHUNK, N))]
        xb = torch.stack([b[0] for b in batch])
        yb = torch.tensor([b[1] for b in batch], dtype=torch.long)
        with torch.no_grad():
            gc   = _per_sample_grads_vmap(model, xb, yb, device)
            nms  = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
            gc_c = gc * (CLIP_C / nms).clamp(max=1.0)
            proj_parts.append((gc_c @ U_r).cpu())         # [chunk, rank]
            norms_sq_parts.append((gc_c ** 2).sum(1).cpu())  # [chunk]
        del gc, gc_c; torch.cuda.empty_cache()

    projections = torch.cat(proj_parts, dim=0)   # [n_priv, rank]
    norms_sq    = torch.cat(norms_sq_parts, dim=0)  # [n_priv]
    return U_r, lam, projections, norms_sq


# ---------------------------------------------------------------------------
# Accumulate one accounting step
# ---------------------------------------------------------------------------

def _accumulate_step(projections, norms_sq, lam, sigma_use, sum_norm2, sum_reduction_k):
    """
    Update accumulated statistics in-place from one accounting step.

    projections : [n_priv, rank] tensor (CPU) — ḡ_j^T u_k for each j, k
    norms_sq    : [n_priv] tensor (CPU) — ||ḡ_j||²
    lam         : [rank] tensor — eigenvalues λ_k(Σ_t)
    sigma_use   : float — noise std = σ_mult · C

    sum_norm2      [n_priv]       += ||ḡ||² / σ_use²
    sum_reduction_k [n_priv, rank] += λ_k · (ḡ^T u_k)² / (σ_use² · (σ_use² + λ_k))
    """
    sigma2  = sigma_use ** 2
    # Clamp squared norms to C² before accumulating.  The float16 round-trip in
    # _eigenpairs_r3 (gc_clipped.half() → float32) can push individual norms
    # fractionally above CLIP_C, causing sum_norm2 to slightly exceed the
    # data-independent worst-case bound and failing the sanity check.
    gn2_np  = np.minimum(norms_sq.numpy().astype(np.float64), CLIP_C ** 2)
    proj_np = projections.numpy().astype(np.float64)
    lam_np  = lam.cpu().numpy().astype(np.float64)

    sum_norm2[:] += gn2_np / sigma2

    # reduction_k[j, k] = λ_k * proj²[j,k] / (σ²*(σ²+λ_k))
    denom  = sigma2 * (sigma2 + lam_np)          # [rank]
    weight = lam_np / denom                       # [rank]
    sum_reduction_k[:] += (proj_np ** 2) * weight[None, :]


# ---------------------------------------------------------------------------
# Public pretraining (R2)
# ---------------------------------------------------------------------------

def _pretrain(model, pub_x, pub_y, device):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=PRETRAIN_LR, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)
    N = pub_x.shape[0]
    for ep in range(1, PRETRAIN_EPOCHS + 1):
        perm = torch.randperm(N)
        for i in range(0, N, PUB_BATCH):
            idx = perm[i:i+PUB_BATCH]; opt.zero_grad()
            F.cross_entropy(model(pub_x[idx].to(device)), pub_y[idx].to(device)).backward(); opt.step()
        sch.step()
    print(f"  [pretrain] done ({PRETRAIN_EPOCHS} ep)")


def calibrate_sigma(eps, delta, q, T_steps):
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(target_epsilon=float(eps), target_delta=float(delta),
                                sample_rate=float(q), steps=int(T_steps), accountant="rdp")


@torch.no_grad()
def evaluate(model, test_ds, device, is_emnist=False, is_clip=False):
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    model.eval(); correct = total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        if is_emnist: y = y - 1
        correct += (model(x.float()).argmax(1) == y).sum().item(); total += y.shape[0]
    return correct / total


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_run(run_id, cfg, seed, device, data_root, cache_dir,
              out_dir, log_dir, ckpt_dir, K_override=None):
    K = K_override if K_override is not None else cfg.get("K", 1)
    tag = (f"p17_{run_id}_{cfg['mech']}_{cfg['dataset']}_{cfg['regime']}"
           f"_eps{cfg['eps']:.0f}_seed{seed}")
    csv_path   = os.path.join(out_dir,  f"{tag}.csv")
    ckpt_path  = os.path.join(ckpt_dir, f"{tag}_ckpt.pt")
    final_path = os.path.join(out_dir,  f"{tag}_final.pt")
    stats_path = os.path.join(log_dir,  f"{tag}_stats.npz")
    meta_path  = os.path.join(log_dir,  f"{tag}_meta.npz")

    if all(os.path.exists(p) for p in [final_path, csv_path, stats_path, meta_path]):
        print(f"[P17] {tag}: already done, skipping.")
        return

    print(f"\n[P17] === {run_id} (seed={seed}) ===")
    print(f"  dataset={cfg['dataset']} regime={cfg['regime']} eps={cfg['eps']} "
          f"batch={cfg['batch']} K={K} seed={seed}")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    regime = cfg["regime"]; dataset = cfg["dataset"]; mech = cfg["mech"]
    eps = cfg["eps"]; batch = cfg["batch"]
    is_clip = regime == "R3"; is_emnist = "emnist" in dataset

    # Build dataset
    priv_feats_all = None; priv_labels_all = None
    if is_clip:
        (pub_feats, pub_labels, priv_ds, test_ds,
         pub_x, pub_y, num_classes, priv_idx, tier_labels, delta,
         priv_feats_all, priv_labels_all) = build_dataset_clip(dataset, data_root, cache_dir, device, seed=42)
    else:
        (pub_ds, priv_ds, test_ds, pub_x, pub_y, num_classes,
         priv_idx, tier_labels, delta) = build_dataset_wrn(dataset, data_root, seed=42)

    n_priv  = len(priv_ds)
    epochs  = get_epochs(regime, dataset)
    steps_per_epoch = max(1, n_priv // batch)
    T_steps = epochs * steps_per_epoch
    q       = batch / n_priv
    sigma   = calibrate_sigma(eps, delta, q, T_steps)
    sigma_use = sigma * CLIP_C

    print(f"  n_priv={n_priv}  T={T_steps}  q={q:.5f}  σ_mult={sigma:.4f}  K={K}")

    os.makedirs(log_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    model    = make_model(regime, num_classes, dataset).to(device)
    d_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {d_params:,}")

    pub_x_dev = pub_x.to(device).float()
    pub_y_dev = pub_y.to(device).long()
    if regime == "R2":
        _pretrain(model, pub_x_dev, pub_y_dev, device)

    # Accumulated accounting statistics
    sum_norm2      = np.zeros(n_priv, dtype=np.float64)
    sum_reduction_k= np.zeros((n_priv, RANK), dtype=np.float64)
    n_accounted    = 0
    eigval_history = []   # list of [RANK] arrays

    optimizer  = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 1; step_global = 0; best_acc = 0.0

    if os.path.exists(ckpt_path):
        print(f"  [resume] Loading: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        rng_state = ckpt["rng_state"]
        if not isinstance(rng_state, torch.ByteTensor):
            rng_state = torch.ByteTensor(rng_state.tolist() if hasattr(rng_state, 'tolist') else list(rng_state))
        torch.set_rng_state(rng_state)
        np.random.set_state(ckpt["np_rng_state"])
        random.setstate(ckpt["py_rng_state"])
        start_epoch      = ckpt["epoch"] + 1
        step_global      = ckpt["step_global"]
        best_acc         = ckpt["best_acc"]
        sum_norm2[:]     = ckpt["sum_norm2"]
        sum_reduction_k[:] = ckpt["sum_reduction_k"]
        n_accounted      = int(ckpt["n_accounted"])
        eigval_history   = list(ckpt.get("eigval_history", []))
        print(f"  [resume] epoch={ckpt['epoch']}  step={step_global}  n_accounted={n_accounted}")

    # Save meta before training (certify needs this even if run dies)
    np.savez(meta_path, run_id=np.str_(run_id), mech=np.str_(mech),
             dataset=np.str_(dataset), regime=np.str_(regime),
             n_priv=np.int32(n_priv), batch_size=np.int32(batch),
             epochs=np.int32(epochs), eps=np.float64(eps), delta=np.float64(delta),
             q=np.float64(q), T_steps=np.int32(T_steps), K=np.int32(K),
             sigma_mult=np.float64(sigma), sigma_use=np.float64(sigma_use),
             tier_labels=(tier_labels if tier_labels is not None else np.array([], dtype=np.int32)),
             priv_idx=priv_idx.astype(np.int32))

    csv_mode = "a" if start_epoch > 1 and os.path.exists(csv_path) else "w"
    csv_file = open(csv_path, csv_mode, newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "test_acc", "lr", "n_accounted"])
    if csv_mode == "w": writer.writeheader()

    priv_loader = DataLoader(priv_ds, batch_size=batch, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train(); total_loss = 0.0; n_batches = 0

        for batch_data in priv_loader:
            x, y, _ = batch_data[0], batch_data[1], batch_data[2]
            optimizer.zero_grad(set_to_none=True)

            # DP-SGD step
            flat_g = _collect_and_update(model, x, y, sigma_use, d_params, device, regime)
            _set_grads(model, flat_g)
            optimizer.step()

            # Accounting step (every K-th step)
            if step_global % K == 0:
                model.eval()
                if is_clip:
                    U_r, lam, projections, norms_sq = _eigenpairs_r3(
                        model, priv_feats_all, priv_labels_all, q, RANK, device)
                else:
                    U_r, lam, projections, norms_sq = _eigenpairs_chunked(
                        model, priv_ds, q, RANK, device)

                _accumulate_step(
                    projections.cpu(), norms_sq.cpu(), lam,
                    sigma_use, sum_norm2, sum_reduction_k)
                n_accounted  += 1
                eigval_history.append(lam.cpu().numpy().astype(np.float32))
                print(f"    [acct] step={step_global}  λ_max={lam[0].item():.4g}  "
                      f"λ_r={lam[-1].item():.4g}  n_accounted={n_accounted}")
                del U_r, lam, projections, norms_sq; torch.cuda.empty_cache()
                model.train()

            with torch.no_grad():
                b_sub = min(x.shape[0], 64)
                out = model(x[:b_sub].to(device).float() if is_clip else x[:b_sub].to(device))
                total_loss += F.cross_entropy(out, y[:b_sub].to(device)).item()
            n_batches += 1; step_global += 1

        scheduler.step()
        test_acc = evaluate(model, test_ds, device, is_emnist=is_emnist, is_clip=is_clip)
        cur_lr   = optimizer.param_groups[0]["lr"]
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}_best.pt"))

        row = {"epoch": epoch, "train_loss": f"{total_loss/max(n_batches,1):.4f}",
               "test_acc": f"{test_acc:.4f}", "lr": f"{cur_lr:.6f}",
               "n_accounted": n_accounted}
        writer.writerow(row); csv_file.flush()
        print(f"  ep {epoch:3d}/{epochs}  acc={test_acc:.4f}  best={best_acc:.4f}"
              f"  n_accounted={n_accounted}")

        # Checkpoint every epoch
        eigval_arr = np.array(eigval_history, dtype=np.float32) if eigval_history else np.empty((0, RANK), dtype=np.float32)
        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "rng_state": torch.get_rng_state(), "np_rng_state": np.random.get_state(),
            "py_rng_state": random.getstate(), "step_global": step_global, "best_acc": best_acc,
            "sum_norm2": sum_norm2.copy(), "sum_reduction_k": sum_reduction_k.copy(),
            "n_accounted": n_accounted, "eigval_history": eigval_arr,
        }, ckpt_path)

    csv_file.close()
    torch.save(model.state_dict(), final_path)

    eigval_arr = np.array(eigval_history, dtype=np.float32) if eigval_history else np.empty((0, RANK), dtype=np.float32)

    # Save accumulated statistics
    np.savez_compressed(stats_path,
        sum_norm2=sum_norm2, sum_reduction_k=sum_reduction_k,
        n_accounted=np.int32(n_accounted), T_steps=np.int32(step_global),
        K=np.int32(K), eigval_history=eigval_arr)

    # Update meta with final T_steps
    meta_d = dict(np.load(meta_path, allow_pickle=True))
    meta_d["T_steps"] = np.int32(step_global)
    np.savez(meta_path, **meta_d)

    if os.path.exists(ckpt_path): os.remove(ckpt_path)
    torch.cuda.empty_cache()
    print(f"[P17] {run_id} seed={seed} done — acc={test_acc:.4f}  best={best_acc:.4f}"
          f"  n_accounted={n_accounted}/{step_global}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 17 v4 training")
    parser.add_argument("--run",    type=str, default=None)
    parser.add_argument("--block",  type=str, default=None, choices=list(BLOCK_ORDER))
    parser.add_argument("--tier",   type=int, default=None, choices=[1, 2])
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--seed",   type=int, default=None)
    parser.add_argument("--K",      type=int, default=None,
                        help="Sparse accounting interval (default: per run matrix)")
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--out_dir",   type=str, default=os.path.join(RESULTS_DIR, "train"))
    parser.add_argument("--log_dir",   type=str, default=os.path.join(RESULTS_DIR, "logs"))
    parser.add_argument("--ckpt_dir",  type=str, default=os.path.join(RESULTS_DIR, "checkpoints"))
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[P17] Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.run:
        run_ids = [args.run]
    elif args.block:
        run_ids = BLOCK_ORDER[args.block]
    elif args.tier:
        run_ids = [rid for rid, c in RUN_MATRIX.items() if c["tier"] == args.tier]
    elif args.all:
        run_ids = [rid for blk in "ABCD" for rid in BLOCK_ORDER.get(blk, [])]
    else:
        print("[P17] No run specified. Running Block A (CLIP). Use --all for full matrix.")
        run_ids = BLOCK_ORDER["A"]

    for run_id in run_ids:
        if run_id not in RUN_MATRIX:
            print(f"[P17] Unknown run: {run_id}"); continue
        cfg = RUN_MATRIX[run_id]
        seeds = [args.seed] if args.seed is not None else list(range(cfg["n_seeds"]))
        for seed in seeds:
            train_run(run_id, cfg, seed, device, args.data_root, args.cache_dir,
                      args.out_dir, args.log_dir, args.ckpt_dir, K_override=args.K)


if __name__ == "__main__":
    main()
