#!/usr/bin/env python3
"""
Phase 17 — Training: Corrected Composition and Loewner Verification
====================================================================

Three changes from Phase 16:

  1. COMPOSITION CORRECTION: accumulate per-sample gradient statistics for ALL
     n examples at every training step (all-example logging), not only for the
     sampled batch.  This lets the certify script charge every step at the
     actual per-instance rate instead of conditioning on the sampling outcome.

  2. LOEWNER VERIFICATION: inline Loewner check at N_LOEWNER_CHECKS=60 evenly-
     spaced steps.  Checks λ_min(Σ_t − Σ̂_r) ≥ 0 in the rank-r subspace
     (always) and also via full Lanczos on GPU for R3 (CLIP, d=5120).
     Results and the surviving κ logged to JSON per run.

  3. EIGENVALUE SCALING: λ̂_k = q(1−q)·n_priv·κ·λ_k^pub.
     κ starts at KAPPA_INIT=1.0 and shrinks by KAPPA_SHRINK=0.9 at each
     Loewner check step until the subspace check passes.  Global minimum
     κ saved to meta for the certify script.

Accumulated statistics saved per example:
  all_sum_gn2   : [n_priv]       — Σ_t ‖ḡ_{i,t}‖²  over logged steps
  all_sum_gproj2: [n_priv, rank] — Σ_t (ḡ_{i,t}ᵀuₖ)² over logged steps
  n_logged      : [n_priv]       — number of steps logged per example

Usage
-----
  python experiments/exp_p17_train.py --block A --gpu 0
  python experiments/exp_p17_train.py --run C3 --seed 0 --gpu 0
  python experiments/exp_p17_train.py --all --gpu 0
  python experiments/exp_p17_train.py --run C3 --no_all_example_logging --gpu 0
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
# Run matrix (Phase 17)
# ---------------------------------------------------------------------------

RUN_MATRIX = {
    # Tier 1: 5 seeds — R1 cold start
    "A1": dict(dataset="cifar10",       regime="R1", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5, tier=1),
    # Tier 1: 5 seeds — R2 warm start, ε sweep + LT + EMNIST
    "B1": dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=1.0, batch=5000,  n_seeds=5, tier=1),
    "B2": dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=2.0, batch=5000,  n_seeds=5, tier=1),
    "B3": dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=4.0, batch=5000,  n_seeds=5, tier=1),
    "B4": dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5, tier=1),
    "B5": dict(dataset="cifar10_lt50",  regime="R2", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5, tier=1),
    "B6": dict(dataset="cifar10_lt100", regime="R2", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5, tier=1),
    "B7": dict(dataset="emnist",        regime="R2", mech="vanilla", eps=8.0, batch=10000, n_seeds=5, tier=1),
    # Tier 1: 5 seeds — R3 CLIP linear probe, ε sweep + LT
    "C1": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=1.0, batch=5000,  n_seeds=5, tier=1),
    "C2": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=2.0, batch=5000,  n_seeds=5, tier=1),
    "C3": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5, tier=1),
    "C4": dict(dataset="cifar10_lt50",  regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5, tier=1),
    "C5": dict(dataset="cifar10_lt100", regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5, tier=1),
    # Tier 2: 3 seeds — CLIP batch sweep
    "D1": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=1000,  n_seeds=3, tier=2),
    "D2": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=3, tier=2),
    "D3": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=10000, n_seeds=3, tier=2),
    "D4": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=25000, n_seeds=3, tier=2),
}

BLOCK_ORDER = {
    "A": ["C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4"],  # CLIP first (fastest)
    "B": ["A1", "B4", "B5", "B6"],
    "C": ["B1", "B2", "B3"],
    "D": ["B7"],
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_C           = 1.0
RANK_V           = 100
N_PUB            = 2000
LR               = 0.1
MOMENTUM         = 0.9
WEIGHT_DECAY     = 5e-4
PRETRAIN_EPOCHS  = 50
PRETRAIN_LR      = 0.01
PUB_BATCH        = 256
GRAD_CHUNK       = 64
CLIP_CHUNK       = 256
ALL_EX_CHUNK     = 256        # chunk size for all-example sweep
N_LOEWNER_CHECKS = 60         # evenly-spaced Loewner checks per training run
LOEWNER_LANCZOS_ITERS = 50    # power-iteration steps (R3 full check)
KAPPA_INIT       = 1.0
KAPPA_SHRINK     = 0.9
KAPPA_MIN        = 0.1
DATA_ROOT        = "./data"
CACHE_DIR        = "./data/clip_features"
RESULTS_DIR      = "./results/exp_p17"

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
# Dataset helpers  (identical to p16)
# ---------------------------------------------------------------------------

class _IndexedSubset(Dataset):
    def __init__(self, base_ds, indices):
        self.base    = base_ds
        self.indices = np.asarray(indices)
    def __len__(self):  return len(self.indices)
    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, y, int(self.indices[i])


class _EMNISTIndexedSubset(Dataset):
    def __init__(self, base_ds, indices):
        self.base    = base_ds
        self.indices = np.asarray(indices)
    def __len__(self):  return len(self.indices)
    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, int(y) - 1, int(self.indices[i])


class _FeatureDataset(Dataset):
    def __init__(self, features, labels, global_indices):
        self.features = features
        self.labels   = labels
        self.indices  = global_indices
    def __len__(self):  return len(self.labels)
    def __getitem__(self, i):
        return self.features[i], int(self.labels[i]), int(self.indices[i])


def _cifar10_raw(data_root, augment):
    mean = (0.4914, 0.4822, 0.4465); std = (0.2023, 0.1994, 0.2010)
    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize(mean, std)]) if augment else \
         T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf)


def _cifar10_test(data_root):
    mean = (0.4914, 0.4822, 0.4465); std = (0.2023, 0.1994, 0.2010)
    return torchvision.datasets.CIFAR10(root=data_root, train=False, download=True,
        transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]))


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
    is_lt     = "lt" in dataset_name
    lt_ir     = 50 if "lt50" in dataset_name else (100 if "lt100" in dataset_name else 1)
    is_emnist = "emnist" in dataset_name

    if is_emnist:
        train_aug   = _emnist_raw(data_root, augment=True)
        train_noaug = _emnist_raw(data_root, augment=False)
        test_ds     = _emnist_test(data_root)
        raw_targets = np.array(train_aug.targets) - 1
        all_idx     = np.arange(len(train_aug))
        num_classes = 26
        pub_idx, priv_idx = make_public_private_split(all_idx, raw_targets, public_frac=0.1, seed=seed)
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
        lt_idx       = make_cifar10_lt_indices(full_targets, lt_ir, seed=seed) if is_lt else all_idx
        lt_targets   = full_targets[lt_idx]
        pub_idx, priv_idx = make_public_private_split(lt_idx, lt_targets, public_frac=0.1, seed=seed)
        rng     = np.random.default_rng(seed)
        pub_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]
        pub_ds  = Subset(train_noaug, pub_use.tolist())
        priv_ds = _IndexedSubset(train_aug, priv_idx)
        tier_labels = np.array([class_to_tier(c) for c in full_targets[priv_idx]],
                                dtype=np.int32) if is_lt else None
        pub_x = torch.stack([pub_ds[i][0] for i in range(len(pub_ds))])
        pub_y = torch.tensor([int(full_targets[pub_use[i]]) for i in range(len(pub_use))],
                              dtype=torch.long)
        return pub_ds, priv_ds, test_ds, pub_x, pub_y, 10, priv_idx, tier_labels, 1e-5


# ---------------------------------------------------------------------------
# CLIP feature extraction  (identical to p16)
# ---------------------------------------------------------------------------

def _load_or_extract_clip_features(dataset_name, data_root, cache_dir, device):
    os.makedirs(cache_dir, exist_ok=True)
    feat_path  = os.path.join(cache_dir, "cifar10_clip_train.pt")
    lab_path   = os.path.join(cache_dir, "cifar10_clip_train_labels.pt")
    tfeat_path = os.path.join(cache_dir, "cifar10_clip_test.pt")
    tlab_path  = os.path.join(cache_dir, "cifar10_clip_test_labels.pt")

    if all(os.path.exists(p) for p in [feat_path, lab_path, tfeat_path, tlab_path]):
        print(f"  [CLIP] Loading cached features from {feat_path}")
        return (torch.load(feat_path,  map_location="cpu"),
                torch.load(lab_path,   map_location="cpu"),
                torch.load(tfeat_path, map_location="cpu"),
                torch.load(tlab_path,  map_location="cpu"))

    print("  [CLIP] Extracting features...")
    try:
        import open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        clip_model = clip_model.to(device).eval()
    except ImportError:
        import clip as openai_clip
        clip_model, preprocess = openai_clip.load("ViT-B/32", device=device)
        clip_model.eval()

    clip_mean = (0.48145466, 0.4578275,  0.40821073)
    clip_std  = (0.26862954, 0.26130258, 0.27577711)
    clip_tf   = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                            T.CenterCrop(224), T.ToTensor(), T.Normalize(clip_mean, clip_std)])

    def _extract(ds_obj):
        feats, labs = [], []
        for batch in DataLoader(ds_obj, batch_size=256, shuffle=False, num_workers=4):
            imgs = batch[0].to(device)
            with torch.no_grad():
                f = clip_model.encode_image(imgs)
            feats.append(f.cpu().float()); labs.append(batch[1])
        return torch.cat(feats), torch.cat(labs)

    train_raw = torchvision.datasets.CIFAR10(root=data_root, train=True,  download=True, transform=clip_tf)
    test_raw  = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=clip_tf)
    train_feats, train_labels = _extract(train_raw)
    test_feats,  test_labels  = _extract(test_raw)

    torch.save(train_feats,  feat_path);  torch.save(train_labels, lab_path)
    torch.save(test_feats,   tfeat_path); torch.save(test_labels,  tlab_path)
    del clip_model; torch.cuda.empty_cache()
    return train_feats, train_labels, test_feats, test_labels


def build_dataset_clip(dataset_name, data_root, cache_dir, device, seed=42):
    is_lt = "lt" in dataset_name
    lt_ir = 50 if "lt50" in dataset_name else (100 if "lt100" in dataset_name else 1)
    train_feats, train_labels, test_feats, test_labels = \
        _load_or_extract_clip_features("cifar10", data_root, cache_dir, device)

    full_targets = train_labels.numpy()
    all_idx      = np.arange(len(full_targets))
    lt_idx       = make_cifar10_lt_indices(full_targets, lt_ir, seed=seed) if is_lt else all_idx
    lt_targets   = full_targets[lt_idx]
    pub_idx, priv_idx = make_public_private_split(lt_idx, lt_targets, public_frac=0.1, seed=seed)
    rng     = np.random.default_rng(seed)
    pub_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]

    pub_feats  = train_feats[pub_use]
    pub_labels = torch.tensor(full_targets[pub_use], dtype=torch.long)
    priv_feats = train_feats[priv_idx]
    priv_labels= torch.tensor(full_targets[priv_idx], dtype=torch.long)
    priv_global= torch.tensor(priv_idx, dtype=torch.long)
    priv_ds    = _FeatureDataset(priv_feats, priv_labels, priv_global)
    tier_labels= np.array([class_to_tier(c) for c in full_targets[priv_idx]],
                           dtype=np.int32) if is_lt else None
    test_ds    = _FeatureDataset(test_feats, test_labels, torch.arange(len(test_labels)))

    # Return full train features/labels for all-example sweep
    all_feats  = train_feats[priv_idx]    # [n_priv, 512] — private split features
    all_labels = torch.tensor(full_targets[priv_idx], dtype=torch.long)

    return (pub_feats, pub_labels, priv_ds, test_ds,
            pub_feats, pub_labels, 10, priv_idx, tier_labels, 1e-5,
            all_feats, all_labels)


# ---------------------------------------------------------------------------
# Models  (identical to p16)
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
    def forward(self, x): return self.fc(x)


def make_model(regime, num_classes, dataset_name):
    if regime == "R3":
        return LinearHead(num_classes, feature_dim=512)
    if "emnist" in dataset_name:
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
                return self.fc(F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1))
        return SimpleCNN(num_classes)
    return ResNet20(num_classes=num_classes, n_groups=16)


# ---------------------------------------------------------------------------
# Per-sample gradients
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
    h = h_chunk.to(device).float()
    y = y_chunk.to(device).long()
    W = model.fc.weight; b = model.fc.bias
    with torch.no_grad():
        logits = h @ W.t() + b
        p      = torch.softmax(logits, dim=1)
        ey     = torch.zeros_like(p)
        ey.scatter_(1, y.unsqueeze(1), 1.0)
        delta  = p - ey
        g_W    = (delta.unsqueeze(2) * h.unsqueeze(1)).reshape(h.shape[0], -1)
        g_b    = delta
    return torch.cat([g_W, g_b], dim=1)


def _set_grads(model, flat_grad):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat_grad[offset:offset+n].view(p.shape).clone()
        offset += n


# ---------------------------------------------------------------------------
# Subspace (PCA of public clipped gradients)
# ---------------------------------------------------------------------------

def _compute_subspace(model, pub_x, pub_y, device, rank=RANK_V, regime="R2"):
    print(f"  [subspace] Computing rank-{rank} PCA (regime={regime})...")
    model.eval()
    parts = []
    N = pub_x.shape[0]
    c = CLIP_CHUNK if regime == "R3" else GRAD_CHUNK
    for i in range(0, N, c):
        xc, yc = pub_x[i:i+c], pub_y[i:i+c]
        g = (_per_sample_grads_linear(model, xc, yc, device) if regime == "R3"
             else _per_sample_grads_vmap(model, xc, yc, device))
        norms = g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        parts.append((g * (CLIP_C / norms).clamp(max=1.0)).cpu())
        del g; torch.cuda.empty_cache()
    G   = torch.cat(parts, dim=0).float()
    k   = min(rank, G.shape[0] - 1, G.shape[1] - 1)
    _, S, V = torch.svd_lowrank(G, q=k, niter=6)
    V   = V[:, :k].cpu()
    S   = S[:k].cpu()
    lambdas = (S ** 2 / G.shape[0]).numpy().astype(np.float64)
    print(f"  [subspace] V={V.shape}  λ_max={lambdas[0]:.4g}  λ_min={lambdas[-1]:.4g}")
    del G; torch.cuda.empty_cache()
    return V, lambdas


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
            idx = perm[i:i+PUB_BATCH]
            opt.zero_grad()
            F.cross_entropy(model(pub_x[idx].to(device)), pub_y[idx].to(device)).backward()
            opt.step()
        sch.step()
    print(f"  [pretrain] done ({PRETRAIN_EPOCHS} ep)")


# ---------------------------------------------------------------------------
# Privacy calibration
# ---------------------------------------------------------------------------

def calibrate_sigma(eps, delta, q, T_steps):
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(target_epsilon=float(eps), target_delta=float(delta),
                                sample_rate=float(q), steps=int(T_steps), accountant="rdp")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, test_ds, device, is_emnist=False, is_clip=False):
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    model.eval(); correct = total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        if is_emnist: y = y - 1
        correct += (model(x.float()).argmax(1) == y).sum().item()
        total   += y.shape[0]
    return correct / total


# ---------------------------------------------------------------------------
# Loewner verification (inline, during training)
# ---------------------------------------------------------------------------

def _loewner_check_subspace(G_proj_gpu, lambdas_hat_gpu, q):
    """
    Check λ_min(q(1−q)·Gᵀ G − diag(λ̂)) ≥ 0 in the rank-r subspace.

    G_proj_gpu : [B, r] tensor — per-sample projections ḡⱼᵀ Uᵣ
    lambdas_hat: [r] tensor   — λ̂ₖ = q(1−q)·n_priv·κ·λₖ^pub
    Returns (lambda_min: float, passed: bool)
    """
    with torch.no_grad():
        cov_sub = q * (1.0 - q) * G_proj_gpu.t() @ G_proj_gpu   # [r, r]
        diff    = cov_sub - torch.diag(lambdas_hat_gpu)           # [r, r]
        eigs    = torch.linalg.eigvalsh(diff)                     # ascending
    lam_min = float(eigs[0].item())
    return lam_min, lam_min >= 0.0


def _loewner_check_full_r3(gc_clip_batch, U_r_gpu, lambdas_hat_gpu, q, n_iter=LOEWNER_LANCZOS_ITERS):
    """
    Full power-iteration Loewner check for R3 (d=5120, small enough to hold all gradients).

    gc_clip_batch : [B, d] GPU tensor — clipped gradients for the current batch
    Returns (lambda_max_of_Sigmahat_minus_Sigmat: float, passed: bool)
    Passed ⟺ λ_max ≤ 0, i.e., Σ_t ≽ Σ̂_r.
    """
    B = gc_clip_batch.shape[0]
    d = gc_clip_batch.shape[1]
    v = F.normalize(torch.randn(d, device=gc_clip_batch.device), dim=0)

    def matvec(v_):
        coords = U_r_gpu.t() @ v_                                    # [r]
        lhs    = U_r_gpu @ (lambdas_hat_gpu * coords)                # [d]  Σ̂_r v
        dots   = gc_clip_batch @ v_                                  # [B]
        rhs    = q * (1.0 - q) * (gc_clip_batch.t() @ dots)         # [d]  Σ_t v
        return lhs - rhs

    with torch.no_grad():
        for _ in range(n_iter):
            v = matvec(v)
            vnorm = v.norm()
            if vnorm < 1e-12: break
            v = v / vnorm
        Av = matvec(v)
        lam_max = float((v @ Av).item())
    return lam_max, lam_max <= 0.0


def run_loewner_check(G_proj_gpu, gc_clip_batch, U_r_gpu, lambdas_pub_gpu,
                      q, n_priv, kappa, is_clip):
    """
    Run Loewner check and shrink κ if it fails.
    Returns (kappa_final, lam_min_subspace, lam_max_full, passed_subspace, passed_full).
    """
    kap = kappa
    lam_sub = None; lam_full = None
    passed_sub = False; passed_full = None

    while kap >= KAPPA_MIN:
        lambdas_hat = q * (1.0 - q) * n_priv * kap * lambdas_pub_gpu   # [r]

        lam_sub, passed_sub = _loewner_check_subspace(G_proj_gpu, lambdas_hat, q)

        if passed_sub:
            # Also run full check for R3 (cheap, d=5120)
            if is_clip and gc_clip_batch is not None:
                lam_full, passed_full = _loewner_check_full_r3(
                    gc_clip_batch, U_r_gpu, lambdas_hat, q)
            break
        kap = kap * KAPPA_SHRINK

    # If we exhausted kappa and still failing, report the last values
    if not passed_sub and kap < KAPPA_MIN:
        kap = KAPPA_MIN
        lambdas_hat = q * (1.0 - q) * n_priv * kap * lambdas_pub_gpu
        lam_sub, passed_sub = _loewner_check_subspace(G_proj_gpu, lambdas_hat, q)
        if is_clip and gc_clip_batch is not None:
            lam_full, passed_full = _loewner_check_full_r3(gc_clip_batch, U_r_gpu, lambdas_hat, q)

    return kap, lam_sub, lam_full, passed_sub, passed_full


# ---------------------------------------------------------------------------
# All-example gradient sweep (online accumulation)
# ---------------------------------------------------------------------------

def sweep_all_examples(model, priv_data_all, V_gpu, device, regime,
                       all_sum_gn2, all_sum_gproj2, n_logged,
                       priv_idx_map):
    """
    Compute per-sample clipped gradient statistics for ALL n_priv examples at
    the current model state and accumulate into all_sum_gn2/all_sum_gproj2.

    priv_data_all: for R3, tuple (all_feats, all_labels) [n_priv, d]
                   for R1/R2, an _IndexedSubset dataset (accessed by position)
    priv_idx_map : [n_priv] int array — maps position → example index (same as priv_idx)
    """
    rank = V_gpu.shape[1]
    model.eval()

    if regime == "R3":
        all_feats, all_labels = priv_data_all
        N = len(all_feats)
        for i in range(0, N, ALL_EX_CHUNK):
            h = all_feats[i:i+ALL_EX_CHUNK].to(device).float()
            y = all_labels[i:i+ALL_EX_CHUNK].to(device).long()
            with torch.no_grad():
                gc     = _per_sample_grads_linear(model, h, y, device)
                norms  = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
                gc_c   = gc * (CLIP_C / norms).clamp(max=1.0)
                gnorms = gc_c.norm(dim=1)           # [chunk]
                coords = gc_c @ V_gpu               # [chunk, rank]
                gn2_c  = gnorms.cpu().numpy().astype(np.float64) ** 2
                gp2_c  = (coords ** 2).cpu().numpy().astype(np.float64)
            all_sum_gn2[i:i+len(h)]    += gn2_c
            all_sum_gproj2[i:i+len(h)] += gp2_c
            n_logged[i:i+len(h)]       += 1
            del gc, gc_c, coords; torch.cuda.empty_cache()
    else:
        # R1/R2: iterate over priv_data_all which is a Dataset yielding (x, y, global_idx)
        # We access it by position (0..n_priv-1)
        N = len(priv_data_all)
        for i in range(0, N, ALL_EX_CHUNK):
            batch = [priv_data_all[j] for j in range(i, min(i + ALL_EX_CHUNK, N))]
            xb = torch.stack([b[0] for b in batch])
            yb = torch.tensor([b[1] for b in batch], dtype=torch.long)
            with torch.no_grad():
                gc     = _per_sample_grads_vmap(model, xb, yb, device)
                norms  = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
                gc_c   = gc * (CLIP_C / norms).clamp(max=1.0)
                gnorms = gc_c.norm(dim=1)
                coords = gc_c @ V_gpu
                gn2_c  = gnorms.cpu().numpy().astype(np.float64) ** 2
                gp2_c  = (coords ** 2).cpu().numpy().astype(np.float64)
            actual = len(batch)
            all_sum_gn2[i:i+actual]    += gn2_c
            all_sum_gproj2[i:i+actual] += gp2_c
            n_logged[i:i+actual]       += 1
            del gc, gc_c, coords; torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Sampled-step gradient collection (returns sum_g + logging data for Loewner)
# ---------------------------------------------------------------------------

def collect_grads_and_log(model, x, y, ex_idx, V_gpu, device, regime,
                          sampled_sum_gn2, sampled_sum_gproj2, n_sampled,
                          priv_pos_map):
    """
    Collect per-sample gradients for sampled batch, clip, accumulate statistics.
    Also returns:
      G_proj_gpu  : [B, rank] float32 GPU tensor — projections (for Loewner check)
      gc_clip_all : [B, d] float32 GPU tensor — full clipped grads (R3 only, else None)

    priv_pos_map: dict {global_idx → local_position} in [0, n_priv)
    """
    B     = x.shape[0]
    d     = sum(p.numel() for p in model.parameters())
    sum_g = torch.zeros(d, device=device)
    c     = CLIP_CHUNK if regime == "R3" else GRAD_CHUNK
    is_r3 = regime == "R3"

    G_proj_parts  = []
    gc_clip_parts = [] if is_r3 else None  # only store for R3 (d=5120)

    for ci in range(0, B, c):
        xc    = x[ci:ci+c]
        yc    = y[ci:ci+c]
        idx_c = ex_idx[ci:ci+c]

        if is_r3:
            gc = _per_sample_grads_linear(model, xc, yc, device)
        else:
            gc = _per_sample_grads_vmap(model, xc, yc, device)

        norms   = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
        gc_clip = gc * (CLIP_C / norms).clamp(max=1.0)
        sum_g  += gc_clip.sum(0)

        with torch.no_grad():
            coords  = gc_clip @ V_gpu        # [chunk, rank]
            gnorms  = gc_clip.norm(dim=1)    # [chunk]
            G_proj_parts.append(coords.detach())
            if is_r3:
                gc_clip_parts.append(gc_clip.detach())

        # Accumulate sampled-step statistics (indexed by local position)
        gn2_c  = gnorms.cpu().numpy().astype(np.float64) ** 2
        gp2_c  = (coords ** 2).cpu().numpy().astype(np.float64)
        for ii, gidx in enumerate(idx_c.tolist()):
            lpos = priv_pos_map.get(int(gidx), -1)
            if lpos >= 0:
                sampled_sum_gn2[lpos]    += gn2_c[ii]
                sampled_sum_gproj2[lpos] += gp2_c[ii]
                n_sampled[lpos]          += 1

        del gc, gc_clip, coords, gnorms
        torch.cuda.empty_cache()

    G_proj_gpu  = torch.cat(G_proj_parts, dim=0)       # [B, rank]
    gc_clip_all = torch.cat(gc_clip_parts, dim=0) if is_r3 else None
    return sum_g, G_proj_gpu, gc_clip_all


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_run(run_id, cfg, seed, device, data_root, cache_dir,
              out_dir, log_dir, ckpt_dir, all_example_logging=True):
    tag       = (f"p17_{run_id}_{cfg['mech']}_{cfg['dataset']}_{cfg['regime']}"
                 f"_eps{cfg['eps']:.0f}_seed{seed}")
    csv_path       = os.path.join(out_dir,  f"{tag}.csv")
    ckpt_path      = os.path.join(ckpt_dir, f"{tag}_ckpt.pt")
    final_path     = os.path.join(out_dir,  f"{tag}_final.pt")
    stats_path     = os.path.join(log_dir,  f"{tag}_stats.npz")
    loewner_path   = os.path.join(log_dir,  f"{tag}_loewner.json")
    meta_path      = os.path.join(log_dir,  f"{tag}_meta.npz")

    if (os.path.exists(final_path) and os.path.exists(csv_path)
            and os.path.exists(stats_path) and os.path.exists(meta_path)):
        print(f"[P17] {tag}: already done, skipping.")
        return

    print(f"\n[P17] === {run_id} (seed={seed}) ===")
    print(f"  dataset={cfg['dataset']} regime={cfg['regime']} mech={cfg['mech']} "
          f"eps={cfg['eps']} batch={cfg['batch']} seed={seed}")
    print(f"  all_example_logging={all_example_logging}")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    regime   = cfg["regime"]
    dataset  = cfg["dataset"]
    mech     = cfg["mech"]
    eps      = cfg["eps"]
    batch    = cfg["batch"]
    is_clip  = regime == "R3"
    is_emnist= "emnist" in dataset

    # Build dataset
    all_feats_np = None; all_labels_np = None  # R3 all-example data
    priv_data_all = None                        # R1/R2 all-example data (the priv dataset)

    if is_clip:
        (pub_feats, pub_labels, priv_ds, test_ds,
         pub_x, pub_y, num_classes, priv_idx, tier_labels, delta,
         all_feats, all_labels) = build_dataset_clip(dataset, data_root, cache_dir, device, seed=42)
        all_feats_np  = all_feats   # [n_priv, 512]
        all_labels_np = all_labels  # [n_priv]
    else:
        (pub_ds, priv_ds, test_ds, pub_x, pub_y, num_classes,
         priv_idx, tier_labels, delta) = build_dataset_wrn(dataset, data_root, seed=42)
        priv_data_all = priv_ds

    n_priv  = len(priv_ds)
    epochs  = get_epochs(regime, dataset)
    steps_per_epoch = max(1, n_priv // batch)
    T_steps = epochs * steps_per_epoch
    q       = batch / n_priv
    sigma   = calibrate_sigma(eps, delta, q, T_steps)
    sigma_use = sigma * CLIP_C

    print(f"  n_priv={n_priv}  T={T_steps}  q={q:.5f}  σ_mult={sigma:.4f}  σ_use={sigma_use:.4f}")

    os.makedirs(log_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Map global example index → local position in [0, n_priv)
    priv_pos_map = {int(priv_idx[i]): i for i in range(len(priv_idx))}

    # Schedule Loewner check steps
    loewner_steps = set(np.linspace(0, T_steps - 1, N_LOEWNER_CHECKS, dtype=int).tolist())

    model     = make_model(regime, num_classes, dataset).to(device)
    d_params  = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {d_params:,}")

    pub_x_dev = pub_x.to(device).float()
    pub_y_dev = pub_y.to(device).long()

    if regime == "R2":
        _pretrain(model, pub_x_dev, pub_y_dev, device)

    pub_x_cpu = pub_x.cpu(); pub_y_cpu = pub_y.cpu()

    # Subspace + eigenvalues
    rank_v = RANK_V
    V_cpu, lambdas_pub = _compute_subspace(model, pub_x_cpu, pub_y_cpu, device,
                                            rank=rank_v, regime=regime)
    V_gpu           = V_cpu.to(device)
    lambdas_pub_gpu = torch.tensor(lambdas_pub, dtype=torch.float32, device=device)

    # Accumulated statistics (per local position in [0, n_priv))
    all_sum_gn2    = np.zeros(n_priv, dtype=np.float64)   # Σ_t ‖ḡ‖² over logged steps
    all_sum_gproj2 = np.zeros((n_priv, rank_v), dtype=np.float64)  # Σ_t (ḡᵀuₖ)²
    n_logged       = np.zeros(n_priv, dtype=np.int32)    # steps logged per example

    # Sampled-step tracking (needed as fallback info for certify when all-ex disabled)
    sampled_sum_gn2    = np.zeros(n_priv, dtype=np.float64)
    sampled_sum_gproj2 = np.zeros((n_priv, rank_v), dtype=np.float64)
    n_sampled          = np.zeros(n_priv, dtype=np.int32)

    # Loewner state
    kappa        = KAPPA_INIT
    loewner_log  = []

    # Save meta BEFORE training (needed for certify even if run dies)
    np.savez(meta_path,
             run_id=np.str_(run_id), mech=np.str_(mech), dataset=np.str_(dataset),
             regime=np.str_(regime), n_priv=np.int32(n_priv), batch_size=np.int32(batch),
             epochs=np.int32(epochs), eps=np.float64(eps), delta=np.float64(delta),
             q=np.float64(q), T_steps=np.int32(T_steps),
             sigma_mult=np.float64(sigma), sigma_use=np.float64(sigma_use),
             lambdas_pub=lambdas_pub,
             kappa_global=np.float64(kappa),    # updated at end
             all_example_logging=np.bool_(all_example_logging),
             tier_labels=(tier_labels if tier_labels is not None else np.array([], dtype=np.int32)),
             priv_idx=priv_idx.astype(np.int32))

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Resume from checkpoint
    start_epoch = 1
    step_global = 0
    best_acc    = 0.0

    if os.path.exists(ckpt_path):
        print(f"  [resume] Loading: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        torch.set_rng_state(ckpt["rng_state"])
        np.random.set_state(ckpt["np_rng_state"])
        random.setstate(ckpt["py_rng_state"])
        start_epoch         = ckpt["epoch"] + 1
        step_global         = ckpt["step_global"]
        best_acc            = ckpt["best_acc"]
        kappa               = float(ckpt.get("kappa", KAPPA_INIT))
        all_sum_gn2[:]      = ckpt["all_sum_gn2"]
        all_sum_gproj2[:]   = ckpt["all_sum_gproj2"]
        n_logged[:]         = ckpt["n_logged"]
        sampled_sum_gn2[:]  = ckpt.get("sampled_sum_gn2", np.zeros(n_priv))
        sampled_sum_gproj2[:] = ckpt.get("sampled_sum_gproj2", np.zeros((n_priv, rank_v)))
        n_sampled[:]        = ckpt.get("n_sampled", np.zeros(n_priv, dtype=np.int32))
        loewner_log         = ckpt.get("loewner_log", [])
        print(f"  [resume] Resumed epoch={ckpt['epoch']}  step={step_global}  kappa={kappa:.4f}")

    # CSV writer
    csv_mode = "a" if start_epoch > 1 and os.path.exists(csv_path) else "w"
    csv_file = open(csv_path, csv_mode, newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "test_acc", "lr", "kappa"])
    if csv_mode == "w": writer.writeheader()

    priv_loader = DataLoader(priv_ds, batch_size=batch, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0; n_batches = 0

        for batch_data in priv_loader:
            x, y, ex_idx = batch_data[0], batch_data[1], batch_data[2]
            optimizer.zero_grad(set_to_none=True)

            # Collect grads + log sampled-step statistics
            sum_g, G_proj_gpu, gc_clip_all = collect_grads_and_log(
                model, x, y, ex_idx, V_gpu, device, regime,
                sampled_sum_gn2, sampled_sum_gproj2, n_sampled, priv_pos_map)

            # DP noise + update
            noise  = torch.randn(d_params, device=device) * sigma_use
            flat_g = (sum_g + noise) / x.shape[0]
            _set_grads(model, flat_g)
            optimizer.step()

            # All-example sweep (accumulate statistics for ALL examples at current θ)
            if all_example_logging:
                with torch.no_grad():
                    sweep_all_examples(
                        model,
                        (all_feats_np, all_labels_np) if is_clip else priv_data_all,
                        V_gpu, device, regime,
                        all_sum_gn2, all_sum_gproj2, n_logged, priv_pos_map)
            # When all_example_logging=False, all_sum_* stays as sampled-step stats
            # (synced at epoch end below; certify adds data-independent fallback).

            # Loewner check at scheduled steps
            if step_global in loewner_steps:
                kappa_new, lam_sub, lam_full, pass_sub, pass_full = run_loewner_check(
                    G_proj_gpu, gc_clip_all, V_gpu, lambdas_pub_gpu,
                    q, n_priv, kappa, is_clip)
                kappa = min(kappa, kappa_new)   # track global minimum
                entry = {"step": int(step_global), "kappa": float(kappa_new),
                         "kappa_global": float(kappa),
                         "lam_min_subspace": float(lam_sub) if lam_sub is not None else None,
                         "lam_max_full": float(lam_full) if lam_full is not None else None,
                         "passed_subspace": bool(pass_sub),
                         "passed_full": bool(pass_full) if pass_full is not None else None}
                loewner_log.append(entry)
                status = "PASS" if pass_sub else "FAIL"
                print(f"    [Loewner] step={step_global}  κ={kappa_new:.4f} (global={kappa:.4f})"
                      f"  λ_min_sub={lam_sub:.4g}  {status}"
                      + (f"  λ_max_full={lam_full:.4g}" if lam_full is not None else ""))

            with torch.no_grad():
                b_sub = min(x.shape[0], 64)
                xb = x[:b_sub].to(device)
                out = model(xb.float() if is_clip else xb)
                total_loss += F.cross_entropy(out, y[:b_sub].to(device)).item()
            n_batches   += 1
            step_global += 1

        scheduler.step()
        test_acc = evaluate(model, test_ds, device, is_emnist=is_emnist, is_clip=is_clip)
        cur_lr   = optimizer.param_groups[0]["lr"]

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}_best.pt"))

        row = {"epoch": epoch, "train_loss": f"{total_loss/max(n_batches,1):.4f}",
               "test_acc": f"{test_acc:.4f}", "lr": f"{cur_lr:.6f}", "kappa": f"{kappa:.6f}"}
        writer.writerow(row); csv_file.flush()

        # When not doing all-example logging, keep all_sum_* in sync with sampled arrays
        if not all_example_logging:
            all_sum_gn2[:]    = sampled_sum_gn2
            all_sum_gproj2[:] = sampled_sum_gproj2
            n_logged[:]       = n_sampled

        n_logged_all = int(n_logged[0]) if all_example_logging else int(n_sampled.max())
        print(f"  ep {epoch:3d}/{epochs}  acc={test_acc:.4f}  best={best_acc:.4f}"
              f"  κ={kappa:.4f}  n_logged_ex={n_logged_all}")

        # Checkpoint (every epoch)
        torch.save({
            "epoch":              epoch,
            "model_state":        model.state_dict(),
            "optimizer_state":    optimizer.state_dict(),
            "scheduler_state":    scheduler.state_dict(),
            "rng_state":          torch.get_rng_state(),
            "np_rng_state":       np.random.get_state(),
            "py_rng_state":       random.getstate(),
            "step_global":        step_global,
            "best_acc":           best_acc,
            "kappa":              kappa,
            "all_sum_gn2":        all_sum_gn2.copy(),
            "all_sum_gproj2":     all_sum_gproj2.copy(),
            "n_logged":           n_logged.copy(),
            "sampled_sum_gn2":    sampled_sum_gn2.copy(),
            "sampled_sum_gproj2": sampled_sum_gproj2.copy(),
            "n_sampled":          n_sampled.copy(),
            "loewner_log":        loewner_log,
        }, ckpt_path)

        # Flush Loewner log every epoch
        with open(loewner_path, "w") as f:
            json.dump(loewner_log, f, indent=2)

    csv_file.close()
    torch.save(model.state_dict(), final_path)

    # Update meta with final kappa
    meta_data = dict(np.load(meta_path, allow_pickle=True))
    meta_data["kappa_global"] = np.float64(kappa)
    np.savez(meta_path, **meta_data)

    # Save final accumulated statistics
    np.savez_compressed(stats_path,
        all_sum_gn2=all_sum_gn2,
        all_sum_gproj2=all_sum_gproj2,
        n_logged=n_logged,
        sampled_sum_gn2=sampled_sum_gn2,
        sampled_sum_gproj2=sampled_sum_gproj2,
        n_sampled=n_sampled,
        T_steps=np.int32(step_global),
        all_example_logging=np.bool_(all_example_logging),
        kappa_global=np.float64(kappa))

    # Save final Loewner log
    with open(loewner_path, "w") as f:
        json.dump(loewner_log, f, indent=2)

    # Remove rolling checkpoint
    if os.path.exists(ckpt_path): os.remove(ckpt_path)

    del V_gpu; torch.cuda.empty_cache()
    print(f"[P17] {run_id} seed={seed} done — acc={test_acc:.4f}  best={best_acc:.4f}"
          f"  κ_global={kappa:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 17 training")
    parser.add_argument("--run",    type=str, default=None, help="Single run ID (e.g. C3)")
    parser.add_argument("--block",  type=str, default=None, choices=list(BLOCK_ORDER))
    parser.add_argument("--tier",   type=int, default=None, choices=[1, 2])
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--seed",   type=int, default=None)
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--no_all_example_logging", action="store_true",
                        help="Disable all-example logging (use sampled-only + data-independent fallback)")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--out_dir",   type=str, default=os.path.join(RESULTS_DIR, "train"))
    parser.add_argument("--log_dir",   type=str, default=os.path.join(RESULTS_DIR, "logs"))
    parser.add_argument("--ckpt_dir",  type=str, default=os.path.join(RESULTS_DIR, "checkpoints"))
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[P17] Device: {device}")

    os.makedirs(args.out_dir,  exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    all_example_logging = not args.no_all_example_logging

    if args.run:
        run_ids = [args.run]
    elif args.block:
        run_ids = BLOCK_ORDER[args.block]
    elif args.tier:
        run_ids = [rid for rid, c in RUN_MATRIX.items() if c["tier"] == args.tier]
    elif args.all:
        run_ids = []
        for blk in "ABCD":
            run_ids.extend(BLOCK_ORDER.get(blk, []))
    else:
        print("[P17] No run specified. Running Block A (CLIP). Use --all for full matrix.")
        run_ids = BLOCK_ORDER["A"]

    for run_id in run_ids:
        if run_id not in RUN_MATRIX:
            print(f"[P17] Unknown run: {run_id}, skipping.")
            continue
        cfg = RUN_MATRIX[run_id]
        seeds = [args.seed] if args.seed is not None else list(range(cfg["n_seeds"]))
        for seed in seeds:
            train_run(run_id, cfg, seed, device, args.data_root, args.cache_dir,
                      args.out_dir, args.log_dir, args.ckpt_dir,
                      all_example_logging=all_example_logging)


if __name__ == "__main__":
    main()
