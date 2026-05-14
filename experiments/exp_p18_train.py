#!/usr/bin/env python3
"""
Phase 18 Training: Nyström PSD-Minorant Direction-Aware DP-SGD
=============================================================

Spec: phase18_spex.md

Logging per accounting step t:
  clipped_norms[i,t]     = ||ḡ_{i,t}||              shape (n, T_acc)
  losses[i,t]            = CE loss for example i      shape (n, T_acc)
  B_matrices[t,:,:]      = Q_t^T Σ_t Q_t             shape (T_acc, r_max, r_max)
  YTY_matrices[t,:,:]    = Y_t^T Y_t                  shape (T_acc, r_max, r_max)
  Y_projections[i,t,:]   = Y_t^T ḡ_{i,t}             shape (n, T_acc, r_max)

where Y_t = Σ_t^full Q_t, Σ_t^full = ρ G_t^T G_t, Q_t = top-r_max eigenvectors.

Usage:
  python experiments/exp_p18_train.py --setting S2 --seed 0 --gpu 0
  python experiments/exp_p18_train.py --setting S1 --seed 0 --gpu 0
  python experiments/exp_p18_train.py --setting S3 --seed 0 --gpu 0
  python experiments/exp_p18_train.py --all_minimal --gpu 0
"""

import os, sys, json, math, argparse, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets import make_public_private_split, make_cifar10_lt_indices
from src.models import WideResNet

import torchvision
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Run matrix (Section 2)
# ---------------------------------------------------------------------------

# Batch sizes are chosen to hold q = batch/n_priv ≈ 1/9 ≈ 0.111 across all settings.
# This keeps σ identical at the same (ε, δ, T) so cross-setting comparisons in Table 5
# are attributable to data distribution, not noise level.
#
# Approximate n_priv per dataset (90% private, 10% public per class):
#   cifar10:       ~45,000  →  batch=5000  q≈0.111  T_train=360 (40ep) / 540 (60ep)
#   cifar10_lt50:  ~12,602  →  batch=1400  q≈0.111  T_train=360 (40ep) / 540 (60ep)
#   cifar10_lt100: ~11,167  →  batch=1200  q≈0.107  T_train=360 (40ep) / 540 (60ep)

# Minimal TMLR-ready set (Section 2.1)
SETTINGS_MINIMAL = {
    "S1": dict(dataset="cifar10",      regime="R3", eps=8.0, batch=5000, n_seeds=3, epochs=40, K=1),
    "S2": dict(dataset="cifar10_lt50", regime="R3", eps=8.0, batch=1400, n_seeds=3, epochs=40, K=1),
    "S3": dict(dataset="cifar10_lt50", regime="R2", eps=8.0, batch=1400, n_seeds=1, epochs=60, K=10),
}

# Optional expanded set (Section 2.2)
SETTINGS_EXPANDED = {
    "S4a": dict(dataset="cifar10",       regime="R3", eps=1.0, batch=5000, n_seeds=3, epochs=40, K=1),
    "S4b": dict(dataset="cifar10",       regime="R3", eps=2.0, batch=5000, n_seeds=3, epochs=40, K=1),
    "S4c": dict(dataset="cifar10",       regime="R3", eps=4.0, batch=5000, n_seeds=3, epochs=40, K=1),
    "S5":  dict(dataset="cifar10_lt100", regime="R3", eps=8.0, batch=1200, n_seeds=3, epochs=40, K=1),
    "S6":  dict(dataset="cifar10",       regime="R2", eps=8.0, batch=5000, n_seeds=1, epochs=60, K=10),
    "S7":  dict(dataset="cifar10_lt100", regime="R2", eps=8.0, batch=1200, n_seeds=1, epochs=60, K=10),
    "S9":  dict(dataset="cifar10",       regime="R1", eps=8.0, batch=5000, n_seeds=1, epochs=60, K=10),
}

ALL_SETTINGS = {**SETTINGS_MINIMAL, **SETTINGS_EXPANDED}

# LiRA targets per setting (Section 6.3)
LIRA_N_TARGETS = {
    "S1": 1000, "S2": 1500, "S3": 300,
    "S4a": 1000, "S4b": 1000, "S4c": 1000, "S5": 1500,
    "S6": 1000, "S7": 1500, "S9": 1000,
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_C      = 1.0
R_MAX       = 200     # Nyström sketch rank (r_max)
CHUNK_R3    = 512     # chunk size for CLIP grad computation
CHUNK_WRN   = 32      # chunk size for WRN vmap (memory-limited)
RSVD_K      = R_MAX + 20  # oversampling for rSVD
DATA_ROOT   = "./data"
CACHE_DIR   = "./data/clip_features"
RUNS_DIR    = "./runs"
N_PUB       = 2000
PRETRAIN_EP = 50
PRETRAIN_LR = 0.01
LR          = 0.1
N_GROUPS    = 16      # GroupNorm groups for WRN

LT_HEAD = {0, 1, 2}
LT_MID  = {3, 4, 5, 6}
LT_TAIL = {7, 8, 9}

def class_to_tier(c):
    if c in LT_HEAD: return 0
    if c in LT_MID:  return 1
    return 2


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class _IndexedSubset(Dataset):
    def __init__(self, base, indices):
        self.base = base; self.indices = np.asarray(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]; return x, y, int(self.indices[i])


class _FeatureDataset(Dataset):
    def __init__(self, feats, labels, global_idx):
        self.feats = feats; self.labels = labels; self.idx = global_idx
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.feats[i], int(self.labels[i]), int(self.idx[i])


def _cifar10_ds(data_root, augment, train=True):
    m, s = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize(m, s)]) if augment else \
         T.Compose([T.ToTensor(), T.Normalize(m, s)])
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
    tr_raw = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=clip_tf)
    te_raw = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=clip_tf)
    tf_f, tf_l = _extract(tr_raw); te_f, te_l = _extract(te_raw)
    torch.save(tf_f, paths["train"]); torch.save(tf_l, paths["train_labels"])
    torch.save(te_f, paths["test"]);  torch.save(te_l, paths["test_labels"])
    del cm; torch.cuda.empty_cache()
    return tf_f, tf_l, te_f, te_l


def build_dataset_clip(dataset_name, data_root, cache_dir, device, seed=42):
    is_lt = "lt" in dataset_name
    lt_ir = 50 if "lt50" in dataset_name else (100 if "lt100" in dataset_name else 1)
    tf_f, tf_l, te_f, te_l = _load_clip_features(data_root, cache_dir, device)
    full_targets = tf_l.numpy(); all_idx = np.arange(len(full_targets))
    lt_idx = make_cifar10_lt_indices(full_targets, lt_ir, seed=42) if is_lt else all_idx
    lt_targets = full_targets[lt_idx]
    pub_idx, priv_idx = make_public_private_split(lt_idx, lt_targets, public_frac=0.1, seed=42)
    rng = np.random.default_rng(seed)
    pub_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]
    priv_feats  = tf_f[priv_idx]
    priv_labels = torch.tensor(full_targets[priv_idx], dtype=torch.long)
    pub_feats   = tf_f[pub_use]
    pub_labels  = torch.tensor(full_targets[pub_use], dtype=torch.long)
    priv_ds = _FeatureDataset(priv_feats, priv_labels, torch.tensor(priv_idx))
    tier_labels = (np.array([class_to_tier(c) for c in full_targets[priv_idx]], dtype=np.int32)
                   if is_lt else None)
    test_ds = _FeatureDataset(te_f, te_l, torch.arange(len(te_l)))
    class_counts = np.bincount(full_targets[priv_idx], minlength=10)
    return (pub_feats, pub_labels, priv_ds, test_ds, priv_idx,
            tier_labels, 10, 1e-5, class_counts,
            priv_feats, priv_labels, te_f, te_l)


def build_dataset_wrn(dataset_name, data_root, seed=42):
    is_lt = "lt" in dataset_name
    lt_ir = 50 if "lt50" in dataset_name else (100 if "lt100" in dataset_name else 1)
    train_aug   = _cifar10_ds(data_root, augment=True,  train=True)
    train_noaug = _cifar10_ds(data_root, augment=False, train=True)
    test_ds     = _cifar10_ds(data_root, augment=False, train=False)
    full_targets = np.array(train_aug.targets)
    lt_idx = make_cifar10_lt_indices(full_targets, lt_ir, seed=42) if is_lt else np.arange(len(train_aug))
    lt_targets = full_targets[lt_idx]
    pub_idx, priv_idx = make_public_private_split(lt_idx, lt_targets, public_frac=0.1, seed=42)
    rng = np.random.default_rng(seed)
    pub_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]
    pub_ds = torch.utils.data.Subset(train_noaug, pub_use.tolist())
    priv_ds = _IndexedSubset(train_aug, priv_idx)
    tier_labels = (np.array([class_to_tier(c) for c in full_targets[priv_idx]], dtype=np.int32)
                   if is_lt else None)
    pub_x = torch.stack([pub_ds[i][0] for i in range(len(pub_ds))])
    pub_y = torch.tensor([int(full_targets[pub_use[i]]) for i in range(len(pub_use))], dtype=torch.long)
    class_counts = np.bincount(full_targets[priv_idx], minlength=10)
    priv_labels = full_targets[priv_idx].astype(np.int32)
    return (pub_x, pub_y, priv_ds, test_ds, priv_idx,
            tier_labels, 10, 1e-5, class_counts, priv_labels)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, num_classes=10, feat_dim=512):
        super().__init__(); self.fc = nn.Linear(feat_dim, num_classes)
    def forward(self, x): return self.fc(x.float())


def make_model(regime, num_classes):
    if regime == "R3":
        return LinearHead(num_classes, feat_dim=512)
    return WideResNet(depth=28, widen_factor=2, num_classes=num_classes, n_groups=N_GROUPS)


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
        ey = torch.zeros_like(p); ey.scatter_(1, y.unsqueeze(1), 1.0)
        delta = p - ey
        g_W = (delta.unsqueeze(2) * h.unsqueeze(1)).reshape(h.shape[0], -1)
    return torch.cat([g_W, delta], dim=1)


def _set_grads(model, flat_g):
    offset = 0
    for p in model.parameters():
        n = p.numel(); p.grad = flat_g[offset:offset+n].view(p.shape).clone(); offset += n


# ---------------------------------------------------------------------------
# DP-SGD training step
# ---------------------------------------------------------------------------

def dp_sgd_step(model, x, y, sigma_use, d, device, regime):
    B = x.shape[0]; sum_g = torch.zeros(d, device=device)
    chunk = CHUNK_R3 if regime == "R3" else CHUNK_WRN
    for i in range(0, B, chunk):
        xc, yc = x[i:i+chunk], y[i:i+chunk]
        if regime == "R3":
            gc = _per_sample_grads_linear(model, xc, yc, device)
        else:
            gc = _per_sample_grads_vmap(model, xc, yc, device)
        nms = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
        gc_c = gc * (CLIP_C / nms).clamp(max=1.0)
        sum_g += gc_c.sum(0)
        del gc, gc_c; torch.cuda.empty_cache()
    noise = torch.randn(d, device=device) * sigma_use
    return (sum_g + noise) / B


# ---------------------------------------------------------------------------
# Nyström sufficient statistics (R3 CLIP)
# ---------------------------------------------------------------------------

def nystrom_stats_r3(model, priv_feats, priv_labels, rho, r_max, device):
    """
    R3 (CLIP linear head): materialize G_t, compute Nyström stats.

    Returns:
      norms_t   [n] float32
      losses_t  [n] float32
      B_t       [r_max, r_max] float32
      M_t       [r_max, r_max] float32
      Y_proj_t  [n, r_max] float32
    """
    n = len(priv_feats)
    G_parts, norm_parts, loss_parts = [], [], []
    model.eval()
    for i in range(0, n, CHUNK_R3):
        h = priv_feats[i:i+CHUNK_R3].to(device).float()
        y = priv_labels[i:i+CHUNK_R3].to(device).long()
        with torch.no_grad():
            gc = _per_sample_grads_linear(model, h, y, device)
            nms = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
            gc_c = gc * (CLIP_C / nms).clamp(max=1.0)
            G_parts.append(gc_c.cpu().float())
            norm_parts.append(gc_c.norm(dim=1).cpu())
            logits = model(h)
            loss_parts.append(F.cross_entropy(logits, y, reduction='none').cpu())
        del gc, gc_c, h, y, logits; torch.cuda.empty_cache()

    G = torch.cat(G_parts, 0)          # (n, d) CPU float32
    norms_t  = torch.cat(norm_parts, 0)
    losses_t = torch.cat(loss_parts, 0)
    del G_parts, norm_parts, loss_parts

    # Covariance Σ_t = ρ G^T G, move to GPU if fits
    G_dev = G.to(device)
    cov = rho * G_dev.T @ G_dev        # (d, d)

    # Top-r_max eigenvectors (ascending → descending)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    Q_t = eigvecs[:, -r_max:].flip(1)  # (d, r_max)

    # Y_t = Σ_t Q_t = cov Q_t
    Y_t = cov @ Q_t                     # (d, r_max)
    B_t = (Q_t.T @ Y_t).cpu()          # (r_max, r_max)
    M_t = (Y_t.T @ Y_t).cpu()          # (r_max, r_max)
    Y_proj_t = (G_dev @ Y_t).cpu()     # (n, r_max): row i = Y_t^T g_i

    del cov, eigvals, eigvecs, Q_t, Y_t, G_dev; torch.cuda.empty_cache()
    return norms_t, losses_t, B_t, M_t, Y_proj_t


# ---------------------------------------------------------------------------
# Nyström sufficient statistics (R1/R2 WRN via rSVD + streaming)
# ---------------------------------------------------------------------------

def nystrom_stats_wrn(model, priv_ds, rho, r_max, device):
    """
    WRN: 4-pass approach (rSVD for Q_t, then streaming for Y_t and Y_proj).

    Pass 1: rSVD sketch (Y_sketch = G Omega), also norms and losses
    Pass 2: B = Q^T G → V_r = Q_t
    Pass 3: P = G Q_t, accumulate Y_t = ρ G^T P
    Pass 4: Y_proj = G Y_t
    """
    model.eval()
    N = len(priv_ds)
    d = sum(p.numel() for p in model.parameters())
    loader = DataLoader(priv_ds, batch_size=CHUNK_WRN, shuffle=False,
                        num_workers=2, pin_memory=True, drop_last=False)
    k = r_max + 20

    # --- Pass 1: sketch + norms + losses --------------------------------
    rng_sketch = torch.Generator(device=device); rng_sketch.manual_seed(99999)
    Omega = torch.randn(d, k, generator=rng_sketch, device=device)
    Y_sketch_parts = []; norm_parts = []; loss_parts = []
    for batch in loader:
        x, y = batch[0], batch[1]
        with torch.no_grad():
            gc = _per_sample_grads_vmap(model, x, y, device)
            nms = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
            gc_c = gc * (CLIP_C / nms).clamp(max=1.0)
            Y_sketch_parts.append((gc_c @ Omega).cpu())
            norm_parts.append(gc_c.norm(dim=1).cpu())
            logits = model(x.to(device).float())
            loss_parts.append(F.cross_entropy(logits, y.to(device).long(),
                                              reduction='none').cpu())
        del gc, gc_c; torch.cuda.empty_cache()
    Y_sketch = torch.cat(Y_sketch_parts, 0)          # (N, k)
    norms_t  = torch.cat(norm_parts, 0)
    losses_t = torch.cat(loss_parts, 0)
    Q_svd, _ = torch.linalg.qr(Y_sketch)             # (N, k)
    del Y_sketch, Y_sketch_parts, Omega; torch.cuda.empty_cache()

    # --- Pass 2: B = Q_svd^T G → rSVD eigenpairs -----------------------
    Bmat = torch.zeros(k, d, device=device, dtype=torch.float32)
    row = 0
    for batch in loader:
        x, y = batch[0], batch[1]; bs = x.shape[0]
        with torch.no_grad():
            gc = _per_sample_grads_vmap(model, x, y, device)
            nms = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
            gc_c = gc * (CLIP_C / nms).clamp(max=1.0)
            Q_row = Q_svd[row:row+bs].to(device)
            Bmat.addmm_(Q_row.T, gc_c)
        row += bs; del gc, gc_c, Q_row; torch.cuda.empty_cache()
    del Q_svd

    with torch.no_grad():
        BBT = Bmat @ Bmat.T
        eigvals_b, U_B = torch.linalg.eigh(BBT)
        top_r = min(r_max, int((eigvals_b > 0).sum().item()))
        eigvals_r = eigvals_b[-top_r:].flip(0).clamp(min=0.0)
        U_B_r = U_B[:, -top_r:].flip(1)
        V_r = (Bmat.T @ U_B_r) / eigvals_r.sqrt().clamp(min=1e-12)
    del Bmat, BBT, U_B; torch.cuda.empty_cache()
    # V_r: (d, top_r) — approximate eigenvectors of Σ_t (before ρ scaling)
    Q_t = V_r  # (d, r_max or top_r)
    actual_r = Q_t.shape[1]

    # --- Pass 3: P = G Q_t, Y_t = ρ G^T P ------------------------------
    P_buf = torch.zeros(N, actual_r, dtype=torch.float32)  # CPU buffer
    Y_t = torch.zeros(d, actual_r, device=device, dtype=torch.float32)
    row = 0
    for batch in loader:
        x, y = batch[0], batch[1]; bs = x.shape[0]
        with torch.no_grad():
            gc = _per_sample_grads_vmap(model, x, y, device)
            nms = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
            gc_c = gc * (CLIP_C / nms).clamp(max=1.0)
            P_chunk = gc_c @ Q_t                                   # (bs, r)
            P_buf[row:row+bs] = P_chunk.cpu()
            Y_t.addmm_(gc_c.T, P_chunk, alpha=rho)                # accumulate
        row += bs; del gc, gc_c, P_chunk; torch.cuda.empty_cache()
    P_buf_dev = P_buf.to(device)
    B_t = (rho * P_buf_dev.T @ P_buf_dev).cpu()                    # (r, r)
    M_t = (Y_t.T @ Y_t).cpu()                                      # (r, r)
    del P_buf_dev

    # Pad B_t and M_t to (r_max, r_max) if needed
    if actual_r < r_max:
        B_pad = torch.zeros(r_max, r_max); B_pad[:actual_r, :actual_r] = B_t; B_t = B_pad
        M_pad = torch.zeros(r_max, r_max); M_pad[:actual_r, :actual_r] = M_t; M_t = M_pad

    # --- Pass 4: Y_proj = G Y_t -----------------------------------------
    Y_proj_parts = []
    for batch in loader:
        x, y = batch[0], batch[1]
        with torch.no_grad():
            gc = _per_sample_grads_vmap(model, x, y, device)
            nms = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
            gc_c = gc * (CLIP_C / nms).clamp(max=1.0)
            yp = gc_c @ Y_t                                        # (bs, r)
        Y_proj_parts.append(yp.cpu()); del gc, gc_c, yp; torch.cuda.empty_cache()
    Y_proj_t = torch.cat(Y_proj_parts, 0)                         # (N, r)

    # Pad Y_proj to (N, r_max)
    if actual_r < r_max:
        tmp = torch.zeros(N, r_max); tmp[:, :actual_r] = Y_proj_t; Y_proj_t = tmp

    del Q_t, V_r, Y_t; torch.cuda.empty_cache()
    return norms_t, losses_t, B_t, M_t, Y_proj_t


# ---------------------------------------------------------------------------
# Public pretraining (R2)
# ---------------------------------------------------------------------------

def pretrain(model, pub_x, pub_y, device):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=PRETRAIN_LR, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EP)
    N = pub_x.shape[0]
    for ep in range(PRETRAIN_EP):
        perm = torch.randperm(N)
        for i in range(0, N, 256):
            idx = perm[i:i+256]; opt.zero_grad()
            F.cross_entropy(model(pub_x[idx].to(device)), pub_y[idx].to(device)).backward()
            opt.step()
        sch.step()
    print(f"  [pretrain] done ({PRETRAIN_EP} epochs)")


# ---------------------------------------------------------------------------
# Calibration
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, test_ds, device, is_clip):
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    model.eval(); correct = total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        correct += (model(x.float()).argmax(1) == y).sum().item(); total += y.shape[0]
    return correct / total


# ---------------------------------------------------------------------------
# LiRA target selection (deterministic)
# ---------------------------------------------------------------------------

def select_lira_targets(setting_id, priv_idx, priv_labels, tier_labels,
                        test_labels_np, n_targets, seed=1000):
    """
    Select member targets from private set and nonmember targets from test set.
    Returns (member_local_idx, nonmember_test_idx) as arrays of local/test positions.
    """
    rng = np.random.default_rng(seed)
    priv_labels_np = np.asarray(priv_labels) if not isinstance(priv_labels, np.ndarray) else priv_labels

    if tier_labels is not None:
        # Stratified by tier: head=0, mid=1, tail=2
        n_per_tier = n_targets // 3
        member_idx = []
        for tier in [0, 1, 2]:
            mask = (tier_labels == tier)
            avail = np.where(mask)[0]
            n_sel = min(n_per_tier, len(avail))
            chosen = rng.choice(avail, size=n_sel, replace=False)
            member_idx.append(chosen)
        # Fill remaining slots from largest available tier
        total = sum(len(m) for m in member_idx)
        if total < n_targets:
            remaining = n_targets - total
            all_used = np.concatenate(member_idx)
            all_avail = np.setdiff1d(np.arange(len(priv_idx)), all_used)
            extra = rng.choice(all_avail, size=min(remaining, len(all_avail)), replace=False)
            member_idx.append(extra)
        member_local = np.concatenate(member_idx)
    else:
        # Stratified by class: 100 per class
        n_per_class = n_targets // 10
        member_idx = []
        for c in range(10):
            mask = (priv_labels_np == c)
            avail = np.where(mask)[0]
            n_sel = min(n_per_class, len(avail))
            chosen = rng.choice(avail, size=n_sel, replace=False)
            member_idx.append(chosen)
        member_local = np.concatenate(member_idx)

    # Nonmembers from test set, stratified by class
    n_nonmembers = len(member_local)
    n_per_class = n_nonmembers // 10
    test_labels_np = np.asarray(test_labels_np)
    nonmember_idx = []
    for c in range(10):
        mask = (test_labels_np == c)
        avail = np.where(mask)[0]
        n_sel = min(n_per_class, len(avail))
        chosen = rng.choice(avail, size=n_sel, replace=False)
        nonmember_idx.append(chosen)
    nonmember_test = np.concatenate(nonmember_idx)

    return member_local.astype(np.int32), nonmember_test.astype(np.int32)


@torch.no_grad()
def compute_logits(model, feats_or_ds, indices, device, is_clip, chunk=256):
    """Compute logits for specific examples (by local index into feats/ds)."""
    model.eval()
    logits_list = []
    for i in range(0, len(indices), chunk):
        idx = indices[i:i+chunk]
        if is_clip:
            x = feats_or_ds[idx].to(device).float()
        else:
            x = torch.stack([feats_or_ds[int(j)][0] for j in idx]).to(device)
        logits_list.append(model(x).cpu())
    return torch.cat(logits_list, 0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_run(setting_id, cfg, seed, device, data_root, cache_dir, runs_dir):
    regime  = cfg["regime"]
    dataset = cfg["dataset"]
    eps     = cfg["eps"]
    batch   = cfg["batch"]
    epochs  = cfg["epochs"]
    K       = cfg["K"]
    is_clip = (regime == "R3")

    run_dir = os.path.join(runs_dir, setting_id, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    ckpt_path  = os.path.join(run_dir, "checkpoint.pt")
    final_path = os.path.join(run_dir, "model_final.pt")
    meta_path  = os.path.join(run_dir, "metadata.json")

    if os.path.exists(final_path) and os.path.exists(meta_path):
        print(f"[P18] {setting_id}/seed_{seed}: already done, skipping.")
        return

    print(f"\n[P18] === {setting_id} seed={seed} ===")
    print(f"  dataset={dataset} regime={regime} eps={eps} batch={batch} K={K}")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Build datasets
    tier_labels = None; test_labels_np = None
    if is_clip:
        (pub_feats, pub_labels, priv_ds, test_ds, priv_idx,
         tier_labels, num_classes, delta, class_counts,
         priv_feats_all, priv_labels_all, te_feats, te_labels) = \
            build_dataset_clip(dataset, data_root, cache_dir, device, seed=42)
        test_labels_np = te_labels.numpy()
    else:
        (pub_x, pub_y, priv_ds, test_ds, priv_idx,
         tier_labels, num_classes, delta, class_counts, priv_labels_wrn) = \
            build_dataset_wrn(dataset, data_root, seed=42)
        test_labels_np = np.array(test_ds.targets if hasattr(test_ds, 'targets') else
                                  [test_ds[i][1] for i in range(len(test_ds))])

    n_priv = len(priv_ds)
    steps_per_ep = n_priv // batch  # drop_last
    T_train = epochs * steps_per_ep
    T_acc   = math.ceil(T_train / K)
    q       = batch / n_priv
    rho     = q * (1.0 - q)
    sigma   = calibrate_sigma(eps, delta, q, T_train)
    sigma_use = sigma * CLIP_C
    a         = sigma ** 2 * CLIP_C ** 2

    print(f"  n={n_priv}  T_train={T_train}  T_acc={T_acc}  q={q:.5f}  σ={sigma:.4f}  a={a:.6f}")

    # Pre-allocate output arrays
    # clipped_norms and losses: (n, T_acc)
    clipped_norms_arr = np.zeros((n_priv, T_acc), dtype=np.float32)
    losses_arr        = np.zeros((n_priv, T_acc), dtype=np.float32)
    B_arr             = np.zeros((T_acc, R_MAX, R_MAX), dtype=np.float32)
    YTY_arr           = np.zeros((T_acc, R_MAX, R_MAX), dtype=np.float32)

    # Y_projections: (n, T_acc, r_max) — use memmap for large cases
    Yproj_path = os.path.join(run_dir, "Y_projections.npy")
    Yproj_size = n_priv * T_acc * R_MAX * 4  # bytes float32
    if Yproj_size > 2e9:  # > 2GB: use .npy-backed memmap
        _mm_mode = 'r+' if (os.path.exists(ckpt_path) and os.path.exists(Yproj_path)) else 'w+'
        if _mm_mode == 'r+':
            Y_proj_mm = np.lib.format.open_memmap(Yproj_path, mode='r+')
        else:
            Y_proj_mm = np.lib.format.open_memmap(
                Yproj_path, mode='w+', dtype=np.float32,
                shape=(n_priv, T_acc, R_MAX))
        use_memmap = True
    else:
        Y_proj_arr = np.zeros((n_priv, T_acc, R_MAX), dtype=np.float32)
        use_memmap = False

    priv_labels_np = (priv_labels_all.numpy().astype(np.int32) if is_clip
                      else priv_labels_wrn)
    priv_idx_np    = np.asarray(priv_idx, dtype=np.int32)

    # Model
    model = make_model(regime, num_classes).to(device)
    d_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {d_params:,}")

    # Pretrain (R2)
    if regime == "R2" and not os.path.exists(ckpt_path):
        if is_clip:
            pretrain(model, pub_feats.to(device), pub_labels.to(device), device)
        else:
            pretrain(model, pub_x.to(device), pub_y.to(device), device)

    # Optimizer: SGD, no momentum (per spec Section 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.0, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 1; step_global = 0; t_idx = 0; best_acc = 0.0
    acct_step_indices = []

    # Resume from checkpoint
    if os.path.exists(ckpt_path):
        print(f"  [resume] Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch    = ckpt["epoch"] + 1
        step_global    = ckpt["step_global"]
        t_idx          = ckpt["t_idx"]
        best_acc       = ckpt["best_acc"]
        acct_step_indices = ckpt["acct_step_indices"]
        clipped_norms_arr[:] = ckpt["clipped_norms_arr"]
        losses_arr[:]         = ckpt["losses_arr"]
        B_arr[:]              = ckpt["B_arr"]
        YTY_arr[:]            = ckpt["YTY_arr"]
        if use_memmap:
            pass  # Y_proj_mm already populated from disk via 'r+' mode
        else:
            Y_proj_arr[:, :t_idx, :] = ckpt["Y_proj_partial"]
        torch.set_rng_state(ckpt["rng_state"])
        np.random.set_state(ckpt["np_rng_state"])
        random.setstate(ckpt["py_rng_state"])
        print(f"  [resume] epoch={ckpt['epoch']}  t_idx={t_idx}")

    priv_loader = DataLoader(priv_ds, batch_size=batch, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        model.train(); total_loss = 0.0; n_batches = 0

        for batch_data in priv_loader:
            x, y = batch_data[0], batch_data[1]
            optimizer.zero_grad(set_to_none=True)

            flat_g = dp_sgd_step(model, x, y, sigma_use, d_params, device, regime)
            _set_grads(model, flat_g)
            optimizer.step()

            # Accounting step
            if step_global % K == 0:
                t_start = time.time()
                model.eval()
                if is_clip:
                    norms_t, loss_t, B_t, M_t, Yp_t = nystrom_stats_r3(
                        model, priv_feats_all, priv_labels_all, rho, R_MAX, device)
                else:
                    norms_t, loss_t, B_t, M_t, Yp_t = nystrom_stats_wrn(
                        model, priv_ds, rho, R_MAX, device)

                clipped_norms_arr[:, t_idx] = norms_t.numpy()
                losses_arr[:, t_idx]         = loss_t.numpy()
                B_arr[t_idx]                 = B_t.numpy()
                YTY_arr[t_idx]               = M_t.numpy()
                if use_memmap:
                    Y_proj_mm[:, t_idx, :] = Yp_t.numpy()
                else:
                    Y_proj_arr[:, t_idx, :] = Yp_t.numpy()

                acct_step_indices.append(step_global)
                t_elapsed = time.time() - t_start
                print(f"    [acct t={t_idx:4d}] step={step_global}  "
                      f"norm_med={norms_t.median():.4f}  "
                      f"B_diag_max={B_t.diagonal().max():.4g}  "
                      f"time={t_elapsed:.1f}s")
                t_idx += 1
                del norms_t, loss_t, B_t, M_t, Yp_t
                torch.cuda.empty_cache()
                model.train()

            with torch.no_grad():
                b_sub = min(x.shape[0], 64)
                out = model(x[:b_sub].to(device).float() if is_clip else x[:b_sub].to(device))
                total_loss += F.cross_entropy(out, y[:b_sub].to(device)).item()
            n_batches += 1; step_global += 1

        scheduler.step()
        test_acc = evaluate(model, test_ds, device, is_clip)
        if test_acc > best_acc:
            best_acc = test_acc
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  ep {epoch:3d}/{epochs}  acc={test_acc:.4f}  best={best_acc:.4f}"
              f"  lr={cur_lr:.5f}  t_idx={t_idx}")

        # Checkpoint — for memmap, flush to disk instead of embedding in checkpoint
        ckpt_dict = {
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "rng_state": torch.get_rng_state(),
            "np_rng_state": np.random.get_state(),
            "py_rng_state": random.getstate(),
            "step_global": step_global, "t_idx": t_idx, "best_acc": best_acc,
            "acct_step_indices": acct_step_indices,
            "clipped_norms_arr": clipped_norms_arr,
            "losses_arr": losses_arr,
            "B_arr": B_arr,
            "YTY_arr": YTY_arr,
        }
        if use_memmap:
            Y_proj_mm.flush()  # persist to Yproj_path; reopened as 'r+' on resume
        else:
            ckpt_dict["Y_proj_partial"] = Y_proj_arr[:, :t_idx, :].copy()
        torch.save(ckpt_dict, ckpt_path)

    # Save final model
    torch.save(model.state_dict(), final_path)

    # Save all numpy arrays
    np.save(os.path.join(run_dir, "example_indices.npy"), priv_idx_np)
    np.save(os.path.join(run_dir, "labels.npy"), priv_labels_np)
    np.save(os.path.join(run_dir, "class_counts.npy"), class_counts)
    np.save(os.path.join(run_dir, "clipped_norms.npy"), clipped_norms_arr[:, :t_idx])
    np.save(os.path.join(run_dir, "losses.npy"), losses_arr[:, :t_idx])
    np.save(os.path.join(run_dir, "B_matrices.npy"), B_arr[:t_idx])
    np.save(os.path.join(run_dir, "YTY_matrices.npy"), YTY_arr[:t_idx])
    if not use_memmap:
        np.save(Yproj_path, Y_proj_arr[:, :t_idx, :])
    else:
        # Trim memmap to actual t_idx if needed
        if t_idx < T_acc:
            Y_proj_mm.flush()
            final_yp = np.lib.format.open_memmap(Yproj_path, mode='r')
            np.save(Yproj_path + "_trim.npy", np.asarray(final_yp[:, :t_idx, :]))
            os.rename(Yproj_path + "_trim.npy", Yproj_path)
        else:
            Y_proj_mm.flush()

    if tier_labels is not None:
        np.save(os.path.join(run_dir, "tier_labels.npy"), tier_labels)

    # Save metadata
    n_lira = LIRA_N_TARGETS.get(setting_id, 1000)
    member_local, nonmember_test = select_lira_targets(
        setting_id, priv_idx_np, priv_labels_np, tier_labels,
        test_labels_np, n_lira, seed=1000)
    np.save(os.path.join(run_dir, "lira_member_local_idx.npy"), member_local)
    np.save(os.path.join(run_dir, "lira_nonmember_test_idx.npy"), nonmember_test)

    # Compute target logits on final model
    if is_clip:
        member_logits = compute_logits(model, priv_feats_all, member_local, device,
                                       is_clip=True)
        nonmember_logits = compute_logits(model, te_feats, nonmember_test, device,
                                          is_clip=True)
    else:
        member_logits = compute_logits(model, priv_ds, member_local, device,
                                       is_clip=False)
        nonmember_logits = compute_logits(model, test_ds, nonmember_test, device,
                                          is_clip=False)
    np.save(os.path.join(run_dir, "target_logits_dp_members.npy"), member_logits)
    np.save(os.path.join(run_dir, "target_logits_dp_nonmembers.npy"), nonmember_logits)
    # Also save combined target_logits_dp.npy (members only, per spec 4.6)
    np.save(os.path.join(run_dir, "target_logits_dp.npy"), member_logits)

    # Sanity check: fallback_mask
    fallback_count = 0
    U_full_max = clipped_norms_arr[:, :t_idx] ** 2 / a
    # (we don't compute U here — that's certify's job, but we log norms)

    metadata = {
        "setting_id": setting_id,
        "seed": seed,
        "dataset": dataset,
        "regime": regime,
        "architecture": "CLIP_ViT_B_32_linear" if is_clip else "WRN-28-2_GroupNorm16",
        "n_train": int(n_priv),
        "batch_size": int(batch),
        "q": float(q),
        "rho": float(rho),
        "C": float(CLIP_C),
        "sigma": float(sigma),
        "a": float(a),
        "delta": float(delta),
        "epsilon_target": float(eps),
        "epochs": int(epochs),
        "T_train": int(step_global),
        "T_acc": int(t_idx),
        "accounting_step_indices": acct_step_indices,
        "r_max": R_MAX,
        "Q_method": "eigenvectors_full_covariance" if is_clip else "rsvd_lanczos",
        "gradient_convention": "summed",
        "accountant": "prv",
        "best_test_acc": float(best_acc),
        "n_lira_members": int(len(member_local)),
        "n_lira_nonmembers": int(len(nonmember_test)),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"\n[P18] {setting_id}/seed_{seed} DONE"
          f"  best_acc={best_acc:.4f}  T_acc={t_idx}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 18 training")
    parser.add_argument("--setting",      type=str, default=None)
    parser.add_argument("--seed",         type=int, default=None)
    parser.add_argument("--all_minimal",  action="store_true")
    parser.add_argument("--all",          action="store_true")
    parser.add_argument("--gpu",          type=int, default=0)
    parser.add_argument("--data_root",    type=str, default=DATA_ROOT)
    parser.add_argument("--cache_dir",    type=str, default=CACHE_DIR)
    parser.add_argument("--runs_dir",     type=str, default=RUNS_DIR)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[P18] Device: {device}")

    if args.setting:
        settings_to_run = {args.setting: ALL_SETTINGS[args.setting]}
    elif args.all_minimal:
        settings_to_run = SETTINGS_MINIMAL
    elif args.all:
        settings_to_run = ALL_SETTINGS
    else:
        print("[P18] No setting specified. Running S2 (headline). Use --all_minimal for S1+S2+S3.")
        settings_to_run = {"S2": SETTINGS_MINIMAL["S2"]}

    for sid, cfg in settings_to_run.items():
        seeds = [args.seed] if args.seed is not None else list(range(cfg["n_seeds"]))
        for seed in seeds:
            train_run(sid, cfg, seed, device, args.data_root, args.cache_dir, args.runs_dir)

    print("[P18] All done.")


if __name__ == "__main__":
    main()
