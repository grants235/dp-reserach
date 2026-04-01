"""
Phase 10: EMNIST (Experiment D) and Pretrained Feature Extractor (Experiment E)
=================================================================================

Experiment D — EMNIST-balanced (from-scratch ResNet-20, 47 classes)
  vanilla          DP-SGD cold start
  vanilla_warm     DP-SGD warm start (pretrain on 2000 public images)
  gep              GEP (variance-PCA subspace, split noise)
  pda_dpmd         PDA-DPMD with linear alpha ramp (alpha_0=0.9, ramp_frac=0.5)
  pda_cw           CW-PDA-DPMD (per-step coherence-weighted public gradient)

EMNIST-balanced: 112,800 train / 18,800 test, 47 classes (digits + upper/lower letters),
~2,400 per class train.  Grayscale 28×28 upsampled to 32×32 and replicated to 3 channels
so the same ResNet-20 architecture works without modification.

Why EMNIST instead of CIFAR-100:
  - 47 classes → more fragmented gradient structure than CIFAR-10's 10
  - Balanced dataset: no long-tail confound (different from the IR experiments)
  - Simpler image content means the challenge is purely from the number of classes
    and privacy noise, isolating the effect of our methods from feature difficulty

Experiment E — Linear probe on pretrained backbone (CIFAR-10, feature-space DP)
  vanilla_ft       DP-SGD on linear head (cold-start random head)
  gep_ft           GEP on linear-head gradients
  pda_ft           PDA-DPMD on linear head
  pda_cw_ft        CW-PDA-DPMD on linear head

Usage
-----
  # Experiment D: EMNIST (quick sweep, 1 seed)
  python experiments/exp_p10_cifar100_ft.py --exp D --gpu 0

  # Experiment E: linear probe (extract features first, then very fast)
  python experiments/exp_p10_cifar100_ft.py --exp E --backbone clip_vitb32 --gpu 0

  # Single arm:
  python experiments/exp_p10_cifar100_ft.py --exp D --arm pda_cw --eps 2 --seed 0 --gpu 0
  python experiments/exp_p10_cifar100_ft.py --exp E --arm pda_cw_ft --eps 1 --seed 0 --gpu 0

  # Extract and cache features (Exp E pre-step):
  python experiments/exp_p10_cifar100_ft.py --extract_features --backbone clip_vitb32 --gpu 0

  # Analysis only:
  python experiments/exp_p10_cifar100_ft.py --analysis_only

Notes
-----
- Exp D arms share code with P9; alpha ramp and norm-matching are default.
- Exp E extracts features once (cached), then trains an nn.Linear under DP.
  Linear probe d ≈ 5130 (CLIP) or 20490 (ResNet-50), vs ~278K for ResNet-20.
  Per-run time: ~2 min (E) vs ~20 min (D).
- Both experiments log cos_pub_priv for PDA arms.
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
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import ResNet20
from src.datasets import make_public_private_split

# ---------------------------------------------------------------------------
# Shared hyperparameters
# ---------------------------------------------------------------------------

DELTA        = 1e-5
CLIP_VAN     = 1.0       # clipping norm (vanilla / PDA); same for D and E
CLIP0        = 5.0       # GEP embedding clip
CLIP1        = 2.0       # GEP residual clip
PRETRAIN_LR  = 0.01
PRETRAIN_MOM = 0.9
PRETRAIN_WD  = 5e-4
GRAD_CHUNK   = 64
DATA_ROOT    = "./data"
RESULTS_DIR  = "./results/exp_p10"
FEAT_CACHE   = "./data/features"

# Alpha schedule (best settings from Exp A / P9):
ALPHA_START  = 0.9
RAMP_FRAC    = 0.5       # ramp alpha 0.9→1.0 over first 50% of steps

# ---------------------------------------------------------------------------
# Experiment D — EMNIST-balanced constants
# ---------------------------------------------------------------------------

D_EPOCHS       = 60
D_BATCH_SIZE   = 1000
D_LR           = 0.1
D_MOMENTUM     = 0.9
D_WEIGHT_DECAY = 5e-4
D_N_CLASSES    = 47        # EMNIST balanced: digits 0-9 + upper/lower letters
D_N_PUB        = 2000      # ~43 per class × 47 classes (cap applied after stratified sample)
D_PUB_FRAC     = 0.02      # 2% per class → ~48 per class × 47 = 2256, capped to 2000
D_R_DIM        = 1000      # GEP subspace rank (< D_N_PUB, valid for svd_lowrank)
D_PUB_BATCH    = 256
D_EPS_LIST     = [1.0, 2.0, 4.0, 8.0]
D_N_SEEDS_Q    = 1         # quick sweep
D_N_SEEDS_F    = 3         # full multi-seed
D_PRETRAIN_EP  = 50
D_ARMS         = ["vanilla", "vanilla_warm", "gep", "pda_dpmd", "pda_cw"]

# EMNIST image preprocessing: grayscale 28×28 → 3-channel 32×32 for ResNet-20
_EMNIST_MEAN = [0.1307, 0.1307, 0.1307]
_EMNIST_STD  = [0.3081, 0.3081, 0.3081]

# ---------------------------------------------------------------------------
# Experiment E — linear probe constants
# ---------------------------------------------------------------------------

E_EPOCHS       = 100
E_BATCH_SIZE   = 256
E_LR           = 0.01
E_MOMENTUM     = 0.9
E_WEIGHT_DECAY = 0.0       # no weight decay for linear probe
E_N_PUB        = 2000      # same split as P9/Exp A
E_R_DIM        = 50        # small rank for low-dim linear probe
E_PUB_BATCH    = 256
E_EPS_LIST     = [0.5, 1.0, 2.0, 4.0]
E_N_SEEDS      = 1         # 3 for winners
E_PRETRAIN_EP  = 20        # warm the linear head on public features (fast)
E_ARMS         = ["vanilla_ft", "gep_ft", "pda_ft", "pda_cw_ft"]

# Arm → experiment mapping
ARM_TO_EXP = {arm: "D" for arm in D_ARMS}
ARM_TO_EXP.update({arm: "E" for arm in E_ARMS})

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """Single linear layer for Exp E. Parameters: [feature_dim × num_classes + num_classes]."""

    def __init__(self, feature_dim: int, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def _make_model_d():
    return ResNet20(num_classes=D_N_CLASSES, n_groups=16)


def _make_model_e(feature_dim: int):
    return LinearProbe(feature_dim, num_classes=10)


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


def _set_grads(model, flat_grad):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat_grad[offset: offset + n].view(p.shape).clone()
        offset += n


# ---------------------------------------------------------------------------
# Pretrained backbone + feature extraction (Exp E)
# ---------------------------------------------------------------------------

def _get_backbone(backbone_name: str, device):
    """
    Returns (feature_extractor, feature_dim, transform, actual_name).
    actual_name is the backbone that was actually loaded (may differ from backbone_name
    if a fallback occurs, e.g. clip_vitb32 → resnet50 when CLIP is not installed).
    """
    if backbone_name == "clip_vitb32":
        try:
            import clip
            model, preprocess = clip.load("ViT-B/32", device=device)
            model.eval()
            return model, 512, preprocess, "clip_vitb32"
        except ImportError:
            print("[P10] CLIP not installed. Falling back to resnet50.")
            backbone_name = "resnet50"

    if backbone_name in ("resnet50", "resnet50_torch"):
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights)
        model.fc = nn.Identity()
        model = model.to(device).eval()
        preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return model, 2048, preprocess, "resnet50"

    raise ValueError(f"Unknown backbone: {backbone_name!r}. Choose clip_vitb32 or resnet50.")


def _extract_or_load(backbone_name: str, data_root: str, cache_dir: str, device):
    """
    Extract CIFAR-10 features from pretrained backbone (L2-normalized) and cache.
    Returns dict with keys: train_x, train_y, test_x, test_y, feature_dim, backbone.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Get the actual backbone (may fall back to resnet50 if CLIP not installed)
    feat_model, feature_dim, preprocess, actual_name = _get_backbone(backbone_name, device)

    cache_path = os.path.join(cache_dir, f"cifar10_feats_{actual_name}.pt")
    if os.path.exists(cache_path):
        print(f"[P10] Loading cached features: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f"[P10] Extracting CIFAR-10 features with {actual_name}...")
    is_clip = actual_name.startswith("clip")

    def _do_extract(train: bool):
        ds = torchvision.datasets.CIFAR10(
            root=data_root, train=train, download=True, transform=preprocess
        )
        loader = DataLoader(ds, batch_size=256, shuffle=False,
                            num_workers=4, pin_memory=True)
        feats, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                if is_clip:
                    f = feat_model.encode_image(x).float()
                else:
                    f = feat_model(x).float()
                f = F.normalize(f, dim=1)   # L2-normalize for stable per-sample clipping
                feats.append(f.cpu())
                labels.append(y)
        return torch.cat(feats, 0), torch.cat(labels, 0)

    train_x, train_y = _do_extract(train=True)
    test_x,  test_y  = _do_extract(train=False)

    result = dict(train_x=train_x, train_y=train_y,
                  test_x=test_x,   test_y=test_y,
                  feature_dim=feature_dim, backbone=actual_name)
    torch.save(result, cache_path)
    print(f"[P10] Saved: {cache_path}  (dim={feature_dim})")
    return result


def _build_e_datasets(feats: dict, seed: int = 42):
    """
    Create public/private feature-space splits matching P9 (public_frac=0.1, seed=42).
    Returns: pub_x, pub_y, priv_dataset (TensorDataset), test_dataset, pub_x, pub_y tensors.
    """
    train_x = feats["train_x"]  # [50000, d]
    train_y = feats["train_y"]  # [50000]
    test_x  = feats["test_x"]   # [10000, d]
    test_y  = feats["test_y"]   # [10000]

    all_idx = np.arange(len(train_x))
    targets = train_y.numpy()

    pub_idx, priv_idx = make_public_private_split(
        all_idx, targets, public_frac=0.1, seed=seed
    )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pub_idx))
    pub_idx_use = pub_idx[perm[:E_N_PUB]]

    pub_x  = train_x[pub_idx_use]
    pub_y  = train_y[pub_idx_use]
    priv_x = train_x[priv_idx]
    priv_y = train_y[priv_idx]

    priv_dataset = TensorDataset(priv_x, priv_y)
    test_dataset  = TensorDataset(test_x, test_y)

    return pub_x, pub_y, priv_dataset, test_dataset


# ---------------------------------------------------------------------------
# EMNIST-balanced datasets (Exp D)
# ---------------------------------------------------------------------------

def _emnist_transforms(augment: bool):
    """
    EMNIST-balanced → ResNet-20 compatible transform.
    Converts 1-channel 28×28 grayscale to 3-channel 32×32.
    No horizontal flip (would flip letters/digits meaninglessly).
    """
    base = [
        T.Grayscale(num_output_channels=3),   # 1-ch → 3-ch (replicate)
        T.Resize(32),                           # 28×28 → 32×32
    ]
    if augment:
        base.append(T.RandomCrop(32, padding=4))
    base += [
        T.ToTensor(),
        T.Normalize(_EMNIST_MEAN, _EMNIST_STD),
    ]
    return T.Compose(base)


def _build_d_datasets(data_root: str, seed: int = 42):
    """
    Public (~43 per class, capped at 2000 total) + private split for EMNIST-balanced.
    Returns: pub_dataset, priv_dataset, test_dataset, pub_x, pub_y, class_sizes.
    """
    tf_noaug = _emnist_transforms(augment=False)
    tf_aug   = _emnist_transforms(augment=True)

    full_train = torchvision.datasets.EMNIST(
        root=data_root, split="balanced", train=True,
        download=True, transform=tf_noaug
    )
    full_targets = np.array(full_train.targets)
    all_idx = np.arange(len(full_train))

    pub_idx, priv_idx = make_public_private_split(
        all_idx, full_targets, public_frac=D_PUB_FRAC, seed=seed
    )

    # Cap public set at D_N_PUB while keeping stratification
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pub_idx))
    pub_idx_use = pub_idx[perm[:D_N_PUB]]

    pub_dataset  = Subset(full_train, pub_idx_use.tolist())
    priv_dataset = Subset(
        torchvision.datasets.EMNIST(
            root=data_root, split="balanced", train=True,
            download=False, transform=tf_aug
        ),
        priv_idx.tolist(),
    )
    test_dataset = torchvision.datasets.EMNIST(
        root=data_root, split="balanced", train=False,
        download=True, transform=tf_noaug
    )

    pub_x = torch.stack([pub_dataset[i][0] for i in range(len(pub_dataset))])
    pub_y = torch.tensor([pub_dataset[i][1] for i in range(len(pub_dataset))],
                         dtype=torch.long)

    class_sizes = np.bincount(full_targets[pub_idx_use], minlength=D_N_CLASSES)
    return pub_dataset, priv_dataset, test_dataset, pub_x, pub_y, class_sizes


# ---------------------------------------------------------------------------
# Per-sample gradients (identical to P9, works for any model)
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


def _per_sample_grads_all(model, x, y, device, chunk=GRAD_CHUNK):
    model.eval()
    parts = []
    for i in range(0, x.shape[0], chunk):
        g = _per_sample_grads_chunk(model, x[i:i+chunk], y[i:i+chunk], device)
        parts.append(g.cpu())
        del g
        torch.cuda.empty_cache()
    return torch.cat(parts, dim=0)  # [B, d] CPU


# ---------------------------------------------------------------------------
# Training step helpers
# ---------------------------------------------------------------------------

def _vanilla_priv_step(model, x, y, sigma, clip_c, device):
    """
    Clip-sum-noise DP-SGD.
    Returns (noised/B, signal/B): both flat [d] CPU.
    signal is the clipped sum without noise (for cosine-similarity diagnostics).
    """
    B     = x.shape[0]
    grads = _per_sample_grads_all(model, x, y, device)
    norms = grads.norm(dim=1, keepdim=True).clamp(min=1e-8)
    clipped = grads * (clip_c / norms).clamp(max=1.0)
    sum_g   = clipped.sum(dim=0)
    noise   = torch.randn_like(sum_g) * (sigma * clip_c)
    signal  = sum_g / B
    noised  = (sum_g + noise) / B
    return noised, signal


def _gep_priv_step(model, x, y, V, sigma_par, sigma_perp, clip0, clip1, device):
    """GEP step: embed in subspace V [d, r], clip par + perp separately, add noise."""
    B     = x.shape[0]
    V_dev = V.to(device)
    r, d  = V.shape[1], V.shape[0]

    grads    = _per_sample_grads_all(model, x, y, device)
    sum_c    = torch.zeros(r, device=device)
    sum_perp = torch.zeros(d, device=device)

    chunk = min(GRAD_CHUNK * 4, B)
    for i in range(0, B, chunk):
        g      = grads[i:i+chunk].to(device)
        c      = g @ V_dev
        g_par  = c @ V_dev.T
        g_perp = g - g_par
        c      = c * (clip0 / c.norm(dim=1, keepdim=True).clamp(min=1e-8)).clamp(max=1.0)
        g_perp = g_perp * (clip1 / g_perp.norm(dim=1, keepdim=True).clamp(min=1e-8)).clamp(max=1.0)
        sum_c    += c.sum(0)
        sum_perp += g_perp.sum(0)
        del g, c, g_par, g_perp
        torch.cuda.empty_cache()

    sum_c    += torch.randn(r, device=device) * (sigma_par  * clip0)
    noise_p   = torch.randn(d, device=device) * (sigma_perp * clip1)
    noise_p  -= V_dev @ (V_dev.T @ noise_p)   # project out subspace component
    sum_perp += noise_p

    return ((V_dev @ sum_c + sum_perp) / B).cpu()


def _build_subspace(model, pub_dataset_or_tensors, r, device, seed=0):
    """Variance-PCA subspace from public gradients. Returns [d, r] CPU."""
    if isinstance(pub_dataset_or_tensors, tuple):
        pub_x, pub_y = pub_dataset_or_tensors
        loader = [(pub_x[i:i+256], pub_y[i:i+256])
                  for i in range(0, pub_x.shape[0], 256)]
    else:
        loader = DataLoader(pub_dataset_or_tensors,
                            batch_size=min(256, len(pub_dataset_or_tensors)),
                            shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    gs = []
    for x, y in loader:
        gs.append(_per_sample_grads_all(model, x, y, device))
    G = torch.cat(gs, 0).float()
    k = min(r, G.shape[0] - 1, G.shape[1] - 1)
    _, _, V = torch.svd_lowrank(G, q=k, niter=4)
    return V[:, :r].cpu()


def _pub_grad_flat(model, pub_x, pub_y, device, weights=None, batch_size=256):
    """Standard (optionally weighted) public gradient via autograd. Returns [d] CPU."""
    model.train()
    model.zero_grad()
    N = pub_x.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    total_w    = 0.0

    for i in range(0, N, batch_size):
        xb = pub_x[i:i+batch_size].to(device)
        yb = pub_y[i:i+batch_size].to(device)
        loss_per = F.cross_entropy(model(xb), yb, reduction="none")
        if weights is not None:
            wb = weights[i:i+batch_size].to(device)
            total_loss = total_loss + (wb * loss_per).sum()
            total_w   += wb.sum().item()
        else:
            total_loss = total_loss + loss_per.sum()
            total_w   += len(yb)

    (total_loss / max(total_w, 1e-12)).backward()
    flat = torch.cat([p.grad.view(-1).detach().cpu() for p in model.parameters()])
    model.zero_grad()
    return flat


def _pub_grad_cw_flat(model, pub_x, pub_y, device, batch_size=256):
    """
    Per-step coherence-weighted public gradient from current model.
    Weights each example by 1/(||g_i|| + 1e-6) — low-norm (coherent) gets
    high weight. Computed on a random mini-batch from public data.
    Returns [d] CPU.
    """
    model.eval()
    N   = pub_x.shape[0]
    idx = torch.randperm(N)[:batch_size]
    xb  = pub_x[idx]
    yb  = pub_y[idx]

    grads = _per_sample_grads_chunk(model, xb, yb, device)  # [B, d] on device
    grads_cpu = grads.cpu()
    del grads
    torch.cuda.empty_cache()

    norms = grads_cpu.norm(dim=1)        # [B]
    w     = 1.0 / (norms + 1e-6)        # high weight for coherent (low-norm)
    w     = w / w.sum()                  # normalize

    return (w.unsqueeze(1) * grads_cpu).sum(0)  # [d]


def _class_grad_flat(model, pub_x, pub_y, device, n_classes=10):
    """Equal-weight average of per-class public gradients. Returns [d] CPU."""
    model.train()
    total_grad   = None
    n_present    = 0
    for k in range(n_classes):
        mask = (pub_y == k)
        if mask.sum() == 0:
            continue
        g_k = _pub_grad_flat(model, pub_x[mask], pub_y[mask], device)
        total_grad = g_k if total_grad is None else total_grad + g_k
        n_present += 1
    if total_grad is None:
        return torch.zeros(_num_params(model))
    return total_grad / n_present


# ---------------------------------------------------------------------------
# Privacy calibration
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


def _gep_sigmas(sigma_van):
    return sigma_van * math.sqrt(2), sigma_van * math.sqrt(2)


# ---------------------------------------------------------------------------
# Pretraining on public data
# ---------------------------------------------------------------------------

def _pretrain_on_public(model, pub_x, pub_y, device, epochs, lr=PRETRAIN_LR,
                        momentum=PRETRAIN_MOM, wd=PRETRAIN_WD):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=momentum, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    N = pub_x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, 256):
            idx = perm[i:i+256]
            optimizer.zero_grad()
            F.cross_entropy(model(pub_x[idx].to(device)),
                            pub_y[idx].to(device)).backward()
            optimizer.step()
        scheduler.step()
    print(f"[P10]   Pretraining done ({epochs} epochs)")
    return model


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
def _probe_loss(model, x, y, device):
    model.eval()
    return F.cross_entropy(model(x.to(device)), y.to(device)).item()


# ---------------------------------------------------------------------------
# Main training run (shared by D and E)
# ---------------------------------------------------------------------------

def _train_run(arm_name, exp, eps, seed,
               pub_x, pub_y,
               priv_dataset, test_dataset,
               device, out_dir,
               clip_c=CLIP_VAN,
               clip0=CLIP0, clip1=CLIP1,
               r_dim=D_R_DIM,
               epochs=D_EPOCHS,
               batch_size=D_BATCH_SIZE,
               lr=D_LR, momentum=D_MOMENTUM, wd=D_WEIGHT_DECAY,
               pretrain_epochs=D_PRETRAIN_EP,
               pub_batch=D_PUB_BATCH,
               make_model_fn=None,
               pub_dataset=None,
               tag_suffix=""):
    """
    Core DP training loop — works for both Exp D (CIFAR-100) and Exp E (linear probe).

    pub_dataset: only needed for GEP subspace building (image datasets);
                 for Exp E (features), pass None and the (pub_x, pub_y) tuple is used.
    """
    tag      = f"{arm_name}_eps{eps:.1f}_seed{seed}{tag_suffix}"
    csv_path = os.path.join(out_dir, f"{tag}.csv")

    if os.path.exists(csv_path):
        print(f"[P10] {tag}: already done, skipping")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return float(rows[-1]["test_acc"])

    print(f"\n[P10] === {tag} ===")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Privacy accounting
    tmp_loader = DataLoader(priv_dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)
    steps_per_epoch = len(tmp_loader)
    T_steps         = epochs * steps_per_epoch
    q               = batch_size / len(priv_dataset)
    del tmp_loader

    sigma_van = _calibrate_sigma(eps, DELTA, q, T_steps)

    # GEP sigmas
    sigma_par, sigma_perp = _gep_sigmas(sigma_van)

    # Model
    model = make_model_fn().to(device)

    # Arm-specific setup
    is_pda  = arm_name in ("pda_dpmd", "pda_cw", "pda_ft", "pda_cw_ft")
    is_gep  = arm_name in ("gep", "gep_ft")
    is_warm = arm_name in ("vanilla_warm",)
    needs_pretrain = is_pda or is_warm

    if needs_pretrain:
        print(f"[P10]   Pretraining ({pretrain_epochs} ep)...")
        _pretrain_on_public(model, pub_x, pub_y, device, epochs=pretrain_epochs)

    # GEP subspace
    V = None
    if is_gep:
        subspace_src = pub_dataset if pub_dataset is not None else (pub_x, pub_y)
        V = _build_subspace(model, subspace_src, r_dim, device, seed=seed)

    # Optimizer + scheduler
    if exp == "E":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    priv_loader = DataLoader(priv_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_dataset, batch_size=512, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Small probe batch for train-loss estimation
    rng_np    = np.random.default_rng(seed + 100)
    probe_idx = rng_np.choice(len(priv_dataset), size=256, replace=False)
    probe_ds  = torch.utils.data.Subset(priv_dataset, probe_idx.tolist())
    probe_x, probe_y = next(iter(DataLoader(probe_ds, batch_size=256)))

    # CSV setup
    fieldnames = ["epoch", "train_loss", "test_acc", "lr"]
    if is_pda:
        fieldnames += ["alpha_mean", "cos_pub_priv"]

    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    best_acc    = 0.0
    step_global = 0
    ramp_steps  = RAMP_FRAC * T_steps

    for epoch in range(1, epochs + 1):
        model.train()

        # Refresh GEP subspace each epoch
        if is_gep:
            subspace_src = pub_dataset if pub_dataset is not None else (pub_x, pub_y)
            V = _build_subspace(model, subspace_src, r_dim, device, seed=seed)

        alpha_accum = []
        cos_accum   = []

        for x, y in priv_loader:
            optimizer.zero_grad(set_to_none=True)

            if not is_pda and not is_gep:
                # Vanilla / vanilla_warm
                flat_priv, _ = _vanilla_priv_step(
                    model, x, y, sigma_van, clip_c, device)
                _set_grads(model, flat_priv.to(device))

            elif is_gep:
                flat_grad = _gep_priv_step(
                    model, x, y, V, sigma_par, sigma_perp, clip0, clip1, device)
                _set_grads(model, flat_grad.to(device))

            else:
                # PDA arms
                flat_priv, priv_signal = _vanilla_priv_step(
                    model, x, y, sigma_van, clip_c, device)

                # Public gradient
                use_cw = arm_name in ("pda_cw", "pda_cw_ft")
                if use_cw:
                    g_pub = _pub_grad_cw_flat(model, pub_x, pub_y, device, pub_batch)
                else:
                    g_pub = _pub_grad_flat(model, pub_x, pub_y, device, batch_size=pub_batch)

                # Cosine similarity between g_pub direction and private signal
                g_pub_norm   = g_pub.norm().clamp(min=1e-8)
                signal_norm  = priv_signal.norm().clamp(min=1e-8)
                cos_sim = (g_pub @ priv_signal / (g_pub_norm * signal_norm)).item()
                cos_accum.append(cos_sim)

                # Norm-match: scale g_pub to private gradient magnitude
                priv_norm = flat_priv.norm()
                g_pub = g_pub * (priv_norm / g_pub_norm)

                # Alpha ramp: 0.9 → 1.0 over first ramp_frac * T steps
                alpha_t = ALPHA_START + (1.0 - ALPHA_START) * min(
                    1.0, step_global / max(ramp_steps, 1.0)
                )
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
        if is_pda:
            row["alpha_mean"]   = f"{np.mean(alpha_accum):.4f}" if alpha_accum else "nan"
            row["cos_pub_priv"] = f"{np.mean(cos_accum):.4f}"   if cos_accum   else "nan"

        writer.writerow(row)
        csv_file.flush()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f"{tag}_best.pt"))

        alpha_str = f"  α={np.mean(alpha_accum):.3f}" if alpha_accum else ""
        cos_str   = f"  cos={np.mean(cos_accum):.3f}"  if cos_accum   else ""
        print(f"  ep {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
              f"acc={test_acc:.4f}  best={best_acc:.4f}{alpha_str}{cos_str}")

    csv_file.close()
    torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}_final.pt"))
    print(f"[P10] done — final={test_acc:.4f}  best={best_acc:.4f}")
    return test_acc


# ---------------------------------------------------------------------------
# Experiment-level sweeps
# ---------------------------------------------------------------------------

def run_exp_d(arm_names, eps_list, n_seeds, data_root, device, out_dir):
    """Run Experiment D: CIFAR-100 from-scratch."""
    pub_ds, priv_ds, test_ds, pub_x, pub_y, class_sizes = \
        _build_d_datasets(data_root, seed=42)
    print(f"[P10] Exp D: EMNIST-balanced ({D_N_CLASSES} classes)  "
          f"pub={len(pub_ds)}  priv={len(priv_ds)}")
    print(f"[P10] Class sizes (pub): {class_sizes[:10]}...{class_sizes[-5:]}")

    for arm in arm_names:
        for eps in eps_list:
            for seed in range(n_seeds):
                _train_run(
                    arm_name=arm, exp="D", eps=eps, seed=seed,
                    pub_x=pub_x, pub_y=pub_y,
                    priv_dataset=priv_ds, test_dataset=test_ds,
                    device=device, out_dir=out_dir,
                    clip_c=CLIP_VAN, clip0=CLIP0, clip1=CLIP1,
                    r_dim=D_R_DIM,
                    epochs=D_EPOCHS, batch_size=D_BATCH_SIZE,
                    lr=D_LR, momentum=D_MOMENTUM, wd=D_WEIGHT_DECAY,
                    pretrain_epochs=D_PRETRAIN_EP,
                    pub_batch=D_PUB_BATCH,
                    make_model_fn=_make_model_d,
                    pub_dataset=pub_ds,
                )


def run_exp_e(arm_names, eps_list, n_seeds, backbone_name, data_root, device, out_dir):
    """Run Experiment E: linear probe on pretrained features."""
    feats = _extract_or_load(backbone_name, data_root, FEAT_CACHE, device)
    feature_dim = feats["feature_dim"]
    print(f"[P10] Exp E: {backbone_name}  feature_dim={feature_dim}")

    pub_x, pub_y, priv_ds, test_ds = _build_e_datasets(feats, seed=42)
    print(f"[P10] Exp E: pub={pub_x.shape[0]}  priv={len(priv_ds)}")

    r_dim = min(E_R_DIM, feature_dim // 2)

    for arm in arm_names:
        for eps in eps_list:
            for seed in range(n_seeds):
                _train_run(
                    arm_name=arm, exp="E", eps=eps, seed=seed,
                    pub_x=pub_x, pub_y=pub_y,
                    priv_dataset=priv_ds, test_dataset=test_ds,
                    device=device, out_dir=out_dir,
                    clip_c=CLIP_VAN, clip0=CLIP0, clip1=CLIP1,
                    r_dim=r_dim,
                    epochs=E_EPOCHS, batch_size=E_BATCH_SIZE,
                    lr=E_LR, momentum=E_MOMENTUM, wd=E_WEIGHT_DECAY,
                    pretrain_epochs=E_PRETRAIN_EP,
                    pub_batch=E_PUB_BATCH,
                    make_model_fn=lambda: _make_model_e(feature_dim),
                    pub_dataset=None,  # use feature tensors directly
                )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _load_csv(arm, eps, seed, out_dir, tag_suffix=""):
    tag  = f"{arm}_eps{eps:.1f}_seed{seed}{tag_suffix}"
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
    if not vals:
        return None, None
    return np.mean(vals), np.std(vals)


def _print_table(title, acc_fn, arm_names, eps_list, n_seeds, out_dir):
    print(f"\n{'='*72}")
    print(f" {title}")
    print(f"{'='*72}")
    header = f"{'arm':<20}" + "".join(f"  ε={e:.1f}  " for e in eps_list)
    print(header)
    print("-" * len(header))
    for arm in arm_names:
        row = f"{arm:<20}"
        for eps in eps_list:
            accs = [acc_fn(_load_csv(arm, eps, s, out_dir)) for s in range(n_seeds)]
            mu, sd = _mean_std(accs)
            row += f"  {'N/A':>12}  " if mu is None else f"  {mu*100:5.2f}±{sd*100:.2f}%  "
        print(row)


def _print_gap_table(baseline, arm_names, eps_list, n_seeds, out_dir):
    print(f"\n  Gap vs {baseline} (pp, final acc)")
    print(f"  {'arm':<20}" + "".join(f"  ε={e:.1f}  " for e in eps_list))
    for arm in arm_names:
        if arm == baseline:
            continue
        row = f"  {arm:<20}"
        for eps in eps_list:
            arm_accs = [_final_acc(_load_csv(arm,      eps, s, out_dir)) for s in range(n_seeds)]
            bas_accs = [_final_acc(_load_csv(baseline, eps, s, out_dir)) for s in range(n_seeds)]
            a_mu, _ = _mean_std(arm_accs)
            b_mu, _ = _mean_std(bas_accs)
            if a_mu is None or b_mu is None:
                row += f"  {'N/A':>10}  "
            else:
                row += f"  {(a_mu-b_mu)*100:+6.2f}pp  "
        print(row)


def _plot_curves(arm_names, eps, n_seeds, out_dir, title_prefix="", metric="test_acc"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(arm_names)))

    for arm, color in zip(arm_names, colors):
        runs = []
        for s in range(n_seeds):
            rows = _load_csv(arm, eps, s, out_dir)
            if rows is None:
                continue
            try:
                runs.append([float(r[metric]) for r in rows])
            except (KeyError, ValueError):
                continue
        if not runs:
            continue
        arr = np.array(runs)
        ep  = np.arange(1, arr.shape[1] + 1)
        mu  = arr.mean(0)
        sd  = arr.std(0)
        ax.plot(ep, mu, label=arm, color=color, lw=1.5)
        ax.fill_between(ep, mu - sd, mu + sd, alpha=0.15, color=color)

    ylabel = "Test Accuracy" if metric == "test_acc" else "Train Loss"
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix}ε={eps:.1f}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    stem = "_".join(arm_names[:4])
    path = os.path.join(out_dir, f"curves_{stem}_eps{eps:.1f}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P10] Saved {path}")


def _plot_cos_sim(arm_names, eps, n_seeds, out_dir, title_prefix=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(arm_names)))
    any_data = False

    for arm, color in zip(arm_names, colors):
        runs = []
        for s in range(n_seeds):
            rows = _load_csv(arm, eps, s, out_dir)
            if rows is None or "cos_pub_priv" not in rows[0]:
                continue
            try:
                runs.append([float(r["cos_pub_priv"]) for r in rows])
            except (KeyError, ValueError):
                continue
        if not runs:
            continue
        arr = np.array(runs)
        ep  = np.arange(1, arr.shape[1] + 1)
        mu  = arr.mean(0)
        sd  = arr.std(0)
        ax.plot(ep, mu, label=arm, color=color, lw=1.5)
        ax.fill_between(ep, mu - sd, mu + sd, alpha=0.15, color=color)
        any_data = True

    if not any_data:
        plt.close(fig)
        return

    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("cos(g_pub, g_priv signal)")
    ax.set_title(f"{title_prefix}Public–private gradient alignment  ε={eps:.1f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    stem = "_".join(arm_names[:4])
    path = os.path.join(out_dir, f"cos_sim_{stem}_eps{eps:.1f}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P10] Saved {path}")


def _run_analysis(out_dir, n_seeds_d=D_N_SEEDS_Q, n_seeds_e=E_N_SEEDS):
    print("\n[P10] ===== EXPERIMENT D: EMNIST-balanced (47 classes) =====")
    _print_table("Final accuracy — Exp D", _final_acc,
                 D_ARMS, D_EPS_LIST, n_seeds_d, out_dir)
    _print_table("Best accuracy — Exp D",  _best_acc,
                 D_ARMS, D_EPS_LIST, n_seeds_d, out_dir)
    _print_gap_table("vanilla", D_ARMS, D_EPS_LIST, n_seeds_d, out_dir)

    print("\n[P10] ===== EXPERIMENT E: LINEAR PROBE =====")
    _print_table("Final accuracy — Exp E", _final_acc,
                 E_ARMS, E_EPS_LIST, n_seeds_e, out_dir)
    _print_table("Best accuracy — Exp E",  _best_acc,
                 E_ARMS, E_EPS_LIST, n_seeds_e, out_dir)
    _print_gap_table("vanilla_ft", E_ARMS, E_EPS_LIST, n_seeds_e, out_dir)

    # Plots
    pda_d = [a for a in D_ARMS if "pda" in a]
    pda_e = [a for a in E_ARMS if "pda" in a]
    for eps in D_EPS_LIST:
        _plot_curves(D_ARMS, eps, n_seeds_d, out_dir, "Exp D EMNIST: ")
        _plot_cos_sim(pda_d, eps, n_seeds_d, out_dir, "Exp D: ")
    for eps in E_EPS_LIST:
        _plot_curves(E_ARMS, eps, n_seeds_e, out_dir, "Exp E linear probe: ")
        _plot_cos_sim(pda_e, eps, n_seeds_e, out_dir, "Exp E: ")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 10: CIFAR-100 and linear probe")
    ap.add_argument("--exp", choices=["D", "E"], help="Run Experiment D or E")
    ap.add_argument("--arm", help="Single arm name")
    ap.add_argument("--eps", type=float, help="Single epsilon value")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="1-seed quick sweep (default for Exp D)")
    ap.add_argument("--full",  action="store_true",
                    help="3-seed full sweep")
    ap.add_argument("--backbone", default="clip_vitb32",
                    choices=["clip_vitb32", "resnet50"],
                    help="Pretrained backbone for Exp E (default: clip_vitb32)")
    ap.add_argument("--extract_features", action="store_true",
                    help="Extract and cache features for Exp E, then exit")
    ap.add_argument("--analysis_only", action="store_true")
    ap.add_argument("--gpu",       type=int, default=0)
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--out_dir",   default=RESULTS_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[P10] Device: {device}  |  backbone: {args.backbone}")

    if args.analysis_only:
        n_seeds_d = D_N_SEEDS_F if args.full else D_N_SEEDS_Q
        _run_analysis(args.out_dir, n_seeds_d=n_seeds_d, n_seeds_e=E_N_SEEDS)
        return

    if args.extract_features:
        _extract_or_load(args.backbone, args.data_root, FEAT_CACHE, device)
        return

    n_seeds_d = D_N_SEEDS_F if args.full else D_N_SEEDS_Q
    n_seeds_e = E_N_SEEDS

    # ---- Single arm mode ----
    if args.arm is not None:
        exp = args.exp or ARM_TO_EXP.get(args.arm, "D")
        eps_list = [args.eps] if args.eps is not None else \
                   (D_EPS_LIST if exp == "D" else E_EPS_LIST)
        if exp == "D":
            pub_ds, priv_ds, test_ds, pub_x, pub_y, _ = \
                _build_d_datasets(args.data_root, seed=42)
            for eps in eps_list:
                _train_run(
                    arm_name=args.arm, exp="D", eps=eps, seed=args.seed,
                    pub_x=pub_x, pub_y=pub_y,
                    priv_dataset=priv_ds, test_dataset=test_ds,
                    device=device, out_dir=args.out_dir,
                    clip_c=CLIP_VAN, clip0=CLIP0, clip1=CLIP1,
                    r_dim=D_R_DIM, epochs=D_EPOCHS, batch_size=D_BATCH_SIZE,
                    lr=D_LR, momentum=D_MOMENTUM, wd=D_WEIGHT_DECAY,
                    pretrain_epochs=D_PRETRAIN_EP, pub_batch=D_PUB_BATCH,
                    n_classes=100, make_model_fn=_make_model_d, pub_dataset=pub_ds,
                )
        else:
            feats = _extract_or_load(args.backbone, args.data_root, FEAT_CACHE, device)
            feature_dim = feats["feature_dim"]
            pub_x, pub_y, priv_ds, test_ds = _build_e_datasets(feats, seed=42)
            r_dim = min(E_R_DIM, feature_dim // 2)
            for eps in eps_list:
                _train_run(
                    arm_name=args.arm, exp="E", eps=eps, seed=args.seed,
                    pub_x=pub_x, pub_y=pub_y,
                    priv_dataset=priv_ds, test_dataset=test_ds,
                    device=device, out_dir=args.out_dir,
                    clip_c=CLIP_VAN, clip0=CLIP0, clip1=CLIP1,
                    r_dim=r_dim, epochs=E_EPOCHS, batch_size=E_BATCH_SIZE,
                    lr=E_LR, momentum=E_MOMENTUM, wd=E_WEIGHT_DECAY,
                    pretrain_epochs=E_PRETRAIN_EP, pub_batch=E_PUB_BATCH,
                    n_classes=10, make_model_fn=lambda: _make_model_e(feature_dim),
                )
        return

    # ---- Experiment-level sweep ----
    if args.exp == "D":
        run_exp_d(D_ARMS, D_EPS_LIST, n_seeds_d, args.data_root, device, args.out_dir)
        _run_analysis(args.out_dir, n_seeds_d=n_seeds_d, n_seeds_e=n_seeds_e)
        return

    if args.exp == "E":
        run_exp_e(E_ARMS, E_EPS_LIST, n_seeds_e, args.backbone,
                  args.data_root, device, args.out_dir)
        _run_analysis(args.out_dir, n_seeds_d=n_seeds_d, n_seeds_e=n_seeds_e)
        return

    # Default: run both
    print("[P10] No --exp specified. Running both D and E sequentially.")
    run_exp_d(D_ARMS, D_EPS_LIST, n_seeds_d, args.data_root, device, args.out_dir)
    run_exp_e(E_ARMS, E_EPS_LIST, n_seeds_e, args.backbone,
              args.data_root, device, args.out_dir)
    _run_analysis(args.out_dir, n_seeds_d=n_seeds_d, n_seeds_e=n_seeds_e)


if __name__ == "__main__":
    main()
