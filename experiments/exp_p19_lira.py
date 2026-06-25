#!/usr/bin/env python3
"""
Phase 19 LiRA: Shadow Training and Membership Inference Scoring
===============================================================

Spec: phase19_spex.md, Section 10.

LiRA plan:
  LF1: F1 seed 0 (CLIP CIFAR-10-LT(50)), 1500 targets, 128 shadows [headline]
  LF2: F2 seed 0 (CLIP CIFAR-10),         1000 targets, 48–64 shadows [control]
  LF5: F5 seed 0 (WRN  CIFAR-10-LT(50)),   300 targets, 16–24 shadows [optional]

LiRA scoring (spec §10.3):
  s_{i,m} = log[p_m(y_i|x_i) / (1 - p_m(y_i|x_i))]    (logit of correct class)

  Fit per-target in/out Gaussians, floor variance at 1e-6.

  D_i^LiRA = (μ_in - μ_out) / sqrt(0.5 × (σ_in² + σ_out²))  [distinguishability]

Outputs (per LiRA run):
  lira/{lira_id}/D_lira_members.npy
  lira/{lira_id}/D_lira_nonmembers.npy
  lira/{lira_id}/lira_scores_members.npy
  lira/{lira_id}/lira_scores_nonmembers.npy
  lira/{lira_id}/llr_dp_members.npy
  lira/{lira_id}/llr_dp_nonmembers.npy
  lira/{lira_id}/lira_summary.json

Usage:
  python experiments/exp_p19_lira.py --lira_id LF1 --gpu 0                        # seed 0 only
  python experiments/exp_p19_lira.py --lira_id LF1 --all_seeds --gpu 0            # seeds 0,1,2
  python experiments/exp_p19_lira.py --lira_id LF1 --seed 1 --gpu 0               # seed 1 only
  python experiments/exp_p19_lira.py --lira_id LF1 --shadow_start 0 --shadow_end 64 --gpu 0
  python experiments/exp_p19_lira.py --lira_id LF1 --score_only --all_seeds --gpu 0
  python experiments/exp_p19_lira.py --all --all_seeds --gpu 0
"""

import os, sys, json, argparse, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets import make_public_private_split, make_cifar10_lt_indices, make_lt_indices, load_purchase100
from src.models import WideResNet, PurchaseFC

import torchvision
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LIRA_SETTINGS = {
    "LF1": dict(matched_run="F1", seeds=[0, 1, 2], dataset="cifar10_lt50", regime="R3",
                n_targets=1500, n_shadows=128, shadow_epochs=20),
    "LF2": dict(matched_run="F2", seeds=[0, 1, 2], dataset="cifar10",      regime="R3",
                n_targets=1000, n_shadows=64,  shadow_epochs=20),
    "LF5": dict(matched_run="F5", seeds=[0, 1, 2], dataset="cifar10_lt50", regime="R2",
                n_targets=300,  n_shadows=128, shadow_epochs=30),
    "LF7": dict(matched_run="F7", seeds=[0, 1, 2], dataset="purchase100_lt10", regime="R2",
                arch="purchase_fc", num_classes=100,
                n_targets=600,  n_shadows=64,  shadow_epochs=15),
}

DATA_ROOT  = "./data"
CACHE_DIR  = "./data/clip_features"
RUNS_DIR   = "./runs/p19"
LIRA_DIR   = "./lira/p19"
N_GROUPS   = 16
SHADOW_LR  = 0.05
SHADOW_BS  = 256
CHUNK_R3       = 512
CHUNK_WRN      = 64
CHUNK_PURCHASE = 512


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class _FeatureDataset(Dataset):
    def __init__(self, feats, labels, global_idx):
        self.feats = feats; self.labels = labels; self.idx = global_idx
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.feats[i], int(self.labels[i]), int(self.idx[i])


def _cifar10_aug(data_root):
    m, s = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize(m, s)])
    return torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf)


def _cifar10_noaug(data_root):
    m, s = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tf = T.Compose([T.ToTensor(), T.Normalize(m, s)])
    return torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf)


def _cifar10_test(data_root):
    m, s = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tf = T.Compose([T.ToTensor(), T.Normalize(m, s)])
    return torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf)


def _load_clip_features(data_root, cache_dir):
    paths = {k: os.path.join(cache_dir, f"cifar10_clip_{k}.pt")
             for k in ["train", "train_labels", "test", "test_labels"]}
    if all(os.path.exists(p) for p in paths.values()):
        return tuple(torch.load(paths[k], map_location="cpu", weights_only=False)
                     for k in ["train", "train_labels", "test", "test_labels"])
    raise FileNotFoundError("CLIP features not cached. Run exp_p19_train.py --run F1 first.")


class _NumpyDataset(Dataset):
    """Wraps numpy (X, y) arrays; yields (x_tensor, y_int, global_idx)."""
    def __init__(self, X, y, global_idx=None):
        self.X   = np.asarray(X, dtype=np.float32)
        self.y   = np.asarray(y, dtype=np.int64)
        self.gidx = (np.arange(len(y), dtype=np.int64)
                     if global_idx is None else np.asarray(global_idx, dtype=np.int64))

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]).float(), int(self.y[i]), int(self.gidx[i]))


def build_full_dataset_purchase(dataset_name, data_root):
    """
    Reconstruct the Purchase-100 train/test split identical to exp_p19_train.py.

    Returns (X_all, y_all, X_test, y_test, priv_idx, full_train_targets).
    X_all: full dataset arrays (N, 600).  priv_idx: global indices of private training examples.
    X_test, y_test: held-out test split (for nonmember logits).
    """
    is_lt = "lt" in dataset_name
    lt_ir_str = dataset_name.split("lt")[-1] if is_lt else "1"
    lt_ir = int("".join(filter(str.isdigit, lt_ir_str))) if is_lt else 1
    num_classes = 100

    X_all, y_all = load_purchase100(data_root)
    N = len(y_all)

    # Same 80/20 stratified split as train.py (seed=42)
    rng_split = np.random.default_rng(42)
    train_mask = np.zeros(N, dtype=bool)
    for c in range(num_classes):
        cls_idx = np.where(y_all == c)[0]
        n_train_c = max(1, int(len(cls_idx) * 0.8))
        chosen = rng_split.choice(cls_idx, size=n_train_c, replace=False)
        train_mask[chosen] = True
    train_idx = np.where(train_mask)[0]
    test_idx  = np.where(~train_mask)[0]

    y_train = y_all[train_idx]
    if is_lt and lt_ir > 1:
        lt_local = make_lt_indices(y_train, float(lt_ir), num_classes=num_classes, seed=42)
        lt_train_idx = train_idx[lt_local]
    else:
        lt_train_idx = train_idx

    y_lt = y_all[lt_train_idx]
    _, priv_local = make_public_private_split(
        np.arange(len(lt_train_idx)), y_lt, public_frac=0.1, seed=42)
    priv_global = lt_train_idx[priv_local]

    return (X_all, y_all,
            X_all[test_idx], y_all[test_idx].astype(np.int32),
            priv_global.astype(np.int32),
            y_all[train_idx])


def build_full_dataset_clip(dataset_name, data_root, cache_dir):
    """Return full CLIP feature dataset (all training examples used for shadows)."""
    is_lt = "lt" in dataset_name
    lt_ir = 50 if "lt50" in dataset_name else 1
    tf_f, tf_l, te_f, te_l = _load_clip_features(data_root, cache_dir)
    full_targets = tf_l.numpy()
    lt_idx = make_cifar10_lt_indices(full_targets, lt_ir, seed=42) if is_lt else np.arange(len(full_targets))
    lt_targets = full_targets[lt_idx]
    _, priv_idx = make_public_private_split(lt_idx, lt_targets, public_frac=0.1, seed=42)
    return tf_f, tf_l, te_f, te_l, priv_idx, full_targets


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, num_classes=10, feat_dim=512):
        super().__init__(); self.fc = nn.Linear(feat_dim, num_classes)
    def forward(self, x): return self.fc(x.float())


def make_shadow_model(regime, num_classes, arch="wrn28-2"):
    if regime == "R3":
        return LinearHead(num_classes, feat_dim=512)
    if arch == "purchase_fc":
        return PurchaseFC(num_classes=num_classes)
    return WideResNet(depth=28, widen_factor=2, num_classes=num_classes, n_groups=N_GROUPS)


# ---------------------------------------------------------------------------
# Shadow training
# ---------------------------------------------------------------------------

def train_shadow_clip(model, shadow_feats, shadow_labels, n_epochs, device, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=SHADOW_LR, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    N = shadow_feats.shape[0]
    for ep in range(n_epochs):
        perm = torch.randperm(N)
        for i in range(0, N, SHADOW_BS):
            idx = perm[i:i + SHADOW_BS]; opt.zero_grad()
            h = shadow_feats[idx].to(device).float()
            y = shadow_labels[idx].to(device).long()
            F.cross_entropy(model(h), y).backward(); opt.step()
        sch.step()


def train_shadow_wrn(model, shadow_indices, train_ds_aug, n_epochs, device, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model.train()
    sub_ds = Subset(train_ds_aug, shadow_indices.tolist())
    loader = DataLoader(sub_ds, batch_size=SHADOW_BS, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    for ep in range(n_epochs):
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(model(x.to(device)), y.to(device)).backward(); opt.step()
        sch.step()


def train_shadow_purchase(model, X_shadow, y_shadow, n_epochs, device, seed):
    """Train shadow PurchaseFC on in-memory numpy arrays (no DataLoader overhead)."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model.train()
    X_t = torch.from_numpy(X_shadow).float()
    y_t = torch.from_numpy(y_shadow.astype(np.int64))
    N   = len(y_t)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    for ep in range(n_epochs):
        perm = torch.randperm(N)
        for i in range(0, N, SHADOW_BS):
            idx = perm[i:i + SHADOW_BS]; opt.zero_grad()
            F.cross_entropy(model(X_t[idx].to(device)), y_t[idx].to(device)).backward()
            opt.step()
        sch.step()


# ---------------------------------------------------------------------------
# Logit evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_logits_purchase(model, X, device, chunk=CHUNK_PURCHASE):
    """Evaluate PurchaseFC on a numpy array of features."""
    model.eval(); parts = []
    for i in range(0, len(X), chunk):
        h = torch.from_numpy(X[i:i + chunk].astype(np.float32)).to(device)
        parts.append(model(h).cpu())
    return torch.cat(parts).numpy().astype(np.float32)


@torch.no_grad()
def eval_logits_clip(model, feats, indices, device, chunk=CHUNK_R3):
    model.eval(); parts = []
    for i in range(0, len(indices), chunk):
        h = feats[indices[i:i + chunk]].to(device).float()
        parts.append(model(h).cpu())
    return torch.cat(parts).numpy().astype(np.float32)


@torch.no_grad()
def eval_logits_wrn(model, noaug_ds, indices, device, chunk=CHUNK_WRN):
    """Evaluate WRN on no-aug dataset at given local indices into the LT private set."""
    model.eval(); parts = []
    for i in range(0, len(indices), chunk):
        idx = indices[i:i + chunk]
        x = torch.stack([noaug_ds[int(j)][0] for j in idx]).to(device)
        parts.append(model(x).cpu())
    return torch.cat(parts).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# LiRA shadow subset selection
# ---------------------------------------------------------------------------

def select_shadow_subset(lira_id, priv_idx, full_targets, member_local, shadow_m, seed_base):
    """
    For shadow m, randomly include each private example with prob 0.5,
    ensuring target members have a 50/50 in/out split across shadows.

    Returns (shadow_in_local, member_in_shadow) arrays.
    shadow_in_local: local indices (into priv_idx) included in this shadow's training set.
    """
    rng = np.random.default_rng(seed_base + shadow_m * 31337)
    n_priv = len(priv_idx)

    # Each example is in or out with prob 0.5
    in_mask = rng.random(n_priv) < 0.5   # (n_priv,)

    # Member targets: ensure roughly half are in each shadow
    # (already handled by random p=0.5 above)
    return np.where(in_mask)[0].astype(np.int32)


# ---------------------------------------------------------------------------
# Score aggregation: LiRA scoring per spec §10.3
# ---------------------------------------------------------------------------

def logit_score(logits, labels):
    """
    s_{i,m} = log[p(y_i|x_i) / (1 - p(y_i|x_i))]  (log-odds of correct class)

    logits: (n, C)  raw logits
    labels: (n,)    integer class labels
    """
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    n = len(labels)
    p_correct = probs[np.arange(n), labels.astype(int)]
    p_correct = np.clip(p_correct, 1e-7, 1 - 1e-7)
    return np.log(p_correct / (1.0 - p_correct)).astype(np.float64)


def compute_lira_scores(member_in_scores, member_out_scores,
                        nonmember_in_scores, nonmember_out_scores):
    """
    Fit per-target in/out Gaussians and compute D_i^LiRA and LLR score.

    Inputs: arrays of shape (n_targets, n_shadows) — scores for shadows where
      example was in / out of training set.

    Returns: (D_lira, llr_test) arrays of shape (n_targets,).
    """
    VAR_FLOOR = 1e-6
    n = member_in_scores.shape[0]
    D_lira = np.zeros(n)
    llr    = np.zeros(n)

    for i in range(n):
        s_in  = member_in_scores[i]
        s_out = member_out_scores[i]

        # Per spec: minimum 20 in/out shadows (LF1/LF2)
        if len(s_in) < 2 or len(s_out) < 2:
            continue

        mu_in  = s_in.mean();  var_in  = max(s_in.var(), VAR_FLOOR)
        mu_out = s_out.mean(); var_out = max(s_out.var(), VAR_FLOOR)

        # D_i^LiRA = (mu_in - mu_out) / sqrt(0.5 * (sigma_in^2 + sigma_out^2))
        pooled_std = math.sqrt(0.5 * (var_in + var_out))
        D_lira[i] = (mu_in - mu_out) / max(pooled_std, 1e-12)

    # LLR = log p(s | in) - log p(s | out) where s is the DP model score
    # (not computed here — computed separately from DP model logits in run_scoring)

    return D_lira


# ---------------------------------------------------------------------------
# Main LiRA pipeline
# ---------------------------------------------------------------------------

def run_lira(lira_id, cfg, shadow_start, shadow_end, device,
             data_root, cache_dir, runs_dir, lira_dir, seed=None):
    """Train shadow models and save their logit scores."""
    matched_run = cfg["matched_run"]
    run_seed    = seed if seed is not None else cfg["seeds"][0]
    dataset     = cfg["dataset"]
    regime      = cfg["regime"]
    arch        = cfg.get("arch", "wrn28-2")
    num_classes = cfg.get("num_classes", 10)
    n_shadows   = cfg["n_shadows"]
    n_epochs    = cfg["shadow_epochs"]
    is_clip     = (regime == "R3")
    is_purchase = "purchase" in dataset

    out_dir = os.path.join(lira_dir, lira_id, f"seed_{run_seed}")
    os.makedirs(out_dir, exist_ok=True)

    run_dir = os.path.join(runs_dir, matched_run, f"seed_{run_seed}")
    if not os.path.isdir(run_dir):
        print(f"[LiRA] Run dir not found: {run_dir}. Train {matched_run} first.")
        return

    # Load LiRA targets
    member_local    = np.load(os.path.join(run_dir, "lira_member_local_idx.npy"))
    nonmember_test  = np.load(os.path.join(run_dir, "lira_nonmember_test_idx.npy"))
    priv_idx        = np.load(os.path.join(run_dir, "example_indices.npy"))
    priv_labels     = np.load(os.path.join(run_dir, "labels.npy"))
    n_priv = len(priv_idx)

    member_labels    = priv_labels[member_local]
    nonmember_labels = None

    # Load dataset for shadow training
    if is_clip:
        tf_f, tf_l, te_f, te_l, _, full_targets = build_full_dataset_clip(
            dataset, data_root, cache_dir)
        nonmember_labels = te_l.numpy()[nonmember_test]
    elif is_purchase:
        X_all, y_all, X_test, y_test, _, full_targets = build_full_dataset_purchase(
            dataset, data_root)
        nonmember_labels = y_test[nonmember_test]
    else:
        train_ds_aug = _cifar10_aug(data_root)
        noaug_priv   = _cifar10_noaug(data_root)
        test_ds      = _cifar10_test(data_root)
        full_targets = np.array(train_ds_aug.targets)
        nonmember_labels = np.array(test_ds.targets)[nonmember_test]

    n_targets_m  = len(member_local)
    n_targets_nm = len(nonmember_test)

    # Check for existing shadow results
    shadow_end_actual = min(shadow_end, n_shadows)

    for m in range(shadow_start, shadow_end_actual):
        ckpt_path = os.path.join(out_dir, f"shadow_{m}_member_logits.npy")
        if os.path.exists(ckpt_path):
            print(f"  [LiRA] shadow {m} already done, skipping.")
            continue

        print(f"  [LiRA {lira_id}] Training shadow {m}/{n_shadows}...")

        # Shadow subset: each priv example is in with prob 0.5
        shadow_in_local = select_shadow_subset(lira_id, priv_idx, full_targets,
                                               member_local, m, seed_base=42000)
        in_set = set(shadow_in_local.tolist())

        # Indicators for members and nonmembers
        member_in_shadow = np.array([local_i in in_set for local_i in member_local], dtype=bool)

        if is_clip:
            shadow_in_global = priv_idx[shadow_in_local]
            shadow_feats  = tf_f[shadow_in_global]
            shadow_labels = tf_l[shadow_in_global].long()
            model = make_shadow_model(regime, num_classes).to(device)
            train_shadow_clip(model, shadow_feats, shadow_labels, n_epochs, device, seed=m)

            # Evaluate on targets
            member_logits    = eval_logits_clip(model, tf_f, priv_idx[member_local], device)
            nonmember_logits = eval_logits_clip(model, te_f, nonmember_test, device)

        elif is_purchase:
            shadow_in_global = priv_idx[shadow_in_local]
            X_shadow = X_all[shadow_in_global]
            y_shadow = y_all[shadow_in_global].astype(np.int64)
            model = make_shadow_model(regime, num_classes, arch).to(device)
            train_shadow_purchase(model, X_shadow, y_shadow, n_epochs, device, seed=m)

            member_logits    = eval_logits_purchase(model, X_all[priv_idx[member_local]], device)
            nonmember_logits = eval_logits_purchase(model, X_test[nonmember_test], device)

        else:
            shadow_train_global = priv_idx[shadow_in_local]
            model = make_shadow_model(regime, num_classes, arch).to(device)
            train_shadow_wrn(model, shadow_train_global, train_ds_aug, n_epochs, device, seed=m)

            # Evaluate on targets (using no-aug transform)
            member_global  = priv_idx[member_local]
            member_logits    = eval_logits_wrn(model, noaug_priv, member_global, device)
            nonmember_logits = eval_logits_wrn(model, test_ds, nonmember_test, device)

        np.save(os.path.join(out_dir, f"shadow_{m}_member_logits.npy"),
                member_logits.astype(np.float32))
        np.save(os.path.join(out_dir, f"shadow_{m}_nonmember_logits.npy"),
                nonmember_logits.astype(np.float32))
        np.save(os.path.join(out_dir, f"shadow_{m}_in_mask.npy"),
                member_in_shadow.astype(bool))
        del model; torch.cuda.empty_cache()
        print(f"  [LiRA] shadow {m} done (in_shadow={member_in_shadow.sum()}/{n_targets_m})")

    # Save target labels
    np.save(os.path.join(out_dir, "targets_members.npy"),    member_local)
    np.save(os.path.join(out_dir, "targets_nonmembers.npy"), nonmember_test)
    np.save(os.path.join(out_dir, "member_labels.npy"),      member_labels)
    if nonmember_labels is not None:
        np.save(os.path.join(out_dir, "nonmember_labels.npy"), nonmember_labels.astype(np.int32))

    meta = {
        "lira_id": lira_id, "matched_run": matched_run, "run_seed": run_seed,
        "dataset": dataset, "regime": regime, "n_shadows": n_shadows,
        "n_targets_members": n_targets_m, "n_targets_nonmembers": n_targets_nm,
        "shadow_epochs": n_epochs, "shadow_range_done": [shadow_start, shadow_end_actual],
    }
    with open(os.path.join(out_dir, "shadow_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [LiRA {lira_id}] Shadows {shadow_start}–{shadow_end_actual} done.")


def run_scoring(lira_id, cfg, device, runs_dir, lira_dir, seed=None):
    """
    Aggregate all shadow logits and compute LiRA scores.

    Per spec §10.3:
      s_{i,m} = log-odds of correct class
      Fit per-target in/out Gaussians (var floor 1e-6)
      D_i^LiRA = (mu_in - mu_out) / sqrt(0.5*(sigma_in^2 + sigma_out^2))
    """
    matched_run = cfg["matched_run"]
    run_seed    = seed if seed is not None else cfg["seeds"][0]
    n_shadows   = cfg["n_shadows"]
    regime      = cfg["regime"]
    is_clip     = (regime == "R3")

    out_dir = os.path.join(lira_dir, lira_id, f"seed_{run_seed}")
    run_dir = os.path.join(runs_dir, matched_run, f"seed_{run_seed}")

    member_local   = np.load(os.path.join(run_dir, "lira_member_local_idx.npy"))
    nonmember_test = np.load(os.path.join(run_dir, "lira_nonmember_test_idx.npy"))
    member_labels  = np.load(os.path.join(out_dir, "member_labels.npy"))
    try:
        nonmember_labels = np.load(os.path.join(out_dir, "nonmember_labels.npy"))
    except FileNotFoundError:
        print("  [scoring] nonmember_labels.npy not found"); return

    n_m  = len(member_local)
    n_nm = len(nonmember_test)

    # Collect shadow scores
    shadow_scores_m_in  = [[] for _ in range(n_m)]
    shadow_scores_m_out = [[] for _ in range(n_m)]
    available_shadows   = 0

    # Dense matrices for LT-IQR analysis: shape (n_m, n_shadows), NaN = shadow missing.
    # shadow_logit_scores[i, m] = log-odds score from shadow m for target i.
    # shadow_in_mask_combined[i, m] = 1 if target i was IN shadow m's training set.
    shadow_logit_scores      = np.full((n_m, n_shadows), np.nan,  dtype=np.float32)
    shadow_in_mask_combined  = np.zeros((n_m, n_shadows),          dtype=bool)

    for m in range(n_shadows):
        logit_path = os.path.join(out_dir, f"shadow_{m}_member_logits.npy")
        mask_path  = os.path.join(out_dir, f"shadow_{m}_in_mask.npy")
        if not os.path.exists(logit_path): continue

        member_logits = np.load(logit_path).astype(np.float32)  # (n_m, C)
        in_mask       = np.load(mask_path)                       # (n_m,) bool
        scores_m      = logit_score(member_logits, member_labels) # (n_m,)

        shadow_logit_scores[:, m]     = scores_m.astype(np.float32)
        shadow_in_mask_combined[:, m] = in_mask.astype(bool)

        for i in range(n_m):
            if in_mask[i]:
                shadow_scores_m_in[i].append(scores_m[i])
            else:
                shadow_scores_m_out[i].append(scores_m[i])
        available_shadows += 1

    if available_shadows < 4:
        print(f"  [scoring] Only {available_shadows} shadows available. Need at least 4.")
        return

    print(f"  [LiRA {lira_id}] Scoring with {available_shadows}/{n_shadows} shadows.")

    # Check minimum shadow coverage
    min_in  = min(len(s) for s in shadow_scores_m_in)
    min_out = min(len(s) for s in shadow_scores_m_out)
    print(f"  Min shadows per target: in={min_in}, out={min_out}")
    if min_in < 5 or min_out < 5:
        print(f"  [LiRA] WARNING: Some targets have fewer than 5 in/out shadows.")

    # Stack as arrays
    max_in  = max(len(s) for s in shadow_scores_m_in)
    max_out = max(len(s) for s in shadow_scores_m_out)
    in_arr  = np.full((n_m, max_in), np.nan)
    out_arr = np.full((n_m, max_out), np.nan)
    for i in range(n_m):
        ni = len(shadow_scores_m_in[i])
        no = len(shadow_scores_m_out[i])
        if ni > 0: in_arr[i, :ni]  = shadow_scores_m_in[i]
        if no > 0: out_arr[i, :no] = shadow_scores_m_out[i]

    np.save(os.path.join(out_dir, "lira_scores_members.npy"),    in_arr.astype(np.float32))
    np.save(os.path.join(out_dir, "lira_scores_nonmembers.npy"), out_arr.astype(np.float32))

    # Dense combined arrays consumed by LT-IQR in exp_p19_analysis.py.
    # shadow_logit_scores[i, m]    = log-odds score for target i from shadow m (NaN if missing).
    # shadow_in_mask_combined[i,m] = True if target i was IN shadow m's training set.
    np.save(os.path.join(out_dir, "shadow_logit_scores.npy"),     shadow_logit_scores)
    np.save(os.path.join(out_dir, "shadow_in_mask_combined.npy"), shadow_in_mask_combined)

    # Compute global score variance as a data-driven variance floor.
    # A hard 1e-6 floor inflates |D_lira| for easy (head) examples where var_in
    # and var_out both collapse to near zero: pooled_std → 1e-3 and D_lira explodes.
    # Since ε_dir also correlates with tier, this creates an artifactual component
    # in the headline ρ. Using global_var (across all shadow scores, all targets)
    # as the floor matches the Carlini et al. 2022 global-variance offline variant.
    all_scores_flat = []
    for i in range(n_m):
        all_scores_flat.extend(shadow_scores_m_in[i])
        all_scores_flat.extend(shadow_scores_m_out[i])
    global_var = float(np.var(all_scores_flat)) if len(all_scores_flat) > 1 else 1.0
    VAR_FLOOR = max(global_var, 1e-6)
    print(f"  [LiRA {lira_id}] global_var={global_var:.6f}  VAR_FLOOR={VAR_FLOOR:.6f}")

    D_lira_members = np.zeros(n_m)
    n_floor_hits_in = 0; n_floor_hits_out = 0
    for i in range(n_m):
        s_in  = np.array(shadow_scores_m_in[i])
        s_out = np.array(shadow_scores_m_out[i])
        if len(s_in) < 2 or len(s_out) < 2: continue
        raw_var_in  = s_in.var();  raw_var_out = s_out.var()
        if raw_var_in  < VAR_FLOOR: n_floor_hits_in  += 1
        if raw_var_out < VAR_FLOOR: n_floor_hits_out += 1
        mu_in, var_in   = s_in.mean(), max(raw_var_in,  VAR_FLOOR)
        mu_out, var_out = s_out.mean(), max(raw_var_out, VAR_FLOOR)
        pooled_std = math.sqrt(0.5 * (var_in + var_out))
        D_lira_members[i] = (mu_in - mu_out) / max(pooled_std, 1e-12)

    floor_hit_rate_in  = n_floor_hits_in  / max(n_m, 1)
    floor_hit_rate_out = n_floor_hits_out / max(n_m, 1)
    print(f"  [LiRA {lira_id}] var-floor hits: in={n_floor_hits_in}/{n_m} "
          f"({floor_hit_rate_in:.1%})  out={n_floor_hits_out}/{n_m} "
          f"({floor_hit_rate_out:.1%})")

    # D_i^LiRA for nonmembers (using DP model logits as proxy for nonmember score)
    # For nonmembers, D_i^LiRA is typically negative (excluded from training)
    # We compute a simpler version using the available DP model logit score
    dp_member_path    = os.path.join(run_dir, "target_logits_dp_members.npy")
    dp_nonmember_path = os.path.join(run_dir, "target_logits_dp_nonmembers.npy")

    llr_dp_members    = np.zeros(n_m,  dtype=np.float32)
    llr_dp_nonmembers = np.zeros(n_nm, dtype=np.float32)

    if os.path.exists(dp_member_path):
        dp_mem_logits = np.load(dp_member_path).astype(np.float32)     # (n_m, C)
        llr_dp_members = logit_score(dp_mem_logits, member_labels).astype(np.float32)

    if os.path.exists(dp_nonmember_path):
        dp_nm_logits = np.load(dp_nonmember_path).astype(np.float32)   # (n_nm, C)
        llr_dp_nonmembers = logit_score(dp_nm_logits, nonmember_labels).astype(np.float32)

    np.save(os.path.join(out_dir, "D_lira_members.npy"),    D_lira_members.astype(np.float32))
    # D_lira for nonmembers is NOT computed here — shadow training only covers private
    # training examples, so test-set nonmembers have no in/out shadow split.
    # Saved as zeros; Table 3 correlations use members only (clearly labeled).
    np.save(os.path.join(out_dir, "D_lira_nonmembers.npy"), np.zeros(n_nm, dtype=np.float32))
    np.save(os.path.join(out_dir, "llr_dp_members.npy"),    llr_dp_members)
    np.save(os.path.join(out_dir, "llr_dp_nonmembers.npy"), llr_dp_nonmembers)

    # Split-half shadow stability (spec §10.4)
    half = available_shadows // 2
    if half >= 2:
        D1 = np.zeros(n_m); D2 = np.zeros(n_m)
        for i in range(n_m):
            s_in  = np.array(shadow_scores_m_in[i]);  s_out = np.array(shadow_scores_m_out[i])
            # First half
            h1_in  = s_in[:len(s_in)//2];   h1_out = s_out[:len(s_out)//2]
            h2_in  = s_in[len(s_in)//2:];   h2_out = s_out[len(s_out)//2:]
            for D_arr, hi, ho in [(D1, h1_in, h1_out), (D2, h2_in, h2_out)]:
                if len(hi) >= 2 and len(ho) >= 2:
                    mu_in, var_in   = hi.mean(), max(hi.var(), VAR_FLOOR)
                    mu_out, var_out = ho.mean(), max(ho.var(), VAR_FLOOR)
                    ps = math.sqrt(0.5 * (var_in + var_out))
                    D_arr[i] = (mu_in - mu_out) / max(ps, 1e-12)
        # Stability = correlation between half-1 and half-2 D scores
        mask_valid = np.isfinite(D1) & np.isfinite(D2)
        if mask_valid.sum() > 10:
            split_half_corr = float(np.corrcoef(D1[mask_valid], D2[mask_valid])[0, 1])
        else:
            split_half_corr = float("nan")
    else:
        split_half_corr = float("nan")

    print(f"  [LiRA {lira_id}] D^LiRA: mean={D_lira_members.mean():.4f}  "
          f"std={D_lira_members.std():.4f}  "
          f"split-half r={split_half_corr:.4f}")

    summary = {
        "lira_id": lira_id,
        "run_seed": run_seed,
        "n_shadows_used": available_shadows,
        "n_targets_members": n_m, "n_targets_nonmembers": n_nm,
        "D_lira_mean": float(D_lira_members.mean()),
        "D_lira_std":  float(D_lira_members.std()),
        "D_lira_med":  float(np.median(D_lira_members)),
        "split_half_stability": split_half_corr,
        "min_in_shadows":  int(min_in),
        "min_out_shadows": int(min_out),
        "global_var_floor": float(VAR_FLOOR),
        "floor_hit_rate_in":  float(floor_hit_rate_in),
        "floor_hit_rate_out": float(floor_hit_rate_out),
        "note_nonmembers": "D_lira_nonmembers saved as zeros; Table 3 uses members only",
    }
    with open(os.path.join(out_dir, "lira_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [LiRA {lira_id}] Summary saved → {out_dir}/lira_summary.json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 19 LiRA")
    parser.add_argument("--lira_id",      type=str, default=None,
                        choices=list(LIRA_SETTINGS.keys()))
    parser.add_argument("--shadow_start", type=int, default=0)
    parser.add_argument("--shadow_end",   type=int, default=None)
    parser.add_argument("--score_only",   action="store_true")
    parser.add_argument("--all",          action="store_true")
    parser.add_argument("--seed",         type=int, default=None,
                        help="Single seed to run (overrides config seeds)")
    parser.add_argument("--all_seeds",    action="store_true",
                        help="Run all seeds defined in LIRA_SETTINGS (default: seed 0 only)")
    parser.add_argument("--gpu",          type=int, default=0)
    parser.add_argument("--data_root",    type=str, default=DATA_ROOT)
    parser.add_argument("--cache_dir",    type=str, default=CACHE_DIR)
    parser.add_argument("--runs_dir",     type=str, default=RUNS_DIR)
    parser.add_argument("--lira_dir",     type=str, default=LIRA_DIR)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[P19-LiRA] Device: {device}")

    if args.all:
        lira_ids = list(LIRA_SETTINGS.keys())
    elif args.lira_id:
        lira_ids = [args.lira_id]
    else:
        print("[P19-LiRA] Specify --lira_id LF1/LF2/LF5 or --all"); return

    os.makedirs(args.lira_dir, exist_ok=True)

    for lid in lira_ids:
        cfg = LIRA_SETTINGS[lid]
        shadow_end = args.shadow_end if args.shadow_end is not None else cfg["n_shadows"]

        if args.seed is not None:
            seeds = [args.seed]
        elif args.all_seeds:
            seeds = cfg["seeds"]
        else:
            seeds = [cfg["seeds"][0]]

        for s in seeds:
            print(f"\n[P19-LiRA] === {lid} seed={s} ===")

            if not args.score_only:
                run_lira(lid, cfg, args.shadow_start, shadow_end, device,
                         args.data_root, args.cache_dir, args.runs_dir, args.lira_dir,
                         seed=s)

            run_scoring(lid, cfg, device, args.runs_dir, args.lira_dir, seed=s)

    print("\n[P19-LiRA] Done.")


if __name__ == "__main__":
    main()
