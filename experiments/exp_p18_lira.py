#!/usr/bin/env python3
"""
Phase 18 LiRA: Shadow Training and Membership Inference Scoring
===============================================================

Spec: phase18_spex.md, Section 6.

For each LiRA setting (L1/L2/L3), trains non-private shadow models and
computes per-example LiRA distinguishability scores.

LiRA plan (Section 6.2):
  L1: S2 CLIP CIFAR-10-LT(50) ε=8 — 1500 targets, 128 shadows
  L2: S1 CLIP CIFAR-10        ε=8 — 1000 targets,  32 shadows
  L3: S3 WRN  CIFAR-10-LT(50) ε=8 —  300 targets,   4 shadows

File structure:
  lira/{setting_id}/targets_members.npy
  lira/{setting_id}/targets_nonmembers.npy
  lira/{setting_id}/shadow_{m}_in_indices.npy
  lira/{setting_id}/shadow_{m}_nonmember_in_indices.npy
  lira/{setting_id}/shadow_{m}_member_logits.npy
  lira/{setting_id}/shadow_{m}_nonmember_logits.npy
  lira/{setting_id}/shadow_{m}_metadata.json
  lira/{setting_id}/dp_member_logits.npy
  lira/{setting_id}/dp_nonmember_logits.npy
  lira/{setting_id}/lira_scores_members.npy
  lira/{setting_id}/lira_scores_nonmembers.npy
  lira/{setting_id}/lira_summary.json

Usage:
  python experiments/exp_p18_lira.py --lira_id L1 --gpu 0
  python experiments/exp_p18_lira.py --lira_id L1 --shadow_start 0 --shadow_end 64 --gpu 0
  python experiments/exp_p18_lira.py --lira_id L1 --score_only --gpu 0
  python experiments/exp_p18_lira.py --all --gpu 0
"""

import os, sys, json, argparse, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets import make_public_private_split, make_cifar10_lt_indices
from src.models import WideResNet

import torchvision
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LIRA_SETTINGS = {
    "L1": dict(matched="S2", dataset="cifar10_lt50", regime="R3",
               n_targets=1500, n_shadows=128, shadow_epochs=20),
    "L2": dict(matched="S1", dataset="cifar10",      regime="R3",
               n_targets=1000, n_shadows=32,  shadow_epochs=20),
    "L3": dict(matched="S3", dataset="cifar10_lt50", regime="R2",
               n_targets=300,  n_shadows=4,   shadow_epochs=30),
}

DATA_ROOT  = "./data"
CACHE_DIR  = "./data/clip_features"
RUNS_DIR   = "./runs"
LIRA_DIR   = "./lira"
N_GROUPS   = 16
SHADOW_LR  = 0.05
SHADOW_BS  = 256
CHUNK_R3   = 512
CHUNK_WRN  = 64

LT_HEAD = {0, 1, 2}
LT_MID  = {3, 4, 5, 6}
LT_TAIL = {7, 8, 9}

def class_to_tier(c):
    if c in LT_HEAD: return 0
    if c in LT_MID:  return 1
    return 2


# ---------------------------------------------------------------------------
# Dataset helpers (reuse from training)
# ---------------------------------------------------------------------------

class _FeatureDataset(Dataset):
    def __init__(self, feats, labels, global_idx):
        self.feats = feats; self.labels = labels; self.idx = global_idx
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.feats[i], int(self.labels[i]), int(self.idx[i])


class _IndexedSubset(Dataset):
    def __init__(self, base, indices):
        self.base = base; self.indices = np.asarray(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]; return x, y, int(self.indices[i])


class _ConcatTupleDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.cum = np.cumsum([len(d) for d in datasets])
    def __len__(self): return int(self.cum[-1]) if len(self.cum) else 0
    def __getitem__(self, i):
        ds_i = int(np.searchsorted(self.cum, i, side="right"))
        prev = 0 if ds_i == 0 else int(self.cum[ds_i - 1])
        return self.datasets[ds_i][i - prev]


def _cifar10_ds(data_root, augment, train=True):
    m, s = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize(m, s)]) if augment else \
         T.Compose([T.ToTensor(), T.Normalize(m, s)])
    return torchvision.datasets.CIFAR10(root=data_root, train=train,
                                        download=True, transform=tf)


def _load_clip_features(data_root, cache_dir, device):
    paths = {k: os.path.join(cache_dir, f"cifar10_clip_{k}.pt")
             for k in ["train", "train_labels", "test", "test_labels"]}
    if all(os.path.exists(p) for p in paths.values()):
        return tuple(torch.load(paths[k], map_location="cpu", weights_only=False)
                     for k in ["train", "train_labels", "test", "test_labels"])
    raise FileNotFoundError("CLIP features not cached. Run exp_p18_train.py --setting S1 first.")


def load_private_indices(matched_setting, seed=0):
    """Load priv_idx from the run directory (same split for all seeds)."""
    run_dir = os.path.join(RUNS_DIR, matched_setting, f"seed_{seed}")
    idx_path = os.path.join(run_dir, "example_indices.npy")
    lbl_path = os.path.join(run_dir, "labels.npy")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Run not found: {run_dir}. Train first.")
    priv_idx    = np.load(idx_path)
    priv_labels = np.load(lbl_path)
    return priv_idx, priv_labels


def load_lira_targets(matched_setting, seed=0):
    """Load pre-selected LiRA targets from run directory."""
    run_dir = os.path.join(RUNS_DIR, matched_setting, f"seed_{seed}")
    ml  = np.load(os.path.join(run_dir, "lira_member_local_idx.npy"))
    nml = np.load(os.path.join(run_dir, "lira_nonmember_test_idx.npy"))
    return ml, nml


def build_dataset_for_lira(dataset_name, data_root, cache_dir, device):
    """Return (priv_idx, priv_feats, priv_labels, test_feats, test_labels, n_classes)."""
    is_lt = "lt" in dataset_name
    lt_ir = 50 if "lt50" in dataset_name else (100 if "lt100" in dataset_name else 1)

    tf_f, tf_l, te_f, te_l = _load_clip_features(data_root, cache_dir, device)
    full_targets = tf_l.numpy()
    lt_idx = make_cifar10_lt_indices(full_targets, lt_ir, seed=42) if is_lt else np.arange(len(full_targets))
    lt_targets = full_targets[lt_idx]
    _, priv_idx = make_public_private_split(lt_idx, lt_targets, public_frac=0.1, seed=42)

    priv_feats  = tf_f[priv_idx]
    priv_labels = tf_l[priv_idx]
    return priv_idx, priv_feats, priv_labels, te_f, te_l, 10


def build_dataset_wrn_for_lira(dataset_name, data_root):
    """Return (priv_idx, priv_labels, train_ds_aug, test_ds) for WRN."""
    is_lt = "lt" in dataset_name
    lt_ir = 50 if "lt50" in dataset_name else (100 if "lt100" in dataset_name else 1)
    train_aug = _cifar10_ds(data_root, augment=True, train=True)
    test_ds   = _cifar10_ds(data_root, augment=False, train=False)
    full_targets = np.array(train_aug.targets)
    lt_idx = make_cifar10_lt_indices(full_targets, lt_ir, seed=42) if is_lt else np.arange(len(train_aug))
    lt_targets = full_targets[lt_idx]
    _, priv_idx = make_public_private_split(lt_idx, lt_targets, public_frac=0.1, seed=42)
    return priv_idx, full_targets[priv_idx], train_aug, test_ds


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, num_classes=10, feat_dim=512):
        super().__init__(); self.fc = nn.Linear(feat_dim, num_classes)
    def forward(self, x): return self.fc(x.float())


def make_shadow_model(regime, num_classes):
    if regime == "R3":
        return LinearHead(num_classes, feat_dim=512)
    return WideResNet(depth=28, widen_factor=2, num_classes=num_classes, n_groups=N_GROUPS)


# ---------------------------------------------------------------------------
# Shadow training (non-private)
# ---------------------------------------------------------------------------

def train_shadow_clip(model, shadow_feats, shadow_labels, n_epochs, device, seed):
    """Train non-private CLIP linear head on shadow subset."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=SHADOW_LR, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    N = shadow_feats.shape[0]
    for ep in range(n_epochs):
        perm = torch.randperm(N)
        for i in range(0, N, SHADOW_BS):
            idx = perm[i:i+SHADOW_BS]; opt.zero_grad()
            h = shadow_feats[idx].to(device).float()
            y = shadow_labels[idx].to(device).long()
            F.cross_entropy(model(h), y).backward(); opt.step()
        sch.step()


def train_shadow_wrn(model, shadow_ds, n_epochs, device, seed):
    """Train non-private WRN on shadow subset."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model.train()
    loader = DataLoader(shadow_ds, batch_size=SHADOW_BS, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    for ep in range(n_epochs):
        for x, y, _ in loader:
            opt.zero_grad()
            F.cross_entropy(model(x.to(device)), y.to(device)).backward(); opt.step()
        sch.step()


# ---------------------------------------------------------------------------
# Logit evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_logits_clip(model, feats, indices, device, chunk=CHUNK_R3):
    """Evaluate model on CLIP feature examples given by local indices."""
    model.eval()
    parts = []
    for i in range(0, len(indices), chunk):
        h = feats[indices[i:i+chunk]].to(device).float()
        parts.append(model(h).cpu())
    return torch.cat(parts).numpy().astype(np.float32)


@torch.no_grad()
def eval_logits_wrn(model, ds, indices, device, chunk=CHUNK_WRN):
    """Evaluate WRN on dataset examples given by local indices into ds."""
    model.eval(); parts = []
    for i in range(0, len(indices), chunk):
        idx = indices[i:i+chunk]
        x = torch.stack([ds[int(j)][0] for j in idx]).to(device)
        parts.append(model(x).cpu())
    return torch.cat(parts).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# LiRA scoring (Section 6.5)
# ---------------------------------------------------------------------------

def lira_feature(logits, labels):
    """
    Compute scalar LiRA feature for each example.
    ℓ_i = log(p_true / (1 - p_true)) = log-odds of true class probability.
    """
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    n = len(labels)
    p_true = probs[np.arange(n), labels]
    p_true = np.clip(p_true, 1e-7, 1.0 - 1e-7)
    return np.log(p_true / (1.0 - p_true)).astype(np.float64)


def _shadow_id_list(shadow_ids):
    if isinstance(shadow_ids, int):
        return list(range(shadow_ids))
    return list(shadow_ids)


def compute_lira_scores(shadow_dir, member_idx, nonmember_idx,
                        member_labels, nonmember_labels,
                        shadow_ids, is_member_arr=None):
    """
    Compute LiRA distinguishability scores D_i^LiRA and LLR scores.

    For each target example i:
      ℓ_{i,m} = lira_feature from shadow m
      In-shadows: m where i ∈ S_m
      Out-shadows: m where i ∉ S_m

    D_i^LiRA = (μ_in - μ_out) / sqrt((σ²_in + σ²_out)/2)
    """
    all_labels  = np.concatenate([member_labels, nonmember_labels])
    N = len(all_labels)

    # Collect ℓ_{i,m} for each shadow
    in_scores  = [[] for _ in range(N)]   # scores when i ∈ S_m
    out_scores = [[] for _ in range(N)]   # scores when i ∉ S_m

    for m in _shadow_id_list(shadow_ids):
        # Load membership mask for this shadow
        in_path = os.path.join(shadow_dir, f"shadow_{m}_in_indices.npy")
        nmem_in_path = os.path.join(shadow_dir, f"shadow_{m}_nonmember_in_indices.npy")
        ml_path = os.path.join(shadow_dir, f"shadow_{m}_member_logits.npy")
        nl_path = os.path.join(shadow_dir, f"shadow_{m}_nonmember_logits.npy")
        if not all(os.path.exists(p) for p in [in_path, ml_path, nl_path]):
            continue

        in_indices = set(np.load(in_path).tolist())   # local indices in shadow
        nmem_in_indices = (set(np.load(nmem_in_path).tolist())
                           if os.path.exists(nmem_in_path) else set())
        mem_logits = np.load(ml_path)                  # [N_members, C]
        nmem_logits = np.load(nl_path)                 # [N_nonmembers, C]

        # Combine into [N, C] aligned as [members, nonmembers]
        mem_labels_arr  = member_labels
        nmem_labels_arr = nonmember_labels

        mem_feat  = lira_feature(mem_logits, mem_labels_arr)    # [N_members]
        nmem_feat = lira_feature(nmem_logits, nmem_labels_arr)  # [N_nonmembers]
        all_feat  = np.concatenate([mem_feat, nmem_feat])        # [N]

        # Distribute into in/out based on target-specific shadow membership.
        # Member indices are local private-set positions; nonmember indices are
        # test-set positions, so they must not share one namespace.
        n_mem = len(member_idx)
        for i, loc_idx in enumerate(member_idx):
            if int(loc_idx) in in_indices:
                in_scores[i].append(all_feat[i])
            else:
                out_scores[i].append(all_feat[i])
        for j, test_idx in enumerate(nonmember_idx):
            out_i = n_mem + j
            if int(test_idx) in nmem_in_indices:
                in_scores[out_i].append(all_feat[out_i])
            else:
                out_scores[out_i].append(all_feat[out_i])

    # Fit Gaussians (Section 6.5) with variance flooring
    EPS_VAR = 1e-6 ** 2
    mu_in   = np.zeros(N); sig_in  = np.ones(N) * 1e-6
    mu_out  = np.zeros(N); sig_out = np.ones(N) * 1e-6

    for i in range(N):
        if len(in_scores[i]) > 0:
            mu_in[i]  = np.mean(in_scores[i])
            sig_in[i] = max(np.std(in_scores[i]), 1e-6)
        if len(out_scores[i]) > 0:
            mu_out[i]  = np.mean(out_scores[i])
            sig_out[i] = max(np.std(out_scores[i]), 1e-6)

    # D_i^LiRA = (μ_in - μ_out) / sqrt((σ²_in + σ²_out)/2)
    sigma_pool = np.sqrt((sig_in ** 2 + sig_out ** 2) / 2.0)
    D_lira = (mu_in - mu_out) / np.maximum(sigma_pool, 1e-8)

    return D_lira[:len(member_idx)], D_lira[len(member_idx):]


def compute_llr_dp(dp_member_logits, dp_nonmember_logits,
                   member_labels, nonmember_labels,
                   shadow_dir, member_local_idx, nonmember_test_idx, shadow_ids):
    """
    Compute LLR_i^DP = log p_in(ℓ_i^DP) - log p_out(ℓ_i^DP) for DP model.
    Uses the fitted Gaussian distributions from shadow models.
    """
    all_labels  = np.concatenate([member_labels, nonmember_labels])
    N = len(all_labels)

    in_scores  = [[] for _ in range(N)]
    out_scores = [[] for _ in range(N)]

    for m in _shadow_id_list(shadow_ids):
        in_path = os.path.join(shadow_dir, f"shadow_{m}_in_indices.npy")
        nmem_in_path = os.path.join(shadow_dir, f"shadow_{m}_nonmember_in_indices.npy")
        ml_path = os.path.join(shadow_dir, f"shadow_{m}_member_logits.npy")
        nl_path = os.path.join(shadow_dir, f"shadow_{m}_nonmember_logits.npy")
        if not all(os.path.exists(p) for p in [in_path, ml_path, nl_path]):
            continue
        in_set = set(np.load(in_path).tolist())
        nmem_in_set = (set(np.load(nmem_in_path).tolist())
                       if os.path.exists(nmem_in_path) else set())
        mem_feat  = lira_feature(np.load(ml_path), member_labels)
        nmem_feat = lira_feature(np.load(nl_path), nonmember_labels)
        all_feat  = np.concatenate([mem_feat, nmem_feat])
        n_mem = len(member_local_idx)
        for i, loc_idx in enumerate(member_local_idx):
            if int(loc_idx) in in_set:
                in_scores[i].append(all_feat[i])
            else:
                out_scores[i].append(all_feat[i])
        for j, test_idx in enumerate(nonmember_test_idx):
            out_i = n_mem + j
            if int(test_idx) in nmem_in_set:
                in_scores[out_i].append(all_feat[out_i])
            else:
                out_scores[out_i].append(all_feat[out_i])

    # DP model features
    dp_all_logits = np.concatenate([dp_member_logits, dp_nonmember_logits], axis=0)
    dp_feat = lira_feature(dp_all_logits, all_labels)  # [N]

    EPS = 1e-6
    llr = np.zeros(N)
    for i in range(N):
        mu_in  = np.mean(in_scores[i])  if len(in_scores[i]) > 0  else 0.0
        mu_out = np.mean(out_scores[i]) if len(out_scores[i]) > 0 else 0.0
        si_in  = max(np.std(in_scores[i]),  EPS) if len(in_scores[i]) > 1  else EPS
        si_out = max(np.std(out_scores[i]), EPS) if len(out_scores[i]) > 1 else EPS
        fi = dp_feat[i]
        log_pin  = -0.5 * ((fi - mu_in)  / si_in)  ** 2 - np.log(si_in)
        log_pout = -0.5 * ((fi - mu_out) / si_out) ** 2 - np.log(si_out)
        llr[i] = log_pin - log_pout

    return llr[:len(member_local_idx)], llr[len(member_local_idx):]


# ---------------------------------------------------------------------------
# AUC and correlation (Table C)
# ---------------------------------------------------------------------------

def compute_auc(scores_members, scores_nonmembers):
    from sklearn.metrics import roc_auc_score
    labels = np.concatenate([np.ones(len(scores_members)),
                             np.zeros(len(scores_nonmembers))])
    scores = np.concatenate([scores_members, scores_nonmembers])
    return float(roc_auc_score(labels, scores))


def spearman_rho(x, y):
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return float(r), float(p)


def rank_r2(x, y):
    """Rank-rank R² (Spearman ρ²)."""
    from scipy.stats import rankdata
    rx = rankdata(x); ry = rankdata(y)
    corr = np.corrcoef(rx, ry)[0, 1]
    return float(corr ** 2)


# ---------------------------------------------------------------------------
# Main LiRA run
# ---------------------------------------------------------------------------

def run_lira(lira_id, cfg, device, data_root, cache_dir, runs_dir, lira_dir,
             shadow_start=0, shadow_end=None, score_only=False):
    matched    = cfg["matched"]
    dataset    = cfg["dataset"]
    regime     = cfg["regime"]
    n_targets  = cfg["n_targets"]
    n_shadows  = cfg["n_shadows"]
    shad_ep    = cfg["shadow_epochs"]
    is_clip    = (regime == "R3")

    lira_setting_dir = os.path.join(lira_dir, lira_id)
    os.makedirs(lira_setting_dir, exist_ok=True)

    print(f"\n[LiRA] === {lira_id} (matched={matched}) ===")

    # Load targets from training run (use seed 0 for consistency)
    member_local_idx, nonmember_test_idx = load_lira_targets(matched, seed=0)
    np.save(os.path.join(lira_setting_dir, "targets_members.npy"),    member_local_idx)
    np.save(os.path.join(lira_setting_dir, "targets_nonmembers.npy"), nonmember_test_idx)
    print(f"  Targets: {len(member_local_idx)} members, {len(nonmember_test_idx)} nonmembers")

    # Load data
    if is_clip:
        priv_idx, priv_feats, priv_labels, te_feats, te_labels, n_classes = \
            build_dataset_for_lira(dataset, data_root, cache_dir, device)
        member_labels    = priv_labels[member_local_idx].numpy().astype(np.int32)
        nonmember_labels = te_labels[nonmember_test_idx].numpy().astype(np.int32)
    else:
        priv_idx, priv_labels_np, train_ds, test_ds = build_dataset_wrn_for_lira(dataset, data_root)
        n_classes = 10
        member_labels    = priv_labels_np[member_local_idx].astype(np.int32)
        test_labels_np   = np.array([test_ds[i][1] for i in range(len(test_ds))], dtype=np.int32)
        nonmember_labels = test_labels_np[nonmember_test_idx].astype(np.int32)

    n_priv = len(priv_idx)

    if not score_only:
        shadow_end = shadow_end or n_shadows
        print(f"  Training shadows {shadow_start}..{shadow_end-1} (total={n_shadows})")

        for m in range(shadow_start, shadow_end):
            shad_ckpt = os.path.join(lira_setting_dir, f"shadow_{m}_member_logits.npy")
            if os.path.exists(shad_ckpt):
                print(f"  shadow {m}: already done, skipping.")
                continue

            print(f"  shadow {m}/{n_shadows}...")
            rng_s = np.random.default_rng(m + 10000)

            # Sample 50% of private set (Section 6.4)
            in_mask  = rng_s.random(n_priv) < 0.5
            in_local = np.where(in_mask)[0]   # local indices in private set
            nmem_in_mask = rng_s.random(len(nonmember_test_idx)) < 0.5
            nmem_in_local = nonmember_test_idx[nmem_in_mask]

            np.save(os.path.join(lira_setting_dir, f"shadow_{m}_in_indices.npy"),
                    in_local.astype(np.int32))
            np.save(os.path.join(lira_setting_dir, f"shadow_{m}_nonmember_in_indices.npy"),
                    nmem_in_local.astype(np.int32))

            shadow_model = make_shadow_model(regime, n_classes).to(device)

            if is_clip:
                shadow_feats  = torch.cat([priv_feats[in_local],
                                           te_feats[nmem_in_local]], dim=0)
                shadow_labels = torch.cat([priv_labels[in_local],
                                           te_labels[nmem_in_local]], dim=0)
                train_shadow_clip(shadow_model, shadow_feats, shadow_labels,
                                  shad_ep, device, seed=m)
            else:
                shadow_ds = _ConcatTupleDataset(
                    _IndexedSubset(train_ds, priv_idx[in_local]),
                    _IndexedSubset(test_ds, nmem_in_local))
                train_shadow_wrn(shadow_model, shadow_ds, shad_ep, device, seed=m)

            # Evaluate on member targets (from private set, all members)
            if is_clip:
                mem_logits  = eval_logits_clip(shadow_model, priv_feats,
                                               member_local_idx, device)
                nmem_logits = eval_logits_clip(shadow_model, te_feats,
                                               nonmember_test_idx, device)
            else:
                mem_logits  = eval_logits_wrn(shadow_model,
                                              _IndexedSubset(train_ds, priv_idx),
                                              member_local_idx, device)
                # For nonmembers: evaluate on test set directly
                nmem_logits = eval_logits_wrn(shadow_model, test_ds,
                                              nonmember_test_idx, device)

            np.save(os.path.join(lira_setting_dir, f"shadow_{m}_member_logits.npy"),
                    mem_logits)
            np.save(os.path.join(lira_setting_dir, f"shadow_{m}_nonmember_logits.npy"),
                    nmem_logits)

            meta_s = {
                "lira_id": lira_id, "shadow_m": m,
                "n_member_in": int(len(in_local)),
                "n_member_out": int((~in_mask).sum()),
                "n_nonmember_in": int(len(nmem_in_local)),
                "n_nonmember_out": int((~nmem_in_mask).sum()),
                "shadow_epochs": shad_ep,
            }
            with open(os.path.join(lira_setting_dir, f"shadow_{m}_metadata.json"), "w") as f:
                json.dump(meta_s, f, indent=2)

            del shadow_model; torch.cuda.empty_cache()

    # --- Scoring phase -------------------------------------------------------
    # Check how many shadows are done
    done_shadows = [m for m in range(n_shadows)
                    if os.path.exists(os.path.join(lira_setting_dir,
                                                   f"shadow_{m}_member_logits.npy"))]
    print(f"\n  [LiRA score] {len(done_shadows)}/{n_shadows} shadows done")

    if len(done_shadows) < 2:
        print("  [LiRA] Not enough shadows to compute scores (need ≥ 2). Run more shadows first.")
        return

    # Load DP model logits (from training run, seed 0)
    run_dir_s0 = os.path.join(runs_dir, matched, "seed_0")
    dp_mem_path  = os.path.join(run_dir_s0, "target_logits_dp_members.npy")
    dp_nmem_path = os.path.join(run_dir_s0, "target_logits_dp_nonmembers.npy")

    if not os.path.exists(dp_mem_path):
        print(f"  [LiRA] DP model logits not found: {dp_mem_path}. Compute from model.")
        # Try to load model and compute
        model_path = os.path.join(run_dir_s0, "model_final.pt")
        if os.path.exists(model_path):
            dp_model = make_shadow_model(regime, n_classes).to(device)
            dp_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            if is_clip:
                dp_mem_logits  = eval_logits_clip(dp_model, priv_feats, member_local_idx, device)
                dp_nmem_logits = eval_logits_clip(dp_model, te_feats, nonmember_test_idx, device)
            else:
                dp_mem_logits  = eval_logits_wrn(dp_model,
                                                 _IndexedSubset(train_ds, priv_idx),
                                                 member_local_idx, device)
                dp_nmem_logits = eval_logits_wrn(dp_model, test_ds, nonmember_test_idx, device)
            np.save(dp_mem_path,  dp_mem_logits)
            np.save(dp_nmem_path, dp_nmem_logits)
            del dp_model; torch.cuda.empty_cache()
        else:
            print(f"  [LiRA] Model not found either: {model_path}"); return
    else:
        dp_mem_logits  = np.load(dp_mem_path)
        dp_nmem_logits = np.load(dp_nmem_path)

    np.save(os.path.join(lira_setting_dir, "dp_member_logits.npy"),    dp_mem_logits)
    np.save(os.path.join(lira_setting_dir, "dp_nonmember_logits.npy"), dp_nmem_logits)

    # Compute D_i^LiRA scores
    D_mem, D_nmem = compute_lira_scores(
        lira_setting_dir, member_local_idx, nonmember_test_idx,
        member_labels, nonmember_labels,
        shadow_ids=done_shadows)

    np.save(os.path.join(lira_setting_dir, "lira_scores_members.npy"),    D_mem.astype(np.float32))
    np.save(os.path.join(lira_setting_dir, "lira_scores_nonmembers.npy"), D_nmem.astype(np.float32))

    # Compute LLR scores for DP model
    llr_mem, llr_nmem = compute_llr_dp(
        dp_mem_logits, dp_nmem_logits,
        member_labels, nonmember_labels,
        lira_setting_dir, member_local_idx, nonmember_test_idx,
        shadow_ids=done_shadows)

    np.save(os.path.join(lira_setting_dir, "llr_dp_members.npy"),    llr_mem.astype(np.float32))
    np.save(os.path.join(lira_setting_dir, "llr_dp_nonmembers.npy"), llr_nmem.astype(np.float32))

    # AUC of LLR as attack score
    try:
        auc_llr = compute_auc(llr_mem, llr_nmem)
    except Exception:
        auc_llr = float("nan")

    # AUC of D_LiRA as member predictor
    try:
        auc_d = compute_auc(D_mem, D_nmem)
    except Exception:
        auc_d = float("nan")

    # Correlate ε^dir and ε^norm with D_LiRA (if cert exists)
    cert_dir = "./certs/p18"
    spearman_dir = float("nan"); spearman_norm = float("nan")
    r2_dir = float("nan"); r2_norm = float("nan")
    for seed in range(3):
        eps_dir_path  = os.path.join(cert_dir, f"p18_{matched}_seed{seed}_eps_dir.npy")
        eps_norm_path = os.path.join(cert_dir, f"p18_{matched}_seed{seed}_eps_norm.npy")
        if os.path.exists(eps_dir_path) and os.path.exists(eps_norm_path):
            eps_dir_all  = np.load(eps_dir_path)   # [n_priv]
            eps_norm_all = np.load(eps_norm_path)   # [n_priv]
            eps_dir_tgt  = eps_dir_all[member_local_idx]
            eps_norm_tgt = eps_norm_all[member_local_idx]
            try:
                spearman_dir,  _ = spearman_rho(eps_dir_tgt,  D_mem)
                spearman_norm, _ = spearman_rho(eps_norm_tgt, D_mem)
                r2_dir    = rank_r2(eps_dir_tgt,  D_mem)
                r2_norm   = rank_r2(eps_norm_tgt, D_mem)
            except Exception:
                pass
            break

    summary = {
        "lira_id": lira_id,
        "matched_setting": matched,
        "n_shadows_done": len(done_shadows),
        "n_targets_members": int(len(member_local_idx)),
        "n_targets_nonmembers": int(len(nonmember_test_idx)),
        "D_lira_members_mean":  float(D_mem.mean()),
        "D_lira_members_med":   float(np.median(D_mem)),
        "auc_llr_dp":           auc_llr,
        "auc_D_lira":           auc_d,
        "spearman_eps_dir_vs_D_lira":  spearman_dir,
        "spearman_eps_norm_vs_D_lira": spearman_norm,
        "rank_r2_eps_dir":  r2_dir,
        "rank_r2_eps_norm": r2_norm,
    }

    with open(os.path.join(lira_setting_dir, "lira_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  [LiRA {lira_id}] AUC(LLR)={auc_llr:.4f}  AUC(D_LiRA)={auc_d:.4f}")
    print(f"  Spearman ρ: ε^dir={spearman_dir:.4f}  ε^norm={spearman_norm:.4f}")
    print(f"  Rank-R²:    ε^dir={r2_dir:.4f}  ε^norm={r2_norm:.4f}")
    print(f"  Summary: {os.path.join(lira_setting_dir, 'lira_summary.json')}")


# ---------------------------------------------------------------------------
# Analysis: Table C and Figure (Section 7)
# ---------------------------------------------------------------------------

def print_table_c(lira_id, cert_dir="./certs/p18"):
    """Table C: headline LiRA correlation."""
    lira_dir_s = os.path.join(LIRA_DIR, lira_id)
    summ_path  = os.path.join(lira_dir_s, "lira_summary.json")
    if not os.path.exists(summ_path):
        print(f"[Table C] LiRA summary not found: {summ_path}"); return

    with open(summ_path) as f:
        s = json.load(f)

    matched = s["matched_setting"]
    D_mem    = np.load(os.path.join(lira_dir_s, "lira_scores_members.npy"))
    llr_mem  = np.load(os.path.join(lira_dir_s, "llr_dp_members.npy"))

    rows = []
    # Load cert results
    for seed in range(3):
        for name, fname in [("eps_dir", "eps_dir"), ("eps_norm", "eps_norm")]:
            p = os.path.join(cert_dir, f"p18_{matched}_seed{seed}_{fname}.npy")
            if not os.path.exists(p): continue
            arr = np.load(p)
            tgt = os.path.join(RUNS_DIR, matched, f"seed_{seed}", "lira_member_local_idx.npy")
            if not os.path.exists(tgt): continue
            local_idx = np.load(tgt)
            arr_tgt = arr[local_idx]
            try:
                rho_s, _ = spearman_rho(arr_tgt, D_mem[:len(local_idx)])
                r2   = rank_r2(arr_tgt, D_mem[:len(local_idx)])
            except Exception:
                rho_s = r2 = float("nan")
            rows.append({"metric": f"{fname} seed{seed}", "spearman": rho_s, "rank_r2": r2})
            break

    print(f"\n{'='*60}")
    print(f"  Table C: LiRA correlation — {lira_id} ({matched})")
    print(f"{'='*60}")
    print(f"  {'Metric':20s}  {'Spearman ρ':10s}  {'Rank R²':8s}")
    for row in rows:
        print(f"  {row['metric']:20s}  {row['spearman']:10.4f}  {row['rank_r2']:8.4f}")
    print(f"  AUC(LLR DP model): {s['auc_llr_dp']:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 18 LiRA")
    parser.add_argument("--lira_id",      type=str, default=None)
    parser.add_argument("--all",          action="store_true")
    parser.add_argument("--score_only",   action="store_true")
    parser.add_argument("--shadow_start", type=int, default=0)
    parser.add_argument("--shadow_end",   type=int, default=None)
    parser.add_argument("--table_c",      action="store_true")
    parser.add_argument("--gpu",          type=int, default=0)
    parser.add_argument("--data_root",    type=str, default=DATA_ROOT)
    parser.add_argument("--cache_dir",    type=str, default=CACHE_DIR)
    parser.add_argument("--runs_dir",     type=str, default=RUNS_DIR)
    parser.add_argument("--lira_dir",     type=str, default=LIRA_DIR)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[LiRA] Device: {device}")

    if args.table_c:
        for lid in (["L1"] if not args.lira_id else [args.lira_id]):
            print_table_c(lid)
        return

    if args.lira_id:
        lira_ids = [args.lira_id]
    elif args.all:
        lira_ids = list(LIRA_SETTINGS.keys())
    else:
        print("[LiRA] No --lira_id specified. Defaulting to L1 (headline).")
        lira_ids = ["L1"]

    for lid in lira_ids:
        if lid not in LIRA_SETTINGS:
            print(f"[LiRA] Unknown LiRA ID: {lid}"); continue
        run_lira(lid, LIRA_SETTINGS[lid], device,
                 args.data_root, args.cache_dir, args.runs_dir, args.lira_dir,
                 shadow_start=args.shadow_start,
                 shadow_end=args.shadow_end,
                 score_only=args.score_only)

    print("[LiRA] Done.")


if __name__ == "__main__":
    main()
