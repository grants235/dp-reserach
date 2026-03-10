"""
Evaluation utilities shared across experiments.

Includes:
- Feature extraction for density-based tiers
- Per-sample gradient norm computation
- LiRA / RMIA membership inference attack
- Result saving/loading helpers
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Any, Tuple


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader,
                     device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract penultimate-layer features.

    Returns:
        features: (n, D)
        labels: (n,)
    """
    model.eval()
    feats, labels = [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        f = model.features(x).cpu().numpy()
        feats.append(f)
        labels.append(y.cpu().numpy())
    return np.concatenate(feats), np.concatenate(labels)


# ---------------------------------------------------------------------------
# Per-sample gradient norm at a checkpoint
# ---------------------------------------------------------------------------

def compute_gradient_norms_checkpoint(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    C: float = None,
) -> Dict[str, np.ndarray]:
    """
    Compute per-sample gradient norms and (optionally) clipped norms.

    Returns dict with:
        'unclipped': array of shape (n,) – ||∇ℓ(θ, z_i)||
        'clipped': array of shape (n,)   – min(||∇ℓ(θ, z_i)||, C) if C given
    """
    from src.calibration import compute_per_sample_gradient_norms
    from src.calibration import compute_per_sample_clipped_norms

    norms = compute_per_sample_gradient_norms(model, loader, device)
    result = {"unclipped": norms}
    if C is not None:
        result["clipped"] = compute_per_sample_clipped_norms(norms, C)
    return result


# ---------------------------------------------------------------------------
# Membership Inference: LiRA (Carlini et al., 2022)
# ---------------------------------------------------------------------------

def lira_scores_online(
    losses_in: np.ndarray,
    losses_out: np.ndarray,
    target_losses: np.ndarray,
    use_augmentation: bool = True,
    aug_losses: np.ndarray = None,
) -> np.ndarray:
    """
    Compute LiRA attack scores (online version with shadow models).

    For each target sample i:
        score_i = log N(L_i; μ_in, σ_in) - log N(L_i; μ_out, σ_out)

    Args:
        losses_in: (n_targets, n_in_models) – losses of each target on in-models
        losses_out: (n_targets, n_out_models) – losses on out-models
        target_losses: (n_targets,) – loss of each target on the target model
        use_augmentation: if True, average scores over augmented versions
        aug_losses: (n_targets,) – loss under horizontal flip augmentation

    Returns:
        scores: (n_targets,) – higher = more likely member
    """
    n_targets = len(target_losses)
    scores = np.zeros(n_targets)

    for i in range(n_targets):
        lin = losses_in[i]
        lout = losses_out[i]
        mu_in = lin.mean()
        sigma_in = lin.std() + 1e-8
        mu_out = lout.mean()
        sigma_out = lout.std() + 1e-8

        def log_gaussian(x, mu, sigma):
            return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma)

        def score_single(loss_val):
            return log_gaussian(loss_val, mu_in, sigma_in) - \
                   log_gaussian(loss_val, mu_out, sigma_out)

        if use_augmentation and aug_losses is not None:
            scores[i] = 0.5 * (score_single(target_losses[i]) +
                                score_single(aug_losses[i]))
        else:
            scores[i] = score_single(target_losses[i])

    return scores


def rmia_scores(
    losses_in: np.ndarray,
    losses_out: np.ndarray,
    target_losses: np.ndarray,
    ref_losses_in: np.ndarray,
    ref_losses_out: np.ndarray,
    ref_losses: np.ndarray,
) -> np.ndarray:
    """
    RMIA (Rezaei et al., 2023) attack scores.

    score_i = LiRA(z_i) - LiRA(z_ref_i)
    where z_ref is a reference (non-member) point.

    Args:
        All arrays are (n_targets, n_models) or (n_targets,) as appropriate.

    Returns:
        scores: (n_targets,)
    """
    target_lira = lira_scores_online(losses_in, losses_out, target_losses,
                                     use_augmentation=False)
    ref_lira = lira_scores_online(ref_losses_in, ref_losses_out, ref_losses,
                                  use_augmentation=False)
    return target_lira - ref_lira


def compute_mia_metrics(
    scores: np.ndarray,
    is_member: np.ndarray,
    fpr_targets: List[float] = None,
) -> Dict[str, float]:
    """
    Compute MIA metrics: AUC and TPR at specified FPR thresholds.

    Args:
        scores: (n,) – attack scores (higher = predicted member)
        is_member: (n,) bool – True for members, False for non-members
        fpr_targets: list of FPR values to compute TPR at

    Returns:
        dict with 'auc', 'tpr_at_fpr_1e-3', 'tpr_at_fpr_1e-2', etc.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    if fpr_targets is None:
        fpr_targets = [1e-3, 1e-2]

    auc = roc_auc_score(is_member, scores)
    fpr, tpr, _ = roc_curve(is_member, scores)

    metrics = {"auc": auc}
    for target_fpr in fpr_targets:
        # Find TPR at the first FPR that exceeds the target
        idx = np.searchsorted(fpr, target_fpr, side="right")
        if idx > 0:
            tpr_val = tpr[idx - 1]
        else:
            tpr_val = 0.0
        key = f"tpr_at_fpr_{target_fpr:.0e}"
        metrics[key] = float(tpr_val)

    return metrics


def compute_roc(scores: np.ndarray, is_member: np.ndarray):
    """Return (fpr, tpr) arrays for ROC curve."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(is_member, scores)
    return fpr, tpr


# ---------------------------------------------------------------------------
# LT-IQR precision (Pollock et al. cross-reference)
# ---------------------------------------------------------------------------

def lt_iqr_precision_at_k(
    iqr_scores: np.ndarray,
    true_vulnerabilities: np.ndarray,
    k_frac: float = 0.01,
) -> float:
    """
    Precision@k of LT-IQR for identifying vulnerable/high-ε samples.

    Args:
        iqr_scores: (n,) – IQR scores (higher = more unstable)
        true_vulnerabilities: (n,) bool – True = actually high-ε sample
        k_frac: fraction of top samples to consider (default 1%)

    Returns:
        precision at k
    """
    n = len(iqr_scores)
    k = max(1, int(k_frac * n))
    top_k_idx = np.argsort(iqr_scores)[::-1][:k]
    return float(true_vulnerabilities[top_k_idx].mean())


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------

def save_results(results: Any, path: str):
    """Save results dict/array to a pickle file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)


def load_results(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_serialize)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
