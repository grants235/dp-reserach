#!/usr/bin/env python3
"""
Phase 21: Tier 1 — Tabular MLP with Real DP-SGD + LiRA
=========================================================

Role: faithfulness and regime workhorse.  Shows that ε^dir predicts LiRA
membership scores better than ε^norm on a real attack, with advantage scaling
with degeneracy (imbalance ratio) and shrinking as privacy tightens (ε-isotropization).

Setup:
  - Synthetic imbalanced tabular dataset (sklearn make_classification)
    mimicking credit/fraud classification.  n=1000 total, d_feat=20.
  - 1-hidden-layer MLP (width=16), d_param = 20×16+16 + 16×1+1 = 353.
  - DP-SGD training (per-sample gradient clipping via vmap or manual loop).
  - 32 pooled shadow models for LiRA (each trained on Bernoulli-0.5 subset).
  - Certificate: exact d² from per-sample gradients at final model; Σ from
    training gradients; LOO correction via Sherman-Morrison; Nyström tightness.
  - 2-D sweep: ε ∈ {1,2,4,8,16} × IR ∈ {1,5,10,20,50}.
    Report Δρ = Spearman(ε^dir, LiRA) − Spearman(ε^norm, LiRA) as heatmap.

Deliverables:
  Fig H1: 2D heatmap of Δρ(ε, IR) — regime map (main deliverable)
  Fig H2: ε-slice at fixed IR — tight-ε decay reproduced on real attack
  Fig H3: Nyström tightness on real gradients (validates Appendix C)
  Tab H: per-cell Spearman ρ^dir, ρ^norm, Δρ with CI across seeds

Usage:
  python experiments/exp_p21_tabular.py --fast      # ~30s, 4 cells, 8 shadows
  python experiments/exp_p21_tabular.py --all       # full grid (15–25 min)
  python experiments/exp_p21_tabular.py --load results/p21/grid_results.json
"""

import os, sys, json, math, argparse, time, warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR = "./results/p21"

# ---------------------------------------------------------------------------
# Shared ε_sgm / calibration (same as p20)
# ---------------------------------------------------------------------------

_GH_T, _GH_W = hermgauss(64)
_GH_T = _GH_T.astype(np.float64)
_GH_W = _GH_W.astype(np.float64)
_LOG_SQRT_PI = 0.5 * math.log(math.pi)
_SQRT2 = math.sqrt(2.0)

ALPHA_GRID = np.concatenate([
    np.arange(1.5, 10, 0.5),
    np.arange(10, 100, 2.0),
    np.arange(100, 1000, 20.0),
    np.arange(1000, 5001, 100.0),
]).astype(np.float64)


def eps_sgm_vec(alpha: float, q: float, mu_array: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu_array, dtype=np.float64).ravel()
    n = len(mu)
    result = np.zeros(n, dtype=np.float64)
    pos_mask = (mu > 1e-15)
    if not pos_mask.any():
        return result.reshape(getattr(mu_array, 'shape', (n,)))
    mu_nz = mu[pos_mask][:, None]
    t = _GH_T[None, :]
    w = _GH_W[None, :]
    v = mu_nz * (_SQRT2 * t) - 0.5 * mu_nz ** 2
    log_base = np.logaddexp(np.log(1.0 - q), np.log(q) + v)
    log_g = alpha * log_base
    log_contrib = np.log(w) - _LOG_SQRT_PI + log_g
    max_lc = log_contrib.max(axis=1, keepdims=True)
    log_E = max_lc[:, 0] + np.log(np.exp(log_contrib - max_lc).sum(axis=1))
    result[pos_mask] = log_E / (alpha - 1.0)
    return result.reshape(getattr(mu_array, 'shape', (n,)))


def eps_sgm_scalar(alpha: float, q: float, mu: float) -> float:
    return float(eps_sgm_vec(alpha, q, np.array([mu]))[0])


def eps_cert_from_rdp(rdp_per_alpha: np.ndarray, alpha_grid: np.ndarray,
                       delta: float = 1e-5) -> float:
    log_inv_delta = math.log(1.0 / delta)
    eps_vals = rdp_per_alpha + log_inv_delta / (alpha_grid - 1.0)
    return float(np.min(eps_vals))


def calibrate_sigma(eps_target: float, q: float, T: int, delta: float = 1e-5,
                    mu_lo: float = 0.01, mu_hi: float = 20.0,
                    n_iter: int = 60) -> float:
    for _ in range(n_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        rdp = np.array([T * eps_sgm_scalar(a, q, mu_mid) for a in ALPHA_GRID])
        eps = eps_cert_from_rdp(rdp, ALPHA_GRID, delta)
        if eps > eps_target:
            mu_hi = mu_mid
        else:
            mu_lo = mu_mid
    return 1.0 / (0.5 * (mu_lo + mu_hi))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_dataset(n_total: int, n_features: int, imbalance_ratio: float,
                 n_informative: int = 10, seed: int = 42):
    """Imbalanced binary tabular dataset (mimics credit/fraud classification)."""
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    minority_frac = 1.0 / (1.0 + imbalance_ratio)
    X, y = make_classification(
        n_samples=n_total,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, n_features - n_informative - 2),
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[1.0 - minority_frac, minority_frac],
        flip_y=0.01,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = y.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _make_model(d_in: int, width: int):
    import torch.nn as nn
    import torch.nn.functional as F

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(d_in, width)
            self.fc2 = nn.Linear(width, 1)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x))).squeeze(-1)

    return TinyMLP()


def _param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# DP-SGD training  (per-sample gradient clipping)
# ---------------------------------------------------------------------------

def _try_vmap():
    try:
        from torch.func import functional_call, grad, vmap
        return functional_call, grad, vmap
    except ImportError:
        return None


def per_sample_grads_vmap(model, X_batch, y_batch):
    """Vectorized per-sample gradients via torch.func.vmap (PyTorch >= 2.0)."""
    import torch
    import torch.nn.functional as F
    from torch.func import functional_call, grad, vmap

    params = {k: v.detach() for k, v in model.named_parameters()}

    def single_loss(params, x, y):
        out = functional_call(model, params, x.unsqueeze(0)).squeeze()
        return F.binary_cross_entropy_with_logits(out, y)

    per_sample = vmap(grad(single_loss), in_dims=(None, 0, 0))
    grad_dicts = per_sample(params, X_batch, y_batch)

    grads = torch.cat([g.reshape(len(X_batch), -1) for g in grad_dicts.values()], dim=1)
    return grads  # (B, d_param)


def per_sample_grads_loop(model, X_batch, y_batch, criterion):
    """Fallback: per-sample gradients via scalar backward loop."""
    import torch
    grads = []
    for x, y in zip(X_batch, y_batch):
        model.zero_grad()
        loss = criterion(model(x.unsqueeze(0)), y.unsqueeze(0))
        loss.backward()
        grads.append(torch.cat([p.grad.detach().flatten() for p in model.parameters()]))
    return torch.stack(grads)  # (B, d_param)


def train_dp_sgd(model, X_np, y_np, sigma: float, C: float,
                 n_steps: int, batch_size: int, lr: float, seed: int,
                 use_vmap: bool = True):
    """
    Train model with per-sample gradient clipping + Gaussian noise.
    Returns the trained model (modified in-place).
    """
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    n = len(X_np)
    X = torch.FloatTensor(X_np)
    y = torch.FloatTensor(y_np)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    d_param = _param_count(model)

    vmap_ok = use_vmap and _try_vmap() is not None

    for _ in range(n_steps):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        X_b = X[idx]
        y_b = y[idx]

        if vmap_ok:
            try:
                G = per_sample_grads_vmap(model, X_b, y_b)
            except Exception:
                G = per_sample_grads_loop(model, X_b, y_b, criterion)
        else:
            G = per_sample_grads_loop(model, X_b, y_b, criterion)

        # Per-sample clip to norm C
        import torch
        norms = G.norm(dim=1, keepdim=True).clamp(min=1e-8)
        clip_factor = (C / norms).clamp(max=1.0)
        G_clipped = G * clip_factor

        # Aggregate + Gaussian noise
        noisy_sum = G_clipped.sum(0) + sigma * C * torch.randn(d_param)

        # Write back as gradient and step
        optimizer.zero_grad()
        with torch.no_grad():
            idx_p = 0
            for p in model.parameters():
                n_p = p.numel()
                p.grad = (noisy_sum[idx_p:idx_p + n_p].reshape(p.shape) / batch_size)
                idx_p += n_p
        optimizer.step()

    return model


# ---------------------------------------------------------------------------
# Per-sample gradient extraction (for certificate)
# ---------------------------------------------------------------------------

def compute_all_per_sample_grads(model, X_np, y_np, C: float = 1.0,
                                  batch: int = 128,
                                  use_vmap: bool = True) -> np.ndarray:
    """Compute per-sample gradients at current model for ALL n examples."""
    import torch
    import torch.nn as nn

    n = len(X_np)
    X = torch.FloatTensor(X_np)
    y = torch.FloatTensor(y_np)
    criterion = nn.BCEWithLogitsLoss()
    d_param = _param_count(model)
    vmap_ok = use_vmap and _try_vmap() is not None

    grads = np.empty((n, d_param), dtype=np.float32)
    i = 0
    while i < n:
        j = min(i + batch, n)
        X_b, y_b = X[i:j], y[i:j]
        if vmap_ok:
            try:
                G = per_sample_grads_vmap(model, X_b, y_b)
            except Exception:
                G = per_sample_grads_loop(model, X_b, y_b, criterion)
        else:
            G = per_sample_grads_loop(model, X_b, y_b, criterion)
        grads[i:j] = G.detach().numpy()
        i = j

    # Clip to C (match DP-SGD clip)
    norms = np.linalg.norm(grads, axis=1, keepdims=True)
    grads = grads * np.minimum(1.0, C / np.maximum(norms, 1e-8))
    return grads


# ---------------------------------------------------------------------------
# Privacy certificate
# ---------------------------------------------------------------------------

def compute_certificate(g: np.ndarray, Sigma_full: np.ndarray,
                         sigma: float, C: float, q: float, T: int,
                         delta: float, rho: float) -> dict:
    """
    Compute ε^dir and ε^norm for one canary gradient g.

    Uses the full Sigma (which includes the canary's contribution) and
    applies the Sherman-Morrison LOO correction: d²_loo = d²_full / (1 - ρ d²_full).

    ε^norm uses only ‖g‖ (direction-blind).
    ε^dir uses d²_loo (direction-aware Mahalanobis shift).
    """
    a = sigma ** 2 * C ** 2
    Sigma_tot = a * np.eye(len(g)) + Sigma_full

    # Exact d² on full Sigma_tot
    try:
        v = np.linalg.solve(Sigma_tot, g)
        d2_full = float(np.dot(g, v))
    except np.linalg.LinAlgError:
        d2_full = float(np.dot(g, g)) / a

    # LOO correction: d²_loo = d²_full / (1 - ρ d²_full)
    denom = 1.0 - rho * d2_full
    if denom > 1e-6:
        d2_loo = d2_full / denom
    else:
        d2_loo = d2_full / 1e-6

    # ε^dir from d²_loo
    mu_dir = math.sqrt(max(d2_loo, 0.0))
    rdp_dir = np.array([T * eps_sgm_scalar(al, q, mu_dir) for al in ALPHA_GRID])
    eps_dir = eps_cert_from_rdp(rdp_dir, ALPHA_GRID, delta)

    # ε^norm from ‖g‖ (direction-blind)
    g_norm = float(np.linalg.norm(g))
    mu_norm = g_norm / (sigma * C)
    rdp_norm = np.array([T * eps_sgm_scalar(al, q, mu_norm) for al in ALPHA_GRID])
    eps_norm = eps_cert_from_rdp(rdp_norm, ALPHA_GRID, delta)

    return {
        "d2_full": d2_full,
        "d2_loo": d2_loo,
        "eps_dir": eps_dir,
        "eps_norm": eps_norm,
        "g_norm": g_norm,
    }


def compute_nystrom_tightness(g: np.ndarray, Sigma_full: np.ndarray,
                               sigma: float, C: float,
                               ranks: List[int]) -> dict:
    """
    Nyström tightness on real gradients: Û² vs d²_exact as rank grows.
    No LOO correction (checking the bound validity on full Sigma).
    """
    a = sigma ** 2 * C ** 2
    Sigma_tot = a * np.eye(len(g)) + Sigma_full

    try:
        v = np.linalg.solve(Sigma_tot, g)
        d2_exact = float(np.dot(g, v))
    except np.linalg.LinAlgError:
        d2_exact = float(np.dot(g, g)) / a

    lambdas_emp, eigvecs_emp = np.linalg.eigh(Sigma_full)
    lambdas_emp = lambdas_emp[::-1].copy()
    eigvecs_emp = eigvecs_emp[:, ::-1].copy()

    results = {"d2_exact": d2_exact, "ranks": {}}
    norm2_a = float(np.dot(g, g)) / a

    for r in ranks:
        r_eff = min(r, len(lambdas_emp))
        U_r = eigvecs_emp[:, :r_eff]
        lam_r = lambdas_emp[:r_eff]
        proj = U_r.T @ g
        discount = (lam_r / (a + lam_r)) * proj ** 2
        U_full = float(np.clip(norm2_a - np.sum(discount) / a, 0.0, norm2_a))
        rel_gap = (U_full - d2_exact) / max(d2_exact, 1e-15)
        results["ranks"][r] = {"U_full": U_full, "rel_gap": rel_gap}

    return results


# ---------------------------------------------------------------------------
# LiRA: pooled shadow model attack
# ---------------------------------------------------------------------------

def train_shadow_pool(X_train: np.ndarray, y_train: np.ndarray,
                       sigma: float, C: float, q: float, T: int,
                       n_shadows: int, n_steps: int, batch_size: int,
                       lr: float, d_in: int, width: int, seed: int,
                       verbose: bool = False):
    """
    Train N shadow models, each on a random Bernoulli(0.5) subset.

    Returns:
      shadow_models : list of trained models
      memberships   : (n_shadows, n_train) bool — which examples were included
    """
    n = len(X_train)
    rng = np.random.default_rng(seed + 7777)
    memberships = rng.random((n_shadows, n)) < 0.5   # (S, n)

    shadow_models = []
    for s in range(n_shadows):
        mask = memberships[s]
        X_s, y_s = X_train[mask], y_train[mask]
        if len(X_s) < 2:
            mask = np.ones(n, dtype=bool)
            X_s, y_s = X_train, y_train

        model_s = _make_model(d_in, width)
        train_dp_sgd(model_s, X_s, y_s, sigma=sigma, C=C,
                     n_steps=n_steps, batch_size=min(batch_size, len(X_s)),
                     lr=lr, seed=seed + s)
        shadow_models.append(model_s)
        if verbose:
            print(f"    shadow {s+1}/{n_shadows} done (n_in={mask.sum()})")

    return shadow_models, memberships


def compute_lira_scores(target_model,
                         shadow_models, memberships: np.ndarray,
                         X_canaries: np.ndarray, y_canaries: np.ndarray,
                         canary_idx_in_train: np.ndarray) -> np.ndarray:
    """
    Offline LiRA: log-likelihood ratio membership score for each canary.

    Uses the shadow model pool where membership[s, canary] records whether
    canary was in shadow model s's training set.

    Returns: (K_canary,) LiRA scores
    """
    import torch
    import torch.nn as nn

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    K = len(X_canaries)
    S = len(shadow_models)

    X_t = torch.FloatTensor(X_canaries)
    y_t = torch.FloatTensor(y_canaries)

    # Shadow model losses on each canary: (S, K)
    shadow_losses = np.empty((S, K), dtype=np.float32)
    with torch.no_grad():
        for s, sm in enumerate(shadow_models):
            sm.eval()
            shadow_losses[s] = criterion(sm(X_t), y_t).numpy()

    # Target model losses on each canary: (K,)
    with torch.no_grad():
        target_model.eval()
        target_losses = criterion(target_model(X_t), y_t).numpy()

    # LiRA score per canary
    scores = np.zeros(K)
    for c in range(K):
        train_idx = canary_idx_in_train[c]
        in_mask = memberships[:, train_idx]

        l_in = shadow_losses[in_mask, c]
        l_out = shadow_losses[~in_mask, c]

        if len(l_in) < 2 or len(l_out) < 2:
            scores[c] = 0.0
            continue

        mu_in, std_in = l_in.mean(), l_in.std() + 1e-6
        mu_out, std_out = l_out.mean(), l_out.std() + 1e-6

        l_t = target_losses[c]
        log_pr_in = -0.5 * ((l_t - mu_in) / std_in) ** 2 - math.log(std_in)
        log_pr_out = -0.5 * ((l_t - mu_out) / std_out) ** 2 - math.log(std_out)
        scores[c] = float(log_pr_in - log_pr_out)

    return scores


# ---------------------------------------------------------------------------
# Spearman with constant-input guard
# ---------------------------------------------------------------------------

def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    if x[mask].std() < 1e-12:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(x[mask], y[mask]).statistic)


# ---------------------------------------------------------------------------
# Run one grid cell
# ---------------------------------------------------------------------------

@dataclass
class CellSpec:
    eps_target: float = 8.0
    imbalance_ratio: float = 10.0
    n_total: int = 1000
    n_features: int = 20
    n_informative: int = 10
    width: int = 16            # MLP hidden width
    C: float = 1.0
    delta: float = 1e-5
    T: int = 50                # training steps
    q: float = 0.1             # approximate sampling rate
    batch_size: int = 32
    lr: float = 0.05
    n_shadows: int = 32        # pooled shadow models
    shadow_steps: int = 50     # steps per shadow model
    n_canaries: int = 20       # canaries to attack (from minority class)
    nystrom_ranks: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    seed: int = 42


def run_one_cell(spec: CellSpec, verbose: bool = True) -> dict:
    t0 = time.time()

    # Calibrate σ
    sigma = calibrate_sigma(spec.eps_target, spec.q, spec.T, spec.delta)
    a = sigma ** 2 * spec.C ** 2
    rho = spec.q * (1.0 - spec.q)

    # Dataset
    X_train, y_train, X_test, y_test = make_dataset(
        spec.n_total, spec.n_features, spec.imbalance_ratio,
        n_informative=spec.n_informative, seed=spec.seed
    )
    n_train = len(X_train)
    d_in = spec.n_features

    # Pick canaries: minority-class training examples (most exposed under imbalance)
    minority_idx = np.where(y_train == 1)[0]
    majority_idx = np.where(y_train == 0)[0]
    rng = np.random.default_rng(spec.seed)

    n_min = min(spec.n_canaries // 2, len(minority_idx))
    n_maj = min(spec.n_canaries - n_min, len(majority_idx))
    canary_min = rng.choice(minority_idx, size=n_min, replace=False)
    canary_maj = rng.choice(majority_idx, size=n_maj, replace=False)
    canary_idx = np.concatenate([canary_min, canary_maj])   # indices into X_train
    canary_labels = y_train[canary_idx]                     # minority=1, majority=0

    X_can = X_train[canary_idx]
    y_can = canary_labels
    K = len(canary_idx)

    if verbose:
        print(f"  ε={spec.eps_target}, IR={spec.imbalance_ratio:.0f}, "
              f"σ={sigma:.3f}, n_train={n_train}, "
              f"minority={len(minority_idx)}, canaries={K}")

    # Train TARGET model
    target_model = _make_model(d_in, spec.width)
    train_dp_sgd(target_model, X_train, y_train,
                 sigma=sigma, C=spec.C,
                 n_steps=spec.T, batch_size=spec.batch_size,
                 lr=spec.lr, seed=spec.seed)

    # Compute per-sample gradients at target model (for certificate)
    all_grads = compute_all_per_sample_grads(target_model, X_train, y_train, C=spec.C)
    d_param = all_grads.shape[1]

    # Sigma from all training gradients (full, canaries included)
    Sigma_full = rho * all_grads.T @ all_grads   # (d_param, d_param)

    # Certificate for each canary
    certs = []
    for i, tidx in enumerate(canary_idx):
        g = all_grads[tidx].astype(np.float64)
        cert = compute_certificate(g, Sigma_full, sigma, spec.C,
                                    spec.q, spec.T, spec.delta, rho)
        cert["canary_class"] = int(y_train[tidx])
        certs.append(cert)

    eps_dir = np.array([c["eps_dir"] for c in certs])
    eps_norm = np.array([c["eps_norm"] for c in certs])

    # Nyström tightness on the median-d² canary
    med_idx = np.argsort([c["d2_loo"] for c in certs])[K // 2]
    g_med = all_grads[canary_idx[med_idx]].astype(np.float64)
    nystrom = compute_nystrom_tightness(g_med, Sigma_full, sigma, spec.C,
                                         spec.nystrom_ranks)

    # LiRA shadow models
    shadow_models, memberships = train_shadow_pool(
        X_train, y_train, sigma=sigma, C=spec.C,
        q=spec.q, T=spec.T, n_shadows=spec.n_shadows,
        n_steps=spec.shadow_steps, batch_size=spec.batch_size,
        lr=spec.lr, d_in=d_in, width=spec.width,
        seed=spec.seed + 1111, verbose=False,
    )

    lira_scores = compute_lira_scores(
        target_model, shadow_models, memberships,
        X_can, y_can, canary_idx,
    )

    # Spearman correlations
    rho_dir = spearman_rho(eps_dir, lira_scores)
    rho_norm = spearman_rho(eps_norm, lira_scores)
    delta_rho = rho_dir - rho_norm

    elapsed = time.time() - t0
    if verbose:
        print(f"    ρ_dir={rho_dir:.3f}  ρ_norm={rho_norm:.3f}  "
              f"Δρ={delta_rho:+.3f}  [{elapsed:.1f}s]")

    return {
        "eps_target": spec.eps_target,
        "imbalance_ratio": spec.imbalance_ratio,
        "sigma": sigma, "a": a, "rho_q": rho,
        "n_train": n_train, "n_minority": int(len(minority_idx)),
        "d_param": d_param, "K": K,
        "rho_dir": rho_dir, "rho_norm": rho_norm, "delta_rho": delta_rho,
        "eps_dir": eps_dir.tolist(),
        "eps_norm": eps_norm.tolist(),
        "lira_scores": lira_scores.tolist(),
        "canary_labels": y_can.tolist(),
        "certificates": certs,
        "nystrom": nystrom,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

DEFAULT_EPS = [1.0, 2.0, 4.0, 8.0, 16.0]
DEFAULT_IR = [1.0, 5.0, 10.0, 20.0, 50.0]
DEFAULT_SEEDS = [42, 123, 999]


def run_grid(eps_list=None, ir_list=None, seeds=None,
             n_total=1000, n_features=20, width=16,
             n_shadows=32, shadow_steps=50, n_steps=50,
             n_canaries=20, out_dir=OUT_DIR, verbose=True) -> list:
    if eps_list is None: eps_list = DEFAULT_EPS
    if ir_list is None: ir_list = DEFAULT_IR
    if seeds is None: seeds = DEFAULT_SEEDS

    os.makedirs(out_dir, exist_ok=True)
    all_results = []
    total = len(eps_list) * len(ir_list) * len(seeds)
    done = 0

    for eps in eps_list:
        for ir in ir_list:
            cell_results = []
            for seed in seeds:
                done += 1
                if verbose:
                    print(f"\n[{done}/{total}] ε={eps}  IR={ir:.0f}  seed={seed}")
                spec = CellSpec(
                    eps_target=eps, imbalance_ratio=ir,
                    n_total=n_total, n_features=n_features, width=width,
                    n_shadows=n_shadows, shadow_steps=shadow_steps,
                    T=n_steps, n_canaries=n_canaries, seed=seed,
                )
                res = run_one_cell(spec, verbose=verbose)
                cell_results.append(res)

            # Aggregate across seeds
            delta_rhos = [r["delta_rho"] for r in cell_results]
            rho_dirs = [r["rho_dir"] for r in cell_results]
            rho_norms = [r["rho_norm"] for r in cell_results]

            agg = {
                "eps_target": eps, "imbalance_ratio": ir,
                "delta_rho_mean": float(np.mean(delta_rhos)),
                "delta_rho_std": float(np.std(delta_rhos)),
                "rho_dir_mean": float(np.mean(rho_dirs)),
                "rho_norm_mean": float(np.mean(rho_norms)),
                "n_seeds": len(seeds),
                "per_seed": cell_results,
            }
            all_results.append(agg)

    path = os.path.join(out_dir, "grid_results.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved] {path}")
    return all_results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_heatmap(all_results: list, out_dir: str) -> None:
    """Fig H1: 2D heatmap of Δρ(ε, IR)."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [fig_heatmap] matplotlib not available"); return

    os.makedirs(out_dir, exist_ok=True)

    eps_vals = sorted(set(r["eps_target"] for r in all_results))
    ir_vals = sorted(set(r["imbalance_ratio"] for r in all_results))

    # Build 2D grids for Δρ, ρ_dir, ρ_norm
    def build_grid(key):
        grid = np.full((len(ir_vals), len(eps_vals)), np.nan)
        for r in all_results:
            ei = eps_vals.index(r["eps_target"])
            ii = ir_vals.index(r["imbalance_ratio"])
            grid[ii, ei] = r[key]
        return grid

    delta_grid = build_grid("delta_rho_mean")
    dir_grid = build_grid("rho_dir_mean")
    norm_grid = build_grid("rho_norm_mean")
    std_grid = build_grid("delta_rho_std")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def plot_heatmap(ax, data, title, cmap, vmin, vmax, std=None):
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                        origin="lower")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(eps_vals)))
        ax.set_xticklabels([f"{e:.0f}" for e in eps_vals])
        ax.set_yticks(range(len(ir_vals)))
        ax.set_yticklabels([f"{ir:.0f}" for ir in ir_vals])
        ax.set_xlabel("Privacy budget ε")
        ax.set_ylabel("Imbalance ratio IR")
        ax.set_title(title)
        for ii in range(len(ir_vals)):
            for ei in range(len(eps_vals)):
                val = data[ii, ei]
                if not np.isnan(val):
                    text = f"{val:.2f}"
                    if std is not None and not np.isnan(std[ii, ei]):
                        text += f"\n±{std[ii, ei]:.2f}"
                    ax.text(ei, ii, text, ha="center", va="center",
                            fontsize=7, color="white" if abs(val) > 0.3 else "black")

    plot_heatmap(axes[0], delta_grid,
                 "Δρ = ρ(ε^dir, LiRA) − ρ(ε^norm, LiRA)\n(direction advantage)",
                 "RdBu", -0.5, 0.5, std=std_grid)
    plot_heatmap(axes[1], dir_grid,
                 "ρ(ε^dir, LiRA)\n(direction-aware certificate)",
                 "viridis", -0.5, 0.8)
    plot_heatmap(axes[2], norm_grid,
                 "ρ(ε^norm, LiRA)\n(norm-only certificate)",
                 "viridis", -0.5, 0.8)

    fig.suptitle("Fig H1: Regime map — direction advantage Δρ across (ε, IR)\n"
                 "High IR + high ε = heaviest clipping + most anisotropy = largest gap",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "figH1_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig_heatmap] → {path}")


def fig_eps_slice(all_results: list, out_dir: str, fixed_ir: float = 10.0) -> None:
    """Fig H2: ε-slice — Δρ vs ε at fixed IR, with CI across seeds."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [fig_eps_slice] matplotlib not available"); return

    os.makedirs(out_dir, exist_ok=True)
    subset = sorted(
        [r for r in all_results if abs(r["imbalance_ratio"] - fixed_ir) < 1.0],
        key=lambda r: r["eps_target"]
    )
    if not subset:
        print(f"  [fig_eps_slice] no results for IR={fixed_ir}"); return

    eps_vals = [r["eps_target"] for r in subset]
    delta_means = [r["delta_rho_mean"] for r in subset]
    delta_stds = [r["delta_rho_std"] for r in subset]
    dir_means = [r["rho_dir_mean"] for r in subset]
    norm_means = [r["rho_norm_mean"] for r in subset]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.errorbar(eps_vals, delta_means, yerr=delta_stds, fmt="o-",
                  color="steelblue", lw=1.8, ms=7, capsize=4,
                  label="Δρ ± 1 std (across seeds)")
    ax1.axhline(0, color="gray", ls=":", lw=0.8)
    ax1.set_xlabel("Privacy budget ε  (larger = less noise)")
    ax1.set_ylabel("Δρ = ρ(ε^dir, LiRA) − ρ(ε^norm, LiRA)")
    ax1.set_title(f"ε-slice at IR={fixed_ir:.0f}\n"
                  "Direction advantage decays as ε tightens\n"
                  "(noise floor isotropizes Σ_tot — scope result)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(eps_vals, dir_means, "o-", color="steelblue", lw=1.5, ms=7,
              label="ρ(ε^dir, LiRA)")
    ax2.plot(eps_vals, norm_means, "s--", color="firebrick", lw=1.5, ms=7,
              label="ρ(ε^norm, LiRA)")
    ax2.set_xlabel("ε")
    ax2.set_ylabel("Spearman ρ with LiRA score")
    ax2.set_title("Both decrease at tight ε\n(real attack harder when noise dominates)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Fig H2: ε-slice (IR={fixed_ir:.0f}) — tight-ε regime on real LiRA attack\n"
                 "Confirms Fig D (geometric) with a real attack as ground truth",
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "figH2_eps_slice.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig_eps_slice] → {path}")


def fig_nystrom_tightness(all_results: list, out_dir: str) -> None:
    """Fig H3: Nyström tightness on real gradients."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [fig_nystrom_tightness] matplotlib not available"); return

    os.makedirs(out_dir, exist_ok=True)

    # Collect all per-seed nystrom results
    from collections import defaultdict
    by_cell = defaultdict(list)
    for r in all_results:
        key = (r["eps_target"], r["imbalance_ratio"])
        for seed_r in r["per_seed"]:
            nys = seed_r["nystrom"]
            d2_exact = nys["d2_exact"]
            for rank_str, rdata in nys["ranks"].items():
                by_cell[(key, int(rank_str))].append(rdata["rel_gap"])

    # Summary: median rel_gap vs rank, for a representative cell
    rep_eps, rep_ir = 8.0, 10.0
    ranks = sorted(set(k[1] for k in by_cell.keys()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for (ir_v) in sorted(set(k[0][1] for k in by_cell.keys())):
        rel_meds = []
        for rank in ranks:
            key = ((rep_eps, ir_v), rank)
            vals = by_cell.get(key, [])
            rel_meds.append(float(np.median(vals)) if vals else np.nan)
        ax1.plot(ranks, rel_meds, "o-", ms=5, lw=1.3, label=f"IR={ir_v:.0f}")

    ax1.axhline(0, color="gray", ls=":", lw=0.8)
    ax1.set_xlabel("Nyström rank r")
    ax1.set_ylabel("Median relative gap (Û² − d²) / d²")
    ax1.set_title(f"Nyström tightness on real gradients (ε={rep_eps})\n"
                  "Bound converges toward exact as rank grows")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: tightness per ε at fixed IR
    for (eps_v) in sorted(set(k[0][0] for k in by_cell.keys())):
        rel_meds = []
        for rank in ranks:
            key = ((eps_v, rep_ir), rank)
            vals = by_cell.get(key, [])
            rel_meds.append(float(np.median(vals)) if vals else np.nan)
        ax2.plot(ranks, rel_meds, "o-", ms=5, lw=1.3, label=f"ε={eps_v:.0f}")

    ax2.axhline(0, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel("Nyström rank r")
    ax2.set_ylabel("Median relative gap")
    ax2.set_title(f"Tightness vs ε (IR={rep_ir:.0f})\n"
                  "Tighter ε → larger a → bound tightens faster")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Fig H3: Nyström certificate tightness on real tabular MLP gradients\n"
                 "Validates Appendix C bound on actual (not synthetic) gradient geometry",
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "figH3_nystrom_tightness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig_nystrom_tightness] → {path}")


def print_table(all_results: list) -> None:
    print(f"\n{'='*80}")
    print("  Tab H: Tier 1 Tabular — Spearman correlations with LiRA")
    print(f"{'='*80}")
    print(f"  {'ε':4s}  {'IR':5s}  {'ρ_dir':7s}  {'ρ_norm':7s}  {'Δρ':7s}  {'±std':6s}  {'seeds':5s}")
    for r in sorted(all_results, key=lambda x: (x["eps_target"], x["imbalance_ratio"])):
        print(f"  {r['eps_target']:4.0f}  {r['imbalance_ratio']:5.0f}  "
              f"{r['rho_dir_mean']:7.3f}  {r['rho_norm_mean']:7.3f}  "
              f"{r['delta_rho_mean']:+7.3f}  {r['delta_rho_std']:6.3f}  "
              f"{r['n_seeds']:5d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Full grid")
    parser.add_argument("--fast", action="store_true",
                        help="Smoke-test: 2ε×2IR, 8 shadows, 1 seed, ~30s")
    parser.add_argument("--eps", nargs="+", type=float, default=DEFAULT_EPS)
    parser.add_argument("--ir", nargs="+", type=float, default=DEFAULT_IR)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--n_shadows", type=int, default=32)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--n_canaries", type=int, default=20)
    parser.add_argument("--n_total", type=int, default=1000)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--out", type=str, default=OUT_DIR)
    args = parser.parse_args()

    if args.fast:
        args.eps = [4.0, 8.0]
        args.ir = [1.0, 10.0]
        args.seeds = [42]
        args.n_shadows = 8
        args.n_steps = 20
        args.n_canaries = 10

    if args.load:
        with open(args.load) as f:
            all_results = json.load(f)
    else:
        all_results = run_grid(
            eps_list=args.eps, ir_list=args.ir, seeds=args.seeds,
            n_total=args.n_total, width=args.width,
            n_shadows=args.n_shadows, shadow_steps=args.n_steps,
            n_steps=args.n_steps, n_canaries=args.n_canaries,
            out_dir=args.out,
        )

    fig_heatmap(all_results, args.out)
    fig_eps_slice(all_results, args.out, fixed_ir=10.0)
    fig_nystrom_tightness(all_results, args.out)
    print_table(all_results)


if __name__ == "__main__":
    main()
