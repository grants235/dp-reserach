#!/usr/bin/env python3
"""
Phase 20: Synthetic Canary Experiment
======================================

Tier-1 transcript-space validation. Works entirely in gradient/transcript
space with no trained model. Runs in minutes on CPU.

Mathematical setup (four-fix version):
  - d=256 gradient space
  - Background pool of n clipped gradients sampled from a power-law target
    spectrum λ_k ∝ (k+1)^{-β}, β = log(κ)/log(d); clipped to ‖·‖=C
  - Sigma_emp = ρ G^T G is computed from the ACTUAL bg_grads (eliminates
    the model/MC mismatch of the analytic-Sigma version)
  - K canaries with ‖ḡ_c‖=C exactly in the eigvec_emp basis
  - Three quantities per canary (on disjoint code paths):
      1. Exact d²_c = ḡ_c^T (aI + Σ_emp)^{-1} ḡ_c  (matrix inversion, d×d)
         Nyström Û²_c = g^T(aI+L_r)^{-1}g ≥ d²_c  (NO LOO correction:
           canaries are not in bg_grads so Σ_emp is already leave-one-out)
         Analytic C²/(a+λ_k) for eigendirection canaries — cross-check
      2. ε^dir = ε_sgm(α*, q, √d²_c)  |  ε^norm = ε_sgm(α*, q, C/(σC)) = const
      3. Gaussianity of the projected LOO aggregate (Assumption 4.2 check):
           project each Poisson batch onto the Mahalanobis direction
           w = Σ_tot^{-1}g_c / ‖…‖ and compare against fitted Gaussian
         (never touches Σ or d² — genuinely non-circular)

Deliverables:
  Fig A: Eigendirection ladder — exact/Nyström/analytic/norm shift vs eigdir k
  Fig B: Anisotropy-vs-gap — Δρ vs median masking discount, κ sweep
  Fig C: Assumption 4.2 — KS distance of projected aggregate vs Gaussian,
         colored by masking discount (should be small and uniform)
  Fig D: ε sweep — gap Δρ vs ε, at fixed κ
  Tab T: Nyström tightness — Û² vs d²_exact as rank r grows (converges to 0)

Usage:
  python experiments/exp_p20_synthetic.py --all
  python experiments/exp_p20_synthetic.py --fig ladder
  python experiments/exp_p20_synthetic.py --fig gap --kappas 1 5 20 100 500
  python experiments/exp_p20_synthetic.py --fig eps_sweep
  python experiments/exp_p20_synthetic.py --fig assumption
  python experiments/exp_p20_synthetic.py --table tightness
"""

import os, sys, json, math, argparse, time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.polynomial.hermite import hermgauss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR = "./results/p20"

# ---------------------------------------------------------------------------
# ε_sgm: exact Poisson-subsampled Gaussian RDP (copied from p19_certify)
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
    """Convert per-α RDP budget to (ε,δ)-DP via standard conversion."""
    log_inv_delta = math.log(1.0 / delta)
    eps_vals = rdp_per_alpha + log_inv_delta / (alpha_grid - 1.0)
    return float(np.min(eps_vals))


def calibrate_sigma(eps_target: float, q: float, T: int, delta: float = 1e-5,
                    mu_lo: float = 0.01, mu_hi: float = 20.0,
                    n_iter: int = 60) -> float:
    """Binary search σ so T-step RDP composition ≈ ε_target.

    ε is monotone increasing in μ=1/σ, so:
      ε > eps_target → μ too large → reduce upper bound: mu_hi = mu_mid
      ε ≤ eps_target → μ too small → raise lower bound: mu_lo = mu_mid
    """
    for _ in range(n_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        rdp = np.array([T * eps_sgm_scalar(a, q, mu_mid) for a in ALPHA_GRID])
        eps = eps_cert_from_rdp(rdp, ALPHA_GRID, delta)
        if eps > eps_target:
            mu_hi = mu_mid   # less shift (more noise) needed
        else:
            mu_lo = mu_mid   # more shift (less noise) needed
    return 1.0 / (0.5 * (mu_lo + mu_hi))


# ---------------------------------------------------------------------------
# Synthetic background pool
# ---------------------------------------------------------------------------

@dataclass
class SyntheticSpec:
    d: int = 256           # gradient dimension
    n: int = 12602         # number of background examples
    m: int = 16            # head subspace dimension
    kappa: float = 20.0    # anisotropy: λ_H / λ_L
    C: float = 1.0         # clip norm
    q: float = 1.0 / 9.0  # Poisson sampling rate  (matches S1 operating point)
    eps_target: float = 8.0
    delta: float = 1e-5
    T: int = 360           # steps (for σ calibration; not actually run)
    K_canaries: int = 200  # number of canaries (excluding eigendirection canaries)
    seed: int = 0
    nystrom_ranks: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64, 128])
    n_mc_samples: int = 100_000  # Monte-Carlo draws for realized distinguishability


def build_background(spec: SyntheticSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build background gradient pool with a power-law target spectrum.

    Target: λ_k ∝ (k+1)^{-β}, β = log(κ)/log(d)
    κ=1 → β=0 → isotropic (negative control), κ>1 → decaying spectrum.
    Gradients are sampled from this target then clipped to ‖·‖=C.
    Sigma_emp is computed empirically from the clipped gradients, so the
    model and the MC code share EXACTLY the same covariance object.

    Returns:
      bg_grads  : (n, d) clipped gradients, all ‖·‖=C
      eigvecs   : (d, d) columns = eigenvectors of Sigma_emp, descending λ
      lambdas   : (d,)   eigenvalues of Sigma_emp, descending
      Sigma_emp : (d, d) ρ * bg_grads.T @ bg_grads  (the shared Σ object)
    """
    rng = np.random.default_rng(spec.seed + 10000)
    d, n = spec.d, spec.n
    rho = spec.q * (1.0 - spec.q)

    # Power-law target: λ_k ∝ (k+1)^{-β}, κ = d^β → β = log(κ)/log(d)
    if spec.kappa <= 1.0 + 1e-9:
        beta = 0.0
        target_lambdas = np.ones(d, dtype=np.float64)
    else:
        beta = math.log(spec.kappa) / math.log(d)
        target_lambdas = np.arange(1, d + 1, dtype=np.float64) ** (-beta)
    target_lambdas *= rho * n * spec.C ** 2 / target_lambdas.sum()

    # Random eigenbasis for the target
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))

    # Sample n gradients in target eigenbasis, then rotate to ambient space
    z = rng.standard_normal((n, d))
    g_raw = (z * np.sqrt(target_lambdas)[None, :]) @ Q.T   # (n, d)

    # Clip to norm C
    norms = np.linalg.norm(g_raw, axis=1, keepdims=True)
    bg_grads = g_raw / np.maximum(norms, 1e-10) * spec.C   # (n, d)

    # Empirical covariance from the clipped gradients (already LOO: no canary in pool)
    Sigma_emp = rho * bg_grads.T @ bg_grads   # (d, d)

    # Spectral decomposition of Sigma_emp, sorted descending
    lambdas_emp, eigvecs_emp = np.linalg.eigh(Sigma_emp)
    lambdas_emp = lambdas_emp[::-1].copy()
    eigvecs_emp = eigvecs_emp[:, ::-1].copy()

    return bg_grads, eigvecs_emp, lambdas_emp, Sigma_emp


def compute_exact_shift(g: np.ndarray, Sigma: np.ndarray,
                         sigma: float, C: float) -> float:
    """d²_exact = g^T (σ²C²I + Σ)^{-1} g"""
    a = sigma ** 2 * C ** 2
    Stot = a * np.eye(len(g)) + Sigma
    return float(g @ np.linalg.solve(Stot, g))




def compute_nystrom_U(g: np.ndarray, eigvecs: np.ndarray,
                      lambdas: np.ndarray, sigma: float, C: float, r: int) -> float:
    """
    Nyström upper bound on d²_exact = g^T(aI+Σ)^{-1}g.

    U_full = g^T(aI+L_r)^{-1}g  (rank-r Nyström minorant L_r ≤ Σ in Loewner order).
    U_full ≥ d²_exact, and converges to d²_exact as r→d.

    No LOO correction: Sigma_emp is already leave-one-out (canaries are not in bg_grads),
    so no Sherman-Morrison adjustment is needed here.
    """
    a = sigma ** 2 * C ** 2
    r = min(r, len(lambdas))
    U_r = eigvecs[:, :r]    # (d, r) top-r eigenvectors
    lam_r = lambdas[:r]     # (r,) descending
    norm2_a = float(np.dot(g, g)) / a
    proj = U_r.T @ g        # (r,)
    # Woodbury: (aI+L_r)^{-1} = (1/a)I - (1/a²) U_r diag(λ/(a+λ)) U_r^T
    discount = (lam_r / (a + lam_r)) * proj ** 2
    return float(np.clip(norm2_a - np.sum(discount) / a, 0.0, norm2_a))


def analytic_shift_eigdir(k: int, lambdas: np.ndarray, sigma: float, C: float) -> float:
    """d²_exact for a canary aligned with eigenvector k: C²/(σ²C² + λ_k)."""
    a = sigma ** 2 * C ** 2
    return float(C ** 2 / (a + lambdas[k]))


# ---------------------------------------------------------------------------
# Monte-Carlo realized distinguishability
# ---------------------------------------------------------------------------

def mc_realized_batch(
    canaries: np.ndarray,      # (K, d) — all ‖canary‖=C (equal-norm)
    bg_gradients: np.ndarray,  # (n, d)
    sigma: float,
    C: float,
    q: float,
    n_samples: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Vectorized Monte-Carlo realized distinguishability for all K canaries at once.

    For each canary direction, D* = (μ₁−μ₀)/√(½(σ₀²+σ₁²)) where the transcript
    is projected onto the canary's unit direction. The Poisson inclusions are shared
    across canaries, so cost is O(n_samples × n + n_samples × K) not O(K × n_samples × n).

    Anti-circularity: we never touch Σ or d² here. The realized D* is computed
    from raw transcripts sampled from the true Poisson-sum law.

    Returns D_star: (K,) array.
    """
    rng = np.random.default_rng(seed)
    n, K = bg_gradients.shape[0], canaries.shape[0]
    C2 = sigma ** 2 * C ** 2  # DP noise variance per dimension

    # Pre-project all background gradients onto each canary direction: (n, K)
    g_units = canaries / (np.linalg.norm(canaries, axis=1, keepdims=True) + 1e-30)
    bg_proj = bg_gradients @ g_units.T   # (n, K)

    # --- World 0: no target ---
    # G0_k = Σ_j I_j ḡ_j·g_unit_k  (each sample row is one Poisson draw)
    I0 = (rng.random((n_samples, n)) < q)           # (S, n)
    G0 = I0.astype(np.float32) @ bg_proj.astype(np.float32)   # (S, K)
    noise0 = rng.standard_normal((n_samples, K)) * math.sqrt(C2)
    T0 = G0 + noise0   # (S, K)

    # --- World 1: target present with prob q ---
    I1 = (rng.random((n_samples, n)) < q)
    G1 = I1.astype(np.float32) @ bg_proj.astype(np.float32)   # (S, K)
    B_star = (rng.random(n_samples) < q).astype(np.float32)   # (S,)
    # g_star_proj_k = g_unit_k · g_k = ‖g_k‖ = C  for all k (equal-norm guarantee)
    g_star_proj = np.linalg.norm(canaries, axis=1)             # (K,)  ≈ C everywhere
    noise1 = rng.standard_normal((n_samples, K)) * math.sqrt(C2)
    T1 = G1 + B_star[:, None] * g_star_proj[None, :] + noise1   # (S, K)

    mu0 = T0.mean(axis=0); mu1 = T1.mean(axis=0)   # (K,)
    s0 = T0.std(axis=0);   s1 = T1.std(axis=0)     # (K,)
    D_star = (mu1 - mu0) / np.sqrt(0.5 * (s0 ** 2 + s1 ** 2) + 1e-30)
    return D_star.astype(np.float64)


# ---------------------------------------------------------------------------
# Assumption 4.2 Gaussianity check
# ---------------------------------------------------------------------------

def mc_gaussianity_check(
    canaries_subset: np.ndarray,  # (K_sub, d)
    bg_grads: np.ndarray,         # (n, d)
    Sigma_tot: np.ndarray,        # (d, d) = aI + Sigma_emp
    q: float,
    n_mc: int = 2000,
    seed: int = 42,
) -> List[dict]:
    """
    Project each Poisson-sum aggregate onto the Mahalanobis direction
    w_c = Sigma_tot^{-1} g_c / ‖…‖, then test Gaussianity against N(0,1).

    Poisson draws are shared across all K_sub canaries (same batching trick
    as mc_realized_batch), so cost is O(n_mc × n + n_mc × K_sub) not
    O(K_sub × n_mc × n).

    Under Assumption 4.2 the projected aggregate should be N(μ, σ²). Reports
    KS distance, skewness, excess kurtosis — should be small and direction-
    uniform (validates the assumption holds for all canary angles).
    """
    from scipy.stats import kstest, skew as sp_skew, kurtosis as sp_kurt
    from scipy.stats import norm as sp_norm

    rng = np.random.default_rng(seed)
    n, K_sub = bg_grads.shape[0], canaries_subset.shape[0]

    # Mahalanobis directions for each canary: W = (d, K_sub)
    W = np.empty((bg_grads.shape[1], K_sub), dtype=np.float64)
    valid = np.ones(K_sub, dtype=bool)
    for i, g in enumerate(canaries_subset):
        w = np.linalg.solve(Sigma_tot, g)
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-30:
            valid[i] = False
            W[:, i] = 0.0
        else:
            W[:, i] = w / w_norm

    # Project all background gradients onto all K_sub directions: (n, K_sub)
    BG_PROJ = bg_grads @ W   # (n, K_sub)

    # Theoretical moments (from projected bg pool, no canary in pool)
    mu_th = q * BG_PROJ.sum(axis=0)           # (K_sub,)
    var_th = q * (1.0 - q) * (BG_PROJ ** 2).sum(axis=0)  # (K_sub,)

    # Shared Poisson draws across all K_sub canaries
    I = (rng.random((n_mc, n)) < q)                           # (n_mc, n)
    P = I.astype(np.float32) @ BG_PROJ.astype(np.float32)    # (n_mc, K_sub)

    results = []
    for i in range(K_sub):
        if not valid[i] or var_th[i] < 1e-30:
            results.append({"i": i, "ks_stat": float("nan"),
                             "skewness": float("nan"), "excess_kurtosis": float("nan")})
            continue
        Z = (P[:, i].astype(np.float64) - mu_th[i]) / math.sqrt(var_th[i])
        ks_stat, _ = kstest(Z, sp_norm.cdf)
        results.append({
            "i": i,
            "ks_stat": float(ks_stat),
            "skewness": float(sp_skew(Z)),
            "excess_kurtosis": float(sp_kurt(Z)),
        })

    return results


# ---------------------------------------------------------------------------
# Build canary set
# ---------------------------------------------------------------------------

def make_canaries(spec: SyntheticSpec, eigvecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      canaries : (K, d) — all ‖canary‖=C  (equal-norm guarantee)
      angles   : (K,)  — cosine of angle with head subspace  (0=quiet, 1=head)

    Sweeps from "fully in head subspace" to "fully in quiet subspace".
    Also includes one canary per eigendirection (for the ladder figure).
    """
    rng = np.random.default_rng(spec.seed + 99)
    d, m, K, C = spec.d, spec.m, spec.K_canaries, spec.C

    # K angled canaries: linear sweep of head-alignment ∈ [0, 1]
    alphas = np.linspace(0.0, 1.0, K)   # alpha=1 → fully in head, 0 → fully quiet
    canaries = []
    angles = []
    for alpha in alphas:
        # Component in head subspace: u_H ~ random unit in span{u_0,...,u_{m-1}}
        w_H = rng.standard_normal(m)
        w_H /= np.linalg.norm(w_H) + 1e-30
        g_H = eigvecs[:, :m] @ w_H

        # Component in quiet subspace
        w_Q = rng.standard_normal(d - m)
        w_Q /= np.linalg.norm(w_Q) + 1e-30
        g_Q = eigvecs[:, m:] @ w_Q

        # Blend
        g = alpha * g_H + math.sqrt(max(1.0 - alpha ** 2, 0.0)) * g_Q
        g = g / (np.linalg.norm(g) + 1e-30) * C
        canaries.append(g)
        angles.append(float(alpha))

    canaries = np.stack(canaries)      # (K, d)
    angles = np.array(angles)          # (K,)
    return canaries, angles


def make_eigdir_canaries(spec: SyntheticSpec, eigvecs: np.ndarray) -> np.ndarray:
    """One canary per eigendirection: g_k = C * u_k."""
    return (eigvecs * spec.C).T   # (d, d), row k = C * eigenvector k



def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ. Returns 0.0 when x is constant (no ranking information)."""
    from scipy.stats import spearmanr
    import warnings
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    if x[mask].std() < 1e-12:
        return 0.0  # constant predictor — no ranking power by definition
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(x[mask], y[mask]).statistic)


# ---------------------------------------------------------------------------
# Main experiment functions
# ---------------------------------------------------------------------------

def run_one_spec(spec: SyntheticSpec, verbose: bool = True) -> dict:
    """
    Full synthetic experiment for one (kappa, eps_target, m) spec.
    Returns a result dict with all deliverable quantities.
    """
    t0 = time.time()
    # Calibrate σ
    sigma = calibrate_sigma(spec.eps_target, spec.q, spec.T, spec.delta)
    a = sigma ** 2 * spec.C ** 2

    if verbose:
        print(f"  κ={spec.kappa:6.1f}  ε={spec.eps_target}  m={spec.m}  "
              f"σ={sigma:.4f}  a={a:.5f}")

    # Build background — single call yields bg_grads AND Sigma_emp (no mismatch)
    bg_grads, eigvecs, lambdas, Sigma = build_background(spec)

    # Canaries
    canaries, angles = make_canaries(spec, eigvecs)
    K = len(canaries)

    # Norm shift (constant for all equal-norm canaries)
    mu_norm = spec.C / (sigma * spec.C)   # = 1/σ  (same for all canaries)
    eps_norm_all = np.full(K, eps_cert_from_rdp(
        np.array([spec.T * eps_sgm_scalar(a_val, spec.q, mu_norm)
                  for a_val in ALPHA_GRID]),
        ALPHA_GRID, spec.delta
    ))

    # Compute exact d² and Nyström upper bound for each canary
    exact_d2 = np.zeros(K)
    nystrom_d2_per_rank = {r: np.zeros(K) for r in spec.nystrom_ranks}

    for i, g in enumerate(canaries):
        exact_d2[i] = compute_exact_shift(g, Sigma, sigma, spec.C)
        for r in spec.nystrom_ranks:
            # No LOO correction: Sigma_emp is already leave-one-out
            nystrom_d2_per_rank[r][i] = compute_nystrom_U(
                g, eigvecs, lambdas, sigma, spec.C, r
            )

    # Realized D* — vectorized across all K canaries (single batch of Poisson draws)
    realized_D = mc_realized_batch(
        canaries, bg_grads, sigma, spec.C, spec.q,
        n_samples=spec.n_mc_samples, seed=spec.seed,
    )

    # ε^dir from exact d²  (what the theory predicts)
    eps_dir_exact = np.array([
        eps_cert_from_rdp(
            np.array([spec.T * eps_sgm_scalar(a_val, spec.q, math.sqrt(max(d2, 0.0)))
                      for a_val in ALPHA_GRID]),
            ALPHA_GRID, spec.delta
        )
        for d2 in exact_d2
    ])

    # ε^dir from Nyström d̂² at headline rank
    r_headline = spec.nystrom_ranks[-2] if len(spec.nystrom_ranks) >= 2 else spec.nystrom_ranks[-1]
    eps_dir_nystrom = np.array([
        eps_cert_from_rdp(
            np.array([spec.T * eps_sgm_scalar(a_val, spec.q, math.sqrt(max(d2h, 0.0)))
                      for a_val in ALPHA_GRID]),
            ALPHA_GRID, spec.delta
        )
        for d2h in nystrom_d2_per_rank[r_headline]
    ])

    # Masking discount (direction vs norm, using exact d²)
    norm_shift2 = (spec.C / (sigma * spec.C)) ** 2   # = 1/σ²
    masking_discount = 1.0 - exact_d2 / norm_shift2
    masking_discount = np.clip(masking_discount, 0.0, 1.0)
    median_masking = float(np.median(masking_discount))

    # Spearman correlations with realized D*
    rho_dir_exact = spearman_rho(eps_dir_exact, realized_D)
    rho_norm = spearman_rho(eps_norm_all, realized_D)
    rho_dir_nystrom = spearman_rho(eps_dir_nystrom, realized_D)
    delta_rho_exact = rho_dir_exact - rho_norm   # THE gap figure y-axis

    # Eigendirection canaries for ladder figure
    eig_canaries = make_eigdir_canaries(spec, eigvecs)  # (d, d)
    eig_exact_d2 = np.array([
        compute_exact_shift(eig_canaries[k], Sigma, sigma, spec.C)
        for k in range(spec.d)
    ])
    # Analytic: C²/(a+λ_k) where λ_k are empirical eigenvalues of Sigma_emp
    eig_analytic_d2 = spec.C ** 2 / (a + lambdas)
    eig_nystrom_d2 = {}
    for r in spec.nystrom_ranks:
        eig_nystrom_d2[r] = np.array([
            compute_nystrom_U(eig_canaries[k], eigvecs, lambdas, sigma, spec.C, r)
            for k in range(spec.d)
        ])

    # Nyström tightness: (Û² - d²_exact) / d²_exact per rank — should → 0 as r→d
    tightness = {}
    for r in spec.nystrom_ranks:
        d2h = nystrom_d2_per_rank[r]
        gap = d2h - exact_d2   # should be ≥ 0 (upper bound)
        rel = gap / np.maximum(exact_d2, 1e-15)
        tightness[r] = {
            "n_violations": int((gap < -1e-10).sum()),
            "gap_med": float(np.median(gap)),
            "gap_p95": float(np.percentile(gap, 95)),
            "rel_gap_med": float(np.median(rel)),
            "rel_gap_p95": float(np.percentile(rel, 95)),
        }

    # Assumption 4.2 Gaussianity check on 50 canaries
    n_check = min(50, K)
    check_idx = np.linspace(0, K - 1, n_check, dtype=int)
    Sigma_tot = a * np.eye(spec.d) + Sigma
    gaussianity = mc_gaussianity_check(
        canaries[check_idx], bg_grads, Sigma_tot, spec.q,
        n_mc=5000, seed=spec.seed + 777,
    )
    for j, idx in enumerate(check_idx):
        gaussianity[j]["masking_discount"] = float(masking_discount[idx])

    elapsed = time.time() - t0
    if verbose:
        print(f"    median_masking={median_masking:.3f}  Δρ(exact)={delta_rho_exact:+.4f}  "
              f"ρ_dir={rho_dir_exact:.4f}  ρ_norm={rho_norm:.4f}  [{elapsed:.1f}s]")

    return {
        "kappa": spec.kappa, "eps_target": spec.eps_target, "m": spec.m,
        "sigma": sigma, "q": spec.q, "T": spec.T, "d": spec.d, "n": spec.n,
        "K": K,
        "median_masking_discount": median_masking,
        "rho_dir_exact": rho_dir_exact,
        "rho_dir_nystrom": rho_dir_nystrom,
        "rho_norm": rho_norm,
        "delta_rho_exact": delta_rho_exact,
        "delta_rho_nystrom": float(rho_dir_nystrom - rho_norm),
        "angles": angles.tolist(),
        "exact_d2": exact_d2.tolist(),
        "realized_D": realized_D.tolist(),
        "eps_dir_exact": eps_dir_exact.tolist(),
        "eps_norm": eps_norm_all.tolist(),
        "masking_discount": masking_discount.tolist(),
        "tightness": tightness,
        "gaussianity": gaussianity,
        "eig_exact_d2": eig_exact_d2.tolist(),
        "eig_analytic_d2": eig_analytic_d2.tolist(),
        "eig_nystrom_d2": {str(r): v.tolist() for r, v in eig_nystrom_d2.items()},
        "lambdas": lambdas.tolist(),
        "elapsed_s": elapsed,
        "nystrom_ranks": spec.nystrom_ranks,
        "r_headline": r_headline,
    }


# ---------------------------------------------------------------------------
# Figure A: Eigendirection ladder
# ---------------------------------------------------------------------------

def fig_ladder(results: dict, out_dir: str) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [fig_ladder] matplotlib not available"); return

    os.makedirs(out_dir, exist_ok=True)
    eig_exact = np.array(results["eig_exact_d2"])
    eig_analytic = np.array(results["eig_analytic_d2"])
    norm_shift2 = 1.0 / results["sigma"] ** 2
    d = len(eig_exact)
    m = results["m"]
    k_idx = np.arange(d)

    # Only show first 64 for readability
    n_show = min(64, d)
    r_headline = results["r_headline"]
    eig_nystrom = np.array(results["eig_nystrom_d2"][str(r_headline)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(k_idx[:n_show], eig_analytic[:n_show], "k-", lw=1.5,
             label=r"analytic $C^2/(a+\lambda_k)$", zorder=3)
    ax1.plot(k_idx[:n_show], eig_exact[:n_show], "b--", lw=1.2,
             label=r"exact $d^2$", zorder=4)
    ax1.plot(k_idx[:n_show], eig_nystrom[:n_show], "g:", lw=1.2,
             label=fr"Nyström $\hat{{d}}^2$ (r={r_headline})", zorder=5)
    ax1.axhline(norm_shift2, color="red", ls="-.", lw=1.2, label=r"norm shift $1/\sigma^2$")
    ax1.axvline(m - 0.5, color="gray", ls=":", lw=1.0, label=f"head/quiet boundary (m={m})")
    ax1.set_xlabel("Eigendirection index k")
    ax1.set_ylabel(r"Shift $d^2$")
    ax1.set_title(f"Eigendirection ladder\n(κ={results['kappa']}, ε={results['eps_target']}, m={m})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right panel: same on log scale
    ax2.semilogy(k_idx[:n_show], np.maximum(eig_analytic[:n_show], 1e-15), "k-", lw=1.5,
                 label=r"analytic")
    ax2.semilogy(k_idx[:n_show], np.maximum(eig_exact[:n_show], 1e-15), "b--", lw=1.2,
                 label=r"exact")
    ax2.semilogy(k_idx[:n_show], np.maximum(eig_nystrom[:n_show], 1e-15), "g:", lw=1.2,
                 label=fr"Nyström r={r_headline}")
    ax2.axhline(norm_shift2, color="red", ls="-.", lw=1.2, label="norm shift")
    ax2.axvline(m - 0.5, color="gray", ls=":", lw=1.0)
    ax2.set_xlabel("Eigendirection index k")
    ax2.set_ylabel(r"$d^2$ (log scale)")
    ax2.set_title("Log scale")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Fig A: Directional shift sweeps full Rayleigh-quotient range", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "figA_eigdir_ladder.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig_ladder] → {path}")


# ---------------------------------------------------------------------------
# Figure B: Anisotropy vs gap
# ---------------------------------------------------------------------------

def fig_gap(all_results: list, out_dir: str) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("  [fig_gap] matplotlib not available"); return

    os.makedirs(out_dir, exist_ok=True)

    # Group by (eps_target, m), sweep kappa
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        groups[(r["eps_target"], r["m"])].append(r)

    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5), squeeze=False)

    cmap = cm.get_cmap("viridis")
    eps_vals = sorted(set(r["eps_target"] for r in all_results))
    eps_colors = {e: cmap(i / max(len(eps_vals) - 1, 1)) for i, e in enumerate(eps_vals)}

    for ax_idx, ((eps, m), group) in enumerate(sorted(groups.items())):
        ax = axes[0][ax_idx]
        group = sorted(group, key=lambda r: r["kappa"])

        x = np.array([r["median_masking_discount"] for r in group])
        y_exact = np.array([r["delta_rho_exact"] for r in group])
        y_nystrom = np.array([r["delta_rho_nystrom"] for r in group])
        kappas = [r["kappa"] for r in group]

        ax.plot(x, y_exact, "o-", color=eps_colors[eps], lw=1.5,
                label=f"ε={eps}, exact", ms=7)
        ax.plot(x, y_nystrom, "s--", color=eps_colors[eps], lw=1.0,
                alpha=0.7, label=f"ε={eps}, Nyström", ms=5)

        # Annotate kappa values
        for xi, yi, k in zip(x, y_exact, kappas):
            ax.annotate(f"κ={k:.0f}", (xi, yi), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")

        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.fill_between([0, 1], -0.1, 0, alpha=0.08, color="red",
                         label="dir worse than norm")
        ax.set_xlabel("Median masking discount (1 − d²/‖ḡ‖²/σ²C²)")
        ax.set_ylabel("Δρ = ρ(ε^dir, D*) − ρ(ε^norm, D*)")
        ax.set_title(f"m={m}, ε={eps}\nGap rises with anisotropy")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle("Fig B: Anisotropy-vs-Advantage\n"
                 "x-axis = free (from logged masking), "
                 "y-axis = faithfulness gain of direction over norm",
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "figB_anisotropy_vs_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig_gap] → {path}")


# ---------------------------------------------------------------------------
# Figure C: Assumption 4.2 Gaussianity check
# ---------------------------------------------------------------------------

def fig_assumption(all_results: list, out_dir: str) -> None:
    """
    Validate Assumption 4.2: the Poisson-sum aggregate projected onto the
    Mahalanobis direction w = Sigma_tot^{-1}g is approximately Gaussian.

    Left panel: KS distance vs masking discount (should be uniformly small).
    Right panel: excess kurtosis vs masking discount (should be near 0).
    Coloured by κ to show assumption holds across anisotropy levels.
    """
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [fig_assumption] matplotlib not available"); return

    os.makedirs(out_dir, exist_ok=True)

    all_masking, all_ks, all_kurt, all_kappa = [], [], [], []
    for r in all_results:
        for g in r.get("gaussianity", []):
            if not math.isfinite(g.get("ks_stat", float("nan"))):
                continue
            all_masking.append(g["masking_discount"])
            all_ks.append(g["ks_stat"])
            all_kurt.append(g["excess_kurtosis"])
            all_kappa.append(r["kappa"])

    if not all_masking:
        print("  [fig_assumption] no gaussianity data"); return

    all_masking = np.array(all_masking)
    all_ks = np.array(all_ks)
    all_kurt = np.array(all_kurt)
    all_kappa = np.array(all_kappa)

    kappas_uniq = sorted(set(all_kappa))
    cmap = plt.colormaps["viridis"]
    kappa_colors = {k: cmap(i / max(len(kappas_uniq) - 1, 1))
                    for i, k in enumerate(kappas_uniq)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for k in kappas_uniq:
        mask = all_kappa == k
        ax1.scatter(all_masking[mask], all_ks[mask],
                    color=kappa_colors[k], s=18, alpha=0.65, label=f"κ={k:.0f}")
        ax2.scatter(all_masking[mask], all_kurt[mask],
                    color=kappa_colors[k], s=18, alpha=0.65, label=f"κ={k:.0f}")

    # Reference bands
    ax1.axhline(0.05, color="gray", ls="--", lw=0.8, label="KS=0.05 reference")
    ax2.axhline(0.0, color="gray", ls="--", lw=0.8, label="kurtosis=0")
    ax2.axhline(0.5, color="gray", ls=":", lw=0.6)
    ax2.axhline(-0.5, color="gray", ls=":", lw=0.6)

    ax1.set_xlabel("Masking discount (1 − d²/‖g‖²σ²)")
    ax1.set_ylabel("KS distance vs N(0,1)")
    ax1.set_title("Assumption 4.2: KS(projected aggregate, Gaussian)\n"
                  "Small ∀ masking values → assumption holds direction-uniformly")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)

    ax2.set_xlabel("Masking discount")
    ax2.set_ylabel("Excess kurtosis")
    ax2.set_title("Excess kurtosis of projected aggregate\n"
                  "Near 0 → Gaussian third+fourth moments match")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)

    fig.suptitle("Fig C: Projection onto Mahalanobis direction w=Σ_tot⁻¹g/‖…‖\n"
                 "validates Assumption 4.2 for all canary directions and all κ",
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "figC_assumption_check.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig_assumption] → {path}")


# ---------------------------------------------------------------------------
# Figure D: ε sweep at fixed kappa
# ---------------------------------------------------------------------------

def fig_eps_sweep(all_results: list, out_dir: str, fixed_kappa: float = 20.0) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [fig_eps_sweep] matplotlib not available"); return

    os.makedirs(out_dir, exist_ok=True)
    subset = [r for r in all_results if abs(r["kappa"] - fixed_kappa) < 0.5]
    if not subset:
        print(f"  [fig_eps_sweep] no results for κ={fixed_kappa}"); return
    subset = sorted(subset, key=lambda r: r["eps_target"])

    eps_vals = [r["eps_target"] for r in subset]
    y_delta = [r["delta_rho_exact"] for r in subset]
    y_dir = [r["rho_dir_exact"] for r in subset]
    y_norm = [r["rho_norm"] for r in subset]
    x_mask = [r["median_masking_discount"] for r in subset]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(eps_vals, y_delta, "o-", color="steelblue", lw=1.8, ms=8, label="Δρ (dir−norm)")
    ax1.axhline(0, color="gray", ls=":", lw=0.8)
    ax1.set_xlabel("Privacy budget ε (higher = less private)")
    ax1.set_ylabel("Δρ = ρ(ε^dir, D*) − ρ(ε^norm, D*)")
    ax1.set_title(f"ε sweep (κ={fixed_kappa:.0f})\nAdvantage shrinks as ε tightens")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(eps_vals, y_dir, "o-", color="steelblue", lw=1.5, ms=7, label="ρ(dir, D*)")
    ax2.plot(eps_vals, y_norm, "s--", color="firebrick", lw=1.5, ms=7, label="ρ(norm, D*)")
    ax2_r = ax2.twinx()
    ax2_r.plot(eps_vals, x_mask, "^:", color="green", lw=1.0, ms=5, label="masking discount")
    ax2_r.set_ylabel("Median masking discount", color="green")
    ax2_r.tick_params(axis="y", labelcolor="green")
    ax2.set_xlabel("ε")
    ax2.set_ylabel("Spearman ρ")
    ax2.set_title("ρ vs ε")
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Fig D: Privacy budget sweep at κ={fixed_kappa:.0f}\n"
                 "Advantage decays as ε tightens (isotropization of Σ_tot)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "figD_eps_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [fig_eps_sweep] → {path}")


# ---------------------------------------------------------------------------
# Table T: Nyström tightness
# ---------------------------------------------------------------------------

def table_tightness(all_results: list, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    print(f"\n{'='*72}")
    print("  Table T: Nyström Tightness (d̂² - d²_exact) / d²_exact")
    print(f"{'='*72}")
    hdr = f"  {'κ':6s} {'ε':4s} {'m':3s} {'r':5s}  {'n_viol':6s}  {'gap_med':8s}  {'gap_p95':8s}  {'rel_med':8s}  {'rel_p95':8s}"
    print(hdr)

    for r in sorted(all_results, key=lambda x: (x["kappa"], x["eps_target"])):
        for rk, t in sorted(r["tightness"].items()):
            row = {"kappa": r["kappa"], "eps": r["eps_target"], "m": r["m"],
                   "rank": rk, **t}
            rows.append(row)
            print(f"  {r['kappa']:6.1f} {r['eps_target']:4.1f} {r['m']:3d} {rk:5d}  "
                  f"{t['n_violations']:6d}  {t['gap_med']:8.5f}  {t['gap_p95']:8.5f}  "
                  f"{t['rel_gap_med']:8.4f}  {t['rel_gap_p95']:8.4f}")

    path = os.path.join(out_dir, "tableT_nystrom_tightness.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  [saved] {path}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

DEFAULT_KAPPAS = [1.0, 2.0, 5.0, 20.0, 100.0, 500.0]
DEFAULT_EPS = [1.0, 2.0, 4.0, 8.0, 16.0]
DEFAULT_M = [16]


def run_grid(kappas=None, eps_list=None, m_list=None,
             d=256, n=12602, q=1.0/9.0, T=360, delta=1e-5,
             K=200, n_mc=50_000, seed=0,
             nystrom_ranks=None, verbose=True, out_dir=OUT_DIR) -> list:
    if kappas is None: kappas = DEFAULT_KAPPAS
    if eps_list is None: eps_list = DEFAULT_EPS
    if m_list is None: m_list = DEFAULT_M
    if nystrom_ranks is None: nystrom_ranks = [8, 16, 32, 64, 128]

    os.makedirs(out_dir, exist_ok=True)
    all_results = []
    total = len(kappas) * len(eps_list) * len(m_list)
    done = 0
    for m in m_list:
        for eps in eps_list:
            for kappa in kappas:
                done += 1
                print(f"\n[{done}/{total}] κ={kappa}  ε={eps}  m={m}")
                spec = SyntheticSpec(
                    d=d, n=n, m=m, kappa=kappa,
                    C=1.0, q=q, eps_target=eps, delta=delta, T=T,
                    K_canaries=K, seed=seed,
                    nystrom_ranks=nystrom_ranks, n_mc_samples=n_mc,
                )
                res = run_one_spec(spec, verbose=verbose)
                all_results.append(res)

    path = os.path.join(out_dir, "synthetic_grid_results.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved] {path}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Run full grid and produce all figures")
    parser.add_argument("--fig", choices=["ladder", "gap", "assumption", "eps_sweep"],
                        help="Run a specific figure only")
    parser.add_argument("--table", choices=["tightness"],
                        help="Run a specific table only")
    parser.add_argument("--kappas", nargs="+", type=float, default=DEFAULT_KAPPAS)
    parser.add_argument("--eps", nargs="+", type=float, default=DEFAULT_EPS)
    parser.add_argument("--m", nargs="+", type=int, default=DEFAULT_M)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--n", type=int, default=12602)
    parser.add_argument("--q", type=float, default=1.0/9.0)
    parser.add_argument("--T", type=int, default=360)
    parser.add_argument("--K", type=int, default=200,
                        help="Number of angled canaries")
    parser.add_argument("--n_mc", type=int, default=50_000,
                        help="Monte-Carlo samples for realized D*")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load", type=str, default=None,
                        help="Load existing grid results JSON instead of rerunning")
    parser.add_argument("--fast", action="store_true",
                        help="Quick smoke-test: 3 kappas, 2 eps, 10k MC samples")
    parser.add_argument("--out", type=str, default=OUT_DIR)
    args = parser.parse_args()

    if args.fast:
        args.kappas = [1.0, 20.0, 100.0]
        args.eps = [4.0, 8.0]
        args.n_mc = 10_000
        args.K = 50

    if args.load:
        print(f"[load] {args.load}")
        with open(args.load) as f:
            all_results = json.load(f)
    else:
        all_results = run_grid(
            kappas=args.kappas, eps_list=args.eps, m_list=args.m,
            d=args.d, n=args.n, q=args.q, T=args.T, delta=1e-5,
            K=args.K, n_mc=args.n_mc, seed=args.seed,
            out_dir=args.out,
        )

    # Pick a reference result for ladder figure (use first kappa, headline eps=8)
    ref_eps = 8.0
    ref_kappa = args.kappas[len(args.kappas) // 2]
    ref_results = [r for r in all_results
                   if abs(r["eps_target"] - ref_eps) < 0.5
                   and abs(r["kappa"] - ref_kappa) < 0.5]
    if not ref_results:
        ref_results = [all_results[0]]

    produce_all = args.all or (args.fig is None and args.table is None)

    if produce_all or args.fig == "ladder":
        fig_ladder(ref_results[0], args.out)

    if produce_all or args.fig == "gap":
        fig_gap(all_results, args.out)

    if produce_all or args.fig == "assumption":
        fig_assumption(all_results, args.out)

    if produce_all or args.fig == "eps_sweep":
        # Use median kappa for sweep
        fig_eps_sweep(all_results, args.out, fixed_kappa=ref_kappa)

    if produce_all or args.table == "tightness":
        table_tightness(all_results, args.out)

    # Print summary
    print(f"\n{'='*72}")
    print("  Summary: anisotropy-vs-gap (headline ε=8)")
    print(f"{'='*72}")
    print(f"  {'κ':8s}  {'masking':8s}  {'Δρ(exact)':10s}  {'ρ_dir':7s}  {'ρ_norm':7s}")
    for r in sorted([x for x in all_results if abs(x["eps_target"] - 8.0) < 0.5],
                    key=lambda x: x["kappa"]):
        print(f"  {r['kappa']:8.1f}  {r['median_masking_discount']:8.3f}  "
              f"{r['delta_rho_exact']:+10.4f}  {r['rho_dir_exact']:7.4f}  "
              f"{r['rho_norm']:7.4f}")


if __name__ == "__main__":
    main()
