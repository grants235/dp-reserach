#!/usr/bin/env python3
"""
Phase 19 Certificate: Fixed-Grid Rényi-Filter Direction-Aware Bound
====================================================================

Spec: phase19_spex.md, Sections 8–9 and 13.

Algorithm:
  1. Load per-step Nyström stats from training run.
  2. Commit budget grid B_max(α) = T × ε_sgm(α, q, 1/σ) BEFORE any per-instance analysis.
     Save filter_budget_grid.json.
  3. For each step t: compute per-instance shifts μ_dir and μ_norm.
  4. For each order α: accumulate C_i^v(α) = Σ_t ε_sgm(α, q, μ_{i,t}^v).
  5. Grid-round: B_{i,cert}^v(α) = Δ_α × ⌈C_i^v(α)/Δ_α⌉  (≤ B_max(α)).
  6. Convert: ε_{i,cert}^v(δ) = min_α [B_{i,cert}^v(α) + log(1/δ)/(α−1)].

ε_sgm implementation:
  Uses 64-point Gauss-Hermite quadrature for the exact Poisson-subsampled
  Gaussian RDP at any real order α > 1:

    ε_sgm(α, q, μ) = (1/(α−1)) log E_{x~N(0,1)}[(1-q + q·e^{μx−μ²/2})^α]

  Computed stably in log-space; vectorized over examples.

Usage:
  python experiments/exp_p19_certify.py --run F1 --seed 0
  python experiments/exp_p19_certify.py --run F1 --all_seeds
  python experiments/exp_p19_certify.py --all
  python experiments/exp_p19_certify.py --table
"""

import os, sys, json, math, argparse, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from numpy.polynomial.hermite import hermgauss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RUNS_DIR = "./runs/p19"
CERT_DIR = "./certs/p19"

RANK_ABLATION = [10, 25, 50, 100, 200]

# α grid: low end densified for small-shift regimes (F7/F8 barely-clipped MNIST).
# Orders below 1.5 carry large log(1/δ)/(α-1) conversion penalties for δ≤1e-5 and
# are never selected by the δ-penalized min; they are included to support future
# tighter conversions (e.g. PRV or fixed-α comparison) that bypass that penalty.
ALPHA_GRID = np.concatenate([
    np.array([1.01, 1.05, 1.1, 1.2, 1.25, 1.35]),
    np.arange(1.5,  10,    0.5),
    np.arange(10,   100,   2.0),
    np.arange(100,  1000,  20.0),
    np.arange(1000, 5001,  100.0),
]).astype(np.float64)

# Gauss-Hermite quadrature constants (64 points)
_GH_T, _GH_W = hermgauss(64)          # t_j, w_j such that ∫f(t)e^{-t²}dt ≈ Σw_j f(t_j)
_GH_T   = _GH_T.astype(np.float64)
_GH_W   = _GH_W.astype(np.float64)
_LOG_SQRT_PI = 0.5 * math.log(math.pi)
_SQRT2  = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# ε_sgm: exact Poisson-subsampled Gaussian RDP via Gauss-Hermite quadrature
# ---------------------------------------------------------------------------

def eps_sgm_vec(alpha: float, q: float, mu_array: np.ndarray) -> np.ndarray:
    """
    Compute ε_sgm(alpha, q, μ) for an array of μ values.

    ε_sgm(α, q, μ) = (1/(α−1)) × log E_{x~N(0,1)}[(1−q + q·e^{μx−μ²/2})^α]

    Uses 64-point Gauss-Hermite quadrature with change of variables x = √2·t:
      E[f(x)] ≈ (1/√π) Σ_j w_j f(√2·t_j)

    Fully vectorized over mu_array; numerically stable via log-space logsumexp.

    Args:
      alpha     : RDP order (real > 1)
      q         : Poisson sampling rate
      mu_array  : array of non-negative shift values ‖ḡ‖/(σC)

    Returns:
      eps array of same shape as mu_array
    """
    assert alpha > 1.0, f"alpha must be > 1, got {alpha}"
    mu = np.asarray(mu_array, dtype=np.float64).ravel()
    n  = len(mu)
    result = np.zeros(n, dtype=np.float64)

    pos_mask = (mu > 1e-15)
    if not pos_mask.any():
        return result.reshape(mu_array.shape if hasattr(mu_array, 'shape') else (n,))

    mu_nz = mu[pos_mask][:, None]     # (n_nz, 1)
    t     = _GH_T[None, :]            # (1, K)
    w     = _GH_W[None, :]            # (1, K)

    # v_{i,j} = μ_i × √2 × t_j − μ_i²/2   (exponent in e^{μx−μ²/2} at x=√2·t_j)
    v = mu_nz * (_SQRT2 * t) - 0.5 * mu_nz ** 2    # (n_nz, K)

    # log((1−q) + q·e^v) = logaddexp(log(1−q), log(q)+v)  [numerically stable]
    log_base = np.logaddexp(np.log(1.0 - q), np.log(q) + v)  # (n_nz, K)

    # log_g_{i,j} = α × log_base_{i,j}
    log_g = alpha * log_base       # (n_nz, K)

    # log(Σ_j (w_j/√π) × exp(log_g_{i,j}))
    # = logsumexp_j ( log(w_j) − log(√π) + log_g_{i,j} )
    log_contrib = np.log(w) - _LOG_SQRT_PI + log_g    # (n_nz, K)

    # Stable logsumexp over K dimension
    max_lc = log_contrib.max(axis=1, keepdims=True)     # (n_nz, 1)
    log_E  = max_lc[:, 0] + np.log(
        np.exp(log_contrib - max_lc).sum(axis=1))       # (n_nz,)

    result[pos_mask] = log_E / (alpha - 1.0)
    return result.reshape(mu_array.shape if hasattr(mu_array, 'shape') else (n,))


def eps_sgm_scalar(alpha: float, q: float, mu: float) -> float:
    """Scalar version of eps_sgm_vec."""
    return float(eps_sgm_vec(alpha, q, np.array([mu]))[0])


# ---------------------------------------------------------------------------
# Budget grid (Section 8.3) — must be committed BEFORE per-instance analysis
# ---------------------------------------------------------------------------

def build_budget_grid(sigma: float, q: float, T: int,
                      alpha_grid: np.ndarray = ALPHA_GRID) -> dict:
    """
    Build the data-independent fixed budget grid.

    B_max(α) = T × ε_sgm(α, q, 1/σ)
    Δ_α      = max(1e-5, B_max(α)/5000)
    M_α      = ⌈B_max(α)/Δ_α⌉
    Grid:    {0, Δ_α, 2Δ_α, ..., M_α × Δ_α}

    Returns dict with keys: Bmax, Delta, M (all indexed by alpha_grid position).
    """
    mu_ceiling = 1.0 / sigma   # maximum possible μ (‖ḡ‖=C=1 → μ=1/σ)
    Bmax  = np.zeros(len(alpha_grid), dtype=np.float64)
    Delta = np.zeros(len(alpha_grid), dtype=np.float64)
    M_arr = np.zeros(len(alpha_grid), dtype=np.int64)

    for j, alpha in enumerate(alpha_grid):
        eps_ceil = eps_sgm_scalar(alpha, q, mu_ceiling)
        bmax     = T * eps_ceil
        # Grid step: B_max/100000 (was /5000) so that barely-clipped regimes
        # (e.g. F7/F8 where μ_dir << 1/σ) land in distinct buckets rather than
        # all collapsing to bucket-1 and producing a constant certified ε.
        delta_a  = max(1e-7, bmax / 100000.0)
        m_a      = math.ceil(bmax / delta_a) if bmax > 0 else 0
        Bmax[j]  = bmax
        Delta[j] = delta_a
        M_arr[j] = m_a

    return {"Bmax": Bmax, "Delta": Delta, "M": M_arr, "alpha_grid": alpha_grid,
            "sigma": sigma, "q": q, "T": T, "mu_ceiling": mu_ceiling}


def save_budget_grid(grid: dict, path: str) -> None:
    """Serialize budget grid to JSON (must be saved BEFORE charge analysis)."""
    alpha_list = grid["alpha_grid"].tolist()
    Bmax_dict  = {str(a): float(b) for a, b in zip(alpha_list, grid["Bmax"])}
    Delta_dict = {str(a): float(d) for a, d in zip(alpha_list, grid["Delta"])}
    M_dict     = {str(a): int(m)   for a, m in zip(alpha_list, grid["M"])}
    doc = {
        "alphas":                  alpha_list,
        "Bmax_by_alpha":           Bmax_dict,
        "Delta_by_alpha":          Delta_dict,
        "M_by_alpha":              M_dict,
        "sigma":                   float(grid["sigma"]),
        "q":                       float(grid["q"]),
        "T":                       int(grid["T"]),
        "mu_ceiling":              float(grid["mu_ceiling"]),
        "grid_rule":               "Delta=max(1e-5, Bmax/5000)",
        "created_before_charge_analysis": True,
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"  [grid] Saved budget grid → {path}")


# ---------------------------------------------------------------------------
# Nyström bound computation (Section 9)
# ---------------------------------------------------------------------------

def compute_B_pseudo_sqrt_inv(B_r: np.ndarray) -> np.ndarray:
    """
    S_r = B_r^{†/2}: symmetric PSD pseudo-square-root inverse.
    Threshold: τ = 1e-8 × max(λ_max, 1e-30).
    """
    eigvals, eigvecs = np.linalg.eigh(B_r.astype(np.float64))
    lam_max = eigvals.max()
    tau = 1e-8 * max(float(lam_max), 1e-30)
    inv_sqrt = np.where(eigvals > tau, eigvals ** (-0.5), 0.0)
    return (eigvecs * inv_sqrt[None, :]) @ eigvecs.T    # (r, r) symmetric PSD


def compute_U_full_and_d2hat(
    clipped_norms_t: np.ndarray,   # (n,)
    B_t: np.ndarray,               # (r_max, r_max)
    YTY_t: np.ndarray,             # (r_max, r_max)
    Y_proj_t: np.ndarray,          # (n, r_max)
    a: float,
    rho: float,
    r: int,
) -> tuple:
    """
    Compute U_{i,t,r}^full and d̂²_{i,t,r} for all n examples at one step.

    Returns (U_full, d2_hat, fallback_mask) all shape (n,).

    U_{i}^full = ‖ḡ_i‖²/a − (1/a²) z_i^T (I + a^{-1} H)^{-1} z_i
    d̂²_i      = U_i/(1−ρU_i)  if ρU_i < 1,  else ‖ḡ_i‖²/a
    Both clipped to [0, ‖ḡ_i‖²/a].
    """
    r = min(r, B_t.shape[0])

    # Truncate to rank r
    B_r   = B_t[:r, :r].astype(np.float64)
    M_r   = YTY_t[:r, :r].astype(np.float64)
    s_r   = Y_proj_t[:, :r].astype(np.float64)    # (n, r)

    norm2   = clipped_norms_t.astype(np.float64) ** 2
    norm2_a = norm2 / a                            # ‖ḡ‖²/a for each i

    # S_r = B_r^{†/2}
    S_r = compute_B_pseudo_sqrt_inv(B_r)           # (r, r)

    # z_i = S_r s_{i,t,r}   (r,) for each i
    z = (S_r @ s_r.T).T                            # (n, r)

    # H = S_r M_r S_r
    H_r = S_r @ M_r @ S_r                         # (r, r)

    # A = I + (1/a) H_r
    A = np.eye(r) + H_r / a                        # (r, r)

    # Solve A x = z^T for all examples simultaneously
    # A is (r×r), z^T is (r×n); result Ainv_z is (n×r)
    Ainv_z = np.linalg.solve(A, z.T).T            # (n, r)

    # z^T A^{-1} z per row
    ztAinvz = (z * Ainv_z).sum(axis=1)            # (n,)

    # U^full = ‖ḡ‖²/a − (1/a²) z^T A^{-1} z
    U_full = norm2_a - ztAinvz / (a ** 2)

    # Clip to [0, ‖ḡ‖²/a]
    U_full = np.clip(U_full, 0.0, norm2_a)

    # Leave-one-out correction: d̂² = U/(1−ρU) capped at ‖ḡ‖²/a
    rho_U = rho * U_full
    fallback = (rho_U >= 1.0)
    denom    = np.where(fallback, 1.0, 1.0 - rho_U)
    denom    = np.maximum(denom, 1e-30)
    d2_hat   = np.where(fallback, norm2_a,
                        np.minimum(U_full / denom, norm2_a))
    d2_hat   = np.clip(d2_hat, 0.0, norm2_a)

    return U_full, d2_hat, fallback


# ---------------------------------------------------------------------------
# Per-step charge computation and filter replay
# ---------------------------------------------------------------------------

def _save_partial_ckpt(prefix, t_completed, C_norm, C_dir,
                       fallback_total, mu_dir_max, mu_norm_max):
    np.save(f"{prefix}_partial_C_norm.npy",   C_norm.astype(np.float64))
    np.save(f"{prefix}_partial_C_dir100.npy", C_dir.astype(np.float64))
    with open(f"{prefix}_partial_ckpt_meta.json", "w") as _f:
        json.dump({"t_completed": t_completed,
                   "fallback_total": int(fallback_total),
                   "mu_dir_max":     float(mu_dir_max),
                   "mu_norm_max":    float(mu_norm_max)}, _f)


def compute_realized_cumulative(
    clipped_norms: np.ndarray,   # (n, T)
    B_matrices: np.ndarray,      # (T, r_max, r_max)
    YTY_matrices: np.ndarray,    # (T, r_max, r_max)
    Y_projections: np.ndarray,   # (n, T, r_max)
    a: float,
    rho: float,
    sigma: float,
    q: float,
    r: int,
    alpha_grid: np.ndarray,
    start_t: int = 0,
    C_norm_init: np.ndarray = None,
    C_dir_init: np.ndarray = None,
    fallback_total_init: int = 0,
    mu_dir_max_init: float = 0.0,
    mu_norm_max_init: float = 0.0,
    ckpt_prefix: str = None,
    ckpt_every: int = 100,
) -> tuple:
    """
    For all examples i and orders α, compute:
      C_realized_norm[i, j]   = Σ_t ε_sgm(α_j, q, μ_norm_{i,t})
      C_realized_dir[i, j]    = Σ_t ε_sgm(α_j, q, μ_dir_{i,t,r})

    Also returns (fallback_rate, mu_dir_max, mu_norm_max) for sanity checks.

    The per-step norm comparator shift:
      μ_norm_{i,t} = ‖ḡ_{i,t}‖ / (σ × C) = ‖ḡ_{i,t}‖ / √a

    The direction-aware shift:
      μ_dir_{i,t,r} = √(d̂²_{i,t,r})

    Both shifts must satisfy μ ≤ 1/σ (checked in sanity checks).

    start_t / C_norm_init / C_dir_init / *_init allow resuming from a
    partial checkpoint; ckpt_prefix/ckpt_every control mid-run saves.
    """
    n, T = clipped_norms.shape
    n_alpha = len(alpha_grid)

    C_norm = C_norm_init.copy() if C_norm_init is not None \
             else np.zeros((n, n_alpha), dtype=np.float64)
    C_dir  = C_dir_init.copy() if C_dir_init is not None \
             else np.zeros((n, n_alpha), dtype=np.float64)

    fallback_total = fallback_total_init
    mu_dir_max  = mu_dir_max_init
    mu_norm_max = mu_norm_max_init

    for t in range(start_t, T):
        norms_t = clipped_norms[:, t]              # (n,)
        B_t     = B_matrices[t]
        YTY_t   = YTY_matrices[t]
        Yp_t    = Y_projections[:, t, :]           # (n, r_max)

        # Norm-based shift: μ_norm = ‖ḡ‖/(σC) = ‖ḡ‖/√a  (since a=σ²C²)
        mu_norm = norms_t.astype(np.float64) / math.sqrt(a)   # (n,)

        # Direction-aware shift: μ_dir = √(d̂²)
        U_full, d2_hat, fallback_t = compute_U_full_and_d2hat(
            norms_t, B_t, YTY_t, Yp_t, a, rho, r)
        mu_dir = np.sqrt(np.maximum(d2_hat, 0.0))             # (n,)

        fallback_total += fallback_t.sum()
        mu_dir_max  = max(mu_dir_max,  float(mu_dir.max()))
        mu_norm_max = max(mu_norm_max, float(mu_norm.max()))

        # Accumulate per-step charges for every α order
        for j, alpha in enumerate(alpha_grid):
            c_norm = eps_sgm_vec(alpha, q, mu_norm)   # (n,)
            c_dir  = eps_sgm_vec(alpha, q, mu_dir)    # (n,)
            C_norm[:, j] += c_norm
            C_dir[:, j]  += c_dir

        if t % 50 == 0:
            print(f"    [certify step {t:4d}/{T}]  "
                  f"mu_norm_max={mu_norm.max():.5f}  "
                  f"mu_dir_max={mu_dir.max():.5f}  "
                  f"fallback={fallback_t.sum()}")

        # Partial checkpoint every ckpt_every steps
        if ckpt_prefix and (t + 1) % ckpt_every == 0 and (t + 1) < T:
            _save_partial_ckpt(ckpt_prefix, t, C_norm, C_dir,
                               fallback_total, mu_dir_max, mu_norm_max)
            print(f"    [certify] partial checkpoint saved at step {t+1}/{T}")

    fallback_rate = fallback_total / (n * T) if T > 0 else 0.0
    return C_norm, C_dir, fallback_rate, mu_dir_max, mu_norm_max


def filter_replay(C_realized: np.ndarray,   # (n, n_alpha)
                  grid: dict) -> np.ndarray:
    """
    Fixed-grid filter replay (spec §8.4).

    For each (i, α), the filter completes for budget B iff C_i(α) ≤ B.
    Since B_{i,cert}(α) = min{B ∈ grid : B ≥ C_i(α)}, this is equivalent to
    ceiling-rounding the realized spend to the nearest grid point.

    B_{i,cert}(α) = Δ_α × ⌈C_i(α)/Δ_α⌉  capped at B_max(α).

    Returns B_cert of shape (n, n_alpha).
    """
    n, n_alpha = C_realized.shape
    Delta = grid["Delta"]   # (n_alpha,)
    Bmax  = grid["Bmax"]    # (n_alpha,)

    B_cert = np.zeros_like(C_realized)
    for j in range(n_alpha):
        d = Delta[j]
        if d <= 0:
            B_cert[:, j] = Bmax[j]
            continue
        # Ceiling-round to grid
        B_cert[:, j] = np.minimum(
            np.ceil(C_realized[:, j] / d) * d,
            Bmax[j])

    return B_cert


def certified_epsilon(B_cert: np.ndarray,      # (n, n_alpha)
                      alpha_grid: np.ndarray,   # (n_alpha,)
                      delta: float) -> tuple:
    """
    ε_{i,cert}(δ) = min_α [B_{i,cert}(α) + log(1/δ)/(α−1)]

    Returns (eps_cert shape (n,), best_alpha_idx shape (n,)).
    """
    log1delta = math.log(1.0 / delta)
    # candidates: (n, n_alpha)
    candidates = B_cert + log1delta / (alpha_grid[None, :] - 1.0)
    best_idx = np.argmin(candidates, axis=1)   # (n,)
    eps_cert = candidates[np.arange(len(best_idx)), best_idx]
    return eps_cert, best_idx


# ---------------------------------------------------------------------------
# Parallel rank-ablation worker (top-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

def _rank_worker(args):
    """
    Compute C_dir for one rank r, loading arrays via mmap to avoid pickling them.
    Returns (r, C_dir_r).
    Checkpoints every 100 steps so a killed worker resumes from the last boundary.
    """
    run_dir, cert_dir, tag, r, a, rho, sigma, q, T_use = args
    c_dir_r_ckpt = os.path.join(cert_dir, f"{tag}_C_realized_dir_rank_{r}.npy")
    if os.path.exists(c_dir_r_ckpt):
        print(f"  [cert] r={r}: loading completed result (cached)...")
        return r, np.load(c_dir_r_ckpt)

    # Per-rank partial checkpoint (prefix distinct per r so workers don't collide)
    partial_prefix = os.path.join(cert_dir, f"{tag}_rankwrk_{r}")
    partial_meta   = partial_prefix + "_partial_ckpt_meta.json"
    partial_cdir   = partial_prefix + "_partial_C_dir100.npy"

    start_t = 0
    C_dir_init = None
    fallback_total_init = 0
    mu_dir_max_init = mu_norm_max_init = 0.0
    if os.path.exists(partial_meta) and os.path.exists(partial_cdir):
        with open(partial_meta) as f:
            _pm = json.load(f)
        start_t             = _pm["t_completed"] + 1
        C_dir_init          = np.load(partial_cdir)
        fallback_total_init = int(_pm.get("fallback_total", 0))
        mu_dir_max_init     = float(_pm.get("mu_dir_max",   0.0))
        mu_norm_max_init    = float(_pm.get("mu_norm_max",  0.0))
        print(f"  [cert] r={r}: resuming from step {start_t}/{T_use} "
              f"(partial checkpoint found)...")

    # mmap_mode='r' lets workers share the on-disk arrays without copying them
    clipped_norms = np.load(os.path.join(run_dir, "clipped_norms.npy"), mmap_mode='r')[:, :T_use]
    B_matrices    = np.load(os.path.join(run_dir, "B_matrices.npy"),    mmap_mode='r')[:T_use]
    YTY_matrices  = np.load(os.path.join(run_dir, "YTY_matrices.npy"),  mmap_mode='r')[:T_use]
    Y_projections = np.load(os.path.join(run_dir, "Y_projections.npy"), mmap_mode='r')[:, :T_use, :]

    _, C_dir_r, _, _, _ = compute_realized_cumulative(
        clipped_norms, B_matrices, YTY_matrices, Y_projections,
        a, rho, sigma, q, r, ALPHA_GRID,
        start_t=start_t,
        C_dir_init=C_dir_init,
        fallback_total_init=fallback_total_init,
        mu_dir_max_init=mu_dir_max_init,
        mu_norm_max_init=mu_norm_max_init,
        ckpt_prefix=partial_prefix,
        ckpt_every=100)

    np.save(c_dir_r_ckpt, C_dir_r.astype(np.float64))
    print(f"  [cert] r={r}: completed, result saved.")
    # Clean up per-rank partial checkpoint files
    for _p in [partial_meta, partial_cdir,
               partial_prefix + "_partial_C_norm.npy"]:
        if os.path.exists(_p):
            os.remove(_p)
    return r, C_dir_r


# ---------------------------------------------------------------------------
# Full certification for one run
# ---------------------------------------------------------------------------

def certify_run(run_dir: str, cert_dir: str, ranks=None,
                delta: float = None, verbose: bool = True) -> dict:
    """
    Full Phase 19 certificate for a single training run.

    Steps:
      1. Load metadata and statistics.
      2. Commit budget grid and save filter_budget_grid.json.
      3. Compute realized cumulative charges for dir (at each rank) and norm.
      4. Filter replay → B_cert.
      5. Convert to certified ε.
      6. Save all output files per spec §8.5–8.6.
      7. Run sanity checks (spec §13).

    Returns summary dict.
    """
    if ranks is None:
        ranks = RANK_ABLATION

    meta_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"  [cert] metadata.json not found: {run_dir}"); return None
    with open(meta_path) as f:
        meta = json.load(f)

    if delta is None:
        delta = float(meta["delta"])
    q     = float(meta["q"])
    rho   = float(meta["rho"])
    a     = float(meta["a"])
    sigma = float(meta["sigma"])
    T     = int(meta["T_train"])
    run_id = meta.get("run_id", "?")
    seed   = meta.get("seed", 0)
    tag    = f"p19_{run_id}_seed{seed}"

    # Load statistics
    def _load(name):
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            print(f"  [cert] Missing: {name}"); return None
        return np.load(p)

    clipped_norms  = _load("clipped_norms.npy")     # (n, T)
    B_matrices     = _load("B_matrices.npy")         # (T, r_max, r_max)
    YTY_matrices   = _load("YTY_matrices.npy")       # (T, r_max, r_max)
    Y_projections  = _load("Y_projections.npy")      # (n, T, r_max)
    tier_labels    = _load("tier_labels.npy")        # (n,) or None

    if any(x is None for x in [clipped_norms, B_matrices, YTY_matrices, Y_projections]):
        return None

    n, T_logged = clipped_norms.shape
    T_use = min(T, T_logged)
    if T_use < T:
        print(f"  [cert] WARNING: expected T={T} steps, only {T_logged} logged. "
              f"Using {T_use} steps (conservative).")

    # Truncate to T_use
    clipped_norms  = clipped_norms[:, :T_use]
    B_matrices     = B_matrices[:T_use]
    YTY_matrices   = YTY_matrices[:T_use]
    Y_projections  = Y_projections[:, :T_use, :]

    # ===================================================================
    # STEP 1: Commit budget grid BEFORE any per-instance charge analysis
    # ===================================================================
    print(f"\n  [cert] Building budget grid (committed before charge analysis)...")
    grid = build_budget_grid(sigma, q, T_use, ALPHA_GRID)
    grid_path = os.path.join(cert_dir, f"{tag}_filter_budget_grid.json")
    save_budget_grid(grid, grid_path)

    # ===================================================================
    # STEP 2: Compute norm-based realized cumulative charges
    #   (norm doesn't depend on rank — compute once)
    # ===================================================================
    headline_rank  = 100
    ckpt_meta_path = os.path.join(cert_dir, f"{tag}_ckpt_meta.json")
    c_norm_ckpt    = os.path.join(cert_dir, f"{tag}_C_realized_norm.npy")
    c_dir100_ckpt  = os.path.join(cert_dir, f"{tag}_C_realized_dir_rank_{headline_rank}.npy")

    partial_meta_path = os.path.join(cert_dir, f"{tag}_partial_ckpt_meta.json")
    partial_norm_path = os.path.join(cert_dir, f"{tag}_partial_C_norm.npy")
    partial_dir_path  = os.path.join(cert_dir, f"{tag}_partial_C_dir100.npy")

    if (os.path.exists(c_norm_ckpt) and os.path.exists(c_dir100_ckpt)
            and os.path.exists(ckpt_meta_path)):
        print(f"  [cert] Resuming: loading cached C_norm + C_dir_r{headline_rank}...")
        C_norm    = np.load(c_norm_ckpt)
        C_dir_100 = np.load(c_dir100_ckpt)
        with open(ckpt_meta_path) as f:
            _cm = json.load(f)
        fallback_rate = _cm["fallback_rate"]
        mu_dir_max    = _cm["mu_dir_max"]
        mu_norm_max   = _cm["mu_norm_max"]
    else:
        # Check for a partial (mid-run) checkpoint from a previous killed run
        start_t = 0
        C_norm_init = C_dir_100_init = None
        fallback_total_init = 0
        mu_dir_max_init = mu_norm_max_init = 0.0
        if (os.path.exists(partial_meta_path) and os.path.exists(partial_norm_path)
                and os.path.exists(partial_dir_path)):
            with open(partial_meta_path) as f:
                _pm = json.load(f)
            start_t             = _pm["t_completed"] + 1
            C_norm_init         = np.load(partial_norm_path)
            C_dir_100_init      = np.load(partial_dir_path)
            fallback_total_init = int(_pm.get("fallback_total", 0))
            mu_dir_max_init     = float(_pm.get("mu_dir_max",   0.0))
            mu_norm_max_init    = float(_pm.get("mu_norm_max",  0.0))
            print(f"  [cert] Resuming from step {start_t}/{T_use} "
                  f"(partial checkpoint found)...")

        print(f"  [cert] Computing norm + headline rank r={headline_rank} "
              f"(T={T_use}, n={n}, start_t={start_t})...")
        C_norm, C_dir_100, fallback_rate, mu_dir_max, mu_norm_max = \
            compute_realized_cumulative(
                clipped_norms, B_matrices, YTY_matrices, Y_projections,
                a, rho, sigma, q, headline_rank, ALPHA_GRID,
                start_t=start_t,
                C_norm_init=C_norm_init, C_dir_init=C_dir_100_init,
                fallback_total_init=fallback_total_init,
                mu_dir_max_init=mu_dir_max_init,
                mu_norm_max_init=mu_norm_max_init,
                ckpt_prefix=os.path.join(cert_dir, tag),
                ckpt_every=100)
        np.save(c_norm_ckpt,   C_norm.astype(np.float64))
        np.save(c_dir100_ckpt, C_dir_100.astype(np.float64))
        with open(ckpt_meta_path, "w") as f:
            json.dump({"fallback_rate": float(fallback_rate),
                       "mu_dir_max":    float(mu_dir_max),
                       "mu_norm_max":   float(mu_norm_max)}, f, indent=2)
        print(f"  [cert] Checkpoint saved: C_norm + C_dir_r{headline_rank}")
        # Clean up partial checkpoint files now that final is written
        for _p in [partial_meta_path, partial_norm_path, partial_dir_path]:
            if os.path.exists(_p):
                os.remove(_p)

    if verbose:
        print(f"  [cert] fallback_rate={fallback_rate:.6%}  "
              f"mu_dir_max={mu_dir_max:.6f}  mu_norm_max={mu_norm_max:.6f}  "
              f"1/sigma={1/sigma:.6f}")

    # ===================================================================
    # STEP 3: Filter replay and certified ε for headline rank
    # ===================================================================
    B_cert_norm    = filter_replay(C_norm,     grid)    # (n, n_alpha)
    B_cert_dir_100 = filter_replay(C_dir_100,  grid)    # (n, n_alpha)

    eps_cert_norm,     best_alpha_norm     = certified_epsilon(B_cert_norm,    ALPHA_GRID, delta)
    eps_cert_dir_100,  best_alpha_dir_100  = certified_epsilon(B_cert_dir_100, ALPHA_GRID, delta)

    if verbose:
        # α* distribution — key floor-pinning diagnostic.
        # If α* is railed at the grid minimum for most examples while ε^dir is
        # constant, the log(1/δ)/(α-1) term is dominating; finer Δ or tighter
        # conversion is needed.
        _adir  = ALPHA_GRID[best_alpha_dir_100.astype(int)]
        _anorm = ALPHA_GRID[best_alpha_norm.astype(int)]
        _at_min_dir  = (best_alpha_dir_100  == 0).mean()
        _at_min_norm = (best_alpha_norm     == 0).mean()
        print(f"  [diag] α* (dir  r={headline_rank}): "
              f"p5={np.percentile(_adir,5):.1f}  med={np.median(_adir):.1f}  "
              f"p95={np.percentile(_adir,95):.1f}  at_grid_min={_at_min_dir:.1%}")
        print(f"  [diag] α* (norm):         "
              f"p5={np.percentile(_anorm,5):.1f}  med={np.median(_anorm):.1f}  "
              f"p95={np.percentile(_anorm,95):.1f}  at_grid_min={_at_min_norm:.1%}")

    # ===================================================================
    # STEP 4: Rank ablation (parallel across ranks)
    # ===================================================================
    C_dir_by_rank = {headline_rank: C_dir_100}

    other_ranks = [r for r in RANK_ABLATION if r != headline_rank]
    if other_ranks:
        n_workers = min(len(other_ranks), os.cpu_count() or 4)
        print(f"  [cert] Rank ablation {other_ranks} — {n_workers} parallel workers")
        worker_args = [
            (run_dir, cert_dir, tag, r, a, rho, sigma, q, T_use)
            for r in other_ranks
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_rank_worker, wa): wa[3] for wa in worker_args}
            for fut in as_completed(futures):
                r_done, C_dir_r = fut.result()
                C_dir_by_rank[r_done] = C_dir_r

    rank_results       = {}
    B_cert_dir_by_rank = {}
    for r in RANK_ABLATION:
        C_dir_r        = C_dir_by_rank[r]
        B_cert_r       = filter_replay(C_dir_r, grid)
        eps_cert_dir_r, best_alpha_r = certified_epsilon(B_cert_r, ALPHA_GRID, delta)
        B_cert_dir_by_rank[r] = B_cert_r

        eps_realized_dir_r, _ = certified_epsilon(C_dir_r, ALPHA_GRID, delta)
        overhead_r = B_cert_r - C_dir_r

        rank_results[r] = {
            "eps_cert_dir":     eps_cert_dir_r,
            "eps_realized_dir": eps_realized_dir_r,
            "B_cert":           B_cert_r,
            "C_realized":       C_dir_r,
            "overhead":         overhead_r,
            "best_alpha":       best_alpha_r,
        }

        if verbose:
            print(f"    r={r:3d}: ε^dir_cert med={np.median(eps_cert_dir_r):.4f}  "
                  f"ε^norm_cert med={np.median(eps_cert_norm):.4f}")

    # ===================================================================
    # STEP 5: Save output files (spec §8.5–8.6)
    # ===================================================================
    np.save(os.path.join(cert_dir, f"{tag}_Bcert_norm.npy"),
            B_cert_norm.astype(np.float64))
    np.save(os.path.join(cert_dir, f"{tag}_epsilon_cert_norm.npy"),
            eps_cert_norm.astype(np.float64))
    np.save(os.path.join(cert_dir, f"{tag}_best_alpha_cert_norm.npy"),
            best_alpha_norm.astype(np.int32))
    np.save(os.path.join(cert_dir, f"{tag}_C_realized_norm.npy"),
            C_norm.astype(np.float64))
    # Realized ε for norm
    eps_realized_norm, _ = certified_epsilon(C_norm, ALPHA_GRID, delta)
    np.save(os.path.join(cert_dir, f"{tag}_epsilon_realized_norm.npy"),
            eps_realized_norm.astype(np.float64))
    # Filter halt step: -1 for all (filter never halts under certified budget)
    halt_norm = np.full(n, -1, dtype=np.int32)
    np.save(os.path.join(cert_dir, f"{tag}_filter_halt_step_norm.npy"), halt_norm)

    for r in RANK_ABLATION:
        rr = rank_results[r]
        np.save(os.path.join(cert_dir, f"{tag}_Bcert_dir_rank_{r}.npy"),
                rr["B_cert"].astype(np.float64))
        np.save(os.path.join(cert_dir, f"{tag}_epsilon_cert_dir_rank_{r}.npy"),
                rr["eps_cert_dir"].astype(np.float64))
        np.save(os.path.join(cert_dir, f"{tag}_best_alpha_cert_dir_rank_{r}.npy"),
                rr["best_alpha"].astype(np.int32))
        np.save(os.path.join(cert_dir, f"{tag}_C_realized_dir_rank_{r}.npy"),
                rr["C_realized"].astype(np.float64))
        np.save(os.path.join(cert_dir, f"{tag}_epsilon_realized_dir_rank_{r}.npy"),
                rr["eps_realized_dir"].astype(np.float64))
        halt_dir = np.full(n, -1, dtype=np.int32)
        np.save(os.path.join(cert_dir, f"{tag}_filter_halt_step_dir_rank_{r}.npy"), halt_dir)

    # Per-step Ufull and d2hat for headline rank (diagnostic)
    ufull_ckpt    = os.path.join(cert_dir, f"{tag}_Ufull_rank_{headline_rank}.npy")
    dhat2_ckpt    = os.path.join(cert_dir, f"{tag}_dhat2_rank_{headline_rank}.npy")
    fallback_ckpt = os.path.join(cert_dir, f"{tag}_fallback_mask_rank_{headline_rank}.npy")
    if os.path.exists(ufull_ckpt) and os.path.exists(dhat2_ckpt) and os.path.exists(fallback_ckpt):
        print(f"  [cert] Resuming: loading cached Ufull/dhat2 diagnostics (r={headline_rank})...")
    else:
        Ufull_100    = np.zeros((n, T_use), dtype=np.float64)
        dhat2_100    = np.zeros((n, T_use), dtype=np.float64)
        fallback_100 = np.zeros((n, T_use), dtype=bool)
        for t in range(T_use):
            U_t, d2_t, fb_t = compute_U_full_and_d2hat(
                clipped_norms[:, t], B_matrices[t], YTY_matrices[t],
                Y_projections[:, t, :], a, rho, headline_rank)
            Ufull_100[:, t]    = U_t
            dhat2_100[:, t]    = d2_t
            fallback_100[:, t] = fb_t
        np.save(ufull_ckpt,    Ufull_100.astype(np.float32))
        np.save(dhat2_ckpt,    dhat2_100.astype(np.float32))
        np.save(fallback_ckpt, fallback_100.astype(bool))
        print(f"  [cert] Checkpoint saved: Ufull/dhat2 diagnostics (r={headline_rank})")

    if verbose:
        # d_hat shift distribution — confirms whether μ_dir has genuine per-example
        # spread or is uniformly near zero (which would floor-pin ε^dir regardless
        # of grid resolution).
        _dhat2 = np.load(dhat2_ckpt).astype(np.float64)
        _mu    = np.sqrt(np.maximum(_dhat2, 0.0)).mean(axis=1)   # (n,) mean over T
        _spread = np.percentile(_mu, 95) - np.percentile(_mu, 5)
        print(f"  [diag] μ_dir per-example (mean over T steps):")
        print(f"         p1={np.percentile(_mu,1):.6f}  p5={np.percentile(_mu,5):.6f}  "
              f"p25={np.percentile(_mu,25):.6f}  med={np.median(_mu):.6f}  "
              f"p75={np.percentile(_mu,75):.6f}  p95={np.percentile(_mu,95):.6f}  "
              f"max={_mu.max():.6f}")
        print(f"         spread(p95-p5)={_spread:.6f}  "
              f"frac_near_zero(μ<1e-4)={((_mu < 1e-4).mean()):.1%}")

    # ===================================================================
    # STEP 6: Sanity checks (spec §13.1–13.5)
    # ===================================================================
    ok = _sanity_checks(
        eps_cert_norm, eps_cert_dir_100, fallback_rate,
        mu_norm_max, mu_dir_max, sigma, a, q, T_use, delta, tag)

    # Sanity: ε_dir ≤ ε_norm
    viol = (eps_cert_dir_100 > eps_cert_norm + 1e-6).sum()
    if viol > 0:
        print(f"  [SANITY FAIL] {viol} examples have ε_dir_cert > ε_norm_cert")
        ok = False
    else:
        print(f"  [SANITY OK]   ε_dir_cert ≤ ε_norm_cert ✓  (tol=1e-6)")

    # Sanity: PSD checks for B_matrices and YTY_matrices
    # eigvalsh returns ascending order; [0] is the minimum eigenvalue
    min_B_eig = min(np.linalg.eigvalsh(B_matrices[t])[0] for t in range(0, T_use, max(1, T_use//10)))
    min_M_eig = min(np.linalg.eigvalsh(YTY_matrices[t])[0] for t in range(0, T_use, max(1, T_use//10)))
    tau_psd = 1e-5
    if min_B_eig < -tau_psd:
        print(f"  [SANITY FAIL] B_matrices min eigenvalue={min_B_eig:.2e} < −{tau_psd:.0e}")
        ok = False
    else:
        print(f"  [SANITY OK]   B_matrices PSD ✓  (min_eigval={min_B_eig:.2e})")
    if min_M_eig < -tau_psd:
        print(f"  [SANITY FAIL] YTY_matrices min eigenvalue={min_M_eig:.2e} < −{tau_psd:.0e}")
        ok = False
    else:
        print(f"  [SANITY OK]   YTY_matrices PSD ✓  (min_eigval={min_M_eig:.2e})")

    # ===================================================================
    # STEP 7: Summary statistics and tier breakdown
    # ===================================================================
    eps_norm = eps_cert_norm
    eps_dir  = eps_cert_dir_100

    if verbose:
        _print_summary(eps_norm, eps_dir, tier_labels, tag, meta)
        _print_rank_ablation(rank_results)
        _print_discretization_overhead(rank_results[headline_rank], tag)

    # Tier-wise stats
    tier_stats = {}
    if tier_labels is not None:
        for t_id, t_name in {0: "head", 1: "mid", 2: "tail"}.items():
            mask = (tier_labels == t_id)
            if mask.sum() == 0: continue
            tier_stats[t_name] = {
                "n":            int(mask.sum()),
                "eps_norm_mean": float(eps_norm[mask].mean()),
                "eps_dir_mean":  float(eps_dir[mask].mean()),
                "masking_ratio": float(1.0 - eps_dir[mask].mean() / max(eps_norm[mask].mean(), 1e-15)),
            }

    summary = {
        "run_id": run_id, "seed": seed,
        "regime": meta.get("regime"), "dataset": meta.get("dataset"),
        "epsilon_target": meta.get("epsilon_target"),
        "n": n, "T_acc": T_use,
        "sanity_ok": bool(ok),
        "fallback_rate": float(fallback_rate),
        "mu_norm_max": float(mu_norm_max),
        "mu_dir_max":  float(mu_dir_max),
        "sigma":       float(sigma),
        "mu_ceiling":  float(1.0/sigma),
        # Norm stats
        "eps_norm_cert_mean":   float(eps_norm.mean()),
        "eps_norm_cert_med":    float(np.median(eps_norm)),
        "eps_norm_cert_cv":     float(eps_norm.std() / max(eps_norm.mean(), 1e-15)),
        "eps_norm_cert_p5":     float(np.percentile(eps_norm, 5)),
        "eps_norm_cert_p95":    float(np.percentile(eps_norm, 95)),
        # Dir stats (r=100)
        "eps_dir_cert_mean":    float(eps_dir.mean()),
        "eps_dir_cert_med":     float(np.median(eps_dir)),
        "eps_dir_cert_cv":      float(eps_dir.std() / max(eps_dir.mean(), 1e-15)),
        "eps_dir_cert_p5":      float(np.percentile(eps_dir, 5)),
        "eps_dir_cert_p95":     float(np.percentile(eps_dir, 95)),
        # Masking
        "masking_gap_med":      float(np.median(eps_norm - eps_dir)),
        "masking_ratio_med":    float(np.median(1.0 - eps_dir / np.maximum(eps_norm, 1e-15))),
        # Rank ablation summary
        "rank_ablation": {
            str(r): {
                "eps_cert_dir_med": float(np.median(rank_results[r]["eps_cert_dir"])),
                "eps_cert_norm_med": float(np.median(eps_cert_norm)),
                "overhead_p95": float(np.percentile(
                    rank_results[r]["overhead"].sum(axis=1) if rank_results[r]["overhead"].ndim > 1
                    else rank_results[r]["overhead"], 95)),
            } for r in RANK_ABLATION
        },
        "tier_stats": tier_stats,
        "best_test_acc": meta.get("best_test_acc"),
        "grid_path": grid_path,
    }
    summ_path = os.path.join(cert_dir, f"{tag}_summary.json")
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [cert] Summary saved → {summ_path}")

    # Plots
    _plot_scatter(eps_norm, eps_dir, tier_labels, meta,
                  os.path.join(cert_dir, f"{tag}_scatter.png"))
    _plot_hist(eps_norm, eps_dir, tier_labels, meta,
               os.path.join(cert_dir, f"{tag}_hist.png"))
    _plot_rank_ablation_fig(rank_results, eps_cert_norm,
                            os.path.join(cert_dir, f"{tag}_rank_ablation.png"), run_id, seed)

    return summary


# ---------------------------------------------------------------------------
# Sanity checks (spec §13)
# ---------------------------------------------------------------------------

def _sanity_checks(eps_norm, eps_dir, fallback_rate,
                   mu_norm_max, mu_dir_max, sigma, a, q, T, delta, tag):
    ok = True
    tau = 1e-5  # float32-heavy pipeline tolerance

    # Grid ceiling condition (spec §13.1)
    mu_ceiling = 1.0 / sigma
    if mu_norm_max > mu_ceiling + tau:
        print(f"  [SANITY FAIL] {tag}: mu_norm_max={mu_norm_max:.6f} > 1/σ+τ={mu_ceiling+tau:.6f}")
        ok = False
    else:
        print(f"  [SANITY OK]   {tag}: mu_norm_max={mu_norm_max:.6f} ≤ 1/σ+τ ✓")

    if mu_dir_max > mu_ceiling + tau:
        print(f"  [SANITY FAIL] {tag}: mu_dir_max={mu_dir_max:.6f} > 1/σ+τ={mu_ceiling+tau:.6f}")
        ok = False
    else:
        print(f"  [SANITY OK]   {tag}: mu_dir_max={mu_dir_max:.6f} ≤ 1/σ+τ ✓")

    # Fallback rate (spec §13.1)
    if fallback_rate > 0.01:
        print(f"  [SANITY WARN] {tag}: fallback_rate={fallback_rate:.4%} (expected ~0)")
    else:
        print(f"  [SANITY OK]   {tag}: fallback_rate={fallback_rate:.6%} ✓")

    return ok


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_summary(eps_norm, eps_dir, tier_labels, tag, meta):
    n = len(eps_norm)
    ratio = eps_dir / np.maximum(eps_norm, 1e-15)
    masking = 1.0 - ratio
    print(f"\n{'='*72}")
    print(f"  {tag}  {meta.get('regime')} {meta.get('dataset')}  ε={meta.get('epsilon_target')}")
    print(f"  n={n}  T_acc={meta.get('T_train')}  r=100  δ={meta.get('delta')}")
    print(f"{'='*72}")
    print(f"  ε^norm cert: med={np.median(eps_norm):.4f}  CV={eps_norm.std()/eps_norm.mean():.3f}"
          f"  p5={np.percentile(eps_norm,5):.4f}  p95={np.percentile(eps_norm,95):.4f}")
    print(f"  ε^dir  cert: med={np.median(eps_dir):.4f}  CV={eps_dir.std()/eps_dir.mean():.3f}"
          f"  p5={np.percentile(eps_dir,5):.4f}  p95={np.percentile(eps_dir,95):.4f}")
    print(f"  masking gap: med={np.median(eps_norm-eps_dir):.4f}  "
          f"ratio med={np.median(masking)*100:.1f}%  mean={masking.mean()*100:.1f}%")
    if tier_labels is not None:
        print(f"\n  Tier breakdown:")
        for t_id, t_name in {0: "head", 1: "mid", 2: "tail"}.items():
            mask = (tier_labels == t_id)
            if mask.sum() == 0: continue
            print(f"    {t_name:4s} n={mask.sum():5d}  "
                  f"ε^norm={eps_norm[mask].mean():.4f}  "
                  f"ε^dir={eps_dir[mask].mean():.4f}  "
                  f"masking={100*(1-eps_dir[mask].mean()/max(eps_norm[mask].mean(),1e-15)):.1f}%")


def _print_rank_ablation(rank_results):
    print(f"\n  Rank ablation:")
    print(f"  {'rank':>5}  {'ε^dir med':>10}  {'ε^dir p5':>8}  {'ε^dir p95':>9}  {'overhead p95':>12}")
    for r in sorted(rank_results.keys()):
        rr = rank_results[r]
        ed  = rr["eps_cert_dir"]
        ovh = rr["overhead"]
        ovh_p95 = float(np.percentile(ovh.flatten(), 95))
        print(f"  {r:5d}  {np.median(ed):10.4f}  "
              f"{np.percentile(ed,5):8.4f}  "
              f"{np.percentile(ed,95):9.4f}  "
              f"{ovh_p95:12.6f}")


def _print_discretization_overhead(rr_100, tag):
    ovh = rr_100["overhead"]   # (n, n_alpha)
    # Use best-alpha overhead per example
    best_idx = np.argmin(rr_100["B_cert"], axis=1)
    ovh_best = np.array([ovh[i, best_idx[i]] for i in range(len(best_idx))])
    certified = np.array([rr_100["B_cert"][i, best_idx[i]] for i in range(len(best_idx))])
    pct = 100.0 * ovh_best / np.maximum(certified, 1e-15)
    print(f"\n  Filter discretization overhead (r=100, at best α):")
    print(f"    median overhead:   {np.median(ovh_best):.6f}")
    print(f"    p95 overhead:      {np.percentile(ovh_best,95):.6f}")
    print(f"    max overhead:      {ovh_best.max():.6f}")
    print(f"    median overhead %: {np.median(pct):.3f}%")
    print(f"    p95 overhead %:    {np.percentile(pct,95):.3f}%")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plt():
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _plot_scatter(eps_norm, eps_dir, tier_labels, meta, path):
    plt = _plt()
    if plt is None: return
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = tier_labels.astype(float) if tier_labels is not None else np.zeros(len(eps_norm))
    sc = ax.scatter(eps_norm, eps_dir, c=colors, cmap="RdYlGn_r", s=4, alpha=0.4)
    mn = min(eps_norm.min(), eps_dir.min()); mx = max(eps_norm.max(), eps_dir.max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, label="y=x")
    ax.set_xlabel("ε^norm certified"); ax.set_ylabel("ε^dir certified (r=100)")
    ax.set_title(f"{meta.get('run_id','?')} s{meta.get('seed','?')}  ε={meta.get('epsilon_target')}")
    if tier_labels is not None: plt.colorbar(sc, ax=ax, label="tier")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [plot] {path}")


def _plot_hist(eps_norm, eps_dir, tier_labels, meta, path):
    plt = _plt()
    if plt is None: return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    tier_colors = {0: "steelblue", 1: "orange", 2: "firebrick"}
    tier_names  = {0: "head", 1: "mid", 2: "tail"}
    if tier_labels is not None:
        for t, tn in tier_names.items():
            mask = (tier_labels == t)
            if mask.sum() == 0: continue
            axes[0].hist(eps_norm[mask], bins=40, alpha=0.6, label=tn, color=tier_colors[t])
            axes[1].hist(eps_dir[mask],  bins=40, alpha=0.6, label=tn, color=tier_colors[t])
    else:
        axes[0].hist(eps_norm, bins=50, alpha=0.7, color="steelblue")
        axes[1].hist(eps_dir,  bins=50, alpha=0.7, color="darkorange")
    for ax, lbl in zip(axes, ["ε^norm cert", "ε^dir cert (r=100)"]):
        ax.axvline(meta.get("epsilon_target", 8), color="k", ls="--", lw=1)
        ax.set_xlabel(lbl); ax.set_ylabel("count"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [plot] {path}")


def _plot_rank_ablation_fig(rank_results, eps_cert_norm, path, run_id, seed):
    plt = _plt()
    if plt is None: return
    ranks  = sorted(rank_results.keys())
    med_dir  = [float(np.median(rank_results[r]["eps_cert_dir"])) for r in ranks]
    med_norm = float(np.median(eps_cert_norm))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranks, med_dir, "o-", color="darkorange", label="ε^dir cert")
    ax.axhline(med_norm, ls="--", color="steelblue", label=f"ε^norm cert = {med_norm:.4f}")
    ax.set_xlabel("Rank r"); ax.set_ylabel("Median certified ε")
    ax.set_title(f"{run_id} s{seed}: rank ablation"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [plot] {path}")


# ---------------------------------------------------------------------------
# Seed aggregation and summary table
# ---------------------------------------------------------------------------

def aggregate_seeds(run_id, cert_dir, n_seeds):
    summaries = []
    for seed in range(n_seeds):
        p = os.path.join(cert_dir, f"p19_{run_id}_seed{seed}_summary.json")
        if os.path.exists(p):
            with open(p) as f: summaries.append(json.load(f))
    if not summaries: return None
    agg = {"n_seeds": len(summaries)}
    keys = ["eps_norm_cert_med", "eps_dir_cert_med", "eps_norm_cert_cv",
            "eps_dir_cert_cv", "masking_gap_med", "masking_ratio_med",
            "best_test_acc", "fallback_rate"]
    for k in keys:
        vals = [s[k] for s in summaries if k in s and s[k] is not None]
        if vals:
            agg[f"{k}_median"] = float(np.median(vals))
            agg[f"{k}_q25"]    = float(np.percentile(vals, 25))
            agg[f"{k}_q75"]    = float(np.percentile(vals, 75))
    return agg


SEEDS_PER_RUN = {"F1": 3, "F2": 3, "F5": 3, "F6": 3, "F7": 3, "F8": 3, "F9": 3}
ALL_RUNS      = list(SEEDS_PER_RUN.keys())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 19 certificate computation")
    parser.add_argument("--run",       type=str, default=None, choices=ALL_RUNS)
    parser.add_argument("--seed",      type=int, default=None)
    parser.add_argument("--all_seeds", action="store_true")
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--table",     action="store_true")
    parser.add_argument("--runs_dir",  type=str, default=RUNS_DIR)
    parser.add_argument("--cert_dir",  type=str, default=CERT_DIR)
    args = parser.parse_args()

    os.makedirs(args.cert_dir, exist_ok=True)

    if args.run and not args.all:
        runs_to_run = [args.run]
    elif args.all or args.table:
        runs_to_run = ALL_RUNS
    else:
        print("[P19-cert] Specify --run F1/F2/F5, --all, or --table.")
        return

    all_agg = {}
    for rid in runs_to_run:
        n_seeds = SEEDS_PER_RUN.get(rid, 1)
        if args.seed is not None:
            seeds = [args.seed]
        elif args.all_seeds:
            seeds = list(range(n_seeds))
        else:
            seeds = [0]

        for seed in seeds:
            run_dir = os.path.join(args.runs_dir, rid, f"seed_{seed}")
            if not os.path.isdir(run_dir):
                print(f"[P19-cert] {rid}/seed_{seed}: run dir not found: {run_dir}")
                continue
            print(f"\n[P19-cert] === {rid} seed={seed} ===")
            _t_cert_start = time.time()
            certify_run(run_dir, args.cert_dir, verbose=True)
            _t_cert_elapsed = time.time() - _t_cert_start
            try:
                import resource as _res
                _peak_mb = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1024.0
                _mem_str = f"  peak_mem={_peak_mb:.0f} MB"
            except Exception:
                _mem_str = ""
            print(f"\n  [B2 certify timing] {rid}/seed_{seed}: "
                  f"wall_clock={_t_cert_elapsed/60:.1f} min ({_t_cert_elapsed:.0f}s){_mem_str}")
            _timing_path = os.path.join(args.cert_dir, f"p19_{rid}_seed{seed}_certify_timing.json")
            with open(_timing_path, "w") as _f:
                json.dump({"run_id": rid, "seed": seed,
                           "certify_wall_s": float(_t_cert_elapsed),
                           "certify_wall_min": float(_t_cert_elapsed / 60.0)}, _f, indent=2)

        agg = aggregate_seeds(rid, args.cert_dir, n_seeds)
        all_agg[rid] = agg
        if agg:
            agg_path = os.path.join(args.cert_dir, f"p19_{rid}_aggregate.json")
            with open(agg_path, "w") as f:
                json.dump({"run_id": rid, **agg}, f, indent=2)

    if args.table or len(all_agg) > 1:
        print(f"\n{'='*80}\n  Phase 19 Certificate Summary\n{'='*80}")
        hdr = f"  {'Run':5s}  {'Regime':6s}  {'ε^norm med':10s}  {'ε^dir med':9s}  " \
              f"{'masking%':8s}  {'CV norm':7s}  {'seeds':5s}  {'acc':6s}"
        print(hdr)
        for rid, agg in sorted(all_agg.items()):
            if agg is None: continue
            def _g(k): return f"{agg.get(k+'_median',float('nan')):.4f}"
            mask_pct = f"{agg.get('masking_ratio_med_median',0)*100:.1f}%"
            print(f"  {rid:5s}  {'R3' if rid in ['F1','F2'] else 'R2':6s}  "
                  f"{_g('eps_norm_cert_med'):10s}  {_g('eps_dir_cert_med'):9s}  "
                  f"{mask_pct:8s}  {_g('eps_norm_cert_cv'):7s}  "
                  f"{agg.get('n_seeds',1):5d}  {_g('best_test_acc'):6s}")

    print(f"\n[P19-cert] Done. Results in {args.cert_dir}")


if __name__ == "__main__":
    main()
