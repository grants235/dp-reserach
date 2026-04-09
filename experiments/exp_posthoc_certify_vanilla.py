#!/usr/bin/env python3
"""
Phase 14 — Experiment B: Post-Hoc Direction-Aware Certificates on Standard DP-SGD
==================================================================================

Loads per-step gradient logs from Experiment A and computes per-instance privacy
certificates using the ACTUAL Opacus subsampled Gaussian mechanism RDP formula
(Mironov 2019 Theorem 4, with Poisson subsampling amplification) — not the
quadratic approximation used in Phase 13.

Two certificates per example:

  eps_norm(i)       — Thudi et al. per-instance RDP, standard adversary
                      ε_i^norm(α) = Σ_{t: i∈B_t} ε_step(α, ||ḡ_it||/C, σ_std)
                      where ε_step uses the Opacus subsampled Gaussian RDP formula

  eps_direction(i)  — direction-aware, aggregate-uncertain adversary
                      ε_i^dir(α)  = Σ_{t: i∈B_t} ε_step(α, ||ḡ_it^V⊥||/C, σ_std)

Both converted to (ε,δ)-DP: ε_i = min_{α>1} [ε_i(α) + log(1/δ)/(α-1)]

Four sanity checks before reporting results:
  Check 1: data-independent (worst-case) ε = 2.0 from Opacus accountant
  Check 2: all eps_norm ≤ 2.0 (guaranteed since ||ḡ|| ≤ C)
  Check 3: distribution of eps_norm clusters near (but below) 2.0
  Check 4: eps_direction ≤ eps_norm for all i

Expected: eps_direction ≈ 0.5 × eps_norm (from β_mean ≈ 0.5 at rank 100).
This gives ~2× tightening of per-instance certificates under standard DP-SGD.

Usage (CPU only, fast)
-----
  python experiments/exp_posthoc_certify_vanilla.py --arm vanilla_warm_log
  python experiments/exp_posthoc_certify_vanilla.py  # all arms
"""

import os
import sys
import csv
import argparse
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Constants — must match exp_posthoc_vanilla.py exactly
# ---------------------------------------------------------------------------

DELTA       = 1e-5
EPS         = 2.0
CLIP_C      = 1.0
C_PERP_FRAC = 0.4
BATCH_SIZE  = 1000
EPOCHS      = 60
ARMS        = ["vanilla_warm_log", "gep_log", "pda_dpmd_log"]

RESULTS_DIR = "./results/exp_p14"
LOG_DIR     = os.path.join(RESULTS_DIR, "logs")
EXP_DIR     = os.path.join(RESULTS_DIR, "exp1")
CERT_DIR    = os.path.join(RESULTS_DIR, "exp2")

# RDP alpha grid for optimization.
# Dense near the optimal (~10 for eps=2, q=0.022, T=2700); extended range for robustness.
# Matches the Opacus DEFAULT_ALPHAS range for compatibility.
ALPHA_GRID = np.concatenate([
    np.arange(1.1,   2.0,   0.1),
    np.arange(2.0,  10.0,   0.5),
    np.arange(10.0,  64.0,  1.0),
    np.arange(64.0, 200.0, 10.0),
    np.arange(200.0, 1001., 50.0),
])

# Lookup table resolution for sensitivity ratio s = Δ/C
N_BINS = 1000   # number of sensitivity ratio bins in [0, 1]


# ---------------------------------------------------------------------------
# Privacy calibration
# ---------------------------------------------------------------------------

def _calibrate_sigma(eps, delta, q, steps):
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(
        target_epsilon=eps, target_delta=delta,
        sample_rate=q, steps=steps, accountant="rdp")


def _get_sigma_for_arm(arm_name, eps, delta, n_train, batch_size, epochs):
    """Return (sigma, q, T_steps) for each arm.

    All arms use single-channel Gaussian mechanism with standard sigma_std.
    GEP sensitivity is ||g_V|| ≤ ||g|| ≤ C, so same calibration as vanilla.
    """
    steps_per_epoch = n_train // batch_size
    T_steps         = epochs * steps_per_epoch
    q               = batch_size / n_train
    sigma = _calibrate_sigma(eps, delta, q, T_steps)
    return sigma, q, T_steps


# ---------------------------------------------------------------------------
# Core: per-step subsampled Gaussian RDP via Opacus
# ---------------------------------------------------------------------------

def _opacus_rdp_step(q, noise_multiplier, orders):
    """
    Per-step RDP of Poisson-subsampled Gaussian mechanism at multiple orders.
    Uses Opacus internal compute_rdp (Mironov 2019 Theorem 4 + subsampling).
    Falls back to Gaussian-only (no subsampling, conservative) if unavailable.

    Returns ndarray of shape [len(orders)].
    """
    orders_list = list(orders) if not np.isscalar(orders) else [float(orders)]
    try:
        # Opacus >= 1.4: opacus.accountants.analysis.rdp
        from opacus.accountants.analysis.rdp import compute_rdp
        result = compute_rdp(q=q, noise_multiplier=noise_multiplier,
                             steps=1, orders=orders_list)
        return np.asarray(result, dtype=np.float64)
    except (ImportError, AttributeError, TypeError):
        pass
    try:
        # Older Opacus: opacus.accountants.rdp_accountant
        from opacus.accountants.rdp_accountant import compute_rdp  # type: ignore
        result = compute_rdp(q=q, noise_multiplier=noise_multiplier,
                             steps=1, orders=orders_list)
        return np.asarray(result, dtype=np.float64)
    except (ImportError, AttributeError, TypeError):
        # Fallback: Gaussian mechanism without subsampling (conservative upper bound)
        alphas = np.asarray(orders_list, dtype=np.float64)
        return alphas / (2.0 * noise_multiplier ** 2)


def _build_rdp_lookup(q, sigma, alpha_grid, n_bins=N_BINS):
    """
    Build a lookup table for per-step subsampled Gaussian RDP.

    For sensitivity ratio s = Δ/C ∈ [0, 1], the effective noise multiplier
    w.r.t. sensitivity Δ is σ/s.  We precompute:
        rdp_table[k, j] = ε_step(α_j, s_grid[k], σ)
    using the Opacus subsampled Gaussian RDP formula.

    s_grid[0] = 0 is handled separately (zero sensitivity → zero RDP).
    s_grid[1:] uses geomspace from s_min to 1.0.

    Returns: (s_grid, rdp_table) where
        s_grid:    [n_bins]  float64, sensitivity ratios
        rdp_table: [n_bins, len(alpha_grid)]  float64, per-step RDP values
    """
    s_min    = 1e-4
    s_grid   = np.zeros(n_bins, dtype=np.float64)
    s_grid[1:] = np.geomspace(s_min, 1.0, n_bins - 1)

    n_alpha  = len(alpha_grid)
    table    = np.zeros((n_bins, n_alpha), dtype=np.float64)
    # s=0 → infinite noise multiplier → zero RDP (table[0] stays 0)

    print(f"[P14-B] Building RDP lookup table (q={q:.5f}, σ={sigma:.4f}, "
          f"{n_bins} bins × {n_alpha} α)...")
    t0 = time.time()
    for k in range(1, n_bins):
        s      = s_grid[k]
        nm     = sigma / s     # effective noise multiplier for sensitivity Δ = s*C
        table[k] = _opacus_rdp_step(q, nm, alpha_grid)
    elapsed = time.time() - t0
    print(f"[P14-B] Lookup table built in {elapsed:.1f}s")
    return s_grid, table


def _accumulate_rdp_from_log(log_df, clip_c, s_grid, rdp_table, max_examples):
    """
    Accumulate per-example per-step RDP from the gradient log.

    For each log row (step t, example i, sensitivity s = grad_norm/C):
        rdp_sum[i] += rdp_table[bin(s)]   (shape [n_alpha])

    Returns:
        rdp_sum: [max_examples, n_alpha]  (indexed by local example index)
        unique_global_idx: sorted array of global example indices
        n_sampled: [max_examples] int, times each example appeared
    """
    n_alpha = rdp_table.shape[1]

    gn   = log_df["grad_norm"].to_numpy(np.float32)
    idx  = log_df["example_idx"].to_numpy(np.int32)

    # Map global indices to contiguous local indices
    unique_idx   = np.array(sorted(np.unique(idx)), dtype=np.int32)
    n_present    = len(unique_idx)
    g2l          = {g: l for l, g in enumerate(unique_idx)}
    local_idx    = np.array([g2l[g] for g in idx], dtype=np.int32)

    rdp_sum  = np.zeros((n_present, n_alpha), dtype=np.float64)
    n_sampled = np.zeros(n_present, dtype=np.int32)

    # Sensitivity ratio and bin lookup
    s_vals     = (gn / clip_c).clip(s_grid[1], 1.0)
    bin_idx    = np.searchsorted(s_grid[1:], s_vals, side="right")
    # bin_idx ∈ [0, n_bins-1]; maps into s_grid[1:] so offset by 1 for table
    bin_idx    = np.minimum(bin_idx, len(s_grid) - 2)
    table_idx  = bin_idx + 1    # since s_grid[0]=0 is the sentinel

    # Count appearances
    np.add.at(n_sampled, local_idx, 1)

    # Accumulate RDP per alpha level (loop over alpha, vectorize over rows)
    for j in range(n_alpha):
        rdp_j = rdp_table[table_idx, j]   # [N_rows]
        np.add.at(rdp_sum[:, j], local_idx, rdp_j)

    return rdp_sum, unique_idx, n_sampled


def _accumulate_rdp_from_log_inorm(log_df, clip_c, s_grid, rdp_table):
    """Same as above but using incoherent_norm/C as the sensitivity ratio."""
    n_alpha = rdp_table.shape[1]

    inn  = log_df["incoherent_norm"].to_numpy(np.float32)
    idx  = log_df["example_idx"].to_numpy(np.int32)

    unique_idx = np.array(sorted(np.unique(idx)), dtype=np.int32)
    n_present  = len(unique_idx)
    g2l        = {g: l for l, g in enumerate(unique_idx)}
    local_idx  = np.array([g2l[g] for g in idx], dtype=np.int32)

    rdp_sum = np.zeros((n_present, n_alpha), dtype=np.float64)

    s_vals    = (inn / clip_c).clip(0.0, 1.0)
    # For incoherent norm = 0, bin goes to 0 (zero RDP)
    nonzero   = s_vals > s_grid[1]
    bin_idx   = np.zeros(len(s_vals), dtype=np.int32)
    if nonzero.any():
        bin_idx[nonzero] = (np.searchsorted(s_grid[1:], s_vals[nonzero],
                                             side="right") + 1)
        bin_idx[nonzero] = np.minimum(bin_idx[nonzero], len(s_grid) - 1)

    for j in range(n_alpha):
        rdp_j = rdp_table[bin_idx, j]
        np.add.at(rdp_sum[:, j], local_idx, rdp_j)

    return rdp_sum


# ---------------------------------------------------------------------------
# RDP to (ε,δ)-DP conversion
# ---------------------------------------------------------------------------

def _rdp_to_dp(rdp_per_example, alpha_grid, delta):
    """
    Convert per-example RDP to (ε,δ)-DP for each example.

    Uses the Balle et al. 2020 (Theorem 21) conversion, matching the formula
    in Opacus get_privacy_spent:
        ε_i = min_α [ rdp(α) - (log(δ) + log(α))/(α-1) + log((α-1)/α) ]

    This is tighter than the simple min_α[rdp + log(1/δ)/(α-1)] and reproduces
    the same eps as get_privacy_spent for the sanity check.

    rdp_per_example: [n_examples, n_alpha]  or  [n_alpha] for a single example
    Returns: eps_arr [n_examples], best_alpha [n_examples]
    """
    scalar = rdp_per_example.ndim == 1
    if scalar:
        rdp_per_example = rdp_per_example[np.newaxis]

    log_delta = np.log(delta)
    log_alpha = np.log(alpha_grid)
    log_am1   = np.log(alpha_grid - 1.0)   # log(α-1); defined for α > 1

    # Balle et al. Theorem 21 (vectorized over examples × alpha values)
    # candidates[i, j] = rdp[i,j] - (log(δ)+log(α_j))/(α_j-1) + log((α_j-1)/α_j)
    candidates = (rdp_per_example
                  - (log_delta + log_alpha) / (alpha_grid - 1.0)
                  + (log_am1   - log_alpha))   # log((α-1)/α)

    best_j     = np.nanargmin(candidates, axis=1)
    eps_arr    = candidates[np.arange(len(best_j)), best_j]
    alpha_best = alpha_grid[best_j]

    if scalar:
        return eps_arr[0], alpha_best[0]
    return eps_arr, alpha_best


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _sanity_check_data_independent(arm_name, q, sigma, T_steps, eps_target,
                                    alpha_grid, delta):
    """
    Check 1: data-independent RDP accounting should reproduce eps_target=2.0.
    Uses Opacus compute_rdp with worst-case sensitivity (noise_multiplier=sigma)
    and get_privacy_spent for the Balle et al. conversion.
    """
    try:
        try:
            from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
        except ImportError:
            from opacus.accountants.rdp_accountant import compute_rdp, get_privacy_spent  # type: ignore
        rdp_total = list(compute_rdp(q=q, noise_multiplier=sigma, steps=T_steps,
                                     orders=list(alpha_grid)))
        eps_check, best_alpha = get_privacy_spent(
            orders=list(alpha_grid), rdp=rdp_total, delta=delta)
        status = "PASS" if eps_check <= eps_target + 0.05 else "WARN"
        print(f"  [Check 1] Data-independent ε = {eps_check:.4f} "
              f"(target≤{eps_target}, best α={best_alpha:.1f}) — {status}")
        return float(eps_check)
    except (ImportError, AttributeError) as e:
        print(f"  [Check 1] Skipped (Opacus unavailable): {e}")
        return None


def _sanity_checks(arm_name, eps_norm, eps_direction, n_sampled,
                   eps_target, q, sigma, T_steps, alpha_grid, delta):
    """Run all 4 sanity checks and print results."""
    print(f"\n  Sanity Checks — {arm_name}")
    print(f"  {'─'*50}")

    # Check 1: data-independent ε ≈ eps_target
    _sanity_check_data_independent(arm_name, q, sigma, T_steps,
                                   eps_target, alpha_grid, delta)

    # Check 2: all eps_norm ≤ eps_target
    max_en = eps_norm.max()
    n_viol = (eps_norm > eps_target).sum()
    status = "PASS" if n_viol == 0 else f"FAIL ({n_viol} violations)"
    print(f"  [Check 2] max eps_norm = {max_en:.4f} ≤ {eps_target} — {status}")

    # Check 3: distribution clusters near eps_target
    frac_near = (eps_norm >= 0.9 * eps_target).mean()
    print(f"  [Check 3] {frac_near*100:.1f}% of examples have "
          f"eps_norm ≥ 0.9×{eps_target} = {0.9*eps_target:.2f} "
          f"(expect ~majority if norms ≈ C)")

    # Check 4: eps_direction ≤ eps_norm for all i
    n_viol4 = (eps_direction > eps_norm + 1e-9).sum()
    status4 = "PASS" if n_viol4 == 0 else f"FAIL ({n_viol4} violations)"
    print(f"  [Check 4] eps_direction ≤ eps_norm — {status4}")


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def _print_summary(arm_name, certs, eps, delta, sigma, q, T_steps):
    en  = certs["eps_norm"]
    ed  = certs["eps_direction"]
    ns  = certs["n_sampled"]
    bm  = certs["beta_mean"]
    rat = certs["tightening_ratio"]

    print(f"\n{'='*70}")
    print(f" Post-Hoc Certificate Summary — {arm_name}  (ε={eps}, δ={delta})")
    print(f"  σ={sigma:.4f}, q={q:.5f}, T={T_steps}")
    print(f"{'='*70}")

    print(f"\n  Norm-based certificate (standard adversary):")
    print(f"    max:    {en.max():.4f}  {'✓ PASS (≤2)' if en.max() <= eps else '✗ FAIL'}")
    print(f"    99th:   {np.percentile(en, 99):.4f}")
    print(f"    95th:   {np.percentile(en, 95):.4f}")
    print(f"    mean:   {en.mean():.4f}")
    print(f"    ≤ eps:  {(en <= eps).mean()*100:.1f}%")

    print(f"\n  Direction-aware certificate (aggregate-uncertain adversary):")
    print(f"    max:    {ed.max():.4f}")
    print(f"    99th:   {np.percentile(ed, 99):.4f}")
    print(f"    95th:   {np.percentile(ed, 95):.4f}")
    print(f"    mean:   {ed.mean():.4f}")
    print(f"    ≤ eps:  {(ed <= eps).mean()*100:.1f}%")

    print(f"\n  Tightening ratio eps_direction / eps_norm:")
    print(f"    mean:   {rat.mean():.4f}  (expected ≈ β_mean ≈ 0.5)")
    print(f"    median: {np.median(rat):.4f}")
    print(f"    2× tightening: {(rat < 0.55).mean()*100:.1f}% of examples")

    print(f"\n  β_mean (incoherent fraction):")
    print(f"    mean:   {bm.mean():.4f}")
    print(f"    median: {np.median(bm):.4f}")

    print(f"\n  Sampling statistics:")
    print(f"    mean sampled: {ns.mean():.1f}  max: {ns.max()}")


# ---------------------------------------------------------------------------
# Main per-arm processing
# ---------------------------------------------------------------------------

def _run_arm(arm_name, eps, seed, log_dir, cert_dir, delta=DELTA):
    tag      = f"{arm_name}_eps{eps:.0f}_seed{seed}"
    log_path = os.path.join(log_dir, f"{tag}.parquet")
    npz_path = os.path.join(log_dir, f"{tag}.npz")
    out_path = os.path.join(cert_dir, f"{tag}_certificates.csv")

    if os.path.exists(out_path):
        print(f"[P14-B] {tag}: certificates already computed.")
        return _load_certificates(out_path)

    # --- Load gradient log ---
    log_df = None
    if os.path.exists(log_path):
        try:
            import pandas as pd
            log_df = pd.read_parquet(log_path)
        except ImportError:
            pass
    if log_df is None and os.path.exists(npz_path):
        import pandas as pd
        data   = np.load(npz_path)
        log_df = pd.DataFrame({
            "step":            data["step"],
            "example_idx":    data["example_idx"],
            "grad_norm":      data["grad_norm"],
            "incoherent_norm": data["incoherent_norm"],
        })
    if log_df is None:
        print(f"[P14-B] Log not found: {log_path}. "
              "Run exp_posthoc_vanilla.py first.")
        return None

    n_rows      = len(log_df)
    n_train_est = int(log_df["example_idx"].nunique())
    print(f"\n[P14-B] {tag}: {n_rows:,} log rows, {n_train_est} unique examples")

    # --- Privacy parameters ---
    sigma, q, T_steps = _get_sigma_for_arm(
        arm_name, eps, delta, n_train_est, BATCH_SIZE, EPOCHS)
    print(f"[P14-B] sigma={sigma:.4f}, q={q:.5f}, T={T_steps}")

    # --- Build RDP lookup table ---
    s_grid, rdp_table = _build_rdp_lookup(q, sigma, ALPHA_GRID, N_BINS)

    # --- Accumulate per-example RDP (norm-based) ---
    print(f"[P14-B] Accumulating norm-based RDP over {n_rows:,} log rows...")
    t0 = time.time()
    rdp_norm, unique_idx, n_sampled = _accumulate_rdp_from_log(
        log_df, CLIP_C, s_grid, rdp_table, n_train_est)
    print(f"[P14-B] Norm RDP accumulated in {time.time()-t0:.1f}s")

    # --- Accumulate per-example RDP (direction-aware) ---
    print(f"[P14-B] Accumulating direction-aware RDP over {n_rows:,} log rows...")
    t0 = time.time()
    rdp_dir = _accumulate_rdp_from_log_inorm(
        log_df, CLIP_C, s_grid, rdp_table)
    print(f"[P14-B] Direction RDP accumulated in {time.time()-t0:.1f}s")

    # --- Convert to (ε,δ)-DP ---
    print(f"[P14-B] Converting RDP to (ε,δ)-DP for {len(unique_idx)} examples...")
    eps_norm,      _ = _rdp_to_dp(rdp_norm, ALPHA_GRID, delta)
    eps_direction, _ = _rdp_to_dp(rdp_dir,  ALPHA_GRID, delta)

    # --- Derived statistics ---
    sum_gn2 = np.zeros(len(unique_idx), dtype=np.float64)
    sum_in2 = np.zeros(len(unique_idx), dtype=np.float64)
    gn  = log_df["grad_norm"].to_numpy(np.float64)
    inn = log_df["incoherent_norm"].to_numpy(np.float64)
    idx = log_df["example_idx"].to_numpy(np.int32)
    g2l = {g: l for l, g in enumerate(unique_idx)}
    local_idx = np.array([g2l[g] for g in idx], dtype=np.int32)
    np.add.at(sum_gn2, local_idx, gn ** 2)
    np.add.at(sum_in2, local_idx, inn ** 2)

    beta_mean = np.where(sum_gn2 > 0, sum_in2 / sum_gn2, 0.0)
    tightening = np.where(eps_norm > 0, eps_direction / eps_norm, 0.0)

    certs = {
        "example_idx":     unique_idx,
        "n_sampled":       n_sampled,
        "eps_norm":        eps_norm,
        "eps_direction":   eps_direction,
        "beta_mean":       beta_mean,
        "tightening_ratio": tightening,
        "sum_gnorm2":      sum_gn2,
        "sum_inorm2":      sum_in2,
    }

    # --- Sanity checks ---
    _sanity_checks(arm_name, eps_norm, eps_direction, n_sampled,
                   eps, q, sigma, T_steps, ALPHA_GRID, delta)

    # --- Summary ---
    _print_summary(arm_name, certs, eps, delta, sigma, q, T_steps)

    # --- Save certificates ---
    os.makedirs(cert_dir, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "example_idx", "n_sampled", "eps_norm", "eps_direction",
            "beta_mean", "tightening_ratio", "sum_gnorm2", "sum_inorm2"])
        w.writeheader()
        for i in range(len(unique_idx)):
            w.writerow({
                "example_idx":     int(unique_idx[i]),
                "n_sampled":       int(n_sampled[i]),
                "eps_norm":        f"{eps_norm[i]:.6f}",
                "eps_direction":   f"{eps_direction[i]:.6f}",
                "beta_mean":       f"{beta_mean[i]:.6f}",
                "tightening_ratio": f"{tightening[i]:.6f}",
                "sum_gnorm2":      f"{sum_gn2[i]:.6f}",
                "sum_inorm2":      f"{sum_in2[i]:.6f}",
            })
    print(f"[P14-B] Certificates saved: {out_path}")
    return certs


def _load_certificates(path):
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return {
            "example_idx":     df["example_idx"].to_numpy(np.int32),
            "n_sampled":       df["n_sampled"].to_numpy(np.int32),
            "eps_norm":        df["eps_norm"].to_numpy(np.float64),
            "eps_direction":   df["eps_direction"].to_numpy(np.float64),
            "beta_mean":       df["beta_mean"].to_numpy(np.float64),
            "tightening_ratio": df["tightening_ratio"].to_numpy(np.float64),
        }
    except Exception:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        return {
            "example_idx":     np.array([int(r["example_idx"]) for r in rows]),
            "n_sampled":       np.array([int(r["n_sampled"]) for r in rows]),
            "eps_norm":        np.array([float(r["eps_norm"]) for r in rows]),
            "eps_direction":   np.array([float(r["eps_direction"]) for r in rows]),
            "beta_mean":       np.array([float(r["beta_mean"]) for r in rows]),
            "tightening_ratio": np.array([float(r["tightening_ratio"]) for r in rows]),
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_all(certs_by_arm, out_dir, eps=EPS):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = {
        "vanilla_warm_log": "steelblue",
        "gep_log":          "darkorange",
        "pda_dpmd_log":     "forestgreen",
    }
    labels = {
        "vanilla_warm_log": "Vanilla DP-SGD",
        "gep_log":          "GEP (two-channel)",
        "pda_dpmd_log":     "PDA-DPMD",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Phase 14: Post-Hoc Direction-Aware Certificates on Standard DP-SGD "
        f"(ε={eps}, δ={DELTA})", fontsize=12)

    # --- Panel 1: histogram of eps_norm ---
    ax = axes[0, 0]
    for arm, certs in certs_by_arm.items():
        ax.hist(certs["eps_norm"], bins=60, alpha=0.6,
                label=labels.get(arm, arm), color=colors.get(arm))
    ax.axvline(eps, color="red", ls="--", lw=1.5, label=f"ε={eps}")
    ax.set_xlabel("ε_norm (norm-based, standard adversary)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of norm-based per-instance certificates")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Panel 2: histogram of eps_direction ---
    ax = axes[0, 1]
    for arm, certs in certs_by_arm.items():
        ax.hist(certs["eps_direction"], bins=60, alpha=0.6,
                label=labels.get(arm, arm), color=colors.get(arm))
    ax.axvline(eps, color="red", ls="--", lw=1.5, label=f"ε={eps}")
    ax.set_xlabel("ε_direction (direction-aware, agg-uncertain adversary)")
    ax.set_title("Distribution of direction-aware per-instance certificates")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Panel 3: scatter eps_norm vs eps_direction colored by beta ---
    ax = axes[1, 0]
    for arm, certs in certs_by_arm.items():
        sc = ax.scatter(certs["eps_norm"], certs["eps_direction"],
                        c=certs["beta_mean"], cmap="viridis",
                        alpha=0.3, s=4, vmin=0, vmax=1,
                        label=labels.get(arm, arm))
    all_en = np.concatenate([c["eps_norm"] for c in certs_by_arm.values()])
    mn, mx = all_en.min(), all_en.max()
    ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, label="y=x (no tightening)")
    ax.plot([mn, mx], [mn*0.5, mx*0.5], "r:", lw=1, label="y=0.5x (2× tightening)")
    ax.axhline(eps, color="red", lw=0.5, alpha=0.4)
    ax.axvline(eps, color="red", lw=0.5, alpha=0.4)
    ax.set_xlabel("ε_norm"); ax.set_ylabel("ε_direction")
    ax.set_title("Direction-aware vs norm-based (color = β_mean)")
    plt.colorbar(sc, ax=ax, label="β_mean")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # --- Panel 4: tightening ratio distribution ---
    ax = axes[1, 1]
    for arm, certs in certs_by_arm.items():
        ax.hist(certs["tightening_ratio"], bins=60, alpha=0.6,
                label=labels.get(arm, arm), color=colors.get(arm))
    ax.axvline(0.5, color="red", ls="--", lw=1.5, label="ratio=0.5 (2× tightening)")
    ax.set_xlabel("ε_direction / ε_norm (tightening ratio)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of tightening ratios (< 1 = direction-aware helps)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "p14_certificates.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P14-B] Figure saved: {path}")


def _print_cross_arm_comparison(certs_by_arm, eps):
    print(f"\n{'='*70}")
    print(f" Cross-Arm Comparison (ε={eps}, δ={DELTA})")
    print(f"{'='*70}")
    header = f"  {'Arm':<22} {'en_mean':>8} {'en_p95':>8} {'en_max':>8} " \
             f"{'ed_mean':>8} {'ed_p95':>8} {'ed_max':>8} {'ratio':>8}"
    print(header)
    print(f"  {'─'*78}")
    for arm, certs in certs_by_arm.items():
        en  = certs["eps_norm"]
        ed  = certs["eps_direction"]
        rat = certs["tightening_ratio"]
        print(f"  {arm:<22} "
              f"{en.mean():8.3f} {np.percentile(en,95):8.3f} {en.max():8.3f} "
              f"{ed.mean():8.3f} {np.percentile(ed,95):8.3f} {ed.max():8.3f} "
              f"{rat.mean():8.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm",      type=str, default=None, choices=ARMS)
    parser.add_argument("--eps",      type=float, default=EPS)
    parser.add_argument("--seed",     type=int,   default=0)
    parser.add_argument("--delta",    type=float, default=DELTA)
    parser.add_argument("--log_dir",  type=str,   default=LOG_DIR)
    parser.add_argument("--cert_dir", type=str,   default=CERT_DIR)
    args = parser.parse_args()

    os.makedirs(args.cert_dir, exist_ok=True)

    arms = [args.arm] if args.arm else ARMS
    certs_by_arm = {}

    for arm in arms:
        certs = _run_arm(arm, args.eps, args.seed,
                         args.log_dir, args.cert_dir, args.delta)
        if certs is not None:
            certs_by_arm[arm] = certs

    if len(certs_by_arm) > 1:
        _print_cross_arm_comparison(certs_by_arm, args.eps)
        _plot_all(certs_by_arm, args.cert_dir, args.eps)

    # Headline result for vanilla_warm_log
    if "vanilla_warm_log" in certs_by_arm:
        certs = certs_by_arm["vanilla_warm_log"]
        en  = certs["eps_norm"]
        ed  = certs["eps_direction"]
        rat = certs["tightening_ratio"]
        print(f"\n{'='*70}")
        print(f" HEADLINE: Standard DP-SGD at ε={args.eps}")
        print(f"{'='*70}")
        print(f"  eps_norm:      mean={en.mean():.3f}, max={en.max():.3f} ≤ {args.eps} "
              f"({'✓' if en.max() <= args.eps else '✗'})")
        print(f"  eps_direction: mean={ed.mean():.3f}, max={ed.max():.3f}")
        print(f"  Tightening:    mean={rat.mean():.3f} "
              f"({(rat < 0.55).mean()*100:.0f}% examples get ≥2× improvement)")
        print(f"  → Standard DP at ε={args.eps} provides ε_i≈{ed.mean():.2f} "
              f"to typical examples under direction-aware accounting.")


if __name__ == "__main__":
    main()
