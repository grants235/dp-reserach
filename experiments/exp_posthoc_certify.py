#!/usr/bin/env python3
"""
Phase 13 — Experiment 2: Post-Hoc Per-Instance Privacy Certificates
=====================================================================

Loads per-step gradient logs from Exp 1 and computes per-instance privacy
certificates for each training example. No GPU required.

Two bounds are computed for each example i:

  eps_norm(i):       standard Thudi et al. bound using full clipped gradient norm
                     ε_i^norm(α) = Σ_{t: i∈B_t} α/2 * ||g_bar_it||² / σ_use²
                     where σ_use = sigma_van * c_eff  (absolute noise std)

  eps_direction(i):  direction-aware bound using incoherent norm only
                     ε_i^dir(α)  = Σ_{t: i∈B_t} α/2 * ||g_bar_it^perp||² / σ_use²

Both are converted to (ε, δ)-DP via RDP-to-DP conversion:
  ε_i = min_{α>1} [ ε_i(α) + log(1/δ) / (α-1) ]

Gate: if max_i eps_norm ≤ 2, the reduced-noise mechanism is unconditionally
      certified for this dataset under the standard adversary.

Usage
-----
  # After running exp_posthoc_train.py:
  python experiments/exp_posthoc_certify.py --arm dist_aware_log --gpu 0
  python experiments/exp_posthoc_certify.py --arm pda_dpmd_da_log
  python experiments/exp_posthoc_certify.py  # both arms
"""

import os
import sys
import csv
import math
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Constants (must match exp_posthoc_train.py)
# ---------------------------------------------------------------------------

DELTA        = 1e-5
EPS          = 2.0
CLIP_C       = 1.0
C_EFF_RATIO  = 0.4
N_TRAIN      = 45000    # approximate, will be read from log
BATCH_SIZE   = 1000
EPOCHS       = 60
ARMS         = ["dist_aware_log", "pda_dpmd_da_log"]

RESULTS_DIR  = "./results/exp_p13"
LOG_DIR      = os.path.join(RESULTS_DIR, "logs")
EXP1_DIR     = os.path.join(RESULTS_DIR, "exp1")
EXP2_DIR     = os.path.join(RESULTS_DIR, "exp2")

# RDP alpha grid for optimization
ALPHA_GRID   = np.concatenate([
    np.arange(1.5,  10,   0.5),
    np.arange(10,   100,  2.0),
    np.arange(100,  1000, 20.0),
])


# ---------------------------------------------------------------------------
# Privacy calibration helper
# ---------------------------------------------------------------------------

def _calibrate_sigma(eps, delta, q, steps):
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(
        target_epsilon=eps, target_delta=delta,
        sample_rate=q, steps=steps, accountant="rdp")


def _get_sigma_use(eps, delta, n_train, batch_size, epochs):
    """Reproduce the sigma_use used in training."""
    from math import ceil
    steps_per_epoch = n_train // batch_size
    T_steps         = epochs * steps_per_epoch
    q               = batch_size / n_train
    sigma_van       = _calibrate_sigma(eps, delta, q, T_steps)
    c_eff           = C_EFF_RATIO * CLIP_C
    sigma_use       = sigma_van * c_eff
    return sigma_use, sigma_van, c_eff, q, T_steps


# ---------------------------------------------------------------------------
# RDP-to-DP conversion
# ---------------------------------------------------------------------------

def _rdp_to_dp(rdp_values, alphas, delta):
    """
    Convert a vector of RDP values rdp_values[i] = ε_i(α_i) to (ε, δ)-DP.
    Returns ε = min_α [ rdp_values(α) + log(1/δ) / (α-1) ].
    """
    candidates = rdp_values + np.log(1.0 / delta) / (alphas - 1.0)
    idx = np.argmin(candidates)
    return float(candidates[idx]), float(alphas[idx])


# ---------------------------------------------------------------------------
# Core: per-instance certificate computation
# ---------------------------------------------------------------------------

def _compute_certificates(log_df, sigma_use, n_train, delta=DELTA):
    """
    Compute eps_norm and eps_direction for every training example.

    log_df columns: step, example_idx, grad_norm, incoherent_norm, beta

    Returns: dict with arrays of shape (n_train,):
      n_sampled       — number of times each example was sampled
      sum_gnorm2      — Σ grad_norm²
      sum_inorm2      — Σ incoherent_norm²
      eps_norm        — post-hoc (ε, δ)-DP via full norm
      eps_direction   — post-hoc (ε, δ)-DP via incoherent norm
    """
    sigma2 = sigma_use ** 2

    # Accumulate per-example sums (streaming, no full log in memory needed)
    n_sampled  = np.zeros(n_train, dtype=np.int32)
    sum_gn2    = np.zeros(n_train, dtype=np.float64)
    sum_in2    = np.zeros(n_train, dtype=np.float64)

    idx  = log_df["example_idx"].to_numpy(dtype=np.int32)
    gn   = log_df["grad_norm"].to_numpy(dtype=np.float64)
    inn  = log_df["incoherent_norm"].to_numpy(dtype=np.float64)

    # Map global indices to [0, n_train)
    unique_idx = np.unique(idx)
    global_to_local = {g: l for l, g in enumerate(sorted(unique_idx))}
    n_present = len(unique_idx)

    local_idx = np.array([global_to_local[g] for g in idx], dtype=np.int32)
    np.add.at(n_sampled[:n_present], local_idx, 1)
    np.add.at(sum_gn2[:n_present],  local_idx, gn ** 2)
    np.add.at(sum_in2[:n_present],  local_idx, inn ** 2)

    # For each example: compute RDP value across alpha grid, then convert to DP
    eps_norm_arr = np.full(n_present, np.inf)
    eps_dir_arr  = np.full(n_present, np.inf)

    for i in range(n_present):
        if n_sampled[i] == 0:
            eps_norm_arr[i] = 0.0
            eps_dir_arr[i]  = 0.0
            continue

        # Per-example RDP: ε_i(α) = Σ_{t: i∈B_t} α/2 * ||g||² / σ²
        # Factored: ε_i(α) = (α/2) * sum_gn2[i] / sigma2
        rdp_norm = (ALPHA_GRID / 2.0) * sum_gn2[i] / sigma2
        rdp_dir  = (ALPHA_GRID / 2.0) * sum_in2[i]  / sigma2

        eps_norm_arr[i], _ = _rdp_to_dp(rdp_norm, ALPHA_GRID, delta)
        eps_dir_arr[i],  _ = _rdp_to_dp(rdp_dir,  ALPHA_GRID, delta)

    return {
        "example_idx":    np.array(sorted(unique_idx), dtype=np.int32),
        "n_sampled":      n_sampled[:n_present],
        "sum_gnorm2":     sum_gn2[:n_present],
        "sum_inorm2":     sum_in2[:n_present],
        "eps_norm":       eps_norm_arr,
        "eps_direction":  eps_dir_arr,
        "beta_mean":      np.where(sum_gn2[:n_present] > 0,
                                   sum_in2[:n_present] / sum_gn2[:n_present],
                                   0.0),
    }


# ---------------------------------------------------------------------------
# Main per-arm processing
# ---------------------------------------------------------------------------

def _run_arm(arm_name, eps, seed, log_dir, out_dir, delta=DELTA):
    tag      = f"{arm_name}_eps{eps:.0f}_seed{seed}"
    log_path = os.path.join(log_dir, f"{tag}.parquet")
    npz_path = os.path.join(log_dir, f"{tag}.npz")
    out_path = os.path.join(out_dir, f"{tag}_certificates.csv")

    if os.path.exists(out_path):
        print(f"[P13-E2] {tag}: certificates already computed.")
        return _load_certificates(out_path)

    # Load log
    if os.path.exists(log_path):
        try:
            import pandas as pd
            log_df = pd.read_parquet(log_path)
        except ImportError:
            print("[P13-E2] pandas not available, trying npz...")
            log_df = None
    elif os.path.exists(npz_path):
        log_df = None
    else:
        print(f"[P13-E2] Log not found: {log_path}. Run exp_posthoc_train.py first.")
        return None

    if log_df is None:
        # npz fallback
        import pandas as pd
        data   = np.load(npz_path)
        log_df = pd.DataFrame({
            "step":            data["step"],
            "example_idx":    data["example_idx"],
            "grad_norm":      data["grad_norm"],
            "incoherent_norm": data["incoherent_norm"],
        })
        log_df["beta"] = (log_df["incoherent_norm"] ** 2 /
                          np.maximum(log_df["grad_norm"] ** 2, 1e-12))

    n_rows = len(log_df)
    n_train_est = int(log_df["example_idx"].nunique())
    print(f"[P13-E2] {tag}: loaded {n_rows:,} log rows, "
          f"{n_train_est} unique examples")

    # Infer sigma_use from training config
    sigma_use, sigma_van, c_eff, q, T_steps = _get_sigma_use(
        eps, delta, n_train_est, BATCH_SIZE, EPOCHS)
    print(f"[P13-E2] sigma_van={sigma_van:.4f}, c_eff={c_eff:.3f}, "
          f"sigma_use={sigma_use:.4f}")

    # Compute certificates
    print(f"[P13-E2] Computing certificates for {n_train_est} examples...")
    certs = _compute_certificates(log_df, sigma_use, n_train_est, delta)

    # Save certificates
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "example_idx", "n_sampled", "eps_norm", "eps_direction",
            "beta_mean", "sum_gnorm2", "sum_inorm2"])
        w.writeheader()
        for i in range(len(certs["example_idx"])):
            w.writerow({
                "example_idx":   int(certs["example_idx"][i]),
                "n_sampled":     int(certs["n_sampled"][i]),
                "eps_norm":      f"{certs['eps_norm'][i]:.6f}",
                "eps_direction": f"{certs['eps_direction'][i]:.6f}",
                "beta_mean":     f"{certs['beta_mean'][i]:.6f}",
                "sum_gnorm2":    f"{certs['sum_gnorm2'][i]:.6f}",
                "sum_inorm2":    f"{certs['sum_inorm2'][i]:.6f}",
            })
    print(f"[P13-E2] Certificates saved: {out_path}")

    # Print gate decision and summary
    _print_summary(arm_name, certs, eps, sigma_use, c_eff)

    return certs


def _load_certificates(path):
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return {
            "example_idx":   df["example_idx"].to_numpy(np.int32),
            "n_sampled":     df["n_sampled"].to_numpy(np.int32),
            "eps_norm":      df["eps_norm"].to_numpy(np.float64),
            "eps_direction": df["eps_direction"].to_numpy(np.float64),
            "beta_mean":     df["beta_mean"].to_numpy(np.float64),
        }
    except Exception:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        return {
            "example_idx":   np.array([int(r["example_idx"]) for r in rows]),
            "n_sampled":     np.array([int(r["n_sampled"]) for r in rows]),
            "eps_norm":      np.array([float(r["eps_norm"]) for r in rows]),
            "eps_direction": np.array([float(r["eps_direction"]) for r in rows]),
            "beta_mean":     np.array([float(r["beta_mean"]) for r in rows]),
        }


def _print_summary(arm_name, certs, eps, sigma_use, c_eff):
    en  = certs["eps_norm"]
    ed  = certs["eps_direction"]
    ns  = certs["n_sampled"]
    bm  = certs["beta_mean"]

    print(f"\n{'='*70}")
    print(f" Post-Hoc Certificate Summary — {arm_name}  (ε={eps}, δ={DELTA})")
    print(f"  sigma_use={sigma_use:.4f}, C_eff={c_eff:.3f}")
    print(f"{'='*70}")

    print(f"\n  Norm-Based Certificate (eps_norm, standard Thudi bound):")
    print(f"    max:    {en.max():.4f}  {'✓ PASS' if en.max() <= eps else '✗ FAIL'}")
    print(f"    99th:   {np.percentile(en, 99):.4f}")
    print(f"    95th:   {np.percentile(en, 95):.4f}")
    print(f"    mean:   {en.mean():.4f}")
    print(f"    ≤ eps:  {(en <= eps).mean()*100:.1f}% of examples")

    print(f"\n  Direction-Aware Certificate (eps_direction, incoherent norm):")
    print(f"    max:    {ed.max():.4f}  {'✓ PASS' if ed.max() <= eps else '✗ FAIL'}")
    print(f"    99th:   {np.percentile(ed, 99):.4f}")
    print(f"    95th:   {np.percentile(ed, 95):.4f}")
    print(f"    mean:   {ed.mean():.4f}")
    print(f"    ≤ eps:  {(ed <= eps).mean()*100:.1f}% of examples")

    print(f"\n  Tightening ratio (eps_direction / eps_norm):")
    ratio = ed / np.maximum(en, 1e-12)
    print(f"    mean:  {ratio.mean():.4f}  (< 1 means direction-aware is tighter)")
    print(f"    median:{np.median(ratio):.4f}")

    print(f"\n  Sampling statistics:")
    print(f"    mean times sampled: {ns.mean():.1f}")
    print(f"    max times sampled:  {ns.max()}")

    print(f"\n  beta_mean (incoherent fraction) statistics:")
    print(f"    mean:   {bm.mean():.4f}  (0=fully coherent, 1=fully incoherent)")
    print(f"    95th:   {np.percentile(bm, 95):.4f}")
    print(f"    max:    {bm.max():.4f}")

    # Gate decision
    print(f"\n{'='*70}")
    if en.max() <= eps:
        print(f"  GATE PASSED (norm-based): max eps_norm={en.max():.4f} ≤ {eps}")
        print(f"  The reduced-noise mechanism is unconditionally certified.")
    elif ed.max() <= eps:
        print(f"  GATE PASSED (direction-aware only): max eps_dir={ed.max():.4f} ≤ {eps}")
        print(f"  Certificate holds under aggregate-uncertain adversary model.")
    else:
        pct_norm = (en <= eps).mean() * 100
        pct_dir  = (ed <= eps).mean() * 100
        print(f"  GATE FAILED: max eps_norm={en.max():.4f}, max eps_dir={ed.max():.4f}")
        print(f"  {pct_norm:.1f}% pass norm-based, {pct_dir:.1f}% pass direction-aware.")
        if pct_norm >= 95:
            print(f"  High-probability result: ≥95% of examples certified.")


def _plot_distributions(certs_by_arm, out_dir, eps=EPS):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"dist_aware_log": "steelblue", "pda_dpmd_da_log": "darkorange"}

    # Left: histogram of eps_norm
    ax = axes[0]
    for arm, certs in certs_by_arm.items():
        ax.hist(certs["eps_norm"], bins=50, alpha=0.6,
                label=arm, color=colors.get(arm))
    ax.axvline(eps, color="red", ls="--", lw=1.5, label=f"ε={eps}")
    ax.set_xlabel("ε_norm (norm-based post-hoc)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of norm-based certificates")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Middle: histogram of eps_direction
    ax = axes[1]
    for arm, certs in certs_by_arm.items():
        ax.hist(certs["eps_direction"], bins=50, alpha=0.6,
                label=arm, color=colors.get(arm))
    ax.axvline(eps, color="red", ls="--", lw=1.5, label=f"ε={eps}")
    ax.set_xlabel("ε_direction (direction-aware post-hoc)")
    ax.set_title("Distribution of direction-aware certificates")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: scatter eps_norm vs eps_direction
    ax = axes[2]
    for arm, certs in certs_by_arm.items():
        ax.scatter(certs["eps_norm"], certs["eps_direction"],
                   alpha=0.3, s=4, label=arm, color=colors.get(arm))
    mn = min(c["eps_norm"].min() for c in certs_by_arm.values())
    mx = max(c["eps_norm"].max() for c in certs_by_arm.values())
    ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, label="y=x (no improvement)")
    ax.axhline(eps, color="red", ls=":", lw=1, alpha=0.5)
    ax.axvline(eps, color="red", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("ε_norm")
    ax.set_ylabel("ε_direction")
    ax.set_title("Direction-aware vs norm-based tightening")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "p13_certificates.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P13-E2] Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm",     type=str, default=None, choices=ARMS)
    parser.add_argument("--eps",     type=float, default=EPS)
    parser.add_argument("--seed",    type=int,   default=0)
    parser.add_argument("--delta",   type=float, default=DELTA)
    parser.add_argument("--log_dir", type=str,   default=LOG_DIR)
    parser.add_argument("--out_dir", type=str,   default=EXP2_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    arms = [args.arm] if args.arm else ARMS
    certs_by_arm = {}

    for arm in arms:
        certs = _run_arm(arm, args.eps, args.seed, args.log_dir, args.out_dir, args.delta)
        if certs is not None:
            certs_by_arm[arm] = certs

    if len(certs_by_arm) > 1:
        _plot_distributions(certs_by_arm, args.out_dir, args.eps)

        # Cross-arm comparison
        print(f"\n{'='*70}")
        print(" Cross-Arm Comparison")
        print(f"{'='*70}")
        for arm, certs in certs_by_arm.items():
            en = certs["eps_norm"]
            ed = certs["eps_direction"]
            print(f"  {arm}:")
            print(f"    eps_norm max={en.max():.3f}  "
                  f"95th={np.percentile(en, 95):.3f}  "
                  f"pass={( en <= args.eps).mean()*100:.1f}%")
            print(f"    eps_dir  max={ed.max():.3f}  "
                  f"95th={np.percentile(ed, 95):.3f}  "
                  f"pass={(ed <= args.eps).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
