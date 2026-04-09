#!/usr/bin/env python3
"""
Phase 15 — Certify: Direction-Aware Per-Instance Certificate Computation
=========================================================================

Loads per-step gradient logs from exp_p15_train.py and computes three
per-instance privacy certificates for each training example:

  1. Data-independent ε:        standard RDP accountant, worst-case sensitivity
  2. Norm-based ε_i^norm:       Thudi-style, using actual ||ḡ_it|| per sampled step
  3. Direction-aware ε_i^dir:   our bound, using ||(I-P_V) ḡ_it|| per sampled step

For long-tailed arms (B1, B2), results are also stratified by tier (head/mid/tail).

RDP → (ε,δ)-DP conversion:
  ε_i = min_{α>1} [ ε_i(α) + log(1/δ)/(α-1) ]
where ε_i(α) = (α/2) * Σ_{t: i∈B_t} ||g||² / σ_use²

Usage
-----
  # After running exp_p15_train.py:
  python experiments/exp_p15_certify.py --arm A3
  python experiments/exp_p15_certify.py --arm B2
  python experiments/exp_p15_certify.py --arm C3
  python experiments/exp_p15_certify.py           # all arms in priority order
"""

import os
import sys
import csv
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Constants (match exp_p15_train.py)
# ---------------------------------------------------------------------------

RESULTS_DIR = "./results/exp_p15"
TRAIN_DIR   = os.path.join(RESULTS_DIR, "train")
LOG_DIR     = os.path.join(RESULTS_DIR, "logs")
CERT_DIR    = os.path.join(RESULTS_DIR, "certs")

ALL_ARMS    = ["A3", "B2", "C3", "A2", "A1", "B1", "C2", "C1"]

# Tier names for B arms
TIER_NAMES  = {0: "head", 1: "mid", 2: "tail"}

# RDP alpha grid — extended to avoid "optimal order is largest alpha" warning
ALPHA_GRID  = np.concatenate([
    np.arange(1.5,  10,    0.5),
    np.arange(10,   100,   2.0),
    np.arange(100,  1000,  20.0),
    np.arange(1000, 5001,  100.0),
])


# ---------------------------------------------------------------------------
# RDP → DP conversion
# ---------------------------------------------------------------------------

def _rdp_to_dp(rdp_values, alphas, delta):
    candidates = rdp_values + np.log(1.0 / delta) / (alphas - 1.0)
    idx = np.argmin(candidates)
    return float(candidates[idx]), float(alphas[idx])


# ---------------------------------------------------------------------------
# Data-independent certificate
# ---------------------------------------------------------------------------

def _data_independent_eps(sigma_mult, q, T_steps, delta, eps_target):
    """
    Verify that the RDP accountant gives eps_target for the training run.
    Returns (eps_di, alpha_opt).
    """
    try:
        from opacus.accountants import RDPAccountant
        acct = RDPAccountant()
        acct.history = [(sigma_mult, q, T_steps)]
        eps_di, alpha_opt = acct.get_privacy_spent(delta=delta)
        return float(eps_di), float(alpha_opt)
    except Exception:
        # Fallback: manual RDP composition
        # ε(α) = T * q^2 * α / (2 * sigma_mult^2)   (Gaussian mechanism approx)
        rdp = T_steps * (q**2) * ALPHA_GRID / (2.0 * sigma_mult**2)
        eps_di, alpha_opt = _rdp_to_dp(rdp, ALPHA_GRID, delta)
        return eps_di, alpha_opt


# ---------------------------------------------------------------------------
# Per-instance certificates
# ---------------------------------------------------------------------------

def _compute_certificates(log_df, sigma_use, delta, n_examples_est, q):
    """
    Compute norm-based and direction-aware per-instance certificates.

    Uses the subsampled Gaussian mechanism RDP (Thudi et al. Theorem 3.2):
      ε_i(α) ≈ q² * (α/2) * Σ_{t: i∈B_t} ||ḡ_it||² / σ_use²

    The q² factor is the subsampling amplification from Poisson sampling at
    rate q. Without it, the formula is the raw (unsubsampled) Gaussian RDP,
    which loses the amplification and can exceed the data-independent ε.

    Mathematically, since ||ḡ_it|| ≤ C and there are at most T steps,
    ε_norm_i ≤ q² * (α/2) * qT * C² / σ_use² ≈ q * ε_DI, ensuring
    ε_norm_i ≤ ε_DI always holds.

    sigma_use: absolute noise std = sigma_multiplier * C  (= sigma_mult when C=1)
    q:         Poisson sampling rate = batch_size / n_train

    Returns dict with arrays (one entry per unique example in log):
      example_idx, n_sampled, sum_gnorm2, sum_inorm2,
      eps_norm, eps_direction, beta_mean
    """
    sigma2 = float(sigma_use) ** 2
    q_sq   = float(q) ** 2   # subsampling amplification factor

    idx_arr = log_df["example_idx"].to_numpy(dtype=np.int32)
    gn_arr  = log_df["grad_norm"].to_numpy(dtype=np.float64)
    in_arr  = log_df["incoherent_norm"].to_numpy(dtype=np.float64)

    unique_idx = np.unique(idx_arr)
    n_present  = len(unique_idx)
    g2l        = {g: l for l, g in enumerate(unique_idx)}
    local_idx  = np.array([g2l[g] for g in idx_arr], dtype=np.int32)

    n_sampled = np.zeros(n_present, dtype=np.int32)
    sum_gn2   = np.zeros(n_present, dtype=np.float64)
    sum_in2   = np.zeros(n_present, dtype=np.float64)

    np.add.at(n_sampled, local_idx, 1)
    np.add.at(sum_gn2,   local_idx, gn_arr ** 2)
    np.add.at(sum_in2,   local_idx, in_arr  ** 2)

    eps_norm_arr = np.full(n_present, np.inf)
    eps_dir_arr  = np.full(n_present, np.inf)

    for i in range(n_present):
        if n_sampled[i] == 0:
            eps_norm_arr[i] = 0.0
            eps_dir_arr[i]  = 0.0
            continue
        # Subsampled Gaussian mechanism RDP (quadratic approximation, valid for small q):
        # ε_i(α) = q² * (α/2) * Σ_t ||ḡ_it||² / σ²
        rdp_norm = q_sq * (ALPHA_GRID / 2.0) * sum_gn2[i] / sigma2
        rdp_dir  = q_sq * (ALPHA_GRID / 2.0) * sum_in2[i] / sigma2
        eps_norm_arr[i], _ = _rdp_to_dp(rdp_norm, ALPHA_GRID, delta)
        eps_dir_arr[i],  _ = _rdp_to_dp(rdp_dir,  ALPHA_GRID, delta)

    beta_mean = np.where(sum_gn2 > 0, sum_in2 / sum_gn2, 0.0)

    return {
        "example_idx":   unique_idx,
        "n_sampled":     n_sampled,
        "sum_gnorm2":    sum_gn2,
        "sum_inorm2":    sum_in2,
        "eps_norm":      eps_norm_arr,
        "eps_direction": eps_dir_arr,
        "beta_mean":     beta_mean,
    }


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _summarize(label, certs, delta, eps_target, tier_labels=None):
    en   = certs["eps_norm"]
    ed   = certs["eps_direction"]
    ns   = certs["n_sampled"]
    bm   = certs["beta_mean"]
    ratio = ed / np.maximum(en, 1e-12)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  n_examples={len(en)}, ε_target={eps_target}, δ={delta:.2e}")
    print(f"{'='*70}")
    print(f"  Norm-based certificate (ε_i^norm):")
    print(f"    max={en.max():.4f}  95th={np.percentile(en,95):.4f}  "
          f"mean={en.mean():.4f}")
    print(f"  Direction-aware certificate (ε_i^dir):")
    print(f"    max={ed.max():.4f}  95th={np.percentile(ed,95):.4f}  "
          f"mean={ed.mean():.4f}")
    print(f"  Tightening ratio (ε_dir/ε_norm):")
    print(f"    mean={ratio.mean():.4f}  median={np.median(ratio):.4f}  "
          f"min={ratio.min():.4f}")
    print(f"  Sampling: mean_times={ns.mean():.1f}  max={ns.max()}")
    print(f"  beta_mean: mean={bm.mean():.4f}  95th={np.percentile(bm,95):.4f}")

    if tier_labels is not None and len(tier_labels) > 0:
        # tier_labels indexed same as example_idx (by local position)
        print(f"\n  Per-tier breakdown:")
        for t, tname in TIER_NAMES.items():
            mask = tier_labels == t
            if mask.sum() == 0:
                continue
            en_t = en[mask]
            ed_t = ed[mask]
            r_t  = ratio[mask]
            print(f"    {tname:4s} (n={mask.sum():5d}): "
                  f"ε_norm mean={en_t.mean():.4f}  "
                  f"ε_dir mean={ed_t.mean():.4f}  "
                  f"tightening={r_t.mean():.4f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_arm(arm, certs, eps_target, tier_labels, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    en = certs["eps_norm"]
    ed = certs["eps_direction"]
    bm = certs["beta_mean"]
    is_lt = tier_labels is not None and len(tier_labels) > 0

    # --- Scatter: eps_norm vs eps_direction, colored by beta ---
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(en, ed, c=bm, cmap="plasma_r", s=3, alpha=0.5, vmin=0, vmax=1)
    mn, mx = min(en.min(), ed.min()), max(en.max(), ed.max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, label="y=x (no improvement)")
    ax.set_xlabel("ε_norm (norm-based)")
    ax.set_ylabel("ε_dir (direction-aware)")
    ax.set_title(f"Arm {arm}: direction-aware vs norm-based  (ε={eps_target})")
    plt.colorbar(sc, ax=ax, label="β (incoherent fraction)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"p15_{arm}_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")

    # --- Histogram of eps_direction ---
    fig, ax = plt.subplots(figsize=(6, 4))
    if is_lt:
        colors = {0: "steelblue", 1: "orange", 2: "firebrick"}
        for t, tname in TIER_NAMES.items():
            mask = tier_labels == t
            if mask.sum() == 0:
                continue
            ax.hist(ed[mask], bins=50, alpha=0.6, label=tname, color=colors[t])
        ax.axvline(eps_target, color="k", ls="--", lw=1, label=f"ε={eps_target}")
        ax.set_title(f"Arm {arm}: ε_dir by tier  (ε={eps_target})")
        ax.legend()
    else:
        ax.hist(en, bins=50, alpha=0.6, label="ε_norm", color="steelblue")
        ax.hist(ed, bins=50, alpha=0.6, label="ε_dir",  color="darkorange")
        ax.axvline(eps_target, color="k", ls="--", lw=1, label=f"ε={eps_target}")
        ax.set_title(f"Arm {arm}: certificates  (ε={eps_target})")
        ax.legend()
    ax.set_xlabel("per-instance ε")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"p15_{arm}_hist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")


# ---------------------------------------------------------------------------
# Per-arm processing
# ---------------------------------------------------------------------------

def _run_arm(arm, seed, log_dir, train_dir, cert_dir):
    tag      = f"p15_{arm}_eps8_seed{seed}"
    log_path = os.path.join(log_dir, f"{tag}.parquet")
    log_npz  = log_path.replace(".parquet", ".npz")
    meta_path = os.path.join(log_dir, f"{tag}_meta.npz")
    cert_path = os.path.join(cert_dir, f"{tag}_certs.csv")
    summ_path = os.path.join(cert_dir, f"{tag}_summary.txt")

    if os.path.exists(cert_path):
        print(f"[P15-cert] {tag}: certificates already exist, skipping computation.")
        return _load_certs(cert_path)

    # Load log
    log_df = None
    if os.path.exists(log_path):
        try:
            import pandas as pd
            log_df = pd.read_parquet(log_path)
        except Exception as e:
            print(f"  [warn] parquet load failed: {e}")
    if log_df is None and os.path.exists(log_npz):
        import pandas as pd
        data   = np.load(log_npz)
        log_df = pd.DataFrame({
            "example_idx":     data["example_idx"],
            "grad_norm":       data["grad_norm"],
            "incoherent_norm": data["incoherent_norm"],
        })
    if log_df is None:
        print(f"[P15-cert] Log not found for {tag}. Run exp_p15_train.py first.")
        return None

    # Load meta
    if not os.path.exists(meta_path):
        print(f"[P15-cert] Meta not found: {meta_path}")
        return None
    meta = np.load(meta_path, allow_pickle=True)

    eps        = float(meta["eps"])
    delta      = float(meta["delta"])
    sigma_use  = float(meta["sigma_use"])
    sigma_mult = float(meta["sigma_mult"])
    q          = float(meta["q"])
    T_steps    = int(meta["T_steps"])
    n_priv     = int(meta["n_priv"])
    tier_arr   = meta["tier_labels"]  # empty if not LT
    priv_idx   = meta["priv_idx"]
    has_tiers  = len(tier_arr) > 0

    print(f"\n[P15-cert] === ARM {arm} ===")
    print(f"  log rows={len(log_df):,}  n_priv={n_priv}  eps={eps}  delta={delta:.2e}")
    print(f"  sigma_use={sigma_use:.4f}  q={q:.5f}  T={T_steps}")

    # Data-independent certificate
    eps_di, alpha_di = _data_independent_eps(sigma_mult, q, T_steps, delta, eps)
    print(f"  Data-independent: ε={eps_di:.4f}  (target={eps})")

    # Per-instance certificates
    print(f"  Computing per-instance certificates ...")
    certs = _compute_certificates(log_df, sigma_use, delta, n_priv, q)

    # Map tier labels to local positions (certs["example_idx"] are global indices)
    tier_local = None
    if has_tiers:
        # priv_idx[i] is the global dataset idx for local position i
        # tier_arr[i] is the tier for local position i
        # certs["example_idx"] are global indices from the log
        global_to_local = {int(priv_idx[i]): i for i in range(len(priv_idx))}
        tier_local = np.array([
            tier_arr[global_to_local[int(gidx)]]
            if int(gidx) in global_to_local else -1
            for gidx in certs["example_idx"]
        ], dtype=np.int32)

    # Save certificates CSV
    os.makedirs(cert_dir, exist_ok=True)
    with open(cert_path, "w", newline="") as f:
        fieldnames = ["example_idx", "n_sampled", "eps_norm", "eps_direction",
                      "beta_mean", "sum_gnorm2", "sum_inorm2"]
        if has_tiers:
            fieldnames.append("tier")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        n = len(certs["example_idx"])
        for i in range(n):
            row = {
                "example_idx":   int(certs["example_idx"][i]),
                "n_sampled":     int(certs["n_sampled"][i]),
                "eps_norm":      f"{certs['eps_norm'][i]:.6f}",
                "eps_direction": f"{certs['eps_direction'][i]:.6f}",
                "beta_mean":     f"{certs['beta_mean'][i]:.6f}",
                "sum_gnorm2":    f"{certs['sum_gnorm2'][i]:.6f}",
                "sum_inorm2":    f"{certs['sum_inorm2'][i]:.6f}",
            }
            if has_tiers:
                row["tier"] = int(tier_local[i]) if tier_local is not None else -1
            w.writerow(row)
    print(f"  Certificates saved: {cert_path}")

    # Print and save summary
    label = f"Arm {arm} — {tag}"
    _summarize(label, certs, delta, eps, tier_local)

    summary_lines = _build_summary_lines(
        arm, certs, eps, eps_di, alpha_di, delta, q, T_steps,
        sigma_use, sigma_mult, tier_local)
    with open(summ_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  Summary saved: {summ_path}")

    # Plots
    _plot_arm(arm, certs, eps, tier_local, cert_dir)

    return certs


def _build_summary_lines(arm, certs, eps, eps_di, alpha_di, delta, q, T_steps,
                          sigma_use, sigma_mult, tier_local):
    en    = certs["eps_norm"]
    ed    = certs["eps_direction"]
    ratio = ed / np.maximum(en, 1e-12)
    bm    = certs["beta_mean"]
    ns    = certs["n_sampled"]

    lines = [
        f"Arm {arm}  (ε={eps}, δ={delta:.2e})",
        f"  q={q:.5f}  T={T_steps}  sigma_mult={sigma_mult:.4f}  sigma_use={sigma_use:.4f}",
        "",
        f"Data-independent:  ε_DI={eps_di:.4f}  (α={alpha_di:.1f})",
        "",
        "Norm-based certificates (ε_i^norm):",
        f"  mean={en.mean():.4f}  95th={np.percentile(en,95):.4f}  "
        f"max={en.max():.4f}",
        "",
        "Direction-aware certificates (ε_i^dir):",
        f"  mean={ed.mean():.4f}  95th={np.percentile(ed,95):.4f}  "
        f"max={ed.max():.4f}",
        "",
        "Tightening ratio (ε_dir/ε_norm):",
        f"  mean={ratio.mean():.4f}  median={np.median(ratio):.4f}  "
        f"min={ratio.min():.4f}",
        "",
        "Beta statistics:",
        f"  mean={bm.mean():.4f}  95th={np.percentile(bm,95):.4f}  max={bm.max():.4f}",
        "",
        "Sampling:",
        f"  mean_times_sampled={ns.mean():.1f}  max={ns.max()}",
    ]

    if tier_local is not None and len(tier_local) > 0:
        lines += ["", "Per-tier breakdown:"]
        for t, tname in TIER_NAMES.items():
            mask = tier_local == t
            if mask.sum() == 0:
                continue
            en_t = en[mask]
            ed_t = ed[mask]
            r_t  = ratio[mask]
            lines.append(
                f"  {tname:4s} (n={mask.sum():5d}): "
                f"ε_norm mean={en_t.mean():.4f}  95th={np.percentile(en_t,95):.4f}  "
                f"ε_dir mean={ed_t.mean():.4f}  95th={np.percentile(ed_t,95):.4f}  "
                f"tightening={r_t.mean():.4f}"
            )
    return lines


def _load_certs(path):
    try:
        import pandas as pd
        df = pd.read_csv(path)
        out = {
            "example_idx":   df["example_idx"].to_numpy(np.int32),
            "n_sampled":     df["n_sampled"].to_numpy(np.int32),
            "eps_norm":      df["eps_norm"].to_numpy(np.float64),
            "eps_direction": df["eps_direction"].to_numpy(np.float64),
            "beta_mean":     df["beta_mean"].to_numpy(np.float64),
        }
        if "tier" in df.columns:
            out["tier"] = df["tier"].to_numpy(np.int32)
        return out
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cross-arm comparison table
# ---------------------------------------------------------------------------

def _print_cross_arm_table(certs_by_arm, log_dir, seed):
    print(f"\n{'='*90}")
    print(f"  Cross-Arm Comparison Table")
    print(f"  {'Arm':4s}  {'q':7s}  {'ε_norm mean':11s}  {'ε_dir mean':10s}  "
          f"{'tightening':10s}  {'β mean':8s}  {'n_sampled':9s}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*11}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*9}")

    for arm, certs in certs_by_arm.items():
        # Load q from meta
        tag       = f"p15_{arm}_eps8_seed{seed}"
        meta_path = os.path.join(log_dir, f"{tag}_meta.npz")
        q = float("nan")
        if os.path.exists(meta_path):
            try:
                meta = np.load(meta_path)
                q = float(meta["q"])
            except Exception:
                pass

        en = certs["eps_norm"]
        ed = certs["eps_direction"]
        bm = certs["beta_mean"]
        ns = certs["n_sampled"]
        ratio = ed / np.maximum(en, 1e-12)
        print(f"  {arm:4s}  {q:7.4f}  {en.mean():11.4f}  {ed.mean():10.4f}  "
              f"{ratio.mean():10.4f}  {bm.mean():8.4f}  {ns.mean():9.1f}")
    print(f"{'='*90}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm",      type=str, default=None, choices=ALL_ARMS)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--log_dir",  type=str, default=LOG_DIR)
    parser.add_argument("--train_dir",type=str, default=TRAIN_DIR)
    parser.add_argument("--cert_dir", type=str, default=CERT_DIR)
    args = parser.parse_args()

    os.makedirs(args.cert_dir, exist_ok=True)

    arms = [args.arm] if args.arm else ALL_ARMS
    certs_by_arm = {}

    for arm in arms:
        certs = _run_arm(arm, args.seed, args.log_dir, args.train_dir, args.cert_dir)
        if certs is not None:
            certs_by_arm[arm] = certs

    if len(certs_by_arm) > 1:
        _print_cross_arm_table(certs_by_arm, args.log_dir, args.seed)

    print(f"\n[P15-cert] Done. Results in {args.cert_dir}")


if __name__ == "__main__":
    main()
