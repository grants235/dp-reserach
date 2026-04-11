#!/usr/bin/env python3
"""
Phase 16 — Certify: Direction-Aware Per-Instance Certificate Computation
=========================================================================

Loads per-step gradient logs from exp_p16_train.py and computes:

  1. Data-independent ε:        standard RDP accountant
  2. Norm-based ε_i^norm:       Thudi-style, using actual ||ḡ_it|| per step
  3. Direction-aware ε_i^dir:   using ||(I-P_V) ḡ_it|| per step

Also computes β-spectrum (rank sweep 1, 5, 9, 10, 50, 100) for R3 runs.
Aggregates over multiple seeds: reports median + IQR.

Usage
-----
  # Single run, single seed:
  python experiments/exp_p16_certify.py --run H1 --seed 0

  # All seeds for a run (aggregated):
  python experiments/exp_p16_certify.py --run H1

  # All completed runs:
  python experiments/exp_p16_certify.py --all

  # Print summary table across all runs:
  python experiments/exp_p16_certify.py --table
"""

import os
import sys
import csv
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Constants (match exp_p16_train.py)
# ---------------------------------------------------------------------------

RESULTS_DIR = "./results/exp_p16"
LOG_DIR     = os.path.join(RESULTS_DIR, "logs")
TRAIN_DIR   = os.path.join(RESULTS_DIR, "train")
CERT_DIR    = os.path.join(RESULTS_DIR, "certs")

TIER_NAMES  = {0: "head", 1: "mid", 2: "tail"}

# Extended RDP alpha grid
ALPHA_GRID  = np.concatenate([
    np.arange(1.5,  10,    0.5),
    np.arange(10,   100,   2.0),
    np.arange(100,  1000,  20.0),
    np.arange(1000, 5001,  100.0),
])

# β-spectrum rank sweep for R3 runs
BETA_RANKS = [1, 5, 9, 10, 50, 100]


# ---------------------------------------------------------------------------
# Run matrix (subset — just metadata for certificate computation)
# ---------------------------------------------------------------------------

RUN_MATRIX = {
    "H1":  dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5),
    "H2":  dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=2.0, batch=5000,  n_seeds=5),
    "H3":  dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=1.0, batch=5000,  n_seeds=5),
    "H4":  dict(dataset="cifar10_lt50",  regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5),
    "H5":  dict(dataset="cifar10_lt100", regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5),
    "H6":  dict(dataset="cifar10",       regime="R1", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5),
    "H7":  dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5),
    "H8":  dict(dataset="cifar10_lt50",  regime="R2", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5),
    "H9":  dict(dataset="emnist",        regime="R2", mech="vanilla", eps=8.0, batch=10000, n_seeds=5),
    "H10": dict(dataset="emnist",        regime="R2", mech="vanilla", eps=8.0, batch=25000, n_seeds=5),
    "S1":  dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=1.0, batch=5000,  n_seeds=3),
    "S2":  dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=2.0, batch=5000,  n_seeds=3),
    "S3":  dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=4.0, batch=5000,  n_seeds=3),
    "S4":  dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=8.0, batch=1000,  n_seeds=3),
    "S5":  dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=8.0, batch=10000, n_seeds=3),
    "S6":  dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=1000,  n_seeds=3),
    "S7":  dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=10000, n_seeds=3),
    "S8":  dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=25000, n_seeds=3),
    "M1":  dict(dataset="cifar10",       regime="R2", mech="gep",     eps=8.0, batch=5000,  n_seeds=1),
    "M2":  dict(dataset="cifar10",       regime="R2", mech="pda",     eps=8.0, batch=5000,  n_seeds=1),
    "M3":  dict(dataset="cifar10",       regime="R2", mech="auto",    eps=8.0, batch=5000,  n_seeds=1),
    "M4":  dict(dataset="cifar10_lt50",  regime="R2", mech="gep",     eps=8.0, batch=5000,  n_seeds=1),
    "M5":  dict(dataset="cifar10_lt50",  regime="R2", mech="pda",     eps=8.0, batch=5000,  n_seeds=1),
    "M6":  dict(dataset="cifar10",       regime="R3", mech="gep",     eps=8.0, batch=5000,  n_seeds=1),
    "M7":  dict(dataset="cifar10",       regime="R3", mech="auto",    eps=8.0, batch=5000,  n_seeds=1),
}


# ---------------------------------------------------------------------------
# RDP → DP conversion
# ---------------------------------------------------------------------------

def rdp_to_dp(rdp_values, alphas, delta):
    candidates = rdp_values + np.log(1.0 / delta) / (alphas - 1.0)
    idx = np.argmin(candidates)
    return float(candidates[idx]), float(alphas[idx])


# ---------------------------------------------------------------------------
# Data-independent certificate
# ---------------------------------------------------------------------------

def data_independent_eps(sigma_mult, q, T_steps, delta):
    try:
        from opacus.accountants import RDPAccountant
        acct = RDPAccountant()
        acct.history = [(sigma_mult, q, T_steps)]
        eps_di, alpha_opt = acct.get_privacy_spent(delta=delta)
        return float(eps_di), float(alpha_opt)
    except Exception:
        rdp = T_steps * (q ** 2) * ALPHA_GRID / (2.0 * sigma_mult ** 2)
        return rdp_to_dp(rdp, ALPHA_GRID, delta)


# ---------------------------------------------------------------------------
# Per-instance certificates
# ---------------------------------------------------------------------------

def compute_certificates(log_df, sigma_use, delta, q):
    """
    Compute norm-based and direction-aware per-instance certificates.

    RDP (subsampled Gaussian, quadratic approximation):
      ε_i(α) = q² * (α/2) * Σ_{t: i∈B_t} ||ḡ_it||² / σ_use²

    Returns dict with arrays indexed by unique example_idx.
    """
    sigma2 = float(sigma_use) ** 2
    q_sq   = float(q) ** 2

    idx_arr = log_df["example_idx"].to_numpy(dtype=np.int32)
    gn_arr  = log_df["grad_norm"].to_numpy(dtype=np.float64)
    in_arr  = log_df["incoherent_norm"].to_numpy(dtype=np.float64)

    unique_idx = np.unique(idx_arr)
    n          = len(unique_idx)
    g2l        = {int(g): l for l, g in enumerate(unique_idx)}
    local_idx  = np.array([g2l[int(g)] for g in idx_arr], dtype=np.int32)

    n_sampled = np.zeros(n, dtype=np.int32)
    sum_gn2   = np.zeros(n, dtype=np.float64)
    sum_in2   = np.zeros(n, dtype=np.float64)
    np.add.at(n_sampled, local_idx, 1)
    np.add.at(sum_gn2,   local_idx, gn_arr ** 2)
    np.add.at(sum_in2,   local_idx, in_arr  ** 2)

    eps_norm_arr = np.zeros(n, dtype=np.float64)
    eps_dir_arr  = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if n_sampled[i] == 0:
            continue
        rdp_norm = q_sq * (ALPHA_GRID / 2.0) * sum_gn2[i] / sigma2
        rdp_dir  = q_sq * (ALPHA_GRID / 2.0) * sum_in2[i] / sigma2
        eps_norm_arr[i], _ = rdp_to_dp(rdp_norm, ALPHA_GRID, delta)
        eps_dir_arr[i],  _ = rdp_to_dp(rdp_dir,  ALPHA_GRID, delta)

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
# β-spectrum computation (rank sweep for R3 runs)
# ---------------------------------------------------------------------------

def compute_beta_spectrum(log_df, sigma_use, delta, q, ranks=BETA_RANKS):
    """
    For each rank r in `ranks`, recompute per-instance direction-aware certificates
    using only the top-r columns of V (V is not reloaded; instead we note that
    incoherent_norm in the log was computed at rank RANK_V=100).

    Since we logged ||g_perp|| at rank 100, we cannot exactly recompute at lower
    ranks from the log alone. Instead, we compute the β-spectrum from the
    aggregate statistics: β̄ = mean(incoherent²) / mean(grad²) is rank-100 β.
    For a theoretical β-spectrum, we would need to re-run with each rank.

    What we can do from the existing log: compute certificates at rank 100 and
    report the aggregate β = sum_inorm2 / sum_gnorm2.
    For the actual β-spectrum, we flag that a re-run with each rank is needed.
    """
    print("  [β-spectrum] Note: recomputing at lower ranks requires per-example "
          "gradient projections. Reporting rank-100 aggregate β here.")
    certs = compute_certificates(log_df, sigma_use, delta, q)
    betas = {}
    for r in ranks:
        if r == 100:
            betas[r] = float(certs["beta_mean"].mean())
        else:
            betas[r] = None   # requires separate per-rank logging
    return betas, certs


# ---------------------------------------------------------------------------
# Load accuracy
# ---------------------------------------------------------------------------

def load_accuracy(tag, train_dir=TRAIN_DIR):
    """
    Load final and best test accuracy from the per-run CSV produced by
    exp_p16_train.py.  Returns (final_acc, best_acc) or (None, None).
    """
    csv_path = os.path.join(train_dir, f"{tag}.csv")
    if not os.path.exists(csv_path):
        return None, None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "test_acc" not in df.columns or df.empty:
            return None, None
        final_acc = float(df["test_acc"].iloc[-1])
        best_acc  = float(df["test_acc"].max())
        return final_acc, best_acc
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Load log
# ---------------------------------------------------------------------------

def load_log(log_path):
    """Load parquet or npz log file."""
    log_npz = log_path.replace(".parquet", ".npz")
    if os.path.exists(log_path):
        try:
            import pandas as pd
            return pd.read_parquet(log_path)
        except Exception as e:
            print(f"  [warn] parquet load failed ({e}), trying npz...")
    if os.path.exists(log_npz):
        import pandas as pd
        data = np.load(log_npz)
        return pd.DataFrame({
            "example_idx":     data["example_idx"],
            "grad_norm":       data["grad_norm"],
            "incoherent_norm": data["incoherent_norm"],
        })
    return None


# ---------------------------------------------------------------------------
# Summarize and print
# ---------------------------------------------------------------------------

def summarize(label, certs, delta, eps_target, tier_local=None):
    en    = certs["eps_norm"]
    ed    = certs["eps_direction"]
    ns    = certs["n_sampled"]
    bm    = certs["beta_mean"]
    ratio = np.where(en > 0, ed / en, np.nan)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  n={len(en)}, ε_target={eps_target}, δ={delta:.2e}")
    print(f"{'='*70}")
    print(f"  Norm-based  (ε^norm):  mean={en.mean():.4f}  "
          f"95th={np.percentile(en,95):.4f}  max={en.max():.4f}")
    print(f"  Dir-aware   (ε^dir):   mean={ed.mean():.4f}  "
          f"95th={np.percentile(ed,95):.4f}  max={ed.max():.4f}")
    ratio_clean = ratio[~np.isnan(ratio)]
    print(f"  Tightening (ε^dir/ε^norm): mean={ratio_clean.mean():.4f}  "
          f"median={np.median(ratio_clean):.4f}  min={ratio_clean.min():.4f}")
    print(f"  β mean:  {bm.mean():.4f}  95th={np.percentile(bm,95):.4f}")
    print(f"  Sampled: mean={ns.mean():.1f}  max={ns.max()}")

    if tier_local is not None and len(tier_local) > 0:
        print("  Per-tier:")
        for t, tname in TIER_NAMES.items():
            mask = tier_local == t
            if mask.sum() == 0:
                continue
            r_t = ratio[mask]; r_t = r_t[~np.isnan(r_t)]
            print(f"    {tname:4s} (n={mask.sum():5d}): ε^norm={en[mask].mean():.4f}  "
                  f"ε^dir={ed[mask].mean():.4f}  tightening={r_t.mean():.4f}")


def build_summary_dict(certs, eps_di, alpha_di, tier_local=None):
    en    = certs["eps_norm"]
    ed    = certs["eps_direction"]
    bm    = certs["beta_mean"]
    ns    = certs["n_sampled"]
    ratio = np.where(en > 0, ed / en, np.nan)
    r_clean = ratio[~np.isnan(ratio)]

    d = {
        "eps_di": float(eps_di), "alpha_di": float(alpha_di),
        "eps_norm_mean": float(en.mean()), "eps_norm_95": float(np.percentile(en, 95)),
        "eps_norm_max":  float(en.max()),
        "eps_dir_mean":  float(ed.mean()),  "eps_dir_95": float(np.percentile(ed, 95)),
        "eps_dir_max":   float(ed.max()),
        "tightening_mean": float(r_clean.mean()) if len(r_clean) > 0 else float("nan"),
        "tightening_med":  float(np.median(r_clean)) if len(r_clean) > 0 else float("nan"),
        "beta_mean": float(bm.mean()), "beta_95": float(np.percentile(bm, 95)),
        "n_sampled_mean": float(ns.mean()),
    }
    if tier_local is not None and len(tier_local) > 0:
        for t, tname in TIER_NAMES.items():
            mask = tier_local == t
            if mask.sum() == 0:
                continue
            r_t = ratio[mask]; r_t = r_t[~np.isnan(r_t)]
            d[f"eps_norm_{tname}"] = float(en[mask].mean())
            d[f"eps_dir_{tname}"]  = float(ed[mask].mean())
            d[f"tight_{tname}"]    = float(r_t.mean()) if len(r_t) > 0 else float("nan")
    return d


# ---------------------------------------------------------------------------
# Save certificates CSV
# ---------------------------------------------------------------------------

def save_certs_csv(certs, cert_path, tier_local=None):
    fieldnames = ["example_idx", "n_sampled", "eps_norm", "eps_direction",
                  "beta_mean", "sum_gnorm2", "sum_inorm2"]
    if tier_local is not None and len(tier_local) > 0:
        fieldnames.append("tier")
    with open(cert_path, "w", newline="") as f:
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
            if tier_local is not None and len(tier_local) > 0:
                row["tier"] = int(tier_local[i])
            w.writerow(row)
    print(f"  [cert] Saved: {cert_path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_certs(run_id, seed, certs, eps_target, tier_local, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    en = certs["eps_norm"]
    ed = certs["eps_direction"]
    bm = certs["beta_mean"]
    is_lt = tier_local is not None and len(tier_local) > 0

    # Scatter: eps_norm vs eps_direction, colored by beta
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(en, ed, c=bm, cmap="plasma_r", s=3, alpha=0.5, vmin=0, vmax=1)
    mn, mx = min(en.min(), ed.min()), max(en.max(), ed.max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, label="y=x (no improvement)")
    ax.set_xlabel("ε^norm"); ax.set_ylabel("ε^dir")
    ax.set_title(f"{run_id} seed={seed}: direction-aware vs norm-based  (ε={eps_target})")
    plt.colorbar(sc, ax=ax, label="β")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"p16_{run_id}_s{seed}_scatter.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [plot] {path}")

    # Histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    if is_lt:
        colors = {0: "steelblue", 1: "orange", 2: "firebrick"}
        for t, tname in TIER_NAMES.items():
            mask = tier_local == t
            if mask.sum() == 0:
                continue
            ax.hist(ed[mask], bins=50, alpha=0.6, label=tname, color=colors[t])
        ax.set_title(f"{run_id} s{seed}: ε^dir by tier  (ε={eps_target})")
    else:
        ax.hist(en, bins=50, alpha=0.6, label="ε^norm", color="steelblue")
        ax.hist(ed, bins=50, alpha=0.6, label="ε^dir",  color="darkorange")
        ax.set_title(f"{run_id} s{seed}: certificates  (ε={eps_target})")
    ax.axvline(eps_target, color="k", ls="--", lw=1, label=f"ε={eps_target}")
    ax.legend(); ax.set_xlabel("per-instance ε"); ax.set_ylabel("count")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    path = os.path.join(out_dir, f"p16_{run_id}_s{seed}_hist.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [plot] {path}")


def plot_beta_spectrum(run_id, seed, betas, out_dir):
    """Plot β^(r) vs rank for R3 runs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ranks = [r for r, v in betas.items() if v is not None]
    vals  = [betas[r] for r in ranks]
    if not ranks:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranks, vals, "o-", color="steelblue")
    ax.set_xlabel("Subspace rank r"); ax.set_ylabel("β^(r) mean")
    ax.set_title(f"{run_id} s{seed}: β-spectrum vs subspace rank")
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.set_ylim(bottom=0); ax.grid(True, alpha=0.3); fig.tight_layout()
    path = os.path.join(out_dir, f"p16_{run_id}_s{seed}_beta_spectrum.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [plot] {path}")


# ---------------------------------------------------------------------------
# Per-run certification
# ---------------------------------------------------------------------------

def certify_run_seed(run_id, cfg, seed, log_dir, cert_dir):
    tag      = f"p16_{run_id}_{cfg['mech']}_{cfg['dataset']}_{cfg['regime']}" \
               f"_eps{cfg['eps']:.0f}_seed{seed}"
    log_path  = os.path.join(log_dir, f"{tag}.parquet")
    meta_path = os.path.join(log_dir, f"{tag}_meta.npz")
    cert_path = os.path.join(cert_dir, f"{tag}_certs.csv")
    summ_path = os.path.join(cert_dir, f"{tag}_summary.json")

    if os.path.exists(cert_path):
        print(f"[P16-cert] {tag}: already certified, loading.")
        try:
            import pandas as pd
            df = pd.read_csv(cert_path)
            certs = {
                "example_idx":   df["example_idx"].to_numpy(np.int32),
                "n_sampled":     df["n_sampled"].to_numpy(np.int32),
                "eps_norm":      df["eps_norm"].to_numpy(np.float64),
                "eps_direction": df["eps_direction"].to_numpy(np.float64),
                "beta_mean":     df["beta_mean"].to_numpy(np.float64),
                "sum_gnorm2":    df["sum_gnorm2"].to_numpy(np.float64),
                "sum_inorm2":    df["sum_inorm2"].to_numpy(np.float64),
            }
            tier_local = df["tier"].to_numpy(np.int32) if "tier" in df.columns else None
            return certs, tier_local
        except Exception:
            pass

    log_df = load_log(log_path)
    if log_df is None:
        print(f"[P16-cert] Log not found for {tag}. Run exp_p16_train.py first.")
        return None, None

    if not os.path.exists(meta_path):
        print(f"[P16-cert] Meta not found: {meta_path}")
        return None, None

    meta      = np.load(meta_path, allow_pickle=True)
    eps       = float(meta["eps"])
    delta     = float(meta["delta"])
    sigma_use = float(meta["sigma_use"])
    sigma_mult= float(meta["sigma_mult"])
    q         = float(meta["q"])
    T_steps   = int(meta["T_steps"])
    n_priv    = int(meta["n_priv"])
    tier_arr  = meta["tier_labels"]
    priv_idx  = meta["priv_idx"]
    has_tiers = len(tier_arr) > 0

    print(f"\n[P16-cert] === {run_id} seed={seed} ===")
    print(f"  log rows={len(log_df):,}  n_priv={n_priv}  eps={eps}  delta={delta:.2e}")
    print(f"  sigma_use={sigma_use:.4f}  q={q:.5f}  T={T_steps}")

    # Data-independent certificate
    eps_di, alpha_di = data_independent_eps(sigma_mult, q, T_steps, delta)
    print(f"  Data-independent: ε={eps_di:.4f}  (target={eps})")

    # Per-instance certificates
    is_r3 = cfg["regime"] == "R3"
    if is_r3:
        betas, certs = compute_beta_spectrum(log_df, sigma_use, delta, q)
    else:
        certs = compute_certificates(log_df, sigma_use, delta, q)
        betas = {}

    # Map tier labels to local positions
    tier_local = None
    if has_tiers:
        global_to_local = {int(priv_idx[i]): i for i in range(len(priv_idx))}
        tier_local = np.array([
            tier_arr[global_to_local[int(gidx)]] if int(gidx) in global_to_local else -1
            for gidx in certs["example_idx"]
        ], dtype=np.int32)

    os.makedirs(cert_dir, exist_ok=True)
    save_certs_csv(certs, cert_path, tier_local)

    label = f"{run_id} seed={seed} — {cfg['regime']} {cfg['mech']} {cfg['dataset']}"
    summarize(label, certs, delta, eps, tier_local)

    # Save summary JSON
    import json
    summ = build_summary_dict(certs, eps_di, alpha_di, tier_local)
    summ["tag"]   = tag
    summ["run_id"]= run_id
    summ["seed"]  = seed
    summ.update({
        "regime": cfg["regime"], "mech": cfg["mech"],
        "dataset": cfg["dataset"], "eps": eps,
    })
    final_acc, best_acc = load_accuracy(tag)
    if final_acc is not None:
        summ["final_acc"] = final_acc
        summ["best_acc"]  = best_acc
    if is_r3 and betas:
        for r, v in betas.items():
            summ[f"beta_rank{r}"] = v
    with open(summ_path, "w") as f:
        json.dump(summ, f, indent=2)
    print(f"  [cert] Summary: {summ_path}")

    # Plots
    plot_certs(run_id, seed, certs, eps, tier_local, cert_dir)
    if is_r3 and betas:
        plot_beta_spectrum(run_id, seed, betas, cert_dir)

    return certs, tier_local


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def aggregate_seeds(run_id, cfg, log_dir, cert_dir):
    """Load all seed certificates for a run and compute median + IQR."""
    all_certs = []
    for seed in range(cfg["n_seeds"]):
        tag      = f"p16_{run_id}_{cfg['mech']}_{cfg['dataset']}_{cfg['regime']}" \
                   f"_eps{cfg['eps']:.0f}_seed{seed}"
        cert_path = os.path.join(cert_dir, f"{tag}_certs.csv")
        if not os.path.exists(cert_path):
            continue
        try:
            import pandas as pd
            df = pd.read_csv(cert_path)
            all_certs.append({
                "eps_norm":      df["eps_norm"].to_numpy(np.float64),
                "eps_direction": df["eps_direction"].to_numpy(np.float64),
                "beta_mean":     df["beta_mean"].to_numpy(np.float64),
            })
        except Exception:
            pass

    if not all_certs:
        return None

    # Compute per-example statistics across seeds (if same examples present)
    # Simpler: aggregate scalar statistics
    norms = [c["eps_norm"].mean()      for c in all_certs]
    dirs  = [c["eps_direction"].mean() for c in all_certs]
    betas = [c["beta_mean"].mean()     for c in all_certs]
    ratios= [d / max(n, 1e-12) for n, d in zip(norms, dirs)]

    # Collect per-seed accuracy
    best_accs = []
    for seed in range(cfg["n_seeds"]):
        tag = (f"p16_{run_id}_{cfg['mech']}_{cfg['dataset']}_{cfg['regime']}"
               f"_eps{cfg['eps']:.0f}_seed{seed}")
        _, ba = load_accuracy(tag)
        if ba is not None:
            best_accs.append(ba)

    print(f"\n[P16-cert] {run_id}: {len(all_certs)}/{cfg['n_seeds']} seeds aggregated")
    print(f"  ε^norm  median={np.median(norms):.4f}  IQR=[{np.percentile(norms,25):.4f}, {np.percentile(norms,75):.4f}]")
    print(f"  ε^dir   median={np.median(dirs):.4f}   IQR=[{np.percentile(dirs,25):.4f}, {np.percentile(dirs,75):.4f}]")
    print(f"  tighten median={np.median(ratios):.4f} IQR=[{np.percentile(ratios,25):.4f}, {np.percentile(ratios,75):.4f}]")
    print(f"  β       median={np.median(betas):.4f}")
    if best_accs:
        print(f"  best_acc median={np.median(best_accs):.4f}  "
              f"IQR=[{np.percentile(best_accs,25):.4f}, {np.percentile(best_accs,75):.4f}]")

    result = {
        "n_seeds_present": len(all_certs),
        "eps_norm_median": float(np.median(norms)),
        "eps_norm_iqr":    [float(np.percentile(norms, 25)), float(np.percentile(norms, 75))],
        "eps_dir_median":  float(np.median(dirs)),
        "eps_dir_iqr":     [float(np.percentile(dirs, 25)), float(np.percentile(dirs, 75))],
        "tightening_median": float(np.median(ratios)),
        "beta_median":     float(np.median(betas)),
    }
    if best_accs:
        result["best_acc_median"] = float(np.median(best_accs))
        result["best_acc_iqr"]    = [float(np.percentile(best_accs, 25)),
                                     float(np.percentile(best_accs, 75))]
    return result


# ---------------------------------------------------------------------------
# Cross-run summary table
# ---------------------------------------------------------------------------

def print_table(all_stats):
    """Print Table 1 from the spec across all completed runs."""
    header = (f"{'Run':5s}  {'Regime':6s}  {'Dataset':14s}  {'Mech':8s}  "
              f"{'ε':4s}  {'ε^norm med':10s}  {'ε^dir med':9s}  "
              f"{'tighten':8s}  {'β med':6s}  {'best_acc':8s}  {'n_seeds':7s}")
    print(f"\n{'='*len(header)}")
    print("  Phase 16 Cross-Run Summary Table")
    print(f"{'='*len(header)}")
    print(f"  {header}")
    print(f"  {'-'*len(header)}")
    for run_id, stats in sorted(all_stats.items()):
        if stats is None:
            continue
        cfg = RUN_MATRIX.get(run_id, {})
        best_acc_str = (f"{stats['best_acc_median']:.4f}"
                        if "best_acc_median" in stats else "  N/A  ")
        print(f"  {run_id:5s}  {cfg.get('regime','?'):6s}  "
              f"{cfg.get('dataset','?'):14s}  {cfg.get('mech','?'):8s}  "
              f"{cfg.get('eps',0):4.0f}  "
              f"{stats['eps_norm_median']:10.4f}  "
              f"{stats['eps_dir_median']:9.4f}  "
              f"{stats['tightening_median']:8.4f}  "
              f"{stats['beta_median']:6.4f}  "
              f"{best_acc_str:8s}  "
              f"{stats['n_seeds_present']:7d}")
    print(f"{'='*len(header)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 16 certificate computation")
    parser.add_argument("--run",      type=str, default=None, help="Run ID (e.g. H1, M3)")
    parser.add_argument("--seed",     type=int, default=None, help="Single seed (default: all)")
    parser.add_argument("--all",      action="store_true",    help="Process all runs")
    parser.add_argument("--table",    action="store_true",    help="Print summary table")
    parser.add_argument("--log_dir",  type=str, default=LOG_DIR)
    parser.add_argument("--cert_dir", type=str, default=CERT_DIR)
    args = parser.parse_args()

    os.makedirs(args.cert_dir, exist_ok=True)

    # Determine runs
    if args.run:
        run_ids = [args.run]
    elif args.all or args.table:
        run_ids = list(RUN_MATRIX.keys())
    else:
        parser.print_help()
        print("\nNo run specified. Use --run <ID>, --all, or --table.")
        return

    all_stats = {}

    for run_id in run_ids:
        if run_id not in RUN_MATRIX:
            print(f"[P16-cert] Unknown run: {run_id}")
            continue
        cfg = RUN_MATRIX[run_id]

        if args.seed is not None:
            certify_run_seed(run_id, cfg, args.seed, args.log_dir, args.cert_dir)
        else:
            for seed in range(cfg["n_seeds"]):
                certify_run_seed(run_id, cfg, seed, args.log_dir, args.cert_dir)
            # Aggregate across seeds
            stats = aggregate_seeds(run_id, cfg, args.log_dir, args.cert_dir)
            all_stats[run_id] = stats

            # Save aggregated stats
            if stats is not None:
                import json
                agg_path = os.path.join(args.cert_dir, f"p16_{run_id}_aggregate.json")
                cfg_info = {k: str(v) for k, v in cfg.items()}
                with open(agg_path, "w") as f:
                    json.dump({"run_id": run_id, "cfg": cfg_info, **stats}, f, indent=2)
                print(f"  [cert] Aggregate saved: {agg_path}")

    if args.table or (not args.seed and len(all_stats) > 1):
        # Load any missing aggregate jsons
        import json
        for run_id in run_ids:
            if run_id not in all_stats or all_stats[run_id] is None:
                agg_path = os.path.join(args.cert_dir, f"p16_{run_id}_aggregate.json")
                if os.path.exists(agg_path):
                    try:
                        with open(agg_path) as f:
                            d = json.load(f)
                        all_stats[run_id] = d
                    except Exception:
                        pass
        print_table(all_stats)

    print(f"\n[P16-cert] Done. Results in {args.cert_dir}")


if __name__ == "__main__":
    main()
