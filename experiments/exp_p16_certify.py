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

# LiRA run matrix (Tier 4)
LIRA_MATRIX = {
    "L1": dict(dataset="cifar10",      regime="R2", mech="vanilla", eps=8.0,
               batch=5000, n_shadows=16, n_targets=200),
    "L2": dict(dataset="cifar10_lt50", regime="R2", mech="vanilla", eps=8.0,
               batch=5000, n_shadows=16, n_targets=200),
}

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

def compute_certificates(log_df, sigma_use, delta, q, lambdas=None):
    """
    Compute norm-based and direction-aware per-instance certificates.

    RDP (subsampled Gaussian, quadratic approximation):
      ε_i(α) = q² * (α/2) * Σ_{t: i∈B_t} d²_it

    where d²_it is the per-step effective squared distance:
      - Norm-based:  d²_it = ||ḡ_it||² / σ_use²
      - Direction-aware (Woodbury, rigorous upper bound on d_eff^(r)²):
            d²_it = Σ_k (g_proj_k)² / (σ_use² + λ_k)  +  ||g_perp||² / σ_use²

        This is equivalent to:
            d²_it = ||ḡ_it||² / σ_use²
                    - Σ_k (g_proj_k)² λ_k / (σ_use² (σ_use² + λ_k))

        At r=0 it reduces to the norm-based bound.  No eigenspace assumption needed.

    Parameters
    ----------
    log_df    : DataFrame with columns example_idx, grad_norm, incoherent_norm,
                and optionally g_proj_0 … g_proj_{r-1}
    sigma_use : actual noise std = sigma_mult * C  (σ_use² = σ_mult² C²)
    delta     : DP delta
    q         : subsampling rate
    lambdas   : (r,) array of PCA eigenvalues Λ, loaded from meta.  When None
                (or g_proj columns absent), falls back to incoherent-norm-only
                formula (not rigorous — kept for backward compatibility).

    Returns dict with arrays indexed by unique example_idx.
    """
    sigma2 = float(sigma_use) ** 2   # σ_use² = σ_mult² C²
    q_sq   = float(q) ** 2

    idx_arr = log_df["example_idx"].to_numpy(dtype=np.int32)
    gn_arr  = log_df["grad_norm"].to_numpy(dtype=np.float64)
    in_arr  = log_df["incoherent_norm"].to_numpy(dtype=np.float64)

    unique_idx = np.unique(idx_arr)
    n          = len(unique_idx)
    g2l        = {int(g): l for l, g in enumerate(unique_idx)}
    local_idx  = np.array([g2l[int(g)] for g in idx_arr], dtype=np.int32)

    # ---- Try to load g_proj projections for Woodbury formula ----------------
    use_woodbury = False
    rank         = 0
    sum_gproj2   = None   # (n, rank) — accumulated Σ_t (g_proj_k)² per example

    if lambdas is not None and len(lambdas) > 0:
        rank      = len(lambdas)
        lambdas   = np.asarray(lambdas, dtype=np.float64)
        proj_cols = [f"g_proj_{k}" for k in range(rank)]
        if all(c in log_df.columns for c in proj_cols):
            g_proj_mat  = log_df[proj_cols].to_numpy(dtype=np.float64)  # (N_rows, rank)
            sum_gproj2  = np.zeros((n, rank), dtype=np.float64)
            for k in range(rank):
                np.add.at(sum_gproj2[:, k], local_idx, g_proj_mat[:, k] ** 2)
            use_woodbury = True
        else:
            missing = [c for c in proj_cols if c not in log_df.columns]
            print(f"  [cert] g_proj columns missing in log ({len(missing)}/{rank} absent); "
                  f"falling back to incoherent-norm-only formula (not rigorous).")

    if not use_woodbury:
        print("  [cert] Woodbury formula NOT used — using incoherent-norm proxy "
              "(lower bound on d_eff; certificates may be overly optimistic).")

    # ---- Accumulate per-example sums ----------------------------------------
    n_sampled = np.zeros(n, dtype=np.int32)
    sum_gn2   = np.zeros(n, dtype=np.float64)
    sum_in2   = np.zeros(n, dtype=np.float64)
    np.add.at(n_sampled, local_idx, 1)
    np.add.at(sum_gn2,   local_idx, gn_arr ** 2)
    np.add.at(sum_in2,   local_idx, in_arr  ** 2)

    # ---- Woodbury sum: Σ_t [ Σ_k (g_proj_k)²/(σ²+λ_k)  +  ||g_perp||²/σ² ] --
    # Equivalently: sum_gn2/σ² - Σ_k sum_gproj2[:,k]*λ_k/(σ²(σ²+λ_k))
    if use_woodbury:
        # (n,) vector: Σ_k sum_gproj2[i,k] / (sigma2 + lambdas[k])
        proj_term    = np.dot(sum_gproj2, 1.0 / (sigma2 + lambdas))  # (n,)
        incoher_term = sum_in2 / sigma2                               # (n,)
        sum_woodbury2 = proj_term + incoher_term
    else:
        # Fallback: only the incoherent term (not a rigorous upper bound)
        sum_woodbury2 = sum_in2 / sigma2

    # ---- RDP → ε conversion -------------------------------------------------
    eps_norm_arr = np.zeros(n, dtype=np.float64)
    eps_dir_arr  = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if n_sampled[i] == 0:
            continue
        rdp_norm = q_sq * (ALPHA_GRID / 2.0) * sum_gn2[i] / sigma2
        # Direction-aware: uses full Woodbury formula when available
        rdp_dir  = q_sq * (ALPHA_GRID / 2.0) * sum_woodbury2[i]
        eps_norm_arr[i], _ = rdp_to_dp(rdp_norm, ALPHA_GRID, delta)
        eps_dir_arr[i],  _ = rdp_to_dp(rdp_dir,  ALPHA_GRID, delta)

    # β = incoherent fraction of gradient energy (unchanged definition)
    beta_mean = np.where(sum_gn2 > 0, sum_in2 / sum_gn2, 0.0)

    return {
        "example_idx":    unique_idx,
        "n_sampled":      n_sampled,
        "sum_gnorm2":     sum_gn2,
        "sum_inorm2":     sum_in2,
        "sum_woodbury2":  sum_woodbury2,
        "eps_norm":       eps_norm_arr,
        "eps_direction":  eps_dir_arr,
        "beta_mean":      beta_mean,
        "woodbury_used":  use_woodbury,
        "woodbury_rank":  rank,
    }


# ---------------------------------------------------------------------------
# β-spectrum computation (rank sweep for R3 runs)
# ---------------------------------------------------------------------------

def compute_beta_spectrum(log_df, sigma_use, delta, q, lambdas=None, ranks=BETA_RANKS):
    """
    For each rank r in `ranks`, compute direction-aware certificates using the
    top-r PCA components via the Woodbury formula.

    When g_proj columns and lambdas are present (Woodbury path):
      - For each rank r, truncate lambdas to the top-r eigenvalues and use only
        g_proj_0 … g_proj_{r-1}.  This gives a valid upper bound at each rank:
        d_eff^(r)² ≥ d_eff^(r')²  for r > r', so eps^dir decreases (tightens)
        as r grows.
      - β^(r) is defined as the mean of (sum_woodbury2^(r) / (sum_gnorm2/sigma2)).

    When g_proj columns are absent (fallback):
      - Only rank-100 β (the incoherent fraction) is reported; lower ranks need
        a separate training run with smaller rank logged.
    """
    sigma2     = float(sigma_use) ** 2
    has_proj   = (lambdas is not None and len(lambdas) > 0 and
                  f"g_proj_0" in log_df.columns)

    if not has_proj:
        print("  [β-spectrum] g_proj columns absent — reporting rank-100 β only.")
        certs = compute_certificates(log_df, sigma_use, delta, q, lambdas=lambdas)
        betas = {}
        for r in ranks:
            if r == max(ranks):
                betas[r] = float(certs["beta_mean"].mean())
            else:
                betas[r] = None
        return betas, certs

    # Full Woodbury β-spectrum: compute certs at each rank
    # Use all components for the "full" cert returned alongside betas
    certs = compute_certificates(log_df, sigma_use, delta, q, lambdas=lambdas)
    betas = {}
    for r in ranks:
        r_use = min(r, len(lambdas))
        lam_r = lambdas[:r_use]
        # Reuse accumulation from certs; rebuild sum_gproj2 at rank r
        idx_arr   = log_df["example_idx"].to_numpy(dtype=np.int32)
        unique_idx = certs["example_idx"]
        n          = len(unique_idx)
        g2l        = {int(g): l for l, g in enumerate(unique_idx)}
        local_idx  = np.array([g2l[int(g)] for g in idx_arr], dtype=np.int32)

        sum_in2 = certs["sum_inorm2"]   # always at full rank (perp to all r cols)
        sum_gn2 = certs["sum_gnorm2"]

        if r_use > 0:
            proj_cols  = [f"g_proj_{k}" for k in range(r_use)]
            g_proj_mat = log_df[proj_cols].to_numpy(dtype=np.float64)
            sgp2       = np.zeros((n, r_use), dtype=np.float64)
            for k in range(r_use):
                np.add.at(sgp2[:, k], local_idx, g_proj_mat[:, k] ** 2)
            proj_term  = np.dot(sgp2, 1.0 / (sigma2 + lam_r))
        else:
            proj_term  = np.zeros(n, dtype=np.float64)

        sw2   = proj_term + sum_in2 / sigma2
        norm2 = np.where(sum_gn2 > 0, sum_gn2 / sigma2, 1.0)
        betas[r] = float(np.mean(sw2 / norm2))

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
    """Load scalar log (parquet/npz) and merge g_proj companion file.

    The scalar log (parquet or npz) contains: example_idx, grad_norm, incoherent_norm.
    The companion _gproj.npy (written by exp_p16_train.py) contains the per-step PCA
    projection vectors g_proj = V^T g_clip, shape (N_rows, rank).  These are merged
    as columns g_proj_0 … g_proj_{rank-1} so that compute_certificates can use the
    Woodbury formula.

    Falls back gracefully if either file is missing.
    """
    import pandas as pd
    log_npz      = log_path.replace(".parquet", ".npz")
    gproj_path   = log_path.replace(".parquet", "_gproj.npy")

    df = None
    if os.path.exists(log_path):
        try:
            df = pd.read_parquet(log_path)
        except Exception as e:
            print(f"  [warn] parquet load failed ({e}), trying npz...")

    if df is None and os.path.exists(log_npz):
        data    = np.load(log_npz)
        df_dict = {
            "example_idx":     data["example_idx"],
            "grad_norm":       data["grad_norm"],
            "incoherent_norm": data["incoherent_norm"],
        }
        # Legacy npz that already embedded g_proj (older format)
        if "g_proj" in data:
            g_proj = data["g_proj"]
            for k in range(g_proj.shape[1]):
                df_dict[f"g_proj_{k}"] = g_proj[:, k]
        df = pd.DataFrame(df_dict)

    if df is None:
        return None

    # Merge companion _gproj.npy when g_proj columns are not already present
    if "g_proj_0" not in df.columns and os.path.exists(gproj_path):
        try:
            g_proj = np.load(gproj_path)        # (N_rows, rank) float32
            if len(g_proj) == len(df):
                for k in range(g_proj.shape[1]):
                    df[f"g_proj_{k}"] = g_proj[:, k]
                print(f"  [log] Merged g_proj ({g_proj.shape[1]} cols, "
                      f"{g_proj.shape[0]:,} rows) from {gproj_path}")
            else:
                print(f"  [warn] g_proj row count mismatch "
                      f"({len(g_proj):,} vs {len(df):,} in scalar log); skipping Woodbury")
        except Exception as e:
            print(f"  [warn] g_proj load failed ({e}); Woodbury will not be used")
    elif "g_proj_0" not in df.columns:
        print(f"  [warn] No g_proj companion found at {gproj_path}; "
              f"Woodbury formula will fall back to incoherent-norm proxy")

    return df


# ---------------------------------------------------------------------------
# Summarize and print
# ---------------------------------------------------------------------------

def summarize(label, certs, delta, eps_target, tier_local=None):
    en    = certs["eps_norm"]
    ed    = certs["eps_direction"]
    ns    = certs["n_sampled"]
    bm    = certs["beta_mean"]
    ratio = np.where(en > 0, ed / en, np.nan)
    woodbury_used = certs.get("woodbury_used", False)
    woodbury_rank = certs.get("woodbury_rank", 0)

    method_tag = (f"Woodbury r={woodbury_rank} [rigorous upper bound]"
                  if woodbury_used
                  else "incoherent-norm proxy [NOT rigorous — lower bound on d_eff]")

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  n={len(en)}, ε_target={eps_target}, δ={delta:.2e}")
    print(f"  ε^dir method: {method_tag}")
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
        # Woodbury metadata: indicates whether ε^dir used the rigorous formula
        "woodbury_used": bool(certs.get("woodbury_used", False)),
        "woodbury_rank": int(certs.get("woodbury_rank", 0)),
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
                  "beta_mean", "sum_gnorm2", "sum_inorm2", "sum_woodbury2"]
    if tier_local is not None and len(tier_local) > 0:
        fieldnames.append("tier")
    woodbury_used = certs.get("woodbury_used", False)
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
                "sum_woodbury2": f"{certs['sum_woodbury2'][i]:.6f}",
            }
            if tier_local is not None and len(tier_local) > 0:
                row["tier"] = int(tier_local[i])
            w.writerow(row)
    rigour = "Woodbury" if woodbury_used else "incoherent-norm proxy (not rigorous)"
    print(f"  [cert] Saved: {cert_path}  [{rigour}]")


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
                "sum_woodbury2": (df["sum_woodbury2"].to_numpy(np.float64)
                                  if "sum_woodbury2" in df.columns
                                  else df["sum_inorm2"].to_numpy(np.float64)),
                # Cached certs don't store the Woodbury flag; mark as unknown
                "woodbury_used": "sum_woodbury2" in df.columns,
                "woodbury_rank": 0,
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

    # PCA eigenvalues for Woodbury formula (stored by exp_p16_train.py as "lambdas")
    lambdas = meta["lambdas"].astype(np.float64) if "lambdas" in meta else None
    if lambdas is not None:
        print(f"  Woodbury: loaded {len(lambdas)} PCA eigenvalues from meta "
              f"(λ_max={lambdas[0]:.4g}, λ_min={lambdas[-1]:.4g})")
    else:
        print("  Woodbury: no lambdas in meta — will use incoherent-norm fallback.")

    print(f"\n[P16-cert] === {run_id} seed={seed} ===")
    print(f"  log rows={len(log_df):,}  n_priv={n_priv}  eps={eps}  delta={delta:.2e}")
    print(f"  sigma_use={sigma_use:.4f}  q={q:.5f}  T={T_steps}")

    # Data-independent certificate
    eps_di, alpha_di = data_independent_eps(sigma_mult, q, T_steps, delta)
    print(f"  Data-independent: ε={eps_di:.4f}  (target={eps})")

    # Per-instance certificates
    is_r3 = cfg["regime"] == "R3"
    if is_r3:
        betas, certs = compute_beta_spectrum(log_df, sigma_use, delta, q,
                                             lambdas=lambdas)
    else:
        certs = compute_certificates(log_df, sigma_use, delta, q,
                                     lambdas=lambdas)
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
# LiRA attack analysis (Tier 4: L1, L2)
# ---------------------------------------------------------------------------

def _load_shadow_loss(model_path, target_loader, device):
    """
    Load a shadow model and return per-target cross-entropy losses (numpy array).
    Falls back to a simple WRN-28-2 inference; caller is responsible for building
    the model skeleton before calling this helper.
    """
    import torch
    import torch.nn.functional as F

    # Dynamically import the model builder from the train script
    train_mod_path = os.path.join(os.path.dirname(__file__), "exp_p16_train.py")
    import importlib.util
    spec_imp = importlib.util.spec_from_file_location("exp_p16_train", train_mod_path)
    train_mod = importlib.util.module_from_spec(spec_imp)
    spec_imp.loader.exec_module(train_mod)

    # build model: L1/L2 use regime=R2 → WRN-28-2 + GroupNorm, 10 classes
    model = train_mod.make_model("R2", 10, "cifar10").to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    losses = []
    with torch.no_grad():
        for x, y in target_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="none")
            losses.append(loss.cpu().numpy())
    return np.concatenate(losses)


def run_lira_attack(run_id, cfg, log_dir, train_dir, cert_dir, device_str="cpu"):
    """
    Offline LiRA (Carlini et al. 2022) for a completed Tier-4 run.

    For each target example i:
      - Collect losses under the n_shadows shadow models where i was IN / OUT.
      - Fit Gaussian to each set, compute likelihood ratio → LiRA score s_i.
      - Compute AUC, TPR@0.1%FPR, TPR@1%FPR.
      - Correlate s_i with ε^norm and ε^dir from the corresponding H-run certs.

    Saves: lira_{run_id}_scores.npz, lira_{run_id}_summary.json
    """
    import json

    n_shadows  = cfg["n_shadows"]
    n_targets  = cfg["n_targets"]
    dataset    = cfg["dataset"]

    target_npy = os.path.join(log_dir, f"lira_{run_id}_target_idx.npy")
    summ_path  = os.path.join(cert_dir, f"lira_{run_id}_summary.json")
    score_path = os.path.join(cert_dir, f"lira_{run_id}_scores.npz")

    if not os.path.exists(target_npy):
        print(f"[P16-LiRA] Target index not found: {target_npy}  (run exp_p16_train.py --run {run_id} first)")
        return None

    target_global_idx = np.load(target_npy)
    n_found = min(len(target_global_idx), n_targets)
    target_global_idx = target_global_idx[:n_found]

    # Build shadow shadow-loss matrix: (n_shadows, n_targets)
    # and membership matrix: (n_shadows, n_targets)  1=in, 0=out
    shadow_losses  = np.full((n_shadows, n_found), np.nan)
    member_matrix  = np.zeros((n_shadows, n_found), dtype=np.int8)

    # We need to run inference on target examples — build a minimal dataset loader.
    # Targets live in the private split (global indices into CIFAR-10 train).
    try:
        import torch
        import torchvision
        import torchvision.transforms as T
        device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
        tf   = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        cifar_train = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=False, transform=tf)

        # Subset to target examples
        from torch.utils.data import Subset, DataLoader
        target_ds     = Subset(cifar_train, target_global_idx.tolist())
        target_loader = DataLoader(target_ds, batch_size=64, shuffle=False,
                                   num_workers=0, pin_memory=False)
        # target labels for cross-entropy
        target_labels = np.array([cifar_train.targets[i] for i in target_global_idx])

    except Exception as e:
        print(f"[P16-LiRA] Dataset load failed: {e}")
        return None

    print(f"\n[P16-LiRA] === {run_id}: scoring {n_found} targets over {n_shadows} shadows ===")

    # Build global→local index mapping from the corresponding H-run meta file.
    # in_mask is indexed by local private index, but target_global_idx are global.
    cert_run = "H7" if run_id == "L1" else "H8"
    meta_tag  = (f"p16_{cert_run}_vanilla_{dataset}_R2"
                 f"_eps{int(cfg['eps'])}_seed0")
    meta_path = os.path.join(log_dir, f"{meta_tag}_meta.npz")
    global_to_local = None
    if os.path.exists(meta_path):
        try:
            meta_np = np.load(meta_path, allow_pickle=True)
            priv_idx = meta_np["priv_idx"]
            global_to_local = {int(g): l for l, g in enumerate(priv_idx)}
            print(f"  [LiRA] Loaded priv_idx ({len(priv_idx)} entries) from {meta_tag}_meta.npz")
        except Exception as e:
            print(f"  [LiRA] Could not load priv_idx from meta: {e}")
    else:
        print(f"  [LiRA] Meta file not found ({meta_path}); "
              f"run --run {cert_run} first or ensure meta is present.")

    if global_to_local is None:
        print(f"[P16-LiRA] Cannot map target global indices to local — aborting.")
        return None

    # Local indices for each target (in the private split)
    target_local_idx = np.array(
        [global_to_local.get(int(g), -1) for g in target_global_idx], dtype=np.int32)
    valid_target_mask = target_local_idx >= 0
    if not valid_target_mask.all():
        n_missing = (~valid_target_mask).sum()
        print(f"  [LiRA] Warning: {n_missing}/{n_found} targets not found in priv_idx; "
              f"they will have NaN scores.")

    if os.path.exists(score_path):
        print(f"[P16-LiRA] Loading cached scores: {score_path}")
        data = np.load(score_path)
        shadow_losses = data["shadow_losses"]
        member_matrix = data["member_matrix"]
    else:
        for sid in range(n_shadows):
            tag_s   = f"p16_lira_{run_id}_shadow{sid:03d}"
            model_p = os.path.join(train_dir, f"{tag_s}_final.pt")
            mem_p   = os.path.join(log_dir,   f"{tag_s}_member_mask.npy")

            if not os.path.exists(model_p):
                print(f"  [LiRA] shadow {sid}: model not found, skipping.")
                continue
            if not os.path.exists(mem_p):
                print(f"  [LiRA] shadow {sid}: membership mask not found, skipping.")
                continue

            in_mask = np.load(mem_p)  # shape (n_priv,) bool — indexed by local idx
            # Map targets via local indices; unresolvable targets stay 0
            local_valid = target_local_idx[valid_target_mask]
            member_matrix[sid, valid_target_mask] = in_mask[local_valid].astype(np.int8)

            try:
                losses = _load_shadow_loss(model_p, target_loader, device)
                shadow_losses[sid] = losses[:n_found]
                print(f"  [LiRA] shadow {sid}: loss mean={losses.mean():.4f}")
            except Exception as e:
                print(f"  [LiRA] shadow {sid}: inference failed ({e})")

        np.savez(score_path, shadow_losses=shadow_losses, member_matrix=member_matrix,
                 target_global_idx=target_global_idx)
        print(f"  [LiRA] Scores cached: {score_path}")

    # Compute LiRA scores via offline Gaussian likelihood ratio
    lira_scores   = np.full(n_found, np.nan)
    true_member   = np.zeros(n_found, dtype=np.int8)  # ground truth for target models

    # The "target" model is the model that was actually trained on the full private set.
    # For offline LiRA we treat: IN shadows give "in" distribution, OUT shadows give "out".
    # For each target example i:
    #   in_losses  = shadow_losses[member_matrix[:,i]==1, i]
    #   out_losses = shadow_losses[member_matrix[:,i]==0, i]
    # LiRA score = log p(loss_i | in) - log p(loss_i | out)
    # Since we don't have a separate target model inference, we use leave-one-out:
    # treat each shadow as both target and reference for the other shadows.
    # Simplified: use mean loss over all shadows as a proxy for the "target" loss.

    for i in range(n_found):
        in_idx  = np.where(member_matrix[:, i] == 1)[0]
        out_idx = np.where(member_matrix[:, i] == 0)[0]
        if len(in_idx) < 2 or len(out_idx) < 2:
            continue

        in_losses  = shadow_losses[in_idx,  i]
        out_losses = shadow_losses[out_idx, i]
        valid_in   = in_losses[~np.isnan(in_losses)]
        valid_out  = out_losses[~np.isnan(out_losses)]
        if len(valid_in) < 2 or len(valid_out) < 2:
            continue

        mu_in,  std_in  = valid_in.mean(),  valid_in.std()  + 1e-8
        mu_out, std_out = valid_out.mean(), valid_out.std() + 1e-8

        # Proxy target loss = mean of out-shadows (conservative; typical offline setup)
        loss_i = valid_out.mean()
        log_p_in  = -0.5 * ((loss_i - mu_in)  / std_in)  ** 2 - np.log(std_in)
        log_p_out = -0.5 * ((loss_i - mu_out) / std_out) ** 2 - np.log(std_out)
        lira_scores[i] = log_p_in - log_p_out

        # True membership: most in-shadows → member
        true_member[i] = 1 if len(valid_in) > len(valid_out) else 0

    valid_mask = ~np.isnan(lira_scores)
    n_valid    = valid_mask.sum()

    if n_valid < 5:
        print(f"[P16-LiRA] Too few valid scores ({n_valid}); cannot compute metrics.")
        return None

    scores_v = lira_scores[valid_mask]
    labels_v = true_member[valid_mask]

    # AUC
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        auc = roc_auc_score(labels_v, scores_v)
        fprs, tprs, _ = roc_curve(labels_v, scores_v)
        # TPR at FPR ≤ 0.1% and 1%
        tpr_01 = float(tprs[np.searchsorted(fprs, 0.001, side="right") - 1])
        tpr_1  = float(tprs[np.searchsorted(fprs, 0.01,  side="right") - 1])
    except Exception as e:
        print(f"  [LiRA] sklearn not available or error: {e}")
        auc, tpr_01, tpr_1 = float("nan"), float("nan"), float("nan")

    # Correlation with direction-aware certificates
    # Try to load the H7/H8 certs that correspond to L1/L2 respectively
    cert_run = "H7" if run_id == "L1" else "H8"
    cert_tag = (f"p16_{cert_run}_vanilla_{dataset}_R2"
                f"_eps{int(cfg['eps'])}_seed0")
    cert_csv = os.path.join(cert_dir, f"{cert_tag}_certs.csv")

    spearman_norm, spearman_dir = float("nan"), float("nan")
    if os.path.exists(cert_csv):
        try:
            import pandas as pd
            from scipy.stats import spearmanr
            cdf = pd.read_csv(cert_csv)
            # Align by example_idx
            cdf = cdf.set_index("example_idx")
            matched_norm = []
            matched_dir  = []
            matched_lira = []
            for ii, gidx in enumerate(target_global_idx[valid_mask]):
                if int(gidx) in cdf.index:
                    matched_norm.append(cdf.loc[int(gidx), "eps_norm"])
                    matched_dir.append( cdf.loc[int(gidx), "eps_direction"])
                    matched_lira.append(scores_v[ii] if ii < len(scores_v) else float("nan"))
            if len(matched_norm) >= 5:
                spearman_norm = float(spearmanr(matched_lira, matched_norm).statistic)
                spearman_dir  = float(spearmanr(matched_lira, matched_dir).statistic)
        except Exception as e:
            print(f"  [LiRA] Correlation failed: {e}")
    else:
        print(f"  [LiRA] Cert CSV not found for correlation ({cert_csv})")

    summary = {
        "run_id":         run_id,
        "dataset":        dataset,
        "n_shadows":      n_shadows,
        "n_targets":      n_found,
        "n_valid":        int(n_valid),
        "auc":            float(auc),
        "tpr_at_fpr01":   tpr_01,
        "tpr_at_fpr1":    tpr_1,
        "lira_score_mean":float(scores_v.mean()),
        "lira_score_std": float(scores_v.std()),
        "spearman_lira_vs_eps_norm": spearman_norm,
        "spearman_lira_vs_eps_dir":  spearman_dir,
        "cert_run_used":  cert_run,
    }

    import json
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [LiRA] Summary: {summ_path}")

    print(f"\n  AUC={auc:.4f}  TPR@0.1%FPR={tpr_01:.4f}  TPR@1%FPR={tpr_1:.4f}")
    print(f"  Spearman(LiRA, ε^norm)={spearman_norm:.4f}  "
          f"Spearman(LiRA, ε^dir)={spearman_dir:.4f}")
    return summary


def print_lira_table(lira_stats):
    """Print Table 4 (LiRA summary) from the spec."""
    header = (f"{'Run':5s}  {'Dataset':14s}  {'Shadows':7s}  {'Targets':7s}  "
              f"{'AUC':6s}  {'TPR@0.1%':8s}  {'TPR@1%':7s}  "
              f"{'ρ(LiRA,ε^norm)':14s}  {'ρ(LiRA,ε^dir)':13s}")
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("  Phase 16 — Table 4: LiRA Validation Summary")
    print(sep)
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for run_id, s in sorted(lira_stats.items()):
        if s is None:
            continue
        def _fmt(v, fmt=".4f"):
            return f"{v:{fmt}}" if not (isinstance(v, float) and np.isnan(v)) else "  N/A "
        print(f"  {run_id:5s}  {s['dataset']:14s}  {s['n_shadows']:7d}  "
              f"{s['n_valid']:7d}  "
              f"{_fmt(s['auc']):6s}  "
              f"{_fmt(s['tpr_at_fpr01']):8s}  "
              f"{_fmt(s['tpr_at_fpr1']):7s}  "
              f"{_fmt(s['spearman_lira_vs_eps_norm']):14s}  "
              f"{_fmt(s['spearman_lira_vs_eps_dir']):13s}")
    print(sep)
    print("  ρ = Spearman correlation between LiRA score and per-instance ε.")
    print("  If ε^dir is a better predictor of attack success, ρ(LiRA,ε^dir) > ρ(LiRA,ε^norm).")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 16 certificate computation")
    parser.add_argument("--run",      type=str, default=None, help="Run ID (e.g. H1, M3)")
    parser.add_argument("--seed",     type=int, default=None, help="Single seed (default: all)")
    parser.add_argument("--all",      action="store_true",    help="Process all runs")
    parser.add_argument("--table",    action="store_true",    help="Print summary table")
    parser.add_argument("--lira",     action="store_true",    help="Run LiRA attack analysis (L1, L2)")
    parser.add_argument("--lira_run", type=str, default=None, help="Single LiRA run (L1 or L2)")
    parser.add_argument("--device",   type=str, default="cpu", help="Device for LiRA inference (e.g. cuda:0)")
    parser.add_argument("--log_dir",  type=str, default=LOG_DIR)
    parser.add_argument("--train_dir",type=str, default=TRAIN_DIR)
    parser.add_argument("--cert_dir", type=str, default=CERT_DIR)
    args = parser.parse_args()

    os.makedirs(args.cert_dir, exist_ok=True)

    # ---- LiRA analysis path ------------------------------------------------
    if args.lira or args.lira_run:
        lira_ids = ([args.lira_run] if args.lira_run else list(LIRA_MATRIX.keys()))
        lira_stats = {}
        for run_id in lira_ids:
            if run_id not in LIRA_MATRIX:
                print(f"[P16-cert] Unknown LiRA run: {run_id}  (valid: {list(LIRA_MATRIX)})")
                continue
            cfg = LIRA_MATRIX[run_id]
            summ = run_lira_attack(run_id, cfg, args.log_dir, args.train_dir,
                                   args.cert_dir, device_str=args.device)
            lira_stats[run_id] = summ

        # Also try to load any previously cached summaries
        import json
        for run_id in LIRA_MATRIX:
            if run_id not in lira_stats:
                sp = os.path.join(args.cert_dir, f"lira_{run_id}_summary.json")
                if os.path.exists(sp):
                    try:
                        with open(sp) as f:
                            lira_stats[run_id] = json.load(f)
                    except Exception:
                        pass

        print_lira_table(lira_stats)
        print(f"\n[P16-cert] LiRA done. Results in {args.cert_dir}")
        return

    # ---- Certificate analysis path -----------------------------------------
    # Determine runs
    if args.run:
        run_ids = [args.run]
    elif args.all or args.table:
        run_ids = list(RUN_MATRIX.keys())
    else:
        parser.print_help()
        print("\nNo run specified. Use --run <ID>, --all, --table, or --lira.")
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
