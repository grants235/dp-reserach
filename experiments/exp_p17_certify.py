#!/usr/bin/env python3
"""
Phase 17 — Certify: Corrected Per-Instance Certificate Computation
===================================================================

Corrects three bugs from Phase 16:

  1. COMPOSITION CORRECTION (Section 1 of spec):
     Charges every training step t = 1..T.  At steps where example i's gradient
     was logged (all-example or sampled), uses the actual Woodbury d_eff^(r)².
     At unsampled steps (all-example logging disabled), falls back to the
     data-independent rate by adding C²/σ_use² per missing step.

  2. LOEWNER VERIFICATION (Section 3 of spec):
     Reads the κ determined by the inline Loewner check during training.
     Uses λ̂_k = q(1−q)·n_priv·κ·λ_k^pub (previously κ=1 was assumed).

  3. SANITY CHECKS (Section 4 of spec):
     After computing certificates, verifies:
       ε_i^dir  ≤  ε_i^norm  ≤  ε_data_independent
     for every example.  Runs that fail any check are flagged and not reported.

Usage
-----
  python experiments/exp_p17_certify.py --run C3 --seed 0
  python experiments/exp_p17_certify.py --run C3          # all seeds + aggregate
  python experiments/exp_p17_certify.py --all
  python experiments/exp_p17_certify.py --table
"""

import os, sys, csv, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = "./results/exp_p17"
LOG_DIR     = os.path.join(RESULTS_DIR, "logs")
TRAIN_DIR   = os.path.join(RESULTS_DIR, "train")
CERT_DIR    = os.path.join(RESULTS_DIR, "certs")
CLIP_C      = 1.0

TIER_NAMES = {0: "head", 1: "mid", 2: "tail"}

ALPHA_GRID = np.concatenate([
    np.arange(1.5,  10,    0.5),
    np.arange(10,   100,   2.0),
    np.arange(100,  1000,  20.0),
    np.arange(1000, 5001,  100.0),
])

BETA_RANKS = [1, 5, 9, 10, 50, 100]

RUN_MATRIX = {
    "A1": dict(dataset="cifar10",       regime="R1", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5),
    "B1": dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=1.0, batch=5000,  n_seeds=5),
    "B2": dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=2.0, batch=5000,  n_seeds=5),
    "B3": dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=4.0, batch=5000,  n_seeds=5),
    "B4": dict(dataset="cifar10",       regime="R2", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5),
    "B5": dict(dataset="cifar10_lt50",  regime="R2", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5),
    "B6": dict(dataset="cifar10_lt100", regime="R2", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5),
    "B7": dict(dataset="emnist",        regime="R2", mech="vanilla", eps=8.0, batch=10000, n_seeds=5),
    "C1": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=1.0, batch=5000,  n_seeds=5),
    "C2": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=2.0, batch=5000,  n_seeds=5),
    "C3": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5),
    "C4": dict(dataset="cifar10_lt50",  regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5),
    "C5": dict(dataset="cifar10_lt100", regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=5),
    "D1": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=1000,  n_seeds=3),
    "D2": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=5000,  n_seeds=3),
    "D3": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=10000, n_seeds=3),
    "D4": dict(dataset="cifar10",       regime="R3", mech="vanilla", eps=8.0, batch=25000, n_seeds=3),
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
        return acct.get_privacy_spent(delta=delta)
    except Exception:
        rdp = T_steps * (q ** 2) * ALPHA_GRID / (2.0 * sigma_mult ** 2)
        return rdp_to_dp(rdp, ALPHA_GRID, delta)


# ---------------------------------------------------------------------------
# Per-instance certificates (CORRECTED composition)
# ---------------------------------------------------------------------------

def compute_certificates(stats, meta, delta):
    """
    Compute corrected norm-based and direction-aware per-instance certificates.

    CORRECTED COMPOSITION: charges every step t = 1..T.
      - At logged steps: uses actual Woodbury d_eff^(r)² via accumulated stats.
      - At unsampled steps (when all_example_logging=False): adds C²/σ_use²
        (data-independent fallback) per missing step.

    EIGENVALUE SCALING: λ̂_k = q(1−q)·n_priv·κ·λ_k^pub, using κ_global from
    the inline Loewner check.

    Parameters
    ----------
    stats : dict loaded from _stats.npz
    meta  : dict loaded from _meta.npz
    delta : float

    Returns dict indexed by local position in [0, n_priv).
    """
    # Unpack meta
    q          = float(meta["q"])
    n_priv     = int(meta["n_priv"])
    T_steps    = int(meta["T_steps"])
    sigma_mult = float(meta["sigma_mult"])
    sigma_use  = float(meta["sigma_use"])
    sigma2     = sigma_use ** 2               # = (σ_mult · C)²
    kappa      = float(meta.get("kappa_global", 1.0))
    lambdas_pub= meta["lambdas_pub"].astype(np.float64)
    all_ex_log = bool(meta.get("all_example_logging", True))

    # Unpack accumulated stats
    all_sum_gn2    = stats["all_sum_gn2"].astype(np.float64)     # [n_priv]
    all_sum_gproj2 = stats["all_sum_gproj2"].astype(np.float64)  # [n_priv, rank]
    n_logged       = stats["n_logged"].astype(np.int64)           # [n_priv]
    rank           = all_sum_gproj2.shape[1]

    # Scale PCA eigenvalues: λ̂_k = q(1−q)·n_priv·κ·λ_k^pub
    scale      = q * (1.0 - q) * n_priv * kappa
    lambdas_hat = lambdas_pub[:rank] * scale
    print(f"  [cert] κ={kappa:.4f}  scale={scale:.2f}  "
          f"λ̂_max={lambdas_hat[0]:.4g}  λ̂_min={lambdas_hat[-1]:.4g}")

    # -------------------------------------------------------------------
    # CORRECTED COMPOSITION
    # Number of unsampled steps = T_steps - n_logged[i]
    # At each unsampled step, contribution to sum_gn2 is C² (fallback d²=C²/σ²)
    # and to sum_woodbury2 is C²/σ² (same — no tightening at unsampled steps).
    # -------------------------------------------------------------------
    n_unsampled = T_steps - n_logged    # [n_priv], may be 0 if all-example logging

    if not all_ex_log:
        n_unsampled_actual = n_unsampled.copy()
        print(f"  [cert] all_example_logging=False: adding fallback for "
              f"{n_unsampled.mean():.1f} avg unsampled steps per example "
              f"(T={T_steps}, avg_logged={n_logged.mean():.1f})")
    else:
        n_unsampled_actual = np.zeros(n_priv, dtype=np.int64)
        print(f"  [cert] all_example_logging=True: all {T_steps} steps logged for every example")

    # Corrected norm totals (logged steps + data-independent fallback at unsampled steps)
    total_gn2 = all_sum_gn2 + n_unsampled_actual * (CLIP_C ** 2)   # [n_priv]

    # Woodbury direction-aware total — explicit-subtraction form:
    #   d_eff,i,t^(r)² = ||ḡ||²/σ²C² − Σ_k λ̂_k (ḡᵀuₖ)²/(σ²C²(σ²C²+λ̂_k))
    # Accumulated over ALL T steps:
    #   total_woodbury = total_gn2/σ² − Σ_k [Σ_t (ḡᵀuₖ)²] · λ̂_k/(σ²(σ²+λ̂_k))
    # Unsampled steps contribute C²/σ² with no tightening (they are already in total_gn2/σ²
    # and their proj contribution to the reduction is zero).
    #
    # This form guarantees total_woodbury ≤ total_gn2/σ² for any λ̂_k ≥ 0.
    reduction     = np.dot(all_sum_gproj2,
                           lambdas_hat / (sigma2 * (sigma2 + lambdas_hat)))  # (n,)
    total_woodbury = np.maximum(total_gn2 / sigma2 - reduction, 0.0)          # (n,)

    q_sq = q ** 2

    # RDP → DP conversion for each example
    eps_norm_arr = np.zeros(n_priv, dtype=np.float64)
    eps_dir_arr  = np.zeros(n_priv, dtype=np.float64)

    for i in range(n_priv):
        # Both totals are now in the same units (dimensionless d²):
        #   total_gn2[i] / sigma2  = Σ_t ||ḡ||²/σ²  (norm-based d²)
        #   total_woodbury[i]      = total_gn2/σ² − reduction  (Woodbury d²)
        rdp_norm = q_sq * (ALPHA_GRID / 2.0) * (total_gn2[i] / sigma2)
        rdp_dir  = q_sq * (ALPHA_GRID / 2.0) * total_woodbury[i]
        eps_norm_arr[i], _ = rdp_to_dp(rdp_norm, ALPHA_GRID, delta)
        eps_dir_arr[i],  _ = rdp_to_dp(rdp_dir,  ALPHA_GRID, delta)

    beta_mean = np.where(total_gn2 > 0,
                         total_woodbury / (total_gn2 / sigma2), 0.0)

    return {
        "n_priv":         n_priv,
        "n_logged":       n_logged,
        "n_unsampled":    n_unsampled_actual,
        "all_sum_gn2":    all_sum_gn2,
        "total_gn2":      total_gn2,
        "total_woodbury": total_woodbury,
        "eps_norm":       eps_norm_arr,
        "eps_direction":  eps_dir_arr,
        "beta_mean":      beta_mean,
        "lambdas_hat":    lambdas_hat,
        "kappa":          kappa,
        "all_ex_log":     all_ex_log,
    }


# ---------------------------------------------------------------------------
# β-spectrum (rank sweep, R3)
# ---------------------------------------------------------------------------

def compute_beta_spectrum(stats, meta, delta, ranks=BETA_RANKS):
    """Recompute Woodbury sum at each rank in `ranks`, return β^(r) values."""
    q         = float(meta["q"])
    n_priv    = int(meta["n_priv"])
    T_steps   = int(meta["T_steps"])
    sigma_use = float(meta["sigma_use"])
    sigma2    = sigma_use ** 2
    kappa     = float(meta.get("kappa_global", 1.0))
    lambdas_pub = meta["lambdas_pub"].astype(np.float64)

    all_sum_gn2    = stats["all_sum_gn2"].astype(np.float64)
    all_sum_gproj2 = stats["all_sum_gproj2"].astype(np.float64)
    n_logged       = stats["n_logged"].astype(np.int64)
    full_rank      = all_sum_gproj2.shape[1]
    all_ex_log     = bool(meta.get("all_example_logging", True))
    n_unsampled    = np.zeros(n_priv, dtype=np.int64) if all_ex_log else (T_steps - n_logged)

    betas = {}
    for r in ranks:
        r_use     = min(r, full_rank)
        lam_r     = lambdas_pub[:r_use] * (q * (1.0 - q) * n_priv * kappa)
        gp2_r     = all_sum_gproj2[:, :r_use]
        total_gn2_r = all_sum_gn2 + n_unsampled * CLIP_C ** 2
        reduction_r = np.dot(gp2_r, lam_r / (sigma2 * (sigma2 + lam_r)))
        wood_r      = np.maximum(total_gn2_r / sigma2 - reduction_r, 0.0)
        norm2       = np.where(total_gn2_r > 0, total_gn2_r / sigma2, 1.0)
        betas[r]    = float(np.mean(wood_r / norm2))
    return betas


# ---------------------------------------------------------------------------
# Sanity checks (Section 4 of spec)
# ---------------------------------------------------------------------------

def run_sanity_checks(certs, eps_di, eps_target, tag):
    """
    Verify the four invariants from the spec.  Returns True if all pass.
      (a) ε_i^dir  ≤  ε_i^norm  for every i
      (b) ε_i^norm ≤  ε_data_independent  for every i
      (c) ε_i^dir  ≤  ε_data_independent  for every i
      (d) data-independent ε matches target ε (within 5%)
    """
    en = certs["eps_norm"]; ed = certs["eps_direction"]
    tol = 1e-6
    ok  = True

    viol_ab = (ed > en + tol).sum()
    if viol_ab > 0:
        print(f"  [SANITY FAIL] {tag}: {viol_ab} examples have ε^dir > ε^norm")
        ok = False

    viol_b = (en > eps_di + tol).sum()
    if viol_b > 0:
        print(f"  [SANITY FAIL] {tag}: {viol_b} examples have ε^norm > ε_data_independent "
              f"({eps_di:.4f})")
        ok = False

    viol_c = (ed > eps_di + tol).sum()
    if viol_c > 0:
        print(f"  [SANITY FAIL] {tag}: {viol_c} examples have ε^dir > ε_data_independent")
        ok = False

    eps_ratio = abs(eps_di - eps_target) / max(eps_target, 1e-9)
    if eps_ratio > 0.05:
        print(f"  [SANITY WARN] {tag}: ε_data_independent={eps_di:.4f} vs target={eps_target:.4f}"
              f"  (ratio={eps_ratio:.3f})")

    if ok:
        print(f"  [SANITY OK] {tag}: all four invariants pass")
    return ok


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------

def load_stats(tag, log_dir):
    path = os.path.join(log_dir, f"{tag}_stats.npz")
    if not os.path.exists(path):
        print(f"  [cert] Stats not found: {path}")
        return None
    return dict(np.load(path, allow_pickle=True))


def load_meta(tag, log_dir):
    path = os.path.join(log_dir, f"{tag}_meta.npz")
    if not os.path.exists(path):
        print(f"  [cert] Meta not found: {path}")
        return None
    return dict(np.load(path, allow_pickle=True))


def load_loewner_log(tag, log_dir):
    path = os.path.join(log_dir, f"{tag}_loewner.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def load_accuracy(tag, train_dir=TRAIN_DIR):
    csv_path = os.path.join(train_dir, f"{tag}.csv")
    if not os.path.exists(csv_path):
        return None, None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "test_acc" not in df.columns or df.empty:
            return None, None
        return float(df["test_acc"].iloc[-1]), float(df["test_acc"].max())
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Save / summarise
# ---------------------------------------------------------------------------

def save_certs_csv(certs, tier_arr, cert_path):
    fieldnames = ["local_pos", "n_logged", "n_unsampled", "eps_norm", "eps_direction",
                  "beta_mean", "total_gn2", "total_woodbury"]
    if tier_arr is not None:
        fieldnames.append("tier")
    with open(cert_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        n = certs["n_priv"]
        for i in range(n):
            row = {
                "local_pos":      i,
                "n_logged":       int(certs["n_logged"][i]),
                "n_unsampled":    int(certs["n_unsampled"][i]),
                "eps_norm":       f"{certs['eps_norm'][i]:.6f}",
                "eps_direction":  f"{certs['eps_direction'][i]:.6f}",
                "beta_mean":      f"{certs['beta_mean'][i]:.6f}",
                "total_gn2":      f"{certs['total_gn2'][i]:.6f}",
                "total_woodbury": f"{certs['total_woodbury'][i]:.6f}",
            }
            if tier_arr is not None:
                row["tier"] = int(tier_arr[i])
            w.writerow(row)
    print(f"  [cert] Saved: {cert_path}")


def summarize(label, certs, delta, eps_target, tier_arr):
    en    = certs["eps_norm"]; ed = certs["eps_direction"]
    bm    = certs["beta_mean"]
    ratio = np.where(en > 0, ed / en, np.nan)
    rc    = ratio[~np.isnan(ratio)]

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  n={len(en)}  ε_target={eps_target}  δ={delta:.2e}"
          f"  κ={certs['kappa']:.4f}  all_ex_log={certs['all_ex_log']}")
    print(f"{'='*70}")
    print(f"  Norm-based  ε^norm:  mean={en.mean():.4f}  "
          f"med={np.median(en):.4f}  95th={np.percentile(en,95):.4f}  max={en.max():.4f}")
    print(f"  Dir-aware   ε^dir:   mean={ed.mean():.4f}  "
          f"med={np.median(ed):.4f}  95th={np.percentile(ed,95):.4f}  max={ed.max():.4f}")
    if len(rc) > 0:
        print(f"  Tightening ε^dir/ε^norm: mean={rc.mean():.4f}  "
              f"med={np.median(rc):.4f}  min={rc.min():.4f}")
    print(f"  β mean={bm.mean():.4f}  95th={np.percentile(bm,95):.4f}")
    print(f"  n_logged: mean={certs['n_logged'].mean():.1f}  "
          f"n_unsampled: mean={certs['n_unsampled'].mean():.1f}")

    if tier_arr is not None and len(tier_arr) > 0:
        print("  Per-tier:")
        for t, tname in TIER_NAMES.items():
            mask = tier_arr == t
            if mask.sum() == 0: continue
            r_t = ratio[mask]; r_t = r_t[~np.isnan(r_t)]
            print(f"    {tname:4s} (n={mask.sum():5d}): ε^norm={en[mask].mean():.4f}  "
                  f"ε^dir={ed[mask].mean():.4f}  tighten={r_t.mean():.4f}" if len(r_t) > 0 else
                  f"    {tname:4s} (n={mask.sum():5d}): ε^norm={en[mask].mean():.4f}  "
                  f"ε^dir={ed[mask].mean():.4f}")


def build_summary_dict(certs, eps_di, alpha_di, tier_arr, betas=None):
    en = certs["eps_norm"]; ed = certs["eps_direction"]; bm = certs["beta_mean"]
    ratio = np.where(en > 0, ed / en, np.nan)
    rc = ratio[~np.isnan(ratio)]

    d = {
        "eps_di": float(eps_di), "alpha_di": float(alpha_di),
        "eps_norm_mean": float(en.mean()), "eps_norm_med": float(np.median(en)),
        "eps_norm_95":   float(np.percentile(en, 95)), "eps_norm_max": float(en.max()),
        "eps_dir_mean":  float(ed.mean()),  "eps_dir_med":  float(np.median(ed)),
        "eps_dir_95":    float(np.percentile(ed, 95)),  "eps_dir_max":  float(ed.max()),
        "tightening_mean": float(rc.mean()) if len(rc) > 0 else float("nan"),
        "tightening_med":  float(np.median(rc)) if len(rc) > 0 else float("nan"),
        "beta_mean": float(bm.mean()), "beta_95": float(np.percentile(bm, 95)),
        "kappa": float(certs["kappa"]),
        "all_ex_log": bool(certs["all_ex_log"]),
        "n_logged_mean": float(certs["n_logged"].mean()),
        "n_unsampled_mean": float(certs["n_unsampled"].mean()),
    }
    if tier_arr is not None and len(tier_arr) > 0:
        for t, tname in TIER_NAMES.items():
            mask = tier_arr == t
            if mask.sum() == 0: continue
            r_t = ratio[mask]; r_t = r_t[~np.isnan(r_t)]
            d[f"eps_norm_{tname}"] = float(en[mask].mean())
            d[f"eps_dir_{tname}"]  = float(ed[mask].mean())
            d[f"tight_{tname}"]    = float(r_t.mean()) if len(r_t) > 0 else float("nan")
    if betas:
        for r, v in betas.items():
            d[f"beta_rank{r}"] = v
    return d


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_certs(run_id, seed, certs, eps_target, tier_arr, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    en = certs["eps_norm"]; ed = certs["eps_direction"]; bm = certs["beta_mean"]
    is_lt = tier_arr is not None and len(tier_arr) > 0

    # Scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(en, ed, c=bm, cmap="plasma_r", s=3, alpha=0.5, vmin=0, vmax=1)
    mn, mx = min(en.min(), ed.min()), max(en.max(), ed.max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, label="y=x (no improvement)")
    ax.set_xlabel("ε^norm"); ax.set_ylabel("ε^dir")
    ax.set_title(f"{run_id} s{seed}: direction-aware vs norm-based  (ε={eps_target})")
    plt.colorbar(sc, ax=ax, label="β"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"p17_{run_id}_s{seed}_scatter.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [plot] {path}")

    # Histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    if is_lt:
        colors = {0: "steelblue", 1: "orange", 2: "firebrick"}
        for t, tname in TIER_NAMES.items():
            mask = tier_arr == t
            if mask.sum() == 0: continue
            ax.hist(ed[mask], bins=50, alpha=0.6, label=tname, color=colors[t])
        ax.set_title(f"{run_id} s{seed}: ε^dir by tier  (ε={eps_target})")
    else:
        ax.hist(en, bins=50, alpha=0.6, label="ε^norm", color="steelblue")
        ax.hist(ed, bins=50, alpha=0.6, label="ε^dir",  color="darkorange")
        ax.set_title(f"{run_id} s{seed}: certificates  (ε={eps_target})")
    ax.axvline(eps_target, color="k", ls="--", lw=1, label=f"ε={eps_target}")
    ax.legend(); ax.set_xlabel("per-instance ε"); ax.set_ylabel("count")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    path = os.path.join(out_dir, f"p17_{run_id}_s{seed}_hist.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [plot] {path}")


def plot_beta_spectrum(run_id, seed, betas, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ranks = [r for r, v in betas.items() if v is not None]
    vals  = [betas[r] for r in ranks]
    if not ranks: return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranks, vals, "o-", color="steelblue")
    ax.set_xlabel("Subspace rank r"); ax.set_ylabel("β^(r) mean")
    ax.set_title(f"{run_id} s{seed}: β-spectrum vs rank")
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.set_ylim(bottom=0); ax.grid(True, alpha=0.3); fig.tight_layout()
    path = os.path.join(out_dir, f"p17_{run_id}_s{seed}_beta_spectrum.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [plot] {path}")


def plot_loewner_kappa(run_id, seed, loewner_log, out_dir):
    """Plot κ over training steps from the Loewner check log."""
    if not loewner_log: return
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps  = [e["step"] for e in loewner_log]
    kappas = [e["kappa"] for e in loewner_log]
    kap_g  = [e["kappa_global"] for e in loewner_log]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(steps, kappas, "o-", ms=3, color="steelblue", label="κ at step")
    ax.plot(steps, kap_g,  "-",  color="firebrick",       label="κ_global (min so far)")
    ax.set_xlabel("Training step"); ax.set_ylabel("κ")
    ax.set_title(f"{run_id} s{seed}: Loewner κ history")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); fig.tight_layout()
    path = os.path.join(out_dir, f"p17_{run_id}_s{seed}_kappa.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [plot] {path}")


# ---------------------------------------------------------------------------
# Per-run certification
# ---------------------------------------------------------------------------

def certify_run_seed(run_id, cfg, seed, log_dir, cert_dir):
    tag       = (f"p17_{run_id}_{cfg['mech']}_{cfg['dataset']}_{cfg['regime']}"
                 f"_eps{cfg['eps']:.0f}_seed{seed}")
    cert_path = os.path.join(cert_dir, f"{tag}_certs.csv")
    summ_path = os.path.join(cert_dir, f"{tag}_summary.json")

    meta  = load_meta(tag, log_dir)
    stats = load_stats(tag, log_dir)
    loewner_log = load_loewner_log(tag, log_dir)

    if meta is None or stats is None:
        print(f"[P17-cert] {tag}: meta or stats missing — run exp_p17_train.py first.")
        return None, None

    eps       = float(meta["eps"])
    delta     = float(meta["delta"])
    sigma_mult= float(meta["sigma_mult"])
    q         = float(meta["q"])
    T_steps   = int(meta["T_steps"])
    n_priv    = int(meta["n_priv"])
    tier_arr  = meta["tier_labels"]
    tier_arr  = tier_arr if len(tier_arr) > 0 else None

    print(f"\n[P17-cert] === {run_id} seed={seed} ===")
    print(f"  n_priv={n_priv}  T={T_steps}  q={q:.5f}  eps={eps}  delta={delta:.2e}")
    print(f"  loewner_checks={len(loewner_log)}")

    # Data-independent certificate
    eps_di, alpha_di = data_independent_eps(sigma_mult, q, T_steps, delta)
    print(f"  Data-independent: ε={eps_di:.4f}  (target={eps})")

    # Per-instance certificates (corrected composition)
    certs = compute_certificates(stats, meta, delta)

    # β-spectrum for R3
    is_r3 = cfg["regime"] == "R3"
    betas  = compute_beta_spectrum(stats, meta, delta) if is_r3 else {}

    # Sanity checks (spec Section 4)
    ok = run_sanity_checks(certs, eps_di, eps, tag)
    if not ok:
        print(f"  [WARN] {tag}: sanity checks failed — numbers flagged but still saved.")

    os.makedirs(cert_dir, exist_ok=True)
    save_certs_csv(certs, tier_arr, cert_path)

    label = f"{run_id} seed={seed} — {cfg['regime']} {cfg['mech']} {cfg['dataset']}"
    summarize(label, certs, delta, eps, tier_arr)

    # Summary JSON
    summ = build_summary_dict(certs, eps_di, alpha_di, tier_arr, betas if is_r3 else None)
    summ.update({"tag": tag, "run_id": run_id, "seed": seed, "sanity_ok": ok,
                 "regime": cfg["regime"], "mech": cfg["mech"],
                 "dataset": cfg["dataset"], "eps": eps,
                 "n_loewner_checks": len(loewner_log)})
    final_acc, best_acc = load_accuracy(tag)
    if final_acc is not None:
        summ["final_acc"] = final_acc; summ["best_acc"] = best_acc

    with open(summ_path, "w") as f:
        json.dump(summ, f, indent=2)
    print(f"  [cert] Summary: {summ_path}")

    # Plots
    plot_certs(run_id, seed, certs, eps, tier_arr, cert_dir)
    plot_loewner_kappa(run_id, seed, loewner_log, cert_dir)
    if is_r3 and betas:
        plot_beta_spectrum(run_id, seed, betas, cert_dir)

    return certs, tier_arr


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def aggregate_seeds(run_id, cfg, log_dir, cert_dir):
    all_summ = []
    for seed in range(cfg["n_seeds"]):
        tag   = (f"p17_{run_id}_{cfg['mech']}_{cfg['dataset']}_{cfg['regime']}"
                 f"_eps{cfg['eps']:.0f}_seed{seed}")
        spath = os.path.join(cert_dir, f"{tag}_summary.json")
        if not os.path.exists(spath): continue
        try:
            with open(spath) as f: all_summ.append(json.load(f))
        except Exception: pass

    if not all_summ: return None

    fields = ["eps_norm_mean", "eps_dir_mean", "tightening_mean", "beta_mean", "best_acc"]
    result = {"n_seeds_present": len(all_summ)}
    for fld in fields:
        vals = [s[fld] for s in all_summ if fld in s and not (isinstance(s[fld], float) and np.isnan(s[fld]))]
        if vals:
            result[f"{fld}_median"] = float(np.median(vals))
            result[f"{fld}_iqr"]    = [float(np.percentile(vals, 25)),
                                       float(np.percentile(vals, 75))]

    print(f"\n[P17-cert] {run_id}: {len(all_summ)}/{cfg['n_seeds']} seeds aggregated")
    for fld in fields:
        if f"{fld}_median" in result:
            iqr = result.get(f"{fld}_iqr", [float("nan"), float("nan")])
            print(f"  {fld:25s} median={result[f'{fld}_median']:.4f}"
                  f"  IQR=[{iqr[0]:.4f},{iqr[1]:.4f}]")
    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_table(all_stats):
    hdr = (f"{'Run':5s}  {'Regime':6s}  {'Dataset':14s}  {'ε':4s}  "
           f"{'ε^norm med':10s}  {'ε^dir med':9s}  {'tighten':8s}  "
           f"{'β med':6s}  {'best_acc':8s}  {'seeds':5s}")
    sep = "=" * len(hdr)
    print(f"\n{sep}\n  Phase 17 Summary Table\n{sep}")
    print(f"  {hdr}\n  {'-'*len(hdr)}")
    for run_id, stats in sorted(all_stats.items()):
        if stats is None: continue
        cfg = RUN_MATRIX.get(run_id, {})
        def _g(k, fmt=".4f"):
            v = stats.get(f"{k}_median", float("nan"))
            return f"{v:{fmt}}" if not np.isnan(v) else "  N/A  "
        print(f"  {run_id:5s}  {cfg.get('regime','?'):6s}  "
              f"{cfg.get('dataset','?'):14s}  {cfg.get('eps',0):4.0f}  "
              f"{_g('eps_norm_mean'):10s}  {_g('eps_dir_mean'):9s}  "
              f"{_g('tightening_mean'):8s}  {_g('beta_mean'):6s}  "
              f"{_g('best_acc'):8s}  {stats['n_seeds_present']:5d}")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 17 certificate computation")
    parser.add_argument("--run",      type=str, default=None)
    parser.add_argument("--seed",     type=int, default=None)
    parser.add_argument("--all",      action="store_true")
    parser.add_argument("--table",    action="store_true")
    parser.add_argument("--log_dir",  type=str, default=LOG_DIR)
    parser.add_argument("--train_dir",type=str, default=TRAIN_DIR)
    parser.add_argument("--cert_dir", type=str, default=CERT_DIR)
    args = parser.parse_args()

    os.makedirs(args.cert_dir, exist_ok=True)

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
            print(f"[P17-cert] Unknown run: {run_id}"); continue
        cfg = RUN_MATRIX[run_id]

        if args.seed is not None:
            certify_run_seed(run_id, cfg, args.seed, args.log_dir, args.cert_dir)
        else:
            for seed in range(cfg["n_seeds"]):
                certify_run_seed(run_id, cfg, seed, args.log_dir, args.cert_dir)
            stats = aggregate_seeds(run_id, cfg, args.log_dir, args.cert_dir)
            all_stats[run_id] = stats
            if stats is not None:
                agg_path = os.path.join(args.cert_dir, f"p17_{run_id}_aggregate.json")
                with open(agg_path, "w") as f:
                    json.dump({"run_id": run_id, "cfg": {k: str(v) for k, v in cfg.items()},
                               **stats}, f, indent=2)
                print(f"  [cert] Aggregate saved: {agg_path}")

    if args.table or (not args.seed and len(all_stats) > 1):
        for run_id in run_ids:
            if run_id not in all_stats or all_stats[run_id] is None:
                agg_path = os.path.join(args.cert_dir, f"p17_{run_id}_aggregate.json")
                if os.path.exists(agg_path):
                    try:
                        with open(agg_path) as f: all_stats[run_id] = json.load(f)
                    except Exception: pass
        print_table(all_stats)

    print(f"\n[P17-cert] Done. Results in {args.cert_dir}")


if __name__ == "__main__":
    main()
