#!/usr/bin/env python3
"""
Phase 18 Certificate: Nyström PSD-Minorant Direction-Aware Bound
=================================================================

Spec: phase18_spex.md, Sections 1.4-1.6 and Section 5.

For each run and rank r ∈ {10, 25, 50, 100, 200}:
  1. Truncate saved statistics to rank r
  2. Compute S_{t,r} = B_{t,r}^{†/2}
  3. Compute U_{i,t,r}^full via Woodbury (Section 1.4)
  4. Apply leave-one-out correction (Section 1.5)
  5. Compose ε_i^dir(δ) over T_acc steps (Section 1.6)

Also computes ε_i^norm for comparison.

Usage:
  python experiments/exp_p18_certify.py --setting S2 --seed 0
  python experiments/exp_p18_certify.py --setting S2
  python experiments/exp_p18_certify.py --all_minimal
  python experiments/exp_p18_certify.py --all
  python experiments/exp_p18_certify.py --table
"""

import os, sys, json, argparse
import numpy as np

RUNS_DIR  = "./runs"
CERT_DIR  = "./certs/p18"

RANK_ABLATION = [10, 25, 50, 100, 200]   # Section 5, Table G

ALPHA_GRID = np.concatenate([
    np.arange(1.5,  10,    0.5),
    np.arange(10,   100,   2.0),
    np.arange(100,  1000,  20.0),
    np.arange(1000, 5001,  100.0),
])

SETTINGS_MINIMAL = {"S1", "S2", "S3"}
SETTINGS_ALL = {"S1", "S2", "S3", "S4a", "S4b", "S4c", "S5", "S6", "S7", "S9"}


# ---------------------------------------------------------------------------
# RDP helpers
# ---------------------------------------------------------------------------

def rdp_to_dp(rdp_per_alpha, delta):
    """Convert RDP array (indexed by ALPHA_GRID) to (ε, δ) via standard formula."""
    candidates = rdp_per_alpha + np.log(1.0 / delta) / (ALPHA_GRID - 1.0)
    idx = np.argmin(candidates)
    return float(candidates[idx]), float(ALPHA_GRID[idx])


def compose_rdp(mu_per_step, q):
    """
    Compose RDP over T steps.

    mu_per_step: [T_acc] array of μ values (one per accounting step)
    q: Poisson sampling rate

    Per-step RDP (leading-order subsampled Gaussian approximation):
      ε_step(α) = q² α μ² / 2

    This matches Phase 17's formula and is valid for the small-q limit.
    Returns rdp: [len(ALPHA_GRID)] array.
    """
    total_d2 = np.sum(mu_per_step ** 2)         # Σ_t μ²_t = Σ_t d̂²_t
    q_sq = q ** 2
    rdp = q_sq * (ALPHA_GRID / 2.0) * total_d2  # [len(ALPHA_GRID)]
    return rdp


def data_independent_eps(sigma, q, T_steps, delta):
    """Data-independent (ε,δ)-DP via opacus RDP accountant."""
    try:
        from opacus.accountants import RDPAccountant
        acct = RDPAccountant(); acct.history = [(sigma, q, T_steps)]
        return acct.get_privacy_spent(delta=delta)
    except Exception:
        rdp = T_steps * (q ** 2) * ALPHA_GRID / (2.0 * sigma ** 2)
        return rdp_to_dp(rdp, delta)


def data_independent_eps_quad(a, q, T_steps, delta):
    """Data-independent bound using same quadratic approx as per-instance formula."""
    rdp = T_steps * (q ** 2) * ALPHA_GRID / (2.0 * a)
    return rdp_to_dp(rdp, delta)


# ---------------------------------------------------------------------------
# Core certificate computation (Section 5)
# ---------------------------------------------------------------------------

def compute_B_pseudo_sqrt_inv(B_r):
    """
    Compute S_r = B_r^{†/2} via eigendecomposition.
    Threshold: τ = 1e-8 × λ_max (spec Section 1.4).
    Returns S_r (r×r) symmetric PSD.
    """
    eigvals, eigvecs = np.linalg.eigh(B_r.astype(np.float64))
    lam_max = eigvals.max()
    tau = 1e-8 * max(lam_max, 1e-30)
    inv_sqrt = np.where(eigvals > tau, eigvals ** (-0.5), 0.0)
    S_r = (eigvecs * inv_sqrt[None, :]) @ eigvecs.T   # (r, r)
    return S_r


def compute_U_full_batch(clipped_norms_t, B_t, YTY_t, Y_proj_t, a, r):
    """
    Compute U_{i,t,r}^full for all n examples at one accounting step.

    clipped_norms_t : [n]          ||ḡ_i||
    B_t             : [r_max, r_max]
    YTY_t           : [r_max, r_max]
    Y_proj_t        : [n, r_max]   Y_t^T ḡ_i for each i
    a               : scalar       σ² C²
    r               : int          rank to use (≤ r_max)

    Returns U_full : [n] clipped to [0, ||g||²/a]
    """
    r = min(r, B_t.shape[0])

    # Truncate to rank r (Section 5.1)
    B_r    = B_t[:r, :r].astype(np.float64)
    M_r    = YTY_t[:r, :r].astype(np.float64)
    s_r    = Y_proj_t[:, :r].astype(np.float64)   # (n, r)

    # S_r = B_r^{†/2} (Section 5.2)
    S_r = compute_B_pseudo_sqrt_inv(B_r)           # (r, r)

    # z = S_r s_r^T → shape (r, n) → transpose → (n, r) (Section 5.3)
    z = (S_r @ s_r.T).T                            # (n, r)

    # H = S_r M_r S_r (Section 5.3)
    H_r = S_r @ M_r @ S_r                          # (r, r)

    # (I + a^{-1} H_r)^{-1} z (n × r)
    A = np.eye(r) + H_r / a                        # (r, r)
    # Solve A x = z^T for x, then x = A^{-1} z^T, then (n, r)
    Ainv_z = np.linalg.solve(A, z.T).T             # (n, r)

    # Quadratic form: z^T A^{-1} z for each row
    ztAinvz = (z * Ainv_z).sum(axis=1)             # (n,)

    # U^full = ||g||²/a - ztAinvz/a² (Section 1.4 / 5.3)
    norm2 = clipped_norms_t.astype(np.float64) ** 2
    U_full = norm2 / a - ztAinvz / (a ** 2)

    # Numerical clipping (Section 1.4)
    U_full = np.clip(U_full, 0.0, norm2 / a)

    return U_full.astype(np.float64)


def compute_d2_hat(U_full, norm2_over_a, rho):
    """
    Leave-one-out correction (Section 1.5 / 5.4).

    d̂² = U / (1 - ρU)  if ρU < 1
        = ||g||²/a      otherwise
    """
    rho_U = rho * U_full
    fallback = (rho_U >= 1.0)
    d2_hat = np.where(
        fallback,
        norm2_over_a,
        np.minimum(U_full / np.maximum(1.0 - rho_U, 1e-30), norm2_over_a)
    )
    return d2_hat, fallback


def certify_run(run_dir, rank=100, delta=None, verbose=True):
    """
    Compute per-instance certificates from a run directory.

    Returns dict with:
      eps_norm  [n]  norm-based bound
      eps_dir   [n]  direction-aware Nyström bound
      d2_hat    [n, T_acc]  per-step d̂²
      fallback_rate  scalar
      meta      dict
    """
    meta_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"  [cert] metadata.json not found in {run_dir}"); return None

    with open(meta_path) as f:
        meta = json.load(f)

    if delta is None:
        delta = meta["delta"]

    q    = meta["q"]
    rho  = meta["rho"]
    a    = meta["a"]
    T_acc = meta["T_acc"]
    T_train = int(meta.get("T_train", T_acc))

    # Load saved statistics
    def load_arr(name):
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            print(f"  [cert] Missing: {name}"); return None
        return np.load(p)

    clipped_norms = load_arr("clipped_norms.npy")  # (n, T_acc)
    B_matrices    = load_arr("B_matrices.npy")     # (T_acc, r_max, r_max)
    YTY_matrices  = load_arr("YTY_matrices.npy")   # (T_acc, r_max, r_max)
    Y_projections = load_arr("Y_projections.npy")  # (n, T_acc, r_max)

    if any(x is None for x in [clipped_norms, B_matrices, YTY_matrices, Y_projections]):
        return None

    n = clipped_norms.shape[0]
    T_use = min(T_acc, clipped_norms.shape[1])

    # Norm-based bound on logged steps:
    # μ_{norm,i,t} = ||g_{i,t}|| / (σC) = ||g||/sqrt(a).
    d2_norm = (clipped_norms[:, :T_use] ** 2).sum(axis=1) / a  # [n]

    # Direction-aware Nyström bound
    d2_dir_sum = np.zeros(n, dtype=np.float64)
    d2_hat_all = np.zeros((n, T_use), dtype=np.float64)
    fallback_total = 0

    for t in range(T_use):
        norms_t  = clipped_norms[:, t]             # [n]
        B_t      = B_matrices[t]                   # [r_max, r_max]
        YTY_t    = YTY_matrices[t]                 # [r_max, r_max]
        Yp_t     = Y_projections[:, t, :]          # [n, r_max]

        U_full_t = compute_U_full_batch(norms_t, B_t, YTY_t, Yp_t, a, rank)
        norm2_a_t = norms_t.astype(np.float64) ** 2 / a

        d2_t, fb_t = compute_d2_hat(U_full_t, norm2_a_t, rho)
        d2_hat_all[:, t] = d2_t
        d2_dir_sum += d2_t
        fallback_total += fb_t.sum()

    fallback_rate = fallback_total / (n * T_use) if T_use > 0 else 0.0

    # Sparse-step accounting for WRN regimes: if only every K-th step was
    # logged, charge all unlogged training steps at the worst-case norm rate.
    # This keeps the composed certificate conservative and matches the spec's
    # requirement that intermediate steps are not silently dropped.
    missing_steps = max(0, T_train - T_use)
    if missing_steps > 0:
        missing_d2 = missing_steps / a
        d2_norm += missing_d2
        d2_dir_sum += missing_d2

    # Per-example ε: compose via RDP then convert to (ε,δ)
    # ε^norm(α) = q²α/2 × d²_norm[i];  ε^dir(α) = q²α/2 × d²_dir[i]
    q_sq = q ** 2
    eps_norm = np.zeros(n, dtype=np.float64)
    eps_dir  = np.zeros(n, dtype=np.float64)

    for i in range(n):
        rdp_norm = q_sq * (ALPHA_GRID / 2.0) * d2_norm[i]
        rdp_dir  = q_sq * (ALPHA_GRID / 2.0) * d2_dir_sum[i]
        eps_norm[i], _ = rdp_to_dp(rdp_norm, delta)
        eps_dir[i],  _ = rdp_to_dp(rdp_dir,  delta)

    if verbose:
        eps_di, _ = data_independent_eps(meta["sigma"], q, T_train, delta)
        eps_di_q, _ = data_independent_eps_quad(a, q, T_train, delta)
        print(f"  [cert r={rank}] T_acc={T_use}  T_train={T_train}  "
              f"unlogged={missing_steps}  fallback={fallback_rate:.4%}")
        print(f"  ε_di_opacus={eps_di:.4f}  ε_di_quad={eps_di_q:.4f}  target={meta['epsilon_target']}")
        print(f"  ε^norm:  med={np.median(eps_norm):.4f}  CV={eps_norm.std()/eps_norm.mean():.3f}"
              f"  max={eps_norm.max():.4f}")
        print(f"  ε^dir:   med={np.median(eps_dir):.4f}  CV={eps_dir.std()/eps_dir.mean():.3f}"
              f"  max={eps_dir.max():.4f}")

    return {
        "eps_norm": eps_norm,
        "eps_dir":  eps_dir,
        "d2_norm":  d2_norm,
        "d2_dir":   d2_dir_sum,
        "d2_hat_all": d2_hat_all,
        "fallback_rate": fallback_rate,
        "missing_steps_charged": missing_steps,
        "meta": meta,
        "n": n,
        "T_acc": T_use,
    }


# ---------------------------------------------------------------------------
# Sanity checks (Section 9)
# ---------------------------------------------------------------------------

def run_sanity_checks(result, tag):
    en = result["eps_norm"]; ed = result["eps_dir"]
    a = result["meta"]["a"]; q = result["meta"]["q"]
    T_acc = int(result["meta"].get("T_train", result["T_acc"]))
    delta = result["meta"]["delta"]
    tol = 1e-6; ok = True

    # (1) d̂² ≤ ||g||²/a at every step
    # (already enforced by clipping in compute_U_full_batch)

    # (2) ε^dir ≤ ε^norm for every example
    viol = (ed > en + tol).sum()
    if viol > 0:
        print(f"  [SANITY FAIL] {tag}: {viol} examples have ε^dir > ε^norm"); ok = False
    else:
        print(f"  [SANITY OK]   {tag}: ε^dir ≤ ε^norm ✓")

    # (3) Fallback rate should be near zero
    fr = result["fallback_rate"]
    if fr > 0.01:
        print(f"  [SANITY WARN] {tag}: fallback rate={fr:.4%} (expected near zero)")
    else:
        print(f"  [SANITY OK]   {tag}: fallback rate={fr:.6%} ✓")

    # (4) ε^norm ≤ ε_di_quad (guaranteed by construction)
    eps_di_q, _ = data_independent_eps_quad(a, q, T_acc, delta)
    viol2 = (en > eps_di_q + tol).sum()
    if viol2 > 0:
        print(f"  [SANITY FAIL] {tag}: {viol2} examples ε^norm > ε_di_quad={eps_di_q:.4f}"); ok = False
    else:
        print(f"  [SANITY OK]   {tag}: ε^norm ≤ ε_di_quad={eps_di_q:.4f} ✓")

    return ok


# ---------------------------------------------------------------------------
# Rank ablation (Table G)
# ---------------------------------------------------------------------------

def rank_ablation(run_dir, delta=None, save_prefix=None):
    """Compute certs at ranks 10,25,50,100,200 for Table G."""
    results = {}
    for r in RANK_ABLATION:
        if delta is not None:
            res = certify_run(run_dir, rank=r, delta=delta, verbose=False)
        else:
            res = certify_run(run_dir, rank=r, verbose=False)
        if res is None: continue
        results[r] = {
            "eps_dir_med": float(np.median(res["eps_dir"])),
            "eps_norm_med": float(np.median(res["eps_norm"])),
            "fallback_rate": float(res["fallback_rate"]),
        }
        if save_prefix is not None:
            np.save(f"{save_prefix}_eps_dir_r{r}.npy", res["eps_dir"].astype(np.float32))
    return results


# ---------------------------------------------------------------------------
# Tier-wise summary (Table E)
# ---------------------------------------------------------------------------

def tier_summary(result, tier_labels, tag=""):
    en = result["eps_norm"]; ed = result["eps_dir"]
    if tier_labels is None:
        print(f"  {tag}: no tier labels (non-LT dataset)")
        return
    tier_names = {0: "head", 1: "mid", 2: "tail"}
    print(f"\n  Tier summary ({tag}):")
    for t, tname in tier_names.items():
        mask = (tier_labels == t)
        if mask.sum() == 0: continue
        print(f"    {tname:4s} n={mask.sum():5d}  "
              f"ε^norm={en[mask].mean():.4f}  ε^dir={ed[mask].mean():.4f}  "
              f"ratio={ed[mask].mean()/en[mask].mean():.4f}")
    tail = (tier_labels == 2); head = (tier_labels == 0)
    if head.sum() > 0 and tail.sum() > 0:
        print(f"    tail/head ratio (ε^dir): {ed[tail].mean()/ed[head].mean():.3f}")


# ---------------------------------------------------------------------------
# Full summary
# ---------------------------------------------------------------------------

def summarize_run(setting_id, seed, result, tag=""):
    meta = result["meta"]
    en = result["eps_norm"]; ed = result["eps_dir"]
    ratio = np.where(en > 0, ed / en, np.nan); rc = ratio[~np.isnan(ratio)]
    print(f"\n{'='*72}")
    print(f"  {setting_id} seed={seed}  {meta['regime']} {meta['dataset']}  ε={meta['epsilon_target']}")
    print(f"  n={result['n']}  T_acc={result['T_acc']}  r=100")
    print(f"{'='*72}")
    print(f"  ε^norm:  mean={en.mean():.4f}  med={np.median(en):.4f}  "
          f"CV={en.std()/en.mean():.3f}  max={en.max():.4f}")
    print(f"  ε^dir:   mean={ed.mean():.4f}  med={np.median(ed):.4f}  "
          f"CV={ed.std()/ed.mean():.3f}  max={ed.max():.4f}")
    if len(rc) > 0:
        print(f"  ε^dir/ε^norm: mean={rc.mean():.4f}  med={np.median(rc):.4f}"
              f"  → {(1-rc.mean())*100:.1f}% mean tightening")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plt():
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_scatter(result, tier_labels, out_path):
    plt = _plt()
    if plt is None: return
    en = result["eps_norm"]; ed = result["eps_dir"]
    meta = result["meta"]

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = np.zeros(result["n"])
    if tier_labels is not None:
        colors = tier_labels.astype(float)
    sc = ax.scatter(en, ed, c=colors, cmap="RdYlGn_r", s=4, alpha=0.4)
    mn, mx = min(en.min(), ed.min()), max(en.max(), ed.max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, label="y=x")
    ax.set_xlabel("ε^norm"); ax.set_ylabel("ε^dir")
    ax.set_title(f"{meta['setting_id']} s{meta['seed']}: dir vs norm  (ε={meta['epsilon_target']})")
    if tier_labels is not None:
        plt.colorbar(sc, ax=ax, label="tier")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  [plot] {out_path}")


def plot_hist(result, tier_labels, out_path):
    plt = _plt()
    if plt is None: return
    en = result["eps_norm"]; ed = result["eps_dir"]
    meta = result["meta"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if tier_labels is not None:
        tier_colors = {0: "steelblue", 1: "orange", 2: "firebrick"}
        tier_names  = {0: "head", 1: "mid", 2: "tail"}
        for t, tname in tier_names.items():
            mask = tier_labels == t
            if mask.sum() == 0: continue
            axes[0].hist(en[mask], bins=40, alpha=0.6, label=tname, color=tier_colors[t])
            axes[1].hist(ed[mask], bins=40, alpha=0.6, label=tname, color=tier_colors[t])
    else:
        axes[0].hist(en, bins=50, alpha=0.7, color="steelblue")
        axes[1].hist(ed, bins=50, alpha=0.7, color="darkorange")
    for ax, label in zip(axes, ["ε^norm", "ε^dir"]):
        ax.axvline(meta["epsilon_target"], color="k", ls="--", lw=1, label=f"ε={meta['epsilon_target']}")
        ax.set_xlabel(label); ax.set_ylabel("count"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[0].set_title(f"{meta['setting_id']} s{meta['seed']}: ε^norm")
    axes[1].set_title(f"ε^dir (r=100)")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  [plot] {out_path}")


def plot_rank_ablation(rank_results, out_path, setting_id, seed):
    plt = _plt()
    if plt is None: return
    if not rank_results: return
    ranks = sorted(rank_results.keys())
    med_dir  = [rank_results[r]["eps_dir_med"]  for r in ranks]
    med_norm = [rank_results[r]["eps_norm_med"] for r in ranks]
    fallback = [rank_results[r]["fallback_rate"] for r in ranks]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ranks, med_dir,  "o-", label="ε^dir (Nyström)", color="darkorange")
    axes[0].plot(ranks, med_norm, "s--", label="ε^norm",          color="steelblue")
    axes[0].set_xlabel("Rank r"); axes[0].set_ylabel("Median per-instance ε")
    axes[0].set_title(f"{setting_id} s{seed}: rank ablation")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(ranks, [max(f, 1e-9) for f in fallback], "o-", color="firebrick")
    axes[1].set_xlabel("Rank r"); axes[1].set_ylabel("Fallback rate")
    axes[1].set_title("Fallback rate vs rank"); axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  [plot] {out_path}")


# ---------------------------------------------------------------------------
# Per-run certification
# ---------------------------------------------------------------------------

def certify_setting_seed(setting_id, seed, runs_dir, cert_dir):
    run_dir = os.path.join(runs_dir, setting_id, f"seed_{seed}")
    if not os.path.isdir(run_dir):
        print(f"[P18-cert] {setting_id}/seed_{seed}: run dir not found: {run_dir}")
        return None

    os.makedirs(cert_dir, exist_ok=True)

    print(f"\n[P18-cert] === {setting_id} seed={seed} ===")

    # Load tier labels if available
    tier_path = os.path.join(run_dir, "tier_labels.npy")
    tier_labels = np.load(tier_path) if os.path.exists(tier_path) else None

    # Main certificate at r=100
    result = certify_run(run_dir, rank=100, verbose=True)
    if result is None: return None

    meta = result["meta"]
    tag = f"p18_{setting_id}_seed{seed}"

    # Sanity checks
    ok = run_sanity_checks(result, tag)

    # Tier summary
    if tier_labels is not None:
        tier_summary(result, tier_labels, tag)

    summarize_run(setting_id, seed, result, tag)

    # Rank ablation (Table G)
    print(f"\n  [rank ablation]")
    rank_results = rank_ablation(run_dir, save_prefix=os.path.join(cert_dir, tag))
    print(f"  rank  |  ε^dir med  |  ε^norm med  |  fallback")
    for r in sorted(rank_results.keys()):
        rr = rank_results[r]
        print(f"  {r:5d}  |  {rr['eps_dir_med']:10.4f}  |  {rr['eps_norm_med']:11.4f}  |  {rr['fallback_rate']:.6f}")

    # Save results
    summary = {
        "setting_id":    setting_id,
        "seed":          seed,
        "regime":        meta.get("regime"),
        "dataset":       meta.get("dataset"),
        "epsilon_target": meta.get("epsilon_target"),
        "n":             result["n"],
        "T_acc":         result["T_acc"],
        "sanity_ok":     ok,
        "fallback_rate": float(result["fallback_rate"]),
        "missing_steps_charged": int(result.get("missing_steps_charged", 0)),
        "eps_norm_mean": float(result["eps_norm"].mean()),
        "eps_norm_med":  float(np.median(result["eps_norm"])),
        "eps_norm_cv":   float(result["eps_norm"].std() / result["eps_norm"].mean()),
        "eps_norm_max":  float(result["eps_norm"].max()),
        "eps_dir_mean":  float(result["eps_dir"].mean()),
        "eps_dir_med":   float(np.median(result["eps_dir"])),
        "eps_dir_cv":    float(result["eps_dir"].std() / result["eps_dir"].mean()),
        "eps_dir_max":   float(result["eps_dir"].max()),
        "rank_ablation": rank_results,
        "best_test_acc": meta.get("best_test_acc"),
    }
    if tier_labels is not None:
        for t, tname in {0: "head", 1: "mid", 2: "tail"}.items():
            mask = (tier_labels == t)
            if mask.sum() == 0: continue
            summary[f"eps_norm_{tname}"] = float(result["eps_norm"][mask].mean())
            summary[f"eps_dir_{tname}"]  = float(result["eps_dir"][mask].mean())

    summ_path = os.path.join(cert_dir, f"{tag}_summary.json")
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [cert] Summary: {summ_path}")

    # Save per-example arrays
    np.save(os.path.join(cert_dir, f"{tag}_eps_norm.npy"), result["eps_norm"].astype(np.float32))
    np.save(os.path.join(cert_dir, f"{tag}_eps_dir.npy"),  result["eps_dir"].astype(np.float32))

    # Plots
    plot_scatter(result, tier_labels,
                 os.path.join(cert_dir, f"{tag}_scatter.png"))
    plot_hist(result, tier_labels,
              os.path.join(cert_dir, f"{tag}_hist.png"))
    plot_rank_ablation(rank_results,
                       os.path.join(cert_dir, f"{tag}_rank_ablation.png"),
                       setting_id, seed)

    return summary


# ---------------------------------------------------------------------------
# Seed aggregation
# ---------------------------------------------------------------------------

def aggregate_seeds(setting_id, cert_dir, n_seeds):
    summaries = []
    for seed in range(n_seeds):
        p = os.path.join(cert_dir, f"p18_{setting_id}_seed{seed}_summary.json")
        if os.path.exists(p):
            with open(p) as f: summaries.append(json.load(f))

    if not summaries: return None
    result = {"n_seeds": len(summaries)}
    for key in ["eps_norm_med", "eps_dir_med", "eps_norm_cv", "eps_dir_cv", "best_test_acc"]:
        vals = [s[key] for s in summaries if key in s and s[key] is not None]
        if vals:
            result[f"{key}_median"] = float(np.median(vals))
            result[f"{key}_q25"]    = float(np.percentile(vals, 25))
            result[f"{key}_q75"]    = float(np.percentile(vals, 75))
    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_summaries):
    hdr = (f"{'Setting':8s}  {'Regime':6s}  {'Dataset':14s}  {'ε':4s}  "
           f"{'ε^norm med':10s}  {'ε^dir med':9s}  {'CV norm':7s}  {'CV dir':7s}  "
           f"{'seeds':5s}  {'acc':6s}")
    print(f"\n{'='*len(hdr)}\n  Phase 18 Certificate Summary\n{'='*len(hdr)}")
    print(f"  {hdr}")
    print(f"  {'-'*len(hdr)}")
    for sid, agg in sorted(all_summaries.items()):
        if agg is None: continue
        def _g(k, fmt=".4f"):
            v = agg.get(f"{k}_median", float("nan"))
            return f"{v:{fmt}}" if not (isinstance(v, float) and np.isnan(v)) else "  N/A  "
        # Get regime/dataset from any seed's summary
        summ = None
        for seed in range(10):
            p = os.path.join(CERT_DIR, f"p18_{sid}_seed{seed}_summary.json")
            if os.path.exists(p):
                with open(p) as f: summ = json.load(f); break
        regime  = summ.get("regime",  "?") if summ else "?"
        dataset = summ.get("dataset", "?") if summ else "?"
        eps_t   = summ.get("epsilon_target", 0) if summ else 0
        acc     = _g("best_test_acc", ".3f")
        print(f"  {sid:8s}  {regime:6s}  {dataset:14s}  {eps_t:4.0f}  "
              f"{_g('eps_norm_med'):10s}  {_g('eps_dir_med'):9s}  "
              f"{_g('eps_norm_cv'):7s}  {_g('eps_dir_cv'):7s}  "
              f"{agg['n_seeds']:5d}  {acc:6s}")
    print(f"{'='*len(hdr)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SEEDS_PER_SETTING = {
    "S1": 3, "S2": 3, "S3": 1,
    "S4a": 3, "S4b": 3, "S4c": 3, "S5": 3,
    "S6": 1, "S7": 1, "S9": 1,
}


def main():
    parser = argparse.ArgumentParser(description="Phase 18 certificate computation")
    parser.add_argument("--setting",     type=str, default=None)
    parser.add_argument("--seed",        type=int, default=None)
    parser.add_argument("--all_minimal", action="store_true")
    parser.add_argument("--all",         action="store_true")
    parser.add_argument("--table",       action="store_true")
    parser.add_argument("--runs_dir",    type=str, default=RUNS_DIR)
    parser.add_argument("--cert_dir",    type=str, default=CERT_DIR)
    args = parser.parse_args()

    os.makedirs(args.cert_dir, exist_ok=True)

    if args.setting:
        settings_to_run = [args.setting]
    elif args.all_minimal:
        settings_to_run = list(SETTINGS_MINIMAL)
    elif args.all or args.table:
        settings_to_run = list(SETTINGS_ALL)
    else:
        print("[P18-cert] No setting specified. Use --setting S2, --all_minimal, or --table.")
        return

    all_agg = {}
    for sid in settings_to_run:
        n_seeds = SEEDS_PER_SETTING.get(sid, 1)
        seeds = [args.seed] if args.seed is not None else list(range(n_seeds))
        for seed in seeds:
            certify_setting_seed(sid, seed, args.runs_dir, args.cert_dir)
        agg = aggregate_seeds(sid, args.cert_dir, n_seeds)
        all_agg[sid] = agg
        if agg is not None:
            agg_path = os.path.join(args.cert_dir, f"p18_{sid}_aggregate.json")
            with open(agg_path, "w") as f:
                json.dump({"setting_id": sid, **agg}, f, indent=2)

    if args.table or (not args.seed and len(all_agg) > 1):
        # Load any existing aggregates
        for sid in settings_to_run:
            if sid not in all_agg or all_agg[sid] is None:
                p = os.path.join(args.cert_dir, f"p18_{sid}_aggregate.json")
                if os.path.exists(p):
                    with open(p) as f: all_agg[sid] = json.load(f)
        print_summary_table(all_agg)

    print(f"\n[P18-cert] Done. Results in {args.cert_dir}")


if __name__ == "__main__":
    main()
