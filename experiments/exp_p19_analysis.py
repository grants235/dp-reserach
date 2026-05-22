#!/usr/bin/env python3
"""
Phase 19 Analysis: Tables, Figures, and Gaussian Validation
============================================================

Spec: phase19_spex.md, Sections 11–12.

Produces:
  Table 1: Norm flatness (mean/median/CV/fraction within 1% of C)
  Table 2: Certified bound spread and masking
  Table 3: Headline LiRA correlation (requires LiRA results from exp_p19_lira.py)
  Table 4: Filter discretization overhead
  Table 5: Non-CLIP robustness (F5 tier separation)
  Figure 1: Headline masking scatter (F1/LF1 with LiRA D_i coloring)
  Figure 2: Non-CLIP masking by tier (F5)
  Gaussian validation: KS test for sampling-uncertainty Gaussian approx (§12)

Usage:
  python experiments/exp_p19_analysis.py --all
  python experiments/exp_p19_analysis.py --table 1
  python experiments/exp_p19_analysis.py --table 2
  python experiments/exp_p19_analysis.py --table 3
  python experiments/exp_p19_analysis.py --gaussian_validation --run F1 --seed 0
"""

import os, sys, json, argparse, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RUNS_DIR = "./runs/p19"
CERT_DIR = "./certs/p19"
LIRA_DIR = "./lira/p19"
OUT_DIR  = "./results/p19"

ALPHA_GRID = np.concatenate([
    np.arange(1.5,  10,    0.5),
    np.arange(10,   100,   2.0),
    np.arange(100,  1000,  20.0),
    np.arange(1000, 5001,  100.0),
]).astype(np.float64)


def _plt():
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  [warn] matplotlib not available"); return None


def _load_cert(run_id, seed, cert_dir=CERT_DIR):
    tag = f"p19_{run_id}_seed{seed}"
    out = {}
    for name in ["epsilon_cert_norm", "epsilon_cert_dir_rank_100",
                 "Bcert_norm", "Bcert_dir_rank_100",
                 "C_realized_norm", "C_realized_dir_rank_100",
                 "epsilon_realized_norm", "epsilon_realized_dir_rank_100"]:
        p = os.path.join(cert_dir, f"{tag}_{name}.npy")
        if os.path.exists(p): out[name] = np.load(p)
    summ_p = os.path.join(cert_dir, f"{tag}_summary.json")
    if os.path.exists(summ_p):
        with open(summ_p) as f: out["summary"] = json.load(f)
    return out


def _load_run(run_id, seed, runs_dir=RUNS_DIR):
    run_dir = os.path.join(runs_dir, run_id, f"seed_{seed}")
    out = {"run_dir": run_dir}
    for name in ["clipped_norms", "losses", "labels", "tier_labels",
                 "realized_batch_sizes", "class_counts"]:
        p = os.path.join(run_dir, f"{name}.npy")
        if os.path.exists(p): out[name] = np.load(p)
    meta_p = os.path.join(run_dir, "metadata.json")
    if os.path.exists(meta_p):
        with open(meta_p) as f: out["meta"] = json.load(f)
    return out


def _load_lira(lira_id, lira_dir=LIRA_DIR):
    d = os.path.join(lira_dir, lira_id)
    out = {}
    for name in ["lira_scores_members", "lira_scores_nonmembers",
                 "llr_dp_members", "llr_dp_nonmembers",
                 "targets_members", "targets_nonmembers",
                 "D_lira_members", "D_lira_nonmembers"]:
        p = os.path.join(d, f"{name}.npy")
        if os.path.exists(p): out[name] = np.load(p)
    summ = os.path.join(d, "lira_summary.json")
    if os.path.exists(summ):
        with open(summ) as f: out["summary"] = json.load(f)
    return out


# ---------------------------------------------------------------------------
# Table 1: Norm flatness
# ---------------------------------------------------------------------------

def table_1(runs_dir=RUNS_DIR, out_dir=OUT_DIR):
    """
    Table 1: Norm distribution diagnostics.
    Rows: F1 seed 0, F2 seed 0, F5 seeds 0,1,2 (aggregated).
    Columns: mean norm, median norm, CV, fraction within 1% of C.
    """
    rows = [("F1", [0]), ("F2", [0]), ("F5", [0, 1, 2])]
    C = 1.0
    print(f"\n{'='*72}")
    print(f"  Table 1: Norm Flatness (C={C})")
    print(f"{'='*72}")
    hdr = f"  {'Run':4s} {'seed':4s}  {'mean':8s}  {'median':8s}  {'CV':7s}  {'frac≤1% C':10s}  {'n':8s}"
    print(hdr); print(f"  {'-'*len(hdr.lstrip())}")

    all_rows = []
    for run_id, seeds in rows:
        for seed in seeds:
            run = _load_run(run_id, seed, runs_dir)
            if "clipped_norms" not in run: continue
            norms = run["clipped_norms"]    # (n, T)
            # Use last 10 steps (or all if T<10) for norm statistics
            T = norms.shape[1]
            norms_use = norms[:, max(0, T-10):].ravel()
            mean_n   = norms_use.mean()
            med_n    = np.median(norms_use)
            cv_n     = norms_use.std() / max(mean_n, 1e-15)
            frac_sat = (norms_use >= 0.99 * C).mean()   # fraction within 1% of C
            n        = norms.shape[0]
            print(f"  {run_id:4s} {seed:4d}  {mean_n:8.5f}  {med_n:8.5f}  "
                  f"{cv_n:7.4f}  {frac_sat:10.4f}  {n:8d}")
            all_rows.append({"run_id": run_id, "seed": seed,
                             "mean_norm": float(mean_n), "median_norm": float(med_n),
                             "cv": float(cv_n), "frac_within_1pct_C": float(frac_sat)})

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "table1_norm_flatness.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n  [saved] {out_dir}/table1_norm_flatness.json")


# ---------------------------------------------------------------------------
# Table 2: Certified bound spread and masking
# ---------------------------------------------------------------------------

def table_2(cert_dir=CERT_DIR, runs_dir=RUNS_DIR, out_dir=OUT_DIR):
    """
    Table 2: Certified ε spread and masking gap.
    Uses CERTIFIED grid-filtered values only (not realized).
    Rows: F1, F2, F5 (each seed, then aggregated).
    """
    print(f"\n{'='*72}")
    print(f"  Table 2: Certified Bound Spread and Masking (r=100)")
    print(f"{'='*72}")
    hdr = (f"  {'Run':4s} {'S':1s}  "
           f"{'norm_med':8s}  {'norm_cv':7s}  {'norm_p5':7s}  {'norm_p95':8s}  "
           f"{'dir_med':7s}  {'dir_cv':6s}  "
           f"{'gap_med':7s}  {'ratio_med':9s}  {'n':6s}")
    print(hdr); print(f"  {'-'*len(hdr.lstrip())}")

    all_rows = []
    for run_id in ["F1", "F2", "F5"]:
        for seed in range(3):
            cert = _load_cert(run_id, seed, cert_dir)
            if "epsilon_cert_norm" not in cert: continue
            en = cert["epsilon_cert_norm"]
            ed = cert["epsilon_cert_dir_rank_100"]
            n  = len(en)
            gap   = en - ed
            ratio = 1.0 - ed / np.maximum(en, 1e-15)
            print(f"  {run_id:4s} {seed:1d}  "
                  f"{np.median(en):8.4f}  {en.std()/en.mean():7.4f}  "
                  f"{np.percentile(en,5):7.4f}  {np.percentile(en,95):8.4f}  "
                  f"{np.median(ed):7.4f}  {ed.std()/ed.mean():6.4f}  "
                  f"{np.median(gap):7.4f}  {np.median(ratio):9.4f}  {n:6d}")
            all_rows.append({
                "run_id": run_id, "seed": seed,
                "norm_med": float(np.median(en)), "norm_cv": float(en.std()/en.mean()),
                "norm_p5": float(np.percentile(en,5)), "norm_p95": float(np.percentile(en,95)),
                "dir_med": float(np.median(ed)), "dir_cv": float(ed.std()/ed.mean()),
                "dir_p5": float(np.percentile(ed,5)), "dir_p95": float(np.percentile(ed,95)),
                "gap_med": float(np.median(gap)), "gap_p5": float(np.percentile(gap,5)),
                "gap_p95": float(np.percentile(gap,95)),
                "ratio_med": float(np.median(ratio)), "ratio_p95": float(np.percentile(ratio,95)),
                "n": n,
            })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "table2_certified_spread.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n  [saved] {out_dir}/table2_certified_spread.json")


# ---------------------------------------------------------------------------
# Table 3: Headline LiRA correlation
# ---------------------------------------------------------------------------

def table_3(cert_dir=CERT_DIR, lira_dir=LIRA_DIR, runs_dir=RUNS_DIR, out_dir=OUT_DIR):
    """
    Table 3: Headline LiRA correlation (F1/LF1).
    Requires LiRA results from exp_p19_lira.py.
    Correlates certified ε with D_i^LiRA using Spearman rank correlation.
    """
    try:
        from scipy.stats import spearmanr
        from scipy.stats import bootstrap
        HAS_SCIPY = True
    except ImportError:
        print("  [Table 3] scipy not available; skipping correlation CIs")
        HAS_SCIPY = False

    print(f"\n{'='*72}")
    print(f"  Table 3: Headline LiRA Correlation (F1/LF1)")
    print(f"{'='*72}")

    lf1 = _load_lira("LF1", lira_dir)
    cert = _load_cert("F1", 0, cert_dir)
    run  = _load_run("F1", 0, runs_dir)

    if not lf1 or "D_lira_members" not in lf1:
        print("  [Table 3] LiRA results not found (run exp_p19_lira.py first).")
        return

    # Align LiRA member targets with certified ε
    # LiRA scores for members: D_i^LiRA
    D_lira = lf1["D_lira_members"]                             # (n_targets,)
    member_local = np.load(os.path.join(runs_dir, "F1", "seed_0", "lira_member_local_idx.npy"))

    if "epsilon_cert_dir_rank_100" not in cert:
        print("  [Table 3] Certified ε not found."); return

    eps_dir_all  = cert["epsilon_cert_dir_rank_100"]           # (n,)
    eps_norm_all = cert["epsilon_cert_norm"]                   # (n,)

    eps_dir_members  = eps_dir_all[member_local]
    eps_norm_members = eps_norm_all[member_local]

    # Also: final losses
    losses_all   = run.get("losses", None)
    loss_final   = losses_all[member_local, -1] if losses_all is not None else None

    # Also: realized ε (diagnostic, clearly labeled)
    if "epsilon_realized_dir_rank_100" in cert:
        eps_real_dir_members = cert["epsilon_realized_dir_rank_100"][member_local]
    else:
        eps_real_dir_members = None

    def _compute_row(y, label):
        """Compute Spearman ρ, CI, rank R², AUC vs D_lira."""
        mask = np.isfinite(y) & np.isfinite(D_lira)
        y_m = y[mask]; D_m = D_lira[mask]
        if len(y_m) < 10: return None

        if HAS_SCIPY:
            rho_val, pval = spearmanr(y_m, D_m)
            # Bootstrap 95% CI
            def _spr(x, y): return spearmanr(x, y).statistic
            bs = bootstrap((y_m, D_m), _spr, n_resamples=1000,
                           paired=True, confidence_level=0.95,
                           random_state=42, method="percentile")
            ci_lo, ci_hi = float(bs.confidence_interval.low), float(bs.confidence_interval.high)
        else:
            from numpy import corrcoef, argsort
            ry = argsort(argsort(y_m)); rd = argsort(argsort(D_m))
            rho_val = float(corrcoef(ry, rd)[0,1])
            ci_lo = ci_hi = float('nan')

        # Rank-rank R²
        ry = np.argsort(np.argsort(y_m)); rd = np.argsort(np.argsort(D_m))
        r2 = float(np.corrcoef(ry, rd)[0, 1]**2)

        print(f"    {label:35s}  ρ={rho_val:+.4f}  CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
              f"R²={r2:.4f}")
        return {"label": label, "rho": float(rho_val), "ci_lo": ci_lo, "ci_hi": ci_hi, "R2": r2}

    rows = []
    print(f"  {'Feature':35s}  {'Spearman ρ':10s}  {'95% CI':18s}  {'R²':6s}")
    r = _compute_row(eps_dir_members, "certified ε^dir (r=100)")
    if r: rows.append(r)
    r = _compute_row(eps_norm_members, "certified ε^norm")
    if r: rows.append(r)
    if eps_real_dir_members is not None:
        r = _compute_row(eps_real_dir_members, "realized ε^dir (diagnostic, not cert)")
        if r: rows.append(r)
    if loss_final is not None:
        r = _compute_row(loss_final, "final training loss")
        if r: rows.append(r)
    if "tier_labels" in run:
        r = _compute_row(run["tier_labels"][member_local].astype(float), "tier label")
        if r: rows.append(r)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "table3_lira_correlation.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  [saved] {out_dir}/table3_lira_correlation.json")


# ---------------------------------------------------------------------------
# Table 4: Filter discretization overhead
# ---------------------------------------------------------------------------

def table_4(cert_dir=CERT_DIR, out_dir=OUT_DIR):
    """
    Table 4: Filter discretization overhead at best α.
    B_{i,cert}(α*) − C_i(α*) for direction and norm.
    """
    print(f"\n{'='*72}")
    print(f"  Table 4: Filter Discretization Overhead (r=100)")
    print(f"{'='*72}")

    rows = []
    for run_id in ["F1", "F5"]:
        for seed in [0]:
            cert = _load_cert(run_id, seed, cert_dir)
            if "Bcert_dir_rank_100" not in cert: continue
            if "C_realized_dir_rank_100" not in cert: continue

            B_cert_dir = cert["Bcert_dir_rank_100"]       # (n, n_alpha)
            C_real_dir = cert["C_realized_dir_rank_100"]  # (n, n_alpha)
            B_cert_norm = cert["Bcert_norm"]              # (n, n_alpha)
            C_real_norm = cert["C_realized_norm"]         # (n, n_alpha)
            en = cert["epsilon_cert_norm"]
            ed = cert["epsilon_cert_dir_rank_100"]

            # Overhead per alpha
            overhead_dir  = B_cert_dir  - C_real_dir   # (n, n_alpha)
            overhead_norm = B_cert_norm - C_real_norm   # (n, n_alpha)

            # Aggregate per example: sum over alpha? No — use at best alpha.
            # Best alpha: argmin over alpha of certified eps value
            # For dir: use the alpha that achieves min B + log(1/delta)/(alpha-1)
            # We can proxy by using the alpha with min overhead in the meaningful range.
            # Simpler: report overhead as the sum Σ_α Δ_α (total budget slack).
            # But spec says "at selected α*" — use the alpha that achieves the min eps.

            # Load best alpha arrays if available
            best_dir_path  = os.path.join(cert_dir, f"p19_{run_id}_seed{seed}_best_alpha_cert_dir_rank_100.npy")
            best_norm_path = os.path.join(cert_dir, f"p19_{run_id}_seed{seed}_best_alpha_cert_norm.npy")

            if os.path.exists(best_dir_path):
                best_alpha_dir = np.load(best_dir_path).astype(int)
                ovh_dir  = overhead_dir[np.arange(len(best_alpha_dir)), best_alpha_dir]
                cert_val_dir = B_cert_dir[np.arange(len(best_alpha_dir)), best_alpha_dir]
            else:
                # Use minimum overhead column as proxy
                ovh_dir  = overhead_dir.min(axis=1)
                cert_val_dir = B_cert_dir.min(axis=1)

            if os.path.exists(best_norm_path):
                best_alpha_norm = np.load(best_norm_path).astype(int)
                ovh_norm  = overhead_norm[np.arange(len(best_alpha_norm)), best_alpha_norm]
                cert_val_norm = B_cert_norm[np.arange(len(best_alpha_norm)), best_alpha_norm]
            else:
                ovh_norm  = overhead_norm.min(axis=1)
                cert_val_norm = B_cert_norm.min(axis=1)

            pct_dir  = 100.0 * ovh_dir  / np.maximum(cert_val_dir,  1e-15)
            pct_norm = 100.0 * ovh_norm / np.maximum(cert_val_norm, 1e-15)

            print(f"\n  {run_id} seed={seed} (direction-aware):")
            print(f"    median overhead:  {np.median(ovh_dir):.6f}")
            print(f"    p95 overhead:     {np.percentile(ovh_dir,95):.6f}")
            print(f"    max overhead:     {ovh_dir.max():.6f}")
            print(f"    median overhead%: {np.median(pct_dir):.3f}%")
            print(f"    p95 overhead%:    {np.percentile(pct_dir,95):.3f}%")

            print(f"  {run_id} seed={seed} (norm-based):")
            print(f"    median overhead:  {np.median(ovh_norm):.6f}")
            print(f"    p95 overhead:     {np.percentile(ovh_norm,95):.6f}")
            print(f"    max overhead:     {ovh_norm.max():.6f}")
            print(f"    median overhead%: {np.median(pct_norm):.3f}%")
            print(f"    p95 overhead%:    {np.percentile(pct_norm,95):.3f}%")

            rows.append({
                "run_id": run_id, "seed": seed,
                "dir_overhead_med": float(np.median(ovh_dir)),
                "dir_overhead_p95": float(np.percentile(ovh_dir,95)),
                "dir_overhead_max": float(ovh_dir.max()),
                "dir_overhead_pct_med": float(np.median(pct_dir)),
                "dir_overhead_pct_p95": float(np.percentile(pct_dir,95)),
                "norm_overhead_med": float(np.median(ovh_norm)),
                "norm_overhead_p95": float(np.percentile(ovh_norm,95)),
                "norm_overhead_max": float(ovh_norm.max()),
                "norm_overhead_pct_med": float(np.median(pct_norm)),
                "norm_overhead_pct_p95": float(np.percentile(pct_norm,95)),
            })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "table4_discretization_overhead.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  [saved] {out_dir}/table4_discretization_overhead.json")


# ---------------------------------------------------------------------------
# Table 5: Non-CLIP robustness (F5)
# ---------------------------------------------------------------------------

def table_5(cert_dir=CERT_DIR, lira_dir=LIRA_DIR, runs_dir=RUNS_DIR, out_dir=OUT_DIR):
    """
    Table 5: Non-CLIP robustness (F5 tier separation).
    Rows: tier means of certified ε^dir and ε^norm, masking discount by tier, seed stability.
    """
    print(f"\n{'='*72}")
    print(f"  Table 5: Non-CLIP Robustness (F5, r=100)")
    print(f"{'='*72}")

    tier_names = {0: "head", 1: "mid", 2: "tail"}
    all_rows = []

    for seed in [0, 1, 2]:
        cert = _load_cert("F5", seed, cert_dir)
        run  = _load_run("F5", seed, runs_dir)
        if "epsilon_cert_norm" not in cert: continue
        if "tier_labels" not in run: continue

        en = cert["epsilon_cert_norm"]
        ed = cert["epsilon_cert_dir_rank_100"]
        tier_labels = run["tier_labels"]

        print(f"\n  F5 seed={seed}:")
        hdr = f"    {'tier':6s}  {'n':5s}  {'ε^norm':7s}  {'ε^dir':6s}  {'masking%':9s}"
        print(hdr)
        row = {"seed": seed, "tiers": {}}
        for t_id, t_name in tier_names.items():
            mask = (tier_labels == t_id)
            if mask.sum() == 0: continue
            mn  = float(en[mask].mean())
            md  = float(ed[mask].mean())
            msk = 100.0 * (1.0 - md / max(mn, 1e-15))
            print(f"    {t_name:6s}  {mask.sum():5d}  {mn:7.4f}  {md:6.4f}  {msk:9.2f}%")
            row["tiers"][t_name] = {
                "n": int(mask.sum()),
                "eps_norm_mean": mn, "eps_dir_mean": md,
                "masking_pct": float(msk),
            }
        all_rows.append(row)

    # Seed stability: compare tier means across seeds
    if len(all_rows) >= 2:
        print(f"\n  Seed stability (coefficient of variation across seeds):")
        for t_id, t_name in tier_names.items():
            dir_vals = [r["tiers"][t_name]["eps_dir_mean"]
                        for r in all_rows if t_name in r["tiers"]]
            if len(dir_vals) < 2: continue
            cv = np.std(dir_vals) / max(np.mean(dir_vals), 1e-15)
            print(f"    {t_name:6s}: ε^dir mean across seeds={np.mean(dir_vals):.4f}  CV={cv:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "table5_wrn_robustness.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n  [saved] {out_dir}/table5_wrn_robustness.json")


# ---------------------------------------------------------------------------
# Figure 1: Headline masking scatter (F1/LF1 with LiRA D_i)
# ---------------------------------------------------------------------------

def figure_1(cert_dir=CERT_DIR, lira_dir=LIRA_DIR, runs_dir=RUNS_DIR, out_dir=OUT_DIR):
    """
    Figure 1: Scatter plot of ε^dir and ε^norm vs D_i^LiRA, colored by tier.
    Requires LiRA results from exp_p19_lira.py.
    """
    plt = _plt(); os.makedirs(out_dir, exist_ok=True)
    lf1  = _load_lira("LF1", lira_dir)
    cert = _load_cert("F1", 0, cert_dir)
    run  = _load_run("F1", 0, runs_dir)

    if not lf1 or "D_lira_members" not in lf1:
        print("  [Fig 1] LiRA results not found."); return

    member_local = np.load(os.path.join(runs_dir, "F1", "seed_0", "lira_member_local_idx.npy"))
    D_lira  = lf1["D_lira_members"]
    eps_dir = cert.get("epsilon_cert_dir_rank_100")
    eps_norm = cert.get("epsilon_cert_norm")
    if eps_dir is None or eps_norm is None: return

    eps_dir_m  = eps_dir[member_local]
    eps_norm_m = eps_norm[member_local]
    tiers = run.get("tier_labels")
    tier_m = tiers[member_local] if tiers is not None else None

    if plt is None: return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tier_colors = {0: "steelblue", 1: "orange", 2: "firebrick"}
    tier_names  = {0: "head", 1: "mid", 2: "tail"}

    for ax, (eps_v, lbl) in zip(axes, [(eps_dir_m, "ε^dir cert"), (eps_norm_m, "ε^norm cert")]):
        if tier_m is not None:
            for t_id, t_name in tier_names.items():
                mask = (tier_m == t_id)
                if mask.sum() == 0: continue
                ax.scatter(D_lira[mask], eps_v[mask], c=tier_colors[t_id],
                           s=6, alpha=0.5, label=t_name)
        else:
            ax.scatter(D_lira, eps_v, s=6, alpha=0.5)
        ax.set_xlabel("D_i^LiRA"); ax.set_ylabel(lbl)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[0].set_title("Figure 1a: D_i^LiRA vs ε^dir (headline)")
    axes[1].set_title("Figure 1b: D_i^LiRA vs ε^norm")
    fig.suptitle("F1/LF1: Direction-Aware Masking vs LiRA Distinguishability", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "figure1_headline_scatter.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [Fig 1] {path}")


# ---------------------------------------------------------------------------
# Figure 2: Non-CLIP masking by tier (F5)
# ---------------------------------------------------------------------------

def figure_2(cert_dir=CERT_DIR, runs_dir=RUNS_DIR, out_dir=OUT_DIR):
    """
    Figure 2: Certified masking discount by tier for F5, across seeds.
    """
    plt = _plt(); os.makedirs(out_dir, exist_ok=True)
    if plt is None: return

    tier_names = {0: "head", 1: "mid", 2: "tail"}
    tier_colors = {0: "steelblue", 1: "orange", 2: "firebrick"}

    fig, ax = plt.subplots(figsize=(7, 4))
    found = False

    for seed in [0, 1, 2]:
        cert = _load_cert("F5", seed, cert_dir)
        run  = _load_run("F5", seed, runs_dir)
        if "epsilon_cert_norm" not in cert or "tier_labels" not in run: continue
        found = True
        en = cert["epsilon_cert_norm"]; ed = cert["epsilon_cert_dir_rank_100"]
        tier_labels = run["tier_labels"]

        for t_id, t_name in tier_names.items():
            mask = (tier_labels == t_id)
            if mask.sum() == 0: continue
            masking = 1.0 - ed[mask].mean() / max(en[mask].mean(), 1e-15)
            ax.scatter(t_id + 0.1 * (seed - 1), masking,
                       color=tier_colors[t_id], marker=["o","s","^"][seed],
                       s=80, label=f"seed {seed}" if t_id == 0 else "")

    if found:
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["head", "mid", "tail"])
        ax.set_xlabel("Tier"); ax.set_ylabel("Certified masking ratio (1 − ε^dir/ε^norm)")
        ax.set_title("Figure 2: F5 WRN Non-CLIP Masking by Tier")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, "figure2_wrn_tier_masking.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  [Fig 2] {path}")
    else:
        plt.close(fig); print("  [Fig 2] No F5 data found.")


# ---------------------------------------------------------------------------
# Section 12: Gaussian approximation validation
# ---------------------------------------------------------------------------

def gaussian_validation(run_id, seed, runs_dir=RUNS_DIR, out_dir=OUT_DIR,
                        n_samples=5000, n_targets_per_tier=3, n_projections=10):
    """
    Validate the sampling-uncertainty Gaussian approximation (spec §12).

    At selected steps (early/middle/late), for representative targets:
      1. Load G_t (or Y_projections as proxy).
      2. Draw M=5000 independent Poisson batches.
      3. For each projection direction u_k, compute <u_k, G_{-i,t}> empirically.
      4. Compare to fitted Gaussian.

    Reports: median KS distance, max KS distance, skewness, excess kurtosis.
    """
    try:
        from scipy.stats import kstest, skew, kurtosis, norm
    except ImportError:
        print("  [GV] scipy not available; skipping Gaussian validation.")
        return

    run_dir  = os.path.join(runs_dir, run_id, f"seed_{seed}")
    meta_p   = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta_p):
        print(f"  [GV] metadata.json not found: {run_dir}"); return
    with open(meta_p) as f:
        meta = json.load(f)

    q   = float(meta["q"])
    rho = float(meta["rho"])
    n   = int(meta["n_train"])
    T   = int(meta["T_train"])

    # Load Y_projections as proxy for G_t directions
    yp_path = os.path.join(run_dir, "Y_projections.npy")
    if not os.path.exists(yp_path):
        print(f"  [GV] Y_projections.npy not found"); return
    Y_proj = np.load(yp_path)      # (n, T, r_max)

    tier_path = os.path.join(run_dir, "tier_labels.npy")
    tier_labels = np.load(tier_path) if os.path.exists(tier_path) else None

    # Select steps: early ≈ 0.1T, middle ≈ 0.5T, late ≈ 0.9T
    steps = {
        "early":  max(0, round(0.1 * T) - 1),
        "middle": max(0, round(0.5 * T) - 1),
        "late":   max(0, round(0.9 * T) - 1),
    }
    # Limit to F1 seed 0 or F5 middle (spec)
    if run_id == "F5":
        steps = {"middle": steps["middle"]}

    # Select target examples: from each tier (head/mid/tail) or uniformly
    rng = np.random.default_rng(12345)
    targets = []
    if tier_labels is not None:
        for t_id in [0, 1, 2]:
            avail = np.where(tier_labels == t_id)[0]
            if len(avail) == 0: continue
            for idx in rng.choice(avail, size=min(n_targets_per_tier, len(avail)), replace=False):
                targets.append((int(idx), {0:"head",1:"mid",2:"tail"}[t_id]))
    else:
        for idx in rng.choice(n, size=min(9, n), replace=False):
            targets.append((int(idx), "all"))

    os.makedirs(out_dir, exist_ok=True)
    results = []

    for phase, t_idx in steps.items():
        t_idx = min(t_idx, T - 1)
        print(f"\n  [GV] {run_id} seed={seed}  phase={phase} step={t_idx}")

        # Use Y_proj[:,t,:] as the gradient proxy for each example
        # Y_proj[i,t,:r] = Y_t^T ḡ_{i,t} — a low-dimensional projection of gradient
        # This is not the full gradient G_t, but it captures the covariance structure
        # For the Gaussian validation of sampling uncertainty, we use:
        #   G_{-i,t} = Σ_{j≠i} I_{j,t} Y_proj[j,t,:]  (projected)
        # vs fitted Gaussian with covariance ρ Σ_{j≠i} Y_proj[j,t,:] Y_proj[j,t,:]^T
        yp_t = Y_proj[:, t_idx, :n_projections].astype(np.float64)  # (n, K)
        K = yp_t.shape[1]

        # Expected Cov(G_{-i,t}) ≈ ρ Σ_{j≠i} yp_j yp_j^T
        # For each target, focus on the top projection direction
        for (i_target, tier_name) in targets[:3]:    # limit for speed
            # Leave-one-out: directions from all j ≠ i
            yp_others = np.concatenate([yp_t[:i_target], yp_t[i_target+1:]], axis=0)
            # (n-1, K)

            # Theoretical distribution of <u_k, G_{-i,t}> for u_k = standard basis
            # Σ_{j≠i} q(1-q) yp_j[k]^2 per projection dimension k
            sigma_k2 = rho * (yp_others ** 2).sum(axis=0)   # (K,)
            mu_k     = q * yp_others.sum(axis=0)             # (K,) expected mean

            # Empirical: draw M Poisson batches
            I_others = rng.random((n_samples, n - 1)) < q    # (M, n-1) Poisson inclusions
            G_proj   = I_others @ yp_others                   # (M, K)

            # KS test per projection dimension
            ks_stats = []
            skewness_vals = []
            kurt_vals = []
            for k in range(K):
                g_k = G_proj[:, k]
                sigma_k = math.sqrt(max(sigma_k2[k], 1e-12))
                # KS test: empirical vs N(mu_k, sigma_k^2)
                stat, _pval = kstest(g_k, "norm", args=(mu_k[k], sigma_k))
                ks_stats.append(float(stat))
                # Standardize
                g_std = (g_k - mu_k[k]) / sigma_k
                skewness_vals.append(float(skew(g_std)))
                kurt_vals.append(float(kurtosis(g_std)))    # excess kurtosis

            med_ks = float(np.median(ks_stats))
            max_ks = float(np.max(ks_stats))
            med_skew = float(np.median(np.abs(skewness_vals)))
            med_kurt = float(np.median(np.abs(kurt_vals)))

            print(f"    i={i_target:5d} ({tier_name:4s}):  "
                  f"KS med={med_ks:.4f}  max={max_ks:.4f}  "
                  f"|skew| med={med_skew:.4f}  |kurt| med={med_kurt:.4f}")

            results.append({
                "run_id": run_id, "seed": seed, "phase": phase, "step": int(t_idx),
                "target_idx": i_target, "tier": tier_name,
                "ks_median": med_ks, "ks_max": max_ks,
                "skewness_abs_med": med_skew, "excess_kurtosis_abs_med": med_kurt,
            })

    out_path = os.path.join(out_dir, f"gaussian_validation_{run_id}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [GV] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 19 analysis")
    parser.add_argument("--all",     action="store_true", help="Run all tables and figures")
    parser.add_argument("--table",   type=str, default=None, choices=["1","2","3","4","5"],
                        help="Run specific table")
    parser.add_argument("--figure",  type=str, default=None, choices=["1","2"],
                        help="Run specific figure")
    parser.add_argument("--gaussian_validation", action="store_true")
    parser.add_argument("--run",      type=str, default="F1", choices=["F1","F2","F5"])
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--runs_dir", type=str, default=RUNS_DIR)
    parser.add_argument("--cert_dir", type=str, default=CERT_DIR)
    parser.add_argument("--lira_dir", type=str, default=LIRA_DIR)
    parser.add_argument("--out_dir",  type=str, default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.all or args.table == "1":
        table_1(args.runs_dir, args.out_dir)
    if args.all or args.table == "2":
        table_2(args.cert_dir, args.runs_dir, args.out_dir)
    if args.all or args.table == "3":
        table_3(args.cert_dir, args.lira_dir, args.runs_dir, args.out_dir)
    if args.all or args.table == "4":
        table_4(args.cert_dir, args.out_dir)
    if args.all or args.table == "5":
        table_5(args.cert_dir, args.lira_dir, args.runs_dir, args.out_dir)
    if args.all or args.figure == "1":
        figure_1(args.cert_dir, args.lira_dir, args.runs_dir, args.out_dir)
    if args.all or args.figure == "2":
        figure_2(args.cert_dir, args.runs_dir, args.out_dir)
    if args.gaussian_validation or args.all:
        run_id = args.run if not args.all else "F1"
        seed   = args.seed if not args.all else 0
        gaussian_validation(run_id, seed, args.runs_dir, args.out_dir)

    print(f"\n[P19-analysis] Done. Results in {args.out_dir}")


if __name__ == "__main__":
    main()
