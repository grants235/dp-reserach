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
  python experiments/exp_p19_analysis.py --all --all_lira_seeds          # use all available LiRA seeds
  python experiments/exp_p19_analysis.py --table 3 --all_lira_seeds
  python experiments/exp_p19_analysis.py --table 3 --lira_seeds 0 1 2   # explicit seeds
  python experiments/exp_p19_analysis.py --table 1
  python experiments/exp_p19_analysis.py --table 2
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

RANK_ABLATION = [10, 25, 50, 100, 200]


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


def _load_lira(lira_id, lira_dir=LIRA_DIR, seed=0):
    d = os.path.join(lira_dir, lira_id, f"seed_{seed}")
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


def _available_lira_seeds(lira_id, lira_dir=LIRA_DIR):
    """Return sorted list of seeds that have a lira_summary.json."""
    base = os.path.join(lira_dir, lira_id)
    seeds = []
    for entry in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        if entry.startswith("seed_"):
            try:
                s = int(entry.split("_")[1])
            except (ValueError, IndexError):
                continue
            if os.path.exists(os.path.join(base, entry, "lira_summary.json")):
                seeds.append(s)
    return seeds


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

def table_3(cert_dir=CERT_DIR, lira_dir=LIRA_DIR, runs_dir=RUNS_DIR, out_dir=OUT_DIR,
            lira_seeds=None):
    """
    Table 3: Headline LiRA correlation (F1/LF1), aggregated across all available seeds.
    Reports per-seed Spearman ρ and mean±std across seeds.
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

    seeds_to_use = lira_seeds if lira_seeds is not None else _available_lira_seeds("LF1", lira_dir)
    if not seeds_to_use:
        seeds_to_use = [0]

    def _compute_row(y, label, D_ref):
        mask = np.isfinite(y) & np.isfinite(D_ref)
        y_m = y[mask]; D_m = D_ref[mask]
        if len(y_m) < 10: return None

        if HAS_SCIPY:
            rho_val, _pval = spearmanr(y_m, D_m)
            def _spr(x, y): return spearmanr(x, y).statistic
            bs = bootstrap((y_m, D_m), _spr, n_resamples=2000,
                           paired=True, confidence_level=0.95,
                           random_state=42, method="percentile")
            ci_lo = float(bs.confidence_interval.low)
            ci_hi = float(bs.confidence_interval.high)
        else:
            ry = np.argsort(np.argsort(y_m)); rd = np.argsort(np.argsort(D_m))
            rho_val = float(np.corrcoef(ry, rd)[0, 1])
            ci_lo = ci_hi = float('nan')

        ry = np.argsort(np.argsort(y_m)); rd = np.argsort(np.argsort(D_m))
        r2 = float(np.corrcoef(ry, rd)[0, 1] ** 2)

        print(f"    {label:52s}  ρ={rho_val:+.4f}  CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
              f"R²={r2:.4f}  n={len(y_m)}")
        return {"label": label, "rho": float(rho_val), "ci_lo": ci_lo, "ci_hi": ci_hi,
                "R2": r2, "n": len(y_m)}

    all_rows = []
    # Collect per-seed ρ values for headline aggregation
    per_seed_dir_rho  = []
    per_seed_norm_rho = []

    print(f"  NOTE: all ρ are Spearman rank correlations across member targets only.")
    print(f"  {'Feature':52s}  {'Spearman ρ':10s}  {'95% CI':18s}  {'R²':6s}  {'n':5s}")

    for s in seeds_to_use:
        lf1  = _load_lira("LF1", lira_dir, seed=s)
        cert = _load_cert("F1", s, cert_dir)
        run  = _load_run("F1", s, runs_dir)

        if not lf1 or "D_lira_members" not in lf1:
            print(f"  [Table 3] LiRA seed={s} not found, skipping.")
            continue
        if "epsilon_cert_dir_rank_100" not in cert:
            print(f"  [Table 3] Certified ε seed={s} not found, skipping.")
            continue

        print(f"\n  -- seed={s} --")
        D_lira       = lf1["D_lira_members"]
        member_local = np.load(
            os.path.join(runs_dir, "F1", f"seed_{s}", "lira_member_local_idx.npy"))

        eps_dir_m  = cert["epsilon_cert_dir_rank_100"][member_local]
        eps_norm_m = cert["epsilon_cert_norm"][member_local]
        losses_all = run.get("losses", None)
        loss_final = losses_all[member_local, -1] if losses_all is not None else None
        eps_real_dir_m = (cert["epsilon_realized_dir_rank_100"][member_local]
                          if "epsilon_realized_dir_rank_100" in cert else None)

        rows_seed = []
        r = _compute_row(eps_dir_m,  f"[s{s}] ε^dir cert r=100", D_lira)
        if r:
            r["seed"] = s; rows_seed.append(r); per_seed_dir_rho.append(r["rho"])
        r = _compute_row(eps_norm_m, f"[s{s}] ε^norm cert", D_lira)
        if r:
            r["seed"] = s; rows_seed.append(r); per_seed_norm_rho.append(r["rho"])
        if eps_real_dir_m is not None:
            r = _compute_row(eps_real_dir_m, f"[s{s}] realized ε^dir (diagnostic)", D_lira)
            if r: r["seed"] = s; rows_seed.append(r)
        if loss_final is not None:
            r = _compute_row(loss_final, f"[s{s}] final training loss", D_lira)
            if r: r["seed"] = s; rows_seed.append(r)
        if "tier_labels" in run:
            r = _compute_row(run["tier_labels"][member_local].astype(float),
                             f"[s{s}] tier label", D_lira)
            if r: r["seed"] = s; rows_seed.append(r)

        # Within-tier partial Spearman
        if "tier_labels" in run:
            tier_arr = run["tier_labels"][member_local]
            print(f"    Within-tier ρ (seed={s}):")
            wt_rhos = []
            for t_id, t_name in {0: "head", 1: "mid", 2: "tail"}.items():
                mask_t = (tier_arr == t_id)
                if mask_t.sum() < 10: continue
                r = _compute_row(eps_dir_m[mask_t],
                                 f"[s{s}]   ε^dir cert [{t_name}]",
                                 D_lira[mask_t])
                if r:
                    r["seed"] = s; r["tier"] = t_name; r["within_tier"] = True
                    wt_rhos.append(r["rho"]); rows_seed.append(r)
                r = _compute_row(eps_norm_m[mask_t],
                                 f"[s{s}]   ε^norm cert [{t_name}]",
                                 D_lira[mask_t])
                if r:
                    r["seed"] = s; r["tier"] = t_name; r["within_tier"] = True
                    rows_seed.append(r)
                if loss_final is not None:
                    r = _compute_row(loss_final[mask_t],
                                     f"[s{s}]   loss [{t_name}]",
                                     D_lira[mask_t])
                    if r:
                        r["seed"] = s; r["tier"] = t_name; r["within_tier"] = True
                        rows_seed.append(r)
            if wt_rhos:
                print(f"    Mean within-tier ρ (ε^dir, seed={s}): {np.mean(wt_rhos):.4f}")

        all_rows.extend(rows_seed)

    # Cross-seed summary
    if len(per_seed_dir_rho) > 1:
        print(f"\n  Cross-seed summary ({len(per_seed_dir_rho)} seeds):")
        print(f"    ε^dir  ρ per seed: {[f'{v:+.4f}' for v in per_seed_dir_rho]}")
        print(f"    ε^dir  mean ρ = {np.mean(per_seed_dir_rho):+.4f}  "
              f"std = {np.std(per_seed_dir_rho):.4f}")
        print(f"    ε^norm ρ per seed: {[f'{v:+.4f}' for v in per_seed_norm_rho]}")
        print(f"    ε^norm mean ρ = {np.mean(per_seed_norm_rho):+.4f}  "
              f"std = {np.std(per_seed_norm_rho):.4f}")
        all_rows.append({
            "label": "AGGREGATE ε^dir cert r=100",
            "seeds": seeds_to_use,
            "rho_per_seed": per_seed_dir_rho,
            "rho_mean": float(np.mean(per_seed_dir_rho)),
            "rho_std":  float(np.std(per_seed_dir_rho)),
        })
        all_rows.append({
            "label": "AGGREGATE ε^norm cert",
            "seeds": seeds_to_use,
            "rho_per_seed": per_seed_norm_rho,
            "rho_mean": float(np.mean(per_seed_norm_rho)),
            "rho_std":  float(np.std(per_seed_norm_rho)),
        })

    if not all_rows:
        print("  [Table 3] No LiRA results found. Run exp_p19_lira.py first.")
        return

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "table3_lira_correlation.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
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

def figure_1(cert_dir=CERT_DIR, lira_dir=LIRA_DIR, runs_dir=RUNS_DIR, out_dir=OUT_DIR,
             lira_seeds=None):
    """
    Figure 1: Scatter plot of ε^dir and ε^norm vs D_i^LiRA, colored by tier.
    Plots one panel per available LiRA seed; if multiple seeds, uses a subplot row per seed.
    """
    plt = _plt(); os.makedirs(out_dir, exist_ok=True)
    if plt is None: return

    seeds_to_use = lira_seeds if lira_seeds is not None else _available_lira_seeds("LF1", lira_dir)
    if not seeds_to_use:
        seeds_to_use = [0]

    tier_colors = {0: "steelblue", 1: "orange", 2: "firebrick"}
    tier_names  = {0: "head", 1: "mid", 2: "tail"}

    valid = []
    for s in seeds_to_use:
        lf1  = _load_lira("LF1", lira_dir, seed=s)
        cert = _load_cert("F1", s, cert_dir)
        run  = _load_run("F1", s, runs_dir)
        if not lf1 or "D_lira_members" not in lf1: continue
        eps_dir  = cert.get("epsilon_cert_dir_rank_100")
        eps_norm = cert.get("epsilon_cert_norm")
        if eps_dir is None or eps_norm is None: continue
        member_local = np.load(
            os.path.join(runs_dir, "F1", f"seed_{s}", "lira_member_local_idx.npy"))
        tiers = run.get("tier_labels")
        tier_m = tiers[member_local] if tiers is not None else None
        valid.append((s, lf1["D_lira_members"],
                      eps_dir[member_local], eps_norm[member_local], tier_m))

    if not valid:
        print("  [Fig 1] LiRA results not found."); return

    n_seeds = len(valid)
    fig, axes = plt.subplots(n_seeds, 2, figsize=(12, 5 * n_seeds), squeeze=False)

    for row_idx, (s, D_lira, eps_dir_m, eps_norm_m, tier_m) in enumerate(valid):
        for col_idx, (eps_v, lbl) in enumerate([(eps_dir_m, "ε^dir cert"),
                                                 (eps_norm_m, "ε^norm cert")]):
            ax = axes[row_idx][col_idx]
            if tier_m is not None:
                for t_id, t_name in tier_names.items():
                    mask = (tier_m == t_id)
                    if mask.sum() == 0: continue
                    ax.scatter(D_lira[mask], eps_v[mask], c=tier_colors[t_id],
                               s=6, alpha=0.5, label=t_name)
                ax.legend(fontsize=8)
            else:
                ax.scatter(D_lira, eps_v, s=6, alpha=0.5)
            ax.set_xlabel("D_i^LiRA"); ax.set_ylabel(lbl)
            ax.grid(True, alpha=0.3)
            prefix = "a" if col_idx == 0 else "b"
            ax.set_title(f"Fig 1{prefix} [seed={s}]: D^LiRA vs {lbl}")

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

        # NOTE: Y_proj[i,t,:r] = Y_t^T ḡ_{i,t} is a projection onto the top-r
        # covariance directions of the gradient covariance matrix. This validation
        # therefore tests Gaussianity only for the *dominant* directions — those
        # where the CLT is most likely to hold by Lyapunov. The low-variance /
        # incoherent directions (where the Gaussian approximation may break, and
        # where the masking argument for tail examples is most sensitive) are
        # truncated away. KS statistics here should be reported with this caveat.
        yp_t = Y_proj[:, t_idx, :n_projections].astype(np.float64)  # (n, K)
        K = yp_t.shape[1]

        for (i_target, tier_name) in targets:
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
# Table 6: Rank ablation
# ---------------------------------------------------------------------------

def table_6(cert_dir=CERT_DIR, out_dir=OUT_DIR):
    """
    Table 6: Certified ε^dir vs Nyström rank r across RANK_ABLATION = [10,25,50,100,200].

    Key question: does masking saturate before r=100 (rank is sufficient) or
    keep shrinking at r=200 (higher rank would tighten the bound further)?
    Reports median/p5/p95 of ε^dir cert and overhead p95 at each rank, with
    ε^norm as the baseline reference.
    """
    print(f"\n{'='*72}")
    print(f"  Table 6: Rank Ablation (ε^dir cert vs Nyström rank r)")
    print(f"{'='*72}")

    all_rows = []
    for run_id, seeds in [("F1", [0]), ("F5", [0, 1, 2])]:
        for seed in seeds:
            tag  = f"p19_{run_id}_seed{seed}"
            cert = _load_cert(run_id, seed, cert_dir)
            if "epsilon_cert_norm" not in cert:
                continue
            eps_norm = cert["epsilon_cert_norm"]

            # Load ε^dir for every rank that was computed
            rank_eps = {}
            for r in RANK_ABLATION:
                p = os.path.join(cert_dir, f"{tag}_epsilon_cert_dir_rank_{r}.npy")
                if os.path.exists(p):
                    rank_eps[r] = np.load(p)

            if not rank_eps:
                print(f"  {run_id} seed={seed}: no rank ablation files found, skipping.")
                continue

            norm_med = float(np.median(eps_norm))
            print(f"\n  {run_id} seed={seed}  (ε^norm med={norm_med:.4f})")
            hdr = (f"    {'r':>5}  {'ε^dir med':>10}  {'p5':>7}  {'p95':>8}  "
                   f"{'masking%':>9}  {'Δ vs r-1':>9}  {'overhead p95':>13}")
            print(hdr)

            prev_med = None
            for r in RANK_ABLATION:
                if r not in rank_eps:
                    continue
                ed  = rank_eps[r]
                med = float(np.median(ed))
                masking_pct = 100.0 * (1.0 - med / max(norm_med, 1e-15))
                delta = (med - prev_med) if prev_med is not None else float('nan')

                overhead_p95 = float('nan')
                b_path = os.path.join(cert_dir, f"{tag}_Bcert_dir_rank_{r}.npy")
                c_path = os.path.join(cert_dir, f"{tag}_C_realized_dir_rank_{r}.npy")
                if os.path.exists(b_path) and os.path.exists(c_path):
                    overhead_p95 = float(np.percentile(
                        (np.load(b_path) - np.load(c_path)).flatten(), 95))

                delta_str = f"{delta:+9.4f}" if np.isfinite(delta) else f"{'—':>9}"
                print(f"    {r:5d}  {med:10.4f}  "
                      f"{np.percentile(ed,5):7.4f}  {np.percentile(ed,95):8.4f}  "
                      f"{masking_pct:9.2f}%  {delta_str}  {overhead_p95:13.6f}")

                all_rows.append({
                    "run_id": run_id, "seed": seed, "rank": r,
                    "eps_dir_med":  med,
                    "eps_dir_p5":   float(np.percentile(ed, 5)),
                    "eps_dir_p95":  float(np.percentile(ed, 95)),
                    "eps_norm_med": norm_med,
                    "masking_pct":  masking_pct,
                    "delta_vs_prev_rank": float(delta),
                    "overhead_p95": overhead_p95,
                })
                prev_med = med

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "table6_rank_ablation.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n  [saved] {out_dir}/table6_rank_ablation.json")


# ===========================================================================
# Item 1: Shadow-free baseline rows  (LT-IQR and generalized leverage)
# ===========================================================================

def _spearman_ci(x, y, n_boot=500, seed=42):
    """Spearman rho plus bootstrap 95% CI.  Returns (rho, ci_lo, ci_hi)."""
    from scipy.stats import spearmanr
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    if len(xm) < 5:
        return float("nan"), float("nan"), float("nan")
    rho_val = float(spearmanr(xm, ym).statistic)
    rng = np.random.default_rng(seed)
    n = len(xm); boot_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot_vals.append(float(spearmanr(xm[idx], ym[idx]).statistic))
        except Exception:
            pass
    if not boot_vals:
        return rho_val, float("nan"), float("nan")
    return rho_val, float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5))


def _rank_r2(x, y):
    from scipy.stats import rankdata
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    return float(np.corrcoef(rankdata(x[mask]), rankdata(y[mask]))[0, 1] ** 2)


def _load_shadow_traces(lira_id, lira_seed, lira_dir=LIRA_DIR):
    """
    Load the combined per-shadow logit-score matrix and in-mask matrix for LT-IQR.

    Primary format (written by exp_p19_lira.py run_scoring):
      shadow_logit_scores.npy      (n_targets, n_shadows)  log-odds per shadow, NaN=missing
      shadow_in_mask_combined.npy  (n_targets, n_shadows)  bool, True=target was IN that shadow

    Fallback formats (older pipelines):
      logit_matrix.npy / shadows/logit_matrix.npy  + membership_matrix.npy (same dir)

    Returns (logit_matrix, membership_matrix) or (None, None) if nothing is found.
    The returned logit_matrix entries are log-odds scores (finite) or NaN (missing shadow).
    The returned membership_matrix entries are 0/1 integers.
    """
    base = os.path.join(lira_dir, lira_id, f"seed_{lira_seed}")

    # Primary: exp_p19_lira.py outputs
    logit_p = os.path.join(base, "shadow_logit_scores.npy")
    mask_p  = os.path.join(base, "shadow_in_mask_combined.npy")
    if os.path.exists(logit_p) and os.path.exists(mask_p):
        return np.load(logit_p).astype(np.float64), np.load(mask_p).astype(np.int32)

    # Fallback: older naming conventions
    logit_candidates = [
        "shadow_logits_members.npy",
        "logit_matrix.npy",
        os.path.join("shadows", "logit_matrix.npy"),
        os.path.join("shadows", "logit_matrix_np.npy"),
    ]
    mem_candidates = [
        "shadow_memberships_members.npy",
        "membership_matrix.npy",
        os.path.join("shadows", "membership_matrix.npy"),
        os.path.join("shadows", "membership_matrix_np.npy"),
    ]
    logit_mat = None
    for name in logit_candidates:
        p = os.path.join(base, name)
        if os.path.exists(p):
            logit_mat = np.load(p).astype(np.float64); break
    mem_mat = None
    for name in mem_candidates:
        p = os.path.join(base, name)
        if os.path.exists(p):
            mem_mat = np.load(p).astype(np.int32); break

    if logit_mat is None or mem_mat is None:
        # Try reconstructing from per-shadow files if the dense matrices were not saved
        per_shadow = sorted(f for f in os.listdir(base)
                            if f.startswith("shadow_") and f.endswith("_member_logits.npy"))
        if per_shadow:
            print(f"    [shadow_traces] dense matrices missing; attempting reconstruction "
                  f"from {len(per_shadow)} per-shadow files in {base}")
            # Load member labels from the companion member_labels.npy
            labels_p = os.path.join(base, "member_labels.npy")
            if not os.path.exists(labels_p):
                print(f"    [shadow_traces] member_labels.npy not found in {base}")
                return None, None
            member_labels = np.load(labels_p)
            n_m = len(member_labels)
            n_s = len(per_shadow)
            logit_mat = np.full((n_m, n_s), np.nan, dtype=np.float64)
            mem_mat   = np.zeros((n_m, n_s), dtype=np.int32)
            for col, fname in enumerate(per_shadow):
                shadow_idx = int(fname.split("_")[1])
                mask_fname = f"shadow_{shadow_idx}_in_mask.npy"
                mask_full  = os.path.join(base, mask_fname)
                logit_full = os.path.join(base, fname)
                if not os.path.exists(mask_full): continue
                logits   = np.load(logit_full).astype(np.float32)   # (n_m, C)
                in_mask  = np.load(mask_full).astype(bool)           # (n_m,)
                # Convert to log-odds score (same formula as logit_score in the LiRA script)
                probs = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs /= probs.sum(axis=1, keepdims=True)
                n = len(member_labels)
                p_correct = probs[np.arange(n), member_labels.astype(int)]
                p_correct = np.clip(p_correct, 1e-7, 1 - 1e-7)
                scores = np.log(p_correct / (1.0 - p_correct))
                logit_mat[:, col] = scores.astype(np.float64)
                mem_mat[:, col]   = in_mask.astype(np.int32)
            # Save for next time
            np.save(os.path.join(base, "shadow_logit_scores.npy"),
                    logit_mat.astype(np.float32))
            np.save(os.path.join(base, "shadow_in_mask_combined.npy"),
                    mem_mat.astype(bool))
            print(f"    [shadow_traces] saved dense matrices for future runs.")

    return logit_mat, mem_mat


def _compute_lt_iqr(logit_matrix, membership_matrix, sign_ref_D=None):
    """
    LT-IQR (Pollock et al. 2024): IQR of loss values across OUT shadow models.

    For each member target i, collect losses from all OUT-model shadows
    (membership_matrix[i, m] == 0), then compute Q75 − Q25.

    Sign convention: higher = more vulnerable.  If Spearman(LT-IQR, D^LiRA) < 0
    on sign_ref_D, flip sign.

    logit_matrix:     (n_targets, n_shadows) scalar confidence (logit) per shadow
    membership_matrix:(n_targets, n_shadows) 1=IN, 0=OUT
    Returns: lt_iqr  (n_targets,) — floats, NaN when < 2 OUT-model shadows
    """
    from scipy.stats import spearmanr
    n_targets, n_shadows = logit_matrix.shape
    lt_iqr = np.full(n_targets, np.nan, dtype=np.float64)
    for i in range(n_targets):
        out_mask = membership_matrix[i] == 0
        if out_mask.sum() < 2:
            continue
        logits_out = logit_matrix[i, out_mask]
        # Cross-entropy proxy: loss = log(1 + exp(−logit)); higher loss = less confident
        losses_out = np.log1p(np.exp(-np.clip(logits_out, -30, 30)))
        lt_iqr[i] = float(np.percentile(losses_out, 75) - np.percentile(losses_out, 25))

    # Sign check: verify Spearman(LT-IQR, D^LiRA) ≥ 0
    if sign_ref_D is not None:
        n = min(len(lt_iqr), len(sign_ref_D))
        valid = np.isfinite(lt_iqr[:n]) & np.isfinite(sign_ref_D[:n])
        if valid.sum() >= 5:
            rho_sign = float(spearmanr(lt_iqr[:n][valid], sign_ref_D[:n][valid]).statistic)
            if rho_sign < 0:
                print(f"    [lt_iqr] Sign flip: Spearman={rho_sign:.4f} < 0 → negating LT-IQR")
                lt_iqr = -lt_iqr
            else:
                print(f"    [lt_iqr] Sign OK: Spearman={rho_sign:.4f} > 0 (no flip)")
    return lt_iqr


def _nystrom_H_from_BM(B, M):
    """
    Compute S = B^{−1/2} and H = S M S (the Nystrom H matrix) used in
    both the eps_dir quadratic form and the generalized leverage computation.
    """
    B = np.asarray(B, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    evals_B, evecs_B = np.linalg.eigh(B)
    evals_B = np.maximum(evals_B, 1e-12)
    S = evecs_B @ np.diag(1.0 / np.sqrt(evals_B)) @ evecs_B.T
    H = S @ M @ S
    return S, H


def _compute_gen_leverage_batch(Y_proj_final, B_final, M_final,
                                mode="pinv", a=None, g_sqnorm=None,
                                eigenvalue_floor_factor=1e-8):
    """
    Generalized leverage (self-influence) h_i = g_i^T Sigma_hat^{+} g_i using
    the Nystrom pseudoinverse, with W = Y B^{-1/2} and H = W^T W = B^{-1/2} M B^{-1/2}.

    IMPORTANT: the correct pseudoinverse form is
        h_i = z_i^T H^{-2} z_i   (mode="pinv")
    NOT z_i^T H^{-1} z_i.  The latter equals ||P_W g_i||^2 (energy in the dominant
    subspace), which has no inverse-covariance action and correlates negatively with
    membership vulnerability.  The correct form weights energy by 1/lambda_k^2,
    emphasising the quiet low-variance directions — the same family as the eps_dir
    Mahalanobis shift but under a different matrix.

    A ridge-regularised variant (matching the DP-noise-aware eps_dir form) is also
    provided for comparison:
        h_i = g_i^T (a I + Sigma_hat)^{-1} g_i
            = ||g_i||^2/a  -  (1/a^2) z_i^T (I + H/a)^{-1} z_i   (mode="ridge")

    Parameters
    ----------
    Y_proj_final : (n, r)   s_i = Y_t^T g_i at the final accounting step.
    B_final, M_final : (r, r)   B and M = Y^T Y at the final step.
    mode : {"pinv", "ridge"}
    a    : float, required for mode="ridge" (e.g. sigma^2 C^2).
    g_sqnorm : (n,) or float, required for mode="ridge" (ambient ||g_i||^2).
    eigenvalue_floor_factor : pseudoinverse floor = factor * lambda_max.

    Returns : (n,) leverage scores >= 0.
    """
    Y_proj = np.asarray(Y_proj_final, dtype=np.float64)
    if Y_proj.ndim == 1:
        Y_proj = Y_proj.reshape(1, -1)
    n, r = Y_proj.shape

    S, H = _nystrom_H_from_BM(B_final, M_final)
    evals_H, evecs_H = np.linalg.eigh(H)           # ascending order
    lmax  = float(evals_H.max()) if evals_H.size and evals_H.max() > 0 else 1.0
    floor = eigenvalue_floor_factor * lmax
    valid = evals_H > floor

    # z_i = S s_i = W^T g_i ; coefficients in H-eigenbasis: C[i,k] = v_k^T z_i
    Z = Y_proj @ S.T          # (n, r)
    C = Z @ evecs_H            # (n, r)

    if mode == "pinv":
        # h_i = z_i^T H^{+2} z_i = sum_k C_{ik}^2 / lambda_k^2  (lambda_k > floor)
        inv2 = np.zeros(r, dtype=np.float64)
        inv2[valid] = 1.0 / (evals_H[valid] ** 2)
        leverage = (C ** 2) @ inv2

    elif mode == "ridge":
        if a is None or g_sqnorm is None:
            raise ValueError("mode='ridge' requires both `a` and `g_sqnorm`.")
        a = float(a)
        # z_i^T (I + H/a)^{-1} z_i = sum_k C_{ik}^2 / (1 + lambda_k/a)
        quad = (C ** 2) @ (1.0 / (1.0 + evals_H / a))      # (n,)
        gsq = np.asarray(g_sqnorm, dtype=np.float64)
        if gsq.ndim == 0:
            gsq = np.full(n, float(gsq))
        leverage = gsq / a - quad / (a ** 2)

    else:
        raise ValueError(f"Unknown mode {mode!r}; use 'pinv' or 'ridge'.")

    return np.maximum(leverage, 0.0)


def _compute_lt_iqr_from_losses(losses_mem, n_epochs, collapse="mean", q1=0.25, q2=0.75):
    """
    LT-IQR per Pollock et al. from a per-step training-loss trace.

    losses_mem : (n_member, T_steps)  per-step CE losses saved during training
    n_epochs   : number of training epochs (used to group steps into epoch blocks)
    collapse   : "mean" (primary) or "last" — how to reduce each epoch's steps to one value
    Returns    : lt_iqr of shape (n_member,), non-negative, no sign flip needed
    """
    losses_mem = np.asarray(losses_mem, dtype=np.float64)
    n_member, T_steps = losses_mem.shape
    steps_per_epoch = T_steps // n_epochs
    if steps_per_epoch < 1:
        return np.full(n_member, np.nan)
    T_use = steps_per_epoch * n_epochs          # truncate to complete epochs
    L_epoched = losses_mem[:, :T_use].reshape(n_member, n_epochs, steps_per_epoch)
    if collapse == "mean":
        L = L_epoched.mean(axis=2)              # (n_member, n_epochs)
    else:
        L = L_epoched[:, :, -1]                 # last step of each epoch
    return np.quantile(L, q2, axis=1) - np.quantile(L, q1, axis=1)


def _precision_at_k(scores, D_lira, ks=(0.01, 0.03, 0.05)):
    """
    Precision@k matching Pollock et al.'s headline metric.

    Vulnerable set V = top-k members by D^LiRA score.
    Predicted set P = top-k members by predictor scores, with tie-aware
    expected precision under random tie-breaking at the k-th boundary.

    When the k-th largest predictor score is tied across multiple members,
    using a deterministic argsort would give arbitrary (index-ordered) results.
    Instead we compute expected precision:
      - slots filled by strictly-above-threshold predictions count exactly
      - remaining slots are filled randomly from the tied pool; we use the
        base rate of vulnerable members in that pool as E[hits from ties]

    ks : fractions of the valid population (0.01 = top-1%)
    Returns dict with keys prec_at_1pct, k_1pct, tie_fraction_at_k_1pct, ...
    """
    scores = np.asarray(scores, dtype=np.float64)
    D = np.asarray(D_lira, dtype=np.float64)
    n = min(len(scores), len(D))
    valid = np.isfinite(scores[:n]) & np.isfinite(D[:n])
    sv, dv = scores[:n][valid], D[:n][valid]
    n_v = len(sv)
    out = {}
    for k_frac in ks:
        k = max(1, int(round(k_frac * n_v)))
        tag = f"{int(round(k_frac * 100))}pct"

        # Vulnerable set V: top-k by D^LiRA (D scores typically continuous; argsort is fine)
        V = set(np.argsort(dv)[-k:].tolist())

        # Predicted set P: expected precision accounting for ties in sv at the boundary
        thresh = np.sort(sv)[-k]  # k-th largest value
        above_thresh_idx = np.where(sv > thresh)[0]
        tied_idx         = np.where(sv == thresh)[0]

        n_above    = len(above_thresh_idx)
        n_tied     = len(tied_idx)
        remaining  = k - n_above  # slots to fill from the tied pool

        if remaining <= 0 or n_tied == 0:
            # No tie spans the boundary — exact count
            P = set(above_thresh_idx.tolist())
            if n_tied > 0:
                P |= set(tied_idx[:remaining].tolist())
            prec     = float(len(V & P) / k)
            tie_frac = 0.0
        else:
            # Expected hits: exact above-threshold portion + proportional tie portion
            exact_hits            = len(V & set(above_thresh_idx.tolist()))
            vulnerable_in_ties    = len(V & set(tied_idx.tolist()))
            frac_from_ties        = vulnerable_in_ties * remaining / n_tied
            prec                  = float((exact_hits + frac_from_ties) / k)
            tie_frac              = float(remaining / k)

        out[f"prec_at_{tag}"]            = prec
        out[f"k_{tag}"]                  = k
        out[f"tie_fraction_at_k_{tag}"]  = tie_frac
    return out


def _corr_row_p19(label, scores, D_lira, tier_arr=None, n_boot=500):
    """Return overall + per-tier correlation dicts for a predictor vs D^LiRA."""
    scores = np.asarray(scores, dtype=np.float64)
    D = np.asarray(D_lira, dtype=np.float64)
    n = min(len(scores), len(D))
    scores, D = scores[:n], D[:n]

    rho, ci_lo, ci_hi = _spearman_ci(scores, D, n_boot=n_boot)
    r2 = _rank_r2(scores, D)
    overall = {"label": label, "scope": "overall",
               "rho": rho, "ci_lo": ci_lo, "ci_hi": ci_hi,
               "rank_r2": r2, "n": int(np.isfinite(scores).sum())}

    tier_rows = []
    if tier_arr is not None:
        tier_names_map = {0: "head", 1: "mid", 2: "tail"}
        for t_id, t_name in tier_names_map.items():
            mask = (tier_arr[:n] == t_id) & np.isfinite(scores) & np.isfinite(D)
            if mask.sum() < 5:
                continue
            rho_t, ci_lo_t, ci_hi_t = _spearman_ci(scores[mask], D[mask], n_boot=n_boot)
            r2_t = _rank_r2(scores[mask], D[mask])
            tier_rows.append({"label": label, "scope": t_name,
                               "rho": rho_t, "ci_lo": ci_lo_t, "ci_hi": ci_hi_t,
                               "rank_r2": r2_t, "n": int(mask.sum())})
    return overall, tier_rows


def table_baselines(cert_dir=CERT_DIR, lira_dir=LIRA_DIR, runs_dir=RUNS_DIR,
                    out_dir=OUT_DIR, setting="F1", lira_id="LF1",
                    seeds=(0, 1, 2), n_boot=500):
    """
    Item 1: Shadow-free baseline rows for table3.

    Computes LT-IQR and generalized leverage for each seed, runs Spearman +
    bootstrap + three-seed aggregation, and reports the gap (eps_dir rho minus
    each baseline rho) per tier so prose can state the comparison honestly.

    Appends to {out_dir}/table3_baselines.json (same schema as table3_lira_
    correlation.json).
    """
    print(f"\n{'='*72}")
    print(f"  Item 1: Shadow-free baselines — {setting} × {lira_id}")
    print(f"{'='*72}")

    # Collect per-seed results for three-seed aggregate
    per_seed_lt_iqr   = {}
    per_seed_leverage = {}
    per_seed_D        = {}
    per_seed_tiers    = {}   # tier labels for member targets
    all_rows          = []

    for seed in seeds:
        print(f"\n  -- seed {seed} --")
        lf1  = _load_lira(lira_id, lira_dir, seed=seed)
        cert = _load_cert(setting, seed, cert_dir)
        run  = _load_run(setting, seed, runs_dir)

        if not lf1 or "D_lira_members" not in lf1:
            print(f"    [skip] D_lira_members not found for {lira_id}/seed_{seed}")
            continue

        D_lira = lf1["D_lira_members"].astype(np.float64)
        member_local_path = os.path.join(runs_dir, setting, f"seed_{seed}",
                                         "lira_member_local_idx.npy")
        if not os.path.exists(member_local_path):
            print(f"    [skip] lira_member_local_idx.npy not found")
            continue
        member_local = np.load(member_local_path)

        n = min(len(D_lira), len(member_local))
        D_use     = D_lira[:n]
        idx       = member_local[:n]
        tier_arr  = run.get("tier_labels")
        tier_mem  = tier_arr[idx] if tier_arr is not None else None

        per_seed_D[seed]     = D_use
        per_seed_tiers[seed] = tier_mem

        # ---------------------------------------------------------------
        # 1a: LT-IQR — per-step losses.npy trace (Pollock et al. §4.2)
        #     No shadow models used.  IQR is computed over per-epoch loss
        #     values (steps collapsed to epochs first) to match Pollock's
        #     construction.  Higher IQR = more loss volatility = more
        #     vulnerable, by construction; no sign flip needed.
        # ---------------------------------------------------------------
        lt_iqr = None
        lt_iqr_last_collapse = None
        losses_all_raw = run.get("losses")
        run_meta = run.get("meta", {})
        n_epochs_run = int(run_meta.get("n_epochs",
                           run_meta.get("T_epochs",
                           run_meta.get("num_epochs", 0))))
        if n_epochs_run <= 0 and losses_all_raw is not None:
            T_steps_raw = losses_all_raw.shape[1]
            spe = int(run_meta.get("steps_per_epoch",
                      run_meta.get("steps_per_ep", 0)))
            if spe > 0:
                n_epochs_run = T_steps_raw // spe
        if losses_all_raw is not None and n_epochs_run > 0:
            losses_mem_raw = losses_all_raw[idx, :]           # (n_member, T_steps)
            lt_iqr_mean        = _compute_lt_iqr_from_losses(
                losses_mem_raw, n_epochs_run, collapse="mean")
            lt_iqr_last_collapse = _compute_lt_iqr_from_losses(
                losses_mem_raw, n_epochs_run, collapse="last")
            sp_rho_collapse, _, _ = _spearman_ci(lt_iqr_mean, lt_iqr_last_collapse, n_boot=0)
            n_finite = int(np.isfinite(lt_iqr_mean).sum())
            n_nonneg = int((lt_iqr_mean[np.isfinite(lt_iqr_mean)] >= 0).sum())
            print(f"    LT-IQR (mean-collapse): {n_finite}/{len(lt_iqr_mean)} targets finite, "
                  f"non-neg={n_nonneg}/{n_finite}, "
                  f"Spearman(mean,last)={sp_rho_collapse:+.4f}  "
                  f"[n_epochs={n_epochs_run}, T_steps={losses_all_raw.shape[1]}]")
            if np.isfinite(sp_rho_collapse) and sp_rho_collapse < 0.95:
                print(f"    [warn] mean/last Spearman={sp_rho_collapse:.4f} < 0.95 — "
                      f"trace fragile at this operating point; both collapse modes reported.")
            lt_iqr = lt_iqr_mean
            per_seed_lt_iqr[seed] = lt_iqr
        else:
            print(f"    LT-IQR: losses.npy not found or n_epochs unknown "
                  f"(n_epochs_run={n_epochs_run}) — skipping.")

        # ---------------------------------------------------------------
        # 1b: Generalized leverage from final Nystrom step
        # ---------------------------------------------------------------
        leverage = None
        yp_path = os.path.join(runs_dir, setting, f"seed_{seed}", "Y_projections.npy")
        b_path  = os.path.join(runs_dir, setting, f"seed_{seed}", "B_matrices.npy")
        m_path  = os.path.join(runs_dir, setting, f"seed_{seed}", "YTY_matrices.npy")
        if os.path.exists(yp_path) and os.path.exists(b_path) and os.path.exists(m_path):
            Y_proj = np.load(yp_path)    # (n_priv, T, r)
            B_mats = np.load(b_path)     # (T, r, r)
            M_mats = np.load(m_path)     # (T, r, r)
            # Final accounting step for each member target
            Y_final = Y_proj[idx, -1, :].astype(np.float64)
            leverage = _compute_gen_leverage_batch(Y_final,
                                                   B_mats[-1].astype(np.float64),
                                                   M_mats[-1].astype(np.float64))
            per_seed_leverage[seed] = leverage
            print(f"    Gen-leverage: {len(leverage)} targets (mean={leverage.mean():.4e})")
        else:
            print(f"    Gen-leverage: Y_projections/B_matrices/YTY_matrices not found — skipping.")

        # ---------------------------------------------------------------
        # 1c: Correlation rows for this seed
        # ---------------------------------------------------------------
        if lt_iqr is not None:
            n_use = min(n, len(lt_iqr))
            # Primary: mean-collapse LT-IQR (Pollock §4.2 construction)
            row_ov, rows_t = _corr_row_p19(f"[s{seed}] LT-IQR (mean)", lt_iqr, D_use[:n_use],
                                            tier_arr=tier_mem[:n_use] if tier_mem is not None else None,
                                            n_boot=n_boot)
            prec_mean = _precision_at_k(lt_iqr[:n_use], D_use[:n_use])
            for r in [row_ov] + rows_t:
                r["seed"] = seed; r["predictor"] = "lt_iqr"; r["collapse"] = "mean"
                r["precision_at_k"] = prec_mean
                all_rows.append(r)
            print(f"    LT-IQR (mean) overall: ρ={row_ov['rho']:+.4f}  "
                  f"CI=[{row_ov['ci_lo']:.3f},{row_ov['ci_hi']:.3f}]  "
                  f"Prec@1%={prec_mean.get('prec_at_1pct', float('nan')):.3f}")

            # Sensitivity check: last-collapse
            if lt_iqr_last_collapse is not None:
                n_use_l = min(n, len(lt_iqr_last_collapse))
                row_ov_l, rows_t_l = _corr_row_p19(
                    f"[s{seed}] LT-IQR (last)", lt_iqr_last_collapse, D_use[:n_use_l],
                    tier_arr=tier_mem[:n_use_l] if tier_mem is not None else None,
                    n_boot=n_boot)
                prec_last = _precision_at_k(lt_iqr_last_collapse[:n_use_l], D_use[:n_use_l])
                for r in [row_ov_l] + rows_t_l:
                    r["seed"] = seed; r["predictor"] = "lt_iqr"; r["collapse"] = "last"
                    r["precision_at_k"] = prec_last
                    all_rows.append(r)
                print(f"    LT-IQR (last) overall: ρ={row_ov_l['rho']:+.4f}  "
                      f"CI=[{row_ov_l['ci_lo']:.3f},{row_ov_l['ci_hi']:.3f}]  "
                      f"Prec@1%={prec_last.get('prec_at_1pct', float('nan')):.3f}")

            # Reference predictors: Precision@k on the same axis for comparison
            for pred_name, pred_arr in [
                ("eps_dir",  cert.get("epsilon_cert_dir_rank_100")),
                ("eps_norm", cert.get("epsilon_cert_norm")),
            ]:
                if pred_arr is None:
                    continue
                pred_mem = pred_arr[idx[:n_use]]
                prec_ref = _precision_at_k(pred_mem, D_use[:n_use])
                ref_row = {"label": f"[s{seed}] {pred_name} Prec@k",
                           "seed": seed, "predictor": pred_name, "scope": "overall",
                           "precision_at_k": prec_ref,
                           "rho": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
                all_rows.append(ref_row)
                print(f"    {pred_name:10s}  "
                      f"Prec@1%={prec_ref.get('prec_at_1pct', float('nan')):.3f}  "
                      f"Prec@3%={prec_ref.get('prec_at_3pct', float('nan')):.3f}  "
                      f"Prec@5%={prec_ref.get('prec_at_5pct', float('nan')):.3f}")
            if losses_all_raw is not None:
                loss_final_mem = losses_all_raw[idx[:n_use], -1]
                prec_loss = _precision_at_k(loss_final_mem, D_use[:n_use])
                loss_row = {"label": f"[s{seed}] loss_final Prec@k",
                            "seed": seed, "predictor": "loss_final", "scope": "overall",
                            "precision_at_k": prec_loss,
                            "rho": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
                all_rows.append(loss_row)
                print(f"    loss_final   "
                      f"Prec@1%={prec_loss.get('prec_at_1pct', float('nan')):.3f}  "
                      f"Prec@3%={prec_loss.get('prec_at_3pct', float('nan')):.3f}  "
                      f"Prec@5%={prec_loss.get('prec_at_5pct', float('nan')):.3f}")

        if leverage is not None:
            n_use = min(n, len(leverage))
            row_ov, rows_t = _corr_row_p19(f"[s{seed}] gen-leverage", leverage, D_use[:n_use],
                                            tier_arr=tier_mem[:n_use] if tier_mem is not None else None,
                                            n_boot=n_boot)
            for r in [row_ov] + rows_t:
                r["seed"] = seed; r["predictor"] = "gen_leverage"
                all_rows.append(r)
            print(f"    Gen-leverage overall: ρ={row_ov['rho']:+.4f}  "
                  f"CI=[{row_ov['ci_lo']:.3f},{row_ov['ci_hi']:.3f}]")

    # -------------------------------------------------------------------
    # Three-seed aggregate rows
    # -------------------------------------------------------------------
    print(f"\n  --- Aggregate rows (seeds={list(seeds)}) ---")
    tier_names_map = {0: "head", 1: "mid", 2: "tail"}
    scopes = ["overall"] + list(tier_names_map.values())

    for predictor, seed_scores_dict in [("lt_iqr", per_seed_lt_iqr),
                                         ("gen_leverage", per_seed_leverage)]:
        if not seed_scores_dict:
            continue
        pred_label = {"lt_iqr": "LT-IQR", "gen_leverage": "gen-leverage"}[predictor]
        for scope in scopes:
            rho_per_seed = []
            for seed, scores in seed_scores_dict.items():
                D_s    = per_seed_D.get(seed, np.array([]))
                tier_s = per_seed_tiers.get(seed)
                n_s    = min(len(scores), len(D_s))
                sc_s, D_s = scores[:n_s], D_s[:n_s]
                if scope == "overall":
                    valid = np.isfinite(sc_s) & np.isfinite(D_s)
                else:
                    t_id = {v: k for k, v in tier_names_map.items()}.get(scope)
                    if tier_s is None or t_id is None: continue
                    valid = (tier_s[:n_s] == t_id) & np.isfinite(sc_s) & np.isfinite(D_s)
                if valid.sum() < 5: continue
                rho, _, _ = _spearman_ci(sc_s[valid], D_s[valid], n_boot=0)
                rho_per_seed.append(float(rho))
            if not rho_per_seed: continue
            agg = {"label": f"AGGREGATE {pred_label}",
                   "predictor": predictor, "scope": scope,
                   "rho_per_seed": rho_per_seed,
                   "rho_mean": float(np.mean(rho_per_seed)),
                   "rho_std":  float(np.std(rho_per_seed)),
                   "n_seeds":  len(rho_per_seed)}
            all_rows.append(agg)
            print(f"  {pred_label:12s} {scope:8s}: "
                  f"ρ_mean={agg['rho_mean']:+.4f} ±{agg['rho_std']:.4f}  "
                  f"per_seed={[f'{v:+.4f}' for v in rho_per_seed]}")

    # -------------------------------------------------------------------
    # 1d: Gap report: eps_dir rho minus each baseline rho, per tier
    # -------------------------------------------------------------------
    print(f"\n  --- Gap: eps_dir ρ minus baseline ρ (three-seed range) ---")
    for pred, seed_dict in [("LT-IQR", per_seed_lt_iqr),
                              ("gen-leverage", per_seed_leverage)]:
        if not seed_dict: continue
        for scope in scopes:
            gaps = []
            for seed, base_scores in seed_dict.items():
                cert = _load_cert(setting, seed, cert_dir)
                if cert is None or "epsilon_cert_dir_rank_100" not in cert: continue
                ml = os.path.join(runs_dir, setting, f"seed_{seed}", "lira_member_local_idx.npy")
                if not os.path.exists(ml): continue
                midx = np.load(ml)
                D_s = per_seed_D.get(seed, np.array([]))
                tier_s = per_seed_tiers.get(seed)
                n_s = min(len(base_scores), len(midx), len(D_s))
                ed  = cert["epsilon_cert_dir_rank_100"][midx[:n_s]]
                if scope == "overall":
                    valid = np.isfinite(ed) & np.isfinite(base_scores[:n_s]) & np.isfinite(D_s[:n_s])
                else:
                    t_id = {v: k for k, v in tier_names_map.items()}.get(scope)
                    if tier_s is None or t_id is None: continue
                    valid = ((tier_s[:n_s] == t_id) & np.isfinite(ed)
                             & np.isfinite(base_scores[:n_s]) & np.isfinite(D_s[:n_s]))
                if valid.sum() < 5: continue
                rho_dir,  _, _ = _spearman_ci(ed[valid], D_s[:n_s][valid], n_boot=0)
                rho_base, _, _ = _spearman_ci(base_scores[:n_s][valid], D_s[:n_s][valid], n_boot=0)
                gaps.append(rho_dir - rho_base)
            if not gaps: continue
            gap_str = f"[{min(gaps):+.4f}, {max(gaps):+.4f}]"
            print(f"  eps_dir vs {pred:12s} {scope:8s}: gap range={gap_str}  mean={np.mean(gaps):+.4f}")

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "table3_baselines.json")
    with open(out_path, "w") as f:
        json.dump({"setting": setting, "lira_id": lira_id, "seeds": list(seeds),
                   "rows": all_rows,
                   "schema": ("Each row: label, predictor, scope, rho, ci_lo, ci_hi, "
                               "rank_r2, n, seed.  LT-IQR rows also carry: collapse "
                               "(mean|last), precision_at_k {prec_at_1pct, k_1pct, "
                               "prec_at_3pct, k_3pct, prec_at_5pct, k_5pct}.  "
                               "Reference-predictor Prec@k rows (eps_dir, eps_norm, "
                               "loss_final) carry precision_at_k but rho=NaN.  "
                               "Aggregate rows: rho_per_seed, rho_mean, rho_std instead.")},
                  f, indent=2)
    print(f"\n  [saved] {out_path}")
    return all_rows


# ===========================================================================
# Item 2: Low-variance-direction Gaussianity check
# ===========================================================================

def _gauss_stats_1d(samples):
    """KS distance to best-fit Gaussian, |skewness|, |excess kurtosis|."""
    from scipy.stats import ks_2samp, skew as _skew, kurtosis as _kurt, norm
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if len(samples) < 20:
        return {"ks": float("nan"), "skewness_abs": float("nan"),
                "excess_kurtosis_abs": float("nan"), "n": len(samples)}
    mu, sd = samples.mean(), samples.std()
    if sd < 1e-12:
        return {"ks": float("nan"), "skewness_abs": float("nan"),
                "excess_kurtosis_abs": float("nan"), "n": len(samples)}
    ref = norm.rvs(loc=mu, scale=sd, size=min(20000, len(samples) * 10), random_state=0)
    ks_stat, _ = ks_2samp(samples, ref)
    return {"ks": float(ks_stat),
            "skewness_abs": float(abs(_skew(samples))),
            "excess_kurtosis_abs": float(abs(_kurt(samples, fisher=True))),
            "n": len(samples)}


def _poisson_proj(scalars_j, q, n_batches, exclude_idx, rng):
    """
    Draw n_batches realisations of <v, G_{-i}> = Σ_{j≠i} I_j (v^T g_j).
    scalars_j[j] = v^T g_j  for all j (including i).
    exclude_idx: index of the leave-one-out target (integer).
    Returns: array of shape (n_batches,).
    """
    n = len(scalars_j)
    mask = np.ones(n, dtype=bool); mask[exclude_idx] = False
    s = scalars_j[mask]           # (n-1,)
    chunk = 512
    out = np.empty(n_batches, dtype=np.float64)
    for start in range(0, n_batches, chunk):
        end = min(start + chunk, n_batches)
        nb  = end - start
        I   = rng.binomial(1, q, size=(len(s), nb)).astype(np.float64)
        out[start:end] = s @ I
    return out


def gaussian_validation_lowvar(run_id, seed, runs_dir=RUNS_DIR, out_dir=OUT_DIR,
                                n_batches=5000, k_top=10, k_bot=5,
                                eigenvalue_floor_factor=1e-6,
                                n_targets_per_tier=3):
    """
    Item 2: Low-variance-direction Gaussianity check.

    Re-runs the existing batch-sampling loop (same n_batches, target set,
    early/middle/late steps) but projects onto two complementary families:

    Family (i):  bottom-k eigenvectors of H = B^{-1/2} M B^{-1/2} that have
                 eigenvalue above eigenvalue_floor_factor * lambda_max.
                 These are the lowest-variance directions within the Nystrom
                 subspace.

    Family (ii): for each tail (and mid/head) target, the component of its
                 clipped gradient direction orthogonal to the top-k subspace.
                 This is the direction-specific test for how exposed the target
                 is in non-dominant space.

    The existing gaussian_validation() tests only top-k directions.  This
    function tests the opposite end, where Gaussianity is least guaranteed
    and tail examples are most exposed.

    Output: {out_dir}/gaussian_validation_{run_id}_lowvar_seed{seed}.json
    Same schema as the existing gaussian_validation output plus a "subspace"
    field in {"top","bottom","tail_orthogonal"}.
    """
    print(f"\n{'='*72}")
    print(f"  Item 2: Low-var Gaussianity — {run_id} seed={seed}")
    print(f"{'='*72}")

    run_dir   = os.path.join(runs_dir, run_id, f"seed_{seed}")
    meta_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"  [GV-lowvar] metadata.json not found: {run_dir}"); return

    with open(meta_path) as f:
        meta = json.load(f)
    q = float(meta.get("q", meta.get("rho_poisson", 1/9)))
    T = int(meta.get("T_train", meta.get("T_steps", 100)))

    for fname in ["Y_projections.npy", "B_matrices.npy", "YTY_matrices.npy"]:
        if not os.path.exists(os.path.join(run_dir, fname)):
            print(f"  [GV-lowvar] {fname} not found — skipping (need Nystrom matrices).")
            return

    Y_proj = np.load(os.path.join(run_dir, "Y_projections.npy"))   # (n, T, r)
    B_mats = np.load(os.path.join(run_dir, "B_matrices.npy"))       # (T, r, r)
    M_mats = np.load(os.path.join(run_dir, "YTY_matrices.npy"))     # (T, r, r)
    tier_path = os.path.join(run_dir, "tier_labels.npy")
    tier_labels = np.load(tier_path) if os.path.exists(tier_path) else None

    n, T_saved, r = Y_proj.shape
    T = min(T, T_saved)

    steps_map = {
        "early":  max(0, int(0.1 * T) - 1),
        "middle": max(0, int(0.5 * T) - 1),
        "late":   max(0, T - 1),
    }
    if run_id == "F5":
        steps_map = {"middle": steps_map["middle"]}

    # Select representative targets from each tier
    rng_tgt = np.random.default_rng(seed * 7 + 13)
    targets = []   # list of (idx, tier_name)
    tier_map = {0: "head", 1: "mid", 2: "tail"}
    if tier_labels is not None:
        for t_id, t_name in tier_map.items():
            avail = np.where(tier_labels == t_id)[0]
            if len(avail) == 0: continue
            chosen = rng_tgt.choice(avail, size=min(n_targets_per_tier, len(avail)), replace=False)
            for idx in chosen:
                targets.append((int(idx), t_name))
    else:
        for idx in rng_tgt.choice(n, size=min(9, n), replace=False):
            targets.append((int(idx), "unknown"))

    results = []
    rng_sam = np.random.default_rng(seed * 31 + 17)

    for phase_name, step_idx in steps_map.items():
        step_idx = int(np.clip(step_idx, 0, T - 1))
        print(f"\n  Phase {phase_name} (step {step_idx}/{T}) ...")

        B  = B_mats[step_idx].astype(np.float64)
        M  = M_mats[step_idx].astype(np.float64)
        S, H = _nystrom_H_from_BM(B, M)

        # Eigendecompose H (ascending order from eigh → reverse for descending)
        evals_H, evecs_H = np.linalg.eigh(H)
        evals_desc  = evals_H[::-1]
        evecs_desc  = evecs_H[:, ::-1]   # (r, r) columns = eigvecs in descending order
        lmax        = evals_desc[0] if evals_desc[0] > 0 else 1.0
        floor       = eigenvalue_floor_factor * lmax

        # Valid directions: eigenvalue > floor
        valid_mask  = evals_desc > floor
        n_valid     = int(valid_mask.sum())

        # Bottom-k from valid directions
        valid_idxs  = np.where(valid_mask)[0]   # positions in desc-ordered evals
        n_bot_use   = min(k_bot, len(valid_idxs))
        bot_idxs    = valid_idxs[-n_bot_use:]   # last k of the valid = smallest valid

        # Precompute z_j = S @ Y_proj[j, step, :] for all j
        Y_step = Y_proj[:, step_idx, :].astype(np.float64)   # (n, r)
        Z = Y_step @ S.T                                       # (n, r)

        print(f"    H eigenvalues: max={evals_desc[0]:.4e}, "
              f"floor={floor:.2e}, n_valid={n_valid}/{r}, n_bot={n_bot_use}")

        # -----------------------------------------------------------------
        # Family (i): bottom-k valid eigenvectors of H
        # -----------------------------------------------------------------
        for ki, dir_idx in enumerate(bot_idxs):
            w      = evecs_desc[:, dir_idx]       # (r,) direction in z-space
            scalars = Z @ w                        # (n,) scalar projections
            ev     = float(evals_desc[dir_idx])

            for (i_target, tier_name) in targets:
                samples = _poisson_proj(scalars, q, n_batches,
                                        exclude_idx=i_target, rng=rng_sam)
                stats = _gauss_stats_1d(samples)
                stats.update({
                    "run_id": run_id, "seed": seed,
                    "phase": phase_name, "step": step_idx,
                    "subspace": "bottom",
                    "direction_idx": ki,
                    "eigenvalue": ev,
                    "tier": tier_name,
                    "target_idx": i_target,
                })
                results.append(stats)

            ks_vals = [r2["ks"] for r2 in results
                       if r2["subspace"] == "bottom" and r2["phase"] == phase_name
                       and r2["direction_idx"] == ki and np.isfinite(r2["ks"])]
            if ks_vals:
                print(f"    [bottom dir {ki} λ={ev:.3e}] KS med={np.median(ks_vals):.4f}  "
                      f"max={np.max(ks_vals):.4f}")

        # -----------------------------------------------------------------
        # Family (ii): tail-orthogonal directions
        # Defined as z_i projected out of the top-k subspace, then normalized.
        # -----------------------------------------------------------------
        top_evecs = evecs_desc[:, :k_top]   # (r, k_top)

        for (i_target, tier_name) in targets:
            z_i     = Z[i_target]                          # (r,)
            proj    = top_evecs @ (top_evecs.T @ z_i)
            z_orth  = z_i - proj
            norm_orth = float(np.linalg.norm(z_orth))

            if norm_orth < 1e-10:
                print(f"    [ii] {tier_name} i={i_target}: gradient fully in top-{k_top} "
                      f"subspace; skipping orthogonal direction.")
                continue

            w_ii    = z_orth / norm_orth                   # (r,) normalised direction
            scalars = Z @ w_ii                              # (n,)

            samples = _poisson_proj(scalars, q, n_batches,
                                    exclude_idx=i_target, rng=rng_sam)
            stats = _gauss_stats_1d(samples)
            stats.update({
                "run_id": run_id, "seed": seed,
                "phase": phase_name, "step": step_idx,
                "subspace": "tail_orthogonal",
                "tier": tier_name,
                "target_idx": i_target,
                "z_orth_norm": norm_orth,
                "frac_orth_of_total": norm_orth / (np.linalg.norm(z_i) + 1e-12),
            })
            results.append(stats)

        # Summary for this phase
        for sub, sub_label in [("bottom", "bottom-k"), ("tail_orthogonal", "tail_orth")]:
            sub_r = [s for s in results if s["phase"] == phase_name and s["subspace"] == sub]
            if not sub_r: continue
            ks_v  = [s["ks"] for s in sub_r if np.isfinite(s["ks"])]
            sk_v  = [s["skewness_abs"] for s in sub_r if np.isfinite(s.get("skewness_abs", np.nan))]
            ek_v  = [s["excess_kurtosis_abs"] for s in sub_r if np.isfinite(s.get("excess_kurtosis_abs", np.nan))]
            if ks_v:
                print(f"    [{sub_label}] KS: med={np.median(ks_v):.4f}  "
                      f"max={np.max(ks_v):.4f}  |skew| med={np.median(sk_v) if sk_v else float('nan'):.4f}  "
                      f"|ekurt| med={np.median(ek_v) if ek_v else float('nan'):.4f}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
                            f"gaussian_validation_{run_id}_lowvar_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [GV-lowvar] Saved → {out_path}")
    return results


# ===========================================================================
# Item 3: Cross-seed rank stability of per-example eps_dir
# ===========================================================================

def rank_stability(cert_dir=CERT_DIR, runs_dir=RUNS_DIR, out_dir=OUT_DIR,
                   setting="F1", seeds=(0, 1, 2)):
    """
    Item 3: Cross-seed rank stability of per-example eps_dir rankings.

    For the common set of member examples, computes pairwise Spearman
    between per-example eps_dir rankings across seeds (0vs1, 0vs2, 1vs2),
    overall and within each tier.  Reports the median pairwise value and
    the three individual values.

    If eps_dir values were saved at the per-example level this is pure
    post-processing; if only tier summaries were saved the function warns
    and exits early.

    Output: {out_dir}/rank_stability_{setting}.json
    """
    print(f"\n{'='*72}")
    print(f"  Item 3: Cross-seed rank stability — {setting}")
    print(f"{'='*72}")

    eps_by_seed  = {}
    tier_by_seed = {}

    for seed in seeds:
        cert = _load_cert(setting, seed, cert_dir)
        if cert is None or "epsilon_cert_dir_rank_100" not in cert:
            print(f"  [skip] eps_dir not found for seed {seed}")
            continue
        eps_by_seed[seed]  = cert["epsilon_cert_dir_rank_100"].astype(np.float64)
        run = _load_run(setting, seed, runs_dir)
        if "tier_labels" in run:
            tier_by_seed[seed] = run["tier_labels"]
        print(f"  Loaded seed {seed}: n={len(eps_by_seed[seed])}  "
              f"med={np.median(eps_by_seed[seed]):.4f}")

    avail_seeds = sorted(eps_by_seed.keys())
    if len(avail_seeds) < 2:
        print(f"  [item3] Need ≥ 2 seeds with saved eps_dir, found {len(avail_seeds)}. Stopping.")
        return

    n_common  = min(len(eps_by_seed[s]) for s in avail_seeds)
    tier_ref  = tier_by_seed.get(avail_seeds[0])
    if tier_ref is not None:
        tier_ref = tier_ref[:n_common]

    tier_map = {0: "head", 1: "mid", 2: "tail"}
    scopes   = ["overall"] + list(tier_map.values())
    pairs    = [(s1, s2) for i, s1 in enumerate(avail_seeds)
                for s2 in avail_seeds[i+1:]]

    rows = []
    for scope in scopes:
        if scope == "overall":
            scope_mask = np.ones(n_common, dtype=bool)
        else:
            if tier_ref is None: continue
            t_id = {v: k for k, v in tier_map.items()}[scope]
            scope_mask = (tier_ref == t_id)
        if scope_mask.sum() < 5: continue

        pair_rhos = []
        for s1, s2 in pairs:
            e1 = eps_by_seed[s1][:n_common][scope_mask]
            e2 = eps_by_seed[s2][:n_common][scope_mask]
            valid = np.isfinite(e1) & np.isfinite(e2)
            if valid.sum() < 5: continue
            rho, _, _ = _spearman_ci(e1[valid], e2[valid], n_boot=0)
            rows.append({"scope": scope, "pair": f"{s1}vs{s2}",
                         "spearman": float(rho), "n": int(valid.sum())})
            pair_rhos.append(float(rho))
            print(f"  {scope:8s} ({s1},{s2}): Spearman ε^dir = {rho:+.4f}  n={valid.sum()}")

        if pair_rhos:
            med_rho = float(np.median(pair_rhos))
            rows.append({"scope": scope, "pair": "median_pairwise",
                         "spearman": med_rho, "n": int(scope_mask.sum())})
            print(f"  {scope:8s} median pairwise: {med_rho:+.4f}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rank_stability_{setting}.json")
    with open(out_path, "w") as f:
        json.dump({"setting": setting, "seeds_available": avail_seeds,
                   "n_common": n_common, "rows": rows,
                   "schema": ("scope ∈ {overall,head,mid,tail}, "
                               "pair ∈ {0vs1,0vs2,1vs2,median_pairwise}, spearman, n")},
                  f, indent=2)
    print(f"\n  [saved] {out_path}")
    return rows


# ===========================================================================
# New paper figures (Section 4 rewrite)
# ===========================================================================

def figure_degeneracy_s2(cert_dir=CERT_DIR, out_dir=OUT_DIR):
    """Figure A (fig:degeneracy): S2=F2 seed 0. Strip rows for ε^norm vs ε^dir."""
    plt = _plt()
    os.makedirs(out_dir, exist_ok=True)
    if plt is None:
        return

    cert = _load_cert("F2", 0, cert_dir)
    if "epsilon_cert_norm" not in cert or "epsilon_cert_dir_rank_100" not in cert:
        print("  [Fig A] F2 seed 0 cert arrays missing — skipping.")
        return

    en = cert["epsilon_cert_norm"]
    ed = cert["epsilon_cert_dir_rank_100"]

    med_norm = float(np.median(en))
    cv_dir   = float(ed.std() / max(ed.mean(), 1e-15))
    p5_dir   = float(np.percentile(ed, 5))
    p95_dir  = float(np.percentile(ed, 95))
    med_mask = float(np.median(1.0 - ed / np.maximum(en, 1e-15)))

    print(f"\n  [Fig A] F2 seed 0:")
    print(f"    median(ε^norm)            = {med_norm:.4f}")
    print(f"    CV(ε^dir)                 = {cv_dir:.4f}")
    print(f"    p5(ε^dir)                 = {p5_dir:.4f}")
    print(f"    p95(ε^dir)                = {p95_dir:.4f}")
    print(f"    median(1 − ε^dir/ε^norm)  = {med_mask:.4f}")

    rng = np.random.default_rng(42)
    jitter = 0.04

    fig, ax = plt.subplots(figsize=(8, 3.5))
    y_norm = np.ones(len(en))  + rng.uniform(-jitter, jitter, len(en))
    y_dir  = np.zeros(len(ed)) + rng.uniform(-jitter, jitter, len(ed))
    ax.scatter(en, y_norm, s=3, alpha=0.25, color="steelblue", rasterized=True, label="ε^norm")
    ax.scatter(ed, y_dir,  s=3, alpha=0.25, color="firebrick",  rasterized=True, label="ε^dir")
    ax.axvline(med_norm, color="steelblue", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.annotate(f"median ε^norm = {med_norm:.2f}",
                xy=(med_norm, 1.28), ha="center", fontsize=8, color="steelblue")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["ε^dir", "ε^norm"])
    ax.set_ylim(-0.35, 1.55)
    ax.set_xlabel("Certified ε value")
    ax.set_title("Figure A (fig:degeneracy): S2 — CLIP linear head, balanced CIFAR-10")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "figure_s2_degeneracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Fig A] saved → {path}")


def figure_withintier_s1(cert_dir=CERT_DIR, lira_dir=LIRA_DIR, runs_dir=RUNS_DIR,
                          out_dir=OUT_DIR, paper_seed=0):
    """Figure B (fig:withintier): S1=F1×LF1, per-tier scatter ε^dir vs D^LiRA."""
    plt = _plt()
    os.makedirs(out_dir, exist_ok=True)
    if plt is None:
        return

    s = paper_seed
    lf1  = _load_lira("LF1", lira_dir, seed=s)
    cert = _load_cert("F1", s, cert_dir)
    run  = _load_run("F1", s, runs_dir)

    if not lf1 or "D_lira_members" not in lf1:
        print(f"  [Fig B] LiRA seed={s} not found — skipping.")
        return
    if "epsilon_cert_dir_rank_100" not in cert or "epsilon_cert_norm" not in cert:
        print(f"  [Fig B] F1 seed={s} cert arrays missing — skipping.")
        return

    ml_path = os.path.join(runs_dir, "F1", f"seed_{s}", "lira_member_local_idx.npy")
    if not os.path.exists(ml_path):
        print(f"  [Fig B] lira_member_local_idx.npy missing for seed={s} — skipping.")
        return

    member_local = np.load(ml_path)
    tier_all = run.get("tier_labels")
    if tier_all is None:
        print(f"  [Fig B] tier_labels.npy missing — skipping.")
        return

    D_lira = lf1["D_lira_members"]
    n_m    = min(len(D_lira), len(member_local))
    D_m    = D_lira[:n_m]
    idx    = member_local[:n_m]
    tier_m = tier_all[idx]
    eps_dir_m  = cert["epsilon_cert_dir_rank_100"][idx]
    eps_norm_m = cert["epsilon_cert_norm"][idx]

    tier_names  = {0: "head", 1: "mid", 2: "tail"}
    tier_colors = {0: "steelblue", 1: "orange", 2: "firebrick"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for col, (t_id, t_name) in enumerate(tier_names.items()):
        ax = axes[col]
        mask = (tier_m == t_id)
        if mask.sum() == 0:
            ax.set_title(f"{t_name} (n=0)"); continue
        ax.scatter(D_m[mask], eps_dir_m[mask], s=8, alpha=0.45,
                   color=tier_colors[t_id], rasterized=True)
        en_tier = eps_norm_m[mask]
        if (en_tier.max() - en_tier.min()) < 0.01:
            ax.axhline(float(en_tier.mean()), color="gray", linestyle="--",
                       linewidth=1.2, alpha=0.8, label=f"ε^norm={en_tier.mean():.2f}")
        else:
            p5, p95 = float(np.percentile(en_tier, 5)), float(np.percentile(en_tier, 95))
            ax.axhspan(p5, p95, color="gray", alpha=0.15, label=f"ε^norm p5–p95")
        ax.set_title(f"{t_name}  (n={mask.sum()})")
        ax.set_xlabel("D$_i^{\\mathrm{LiRA}}$")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("ε^dir cert (r=100)")
    fig.suptitle("Figure B (fig:withintier): S1 — Within-tier ε^dir vs LiRA score", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "figure_withintier_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Fig B] saved → {path}")


def figure_wrn_distribution_s3(cert_dir=CERT_DIR, runs_dir=RUNS_DIR, out_dir=OUT_DIR,
                                 also_f1=False):
    """Figure C (fig:wrn): S3=F5, pooled seeds 0-2, per-tier ε^dir violin."""
    plt = _plt()
    os.makedirs(out_dir, exist_ok=True)
    if plt is None:
        return

    tier_names  = {0: "head", 1: "mid", 2: "tail"}
    tier_colors = {0: "steelblue", 1: "orange", 2: "firebrick"}
    norm_ref    = 9.50

    ed_by_tier = {0: [], 1: [], 2: []}
    en_by_tier = {0: [], 1: [], 2: []}
    found_any = False
    for seed in [0, 1, 2]:
        cert = _load_cert("F5", seed, cert_dir)
        run  = _load_run("F5", seed, runs_dir)
        if "epsilon_cert_norm" not in cert or "epsilon_cert_dir_rank_100" not in cert:
            print(f"  [Fig C] F5 seed={seed} cert missing — skipping seed."); continue
        if "tier_labels" not in run:
            print(f"  [Fig C] F5 seed={seed} tier_labels missing — skipping seed."); continue
        found_any = True
        for t_id in [0, 1, 2]:
            mask = (run["tier_labels"] == t_id)
            ed_by_tier[t_id].append(cert["epsilon_cert_dir_rank_100"][mask])
            en_by_tier[t_id].append(cert["epsilon_cert_norm"][mask])

    if not found_any:
        print("  [Fig C] No F5 data found — skipping."); return

    ed_pooled = {t: np.concatenate(v) for t, v in ed_by_tier.items() if v}
    en_pooled = {t: np.concatenate(v) for t, v in en_by_tier.items() if v}

    print(f"\n  [Fig C] F5 per-tier summary (pooled seeds 0-2):")
    for t_id, t_name in tier_names.items():
        if t_id not in ed_pooled or len(ed_pooled[t_id]) == 0: continue
        mean_dir  = float(ed_pooled[t_id].mean())
        mean_norm = float(en_pooled.get(t_id, np.array([norm_ref])).mean())
        masking   = 100.0 * (1.0 - mean_dir / max(mean_norm, 1e-15))
        print(f"    {t_name:6s}: mean(ε^dir)={mean_dir:.4f}  "
              f"mean(ε^norm)={mean_norm:.4f}  masking={masking:.1f}%")

    n_panels = 2 if also_f1 else 1
    fig, axes_all = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4.5), squeeze=False)
    ax_wrn = axes_all[0, 0]

    data_wrn = [ed_pooled.get(t, np.array([0.0])) for t in [0, 1, 2]]
    parts = ax_wrn.violinplot(data_wrn, positions=[0, 1, 2],
                               showmedians=True, showextrema=False)
    for pc, t_id in zip(parts["bodies"], [0, 1, 2]):
        pc.set_facecolor(tier_colors[t_id]); pc.set_alpha(0.6)
    ax_wrn.axhline(norm_ref, color="gray", linestyle="--", linewidth=1.2,
                   alpha=0.8, label=f"ε^norm = {norm_ref:.2f}")
    ax_wrn.set_xticks([0, 1, 2])
    ax_wrn.set_xticklabels([tier_names[t] for t in [0, 1, 2]])
    ax_wrn.set_xlabel("Tier"); ax_wrn.set_ylabel("ε^dir cert (r=100)")
    ax_wrn.set_title("S3 = WideResNet-28-2, CIFAR-10-LT(50)")
    ax_wrn.legend(fontsize=9); ax_wrn.grid(True, axis="y", alpha=0.3)

    if also_f1:
        ax_f1 = axes_all[0, 1]
        ed_f1 = {0: [], 1: [], 2: []}
        for seed in [0, 1, 2]:
            cert = _load_cert("F1", seed, cert_dir)
            run  = _load_run("F1", seed, runs_dir)
            if "epsilon_cert_dir_rank_100" not in cert or "tier_labels" not in run: continue
            for t_id in [0, 1, 2]:
                mask = (run["tier_labels"] == t_id)
                ed_f1[t_id].append(cert["epsilon_cert_dir_rank_100"][mask])
        ed_f1p = {t: np.concatenate(v) for t, v in ed_f1.items() if v}
        data_f1 = [ed_f1p.get(t, np.array([0.0])) for t in [0, 1, 2]]
        parts2 = ax_f1.violinplot(data_f1, positions=[0, 1, 2],
                                   showmedians=True, showextrema=False)
        for pc, t_id in zip(parts2["bodies"], [0, 1, 2]):
            pc.set_facecolor(tier_colors[t_id]); pc.set_alpha(0.6)
        ax_f1.axhline(norm_ref, color="gray", linestyle="--", linewidth=1.2,
                      alpha=0.8, label=f"ε^norm ≈ {norm_ref:.2f}")
        ax_f1.set_xticks([0, 1, 2])
        ax_f1.set_xticklabels([tier_names[t] for t in [0, 1, 2]])
        ax_f1.set_xlabel("Tier"); ax_f1.set_title("S1 = CLIP linear head, CIFAR-10-LT(50)")
        ax_f1.legend(fontsize=9); ax_f1.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Figure C (fig:wrn): S3 = WideResNet-28-2, CIFAR-10-LT(50)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "figure_s3_dir_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Fig C] saved → {path}")


# ===========================================================================
# Consolidated correlation table for paper (tab:lira)
# ===========================================================================

def table_lira_paper(cert_dir=CERT_DIR, lira_dir=LIRA_DIR, runs_dir=RUNS_DIR,
                     out_dir=OUT_DIR, seeds=(0, 1, 2)):
    """
    Four rows (ε^dir, ε^norm, loss, gen-leverage) × four scopes (overall/head/mid/tail).
    Mean Spearman across seeds 0,1,2; ε^norm constant in mid/tail → 'undef.'.
    Outputs table_lira_paper.json and table_lira_paper.tex.
    """
    print(f"\n{'='*72}")
    print(f"  table_lira_paper: consolidated correlation table (tab:lira)")
    print(f"{'='*72}")

    tier_names_map = {0: "head", 1: "mid", 2: "tail"}
    scopes         = ["overall", "head", "mid", "tail"]
    row_keys       = ["eps_dir", "eps_norm", "loss", "gen_leverage"]
    accum          = {rk: {sc: [] for sc in scopes} for rk in row_keys}

    for seed in seeds:
        lf1  = _load_lira("LF1", lira_dir, seed=seed)
        cert = _load_cert("F1",  seed, cert_dir)
        run  = _load_run("F1",   seed, runs_dir)

        if not lf1 or "D_lira_members" not in lf1:
            print(f"  [missing] D_lira_members LF1 seed={seed}"); continue
        if "epsilon_cert_dir_rank_100" not in cert or "epsilon_cert_norm" not in cert:
            print(f"  [missing] cert arrays F1 seed={seed}"); continue
        ml_path = os.path.join(runs_dir, "F1", f"seed_{seed}", "lira_member_local_idx.npy")
        if not os.path.exists(ml_path):
            print(f"  [missing] lira_member_local_idx.npy seed={seed}"); continue

        member_local = np.load(ml_path)
        D_lira = lf1["D_lira_members"]
        n_m    = min(len(D_lira), len(member_local))
        D_use  = D_lira[:n_m]
        idx    = member_local[:n_m]
        tier_all = run.get("tier_labels")
        tier_m   = tier_all[idx] if tier_all is not None else None

        eps_dir_m  = cert["epsilon_cert_dir_rank_100"][idx]
        eps_norm_m = cert["epsilon_cert_norm"][idx]
        losses_all = run.get("losses")
        loss_m     = losses_all[idx, -1] if losses_all is not None else None

        leverage = None
        yp_path = os.path.join(runs_dir, "F1", f"seed_{seed}", "Y_projections.npy")
        b_path  = os.path.join(runs_dir, "F1", f"seed_{seed}", "B_matrices.npy")
        m_path  = os.path.join(runs_dir, "F1", f"seed_{seed}", "YTY_matrices.npy")
        if os.path.exists(yp_path) and os.path.exists(b_path) and os.path.exists(m_path):
            Y_proj  = np.load(yp_path)
            B_mats  = np.load(b_path)
            M_mats  = np.load(m_path)
            Y_final = Y_proj[idx, -1, :].astype(np.float64)
            leverage = _compute_gen_leverage_batch(
                Y_final, B_mats[-1].astype(np.float64),
                M_mats[-1].astype(np.float64), mode="pinv")
        else:
            print(f"  [missing] Nystrom matrices F1 seed={seed} — gen-leverage skipped.")

        def _rho(arr, D, mask=None):
            if arr is None: return float("nan")
            a, d = (np.asarray(arr, np.float64),
                    np.asarray(D,   np.float64))
            if mask is not None: a, d = a[mask], d[mask]
            rho, _, _ = _spearman_ci(a, d, n_boot=0)
            return rho

        for scope in scopes:
            if scope == "overall":
                mask = None
                D_sc = D_use
            else:
                if tier_m is None: continue
                t_id = {v: k for k, v in tier_names_map.items()}[scope]
                mask = (tier_m == t_id)
                if mask.sum() < 5: continue
                D_sc = D_use[mask]

            def _apply(arr):
                if arr is None: return float("nan")
                a = np.asarray(arr, np.float64)
                return _rho(a if mask is None else a[mask], D_sc)

            # ε^norm: undef. if constant within mid/tail
            en_sc = eps_norm_m if mask is None else eps_norm_m[mask]
            if scope in ("mid", "tail") and (en_sc.max() - en_sc.min()) < 0.01:
                accum["eps_norm"][scope].append("undef.")
            else:
                accum["eps_norm"][scope].append(_apply(eps_norm_m))

            accum["eps_dir"][scope].append(_apply(eps_dir_m))
            accum["loss"][scope].append(_apply(loss_m))
            accum["gen_leverage"][scope].append(
                _rho(leverage if mask is None else leverage[mask], D_sc)
                if leverage is not None else float("nan"))

    def _mean_or_undef(vals):
        numeric = [v for v in vals if isinstance(v, float) and np.isfinite(v)]
        has_undef = any(v == "undef." for v in vals)
        if has_undef and not numeric: return "undef."
        if not numeric: return float("nan")
        return float(np.mean(numeric))

    cells = {(rk, sc): _mean_or_undef(accum[rk][sc])
             for rk in row_keys for sc in scopes}

    # Print table
    row_labels = {"eps_dir": "ε^dir", "eps_norm": "ε^norm",
                  "loss": "loss", "gen_leverage": "gen-leverage"}
    print(f"\n  {'Row':14s} | {'Overall':8s} | {'Head':8s} | {'Mid':8s} | {'Tail':8s}")
    print(f"  {'-'*60}")
    for rk in row_keys:
        def _f(v):
            if v == "undef.": return "undef.  "
            if isinstance(v, float) and not np.isfinite(v): return "nan     "
            return f"{float(v):8.3f}"
        print(f"  {row_labels[rk]:14s} | " +
              " | ".join(_f(cells[(rk, sc)]) for sc in scopes))

    # Footnote scalars from existing outputs
    ft_tier = float("nan")
    ft_ltiqr = float("nan")
    t3_path = os.path.join(out_dir, "table3_lira_correlation.json")
    if os.path.exists(t3_path):
        with open(t3_path) as f:
            t3_rows = json.load(f)
        for r in t3_rows:
            if isinstance(r.get("label"), str) and "tier label" in r["label"].lower():
                v = r.get("rho")
                if v is not None and np.isfinite(float(v)):
                    ft_tier = float(v); break
    bl_path = os.path.join(out_dir, "table3_baselines.json")
    if os.path.exists(bl_path):
        with open(bl_path) as f:
            bl_data = json.load(f)
        bl_rows = bl_data.get("rows", bl_data) if isinstance(bl_data, dict) else bl_data
        rhos = [float(r["rho_mean"]) for r in bl_rows
                if isinstance(r.get("label"), str)
                and "AGGREGATE" in r["label"] and "LT-IQR" in r["label"]
                and r.get("scope") == "overall" and "rho_mean" in r]
        if rhos: ft_ltiqr = float(np.mean(rhos))

    print(f"\n  Footnote: tier-label ρ={ft_tier:.3f}  LT-IQR ρ={ft_ltiqr:.3f}")

    # Serialise
    cells_json = {f"{rk}|{sc}":
                  (v if isinstance(v, str) else
                   (None if isinstance(v, float) and not np.isfinite(v) else v))
                  for (rk, sc), v in cells.items()}
    out_json = {
        "cells": cells_json,
        "footnote_tier_label_spearman":
            (float(ft_tier)  if np.isfinite(ft_tier)  else None),
        "footnote_lt_iqr_spearman":
            (float(ft_ltiqr) if np.isfinite(ft_ltiqr) else None),
        "provenance": ("F1×LF1 seeds 0,1,2 — mean Spearman; "
                       "no new data generated"),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "table_lira_paper.json"), "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"  [saved] {out_dir}/table_lira_paper.json")

    # LaTeX
    def _tex(v):
        if v == "undef.": return r"\text{undef.}"
        if v is None or (isinstance(v, float) and not np.isfinite(v)): return "--"
        return f"{float(v):.3f}"

    row_tex = {
        "eps_dir":      r"$\varepsilon^{\mathrm{dir}}$",
        "eps_norm":     r"$\varepsilon^{\mathrm{norm}}$",
        "loss":         r"loss",
        "gen_leverage": r"gen-leverage",
    }
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Predictor & Overall & Head & Mid & Tail \\", r"\midrule"]
    for rk in row_keys:
        vals = " & ".join(_tex(cells[(rk, sc)]) for sc in scopes)
        lines.append(f"{row_tex[rk]} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              f"% Footnote: tier-label ρ={ft_tier:.3f}; LT-IQR ρ={ft_ltiqr:.3f}"]
    tex_path = os.path.join(out_dir, "table_lira_paper.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [saved] {tex_path}")
    return cells, ft_tier, ft_ltiqr


# ===========================================================================
# Accounting note (appendix calibration)
# ===========================================================================

def accounting_note(runs_dir=RUNS_DIR, cert_dir=CERT_DIR, out_dir=OUT_DIR):
    """Deterministic RDP/PRV accounting on saved metadata — no retraining."""
    print(f"\n{'='*72}")
    print(f"  accounting_note: F1 seed 0")
    print(f"{'='*72}")

    meta_path = os.path.join(runs_dir, "F1", "seed_0", "metadata.json")
    if not os.path.exists(meta_path):
        print(f"  [missing] {meta_path}"); return {}
    with open(meta_path) as f:
        meta = json.load(f)

    sigma = float(meta.get("sigma", meta.get("noise_multiplier", 1.484)))
    q     = float(meta.get("q",     meta.get("sampling_rate",   0.111)))
    T     = int(  meta.get("T_train", meta.get("T_steps", meta.get("num_steps", 360))))
    delta = float(meta.get("delta", 1e-5))
    print(f"  σ={sigma}  q={q}  T={T}  δ={delta:.1e}")

    # ε_norm_fullclip: subsampled-Gaussian RDP composed T steps, minimized over α.
    # Must include Poisson subsampling amplification (rate q) — the bare
    # α/(2σ²) formula without subsampling gives ~145 and is wrong here.
    log_inv_delta = math.log(1.0 / delta)
    best_eps = float("inf"); best_alpha = float("nan")
    _rdp_ok = False
    try:
        orders = [float(a) for a in ALPHA_GRID if a > 1.0]
        try:
            from opacus.accountants.analysis.rdp import compute_rdp as _crdp
        except ImportError:
            from opacus.accountants.rdp_accountant import compute_rdp as _crdp
        rdp_step = _crdp(q=q, noise_multiplier=sigma, steps=1, orders=orders)
        rdp_composed = T * np.asarray(rdp_step, dtype=np.float64)
        alpha_arr    = np.asarray(orders, dtype=np.float64)
        eps_cands    = rdp_composed + log_inv_delta / (alpha_arr - 1.0)
        best_idx     = int(np.argmin(eps_cands))
        best_eps     = float(eps_cands[best_idx])
        best_alpha   = float(alpha_arr[best_idx])
        _rdp_ok      = True
        print(f"  ε_norm_fullclip (RDP+subsampling) = {best_eps:.4f}  at α* = {best_alpha:.2f}")
    except Exception as e:
        print(f"  [warn] Opacus RDP compute failed ({e}); trying table2 fallback.")

    if not _rdp_ok:
        # Fallback: read median ε^norm for F1 seed 0 from table2 (same value by construction).
        t2_path = os.path.join(out_dir, "table2_certified_spread.json")
        if os.path.exists(t2_path):
            with open(t2_path) as f:
                _t2 = json.load(f)
            _f1s0 = next((r for r in _t2 if r["run_id"] == "F1" and r.get("seed") == 0), None)
            if _f1s0:
                best_eps = float(_f1s0["norm_med"])
                print(f"  ε_norm_fullclip (fallback table2 F1 seed0 norm_med) = {best_eps:.4f}")
        if best_eps == float("inf"):
            print("  [warn] ε_norm_fullclip could not be determined.")

    # ε_global_prv: try metadata / cert summary first, else PRV accountant
    eps_prv = None
    for src_dict in [meta]:
        for key in ["global_eps_prv", "eps_prv", "epsilon_prv",
                    "global_epsilon_prv", "epsilon_global", "eps_global"]:
            if key in src_dict:
                eps_prv = float(src_dict[key])
                print(f"  ε_global_prv (metadata) = {eps_prv:.4f}"); break
        if eps_prv is not None: break

    cert_summ = os.path.join(cert_dir, "p19_F1_seed0_summary.json")
    if eps_prv is None and os.path.exists(cert_summ):
        with open(cert_summ) as f:
            summ = json.load(f)
        for key in ["global_eps_prv", "eps_prv", "epsilon_prv", "global_epsilon_prv"]:
            if key in summ:
                eps_prv = float(summ[key])
                print(f"  ε_global_prv (cert summary) = {eps_prv:.4f}"); break

    if eps_prv is None:
        try:
            from opacus.accountants import PRVAccountant
            acc = PRVAccountant()
            acc.history = [(sigma, q, T)]
            eps_prv = float(acc.get_epsilon(delta=delta))
            print(f"  ε_global_prv (PRV recompute) = {eps_prv:.4f}")
        except Exception as e:
            print(f"  [warn] PRV recompute failed: {e}")
            eps_prv = float("nan")

    result = {
        "sigma": sigma, "q": q, "T": T, "delta": delta,
        "eps_norm_fullclip": float(best_eps),
        "best_alpha": float(best_alpha),
        "eps_global_prv": eps_prv,
        "provenance": "F1 seed_0 metadata.json — deterministic post-processing only",
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "accounting_note.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [saved] {out_path}")
    return result


# ===========================================================================
# Numbers manifest
# ===========================================================================

def emit_paper_numbers(cert_dir=CERT_DIR, lira_dir=LIRA_DIR, runs_dir=RUNS_DIR,
                       out_dir=OUT_DIR):
    """Assemble paper_numbers.json from all saved JSON outputs."""
    print(f"\n{'='*72}")
    print(f"  emit_paper_numbers")
    print(f"{'='*72}")

    def _load_or_run(path, fn):
        if not os.path.exists(path):
            print(f"  [auto-run] {os.path.basename(path)} absent — running prerequisite.")
            fn()
        if not os.path.exists(path):
            print(f"  [missing] {path}"); return None
        with open(path) as f:
            return json.load(f)

    numbers = {}
    prov    = []

    # Table 1
    t1 = _load_or_run(os.path.join(out_dir, "table1_norm_flatness.json"),
                      lambda: table_1(runs_dir, out_dir))
    if t1:
        f1r = [r for r in t1 if r["run_id"] == "F1"]
        f2r = [r for r in t1 if r["run_id"] == "F2"]
        f5r = [r for r in t1 if r["run_id"] == "F5"]
        if f1r: numbers["sec6_p2_frac_clipped_S1"] = float(f1r[0]["frac_within_1pct_C"])
        if f5r: numbers["sec6_p2_frac_clipped_S3"] = float(np.mean([r["frac_within_1pct_C"] for r in f5r]))
        if f2r: numbers["sec6_p2_S2_norm_CV"]      = float(f2r[0]["cv"])
        # median_norm in table1 is the median clipped norm (~1.0 = C), not ε^norm.
        # fig_wrn_S3_norm_reference is set from table2 norm_med below.
        prov.append("table1_norm_flatness.json")

    # Table 2
    t2 = _load_or_run(os.path.join(out_dir, "table2_certified_spread.json"),
                      lambda: table_2(cert_dir, runs_dir, out_dir))
    if t2:
        f2r    = [r for r in t2 if r["run_id"] == "F2"]
        f5r_t2 = [r for r in t2 if r["run_id"] == "F5"]
        if f2r:
            numbers["sec6_p2_S2_dir_CV"]         = float(f2r[0]["dir_cv"])
            numbers["fig_degeneracy_norm_value"]  = float(f2r[0]["norm_med"])
            numbers["fig_degeneracy_dir_p5"]      = float(f2r[0]["dir_p5"])
            numbers["fig_degeneracy_dir_p95"]     = float(f2r[0]["dir_p95"])
        if f5r_t2:
            # norm_med in table2 = median(ε^norm) for F5 — this is the ~9.50 reference.
            numbers["fig_wrn_S3_norm_reference"] = float(
                np.mean([r["norm_med"] for r in f5r_t2]))
        prov.append("table2_certified_spread.json")

    # Table 3
    t3 = _load_or_run(os.path.join(out_dir, "table3_lira_correlation.json"),
                      lambda: table_3(cert_dir, lira_dir, runs_dir, out_dir))
    if t3:
        agg_dir = next((r for r in t3 if isinstance(r.get("label"), str)
                        and "AGGREGATE ε^dir" in r["label"]), None)
        if agg_dir:
            numbers["appendix_overall_seed_std"] = float(agg_dir.get("rho_std", float("nan")))
        for r in t3:
            lbl = str(r.get("label", ""))
            ci_lo = r.get("ci_lo"); ci_hi = r.get("ci_hi")
            if ci_lo is None or not np.isfinite(float(ci_lo)): continue
            if "ε^dir cert" in lbl and not r.get("within_tier") and "AGGREGATE" not in lbl:
                numbers.setdefault("appendix_dir_CI_overall_lo", float(ci_lo))
                numbers.setdefault("appendix_dir_CI_overall_hi", float(ci_hi))
            if "ε^dir cert" in lbl and r.get("tier") == "tail":
                numbers.setdefault("appendix_dir_CI_tail_lo", float(ci_lo))
                numbers.setdefault("appendix_dir_CI_tail_hi", float(ci_hi))
            # Overall loss: table3 labels "final training loss" (no within_tier flag)
            if "final training loss" in lbl and not r.get("within_tier") and "AGGREGATE" not in lbl:
                numbers.setdefault("appendix_loss_CI_overall_lo", float(ci_lo))
                numbers.setdefault("appendix_loss_CI_overall_hi", float(ci_hi))
            # Within-tier tail loss: labels are "[s0]   loss [tail]" — no "final training loss"
            _is_loss_row = ("final training loss" in lbl or
                            ("loss" in lbl and "[tail]" in lbl))
            if _is_loss_row and r.get("tier") == "tail":
                numbers.setdefault("appendix_loss_CI_tail_lo", float(ci_lo))
                numbers.setdefault("appendix_loss_CI_tail_hi", float(ci_hi))
        prov.append("table3_lira_correlation.json")

    # Table 4
    t4 = _load_or_run(os.path.join(out_dir, "table4_discretization_overhead.json"),
                      lambda: table_4(cert_dir, out_dir))
    if t4:
        for r in t4:
            rid = r["run_id"]
            if rid == "F1":
                numbers["appendix_filter_S1_med_pct"]  = float(r["dir_overhead_pct_med"])
                numbers["appendix_filter_S1_p95_pct"]  = float(r["dir_overhead_pct_p95"])
                numbers["appendix_filter_S1_max"]      = float(r["dir_overhead_max"])
            if rid == "F5":
                numbers["appendix_filter_S3_med_pct"]  = float(r["dir_overhead_pct_med"])
                numbers["appendix_filter_S3_p95_pct"]  = float(r["dir_overhead_pct_p95"])
        prov.append("table4_discretization_overhead.json")

    # Table 5
    t5 = _load_or_run(os.path.join(out_dir, "table5_wrn_robustness.json"),
                      lambda: table_5(cert_dir, lira_dir, runs_dir, out_dir))
    if t5:
        tier_mask_acc = {"head": [], "mid": [], "tail": []}
        for row in t5:
            for t_name, td in row.get("tiers", {}).items():
                if t_name in tier_mask_acc:
                    tier_mask_acc[t_name].append(float(td["masking_pct"]))
        all_pct = []
        for t_name, vals in tier_mask_acc.items():
            if vals:
                m = float(np.mean(vals))
                numbers[f"sec6_p4_S3_{t_name}_masking_pct"] = m
                all_pct.extend(vals)
        if all_pct:
            numbers["sec6_p4_S3_masking_range_lo"] = float(min(all_pct))
            numbers["sec6_p4_S3_masking_range_hi"] = float(max(all_pct))
        prov.append("table5_wrn_robustness.json")

    # Table 6
    t6 = _load_or_run(os.path.join(out_dir, "table6_rank_ablation.json"),
                      lambda: table_6(cert_dir, out_dir))
    if t6:
        f5_rank_acc = {}
        for r in t6:
            rk = r["rank"]
            if r["run_id"] == "F1":
                numbers[f"appendix_minorant_S1_r{rk}_med"] = float(r["eps_dir_med"])
            if r["run_id"] == "F5":
                f5_rank_acc.setdefault(rk, []).append(float(r["eps_dir_med"]))
            if rk == 200:
                numbers[f"appendix_minorant_{r['run_id']}_r200_delta"] = \
                    float(r.get("delta_vs_prev_rank", float("nan")))
        for rk, vals in f5_rank_acc.items():
            if vals:
                numbers[f"appendix_minorant_S3_r{rk}_med"] = float(np.mean(vals))
        prov.append("table6_rank_ablation.json")

    # Table baselines
    tb_raw = _load_or_run(os.path.join(out_dir, "table3_baselines.json"),
                          lambda: table_baselines(cert_dir, lira_dir, runs_dir, out_dir))
    if tb_raw:
        bl_rows = tb_raw.get("rows", tb_raw) if isinstance(tb_raw, dict) else tb_raw
        prec_acc: dict = {}
        for r in bl_rows:
            pred  = r.get("predictor", "")
            scope = r.get("scope", "")
            for k_tag in ["prec_at_1pct", "prec_at_3pct", "prec_at_5pct"]:
                val = (r.get("precision_at_k") or {}).get(k_tag)
                if val is not None and np.isfinite(float(val)):
                    prec_acc.setdefault((pred, scope, k_tag), []).append(float(val))
        for pred, nlabel in [("eps_dir","dir"), ("eps_norm","norm"),
                              ("loss_final","loss"), ("lt_iqr","ltiqr")]:
            for k_tag, k_lbl in [("prec_at_1pct","1"), ("prec_at_3pct","3"), ("prec_at_5pct","5")]:
                key = (pred, "overall", k_tag)
                if key in prec_acc:
                    numbers[f"appendix_prec_{nlabel}_at_{k_lbl}pct"] = \
                        float(np.mean(prec_acc[key]))
        # Expose eps_dir precision also under sec6 key
        for k_tag, k_lbl in [("prec_at_1pct","1"), ("prec_at_3pct","3"), ("prec_at_5pct","5")]:
            src = f"appendix_prec_dir_at_{k_lbl}pct"
            if src in numbers:
                numbers[f"sec6_p3_precision_dir_at_{k_lbl}pct"] = numbers[src]
            src2 = f"appendix_prec_norm_at_{k_lbl}pct"
            if src2 in numbers:
                numbers[f"sec6_p3_precision_norm_at_{k_lbl}pct"] = numbers[src2]
        for r in bl_rows:
            if (isinstance(r.get("label"), str) and "AGGREGATE" in r["label"]
                    and "gen-leverage" in r["label"] and "rho_mean" in r):
                sc  = r.get("scope", "")
                val = float(r["rho_mean"])
                if np.isfinite(val):
                    numbers[f"sec6_p3_gen_leverage_{sc}"] = val
        prov.append("table3_baselines.json")

    # table_lira_paper.json
    tlp = _load_or_run(os.path.join(out_dir, "table_lira_paper.json"),
                       lambda: table_lira_paper(cert_dir, lira_dir, runs_dir, out_dir))
    if tlp:
        for rk, rl in [("eps_dir","dir"), ("eps_norm","norm"),
                        ("loss","loss"), ("gen_leverage","gen_leverage")]:
            for sc in ["overall", "head", "mid", "tail"]:
                v = tlp.get("cells", {}).get(f"{rk}|{sc}")
                if v == "undef.":
                    numbers[f"tab_lira_{rl}_{sc}"] = "undef."
                elif v is not None:
                    try:
                        fv = float(v)
                        if np.isfinite(fv):
                            numbers[f"tab_lira_{rl}_{sc}"] = fv
                    except (TypeError, ValueError):
                        pass
        numbers["footnote_tier_label_spearman"] = tlp.get("footnote_tier_label_spearman")
        numbers["footnote_lt_iqr_spearman"]     = tlp.get("footnote_lt_iqr_spearman")
        prov.append("table_lira_paper.json")

    # Rank stability F1
    rs_raw = _load_or_run(os.path.join(out_dir, "rank_stability_F1.json"),
                          lambda: rank_stability(cert_dir, runs_dir, out_dir, "F1"))
    if rs_raw:
        rs_rows = rs_raw.get("rows", rs_raw) if isinstance(rs_raw, dict) else rs_raw
        for r in rs_rows:
            if r.get("pair") == "median_pairwise":
                numbers[f"appendix_rank_stability_median_pairwise_{r.get('scope','')}"] = \
                    float(r["spearman"])
        prov.append("rank_stability_F1.json")

    # Gaussian validation dominant (F1 seed 0)
    gv = _load_or_run(os.path.join(out_dir, "gaussian_validation_F1_seed0.json"),
                      lambda: gaussian_validation("F1", 0, runs_dir, out_dir))
    if gv:
        ks_med  = [r["ks_median"] for r in gv if np.isfinite(r.get("ks_median", float("nan")))]
        ks_max  = [r["ks_max"]    for r in gv if np.isfinite(r.get("ks_max",    float("nan")))]
        sk_meds = [r["skewness_abs_med"] for r in gv
                   if np.isfinite(r.get("skewness_abs_med", float("nan")))]
        ku_meds = [r["excess_kurtosis_abs_med"] for r in gv
                   if np.isfinite(r.get("excess_kurtosis_abs_med", float("nan")))]
        if ks_med:
            numbers["appendix_gauss_dom_ks_median"] = float(np.median(ks_med))
            numbers["appendix_gauss_dom_ks_max"]    = float(np.max(ks_max)) if ks_max else float("nan")
        if sk_meds: numbers["appendix_gauss_dom_skew_max"]  = float(np.max(sk_meds))
        if ku_meds: numbers["appendix_gauss_dom_kurt_max"]  = float(np.max(ku_meds))
        prov.append("gaussian_validation_F1_seed0.json")

    # Gaussian validation lowvar F1 seeds 0-2
    for sub, nlabel in [("bottom", "bot"), ("tail_orthogonal", "tail_orth")]:
        all_ks: list = []
        for seed in [0, 1, 2]:
            gvlv = _load_or_run(
                os.path.join(out_dir, f"gaussian_validation_F1_lowvar_seed{seed}.json"),
                lambda s=seed: gaussian_validation_lowvar("F1", s, runs_dir, out_dir))
            if gvlv:
                all_ks.extend(r["ks"] for r in gvlv
                               if r.get("subspace") == sub
                               and np.isfinite(r.get("ks", float("nan"))))
        if all_ks:
            numbers[f"appendix_gauss_{nlabel}_ks_median"] = float(np.median(all_ks))
            numbers[f"appendix_gauss_{nlabel}_ks_max"]    = float(np.max(all_ks))
        prov.append(f"gaussian_validation_F1_lowvar_seed{{0,1,2}}.json subspace={sub}")

    # Tier populations and member-target counts
    run_f1 = _load_run("F1", 0, runs_dir)
    if "tier_labels" in run_f1:
        tl = run_f1["tier_labels"]
        counts = np.bincount(tl.astype(int), minlength=3)
        numbers["data_tier_pop_head"] = int(counts[0])
        numbers["data_tier_pop_mid"]  = int(counts[1])
        numbers["data_tier_pop_tail"] = int(counts[2])
        print(f"  Tier populations (F1 seed 0): {counts[0]}/{counts[1]}/{counts[2]}")
        ml_path = os.path.join(runs_dir, "F1", "seed_0", "lira_member_local_idx.npy")
        if os.path.exists(ml_path):
            ml = np.load(ml_path)
            mc = np.bincount(tl[ml].astype(int), minlength=3)
            numbers["data_member_tier_count_head"]  = int(mc[0])
            numbers["data_member_tier_count_mid"]   = int(mc[1])
            numbers["data_member_tier_count_tail"]  = int(mc[2])
            numbers["data_member_tier_count_total"] = int(mc.sum())
            print(f"  Member-target counts: {mc[0]}/{mc[1]}/{mc[2]} (total={mc.sum()})")
        prov.append("tier_labels.npy + lira_member_local_idx.npy F1 seed 0")

    # Accounting note
    acc = _load_or_run(os.path.join(out_dir, "accounting_note.json"),
                       lambda: accounting_note(runs_dir, cert_dir, out_dir))
    if acc:
        numbers["appendix_accounting_sigma"] = acc.get("sigma")
        numbers["appendix_accounting_q"]     = acc.get("q")
        numbers["appendix_accounting_T"]     = acc.get("T")
        numbers["appendix_fullclip_eps"]     = acc.get("eps_norm_fullclip")
        numbers["appendix_fullclip_alpha"]   = acc.get("best_alpha")
        numbers["appendix_prv_eps"]          = acc.get("eps_global_prv")
        prov.append("accounting_note.json")

    # Write
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "paper_numbers.json")
    with open(out_path, "w") as f:
        json.dump({"numbers": numbers, "provenance_log": prov,
                   "note": "All values from saved arrays; no training/cert/LiRA calls."},
                  f, indent=2, default=str)
    print(f"\n  [saved] {out_path}")

    # Pretty-print groups
    groups = [
        ("§6 ¶2", ["sec6_p2_frac_clipped_S1","sec6_p2_frac_clipped_S3",
                    "sec6_p2_S2_dir_CV","sec6_p2_S2_norm_CV"]),
        ("§6 ¶3", ["tab_lira_dir_mid","tab_lira_dir_tail","tab_lira_norm_mid","tab_lira_norm_tail",
                    "sec6_p3_precision_dir_at_1pct","sec6_p3_precision_norm_at_1pct",
                    "sec6_p3_gen_leverage_overall","sec6_p3_gen_leverage_head",
                    "sec6_p3_gen_leverage_mid","sec6_p3_gen_leverage_tail"]),
        ("§6 ¶4", ["sec6_p4_S3_head_masking_pct","sec6_p4_S3_mid_masking_pct",
                    "sec6_p4_S3_tail_masking_pct",
                    "sec6_p4_S3_masking_range_lo","sec6_p4_S3_masking_range_hi"]),
        ("Fig captions", ["fig_degeneracy_norm_value","fig_degeneracy_dir_p5",
                           "fig_degeneracy_dir_p95","fig_wrn_S3_norm_reference"]),
        ("tab:lira",    [k for k in numbers if k.startswith("tab_lira_")
                         or k.startswith("footnote_")]),
        ("Tier counts", ["data_tier_pop_head","data_tier_pop_mid","data_tier_pop_tail",
                          "data_member_tier_count_head","data_member_tier_count_mid",
                          "data_member_tier_count_tail","data_member_tier_count_total"]),
        ("Appendix CIs", ["appendix_overall_seed_std",
                           "appendix_dir_CI_overall_lo","appendix_dir_CI_overall_hi",
                           "appendix_dir_CI_tail_lo","appendix_dir_CI_tail_hi",
                           "appendix_loss_CI_overall_lo","appendix_loss_CI_overall_hi",
                           "appendix_loss_CI_tail_lo","appendix_loss_CI_tail_hi"]),
        ("Rank stability", [k for k in numbers if "rank_stability" in k]),
        ("Precision grids", [k for k in numbers if "prec_" in k or "precision_" in k]),
        ("Accounting",  ["appendix_accounting_sigma","appendix_accounting_q",
                          "appendix_accounting_T","appendix_fullclip_eps",
                          "appendix_fullclip_alpha","appendix_prv_eps"]),
        ("Minorant ranks", [k for k in numbers if "minorant" in k]),
        ("Filter",      [k for k in numbers if "filter" in k]),
        ("Gaussianity", [k for k in numbers if "gauss" in k]),
    ]
    for grp, keys in groups:
        found = {k: numbers[k] for k in keys if k in numbers}
        if not found: continue
        print(f"\n  [{grp}]")
        for k, v in found.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    return numbers


# ===========================================================================
# Acceptance check
# ===========================================================================

def paper_check(out_dir=OUT_DIR):
    """--paper_check: compare paper_numbers.json to §9 draft targets."""
    path = os.path.join(out_dir, "paper_numbers.json")
    if not os.path.exists(path):
        print(f"  [paper_check] paper_numbers.json not found. Run --paper first.")
        return

    with open(path) as f:
        pn = json.load(f)
    numbers = pn.get("numbers", pn)

    DRAFT = {
        "sec6_p2_frac_clipped_S1":            (0.54,   0.05),
        "sec6_p2_frac_clipped_S3":            (0.78,   0.05),
        "sec6_p2_S2_dir_CV":                  (0.169,  0.02),
        "tab_lira_dir_overall":               (0.700,  0.01),
        "tab_lira_dir_head":                  (0.313,  0.01),
        "tab_lira_dir_mid":                   (0.548,  0.01),
        "tab_lira_dir_tail":                  (0.718,  0.01),
        "tab_lira_norm_overall":              (0.467,  0.01),
        "tab_lira_norm_head":                 (0.290,  0.01),
        "tab_lira_loss_overall":              (0.682,  0.01),
        "tab_lira_loss_head":                 (0.312,  0.01),
        "tab_lira_loss_mid":                  (0.497,  0.01),
        "tab_lira_loss_tail":                 (0.596,  0.01),
        "tab_lira_gen_leverage_overall":      (0.57,   0.01),
        "tab_lira_gen_leverage_head":         (0.29,   0.01),
        "tab_lira_gen_leverage_mid":          (0.42,   0.01),
        "tab_lira_gen_leverage_tail":         (0.38,   0.01),
        "sec6_p3_precision_dir_at_1pct":      (0.29,   0.01),
        "sec6_p3_precision_norm_at_1pct":     (0.013,  0.01),
        "fig_degeneracy_norm_value":          (9.53,   0.05),
        "appendix_accounting_sigma":          (1.484,  0.001),
        "appendix_accounting_q":              (0.111,  0.005),
        "appendix_accounting_T":              (360,    1),
        "appendix_fullclip_eps":              (9.53,   0.05),
        "appendix_fullclip_alpha":            (3.5,    0.5),
        "appendix_prv_eps":                   (7.99,   0.1),
        "appendix_gauss_dom_ks_median":       (0.013,  0.002),
        "appendix_gauss_dom_ks_max":          (0.029,  0.002),
        "appendix_gauss_bot_ks_median":       (0.011,  0.002),
        "appendix_gauss_bot_ks_max":          (0.024,  0.002),
        "appendix_gauss_tail_orth_ks_median": (0.011,  0.002),
        "appendix_gauss_tail_orth_ks_max":    (0.021,  0.002),
        "data_tier_pop_head":                 (9301,   100),
        "data_tier_pop_mid":                  (2857,   100),
        "data_tier_pop_tail":                 (444,    50),
        "data_member_tier_count_head":        (544,    50),
        "data_member_tier_count_mid":         (512,    50),
        "data_member_tier_count_tail":        (444,    50),
        "data_member_tier_count_total":       (1500,   10),
    }
    UNDEF_EXPECTED = {"tab_lira_norm_mid", "tab_lira_norm_tail"}

    print(f"\n{'='*72}")
    print(f"  paper_check: §9 draft comparison")
    print(f"{'='*72}")
    n_pass = n_fail = n_miss = 0

    for key, (draft, tol) in sorted(DRAFT.items()):
        actual = numbers.get(key)
        if actual is None:
            print(f"  [MISSING] {key:55s}  draft={draft}")
            n_miss += 1; continue
        try:
            af = float(actual)
        except (TypeError, ValueError):
            print(f"  [SKIP]    {key:55s}  actual={actual!r}"); continue
        diff = abs(af - float(draft))
        tag = "PASS" if diff <= tol else "FAIL"
        print(f"  [{tag}]    {key:55s}  actual={af:.4f}  draft={draft}  |Δ|={diff:.4f}"
              + (f"  tol={tol}" if tag == "FAIL" else ""))
        if tag == "PASS": n_pass += 1
        else:             n_fail += 1

    for key in sorted(UNDEF_EXPECTED):
        actual = numbers.get(key)
        if actual == "undef.":
            print(f"  [PASS]    {key:55s}  = undef. (expected)")
            n_pass += 1
        else:
            print(f"  [FAIL]    {key:55s}  = {actual!r} (expected undef.)")
            n_fail += 1

    print(f"\n  Summary: {n_pass} PASS / {n_fail} FAIL / {n_miss} MISSING")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 19 analysis")
    parser.add_argument("--all",     action="store_true", help="Run all tables and figures")
    parser.add_argument("--table",   type=str, default=None, choices=["1","2","3","4","5","6"],
                        help="Run specific table")
    parser.add_argument("--figure",  type=str, default=None, choices=["1","2"],
                        help="Run specific figure")
    parser.add_argument("--gaussian_validation", action="store_true")
    parser.add_argument("--run",      type=str, default="F1", choices=["F1","F2","F5"])
    parser.add_argument("--seed",     type=int, default=0,
                        help="Seed for --gaussian_validation / --gaussian_lowvar")
    parser.add_argument("--lira_seeds", type=int, nargs="+", default=None,
                        help="LiRA seeds to use for Table 3 / Figure 1 (e.g. --lira_seeds 0 1 2)")
    parser.add_argument("--all_lira_seeds", action="store_true",
                        help="Auto-detect all available LiRA seeds (overrides --lira_seeds)")
    parser.add_argument("--runs_dir", type=str, default=RUNS_DIR)
    parser.add_argument("--cert_dir", type=str, default=CERT_DIR)
    parser.add_argument("--lira_dir", type=str, default=LIRA_DIR)
    parser.add_argument("--out_dir",  type=str, default=OUT_DIR)
    # ------------------------------------------------------------------
    # New items 1-3 (phase19_spex.md remaining analysis spec)
    # ------------------------------------------------------------------
    parser.add_argument("--baselines", action="store_true",
                        help="Item 1: shadow-free baseline rows (LT-IQR, gen-leverage)")
    parser.add_argument("--gaussian_lowvar", action="store_true",
                        help="Item 2: low-variance-direction Gaussianity check")
    parser.add_argument("--rank_stability", action="store_true",
                        help="Item 3: cross-seed rank stability of eps_dir rankings")
    parser.add_argument("--setting",  type=str, default="F1",
                        help="Run ID for items 1-3 (default: F1)")
    parser.add_argument("--lira_id",  type=str, default="LF1",
                        help="LiRA run ID for item 1 (default: LF1)")
    parser.add_argument("--seeds",    type=int, nargs="+", default=[0, 1, 2],
                        help="Seeds for items 1-3 (default: 0 1 2)")
    parser.add_argument("--n_batches", type=int, default=5000,
                        help="Poisson batches for item 2 Gaussianity check")
    parser.add_argument("--k_top",    type=int, default=10,
                        help="Top eigenvectors defining dominant subspace (item 2)")
    parser.add_argument("--k_bot",    type=int, default=5,
                        help="Bottom eigenvectors to test for family (i) (item 2)")
    # ------------------------------------------------------------------
    # Paper-mode flags (Section 4 / 5 rewrite artifacts)
    # ------------------------------------------------------------------
    parser.add_argument("--paper",         action="store_true",
                        help="Produce all paper artifacts: figures A/B/C, tab:lira, paper_numbers.json")
    parser.add_argument("--paper_figures", action="store_true",
                        help="Figures A (fig:degeneracy), B (fig:withintier), C (fig:wrn)")
    parser.add_argument("--paper_table",   action="store_true",
                        help="table_lira_paper() — consolidated 4×4 correlation table")
    parser.add_argument("--paper_numbers", action="store_true",
                        help="emit_paper_numbers() — paper_numbers.json manifest")
    parser.add_argument("--paper_seed",    type=int, default=0,
                        help="Seed for Figure B (default 0)")
    parser.add_argument("--paper_check",   action="store_true",
                        help="Compare paper_numbers.json to §9 draft targets; print PASS/FAIL")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve which LiRA seeds to use for Table 3 / Figure 1
    if args.all_lira_seeds:
        lira_seeds = None   # _available_lira_seeds will auto-detect
    elif args.lira_seeds is not None:
        lira_seeds = args.lira_seeds
    else:
        lira_seeds = None   # auto-detect (falls back to [0] if dir missing)

    if args.all or args.table == "1":
        table_1(args.runs_dir, args.out_dir)
    if args.all or args.table == "2":
        table_2(args.cert_dir, args.runs_dir, args.out_dir)
    if args.all or args.table == "3":
        table_3(args.cert_dir, args.lira_dir, args.runs_dir, args.out_dir,
                lira_seeds=lira_seeds)
    if args.all or args.table == "4":
        table_4(args.cert_dir, args.out_dir)
    if args.all or args.table == "5":
        table_5(args.cert_dir, args.lira_dir, args.runs_dir, args.out_dir)
    if args.all or args.table == "6":
        table_6(args.cert_dir, args.out_dir)
    if args.all or args.figure == "1":
        figure_1(args.cert_dir, args.lira_dir, args.runs_dir, args.out_dir,
                 lira_seeds=lira_seeds)
    if args.all or args.figure == "2":
        figure_2(args.cert_dir, args.runs_dir, args.out_dir)
    if args.gaussian_validation or args.all:
        run_id = args.run if not args.all else "F1"
        seed   = args.seed if not args.all else 0
        gaussian_validation(run_id, seed, args.runs_dir, args.out_dir)

    # ------------------------------------------------------------------
    # Items 1-3: remaining analysis
    # ------------------------------------------------------------------
    if args.baselines or args.all:
        table_baselines(
            cert_dir=args.cert_dir, lira_dir=args.lira_dir,
            runs_dir=args.runs_dir, out_dir=args.out_dir,
            setting=args.setting, lira_id=args.lira_id,
            seeds=tuple(args.seeds),
        )

    if args.gaussian_lowvar or args.all:
        for s in args.seeds:
            gaussian_validation_lowvar(
                run_id=args.setting, seed=s,
                runs_dir=args.runs_dir, out_dir=args.out_dir,
                n_batches=args.n_batches, k_top=args.k_top, k_bot=args.k_bot,
            )

    if args.rank_stability or args.all:
        rank_stability(
            cert_dir=args.cert_dir, runs_dir=args.runs_dir, out_dir=args.out_dir,
            setting=args.setting, seeds=tuple(args.seeds),
        )

    # ------------------------------------------------------------------
    # Paper-mode dispatch
    # ------------------------------------------------------------------
    do_figures  = args.paper or args.paper_figures
    do_table    = args.paper or args.paper_table
    do_numbers  = args.paper or args.paper_numbers

    if do_figures:
        figure_degeneracy_s2(args.cert_dir, args.out_dir)
        figure_withintier_s1(args.cert_dir, args.lira_dir, args.runs_dir,
                              args.out_dir, paper_seed=args.paper_seed)
        figure_wrn_distribution_s3(args.cert_dir, args.runs_dir, args.out_dir)

    if do_table:
        table_lira_paper(args.cert_dir, args.lira_dir, args.runs_dir, args.out_dir)

    if do_numbers:
        emit_paper_numbers(args.cert_dir, args.lira_dir, args.runs_dir, args.out_dir)

    if args.paper_check:
        paper_check(args.out_dir)

    print(f"\n[P19-analysis] Done. Results in {args.out_dir}")


if __name__ == "__main__":
    main()
