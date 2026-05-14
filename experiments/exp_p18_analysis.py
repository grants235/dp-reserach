#!/usr/bin/env python3
"""
Phase 18 Analysis: Tables and Figures (Section 7 of spec)

Produces:
  Table A: norm distribution diagnostics
  Table B: norm-based bound distribution
  Table C: headline LiRA correlation
  Table D: LiRA decile calibration
  Table E: tier separation
  Table F: seed stability
  Table G: rank ablation
  Figure:  LiRA scatter (ε^dir and ε^norm vs D_LiRA)

Usage:
  python experiments/exp_p18_analysis.py --all
  python experiments/exp_p18_analysis.py --table A
  python experiments/exp_p18_analysis.py --figure lira_scatter
"""

import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RUNS_DIR = "./runs"
CERT_DIR = "./certs/p18"
LIRA_DIR = "./lira"
OUT_DIR  = "./results/p18"


def _plt():
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  [warn] matplotlib not available"); return None


def load_cert(setting_id, seed, cert_dir=CERT_DIR):
    tag = f"p18_{setting_id}_seed{seed}"
    eps_dir  = os.path.join(cert_dir, f"{tag}_eps_dir.npy")
    eps_norm = os.path.join(cert_dir, f"{tag}_eps_norm.npy")
    summ     = os.path.join(cert_dir, f"{tag}_summary.json")
    if not all(os.path.exists(p) for p in [eps_dir, eps_norm]):
        return None
    result = {
        "eps_dir":  np.load(eps_dir).astype(np.float64),
        "eps_norm": np.load(eps_norm).astype(np.float64),
    }
    if os.path.exists(summ):
        with open(summ) as f: result["summary"] = json.load(f)
    return result


def load_run_arrays(setting_id, seed, runs_dir=RUNS_DIR):
    run_dir = os.path.join(runs_dir, setting_id, f"seed_{seed}")
    out = {}
    for name in ["clipped_norms", "losses", "labels", "tier_labels",
                 "lira_member_local_idx", "lira_nonmember_test_idx"]:
        p = os.path.join(run_dir, f"{name}.npy")
        if os.path.exists(p): out[name] = np.load(p)
    meta_p = os.path.join(run_dir, "metadata.json")
    if os.path.exists(meta_p):
        with open(meta_p) as f: out["meta"] = json.load(f)
    return out


def load_lira(lira_id, lira_dir=LIRA_DIR):
    d = os.path.join(lira_dir, lira_id)
    out = {}
    for name in ["lira_scores_members", "lira_scores_nonmembers",
                 "llr_dp_members", "llr_dp_nonmembers",
                 "targets_members", "targets_nonmembers"]:
        p = os.path.join(d, f"{name}.npy")
        if os.path.exists(p): out[name] = np.load(p)
    summ = os.path.join(d, "lira_summary.json")
    if os.path.exists(summ):
        with open(summ) as f: out["summary"] = json.load(f)
    return out


# ---------------------------------------------------------------------------
# Table A: Norm distribution diagnostics
# ---------------------------------------------------------------------------

def table_A(setting_id="S2", seed=0):
    """Table A: From clipped_norms at final accounting step."""
    run = load_run_arrays(setting_id, seed)
    if "clipped_norms" not in run:
        print(f"  [Table A] clipped_norms not found for {setting_id}/seed_{seed}"); return

    norms = run["clipped_norms"]     # (n, T_acc)
    C = run["meta"]["C"] if "meta" in run else 1.0
    T_acc = norms.shape[1]

    # Final step diagnostics
    norms_final = norms[:, -1]
    mean_n  = norms_final.mean()
    med_n   = np.median(norms_final)
    cv_n    = norms_final.std() / mean_n if mean_n > 0 else float("nan")
    frac_at_C = (norms_final >= C * 0.99).mean()

    print(f"\n{'='*60}")
    print(f"  Table A: Norm distribution ({setting_id} seed={seed}, T_acc={T_acc})")
    print(f"{'='*60}")
    print(f"  Final-step ||ḡ||:")
    print(f"    mean     = {mean_n:.4f}")
    print(f"    median   = {med_n:.4f}")
    print(f"    CV       = {cv_n:.4f}")
    print(f"    frac≥C   = {frac_at_C:.4f}  (C={C})")

    # Aggregate over all steps
    all_norms = norms.ravel()
    print(f"  All steps (n={len(all_norms):,}):")
    print(f"    mean     = {all_norms.mean():.4f}  median = {np.median(all_norms):.4f}")
    print(f"    CV       = {all_norms.std()/all_norms.mean():.4f}")
    print(f"    frac≥C   = {(all_norms >= C*0.99).mean():.4f}")


# ---------------------------------------------------------------------------
# Table B: Norm-based bound distribution
# ---------------------------------------------------------------------------

def table_B(setting_id="S2", seed=0):
    """Table B: From ε_i^norm."""
    cert = load_cert(setting_id, seed)
    if cert is None:
        print(f"  [Table B] cert not found for {setting_id}/seed_{seed}"); return

    en = cert["eps_norm"]
    print(f"\n{'='*60}")
    print(f"  Table B: ε^norm distribution ({setting_id} seed={seed})")
    print(f"{'='*60}")
    print(f"  n = {len(en)}")
    print(f"  median   = {np.median(en):.4f}")
    print(f"  mean     = {en.mean():.4f}")
    print(f"  CV       = {en.std()/en.mean():.4f}")
    print(f"  max/min  = {en.max():.4f} / {en.min():.4f}")
    print(f"  95th pct = {np.percentile(en,95):.4f}")

    if "summary" in cert:
        s = cert["summary"]
        print(f"  ε^dir:  med={np.median(cert['eps_dir']):.4f}  "
              f"CV={cert['eps_dir'].std()/cert['eps_dir'].mean():.4f}")


# ---------------------------------------------------------------------------
# Table C: Headline LiRA correlation
# ---------------------------------------------------------------------------

def spearman_rho(x, y):
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return float(r), float(p)


def rank_r2(x, y):
    from scipy.stats import rankdata
    rx = rankdata(x); ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1] ** 2)


def bootstrap_ci(x, y, func, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try: vals.append(func(x[idx], y[idx]))
        except Exception: pass
    if not vals: return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def compute_auc(pos_scores, neg_scores):
    from sklearn.metrics import roc_auc_score
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])
    return float(roc_auc_score(labels, scores))


def table_C(setting_id="S2", lira_id="L1", seed=0):
    """Table C: Spearman ρ, rank-rank R², AUC vs D^LiRA."""
    lira  = load_lira(lira_id)
    if not lira or "lira_scores_members" not in lira:
        print(f"  [Table C] LiRA scores not found for {lira_id}"); return

    D_mem = lira["lira_scores_members"].astype(np.float64)
    D_nmem = lira.get("lira_scores_nonmembers", np.zeros(len(D_mem)))
    llr_mem  = lira.get("llr_dp_members",    np.zeros(len(D_mem)))
    llr_nmem = lira.get("llr_dp_nonmembers", np.zeros(len(D_nmem)))

    # Get attack-success threshold τ_{0.1}: FPR=0.1 on nonmembers
    if len(llr_nmem) > 0:
        tau_01 = float(np.percentile(llr_nmem, 90))  # top 10% of nonmembers
    else:
        tau_01 = float("nan")

    cert = load_cert(setting_id, seed)
    run  = load_run_arrays(setting_id, seed)
    member_local_idx = run.get("lira_member_local_idx")

    print(f"\n{'='*70}")
    print(f"  Table C: LiRA Correlation — {setting_id}/seed_{seed} × {lira_id}")
    print(f"{'='*70}")
    print(f"  {'Metric':25s}  {'Spearman':8s}  {'95% CI':15s}  {'Rank R²':8s}  {'AUC':6s}")
    print(f"  {'-'*70}")

    metrics = {}

    if cert is not None and member_local_idx is not None:
        n = min(len(D_mem), len(member_local_idx))
        D_use = D_mem[:n]

        for name, arr in [("ε^dir",  cert["eps_dir"][member_local_idx[:n]]),
                          ("ε^norm", cert["eps_norm"][member_local_idx[:n]])]:
            rho_v, _  = spearman_rho(arr, D_use)
            r2        = rank_r2(arr, D_use)
            ci_lo, ci_hi = bootstrap_ci(arr, D_use,
                                        lambda a, b: spearman_rho(a, b)[0])
            if len(llr_mem) >= n and not np.isnan(tau_01):
                atk_success = (llr_mem[:n] >= tau_01).astype(float)
                try: auc_v = compute_auc(arr[atk_success.astype(bool)],
                                          arr[~atk_success.astype(bool)])
                except Exception: auc_v = float("nan")
            else:
                auc_v = float("nan")
            metrics[name] = {"spearman": rho_v, "ci": (ci_lo, ci_hi),
                              "rank_r2": r2, "auc": auc_v}
            print(f"  {name:25s}  {rho_v:8.4f}  [{ci_lo:.3f},{ci_hi:.3f}]    "
                  f"{r2:8.4f}  {auc_v:6.4f}")

    # Final loss
    if "losses" in run and member_local_idx is not None:
        losses_final = run["losses"][:, -1][member_local_idx[:len(D_mem)]]
        rho_v, _ = spearman_rho(losses_final, D_mem[:len(losses_final)])
        r2 = rank_r2(losses_final, D_mem[:len(losses_final)])
        auc_v = float("nan")
        if len(llr_mem) >= len(losses_final) and not np.isnan(tau_01):
            atk_success = (llr_mem[:len(losses_final)] >= tau_01).astype(bool)
            try: auc_v = compute_auc(losses_final[atk_success], losses_final[~atk_success])
            except Exception: pass
        print(f"  {'Final loss':25s}  {rho_v:8.4f}  {'':15s}  {r2:8.4f}  {auc_v:6.4f}")

    # Class frequency
    if "labels" in run and member_local_idx is not None:
        labels = run["labels"][member_local_idx[:len(D_mem)]]
        cls_counts = np.bincount(run["labels"].astype(int), minlength=int(run["labels"].max()) + 1)
        freq = cls_counts[labels]
        rho_v, _ = spearman_rho(-freq.astype(float), D_mem[:len(freq)])
        r2 = rank_r2(-freq.astype(float), D_mem[:len(freq)])
        auc_v = float("nan")
        if len(llr_mem) >= len(freq) and not np.isnan(tau_01):
            atk_success = (llr_mem[:len(freq)] >= tau_01).astype(bool)
            try: auc_v = compute_auc((-freq.astype(float))[atk_success],
                                      (-freq.astype(float))[~atk_success])
            except Exception: pass
        print(f"  {'Class freq (neg)':25s}  {rho_v:8.4f}  {'':15s}  {r2:8.4f}  {auc_v:6.4f}")

    print(f"  τ_0.1 (FPR=0.1 on nonmembers) = {tau_01:.4f}")


# ---------------------------------------------------------------------------
# Table D: LiRA decile calibration
# ---------------------------------------------------------------------------

def table_D(setting_id="S2", lira_id="L1", seed=0):
    lira = load_lira(lira_id)
    cert = load_cert(setting_id, seed)
    run  = load_run_arrays(setting_id, seed)
    if not lira or cert is None: return

    D_mem = lira["lira_scores_members"].astype(np.float64)
    member_local_idx = run.get("lira_member_local_idx")
    if member_local_idx is None: return

    n = min(len(D_mem), len(member_local_idx))
    eps_dir  = cert["eps_dir"][member_local_idx[:n]]
    eps_norm = cert["eps_norm"][member_local_idx[:n]]
    D_use    = D_mem[:n]
    losses_f = run.get("losses", np.zeros((1,1)))
    loss_final = losses_f[:, -1][member_local_idx[:n]] if losses_f.ndim > 1 else np.zeros(n)

    deciles = np.percentile(D_use, np.arange(0, 110, 10))
    print(f"\n{'='*70}")
    print(f"  Table D: LiRA Decile Calibration ({setting_id}/seed={seed})")
    print(f"{'='*70}")
    print(f"  {'Decile':7s}  {'D_LiRA':8s}  {'ε^dir':8s}  {'ε^norm':8s}  {'loss':8s}")
    for d in range(10):
        lo, hi = deciles[d], deciles[d+1]
        mask = (D_use >= lo) & (D_use < hi) if d < 9 else (D_use >= lo)
        if mask.sum() == 0: continue
        print(f"  {d+1:7d}  {D_use[mask].mean():8.4f}  "
              f"{eps_dir[mask].mean():8.4f}  "
              f"{eps_norm[mask].mean():8.4f}  "
              f"{loss_final[mask].mean():8.4f}")


# ---------------------------------------------------------------------------
# Table E: Tier separation
# ---------------------------------------------------------------------------

def table_E(setting_id="S2", lira_id="L1", seed=0):
    lira = load_lira(lira_id)
    cert = load_cert(setting_id, seed)
    run  = load_run_arrays(setting_id, seed)
    if cert is None: return

    D_mem = lira.get("lira_scores_members", np.zeros(1)).astype(np.float64)
    member_local_idx = run.get("lira_member_local_idx")
    tier_labels = run.get("tier_labels")

    if tier_labels is None:
        print(f"  [Table E] No tier labels for {setting_id}"); return

    n = min(len(D_mem), len(member_local_idx)) if member_local_idx is not None else 0

    print(f"\n{'='*72}")
    print(f"  Table E: Tier Separation ({setting_id}/seed={seed})")
    print(f"{'='*72}")
    print(f"  {'Tier':8s}  {'n':6s}  {'D_LiRA':8s}  {'||g||/C':8s}  {'ε^norm':8s}  {'ε^dir':8s}")

    eps_dir_all  = cert["eps_dir"]
    eps_norm_all = cert["eps_norm"]
    norms_all    = run.get("clipped_norms", np.zeros((1,1)))

    tier_names = {0: "head", 1: "mid", 2: "tail"}
    tier_eps_dir = {}; tier_eps_norm = {}

    for t, tname in tier_names.items():
        mask_priv = (tier_labels == t)
        n_t = mask_priv.sum()
        eps_n = eps_norm_all[mask_priv].mean()
        eps_d = eps_dir_all[mask_priv].mean()
        g_norm_mean = (norms_all[mask_priv].mean() if norms_all.ndim > 1
                       else float("nan"))

        # D_LiRA for members in this tier
        D_t = float("nan")
        if n > 0 and member_local_idx is not None:
            member_tiers = tier_labels[member_local_idx[:n]]
            D_tgt = D_mem[:n]
            mask_t = (member_tiers == t)
            if mask_t.sum() > 0:
                D_t = D_tgt[mask_t].mean()

        tier_eps_dir[tname] = eps_d; tier_eps_norm[tname] = eps_n
        print(f"  {tname:8s}  {n_t:6d}  {D_t:8.4f}  {g_norm_mean:8.4f}  "
              f"{eps_n:8.4f}  {eps_d:8.4f}")

    if "head" in tier_eps_dir and "tail" in tier_eps_dir:
        r_dir  = tier_eps_dir["tail"]  / tier_eps_dir["head"]
        r_norm = tier_eps_norm["tail"] / tier_eps_norm["head"]
        print(f"  {'tail/head':8s}  {'':6s}  {'':8s}  {'':8s}  "
              f"{r_norm:8.4f}  {r_dir:8.4f}")


# ---------------------------------------------------------------------------
# Table F: Seed stability
# ---------------------------------------------------------------------------

def table_F(setting_id="S2"):
    from scipy.stats import spearmanr
    print(f"\n{'='*60}")
    print(f"  Table F: Seed Stability ({setting_id})")
    print(f"{'='*60}")

    certs = {}
    for seed in range(3):
        c = load_cert(setting_id, seed)
        if c is not None: certs[seed] = c

    seeds = sorted(certs.keys())
    if len(seeds) < 2:
        print(f"  Need ≥ 2 seeds (found {len(seeds)})"); return

    pairs = [(s1, s2) for i, s1 in enumerate(seeds) for s2 in seeds[i+1:]]
    rho_dir_vals = []; rho_norm_vals = []
    for s1, s2 in pairs:
        n = min(len(certs[s1]["eps_dir"]), len(certs[s2]["eps_dir"]))
        r_dir, _  = spearmanr(certs[s1]["eps_dir"][:n], certs[s2]["eps_dir"][:n])
        r_norm, _ = spearmanr(certs[s1]["eps_norm"][:n], certs[s2]["eps_norm"][:n])
        rho_dir_vals.append(r_dir); rho_norm_vals.append(r_norm)
        print(f"  seeds ({s1},{s2}):  Spearman ε^dir={r_dir:.4f}  ε^norm={r_norm:.4f}")

    print(f"  Median ε^dir: {np.median(rho_dir_vals):.4f}")
    print(f"  Median ε^norm: {np.median(rho_norm_vals):.4f}")


# ---------------------------------------------------------------------------
# Table G: Rank ablation
# ---------------------------------------------------------------------------

def table_G(setting_id="S2", seed=0, lira_id="L1"):
    summ_path = os.path.join(CERT_DIR, f"p18_{setting_id}_seed{seed}_summary.json")
    if not os.path.exists(summ_path):
        print(f"  [Table G] summary not found for {setting_id}/seed_{seed}"); return

    with open(summ_path) as f:
        summ = json.load(f)

    rank_abl = summ.get("rank_ablation", {})
    if not rank_abl:
        print(f"  [Table G] rank ablation not in summary"); return

    print(f"\n{'='*60}")
    print(f"  Table G: Rank Ablation ({setting_id} seed={seed})")
    print(f"{'='*60}")
    lira = load_lira(lira_id)
    D_mem = lira.get("lira_scores_members") if lira else None
    run = load_run_arrays(setting_id, seed)
    member_local_idx = run.get("lira_member_local_idx")

    print(f"  {'r':6s}  {'ε^dir med':10s}  {'ε^norm med':10s}  {'Spearman':8s}  {'fallback':8s}")
    for r in sorted(int(k) for k in rank_abl.keys()):
        rr = rank_abl[str(r)]
        rho_v = rr.get("spearman_vs_lira", float("nan"))
        rank_cert = os.path.join(CERT_DIR, f"p18_{setting_id}_seed{seed}_eps_dir_r{r}.npy")
        if np.isnan(rho_v) and D_mem is not None and member_local_idx is not None and os.path.exists(rank_cert):
            arr = np.load(rank_cert)
            n = min(len(D_mem), len(member_local_idx))
            try: rho_v, _ = spearman_rho(arr[member_local_idx[:n]], D_mem[:n])
            except Exception: rho_v = float("nan")
        print(f"  {r:6d}  {rr['eps_dir_med']:10.4f}  {rr['eps_norm_med']:10.4f}  "
              f"{rho_v:8.4f}  {rr['fallback_rate']:.6f}")


# ---------------------------------------------------------------------------
# Figure: LiRA scatter
# ---------------------------------------------------------------------------

def figure_lira_scatter(setting_id="S2", lira_id="L1", seed=0, out_dir=OUT_DIR):
    plt = _plt()
    if plt is None: return

    lira = load_lira(lira_id)
    cert = load_cert(setting_id, seed)
    run  = load_run_arrays(setting_id, seed)
    if not lira or cert is None: return

    D_mem = lira["lira_scores_members"].astype(np.float64)
    member_local_idx = run.get("lira_member_local_idx")
    tier_labels = run.get("tier_labels")
    if member_local_idx is None: return

    n = min(len(D_mem), len(member_local_idx))
    eps_dir  = cert["eps_dir"][member_local_idx[:n]]
    eps_norm = cert["eps_norm"][member_local_idx[:n]]
    D_use    = D_mem[:n]

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (y_vals, ylabel) in zip(axes, [
        (eps_dir,  "ε^dir (Nyström)"),
        (eps_norm, "ε^norm"),
    ]):
        colors = np.zeros(n)
        if tier_labels is not None:
            colors = tier_labels[member_local_idx[:n]].astype(float)
        sc = ax.scatter(D_use, y_vals, c=colors, cmap="RdYlGn_r",
                        s=8, alpha=0.4, vmin=0, vmax=2)
        # Add binned means (LOESS-like)
        n_bins = 15
        bins = np.percentile(D_use, np.linspace(0, 100, n_bins+1))
        bin_x, bin_y = [], []
        for i in range(n_bins):
            mask = (D_use >= bins[i]) & (D_use < bins[i+1])
            if mask.sum() > 5:
                bin_x.append(D_use[mask].mean())
                bin_y.append(y_vals[mask].mean())
        if bin_x:
            ax.plot(bin_x, bin_y, "r-", lw=2.5, label="Binned mean")
        ax.set_xlabel("D_LiRA (distinguishability)"); ax.set_ylabel(ylabel)
        ax.set_title(f"{setting_id}/seed={seed}: {ylabel}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="tier")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"p18_{setting_id}_seed{seed}_{lira_id}_lira_scatter.png")
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  [fig] {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global CERT_DIR, OUT_DIR
    parser = argparse.ArgumentParser(description="Phase 18 Analysis")
    parser.add_argument("--table",   type=str, default=None,
                        choices=["A","B","C","D","E","F","G"])
    parser.add_argument("--figure",  type=str, default=None,
                        choices=["lira_scatter"])
    parser.add_argument("--all",     action="store_true")
    parser.add_argument("--setting", type=str, default="S2")
    parser.add_argument("--lira_id", type=str, default="L1")
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--cert_dir",type=str, default=CERT_DIR)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    CERT_DIR = args.cert_dir; OUT_DIR = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    if args.all or args.table == "A":
        table_A(args.setting, args.seed)
    if args.all or args.table == "B":
        table_B(args.setting, args.seed)
    if args.all or args.table == "C":
        table_C(args.setting, args.lira_id, args.seed)
    if args.all or args.table == "D":
        table_D(args.setting, args.lira_id, args.seed)
    if args.all or args.table == "E":
        table_E(args.setting, args.lira_id, args.seed)
    if args.all or args.table == "F":
        table_F(args.setting)
    if args.all or args.table == "G":
        table_G(args.setting, args.seed, args.lira_id)
    if args.all or args.figure == "lira_scatter":
        figure_lira_scatter(args.setting, args.lira_id, args.seed, args.out_dir)

    if not (args.all or args.table or args.figure):
        parser.print_help()


if __name__ == "__main__":
    main()
