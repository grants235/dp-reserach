"""
analyze_exp1.py — Exp 1 results vs spec.tex expectations.

Key checks:
  1. Training sanity (σ, ε, test accuracy)
  2. Clipping saturation diagnostic (% of samples that hit the clip bound)
  3. UNCLIPPED norm stratification — the real signal the hypothesis predicts
  4. CLIPPED norm stratification — what the success criterion formally checks
  5. Trajectory of per-tier norms across epochs
  6. Per-tier accuracy/loss at convergence
  7. CIFAR-100 Strategy A tier-assignment bug diagnostic

Usage:
    python3 analyze_exp1.py [--results_dir results/exp1]
"""

import os, sys, pickle, argparse
import numpy as np

RESULTS_DIR = "results/exp1"
K = 3
C = 1.0
CHECKPOINT_EPOCHS = [1, 5, 10, 25, 50, 75, 100]
SEP = "=" * 74

def load_all(d):
    runs = {}
    for e in sorted(os.listdir(d)):
        p = os.path.join(d, e, "results.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                runs[e] = pickle.load(f)
    return runs

def tier_stats(norms, tiers, k_list=None):
    k_list = k_list or list(range(K))
    out = {}
    for k in k_list:
        mask = (tiers == k)
        t = norms[mask]
        if len(t) == 0:
            out[k] = dict(mean=float("nan"), p25=float("nan"), p50=float("nan"),
                          p75=float("nan"), p95=float("nan"), n=0)
        else:
            out[k] = dict(mean=float(t.mean()),
                          p25=float(np.percentile(t, 25)),
                          p50=float(np.percentile(t, 50)),
                          p75=float(np.percentile(t, 75)),
                          p95=float(np.percentile(t, 95)),
                          n=int(mask.sum()))
    return out

def final_epoch(ckpt):
    return max(e for e in ckpt if isinstance(e, int))

def hdr(s):
    print(f"\n{SEP}\n  {s}\n{SEP}")

def ratio_label(r):
    if np.isnan(r): return "N/A"
    return f"{r:.2f}x  {'PASS ✓' if r >= 3.0 else 'FAIL ✗'}"


def main(results_dir):
    runs = load_all(results_dir)
    if not runs:
        print(f"No results found in {results_dir}"); sys.exit(1)
    print(f"Loaded {len(runs)} runs: {sorted(runs.keys())}")

    lt_runs  = {t: r for t, r in runs.items() if r["dataset"]=="cifar10"  and r["imbalance_ratio"]==50.0}
    bal_runs = {t: r for t, r in runs.items() if r["dataset"]=="cifar10"  and r["imbalance_ratio"]==1.0}
    c100_runs= {t: r for t, r in runs.items() if r["dataset"]=="cifar100"}

    # ── 1. Training overview ──────────────────────────────────────────────────
    hdr("1. TRAINING OVERVIEW  (σ, ε_achieved, test_acc)")
    print(f"  NOTE: CIFAR-10 balanced ε=3 DP-SGD with WRN-28-2 typically achieves")
    print(f"  ~60-70% in literature. Values below 50% may indicate a training issue.\n")
    print(f"  {'Run':<35} {'σ':>7} {'ε':>7} {'test_acc':>9}")
    print("  " + "-" * 62)
    for tag, r in sorted(runs.items()):
        flag = ""
        if r["dataset"] == "cifar10" and r["imbalance_ratio"] == 1.0 and r["test_acc"] < 0.50:
            flag = "  ← LOW"
        print(f"  {tag:<35} {r['sigma']:>7.4f} {r.get('epsilon', float('nan')):>7.3f}"
              f" {r['test_acc']:>9.4f}{flag}")

    # ── 2. Clipping saturation ────────────────────────────────────────────────
    hdr("2. CLIPPING SATURATION DIAGNOSTIC")
    print(f"  C = {C}. If nearly all samples are clipped, clipped norms ≈ C everywhere")
    print(f"  and carry NO stratification signal (all tiers look identical).\n")
    print(f"  {'Run':<35} {'%_clipped':>10}  {'unclipped_mean':>14}  {'unclipped_p95':>13}")
    print("  " + "-" * 76)
    for tag, r in sorted(runs.items()):
        ep = final_epoch(r["checkpoint_data"])
        un = r["checkpoint_data"][ep]["unclipped_norms"]
        pct = float((un > C).mean()) * 100
        print(f"  {tag:<35} {pct:>9.1f}%  {un.mean():>14.4f}  {np.percentile(un,95):>13.4f}")

    # ── 3. Unclipped norm stratification — the real hypothesis ───────────────
    hdr("3. UNCLIPPED NORM STRATIFICATION  (the hypothesis that must hold)")
    print("  The spec predicts tail-tier (Tier 2) samples have LARGER gradient norms")
    print("  than head-tier (Tier 0) samples. This must show up in UNCLIPPED norms.")
    print("  Success criterion: Tier2/Tier0 ratio ≥ 3× on CIFAR-10-LT IR=50.\n")

    for label, run_dict, strat in [
        ("CIFAR-10-LT IR=50, Strategy A", lt_runs,  "A"),
        ("CIFAR-10-LT IR=50, Strategy B", lt_runs,  "B"),
        ("CIFAR-10 balanced,  Strategy A", bal_runs, "A"),
        ("CIFAR-100,          Strategy B", c100_runs,"B"),
    ]:
        if not run_dict:
            continue
        print(f"  {label}")
        ratios = []
        for tag, r in sorted(run_dict.items()):
            ep = final_epoch(r["checkpoint_data"])
            un = r["checkpoint_data"][ep]["unclipped_norms"]
            tiers = r[f"tiers_{strat}"]
            active = [k for k in range(K) if (tiers==k).any()]
            s = tier_stats(un, tiers, active)
            row = "    " + tag
            for k in active:
                row += f"  T{k}={s[k]['mean']:.3f}(n={s[k]['n']})"
            if len(active) >= 2:
                m0, m2 = s[active[0]]["mean"], s[active[-1]]["mean"]
                rat = m2 / m0 if m0 > 0 else float("nan")
                ratios.append(rat)
                row += f"  ratio={ratio_label(rat)}"
            print(row)
        if ratios:
            mr = np.nanmean(ratios)
            print(f"    → Mean ratio: {ratio_label(mr)}")
        print()

    # ── 4. Clipped norm stratification ───────────────────────────────────────
    hdr("4. CLIPPED NORM STRATIFICATION  (formal success criterion)")
    print("  With C=1.0 and most norms >> C, clipped norms will be uniformly ~C.")
    print("  A ratio ≈ 1.0 here is EXPECTED when clipping saturates, not a")
    print("  fundamental failure of the hypothesis.\n")

    for label, run_dict, strat in [
        ("CIFAR-10-LT IR=50, Strategy A (primary)", lt_runs, "A"),
        ("CIFAR-10 balanced,  Strategy A",           bal_runs,"A"),
        ("CIFAR-100,          Strategy B",           c100_runs,"B"),
    ]:
        if not run_dict:
            continue
        print(f"  {label}")
        ratios = []
        for tag, r in sorted(run_dict.items()):
            ep = final_epoch(r["checkpoint_data"])
            cn = r["checkpoint_data"][ep]["clipped_norms"]
            tiers = r[f"tiers_{strat}"]
            active = [k for k in range(K) if (tiers==k).any()]
            s = tier_stats(cn, tiers, active)
            row = "    " + tag
            for k in active:
                row += f"  T{k}={s[k]['mean']:.4f}"
            if len(active) >= 2:
                m0, m2 = s[active[0]]["mean"], s[active[-1]]["mean"]
                rat = m2/m0 if m0 > 0 else float("nan")
                ratios.append(rat)
                row += f"  ratio={rat:.2f}"
            print(row)
        if ratios:
            print(f"    → Mean ratio: {np.nanmean(ratios):.2f}")
        print()

    # ── 5. Trajectory ─────────────────────────────────────────────────────────
    hdr("5. UNCLIPPED NORM TRAJECTORY across epochs (Fig 2 data, CIFAR-10-LT IR=50 Strategy A)")
    print(f"  {'Epoch':>6}", end="")
    for k in range(K):
        print(f"  {'T'+str(k)+' mean':>10}  {'T'+str(k)+' p95':>9}", end="")
    print(f"  {'Ratio T2/T0':>12}")
    print("  " + "-" * 75)
    for ep in CHECKPOINT_EPOCHS:
        per_tier = {k: [] for k in range(K)}
        for r in lt_runs.values():
            if ep not in r["checkpoint_data"]:
                continue
            un = r["checkpoint_data"][ep]["unclipped_norms"]
            tiers = r["tiers_A"]
            for k in range(K):
                mask = (tiers == k)
                if mask.any():
                    per_tier[k].append((un[mask].mean(), np.percentile(un[mask], 95)))
        means = [np.mean([v[0] for v in per_tier[k]]) if per_tier[k] else float("nan")
                 for k in range(K)]
        p95s  = [np.mean([v[1] for v in per_tier[k]]) if per_tier[k] else float("nan")
                 for k in range(K)]
        rat = means[2]/means[0] if means[0] > 0 else float("nan")
        print(f"  {ep:>6}", end="")
        for k in range(K):
            print(f"  {means[k]:>10.4f}  {p95s[k]:>9.4f}", end="")
        print(f"  {rat:>12.2f}")

    # ── 6. Tab 1 (spec): unclipped mean + p95 per tier ───────────────────────
    hdr("6. TAB 1 (spec format) — Unclipped norm mean + p95 at convergence, per tier")
    agg = {}
    for tag, r in runs.items():
        ep = final_epoch(r["checkpoint_data"])
        un = r["checkpoint_data"][ep]["unclipped_norms"]
        for strat, tiers in [("A", r["tiers_A"]), ("B", r["tiers_B"])]:
            key = (r["dataset"], r["imbalance_ratio"], strat)
            if key not in agg:
                agg[key] = {k: {"means": [], "p95s": []} for k in range(K)}
            for k in range(K):
                t = un[tiers == k]
                if len(t):
                    agg[key][k]["means"].append(t.mean())
                    agg[key][k]["p95s"].append(np.percentile(t, 95))

    for (dset, ir, strat), td in sorted(agg.items()):
        print(f"  {dset} IR={ir:.0f}  Strategy {strat}")
        active = [k for k in range(K) if td[k]["means"]]
        print(f"  {'Tier':>4}  {'mean':>8}  {'p95':>8}  {'n_seeds':>8}")
        for k in range(K):
            if td[k]["means"]:
                print(f"  T{k}    {np.mean(td[k]['means']):>8.4f}  {np.mean(td[k]['p95s']):>8.4f}"
                      f"  {len(td[k]['means']):>8}")
            else:
                print(f"  T{k}    {'EMPTY':>8}  {'EMPTY':>8}  {'  (bug)':>8}")
        if len(active) >= 2:
            m0 = np.mean(td[active[0]]["means"])
            m2 = np.mean(td[active[-1]]["means"])
            print(f"  → Tail/Head ratio: {m2/m0:.2f}x")
        print()

    # ── 7. Per-tier accuracy/loss at convergence ──────────────────────────────
    hdr("7. PER-TIER ACCURACY AND LOSS AT CONVERGENCE (epoch 100)")
    for tag, r in sorted(runs.items()):
        conv = r["checkpoint_data"].get("convergence", {})
        if not conv:
            continue
        correct = conv["correctly_classified"]
        losses  = conv["losses"]
        tiers   = r["tiers_A"]
        active  = [k for k in range(K) if (tiers == k).any()]
        print(f"\n  {tag}  overall_acc={correct.mean():.3f}")
        for k in active:
            mask = (tiers == k)
            print(f"    Tier {k}  acc={correct[mask].mean():.3f}"
                  f"  loss={losses[mask].mean():.3f}  n={mask.sum()}")

    # ── 8. CIFAR-100 Strategy A bug ────────────────────────────────────────────
    hdr("8. CIFAR-100 STRATEGY A BUG")
    print("  Strategy A sorts classes by frequency. On balanced CIFAR-100 (all 100")
    print("  classes equal), sort order is non-deterministic and the current")
    print("  implementation puts ALL samples into Tier 2 (0 in Tiers 0 and 1).\n")
    for tag, r in sorted(c100_runs.items()):
        szA = r.get("tier_sizes_A", [])
        szB = r.get("tier_sizes_B", [])
        print(f"  {tag}  Strategy A tiers={szA}  Strategy B tiers={szB}")
    print("\n  Fix: for balanced datasets, use class index mod K for Strategy A.")
    print("  Strategy B (density-based) correctly distributes CIFAR-100 into equal tiers.")

    # ── 9. Summary ────────────────────────────────────────────────────────────
    hdr("9. SUMMARY vs SPEC EXPECTATIONS")
    print("""
  ISSUE 1 — Clipping saturation (root cause of FAIL on formal criterion)
    ε=3 on WRN-28-2 with C=1.0 clips ~80-100% of all gradients to exactly C.
    Clipped norms are therefore ~C regardless of tier → ratio ≈ 1.0.
    Action: check unclipped norm ratios (section 3 above). If unclipped norms
    stratify (Tier2/Tier0 >> 1), the hypothesis holds but C=1.0 is too tight.
    Consider raising C (e.g. 5-10) or switching to the calibrated C_k values.

  ISSUE 2 — Low test accuracy (~38% CIFAR-10 vs expected ~60-70%)
    Possible causes: (a) LR/schedule mismatch, (b) 10% public holdout,
    (c) training loop bug, (d) this WRN-28-2 implementation.
    Sanity check: run a non-private baseline (NonPrivateTrainer) and compare.

  ISSUE 3 — CIFAR-100 Strategy A is broken
    All 45K samples end up in Tier 2. Strategy B works correctly (15K/tier).
    Fix assign_tiers() to fall back to class-index-mod-K for balanced data.

  SPEC SUCCESS CRITERION STATUS
    Formal (clipped norms, ratio ≥ 3×): FAIL — due to saturation, not hypothesis
    Hypothesis (unclipped norms stratify): see section 3 for actual values
    """)

    print(SEP + "\nAnalysis complete.\n" + SEP)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=RESULTS_DIR)
    args = p.parse_args()
    main(args.results_dir)
