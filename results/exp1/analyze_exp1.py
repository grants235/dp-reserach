"""
analyze_exp1.py — Analyze Experiment 1 results against spec.tex expectations.

Usage:
    python3 analyze_exp1.py [--results_dir results/exp1]

Checks:
  1. SUCCESS CRITERION  (spec §5.4): Tier 0 mean norm ≥ 3× smaller than Tier 2
     at convergence on CIFAR-10-LT (IR=50), Strategy A.
  2. TAB 1              Per-tier mean and p95 clipped gradient norms at convergence.
  3. TRAJECTORY         Per-tier mean norm across checkpoint epochs (Fig 2 data).
  4. DISTRIBUTION       Norm distribution statistics by tier (Fig 1 data).
  5. TRAINING           Test accuracy, achieved ε, σ for each run.
"""

import os
import sys
import pickle
import argparse
import numpy as np

RESULTS_DIR = "results/exp1"
K = 3
CHECKPOINT_EPOCHS = [1, 5, 10, 25, 50, 75, 100]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_all(results_dir):
    runs = {}
    for entry in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, entry, "results.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                r = pickle.load(f)
            runs[entry] = r
    return runs


def tier_stats(norms, tiers):
    """Return {tier: {mean, p95, n}} for each tier."""
    out = {}
    for k in range(K):
        mask = (tiers == k)
        t = norms[mask]
        if len(t) == 0:
            out[k] = {"mean": float("nan"), "p95": float("nan"), "n": 0}
        else:
            out[k] = {"mean": float(t.mean()), "p95": float(np.percentile(t, 95)), "n": int(mask.sum())}
    return out


def final_epoch(ckpt):
    return max(e for e in ckpt.keys() if isinstance(e, int))


# ── print helpers ─────────────────────────────────────────────────────────────

SEP = "=" * 72

def header(s):
    print(f"\n{SEP}")
    print(f"  {s}")
    print(SEP)

# ── main analysis ─────────────────────────────────────────────────────────────

def main(results_dir):
    runs = load_all(results_dir)
    if not runs:
        print(f"No results found in {results_dir}")
        sys.exit(1)

    print(f"Loaded {len(runs)} runs: {sorted(runs.keys())}")

    # ── 1. Training overview ──────────────────────────────────────────────────
    header("TRAINING OVERVIEW  (test accuracy, achieved ε, σ)")
    print(f"{'Run':<35} {'σ':>7} {'ε_target':>9} {'ε_achieved':>11} {'test_acc':>9}")
    print("-" * 75)
    for tag, r in sorted(runs.items()):
        eps_achieved = r.get("epsilon", float("nan"))
        print(f"{tag:<35} {r['sigma']:>7.4f} {'3.0':>9} {eps_achieved:>11.3f} {r['test_acc']:>9.4f}")

    # ── 2. Table 1: per-tier norms at convergence ─────────────────────────────
    header("TAB 1 — Per-tier clipped gradient norms at convergence (mean ± std, p95)")
    print("Averaged over seeds within each (dataset, IR, strategy) group.\n")

    # Aggregate across seeds
    agg = {}  # key: (dataset, IR, strategy) → {tier: [mean_list, p95_list]}
    for tag, r in runs.items():
        ckpt = r["checkpoint_data"]
        ep = final_epoch(ckpt)
        cn = ckpt[ep]["clipped_norms"]
        for strat, tiers in [("A", r["tiers_A"]), ("B", r["tiers_B"])]:
            key = (r["dataset"], r["imbalance_ratio"], strat)
            if key not in agg:
                agg[key] = {k: {"means": [], "p95s": []} for k in range(K)}
            for k in range(K):
                mask = (tiers == k)
                t = cn[mask]
                if len(t):
                    agg[key][k]["means"].append(float(t.mean()))
                    agg[key][k]["p95s"].append(float(np.percentile(t, 95)))

    for (dset, ir, strat), tier_data in sorted(agg.items()):
        print(f"  {dset} IR={ir:.0f}  Strategy {strat}")
        print(f"  {'Tier':>6} {'mean':>8} {'p95':>8} {'n_seeds':>8}")
        for k in range(K):
            m_list = tier_data[k]["means"]
            p_list = tier_data[k]["p95s"]
            if m_list:
                print(f"  Tier {k}  {np.mean(m_list):>8.4f}  {np.mean(p_list):>8.4f}  {len(m_list):>8}")
            else:
                print(f"  Tier {k}  {'N/A':>8}  {'N/A':>8}")
        print()

    # ── 3. Success criterion ──────────────────────────────────────────────────
    header("SUCCESS CRITERION  (spec §5.4)")
    print("Tier 0 (head) mean norm must be ≥ 3× SMALLER than Tier 2 (tail)")
    print("at convergence on CIFAR-10-LT (IR=50), Strategy A.\n")

    lt_runs = {tag: r for tag, r in runs.items()
               if r["dataset"] == "cifar10" and r["imbalance_ratio"] == 50.0}

    if not lt_runs:
        print("  WARNING: No CIFAR-10-LT (IR=50) results found.")
    else:
        ratios = []
        for tag, r in sorted(lt_runs.items()):
            ep = final_epoch(r["checkpoint_data"])
            cn = r["checkpoint_data"][ep]["clipped_norms"]
            tiers = r["tiers_A"]
            stats = tier_stats(cn, tiers)
            m0, m2 = stats[0]["mean"], stats[2]["mean"]
            ratio = m2 / m0 if m0 > 0 else float("nan")
            ratios.append(ratio)
            status = "PASS ✓" if ratio >= 3.0 else "FAIL ✗"
            print(f"  {tag:<35}  Tier0={m0:.4f}  Tier2={m2:.4f}  ratio={ratio:.2f}  [{status}]")

        mean_ratio = np.nanmean(ratios)
        overall = "PASS ✓" if mean_ratio >= 3.0 else "FAIL ✗"
        print(f"\n  Mean ratio (across seeds): {mean_ratio:.2f}  [{overall}]")
        print(f"  Required: ≥ 3.0")

    # ── 4. Balanced CIFAR-10 — smaller gap expected ───────────────────────────
    header("GRADIENT STRATIFICATION — Balanced CIFAR-10")
    print("Spec says: gap may be smaller on balanced data but should be significant.\n")

    bal_runs = {tag: r for tag, r in runs.items()
                if r["dataset"] == "cifar10" and r["imbalance_ratio"] == 1.0}

    for tag, r in sorted(bal_runs.items()):
        ep = final_epoch(r["checkpoint_data"])
        cn = r["checkpoint_data"][ep]["clipped_norms"]
        for strat, tiers in [("A", r["tiers_A"]), ("B", r["tiers_B"])]:
            stats = tier_stats(cn, tiers)
            m0, m2 = stats[0]["mean"], stats[2]["mean"]
            ratio = m2 / m0 if m0 > 0 else float("nan")
            print(f"  {tag}  Strategy {strat}  Tier0={m0:.4f}  Tier2={m2:.4f}  ratio={ratio:.2f}")

    # ── 5. Trajectory: per-tier mean norm across epochs ───────────────────────
    header("TRAJECTORY — Per-tier mean clipped norm across checkpoint epochs (Fig 2 data)")
    print("Averaged over seeds for CIFAR-10-LT IR=50, Strategy A.\n")

    epoch_data = {ep: {k: [] for k in range(K)} for ep in CHECKPOINT_EPOCHS}
    for tag, r in lt_runs.items():
        for ep in CHECKPOINT_EPOCHS:
            if ep not in r["checkpoint_data"]:
                continue
            cn = r["checkpoint_data"][ep]["clipped_norms"]
            tiers = r["tiers_A"]
            for k in range(K):
                mask = (tiers == k)
                if mask.any():
                    epoch_data[ep][k].append(cn[mask].mean())

    print(f"  {'Epoch':>6}", end="")
    for k in range(K):
        print(f"  {'Tier'+str(k)+' mean':>12}", end="")
    print(f"  {'Ratio T2/T0':>12}")
    print("  " + "-" * 60)
    for ep in CHECKPOINT_EPOCHS:
        means = [np.mean(epoch_data[ep][k]) if epoch_data[ep][k] else float("nan")
                 for k in range(K)]
        ratio = means[2] / means[0] if means[0] > 0 else float("nan")
        print(f"  {ep:>6}", end="")
        for m in means:
            print(f"  {m:>12.4f}", end="")
        print(f"  {ratio:>12.2f}")

    # ── 6. CIFAR-100 stratification ───────────────────────────────────────────
    header("GRADIENT STRATIFICATION — CIFAR-100")

    c100_runs = {tag: r for tag, r in runs.items() if r["dataset"] == "cifar100"}
    for tag, r in sorted(c100_runs.items()):
        ep = final_epoch(r["checkpoint_data"])
        cn = r["checkpoint_data"][ep]["clipped_norms"]
        for strat, tiers in [("A", r["tiers_A"]), ("B", r["tiers_B"])]:
            stats = tier_stats(cn, tiers)
            m0, m2 = stats[0]["mean"], stats[2]["mean"]
            ratio = m2 / m0 if m0 > 0 else float("nan")
            ns = [stats[k]["n"] for k in range(K)]
            print(f"  {tag}  Strategy {strat}  "
                  f"Tier0={m0:.4f}(n={ns[0]})  "
                  f"Tier2={m2:.4f}(n={ns[2]})  ratio={ratio:.2f}")

    # ── 7. Tier sizes ─────────────────────────────────────────────────────────
    header("TIER SIZES per run")
    for tag, r in sorted(runs.items()):
        szA = r.get("tier_sizes_A", [])
        szB = r.get("tier_sizes_B", [])
        print(f"  {tag:<35}  A={szA}  B={szB}")

    # ── 8. Convergence quality ────────────────────────────────────────────────
    header("CONVERGENCE — Per-tier loss and accuracy at epoch 100")
    for tag, r in sorted(runs.items()):
        conv = r["checkpoint_data"].get("convergence", {})
        if not conv:
            continue
        labels = conv["labels"]
        correct = conv["correctly_classified"]
        losses = conv["losses"]
        tiers = r["tiers_A"]
        print(f"\n  {tag}  (overall acc={correct.mean():.3f})")
        for k in range(K):
            mask = (tiers == k)
            if not mask.any():
                continue
            acc_k = correct[mask].mean()
            loss_k = losses[mask].mean()
            print(f"    Tier {k}: acc={acc_k:.3f}  loss={loss_k:.3f}  n={mask.sum()}")

    print(f"\n{SEP}")
    print("Analysis complete.")
    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    args = parser.parse_args()
    main(args.results_dir)
