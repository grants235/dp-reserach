#!/usr/bin/env python3
"""
exp_p19_appendix.py — retrieve fill-in values for all appendix TBDs.

Reads only from ./runs/p19, ./certs/p19, ./lira/p19, ./results/p19.
No training, certification, or LiRA-scoring calls are made.

Usage:
  python experiments/exp_p19_appendix.py              # all sections
  python experiments/exp_p19_appendix.py --settings   # S2/S3 operating points
  python experiments/exp_p19_appendix.py --tiers      # tier class cutoffs
  python experiments/exp_p19_appendix.py --accounting # step freq + Renyi grid
  python experiments/exp_p19_appendix.py --gaussianity
  python experiments/exp_p19_appendix.py --lira
"""

import os, sys, json, math, argparse
import numpy as np

RUNS_DIR    = "./runs/p19"
CERT_DIR    = "./certs/p19"
LIRA_DIR    = "./lira/p19"
RESULTS_DIR = "./results/p19"

ALPHA_GRID = np.concatenate([
    np.arange(1.5,  10,    0.5),
    np.arange(10,   100,   2.0),
    np.arange(100,  1000,  20.0),
    np.arange(1000, 5001,  100.0),
]).astype(np.float64)

TIER_NAMES = {0: "head", 1: "mid", 2: "tail"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta(run_id, seed=0):
    p = os.path.join(RUNS_DIR, run_id, f"seed_{seed}", "metadata.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _npy(run_id, seed, name):
    p = os.path.join(RUNS_DIR, run_id, f"seed_{seed}", f"{name}.npy")
    return np.load(p) if os.path.exists(p) else None


def _get(d, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _div(header):
    print(f"\n{'='*70}")
    print(f"  {header}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# TBD 1  §Settings — S2 (F2) and S3 (F5) operating points
# ---------------------------------------------------------------------------

def sec_settings():
    """Report σ, q, T, qn, n for every setting (S1/S2/S3)."""
    _div("TBD §Settings — operating points for S1, S2, S3")

    for label, run_id in [("S1", "F1"), ("S2", "F2"), ("S3", "F5")]:
        meta = _meta(run_id, 0)
        if meta is None:
            print(f"\n  [{run_id}] metadata.json not found — [missing]")
            continue

        sigma = float(_get(meta, "sigma", "noise_multiplier", default=float("nan")))
        q     = float(_get(meta, "q", "sampling_rate", default=float("nan")))
        T     = int(  _get(meta, "T_train", "T_steps", "num_steps", default=0))
        delta = float(_get(meta, "delta", default=1e-5))
        n_eps = float(_get(meta, "epsilon", "eps_target", "target_epsilon", default=float("nan")))

        # n_train: try multiple keys, fall back to len(tier_labels)
        n = int(_get(meta, "n_train", "n_private", "n", default=0))
        if n == 0:
            tl = _npy(run_id, 0, "tier_labels")
            if tl is not None:
                n = len(tl)

        n_epochs = int(_get(meta, "n_epochs", "T_epochs", "num_epochs", default=0))
        spe      = T // n_epochs if n_epochs > 0 and T > 0 else None

        print(f"\n  {label} ({run_id}) seed 0:")
        print(f"    σ (noise multiplier)  = {sigma:.4f}")
        print(f"    q (sampling rate)     = {q:.4f}")
        print(f"    T (training steps)    = {T}")
        print(f"    n (private examples)  = {n}")
        print(f"    qn (expected batch)   = {q * n:.1f}" if n > 0 else "    qn = n/a")
        print(f"    δ                     = {delta:.1e}")
        if not math.isnan(n_eps):
            print(f"    target ε              = {n_eps}")
        if n_epochs > 0:
            print(f"    n_epochs              = {n_epochs}")
        if spe is not None:
            print(f"    steps / epoch         = {spe}")

        # Print any extra keys that might be useful
        extra_keys = [k for k in meta
                      if k not in {"sigma","noise_multiplier","q","sampling_rate",
                                   "T_train","T_steps","num_steps","delta","epsilon",
                                   "eps_target","target_epsilon","n_train","n_private","n",
                                   "n_epochs","T_epochs","num_epochs"}]
        if extra_keys:
            print(f"    (other metadata keys: {extra_keys})")

    print(f"\n  Appendix sentence to fill:")
    print(f"    For S2: σ=<>, q=<>, T=<>, qn=<> over n=<> private examples.")
    print(f"    For S3: σ=<>, q=<>, T=<>, qn=<> over n=<> private examples.")


# ---------------------------------------------------------------------------
# TBD 2  §Data and tiers — class-count cutoffs
# ---------------------------------------------------------------------------

def sec_tiers():
    """Report per-class counts and the count thresholds that define tiers."""
    _div("TBD §Data and tiers — class-count cutoffs for head / mid / tail")

    tier_labels = _npy("F1", 0, "tier_labels")
    labels      = _npy("F1", 0, "labels")

    if tier_labels is None:
        print("  [missing] tier_labels.npy for F1 seed 0")
        return

    bc = np.bincount(tier_labels.astype(int), minlength=3)
    print(f"\n  Private-set tier populations (F1 seed 0):")
    for t_id, t_name in TIER_NAMES.items():
        print(f"    {t_name}: {bc[t_id]} examples")
    print(f"    total:  {bc.sum()}")

    if labels is None:
        print("\n  [missing] labels.npy — cannot report per-class breakdown")
        return

    n_classes = int(labels.max()) + 1
    class_info = {}   # class_id → (count, tier_id)
    for c in range(n_classes):
        mask   = (labels == c)
        n_c    = int(mask.sum())
        tiers_c = tier_labels[mask]
        # Every member of a class should be in the same tier (strategy A = class frequency)
        tier_c  = int(np.bincount(tiers_c.astype(int)).argmax()) if n_c > 0 else -1
        class_info[c] = (n_c, tier_c)

    print(f"\n  Per-class sample counts and tier assignments:")
    print(f"    {'Class':>5}  {'Count':>7}  {'Tier':>5}")
    for c in range(n_classes):
        n_c, t_id = class_info[c]
        print(f"    {c:5d}  {n_c:7d}  {TIER_NAMES.get(t_id,'?'):>5}")

    # Report the count thresholds at tier boundaries
    print(f"\n  Tier boundary thresholds (by class count):")
    # Sort classes by count descending
    sorted_classes = sorted(range(n_classes), key=lambda c: class_info[c][0], reverse=True)
    prev_tier = None
    for c in sorted_classes:
        n_c, t_id = class_info[c]
        if t_id != prev_tier:
            print(f"    → {TIER_NAMES.get(t_id,'?')} tier starts at count = {n_c} "
                  f"(class {c})")
            prev_tier = t_id

    # Summary ranges per tier
    print(f"\n  Count range per tier:")
    for t_id, t_name in TIER_NAMES.items():
        t_classes = [c for c in range(n_classes) if class_info[c][1] == t_id]
        if not t_classes:
            continue
        counts = [class_info[c][0] for c in t_classes]
        print(f"    {t_name}: classes {t_classes}, counts [{min(counts)} – {max(counts)}]")


# ---------------------------------------------------------------------------
# TBD 3  §Accounting — step frequency and Rényi order grid
# ---------------------------------------------------------------------------

def sec_accounting():
    """Report accounting-step frequency and Rényi order grid from metadata."""
    _div("TBD §Accounting — step frequency and Rényi order grid")

    meta = _meta("F1", 0)
    if meta is None:
        print("  [missing] F1 seed 0 metadata.json")
        return

    T        = int(_get(meta, "T_train", "T_steps", "num_steps", default=0))
    n_epochs = int(_get(meta, "n_epochs", "T_epochs", "num_epochs", default=0))

    print(f"\n  F1 seed 0 base values: T={T}, n_epochs={n_epochs}")

    # Accounting frequency
    freq = _get(meta, "accounting_freq", "cert_freq", "log_freq",
                "acct_freq", "accounting_interval")
    if freq is not None:
        print(f"\n  Accounting frequency (from metadata): every {freq} steps")
    else:
        print(f"\n  Accounting frequency not explicit in metadata.")
        if T > 0 and n_epochs > 0:
            spe = T // n_epochs
            print(f"    Inferred {spe} steps/epoch over {n_epochs} epochs.")
            print(f"    If per-epoch: {n_epochs} accounting steps; "
                  f"if per-step: {T} accounting steps.")
        # Check how many rows are in Y_projections (= number of saved accounting steps)
        yp_path = os.path.join(RUNS_DIR, "F1", "seed_0", "Y_projections.npy")
        if os.path.exists(yp_path):
            yp_shape = np.load(yp_path, mmap_mode="r").shape   # (n, T_acct, r)
            T_acct = yp_shape[1]
            print(f"    Y_projections.npy shape: {yp_shape}  → {T_acct} accounting steps saved")
            if T > 0:
                ratio = T / T_acct
                print(f"    Ratio T/T_acct = {ratio:.2f} — "
                      f"{'per-step' if abs(ratio-1)<0.01 else f'every ~{ratio:.0f} steps'}")

    # Nystrom rank
    rank = _get(meta, "rank", "nystrom_rank", "r", "nyström_rank")
    if rank is not None:
        print(f"\n  Nystrom rank (r): {rank}")
    else:
        # Check cert summary
        summ_path = os.path.join(CERT_DIR, "p19_F1_seed0_summary.json")
        if os.path.exists(summ_path):
            with open(summ_path) as f:
                summ = json.load(f)
            rank = _get(summ, "rank", "nystrom_rank", "r")
        print(f"\n  Nystrom rank (r): "
              f"{rank if rank is not None else '[not in metadata or cert summary; default r=100]'}")

    # Rényi order grid
    print(f"\n  ALPHA_GRID (module-level constant, used in both certification and accounting):")
    print(f"    {len(ALPHA_GRID)} orders, range [{ALPHA_GRID.min():.1f}, {ALPHA_GRID.max():.0f}]")
    print(f"    Segments:")
    print(f"      [1.5, 9.5]    step 0.5   → {len(np.arange(1.5, 10, 0.5))} orders")
    print(f"      [10, 98]      step 2.0   → {len(np.arange(10, 100, 2.0))} orders")
    print(f"      [100, 980]    step 20.0  → {len(np.arange(100, 1000, 20.0))} orders")
    print(f"      [1000, 5000]  step 100.0 → {len(np.arange(1000, 5001, 100.0))} orders")

    # Check if metadata overrides the grid
    alpha_meta = _get(meta, "alpha_grid", "rdp_orders", "renyi_orders")
    if alpha_meta is not None:
        print(f"\n  alpha_grid from metadata: {alpha_meta}")


# ---------------------------------------------------------------------------
# §Gaussianity — confirm step indices and test configuration
# ---------------------------------------------------------------------------

def sec_gaussianity():
    """Confirm the three phase steps and test parameters from saved JSON."""
    _div("§Gaussianity tests — step indices and configuration")

    meta = _meta("F1", 0)
    T = 0
    if meta is not None:
        T = int(_get(meta, "T_train", "T_steps", "num_steps", default=0))
    print(f"\n  F1 (S1): T={T} training steps")

    steps_theory = {
        "early":  max(0, round(0.1 * T) - 1),
        "middle": max(0, round(0.5 * T) - 1),
        "late":   max(0, round(0.9 * T) - 1),
    }
    print(f"\n  Theoretical phase steps (0-indexed, from code formula):")
    for phase, idx in steps_theory.items():
        print(f"    {phase:6s}: step {idx:4d}  (1-indexed: step {idx+1})")

    # Load saved Gaussianity validation output
    gv_path = os.path.join(RESULTS_DIR, "gaussian_validation_F1_seed0.json")
    if os.path.exists(gv_path):
        with open(gv_path) as f:
            gv = json.load(f)
        actual_steps = {r["phase"]: r["step"] for r in gv}
        print(f"\n  Actual steps in gaussian_validation_F1_seed0.json:")
        for phase, step in sorted(actual_steps.items()):
            print(f"    {phase:6s}: step {step:4d}  (1-indexed: step {step+1})")

        # Per-phase KS summary (dominant directions)
        print(f"\n  KS summary (dominant projections, per phase):")
        for phase in ["early", "middle", "late"]:
            rows = [r for r in gv if r.get("phase") == phase]
            ks   = [r["ks_median"] for r in rows if r.get("ks_median") is not None]
            if ks:
                print(f"    {phase:6s}: KS median={np.median(ks):.4f}  max={np.max(ks):.4f}  "
                      f"(n_targets={len(rows)})")
    else:
        print(f"\n  [missing] {gv_path} — run --gaussian_validation --run F1 --seed 0 first")

    print(f"\n  Test parameters (hardcoded in gaussian_validation()):")
    print(f"    n_samples (Poisson draws):  5000")
    print(f"    n_targets_per_tier:         3   → 9 targets (3 head, 3 mid, 3 tail)")
    print(f"    n_projections (dominant):   10")

    # Low-variance validation
    gvlv_path = os.path.join(RESULTS_DIR, "gaussian_validation_F1_lowvar_seed0.json")
    if os.path.exists(gvlv_path):
        with open(gvlv_path) as f:
            gvlv = json.load(f)
        print(f"\n  Low-variance validation (gaussian_validation_F1_lowvar_seed0.json):")
        for sub in ["bottom", "tail_orthogonal"]:
            rows = [r for r in gvlv if r.get("subspace") == sub]
            ks   = [r["ks"] for r in rows if r.get("ks") is not None
                    and not math.isnan(r["ks"])]
            if ks:
                print(f"    {sub}: KS median={np.median(ks):.4f}  max={np.max(ks):.4f}  "
                      f"(n={len(ks)})")


# ---------------------------------------------------------------------------
# §Attack — LiRA configuration
# ---------------------------------------------------------------------------

def sec_lira():
    """Report LiRA shadow-model count and target breakdown."""
    _div("§Attack — LiRA configuration")

    # LiRA summary
    for seed in [0, 1, 2]:
        summ_path = os.path.join(LIRA_DIR, "LF1", f"seed_{seed}", "lira_summary.json")
        if not os.path.exists(summ_path):
            continue
        with open(summ_path) as f:
            summ = json.load(f)
        print(f"\n  LF1 seed {seed} lira_summary.json:")
        for k, v in sorted(summ.items()):
            print(f"    {k}: {v}")
        break

    # Member-target tier counts from paper_numbers.json
    pn_path = os.path.join(RESULTS_DIR, "paper_numbers.json")
    if os.path.exists(pn_path):
        with open(pn_path) as f:
            pn = json.load(f)
        nums = pn.get("numbers", {})
        print(f"\n  Member-target tier breakdown (from paper_numbers.json):")
        for k in ["data_member_tier_count_head", "data_member_tier_count_mid",
                  "data_member_tier_count_tail", "data_member_tier_count_total"]:
            v = nums.get(k, "[missing]")
            print(f"    {k}: {v}")
    else:
        # Compute directly
        tier_labels = _npy("F1", 0, "tier_labels")
        ml_path = os.path.join(RUNS_DIR, "F1", "seed_0", "lira_member_local_idx.npy")
        if tier_labels is not None and os.path.exists(ml_path):
            ml = np.load(ml_path)
            mc = np.bincount(tier_labels[ml].astype(int), minlength=3)
            print(f"\n  Member-target tier counts (computed directly):")
            print(f"    head: {mc[0]}  mid: {mc[1]}  tail: {mc[2]}  total: {mc.sum()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Appendix TBD retrieval for p19 runs")
    parser.add_argument("--settings",    action="store_true", help="S2/S3 σ,q,T,qn,n")
    parser.add_argument("--tiers",       action="store_true", help="Tier class-count cutoffs")
    parser.add_argument("--accounting",  action="store_true", help="Acct freq + Rényi grid")
    parser.add_argument("--gaussianity", action="store_true", help="Gaussianity step indices")
    parser.add_argument("--lira",        action="store_true", help="LiRA configuration")
    parser.add_argument("--runs_dir",    default="./runs/p19")
    parser.add_argument("--cert_dir",    default="./certs/p19")
    parser.add_argument("--lira_dir",    default="./lira/p19")
    parser.add_argument("--results_dir", default="./results/p19")
    args = parser.parse_args()

    global RUNS_DIR, CERT_DIR, LIRA_DIR, RESULTS_DIR
    RUNS_DIR    = args.runs_dir
    CERT_DIR    = args.cert_dir
    LIRA_DIR    = args.lira_dir
    RESULTS_DIR = args.results_dir

    explicit = any([args.settings, args.tiers, args.accounting,
                    args.gaussianity, args.lira])

    if not explicit or args.settings:    sec_settings()
    if not explicit or args.tiers:       sec_tiers()
    if not explicit or args.accounting:  sec_accounting()
    if not explicit or args.gaussianity: sec_gaussianity()
    if not explicit or args.lira:        sec_lira()

    print(f"\n[done]")


if __name__ == "__main__":
    main()
