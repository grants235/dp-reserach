#!/usr/bin/env python3
"""
A5: Verify ε_norm = 9.53 > ε_target=8 against the accountants.

Spec §A5: Take σ, q, T, δ from metadata.json; compute the global RDP-conversion ε
for a worst-case example (μ=1/σ every step) and confirm it matches the observed
median ε_norm (~9.53); print the PRV accountant ε at the same σ (should be ≈8).

Usage:
  python experiments/verify_accounting.py --run F1 --seed 0
  python experiments/verify_accounting.py --run F1 --seed 0 --runs_dir ./runs/p19
"""

import os, sys, json, math, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RUNS_DIR = "./runs/p19"
CERT_DIR = "./certs/p19"

# ---------------------------------------------------------------------------
# Import ε_sgm and alpha grid from certify module (to guarantee identical impl)
# ---------------------------------------------------------------------------
from exp_p19_certify import eps_sgm_scalar, ALPHA_GRID


def rdp_to_dp_epsilon(rdp_budget_by_alpha, alpha_grid, delta):
    """
    Convert RDP budget B(α) to (ε,δ)-DP via standard conversion:
      ε = min_α [ B(α) + log(1/δ) / (α−1) ]
    """
    log1d = math.log(1.0 / delta)
    best = float("inf")
    best_alpha = None
    for j, alpha in enumerate(alpha_grid):
        val = rdp_budget_by_alpha[j] + log1d / (alpha - 1.0)
        if val < best:
            best = val
            best_alpha = alpha
    return best, best_alpha


def rdp_worst_case_epsilon(sigma, q, T, delta, alpha_grid=ALPHA_GRID):
    """
    RDP-conversion ε for a worst-case fully-clipped example (μ=1/σ every step).
    B(α) = T × ε_sgm(α, q, 1/σ)
    """
    mu_ceiling = 1.0 / sigma
    B = np.array([T * eps_sgm_scalar(alpha, q, mu_ceiling) for alpha in alpha_grid])
    eps, best_alpha = rdp_to_dp_epsilon(B, alpha_grid, delta)
    return eps, best_alpha, B


def prv_epsilon(eps_target, q, T, delta):
    """
    Re-derive σ via PRV and compute the PRV ε at that σ.
    Returns (sigma, prv_eps).
    """
    try:
        from opacus.accountants.utils import get_noise_multiplier
        from opacus.accountants import create_accountant
        sigma = get_noise_multiplier(
            target_epsilon=float(eps_target), target_delta=float(delta),
            sample_rate=float(q), steps=int(T), accountant="prv")

        # Compute forward PRV ε at this σ
        acct = create_accountant("prv")
        for _ in range(int(T)):
            acct.step(noise_multiplier=float(sigma), sample_rate=float(q))
        prv_eps = acct.get_epsilon(delta=float(delta))
        return float(sigma), float(prv_eps)
    except Exception as e:
        print(f"  [warn] PRV computation failed: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="A5: Verify per-instance ε_norm vs PRV")
    parser.add_argument("--run",      type=str, default="F1", help="Run ID (default F1)")
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--runs_dir", type=str, default=RUNS_DIR)
    parser.add_argument("--cert_dir", type=str, default=CERT_DIR)
    args = parser.parse_args()

    run_dir  = os.path.join(args.runs_dir, args.run, f"seed_{args.seed}")
    meta_p   = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta_p):
        print(f"[A5] metadata.json not found: {meta_p}")
        return

    with open(meta_p) as f:
        meta = json.load(f)

    sigma       = float(meta["sigma"])
    q           = float(meta["q"])
    T           = int(meta["T_train"])
    delta       = float(meta.get("delta", 1e-5))
    eps_target  = float(meta.get("epsilon_target", 8.0))
    accountant  = meta.get("accountant", "prv")

    print(f"\n[A5] Accounting verification for {args.run}/seed_{args.seed}")
    print(f"     sigma={sigma:.6f}  q={q:.6f}  T={T}  delta={delta:.2e}  "
          f"ε_target={eps_target}  accountant={accountant}")

    # --- RDP-conversion worst-case ε (fully clipped example) ---
    eps_rdp, best_alpha_rdp, B_rdp = rdp_worst_case_epsilon(sigma, q, T, delta, ALPHA_GRID)
    print(f"\n  RDP-conversion worst-case ε (μ=1/σ every step):")
    print(f"    ε_RDP = {eps_rdp:.4f}  (best α = {best_alpha_rdp:.1f})")
    print(f"    This is the expected median ε_norm for a fully clipped example.")
    print(f"    If the paper reports median ε_norm ≈ 9.53 this should match.")

    # --- PRV ε at the same σ ---
    sigma_prv, eps_prv = prv_epsilon(eps_target, q, T, delta)
    if sigma_prv is not None:
        print(f"\n  PRV accountant at same ε_target={eps_target}:")
        print(f"    σ_PRV = {sigma_prv:.6f}  (should equal {sigma:.6f})")
        print(f"    ε_PRV = {eps_prv:.4f}  (should be ≈ {eps_target})")
        if abs(sigma_prv - sigma) > 1e-3:
            print(f"    [warn] σ mismatch: PRV calibrated σ={sigma_prv:.6f} "
                  f"vs saved σ={sigma:.6f}")
    else:
        print(f"\n  PRV accountant: not available (opacus not installed or failed).")
        print(f"  Manually: ε_PRV ≈ {eps_target} (PRV was used to calibrate σ).")

    # --- Compare ---
    print(f"\n  Summary:")
    print(f"    RDP-conversion ε (worst case) = {eps_rdp:.4f}")
    if eps_prv is not None:
        print(f"    PRV ε (forward check)         = {eps_prv:.4f}")
        gap = eps_rdp - eps_prv
        print(f"    Gap (RDP − PRV)               = {gap:.4f}  "
              f"({'RDP is looser, as expected' if gap > 0 else 'UNEXPECTED: PRV is looser'})")

    # --- Load observed ε_norm and compare ---
    tag     = f"p19_{args.run}_seed{args.seed}"
    en_path = os.path.join(args.cert_dir, f"{tag}_epsilon_cert_norm.npy")
    if os.path.exists(en_path):
        eps_norm_arr = np.load(en_path)
        eps_norm_med = float(np.median(eps_norm_arr))
        print(f"\n  Observed certified ε_norm (median over n={len(eps_norm_arr)} examples):")
        print(f"    median ε_norm = {eps_norm_med:.4f}")
        match = abs(eps_norm_med - eps_rdp) < 0.1
        print(f"    RDP worst-case = {eps_rdp:.4f}  "
              f"→ {'MATCH ✓' if match else f'MISMATCH (diff={eps_norm_med-eps_rdp:+.4f})'}")
        if not match:
            print(f"    [note] A small discrepancy is expected if not all examples are fully "
                  f"clipped (ε_norm is the per-example certified value, not uniformly worst-case).")
    else:
        print(f"\n  Certified ε_norm not found at {en_path}")
        print(f"  Run exp_p19_certify.py --run {args.run} --seed {args.seed} first.")

    # --- Save result ---
    out = {
        "run_id":          args.run,
        "seed":            args.seed,
        "sigma":           sigma,
        "q":               q,
        "T":               T,
        "delta":           delta,
        "eps_target":      eps_target,
        "eps_rdp_worst":   float(eps_rdp),
        "best_alpha_rdp":  float(best_alpha_rdp) if best_alpha_rdp is not None else None,
        "eps_prv_forward": float(eps_prv) if eps_prv is not None else None,
        "eps_norm_med_observed": float(np.median(np.load(en_path))) if os.path.exists(en_path) else None,
    }
    os.makedirs(os.path.join("results", "p19"), exist_ok=True)
    out_path = os.path.join("results", "p19", f"verify_accounting_{args.run}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  [saved] {out_path}")


if __name__ == "__main__":
    main()
