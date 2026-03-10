"""
run_all.py – Main entry point for Channeled DP-SGD experiments.

Executes experiments in priority order per the spec (Section 10):
  1. Exp 1: Gradient structure validation (P0 – stop if fails)
  2. Exp 3: Main comparison on CIFAR-10-LT (IR=50), ε=3 (P0)
  3. Exp 5.4: Imbalance ratio ablation (P0)
  4. Exp 2: Per-instance privacy / scaling law (P1)
  5. Exp 3: Remaining datasets and ε values (P1)
  6. Exp 5.1–5.3: Other ablations (P1/P2)
  7. Exp 4: MIA (P1)
  8. Exp 6: Structural predictions (P2)

Usage:
    python run_all.py [--priority P0|P1|P2|all] [--gpu 0] [--data_root ./data]
"""

import os
import sys
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Channeled DP-SGD Experiments"
    )
    parser.add_argument(
        "--priority", default="P0",
        choices=["P0", "P1", "P2", "all"],
        help="Run experiments up to this priority level."
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument(
        "--exp", default=None,
        help="Run only a specific experiment: exp1, exp2, exp3, exp4, exp5, exp6, plots"
    )
    return parser.parse_args()


def get_device(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU available. Experiments will be very slow on CPU.")
    return device


def check_gradient_stratification(exp1_results: dict) -> bool:
    """
    Success criterion for Exp 1: Tier 0 mean norm ≥ 3× smaller than Tier 2
    on CIFAR-10-LT (IR=50).
    """
    from experiments.exp1_gradient_structure import compute_tab1
    tab1 = compute_tab1(exp1_results)
    for key, tier_stats in tab1.items():
        if "IR50" in key and "_A" in key:  # Strategy A on LT data
            tier0_mean = tier_stats[0]["mean"]
            tier2_mean = tier_stats[2]["mean"]
            if tier2_mean > 0 and tier0_mean > 0:
                ratio = tier2_mean / tier0_mean
                print(f"\n[Success check] {key}: "
                      f"Tier2/Tier0 norm ratio = {ratio:.2f} "
                      f"(need ≥3.0 for success)")
                if ratio < 3.0:
                    print("  WARNING: Gradient stratification criterion NOT met.")
                    print("  Per spec: 'Stop if this fails.'")
                    return False
                else:
                    print("  SUCCESS: Gradient norms stratify by tier.")
                    return True
    return True  # if no LT data found, don't block


def run_experiment_1(device, data_root):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Gradient Structure Validation")
    print("=" * 60)
    from experiments.exp1_gradient_structure import run_all
    return run_all(device=device, data_root=data_root)


def run_experiment_2(device, data_root):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Per-Instance Privacy Validation")
    print("=" * 60)
    from experiments.exp2_per_instance_privacy import run_exp2
    return run_exp2(device=device, data_root=data_root)


def run_experiment_3(device, data_root, priority="P0"):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Main Accuracy Comparison")
    print("=" * 60)
    from experiments.exp3_main_comparison import (
        run_dataset_epsilon, DATASETS, EPSILONS
    )

    if priority == "P0":
        # Only CIFAR-10-LT (IR=50) at ε=3
        print("  [P0] Running CIFAR-10-LT IR=50 at ε=3 only.")
        run_dataset_epsilon("cifar10", 50.0, "wrn28-2", 100, 3.0, device,
                             data_root=data_root)
    else:
        # All datasets and ε values
        for (dname, ir, arch, epochs) in DATASETS:
            for eps in EPSILONS:
                print(f"\n  Dataset={dname} IR={ir} ε={eps}")
                run_dataset_epsilon(dname, ir, arch, epochs, eps, device,
                                    data_root=data_root)


def run_experiment_4(device, data_root):
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Membership Inference Attacks")
    print("=" * 60)
    print("  NOTE: This experiment requires ~600 GPU-hours (128 shadow models).")
    from experiments.exp4_mia import run_exp4
    return run_exp4(device=device, data_root=data_root)


def run_experiment_5(device, data_root, ablations="all"):
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Ablations")
    print("=" * 60)
    from experiments.exp5_ablations import (
        ablation_num_tiers, ablation_tier_strategy, ablation_calibration,
        ablation_imbalance_ratio, ablation_architecture
    )

    results = {}
    if ablations in ("5.4", "all"):
        print("\n  5.4: Imbalance ratio ablation")
        results["tab11"] = ablation_imbalance_ratio(device, data_root)

    if ablations in ("5.1", "all"):
        print("\n  5.1: Number of tiers")
        results["tab8"] = ablation_num_tiers(device, data_root)

    if ablations in ("5.2", "all"):
        print("\n  5.2: Tier assignment strategy")
        results["tab9"] = ablation_tier_strategy(device, data_root)

    if ablations in ("5.3", "all"):
        print("\n  5.3: Clipping calibration sensitivity")
        results["tab10"] = ablation_calibration(device, data_root)

    if ablations in ("5.5", "all"):
        print("\n  5.5: Architecture")
        results["tab12"] = ablation_architecture(device, data_root)

    return results


def run_experiment_6(device, data_root):
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Structural Predictions")
    print("=" * 60)
    from experiments.exp6_structural import run_exp6
    return run_exp6(device=device, data_root=data_root)


def generate_plots(exp1_r, exp2_r, exp3_r, exp4_r, exp5_r, exp6_r):
    print("\n" + "=" * 60)
    print("GENERATING ALL FIGURES")
    print("=" * 60)
    from plotting.figures import generate_all_figures
    generate_all_figures(
        exp1_results=exp1_r,
        exp2_results=exp2_r,
        exp3_results=exp3_r,
        exp4_results=exp4_r,
        exp5_results=(
            exp5_r.get("tab8"), exp5_r.get("tab9"),
            exp5_r.get("tab10"), exp5_r.get("tab11"), exp5_r.get("tab12")
        ) if exp5_r else None,
        exp6_results=exp6_r,
    )


def ensure_data(data_root: str):
    """
    Download datasets and pre-compute fixed splits if not already done.
    Idempotent: safe to call every run.
    """
    splits_dir = os.path.join(data_root, "splits")
    expected_keys = [
        "cifar10_IR1", "cifar10_IR10", "cifar10_IR50", "cifar10_IR100", "cifar100_IR1",
    ]
    missing = [k for k in expected_keys
               if not os.path.exists(os.path.join(splits_dir, f"{k}.npz"))]

    if not missing:
        print(f"[Data] All splits present in {splits_dir}.")
        return

    print(f"[Data] Missing splits: {missing}")
    print(f"[Data] Running setup_data.py ...")

    import subprocess
    result = subprocess.run(
        [sys.executable, "setup_data.py", "--data_root", data_root],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[Data] setup_data.py failed. Experiments will compute splits on-the-fly.")


def main():
    args = parse_args()
    device = get_device(args.gpu)

    print(f"\nChanneled DP-SGD Experiments")
    print(f"Priority: {args.priority}")
    print(f"Data root: {args.data_root}")
    print(f"Results will be saved to: ./results/\n")

    # Download + pre-compute splits (idempotent)
    if args.exp != "plots":
        ensure_data(args.data_root)

    exp1_r = exp2_r = exp3_r = exp4_r = exp5_r = exp6_r = None

    if args.exp is not None:
        # Run only a specific experiment
        if args.exp == "exp1":
            exp1_r = run_experiment_1(device, args.data_root)
        elif args.exp == "exp2":
            exp2_r = run_experiment_2(device, args.data_root)
        elif args.exp == "exp3":
            run_experiment_3(device, args.data_root, priority="P1")
        elif args.exp == "exp4":
            exp4_r = run_experiment_4(device, args.data_root)
        elif args.exp == "exp5":
            exp5_r = run_experiment_5(device, args.data_root)
        elif args.exp == "exp6":
            exp6_r = run_experiment_6(device, args.data_root)
        elif args.exp == "plots":
            # Load existing results and generate plots
            import pickle, json

            def _load(path):
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        return pickle.load(f)
                return None

            exp2_r = _load("results/exp2/results.pkl")
            exp6_r = _load("results/exp6/results.pkl")
            generate_plots(None, exp2_r, None, None, None, exp6_r)
        return

    # Priority-ordered execution
    # -------------------------------------------------------------------------
    # P0: Experiments 1, 3 (LT only), 5.4
    # -------------------------------------------------------------------------

    # Exp 1 (P0)
    exp1_r = run_experiment_1(device, args.data_root)

    # Critical success check
    if not check_gradient_stratification(exp1_r):
        print("\nERROR: Gradient structure validation FAILED.")
        print("Per spec Section 10: 'Stop if this fails.'")
        sys.exit(1)

    # Exp 3 (P0: LT only, ε=3)
    run_experiment_3(device, args.data_root, priority="P0")

    # Exp 5.4 (P0)
    exp5_r = run_experiment_5(device, args.data_root, ablations="5.4")

    if args.priority == "P0":
        generate_plots(exp1_r, None, None, None, exp5_r, None)
        print("\nP0 experiments complete.")
        return

    # -------------------------------------------------------------------------
    # P1: Exp 2, Exp 3 (full), Exp 5.1–5.3, Exp 4
    # -------------------------------------------------------------------------

    # Exp 2 (P1)
    exp2_r = run_experiment_2(device, args.data_root)

    # Exp 3 full (P1)
    run_experiment_3(device, args.data_root, priority="P1")

    # Exp 5 remaining ablations (P1)
    for abl in ["5.1", "5.2", "5.3"]:
        abl_r = run_experiment_5(device, args.data_root, ablations=abl)
        if exp5_r is None:
            exp5_r = {}
        exp5_r.update(abl_r)

    if args.priority == "P1":
        generate_plots(exp1_r, exp2_r, None, None, exp5_r, None)
        print("\nP1 experiments complete.")
        return

    # -------------------------------------------------------------------------
    # P2: Exp 4 (MIA), Exp 5.5, Exp 6
    # -------------------------------------------------------------------------

    # Exp 4 (P1 for core, P2 for per-tier)
    exp4_r = run_experiment_4(device, args.data_root)

    # Exp 5.5 (P2)
    abl_r = run_experiment_5(device, args.data_root, ablations="5.5")
    if exp5_r is None:
        exp5_r = {}
    exp5_r.update(abl_r)

    # Exp 6 (P2)
    exp6_r = run_experiment_6(device, args.data_root)

    # All figures
    generate_plots(exp1_r, exp2_r, None, exp4_r, exp5_r, exp6_r)

    print("\nAll experiments complete.")
    print("Results saved to ./results/")
    print("Figures saved to ./results/figures/")


if __name__ == "__main__":
    main()
