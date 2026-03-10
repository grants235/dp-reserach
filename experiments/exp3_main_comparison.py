"""
Experiment 3: Main Accuracy Comparison
=======================================

Goal: Compare test accuracy of Channeled DP-SGD vs. baselines at matched ε.

Methods:
  - Standard (grid): grid search C ∈ {0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0}
  - Standard (median): C = median of public-model gradient norms
  - Adaptive clip: Andrew et al. adaptive clipping
  - Channeled K=2: Strategy A, c=2
  - Channeled K=3: Strategy A, c=2
  - Channeled K=3 (density): Strategy B, c=2
  - Non-private: no DP

Datasets: CIFAR-10, CIFAR-10-LT (IR=50), CIFAR-100
ε ∈ {1, 3, 8}
5 seeds per configuration

Outputs: Tab 3–5, Fig 5 (saved to results/exp3/).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from src.datasets import load_datasets, make_data_loaders, TieredDataset
from src.models import make_model, validate_model_for_dp
from src.dp_training import (
    StandardDPSGDTrainer, ChanneledDPSGDTrainer, AdaptiveClipTrainer,
    NonPrivateTrainer, evaluate, evaluate_per_class, set_seed,
)
from src.tiers import assign_tiers
from src.calibration import (
    compute_per_sample_gradient_norms, calibrate_clipping_bounds,
    train_public_model,
)
from src.privacy_accounting import compute_sigma
from src.evaluation import save_results, save_json, extract_features


RESULTS_DIR = "results/exp3"
DATA_ROOT = "./data"
BATCH_SIZE = 256
DELTA = 1e-5
N_SEEDS = 5
K_VALUES = [2, 3]
C_GRID = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
C_CALIBRATION_MULTIPLIER = 2.0

DATASETS = [
    ("cifar10",  1.0,  "wrn28-2", 100),    # CIFAR-10 balanced
    ("cifar10",  50.0, "wrn28-2", 100),    # CIFAR-10-LT IR=50
    ("cifar100", 1.0,  "wrn28-2", 200),    # CIFAR-100
]
EPSILONS = [1.0, 3.0, 8.0]


def run_config(
    method: str,
    dataset_name: str,
    imbalance_ratio: float,
    arch: str,
    epochs: int,
    epsilon: float,
    seed: int,
    device: torch.device,
    data_root: str = DATA_ROOT,
    C: float = 1.0,               # for Standard (grid)
    K: int = 3,                   # for Channeled
    tier_strategy: str = "A",     # for Channeled
) -> dict:
    """
    Run one (method, dataset, epsilon, seed) configuration.
    Returns dict with test_acc, train_acc, per_class_acc, etc.
    """
    set_seed(seed)

    data = load_datasets(dataset_name, data_root=data_root,
                         imbalance_ratio=imbalance_ratio,
                         public_frac=0.1, split_seed=42)
    num_classes = data["num_classes"]
    n_train = data["n_train"]
    class_counts = data["class_counts"]
    private_targets = np.array(data["private_dataset"].targets)
    private_loader, public_loader, test_loader = make_data_loaders(data, BATCH_SIZE)

    q = BATCH_SIZE / n_train
    T = epochs * int(np.ceil(n_train / BATCH_SIZE))
    sigma = compute_sigma(epsilon, DELTA, q, T)

    model = make_model(arch, num_classes)
    assert validate_model_for_dp(model)

    # ------------------------------------------------------------------
    # Dispatch to trainer
    # ------------------------------------------------------------------

    if method == "non_private":
        trainer = NonPrivateTrainer(
            model=model, n_train=n_train, epochs=epochs, device=device
        )
        result = trainer.train(private_loader, test_loader, verbose=False)
        per_class = evaluate_per_class(trainer.model, test_loader, num_classes, device)
        result["per_class_acc"] = per_class.tolist()
        return result

    elif method.startswith("standard"):
        if method == "standard_median":
            # C = median of public-model gradient norms
            pub_model = make_model(arch, num_classes)
            pub_model = train_public_model(pub_model, public_loader, device,
                                           epochs=50, verbose=False)
            pub_norms = compute_per_sample_gradient_norms(pub_model, private_loader, device)
            C = float(np.median(pub_norms))

        trainer = StandardDPSGDTrainer(
            model=model, sigma=sigma, C=C,
            n_train=n_train, batch_size=BATCH_SIZE, epochs=epochs,
            device=device, delta=DELTA,
        )
        loader = trainer.make_private(private_loader)

        # Manual training loop so we can control scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=epochs
        )
        for epoch in range(1, epochs + 1):
            trainer.model.train()
            for batch in loader:
                x, y = batch[0].to(device), batch[1].to(device)
                trainer.optimizer.zero_grad()
                out = trainer.model(x)
                loss = torch.nn.functional.cross_entropy(out, y)
                loss.backward()
                trainer.optimizer.step()
            scheduler.step()

        test_acc = evaluate(trainer.model, test_loader, device)
        train_acc = evaluate(trainer.model, private_loader, device)
        per_class = evaluate_per_class(trainer.model, test_loader, num_classes, device)
        eps_ach = trainer.privacy_engine.get_epsilon(DELTA)
        return dict(test_acc=test_acc, train_acc=train_acc,
                    per_class_acc=per_class.tolist(),
                    epsilon=eps_ach, sigma=sigma, C=C)

    elif method == "adaptive":
        trainer = AdaptiveClipTrainer(
            model=model, sigma=sigma, n_train=n_train, batch_size=BATCH_SIZE,
            epochs=epochs, device=device, delta=DELTA,
        )
        result = trainer.train(private_loader, test_loader, verbose=False)
        per_class = evaluate_per_class(trainer.model, test_loader, num_classes, device)
        result["per_class_acc"] = per_class.tolist()
        return result

    elif method.startswith("channeled"):
        # Train public model for calibration
        pub_model = make_model(arch, num_classes)
        pub_model = train_public_model(pub_model, public_loader, device,
                                       epochs=50, verbose=False)

        # Tier assignment
        if tier_strategy == "A":
            tiers = assign_tiers("A", private_targets, class_counts, K=K)
        elif tier_strategy == "B":
            feats_pub, _ = extract_features(pub_model, public_loader, device)
            feats_priv, _ = extract_features(pub_model, private_loader, device)
            tiers = assign_tiers("B", private_targets, class_counts, K=K,
                                 features_public=feats_pub, features_all=feats_priv)
        else:
            raise ValueError(f"Unknown tier strategy: {tier_strategy}")

        # Calibrate C_k
        C_list = calibrate_clipping_bounds(
            pub_model, private_loader, tiers, K,
            c=C_CALIBRATION_MULTIPLIER, device=device,
        )
        print(f"    C_per_tier = {[f'{c:.3f}' for c in C_list]}")

        # Wrap dataset with tier labels for ChanneledDPSGDTrainer
        tiered_dataset = TieredDataset(data["private_dataset"], tiers)
        from torch.utils.data import DataLoader as DL
        tiered_loader = DL(tiered_dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=False)

        trainer = ChanneledDPSGDTrainer(
            model=model, sigma=sigma, C_per_tier=C_list,
            n_train=n_train, batch_size=BATCH_SIZE, epochs=epochs,
            device=device, delta=DELTA,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=epochs
        )
        for epoch in range(1, epochs + 1):
            trainer.train_epoch(tiered_loader)
            scheduler.step()

        test_acc = evaluate(trainer.model, test_loader, device)
        train_acc = evaluate(trainer.model, private_loader, device)
        per_class = evaluate_per_class(trainer.model, test_loader, num_classes, device)
        return dict(test_acc=test_acc, train_acc=train_acc,
                    per_class_acc=per_class.tolist(),
                    sigma=sigma, C_per_tier=C_list, K=K)

    else:
        raise ValueError(f"Unknown method: {method}")


def run_dataset_epsilon(
    dataset_name: str,
    imbalance_ratio: float,
    arch: str,
    epochs: int,
    epsilon: float,
    device: torch.device,
    data_root: str = DATA_ROOT,
    results_dir: str = RESULTS_DIR,
) -> dict:
    """
    Run all methods × all seeds for one (dataset, ε) pair.
    Returns nested dict: results[method][seed] = metrics_dict.
    """
    tag = f"{dataset_name}_IR{imbalance_ratio:.0f}_eps{epsilon:.0f}"
    out_dir = os.path.join(results_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    # Define all methods with their configs
    methods_configs = []

    # Standard grid search – run all C values, pick best at the end
    for C_val in C_GRID:
        methods_configs.append({
            "method": "standard_grid",
            "C": C_val,
            "label": f"standard_C{C_val}",
        })
    # Standard (median C)
    methods_configs.append({"method": "standard_median", "label": "standard_median"})
    # Adaptive clipping
    methods_configs.append({"method": "adaptive", "label": "adaptive"})
    # Channeled K=2, Strategy A
    methods_configs.append({"method": "channeled", "K": 2, "tier_strategy": "A",
                             "label": "channeled_K2_A"})
    # Channeled K=3, Strategy A
    methods_configs.append({"method": "channeled", "K": 3, "tier_strategy": "A",
                             "label": "channeled_K3_A"})
    # Channeled K=3, Strategy B
    methods_configs.append({"method": "channeled", "K": 3, "tier_strategy": "B",
                             "label": "channeled_K3_B"})
    # Non-private (run once, epsilon-independent)
    methods_configs.append({"method": "non_private", "label": "non_private"})

    for cfg in methods_configs:
        label = cfg["label"]
        method = cfg["method"]
        method_dir = os.path.join(out_dir, label)
        os.makedirs(method_dir, exist_ok=True)
        seed_results = []

        for seed in range(N_SEEDS):
            seed_path = os.path.join(method_dir, f"seed{seed}.json")
            if os.path.exists(seed_path):
                from src.evaluation import load_json
                r = load_json(seed_path)
                seed_results.append(r)
                continue

            print(f"  [{tag}] {label} seed={seed} ...")
            try:
                r = run_config(
                    method=method,
                    dataset_name=dataset_name,
                    imbalance_ratio=imbalance_ratio,
                    arch=arch,
                    epochs=epochs,
                    epsilon=epsilon,
                    seed=seed,
                    device=device,
                    data_root=data_root,
                    C=cfg.get("C", 1.0),
                    K=cfg.get("K", 3),
                    tier_strategy=cfg.get("tier_strategy", "A"),
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                r = {"test_acc": float("nan"), "error": str(e)}

            save_json(r, seed_path)
            seed_results.append(r)

        all_results[label] = seed_results

    # Summarize
    summary = summarize_results(all_results)
    save_json(summary, os.path.join(out_dir, "summary.json"))
    return all_results


def summarize_results(all_results: dict) -> dict:
    """Compute mean ± std test accuracy for each method."""
    summary = {}
    for label, seed_results in all_results.items():
        accs = [r["test_acc"] for r in seed_results if "test_acc" in r and
                not (isinstance(r["test_acc"], float) and
                     r["test_acc"] != r["test_acc"])]  # filter NaN
        if accs:
            summary[label] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "n_seeds": len(accs),
            }
    # Best standard C
    grid_entries = {k: v for k, v in summary.items() if k.startswith("standard_C")}
    if grid_entries:
        best_label = max(grid_entries, key=lambda k: grid_entries[k]["mean"])
        summary["standard_best"] = {
            **grid_entries[best_label],
            "best_C": best_label.replace("standard_C", ""),
        }
    return summary


def run_all(device: torch.device = None, data_root: str = DATA_ROOT):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp3] Using device: {device}")

    for (dname, ir, arch, epochs) in DATASETS:
        for eps in EPSILONS:
            print(f"\n[Exp3] Dataset={dname} IR={ir} ε={eps}")
            run_dataset_epsilon(dname, ir, arch, epochs, eps, device,
                                data_root=data_root)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 3: Main comparison")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", default="all",
                        help="cifar10, cifar10_lt, cifar100, or all")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Run only this epsilon value")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    run_all(device=device, data_root=args.data_root)
