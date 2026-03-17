"""
Experiment 1: Gradient Structure Validation
============================================

Goal: Confirm that gradient norms stratify by tier (Hypothesis 1).

Procedure:
- Train WRN-28-2 on CIFAR-10 with standard DP-SGD (ε=3, C=1.0) for 100 epochs.
- Save checkpoints at epochs {1, 5, 10, 25, 50, 75, 100}.
- At each checkpoint, compute per-sample unclipped and clipped gradient norms.
- Assign tiers using Strategies A and B with K=3.
- Repeat for CIFAR-10-LT (IR=50), CIFAR-100 with 3 seeds.

Outputs: Fig 1, Fig 2, Tab 1 (saved to results/exp1/).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pickle

from src.datasets import load_datasets, make_data_loaders, TieredDataset
from src.models import make_model, validate_model_for_dp
from src.dp_training import (
    StandardDPSGDTrainer, evaluate, compute_per_sample_losses, set_seed
)
from src.calibration import (
    compute_per_sample_gradient_norms,
    compute_per_sample_clipped_norms,
    train_public_model,
)
from src.tiers import assign_tiers, get_tier_sizes
from src.privacy_accounting import compute_sigma
from src.evaluation import save_results, extract_features


RESULTS_DIR = "results/exp1"
DATA_ROOT = "./data"
ARCH = "wrn28-2"
BATCH_SIZE = 256
EPOCHS = 100
EPS_TARGET = 3.0
DELTA = 1e-5
C = 1.0  # global clipping bound for standard DP-SGD
K = 3
CHECKPOINT_EPOCHS = [1, 5, 10, 25, 50, 75, 100]
SEEDS = [0, 1, 2]
DATASETS = [
    ("cifar10", 1.0),
    ("cifar10", 50.0),   # CIFAR-10-LT IR=50
    ("cifar100", 1.0),
]


def run_exp1(
    dataset_name: str,
    imbalance_ratio: float,
    seed: int,
    device: torch.device,
    data_root: str = DATA_ROOT,
    results_dir: str = RESULTS_DIR,
):
    """Run Experiment 1 for a single (dataset, IR, seed) configuration."""
    tag = f"{dataset_name}_IR{imbalance_ratio:.0f}_seed{seed}"
    out_dir = os.path.join(results_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    result_path = os.path.join(out_dir, "results.pkl")
    if os.path.exists(result_path):
        print(f"[Exp1] {tag}: already computed, skipping.")
        return load_existing(result_path)

    set_seed(seed)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    data = load_datasets(
        dataset_name=dataset_name,
        data_root=data_root,
        imbalance_ratio=imbalance_ratio,
        public_frac=0.1,
        split_seed=42,  # fixed across all seeds/methods
    )
    num_classes = data["num_classes"]
    n_train = data["n_train"]
    class_counts = data["class_counts"]
    private_indices = data["private_indices"]

    private_loader, public_loader, test_loader = make_data_loaders(
        data, batch_size=BATCH_SIZE
    )

    # Non-shuffled, no-augmentation loader for gradient norm evaluation.
    # CRITICAL: must match the order of private_targets / tiers_A / tiers_B
    # so that norms[i] corresponds to sample i in private_dataset.
    from torch.utils.data import DataLoader as _DataLoader
    eval_loader = _DataLoader(
        data["private_dataset_noaug"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # Private targets (for tier assignment)
    private_targets = np.array(data["private_dataset"].targets)

    # ------------------------------------------------------------------
    # 2. Compute σ
    # ------------------------------------------------------------------
    q = BATCH_SIZE / n_train
    T = EPOCHS * int(np.ceil(n_train / BATCH_SIZE))
    sigma = compute_sigma(EPS_TARGET, DELTA, q, T)
    print(f"[Exp1] {tag}: σ={sigma:.4f}, q={q:.5f}, T={T}")

    # ------------------------------------------------------------------
    # 3. Train public model (for Strategy B density tiers)
    # ------------------------------------------------------------------
    public_model = make_model(ARCH, num_classes)
    public_model = train_public_model(
        public_model, public_loader, device, epochs=50, verbose=False
    )

    # Strategy A tiers (for LT) or class-index-mod-K (for balanced)
    tiers_A = assign_tiers("A", private_targets, class_counts, K=K)
    tier_sizes_A = get_tier_sizes(tiers_A, K)
    print(f"[Exp1] Strategy A tier sizes: {tier_sizes_A}")

    # Strategy B tiers (density-based)
    feats_public, _ = extract_features(public_model, public_loader, device)
    feats_private, _ = extract_features(public_model, private_loader, device)
    tiers_B = assign_tiers(
        "B", private_targets, class_counts, K=K,
        features_public=feats_public, features_all=feats_private
    )
    tier_sizes_B = get_tier_sizes(tiers_B, K)
    print(f"[Exp1] Strategy B tier sizes: {tier_sizes_B}")

    # ------------------------------------------------------------------
    # 4. Train with standard DP-SGD, save checkpoints
    # ------------------------------------------------------------------
    model = make_model(ARCH, num_classes)
    assert validate_model_for_dp(model), "Model has incompatible layers for DP-SGD"

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    trainer = StandardDPSGDTrainer(
        model=model,
        sigma=sigma,
        C=C,
        n_train=n_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=0.1,  # spec §3.2: initial LR 0.1 with cosine annealing
        device=device,
        delta=DELTA,
    )
    loader = trainer.make_private(private_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=EPOCHS
    )

    # Per-epoch data collection
    checkpoint_data = {}

    for epoch in range(1, EPOCHS + 1):
        trainer.model.train()
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            trainer.optimizer.zero_grad()
            out = trainer.model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
            loss.backward()
            trainer.optimizer.step()
        scheduler.step()

        if epoch in CHECKPOINT_EPOCHS:
            print(f"[Exp1] {tag}: epoch {epoch}, computing gradient norms...")
            # Use eval_loader (shuffle=False, no-aug) so that norms[i]
            # aligns with private_targets[i] / tiers_A[i] / tiers_B[i].
            unclipped_norms = compute_per_sample_gradient_norms(
                trainer.model, eval_loader, device
            )
            clipped_norms = compute_per_sample_clipped_norms(unclipped_norms, C)

            checkpoint_data[epoch] = {
                "unclipped_norms": unclipped_norms,  # (n_train,)
                "clipped_norms": clipped_norms,       # (n_train,)
                "tiers_A": tiers_A,
                "tiers_B": tiers_B,
                "labels": private_targets,
            }

    # Per-sample loss and prediction at convergence (epoch 100).
    # Use eval_loader to keep alignment with tiers_A / tiers_B.
    losses, labels, preds, confs = compute_per_sample_losses(
        trainer.model, eval_loader, device
    )
    checkpoint_data["convergence"] = {
        "losses": losses,
        "labels": labels,
        "predictions": preds,
        "confidences": confs,
        "correctly_classified": (preds == labels).astype(bool),
    }

    # Final test accuracy
    test_acc = evaluate(trainer.model, test_loader, device)
    eps_achieved = trainer.privacy_engine.get_epsilon(DELTA)
    print(f"[Exp1] {tag}: final test_acc={test_acc:.4f}, ε={eps_achieved:.3f}")

    results = {
        "tag": tag,
        "dataset": dataset_name,
        "imbalance_ratio": imbalance_ratio,
        "seed": seed,
        "sigma": sigma,
        "C": C,
        "tiers_A": tiers_A,
        "tiers_B": tiers_B,
        "tier_sizes_A": tier_sizes_A,
        "tier_sizes_B": tier_sizes_B,
        "class_counts": class_counts,
        "checkpoint_data": checkpoint_data,
        "test_acc": test_acc,
        "epsilon": eps_achieved,
    }

    save_results(results, result_path)
    print(f"[Exp1] {tag}: saved to {result_path}")
    return results


def load_existing(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def run_all(device: torch.device = None, data_root: str = DATA_ROOT):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp1] Using device: {device}")

    all_results = {}
    for (dname, ir) in DATASETS:
        for seed in SEEDS:
            r = run_exp1(dname, ir, seed, device, data_root=data_root)
            key = f"{dname}_IR{ir:.0f}_seed{seed}"
            all_results[key] = r

    return all_results


# ---------------------------------------------------------------------------
# Summary statistics (Tab 1)
# ---------------------------------------------------------------------------

def compute_tab1(all_results: dict) -> dict:
    """
    Compute per-tier mean and 95th-percentile clipped gradient norms at convergence.

    Returns nested dict: tab1[dataset][strategy][tier] = {'mean': ..., 'p95': ...}
    """
    tab1 = {}
    for key, r in all_results.items():
        dname = r["dataset"]
        ir = r["imbalance_ratio"]
        ckpt = r["checkpoint_data"]
        final_epoch = max(e for e in ckpt.keys() if isinstance(e, int))
        clipped_norms = ckpt[final_epoch]["clipped_norms"]

        for strat_name, tiers in [("A", r["tiers_A"]), ("B", r["tiers_B"])]:
            combo_key = f"{dname}_IR{ir:.0f}_{strat_name}"
            if combo_key not in tab1:
                tab1[combo_key] = {k: {"mean_list": [], "p95_list": []} for k in range(K)}

            for k in range(K):
                mask = (tiers == k)
                tier_norms = clipped_norms[mask]
                if len(tier_norms) > 0:
                    tab1[combo_key][k]["mean_list"].append(float(tier_norms.mean()))
                    tab1[combo_key][k]["p95_list"].append(float(np.percentile(tier_norms, 95)))

    # Average across seeds
    summary = {}
    for combo_key, tier_dict in tab1.items():
        summary[combo_key] = {}
        for k, stats in tier_dict.items():
            summary[combo_key][k] = {
                "mean": np.mean(stats["mean_list"]),
                "p95": np.mean(stats["p95_list"]),
            }
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 1: Gradient Structure")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    all_results = run_all(device=device, data_root=args.data_root)
    tab1 = compute_tab1(all_results)
    print("\n=== Tab 1: Per-tier gradient norm statistics at convergence ===")
    for combo_key, tier_dict in tab1.items():
        print(f"\n{combo_key}:")
        for k, stats in tier_dict.items():
            print(f"  Tier {k}: mean={stats['mean']:.4f}, p95={stats['p95']:.4f}")
