"""
Experiment 5: Ablations
=======================

All ablations: CIFAR-10-LT (IR=50), ε=3, WRN-28-2, 5 seeds.

5.1: Number of tiers  K ∈ {1, 2, 3, 5, 10}
5.2: Tier assignment strategy  {A, B, C}
5.3: Clipping calibration  c ∈ {1.5, 2, 3, 5}, public-95th-pctile, fixed-ratio, all-equal
5.4: Imbalance ratio  IR ∈ {1, 10, 50, 100}
5.5: Architecture  WRN-28-2 vs ResNet-20

Outputs: Tab 8–12, Fig 8–9 (saved to results/exp5/).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as DL

from src.datasets import load_datasets, make_data_loaders, TieredDataset
from src.models import make_model, validate_model_for_dp
from src.dp_training import (
    ChanneledDPSGDTrainer, StandardDPSGDTrainer,
    evaluate, set_seed,
)
from src.tiers import assign_tiers, get_tier_sizes
from src.calibration import (
    calibrate_clipping_bounds, calibrate_fixed_ratio,
    calibrate_all_equal, train_public_model,
    compute_per_sample_gradient_norms,
)
from src.privacy_accounting import compute_sigma
from src.evaluation import save_json, extract_features


RESULTS_DIR = "results/exp5"
DATA_ROOT = "./data"
BATCH_SIZE = 256
EPOCHS = 100
EPSILON = 3.0
DELTA = 1e-5
N_SEEDS = 5


# ---------------------------------------------------------------------------
# Core training helpers
# ---------------------------------------------------------------------------

def run_channeled(
    model, sigma, C_per_tier, tiers, n_train, epochs, device,
    private_dataset, test_loader, verbose=False
) -> dict:
    """
    Run channeled DP-SGD and return test/train accuracy.

    Wraps private_dataset with TieredDataset so the trainer receives
    tier labels directly in each batch.
    """
    tiered_dataset = TieredDataset(private_dataset, tiers)
    tiered_loader = DL(tiered_dataset, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=False)

    trainer = ChanneledDPSGDTrainer(
        model=model, sigma=sigma, C_per_tier=C_per_tier,
        n_train=n_train, batch_size=BATCH_SIZE,
        epochs=epochs, device=device, delta=DELTA,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=epochs
    )
    for epoch in range(1, epochs + 1):
        trainer.train_epoch(tiered_loader)
        sched.step()

    test_acc = evaluate(trainer.model, test_loader, device)
    train_acc = evaluate(trainer.model, tiered_loader, device)
    return dict(test_acc=test_acc, train_acc=train_acc,
                C_per_tier=C_per_tier, K=len(C_per_tier))


def run_standard(
    model, sigma, C, n_train, epochs, device,
    private_loader, test_loader
) -> dict:
    """Run standard DP-SGD and return test/train accuracy."""
    trainer = StandardDPSGDTrainer(
        model=model, sigma=sigma, C=C, n_train=n_train, batch_size=BATCH_SIZE,
        epochs=epochs, device=device, delta=DELTA,
    )
    loader = trainer.make_private(private_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=epochs
    )
    for epoch in range(1, epochs + 1):
        trainer.model.train()
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            trainer.optimizer.zero_grad()
            out = trainer.model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            trainer.optimizer.step()
        sched.step()
    test_acc = evaluate(trainer.model, test_loader, device)
    train_acc = evaluate(trainer.model, private_loader, device)
    eps = trainer.privacy_engine.get_epsilon(DELTA)
    return dict(test_acc=test_acc, train_acc=train_acc, epsilon=eps, C=C)


def _load_base_data(dataset_name, imbalance_ratio, arch, data_root, device, seed):
    """Load data, compute sigma, train public model, return everything needed."""
    set_seed(seed)
    data = load_datasets(dataset_name, data_root=data_root,
                         imbalance_ratio=imbalance_ratio,
                         public_frac=0.1, split_seed=42)
    n_train = data["n_train"]
    num_classes = data["num_classes"]
    class_counts = data["class_counts"]
    private_targets = np.array(data["private_dataset"].targets)
    private_loader, public_loader, test_loader = make_data_loaders(data, BATCH_SIZE)

    q = BATCH_SIZE / n_train
    T = EPOCHS * int(np.ceil(n_train / BATCH_SIZE))
    sigma = compute_sigma(EPSILON, DELTA, q, T)

    # Public model for calibration
    pub_model = make_model(arch, num_classes)
    pub_model = train_public_model(pub_model, public_loader, device,
                                   epochs=50, verbose=False)

    private_dataset = data["private_dataset"]
    return (data, n_train, num_classes, class_counts, private_targets,
            private_loader, public_loader, test_loader, sigma, pub_model,
            private_dataset)


# ---------------------------------------------------------------------------
# 5.1: Number of Tiers
# ---------------------------------------------------------------------------

def ablation_num_tiers(device, data_root=DATA_ROOT, results_dir=RESULTS_DIR):
    """Tab 8, Fig 8: Test accuracy vs K ∈ {1, 2, 3, 5, 10}."""
    K_VALUES = [1, 2, 3, 5, 10]
    arch = "wrn28-2"
    out_dir = os.path.join(results_dir, "5.1_num_tiers")
    os.makedirs(out_dir, exist_ok=True)
    tab8 = {}

    for K in K_VALUES:
        accs_by_seed = []
        for seed in range(N_SEEDS):
            fname = os.path.join(out_dir, f"K{K}_seed{seed}.json")
            if os.path.exists(fname):
                from src.evaluation import load_json
                r = load_json(fname)
                accs_by_seed.append(r)
                continue

            print(f"  [Exp5.1] K={K}, seed={seed}")
            (data, n_train, num_classes, class_counts, private_targets,
             private_loader, public_loader, test_loader, sigma, pub_model,
             private_dataset) = _load_base_data(
                 "cifar10", 50.0, arch, data_root, device, seed)

            tiers_A = assign_tiers("A", private_targets, class_counts, K=K)
            C_list = calibrate_clipping_bounds(pub_model, private_loader, tiers_A,
                                                K=K, c=2.0, device=device)
            model = make_model(arch, num_classes)

            if K == 1:
                # K=1 is equivalent to standard DP-SGD with calibrated C
                r = run_standard(model, sigma, C_list[0], n_train, EPOCHS,
                                  device, private_loader, test_loader)
            else:
                r = run_channeled(model, sigma, C_list, tiers_A, n_train, EPOCHS,
                                   device, private_dataset, test_loader)

            save_json(r, fname)
            accs_by_seed.append(r)

        accs = [r["test_acc"] for r in accs_by_seed]
        tab8[K] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
        print(f"  K={K}: mean={tab8[K]['mean']:.4f} ± {tab8[K]['std']:.4f}")

    save_json(tab8, os.path.join(out_dir, "tab8.json"))
    return tab8


# ---------------------------------------------------------------------------
# 5.2: Tier Assignment Strategy
# ---------------------------------------------------------------------------

def ablation_tier_strategy(device, data_root=DATA_ROOT, results_dir=RESULTS_DIR):
    """Tab 9: Test accuracy by tier strategy {A, B, C}."""
    arch = "wrn28-2"
    K = 3
    out_dir = os.path.join(results_dir, "5.2_tier_strategy")
    os.makedirs(out_dir, exist_ok=True)
    tab9 = {}

    for strategy in ["A", "B", "C"]:
        accs_by_seed = []
        for seed in range(N_SEEDS):
            fname = os.path.join(out_dir, f"strat{strategy}_seed{seed}.json")
            if os.path.exists(fname):
                from src.evaluation import load_json
                r = load_json(fname)
                accs_by_seed.append(r)
                continue

            print(f"  [Exp5.2] Strategy={strategy}, seed={seed}")
            (data, n_train, num_classes, class_counts, private_targets,
             private_loader, public_loader, test_loader, sigma, pub_model,
             private_dataset) = _load_base_data(
                 "cifar10", 50.0, arch, data_root, device, seed)

            if strategy == "B":
                feats_pub, _ = extract_features(pub_model, public_loader, device)
                feats_priv, _ = extract_features(pub_model, private_loader, device)
                tiers = assign_tiers("B", private_targets, class_counts, K=K,
                                     features_public=feats_pub, features_all=feats_priv)
            elif strategy == "C":
                tiers_a = assign_tiers("A", private_targets, class_counts, K=K)
                tier_sizes = get_tier_sizes(tiers_a, K)
                from src.tiers import tier_by_random
                tiers = tier_by_random(len(private_targets), tier_sizes, seed=seed)
            else:
                tiers = assign_tiers("A", private_targets, class_counts, K=K)

            C_list = calibrate_clipping_bounds(pub_model, private_loader, tiers,
                                                K=K, c=2.0, device=device)
            model = make_model(arch, num_classes)
            r = run_channeled(model, sigma, C_list, tiers, n_train, EPOCHS,
                               device, private_dataset, test_loader)
            save_json(r, fname)
            accs_by_seed.append(r)

        accs = [r["test_acc"] for r in accs_by_seed]
        tab9[strategy] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
        print(f"  Strategy {strategy}: mean={tab9[strategy]['mean']:.4f}")

    save_json(tab9, os.path.join(out_dir, "tab9.json"))
    return tab9


# ---------------------------------------------------------------------------
# 5.3: Clipping Calibration Sensitivity
# ---------------------------------------------------------------------------

def ablation_calibration(device, data_root=DATA_ROOT, results_dir=RESULTS_DIR):
    """Tab 10: Test accuracy by calibration method."""
    arch = "wrn28-2"
    K = 3
    out_dir = os.path.join(results_dir, "5.3_calibration")
    os.makedirs(out_dir, exist_ok=True)
    tab10 = {}

    configs = []
    for c in [1.5, 2.0, 3.0, 5.0]:
        configs.append((f"public_p95_c{c}", "public_c", c))
    configs.append(("fixed_ratio", "fixed_ratio", None))
    configs.append(("all_equal", "all_equal", None))

    for (config_name, calib_type, c_val) in configs:
        accs_by_seed = []
        for seed in range(N_SEEDS):
            fname = os.path.join(out_dir, f"{config_name}_seed{seed}.json")
            if os.path.exists(fname):
                from src.evaluation import load_json
                r = load_json(fname)
                accs_by_seed.append(r)
                continue

            print(f"  [Exp5.3] {config_name}, seed={seed}")
            (data, n_train, num_classes, class_counts, private_targets,
             private_loader, public_loader, test_loader, sigma, pub_model,
             private_dataset) = _load_base_data(
                 "cifar10", 50.0, arch, data_root, device, seed)

            tiers = assign_tiers("A", private_targets, class_counts, K=K)

            if calib_type == "public_c":
                C_list = calibrate_clipping_bounds(pub_model, private_loader, tiers,
                                                    K=K, c=c_val, device=device)
            elif calib_type == "fixed_ratio":
                C_base = calibrate_clipping_bounds(pub_model, private_loader, tiers,
                                                    K=K, c=2.0, device=device)
                C_list = calibrate_fixed_ratio(max(C_base), K)
            else:  # all_equal
                C_base = calibrate_clipping_bounds(pub_model, private_loader, tiers,
                                                    K=K, c=2.0, device=device)
                C_list = calibrate_all_equal(max(C_base), K)

            model = make_model(arch, num_classes)
            if calib_type == "all_equal":
                # All equal = standard DP-SGD with C = C_max
                r = run_standard(model, sigma, C_list[0], n_train, EPOCHS,
                                  device, private_loader, test_loader)
            else:
                r = run_channeled(model, sigma, C_list, tiers, n_train, EPOCHS,
                                   device, private_dataset, test_loader)

            save_json(r, fname)
            accs_by_seed.append(r)

        accs = [r["test_acc"] for r in accs_by_seed]
        tab10[config_name] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
        print(f"  {config_name}: mean={tab10[config_name]['mean']:.4f}")

    save_json(tab10, os.path.join(out_dir, "tab10.json"))
    return tab10


# ---------------------------------------------------------------------------
# 5.4: Imbalance Ratio
# ---------------------------------------------------------------------------

def ablation_imbalance_ratio(device, data_root=DATA_ROOT, results_dir=RESULTS_DIR):
    """Tab 11, Fig 9: Standard vs Channeled at each IR ∈ {1, 10, 50, 100}."""
    arch = "wrn28-2"
    K = 3
    IR_VALUES = [1.0, 10.0, 50.0, 100.0]
    out_dir = os.path.join(results_dir, "5.4_imbalance_ratio")
    os.makedirs(out_dir, exist_ok=True)
    tab11 = {}

    for ir in IR_VALUES:
        tab11[ir] = {}
        for method_name in ["standard", "channeled"]:
            accs_by_seed = []
            for seed in range(N_SEEDS):
                fname = os.path.join(out_dir, f"IR{ir:.0f}_{method_name}_seed{seed}.json")
                if os.path.exists(fname):
                    from src.evaluation import load_json
                    r = load_json(fname)
                    accs_by_seed.append(r)
                    continue

                print(f"  [Exp5.4] IR={ir}, method={method_name}, seed={seed}")
                (data, n_train, num_classes, class_counts, private_targets,
                 private_loader, public_loader, test_loader, sigma, pub_model,
                 private_dataset) = _load_base_data(
                     "cifar10", ir, arch, data_root, device, seed)

                if method_name == "standard":
                    # Use median C from public model gradient norms
                    pub_norms = compute_per_sample_gradient_norms(
                        pub_model, private_loader, device
                    )
                    C = float(np.median(pub_norms))
                    model = make_model(arch, num_classes)
                    r = run_standard(model, sigma, C, n_train, EPOCHS,
                                     device, private_loader, test_loader)
                else:
                    tiers = assign_tiers("A", private_targets, class_counts, K=K)
                    C_list = calibrate_clipping_bounds(pub_model, private_loader, tiers,
                                                        K=K, c=2.0, device=device)
                    model = make_model(arch, num_classes)
                    r = run_channeled(model, sigma, C_list, tiers, n_train, EPOCHS,
                                       device, private_dataset, test_loader)

                save_json(r, fname)
                accs_by_seed.append(r)

            accs = [r["test_acc"] for r in accs_by_seed]
            tab11[ir][method_name] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
            }

        gap = tab11[ir]["channeled"]["mean"] - tab11[ir]["standard"]["mean"]
        print(f"  IR={ir}: standard={tab11[ir]['standard']['mean']:.4f}, "
              f"channeled={tab11[ir]['channeled']['mean']:.4f}, gap={gap:+.4f}")

    save_json({str(k): v for k, v in tab11.items()},
              os.path.join(out_dir, "tab11.json"))
    return tab11


# ---------------------------------------------------------------------------
# 5.5: Architecture
# ---------------------------------------------------------------------------

def ablation_architecture(device, data_root=DATA_ROOT, results_dir=RESULTS_DIR):
    """Tab 12: Test accuracy by architecture × method (CIFAR-10 balanced, ε=3)."""
    out_dir = os.path.join(results_dir, "5.5_architecture")
    os.makedirs(out_dir, exist_ok=True)
    tab12 = {}
    K = 3

    for arch in ["wrn28-2", "resnet20"]:
        tab12[arch] = {}
        for method_name in ["standard", "channeled"]:
            accs_by_seed = []
            for seed in range(N_SEEDS):
                fname = os.path.join(out_dir, f"{arch}_{method_name}_seed{seed}.json")
                if os.path.exists(fname):
                    from src.evaluation import load_json
                    r = load_json(fname)
                    accs_by_seed.append(r)
                    continue

                print(f"  [Exp5.5] arch={arch}, method={method_name}, seed={seed}")
                (data, n_train, num_classes, class_counts, private_targets,
                 private_loader, public_loader, test_loader, sigma, pub_model,
                 private_dataset) = _load_base_data(
                     "cifar10", 1.0, arch, data_root, device, seed)

                if method_name == "standard":
                    pub_norms = compute_per_sample_gradient_norms(
                        pub_model, private_loader, device
                    )
                    C = float(np.median(pub_norms))
                    model = make_model(arch, num_classes)
                    r = run_standard(model, sigma, C, n_train, EPOCHS,
                                     device, private_loader, test_loader)
                else:
                    tiers = assign_tiers("A", private_targets, class_counts, K=K)
                    C_list = calibrate_clipping_bounds(pub_model, private_loader, tiers,
                                                        K=K, c=2.0, device=device)
                    model = make_model(arch, num_classes)
                    r = run_channeled(model, sigma, C_list, tiers, n_train, EPOCHS,
                                       device, private_dataset, test_loader)

                save_json(r, fname)
                accs_by_seed.append(r)

            accs = [r["test_acc"] for r in accs_by_seed]
            tab12[arch][method_name] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
            }
            print(f"  {arch}/{method_name}: {tab12[arch][method_name]['mean']:.4f}")

    save_json(tab12, os.path.join(out_dir, "tab12.json"))
    return tab12


def run_all(device: torch.device = None, data_root: str = DATA_ROOT):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp5] Using device: {device}")

    tab8 = ablation_num_tiers(device, data_root)
    tab9 = ablation_tier_strategy(device, data_root)
    tab10 = ablation_calibration(device, data_root)
    tab11 = ablation_imbalance_ratio(device, data_root)
    tab12 = ablation_architecture(device, data_root)

    print("\n=== Exp5 Summary ===")
    print("Tab 8 (K vs accuracy):", {k: f"{v['mean']:.4f}±{v['std']:.4f}"
                                      for k, v in tab8.items()})
    print("Tab 9 (strategy):", {s: f"{v['mean']:.4f}±{v['std']:.4f}"
                                  for s, v in tab9.items()})
    return tab8, tab9, tab10, tab11, tab12


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 5: Ablations")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ablation", default="all",
                        help="5.1, 5.2, 5.3, 5.4, 5.5, or all")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.ablation == "5.1":
        ablation_num_tiers(device, args.data_root)
    elif args.ablation == "5.2":
        ablation_tier_strategy(device, args.data_root)
    elif args.ablation == "5.3":
        ablation_calibration(device, args.data_root)
    elif args.ablation == "5.4":
        ablation_imbalance_ratio(device, args.data_root)
    elif args.ablation == "5.5":
        ablation_architecture(device, args.data_root)
    else:
        run_all(device, args.data_root)
