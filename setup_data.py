"""
setup_data.py – Initialize data directory for Channeled DP-SGD experiments.

Downloads CIFAR-10 and CIFAR-100, pre-computes and saves all dataset splits
that must remain fixed across experiments:
  - Public/private splits for each dataset variant
  - Long-tailed CIFAR-10 indices (IR = 10, 50, 100)

Saved to: data/splits/<dataset_key>.npz

Run once before any experiment:
    python setup_data.py [--data_root ./data]
"""

import os
import sys
import argparse
import numpy as np

SPLIT_SEED = 42       # never change – all experiments depend on this
PUBLIC_FRAC = 0.10    # 10% of each class → public

IMBALANCE_RATIOS = [1.0, 10.0, 50.0, 100.0]  # IR=1 means balanced

DATASET_CONFIGS = [
    # (name, imbalance_ratio)
    ("cifar10",   1.0),
    ("cifar10",  10.0),
    ("cifar10",  50.0),
    ("cifar10", 100.0),
    ("cifar100",  1.0),
]


def download_datasets(data_root: str):
    """Download CIFAR-10 and CIFAR-100 via torchvision (idempotent)."""
    import torchvision
    import torchvision.transforms as T

    tf = T.ToTensor()
    print("Downloading CIFAR-10 ...")
    torchvision.datasets.CIFAR10(root=data_root, train=True,  download=True, transform=tf)
    torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf)
    print("Downloading CIFAR-100 ...")
    torchvision.datasets.CIFAR100(root=data_root, train=True,  download=True, transform=tf)
    torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=tf)
    print("Downloads complete.")


def make_lt_indices(targets: np.ndarray, imbalance_ratio: float, seed: int) -> np.ndarray:
    """
    Sample long-tailed indices from a full CIFAR-10 training set.
    n_c = floor(5000 * IR^{-c/9}) for class c in {0,...,9}.
    """
    rng = np.random.default_rng(seed)
    selected = []
    for c in range(10):
        n_c = max(1, int(5000 * (imbalance_ratio ** (-c / 9))))
        cls_idx = np.where(targets == c)[0]
        assert len(cls_idx) >= n_c, f"Class {c}: need {n_c} but only {len(cls_idx)} available"
        chosen = rng.choice(cls_idx, size=n_c, replace=False)
        selected.append(chosen)
    indices = np.concatenate(selected)
    return indices


def make_split(indices: np.ndarray, targets: np.ndarray,
               public_frac: float, seed: int):
    """
    Stratified public/private split.
    Returns (public_indices, private_indices) both into the full original dataset.
    """
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    class_labels = targets[indices]

    pub, priv = [], []
    for c in np.unique(class_labels):
        mask = (class_labels == c)
        cls_idx = indices[mask]
        n_pub = max(1, int(len(cls_idx) * public_frac))
        perm = rng.permutation(len(cls_idx))
        pub.append(cls_idx[perm[:n_pub]])
        priv.append(cls_idx[perm[n_pub:]])

    return np.concatenate(pub), np.concatenate(priv)


def build_splits(data_root: str):
    """Pre-compute all splits and save to data/splits/<key>.npz."""
    import torchvision
    import torchvision.transforms as T

    splits_dir = os.path.join(data_root, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    tf = T.ToTensor()

    # Load raw datasets (no transform needed for index computation)
    cifar10_train = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=tf)
    cifar100_train = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=False, transform=tf)

    c10_targets  = np.array(cifar10_train.targets)
    c100_targets = np.array(cifar100_train.targets)

    for (dname, ir) in DATASET_CONFIGS:
        key = f"{dname}_IR{ir:.0f}"
        out_path = os.path.join(splits_dir, f"{key}.npz")
        if os.path.exists(out_path):
            print(f"  {key}: split already exists, skipping.")
            continue

        if dname == "cifar10":
            full_targets = c10_targets
            n_full = len(c10_targets)
        else:
            full_targets = c100_targets
            n_full = len(c100_targets)
            if ir != 1.0:
                print(f"  Skipping {key}: long-tail not used for CIFAR-100.")
                continue

        # Long-tailed subsetting (IR > 1 only for CIFAR-10)
        if ir > 1.0:
            lt_indices = make_lt_indices(full_targets, ir, seed=SPLIT_SEED)
        else:
            lt_indices = np.arange(n_full)

        lt_targets = full_targets[lt_indices]

        # Class counts in this split
        num_classes = 10 if dname == "cifar10" else 100
        class_counts = np.bincount(lt_targets, minlength=num_classes)

        # Public/private split
        pub_idx, priv_idx = make_split(lt_indices, full_targets,
                                        PUBLIC_FRAC, SPLIT_SEED)
        pub_targets  = full_targets[pub_idx]
        priv_targets = full_targets[priv_idx]

        np.savez(
            out_path,
            lt_indices=lt_indices,
            public_indices=pub_idx,
            private_indices=priv_idx,
            lt_targets=lt_targets,
            public_targets=pub_targets,
            private_targets=priv_targets,
            class_counts=class_counts,
            imbalance_ratio=np.float64(ir),
            split_seed=np.int64(SPLIT_SEED),
            public_frac=np.float64(PUBLIC_FRAC),
        )

        priv_per_class = np.bincount(priv_targets, minlength=num_classes)
        pub_per_class  = np.bincount(pub_targets, minlength=num_classes)
        print(
            f"  {key}: total={len(lt_indices)}, "
            f"private={len(priv_idx)}, public={len(pub_idx)}, "
            f"classes={num_classes}"
        )
        if ir > 1.0:
            print(
                f"    class 0: {priv_per_class[0]} private / {pub_per_class[0]} public  |  "
                f"class 9: {priv_per_class[9]} private / {pub_per_class[9]} public"
            )

    print(f"All splits saved to {splits_dir}")


def verify_splits(data_root: str) -> bool:
    """Sanity-check all saved splits. Returns True if all pass."""
    splits_dir = os.path.join(data_root, "splits")
    ok = True

    for (dname, ir) in DATASET_CONFIGS:
        if dname == "cifar100" and ir != 1.0:
            continue
        key = f"{dname}_IR{ir:.0f}"
        path = os.path.join(splits_dir, f"{key}.npz")
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            ok = False
            continue

        d = np.load(path)

        # Check no overlap between public and private
        pub = set(d["public_indices"].tolist())
        priv = set(d["private_indices"].tolist())
        overlap = pub & priv
        if overlap:
            print(f"  ERROR {key}: {len(overlap)} indices appear in both public and private!")
            ok = False
        else:
            print(f"  OK {key}: {len(priv)} private, {len(pub)} public, no overlap")

        # Check LT class distribution
        if ir > 1.0:
            targets = d["lt_targets"]
            for c in range(10):
                n_c = (targets == c).sum()
                expected = max(1, int(5000 * (ir ** (-c / 9))))
                # Allow ±1 due to rounding
                if abs(n_c - expected) > 1:
                    print(f"  WARNING {key}: class {c} has {n_c} samples, expected ~{expected}")

    return ok


def print_summary(data_root: str):
    """Print a table of all splits."""
    splits_dir = os.path.join(data_root, "splits")
    print("\n" + "=" * 70)
    print(f"{'Dataset':<25} {'IR':>6} {'Total':>8} {'Private':>9} {'Public':>8}")
    print("=" * 70)
    for (dname, ir) in DATASET_CONFIGS:
        if dname == "cifar100" and ir != 1.0:
            continue
        key = f"{dname}_IR{ir:.0f}"
        path = os.path.join(splits_dir, f"{key}.npz")
        if not os.path.exists(path):
            print(f"  {key:<25} MISSING")
            continue
        d = np.load(path)
        total = len(d["lt_indices"])
        priv  = len(d["private_indices"])
        pub   = len(d["public_indices"])
        print(f"  {dname:<23} {ir:>6.0f} {total:>8,} {priv:>9,} {pub:>8,}")
    print("=" * 70)


def load_split(data_root: str, dataset_name: str, imbalance_ratio: float) -> dict:
    """
    Load pre-computed split for a (dataset, IR) pair.

    Returns dict with keys:
        lt_indices, public_indices, private_indices,
        lt_targets, public_targets, private_targets,
        class_counts, imbalance_ratio, split_seed, public_frac
    """
    key = f"{dataset_name}_IR{imbalance_ratio:.0f}"
    path = os.path.join(data_root, "splits", f"{key}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Split not found: {path}\n"
            f"Run `python setup_data.py --data_root {data_root}` first."
        )
    d = np.load(path)
    return {k: d[k] for k in d.files}


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets and pre-compute fixed splits."
    )
    parser.add_argument("--data_root", default="./data",
                        help="Directory to store datasets and splits (default: ./data)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip torchvision download (if datasets already present)")
    parser.add_argument("--verify_only", action="store_true",
                        help="Only verify existing splits, do not re-compute")
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)

    if args.verify_only:
        print("Verifying existing splits ...")
        ok = verify_splits(args.data_root)
        print_summary(args.data_root)
        sys.exit(0 if ok else 1)

    # 1. Download
    if not args.skip_download:
        print(f"\nStep 1/3: Downloading datasets to {args.data_root} ...")
        try:
            download_datasets(args.data_root)
        except Exception as e:
            print(f"Download failed: {e}")
            sys.exit(1)
    else:
        print("Step 1/3: Skipping download (--skip_download set).")

    # 2. Build splits
    print(f"\nStep 2/3: Computing fixed splits (seed={SPLIT_SEED}) ...")
    build_splits(args.data_root)

    # 3. Verify
    print("\nStep 3/3: Verifying splits ...")
    ok = verify_splits(args.data_root)
    print_summary(args.data_root)

    if ok:
        print("\nData setup complete. Ready to run experiments.")
        print(f"  python run_all.py --data_root {args.data_root} --priority P0")
    else:
        print("\nERROR: Some splits failed verification. Re-run setup_data.py.")
        sys.exit(1)


if __name__ == "__main__":
    main()
