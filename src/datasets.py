"""
Dataset utilities for Channeled DP-SGD experiments.

Implements:
- CIFAR-10 / CIFAR-100 loading
- Long-tailed CIFAR-10 construction (exponential decay, Cui et al. 2019)
- Fixed public/private split (10% public per class, same across all methods/seeds)
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Standard transforms
# ---------------------------------------------------------------------------

def get_transforms(augment: bool = True):
    """Return (train_transform, test_transform) for CIFAR-10/100."""
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([transforms.ToTensor(), normalize])
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])
    return train_tf, test_tf


def get_cifar100_transforms(augment: bool = True):
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761],
    )
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([transforms.ToTensor(), normalize])
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])
    return train_tf, test_tf


# ---------------------------------------------------------------------------
# Long-tailed CIFAR-10
# ---------------------------------------------------------------------------

class IndexedDataset(Dataset):
    """Wraps a dataset and exposes the original dataset-level index."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.dataset[self.indices[i]]
        return x, y, self.indices[i]  # (image, label, global_idx)

    @property
    def targets(self):
        return [self.dataset.targets[idx] for idx in self.indices]


def make_cifar10_lt_indices(full_targets, imbalance_ratio: float, seed: int = 42):
    """
    Compute per-class sample indices for long-tailed CIFAR-10.

    n_c = 5000 * imbalance_ratio^{-c/9}  for class c in {0, ..., 9}.
    Class 0 has 5000 examples; class 9 has 5000 / imbalance_ratio.

    Returns list of indices (into the original full training set).
    """
    rng = np.random.default_rng(seed)
    targets = np.asarray(full_targets)
    num_classes = 10

    selected = []
    for c in range(num_classes):
        n_c = int(5000 * (imbalance_ratio ** (-c / 9)))
        n_c = max(n_c, 1)
        class_idx = np.where(targets == c)[0]
        assert len(class_idx) >= n_c, (
            f"Class {c}: requested {n_c} but only {len(class_idx)} available"
        )
        chosen = rng.choice(class_idx, size=n_c, replace=False)
        selected.append(chosen)

    return np.concatenate(selected)


# ---------------------------------------------------------------------------
# Public / private split
# ---------------------------------------------------------------------------

def make_public_private_split(indices, targets, public_frac: float = 0.1, seed: int = 42):
    """
    Split indices into public (10% per class) and private (90% per class).

    The split is done stratified by class so each class contributes equally
    to the public set. Returns (public_indices, private_indices).

    The same split must be used across all methods and seeds.
    """
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices)
    targets = np.asarray(targets)
    class_labels = targets[np.arange(len(targets))]  # already aligned

    # Build per-class subsets
    unique_classes = np.unique(class_labels)
    public_idx, private_idx = [], []

    for c in unique_classes:
        mask = (class_labels == c)
        cls_indices = indices[mask]
        n_public = max(1, int(len(cls_indices) * public_frac))
        perm = rng.permutation(len(cls_indices))
        public_idx.append(cls_indices[perm[:n_public]])
        private_idx.append(cls_indices[perm[n_public:]])

    return np.concatenate(public_idx), np.concatenate(private_idx)


# ---------------------------------------------------------------------------
# Main data-loading entry point
# ---------------------------------------------------------------------------

def load_split_file(data_root: str, dataset_name: str, imbalance_ratio: float) -> dict:
    """
    Load pre-computed split from data/splits/<key>.npz if it exists.
    Returns None if the file is not found (caller falls back to on-the-fly computation).
    """
    key = f"{dataset_name}_IR{imbalance_ratio:.0f}"
    path = os.path.join(data_root, "splits", f"{key}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}


def load_datasets(
    dataset_name: str,
    data_root: str = "./data",
    imbalance_ratio: float = 1.0,
    public_frac: float = 0.1,
    split_seed: int = 42,
):
    """
    Load and split a dataset according to the spec.

    Returns a dict with keys:
        public_dataset  : IndexedDataset  (public split, no augmentation)
        private_dataset : IndexedDataset  (private split, with augmentation)
        test_dataset    : standard Dataset (full test set)
        public_indices  : np.ndarray
        private_indices : np.ndarray
        class_counts    : np.ndarray of shape (num_classes,) – private counts
        num_classes     : int
        n_train         : int  (= len(private_dataset))
    """
    os.makedirs(data_root, exist_ok=True)

    if dataset_name.startswith("cifar10"):
        train_tf_aug, test_tf = get_transforms(augment=True)
        train_tf_noaug, _ = get_transforms(augment=False)

        full_train = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=train_tf_aug
        )
        full_train_noaug = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=train_tf_noaug
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=test_tf
        )
        num_classes = 10

    elif dataset_name == "cifar100":
        train_tf_aug, test_tf = get_cifar100_transforms(augment=True)
        train_tf_noaug, _ = get_cifar100_transforms(augment=False)

        full_train = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=train_tf_aug
        )
        full_train_noaug = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=train_tf_noaug
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=test_tf
        )
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ------------------------------------------------------------------
    # Use pre-computed split from setup_data.py if available;
    # fall back to on-the-fly computation (results are identical when
    # using the same split_seed and public_frac).
    # ------------------------------------------------------------------
    precomputed = load_split_file(data_root, dataset_name, imbalance_ratio)

    if precomputed is not None:
        public_indices  = precomputed["public_indices"]
        private_indices = precomputed["private_indices"]
        class_counts    = precomputed["class_counts"]
    else:
        # On-the-fly: compute LT indices + public/private split
        full_targets = np.array(full_train.targets)

        if dataset_name.startswith("cifar10") and imbalance_ratio > 1.0:
            lt_indices = make_cifar10_lt_indices(
                full_targets, imbalance_ratio, seed=split_seed
            )
        else:
            lt_indices = np.arange(len(full_train))

        lt_targets = full_targets[lt_indices]
        public_indices, private_indices = make_public_private_split(
            lt_indices, lt_targets, public_frac=public_frac, seed=split_seed
        )
        private_targets_arr = full_targets[private_indices]
        class_counts = np.bincount(private_targets_arr, minlength=num_classes)

    # Build IndexedDatasets
    public_dataset = IndexedDataset(full_train_noaug, public_indices)
    private_dataset = IndexedDataset(full_train, private_indices)

    return dict(
        public_dataset=public_dataset,
        private_dataset=private_dataset,
        test_dataset=test_dataset,
        public_indices=public_indices,
        private_indices=private_indices,
        class_counts=class_counts,
        num_classes=num_classes,
        n_train=len(private_dataset),
    )


def make_data_loaders(
    data_dict: dict,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """Build (private_loader, public_loader, test_loader) from a data_dict."""
    private_loader = DataLoader(
        data_dict["private_dataset"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    public_loader = DataLoader(
        data_dict["public_dataset"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        data_dict["test_dataset"],
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return private_loader, public_loader, test_loader


def make_poisson_loader(dataset, sample_rate: float, num_workers: int = 4):
    """
    DataLoader with Poisson subsampling (each sample included independently
    with probability sample_rate). Expected batch size = sample_rate * len(dataset).
    """
    from opacus.data_loader import DPDataLoader
    return DPDataLoader.from_data_loader(
        DataLoader(
            dataset,
            batch_size=int(sample_rate * len(dataset)),
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        ),
        distributed=False,
    )


class TieredDataset(Dataset):
    """
    Wraps any dataset and appends per-sample tier labels to each batch.

    __getitem__ returns (x, y, tier) so the ChanneledDPSGDTrainer can access
    tier labels directly without global-index bookkeeping.

    Args:
        dataset: base dataset whose __getitem__ returns at least (x, y)
        tier_labels: array of shape (len(dataset),) with tier indices
    """

    def __init__(self, dataset, tier_labels: np.ndarray):
        self.dataset = dataset
        self.tier_labels = np.asarray(tier_labels, dtype=np.int64)
        assert len(tier_labels) == len(dataset), (
            f"tier_labels length {len(tier_labels)} != dataset length {len(dataset)}"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        x, y = item[0], item[1]
        return x, y, int(self.tier_labels[i])

    @property
    def targets(self):
        return [self.dataset[i][1] for i in range(len(self.dataset))]
