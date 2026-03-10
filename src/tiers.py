"""
Tier assignment strategies for Channeled DP-SGD.

Strategy A: Class frequency (for long-tailed data)
Strategy B: Public-set k-NN density estimation
Strategy C: Random assignment (control)
"""

import numpy as np
from typing import Optional


def tier_by_class_frequency(
    labels: np.ndarray,
    class_counts: np.ndarray,
    K: int = 3,
) -> np.ndarray:
    """
    Strategy A: Assign tiers based on class frequency.

    Classes are sorted by count (descending). The top K roughly equal groups
    become tiers 0, 1, ..., K-1. Tier 0 = most frequent (head); Tier K-1 =
    least frequent (tail).

    For CIFAR-10-LT with K=3:
        Tier 0 = 3 most frequent classes
        Tier 1 = middle 4 classes
        Tier 2 = 3 least frequent classes

    For balanced CIFAR-10, class_counts are approximately equal, so tiers are
    assigned by class index modulo K (arbitrary but fixed).

    Args:
        labels: array of shape (n,) with class labels
        class_counts: array of shape (num_classes,) with per-class sample counts
        K: number of tiers

    Returns:
        tiers: array of shape (n,) with tier indices in {0, ..., K-1}
    """
    num_classes = len(class_counts)
    sorted_classes = np.argsort(class_counts)[::-1]  # descending frequency

    classes_per_tier = num_classes // K
    class_to_tier = np.zeros(num_classes, dtype=int)

    for i, c in enumerate(sorted_classes):
        class_to_tier[c] = min(i // classes_per_tier, K - 1)

    labels = np.asarray(labels)
    return class_to_tier[labels]


def tier_by_density(
    features_public: np.ndarray,
    features_all: np.ndarray,
    K: int = 3,
    k_nn: int = 10,
) -> np.ndarray:
    """
    Strategy B: Assign tiers via k-NN density from public features.

    Density is estimated as 1 / (mean k-NN distance + ε).
    Samples are partitioned by density quantiles:
        Tier 0 = highest density (head region)
        Tier K-1 = lowest density (tail region)

    Args:
        features_public: array of shape (n_public, D) – public-set embeddings
        features_all: array of shape (n_all, D) – all training embeddings
        K: number of tiers
        k_nn: number of neighbors for density estimation

    Returns:
        tiers: array of shape (n_all,) with tier indices in {0, ..., K-1}
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k_nn, algorithm="auto", n_jobs=-1)
    nn.fit(features_public)
    distances, _ = nn.kneighbors(features_all)  # (n_all, k_nn)
    densities = 1.0 / (distances.mean(axis=1) + 1e-8)  # (n_all,)

    # Partition by density quantiles; tier 0 = high density (head)
    quantiles = np.quantile(densities, np.linspace(0, 1, K + 1))
    # digitize returns 1-indexed; subtract 1 for 0-indexed tiers
    tiers = np.digitize(densities, quantiles[1:-1])  # in {0, ..., K-1}
    # Invert so tier 0 = highest density
    tiers = K - 1 - tiers
    return tiers.astype(int)


def tier_by_random(
    n_samples: int,
    tier_sizes: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """
    Strategy C: Random tier assignment preserving tier size distribution.

    Args:
        n_samples: total number of samples
        tier_sizes: array of shape (K,) specifying number of samples per tier
                    (must sum to n_samples)
        seed: random seed for reproducibility

    Returns:
        tiers: array of shape (n_samples,) with tier indices in {0, ..., K-1}
    """
    assert tier_sizes.sum() == n_samples, (
        f"tier_sizes sum {tier_sizes.sum()} != n_samples {n_samples}"
    )
    rng = np.random.default_rng(seed)
    tiers = np.empty(n_samples, dtype=int)
    perm = rng.permutation(n_samples)
    offset = 0
    for k, size in enumerate(tier_sizes):
        tiers[perm[offset:offset + size]] = k
        offset += size
    return tiers


def get_tier_sizes(tiers: np.ndarray, K: int) -> np.ndarray:
    """Return array of shape (K,) with number of samples per tier."""
    return np.bincount(tiers, minlength=K)


def assign_tiers(
    strategy: str,
    labels: np.ndarray,
    class_counts: np.ndarray,
    K: int = 3,
    features_public: Optional[np.ndarray] = None,
    features_all: Optional[np.ndarray] = None,
    k_nn: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """
    Dispatch function: assign tiers using the specified strategy.

    Args:
        strategy: 'A' (class frequency), 'B' (density), 'C' (random)
        labels: array of class labels (n,)
        class_counts: per-class sample counts (num_classes,)
        K: number of tiers
        features_public: required for strategy B
        features_all: required for strategy B
        k_nn: k for strategy B k-NN
        seed: random seed for strategy C

    Returns:
        tiers: array of shape (n,) in {0, ..., K-1}
    """
    if strategy == "A":
        return tier_by_class_frequency(labels, class_counts, K)
    elif strategy == "B":
        if features_public is None or features_all is None:
            raise ValueError("Strategy B requires features_public and features_all")
        return tier_by_density(features_public, features_all, K, k_nn)
    elif strategy == "C":
        # Use same tier-size distribution as Strategy A
        tiers_a = tier_by_class_frequency(labels, class_counts, K)
        tier_sizes = get_tier_sizes(tiers_a, K)
        return tier_by_random(len(labels), tier_sizes, seed=seed)
    else:
        raise ValueError(f"Unknown tier strategy: {strategy!r}")
