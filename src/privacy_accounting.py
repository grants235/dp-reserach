"""
Privacy accounting utilities for Channeled DP-SGD experiments.

Computes:
- Noise multiplier σ from (ε, δ, T, q) via RDP accountant
- ε from (σ, δ, T, q)
- Per-instance ε via the quadratic scaling law (Thudi et al., 2022)
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# σ ↔ ε conversion via Opacus RDP accountant
# ---------------------------------------------------------------------------

def compute_sigma(
    target_eps: float,
    target_delta: float,
    sample_rate: float,
    steps: int,
    accountant: str = "rdp",
    eps_error: float = 0.01,
) -> float:
    """
    Find the noise multiplier σ such that (σ, sample_rate, steps) achieves
    (target_eps, target_delta)-DP.

    Uses Opacus's get_noise_multiplier utility with the RDP accountant.
    σ does NOT depend on the clipping bound C.
    """
    from opacus.accountants.utils import get_noise_multiplier
    sigma = get_noise_multiplier(
        target_epsilon=target_eps,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=steps,
        accountant=accountant,
        eps_error=eps_error,
    )
    return sigma


def compute_epsilon(
    sigma: float,
    target_delta: float,
    sample_rate: float,
    steps: int,
    accountant: str = "rdp",
) -> float:
    """
    Compute the (ε, δ)-DP guarantee for a given σ.
    """
    from opacus.accountants import RDPAccountant
    from opacus.accountants import GaussianAccountant

    if accountant == "rdp":
        acc = RDPAccountant()
    else:
        acc = GaussianAccountant()

    acc.history = [(sigma, sample_rate, steps)]
    return acc.get_epsilon(delta=target_delta)


def compute_sigma_table(
    epsilons: list,
    delta: float,
    n_train: int,
    batch_size: int,
    epochs: int,
) -> dict:
    """
    Compute σ for each target ε in epsilons.

    Returns dict mapping ε -> σ, also records q and T.
    """
    q = batch_size / n_train
    T = epochs * int(np.ceil(n_train / batch_size))
    result = {}
    for eps in epsilons:
        sigma = compute_sigma(
            target_eps=eps,
            target_delta=delta,
            sample_rate=q,
            steps=T,
        )
        result[eps] = dict(sigma=sigma, q=q, T=T, n_train=n_train)
    return result


# ---------------------------------------------------------------------------
# Per-instance ε (Thudi et al., 2022 quadratic scaling law)
# ---------------------------------------------------------------------------

def per_instance_epsilon_quadratic(
    rms_clipped_norms: np.ndarray,
    C: float,
    global_eps: float,
) -> np.ndarray:
    """
    Theoretical per-instance ε via the quadratic scaling law:

        ε_i ≈ (Δ̄_i / C)² · ε

    where Δ̄_i is the RMS clipped gradient norm:

        Δ̄_i = sqrt( (1/T) Σ_t ||ḡ_i(θ_t)||² )

    and ε is the global (ε, δ)-DP guarantee.

    Args:
        rms_clipped_norms: array of shape (n,) containing Δ̄_i for each sample
        C: global clipping bound used in standard DP-SGD
        global_eps: global ε from (ε, δ)-DP guarantee

    Returns:
        per_instance_eps: array of shape (n,) with ε_i for each sample
    """
    ratio_sq = (rms_clipped_norms / C) ** 2
    # Clip ratio to [0, 1] – the bound only makes sense for ratio ≤ 1
    ratio_sq = np.clip(ratio_sq, 0.0, 1.0)
    return ratio_sq * global_eps


def thudi_per_instance_epsilon(
    clipped_norms_per_step: np.ndarray,
    C: float,
    sigma: float,
    sample_rate: float,
    n_train: int,
    target_delta: float,
    composition_constant: float = 3.0,
) -> np.ndarray:
    """
    Per-instance ε based on Thudi et al. (2022) Theorem 3.2 + 3.3.

    Per-step RDP bound for sample i (Gaussian mechanism, Poisson subsampling):
        ρ_step_i(α) ≈ (Δ_i / C)² · ρ_step(α)
    where ρ_step(α) is the standard per-step RDP for the Gaussian mechanism.

    Composition over T steps (Theorem 3.3):
        ρ_total_i = T · ρ_step_i
    Converting RDP to (ε, δ)-DP:
        ε_i(δ) = min_α [ ρ_total_i(α) + log(1/δ) / (α - 1) ]

    Args:
        clipped_norms_per_step: array of shape (n, T) – per-step clipped norms
        C: clipping bound
        sigma: noise multiplier
        sample_rate: Poisson sampling rate q
        n_train: total training set size
        target_delta: δ for ε conversion
        composition_constant: p = composition_constant * T (Thudi Thm 3.3)

    Returns:
        per_instance_eps: array of shape (n,) with ε_i for each sample
    """
    from opacus.accountants import RDPAccountant

    n, T = clipped_norms_per_step.shape

    # RMS clipped norm per sample
    rms_norms = np.sqrt(np.mean(clipped_norms_per_step ** 2, axis=1))  # (n,)

    # Sensitivity ratio
    ratio_sq = np.clip((rms_norms / C) ** 2, 0.0, 1.0)  # (n,)

    per_eps = np.zeros(n)
    for i in range(n):
        # Effective sigma_i: since sensitivity is Δ_i = ratio_i * C and noise is
        # σ * C, the effective noise multiplier is σ / ratio_i (ratio ≥ 0).
        # But for ratio_i = 0, epsilon_i = 0.
        if ratio_sq[i] < 1e-10:
            per_eps[i] = 0.0
            continue

        # Per-instance effective sigma: noise_std / sensitivity = σ*C / (ratio_i*C)
        # = σ / ratio_i (higher ratio → lower effective sigma → higher ε)
        effective_sigma = sigma / np.sqrt(ratio_sq[i])

        acc = RDPAccountant()
        acc.history = [(effective_sigma, sample_rate, T)]
        per_eps[i] = acc.get_epsilon(delta=target_delta)

    return per_eps


# ---------------------------------------------------------------------------
# Mahalanobis deviation
# ---------------------------------------------------------------------------

def mahalanobis_deviation(grad_vecs: np.ndarray) -> np.ndarray:
    """
    Compute Mahalanobis-like deviation for each sample:
        δ̂_i = ||∇ℓ(θ*, z_i) - (1/n) Σ_j ∇ℓ(θ*, z_j)||²

    This is equivalent to the squared Euclidean distance to the mean gradient,
    which is a proxy for gradient norm deviation (cheaper than full Mahalanobis).

    Args:
        grad_vecs: array of shape (n, D)
    Returns:
        deviations: array of shape (n,)
    """
    mean_grad = grad_vecs.mean(axis=0, keepdims=True)  # (1, D)
    diff = grad_vecs - mean_grad  # (n, D)
    return np.sum(diff ** 2, axis=1)  # (n,)


def lt_iqr_score(loss_traces: np.ndarray) -> np.ndarray:
    """
    LT-IQR: interquartile range of per-sample loss across epochs.
    High IQR indicates unstable training (typically tail/hard samples).

    Args:
        loss_traces: array of shape (n, T) – per-epoch training loss per sample
    Returns:
        iqr_scores: array of shape (n,)
    """
    q75 = np.percentile(loss_traces, 75, axis=1)
    q25 = np.percentile(loss_traces, 25, axis=1)
    return q75 - q25
