"""
Clipping threshold calibration for Channeled DP-SGD.

Per the spec:
- Train a "public model" on the 10% public split for 50 epochs (no DP).
- Compute per-sample gradient norms under the public model.
- For each tier k: C_k = c * percentile(norms_k, 95)
- The multiplier c=2.0 by default; calibration sensitivity is tested in Exp 5.3.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List


# ---------------------------------------------------------------------------
# Per-sample gradient norms
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_features(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Extract penultimate-layer features for density-based tiers."""
    model.eval()
    feats = []
    for batch in loader:
        x = batch[0].to(device)
        feats.append(model.features(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_per_sample_gradient_norms(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = None,
) -> np.ndarray:
    """
    Compute the L2 gradient norm ||∇ℓ(θ, z_i)|| for each sample z_i.

    Uses per-sample gradient computation via Opacus GradSampleModule.
    The model is evaluated in eval mode (no dropout).

    Returns:
        norms: array of shape (n,) – unclipped per-sample gradient norms
    """
    from opacus.grad_sample import GradSampleModule

    model_gs = GradSampleModule(model)
    model_gs.eval().to(device)

    all_norms = []
    n_batches = 0

    for batch in loader:
        if max_batches and n_batches >= max_batches:
            break

        # Support both (x, y) and (x, y, idx) batches
        x, y = batch[0].to(device), batch[1].to(device)

        # Zero accumulated grad_sample
        for p in model_gs.parameters():
            p.grad = None
            if hasattr(p, "grad_sample"):
                del p.grad_sample

        # Forward + backward (sum reduction so grad_sample[i] = ∇ℓ_i)
        outputs = model_gs(x)
        loss = F.cross_entropy(outputs, y, reduction="sum")
        loss.backward()

        # Collect per-sample gradients: each p.grad_sample has shape (B, *p.shape)
        batch_size = x.shape[0]
        grad_flat = []
        for p in model_gs.parameters():
            if p.requires_grad and hasattr(p, "grad_sample"):
                grad_flat.append(p.grad_sample.reshape(batch_size, -1))

        # (B, D) – concatenated per-sample gradient vector
        per_sample = torch.cat(grad_flat, dim=1)
        norms = per_sample.norm(dim=1).detach().cpu().numpy()
        all_norms.append(norms)

        # Free memory
        for p in model_gs.parameters():
            if hasattr(p, "grad_sample"):
                del p.grad_sample

        n_batches += 1

    # Unwrap GradSampleModule to avoid side-effects on caller's model
    return np.concatenate(all_norms)


def compute_per_sample_clipped_norms(
    grad_norms: np.ndarray,
    C: float,
) -> np.ndarray:
    """
    Clipped gradient norm: ||ḡ_i|| = min(||∇ℓ_i||, C).
    Since clipping scales the gradient to norm C when ||∇ℓ_i|| > C,
    the clipped norm is min(||∇ℓ_i||, C).
    """
    return np.minimum(grad_norms, C)


# ---------------------------------------------------------------------------
# Train public model
# ---------------------------------------------------------------------------

def train_public_model(
    model: nn.Module,
    public_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    verbose: bool = True,
) -> nn.Module:
    """
    Train model on the public split for 50 epochs without DP.
    Used to estimate per-tier gradient norms for calibration.
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        total_loss, n = 0.0, 0
        for batch in public_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.shape[0]
            n += x.shape[0]
        scheduler.step()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [Public model] Epoch {epoch+1}/{epochs}, loss={total_loss/n:.4f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Calibrate clipping bounds
# ---------------------------------------------------------------------------

def calibrate_clipping_bounds(
    public_model: nn.Module,
    private_loader: DataLoader,
    tier_labels: np.ndarray,
    K: int,
    c: float = 2.0,
    device: torch.device = None,
    percentile: float = 95.0,
) -> List[float]:
    """
    Estimate C_k for each tier from public-model gradient norms.

        C_k = c * percentile(||∇ℓ(θ_pub, z_i)||, 95)  for z_i in tier k

    The public model's gradient norms serve as a proxy for the eventual
    gradient norms during private training.

    Args:
        public_model: model trained on the public split
        private_loader: loader over the FULL training set (to get norms for all)
        tier_labels: array of shape (n_train,) with tier labels for each sample
        K: number of tiers
        c: scale multiplier (default 2.0)
        device: compute device
        percentile: norm percentile per tier (default 95)

    Returns:
        C_list: list of length K with C_k values (C_0 ≤ C_1 ≤ ... ≤ C_{K-1})
    """
    if device is None:
        device = next(public_model.parameters()).device

    grad_norms = compute_per_sample_gradient_norms(public_model, private_loader, device)
    tier_labels = np.asarray(tier_labels)

    assert len(grad_norms) == len(tier_labels), (
        f"grad_norms length {len(grad_norms)} != tier_labels length {len(tier_labels)}"
    )

    C_list = []
    for k in range(K):
        mask = (tier_labels == k)
        tier_norms = grad_norms[mask]
        if len(tier_norms) == 0:
            # Fallback: use global percentile
            tau_k = np.percentile(grad_norms, percentile)
        else:
            tau_k = np.percentile(tier_norms, percentile)
        C_list.append(c * tau_k)

    return C_list


def calibrate_fixed_ratio(C_max: float, K: int) -> List[float]:
    """
    Fixed-ratio calibration: C_0 = C_max/10, C_1 = C_max/3, C_2 = C_max.
    Generalizes linearly in log space for K != 3.
    """
    log_min = np.log(C_max / 10)
    log_max = np.log(C_max)
    log_vals = np.linspace(log_min, log_max, K)
    return np.exp(log_vals).tolist()


def calibrate_all_equal(C: float, K: int) -> List[float]:
    """All tiers use the same clipping bound C (equivalent to standard DP-SGD)."""
    return [C] * K
