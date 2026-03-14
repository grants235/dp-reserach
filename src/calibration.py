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


def _get_raw_model(model: nn.Module):
    """
    Strip GradSampleModule wrappers.
    Returns (raw_nn_module, gsm_or_None) without modifying any hooks.
    """
    from opacus.grad_sample import GradSampleModule
    if isinstance(model, GradSampleModule):
        return model._module, model
    probe = model
    while hasattr(probe, "_module"):
        inner = probe._module
        if isinstance(inner, GradSampleModule):
            return inner._module, inner
        probe = inner
    return model, None


def compute_per_sample_gradient_norms(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = None,
) -> np.ndarray:
    """
    Compute the L2 gradient norm ||∇ℓ(θ, z_i)|| for each sample z_i.

    Uses torch.func (vmap + grad + functional_call).  Two hook issues must both
    be suppressed before vmap is called:

      1. Opacus *forward* hook (capture_activations_hook): accesses
         p._forward_counter on the module's parameters.  Under functional_call
         those are replaced with plain detached tensors that lack the attribute.
         Fix: gsm.disable_hooks() makes the Python callback return immediately.

      2. Opacus *backward* hook (capture_backprops_hook): registered via
         register_full_backward_hook.  PyTorch injects a BackwardHookFunction
         wrapper into the forward graph whenever _backward_hooks is non-empty,
         and BackwardHookFunction does not implement setup_context, so vmap
         raises RuntimeError.  gsm.disable_hooks() only silences the Python
         callback — it does NOT prevent the wrapper from being inserted.
         Fix: temporarily empty _backward_hooks (and _backward_pre_hooks) on
         every submodule of raw, then restore in the finally block.

    Works for both plain nn.Module and Opacus-wrapped GradSampleModule.
    """
    from torch.func import vmap, grad, functional_call

    raw, gsm = _get_raw_model(model)
    raw = raw.to(device)

    params  = {n: p.detach() for n, p in raw.named_parameters()}
    buffers = {n: b.detach() for n, b in raw.named_buffers()}

    def loss_single(params, x_i, y_i):
        # x_i: (C, H, W); y_i: scalar → (1,) for cross_entropy
        out = functional_call(raw, {**params, **buffers}, (x_i.unsqueeze(0),))
        return F.cross_entropy(out, y_i.unsqueeze(0))

    per_sample_grad_fn = vmap(grad(loss_single), in_dims=(None, 0, 0))

    # --- Fix 1: silence Opacus forward-hook _forward_counter access ---
    if gsm is not None:
        gsm.disable_hooks()

    # --- Fix 2: remove backward hooks so BackwardHookFunction is never injected ---
    saved_bw     = {}
    saved_bw_pre = {}
    for name, m in raw.named_modules():
        if m._backward_hooks:
            saved_bw[name] = dict(m._backward_hooks)
            m._backward_hooks.clear()
        bwp = getattr(m, "_backward_pre_hooks", None)
        if bwp:
            saved_bw_pre[name] = dict(bwp)
            bwp.clear()

    all_norms = []
    n_batches  = 0

    try:
        for batch in loader:
            if max_batches and n_batches >= max_batches:
                break

            x, y = batch[0].to(device), batch[1].to(device)
            B = x.shape[0]

            per_sample_grads = per_sample_grad_fn(params, x, y)
            flat  = torch.cat([g.reshape(B, -1) for g in per_sample_grads.values()], dim=1)
            norms = flat.norm(dim=1).detach().cpu().numpy()
            all_norms.append(norms)
            n_batches += 1

    finally:
        # Restore backward hooks before resuming training
        for name, m in raw.named_modules():
            if name in saved_bw:
                m._backward_hooks.update(saved_bw[name])
            bwp = getattr(m, "_backward_pre_hooks", None)
            if bwp is not None and name in saved_bw_pre:
                bwp.update(saved_bw_pre[name])
        if gsm is not None:
            gsm.enable_hooks()

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
