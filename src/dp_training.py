"""
DP-SGD training implementations for Channeled DP-SGD experiments.

Implements:
- Standard DP-SGD (via Opacus PrivacyEngine)
- Channeled DP-SGD (per-tier clipping + noise, via GradSampleModule)
- Adaptive Clipping (Andrew et al., 2021)
- Non-private baseline

Critical constraint (per spec): σ is identical across all methods for a given
(ε, δ, T, q). σ is computed by privacy_accounting.compute_sigma() and passed
to every trainer. Channeled DP-SGD changes only C_k, not σ.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model: nn.Module, lr: float = 0.1, momentum: float = 0.9,
                   weight_decay: float = 5e-4) -> torch.optim.SGD:
    return torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )


def make_scheduler(optimizer, epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def _collect_per_sample_grads(model: nn.Module, batch_size: int, device: torch.device):
    """
    Concatenate per-sample gradient tensors from GradSampleModule.
    Returns per_sample_grads of shape (batch_size, D) and D.
    """
    grad_list = []
    for p in model.parameters():
        if p.requires_grad and hasattr(p, "grad_sample"):
            grad_list.append(p.grad_sample.reshape(batch_size, -1))
    per_sample = torch.cat(grad_list, dim=1)  # (B, D)
    return per_sample


def _write_grads_to_params(model: nn.Module, flat_grad: torch.Tensor):
    """Write a flat gradient vector back into model parameter .grad fields."""
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.grad = flat_grad[offset:offset + numel].reshape(p.shape).clone()
            offset += numel


def _clear_grad_samples(model: nn.Module):
    for p in model.parameters():
        if hasattr(p, "grad_sample"):
            p.grad_sample = None   # reset to None; do NOT del (Opacus hooks require attribute to exist)
        p.grad = None


# ---------------------------------------------------------------------------
# Standard DP-SGD (Opacus)
# ---------------------------------------------------------------------------

class StandardDPSGDTrainer:
    """
    Standard DP-SGD via Opacus PrivacyEngine.
    - One global clipping bound C.
    - Noise: N(0, σ²C²I) added to gradient sum at each step.
    """

    def __init__(
        self,
        model: nn.Module,
        sigma: float,
        C: float,
        n_train: int,
        batch_size: int,
        epochs: int,
        lr: float = 0.1,
        device: torch.device = None,
        delta: float = 1e-5,
    ):
        from opacus import PrivacyEngine
        from opacus.data_loader import DPDataLoader

        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.sigma = sigma
        self.C = C
        self.n_train = n_train
        self.batch_size = batch_size
        self.epochs = epochs
        self.delta = delta

        self.optimizer = make_optimizer(self.model, lr=lr)
        self.scheduler = make_scheduler(self.optimizer, epochs)
        self.privacy_engine = PrivacyEngine()

        self._DPDataLoader = DPDataLoader
        self._lr = lr

    def make_private(self, train_loader: DataLoader):
        """Attach PrivacyEngine to model/optimizer/loader."""
        model_dp, optimizer_dp, loader_dp = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=train_loader,
            noise_multiplier=self.sigma,
            max_grad_norm=self.C,
            poisson_sampling=True,
        )
        self.model = model_dp
        self.optimizer = optimizer_dp
        return loader_dp

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch. Returns dict with loss and accuracy."""
        self.model.train()
        total_loss, correct, n = 0.0, 0, 0

        for batch in loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.shape[0]
            correct += (out.argmax(1) == y).sum().item()
            n += x.shape[0]

        return {"loss": total_loss / n, "accuracy": correct / n}

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        checkpoint_epochs: List[int] = None,
        checkpoint_dir: str = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training run.

        Returns:
            results: dict with 'test_acc', 'train_acc', 'epsilon', 'history'
        """
        loader = self.make_private(train_loader)
        self.scheduler = make_scheduler(self.optimizer, self.epochs)

        history = []
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(loader)
            self.scheduler.step()

            eps = self.privacy_engine.get_epsilon(self.delta)

            if verbose and epoch % 10 == 0:
                test_acc = evaluate(self.model, test_loader, self.device)
                print(
                    f"  Epoch {epoch:3d}/{self.epochs}: "
                    f"loss={train_metrics['loss']:.4f}, "
                    f"train_acc={train_metrics['accuracy']:.4f}, "
                    f"test_acc={test_acc:.4f}, "
                    f"ε={eps:.3f}"
                )
                history.append(dict(epoch=epoch, test_acc=test_acc, eps=eps,
                                    **train_metrics))

            if checkpoint_epochs and epoch in checkpoint_epochs and checkpoint_dir:
                _save_checkpoint(self.model, epoch, checkpoint_dir)

        test_acc = evaluate(self.model, test_loader, self.device)
        train_acc = evaluate(self.model, train_loader, self.device)
        eps = self.privacy_engine.get_epsilon(self.delta)

        return dict(
            test_acc=test_acc,
            train_acc=train_acc,
            epsilon=eps,
            sigma=self.sigma,
            C=self.C,
            history=history,
        )


# ---------------------------------------------------------------------------
# Channeled DP-SGD
# ---------------------------------------------------------------------------

class ChanneledDPSGDTrainer:
    """
    Channeled DP-SGD per the spec pseudocode.

    For each step:
      1. Compute per-sample gradients via GradSampleModule.
      2. For each tier k:
         a. Clip gradients of tier-k samples to C_k.
         b. Sum clipped gradients.
         c. Add N(0, σ²C_k²I) noise (always, even if no tier-k samples).
      3. Sum across all tiers.
      4. Normalize by (q * n_train).
      5. Apply via SGD optimizer.

    Privacy: each tier k applies Gaussian mechanism with sensitivity C_k and
    noise multiplier σ. Since sample i only participates in tier k_i, the
    per-sample privacy guarantee is identical to standard DP-SGD with the same σ.
    """

    def __init__(
        self,
        model: nn.Module,
        sigma: float,
        C_per_tier: List[float],
        n_train: int,
        batch_size: int,
        epochs: int,
        lr: float = 0.1,
        device: torch.device = None,
        delta: float = 1e-5,
    ):
        """
        Args:
            model: PyTorch model (must be DP-compatible, i.e. no BatchNorm)
            sigma: noise multiplier (same as standard DP-SGD)
            C_per_tier: list of K clipping bounds [C_0, ..., C_{K-1}]
            n_train: total private training set size
            batch_size: logical batch size for q = batch_size / n_train
            epochs: number of training epochs
            device: compute device

        NOTE: The training loader MUST yield batches of (x, y, tier_label).
              Use datasets.TieredDataset to wrap the training dataset.
        """
        from opacus.grad_sample import GradSampleModule

        self.device = device or torch.device("cpu")
        self.sigma = sigma
        self.C_per_tier = list(C_per_tier)
        self.K = len(C_per_tier)
        self.n_train = n_train
        self.q = batch_size / n_train
        self.batch_size = batch_size
        self.epochs = epochs
        self.delta = delta

        # Wrap model for per-sample gradient computation
        self.model = GradSampleModule(model).to(self.device)
        self.optimizer = make_optimizer(self.model, lr=lr)
        self.scheduler = make_scheduler(self.optimizer, epochs)

    def _channeled_step(self, x: torch.Tensor, y: torch.Tensor,
                         sample_tiers: np.ndarray):
        """
        Execute one Channeled DP-SGD step.

        sample_tiers: tier index in {0, ..., K-1} for each sample in the batch.
                      Provided directly from the TieredDataset (batch[2]).
        """
        _clear_grad_samples(self.model)
        self.model.train()

        # Forward + backward (sum reduction → grad_sample[i] = ∇ℓ_i)
        out = self.model(x)
        loss = F.cross_entropy(out, y, reduction="sum")
        loss.backward()

        batch_size = x.shape[0]
        per_sample_grads = _collect_per_sample_grads(self.model, batch_size, self.device)
        D = per_sample_grads.shape[1]

        batch_tiers = np.asarray(sample_tiers)

        total_update = torch.zeros(D, device=self.device)

        for k in range(self.K):
            C_k = self.C_per_tier[k]
            mask = torch.tensor(batch_tiers == k, device=self.device)

            if mask.any():
                tier_grads = per_sample_grads[mask]  # (n_k, D)
                norms = tier_grads.norm(dim=1, keepdim=True)  # (n_k, 1)
                # Per-sample clip factor: min(1, C_k / ||g_i||)
                scale = torch.clamp(C_k / norms.clamp(min=1e-8), max=1.0)
                clipped = tier_grads * scale  # (n_k, D)
                aggregate = clipped.sum(dim=0)  # (D,)
            else:
                aggregate = torch.zeros(D, device=self.device)

            # Gaussian noise with std = σ * C_k (always drawn, even if empty)
            noise = torch.randn(D, device=self.device) * (self.sigma * C_k)
            total_update += aggregate + noise

        # Normalize by expected batch size
        total_update /= (self.q * self.n_train)

        # Write into .grad and step
        _clear_grad_samples(self.model)
        _write_grads_to_params(self.model, total_update)
        self.optimizer.step()

        return out.detach(), loss.item() / batch_size

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch using Channeled DP-SGD.

        The loader MUST yield (x, y, tier) triples (use TieredDataset).
        """
        self.model.train()
        total_loss, correct, n = 0.0, 0, 0

        for batch in loader:
            if len(batch) < 3:
                raise ValueError(
                    "ChanneledDPSGDTrainer expects batches of (x, y, tier). "
                    "Wrap your dataset with datasets.TieredDataset."
                )
            x, y, tiers = batch[0], batch[1], batch[2]
            tiers_np = tiers.numpy() if hasattr(tiers, "numpy") else np.asarray(tiers)

            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out, loss_val = self._channeled_step(x, y, tiers_np)

            correct += (out.argmax(1) == y).sum().item()
            total_loss += loss_val * x.shape[0]
            n += x.shape[0]

        return {"loss": total_loss / n, "accuracy": correct / n}

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        checkpoint_epochs: List[int] = None,
        checkpoint_dir: str = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Full training run. Returns results dict."""
        history = []

        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            self.scheduler.step()

            if verbose and epoch % 10 == 0:
                test_acc = evaluate(self.model, test_loader, self.device)
                print(
                    f"  [Channeled K={self.K}] Epoch {epoch:3d}/{self.epochs}: "
                    f"loss={train_metrics['loss']:.4f}, "
                    f"train_acc={train_metrics['accuracy']:.4f}, "
                    f"test_acc={test_acc:.4f}"
                )
                history.append(dict(epoch=epoch, test_acc=test_acc, **train_metrics))

            if checkpoint_epochs and epoch in checkpoint_epochs and checkpoint_dir:
                _save_checkpoint(self.model, epoch, checkpoint_dir)

        test_acc = evaluate(self.model, test_loader, self.device)
        train_acc = evaluate(self.model, train_loader, self.device)

        return dict(
            test_acc=test_acc,
            train_acc=train_acc,
            sigma=self.sigma,
            C_per_tier=self.C_per_tier,
            K=self.K,
            history=history,
        )


# ---------------------------------------------------------------------------
# Adaptive Clipping (Andrew et al., 2021)
# ---------------------------------------------------------------------------

class AdaptiveClipTrainer:
    """
    Andrew et al. (2021) adaptive clipping via Opacus.

    The quantile target (default 0.5) is estimated privately, with privacy
    cost included in the total ε budget via Opacus's privacy accounting.
    """

    def __init__(
        self,
        model: nn.Module,
        sigma: float,
        n_train: int,
        batch_size: int,
        epochs: int,
        lr: float = 0.1,
        quantile: float = 0.5,
        device: torch.device = None,
        delta: float = 1e-5,
        initial_C: float = 1.0,
    ):
        from opacus import PrivacyEngine

        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.sigma = sigma
        self.quantile = quantile
        self.n_train = n_train
        self.batch_size = batch_size
        self.epochs = epochs
        self.delta = delta
        self.initial_C = initial_C

        self.optimizer = make_optimizer(self.model, lr=lr)
        self.scheduler = make_scheduler(self.optimizer, epochs)
        self.privacy_engine = PrivacyEngine()

    def make_private(self, train_loader: DataLoader):
        from opacus.optimizers import DPOptimizerFastGradientClipping

        model_dp, optimizer_dp, loader_dp = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=train_loader,
            noise_multiplier=self.sigma,
            max_grad_norm=self.initial_C,
            clipping="adaptive",
            grad_sample_mode="hooks",
            poisson_sampling=True,
        )
        self.model = model_dp
        self.optimizer = optimizer_dp
        return loader_dp

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, n = 0.0, 0, 0
        for batch in loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.shape[0]
            correct += (out.argmax(1) == y).sum().item()
            n += x.shape[0]
        return {"loss": total_loss / n, "accuracy": correct / n}

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        loader = self.make_private(train_loader)
        self.scheduler = make_scheduler(self.optimizer, self.epochs)

        for epoch in range(1, self.epochs + 1):
            self.train_epoch(loader)
            self.scheduler.step()
            if verbose and epoch % 10 == 0:
                test_acc = evaluate(self.model, test_loader, self.device)
                eps = self.privacy_engine.get_epsilon(self.delta)
                print(f"  [Adaptive] Epoch {epoch}/{self.epochs}: "
                      f"test_acc={test_acc:.4f}, ε={eps:.3f}")

        test_acc = evaluate(self.model, test_loader, self.device)
        train_acc = evaluate(self.model, train_loader, self.device)
        eps = self.privacy_engine.get_epsilon(self.delta)

        return dict(
            test_acc=test_acc,
            train_acc=train_acc,
            epsilon=eps,
            sigma=self.sigma,
        )


# ---------------------------------------------------------------------------
# Non-private baseline
# ---------------------------------------------------------------------------

class NonPrivateTrainer:
    """Standard SGD without any DP noise."""

    def __init__(
        self,
        model: nn.Module,
        n_train: int,
        epochs: int,
        lr: float = 0.1,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.epochs = epochs
        self.optimizer = make_optimizer(self.model, lr=lr)
        self.scheduler = make_scheduler(self.optimizer, epochs)

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss, correct, n = 0.0, 0, 0
            for batch in train_loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * x.shape[0]
                correct += (out.argmax(1) == y).sum().item()
                n += x.shape[0]
            self.scheduler.step()
            if verbose and epoch % 20 == 0:
                test_acc = evaluate(self.model, test_loader, self.device)
                print(f"  [Non-private] Epoch {epoch}/{self.epochs}: "
                      f"loss={total_loss/n:.4f}, test_acc={test_acc:.4f}")

        test_acc = evaluate(self.model, test_loader, self.device)
        train_acc = evaluate(self.model, train_loader, self.device)
        return dict(test_acc=test_acc, train_acc=train_acc, epsilon=float("inf"))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute top-1 accuracy."""
    model.eval()
    correct, n = 0, 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        n += x.shape[0]
    return correct / n if n > 0 else 0.0


@torch.no_grad()
def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """Compute per-class test accuracy. Returns array of shape (num_classes,)."""
    model.eval()
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x)
        preds = out.argmax(1).cpu().numpy()
        y_np = y.cpu().numpy()
        for c in range(num_classes):
            mask = y_np == c
            correct[c] += (preds[mask] == c).sum()
            total[c] += mask.sum()
    return np.where(total > 0, correct / total, 0.0)


@torch.no_grad()
def compute_per_sample_losses(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Compute per-sample cross-entropy losses.

    Returns:
        losses: np.ndarray of shape (n,)
        labels: np.ndarray of shape (n,)
        predictions: np.ndarray of shape (n,) with predicted class indices
        confidences: np.ndarray of shape (n,) with max softmax probabilities
    """
    model.eval()
    all_losses, all_labels, all_preds, all_confs = [], [], [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x)
        losses = F.cross_entropy(out, y, reduction="none").cpu().numpy()
        probs = F.softmax(out, dim=1).cpu().numpy()
        preds = out.argmax(1).cpu().numpy()
        confs = probs.max(axis=1)
        all_losses.append(losses)
        all_labels.append(y.cpu().numpy())
        all_preds.append(preds)
        all_confs.append(confs)
    return (
        np.concatenate(all_losses),
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_confs),
    )


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def _save_checkpoint(model: nn.Module, epoch: int, directory: str):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"epoch_{epoch:04d}.pt")
    # Unwrap Opacus/GradSampleModule wrappers before saving
    actual_model = model
    if hasattr(model, "_module"):
        actual_model = model._module
    torch.save({"epoch": epoch, "state_dict": actual_model.state_dict()}, path)


def load_checkpoint(model: nn.Module, path: str) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    return ckpt["epoch"]
