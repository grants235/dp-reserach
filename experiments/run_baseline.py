"""
run_baseline.py — Establish non-private and DP-SGD baselines.

Trains WRN-28-2 in three modes and saves models + configs for reuse
in all downstream experiments.

  1. nonprivate  — SGD, no DP noise.  Target: ~94% CIFAR-10.
  2. dp_standard — spec default (bs=256, no aug-mult).  Target: ~38%.
  3. dp_opt      — aug-mult + tuned LR + more epochs.   Target: ~60-65%.

Key improvement vs exp1 for dp_opt:
  • Augmentation multiplicity (k): k augmented views of each sample are
    averaged before clipping.  Reduces per-sample gradient variance at
    zero extra privacy cost (Tramèr & Boneh 2021).
  • LR scaled with effective signal: LR ≈ lr_base * sqrt(bs*aug_mult/256).
  • 200 epochs instead of 100.

Usage:
    python experiments/run_baseline.py --dataset cifar10 --gpu 0
    python experiments/run_baseline.py --dataset cifar10 --mode dp_opt --gpu 0
    python experiments/run_baseline.py --sweep --gpu 0   # try all configs

Outputs:  results/baseline/<tag>/  {model.pt, config.json, results.pkl}
"""

import os, sys, json, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets import load_datasets, make_data_loaders, IndexedDataset
from src.models import make_model, validate_model_for_dp
from src.dp_training import (
    _collect_per_sample_grads, _write_grads_to_params, _clear_grad_samples,
    evaluate, set_seed,
)
from src.privacy_accounting import compute_sigma


# ---------------------------------------------------------------------------
# Defaults — overridden per config below
# ---------------------------------------------------------------------------
DATA_ROOT   = "./data"
RESULTS_DIR = "results/baseline"
DELTA       = 1e-5
EPS_TARGET  = 3.0
ARCH        = "wrn28-2"
SEEDS       = [0, 1, 2]


# ---------------------------------------------------------------------------
# GPU-side augmentation (applied to normalized CIFAR tensors, batch-wise)
# ---------------------------------------------------------------------------

def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Random 4-px-pad crop + horizontal flip on a batch of CIFAR tensors.

    Works on CPU or CUDA.  Each sample in the batch gets an independent
    random transform, so repeated calls produce different augmented views.
    Fully vectorized — no Python loop over batch elements.
    """
    B, C, H, W = x.shape
    # Reflect-pad by 4 on each spatial side → (B, C, H+8, W+8)
    xp = F.pad(x, (4, 4, 4, 4), mode='reflect')
    # Random crop offsets per sample
    oi = torch.randint(0, 8, (B,), device=x.device)
    oj = torch.randint(0, 8, (B,), device=x.device)
    # Vectorized crop: build row/col index tensors and gather in one shot
    rows = oi[:, None] + torch.arange(H, device=x.device)   # (B, H)
    cols = oj[:, None] + torch.arange(W, device=x.device)   # (B, W)
    b_idx = torch.arange(B, device=x.device)[:, None, None]  # (B, 1, 1)
    # Permute to (B, H+8, W+8, C) so channels come along for free
    xp_t = xp.permute(0, 2, 3, 1)                           # (B, H+8, W+8, C)
    crops = xp_t[b_idx, rows[:, :, None], cols[:, None, :]]  # (B, H, W, C)
    crops = crops.permute(0, 3, 1, 2).contiguous()           # (B, C, H, W)
    # Random horizontal flip
    flip = torch.rand(B, device=x.device) > 0.5
    crops[flip] = crops[flip].flip(-1)
    return crops


# ---------------------------------------------------------------------------
# Non-private training
# ---------------------------------------------------------------------------

def train_nonprivate(
    dataset_name: str,
    seed: int,
    device: torch.device,
    data_root: str = DATA_ROOT,
    results_dir: str = RESULTS_DIR,
    lr: float = 0.1,
    batch_size: int = 128,
    epochs: int = 200,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
):
    tag = f"nonprivate_{dataset_name}_seed{seed}"
    out_dir = os.path.join(results_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "results.pkl")
    if os.path.exists(result_path):
        print(f"[baseline] {tag}: cached, skipping.")
        with open(result_path, "rb") as f:
            return pickle.load(f)

    set_seed(seed)
    data = load_datasets(dataset_name, data_root=data_root, imbalance_ratio=1.0,
                         public_frac=0.1, split_seed=42)
    num_classes = data["num_classes"]
    _, _, test_loader = make_data_loaders(data, batch_size=256)

    # Full-augmentation private loader
    train_loader = DataLoader(data["private_dataset"], batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    model = make_model(ARCH, num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
        scheduler.step()
        if epoch % 20 == 0 or epoch == epochs:
            acc = evaluate(model, test_loader, device)
            print(f"  [nonprivate {tag}] epoch {epoch}/{epochs}  test_acc={acc:.4f}")

    test_acc = evaluate(model, test_loader, device)
    config = dict(mode="nonprivate", dataset=dataset_name, seed=seed, lr=lr,
                  batch_size=batch_size, epochs=epochs, momentum=momentum,
                  weight_decay=weight_decay)
    results = dict(tag=tag, test_acc=test_acc, config=config)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(result_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  [nonprivate {tag}] DONE  test_acc={test_acc:.4f}")

    # Explicit memory cleanup
    del model, optimizer, scheduler, train_loader, test_loader
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# DP-SGD with augmentation multiplicity (core routine)
# ---------------------------------------------------------------------------

def _dp_aug_mult_step(
    gsm: nn.Module,
    optimizer: torch.optim.Optimizer,
    x_clean: torch.Tensor,          # (B, C, H, W) — no augmentation applied yet
    y: torch.Tensor,                # (B,)
    aug_mult: int,
    C: float,
    sigma: float,
    q: float,
    n_train: int,
    device: torch.device,
    chunk_size: int = None,         # split batch into chunks to cap Opacus grad_sample memory
):
    """One DP-SGD step with augmentation multiplicity.

    For each of the B samples, draws aug_mult independent augmented views,
    computes per-sample gradients for all B*aug_mult views, reshapes to
    (B, aug_mult, D), averages over the aug dimension → (B, D), then clips
    the averaged gradient per sample, aggregates, and adds Gaussian noise.

    Privacy analysis: averaging over aug_mult views of the SAME sample does
    not change privacy cost — it is variance reduction within a single
    sample's gradient, exactly as in Tramèr & Boneh (2021).

    Memory: Opacus stores grad_samples of shape (chunk * aug_mult, *p_shape)
    per parameter.  chunk_size caps this at chunk_size * aug_mult samples per
    backward pass.  Clipped sums are accumulated across chunks; noise is added
    ONCE at the end, so privacy accounting is unchanged.
    """
    B = x_clean.shape[0]
    eff_chunk = B if (chunk_size is None or chunk_size >= B) else chunk_size

    agg_parts = None   # list of per-param clipped-sum tensors (D_p,)
    first_preds = None

    for start in range(0, B, eff_chunk):
        end = min(start + eff_chunk, B)
        Bc = end - start
        xc = x_clean[start:end]
        yc = y[start:end]

        x_views = torch.cat([augment_batch(xc) for _ in range(aug_mult)], dim=0)
        y_views = yc.repeat(aug_mult)

        _clear_grad_samples(gsm)
        gsm.train()
        out = gsm(x_views)
        F.cross_entropy(out, y_views, reduction='sum').backward()

        if first_preds is None:
            first_preds = (out[:Bc].detach(), yc)

        # Pass 1 — per-sample squared norms for this chunk
        sq_norms = torch.zeros(Bc, device=xc.device)
        for p in gsm.parameters():
            if p.requires_grad and hasattr(p, "grad_sample") and p.grad_sample is not None:
                gs = p.grad_sample.reshape(Bc, aug_mult, -1).mean(dim=1)  # (Bc, d_p)
                sq_norms += gs.pow(2).sum(dim=1)
        scale = torch.clamp(C / sq_norms.sqrt_().clamp_(min=1e-8), max=1.0)  # (Bc,)

        # Pass 2 — clip and accumulate into agg_parts (no noise yet)
        chunk_parts = []
        for p in gsm.parameters():
            if not p.requires_grad:
                continue
            if hasattr(p, "grad_sample") and p.grad_sample is not None:
                gs = p.grad_sample.reshape(Bc, aug_mult, -1).mean(dim=1)  # (Bc, d_p)
                clipped_p = (gs * scale[:, None]).sum(dim=0)               # (d_p,)
            else:
                clipped_p = torch.zeros(p.numel(), device=xc.device)
            chunk_parts.append(clipped_p.reshape(p.shape))

        _clear_grad_samples(gsm)

        if agg_parts is None:
            agg_parts = chunk_parts
        else:
            for i in range(len(agg_parts)):
                agg_parts[i] = agg_parts[i] + chunk_parts[i]

    # Add noise ONCE over the full aggregated clipped gradient
    param_iter = (p for p in gsm.parameters() if p.requires_grad)
    for p, agg in zip(param_iter, agg_parts):
        noise = torch.randn_like(agg) * (sigma * C)
        p.grad = (agg + noise) / (q * n_train)

    optimizer.step()
    optimizer.zero_grad()

    return first_preds   # (out[:Bc], yc) from first chunk — for train-acc logging


# ---------------------------------------------------------------------------
# DP training wrapper
# ---------------------------------------------------------------------------

def train_dp(
    dataset_name: str,
    imbalance_ratio: float,
    seed: int,
    device: torch.device,
    data_root: str = DATA_ROOT,
    results_dir: str = RESULTS_DIR,
    # -- DP hyperparameters --
    batch_size: int = 256,
    aug_mult: int   = 1,
    lr: float       = 0.1,
    epochs: int     = 100,
    C: float        = 1.0,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    mode_label: str = "dp_standard",
    chunk_size: int = None,   # max samples per Opacus backward; None = full batch
):
    tag = f"{mode_label}_{dataset_name}_IR{imbalance_ratio:.0f}_seed{seed}"
    out_dir = os.path.join(results_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "results.pkl")
    if os.path.exists(result_path):
        print(f"[baseline] {tag}: cached, skipping.")
        with open(result_path, "rb") as f:
            return pickle.load(f)

    set_seed(seed)
    data = load_datasets(dataset_name, data_root=data_root,
                         imbalance_ratio=imbalance_ratio,
                         public_frac=0.1, split_seed=42)
    num_classes = data["num_classes"]
    n_train     = data["n_train"]
    _, _, test_loader = make_data_loaders(data, batch_size=256)

    # Non-shuffled no-aug loader (clean tensors for manual augmentation)
    train_loader = DataLoader(
        data["private_dataset_noaug"],
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    q = batch_size / n_train
    T = epochs * len(train_loader)
    sigma = compute_sigma(EPS_TARGET, DELTA, q, T)
    print(f"[baseline] {tag}: σ={sigma:.4f} q={q:.5f} T={T} aug_mult={aug_mult}")

    from opacus.grad_sample import GradSampleModule
    model = make_model(ARCH, num_classes)
    assert validate_model_for_dp(model)
    gsm = GradSampleModule(model).to(device)

    optimizer = torch.optim.SGD(gsm.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        gsm.train()
        correct, total = 0, 0
        for batch in train_loader:
            x_clean = batch[0].to(device)
            y       = batch[1].to(device)
            preds, yt = _dp_aug_mult_step(
                gsm, optimizer, x_clean, y,
                aug_mult=aug_mult, C=C, sigma=sigma,
                q=q, n_train=n_train, device=device,
                chunk_size=chunk_size,
            )
            correct += (preds.argmax(1) == yt).sum().item()
            total   += yt.shape[0]
        scheduler.step()

        if epoch % 20 == 0 or epoch == epochs:
            test_acc = evaluate(gsm, test_loader, device)
            print(f"  [{tag}] epoch {epoch}/{epochs}"
                  f"  train={correct/max(total,1):.3f}  test={test_acc:.4f}")

    test_acc = evaluate(gsm, test_loader, device)

    # Compute achieved epsilon using Opacus accountant
    from opacus.accountants import RDPAccountant
    acc_rdp = RDPAccountant()
    acc_rdp.history = [(sigma, q, T)]
    eps_achieved = acc_rdp.get_epsilon(delta=DELTA)

    config = dict(
        mode=mode_label, dataset=dataset_name, imbalance_ratio=imbalance_ratio,
        seed=seed, batch_size=batch_size, aug_mult=aug_mult, lr=lr,
        epochs=epochs, C=C, momentum=momentum, weight_decay=weight_decay,
        sigma=sigma, eps_target=EPS_TARGET, eps_achieved=eps_achieved,
        delta=DELTA, q=q, T=T, chunk_size=chunk_size,
    )
    results = dict(tag=tag, test_acc=test_acc, config=config)

    # Save raw model (unwrap GradSampleModule)
    raw = gsm._module
    torch.save(raw.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(result_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  [{tag}] DONE  test_acc={test_acc:.4f}  ε={eps_achieved:.3f}")

    # Explicit memory cleanup
    del raw, gsm, model, optimizer, scheduler, train_loader, test_loader
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Predefined configurations
# Tune ranges documented inline; change to sweep with --sweep flag.
# ---------------------------------------------------------------------------

# LR scaling rule for DP-SGD (linear with sqrt of effective batch size):
#   lr = lr_base * sqrt(bs * aug_mult / 256)
# where lr_base=0.5 is tuned empirically.  Adjust lr_base ∈ [0.2, 1.0].
_LR_BASE = 0.5


def _dp_lr(bs, aug_mult, lr_base=_LR_BASE):
    return lr_base * (bs * aug_mult / 256) ** 0.5


CONFIGS = {
    # ── baseline (reproduce exp1 for comparison) ──────────────────────────
    "dp_standard": dict(
        batch_size=256, aug_mult=1, lr=0.1, epochs=100, C=1.0,
    ),
    # ── aug-mult only (most impactful single change) ──────────────────────
    # aug_mult ∈ {2, 4, 8, 16}; 8 is the sweet spot (Tramèr & Boneh 2021)
    "dp_augmult8": dict(
        batch_size=256, aug_mult=8, lr=_dp_lr(256, 8), epochs=200, C=1.0,
    ),
    # ── larger batch + aug-mult (recommended for best accuracy) ──────────
    # batch_size ∈ {512, 1024, 2048}; lr ∈ [0.5, 4.0]
    # chunk_size=512: processes 512*4=2048 aug views per backward pass,
    # matching dp_augmult8's memory footprint (256*8=2048). 2 chunks/step.
    "dp_opt": dict(
        batch_size=1024, aug_mult=4, lr=_dp_lr(1024, 4), epochs=200, C=1.0,
        chunk_size=512,
    ),
}



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all(dataset: str, mode: str, device: torch.device,
            data_root: str, results_dir: str, seeds):
    all_results = {}

    if mode in ("nonprivate", "all"):
        for seed in seeds:
            r = train_nonprivate(
                dataset, seed=seed, device=device,
                data_root=data_root, results_dir=results_dir,
            )
            all_results[r["tag"]] = r

    dp_modes = list(CONFIGS.keys()) if mode in ("all", "sweep") else \
               [mode] if mode in CONFIGS else []

    for dp_mode in dp_modes:
        cfg = CONFIGS[dp_mode]
        for seed in seeds:
            r = train_dp(
                dataset, imbalance_ratio=1.0, seed=seed, device=device,
                data_root=data_root, results_dir=results_dir,
                mode_label=dp_mode, **cfg,
            )
            all_results[r["tag"]] = r

    # Summary
    print("\n" + "="*70)
    print("BASELINE SUMMARY")
    print("="*70)
    print(f"{'Tag':<50} {'test_acc':>9}  {'ε':>6}")
    print("-"*70)
    for tag, r in sorted(all_results.items()):
        eps = r["config"].get("eps_achieved", float("inf"))
        eps_str = f"{eps:.3f}" if eps < float("inf") else "  ∞  "
        print(f"  {tag:<48} {r['test_acc']:>9.4f}  {eps_str:>6}")

    # Save aggregated summary
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(
            {tag: {"test_acc": r["test_acc"],
                   "eps": r["config"].get("eps_achieved"),
                   "config": r["config"]}
             for tag, r in all_results.items()},
            f, indent=2,
        )
    print(f"\nSummary saved to {os.path.join(results_dir, 'summary.json')}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline accuracy profiling")
    parser.add_argument("--dataset",  default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--mode",     default="dp_opt",
                        choices=["nonprivate", "dp_standard", "dp_augmult8",
                                 "dp_opt", "all", "sweep"])
    parser.add_argument("--gpu",      type=int, default=0)
    parser.add_argument("--seeds",    type=int, nargs="+", default=[0])
    parser.add_argument("--data_root",default=DATA_ROOT)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Configs available: {list(CONFIGS.keys())}")
    lr_summary = ", ".join(f"{k}={CONFIGS[k]['lr']:.3f}" for k in CONFIGS)
    print(f"LR values: {lr_summary}")

    run_all(
        dataset=args.dataset, mode=args.mode, device=device,
        data_root=args.data_root, results_dir=args.results_dir,
        seeds=args.seeds,
    )
