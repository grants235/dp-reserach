"""
Experiment 1: Gradient Structure Validation — Phase 1
======================================================

Goal: Test whether gradient norms stratify by tier with a properly-tuned
training pipeline.  Phase-0 validated: dp_augmult8 hit 55.51% on CIFAR-10
at ε=3, clearing the ≥55% go/no-go gate.

Phase-0 best config (dp_augmult8 from run_baseline.py):
  aug_mult=8, B=256, LR=1.414, 200 epochs, C=1.0 → 55.51% @ ε≈3.0

Procedure (spec2.tex §4.2):
- Train WRN-28-2 on CIFAR-10 (balanced) and CIFAR-10-LT (IR=50) at ε=3
  using the Phase-0 best config.
- Save checkpoints at epochs {1, 5, 10, 25, 50, 75, 100, 150, 200}.
- At each checkpoint compute for ALL training examples:
    • Unclipped gradient norm ||∇ℓ(θ_t, z_i)||
    • Clipped gradient norm ||ḡ_i(θ_t)|| (min(unclipped, C))
    • Training loss ℓ(θ_t, z_i)
    • Correctly classified (yes/no)
- Assign tiers (Strategy A: class frequency for LT; Strategy B: density).
- Report: per-tier gradient norm mean, p50, p95, and Tier2/Tier0 ratio.

Key question (spec2.tex §4.2): Does the tier ratio improve when training is
properly tuned?  Success criterion: ≥2× ratio Tier2/Tier0 in clipped norms.

Outputs: results/exp1_p1/<tag>/{results.pkl, checkpoints/}
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import pickle

from torch.utils.data import DataLoader
from opacus.grad_sample import GradSampleModule
from opacus.accountants import RDPAccountant

from src.datasets import load_datasets, make_data_loaders
from src.models import make_model, validate_model_for_dp
from src.dp_training import (
    _clear_grad_samples, evaluate, compute_per_sample_losses, set_seed
)
from src.calibration import (
    compute_per_sample_gradient_norms,
    compute_per_sample_clipped_norms,
    train_public_model,
)
from src.tiers import assign_tiers, get_tier_sizes
from src.privacy_accounting import compute_sigma
from src.evaluation import save_results, extract_features


# ---------------------------------------------------------------------------
# Constants — Phase-0 best config (dp_augmult8)
# ---------------------------------------------------------------------------
RESULTS_DIR = "results/exp1_p1"
DATA_ROOT   = "./data"
ARCH        = "wrn28-2"
BATCH_SIZE  = 256
AUG_MULT    = 8
LR          = 0.5 * (BATCH_SIZE * AUG_MULT / 256) ** 0.5   # ≈ 1.4142
EPOCHS      = 200
EPS_TARGET  = 3.0
DELTA       = 1e-5
C           = 1.0   # Phase-0 C; 98% clipping saturation observed at this value
K           = 3
# Extended to cover the full 200-epoch arc (original spec had up to 100)
CHECKPOINT_EPOCHS = [1, 5, 10, 25, 50, 75, 100, 150, 200]
SEEDS       = [0, 1, 2]
# Phase 1 focuses on CIFAR-10 balanced and LT; CIFAR-100 is deferred
DATASETS    = [
    ("cifar10",  1.0),
    ("cifar10", 50.0),   # CIFAR-10-LT IR=50
]


# ---------------------------------------------------------------------------
# GPU-side batch augmentation (vectorized — no Python loop over samples)
# ---------------------------------------------------------------------------

def _augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Random 4-px-pad crop + horizontal flip, fully vectorized."""
    B, C_, H, W = x.shape
    xp = F.pad(x, (4, 4, 4, 4), mode='reflect')
    oi = torch.randint(0, 8, (B,), device=x.device)
    oj = torch.randint(0, 8, (B,), device=x.device)
    rows  = oi[:, None] + torch.arange(H, device=x.device)   # (B, H)
    cols  = oj[:, None] + torch.arange(W, device=x.device)   # (B, W)
    b_idx = torch.arange(B, device=x.device)[:, None, None]
    xp_t  = xp.permute(0, 2, 3, 1)                           # (B, H+8, W+8, C_)
    crops = xp_t[b_idx, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2).contiguous()
    flip  = torch.rand(B, device=x.device) > 0.5
    crops[flip] = crops[flip].flip(-1)
    return crops


# ---------------------------------------------------------------------------
# One DP-SGD aug-mult step (two-pass clipping, chunked for memory safety)
# Noise added ONCE after accumulating clipped sums across all chunks.
# ---------------------------------------------------------------------------

def _aug_mult_step(
    gsm: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x_clean: torch.Tensor,
    y: torch.Tensor,
    sigma: float,
    C: float,
    q: float,
    n_train: int,
    aug_mult: int,
    chunk_size: int = None,
):
    """One DP-SGD step with augmentation multiplicity and chunked processing.

    chunk_size caps the number of samples per Opacus backward pass to bound
    grad_sample memory.  With B=256, aug_mult=8, chunk_size=None is fine
    (2048 total aug views, matching the dp_augmult8 baseline that ran without OOM).
    """
    B = x_clean.shape[0]
    eff_chunk = B if (chunk_size is None or chunk_size >= B) else chunk_size

    agg_parts   = None
    first_preds = None

    for start in range(0, B, eff_chunk):
        end = min(start + eff_chunk, B)
        Bc  = end - start
        xc  = x_clean[start:end]
        yc  = y[start:end]

        x_views = torch.cat([_augment_batch(xc) for _ in range(aug_mult)], dim=0)
        y_views = yc.repeat(aug_mult)

        _clear_grad_samples(gsm)
        gsm.train()
        out = gsm(x_views)
        F.cross_entropy(out, y_views, reduction='sum').backward()

        if first_preds is None:
            first_preds = (out[:Bc].detach(), yc)

        # Pass 1: accumulate per-sample squared norms
        sq_norms = torch.zeros(Bc, device=xc.device)
        for p in gsm.parameters():
            if p.requires_grad and hasattr(p, "grad_sample") and p.grad_sample is not None:
                gs = p.grad_sample.reshape(Bc, aug_mult, -1).mean(dim=1)  # (Bc, d_p)
                sq_norms += gs.pow(2).sum(dim=1)
        scale = torch.clamp(C / sq_norms.sqrt_().clamp_(min=1e-8), max=1.0)  # (Bc,)

        # Pass 2: clip and accumulate per-parameter clipped sums
        chunk_parts = []
        for p in gsm.parameters():
            if not p.requires_grad:
                continue
            if hasattr(p, "grad_sample") and p.grad_sample is not None:
                gs = p.grad_sample.reshape(Bc, aug_mult, -1).mean(dim=1)
                chunk_parts.append((gs * scale[:, None]).sum(dim=0).reshape(p.shape))
            else:
                chunk_parts.append(torch.zeros_like(p))

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
    return first_preds   # (preds[:Bc], yc) for train-acc logging


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp1(
    dataset_name: str,
    imbalance_ratio: float,
    seed: int,
    device: torch.device,
    data_root: str = DATA_ROOT,
    results_dir: str = RESULTS_DIR,
):
    """Run Phase-1 Experiment 1 for a single (dataset, IR, seed) config."""
    tag     = f"{dataset_name}_IR{imbalance_ratio:.0f}_seed{seed}"
    out_dir = os.path.join(results_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    result_path = os.path.join(out_dir, "results.pkl")
    if os.path.exists(result_path):
        print(f"[Exp1-P1] {tag}: already computed, skipping.")
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
        split_seed=42,   # fixed across all seeds/methods
    )
    num_classes   = data["num_classes"]
    n_train       = data["n_train"]
    class_counts  = data["class_counts"]
    private_targets = np.array(data["private_dataset"].targets)

    _, public_loader, test_loader = make_data_loaders(data, batch_size=256)

    # Shuffled no-aug loader for DP training (we apply augmentation manually)
    train_noaug_loader = DataLoader(
        data["private_dataset_noaug"],
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    # Ordered no-aug loader for gradient/loss evaluation — index-aligned with
    # private_targets, tiers_A, tiers_B
    eval_loader = DataLoader(
        data["private_dataset_noaug"],
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
    )

    # ------------------------------------------------------------------
    # 2. Compute σ — same formula as dp_augmult8 in run_baseline
    #    aug_mult does NOT change q or T (Tramèr & Boneh 2021)
    # ------------------------------------------------------------------
    q     = BATCH_SIZE / n_train
    T     = EPOCHS * len(train_noaug_loader)   # drop_last=True → consistent
    sigma = compute_sigma(EPS_TARGET, DELTA, q, T)
    print(f"[Exp1-P1] {tag}: σ={sigma:.4f}, q={q:.5f}, T={T}, "
          f"aug_mult={AUG_MULT}, LR={LR:.4f}")

    # ------------------------------------------------------------------
    # 3. Tier assignment
    #    Strategy A: class frequency (natural for LT; for balanced, assigns
    #                by class index mod K)
    #    Strategy B: feature-density-based using a public model
    # ------------------------------------------------------------------
    public_model = make_model(ARCH, num_classes)
    public_model = train_public_model(
        public_model, public_loader, device, epochs=50, verbose=False
    )

    tiers_A      = assign_tiers("A", private_targets, class_counts, K=K)
    tier_sizes_A = get_tier_sizes(tiers_A, K)
    print(f"[Exp1-P1] Strategy A tier sizes: {tier_sizes_A}")

    feats_public, _ = extract_features(public_model, public_loader, device)
    # Use ordered eval_loader so feature indices align with private_targets
    feats_private, _ = extract_features(public_model, eval_loader, device)
    tiers_B      = assign_tiers(
        "B", private_targets, class_counts, K=K,
        features_public=feats_public, features_all=feats_private,
    )
    tier_sizes_B = get_tier_sizes(tiers_B, K)
    print(f"[Exp1-P1] Strategy B tier sizes: {tier_sizes_B}")

    # ------------------------------------------------------------------
    # 4. Build model and wrap with GradSampleModule
    # ------------------------------------------------------------------
    model = make_model(ARCH, num_classes)
    assert validate_model_for_dp(model), "Model has incompatible layers for DP"
    gsm = GradSampleModule(model).to(device)

    optimizer = torch.optim.SGD(
        gsm.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    # ------------------------------------------------------------------
    # 5. Training loop with checkpoints
    # ------------------------------------------------------------------
    checkpoint_data = {}

    for epoch in range(1, EPOCHS + 1):
        correct, total = 0, 0
        for batch in train_noaug_loader:
            x_clean = batch[0].to(device)
            y       = batch[1].to(device)
            preds, yt = _aug_mult_step(
                gsm, optimizer, x_clean, y,
                sigma=sigma, C=C, q=q, n_train=n_train, aug_mult=AUG_MULT,
            )
            correct += (preds.argmax(1) == yt).sum().item()
            total   += yt.shape[0]
        scheduler.step()

        if epoch in CHECKPOINT_EPOCHS:
            raw = gsm._module   # unwrapped model for evaluation
            print(f"[Exp1-P1] {tag}: epoch {epoch} — computing gradient norms...")
            unclipped_norms = compute_per_sample_gradient_norms(
                gsm, eval_loader, device
            )
            clipped_norms = compute_per_sample_clipped_norms(unclipped_norms, C)

            # Per-sample loss + correct/incorrect at this checkpoint
            losses, labels, preds_eval, _ = compute_per_sample_losses(
                raw, eval_loader, device
            )
            clipping_frac = float((unclipped_norms > C).mean())

            checkpoint_data[epoch] = {
                "unclipped_norms":      unclipped_norms,            # (n_train,)
                "clipped_norms":        clipped_norms,              # (n_train,)
                "losses":               losses,                     # (n_train,)
                "correctly_classified": (preds_eval == labels).astype(bool),
                "clipping_fraction":    clipping_frac,
                "tiers_A":              tiers_A,
                "tiers_B":              tiers_B,
                "labels":               private_targets,
            }
            test_acc = evaluate(raw, test_loader, device)
            print(f"  epoch {epoch}: train={correct/max(total,1):.3f}  "
                  f"test={test_acc:.4f}  clip_frac={clipping_frac:.3f}")

    # ------------------------------------------------------------------
    # 6. Final privacy accounting
    # ------------------------------------------------------------------
    rdp = RDPAccountant()
    rdp.history = [(sigma, q, T)]
    eps_achieved = rdp.get_epsilon(delta=DELTA)

    test_acc = evaluate(gsm._module, test_loader, device)
    print(f"[Exp1-P1] {tag}: DONE  test_acc={test_acc:.4f}  ε={eps_achieved:.4f}")

    results = {
        "tag":            tag,
        "dataset":        dataset_name,
        "imbalance_ratio": imbalance_ratio,
        "seed":           seed,
        "sigma":          sigma,
        "C":              C,
        "aug_mult":       AUG_MULT,
        "lr":             LR,
        "epochs":         EPOCHS,
        "tiers_A":        tiers_A,
        "tiers_B":        tiers_B,
        "tier_sizes_A":   tier_sizes_A,
        "tier_sizes_B":   tier_sizes_B,
        "class_counts":   class_counts,
        "checkpoint_data": checkpoint_data,
        "test_acc":       test_acc,
        "epsilon":        eps_achieved,
    }

    save_results(results, result_path)
    torch.save(gsm._module.state_dict(), os.path.join(out_dir, "model_final.pt"))

    # Free GPU memory before next run
    del gsm, model, optimizer, scheduler
    import gc; gc.collect(); torch.cuda.empty_cache()

    print(f"[Exp1-P1] {tag}: saved to {result_path}")
    return results


def load_existing(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def run_all(device: torch.device = None, data_root: str = DATA_ROOT):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp1-P1] Using device: {device}")

    all_results = {}
    for (dname, ir) in DATASETS:
        for seed in SEEDS:
            r   = run_exp1(dname, ir, seed, device, data_root=data_root)
            key = f"{dname}_IR{ir:.0f}_seed{seed}"
            all_results[key] = r
    return all_results


# ---------------------------------------------------------------------------
# Summary statistics (Tab 1) — spec2 §4.2
# Reports mean, p50, p95, and Tier2/Tier0 ratio per strategy at each
# checkpoint epoch, averaged across seeds.
# ---------------------------------------------------------------------------

def compute_tab1(all_results: dict, at_epoch: int = None) -> dict:
    """
    Compute per-tier gradient norm statistics at a given checkpoint epoch.

    If at_epoch is None, uses the final checkpoint epoch in the data.
    Returns nested dict:
        tab1[dataset_strategy][tier] = {'mean', 'p50', 'p95'}
    plus tab1[dataset_strategy]['ratio_2_0'] = Tier2/Tier0 mean ratio.
    """
    # Collect per-seed stats
    accum = {}   # key → tier → {'mean_list', 'p50_list', 'p95_list'}

    for key, r in all_results.items():
        dname = r["dataset"]
        ir    = r["imbalance_ratio"]
        ckpt  = r["checkpoint_data"]

        epoch = at_epoch
        if epoch is None:
            epoch = max(e for e in ckpt.keys() if isinstance(e, int))
        if epoch not in ckpt:
            print(f"[Tab1] Warning: epoch {epoch} not in {key}, skipping.")
            continue

        clipped_norms = ckpt[epoch]["clipped_norms"]

        for strat_name, tiers in [("A", r["tiers_A"]), ("B", r["tiers_B"])]:
            combo = f"{dname}_IR{ir:.0f}_{strat_name}"
            if combo not in accum:
                accum[combo] = {k: {"mean_list": [], "p50_list": [], "p95_list": []}
                                for k in range(K)}
            for k in range(K):
                mask = tiers == k
                tn   = clipped_norms[mask]
                if len(tn) == 0:
                    continue
                accum[combo][k]["mean_list"].append(float(tn.mean()))
                accum[combo][k]["p50_list"].append(float(np.percentile(tn, 50)))
                accum[combo][k]["p95_list"].append(float(np.percentile(tn, 95)))

    # Average across seeds and compute Tier2/Tier0 ratio
    summary = {}
    for combo, tier_dict in accum.items():
        summary[combo] = {}
        for k, stats in tier_dict.items():
            if not stats["mean_list"]:
                continue
            summary[combo][k] = {
                "mean": float(np.mean(stats["mean_list"])),
                "p50":  float(np.mean(stats["p50_list"])),
                "p95":  float(np.mean(stats["p95_list"])),
            }
        # Tier2/Tier0 ratio (key diagnostic: ≥2× is the success criterion)
        if 0 in summary[combo] and 2 in summary[combo]:
            t0_mean = summary[combo][0]["mean"]
            t2_mean = summary[combo][2]["mean"]
            summary[combo]["ratio_2_0"] = t2_mean / max(t0_mean, 1e-8)
    return summary


def print_tab1(tab1: dict):
    print("\n=== Tab 1: Per-tier clipped gradient norm statistics ===")
    print(f"  (success criterion: Tier2/Tier0 ratio ≥ 2.0)\n")
    for combo, tier_dict in sorted(tab1.items()):
        ratio = tier_dict.get("ratio_2_0", float("nan"))
        flag  = " ✓" if ratio >= 2.0 else " ✗"
        print(f"{combo}  [Tier2/Tier0 = {ratio:.3f}{flag}]")
        for k in range(K):
            if k not in tier_dict:
                continue
            s = tier_dict[k]
            print(f"  Tier {k}: mean={s['mean']:.4f}  p50={s['p50']:.4f}  p95={s['p95']:.4f}")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 1 Phase 1: Gradient Structure")
    parser.add_argument("--data_root",   default="./data")
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--gpu",         type=int, default=0)
    parser.add_argument("--dataset",     default=None, choices=["cifar10"])
    parser.add_argument("--ir",          type=float, default=None)
    parser.add_argument("--seed",        type=int, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Allow single-run mode for debugging / incremental execution
    if args.dataset is not None and args.seed is not None:
        ir = args.ir if args.ir is not None else 1.0
        run_exp1(args.dataset, ir, args.seed, device,
                 data_root=args.data_root, results_dir=args.results_dir)
    else:
        all_results = run_all(device=device, data_root=args.data_root)
        tab1 = compute_tab1(all_results)
        print_tab1(tab1)
