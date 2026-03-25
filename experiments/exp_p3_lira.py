#!/usr/bin/env python3
"""
Phase 3 (Minimal): Coherence–Privacy Correlation via LiRA.

Tests whether gradient coherence (M1) correlates with per-instance privacy
leakage (LiRA score) better than gradient norm alone.

Pipeline:
  Step 0 – Select 200 target examples (100 T0, 60 T1, 40 T2)
  Step 1 – Train 16 shadow models (standard DP-SGD, 100 epochs, no augmult)
  Step 2 – Collect true-class logits for each target from each shadow
  Step 3 – Compute per-instance LiRA scores
  Step 4 – Compute M1, M3, grad norms, losses via reference model
  Step 5 – Correlations (Tab P3-1/P3-2) and figures (Fig P3-1/P3-2/P3-3)

Outputs: results/exp_p3/<tag>/

Run:
  # GPU training phase
  venv/bin/python experiments/exp_p3_lira.py --dataset cifar10 --ir 50 --gpu 0

  # Train specific shadow range (e.g. on second GPU in parallel)
  venv/bin/python experiments/exp_p3_lira.py --dataset cifar10 --ir 50 --gpu 1 \\
      --shadow_start 8 --shadow_end 15

  # CPU-only analysis after all shadows are trained
  venv/bin/python experiments/exp_p3_lira.py --dataset cifar10 --ir 50 --analysis_only
"""

import os
import sys
import gc
import argparse
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import load_datasets, make_data_loaders
from src.models import make_model, validate_model_for_dp
from src.dp_training import _clear_grad_samples, set_seed, evaluate
from src.tiers import assign_tiers, get_tier_sizes
from src.privacy_accounting import compute_sigma
from src.evaluation import save_results

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARCH          = "wrn28-2"
K             = 3
N_SHADOWS     = 16
TIER_TARGETS  = {0: 100, 1: 60, 2: 40}   # total = 200
N_TARGETS     = sum(TIER_TARGETS.values())
EPOCHS        = 100
BATCH_SIZE    = 256
C_TRAIN       = 1.0
LR            = 0.1
EPS_TARGET    = 3.0
DELTA         = 1e-5
GRAD_BATCH    = 16     # vmap batch size for per-sample gradient computation
TARGET_SEED   = 777    # for reproducible target selection
DATA_ROOT     = "./data"
RESULTS_DIR   = "./results/exp_p3"
EXP1_DIR      = "./results/exp1_p1"   # Phase-1 models


# ---------------------------------------------------------------------------
# Standard DP-SGD step (no augmult)
# ---------------------------------------------------------------------------

def _dp_step(gsm, optimizer, x, y, sigma, C, q, n_train, device):
    """
    One standard DP-SGD gradient update.
    Clips per-sample gradients to C, adds Gaussian noise, divides by q*n_train.
    """
    B = x.shape[0]
    _clear_grad_samples(gsm)
    gsm.train()

    out = gsm(x)
    F.cross_entropy(out, y, reduction="sum").backward()

    # Per-sample gradient norms
    sq_norms = torch.zeros(B, device=device)
    for p in gsm.parameters():
        if p.requires_grad and hasattr(p, "grad_sample") and p.grad_sample is not None:
            sq_norms += p.grad_sample.reshape(B, -1).pow(2).sum(1)
    scale = torch.clamp(C / sq_norms.sqrt_().clamp_(min=1e-8), max=1.0)

    # Clipped sum + noise → p.grad
    with torch.no_grad():
        for p in gsm.parameters():
            if not p.requires_grad:
                continue
            if hasattr(p, "grad_sample") and p.grad_sample is not None:
                clipped_sum = (
                    p.grad_sample.reshape(B, -1) * scale[:, None]
                ).sum(0).reshape(p.shape)
                noise = torch.randn_like(p) * (sigma * C)
                p.grad = (clipped_sum + noise) / (q * n_train)
            else:
                p.grad = torch.zeros_like(p)

    _clear_grad_samples(gsm)
    optimizer.step()
    optimizer.zero_grad()


# ---------------------------------------------------------------------------
# Step 0: Select target examples
# ---------------------------------------------------------------------------

def _select_targets(data, tiers, out_dir):
    """
    Select N_TARGETS examples stratified by tier.
    Saves target_indices.npy and returns (target_indices, tiers_target).
    """
    path = os.path.join(out_dir, "target_indices.npy")
    if os.path.exists(path):
        print("[P3] target_indices already exist, loading.")
        target_indices = np.load(path)
        return target_indices, tiers[target_indices]

    rng = np.random.default_rng(TARGET_SEED)
    selected = []
    for tier_k, n_want in TIER_TARGETS.items():
        pool = np.where(tiers == tier_k)[0]
        if len(pool) < n_want:
            raise RuntimeError(
                f"Tier {tier_k} has only {len(pool)} examples, need {n_want}."
            )
        chosen = rng.choice(pool, size=n_want, replace=False)
        selected.extend(chosen.tolist())

    target_indices = np.array(selected, dtype=np.int64)
    np.save(path, target_indices)
    print(
        f"[P3] {N_TARGETS} targets saved to {path}  "
        f"(tier counts: {[(tiers[target_indices]==k).sum() for k in range(K)]})"
    )
    return target_indices, tiers[target_indices]


# ---------------------------------------------------------------------------
# Step 1: Train shadow models
# ---------------------------------------------------------------------------

def _train_shadow(m, data, target_indices, device, shadow_dir):
    """
    Train shadow model m.
    Saves shadow_{m:02d}.pt and membership_{m:02d}.npy.
    Skips if both already exist.
    """
    from opacus.grad_sample import GradSampleModule

    model_path = os.path.join(shadow_dir, f"shadow_{m:02d}.pt")
    mem_path   = os.path.join(shadow_dir, f"membership_{m:02d}.npy")
    if os.path.exists(model_path) and os.path.exists(mem_path):
        print(f"[P3] shadow {m:02d}: already trained, skipping.")
        return

    # Bernoulli(0.5) membership for each target
    rng = np.random.RandomState(seed=1000 + m)
    membership = rng.binomial(1, 0.5, size=len(target_indices)).astype(np.int8)

    # Training set: all non-target private examples + targets where in=1
    n_priv    = len(data["private_dataset_noaug"])
    target_set = set(int(idx) for idx in target_indices)
    non_target = [i for i in range(n_priv) if i not in target_set]
    included   = [
        int(target_indices[j])
        for j in range(len(target_indices))
        if membership[j] == 1
    ]
    train_idxs = np.array(non_target + included, dtype=np.int64)
    n_train    = len(train_idxs)
    q          = BATCH_SIZE / n_train

    # Use augmented dataset for training
    subset = Subset(data["private_dataset"], train_idxs)
    use_pin = device.type == "cuda"
    loader = DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=use_pin, drop_last=True,
    )

    T     = EPOCHS * len(loader)
    sigma = compute_sigma(EPS_TARGET, DELTA, q, T)
    print(
        f"[P3] shadow {m:02d}: n_train={n_train}, q={q:.4f}, "
        f"T={T}, sigma={sigma:.4f}, mem_in={membership.sum()}"
    )

    set_seed(2000 + m)
    model = make_model(ARCH, data["num_classes"])
    assert validate_model_for_dp(model)
    gsm = GradSampleModule(model).to(device)
    opt = torch.optim.SGD(gsm.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    _, _, test_loader = make_data_loaders(data, batch_size=256)

    for epoch in range(1, EPOCHS + 1):
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            _dp_step(gsm, opt, x, y, sigma, C_TRAIN, q, n_train, device)
        sch.step()
        if epoch % 25 == 0 or epoch == EPOCHS:
            acc = evaluate(gsm._module, test_loader, device)
            print(f"  shadow {m:02d} epoch {epoch:3d}/{EPOCHS}: acc={acc:.4f}")

    torch.save(gsm._module.state_dict(), model_path)
    np.save(mem_path, membership)
    print(f"[P3] shadow {m:02d}: saved to {model_path}")

    del gsm, opt, sch, loader, subset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Step 1b: Reference model (for predictor computation)
# ---------------------------------------------------------------------------

def _get_or_train_reference(data, device, ref_path):
    """
    Return reference model state dict (CPU).
    Priority: existing ref_path → Phase-1 model → train fresh.
    """
    if os.path.exists(ref_path):
        print(f"[P3] reference model loaded from {ref_path}")
        return torch.load(ref_path, map_location="cpu")

    # Try Phase-1 model (aug_mult=8, 200 epochs — better quality)
    p1_path = os.path.join(EXP1_DIR, "cifar10_IR50_seed0", "model_final.pt")
    if os.path.exists(p1_path):
        import shutil
        shutil.copy(p1_path, ref_path)
        print(f"[P3] reference model: copied from Phase-1 model {p1_path}")
        return torch.load(ref_path, map_location="cpu")

    # Try shadow_00 as reference (already trained, targets mixed in/out)
    shadow_00 = os.path.join(os.path.dirname(ref_path), "shadow_00.pt")
    if os.path.exists(shadow_00):
        import shutil
        shutil.copy(shadow_00, ref_path)
        print(f"[P3] reference model: using shadow_00 as reference ({ref_path})")
        return torch.load(ref_path, map_location="cpu")

    # Train fresh: standard DP-SGD on full private set (all targets in)
    from opacus.grad_sample import GradSampleModule
    print("[P3] training fresh reference model (all targets included)...")

    n_train = len(data["private_dataset"])
    q       = BATCH_SIZE / n_train
    use_pin = device.type == "cuda"
    loader  = DataLoader(
        data["private_dataset"], batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=use_pin, drop_last=True,
    )
    T     = EPOCHS * len(loader)
    sigma = compute_sigma(EPS_TARGET, DELTA, q, T)

    model = make_model(ARCH, data["num_classes"])
    assert validate_model_for_dp(model)
    gsm = GradSampleModule(model).to(device)
    opt = torch.optim.SGD(gsm.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    _, _, test_loader = make_data_loaders(data, batch_size=256)

    for epoch in range(1, EPOCHS + 1):
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            _dp_step(gsm, opt, x, y, sigma, C_TRAIN, q, n_train, device)
        sch.step()
        if epoch % 25 == 0 or epoch == EPOCHS:
            acc = evaluate(gsm._module, test_loader, device)
            print(f"  reference epoch {epoch:3d}/{EPOCHS}: acc={acc:.4f}")

    state = gsm._module.state_dict()
    torch.save(state, ref_path)
    del gsm, opt, sch, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state


# ---------------------------------------------------------------------------
# Step 2: Collect logits
# ---------------------------------------------------------------------------

def _collect_logits(target_indices, shadow_dir, data, device):
    """
    For each shadow model, collect true-class logit for each target example.
    Returns (logit_matrix [N_TARGETS, N_SHADOWS], membership_matrix [N_TARGETS, N_SHADOWS]).
    """
    logit_path = os.path.join(shadow_dir, "logit_matrix.npy")
    mem_path   = os.path.join(shadow_dir, "membership_matrix.npy")
    if os.path.exists(logit_path) and os.path.exists(mem_path):
        print("[P3] logit + membership matrix already collected, loading.")
        return np.load(logit_path), np.load(mem_path)

    n_tgt = len(target_indices)
    logit_matrix      = np.full((n_tgt, N_SHADOWS), np.nan, dtype=np.float32)
    membership_matrix = np.zeros((n_tgt, N_SHADOWS), dtype=np.int8)

    # Pre-load all target images + labels (no aug, deterministic)
    target_imgs, target_lbls = [], []
    for idx in target_indices:
        x, y, _ = data["private_dataset_noaug"][int(idx)]
        target_imgs.append(x)
        target_lbls.append(int(y))
    target_imgs_t = torch.stack(target_imgs).to(device)          # (n_tgt, C, H, W)
    target_lbls_t = torch.tensor(target_lbls, device=device)     # (n_tgt,)

    for m in range(N_SHADOWS):
        model_path = os.path.join(shadow_dir, f"shadow_{m:02d}.pt")
        mem_f      = os.path.join(shadow_dir, f"membership_{m:02d}.npy")
        if not os.path.exists(model_path) or not os.path.exists(mem_f):
            print(f"[P3] shadow {m:02d} not found — skipping logit collection for this shadow.")
            continue

        state = torch.load(model_path, map_location=device)
        model = make_model(ARCH, data["num_classes"])
        model.load_state_dict(state)
        model.eval().to(device)

        with torch.no_grad():
            logits = model(target_imgs_t)                          # (n_tgt, n_classes)
            true_logits = logits[torch.arange(n_tgt, device=device), target_lbls_t]
            logit_matrix[:, m] = true_logits.cpu().numpy()

        membership_matrix[:, m] = np.load(mem_f).astype(np.int8)
        del model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[P3] shadow {m:02d}: logits collected.")

    np.save(logit_path, logit_matrix)
    np.save(mem_path, membership_matrix)
    print(f"[P3] logit matrix saved ({(~np.isnan(logit_matrix)).sum()} valid entries).")
    return logit_matrix, membership_matrix


# ---------------------------------------------------------------------------
# Step 3: LiRA scores
# ---------------------------------------------------------------------------

def _compute_lira_scores(logit_matrix, membership_matrix, out_dir):
    """
    Offline LiRA: per-example score = (mu_in - mu_out) / std_out.
    Also computes per-example AUROC.
    """
    from sklearn.metrics import roc_auc_score

    score_path = os.path.join(out_dir, "lira_scores.npy")
    auroc_path = os.path.join(out_dir, "aurocs.npy")
    if os.path.exists(score_path) and os.path.exists(auroc_path):
        print("[P3] LiRA scores already computed, loading.")
        return np.load(score_path), np.load(auroc_path)

    n_tgt = logit_matrix.shape[0]
    lira_scores = np.full(n_tgt, np.nan, dtype=np.float32)
    aurocs      = np.full(n_tgt, np.nan, dtype=np.float32)

    for j in range(n_tgt):
        valid_mask = ~np.isnan(logit_matrix[j])
        in_logits  = logit_matrix[j, valid_mask & (membership_matrix[j] == 1)]
        out_logits = logit_matrix[j, valid_mask & (membership_matrix[j] == 0)]

        if len(in_logits) < 3 or len(out_logits) < 3:
            continue

        mu_out  = out_logits.mean()
        std_out = out_logits.std(ddof=1) + 1e-8
        mu_in   = in_logits.mean()
        lira_scores[j] = (mu_in - mu_out) / std_out

        labels = np.concatenate([np.ones(len(in_logits)), np.zeros(len(out_logits))])
        scores = np.concatenate([in_logits, out_logits])
        try:
            aurocs[j] = roc_auc_score(labels, scores)
        except Exception:
            aurocs[j] = 0.5

    valid = ~np.isnan(lira_scores)
    print(
        f"[P3] LiRA scores: {valid.sum()}/{n_tgt} valid  "
        f"mean={lira_scores[valid].mean():.3f}  "
        f"std={lira_scores[valid].std():.3f}  "
        f"mean_auroc={aurocs[valid].mean():.3f}"
    )
    np.save(score_path, lira_scores)
    np.save(auroc_path, aurocs)
    return lira_scores, aurocs


# ---------------------------------------------------------------------------
# Step 4: Gradient predictors (M1, M3, norms, losses)
# ---------------------------------------------------------------------------

def _compute_predictors(ref_state, data, target_indices, device, pred_path):
    """
    Compute M1, M3, grad_norm_clip, grad_norm_raw, loss for each target.

    Algorithm:
      Pass 1 – iterate all n_train examples (via vmap), accumulate G_sum
               (mean clipped gradient) and store clipped flat grad for each target.
      Derive  – G_hat = G_sum / ||G_sum||
      Compute – M1 = cos(clipped_g_target, G_hat)
                M3 = <clipped_g_target, G> / ||G||^2 * n_train  (Chatterjee f_i)
      Pass 2  – forward-only to get losses for 200 targets.
    """
    from torch.func import vmap, grad, functional_call

    if os.path.exists(pred_path):
        print(f"[P3] predictors loaded from {pred_path}")
        return dict(np.load(pred_path))

    n_classes = data["num_classes"]
    model = make_model(ARCH, n_classes)
    model.load_state_dict(ref_state)
    model.eval().to(device)

    params      = {n: p.detach() for n, p in model.named_parameters()}
    buffers     = {n: b.detach() for n, b in model.named_buffers()}
    param_names = list(params.keys())
    d_total     = sum(params[n].numel() for n in param_names)

    def loss_single(params, x_i, y_i):
        out = functional_call(model, {**params, **buffers}, (x_i.unsqueeze(0),))
        return F.cross_entropy(out, y_i.unsqueeze(0))

    per_sample_grad_fn = vmap(grad(loss_single), in_dims=(None, 0, 0))

    # ── Pass 1: iterate full private dataset ──────────────────────────────
    print(f"[P3] Pass 1: computing per-sample gradients over all n_train "
          f"(d_total={d_total:,}, batch={GRAD_BATCH})...")

    use_pin = device.type == "cuda"
    all_loader = DataLoader(
        data["private_dataset_noaug"],
        batch_size=GRAD_BATCH, shuffle=False,
        num_workers=2, pin_memory=use_pin, drop_last=False,
    )

    target_pos_to_j  = {int(target_indices[j]): j for j in range(len(target_indices))}
    target_clipped   = {}   # pos → (d_total,) CPU tensor
    all_orig_norms   = torch.zeros(len(data["private_dataset_noaug"]))
    G_sum            = torch.zeros(d_total, dtype=torch.float64)

    sample_offset = 0
    for batch_idx, batch in enumerate(all_loader):
        x, y = batch[0].to(device), batch[1].to(device)
        B = x.shape[0]

        per_grads = per_sample_grad_fn(params, x, y)
        flat = torch.cat(
            [per_grads[n].detach().reshape(B, -1) for n in param_names], dim=1
        )  # (B, d_total) on device

        raw_norms = flat.norm(dim=1)                                  # (B,)
        clip_fac  = torch.clamp(C_TRAIN / raw_norms.clamp(min=1e-8), max=1.0)
        clipped   = flat * clip_fac[:, None]                          # (B, d_total)

        all_orig_norms[sample_offset:sample_offset + B] = raw_norms.cpu()
        G_sum += clipped.sum(0).cpu().double()

        # Store clipped grads for target positions
        clipped_cpu = clipped.cpu()
        for b in range(B):
            pos = sample_offset + b
            if pos in target_pos_to_j:
                target_clipped[pos] = clipped_cpu[b]

        sample_offset += B
        if (batch_idx + 1) % 100 == 0:
            print(f"  ... {sample_offset}/{len(data['private_dataset_noaug'])} examples")

        del flat, clipped, per_grads
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_train = sample_offset
    G_flat  = (G_sum / n_train).float()     # mean clipped gradient, (d_total,)
    G_norm  = G_flat.norm().item()
    G_hat   = G_flat / max(G_norm, 1e-8)

    print(f"[P3] G_norm={G_norm:.4f}, n_train={n_train}")

    # ── Compute M1, M3, norms for each target ─────────────────────────────
    n_tgt          = len(target_indices)
    M1             = np.zeros(n_tgt, dtype=np.float32)
    M3             = np.zeros(n_tgt, dtype=np.float32)
    grad_norm_clip = np.zeros(n_tgt, dtype=np.float32)
    grad_norm_raw  = np.zeros(n_tgt, dtype=np.float32)

    for j, pos in enumerate(target_indices):
        pos = int(pos)
        g   = target_clipped[pos]           # (d_total,) CPU float32
        gnc = g.norm().item()
        gnr = all_orig_norms[pos].item()
        dot = (g * G_hat).sum().item()

        M1[j]             = dot / max(gnc, 1e-8)
        M3[j]             = (g * G_flat).sum().item() / max(G_norm ** 2, 1e-16) * n_train
        grad_norm_clip[j] = gnc
        grad_norm_raw[j]  = gnr

    # ── Pass 2: losses for targets (forward-only) ─────────────────────────
    print("[P3] Pass 2: computing losses for 200 targets...")
    losses = np.zeros(n_tgt, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for j, pos in enumerate(target_indices):
            x, y, _ = data["private_dataset_noaug"][int(pos)]
            x = x.unsqueeze(0).to(device)
            logit = model(x)
            losses[j] = F.cross_entropy(logit, torch.tensor([y], device=device)).item()

    del model, G_sum, G_flat, G_hat, target_clipped, all_orig_norms
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pred = dict(
        M1=M1, M3=M3,
        grad_norm_clip=grad_norm_clip,
        grad_norm_raw=grad_norm_raw,
        loss=losses,
    )
    np.savez(pred_path, **pred)
    print(f"[P3] predictors saved to {pred_path}")
    return pred


# ---------------------------------------------------------------------------
# Step 5: Analysis — correlations, tables, figures
# ---------------------------------------------------------------------------

def _run_analysis(pred, lira_scores, aurocs, tiers_target, out_dir):
    """
    Produce Tab P3-1, Tab P3-2, Fig P3-1, P3-2, P3-3.
    All outputs saved to out_dir/analysis/.
    """
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression

    ana_dir = os.path.join(out_dir, "analysis")
    os.makedirs(ana_dir, exist_ok=True)

    valid = ~np.isnan(lira_scores)
    lira  = lira_scores[valid]
    t     = tiers_target[valid]

    print(f"\n[P3] Analysis: {valid.sum()} valid targets  "
          f"overall LiRA mean={lira.mean():.3f}  std={lira.std():.3f}")
    print(f"     Overall mean AUROC: {aurocs[valid].mean():.3f}")

    # ── Tab P3-1: Spearman ρ table ────────────────────────────────────────
    predictor_names = ["M1", "M3", "grad_norm_clip", "grad_norm_raw", "loss"]
    predictor_labels = {
        "M1":             "M1 (global coherence)",
        "M3":             "M3 (Chatterjee f_i)",
        "grad_norm_clip": "grad_norm_clip",
        "grad_norm_raw":  "grad_norm_raw",
        "loss":           "loss",
    }

    lines = ["=== Tab P3-1: Spearman ρ vs LiRA score ===",
             f"{'Predictor':<24} {'rho':>8} {'p-value':>10} {'note'}"]
    lines.append("-" * 55)
    rho_results = {}
    for name in predictor_names:
        v = pred[name][valid]
        rho, p = spearmanr(v, lira)
        sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
        lines.append(f"  {predictor_labels[name]:<22} {rho:>+8.3f} {p:>10.4f}{sig}")
        rho_results[name] = (rho, p)
    lines.append("")

    # Success criteria S1, S2
    rho_m1,   _ = rho_results["M1"]
    rho_norm, _ = rho_results["grad_norm_raw"]
    delta_rho = abs(rho_m1) - abs(rho_norm)
    s1 = "PASS" if delta_rho >= 0.15 else "FAIL"
    s2 = "PASS" if abs(rho_m1) > 0.3 else "FAIL"
    lines += [
        f"  S1 |ρ(M1)| > |ρ(norm)| by ≥0.15:  Δρ = {delta_rho:+.3f}  [{s1}]",
        f"  S2 |ρ(M1)| > 0.3:                  |ρ| = {abs(rho_m1):.3f}   [{s2}]",
        "",
    ]

    # ── Tab P3-2: R² analysis ─────────────────────────────────────────────
    y_lr = lira
    def r2(X):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        return LinearRegression().fit(X, y_lr).score(X, y_lr)

    X_loss  = pred["loss"][valid]
    X_m1    = pred["M1"][valid]
    X_norm  = pred["grad_norm_raw"][valid]

    r2_loss      = r2(X_loss)
    r2_m1        = r2(X_m1)
    r2_norm      = r2(X_norm)
    r2_loss_m1   = r2(np.column_stack([X_loss, X_m1]))
    r2_loss_norm = r2(np.column_stack([X_loss, X_norm]))

    s4 = "PASS" if (r2_loss_m1 - r2_loss) > 0 else "FAIL"
    lines += [
        "=== Tab P3-2: R² analysis ===",
        f"  loss only:       R² = {r2_loss:.3f}",
        f"  M1 only:         R² = {r2_m1:.3f}",
        f"  norm only:       R² = {r2_norm:.3f}",
        f"  loss + M1:       R² = {r2_loss_m1:.3f}   (ΔR² = {r2_loss_m1 - r2_loss:+.3f})",
        f"  loss + norm:     R² = {r2_loss_norm:.3f}   (ΔR² = {r2_loss_norm - r2_loss:+.3f})",
        f"  S4 M1 adds power beyond loss: [{s4}]",
        "",
    ]

    # ── Per-tier LiRA stats ───────────────────────────────────────────────
    tier_names = ["T0 (head)", "T1 (mid)", "T2 (tail)"]
    lines.append("=== Per-tier LiRA statistics ===")
    lines.append(f"{'Tier':<12} {'n':>5} {'mean':>8} {'median':>8} {'std':>8}")
    lines.append("-" * 45)
    tier_medians = []
    for k in range(K):
        mask_k = t == k
        v_k = lira[mask_k]
        med = float(np.median(v_k)) if len(v_k) > 0 else float("nan")
        tier_medians.append(med)
        lines.append(
            f"  {tier_names[k]:<10} {mask_k.sum():>5} "
            f"{v_k.mean():>8.3f} {med:>8.3f} {v_k.std():>8.3f}"
        )
    s3 = "PASS" if (len(tier_medians) >= 3 and
                    tier_medians[2] > tier_medians[0]) else "FAIL"
    lines.append(f"  S3 T2 median > T0 median: "
                 f"{tier_medians[2]:.3f} > {tier_medians[0]:.3f}  [{s3}]")
    lines.append("")

    tab_text = "\n".join(lines)
    print(tab_text)
    tab_path = os.path.join(ana_dir, "tables.txt")
    with open(tab_path, "w") as f:
        f.write(tab_text)

    # ── Figures ───────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[P3] matplotlib not available — skipping figures.")
        return

    colors     = ["tab:blue", "tab:orange", "tab:red"]
    tier_labs  = ["Tier 0 (head)", "Tier 1 (mid)", "Tier 2 (tail)"]

    def scatter_by_tier(ax, x_vals, y_vals, tiers_v, title, xlabel):
        for k in range(K):
            m = tiers_v == k
            ax.scatter(x_vals[m], y_vals[m],
                       c=colors[k], alpha=0.5, s=15, label=tier_labs[k])
        rho_v, _ = spearmanr(x_vals, y_vals)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("LiRA score", fontsize=9)
        ax.set_title(f"{title}  (ρ = {rho_v:+.3f})", fontsize=9)
        ax.legend(fontsize=7)

    # Fig P3-1: 4-panel scatter
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        ("grad_norm_clip", "||ḡ_i|| (clipped norm)"),
        ("M1",             "cos θ_i^global  (M1)"),
        ("loss",           "Loss"),
        ("M3",             "f_i  (Chatterjee M3)"),
    ]
    for ax, (key, lbl) in zip(axes.flat, panels):
        scatter_by_tier(ax, pred[key][valid], lira, t, lbl, lbl)
    fig.suptitle("Gradient Predictors vs LiRA Score (CIFAR-10-LT IR=50)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(ana_dir, "fig_P3_1.png"), dpi=150)
    plt.close(fig)

    # Fig P3-2: M1 vs LiRA hero figure
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter_by_tier(ax, pred["M1"][valid], lira, t, "", "cos θ_i^global  (M1)")
    x_line = np.linspace(pred["M1"][valid].min(), pred["M1"][valid].max(), 200)
    coeffs = np.polyfit(pred["M1"][valid], lira, 1)
    ax.plot(x_line, np.polyval(coeffs, x_line), "k--", alpha=0.7, lw=1.5)
    rho_v, p_v = spearmanr(pred["M1"][valid], lira)
    ax.set_title(
        f"Gradient Coherence vs Per-Instance Privacy Vulnerability\n"
        f"Spearman ρ = {rho_v:+.3f},  p = {p_v:.2e}",
        fontsize=10,
    )
    ax.set_xlabel("cos θ_i^global  (M1 — gradient coherence)", fontsize=10)
    ax.set_ylabel("LiRA score  (higher = more vulnerable)", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ana_dir, "fig_P3_2.png"), dpi=150)
    plt.close(fig)

    # Fig P3-3: LiRA boxplot by tier
    fig, ax = plt.subplots(figsize=(6, 4))
    data_by_tier = [lira[t == k] for k in range(K)]
    bp = ax.boxplot(data_by_tier, labels=tier_labs, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("LiRA score", fontsize=10)
    ax.set_title("Per-instance vulnerability by tier", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(ana_dir, "fig_P3_3.png"), dpi=150)
    plt.close(fig)

    print(f"[P3] figures saved to {ana_dir}/")


# ---------------------------------------------------------------------------
# Main experiment orchestrator
# ---------------------------------------------------------------------------

def run_exp_p3(
    dataset_name: str,
    imbalance_ratio: float,
    seed: int,
    device: torch.device,
    shadow_start: int = 0,
    shadow_end: int = N_SHADOWS - 1,
    analysis_only: bool = False,
    data_root: str = DATA_ROOT,
    results_dir: str = RESULTS_DIR,
):
    tag     = f"{dataset_name}_IR{imbalance_ratio:.0f}_seed{seed}"
    out_dir = os.path.join(results_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    shadow_dir = os.path.join(out_dir, "shadows")
    os.makedirs(shadow_dir, exist_ok=True)

    set_seed(seed)

    # ── Load data + tiers ─────────────────────────────────────────────────
    print(f"[P3] loading data: {dataset_name} IR={imbalance_ratio}")
    data = load_datasets(
        dataset_name=dataset_name, data_root=data_root,
        imbalance_ratio=imbalance_ratio,
        public_frac=0.1, split_seed=42,
    )
    private_targets = np.array(data["private_dataset"].targets)
    class_counts    = data["class_counts"]
    tiers           = assign_tiers("A", private_targets, class_counts, K=K)

    tier_sizes = get_tier_sizes(tiers, K)
    print(f"[P3] tier sizes: {tier_sizes}  (T0=head, T2=tail)")

    # ── Step 0: Select targets ────────────────────────────────────────────
    target_indices, tiers_target = _select_targets(data, tiers, out_dir)

    if analysis_only:
        print("[P3] --analysis_only: skipping training and logit collection.")
    else:
        # ── Step 1: Train shadow models ───────────────────────────────────
        for m in range(shadow_start, shadow_end + 1):
            _train_shadow(m, data, target_indices, device, shadow_dir)

        # ── Step 1b: Reference model ──────────────────────────────────────
        ref_path = os.path.join(out_dir, "reference.pt")
        ref_state = _get_or_train_reference(data, device, ref_path)

        # ── Step 2: Collect logits ────────────────────────────────────────
        logit_matrix, membership_matrix = _collect_logits(
            target_indices, shadow_dir, data, device
        )
    # If analysis_only, load existing logits
    if analysis_only:
        logit_path = os.path.join(shadow_dir, "logit_matrix.npy")
        mem_path   = os.path.join(shadow_dir, "membership_matrix.npy")
        if not os.path.exists(logit_path):
            raise FileNotFoundError(
                f"logit_matrix.npy not found at {logit_path}. "
                "Run without --analysis_only first to collect logits."
            )
        logit_matrix      = np.load(logit_path)
        membership_matrix = np.load(mem_path)

    # ── Step 3: LiRA scores ───────────────────────────────────────────────
    lira_scores, aurocs = _compute_lira_scores(
        logit_matrix, membership_matrix, out_dir
    )

    # ── Step 4: Gradient predictors ───────────────────────────────────────
    pred_path = os.path.join(out_dir, "predictors.npz")
    if not os.path.exists(pred_path):
        ref_path  = os.path.join(out_dir, "reference.pt")
        if not os.path.exists(ref_path):
            # Try to find/train reference
            ref_state = _get_or_train_reference(data, device, ref_path)
        else:
            ref_state = torch.load(ref_path, map_location="cpu")
        pred = _compute_predictors(ref_state, data, target_indices, device, pred_path)
    else:
        pred = dict(np.load(pred_path))

    # ── Step 5: Analysis ──────────────────────────────────────────────────
    _run_analysis(pred, lira_scores, aurocs, tiers_target, out_dir)

    # ── Save full results ─────────────────────────────────────────────────
    results = {
        "tag":             tag,
        "dataset":         dataset_name,
        "imbalance_ratio": imbalance_ratio,
        "seed":            seed,
        "target_indices":  target_indices,
        "tiers_target":    tiers_target,
        "lira_scores":     lira_scores,
        "aurocs":          aurocs,
        "predictors":      {k: v for k, v in pred.items()},
    }
    save_results(results, os.path.join(out_dir, "results.pkl"))
    print(f"[P3] experiment complete — results in {out_dir}")
    return results


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_all(device=None, data_root=DATA_ROOT, results_dir=RESULTS_DIR):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Primary: CIFAR-10-LT (IR=50)
    run_exp_p3("cifar10", 50.0, 0, device,
               data_root=data_root, results_dir=results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Coherence–Privacy via LiRA")
    parser.add_argument("--dataset",       default="cifar10")
    parser.add_argument("--ir",            type=float, default=50.0)
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--gpu",           type=int,   default=0)
    parser.add_argument("--data_root",     default=DATA_ROOT)
    parser.add_argument("--results_dir",   default=RESULTS_DIR)
    parser.add_argument(
        "--shadow_start", type=int, default=0,
        help="First shadow model index to train (inclusive). "
             "Use with --shadow_end to train a subset on a specific GPU.",
    )
    parser.add_argument(
        "--shadow_end", type=int, default=N_SHADOWS - 1,
        help="Last shadow model index to train (inclusive).",
    )
    parser.add_argument(
        "--analysis_only", action="store_true",
        help="Skip training and logit collection; only run Steps 3-5 "
             "(requires existing logit_matrix.npy and predictors.npz).",
    )
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    run_exp_p3(
        dataset_name=args.dataset,
        imbalance_ratio=args.ir,
        seed=args.seed,
        device=device,
        shadow_start=args.shadow_start,
        shadow_end=args.shadow_end,
        analysis_only=args.analysis_only,
        data_root=args.data_root,
        results_dir=args.results_dir,
    )
