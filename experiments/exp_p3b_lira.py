#!/usr/bin/env python3
"""
Phase 3b: Coherence–Privacy Correlation on Non-Private Models.

Tests the same H1 hypothesis (does gradient coherence predict per-instance
privacy?) on standard (non-private) models where accuracy is high (~85-93%)
and LiRA has strong membership signal.

Key differences from Phase 3:
  - No DP: standard SGD, no clipping, no noise
  - WRN-28-2 with BatchNorm (no GroupNorm needed)
  - Gradient norms are NOT clipped — may stratify by tier (unlike Phase 2)
  - Extra figure: Fig P3b-4 — raw gradient norm histogram by tier

Run:
  venv/bin/python experiments/exp_p3b_lira.py --dataset cifar10 --ir 50 --gpu 0

  # Parallel (split shadows across GPUs)
  venv/bin/python experiments/exp_p3b_lira.py --ir 50 --gpu 0 --shadow_start 0 --shadow_end 7
  venv/bin/python experiments/exp_p3b_lira.py --ir 50 --gpu 1 --shadow_start 8 --shadow_end 15

  # Analysis only (CPU, after all shadows trained)
  venv/bin/python experiments/exp_p3b_lira.py --ir 50 --analysis_only
"""

import os
import sys
import gc
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import load_datasets, make_data_loaders
from src.dp_training import set_seed, evaluate
from src.tiers import assign_tiers, get_tier_sizes
from src.evaluation import save_results

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K            = 3
N_SHADOWS    = 16
TIER_TARGETS = {0: 100, 1: 60, 2: 40}    # total = 200
N_TARGETS    = sum(TIER_TARGETS.values())
EPOCHS       = 200
BATCH_SIZE   = 128
LR           = 0.1
WD           = 5e-4
GRAD_BATCH   = 16     # vmap batch for per-sample gradient computation
TARGET_SEED  = 777
DATA_ROOT    = "./data"
RESULTS_DIR  = "./results/exp_p3b"
P3_DIR       = "./results/exp_p3"    # check for reusable target_indices


# ---------------------------------------------------------------------------
# WideResNet-28-2 with standard BatchNorm (no GroupNorm)
# ---------------------------------------------------------------------------

class _WideBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=stride,
                               padding=1, bias=False)
        self.shortcut = (
            nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)
            if stride != 1 or in_planes != out_planes
            else nn.Identity()
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.shortcut(x)


class WRN28_2_BN(nn.Module):
    """WideResNet-28-2 with BatchNorm2d. ~1.47M params. For non-private training."""

    def __init__(self, num_classes=10):
        super().__init__()
        c = [16, 32, 64, 128]
        n = 4   # (28 - 4) // 6

        self.conv1  = nn.Conv2d(3, c[0], 3, padding=1, bias=False)
        self.layer1 = self._make(c[0], c[1], n, stride=1)
        self.layer2 = self._make(c[1], c[2], n, stride=2)
        self.layer3 = self._make(c[2], c[3], n, stride=2)
        self.bn     = nn.BatchNorm2d(c[3])
        self.fc     = nn.Linear(c[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    @staticmethod
    def _make(in_planes, out_planes, n, stride):
        layers = [_WideBlock(in_planes, out_planes, stride)]
        for _ in range(1, n):
            layers.append(_WideBlock(out_planes, out_planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def _make_model(num_classes):
    return WRN28_2_BN(num_classes=num_classes)


# ---------------------------------------------------------------------------
# Standard (non-private) training step
# ---------------------------------------------------------------------------

def _train_model(model, loader, epochs, device, tag="model"):
    """Train model with standard SGD + cosine LR. Returns final test accuracy."""
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    model.train()
    for epoch in range(1, epochs + 1):
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        sch.step()
        if epoch % 50 == 0 or epoch == epochs:
            model.eval()
            # Quick train-batch accuracy proxy (full eval done at caller)
            model.train()
            print(f"  {tag} epoch {epoch:3d}/{epochs}")

    return model


# ---------------------------------------------------------------------------
# Step 0: Select target examples
# ---------------------------------------------------------------------------

def _select_targets(data, tiers, out_dir):
    """
    Select 200 target examples stratified by tier.
    Reuses Phase-3 target_indices.npy if it exists (for comparability).
    """
    path = os.path.join(out_dir, "target_indices.npy")
    if os.path.exists(path):
        target_indices = np.load(path)
        print(f"[P3b] target_indices loaded from {path}")
        return target_indices, tiers[target_indices]

    # Try reusing Phase-3 targets
    p3_path = os.path.join(P3_DIR, "cifar10_IR50_seed0", "target_indices.npy")
    if os.path.exists(p3_path):
        import shutil
        shutil.copy(p3_path, path)
        target_indices = np.load(path)
        print(f"[P3b] reusing Phase-3 target_indices from {p3_path}")
        return target_indices, tiers[target_indices]

    # Select fresh
    rng = np.random.default_rng(TARGET_SEED)
    selected = []
    for tier_k, n_want in TIER_TARGETS.items():
        pool = np.where(tiers == tier_k)[0]
        if len(pool) < n_want:
            raise RuntimeError(f"Tier {tier_k}: only {len(pool)} examples, need {n_want}.")
        selected.extend(rng.choice(pool, size=n_want, replace=False).tolist())

    target_indices = np.array(selected, dtype=np.int64)
    np.save(path, target_indices)
    print(
        f"[P3b] {N_TARGETS} targets saved  "
        f"(tier counts: {[(tiers[target_indices]==k).sum() for k in range(K)]})"
    )
    return target_indices, tiers[target_indices]


# ---------------------------------------------------------------------------
# Step 1: Train shadow models
# ---------------------------------------------------------------------------

def _train_shadow(m, data, target_indices, device, shadow_dir):
    """
    Train non-private shadow model m.
    Saves shadow_np_{m:02d}.pt and membership_np_{m:02d}.npy.
    """
    model_path = os.path.join(shadow_dir, f"shadow_np_{m:02d}.pt")
    mem_path   = os.path.join(shadow_dir, f"membership_np_{m:02d}.npy")
    if os.path.exists(model_path) and os.path.exists(mem_path):
        print(f"[P3b] shadow {m:02d}: already trained, skipping.")
        return

    rng = np.random.RandomState(seed=1000 + m)
    membership = rng.binomial(1, 0.5, size=len(target_indices)).astype(np.int8)

    n_priv     = len(data["private_dataset"])
    target_set = set(int(idx) for idx in target_indices)
    non_target = [i for i in range(n_priv) if i not in target_set]
    included   = [int(target_indices[j]) for j in range(len(target_indices))
                  if membership[j] == 1]
    train_idxs = np.array(non_target + included, dtype=np.int64)
    n_train    = len(train_idxs)

    print(f"[P3b] shadow {m:02d}: n_train={n_train}, mem_in={membership.sum()}")

    # Use augmented dataset for training (has random crop + flip transforms)
    use_pin = device.type == "cuda"
    loader  = DataLoader(
        Subset(data["private_dataset"], train_idxs),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=use_pin, drop_last=True,
    )

    set_seed(2000 + m)
    model = _make_model(data["num_classes"]).to(device)
    _train_model(model, loader, EPOCHS, device, tag=f"shadow {m:02d}")

    # Final test accuracy
    _, _, test_loader = make_data_loaders(data, batch_size=256)
    acc = evaluate(model, test_loader, device)
    print(f"[P3b] shadow {m:02d}: final acc={acc:.4f}")

    torch.save(model.state_dict(), model_path)
    np.save(mem_path, membership)

    del model, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Step 1b: Reference model (for predictor computation)
# ---------------------------------------------------------------------------

def _get_or_train_reference(data, device, ref_path):
    """Return reference model state dict. Trains if not found."""
    if os.path.exists(ref_path):
        print(f"[P3b] reference model loaded from {ref_path}")
        return torch.load(ref_path, map_location="cpu")

    print("[P3b] training reference model (all targets included)...")
    use_pin = device.type == "cuda"
    loader  = DataLoader(
        data["private_dataset"],
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=use_pin, drop_last=True,
    )

    set_seed(9999)
    model = _make_model(data["num_classes"]).to(device)
    _train_model(model, loader, EPOCHS, device, tag="reference")

    _, _, test_loader = make_data_loaders(data, batch_size=256)
    acc = evaluate(model, test_loader, device)
    print(f"[P3b] reference: final acc={acc:.4f}")

    state = model.state_dict()
    torch.save(state, ref_path)
    del model, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state


# ---------------------------------------------------------------------------
# Step 2: Collect logits
# ---------------------------------------------------------------------------

def _collect_logits(target_indices, shadow_dir, data, device):
    """Collect true-class logits from each shadow for each target (no aug)."""
    logit_path = os.path.join(shadow_dir, "logit_matrix_np.npy")
    mem_path   = os.path.join(shadow_dir, "membership_matrix_np.npy")
    if os.path.exists(logit_path) and os.path.exists(mem_path):
        print("[P3b] logit matrix already collected, loading.")
        return np.load(logit_path), np.load(mem_path)

    n_tgt = len(target_indices)
    logit_matrix      = np.full((n_tgt, N_SHADOWS), np.nan, dtype=np.float32)
    membership_matrix = np.zeros((n_tgt, N_SHADOWS), dtype=np.int8)

    # Pre-load all target images (no aug, deterministic)
    target_imgs, target_lbls = [], []
    for idx in target_indices:
        x, y, _ = data["private_dataset_noaug"][int(idx)]
        target_imgs.append(x)
        target_lbls.append(int(y))
    target_imgs_t = torch.stack(target_imgs).to(device)
    target_lbls_t = torch.tensor(target_lbls, device=device)

    for m in range(N_SHADOWS):
        model_path = os.path.join(shadow_dir, f"shadow_np_{m:02d}.pt")
        mem_f      = os.path.join(shadow_dir, f"membership_np_{m:02d}.npy")
        if not os.path.exists(model_path) or not os.path.exists(mem_f):
            print(f"[P3b] shadow {m:02d} not found, skipping.")
            continue

        state = torch.load(model_path, map_location=device)
        model = _make_model(data["num_classes"])
        model.load_state_dict(state)
        model.eval().to(device)

        with torch.no_grad():
            logits      = model(target_imgs_t)                   # (n_tgt, C)
            true_logits = logits[torch.arange(n_tgt, device=device), target_lbls_t]
            logit_matrix[:, m] = true_logits.cpu().numpy()

        membership_matrix[:, m] = np.load(mem_f).astype(np.int8)
        del model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[P3b] shadow {m:02d}: logits collected.")

    np.save(logit_path, logit_matrix)
    np.save(mem_path, membership_matrix)
    return logit_matrix, membership_matrix


# ---------------------------------------------------------------------------
# Step 3: LiRA scores
# ---------------------------------------------------------------------------

def _compute_lira_scores(logit_matrix, membership_matrix, out_dir):
    """Offline LiRA: (mu_in - mu_out) / std_out per example."""
    from sklearn.metrics import roc_auc_score

    score_path = os.path.join(out_dir, "lira_scores_np.npy")
    auroc_path = os.path.join(out_dir, "aurocs_np.npy")
    if os.path.exists(score_path) and os.path.exists(auroc_path):
        print("[P3b] LiRA scores already computed, loading.")
        return np.load(score_path), np.load(auroc_path)

    n_tgt       = logit_matrix.shape[0]
    lira_scores = np.full(n_tgt, np.nan, dtype=np.float32)
    aurocs      = np.full(n_tgt, np.nan, dtype=np.float32)

    for j in range(n_tgt):
        valid      = ~np.isnan(logit_matrix[j])
        in_logits  = logit_matrix[j, valid & (membership_matrix[j] == 1)]
        out_logits = logit_matrix[j, valid & (membership_matrix[j] == 0)]
        if len(in_logits) < 3 or len(out_logits) < 3:
            continue
        lira_scores[j] = (in_logits.mean() - out_logits.mean()) / (out_logits.std(ddof=1) + 1e-8)
        labels = np.concatenate([np.ones(len(in_logits)), np.zeros(len(out_logits))])
        try:
            aurocs[j] = roc_auc_score(labels, np.concatenate([in_logits, out_logits]))
        except Exception:
            aurocs[j] = 0.5

    valid = ~np.isnan(lira_scores)
    mean_auroc = aurocs[valid].mean() if valid.any() else float("nan")
    print(
        f"[P3b] LiRA: {valid.sum()}/{n_tgt} valid  "
        f"mean={lira_scores[valid].mean():.3f}  std={lira_scores[valid].std():.3f}  "
        f"mean_AUROC={mean_auroc:.3f}"
    )
    s0 = "PASS" if mean_auroc > 0.6 else "FAIL"
    print(f"[P3b] S0 (mean AUROC > 0.6): [{s0}]  AUROC={mean_auroc:.3f}")

    np.save(score_path, lira_scores)
    np.save(auroc_path, aurocs)
    return lira_scores, aurocs


# ---------------------------------------------------------------------------
# Step 4: Gradient predictors (M1, M3, raw norms, losses)
# ---------------------------------------------------------------------------

def _compute_predictors(ref_state, data, target_indices, device, pred_path):
    """
    Compute M1, M3, grad_norm_raw, loss for each target using the reference model.

    No clipping — gradients are used raw. Aggregate G = mean over all n_train.
    M1 = cos(g_i, G_hat),  M3 = <g_i, G> / ||G||^2 * n_train.
    """
    from torch.func import vmap, grad, functional_call

    if os.path.exists(pred_path):
        print(f"[P3b] predictors loaded from {pred_path}")
        return dict(np.load(pred_path))

    n_classes = data["num_classes"]
    model = _make_model(n_classes)
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

    # ── Pass 1: accumulate G_sum over all n_train ─────────────────────────
    print(f"[P3b] Pass 1: per-sample gradients over all n_train "
          f"(d={d_total:,}, batch={GRAD_BATCH})...")

    use_pin    = device.type == "cuda"
    all_loader = DataLoader(
        data["private_dataset_noaug"],
        batch_size=GRAD_BATCH, shuffle=False,
        num_workers=2, pin_memory=use_pin, drop_last=False,
    )

    target_pos_to_j = {int(target_indices[j]): j for j in range(len(target_indices))}
    target_raw_grads = {}         # pos → (d_total,) CPU
    target_raw_norms = {}         # pos → float
    G_sum = torch.zeros(d_total, dtype=torch.float64)

    sample_offset = 0
    for batch_idx, batch in enumerate(all_loader):
        x, y = batch[0].to(device), batch[1].to(device)
        B = x.shape[0]

        per_grads = per_sample_grad_fn(params, x, y)
        flat = torch.cat(
            [per_grads[n].detach().reshape(B, -1) for n in param_names], dim=1
        )  # (B, d_total)

        G_sum += flat.sum(0).cpu().double()

        flat_cpu = flat.cpu()
        for b in range(B):
            pos = sample_offset + b
            if pos in target_pos_to_j:
                target_raw_grads[pos] = flat_cpu[b]
                target_raw_norms[pos] = flat_cpu[b].norm().item()

        sample_offset += B
        if (batch_idx + 1) % 100 == 0:
            print(f"  ... {sample_offset}/{len(data['private_dataset_noaug'])}")

        del flat, per_grads
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_train = sample_offset
    G_flat  = (G_sum / n_train).float()
    G_norm  = G_flat.norm().item()
    G_hat   = G_flat / max(G_norm, 1e-8)
    print(f"[P3b] G_norm={G_norm:.4f}, n_train={n_train}")

    # ── Compute M1, M3, norms for each target ─────────────────────────────
    n_tgt         = len(target_indices)
    M1            = np.zeros(n_tgt, dtype=np.float32)
    M3            = np.zeros(n_tgt, dtype=np.float32)
    grad_norm_raw = np.zeros(n_tgt, dtype=np.float32)

    for j, pos in enumerate(target_indices):
        pos = int(pos)
        g   = target_raw_grads[pos]
        gnr = target_raw_norms[pos]
        dot = (g * G_hat).sum().item()

        M1[j]            = dot / max(gnr, 1e-8)
        M3[j]            = (g * G_flat).sum().item() / max(G_norm ** 2, 1e-16) * n_train
        grad_norm_raw[j] = gnr

    # ── Pass 2: losses for 200 targets ────────────────────────────────────
    print("[P3b] Pass 2: losses for 200 targets...")
    losses = np.zeros(n_tgt, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for j, pos in enumerate(target_indices):
            x, y, _ = data["private_dataset_noaug"][int(pos)]
            x = x.unsqueeze(0).to(device)
            logit = model(x)
            losses[j] = F.cross_entropy(logit, torch.tensor([y], device=device)).item()

    del model, G_sum, G_flat, G_hat, target_raw_grads, target_raw_norms
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pred = dict(M1=M1, M3=M3, grad_norm_raw=grad_norm_raw, loss=losses)
    np.savez(pred_path, **pred)
    print(f"[P3b] predictors saved to {pred_path}")
    return pred


# ---------------------------------------------------------------------------
# Step 5: Analysis
# ---------------------------------------------------------------------------

def _run_analysis(pred, lira_scores, aurocs, tiers_target, out_dir):
    """Tab P3b-1, P3b-2, Fig P3b-1 through P3b-4."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression

    ana_dir = os.path.join(out_dir, "analysis")
    os.makedirs(ana_dir, exist_ok=True)

    valid = ~np.isnan(lira_scores)
    lira  = lira_scores[valid]
    t     = tiers_target[valid]
    mean_auroc = aurocs[valid].mean() if valid.any() else float("nan")

    # ── Tab P3b-1: Spearman ρ ─────────────────────────────────────────────
    predictor_names = ["M1", "M3", "grad_norm_raw", "loss"]
    lines = [
        f"=== Tab P3b-1: Spearman ρ vs LiRA score  "
        f"(n={valid.sum()}, mean_AUROC={mean_auroc:.3f}) ===",
        f"{'Predictor':<22} {'rho':>8} {'p-value':>10}",
        "-" * 45,
    ]
    rho_results = {}
    for name in predictor_names:
        rho, p = spearmanr(pred[name][valid], lira)
        sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
        lines.append(f"  {name:<20} {rho:>+8.3f} {p:>10.4f}{sig}")
        rho_results[name] = (rho, p)
    lines.append("")

    rho_m1   = abs(rho_results["M1"][0])
    rho_norm = abs(rho_results["grad_norm_raw"][0])
    delta_rho = rho_m1 - rho_norm
    s0 = "PASS" if mean_auroc > 0.6  else "FAIL"
    s1 = "PASS" if delta_rho  >= 0.10 else "FAIL"
    s2 = "PASS" if rho_m1     >  0.2  else "FAIL"
    lines += [
        f"  S0 mean AUROC > 0.6:              {mean_auroc:.3f}  [{s0}]",
        f"  S1 |ρ(M1)| > |ρ(norm)| by ≥0.10: Δρ = {delta_rho:+.3f}  [{s1}]",
        f"  S2 |ρ(M1)| > 0.2:                {rho_m1:.3f}  [{s2}]",
        "",
    ]

    # ── Tab P3b-2: R² analysis ────────────────────────────────────────────
    y_lr = lira
    def r2(X):
        X = np.atleast_2d(X).T if X.ndim == 1 else X
        return LinearRegression().fit(X, y_lr).score(X, y_lr)

    X_loss = pred["loss"][valid]
    X_m1   = pred["M1"][valid]
    X_norm = pred["grad_norm_raw"][valid]

    r2_loss      = r2(X_loss)
    r2_m1        = r2(X_m1)
    r2_norm      = r2(X_norm)
    r2_loss_m1   = r2(np.column_stack([X_loss, X_m1]))
    r2_loss_norm = r2(np.column_stack([X_loss, X_norm]))

    s4 = "PASS" if (r2_loss_m1 - r2_loss) > 0 else "FAIL"
    lines += [
        "=== Tab P3b-2: R² analysis ===",
        f"  loss only:      R² = {r2_loss:.3f}",
        f"  M1 only:        R² = {r2_m1:.3f}",
        f"  norm only:      R² = {r2_norm:.3f}",
        f"  loss + M1:      R² = {r2_loss_m1:.3f}   (ΔR² = {r2_loss_m1 - r2_loss:+.3f})",
        f"  loss + norm:    R² = {r2_loss_norm:.3f}   (ΔR² = {r2_loss_norm - r2_loss:+.3f})",
        f"  S4 M1 adds beyond loss: [{s4}]",
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
        v_k    = lira[mask_k]
        med    = float(np.median(v_k)) if len(v_k) else float("nan")
        tier_medians.append(med)
        lines.append(
            f"  {tier_names[k]:<10} {mask_k.sum():>5} "
            f"{v_k.mean():>8.3f} {med:>8.3f} {v_k.std():>8.3f}"
        )
    s3 = "PASS" if (len(tier_medians) >= 3 and tier_medians[2] > tier_medians[0]) else "FAIL"
    lines.append(f"  S3 T2 median > T0 median: "
                 f"{tier_medians[2]:.3f} > {tier_medians[0]:.3f}  [{s3}]")
    lines.append("")

    # ── Per-tier gradient norm stats ──────────────────────────────────────
    lines.append("=== Per-tier gradient norm (raw, unclipped) ===")
    lines.append(f"{'Tier':<12} {'mean':>10} {'median':>10} {'p25':>10} {'p75':>10}")
    lines.append("-" * 50)
    for k in range(K):
        mask_k = tiers_target[valid] == k
        v_k    = pred["grad_norm_raw"][valid][mask_k]
        if len(v_k):
            lines.append(
                f"  {tier_names[k]:<10} "
                f"{v_k.mean():>10.3f} {np.median(v_k):>10.3f} "
                f"{np.percentile(v_k,25):>10.3f} {np.percentile(v_k,75):>10.3f}"
            )
    lines.append("")

    tab_text = "\n".join(lines)
    print(tab_text)
    with open(os.path.join(ana_dir, "tables.txt"), "w") as f:
        f.write(tab_text)

    # ── Figures ───────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[P3b] matplotlib not available — skipping figures.")
        return

    colors    = ["tab:blue", "tab:orange", "tab:red"]
    tier_labs = ["Tier 0 (head)", "Tier 1 (mid)", "Tier 2 (tail)"]

    def scatter_by_tier(ax, x_vals, y_vals, t_vals, xlabel):
        for k in range(K):
            m = t_vals == k
            ax.scatter(x_vals[m], y_vals[m], c=colors[k], alpha=0.5, s=15,
                       label=tier_labs[k])
        rho_v, _ = spearmanr(x_vals, y_vals)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("LiRA score", fontsize=9)
        ax.set_title(f"ρ = {rho_v:+.3f}", fontsize=9)
        ax.legend(fontsize=7)

    # Fig P3b-1: 4-panel scatter
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        ("grad_norm_raw", "||g_i|| (raw norm)"),
        ("M1",            "cos θ_i^global  (M1)"),
        ("loss",          "Loss"),
        ("M3",            "f_i  (Chatterjee M3)"),
    ]
    for ax, (key, lbl) in zip(axes.flat, panels):
        scatter_by_tier(ax, pred[key][valid], lira, t, lbl)
    fig.suptitle("Gradient Predictors vs LiRA Score — Non-Private (CIFAR-10-LT IR=50)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(ana_dir, "fig_P3b_1.png"), dpi=150)
    plt.close(fig)

    # Fig P3b-2: M1 vs LiRA hero
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter_by_tier(ax, pred["M1"][valid], lira, t, "cos θ_i^global  (M1)")
    x_line = np.linspace(pred["M1"][valid].min(), pred["M1"][valid].max(), 200)
    ax.plot(x_line, np.polyval(np.polyfit(pred["M1"][valid], lira, 1), x_line),
            "k--", alpha=0.7, lw=1.5)
    rho_v, p_v = spearmanr(pred["M1"][valid], lira)
    ax.set_title(
        f"Gradient Coherence vs Per-Instance Privacy (Non-Private Model)\n"
        f"Spearman ρ = {rho_v:+.3f},  p = {p_v:.2e}", fontsize=10)
    ax.set_xlabel("cos θ_i^global  (M1 — gradient coherence)", fontsize=10)
    ax.set_ylabel("LiRA score  (higher = more vulnerable)", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ana_dir, "fig_P3b_2.png"), dpi=150)
    plt.close(fig)

    # Fig P3b-3: LiRA boxplot by tier
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([lira[t == k] for k in range(K)],
                    labels=tier_labs, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    ax.set_ylabel("LiRA score", fontsize=10)
    ax.set_title("Per-instance vulnerability by tier (non-private)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(ana_dir, "fig_P3b_3.png"), dpi=150)
    plt.close(fig)

    # Fig P3b-4: Raw gradient norm histogram by tier (new vs Phase 2)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in range(K):
        mask_k = tiers_target == k
        v_k    = pred["grad_norm_raw"][mask_k]
        ax.hist(v_k, bins=30, alpha=0.5, color=colors[k], label=tier_labs[k],
                density=True)
    ax.set_xlabel("||g_i||  (raw gradient norm, no clipping)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Gradient Norm Distribution by Tier (Non-Private)\n"
                 "If norms stratify here → clipping was destroying tier signal in DP",
                 fontsize=9)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ana_dir, "fig_P3b_4.png"), dpi=150)
    plt.close(fig)

    print(f"[P3b] figures saved to {ana_dir}/")


# ---------------------------------------------------------------------------
# Main experiment orchestrator
# ---------------------------------------------------------------------------

def run_exp_p3b(
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
    tag        = f"{dataset_name}_IR{imbalance_ratio:.0f}_seed{seed}"
    out_dir    = os.path.join(results_dir, tag)
    shadow_dir = os.path.join(out_dir, "shadows")
    os.makedirs(shadow_dir, exist_ok=True)

    set_seed(seed)

    # ── Data + tiers ──────────────────────────────────────────────────────
    print(f"[P3b] loading data: {dataset_name} IR={imbalance_ratio}")
    data = load_datasets(
        dataset_name=dataset_name, data_root=data_root,
        imbalance_ratio=imbalance_ratio,
        public_frac=0.1, split_seed=42,
    )
    private_targets = np.array(data["private_dataset"].targets)
    tiers           = assign_tiers("A", private_targets, data["class_counts"], K=K)
    print(f"[P3b] tier sizes: {get_tier_sizes(tiers, K)}  (T0=head, T2=tail)")

    # ── Step 0: Targets ───────────────────────────────────────────────────
    target_indices, tiers_target = _select_targets(data, tiers, out_dir)

    if not analysis_only:
        # ── Step 1: Shadow models ─────────────────────────────────────────
        for m in range(shadow_start, shadow_end + 1):
            _train_shadow(m, data, target_indices, device, shadow_dir)

        # ── Step 1b: Reference model ──────────────────────────────────────
        ref_path  = os.path.join(out_dir, "reference_np.pt")
        ref_state = _get_or_train_reference(data, device, ref_path)

        # ── Step 2: Logits ────────────────────────────────────────────────
        logit_matrix, membership_matrix = _collect_logits(
            target_indices, shadow_dir, data, device
        )
    else:
        print("[P3b] --analysis_only: skipping training.")
        logit_path = os.path.join(shadow_dir, "logit_matrix_np.npy")
        mem_path   = os.path.join(shadow_dir, "membership_matrix_np.npy")
        if not os.path.exists(logit_path):
            raise FileNotFoundError(
                f"{logit_path} not found. Run without --analysis_only first."
            )
        logit_matrix      = np.load(logit_path)
        membership_matrix = np.load(mem_path)

    # ── Step 3: LiRA ──────────────────────────────────────────────────────
    lira_scores, aurocs = _compute_lira_scores(logit_matrix, membership_matrix, out_dir)

    # ── Step 4: Predictors ────────────────────────────────────────────────
    pred_path = os.path.join(out_dir, "predictors_np.npz")
    if not os.path.exists(pred_path):
        ref_path = os.path.join(out_dir, "reference_np.pt")
        if not os.path.exists(ref_path):
            ref_state = _get_or_train_reference(data, device, ref_path)
        else:
            ref_state = torch.load(ref_path, map_location="cpu")
        pred = _compute_predictors(ref_state, data, target_indices, device, pred_path)
    else:
        pred = dict(np.load(pred_path))

    # ── Step 5: Analysis ──────────────────────────────────────────────────
    _run_analysis(pred, lira_scores, aurocs, tiers_target, out_dir)

    results = {
        "tag":            tag,
        "dataset":        dataset_name,
        "imbalance_ratio": imbalance_ratio,
        "seed":           seed,
        "target_indices": target_indices,
        "tiers_target":   tiers_target,
        "lira_scores":    lira_scores,
        "aurocs":         aurocs,
        "predictors":     dict(pred),
    }
    save_results(results, os.path.join(out_dir, "results.pkl"))
    print(f"[P3b] complete — results in {out_dir}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3b: Coherence–Privacy via LiRA (non-private models)"
    )
    parser.add_argument("--dataset",       default="cifar10")
    parser.add_argument("--ir",            type=float, default=50.0)
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--gpu",           type=int,   default=0)
    parser.add_argument("--data_root",     default=DATA_ROOT)
    parser.add_argument("--results_dir",   default=RESULTS_DIR)
    parser.add_argument("--shadow_start",  type=int,   default=0)
    parser.add_argument("--shadow_end",    type=int,   default=N_SHADOWS - 1)
    parser.add_argument("--analysis_only", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    run_exp_p3b(
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
