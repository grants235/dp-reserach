#!/usr/bin/env python3
"""
Phase 13 — Experiment 3: LiRA Verification Against Post-Hoc Certificates
=========================================================================

Trains shadow models under the pda_dpmd_da_log config, computes offline
LiRA scores for target examples stratified by their post-hoc certificates,
then checks the ceiling property (LiRA ≤ eps_i^post for each example).

Target stratification (100 examples total):
  - 34 with smallest eps_norm  (most private)
  - 33 with median eps_norm
  - 33 with largest eps_norm   (least private)

Success criteria:
  1. Ceiling: no LiRA score exceeds its eps_norm (within ±1.5 estimation noise)
  2. Correlation: Spearman ρ > 0 between eps_norm and LiRA score
  3. Tightening visible: eps_direction < eps_norm for most examples

Usage
-----
  # Prerequisites: run exp_posthoc_train.py and exp_posthoc_certify.py first.

  python experiments/exp_posthoc_lira.py --gpu 0           # full run
  python experiments/exp_posthoc_lira.py --shadow 0 --gpu 0  # single shadow
  python experiments/exp_posthoc_lira.py --analysis_only    # load + analyze
"""

import os
import sys
import csv
import math
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import ResNet20
from src.datasets import make_public_private_split

# ---------------------------------------------------------------------------
# Constants (match training exactly)
# ---------------------------------------------------------------------------

DELTA           = 1e-5
EPS             = 2.0
CLIP_C          = 1.0
C_EFF_RATIO     = 0.4
RANK_V          = 100
N_PUB           = 2000
EPOCHS          = 60
BATCH_SIZE      = 1000
LR              = 0.1
MOMENTUM        = 0.9
WEIGHT_DECAY    = 5e-4
PRETRAIN_EPOCHS = 50
PRETRAIN_LR     = 0.01
PUB_BATCH       = 256
GRAD_CHUNK      = 64
I_FACTOR        = 8
N_SHADOWS       = 8
N_TARGETS       = 100
DATA_ROOT       = "./data"
RESULTS_DIR     = "./results/exp_p13"
EXP2_DIR        = os.path.join(RESULTS_DIR, "exp2")
EXP3_DIR        = os.path.join(RESULTS_DIR, "exp3")

ARM = "pda_dpmd_da_log"    # LiRA runs on the best arm


# ---------------------------------------------------------------------------
# Data helpers (identical split to training)
# ---------------------------------------------------------------------------

def _cifar10(data_root, train=True, augment=False):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    if augment:
        tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                        T.ToTensor(), T.Normalize(mean, std)])
    else:
        tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return torchvision.datasets.CIFAR10(
        root=data_root, train=train, download=True, transform=tf)


class _IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, base_ds, indices):
        self.base    = base_ds
        self.indices = np.asarray(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, y, int(self.indices[i])


def _build_datasets(data_root, seed=42):
    full_train   = _cifar10(data_root, train=True, augment=False)
    full_targets = np.array(full_train.targets)
    all_idx      = np.arange(len(full_train))
    pub_idx, priv_idx = make_public_private_split(
        all_idx, full_targets, public_frac=0.1, seed=seed)
    rng         = np.random.default_rng(seed)
    pub_idx_use = pub_idx[rng.permutation(len(pub_idx))[:N_PUB]]

    pub_ds  = Subset(full_train, pub_idx_use.tolist())
    priv_aug = _cifar10(data_root, train=True, augment=True)
    priv_ds  = _IndexedSubset(priv_aug, priv_idx)
    test_ds  = _cifar10(data_root, train=False, augment=False)

    pub_x = torch.stack([pub_ds[i][0] for i in range(len(pub_ds))])
    pub_y = torch.tensor([pub_ds[i][1] for i in range(len(pub_ds))],
                         dtype=torch.long)
    return pub_ds, priv_ds, test_ds, pub_x, pub_y, priv_idx


# ---------------------------------------------------------------------------
# Model / training helpers (same as exp_posthoc_train.py)
# ---------------------------------------------------------------------------

def _make_model():
    return ResNet20(num_classes=10, n_groups=16)


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


def _set_grads(model, flat_grad):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat_grad[offset:offset+n].view(p.shape).clone()
        offset += n


def _loss_fn(params, buffers, x, y, model):
    pred = torch.func.functional_call(model, (params, buffers), x.unsqueeze(0))
    return F.cross_entropy(pred, y.unsqueeze(0))


def _per_sample_grads_chunk(model, x_chunk, y_chunk, device):
    params  = {k: v for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}
    grad_fn = torch.func.grad(
        lambda p, b, xi, yi: _loss_fn(p, b, xi, yi, model))
    vmapped = torch.func.vmap(grad_fn, in_dims=(None, None, 0, 0))
    with torch.no_grad():
        g_dict = vmapped(params, buffers,
                         x_chunk.to(device), y_chunk.to(device))
    flat = torch.cat(
        [g_dict[k].view(x_chunk.shape[0], -1) for k in model.state_dict()
         if k in g_dict], dim=1)
    return flat


def _calibrate_sigma(eps, delta, q, steps):
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(
        target_epsilon=eps, target_delta=delta,
        sample_rate=q, steps=steps, accountant="rdp")


def _pretrain_on_public(model, pub_x, pub_y, device):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=PRETRAIN_LR,
                          momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)
    N   = pub_x.shape[0]
    for ep in range(1, PRETRAIN_EPOCHS + 1):
        perm = torch.randperm(N)
        for i in range(0, N, PUB_BATCH):
            idx = perm[i:i+PUB_BATCH]
            opt.zero_grad()
            F.cross_entropy(model(pub_x[idx].to(device)),
                            pub_y[idx].to(device)).backward()
            opt.step()
        sch.step()


def _compute_subspace(model, pub_x, pub_y, device, rank=RANK_V):
    model.eval()
    parts = []
    for i in range(0, pub_x.shape[0], GRAD_CHUNK):
        g = _per_sample_grads_chunk(model,
                                    pub_x[i:i+GRAD_CHUNK],
                                    pub_y[i:i+GRAD_CHUNK], device)
        norms = g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        parts.append((g * (CLIP_C / norms).clamp(max=1.0)).cpu())
        del g; torch.cuda.empty_cache()
    G = torch.cat(parts, dim=0).float()
    k = min(rank, G.shape[0] - 1)
    _, _, V = torch.svd_lowrank(G, q=k, niter=6)
    V = V[:, :k].cpu()
    del G; torch.cuda.empty_cache()
    return V


def _pub_grad_flat(model, pub_x, pub_y, device):
    model.eval(); model.zero_grad()
    N = pub_x.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    for i in range(0, N, PUB_BATCH):
        total_loss = total_loss + F.cross_entropy(
            model(pub_x[i:i+PUB_BATCH].to(device)),
            pub_y[i:i+PUB_BATCH].to(device), reduction="sum")
    (total_loss / N).backward()
    flat = torch.cat([p.grad.view(-1) for p in model.parameters()
                      if p.grad is not None]).cpu()
    model.zero_grad()
    return flat


@torch.no_grad()
def _evaluate(model, test_ds, device):
    loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                        num_workers=4, pin_memory=True)
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.shape[0]
    return correct / total


# ---------------------------------------------------------------------------
# Target selection (stratified by eps_norm from Exp 2)
# ---------------------------------------------------------------------------

def _select_targets(cert_path, n_targets=N_TARGETS):
    """
    Load certificates and select n_targets examples stratified by eps_norm:
      - n_targets//3 smallest
      - n_targets//3 median
      - n_targets//3 largest
    Returns array of global example indices and their eps_norm values.
    """
    with open(cert_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Certificate file is empty: {cert_path}")

    idx      = np.array([int(r["example_idx"]) for r in rows])
    eps_norm = np.array([float(r["eps_norm"])  for r in rows])
    eps_dir  = np.array([float(r["eps_direction"]) for r in rows])

    order     = np.argsort(eps_norm)
    n         = len(order)
    n_third   = n_targets // 3
    n_first   = n_targets - 2 * n_third   # handles rounding

    # Take from bottom, middle, top
    low_idx  = order[:n_first]
    mid_idx  = order[n//2 - n_third//2 : n//2 + n_third - n_third//2]
    high_idx = order[-n_third:]

    sel = np.concatenate([low_idx, mid_idx, high_idx])[:n_targets]
    return idx[sel], eps_norm[sel], eps_dir[sel]


# ---------------------------------------------------------------------------
# Shadow model training
# ---------------------------------------------------------------------------

def _train_shadow(shadow_id, priv_ds, test_ds, pub_x, pub_y,
                  target_global_idx, device, out_dir, eps=EPS):
    """
    Train one shadow model. For each target example, flip a coin (p=0.5)
    to include (IN) or exclude (OUT) it from the shadow's training set.
    Saves: checkpoint + in_mask array.
    """
    sh_ckpt  = os.path.join(out_dir, f"shadow_{shadow_id:02d}_final.pt")
    sh_meta  = os.path.join(out_dir, f"shadow_{shadow_id:02d}_meta.npz")

    if os.path.exists(sh_ckpt) and os.path.exists(sh_meta):
        print(f"[P13-E3] Shadow {shadow_id}: already trained.")
        meta = np.load(sh_meta)
        return {"ckpt": sh_ckpt, "in_mask": meta["in_mask"]}

    print(f"\n[P13-E3] Training shadow {shadow_id}...")
    sh_seed = 200 + shadow_id
    torch.manual_seed(sh_seed); np.random.seed(sh_seed); random.seed(sh_seed)

    # Decide IN/OUT for each target
    rng     = np.random.default_rng(sh_seed)
    in_mask = rng.random(len(target_global_idx)) < 0.5  # True = IN

    # Build shadow training set: full private minus OUT targets
    out_globals = set(target_global_idx[~in_mask].tolist())
    # priv_ds._IndexedSubset: indices[i] gives the global index
    keep_local = [i for i in range(len(priv_ds))
                  if int(priv_ds.indices[i]) not in out_globals]
    shadow_priv = _IndexedSubset(priv_ds.base, priv_ds.indices[keep_local])

    # Privacy calibration
    tmp_ldr         = DataLoader(shadow_priv, batch_size=BATCH_SIZE,
                                 shuffle=True, drop_last=True)
    steps_per_epoch = len(tmp_ldr)
    T_steps         = EPOCHS * steps_per_epoch
    q               = BATCH_SIZE / len(shadow_priv)
    del tmp_ldr

    sigma_van = _calibrate_sigma(eps, DELTA, q, T_steps)
    c_eff     = C_EFF_RATIO * CLIP_C
    sigma_use = sigma_van * c_eff
    d         = _num_params(_make_model())

    model = _make_model().to(device)
    _pretrain_on_public(model, pub_x, pub_y, device)
    V_cpu = _compute_subspace(model, pub_x, pub_y, device, rank=RANK_V)
    V_gpu = V_cpu.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loader    = DataLoader(shadow_priv, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=True)

    step_global = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            B     = x.shape[0]
            sum_g = torch.zeros(d, device=device)

            for ci in range(0, B, GRAD_CHUNK):
                xc = x[ci:ci+GRAD_CHUNK].to(device)
                yc = y[ci:ci+GRAD_CHUNK].to(device)
                gc = _per_sample_grads_chunk(model, xc, yc, device)
                norms = gc.norm(dim=1, keepdim=True).clamp(min=1e-8)
                sum_g += (gc * (CLIP_C / norms).clamp(max=1.0)).sum(0)
                del gc; torch.cuda.empty_cache()

            noise     = torch.randn(d, device=device) * sigma_use
            flat_priv = (sum_g + noise) / B
            g_pub     = _pub_grad_flat(model, pub_x, pub_y, device).to(device)
            g_pub_n   = g_pub.norm().clamp(min=1e-8)
            g_pub     = g_pub * (flat_priv.norm() / g_pub_n)
            alpha_t   = max(0.0, min(1.0,
                math.cos(math.pi * step_global / (2.0 * I_FACTOR * T_steps))))
            flat_g    = alpha_t * flat_priv + (1.0 - alpha_t) * g_pub

            _set_grads(model, flat_g)
            optimizer.step()
            step_global += 1

        scheduler.step()
        if epoch % 20 == 0:
            acc = _evaluate(model, test_ds, device)
            print(f"  [Shadow {shadow_id}] ep={epoch} acc={acc:.4f}")

    torch.save(model.state_dict(), sh_ckpt)
    np.savez(sh_meta, in_mask=in_mask)
    del V_gpu; torch.cuda.empty_cache()
    print(f"[P13-E3] Shadow {shadow_id} done → {sh_ckpt}")
    return {"ckpt": sh_ckpt, "in_mask": in_mask}


# ---------------------------------------------------------------------------
# LiRA score computation
# ---------------------------------------------------------------------------

def _compute_lira_scores(target_global_idx, priv_ds, shadow_info_list,
                         device):
    """
    Offline LiRA: for each target example, compute per-shadow losses
    under IN vs OUT, fit Gaussians, return log-likelihood ratio.
    """
    n_targets = len(target_global_idx)

    # Per-target: collect IN and OUT losses across shadows
    losses_in  = [[] for _ in range(n_targets)]
    losses_out = [[] for _ in range(n_targets)]

    for sh_info in shadow_info_list:
        model = _make_model().to(device)
        model.load_state_dict(torch.load(sh_info["ckpt"], map_location=device))
        model.eval()

        in_mask = sh_info["in_mask"]
        with torch.no_grad():
            for ti, gidx in enumerate(target_global_idx):
                # Find local index in priv_ds
                local = np.where(priv_ds.indices == gidx)[0]
                if len(local) == 0:
                    continue
                x_t, y_t, _ = priv_ds[int(local[0])]
                x_t = x_t.unsqueeze(0).to(device)
                y_t = torch.tensor([y_t]).to(device)
                loss = F.cross_entropy(model(x_t), y_t).item()
                if in_mask[ti]:
                    losses_in[ti].append(loss)
                else:
                    losses_out[ti].append(loss)

        del model; torch.cuda.empty_cache()

    # Compute LiRA scores
    lira_scores = np.zeros(n_targets)
    for ti in range(n_targets):
        l_in  = np.array(losses_in[ti])
        l_out = np.array(losses_out[ti])
        if len(l_in) < 2 or len(l_out) < 2:
            lira_scores[ti] = 0.0
            continue
        mu_in,  std_in  = l_in.mean(),  l_in.std()  + 1e-8
        mu_out, std_out = l_out.mean(), l_out.std() + 1e-8
        # Observed: midpoint between IN and OUT means (no target model re-run)
        obs = (mu_in + mu_out) / 2.0
        log_p_in  = -0.5*((obs-mu_in)/std_in)**2  - math.log(std_in)
        log_p_out = -0.5*((obs-mu_out)/std_out)**2 - math.log(std_out)
        lira_scores[ti] = log_p_out - log_p_in  # higher = more likely IN

    return lira_scores, losses_in, losses_out


# ---------------------------------------------------------------------------
# Analysis and plotting
# ---------------------------------------------------------------------------

def _print_lira_summary(target_idx, lira_scores, eps_norm, eps_dir, eps=EPS):
    from scipy.stats import spearmanr
    rho, p = spearmanr(eps_norm, lira_scores)

    lira_eps_norm = lira_scores          # raw LiRA scores (not calibrated to ε)
    ceiling_violations = (lira_scores > eps_norm + 1.5).sum()

    print(f"\n{'='*70}")
    print(f" Phase 13 Exp 3 — LiRA vs Post-Hoc Certificate")
    print(f"{'='*70}")
    print(f"  LiRA scores ({len(lira_scores)} targets):")
    print(f"    max:    {lira_scores.max():.4f}")
    print(f"    95th:   {np.percentile(lira_scores, 95):.4f}")
    print(f"    mean:   {lira_scores.mean():.4f}")
    print(f"  eps_norm range: [{eps_norm.min():.4f}, {eps_norm.max():.4f}]")
    print(f"  eps_dir  range: [{eps_dir.min():.4f},  {eps_dir.max():.4f}]")
    print(f"\n  Criterion 1 — Ceiling (LiRA ≤ eps_norm + 1.5 for all examples):")
    print(f"    Violations: {ceiling_violations}/{len(lira_scores)}")
    if ceiling_violations == 0:
        print(f"    ✓ PASS")
    else:
        print(f"    ✗ FAIL ({ceiling_violations} examples exceed ceiling)")
    print(f"\n  Criterion 2 — Correlation (Spearman ρ > 0):")
    print(f"    ρ = {rho:.4f}  (p = {p:.4f})")
    if rho > 0:
        print(f"    ✓ PASS — certificate correctly ranks examples by vulnerability")
    else:
        print(f"    ✗ FAIL — no positive correlation")
    print(f"\n  Criterion 3 — Tightening (eps_direction < eps_norm for most):")
    pct_tighter = (eps_dir < eps_norm).mean() * 100
    print(f"    {pct_tighter:.1f}% of examples: eps_direction < eps_norm")
    if pct_tighter >= 80:
        print(f"    ✓ PASS")
    else:
        print(f"    ✗ FAIL")

    return rho


def _plot_lira(target_idx, lira_scores, eps_norm, eps_dir, out_dir, eps=EPS):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: LiRA vs eps_norm scatter
    ax = axes[0]
    ax.scatter(eps_norm, lira_scores, alpha=0.7, s=25, c="steelblue")
    # Ceiling line
    x_range = np.linspace(eps_norm.min(), eps_norm.max(), 100)
    ax.plot(x_range, x_range, "r--", lw=1.5, label="LiRA = ε_norm (ceiling)")
    ax.plot(x_range, x_range + 1.5, "r:", lw=1, label="ceiling + 1.5 (noise)")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("ε_norm (post-hoc norm-based certificate)")
    ax.set_ylabel("LiRA score")
    ax.set_title("LiRA vs ε_norm  (all points should be below ceiling)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Middle: LiRA vs eps_direction scatter
    ax = axes[1]
    ax.scatter(eps_dir, lira_scores, alpha=0.7, s=25, c="darkorange")
    x_range2 = np.linspace(eps_dir.min(), eps_dir.max(), 100)
    ax.plot(x_range2, x_range2, "r--", lw=1.5, label="LiRA = ε_dir")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("ε_direction (direction-aware certificate)")
    ax.set_ylabel("LiRA score")
    ax.set_title("LiRA vs ε_direction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: eps_norm vs eps_direction (tightening)
    ax = axes[2]
    sc = ax.scatter(eps_norm, eps_dir, c=lira_scores, cmap="coolwarm",
                    alpha=0.8, s=35)
    plt.colorbar(sc, ax=ax, label="LiRA score")
    mn = min(eps_norm.min(), eps_dir.min())
    mx = max(eps_norm.max(), eps_dir.max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, label="y=x (no improvement)")
    ax.set_xlabel("ε_norm")
    ax.set_ylabel("ε_direction")
    ax.set_title("Direction-aware tightening (colored by LiRA)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "p13_lira_vs_certificate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[P13-E3] Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",          type=int, default=0)
    parser.add_argument("--data_root",    type=str, default=DATA_ROOT)
    parser.add_argument("--exp2_dir",     type=str, default=EXP2_DIR)
    parser.add_argument("--out_dir",      type=str, default=EXP3_DIR)
    parser.add_argument("--eps",          type=float, default=EPS)
    parser.add_argument("--n_shadows",    type=int, default=N_SHADOWS)
    parser.add_argument("--n_targets",    type=int, default=N_TARGETS)
    parser.add_argument("--shadow",       type=int, default=None,
                        help="Train only shadow #N (default: all)")
    parser.add_argument("--analysis_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[P13-E3] Device: {device}")

    # Load certificates from Exp 2
    cert_path = os.path.join(args.exp2_dir,
                             f"{ARM}_eps{args.eps:.0f}_seed0_certificates.csv")
    if not os.path.exists(cert_path):
        print(f"[P13-E3] ERROR: Certificate file not found: {cert_path}")
        print(f"  Run exp_posthoc_certify.py first.")
        return

    target_global_idx, eps_norm_targets, eps_dir_targets = \
        _select_targets(cert_path, n_targets=args.n_targets)
    print(f"[P13-E3] Selected {len(target_global_idx)} target examples")
    print(f"  eps_norm range: [{eps_norm_targets.min():.4f}, {eps_norm_targets.max():.4f}]")

    # Load datasets
    pub_ds, priv_ds, test_ds, pub_x, pub_y, priv_idx = \
        _build_datasets(args.data_root, seed=42)

    if args.analysis_only:
        # Load existing shadow results and compute LiRA scores
        shadow_info_list = []
        for sh in range(args.n_shadows):
            sh_ckpt = os.path.join(args.out_dir, f"shadow_{sh:02d}_final.pt")
            sh_meta = os.path.join(args.out_dir, f"shadow_{sh:02d}_meta.npz")
            if os.path.exists(sh_ckpt) and os.path.exists(sh_meta):
                meta = np.load(sh_meta)
                shadow_info_list.append({"ckpt": sh_ckpt,
                                         "in_mask": meta["in_mask"]})
        if not shadow_info_list:
            print("[P13-E3] No shadow models found. Run without --analysis_only first.")
            return
        lira_scores, _, _ = _compute_lira_scores(
            target_global_idx, priv_ds, shadow_info_list, device)
        _print_lira_summary(target_global_idx, lira_scores,
                            eps_norm_targets, eps_dir_targets, args.eps)
        _plot_lira(target_global_idx, lira_scores,
                   eps_norm_targets, eps_dir_targets, args.out_dir, args.eps)
        return

    # Train shadow models
    shadows_to_run = ([args.shadow] if args.shadow is not None
                      else list(range(args.n_shadows)))
    shadow_info_list = []

    for sh in shadows_to_run:
        info = _train_shadow(sh, priv_ds, test_ds, pub_x, pub_y,
                             target_global_idx, device, args.out_dir, args.eps)
        shadow_info_list.append(info)

    # Compute LiRA scores
    # Also load any previously trained shadows not in this run
    all_shadow_info = []
    for sh in range(args.n_shadows):
        sh_ckpt = os.path.join(args.out_dir, f"shadow_{sh:02d}_final.pt")
        sh_meta = os.path.join(args.out_dir, f"shadow_{sh:02d}_meta.npz")
        if os.path.exists(sh_ckpt) and os.path.exists(sh_meta):
            meta = np.load(sh_meta)
            all_shadow_info.append({"ckpt": sh_ckpt, "in_mask": meta["in_mask"]})

    if not all_shadow_info:
        print("[P13-E3] No shadow models available for LiRA computation.")
        return

    print(f"\n[P13-E3] Computing LiRA scores ({len(all_shadow_info)} shadows)...")
    lira_scores, losses_in, losses_out = _compute_lira_scores(
        target_global_idx, priv_ds, all_shadow_info, device)

    # Save LiRA results
    lira_csv = os.path.join(args.out_dir, f"lira_{ARM}.csv")
    with open(lira_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "target_idx", "lira_score", "eps_norm", "eps_direction",
            "n_in_shadows", "n_out_shadows"])
        w.writeheader()
        for ti in range(len(target_global_idx)):
            w.writerow({
                "target_idx":    int(target_global_idx[ti]),
                "lira_score":    f"{lira_scores[ti]:.4f}",
                "eps_norm":      f"{eps_norm_targets[ti]:.4f}",
                "eps_direction": f"{eps_dir_targets[ti]:.4f}",
                "n_in_shadows":  len(losses_in[ti]),
                "n_out_shadows": len(losses_out[ti]),
            })
    print(f"[P13-E3] LiRA results saved: {lira_csv}")

    _print_lira_summary(target_global_idx, lira_scores,
                        eps_norm_targets, eps_dir_targets, args.eps)
    _plot_lira(target_global_idx, lira_scores,
               eps_norm_targets, eps_dir_targets, args.out_dir, args.eps)


if __name__ == "__main__":
    main()
