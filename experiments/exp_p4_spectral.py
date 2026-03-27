#!/usr/bin/env python3
"""
Phase 4: Gradient Low-Rank Structure and Projected DP-SGD Feasibility.

Part A (~3h):  Gradient spectral analysis — does useful signal concentrate
               in a low-dimensional subspace?
Part B (~0.5h): Non-private projected training — does the subspace preserve
               learning quality?
Part C (~12h): Projected DP-SGD — does the noise reduction translate to
               accuracy improvement at matched (ε, δ)?

Run:
  # Full experiment (auto-skips Part C if Part B fails threshold)
  venv/bin/python experiments/exp_p4_spectral.py --gpu 0

  # Stop after Part A (spectral analysis only)
  venv/bin/python experiments/exp_p4_spectral.py --gpu 0 --parts A

  # Parts A + B
  venv/bin/python experiments/exp_p4_spectral.py --gpu 0 --parts AB

  # Skip to Part C (if Parts A+B already done)
  venv/bin/python experiments/exp_p4_spectral.py --gpu 0 --parts C
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
from src.models import make_model, validate_model_for_dp
from src.dp_training import _clear_grad_samples, set_seed, evaluate
from src.tiers import assign_tiers, get_tier_sizes
from src.privacy_accounting import compute_sigma
from src.evaluation import save_results

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K               = 3
N_GRAD_SAMPLES  = 2000
TIER_GRAD       = {0: 1000, 1: 600, 2: 400}   # stratified sample for Part A
SVD_COMPONENTS  = 1000
RANK_R_PART_B   = [50, 100, 500, 1000]
RANK_R_PART_C   = [100, 500]
EPOCHS_NP       = 200    # Parts A ref-model + Part B
EPOCHS_DP       = 100    # Part C
BATCH_SIZE_NP   = 128    # Part B
BATCH_SIZE_DP   = 256    # Part C
AUG_MULT_DP     = 8      # Part C (matches Phase-0 best config)
C_TRAIN         = 1.0
EPS_TARGET      = 3.0
DELTA           = 1e-5
GRAD_BATCH      = 16     # vmap batch size for Part A gradient extraction
DP_CHUNK        = 32     # chunk size for Part C Opacus step
GRAD_SEED       = 42     # for stratified sample selection
LR_NP           = 0.1
WD              = 5e-4
DATA_ROOT       = "./data"
RESULTS_DIR     = "./results/exp_p4"
P3B_DIR         = "./results/exp_p3b"  # reference_np.pt lives here


# ---------------------------------------------------------------------------
# WideResNet-28-2 with BatchNorm (for non-private Parts A and B)
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
            if stride != 1 or in_planes != out_planes else nn.Identity()
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.shortcut(x)


class WRN28_2_BN(nn.Module):
    """WideResNet-28-2 with BatchNorm2d. For non-private training."""
    def __init__(self, num_classes=10):
        super().__init__()
        c = [16, 32, 64, 128]
        n = 4
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
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = F.relu(self.bn(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def _make_np_model(num_classes):
    return WRN28_2_BN(num_classes=num_classes)


# ---------------------------------------------------------------------------
# Augmentation + DP helpers (Parts B / C)
# ---------------------------------------------------------------------------

def _augment_batch(x):
    B, C_, H, W = x.shape
    xp    = F.pad(x, (4, 4, 4, 4), mode="reflect")
    oi    = torch.randint(0, 8, (B,), device=x.device)
    oj    = torch.randint(0, 8, (B,), device=x.device)
    rows  = oi[:, None] + torch.arange(H, device=x.device)
    cols  = oj[:, None] + torch.arange(W, device=x.device)
    b_idx = torch.arange(B, device=x.device)[:, None, None]
    crops = (
        xp.permute(0, 2, 3, 1)[b_idx, rows[:, :, None], cols[:, None, :]]
        .permute(0, 3, 1, 2).contiguous()
    )
    flip = torch.rand(B, device=x.device) > 0.5
    crops[flip] = crops[flip].flip(-1)
    return crops


def _project_gradients_inplace(model, V_r):
    """
    Project all parameter gradients onto subspace V_r in-place.
    V_r: (d_total, r) on the same device as model.
    g_proj = V_r @ (V_r^T @ g)
    """
    g = torch.cat([p.grad.flatten() for p in model.parameters()])
    g_proj = V_r @ (V_r.t() @ g)   # (d_total,)
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.grad.copy_(g_proj[offset:offset + numel].reshape(p.shape))
        offset += numel


# ---------------------------------------------------------------------------
# Part A — Step 1: Get / train reference model
# ---------------------------------------------------------------------------

def _get_reference_model(data, device, ref_path):
    """Return state dict of non-private reference model (CPU)."""
    if os.path.exists(ref_path):
        print(f"[P4] reference model loaded from {ref_path}")
        return torch.load(ref_path, map_location="cpu")

    # Check Phase-3b reference
    p3b_ref = os.path.join(P3B_DIR, "cifar10_IR50_seed0", "reference_np.pt")
    if os.path.exists(p3b_ref):
        import shutil; shutil.copy(p3b_ref, ref_path)
        print(f"[P4] reference: copied from Phase-3b ({p3b_ref})")
        return torch.load(ref_path, map_location="cpu")

    # Train fresh
    print("[P4] training reference model (non-private, 200 epochs)...")
    use_pin = device.type == "cuda"
    loader = DataLoader(
        data["private_dataset"], batch_size=BATCH_SIZE_NP, shuffle=True,
        num_workers=4, pin_memory=use_pin, drop_last=True,
    )
    set_seed(1234)
    model = _make_np_model(data["num_classes"]).to(device)
    opt   = torch.optim.SGD(model.parameters(), lr=LR_NP, momentum=0.9, weight_decay=WD)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_NP)
    _, _, test_loader = make_data_loaders(data, batch_size=256)
    for epoch in range(1, EPOCHS_NP + 1):
        model.train()
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        sch.step()
        if epoch % 50 == 0 or epoch == EPOCHS_NP:
            acc = evaluate(model, test_loader, device)
            print(f"  reference epoch {epoch}/{EPOCHS_NP}: acc={acc:.4f}")
    state = model.state_dict()
    torch.save(state, ref_path)
    del model, loader; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return state


# ---------------------------------------------------------------------------
# Part A — Step 2: Compute gradient matrix
# ---------------------------------------------------------------------------

def _select_grad_samples(data, tiers, out_dir):
    """Select 2000 stratified examples for gradient computation."""
    path = os.path.join(out_dir, "grad_sample_indices.npy")
    if os.path.exists(path):
        idx = np.load(path)
        return idx, tiers[idx]
    rng = np.random.default_rng(GRAD_SEED)
    selected = []
    for tier_k, n_want in TIER_GRAD.items():
        pool = np.where(tiers == tier_k)[0]
        selected.extend(rng.choice(pool, size=n_want, replace=False).tolist())
    idx = np.array(selected, dtype=np.int64)
    np.save(path, idx)
    print(f"[P4] grad samples: {len(idx)} selected "
          f"({[(tiers[idx]==k).sum() for k in range(K)]})")
    return idx, tiers[idx]


def _compute_gradient_matrix(ref_state, data, indices, device, out_dir):
    """
    Compute full d-dim gradient for each of the 2000 selected examples.
    Returns G: (N, d_total) float32 numpy array (held in RAM, ~12 GB fp32).
    Saves as float16 to disk if not already done.
    """
    from torch.func import vmap, grad, functional_call

    mat_path = os.path.join(out_dir, "grad_matrix_f16.npy")
    if os.path.exists(mat_path):
        print(f"[P4] gradient matrix loaded from {mat_path}")
        return np.load(mat_path).astype(np.float32)

    print(f"[P4] computing gradient matrix for {len(indices)} examples...")
    model = _make_np_model(data["num_classes"])
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

    subset = Subset(data["private_dataset_noaug"], indices.tolist())
    use_pin = device.type == "cuda"
    loader  = DataLoader(subset, batch_size=GRAD_BATCH, shuffle=False,
                         num_workers=2, pin_memory=use_pin, drop_last=False)

    G = np.zeros((len(indices), d_total), dtype=np.float16)
    offset = 0
    for bi, batch in enumerate(loader):
        x, y = batch[0].to(device), batch[1].to(device)
        B = x.shape[0]
        per_grads = per_sample_grad_fn(params, x, y)
        flat = torch.cat(
            [per_grads[n].detach().reshape(B, -1) for n in param_names], dim=1
        )  # (B, d_total) on device
        G[offset:offset + B] = flat.cpu().half().numpy()
        offset += B
        del flat, per_grads
        if device.type == "cuda": torch.cuda.empty_cache()
        if (bi + 1) % 25 == 0:
            print(f"  ... {offset}/{len(indices)} examples")

    del model; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()

    np.save(mat_path, G)
    print(f"[P4] gradient matrix saved ({G.nbytes / 1e9:.1f} GB fp16)")
    return G.astype(np.float32)


# ---------------------------------------------------------------------------
# Part A — Step 3: SVD
# ---------------------------------------------------------------------------

def _run_svd(G_fp32, device, out_dir):
    """
    Truncated SVD of the centered gradient matrix via the Gram matrix trick.

    Since N=2000 << d=1.47M we compute M = G_c G_c^T (2000x2000, ~16 MB) on
    CPU, eigendecompose it, then recover the right singular vectors:
        Vt = U_top^T @ G_c / S[:, None]
    This never allocates a large matrix on GPU, keeping peak CPU RAM ~20 GB.

    Returns S (SVD_COMPONENTS,), Vt (SVD_COMPONENTS, d) as float32 numpy arrays.
    """
    S_path  = os.path.join(out_dir, "svd_S.npy")
    Vt_path = os.path.join(out_dir, "svd_Vt_f16.npy")
    if os.path.exists(S_path) and os.path.exists(Vt_path):
        print("[P4] SVD results loaded from disk.")
        return np.load(S_path), np.load(Vt_path).astype(np.float32)

    print(f"[P4] running SVD on ({G_fp32.shape[0]}, {G_fp32.shape[1]}) matrix (CPU, Gram trick)...")
    N, d = G_fp32.shape

    # --- Step 1: centered Gram matrix without forming G_c explicitly ---
    # G_c G_c^T = G G^T - v 1^T - 1 v^T + mu·mu  (scalar)
    # where v = G @ mu, mu = col-mean of G
    print("[P4]   computing Gram matrix...")
    g_mean = G_fp32.mean(axis=0)                  # (d,)  ~6 MB
    M      = G_fp32 @ G_fp32.T                    # (N, N) = (2000, 2000)  16 MB
    v      = G_fp32 @ g_mean                      # (N,)
    M     -= v[:, None] + v[None, :]
    M     += float(np.dot(g_mean, g_mean))        # scalar correction
    # M is now G_c G_c^T (symmetric PSD)

    # --- Step 2: eigendecompose small (N, N) matrix ---
    print("[P4]   eigendecomposition of Gram matrix...")
    eigenvalues, U = np.linalg.eigh(M.astype(np.float64))  # ascending order
    del M; gc.collect()

    k   = min(SVD_COMPONENTS, N)
    idx = np.argsort(eigenvalues)[::-1][:k]       # sort descending
    eigenvalues_top = eigenvalues[idx].clip(0)
    U_top = U[:, idx].astype(np.float32)          # (N, k)
    S     = np.sqrt(eigenvalues_top).astype(np.float32)  # (k,)
    del eigenvalues, U; gc.collect()
    print(f"[P4]   top-5 singular values: {S[:5]}")

    # --- Step 3: right singular vectors Vt = U_top^T @ G_c / S ---
    # G_c = G - g_mean, so:
    # U_top^T @ G_c = U_top^T @ G - (U_top.sum(axis=0))[:, None] * g_mean[None, :]
    print("[P4]   computing Vt = U^T @ G_c  (~6 GB)...")
    Vt = U_top.T @ G_fp32                         # (k, d)  ~5.88 GB
    col_sums = U_top.sum(axis=0)                  # (k,)
    Vt -= col_sums[:, None] * g_mean[None, :]     # centering correction
    del g_mean, col_sums, U_top; gc.collect()

    S_safe = np.where(S > 1e-12, S, 1.0)
    Vt    /= S_safe[:, None]

    Vt16 = Vt.astype(np.float16)
    np.save(S_path, S)
    np.save(Vt_path, Vt16)
    print(f"[P4] SVD saved. Top-5 singular values: {S[:5]}")
    return S, Vt.astype(np.float32)


# ---------------------------------------------------------------------------
# Part A — Step 4: Analysis (capture fractions, SNR)
# ---------------------------------------------------------------------------

def _analyze_part_a(G_fp32, tiers_idx, S, Vt, out_dir):
    """
    Compute and print Tab S-1 (variance, aggregate capture, SNR).
    Returns dict of results for plotting.
    """
    N, d       = G_fp32.shape
    G_agg      = G_fp32.mean(axis=0)                        # (d,) aggregate gradient
    G_agg_norm = np.linalg.norm(G_agg)
    total_var  = (S ** 2).sum()

    r_values   = [1, 5, 10, 25, 50, 100, 200, 500, 1000]
    r_values   = [r for r in r_values if r <= len(S)]

    lines = ["=== Tab S-1: Gradient Subspace Analysis ===",
             f"{'r':>6} {'var_frac':>10} {'agg_capture':>13} {'SNR_ratio':>11}"]
    lines.append("-" * 45)

    tab_data = {"r": [], "var_frac": [], "agg_capture": [], "snr_ratio": []}
    for r in r_values:
        V_r     = Vt[:r].T                                  # (d, r)
        coeffs  = G_agg @ V_r                               # (r,) projection of agg
        proj    = V_r @ coeffs                              # (d,) projected agg
        cap     = np.linalg.norm(proj) / (G_agg_norm + 1e-12)
        var_frac = (S[:r] ** 2).sum() / (total_var + 1e-12)
        snr_ratio = (cap ** 2) / max(r / d, 1e-12)
        lines.append(f"  {r:>4}   {var_frac:>9.4f}   {cap:>11.4f}   {snr_ratio:>10.1f}x")
        tab_data["r"].append(r)
        tab_data["var_frac"].append(float(var_frac))
        tab_data["agg_capture"].append(float(cap))
        tab_data["snr_ratio"].append(float(snr_ratio))

    lines.append("")

    # Per-tier aggregate capture at selected r values
    r_tier = [10, 50, 100, 500]
    r_tier = [r for r in r_tier if r <= len(S)]
    lines.append("=== Per-tier aggregate gradient capture ===")
    lines.append(f"{'Tier':<10}" + "".join(f"  r={r:>4}" for r in r_tier))
    lines.append("-" * (10 + len(r_tier) * 10))

    tier_names   = ["T0 (head)", "T1 (mid)", "T2 (tail)"]
    tier_capture = {r: [] for r in r_tier}
    for k in range(K):
        mask    = tiers_idx == k
        G_t     = G_fp32[mask].mean(axis=0)
        G_t_n   = np.linalg.norm(G_t)
        row = f"  {tier_names[k]:<8}"
        for r in r_tier:
            V_r = Vt[:r].T
            proj_t = V_r @ (G_t @ V_r)
            cap_t  = np.linalg.norm(proj_t) / (G_t_n + 1e-12)
            row += f"  {cap_t:>7.4f}"
            tier_capture[r].append(float(cap_t))
        lines.append(row)
    lines.append("")

    # Go/no-go gate
    cap_at_100 = tab_data["agg_capture"][tab_data["r"].index(100)] if 100 in tab_data["r"] else 0.0
    go_nogo = "GO" if cap_at_100 >= 0.50 else "NO-GO"
    lines.append(f"Go/no-go (r=100 capture ≥ 0.50): {cap_at_100:.4f}  [{go_nogo}]")
    lines.append("")

    tab_text = "\n".join(lines)
    print(tab_text)
    with open(os.path.join(out_dir, "tab_S1.txt"), "w") as f:
        f.write(tab_text)

    return {
        "tab_data":     tab_data,
        "tier_capture": tier_capture,
        "r_tier":       r_tier,
        "cap_at_100":   cap_at_100,
        "go_nogo":      go_nogo,
    }


def _save_part_a_figures(S, analysis, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[P4] matplotlib not available — skipping figures.")
        return

    fig_dir = os.path.join(out_dir, "figs_A")
    os.makedirs(fig_dir, exist_ok=True)
    td = analysis["tab_data"]

    # Fig S-1: Singular value spectrum (log-log)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(np.arange(1, len(S) + 1), S, lw=1.5)
    ax.set_xlabel("Component index $i$"); ax.set_ylabel("Singular value $S_i$")
    ax.set_title("Gradient singular value spectrum")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_S1_spectrum.png"), dpi=150)
    plt.close(fig)

    # Fig S-2: Cumulative variance captured
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(td["r"], td["var_frac"], "o-", lw=2)
    ax.axhline(0.9, color="r", linestyle="--", alpha=0.6, label="90%")
    ax.axhline(0.5, color="orange", linestyle="--", alpha=0.6, label="50%")
    ax.set_xlabel("Subspace dimension $r$"); ax.set_ylabel("Fraction of gradient variance")
    ax.set_title("Cumulative gradient variance captured by top-$r$ subspace")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_S2_variance.png"), dpi=150)
    plt.close(fig)

    # Fig S-3: Aggregate gradient capture
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(td["r"], td["agg_capture"], "o-", lw=2, color="tab:green")
    ax.axhline(0.9, color="r", linestyle="--", alpha=0.6, label="90%")
    ax.axhline(0.5, color="orange", linestyle="--", alpha=0.6, label="50% (go/no-go)")
    ax.set_xlabel("Subspace dimension $r$")
    ax.set_ylabel(r"$\|P_V \bar{G}\| / \|\bar{G}\|$  (aggregate capture)")
    ax.set_title("Fraction of aggregate gradient captured by top-$r$ subspace")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_S3_agg_capture.png"), dpi=150)
    plt.close(fig)

    # Fig S-4: Per-tier capture grouped bar
    r_tier  = analysis["r_tier"]
    tc      = analysis["tier_capture"]
    tier_names = ["T0 (head)", "T1 (mid)", "T2 (tail)"]
    colors  = ["tab:blue", "tab:orange", "tab:red"]
    x       = np.arange(len(r_tier))
    width   = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(K):
        vals = [tc[r][k] for r in r_tier]
        ax.bar(x + (k - 1) * width, vals, width, label=tier_names[k],
               color=colors[k], alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"r={r}" for r in r_tier])
    ax.set_ylabel("Aggregate gradient capture"); ax.set_ylim(0, 1.05)
    ax.set_title("Per-tier gradient capture by subspace"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_S4_tier_capture.png"), dpi=150)
    plt.close(fig)

    print(f"[P4] Part A figures saved to {fig_dir}/")


# ---------------------------------------------------------------------------
# Part B — Projected non-private training
# ---------------------------------------------------------------------------

def _train_with_projection(r, Vt, data, device, out_dir):
    """
    Train WRN28_2_BN for EPOCHS_NP with gradient projection onto top-r subspace.
    r=None means full-gradient baseline.
    Returns history dict {epoch: test_acc}.
    """
    import pickle
    tag  = "baseline" if r is None else f"proj_{r}"
    hpath = os.path.join(out_dir, f"part_b_{tag}_history.pkl")
    if os.path.exists(hpath):
        with open(hpath, "rb") as f: return pickle.load(f)
    set_seed(7777 + (0 if r is None else r))
    use_pin = device.type == "cuda"
    loader  = DataLoader(
        data["private_dataset"], batch_size=BATCH_SIZE_NP, shuffle=True,
        num_workers=4, pin_memory=use_pin, drop_last=True,
    )
    _, _, test_loader = make_data_loaders(data, batch_size=256)

    model = _make_np_model(data["num_classes"]).to(device)
    opt   = torch.optim.SGD(model.parameters(), lr=LR_NP, momentum=0.9, weight_decay=WD)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_NP)

    # Pre-load V_r onto device once
    V_r = None
    if r is not None:
        V_r = torch.from_numpy(Vt[:r].T.astype(np.float32)).to(device)  # (d, r)

    history = {}
    for epoch in range(1, EPOCHS_NP + 1):
        model.train()
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            if V_r is not None:
                _project_gradients_inplace(model, V_r)
            opt.step()
        sch.step()
        if epoch % 25 == 0 or epoch == EPOCHS_NP:
            acc = evaluate(model, test_loader, device)
            history[epoch] = acc
            print(f"  [P4-B] {tag} epoch {epoch}/{EPOCHS_NP}: acc={acc:.4f}")

    del model, V_r, loader; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()

    with open(hpath, "wb") as f: pickle.dump(history, f)
    return history


def _run_part_b(Vt, data, device, out_dir):
    """Run all Part B configurations and produce Tab B-1 + Fig B-1."""
    import pickle
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "part_b_results.pkl")
    if os.path.exists(results_path):
        with open(results_path, "rb") as f: return pickle.load(f)

    print("\n[P4] === Part B: Projected Non-Private Training ===")
    histories = {}
    histories[None] = _train_with_projection(None, Vt, data, device, out_dir)
    for r in RANK_R_PART_B:
        histories[r] = _train_with_projection(r, Vt, data, device, out_dir)

    baseline_acc = histories[None][EPOCHS_NP]
    lines = ["=== Tab B-1: Part B Final Test Accuracy ===",
             f"{'Config':<15} {'Final acc':>10} {'Rel. to baseline':>18}"]
    lines.append("-" * 46)
    lines.append(f"  {'Baseline':<13} {baseline_acc:>10.4f}   {'1.000':>10}")
    for r in RANK_R_PART_B:
        acc = histories[r][EPOCHS_NP]
        rel = acc / baseline_acc
        lines.append(f"  {'Proj-' + str(r):<13} {acc:>10.4f}   {rel:>10.3f}")
    lines.append("")
    tab_text = "\n".join(lines)
    print(tab_text)
    with open(os.path.join(out_dir, "tab_B1.txt"), "w") as f: f.write(tab_text)

    # Fig B-1: learning curves
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        configs = [None] + RANK_R_PART_B
        colors  = ["black"] + ["tab:blue", "tab:green", "tab:orange", "tab:red"]
        labels  = ["Baseline (full grad)"] + [f"Proj-{r}" for r in RANK_R_PART_B]
        for cfg, col, lbl in zip(configs, colors, labels):
            h = histories[cfg]
            epochs_ = sorted(h.keys())
            ax.plot(epochs_, [h[e] for e in epochs_], "o-", color=col, lw=2,
                    markersize=4, label=lbl)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Test accuracy")
        ax.set_title("Part B: Projected vs Full Gradient Training (Non-Private)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_B1_curves.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    results = {"histories": histories, "baseline_acc": baseline_acc}
    with open(results_path, "wb") as f: pickle.dump(results, f)
    return results


# ---------------------------------------------------------------------------
# Part C — Projected DP-SGD
# ---------------------------------------------------------------------------

def _projected_dp_step(gsm, optimizer, x_clean, y, V_r, sigma, C, q,
                        n_train, aug_mult, chunk_size):
    """
    Chunked Projected DP-SGD step.
    Clips per-sample gradients in the r-dimensional projected space.
    Privacy: identical to standard DP-SGD (same σ, sensitivity C).
    Noise energy: r × σ²C² instead of d × σ²C².
    """
    B      = x_clean.shape[0]
    r      = V_r.shape[1]
    device = x_clean.device
    agg_coeffs = torch.zeros(r, device=device)

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        Bc  = end - start
        xc  = x_clean[start:end]; yc = y[start:end]

        x_views = torch.cat([_augment_batch(xc) for _ in range(aug_mult)], 0)
        y_views = yc.repeat(aug_mult)

        _clear_grad_samples(gsm)
        gsm.train()
        out = gsm(x_views)
        F.cross_entropy(out, y_views, reduction="sum").backward()

        # Flatten + average over aug_mult views → (Bc, d)
        flat_parts = []
        for p in gsm.parameters():
            if p.requires_grad and hasattr(p, "grad_sample") and p.grad_sample is not None:
                flat_parts.append(p.grad_sample.reshape(Bc, aug_mult, -1).mean(1))
            else:
                flat_parts.append(torch.zeros(Bc, p.numel(), device=device))
        flat = torch.cat(flat_parts, dim=1)   # (Bc, d)

        # Project: coeffs_i = g_i @ V_r  →  (Bc, r)
        coeffs = flat @ V_r

        # Clip in r-dimensional projected space
        norms    = coeffs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        clip_fac = torch.clamp(C / norms, max=1.0)
        agg_coeffs += (coeffs * clip_fac).sum(0)

        _clear_grad_samples(gsm)
        del flat, coeffs, flat_parts
        if device.type == "cuda": torch.cuda.empty_cache()

    # Add r-dimensional Gaussian noise
    noise  = torch.randn(r, device=device) * (sigma * C)
    noised = (agg_coeffs + noise) / (q * n_train)   # (r,)

    # Embed back: update = V_r @ noised  →  (d,)
    update = V_r @ noised

    offset = 0
    for p in gsm.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.grad = update[offset:offset + numel].reshape(p.shape).clone()
            offset += numel

    optimizer.step(); optimizer.zero_grad()


def _train_projected_dp(r_or_none, Vt, data, device, out_dir):
    """
    Train DP-SGD with (or without) projected gradients.
    r_or_none=None → standard DP-SGD (no projection).
    Returns history dict {epoch: test_acc}.
    """
    import pickle
    from opacus.grad_sample import GradSampleModule
    from src.dp_training import _aug_mult_step  # standard augmult step

    tag   = "standard" if r_or_none is None else f"proj_{r_or_none}"
    hpath = os.path.join(out_dir, f"part_c_{tag}_history.pkl")
    if os.path.exists(hpath):
        with open(hpath, "rb") as f: return pickle.load(f)

    n_train = len(data["private_dataset_noaug"])
    q       = BATCH_SIZE_DP / n_train
    use_pin = device.type == "cuda"
    loader  = DataLoader(
        data["private_dataset_noaug"], batch_size=BATCH_SIZE_DP,
        shuffle=True, num_workers=4, pin_memory=use_pin, drop_last=True,
    )
    T     = EPOCHS_DP * len(loader)
    sigma = compute_sigma(EPS_TARGET, DELTA, q, T)
    print(f"[P4-C] {tag}: n_train={n_train}, σ={sigma:.4f}, "
          f"T={T}, aug_mult={AUG_MULT_DP}, q={q:.4f}")

    from src.dp_training import _aug_mult_step as _std_step
    set_seed(5555 + (0 if r_or_none is None else r_or_none))
    model = make_model("wrn28-2", data["num_classes"])
    assert validate_model_for_dp(model)
    gsm = GradSampleModule(model).to(device)
    opt = torch.optim.SGD(gsm.parameters(), lr=0.5 * (BATCH_SIZE_DP * AUG_MULT_DP / 256) ** 0.5,
                          momentum=0.9, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_DP)

    # Pre-load V_r onto device
    V_r = None
    if r_or_none is not None:
        V_r = torch.from_numpy(Vt[:r_or_none].T.astype(np.float32)).to(device)

    _, _, test_loader = make_data_loaders(data, batch_size=256)
    history = {}

    for epoch in range(1, EPOCHS_DP + 1):
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            if V_r is not None:
                _projected_dp_step(gsm, opt, x, y, V_r, sigma, C_TRAIN,
                                   q, n_train, AUG_MULT_DP, DP_CHUNK)
            else:
                # Standard augmult DP step
                _std_aug_mult_dp_step(gsm, opt, x, y, sigma, C_TRAIN,
                                      q, n_train, AUG_MULT_DP)
        sch.step()
        if epoch % 25 == 0 or epoch == EPOCHS_DP:
            acc = evaluate(gsm._module, test_loader, device)
            history[epoch] = acc
            print(f"  [P4-C] {tag} epoch {epoch}/{EPOCHS_DP}: acc={acc:.4f}")

    del gsm, opt, sch, loader, V_r; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()

    with open(hpath, "wb") as f: pickle.dump(history, f)
    return history


def _std_aug_mult_dp_step(gsm, optimizer, x_clean, y, sigma, C, q, n_train, aug_mult):
    """Standard augmult DP-SGD step (no projection)."""
    B       = x_clean.shape[0]
    x_views = torch.cat([_augment_batch(x_clean) for _ in range(aug_mult)], 0)
    y_views = y.repeat(aug_mult)
    _clear_grad_samples(gsm); gsm.train()
    out = gsm(x_views)
    F.cross_entropy(out, y_views, reduction="sum").backward()
    sq_norms = torch.zeros(B, device=x_clean.device)
    for p in gsm.parameters():
        if p.requires_grad and hasattr(p, "grad_sample") and p.grad_sample is not None:
            sq_norms += p.grad_sample.reshape(B, aug_mult, -1).mean(1).pow(2).sum(1)
    scale = torch.clamp(C / sq_norms.sqrt_().clamp_(min=1e-8), max=1.0)
    agg = []
    for p in gsm.parameters():
        if not p.requires_grad: continue
        if hasattr(p, "grad_sample") and p.grad_sample is not None:
            gs = p.grad_sample.reshape(B, aug_mult, -1).mean(1)
            agg.append((gs * scale[:, None]).sum(0).reshape(p.shape))
        else:
            agg.append(torch.zeros_like(p))
    _clear_grad_samples(gsm)
    for p, a in zip((p for p in gsm.parameters() if p.requires_grad), agg):
        p.grad = (a + torch.randn_like(a) * (sigma * C)) / (q * n_train)
    optimizer.step(); optimizer.zero_grad()


def _run_part_c(Vt, data, device, out_dir):
    """Run Part C: standard + projected DP-SGD at matched ε."""
    import pickle
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "part_c_results.pkl")
    if os.path.exists(results_path):
        with open(results_path, "rb") as f: return pickle.load(f)

    print("\n[P4] === Part C: Projected DP-SGD ===")
    histories = {}
    histories["standard"] = _train_projected_dp(None, Vt, data, device, out_dir)
    for r in RANK_R_PART_C:
        histories[f"proj_{r}"] = _train_projected_dp(r, Vt, data, device, out_dir)

    std_acc = histories["standard"][EPOCHS_DP]
    lines   = ["=== Tab C-1: Part C Final Test Accuracy (ε ≈ 3, matched) ===",
               f"{'Config':<15} {'Final acc':>10} {'Delta vs std':>14}"]
    lines.append("-" * 42)
    lines.append(f"  {'Standard':<13} {std_acc:>10.4f}   {'---':>10}")
    for r in RANK_R_PART_C:
        acc  = histories[f"proj_{r}"][EPOCHS_DP]
        diff = acc - std_acc
        flag = " ✓" if diff >= 0.05 else ""
        lines.append(f"  {'Proj-' + str(r):<13} {acc:>10.4f}   {diff:>+9.4f}{flag}")
    lines.append("")
    tab_text = "\n".join(lines)
    print(tab_text)
    with open(os.path.join(out_dir, "tab_C1.txt"), "w") as f: f.write(tab_text)

    # Fig C-1: learning curves
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"standard": "black", "proj_100": "tab:blue", "proj_500": "tab:green"}
        labels = {"standard": "Standard DP-SGD", "proj_100": "Proj-100", "proj_500": "Proj-500"}
        for key, h in histories.items():
            ep = sorted(h.keys())
            ax.plot(ep, [h[e] for e in ep], "o-", color=colors.get(key, "gray"),
                    lw=2, markersize=4, label=labels.get(key, key))
        ax.set_xlabel("Epoch"); ax.set_ylabel("Test accuracy")
        ax.set_title(f"Part C: Projected DP-SGD (ε ≈ {EPS_TARGET}, matched σ)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_C1_curves.png"), dpi=150)
        plt.close(fig)

        # Fig C-2: bar chart accuracy vs r
        fig, ax = plt.subplots(figsize=(6, 4))
        labels_ = ["Std\n(r=d)", "Proj-100", "Proj-500"]
        vals    = [std_acc] + [histories[f"proj_{r}"][EPOCHS_DP] for r in RANK_R_PART_C]
        bars    = ax.bar(labels_, vals, color=["gray", "tab:blue", "tab:green"], alpha=0.8)
        ax.axhline(std_acc, color="gray", linestyle="--", alpha=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Test accuracy"); ax.set_ylim(max(0, min(vals) - 0.05), min(1, max(vals) + 0.05))
        ax.set_title(f"Accuracy vs subspace dim r  (ε ≈ {EPS_TARGET}, same σ)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_C2_bar.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    results = {"histories": histories, "std_acc": std_acc}
    with open(results_path, "wb") as f: pickle.dump(results, f)
    return results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_exp_p4(
    dataset_name: str = "cifar10",
    imbalance_ratio: float = 50.0,
    seed: int = 0,
    device: torch.device = None,
    parts: str = "ABC",
    data_root: str = DATA_ROOT,
    results_dir: str = RESULTS_DIR,
):
    import pickle
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag     = f"{dataset_name}_IR{imbalance_ratio:.0f}_seed{seed}"
    out_dir = os.path.join(results_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    set_seed(seed)
    print(f"[P4] device={device}, parts={parts}")

    # Data + tiers
    data = load_datasets(
        dataset_name=dataset_name, data_root=data_root,
        imbalance_ratio=imbalance_ratio, public_frac=0.1, split_seed=42,
    )
    private_targets = np.array(data["private_dataset"].targets)
    tiers           = assign_tiers("A", private_targets, data["class_counts"], K=K)
    print(f"[P4] tier sizes: {get_tier_sizes(tiers, K)}")

    # ── Part A ────────────────────────────────────────────────────────────
    if "A" in parts:
        print("\n[P4] === Part A: Gradient Spectral Analysis ===")
        ref_state = _get_reference_model(
            data, device, os.path.join(out_dir, "reference_np.pt")
        )
        indices, tiers_idx = _select_grad_samples(data, tiers, out_dir)
        G = _compute_gradient_matrix(ref_state, data, indices, device, out_dir)
        S, Vt = _run_svd(G, device, out_dir)
        analysis = _analyze_part_a(G, tiers_idx, S, Vt, out_dir)
        _save_part_a_figures(S, analysis, out_dir)
        print(f"\n[P4] Part A complete. Go/no-go: {analysis['go_nogo']} "
              f"(r=100 capture={analysis['cap_at_100']:.4f})")
    else:
        # Load pre-computed SVD
        S_path  = os.path.join(out_dir, "svd_S.npy")
        Vt_path = os.path.join(out_dir, "svd_Vt_f16.npy")
        if not os.path.exists(S_path) or not os.path.exists(Vt_path):
            raise FileNotFoundError(
                "SVD results not found. Run with --parts A first."
            )
        S  = np.load(S_path)
        Vt = np.load(Vt_path).astype(np.float32)
        analysis = {"go_nogo": "GO", "cap_at_100": 1.0}  # assume pass if skipping

    # ── Part B ────────────────────────────────────────────────────────────
    if "B" in parts:
        part_b_dir = os.path.join(out_dir, "part_b")
        os.makedirs(part_b_dir, exist_ok=True)
        b_results = _run_part_b(Vt, data, device, part_b_dir)
        baseline_acc   = b_results["baseline_acc"]
        proj100_acc    = b_results["histories"].get(100, {}).get(EPOCHS_NP, 0.0)
        part_b_viable  = proj100_acc >= 0.80 * baseline_acc
        print(f"\n[P4] Part B: Proj-100 acc={proj100_acc:.4f}, "
              f"baseline={baseline_acc:.4f}, "
              f"ratio={proj100_acc/baseline_acc:.3f} "
              f"→ {'viable' if part_b_viable else 'NOT viable'} for Part C")
    else:
        part_b_viable = True  # assume viable if skipping Part B

    # ── Part C ────────────────────────────────────────────────────────────
    if "C" in parts:
        if not part_b_viable:
            print("[P4] Part B viability threshold not met — skipping Part C.")
        elif analysis["go_nogo"] == "NO-GO":
            print("[P4] Part A go/no-go FAILED — skipping Part C.")
        else:
            part_c_dir = os.path.join(out_dir, "part_c")
            os.makedirs(part_c_dir, exist_ok=True)
            _run_part_c(Vt, data, device, part_c_dir)

    print(f"\n[P4] Experiment complete — results in {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pickle

    parser = argparse.ArgumentParser(
        description="Phase 4: Gradient Low-Rank Structure + Projected DP-SGD"
    )
    parser.add_argument("--dataset",     default="cifar10")
    parser.add_argument("--ir",          type=float, default=50.0)
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--gpu",         type=int,   default=0)
    parser.add_argument("--data_root",   default=DATA_ROOT)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument(
        "--parts", default="ABC",
        help="Which parts to run: any combination of A, B, C  (e.g. 'AB', 'C')"
    )
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    run_exp_p4(
        dataset_name=args.dataset,
        imbalance_ratio=args.ir,
        seed=args.seed,
        device=device,
        parts=args.parts.upper(),
        data_root=args.data_root,
        results_dir=args.results_dir,
    )
