"""
exp_h1_direction.py — Phase 2: Coherence–Privacy Connection (phase2_spex.tex)
==============================================================================

Theoretical motivation (Chatterjee 2020 "Coherent Gradients"):
  ||G||² = Σ_i ||g_i||² + Σ_{i≠j} <g_i, g_j>
  Coherent directions (large cross terms) are stable — removing one example
  barely changes the aggregate → low privacy leakage.
  Incoherent directions are unstable → removing one example eliminates
  signal in that direction → high privacy leakage.

  The norm-based bound uses ||ḡ_i|| ≈ C for all i (Phase 1 confirmed: 99%+
  clipping).  Our hypothesis: the INCOHERENT COMPONENT ||ḡ_i^⊥|| stratifies
  by tier even when norms do not.

Measures computed (phase2_spex.tex §2):
  M1  cos θ_i^global  = <ḡ_i, Ḡ> / (||ḡ_i|| ||Ḡ||)                global coherence
  M2  cos θ_i^class   = <ḡ_i, Ḡ_c> / (||ḡ_i|| ||Ḡ_c||)            class coherence
  M3  f_i             = <Ḡ, ḡ_i> / ||Ḡ||²                          Chatterjee contrib
  M4  ||ḡ_i^⊥||      = ||ḡ_i|| √(1 − cos²θ_i^global)              incoherent norm
  M5  coh_i^kNN       = (1/k) Σ_{j∈kNN} cos(ḡ_i, ḡ_j)             k-NN coherence
  M6  outlier_score_i = fraction of 5000 sampled coords in tail      per-coord outlier

All measures computed on CLIPPED gradients ḡ_i = g_i · min(1, C/||g_i||).

Procedure:
  1. Train fresh model with Phase-0 best config (aug_mult=8, 200 epochs),
     saving checkpoints at epochs {10, 25, 50, 100, 150, 200}.
     (Loads Phase-1 checkpoint if available.)
  2. For each checkpoint: compute M1-M6 for the SAME 2000 stratified examples.
  3. Report Tab A (per-tier mean, Tier2/Tier0 ratio) at convergence (epoch 200).
  4. Generate Fig A-G per spec.

Success criteria:
  C1: M1 ratio (Tier0/Tier2) ≥ 1.5  OR  M4 ratio (Tier2/Tier0) ≥ 1.5
  C2: Gradient norm ratio ≈ 1.0  (confirm Phase-1 finding)
  C3: Loss ratio Tier2/Tier0 ≥ 2×
  C4: Coherence stratification increases through training (Fig D monotonic)

Datasets:
  CIFAR-10-LT (IR=50): primary, Strategy A (class frequency) tiers
  CIFAR-10 balanced:   secondary, Strategy B (density) tiers

Outputs: results/exp_h1/<tag>/{results.pkl, tab_a.txt, figs/}
"""

import os, sys, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.datasets import load_datasets, make_data_loaders
from src.models import make_model, validate_model_for_dp
from src.dp_training import _clear_grad_samples, evaluate, set_seed
from src.calibration import train_public_model, _get_raw_model
from src.tiers import assign_tiers, get_tier_sizes
from src.privacy_accounting import compute_sigma
from src.evaluation import save_results, extract_features


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = "results/exp_h1"
EXP1_DIR    = "results/exp1_p1"
DATA_ROOT   = "./data"
ARCH        = "wrn28-2"

DATASETS = [
    ("cifar10", 50.0, "A"),   # LT — primary, Strategy A
    ("cifar10",  1.0, "B"),   # balanced — secondary, Strategy B
]
SEEDS    = [0, 1, 2]
K        = 3

CHECKPOINT_EPOCHS = [10, 25, 50, 100, 150, 200]   # spec §3.4
N_SAMPLES  = 2000    # total stratified samples (≈667 per tier)
K_NN       = 10      # k for M5 k-NN coherence
PROJ_DIM   = 5000    # random projection dim for M5 (JL — spec recommends d'=5000)
PROJ_SEED  = 42
COORD_DIM  = 5000    # number of coordinates subsampled for M6 outlier score
COORD_SEED = 123     # fixed seed — SAME coordinates across all checkpoints/seeds
PROJ_CHUNK = 200     # rows of projection matrix generated at a time (memory)
GRAD_BATCH = 32      # samples per vmap call

# Phase-0 best config (must match exp1_p1)
BATCH_SIZE = 256
AUG_MULT   = 8
LR         = 0.5 * (BATCH_SIZE * AUG_MULT / 256) ** 0.5   # ≈1.4142
EPOCHS     = 200
EPS_TARGET = 3.0
DELTA      = 1e-5
C_TRAIN    = 1.0


# ---------------------------------------------------------------------------
# Augmentation helpers (local copies — avoid circular imports)
# ---------------------------------------------------------------------------

def _augment_batch(x):
    B, C_, H, W = x.shape
    xp = F.pad(x, (4, 4, 4, 4), mode='reflect')
    oi = torch.randint(0, 8, (B,), device=x.device)
    oj = torch.randint(0, 8, (B,), device=x.device)
    rows  = oi[:, None] + torch.arange(H, device=x.device)
    cols  = oj[:, None] + torch.arange(W, device=x.device)
    b_idx = torch.arange(B, device=x.device)[:, None, None]
    crops = xp.permute(0, 2, 3, 1)[b_idx, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2).contiguous()
    flip  = torch.rand(B, device=x.device) > 0.5
    crops[flip] = crops[flip].flip(-1)
    return crops


def _aug_mult_step(gsm, optimizer, x_clean, y, sigma, C, q, n_train, aug_mult):
    B = x_clean.shape[0]
    x_views = torch.cat([_augment_batch(x_clean) for _ in range(aug_mult)], dim=0)
    y_views = y.repeat(aug_mult)
    _clear_grad_samples(gsm)
    gsm.train()
    out = gsm(x_views)
    F.cross_entropy(out, y_views, reduction='sum').backward()
    sq_norms = torch.zeros(B, device=x_clean.device)
    for p in gsm.parameters():
        if p.requires_grad and hasattr(p, 'grad_sample') and p.grad_sample is not None:
            gs = p.grad_sample.reshape(B, aug_mult, -1).mean(1)
            sq_norms += gs.pow(2).sum(1)
    scale = torch.clamp(C / sq_norms.sqrt_().clamp_(min=1e-8), max=1.0)
    agg = []
    for p in gsm.parameters():
        if not p.requires_grad:
            continue
        if hasattr(p, 'grad_sample') and p.grad_sample is not None:
            gs = p.grad_sample.reshape(B, aug_mult, -1).mean(1)
            agg.append((gs * scale[:, None]).sum(0).reshape(p.shape))
        else:
            agg.append(torch.zeros_like(p))
    _clear_grad_samples(gsm)
    param_iter = (p for p in gsm.parameters() if p.requires_grad)
    for p, a in zip(param_iter, agg):
        p.grad = (a + torch.randn_like(a) * (sigma * C)) / (q * n_train)
    optimizer.step(); optimizer.zero_grad()


# ---------------------------------------------------------------------------
# Training with checkpoint saving
# ---------------------------------------------------------------------------

def _train_with_checkpoints(dataset_name, ir, seed, data, device, ckpt_dir):
    """
    Train Phase-0 best config model, saving state dict at each
    CHECKPOINT_EPOCHS epoch.  Skips epochs already saved.
    Returns path to final checkpoint.
    """
    from opacus.grad_sample import GradSampleModule
    from opacus.accountants import RDPAccountant

    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"epoch_{CHECKPOINT_EPOCHS[-1]}.pt")

    # Check which checkpoints already exist
    missing = [e for e in CHECKPOINT_EPOCHS
               if not os.path.exists(os.path.join(ckpt_dir, f"epoch_{e}.pt"))]
    if not missing:
        print(f"[H1] all checkpoints present in {ckpt_dir}, skipping training.")
        return final_path

    # Try to reuse Phase-1 final model (epoch 200 only)
    p1_model = os.path.join(EXP1_DIR,
                            f"{dataset_name}_IR{ir:.0f}_seed{seed}",
                            "model_final.pt")
    if os.path.exists(p1_model) and missing == [] or \
       (os.path.exists(p1_model) and missing == [CHECKPOINT_EPOCHS[-1]]):
        import shutil
        shutil.copy(p1_model, final_path)
        print(f"[H1] copied Phase-1 model to {final_path}")
        missing = [e for e in CHECKPOINT_EPOCHS
                   if not os.path.exists(os.path.join(ckpt_dir, f"epoch_{e}.pt"))]
        if not missing:
            return final_path

    print(f"[H1] training fresh model — checkpoints needed: {missing}")
    num_classes = data["num_classes"]
    n_train     = data["n_train"]
    q           = BATCH_SIZE / n_train

    train_loader = DataLoader(
        data["private_dataset_noaug"], batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    T     = EPOCHS * len(train_loader)
    sigma = compute_sigma(EPS_TARGET, DELTA, q, T)

    model = make_model(ARCH, num_classes)
    assert validate_model_for_dp(model)
    gsm  = GradSampleModule(model).to(device)
    opt  = torch.optim.SGD(gsm.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    _, _, test_loader = make_data_loaders(data, batch_size=256)

    for epoch in range(1, EPOCHS + 1):
        for batch in train_loader:
            _aug_mult_step(gsm, opt, batch[0].to(device), batch[1].to(device),
                           sigma=sigma, C=C_TRAIN, q=q, n_train=n_train,
                           aug_mult=AUG_MULT)
        sch.step()
        if epoch in CHECKPOINT_EPOCHS:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
            torch.save(gsm._module.state_dict(), ckpt_path)
            acc = evaluate(gsm._module, test_loader, device)
            print(f"  epoch {epoch}: test_acc={acc:.4f}  saved → {ckpt_path}")

    del gsm, opt, sch, train_loader
    import gc; gc.collect(); torch.cuda.empty_cache()
    return final_path


# ---------------------------------------------------------------------------
# Gradient feature computation (two-pass, memory-efficient)
# ---------------------------------------------------------------------------

def _compute_measures_at_checkpoint(
    model_state, num_classes, subset_loader, device,
    labels_all, coord_subset,
):
    """
    Compute M1-M6 for all N samples using the given model state dict.

    Two passes over subset_loader:
      Pass 1: accumulate Ḡ (global + class-cond), clip, store norms, losses
      Pass 2: compute M1-M4 exactly; store [N, PROJ_DIM] for M5;
              store [N, COORD_DIM] subsampled coords for M6

    coord_subset: (COORD_DIM,) int tensor — fixed coordinate indices

    Returns dict of arrays, each (N,).
    """
    from torch.func import vmap, grad, functional_call

    raw_model = make_model(ARCH, num_classes)
    raw_model.load_state_dict(model_state)
    raw_model.eval().to(device)

    params  = {n: p.detach() for n, p in raw_model.named_parameters()}
    buffers = {n: b.detach() for n, b in raw_model.named_buffers()}
    param_names = list(params.keys())
    d_total = sum(params[n].numel() for n in param_names)

    def loss_single(params, x_i, y_i):
        out = functional_call(raw_model, {**params, **buffers}, (x_i.unsqueeze(0),))
        return F.cross_entropy(out, y_i.unsqueeze(0))

    per_sample_grad_fn = vmap(grad(loss_single), in_dims=(None, 0, 0))

    # Suppress backward hooks (safety — mirrors calibration.py)
    saved_bw, saved_bw_pre = {}, {}
    for name, m in raw_model.named_modules():
        if m._backward_hooks:
            saved_bw[name] = dict(m._backward_hooks); m._backward_hooks.clear()
        bwp = getattr(m, '_backward_pre_hooks', None)
        if bwp:
            saved_bw_pre[name] = dict(bwp); bwp.clear()

    try:
        # ── Pass 1: raw grads → clip → accumulate Ḡ / Ḡ_c, norms, losses ──
        grad_sum_flat = torch.zeros(d_total)           # CPU, global sum
        class_sums    = {}                             # class_int → (d_total,)
        class_cnts    = {}                             # class_int → int
        all_norms_raw, all_losses = [], []
        N = 0

        for batch in subset_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            B = x.shape[0]

            per_grads = per_sample_grad_fn(params, x, y)
            flat = torch.cat(
                [per_grads[n].detach().reshape(B, -1) for n in param_names], dim=1
            )  # (B, d_total) on GPU

            # Clip: ḡ_i = g_i * min(1, C/||g_i||)
            raw_norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
            clip_fac  = torch.clamp(C_TRAIN / raw_norms, max=1.0)
            clipped   = flat * clip_fac                # (B, d_total) GPU

            all_norms_raw.append(raw_norms.squeeze(1).cpu())

            with torch.no_grad():
                out_l = raw_model(x)
                all_losses.append(F.cross_entropy(out_l, y, reduction='none').cpu())

            clipped_cpu = clipped.cpu()
            grad_sum_flat += clipped_cpu.sum(0)

            for b in range(B):
                c = int(y[b].item())
                if c not in class_sums:
                    class_sums[c] = torch.zeros(d_total)
                    class_cnts[c] = 0
                class_sums[c] += clipped_cpu[b]
                class_cnts[c] += 1
            N += B

        # Ḡ  (global mean clipped gradient)
        G_flat  = grad_sum_flat / N                    # (d_total,) CPU
        G_norm  = G_flat.norm().item()
        G_hat   = G_flat / max(G_norm, 1e-8)          # unit vector

        # Ḡ_c  (class-conditional means)
        G_class = {c: class_sums[c] / class_cnts[c]
                   for c in class_sums if class_cnts[c] > 0}
        G_class_norms = {c: v.norm().item() for c, v in G_class.items()}

        G_flat_dev = G_flat.to(device)
        G_hat_dev  = G_hat.to(device)
        G_class_dev = {c: v.to(device) for c, v in G_class.items()}

        # ── Pass 2: M1-M4 (exact), M5 (projected), M6 (subsampled coords) ──
        all_norms_clip = []
        all_dot_G      = []    # <ḡ_i, Ḡ>  for M1, M3, M4
        all_dot_Gc     = []    # <ḡ_i, Ḡ_c> for M2
        all_proj       = []    # (N, PROJ_DIM) for M5
        all_coords     = []    # (N, COORD_DIM) for M6

        for batch in subset_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            B = x.shape[0]

            per_grads = per_sample_grad_fn(params, x, y)
            flat = torch.cat(
                [per_grads[n].detach().reshape(B, -1) for n in param_names], dim=1
            )  # (B, d_total) GPU

            raw_norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
            clipped   = flat * torch.clamp(C_TRAIN / raw_norms, max=1.0)

            clip_norms = clipped.norm(dim=1)           # (B,) ≈ C for saturated
            all_norms_clip.append(clip_norms.cpu())

            # <ḡ_i, Ḡ>  (for M1, M3, M4)
            dots_G = (clipped * G_flat_dev).sum(1)     # (B,)
            all_dot_G.append(dots_G.cpu())

            # <ḡ_i, Ḡ_c>  (for M2, class-conditional)
            dots_Gc = torch.zeros(B)
            for b in range(B):
                c = int(y[b].item())
                if c in G_class_dev:
                    dots_Gc[b] = (clipped[b] * G_class_dev[c]).sum().item()
            all_dot_Gc.append(dots_Gc)

            # Random projection for M5 k-NN  (CPU, chunked)
            clipped_cpu = clipped.cpu()
            proj_accum  = torch.zeros(B, PROJ_DIM)
            for c_start in range(0, PROJ_DIM, PROJ_CHUNK):
                c_end = min(c_start + PROJ_CHUNK, PROJ_DIM)
                gen   = torch.Generator()
                gen.manual_seed(PROJ_SEED * 137 + c_start)
                P_chunk = torch.randn(c_end - c_start, d_total, generator=gen) \
                          / (PROJ_DIM ** 0.5)
                proj_accum[:, c_start:c_end] = clipped_cpu @ P_chunk.T
            all_proj.append(proj_accum)

            # Subsampled coordinates for M6
            all_coords.append(clipped_cpu[:, coord_subset])  # (B, COORD_DIM)

    finally:
        for name, m in raw_model.named_modules():
            if name in saved_bw:
                m._backward_hooks.update(saved_bw[name])
            bwp = getattr(m, '_backward_pre_hooks', None)
            if bwp is not None and name in saved_bw_pre:
                bwp.update(saved_bw_pre[name])
        del raw_model
        torch.cuda.empty_cache()

    # ── Compute M1-M6 ─────────────────────────────────────────────────────
    norms_raw  = torch.cat(all_norms_raw).numpy()      # (N,)
    norms_clip = torch.cat(all_norms_clip).numpy()      # (N,) ≈ C everywhere
    dot_G      = torch.cat(all_dot_G).numpy()           # (N,) <ḡ_i, Ḡ>
    dot_Gc     = torch.cat(all_dot_Gc).numpy()          # (N,) <ḡ_i, Ḡ_c>
    proj       = torch.cat(all_proj)                    # (N, PROJ_DIM)
    coords_mat = torch.cat(all_coords).numpy()          # (N, COORD_DIM)
    losses     = torch.cat(all_losses).numpy()          # (N,)

    # M1: global coherence
    M1 = dot_G / (norms_clip.clip(1e-8) * max(G_norm, 1e-8))

    # M2: class-conditional coherence
    M2 = np.array([
        dot_Gc[i] / (norms_clip[i] * max(G_class_norms.get(int(labels_all[i]), 1e-8), 1e-8))
        for i in range(N)
    ])

    # M3: Chatterjee per-example contribution  f_i = <Ḡ, ḡ_i> / ||Ḡ||²
    M3 = dot_G / max(G_norm ** 2, 1e-16)

    # M4: incoherent component norm  ||ḡ_i^⊥|| = ||ḡ_i|| √(1 − cos²θ)
    M4 = norms_clip * np.sqrt(np.maximum(1.0 - M1 ** 2, 0.0))

    # M5: k-NN coherence in projected gradient space
    proj_norms = proj.norm(dim=1, keepdim=True).clamp(min=1e-8)
    proj_n     = proj / proj_norms                       # (N, PROJ_DIM) normalised
    cos_sim    = proj_n @ proj_n.T                       # (N, N)
    cos_sim.fill_diagonal_(-2.0)
    topk_sim, _ = cos_sim.topk(K_NN, dim=1)             # (N, K_NN)
    M5 = topk_sim.mean(dim=1).numpy()                   # high = coherent

    # M6: per-coordinate outlier score (fraction of COORD_DIM coords in tails)
    q05 = np.percentile(coords_mat, 5,  axis=0, keepdims=True)   # (1, COORD_DIM)
    q95 = np.percentile(coords_mat, 95, axis=0, keepdims=True)   # (1, COORD_DIM)
    is_out = (coords_mat > q95) | (coords_mat < q05)              # (N, COORD_DIM)
    M6 = is_out.astype(np.float32).mean(axis=1)                   # (N,)

    return {
        "grad_norm_raw":  norms_raw,
        "grad_norm_clip": norms_clip,
        "losses":         losses,
        "M1":  M1,
        "M2":  M2,
        "M3":  M3,
        "M4":  M4,
        "M5":  M5,
        "M6":  M6,
        "G_norm":         G_norm,
    }


# ---------------------------------------------------------------------------
# Tab A — per-tier statistics at convergence
# ---------------------------------------------------------------------------

def compute_tab_a(measures, tiers):
    """
    Per-tier mean, std, median for each measure + Tier2/Tier0 ratio.
    Returns dict: measure_name → {tier_k: {mean, std, median, n},
                                  'ratio_2_0': float}
    """
    K_actual = int(tiers.max()) + 1
    tab = {}
    for name, vals in measures.items():
        if name in ("G_norm",):
            continue
        tab[name] = {}
        tier_means = []
        for k in range(K_actual):
            mask = tiers == k
            v = vals[mask]
            if len(v) == 0:
                tier_means.append(0.0)
                continue
            tab[name][k] = {
                "mean":   float(v.mean()),
                "std":    float(v.std()),
                "median": float(np.median(v)),
                "n":      int(mask.sum()),
            }
            tier_means.append(float(v.mean()))
        if len(tier_means) >= 3 and abs(tier_means[0]) > 1e-10:
            tab[name]["ratio_2_0"] = tier_means[2] / tier_means[0]
    return tab


def print_tab_a(tab, dataset_tag=""):
    # Decide expected direction per measure (for pass/fail flags)
    # M1, M2, M5: head > tail → ratio T0/T2 should be > 1.5
    # M3:         head > tail (head examples reinforce aggregate more)
    # M4, M6:     tail > head → ratio T2/T0 should be > 1.5
    # norm:       should be ≈ 1.0 (confirm Phase-1)
    # loss:       tail > head → T2/T0 > 2
    head_high = {"M1", "M2", "M3", "M5", "grad_norm_clip", "grad_norm_raw"}

    print(f"\n=== Tab A — {dataset_tag} ===")
    print(f"{'Measure':<20} {'T0 mean':>10} {'T1 mean':>10} {'T2 mean':>10} "
          f"{'Ratio':>8}  {'Note'}")
    print("-" * 75)

    measure_order = ["grad_norm_clip", "grad_norm_raw",
                     "M1", "M2", "M3", "M4", "M5", "M6", "losses"]
    for name in measure_order:
        if name not in tab:
            continue
        d = tab[name]
        t0 = d.get(0, {}).get("mean", float("nan"))
        t1 = d.get(1, {}).get("mean", float("nan"))
        t2 = d.get(2, {}).get("mean", float("nan"))
        raw_ratio = d.get("ratio_2_0", float("nan"))

        if name in head_high:
            disp_ratio = (t0 / max(abs(t2), 1e-10)) if not np.isnan(t0) else float("nan")
            ratio_label = "T0/T2"
        else:
            disp_ratio = raw_ratio
            ratio_label = "T2/T0"

        # Flags
        flag = ""
        if name == "M1" and disp_ratio >= 1.5:
            flag = " ✓ C1"
        elif name == "M4" and (raw_ratio >= 1.5):
            flag = " ✓ C1"
        elif name == "grad_norm_clip" and 0.95 <= disp_ratio <= 1.05:
            flag = " ✓ C2"
        elif name == "losses" and raw_ratio >= 2.0:
            flag = " ✓ C3"

        print(f"  {name:<18} {t0:>10.4f} {t1:>10.4f} {t2:>10.4f} "
              f"  {disp_ratio:>6.3f} ({ratio_label}){flag}")
    print()


# ---------------------------------------------------------------------------
# Figures (Fig A–G per spec)
# ---------------------------------------------------------------------------

def _save_figures(all_ckpt_measures, tiers, selected_idx, tag, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[H1] matplotlib not available — skipping figures.")
        return

    os.makedirs(out_dir, exist_ok=True)
    K_actual = int(tiers.max()) + 1
    colors   = ["tab:blue", "tab:orange", "tab:red"]
    tlabels  = {0: "Tier 0 (head)", 1: "Tier 1 (mid)", 2: "Tier 2 (tail)"}

    final_epoch = CHECKPOINT_EPOCHS[-1]
    m = all_ckpt_measures[final_epoch]

    def hist2(ax_l, ax_r, vals_l, vals_r, xl, xr, title_l, title_r):
        for k in range(K_actual):
            mask = tiers == k
            ax_l.hist(vals_l[mask], bins=40, alpha=0.5, color=colors[k],
                      label=tlabels[k], density=True)
            ax_r.hist(vals_r[mask], bins=40, alpha=0.5, color=colors[k],
                      label=tlabels[k], density=True)
        for ax, xl_, tl in [(ax_l, xl, title_l), (ax_r, xr, title_r)]:
            ax.set_xlabel(xl_); ax.set_ylabel("Density"); ax.set_title(tl)
            ax.legend(fontsize=8)

    # Fig A: gradient norm (left) vs global coherence M1 (right) — THE central figure
    fig, (al, ar) = plt.subplots(1, 2, figsize=(12, 4))
    hist2(al, ar, m["grad_norm_clip"], m["M1"],
          "||ḡ_i|| (clipped norm)",
          "cos θ_i^global  (M1 — global coherence)",
          f"Clipped norm — {tag}", f"Global coherence — {tag}")
    fig.suptitle("Fig A: Norm vs Coherence (central result)", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_A.png"), dpi=120); plt.close(fig)

    # Fig B: incoherent norm M4 (left) vs Chatterjee f_i M3 (right)
    fig, (al, ar) = plt.subplots(1, 2, figsize=(12, 4))
    hist2(al, ar, m["M4"], m["M3"],
          "||ḡ_i^⊥||  (M4 — incoherent norm)",
          "f_i  (M3 — Chatterjee contribution)",
          f"Incoherent norm — {tag}", f"Chatterjee f_i — {tag}")
    fig.suptitle("Fig B", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_B.png"), dpi=120); plt.close(fig)

    # Fig C: k-NN coherence M5 (left) vs outlier score M6 (right)
    fig, (al, ar) = plt.subplots(1, 2, figsize=(12, 4))
    hist2(al, ar, m["M5"], m["M6"],
          "coh_i^kNN  (M5 — k-NN coherence)",
          "outlier_score_i  (M6 — per-coord outlier frac)",
          f"k-NN coherence — {tag}", f"Outlier score — {tag}")
    fig.suptitle("Fig C", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_C.png"), dpi=120); plt.close(fig)

    # Fig D: per-tier mean of each measure vs epoch (trajectory)
    epochs_available = sorted(all_ckpt_measures.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax_i, mname in enumerate(["M1", "M2", "M3", "M4", "M5", "M6"]):
        ax = axes[ax_i]
        for k in range(K_actual):
            mask = tiers == k
            means = [all_ckpt_measures[e][mname][mask].mean()
                     for e in epochs_available]
            ax.plot(epochs_available, means, marker='o', color=colors[k],
                    label=tlabels[k])
        ax.set_xlabel("Epoch"); ax.set_ylabel(f"Mean {mname}")
        ax.set_title(f"{mname} trajectory"); ax.legend(fontsize=7)
    fig.suptitle(f"Fig D: Coherence measure trajectories — {tag}", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_D.png"), dpi=120); plt.close(fig)

    # Fig E: gradient norm (x) vs M1 global coherence (y), colored by tier
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in range(K_actual):
        mask = tiers == k
        ax.scatter(m["grad_norm_raw"][mask], m["M1"][mask],
                   alpha=0.3, s=8, color=colors[k], label=tlabels[k])
    ax.set_xlabel("||g_i||  (unclipped norm)"); ax.set_ylabel("cos θ_i^global  (M1)")
    ax.set_title(f"Fig E: Norm vs coherence — {tag}"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_E.png"), dpi=120); plt.close(fig)

    # Fig F: loss (x) vs M1 coherence (y), colored by tier
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in range(K_actual):
        mask = tiers == k
        ax.scatter(m["losses"][mask], m["M1"][mask],
                   alpha=0.3, s=8, color=colors[k], label=tlabels[k])
    ax.set_xlabel("Loss ℓ(θ*, z_i)"); ax.set_ylabel("cos θ_i^global  (M1)")
    ax.set_title(f"Fig F: Loss vs coherence — {tag}"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_F.png"), dpi=120); plt.close(fig)

    # Fig G: M1 global coherence (x) vs M4 incoherent norm (y)
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in range(K_actual):
        mask = tiers == k
        ax.scatter(m["M1"][mask], m["M4"][mask],
                   alpha=0.3, s=8, color=colors[k], label=tlabels[k])
    ax.set_xlabel("cos θ_i^global  (M1)"); ax.set_ylabel("||ḡ_i^⊥||  (M4)")
    ax.set_title(f"Fig G: Coherence vs incoherent norm — {tag}"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_G.png"), dpi=120); plt.close(fig)

    print(f"[H1] figures saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp_h1(
    dataset_name: str,
    imbalance_ratio: float,
    tier_strategy: str,
    seed: int,
    device: torch.device,
    data_root: str = DATA_ROOT,
    results_dir: str = RESULTS_DIR,
):
    tag     = f"{dataset_name}_IR{imbalance_ratio:.0f}_{tier_strategy}_seed{seed}"
    out_dir = os.path.join(results_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")

    result_path = os.path.join(out_dir, "results.pkl")
    if os.path.exists(result_path):
        print(f"[H1] {tag}: cached, loading.")
        with open(result_path, "rb") as f:
            return pickle.load(f)

    set_seed(seed)

    # ── 1. Data & tiers ───────────────────────────────────────────────────
    data = load_datasets(
        dataset_name=dataset_name, data_root=data_root,
        imbalance_ratio=imbalance_ratio,
        public_frac=0.1, split_seed=42,
    )
    num_classes     = data["num_classes"]
    private_targets = np.array(data["private_dataset"].targets)
    class_counts    = data["class_counts"]

    _, public_loader, _ = make_data_loaders(data, batch_size=256)
    eval_loader = DataLoader(
        data["private_dataset_noaug"], batch_size=256,
        shuffle=False, num_workers=4, pin_memory=True, drop_last=False,
    )

    public_model = make_model(ARCH, num_classes)
    public_model = train_public_model(
        public_model, public_loader, device, epochs=50, verbose=False
    )

    if tier_strategy == "A":
        tiers = assign_tiers("A", private_targets, class_counts, K=K)
    else:
        feats_pub, _ = extract_features(public_model, public_loader, device)
        feats_prv, _ = extract_features(public_model, eval_loader, device)
        tiers = assign_tiers("B", private_targets, class_counts, K=K,
                             features_public=feats_pub, features_all=feats_prv)

    tier_sizes = get_tier_sizes(tiers, K)
    print(f"[H1] {tag}: Strategy {tier_strategy} tier sizes: {tier_sizes}")

    # ── 2. Stratified sample selection (same 2000 for all checkpoints) ────
    rng         = np.random.default_rng(seed + 200)
    n_per_tier  = N_SAMPLES // K
    selected    = []
    for k in range(K):
        idx_k  = np.where(tiers == k)[0]
        chosen = rng.choice(idx_k, size=min(n_per_tier, len(idx_k)), replace=False)
        selected.extend(chosen.tolist())
    selected = np.array(selected)
    labels_sel = private_targets[selected]
    tiers_sel  = tiers[selected]
    print(f"[H1] {tag}: {len(selected)} samples selected "
          f"({n_per_tier}/tier), tier distribution: "
          f"{[(tiers_sel==k).sum() for k in range(K)]}")

    subset_loader = DataLoader(
        Subset(data["private_dataset_noaug"], selected),
        batch_size=GRAD_BATCH, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False,
    )

    # ── 3. Fixed coordinate subset for M6 (same across all checkpoints) ──
    d_total = sum(p.numel() for p in make_model(ARCH, num_classes).parameters()
                  if p.requires_grad)
    gen_coord = torch.Generator(); gen_coord.manual_seed(COORD_SEED)
    coord_subset = torch.randperm(d_total, generator=gen_coord)[:COORD_DIM]

    # ── 4. Train / load checkpoints ───────────────────────────────────────
    _train_with_checkpoints(
        dataset_name, imbalance_ratio, seed, data, device, ckpt_dir
    )

    # ── 5. Compute M1-M6 at each checkpoint ──────────────────────────────
    all_ckpt_measures = {}
    for epoch in CHECKPOINT_EPOCHS:
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[H1] {tag}: checkpoint epoch {epoch} not found, skipping.")
            continue
        print(f"[H1] {tag}: computing measures at epoch {epoch}...")
        state = torch.load(ckpt_path, map_location="cpu")
        measures = _compute_measures_at_checkpoint(
            state, num_classes, subset_loader, device, labels_sel, coord_subset
        )
        all_ckpt_measures[epoch] = measures
        print(f"  M1 mean/tier: {[(measures['M1'][tiers_sel==k].mean()) for k in range(K)]}")
        print(f"  M4 mean/tier: {[(measures['M4'][tiers_sel==k].mean()) for k in range(K)]}")

    # ── 6. Tab A (at convergence) ─────────────────────────────────────────
    final_measures = all_ckpt_measures.get(CHECKPOINT_EPOCHS[-1], {})
    tab_a = compute_tab_a(final_measures, tiers_sel)
    print_tab_a(tab_a, dataset_tag=tag)

    tab_path = os.path.join(out_dir, "tab_a.txt")
    with open(tab_path, "w") as f:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_tab_a(tab_a, dataset_tag=tag)
        f.write(buf.getvalue())

    # ── 7. Figures ────────────────────────────────────────────────────────
    _save_figures(all_ckpt_measures, tiers_sel, selected,
                  tag, os.path.join(out_dir, "figs"))

    # ── 8. Save ──────────────────────────────────────────────────────────
    results = {
        "tag":                tag,
        "dataset":            dataset_name,
        "imbalance_ratio":    imbalance_ratio,
        "tier_strategy":      tier_strategy,
        "seed":               seed,
        "selected_idx":       selected,
        "labels_sel":         labels_sel,
        "tiers_sel":          tiers_sel,
        "all_ckpt_measures":  all_ckpt_measures,
        "tab_a":              tab_a,
    }
    save_results(results, result_path)
    print(f"[H1] {tag}: saved to {result_path}")
    return results


def run_all(device=None, data_root=DATA_ROOT, results_dir=RESULTS_DIR):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[H1] device: {device}")
    for dname, ir, strat in DATASETS:
        for seed in SEEDS:
            run_exp_h1(dname, ir, strat, seed, device,
                       data_root=data_root, results_dir=results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Coherence–Privacy Analysis")
    parser.add_argument("--dataset",     default="cifar10")
    parser.add_argument("--ir",          type=float, default=50.0,
                        help="Imbalance ratio (50=LT, 1=balanced)")
    parser.add_argument("--strategy",    default=None,
                        help="Tier strategy override (A or B); auto-selected if omitted")
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--gpu",         type=int, default=0)
    parser.add_argument("--data_root",   default=DATA_ROOT)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--all",         action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.all:
        run_all(device=device, data_root=args.data_root, results_dir=args.results_dir)
    else:
        # Auto-select strategy: A for LT, B for balanced
        strat = args.strategy or ("A" if args.ir > 1.0 else "B")
        run_exp_h1(args.dataset, args.ir, strat, args.seed, device,
                   data_root=args.data_root, results_dir=args.results_dir)
