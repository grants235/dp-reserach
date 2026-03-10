"""
Figure generation for all Channeled DP-SGD experiments.

All figures follow the spec's output inventory.
Figures are saved as PDF (for LaTeX) and PNG (for preview).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List, Optional, Any

FIGURES_DIR = "results/figures"
DPI = 150
FIGSIZE_SINGLE = (5, 4)
FIGSIZE_WIDE = (10, 4)
FIGSIZE_TRIPLE = (12, 4)
TIER_COLORS = ["tab:blue", "tab:orange", "tab:red"]
TIER_LABELS = ["Tier 0 (head)", "Tier 1 (mid)", "Tier 2 (tail)"]


def _savefig(fig, name: str, out_dir: str = FIGURES_DIR):
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{name}.pdf")
    png_path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {png_path}")


# ---------------------------------------------------------------------------
# Fig 1: Gradient norm histogram by tier (Exp 1)
# ---------------------------------------------------------------------------

def fig1_gradient_norm_histograms(
    all_results: Dict,
    out_dir: str = FIGURES_DIR,
):
    """
    Histogram of ||ḡ_i|| at convergence, colored by tier.
    Three panels: CIFAR-10, CIFAR-10-LT (IR=50), CIFAR-100.
    """
    datasets = [
        ("cifar10_IR1", "CIFAR-10", "B"),
        ("cifar10_IR50", "CIFAR-10-LT (IR=50)", "A"),
        ("cifar100_IR1", "CIFAR-100", "B"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_TRIPLE)

    for ax, (dataset_key_prefix, title, strategy) in zip(axes, datasets):
        # Collect data across seeds
        norms_by_tier = {k: [] for k in range(3)}
        for key, r in all_results.items():
            if not key.startswith(dataset_key_prefix):
                continue
            ckpts = r["checkpoint_data"]
            final_epoch = max(e for e in ckpts.keys() if isinstance(e, int))
            clipped = ckpts[final_epoch]["clipped_norms"]
            tiers = r[f"tiers_{strategy}"]
            for k in range(3):
                norms_by_tier[k].extend(clipped[tiers == k].tolist())

        for k in range(3):
            vals = np.array(norms_by_tier[k])
            if len(vals) > 0:
                ax.hist(vals, bins=50, alpha=0.6, color=TIER_COLORS[k],
                        label=TIER_LABELS[k], density=True)

        ax.set_xlabel(r"$\|\bar{g}_i\|$", fontsize=11)
        ax.set_ylabel("Density" if ax == axes[0] else "", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xlim(left=0)

    fig.suptitle(r"Clipped gradient norm $\|\bar{g}_i\|$ by tier at convergence",
                 fontsize=12)
    fig.tight_layout()
    _savefig(fig, "fig1_gradient_histograms", out_dir)


# ---------------------------------------------------------------------------
# Fig 2: Per-tier mean gradient norm vs. epoch (Exp 1)
# ---------------------------------------------------------------------------

def fig2_gradnorm_vs_epoch(
    all_results: Dict,
    out_dir: str = FIGURES_DIR,
):
    """
    Per-tier mean ||ḡ_i|| vs epoch. One line per tier, three panels.
    """
    datasets = [
        ("cifar10_IR1", "CIFAR-10", "B"),
        ("cifar10_IR50", "CIFAR-10-LT (IR=50)", "A"),
        ("cifar100_IR1", "CIFAR-100", "B"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_TRIPLE)

    for ax, (dataset_key_prefix, title, strategy) in zip(axes, datasets):
        # Collect per-epoch per-tier means across seeds
        epochs_seen = set()
        tier_means = {k: {} for k in range(3)}

        for key, r in all_results.items():
            if not key.startswith(dataset_key_prefix):
                continue
            tiers = r[f"tiers_{strategy}"]
            for epoch, ckpt_data in r["checkpoint_data"].items():
                if not isinstance(epoch, int):
                    continue
                epochs_seen.add(epoch)
                clipped = ckpt_data["clipped_norms"]
                for k in range(3):
                    mask = (tiers == k)
                    mean_k = clipped[mask].mean() if mask.sum() > 0 else 0.0
                    tier_means[k].setdefault(epoch, []).append(mean_k)

        sorted_epochs = sorted(epochs_seen)
        for k in range(3):
            ys = [np.mean(tier_means[k].get(e, [0.0])) for e in sorted_epochs]
            ax.plot(sorted_epochs, ys, color=TIER_COLORS[k], label=TIER_LABELS[k],
                    marker="o", markersize=4)

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(r"Mean $\|\bar{g}_i\|$" if ax == axes[0] else "", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle(r"Per-tier mean clipped gradient norm $\|\bar{g}_i\|$ vs epoch",
                 fontsize=12)
    fig.tight_layout()
    _savefig(fig, "fig2_gradnorm_vs_epoch", out_dir)


# ---------------------------------------------------------------------------
# Fig 3: Log-log scatter of scaling law (Exp 2)
# ---------------------------------------------------------------------------

def fig3_scaling_law(exp2_results: Dict, out_dir: str = FIGURES_DIR):
    """
    Log-log scatter: (Δ̄_i/C)² (x-axis) vs ε_i/ε (y-axis).
    Color by tier. Overlay line with slope 1.
    """
    rms = np.array(exp2_results["rms_clipped_norms"])
    eps_i = np.array(exp2_results["eps_i_thudi"])
    tiers = np.array(exp2_results["selected_tiers"])
    C = exp2_results["C"]
    global_eps = exp2_results["global_eps"]

    x = (rms / C) ** 2
    y = eps_i / global_eps

    # Filter out zeros for log scale
    mask = (x > 0) & (y > 0)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    for k in range(3):
        tmask = mask & (tiers == k)
        ax.scatter(x[tmask], y[tmask], c=TIER_COLORS[k], label=TIER_LABELS[k],
                   alpha=0.6, s=20, edgecolors="none")

    # Slope-1 reference line
    xmin, xmax = x[mask].min() * 0.5, x[mask].max() * 2
    xs = np.array([xmin, xmax])
    ax.plot(xs, xs, "k--", linewidth=1.5, label="Slope 1 (theory)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$(\bar{\Delta}_i / C)^2$", fontsize=12)
    ax.set_ylabel(r"$\varepsilon_i / \varepsilon$", fontsize=12)
    ax.set_title("Per-instance privacy scaling law", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, "fig3_scaling_law", out_dir)


# ---------------------------------------------------------------------------
# Fig 4: 6-panel scatter of predictors vs ε_i (Exp 2)
# ---------------------------------------------------------------------------

def fig4_predictor_scatter(exp2_results: Dict, out_dir: str = FIGURES_DIR):
    """6-panel scatter: each predictor vs ε_i."""
    eps_i = np.array(exp2_results["eps_i_thudi"])
    tiers = np.array(exp2_results["selected_tiers"])
    preds = exp2_results["predictors"]

    pred_labels = {
        "grad_norm_convergence": r"$\|\nabla\ell(\theta^*, z_i)\|$",
        "loss_convergence": r"$\ell(\theta^*, z_i)$",
        "confidence": "1 – confidence",
        "mahalanobis": r"$\hat{\delta}_i$ (Mahalanobis)",
        "lt_iqr": "LT-IQR",
        "rms_clipped_norm_sq": r"$\bar{\Delta}_i^2$",
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes_flat = axes.flatten()

    for ax, (pred_key, pred_label) in zip(axes_flat, pred_labels.items()):
        x = np.array(preds[pred_key])
        for k in range(3):
            mask = (tiers == k)
            ax.scatter(x[mask], eps_i[mask], c=TIER_COLORS[k],
                       label=TIER_LABELS[k], alpha=0.5, s=15, edgecolors="none")
        ax.set_xlabel(pred_label, fontsize=9)
        ax.set_ylabel(r"$\varepsilon_i$" if ax in [axes_flat[0], axes_flat[3]] else "",
                      fontsize=9)
        ax.set_title(pred_label, fontsize=9)

    axes_flat[-1].legend(*axes_flat[0].get_legend_handles_labels(), fontsize=8)
    fig.suptitle(r"Predictors vs per-instance $\varepsilon_i$", fontsize=12)
    fig.tight_layout()
    _savefig(fig, "fig4_predictor_scatter", out_dir)


# ---------------------------------------------------------------------------
# Fig 5: Accuracy vs ε (Exp 3)
# ---------------------------------------------------------------------------

def fig5_accuracy_vs_epsilon(
    exp3_summary_cifar10: Dict,
    exp3_summary_cifar10lt: Dict,
    out_dir: str = FIGURES_DIR,
):
    """Line plot: test accuracy vs ε for top methods, two panels."""
    epsilons = [1.0, 3.0, 8.0]
    methods_to_plot = {
        "standard_best": ("Standard (best C)", "k-o"),
        "standard_median": ("Standard (median C)", "k--s"),
        "adaptive": ("Adaptive clip", "g-^"),
        "channeled_K3_A": ("Channeled K=3 (freq)", "b-D"),
        "channeled_K3_B": ("Channeled K=3 (density)", "r-v"),
        "non_private": ("Non-private", "gray"),
    }

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    for ax, (title, summary) in zip(axes, [
        ("CIFAR-10", exp3_summary_cifar10),
        ("CIFAR-10-LT (IR=50)", exp3_summary_cifar10lt),
    ]):
        for method_key, (label, fmt) in methods_to_plot.items():
            ys, yerrs = [], []
            for eps in epsilons:
                eps_key = f"eps{eps:.0f}"
                entry = summary.get(eps_key, {}).get(method_key, {})
                if "mean" in entry:
                    ys.append(entry["mean"])
                    yerrs.append(entry.get("std", 0.0))
                else:
                    ys.append(float("nan"))
                    yerrs.append(0.0)

            color = fmt.split("-")[0] if fmt != "gray" else "gray"
            linestyle = "-" if "--" not in fmt else "--"
            marker = fmt[-1] if not fmt.endswith("-") else None

            ax.errorbar(epsilons, ys, yerr=yerrs, label=label, capsize=3,
                        marker=marker, color=color, linestyle=linestyle)

        ax.set_xlabel(r"$\varepsilon$", fontsize=11)
        ax.set_ylabel("Test accuracy" if ax == axes[0] else "", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(epsilons)
        ax.legend(fontsize=7)

    fig.suptitle("Test accuracy vs privacy budget ε", fontsize=12)
    fig.tight_layout()
    _savefig(fig, "fig5_accuracy_vs_epsilon", out_dir)


# ---------------------------------------------------------------------------
# Fig 6: LiRA ROC (Exp 4)
# ---------------------------------------------------------------------------

def fig6_lira_roc(
    standard_roc: Dict,
    channeled_roc: Dict,
    out_dir: str = FIGURES_DIR,
):
    """LiRA ROC curves (log scale) for standard vs Channeled."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    ax.plot(standard_roc["fpr"], standard_roc["tpr"],
            label="Standard DP-SGD", color="tab:blue")
    ax.plot(channeled_roc["fpr"], channeled_roc["tpr"],
            label="Channeled DP-SGD (K=3)", color="tab:red")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")

    ax.set_xscale("log")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("LiRA ROC (log scale)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "fig6_lira_roc", out_dir)


# ---------------------------------------------------------------------------
# Fig 7: Per-tier TPR@FPR=1e-3 (Exp 4)
# ---------------------------------------------------------------------------

def fig7_per_tier_tpr(
    standard_tier_metrics: Dict,
    channeled_tier_metrics: Dict,
    out_dir: str = FIGURES_DIR,
):
    """Bar chart: per-tier TPR@FPR=1e-3 under Channeled DP-SGD."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    tiers = [0, 1, 2]
    x = np.arange(len(tiers))
    width = 0.35

    tpr_std = [standard_tier_metrics.get(k, {}).get("tpr_at_fpr_1e-03", 0.0)
               for k in tiers]
    tpr_chan = [channeled_tier_metrics.get(k, {}).get("tpr_at_fpr_1e-03", 0.0)
                for k in tiers]

    ax.bar(x - width / 2, tpr_std, width, label="Standard", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, tpr_chan, width, label="Channeled K=3", color="tab:red", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Tier {k}" for k in tiers])
    ax.set_ylabel(r"TPR @ FPR=$10^{-3}$", fontsize=10)
    ax.set_title("Per-tier MIA vulnerability", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, "fig7_per_tier_tpr", out_dir)


# ---------------------------------------------------------------------------
# Fig 8: Accuracy vs K (Exp 5.1)
# ---------------------------------------------------------------------------

def fig8_accuracy_vs_K(tab8: Dict, out_dir: str = FIGURES_DIR):
    """Bar chart: test accuracy vs number of tiers K."""
    fig, ax = plt.subplots(figsize=(6, 4))

    K_vals = sorted(tab8.keys())
    means = [tab8[K]["mean"] for K in K_vals]
    stds = [tab8[K]["std"] for K in K_vals]

    ax.bar([str(K) for K in K_vals], means, yerr=stds, capsize=4,
           color="tab:blue", alpha=0.8)
    ax.set_xlabel("Number of tiers K", fontsize=11)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title("CIFAR-10-LT (IR=50), ε=3: accuracy vs K", fontsize=11)
    ax.set_ylim(bottom=max(0, min(means) - 0.05))
    fig.tight_layout()
    _savefig(fig, "fig8_accuracy_vs_K", out_dir)


# ---------------------------------------------------------------------------
# Fig 9: Accuracy gap vs imbalance ratio (Exp 5.4)
# ---------------------------------------------------------------------------

def fig9_accuracy_gap_vs_IR(tab11: Dict, out_dir: str = FIGURES_DIR):
    """Line plot: accuracy gap (Channeled – Standard) vs IR."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    ir_vals = sorted(tab11.keys())
    gaps = [
        tab11[ir]["channeled"]["mean"] - tab11[ir]["standard"]["mean"]
        for ir in ir_vals
    ]

    ax.plot(ir_vals, gaps, "b-o", label="Channeled – Standard")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")

    ax.set_xscale("log")
    ax.set_xlabel("Imbalance ratio (IR)", fontsize=11)
    ax.set_ylabel("Accuracy gap (pp)", fontsize=11)
    ax.set_title("Accuracy improvement vs imbalance ratio", fontsize=11)
    ax.set_xticks(ir_vals)
    ax.set_xticklabels([str(int(ir)) for ir in ir_vals])
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, "fig9_accuracy_gap_vs_IR", out_dir)


# ---------------------------------------------------------------------------
# Fig 10: Per-instance privacy through training (Exp 6.1)
# ---------------------------------------------------------------------------

def fig10_privacy_through_training(fig10_data: Dict, out_dir: str = FIGURES_DIR):
    """Two-panel plot: per-step and cumulative ε vs epoch."""
    epochs = fig10_data["epochs"]
    # Downsample epoch axis to per-epoch (from per-step if needed)
    # Assume each entry is one epoch (T_actual epochs = EPOCHS in Exp 2)
    n_epochs = len(epochs)
    x = np.arange(1, n_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    ax1.plot(x, fig10_data["eps_head_per_step"], "b-", label="Head (tier 0)")
    ax1.plot(x, fig10_data["eps_tail_per_step"], "r-", label="Tail (tier 2)")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel(r"$\varepsilon_{\mathrm{step},t}$", fontsize=11)
    ax1.set_title("Per-step privacy cost", fontsize=11)
    ax1.legend(fontsize=9)

    ax2.plot(x, fig10_data["eps_head_cumulative"], "b-", label="Head (tier 0)")
    ax2.plot(x, fig10_data["eps_tail_cumulative"], "r-", label="Tail (tier 2)")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel(r"$\varepsilon_i(t)$ cumulative", fontsize=11)
    ax2.set_title("Cumulative privacy cost", fontsize=11)
    ax2.legend(fontsize=9)

    fig.suptitle("Per-instance privacy through training (head vs tail)", fontsize=12)
    fig.tight_layout()
    _savefig(fig, "fig10_privacy_through_training", out_dir)


# ---------------------------------------------------------------------------
# Fig 11: Box plot of ε_i by correct/incorrect × tier (Exp 6.2)
# ---------------------------------------------------------------------------

def fig11_correct_vs_incorrect(fig11_data: Dict, out_dir: str = FIGURES_DIR):
    """Box plot: 6 groups = (correct/incorrect) × (tier 0/1/2)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    group_order = []
    group_data = []
    group_colors = []
    group_labels = []

    for k in range(3):
        for label, color in [("correct", TIER_COLORS[k]),
                              ("incorrect", "lightgray")]:
            key = f"{label}_tier{k}"
            vals = fig11_data.get(key, [])
            if vals:
                group_data.append(vals)
                group_order.append(key)
                group_colors.append(color)
                group_labels.append(f"T{k} {'✓' if label == 'correct' else '✗'}")

    bp = ax.boxplot(group_data, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], group_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(group_labels) + 1))
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel(r"Per-instance $\varepsilon_i$", fontsize=11)
    ax.set_title(r"$\varepsilon_i$ by tier × correct/incorrect at convergence",
                 fontsize=11)
    fig.tight_layout()
    _savefig(fig, "fig11_correct_vs_incorrect", out_dir)


# ---------------------------------------------------------------------------
# Table formatting helpers
# ---------------------------------------------------------------------------

def format_table_accuracy(
    results_by_method: Dict,
    epsilons: List[float],
    methods_display: Dict[str, str],
) -> str:
    """Format accuracy table as LaTeX. Returns LaTeX string."""
    lines = []
    lines.append(r"\begin{tabular}{l" + "cc" * len(epsilons) + "}")
    lines.append(r"\toprule")
    header = "Method"
    for eps in epsilons:
        header += f" & \\multicolumn{{2}}{{c}}{{$\\varepsilon={eps:.0f}$}}"
    lines.append(header + r" \\")
    subheader = ""
    for _ in epsilons:
        subheader += " & Mean & Std"
    lines.append(subheader + r" \\")
    lines.append(r"\midrule")

    for method_key, method_label in methods_display.items():
        row = method_label
        for eps in epsilons:
            eps_key = f"eps{eps:.0f}"
            entry = results_by_method.get(eps_key, {}).get(method_key, {})
            mean = entry.get("mean", float("nan"))
            std = entry.get("std", float("nan"))
            row += f" & {mean:.4f} & {std:.4f}"
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def generate_all_figures(
    exp1_results=None,
    exp2_results=None,
    exp3_results=None,
    exp4_results=None,
    exp5_results=None,
    exp6_results=None,
    out_dir: str = FIGURES_DIR,
):
    """Generate all figures from experiment results."""
    import json

    def _load_json(path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    # Fig 1, 2, Tab 1 (Exp 1)
    if exp1_results:
        try:
            fig1_gradient_norm_histograms(exp1_results, out_dir)
            fig2_gradnorm_vs_epoch(exp1_results, out_dir)
        except Exception as e:
            print(f"  [Warning] Fig 1/2 error: {e}")

    # Fig 3, 4, Tab 2 (Exp 2)
    if exp2_results:
        try:
            fig3_scaling_law(exp2_results, out_dir)
            fig4_predictor_scatter(exp2_results, out_dir)
        except Exception as e:
            print(f"  [Warning] Fig 3/4 error: {e}")

    # Fig 5 (Exp 3)
    if exp3_results:
        try:
            c10_summary = exp3_results.get("cifar10_IR1", {})
            c10lt_summary = exp3_results.get("cifar10_IR50", {})
            if c10_summary and c10lt_summary:
                fig5_accuracy_vs_epsilon(c10_summary, c10lt_summary, out_dir)
        except Exception as e:
            print(f"  [Warning] Fig 5 error: {e}")

    # Fig 6, 7 (Exp 4)
    if exp4_results:
        try:
            std_r = exp4_results.get("standard", {})
            chan_r = exp4_results.get("channeled", {})
            if std_r and chan_r:
                fig6_lira_roc(
                    {"fpr": std_r.get("roc_fpr", []), "tpr": std_r.get("roc_tpr", [])},
                    {"fpr": chan_r.get("roc_fpr", []), "tpr": chan_r.get("roc_tpr", [])},
                    out_dir,
                )
                fig7_per_tier_tpr(
                    std_r.get("tier_metrics", {}),
                    chan_r.get("tier_metrics", {}),
                    out_dir,
                )
        except Exception as e:
            print(f"  [Warning] Fig 6/7 error: {e}")

    # Fig 8, 9 (Exp 5)
    if exp5_results:
        tab8, _, _, tab11, _ = exp5_results
        try:
            if tab8:
                fig8_accuracy_vs_K(tab8, out_dir)
        except Exception as e:
            print(f"  [Warning] Fig 8 error: {e}")
        try:
            if tab11:
                fig9_accuracy_gap_vs_IR(tab11, out_dir)
        except Exception as e:
            print(f"  [Warning] Fig 9 error: {e}")

    # Fig 10, 11 (Exp 6)
    if exp6_results:
        try:
            fig10_data = exp6_results.get("fig10") or \
                         _load_json("results/exp6/fig10_data.json")
            if fig10_data:
                fig10_privacy_through_training(fig10_data, out_dir)
        except Exception as e:
            print(f"  [Warning] Fig 10 error: {e}")
        try:
            fig11_data = exp6_results.get("fig11") or \
                         _load_json("results/exp6/fig11_data.json")
            if fig11_data:
                fig11_correct_vs_incorrect(fig11_data, out_dir)
        except Exception as e:
            print(f"  [Warning] Fig 11 error: {e}")

    print(f"\nAll figures saved to {out_dir}")
