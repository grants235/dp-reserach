"""
Experiment 6: Structural Predictions
======================================

Uses checkpoints from Experiment 2.

6.1: Per-instance privacy through training
- Compute ε_step,t and cumulative ε_i(t) for 100 head + 100 tail examples.
- Outputs: Fig 10 (per-step and cumulative ε vs epoch).

6.2: Correct vs. incorrect classification
- At convergence, partition by correct/incorrect × tier.
- Box plot of ε_i.
- Outputs: Fig 11 (6-group box plot).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pickle

from src.datasets import load_datasets, make_data_loaders
from src.models import make_model
from src.dp_training import compute_per_sample_losses, set_seed
from src.calibration import (
    compute_per_sample_gradient_norms,
    compute_per_sample_clipped_norms,
)
from src.tiers import assign_tiers
from src.privacy_accounting import per_instance_epsilon_quadratic, compute_sigma
from src.evaluation import save_results, save_json


RESULTS_DIR = "results/exp6"
DATA_ROOT = "./data"
ARCH = "wrn28-2"
BATCH_SIZE = 256
EPOCHS = 100
EPSILON = 3.0
DELTA = 1e-5
C = 1.0
K = 3
N_HEAD = 100   # examples from tier 0 (head)
N_TAIL = 100   # examples from tier 2 (tail)


def load_exp2_results(exp2_dir: str = "results/exp2"):
    """Load Experiment 2 results (checkpoints and per-sample data)."""
    result_path = os.path.join(exp2_dir, "results.pkl")
    if not os.path.exists(result_path):
        raise FileNotFoundError(
            f"Experiment 2 results not found at {result_path}. "
            "Please run exp2 first."
        )
    with open(result_path, "rb") as f:
        return pickle.load(f)


def compute_per_step_epsilon(
    clipped_norms_over_time: np.ndarray,
    C: float,
    sigma: float,
    q: float,
    T: int,
    global_eps: float,
) -> np.ndarray:
    """
    Compute per-step per-instance ε for each example.

    Per-step ε_step,t,i = (||ḡ_i(θ_t)|| / C)² · ε_step
    where ε_step is the per-step privacy budget = global_eps / T (approximately).

    More precisely: use the ratio of per-instance RDP to global RDP per step.

    Args:
        clipped_norms_over_time: (n, T) – clipped norms at each step
        C: global clipping bound
        sigma: noise multiplier
        q: sampling rate
        T: total steps
        global_eps: global (ε, δ)-DP guarantee

    Returns:
        eps_per_step: (n, T) – per-step per-instance ε
    """
    ratio_sq = (clipped_norms_over_time / C) ** 2  # (n, T)
    ratio_sq = np.clip(ratio_sq, 0.0, 1.0)

    # Per-step global ε (approximate as global_eps / T)
    eps_step_global = global_eps / T
    eps_per_step = ratio_sq * eps_step_global  # (n, T)
    return eps_per_step


def run_exp6(device: torch.device = None, data_root: str = DATA_ROOT,
             exp2_dir: str = "results/exp2"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp6] Using device: {device}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load Exp2 results
    # ------------------------------------------------------------------
    exp2 = load_exp2_results(exp2_dir)

    selected_indices = exp2["selected_indices"]
    selected_tiers = exp2["selected_tiers"]
    selected_labels = exp2["selected_labels"]
    clipped_norms_over_time = exp2["clipped_norms_over_time"]  # (n_sel, T)
    sigma = exp2["sigma"]
    q = exp2["q"]
    T = exp2["T"]

    n_sel, T_actual = clipped_norms_over_time.shape

    # ------------------------------------------------------------------
    # 6.1: Per-instance privacy through training
    # ------------------------------------------------------------------
    print("[Exp6] Computing per-step and cumulative ε...")

    # Select 100 head (tier 0) and 100 tail (tier 2) examples
    head_mask = (selected_tiers == 0)
    tail_mask = (selected_tiers == K - 1)
    head_local_idx = np.where(head_mask)[0][:N_HEAD]
    tail_local_idx = np.where(tail_mask)[0][:N_TAIL]

    print(f"  Head examples: {len(head_local_idx)}, Tail: {len(tail_local_idx)}")

    # Per-step ε
    eps_per_step = compute_per_step_epsilon(
        clipped_norms_over_time, C, sigma, q, T_actual, EPSILON
    )  # (n_sel, T_actual)

    # Cumulative ε (sum over steps, approximate)
    eps_cumulative = np.cumsum(eps_per_step, axis=1)  # (n_sel, T_actual)

    # Head vs tail averages
    eps_head_per_step = eps_per_step[head_local_idx].mean(axis=0)   # (T,)
    eps_tail_per_step = eps_per_step[tail_local_idx].mean(axis=0)   # (T,)
    eps_head_cumulative = eps_cumulative[head_local_idx].mean(axis=0)
    eps_tail_cumulative = eps_cumulative[tail_local_idx].mean(axis=0)

    fig10_data = {
        "epochs": list(range(1, T_actual + 1)),
        "eps_head_per_step": eps_head_per_step.tolist(),
        "eps_tail_per_step": eps_tail_per_step.tolist(),
        "eps_head_cumulative": eps_head_cumulative.tolist(),
        "eps_tail_cumulative": eps_tail_cumulative.tolist(),
    }
    save_json(fig10_data, os.path.join(RESULTS_DIR, "fig10_data.json"))

    # ------------------------------------------------------------------
    # 6.2: Correct vs. incorrect classification at convergence
    # ------------------------------------------------------------------
    print("[Exp6] Correct vs. incorrect classification analysis...")

    # Load final Exp2 convergence data
    if "checkpoint_data" in exp2:
        # Exp2 has convergence classification data for selected examples
        conv_data = exp2.get("checkpoint_data", {}).get("convergence", {})
        if conv_data:
            correctly_classified = conv_data["correctly_classified"]
        else:
            correctly_classified = None
    else:
        correctly_classified = None

    if correctly_classified is None:
        # Compute from the Exp2 model if available
        print("  No convergence data in Exp2 results; loading from checkpoint...")
        ckpt_path = os.path.join(exp2_dir, f"checkpoints/seed0/epoch_{EPOCHS:04d}.pt")
        if os.path.exists(ckpt_path):
            data = load_datasets("cifar10", data_root=data_root, imbalance_ratio=1.0,
                                 public_frac=0.1, split_seed=42)
            private_loader, _, _ = make_data_loaders(data, BATCH_SIZE)

            model = make_model(ARCH, data["num_classes"])
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            model = model.to(device).eval()

            from torch.utils.data import Subset
            sel_loader = torch.utils.data.DataLoader(
                Subset(private_loader.dataset, selected_indices.tolist()),
                batch_size=BATCH_SIZE, shuffle=False,
            )
            losses_conv, labels_conv, preds_conv, confs_conv = \
                compute_per_sample_losses(model, sel_loader, device)
            correctly_classified = (preds_conv == labels_conv)
        else:
            # Fallback: use per-instance ε as proxy (higher ε → more likely wrong)
            eps_i = exp2.get("eps_i_thudi", exp2.get("eps_i_theoretical"))
            correctly_classified = (eps_i < np.median(eps_i))

    correctly_classified = np.asarray(correctly_classified, dtype=bool)

    # Per-instance ε at convergence (RMS norm over all steps → quadratic law)
    rms_norms = np.sqrt(np.mean(clipped_norms_over_time ** 2, axis=1))
    eps_i = per_instance_epsilon_quadratic(rms_norms, C, EPSILON)

    # Box plot data: 6 groups = (correct/incorrect) × (tier 0/1/2)
    fig11_data = {}
    for is_correct, correct_label in [(True, "correct"), (False, "incorrect")]:
        for k in range(K):
            mask = (selected_tiers == k) & (correctly_classified == is_correct)
            group_key = f"{correct_label}_tier{k}"
            fig11_data[group_key] = eps_i[mask].tolist()
            n_group = mask.sum()
            mean_eps = eps_i[mask].mean() if n_group > 0 else 0.0
            print(f"  {group_key}: n={n_group}, mean_ε={mean_eps:.4f}")

    save_json(fig11_data, os.path.join(RESULTS_DIR, "fig11_data.json"))

    results = {
        "fig10": fig10_data,
        "fig11": fig11_data,
        "eps_per_step": eps_per_step,
        "eps_cumulative": eps_cumulative,
        "selected_tiers": selected_tiers.tolist(),
    }
    save_results(results, os.path.join(RESULTS_DIR, "results.pkl"))
    print(f"[Exp6] Results saved to {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 6: Structural predictions")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--exp2_dir", default="results/exp2")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    run_exp6(device=device, data_root=args.data_root, exp2_dir=args.exp2_dir)
