# Direction-Aware Per-Instance Privacy Accounting for DP-SGD

**CPSC 5520 Final Project — Grant Shanklin, Yale University**

Paper: *Direction-Aware Per-Instance Privacy Accounting for DP-SGD*
Video: https://www.youtube.com/watch?v=fkinSyM-GOk

---

## Overview

DP-SGD reports the same (ε, δ) guarantee for every training example regardless of how much each example actually influenced the model.
Existing per-instance accountants (Yu et al. 2022; Thudi et al. 2024) refine this using each example's **clipped gradient norm**, but in deep networks the clipped norms are nearly uniform across the dataset — the per-instance bound is nearly flat even when membership-inference vulnerability varies by an order of magnitude across examples.

This project proposes using gradient **direction** instead of just magnitude.
Under a Gaussian aggregate-uncertainty adversary, the per-step Rényi distinguishability is a Mahalanobis quadratic form: gradient energy along directions where the batch aggregate varies naturally is **masked** by that variation.
A Woodbury low-rank approximation gives a computable conservative certificate:

```
d²_eff = ‖ḡ‖²/σ²C²  −  Σ_k  λ_k·(ḡᵀuₖ)² / (σ²C²·(σ²C² + λ_k))
```

where uₖ, λₖ are the top-r eigenpairs of the private gradient covariance Σ_t.

**Results on CIFAR-10 CLIP linear probe:**

| ε | ε^norm (med) | ε^dir (med) | Tightening |
|---|---|---|---|
| 1 | 0.628 | 0.458 | **30%** |
| 2 | 1.192 | 0.677 | **46%** |
| 8 | 4.066 | 1.548 | **63%** |

The mechanism and its (ε, δ)-DP guarantee are unchanged; this is a per-instance refinement of the accountant, not a new mechanism.

---

## Installation

```bash
pip install -r requirements.txt
```

**Required:** `torch>=2.0`, `torchvision>=0.15`, `opacus>=1.4`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`

For R3 (CLIP), also install one of:
```bash
pip install open_clip_torch   # preferred
# or: pip install git+https://github.com/openai/CLIP
```

---

## Data Setup

Download CIFAR-10 and pre-compute the fixed public/private splits (run once):

```bash
python setup_data.py --data_root ./data
```

This downloads CIFAR-10/100, constructs long-tailed subsets at IR ∈ {10, 50, 100}, and saves stratified 90/10 private/public splits with seed 42.

For EMNIST, torchvision downloads it automatically on first use.

For R3 (CLIP), features are extracted and cached automatically on the first training run at `./data/clip_features/`.

---

## Training a Model

Use `main.py --mode train`. All parameters are required — there are no defaults for the core experiment configuration.

```bash
python main.py --mode train \
    --dataset   cifar10     \   # cifar10 | cifar10_lt50 | cifar10_lt100 | emnist
    --regime    R3          \   # R1=cold start | R2=warm start | R3=CLIP linear probe
    --eps       8           \   # privacy budget (1 | 2 | 4 | 8)
    --batch     5000        \   # batch size
    --seed      0           \   # random seed
    --K         1           \   # accounting interval (K=1 for R3, K=8 for R1/R2)
    --gpu       0           \   # GPU index
    --name      myrun           # output sub-directory name (default: run)
```

Outputs are written to `./results/runs/<name>/`:
- `train/<tag>.csv` — per-epoch train loss and test accuracy
- `train/<tag>_final.pt` — final model weights
- `train/<tag>_best.pt` — best checkpoint by test accuracy
- `logs/<tag>_stats.npz` — accumulated gradient statistics (sum_norm2, sum_reduction_k, eigval_history)
- `logs/<tag>_meta.npz` — run hyperparameters (σ, q, T, K, tier labels, etc.)
- `checkpoints/<tag>_ckpt.pt` — latest checkpoint for resuming (deleted on completion)

The run can be interrupted and resumed: the checkpoint is loaded automatically.

### Regime reference

| Regime | Architecture | Initialization | Epochs |
|--------|-------------|----------------|--------|
| R1 | WRN-28-2 (GroupNorm) | Random | 60 (CIFAR), 30 (EMNIST) |
| R2 | WRN-28-2 (GroupNorm) | 50-ep public pre-train | 60 (CIFAR), 30 (EMNIST) |
| R3 | Linear head on CLIP ViT-B/32 | CLIP (frozen) | 100 |

### Accounting interval K

At every K-th training step, `main.py` computes all-example per-sample gradients,
extracts top-r eigenpairs of Σ_t, and accumulates the direction-aware statistics.
Non-accounted steps fall back to the data-independent (worst-case) bound.

- **R3 (CLIP, d=5130):** K=1 is affordable; full covariance fits on GPU.
- **R1/R2 (WRN, d≈270k):** Use K=8 or K=16 to reduce cost; randomized SVD is used.

---

## Certifying a Trained Model

After training completes, compute per-instance certificates:

```bash
python main.py --mode certify \
    --dataset  cifar10  \
    --regime   R3       \
    --eps      8        \
    --batch    5000     \
    --seed     0        \
    --name     myrun        # must match the --name used during training
```

Outputs are written to `./results/runs/<name>/certs/`:
- `<tag>_certs.csv` — per-example ε^norm, ε^dir, β (one row per training example)
- `<tag>_summary.json` — aggregate statistics (medians, tightening %, β, accuracy)
- `<tag>_scatter.png` — ε^dir vs ε^norm scatter plot colored by β
- `<tag>_hist.png` — histogram of per-instance certificates
- `<tag>_eigvals.png` — top eigenvalue evolution over training
- `<tag>_beta_spectrum.png` — β as a function of rank r (R3 only)

### Certificate fields

| Field | Meaning |
|-------|---------|
| `eps_norm` | Per-instance norm-based ε (Thudi et al. 2024) |
| `eps_direction` | Direction-aware ε (this work) — always ≤ `eps_norm` |
| `beta_mean` | d²_eff / d²_norm ∈ [0,1]; lower = more masking = tighter bound |
| `tier` | 0=head, 1=mid, 2=tail (long-tailed datasets only) |

---

## Train + Certify in One Shot

```bash
python main.py --mode both \
    --dataset cifar10 --regime R3 --eps 8 \
    --batch 5000 --seed 0 --K 1 --gpu 0 \
    --name myrun
```

---

## Summary Table

Print a summary across all certified runs under `--results_dir`:

```bash
python main.py --mode table
# or for a custom path:
python main.py --mode table --results_dir ./results/runs
```

---

## Reproducing Paper Results

The paper's headline table (Table 2) uses CLIP R3 on CIFAR-10 across ε ∈ {1, 2, 8}:

```bash
for eps in 1 2 8; do
    for seed in 0 1 2; do
        python main.py --mode both \
            --dataset cifar10 --regime R3 --eps $eps \
            --batch 5000 --seed $seed --K 1 --gpu 0 \
            --name "clip_c10_eps${eps}_s${seed}"
    done
done
python main.py --mode table
```

Long-tailed variants (Table 2, bottom rows):

```bash
for ir in lt50 lt100; do
    for seed in 0 1 2; do
        python main.py --mode both \
            --dataset cifar10_${ir} --regime R3 --eps 8 \
            --batch 5000 --seed $seed --K 1 --gpu 0 \
            --name "clip_${ir}_eps8_s${seed}"
    done
done
```

---

## Project Structure

```
final-project/
├── main.py                   # flags-driven train + certify interface (this file)
├── setup_data.py             # one-time data download and split setup
├── requirements.txt
├── src/
│   ├── datasets.py           # dataset helpers, public/private split
│   ├── models.py             # WRN-28-2, ResNet-20 (GroupNorm)
│   └── ...
├── experiments/
│   ├── exp_p17_train.py      # core training loop with per-instance gradient logging
│   └── exp_p17_certify.py    # certificate computation (norm-based + direction-aware)
└── results/
    └── runs/
        └── <name>/
            ├── train/        # CSVs, model weights
            ├── logs/         # gradient statistics, meta
            ├── checkpoints/  # resume checkpoints
            └── certs/        # certificates, plots, summaries
```

---

## Citation

```
Shanklin, G. (2026). Direction-Aware Per-Instance Privacy Accounting for DP-SGD.
Yale University CPSC 5520 Final Project.
```
