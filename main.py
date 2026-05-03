#!/usr/bin/env python3
"""
main.py — Direction-Aware Per-Instance DP Accounting

Flags-driven interface for training models with per-instance gradient logging
and computing direction-aware privacy certificates.

Modes
-----
  train    train model with DP-SGD + per-instance gradient accounting
  certify  compute norm-based and direction-aware certificates from saved logs
  both     train then certify in one shot
  table    print summary table over all certified runs in --results_dir

Examples
--------
  # CLIP linear probe, CIFAR-10, eps=8
  python main.py --mode train --dataset cifar10 --regime R3 --eps 8 \\
                 --batch 5000 --seed 0 --K 1 --gpu 0

  # Certify the above run
  python main.py --mode certify --dataset cifar10 --regime R3 --eps 8 \\
                 --batch 5000 --seed 0

  # Train + certify in one shot
  python main.py --mode both --dataset cifar10 --regime R3 --eps 8 \\
                 --batch 5000 --seed 0 --K 1 --gpu 0

  # WRN-28-2 cold start, CIFAR-10-LT (IR=50), eps=8, sparse accounting
  python main.py --mode train --dataset cifar10_lt50 --regime R1 --eps 8 \\
                 --batch 5000 --seed 0 --K 8 --gpu 0

  # Summary table for everything under results/runs/
  python main.py --mode table
"""

import csv
import json
import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.exp_p17_train import train_run
from experiments.exp_p17_certify import (
    ALPHA_GRID,
    build_summary_dict,
    compute_beta_spectrum,
    compute_certificates,
    data_independent_eps,
    data_independent_eps_quad,
    load_meta,
    load_stats,
    plot_beta_spectrum,
    plot_certs,
    plot_eigenvalue_history,
    run_sanity_checks,
    save_certs_csv,
    summarize,
)


# ---------------------------------------------------------------------------
# Tag helpers
# ---------------------------------------------------------------------------

def make_tag(name: str, dataset: str, regime: str, eps: float, seed: int,
             mech: str = "vanilla") -> str:
    return f"p17_{name}_{mech}_{dataset}_{regime}_eps{eps:.0f}_seed{seed}"


# ---------------------------------------------------------------------------
# Certify one run
# ---------------------------------------------------------------------------

def certify_one(tag: str, cfg: dict, seed: int,
                log_dir: str, cert_dir: str, train_dir: str) -> None:
    cert_path = os.path.join(cert_dir, f"{tag}_certs.csv")
    summ_path = os.path.join(cert_dir, f"{tag}_summary.json")

    meta  = load_meta(tag, log_dir)
    stats = load_stats(tag, log_dir)
    if meta is None or stats is None:
        print(f"[certify] Stats/meta missing for {tag}. Run --mode train first.")
        return

    eps        = float(meta["eps"])
    delta      = float(meta["delta"])
    sigma_mult = float(meta["sigma_mult"])
    q          = float(meta["q"])
    T_steps    = int(meta["T_steps"])
    n_priv     = int(meta["n_priv"])
    tier_arr   = meta["tier_labels"]
    tier_arr   = tier_arr if len(tier_arr) > 0 else None

    print(f"\n[certify] {tag}")
    print(f"  n={n_priv}  T={T_steps}  q={q:.5f}  eps={eps}  delta={delta:.2e}")
    print(f"  n_accounted={int(stats['n_accounted'])}  K={int(meta.get('K', 1))}")

    eps_di,      alpha_di = data_independent_eps(sigma_mult, q, T_steps, delta)
    eps_di_quad, _        = data_independent_eps_quad(
        float(meta["sigma_use"]), q, T_steps, delta)
    print(f"  eps_di={eps_di:.4f}  eps_di_quad={eps_di_quad:.4f}  target={eps}")

    certs  = compute_certificates(stats, meta, delta)
    is_r3  = cfg["regime"] == "R3"
    betas  = compute_beta_spectrum(stats, meta, delta) if is_r3 else {}

    ok = run_sanity_checks(certs, eps_di, eps_di_quad, eps, tag)
    if not ok:
        print("  [WARN] Sanity check(s) failed — results still saved.")

    os.makedirs(cert_dir, exist_ok=True)
    save_certs_csv(certs, tier_arr, cert_path)
    summarize(tag, certs, delta, eps, tier_arr)

    summ = build_summary_dict(
        certs, eps_di, alpha_di, tier_arr, betas if is_r3 else None)
    summ["eps_di_quad"] = float(eps_di_quad)
    summ.update({
        "tag":      tag,
        "seed":     seed,
        "sanity_ok": ok,
        "regime":   cfg["regime"],
        "mech":     cfg.get("mech", "vanilla"),
        "dataset":  cfg["dataset"],
        "eps":      float(cfg["eps"]),
    })

    # Load accuracy from train CSV
    csv_path = os.path.join(train_dir, f"{tag}.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            if rows and "test_acc" in rows[0]:
                summ["final_acc"] = float(rows[-1]["test_acc"])
                summ["best_acc"]  = max(float(r["test_acc"]) for r in rows)
        except Exception:
            pass

    with open(summ_path, "w") as fh:
        json.dump(summ, fh, indent=2)
    print(f"  [cert] Summary saved: {summ_path}")

    plot_certs(tag, seed, certs, eps, tier_arr, cert_dir)
    plot_eigenvalue_history(tag, seed, stats, cert_dir)
    if is_r3 and betas:
        plot_beta_spectrum(tag, seed, betas, cert_dir)


# ---------------------------------------------------------------------------
# Table mode
# ---------------------------------------------------------------------------

def print_results_table(results_dir: str) -> None:
    summaries = []
    for root, _, files in os.walk(results_dir):
        for fname in files:
            if fname.endswith("_summary.json"):
                path = os.path.join(root, fname)
                try:
                    with open(path) as fh:
                        summaries.append(json.load(fh))
                except Exception:
                    pass

    if not summaries:
        print(f"[table] No summary files found under {results_dir}")
        return

    col_tag = 50
    hdr = (f"{'Tag':{col_tag}s}  {'eps':>4}  {'regime':>6}  {'dataset':>14}  "
           f"{'eps^norm':>10}  {'eps^dir':>9}  {'tight%':>7}  {'best_acc':>8}")
    sep = "=" * (len(hdr) + 2)
    print(f"\n{sep}")
    print("  Per-Instance Certificate Summary")
    print(sep)
    print(f"  {hdr}")
    print("  " + "-" * len(hdr))

    def _f(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) and not np.isnan(float(v)) else "  N/A  "

    for d in sorted(summaries, key=lambda x: x.get("tag", "")):
        tag    = d.get("tag", "?")[:col_tag]
        eps    = d.get("eps", float("nan"))
        regime = d.get("regime", "?")
        ds     = d.get("dataset", "?")
        en     = d.get("eps_norm_mean", float("nan"))
        ed     = d.get("eps_dir_mean",  float("nan"))
        tight  = (1.0 - ed / en) * 100 if (en and en > 0) else float("nan")
        acc    = d.get("best_acc", float("nan"))
        print(f"  {tag:{col_tag}s}  {eps:>4.0f}  {regime:>6}  {ds:>14}  "
              f"{_f(en):>10}  {_f(ed):>9}  "
              f"{(f'{tight:.1f}%' if not np.isnan(tight) else '  N/A  '):>7}  "
              f"{_f(acc):>8}")

    print(sep)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Direction-Aware Per-Instance DP Accounting — train and certify",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--mode", required=True,
                        choices=["train", "certify", "both", "table"],
                        help="Operation mode (required)")

    # Core run parameters (required for train/certify/both)
    core = parser.add_argument_group("run parameters (required for train/certify/both)")
    core.add_argument("--dataset",
                      choices=["cifar10", "cifar10_lt50", "cifar10_lt100", "emnist"],
                      help="Dataset")
    core.add_argument("--regime", choices=["R1", "R2", "R3"],
                      help="R1=cold start (WRN-28-2), R2=warm start (WRN-28-2), "
                           "R3=CLIP ViT-B/32 linear probe")
    core.add_argument("--eps", type=float,
                      help="Target privacy budget ε  (e.g. 1, 2, 4, 8)")
    core.add_argument("--batch", type=int,
                      help="Training batch size (e.g. 5000 for CIFAR, 10000 for EMNIST)")
    core.add_argument("--seed", type=int,
                      help="Random seed for reproducibility")

    # Training options
    train_g = parser.add_argument_group("training options")
    train_g.add_argument("--K", type=int, default=1,
                         help="Accounting interval — log every K-th step. "
                              "K=1 for R3 (CLIP), K=8 for R1/R2. (default: 1)")
    train_g.add_argument("--gpu", type=int, default=0,
                         help="Primary GPU index (default: 0)")
    train_g.add_argument("--extra_gpus", type=int, nargs="*", default=None,
                         help="Additional GPU indices for algebra passes (R1/R2 only). "
                              "E.g. --extra_gpus 1 2")

    # Path options
    path_g = parser.add_argument_group("paths")
    path_g.add_argument("--name", type=str, default="run",
                        help="Run name used as output sub-directory and file prefix "
                             "(default: run)")
    path_g.add_argument("--data_root", type=str, default="./data",
                        help="Dataset root directory (default: ./data)")
    path_g.add_argument("--cache_dir", type=str, default="./data/clip_features",
                        help="CLIP feature cache directory (default: ./data/clip_features)")
    path_g.add_argument("--results_dir", type=str, default="./results/runs",
                        help="Base results directory; outputs go to "
                             "<results_dir>/<name>/{train,logs,checkpoints,certs}/ "
                             "(default: ./results/runs)")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Validate required args for non-table modes
    if args.mode in ("train", "certify", "both"):
        missing = [f for f in ("dataset", "regime", "eps", "batch", "seed")
                   if getattr(args, f) is None]
        if missing:
            parser.error(
                f"--mode {args.mode} requires: "
                + ", ".join(f"--{m}" for m in missing)
            )

    # Table mode: just scan results_dir
    if args.mode == "table":
        print_results_table(args.results_dir)
        return

    # Output layout: <results_dir>/<name>/
    base_dir  = os.path.join(args.results_dir, args.name)
    train_dir = os.path.join(base_dir, "train")
    log_dir   = os.path.join(base_dir, "logs")
    ckpt_dir  = os.path.join(base_dir, "checkpoints")
    cert_dir  = os.path.join(base_dir, "certs")
    for d in (train_dir, log_dir, ckpt_dir, cert_dir):
        os.makedirs(d, exist_ok=True)

    # Config dict (mirrors RUN_MATRIX schema expected by train_run)
    cfg = {
        "dataset": args.dataset,
        "regime":  args.regime,
        "mech":    "vanilla",
        "eps":     args.eps,
        "batch":   args.batch,
        "n_seeds": 1,
        "K":       args.K,
    }

    tag = make_tag(args.name, args.dataset, args.regime, args.eps, args.seed)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    extra_devices = (
        [torch.device(f"cuda:{g}") for g in args.extra_gpus]
        if args.extra_gpus else None
    )

    print(f"[main] mode={args.mode}  tag={tag}")
    print(f"[main] device={device}"
          + (f"  extra_gpus={args.extra_gpus}" if extra_devices else ""))
    print(f"[main] output dir: {base_dir}")

    if args.mode in ("train", "both"):
        print(f"\n[main] === Training ===")
        train_run(
            run_id=args.name,
            cfg=cfg,
            seed=args.seed,
            device=device,
            data_root=args.data_root,
            cache_dir=args.cache_dir,
            out_dir=train_dir,
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
            K_override=args.K,
            extra_devices=extra_devices,
        )

    if args.mode in ("certify", "both"):
        print(f"\n[main] === Certify ===")
        certify_one(tag, cfg, args.seed, log_dir, cert_dir, train_dir)

    print(f"\n[main] Done. All outputs in: {base_dir}")


if __name__ == "__main__":
    main()
