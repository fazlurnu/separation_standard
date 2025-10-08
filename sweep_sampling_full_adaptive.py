#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep an IPR sampling script over:
- dpsi: FIXED_DPSI
- reception_prob: RECEP_LIST
- pos_uncertainty_sigma: POS_LIST
- vel_uncertainty_sigma: VEL_LIST

Produces (under --out-root, default ./sampling_sweep_full_adaptive):
- summary.csv
- all_evals.csv
- combined_metrics.npy

Supports an additional sweep method "adaptive" using the shifted-tanh logistic
sampler in `adaptive_sampling_tanh.py`. When --method adaptive is used, this
driver forwards the exact flags you’d run manually, e.g.:

    python adaptive_sampling_tanh.py \
      --adaptive \
      --budget 96 \
      --n-init 32 \
      --batch-size 4 \
      --iterations 1 \
      --sobol-pool-size 1024 \
      --plot-grid-res 200

Usage examples:
  python sweep_sampling_full_adaptive.py --script adaptive_sampling_tanh.py --method sobol --n-samples 96
  python sweep_sampling_full_adaptive.py --script adaptive_sampling_tanh.py --method adaptive
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
import argparse
import json
import csv
import numpy as np
from itertools import product
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# You can change these lists as needed
FIXED_DPSI   = [24, 26, 28, 30, 36, 72, 4]
RECEP_LIST   = [0.95]      # reception_prob
POS_LIST     = [15.0]      # pos_uncertainty_sigma
VEL_LIST     = [1.5]       # vel_uncertainty_sigma


def _tagify(dpsi, rprob, pos, vel):
    r = int(round(rprob * 100))      # 0.90 -> 90, 0.95 -> 95
    p = int(round(pos * 10))         # 1.5 -> 15, 5.0 -> 50, 15.0 -> 150
    v = int(round(vel * 10))         # 0.5 -> 5, 1.5 -> 15
    return f"dpsi{dpsi:03d}_r{r:03d}_pos{p:03d}_vel{v:03d}"


def run_one(python_exe, script_path, out_root, dpsi, rprob, pos, vel,
            tag, method, n_samples, grid_la, grid_rf, seed, resume=False,
            *,
            # --- Adaptive passthrough knobs ---
            n_init=32, batch_size=4, iterations=16,
            sobol_pool_size=1024, plot_grid_res=200):
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_json = out_dir / f"summary-{tag}.json"
    eval_csv     = out_dir / f"eval_log-{tag}.csv"
    metrics_npy  = out_dir / f"metrics-{tag}.npy"

    if resume and summary_json.exists() and eval_csv.exists() and metrics_npy.exists():
        return summary_json, eval_csv, metrics_npy, None

    # Build command to call the sampling script (e.g., adaptive_sampling_tanh.py)
    cmd = [
        python_exe, str(script_path),
        "--out-dir", str(out_dir),
        "--tag", tag,
        "--dpsi", str(dpsi),
        "--reception-prob", str(rprob),
        "--pos-uncertainty", str(pos),
        "--vel-uncertainty", str(vel),
        "--seed", str(seed),
    ]

    # Branching by method. For adaptive, we forward the plotting & pool options too.
    if method == "grid":
        cmd += ["--method", "grid",
                "--grid-lookahead", str(grid_la),
                "--grid-resofach", str(grid_rf)]
    elif method == "adaptive":
        budget = n_init + batch_size * iterations
        cmd += [
            "--adaptive",
            "--budget", str(budget),
            "--n-init", str(n_init),
            "--batch-size", str(batch_size),
            "--iterations", str(iterations),
            "--sobol-pool-size", str(sobol_pool_size),
            "--plot-grid-res", str(plot_grid_res),
        ]
    else:
        cmd += ["--method", method, "--n-samples", str(n_samples)]

    try:
        subprocess.run(cmd, check=True)
        return summary_json, eval_csv, metrics_npy, None
    except subprocess.CalledProcessError as e:
        return summary_json, eval_csv, metrics_npy, f"run failed: {e}"
    except FileNotFoundError as e:
        return summary_json, eval_csv, metrics_npy, f"python or script not found: {e}"


def main():
    ap = argparse.ArgumentParser(description="Full sweep driver for IPR sampling scripts")
    ap.add_argument("--script", type=Path, default=Path("adaptive_sampling_tanh.py"),
                    help="Path to the sampling script (default: ./adaptive_sampling_tanh.py)")
    ap.add_argument("--out-root", type=Path, default=Path("sampling_sweep_full_adaptive"),
                    help="Root directory for all runs (default: ./sampling_sweep_full_adaptive)")
    ap.add_argument("--python", type=str, default=sys.executable,
                    help="Python executable to use (default: current interpreter)")

    # We allow "adaptive" here to control branching; the underlying script
    # should accept --adaptive rather than a --method value of "adaptive".
    ap.add_argument("--method", type=str, default="sobol",
        choices=["sobol", "lhs", "halton", "random", "grid", "adaptive"],
        help="Sampling method for each run (default: sobol)")
    ap.add_argument("--n-samples", type=int, default=96,
                    help="Number of samples per combination (ignored for grid/adaptive)")
    ap.add_argument("--grid-lookahead", type=int, default=41,
                    help="Grid steps for lookahead_time (grid only)")
    ap.add_argument("--grid-resofach", type=int, default=26,
                    help="Grid steps for resofach (grid only)")
    ap.add_argument("--seed", type=int, default=42, help="Random/scramble seed")
    ap.add_argument("--resume", action="store_true",
                    help="Skip combinations whose artifacts already exist")

    # --- Adaptive passthrough options (plots & pool size included) ---
    ap.add_argument("--n-init", type=int, default=32,
                    help="Adaptive: initial labeled points")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Adaptive: points added per iteration")
    ap.add_argument("--iterations", type=int, default=16,
                    help="Adaptive: number of iterations")
    ap.add_argument("--sobol-pool-size", type=int, default=1024,
                    help="Adaptive: Sobol pool size")
    ap.add_argument("--plot-grid-res", type=int, default=200,
                    help="Adaptive: decision-boundary plot grid resolution per axis")

    args = ap.parse_args()

    args.out_root = args.out_root.expanduser().resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)

    combos = list(product(RECEP_LIST, POS_LIST, VEL_LIST, FIXED_DPSI))
    total = len(combos)

    summary_rows = []
    all_eval_rows = []
    combined_metrics = []

    with tqdm(total=total, desc="Sweeping dpsi×reception×pos×vel", dynamic_ncols=True) as pbar:
        for rprob, pos, vel, dpsi in combos:
            tag = _tagify(dpsi, rprob, pos, vel)

            summary_json, eval_csv, metrics_npy, err = run_one(
                args.python, args.script, args.out_root,
                dpsi, rprob, pos, vel,
                tag, args.method, args.n_samples, args.grid_lookahead, args.grid_resofach,
                args.seed, resume=args.resume,
                n_init=args.n_init, batch_size=args.batch_size, iterations=args.iterations,
                sobol_pool_size=args.sobol_pool_size, plot_grid_res=args.plot_grid_res
            )

            row = {
                "dpsi": dpsi,
                "reception_prob": rprob,
                "pos_uncertainty_sigma": pos,
                "vel_uncertainty_sigma": vel,
                "lookahead_time": "",
                "resofach": "",
                "total_ipr": "",
                "mean_mean_abs_xtrack": "",
                "feasible": "",
                "accuracy": "",
                "confusion_tn": "",
                "confusion_fp": "",
                "confusion_fn": "",
                "confusion_tp": "",
                "summary_json": str(summary_json),
                "eval_csv": str(eval_csv),
                "metrics_npy": str(metrics_npy),
                "error": err or "",
            }

            if err is None:
                # Read summary
                if summary_json.exists():
                    try:
                        with summary_json.open() as f:
                            payload = json.load(f)
                        best = payload.get("best", {})
                        row.update({
                            "lookahead_time": best.get("lookahead_time", ""),
                            "resofach": best.get("resofach", ""),
                            "total_ipr": best.get("total_ipr", ""),
                            "mean_mean_abs_xtrack": best.get("mean_mean_abs_xtrack", ""),
                            "feasible": best.get("feasible", ""),
                        })
                    except Exception as e:
                        row["error"] = f"read summary json failed: {e}"

                # Read eval log
                if eval_csv.exists():
                    try:
                        with eval_csv.open(newline="") as f:
                            reader = list(csv.DictReader(f))
                        y_true = [int(r["feasible"]) for r in reader]
                        cm = confusion_matrix(y_true, y_true, labels=[0, 1])  # trivial CM vs itself
                        tn, fp, fn, tp = cm.ravel()
                        acc = (tp + tn) / max(1, (tp + tn + fp + fn))

                        # Save diagnostics into JSON summary
                        if summary_json.exists():
                            try:
                                with summary_json.open() as f:
                                    payload = json.load(f)
                            except Exception:
                                payload = {}
                        else:
                            payload = {}
                        payload["accuracy"] = acc
                        payload["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
                        with summary_json.open("w") as f:
                            json.dump(payload, f, indent=2)

                        row["accuracy"] = acc
                        row["confusion_tn"] = int(tn)
                        row["confusion_fp"] = int(fp)
                        row["confusion_fn"] = int(fn)
                        row["confusion_tp"] = int(tp)

                        # Keep full eval rows
                        for r in reader:
                            r2 = dict(r)
                            r2["dpsi"] = dpsi
                            r2["reception_prob"] = rprob
                            r2["pos_uncertainty_sigma"] = pos
                            r2["vel_uncertainty_sigma"] = vel
                            all_eval_rows.append(r2)
                    except Exception as e:
                        row["error"] += f" | read eval csv failed: {e}"

                # Read metrics
                if metrics_npy.exists():
                    try:
                        m = np.load(metrics_npy)
                        if m.ndim != 2 or m.shape[1] != 2:
                            raise ValueError(f"metrics shape expected (N,2), got {m.shape}")
                        cols = [
                            np.full((m.shape[0], 1), float(dpsi)),
                            np.full((m.shape[0], 1), float(rprob)),
                            np.full((m.shape[0], 1), float(pos)),
                            np.full((m.shape[0], 1), float(vel)),
                            m[:, 0:1],  # mean_mean_abs_xtrack
                            m[:, 1:2],  # total_ipr
                        ]
                        combined_metrics.append(np.hstack(cols))
                    except Exception as e:
                        row["error"] += f" | read metrics npy failed: {e}"

            summary_rows.append(row)
            pbar.update(1)

    # Write summary.csv
    summary_csv = args.out_root / "summary.csv"
    fieldnames = [
        "dpsi", "reception_prob", "pos_uncertainty_sigma", "vel_uncertainty_sigma",
        "lookahead_time", "resofach", "total_ipr", "mean_mean_abs_xtrack",
        "feasible", "accuracy",
        "confusion_tn", "confusion_fp", "confusion_fn", "confusion_tp",
        "summary_json", "eval_csv", "metrics_npy", "error"
    ]
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    # Write all_evals.csv
    all_evals_csv = args.out_root / "all_evals.csv"
    if all_eval_rows:
        eval_fieldnames = [
            "dpsi", "reception_prob", "pos_uncertainty_sigma", "vel_uncertainty_sigma",
            "idx", "lookahead_time", "resofach", "total_ipr", "mean_mean_abs_xtrack", "feasible"
        ]
        with all_evals_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=eval_fieldnames)
            w.writeheader()
            w.writerows(all_eval_rows)

    # Write combined_metrics.npy
    if combined_metrics:
        M = np.vstack(combined_metrics)
        np.save(args.out_root / "combined_metrics.npy", M)

    print("\nSweep complete.")
    print(f"Best-per-combination summary: {summary_csv}")
    if all_eval_rows:
        print(f"All evaluations CSV:  {all_evals_csv}")
    if combined_metrics:
        print(f"Combined metrics NPY: {args.out_root / 'combined_metrics.npy'}")
    print(f"Per-run outputs under: {args.out_root}/<tag>/")
    print("If you used --method adaptive and your sampling script saves per-iteration plots,")
    print("check each run's folder for a 'per_iter-<tag>/' directory with boundary_*.png and samples_*.csv.")


if __name__ == "__main__":
    main()
