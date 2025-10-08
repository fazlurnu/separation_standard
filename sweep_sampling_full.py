#!/usr/bin/env python3
"""
Sweep sample_ipr.py over:
- dpsi: FIXED_DPSI
- reception_prob: RECEP_LIST
- pos_uncertainty_sigma: POS_LIST
- vel_uncertainty_sigma: VEL_LIST

Produces (under --out-root, default ./sampling_sweep_full_adaptive):
- summary.csv
- all_evals.csv
- combined_metrics.npy

Supports an additional sweep method "adaptive":
  - Margin-based active learning (SVM with poly kernel)
  - Starts with 16 random points
  - Adds 4 points per iteration × 20 iterations
  - Total 96 samples

Usage examples:
  python sweep_sampling_full.py --script sample_ipr.py --method sobol --n-samples 96
  python sweep_sampling_full.py --script sample_ipr.py --method adaptive
"""

import sys
import subprocess
from pathlib import Path
import argparse
import json
import csv
import numpy as np
from itertools import product
from tqdm import tqdm

# You can change these lists as needed
FIXED_DPSI   = [2, 6, 24]
RECEP_LIST   = [0.95]      # reception_prob
POS_LIST     = [15.0]      # pos_uncertainty_sigma
VEL_LIST     = [1.5]       # vel_uncertainty_sigma


def _tagify(dpsi, rprob, pos, vel):
    r = int(round(rprob * 100))      # 0.90 -> 90, 0.95 -> 95
    p = int(round(pos * 10))         # 1.5 -> 15, 5.0 -> 50, 15.0 -> 150
    v = int(round(vel * 10))         # 0.5 -> 5, 1.5 -> 15
    return f"dpsi{dpsi:06.1f}_r{r:03d}_pos{p:03d}_vel{v:03d}"


def run_one(python_exe, script_path, out_root, dpsi, rprob, pos, vel,
            tag, method, n_samples, grid_la, grid_rf, seed, resume=False):
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_json = out_dir / f"summary-{tag}.json"
    eval_csv     = out_dir / f"eval_log-{tag}.csv"
    metrics_npy  = out_dir / f"metrics-{tag}.npy"

    if resume and summary_json.exists() and eval_csv.exists() and metrics_npy.exists():
        return summary_json, eval_csv, metrics_npy, None

    # Build command to call sample_ipr.py
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

    # Important: sample_ipr.py uses --adaptive flag to enable adaptive mode,
    # and its --method does NOT accept "adaptive".
    if method == "grid":
        cmd += ["--method", "grid",
                "--grid-lookahead", str(grid_la),
                "--grid-resofach", str(grid_rf)]
    elif method == "adaptive":
        # Do NOT pass --method here (or pass a benign one if you prefer, e.g., sobol).
        cmd += ["--adaptive", "--batch-size", "4", "--iterations", "16"]
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
    ap = argparse.ArgumentParser(description="Full sweep for sample_ipr.py (sampling, not BO)")
    ap.add_argument("--script", type=Path, default=Path("sample_ipr.py"),
                    help="Path to sample_ipr.py (default: ./sample_ipr.py)")
    ap.add_argument("--out-root", type=Path, default=Path("sampling_sweep_full_adaptive"),
                    help="Root directory for all runs (default: ./sampling_sweep_full_adaptive)")
    ap.add_argument("--python", type=str, default=sys.executable,
                    help="Python executable to use (default: current interpreter)")

    # We allow "adaptive" here in the sweep CLI to control branching, but
    # note sample_ipr.py itself does NOT accept --method adaptive.
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
                args.seed, resume=args.resume
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
                            reader = csv.DictReader(f)
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
        "feasible", "summary_json", "eval_csv", "metrics_npy", "error"
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


if __name__ == "__main__":
    main()
