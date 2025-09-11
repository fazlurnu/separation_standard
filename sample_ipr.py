#!/usr/bin/env python3
# pip install scipy tqdm joblib scikit-learn

import json
import csv
from pathlib import Path
from datetime import datetime, timezone
import argparse
import numpy as np
from typing import Tuple
from functools import lru_cache
from tqdm import tqdm
from joblib import dump as joblib_dump

from scipy.stats.qmc import Sobol, LatinHypercube, Halton

# For adaptive sampling
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pairwise_ipr_simulator import PairwiseIPRSimulator

# --- Fixed settings (tweak as needed) ---
COMMON_KW = dict(
    pos_uncertainty_sigma=15,
    vel_uncertainty_sigma=1.5,
    reception_prob=0.95,
    init_speed_intruder=20,
    dpsi=4,
    max_tr=15,
    max_dtr2=10,
    width=10,
    height=10,
    nb_of_repetition=100,
    n_jobs=3,
)

# Constraint
IPR_MIN = 0.999

# --- Search space ---
LOOKAHEAD_LOW, LOOKAHEAD_HIGH = 5, 90     # integer
RESOFACH_LOW, RESOFACH_HIGH = 0.925, 1.20  # real


# Optional: cache repeated evaluations
@lru_cache(maxsize=512)
def _eval_one(lookahead_time: int, resofach: float) -> Tuple[float, float]:
    sim = PairwiseIPRSimulator(
        lookahead_time=int(lookahead_time),
        resofach=float(resofach),
        **COMMON_KW,
    )
    total_ipr, distance_cpa, xtrack_area_norm_arr = sim.compute()
    mean_mean_abs_xtrack = (
        float(np.mean(xtrack_area_norm_arr)) if xtrack_area_norm_arr.size else float("inf")
    )
    return total_ipr, mean_mean_abs_xtrack


def _to_py(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [_to_py(x) for x in o]
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    return o


def _scale_unit_to_space(u_lookahead: np.ndarray, u_resofach: np.ndarray):
    """Map [0,1] samples to the problem space."""
    lookahead = LOOKAHEAD_LOW + u_lookahead * (LOOKAHEAD_HIGH - LOOKAHEAD_LOW)
    lookahead = np.rint(lookahead).astype(int)
    lookahead = np.clip(lookahead, LOOKAHEAD_LOW, LOOKAHEAD_HIGH)

    resofach = RESOFACH_LOW + u_resofach * (RESOFACH_HIGH - RESOFACH_LOW)
    resofach = np.clip(resofach, RESOFACH_LOW, RESOFACH_HIGH)
    return lookahead, resofach


def _make_samples(method: str, n_samples: int, seed: int):
    method = method.lower()
    if method == "sobol":
        eng = Sobol(d=2, scramble=True, seed=seed)
        U = eng.random_base2(int(np.ceil(np.log2(max(2, n_samples)))))
        U = U[:n_samples]
    elif method == "lhs":
        eng = LatinHypercube(d=2, seed=seed)
        U = eng.random(n_samples)
    elif method == "halton":
        eng = Halton(d=2, scramble=True, seed=seed)
        U = eng.random(n_samples)
    elif method == "random":
        rng = np.random.default_rng(seed)
        U = rng.random((n_samples, 2))
    else:
        raise ValueError(f"Unknown method: {method}")

    la, rf = _scale_unit_to_space(U[:, 0], U[:, 1])
    return la, rf


def _make_grid(n_lookahead: int, n_resofach: int):
    la = np.linspace(LOOKAHEAD_LOW, LOOKAHEAD_HIGH, n_lookahead)
    la = np.rint(la).astype(int)
    la = np.unique(np.clip(la, LOOKAHEAD_LOW, LOOKAHEAD_HIGH))
    rf = np.linspace(RESOFACH_LOW, RESOFACH_HIGH, n_resofach)
    LA, RF = np.meshgrid(la, rf, indexing="ij")
    return LA.ravel(), RF.ravel()


def _make_adaptive_samples(n_init: int, batch_size: int, iterations: int, seed: int):
    """Margin-based adaptive sampling."""
    rng = np.random.default_rng(seed)
    total_budget = n_init + batch_size * iterations

    # Candidate pool
    n_pool = 5000
    U_pool = rng.random((n_pool, 2))
    la_pool, rf_pool = _scale_unit_to_space(U_pool[:, 0], U_pool[:, 1])
    X_pool = np.column_stack([la_pool, rf_pool])

    # Init set
    init_idx = rng.choice(n_pool, size=n_init, replace=False)
    labeled_idx = set(init_idx.tolist())

    X_labeled = X_pool[list(labeled_idx)]
    y_labeled = []
    for la, rf in X_labeled:
        total_ipr, _ = _eval_one(int(la), float(rf))
        y_labeled.append(1 if total_ipr >= IPR_MIN else -1)
    y_labeled = np.array(y_labeled)

    clf = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=3, C=5.0, coef0=1.0))

    for _ in range(iterations):
        clf.fit(X_labeled, y_labeled)
        unlabeled = [i for i in range(n_pool) if i not in labeled_idx]
        if not unlabeled:
            break
        scores = clf.decision_function(X_pool[unlabeled])
        pick = np.argsort(np.abs(scores))[:batch_size]
        pick_idx = [unlabeled[i] for i in pick]

        new_X, new_y = [], []
        for idx in pick_idx:
            la, rf = X_pool[idx]
            total_ipr, _ = _eval_one(int(la), float(rf))
            new_X.append([la, rf])
            new_y.append(1 if total_ipr >= IPR_MIN else -1)
        new_X = np.array(new_X)
        new_y = np.array(new_y)

        X_labeled = np.vstack([X_labeled, new_X])
        y_labeled = np.concatenate([y_labeled, new_y])
        labeled_idx.update(pick_idx)

        if len(X_labeled) >= total_budget:
            break

    la_arr = X_labeled[:, 0].astype(int)
    rf_arr = X_labeled[:, 1].astype(float)
    return la_arr, rf_arr


def main():
    parser = argparse.ArgumentParser(description="Sampling (not BO) for Pairwise IPR.")
    parser.add_argument("--out-dir", type=str, default="sample_runs")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")
    parser.add_argument("--method", type=str, default="sobol",
                        choices=["sobol", "lhs", "halton", "random", "grid"],
                        help="Sampling method")
    parser.add_argument("--adaptive", action="store_true",
                        help="Use adaptive (margin-based) sampling")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for adaptive sampling")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of adaptive iterations")
    parser.add_argument("--n-samples", type=int, default=96,
                        help="Number of samples (ignored for grid/adaptive)")
    parser.add_argument("--grid-lookahead", type=int, default=41)
    parser.add_argument("--grid-resofach", type=int, default=26)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpsi", type=int, default=None)
    parser.add_argument("--reception-prob", type=float, default=None)
    parser.add_argument("--pos-uncertainty", type=float, default=None)
    parser.add_argument("--vel-uncertainty", type=float, default=None)
    args = parser.parse_args()

    # Override COMMON_KW
    if args.dpsi is not None:
        COMMON_KW["dpsi"] = int(args.dpsi)
    if args.reception_prob is not None:
        COMMON_KW["reception_prob"] = float(args.reception_prob)
    if args.pos_uncertainty is not None:
        COMMON_KW["pos_uncertainty_sigma"] = float(args.pos_uncertainty)
    if args.vel_uncertainty is not None:
        COMMON_KW["vel_uncertainty_sigma"] = float(args.vel_uncertainty)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Sample selection ---
    if args.adaptive:
        la_arr, rf_arr = _make_adaptive_samples(
            n_init=16, batch_size=args.batch_size,
            iterations=args.iterations, seed=args.seed
        )
    elif args.method == "grid":
        la_arr, rf_arr = _make_grid(args.grid_lookahead, args.grid_resofach)
    else:
        la_arr, rf_arr = _make_samples(args.method, args.n_samples, args.seed)

    n = len(la_arr)
    print(f"Evaluating {n} samples using "
          f"{'adaptive' if args.adaptive else args.method}...")

    # --- Evaluate ---
    eval_log = []
    metrics = np.zeros((n, 2), dtype=float)

    with tqdm(total=n, desc="Sampling", dynamic_ncols=True) as pbar:
        for i, (la, rf) in enumerate(zip(la_arr, rf_arr)):
            total_ipr, mean_mean_abs_xtrack = _eval_one(int(la), float(rf))
            feasible = bool(total_ipr >= IPR_MIN)
            eval_log.append({
                "idx": i,
                "lookahead_time": int(la),
                "resofach": float(rf),
                "total_ipr": float(total_ipr),
                "mean_mean_abs_xtrack": float(mean_mean_abs_xtrack),
                "feasible": feasible,
            })
            metrics[i, 0] = mean_mean_abs_xtrack
            metrics[i, 1] = total_ipr

            pbar.set_postfix(idx=i, lookahead=la, resofach=f"{rf:.4f}",
                             IPR=f"{total_ipr:.6f}", feasible=feasible)
            pbar.update(1)

    # --- Pick best ---
    feasibles = [r for r in eval_log if r["feasible"]]
    if feasibles:
        best = min(feasibles, key=lambda r: r["mean_mean_abs_xtrack"])
    else:
        best = max(eval_log, key=lambda r: (r["total_ipr"], -r["mean_mean_abs_xtrack"]))

    # --- Save artifacts ---
    tag_suffix = f"-{args.tag}" if args.tag else ""
    csv_path = out_dir / f"eval_log{tag_suffix}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=eval_log[0].keys())
        w.writeheader()
        w.writerows(eval_log)

    npy_path = out_dir / f"metrics{tag_suffix}.npy"
    np.save(npy_path, metrics)

    json_path = out_dir / f"summary{tag_suffix}.json"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "constraint": {"IPR_MIN": IPR_MIN},
        "search_space": {
            "lookahead_time": {"type": "Integer", "low": LOOKAHEAD_LOW, "high": LOOKAHEAD_HIGH},
            "resofach": {"type": "Real", "low": RESOFACH_LOW, "high": RESOFACH_HIGH},
        },
        "sampling": {
            "method": "adaptive" if args.adaptive else args.method,
            "n_samples": n,
            "batch_size": args.batch_size if args.adaptive else None,
            "iterations": args.iterations if args.adaptive else None,
            "seed": args.seed,
        },
        "sim_common_kw": _to_py(COMMON_KW),
        "best": best,
        "files": {"csv": str(csv_path), "metrics_npy": str(npy_path)},
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    # --- Print summary ---
    print("\n=== Best (feasible preferred) ===")
    print(f"lookahead_time: {best['lookahead_time']}")
    print(f"resofach      : {best['resofach']:.4f}")
    print(f"IPR (>= {IPR_MIN}): {best['total_ipr']:.6f} "
          f"({'OK' if best['feasible'] else 'FAIL'})")
    print(f"mean(mean_abs_xtrack): {best['mean_mean_abs_xtrack']:.3f}")
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {npy_path}")


if __name__ == "__main__":
    main()
