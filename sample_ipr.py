#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive sampling for Pairwise IPR using a shifted-tanh logistic classifier
with fixed normalization and per-iteration debugging artifacts.

Mapping (important):
- x1 := resofach
- x2 := lookahead_time
The rest of the code still stores/handles raw samples as [lookahead_time, resofach].
Normalization functions remap columns so the model sees (x1, x2) as above.

What’s new vs. the previous script:
- For every active-learning iteration, we save:
  * a decision-boundary PNG (probability field + p=0.5 contour),
  * a CSV snapshot of all labeled samples so far, with a flag for those
    picked in the current iteration.
- An initial "iter 000 init" snapshot is saved before the first addition.

Artifacts live under:  <out-dir>/per_iter[-tag]/boundary_XXX_*.png and samples_XXX_*.csv

This file is self-contained (besides `pairwise_ipr_simulator`) and supports:
- Non-adaptive sampling (Sobol/LHS/Halton/Random/Grid).
- A single adaptive run with (n_init, batch_size, iterations).
- A budget search over (start_head, batch_size) combos that exactly fit --budget.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from datetime import datetime, timezone
import argparse
import numpy as np
from typing import Tuple, Iterable, Optional, List, Dict
from functools import lru_cache
from tqdm import tqdm
from joblib import dump as joblib_dump  # kept for compatibility
import multiprocessing
from scipy.stats.qmc import Sobol, LatinHypercube, Halton

# Plotting enabled for per-iteration artifacts
import matplotlib.pyplot as plt

from pairwise_ipr_simulator import PairwiseIPRSimulator

###############################################
# Shifted-tanh logistic (learning core)
###############################################

class ShiftedTanhLogitModel:
    def __init__(self, a, be, ga, w1, w2, b, sh_n,
                 x1_mu, x1_sd, x2_mu, x2_sd, best_bce, acc: float = 0.0):
        # Core parameters
        self.a = float(a); self.be = float(be); self.ga = float(ga)
        self.w1 = float(w1); self.w2 = float(w2); self.b = float(b)
        self.sh_n = float(sh_n)
        # Normalization stats
        self.x1_mu = float(x1_mu); self.x1_sd = float(x1_sd)  # resofach
        self.x2_mu = float(x2_mu); self.x2_sd = float(x2_sd)  # lookahead_time
        # Logs
        self.best_bce = float(best_bce)
        self.acc = float(acc)

    @staticmethod
    def _sigmoid(u: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-u))

    def _z_norm(self, X: np.ndarray) -> np.ndarray:
        """
        Map raw [lookahead_time, resofach] -> normalized [x1, x2]
        with x1 := resofach, x2 := lookahead_time.
        """
        X = np.asarray(X, float)
        Xn = X.copy()
        # x1 (col 0 of normalized) pulls from resofach (col 1 of raw)
        Xn[:, 0] = (X[:, 1] - self.x1_mu) / (self.x1_sd + 1e-12)
        # x2 (col 1 of normalized) pulls from lookahead_time (col 0 of raw)
        Xn[:, 1] = (X[:, 0] - self.x2_mu) / (self.x2_sd + 1e-12)
        return Xn

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        Xn = self._z_norm(X_raw)
        x1 = Xn[:, 0:1]  # resofach (normalized)
        x2 = Xn[:, 1:2]  # lookahead_time (normalized)
        s  = self.be * (x1 - self.sh_n)
        z  = self.a * np.tanh(s) + self.ga
        return self._sigmoid(self.w1 * z + self.w2 * x2 + self.b).ravel()

    def predict(self, X_raw: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X_raw) >= threshold).astype(int)

def _train_on_normalized(
    Xn: np.ndarray, y: np.ndarray,
    *, steps: int = 20_000, lr0: float = 0.05, decay: float = 5e-5,
    clip: float = 5.0, patience: int = 4000, seed: int = 1
):
    rng = np.random.RandomState(seed)
    x1_t = Xn[:, 0:1]
    x2_t = Xn[:, 1:2]
    y_t  = y.reshape(-1, 1)

    sh  = np.median(x1_t)
    a   = 0.5 + 0.1 * rng.randn()
    be  = 1.5 + 0.05 * rng.randn()
    ga  = 0.0 + 0.05 * rng.randn()
    w1  = 1.0
    w2  = 0.5
    b   = 0.0

    best_loss = np.inf
    since_improve = 0
    best_params = None
    eps = 1e-9

    for t in range(steps):
        s   = be * (x1_t - sh)
        th  = np.tanh(s)
        z   = a * th + ga
        lin = w1 * z + w2 * x2_t + b
        p   = 1.0 / (1.0 + np.exp(-lin))

        loss = -(y_t * np.log(p + eps) + (1 - y_t) * np.log(1 - p + eps)).mean()
        d = (p - y_t) / len(y_t)

        grad_w1 = np.sum(d * z)
        grad_w2 = np.sum(d * x2_t)
        grad_b  = np.sum(d)

        d_z   = d * w1
        sech2 = 1.0 - th**2
        grad_a  = np.sum(d_z * th)
        grad_ga = np.sum(d_z)
        grad_be = np.sum(d_z * a * sech2 * (x1_t - sh))
        grad_sh = np.sum(d_z * a * sech2 * (-be))

        grads = np.array([grad_a, grad_be, grad_ga, grad_w1, grad_w2, grad_b, grad_sh], float)
        nrm = np.linalg.norm(grads)
        if nrm > clip:
            grads *= clip / (nrm + 1e-12)
        grad_a, grad_be, grad_ga, grad_w1, grad_w2, grad_b, grad_sh = grads

        lr = lr0 / (1 + decay * t)
        a  -= lr * grad_a
        be -= lr * grad_be
        ga -= lr * grad_ga
        w1 -= lr * grad_w1
        w2 -= lr * grad_w2
        b  -= lr * grad_b
        sh -= lr * grad_sh

        if loss + 1e-10 < best_loss:
            best_loss   = float(loss)
            best_params = (float(a), float(be), float(ga), float(w1), float(w2), float(b), float(sh))
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= patience and best_params is not None:
                a, be, ga, w1, w2, b, sh = best_params
                break

    return best_params, float(best_loss)


def _fit_with_fixed_stats(
    X_raw: np.ndarray, y: np.ndarray,
    *, stats: Tuple[float, float, float, float],
    steps: int = 20_000, lr0: float = 0.05, decay: float = 5e-5, clip: float = 5.0, patience: int = 4000,
    seed: int = 42
) -> ShiftedTanhLogitModel:
    """
    stats = (x1_mu, x1_sd, x2_mu, x2_sd) where
      x1 := resofach (raw col 1)
      x2 := lookahead_time (raw col 0)
    """
    x1_mu, x1_sd, x2_mu, x2_sd = stats
    Xn = X_raw.copy().astype(float)
    # Normalize to [x1, x2] order used by the learner
    Xn[:, 0] = (X_raw[:, 1] - x1_mu) / x1_sd       # x1 from resofach
    Xn[:, 1] = (X_raw[:, 0] - x2_mu) / x2_sd       # x2 from lookahead

    params, best_loss = _train_on_normalized(
        Xn, y.astype(float), steps=steps, lr0=lr0, decay=decay, clip=clip, patience=patience, seed=seed
    )
    a, be, ga, w1, w2, b, sh = params
    return ShiftedTanhLogitModel(a, be, ga, w1, w2, b, sh, x1_mu, x1_sd, x2_mu, x2_sd, best_loss)


def _global_stats_from_pool(X_pool: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute stats for normalization with the x1/x2 mapping:
      x1 stats from resofach (raw col 1)
      x2 stats from lookahead_time (raw col 0)
    """
    x1_mu = float(X_pool[:, 1].mean()); x1_sd = float(X_pool[:, 1].std(ddof=0) + 1e-12)
    x2_mu = float(X_pool[:, 0].mean()); x2_sd = float(X_pool[:, 0].std(ddof=0) + 1e-12)
    return x1_mu, x1_sd, x2_mu, x2_sd

###############################################
# Problem setup
###############################################

# Detect number of available CPUs
cpu_count = multiprocessing.cpu_count()

# this is hardcoded to have best performance in my machines
# Define logic for n_jobs
if cpu_count >= 100:
    n_jobs = 100
elif cpu_count >= 31:
    n_jobs = 20
elif cpu_count >= 6:
    n_jobs = 3
else:
    n_jobs = int(max(1, cpu_count // 2))  # reasonable fallback

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
    n_jobs=n_jobs,
)

IPR_MIN = 0.999

LOOKAHEAD_LOW, LOOKAHEAD_HIGH = 1, 90     # integer
RESOFACH_LOW, RESOFACH_HIGH = 0.90, 1.20  # real

###############################################
# Simulator wrapper + utilities
###############################################

@lru_cache(maxsize=1024)
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

###############################################
# Per-iteration artifact helpers
###############################################

def _save_iter_artifacts(
    *, save_dir: Optional[Path],
    iter_name: str,
    model: ShiftedTanhLogitModel,
    X_pool: np.ndarray,
    labeled_idx: List[int],
    added_idx: List[int],
    y_labeled: np.ndarray,
    grid_res: int = 200,
):
    """Save a decision-boundary PNG and a CSV of samples for this iteration."""
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSV snapshot of labeled samples so far ----
    # Columns: pool_index, lookahead_time, resofach, label, picked_this_iter
    added_set = set(added_idx)
    rows = []
    for j, i_pool in enumerate(labeled_idx):
        la, rf = X_pool[i_pool]
        rows.append({
            "pool_index": int(i_pool),
            "lookahead_time": int(la),
            "resofach": float(rf),
            "label": int(y_labeled[j]),
            "picked_this_iter": bool(i_pool in added_set),
        })
    csv_path = save_dir / f"samples_{iter_name}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- Decision boundary plot ----
    la_lin = np.linspace(LOOKAHEAD_LOW, LOOKAHEAD_HIGH, grid_res)
    rf_lin = np.linspace(RESOFACH_LOW, RESOFACH_HIGH, grid_res)
    LA, RF = np.meshgrid(la_lin, rf_lin, indexing="ij")
    grid = np.column_stack([LA.ravel(), RF.ravel()])  # raw: [lookahead, resofach]
    P = model.predict_proba(grid).reshape(LA.shape)

    X_labeled = X_pool[labeled_idx]
    added_mask = np.array([idx in added_set for idx in labeled_idx], dtype=bool)
    prev_mask = ~added_mask

    plt.figure(figsize=(7.2, 6.2))
    plt.contour(RF, LA, P, levels=[0.5], linewidths=2)

    if prev_mask.any():
        plt.scatter(X_labeled[prev_mask, 1], X_labeled[prev_mask, 0],
                    s=26, marker="o", label="labeled (prev)")
    if added_mask.any():
        plt.scatter(X_labeled[added_mask, 1], X_labeled[added_mask, 0],
                    s=48, marker="^", label="added (this iter)")

    plt.xlabel("resofach")
    plt.ylabel("lookahead_time")
    plt.title(f"Decision boundary — {iter_name}")
    plt.legend(loc="best")
    plt.tight_layout()
    png_path = save_dir / f"boundary_{iter_name}.png"
    plt.savefig(png_path, dpi=150)
    plt.close()

###############################################
# Active refinement (eval_one labels, fixed normalization)
###############################################

def _build_sobol_pool(sobol_pool_size: int, seed: int) -> np.ndarray:
    eng = Sobol(d=2, scramble=True, seed=seed)
    U = eng.random_base2(int(np.ceil(np.log2(max(2, sobol_pool_size)))))
    U = U[:sobol_pool_size]
    la_pool, rf_pool = _scale_unit_to_space(U[:, 0], U[:, 1])
    X_pool = np.column_stack([la_pool.astype(float), rf_pool.astype(float)])
    return X_pool

def active_refine_fixednorm_evalone(
    *, X_pool: np.ndarray, n_init: int, batch_size: int, iters: int, seed: int,
    steps: int = 20_000, lr0: float = 0.05, decay: float = 5e-5, clip: float = 5.0, patience: int = 4000,
    save_debug_dir: Optional[Path] = "./debug/", grid_res: int = 200
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Modified policy:
      - Minimum labeled samples = 96
      - Preferentially sample within the first 1024 pool indices by uncertainty (|p-0.5|);
        top-up sequentially beyond 1024 if needed.
      - Stop when train accuracy >= 0.85 and at least 96 samples, or when reaching 128 samples.
      - Save per-iteration boundary + samples to save_debug_dir if provided.
    """
    MIN_SAMPLES = 64
    MAX_SAMPLES = 128
    PRIMARY_BLOCK = 1024
    ACC_TARGET = 0.9

    rng = np.random.default_rng(seed)

    # Fixed normalization from the *full pool*
    stats = _global_stats_from_pool(X_pool)

    # Initial labeled set: first n_init from pool
    init_k = min(n_init, len(X_pool))
    labeled_idx = list(range(init_k))

    def _label_points(idxs: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
        X = X_pool[list(idxs)]
        labs = []
        for la, rf in X:
            total_ipr, _ = _eval_one(int(la), float(rf))
            labs.append(1.0 if total_ipr >= IPR_MIN else 0.0)
        return X, np.asarray(labs, float)

    X_labeled, y_labeled = _label_points(labeled_idx)

    # Save initial snapshot
    model_init = _fit_with_fixed_stats(
        X_labeled, y_labeled, stats=stats,
        steps=steps, lr0=lr0, decay=decay, clip=clip, patience=patience, seed=seed
    )
    _save_iter_artifacts(
        save_dir=save_debug_dir, iter_name="000_init",
        model=model_init, X_pool=X_pool, labeled_idx=labeled_idx, added_idx=[],
        y_labeled=y_labeled, grid_res=grid_res
    )

    history: List[Dict] = []
    it = 0

    while True:
        # Train on current labeled set
        model = _fit_with_fixed_stats(
            X_labeled, y_labeled, stats=stats,
            steps=steps, lr0=lr0, decay=decay, clip=clip, patience=patience, seed=seed
        )
        yhat = model.predict(X_labeled)
        acc_now = float((yhat == y_labeled.astype(int)).mean()) if len(y_labeled) else 0.0

        # Stopping conditions
        if len(X_labeled) >= MIN_SAMPLES and acc_now >= ACC_TARGET:
            history.append({
                "iter": it,
                "added_indices": [],
                "train_size": int(len(X_labeled)),
                "train_acc": acc_now,
                "bce": float(model.best_bce),
                "feasible_found": int(np.sum(y_labeled)),
                "note": "Reached accuracy target"
            })
            break

        if len(X_labeled) >= MAX_SAMPLES:
            history.append({
                "iter": it,
                "added_indices": [],
                "train_size": int(len(X_labeled)),
                "train_acc": acc_now,
                "bce": float(model.best_bce),
                "feasible_found": int(np.sum(y_labeled)),
                "note": "Reached MAX_SAMPLES"
            })
            break

        if len(labeled_idx) == len(X_pool):
            history.append({
                "iter": it,
                "added_indices": [],
                "train_size": int(len(X_labeled)),
                "train_acc": acc_now,
                "bce": float(model.best_bce),
                "feasible_found": int(np.sum(y_labeled)),
                "note": "Pool exhausted"
            })
            break

        # # Batch size this round
        # k = min(batch_size, MAX_SAMPLES - len(X_labeled))

        # primary_unlabeled = [i for i in range(min(PRIMARY_BLOCK, len(X_pool))) if i not in labeled_idx]
        # pick: List[int] = []

        # def _balanced_pick(unlabeled_idxs: List[int], want: int) -> List[int]:
        #     """
        #     Picks ~half explore (closest to p=0.5) and ~half exploit.
        #     EXPLORE: by uncertainty (smallest |p-0.5|).
        #     EXPLOIT: by *index order* within unlabeled_idxs, skipping ones already taken.
        #     """
        #     if not unlabeled_idxs or want <= 0:
        #         return []

        #     proba = model.predict_proba(X_pool[unlabeled_idxs])
        #     dist = np.abs(proba - 0.5)

        #     want = min(want, len(unlabeled_idxs))
        #     k_explore = want // 2
        #     k_exploit = want - k_explore

        #     # Explore set: closest to boundary (by uncertainty)
        #     order_by_uncert = np.argsort(dist)  # ascending
        #     explore_local = list(order_by_uncert[:k_explore])

        #     # Exploit set: in *index order* within unlabeled_idxs, skipping explore picks
        #     chosen_mask = set(explore_local)
        #     exploit_local = []
        #     for j in range(len(unlabeled_idxs)):
        #         if j not in chosen_mask:
        #             exploit_local.append(j)
        #             if len(exploit_local) >= k_exploit:
        #                 break

        #     chosen_local = explore_local + exploit_local
        #     return [unlabeled_idxs[j] for j in chosen_local]

        # # Preferentially pick inside the primary block with balanced explore/exploit
        # if primary_unlabeled:
        #     want_primary = min(k, len(primary_unlabeled))
        #     pick.extend(_balanced_pick(primary_unlabeled, want_primary))

        # # If still short, top up from secondary block (balanced as well)
        # if len(pick) < k:
        #     secondary_unlabeled = [
        #         i for i in range(PRIMARY_BLOCK, len(X_pool))
        #         if i not in labeled_idx and i not in pick
        #     ]
        #     want_secondary = k - len(pick)
        #     pick.extend(_balanced_pick(secondary_unlabeled, want_secondary))

        # Batch size this round
        k = min(batch_size, MAX_SAMPLES - len(X_labeled))

        # Build primary/secondary pools of *indices*
        primary_unlabeled = [i for i in range(min(PRIMARY_BLOCK, len(X_pool))) if i not in labeled_idx]
        secondary_unlabeled = [
            i for i in range(PRIMARY_BLOCK, len(X_pool))
            if i not in labeled_idx
        ]

        pick: List[int] = []

        def pick_by_uncertainty(cands: List[int], want: int) -> List[int]:
            if not cands or want <= 0:
                return []
            proba = model.predict_proba(X_pool[cands])
            order = np.argsort(np.abs(proba - 0.5))  # ascending uncertainty distance
            order = order[:min(want, len(cands))]
            return [cands[j] for j in order]

        # Prefer primary block; top-up from secondary
        want_primary = min(k, len(primary_unlabeled))
        pick.extend(pick_by_uncertainty(primary_unlabeled, want_primary))

        if len(pick) < k and secondary_unlabeled:
            pick.extend(pick_by_uncertainty(secondary_unlabeled, k - len(pick)))

        if not pick:
            history.append({
                "iter": it,
                "added_indices": [],
                "train_size": int(len(X_labeled)),
                "train_acc": acc_now,
                "bce": float(model.best_bce),
                "feasible_found": int(np.sum(y_labeled)),
                "note": "No candidates to add"
            })
            break

        # Label and update
        X_new, y_new = _label_points(pick)
        X_labeled = np.vstack([X_labeled, X_new])
        y_labeled = np.concatenate([y_labeled, y_new])
        labeled_idx.extend(pick)

        # Retrain for logging and artifact saving "after" this addition
        model_post = _fit_with_fixed_stats(
            X_labeled, y_labeled, stats=stats,
            steps=steps, lr0=lr0, decay=decay, clip=clip, patience=patience, seed=seed
        )
        yhat_post = model_post.predict(X_labeled)
        acc_post = float((yhat_post == y_labeled.astype(int)).mean()) if len(y_labeled) else 0.0

        # Save "after" snapshot
        _save_iter_artifacts(
            save_dir=save_debug_dir,
            iter_name=f"{it:03d}_after",
            model=model_post,
            X_pool=X_pool,
            labeled_idx=labeled_idx,
            added_idx=pick,
            y_labeled=y_labeled,
            grid_res=grid_res,
        )

        history.append({
            "iter": it,
            "added_indices": pick,
            "train_size": int(len(X_labeled)),
            "train_acc": acc_post,
            "bce": float(model_post.best_bce),
            "feasible_found": int(np.sum(y_labeled)),
        })

        if len(X_labeled) >= MIN_SAMPLES and acc_post >= ACC_TARGET:
            break

        it += 1

    la_arr = X_labeled[:, 0].astype(int)
    rf_arr = X_labeled[:, 1].astype(float)
    return la_arr, rf_arr, history


def fine_tune_to_budget_evalone(
    *, X_pool: np.ndarray, n_init: int, batch_size: int, iters: int, seed: int,
    steps: int = 20_000, lr0: float = 0.05, decay: float = 5e-5, clip: float = 5.0, patience: int = 4000,
    save_debug_dir: Optional[Path] = None, grid_res: int = 200
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    # The policy loop already enforces min/max; `iters` is kept for compatibility.
    la_arr, rf_arr, history = active_refine_fixednorm_evalone(
        X_pool=X_pool, n_init=n_init, batch_size=batch_size, iters=iters, seed=seed,
        steps=steps, lr0=lr0, decay=decay, clip=clip, patience=patience,
        save_debug_dir=save_debug_dir, grid_res=grid_res
    )
    return la_arr, rf_arr, history

###############################################
# Config search (budget) — objective helper
###############################################

def _objective_for_config(eval_log: List[Dict]) -> Tuple[int, float, float]:
    """Return a tuple to maximize: (feasible_count, -best_mean_abs_xtrack, total_ipr_of_best)"""
    feas = [r for r in eval_log if r["feasible"]]
    if not feas:
        best_feas_xtrack = float("inf")
        best_feas_ipr = 0.0
    else:
        best = min(feas, key=lambda r: r["mean_mean_abs_xtrack"])
        best_feas_xtrack = float(best["mean_mean_abs_xtrack"])
        best_feas_ipr = float(best["total_ipr"])
    return (len(feas), -best_feas_xtrack, best_feas_ipr)

###############################################
# CLI
###############################################

def main():
    parser = argparse.ArgumentParser(description="Adaptive / non-adaptive sampling for Pairwise IPR with shifted-tanh active refinement.")
    parser.add_argument("--out-dir", type=str, default="sample_runs")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")

    parser.add_argument("--method", type=str, default="sobol",
                        choices=["sobol", "lhs", "halton", "random", "grid"],
                        help="Sampling method when not using --adaptive")

    parser.add_argument("--adaptive", action="store_true", help="Use active (margin) sampling with fixed normalization")

    # Budgeted adaptive config
    parser.add_argument("--budget", type=int, default=96, help="Total labeling budget (n_init + batch_size * iterations)")
    parser.add_argument("--n-init", type=int, default=32, help="Seed size for adaptive run")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for adaptive run")
    parser.add_argument("--iterations", type=int, default=16, help="Iterations for adaptive run")

    # Config search toggles (kept for compatibility; not used directly here)
    parser.add_argument("--search-configs", action="store_true", help="Search (start_head, batch_size, iters) under --budget")
    parser.add_argument("--start-heads", type=str, default="16,32,64", help="CSV of start_head candidates")
    parser.add_argument("--batch-sizes", type=str, default="4,8", help="CSV of batch_size candidates")

    # Non-adaptive controls
    parser.add_argument("--n-samples", type=int, default=96, help="Number of samples (ignored for grid/adaptive)")
    parser.add_argument("--grid-lookahead", type=int, default=41)
    parser.add_argument("--grid-resofach", type=int, default=26)

    # Pool and seeds
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sobol-pool-size", type=int, default=1024, help="Total Sobol candidates to build the adaptive pool")

    # Overrides for simulator
    parser.add_argument("--dpsi", type=float, default=None)
    parser.add_argument("--reception-prob", type=float, default=None)
    parser.add_argument("--pos-uncertainty", type=float, default=None)
    parser.add_argument("--vel-uncertainty", type=float, default=None)

    # Per-iteration artifact controls
    parser.add_argument("--plot-grid-res", type=int, default=200, help="Resolution of the boundary plot grid (per axis)")

    args = parser.parse_args()

    # Apply overrides
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

    tag_suffix = f"-{args.tag}" if args.tag else ""

    # --- Adaptive path ---
    if args.adaptive:
        # Single adaptive run with explicit (n_init, batch_size, iterations)
        assert args.n_init + args.batch_size * args.iterations == args.budget, \
            "Budget must equal n_init + batch_size * iterations."

        X_pool = _build_sobol_pool(args.sobol_pool_size, args.seed)

        # Where to save per-iteration artifacts
        debug_dir = out_dir / f"per_iter{tag_suffix}"
        debug_dir.mkdir(parents=True, exist_ok=True)

        la_arr, rf_arr, history = fine_tune_to_budget_evalone(
            X_pool=X_pool, n_init=args.n_init, batch_size=args.batch_size, iters=args.iterations, seed=args.seed,
            save_debug_dir=debug_dir, grid_res=args.plot_grid_res
        )

        # Evaluate labeled set in order
        n = len(la_arr)
        eval_log = []
        metrics = np.zeros((n, 2), dtype=float)
        feasibles = []

        with tqdm(total=n, desc="Sampling", dynamic_ncols=True) as pbar:
            for i, (la, rf) in enumerate(zip(la_arr, rf_arr)):
                total_ipr, mma = _eval_one(int(la), float(rf))
                feasible = bool(total_ipr >= IPR_MIN)
                eval_log.append({
                    "idx": i,
                    "lookahead_time": int(la),
                    "resofach": float(rf),
                    "total_ipr": float(total_ipr),
                    "mean_mean_abs_xtrack": float(mma),
                    "feasible": feasible,
                })
                metrics[i, 0] = mma
                metrics[i, 1] = total_ipr
                if feasible:
                    feasibles.append(eval_log[-1])
                pbar.set_postfix(idx=i, lookahead=la, resofach=f"{rf:.4f}", IPR=f"{total_ipr:.6f}", feasible=feasible)
                pbar.update(1)

        if feasibles:
            best_feas = min(feasibles, key=lambda r: r["mean_mean_abs_xtrack"])
        else:
            best_feas = max(eval_log, key=lambda r: (r["total_ipr"], -r["mean_mean_abs_xtrack"]))

        # Save artifacts
        csv_path = out_dir / f"eval_log{tag_suffix}.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=eval_log[0].keys())
            w.writeheader(); w.writerows(eval_log)

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
                "method": "adaptive",
                "budget": args.budget,
                "n": int(n),
                "n_init": int(args.n_init),
                "batch_size": int(args.batch_size),
                "iterations": int(args.iterations),
                "sobol_pool_size": int(args.sobol_pool_size),
                "seed": int(args.seed),
            },
            "sim_common_kw": _to_py(COMMON_KW),
            "best": best_feas,
            "history": history,
            "files": {
                "csv": str(csv_path),
                "metrics_npy": str(npy_path),
                "per_iter_dir": str(debug_dir),
            },
        }
        with json_path.open("w") as f:
            json.dump(payload, f, indent=2)

        # Print summary
        print("\n=== Best (feasible preferred) ===")
        print(f"lookahead_time: {best_feas['lookahead_time']}")
        print(f"resofach      : {best_feas['resofach']:.4f}")
        print(f"IPR (>= {IPR_MIN}): {best_feas['total_ipr']:.6f} "
              f"({'OK' if best_feas['feasible'] else 'FAIL'})")
        print(f"mean(mean_abs_xtrack): {best_feas['mean_mean_abs_xtrack']:.3f}")
        print(f"\nSaved: {json_path}")
        print(f"Saved: {csv_path}")
        print(f"Saved: {npy_path}")
        print(f"Per-iteration artifacts in: {debug_dir}")
        return

    # --- Non-adaptive path ---
    if args.method == "grid":
        la_arr, rf_arr = _make_grid(args.grid_lookahead, args.grid_resofach)
    else:
        la_arr, rf_arr = _make_samples(args.method, args.n_samples, args.seed)

    n = len(la_arr)
    print(f"Evaluating {n} samples using {args.method}...")

    eval_log = []
    metrics = np.zeros((n, 2), dtype=float)

    with tqdm(total=n, desc="Sampling", dynamic_ncols=True) as pbar:
        for i, (la, rf) in enumerate(zip(la_arr, rf_arr)):
            total_ipr, mma = _eval_one(int(la), float(rf))
            feasible = bool(total_ipr >= IPR_MIN)
            eval_log.append({
                "idx": i,
                "lookahead_time": int(la),
                "resofach": float(rf),
                "total_ipr": float(total_ipr),
                "mean_mean_abs_xtrack": float(mma),
                "feasible": feasible,
            })
            metrics[i, 0] = mma
            metrics[i, 1] = total_ipr
            pbar.set_postfix(idx=i, lookahead=la, resofach=f"{rf:.4f}", IPR=f"{total_ipr:.6f}", feasible=feasible)
            pbar.update(1)

    feasibles = [r for r in eval_log if r["feasible"]]
    if feasibles:
        best = min(feasibles, key=lambda r: r["mean_mean_abs_xtrack"])
    else:
        best = max(eval_log, key=lambda r: (r["total_ipr"], -r["mean_mean_abs_xtrack"]))

    # Save artifacts
    csv_path = out_dir / f"eval_log{tag_suffix}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=eval_log[0].keys())
        w.writeheader(); w.writerows(eval_log)

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
            "method": args.method,
            "n_samples": n,
            "seed": args.seed,
        },
        "sim_common_kw": _to_py(COMMON_KW),
        "best": best,
        "files": {"csv": str(csv_path), "metrics_npy": str(npy_path)},
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    # Print summary
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