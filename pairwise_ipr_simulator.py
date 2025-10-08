# pairwise_ipr_simulator.py

import os
import math
import numpy as np
import pandas as pd

from joblib import Parallel, delayed, parallel_backend

from tqdm import tqdm
from typing import Tuple

import contextlib, io, sys
import bluesky as bs

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Temporarily suppress stdout/stderr (for noisy library init)."""
    save_stdout, save_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        yield
    finally:
        sys.stdout, sys.stderr = save_stdout, save_stderr

class ReportingParallel(Parallel):
    """Parallel that prints 'Done X out of Y' every `report_every` completed tasks."""
    def __init__(self, *args, report_every: int = 10, **kwargs):
        self._report_every = max(1, int(report_every))
        super().__init__(*args, **kwargs)

    def print_progress(self):
        # Called by joblib in the main process as tasks complete
        completed = getattr(self, "n_completed_tasks", 0)
        total = getattr(self, "total_tasks", None)

        if total is not None:
            if (completed % self._report_every == 0) or (completed == total):
                # Print a compact line (no joblib default spam)
                print(f"[Parallel] Done {completed} out of {total}")

        # Don't call super().print_progress() to avoid default verbose spam
        # (super() would print detailed timing lines every batch).
        return
    
# --- Cross-track helpers ------------------------------------------------------
EARTH_R = 6371000.0  # meters

from typing import Tuple  # ensure available here

def _xtrack_area_norm_all_ac(
    time_list, id_list, lat_list, lon_list, hdg_list,
    tmax: float,
    init_speed_ownship: float,
    init_speed_intruder: float,
) -> float:
    """
    Compute the normalized x-track area metric for *all* aircraft and average them.

    For each aircraft i:
      1) Build its own reference line from its first (lat, lon, hdg).
      2) Project positions to local ENU, compute along-track s_i and cross-track x_i (signed).
      3) Integrate area_i = ∫ |x_i(s)| ds (trapezoid).
      4) Normalize by L0_i = v0_i * tmax, where v0_0=init_speed_ownship, v0_1=init_speed_intruder.
    Return the mean over aircraft: mean_i (area_i / L0_i).  Units: meters.
    """
    import numpy as _np

    n_ac = len(id_list[0])
    vals = []

    for ac_idx in range(n_ac):
        lat = _np.array([row[ac_idx] for row in lat_list], dtype=float)
        lon = _np.array([row[ac_idx] for row in lon_list], dtype=float)
        hdg = _np.array([row[ac_idx] for row in hdg_list], dtype=float)

        # Reference from aircraft's own first position & heading
        lat0, lon0 = float(lat[0]), float(lon[0])
        hdg0 = _np.deg2rad(float(hdg[0]))

        # Local EN projection
        lat0r = _np.deg2rad(lat0)
        lam0r = _np.deg2rad(lon0)
        latr  = _np.deg2rad(lat)
        lamr  = _np.deg2rad(lon)

        x_east  = EARTH_R * _np.cos(lat0r) * (lamr - lam0r)
        y_north = EARTH_R * (latr - lat0r)
        p = _np.column_stack([x_east, y_north])

        # Unit vectors from initial heading (0°=North, 90°=East)
        vhat  = _np.array([_np.sin(hdg0), _np.cos(hdg0)])
        nperp = _np.array([-vhat[1], vhat[0]])

        s  = p @ vhat         # along-track [m]
        ax = _np.abs(p @ nperp)  # |cross-track| [m]

        # ∫ |x| ds by trapezoid; only forward progress contributes
        ds     = _np.diff(s)
        ds_pos = _np.maximum(ds, 0.0)
        ax_mid = 0.5 * (ax[:-1] + ax[1:])
        area   = float(_np.sum(ax_mid * ds_pos))  # m^2

        v0 = float(init_speed_ownship if ac_idx == 0 else init_speed_intruder)  # m/s
        L0 = max(v0 * tmax, 1e-6)  # guard
        vals.append(area / L0)     # meters

    return float(_np.mean(vals)) if vals else 0.0
# -----------------------------------------------------------------------------#

def _lookup_max_time_norm(reception_prob: float, dpsi: int, dtlook: int, csv_path: str = "max_time_norm.csv") -> float:
    """
    Returns max_time_norm for a given combination of reception_prob, dpsi, and dtlook.
    Falls back to 10 if dpsi < 10, otherwise 4 if no exact match is found or file is missing.
    """
    try:
        df = pd.read_csv(csv_path)
        dpsi = 180 - abs((dpsi % 360) - 180)  # create triangle wave function
        dtlook = math.ceil(dtlook / 5) * 5
        dtlook = 15 if dtlook > 15 else dtlook  # cap dtlook at 15
        reception_prob = math.floor(reception_prob / 0.1) * 0.1

        result = df[
            (df['reception_prob'] == reception_prob) &
            (df['dpsi'] == dpsi) &
            (df['dtlook'] == dtlook)
        ]
        if not result.empty:
            return float(result.iloc[0]['max_time_norm'])
    except Exception:
        pass

    # fallback behavior
    return 10.0 if dpsi < 10 else 4.0

def _run_repetition(
    rep: int,
    dpsi: int,
    tmax: float,
    init_speed_intruder: float,
    pos_uncertainty_sigma: float,
    vel_uncertainty_sigma: float,
    reception_prob: float,
    lookahead_time: float,
    width: int,
    height: int,
    horizontal_sep: float,
    init_speed_ownship: float,
    aircraft_type: str,
    SIMDT_FACTOR: int,
    vehicle_uncertainty: str,
    conf_reso_algo_select: str,
    resofach: float,
    max_tr: float,
    max_dtr2: float,
    trajectory_dir: str | None = None,
):
    """
    One Monte-Carlo repetition. Kept as a module-level function for Joblib pickling.
    Returns (los_count, distance_cpa_array)
    """
    # Import/construct inside worker to avoid pickling stateful objects
    import numpy as _np
    # import bluesky as bs
    from cd_statebased import StateBased as _StateBased
    from cr_mvp import MVP as _MVP
    from cr_vo import VO as _VO
    from cns_adsl import ADSL as _ADSL
    from pairwise_conflict import PairwiseHorConflict as _PairwiseHorConflict

    if not getattr(bs, "_joblib_inited", False):
        with suppress_stdout_stderr():
            bs.init(mode="sim", detached=True)
        bs._joblib_inited = True

    # Apply traffic limits globally
    bs.traf.MAX_TR = max_tr
    bs.traf.MAX_DTR2 = max_dtr2

    # Resolution algorithm
    conf_detection = _StateBased()
    conf_detection_ground_truth = _StateBased()
    conf_resolution = _VO() if conf_reso_algo_select == 'v' else _MVP()

    pairwise = _PairwiseHorConflict(
        pair_width=width, pair_height=height,
        asas_pzr_m=horizontal_sep, dtlookahead=lookahead_time,
        init_speed_ownship=init_speed_ownship, init_speed_intruder=init_speed_intruder,
        init_dpsi=dpsi, aircraft_type_ownship=aircraft_type
    )

    # Apply resolution factor scaling (if your VO/MVP uses it)
    if hasattr(conf_resolution, "resofach"):
        conf_resolution.resofach = resofach

    _np.random.seed(42 + rep)  # distinct seeds per repetition

    simdt = bs.settings.simdt * SIMDT_FACTOR
    distance_array = []
    sim_timer_second = 0.0

    # ADSL models (created inside worker)
    adsl_bus = _ADSL(pos_uncertainty_sigma, vel_uncertainty_sigma, reception_prob)

    ownship_adsl = None
    intruder_adsl = None
    prev_intruder_adsl = None

    # --- record trajectory ---
    time_list, id_list = [], []
    lat_list, lon_list = [], []
    gs_list, hdg_list = [], []

    while sim_timer_second < tmax:
        states = pairwise._get_states()

        if sim_timer_second == 0.0:
            ownship_adsl = _ADSL(pos_uncertainty_sigma, vel_uncertainty_sigma, reception_prob=1.0)
            intruder_adsl = _ADSL(pos_uncertainty_sigma, vel_uncertainty_sigma, reception_prob=reception_prob)
            prev_intruder_adsl = _ADSL(pos_uncertainty_sigma, vel_uncertainty_sigma, reception_prob=None)

            ownship_adsl._get_noisy_states(states)
            adsl_bus.send_data(intruder_adsl, ownship_adsl, None)
            adsl_bus.send_data(prev_intruder_adsl, intruder_adsl, None)

        if (round(sim_timer_second, 2) % bs.settings.asas_dt == 0):
            ownship_adsl._get_noisy_states(states)

            rx_mask = _np.random.random(size=states.ntraf) <= intruder_adsl.reception_prob
            update_indices = _np.where(rx_mask)[0]
            missed_indices = _np.where(~rx_mask)[0]

            if update_indices.size > 0:
                adsl_bus.send_data(intruder_adsl, ownship_adsl, update_indices)
            if missed_indices.size > 0:
                adsl_bus.send_data(intruder_adsl, prev_intruder_adsl, missed_indices)

            adsl_bus.send_data(prev_intruder_adsl, intruder_adsl)

            ownship = ownship_adsl if "o" in vehicle_uncertainty else states
            intruder = intruder_adsl if "i" in vehicle_uncertainty else states

            conf_detection.detect(ownship, intruder, horizontal_sep, 100, lookahead_time)
            conf_detection_ground_truth.detect(states, states, horizontal_sep, 100, 100)
            reso = conf_resolution.resolve(conf_detection, ownship, intruder)

        distance_ = pairwise.step(conf_detection, reso, simdt_factor=SIMDT_FACTOR)
        distance_array.append(distance_)

        # collect
        time_list.append(sim_timer_second)
        id_list.append(states.id)
        lat_list.append(states.lat)
        lon_list.append(states.lon)
        gs_list.append(states.gs)
        hdg_list.append(states.hdg)

        sim_timer_second += simdt

    distance_cpa = np.min(distance_array, axis=0)
    los = (distance_cpa < horizontal_sep).sum()

    # --- normalized x-track area metric (mean over aircraft) ---
    xtrack_area_norm = _xtrack_area_norm_all_ac(
        time_list, id_list, lat_list, lon_list, hdg_list,
        tmax=tmax,
        init_speed_ownship=init_speed_ownship,
        init_speed_intruder=init_speed_intruder,
    )

    # --- save CSV if requested ---
    if trajectory_dir:
        out_dir = os.path.join(trajectory_dir, str(int(init_speed_intruder)), f"dpsi_{dpsi}")
        os.makedirs(out_dir, exist_ok=True)
        trajectory_df = pd.DataFrame({
            "time": time_list,
            "id":   id_list,
            "lat":  lat_list,
            "lon":  lon_list,
            "gs":   gs_list,
            "hdg":  hdg_list,
        })
        if (rep < 2):
            out_name = f"trajectory_look{int(lookahead_time)}_resof{resofach:.4f}_rep{rep}.csv"
            trajectory_df.to_csv(os.path.join(out_dir, out_name), index=False)

    pairwise.reset()

    return int(los), np.round(distance_cpa, 3), float(xtrack_area_norm)


class PairwiseIPRSimulator:
    """
    Usage:
        sim = PairwiseIPRSimulator(
            pos_uncertainty_sigma=..., vel_uncertainty_sigma=...,
            reception_prob=..., lookahead_time=..., init_speed_intruder=..., dpsi=...,
            max_tr=15, max_dtr2=10, resofach=1.05
        )
        total_ipr = sim.compute()
    """

    def __init__(
        self,
        pos_uncertainty_sigma: float,
        vel_uncertainty_sigma: float,
        reception_prob: float,
        lookahead_time: float,
        init_speed_intruder: float,
        dpsi: int,
        *,
        max_tr: int = 15,
        max_dtr2: int = 10,
        resofach: float = 1.05,
        width: int = 5,
        height: int = 5,
        horizontal_sep: float = 50.0,
        init_speed_ownship: float = 20.0,
        aircraft_type: str = "M600",
        nb_of_repetition: int = 1,
        SIMDT_FACTOR: int = 4,
        vehicle_uncertainty: str = "oi",
        conf_reso_algo_select: str = "m",
        n_jobs: int = 1,
        max_time_norm_csv: str = "max_time_norm.csv",
        trajectory_dir: str | None = None,
    ):
        # Required inputs
        self.pos_uncertainty_sigma = pos_uncertainty_sigma
        self.vel_uncertainty_sigma = vel_uncertainty_sigma
        self.reception_prob = reception_prob
        self.lookahead_time = lookahead_time
        self.init_speed_intruder = init_speed_intruder
        self.dpsi = dpsi

        # Extra control
        self.max_tr = max_tr
        self.max_dtr2 = max_dtr2
        self.resofach = resofach

        # Optional/config
        self.width = width
        self.height = height
        self.horizontal_sep = horizontal_sep
        self.init_speed_ownship = init_speed_ownship
        self.aircraft_type = aircraft_type
        self.nb_of_repetition = nb_of_repetition
        self.SIMDT_FACTOR = SIMDT_FACTOR
        self.vehicle_uncertainty = vehicle_uncertainty
        self.conf_reso_algo_select = conf_reso_algo_select
        cpu_workers = max(1, os.cpu_count() or 1)
        self.n_jobs = max(1, min(int(n_jobs), cpu_workers))
        self.max_time_norm_csv = max_time_norm_csv
        self.trajectory_dir = trajectory_dir
        # # Init BlueSky once in the main process
        # if not getattr(bs, "_main_inited", False):
        #     bs.init(mode='sim', detached=True)
        #     bs._main_inited = True

    def compute(self) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
        - total_ipr (float)
        - distance_cpa (np.ndarray) concatenated across repetitions
        - max_abs_xtrack_arr (np.ndarray)   shape: (nb_of_repetition,)
        - mean_abs_xtrack_arr (np.ndarray)  shape: (nb_of_repetition,)
        - p95_abs_xtrack_arr (np.ndarray)   shape: (nb_of_repetition,)
        """
        max_time_norm = _lookup_max_time_norm(
            self.reception_prob, self.dpsi, int(self.lookahead_time), csv_path=self.max_time_norm_csv
        )
        tmax = max_time_norm * 1.25 * self.lookahead_time

        with parallel_backend("loky", inner_max_num_threads=1):
            results_parallel = ReportingParallel(
                n_jobs=self.n_jobs,
                verbose=0,
                report_every=100,
            )(
                delayed(_run_repetition)(  # unchanged args
                    rep,
                    self.dpsi,
                    tmax,
                    self.init_speed_intruder,
                    self.pos_uncertainty_sigma,
                    self.vel_uncertainty_sigma,
                    self.reception_prob,
                    self.lookahead_time,
                    self.width,
                    self.height,
                    self.horizontal_sep,
                    self.init_speed_ownship,
                    self.aircraft_type,
                    self.SIMDT_FACTOR,
                    self.vehicle_uncertainty,
                    self.conf_reso_algo_select,
                    self.resofach,
                    self.max_tr,
                    self.max_dtr2,
                    self.trajectory_dir
                )
                for rep in range(self.nb_of_repetition)
            )

            if results_parallel:
                los_list, distance_cpa_list, area_list = zip(*results_parallel)
                total_los = int(sum(los_list))
                distance_cpa = np.concatenate(distance_cpa_list, axis=0)
                xtrack_area_norm_arr = np.array(area_list, dtype=float)
            else:
                total_los = 0
                distance_cpa = np.array([])
                xtrack_area_norm_arr = np.array([], dtype=float)

            denom = self.width * self.height * max(1, self.nb_of_repetition)
            total_ipr = ((denom - total_los) / denom) if denom > 0 else 0.0
            return float(total_ipr), distance_cpa, xtrack_area_norm_arr
