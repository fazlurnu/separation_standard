import sys
import pandas as pd
import numpy as np
import time
import json
import os
import math

from tqdm import tqdm

from pairwise_conflict import *
import bluesky as bs

from cd_statebased import StateBased
from cr_mvp import MVP
from cr_vo import VO
from cns_adsl import ADSL

# Load simulation parameters
with open("pairwise_params.json", "r") as f:
    params = json.load(f)

from joblib import Parallel, delayed, parallel_backend

def run_repetition(
    rep,
    dpsi,
    tmax,
    init_speed_intruder,
    pos_uncertainty_sigma,
    vel_uncertainty_sigma,
    reception_prob,
    lookahead_time,
    width,
    height,
    horizontal_sep,
    init_speed_ownship,
    aircraft_type,
    SIMDT_FACTOR,
    vehicle_uncertainty,
    conf_reso_algo_select,
    show_viz,
):
    # Import/construct inside worker to avoid pickling stateful objects
    from cd_statebased import StateBased
    from cr_mvp import MVP
    from cr_vo import VO
    from cns_adsl import ADSL
    import bluesky as bs
    import numpy as np
    import os, psutil

    pid = os.getpid()
    # cpu = psutil.Process().cpu_num()
    print(f"[PID {pid} | Starting repetition {rep} (dpsi={dpsi}, speed={init_speed_intruder})")

    # Each process needs BlueSky settings; init once per process
    # If you see multiple "Reading config..." lines, that's expected with multiprocessing.
    if not getattr(bs, "_joblib_inited", False):
        bs.init(mode='sim', detached=True)
        bs._joblib_inited = True

    conf_detection = StateBased()
    conf_resolution = VO() if conf_reso_algo_select == 'v' else MVP()

    pairwise = PairwiseHorConflict(
        pair_width=width, pair_height=height,
        asas_pzr_m=horizontal_sep, dtlookahead=lookahead_time,
        init_speed_ownship=init_speed_ownship, init_speed_intruder=init_speed_intruder,
        init_dpsi=dpsi, aircraft_type_ownship=aircraft_type
    )

    # Optional: make runs reproducible but distinct
    np.random.seed((hash((rep, dpsi, init_speed_intruder)) & 0xFFFFFFFF))

    simdt = bs.settings.simdt * SIMDT_FACTOR
    distance_array = []
    sim_timer_second = 0.0

    # ADSL models (created inside worker)
    adsl_bus = ADSL(pos_uncertainty_sigma, vel_uncertainty_sigma, reception_prob)

    ownship_adsl = None
    intruder_adsl = None
    prev_intruder_adsl = None

    while sim_timer_second < tmax:
        states = pairwise._get_states()

        if sim_timer_second == 0:
            ownship_adsl = ADSL(pos_uncertainty_sigma, vel_uncertainty_sigma, reception_prob=1.0)
            intruder_adsl = ADSL(pos_uncertainty_sigma, vel_uncertainty_sigma, reception_prob=reception_prob)
            prev_intruder_adsl = ADSL(pos_uncertainty_sigma, vel_uncertainty_sigma, reception_prob=None)

            # First full update
            ownship_adsl._get_noisy_states(states)
            adsl_bus.send_data(intruder_adsl, ownship_adsl, None)
            adsl_bus.send_data(prev_intruder_adsl, intruder_adsl, None)  # store t=0 snapshot

        # Update at each time step
        if ((round(sim_timer_second, 2) % bs.settings.asas_dt == 0)):
            ownship_adsl._get_noisy_states(states)

            # Per-aircraft reception sampling
            rx_mask = np.random.random(size=states.ntraf) <= intruder_adsl.reception_prob
            update_indices = np.where(rx_mask)[0]
            missed_indices = np.where(~rx_mask)[0]

            # Only update aircraft that received data
            if update_indices.size > 0:
                adsl_bus.send_data(intruder_adsl, ownship_adsl, update_indices)

            # Fill in missed aircraft with previous state
            if missed_indices.size > 0:
                adsl_bus.send_data(intruder_adsl, prev_intruder_adsl, missed_indices)

            # Store this state for next tick fallback
            adsl_bus.send_data(prev_intruder_adsl, intruder_adsl)

            # Choose data source
            ownship = ownship_adsl if "o" in vehicle_uncertainty else states
            intruder = intruder_adsl if "i" in vehicle_uncertainty else states

            # Run detection and resolution
            conf_detection.detect(ownship, intruder, horizontal_sep, 100, lookahead_time)
            reso = conf_resolution.resolve(conf_detection, ownship, intruder)

        distance_ = pairwise.step(conf_detection, reso, simdt_factor=SIMDT_FACTOR)
        distance_array.append(distance_)
        sim_timer_second += simdt

    distance_cpa = np.min(distance_array, axis=0)
    los = (distance_cpa < horizontal_sep).sum()

    return int(los), np.round(distance_cpa, 3)

def lookup_max_time_norm(reception_prob: float, dpsi: int, dtlook: int) -> float:
    """
    Returns max_time_norm for a given combination of reception_prob, dpsi, and dtlook.
    Returns None if no match is found.
    """

    # Load the dataset
    file_path = "max_time_norm.csv"
    df = pd.read_csv(file_path)

    dtlook = math.ceil(dtlook / 5) * 5

    result = df[
        (df['reception_prob'] == reception_prob) &
        (df['dpsi'] == dpsi) &
        (df['dtlook'] == dtlook)
    ]
    
    if not result.empty:
        return result.iloc[0]['max_time_norm']
    else:
        return 4 ## safe to say they should be done with the conf

def main():
    start_lat = params["start_lat"]
    start_lon = params["start_lon"]
    delta_lat_lon = params["delta_lat_lon"]
    reception_prob = 0.9

    SIMDT_FACTOR = 10

    # Simulation grid and conflict parameters
    width = 10
    height = 10
    horizontal_sep = 50  # in meters

    init_speed_ownship = 20  # kts
    aircraft_type = 'M600'
    nb_of_repetition = 500
    show_viz = False

    # Argument parsing
    if len(sys.argv) != 4:
        print("Usage: python main.py <nav_uncertainty> <vehicle_uncertainty> <conf_reso_algo_select>")
        print("Settings is set to default: vp oi m")
        nav_uncertainty = 'vp'
        vehicle_uncertainty = 'oi'
        conf_reso_algo_select = 'm'
    else:
        nav_uncertainty = sys.argv[1].lower().strip()
        vehicle_uncertainty = sys.argv[2].lower().strip()
        conf_reso_algo_select = sys.argv[3].lower().strip()
        print(f"Settings is set to: {nav_uncertainty} {vehicle_uncertainty} {conf_reso_algo_select}")

    # Validate inputs
    if any(c not in {'v', 'p'} for c in nav_uncertainty):
        raise ValueError("nav_uncertainty can only contain 'v', 'p'")
    if any(c not in {'o', 'i'} for c in vehicle_uncertainty):
        raise ValueError("vehicle_uncertainty can only be 'o' or 'i'")
    if conf_reso_algo_select not in {'m', 'v'}:
        raise ValueError("conf_reso_algo_select can only be 'm' or 'v'")

    print(f"Selected source of uncertainty: {nav_uncertainty}")
    print(f"Selected vehicle uncertainty: {vehicle_uncertainty}")
    print(f"Selected resolution algorithm: {conf_reso_algo_select}")

    # Uncertainty settings
    pos_uncertainty_sigma = 1.5 if 'p' in nav_uncertainty else 0
    vel_uncertainty_sigma = 0.5 if 'v' in nav_uncertainty else 0

    # Initialize simulation
    bs.init(mode='sim', detached=True)
    conf_detection = StateBased()
    conf_resolution = VO() if conf_reso_algo_select == 'v' else MVP()
    fname = f"{nav_uncertainty}_{vehicle_uncertainty}_{conf_reso_algo_select}"

    # Main simulation loops with tqdm
    for pos_uncertainty_sigma in tqdm([15], desc="pos_uncertainty_sigma"):
        for vel_uncertainty_sigma in tqdm([0], desc="vel_uncertainty_sigma"):
            adsl = ADSL(pos_uncertainty_sigma, vel_uncertainty_sigma, reception_prob)
            for reception_prob in [0.8]:
                for lookahead_time in tqdm([15], desc="Lookahead time"):
                    # for init_speed_intruder in tqdm([20], desc="Intruder Speeds"):
                    for init_speed_intruder in [5, 15, 20]:
                        # init_speed_ownship = init_speed_intruder
                        # Create directory for this speed if it doesn't exist
                        speed_dir = f"trajectories/{init_speed_intruder}"
                        os.makedirs(speed_dir, exist_ok=True)
                        
                        results = {'angles': [], 'ipr': [], 'los_count': [], 'distance_cpa': []}

                        # for dpsi in [4, 10]:
                        for dpsi in tqdm(range(2, 41, 2), desc=f"DPSI"):

                            # Example usage
                            tmax = lookup_max_time_norm(reception_prob, dpsi, lookahead_time) * 1.5 * lookahead_time
                            
                            # Create directory for this dpsi if it doesn't exist
                            dpsi_dir = f"{speed_dir}/dpsi_{dpsi}"
                            os.makedirs(dpsi_dir, exist_ok=True)
                            
                            los_list = []
                            distance_cpa_list = []

                            # If on macOS/Linux this is fine; still wrap in __main__ guard (below)
                            with parallel_backend("loky", inner_max_num_threads=1):
                                results_parallel = Parallel(n_jobs=4, verbose=5)(
                                    delayed(run_repetition)(
                                        rep,
                                        dpsi,
                                        tmax,
                                        init_speed_intruder,
                                        pos_uncertainty_sigma,
                                        vel_uncertainty_sigma,
                                        reception_prob,
                                        lookahead_time,
                                        width,
                                        height,
                                        horizontal_sep,
                                        init_speed_ownship,
                                        aircraft_type,
                                        SIMDT_FACTOR,
                                        vehicle_uncertainty,
                                        conf_reso_algo_select,
                                        show_viz,
                                    )
                                    for rep in range(nb_of_repetition)
                                )

                            los_list, distance_cpa_list = zip(*results_parallel)

                            total_los = sum(los_list)
                            total_ipr = ((width * height * nb_of_repetition) - total_los) / (width * height * nb_of_repetition)

                            results['angles'].append(dpsi)
                            results['ipr'].append(total_ipr)
                            results['los_count'].append(total_los)
                            results['distance_cpa'].append(distance_cpa_list)

                        df = pd.DataFrame(results)
                        df.to_csv(
                            f'results/results_{fname}_{init_speed_intruder}_{pos_uncertainty_sigma}_{vel_uncertainty_sigma}_{lookahead_time}_receptionprob{int(reception_prob*100)}.csv',
                            index=False)
    pass

if __name__ == "__main__":
    main()