import sys
import pandas as pd
import numpy as np
import time
import json
import os

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

start_lat = params["start_lat"]
start_lon = params["start_lon"]
delta_lat_lon = params["delta_lat_lon"]
reception_prob = 0.8

# Simulation grid and conflict parameters
width = 10
height = 10
horizontal_sep = 50  # in meters
lookahead_time = 1  # seconds

init_speed_ownship = 20  # kts
aircraft_type = 'M600'
nb_of_repetition = 100
show_viz = False

# Argument parsing
if len(sys.argv) != 4:
    print("Usage: python main.py <nav_uncertainty> <vehicle_uncertainty> <conf_reso_algo_select>")
    print("Settings is set to default: shp oi m")
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
pos_uncertainty_sigma = 15 if 'p' in nav_uncertainty else 0
vel_uncertainty_sigma = 0.5 if 'v' in nav_uncertainty else 0

# Initialize simulation
bs.init(mode='sim', detached=True)
conf_detection = StateBased()
conf_resolution = VO() if conf_reso_algo_select == 'v' else MVP()
adsl = ADSL(pos_uncertainty_sigma, vel_uncertainty_sigma, reception_prob)
fname = f"{nav_uncertainty}_{vehicle_uncertainty}_{conf_reso_algo_select}"

# Main simulation loops with tqdm
for lookahead_time in tqdm([10, 15], desc="Lookahead time"):
    tmax = 10 * lookahead_time
    minsimtime = tmax / 30

    for init_speed_intruder in tqdm([20], desc="Intruder Speeds"):
        # Create directory for this speed if it doesn't exist
        speed_dir = f"trajectories/{init_speed_intruder}"
        os.makedirs(speed_dir, exist_ok=True)
        
        results = {'angles': [], 'ipr': [], 'los_count': [], 'distance_cpa': []}

        for dpsi in tqdm([2], desc=f"DPSI for SPD {init_speed_intruder}", leave=False):
        # for dpsi in tqdm(range(2, 4, 2), desc=f"DPSI for SPD {init_speed_intruder}", leave=False):
            
            if((dpsi < 10) & (init_speed_intruder == init_speed_ownship)):
                tmax = 10 * lookahead_time
            elif(dpsi < 20):
                tmax = 7 * lookahead_time
            else:
                tmax = 5 * lookahead_time

            los_list = []
            distance_cpa_list = []
            
            # Create directory for this dpsi if it doesn't exist
            dpsi_dir = f"{speed_dir}/dpsi_{dpsi}"
            os.makedirs(dpsi_dir, exist_ok=True)
            
            for rep in tqdm(range(nb_of_repetition), desc="rep", leave=False):  # Changed _ to rep to track repetition number
                start_time = time.time()

                pairwise = PairwiseHorConflict(
                    pair_width=width, pair_height=height,
                    asas_pzr_m=horizontal_sep, dtlookahead=lookahead_time,
                    init_speed_ownship=init_speed_ownship, init_speed_intruder=init_speed_intruder,
                    init_dpsi=dpsi, aircraft_type_ownship=aircraft_type
                )

                simdt = bs.settings.simdt
                t = np.arange(0, tmax + simdt, simdt)
                distance_array = []

                sim_timer_second = 0
                still_in_conflict = True
                no_conflict_counter = 0
                nb_check_last_in_conflicts = 100

                lat_list, lon_list = [], []
                gs_list, hdg_list = [], []
                lat_list_noise, lon_list_noise = [], []
                id_list = []

                while sim_timer_second < tmax and True:
                    states = pairwise._get_states()

                    if round(sim_timer_second, 2) % bs.settings.asas_dt == 0:
                        adsl._get_noisy_states(states)

                        ownship = adsl if "o" in vehicle_uncertainty else states
                        intruder = adsl if "i" in vehicle_uncertainty else states

                        conf_detection.detect(ownship, intruder, horizontal_sep, 100, lookahead_time)
                        reso = conf_resolution.resolve(conf_detection, ownship, intruder)

                    distance_ = pairwise.step(conf_detection, reso)
                    distance_array.append(distance_)
                    sim_timer_second += simdt

                    id_list.append(states.id)
                    lat_list.append(states.lat)
                    lon_list.append(states.lon)
                    gs_list.append(states.gs)
                    hdg_list.append(states.hdg)
                    lat_list_noise.append(adsl.lat)
                    lon_list_noise.append(adsl.lon)

                    if len(reso[-1]) == 0:
                        no_conflict_counter += 1
                    else:
                        no_conflict_counter = 0

                    if no_conflict_counter > nb_check_last_in_conflicts:
                        still_in_conflict = reception_prob < 0.2
                    else:
                        still_in_conflict = True

                # Save trajectory data for this repetition
                trajectory_df = pd.DataFrame({
                    'id': id_list,
                    'lat': lat_list,
                    'lon': lon_list,
                    'gs': gs_list,
                    'hdg': hdg_list
                })

                # if(rep < 2):
                    # trajectory_df.to_csv(f"{dpsi_dir}/trajectory_{fname}_{init_speed_intruder}_{pos_uncertainty_sigma}_{vel_uncertainty_sigma}_{rep}.csv", index=False)
                
                # if(dpsi == 2):
                #     trajectory_df.to_csv(f"{dpsi_dir}/trajectory_{fname}_{init_speed_intruder}_{pos_uncertainty_sigma}_{vel_uncertainty_sigma}_{rep}.csv", index=False)
                
                # trajectory_df.to_csv(f"{dpsi_dir}/trajectory_{fname}_{init_speed_intruder}_{pos_uncertainty_sigma}_{vel_uncertainty_sigma}_{rep}.csv", index=False)

                distance_cpa = np.min(distance_array, axis=0)

                los = (distance_cpa < horizontal_sep).sum()
                ipr = ((width * height) - los) / (width * height)

                los_list.append(los)
                # distance_cpa_list.append(distance_cpa[distance_cpa < horizontal_sep])
                distance_cpa_list.append(distance_cpa)

                # print(f"Intruder_SPD: {init_speed_intruder}, DPSI: {dpsi}, IPR: {ipr}, LOS: {los}")
                # print(f"Distance CPA: {distance_cpa}")
                # print(f"Execution time: {time.time() - start_time:.2f} sec | Sim time: {sim_timer_second:.2f} sec")

                id_list, lat_list, lon_list
                if show_viz:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 5))

                    lon_list = np.array(lon_list)
                    lat_list = np.array(lat_list)
                    lon_list_noise = np.array(lon_list_noise)
                    lat_list_noise = np.array(lat_list_noise)

                    for i in range(lon_list.shape[1]):
                        plt.plot(lon_list[:, i], lat_list[:, i], color='tab:blue')
                        # plt.scatter(lon_list_noise[:, i], lat_list_noise[:, i], color='tab:red', alpha=0.2)

                    # for pair in conf_detection.confpairs_unique:
                    #     idx = states.id2idx(list(pair)[0])
                    #     plt.plot([states.lon[idx] - delta_lat_lon / 2, states.lon[idx] + delta_lat_lon / 2],
                    #              [states.lat[idx], states.lat[idx]], color='red', alpha=0.4)
                    #     plt.plot([states.lon[idx], states.lon[idx]],
                    #              [states.lat[idx] - delta_lat_lon / 2, states.lat[idx] + delta_lat_lon / 2],
                    #              color='red', alpha=0.4)

                    plt.xlabel("Longitude")
                    plt.ylabel("Latitude")
                    plt.show()

                pairwise.reset()

            total_los = sum(los_list)
            total_ipr = ((width * height * nb_of_repetition) - total_los) / (width * height * nb_of_repetition)

            results['angles'].append(dpsi)
            results['ipr'].append(total_ipr)
            results['los_count'].append(total_los)
            results['distance_cpa'].append(distance_cpa_list)

        df = pd.DataFrame(results)
        df.to_csv(
            f'results/results_{fname}_{init_speed_intruder}_{pos_uncertainty_sigma}_{vel_uncertainty_sigma}_{lookahead_time}.csv',
            index=False)