import pandas as pd

from pairwise_conflict import *
import bluesky as bs

from cd_statebased import StateBased
from cr_mvp import MVP
from cr_vo import VO
from cns_adsl import ADSL

import time

import json

## read params
with open("pairwise_params.json", "r") as f:
    params = json.load(f)

# Access parameters
start_lat = params["start_lat"]
start_lon = params["start_lon"]
delta_lat_lon = params["delta_lat_lon"]

time_list = []
pairlist = []

## make a conflict pair pair_width * pair_height,
## with asas_pzr_m as the horizontal separation
## and dtlookahead as the lookahead time
width = 10
height = 10

horizontal_sep = 50 ## in meters
lookahead_time = 15 ## seconds
tmax = 4 * lookahead_time ## seconds
minsimtime = tmax/30 ## seconds

init_speed_ownship = 20 ## in kts
aircraft_type = 'M600'

def get_valid_input(prompt, allowed_letters):
    while True:
        user_input = input(prompt).lower().strip()
        if all(char in allowed_letters for char in user_input) and len(user_input) == len(set(user_input)):
            return user_input
        else:
            print(f"Invalid input. Please enter only the allowed initials: {', '.join(allowed_letters)}")

source_of_uncertainty = get_valid_input(
    "Speed, Heading, Position.\nWrite the initials to include it as part of the uncertainty (s/h/p): ",
    {'s', 'h', 'p'}
)

vehicle_uncertainty = get_valid_input(
    "Ownship, Intruder.\nWrite the initials to include it as source of uncertainty (o/i): ",
    {'o', 'i'}
)

print(f"Selected source of uncertainty: {source_of_uncertainty}")
print(f"Selected vehicle uncertainty: {vehicle_uncertainty}")

pos_uncertainty_sigma = 0
hdg_uncertainty_sigma = 0
spd_uncertainty_sigma = 0

if('s' in source_of_uncertainty):
    spd_uncertainty_sigma = 3
if('h' in source_of_uncertainty):
    hdg_uncertainty_sigma = 5
if('p' in source_of_uncertainty):
    pos_uncertainty_sigma = 15

show_viz = False

nb_of_repetition = 5

## TO DO:
## Implement ownship only or intruder only for the hdg and spd uncertainty

bs.init(mode='sim', detached=True)

conf_detection = StateBased()
conf_resolution = VO()
adsl = ADSL(pos_uncertainty_sigma, spd_uncertainty_sigma, hdg_uncertainty_sigma)

fname = "hdg_intruder_vo"

for init_speed_intruder in [20]:
    results = {'angles': [], 'ipr': [], 'los_count': [], 'distance_cpa': []}

    for dpsi in range(2, 181, 2):
        los_list = []
        distance_cpa_list = []
        
        for i in range(nb_of_repetition):
            start_time = time.time()

            ## initiate things
            pairwise = PairwiseHorConflict(pair_width=width, pair_height=height,
                                        asas_pzr_m=horizontal_sep, dtlookahead=lookahead_time,
                                        init_speed_ownship=init_speed_ownship, init_speed_intruder=init_speed_intruder,
                                        init_dpsi = dpsi,
                                        drone_type= aircraft_type)

            ## simulations
            simdt = bs.settings.simdt
            t = np.arange(0, tmax + simdt, simdt)
            distance_array = []


            sim_timer_second = 0
            still_in_conflict = True
            no_conflict_counter = 0
            nb_check_last_in_conflicts = 100 ## val * simdt is the duration of the checking

            lat_list = []
            lon_list = []

            lat_list_noise = []
            lon_list_noise = []

            while ((sim_timer_second < tmax) & (still_in_conflict)):
                states = pairwise._get_states()
                
                ## make sure the conf detect and reso only done every asas_dt
                if(round(sim_timer_second, 2) % bs.settings.asas_dt == 0):
                    adsl._get_noisy_pos(states)
                    adsl._get_noisy_hdg(states)
                    adsl._get_noisy_spd(states)

                    ## select here the source of uncertainty
                    ownship = states
                    intruder = states

                    if("o" in vehicle_uncertainty):
                        ownship = adsl
                    if("i" in vehicle_uncertainty):
                        intruder = adsl

                    conf_detection.detect(ownship, intruder, horizontal_sep, 100, lookahead_time)
                    reso = conf_resolution.resolve(conf_detection, ownship, intruder)

                distance_ = pairwise.step(conf_detection, reso)
                    
                distance_array.append(distance_)
                    
                sim_timer_second += simdt

                lat_list.append(states.lat)
                lon_list.append(states.lon)

                lat_list_noise.append(adsl.lat)
                lon_list_noise.append(adsl.lon)

                ## this might be useful in case want to optimize the sim time
                if(len(reso[-1]) == 0):
                    no_conflict_counter += 1
                else:
                    no_conflict_counter = 0

                if(no_conflict_counter > nb_check_last_in_conflicts):
                    still_in_conflict = False

            ## calcualte the metrics
            distance_cpa = np.min(distance_array, axis=0)

            los = (distance_cpa < horizontal_sep).sum()
            ipr = ((width*height) - los)/(width*height)

            los_list.append(los)
            distance_cpa_list.append(distance_cpa[distance_cpa < horizontal_sep])
            
            print(f"Intruder_SPD: {init_speed_intruder}, DPSI: {dpsi}, IPR: {ipr}, LOS: {los}")
            print(f"Distance CPA: {distance_cpa[distance_cpa < horizontal_sep]}")
            end_time = time.time()

            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")
            print(f"Simulation time: {sim_timer_second} seconds")

            if(show_viz):
                import matplotlib.pyplot as plt

                lon_list = np.array(lon_list)
                lat_list = np.array(lat_list)

                lon_list_noise = np.array(lon_list_noise)
                lat_list_noise = np.array(lat_list_noise)

                plt.figure(figsize=(10, 5))  # Create a new figure for each plot

                for i in range(lon_list.shape[1]):  # Loop over columns (50 iterations)
                    plt.plot(lon_list[:, i], lat_list[:, i], color = 'tab:blue')  # Plot all rows against this column
                    plt.scatter(lon_list_noise[:, i], lat_list_noise[:, i], color = 'tab:red', alpha = 0.2)  # Plot all rows against this column
                    plt.xlabel("Longitude")
                    plt.ylabel("Latitude")

                for pair in conf_detection.confpairs_unique:

                    idx = states.id2idx(list(pair)[0])

                    plt.plot([states.lon[idx] - delta_lat_lon/2, states.lon[idx] + delta_lat_lon/2], [states.lat[idx], states.lat[idx]], color = 'red', alpha = 0.4)
                    plt.plot([states.lon[idx], states.lon[idx]], [states.lat[idx] - delta_lat_lon/2, states.lat[idx] + delta_lat_lon/2], color = 'red', alpha = 0.4)

                plt.show()  # Display the plot

            pairwise.reset()

        total_los = sum(los_list)
        total_ipr = ((width*height*nb_of_repetition) - total_los)/(width*height*nb_of_repetition)

        results['angles'].append(dpsi)
        results['ipr'].append(total_ipr)
        results['los_count'].append(total_los)
        results['distance_cpa'].append(distance_cpa_list)

    df = pd.DataFrame.from_dict(results)
    df.to_csv(f'results_{fname}_{init_speed_intruder}_{pos_uncertainty_sigma}_{spd_uncertainty_sigma}_{hdg_uncertainty_sigma}.csv', index = False)