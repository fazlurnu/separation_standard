from pairwise_conflict import *
import bluesky as bs

from cd_statebased import StateBased
from cr_mvp import MVP
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

horizontal_sep = 50
lookahead_time = 15 ## seconds
tmax = 4 * lookahead_time ## seconds
minsimtime = tmax/30 ## seconds

init_speed_ownship = 20 ## in kts
init_speed_intruder = 15 ## in kts
aircraft_type = 'M600'

pos_uncertainty_sigma = 5
hdg_uncertainty_sigma = 2
spd_uncertainty_sigma = 3

show_viz = False

los_list = []
nb_of_repetition = 5

## TO DO:
## Implement ownship only or intruder only for the hdg and spd uncertainty
## Also add the conflict logger

for i in range(nb_of_repetition):
    start_time = time.time()

    ## initiate things
    pairwise = PairwiseHorConflict(pair_width=width, pair_height=height,
                                asas_pzr_m=horizontal_sep, dtlookahead=lookahead_time,
                                init_speed_ownship=init_speed_ownship, init_speed_intruder=init_speed_intruder,
                                drone_type= aircraft_type)

    conf_detection = StateBased()
    conf_resolution = MVP()
    adsl = ADSL(pos_uncertainty_sigma, spd_uncertainty_sigma, hdg_uncertainty_sigma)

    ## simulations
    simdt = bs.settings.simdt
    t = np.arange(0, tmax + simdt, simdt)
    distance_array = []

    lat_array = []
    lon_array = []

    sim_timer_second = 0
    still_in_conflict = True
    last_in_conflicts = []
    nb_check_last_in_conflicts = 20 ## val * simdt is the duration of the checking

    lat_list = []
    lon_list = []

    while ((sim_timer_second < tmax) & (still_in_conflict)):
        states = pairwise._get_states()
        adsl._get_noisy_pos(states)
        adsl._get_noisy_hdg(states)
        adsl._get_noisy_spd(states)
        
        ## make sure the conf detect and reso only done every asas_dt
        if(round(sim_timer_second, 2) % bs.settings.asas_dt == 0):
            conf_detection.detect(adsl, adsl, horizontal_sep, 100, lookahead_time)
            reso = conf_resolution.resolve(conf_detection, adsl, adsl)
        
        distance_ = pairwise.step(conf_detection, reso)
        distance_array.append(distance_)
            
        sim_timer_second += simdt

        lat_list.append(states.lat)
        lon_list.append(states.lon)

        ## this might be useful in case want to optimize the sim time
        # if(sim_timer_second >= minsimtime):
        #     ## later here add also the conflicting based on ADS-B
        #     in_conflict = len(conf_detection.confpairs_unique) > 0

        #     last_in_conflicts.append(in_conflict)
        #     if len(last_in_conflicts) > nb_check_last_in_conflicts:
        #         last_in_conflicts.pop(0)

        #     still_in_conflict = any(last_in_conflicts)

    ## calcualte the metrics
    distance_cpa = np.min(distance_array, axis=0)

    los = (distance_cpa < horizontal_sep).sum()
    ipr = ((width*height) - los)/(width*height)

    los_list.append(los)
    print(f"IPR: {ipr}, LOS: {los}")
    print(f"Distance CPA: {distance_cpa[distance_cpa < horizontal_sep]}")
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    print(f"Simulation time: {sim_timer_second} seconds")

    if(show_viz):
        import matplotlib.pyplot as plt

        lon_list = np.array(lon_list)
        lat_list = np.array(lat_list)

        plt.figure(figsize=(10, 5))  # Create a new figure for each plot

        for i in range(lon_list.shape[1]):  # Loop over columns (50 iterations)
            plt.plot(lon_list[:, i], lat_list[:, i], color = 'tab:blue')  # Plot all rows against this column
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

print(f"Final IPR: {total_ipr}. Final LOS: {total_los}")