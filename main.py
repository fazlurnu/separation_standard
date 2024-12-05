from pairwise_conflict import *
import bluesky as bs

from cd_statebased import StateBased
from cr_mvp import MVP

import time

time_list = []
pairlist = []

## make a conflict pair pair_width * pair_height,
## with asas_pzr_m as the horizontal separation
## and dtlookahead as the lookahead time
width = 5
height = 5

horizontal_sep = 200
lookahead_time = 15 ## seconds
tmax = 20 * lookahead_time ## seconds
minsimtime = 10 ## seconds

init_speed = 20
aircraft_type = 'EC35'

start_time = time.time()

## initiate things
pairwise = PairwiseHorConflict(pair_width=width, pair_height=height,
                            asas_pzr_m=horizontal_sep, dtlookahead=lookahead_time,
                            init_speed=init_speed, drone_type= aircraft_type)

conf_detection = StateBased()
conf_resolution = MVP()

## simulations
simdt = bs.settings.simdt
t = np.arange(0, tmax + simdt, simdt)
print(len(t))
distance_array = []

lat_array = []
lon_array = []

sim_timer_second = 0
still_in_conflict = True
last_in_conflicts = []
nb_check_last_in_conflicts = 20 ## val * simdt is the duration of the checking

while ((sim_timer_second < tmax) & (still_in_conflict)):
    states = pairwise._get_states()
    conf_detection.detect(states, states, horizontal_sep, 100, lookahead_time)
    reso = conf_resolution.resolve(conf_detection, states, states)
    
    distance_ = pairwise.step(conf_detection, reso)
    distance_array.append(distance_)
        
    sim_timer_second += simdt


    if(sim_timer_second > minsimtime):
        ## later here add also the conflicting based on ADS-B
        in_conflict = len(conf_detection.confpairs_unique) > 0

        last_in_conflicts.append(in_conflict)
        if len(last_in_conflicts) > nb_check_last_in_conflicts:
            last_in_conflicts.pop(0)
            
        print(last_in_conflicts)

        still_in_conflict = any(last_in_conflicts)

## calcualte the metrics
distance_cpa = np.min(distance_array, axis=0)

los = (distance_cpa < horizontal_sep).sum()
ipr = ((width*height) - los)/(width*height)

print(conf_detection.confpairs_unique)
print(ipr, los)

pairwise.reset()

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")