from pairwise_conflict import *
import bluesky as bs

from cd_statebased import StateBased
from cr_mvp import MVP

## make a conflict pair pair_width * pair_height,
## with asas_pzr_m as the horizontal separation
## and dtlookahead as the lookahead time
width = 2
height = 2

horizontal_sep = 50
lookahead_time = 15 ## seconds
tmax = 1*lookahead_time ## seconds

init_speed = 20

## initiate things

pairwise = PairwiseHorConflict(pair_width=width, pair_height=height,
                               asas_pzr_m=horizontal_sep, dtlookahead=lookahead_time,
                               init_speed=init_speed, drone_type= 'M600')

conf_detection = StateBased()
print(type(conf_detection))  # This should now output <class 'cd_statebased.StateBased'>

conf_resolution = MVP()

## simulations
simdt = bs.settings.simdt
t = np.arange(0, tmax + simdt, simdt)

distance_array = []

lat_array = []
lon_array = []

for i in range(len(t)):
    states = pairwise._get_states()
    conf = conf_detection.detect(states, states, horizontal_sep, 100, lookahead_time)

    distance_ = pairwise.step(None)
    distance_array.append(distance_)

## calcualte the metrics
print(states.lat, states.ntraf)
print(conf)

distance_cpa = np.min(distance_array, axis=0)

los = (distance_cpa < horizontal_sep).sum()
ipr = ((width*height) - los)/(width*height)

print(distance_cpa)
print(ipr, los)