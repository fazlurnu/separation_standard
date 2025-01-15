import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from bluesky.tools import geo

from bluesky.tools.aero import cas2tas, casormach2tas, fpm, kts

import bluesky as bs

import json

M2NM = 1/1852
NM2M = 1852

DCPA_M = 0

ALT = 100

## read params
with open("pairwise_params.json", "r") as f:
    params = json.load(f)

# Access parameters
start_lat = params["start_lat"]
start_lon = params["start_lon"]
delta_lat_lon = params["delta_lat_lon"]

class PairwiseHorConflict():
    """ 
    PairwiseHorConflict

    TODO:
    - make an ADS-L class -> position and communication uncertainty
    - implement the ADS-L class -> for instance bs.traf.lat + noise (position)
    - implement the ADS-L class -> for instnace if(random() < reception_prob): update_pos (communication uncertainty)
    """

    def __init__(self, 
                pair_width: int, pair_height: int,      ## number of spawned aircraft
                asas_pzr_m: float, dtlookahead: float,  ## separation standard params
                init_speed_ownship: float, init_speed_intruder: float, drone_type: str,     ## aircraft params
                inherent_asas_on: bool = False) -> None:
        
        self.nb_pair = pair_width * pair_height

        self.asas_pzr_m = asas_pzr_m
        self.dtlookahead = dtlookahead

        self.init_speed_ownship = init_speed_ownship
        self.init_speed_intruder = init_speed_intruder
        self.drone_type = drone_type

        self.init_heading = np.array([
                                        0 if i % 2 == 0 else np.random.randint(0, 360)
                                        for i in range(2 * pair_width * pair_height)
                                    ])
        
        bs.init(mode='sim', detached=True)

        # set conflict definition
        bs.settings.asas_pzr = self.asas_pzr_m * M2NM
        bs.settings.asas_dtlookahead = self.dtlookahead

        dcpa = DCPA_M * M2NM

        # create drones
        counter = 0
        idx = 0

        for i in range(pair_width):
            for j in range(pair_height):
                ownship_id = f"DRO{counter:03}"
                intruder_id = f"DRI{counter:03}"

                aclats = start_lat + i * delta_lat_lon
                aclons = start_lon + j * delta_lat_lon
                
                ## the heading of this one is always zero
                bs.traf.cre(acid=ownship_id, actype= self.drone_type, aclat=aclats, aclon=aclons,
                    achdg=self.init_heading[idx], acalt=ALT, acspd=self.init_speed_ownship)
                
                idx += 1

                ## make intruder, dpsi is random
                bs.traf.creconfs(acid=intruder_id, actype = self.drone_type, targetidx=bs.traf.id2idx(ownship_id),
                                dpsi=self.init_heading[idx], dcpa = dcpa, tlosh = bs.settings.asas_dtlookahead, spd = self.init_speed_intruder)
                idx += 1
                
                counter += 1

        if(inherent_asas_on):
            bs.stack.stack("ASAS ON")
            bs.stack.stack("RESO MVP")
            
    def reset(self, seed=None, options=None) -> None:      
        bs.traf.reset()

    def _get_states(self):
        return bs.traf

    def step(self, detection, resolution) -> np.ndarray:
        # set simulation time step, and enable fast-time running
        simdt = bs.settings.simdt
        bs.stack.stack(f"DT {simdt};FF")

        # allocate some empty arrays for the results
        ntraf = bs.traf.ntraf

        self.distance_array = np.zeros((self.nb_pair))

        bs.sim.step()
        
        for i in range(ntraf):
            target_id = bs.traf.id[i]

            if resolution != None:
                if(any(target_id in pair for pair in detection.confpairs)):
                    bs.stack.stack(f"HDG {target_id}, {resolution[0][i]}")
                    bs.stack.stack(f"SPD {target_id}, {resolution[1][i] / kts}")
                else:
                    bs.stack.stack(f"HDG {target_id}, {self.init_heading[i]}") # the DRI
                    bs.stack.stack(f"SPD {target_id}, {self.init_speed_ownship}")

        ## FIX HERE THE DISTANCE CALCULATION
        for pair in range(self.nb_pair):
            ownship_id = f"DRO{pair:03}"
            intruder_id = f"DRI{pair:03}"

            lat_dro_0 = bs.traf.lat[bs.traf.id2idx(ownship_id)]
            lon_dro_0 = bs.traf.lon[bs.traf.id2idx(ownship_id)]
            point_dro = (lat_dro_0, lon_dro_0)

            lat_dri_0 = bs.traf.lat[bs.traf.id2idx(intruder_id)]
            lon_dri_0 = bs.traf.lon[bs.traf.id2idx(intruder_id)]
            point_dri = (lat_dri_0, lon_dri_0)

            qdr, dist = geo.kwikqdrdist_matrix(np.asmatrix(lat_dro_0), np.asmatrix(lon_dro_0),
                                    np.asmatrix(lat_dri_0), np.asmatrix(lon_dri_0))
            
            distance = geodesic(point_dro, point_dri).meters
            self.distance_array[pair] = dist * NM2M
        
        return np.array(self.distance_array)