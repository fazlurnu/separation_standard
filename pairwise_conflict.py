import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from bluesky.tools import geo

from bluesky.tools.aero import cas2tas, casormach2tas, fpm, kts

import bluesky as bs

import json

M2NM = 1/1852
NM2M = 1852

DCPA_M = 45

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
    - make an ADS-L class -> communication uncertainty
    - implement the ADS-L class -> for instnace if(random() < reception_prob): update_pos (communication uncertainty)
    """

    def __init__(self, 
                pair_width: int, pair_height: int,      ## number of spawned aircraft
                asas_pzr_m: float, dtlookahead: float,  ## separation standard params
                init_speed_ownship: float, init_speed_intruder: float,      ## aircraft params
                aircraft_type_ownship: str, aircraft_type_intruder: str = None,     ## aircraft params
                init_dpsi: float = None,
                inherent_asas_on: bool = False) -> None:
        
        self.nb_pair = pair_width * pair_height

        self.asas_pzr_m = asas_pzr_m
        self.dtlookahead = dtlookahead

        self.init_speed_ownship = init_speed_ownship
        self.init_speed_intruder = init_speed_intruder

        self.aircraft_type_ownship = aircraft_type_ownship
        if(aircraft_type_intruder == None):
            self.aircraft_type_intruder = aircraft_type_ownship
        else:
            self.aircraft_type_intruder = aircraft_type_intruder

        if(init_dpsi != None):
            self.init_heading = np.array([
                                        0 if i % 2 == 0 else init_dpsi
                                        for i in range(2 * pair_width * pair_height)
                                    ])
        else:
            self.init_heading = np.array([
                                            0 if i % 2 == 0 else np.random.randint(0, 360)
                                            for i in range(2 * pair_width * pair_height)
                                        ])

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
                bs.traf.cre(acid=ownship_id, actype= self.aircraft_type_ownship, aclat=aclats, aclon=aclons,
                    achdg=self.init_heading[idx], acalt=ALT, acspd=self.init_speed_ownship)
                
                idx += 1

                ## make intruder, dpsi is random
                bs.traf.creconfs(acid=intruder_id, actype = self.aircraft_type_intruder, targetidx=bs.traf.id2idx(ownship_id),
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

    def step(self, detection, resolution, simdt_factor = 1) -> np.ndarray:
        # set simulation time step, and enable fast-time running
        simdt = bs.settings.simdt * simdt_factor
        bs.stack.stack(f"DT {simdt};FF")

        # allocate some empty arrays for the results
        ntraf = bs.traf.ntraf

        self.distance_array = np.zeros((self.nb_pair))

        bs.sim.step()
        
        reso_hdg, reso_spd, _, _, resopairs = resolution

        for i in range(ntraf):
            target_id = bs.traf.id[i]

            if resolution != None:
                if(any(target_id in pair for pair in resopairs)):
                    bs.stack.stack(f"HDG {target_id}, {reso_hdg[i]}")
                    bs.stack.stack(f"SPD {target_id}, {reso_spd[i] / kts}")
                else:
                    bs.stack.stack(f"HDG {target_id}, {self.init_heading[i]}")
                    if("DRO" in target_id):
                        bs.stack.stack(f"SPD {target_id}, {self.init_speed_ownship}")
                    else:
                        bs.stack.stack(f"SPD {target_id}, {self.init_speed_intruder}")
            else:
                bs.stack.stack(f"HDG {target_id}, {self.init_heading[i]}")
                if("DRO" in target_id):
                    bs.stack.stack(f"SPD {target_id}, {self.init_speed_ownship}")
                else: 
                    bs.stack.stack(f"SPD {target_id}, {self.init_speed_intruder}")

        # Precompute IDs only once
        ownship_ids   = [f"DRO{i:03}" for i in range(self.nb_pair)]
        intruder_ids  = [f"DRI{i:03}" for i in range(self.nb_pair)]

        # Convert to indices just once
        ownship_idx   = [bs.traf.id2idx(oid) for oid in ownship_ids]
        intruder_idx  = [bs.traf.id2idx(iid) for iid in intruder_ids]

        # Gather lat/lon arrays for all pairs
        lat_dro = np.array([bs.traf.lat[idx] for idx in ownship_idx])
        lon_dro = np.array([bs.traf.lon[idx] for idx in ownship_idx])
        lat_dri = np.array([bs.traf.lat[idx] for idx in intruder_idx])
        lon_dri = np.array([bs.traf.lon[idx] for idx in intruder_idx])

        # Compute all distances in one vectorized call
        _, dist = geo.kwikqdrdist_matrix(
            np.asmatrix(lat_dro),
            np.asmatrix(lon_dro),
            np.asmatrix(lat_dri),
            np.asmatrix(lon_dri)
        )

        # Store results in meters
        self.distance_array[:] = np.diag(dist) * NM2M
        
        return np.array(self.distance_array)