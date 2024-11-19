import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

import bluesky as bs

M2NM = 1/1852
NM2M = 1852

DCPA_M = 0
DPSI = 45

ASAS_PZR_M = 50 # in m
DTLOOKAHEAD = 15 # in seconds

DRONE_TYPE = 'M600'
ALT = 100

TMAX = DTLOOKAHEAD * 4

class PairwiseHorConflict():
    """ 
    PairwiseHorConflict

    TODO:
    - make a conflict detection class outside this script
    - call this conflict detection script here
    - make a conflict resolution class (MVP) outside this script
    - call this conflict resolution script here
    - make an ADS-L class -> position and communication uncertainty
    - implement the ADS-L class -> for instance bs.traf.lat + noise (position)
    - implement the ADS-L class -> for instnace if(random() < reception_prob): update_pos (communication uncertainty)
    """

    def __init__(self, 
                pair_width: int, 
                pair_height: int, 
                asas_pzr_m: float, 
                dtlookahead: float,
                speed: float,
                inherent_asas_on: bool = False) -> None:
        
        self.nb_pair = pair_width * pair_height
        self.asas_pzr_m = asas_pzr_m
        self.dtlookahead = dtlookahead

        bs.init(mode='sim', detached=True)

        start_lat = 52.3
        start_lon = 4.7
        delta_lat_lon = 0.02

        # set conflict definition
        bs.settings.asas_pzr = self.asas_pzr_m * M2NM
        bs.settings.asas_dtlookahead = self.dtlookahead

        dcpa = DCPA_M * M2NM

        # create drones
        counter = 0
        for i in range(pair_width):
            for j in range(pair_height):
                ownship_id = f"DRO{counter:03}"
                intruder_id = f"DRI{counter:03}"

                aclats = start_lat + i * delta_lat_lon
                aclons = start_lon + j * delta_lat_lon
                achdgs = 0 ## in degrees
                
                bs.traf.cre(acid=ownship_id, actype=DRONE_TYPE, aclat=aclats, aclon=aclons,
                    achdg=achdgs, acalt=ALT, acspd=speed)

                ## make intruder, dpsi is random
                bs.traf.creconfs(acid=intruder_id, actype = DRONE_TYPE, targetidx=bs.traf.id2idx(ownship_id),
                                dpsi=np.random.uniform(0, 360), dcpa = dcpa, tlosh = bs.settings.asas_dtlookahead, spd = speed)
                
                counter += 1

        if(inherent_asas_on):
            bs.stack.stack("ASAS ON")
            bs.stack.stack("RESO MVP")
            
    def step(self) -> np.ndarray:
        # set simulation time step, and enable fast-time running
        simdt = bs.settings.simdt
        bs.stack.stack(f"DT {simdt};FF")
        t = np.arange(0, TMAX + simdt, simdt)

        # allocate some empty arrays for the results
        ntraf = bs.traf.ntraf
        self.res = np.zeros((len(t), 4, ntraf))

        self.distance_array = np.zeros((len(t), self.nb_pair))

        # iteratively simulate the traffic
        for i in range(len(t)):
            # Perform one step of the simulation
            bs.sim.step()

            for pair in range(self.nb_pair):
                ownship_id = f"DRO{pair:03}"
                intruder_id = f"DRI{pair:03}"

                lat_dro_0 = bs.traf.lat[bs.traf.id2idx(ownship_id)]
                lon_dro_0 = bs.traf.lon[bs.traf.id2idx(ownship_id)]
                point_dro = (lat_dro_0, lon_dro_0)

                lat_dri_0 = bs.traf.lat[bs.traf.id2idx(intruder_id)]
                lon_dri_0 = bs.traf.lon[bs.traf.id2idx(intruder_id)]
                point_dri = (lat_dri_0, lon_dri_0)

                distance = geodesic(point_dro, point_dri).meters
                self.distance_array[i][pair] = distance

            self.res[i] = [bs.traf.lat,
                        bs.traf.lon,
                        bs.traf.alt,
                        bs.traf.tas,
                        ]
            
        self.distance_array = np.array(self.distance_array)
        distance_cpa = np.min(self.distance_array, axis=0)
        
        return distance_cpa

    def plot(self):
        plt.figure()

        for idx in range(bs.traf.ntraf):
            plt.plot(self.res[:, 1, idx], self.res[:, 0, idx])

        plt.show()