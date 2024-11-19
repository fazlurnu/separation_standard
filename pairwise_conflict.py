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
SPEED = 20

TMAX = DTLOOKAHEAD * 4

PAIR_WIDTH = 4
PAIR_HEIGHT = 4
NB_PAIR = PAIR_WIDTH * PAIR_HEIGHT

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

    def __init__(self):
        bs.init(mode='sim', detached=True)

        nb_pair = 4

        start_lat = 52.3
        start_lon = 4.7
        delta_lat_lon = 0.02

        # set conflict definition
        bs.settings.asas_pzr = ASAS_PZR_M * M2NM
        bs.settings.asas_dtlookahead = DTLOOKAHEAD

        dcpa = DCPA_M * M2NM
        dpsi = 45

        # create drones
        counter = 0

        for i in range(PAIR_WIDTH):
            for j in range(PAIR_HEIGHT):
                ownship_id = f"DRO{counter:03}"
                intruder_id = f"DRI{counter:03}"

                aclats = start_lat + i * delta_lat_lon
                aclons = start_lon + j * delta_lat_lon
                achdgs = 0 ## in degrees
                
                bs.traf.cre(acid=ownship_id, actype=DRONE_TYPE, aclat=aclats, aclon=aclons,
                    achdg=achdgs, acalt=ALT, acspd=SPEED)

                bs.traf.creconfs(acid=intruder_id, actype = DRONE_TYPE, targetidx=bs.traf.id2idx(ownship_id),
                                dpsi=dpsi, dcpa = dcpa, tlosh = bs.settings.asas_dtlookahead, spd = SPEED)
                
                counter += 1
            
    def step(self):
        # set simulation time step, and enable fast-time running
        simdt = bs.settings.simdt
        bs.stack.stack(f"DT {simdt};FF")

        t = np.arange(0, TMAX + simdt, simdt)

        # allocate some empty arrays for the results
        ntraf = bs.traf.ntraf
        self.res = np.zeros((len(t), 4, ntraf))

        # iteratively simulate the traffic
        distance_array = np.zeros((len(t)))

        for i in range(len(t)):
            # Perform one step of the simulation
            bs.sim.step()

            lat_dro_0 = bs.traf.lat[0]
            lon_dro_0 = bs.traf.lon[0]
            point_dro = (lat_dro_0, lon_dro_0)

            lat_dri_0 = bs.traf.lat[1]
            lon_dri_0 = bs.traf.lon[1]
            point_dri = (lat_dri_0, lon_dri_0)

            distance = geodesic(point_dro, point_dri).meters
            distance_array[i] = distance

            # save the results from the simulator in the results array,
            # here we keep the latitude, longitude, altitude and TAS
            self.res[i] = [bs.traf.lat,
                        bs.traf.lon,
                        bs.traf.alt,
                        bs.traf.tas,
                        ]
            
        print(min(distance_array))

    def plot(self):
        plt.figure()

        for idx in range(bs.traf.ntraf):
            plt.plot(self.res[:, 1, idx], self.res[:, 0, idx])

        plt.show()