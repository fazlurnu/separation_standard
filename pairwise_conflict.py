import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

import bluesky as bs

bs.init(mode='sim', detached=True)

nb_pair = 4

start_lat = 52.3
start_lon = 4.7
delta_lat_lon = 0.02

# drone config
drone_type = 'M600'
alt = 100 # in what?
speed = 20 # in what?

# set conflict definition
asas_pzr = 50 # in m
nautical_mile = 1852 # in m
bs.settings.asas_pzr = asas_pzr/nautical_mile # in nautical mile
bs.settings.asas_dtlookahead = 15 # in seconds

dcpa_m = 20
dcpa = dcpa_m/nautical_mile
dpsi = 45

# create drones
for pair_id in range(nb_pair):
    ownship_id = f"DRO{pair_id:03}"
    intruder_id = f"DRI{pair_id:03}"

    aclats = start_lat
    aclons = start_lon + pair_id * delta_lat_lon
    achdgs = 0 ## in degrees
    
    bs.traf.cre(acid=ownship_id, actype=drone_type, aclat=aclats, aclon=aclons,
        achdg=achdgs, acalt=alt, acspd=speed)

    bs.traf.creconfs(acid=intruder_id, actype = drone_type, targetidx=bs.traf.id2idx(ownship_id),
                     dpsi=dpsi, dcpa = dcpa, tlosh = bs.settings.asas_dtlookahead, spd = speed)
    
# run simulations
    
# set simulation time step, and enable fast-time running
simdt = bs.settings.simdt
bs.stack.stack(f"DT {simdt};FF")

# we'll run the simulation for up to 120 seconds
t_max = bs.settings.asas_dtlookahead * 4
t = np.arange(0, t_max + simdt, simdt)

# allocate some empty arrays for the results
ntraf = bs.traf.ntraf
res = np.zeros((len(t), 4, ntraf))

# iteratively simulate the traffic
distance_array = np.zeros((len(t)))

bs.stack.stack(f"ASAS ON")
bs.stack.stack(f"RESO MVP")

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
    res[i] = [bs.traf.lat,
                bs.traf.lon,
                bs.traf.alt,
                bs.traf.tas,
                ]

plt.figure()

for idx in range(ntraf):
    plt.plot(res[:, 1, idx], res[:, 0, idx])

plt.figure()
plt.plot(t, distance_array)
plt.axhline(dcpa_m, min(t), max(t), color = 'r', linestyle = '--')

plt.show()

print(min(distance_array))