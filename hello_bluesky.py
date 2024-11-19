import numpy as np

import bluesky as bs
from bluesky.simulation import ScreenIO

class ScreenDummy(ScreenIO):
    """
    Dummy class for the screen. Inherits from ScreenIO to make sure all the
    necessary methods are there. This class is there to reimplement the echo
    method so that console messages are printed.
    """
    def echo(self, text='', flags=0):
        """Just print echo messages"""
        print("BlueSky console:", text)

# initialize bluesky as non-networked simulation node
bs.init(mode='sim', detached=True)

# initialize dummy screen
bs.scr = ScreenDummy()

# generate some trajectories
n = 3

# create n aircraft with random positions, altitudes, speeds
bs.traf.mcre(n, actype="A320")

for acid in bs.traf.id:
    bs.stack.stack(f'ORIG {acid} EGLL;'
                   f'ADDWPT {acid} BPK FL60;'
                   f'ADDWPT {acid} TOTRI FL107;'
                   f'ADDWPT {acid} MATCH FL115;'
                   f'ADDWPT {acid} BRAIN FL164;'
                   f'VNAV {acid} ON')
    
# set simulation time step, and enable fast-time running
bs.stack.stack('DT 0.05;FF')

# we'll run the simulation for up to 4000 seconds
t_max = 4000

ntraf = bs.traf.ntraf
n_steps = int(t_max + 1)
t = np.linspace(0, t_max, n_steps)

# allocate some empty arrays for the results
res = np.zeros((n_steps, 4, ntraf))

# iteratively simulate the traffic
for i in range(n_steps):
    # Perform one step of the simulation
    bs.sim.step()

    # save the results from the simulator in the results array,
    # here we keep the latitude, longitude, altitude and TAS
    res[i] = [bs.traf.lat,
                bs.traf.lon,
                bs.traf.alt,
                bs.traf.tas]
    
# plot
import matplotlib.pyplot as plt

for idx, acid in enumerate(bs.traf.id):
    fig = plt.figure(figsize=(10, 15))
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 1), (2, 0))
    ax3 = plt.subplot2grid((4, 1), (3, 0))

    ax1.plot(res[:, 1, idx], res[:, 0, idx])

    ax2.plot(t, res[:, 2, idx])
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('alt [m]')

    ax3.plot(t, res[:, 3, idx])
    ax3.set_xlabel('t [s]')
    ax3.set_ylabel('TAS [m/s]')
    
    fig.suptitle(f'Trajectory {acid}')

plt.show()