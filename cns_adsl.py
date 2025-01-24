''' Communication, navigation, surveillance model. '''
import numpy as np
from bluesky.tools import geo
from bluesky.tools.aero import nm

class ADSL():
    """ 
    
    """
    def __init__(self, confidence_interval, spd_uncertainty_sigma, hdg_uncertainty_sigma,
                 reception_prob = 1.0):
        # Calculate standard deviation from confidence interval
        # For 2D, 95% confidence interval is approximately 2.448 standard deviations
        self.reception_prob = reception_prob

        self.std_dev = confidence_interval / 2.448
        self.spd_uncertainty_sigma = spd_uncertainty_sigma
        self.hdg_uncertainty_sigma = hdg_uncertainty_sigma

        self.ntraf = 0
        self.lat     = np.array([])  # latitude [deg]
        self.lon     = np.array([])  # longitude [deg]
        self.alt     = np.array([])  # altitude [m]
        self.hdg     = np.array([])  # traffic heading [deg]
        self.trk     = np.array([])  # track angle [deg]
        self.gs      = np.array([])  # ground speed [m/s]
        self.vs      = np.array([])  # vertical speed [m/s]
        self.id      = []  # identifier (string)

        self.first_update_done = False
        
    def _get_noisy_pos(self, states, update_array = None):
        self.ntraf = states.ntraf
        self.alt = states.alt.copy()
        self.hdg = states.hdg.copy()
        self.trk = states.hdg.copy()
        self.gs = states.gs.copy()
        self.gseast = states.gseast.copy()
        self.gsnorth = states.gsnorth.copy()
        self.tas = states.tas.copy()

        self.vs = states.vs.copy()
        self.id = states.id.copy()

        self.perf = states.perf
        self.ap = states.ap
        self.selalt = states.selalt

        lat = states.lat.copy()
        lon = states.lon.copy()

        lat_noise = np.zeros_like(lat)
        lon_noise = np.zeros_like(lon)

        for i, (mean_lat, mean_lon) in enumerate(zip(lat, lon)):
            # Covariance matrix (assuming circular symmetric distribution)
            self.cov = np.array([[self.std_dev**2, 0], 
                                [0, self.std_dev**2]])
            
            # Generate random samples from multivariate normal distribution
            x, y = np.random.multivariate_normal((0, 0), self.cov).T

            # Convert meters to degrees for latitude
            # 1 degree of latitude is approximately 111,320 meters
            lat_noise[i] = y / 111320

            # Convert meters to degrees for longitude 
            # Longitude degrees depend on the latitude (they get smaller closer to the poles)
            # Use the length of a degree at the current latitude
            lon_noise[i] = x / (111320 * np.cos(np.deg2rad(mean_lat)))

        if(update_array != None):
            self.lat[update_array] = lat[update_array] + lat_noise[update_array]
            self.lon[update_array] = lon[update_array] + lon_noise[update_array]
        else:
            self.lat = lat + lat_noise
            self.lon = lon + lon_noise

    def _get_noisy_hdg(self, states, update_array = None):
        self.ntraf = states.ntraf

        if(update_array != None):
            trk      = states.trk[update_array]
            self.trk[update_array] = trk + (np.random.normal(0, self.hdg_uncertainty_sigma, self.ntraf))[update_array]
            self.gsnorth[update_array] = self.gs[update_array] * np.cos(np.deg2rad(self.trk[update_array]))
            self.gseast [update_array] = self.gs[update_array] * np.sin(np.deg2rad(self.trk[update_array]))
        else:
            self.trk = states.trk + (np.random.normal(0, self.hdg_uncertainty_sigma, self.ntraf))
            self.gsnorth = self.gs * np.cos(np.deg2rad(self.trk))
            self.gseast  = self.gs * np.sin(np.deg2rad(self.trk))

    def _get_noisy_spd(self, states, update_array = None):
        self.ntraf = states.ntraf    
        
        if(update_array != None):
            self.gs[update_array] = states.gs[update_array] + (np.random.normal(0, self.spd_uncertainty_sigma, self.ntraf))[update_array]
            self.gsnorth[update_array] = self.gs[update_array] * np.cos(np.deg2rad(self.trk[update_array]))
            self.gseast[update_array]  = self.gs[update_array] * np.sin(np.deg2rad(self.trk[update_array]))
        else:
            self.gs = states.gs + (np.random.normal(0, self.spd_uncertainty_sigma, self.ntraf))
            self.gsnorth = self.gs * np.cos(np.deg2rad(self.trk))
            self.gseast  = self.gs * np.sin(np.deg2rad(self.trk))
    
    def _get_noisy_states(self, states):
        ## Still buggy the update_prob_cond
        update_prob_cond = (np.random.random(size = states.ntraf) <= self.reception_prob)
        up = np.where(update_prob_cond)

        if not self.first_update_done:
            self._get_noisy_pos(states)
            self._get_noisy_hdg(states)
            self._get_noisy_spd(states)

            self.first_update_done = True
        else:
            self._get_noisy_pos(states, up)
            self._get_noisy_hdg(states, up)
            self._get_noisy_spd(states, up)