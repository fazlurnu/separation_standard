''' Communication, navigation, surveillance model. '''
import numpy as np
from bluesky.tools import geo
from bluesky.tools.aero import nm

class ADSL():
    """ 
    
    """
    def __init__(self, confidence_interval):
        # Calculate standard deviation from confidence interval
        # For 2D, 95% confidence interval is approximately 2.448 standard deviations
        self.std_dev = confidence_interval / 2.448

        self.ntraf = 0
        self.lat     = np.array([])  # latitude [deg]
        self.lon     = np.array([])  # longitude [deg]
        self.alt     = np.array([])  # altitude [m]
        self.hdg     = np.array([])  # traffic heading [deg]
        self.trk     = np.array([])  # track angle [deg]
        self.gs      = np.array([])  # ground speed [m/s]
        self.vs      = np.array([])  # vertical speed [m/s]
        self.id      = []  # identifier (string)

        
    def _get_noisy_pos(self, states):
        self.ntraf = states.ntraf
        self.alt = states.alt.copy()
        self.hdg = states.hdg.copy()
        self.trk = states.hdg.copy()
        self.gs = states.gs.copy()
        self.vs = states.vs.copy()
        self.id = states.id.copy()

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
            lat_noise[i] = y / 111_320

            # Convert meters to degrees for longitude 
            # Longitude degrees depend on the latitude (they get smaller closer to the poles)
            # Use the length of a degree at the current latitude
            lon_noise[i] = x / (111_320 * np.cos(np.deg2rad(mean_lat)))

        # Noisy positions
        self.lat = lat + lat_noise
        self.lon = lon + lon_noise