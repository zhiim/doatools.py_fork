import sys
sys.path.append('../')

import numpy as np
from doatools.model.sources import NearField2DSourcePlacement

location = np.array([[1, 2],[3, 4]])
sensor = np.array([[0], [1], [2]])
signal = NearField2DSourcePlacement(locations=location)

delay = signal.phase_delay_matrix(sensor_locations=sensor, wavelength=2)
