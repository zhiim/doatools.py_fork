# music algorithm for near field DOA estimation

import sys
sys.path.append('../../')

import numpy as np
from matplotlib import cm
from doatools.model.arrays import UniformLinearArray
from doatools.model.sources import NearField2DSourcePlacement
from doatools.model.signals import ComplexStochasticSignal
from doatools.model.snapshots import get_narrowband_snapshots
from doatools.estimation.grid import NearField2DSearchGrid
from doatools.estimation.music import MUSIC
import doatools.plotting as doaplot

np.random.seed(128)

# Parameters
wavelength = 1.0 # Normalized wavelength. Recall that d0 = wavelength / 2.
d0 = wavelength / 2
power_source = 1.0
power_noise = 1.0 # SNR = 0 dB
n_snapshots = 300

# Create a 12-element ULA
ula = UniformLinearArray(12, d0)

distance = np.array([5, 10])
angle = np.array([30, 120]) * np.pi / 180

distance = distance.reshape((-1, 1))
angle = angle.reshape((-1, 1))
locations = np.concatenate((distance, angle), axis=1)
sources = NearField2DSourcePlacement(locations=locations)

print(sources.locations)

# Use the stochastic signal model.
source_signal = ComplexStochasticSignal(sources.size, power_source)
noise_signal = ComplexStochasticSignal(ula.size, power_noise)

# Get the estimated covariance matrix.
_, R = get_narrowband_snapshots(ula, sources, wavelength, source_signal,
                                noise_signal, n_snapshots,
                                return_covariance=True)

# Create a MUSIC-based estimator.
grid_start = (0, 0)
grid_stop = (20, np.pi)
grid = NearField2DSearchGrid(start=grid_start, stop=grid_stop, size=(180, 100))
estimator = MUSIC(ula, wavelength, grid)

# Get the estimates.
resolved, estimates, sp = estimator.estimate(R, sources.size,
                                             return_spectrum=True)
print('Estimates: {0}'.format(estimates.locations))
print('Ground truth: {0}'.format(sources.locations))

# Plot the MUSIC-spectrum.
# doaplot.plot_spectrum({'MUSIC': sp}, grid, estimates=estimates,
#                       ground_truth=sources, use_log_scale=True)
doaplot.plot_spectrum(sp, grid, ground_truth=sources,
                      use_log_scale=True, plot_2d_mode='surface',
                      color_map=cm.coolwarm)
