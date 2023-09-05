# MUSIC algorithm under uniform ractangle array

import sys
sys.path.append('../../')

import numpy as np
import doatools.model as model
import doatools.estimation as estimation
import doatools.plotting as doaplot
from matplotlib import cm

np.random.seed(128)

# Parameters
wavelength = 1.0 # Normalized wavelength. Recall that d0 = wavelength / 2.
d0 = wavelength / 2
power_source = 1.0
power_noise = 1.0 # SNR = 0 dB
n_snapshots = 100

# Create a 8x8 uniform rectangle array
ura = model.UniformRectangularArray(m=8, n=8, d0=d0)

# Place 2 far-field 2D narrow-band sources
# 2 sources with locations in (10, 20) and (70, 60) in degree
sources_loaction = np.array([[10, 20], [70, 60]]) / 180 * np.pi
sources = model.FarField2DSourcePlacement(locations=sources_loaction,
                                          unit='rad')

# Use the stochastic signal model.
source_signal = model.ComplexStochasticSignal(sources.size, power_source)
noise_signal = model.ComplexStochasticSignal(ura.size, power_noise)

# Get the estimated covariance matrix.
_, R = model.get_narrowband_snapshots(ura, sources, wavelength, source_signal,
                                      noise_signal, n_snapshots,
                                      return_covariance=True)

# Create a MUSIC-based estimator.
grid = estimation.FarField2DSearchGrid()  # use default attributes
estimator = estimation.MUSIC(ura, wavelength, grid)

# Get the estimates.
resolved, estimates, sp = estimator.estimate(R, sources.size,
                                             return_spectrum=True)
print('Estimates: {0}'.format(estimates.locations))
print('Ground truth: {0}'.format(sources.locations))

# Plot the MUSIC-spectrum.
doaplot.plot_spectrum(sp, grid, ground_truth=sources,
                      use_log_scale=True, plot_2d_mode='surface',
                      color_map=cm.coolwarm)
