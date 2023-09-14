import sys
sys.path.append('../')

import numpy as np
import doatools.model as model
import doatools.estimation as estimation
import doatools.plotting as doaplot


np.random.seed(128)

# Parameters
wavelength = 1.0 # Normalized wavelength. Recall that d0 = wavelength / 2.
d0 = wavelength / 2
power_source = 1.0
power_noise = 1.0 # SNR = 0 dB
n_snapshots = 100

# Create a 12-element ULA
ula = model.UniformLinearArray(12, d0)

# Place 7 far-field narrow-band sources uniformly between (-pi/4, pi/4)
sources = model.FarField1DSourcePlacement(np.linspace(-np.pi/4, np.pi/4, 5))

# Use the stochastic signal model.
source_signal = model.ComplexStochasticSignal(sources.size, power_source)
noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)

# Get the estimated covariance matrix.
y_n = model.get_narrowband_snapshots(ula, sources, wavelength, source_signal,
                                    #   noise_signal=noise_signal,
                                      n_snapshots=n_snapshots,
                                      return_covariance=False)

print(np.mean(np.abs(y_n[1, :]) ** 2))
print(np.mean(np.abs(y_n) ** 2))
