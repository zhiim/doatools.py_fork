import sys
sys.path.append('../')

import numpy as np
from doatools.model.signals import PeriodicChirpSignal
from doatools.model.arrays import UniformLinearArray
from doatools.model.sources import FarField1DSourcePlacement
from doatools.estimation.grid import FarField1DSearchGrid
from doatools.estimation.wideband.issm import ISSM
from doatools.model.snapshots import get_wideband_snapshots
import matplotlib.pyplot as plt

f0 = (6e6, 10e6)
f1 = (14e6, 14e6)
f_start = min(f0)
f_end = max(f1)

fs = 2 * f_end

n_fft = 512

t1 = (1e-4, 1e-4)
s_period = max(t1)

d0 = 3e8 / f_end / 2

ula = UniformLinearArray(n=6, d0=d0)

source = FarField1DSourcePlacement([-np.pi/180 * 25, np.pi / 180 * 50])

pcs = PeriodicChirpSignal(dim=2, f0=f0, f1=f1, t1=t1, s_period=s_period, fs=fs)

received = get_wideband_snapshots(array=ula, source=source, source_signal=pcs, add_noise=True, snr=0)

grid = FarField1DSearchGrid()
estimator = ISSM(array=ula, search_grid=grid)

num_snapshot = received[0, :].size

sp = estimator.estimate(received, fs=fs, f_start=f_start, f_end=f_end,
                        n_fft=n_fft, k=2)

plt.plot(np.arange(-90, 90), np.abs(sp))
plt.show()
