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

f0 = (2e8, 3e8)
t1 = (1e-6, 1e-6)
f1 = (5e8, 6e8)
f_start = min(f0)
f_end = max(f1)

s_period = 2 * max(t1)
fs = 2 * f_end

d0 = 3e8 / f_end / 2
ula = UniformLinearArray(n=3, d0=d0)

source = FarField1DSourcePlacement([-np.pi / 3, np.pi / 6])

pcs = PeriodicChirpSignal(dim=2, f0=f0, f1=f1, t1=t1, s_period=s_period, fs=fs)

received = get_wideband_snapshots(array=ula, source=source, source_signal=pcs)

grid = FarField1DSearchGrid()
estimator = ISSM(array=ula, search_grid=grid)

num_snapshot = received[0, :].size

n_fft = 2048

freq_start_index = int((f_start - (-fs / 2)) / (fs / n_fft))
freq_end_index = int((f_end - (-fs / 2)) / (fs / n_fft))
# frequency points of narrowband signals
freq_bins = np.linspace(f_start, f_end,
                        (freq_end_index - freq_start_index))

sp = estimator.estimate(received, freq_start=freq_start_index,
                        freq_end=freq_end_index, freq_bins=freq_bins,
                        n_fft=2048, k=2)

plt.plot(np.arange(-90, 90), sp)
plt.show()
