import sys
sys.path.append('../')

import numpy as np
from doatools.model.signals import PeriodicChirpSignal
from doatools.model.arrays import UniformLinearArray
from doatools.model.sources import FarField1DSourcePlacement
from doatools.model.snapshots import get_wideband_snapshots
import matplotlib.pyplot as plt

d0 = 1
ula = UniformLinearArray(n=3, d0=d0)

source = FarField1DSourcePlacement([np.pi / 6, np.pi / 3])

f0 = (20, 30)
t1 = (10, 10)
f1 = (50, 60)

s_period=10
fs = 4 * 60

pcs = PeriodicChirpSignal(dim=2, f0=f0, f1=f1, t1=t1, s_period=s_period, fs=fs)

received = get_wideband_snapshots(array=ula, source=source, source_signal=pcs)

plt.plot(np.abs(np.fft.fftshift(np.fft.fft(received[0, :]))))
plt.show()
