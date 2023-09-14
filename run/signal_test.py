import sys
sys.path.append('../')

import numpy as np
from doatools.model.signals import PeriodicChirpSignal
import matplotlib.pyplot as plt

f0 = (20, 30)
t1 = (10, 10)
f1 = (50, 60)
f_start = min(f0)
f_end = max(f1)

s_period = max(t1)
fs = 4 * f_end


pcs = PeriodicChirpSignal(dim=2, f0=f0, f1=f1, t1=t1, s_period=s_period, fs=fs)

s = pcs.emit()
num_snapshot = s.shape[1]

s_fft = np.fft.fft(s[0, :])

f = np.arange(0, int(num_snapshot/2)) * fs / num_snapshot

plt.plot(np.abs(s_fft))
plt.show()
