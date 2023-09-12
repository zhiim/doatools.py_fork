import sys
sys.path.append('../')

import numpy as np
from doatools.model.signals import PeriodicChirpSignal
import matplotlib.pyplot as plt

f0 = (20, 30)
t1 = (10, 10)
f1 = (50, 60)

s_period=10

pcs = PeriodicChirpSignal(dim=2, f0=f0, f1=f1, t1=t1, s_period=s_period)

s_start=2
fs = 2 * 60

s = pcs.emit(s_start)

t = np.arange(s_start, 10 + s_start, 1 / fs)

S = np.fft.fftshift(np.fft.fft(s[0, :]))
f = np.arange(-fs/2, fs/2, fs/s[0, :].size)

plt.figure(1)
plt.plot(t, s[0, :])
plt.figure(2)
plt.plot(f, np.abs(S))
plt.show()

