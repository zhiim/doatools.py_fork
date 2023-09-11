import sys
sys.path.append('../')

import numpy as np
from doatools.model.signals import PeriodicChirpSignal
import matplotlib.pyplot as plt

f0 = (20, 30)
t1 = (8, 10)
f1 = (50, 60)

pcs = PeriodicChirpSignal(2)

s_start=2
s_period=10
fs = 2 * 60

s = pcs.emit(f0, f1, t1, s_start, s_period)

t = np.arange(s_start, 10 + s_start, 1 / fs)

S = np.fft.fftshift(np.fft.fft(s[0, :]))
f = np.arange(-fs/2, fs/2, fs/s[0, :].size)

plt.figure(1)
plt.plot(t, s[0, :])
plt.figure(2)
plt.plot(f, np.abs(S))
plt.show()

