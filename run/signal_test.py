import sys
sys.path.append('../')

import numpy as np
from doatools.model.signals import PeriodicChirpSignal
import matplotlib.pyplot as plt

f0 = (20, 20)
t1 = (10, 10)
f1 = (50, 50)
f_start = min(f0)
f_end = max(f1)

s_period = max(t1)
fs = 4 * f_end
s_start = (5, 0)

pcs = PeriodicChirpSignal(dim=2, f0=f0, f1=f1, t1=t1, s_period=s_period, fs=fs)

s = pcs.emit(s_start=s_start)
num_snapshot = s.shape[1]

s_fft = np.fft.fftshift(np.fft.fft(s[0, :].real))
s_fft_2 = np.fft.fftshift(np.fft.fft(s[1, :].real))

f = np.arange(-fs/2, fs/2, fs/num_snapshot)

plt.figure(1)
plt.plot(f, np.abs(s_fft))
plt.figure(2)
plt.plot(f, np.abs(s_fft_2))
# plt.plot(s[0, :].real)
plt.show()
