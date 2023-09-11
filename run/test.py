import sys
sys.path.append('../')

import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt

f0 = 20
f1 = 40
t1 = 10
fs = 4 * f1
t = np.arange(5, 15, 1/fs)

s = chirp(t, f0, t1, f1)

S = np.fft.fftshift(np.fft.fft(s))
f = np.arange(-fs/2, fs/2, fs/t.size)

plt.figure(1)
plt.plot(t, s)
plt.figure(2)
plt.plot(f, np.abs(S))
plt.show()

