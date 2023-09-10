import sys
sys.path.append('../')

import numpy as np
from doatools.model.signals import ChirpSignal
import matplotlib.pyplot as plt

f0 = (20, 30)
f1 = (40, 50)
fs = 4 * 50

chr = ChirpSignal(dim=2)
s = chr.emit(500, f0=f0, f1=f1, fs=fs)

spectrum = np.fft.fftshift(np.fft.fft(s[0, :]))
f = np.arange(-fs/2, fs/2, fs/spectrum.size)

# plt.plot(t, w)
plt.plot(f, np.abs(spectrum))
plt.show()

