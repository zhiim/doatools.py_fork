import numpy as np

snr = 20
sensor_num = 8
theta_s = (-np.pi / 6, np.pi / 6)
source_num = len(theta_s)

f0 = (6e6, 10e6)
f1 = (14e6, 14e6)
fc = 10e6
fs = 2 * max(f1)

freq_snapshots = 100
nfft = 512
snapshots = freq_snapshots * nfft
ts = (1/fs) * np.arange(0, snapshots - 1) + 0.005

c = 3e8
d0 = (c/max(f0)) / 2

received_data = np.zeros((sensor_num, snapshots), dtype=np.complex_)
for m in range(sensor_num):
    for n in range(source_num):
        delay = m * d0 * np.sin(theta_s[n]) / c
        k = (f1[n] - f0[n]) / ts[-1]
        received_data[m, :] += np.exp(1j * 2 * np.pi * (f0[n] * (ts - delay) + 
                                                        0.5 * k * (ts - delay)**2))

dataset = np.zeros((sensor_num, freq_snapshots), nfft)
