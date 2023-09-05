# compare degree of freedom of co-arrays drived from ula, nested array and
# coprime array

import sys
sys.path.append('../../')

import doatools.model as model
from doatools.plotting.plot_array import plot_array, plot_coarray
import matplotlib.pyplot as plt

wavelength = 1.0 # Normalized
d0 = wavelength / 2.0

# Create some 1D arrays.
ula = model.UniformLinearArray(10, d0)
cpa = model.CoPrimeArray(2, 3, d0)
nea = model.NestedArray(3, 3, d0)

# Visualize these arrays and their difference coarrays.
arrays_1d = [ula, cpa, nea]
plt.figure(figsize=(8, 6))
for i, array in enumerate(arrays_1d):
    ax = plt.subplot(len(arrays_1d), 2, i * 2 + 1)
    plot_array(array, ax=ax)
    ax.set_title(array.name)
    ax = plt.subplot(len(arrays_1d), 2, i * 2 + 2)
    plot_coarray(array, ax=ax)
    ax.set_title('Coarray of ' + array.name)
plt.tight_layout()
plt.show()
