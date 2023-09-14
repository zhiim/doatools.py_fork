from ..music import MUSIC
from .wideband_core import divide_wideband_into_sub
import numpy as np

class ISSM(MUSIC):
    def __init__(self, array, search_grid):
        self._array = array
        self._search_grid = search_grid

    def estimate(self, signal, freq_start, freq_end, freq_bins, n_fft, k):

        signal_spectrum = divide_wideband_into_sub(signal, n_fft,
                                                   freq_start,
                                                   freq_end)
        spatial_spectrums = np.zeros((freq_bins.size, self._search_grid.size),
                                     dtype=np.complex_)
        for i, freq in enumerate(freq_bins):
            wavelength = 3e8 / freq
            matrix_r = 1 / signal_spectrum.shape[2] *\
                  signal_spectrum[:, i, :] @ signal_spectrum[:, i, :].conj().T
            music = MUSIC(self._array, wavelength, self._search_grid)
            spatial_spectrums[i, :] = music.get_spatial_spectrum(matrix_r,
                                                                    k)
        spatial_spectrum = 1 / np.sum(spatial_spectrums, axis=0)
        return spatial_spectrum
