from ..music import MUSIC
from .wideband_core import divide_wideband_into_sub
import numpy as np

class ISSM(MUSIC):
    def __init__(self, array, search_grid):
        self._array = array
        self._search_grid = search_grid

    def estimate(self, signal, fs, f_start, f_end, n_fft, k):

        # index of start frequency point of signal band in fft output
        f_start_index = int(f_start / (fs / n_fft))
        # index of end frequency point of signal band in fft output
        f_end_index = int(f_end / (fs / n_fft))

        signal_spectrum = divide_wideband_into_sub(signal, n_fft, f_start_index,
                                                   f_end_index)
        # frequency points of narrowband signals
        freq_bins = np.linspace(f_start, f_end,
                                (f_end_index - f_start_index))
        spatial_spectrums = np.zeros((n_fft, self._search_grid.size),
                                     dtype=np.complex_)
        for i, freq in enumerate(freq_bins):
            wavelength = 3e8 / freq
            matrix_r = 1 / signal_spectrum.shape[2] *\
                signal_spectrum[:, i, :] @ signal_spectrum[:, i, :].conj().T
            music = MUSIC(self._array, wavelength, self._search_grid)
            spatial_spectrums[i, :] = music.get_spatial_spectrum(matrix_r,
                                                                    k)
        spatial_spectrum =  np.sum(spatial_spectrums, axis=0)
        return spatial_spectrum
