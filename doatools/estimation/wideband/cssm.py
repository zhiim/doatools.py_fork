import numpy as np
from ..music import MUSIC
from .wideband_core import divide_wideband_into_sub

class CSSM(MUSIC):
    def __init__(self, array, search_grid, **kwargs):
        wavelength = None
        super().__init__(array, wavelength, search_grid, enable_caching=False,
                         **kwargs)
    def _spatial_spectrum(self, signal, fs, f_start, f_end, pre_estimate,
                          n_fft, k):
        # 选取频带的中心频率作为参考频点
        f_reference = (f_start + f_end) / 2
        # divide wideband signal into frequency points
        signal_subs, freq_bins = divide_wideband_into_sub(signal=signal,
                                                          n_fft=n_fft, fs=fs,
                                                          f_start=f_start,
                                                          f_end=f_end)
        matrix_r = np.zeros((self._array.size, self._array.size),
                            dtype=np.complex_)  # 协方差矩阵
        for i, freq in enumerate(freq_bins):
            

