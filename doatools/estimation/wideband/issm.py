from ..music import MUSIC
from .wideband_core import divide_wideband_into_sub, get_estimates_from_sp
import numpy as np

class ISSM(MUSIC):
    """Incoherent Signal Subspace Method (ISSM) estimator for wideband DOA esti-
    mation.

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        search_grid (~doatools.estimation.grid.SearchGrid): The search grid
            used to locate the sources.
        **kwargs: Other keyword arguments supported by
            :class:`~doatools.estimation.core.SpectrumBasedEstimatorBase`.
    """
    def __init__(self, array, search_grid, **kwargs):
        # we need to specific wavelength for every frequency points,
        # so we'll pass wavelength later
        wavelength = None

        # `enable_caching` should be set to False to recompute atom matrix for
        # every frequency point
        super().__init__(array, wavelength, search_grid, enable_caching=False,
                         **kwargs)

    def _spatial_spectrum(self, signal, fs, f_start, f_end, n_fft, k):
        """Get spatial spectrum using ISSM"""
        # divide wideband signal into frequency points
        signal_subs, freq_bins = divide_wideband_into_sub(signal, n_fft, fs,
                                                          f_start, f_end)

        spatial_spectrums = np.zeros((n_fft, self._search_grid.size),
                                     dtype=np.complex_)
        for i, freq in enumerate(freq_bins):
            wavelength = 3e8 / freq  # wavelength of every frequency points
            self._wavelength = wavelength

            # compute covariance matrix under every frequency points
            matrix_r = 1 / signal_subs.shape[2] *\
                signal_subs[:, i, :] @ signal_subs[:, i, :].conj().T

            # compute spatial spectrum under every frequency points
            spatial_spectrums[i, :] = super()._spatial_spectrum(matrix_r,
                                                                    k)

        # Average all spatial spectrums to obtain the final spatial spectrum
        spatial_spectrum =  np.sum(spatial_spectrums, axis=0)
        return spatial_spectrum

    def estimate(self, signal, fs, f_start, f_end, n_fft, k,
                 return_spectrum=True):
        """Get DOA estimation using ISSM.

        Args:
            signal (np.array): sampled wideband signal.
            fs (float): sampling frequency.
            f_start (float): start frequency of wideband signal.
            f_end (float): end frequency of wideband signal.
            n_fft (int): number of points of FFT.
            k (int): number of sources.
            return_spectrum (bool, optional): return spatial spectrum or not.
                Defaults to True.

        Returns:
            A tuple with the following elements.

            * resolved (:class:`bool`): A boolean indicating if the desired
              number of sources are found. This flag does **not** guarantee that
              the estimated source locations are correct. The estimated source
              locations may be completely wrong!
              If resolved is False, both ``estimates`` and ``spectrum`` will be
              ``None``.
            * estimates (:class:`~doatools.model.sources.SourcePlacement`):
              A :class:`~doatools.model.sources.SourcePlacement` instance of the
              same type as the one used in the search grid, represeting the
              estimated source locations. Will be ``None`` if resolved is
              ``False``.
            * spectrum (:class:`~numpy.ndarray`): An numpy array of the same
              shape of the specified search grid, consisting of values evaluated
              at the grid points. Only present if ``return_spectrum`` is
              ``True``.
        """
        sp = self._spatial_spectrum(signal, fs, f_start, f_end, n_fft, k)

        return get_estimates_from_sp(sp=sp, k=k, search_grid=self._search_grid,
                                     peak_finder=self._peak_finder,
                                     return_spectrum=return_spectrum)
