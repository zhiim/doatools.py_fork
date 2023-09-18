from ..music import MUSIC
from .wideband_core import divide_wideband_into_sub
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
        """Compute spatial spectrum."""
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
        """Get DOA estimation using ISSM algorithm.

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

        # Find peak locations.
        peak_indices = self._peak_finder(np.abs(sp))
        # The peak finder returns a tuple whose length is at least one. Hence
        # we can get the number of peaks by checking the length of the first
        # element in the tuple.
        n_peaks = len(peak_indices[0])

        if n_peaks < k:
            # Not enough peaks.
            if return_spectrum:
                return False, None, sp
            else:
                return False, None
        else:
            # Obtain the peak values for sorting. Remember that `peak_indices`
            # is a tuple of 1D numpy arrays, and `sp` has been reshaped.
            peak_values = sp[peak_indices]
            # Identify the k largest peaks.
            top_indices = np.argsort(peak_values)[-k:]
            # Filter out the peak indices of the k largest peaks.
            peak_indices = [axis[top_indices] for axis in peak_indices]
            # Obtain the estimates.
            # Note that we need to convert n-d indices to flattened indices.
            # We sorted the flattened indices here to respect the ordering of
            # source locations in the search grid.
            flattened_indices = np.ravel_multi_index(peak_indices,
                                                     self._search_grid.shape)
            flattened_indices.sort()
            estimates = self._search_grid.source_placement[flattened_indices]

            if return_spectrum:
                return True, estimates, sp
            else:
                return True, estimates
