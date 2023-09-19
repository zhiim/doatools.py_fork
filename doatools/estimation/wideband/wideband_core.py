import numpy as np

def divide_wideband_into_sub(signal, n_fft, fs, f_start, f_end):
    """Divide sampling points of wideband siganl into p groups, than decompose
    wideband signal in every group into lots of narrowband signals using DFT.

    Args:
        signal (np.ndarray): wideband signal to be decomposed
        n_fft (int): FFT length in every group
        f_start (float): start frequency of wideband signal
        f_end (float): end frequency of wideband signal
        fs (float): sampling frequency of wideband signal

    Returns:
        a m*n*p matrix consists of FFT output of p groups (m is the number of
        antennas, n is the number of points in FFT output within band of wide-
        band signal) and a array consists of freqency points of within band of
        wideband signal in FFT output.
    """
    num_snapshot = signal.shape[1]
    # divide all sampling points of signal into `num_group` groups,
    # each group has `n_fft` sampling points
    num_group = num_snapshot // n_fft

    # index of start frequency point of signal band in fft output
    f_start_index = int(f_start / (fs / n_fft))
    # index of end frequency point of signal band in fft output
    f_end_index = int(f_end / (fs / n_fft))

    # frequency points of narrowband signals
    freq_bins = np.linspace(f_start, f_end,
                            (f_end_index - f_start_index))

    # do fft to `n_fft` points in every group
    signal_subs = np.zeros((signal.shape[0], (f_end_index - f_start_index),
                                num_group), dtype=np.complex_)
    for group_i in range(num_group):
        # only use fft output within frequency band of signal
        signal_subs[:, :, group_i] = np.fft.fft(
            signal[:, group_i * n_fft: (group_i + 1) * n_fft])\
                                                [:, f_start_index: f_end_index]
    return signal_subs, freq_bins

def get_estimates_from_sp(sp, k, search_grid, peak_finder, return_spectrum):
    """Get DOA estimation from spatial spectrum.

    Args:
        sp (np.array): spatial spectrum.
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
    # Find peak locations.
    peak_indices = peak_finder(np.abs(sp))
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
                                                    search_grid.shape)
        flattened_indices.sort()
        estimates = search_grid.source_placement[flattened_indices]

        if return_spectrum:
            return True, estimates, sp
        else:
            return True, estimates
