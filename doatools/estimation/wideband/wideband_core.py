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
