import numpy as np

def divide_wideband_into_sub(signal, n_fft, freq_start, freq_end):
    """Divide sampling points of wideband siganl into p groups, than decompose
    wideband signal in every group into lots of narrowband signals using DFT.

    Args:
        signal (np.ndarray): wideband signal to be decomposed
        n_fft (int): FFT length in every group
        fre_start (int): index of start frequency of wideband signal in FFT out-
            put
        fre_end (int): index of end frequency of wideband signal in FFT output

    Returns:
        np.array: a m*n*p matrix consists of FFT output of p groups (m is the
            number of antennas, n is the number of points in FFT output within
            band of wideband signal)
    """
    num_snapshot = signal.shape[1]
    # divide all sampling points of signal into `num_group` groups,
    # each group has `n_fft` sampling points
    num_group = num_snapshot - n_fft + 1

    # do fft to `n_fft` points in every group
    signal_spectrum = np.zeros((signal.shape[0], (freq_end - freq_start),
                                num_group), dtype=np.complex_)
    for group_i in range(num_group):
        # only use fft output within frequency band of signal
        signal_spectrum[:, :, group_i] = np.fft.fftshift(
        np.fft.fft(signal[:, group_i: group_i + n_fft]))[:, freq_start: freq_end]
    return signal_spectrum
