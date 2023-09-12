import numpy as np
from .sources import FarField1DSourcePlacement, FarField2DSourcePlacement

def get_narrowband_snapshots(array, sources, wavelength, source_signal,
                             noise_signal=None, n_snapshots=1,
                             return_covariance=False):
    r"""Generates snapshots based on the narrowband snapshot model (see
    Chapter 8.1 of [1]).

    Let :math:`\mathbf{A}` be the steering matrix, :math:`\mathbf{s}(t)` be the
    source signal vector, and :math:`\mathbf{n}(t)` be the noise signal matrix.
    Then the snapshots received at the array is given by

    .. math::

        \mathbf{y}(t) = \mathbf{A}\mathbf{s}(t) + \mathbf{n}(t),
        t = 1, 2, ..., N,

    where :math:`N` denotes the number of snapshots.

    Args:
        array (~doatools.model.arrays.ArrayDesign): The array receiving the
            snapshots.
        sources (~doatools.model.sources.SourcePlacement): Source placement.
        wavelength (float): Wavelength of the carrier wave.
        source_signal (~doatools.model.signals.SignalGenerator):
            Source signal generator.
        noise_signal (~doatools.model.signals.SignalGenerator):
            Noise signal generator. Default value is ``None``, meaning no
            additive noise.
        n_snapshots (int): Number of snapshots. Default value is 1.
        return_covariance (bool): If set to ``True``, also returns the sample
            covariance matrix. Default value is ``False``.

    Returns:
        Depending on ``return_covariance``.

        * If ``return_covariance`` is ``False``, returns the snapshots matrix,
          :math:`\mathbf{Y}`, where each column represents a snapshot.
        * If ``return_covariance`` is ``True``, also returns sample covariance
          matrix, which is computed by

          .. math::
              \mathbf{R} = \frac{1}{N} \mathbf{Y} \mathbf{Y}^H.

    References:
        [1] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
    """
    matrix_a = array.steering_matrix(sources, wavelength)  # steering matrix
    matrix_s = source_signal.emit(n_snapshots)  # sources
    matrix_y = matrix_a @ matrix_s
    if noise_signal is not None:
        matrix_n = noise_signal.emit(n_snapshots)  # noise
        matrix_y += matrix_n
    if return_covariance:
        # covariance matirx
        matrix_r = (matrix_y @ matrix_y.conj().T) / n_snapshots
        return matrix_y, matrix_r
    else:
        return matrix_y

def get_wideband_snapshots(array, source, source_signal,
                           noise_signal=None, n_snapshot=1,
                           return_covariance=False):
    c = 3e8  # wave speed
    num_element = array.size  # number of array elements
    array_location = array.actual_element_locations
    num_source = source.size
    if source.units[0] == 'deg':
        source_location = np.deg2rad(source.locations)
    else:
        source_location = source.locations

    # if the source is 1D source
    if isinstance(source, FarField1DSourcePlacement):
        # if 1D array
        if array_location.shape[1] == 1:
            # time delay
            tau = 1 / c * np.outer(array_location,
                                    np.sin(source_location))
        # if 2D or 3D array
        else:
            tau = 1/c * (np.outer(array_location[:, 0], np.sin(source_location))
                      + np.outer(array_location[:, 1], np.cos(source_location)))

    if isinstance(source, FarField2DSourcePlacement):
        # if 1D array
        if array_location.shape[1] == 1:
            # time delay
            tau = 1 / c * np.outer(array_location,
                                    np.sin(source_location))
        # if 2D or 3D array
        else:
            tau = 1/c * (np.outer(array_location[:, 0], np.sin(source_location))
                      + np.outer(array_location[:, 1], np.cos(source_location)))

    array_received = np.zeros((num_element, num_source))
    for element_i in range(num_element):
