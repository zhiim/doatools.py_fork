from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import sqrtm
from scipy.signal import chirp
from ..utils.math import randcn

class SignalGenerator(ABC):
    """Abstrace base class for all signal generators.

    Extend this class to create your own signal generators.
    """

    @property
    @abstractmethod
    def dim(self):
        """Retrieves the dimension of the signal generator."""
        pass

    @abstractmethod
    def emit(self, n):
        """Emits the signal matrix.

        Generates a k x n matrix where k is the dimension of the signal and
        each column represents a sample (n is the number of snapshots).
        """
        pass

class ComplexStochasticSignal(SignalGenerator):
    """Creates a signal generator that generates zero-mean complex
    circularly-symmetric Gaussian signals.

    Args:
        dim (int): Dimension of the complex Gaussian distribution. Must match
            the size of ``C`` if ``C`` is not a scalar.
        C: Covariance matrix of the complex Gaussian distribution.
            Can be specified by

            1. A full covariance matrix. (related sources)
            2. An real vector (k x 1) denoting the diagonals of the covariance
               matrix if the covariance matrix is diagonal. (unrelated sources)
            3. A scalar if the covariance matrix is diagonal and all diagonal
               elements share the same value. In this case, parameter n must be
               specified. (unrelated sources with a same power)
    """

    def __init__(self, dim, C):
        self._dim = dim
        if np.isscalar(C):
            # Scalar
            self._C2 = np.sqrt(C)  # amplitude of signals
            self._generator = lambda n: self._C2 * randcn((self._dim, n))
        elif C.ndim == 1:
            # Vector
            if C.size != dim:
                raise ValueError('The size of C must be {0}.'.format(dim))
            self._C2 = np.sqrt(C).reshape((-1, 1))
            self._generator = lambda n: self._C2 * randcn((self._dim, n))
        elif C.ndim == 2:
            # Matrix
            if C.shape[0] != dim or C.shape[1] != dim:
                raise ValueError('The shape of C must be ({0}, {0}).'
                                 .format(dim))
            self._C2 = sqrtm(C)
            self._generator = lambda n: self._C2 @ randcn((self._dim, n))
        else:
            raise ValueError(
                'The covariance must be specified by a scalar, a vector of'
                'size {0}, or a matrix of {0}x{0}.'.format(dim)
            )
        self._C = C

    @property
    def dim(self):
        return self._dim

    def emit(self, n):
        return self._generator(n)

class RandomPhaseSignal(SignalGenerator):
    r"""Creates a random phase signal generator.

    The phases are uniformly and independently sampled from :math:`[-\pi, \pi]`.

    Args:
        dim (int): Dimension of the signal (usually equal to the number of
            sources).
        amplitudes: Amplitudes of the signal. Can be specified by

            1. A scalar if all sources have the same amplitude.
            2. A vector if the sources have different amplitudes.
    """

    def __init__(self, dim, amplitudes=1.0):
        self._dim = dim
        if np.isscalar(amplitudes):
            self._amplitudes = np.full((dim, 1), amplitudes)
        else:
            if amplitudes.size != dim:
                raise ValueError("The size of 'amplitudes' does not match the\
                                  value of 'dim'.")
            self._amplitudes = amplitudes.reshape((-1, 1))

    @property
    def dim(self):
        return self._dim

    def emit(self, n):
        phases = np.random.uniform(-np.pi, np.pi, (self._dim, n))
        c = np.sin(phases) * 1j
        c += np.cos(phases)
        return self._amplitudes * c

class PeriodicChirpSignal(SignalGenerator):
    """Generate periodic chirp signal (Frequency-swept signal) as the incident
    signal in wideband DOA estimation.

    Args:
        dim (int): Dimension of the signal (usually equal to the number of
            sources).
        amplitudes: Amplitudes of the signal. Can be specified by

            1. A scalar if all sources have the same amplitude.
            2. A vector if the sources have different amplitudes.

    Raises:
        ValueError: `amplitudes` has a wrong dimension which isn't match with
            `dim`
    """
    def __init__(self, dim, amplitudes=1.0):
        self._dim = dim

        if np.isscalar(amplitudes):
            self._amplitudes = np.full((dim, 1), amplitudes)
        else:
            if amplitudes.size != dim:
                raise ValueError("The size of 'amplitudes' does not match the\
                                  value of 'dim'.")
            self._amplitudes = amplitudes.reshape((-1, 1))

    @property
    def dim(self):
        return self._dim

    def emit(self, f0, f1, t1, s_start, s_period, fs=None, method='linear'):
        """Generates a k x n matrix where k is the dimension of the signal and
        each column represents a sample.

        Args:
            f0 (float): start frequency of chirp signal.
            f1 (float): end frequency of chirp signal.
            t1 (int): how much time it takes to reach f1 from f0 (seconds)
            s_start (int): time point when sampling of chirp signal start.
            s_period (int): the period of time the sampling lasts. `s_period`
                should be no less than `t1`.
            fs (float, optional): sampling frequency (at least twice the maxim-
                um of f0 and f1). Defaults to twice the maximum of f0 and f1.
            method (str, optional): kind of frequency sweep, can be specified as
                {'linear', 'quadratic', 'logarithmic', 'hyperbolic'}. Defaults
                to 'linear'.

        Returns:
            numpy.ndarray: sampled chirp signals.
        """
        # if fs is not specified, set fs to twice the maximum of f0 and f1
        if fs is None:
            fs = 2 * max(max(f0), max(f1))
        if s_start < 0:
            raise ValueError("Sampling of chirp signal should start after time\
                              0.")
        # if the sampling time period less than t1, we can not get a full
        # frequency sweept from f0 to f1
        if s_period < t1:
            s_period = s_start + t1

        signal = np.zeros((self._dim, t.size))
        # generate sampled chirp signal
        for i in np.arange(self._dim):
            # 1. sampling from s_start to t1
            time_1 = np.arange(s_start, t1, 1 / fs)
            s1 = chirp(t=time_1, f0=f0[i], f1=f1[i], t1=t1,
                                 method=method)
            # 2. sampling every full periods after t1
            period_num = (s_period - (t1 - s_start)) / t1
        return self._amplitudes * signal
