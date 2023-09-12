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
        f0 (tuple | np.array): start frequency of every chirp signal.
        f1 (tuple | np.array): end frequency of every chirp signal.
        t1 (tuple | np.array): how much time it takes to reach f1 from f0
            for every cirp signal.
        s_period (int): the period of time the sampling lasts. `s_period`
            should be no less than `t1`.
        amplitudes: Amplitudes of the signal. Can be specified by

            1. A scalar if all sources have the same amplitude.
            2. A vector if the sources have different amplitudes.
        method (str, optional): kind of frequency sweep, can be specified as
            {'linear', 'quadratic', 'logarithmic', 'hyperbolic'}. Defaults
            to 'linear'.

    Raises:
        ValueError: `amplitudes` has a wrong dimension which isn't match with
            `dim`
    """
    def __init__(self, dim, f0, f1, t1, s_period, amplitudes=1.0,
                 method='linear'):
        self._dim = dim

        if np.isscalar(amplitudes):
            self._amplitudes = np.full((dim, 1), amplitudes)
        else:
            if amplitudes.size != dim:
                raise ValueError("The size of 'amplitudes' does not match the\
                                  value of 'dim'.")
            self._amplitudes = amplitudes.reshape((-1, 1))

        # if the sampling time period less than t1, we can not get a full
        # frequency swept from f0 to f1
        if s_period < max(t1):
            raise ValueError("Sampling period less than t1, can't sweep full\
                              frequency range.")
        self._f0 = f0
        self._f1 = f1
        self._t1 = t1
        self._s_period = s_period
        self._method = method

    @property
    def dim(self):
        return self._dim

    def emit(self, s_start=0, fs=None):
        """Generates a k x n matrix where k is the dimension of the signal and
        each column represents a sample.

        Args:
            s_start (int): time point when sampling of chirp signal start.
            fs (float, optional): sampling frequency (at least twice the maxim-
                um of f0 and f1). Defaults to twice the maximum of f0 and f1.

        Returns:
            numpy.ndarray: sampled chirp signals.
        """
        # if fs is not specified, set fs to twice the maximum of f0 and f1
        if fs is None:
            fs = 2 * max(max(self._f0), max(self._f1))
        if s_start < 0:
            raise ValueError("Sampling of chirp signal should start after time\
                              0.")

        signal = np.zeros((self._dim, self._s_period * fs))
        # generate sampled chirp signal
        for dim_i in range(self._dim):
            # 1. sampling from s_start to t1
            time_1 = np.arange(s_start, self._t1[dim_i], 1 / fs)
            s = chirp(t=time_1, f0=self._f0[dim_i], f1=self._f1[dim_i],
                                 t1=self._t1[dim_i],
                                 method=self._method)
            # 2. sampling every full periods after t1
            period_num = (self._s_period - (self._t1[dim_i] - s_start)) //\
                  self._t1[dim_i]
            if period_num > 0:
                for i in range(period_num):
                    s = np.concatenate((s, chirp(t=np.arange(0, self._t1[dim_i],
                                                             1/fs),
                                                 f0=self._f0[dim_i],
                                                 t1=self._t1[dim_i],
                                                 f1=self._f1[dim_i],
                                                 method=self._method)))
            # 3. sampling remainder
            remainder = (self._s_period - (self._t1[dim_i] - s_start)) %\
                        self._t1[dim_i]
            if remainder > 0:
                s = np.concatenate((s, chirp(t=np.arange(0, remainder, 1 / fs),
                                            f0=self._f0[dim_i],
                                            t1=self._t1[dim_i],
                                            f1=self._f1[dim_i],
                                            method=self._method)))
            signal[dim_i, :] = s

        return self._amplitudes * signal
