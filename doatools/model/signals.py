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

class ChirpSignal(SignalGenerator):
    """Generate chirp signal (Frequency-swept signal) as the incident signal in
    wideband DOA estimation.

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

    def emit(self, n, f0, f1, fs=None, method='linear'):
        """Generates a k x n matrix where k is the dimension of the signal and
        each column represents a sample.

        Args:
            n (int): number of snapshot.
            f0 (float): start frequency of chirp signal.
            f1 (float): end frequency of chirp signal.
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

        # time the signal will be sample
        t = np.arange(n) * 1 / fs

        signal = np.zeros((self._dim, n))
        # generate sampled chirp signal
        for i in np.arange(self._dim):
            signal[i, :] = chirp(t=t, f0=f0[i], f1=f1[i], t1=t[-1],
                                 method=method)
        return self._amplitudes * signal
