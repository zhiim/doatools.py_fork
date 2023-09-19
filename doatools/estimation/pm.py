import numpy as np
from .core import SpectrumBasedEstimatorBase, ensure_covariance_size

def _compute_spectrum(grid_matrix_a, matrix_r, num_sources):
    """Compute spatial spectrum using PM algorithm

    Args:
        grid_matrix_a (~numpy.ndarray): the steering matrix of sources located
            in all search grids, which is used to compute spatial spectrum
        matrix_r (~numpy.ndarray): covariance matrix of sampled signal received
            by antenna array
        num_sources (int): number of sources

    Returns:
        numpy.ndarray: spatial spectrum in all search girds

    References:
        [1] Marcos S, Marsal A, Benidir M. The propagator method for source
        bearing estimation[J]. Signal Processing, 1995, 42(2): 121-138.
    """
    # block the covariance matrix R into G and H
    matrix_g = matrix_r[:, 0:num_sources]
    matrix_h = matrix_r[:, num_sources:]
    # compute the propagation operator P
    matrix_p = (np.linalg.inv(matrix_g.T.conj() @ matrix_g)) @\
                         (matrix_g.T.conj() @ matrix_h)
    # build the orthogonal matrix Q
    matrix_q_h = np.concatenate((matrix_p.T.conj(),
                               -np.eye(matrix_r.shape[0] - num_sources)),
                               axis=1)
    matrix_v = matrix_q_h @ grid_matrix_a
    return np.reciprocal(np.absolute(
        np.sum(matrix_v * matrix_v.conj(), axis=0)
    ))

class PM(SpectrumBasedEstimatorBase):
    """Propagator Method (PM) algorithm used for DOA estimation

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid): The search grid
            used to locate the sources.
        **kwargs: Other keyword arguments supported by
            :class:`~doatools.estimation.core.SpectrumBasedEstimatorBase`.
    """
    def __init__(self, array, wavelength, search_grid, **kwargs):
        super().__init__(array, wavelength, search_grid, **kwargs)

    def estimate(self, matrix_r, k, **kwargs):
        """Get the estimated DOAs using PM algorithm

        Args:
            matrix_r (~numpy.ndarray): Covariance matrix input. The size of R
                must match that of the array design used when creating this
                estimator.
            k (int): Expected number of sources.
            return_spectrum (bool): Set to ``True`` to also output the spectrum
                for visualization. Default value if ``False``.
            refine_estimates (bool): Set to True to enable grid refinement to
                obtain potentially more accurate estimates.
            refinement_density (int): Density of the refinement grids. Higher
                density values lead to denser refinement grids and increased
                computational complexity. Default value is 10.
            refinement_iters (int): Number of refinement iterations. More
                iterations generally lead to better results, at the cost of
                increased computational complexity. Default value is 3.

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
        # make sure that the size of covariance matrix match the array size
        ensure_covariance_size(matrix_r, self._array)
        return self._estimate(
            f_sp=lambda grid_matrix_a: _compute_spectrum(grid_matrix_a,
                                                         matrix_r,
                                                         k),
            k=k,
            **kwargs
        )

