import numpy as np
from .core import SpectrumBasedEstimatorBase, ensure_covariance_size

def _compute_spectrum(matrix_r, num_sources):
    # block the covariance matrix R into G and H
    matrix_g = matrix_r[:, num_sources]
    matrix_h = matrix_r[:, num_sources:]
    # compute the propagation operator P
    matrix_p = np.matmul(np.linalg.inv(np.matmul(matrix_g.H, matrix_g)),
                         np.matmul(matrix_g.H, matrix_h))
    # build the orthogonal matrix Q
    matrix_q = np.concatenate((matrix_p,
                               np.eye(matrix_r.shape[0] - num_sources)),
                               axis=1)

class PM(SpectrumBasedEstimatorBase):
    def __init__(self, array, wavelength, search_grid, **kwargs):
        super().__init__(array, wavelength, search_grid, **kwargs)


    def estimate(self, matrix_r, k, **kwargs):
       # make sure that the size of covariance matrix match the array size
       ensure_covariance_size(matrix_r, self._array)


