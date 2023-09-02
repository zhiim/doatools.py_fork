import sys
sys.path.append('../')

import numpy as np
from doatools.model.arrays import GridBasedArrayDesign
from doatools.model.coarray import WeightFunction1D

array_indice = np.array([0, 1, 4]).reshape(-1, 1)
array = GridBasedArrayDesign(indices=array_indice, d0=1)

weight_fun = WeightFunction1D(array=array)

print(weight_fun.get_coarray_selection_matrix())

