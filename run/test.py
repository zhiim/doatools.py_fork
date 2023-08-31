import sys
sys.path.append('../')

from doatools.model.arrays import NestedArray
from doatools.model.coarray import compute_location_differences, \
    WeightFunction1D

nested = NestedArray(n1=3, n2=3, d0=1)

diff = compute_location_differences(nested.element_locations)

weight_fun = WeightFunction1D(nested)

print(weight_fun.differences())
print(weight_fun.get_central_ula_size(True))

