from numba import cuda, float32
import math
import numpy as np

@cuda.jit
def increment_by_one(an_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < an_array.size:  # Check array boundaries
        an_array[pos] += 1



ary = np.arange(10)

print(ary)

d_ary = cuda.to_device(ary)

threadsperblock = 32
blockspergrid = (d_ary.size + (threadsperblock - 1)) // threadsperblock
increment_by_one[blockspergrid, threadsperblock](d_ary)

hary = d_ary.copy_to_host()

print(hary)
