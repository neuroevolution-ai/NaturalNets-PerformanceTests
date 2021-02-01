import numpy as np
import time
from scipy.sparse import csr_matrix

inputs = 64*64*3
N = 200

A = np.random.rand(N, inputs)
x = np.random.rand(inputs, 1)

#k=0
#for i in range(N):
#    for j in range(inputs):
#        if k % 2000 != 0:
#            A[i, j] = 0
#        k=k+1

start = time.time()

for i in range(1000):
    np.matmul(A,x)
end = time.time()
print("Numpy: " + str(end - start) + " ms")

A = csr_matrix(A,dtype=float)

start = time.time()

for i in range(1000):
    A.dot(x)

end = time.time()
print("Scipy: " + str(end - start) + " ms")