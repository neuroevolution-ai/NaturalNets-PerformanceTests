from numba import cuda, float32
import math
import numpy as np

m = 100
p = 100

@cuda.jit
def matmul_fast(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    tx = cuda.threadIdx.x

    sA = cuda.shared.array(shape=(m,p), dtype=float32)
    sB = cuda.shared.array(shape=p, dtype=float32)

    for j in range(p):
        sA[tx,j] = A[tx,j]

    if tx == 0:
        for j in range(p):
            sB[j] = B[j]

    # Wait until all threads finish preloading
    cuda.syncthreads()

    for j in range(2*1000):
        Cvalue = 0.0

        if tx < m:
            for i in range(p):
                Cvalue += sA[tx, i] * sB[i]

            C[tx] = Cvalue


@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    tx = cuda.threadIdx.x
    for j in range(2*1000):
        Cvalue = 0.0

        if tx < m:
            for i in range(p):
                Cvalue += A[tx, i] * B[i]

            C[tx] = Cvalue



A = np.random.rand(m, p).astype(np.float32)
B = np.random.rand(p).astype(np.float32)
C = np.zeros(m, dtype=np.float32)

A_d = cuda.to_device(A)
B_d = cuda.to_device(B)
C_d = cuda.to_device(C)

for i in range(100):
    matmul_fast[112, m](A_d, B_d, C_d)
    cuda.synchronize()

C_r = C_d.copy_to_host()

#print(np.matmul(A,B) - C_r)

cuda.profile_stop()
cuda.close()
