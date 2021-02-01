import time
from numba import jit
import numba
import numpy as np
import random
from scipy.sparse import csr_matrix


#@jit(nopython=True)
def step(ob, y, V, W, T, delta_t):

    # Differential equation
    dydt = np.dot(W, y) + np.dot(V, ob)

    # Euler forward discretization
    y = y + delta_t * dydt

    return np.dot(y, T)

#@jit(nopython=True)
def mult(A, B):

    A[300:600, 100:400] = B

    return


@jit(nopython=True)
def tensordot(x, y, kernel, kernel_size, input):

    dotproduct = 0

    for i in range(kernel_size):
        for j in range(kernel_size):
            dotproduct += input[x + i, y + j] * kernel[i, j]

    return dotproduct


@jit(nopython=True)
def conv2d(out_r, out_c, kernel, kernel_size, N, out, input):
    for _ in range(N):
        for i in range(out_r):
            for j in range(out_c):
                #if(i+j)%100 == 0:
                #out[i, j] = np.tensordot(input[i:i + kernel_size, j:j + kernel_size], kernel)
                out[i, j] = tensordot(i, j, kernel, kernel_size, input)

    return


@jit(nopython=True)
def initialize_matrix(out_r, out_c, kernel_size):
    A1 = np.random.rand(out_r * out_c, out_r * out_c)

    for i in range(out_r*out_c):
        for j in range(out_r*out_c):
            if A1[i, j] > kernel_size*kernel_size/(out_r*out_c):
                A1[i, j] = 0

    return A1

@jit(nopython=True)
def initialize_weight_matrix(A, N):

    for i in range(N):
        for j in range(N):
            A[i, j] = i*j


delta_t = 0.05
number_of_inputs = 150
number_of_neurons = 4000
number_of_outputs = 30

V = np.random.rand(number_of_neurons, number_of_inputs)
W = np.random.rand(number_of_neurons, number_of_neurons)
T = np.random.rand(number_of_neurons, number_of_outputs)
ob = np.random.rand(number_of_inputs)
y = np.random.rand(number_of_neurons)

A = np.random.rand(800, 500)
B = np.ones(300)

kernel_size = 8
out_r = 64*3
out_c = 64
N1 = 15

kernel = numba.float32(np.random.rand(kernel_size, kernel_size))
input = numba.float32(np.random.rand(out_r+kernel_size, out_c+kernel_size))
out = numba.float32(np.random.rand(out_r, out_c))

kernel = np.random.rand(kernel_size, kernel_size)
input = np.random.rand(out_r+kernel_size, out_c+kernel_size)
out = np.random.rand(out_r, out_c)

print("test1")

N = out_r*out_c
#num_elements = N*kernel_size*kernel_size
num_elements = 300000

data = np.random.rand(num_elements)
row = np.random.randint(N, size=num_elements)
col = np.random.randint(N, size=num_elements)
A = csr_matrix((data, (row, col)), shape=(N, N))


N=10000

A = np.random.rand(N, N)
x = np.random.rand(N, 1)
k=0

#for i in range(N):
#    for j in range(N):
#        if k % 499 != 0:
#            A1[i,j] = 0
#        k=k+1

#np.random.shuffle(A1)
#A = csr_matrix(A1,dtype=float)

print("test2")

#conv2d(out_r, out_c, kernel, kernel_size, N1, out, input)
#step(ob, y, V, W, T, delta_t)
#mult(A, B)

start = time.time()

for i in range(1000):
    #step(ob, y, V, W, T, delta_t)
    #mult(A, B)
    #A.dot(x)
    #conv2d(out_r, out_c, kernel, kernel_size, N1, out, input)

    initialize_weight_matrix(A, N)

end = time.time()

print(end - start)

