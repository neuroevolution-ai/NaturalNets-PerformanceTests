import numpy as np
import time
import os
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

os.environ['OPENBLAS_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MPI_NUM_THREADS'] = '12'

number_neuron_connections = 3000000
u = list()
population_size = 112


for i in range(population_size):
    u.append(np.random.rand(number_neuron_connections, 1, 6).astype(np.float32))

# Weight Matrizes
A = np.random.rand(6, 16).astype(np.float32)
B = np.random.rand(16, 32).astype(np.float32)
C = np.random.rand(32, 16).astype(np.float32)
D = np.random.rand(16, 8).astype(np.float32)
E = np.random.rand(8, 1).astype(np.float32)
weights = [A, B, C, D, E]

# Biases
a = np.random.rand(1, 16).astype(np.float32)
b = np.random.rand(1, 32).astype(np.float32)
c = np.random.rand(1, 16).astype(np.float32)
d = np.random.rand(1, 8).astype(np.float32)
e = np.random.rand(1, 1).astype(np.float32)
biases = [a, b, c, d, e]

def predict(u):

    x = u

    for weight, bias in zip(weights, biases):
        x = np.matmul(x, weight)
        x = np.add(x, bias)
        x = np.maximum(0, x)

    return x

# No map

t_start = time.time()

for u_i in u:
    y = predict(u_i)

print((time.time()-t_start))

# Multithreading

pool = ThreadPool()

t_start = time.time()

s_multithreading = pool.map(predict, u)

for i in s_multithreading:
    print("test multithreading")

print((time.time()-t_start))


# Multiprocessing
pool = multiprocessing.Pool()

t_start = time.time()

s_mp = pool.map(predict, u)

for i in s_mp:
    print("test mp")

print((time.time()-t_start))


# Sequential
t_start = time.time()

s_sequential = map(predict, u)

for i in s_sequential:
    print("test sequential")

print((time.time()-t_start))

print("Finished")
