import time
import os
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

mode = "step_batch"
set_numpy_variables = False
test_gpu = False

if set_numpy_variables:
    os.environ['OPENBLAS_NUM_THREADS'] = '12'
    os.environ['MKL_NUM_THREADS'] = '12'
    os.environ['OMP_NUM_THREADS'] = '12'
    os.environ['MPI_NUM_THREADS'] = '12'


if test_gpu:
    import cupy as np
else:
    import numpy as np

number_inputs = 64*64*3
number_neurons = 200
number_outputs = 16

number_neuron_connections = 3000000
population_size = 112

number_env_steps = 1000


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

V = np.random.rand(number_neurons, number_inputs).astype(np.float32)
W = np.random.rand(number_neurons, number_neurons).astype(np.float32)
T = np.random.rand(number_outputs, number_neurons).astype(np.float32)


def predict(u):
    x = u

    for weight, bias in zip(weights, biases):
        x = np.matmul(x, weight)
        x = np.add(x, bias)
        x = np.maximum(0, x)

    return x


def step(u):
    x = np.random.rand(u.shape[0], number_neurons, 1).astype(np.float32)
    dx = np.matmul(W, x) + np.matmul(V, u)
    x += 0.05*dx

    return np.matmul(T, x)


def run_episode(u):

    for _ in range(number_env_steps):
        o = step(u)

    return o


inputs = []

if mode == "step_batch":
    function_to_test = step

    for _ in range(number_env_steps):
        inputs.append(np.random.rand(population_size, number_inputs, 1).astype(np.float32))

elif mode == "step_episode_runner":
    function_to_test = run_episode

    for _ in range(population_size):
        inputs.append(np.random.rand(1, number_inputs, 1).astype(np.float32))

elif mode == "predict":
    function_to_test = predict

    for _ in range(population_size):
        inputs.append(np.random.rand(number_neuron_connections, 1, 6).astype(np.float32))

else:
    raise RuntimeError("No valid mode")


print("Start")

# No map
t_start = time.time()

for inp in inputs:
    y = function_to_test(inp)

print("Sequential without map took: {:6.3f}s".format(time.time()-t_start))


# Multithreading
pool = ThreadPool()

t_start = time.time()

s_multithreading = pool.map(function_to_test, inputs)

print("Multi-Threading map took: {:6.3f}s".format(time.time()-t_start))


# Multiprocessing
pool = multiprocessing.Pool()

t_start = time.time()

s_mp = pool.map(function_to_test, inputs)

print("Multi-Processing map took: {:6.3f}s".format(time.time()-t_start))


# Sequential
t_start = time.time()

s_sequential = map(function_to_test, inputs)

for i in s_sequential:
    y = s_sequential

print("Sequential map took: {:6.3f}s".format(time.time()-t_start))
