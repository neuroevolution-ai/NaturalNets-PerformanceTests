import numpy as np
import time

number_neuron_connections = 3000000
u = np.random.rand(number_neuron_connections, 1, 6).astype('d')
A = np.random.rand(6, 16).astype('d')
B = np.random.rand(16, 32).astype('d')
C = np.random.rand(32, 16).astype('d')
D = np.random.rand(16, 8).astype('d')
E = np.random.rand(8, 1).astype('d')


def predict(u):
    x = np.matmul(u, A)
    x = np.maximum(0, x)
    x = np.matmul(x, B)
    x = np.maximum(0, x)
    x = np.matmul(x, C)
    x = np.maximum(0, x)
    x = np.matmul(x, D)
    x = np.maximum(0, x)
    x = np.matmul(x, E)

    return x


t1 = time.time()

#cp.cuda.Device(0).synchronize()

x1 = predict(u)

t2 = time.time()

print((t2-t1))

x2 = x1*0


t1 = time.time()

#cp.cuda.Device(0).synchronize()

for i in range(number_neuron_connections):
    x2[i] = predict(u[i])

t2 = time.time()

print((t2-t1))