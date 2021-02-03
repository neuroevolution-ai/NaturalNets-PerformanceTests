import numpy as np
import time

number_neuron_connections = 3000000
u = np.random.rand(number_neuron_connections, 1, 6).astype(np.float32)

# Weight Matrizes
A = np.random.rand(6, 16).astype(np.float32)
B = np.random.rand(16, 32).astype(np.float32)
C = np.random.rand(32, 16).astype(np.float32)
D = np.random.rand(16, 8).astype(np.float32)
E = np.random.rand(8, 1).astype(np.float32)

# Biases
a = np.random.rand(1, 16).astype(np.float32)
b = np.random.rand(1, 32).astype(np.float32)
c = np.random.rand(1, 16).astype(np.float32)
d = np.random.rand(1, 8).astype(np.float32)
e = np.random.rand(1, 1).astype(np.float32)

def predict(u):
    x = np.matmul(u, A)
    x = np.add(x, a)
    x = np.maximum(0, x)

    x = np.matmul(x, B)
    x = np.add(x, b)
    x = np.maximum(0, x)

    x = np.matmul(x, C)
    x = np.add(x, c)
    x = np.maximum(0, x)

    x = np.matmul(x, D)
    x = np.add(x, d)
    x = np.maximum(0, x)

    x = np.matmul(x, E)
    x = np.add(x, e)

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