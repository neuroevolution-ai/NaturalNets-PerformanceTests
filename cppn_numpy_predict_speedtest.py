import numpy as np
import time

number_neuron_connections = 3000000
u = np.random.rand(number_neuron_connections, 1, 6).astype(np.float32)

u_t = np.random.rand(number_neuron_connections, 6, 1).astype(np.float32)*0

for i in range(len(u)):
    k = u[i].T
    u_t[i] = u[i].T

#u_t = np.transpose(u)  #np.random.rand(number_neuron_connections, 6, 1).astype(np.float32)

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

# Weight Matrizes transposed
A_t = A.T
B_t = B.T
C_t = C.T
D_t = D.T
E_t = E.T
weights_t = [A_t, B_t, C_t, D_t, E_t]

# Biases transposed
a_t = a.T
b_t = b.T
c_t = c.T
d_t = d.T
e_t = e.T
biases_t = [a_t, b_t, c_t, d_t, e_t]


def predict_direct(u):
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

def predict_direct_transposed(u):
    x = np.matmul(A_t, u)
    x = np.add(x, a_t)
    x = np.maximum(0, x)

    x = np.matmul(B_t, x)
    x = np.add(x, b_t)
    x = np.maximum(0, x)

    x = np.matmul(C_t, x)
    x = np.add(x, c_t)
    x = np.maximum(0, x)

    x = np.matmul(D_t, x)
    x = np.add(x, d_t)
    x = np.maximum(0, x)

    x = np.matmul(E_t, x)
    x = np.add(x, e_t)

    return x

def predict_loop(u):

    x = u

    for weight, bias in zip(weights, biases):
        x = np.matmul(x, weight)
        x = np.add(x, bias)
        x = np.maximum(0, x)

    return x

def predict_loop_transposed(u):

    x = u

    for weight, bias in zip(weights_t, biases_t):
        x = np.matmul(weight, x)
        x = np.add(x, bias)
        x = np.maximum(0, x)

    return x

t_start = time.time()
x_direct = predict_direct(u)
print((time.time()-t_start))

t_start = time.time()
x_direct_transposed = predict_direct_transposed(u_t)
print((time.time()-t_start))


t_start = time.time()
x_loop = predict_loop(u)
print((time.time()-t_start))

t_start = time.time()
x_loop_transposed = predict_loop_transposed(u_t)
print((time.time()-t_start))

print(np.max(np.abs(x_direct - x_direct_transposed)))
print(np.max(np.abs(x_direct - x_loop)))
print(np.max(np.abs(x_direct - x_loop_transposed)))

#x2 = x1*0


#t1 = time.time()

#cp.cuda.Device(0).synchronize()

#for i in range(number_neuron_connections):
#    x2[i] = predict(u[i])

#t2 = time.time()

#print((t2-t1))