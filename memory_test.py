import numpy as np

dt = 0.05

number_inputs = 3
number_neurons = 6

V = np.random.rand(number_neurons, number_inputs)*4-2
W = np.random.rand(number_neurons, number_neurons)*4-2

net_type = 'CTRNN3'

u = np.random.rand(number_inputs)*4-2

N = 100

x = np.random.rand(number_neurons)*4-2

for t in range(N):

    print("t=" + str(t) + ": x=" + str(x))

    if net_type == 'Elman1':
        x = np.arctan(np.matmul(W, x) + np.matmul(V, u))

    elif net_type == 'Elman2':
        x = np.clip(np.matmul(W, x) + np.matmul(V, u), -1, 1)

    elif net_type == 'CTRNN1':
        x = x + dt*(np.matmul(W, np.arctan(x)) + np.matmul(V, u))
        x = np.clip(x, -1, 1)

    elif net_type == 'CTRNN2':
        x = x + dt*(np.matmul(W, x) + np.matmul(V, u))
        x = np.clip(x, -1, 1)

    elif net_type == 'CTRNN3':
        x = x + dt * (np.matmul(W, x) + np.matmul(V, u))
        x = np.arctan(x)


print("Finished")
