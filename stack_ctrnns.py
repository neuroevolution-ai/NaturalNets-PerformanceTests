import numpy as np

dt = 0.05

V_a = np.random.rand(3,3)*2-1
W_a = np.random.rand(3,3)*2-1
V_b = np.random.rand(3,3)*2-1
W_b = np.random.rand(3,3)*2-1
V_c = np.block([[V_a, np.zeros((3, 3))], [np.zeros((3, 3)), np.zeros((3, 3))]])
W_c = np.block([[W_a, np.zeros((3, 3))], [V_b, W_b]])

net_type = 'CTRNN'

u_a = np.random.rand(3)*2-1
u_c = np.hstack((u_a, np.zeros(3)))

x_a = np.random.rand(3)*2-1
x_b = np.random.rand(3)*2-1
x_c = np.hstack((x_a, x_b))

N = 150

for i in range(N):

    if net_type == 'CTRNN':
        x_a_new = x_a + dt * (np.matmul(W_a, np.arctan(x_a)) + np.matmul(V_a, u_a))
        x_b_new = x_b + dt * (np.matmul(W_b, np.arctan(x_b)) + np.matmul(V_b, np.arctan(x_a)))

        x_c = x_c + dt * (np.matmul(W_c, np.arctan(x_c)) + np.matmul(V_c, u_c))

    elif net_type == 'Elman':
        x_a_new = np.arctan(np.matmul(W_a, x_a) + np.matmul(V_a, u_a))
        x_b_new = np.arctan(np.matmul(W_b, x_b) + np.matmul(V_b, x_a))

        x_c = np.arctan(np.matmul(W_c, x_c) + np.matmul(V_c, u_c))

    x_a = x_a_new
    x_b = x_b_new

    print(x_a, x_b)
    print(x_c)

print("Finished")
