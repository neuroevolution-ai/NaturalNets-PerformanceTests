import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import random


def tensordot(x, y, kernel, kernel_size, input, data):

    dotproduct = 0

    for i in range(kernel_size):
        for j in range(kernel_size):
            dotproduct += input[x + i, y + j] * kernel[i, j]
            data.append([x+i, y+j, 0.102, x, y, 0.1, kernel[i, j]])

    return dotproduct


def conv2d(out_r, out_c, kernel, kernel_size, out, input, data):
    for i in range(out_r):
        for j in range(out_c):
            out[i, j] = tensordot(i, j, kernel, kernel_size, input, data)

    return


kernel_size = 5
out_r = 64
out_c = 64

kernel = np.random.rand(kernel_size, kernel_size)*2.0-1.0
input = np.random.rand(out_r+kernel_size, out_c+kernel_size)
out = np.random.rand(out_r, out_c)

data = list()

conv2d(out_r, out_c, kernel, kernel_size, out, input, data)

# Normalize data
data_norm = list()
for data_row in data:
    data_norm.append([data_row[0]/(out_r+kernel_size-2), data_row[1]/(out_c+kernel_size-2), data_row[2], (data_row[3]+1)/(out_r+kernel_size-2), (data_row[4]+1)/(out_c+kernel_size-2), data_row[5], data_row[6]])

data_norm_np = np.asarray(data_norm)
np.random.shuffle(data_norm_np)

input_labels_np = data_norm_np[:,0:6]
output_labels_np = data_norm_np[:,6]

trainX = input_labels_np[0:30000]
trainY = output_labels_np[0:30000]
testX = input_labels_np[30000:]
testY = output_labels_np[30000:]

np.save('trainX.npy', trainX)
np.save('trainY.npy', trainY)
np.save('testX.npy', testX)
np.save('testY.npy', testY)

#fig = plt.figure()
#ax = fig.gca(projection='3d')

#colormap = cm.inferno

#colors = output_labels_np
#norm = Normalize()
#norm.autoscale(colors)

#ax.quiver(input_labels_np[:,0], input_labels_np[:,1], input_labels_np[:,2], input_labels_np[:,3]-input_labels_np[:,0], input_labels_np[:,4]-input_labels_np[:,1], input_labels_np[:,5]-input_labels_np[:,2], color=colormap(norm(output_labels_np)), linewidth=0.7, arrow_length_ratio=0)

#plt.show()