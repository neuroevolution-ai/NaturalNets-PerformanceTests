import numpy as np
import time

dim = 5000

t_start = time.time()

C = np.identity(dim)
diagD, B = np.linalg.eigh(C)

print(time.time()-t_start)

print("Finished")