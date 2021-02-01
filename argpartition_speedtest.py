import numpy as np
import time

a = np.arange(100000000)

np.random.shuffle(a)

start = time.time()

b = np.argpartition(a, 100000)[:100000]

end = time.time()
print(end - start)

print(b)