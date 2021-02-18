import numpy as np
import time
from random import randint
import gym

#env = gym.make('procgen:procgen-heist-v0')
#obs = env.reset()

#start = time.time()

#for i in range(1000):
#    obs, rew, done, info = env.step(env.action_space.sample())


#end = time.time()
#print("Elapsed time: " + str(end - start) + "ms")


#A = np.random.rand(600, 400)
#B = np.random.rand(80, 80)
A = np.ones([400, 400])
B = np.ones([80, 80])*5
C = np.zeros([400, 400])
n = 300


a, b = 50, 50
n = 400
r = 100

y,x = np.ogrid[-a:n-a, -b:n-b]
mask = x*x + y*y <= r*r

start = time.time()

for i in range(1000):
    n = randint(0, 300)
    #C = np.array(A, copy=True)
    #C = A
    C[0:200, 0:200] = A[200:400, 200:400]

    for i in range(50):
        C[0:80, 0:80] = B

    for i in range(50):
        C[mask] = 255

    #C = A[n:n+64, n:n+64]
    #C[n:80+n, n:80+n] = B


end = time.time()
print("Elapsed time: " + str(end - start) + "ms")



