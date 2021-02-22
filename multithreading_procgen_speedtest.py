# https://code.activestate.com/recipes/577360-a-multithreaded-concurrent-version-of-map/

import time
import threading
import numpy as np
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

import gym


def concurrent_map(func, env):
    """
    Similar to the bultin function map(). But spawn a thread for each argument
    and apply `func` concurrently.

    Note: unlike map(), we cannot take an iterable argument. `data` should be an
    indexable sequence.
    """

    N = len(env)
    result = [None] * N

    # wrapper to dispose the result in the right slot
    def task_wrapper(i):
        result[i] = func(env[i], env[i].action_space.sample())

    threads = [threading.Thread(target=task_wrapper, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return result


def env_step(environment):
    action = environment.action_space.sample()
    obs, rew, done, info = environment.step(action)
    return obs


population_size = 112

envs = []
obs = []
for i in range(population_size):
    e = gym.make('procgen:procgen-heist-v0')
    envs.append(e)
    obs.append(e.reset())


#pool = multiprocessing.Pool()

pool = ThreadPool()


t_start = time.time()

for _ in range(1000):
    #for env in envs:
    #    env_step(env, env.action_space.sample())

    # concurrent_map(env_step, envs)

    pool.map(env_step, envs)

print("Envs steps took: {:6.2f}s".format(time.time()-t_start))

