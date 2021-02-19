import time
import gym
from gym.vector import make as make_vec_env
import multiprocessing
import numpy as np


population_size = 112

number_env_steps = 1000


def run_episode_full(u):

    env = gym.make('procgen:procgen-heist-v0')
    obs = env.reset()
    reward = 0

    for _ in range(number_env_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)

        reward += rew

    return reward


def run_episode_vec_env(u):

    env = make_vec_env(id="procgen:procgen-heist-v0", num_envs=population_size, asynchronous=True)
    obs = env.reset()
    rewards = np.zeros((population_size))
    for _ in range(number_env_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)

        rewards += rew

    return rewards

inputs = np.zeros((population_size))

# Multiprocessing
pool = multiprocessing.Pool()

t_start = time.time()

result_mp = pool.map(run_episode_full, inputs)

print("Multi-Processing map took: {:6.3f}s".format(time.time()-t_start))

# Vectorized environment

t_start = time.time()

result_vec = run_episode_vec_env([])

print("Vectorized environment took: {:6.3f}s".format(time.time()-t_start))

assert len(result_mp) == len(result_vec) and len(result_mp) == population_size
