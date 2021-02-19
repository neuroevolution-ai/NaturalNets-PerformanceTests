import multiprocessing
import time

import gym
import gym3
import numpy as np
from gym.vector import make as make_vec_env
from procgen import ProcgenGym3Env

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
    rewards = np.zeros(population_size)
    for _ in range(number_env_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)

        rewards += rew

    return rewards


def run_episode_gym3_vec_env(u):

    env = ProcgenGym3Env(num=population_size, env_name="heist")
    rewards = np.zeros(population_size)
    for _ in range(number_env_steps):
        env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
        rew, obs, first = env.observe()

        rewards += rew
    return rewards


def main():
    inputs = np.zeros(population_size)

    # Multiprocessing
    pool = multiprocessing.Pool()

    t_start = time.time()

    result_mp = pool.map(run_episode_full, inputs)

    print("Multi-Processing map took: {:6.3f}s".format(time.time()-t_start))

    # Vectorized environment
    t_start = time.time()

    result_vec = run_episode_vec_env([])

    print("Vectorized environment took: {:6.3f}s".format(time.time()-t_start))

    # Gym3 Vectorized environment
    t_start = time.time()

    result_gym3_vec = run_episode_gym3_vec_env([])

    print("Gym3 vec environment took: {:6.3f}s".format(time.time()-t_start))

    assert (len(result_mp) == len(result_vec)
            and len(result_mp) == len(result_gym3_vec)
            and  len(result_mp) == population_size)


if __name__ == "__main__":
    main()
