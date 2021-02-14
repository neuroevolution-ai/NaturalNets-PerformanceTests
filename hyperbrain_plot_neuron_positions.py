import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import gym
import matplotlib.animation as animation

matplotlib.use('Qt5Agg')


def get_circle_points(n):

    circle_x = np.zeros(n)
    circle_y = np.zeros(n)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)

    for i in range(n):
        circle_x[i] = 0.45*np.cos(t[i])+0.5
        circle_y[i] = 0.45*np.sin(t[i])+0.5

    return circle_x, circle_y


def get_rgb_points(width, height):

    space_x = np.linspace(0, 1, width)
    space_y = np.linspace(0, 1, height)

    grid_x, grid_y = np.meshgrid(space_x, space_y)

    return grid_x.flatten(), grid_y.flatten()


def get_rgb_from_observation(ob):

    ob_scaled = ob/255.0

    ob_red = ob_scaled[:, :, 0]
    ob_green = ob_scaled[:, :, 1]
    ob_blue = ob_scaled[:, :, 2]

    return ob_red.flatten(), ob_green.flatten(), ob_blue.flatten()


def update_plot(i):

    obs, rew, done, info = env.step(env.action_space.sample())

    red, green, blue = get_rgb_from_observation(obs)

    print(i)

    if i == 0:
        global t_start
        t_start = time.time()

    scat_red_inputs.set_array(red)
    scat_green_inputs.set_array(green)
    scat_blue_inputs.set_array(blue)

    if i == number_frames-1:
        print((time.time() - t_start))

    return


# Parameters
number_frames = 1000
input_image_width = 64
input_image_height = 64
red_z = 2
green_z = 3
blue_z = 4
number_neurons = 2000
number_outputs = 16
outputs_z = -1
record_video = False

# Inputs from rgb meshgrids
rgb_x, rgb_y = get_rgb_points(input_image_width, input_image_height)

# CTRNN Neurons are randomly placed in a cube with length = 1
neurons_x, neurons_y, neurons_z = np.random.random((3, number_neurons))

# Outputs as a circle
outputs_x, outputs_y = get_circle_points(number_outputs)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

env = gym.make("procgen:procgen-heist-v0", distribution_mode="memory")

# Procgen Env
obs_init = env.reset()
red_init, green_init, blue_init = get_rgb_from_observation(obs_init)

# Place rgb inputs
scat_red_inputs = ax.scatter(rgb_x, rgb_y, red_z, c=red_init, s=15, edgecolors='none', cmap="Reds", alpha=0.9)
scat_green_inputs = ax.scatter(rgb_x, rgb_y, green_z, c=green_init, s=15, edgecolors='none', cmap="Greens", alpha=0.9)
scat_blue_inputs = ax.scatter(rgb_x, rgb_y, blue_z, c=blue_init, s=15, edgecolors='none', cmap="Blues", alpha=0.9)

# Place CTRNN neurons
scat_neurons = ax.scatter(neurons_x, neurons_y, neurons_z, c="Black", s=15, edgecolors='none')

# Place outputs
scat_outputs = ax.scatter(outputs_x, outputs_y, outputs_z, c="Orange", s=30, edgecolors='none')

ani = animation.FuncAnimation(fig, update_plot, frames=number_frames, repeat=False, interval=20)

# Set Plot to Fullscreen
manager = plt.get_current_fig_manager()
manager.window.showMaximized()

if record_video:
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=3, metadata=dict(artist='Me'), bitrate=-1)

    ani.save('hyperbrain.mp4', writer=writer)

print("Finished")