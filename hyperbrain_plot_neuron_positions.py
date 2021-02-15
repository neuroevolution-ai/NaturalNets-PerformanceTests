import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import gym
import matplotlib.animation as animation

matplotlib.use('Qt5Agg')


def get_rgb_points(width, height, red_z, green_z, blue_z):

    space_x = np.linspace(0, 1, width)
    space_y = np.linspace(0, 1, height)

    grid_x, grid_y = np.meshgrid(space_x, space_y)
    rgb_points_x = grid_x.flatten()
    rgb_points_y = grid_y.flatten()

    positions_red = np.column_stack((rgb_points_x, rgb_points_y, np.ones(width*height) * red_z))
    positions_green = np.column_stack((rgb_points_x, rgb_points_y, np.ones(width*height) * green_z))
    positions_blue = np.column_stack((rgb_points_x, rgb_points_y, np.ones(width*height) * blue_z))

    return np.vstack((positions_red, positions_green, positions_blue))


def get_circle_points(n, position_z):

    circle_x = np.zeros(n)
    circle_y = np.zeros(n)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)

    for i in range(n):
        circle_x[i] = 0.45*np.cos(t[i])+0.5
        circle_y[i] = 0.45*np.sin(t[i])+0.5

    circle_z = np.ones(n)*position_z

    return np.column_stack((circle_x, circle_y, circle_z))


def get_rgb_from_observation(ob):

    ob_scaled = ob/255.0

    ob_red = ob_scaled[:, :, 0].flatten()
    ob_green = ob_scaled[:, :, 1].flatten()
    ob_blue = ob_scaled[:, :, 2].flatten()

    return np.concatenate((ob_red, ob_green, ob_blue))


def split_coordinates(coordinates):

    return coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]


def update_plot(k):

    obs, rew, done, info = env.step(env.action_space.sample())
    inputs = get_rgb_from_observation(obs)
    red = inputs[0:image_size]
    green = inputs[image_size:2 * image_size]
    blue = inputs[2 * image_size:3 * image_size]

    print(k)

    if k == 0:
        global t_start
        t_start = time.time()

    if unified_inputs:
        scat_inputs.set_array(inputs)
    else:
        scat_red_inputs.set_array(red)
        scat_green_inputs.set_array(green)
        scat_blue_inputs.set_array(blue)

    if k == number_frames-1:
        print((time.time() - t_start))

    return


# Parameters
environment = "procgen:procgen-heist-v0"
memory_mode = True
number_frames = 1000
input_image_width = 64
input_image_height = 64
red_z = 2
green_z = 3
blue_z = 4
number_neurons = 5000
number_outputs = 16
outputs_z = -1
record_video = False
unified_inputs = False

# Input positions
image_size = input_image_width * input_image_height
input_positions = get_rgb_points(input_image_width, input_image_height, red_z, green_z, blue_z)
inputs_red_positions = input_positions[0:image_size, :]
inputs_green_positions = input_positions[image_size:2 * image_size, :]
inputs_blue_positions = input_positions[2 * image_size:3 * image_size, :]

# CTRNN Neurons are randomly placed in a cube with length = 1
neurons_positions = np.random.random((number_neurons, 3))

# Outputs as a circle
outputs_positions = get_circle_points(number_outputs, outputs_z)

if memory_mode:
    env = gym.make(environment, distribution_mode="memory")
else:
    env = gym.make(environment)

# Reset Procgen Env and get initial observation
obs_init = env.reset()
inputs_init = get_rgb_from_observation(obs_init)
red_init = inputs_init[0:image_size]
green_init = inputs_init[image_size:2 * image_size]
blue_init = inputs_init[2 * image_size:3 * image_size]

# Split all positions from 2D matrizes to 1D vectors (required for scatter function)
inputs_x, inputs_y, inputs_z = split_coordinates(input_positions)
inputs_red_x, inputs_red_y, inputs_red_z = split_coordinates(inputs_red_positions)
inputs_green_x, inputs_green_y, inputs_green_z = split_coordinates(inputs_green_positions)
inputs_blue_x, inputs_blue_y, inputs_blue_z = split_coordinates(inputs_blue_positions)
neurons_x, neurons_y, neurons_z = split_coordinates(neurons_positions)
outputs_x, outputs_y, outputs_z = split_coordinates(outputs_positions)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Place RGB inputs
if unified_inputs:
    scat_inputs = ax.scatter(inputs_x, inputs_y, inputs_z, c=np.concatenate((red_init, green_init, blue_init)), s=15,
                             edgecolors='none', cmap="Greys", alpha=0.9)
else:
    scat_red_inputs = ax.scatter(inputs_red_x, inputs_red_y, inputs_red_z, c=red_init, s=15,
                                 edgecolors='none', cmap="Reds", alpha=0.9)
    scat_green_inputs = ax.scatter(inputs_green_x, inputs_green_y, inputs_green_z, c=green_init, s=15,
                                   edgecolors='none', cmap="Greens", alpha=0.9)
    scat_blue_inputs = ax.scatter(inputs_blue_x, inputs_blue_y, inputs_blue_z, c=blue_init, s=15,
                                  edgecolors='none', cmap="Blues", alpha=0.9)

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
