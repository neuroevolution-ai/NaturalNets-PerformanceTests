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

    ob_red = ob_scaled[:, :, 0]
    ob_green = ob_scaled[:, :, 1]
    ob_blue = ob_scaled[:, :, 2]

    return ob_red.flatten(), ob_green.flatten(), ob_blue.flatten()


def split_coordinates(coordinates):

    return coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]


def update_plot(i):

    obs, rew, done, info = env.step(env.action_space.sample())
    red, green, blue = get_rgb_from_observation(obs)

    print(i)

    if i == 0:
        global t_start
        t_start = time.time()

    if unified_inputs:
        inputs = np.concatenate((red, green, blue))
        scat_inputs.set_array(inputs)
    else:
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
number_neurons = 5000
number_outputs = 16
outputs_z = -1
record_video = False
unified_inputs = False

# Inputs from rgb meshgrids
image_size = input_image_width * input_image_height
input_positions = get_rgb_points(input_image_width, input_image_height, red_z, green_z, blue_z)
inputs_red_positions = input_positions[0:image_size, :]
inputs_green_positions = input_positions[image_size:2 * image_size, :]
inputs_blue_positions = input_positions[2 * image_size:3 * image_size, :]

for i in input_positions:
    print(i[0])
    print("test")

# CTRNN Neurons are randomly placed in a cube with length = 1
neurons_positions = np.random.random((number_neurons, 3))

# Outputs as a circle
outputs_positions = get_circle_points(number_outputs, outputs_z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#env = gym.make("procgen:procgen-chaser-v0", distribution_mode="memory")
env = gym.make("procgen:procgen-heist-v0")

# Procgen Env
obs_init = env.reset()
red_init, green_init, blue_init = get_rgb_from_observation(obs_init)

test = obs_init.flatten()/255.0

# Split all positions from 2D arrays to 1D vectors
inputs_x, inputs_y, inputs_z = split_coordinates(input_positions)
inputs_red_x, inputs_red_y, inputs_red_z = split_coordinates(inputs_red_positions)
inputs_green_x, inputs_green_y, inputs_green_z = split_coordinates(inputs_green_positions)
inputs_blue_x, inputs_blue_y, inputs_blue_z = split_coordinates(inputs_blue_positions)
neurons_x, neurons_y, neurons_z = split_coordinates(neurons_positions)
outputs_x, outputs_y, outputs_z = split_coordinates(outputs_positions)

# Place rgb inputs
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
