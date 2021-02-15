import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import gym
import matplotlib.animation as animation

matplotlib.use('Qt5Agg')


def get_rgb_positions(width: int, height: int,
                      red_position_z: float, green_position_z: float, blue_position_z: float) -> np.ndarray:

    space_x = np.linspace(0, 1, width)
    space_y = np.linspace(0, 1, height)

    grid_x, grid_y = np.meshgrid(space_x, space_y)
    rgb_positions_x = grid_x.flatten()
    rgb_positions_y = grid_y.flatten()

    red_positions = np.column_stack((rgb_positions_x, rgb_positions_y, np.ones(width*height) * red_position_z))
    green_positions = np.column_stack((rgb_positions_x, rgb_positions_y, np.ones(width*height) * green_position_z))
    blue_positions = np.column_stack((rgb_positions_x, rgb_positions_y, np.ones(width*height) * blue_position_z))

    return np.vstack((red_positions, green_positions, blue_positions))


def get_circle_positions(number_positions: int, radius: float, position_z: float) -> np.ndarray:

    circle_positions_x = np.zeros(number_positions)
    circle_positions_y = np.zeros(number_positions)

    angles = np.linspace(0, 2 * np.pi, number_positions, endpoint=False)

    for i in range(number_positions):
        circle_positions_x[i] = radius*np.cos(angles[i])+0.5
        circle_positions_y[i] = radius*np.sin(angles[i])+0.5

    circle_positions_z = np.ones(number_positions) * position_z

    return np.column_stack((circle_positions_x, circle_positions_y, circle_positions_z))


def get_input_vector_from_observation(observation: np.ndarray) -> np.ndarray:

    observation_scaled = observation / 255.0

    observation_red = observation_scaled[:, :, 0].flatten()
    observation_green = observation_scaled[:, :, 1].flatten()
    observation_blue = observation_scaled[:, :, 2].flatten()

    return np.concatenate((observation_red, observation_green, observation_blue))


def place_points(points: np.ndarray, c, s, cmap):

    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = points[:, 2]

    return ax.scatter(points_x, points_y, points_z, c=c, s=s, edgecolors='none', cmap=cmap, alpha=0.9)


def update_plot(k: int) -> None:

    obs, rew, done, info = env.step(env.action_space.sample())
    input_data = get_input_vector_from_observation(obs)

    print(k)

    if k == 0:
        global t_start
        t_start = time.time()

    if unified_inputs:
        scat_inputs.set_array(input_data)
    else:
        red = input_data[0:image_size]
        green = input_data[image_size:2 * image_size]
        blue = input_data[2 * image_size:3 * image_size]

        scat_red_inputs.set_array(red)
        scat_green_inputs.set_array(green)
        scat_blue_inputs.set_array(blue)

    if k == number_frames-1:
        print((time.time() - t_start))

    return


# Parameters
environment: str = "procgen:procgen-heist-v0"
memory_mode: bool = True
number_frames: int = 1000
input_image_width: int = 64
input_image_height: int = 64
red_z: float = 2.0
green_z: float = 3.0
blue_z: float = 4.0
number_neurons: int = 5000
number_outputs: int = 16
outputs_radius: float = 0.45
outputs_z: float = -1.0
record_video: bool = False
unified_inputs: bool = False


# Input positions
input_positions = get_rgb_positions(input_image_width, input_image_height, red_z, green_z, blue_z)

# CTRNN neurons are randomly placed in a cube with length = 1
neuron_positions = np.random.random((number_neurons, 3))

# Outputs as a circle
output_positions = get_circle_positions(number_outputs, outputs_radius, outputs_z)

if memory_mode:
    env = gym.make(environment, distribution_mode="memory")
else:
    env = gym.make(environment)

# Reset Procgen env and get initial observation
observation_init = env.reset()
input_data_init = get_input_vector_from_observation(observation_init)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Place RGB inputs
if unified_inputs:
    scat_inputs = place_points(input_positions, c=input_data_init, s=15, cmap="Greys")
else:
    image_size = input_image_width * input_image_height

    input_data_red_init = input_data_init[0:image_size]
    input_data_green_init = input_data_init[image_size:2 * image_size]
    input_data_blue_init = input_data_init[2 * image_size:3 * image_size]

    input_positions_red = input_positions[0:image_size, :]
    input_positions_green = input_positions[image_size:2 * image_size, :]
    input_positions_blue = input_positions[2 * image_size:3 * image_size, :]

    scat_red_inputs = place_points(input_positions_red, c=input_data_red_init, s=15, cmap="Reds")
    scat_green_inputs = place_points(input_positions_green, c=input_data_green_init, s=15, cmap="Greens")
    scat_blue_inputs = place_points(input_positions_blue, c=input_data_blue_init, s=15, cmap="Blues")

# Place CTRNN neurons
scat_neurons = place_points(neuron_positions, c="Black", s=15, cmap=None)

# Place outputs
scat_outputs = place_points(output_positions, c="Orange", s=30, cmap=None)

ani = animation.FuncAnimation(fig, update_plot, frames=number_frames, repeat=False, interval=20)

# Set Plot to Fullscreen
manager = plt.get_current_fig_manager()
manager.window.showMaximized()

if record_video:
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=-1)

    ani.save('hyperbrain.mp4', writer=writer)

print("Finished")
