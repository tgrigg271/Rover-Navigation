import simulation
import environment
import sensors
import navigation
import control
import dynamics
import numpy as np
import matplotlib.pyplot as plt


# Initialization -------------------------------------------------------------------------------------------------------
def init_sim(seed=0):
    # Control random number generation
    np.random.seed(seed)
    sim_params = dict()
    # Time
    sim_params['t0'] = 0  # Seconds
    sim_params['dt'] = 0.01
    sim_params['tf'] = 10
    # Rover Dynamics
    sim_params['rover'] = dynamics.init_rover()
    # Sensors
    sim_params['sensors'] = dict()
    sim_params['sensors']['accelerometer'] = sensors.init_accelerometer()
    sim_params['sensors']['gyroscope'] = sensors.init_gyroscope()
    sim_params['sensors']['magnetometer'] = sensors.init_magnetometer()
    sim_params['sensors']['camera'] = sensors.init_camera()
    # Environment
    sim_params['environment'] = environment.init_circular_table()
    # SLAM
    sim_params['slam'] = navigation.init_slam()
    return sim_params


# Output/Plots ---------------------------------------------------------------------------------------------------------
def write_sim_output(t, sim_out, env, measurements, estimate, cmd, rover_state):
    """
    Writes the state of the simulation over time for convenient post-processing and visualization. Currently stores
    :param t: Simulation time, seconds
    :param sim_out: Prior sim_out dictionary. Use None for initialization
    :param env:
    :param measurements:
    :param estimate:
    :param cmd:
    :param rover_state:
    :return: sim_out
    """
    if sim_out is None:
        sim_out = {'time': [], 'environment': [], 'measurements': [], 'estimate': [], 'command': [], 'rover_state': []}
    else:
        sim_out['time'].append(t)
        sim_out['environment'].append(env)
        sim_out['measurements'].append(measurements)
        sim_out['estimate'].append(estimate)
        sim_out['command'].append(cmd)
        sim_out['rover_state'].append(rover_state)
    return sim_out


def extract_measurements(sim_out, meas_name):
    samples = sim_out['measurements']
    ts = []
    measurements = []
    for i_sample, sample in enumerate(samples):
        meas = sample[meas_name]
        if meas is not None:
            ts.append(sim_out['time'][i_sample])
            measurements.append(meas)
    return ts, measurements


def plot_rover_states(sim_out):
    time = sim_out['time']
    rover_state = np.array(sim_out['rover_state'])
    x, y, vbx, psi, r, w_l, w_r = (rover_state[:, 0], rover_state[:, 1], rover_state[:, 2], rover_state[:, 3],
                                   rover_state[:, 4], rover_state[:, 5], rover_state[:, 6])
    # Rover XY Trajectory
    fig, ax = plt.subplots()
    ax.plot(y, x)
    ax.set_title('Rover Trajectory')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    # Rover Position vs. Time
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(time, x)
    ax[0].set_title('Rover Trajectory')
    ax[0].set_ylabel('X (m)')
    ax[1].plot(time, y)
    ax[1].set_ylabel('Y (m)')
    ax[1].set_xlabel('Time (s)')
    # Rover Tangential Velocity vs. Time
    fig, ax = plt.subplots()
    ax.plot(time, vbx)
    ax.set_title('Rover Body X Velocity')
    ax.set_ylabel('Vbx (m/s)')
    ax.set_xlabel('Time (s)')
    # Rover Angular Position & Velocity vs. Time
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(time, psi)
    ax[0].set_title('Rover Angular States')
    ax[0].set_ylabel('Psi (m)')
    ax[1].plot(time, r)
    ax[1].set_ylabel('r (m)')
    ax[1].set_xlabel('Time (s)')
    plt.show()
    pass


def plot_map(sim_out, sim_params):
    environment = sim_out['environment'][0]
    rover_state = sim_out['rover_state'][0]
    fig, ax = plt.subplots()
    # Plot rover position
    ax.plot(rover_state[0], rover_state[1], 'r^')
    # Plot table
    radius = sim_params['environment']['radius']
    table_outline = plt.Circle((0, 0), radius, color='b', fill=False)
    ax.add_patch(table_outline)
    # Plot AprilTags/Camera Markers
    markers = environment['markers']
    mark_xs = []
    mark_ys = []
    for marker in markers:
        x, y = (marker['position'][0], marker['position'][1])
        mark_xs.append(x)
        mark_ys.append(y)
    ax.plot(mark_ys, mark_xs, 'bo')
    # Let the table fill the figure
    ax.axis('equal')
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius))
    plt.show()
    pass


def plot_measurements(sim_out):
    cam_meas = extract_measurements(sim_out, 'camera')
    pass


# Main -----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    sim_params = init_sim()
    # Overwrite fields here as desired.
    # sim_params['tf'] = 15

    # Initialize simulation
    times = np.arange(sim_params['t0'], sim_params['tf'], sim_params['dt'])
    rover_state = sim_params['rover']['ics']
    estimate = sim_params['slam']['estimate']
    cmd = control.waypoint_following(times[0], rover_state, estimate, sim_params, use_truth=True)
    sim_out = None  # Initialize as empty
    # sim_out = write_sim_output(t, sim_out, env, measurements, estimate, cmd, rover_state)

    # Run simulation
    for t in times:
        env = environment.circular_table(t, sim_params)
        measurements = sensors.imu_camera(t, rover_state, env, sim_params)
        estimate = navigation.slam(t, estimate, measurements, cmd, sim_params)
        cmd = control.waypoint_following(t, rover_state, estimate, sim_params, use_truth=True)
        rover_state, d_rover_state = dynamics.no_slip_dynamics(t, rover_state, env, cmd, sim_params)
        sim_out = write_sim_output(t, sim_out, env, measurements, estimate, cmd, rover_state)
    # Print/store outputs
    plot_rover_states(sim_out)
    plot_measurements(sim_out)
    plot_map(sim_out, sim_params)
    pass
