import simulation
import environment
import sensors
import navigation
import control
import dynamics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


# Initialization -------------------------------------------------------------------------------------------------------
def init_sim(seed=1):
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
    # Control
    sim_params['control'] = control.init_waypoint_following()
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
    plt.draw()
    pass


def plot_map(sim_out, sim_params, i_sample=0):
    environment = sim_out['environment'][i_sample]
    rover_state = sim_out['rover_state'][i_sample]
    estimate = sim_out['estimate'][i_sample]
    fig, ax = plt.subplots()

    # Plot rover trajectory
    rover_state_hist = np.array(sim_out['rover_state'][0:i_sample+1])
    x, y = (rover_state_hist[:, 0], rover_state_hist[:, 1])
    # Rover XY Trajectory
    ax.plot(x, y, 'k')
    ax.set_title(f"Environment Map, t={i_sample*sim_params['dt']:.2f}")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')


    # Plot rover position
    ax.plot(rover_state[0], rover_state[1], 'g^')

    # Plot camera field of view
    rang = sim_params['sensors']['camera']['range']
    fov = sim_params['sensors']['camera']['field_of_view']
    psi = rover_state[3, 0]
    fov_wedge = Wedge(rover_state[0:2, 0], rang, (fov[0]+psi)*180/np.pi, (fov[1]+psi)*180/np.pi,
                      edgecolor='black', facecolor='orange')
    ax.add_patch(fov_wedge)

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
    ax.plot(mark_xs, mark_ys, 'bo')

    # Plot rover position estimate
    rover_pos = estimate['state'][0:2]
    rover_pos_cov = estimate['cov'][0:2, 0:2]
    ax.plot(rover_pos[0], rover_pos[1], 'r^')
    navigation.covariance_ellipse(rover_pos, rover_pos_cov, ax, edgecolor='red')

    # Plot landmark position estimate
    marker_states = estimate['state'][3:]
    marker_cov = estimate['cov'][2:, 2:]
    n_markers = int(len(marker_states)/2)  # Guaranteed to be even number, so no worries.
    mark_xs = []
    mark_ys = []
    for i_marker in range(n_markers):
        k_mark = 2*i_marker
        mark_xs.append(marker_states[k_mark, 0])
        mark_ys.append(marker_states[k_mark+1, 0])
    ax.plot(mark_xs, mark_ys, 'gs')
    for i_marker in range(n_markers):
        k_mark = 2*i_marker
        mark_pos = np.array([[mark_xs[i_marker]], [mark_ys[i_marker]]])
        mark_cov = marker_cov[k_mark:k_mark+2, k_mark:k_mark+2]
        navigation.covariance_ellipse(mark_pos, mark_cov, ax, edgecolor='green')

    # Let the table fill the figure
    ax.axis('equal')
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius))
    plt.draw()
    pass


def plot_estimate(sim_out):
    time = sim_out['time']
    # Extract state histories
    rover_state = np.array(sim_out['rover_state'])
    x, y, vbx, psi, r, w_l, w_r = (rover_state[:, 0], rover_state[:, 1], rover_state[:, 2], rover_state[:, 3],
                                   rover_state[:, 4], rover_state[:, 5], rover_state[:, 6])
    # Extract estimate histories
    estimates = sim_out['estimate']
    x_hat = []
    y_hat = []
    psi_hat = []
    x_hat_cov = []
    y_hat_cov = []
    psi_hat_cov = []
    for estimate in estimates:
        x_hat.append(estimate['state'][0, 0])
        y_hat.append(estimate['state'][1, 0])
        psi_hat.append(estimate['state'][2, 0])
        x_hat_cov.append(estimate['cov'][0, 0])
        y_hat_cov.append(estimate['cov'][1, 1])
        psi_hat_cov.append(estimate['cov'][2, 2])
    x_hat = np.reshape(np.array(x_hat), x.shape)
    y_hat = np.reshape(np.array(y_hat), x.shape)
    psi_hat = np.reshape(np.array(psi_hat), x.shape)
    x_hat_cov = np.reshape(np.array(x_hat_cov), x.shape)
    y_hat_cov = np.reshape(np.array(y_hat_cov), x.shape)
    psi_hat_cov = np.reshape(np.array(psi_hat_cov), x.shape)
    # Plot Residuals
    fig, ax = plt.subplots(3, 1)
    # X Position
    ax[0].plot(time, x_hat - x, 'r')
    ax[0].plot(time, 3*np.sqrt(x_hat_cov), 'k')
    ax[0].plot(time, -3*np.sqrt(x_hat_cov), 'k')
    ax[0].set_title('Rover State Estimate')
    ax[0].set_ylabel('dX')
    # Y Position
    ax[1].plot(time, y_hat - y, 'r')
    ax[1].plot(time, 3*np.sqrt(y_hat_cov), 'k')
    ax[1].plot(time, -3*np.sqrt(y_hat_cov), 'k')
    ax[1].set_ylabel('dY')
    # Psi
    ax[2].plot(time, control.wrap(psi_hat - psi, -np.pi, np.pi), 'r')
    ax[2].plot(time, 3*np.sqrt(psi_hat_cov), 'k')
    ax[2].plot(time, -3*np.sqrt(psi_hat_cov), 'k')
    ax[2].set_ylabel('dPsi')
    ax[2].set_xlabel('Time (s)')
    plt.draw()

def plot_measurements(sim_out):
    cam_meas = extract_measurements(sim_out, 'camera')
    pass


# Main -----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    sim_params = init_sim()
    # Overwrite fields here as desired.
    sim_params['tf'] = 100
    use_truth = True

    # Initialize simulation
    times = np.arange(sim_params['t0'], sim_params['tf'], sim_params['dt'])
    rover_state = sim_params['rover']['ics']
    a_pri_estimate = sim_params['slam']['estimate']
    cmd = control.waypoint_following(times[0], rover_state, a_pri_estimate, sim_params, use_truth)
    sim_out = None  # Initialize as empty
    # sim_out = write_sim_output(t, sim_out, env, measurements, estimate, cmd, rover_state)

    # Run simulation
    for t in times:
        env = environment.circular_table(t, sim_params)
        measurements = sensors.imu_camera(t, rover_state, env, sim_params)
        post_estimate, a_pri_estimate = navigation.slam(t, a_pri_estimate, measurements, cmd, sim_params)
        cmd, sim_params = control.waypoint_following(t, rover_state, post_estimate, sim_params, use_truth)
        rover_state, d_rover_state = dynamics.no_slip_dynamics(t, rover_state, env, cmd, sim_params)
        sim_out = write_sim_output(t, sim_out, env, measurements, post_estimate, cmd, rover_state)
    # Print/store outputs
    plot_rover_states(sim_out)
    plot_measurements(sim_out)
    plot_map(sim_out, sim_params)
    plot_map(sim_out, sim_params, len(times)-1)
    plot_estimate(sim_out)
    plt.show()
    pass
