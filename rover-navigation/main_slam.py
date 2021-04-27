import simulation
import environment
import sensors
import navigation
import control
import dynamics
import numpy as np
import matplotlib.pyplot as plt


def init_sim():
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
    return sim_params


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


# Main -----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    sim_params = init_sim()
    # Overwrite fields here as desired.
    # sim_params['tf'] = 15
    # Initialize simulation
    times = np.arange(sim_params['t0'], sim_params['tf'], sim_params['dt'])
    rover_state = sim_params['rover']['ics']
    estimate = navigation.init_slam()
    sim_out = None  # Initialize as empty
    # sim_out = write_sim_output(t, sim_out, env, measurements, estimate, cmd, rover_state)
    # Run simulation
    for t in times:
        env = environment.circular_table(t, sim_params)
        measurements = sensors.imu_camera(t, rover_state, env, sim_params)
        estimate = navigation.slam(t, estimate, measurements, sim_params)
        cmd = control.waypoint_following(t, rover_state, estimate, sim_params, use_truth=True)
        rover_state, d_rover_state = dynamics.no_slip_dynamics(t, rover_state, env, cmd, sim_params)
        sim_out = write_sim_output(t, sim_out, env, measurements, estimate, cmd, rover_state)
    # Print/store outputs
    print(rover_state)
    plot_rover_states(sim_out)
