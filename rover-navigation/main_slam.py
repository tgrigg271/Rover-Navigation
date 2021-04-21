import simulation
from environment import environment
from sensors import sensors
import navigation
import control
import dynamics
import numpy as np


def init_sim():
    sim_params = dict()
    # Time
    sim_params['t0'] = 0  # Seconds
    sim_params['dt'] = 0.1
    sim_params['tf'] = 10
    # Rover Dynamics
    sim_params['mass'] = 1  # Kilograms
    sim_params['r_wheel'] = 0.05  # Meters, radius of wheel
    sim_params['l_axle'] = 0.05  # Meters, length of axle from vehicle center line to wheel contact point
    # Sensors
    # Environment
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


# Main -----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    sim_params = init_sim()
    # Overwrite fields here as desired.
    # sim_params['tf'] = 15
    # Initialize simulation
    times = np.arange(sim_params['t0'], sim_params['tf'], sim_params['dt'])
    rover_state = sim_params['rover_ics']
    sim_out = None  # Initialize as empty
    # Run simulation
    for t in times:
        env = environment(t, sim_params)
        measurements = sensors(t, rover_state, env, sim_params, use_truth=True)
        estimate = navigation.slam(t, estimate, measurements, sim_params, use_truth=True)
        cmd = control.waypoint_following(t, estimate, sim_params)
        rover_state = dynamics.unicycle_dynamics(t, rover_state, env, cmd)
        sim_out = write_sim_output(t, sim_out, env, measurements, estimate, cmd, rover_state)
    # Print/store outputs
