import numpy as np


def init_waypoint_following(rover_speed=0.25, heading_gain=1.0):
    control_params = dict()
    control_params['speed'] = rover_speed  # meters/second
    control_params['heading_gain'] = heading_gain
    control_params['waypoints'] = 2.5*np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    control_params['waypoint_radius'] = 0.1  # meters
    control_params['waypoint_index'] = 0
    return control_params


def wrap(value, low=0, high=1):
    diff = high - low
    return ((value - low) % diff) + low


def waypoint_following(t, state, estimate, sim_params, use_truth=False):
    # Parse Inputs
    vbx = sim_params['control']['speed']
    k_heading = sim_params['control']['heading_gain']
    waypoints = sim_params['control']['waypoints']
    n_wps = len(waypoints)
    r_wp = sim_params['control']['waypoint_radius']
    wp_idx = sim_params['control']['waypoint_index']
    r_wheel = sim_params['rover']['r_wheel']
    l_axle = sim_params['rover']['l_axle']

    # Select rover states
    if use_truth:
        rover_pos = state[0:2]
        psi = state[3, 0]
    else:
        rover_pos = estimate['state'][0:2]
        psi = estimate['state'][2, 0]

    # Calculate nominal wheel speed
    w_nom = vbx/r_wheel
    # Choose waypoint to head towards

    waypoint = waypoints[wp_idx]
    waypoint = np.reshape(waypoint, (2, 1))
    rel_pos = waypoint - rover_pos
    wp_bearing = np.arctan2(rel_pos[1, 0], rel_pos[0, 0])

    # Command heading to waypoint
    err = wrap(wp_bearing - psi, -np.pi, np.pi)
    heading_rate = k_heading*err  # Just proportional control for now
    w_offset = l_axle/r_wheel*heading_rate
    cmd = np.array([[w_nom + w_offset], [w_nom - w_offset]])

    # Cycle waypoints
    if np.linalg.norm(rel_pos) < r_wp:
        wp_idx += 1
        sim_params['control']['waypoint_index'] = wrap(wp_idx, 0, n_wps)

    return cmd, sim_params
