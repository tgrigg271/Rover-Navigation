import numpy as np
import simulation as sim


def init_rover():
    rover = dict()
    # Rover parameters
    rover['mass'] = 1  # Kilograms
    rover['moi'] = 1  # Kilogram-meter^2, moment of inertia about the center of gravity
    rover['r_wheel'] = 0.03  # Meters, radius of wheel
    rover['l_axle'] = 0.05  # Meters, length of axle from vehicle center line to wheel contact point
    rover['w_max'] = 15  # radians/second, max speed of the rover drive motors.
    rover['tau_motor'] = 0.1  # Seconds, time constant of wheel motor drives
    # Initial conditions
    rover['ics'] = np.zeros([7, 1])
    return rover


def __f_rover_noslip(t, rover_state, env, cmd, params):
    # Parse Inputs
    x, y, vbx, psi, r, w_l, w_r = (rover_state[0], rover_state[1], rover_state[2], rover_state[3], rover_state[4],
                                   rover_state[5], rover_state[6])
    w_l_cmd, w_r_cmd = (cmd[0], cmd[1])
    tau_w = params['rover']['tau_motor']
    r_wheel = params['rover']['r_wheel']
    l_axle = params['rover']['l_axle']

    # xdot = f(t,x)
    # Calculate wheel speed changes first, used in velocity and heading rate dynamics
    dw_l = (w_l_cmd - w_l)/tau_w
    dw_r = (w_r_cmd - w_r)/tau_w
    d_rover_state = np.zeros(np.shape(rover_state))
    d_rover_state[0] = vbx*np.cos(psi)
    d_rover_state[1] = vbx*np.sin(psi)
    d_rover_state[2] = r_wheel/2*(dw_l + dw_r)
    d_rover_state[3] = r
    d_rover_state[4] = r_wheel/(2*l_axle)*(dw_l - dw_r)
    d_rover_state[5] = dw_l
    d_rover_state[6] = dw_r
    return d_rover_state


def no_slip_dynamics(t, rover_state, env, cmd, params):
    # Rover State:
    # [Inertial X Pos, Inertial Y Pos, Body X Vel, Heading, Heading Rate, Left Wheel Speed, Right Wheel Speed]
    # [x, y, vbx, psi, r, w_l, w_r]

    # Wrap rover dynamics function for numeric integration
    def f_rover(t, x): return __f_rover_noslip(t, x, env, cmd, params)
    # Set up timestep
    t0 = t
    dt = params['dt']
    tf = t0 + dt
    # Perform numeric integration
    ts, xs = sim.runge_kutta4(t0, dt, tf, rover_state, f_rover)
    return xs[-1]
