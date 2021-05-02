import numpy as np


def runge_kutta4(t0, dt, tf, x0, f):
    """
    Implements the Runge Kutta 4 algorithm for numerically integrating the solution of a system of first order
    differential equations.
    :param t0: Initial time
    :param dt: Fixed time step
    :param tf: Final time
    :param x0: Initial state; NOTE: must be numpy array for non-scalar case
    :param f: System dynamics as f(t,x)
    :return: t, x; the time and state histories
    """
    # Initialize time, state history variables
    ts = [t0]
    xs = [x0]
    dxs = []
    # Determine number of time steps
    n_steps = round((tf - t0) / dt)
    # Initialize time, state
    t = t0
    x = x0

    for i_step in range(0, n_steps):
        # Perform Integration
        k1 = dt*f(t, x)
        k2 = dt*f(t + 0.5*dt, x + 0.5*k1)
        k3 = dt*f(t + 0.5*dt, x + 0.5*k2)
        k4 = dt*f(t + dt, x + k3)
        # Update time, state
        t += dt
        dx = (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        x = x + dx
        # Record Time, State
        ts.append(t)
        xs.append(x)
        dxs.append(dx/dt)
    return ts, xs, dxs


def sample_time_update_flag(t, dt, t_sample, tol=1e-5):
    x = np.mod(t, t_sample)
    hit = x < tol*dt
    diff = t_sample - x - dt*0.9
    near_miss = diff < 0
    return hit or near_miss  # Second term should handle misses due to numeric precision.
