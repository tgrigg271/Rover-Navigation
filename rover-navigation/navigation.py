import numpy as np
from copy import deepcopy
from scipy.spatial.transform import rotation as rot
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def covariance_ellipse(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    Take from: https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    mean : array-like, shape (n, )
        Sample mean, center of ellipse

    cov: array-like, shape (n, n)
        Sample covariance

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0, 0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1, 0]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def cv_state_prediction(dt, rover_state_est, cmd, sim_params):
    # Parse Inputs
    x, y, psi = (rover_state_est[0], rover_state_est[1], rover_state_est[2])
    vbx = sim_params['control']['speed']  # Assume wheels are at commanded speed
    w_l, w_r = cmd  # Assume wheels are at commanded speed
    r_wheel = sim_params['rover']['r_wheel']
    l_axle = sim_params['rover']['l_axle']

    # Reconstruct command (Hardcoded relationship for now)
    u = r_wheel/(2*l_axle)*(w_l - w_r)
    rover_state_proj = np.zeros((3, 1))
    rover_state_proj[0] = x + vbx*dt*np.cos(psi)
    rover_state_proj[1] = y + vbx*dt*np.sin(psi)
    rover_state_proj[2] = psi + u*dt
    return rover_state_proj


def cv_state_transition(dt, rover_state, sim_params):
    vbx = sim_params['control']['speed']  # Assume wheels are at commanded speed
    psi = rover_state[2, 0]
    phi = np.array([
        [1, 0, -vbx*dt*np.sin(psi)],
        [0, 1, vbx*dt*np.cos(psi)],
        [0, 0, 1]])
    return phi


def cart_to_polar(cart, psi=0):
    r = np.linalg.norm(cart)
    x, y = (cart[0, 0], cart[1, 0])
    theta = np.arctan2(y, x) - psi
    return np.array([[r], [theta]])


def polar_to_cart(pol, psi=0):
    r, theta = (pol[0, 0], pol[1, 0])
    x = r*np.cos(theta + psi)
    y = r*np.sin(theta + psi)
    cart = np.array([[x], [y]])  # Probably shouldn't hardcode this to be a two element column vector?
    return cart


def cv_meas_model(rover_state, marker):
    x, y, psi = (rover_state[0], rover_state[1], rover_state[2])
    dx = marker['position'][0] - x
    dy = marker['position'][1] - y
    meas = cart_to_polar([dx, dy], psi)
    return meas


def cv_meas_jacobian_rover(rover_state, marker_state):
    H = np.zeros((2, 3))
    dx = marker_state[0] - rover_state[0]
    dy = marker_state[1] - rover_state[1]
    rang = np.linalg.norm([dx, dy])
    rang2 = rang**2
    H[0][0] = -dx/rang
    H[0][1] = -dy/rang
    H[1][0] = dy/rang2
    H[1][1] = -dx/rang2
    H[1][2] = -1
    return H


def cv_meas_jacobian_mark(rover_state, marker_state):
    H = np.zeros((2,2))
    dx = marker_state[0] - rover_state[0]
    dy = marker_state[1] - rover_state[1]
    rang = np.linalg.norm([dx, dy])
    rang2 = rang**2
    H[0][0] = dx/rang
    H[0][1] = dy/rang
    H[1][0] = -dy/rang2
    H[1][1] = dx/rang2
    return H


def cv_meas_inv_rover_jacobian(rover_state, meas):
    """
    Jacobian of the inverse measurement model with respect to the rover state
    :param rover_state:
    :param meas:
    :return:
    """
    # Heading
    psi = rover_state[2]
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    # Angle Measurement
    theta = meas[1]
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    r = meas[0]
    # Populate Jacobian matrix
    G_r = np.zeros((2, 3))
    G_r[0, 0] = 1
    G_r[0, 1] = 0
    G_r[0, 2] = -r*s_psi*c_theta - r*c_psi*s_theta
    G_r[1, 0] = 0
    G_r[1, 1] = 1
    G_r[1, 2] = r*c_psi*c_theta - r*s_psi*s_theta
    return G_r



def cv_meas_inv_meas_jacobian(rover_state, meas):
    """
    Jacobian of the inverse measurement model with respect to the measurement
    :param rover_state:
    :param meas:
    :return:
    """
    # Heading
    psi = rover_state[2]
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    # Angle Measurement
    theta = meas[1]
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    r = meas[0]
    # Populate Jacobian matrix
    G_z = np.zeros((2, 2))
    G_z[0, 0] = c_psi*c_theta - s_psi*s_theta
    G_z[0, 1] = -r*c_psi*s_theta - r*s_psi*c_theta
    G_z[1, 0] = s_psi*c_theta + c_psi*s_theta
    G_z[1, 1] = -r*s_psi*s_theta + r*c_psi*c_theta
    return G_z


def find_old_makers(vis_markers, estimate):
    # Get set of new IDs not in the SLAM filter memory.
    vis_ids = set([marker['id'] for marker in vis_markers])
    known_ids = set(estimate['mapped_ids'])
    old_ids = known_ids.intersection(vis_ids)
    # Collect marker observations associated with new IDs.
    old_markers = []
    for marker in vis_markers:
        if old_ids.__contains__(marker['id']):
            old_markers.append(marker)
    return old_markers


def find_new_makers(vis_markers, estimate):
    # Get set of new IDs not in the SLAM filter memory.
    vis_ids = set([marker['id'] for marker in vis_markers])
    known_ids = set(estimate['mapped_ids'])
    new_ids = vis_ids - known_ids
    # Collect marker observations associated with new IDs.
    new_markers = []
    for marker in vis_markers:
        if new_ids.__contains__(marker['id']):
            new_markers.append(marker)
    return new_markers


def marker_lookup(markers, tag_id):
    for marker in markers:
        if marker['id'] == tag_id:
            return marker
    return None  # Return none if no marker matches


def init_slam(pos_var=1e-3, vbx_var=1, heading_var=1e-4, r_var=0.1, rang_var=0.1**2, ang_var=0.001**2):
    slam_params = dict()
    # Dynamics
    slam_params['f_prediction'] = cv_state_prediction
    slam_params['f_transition'] = cv_state_transition
    # Process Noise
    slam_params['Q'] = 5e-2*np.diag((vbx_var, r_var))
    # Measurement Model
    slam_params['f_meas'] = cv_meas_model  # Not using this now, but we should. I chose dev time over flexibility.
    slam_params['H_meas'] = cv_meas_jacobian_rover
    # Sensor Noise
    slam_params['R'] = np.diag((rang_var, ang_var))
    # Initialize Estimate
    estimate = dict()
    # State: x, y, heading. I want to add vbx, but we'll get there.
    estimate['state'] = np.zeros((3, 1))
    estimate['cov'] = np.diag((pos_var, pos_var, heading_var))  # Just start with diagonal.
    estimate['mapped_ids'] = []
    slam_params['estimate'] = estimate
    return slam_params


def slam_predict(t, estimate, cmd, sim_params):
    # No-op for t==0,
    if t == 0:
        return estimate

    # Parse Inputs
    dt = sim_params['dt']  # Propagate at every sim time step
    n_rover_states = 3  # Hardcode it lol
    Q = sim_params['slam']['Q']
    n_Q = Q.shape[0]
    state = estimate['state']
    rover_state = state[0:n_rover_states, 0]
    f_prop = sim_params['slam']['f_prediction']
    f_phi = sim_params['slam']['f_transition']

    # Propagate rover state, marker states unchanged
    rover_state = f_prop(dt, rover_state, cmd, sim_params)
    estimate['state'][0:n_rover_states] = rover_state

    # Propagate rover covariance
    phi = f_phi(dt, rover_state, sim_params)
    psi = rover_state[2, 0]
    G = np.zeros((n_rover_states, n_Q))
    G[0, 0] = dt*np.cos(psi)
    G[1, 0] = dt*np.sin(psi)
    G[2, 1] = dt
    rover_cov = estimate['cov'][0:n_rover_states, 0:n_rover_states]
    rover_cov = phi @ rover_cov @ phi.T + G @ Q @ G.T
    estimate['cov'][0:n_rover_states, 0:n_rover_states] = rover_cov

    # Update rover/landmark covariance
    cov = estimate['cov']
    for i_id, tag_id in enumerate(estimate['mapped_ids']):
        k_mark = 2*i_id + n_rover_states
        P_rl = cov[0:n_rover_states, k_mark:k_mark+2]
        P_rl = phi @ P_rl
        cov[0:n_rover_states, k_mark:k_mark+2] = P_rl
        cov[k_mark:k_mark+2, 0:n_rover_states] = P_rl.T
    estimate['cov'] = cov
    return estimate


def slam_update(t, estimate, measurements, sim_params):
    # Parse inputs
    state = estimate['state']
    n_states = state.size
    n_rover_states = 3
    rover_state = state[0:n_rover_states]
    cov = estimate['cov']
    mapped_ids = estimate['mapped_ids']
    vis_markers = measurements['camera']
    R = sim_params['slam']['R']

    # No-op when no new camera measurement is available.
    if vis_markers is None:
        return estimate

    # Find known markers we can see
    vis_old_markers = find_old_makers(vis_markers, estimate)

    # Build up block diagonal measurement matrix
    n_meas = len(vis_old_markers)
    if n_meas > 0:  # Perform update if measurements are available
        R_vec = np.diagonal(R)  # Assume no off-diagonal terms
        RR = np.diag(np.tile(R_vec, n_meas))
        # Probably trash.
        # n_meas = R.shape[0]
        # n_RR = RR.shape[0]
        # z_RR_R = np.zeros((n_meas, n_RR))
        # z_R_RR = np.zeros((n_RR, n_meas))
        # RR = np.block([[RR,     z_RR_R],
        #                [z_R_RR, R]])

        # Build up measurement matrix, vector
        i_meas = 0
        meas_dim = R.shape[0]
        n_states = len(state)
        H = np.zeros((meas_dim*n_meas, n_states))
        z = np.zeros((0, 1))
        zhat = np.zeros((0, 1))
        for i_id, mapped_id in enumerate(mapped_ids):
            marker = marker_lookup(vis_old_markers, mapped_id)
            # Only update for known, visible markers
            if marker is not None:
                pose = marker['pose']
                # Array indices for building H
                k_meas = i_meas*meas_dim  # Measurement index
                k_mark = i_id*meas_dim  # Marker index
                # Build measurement matrix, H
                marker_state = state[k_mark+3:k_mark+5]
                H[k_meas:k_meas+2, 0:n_rover_states] = cv_meas_jacobian_rover(rover_state, marker_state)
                H[k_meas:k_meas+2, k_mark+n_rover_states:k_mark+n_rover_states+2] = cv_meas_jacobian_mark(rover_state,
                                                                                                          marker_state)
                # Build measurement vector, z
                rel_pos_b = pose[0:2, 3:]
                meas = cart_to_polar(rel_pos_b)
                z = np.concatenate((z, meas), 0)
                # Build predicted measurement vector, zhat
                rel_pos = marker_state - rover_state[0:2]  # Rel position in inertial frame
                meas_hat = cart_to_polar(rel_pos, rover_state[2, 0])  # Range/bearing in body frame
                zhat = np.concatenate((zhat, meas_hat), 0)
                i_meas += 1
            if i_meas == n_meas:
                break
        # Calculate innovation
        innov = z - zhat

        # Update estimate
        K = cov @ H.T @ np.linalg.inv(H @ cov @ H.T + RR)
        state = state + K @ innov
        cov = (np.eye(n_states) - K @ H) @ cov
        cov = 1/2*(cov + np.transpose(cov))  # Enforce symmetry in the covariance matrix
        estimate['state'] = state
        estimate['cov'] = cov
    return estimate


def slam_augment(t, estimate, measurements, sim_params):
    # Parse inputs
    state = estimate['state']
    x, y, psi = (state[0], state[1], state[2])
    cov = estimate['cov']
    vis_markers = measurements['camera']
    R = sim_params['slam']['R']

    # No-op when no new camera measurement is available.
    if vis_markers is None:
        return estimate

    # Find and add new markers
    new_markers = find_new_makers(vis_markers, estimate)
    for marker in new_markers:
        pose = marker['pose']
        tag_id = marker['id']

        # Compute marker inertial position
        rel_pos_b = pose[0:2, 3:]
        meas = cart_to_polar(rel_pos_b)
        r_mat = rot.Rotation.from_euler('z', psi)
        r_mat = r_mat.as_matrix()
        r_mat = r_mat[0, 0:2, 0:2]  # Provided as 3,3 but we want 2,2
        marker_pos = state[0:2] + np.matmul(r_mat, rel_pos_b)

        # Compute marker inertial position covariance
        P_rr = cov[0:3, 0:3]  # Rover state covariance
        G_r = cv_meas_inv_rover_jacobian(state, meas)
        G_z = cv_meas_inv_meas_jacobian(state, meas)
        P_ll = np.matmul(np.matmul(G_r, P_rr), np.transpose(G_r)) + np.matmul(np.matmul(G_z, R), np.transpose(G_z))
        P_lr = np.matmul(G_r, P_rr)
        P_lm = np.matmul(G_r, cov[0:3, 3:])  # Covariance between known and new landmarks
        P_lr_lm = np.concatenate((P_lr, P_lm), 1)
        P_rl_ml = np.transpose(P_lr_lm)

        # Augment list of known IDs
        estimate['mapped_ids'].append(tag_id)
        # Augment state
        state = np.concatenate((state, marker_pos))
        # Augment covariance
        cov = np.block([
            [cov,     P_rl_ml],
            [P_lr_lm, P_ll]])
    estimate['state'] = state
    estimate['cov'] = cov
    return estimate


def slam(t, estimate, measurements, cmd, sim_params):
    estimate = deepcopy(estimate)  # Break link between estiamtes at different timesteps.

    # Project state estimate
    estimate = slam_predict(t, estimate, cmd, sim_params)
    # Update SLAM
    a_post_estimate = slam_update(t, estimate, measurements, sim_params)

    # Augment with new landmarks
    a_priori_estimate = slam_augment(t, estimate, measurements, sim_params)

    return a_post_estimate, a_priori_estimate
