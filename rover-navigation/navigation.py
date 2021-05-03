
import numpy as np
from scipy.spatial.transform import rotation as rot


def cv_state_prediction(dt, rover_state, v=1):
    x, y, psi = (rover_state[0], rover_state[1], rover_state[2])
    u = 0
    rover_state_proj = np.zeros((3, 1))
    rover_state_proj[0] = x + v*dt*np.cos(psi)
    rover_state_proj[1] = y + v*dt*np.sin(psi)
    rover_state_proj[2] = psi + u*dt  # Should add heading rate control here.


def cv_state_transition(dt, rover_state, v=1):
    psi = rover_state[2]
    phi = np.array([
        [1, 0, -v*dt*np.sin(psi)],
        [0, 1, v*dt*np.cos(psi)],
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


def init_slam(pos_var=1e-3, vbx_var=1, heading_var=1e-4, r_var=0.1, rang_var=0.5, ang_var=0.01):
    slam_params = dict()
    # Dynamics
    slam_params['f_prediction'] = cv_state_prediction
    slam_params['f_transition'] = cv_state_transition
    # Process Noise
    slam_params['Q'] = np.diag((vbx_var, r_var))
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


def slam_predict(t, estimate, cmd):
    # No-op for t==0,
    if t == 0:
        return estimate
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
            marker = vis_markers[i_meas]
            pose = marker['pose']
            # Only update for known, visible markers
            if marker['id'] == mapped_id:
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
    # Project state estimate
    estimate = slam_predict(t, estimate, cmd)
    # Update SLAM
    estimate = slam_update(t, estimate, measurements, sim_params)

    # Augment with new landmarks
    estimate = slam_augment(t, estimate, measurements, sim_params)

    return estimate
