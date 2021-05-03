import numpy as np
import simulation


def init_accelerometer(sigma=1.0, limits=(-10, 10)):
    accel_params = dict()
    accel_params['sigma'] = sigma
    accel_params['range'] = limits
    return accel_params


def accelerometer(t, state, params):
    # The accelerometer will prove tricky since we haven't considered this to be a state up to this point... Let's just
    # leave it alone for the moment. We could track a separate 'change in state' variable, or add acceleration as a
    # state. I consider the first to be more viable since our dynamics are driven by acceleration, not jerk.
    meas = np.zeros((3, 1))
    return meas


# Gyroscope ------------------------------------------------------------------------------------------------------------
def init_gyroscope(sigma=1.0, limits=(-10, 10)):
    gyro_params = dict()
    gyro_params['sigma'] = sigma
    gyro_params['range'] = limits
    return gyro_params


def gyroscope(t, state, params):
    # True Measurement
    meas = np.zeros((3, 1))
    meas[2, 0] = state[4, 0]  # Yaw rate
    # Noise
    bias = 0  # Hardcode for now
    sigma = params['sensors']['gyroscope']['sigma']
    meas += np.random.normal(bias, sigma, (3, 1))
    # Saturation
    low_lim, up_lim = params['sensors']['gyroscope']['range']
    meas = np.clip(meas, low_lim, up_lim)
    return meas


# Magnetometer ---------------------------------------------------------------------------------------------------------
def init_magnetometer(sigma=1.0, limits=(-1000, 1000)):
    mag_params = dict()
    mag_params['sigma'] = sigma
    mag_params['range'] = limits
    return mag_params


def magnetometer(t, state, environment, params):
    # True Measurement
    meas = np.zeros((3, 1))
    meas[2] = state[3]  # Heading Angle
    # Noise
    bias = 0  # Hardcode for now
    sigma = params['sensors']['magnetometer']['sigma']
    meas += np.random.normal(bias, sigma, (3, 1))
    # Saturation
    low_lim, up_lim = params['sensors']['magnetometer']['range']
    meas = np.clip(meas, low_lim, up_lim)
    return meas


# Camera ---------------------------------------------------------------------------------------------------------------
def init_camera(rang=2.5, fov=(-0.5, 0.5), r_sigma=0.1, az_sigma=0.001, t_sample=0.1):
    camera_params = dict()
    camera_params['range'] = rang  # Detection range, meters
    camera_params['field_of_view'] = fov  # +/-, radians
    camera_params['range_sigma'] = r_sigma  # Range standard deviation, meters
    camera_params['angle_sigma'] = az_sigma  # Angle standard deviation, radians
    camera_params['t_sample'] = t_sample
    return camera_params


def camera(t, state, environment, params):
    # Parse inputs
    rover_pos = state[0:2]  # X,Y position
    heading = state[3, 0]  # Heading angle
    markers = environment['markers']
    # Sim params
    dt = params['dt']  # For measurement sample time
    # Camera params
    max_range = params['sensors']['camera']['range']
    fov = params['sensors']['camera']['field_of_view']
    r_sigma = params['sensors']['camera']['range_sigma']
    az_sigma = params['sensors']['camera']['angle_sigma']
    t_sample = params['sensors']['camera']['t_sample']
    # Initialize measurement list, remains empty if no measurements available
    measurements = []
    new_sample = simulation.sample_time_update_flag(t, dt, t_sample)  # Check for new frame from camera.
    if new_sample:
        for marker in markers:
            tag_pos = marker['position']
            tag_id = marker['id']
            rel_pos = tag_pos - rover_pos
            slant_range = np.linalg.norm(rel_pos)
            los = np.arctan2(rel_pos[1, 0], rel_pos[0, 0]) - heading
            in_view = slant_range < max_range and fov[0] < los < fov[1]
            if in_view:
                # Add measurement noise
                slant_range += np.random.normal(0, r_sigma)
                los += np.random.normal(0, az_sigma)
                # Define pose matrix, assumes no rotation of tag and no vertical displacement. We may want to relax these
                # assumptions
                # Transpose of matrix T defined at https://www.mathworks.com/help/images/ref/rigid3d.html
                pose = np.array([
                    [1, 0, 0, slant_range*np.cos(los)],
                    [0, 1, 0, slant_range*np.sin(los)],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
                meas = {'pose': np.array(pose), 'id': tag_id}
                measurements.append(meas)
    else:
        measurements = None
    return measurements  # None for no available measurement, [] for no tags in view.


# Define Sensor Packages
def imu_camera(t, state, environment, params):
    acc_meas = accelerometer(t, state, params)
    gyro_meas = gyroscope(t, state, params)
    mag_meas = magnetometer(t, state, environment, params)
    cam_meas = camera(t, state, environment, params)
    measurements = {'accelerometer': acc_meas, 'gyroscope': gyro_meas, 'magnetometer': mag_meas, 'camera': cam_meas}
    return measurements
