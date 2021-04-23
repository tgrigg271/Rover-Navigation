import numpy as np


def accelerometer(state, params):
    meas = np.zeros((3, 1))
    return meas


def gyroscope(state, params):
    meas = np.zeros((3, 1))
    return meas


def magnetometer(state, environment, params):
    meas = np.zeros((3, 1))
    return meas


def camera(state, environment, params):
    meas = 0  # This will be more sophisticated, including range, bearing, and marker ID.
    return meas


# Define Sensor Packages
def imu_camera(t, state, environment, params, use_truth=False):
    if use_truth:
        return state
    else:
        acc_meas = accelerometer(state, params)
        gyro_meas = gyroscope(state, params)
        mag_meas = magnetometer(state, environment, params)
        cam_meas = camera(state, environment, params)
        return [acc_meas, gyro_meas, mag_meas, cam_meas]  # Should this be dict/tuple?
