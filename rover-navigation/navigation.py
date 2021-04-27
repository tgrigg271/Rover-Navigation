import numpy as np


def init_slam():
    return np.zeros([7, 1])


def slam(t, estimate, measurements, sim_params):
    estimate = measurements  # Temporary placeholder
