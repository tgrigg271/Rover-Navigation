import numpy as np


def earth_magnetic_field():
    return np.array([1, 0, 0])  # NED frame?


def make_april_tag(position, tag_id):
    tag = {'position': position, 'id': tag_id}
    return tag


def init_circular_table(radius=5, n_markers=25):
    env_params = dict()
    env_params['radius'] = radius
    env_params['n_markers'] = n_markers
    # Generate camera markers
    markers = []
    rng = np.random.default_rng()
    ids = rng.choice(586, size=n_markers, replace=False)  # Tag family 36h11.
    ids = list(ids)
    for i_marker in range(n_markers):
        r = np.random.uniform(radius)
        theta = np.random.uniform(2*np.pi)
        x, y = (r*np.cos(theta), r*np.sin(theta))
        position = np.array([[x], [y]])
        tag_id = ids.pop()
        markers.append(make_april_tag(position, tag_id))
    env_params['markers'] = markers
    return env_params


def circular_table(t, sim_params):
    env = dict()
    # Magnetic field
    env['b_field'] = earth_magnetic_field()
    # Camera markers (April Tags)
    env['markers'] = sim_params['environment']['markers']  # Currently using static camera reference points
    return env
