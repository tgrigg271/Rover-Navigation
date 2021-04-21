def slam(t, estimate, measurements, sim_params, use_truth=False):
    if use_truth:
        return measurements  # measurements are the true state with use_truth=True
    else:
        return estimate
