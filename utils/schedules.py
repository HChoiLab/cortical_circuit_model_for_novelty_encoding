import numpy as np

# annealing schedules for the regularization coefficients
def linear_cyclical_schedule(n_epochs, start=0.0, stop=1.0, n_cycles=4, ratio=0.5):
    L = np.ones(n_epochs) * stop
    period = n_epochs / n_cycles 
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycles):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_epochs):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def ramp_schedule(n_epochs, epoch_thresh, start=0.0, stop=1.0):
    L = np.ones(n_epochs) * start
    eps = np.arange(epoch_thresh, n_epochs+1) - epoch_thresh
    L[int(epoch_thresh - 1):] = (eps / (n_epochs - epoch_thresh)) * stop
    return L

def decreasing_ramp_schedule(n_epochs, epoch_thresh, start=0.25, stop=0.05, decay_episodes=None):
    if decay_episodes is None:
        decay_episodes = n_epochs
    L = np.ones(n_epochs) * start
    eps = np.arange(epoch_thresh, n_epochs+1) - epoch_thresh
    L[int(epoch_thresh - 1):] = np.maximum(stop, start - (start - stop) * (eps / (decay_episodes - epoch_thresh)))
    
    return L


def step_schedule(n_epochs, epoch_thresh, step_from=0.0, step_to=1.0):
    L = step_from * np.ones(n_epochs)
    L[int(epoch_thresh - 1):] = step_to
    return L