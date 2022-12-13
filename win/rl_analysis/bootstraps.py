import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def bootstrap_ci(data, n_boots=1000):
    '''data = M x N matrix, where M is the number of trials, N is the number of observations
    Returns:
        mus: n_boots x N matrix'''
    mus = np.zeros((n_boots, data.shape[1]))
    n_trials = len(data)
    for i in prange(n_boots):
        choices = np.random.choice(np.arange(n_trials), n_trials, replace=True)
        arr = np.zeros_like(data)
        for j, c in enumerate(choices):
            arr[j] = data[c]
        _mu = np.full(data.shape[1:], np.nan)
        for j in range(data.shape[1]):
            _mu[j] = np.nanmean(arr[:, j])
        mus[i] = _mu
    return mus