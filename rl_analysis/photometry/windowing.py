import numba
import numpy as np
import pandas as pd
from numba import prange


@numba.jit(parallel=True, nopython=True)
def avg_da(data, time, start, stop):
    inds = (time >= start) & (time <= stop)
    tmp = data[:, inds]
    mu = np.zeros(len(tmp))
    for i in prange(len(mu)):
        mu[i] = np.nanmean(tmp[i])
    return mu


@numba.jit(parallel=True, nopython=True)
def std_da(data, time, start, stop):
    inds = (time >= start) & (time <= stop)
    tmp = data[:, inds]
    std = np.zeros(len(tmp))
    for i in prange(len(std)):
        std[i] = np.nanstd(tmp[i])
    return std


@numba.jit(parallel=True, nopython=True)
def max_da(data, time, start, stop):
    inds = (time >= start) & (time <= stop)
    tmp = data[:, inds]
    _max = np.zeros(len(tmp))
    for i in prange(len(_max)):
        _max[i] = np.nanmax(tmp[i])
    return _max


@numba.jit(parallel=True, nopython=True)
def min_da(data, time, start, stop):
    inds = (time >= start) & (time <= stop)
    tmp = data[:, inds]
    _min = np.zeros(len(tmp))
    for i in prange(len(_min)):
        _min[i] = np.nanmin(tmp[i])
    return _min


@numba.jit(parallel=True, nopython=True)
def robust_range(data, time, onset, start, stop):
    onset_inds = (time >= onset[0]) & (time <= onset[1])
    tmp = data[:, onset_inds]
    mins = np.zeros(len(tmp))
    for i in prange(len(mins)):
        mins[i] = np.nanquantile(tmp[i], 0.05)
        
    inds = (time >= start) & (time <= stop)
    tmp = data[:, inds]
    maxes = np.zeros(len(tmp))
    for i in prange(len(maxes)):
        maxes[i] = np.nanquantile(tmp[i], 0.95)
    
    return maxes - mins


@numba.jit(parallel=True, nopython=True)
def normal_range(data, time, onset, start, stop):
    onset_inds = (time >= onset[0]) & (time <= onset[1])
    tmp = data[:, onset_inds]
    mins = np.zeros(len(tmp))
    for i in prange(len(mins)):
        mins[i] = np.nanmin(tmp[i])
        
    inds = (time >= start) & (time <= stop)
    tmp = data[:, inds]
    maxes = np.zeros(len(tmp))
    for i in prange(len(maxes)):
        maxes[i] = np.nanmax(tmp[i])
    
    return maxes - mins


@numba.jit(parallel=True, nopython=True)
def get_syllables(labels, indices, window):
    mask = np.zeros(len(indices), np.bool_)
    mask = ((indices + window[0]) > 0) & ((indices + window[-1]) < len(labels))
    indices = indices[mask]
    
    lbls = np.zeros(len(indices))
    for i in prange(len(lbls)):
        lbls[i] = labels[indices[i]]
        
    return lbls


@numba.jit(parallel=True, nopython=True)
def max_slope(data, time, start, stop, n=2):
    inds = (time >= start) & (time <= stop)
    tmp = data[:, inds]
    slope = np.zeros(len(tmp))
    for i in prange(len(slope)):
        d = tmp[i, n:] - tmp[i, :-n]
        slope[i] = np.nanmax(d) / n
    return slope


@numba.jit(nopython=True, parallel=True)
def _align_matrix(data, indices, window, zscore=True):
    mask = np.zeros(len(indices), np.bool_)
    mask = ((indices + window[0]) > 0) & ((indices + window[-1]) < len(data))
    indices = indices[mask]
    
    mtx = np.zeros((len(indices), len(window)))
    for i in prange(len(mtx)):
        _trial = data[window + indices[i]]
        if zscore:
            _trial = (_trial - np.nanmean(_trial)) / np.nanstd(_trial)
        mtx[i] = _trial
    return mtx


def window_trials_as_mtx(
    data,
    trial_indices,
    signal_key,
    zscore_window=(-10, 10),
    truncation_window=(-0.2, 1),
    fps=30,
    zscore_signal=True
):

    window = np.arange(fps * np.ptp(zscore_window))
    window += zscore_window[0] * fps
    window = np.int64(window)
    time = window / fps

    new_time = time[(time >= truncation_window[0]) & (time <= truncation_window[1])]

    signal_matrix = _align_matrix(data[signal_key].to_numpy(), trial_indices, window, zscore=zscore_signal)
    inds = np.where((time >= truncation_window[0]) & (time <= truncation_window[-1]))[0]
    signal_matrix = signal_matrix[:, inds]

    return new_time, signal_matrix


def window_trials_as_df(
    data,
    trial_indices,
    signal_key,
    syllable_key="predicted_syllable (offline)",
    zscore_window=(-10, 10),
    truncation_window=(-0.2, 1),
    fps=30,
    agg_window=(-0.1, 1),
    avg_window=(-0.4, 1),
    robust_min_window=(-0.2, 0.1),
    deriv_size=2,
    zscore_signal=True
):
    """
    Returns:
        agg_df (pd.DataFrame): dataframe containing syllable-associated
            signal statistics (max, min, mean, etc)
        trial_df (pd.DataFrame): tidy-form dataframe containing windowed trials
            of signal aligned to an onset.
    """

    window = np.arange(fps * np.ptp(zscore_window))
    window += zscore_window[0] * fps
    window = np.int64(window)
    time = window / fps

    new_time = time[(time >= truncation_window[0]) & (time <= truncation_window[1])]
    

    signal_matrix = _align_matrix(data[signal_key].to_numpy(), trial_indices, window, zscore=zscore_signal)
    syllable_matrix = _align_matrix(
        data[syllable_key].to_numpy().astype(float), trial_indices, window, zscore=False
    )

    _mu = avg_da(signal_matrix, time, *avg_window)
    _std = std_da(signal_matrix, time, *agg_window)
    _max = max_da(signal_matrix, time, *agg_window)
    _min = min_da(signal_matrix, time, *agg_window)
    _range = robust_range(signal_matrix, time, robust_min_window, *agg_window)
    _range2 = normal_range(signal_matrix, time, robust_min_window, *agg_window)
    _max_slope = max_slope(signal_matrix, time, *agg_window, n=deriv_size)
    _labels = get_syllables(data[syllable_key].to_numpy(), trial_indices, window)
    
    mask = ((trial_indices + window[0]) > 0) & ((trial_indices + window[-1]) < len(data))
    agg_df = pd.DataFrame(
        {
            "mu": _mu,
            "std": _std,
            "max": _max,
            "min": _min,
            "robust_range": _range,
            "range": _range2,
            "slope": _max_slope,
            "labels": _labels,
            "trial_index": trial_indices[mask],
        }
    )

    inds = np.where((time >= truncation_window[0]) & (time <= truncation_window[-1]))[0]
    signal_matrix = signal_matrix[:, inds]
    
    syllable_matrix = syllable_matrix[:, inds].astype("int16")
    new_time = np.tile(new_time, (len(signal_matrix), 1))
    trials = np.repeat(np.arange(len(signal_matrix)), signal_matrix.shape[1])

    trial_df = pd.DataFrame(
        {
            "dlight": signal_matrix.flatten(),
            "time": new_time.flatten(),
            "real_syllable": syllable_matrix.flatten(),
            "aligned_syllable": np.repeat(_labels.astype("int16"), signal_matrix.shape[1]),
            "trials": trials,
        }
    )
    return agg_df, trial_df


def window_trials_for_decoder(
    data,
    trial_indices,
    signal_key,
    syllable_key="predicted_syllable (offline)",
    zscore_window=(-10, 10),
    truncation_window=(-0.2, 1),
    fps=30,
    deriv_size=2,
):

    window = np.arange(fps * np.ptp(zscore_window))
    window += zscore_window[0] * fps
    window = np.int64(window)
    time = window / fps

    new_time = time[(time >= truncation_window[0]) & (time <= truncation_window[1])]

    signal_matrix = _align_matrix(data[signal_key].to_numpy(), trial_indices, window, zscore=True)

    _labels = get_syllables(data[syllable_key].to_numpy(), trial_indices, window)

    inds = np.where((time >= truncation_window[0]) & (time <= truncation_window[-1]))[0]

    deriv_matrix = signal_matrix[:, deriv_size:] - signal_matrix[:, :-deriv_size]
    signal_matrix = signal_matrix[:, inds]
    deriv_matrix = deriv_matrix[:, inds - deriv_size]

    trial_df = pd.DataFrame(
        signal_matrix,
        columns=pd.Index(new_time, name="time"),
        index=pd.Index(np.arange(len(signal_matrix)), name="trial"),
    )
    deriv_df = pd.DataFrame(
        deriv_matrix,
        columns=pd.Index(new_time, name="time"),
        index=pd.Index(np.arange(len(deriv_matrix)), name="trial"),
    )
    trial_df = pd.concat([trial_df, deriv_df], axis=1)
    trial_df["syllable"] = _labels
    return trial_df