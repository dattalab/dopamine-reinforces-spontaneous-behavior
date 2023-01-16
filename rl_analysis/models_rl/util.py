import numpy as np
from copy import deepcopy
from scipy.stats import pearsonr
from numba import jit, prange


def pearsonr_missing_data(x, y):
    nans = np.isnan(x) | np.isnan(y)
    return pearsonr(x[~nans], y[~nans])


def zscore_missing_data(x, axis=1):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x - np.nanmean(x, axis=axis, keepdims=True)) / np.nanstd(
            x, axis=axis, keepdims=True
        )


def compare_usages(
    predicted_usages,
    true_usages,
    compare_func=lambda x, y: pearsonr_missing_data(x, y)[0],
):

    usages1 = predicted_usages.copy().astype("float")
    usages2 = true_usages.copy().astype("float")
    usages2 = usages2[: len(usages1)]

    # print(usages1)
    # print(usages2)

    try:
        return compare_func(usages1, usages2)
    except:
        return np.nan


def compare_tms(
    predicted_tm,
    true_tm,
    normalization_func=lambda x: zscore_missing_data(x, axis=1),
    compare_func=lambda x, y: pearsonr_missing_data(x, y)[0],
    ignore_diagonal=True,
):
    tm1 = predicted_tm.copy()
    tm2 = true_tm.copy()
    tm2 = tm2[: tm1.shape[0], : tm1.shape[1]]

    if ignore_diagonal:
        np.fill_diagonal(tm1, np.nan)
        np.fill_diagonal(tm2, np.nan)

    if normalization_func is not None:
        tm1 = normalization_func(tm1)
        tm2 = normalization_func(tm2)

    try:
        return compare_func(tm1.ravel(), tm2.ravel())
    except Exception as e:
        print(e)
        return np.nan


@jit(nopython=True, parallel=True)
def shuffle_rows_copy(x):
    idx = np.arange(x.shape[1])
    y = np.empty((x.shape[0], x.shape[1]))
    for i in prange(x.shape[0]):
        tmp = np.random.randint(-x.shape[1], +x.shape[1])
        rolled_idx = np.roll(idx, int(tmp))
        y[i] = x[i][rolled_idx]
    return y



# def stable_softmax(p, temperature=1, axis=1):
#     use_p = p.copy()
#     if np.ndim(use_p) == 1:
#         use_p -= np.nanmax(use_p)
#         use_p = np.exp(use_p / temperature)
#         return use_p / np.nansum(use_p)
#     else:
#         use_p -= np.nanmax(use_p, axis=axis, keepdims=True)
#         use_p = np.exp(use_p / temperature)
#         return use_p / np.nansum(use_p, axis=axis, keepdims=True)


# def input_to_sim_format(data, nsyllables=10):
#     cat_data = np.concatenate(data.tolist(), axis=1)
#     rewards = np.empty((nsyllables, nsyllables), dtype="object")
#     sz = np.zeros((nsyllables, nsyllables), dtype="int")
#     for i in range(nsyllables):
#         for j in range(nsyllables):
#             idx = np.flatnonzero((cat_data[0] == i) & (cat_data[1] == j))
#             # idx = idx[np.mod(idx,input_data.shape[2]) < input_data.shape[2] - 1]
#             rewards[i, j] = cat_data[2, idx]
#             sz[i, j] = len(idx)
#             # ave_rew[i,j] = np.mean(cat_data[2,idx])

#     return rewards, sz