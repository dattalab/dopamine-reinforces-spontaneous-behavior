import numpy as np
from scipy.special import psi


# hutter mutual information per Gershman
# Args:
#  tm: 2d numpy array of raw transition counts, assumes rows are states and columns are actions
#  alpha: specifies regularization, typical values are between 0-1
#
# Returns:
#  I: mutual information between "states" and "actions"
def hutter_mi(tm, alpha=0.1):
    N = dm_prior(tm, alpha=alpha)
    n = tm.sum()

    # number of actions per state, sum across columns
    # number of states per action sum across rows
    nA, nS = tm.sum(axis=1, keepdims=True), tm.sum(axis=0, keepdims=True)

    P = psi(N + 1) - psi(nA + 1) - psi(nS + 1) + psi(n + 1)
    I = (N * P).sum() / n

    return I


# direct calculation of mutual information
# Args:
#  tm: 2d numpy array of raw transition counts, assumes rows are states and columns are actions
#  alpha: specifies regularization, typical values are between 0-1
#
# Returns:
#  I: mutual information between "states" and "actions"
def plugin_mi(tm):
    # potentially want to mask out the diagonal from this calculation, but shouldn't factor since tm_p
    # along the diagonal is 0 by our label definition
    with np.errstate(invalid="ignore", divide="ignore"):
        col_p = tm.sum(axis=0)
        col_p /= col_p.sum()
        
        row_p = tm.sum(axis=1)
        row_p /= row_p.sum()

        tm_p = tm / tm.sum()

        log_term = np.log(tm_p / np.outer(row_p, col_p))
        log_term = np.nan_to_num(log_term, 0, 0, 0)
        mutual_info = np.nansum(tm_p * log_term)
    if ~np.isfinite(mutual_info):
        mutual_info = np.nan
    return mutual_info


# direct calculation of mutual information
# Args:
#  tm: 2d numpy array of raw transition counts, assumes rows are states and columns are actions
#  alpha: specifies regularization, typical values are between 0-1
#
# Returns:
#  I: mutual information between "states" and "actions"
def plugin_cond_entropy(tm):
    # potentially want to mask out the diagonal from this calculation, but shouldn't factor since tm_p
    # along the diagonal is 0 by our label definition
    with np.errstate(invalid="ignore", divide="ignore"):
        usages = tm.sum(axis=1, keepdims=True)
        usages /= usages.sum()
        tm_p = tm / tm.sum()

        log_term = np.log(tm_p / usages)
        log_term = np.nan_to_num(log_term, 0, 0, 0)
        ent = -np.nansum(tm_p * log_term)

    if ~np.isfinite(ent):
        ent = np.nan
    return ent


def dm_mi(tm, alpha=None):
    return plugin_mi(dm_prior(tm, alpha=alpha))


def dm_cond_entropy(tm, alpha=None, marginalize=True):
    if marginalize:
        use_x = marginalize_tm(tm)
    else:
        use_x = tm
    return plugin_cond_entropy(dm_prior(use_x, alpha=alpha))


# great resource...
#
# https://rdrr.io/cran/entropy/f/
def dm_prior(tm, alpha=None, axis=None, normalize=True):
    # tm is a matrix of counts
    # some choices for a:
    # a = 0          :   empirical estimate
    # a = 1          :   Laplace
    # a = 1/2        :   Jeffreys
    # a = 1/m        :   Schurmann-Grassberger  (m: number of bins)
    # a = sqrt(n)/m  :   minimax
    if alpha == "perks":
        alpha = 1. / tm.size
    elif alpha == "minimax":
        alpha = np.sqrt(tm.sum()) / tm.size
    elif alpha is None:
        alpha = 1. / 2.

    # return the normalized bigram matrix
    tm_alpha = tm.astype("float") + alpha
    if axis is None and normalize:
        tm_alpha /= tm_alpha.sum()
    elif normalize:
        tm_alpha /= tm_alpha.sum(axis=axis, keepdims=True)
    return tm_alpha


def marginalize_tm(tm):
    with np.errstate(invalid="ignore", divide="ignore"):
        tm_marginal = tm.copy().astype("float")
        usages = tm_marginal.sum(axis=1)
        usages /= usages.sum()
        expectation = np.outer(usages, usages)
        tm_marginal -= expectation * tm_marginal.sum()
        tm_marginal = np.clip(tm_marginal, 0, np.inf)

    return tm_marginal


def plugin_entropy(y, marginalize=False):
    if marginalize:
        y = marginalize_tm(y)
    
    yhat = y / y.sum()
    
    # nans are skipped here...
    with np.errstate(invalid="ignore", divide="ignore"):
        return -np.nansum(yhat * np.log2(yhat))


# for some reason I set marginalize to 
def dm_entropy(tm, alpha=None, marginalize=False, axis=None, min_threshold=None):
    # marginalize out usages prior to other things...
    if len(tm) < 1:
        return np.nan

    if marginalize:
        use_x = marginalize_tm(tm)
    else:
        use_x = tm

    if min_threshold is not None:
        use_x = use_x[use_x.sum(axis=1) > min_threshold]

    if len(use_x) < 1:
        return np.nan
    return plugin_entropy(dm_prior(use_x, alpha=alpha, axis=axis), marginalize=False)


def _pearsonr(x, y):
    from scipy.stats import pearsonr
    nans = np.isnan(x) | np.isnan(y)
    return pearsonr(x[~nans], y[~nans])


def _zscore(x, axis=1):
    return (x - np.nanmean(x, axis=axis, keepdims=True)) / np.nanstd(x, axis=axis, keepdims=True)


def compare_tms(
    predicted_tm,
    true_tm,
    normalization_func=lambda x: _zscore(x, axis=1),
    compare_func=lambda x, y: _pearsonr(x, y)[0],
):
    tm1 = predicted_tm.copy()
    tm2 = true_tm.copy()
    tm2 = tm2[: tm1.shape[0], : tm1.shape[1]]

    if normalization_func is not None:
        tm1 = normalization_func(tm1)
        tm2 = normalization_func(tm2)
    
    # any off-diagonal nans?  if so bail
    isnan1 = np.isnan(tm1)
    isnan2 = np.isnan(tm2)
    
    try:
        return compare_func(tm1.ravel(), tm2.ravel())
    except:
        return np.nan