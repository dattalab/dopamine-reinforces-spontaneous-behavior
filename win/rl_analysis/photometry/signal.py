import numpy as np
import pandas as pd
from scipy import signal
from sklearn import linear_model


def peak_indices(ser, thresh=1.5, min_peak_distance=4):
    if ser.isna().all():
        return np.nan
    inds, _ = signal.find_peaks(ser, height=thresh, distance=min_peak_distance)
    return inds


def rereference(
    x,
    y,
    center=False,
    center_fun=np.median,
    npoints=None,
    clip=False,
    clf=linear_model.RANSACRegressor(linear_model.LinearRegression(fit_intercept=False)),
):

    try:
        _x = x.copy().to_numpy()
    except AttributeError:
        _x = x.copy()
    
    try:
        _y = y.copy().to_numpy()
    except AttributeError:
        _y = y.copy()

    if npoints is None:
        npoints = len(_x)
    else:
        npoints = np.minimum(npoints, len(_x))

    nans = np.logical_or(np.isnan(_x), np.isnan(_y))

    use_x = _x[~nans][:npoints][:, None]
    use_y = _y[~nans][:npoints].ravel()

    if len(use_x) == 0:
        return x

    if center:
        mu_x = center_fun(use_x)
        mu_y = center_fun(use_y)
    else:
        mu_x = 0
        mu_y = 0

    # if we're centering, subtract off the mean/median/whatever prior to fitting
    clf.fit(use_x - mu_x, use_y - mu_y)

    # then we predict on centered data, put the center back in 
    newx = np.zeros_like(_x)
    newx[~nans] = clf.predict(_x[~nans][:, None] - mu_x) + mu_x

    # subtract off for our referenced signal
    newy = _y - newx

    if clip:
        newy = np.clip(newy, 0, np.inf)

    return_data = pd.DataFrame(index=y.index)
    return_data["reference_fit"] = newx
    return_data["rereference"] = newy

    try:
        return_data["reference_fit_slope"] = clf.estimator_.coef_[0]
        try:
            return_data["reference_fit_intercept"] = clf.estimator_.coef_[1]
        except Exception:
            pass
    except Exception:
        return_data["reference_fit_slope"] = clf.coef_[0]
        try:
            return_data["reference_fit_intercept"] = clf.coef_[1]
        except Exception:
            pass

    return return_data


def rolling_mad(data, scale=0.6744897501960817, **rolling_kwargs):
    med = data.rolling(**rolling_kwargs).median()
    _abs = (data - med).abs()
    return _abs.rolling(**rolling_kwargs).median() / scale


def rolling_fluor_normalization(data, window_size=10, fps=30,
                                normalizer='rdff', quantile=0.1):
    '''
    Args:
        window_size (float): size of the rolling window in seconds
    '''
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    window = int(window_size * fps)
    rolling = data.rolling(window, min_periods=5, center=True)
    # first, baseline the data
    if normalizer in ('rdff', 'quantile baseline', 'dff'):
        baselined = data - rolling.quantile(quantile)
    elif normalizer in ('zscore', 'mean baseline'):
        baselined = data - rolling.mean()
    elif normalizer == 'rzscore':
        baselined = data - rolling.median()
    else:
        raise Exception('Not a normalizer option. Choose: rdff, rzscore, zscore, quantile baseline, mean baseline')

    if 'baseline' in normalizer:
        return baselined
    # now normalize
    if normalizer == 'dff':
        return baselined / rolling.quantile(quantile)
    
    rolling = baselined.rolling(window, min_periods=5, center=True)
    if normalizer in ('rdff', 'rzscore'):
        normalized = baselined / rolling_mad(baselined, window=window, min_periods=5, center=True)
    elif normalizer == 'zscore':
        normalized = baselined / rolling.std()
    return normalized