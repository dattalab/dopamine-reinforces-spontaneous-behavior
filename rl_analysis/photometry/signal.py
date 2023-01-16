from typing import Union, Callable, Optional
from typing import Union
from scipy import signal
from sklearn import linear_model
from rl_analysis.util import rolling_mad
from numba import jit
import numpy.typing as npt
import numpy as np
import pandas as pd


def compute_photometry_quality_metrics(
	df_phot: pd.DataFrame,
	pmts: list[str] = ["signal_dff", "reference_dff", "signal_reref_dff"],
	fs: float = 30.0,
) -> pd.DataFrame:

	from scipy.stats import pearsonr
	df_phot_smooth = df_phot.groupby("uuid")[pmts].transform(
		lambda x: x.rolling(window=50, win_type="exponential", center=True).mean(tau=10)
	)
	df_phot_smooth = df_phot_smooth.loc[df_phot.index]

	snr = (
		df_phot_smooth.groupby(df_phot["uuid"])["signal_dff"].quantile(0.9)
		/ df_phot_smooth.groupby(df_phot["uuid"])["reference_dff"].std()
	)

	signal_max = df_phot_smooth.groupby(df_phot["uuid"])["signal_dff"].quantile(0.9)
	reference_max = df_phot_smooth.groupby(df_phot["uuid"])["reference_dff"].quantile(0.9)

	def _pearsonr(x, y):
		x = x.copy()
		y = y.copy()
		nans = np.isnan(x) | np.isnan(y)
		x = x[~nans]
		y = y[~nans]
		if len(x) < 2:
			return np.nan
		else:
			return pearsonr(x, y)[0]

	signal_ref_corr = df_phot_smooth.groupby(df_phot["uuid"]).apply(
		lambda x: _pearsonr(x["signal_dff"], x["reference_dff"])
	)

	df_phot["snr"] = df_phot["uuid"].map(snr)
	df_phot["signal_reference_corr"] = df_phot["uuid"].map(signal_ref_corr)
	df_phot["signal_max"] = df_phot["uuid"].map(signal_max) * 100
	df_phot["reference_max"] = df_phot["uuid"].map(reference_max) * 100
	return df_phot


def peak_indices(ser, thresh=1.5, min_peak_distance=4):
    if ser.isna().all():
        return np.nan
    inds, _ = signal.find_peaks(ser, height=thresh, distance=min_peak_distance)
    return inds


# WIN'S ORIGINAL VERSION
# def rereference(
#     x,
#     y,
#     center=False,
#     center_fun=np.median,
#     npoints=None,
#     clip=False,
#     clf=linear_model.RANSACRegressor(linear_model.LinearRegression(fit_intercept=False)),
# ):

#     try:
#         _x = x.copy().to_numpy()
#     except AttributeError:
#         _x = x.copy()
    
#     try:
#         _y = y.copy().to_numpy()
#     except AttributeError:
#         _y = y.copy()

#     if npoints is None:
#         npoints = len(_x)
#     else:
#         npoints = np.minimum(npoints, len(_x))

#     nans = np.logical_or(np.isnan(_x), np.isnan(_y))

#     use_x = _x[~nans][:npoints][:, None]
#     use_y = _y[~nans][:npoints].ravel()

#     if len(use_x) == 0:
#         return x

#     if center:
#         mu_x = center_fun(use_x)
#         mu_y = center_fun(use_y)
#     else:
#         mu_x = 0
#         mu_y = 0

#     # if we're centering, subtract off the mean/median/whatever prior to fitting
#     clf.fit(use_x - mu_x, use_y - mu_y)

#     # then we predict on centered data, put the center back in 
#     newx = np.zeros_like(_x)
#     newx[~nans] = clf.predict(_x[~nans][:, None] - mu_x) + mu_x

#     # subtract off for our referenced signal
#     newy = _y - newx

#     if clip:
#         newy = np.clip(newy, 0, np.inf)

#     return_data = pd.DataFrame(index=y.index)
#     return_data["reference_fit"] = newx
#     return_data["rereference"] = newy

#     try:
#         return_data["reference_fit_slope"] = clf.estimator_.coef_[0]
#         try:
#             return_data["reference_fit_intercept"] = clf.estimator_.coef_[1]
#         except Exception:
#             pass
#     except Exception:
#         return_data["reference_fit_slope"] = clf.coef_[0]
#         try:
#             return_data["reference_fit_intercept"] = clf.coef_[1]
#         except Exception:
#             pass

#     return return_data


acceptable_models = Union[linear_model.LinearRegression, linear_model.RANSACRegressor]


# JEFF'S VERSION with type annotations
def rereference(
    x: pd.DataFrame,
    y: pd.DataFrame,
    center: bool = False,
    center_fun: Callable[[npt.ArrayLike], np.ndarray] = np.median,
    npoints: Optional[int] = None,
    clip: Optional[bool] = False,
    clf: acceptable_models = linear_model.RANSACRegressor(
        linear_model.LinearRegression(fit_intercept=False)
    ),
) -> pd.DataFrame:

    _x = x.copy().to_numpy()
    _y = y.copy().to_numpy()

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
    newy = _y - newx  # type: ignore

    if clip:
        newy = np.clip(newy, 0, np.inf)

    return_data = pd.DataFrame(index=y.index)
    return_data["reference_fit"] = newx
    return_data["rereference"] = newy

    try:
        return_data["reference_fit_slope"] = clf.estimator_.coef_[0]  # type: ignore
        try:
            return_data["reference_fit_intercept"] = clf.estimator_.coef_[1]  # type: ignore
        except Exception:
            pass
    except Exception:
        return_data["reference_fit_slope"] = clf.coef_[0]  # type: ignore
        try:
            return_data["reference_fit_intercept"] = clf.coef_[1]  # type: ignore
        except Exception:
            pass

    return return_data
    # return pd.Series(data=newy, index=y.index)




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


@jit(nopython=True)
def get_crossings(
    values: np.ndarray,
    threshold: float,
    schmitt: int,
    schmitt_threshold: float,
) -> np.ndarray:

    # find every point we go below then above
    crossings = np.logical_and(values[1:] > threshold, values[:-1] < threshold)
    crossings = np.flatnonzero(crossings)
    is_good = np.ones((len(crossings)), dtype="bool")

    for i, _crossing in enumerate(crossings):
        if (values[_crossing + 1 : _crossing + schmitt + 1] < schmitt_threshold).any():
            is_good[i] = False

    return_vec = np.zeros(
        (
            len(
                values,
            )
        ),
        dtype="bool",
    )
    return_vec[crossings[is_good]] = True
    return return_vec


def get_ncrossings(
    df: pd.DataFrame,
    threshold: float = 1.,
    schmitt: int = 3,
    schmitt_threshold: float = 0.25,
) -> float:
    # convert to rate via inde
    values, index = df.to_numpy(), df.index.to_numpy()
    crossings_vec = get_crossings(
        values,
        threshold=threshold,
        schmitt=schmitt,
        schmitt_threshold=schmitt_threshold,
    )
    return np.sum(crossings_vec) / (index[-1] - index[0])