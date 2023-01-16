import numpy as np
import pandas as pd
import numpy.typing as npt
# try except to accomodate older versions of Python
try:
    from typing import Optional, Sequence, Union, Protocol, Iterable, Tuple
except ImportError:
    from typing import Optional, Sequence, Union, Iterable, Tuple
    from typing_extensions import Protocol


def merge_dict(dct, clobbering_dct):
	return {**dct, **clobbering_dct}


def rle(array: Union[pd.DataFrame, pd.Series, np.ndarray], convert=True) -> pd.Series:
    diffs_raw = np.abs(np.diff(array)) > 0
    diffs = np.ones((len(array),), dtype="bool")
    diffs[1:] = diffs_raw
    array_rle = array[diffs]
    if convert:
        return pd.Series(array_rle).astype("category")
    else:
        return pd.Series(array_rle)


def zscore(arr):
    '''This function does not implement dimension-checking, so make
    sure the dimensionality of your array is 1-D for numpy arrays.
    Handles pandas Series and DataFrames fine.
    '''
    if isinstance(arr, np.ndarray):
        return (arr - np.nanmean(arr)) / np.nanstd(arr)
    # otherwise, assume pandas datastruct
    return (arr - arr.mean()) / arr.std()


# hampel_filter
default_window = {"window": 10, "min_periods": 1, "center": True}


def hampel_filter(trace: pd.DataFrame, threshold: float = 10, **rolling_kwargs) -> pd.DataFrame:
    use_kwargs = {**default_window, **rolling_kwargs}
    meds = trace.rolling(**use_kwargs).median()
    med_dev = (trace - meds).abs()
    mads = rolling_mad(trace, **use_kwargs)
    diffs = med_dev / mads
    threshold_cross = diffs > threshold
    trace[threshold_cross] = meds[threshold_cross]
    return trace


# WIN'S ORIGINAL
# def rolling_mad(data, scale=0.6744897501960817, **rolling_kwargs):
#     med = data.rolling(**rolling_kwargs).median()
#     _abs = (data - med).abs()
#     return _abs.rolling(**rolling_kwargs).median() / scale


# Jeff's type annotated version
def rolling_mad(
    data: pd.DataFrame, scale: float = 0.6744897501960817, **rolling_kwargs
) -> pd.DataFrame:
    use_kwargs = {**default_window, **rolling_kwargs}
    med = data.rolling(**use_kwargs).median()
    _abs = (data - med).abs()
    return _abs.rolling(**use_kwargs).median() / scale


def peak_score(
    x: pd.DataFrame,
    smoothing: int = 50,
    power: float = 3,
    diff_order: int = 2,
    zscore: bool = True,
    rectify: Optional[str] = None,
) -> pd.DataFrame:
    if rectify == "pos":
        x[x < 0] = 0
    elif rectify == "neg":
        x[x > 0] = 0

    if zscore:
        use_vals = (x - x.mean()) / x.std()
    else:
        use_vals = x

    return use_vals.rolling(smoothing, center=True).mean().diff(diff_order).pow(power)


def get_peaks(
    x: pd.DataFrame,
    # trim_border: Tuple[int, int] = (100, 100),
    negate: bool = False,
    duration_threshold: float = 5,
    zscore: bool = False,
    **kwargs,
) -> pd.Series:
    from scipy.signal import find_peaks

    defaults = {"height": 0.001, "prominence": 0.001}
    use_vals = x.fillna(0).values
    if negate:
        use_vals *= -1
    # use_vals[use_vals<rectify] = rectify

    if zscore:
        use_vals = (x - np.nanmean(x)) / np.nanstd(x)
    peak_loc = find_peaks(use_vals, **{**defaults, **kwargs})[0]
    neg_peak_loc = find_peaks(-use_vals, **{**defaults, **kwargs})[0]

    if duration_threshold is not None:
        durs = []
        for _peak in peak_loc:
            if negate:
                nearest_neg = _peak - neg_peak_loc
            else:
                nearest_neg = neg_peak_loc - _peak
            nearest_neg = nearest_neg[nearest_neg > 0]
            if len(nearest_neg) > 0:
                durs.append(np.min(nearest_neg))
            else:
                durs.append(np.nan)

        durs = np.array(durs)
        if duration_threshold is not None:
            peak_loc = peak_loc[durs > duration_threshold]

    return_vals = pd.Series(data=np.zeros((len(x),), dtype="bool"), index=x.index)
    return_vals.iloc[peak_loc] = True

    return return_vals


def count_transitions(seq: Union[np.ndarray, pd.DataFrame], K: int = 100):
    trans = np.zeros((K, K))
    # coerce to ints just in case...
    seq = np.asarray(seq.copy(), dtype="int")
    if seq.ndim == 1:
        np.add.at(trans, (seq[:-1], seq[1:]), 1)
    elif seq.ndim == 2:
        np.add.at(trans, (seq[:, 0], seq[:, 1]), 1)
        for i in range(1, seq.shape[1] - 1):
            np.add.at(trans, (seq[:, i], seq[:, i + 1]), 1) # 10/6/2022 found this without a third arg
    return trans.astype("float")


def randomize_rows(values: np.ndarray) -> None:
    [np.random.shuffle(values[:, i]) for i in range(values.shape[1])]
    return None


def randomize_cols(values: np.ndarray) -> None:
    [np.random.shuffle(values[i]) for i in range(values.shape[0])]
    return None


def pd_zscore(x: pd.DataFrame) -> pd.DataFrame:
    return (x - x.mean()) / x.std()


# https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model
class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...
    def score(self, X, y, sample_weight=None): ...
    def set_params(self, **params): ...


class ScikitSplit(Protocol):
    def split(self, X, y=None, groups=None) -> Iterable: ...


# adapted from https://gist.github.com/joelouismarino/ce239b5601fff2698895f48003f7464b
def whiten(
    X: np.ndarray, method: str = "zca", eps: float = 1e-5
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
        eps: smoothing constant
    """

    # remove nans, put back in at the end

    result = np.zeros_like(X)
    result[:] = np.nan
    nans = np.isnan(X).any(axis=1)

    X_hat = X[~nans]
    X_hat = X_hat.reshape((-1, np.prod(X_hat.shape[1:])))
    mu = np.mean(X_hat, axis=0)
    X_centered = X_hat - mu
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None

    if method in ["zca", "pca", "cholesky"]:
        U, Lambda, _ = np.linalg.svd(Sigma)

        # adjust column signs for PCA
        U = U * np.sign(np.diag(U))[None, :]

        if method == "zca":
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + eps)), U.T))
        elif method == "pca":
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + eps)), U.T)
        elif method == "cholesky":
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + eps)), U.T))).T
        else:
            raise RuntimeError("Did not understand method")
    elif method in ["zca_cor", "pca_cor"]:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)

        # adjust column signs for PCA
        G = G * np.sign(np.diag(G))[None, :]

        if method == "zca_cor":
            W = np.dot(
                np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + eps)), G.T)), np.linalg.inv(V_sqrt)
            )
        elif method == "pca_cor":
            W = np.dot(np.dot(np.diag(1.0 / np.sqrt(Theta + eps)), G.T), np.linalg.inv(V_sqrt))
        else:
            raise RuntimeError("Did not understand method")
    else:
        raise Exception("Whitening method not found.")

    result[~nans] = np.dot(X_centered, W.T)
    return result, W, mu


def get_snippets_vec(
    *args,
    win=[0.25, 2],
    meta_keys=["mouse_id", "timestamp"],
    fs=30.0,
    targets=np.arange(100),
    label_key="predicted_syllable",
    data_keys=["pc{:02d}".format(i) for i in range(10)],
):

    grp_df = args[-1].copy()

    # add support for additional labels,
    #     grp_df = grp_df.copy().set_index("timestamps")
    grp_df = grp_df.copy().set_index(np.arange(len(grp_df)))
    win_samples = (-np.round(win[0] * 30), np.round(win[1] * 30))

    with np.errstate(invalid="ignore"):
        label_seq = rle(grp_df[label_key])
    label_seq = label_seq.loc[label_seq.isin(targets)]
    label_seq = label_seq[
        (label_seq.index > -win_samples[0]) & (label_seq.index < (len(grp_df) - win_samples[1]))
    ]

    capture_idx = np.arange(*win_samples) + label_seq.index.to_numpy()[:, np.newaxis]
    capture_idx = capture_idx.ravel().astype("int")
    snippet = np.repeat(np.arange(len(label_seq)), len(np.arange(*win_samples)))
    x = np.tile(np.arange(*win_samples) / fs, len(label_seq))
    syllable = np.repeat(label_seq.to_numpy(), len(np.arange(*win_samples)))

    # prev_syllable = np.repeat(
    #     label_seq.shift(1, fill_value=np.nan).to_numpy(), len(np.arange(*win_samples))
    # )
    # next_syllable = np.repeat(
    #     label_seq.shift(-1, fill_value=np.nan).to_numpy(), len(np.arange(*win_samples))
    # )
    duration = np.repeat(
        np.append(np.diff(label_seq.index.to_numpy()) / fs, np.nan),
        len(np.arange(*win_samples)),
    )

    _df = pd.DataFrame()
    _df["x"] = x
    for _key in data_keys:
        _df[_key] = grp_df[_key].to_numpy().ravel()[capture_idx]
    for _key in meta_keys:
        _df[_key] = np.repeat(
            grp_df.loc[label_seq.index, _key].to_numpy(), len(np.arange(*win_samples))
        )
    _df["snippet"] = snippet
    # _df["uuid"] = grp_df.iloc[0]["uuid"]
    _df["syllable"] = syllable
    # _df["prev_syllable"] = prev_syllable
    # _df["next_syllable"] = next_syllable
    _df["duration"] = duration
    # _df["timestamp"] = timestamp

    # for v in meta_mapping.values():
    #     _df[f"{v}"] = grp_df.iloc[0][f"{v}"]
    return _df
