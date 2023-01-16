import numpy as np
import pandas as pd
from rl_analysis.util import rle
from typing import Callable, Sequence, Optional, Union
from functools import partial


def align_window_to_label(
    label_key: str = "labels",
    include_labels: np.ndarray = np.arange(100),
    window_bounds: tuple[float, float] = (-10, 10),
    fs: float = 30,
    data_keys: Sequence[str] = [
        "signal_dff",
        "reference_dff",
        "dlight normalized",
        "mcherry normalized",
        "velocity_3d_mm",
        "velocity_2d_mm",
        "height_ave_mm",
        "timestamps",
    ],
    meta_keys: Sequence[str] = ["uuid", "session_name", "subject_name", "mouse_id"],
) -> Callable:
    """Equivalent to jeff's `get_snippets_vec` function. Apply a window
    to each onset of a label. For instance, generate windows for each
    syllable transitions. Useful for plotting dlight dynamics. Calls a
    second function designed to run on the elements of a dataframe's
    groupby object

    Parameters:
        window_bounds (tuple[int]): time in seconds before and after label onset"""
    func = partial(
        window_dataframe,
        label_key=label_key,
        include_labels=include_labels,
        window_bounds=window_bounds,
        data_keys=data_keys,
        meta_keys=meta_keys,
        fs=fs,
        use_onset=False,
    )
    return func
    # out = groupby_parallel(df.groupby(gb_key, observed=True, sort=False), func, num_cpus)
    # return out


def window_dataframe(
    grp_df: pd.DataFrame,
    label_key: str,
    data_keys: Sequence[str],
    meta_keys: Sequence[str],
    include_labels: np.ndarray,
    window_bounds: tuple[float, float] = (-10, 10),
    fs: float = 30,
    use_onset: bool = False,
) -> pd.DataFrame:
    win_samples = np.round(np.array(window_bounds) * fs)
    window = np.arange(*win_samples)

    grp_df = grp_df.reset_index(drop=True)

    if use_onset and ("onset" in grp_df):
        label_seq = grp_df.loc[grp_df["onset"].astype("bool"), label_key]
    else:
        with np.errstate(invalid="ignore"):
            label_seq = rle(grp_df[label_key])
    label_seq = label_seq[label_seq.isin(include_labels)]
    label_seq = label_seq[
        ((label_seq.index + win_samples[0]) > 0)
        & ((label_seq.index + win_samples[1]) < len(grp_df))
    ]

    capture_idx = window + label_seq.index.to_numpy()[:, None]
    incl_idx = np.all(capture_idx > 0, axis=1) & np.all(capture_idx < len(grp_df), axis=1)
    capture_idx = capture_idx[incl_idx]

    # 3/10/2022 MODIFIED DURATION TO ACCOUNT FOR EXCLUSIONS!
    duration = np.repeat(
        np.append(np.diff(label_seq.index.to_numpy()) / fs, np.nan)[incl_idx], len(window)
    )
    capture_idx = capture_idx.ravel().astype("int")

    snippet = np.repeat(np.arange(len(label_seq)), len(window))
    x = np.tile(window / fs, len(label_seq))
    syllable = np.repeat(label_seq.to_numpy(), len(window))

    out = {}
    for _key in data_keys:
        out[_key] = grp_df[_key].to_numpy().ravel()[capture_idx]
    for _key in meta_keys:
        out[_key] = np.repeat(grp_df.loc[label_seq.index, _key].to_numpy(), len(window))

    out["x"] = x.astype("float32")
    out["snippet"] = snippet.astype("uint16")
    out["syllable"] = syllable
    # out["prev_syllable"] = prev_syllable
    # out["next_syllable"] = next_syllable
    out["duration"] = duration.astype("float32")

    out = pd.DataFrame(out)
    out["syllable"] = out["syllable"].astype("UInt16")
    return out


def renormalize_df(
    df: pd.DataFrame,
    normalize_keys: list[str],
    norm_range: tuple[float, float] = (-10, +10),
    recenter_range: Optional[tuple[float, float]] = None,
) -> pd.DataFrame:
    _idx = pd.IndexSlice
    df = df.set_index(["snippet", "x"])
    df = df.sort_index()

    new_keys = [f"{_key}_z" for _key in normalize_keys]

    for _zkey, _key in tqdm(zip(new_keys, normalize_keys), total=len(new_keys)):
        df_mu = df.loc[_idx[:, norm_range[0] : norm_range[1]], :].groupby(["snippet"])[_key].mean()
        df_sig = df.loc[_idx[:, norm_range[0] : norm_range[1]], :].groupby(["snippet"])[_key].std()
        df[_zkey] = np.nan
        df[_zkey] = (df[_key] - df_mu) / df_sig

    if recenter_range is not None:
        for _key in tqdm(new_keys):
            df_mu = (
                df.loc[_idx[:, recenter_range[0] : recenter_range[1]], :]
                .groupby(["snippet"])[_key]
                .mean()
            )
            df[_key].values[:] = (df[_key] - df_mu).values

    df = df.replace([np.inf, -np.inf], np.nan).reset_index()
    return df


# THIS ONLY WORKS IF ALL SNIPPETS ARE COMPLETE/HAVE THE SAME NUMBER OF SAMPLES
def get_shuffle_average(
    idx: int,
    use_df: Optional[pd.DataFrame],
    dlight_bin: list[str] = ["max_bin", "x"],
    shuffle_x: bool = True,
    shuffle_quantity: Optional[pd.Series] = None,
    max_shift: int = 60,
) -> Optional[pd.DataFrame]:

    if use_df is None:
        return None

    proc_df = use_df.copy()
    rng = np.random.default_rng(idx)
    groupby_arrays = [proc_df[_] for _ in dlight_bin]

    if shuffle_x:
        rando_x = proc_df.groupby("snippet")["x"].transform(
            lambda x: np.roll(x, rng.integers(-max_shift, +max_shift))
        )
        # rando_x = pd.Series(
        #     np.concatenate(
        #         [
        #             np.roll(xvec, rng.integers(-max_shift, max_shift))
        #             for _ in range(nsnippets)
        #         ]
        #     ),
        #     index=proc_df.index,
        # ).rename("x")

        if "x" in dlight_bin:
            del groupby_arrays[dlight_bin.index("x")]

        groupby_arrays.append(rando_x)

    if shuffle_quantity is not None:
        use_idx = rng.choice(shuffle_quantity, len(shuffle_quantity), replace=True)
        use_idx = proc_df[shuffle_quantity.name].isin(use_idx)  # type: ignore
        groupby_arrays = [_.loc[use_idx] for _ in groupby_arrays]
        ave = (
            proc_df.loc[use_idx]
            .drop(["x"] + dlight_bin, axis=1)
            .groupby(groupby_arrays, observed=True)
            .mean()
        )
    else:
        ave = proc_df.drop(["x"] + dlight_bin, axis=1).groupby(groupby_arrays, observed=True).mean()

    ave["idx"] = idx
    return ave


def compute_numba_feature(
    time_win,
    use_df: pd.DataFrame,
    numba_feature: Callable,
    keys: list[str] = ["signal_reref_dff_z"],
) -> pd.DataFrame:

    if time_win[1] == np.inf:
        group_obj = use_df.loc[
            (use_df["x"] >= time_win[0]) & (use_df["x"] <= use_df["duration"])
        ].groupby("snippet")[keys]
    else:
        # change from LEFT TO BOTH 1/9/2023
        # change back to LEFT from BOTH 1/9/2023
        group_obj = use_df.loc[use_df["x"].between(*time_win, inclusive="both")].groupby("snippet")[
            keys
        ]

    feature = group_obj.agg(numba_feature, engine="numba").astype("float32")
    feature.columns = pd.MultiIndex.from_tuples(
        [(_, numba_feature.__name__) for _ in feature.columns]
    )
    feature["window"] = pd.arrays.IntervalArray.from_tuples(
        [time_win] * len(feature), closed="both"
    )
    return feature


def compute_pandas_feature(
    time_win,
    use_df: pd.DataFrame,
    pandas_feature: Union[str, Callable],
    keys: list[str] = ["signal_reref_dff_z"],
) -> pd.DataFrame:

    if time_win[1] == np.inf:
        group_obj = (
            use_df.loc[(use_df["x"] >= time_win[0]) & (use_df["x"] <= use_df["duration"])]
            .set_index("x_samples")
            .groupby("snippet")[keys]
        )
    else:
        # change from LEFT TO BOTH 1/9/2023
        group_obj = (
            use_df.loc[use_df["x"].between(*time_win, inclusive="both")]
            .set_index("x_samples")
            .groupby("snippet")[keys]
        )

    feature = group_obj.agg(pandas_feature).astype("float32")

    try:
        func_name = pandas_feature.__name__
    except AttributeError:
        func_name = pandas_feature
    feature.columns = pd.MultiIndex.from_tuples([(_, func_name) for _ in feature.columns])
    feature["window"] = pd.arrays.IntervalArray.from_tuples(
        [time_win] * len(feature), closed="both"
    )
    return feature
