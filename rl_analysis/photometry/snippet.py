import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, Union, Callable


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
