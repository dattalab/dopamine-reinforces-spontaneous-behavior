from functools import reduce
from rl_analysis.util import rle, count_transitions
from typing import Sequence, Optional
import pandas as pd
import numpy as np


def normalize_df(
    *args,
    time_bins=[(0, 30)],
    eps=0,  # this is only added for division
    baseline_smoothing=None,
    outer_loop_key="target_syllable",
    target_only=False,
    label_key="predicted_syllable",
    use_rle=True,
    meta_keys=[],
    syllable_list=list(range(100)),
    adjust_to_bin_size=False,
    baseline="monday",
    monday_lookback=-5,
    week_lookback=-5,
):
    import warnings

    # throughout this function, COUNT (CNT) refers to raw counts and USAGE refers to fractional counts
    dfs = []
    df = args[-1]

    # (a)bsolute baseline uses the first baseline session for each mouse...
    # (l)ocal baseline uses the previous recording day
    # (m)onday baseline uses the previous Monday
    if baseline[0].lower() == "a":
        # baseline is the first day w/ session numbers <= 0
        baselines = df[df["session_number"] <= 0]
        rnd_dates = baselines["date"].dt.floor("d")
        baseline_days = baselines[rnd_dates == rnd_dates.min()]
    elif baseline[0].lower() == "all":
        baselines = df[df["session_number"] <= 0]

    for _group, grp in df.groupby(outer_loop_key, as_index=False):

        # print_key = list(_group)
        # if len(args) > 1:
        #     print_key += [args[0]]

        if baseline[0].lower() == "l":
            # baseline_days = grp[grp["session_number"] <= 0]
            baselines = df[df["session_number"] <= 0]
            rnd_dates = baselines["date"].dt.floor("d")
            baseline_days = baselines[rnd_dates == rnd_dates.max()]
        elif baseline[0].lower() == "m":
            baselines = df[df["session_number"] <= 0]
            # get the baseline from monday
            first_stim_day = grp[grp["session_number"] == 1]["date"]
            if len(first_stim_day) > 0:
                first_stim_day = first_stim_day.iloc[0]
            else:
                continue

            # get all mondays before, don't go beyond monday_lookback
            diffs = (baselines["date"].dt.floor("d") - first_stim_day.floor("d")).dt.days
            baselines = baselines[(diffs < 0) & (diffs >= monday_lookback)]
            mondays = baselines["date"].dt.dayofweek == 0

            if mondays.any():
                baselines = baselines[mondays]
            rnd_dates = baselines["date"].dt.floor("d")
            # get the most recent Monday (or baseline day if it doesn't exist)...
            baseline_days = baselines[rnd_dates == rnd_dates.max()]
        elif baseline[0].lower() == "w":
            baselines = df[df["session_number"] <= 0]
            # get the baseline from monday
            # baselines = baselines[baselines["date"].dt.dayofweek == 0]
            first_stim_day = grp[grp["session_number"] == 1]["date"].iloc[0]
            # get all mondays before
            baselines = baselines[baselines["date"].dt.floor("d") < first_stim_day.floor("d")]
            baselines = baselines[
                (baselines["date"].dt.floor("d") - first_stim_day.floor("d")).dt.days
                >= week_lookback
            ]
            rnd_dates = baselines["date"].dt.floor("d")
            # get the most recent Monday...
            baseline_days = baselines[rnd_dates == rnd_dates.min()]
        else:
            raise RuntimeError("Did not understand baseline option")

        # get usage for all baseline days
        # add an option to average across time (compute some more baselines son)
        baseline_usages = []
        for k, _baseline in baseline_days.groupby("uuid"):
            _baseline_usage = get_binned_usage(
                _baseline,
                label_key=label_key,
                use_rle=use_rle,
                time_bins=time_bins,
                syllable_list=syllable_list,
                adjust_to_bin_size=adjust_to_bin_size,
            )
            baseline_usages.append(_baseline_usage)

        try:
            print_key = list(_group)
        except TypeError:
            print_key = [_group]
        print_key += [grp.iloc[0]["mouse_id"]]
        # do we care about this?
        if len(baseline_usages) == 1:
            warnings.warn(f"Less than two baseline sessions for {print_key}")
        elif len(baseline_usages) < 1:
            warnings.warn(f"Less than one baseline session for {print_key}")
            continue

        # gets the average across different baseline sessions
        baseline_count = reduce(lambda x, y: x.add(y, fill_value=np.nan), baseline_usages) / len(
            baseline_usages
        )
        if baseline_smoothing is not None:
            baseline_count = baseline_count.rolling(**baseline_smoothing, axis=1).mean()

        # this may be obselete now...(this would average across time bins...)
        # if agg_baseline:
        # baseline_count = baseline_count.mean(axis=1)

        # now we have a dataframe with syllable counts for each time bin
        # let's keep track of raw counts and fractional counts (i.e. usages)

        baseline_count = baseline_count.rename("baseline_count")
        baseline_count_eps = baseline_count + eps

        baseline_usage = (baseline_count / baseline_count.groupby("bin").sum()).rename(
            "baseline_usage"
        )
        baseline_usage_eps = baseline_count_eps / baseline_count_eps.groupby("bin").sum()

        for k, _session in grp.groupby(["session_number", "uuid"]):
            # get raw counts, normalize myriad ways, then subtract/divide baseline
            cnt_df = get_binned_usage(
                _session,
                label_key=label_key,
                use_rle=use_rle,
                time_bins=time_bins,
                syllable_list=syllable_list,
                adjust_to_bin_size=adjust_to_bin_size,
            )

            usage_df = cnt_df.copy().rename("usage")
            usage_df = usage_df / usage_df.groupby("bin").sum()

            # ONLY used for division
            usage_df_eps = cnt_df + eps
            usage_df_eps = usage_df_eps / usage_df_eps.groupby("bin").sum()

            # reformat the dataframes for downstream processing
            change_usage = (usage_df - baseline_usage).rename("change_usage")
            fold_change_usage = (usage_df_eps / baseline_usage_eps).rename("fold_change_usage")

            change_count = (cnt_df - baseline_count).rename("change_count")
            fold_change_count = ((cnt_df + eps) / baseline_count_eps).rename("fold_change_count")

            return_df = pd.concat(
                [
                    cnt_df,
                    usage_df,
                    change_count,
                    change_usage,
                    fold_change_count,
                    fold_change_usage,
                    baseline_count,
                    baseline_usage,
                ],
                axis=1,
            )
            return_df = return_df.reset_index()
            return_df["bin_start"] = return_df["bin"].apply(lambda x: x.left).astype("float")
            return_df["bin_end"] = return_df["bin"].apply(lambda x: x.right).astype("float")
            return_df["bin_mid"] = return_df["bin"].apply(lambda x: x.mid).astype("float")

            # return_df["bin_mid"] = return_df["bin"].apply(np.mean)

            return_df = return_df.drop("bin", axis=1)
            for _meta in meta_keys:
                return_df[_meta] = _session.iloc[0][_meta]

            if target_only:
                return_df = return_df[return_df["index"] == _session.iloc[0]["target_syllable"]]

            dfs.append(return_df)

    if len(dfs) > 0:
        return pd.concat(dfs)
    else:
        return None


def filter_feedback_dataframe(
    df,
    first_timestamp_cutoff=10,
    last_timestamp_cutoff=1750,
    missing_frame_cutoff=0.05,
    largest_gap_cutoff=300,
    fs=30.0,
):

    # no need for habituation data...
    df = df.loc[df["target_syllable"] >= 0].copy()

    # first_timestamp_cutoff in seconds, first timestamp must be less than this
    # last_timestamp_cutoff in seconds, last timestamp must be greater than this
    # missing_frame_cutoff fraction of missing frames must be less than this
    # largest_gap_cutoff largest gap in frames must be less than this
    # filters out any sessions that where prematurely cut off
    df = df.groupby("uniq_id").filter(lambda x: x["timestamp"].max() > last_timestamp_cutoff)
    df = df.groupby("uniq_id").filter(lambda x: x["timestamp"].min() < first_timestamp_cutoff)

    missing_frames = df.groupby(["uniq_id"])["timestamp"].apply(
        lambda x: (x.diff() / (1 / fs) - 1).round().sum()
    )

    largest_gap = df.groupby(["uniq_id"])["timestamp"].apply(
        lambda x: (x.diff() / (1 / fs) - 1).round().max()
    )
    fraction_missing = missing_frames / (
        (df.groupby("uniq_id")["timestamp"].max() - df.groupby("uniq_id")["timestamp"].min()) * fs
    )

    keep_df = (fraction_missing < missing_frame_cutoff) & (largest_gap < largest_gap_cutoff)
    preserve_ids = keep_df.loc[keep_df].index
    df = df.loc[df["uniq_id"].isin(preserve_ids)]

    return df


def get_binned_usage(
    df: pd.DataFrame,
    label_key: str = "predicted_syllable",
    time_key: str = "timestamp",
    use_rle: bool = False,
    syllable_list: Sequence[int] = list(range(100)),
    time_bins: Sequence[float] = [0, 30 * 60],
    adjust_to_bin_size: bool = True,
):
    use_df = df[[label_key, time_key]].rename(columns={label_key: "syllable"}).copy()
    use_df["bin"] = pd.cut(use_df[time_key], time_bins)

    bin_size_actual = use_df.groupby("bin")[time_key].apply(lambda x: x.max() - x.min())
    bin_size_desired = [_.right - _.left for _ in use_df["bin"].cat.categories]
    bin_size_desired = pd.Series(data=bin_size_desired, index=use_df["bin"].cat.categories)

    correction = bin_size_desired / bin_size_actual
    correction.index.name = "bin"

    if use_rle:
        count_df = rle(use_df["syllable"])
        count_df = use_df.loc[count_df.index].dropna().copy()
    else:
        count_df = use_df.dropna().copy()

    count_df["syllable"] = count_df["syllable"].astype("category").cat.set_categories(syllable_list)
    return_count = count_df.groupby("bin")["syllable"].value_counts().sort_index()
    return_count.index = return_count.index.set_names("syllable", level=-1)
    return_count = return_count.reorder_levels(["syllable", "bin"]).rename("count")  # type: ignore

    if adjust_to_bin_size:
        return_count *= correction

    return return_count


def get_stim_effects(
    #     *args,
    values: pd.DataFrame,
    dlight_features: list[str] = [],
    syllable_key: str = "syllable",
    chk_syllables: np.ndarray = np.arange(100),
    usage_bins: np.ndarray = np.arange(0,100,10),
    meta_keys: list[str] = [
        "timestamp",
        "mouse_id",
        "syllable_number",
        "session_number",
        "stim_status",
        "opsin",
        "power",
        "area",
        "target_syllable",
        "target_syllable_original_id",
        "stim_duration",
        "experiment_type",
    ],
    additional_scalar_keys: list[str] = ["duration"],
    additional_syllable_keys: list[str] = ["duration"],
    target_only: bool = True,
    tm_dtype: str = "uint16",
    K: int = 100,
    truncate: int = 37
) -> Optional[pd.DataFrame]:
    dcts = []
    #     values = args[-1]
    syllables = values[syllable_key]
    syllable_lst = list(syllables.unique())
    if target_only:
        use_syllables = [values["target_syllable"].iloc[0]]
    else:
        use_syllables = [_ for _ in chk_syllables if _ in syllable_lst]

    # pad values
    # loop through each syllable, get dlight feature and then count outbound transitions in
    # the bins...

    # every transition it's own dataframe?
    for _syllable in use_syllables:

        use_trans = np.flatnonzero(syllables == _syllable)

        if len(use_trans) == 0:
            continue

        # hack
        max_abs_bin = np.max(np.abs(usage_bins))
        use_trans = use_trans[(use_trans > max_abs_bin) & (use_trans < len(syllables) - max_abs_bin)]
#         use_trans = use_trans[(use_trans < len(syllables) - np.max(usage_bins))]

        syllable_dfs = []
        for _trans in use_trans:

            # get transitions in the right range
            for _bin in usage_bins:

                if np.sign(_bin) < 0:
                    left_edge = _trans + _bin
                    right_edge = _trans
                else:
                    # note looked good with +1 shift here...
                    left_edge = _trans
                    right_edge = _trans + _bin
                # all transitions to the right of the current transition and less than _bin left inclusive
                syllable_sequence = syllables.iloc[left_edge:right_edge].values
                binned_values = values.iloc[left_edge:right_edge]

                # now we can index other values by syllable occurrences left inclusive
                future_syllable_trans = use_trans[
                    (use_trans >= left_edge) & (use_trans < right_edge)
                ]
                syllable_values = values.iloc[future_syllable_trans]

                if len(syllable_sequence) != np.abs(_bin):
                    RuntimeError("Indexing issue")
                #                     continue

                dct = {}
                dct["tm"] = count_transitions(syllable_sequence, K=K).astype(tm_dtype)

                for _feature in dlight_features:
                    dct[_feature] = values[_feature].iat[_trans]

                for _key in meta_keys:
                    dct[_key] = values[_key].iat[_trans]

                for _scalar in additional_scalar_keys:
                    dct[f"{_scalar}_bin"] = binned_values[_scalar].mean()

                for _key in additional_syllable_keys:
                    dct[f"{_key}"] = syllable_values[_key].mean()

                dct["tm"] = dct["tm"][:truncate,:truncate]
                dct["syllable"] = _syllable
                dct["count"] = len(future_syllable_trans)
                dct["bin"] = _bin
                dct["trans_number"] = int(_trans)
                dcts.append(dct)

    if len(dcts) == 0:
        return None
    else:
        return pd.DataFrame(dcts).reset_index()