import pandas as pd
import numpy as np
import warnings
from typing import Optional
from rl_analysis.util import count_transitions


def get_syllable_and_scalar_rates(
    values: pd.DataFrame,
    dlight_features: list[str] = [
        "signal_reref_dff_norm_rising_range",
        "signal_reref_dff_norm_rising_slope",
        "signal_reref_dff_norm_rising_max",
    ],
    syllable_key: str = "syllable",
    chk_syllables: np.ndarray = np.arange(100),
    usage_bins: np.ndarray = np.arange(0, 100, 10),
    meta_keys: list[str] = [
        "timestamp",
        "duration",
        "mouse_id",
        "snippet",
        "area",
        "opsin",
        "signal_max",
        "signal_reference_corr",
        "stim_duration",
        "target_syllable",
        "session_number",
    ],
    additional_keys_global: list[str] = [],
    additional_keys_specific: list[str] = [],
    specific_bin_width: int = 10,
) -> pd.DataFrame:
    #     syllables = values[syllable_key].unique().tolist()
    # for every syllable, bin dlight and check usage patterns
    dcts = []

    # one hot syllables
    onehot_syllables = pd.get_dummies(values[syllable_key])
    #     onehot_syllables = onehot_syllables.reset_index()

    all_syllables = list(onehot_syllables.columns)
    use_syllables = onehot_syllables.columns.intersection(chk_syllables)  # type: ignore

    # for each column we need the syllable numbe
    onehot_syllables = onehot_syllables.loc[:, use_syllables]  # type: ignore

    excluded_syllables = set(all_syllables) - set(use_syllables)
    if len(excluded_syllables) > 0:
        raise UserWarning(f"{excluded_syllables} will be excluded in downstream analysis")

    # pad by max binsize
    pad = np.max(usage_bins)

    # pad values
    use_values = values.copy()
    onehot_syllables.index = np.arange(len(onehot_syllables))
    use_values.index = onehot_syllables.index

    # pad everything out
    new_index = np.arange(-pad, len(onehot_syllables) + pad)

    # onehot gets padded with zeros
    onehot_syllables = onehot_syllables.reindex(new_index, fill_value=0)

    # feature vectors are padded with nans
    use_values = use_values.reindex(new_index, fill_value=np.nan)

    # reset index to consecutive numbers again so we can access via numpy
    onehot_syllables.index = np.arange(len(onehot_syllables))
    use_values.index = onehot_syllables.index

    # make a copy of onehot syllables that factors in duration (replace 1 with duration in seconds)
    onehot_syllables_dur = onehot_syllables.copy().astype("float")
    onehot_syllables_dur.values[onehot_syllables_dur != 0] = values["duration"]

    for _syllable in onehot_syllables:

        use_idx = onehot_syllables.index[onehot_syllables[_syllable] == True].astype(
            "int"
        )

        if len(use_idx) == 0:
            continue

        syll_counts = onehot_syllables[_syllable]
        syll_counts_dur = onehot_syllables_dur[_syllable]
        use_values_specific = use_values.copy()
        use_values_specific.loc[
            onehot_syllables[_syllable] != True
        ] = np.nan  # nan out where != syllable

        for _bin in usage_bins:

            # don't include the current syllable in counts
            syll_capture_idx = np.arange(1, _bin) + use_idx.to_numpy()[:, None]

            # we can include the current syllable in scalars
            scalar_capture_idx = np.arange(0, _bin) + use_idx.to_numpy()[:, None]

            counts = np.nansum(syll_counts.values[syll_capture_idx], axis=1)

            # total seconds the syllable was used
            counts_dur = np.nansum(syll_counts_dur.values[syll_capture_idx], axis=1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                # get additional data
                tmp = {}
                for (_offset, _key) in additional_keys_global:
                    capture_idx = np.arange(_offset, _bin) + use_idx.to_numpy()[:, None]
                    tmp[f"{_key}_global_bin"] = np.nanmean(
                        use_values[_key].values[capture_idx], axis=1
                    )

                # we can't use the same binning logic here, need to use a sliding window
                for (_offset, _key) in additional_keys_specific:
                    left_edge = np.maximum(0, _bin - specific_bin_width)
                    capture_idx = (
                        np.arange(left_edge + _offset, _bin)
                        + use_idx.to_numpy()[:, None]
                    )
                    tmp[f"{_key}_specific_bin"] = np.nanmean(
                        use_values_specific[_key].values[capture_idx], axis=1
                    )

            df = pd.DataFrame(data=counts, columns=["count"])
            df["total_duration"] = counts_dur
            df[dlight_features] = use_values.loc[use_idx, dlight_features].values
            df[meta_keys] = use_values.loc[use_idx, meta_keys].values
            df["syllable"] = _syllable
            df["bin"] = _bin
            for k, v in tmp.items():
                df[k] = v
            dcts.append(df)
    return pd.concat(dcts)


def get_syllable_sequence_stats(
#     *args,
    values: pd.DataFrame,
    dlight_features: list[str] = [
        "signal_reref_dff_norm_rising_range",
        "signal_reref_dff_norm_rising_slope",
        "signal_reref_dff_norm_rising_max",
    ],
    syllable_key: str = "syllable",
    chk_syllables: np.ndarray = np.arange(100),
    usage_bins: np.ndarray = np.arange(0, 100, 10),
    meta_keys: list[str] = [
        "timestamp",
        "duration",
        "mouse_id",
        "snippet",
        "area",
        "opsin",
        "signal_max",
        "signal_reference_corr",
        "stim_duration",
        "target_syllable",
        "session_number",
    ],
    K: int = 100,
    truncate: int = 37
) -> Optional[pd.DataFrame]:
    dcts = [] 
#     values = args[-1]
    syllables = values[syllable_key]
    syllable_lst = list(syllables.unique())
    use_syllables = [_ for _ in chk_syllables if _ in syllable_lst]
    
    # pad values
    use_values = values.copy()
    
    excluded_syllables = set(syllable_lst) - set(use_syllables)
    if len(excluded_syllables) > 0:
        raise UserWarning(f"{excluded_syllables} will be excluded in downstream analysis")

    # loop through each syllable, get dlight feature and then count outbound transitions in 
    # the bins...

    # every transition it's own dataframe?
    for _syllable in use_syllables:
        
        use_trans = np.flatnonzero(syllables == _syllable)
        
        if len(use_trans) == 0:
            continue
    
        # why you wanna do me like that?
        syllable_dfs = []
        for _trans in use_trans:
            
            # get transitions in the right range 
            for _bin in usage_bins:    
                
                # all transitions to the right of the current transition and less than _bin
                syllable_sequence = syllables.iloc[_trans:_trans + _bin].values
                
                if len(syllable_sequence) != _bin:
                    continue
                
                dct = {}
                
                dct["tm"] = count_transitions(syllable_sequence, K=K)
                if dct["tm"].max() < 2**16:
                    dct["tm"] = dct["tm"].astype("uint16")
                else:
                    raise RuntimeError("Could not cast tm")
                
                for _feature in dlight_features:
                    dct[_feature] = use_values[_feature].iat[_trans]
                
                for _key in meta_keys:
                    dct[_key] = use_values[_key].iat[_trans]
                dct["count"] = dct["tm"][_syllable].sum()
                dct["tm"] = dct["tm"][:truncate, :truncate]
                dct["syllable"] = _syllable 
                dct["bin"] = _bin
                dcts.append(dct)
    if len(dcts) == 0:
        return None
    else:
        return pd.DataFrame(dcts).reset_index()


def get_syllable_rates_lagged(
    #     *args,
    values: pd.DataFrame,
    dlight_features: list[str] = [
        "signal_reref_dff_norm_rising_range",
        "signal_reref_dff_norm_rising_slope",
        "signal_reref_dff_norm_rising_max",
    ],
    syllable_key: str = "syllable",
    chk_syllables: np.ndarray = np.arange(100),
    usage_bins: np.ndarray = np.arange(0, 100, 10),
    meta_keys: list[str] = [
        "timestamp",
        "duration",
        "mouse_id",
        "snippet",
        "area",
        "opsin",
        "signal_max",
        "signal_reference_corr",
        "stim_duration",
        "target_syllable",
        "session_number",
    ],
    additional_syllable_keys: list[str] = ["duration"],
    K: int = 100,
    lags: np.ndarray = np.arange(-10, 9),
) -> Optional[pd.DataFrame]:
    dcts = []
    #     values = args[-1]
    syllables = values[syllable_key]
    syllable_lst = list(syllables.unique())
    use_syllables = [_ for _ in chk_syllables if _ in syllable_lst]

    excluded_syllables = set(syllable_lst) - set(use_syllables)
    if len(excluded_syllables) > 0:
        raise UserWarning(f"{excluded_syllables} will be excluded in downstream analysis")

    # NEED TO ADD SPECIFIC STUFF IN HERE SO WE CAN DO AVERAGE SYLLABLE DURATION...
    # pad values
    # loop through each syllable, get dlight feature and then count outbound transitions in
    # the bins...

    # every transition it's own dataframe?
    for _syllable in use_syllables:

        use_trans = np.flatnonzero(syllables == _syllable)

        # okay specific back in, let's clone the global/specific designation...
        values_specific = values.copy()
        values_specific[syllables != _syllable] = np.nan

        if len(use_trans) == 0:
            continue

        #         use_trans = use_trans[use_trans < len(syllables) - np.max(usage_bins)]
        # why you wanna do me like that?
        syllable_dfs = []
        for _trans in use_trans:

            # get transitions in the right range
            for _bin in usage_bins:

                # now we can index other values by syllable occurrences left inclusive
                for _lag in lags:
                    try:
                        use_syllable = syllables.iat[_trans + _lag]
                    except IndexError:
                        continue
                    use_trans = np.flatnonzero(syllables == use_syllable)
                    use_trans = use_trans[use_trans < len(syllables) - np.max(usage_bins)]
                    future_syllable_trans = use_trans[
                        (use_trans >= (_trans + _lag)) & (use_trans < (_trans + _lag) + _bin)
                    ]
                    syllable_values = values.iloc[future_syllable_trans]
                    #                 if len(syllable_sequence) != _bin:
                    #                     RuntimeError("Indexing issue")
                    #                     continue

                    dct = {}
                    for _feature in dlight_features:
                        dct[_feature] = values[_feature].iat[_trans]

                    for _key in meta_keys:
                        dct[_key] = values[_key].iat[_trans]

                    for _key in additional_syllable_keys:
                        dct[f"{_key}"] = syllable_values[_key].mean()

                    dct["syllable"] = _syllable
                    dct["count"] = len(future_syllable_trans)
                    dct["bin"] = _bin
                    dct["trans_number"] = int(_trans)
                    dct["lag"] = _lag
                    dcts.append(dct)

    if len(dcts) == 0:
        return None
    else:
        return pd.DataFrame(dcts).reset_index()