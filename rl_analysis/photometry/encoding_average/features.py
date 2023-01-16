import pandas as pd
import numpy as np
from rl_analysis.util import count_transitions
from typing import Optional


def get_lagged_features(
    values: pd.DataFrame,
    dlight_features: list[str] = [
        "signal_reref_dff_norm_rising_range",
        "signal_reref_dff_norm_rising_slope",
        "signal_reref_dff_norm_rising_max",
    ],
    syllable_key: str ="syllable",
    chk_syllables: np.ndarray = np.arange(100),
    usage_bins: np.ndarray = np.arange(0, 100, 10),
    meta_keys: list[str] = [
        "timestamp",
        "mouse_id",
        "snippet",
        "session_number",
        "duration",
        "prev_duration",
        "signal_reference_corr",
        "signal_max",
        "time_bin",
        "date",
        "opsin",
        "area",
        "target_syllable",
#         "uuid",
    ],
    additional_scalar_keys: list[str] = ["duration"],
    additional_syllable_keys: list[str] = ["duration"],
    K: int = 100,
	truncate: int = 37,
    tm_dtype: str = "uint16",
) -> Optional[pd.DataFrame]:
    dcts = []
    syllables = values[syllable_key]
    syllable_lst = list(syllables.unique())
    use_syllables = [_ for _ in chk_syllables if _ in syllable_lst]
    for _syllable in use_syllables:
        use_trans = np.flatnonzero(syllables == _syllable)

        # okay specific back in, let's clone the global/specific designation...
        values_specific = values.copy()
        values_specific[syllables != _syllable] = np.nan

        if len(use_trans) == 0:
            continue

        use_trans = use_trans[use_trans < len(syllables) - np.max(usage_bins)]
        for _trans in use_trans:
            # get transitions in the right range
            for _bin in usage_bins:

                # all transitions to the right of the current transition and less than _bin left inclusive
                syllable_sequence = syllables.iloc[_trans : _trans + _bin].values
                binned_values = values.iloc[_trans : _trans + _bin]
                
                # exclude the current syllable for measures of autocorrelation.
                binned_autocorr_values = values.iloc[_trans + 1: _trans + _bin + 1]
                
                # now we can index other values by syllable occurrences left inclusive
                future_syllable_trans = use_trans[
                    (use_trans >= _trans) & (use_trans < _trans + _bin)
                ]
                syllable_values = values.iloc[future_syllable_trans]

                if len(syllable_sequence) != _bin:
                    RuntimeError("Indexing issue")

                dct = {}
                dct["tm"] = count_transitions(syllable_sequence, K=K).astype(tm_dtype)

                if dct["tm"].max() < 2**16:
                    dct["tm"] = dct["tm"].astype("uint16")
                else:
                    raise RuntimeError("Could not cast tm")
                
                for _feature in dlight_features:
                    dct[_feature] = values[_feature].iat[_trans]
                    dct[f"{_feature}_global_bin"] = binned_autocorr_values[_feature].mean()

                for _key in meta_keys:
                    dct[_key] = values[_key].iat[_trans]

                for _scalar in additional_scalar_keys:
                    dct[f"{_scalar}_global_bin"] = binned_values[_scalar].mean()

                for _key in additional_syllable_keys:
                    dct[f"{_key}_specific_bin"] = syllable_values[_key].mean()

                dct["syllable"] = _syllable
                dct["count"] = len(future_syllable_trans)
                dct["bin"] = _bin
                dct["tm"] = dct["tm"][:truncate, :truncate]
                dct["trans_number"] = int(_trans)
                dcts.append(dct)
                
    if len(dcts) == 0:
        return None
    else:
        return pd.DataFrame(dcts).reset_index()