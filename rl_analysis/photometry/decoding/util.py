from numba import jit
from rl_analysis.photometry.decoding.models import exp_smoother
from copy import deepcopy
import numpy as np
import pandas as pd


@jit
def stable_softmax(p, temperature=100.0):
    use_p = p.copy()
    use_p -= np.nanmax(use_p)
    use_p = np.exp(use_p / temperature)
    return use_p / np.nansum(use_p)


def run_simulation(seqs, features, uniform_init=True, **kwargs):
    results = []
    for _seq, _feature in zip(seqs, features):
        conv_feature = (
            pd.get_dummies(
                pd.Series(_seq).astype("category").cat.set_categories(np.arange(100))
            )
            * _feature[:, None]
        )
        conv_feature = conv_feature.values
        _result = exp_smoother(_seq, conv_feature, **kwargs)
        results.append(_result)
    return results


def run_simulation_stim(
    seqs, features, targets, feedback_status, control=False, stim_offset=0, **kwargs
):
    results = []
    use_kwargs = deepcopy(kwargs)
    truncate = use_kwargs.pop("nsyllables", 100)

    for _seq, _feature, _target, _status in zip(
        seqs, features, targets, feedback_status
    ):
        use_seq = _seq.copy()
        if _target >= truncate:
            # remap if the target is beyond the threshold
            # this leaves an empty slot at the target value
            use_seq[_seq == _target] = truncate + 1

            # shift everything between the new value and the target value by one
            use_seq[(_seq >= (truncate + 1)) & (_seq < _target)] += 1
            use_target = truncate + 1

            # don't forget to increment the truncation
            use_truncate = truncate + 1
        else:
            use_truncate = truncate

        use_feature = _feature.copy()

        # get catch value for target, add offset
        catch_values = use_feature[(_seq == _target) & (_status == 2)]

        if len(catch_values) == 0:
            continue

        # use_feature[(_seq == _target) & (_status != 1)] = catch_values.mean()
        if control is False:
            use_feature[(_status == 1)] = np.nanmean(catch_values) + stim_offset
            # use_feature[(_seq == _target) & (_status == 1)] = np.nanmean(catch_values) + stim_offset
        # print(np.isnan(use_feature).sum())
        # else:
        # use_seq = _seq
        # use_feature = _feature

        # re-zscore to account for mean shift
        conv_feature = pd.get_dummies(
            pd.Series(use_seq).astype("category").cat.set_categories(np.arange(100))
        ) * (use_feature[:, None])
        conv_feature = conv_feature.values
        result = exp_smoother(
            _seq, conv_feature, nsyllables=use_truncate, **use_kwargs
        )
        results.append(result)
    return results