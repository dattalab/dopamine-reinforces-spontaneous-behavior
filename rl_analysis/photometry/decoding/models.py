import numpy as np
import pandas as pd


def exp_smoother(
    sequence,
    da,
    nsyllables=10,
    baseline_temperature=0.1,
    entropy_tau=10,
    usage_tau=200.0,
    entropy_scaling=0.3,
    usage_scaling=0.3,
    eps=1e-10,
    log_variables=False,
):

    from rl_analysis.photometry.decoding.util import stable_softmax
    nsyllables = np.round(nsyllables).astype("int")
    cur_usage_weights = np.zeros((da.shape[1],), dtype="float")
    cur_usage_tracker = np.zeros((da.shape[1],), dtype="float")
    cur_entropy_weights = 0.0

    entropy_alpha = 1.0 / entropy_tau
    usage_alpha = 1.0 / usage_tau
    cur_state = sequence[0]
    nancount = 0

    return_dct = {}

    bin_matrix = pd.get_dummies(
        pd.Series(sequence)
        .astype("category")
        .cat.set_categories(np.arange(da.shape[1]))
    )
    oracle_probas = bin_matrix.rolling(
        np.round(usage_tau).astype("int"), 1, True
    ).mean()

    bin_matrix = bin_matrix.values
    oracle_probas = oracle_probas.values
    # bin_matrix = (da > 0).astype("float")

    if log_variables:
        return_dct["probas_history"] = []
        return_dct["usage_tracker_history"] = []

    return_dct["ll"] = 0.0
    return_dct["ll_oracle"] = 0.0

    # use prior DA
    counter = 0
    for _syllable, _da, _bin_vec, _oracle_vec in zip(
        sequence[1:], da[1:], bin_matrix[1:], oracle_probas[1:]
    ):

        # note took usage scaling out...
        cur_usage_tracker = (1 - usage_alpha) * cur_usage_tracker + usage_alpha * (
            _bin_vec
        )
        cur_usage_tracker_norm = (cur_usage_tracker + eps) / (
            cur_usage_tracker + eps
        ).sum()

        # can happen occassionally, safely continue (should warn if it's excessive...
        if np.isnan(_da).any():
            nancount += 1
            cur_state = _syllable
            # if nancount > 300:
            #     warnings.warn("Encountered more nans than expected...")
            continue

        cur_usage_weights = (1 - usage_alpha) * cur_usage_weights + usage_alpha * (
            _da * usage_scaling
        )

        # global version, more in line with fig 2.
        cur_entropy_weights = (
            1 - entropy_alpha
        ) * cur_entropy_weights + entropy_alpha * (_da[_syllable] * entropy_scaling)

        # if we're truncating just keep rolling
        if (cur_state >= nsyllables) | (_syllable >= nsyllables):
            cur_state = _syllable
            continue

        probas = cur_usage_weights[:nsyllables]
        probas = stable_softmax(
            probas + eps, np.clip(baseline_temperature + cur_entropy_weights, 1e-2, 100)
        )
        oracle_probas = _oracle_vec[:nsyllables]
        oracle_probas = stable_softmax(
            oracle_probas + eps,
            np.clip(baseline_temperature + cur_entropy_weights, 1e-2, 100),
        )

        return_dct["ll"] += np.log(probas[_syllable] + eps)
        return_dct["ll_oracle"] += np.log(oracle_probas[_syllable] + eps)
        counter += 1
        cur_state = _syllable

        if log_variables:
            return_dct["probas_history"].append(probas)
            return_dct["usage_tracker_history"].append(cur_usage_tracker_norm)

    if log_variables:
        return_dct["probas_history"] = np.array(return_dct["probas_history"]).astype(
            "float32"
        )
        return_dct["usage_tracker_history"] = np.array(
            return_dct["usage_tracker_history"]
        ).astype("float32")

    return_dct["ll"] /= counter
    return_dct["ll_oracle"] /= counter
    return return_dct