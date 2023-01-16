from tqdm.auto import tqdm
from copy import deepcopy
import numpy as np



# prepare data for kernel regression
def prepare_data(
    df,
    standardize_features=False,
    dlight_key="signal_reref_dff_z",
    sz_cutoff=23 * 30 * 60,
    feature_variables=[],
    powers=None,
):

    szs = df.groupby("uuid").size()
    if sz_cutoff < 1:
        cutoff = int(np.quantile(szs, sz_cutoff))
    else:
        cutoff = sz_cutoff
    proc_df = df.groupby("uuid").filter(lambda x: len(x) > cutoff)
    feature_names = deepcopy(feature_variables)
    # proc_df = proc_df.sort_values(["uuid","timestamp"]).reset_index()

    grouper = proc_df.groupby("uuid")
    dlight_traces = np.array(
        [v.iloc[:cutoff].to_numpy() for k, v in tqdm(grouper[dlight_key])]
    )
    uuids = np.array([k for k, v in grouper[dlight_key]])

    feature_lst = []
    for _variable in tqdm(feature_variables):
        feature_lst.append([v.iloc[:cutoff].to_numpy() for k, v in grouper[_variable]])
    feature_matrices = np.stack(feature_lst, axis=-1).astype("float")

    power_features = []
    if (powers is not None) and isinstance(powers, list):
        for _var_name, _power in powers:
            idx = feature_names.index(_var_name)
            power_features.append(np.power(feature_lst[idx], _power))
            feature_names.append(f"{_var_name} ** {_power}")
        power_features = np.stack(power_features, axis=-1).astype("float")
    elif (powers is not None) and isinstance(powers, int):
        power_features = feature_matrices**powers
        for _feature in feature_names:
            feature_names.append(f"{_feature} ** {powers}")

    if powers is not None:
        feature_matrices = np.concatenate([feature_matrices, power_features], axis=-1)

    if standardize_features:
        feature_matrices -= np.mean(feature_matrices, axis=1, keepdims=True)
        feature_matrices /= np.std(feature_matrices, axis=1, keepdims=True) + 1e-6

    return dlight_traces, feature_matrices, uuids, feature_names


def early_stopping(losses, threshold=1e-4, patience=10):
    try:
        delta = -np.diff(np.array(losses)[-patience:])
        if (len(losses) >= patience) & np.all(delta < threshold):
            return True
    except IndexError:
        return False