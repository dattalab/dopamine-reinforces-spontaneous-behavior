import os
import toml
import numpy as np
import pandas as pd
from typing import Union, Optional
from copy import deepcopy
from tqdm.auto import tqdm


dlight_exclude_uuids = [
    "5bfb8680-f48d-411c-94ae-3c7cbb3652c7",
    "1ad1d09a-bb6e-4158-8bfd-cc75e2d0747b",
    "eab24978-761d-402f-9e53-f2b6e2b56f68",
    "153719c8-6fae-486d-8cbb-bc73bb883ad6",
]


NONSYLL_COLS = [
    "velocity_3d_mm",
    "velocity_2d_mm",
    "height_ave_mm",
    "width_mm",
    "length_mm",
    "angle",
    "centroid_x_mm",
    "centroid_y_mm",
    "camera_timestamp",
    "realtime-package",
    "trigger_syllable_scalar_comparison",
    "trigger_syllable_scalar_baseline_duration",
    "angle_unwrapped",
    "area",
    "fs",
    "StartTime",
    "SubjectName",
    "SessionName",
    "filter_params",
    "labels",
    "acceleration_3d_mm",
    "acceleration_2d_mm",
    "jerk_3d_mm",
    "jerk_2d_mm",
    "velocity_angle",
    "velocity_height",
    "velocity_2d_mm_z_peak_score",
    "movement_initiations",
    "ir_indices",
    *tuple(f"pc{x:02d}" for x in range(10))
]


def dlight_exclude(
    df: pd.DataFrame,
    dff_threshold: Optional[float] = None,
    dlight_reference_corr_threshold: Optional[float] = None,
    exclude_target: bool = True,
    exclude_stim: bool = True,
    exclude_3s: bool = False,
    syllable_key: str = "predicted_syllable (offline)",
    raw_dff_threshold: float = 1.0,
) -> pd.DataFrame:

    if exclude_3s:
        df = df.loc[df["stim_duration"] != 3].copy()

    if dff_threshold is not None:
        df = df.loc[df["signal_max"] > dff_threshold].copy()

    if dlight_reference_corr_threshold is not None:
        df = df.loc[df["signal_reference_corr"] < dlight_reference_corr_threshold].copy()

    if exclude_target:
        df = df.loc[
            ~(
                (df["opsin"] == "chrimson")
                & (df[syllable_key] == df["target_syllable"])
                & (df["session_number"].isin([1, 2, 3, 4]))
            )
        ].copy()

    if exclude_stim:
        df = df.loc[~(df["session_number"].isin([1, 2]))].copy()

    try:
        if raw_dff_threshold is not None:  # exclude unnatural dff values, i.e. >>> 10%
            uuids = df.loc[df["signal_dff"] > raw_dff_threshold, "uuid"].unique()
            df = df.loc[~df["uuid"].isin(uuids)].copy()
    except KeyError:
        if "uuid" in df.index.names:
            df.loc[~(df.index.get_level_values("uuid").isin(dlight_exclude_uuids))].copy()
        elif "uuid" in df.columns:
            df.loc[~df["uuid"].isin(dlight_exclude_uuids)].copy()
        else:
            raise RuntimeError("Could not find uuid in dataframe")

    return df

# WIN'S ORIGINAL VERSION
# def dlight_exclude_toml(
#     dct: Union[dict, str],
#     dff_threshold=None,
#     dlight_reference_corr_threshold=None,
#     exclude_stim=True,
#     exclude_3s=False,
#     raw_dff_threshold=1.0,
# ):

#     if not isinstance(dct, dict) and isinstance(dct, str):
#         assert os.path.exists(dct), "Path to toml file must exist"
#         with open(dct, "r") as f:
#             use_dct = toml.load(f)
#     else:
#         use_dct = deepcopy(dct)

#     if exclude_3s:
#         use_dct = {k: v for k, v in use_dct.items() if v.get("stim_duration", 0) != 3}

#     if dff_threshold is not None:
#         use_dct = {k: v for k, v in use_dct.items() if v.get("signal_max", -np.inf) > dff_threshold}

#     if dlight_reference_corr_threshold is not None:
#         use_dct = {
#             k: v
#             for k, v in use_dct.items()
#             if v.get("signal_reference_corr", np.inf) < dlight_reference_corr_threshold
#         }

#     if exclude_stim:
#         use_dct = {k: v for k, v in use_dct.items() if v.get("session_number", 0) not in [1, 2]}

#     if raw_dff_threshold is not None:
#         use_dct = {
#             k: v for k, v in use_dct.items() if v.get("signal_dff_max", np.inf) < raw_dff_threshold
#         }

#     return use_dct


# Jeff's version with more complete type annotations
def dlight_exclude_toml(
    dct: Union[dict, str],
    dff_threshold: Optional[float] = None,
    dlight_reference_corr_threshold: Optional[float] = None,
    exclude_stim: bool = True,
    exclude_3s: bool = False,
    raw_dff_threshold: float = 1.0,
) -> dict:
    from copy import deepcopy
    import toml

    if isinstance(dct, str):
        with open(dct, "r") as f:
            use_dct = toml.load(f)
    else:
        use_dct = deepcopy(dct)

    if exclude_3s:
        use_dct = {k: v for k, v in use_dct.items() if v.get("stim_duration", 0) != 3}

    if dff_threshold is not None:
        use_dct = {k: v for k, v in use_dct.items() if v.get("signal_max", -np.inf) > dff_threshold}

    if dlight_reference_corr_threshold is not None:
        use_dct = {
            k: v
            for k, v in use_dct.items()
            if v.get("signal_reference_corr", np.inf) < dlight_reference_corr_threshold
        }

    if exclude_stim:
        use_dct = {k: v for k, v in use_dct.items() if v.get("session_number", 0) not in [1, 2]}

    if raw_dff_threshold is not None:
        use_dct = {
            k: v for k, v in use_dct.items() if v.get("signal_dff_max", np.inf) < raw_dff_threshold
        }

    return use_dct


def load_dlight_features(path, df_filter=None, win_left=0, win_right=0.3):
    df = pd.read_parquet(path)
            
    if df_filter is not None:
        df = df_filter(df)

    df = df.query('@win_left == win_left & win_right == @win_right').copy()

    dtype = dict(
        is_feedback_any='uint8',
        is_catch_any='uint8',
    )

    return df.astype(dtype)


# save memory by converting integers to 8bit
default_to_int8 = [
    "target_syllable",
    "cohort",
    "session_number",
    "syllable_group",
    "feedback_status",
]

default_to_cat = [
    "mouse_id",
    "uuid",
    "genotype",
]


default_load_keys = [
    "mouse_id",
    "predicted_syllable",
    "predicted_syllable (offline)",
    "date",
    "uuid",
    "genotype",
    "target_syllable",
    "syllable_group",
    "timestamp",
    "realtime-package",
    "session_number",
    "SessionName",
    "session_repeat",
    "feedback_status",
]

def _load_feedback_parquet(
    fname,
    cohort=None,
    load_keys=default_load_keys,
    to_int8=default_to_int8,
    to_cat=default_to_cat,
    apply_func=None,
    extra_keys=[],
    stim_sessions=None,
    **kwargs,
):
    import warnings

    use_keys = sorted(list(set(load_keys + extra_keys)))
    feedback_df = []
    # _tmp = pd.read_parquet(_df, columns=load_keys + extra_keys)
    if not os.path.exists(fname):
        warnings.warn(f"{fname} does not exist", UserWarning)
        return None
    try:
        data = pd.read_parquet(fname, columns=use_keys, **kwargs)
    except Exception as e:
        print(e)
        try:
            if "predicted_syllable (offline)" in use_keys:
                use_keys.remove("predicted_syllable (offline)")
            data = pd.read_parquet(fname, columns=use_keys, **kwargs)
        except Exception as e:
            print(e)
            return None

    if callable(apply_func):
        data = apply_func(data)

    data["cohort"] = cohort
    # _tmp["cohort"] = i

    int8s = data.columns.intersection(to_int8)
    data[int8s] = data[int8s].astype("int8")

    cats = data.columns.intersection(to_cat)
    data[cats] = data[cats].astype("category")

    # downcast expensive cols to float32
    like_cols = data.filter(regex="likes[0-9]+").columns

    if len(like_cols) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data[like_cols] = np.log10(data[like_cols]).astype("float32")

    # clamp feedback status in non stim sessions (if provided)
    if stim_sessions is not None:
        data.loc[~data["session_number"].isin(stim_sessions), "feedback_status"] = -1
    else:
        data["feedback_status"] = -1

    return data


def load_feedback_parquet(
    fnames,
    parallel_load=False,
    n_jobs=None,
    backend="loky",
    load_keys=default_load_keys,
    to_int8=default_to_int8,
    to_cat=default_to_cat,
    apply_func=None,
    extra_keys=[],
    **kwargs,
):
    if isinstance(fnames, str):
        fnames = [fnames]

    if not parallel_load:
        dfs = []
        for i, _fname in tqdm(enumerate(fnames)):
            _df = _load_feedback_parquet(
                _fname,
                load_keys=load_keys,
                cohort=i,
                to_int8=to_int8,
                to_cat=to_cat,
                apply_func=apply_func,
                extra_keys=extra_keys,
                **kwargs,
            )
            dfs.append(_df)
    else:
        from functools import partial
        from joblib import Parallel, delayed

        delays = []
        for i, _fname in enumerate(fnames):
            _func = partial(
                _load_feedback_parquet,
                cohort=i,
                load_keys=load_keys,
                to_int8=to_int8,
                to_cat=to_cat,
                apply_func=apply_func,
                extra_keys=extra_keys,
                **kwargs,
            )
            delays.append(delayed(_func)(_fname))
        dfs = Parallel(
            verbose=10, n_jobs=len(delays) if n_jobs is None else n_jobs, backend=backend
        )(delays)

    if dfs is not None:
        feedback_df = pd.concat(dfs).reset_index(drop=True)
    else:
        feedback_df = None
    return feedback_df


exclude_mice = {
    "bad coordinates": [
        "186",
        "189",
        "194",
        "1737",
        "1736",
        "427",
        "15836",
        "15839",
        "15848",
    ],
    "jrcamp1a unilateral": ["1734", "1738"],
    "id confusion": ["1562"],
}

exclude_uuids = [
    "34ef577c-8f80-43f3-9c40-7dccb5b96709",
    "ac13dbbb-fb6c-4704-8827-f384a943113d",
    "0e3e3649-af22-4a1f-8144-f6db841c08b9",
    "54f839b5-439f-4a45-9d8e-7bac97f41447",
    "95cbebb5-308e-4fe0-bd6b-f301269d1741",
    "5cec5800-8218-4fcb-a00e-f2ec0482922d",
    "655f2fdf-ff39-4817-9160-2b8c5d1446f8",
    "7fa3ae69-5541-4508-b69a-4727db61dea5",
    "d5e9b32e-5de4-40b2-9c9b-f8995a0dc906",
    "6df691fe-a538-409f-b6ee-cd6fdbdfb714",
    "1bd2bef9-2b90-4fc9-a7a7-05430b1ba362",
    "fc9be484-cb88-4701-af0f-dc61ee7391bc",
    "f10779aa-986b-48a7-b0ed-3c1695c344a9",
    "95a70dfe-ee7f-43a2-84f0-b08c6d5c4fec",
    "4e47db6c-140a-40d7-9d09-373d97356f4b",
    "2f5ac601-9ae6-45eb-b129-4b7eef15e175",
    "c0552c5a-db7f-4baa-b6ab-7f9a74459ef5",
    "a32c2020-d6d5-420c-acc8-ab10c5af0a83", # bad background
    "6b1b613d-ae86-452c-a85e-c74cfbb5ce2f", # bad background
    "39115867-72c1-4cd9-93e0-a53ec2ff5cd6", # bad background
    "17765325-b7cf-4059-8571-ed30d59491f2", # bad background
    "fae564aa-84df-47c0-a6fa-4cb691029d2e", # bad background
    "34c5702e-10d3-43d8-8871-32b79d2a93a1", # bad background
    "3c08a9f3-f86d-41a7-ad5f-2a92bce0518f", # bad background
]


sex_map = {
        "1520": "female",
        "1515": "female",
        "1521": "female",
        "1519": "female",
        "1524": "male",
        "1525": "male",
        "1529": "female",
        "1527": "male",
        "1528": "male",
        "1546": "male",
        "1544": "male",
        "1561": "female",
        "1562": "female",
        "1778": "male",
        "197": "male",
        "200": "male",
        "208": "female",
        "209": "female",
        "184": "female",
        "185": "female",
        "186": "female",
        "189": "female",
        "194": "male",
        "1738": "male",
        "1734": "male",
        "1737": "male",
        "1736": "male",
        "15809": "female",
        "15814": "female",
        "15825": "female",
        "15817": "female",
        "15827": "female",
        "15816": "female",
        "15847": "female",
        "15848": "female",
        "15836": "female",
        "15839": "female",
        "15822": "male",
        "15823": "male",
        "211": "male",
        "12": "male",
        "10": "male",
        "8": "male",
        "357": "female",
        "358": "female",
        "355": "female",
        "356": "female",
        "361": "male",
        "363": "male",
        "414": "female",
        "413": "female",
        "416": "female",
        "417": "female",
        "368": "male",
        "364": "male",
        "408": "male",
        "410": "male",
        "429": "female",
        "427": "female",
        "428": "female",
        "810": "male",
        "136": "female",
        "137": "female",
        "133": "female",
        "806": "male",
        "807": "female",
        "768": "male",
        "770": "male",
        "769": "male",
        "767": "male",
        "779": "female",
        "778": "female",
        "776": "female",
        "127": "male",
        "126": "male",
        "780": "female",
        "784": "female",
        "781": "female",
        "782": "female",
        "138": "female",
        "2273": "female",
        "2275": "female",
        "2274": "female",
        "2270": "male",
        "2269": "male",
        "2271": "male",
        "2272": "female",
        "240": "female",
        "239": "female",
        "242": "female",
        "241": "female",
        "snc-acr-1": "female",
        "vta-acr-1": "female",
        "vta-acr-2": "female",
        "snc-acr-2": "female",
        "snc-acr-3": "female",
        "vta-acr-3": "female",
        "snc-acr-4": "male",
        "snc-acr-5": "male",
        "vta-acr-4": "male",
        "vta-acr-5": "male",
        "snc-acr-6": "male",
        "snc-acr-7": "male",
        "vta-acr-6": "male",
        "vta-acr-7": "male",
        "3172": "female",
        "3169": "female",
        "3173": "female",
        "2865": "female",
        "2860": "female",
        "2859": "female",
        "2863": "female",
        "2864": "female",
        "2862": "female",
        "3158": "female",
        "3155": "female",
        "3157": "female",
        "3214": "male",
        "3216": "male",
        "3439": "female",
        "3440": "female",
        "3441": "female",
        "3442": "female",
        "3474": "male",
        "3472": "male",
        "3473": "male",
        "3475": "male",
        "vta-nacc-ctrl-6": "male",
        "snc-dls-ctrl-6": "female",
        "dlight-chrimson-1": "female",
        "dlight-chrimson-2": "female",
        "dlight-chrimson-3": "male",
        "dlight-chrimson-4": "male",
        "dlight-chrimson-5": "female",
        "dlight-chrimson-6": "male",
        "dlight-chrimson-7": "male",
        "dlight-chrimson-8": "male",
        "dlight-chrimson-9": "male",
        "dls-ai32jr-1": "male",
        "dls-ai32jr-2": "male",
        "dls-ai32jr-3": "male",
        "dls-ai32jr-4": "female",
        "dls-ai32jr-5": "female",
        "dms-ai32-1": "male",
        "dms-ai32-2": "male",
        "dms-ai32-3": "female",
        "dms-ai32-4": "female",
        "dms-ai32-5": "female",
        "dms-ai32-6": "male",
        "dms-ai32-7": "male",
        "dms-ai32-8": "male",
        "dms-ai32-9": "male",
        "dms-ai32-10": "female",
        "dms-ai32-11": "female",
        "snc-dls-ctrl-7": "male",
        "vta-nacc-ai32-18": "male",
        "vta-nacc-ai32-19": "male",
        "vta-nacc-ai32-20": "female",
        "dls-dlight-1": "male",
        "dls-dlight-2": "male",
        "dls-dlight-3": "male",
        "dls-dlight-4": "male",
        "dls-dlight-5": "male",
        "dls-dlight-6": "male",
        "dls-dlight-7": "male",
        "dls-dlight-8": "male",
        "dls-dlight-9": "male",
        "dls-dlight-10": "male",
        "dls-dlight-11": "male",
        "dls-dlight-12": "male",
        "dls-dlight-13": "male",
        "dms-dlight-1": "male",
        "dms-dlight-2": "male",
        "dms-dlight-3": "male",
        "dms-dlight-4": "male",
        "dms-dlight-5": "female",
        "dms-dlight-6": "female",
        "dms-dlight-7": "male",
        "dms-dlight-8": "male",
        "dms-dlight-9": "male",
        "dms-dlight-10": "male",
        "dms-dlight-11": "male",
        "dms-dlight-12": "male",
        "dms-dlight-13": "male",
        "dms-dlight-14": "male",
    }

def assign_feedback_metadata(
    df,
    exclude_mice=exclude_mice,
):

    df["genotype"] = df["genotype"].fillna("NULL")
    df["area"] = "NULL"
    df["area (pooled)"] = "NULL"
    
    # metadata fix for 1527
    df.loc[(df["mouse_id"] == "1527").values, "genotype"] = "ctrl"
    
    map1 = {
        "snc-dls-ai32": "snc (axon)",
        "snc-dms-ai32": "snc (axon, dms)",
        "snc-ai32": "snc (cell)",
        "vta-nacc-ai32": "vta (axon)",
        "vta-ai32": "vta (cell)",
        "vta-nacc-ai32-jrcamp1a": "vta (axon, jrcamp1a)",
        "snc-dls-ai32-jrcamp1a": "snc (axon)",  # JM changed this 5/21/2022 since we're including bilateral jrcamp1a animals
        "dls-chrimson-dlight": "snc (axon)",
        "ctrl": "ctrl",
        "vta-nacc-ctrl": "ctrl",
        "snc-dls-ctrl": "ctrl",
        "vta-ctrl": "ctrl",
        "snc-ctrl": "ctrl",
        "snc-eyfp": "ctrl",
        "snc-ai32-nacc-lesion": "snc (cell, nacc lesion)",
        "snc-ai32-dls-lesion": "snc (cell, dls lesion)",
        "vta-ai32-dls-lesion": "vta (cell, dls lesion)",
        "vta-ai32-nacc-lesion": "vta (cell, nacc lesion)",
        "vta-nacc": "vta (axon, virus)",
        "snc-dls": "snc (axon, virus)",
        "snc-acr2": "snc (cell, acr2)",
        "vta-acr2": "vta (cell, acr2)",
        "snc-chr2": "snc (cell, chr2)",
        "vta-chr2": "vta (cell, chr2)",
        "snc-dls-halo": "snc (axon)",
        "snc-dls-eyfp": "ctrl",
    }

    opsin_map = {
        "snc-dls-ai32": "chr2",
        "snc-dms-ai32": "chr2",
        "snc-ai32": "chr2",
        "vta-nacc-ai32": "chr2",
        "vta-ai32": "chr2",
        "vta-nacc-ai32-jrcamp1a": "chr2",
        "snc-dls-ai32-jrcamp1a": "chr2",
        "dls-chrimson-dlight": "chrimson",
        "ctrl": "ctrl",
        "vta-nacc-ctrl": "ctrl",
        "snc-dls-ctrl": "ctrl",
        "snc-eyfp": "ctrl",
        "vta-ctrl": "ctrl",
        "snc-ctrl": "ctrl",
        "snc-ai32-nacc-lesion": "chr2",
        "snc-ai32-dls-lesion": "chr2",
        "vta-ai32-dls-lesion": "chr2",
        "vta-ai32-nacc-lesion": "chr2",
        "vta-nacc": "chr2",
        "snc-dls": "chr2",
        "snc-acr2": "acr2",
        "vta-acr2": "acr2",
        "snc-chr2": "chr2 (viral)",
        "vta-chr2": "chr2 (viral)",
        "snc-dls-halo": "halo",
        "snc-dls-eyfp": "ctrl",
    }


    df["area"] = df["genotype"].map(map1)
    df["opsin"] = df["genotype"].map(opsin_map)
    df["sex"] = df["mouse_id"].map(sex_map)
    df["exclude"] = False

    # per-mouse exclusions
    for k, v in exclude_mice.items():
        idx = df["mouse_id"].isin(v)
        df.loc[idx, "exclude"] = True
        df.loc[idx, "exclude_reason"] = k
        df.loc[idx, "area"] = "NULL"

    # per-session exclusions
    df.loc[df["uuid"].isin(exclude_uuids), "exclude"] = True

    return df


def get_closed_loop_parquet_columns(fname, pcs=False, likes=False):
    # get all column names, potentially filter out pcs or likelihoods
    # since these columns eat up a fair amount of memory
    import pyarrow.parquet

    try:
        _tmp = pyarrow.parquet.ParquetFile(fname)
    except OSError:
        _tmp = pyarrow.parquet.ParquetDataset(fname, use_legacy_dataset=False)

    schema = _tmp.schema
    if schema is not None:
        use_names = [_.name for _ in schema]
        col_names = [_ for _ in use_names if not _.startswith("__")]
        if not pcs:
            col_names = [_ for _ in col_names if not _.startswith("pc")]
        if not likes:
            col_names = [_ for _ in col_names if not _.startswith("like")]
        return col_names
    else:
        return None

