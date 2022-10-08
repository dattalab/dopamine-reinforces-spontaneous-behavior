import os
import toml
import numpy as np
from typing import Union
from copy import deepcopy


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


def dlight_exclude_toml(
    dct: Union[dict, str],
    dff_threshold=None,
    dlight_reference_corr_threshold=None,
    exclude_stim=True,
    exclude_3s=False,
    raw_dff_threshold=1.0,
):

    if not isinstance(dct, dict) and isinstance(dct, str):
        assert os.path.exists(dct), "Path to toml file must exist"
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