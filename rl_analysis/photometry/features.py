import numpy as np
import pandas as pd
from numba import jit
from typing import Union, Optional

# NOTE: these are features used in computing the "features" dataframe
# e.g. syllable-associated DA is computed via these functions
__all__ = [
	"nanargmax",
    "argmax",
	"idxmax",
	"peak_to_peak",
	"peak_to_peak_signed",
	"max_abs_peak",
	"rising_slope",
	"falling_slope",
	# "get_peak_edges",
	"rising_width",
	"falling_width",
	"rising_range",
	"falling_range",
	"rising_max",
	"falling_max",
	"rising_min",
	"falling_min",
]


def nanargmax(arr: pd.DataFrame, axis: int = 0) -> float:
    try:
        return arr.index[np.nanargmax(arr, axis)]  # type: ignore
    except ValueError:
        return np.nan

nanargmax.__name__ = "idxmax"


@jit(nopython=True)
def argmax(values: np.ndarray, index: np.ndarray) -> np.integer:
    return np.argmax(values)


@jit(nopython=True)
def idxmax(values: np.ndarray, index: np.ndarray) -> float:
    tmp = values[~np.isnan(values)]
    if len(tmp) == 0:
        return np.nan
    else:
        return index[np.argmax[tmp]]  # type: ignore


@jit(nopython=True)
def peak_to_peak(values: np.ndarray, index: np.ndarray) -> np.floating:
    return np.max(values) - np.min(values)


@jit(nopython=True)
def max_abs_peak(values: np.ndarray, index: np.ndarray) -> np.floating:
    return values[np.argmax(np.abs(values))]


# returns edges
@jit(nopython=True)
def get_peak_edges(
    values: np.ndarray, rising: bool = True, turn_threshold: float = 1
) -> Optional[tuple[tuple[int, int], np.ndarray, np.integer, np.integer]]:
    peak_idx = np.argmax(np.abs(values[2:-2])) + 2
    peak_sign = np.sign(values[peak_idx])

    deriv = np.zeros_like(values)
    deriv[0] = np.nan
    # np diff craps out if the array isn't contiguous so do it the old-fashioned way
    deriv[1:] = values[1:] - values[:-1]

    # flip the signs for falling
    if rising is False:
        peak_sign *= -1
        deriv *= -1

    if peak_sign < 0:
        itr = np.arange(peak_idx + 1, len(values))
    elif peak_sign > 0:
        itr = np.arange(peak_idx, 0, -1)
    else:
        return None

    turn_count = 0
    idx = 0

    for i in itr:
        if deriv[i] <= 0:
            idx += 1
            turn_count += 1
        else:
            idx += 1
            turn_count = 0
        if turn_count >= turn_threshold:
            idx -= turn_count
            break

    if peak_sign < 0:
        left_edge = peak_idx
        right_edge = peak_idx + idx
    else:
        left_edge = peak_idx - idx
        right_edge = peak_idx

    left_edge = np.maximum(left_edge, 0)
    right_edge = np.maximum(np.minimum(right_edge, len(values)), left_edge + 1)

    # return original sign
    if rising is False:
        peak_sign *= -1
        deriv *= -1

    return (left_edge, right_edge), deriv, peak_idx, peak_sign


@jit(nopython=True)
def rising_slope(values: np.ndarray, index: np.ndarray) -> float:
    result = get_peak_edges(values, rising=True)
    if result is not None:
        edges, deriv, peak_idx, peak_sign = result
        if (edges[1] - edges[0]) > 1:
            slope = np.nanmean(deriv[(edges[0] + 1) : edges[1]])
        else:
            slope = np.nan
    else:
        slope = np.nan
    return slope


@jit(nopython=True)
def falling_slope(values: np.ndarray, index: np.ndarray) -> float:
    result = get_peak_edges(values, rising=False)
    if result is not None:
        edges, deriv, peak_idx, peak_sign = result
        if (edges[1] - edges[0]) > 1:
            slope = np.nanmean(deriv[(edges[0] + 1) : edges[1]])
        else:
            slope = np.nan
    else:
        slope = np.nan
    return slope


@jit(nopython=True)
def rising_width(values: np.ndarray, index: np.ndarray) -> float:
    result = get_peak_edges(values, rising=True)
    if result is not None:
        edges, deriv, peak_idx, peak_sign = result
        if (edges[1] - edges[0]) > 1:
            width = edges[1] - edges[0]
        else:
            width = np.nan
    else:
        width = np.nan
    return width


@jit(nopython=True)
def falling_width(values: np.ndarray, index: np.ndarray) -> float:
    result = get_peak_edges(values, rising=False)
    if result is not None:
        edges, deriv, peak_idx, peak_sign = result
        if (edges[1] - edges[0]) > 1:
            width = edges[1] - edges[0]
        else:
            width = np.nan
    else:
        width = np.nan
    return width


@jit(nopython=True)
def rising_range(values: np.ndarray, index: np.ndarray) -> float:
    result = get_peak_edges(values, rising=True)
    if result is not None:
        edges, deriv, peak_idx, peak_sign = result
        if (edges[1] - edges[0]) > 1:
            _range = values[edges[1]] - values[edges[0]]
        else:
            _range = np.nan
    else:
        _range = np.nan
    return _range


@jit(nopython=True)
def falling_range(values: np.ndarray, index: np.ndarray) -> float:
    result = get_peak_edges(values, rising=False)
    if result is not None:
        edges, deriv, peak_idx, peak_sign = result
        if (edges[1] - edges[0]) > 1:
            _range = values[edges[1]] - values[edges[0]]
        else:
            _range = np.nan
    else:
        _range = np.nan
    return _range


@jit(nopython=True)
def rising_max(values: np.ndarray, index: np.ndarray) -> float:
    result = get_peak_edges(values, rising=True)
    if result is not None:
        edges, deriv, peak_idx, peak_sign = result
        if (edges[1] - edges[0]) > 1:
            _range = values[edges[1]]
        else:
            _range = np.nan
    else:
        _range = np.nan
    return _range


@jit(nopython=True)
def rising_min(values: np.ndarray, index: np.ndarray) -> float:
    result = get_peak_edges(values, rising=True)
    if result is not None:
        edges, deriv, peak_idx, peak_sign = result
        if (edges[1] - edges[0]) > 1:
            _range = values[edges[0]]
        else:
            _range = np.nan
    else:
        _range = np.nan
    return _range


@jit(nopython=True)
def falling_max(values: np.ndarray, index: np.ndarray) -> float:
    result = get_peak_edges(values, rising=False)
    if result is not None:
        edges, deriv, peak_idx, peak_sign = result
        if (edges[1] - edges[0]) > 1:
            _range = values[edges[0]]
        else:
            _range = np.nan
    else:
        _range = np.nan
    return _range


@jit(nopython=True)
def falling_min(values: np.ndarray, index: np.ndarray) -> float:
    result = get_peak_edges(values, rising=False)
    if result is not None:
        edges, deriv, peak_idx, peak_sign = result
        if (edges[1] - edges[0]) > 1:
            _range = values[edges[1]]
        else:
            _range = np.nan
    else:
        _range = np.nan
    return _range


@jit(nopython=True)
def peak_to_peak_signed(values: np.ndarray, index: np.ndarray) -> float:
    min_value = np.min(values)
    max_value = np.max(values)

    if min_value < 0:
        return_val = max_value + min_value
    else:
        return_val = max_value - min_value

    return return_val
