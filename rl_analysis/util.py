import numpy as np

def zscore(arr):
    '''This function does not implement dimension-checking, so make
    sure the dimensionality of your array is 1-D for numpy arrays.
    Handles pandas Series and DataFrames fine.
    '''
    if isinstance(arr, np.ndarray):
        return (arr - np.nanmean(arr)) / np.nanstd(arr)
    # otherwise, assume pandas datastruct
    return (arr - arr.mean()) / arr.std()
