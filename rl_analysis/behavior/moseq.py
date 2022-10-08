import numpy as np


def syll_onset(labels) -> np.ndarray:
    '''Finds the indices of syllable beginnings
    Args:
        labels (np.ndarray): array of syllable labels from one mouse
    Returns:
        an array of indices denoting the beginning of each syllable
    '''
    change = np.diff(labels) != 0
    indices = np.where(change)[0]
    indices = indices + 1
    indices = np.concatenate((np.array([0]), indices))
    return indices