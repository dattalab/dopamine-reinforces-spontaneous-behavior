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


def syll_duration(labels):
    onsets = syll_onset(labels)
    end = np.array([labels.size], dtype=np.int64)

    onsets = np.concatenate((onsets, end))
    durations = np.diff(onsets)
    return durations


def add_onset_and_duration(df, syll_key='predicted_syllable'):
    sylls = df[syll_key].to_numpy()

    durs = syll_duration(sylls)

    df['onset'] = df[syll_key].diff().fillna(1) != 0
    df.loc[df['onset'], 'dur'] = durs
    df['dur'] = df['dur'].fillna(method='ffill').astype('uint16')
    return df
