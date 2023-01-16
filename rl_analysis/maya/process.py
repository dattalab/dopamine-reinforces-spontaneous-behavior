import fnmatch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
#from rl_analysis.util import star
from toolz.itertoolz import concat
from itertools import repeat, count
#from rl_analysis.util import syll_onset, syll_duration

# mice with bad fiber placement
EXCLUDE_MICE = (
    '186',
    '189',
    '194',
    '1737',
    '1736',
    '427',
    '15836',
    '15839',
    '15848',
)

TARGETS = (17, 20, 27, 30, 59, 76)

SAVE_DIR = '/n/groups/datta/win/reinforcement-data/processed-dfs'

def syll_onset(labels) -> np.ndarray:
    """Finds the indices of syllable beginnings
    Args:
        labels (np.ndarray): array of syllable labels from one mouse
    Returns:
        an array of indices denoting the beginning of each syllable
    """
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



# from Jeff's mouse-rt-classifier repository
def rle(array):
    diffs_raw = np.abs(np.diff(array)) > 0
    diffs = np.ones((len(array),), dtype="bool")
    diffs[1:] = diffs_raw
    array_rle = array[diffs]
    return pd.Series(array_rle).astype("category")


def rle_from_df(df, key='predicted_syllable'):
    assert 'onset' in df
    return df.loc[df['onset'], key].astype('category')


def zscore(df):
    return (df - df.mean()) / (df.std(ddof=0))


def remove_recording_start(df, remove_minutes=5):
    return df[df['timestamp'] > (remove_minutes * 60)].copy()


def backward(arr):
    return list(reversed(list(arr)))


def filter_first_5mins(df):
    return df.query('timestamp >= 300').copy()


def remove_likes(df):
    likes = fnmatch.filter(df.columns, 'likes*')
    return df.drop(columns=likes)


def reset_index(df):
    return df.reset_index(drop=True)


def rename_controls(df):
    df.loc[df['genotype'].str.contains('ctrl'), 'genotype'] = 'control'
    return df


def add_trial_count(df):
    assert 'onset' in df, 'run add_onset_and_duration(df) to add onset column'
    df['trial_count'] = df['onset'].cumsum()
    return df


def add_onset_and_duration(df, syll_key='predicted_syllable'):
    sylls = df[syll_key].to_numpy()

    durs = syll_duration(sylls)

    df['onset'] = df[syll_key].diff().fillna(1) != 0
    df.loc[df['onset'], 'dur'] = durs
    df['dur'] = df['dur'].fillna(method='ffill').astype('uint16')
    return df


def add_previous_syllable(df, syll_key='predicted_syllable'):
    sylls = df[syll_key].to_numpy()
    idx = df.index
    first_syllable = syll_onset(sylls)
    next_syllable = first_syllable[1:]
    df['previous_syllable'] = np.nan

    df.loc[idx[next_syllable],
           'previous_syllable'] = df.loc[idx[first_syllable[:-1]], syll_key]
    df['previous_syllable'] = df['previous_syllable'].fillna(
        method='ffill').astype('int16')

    return df


def apply_zscore_per_mouse(df, keys):
    '''zscore the list of `keys` on a mouse by mouse basis, assuming
    the mouse identifier is located in the dataframe column "mouse_id"
    '''
    zdf = df[list(keys) + ['mouse_id']].copy()
    zdf = zdf.groupby('mouse_id').apply(zscore)
    for k in keys:
        df[f'z{k}'] = zdf[k].astype('float32')
    return df


def smooth_cols_by_uuid(df, keys, win, win_type=None, win_args={}, silent=False):
    new_keys = [f'{k}_smooth' for k in keys]
    # create new keys
    for k in new_keys:
        df[k] = np.nan
    for _, sesh_df in tqdm(df.groupby('uuid'), disable=silent):
        val = sesh_df[keys].rolling(win, center=True, win_type=win_type,
                                    min_periods=1).mean(**win_args)
        for k in keys:
            df.loc[val.index, f'{k}_smooth'] = val[k].astype('float32')
    return df


def pca_zscore_per_mouse(df):
    return apply_zscore_per_mouse(df, [f'pc{p:02d}' for p in range(10)])


def add_area(df):
    df['genotype'] = df['genotype'].fillna('n/a')

    df.loc[(df['mouse_id'] == '1527').values, 'genotype'] = 'ctrl'

    area_map = {
        "snc-dls-ai32": "snc (axon)",
        "snc-ai32": "snc (cell)",
        "vta-nacc-ai32": "vta (axon)",
        "vta-ai32": "vta (cell)",
        "vta-nacc-ai32-jrcamp1a": "vta (axon)",
        "snc-dls-ai32-jrcamp1a": "snc (axon)",
        "ctrl": "ctrl",
        "vta-nacc-ctrl": "ctrl",
        "snc-dls-ctrl": "ctrl",
        "vta-ctrl": "ctrl",
        "snc-ctrl": "ctrl",
        "dls-chrimson-dlight": "snc (axon)",
    }

    area_pooled_map = {
        "snc-dls-ai32": "snc",
        "snc-ai32": "snc",
        "snc-dls-ai32-jrcamp1a": "snc",
        "vta-nacc-ai32": "vta",
        "vta-ai32": "vta",
        "vta-nacc-ai32-jrcamp1a": "vta",
        "ctrl": "ctrl",
        "vta-nacc-ctrl": "ctrl",
        "snc-dls-ctrl": "ctrl",
        "vta-ctrl": "ctrl",
        "snc-ctrl": "ctrl",
    }

    df['area'] = df['genotype'].map(area_map)
    df['area (pooled)'] = df['genotype'].map(area_pooled_map)

    idx = df['mouse_id'].isin(EXCLUDE_MICE)
    df.loc[idx, 'area'] = 'ctrl (thal)'
    df.loc[idx, 'area (pooled)'] = 'ctrl (thal)'

    return df

