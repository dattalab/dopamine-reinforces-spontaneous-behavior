import pandas as pd
import numpy as np
from rl_analysis.util import count_transitions
from rl_analysis.info.util import dm_entropy


# takes a syllable-RLE time-series and spits out the number of
# times the current syllable happens over the next N timesteps
#
# e.g. if 45 50 45 45 37 37 50 happens, then
# counts_5 would return 3 1 2 for the 3 indices
def get_counts_persample(
    df: pd.DataFrame, bin_sizes: list[int] = [60, 120], truncate: int = 37
) -> pd.DataFrame:
    nvalues = len(df)
    index = df.index
    all_series = []
    for _sz in bin_sizes:
        new_series = pd.Series(index=index, name=f"counts_{_sz}", dtype="float32")
        for _idx in index:
            if (_idx + _sz) > nvalues:
                new_series.loc[_idx] = np.nan
            else:
                slc = df.loc[_idx : _idx + _sz].values
                cur_syllable = df.iat[_idx]
                new_series.loc[_idx] = (slc == cur_syllable).sum()
        all_series.append(new_series)
    return pd.concat(all_series, axis=1)


# takes a syllable-RLE time-series and returns the entropy
# for the syllable at each index
def get_entropy_persample(
    df: pd.DataFrame, bin_sizes: list[int] = [60, 120], truncate: int = 37
) -> pd.DataFrame:
    nvalues = len(df)
    index = df.index
    all_series = []
    for _sz in bin_sizes:
        new_series = pd.Series(index=index, name=f"entropy_{_sz}", dtype="float32")
        for _idx in index:
            if (_idx + _sz) > nvalues:
                new_series.loc[_idx] = np.nan
            else:
                slc = df.loc[_idx: _idx + _sz].values
                tm = count_transitions(slc, K=100)
                ent = dm_entropy(
                    tm[:truncate, :truncate], alpha="perks", marginalize=False, axis=1
                )
                new_series.loc[_idx] = ent
        all_series.append(new_series)
    return pd.concat(all_series, axis=1)


# if we have a pandas column with 2D arrays representing TMs,
# this splits each row of the TM into a row of the series. Used
# to convert a 2D tm to a series of rows representing outbounds...
def split_array(x: np.ndarray, mapping: dict = {}) -> pd.Series:
	index = pd.Series(np.arange(len(x)))
	rows = pd.Series(data=x.tolist(), index=index.map(mapping))
	rows.index.name = "labels"
	rows = rows[rows.index.notnull()]
	rows.index = rows.index.astype("int")
	return rows