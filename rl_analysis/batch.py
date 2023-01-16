from joblib import Parallel, delayed
from typing import Callable, Optional
from tqdm.auto import tqdm
import pandas as pd
import numpy as np


def apply_parallel_joblib(
    group_obj: pd.core.groupby.generic.DataFrameGroupBy,
    func: Callable,
    n_jobs: int = 10,
    verbose: int = 10,
    backend: str = "loky",
    batch_size: str = "auto",
    dask_address: Optional[str] = None,
    dask_extra: dict = {},
) -> Optional[pd.DataFrame]:

    if (backend != "dask") or dask_address is None:
        ret_list = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend, batch_size=batch_size)(
            delayed(func)(group) for name, group in group_obj
        )
    else:
        from dask.distributed import Client, progress
        import dask.delayed as dask_delayed

        client = Client(dask_address)
        futures = client.compute([dask_delayed(func)(group) for name, group in group_obj])
        # progress(futures)
        ret_list = client.gather(futures)

    # get the name of the grouper columns
    grouper_cols = group_obj.grouper.names

    # get a tuple corresponding to key per group
    # grouper_keys = group_obj.groups.keys() # NOT GUARANTEED TO PRESERVE ORDER
    grouper_keys = []
    for k, v in group_obj:
        grouper_keys.append(k)

    if ret_list is None:
        return None

    # fix up a multi index  pandas style
    for _keys, _df in zip(grouper_keys, ret_list):
        if _df is None:
            continue

        # now we form a multi-index, first get the original index
        idx = _df.index.to_numpy()
        indices = []

        # now for each grouper expand to make a multi-index
        if isinstance(_keys, tuple):
            for _key in _keys:
                indices.append([_key] * len(idx))
        else:
            indices.append([_keys] * len(idx))
        indices.append(idx)

        # get the names of our grouper columns
        indices_names = []
        for _col in grouper_cols:
            indices_names.append(_col)
        indices_names.append(_df.index.name)

        # build a multi-index and attach
        new_idx = pd.MultiIndex.from_arrays(indices, names=indices_names)
        _df.index = new_idx

    ret_list = [_ for _ in ret_list if _ is not None]

    # stitch the results into a dataframe
    return pd.concat(ret_list)



# def digest_futures(client, futures):
#     # make sure we maintain the key for sorting
#     results = {}
#     seq = as_completed(futures)
#     pbar = tqdm(seq, total=len(futures))
#     for _future in seq:
#         try:
#             _result = _future.result()
#             _key = _future.key
#             results[_key] = _result
#             #         client.cancel(_future)
#             pbar.update(n=1)
#         except KilledWorker:
#             #         client.cancel(_future)
#             client.retry(_future)
#             seq.add(_future)
#         except CancelledError:
#             client.retry(_future)
#             seq.add(_future)
#     #         client.cancel(_future)
#     #         continue
#     pbar.close()
#     return results


def sort_futures(results, futures, keys):
    # assumes we have a dictionary of results, a list of futures, and a list of matching keys
    sorted_results = {}
    future_keys = [_.key for _ in futures]
    for _key, _result in tqdm(results.items()):
        _idx = future_keys.index(_key)
        new_key = keys[_idx]
        sorted_results[new_key] = _result
    
    return sorted_results


def dask_batched_submission(client, delays, batch_size=5000):
    from functools import reduce

    all_results = []
    for _chunk in tqdm(np.arange(0, len(delays), batch_size), smoothing=0):
        futures = client.compute(delays[_chunk : _chunk + batch_size], retries=20)
        results = digest_futures(futures)
        all_results.append(results)

    return reduce(lambda x, y: {**x, **y}, all_results)


def dask_batched_submission_simple(client, delays, batch_size=5000):
    from functools import reduce

    all_results = []
    for _chunk in tqdm(np.arange(0, len(delays), batch_size), smoothing=0):
        futures = client.compute(delays[_chunk : _chunk + batch_size])
        results = client.gather(futures)
        for _result in results:
            all_results.append(_result)

    return all_results
    # return reduce(lambda x, y: {**x, **y}, all_results)


def digest_futures(futures):
    # make sure we maintain the key for sorting
    from dask.distributed import as_completed
    results = {}
    seq = as_completed(futures, with_results=True, raise_errors=False)
    pbar = tqdm(seq, total=len(futures), smoothing=0)
    for batch in seq.batches():
        for _future, _result in batch:
            if _future.status != "finished":
                pass
                # _future.retry()
                # seq.add(_future)
            else:
                results[_future.key] = _result
                pbar.update(n=1)

    pbar.close()
    return results