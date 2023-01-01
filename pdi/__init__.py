import numpy as np
import pandas as pd

from .drop import drop
from .sidebyside import sidebyside

__all__ = [
    "find",
    "findall",
    "drop",
]

def find(s, x, pos=False):
    """
    Finds element x in Series s. 
    Returns index label (default) or positional index (pos=True).
    Raises ValueError if x is missing from s.
    """
    if len(s) < 1000:
        try:
            idx = s.tolist().index(x)
        except ValueError:
            raise ValueError(f'{x!r} is not in Series')
    else:
        indices = np.where(s == x)[0]
        if not len(indices):
            raise ValueError(f'{x!r} is not in Series')
        idx = indices[0]
    if pos:
        return idx
    else:
        return s.index[idx]

def findall(s, x, pos=False):
    """
    Finds all occurrences of element x in Series s.
    Returns a list of index labels (default) or positional indexes (pos=True).
    """
    idx = np.where(s == x)[0]
    if pos:
        return idx
    else:
        return s.index[idx]

def insert(dst: 'NDFrame', 
           loc: 'int', 
           value: 'Scalar | AnyArrayLike', 
           label: 'Hashable' = 0, 
           ignore_index: 'bool' = False, 
           allow_duplicates: 'bool' = None):
    """
    Inserts DataFrame, Series, list, tuple or dict into DataFrame as a row.
    Also inserts Series, list or a tuple into a Series.

    dst : 
        Series or DataFrame to insert into.
    loc : int
        Insertion index. Must verify 0 <= loc <= len(dst).
    value : 
        Row(s) to insert. 
    label : scalar or tuple
        Index label (use tuple for a MultiIndex label).
    ignore_index : bool, default False
        If True, do not use the existing index. The resulting rows will be 
        labeled 0, ..., n - 1. This is useful if you are inserting into a
        Series or a DataFrame which index does not have meaningful information.
    allow_duplicates : bool, default False 
        Check for duplicates before inserting
    """
    
    n = len(dst)
    if not 0 <= loc <= n:
        raise ValueError('Must verify 0 <= loc <= len(dst)')
    
    if isinstance(dst, pd.DataFrame):
        if isinstance(value, (list, tuple)):
            dst1 = pd.DataFrame([value], columns=dst.columns, index=[label])
        elif isinstance(value, dict):
            dst1 = pd.DataFrame(value, index=[label])
        elif isinstance(value, pd.DataFrame):
            dst1 = value
        else:
            raise TypeError(f'Received value of type {type(value)}')
    
    elif isinstance(dst, pd.Series):
        if isinstance(value, pd.Series):
            dst1 = value
        else:
            dst1 = pd.Series(value, index=[label])
    
    if allow_duplicates is False and len(dst.index.intersection(dst1.index)):
        raise ValueError(f'cannot insert label {label!r}, already exists; consider ignore_index=True')

    if loc==n:  # just for speed, not really necessary
        return pd.concat([dst, dst1], ignore_index=ignore_index)
    else:
        return pd.concat([dst[:loc], dst1, dst[loc:]], ignore_index=ignore_index)

def join(dfs, on=None, how='left', suffixes=None):
    """
    Joins several dataframes. 

    dfs : 
        A list of Series or DataFrames.
    on :
        Specifies columns of the first frame. Other dataframes are always joined by index.
    how : {'left', 'right', 'outer', 'inner'}, default 'left'
        Can either be a string or a list of len(dfs)-1 strings.
    suffixes :
        A list of len(dfs) strings. Suffixes support is not perfect, yet covers 
        basic use cases.
    """
    if len(dfs) < 2:
        raise ValueError('Two or more dataframes expected')
    if on is None and suffixes is None and how == 'left':
        return dfs[0].join(dfs[1:])
    else:
        df = dfs[0]
        n = len(dfs)-1
        
        if on is None:
            on = [None]*n
        elif isinstance(on, str):
            on = [on] * n
        elif isinstance(on, (list, tuple)) and len(on) != n:
            raise ValueError(f'"on" is expected to be either a string or a list of length {n}')

        if isinstance(how, str):
            how = [how] * n
        elif isinstance(how, (list, tuple)) and len(on) != n:
            raise ValueError(f'"how" is expected to be either a string or a list of length {n}')
        
        if suffixes is None:
            suffixes=[''] * (n+1)
        elif isinstance(suffixes, (list, tuple)) and len(suffixes) != n+1:
            raise ValueError(f'"suffixes" is expected to be a list of length {n+1}')
        
        for i, df_i in enumerate(dfs[1:]):
            df = df.join(df_i, on=on[i], how=how[i], lsuffix=suffixes[i], rsuffix=suffixes[i+1])
        return df
