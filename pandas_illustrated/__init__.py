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

def findall(s, x):
    return s.index[np.where(s == x)[0]]

def insert(dst: 'NDFrame', 
           loc: 'int', 
           value: 'Scalar | AnyArrayLike', 
           label: 'Hashable' = 0, 
           ignore_index: 'bool' = False, 
           allow_duplicates: 'bool' = None):
    """
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

