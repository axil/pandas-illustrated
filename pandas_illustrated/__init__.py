import numpy as np
import pandas as pd

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

def insert(df, loc, obj, label=None):
    """
    loc : int
        Insertion index. Must verify 0 <= loc <= len(df).
    obj : row or column
    index : index value (if not provided index will be reset)"""
    
    n = len(df)
    if not 0 <= loc <= n:
        raise ValueError('Must verify 0 <= loc <= len(df)')
    if isinstance(obj, (list, tuple)):
        df1 = pd.DataFrame([obj], columns=df.columns, 
                           index=[0 if label is None else label])
    elif isinstance(obj, dict):
        df1 = pd.DataFrame(obj, 
                           index=[0 if label is None else label])
    elif isinstance(obj, pd.DataFrame):
        df1 = obj
    else:
        raise TypeError(f'Received obj of type {type(obj)}')
    if loc==n:  # just for speed, not really necessary
        return pd.concat([df, df1], ignore_index=label is None)
    else:
        return pd.concat([df[:loc], df1, df[loc:]], ignore_index=label is None)
