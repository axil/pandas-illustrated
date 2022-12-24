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

def insert(df, loc, row, index=None):
    """
    loc : int
        Insertion index. Must verify 0 <= loc <= len(df).
    row : list, tuple or dict
    index : index value (if not provided index will be reset)"""
    
    n = len(df)
    if not 0 <= loc <= n:
        raise ValueError('Must verify 0 <= loc <= len(df)')
    if isinstance(row, (list, tuple)):
        df1 = pd.DataFrame([row], columns=df.columns, 
                           index=[0 if index is None else index])
    elif isinstance(row, dict):
        df1 = pd.DataFrame(row, 
                           index=[0 if index is None else index])
    if loc==0:    # just for speed, not really necessary
        return pd.concat([df1, df], ignore_index=index is None)
    elif loc==n:  # also just for speed, not really necessary
        return pd.concat([df, df1], ignore_index=index is None)
    else:
        return pd.concat([df[:loc], df1, df[loc:]], ignore_index=index is None)
