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

def insert(dst, loc, obj, label=None, ignore_index=False):
    """
    loc : int
        Insertion index. Must verify 0 <= loc <= len(dst).
    obj : row or column
    index : index value (if not provided index will be reset)"""
    
    n = len(dst)
    if not 0 <= loc <= n:
        raise ValueError('Must verify 0 <= loc <= len(dst)')
    if isinstance(dst, pd.DataFrame):
        if isinstance(obj, (list, tuple)):
            dst1 = pd.DataFrame([obj], columns=dst.columns, 
                               index=[0 if label is None else label])
        elif isinstance(obj, dict):
            dst1 = pd.DataFrame(obj, 
                               index=[0 if label is None else label])
        elif isinstance(obj, pd.DataFrame):
            dst1 = obj
        else:
            raise TypeError(f'Received obj of type {type(obj)}')
    elif isinstance(dst, pd.Series):
        if isinstance(obj, pd.Series):
            dst1 = obj
        else:
            dst1 = pd.Series(obj, index=[0 if label is None else label])
#     if loc==0:    # just for speed, not really necessary
#         return pd.concat([dst1, dst], ignore_index=label is None)
    if loc==n:  # just for speed, not really necessary
        return pd.concat([dst, dst1], ignore_index=ignore_index)
    else:
        return pd.concat([dst[:loc], dst1, dst[loc:]], ignore_index=ignore_index)
