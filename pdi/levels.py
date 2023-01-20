from __future__ import annotations

import pandas as pd
from typing import Hashable, Sequence

from pandas._typing import (
    AnyArrayLike,
    Axis,
    Scalar,
)
from pandas.core.dtypes.common import (
    is_list_like,
    is_scalar,
)
from pandas.core.indexes.multi import _require_listlike
from pandas.core.generic import NDFrame
from pandas._libs import lib

def get_level(obj, level, axis=0):
    """
    Returns a complete level of a MultiIndex (alias to .get_level_values).
    
    obj :
        Series, DataFrame or MultiIndex

    level : int or scalar
        Positional index or name of the level to return
    
    axis : int or str
        0, index, rows = index;
        1, columns = columns.
    """

    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return get_level(obj._get_axis(axis), level)
    elif isinstance(obj, pd.MultiIndex):
        mi = obj
    elif isinstance(obj, pd.Index):
        if isinstance(level, int):
            if level != 0:
                raise IndexError(f"Index only has level=0, not {level}")
        elif level != obj.name:
            raise KeyError(f"Index has name={obj.name}, not {level}")
        return obj
    else:
        raise TypeError(
                f"The first argument must a DataFrame, a Series or "
                "a MultiIndex, not {type(mi)}."
        )
    return mi.get_level_values(level=level)

def set_level(obj, level, labels, name=lib.no_default, axis=0, inplace=False):
    """
    Replaces a complete level of a MultiIndex.

    obj :
        Series, DataFrame or MultiIndex

    level : int or scalar
        Positional index or name of the level

    labels :
        One of: 
           - Index, attaches as is
           - Scalar, broadcasts to match MultiIndex length and attaches
           - list/NumPy array sized as proper divisor of MultiIndex length: broadcasts
           to match MultiIndex length and attaches
           - list/NumPy array of size equal to MultiIndex: builds and attaches

    name :
        New name of the level. If not specified:
          - keeps original name if `level` is a list or array or 
          - keeps name of the `level` if `level` is an Index or a Series.

    axis : int or str
        0, index, rows = index;
        1, columns = columns.
    
    inplace : 
        Return a fresh copy or modify inplace. If mi is a simple pd.Index 
        (=not a MultiIndex) always returns a copy because the Index is immutable.
    """
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        ax, mi = obj._get_axis_name(axis), obj._get_axis(axis)
        if not isinstance(mi, pd.MultiIndex):
            raise TypeError(f"MultiIndex expected in the {ax}, got {type(mi)}.")
        if inplace is False:
            obj = obj.copy()
        index = set_level(mi, level, labels, name=name, inplace=False)
        setattr(obj, ax, index)
        if inplace:
            return
        else:
            return obj
    elif isinstance(obj, pd.MultiIndex):
        ax, mi = "index", obj
    else:
        raise TypeError(
                f"The first argument must a DataFrame, a Series or "
                "a MultiIndex, not {type(mi)}."
        )
    
    level_num = mi._get_level_number(level)
    
    if isinstance(labels, pd.Index):
        index = labels
        if name is not lib.no_default:
            index.name = name
    else:       # scalar or array
        n = len(mi)
        if is_scalar(labels):
            labels = [labels] * n
        elif len(labels) == n:
            pass
        else:
            raise ValueError(
                f"len(labels)={len(labels)}; must be 1, {n} or a proper divisor of {n}"
            )
        index = pd.Index(labels)
        if name is not lib.no_default:
            index.name = name
        elif not isinstance(labels, pd.Series):
            index.name = mi.names[level_num]
    
    levels = []
    for i in range(mi.nlevels):
        if i == level_num:
            levels.append(index)
        else:
            levels.append(mi.get_level_values(i))

    idx = pd.MultiIndex.from_arrays(levels)

    if inplace:
        mi._reset_identity()
        mi._names = idx._names
        mi._levels = idx._levels
        mi._codes = idx._codes
        mi.sortorder = idx.sortorder
        mi._reset_cache()
    else:
        return idx

def insert_level(obj, level, labels, name=lib.no_default, axis=0, inplace=False, sort=False):
    """
    Inserts a complete level of a MultiIndex

    obj :
        Series, DataFrame or MultiIndex

    level :
        Positional index or name of the level to prepend to. 
        Use level=index.nlevels to append

    labels :
        One of: 
           - Index, attaches as is
           - Scalar, broadcasts to match MultiIndex length and attaches
           - list/NumPy array of size equal to MultiIndex: builds and attaches

    name :
        New name of the level. If not specified:
          - None if `labels` is a list or array or
          - keeps name of the `labels` if `labels` is an Index or a Series

    axis : int or str
        0, index, rows = index;
        1, columns = columns.

    inplace : bool
        Return a fresh copy or modify inplace.

    sort : bool
        Sorts the resulting index. Only works if obj is a Series or a DataFrame
    """
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        ax, mi = obj._get_axis_name(axis), obj._get_axis(axis)
        if not isinstance(mi, pd.MultiIndex):
            raise TypeError(f"MultiIndex expected in the {ax}, got {type(mi)}.")
        if inplace is False:
            obj = obj.copy()
        index = insert_level(mi, level, labels, name=name, inplace=False, sort=False)
        setattr(obj, ax, index)
        if sort:
            obj.sort_index(axis=axis, inplace=True)
        if inplace:
            return
        else:
            return obj
    elif isinstance(obj, pd.MultiIndex):
        ax, mi = "index", obj
    else:
        raise TypeError(
                f"The first argument must a DataFrame, a Series or "
                "a MultiIndex, not {type(mi)}."
        )
    
    if sort is True:
        raise ValueError(
            "Cannot sort MultiIndex, use pdi.move_level(df, axis=..., sort=True) instead"
        )

    if isinstance(level, int) and level >= mi.nlevels:
        level_num = level  # append mode
    else:
        level_num = mi._get_level_number(level)
    
    if isinstance(labels, pd.Index):
        index = labels
        if name is not lib.no_default:
            index.name = name
    else:
        n = len(mi)
        if is_scalar(labels):
            labels = [labels] * n
        elif len(labels) == n:
            pass
        else:
            raise ValueError(f"len(labels)={len(labels)}; must be 1 or {n}")
        
        if isinstance(labels, pd.Index):
            index = labels
        else:
            index = pd.Index(labels)
        
        if name is not lib.no_default:
            index.name = name
        elif not isinstance(labels, (pd.Series, pd.Index)):
            index.name = None
    
    levels = []
    for i in range(mi.nlevels):
        levels.append(mi.get_level_values(i))
    levels.insert(level_num, index)

    idx = pd.MultiIndex.from_arrays(levels)

    if inplace:
        mi._reset_identity()
        mi._names = idx._names
        mi._levels = idx._levels
        mi._codes = idx._codes
        mi.sortorder = idx.sortorder
        mi._reset_cache()
    else:
        return idx

def drop_level(obj, level, axis=0, inplace=False):
    """
    Drops a complete level of a MultiIndex

    obj :
        Series, DataFrame or MultiIndex

    level : int, scalar or list of ints or scalars
        Positional index(es) or name(s) of the level to drop. 

    axis : int or str
        0, index, rows = index;
        1, columns = columns.

    inplace : bool
        Return a fresh copy or modify inplace.
    """
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        ax, mi = obj._get_axis_name(axis), obj._get_axis(axis)
        if not isinstance(mi, pd.MultiIndex):
            raise TypeError(f"MultiIndex expected in the {ax}, got {type(mi)}.")
        if inplace is False:
            obj = obj.copy()
        index = drop_level(mi, level, inplace=False)
        setattr(obj, ax, index)
        if not inplace:
            return obj
    elif isinstance(obj, pd.MultiIndex):
        ax, mi = "index", obj
    else:
        raise TypeError(f"The first argument must a DataFrame, a Series or "
                        "a MultiIndex, not {type(mi)}.")

    to_drop = []
    if is_scalar(level):
        level = [level]

    to_drop = [mi._get_level_number(lev) for lev in level]
    
    levels = []
    for i in range(mi.nlevels):
        if i not in to_drop:
            levels.append(mi.get_level_values(i))

    idx = pd.MultiIndex.from_arrays(levels)

    if inplace:
        mi._reset_identity()
        mi._names = idx._names
        mi._levels = idx._levels
        mi._codes = idx._codes
        mi.sortorder = idx.sortorder
        mi._reset_cache()
    else:
        return idx

def move_level(obj, src, dst, axis=0, inplace=False, sort=False):
    """
    Moves a complete level of a MultiIndex to a designated position

    obj :
        Series, DataFrame or MultiIndex

    src :
        Positional index or name of the level to move. 
    
    dst :
        Positional index or name of the level to prepend. 
        Use `dst=index.nlevels` to move after the last level.

    axis : int or str
        0, index, rows = index;
        1, columns = columns.

    inplace : bool
        Return a fresh copy or modify inplace.

    sort : bool
        Sorts the resulting index. Only works if obj is a Series or a DataFrame
    """
    
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        ax, mi = obj._get_axis_name(axis), obj._get_axis(axis)
        if not isinstance(mi, pd.MultiIndex):
            raise TypeError(f"MultiIndex expected in the {ax}, got {type(mi)}.")
        if inplace is False:
            obj = obj.copy()
        index = move_level(mi, src, dst, inplace=False, sort=False)
        setattr(obj, ax, index)
        if sort:
            obj.sort_index(axis=axis, inplace=True)
        if inplace:
            return
        else:
            return obj
    elif isinstance(obj, pd.MultiIndex):
        ax, mi = "index", obj
    else:
        raise TypeError(
                f"The first argument must a DataFrame, a Series or "
                "a MultiIndex, not {type(mi)}."
        )
    
    if sort is True:
        raise ValueError(
            "Cannot sort MultiIndex, use pdi.move_level(df, axis=..., sort=True) instead"
        )
    src_num = mi._get_level_number(src)
    if isinstance(dst, int) and dst >= mi.nlevels:
        dst_num = dst    # append mode
    else:
        dst_num = mi._get_level_number(dst)
    
    levels = []
    for i in range(mi.nlevels):
        levels.append(mi.get_level_values(i))

    if src_num < dst_num:
        levels.insert(dst_num, levels[src_num])
        del levels[src_num]
    elif src_num > dst_num:
        levels.insert(dst_num,levels.pop(src_num))
    
    idx = pd.MultiIndex.from_arrays(levels)

    if inplace:
        mi._reset_identity()
        mi._names = idx._names
        mi._levels = idx._levels
        mi._codes = idx._codes
        mi.sortorder = idx.sortorder
        mi._reset_cache()
    else:
        return idx

def swap_levels(obj, i: Axis = -2, j: Axis = -1, axis: Axis = 0, inplace=False, sort=False):
    """
    Moves a complete level of a MultiIndex to a designated position

    obj :
        Series, DataFrame or MultiIndex

    i, j :
        Positional indices or names of the levels to swap. 
    
    axis : int or str
        0, index, rows = index;
        1, columns = columns.

    inplace : bool
        Return a fresh copy or modify inplace.

    sort : bool
        Sorts the resulting index. Only works if obj is a Series or a DataFrame
    """
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        ax, mi = obj._get_axis_name(axis), obj._get_axis(axis)
        if not isinstance(mi, pd.MultiIndex):
            raise TypeError(f"MultiIndex expected in the {ax}, got {type(mi)}.")
        if inplace is False:
            obj = obj.copy()
        index = swap_levels(mi, i, j, inplace=False, sort=False)
        setattr(obj, ax, index)
        if sort:
            obj.sort_index(axis=axis, inplace=True)
        return None if inplace else obj
    elif isinstance(obj, pd.MultiIndex):
        ax, mi = "index", obj
    else:
        raise TypeError(
                f"The first argument must a DataFrame, a Series or "
                "a MultiIndex, not {type(mi)}."
        )
    
    if sort is True:
        raise ValueError(
            "Cannot sort MultiIndex, use pdi.move_level(df, axis=..., sort=True) instead"
        )
    
    idx = mi.swaplevel(i, j)

    if inplace:
        mi._reset_identity()
        mi._names = idx._names
        mi._levels = idx._levels
        mi._codes = idx._codes
        mi.sortorder = idx.sortorder
        mi._reset_cache()
    else:
        return idx


