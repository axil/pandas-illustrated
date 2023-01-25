from __future__ import annotations

import pandas as pd
import numpy as np
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

def get_level(obj, level_id, axis=None):
    """
    Returns a complete level of a MultiIndex (alias to .get_level_values).
    
    obj :
        Series, DataFrame or MultiIndex

    level_id : int or scalar
        Positional index or name of the level to return
    
    axis : int, str or None
        0, 'index', 'rows' = index;
        1, 'columns' = columns;
        None = index for Series, columns for DataFrame.
    """

    if isinstance(obj, (pd.Series, pd.DataFrame)):
        if axis is None:
            axis = obj._info_axis_name
        return get_level(obj._get_axis(axis), level_id)
    elif isinstance(obj, pd.MultiIndex):
        mi = obj
    elif isinstance(obj, pd.Index):
        if isinstance(level_id, int):
            if level_id != 0:
                raise IndexError(f"Index only has level_id=0, not {level_id}")
        elif level_id != obj.name:
            raise KeyError(f"Index has name={obj.name}, not {level_id}")
        return obj
    else:
        raise TypeError(
                f"The first argument must a DataFrame, a Series or "
                "a MultiIndex, not {type(mi)}."
        )
    return mi.get_level_values(level_id)

def set_level(obj, level_id, labels, name=lib.no_default, axis=None, inplace=False):
    """
    Replaces a complete level of a MultiIndex.

    obj :
        Series, DataFrame or MultiIndex

    level_id : int or scalar
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
          - keeps original level name if `labels` is a list or an array or 
          - keeps name of the `labels` if `labels` is an Index or a Series.

    axis : int, str or None
        0, 'index', 'rows' = index;
        1, 'columns' = columns;
        None = index for Series, columns for DataFrame.
    
    inplace : 
        Return a fresh copy or modify inplace. If mi is a simple pd.Index 
        (=not a MultiIndex) always returns a copy because the Index is immutable.
    """
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        if axis is None:
            axis = obj._info_axis_name
        ax, mi = obj._get_axis_name(axis), obj._get_axis(axis)
        if not isinstance(mi, pd.MultiIndex):
            raise TypeError(f"MultiIndex expected in the {ax}, got {type(mi)}.")
        if inplace is False:
            obj = obj.copy()
        index = set_level(mi, level_id, labels, name=name, inplace=False)
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
    
    level_num = mi._get_level_number(level_id)
    
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
            msg = f"len(labels)={len(labels)}; must be 1"
            if n != 1:
                msg += " or {n}."
            raise ValueError(msg)
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

def insert_level(obj, pos, labels, name=lib.no_default, axis=None, inplace=False, sort=False):
    """
    Inserts a complete level of a MultiIndex

    obj :
        Series, DataFrame or MultiIndex

    pos :
        Positional index or name of the level to prepend to. 
        Use pos=index.nlevels to append

    labels :
        One of: 
           - Index, attaches as is
           - Scalar, broadcasts to match MultiIndex length, builds Index and attaches
           - list/NumPy array of size equal to MultiIndex: builds Index and attaches

    name :
        New name of the level. If not specified:
          - None if `labels` is a list or array or
          - keeps name of the `labels` if `labels` is an Index or a Series

    axis : int, str or None
        0, 'index', 'rows' = index;
        1, 'columns' = columns;
        None = index for Series, columns for DataFrame.
    
    inplace : bool
        Return a fresh copy or modify inplace.

    sort : bool
        Sorts the resulting index. Only works if obj is a Series or a DataFrame
    """

    if isinstance(obj, (pd.Series, pd.DataFrame)):
        if axis is None:
            axis = obj._info_axis_name
        ax, mi = obj._get_axis_name(axis), obj._get_axis(axis)
#        if not isinstance(mi, pd.MultiIndex):
#            raise TypeError(f"MultiIndex expected in the {ax}, got {type(mi)}.")
        if inplace is False:
            obj = obj.copy()
        index = insert_level(mi, pos, labels, name=name, inplace=False, sort=False)
        setattr(obj, ax, index)
        if sort:
            obj.sort_index(axis=axis, inplace=True)
        if inplace:
            return
        else:
            return obj
    elif isinstance(obj, pd.MultiIndex):
        ax, mi = "index", obj
    elif isinstance(obj, pd.Index):
        if inplace is True:
            raise ValueError("Index is immutable. Use `inplace=False`.")
        ax, mi = "index", obj
    else:
        raise TypeError(
                "The first argument must a DataFrame, a Series or an Index, "
                f"not {type(obj)}."
        )
    
    if sort is True:
        raise ValueError(
            "Cannot sort MultiIndex, use pdi.move_level(df, axis=..., sort=True) instead"
        )

    if isinstance(pos, int) and pos >= mi.nlevels:
        level_num = pos  # append mode
    else:
        level_num = mi._get_level_number(pos)
    
    n = len(mi)
    if is_scalar(labels):
        index = pd.Index([labels] * n)
    elif isinstance(labels, pd.Index):
        index = labels
    elif isinstance(labels, (list, np.ndarray, pd.Series)):
        index = pd.Index(labels)
    else:
        raise TypeError(
            f"Expected Index, scalar, list or numpy array, got {type(labels)}"
        )
    
    if len(index) != n:
        raise ValueError(f"len(labels)={len(labels)}; must be 1 or {n}")
    
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

def drop_level(obj, level_id, axis=None, inplace=False):
    """
    Drops a complete level of a MultiIndex.
    If only one level remains (and `inplace` is not True),
    a simple Index is returned.

    obj :
        Series, DataFrame or MultiIndex

    level_id : int, scalar or list of ints or scalars
        Positional index(es) or name(s) of the level to drop. 

    axis : int, str or None
        0, 'index', 'rows' = index;
        1, 'columns' = columns;
        None = index for Series, columns for DataFrame.
    
    inplace : bool
        Return a fresh copy or modify inplace.
    """
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        if axis is None:
            axis = obj._info_axis_name
        ax, mi = obj._get_axis_name(axis), obj._get_axis(axis)
        if not isinstance(mi, pd.MultiIndex):
            raise TypeError(f"MultiIndex expected in the {ax}, got {type(mi)}.")
        if inplace is False:
            obj = obj.copy()
        index = drop_level(mi, level_id, inplace=False)
        setattr(obj, ax, index)
        if not inplace:
            return obj
    elif isinstance(obj, pd.MultiIndex):
        ax, mi = "index", obj
    else:
        raise TypeError("The first argument must a DataFrame, a Series or "
                        f"a MultiIndex, not {type(obj)}.")

    to_drop = []
    if is_scalar(level_id):
        level_id = [level_id]

    to_drop = [mi._get_level_number(lev) for lev in level_id]
    
    levels = []
    for i in range(mi.nlevels):
        if i not in to_drop:
            levels.append(mi.get_level_values(i))

    if inplace is False and len(levels) == 1:
        idx = pd.Index(levels[0])
    else:
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

def move_level(obj, src, dst, axis=None, inplace=False, sort=False):
    """
    Moves a complete level of a MultiIndex to a designated position

    obj :
        Series, DataFrame or MultiIndex

    src :
        Positional index or name of the level to move. 
    
    dst :
        Positional index or name of the level to prepend. 
        Use `dst=index.nlevels` to move after the last level.

    axis : int, str or None
        0, 'index', 'rows' = index;
        1, 'columns' = columns;
        None = index for Series, columns for DataFrame.
    
    inplace : bool
        Return a fresh copy or modify inplace.

    sort : bool
        Sorts the resulting index. Only works if obj is a Series or a DataFrame
    """
    
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        if axis is None:
            axis = obj._info_axis_name
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
            f"First argument must a DataFrame, a Series or a MultiIndex, got {type(obj)}."
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

def swap_levels(obj, i: Axis = -2, j: Axis = -1, axis: Axis = None, inplace=False, sort=False):
    """
    Moves a complete level of a MultiIndex to a designated position

    obj :
        Series, DataFrame or MultiIndex

    i, j :
        Positional indices or names of the levels to swap. 
    
    axis : int, str or None
        0, 'index', 'rows' = index;
        1, 'columns' = columns;
        None = index for Series, columns for DataFrame.
    
    inplace : bool
        Return a fresh copy or modify inplace.

    sort : bool
        Sorts the resulting index. Only works if obj is a Series or a DataFrame
    """
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        if axis is None:
            axis = obj._info_axis_name
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
            f"First argument must a DataFrame, a Series or a MultiIndex, not {type(obj)}."
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

def join_levels(obj, name=None, sep='_', axis=None, inplace=False):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        if axis is None:
            axis = obj._info_axis_name
        ax, mi = obj._get_axis_name(axis), obj._get_axis(axis)
        if not isinstance(mi, pd.MultiIndex):
            raise TypeError(f"MultiIndex expected in the {ax}, got {type(mi)}.")
        if inplace is False:
            obj = obj.copy()
        index = join_levels(mi, name=name, inplace=False)
        setattr(obj, ax, index)
        return None if inplace else obj
    elif isinstance(obj, pd.MultiIndex):
        ax, mi = "index", obj
    else:
        raise TypeError(
            f"First argument must a DataFrame, a Series or a MultiIndex, not {type(obj)}."
        )
    
    idx = pd.Index(
        [sep.join(x) for x in list(zip(*[get_level(mi, i) for i in range(mi.nlevels)]))],
        name=name,
    )

    if inplace:
        raise ValueError(
            "Cannot join levels of MultiIndex inplace, consider "
            "`join_levels(df, inplace=True)`"
        )
    else:
        return idx

def split_level(obj, names=None, sep='_', axis=None, inplace=False):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        if axis is None:
            axis = obj._info_axis_name
        ax, mi = obj._get_axis_name(axis), obj._get_axis(axis)
        if isinstance(mi, pd.MultiIndex) and mi.nlevels > 1:
            raise TypeError("MultiIndex is expected to have only one level")
        if inplace is False:
            obj = obj.copy()
        index = split_level(mi, names=names, inplace=False)
        setattr(obj, ax, index)
        return None if inplace else obj
    elif isinstance(obj, pd.MultiIndex):
        ax, mi = "index", obj.get_level_values(0)
    elif isinstance(obj, pd.Index):
        ax, mi = "index", obj
    else:
        raise TypeError(
            f"First argument must a DataFrame, a Series or a MultiIndex, not {type(obj)}."
        )
    
    idx = pd.MultiIndex.from_tuples(
            [x.split(sep) for x in mi],
            names=names)

    if inplace:
        raise ValueError(
            "Cannot split levels of an Index inplace, consider "
            "`split_levels(df, inplace=True)`"
        )
    else:
        return idx

def rename_level(obj, mapping, axis=None, inplace=False):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        if axis is None:
            axis = obj._info_axis_name
        ax, mi = obj._get_axis_name(axis), obj._get_axis(axis)
        if inplace is False:
            obj = obj.copy()
        index = rename_level(mi, mapping, inplace=False)
        setattr(obj, ax, index)
        return None if inplace else obj
    elif isinstance(obj, pd.MultiIndex):
        ax, mi = "index", obj
    elif isinstance(obj, pd.Index):
        ax, mi = "index", obj
    else:
        raise TypeError(
            f"First argument must a DataFrame, a Series or a MultiIndex, not {type(obj)}."
        )
    
    if isinstance(mi, pd.MultiIndex):
        if isinstance(mapping, dict):
            levels, names = list(mapping.keys()), list(mapping.values())
            idx = mi.set_names(names, level=levels)
        elif mi.nlevels == 1 and is_scalar(mapping):
            idx = mi.set_names(mapping, level=0)
        else:
            raise TypeError("'mapping' must be a dict or a scalar")
    elif isinstance(mapping, dict):
        if mi.name in mapping:
            idx = mi.set_names(mapping[mi.name])
        else:
            idx = mi.copy()
    elif is_scalar(mapping):
        idx = mi.set_names(mapping)
    else:
        raise TypeError("'mapping' must be a dict or a scalar")

    if inplace:
        raise ValueError(
            "Cannot rename levels of an Index inplace, consider "
            "`rename_level(df, inplace=True)`"
        )
    else:
        return idx
