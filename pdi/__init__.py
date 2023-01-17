from __future__ import annotations

import numpy as np
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
try:
    from pandas._typing import NDFrameT
except:  # pandas < 1.4
    from pandas._typing import FrameOrSeries
from pandas._libs import lib

from .drop import drop
from .visuals import patch_series, unpatch_series, sidebyside
from .categoricals import lock_order, from_product, vis_lock

__all__ = [
    "find",
    "findall",
    "insert",
    "drop",
    "move",
    "join",
    "patch_series",
    "unpatch_series",
    "sidebyside",
    "patch_dataframe",
    "from_dict",
    "swap_levels",
    "lock_order",
    "from_product",
    "vis_lock",
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
            raise ValueError(f"{x!r} is not in Series")
    else:
        indices = np.where(s == x)[0]
        if not len(indices):
            raise ValueError(f"{x!r} is not in Series")
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


def insert(
    dst: NDFrame,
    pos: int,
    value: Scalar | AnyArrayLike | Sequence,
    label: Hashable = 0,
    ignore_index: bool = False,
    allow_duplicates: bool = None,
) -> NDFrame:
    """
    Inserts DataFrame, Series, list, tuple or dict into DataFrame as a row.
    Also inserts Series, list or a tuple into a Series.

    dst :
        Series or DataFrame to insert into.
    pos : int
        Insertion index. Must verify 0 <= pos <= len(dst).
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
    if not 0 <= pos <= n:
        raise ValueError("Must verify 0 <= pos <= len(dst)")

    if isinstance(dst, pd.DataFrame):
        if isinstance(value, (list, tuple)):
            dst1 = pd.DataFrame([value], columns=dst.columns, index=[label])
        elif isinstance(value, dict):
            dst1 = pd.DataFrame(value, index=[label])
        elif isinstance(value, pd.DataFrame):
            dst1 = value
        elif isinstance(value, pd.Series):
            dst1 = value.to_frame().T
        else:
            raise TypeError(f"Received value of type {type(value)}")

    elif isinstance(dst, pd.Series):
        if isinstance(value, pd.Series):
            dst1 = value
        else:
            dst1 = pd.Series(value, index=[label])

    else:
        raise TypeError(
            f"Supported dst types are: DataFrame, Series. Got type {type(dst)}"
        )

    if (
        allow_duplicates is False or dst.flags.allows_duplicate_labels is False
    ) and len(dst.index.intersection(dst1.index)):
        msg = f"Cannot insert label {label!r}, already exists and "
        if allow_duplicates is False:
            msg += "allow_duplicates is False, "
        if dst.flags.allows_duplicate_labels is False:
            msg += "dst.flags.allows_duplicate_labels is False, "
        msg += "consider ignore_index=True"
        raise ValueError(msg)

    if pos == n:  # just for speed, not really necessary
        return pd.concat([dst, dst1], ignore_index=ignore_index)
    else:
        return pd.concat([dst[:pos], dst1, dst[pos:]], ignore_index=ignore_index)


def _move(a, pos, val):
    a = a.copy()
    i = a.index(val)
    if i > pos:
        del a[i]
        a.insert(pos, val)
    else:
        a.insert(pos, val)
        del a[i]
    return a


def move(obj, pos, label=None, column=None, index=None, axis=None, reset_index=False):
    """
    Moves DataFrame column or row or Series element to positional index 'pos'.
    One of the following formats can be used:
       — column='A'
       — index='a'
       — label='A', axis=1         # same as column='A'
       — label='a', axis=0         # same as index='a'
       — label='a'                 # axis is not required for Series
    Must verify 0 <= pos <= len(columns or index)
    Renumbers the rows from 0 if drop_index is True
    For example:
       >>> move(df, 0, column='C')    # columns: 'A','B','C' -> 'C','B','A'
       >>> move(df, 3, index='a')       # rows: 'a','b','c' -> 'b','c','a'
       >>> move(df, 0, index=2, reset_index=True)
           A  B         A  B
        0  1  2      0  5  6
        1  3  4  ->  1  1  2
        2  5  6      2  3  4


    """
    if index is not None:
        label = index
        axis = 0
    elif column is not None:
        label = column
        axis = 1

    if isinstance(obj, pd.Series) or axis in (0, "rows"):
        obj = obj.reindex(_move(obj.index.tolist(), pos, label))
        if reset_index:
            obj.reset_index(drop=True, inplace=True)
        return obj
    else:
        return obj.reindex(columns=_move(obj.columns.tolist(), pos, label))


def join(dfs, on=None, how="left", suffixes=None):
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
        raise ValueError("Two or more series or dataframes expected")
    if on is None and suffixes is None and how == "left":
        return dfs[0].join(dfs[1:])
    else:
        df = dfs[0]
        n = len(dfs) - 1

        if on is None:
            on = [None] * n
        elif isinstance(on, str):
            on = [on] * n
        elif isinstance(on, (list, tuple)) and len(on) != n:
            raise ValueError(
                f'"on" is expected to be either a string or a list of length {n}'
            )

        if isinstance(how, str):
            how = [how] * n
        elif isinstance(how, (list, tuple)) and len(how) != n:
            raise ValueError(
                f'"how" is expected to be either a string or a list of length {n}'
            )

        if suffixes is None:
            suffixes = [""] * (n + 1)
        elif isinstance(suffixes, (list, tuple)) and len(suffixes) != n + 1:
            raise ValueError(f'"suffixes" is expected to be a list of length {n+1}')

        for i, df_i in enumerate(dfs[1:]):
            df = df.join(
                df_i, on=on[i], how=how[i], lsuffix=suffixes[i], rsuffix=suffixes[i + 1]
            )
        return df


class Mi:
    def __init__(self, df):
        self.df = df

    def __repr__(self):
        return 'Row indexer'

    def __getitem__(self, args):
        return self.df.loc[args, :]

    def __setitem__(self, k, v):
        self.df.loc[k, :] = v

    def __call__(self, *args, **kwargs):
        levels, keys = tuple(kwargs.keys()), tuple(kwargs.values())
        return self.df.xs(keys, level=levels, drop_level=False)


@property
def get_mi(self):
    return Mi(self)


class Co:
    def __init__(self, df):
        self.df = df

    def __repr__(self):
        return 'Column indexer'

    def __getitem__(self, args):
        return self.df.loc[:, args]

    def __setitem__(self, k, v):
        self.df.loc[:, k] = v

    def __call__(self, *args, **kwargs):
        levels, keys = tuple(kwargs.keys()), tuple(kwargs.values())
        return self.df.xs(keys, level=levels, drop_level=False, axis=1)


@property
def get_co(self):
    return Co(self)


def patch_dataframe():
    pd.DataFrame.mi = get_mi
    pd.DataFrame.co = get_co


def from_dict(d):
    """
    dict with a single item => Index from this item
    dict of lists => MultiIndex from values
    dict of tuples => MultiIndex from product
    """
    if not isinstance(d, dict):
        raise TypeError('Argument of `from_dict` must be a dict.')
    if len(d) == 1:
        k, v = next(iter(d.items()))
        return pd.Index(v, name=k)
    elif d and isinstance(next(iter(d.values())), tuple):
        return pd.MultiIndex.from_product(list(d.values()), names=d.keys())
    else:
        return pd.MultiIndex.from_tuples(zip(*d.values()), names=d.keys())


def swap_levels(obj: NDFrameT, i: Axis = -2, j: Axis = -1, axis: Axis = 0, sort=True) -> NDFrameT:
    df1 = obj.swaplevel(i, j, axis=axis)
    if sort:
        df1.sort_index(axis=axis, inplace=True)
    return df1

def get_level(mi, level):
    """
    Returns a complete level of a MultiIndex (alias to .get_level_values)
    """

    return mi.get_level_values(level=level)

def set_level(mi, level, labels, name=lib.no_default, inplace=True):
    """
    Replaces a complete level of a MultiIndex

    level :
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
          - keeps name of the `level` if `level` is an Index.

    inplace : 
        Return a fresh copy or modify inplace. If mi is a simple pd.Index 
        (=not a MultiIndex) always returns a copy because the Index is immutable.

    verify_integrity :
        Performs some sanity checks
    """
    if not isinstance(mi, pd.MultiIndex):
        raise TypeError(f"The first argument must be a MultiIndex, not {type(mi)}.")
    
    level_num = mi._get_level_number(level)
    
    if isinstance(labels, pd.Index):
        index = labels
        if name is not lib.no_default:
            index.name = name
    else:
        n = len(mi)
        if is_scalar(labels):
            labels = [labels] * n
        elif n == len(labels):
            pass
        elif n % len(labels) == 0:
            labels = list(labels) * (n // len(labels))
        else:
            raise ValueError(f"len(labels)={len(labels)}; must be 1, {n} or a proper divisor of {n}")
        index = pd.Index(labels)
        if name is not lib.no_default:
            index.name = name
        else:
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
