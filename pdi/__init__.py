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
    from pandas._typing import FrameOrSeries as NDFrameT
from pandas._libs import lib
from pandas import CategoricalIndex as CatIndex

from .drop import drop
from .visuals import patch_series_repr, unpatch_series_repr, sidebyside, sbs
from .categoricals import locked, lock, from_product, vis_lock, vis, \
                          vis_patch, vis_unpatch
from .levels import get_level, set_level, move_level, insert_level, drop_level, \
                    swap_levels, join_levels, split_level, rename_level

__all__ = [
    "find",
    "findall",
    "insert",
    "drop",
    "move",
    "join",
    "patch_series_repr",
    "unpatch_series_repr",
    "sidebyside",
    "sbs",
    "patch_mi_co",
    "from_dict",
    "from_kw",
    "swap_levels",
    "locked",
    "lock",
    "vis_lock",
    "vis_patch",
    "vis_unpatch",
    "from_product",
    "get_level",
    "set_level",
    "move_level",
    "insert_level",
    "drop_level",
    "swap_levels",
    "join_levels",
    "split_level",
    "rename_level",
]


def find(s, x, pos=False):
    """
    Finds element x in Series s.
    Returns index label (default) or positional index (pos=True).
    Raises ValueError if x is missing from s.
    """
    if isinstance(s, pd.Series):
        pass
    elif isinstance(s, np.ndarray):    
        pos=True
    else:
        raise TypeError(
            f"First argument must be a Series or np.array. Got {type(s)}."
        )

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


def _gen_labels(dst, axis, n, ignore_index):
    index = dst._get_axis(axis)
    if index.is_numeric():
        start = index.max()+1
        labels = np.arange(start, start+n)
    elif ignore_index is True:
        labels = dst.index[:1].tolist() * n
    else:
        raise ValueError(
            "The `label` argument is required for non-numeric index."
        )
    return labels

def _ensure_list(a, n):
    if is_scalar(a):
        a = [a]*n
    elif isinstance(a, tuple):
        a = list(map(list, a))
    elif isinstance(a, list):
        if len(a) == 1:
            a *= n
        elif len(a) != n:
            msg = f"If `label` is a list, it is expected to be of length 1"
            if n != 1:
                msg += f" or {n}"
            msg += f", not {len(a)}"
            f'`label` is expected to be a scalar or a list of length 1 or {n}, '
            raise ValueError(msg)
        if isinstance(a[0], tuple):
            a = list(map(list, zip(*a)))
    else:
        raise TypeError(
            '`label` is expected to be scalar, tuple, or list thereof, '
            f'got {type(a)}.'
        )
    return a

def _build_labels(dst, axis, label, n, ignore_index):
    if label is lib.no_default:
        labels = _gen_labels(dst, axis, n, ignore_index=ignore_index)
    else:
        labels = _ensure_list(label, n)
    return labels

#def _match_cats(dst, dst_index, axis, pos, labels, order) -> (NDFrame, pd.Index):
def _match_cats(dst, dst1, axis, pos, order) -> (NDFrame, pd.Index):
    dst_index = dst._get_axis(axis)
    labels = dst1._get_axis(axis).tolist()
    o = dst_index.ordered
    if not o:
        order = 'last'
    orig_cats = dst_index.categories.tolist()
    if order == 'strict':
        if dst_index.tolist() == orig_cats:
            order = pos
        else:
            raise ValueError(
                f"Order of the {dst._get_axis_name(axis)} labels does not "
                "match the locked order. Can't guess new label(s) position "
                "in the 'categories'."
            )
    if order == 'guess':
        if dst_index.tolist() == orig_cats:
            order = pos
        elif pos == 0:
            order = 'first'
        else:
            order = 'last'
    if order == 'last':
        cats = pd.unique(orig_cats + labels)
    elif order == 'first':
        cats = pd.unique(labels + orig_cats)
    elif isinstance(order, int):
        cats = pd.unique(orig_cats[:order] + labels + orig_cats[order:])
    else:
        raise ValueError(
            f"Unrecognized order: {order}. Must be one of: 'last', 'first',"
            "'guess', 'strict' or an int."
        )
    cats = cats.tolist()
    idx = CatIndex(labels, categories=cats, ordered=o)
    if cats != orig_cats:
        dst = dst.copy()
        if axis == 0:
            dst.index = CatIndex(dst.index, categories=cats, ordered=o)
        else:
            dst.columns = CatIndex(dst.columns, categories=cats, ordered=o)
    if axis == 0:
        dst1.index = idx
    else:
        dst1.columns = idx
    return dst, dst1

def is_categorical(obj, axis):
    return isinstance(obj._get_axis(axis), pd.CategoricalIndex)


def insert(
    dst: NDFrameT,
    pos: int,
    value: Scalar | AnyArrayLike | Sequence,
    label: Hashable = lib.no_default,
    axis=0,
    ignore_index: bool = False,
    order = None,
    allow_duplicates: bool = False,
    inplace=False,
) -> NDFrameT:
    """
    Inserts list, tuple, np.array, DataFrame or Series into a DataFrame.
    Also inserts scalar, list, tuple or Series into a Series.

    dst :
        Series or DataFrame to insert into.
    pos : int
        Insertion index. Must verify 0 <= pos <= len(dst).
    value :
        Row(s) to insert.
    label : scalar or tuple
        Index label (use tuple for a MultiIndex label). 
        If not specified: 
          - if the index is numeric, index.max()+1
          - when inserting Series with a name into a DataFrame, keeps its name 
            as a label.
    ignore_index : bool, default False
        If True, do not use the existing index. The resulting rows will be
        labeled 0, ..., n - 1. This is useful if you are inserting into a
        Series or a DataFrame which index does not have meaningful information.
    allow_duplicates : bool, default False
        Check for duplicates before inserting
    """

    if isinstance(dst, pd.Series) and axis not in (0, 'index', 'rows'):
        raise ValueError(
            f"Series has no axis {axis}."
        )

    if not isinstance(dst, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"First argument must be a DataFrame or a Series. Got {type(dst)}"
        )

    if axis == 1 and ignore_index is True:
        raise ValueError("ignore_index=True is not compatible with axis=1")
    
    if ignore_index is True and label is not lib.no_default:
        raise ValueError(
            "`label` and `ignore_index=True` are mutually exclusive"
        )
        
    n = dst.shape[axis]
    if not 0 <= pos <= n:
        raise IndexError(f"Must verify 0 <= pos <= {n}")

    if isinstance(dst, pd.DataFrame):
        if is_scalar(value):
            if axis == 0:
                dst1 = pd.DataFrame(
                    [[value]*dst.shape[1-axis]], 
                    index=_build_labels(dst, axis, label, 1, ignore_index),
                    columns=dst.columns, 
                )
            else:
                dst1 = pd.DataFrame(
                    [[value]]*dst.shape[1-axis], 
                    index=dst.index,
                    columns=_build_labels(dst, axis, label, 1, ignore_index),
                )

        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                return dst
            elif isinstance(value[0], (list, tuple)):     # 2D case
                k = len(value)
            elif is_scalar(value[0]):     # 1D case
                if axis == 0:
                    value = [value]
                else:
                    value = [[v] for v in value]
                k = 1
            else:
                raise TypeError(
                    'If value is a list, its elements should either be lists or scalars, '
                    f'not {type(value[0])}'
                )
            idx = _build_labels(dst, axis, label, k, ignore_index)
            if axis == 0:
                dst1 = pd.DataFrame(value, index=idx, columns=dst.columns)
            else:
                dst1 = pd.DataFrame(value, index=dst.index, columns=idx)
            
        elif isinstance(value, np.ndarray):
            if value.ndim == 1:
                value = value.reshape(1, -1) if axis==0 else value.reshape(-1, 1)
            elif value.ndim != 2:
                raise ValueError(
                    f"The `value` has wrong number of dimensions: {value.ndim}. "
                    "Expected 1 or 2."
                )
            idx = _build_labels(dst, axis, label, value.shape[axis], ignore_index)
            if axis == 0:
                dst1 = pd.DataFrame(value, index=idx, columns=dst.columns)
            else:
                dst1 = pd.DataFrame(value, index=dst.index, columns=idx)

        elif isinstance(value, pd.DataFrame):
            dst1 = value
            if label is not lib.no_default:
                idx = _ensure_list(label, value.shape[axis])
                if axis == 0:
                    dst1.index = idx
                else:
                    dst1.columns = idx

        elif isinstance(value, pd.Series):
            dst1 = value.to_frame()
            if axis == 0:
                dst1 = dst1.T
            if label is not lib.no_default:
                idx = _ensure_list(label, 1)
            elif value.name is None: 
                idx = _gen_labels(dst, axis, 1, ignore_index)
            else:
                idx = None
            if idx is not None:
                if axis == 0:
                    dst1.index = idx
                else:
                    dst1.columns = idx
        else:
            raise TypeError(
                "Supported value types: list, tuple, ndarray, DataFrame, Series. "
                f"Got {type(value)}"
            )

    else:  # = isinstance(dst, pd.Series):
        if is_scalar(value):
            dst1 = pd.Series(
                [value], 
                index=_build_labels(dst, axis, label, 1, ignore_index)
            )
        elif isinstance(value, (list, tuple, np.ndarray)):
            dst1 = pd.Series(
                value, 
                index=_build_labels(dst, axis, label, len(value), ignore_index)
            )
        elif isinstance(value, pd.Series):
            dst1 = value
            if label is not lib.no_default:
                dst1.index = _ensure_list(label, len(value))
        else:
            raise TypeError(
                "Supported value types: scalar, list, tuple, np.array, Series. "
                "Got {type(value)}"
            )

    if is_categorical(dst, axis) and order is not None:
        dst, dst1 = _match_cats(dst, dst1, axis, pos, order)

    if (
        allow_duplicates is False and \
        ignore_index is not True and \
        len(dst._get_axis(axis).intersection(dst1._get_axis(axis)))
    ):
        if label is not lib.no_default:
            msg = f"Cannot insert label {label!r}, already exists and " \
                  "allow_duplicates is False"
            if axis == 0:
                msg += ", consider ignore_index=True."
            else:
                msg += '.'
        else:
            if axis == 0:
                msg = "Duplicates detected in the index. Consider `ignore_index=True`"
                if label is lib.no_default:
                    msg += " or provide index label(s) manually in the `label` argument."
                else:
                    msg += '.'
            else:
                msg = "Duplicates detected in the index."
                if label is lib.no_default:
                    msg += " Consider providing column label(s) manually in the " \
                           "`label` argument."
        raise ValueError(msg)

    if pos == n:  # just for speed, not really necessary
        res = pd.concat([dst, dst1], axis=axis, ignore_index=ignore_index)
    elif axis == 0:
        res = pd.concat([dst[:pos], dst1, dst[pos:]], axis=axis, ignore_index=ignore_index)
    else:
        res = pd.concat([dst.iloc[:, :pos], dst1, dst.iloc[:, pos:]], axis=axis, ignore_index=ignore_index)

    if isinstance(dst, pd.Series):
        res.name = dst.name
    return res


def append(
    dst: NDFrameT,
    value: Scalar | AnyArrayLike | Sequence,
    label: Hashable = lib.no_default,
    axis=0,
    ignore_index: bool = False,
    order = None,
    allow_duplicates: bool = False,
    inplace=False,
) -> NDFrameT:
    return insert(
        dst,
        dst.shape[axis], 
        value, 
        label=label, 
        axis=axis, 
        ignore_index=ignore_index,
        order=order,
        allow_duplicates=allow_duplicates,
        inplace=inplace
    )

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
    """
    Helps indexing MultiIndex in the rows (read and write access).
    Same policy for keeping and removing the filtered levels as in `.loc`.
    e.g. df.mi[:, 'a', :] returns all rows that have 'a' in the second level:
    
    >>> df
           A  B
    k l m      
    a d g  1  2
    b e h  3  4
    c d i  5  6
    
    >>> df.mi[:, 'a', :]
           A  B
    k l m      
    a d g  1  2
    c d i  5  6

    >>> df.mi[:, 'a', :] = 0
    >>> df
           A  B
    k l m      
    a d g  0  0
    b e h  3  4
    c d i  0  0

    Careful: once the result is created, it becomes a copy, so its changes
    are not propagated to the original dataframe.

    Also you can use df.mi(k='a'). Always keeps all the levels. Not writable.
    If you don't need some levels, you can drop them with pdi.drop_level()
    """
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
    """
    Helps indexing MultiIndex in the colums (read and write access).
    Same policy for keeping and removing the filtered levels as in `.loc`.
    e.g. df.co[:, 'a', :] returns all columns that have 'a' in the second level:
    
    >>> df
    K  A               B            
    L  C       D       C       D    
    M  E   F   E   F   E   F   E   F
    a  1   2   3   4   5   6   7   8
    b  9  10  11  12  13  14  15  16
    
    >>> df.co[:, 'C', :]
    K  A       B    
    L  C       C    
    M  E   F   E   F
    a  1   2   5   6
    b  9  10  13  14

    >>> df.co[:, 'C', :] = 0
    >>> df
    K  A             B           
    L  C      D      C      D    
    M  E  F   E   F  E  F   E   F
    a  0  0   3   4  0  0   7   8
    b  0  0  11  12  0  0  15  16

    Careful: once the result is created, it becomes a copy, so its changes
    are not propagated to the original dataframe.

    Also you can use df.co(K='A'). Always keeps all the levels. Not writable.
    If you don't need some levels, you can drop them with pdi.drop_level()
    """
    return Co(self)


def patch_mi_co():
    pd.DataFrame.mi = get_mi
    pd.DataFrame.co = get_co
    pd.Series.mi = get_mi


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

def from_kw(**args):
    return from_dict(args)

