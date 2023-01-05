from __future__ import annotations

import re

from pandas._typing import (
    ArrayLike,
    Axis,
    CompressionOptions,
    Dtype,
    DtypeArg,
    DtypeObj,
    IndexKeyFunc,
    IndexLabel,
    JSONSerializable,
    Level,
    Manager,
    NDFrameT,
    RandomState,
    Renamer,
    StorageOptions,
    T,
    TimedeltaConvertibleTypes,
    TimestampConvertibleTypes,
    ValueKeyFunc,
)

import pandas.core.common as com

bool_t = bool  # Need alias because NDFrame has def bool:

from pandas.core.dtypes.common import (
    ensure_object,
    ensure_platform_int,
    ensure_str,
    is_bool,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_dict_like,
    is_dtype_equal,
    is_extension_array_dtype,
    is_float,
    is_list_like,
    is_number,
    is_numeric_dtype,
    is_re_compilable,
    is_scalar,
    is_timedelta64_dtype,
    pandas_dtype,
)

# the inverse of "filter"
def _drop(obj: NDFrameT,
        items=None,
        like: str | None = None,
        regex: str | None = None,
        axis=None,
) -> NDFrameT:
    nkw = com.count_not_none(items, like, regex)
    if nkw > 1:
        raise TypeError(
            "Keyword arguments `items`, `like`, or `regex` "
            "are mutually exclusive"
        )

    if axis is None:
        axis = obj._info_axis_name
    labels = obj._get_axis(axis)

    if items is not None:
        name = obj._get_axis_name(axis)
        return obj.reindex(**{name: [r for r in labels if r not in items]})
    
    elif like:
        def f(x) -> bool_t:
            assert like is not None  # needed for mypy
            return like not in ensure_str(x)

        values = labels.map(f)
        return obj.loc(axis=axis)[values]

    elif regex:
        def f(x) -> bool_t:
            return matcher.search(ensure_str(x)) is None

        matcher = re.compile(regex)
        values = labels.map(f)
        return obj.loc(axis=axis)[values]
    else:
        raise TypeError("Must pass either `items`, `like`, or `regex`")

def drop(obj: NDFrameT, 
         index=None, index_like=None, index_re=None, 
         columns=None, columns_like=None, columns_re=None, 
) -> NDFrameT:
    nkw = com.count_not_none(
            index, index_re, index_like,
            columns, columns_re, columns_like, 
    )
    if nkw > 1:
        raise TypeError(
            "Keyword arguments "
            "`index`, `index_re`, `index_like` "
            "`columns`, `columns_re`, and `columns_like` "
            "are mutually exclusive"
        )
    if columns is not None:
        if com.is_bool_indexer(columns):   # can raise ValueError
            return obj.loc(axis=1)[~columns]
        else:
            return _drop(obj, items=columns, axis=1)
    elif index is not None:
        if com.is_bool_indexer(index):     # can raise ValueError
            return obj.loc[~index]
        else:
            return _drop(obj, items=index, axis=0)
    
    elif columns_like is not None:
        return _drop(obj, regex=columns_like, axis=1)
    elif index_like is not None:
        return _drop(obj, regex=index_like, axis=0)
    
    elif columns_re is not None:
        return _drop(obj, regex=columns_re, axis=1)
    elif index_re is not None:
        return _drop(obj, regex=index_re, axis=0)
