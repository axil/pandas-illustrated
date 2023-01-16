import numpy as np
import pandas as pd
from pandas._libs import lib


def show_mi(mi):
    for i in range(mi.nlevels):
        print(mi.get_level_values(i))


def show_df(df):
    show_mi(df.index)
    show_mi(df.columns)


def _is_monotonic_increasing(a):
    return np.all(a[1:] >= a[:-1])


def _get_categories(mi, level, ax):
    if level == 0:
        index = mi.get_level_values(level)
        codes, uniques = pd.factorize(index.values)
        if _is_monotonic_increasing(codes):
            categories = uniques
        else:
            raise ValueError(
                f"Non-monotonic labels in level={level}. "
                "Use lock_order(..., categories=[list of categories])."
            )
    else:
        fr = mi.to_frame(index=False)
        seq = None
        cumcol = [fr.iloc[:, i] for i in range(level)]
        for k, v in fr.groupby(cumcol, sort=False):
            if seq is None:
                seq = v.iloc[:, level]
            else:
                if not np.all(seq.values == v.iloc[:, level].values):
                    raise ValueError(
                        f"Missing label(s) in level {level} of {ax}. "
                        "Use lock_order(..., categories=[list of categories])."
                    )
        categories = seq.unique()
    return categories


def lock_order(obj, level=None, axis=None, categories=None, inplace=True):
    """
    Converts Index or level(s) MultiIndex of DataFrame/Series/Index/MultiIndex to Categoricals.
    Requires either:
        - structure similar to the result of 'from_product', or
        - order is provided explicitly via categories argument

    obj :
        DataFrame, Series, Index or MultiIndex

    level :
        int (single level) or None (all levels)

    axis :
        0/1/'index'/'columns' (single axis) or None (all axes)

    categories :
        provide if the MultiIndex has non-regular structure

    inplace : bool
        return a copy (False) or change in place (True);
        ignored for (multi)index: it is immutable, so
        always return a copy.

    For example:
    1) MultiIndex([('B', 'D'),
                   ('B', 'C'),
                   ('A', 'D'),
                   ('A', 'C')])
    here lock_order works for both level 0 and level 1.

    2) MultiIndex([('B', 'D'),
                   ('B', 'C'),
                   ('A', 'D')])
    here lock_order works for level 0 but not for level 1;
    use `lock_order(level=1, categories=['D','C'])`.

    3) MultiIndex([('B', 'D'),
                   ('A', 'D'),
                   ('B', 'C'),
                   ('A', 'C')])
    here lock_order works for level 1 but not for level 0;
    use `lock_order(level=0, categories=['B','A'])`.
    """

    if isinstance(obj, pd.DataFrame) and level is not None and axis is None:
        raise ValueError(
            'When "level" is specified, "axis" becomes a required argument'
        )

    if inplace is False:
        obj = obj.copy()
    if isinstance(obj, pd.Series):
        mis = {"index": obj.index}
    elif isinstance(obj, pd.Index):
        inplace = False
        mis = {"index": obj}
    elif axis in (0, "index"):
        mis = {"index": obj.index}
    elif axis in (1, "columns"):
        mis = {"columns": obj.columns}
    elif axis is None:
        mis = {"index": obj.index, "columns": obj.columns}
    else:
        raise ValueError('"axis" must be one of [0, 1, "index", "columns"]')

    for ax, mi in mis.items():
        if mi.nlevels == 1:
            new_mi = pd.CategoricalIndex(mi, mi, ordered=True)
        elif level is None:
            if categories is not None and len(categories) != mi.nlevels:
                raise ValueError(
                    f'For level=None "categories" must be of the same len as MultiIndex ({mi.nlevels})'
                )
            indices = []
            for i in range(mi.nlevels):
                if categories is not None:
                    _categories = categories[i]
                else:
                    _categories = _get_categories(mi, i, ax)
                cur_index = mi.get_level_values(i)
                indices.append(
                    pd.CategoricalIndex(cur_index, _categories, ordered=True)
                )
            new_mi = pd.MultiIndex.from_arrays(indices)
        else:
            level_num = mi._get_level_number(level)
            if categories is None:
                categories = _get_categories(mi, level_num, ax)
            indices = [mi.get_level_values(i) for i in range(mi.nlevels)]
            indices[level_num] = pd.CategoricalIndex(
                indices[level_num], categories, ordered=True
            )
            new_mi = pd.MultiIndex.from_arrays(indices)
        #    return df.reindex(index=new_mi, copy=False)     # it's better to do it in-place
        if isinstance(obj, pd.Index):
            obj = new_mi
        else:
            setattr(obj, ax, new_mi)
    if inplace is False:
        return obj


def vis_lock(obj, checkmark="✓"):
    """
    Displays a checkmark next to each Index/MultiIndex level name of a DataFrame, a Series
    or an Index/MultiIndex object in case the level is Categorical
    """

    def _mark_i(mi):
        if mi.name is None:
            mi.name = checkmark
        else:
            mi.name = str(mi.name) + checkmark

    def _mark_mi(mi, level):
        name = mi.names[level]
        if name is None:
            name = checkmark
        else:
            name = str(name) + checkmark
        mi.rename(name, level=level, inplace=True)

    obj1 = obj.copy()  # the proper solution would be to avoid this copy
    if isinstance(obj1, pd.Series):
        mis = [obj1.index]
    elif isinstance(obj1, pd.Index):
        mis = [obj1]
    else:
        mis = [obj1.index, obj1.columns]
    for mi in mis:
        if mi.nlevels == 1:
            if isinstance(mi, pd.CategoricalIndex):
                _mark_i(mi)
        else:
            for i in range(mi.nlevels):
                if isinstance(mi.get_level_values(i), pd.CategoricalIndex):
                    _mark_mi(mi, i)
    return obj1


def from_product(
    iterables, sortorder=None, names=lib.no_default, lock_order=True
) -> pd.MultiIndex:
    mi = pd.MultiIndex.from_product(iterables, sortorder=sortorder, names=names)
    if lock_order is True:
        return globals()["lock_order"](mi)
    else:
        return mi
