import numpy as np
import pandas as pd


def show_mi(mi):
    for i in range(mi.nlevels):
        print(mi.get_level_values(i))


def show_df(df):
    show_mi(df.index)
    show_mi(df.columns)


def _is_monotonic_increasing(a):
    return np.all(a[1:]>=a[:-1])


def _get_categories(mi, level):
    if level==0:
        index = mi.get_level_values(level)
        codes, uniques = pd.factorize(index.values)
        if _is_monotonic_increasing(codes):
            categories = uniques
        else:
            raise ValueError(f'Non-monotonic labels in level={level}. '
                             'Use lock_order(..., categories=[list of categories]).')
    else:
        try:
            fr = mi.to_frame(index=False)
            seq = None
            cumcol = [fr.iloc[:, i] for i in range(level)]
            for k, v in fr.groupby(cumcol, sort=False):
                if seq is None:
                    seq = v.iloc[:, level]
                else:
                    if not np.all(seq.values==v.iloc[:, level].values):
                        raise ValueError(f'Missing labels in level {level}. '
                                         'Use lock_order(..., categories=[list of categories]).')
            categories = seq.unique()
        except:
            from IPython.core.debugger import Pdb; Pdb().set_trace()
    return categories


def lock_order(df, level=None, axis=None, categories=None, inplace=True):
    if inplace is False:
        df = df.copy()
    if axis in (0, 'index'):
        mis = {'index': df.index}
    elif axis in (1, 'columns'):
        mis = {'columns': df.columns}
    elif axis is None:
        mis = {'index': df.index, 'columns': df.columns}
    else:
        raise ValueError('"axis" must be one of [0, 1, "index", "columns"]')
     
    for ax, mi in mis.items():
        if mi.nlevels == 1:
            new_mi = pd.CategoricalIndex(mi, mi, ordered=True)
        elif level is None:
            if categories is not None and len(categories) != mi.nlevels:
                raise ValueError(f'For level=None "categories" must be of the same len as MultiIndex ({mi.nlevels})')
            indices = []
            for level in range(mi.nlevels):
                if categories is not None:
                    _categories = categories[level]
                else:
                    _categories = _get_categories(mi, level)
                cur_index = mi.get_level_values(level)
                indices.append(pd.CategoricalIndex(cur_index, _categories, ordered=True))
            new_mi = pd.MultiIndex.from_arrays(indices)
        else:
            if categories is None:
                categories = _get_categories(mi, level)
            indices = [mi.get_level_values(i) for i in range(mi.nlevels)]
            indices[level] = pd.CategoricalIndex(indices[level], categories, ordered=True)
            new_mi = pd.MultiIndex.from_arrays(indices)
    #    return df.reindex(index=new_mi, copy=False)     # it'd be better to do it in-place
        setattr(df, ax, new_mi)
    if inplace is False:
        return df


def vis_lock(df, checkmark='✓'):
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

    df1 = df.copy()  # would be faster to avoid this copy
    for mi in df1.index, df1.columns:
        if mi.nlevels == 1:
            if isinstance(mi, pd.CategoricalIndex):
                _mark_i(mi)
        else:
            for i in range(mi.nlevels):
                if isinstance(mi.get_level_values(i), pd.CategoricalIndex):
                    _mark_mi(mi, i)
    return df1
