from math import inf

import pytest

import numpy as np
import pandas as pd

import pdi
from pdi import lock_order, from_product, from_dict
from pdi.categoricals import _get_categories
from pdi.testing import range2d, gen_df


def vi(s):
    return s.values.tolist(), s.index.to_list()


def vic(s):
    return s.values.tolist(), s.index.to_list(), s.columns.to_list()

def vicn(s):
    return (
        s.fillna(inf).values.tolist(),
        s.index.to_list(),
        s.columns.to_list(),
        [list(s.index.names), list(s.columns.names)],
    )

def is_categorical(index, level):
    return isinstance(index.get_level_values(level), pd.CategoricalIndex)

def get_categories(index, level):
    if index.nlevels == 1:
        return index.categories.tolist()
    else:
        return index.levels[level].categories.tolist()


def check_all_categorical(index):
    for i in range(index.nlevels):
        assert is_categorical(index, i)


def check_none_categorical(index):
    for i in range(index.nlevels):
        assert not is_categorical(index, i)


def check_same_labels(df, df0):
    assert df.index.values.tolist() == df0.index.values.tolist()
    assert df.index.values.tolist() == df0.index.values.tolist()


CAT = {
    'index':
    [
        ['b', 'a'],
        ['d', 'c'],
        ['f', 'e'],
    ], 
    'columns':
    [
        ['B', 'A'],
        ['D', 'C'],
        ['F', 'E'],
    ]
}

DF = []
DF_AXIS = []
for i in range(1, 4):
    for j in range(i, 4):
        df = gen_df(i, j, False)
        DF.append(df)
        DF_AXIS.append((df, 'columns'))

OTHER = {"index": "columns", "columns": "index"}

@pytest.mark.parametrize("df0, axis", DF_AXIS)
def test_per_level_inplace(df0, axis):
    index0 = getattr(df0, axis)
    orig = vicn(df0)

    if index0.nlevels == 1:
        df = df0.copy()
        lock_order(df, axis=axis, inplace=True)
        mi = getattr(df, axis)
        assert isinstance(mi, pd.CategoricalIndex)
        assert mi.categories.tolist() == CAT[axis][0]
        assert vicn(df) == orig
        return

    # per level by position
    for i in range(index0.nlevels):
        df = df0.copy()
        lock_order(df, level=i, axis=axis, inplace=True)
        mi = getattr(df, axis)
        for j in range(mi.nlevels):
            if j == i:
                assert is_categorical(mi, j), (i, j)
                assert mi.levels[i].categories.tolist() == CAT[axis][j]
            else:
                assert not is_categorical(mi, j), (i, j)
        check_none_categorical(getattr(df, OTHER[axis]))
        assert vicn(df) == orig

    # per level by name
    for i, name in enumerate(index0.names):
        df = df0.copy()
        lock_order(df, level=name, axis=axis, inplace=True)
        mi = getattr(df, axis)
        for j in range(mi.nlevels):
            if j == i:
                assert is_categorical(mi, j), (i, j)
                assert mi.levels[i].categories.tolist() == CAT[axis][j]
            else:
                assert not is_categorical(mi, j), (i, j)
        check_none_categorical(getattr(df, OTHER[axis]))
        assert vicn(df) == orig

@pytest.mark.parametrize("df0, axis", DF_AXIS)
def test_per_level_not_inplace(df0, axis):
    OTHER = {"index": "columns", "columns": "index"}
    index0 = getattr(df0, axis)
    orig = vicn(df0)

    if index0.nlevels == 1:
        df = df0.copy()
        df1 = lock_order(df, axis=axis, inplace=False)
        mi = getattr(df1, axis)
        assert isinstance(mi, pd.CategoricalIndex)
        assert mi.categories.tolist() == CAT[axis][0]
        assert vicn(df1) == orig
        assert vicn(df) == orig
        check_none_categorical(getattr(df, OTHER[axis]))
        check_none_categorical(df.index)
        check_none_categorical(df.columns)
        return

    # per level by position
    for i in range(index0.nlevels):
        df = df0.copy()
        df1 = lock_order(df, level=i, axis=axis, inplace=False)
        mi = getattr(df1, axis)
        for j in range(mi.nlevels):
            if j == i:
                assert is_categorical(mi, j), i
                assert get_categories(mi, j) == CAT[axis][j], i
            else:
                assert not is_categorical(mi, j), (i, j)
        check_none_categorical(getattr(df1, OTHER[axis]))
        assert vicn(df1) == orig
        assert vicn(df) == orig
        check_none_categorical(df.index)
        check_none_categorical(df.columns)

    # per level by name
    for i, name in enumerate(index0.names):
        df = df0.copy()
        df1 = lock_order(df, level=name, axis=axis, inplace=False)
        mi = getattr(df1, axis)
        for j in range(mi.nlevels):
            if j == i:
                assert is_categorical(mi, j), j
                assert get_categories(mi, j) == CAT[axis][j], i
            else:
                assert not is_categorical(mi, j), (i, j)
        check_none_categorical(getattr(df1, OTHER[axis]))
        assert vicn(df1) == orig
        assert vicn(df) == orig
        check_none_categorical(df.index)
        check_none_categorical(df.columns)

@pytest.mark.parametrize("df0, axis", DF_AXIS)
def test_per_axis_not_inplace(df0, axis):
    orig = vicn(df0)

    df = df0.copy()
    df1 = lock_order(df, axis=axis, inplace=False)
    mi = getattr(df1, axis)
    for i in range(mi.nlevels):
        assert is_categorical(mi, i), i
        assert get_categories(mi, i) == CAT[axis][i], i
    check_none_categorical(getattr(df, OTHER[axis]))
    assert vicn(df1) == orig
    assert vicn(df) == orig
    check_none_categorical(df.index)
    check_none_categorical(df.columns)
    
@pytest.mark.parametrize("df0, axis", DF_AXIS)
def test_per_axis_inplace(df0, axis):
    orig = vicn(df0)

    df = df0.copy()
    lock_order(df, axis=axis, inplace=True)
    mi = getattr(df, axis)
    for i in range(mi.nlevels):
        assert is_categorical(mi, i), i
        assert get_categories(mi, i) == CAT[axis][i], i
    check_none_categorical(getattr(df, OTHER[axis]))
    assert vicn(df) == orig
    
@pytest.mark.parametrize("df0", DF)
def test_both_axes_inplace(df0):
    orig = vicn(df0)

    df = df0.copy()
    lock_order(df, inplace=True)
    for axis in 'index', 'columns':
        mi = getattr(df, axis)
        for i in range(mi.nlevels):
            assert is_categorical(mi, i), i
            assert get_categories(mi, i) == CAT[axis][i], i
    assert vicn(df) == orig
    
@pytest.mark.parametrize("df0", DF)
def test_both_axes_not_inplace(df0):
    orig = vicn(df0)

    df = df0.copy()
    df1 = lock_order(df, inplace=False)
    for axis in 'index', 'columns':
        mi = getattr(df1, axis)
        for i in range(mi.nlevels):
            assert is_categorical(mi, i), i
            assert get_categories(mi, i) == CAT[axis][i], i
        check_none_categorical(getattr(df, axis))
    assert vicn(df1) == orig
    assert vicn(df) == orig


def test_get_categories():
    df = pd.DataFrame(
        np.zeros((8, 2), int),
        index=pdi.from_dict(
            {
                "K": ("b", "a"),
                "L": ("d", "c"),
                "M": ("f", "e"),
            }
        ),
        columns=["A", "B"],
    )
    assert _get_categories(df.index, 0, "index").tolist() == ["b", "a"]
    assert _get_categories(df.index, 1, "index").tolist() == ["d", "c"]
    assert _get_categories(df.index, 2, "index").tolist() == ["f", "e"]


def test_dropped_column():
    df = pd.DataFrame(
        np.arange(1, 9).reshape(2, 4),
        index=list("ab"),
        columns=pd.MultiIndex.from_product([list("AB"), list("CD")]),
    )

    df1 = df.drop(columns=("A", "D"))

    with pytest.raises(ValueError):
        pdi.lock_order(df1)

    df1 = lock_order(df, level=0, axis=1, inplace=False)
    assert isinstance(df1.columns.get_level_values(0), pd.CategoricalIndex)
    assert df1.columns.values.tolist() == df.columns.values.tolist()

    df1 = lock_order(df, axis=0, inplace=False)
    assert isinstance(df1.index, pd.CategoricalIndex)
    assert df1.index.values.tolist() == df.index.values.tolist()


def test_incorrect_sorting():
    df = pd.DataFrame(
        np.arange(1, 9).reshape(4, 2),
        index=pdi.from_dict({"K": list("baab"), "L": list("ddcc")}),
        columns=list("AB"),
    )

    df1 = pdi.lock_order(df, axis=0, level=1, inplace=False)
    assert isinstance(df1.index.get_level_values(1), pd.CategoricalIndex)
    assert df1.columns.values.tolist() == df.columns.values.tolist()
    assert df1.index.values.tolist() == df.index.values.tolist()

    with pytest.raises(ValueError):
        pdi.lock_order(df, axis=0, level=0)


def test_series():
    s = pd.Series(
        np.arange(12), index=pdi.from_dict({"K": tuple("BA"), "L": tuple("PYTHON")})
    )
    s1 = lock_order(s, inplace=False)
    assert isinstance(s1.index.get_level_values(0), pd.CategoricalIndex)
    assert isinstance(s1.index.get_level_values(1), pd.CategoricalIndex)
    assert s1.index.values.tolist() == s.index.values.tolist()


def test_index():
    idx = pd.Index(list("PYTHON"))
    idx1 = pdi.lock_order(idx)
    assert isinstance(idx1, pd.CategoricalIndex)
    assert idx1.values.tolist() == list("PYTHON")
    assert idx1.categories.tolist() == list("PYTHON")
    assert idx1.ordered == True


def test_multiindex():
    idx = pd.MultiIndex.from_product([("B", "A"), ("D", "C")])
    idx1 = pdi.lock_order(idx)
    L1 = idx1.get_level_values(0)
    L2 = idx1.get_level_values(1)

    assert isinstance(L1, pd.CategoricalIndex)
    assert L1.values.tolist() == ["B", "B", "A", "A"]
    assert L1.categories.tolist() == ["B", "A"]
    assert L1.ordered == True

    assert isinstance(L2, pd.CategoricalIndex)
    assert L2.values.tolist() == ["D", "C", "D", "C"]
    assert L2.categories.tolist() == ["D", "C"]
    assert L2.ordered == True


def test_categories():
    idx = pd.MultiIndex.from_product([("B", "A"), tuple("PTHYON")])
    idx1 = pdi.lock_order(idx, level=1, categories=list("PYTHON"))
    s = pd.Series(np.arange(1, 13), index=idx1)
    assert s.index.get_level_values(1).categories.tolist() == list("PYTHON")
    s1 = s.sort_index()
    assert vi(s1) == (
        [7, 10, 8, 9, 11, 12, 1, 4, 2, 3, 5, 6],
        [
            ("A", "P"),
            ("A", "Y"),
            ("A", "T"),
            ("A", "H"),
            ("A", "O"),
            ("A", "N"),
            ("B", "P"),
            ("B", "Y"),
            ("B", "T"),
            ("B", "H"),
            ("B", "O"),
            ("B", "N"),
        ],
    )


def test_from_product():
    mi = from_product([[2022, 2021], list("PYTHON")], names=["K", "L"])
    check_all_categorical(mi)


@pytest.mark.parametrize("axis, res_names, index_cats, columns_cats", [
    (0, ['k✓', 'K'], ['c', 'b', 'a'], None),
    (1, ['k', 'K✓'], None, ['C', 'B', 'A']),
    (None, ['k✓', 'K✓'], ['c', 'b', 'a'], ['C', 'B', 'A']),
])
def test_vis_lock1(axis, res_names, index_cats, columns_cats):
    # * auto categories
    df0 =  pd.DataFrame(
            range2d(3, 3), 
            index=from_dict({'k': ["c", "b", "a"]}), 
            columns=from_dict({'K': ["C", "B", "A"]}),
    )
    orig = vicn(df0)

    #   - inplace=False
    df = df0.copy()
    df1 = lock_order(df, axis=axis, inplace=False)
    df2 = pdi.vis_lock(df1)
    assert [df2.index.name, df2.columns.name] == res_names
    if index_cats is not None:
        assert df2.index.categories.tolist() == index_cats 
    if columns_cats is not None:
        assert df2.columns.categories.tolist() == columns_cats
    assert vicn(df) == orig
    
    #   - inplace=True
    df = df0.copy()
    lock_order(df, axis=axis, inplace=True)
    df2 = pdi.vis_lock(df)
    assert [df2.index.name, df2.columns.name] == res_names
    if index_cats is not None:
        assert df2.index.categories.tolist() == index_cats 
    if columns_cats is not None:
        assert df2.columns.categories.tolist() == columns_cats
    
    # manual categories
    df0 =  pd.DataFrame(
            range2d(3, 3), 
            index=from_dict({'k': ["a", "c", "b"]}), 
            columns=from_dict({'K': ["A", "C", "B"]}),
    )
    orig = vicn(df0)
    
    cats = ['c', 'b', 'a'], ['C', 'B', 'A']
    categories = {None: cats, 0: cats[0], 1: cats[1]}
    
    #   - inplace=False
    df = df0.copy()
    df1 = lock_order(df, axis=axis, categories=categories[axis], inplace=False)
    df2 = pdi.vis_lock(df1)
    assert [df2.index.name, df2.columns.name] == res_names
    if index_cats is not None:
        assert df2.index.categories.tolist() == index_cats 
    if columns_cats is not None:
        assert df2.columns.categories.tolist() == columns_cats 
    assert vicn(df) == orig
    
    #   - inplace=True
    df = df0.copy()
    lock_order(df, axis=axis, categories=categories[axis], inplace=True)
    df2 = pdi.vis_lock(df)
    assert [df2.index.name, df2.columns.name] == res_names
    if index_cats is not None:
        assert df2.index.categories.tolist() == index_cats 
    if columns_cats is not None:
        assert df2.columns.categories.tolist() == columns_cats

C = (['b', 'a'], ['d', 'c'], ['B', 'A'], ['D', 'C'])

@pytest.mark.parametrize("axis, level, names, cats, res_cats", [
    (0, 0, (['k✓', 'l'], ['K', 'L']), C[0], (C[0], None, None, None)),
    (0, 1, (['k', 'l✓'], ['K', 'L']), C[1], (None, C[1], None, None)),
    (1, 0, (['k', 'l'], ['K✓', 'L']), C[2], (None, None, C[2], None)),
    (1, 1, (['k', 'l'], ['K', 'L✓']), C[3], (None, None, None, C[3])),
    (0, None, (['k✓', 'l✓'], ['K', 'L']), C[:2], (C[0], C[1], None, None)),
    (1, None, (['k', 'l'], ['K✓', 'L✓']), C[2:], (None, None, C[2], C[3])),
    (None, None, (['k✓', 'l✓'], ['K✓', 'L✓']), [C[:2], C[2:]], C),
])
def test_vis_lock2(axis, level, names, cats, res_cats):
    # * auto categories
    df0 = pd.DataFrame(
            range2d(4, 4),
            index=pdi.from_dict(
                {
                    "k": ("b", "a"),
                    "l": ("d", "c"),
                }
            ),
            columns=pdi.from_dict(
                {
                    "K": ("B", "A"),
                    "L": ("D", "C"),
                }
            ),
        )  # 2Lx2L
    orig = vicn(df0)

    #   - inplace=False
    df = df0.copy()
    df1 = lock_order(df, axis=axis, level=level, inplace=False)
    df2 = pdi.vis_lock(df1)
    assert df2.index.names == names[0]
    assert df2.columns.names == names[1]
    for i in range(4):
        _axis, _level = divmod(i, 2)
        if res_cats[i] is not None:
            df2._get_axis(_axis).levels[_level].categories.tolist() == res_cats[i]
    assert vicn(df) == orig
    
    #   - inplace=True
    df = df0.copy()
    lock_order(df, axis=axis, level=level, inplace=True)
    df2 = pdi.vis_lock(df)
    assert df2.index.names == names[0]
    assert df2.columns.names == names[1]
    for i in range(4):
        _axis, _level = divmod(i, 2)
        if res_cats[i] is not None:
            df2._get_axis(_axis).levels[_level].categories.tolist() == res_cats[i]
    
    # * manual categories
    df0 = pd.DataFrame(
            range2d(4, 4),
            index=pdi.from_dict(
                {
                    "k": ("a", "b"),
                    "l": ("c", "d"),
                }
            ),
            columns=pdi.from_dict(
                {
                    "K": ("A", "B"),
                    "L": ("C", "D"),
                }
            ),
        )  # 2Lx2L
    orig = vicn(df0)

    #   - inplace=False
    df = df0.copy()
    df1 = lock_order(
            df, 
            axis=axis,
            level=level,
            categories=cats,
            inplace=False,
    )
    df2 = pdi.vis_lock(df1)
    assert df2.index.names == names[0]
    assert df2.columns.names == names[1]
    for i in range(4):
        axis, level = divmod(i, 2)
        if res_cats[i] is not None:
            df2._get_axis(axis).levels[level].categories.tolist() == res_cats[i]
    assert vicn(df) == orig

    #   - inplace=True
    df = df0.copy()
    lock_order(
            df, 
            axis=axis, 
            level=level, 
            categories=cats, 
            inplace=True,
    )
    df2 = pdi.vis_lock(df)
    assert df2.index.names == names[0]
    assert df2.columns.names == names[1]
    for i in range(4):
        axis, level = divmod(i, 2)
        if res_cats[i] is not None:
            df2._get_axis(axis).levels[level].categories.tolist() == res_cats[i]


def test_raises():
    df = gen_df(2, 2)
    with pytest.raises(ValueError):
        lock_order(df, level=1)     # axis is required

    with pytest.raises(ValueError):
        lock_order(df.columns, level=1, inplace=True)  # index is immutable

    with pytest.raises(TypeError):
        lock_order(np.array([1,2,3]))   # numpy array has no index

if __name__ == "__main__":
    pytest.main(["-x", "-s", __file__])# + '::test_vis_lock2'])
#    pytest.main(["-x", "-s", __file__ + '::test_per_axis_inplace'])
#    pytest.main(["-x", "-s", __file__ + '::test_per_axis_not_inplace'])
    #input()
