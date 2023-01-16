from math import inf

import pytest

import numpy as np
import pandas as pd

import pdi
from pdi import lock_order, from_product
from pdi.categoricals import _get_categories


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


def check_all_categorical(index):
    for i in range(index.nlevels):
        assert is_categorical(index, i)


def check_none_categorical(index):
    for i in range(index.nlevels):
        assert not is_categorical(index, i)


def check_same_labels(df, df0):
    assert df.index.values.tolist() == df0.index.values.tolist()
    assert df.index.values.tolist() == df0.index.values.tolist()


def check_per_level(df0, axis="index"):
    other_axis = {"index": "columns", "columns": "index"}
    index0 = getattr(df0, axis)

    if index0.nlevels == 1:
        df = df0.copy()
        lock_order(df, axis=axis)
        assert isinstance(getattr(df, axis), pd.CategoricalIndex)
        check_same_labels(df, df0)
        return

    # per level by position
    for i in range(index0.nlevels):
        df = df0.copy()
        lock_order(df, level=i, axis=axis)
        index = getattr(df, axis)
        for j in range(index.nlevels):
            if j == i:
                assert is_categorical(index, j), (i, j)
            else:
                assert not is_categorical(index, j), (i, j)
        check_none_categorical(getattr(df, other_axis[axis]))
        check_same_labels(df, df0)

    # per level by name
    for i, name in enumerate(index0.names):
        df = df0.copy()
        lock_order(df, level=name, axis=axis)
        index = getattr(df, axis)
        for j in range(index.nlevels):
            if j == i:
                assert is_categorical(index, j), (i, j)
            else:
                assert not is_categorical(index, j), (i, j)
        check_same_labels(df, df0)


def range2d(n, m):
    return np.arange(1, n * m + 1).reshape(n, m)


def test_types():
    with pytest.raises(ValueError):
        pdi.lock_order('hmm')
    

@pytest.mark.parametrize(
    "df0",
    [
        # 1L
        pd.DataFrame(
            range2d(3, 3), index=["c", "b", "a"], columns=["C", "B", "A"]
        ),  # 1Lx1L
        # 2L
        pd.DataFrame(
            range2d(4, 2),
            index=pdi.from_dict(
                {
                    "k": ("b", "a"),
                    "l": ("d", "c"),
                }
            ),
            columns=["B", "A"],
        ),  # 2Lx1L
        pd.DataFrame(
            range2d(2, 4),
            index=["b", "a"],
            columns=pdi.from_dict(
                {
                    "K": ("B", "A"),
                    "L": ("D", "C"),
                }
            ),
        ),  # 1Lx2L
        pd.DataFrame(
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
        ),  # 2Lx2L
        # 3L
        pd.DataFrame(
            range2d(8, 2),
            index=pdi.from_dict(
                {
                    "k": ("b", "a"),
                    "l": ("d", "c"),
                    "m": ("f", "e"),
                }
            ),
            columns=["B", "A"],
        ),  # 3Lx1L
        pd.DataFrame(
            range2d(2, 8),
            index=["b", "a"],
            columns=pdi.from_dict(
                {
                    "K": ("B", "A"),
                    "L": ("D", "C"),
                    "M": ("F", "E"),
                }
            ),
        ),  # 1Lx3L
        pd.DataFrame(
            range2d(8, 4),
            index=pdi.from_dict(
                {
                    "k": ("b", "a"),
                    "l": ("d", "c"),
                    "m": ("f", "e"),
                }
            ),
            columns=pdi.from_dict(
                {
                    "K": ("B", "A"),
                    "L": ("D", "C"),
                }
            ),
        ),  # 3Lx2L
        pd.DataFrame(
            range2d(4, 8),
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
                    "M": ("F", "E"),
                }
            ),
        ),  # 2Lx3L
        pd.DataFrame(
            range2d(8, 8),
            index=pdi.from_dict(
                {
                    "k": ("b", "a"),
                    "l": ("d", "c"),
                    "m": ("f", "e"),
                }
            ),
            columns=pdi.from_dict(
                {
                    "K": ("B", "A"),
                    "L": ("D", "C"),
                    "M": ("F", "E"),
                }
            ),
        ),  # 3Lx3L
    ],
)
def test_lock_order(df0):
    # per level
    check_per_level(df0, "index")
    check_per_level(df0, "columns")

    # axis=0
    df = lock_order(df0, axis=0, inplace=False)
    check_all_categorical(df.index)
    check_none_categorical(df.columns)
    check_same_labels(df, df0)

    # axis=1
    df = lock_order(df0, axis=1, inplace=False)
    check_none_categorical(df.index)
    check_all_categorical(df.columns)
    check_same_labels(df, df0)

    # axis=None
    df = lock_order(df0, inplace=False)
    check_all_categorical(df.index)
    check_all_categorical(df.columns)
    check_same_labels(df, df0)


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


def test_vis_lock():
    df = pd.DataFrame(
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

    df1 = lock_order(df, axis=0, level=0, inplace=False)
    df2 = pdi.vis_lock(df1)
    assert df2.index.names == ['k✓', 'l']
    assert df2.columns.names == ['K', 'L']
    
    df1 = lock_order(df, axis=0, level=1, inplace=False)
    df2 = pdi.vis_lock(df1)
    assert df2.index.names == ['k', 'l✓']
    assert df2.columns.names == ['K', 'L']
    
    df1 = lock_order(df, axis=1, level=0, inplace=False)
    df2 = pdi.vis_lock(df1)
    assert df2.index.names == ['k', 'l']
    assert df2.columns.names == ['K✓', 'L']
    
    df1 = lock_order(df, axis=1, level=1, inplace=False)
    df2 = pdi.vis_lock(df1)
    assert df2.index.names == ['k', 'l']
    assert df2.columns.names == ['K', 'L✓']

def test_swap_levels():
    df = pd.DataFrame(
        range2d(4, 2),
        index=pdi.from_dict(
            {
                "k": ("b", "a"),
                "l": ("d", "c"),
            }
        ),
        columns=["B", "A"],
    )  # 2Lx1L
    
    df1 = pdi.swap_levels(df)
    assert vicn(df1) == \
        ([[7, 8], [3, 4], [5, 6], [1, 2]],
        [('c', 'a'), ('c', 'b'), ('d', 'a'), ('d', 'b')],
        ['B', 'A'],
        [['l', 'k'], [None]])

    df = pd.DataFrame(
            range2d(2, 4),
            index=["b", "a"],
            columns=pdi.from_dict(
                {
                    "K": ("B", "A"),
                    "L": ("D", "C"),
                }
            ),
        )  # 1Lx2L

    df1 = pdi.swap_levels(df, axis=1)
    assert vicn(df1) == \
        ([[4, 2, 3, 1], [8, 6, 7, 5]],
        ['b', 'a'],
        [('C', 'A'), ('C', 'B'), ('D', 'A'), ('D', 'B')],
        [[None], ['L', 'K']])


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
