import pytest
import numpy as np
from numpy import inf
import pandas as pd
from pandas._libs import lib

from pdi import insert, append, locked, lock
from pdi.testing import range2d, gen_df1, vi, vic, vin, vicn, gen_df


@pytest.mark.parametrize("row", [
    [1, 2, 3],
    np.array([1, 2, 3]),
    pd.Series({"A": 1, "B": 2, "C": 3}),
])
def test_df_numeric(row):
    # inserting to a dataframe with a numeric index
    df = pd.DataFrame(
        [
            [4, 5, 6],
            [7, 8, 9],
        ],
        columns=["A", "B", "C"],
    )

    df1 = insert(df, 0, row)
    assert vi(df1) == (
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        [2, 0, 1],
    )

    df1 = insert(df, 1, row)
    assert vi(df1) == (
        [
            [4, 5, 6],
            [1, 2, 3],
            [7, 8, 9],
        ],
        [0, 2, 1],
    )

    df1 = insert(df, 2, row)
    assert vi(df1) == (
        [
            [4, 5, 6],
            [7, 8, 9],
            [1, 2, 3],
        ],
        [0, 1, 2],
    )

    df1 = insert(df, 0, row, ignore_index=True)
    assert vi(df1) == (
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        [0, 1, 2],
    )

    df1 = insert(df, 1, row, ignore_index=True)
    assert vi(df1) == (
        [
            [4, 5, 6],
            [1, 2, 3],
            [7, 8, 9],
        ],
        [0, 1, 2],
    )

    df1 = insert(df, 2, row, ignore_index=True)
    assert vi(df1) == (
        [
            [4, 5, 6],
            [7, 8, 9],
            [1, 2, 3],
        ],
        [0, 1, 2],
    )


def test_df_numeric_named_series():
    # inserting a named series to a dataframe with a numeric index
    df = pd.DataFrame(
        [
            [4, 5, 6],
            [7, 8, 9],
        ],
        columns=["A", "B", "C"],
    )
    row = pd.Series([1,2,3], index=list('ABC'), name=5)

    df1 = insert(df, 0, row)
    assert vi(df1) == ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [5, 0, 1])

    df1 = insert(df, 1, row)
    assert vi(df1) == ( [ [4, 5, 6], [1, 2, 3], [7, 8, 9]], [0, 5, 1])

    df1 = insert(df, 2, row)
    assert vi(df1) == ( [ [4, 5, 6], [7, 8, 9], [1, 2, 3]], [0, 1, 5])

@pytest.mark.parametrize("row", [
    [1, 2, 3],
    np.array([1, 2, 3]),
    pd.Series({"A": 1, "B": 2, "C": 3}),
    pd.DataFrame([[1, 2, 3]], columns=["A", "B", "C"]),
])
def test_df_non_numeric(row):
    # inserting to a dataframe with a non-numeric index
    df = pd.DataFrame(
        [
            [4, 5, 6],
            [7, 8, 9],
        ],
        index=["b", "c"],
        columns=["A", "B", "C"],
    )

    df1 = insert(df, 0, row, label="a")
    assert vi(df1) == (
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        ['a', 'b', 'c'],
    )

    df1 = insert(df, 1, row, label="a")
    assert vi(df1) == (
        [
            [4, 5, 6],
            [1, 2, 3],
            [7, 8, 9],
        ],
        ['b', 'a', 'c'],
    )

    df1 = insert(df, 2, row, label="a")
    assert vi(df1) == (
        [
            [4, 5, 6],
            [7, 8, 9],
            [1, 2, 3],
        ],
        ['b', 'c', 'a'],
    )

    df1 = insert(df, 0, row, ignore_index=True)
    assert vi(df1) == (
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        [0, 1, 2],
    )

    df1 = insert(df, 1, row, ignore_index=True)
    assert vi(df1) == (
        [
            [4, 5, 6],
            [1, 2, 3],
            [7, 8, 9],
        ],
        [0, 1, 2],
    )

    df1 = insert(df, 2, row, ignore_index=True)
    assert vi(df1) == (
        [
            [4, 5, 6],
            [7, 8, 9],
            [1, 2, 3],
        ],
        [0, 1, 2],
    )


@pytest.mark.parametrize("row, label", [
    (pd.Series([1,2,3], index=list('ABC'), name="a"), lib.no_default),
    (pd.Series([1,2,3], index=list('ABC'), name="z"), "a"),
])
def test_df_non_numeric_named_series(row, label):
    # inserting a named series to a dataframe with a non-numeric index
    df = pd.DataFrame(
        [
            [4, 5, 6],
            [7, 8, 9],
        ],
        index=["b", "c"],
        columns=["A", "B", "C"],
    )
    df1 = insert(df, 0, row, label=label)
    assert vi(df1) == ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ['a', 'b', 'c'])

    df1 = insert(df, 1, row, label=label)
    assert vi(df1) == ( [ [4, 5, 6], [1, 2, 3], [7, 8, 9]], ['b', 'a', 'c'])

    df1 = insert(df, 2, row, label=label)
    assert vi(df1) == ( [ [4, 5, 6], [7, 8, 9], [1, 2, 3]], ['b', 'c', 'a'])


def test3():
    # testing exceptions
    df = pd.DataFrame([[4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"])
    with pytest.raises(IndexError) as exinfo:
        insert(df, -1, [1, 2, 3])
    assert str(exinfo.value) == "Must verify 0 <= pos <= 2"


def test4():
    # explicit row label
    df = pd.DataFrame([[4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"], index=["a", "b"])
    row = [1, 2, 3]

    df1 = insert(df, 0, row, "c")
    assert df1.values.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert df1.index.tolist() == ["c", "a", "b"]

    df1 = insert(df, 1, row, "c")
    assert df1.values.tolist() == [[4, 5, 6], [1, 2, 3], [7, 8, 9]]
    assert df1.index.tolist() == ["a", "c", "b"]

    df1 = insert(df, 2, row, "c")
    assert df1.values.tolist() == [[4, 5, 6], [7, 8, 9], [1, 2, 3]]
    assert df1.index.tolist() == ["a", "b", "c"]


def test5():
    # heterogenous dtypes
    df = pd.DataFrame(
        [[4, 5.0, "6"], [7, 8.0, "9"]], columns=["A", "B", "C"], index=["a", "b"]
    )
    row = [1, 2.0, "3"]

    df1 = insert(df, 0, row, "c")
    assert df1.values.tolist() == [[1, 2.0, "3"], [4, 5.0, "6"], [7, 8.0, "9"]]
    assert df1.index.tolist() == ["c", "a", "b"]

    df1 = insert(df, 1, row, "c")
    assert df1.values.tolist() == [[4, 5.0, "6"], [1, 2.0, "3"], [7, 8.0, "9"]]
    assert df1.index.tolist() == ["a", "c", "b"]

    df1 = insert(df, 2, row, "c")
    assert df1.values.tolist() == [[4, 5.0, "6"], [7, 8.0, "9"], [1, 2.0, "3"]]
    assert df1.index.tolist() == ["a", "b", "c"]


def test6():
    # MultiIndex
    df = pd.DataFrame(
        [[4, 5, 6], [7, 8, 9]],
        columns=["A", "B", "C"],
        index=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")]),
    )
    row = [1, 2, 3]

    df1 = insert(df, 0, row, ("b", "z"))
    assert df1.index.tolist() == [("b", "z"), ("a", "x"), ("a", "y")]

    df1 = insert(df, 1, row, ("b", "z"))
    assert df1.index.tolist() == [("a", "x"), ("b", "z"), ("a", "y")]

    df1 = insert(df, 2, row, ("b", "z"))
    assert df1.index.tolist() == [("a", "x"), ("a", "y"), ("b", "z")]



def test8():
    # testing alignment
    df = pd.DataFrame([[4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"])
    row = pd.DataFrame([[1, 2, 3]], columns=["B", "C", "D"])

    df1 = insert(df, 0, row, ignore_index=True).fillna(inf)
    assert df1.values.tolist() == [
        [inf, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, inf],
        [7.0, 8.0, 9.0, inf],
    ]

    df1 = insert(df, 1, row, ignore_index=True).fillna(inf)
    assert df1.values.tolist() == [
        [4.0, 5.0, 6.0, inf],
        [inf, 1.0, 2.0, 3.0],
        [7.0, 8.0, 9.0, inf],
    ]

    df1 = insert(df, 2, row, ignore_index=True).fillna(inf)
    assert df1.values.tolist() == [
        [4.0, 5.0, 6.0, inf],
        [7.0, 8.0, 9.0, inf],
        [inf, 1.0, 2.0, 3.0],
    ]


def test9():
    # inserting DataFrame with several rows
    df = pd.DataFrame([[4, 5, 6], 
                       [7, 8, 9],], columns=["A", "B", "C"])
    rows = pd.DataFrame([
        [40, 50, 60], 
        [70, 80, 90],
    ], columns=["A", "B", "C"])

    df1 = insert(df, 0, rows, ignore_index=True)
    assert df1.values.tolist() == [
        [40, 50, 60],
        [70, 80, 90],
        [4, 5, 6],
        [7, 8, 9],
    ]

    df1 = insert(df, 1, rows, ignore_index=True)
    assert df1.values.tolist() == [
        [4, 5, 6],
        [40, 50, 60],
        [70, 80, 90],
        [7, 8, 9],
    ]

    df1 = insert(df, 2, rows, ignore_index=True)
    assert df1.values.tolist() == [
        [4, 5, 6],
        [7, 8, 9],
        [40, 50, 60],
        [70, 80, 90],
    ]


def test_series_scalar_numeric():
    s = pd.Series([20, 30], name='A')

    assert vin(insert(s, 0, 10)) == ([10, 20, 30], [2, 0, 1], 'A')
    assert vin(insert(s, 1, 10)) == ([20, 10, 30], [0, 2, 1], 'A')
    assert vin(insert(s, 2, 10)) == ([20, 30, 10], [0, 1, 2], 'A')

    assert vin(insert(s, 0, 10, ignore_index=True)) == ([10, 20, 30], [0, 1, 2], 'A')
    assert vin(insert(s, 1, 10, ignore_index=True)) == ([20, 10, 30], [0, 1, 2], 'A')
    assert vin(insert(s, 2, 10, ignore_index=True)) == ([20, 30, 10], [0, 1, 2], 'A')


def test_series_scalar_non_numeric():
    s = pd.Series([20, 30], index=["b", "c"], name='B')

    assert vin(insert(s, 0, 10, "a")) == ([10, 20, 30], ["a", "b", "c"], 'B')
    assert vin(insert(s, 1, 10, "a")) == ([20, 10, 30], ["b", "a", "c"], 'B')
    assert vin(insert(s, 2, 10, "a")) == ([20, 30, 10], ["b", "c", "a"], 'B')

    assert vin(insert(s, 0, 10, ignore_index=True)) == ([10, 20, 30], [0, 1, 2], 'B')
    assert vin(insert(s, 1, 10, ignore_index=True)) == ([20, 10, 30], [0, 1, 2], 'B')
    assert vin(insert(s, 2, 10, ignore_index=True)) == ([20, 30, 10], [0, 1, 2], 'B')

    with pytest.raises(ValueError):
        assert vin(insert(s, 0, 10))


@pytest.mark.parametrize("s1", [
    [11, 12],
    [11, 12],
    np.array([11, 12]),
])
def test_series_1D_numeric(s1):
    s = pd.Series([20, 30], name='A')
    
    # automatic index
    assert vin(insert(s, 0, s1)) == ([11, 12, 20, 30], [2, 3, 0, 1], 'A')
    assert vin(insert(s, 1, s1)) == ([20, 11, 12, 30], [0, 2, 3, 1], 'A')
    assert vin(insert(s, 2, s1)) == ([20, 30, 11, 12], [0, 1, 2, 3], 'A')
    
    # ignore_index
    assert vin(insert(s, 0, s1, ignore_index=True)) == ([11, 12, 20, 30], [0, 1, 2, 3], 'A')
    assert vin(insert(s, 1, s1, ignore_index=True)) == ([20, 11, 12, 30], [0, 1, 2, 3], 'A')
    assert vin(insert(s, 2, s1, ignore_index=True)) == ([20, 30, 11, 12], [0, 1, 2, 3], 'A')

@pytest.mark.parametrize("s1, label", [
    ([11, 12], ['x', 'y']),
    (np.array([11, 12]), ['x', 'y']),
    (pd.Series([11, 12], index=["x", "y"], name='B'), lib.no_default),
    (pd.Series([11, 12], index=["v", "u"], name='B'), ['x', 'y']),
])
def test_series_1D_non_numeric(s1, label):
    s = pd.Series([20, 30], index=["b", "c"], name='A')

    assert vin(insert(s, 0, s1, label=label)) == ([11, 12, 20, 30], ["x", "y", "b", "c"], 'A')
    assert vin(insert(s, 1, s1, label=label)) == ([20, 11, 12, 30], ["b", "x", "y", "c"], 'A')
    assert vin(insert(s, 2, s1, label=label)) == ([20, 30, 11, 12], ["b", "c", "x", "y"], 'A')
    
    assert vin(insert(s, 0, s1, ignore_index=True)) == ([11, 12, 20, 30], [0, 1, 2, 3], 'A')
    assert vin(insert(s, 1, s1, ignore_index=True)) == ([20, 11, 12, 30], [0, 1, 2, 3], 'A')
    assert vin(insert(s, 2, s1, ignore_index=True)) == ([20, 30, 11, 12], [0, 1, 2, 3], 'A')


def test_allow_duplicates():
    df = pd.DataFrame([[4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"])

    with pytest.raises(ValueError) as exinfo:
        insert(df, 0, [1, 2, 3], label=0)
    assert "Cannot insert label 0" in str(exinfo.value)

#    df.flags.allows_duplicate_labels = False
#    with pytest.raises(ValueError) as exinfo:
#        insert(df, 0, [1, 2, 3], label=0)
#    assert "Cannot insert label 0" in str(exinfo.value)
#
    df1 = insert(df, 0, [1, 2, 3], label=0, allow_duplicates=True)
    assert vi(df1) == ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [0, 0, 1])
    
    df = pd.DataFrame(
        [
            [4, 5, 6],
            [7, 8, 9],
        ],
        columns=["A", "B", "C"],
    )
    rows = pd.DataFrame([[1, 2, 3]], columns=["A", "B", "C"])
    with pytest.raises(ValueError):
        insert(df, 0, rows)
    with pytest.raises(ValueError):
        insert(df, 0, rows, label=0)
    with pytest.raises(ValueError):
        insert(df.T, 0, rows.T, label=0, axis=1)

    s = pd.Series([20, 30], name='A')
    s1 = pd.Series([11, 12], name='B')
    s2 = insert(s, 0, s1, allow_duplicates=True)
    assert vin(s2) == ([11, 12, 20, 30], [0, 1, 0, 1], 'A')

    df = pd.DataFrame(range2d(3,3))
    with pytest.raises(ValueError):
        insert(df, 0, df[:2])
    with pytest.raises(ValueError):
        insert(df, 0, df.loc[:, :2], axis=1)

def test_label_required():
    s = pd.Series([1, 2, 3], name='A')
    s1 = insert(s, 0, 10)
    assert vin(s1) == ([10, 1, 2, 3], [3, 0, 1, 2], 'A')

    s = pd.Series([1,2,3], index=['a','b','c'], name='B')
    with pytest.raises(ValueError):
        insert(s, 0, 10)
    
    s1 = insert(s, 0, 10, 'd')
    assert vin(s1) == ([10, 1, 2, 3], ['d', 'a', 'b', 'c'], 'B')

def test_ignore_index_with_label():
    s = pd.Series([1,2,3])
    with pytest.raises(ValueError):
        insert(s, 0, 4, label=5, ignore_index=True)


@pytest.mark.parametrize('rows', [
    (range2d(2, 3)*10).tolist(),
    range2d(2, 3)*10,
    pd.DataFrame(range2d(2, 3)*10, columns=list('ABC'), index=[3, 4]),
])
def test_2D_rowwise1(rows):
    # * numeric index
    #   - automatic labels
    df = pd.DataFrame(range2d(3,3), columns=list('ABC'))
    df1 = insert(df, 1, rows, axis=0)
    assert vic(df1) == \
    ([[1, 2, 3], [10, 20, 30], [40, 50, 60], [4, 5, 6], [7, 8, 9]],
    [0, 3, 4, 1, 2],
    ['A', 'B', 'C'])

@pytest.mark.parametrize('rows', [
    (range2d(2, 3)*10).tolist(),
    range2d(2, 3)*10,
    pd.DataFrame(range2d(2, 3)*10, columns=list('ABC'), index=['x', 'y']),
])
def test_2D_rowwise2(rows):
    #   - ignore_index
    df = pd.DataFrame(range2d(3,3), columns=list('ABC'))
    df1 = insert(df, 1, rows, axis=0, ignore_index=True)
    assert vic(df1) == \
    ([[1, 2, 3], [10, 20, 30], [40, 50, 60], [4, 5, 6], [7, 8, 9]],
    [0, 1, 2, 3, 4],
    ['A', 'B', 'C'])

@pytest.mark.parametrize('rows, label', [
    ((range2d(2, 3)*10).tolist(), [5, 6]),
    (range2d(2, 3)*10, [5, 6]),
    (pd.DataFrame(range2d(2, 3)*10, columns=list('ABC')), [5, 6]),
    (pd.DataFrame(range2d(2, 3)*10, columns=list('ABC'), index=[5, 6]), lib.no_default),
])
def test_2D_rowwise3(rows, label):
    #   - manual labels
    df = pd.DataFrame(range2d(3,3), columns=list('ABC'))
    df1 = insert(df, 1, rows, label=[5, 6], axis=0)
    assert vic(df1) == \
    ([[1, 2, 3], [10, 20, 30], [40, 50, 60], [4, 5, 6], [7, 8, 9]],
     [0, 5, 6, 1, 2],
     ['A', 'B', 'C'])

@pytest.mark.parametrize('rows', [
    (range2d(2, 3)*10).tolist(),
    range2d(2, 3)*10,
    pd.DataFrame(range2d(2, 3)*10, columns=list('ABC'), index=['x', 'y']),
])
def test_2D_rowwise4(rows):
    # * non-numeric index
    #   - ignore_index
    df = gen_df1(3, 3)
    df1 = insert(df, 1, rows, axis=0, ignore_index=True)
    assert vic(df1) == \
    ([[1, 2, 3], [10, 20, 30], [40, 50, 60], [4, 5, 6], [7, 8, 9]],
    [0, 1, 2, 3, 4],
    ['A', 'B', 'C'])
    
@pytest.mark.parametrize('rows, label', [
    ((range2d(2, 3)*10).tolist(), ['x', 'y']),
    (range2d(2, 3)*10, ['x', 'y']),
    (pd.DataFrame(range2d(2, 3)*10, columns=list('ABC'), index=['p', 'q']), ['x', 'y']),
    (pd.DataFrame(range2d(2, 3)*10, columns=list('ABC'), index=['x', 'y']), lib.no_default),
])
def test_2D_rowwise5(rows, label):
    #   - manual labels
    df = gen_df1(3, 3)
    df1 = insert(df, 1, rows, axis=0, label=label)
    assert vic(df1) == \
    ([[1, 2, 3], [10, 20, 30], [40, 50, 60], [4, 5, 6], [7, 8, 9]],
    ['a', 'x', 'y', 'b', 'c'],
    ['A', 'B', 'C'])

@pytest.mark.parametrize('rows', [
    (range2d(2, 3)*10).tolist(),
    range2d(2, 3)*10,
])
def test_2D_rowwise6(rows):
    #   - automatic labels
    df = gen_df1(3, 3)
    with pytest.raises(ValueError):
        insert(df, 1, rows, axis=0)
    
def test_rows_size_mismatch():
    df = pd.DataFrame(range2d(3, 3))
    with pytest.raises(ValueError):
        insert(df, 0, [1,1,1], label=['a', 'b'])

    with pytest.raises(ValueError):
        insert(df, 0, [1,1], label='a')

    with pytest.raises(ValueError):
        insert(df, 0, np.zeros((2,3,4), int))
    
    df1 = insert(df, 0, [1,1,1], label=['a'])
    assert vic(df1) == \
        ([[1, 1, 1], [1, 2, 3], [4, 5, 6], [7, 8, 9]], ['a', 0, 1, 2], [0, 1, 2])

    df1 = insert(df, 0, [1,1,1], label='a')
    assert vic(df1) == \
        ([[1, 1, 1], [1, 2, 3], [4, 5, 6], [7, 8, 9]], ['a', 0, 1, 2], [0, 1, 2])
    
    df = gen_df1(3, 3)
    assert vic(insert(df, 0, [])) == \
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ['a', 'b', 'c'], ['A', 'B', 'C'])

    s = pd.Series([1,2,3])
    with pytest.raises(ValueError):
        insert(s, 0, 5, axis=1)

    s = pd.Series([1,2,3])
    with pytest.raises(ValueError):
        insert(s, 0, [4,5,6], [10,11])

def test_wrong_types():
    with pytest.raises(TypeError):
        insert([1,2,3], 0, 1)

    s = pd.Series([1,2,3])
    with pytest.raises(TypeError):
        insert(s, 0, 5, {'a': 4})
    
    s = pd.Series([1,2,3])
    with pytest.raises(TypeError):
        insert(s, 0, {'a': 4})
    
    df = pd.DataFrame([[1,2,3]])
    with pytest.raises(TypeError):
        insert(df, 0, {'a': 4})

    df = gen_df1(3, 3)
    with pytest.raises(TypeError):
        insert(df, 0, [{'q': 10}])

def test_columns():
    df = pd.DataFrame(range2d(2,3), columns=list('ABC'))
    df1 = insert(df, 2, [1,2], 'D', axis=1)
    assert vic(df1) == \
        ([[1, 2, 1, 3], [4, 5, 2, 6]], [0, 1], ['A', 'B', 'D', 'C'])

def test_insert_scalar():
    df = gen_df1(3, 3)
    df1 = insert(df, 2, 0, ignore_index=True)
    assert vic(df1) == \
        ([[1, 2, 3], [4, 5, 6], [0, 0, 0], [7, 8, 9]], [0, 1, 2, 3], ['A', 'B', 'C'])

    df = gen_df1(3, 3)
    df1 = insert(df, 2, 0, label='z')
    assert vic(df1) == \
        ([[1, 2, 3], [4, 5, 6], [0, 0, 0], [7, 8, 9]],
        ['a', 'b', 'z', 'c'],
        ['A', 'B', 'C'])

    df = gen_df1(2,3)
    assert vic(insert(df, 2, 7, axis=1, label='D')) == \
        ([[1, 2, 7, 3], [4, 5, 7, 6]], ['a', 'b'], ['A', 'B', 'D', 'C'])

def test_append():
    df = gen_df1(2,3)
    df1 = append(df, pd.DataFrame([[7,8],[9,10]], columns=['D','E'], index=['a', 'b']), axis=1)
    assert vicn(df1) == \
    ([[1, 2, 3, 7, 8], [4, 5, 6, 9, 10]],
 ['a', 'b'],
 ['A', 'B', 'C', 'D', 'E'],
 [[None], [None]])

CAT_RESULTS = [
    ([[1, 2, 1, 3], [4, 5, 2, 6]], [0, 1], ['C', 'B', 'D', 'A']),
    ([[1, 2, 3, 1], [4, 5, 6, 2]], [0, 1], ['C', 'B', 'A', 'D']),
    ([[1, 1, 2, 3], [2, 4, 5, 6]], [0, 1], ['D', 'C', 'B', 'A']),
    ([[1, 2, 1, 3], [4, 5, 2, 6]], [0, 1], ['C', 'B', 'D', 'A']),
    ([[1, 2, 1, 3], [4, 5, 2, 6]], [0, 1], ['C', 'B', 'D', 'A']),
]

@pytest.mark.parametrize("col", [
    [1, 2],
    np.array([1,2]),
    pd.Series([1, 2]),
    pd.DataFrame([[1], [2]]),
])
def test_categoricals(col):
    df0 = locked(pd.DataFrame(range2d(2,3), columns=list('CBA')))

    for i, order in enumerate(['last', 'first', 'guess', 'strict']):
        df = df0.copy()
        df1 = insert(df, 2, col, 'D', axis=1, order=order)
        assert vic(df1) == CAT_RESULTS[0], i
        df2 = df1.sort_index(axis=1)
        assert vic(df2) == CAT_RESULTS[i+1], i

    df0.sort_index(axis=1, ascending=False, inplace=True)

    df = df0.copy()
    df1 = insert(df, 2, col, 'D', axis=1, order='guess')
    assert vic(df1) == \
        ([[3, 2, 1, 1], [6, 5, 2, 4]], [0, 1], ['A', 'B', 'D', 'C'])
    df2 = df1.sort_index(axis=1)
    assert vic(df2) == \
        ([[1, 2, 3, 1], [4, 5, 6, 2]], [0, 1], ['C', 'B', 'A', 'D'])
    
    with pytest.raises(ValueError):
        df = df0.copy()
        df1 = insert(df, 2, col, 'D', axis=1, order='strict')

def test_mi():
    df = gen_df(2,1)
    df1 = insert(df, 1, [[9,10],[11,12]], label=[('d','e'),('f','g')])
    assert vicn(df1) == \
        ([[1, 2], [9, 10], [11, 12], [3, 4], [5, 6], [7, 8]],
        [('a', 'c'), ('d', 'e'), ('f', 'g'), ('a', 'd'), ('b', 'c'), ('b', 'd')],
        ['A', 'B'],
        [['k', 'l'], ['K']])

def test_unordered():
    df = gen_df1(2,3)
    df.columns = pd.CategoricalIndex(['A', 'B', 'C'], categories=['A', 'B', 'C'], 
            ordered=False, dtype='category')
    df1 = insert(df, 1, [7,8], axis=1, label='D', order='first')
    assert df1.columns.categories.tolist() == ['A', 'B', 'C', 'D']

def test_guess():
    df = gen_df1(2,3)
    lock(df)
    df1 = insert(df, 0, [7, 8], axis=1, label='D', order='guess')
    assert df1.columns.categories.tolist() == ['D', 'A', 'B', 'C']

    df = gen_df1(3,2)
    lock(df)
    df1 = df.sort_index(axis=0, ascending=False)
    df2 = insert(df1, 0, [7, 8], axis=0, label='d', order='guess')
    assert df2.index.categories.tolist() == ['d', 'a', 'b', 'c']

    df = gen_df1(2,2)
    lock(df)
    with pytest.raises(ValueError):
        df1 = insert(df, 0, [7,8], axis=1, label='D', order='zzz')

def test_ignore_columns():
    df = gen_df1(3, 3)
    with pytest.raises(ValueError):
        insert(df, 0, [1,2,3], axis=1, ignore_index=True)
    
if __name__ == "__main__":
    #pytest.main(["-x", "-s", __file__])  # + '::test7'])
    pytest.main(["-x", "-s", __file__ + '::test_categoricals'])
    input()
