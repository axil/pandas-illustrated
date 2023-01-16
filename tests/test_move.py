import pytest
import pandas as pd
from pdi import move


def vi(s):
    return s.values.tolist(), s.index.to_list()


def vic(s):
    return s.values.tolist(), s.index.to_list(), s.columns.to_list()


def test_columns():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=list("ABC"), index=list("ab"))
    df

    df1 = move(df, 0, "C")
    assert vic(df1) == ([[3, 1, 2], [6, 4, 5]], ["a", "b"], ["C", "A", "B"])

    df1 = move(df, 1, "C")
    assert vic(df1) == ([[1, 3, 2], [4, 6, 5]], ["a", "b"], ["A", "C", "B"])

    df1 = move(df, 3, "A")
    assert vic(df1) == ([[2, 3, 1], [5, 6, 4]], ["a", "b"], ["B", "C", "A"])


def test_columns1():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=list("ABC"), index=list("ab"))
    df

    df1 = move(df, 0, column="C")
    assert vic(df1) == ([[3, 1, 2], [6, 4, 5]], ["a", "b"], ["C", "A", "B"])

    df1 = move(df, 1, column="C")
    assert vic(df1) == ([[1, 3, 2], [4, 6, 5]], ["a", "b"], ["A", "C", "B"])

    df1 = move(df, 3, column="A")
    assert vic(df1) == ([[2, 3, 1], [5, 6, 4]], ["a", "b"], ["B", "C", "A"])


def test_rows():
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=list("AB"), index=list("abc"))
    df

    df1 = move(df, 0, "c", axis=0)
    assert vic(df1) == ([[5, 6], [1, 2], [3, 4]], ["c", "a", "b"], ["A", "B"])

    df1 = move(df, 1, "c", axis=0)
    assert vic(df1) == ([[1, 2], [5, 6], [3, 4]], ["a", "c", "b"], ["A", "B"])

    df1 = move(df, 3, "a", axis=0)
    assert vic(df1) == ([[3, 4], [5, 6], [1, 2]], ["b", "c", "a"], ["A", "B"])


def test_rows1():
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=list("AB"), index=list("abc"))
    df

    df1 = move(df, 0, index="c")
    assert vic(df1) == ([[5, 6], [1, 2], [3, 4]], ["c", "a", "b"], ["A", "B"])

    df1 = move(df, 1, index="c")
    assert vic(df1) == ([[1, 2], [5, 6], [3, 4]], ["a", "c", "b"], ["A", "B"])

    df1 = move(df, 3, index="a")
    assert vic(df1) == ([[3, 4], [5, 6], [1, 2]], ["b", "c", "a"], ["A", "B"])


def test_reset_index():
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=list("AB"))
    df

    df1 = move(df, 0, 2, axis=0, reset_index=True)
    assert vic(df1) == ([[5, 6], [1, 2], [3, 4]], [0, 1, 2], ["A", "B"])

    df1 = move(df, 1, 2, axis=0, reset_index=True)
    assert vic(df1) == ([[1, 2], [5, 6], [3, 4]], [0, 1, 2], ["A", "B"])

    df1 = move(df, 3, 0, axis=0, reset_index=True)
    assert vic(df1) == ([[3, 4], [5, 6], [1, 2]], [0, 1, 2], ["A", "B"])


def test_series():
    s = pd.Series([1, 2, 3], index=list("abc"))

    s1 = move(s, 0, "c")
    assert vi(s1) == ([3, 1, 2], ["c", "a", "b"])

    s1 = move(s, 1, "c")
    assert vi(s1) == ([1, 3, 2], ["a", "c", "b"])

    s1 = move(s, 3, "a")
    assert vi(s1) == ([2, 3, 1], ["b", "c", "a"])


def test_series1():
    s = pd.Series([1, 2, 3], index=list("abc"))

    s1 = move(s, 0, index="c")
    assert vi(s1) == ([3, 1, 2], ["c", "a", "b"])

    s1 = move(s, 1, index="c")
    assert vi(s1) == ([1, 3, 2], ["a", "c", "b"])

    s1 = move(s, 3, index="a")
    assert vi(s1) == ([2, 3, 1], ["b", "c", "a"])


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
