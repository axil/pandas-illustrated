import pytest

import numpy as np
import pandas as pd

from pdi import drop
from pdi.drop import _drop
from pdi.testing import gen_df, gen_df1, vi, vic


def test_series():
    s = pd.Series(np.arange(1, 7), index=["a1", "a2", "a3", "b1", "b2", "b3"])

    assert vi(drop(s, s > 4)) == ([1, 2, 3, 4], ["a1", "a2", "a3", "b1"])  # noqa: E225
    assert vi(drop(s, index=["a3", "b1"])) == ([1, 2, 5, 6], ["a1", "a2", "b2", "b3"])
    assert vi(drop(s, index_like="1")) == ([2, 3, 5, 6], ["a2", "a3", "b2", "b3"])
    assert vi(drop(s, index_re="[ab][23]")) == ([1, 4], ["a1", "b1"])

    with pytest.raises(ValueError):
        drop(s, np.array([True] * 5 + [None]))


def test_rows():
    df = pd.DataFrame(
        np.arange(1, 13).reshape(2, 6),
        columns=["a1", "a2", "a3", "b1", "b2", "b3"],
        index=["A", "B"],
    ).T

    assert vic(drop(df, df.A > 4)) == (
        [[1, 7], [2, 8], [3, 9], [4, 10]],
        ["a1", "a2", "a3", "b1"],
        ["A", "B"],
    )  # noqa: E225
    assert vic(drop(df, index=["a3", "b1"])) == (
        [[1, 7], [2, 8], [5, 11], [6, 12]],
        ["a1", "a2", "b2", "b3"],
        ["A", "B"],
    )
    assert vic(drop(df, index_like="1")) == (
        [[2, 8], [3, 9], [5, 11], [6, 12]],
        ["a2", "a3", "b2", "b3"],
        ["A", "B"],
    )
    assert vic(drop(df, index_re="[ab][23]")) == (
        [[1, 7], [4, 10]],
        ["a1", "b1"],
        ["A", "B"],
    )


def test_columns():
    df = pd.DataFrame(
        np.arange(1, 13).reshape(2, 6),
        columns=["a1", "a2", "a3", "b1", "b2", "b3"],
        index=["A", "B"],
    )

    assert vic(drop(df, columns=df.loc["B"].isin([10, 11]))) == (
        [[1, 2, 3, 6], [7, 8, 9, 12]],
        ["A", "B"],
        ["a1", "a2", "a3", "b3"],
    )
    assert vic(drop(df, columns=["a3", "b1"])) == (
        [[1, 2, 5, 6], [7, 8, 11, 12]],
        ["A", "B"],
        ["a1", "a2", "b2", "b3"],
    )
    assert vic(drop(df, columns_like="1")) == (
        [[2, 3, 5, 6], [8, 9, 11, 12]],
        ["A", "B"],
        ["a2", "a3", "b2", "b3"],
    )
    assert vic(drop(df, columns_re="[ab][23]")) == (
        [[1, 4], [7, 10]],
        ["A", "B"],
        ["a1", "b1"],
    )


def test_multiindex1_index():
    df = gen_df(2, 1)
    df1 = drop(df, index=("a", "d"))
    assert vic(df1) == (
        [[1, 2], [5, 6], [7, 8]],
        [("a", "c"), ("b", "c"), ("b", "d")],
        ["A", "B"],
    )
    df1 = drop(df, index=[("a", "d"), ("b", "c")])
    assert vic(df1) == ([[1, 2], [7, 8]], [("a", "c"), ("b", "d")], ["A", "B"])


def test_multiindex2_columns():
    df = gen_df(1, 2)
    df1 = drop(df, columns=("A", "D"))
    assert vic(df1) == (
        [[1, 3, 4], [5, 7, 8]],
        ["a", "b"],
        [("A", "C"), ("B", "C"), ("B", "D")],
    )
    df1 = drop(df, columns=[("A", "D"), ("B", "C")])
    assert vic(df1) == ([[1, 4], [5, 8]], ["a", "b"], [("A", "C"), ("B", "D")])


def test_raises():
    df = gen_df(1, 2)
    with pytest.raises(TypeError):
        _drop(df)

    with pytest.raises(TypeError):
        _drop(df, items="A", like="B", axis=1)

    with pytest.raises(TypeError):
        drop(df, columns="A", columns_re="B")


def test_missing_axis():
    df = gen_df1(3, 3)

    assert vic(_drop(df, "B")) == (
        [[1, 3], [4, 6], [7, 9]],
        ["a", "b", "c"],
        ["A", "C"],
    )
    assert vic(drop(df, columns="B")) == (
        [[1, 3], [4, 6], [7, 9]],
        ["a", "b", "c"],
        ["A", "C"],
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
