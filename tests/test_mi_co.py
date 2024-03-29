﻿import pytest
import numpy as np
import pandas as pd

from pdi import patch_mi_co
from pdi.testing import gen_df, vic, vin, vicn
import pdi


def range2d(n, m):
    return np.arange(1, n * m + 1).reshape(n, m)


def test_patch_series():
    patch_mi_co()

    df = pd.DataFrame(
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
    )  # 3Lx3L

    # * square brackets
    #   - rows
    assert df.mi["a", "d", "f"].tolist() == [33, 34, 35, 36, 37, 38, 39, 40]

    assert df.mi[:, "d", "f"].values.tolist() == [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [33, 34, 35, 36, 37, 38, 39, 40],
    ]

    assert df.mi[:, "d", :].index.tolist() == [
        ("b", "d", "f"),
        ("b", "d", "e"),
        ("a", "d", "f"),
        ("a", "d", "e"),
    ]

    #   - columns
    assert df.co["B", "C", "F"].values.tolist() == [3, 11, 19, 27, 35, 43, 51, 59]

    assert df.co[:, "C", "F"].columns.tolist() == [("B", "C", "F"), ("A", "C", "F")]

    assert df.co[:, "C", :].columns.tolist() == [
        ("B", "C", "F"),
        ("B", "C", "E"),
        ("A", "C", "F"),
        ("A", "C", "E"),
    ]

    # * keyword args:
    #   - rows
    assert df.mi(k="a", l="d", m="f").values.tolist() == [
        [33, 34, 35, 36, 37, 38, 39, 40]
    ]

    assert df.mi(l="d", m="f").values.tolist() == [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [33, 34, 35, 36, 37, 38, 39, 40],
    ]

    assert df.mi(l="d").index.tolist() == [
        ("b", "d", "f"),
        ("b", "d", "e"),
        ("a", "d", "f"),
        ("a", "d", "e"),
    ]

    #   - columns
    assert df.co(K="B", L="C", M="F").T.values.tolist() == [
        [3, 11, 19, 27, 35, 43, 51, 59]
    ]

    assert df.co(L="C", M="F").columns.tolist() == [("B", "C", "F"), ("A", "C", "F")]

    assert df.co(L="C").columns.tolist() == [
        ("B", "C", "F"),
        ("B", "C", "E"),
        ("A", "C", "F"),
        ("A", "C", "E"),
    ]

    repr(df.mi) == "Row indexer"
    repr(df.co) == "Column indexer"


def test_assignments():
    df = gen_df(1, 3)
    df.co[:, "C", :] = 0
    assert vic(df) == (
        [[0, 0, 3, 4, 0, 0, 7, 8], [0, 0, 11, 12, 0, 0, 15, 16]],
        ["a", "b"],
        [
            ("A", "C", "E"),
            ("A", "C", "F"),
            ("A", "D", "E"),
            ("A", "D", "F"),
            ("B", "C", "E"),
            ("B", "C", "F"),
            ("B", "D", "E"),
            ("B", "D", "F"),
        ],
    )

    df = gen_df(3, 1)
    df.mi[:, "d", :] = 0
    assert vic(df) == (
        [[1, 2], [3, 4], [0, 0], [0, 0], [9, 10], [11, 12], [0, 0], [0, 0]],
        [
            ("a", "c", "e"),
            ("a", "c", "f"),
            ("a", "d", "e"),
            ("a", "d", "f"),
            ("b", "c", "e"),
            ("b", "c", "f"),
            ("b", "d", "e"),
            ("b", "d", "f"),
        ],
        ["A", "B"],
    )


def test_from_not_dict():
    with pytest.raises(TypeError):
        pdi.from_dict("hmm")


def test_mi_():
    patch_mi_co()
    df = gen_df(3, 1)

    assert vicn(df.mi_["a"]) == (
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        [("c", "e"), ("c", "f"), ("d", "e"), ("d", "f")],
        ["A", "B"],
        [["l", "m"], ["K"]],
    )

    assert vicn(df.mi_["a", "c"]) == (
        [[1, 2], [3, 4]],
        ["e", "f"],
        ["A", "B"],
        [["m"], ["K"]],
    )

    assert isinstance(df.mi_["a", "c", "e"], pd.Series)

    assert vin(df.mi_["a", "c", "e"]) == ([1, 2], ["A", "B"], ("a", "c", "e"))

    assert vicn(df.mi_[:, "c", :]) == (
        [[1, 2], [3, 4], [9, 10], [11, 12]],
        [("a", "e"), ("a", "f"), ("b", "e"), ("b", "f")],
        ["A", "B"],
        [["k", "m"], ["K"]],
    )

    assert vicn(df.mi_[:, "c", "a":"z"]) == (
        [[1, 2], [3, 4], [9, 10], [11, 12]],
        [("a", "e"), ("a", "f"), ("b", "e"), ("b", "f")],
        ["A", "B"],
        [["k", "m"], ["K"]],
    )

    assert vicn(df.mi_[:, "c", "a":"e"]) == (
        [[1, 2], [9, 10]],
        [("a", "e"), ("b", "e")],
        ["A", "B"],
        [["k", "m"], ["K"]],
    )

    assert vicn(df.mi_[:, :, :]) == vicn(df)


def test_co_():
    patch_mi_co()
    df = gen_df(1, 3)

    assert vicn(df.co_["A"]) == (
        [[1, 2, 3, 4], [9, 10, 11, 12]],
        ["a", "b"],
        [("C", "E"), ("C", "F"), ("D", "E"), ("D", "F")],
        [["k"], ["L", "M"]],
    )

    assert vicn(df.co_["A", "C"]) == (
        [[1, 2], [9, 10]],
        ["a", "b"],
        ["E", "F"],
        [["k"], ["M"]],
    )

    assert isinstance(df.co_["A", "C", "E"], pd.Series)

    assert vin(df.co_["A", "C", "E"]) == ([1, 9], ["a", "b"], ("A", "C", "E"))

    assert vicn(df.co_[:, "C", :]) == (
        [[1, 2, 5, 6], [9, 10, 13, 14]],
        ["a", "b"],
        [("A", "E"), ("A", "F"), ("B", "E"), ("B", "F")],
        [["k"], ["K", "M"]],
    )

    assert vicn(df.co_[:, "C", "A":"Z"]) == (
        [[1, 2, 5, 6], [9, 10, 13, 14]],
        ["a", "b"],
        [("A", "E"), ("A", "F"), ("B", "E"), ("B", "F")],
        [["k"], ["K", "M"]],
    )

    assert vicn(df.co_[:, "C", "A":"E"]) == (
        [[1, 5], [9, 13]],
        ["a", "b"],
        [("A", "E"), ("B", "E")],
        [["k"], ["K", "M"]],
    )

    assert vicn(df.co_[:, :, :]) == vicn(df)


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
