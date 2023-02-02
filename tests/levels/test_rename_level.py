from inspect import isclass

import pytest
import pandas as pd
import numpy as np
from pandas._libs import lib

from pdi import rename_level
from pdi.testing import gen_df, vn, vin, vicn, range2d, gen_df1


@pytest.mark.parametrize(
    "df0,mapping,axis,result",
    [
        (
            gen_df(1, 1),
            {"K": "L"},
            None,
            ([[1, 2], [3, 4]], ["a", "b"], ["A", "B"], [["k"], ["L"]]),
        ),
        (
            gen_df(1, 1),
            "L",
            None,
            ([[1, 2], [3, 4]], ["a", "b"], ["A", "B"], [["k"], ["L"]]),
        ),
        (
            gen_df(1, 1),
            {"k": "l"},
            0,
            ([[1, 2], [3, 4]], ["a", "b"], ["A", "B"], [["l"], ["K"]]),
        ),
        (
            gen_df(1, 1),
            "l",
            0,
            ([[1, 2], [3, 4]], ["a", "b"], ["A", "B"], [["l"], ["K"]]),
        ),
        (
            gen_df(1, 2),
            {"K": "M"},
            None,
            (
                [[1, 2, 3, 4], [5, 6, 7, 8]],
                ["a", "b"],
                [("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")],
                [["k"], ["M", "L"]],
            ),
        ),
        (
            gen_df(2, 1),
            {"k": "m"},
            0,
            (
                [[1, 2], [3, 4], [5, 6], [7, 8]],
                [("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")],
                ["A", "B"],
                [["m", "l"], ["K"]],
            ),
        ),
        (gen_df(2, 1), {"K": "M"}, 0, KeyError),
    ],
)
def test1(df0, mapping, axis, result):
    if isclass(result) and issubclass(result, Exception):
        with pytest.raises(result):
            rename_level(df0, mapping, axis=axis)
    else:
        orig = vicn(df0)

        df = df0.copy()
        df1 = rename_level(df, mapping, axis=axis)
        assert vicn(df1) == result
        assert vicn(df) == orig

        df = df0.copy()
        rename_level(df, mapping, axis=axis, inplace=True)
        assert vicn(df) == result


def test2():
    df = pd.DataFrame(range2d(2, 2), index=[["a", "b"]], columns=["A", "B"])
    assert vicn(rename_level(df, "K", axis=0)) == (
        [[1, 2], [3, 4]],
        [("a",), ("b",)],
        ["A", "B"],
        [["K"], [None]],
    )


def test3():
    df = gen_df(2, 2)
    df1 = rename_level(df, "a", level_id=0, axis=0)
    assert list(df1.index.names) == ["a", "l"]

    with pytest.raises(TypeError):
        rename_level(df, ["a", "b"], level_id=0, axis=0)


def test_wrong_types():
    with pytest.raises(TypeError):
        rename_level(np.array([1, 2, 3]), {"a": "b"})

    df = gen_df(2, 2)
    with pytest.raises(TypeError):
        rename_level(df, ["a", "b"])

    df = gen_df1(2, 2)
    with pytest.raises(ValueError):
        rename_level(df.index, {"x": "y"}, inplace=True)

    df = gen_df1(2, 2)
    with pytest.raises(TypeError):
        rename_level(df.index, ["a", "b"])


def test4():
    df = gen_df1(2, 2)
    assert vicn(rename_level(df, {"t": "y"})) == (
        [[1, 2], [3, 4]],
        ["a", "b"],
        ["A", "B"],
        [[None], [None]],
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
    # pytest.main(["-s", __file__ + '::test0'])
