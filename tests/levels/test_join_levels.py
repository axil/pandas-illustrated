from math import inf

import pytest
import pandas as pd
import numpy as np

from pandas.core.generic import NDFrame

from pdi import join_levels
from pdi.testing import gen_df, vn, vin, vicn


@pytest.mark.parametrize(
    "name, result",
    [
        (
            None,
            (
                [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
                ["a", "b"],
                [
                    "A_C_E",
                    "A_C_F",
                    "A_D_E",
                    "A_D_F",
                    "B_C_E",
                    "B_C_F",
                    "B_D_E",
                    "B_D_F",
                ],
                [["k"], [None]],
            ),
        ),
        (
            "M",
            (
                [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
                ["a", "b"],
                [
                    "A_C_E",
                    "A_C_F",
                    "A_D_E",
                    "A_D_F",
                    "B_C_E",
                    "B_C_F",
                    "B_D_E",
                    "B_D_F",
                ],
                [["k"], ["M"]],
            ),
        ),
    ],
)
def test_join_levels(name, result):
    df0 = gen_df(1, 3)
    orig = vicn(df0)

    # * inplace=False
    df = df0.copy()
    df1 = join_levels(df, name=name)
    assert vicn(df1) == result
    assert vicn(df) == orig

    # inplace=True
    df = df0.copy()
    join_levels(df, name=name, inplace=True)
    assert vicn(df) == result


def test_wrong_types():
    with pytest.raises(TypeError):
        join_levels(np.array([1, 2, 3]))

    df = gen_df(1, 1)
    with pytest.raises(TypeError):
        join_levels(df)


def test_joining_multiindex_inplace():
    df = gen_df(2, 2)
    with pytest.raises(ValueError):
        join_levels(df.columns, inplace=True)


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
    # pytest.main(["-s", __file__ + '::test0'])
