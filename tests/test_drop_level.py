from math import inf

import pytest
import pandas as pd

# try:
#    from pandas._typing import NDFrameT
# except:  # pandas < 1.4
#    from pandas._typing import FrameOrSeries as NDFrameT
from pandas.core.generic import NDFrame


from pdi import set_level, get_level, drop_level, move_level, swap_levels
from pdi.testing import gen_df


def vn(idx):
    return idx.values.tolist(), list(idx.names)


def vicn(df):
    assert isinstance(df, NDFrame)  # Frame or Series
    return (
        df.fillna(inf).values.tolist(),
        df.index.to_list(),
        df.columns.to_list(),
        [list(df.index.names), list(df.columns.names)],
    )


_DROP_RESULTS = (
    (
        [
            ("C", "E"),
            ("C", "F"),
            ("D", "E"),
            ("D", "F"),
            ("C", "E"),
            ("C", "F"),
            ("D", "E"),
            ("D", "F"),
        ],
        ["L", "M"],
    ),
    (
        [
            ("A", "E"),
            ("A", "F"),
            ("A", "E"),
            ("A", "F"),
            ("B", "E"),
            ("B", "F"),
            ("B", "E"),
            ("B", "F"),
        ],
        ["K", "M"],
    ),
    (
        [
            ("A", "C"),
            ("A", "C"),
            ("A", "D"),
            ("A", "D"),
            ("B", "C"),
            ("B", "C"),
            ("B", "D"),
            ("B", "D"),
        ],
        ["K", "L"],
    ),
)


def test_drop_level_mi():
    df = gen_df(1, 3)
    orig = vn(df.columns)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        df = gen_df(1, 3)
        drop_level(df.columns, i)
        assert vn(df.columns) == _DROP_RESULTS[i]

    df = gen_df(1, 3)
    with pytest.raises(IndexError):
        drop_level(df.columns, 3)

    #   - by name
    for i, name in enumerate("KLM"):
        df = gen_df(1, 3)
        drop_level(df.columns, name)
        assert vn(df.columns) == _DROP_RESULTS[i]

    df = gen_df(1, 3)
    with pytest.raises(KeyError):
        drop_level(df.columns, "Q")

    # * inplace = False
    #   - by positional index
    for i in range(3):
        df = gen_df(1, 3)
        idx = drop_level(df.columns, i, inplace=False)
        assert vn(df.columns) == orig
        assert vn(idx) == _DROP_RESULTS[i]

    df = gen_df(1, 3)
    with pytest.raises(IndexError):
        drop_level(df.columns, 3, inplace=False)

    #   - by name
    for i, name in enumerate("KLM"):
        df = gen_df(1, 3)
        idx = drop_level(df.columns, name, inplace=False)
        assert vn(df.columns) == orig
        assert vn(idx) == _DROP_RESULTS[i]

    df = gen_df(1, 3)
    with pytest.raises(KeyError):
        drop_level(df.columns, "Q", inplace=False)


def test_drop_level_df_col():
    df = gen_df(1, 3)
    orig = vn(df.columns)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        df = gen_df(1, 3)
        drop_level(df, i, axis=1)
        assert vn(df.columns) == _DROP_RESULTS[i]

    df = gen_df(1, 3)
    with pytest.raises(IndexError):
        drop_level(df, 3, axis=1)

    #   - by name
    for i, name in enumerate("KLM"):
        df = gen_df(1, 3)
        drop_level(df, name, axis=1)
        assert vn(df.columns) == _DROP_RESULTS[i]

    df = gen_df(1, 3)
    with pytest.raises(KeyError):
        drop_level(df, "Q", axis=1)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        df = gen_df(1, 3)
        df1 = drop_level(df, i, inplace=False, axis=1)
        assert vn(df.columns) == orig
        assert vn(df1.columns) == _DROP_RESULTS[i]

    df = gen_df(1, 3)
    with pytest.raises(IndexError):
        drop_level(df, 3, inplace=False, axis=1)

    #   - by name
    for i, name in enumerate("KLM"):
        df = gen_df(1, 3)
        df1 = drop_level(df, name, inplace=False, axis=1)
        assert vn(df.columns) == orig
        assert vn(df1.columns) == _DROP_RESULTS[i]

    df = gen_df(1, 3)
    with pytest.raises(KeyError):
        drop_level(df.columns, "Q", inplace=False)


def test_drop_level_df_row():
    df0 = gen_df(1, 3).T
    orig = vn(df0.index)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        df = df0.copy()
        drop_level(df, i, axis=0)
        assert vn(df.index) == _DROP_RESULTS[i]

    df = df0.copy()
    with pytest.raises(IndexError):
        drop_level(df, 3, axis=0)

    #   - by name
    for i, name in enumerate("KLM"):
        df = df0.copy()
        drop_level(df, name, axis=0)
        assert vn(df.index) == _DROP_RESULTS[i]

    df = df0.copy()
    with pytest.raises(KeyError):
        drop_level(df, "Q", axis=0)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        df = df0.copy()
        df1 = drop_level(df, i, inplace=False, axis=0)
        assert vn(df.index) == orig
        assert vn(df1.index) == _DROP_RESULTS[i]

    df = df0.copy()
    with pytest.raises(IndexError):
        drop_level(df, 3, inplace=False, axis=0)

    #   - by name
    for i, name in enumerate("KLM"):
        df = df0.copy()
        df1 = drop_level(df, name, inplace=False, axis=0)
        assert vn(df.index) == orig
        assert vn(df1.index) == _DROP_RESULTS[i]

    df = df0.copy()
    with pytest.raises(KeyError):
        drop_level(df.index, "Q", inplace=False)


def test_drop_level_s():
    s0 = gen_df(1, 3).loc["a"]
    orig = vn(s0.index)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        s = s0.copy()
        drop_level(s, i, axis=0)
        assert vn(s.index) == _DROP_RESULTS[i]

    s = s0.copy()
    with pytest.raises(IndexError):
        drop_level(s, 3, axis=0)

    #   - by name
    for i, name in enumerate("KLM"):
        s = s0.copy()
        drop_level(s, name, axis=0)
        assert vn(s.index) == _DROP_RESULTS[i]

    s = s0.copy()
    with pytest.raises(KeyError):
        drop_level(s, "Q", axis=0)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        s = s0.copy()
        s1 = drop_level(s, i, inplace=False, axis=0)
        assert vn(s.index) == orig
        assert vn(s1.index) == _DROP_RESULTS[i]

    s = s0.copy()
    with pytest.raises(IndexError):
        drop_level(s, 3, inplace=False, axis=0)

    #   - by name
    for i, name in enumerate("KLM"):
        s = s0.copy()
        s1 = drop_level(s, name, inplace=False, axis=0)
        assert vn(s.index) == orig
        assert vn(s1.index) == _DROP_RESULTS[i]

    s = s0.copy()
    with pytest.raises(KeyError):
        drop_level(s.index, "Q", inplace=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
