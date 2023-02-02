from math import inf

import pytest
import pandas as pd
import numpy as np

from pandas.core.generic import NDFrame

from pdi import set_level, get_level, drop_level, move_level, swap_levels
from pdi.testing import gen_df, vn, vin, vicn


_MOVE_RESULTS = {
    (0, 1): (
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
        ["K", "L", "M"],
    ),
    (0, 2): (
        [
            ("C", "A", "E"),
            ("C", "A", "F"),
            ("D", "A", "E"),
            ("D", "A", "F"),
            ("C", "B", "E"),
            ("C", "B", "F"),
            ("D", "B", "E"),
            ("D", "B", "F"),
        ],
        ["L", "K", "M"],
    ),
    (0, 3): (
        [
            ("C", "E", "A"),
            ("C", "F", "A"),
            ("D", "E", "A"),
            ("D", "F", "A"),
            ("C", "E", "B"),
            ("C", "F", "B"),
            ("D", "E", "B"),
            ("D", "F", "B"),
        ],
        ["L", "M", "K"],
    ),
    (1, 0): (
        [
            ("C", "A", "E"),
            ("C", "A", "F"),
            ("D", "A", "E"),
            ("D", "A", "F"),
            ("C", "B", "E"),
            ("C", "B", "F"),
            ("D", "B", "E"),
            ("D", "B", "F"),
        ],
        ["L", "K", "M"],
    ),
    (1, 2): (
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
        ["K", "L", "M"],
    ),
    (1, 3): (
        [
            ("A", "E", "C"),
            ("A", "F", "C"),
            ("A", "E", "D"),
            ("A", "F", "D"),
            ("B", "E", "C"),
            ("B", "F", "C"),
            ("B", "E", "D"),
            ("B", "F", "D"),
        ],
        ["K", "M", "L"],
    ),
    (2, 0): (
        [
            ("E", "A", "C"),
            ("F", "A", "C"),
            ("E", "A", "D"),
            ("F", "A", "D"),
            ("E", "B", "C"),
            ("F", "B", "C"),
            ("E", "B", "D"),
            ("F", "B", "D"),
        ],
        ["M", "K", "L"],
    ),
    (2, 1): (
        [
            ("A", "E", "C"),
            ("A", "F", "C"),
            ("A", "E", "D"),
            ("A", "F", "D"),
            ("B", "E", "C"),
            ("B", "F", "C"),
            ("B", "E", "D"),
            ("B", "F", "D"),
        ],
        ["K", "M", "L"],
    ),
    (2, 3): (
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
        ["K", "L", "M"],
    ),
}


def test_move_level_mi():
    df0 = gen_df(1, 3)
    orig = vn(df0.columns)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        for j in range(4):
            if i != j:
                df = df0.copy()
                move_level(df.columns, i, j, inplace=True)
                assert vn(df.columns) == _MOVE_RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(IndexError):
        move_level(df.columns, 0, -5, inplace=True)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                df = df0.copy()
                move_level(df.columns, i_name, j_name, inplace=True)
                assert vn(df.columns) == _MOVE_RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(KeyError):
        move_level(df.columns, "K", "Q", inplace=True)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        for j in range(4):
            if i != j:
                df = df0.copy()
                idx = move_level(df.columns, i, j, inplace=False)
                assert vn(df.columns) == orig, (i, j)
                assert vn(idx) == _MOVE_RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(IndexError):
        move_level(df.columns, 0, -5, inplace=False)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                df = df0.copy()
                idx = move_level(df.columns, i_name, j_name, inplace=False)
                assert vn(df.columns) == orig, (i, j)
                assert vn(idx) == _MOVE_RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(KeyError):
        move_level(df.columns, "K", "Q", inplace=False)


def test_move_level_df_col():
    df0 = gen_df(1, 3)
    orig = vicn(df0)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        for j in range(4):
            if i != j:
                df = df0.copy()
                move_level(df, i, j, axis=1, inplace=True)
                assert vn(df.columns) == _MOVE_RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(IndexError):
        move_level(df, 0, -5, axis=1, inplace=True)

    df = df0.copy()
    with pytest.raises(TypeError):
        move_level(df, 1, 0, axis=0, inplace=True)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                df = df0.copy()
                move_level(df, i_name, j_name, axis=1, inplace=True)
                assert vn(df.columns) == _MOVE_RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(KeyError):
        move_level(df, "K", "Q", axis=1, inplace=True)

    df = df0.copy()
    with pytest.raises(TypeError):
        move_level(df, "k", "k", axis=0, inplace=True)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        for j in range(4):
            if i != j:
                df = df0.copy()
                df1 = move_level(df, i, j, axis=1, inplace=False)
                assert vicn(df) == orig, (i, j)
                assert vn(df1.columns) == _MOVE_RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(IndexError):
        move_level(df, 0, -5, axis=1, inplace=False)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                df = df0.copy()
                df1 = move_level(df, i_name, j_name, axis=1, inplace=False)
                assert vicn(df) == orig, (i, j)
                assert vn(df1.columns) == _MOVE_RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(KeyError):
        move_level(df, "K", "Q", axis=1, inplace=False)

    # * axis = None
    for i in range(3):
        for j in range(4):
            if i != j:
                df = df0.copy()
                move_level(df, i, j, inplace=True)
                assert vn(df.columns) == _MOVE_RESULTS[i, j], (i, j)


def test_move_level_df_row():
    df0 = gen_df(1, 3).T
    orig = vicn(df0)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        for j in range(4):
            if i != j:
                df = df0.copy()
                move_level(df, i, j, axis=0, inplace=True)
                assert vn(df.index) == _MOVE_RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(IndexError):
        move_level(df, 0, -5, axis=0, inplace=True)

    df = df0.copy()
    with pytest.raises(TypeError):
        move_level(df, 1, 0, axis=1, inplace=True)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                df = df0.copy()
                move_level(df, i_name, j_name, axis=0, inplace=True)
                assert vn(df.index) == _MOVE_RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(KeyError):
        move_level(df, "K", "Q", axis=0, inplace=True)

    df = df0.copy()
    with pytest.raises(TypeError):
        move_level(df, "k", "k", axis=1, inplace=True)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        for j in range(4):
            if i != j:
                df = df0.copy()
                df1 = move_level(df, i, j, axis=0, inplace=False)
                assert vicn(df) == orig, (i, j)
                assert vn(df1.index) == _MOVE_RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(IndexError):
        move_level(df, 0, -5, axis=0, inplace=False)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                df = df0.copy()
                df1 = move_level(df, i_name, j_name, axis=0, inplace=False)
                assert vicn(df) == orig, (i, j)
                assert vn(df1.index) == _MOVE_RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(KeyError):
        move_level(df, "K", "Q", axis=0, inplace=False)


def test_move_level_s():
    s0 = gen_df(1, 3).loc["a"]
    orig = vin(s0)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        for j in range(4):
            if i != j:
                s = s0.copy()
                move_level(s, i, j, axis=0, inplace=True)
                assert vn(s.index) == _MOVE_RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(IndexError):
        move_level(s, 0, -5, axis=0, inplace=True)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                s = s0.copy()
                move_level(s, i_name, j_name, axis=0, inplace=True)
                assert vn(s.index) == _MOVE_RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(KeyError):
        move_level(s, "K", "Q", axis=0, inplace=True)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        for j in range(4):
            if i != j:
                s = s0.copy()
                df1 = move_level(s, i, j, axis=0, inplace=False)
                assert vin(s) == orig, (i, j)
                assert vn(df1.index) == _MOVE_RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(IndexError):
        move_level(s, 0, -5, axis=0, inplace=False)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                s = s0.copy()
                df1 = move_level(s, i_name, j_name, axis=0, inplace=False)
                assert vin(s) == orig, (i, j)
                assert vn(df1.index) == _MOVE_RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(KeyError):
        move_level(s, "K", "Q", axis=0, inplace=False)


SORTED_RESULTS = {
    (0, 1): (
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
        ["K", "L", "M"],
    ),
    (0, 2): (
        [
            ("C", "A", "E"),
            ("C", "A", "F"),
            ("C", "B", "E"),
            ("C", "B", "F"),
            ("D", "A", "E"),
            ("D", "A", "F"),
            ("D", "B", "E"),
            ("D", "B", "F"),
        ],
        ["L", "K", "M"],
    ),
    (0, 3): (
        [
            ("C", "E", "A"),
            ("C", "E", "B"),
            ("C", "F", "A"),
            ("C", "F", "B"),
            ("D", "E", "A"),
            ("D", "E", "B"),
            ("D", "F", "A"),
            ("D", "F", "B"),
        ],
        ["L", "M", "K"],
    ),
    (1, 0): (
        [
            ("C", "A", "E"),
            ("C", "A", "F"),
            ("C", "B", "E"),
            ("C", "B", "F"),
            ("D", "A", "E"),
            ("D", "A", "F"),
            ("D", "B", "E"),
            ("D", "B", "F"),
        ],
        ["L", "K", "M"],
    ),
    (1, 2): (
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
        ["K", "L", "M"],
    ),
    (1, 3): (
        [
            ("A", "E", "C"),
            ("A", "E", "D"),
            ("A", "F", "C"),
            ("A", "F", "D"),
            ("B", "E", "C"),
            ("B", "E", "D"),
            ("B", "F", "C"),
            ("B", "F", "D"),
        ],
        ["K", "M", "L"],
    ),
    (2, 0): (
        [
            ("E", "A", "C"),
            ("E", "A", "D"),
            ("E", "B", "C"),
            ("E", "B", "D"),
            ("F", "A", "C"),
            ("F", "A", "D"),
            ("F", "B", "C"),
            ("F", "B", "D"),
        ],
        ["M", "K", "L"],
    ),
    (2, 1): (
        [
            ("A", "E", "C"),
            ("A", "E", "D"),
            ("A", "F", "C"),
            ("A", "F", "D"),
            ("B", "E", "C"),
            ("B", "E", "D"),
            ("B", "F", "C"),
            ("B", "F", "D"),
        ],
        ["K", "M", "L"],
    ),
    (2, 3): (
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
        ["K", "L", "M"],
    ),
}


def test_move_level_s_sort():
    s0 = gen_df(1, 3).loc["a"]
    orig = vin(s0)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        for j in range(4):
            if i != j:
                s = s0.copy()
                move_level(s, i, j, axis=0, inplace=True, sort=True)
                assert vn(s.index) == SORTED_RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(IndexError):
        move_level(s, 0, -5, axis=0, inplace=True, sort=True)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                s = s0.copy()
                move_level(s, i_name, j_name, axis=0, inplace=True, sort=True)
                assert vn(s.index) == SORTED_RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(KeyError):
        move_level(s, "K", "Q", axis=0, inplace=True)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        for j in range(4):
            if i != j:
                s = s0.copy()
                df1 = move_level(s, i, j, axis=0, inplace=False, sort=True)
                assert vin(s) == orig, (i, j)
                assert vn(df1.index) == SORTED_RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(IndexError):
        move_level(s, 0, -5, axis=0, inplace=False, sort=True)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                s = s0.copy()
                df1 = move_level(s, i_name, j_name, axis=0, inplace=False, sort=True)
                assert vin(s) == orig, (i, j)
                assert vn(df1.index) == SORTED_RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(KeyError):
        move_level(s, "K", "Q", axis=0, inplace=False, sort=True)


def test_wrong_types():
    with pytest.raises(TypeError):
        move_level(np.array([1, 2, 3]), 0, 1)


def test_sort_multiindex():
    df = gen_df(2, 2)
    with pytest.raises(ValueError):
        move_level(df.columns, 0, 2, sort=True)


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
    # pytest.main(["-s", __file__ + '::test0'])
