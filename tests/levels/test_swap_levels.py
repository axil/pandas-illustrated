from math import inf

import pytest
import pandas as pd

from pandas.core.generic import NDFrame

from pdi import set_level, get_level, drop_level, swap_levels, swap_levels
from pdi.testing import gen_df


def vn(idx):
    return idx.values.tolist(), list(idx.names)

def vin(s):
    return s.values.tolist(), s.index.to_list(), [s.name, s.index.name]

def vicn(df):
    assert isinstance(df, NDFrame)  # Frame or Series
    return (
        df.fillna(inf).values.tolist(),
        df.index.to_list(),
        df.columns.to_list(),
        [list(df.index.names), list(df.columns.names)],
    )


RESULTS = \
{(0,
  1): ([('C', 'A', 'E'),
   ('C', 'A', 'F'),
   ('D', 'A', 'E'),
   ('D', 'A', 'F'),
   ('C', 'B', 'E'),
   ('C', 'B', 'F'),
   ('D', 'B', 'E'),
   ('D', 'B', 'F')], ['L', 'K', 'M']),
 (0,
  2): ([('E', 'C', 'A'),
   ('F', 'C', 'A'),
   ('E', 'D', 'A'),
   ('F', 'D', 'A'),
   ('E', 'C', 'B'),
   ('F', 'C', 'B'),
   ('E', 'D', 'B'),
   ('F', 'D', 'B')], ['M', 'L', 'K']),
 (1,
  0): ([('C', 'A', 'E'),
   ('C', 'A', 'F'),
   ('D', 'A', 'E'),
   ('D', 'A', 'F'),
   ('C', 'B', 'E'),
   ('C', 'B', 'F'),
   ('D', 'B', 'E'),
   ('D', 'B', 'F')], ['L', 'K', 'M']),
 (1,
  2): ([('A', 'E', 'C'),
   ('A', 'F', 'C'),
   ('A', 'E', 'D'),
   ('A', 'F', 'D'),
   ('B', 'E', 'C'),
   ('B', 'F', 'C'),
   ('B', 'E', 'D'),
   ('B', 'F', 'D')], ['K', 'M', 'L']),
 (2,
  0): ([('E', 'C', 'A'),
   ('F', 'C', 'A'),
   ('E', 'D', 'A'),
   ('F', 'D', 'A'),
   ('E', 'C', 'B'),
   ('F', 'C', 'B'),
   ('E', 'D', 'B'),
   ('F', 'D', 'B')], ['M', 'L', 'K']),
 (2,
  1): ([('A', 'E', 'C'),
   ('A', 'F', 'C'),
   ('A', 'E', 'D'),
   ('A', 'F', 'D'),
   ('B', 'E', 'C'),
   ('B', 'F', 'C'),
   ('B', 'E', 'D'),
   ('B', 'F', 'D')], ['K', 'M', 'L'])}

def test_swap_levels_mi():
    df0 = gen_df(1, 3)
    orig = vn(df0.columns)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        for j in range(3):
            if i != j:
                df = df0.copy()
                swap_levels(df.columns, i, j, inplace=True)
                assert vn(df.columns) == RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(IndexError):
        swap_levels(df.columns, 0, -5, inplace=True)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                df = df0.copy()
                swap_levels(df.columns, i_name, j_name, inplace=True)
                assert vn(df.columns) == RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(KeyError):
        swap_levels(df.columns, "K", "Q", inplace=True)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        for j in range(3):
            if i != j:
                df = df0.copy()
                idx = swap_levels(df.columns, i, j, inplace=False)
                assert vn(df.columns) == orig, (i, j)
                assert vn(idx) == RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(IndexError):
        swap_levels(df.columns, 0, -5, inplace=False)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                df = df0.copy()
                idx = swap_levels(df.columns, i_name, j_name, inplace=False)
                assert vn(df.columns) == orig, (i, j)
                assert vn(idx) == RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(KeyError):
        swap_levels(df.columns, "K", "Q", inplace=False)


def test_swap_levels_df_col():
    df0 = gen_df(1, 3)
    orig = vicn(df0)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        for j in range(3):
            if i != j:
                df = df0.copy()
                swap_levels(df, i, j, axis=1, inplace=True)
                assert vn(df.columns) == RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(IndexError):
        swap_levels(df, 0, -5, axis=1, inplace=True)

    df = df0.copy()
    with pytest.raises(TypeError):
        swap_levels(df, 1, 0, axis=0, inplace=True)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                df = df0.copy()
                swap_levels(df, i_name, j_name, axis=1, inplace=True)
                assert vn(df.columns) == RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(KeyError):
        swap_levels(df, "K", "Q", axis=1, inplace=True)

    df = df0.copy()
    with pytest.raises(TypeError):
        swap_levels(df, "k", "k", axis=0, inplace=True)
    
    # * inplace = False
    #   - by positional index
    for i in range(3):
        for j in range(3):
            if i != j:
                df = df0.copy()
                df1 = swap_levels(df, i, j, axis=1, inplace=False)
                assert vicn(df) == orig, (i, j)
                assert vn(df1.columns) == RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(IndexError):
        swap_levels(df, 0, -5, axis=1, inplace=False)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                df = df0.copy()
                df1 = swap_levels(df, i_name, j_name, axis=1, inplace=False)
                assert vicn(df) == orig, (i, j)
                assert vn(df1.columns) == RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(KeyError):
        swap_levels(df, "K", "Q", axis=1, inplace=False)


def test_swap_levels_df_row():
    df0 = gen_df(1, 3).T
    orig = vicn(df0)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        for j in range(3):
            if i != j:
                df = df0.copy()
                swap_levels(df, i, j, axis=0, inplace=True)
                assert vn(df.index) == RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(IndexError):
        swap_levels(df, 0, -5, axis=0, inplace=True)

    df = df0.copy()
    with pytest.raises(TypeError):
        swap_levels(df, 1, 0, axis=1, inplace=True)
    
    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                df = df0.copy()
                swap_levels(df, i_name, j_name, axis=0, inplace=True)
                assert vn(df.index) == RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(KeyError):
        swap_levels(df, "K", "Q", axis=0, inplace=True)

    df = df0.copy()
    with pytest.raises(TypeError):
        swap_levels(df, "k", "k", axis=1, inplace=True)
    
    # * inplace = False
    #   - by positional index
    for i in range(3):
        for j in range(3):
            if i != j:
                df = df0.copy()
                df1 = swap_levels(df, i, j, axis=0, inplace=False)
                assert vicn(df) == orig, (i, j)
                assert vn(df1.index) == RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(IndexError):
        swap_levels(df, 0, -5, axis=0, inplace=False)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                df = df0.copy()
                df1 = swap_levels(df, i_name, j_name, axis=0, inplace=False)
                assert vicn(df) == orig, (i, j)
                assert vn(df1.index) == RESULTS[i, j], (i, j)

    df = df0.copy()
    with pytest.raises(KeyError):
        swap_levels(df, "K", "Q", axis=0, inplace=False)


def test_swap_levels_s():
    s0 = gen_df(1, 3).loc['a']
    orig = vin(s0)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        for j in range(3):
            if i != j:
                s = s0.copy()
                swap_levels(s, i, j, axis=0, inplace=True)
                assert vn(s.index) == RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(IndexError):
        swap_levels(s, 0, -5, axis=0, inplace=True)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                s = s0.copy()
                swap_levels(s, i_name, j_name, axis=0, inplace=True)
                assert vn(s.index) == RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(KeyError):
        swap_levels(s, "K", "Q", axis=0, inplace=True)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        for j in range(3):
            if i != j:
                s = s0.copy()
                df1 = swap_levels(s, i, j, axis=0, inplace=False)
                assert vin(s) == orig, (i, j)
                assert vn(df1.index) == RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(IndexError):
        swap_levels(s, 0, -5, axis=0, inplace=False)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                s = s0.copy()
                df1 = swap_levels(s, i_name, j_name, axis=0, inplace=False)
                assert vin(s) == orig, (i, j)
                assert vn(df1.index) == RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(KeyError):
        swap_levels(s, "K", "Q", axis=0, inplace=False)

SORTED_RESULTS = \
{(0,
  1): ([('C', 'A', 'E'),
   ('C', 'A', 'F'),
   ('C', 'B', 'E'),
   ('C', 'B', 'F'),
   ('D', 'A', 'E'),
   ('D', 'A', 'F'),
   ('D', 'B', 'E'),
   ('D', 'B', 'F')], ['L', 'K', 'M']),
 (0,
  2): ([('E', 'C', 'A'),
   ('E', 'C', 'B'),
   ('E', 'D', 'A'),
   ('E', 'D', 'B'),
   ('F', 'C', 'A'),
   ('F', 'C', 'B'),
   ('F', 'D', 'A'),
   ('F', 'D', 'B')], ['M', 'L', 'K']),
 (1,
  0): ([('C', 'A', 'E'),
   ('C', 'A', 'F'),
   ('C', 'B', 'E'),
   ('C', 'B', 'F'),
   ('D', 'A', 'E'),
   ('D', 'A', 'F'),
   ('D', 'B', 'E'),
   ('D', 'B', 'F')], ['L', 'K', 'M']),
 (1,
  2): ([('A', 'E', 'C'),
   ('A', 'E', 'D'),
   ('A', 'F', 'C'),
   ('A', 'F', 'D'),
   ('B', 'E', 'C'),
   ('B', 'E', 'D'),
   ('B', 'F', 'C'),
   ('B', 'F', 'D')], ['K', 'M', 'L']),
 (2,
  0): ([('E', 'C', 'A'),
   ('E', 'C', 'B'),
   ('E', 'D', 'A'),
   ('E', 'D', 'B'),
   ('F', 'C', 'A'),
   ('F', 'C', 'B'),
   ('F', 'D', 'A'),
   ('F', 'D', 'B')], ['M', 'L', 'K']),
 (2,
  1): ([('A', 'E', 'C'),
   ('A', 'E', 'D'),
   ('A', 'F', 'C'),
   ('A', 'F', 'D'),
   ('B', 'E', 'C'),
   ('B', 'E', 'D'),
   ('B', 'F', 'C'),
   ('B', 'F', 'D')], ['K', 'M', 'L'])}

def test_swap_levels_s_sort():
    s0 = gen_df(1, 3).loc['a']
    orig = vin(s0)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        for j in range(3):
            if i != j:
                s = s0.copy()
                swap_levels(s, i, j, axis=0, inplace=True, sort=True)
                assert vn(s.index) == SORTED_RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(IndexError):
        swap_levels(s, 0, -5, axis=0, inplace=True, sort=True)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                s = s0.copy()
                swap_levels(s, i_name, j_name, axis=0, inplace=True, sort=True)
                assert vn(s.index) == SORTED_RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(KeyError):
        swap_levels(s, "K", "Q", axis=0, inplace=True)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        for j in range(3):
            if i != j:
                s = s0.copy()
                df1 = swap_levels(s, i, j, axis=0, inplace=False, sort=True)
                assert vin(s) == orig, (i, j)
                assert vn(df1.index) == SORTED_RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(IndexError):
        swap_levels(s, 0, -5, axis=0, inplace=False, sort=True)

    #   - by name
    for i, i_name in enumerate("KLM"):
        for j, j_name in enumerate("KLM"):
            if i != j:
                s = s0.copy()
                df1 = swap_levels(s, i_name, j_name, axis=0, inplace=False, sort=True)
                assert vin(s) == orig, (i, j)
                assert vn(df1.index) == SORTED_RESULTS[i, j], (i, j)

    s = s0.copy()
    with pytest.raises(KeyError):
        swap_levels(s, "K", "Q", axis=0, inplace=False, sort=True)

if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
    #pytest.main(["-s", __file__ + '::test0'])
