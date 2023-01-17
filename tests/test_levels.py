from math import inf

import pytest
import pandas as pd
#try:
#    from pandas._typing import NDFrameT
#except:  # pandas < 1.4
#    from pandas._typing import FrameOrSeries as NDFrameT
from pandas.core.generic import NDFrame


from pdi import set_level
from pdi.testing import gen_df

def vn(idx):
    return idx.values.tolist(), list(idx.names)

def vicn(df):
    assert isinstance(df, NDFrame)      # Frame or Series
    return (
        df.fillna(inf).values.tolist(),
        df.index.to_list(),
        df.columns.to_list(),
        [list(df.index.names), list(df.columns.names)],
    )

def test_set_level_11():
    df = gen_df(1, 2)

    with pytest.raises(TypeError):
        set_level(df.index, 0, 'z')

    df = gen_df(2, 1)

    with pytest.raises(TypeError):
        set_level(df.columns, 0, 'z')
    
def test_set_level_12_0():
    df = gen_df(1, 2)
    set_level(df.columns, 0, 'Z')
    assert vicn(df) == \
        ([[1, 2, 3, 4], [5, 6, 7, 8]],
        ['a', 'b'],
        [('Z', 'C'), ('Z', 'D'), ('Z', 'C'), ('Z', 'D')],
        [['k'], ['K', 'L']])

def test_set_level_12_1():
    df = gen_df(1, 2)
    set_level(df.columns, 1, 'Z')
    assert vicn(df) == \
   ([[1, 2, 3, 4], [5, 6, 7, 8]],
    ['a', 'b'],
    [('A', 'Z'), ('A', 'Z'), ('B', 'Z'), ('B', 'Z')],
    [['k'], ['K', 'L']]) 

def test_set_level_21_0():
    df = gen_df(2, 1)
    set_level(df.index, 0, 'z')
    assert vicn(df) == \
        ([[1, 2], [3, 4], [5, 6], [7, 8]],
        [('z', 'c'), ('z', 'd'), ('z', 'c'), ('z', 'd')],
        ['A', 'B'],
        [['k', 'l'], ['K']])

def test_set_level_21_1():
    df = gen_df(2, 1)
    set_level(df.index, 1, 'z')
    assert vicn(df) == \
        ([[1, 2], [3, 4], [5, 6], [7, 8]],
        [('a', 'z'), ('a', 'z'), ('b', 'z'), ('b', 'z')],
        ['A', 'B'],
        [['k', 'l'], ['K']])

def test_set_level_12_inplace():
    df = gen_df(1, 2)
    df.columns = set_level(df.columns, 0, 'Z', inplace=False)
    assert vicn(df) == \
        ([[1, 2, 3, 4], [5, 6, 7, 8]],
        ['a', 'b'],
        [('Z', 'C'), ('Z', 'D'), ('Z', 'C'), ('Z', 'D')],
        [['k'], ['K', 'L']])

def test_set_level_12_Q():
    res = ([('Q', 'C'), ('Q', 'D'), ('Q', 'C'), ('Q', 'D')], ['K', 'L'])
    
    df = gen_df(1, 2)
    set_level(df.columns, 0, 'Q')
    assert vn(df.columns) == res

    df = gen_df(1, 2)
    df.columns = set_level(df.columns, 0, 'Q', inplace=False)
    assert vn(df.columns) == res
    
def test_set_level_12_QR():
    res = ([('Q', 'C'), ('R', 'D'), ('Q', 'C'), ('R', 'D')], ['K', 'L'])
    
    df = gen_df(1, 2)
    set_level(df.columns, 0, ['Q', 'R'])
    assert vn(df.columns) == res

    df = gen_df(1, 2)
    df.columns = set_level(df.columns, 0, ['Q', 'R'], inplace=False)
    assert vn(df.columns) == res
    
def test_set_level_12_QRS():
    df = gen_df(1, 2)

    with pytest.raises(ValueError):
        set_level(df.columns, 0, ['Q', 'R', 'S'])

    df = gen_df(1, 2)
    
    with pytest.raises(ValueError):
        df.columns = set_level(df.columns, 0, ['Q', 'R', 'S'], inplace=False)
    
def test_set_level_12_QRST():
    res = [('Q', 'C'), ('R', 'D'), ('S', 'C'), ('T', 'D')], ['K', 'L']

    df = gen_df(1, 2)
    set_level(df.columns, 0, ['Q', 'R', 'S', 'T'])
    assert vn(df.columns) == res

    df = gen_df(1, 2)
    df.columns = set_level(df.columns, 0, ['Q', 'R', 'S', 'T'], inplace=False)
    assert vn(df.columns) == res
    
def test_set_level_12_name():
    res = ([('A', 'Q'), ('A', 'R'), ('B', 'Q'), ('B', 'R')], ['K', 'M'])
    
    df = gen_df(1, 2)
    set_level(df.columns, 1, ['Q', 'R'], name='M')
    assert vn(df.columns) == res

    df = gen_df(1, 2)
    df.columns = set_level(df.columns, 1, ['Q', 'R'], name='M', inplace=False)
    assert vn(df.columns) == res
    

if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
