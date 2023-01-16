import pytest
import numpy as np
import pandas as pd
from pdi import stack, unstack, from_dict
from math import inf


def vin(s): 
    return s.values.tolist(), s.index.to_list(), [s.name, s.index.name]


def vicn(s): 
    return s.fillna(inf).values.tolist(), \
           s.index.to_list(), s.columns.to_list(), \
           [list(s.index.names), list(s.columns.names)]


def test_1Lx1L():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=list('CBA'), index=list('ba')); df

    s = stack(df)
    assert vin(s) == \
        ([1, 2, 3, 4, 5, 6],
         [('b', 'C'), ('b', 'B'), ('b', 'A'), ('a', 'C'), ('a', 'B'), ('a', 'A')],
         [None, None])

    s = unstack(df)
    assert vin(s) == \
        ([1, 4, 2, 5, 3, 6],
         [('C', 'b'), ('C', 'a'), ('B', 'b'), ('B', 'a'), ('A', 'b'), ('A', 'a')],
         [None, None])

    assert vicn(unstack(stack(df))) == vicn(df)

    assert vicn(unstack(unstack(df))) == vicn(df.T)


def test_2Lx2L():
    df = pd.DataFrame(
        np.arange(1, 17).reshape(4, 4), 
        index=from_dict({'K': ('b', 'a'), 
                         'L': ('d', 'c')}), 
        columns=from_dict({'M': ('B', 'A'), 
                           'N': ('D', 'C')})) 

    df1 = stack(df)
    assert vicn(df1) == \
        ([[1, 3], [2, 4], [5, 7], [6, 8], [9, 11], [10, 12], [13, 15], [14, 16]],
         [('b', 'd', 'D'),
          ('b', 'd', 'C'),
          ('b', 'c', 'D'),
          ('b', 'c', 'C'),
          ('a', 'd', 'D'),
          ('a', 'd', 'C'),
          ('a', 'c', 'D'),
          ('a', 'c', 'C')],
         ['B', 'A'],
         [['K', 'L', 'N'], ['M']])

    df1 = unstack(df)
    assert vicn(df1) == \
        ([[1, 5, 2, 6, 3, 7, 4, 8], [9, 13, 10, 14, 11, 15, 12, 16]],
         ['b', 'a'],
         [('B', 'D', 'd'),
          ('B', 'D', 'c'),
          ('B', 'C', 'd'),
          ('B', 'C', 'c'),
          ('A', 'D', 'd'),
          ('A', 'D', 'c'),
          ('A', 'C', 'd'),
          ('A', 'C', 'c')],
         [['K'], ['M', 'N', 'L']])

    assert vicn(unstack(stack(df))) == vicn(df)

    s = unstack(unstack(df))
    assert vin(s) == \
        ([1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15, 4, 12, 8, 16],
         [('B', 'D', 'd', 'b'),
          ('B', 'D', 'd', 'a'),
          ('B', 'D', 'c', 'b'),
          ('B', 'D', 'c', 'a'),
          ('B', 'C', 'd', 'b'),
          ('B', 'C', 'd', 'a'),
          ('B', 'C', 'c', 'b'),
          ('B', 'C', 'c', 'a'),
          ('A', 'D', 'd', 'b'),
          ('A', 'D', 'd', 'a'),
          ('A', 'D', 'c', 'b'),
          ('A', 'D', 'c', 'a'),
          ('A', 'C', 'd', 'b'),
          ('A', 'C', 'd', 'a'),
          ('A', 'C', 'c', 'b'),
          ('A', 'C', 'c', 'a')],
         [None, None])

    s = stack(stack(df))
    assert vin(s) == \
        ([1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16],
         [('b', 'd', 'D', 'B'),
          ('b', 'd', 'D', 'A'),
          ('b', 'd', 'C', 'B'),
          ('b', 'd', 'C', 'A'),
          ('b', 'c', 'D', 'B'),
          ('b', 'c', 'D', 'A'),
          ('b', 'c', 'C', 'B'),
          ('b', 'c', 'C', 'A'),
          ('a', 'd', 'D', 'B'),
          ('a', 'd', 'D', 'A'),
          ('a', 'd', 'C', 'B'),
          ('a', 'd', 'C', 'A'),
          ('a', 'c', 'D', 'B'),
          ('a', 'c', 'D', 'A'),
          ('a', 'c', 'C', 'B'),
          ('a', 'c', 'C', 'A')],
         [None, None])

    df1 = stack(df, 0)
    assert vicn(df1) == \
        ([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]],
         [('b', 'd', 'B'),
          ('b', 'd', 'A'),
          ('b', 'c', 'B'),
          ('b', 'c', 'A'),
          ('a', 'd', 'B'),
          ('a', 'd', 'A'),
          ('a', 'c', 'B'),
          ('a', 'c', 'A')],
         ['D', 'C'],
         [['K', 'L', 'M'], ['N']])

    df1 = unstack(df, 0)
    assert vicn(df1) == \
        ([[1, 9, 2, 10, 3, 11, 4, 12], [5, 13, 6, 14, 7, 15, 8, 16]],
         ['d', 'c'],
         [('B', 'D', 'b'),
          ('B', 'D', 'a'),
          ('B', 'C', 'b'),
          ('B', 'C', 'a'),
          ('A', 'D', 'b'),
          ('A', 'D', 'a'),
          ('A', 'C', 'b'),
          ('A', 'C', 'a')],
         [['L'], ['M', 'N', 'K']])
    
    
def test_topo():
    df = pd.DataFrame(np.arange(1, 9).reshape(2, 4), 
                      index=['b', 'a'], 
                      columns=from_dict({'K': list('BBAA'), 
                                         'L': ['Wed', 'Fri', 'Mon', 'Wed']}))

    df1 = stack(df)
    assert vicn(df1) == \
        ([[inf, 3.0], [1.0, 4.0], [2.0, inf], [inf, 7.0], [5.0, 8.0], [6.0, inf]],
         [('b', 'Mon'),
          ('b', 'Wed'),
          ('b', 'Fri'),
          ('a', 'Mon'),
          ('a', 'Wed'),
          ('a', 'Fri')],
         ['B', 'A'],
         [[None, 'L'], ['K']])
            

if __name__ == '__main__':
    pytest.main(['-s', __file__])  # + '::test7'])
