import pytest
import numpy as np
from numpy import inf
import pandas as pd

from pdi import join

def vic(s): 
    return s.values.tolist(), s.index.to_list(), s.columns.to_list()

def test_basic():
    a = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    b = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})

    with pytest.raises(ValueError):
        join([])
    with pytest.raises(ValueError):
        join([a])
    assert vic(join([a,b])) == \
        ([[1, 3, 5, 7], 
          [2, 4, 6, 8]], [0, 1], ['A', 'B', 'C', 'D'])
    with pytest.raises(ValueError):
        join([a, b], on=['A', 'B'])
    with pytest.raises(ValueError):
        join([a, b], how=['left', 'right'])
    with pytest.raises(ValueError):
        join([a, b], suffixes=['_x'])

def test_on():
    a = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}, index=list('abc'))
    b = pd.DataFrame({'A': [1, 2, 3], 'D': [10, 20, 30]}).set_index('A')
    c = pd.DataFrame({'B': [4, 5, 6], 'E': [40, 50, 60]}).set_index('B')
    assert vic(join([a, b, c], on=['A', 'B'])) == \
        ([[1, 4, 7, 10, 40], 
          [2, 5, 8, 20, 50], 
          [3, 6, 9, 30, 60]], ['a', 'b', 'c'], ['A', 'B', 'C', 'D', 'E'])

def test_how():
    a = pd.DataFrame(np.arange(1, 4), columns=list('A'), index=list('abc'))
    b = pd.DataFrame(np.arange(4, 7), columns=list('B'), index=list('bcd'))
    c = pd.DataFrame(np.arange(7, 10), columns=list('C'), index=list('cde'))
    assert vic(join([a, b, c]).fillna(inf)) == \
        ([[1.0, inf, inf], 
          [2.0, 4.0, inf], 
          [3.0, 5.0, 7.0]], ['a', 'b', 'c'], ['A', 'B', 'C'])
    assert vic(join([a, b, c], how='right').fillna(inf)) == \
        ([[3.0, 5.0, 7.0], 
          [inf, 6.0, 8.0], 
          [inf, inf, 9.0]], ['c', 'd', 'e'], ['A', 'B', 'C']) 
    assert vic(join([a, b, c], how='outer').fillna(inf)) == \
        ([[1.0, inf, inf],
          [2.0, 4.0, inf],
          [3.0, 5.0, 7.0],
          [inf, 6.0, 8.0],
          [inf, inf, 9.0]], ['a', 'b', 'c', 'd', 'e'], ['A', 'B', 'C'])
    assert vic(join([a, b, c], how='inner').fillna(inf)) == \
        ([[3, 5, 7]], ['c'], ['A', 'B', 'C'])
    assert vic(join([a, b, c], how=['right', 'left']).fillna(inf)) == \
        ([[2.0, 4.0, inf], 
          [3.0, 5.0, 7.0], 
          [inf, 6.0, 8.0]], ['b', 'c', 'd'], ['A', 'B', 'C'])

def test_suffixes():
    a = pd.DataFrame(np.arange(1, 4), columns=list('X'), index=list('abc')); a
    b = pd.DataFrame(np.arange(4, 7), columns=list('Y'), index=list('abc'), dtype=int); b
    c = pd.DataFrame(np.arange(7, 10), columns=list('Y'), index=list('abc'), dtype=int); c
    assert vic(join([a, b, c], suffixes=['', '_b', '_c'])) == \
        ([[1, 4, 7], 
          [2, 5, 8], 
          [3, 6, 9]], ['a', 'b', 'c'], ['X', 'Y_b', 'Y_c'])

if __name__ == '__main__':
    pytest.main(['-s', __file__])  # + '::test7'])

