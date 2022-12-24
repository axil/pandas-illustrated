﻿import pytest
import numpy as np
import pandas as pd
from pandas_illustrated import insert
from numpy import inf

def test1():
    df = pd.DataFrame([[4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'])
    row = [1, 2, 3]
    
    df1 = insert(df, 0, row)
    assert df1.values.tolist() == [[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]]

    df1 = insert(df, 1, row)
    assert df1.values.tolist() == [[4, 5, 6],
                                   [1, 2, 3],
                                   [7, 8, 9]]

    df1 = insert(df, 2, row)
    assert df1.values.tolist() == [[4, 5, 6],
                                   [7, 8, 9],
                                   [1, 2, 3]]

def test2():
    df = pd.DataFrame([[4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'])
    row = {'A': 1, 'B': 2, 'C': 3}
    
    df1 = insert(df, 0, row)
    assert df1.values.tolist() == [[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]]

    df1 = insert(df, 1, row)
    assert df1.values.tolist() == [[4, 5, 6],
                                   [1, 2, 3],
                                   [7, 8, 9]]

    df1 = insert(df, 2, row)
    assert df1.values.tolist() == [[4, 5, 6],
                                   [7, 8, 9],
                                   [1, 2, 3]]

def test3():
    df = pd.DataFrame([[4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'])
    with pytest.raises(ValueError) as exinfo:
        insert(df, -1, [1, 2, 3])
    assert str(exinfo.value) == 'Must verify 0 <= loc <= len(dst)'

def test4():
    df = pd.DataFrame([[4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'], index=['a', 'b'])
    row = [1, 2, 3]
    
    df1 = insert(df, 0, row, 'c')
    assert df1.values.tolist() == [[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]]
    assert df1.index.tolist() == ['c', 'a', 'b']
    
    df1 = insert(df, 1, row, 'c')
    assert df1.values.tolist() == [[4, 5, 6],
                                   [1, 2, 3],
                                   [7, 8, 9]]
    assert df1.index.tolist() == ['a', 'c', 'b']
    
    df1 = insert(df, 2, row, 'c')
    assert df1.values.tolist() == [[4, 5, 6],
                                   [7, 8, 9],
                                   [1, 2, 3]]
    assert df1.index.tolist() == ['a', 'b', 'c']

def test5():
    df = pd.DataFrame([[4, 5., '6'], [7, 8., '9']], columns=['A', 'B', 'C'], index=['a', 'b'])
    row = [1, 2., '3']
    
    df1 = insert(df, 0, row, 'c')
    assert df1.values.tolist() == [[1, 2., '3'],
                                   [4, 5., '6'],
                                   [7, 8., '9']]
    assert df1.index.tolist() == ['c', 'a', 'b']
    
    df1 = insert(df, 1, row, 'c')
    assert df1.values.tolist() == [[4, 5., '6'],
                                   [1, 2., '3'],
                                   [7, 8., '9']]
    assert df1.index.tolist() == ['a', 'c', 'b']
    
    df1 = insert(df, 2, row, 'c')
    assert df1.values.tolist() == [[4, 5., '6'],
                                   [7, 8., '9'],
                                   [1, 2., '3']]
    assert df1.index.tolist() == ['a', 'b', 'c']

def test6():
    df = pd.DataFrame([[4, 5, 6], [7, 8, 9]], 
            columns=['A', 'B', 'C'], 
            index=pd.MultiIndex.from_tuples([('a', 'x'), ('a', 'y')]))
    row = [1, 2, 3]
    
    df1 = insert(df, 0, row, ('b', 'z'))
    assert df1.index.tolist() == [('b', 'z'), ('a', 'x'), ('a', 'y')]
    
    df1 = insert(df, 1, row, ('b', 'z'))
    assert df1.index.tolist() == [('a', 'x'), ('b', 'z'), ('a', 'y')]
    
    df1 = insert(df, 2, row, ('b', 'z'))
    assert df1.index.tolist() == [('a', 'x'), ('a', 'y'), ('b', 'z')]

def test7():
    df = pd.DataFrame([[4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'])
    row = pd.DataFrame([[1, 2, 3]], columns=['A', 'B', 'C'])
    
    df1 = insert(df, 0, row)
    assert df1.values.tolist() == [[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]]

    df1 = insert(df, 1, row)
    assert df1.values.tolist() == [[4, 5, 6],
                                   [1, 2, 3],
                                   [7, 8, 9]]

    df1 = insert(df, 2, row)
    assert df1.values.tolist() == [[4, 5, 6],
                                   [7, 8, 9],
                                   [1, 2, 3]]

def test8():
    df = pd.DataFrame([[4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'])
    row = pd.DataFrame([[1, 2, 3]], columns=['B', 'C', 'D'])

    df1 = insert(df, 0, row).fillna(inf)
    assert df1.values.tolist() == [[inf, 1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0, inf], 
                                   [7.0, 8.0, 9.0, inf]]
    
    df1 = insert(df, 1, row).fillna(inf)
    assert df1.values.tolist() == [[4.0, 5.0, 6.0, inf], 
                                   [inf, 1.0, 2.0, 3.0],
                                   [7.0, 8.0, 9.0, inf]] 

    df1 = insert(df, 2, row).fillna(inf)
    assert df1.values.tolist() == [[4.0, 5.0, 6.0, inf], 
                                   [7.0, 8.0, 9.0, inf], 
                                   [inf, 1.0, 2.0, 3.0]]

def test9():
    df = pd.DataFrame([[4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'])
    rows = pd.DataFrame([[40, 50, 60], [70, 80, 90]], columns=['A', 'B', 'C'])

    df1 = insert(df, 0, rows)
    assert df1.values.tolist() == [[40, 50, 60],
                                   [70, 80, 90],
                                   [ 4,  5,  6],
                                   [ 7,  8,  9]]

    df1 = insert(df, 1, rows)
    assert df1.values.tolist() == [[ 4,  5,  6],
                                   [40, 50, 60],
                                   [70, 80, 90],
                                   [ 7,  8,  9]]

    df1 = insert(df, 2, rows)
    assert df1.values.tolist() == [[ 4,  5,  6],
                                   [ 7,  8,  9],
                                   [40, 50, 60],
                                   [70, 80, 90]]

def vi(s): 
    return s.values.tolist(), s.index.to_list()

def test_series1():
    s = pd.Series([20, 30])

    assert vi(insert(s, 0, 10)) == ([10, 20, 30], [0, 1, 2])
    assert vi(insert(s, 1, 10)) == ([20, 10, 30], [0, 1, 2])
    assert vi(insert(s, 2, 10)) == ([20, 30, 10], [0, 1, 2])

def test_series2():
    s = pd.Series([20, 30], index=['b', 'c'])
    
    assert vi(insert(s, 0, 10)) == ([10, 20, 30], [0, 1, 2])
    assert vi(insert(s, 1, 10)) == ([20, 10, 30], [0, 1, 2])
    assert vi(insert(s, 2, 10)) == ([20, 30, 10], [0, 1, 2])
    
    assert vi(insert(s, 0, 10, 'a')) == ([10, 20, 30], ['a', 'b', 'c'])
    assert vi(insert(s, 1, 10, 'a')) == ([20, 10, 30], ['b', 'a', 'c'])
    assert vi(insert(s, 2, 10, 'a')) == ([20, 30, 10], ['b', 'c', 'a'])


if __name__ == '__main__':
    pytest.main(['-s', __file__])  # + '::test7'])
