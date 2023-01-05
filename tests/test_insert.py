import pytest
import numpy as np
from numpy import inf
import pandas as pd
from pdi import insert

def vi(s): 
    return s.values.tolist(), s.index.to_list()

def test1():
    df = pd.DataFrame([[4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'])
    row = [1, 2, 3]
    
    df1 = insert(df, 0, row)
    assert vi(df1) == ([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]], [0, 0, 1])

    df1 = insert(df, 1, row)
    assert vi(df1) == ([[4, 5, 6],
                        [1, 2, 3],
                        [7, 8, 9]], [0, 0, 1])

    df1 = insert(df, 2, row)
    assert vi(df1) == ([[4, 5, 6],
                        [7, 8, 9],
                        [1, 2, 3]], [0, 1, 0])
    
    df1 = insert(df, 0, row, ignore_index=True)
    assert vi(df1) == ([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]], [0, 1, 2])

    df1 = insert(df, 1, row, ignore_index=True)
    assert vi(df1) == ([[4, 5, 6],
                        [1, 2, 3],
                        [7, 8, 9]], [0, 1, 2])

    df1 = insert(df, 2, row, ignore_index=True)
    assert vi(df1) == ([[4, 5, 6],
                        [7, 8, 9],
                        [1, 2, 3]], [0, 1, 2])

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

def test_series1():
    s = pd.Series([20, 30])

    assert vi(insert(s, 0, 10)) == ([10, 20, 30], [0, 0, 1])
    assert vi(insert(s, 1, 10)) == ([20, 10, 30], [0, 0, 1])
    assert vi(insert(s, 2, 10)) == ([20, 30, 10], [0, 1, 0])

    assert vi(insert(s, 0, 10, ignore_index=True)) == ([10, 20, 30], [0, 1, 2])
    assert vi(insert(s, 1, 10, ignore_index=True)) == ([20, 10, 30], [0, 1, 2])
    assert vi(insert(s, 2, 10, ignore_index=True)) == ([20, 30, 10], [0, 1, 2])

def test_series2():
    s = pd.Series([20, 30], index=['b', 'c'])
    
    assert vi(insert(s, 0, 10, 'a')) == ([10, 20, 30], ['a', 'b', 'c'])
    assert vi(insert(s, 1, 10, 'a')) == ([20, 10, 30], ['b', 'a', 'c'])
    assert vi(insert(s, 2, 10, 'a')) == ([20, 30, 10], ['b', 'c', 'a'])
    
    assert vi(insert(s, 0, 10, ignore_index=True)) == ([10, 20, 30], [0, 1, 2])
    assert vi(insert(s, 1, 10, ignore_index=True)) == ([20, 10, 30], [0, 1, 2])
    assert vi(insert(s, 2, 10, ignore_index=True)) == ([20, 30, 10], [0, 1, 2])

    assert vi(insert(s, 0, 10)) == ([10, 20, 30], [0, 'b', 'c'])
    assert vi(insert(s, 1, 10)) == ([20, 10, 30], ['b', 0, 'c'])
    assert vi(insert(s, 2, 10)) == ([20, 30, 10], ['b', 'c', 0])
    
def test_series3():
    s = pd.Series([20, 30], index=['b', 'c'])
    s1 = pd.Series([11, 12], index=['x', 'y'])

    assert vi(insert(s, 0, s1)) == ([11, 12, 20, 30], ['x', 'y', 'b', 'c'])
    assert vi(insert(s, 1, s1)) == ([20, 11, 12, 30], ['b', 'x', 'y', 'c'])
    assert vi(insert(s, 2, s1)) == ([20, 30, 11, 12], ['b', 'c', 'x', 'y'])

    assert vi(insert(s, 0, s1, ignore_index=True)) == ([11, 12, 20, 30], [0, 1, 2, 3])
    assert vi(insert(s, 1, s1, ignore_index=True)) == ([20, 11, 12, 30], [0, 1, 2, 3])
    assert vi(insert(s, 2, s1, ignore_index=True)) == ([20, 30, 11, 12], [0, 1, 2, 3])

def test_allow_duplicates():
    df = pd.DataFrame([[4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'])
    with pytest.raises(ValueError) as exinfo:
        insert(df, 0, [1,2,3], allow_duplicates=False)
    assert str(exinfo.value) == 'cannot insert label 0, already exists; consider ignore_index=True'
    df1 = insert(df, 0, [1, 2, 3], label=2, allow_duplicates=False)
    assert vi(df1) == ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [2, 0, 1])

if __name__ == '__main__':
    pytest.main(['-s', __file__])  # + '::test7'])
