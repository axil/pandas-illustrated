import pytest
import numpy as np
import pandas as pd
from pandas_illustrated import insert

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
    assert str(exinfo.value) == 'Must verify 0 <= loc <= len(df)'

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

if __name__ == '__main__':
    pytest.main(['-s', __file__])
