import pytest

import numpy as np
import pandas as pd

import pdi
from pdi import lock_order, from_product
from pdi.categoricals import _get_categories


def vi(s): 
    return s.values.tolist(), s.index.to_list()


def vic(s): 
    return s.values.tolist(), s.index.to_list(), s.columns.to_list()


def test_lock_order1_3Lx1L():
    df0 = pd.DataFrame(
        np.zeros((8, 2), int), 
        index=pdi.from_dict({'K': ('b', 'a'), 
                             'L': ('d', 'c'),
                             'M': ('f', 'e')}), 
        columns=['B', 'A'])
    for i in range(3):
        df = df0.copy()
        old_index = df.index.values.tolist()
        lock_order(df, level=i, axis=0)
        assert isinstance(df.index.get_level_values(i), pd.CategoricalIndex)
        assert df.index.values.tolist() == old_index
    
    df = df0.copy()
    old_index = df.columns.values.tolist()
    lock_order(df, axis=1)
    assert isinstance(df.columns, pd.CategoricalIndex)
    assert df.columns.values.tolist() == old_index


def test_lock_order2_1Lx3L():
    df0 = pd.DataFrame(
        np.zeros((2, 8), int), 
        columns=pdi.from_dict({'K': ('B', 'A'), 
                               'L': ('D', 'C'),
                               'M': ('F', 'E')}), 
        index=['b', 'a'])
    for i in range(3):
        df = df0.copy()
        old_index = df.columns.values.tolist()
        lock_order(df, level=i, axis=1)
        assert isinstance(df.columns.get_level_values(i), pd.CategoricalIndex)
        assert df.columns.values.tolist() == old_index
        
    df = df0.copy()
    old_index = df.index.values.tolist()
    lock_order(df, axis=0)
    assert isinstance(df.index, pd.CategoricalIndex)
    assert df.index.values.tolist() == old_index


def test_get_categories():
    df = pd.DataFrame(
        np.zeros((8, 2), int), 
        index=pdi.from_dict({'K': ('b', 'a'), 
                             'L': ('d', 'c'),
                             'M': ('f', 'e')}), 
        columns=['A', 'B'])
    assert _get_categories(df.index, 0, 'index').tolist() == ['b', 'a']
    assert _get_categories(df.index, 1, 'index').tolist() == ['d', 'c']
    assert _get_categories(df.index, 2, 'index').tolist() == ['f', 'e']


def test_dropped_column():
    df = pd.DataFrame(
        np.arange(1, 9).reshape(2, 4),
        index=list('ab'),
        columns=pd.MultiIndex.from_product([list('AB'), list('CD')]))

    df1 = df.drop(columns=('A', 'D'))

    with pytest.raises(ValueError):
        pdi.lock_order(df1)

    df1 = lock_order(df, level=0, axis=1, inplace=False)
    assert isinstance(df1.columns.get_level_values(0), pd.CategoricalIndex)
    assert df1.columns.values.tolist() == df.columns.values.tolist()
    
    df1 = lock_order(df, axis=0, inplace=False)
    assert isinstance(df1.index, pd.CategoricalIndex)
    assert df1.index.values.tolist() == df.index.values.tolist()


def test_incorrect_sorting():
    df = pd.DataFrame(
        np.arange(1, 9).reshape(4, 2),
        index=pdi.from_dict({'K': list('baab'), 'L': list('ddcc')}),
        columns=list('AB'))
    
    df1 = pdi.lock_order(df, axis=0, level=1, inplace=False)
    assert isinstance(df1.index.get_level_values(1), pd.CategoricalIndex)
    assert df1.columns.values.tolist() == df.columns.values.tolist()
    assert df1.index.values.tolist() == df.index.values.tolist()
    
    with pytest.raises(ValueError):
        pdi.lock_order(df, axis=0, level=0)


def test_series():
    s = pd.Series(
            np.arange(12),          
            index=pdi.from_dict({'K': tuple('BA'), 
                                 'L': tuple('PYTHON')}))
    s1 = lock_order(s, inplace=False)
    assert isinstance(s1.index.get_level_values(0), pd.CategoricalIndex)
    assert isinstance(s1.index.get_level_values(1), pd.CategoricalIndex)
    assert s1.index.values.tolist() == s.index.values.tolist()


def test_index():
    idx = pd.Index(list('PYTHON'))
    idx1 = pdi.lock_order(idx)
    assert isinstance(idx1, pd.CategoricalIndex)
    assert idx1.values.tolist() == list('PYTHON')
    assert idx1.categories.tolist() == list('PYTHON')
    assert idx1.ordered == True

def test_multiindex():
    idx = pd.MultiIndex.from_product([('B', 'A'), ('D', 'C')])
    idx1 = pdi.lock_order(idx)
    L1 = idx1.get_level_values(0)
    L2 = idx1.get_level_values(1)
    
    assert isinstance(L1, pd.CategoricalIndex)
    assert L1.values.tolist() == ['B', 'B', 'A', 'A']
    assert L1.categories.tolist() == ['B', 'A']
    assert L1.ordered == True
    
    assert isinstance(L2, pd.CategoricalIndex)
    assert L2.values.tolist() == ['D', 'C', 'D', 'C']
    assert L2.categories.tolist() == ['D', 'C']
    assert L2.ordered == True


def test_categories():
    idx = pd.MultiIndex.from_product([('B','A'), tuple('PTHYON')])
    idx1 = pdi.lock_order(idx, level=1, categories=list('PYTHON'))
    s = pd.Series(np.arange(1, 13), index=idx1)
    assert s.index.get_level_values(1).categories.tolist() == list('PYTHON')
    s1 = s.sort_index()
    assert vi(s1) == \
        ([7, 10, 8, 9, 11, 12, 1, 4, 2, 3, 5, 6],
         [('A', 'P'),
          ('A', 'Y'),
          ('A', 'T'),
          ('A', 'H'),
          ('A', 'O'),
          ('A', 'N'),
          ('B', 'P'),
          ('B', 'Y'),
          ('B', 'T'),
          ('B', 'H'),
          ('B', 'O'),
          ('B', 'N')])


def test_from_product():
    mi = from_product([[2022, 2021], list('PYTHON')], names=['K', 'L'])
    assert isinstance(mi.get_level_values(0), pd.CategoricalIndex)
    assert isinstance(mi.get_level_values(1), pd.CategoricalIndex)


if __name__ == '__main__':
    pytest.main(['-s', __file__])  # + '::test7'])
