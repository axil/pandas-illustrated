import pytest

import numpy as np
import pandas as pd

import pdi
from pdi import lock_order
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


if __name__ == '__main__':
    pytest.main(['-s', __file__])  # + '::test7'])
