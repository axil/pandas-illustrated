from math import inf

import pytest
import pandas as pd
import numpy as np

from pandas.core.generic import NDFrame

from pdi import join_levels, split_level
from pdi.testing import gen_df, vn, vin, vicn

@pytest.mark.parametrize("names, result", [
    (None, 
    ([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
 ['a', 'b'],
 [('A', 'C', 'E'),
  ('A', 'C', 'F'),
  ('A', 'D', 'E'),
  ('A', 'D', 'F'),
  ('B', 'C', 'E'),
  ('B', 'C', 'F'),
  ('B', 'D', 'E'),
  ('B', 'D', 'F')],
 [['k'], [None, None, None]])
    ),
    (['K', 'L', 'M'], 
    ([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
 ['a', 'b'],
 [('A', 'C', 'E'),
  ('A', 'C', 'F'),
  ('A', 'D', 'E'),
  ('A', 'D', 'F'),
  ('B', 'C', 'E'),
  ('B', 'C', 'F'),
  ('B', 'D', 'E'),
  ('B', 'D', 'F')],
 [['k'], ['K', 'L', 'M']])
    ),
])
def test_join_levels(names, result):
    df0 = join_levels(gen_df(1,3))
    orig = vicn(df0)

    # * inplace=False
    df = df0.copy()
    df1 = split_level(df, names=names)
    assert vicn(df1) == result
    assert vicn(df) == orig

    # inplace=True
    df = df0.copy()
    split_level(df, names=names, inplace=True)
    assert vicn(df) == result

def test_wrong_types():
    with pytest.raises(TypeError):
        split_level(np.array([1,2,3]))

    df = gen_df(2, 2)
    with pytest.raises(TypeError):
        split_level(df)

def test_joining_multiindex_inplace():
    df = gen_df(1, 1)
    with pytest.raises(ValueError):
        split_level(df.columns, inplace=True)
    
if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
    #pytest.main(["-s", __file__ + '::test0'])
