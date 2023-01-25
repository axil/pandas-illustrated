from inspect import isclass

import pytest
import pandas as pd
import numpy as np
from pandas._libs import lib

from pdi import rename_level
from pdi.testing import gen_df, vn, vin, vicn, range2d

@pytest.mark.parametrize('df0,mapping,axis,result', [
    (gen_df(1,1), {'K': 'L'}, None, 
    ([[1, 2], [3, 4]], ['a', 'b'], ['A', 'B'], [['k'], ['L']])),
    
    (gen_df(1,1), 'L', None, 
    ([[1, 2], [3, 4]], ['a', 'b'], ['A', 'B'], [['k'], ['L']])),
    
    (gen_df(1,1), {'k': 'l'}, 0,
    ([[1, 2], [3, 4]], ['a', 'b'], ['A', 'B'], [['l'], ['K']])),
    
    (gen_df(1,1), 'l', 0,
    ([[1, 2], [3, 4]], ['a', 'b'], ['A', 'B'], [['l'], ['K']])),

    (gen_df(1,2), {'K': 'M'}, None,
    ([[1, 2, 3, 4], [5, 6, 7, 8]],
    ['a', 'b'],
    [('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D')],
    [['k'], ['M', 'L']])),
    
    (gen_df(2,1), {'k': 'm'}, 0,
    ([[1, 2], [3, 4], [5, 6], [7, 8]],
    [('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd')],
    ['A', 'B'],
    [['m', 'l'], ['K']])),
    
    (gen_df(2,1), {'K': 'M'}, 0, KeyError),
])
def test1(df0, mapping, axis, result):
    if isclass(result) and issubclass(result, Exception):
        with pytest.raises(result):
            rename_level(df0, mapping, axis=axis)
    else:
        orig = vicn(df0)
        
        df = df0.copy()
        df1 = rename_level(df, mapping, axis=axis)
        assert vicn(df1) == result
        assert vicn(df) == orig

        df = df0.copy()
        rename_level(df, mapping, axis=axis, inplace=True)
        assert vicn(df) == result


def test2():
    df = pd.DataFrame(range2d(2,2), index=[['a', 'b']], columns=['A', 'B'])
    assert vicn(rename_level(df, 'K', axis=0)) == \
        ([[1, 2], [3, 4]], [('a',), ('b',)], ['A', 'B'], [['K'], [None]])

if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
    #pytest.main(["-s", __file__ + '::test0'])
