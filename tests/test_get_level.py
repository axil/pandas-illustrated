from math import inf

import pytest
import pandas as pd

from pandas.core.generic import NDFrame
from pandas._libs import lib

from pdi import get_level
from pdi.testing import gen_df


def vn(idx):
    return idx.values.tolist(), list(idx.names)

def vin(s):
    return s.values.tolist(), s.index.to_list(), [s.name, s.index.name]

def vicn(df):
    assert isinstance(df, NDFrame)  # Frame or Series
    return (
        df.fillna(inf).values.tolist(),
        df.index.to_list(),
        df.columns.to_list(),
        [list(df.index.names), list(df.columns.names)],
    )


def test_get_level():
    df = gen_df(1, 2)
    
    assert vn(get_level(df.columns, 0)) == (['A', 'A', 'B', 'B'], ['K'])
    assert vn(get_level(df.columns, 1)) == (['C', 'D', 'C', 'D'], ['L'])
    assert vn(get_level(df.index, 0)) == (['a', 'b'], ['k'])
    with pytest.raises(IndexError):
        assert vn(get_level(df.index, 1))

    assert vn(get_level(df.columns, "K")) == (['A', 'A', 'B', 'B'], ['K'])
    assert vn(get_level(df.columns, "L")) == (['C', 'D', 'C', 'D'], ['L'])
    assert vn(get_level(df.index, "k")) == (['a', 'b'], ['k'])
    with pytest.raises(KeyError):
        assert vn(get_level(df.index, "l"))

    assert vn(get_level(df, 0, axis=1)) == (['A', 'A', 'B', 'B'], ['K'])
    assert vn(get_level(df, 1, axis=1)) == (['C', 'D', 'C', 'D'], ['L'])
    assert vn(get_level(df, 0, axis=0)) == (['a', 'b'], ['k'])
    with pytest.raises(IndexError):
        assert vn(get_level(df, 1, axis=0))
    
    assert vn(get_level(df, "K", axis=1)) == (['A', 'A', 'B', 'B'], ['K'])
    assert vn(get_level(df, "L", axis=1)) == (['C', 'D', 'C', 'D'], ['L'])
    assert vn(get_level(df, "k", axis=0)) == (['a', 'b'], ['k'])
    with pytest.raises(KeyError):
        assert vn(get_level(df, "l", axis=0))

def test_raises():
    with pytest.raises(TypeError):
        get_level("hmm", 0)

if __name__ == "__main__":
    pytest.main(["-x", "-s", __file__])  # + '::test7'])
