from itertools import islice
from math import inf

import pandas as pd
from pandas.core.generic import NDFrame
import numpy as np

import pdi

def range2d(n, m):
    return np.arange(1, n * m + 1).reshape(n, m)


def gen_df(n, m, ascending=True):
    index_dict = {
        "k": ("a", "b"),
        "l": ("c", "d"),
        "m": ("e", "f"),
    }
    columns_dict = {
        "K": ("A", "B"),
        "L": ("C", "D"),
        "M": ("E", "F"),
    }
#    if n == 1:
#        index = pd.Index(['a', 'b'], name='k')
#    else:
    index = pdi.from_dict(dict(islice(index_dict.items(), n)))
    columns = pdi.from_dict(dict(islice(columns_dict.items(), m)))

    df = pd.DataFrame(range2d(2**n, 2**m), index=index, columns=columns)
    if ascending is False:
        df.sort_index(ascending=False, inplace=True, axis=0)
        df.sort_index(ascending=False, inplace=True, axis=1)
    return df

def gen_df1(n, m, ascending=True):
    return pd.DataFrame(range2d(n, m), index=list('abcdef')[:n], columns=list('ABCDEF')[:m])

def vi(s):
    return s.values.tolist(), s.index.to_list()

def vic(s):
    return s.values.tolist(), s.index.to_list(), s.columns.to_list()

def vin(s):
    return s.values.tolist(), s.index.to_list(), s.name

def vicn(df):
    assert isinstance(df, NDFrame)      # Frame or Series
    return (
        df.fillna(inf).values.tolist(),
        df.index.to_list(),
        df.columns.to_list(),
        [list(df.index.names), list(df.columns.names)],
    )
