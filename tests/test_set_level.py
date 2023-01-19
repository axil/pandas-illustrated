from math import inf

import pytest
import pandas as pd

from pandas.core.generic import NDFrame
from pandas._libs import lib

from pdi import set_level
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


def test_set_level_11():
    df = gen_df(1, 2)

    with pytest.raises(TypeError):
        set_level(df.index, 0, "z")
    
    with pytest.raises(TypeError):
        set_level(df, 0, "z", axis=0)

    df = gen_df(2, 1)

    with pytest.raises(TypeError):
        set_level(df.columns, 0, "z")
    
    with pytest.raises(TypeError):
        set_level(df, 0, "z", axis=1)


Q_RESULT = [
    ([("Q", "C"), ("Q", "D"), ("Q", "C"), ("Q", "D")], ["K", "L"]),
    ([("A", "Q"), ("A", "Q"), ("B", "Q"), ("B", "Q")], ["K", "L"]),
]
Q_RESULT_M = [
    ([("Q", "C"), ("Q", "D"), ("Q", "C"), ("Q", "D")], ["M", "L"]),
    ([("A", "Q"), ("A", "Q"), ("B", "Q"), ("B", "Q")], ["K", "M"]),
]
Q_RESULT_NONE = [
    ([("Q", "C"), ("Q", "D"), ("Q", "C"), ("Q", "D")], [None, "L"]),
    ([("A", "Q"), ("A", "Q"), ("B", "Q"), ("B", "Q")], ["K", None]),
]
QRST = ["Q", "R", "S", "T"]
QRST_RESULT = [
    ([("Q", "C"), ("R", "D"), ("S", "C"), ("T", "D")], ["K", "L"]),
    ([("A", "Q"), ("A", "R"), ("B", "S"), ("B", "T")], ["K", "L"]),
]
QRST_RESULT_M = [
    ([("Q", "C"), ("R", "D"), ("S", "C"), ("T", "D")], ["M", "L"]),
    ([("A", "Q"), ("A", "R"), ("B", "S"), ("B", "T")], ["K", "M"]),
]
QRST_RESULT_NONE = [
    ([("Q", "C"), ("R", "D"), ("S", "C"), ("T", "D")], [None, "L"]),
    ([("A", "Q"), ("A", "R"), ("B", "S"), ("B", "T")], ["K", None]),
]

@pytest.mark.parametrize("a, name, result", [
    ("Q", "M", Q_RESULT_M),
    ("Q", None, Q_RESULT_NONE),
    ("Q", lib.no_default, Q_RESULT),
    
    (QRST, "M", QRST_RESULT_M),
    (pd.Series(QRST), "M", QRST_RESULT_M),
    (pd.Index(QRST), "M", QRST_RESULT_M),
    (pd.Series(QRST, name="Z"), "M", QRST_RESULT_M),
    (pd.Index(QRST, name="Z"), "M", QRST_RESULT_M),

    (QRST, None, QRST_RESULT_NONE),
    (pd.Series(QRST), None, QRST_RESULT_NONE),
    (pd.Index(QRST), None, QRST_RESULT_NONE),
    (pd.Series(QRST, name="M"), None, QRST_RESULT_NONE),
    (pd.Index(QRST, name="M"), None, QRST_RESULT_NONE),
    
    (QRST, lib.no_default, QRST_RESULT),
    (pd.Series(QRST), lib.no_default, QRST_RESULT_NONE),
    (pd.Index(QRST), lib.no_default, QRST_RESULT_NONE),
    (pd.Series(QRST, name="M"), lib.no_default, QRST_RESULT_M),
    (pd.Index(QRST, name="M"), lib.no_default, QRST_RESULT_M),
])
def test_mi_12_QRST(a, name, result):
    # mi
    df0 = gen_df(1, 2)
    orig = vicn(df0)

    for i in range(2):
        df = df0.copy()
        set_level(df.columns, i, a, name=name, inplace=True)
        assert vn(df.columns) == result[i], i

    for i, i_name in enumerate("KL"):
        df = df0.copy()
        set_level(df.columns, i_name, a, name=name, inplace=True)
        assert vn(df.columns) == result[i], i

    for i in range(2):
        df = df0.copy()
        idx = set_level(df.columns, i, a, name=name, inplace=False)
        assert vn(idx) == result[i], i
        assert vicn(df) == orig, i

    for i, i_name in enumerate("KL"):
        df = df0.copy()
        idx = set_level(df.columns, i_name, a, name=name, inplace=False)
        assert vn(idx) == result[i], i
        assert vicn(df) == orig, i
    
    # df, axis=1
    for i in range(2):
        df = df0.copy()
        set_level(df, i, a, name=name, axis=1, inplace=True)
        assert vn(df.columns) == result[i], i

    for i, i_name in enumerate("KL"):
        df = df0.copy()
        set_level(df, i_name, a, name=name, axis=1, inplace=True)
        assert vn(df.columns) == result[i], i

    for i in range(2):
        df = df0.copy()
        df1 = set_level(df, i, a, name=name, axis=1, inplace=False)
        assert vn(df1.columns) == result[i], i
        assert vicn(df) == orig, i

    for i, i_name in enumerate("KL"):
        df = df0.copy()
        df1 = set_level(df, i_name, a, name=name, axis=1, inplace=False)
        assert vn(df1.columns) == result[i], i
        assert vicn(df) == orig, i

    # df, axis=0
    df0 = gen_df(1, 2).T
    orig = vicn(df0)

    for i in range(2):
        df = df0.copy()
        set_level(df, i, a, name=name, axis=0, inplace=True)
        assert vn(df.index) == result[i], i

    for i, i_name in enumerate("KL"):
        df = df0.copy()
        set_level(df, i_name, a, name=name, axis=0, inplace=True)
        assert vn(df.index) == result[i], i

    for i in range(2):
        df = df0.copy()
        df1 = set_level(df, i, a, name=name, axis=0, inplace=False)
        assert vn(df1.index) == result[i], i
        assert vicn(df) == orig, i

    for i, i_name in enumerate("KL"):
        df = df0.copy()
        df1 = set_level(df, i_name, a, name=name, axis=0, inplace=False)
        assert vn(df1.index) == result[i], i
        assert vicn(df) == orig, i

    # s
    s0 = gen_df(1, 2).loc['a']
    orig = vin(s0)

    for i in range(2):
        s = s0.copy()
        set_level(s, i, a, name=name, axis=0, inplace=True)
        assert vn(s.index) == result[i], i

    for i, i_name in enumerate("KL"):
        s = s0.copy()
        set_level(s, i_name, a, name=name, axis=0, inplace=True)
        assert vn(s.index) == result[i], i

    for i in range(2):
        s = s0.copy()
        s1 = set_level(s, i, a, name=name, axis=0, inplace=False)
        assert vn(s1.index) == result[i], i
        assert vin(s) == orig, i

    for i, i_name in enumerate("KL"):
        s = s0.copy()
        s1 = set_level(s, i_name, a, name=name, axis=0, inplace=False)
        assert vn(s1.index) == result[i], i
        assert vin(s) == orig, i


if __name__ == "__main__":
    pytest.main(["-x", "-s", __file__])  # + '::test7'])
