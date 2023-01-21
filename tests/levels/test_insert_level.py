import pytest
import numpy as np
import pandas as pd

from pdi import insert_level
from pdi.testing import gen_df, gen_df1, vn, vic, vin, vicn


RESULTS_SCALAR = \
[([('Q', 'A', 'C'), ('Q', 'A', 'D'), ('Q', 'B', 'C'), ('Q', 'B', 'D')],
  ['M', 'K', 'L']),
 ([('A', 'Q', 'C'), ('A', 'Q', 'D'), ('B', 'Q', 'C'), ('B', 'Q', 'D')],
  ['K', 'M', 'L']),
 ([('A', 'C', 'Q'), ('A', 'D', 'Q'), ('B', 'C', 'Q'), ('B', 'D', 'Q')],
  ['K', 'L', 'M'])]

RESULTS_SCALAR_NONAME = \
[([('Q', 'A', 'C'), ('Q', 'A', 'D'), ('Q', 'B', 'C'), ('Q', 'B', 'D')],
  [None, 'K', 'L']),
 ([('A', 'Q', 'C'), ('A', 'Q', 'D'), ('B', 'Q', 'C'), ('B', 'Q', 'D')],
  ['K', None, 'L']),
 ([('A', 'C', 'Q'), ('A', 'D', 'Q'), ('B', 'C', 'Q'), ('B', 'D', 'Q')],
  ['K', 'L', None])]


@pytest.mark.parametrize("name, results", [
    ("M", RESULTS_SCALAR),
    (None, RESULTS_SCALAR_NONAME),
])
def test_scalar_mi(name, results):
    df0 = gen_df(1, 2)
    orig = vicn(df0)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        df = df0.copy()
        idx = insert_level(df.columns, i, "Q", name=name, inplace=False)
        assert vicn(df) == orig, i
        assert vn(idx) == results[i], i

    df = df0.copy()
    with pytest.raises(IndexError):
        insert_level(df.columns, -5, "Q", name=name, inplace=False)

    #   - by name
    for i, i_name in enumerate("KL"):
        df = df0.copy()
        idx = insert_level(df.columns, i_name, "Q", name=name, inplace=False)
        assert vicn(df) == orig, i
        assert vn(idx) == results[i], i

    df = df0.copy()
    with pytest.raises(KeyError):
        insert_level(df.columns, "Z", "Q", name=name, inplace=False)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        df = df0.copy()
        insert_level(df.columns, i, "Q", name=name, inplace=True)
        assert vn(df.columns) == results[i], i

    df = df0.copy()
    with pytest.raises(IndexError):
        insert_level(df.columns, -5, "Q", name=name, inplace=True)

    #   - by name
    for i, i_name in enumerate("KL"):
        df = df0.copy()
        insert_level(df.columns, i_name, "Q", name=name, inplace=True)
        assert vn(df.columns) == results[i], i

    df = df0.copy()
    with pytest.raises(KeyError):
        insert_level(df.columns, "Z", "Q", name=name, inplace=True)


@pytest.mark.parametrize("name, results", [
    ("M", RESULTS_SCALAR),
    (None, RESULTS_SCALAR_NONAME),
])
def test_scalar_df_col(name, results):
    df0 = gen_df(1, 2)
    orig = vicn(df0)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        df = df0.copy()
        df1 = insert_level(df, i, "Q", name=name, axis=1, inplace=False)
        assert vicn(df) == orig, i
        assert vn(df1.columns) == results[i], i

    df = df0.copy()
    with pytest.raises(IndexError):
        insert_level(df, -5, "Q", name=name, axis=1, inplace=False)

    #   - by name
    for i, i_name in enumerate("KL"):
        df = df0.copy()
        df1 = insert_level(df, i_name, "Q", name=name, axis=1, inplace=False)
        assert vicn(df) == orig, i
        assert vn(df1.columns) == results[i], i

    df = df0.copy()
    with pytest.raises(KeyError):
        insert_level(df, "Z", "Q", name=name, axis=1, inplace=False)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        df = df0.copy()
        insert_level(df, i, "Q", name=name, axis=1, inplace=True)
        assert vn(df.columns) == results[i], i

    df = df0.copy()
    with pytest.raises(IndexError):
        insert_level(df, -5, "Q", name=name, axis=1, inplace=True)

    #   - by name
    for i, i_name in enumerate("KL"):
        df = df0.copy()
        insert_level(df, i_name, "Q", name=name, axis=1, inplace=True)
        assert vn(df.columns) == results[i], i

    df = df0.copy()
    with pytest.raises(KeyError):
        insert_level(df, "Z", "Q", name=name, axis=1, inplace=True)


@pytest.mark.parametrize("name, results", [
    ("M", RESULTS_SCALAR),
    (None, RESULTS_SCALAR_NONAME),
])
def test_scalar_df_row(name, results):
    df0 = gen_df(1, 2).T
    orig = vicn(df0)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        df = df0.copy()
        df1 = insert_level(df, i, "Q", name="M", axis=0, inplace=False)
        assert vicn(df) == orig, i
        assert vn(df1.index) == RESULTS_SCALAR[i], i

    df = df0.copy()
    with pytest.raises(IndexError):
        insert_level(df, -5, "Q", name="M", axis=0, inplace=False)

    #   - by name
    for i, name in enumerate("KL"):
        df = df0.copy()
        df1 = insert_level(df, name, "Q", name="M", axis=0, inplace=False)
        assert vicn(df) == orig, i
        assert vn(df1.index) == RESULTS_SCALAR[i], i

    df = df0.copy()
    with pytest.raises(KeyError):
        insert_level(df, "Z", "Q", name="M", axis=0, inplace=False)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        df = df0.copy()
        insert_level(df, i, "Q", name="M", axis=0, inplace=True)
        assert vn(df.index) == RESULTS_SCALAR[i], i

    df = df0.copy()
    with pytest.raises(IndexError):
        insert_level(df, -5, "Q", name="M", axis=0, inplace=True)

    #   - by name
    for i, name in enumerate("KL"):
        df = df0.copy()
        insert_level(df, name, "Q", name="M", axis=0, inplace=True)
        assert vn(df.index) == RESULTS_SCALAR[i], i

    df = df0.copy()
    with pytest.raises(KeyError):
        insert_level(df, "Z", "Q", name="M", axis=0, inplace=True)

@pytest.mark.parametrize("name, results", [
    ("M", RESULTS_SCALAR),
    (None, RESULTS_SCALAR_NONAME),
])
def test_scalar_s(name, results):
    s0 = gen_df(1, 2).loc['a']
    orig = vin(s0)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        s = s0.copy()
        df1 = insert_level(s, i, "Q", name=name, axis=0, inplace=False)
        assert vin(s) == orig, i
        assert vn(df1.index) == results[i], i

    s = s0.copy()
    with pytest.raises(IndexError):
        insert_level(s, -5, "Q", name=name, axis=0, inplace=False)

    #   - by name
    for i, i_name in enumerate("KL"):
        s = s0.copy()
        df1 = insert_level(s, i_name, "Q", name=name, axis=0, inplace=False)
        assert vin(s) == orig, i
        assert vn(df1.index) == results[i], i

    s = s0.copy()
    with pytest.raises(KeyError):
        insert_level(s, "Z", "Q", name=name, axis=0, inplace=False)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        s = s0.copy()
        insert_level(s, i, "Q", name=name, axis=0, inplace=True)
        assert vn(s.index) == results[i], i

    s = s0.copy()
    with pytest.raises(IndexError):
        insert_level(s, -5, "Q", name=name, axis=0, inplace=True)

    #   - by name
    for i, i_name in enumerate("KL"):
        s = s0.copy()
        insert_level(s, i_name, "Q", name=name, axis=0, inplace=True)
        assert vn(s.index) == results[i], i

    s = s0.copy()
    with pytest.raises(KeyError):
        insert_level(s, "Z", "Q", name=name, axis=0, inplace=True)


def test_scalar_s_noname():
    s0 = gen_df(1, 2).loc['a']
    orig = vin(s0)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        s = s0.copy()
        df1 = insert_level(s, i, "Q", axis=0, inplace=False)
        assert vin(s) == orig, i
        assert vn(df1.index) == RESULTS_SCALAR_NONAME[i], i

    s = s0.copy()
    with pytest.raises(IndexError):
        insert_level(s, -5, "Q", axis=0, inplace=False)

    #   - by name
    for i, name in enumerate("KL"):
        s = s0.copy()
        df1 = insert_level(s, name, "Q", axis=0, inplace=False)
        assert vin(s) == orig, i
        assert vn(df1.index) == RESULTS_SCALAR_NONAME[i], i

    s = s0.copy()
    with pytest.raises(KeyError):
        insert_level(s, "Z", "Q", axis=0, inplace=False)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        s = s0.copy()
        insert_level(s, i, "Q", axis=0, inplace=True)
        assert vn(s.index) == RESULTS_SCALAR_NONAME[i], i

    s = s0.copy()
    with pytest.raises(IndexError):
        insert_level(s, -5, "Q", axis=0, inplace=True)

    #   - by name
    for i, name in enumerate("KL"):
        s = s0.copy()
        insert_level(s, name, "Q", axis=0, inplace=True)
        assert vn(s.index) == RESULTS_SCALAR_NONAME[i], i

    s = s0.copy()
    with pytest.raises(KeyError):
        insert_level(s, "Z", "Q", axis=0, inplace=True)

@pytest.mark.parametrize("name, results", [
    ("M", RESULTS_SCALAR),
    (None, RESULTS_SCALAR_NONAME),
])
def test_scalar_s_sort(name, results):
    s0 = gen_df(1, 2, ascending=False).loc['a']
    orig = vin(s0)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        s = s0.copy()
        df1 = insert_level(s, i, "Q", name=name, axis=0, inplace=False, sort=True)
        assert vin(s) == orig, i
        assert vn(df1.index) == results[i], i

    s = s0.copy()
    with pytest.raises(IndexError):
        insert_level(s, -5, "Q", name=name, axis=0, inplace=False, sort=True)

    #   - by name
    for i, i_name in enumerate("KL"):
        s = s0.copy()
        df1 = insert_level(s, i_name, "Q", name=name, axis=0, inplace=False, sort=True)
        assert vin(s) == orig, i
        assert vn(df1.index) == results[i], i

    s = s0.copy()
    with pytest.raises(KeyError):
        insert_level(s, "Z", "Q", name=name, axis=0, inplace=False, sort=True)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        s = s0.copy()
        insert_level(s, i, "Q", name=name, axis=0, inplace=True, sort=True)
        assert vn(s.index) == results[i], i

    s = s0.copy()
    with pytest.raises(IndexError):
        insert_level(s, -5, "Q", name=name, axis=0, inplace=True, sort=True)

    #   - by name
    for i, i_name in enumerate("KL"):
        s = s0.copy()
        insert_level(s, i_name, "Q", name=name, axis=0, inplace=True, sort=True)
        assert vn(s.index) == results[i], i

    s = s0.copy()
    with pytest.raises(KeyError):
        insert_level(s, "Z", "Q", name=name, axis=0, inplace=True, sort=True)

RESULTS_ARRAY = \
[([('Q', 'A', 'C'), ('R', 'A', 'D'), ('S', 'B', 'C'), ('T', 'B', 'D')],
  ['M', 'K', 'L']),
 ([('A', 'Q', 'C'), ('A', 'R', 'D'), ('B', 'S', 'C'), ('B', 'T', 'D')],
  ['K', 'M', 'L']),
 ([('A', 'C', 'Q'), ('A', 'D', 'R'), ('B', 'C', 'S'), ('B', 'D', 'T')],
  ['K', 'L', 'M'])]

RESULTS_ARRAY_NONAME = \
[([('Q', 'A', 'C'), ('R', 'A', 'D'), ('S', 'B', 'C'), ('T', 'B', 'D')],
  [None, 'K', 'L']),
 ([('A', 'Q', 'C'), ('A', 'R', 'D'), ('B', 'S', 'C'), ('B', 'T', 'D')],
  ['K', None, 'L']),
 ([('A', 'C', 'Q'), ('A', 'D', 'R'), ('B', 'C', 'S'), ('B', 'D', 'T')],
  ['K', 'L', None])]


ARRAY = ["Q", "R", "S", "T"]
@pytest.mark.parametrize("a,name,results", [
    (ARRAY, "M", RESULTS_ARRAY),
    (np.array(ARRAY), "M", RESULTS_ARRAY),
    (pd.Series(ARRAY), "M", RESULTS_ARRAY),
    (pd.Index(ARRAY), "M", RESULTS_ARRAY),
    (ARRAY, None, RESULTS_ARRAY_NONAME),
    (np.array(ARRAY), None, RESULTS_ARRAY_NONAME),
    (pd.Series(ARRAY), None, RESULTS_ARRAY_NONAME),
    (pd.Index(ARRAY), None, RESULTS_ARRAY_NONAME),

])
def test_array_s(a, name, results):
    s0 = gen_df(1, 2).loc['a']
    orig = vin(s0)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        s = s0.copy()
        df1 = insert_level(s, i, a, name=name, axis=0, inplace=False)
        assert vin(s) == orig, i
        assert vn(df1.index) == results[i], i

    s = s0.copy()
    with pytest.raises(IndexError):
        insert_level(s, -5, a, name=name, axis=0, inplace=False)

    #   - by name
    for i, i_name in enumerate("KL"):
        s = s0.copy()
        df1 = insert_level(s, i_name, a, name=name, axis=0, inplace=False)
        assert vin(s) == orig, i
        assert vn(df1.index) == results[i], i

    s = s0.copy()
    with pytest.raises(KeyError):
        insert_level(s, "Z", a, name=name, axis=0, inplace=False)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        s = s0.copy()
        insert_level(s, i, a, name=name, axis=0, inplace=True)
        assert vn(s.index) == results[i], i

    s = s0.copy()
    with pytest.raises(IndexError):
        insert_level(s, -5, a, name=name, axis=0, inplace=True)

    #   - by name
    for i, i_name in enumerate("KL"):
        s = s0.copy()
        insert_level(s, i_name, a, name=name, axis=0, inplace=True)
        assert vn(s.index) == results[i], i

    s = s0.copy()
    with pytest.raises(KeyError):
        insert_level(s, "Z", a, name=name, axis=0, inplace=True)

@pytest.mark.parametrize("a, results", [
    (ARRAY, RESULTS_ARRAY_NONAME),
    (np.array(ARRAY), RESULTS_ARRAY_NONAME),
    (pd.Series(ARRAY), RESULTS_ARRAY_NONAME),
    (pd.Index(ARRAY), RESULTS_ARRAY_NONAME),
    (pd.Series(ARRAY, name="M"), RESULTS_ARRAY),
    (pd.Index(ARRAY, name="M"), RESULTS_ARRAY),
])
def test_array_s_noname(a, results):
    s0 = gen_df(1, 2).loc['a']
    orig = vin(s0)

    # * inplace = False
    #   - by positional index
    for i in range(3):
        s = s0.copy()
        df1 = insert_level(s, i, a, axis=0, inplace=False)
        assert vin(s) == orig, i
        assert vn(df1.index) == results[i], i

    s = s0.copy()
    with pytest.raises(IndexError):
        insert_level(s, -5, a, axis=0, inplace=False)

    #   - by name
    for i, name in enumerate("KL"):
        s = s0.copy()
        df1 = insert_level(s, name, a, axis=0, inplace=False)
        assert vin(s) == orig, i
        assert vn(df1.index) == results[i], i

    s = s0.copy()
    with pytest.raises(KeyError):
        insert_level(s, "Z", a, axis=0, inplace=False)

    # * inplace = True
    #   - by positional index
    for i in range(3):
        s = s0.copy()
        insert_level(s, i, a, axis=0, inplace=True)
        assert vn(s.index) == results[i], i

    s = s0.copy()
    with pytest.raises(IndexError):
        insert_level(s, -5, a, axis=0, inplace=True)

    #   - by name
    for i, name in enumerate("KL"):
        s = s0.copy()
        insert_level(s, name, a, axis=0, inplace=True)
        assert vn(s.index) == results[i], i

    s = s0.copy()
    with pytest.raises(KeyError):
        insert_level(s, "Z", a, axis=0, inplace=True)

def test_insert_into_simple_index():
    df = gen_df1(3,3); df
    df1 = insert_level(df, 1, ['D','E','F'], 'L')
    assert vic(df1) == \
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ['a', 'b', 'c'],
    [('A', 'D'), ('B', 'E'), ('C', 'F')])

    with pytest.raises(ValueError):
        insert_level(df.columns, 1, ['D','E','F'], 'L', inplace=True)

def test_incorrect_types():
    with pytest.raises(TypeError):
        insert_level(np.array([1,2,3]), 0, 'L')
    
    df = gen_df1(3, 3)
    with pytest.raises(TypeError):
        insert_level(df, 1, {'a': 10})

def test_incorrect_sizes():
    df = gen_df1(3, 3)
    with pytest.raises(ValueError):
        insert_level(df, 1, ['D','E'], 'L')

def test_mi_inplace_sort():
    df = gen_df(1,2)
    with pytest.raises(ValueError):
        insert_level(df.columns, 0, list('EFGH'), inplace=True, sort=True)

if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
