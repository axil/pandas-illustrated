import pytest
import numpy as np
from numpy import inf
import pandas as pd
from pandas._libs import lib

from pdi import from_dict, from_kw
from pdi.testing import range2d, gen_df1, vn


def test1():
    mi0 = pd.MultiIndex.from_product(
        [list("AB"), list("CDE"), list("FGHI")], names=list("KLM")
    )
    mi1 = from_kw(K=tuple("AB"), L=tuple("CDE"), M=tuple("FGHI"))
    mi2 = from_dict(dict(K=tuple("AB"), L=tuple("CDE"), M=tuple("FGHI")))
    assert vn(mi0) == vn(mi1)
    assert vn(mi0) == vn(mi2)


def test2():
    mi0 = pd.MultiIndex.from_tuples([tuple("ABC"), tuple("DEF")], names=list("KLM"))
    mi1 = from_kw(K=list("AD"), L=list("BE"), M=list("CF"))
    mi1
    mi2 = from_dict(dict(K=list("AD"), L=list("BE"), M=list("CF")))
    assert vn(mi0) == vn(mi1)
    assert vn(mi0) == vn(mi2)


if __name__ == "__main__":
    pytest.main(["-x", "-s", __file__])  # + '::test7'])
    # pytest.main(["-x", "-s", __file__ + '::test_df_numeric_named_series'])
    input()
