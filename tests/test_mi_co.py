import pytest
import numpy as np
import pandas as pd

from pdi import patch_dataframe
import pdi

def range2d(n, m):
    return np.arange(1, n * m + 1).reshape(n, m)

def test_patch_series():
    patch_dataframe()

    df = pd.DataFrame(
        range2d(8, 8),
        index=pdi.from_dict(
            {
                "k": ("b", "a"),
                "l": ("d", "c"),
                "m": ("f", "e"),
            }
        ),
        columns=pdi.from_dict(
            {
                "K": ("B", "A"),
                "L": ("D", "C"),
                "M": ("F", "E"),
            }
        ),
    )  # 3Lx3L

    assert df.mi['a','d','f'].tolist() == \
            [33, 34, 35, 36, 37, 38, 39, 40]

    assert df.mi[:,'d','f'].values.tolist() == \
            [[1, 2, 3, 4, 5, 6, 7, 8], [33, 34, 35, 36, 37, 38, 39, 40]]

    assert df.mi[:,'d',:].index.tolist() == \
            [('b', 'd', 'f'), ('b', 'd', 'e'), ('a', 'd', 'f'), ('a', 'd', 'e')]

    assert df.co['B','C','F'].values.tolist() == \
            [3, 11, 19, 27, 35, 43, 51, 59]

    assert df.co[:,'C','F'].columns.tolist() == \
            [('B', 'C', 'F'), ('A', 'C', 'F')]

    assert df.co[:,'C',:].columns.tolist() == \
            [('B', 'C', 'F'), ('B', 'C', 'E'), ('A', 'C', 'F'), ('A', 'C', 'E')]
    
if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
