from itertools import islice

import pandas as pd
import numpy as np
import pdi

def range2d(n, m):
    return np.arange(1, n * m + 1).reshape(n, m)


def gen_df(n, m):
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

    return pd.DataFrame(range2d(2**n, 2**m), index=index, columns=columns)
