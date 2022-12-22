﻿import pytest
import numpy as np
import pandas as pd
from pandas_illustrated import find, findall

def test_small():
    s = pd.Series([4, 2, 4, 6], index=['cat', 'penguin', 'dog', 'butterfly'])

    assert find(s, 2) == 'penguin'
    assert find(s, 2, pos=True) == 1

    assert find(s, 4) == 'cat'
    assert find(s, 4, pos=True) == 0

    with pytest.raises(ValueError) as exinfo:
        find(s, 3)
    assert str(exinfo.value) == '3 is not in Series'

    with pytest.raises(ValueError) as exinfo:
        find(s, 3, pos=True)
    assert str(exinfo.value) == '3 is not in Series'
    
def test_large():
    n = 10**4
    a = np.arange(n)
    np.random.shuffle(a)
    s = pd.Series(a, index=np.arange(1, len(a)+1))
    x = a[-1]
    
    assert find(s, x) == n
    assert find(s, x, pos=True) == n-1
    
    with pytest.raises(ValueError) as exinfo:
        find(s, n)
    assert str(exinfo.value) == f'{n} is not in Series'

    with pytest.raises(ValueError) as exinfo:
        find(s, n, pos=True)
    assert str(exinfo.value) == f'{n} is not in Series'

def test_strings():
    s = pd.Series(['dog', 'cat', 'penguin', 'cat'], index=list(range(1, 5)))
    
    assert find(s, 'dog') == 1
    assert find(s, 'dog', pos=True) == 0

    assert find(s, 'cat') == 2
    assert find(s, 'cat', pos=True) == 1

    x = 'butterfly'
    with pytest.raises(ValueError) as exinfo:
        find(s, x)
    assert str(exinfo.value) == f'{x!r} is not in Series'

    x = 'butterfly'
    with pytest.raises(ValueError) as exinfo:
        find(s, x, pos=True)
    assert str(exinfo.value) == f'{x!r} is not in Series'

if __name__ == '__main__':
    pytest.main(['-s', __file__])
