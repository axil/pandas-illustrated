import pytest
import numpy as np
import pandas as pd

from pdi import find, findall
from pdi.testing import gen_df1


def test_small():
    s = pd.Series([4, 2, 4, 6], index=["cat", "penguin", "dog", "butterfly"])

    assert find(s, 2) == "penguin"
    assert find(s, 2, pos=True) == 1

    assert find(s, 4) == "cat"
    assert find(s, 4, pos=True) == 0

    with pytest.raises(ValueError) as exinfo:
        find(s, 3)
    assert str(exinfo.value) == "3 is not in Series"

    with pytest.raises(ValueError) as exinfo:
        find(s, 3, pos=True)
    assert str(exinfo.value) == "3 is not in Series"


def test_large():
    n = 10**4
    a = np.arange(n)
    np.random.shuffle(a)
    s = pd.Series(a, index=np.arange(1, len(a) + 1))
    x = a[-1]

    assert find(s, x) == n
    assert find(s, x, pos=True) == n - 1

    with pytest.raises(ValueError) as exinfo:
        find(s, n)
    assert str(exinfo.value) == f"{n} is not in Series"

    with pytest.raises(ValueError) as exinfo:
        find(s, n, pos=True)
    assert str(exinfo.value) == f"{n} is not in Series"


def test_strings():
    s = pd.Series(["dog", "cat", "penguin", "cat"], index=list(range(1, 5)))

    assert find(s, "dog") == 1
    assert find(s, "dog", pos=True) == 0

    assert find(s, "cat") == 2
    assert find(s, "cat", pos=True) == 1

    assert findall(s, "cat").tolist() == [2, 4]
    assert findall(s, "cat", pos=True).tolist() == [1, 3]

    x = "butterfly"
    with pytest.raises(ValueError) as exinfo:
        find(s, x)
    assert str(exinfo.value) == f"{x!r} is not in Series"

    x = "butterfly"
    with pytest.raises(ValueError) as exinfo:
        find(s, x, pos=True)
    assert str(exinfo.value) == f"{x!r} is not in Series"


def test_ndarray():
    a = np.array([1, 2, 3])
    assert find(a, 3) == 2
    with pytest.raises(ValueError):
        assert find(a, 4)


def test_types():
    df = gen_df1(3, 3)
    with pytest.raises(TypeError):
        find(df, 1)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
