import pytest
import pandas as pd
import pandas_illustrated as pdi

def test1():
    s = pd.Series([4,2,4,6], index=['cat', 'penguin', 'dog', 'butterfly'])
    assert pdi.find(s, 2) == 'penguin'
    assert pdi.find(s, 4) == 'cat'
    with pytest.raises(ValueError):
        pdi.find(s, 3)


if __name__ == '__main__':
    pytest.main(['-s', __file__])
