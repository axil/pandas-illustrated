import pytest
import pandas as pd
from pdi import patch_series, unpatch_series, sidebyside

MARKER = "border-right: 1px solid"

def test_patch_series():
    assert not hasattr(pd.Series([1,2,3]), 'to_html')
    assert not hasattr(pd.Series([1,2,3]), '_repr_html_')
    
    patch_series()
    
    assert MARKER in pd.Series([1,2,3]).to_html()
    assert MARKER in pd.Series([1,2,3])._repr_html_()

    unpatch_series()
    
    assert not hasattr(pd.Series([1,2,3]), 'to_html')
    assert not hasattr(pd.Series([1,2,3]), '_repr_html_')

def test_sidebyside(mocker):
    def gen_display_html(valign):
        def display_html(html, raw):
            assert 'vertical-align:' + valign in html
            for v in [11, 22, 33, 44, 55, 66]:
                assert str(v) in html
            assert raw is True
        return display_html

    mocker.patch('pdi.visuals.display_html', gen_display_html('top'))
    sidebyside(pd.DataFrame([[11,22],[33,44]]), pd.Series([55,66]))
    mocker.patch('pdi.visuals.display_html', gen_display_html('bottom'))
    sidebyside(pd.DataFrame([[11,22],[33,44]]), pd.Series([55,66]), valign='bottom')

if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
