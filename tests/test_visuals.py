import pytest
import pandas as pd
from pdi import patch_series_repr, unpatch_series_repr, sidebyside
from pdi.testing import gen_df

MARKER = "border-right: 1px solid"


def test_patch_series_repr():
    assert not hasattr(pd.Series([1, 2, 3]), "to_html")
    assert not hasattr(pd.Series([1, 2, 3]), "_repr_html_")

    patch_series_repr()

    assert MARKER in pd.Series([1, 2, 3]).to_html()
    assert MARKER in pd.Series([1, 2, 3])._repr_html_()
    assert MARKER in pd.Series([1, 2, 3], name="A").to_html()
    assert MARKER in pd.Series([1, 2, 3], name="A")._repr_html_()

    unpatch_series_repr()

    assert not hasattr(pd.Series([1, 2, 3]), "to_html")
    assert not hasattr(pd.Series([1, 2, 3]), "_repr_html_")


def test_sidebyside(mocker):
    def gen_display_html(valign):
        def display_html(html, raw):
            assert "vertical-align:" + valign in html
            for v in [11, 22, 33, 44]:
                assert str(v) in html
            assert raw is True

        return display_html

    df = gen_df(1, 1) * 11
    mocker.patch("pdi.visuals.display_html", gen_display_html("top"))
    sidebyside(pd.Series([11, 22, 33, 44]))
    sidebyside(df, df.A, df.index)
    sidebyside(df, df.A, df.index, names=["aa", "bb", "cc"])
    mocker.patch("pdi.visuals.display_html", gen_display_html("bottom"))
    sidebyside(df, df.A, df.index, valign="bottom")


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
