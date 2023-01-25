from IPython.display import display_html

import pandas as pd
import pandas.io.formats.format as fmt

def _fix_html(html):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    soup.tbody.find_all("th")[0].style
    soup.thead["style"] = "border-bottom: none;"
    for tr in soup.tbody.find_all("tr"):
        th = tr.find_all("th")[-1]
        th["style"] = "border-right: 1px solid #666;"
    #    html = html.replace('<th>', '<th style="border-right: 1px solid black;">')  # #009dff, yellowgreen
    #    html_list.append(html)
    return str(soup.body.decode_contents())


def patch_series_repr(footer=True):
    """
    Improves visual representation of the Series in Jupyter notebooks
    """

    def get_footer(s):
        if hasattr(fmt, 'get_series_repr_params'):
            repr_params = fmt.get_series_repr_params()
        else:       # pandas < 1.4
            from pandas._config import get_option
            from shutil import get_terminal_size
            
            width, height = get_terminal_size()
            repr_params = dict(
                max_rows = (
                    height
                    if get_option("display.max_rows") == 0
                    else get_option("display.max_rows")
                ),
                min_rows = (
                    height
                    if get_option("display.max_rows") == 0
                    else get_option("display.min_rows")
                ),
                length = get_option("display.show_dimensions"),
            )
        msg = fmt.SeriesFormatter(s, **repr_params)._get_footer()
        return f'<pre style="margin-top:3px">{msg}</pre>'

    def _repr_html_(s):
        if s.name is None:
            s1 = s.copy()
            s1.name = ""
        else:
            s1 = s
        html = s1.to_frame()._repr_html_()
        if footer:
            html += get_footer(s)
        return _fix_html(html)

    def to_html(s, *args, **kwargs):
        if s.name is None:
            s1 = s.copy()
            s1.name = ""
        else:
            s1 = s
        html = s1.to_frame().to_html(*args, **kwargs)
        if footer:
            html += get_footer(s)
        return _fix_html(html)

    pd.Series._repr_html_ = _repr_html_
    pd.Series.to_html = to_html


def unpatch_series_repr():
    """
    Reverts changes applied by patch_series()
    """
    if hasattr(pd.Series, '_repr_html_'):
        del pd.Series._repr_html_
    if hasattr(pd.Series, 'to_html'):
        del pd.Series.to_html


def sidebyside(*dfs, names=[], index=True, valign="top"):
    """
    Displays several DataFrames (or Series, or Indices) side by side.

    Optionally display names above each.
    Index can be omitted with index=False.
    Normally vertically aligned tables look good. If the objects
    of the same length are misaligned due to different header height,
    valign='bottom' results in a better alignment.
    Eg: sidebyside(df1, df2, names=['df1', 'df2'], index=False)
    """

    def to_html(x, index: bool):
        if isinstance(x, pd.Index):
            x = pd.DataFrame([""] * len(x), index=x, columns=[""])
            return x.to_html().replace("<th></th>", "<th>&nbsp;</th>")

        if isinstance(x, pd.DataFrame):
            return x.to_html(index=index)

        # x is expected to be a Series here
        if x.name is None:
            x = x.copy()
            x.name = ""
        html = x.to_frame().to_html(index=index)
        return _fix_html(html) if index else html

    html_str = ""
    if names:
        html_str += (
            "<tr>"
            + "".join(
                '<td style="text-align:center; font-weight:bold">'  # noqa: E131
                f"{name}"
                "</td>"
                for name in names
            )
            + "</tr>"
        )
    html_str += (
        "<tr>"
        + "".join(
            f'<td style="vertical-align:{valign}"> '  # noqa: E131
            f"{to_html(df, index=index)}"
            "</td>"
            for df in dfs
        )
        + "</tr>"
    )
    html_str = f"<table>{html_str}</table>"
    html_str = html_str.replace("table", 'table style="display:inline"')
    display_html(html_str, raw=True)

sbs = sidebyside
