from IPython.display import display_html

def sidebyside(*dfs, names=[], index=True):
    def to_df(x):
        if isinstance(x, pd.Series):
            return x.to_frame()
        elif isinstance(x, pd.Index):
            name = x.name if x.name is not None else 'index'
            return pd.Series(x, name=name).to_frame()
        else:
            return x
    html_str = ''
    if names:
        html_str += (
            '<tr>' + 
                ''.join('<td style="text-align:center; font-weight:bold">'
                            f'{name}'
                        '</td>' for name in names) + 
            '</tr>')
    html_str += (
        '<tr>' + 
            ''.join('<td style="vertical-align:top"> '
                        f'{to_df(df).to_html(index=index)}'
                    '</td>' 
                for df in dfs) + 
        '</tr>')
    html_str = f'<table>{html_str}</table>'
    html_str = html_str.replace('table',
                                'table style="display:inline"')
    display_html(html_str, raw=True)