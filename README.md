# pandas-illustrated

[![pypi](https://img.shields.io/pypi/v/pandas-illustrated.svg)](https://pypi.python.org/pypi/pandas-illustrated)
[![python](https://img.shields.io/pypi/pyversions/pandas-illustrated.svg)](https://pypi.org/project/pandas-illustrated/)
![pytest](https://github.com/axil/pandas-illustrated/actions/workflows/python-package.yml/badge.svg)
![Coverage Badge](img/coverage.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/pypi/l/pandas-illustrated)](https://pypi.org/project/pandas-illustrated/)

This repo contains code for a number of helper functions mentioned in the [Pandas Illustrated](https://betterprogramming.pub/pandas-illustrated-the-definitive-visual-guide-to-pandas-c31fa921a43?sk=50184a8a8b46ffca16664f6529741abc) guide.

## Installation: 

    pip install pandas-illustrated

## Contents

Basic operations:
- `find(s, x, pos=False)`
- `findall(s, x, pos=False)`
- `insert(dst, pos, value, label, axis=0, ignore_index = False, 
    order=None, allow_duplicates=False, inplace=False)`
- `append(dst, value, label = lib.no_default, axis=0, ignore_index = False,
    order=None, allow_duplicates: bool = False, inplace=False)`
- `drop(obj, items=None, like=None, regex=None, axis=None)`
- `move(obj, pos, label=None, column=None, index=None, axis=None, reset_index=False)`
- `join(dfs, on=None, how="left", suffixes=None)`

Visualization improvements:
- `patch_series_repr(footer=True)`
- `unpatch_series_repr()`
- `sidebyside(*dfs, names=[], index=True, valign="top")`
- `sbs = sidebyside`

MultiIndex helpers:
- `patch_mi_co()`
- `from_dict(d)`
- `from_kw(**kwargs)`

Locking columns order:
- `locked(obj, level=None, axis=None, categories=None, inplace=False)`
- `lock = locked with inplace=True`
- `vis_lock(obj, checkmark="✓")`
- `vis_patch()`
- `vis_unpatch()`
- `from_product(iterables, sortorder=None, names=lib.no_default, lock=True)`

MultiIndex manipulations:
- `get_level(obj, level_id, axis=None)`
- `set_level(obj, level_id, labels, name=lib.no_default, axis=None, inplace=False)`
- `move_level(obj, src, dst, axis=None, inplace=False, sort=False)`
- `insert_level(obj, pos, labels, name=lib.no_default, axis=None, inplace=False, sort=False)`
- `drop_level(obj, level_id, axis=None, inplace=False)`
- `swap_levels(obj, i: Axis = -2, j: Axis = -1, axis: Axis = None, inplace=False, sort=False)`
- `join_levels(obj, name=None, sep="_", axis=None, inplace=False)`
- `split_level(obj, names=None, sep="_", axis=None, inplace=False)`
- `rename_level(obj, mapping, level_id=None, axis=None, inplace=False)`


## Usage

### find and findall

By default `find(series, value)` looks for the first occurrence of the given *value* in a *series* and returns the corresponsing index label.

```python
>>> import pandas as pd
>>> import pdi

>>> s = pd.Series([4, 2, 4, 6], index=['cat', 'penguin', 'dog', 'butterfly'])

>>> pdi.find(s, 2)
'penguin' 

>>> pdi.find(s, 4)
'cat' 
```

When the value is not found raises a `ValueError`.

`findall(series, value)` returns a (possibly empty) index of all matching occurrences:

```python
>>> pdi.findall(s, 4)
Index(['cat', 'dog'], dtype='object')
```

With `pos=True` keyword argument `find()` and `findall()` return the positional index instead:

```python
>>> pdi.find(s, 2, pos=True)
1 

>>> pdi.find(s, 4, pos=True)
0
```
There is a number of ways to find index label for a given value. The most efficient of them are:

```python
— s.index[s.tolist().index(x)]       # faster for Series with less than 1000 elements
— s.index[np.where(s == x)[0][0]]    # faster for Series with over 1000 elements  
```

<img src="https://user-images.githubusercontent.com/170910/209191163-52b8cc6a-425d-41e0-a7f9-c2efb4a31bbb.png" width="600">

`find()` chooses optimal implementation depending on the series size; `findall()` always uses the `where` implementation.

### Improving Series Representation

Run `pdi.patch_series_repr()` to make Series look better:

<img src="https://user-images.githubusercontent.com/170910/211085821-544b42b0-561a-47e7-8f32-6f31a05ed978.png" width="600">

If you want to display several Series from one cell, call `display(s)` for each.

### Displaying several Pandas objects side vy side

To display several dataframes, series or indices side by side run `pdi.sidebyside(s1, s2, ...)`

<img src="img/sbs.png" width="450"/>

## Testing

Run `pytest` in the project root.
