# pandas_illustrated

This repo contains code for `find()` and `findall()` functions that help in searching Series by value.
Chooses optimal implementation depending on the size of the Series.
See Pandas Illustrated article for details.

## Installation: 

    pip install pandas-illustrated

## Usage
   
By default `find(series, value)` looks for the first occurrence of the given *value* in a *series* and returns the corresponsing index label.

```python
>>> import pandas as pd
>>> import pandas_illustrated as pdi

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

## Performance

There's a number of ways to find index label for a given value. The most efficient of them are:

```python
— s.index[s.tolist().index(x)]       # faster for Series with less than 1000 elements
— s.index[np.where(s == x)[0][0]]    # faster for Series with over 1000 elements  
```

<img src="https://user-images.githubusercontent.com/170910/209191163-52b8cc6a-425d-41e0-a7f9-c2efb4a31bbb.png" width="600">

`find()` chooses optimal implementation depending on the series size; `findall()` always uses the `where` implementation.

## Testing

Run `pytest` in project root.
