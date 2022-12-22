# pdi

pdi contains code for find and findall functions from the 'Pandas Illustrated' article.
Chooses optimal implementation depending on the size of the Series.

## Installation: 

    pip install pdi

## Usage
   
By default `find()` looks for the first occurrence of the given value in a `Series` and returns the corresponsing index label.

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

`findall()` returns a (possibly empty) index of all matching occurrences:

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
