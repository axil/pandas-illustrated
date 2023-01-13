﻿from collections import defaultdict
from itertools import tee
from functools import reduce as _reduce

from orderedset import OrderedSet as oset


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def process(chunks):
    def f(x):
        if len(x)>1:
            for a, b in pairwise(x):
                d[b].add(a)
        else:
            d[x[0]]
    d = defaultdict(oset)
    for chunk in chunks:
        f(chunk)
    return d


class CircularDependencyError(ValueError):
    def __init__(self, data):
        # Sort the data just to make the output consistent, for use in
        #  error messages.  That's convenient for doctests.
        s = "Circular dependencies exist among these items: {{{}}}".format(
            ", ".join(
                "{!r}:{!r}".format(key, value) for key, value in sorted(data.items())
            )
        )
        super(CircularDependencyError, self).__init__(s)
        self.data = data


def toposort(data):
    # Special case empty input.
    if len(data) == 0:
        return

    # Copy the input so as to leave it unmodified.
    # Discard self-dependencies and copy two levels deep.
    data = {item: oset(e for e in dep if e != item) for item, dep in data.items()}

    # Find all items that don't depend on anything.
    extra_items_in_deps = _reduce(oset.union, data.values()) - oset(data.keys())
    # Add empty dependences where needed.
    data.update({item: oset() for item in extra_items_in_deps})
    while True:
        ordered = oset(item for item, dep in data.items() if len(dep) == 0)
        if not ordered:
            break
        yield ordered
        data = {
            item: (dep - ordered) for item, dep in data.items() if item not in ordered
        }
    if len(data) != 0:
        raise CircularDependencyError(data)


def toposort_flatten(data, sort=True):
    result = []
    for d in toposort(data):
        result.extend((sorted if sort else list)(d))
    return result


def stack(df, level=-1, dropna=False):
    n = df.columns.nlevels
    if isinstance(level, int):
        if level >= n:
            raise IndexError(f'Level number is out of bounds: {level} > {n-1}')
        elif level >= 0:
            level_pos = level
        elif level < -n:
            raise IndexError(f'Level number is out of bounds: {level} < -{n}')
        else:
            level_pos = n + level
    else:
        try:
            level_pos = df.columns.names.index(level)
        except ValueError:
            raise ValueError(f'level name {level} not found in the index level names')
    fr = df.columns.to_frame(index=False)
    others = [fr.iloc[:, i] for i in range(fr.shape[1]) if i != level_pos]
    others_chunks = oset()
    chunks = oset()
    for k, v in fr.groupby(others, sort=False):
        others_chunks.add(k)
        chunks.add(tuple(v.iloc[:,level_pos].tolist()))
    d = process(chunks); d
    stacked = toposort_flatten(d, sort=False)
    df1 = df.stack(level, dropna=dropna)
    df1 = df1.reindex(others_chunks, axis=1)
    df1 = df1.reindex(stacked, axis=0, level=-1)
    return df1


def unstack(df, level=-1, dropna=False):
    n = df.index.nlevels
    if n == 1:      # nothing to be done
        return df.unstack(level, dropna=dropna)
    if isinstance(level, int):
        if level >= n:
            raise IndexError(f'Level number is out of bounds: {level} > {n-1}')
        elif level >= 0:
            level_pos = level
        elif level < -n:
            raise IndexError(f'Level number is out of bounds: {level} < -{n}')
        else:
            level_pos = n + level
    else:
        try:
            level_pos = df.index.names.index(level)
        except ValueError:
            raise ValueError(f'level name {level} not found in the index level names')
    fr = df.index.to_frame(index=False)
    others = [fr.iloc[:, i] for i in range(fr.shape[1]) if i != level_pos]
    others_chunks = oset()
    chunks = oset()
    for k, v in fr.groupby(others, sort=False):
        others_chunks.add(k)
        chunks.add(tuple(v.iloc[:,level_pos].tolist()))
    d = process(chunks); d
    stacked = toposort_flatten(d, sort=False)
    df1 = df.unstack(level, dropna=dropna)
    df1 = df1.reindex(others_chunks)
    df1 = df1.reindex(stacked, axis=1, level=-1)
    return df1