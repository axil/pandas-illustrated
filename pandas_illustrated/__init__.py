import numpy as np

def find(s, x, pos=False):
    if len(s) < 1000:
        idx = s.tolist().index(x)
    else:
        idx = np.where(s == x)[0][0]
    if pos:
        return idx
    else:
        return s.index[idx]

def findall(s, x):
    return s.index[np.where(s == x)[0]]