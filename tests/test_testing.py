import pytest

from pdi.testing import range2d, gen_df, gen_df1, vicn


def test_range2d():
    assert range2d(2,3).tolist() == [[1, 2, 3], [4, 5, 6]]
    assert range2d(3,4).tolist() == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

def test_gen_df_12():
    df = gen_df(1, 2)
    assert vicn(df) == \
([[1, 2, 3, 4], [5, 6, 7, 8]],
 ['a', 'b'],
 [('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D')],
 [['k'], ['K', 'L']])


def test_gen_df_23():
    df = gen_df(2, 3)
    assert vicn(df) == \
([[1, 2, 3, 4, 5, 6, 7, 8],
  [9, 10, 11, 12, 13, 14, 15, 16],
  [17, 18, 19, 20, 21, 22, 23, 24],
  [25, 26, 27, 28, 29, 30, 31, 32]],
 [('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd')],
 [('A', 'C', 'E'),
  ('A', 'C', 'F'),
  ('A', 'D', 'E'),
  ('A', 'D', 'F'),
  ('B', 'C', 'E'),
  ('B', 'C', 'F'),
  ('B', 'D', 'E'),
  ('B', 'D', 'F')],
 [['k', 'l'], ['K', 'L', 'M']])

def test_gen_df1():
    assert vicn(gen_df1(2,3)) == \
        ([[1, 2, 3], [4, 5, 6]], ['a', 'b'], ['A', 'B', 'C'], [[None], [None]])
    assert vicn(gen_df1(2,3, False)) == \
        ([[1, 2, 3], [4, 5, 6]], ['b', 'a'], ['C', 'B', 'A'], [[None], [None]])

if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
