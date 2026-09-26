"""快速排序的测试：正确性、原地语义、边界与退化输入。"""

from __future__ import annotations

import random

from examples.quicksort import quicksort, quicksort_inplace


def test_sorts_int_and_str():
    assert quicksort([3, 1, 2]) == [1, 2, 3]
    assert quicksort(["b", "a", "c"]) == ["a", "b", "c"]
    assert quicksort([]) == []
    assert quicksort([1]) == [1]


def test_matches_builtin_sorted_on_random_input():
    rng = random.Random(0)
    for _ in range(200):
        data = [rng.randrange(50) for _ in range(rng.randrange(40))]
        assert quicksort(data) == sorted(data)


def test_inplace_returns_same_object_and_mutates():
    data = [5, 3, 8, 1]
    out = quicksort_inplace(data)
    assert out is data
    assert data == [1, 3, 5, 8]


def test_quicksort_does_not_mutate_input():
    data = [3, 1, 2]
    quicksort(data)
    assert data == [3, 1, 2]


def test_already_sorted_and_reversed_do_not_blow_stack():
    n = 5000
    assert quicksort_inplace(list(range(n))) == list(range(n))
    assert quicksort_inplace(list(range(n, 0, -1))) == list(range(1, n + 1))


def test_duplicates():
    assert quicksort([2, 1, 2, 1, 2]) == [1, 1, 2, 2, 2]
