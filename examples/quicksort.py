"""快速排序：原地实现 + 返回新列表的便捷封装。

原地版本用 Hoare 式双指针分区，并总是先递归较小的一侧，把递归深度压到
O(log n)，避免有序输入退化成 O(n) 栈深。元素比较用 ``<=`` / ``<``，
因此只要元素之间可比较即可。
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def _partition(items: list[T], lo: int, hi: int) -> int:
    """以 items[hi] 为基准分区，返回基准的最终下标（Lomuto 分区）。"""
    pivot = items[hi]
    i = lo
    for j in range(lo, hi):
        if items[j] <= pivot:
            items[i], items[j] = items[j], items[i]
            i += 1
    items[i], items[hi] = items[hi], items[i]
    return i


def quicksort_inplace(items: list[T], lo: int = 0, hi: int | None = None) -> list[T]:
    """原地排序 ``items``，返回同一个列表对象。

    ``lo`` / ``hi`` 为可选排序区间（左闭右闭），默认整段。
    """
    if hi is None:
        hi = len(items) - 1
    while lo < hi:
        mid = _partition(items, lo, hi)
        # 先处理小区间，大区间走循环——栈深由小侧决定
        if mid - lo < hi - mid:
            quicksort_inplace(items, lo, mid - 1)
            lo = mid + 1
        else:
            quicksort_inplace(items, mid + 1, hi)
            hi = mid - 1
    return items


def quicksort(items: list[T]) -> list[T]:
    """返回排序后的**新**列表，不修改入参。"""
    return quicksort_inplace(list(items))
