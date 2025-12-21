"""
ep_intervals.py — utilities for encoding extra predecessor patterns

NW-flex uses row-wise extra predecessor sets E(i) to extend the usual
Gotoh DP. In Python, these sets are naturally represented as a list of
lists:

    EP[i] = [r0, r1, ...]

where each entry r is a predecessor row index for DP row i.

For efficient use in the Cython/C core, we convert this list-of-lists
representation into a compact interval form:

    ep_counts[i]          = number of intervals on row i
    ep_starts[i, k], ep_ends[i, k] = start/end of k-th interval

so that each E(i) can be iterated as contiguous ranges [start..end] in
tight inner loops.
"""
from typing import Sequence, List, Tuple
import numpy as np

def ep_to_intervals(
    ep: Sequence[Sequence[int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert EP[i] = list of extra predecessor rows into interval form.

    For each DP row i, compute disjoint contiguous intervals [a, b]
    covering the sorted unique elements of EP[i].

    Returns
    -------
    ep_counts : int32[ n_rows ]
    ep_starts : int32[ n_rows x max_intervals ]
    ep_ends   : int32[ n_rows x max_intervals ]
    """
    n_rows = len(ep)
    intervals_per_row: List[List[Tuple[int, int]]] = []
    ep_counts = np.zeros(n_rows, dtype=np.int32)
    max_intervals = 0

    for row_index, rows in enumerate(ep):
        if not rows:
            intervals_per_row.append([])
            continue

        sorted_rows = sorted(set(rows))
        intervals: List[Tuple[int, int]] = []
        start = sorted_rows[0]
        prev = sorted_rows[0]

        for r in sorted_rows[1:]:
            if r == prev + 1:
                prev = r
            else:
                intervals.append((start, prev))
                start = prev = r
        intervals.append((start, prev))

        intervals_per_row.append(intervals)
        ep_counts[row_index] = len(intervals)
        max_intervals = max(max_intervals, len(intervals))

    ep_starts = np.zeros((n_rows, max_intervals), dtype=np.int32)
    ep_ends   = np.zeros((n_rows, max_intervals), dtype=np.int32)

    for i, row_intervals in enumerate(intervals_per_row):
        for k, (a, b) in enumerate(row_intervals):
            ep_starts[i, k] = a
            ep_ends[i, k]   = b

    return ep_counts, ep_starts, ep_ends
