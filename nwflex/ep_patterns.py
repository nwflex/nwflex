"""
ep_patterns.py — Extra-predecessor (EP) configurations for NW-flex

This module defines helpers for constructing the row-wise extra
predecessor sets E(i) used by the NW-flex DP core.

All functions assume:
  - X has length n, with DP rows indexed 0..n,
  - row 0 is the empty prefix,
  - for each row i in 1..n, E(i) ⊆ {0, ..., i-2} (baseline i-1 is implicit
    and must NOT be included).
  - EP[n+1] is the *terminal predecessor set*: it specifies which rows
    are valid starting points for traceback (i.e., the set of rows r such
    that (r, m) can be the final cell).  For global alignment, EP[n+1] is
    empty (implying {n}); for semi-global (local in X), EP[n+1] = {0..n-1}
    (baseline n is implicit).

The patterns here correspond to the configurations described in the
manuscript:

  * Standard NW/Gotoh (no extra predecessors)
  * Single flexible block X = A·Z·B
  * STR specialization Z = R^N, |R| = k
  * Multiple STR blocks in the same reference

Note: in the text, we sometimes let the predecessor set E(i) include i-1 to avoid clutter; 
      here we always exclude i-1 since it is implicit in the baseline.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


# ---------------------------------------------------------------------------
# 0. Standard NW/Gotoh: no extra predecessors
# ---------------------------------------------------------------------------

def build_EP_standard(n: int) -> List[List[int]]:
    """
    Standard Needleman-Wunsch/Gotoh: no extra predecessors.

    Parameters
    ----------
    n : int
        Length of X (number of sequence characters).  DP rows are 0..n.

    Returns
    -------
    EP : list of list of int
        EP[i] is empty for all i in 0..n+1, so each row uses only the
        baseline predecessor i-1 as in standard NW/Gotoh.
        EP[n+1] is empty, meaning traceback starts at row n (global).
    """
    # n+2 entries: indices 0..n for DP rows, index n+1 for terminal set
    return [[] for _ in range(n + 2)]


# ---------------------------------------------------------------------------
# 1. Single flexible block X = A·Z·B
# ---------------------------------------------------------------------------

def build_EP_single_block(n: int, s: int, e: int) -> List[List[int]]:
    """
    Single flexible block X = A·Z·B with rows 0..n.

    Here
        s = |A|
        e = s + |Z|

    and rows are interpreted as:
        0          : empty prefix
        1..s       : block A
        s+1..e     : block Z
        e+1..n     : block B

    We use the EP pattern from the paper:

        E(i) =
          {s}               if s < i ≤ e
          {s, s+1, …, e-1}  if i = e+1  [note: excluding baseline e]
          ∅                 otherwise

    i.e. every row in Z sees the leader row s as an extra predecessor,
    and the first row of B (the "closer" row) sees all rows in Z (and the
    leader) as extra predecessors.

    Notes
    -----
    * We require 0 ≤ s < e ≤ n.
    * If e == n (no B), there is no closer row e+1 in the DP table, so we
      apply the closer pattern to the terminal predecessor set EP[n+1].
      This allows traceback to start from any row in the flex block,
      effectively treating the terminal as the closer.
    * EP[n+1] is empty for global alignment (traceback starts at row n),
      unless e == n, in which case EP[n+1] = {s, ..., e-1}.
    """
    if not (0 <= s < e <= n):
        raise ValueError(f"Invalid block indices: need 0 ≤ s < e ≤ n, got s={s}, e={e}, n={n}")

    # n+2 entries: indices 0..n for DP rows, index n+1 for terminal set
    EP: List[List[int]] = [[] for _ in range(n + 2)]

    for i in range(1, n + 1):
        if s < i <= e:
            # Inside Z: EP(i) = {s}
            EP[i].append(s)
        elif i == e + 1 and e < n:
            # Closer row: EP(e+1) = {s, ..., e-1} (since e is in the baseline)
            EP[i].extend(range(s, e))

    # If e == n (B is empty), apply closer pattern to terminal set
    if e == n:
        # Terminal predecessor set: {s, ..., e-1} (baseline e=n is implicit)
        EP[n + 1].extend(range(s, e))

    return EP


# ---------------------------------------------------------------------------
# 2. STR specialization: single STR block Z = R^N
# ---------------------------------------------------------------------------

def build_EP_STR_phase(n: int, s: int, e: int, k: int) -> List[List[int]]:
    """
    STR specialization: flexible block Z = R^N with motif length k.

    Here
        s = |A|
        e = s + |Z| (end of the repeat block in DP-row indexing)

    We use the phase-preserving pattern:

        E(i)   = {s}                    if s < i ≤ e
        E(e+1) = {s, e-k+1, …, e-1}     if e < n [note: excluding baseline e]
        E(i)   = ∅                      otherwise

    This preserves motif phase at the exit by providing one exit row
    for each phase class modulo k among the last k rows of Z, plus the
    leader row s.

    Notes
    -----
    * As above, 0 ≤ s < e ≤ n.
    * If e == n (no B), there is no closer row e+1 in the DP table, so we
      apply the closer pattern to the terminal predecessor set EP[n+1].
      This allows traceback to start from the correct phase row within the
      flex block, matching the behaviour of build_EP_single_block.
    """
    if not (0 <= s < e <= n):
        raise ValueError(f"Invalid block indices: need 0 ≤ s < e ≤ n, got s={s}, e={e}, n={n}")
    if k <= 0:
        raise ValueError(f"Motif length k must be positive, got k={k}")

    # n+2 entries: indices 0..n for DP rows, index n+1 for terminal set
    EP: List[List[int]] = [[] for _ in range(n + 2)]
    
    for i in range(1, n + 1):
        if s < i <= e:
            # Inside Z: EP(i) = {s}
            EP[i].append(s)
        elif i == e + 1 and e + 1 <= n:
            # Closer row: EP(e+1) = {s, e-k+1, ..., e-1}
            EP[i].append(s)
            start = max(s + 1, e - k + 1)
            EP[i].extend(range(start, e))  # note: excluding baseline e

    # If e == n (B is empty), apply closer pattern to terminal set
    if e == n:
        EP[n + 1].append(s)
        start = max(s + 1, e - k + 1)
        EP[n + 1].extend(range(start, e))

    return EP

# ---------------------------------------------------------------------------
# 3. Union of two EP configurations
# ---------------------------------------------------------------------------

def union_EP_pair(ep1: Sequence[Sequence[int]], ep2: Sequence[Sequence[int]]) -> List[List[int]]:
    """
    Take the row-wise union of two EP configurations.

    Parameters
    ----------
    ep1, ep2 : sequences of sequences of int
        Two extra-predecessor patterns for the *same* reference length.
        Each ep[k] is interpreted as the list of extra predecessors E(k).

    Returns
    -------
    EP : list of list of int
        Row-wise union:
            E_union(i) = E1(i) union E2(i)
        Duplicates are removed, and the result is sorted for reproducibility.


    Raises
    ------
    ValueError
        If ep1 and ep2 have different lengths.
    """
    if len(ep1) != len(ep2):
        raise ValueError(
            f"Cannot union EPs of different lengths: len(ep1)={len(ep1)}, len(ep2)={len(ep2)}"
        )

    EP = []
    for e1, e2 in zip(ep1, ep2):
        # Use a set to remove duplicates; convert back to sorted list for stability
        merged = sorted(set(e1) | set(e2))
        EP.append(merged)
    return EP

# ---------------------------------------------------------------------------
# 4: Union of multiple EP configurations
# ---------------------------------------------------------------------------
def union_EP_sequence(eps: Sequence[Sequence[Sequence[int]]]) -> List[List[int]]:
    """
    Row-wise union of multiple EP configurations.

    Parameters
    ----------
    eps : sequence of EP configurations
        Each element is a list of lists of integers: eps[j][i] = E_j(i),
        the extra-predecessor set for DP row i in configuration j.
        All EP configurations must have the same number of rows.

    Returns
    -------
    EP : list of list of int
        Combined extra-predecessor pattern, where each row i contains
        the sorted union of all E_j(i), with duplicates removed.

    Raises
    ------
    ValueError
        If eps is empty or the EPs have different lengths.
    """
    if not eps:
        raise ValueError("union_EP_sequence requires at least one EP configuration.")

    n = len(eps[0])  # number of DP rows
    if any(len(ep) != n for ep in eps):
        lengths = {len(ep) for ep in eps}
        raise ValueError(f"All EPs must have the same length, got lengths={lengths}")

    EP: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        row_union = set()
        for ep in eps:
            row_union.update(ep[i])
        EP[i] = sorted(row_union)

    return EP

# ---------------------------------------------------------------------------
# 5. Union of multiple STR blocks
# ---------------------------------------------------------------------------
def build_EP_multi_STR_phase(
    n: int,
    blocks: Sequence[Tuple[int, int, int]],
) -> List[List[int]]:
    """
    Extra-predecessor (EP) pattern for multiple STR blocks in the same reference.

    Each STR block is defined by a triple (s, e, k), where:
        s = leader row index (length of X before the repeat block)
        e = s + |Z| (inclusive end row of the repeat block)
        k = motif length |R|

    For each block, `build_EP_STR_phase(n, s, e, k)` returns the appropriate
    phase-preserving EP pattern for that block alone. When multiple STR blocks
    are present in the same reference X, we simply take the row-wise union:

        E(i) = Union_{block ∈ blocks} E_block(i)

    where each E_block(i) is contributed by one block's EP pattern.

    Parameters
    ----------
    n : int
        Length of the reference X (DP rows 0..n).

    blocks : sequence of (s, e, k)
        One triple per STR block.  We require 0 ≤ s < e ≤ n and k > 0 for each
        block. Each block is treated independently; their EP sets are merged.

    Returns
    -------
    EP : list of list of int
        Combined extra-predecessor sets E(i), one list per DP row.

    Notes
    -----
    * Blocks may be disjoint or adjacent. Overlaps are permitted; in overlapping
      regions the E(i) sets accumulate extra predecessors from all contributing
      blocks.
    * Baseline predecessor i-1 is implicit and is never included.
    """
    # Build EP pattern for each block independently
    ep_list = [build_EP_STR_phase(n, s, e, k) for (s, e, k) in blocks]

    # Merge them using the general multi-union helper
    return union_EP_sequence(ep_list)


# ---------------------------------------------------------------------------
# 6. Semi-global configuration: global in Y, local in X
# ---------------------------------------------------------------------------
def build_EP_semiglobal(n: int) -> List[List[int]]:
    """
    Semi-global alignment configuration (global in Y, local in X).

    This uses a single flexible block that spans the entire reference X,
    with leader row s = 0 and end row e = n:

        X = A·Z·B with A = ε, Z = X[1..n], B = ε.

    The corresponding EP pattern for DP rows 1..n is the single-block pattern

        build_EP_single_block(n, s=0, e=n),

    i.e. every row in Z (rows 1..n) sees row 0 as an extra predecessor.
    Since e = n, there is no closer row within the DP range.

    Additionally, EP[n+1] = {0, 1, ..., n-1} specifies that traceback can
    start at any row (local in X). The baseline row n is implicit.

    Parameters
    ----------
    n : int
        Length of the reference X (number of sequence characters).
        If n ≤ 1, there is nothing to flex.

    Returns
    -------
    EP : list of list of int
        Extra-predecessor sets E(i) implementing this semi-global mode.
        EP[n+1] = {0..n-1} enables local-in-X traceback.
    """
    if n <= 1:
        # nothing meaningful to flex; fall back to standard NW
        return build_EP_standard(n)
    
    # Start with the single-block pattern for DP rows
    EP = build_EP_single_block(n, s=0, e=n)
    
    # Set terminal predecessor set for semi-global: traceback can start at any row
    # EP[n+1] = {0, 1, ..., n-1}, excluding baseline n
    EP[n + 1] = list(range(n))
    
    return EP
