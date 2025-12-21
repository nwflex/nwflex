
"""
nw_basics.py — simple Needleman-Wunsch dynamic programming (no affine gaps)

This module provides a minimal Needleman-Wunsch implementation for instruction
and visualization:

  - NWStep                : records a single cell update (candidates, best move, score).
  - needleman_wunsch_steps: fills a score matrix and traceback, returns
                                a list of NWStep objects and the traceback path.
  - traceback_simple      : reconstructs one global alignment from a traceback matrix.
  - GotohResult           : result container for affine-gap alignment.
  - nw_gotoh_matrix       : Gotoh affine-gap alignment returning full DP matrices.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, NamedTuple

import numpy as np
from numpy.typing import NDArray


# ============================================================================
# RESULT CONTAINERS
# ============================================================================

class GotohResult(NamedTuple):
    """
    Result from nw_gotoh_matrix affine-gap alignment.
    
    Attributes
    ----------
    total_score : float
        Optimal alignment score.
    seq_align1 : str
        Aligned sequence 1 with gaps.
    seq_align2 : str
        Aligned sequence 2 with gaps.
    aligned_base_1 : ndarray or None
        Original indices of aligned bases in seq1 (-1 for gaps).
    aligned_base_2 : ndarray or None
        Original indices of aligned bases in seq2 (-1 for gaps).
    M : ndarray or None
        Match state DP matrix.
    X : ndarray or None
        Gap-in-seq2 (vertical) state DP matrix.
    Y : ndarray or None
        Gap-in-seq1 (horizontal) state DP matrix.
    best_path : list or None
        Traceback path as [(i, j, state), ...], state in {0:M, 1:X, 2:Y}.
    """
    total_score: float
    seq_align1: str
    seq_align2: str
    aligned_base_1: Optional[np.ndarray] = None
    aligned_base_2: Optional[np.ndarray] = None
    M: Optional[np.ndarray] = None
    X: Optional[np.ndarray] = None
    Y: Optional[np.ndarray] = None
    best_path: Optional[List[Tuple[int, int, int]]] = None


# ============================================================================
# SIMPLE NW (LINEAR GAPS)
# ============================================================================

@dataclass
class NWStep:
    """
    One update step in the Needleman-Wunsch DP fill.

    Attributes
    ----------
    i, j : int
        DP coordinates of the updated cell (row i, column j).

    candidates : dict
        Mapping {"diag","up","left"} -> candidate scores for this cell.

    best_move : str
        One of {"diag","up","left"} indicating which move was chosen.

    score : int
        Final score stored in F[i,j] for this cell.
    """
    i: int
    j: int
    candidates: Dict[str, int]
    best_move: str
    score: int


def needleman_wunsch_steps(
    seq1: str,
    seq2: str,
    match: int = 1,
    mismatch: int = -1,
    gap: int = -1,
) -> Tuple[NDArray[np.int_], np.ndarray, List[NWStep], List[Tuple[int, int]]]:
    """
    Run a simple Needleman-Wunsch global alignment (no affine gaps),
    recording each DP update as an NWStep for later visualization.

    Parameters
    ----------
    seq1, seq2 : str
        Sequences to align.

    match, mismatch : int
        Scores for match and mismatch, used in diagonal transitions.

    gap : int
        Gap penalty (applied for both insertions and deletions).

    Returns
    -------
    F : (m+1, n+1) array of int
        DP score matrix, where F[i,j] is the best score for aligning
        seq1[0:i] with seq2[0:j].

    trace : (m+1, n+1) ndarray
        Traceback matrix with entries in {"diag","up","left"} (or "" at (0,0)).

    steps : list of NWStep
        One NWStep per inner cell (i >= 1, j >= 1).

    path : list of (i, j)
        Traceback path from (0,0) to (m,n) for one best alignment.
    """
    m, n = len(seq1), len(seq2)
    F: NDArray[np.int_] = np.zeros((m + 1, n + 1), dtype=int)
    trace: np.ndarray = np.full((m + 1, n + 1), "", dtype=object)

    # initialization: global alignment
    for i in range(1, m + 1):
        F[i, 0] = i * gap
        trace[i, 0] = "up"
    for j in range(1, n + 1):
        F[0, j] = j * gap
        trace[0, j] = "left"

    steps: List[NWStep] = []

    # fill DP matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = F[i - 1, j - 1] + (match if seq1[i - 1] == seq2[j - 1] else mismatch)
            up   = F[i - 1, j] + gap
            left = F[i, j - 1] + gap

            candidates = {"diag": diag, "up": up, "left": left}
            best_move = max(candidates, key=candidates.get)
            score = candidates[best_move]

            F[i, j] = score
            trace[i, j] = best_move
            steps.append(NWStep(i=i, j=j, candidates=candidates, best_move=best_move, score=score))

    # traceback to recover one optimal path in the DP grid
    path: List[Tuple[int, int]] = []
    i, j = m, n
    while i > 0 or j > 0:
        path.append((i, j))
        move = trace[i, j]
        if move == "diag":
            i -= 1
            j -= 1
        elif move == "up":
            i -= 1
        elif move == "left":
            j -= 1
        else:
            break
    path.append((0, 0))
    path.reverse()

    return F, trace, steps, path


def traceback_simple(
    seq1: str,
    seq2: str,
    trace: np.ndarray,
) -> Tuple[str, str, List[Tuple[int, int]]]:
    """
    Reconstruct one global alignment from a traceback matrix produced by
    needleman_wunsch_steps.

    Parameters
    ----------
    seq1, seq2 : str
        Sequences that were aligned.

    trace : (m+1, n+1) ndarray of str
        Traceback matrix with entries in {"diag","up","left"}.

    Returns
    -------
    aln1, aln2 : str
        Aligned versions of seq1 and seq2 (including '-' gaps).

    path : list of (i, j)
        Traceback path from (0,0) to (m,n).
    """
    m, n = len(seq1), len(seq2)
    i, j = m, n
    aln1: List[str] = []
    aln2: List[str] = []
    path: List[Tuple[int, int]] = []

    while i > 0 or j > 0:
        path.append((i, j))
        move = trace[i, j]
        if move == "diag":
            aln1.append(seq1[i - 1])
            aln2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif move == "up":
            aln1.append(seq1[i - 1])
            aln2.append("-")
            i -= 1
        elif move == "left":
            aln1.append("-")
            aln2.append(seq2[j - 1])
            j -= 1
        else:
            break
    path.append((0, 0))
    path.reverse()

    aln1_str = "".join(reversed(aln1))
    aln2_str = "".join(reversed(aln2))

    return aln1_str, aln2_str, path


# ============================================================================
# GOTOH AFFINE-GAP ALIGNMENT
# ============================================================================

def nw_gotoh_matrix(
    seq1: str,
    seq2: str,
    alphabet_to_index: Dict[str, int],
    score_matrix: np.ndarray,
    gap_open: float,
    gap_extend: float,
    return_matrices: bool = True,
) -> GotohResult:
    """
    Global alignment (Needleman–Wunsch) with Gotoh affine gaps.
    
    This function returns full DP matrices for visualization. For production
    alignment, use aligners.align_standard() instead.

    Parameters
    ----------
    seq1, seq2 : str
        Sequences to align.
    alphabet_to_index : dict
        Mapping from nucleotide characters to matrix indices (e.g., {'A': 0, 'C': 1, ...}).
    score_matrix : ndarray
        Substitution matrix where score_matrix[i, j] is the score for aligning
        alphabet character with index i to alphabet character with index j.
    gap_open : float
        Gap opening penalty (should be negative).
    gap_extend : float
        Gap extension penalty (should be negative).
    return_matrices : bool
        If True, include full DP matrices in result; otherwise only score and alignment.

    Returns
    -------
    GotohResult
        Named tuple with alignment score, aligned sequences, and optionally DP matrices.
    """
    m, n = len(seq1), len(seq2)
    M = np.full((m + 1, n + 1), -np.inf)
    X = np.full((m + 1, n + 1), -np.inf)  # gap in seq2 (vertical move)
    Y = np.full((m + 1, n + 1), -np.inf)  # gap in seq1 (horizontal move)

    # Initialization
    M[0, 0] = 0.0
    # First column: a gap in seq2 of length i
    for i in range(1, m + 1):
        X[i, 0] = gap_open + (i - 1) * gap_extend
    # First row: a gap in seq1 of length j
    for j in range(1, n + 1):
        Y[0, j] = gap_open + (j - 1) * gap_extend

    # DP fill
    for i in range(1, m + 1):
        ai = seq1[i - 1]
        ai_idx = alphabet_to_index[ai]
        for j in range(1, n + 1):
            bj = seq2[j - 1]
            bj_idx = alphabet_to_index[bj]
            s = score_matrix[ai_idx, bj_idx]

            # Match/mismatch
            M[i, j] = s + max(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1])

            # X: gap in seq2 (vertical)
            X[i, j] = max(
                gap_open   + M[i - 1, j],   # open from M
                gap_extend + X[i - 1, j],   # extend from X
                gap_open   + Y[i - 1, j],   # open from Y
            )

            # Y: gap in seq1 (horizontal)
            Y[i, j] = max(
                gap_open   + M[i, j - 1],   # open from M
                gap_open   + X[i, j - 1],   # open from X
                gap_extend + Y[i, j - 1],   # extend from Y
            )

    finals = (M[m, n], X[m, n], Y[m, n])
    total = float(max(finals))
    state = int(np.argmax(finals))

    # Traceback
    i, j = m, n
    aln1, aln2 = [], []
    p1, p2 = [], []
    path: List[Tuple[int, int, int]] = []

    while i > 0 or j > 0:
        path.append((i, j, state))
        if state == 0:
            # Came from max of previous diagonal states
            aln1.append(seq1[i - 1]); p1.append(i - 1)
            aln2.append(seq2[j - 1]); p2.append(j - 1)
            prev = np.array([M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1]])
            state = int(np.argmax(prev))
            i -= 1; j -= 1
        elif state == 1:
            # Vertical move (gap in seq2): up
            aln1.append(seq1[i - 1]); p1.append(i - 1)
            aln2.append('-');          p2.append(-1)
            # Decide predecessor for X
            if np.isclose(X[i, j], gap_extend + X[i - 1, j]):
                state = 1
            elif np.isclose(X[i, j], gap_open + M[i - 1, j]):
                state = 0
            else:
                state = 2
            i -= 1
        else:
            # Horizontal move (gap in seq1): left
            aln1.append('-');          p1.append(-1)
            aln2.append(seq2[j - 1]); p2.append(j - 1)
            # Decide predecessor for Y
            if np.isclose(Y[i, j], gap_extend + Y[i, j - 1]):
                state = 2
            elif np.isclose(Y[i, j], gap_open + M[i, j - 1]):
                state = 0
            else:
                state = 1
            j -= 1

    seq_align1 = ''.join(reversed(aln1))
    seq_align2 = ''.join(reversed(aln2))
    aligned_base_1 = np.array(list(reversed(p1)))
    aligned_base_2 = np.array(list(reversed(p2)))
    path = list(reversed(path))

    return GotohResult(
        total_score=total,
        seq_align1=seq_align1,
        seq_align2=seq_align2,
        aligned_base_1=aligned_base_1 if return_matrices else None,
        aligned_base_2=aligned_base_2 if return_matrices else None,
        M=M if return_matrices else None,
        X=X if return_matrices else None,
        Y=Y if return_matrices else None,
        best_path=path if return_matrices else None,
    )
