"""
aligners.py — User-facing alignment helpers for NW-flex

This module wraps the NW-flex DP core and the EP pattern builders.  
Each function constructs the
appropriate extra-predecessor configuration E(i), builds a FlexInput,
runs the DP, and returns an AlignmentResult.

The functions here do not change scoring: they 
use the same substitution matrix and affine gap parameters as the
underlying Needleman-Wunsch/Gotoh recurrence.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple, Optional

import numpy as np
from numpy.typing import NDArray

from nwflex.dp_core import AlignmentResult

from .dp_core import FlexInput, run_flex_dp
from .ep_patterns import (
    build_EP_standard,
    build_EP_single_block,
    build_EP_STR_phase,
    build_EP_multi_STR_phase,
    build_EP_semiglobal,
)


# ---------------------------------------------------------------------------
# Alignment expansion: insert gaps for jumped positions
# ---------------------------------------------------------------------------
def expand_alignment_with_jumps(X, Y, path, jumps):
    jump_dict = {jump.from_row: jump for jump in jumps}
    X_expand = []
    Y_expand = []
    for i, j, s in path:
        jump = jump_dict.get(i, None)
        if s == 0:
            X_expand.append("-")
            Y_expand.append(Y[j-1])
        elif s == 1:
            X_expand.append(X[i-1])
            Y_expand.append(Y[j-1])
        else:
            X_expand.append(X[i-1])
            Y_expand.append("-")
        if jump is not None:        
            for i in range(jump.from_row + 1, jump.to_row):
                X_expand.append(X[i-1])
                Y_expand.append("-")
            s = jump.state
    return "".join(X_expand), "".join(Y_expand)


def get_aligned_bases(X: str, Y: str) -> tuple[str, str]:
    """
    Extracts bases of Y aligned to X, and bases of X aligned to Y.

    This function projects the alignment onto the coordinates of the unaligned sequences.
    
    Args:
        X: Aligned reference string (may contain gaps '-').
        Y: Aligned query string (may contain gaps '-').

    Returns:
        A tuple (y_aligned_to_x, x_aligned_to_y) where:
        - y_aligned_to_x: The characters in Y corresponding to non-gap positions in X.
                          (Has the same length as the unaligned X).
        - x_aligned_to_y: The characters in X corresponding to non-gap positions in Y.
                          (Has the same length as the unaligned Y).
    """
    y_aligned_to_x = []
    x_aligned_to_y = []

    for x, y in zip(X, Y):    
        if x != "-":
            y_aligned_to_x.append(y)        
        if y != "-":
            x_aligned_to_y.append(x)
    return "".join(y_aligned_to_x), "".join(x_aligned_to_y)


# ---------------------------------------------------------------------------
# Core helper: run NW-flex with a supplied EP configuration
# ---------------------------------------------------------------------------

def align_with_EP(
    X: str,
    Y: str,
    score_matrix: NDArray[np.floating],
    gap_open: float,
    gap_extend: float,
    alphabet_to_index: Mapping[str, int],
    extra_predecessors: Sequence[Sequence[int]],
    return_data: bool = False,
) -> AlignmentResult:
    """
    Align (X, Y) with a user-supplied EP configuration.

    This is the most general function for accessing the NW-flex core: 
    callers construct extra_predecessors[i] = E(i) and pass it directly.

    Parameters
    ----------
    X, Y : str
        Reference (rows) and read (columns).

    score_matrix : (K, K) array
        Substitution matrix σ, indexed by alphabet_to_index[base] for X and Y.

    gap_open, gap_extend : float
        Affine gap penalties g_s, g_e.

    alphabet_to_index : mapping str -> int
        Maps symbols to indices in score_matrix.

    extra_predecessors : sequence of sequence of int
        Row-wise extra predecessor sets E(i).  Must have length len(X)+1
        (rows 0..n), and E(i) ⊆ {0, ..., i-1}.  The baseline predecessor
        i-1 is implicit and should not be included (no harm but runtime.)

    return_data : bool, default False
        If True, also return the full DP tables (FlexData).

    Returns
    -------
    AlignmentResult
        Contains score, aligned sequences, row-jump events, and
        optionally the full DP state.
    """
    config = FlexInput(
        X=X,
        Y=Y,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        extra_predecessors=extra_predecessors,
        alphabet_to_index=alphabet_to_index,
    )
    return run_flex_dp(config, return_data=return_data)


# ---------------------------------------------------------------------------
# Standard NW/Gotoh (no flex) 
# ---------------------------------------------------------------------------

def align_standard(
    X: str,
    Y: str,
    score_matrix: NDArray[np.floating],
    gap_open: float,
    gap_extend: float,
    alphabet_to_index: Mapping[str, int],
    return_data: bool = False,
) -> AlignmentResult:
    """
    Standard global Needleman-Wunsch/Gotoh alignment: no extra predecessors.

    This is equivalent to NW/Gotoh with the same scoring scheme; NW-flex
    is used only as an implementation vehicle with E(i) = ∅ for all i.
    """
    n = len(X)
    EP = build_EP_standard(n)
    return align_with_EP(
        X=X,
        Y=Y,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        alphabet_to_index=alphabet_to_index,
        extra_predecessors=EP,
        return_data=return_data,
    )

# ---------------------------------------------------------------------------
# Standard NW/Gotoh (no flex) with semi-global alignment (global in Y)
# ---------------------------------------------------------------------------

def align_semiglobal(
    X: str,
    Y: str,
    score_matrix: NDArray[np.floating],
    gap_open: float,
    gap_extend: float,
    alphabet_to_index: Mapping[str, int],
    return_data: bool = False,
) -> AlignmentResult:
    """
    Semi-global alignment: global in Y, local in X.

    This uses a single flexible block spanning the reference X, via
    build_EP_semiglobal(len(X)).  Intuitively, the entire reference is
    allowed to flex (act as the Z block) while the read Y remains global.
    """
    n = len(X)
    EP = build_EP_semiglobal(n)
    return align_with_EP(
        X=X,
        Y=Y,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        alphabet_to_index=alphabet_to_index,
        extra_predecessors=EP,
        return_data=return_data,
    )


# ---------------------------------------------------------------------------
# Single-block A·Z·B configuration
# ---------------------------------------------------------------------------

def align_single_block(
    X: str,
    Y: str,
    s: int,
    e: int,
    score_matrix: NDArray[np.floating],
    gap_open: float,
    gap_extend: float,
    alphabet_to_index: Mapping[str, int],
    return_data: bool = False,
) -> AlignmentResult:
    """
    Align (X, Y) with a single flexible block X = A·Z·B.

    The block is specified via leader and end rows (s, e):

        s = |A|     (leader row index)
        e = s + |Z| (inclusive end row of Z)

    DP rows are 0..n with row i corresponding to the prefix X[0:i].

    Parameters
    ----------
    X, Y : str
        Reference and read sequences.

    s, e : int
        Leader and end rows for the flexible block, as above.  Must
        satisfy 0 ≤ s < e ≤ len(X).

    Other parameters
    ----------------
    score_matrix, gap_open, gap_extend, alphabet_to_index, return_data
        As in align_with_EP.

    Returns
    -------
    AlignmentResult
        Flex alignment for the given A·Z·B configuration.
    """
    n = len(X)
    EP = build_EP_single_block(n, s, e)
    return align_with_EP(
        X=X,
        Y=Y,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        alphabet_to_index=alphabet_to_index,
        extra_predecessors=EP,
        return_data=return_data,
    )


# ---------------------------------------------------------------------------
# STR specializations
# ---------------------------------------------------------------------------

def align_STR_block(
    X: str,
    Y: str,
    s: int,
    e: int,
    k: int,
    score_matrix: NDArray[np.floating],
    gap_open: float,
    gap_extend: float,
    alphabet_to_index: Mapping[str, int],
    return_data: bool = False,
) -> AlignmentResult:
    """
    Align (X, Y) with a single STR block Z = R^N of motif length k.

    The STR block is specified by (s, e, k):

        s = |A|       leader row index
        e = s + |Z|   end row of the repeat block (inclusive)
        k = |R|       motif length

    This uses the phase-preserving EP pattern from build_EP_STR_phase,
    which ties exits from the STR block to motif-phase classes.

    Parameters
    ----------
    X, Y : str
        Reference and read sequences.

    s, e, k : int
        STR block configuration as above.

    Other parameters
    ----------------
    score_matrix, gap_open, gap_extend, alphabet_to_index, return_data
        As in align_with_EP.

    Returns
    -------
    AlignmentResult
        Flex alignment specialized to the STR configuration.
    """
    n = len(X)
    EP = build_EP_STR_phase(n, s, e, k)
    return align_with_EP(
        X=X,
        Y=Y,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        alphabet_to_index=alphabet_to_index,
        extra_predecessors=EP,
        return_data=return_data,
    )


def align_multi_STR(
    X: str,
    Y: str,
    blocks: Sequence[Tuple[int, int, int]],
    score_matrix: NDArray[np.floating],
    gap_open: float,
    gap_extend: float,
    alphabet_to_index: Mapping[str, int],
    return_data: bool = False,
) -> AlignmentResult:
    """
    Align (X, Y) with multiple STR blocks in the same reference.

    Each STR block is given as a triple (s, e, k), and the EP pattern
    for all blocks is built by build_EP_multi_STR_phase.

    Parameters
    ----------
    X, Y : str
        Reference and read sequences.

    blocks : sequence of (s, e, k)
        STR block configurations.  For each block we require 0 ≤ s < e ≤ len(X)
        and k > 0.

    Other parameters
    ----------------
    score_matrix, gap_open, gap_extend, alphabet_to_index, return_data
        As in align_with_EP.

    Returns
    -------
    AlignmentResult
        Flex alignment for the multi-STR configuration.
    """
    n = len(X)
    EP = build_EP_multi_STR_phase(n, blocks)
    return align_with_EP(
        X=X,
        Y=Y,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        alphabet_to_index=alphabet_to_index,
        extra_predecessors=EP,
        return_data=return_data,
    )
