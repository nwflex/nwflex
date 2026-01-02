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

from typing import Iterable, Mapping, Sequence, Tuple, Set, Optional
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import re
from itertools import groupby

from nwflex.dp_core import AlignmentResult
from nwflex.repeats import STRLocus

from .dp_core import FlexInput, run_flex_dp
from .fast import run_flex_dp_fast

from .ep_patterns import (
    build_EP_standard,
    build_EP_single_block,
    build_EP_STR_phase,
    build_EP_multi_STR_phase,
    build_EP_semiglobal,
)

## stuff for CIGAR parsing and writing
_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")  # regex to parse CIGAR strings

def _parse_cigar(cigar: str) -> Sequence[Tuple[int, str]]:
    """Parse CIGAR string into list of (count, op) tuples."""
    parts = [(int(n), op) for n, op in _CIGAR_RE.findall(cigar)]
    ## rebuild to validate
    if "".join(f"{n}{op}" for n, op in parts) != cigar:
        raise ValueError(f"Invalid CIGAR: {cigar!r}")
    return parts

def _op_length_total(cigar: str, ops: Set[str]) -> int:
    """Count number of bases in CIGAR for given operations."""
    parts = _parse_cigar(cigar)
    return sum(n for n, op in parts if op in ops)

def _write_cigar(parts: Sequence[Tuple[int, str]]) -> str:
    """Write CIGAR string from list of (count, op) tuples."""
    return "".join(f"{n}{op}" for n, op in parts)


def rle_ops(ops: str) -> Sequence[Tuple[int, str]]:
    """Run-length encode CIGAR operations string."""
    return [(sum(1 for _ in group), op) for op, group in groupby(ops)]


def alignment_to_cigar(
        path: Sequence[Tuple[int, int, int]],
        lenX: Optional[int] = None,
        lenY: Optional[int] = None,
) -> Tuple[int, str]:
    """
    Convert an AlignmentResult path to a CIGAR string.

    Parameters
    ----------
    path : sequence of (i, j, state)
        Alignment path as returned in AlignmentResult.path.
    
    lenX, lenY : int
        Lengths of the reference and read sequences.
    If not provided, they are inferred from the path.

    Returns
    -------
    start_index : int
        Position of first aligned base

    cigar : str
        CIGAR string representing the alignment.
    """
    if lenX is None:
        lenX = max(i for i, j, s in path)
    if lenY is None:
        lenY = max(j for i, j, s in path)        
    pi = 0
    cigar_states = []
    start_pos = -1
    for i, j, state in path:
        ## skip leading deletions
        if j == 0:
            pi = i
            continue
        ## soft-clip leading insertions
        if i == 0:
            cigar_states.append('S')
            continue
        if j == lenY and state == 2:
            break
        ## the first algined base is the start position
        if j > 0 and i > 0 and start_pos == -1:
            start_pos = i    
        ## if there was a jump, record Ns
        if i - pi > 1:
            cigar_states.extend(['N'] * (i - pi - 1))  
        ## if there is a gap in X  
        if state == 0:
            ## if we are at the end of X, soft clip
            ## otherwise, insertion
            if i == lenX:
                cigar_states.append('S')
            else:
                cigar_states.append('I')
        elif state == 1:
            ## match/mismatch
            cigar_states.append('M')
        else:
            ## gap in Y
            cigar_states.append('D')
        pi= i

    cigar_list = rle_ops(cigar_states)
    cigar_str = _write_cigar(cigar_list)
    return int(start_pos), cigar_str




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
    free_X: bool = False,
    free_Y: bool = False,
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

    free_X: semiglobal in X (allow free gaps at the ends of X)

    free_Y: semiglobal in Y (allow free gaps at the ends of Y)

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
        free_X=free_X,
        free_Y=free_Y,
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
    free_X: bool = False,
    free_Y: bool = False,
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
        free_X=free_X,
        free_Y=free_Y,
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

    We do NOT turn on semi-global alignment intialization conditions.
    This function is really for demonstration purposes.
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
    free_X: bool = False,
    free_Y: bool = False,
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
    score_matrix, gap_open, gap_extend, alphabet_to_index, return_data, free_X, free_Y
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
        free_X=free_X,
        free_Y=free_Y,
        return_data=return_data,
    )


# ---------------------------------------------------------------------------
# STR specializations
# ---------------------------------------------------------------------------

def align_STR_block(
        strLocus: STRLocus,
        Y: str,
        score_matrix: NDArray[np.floating],
        gap_open: float,
        gap_extend: float,
        alphabet_to_index: Mapping[str, int],
        free_X: bool = False,
        free_Y: bool = False,
        return_data: bool = False,
    ) -> AlignmentResult:
    """
    Align (X, Y) with a single STR block Z = R^N of motif length k.
    X, R, N and k are specified by the STRLocus object.

    The STR block is specified by (s, e, k):

        s = |A|       leader row index (determined by STRLocus)
        e = s + |Z|   end row of the repeat block (inclusive)
        k = |R|       motif length

    This uses the phase-preserving EP pattern from build_EP_STR_phase,
    which ties exits from the STR block to motif-phase classes.

    Parameters
    ----------
    strLocus: STRLocus
        The STR locus object specifying reference flanks, motif, and repeat count.
    Y: str
        The read sequence to be aligned.

    Other parameters
    ----------------
    score_matrix, gap_open, gap_extend, alphabet_to_index, return_data, free_X, free_Y
        As in align_with_EP.

    Returns
    -------
    AlignmentResult
        Flex alignment specialized to the STR configuration.
    """
    n = strLocus.n
    EP = build_EP_STR_phase(strLocus.n, 
                            strLocus.s, 
                            strLocus.e, 
                            strLocus.k)
    return align_with_EP(
        X=strLocus.X,
        Y=Y,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        alphabet_to_index=alphabet_to_index,
        extra_predecessors=EP,
        free_X=free_X,
        free_Y=free_Y,
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
    free_X: bool = False,
    free_Y: bool = False,
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
    score_matrix, gap_open, gap_extend, alphabet_to_index, return_data, free_X, free_Y
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
        free_X=free_X,
        free_Y=free_Y,
        return_data=return_data,
    )

class RefAligner:
    """
    Given a common reference sequence X, scoring parameters, 
    extra_predecessors, and alphabet, 
    create an object that can align multiple reads Y1, Y2, ... against X.
    """

    def __init__(
        self,
        ref: str,
        extra_predecessors: Sequence[Sequence[int]],
        score_matrix: NDArray[np.floating],
        gap_open: float,
        gap_extend: float,
        alphabet_to_index: Mapping[str, int],
        free_X: bool = False,
        free_Y: bool = False,
        return_data: bool = False,
        fast_mode: bool = False,
    ):
        self.config = dict()
        self.config["X"] = ref
        self.config["extra_predecessors"] = extra_predecessors                
        self.config["score_matrix"]      = score_matrix
        self.config["gap_open"]          = gap_open
        self.config["gap_extend"]        = gap_extend
        self.config["alphabet_to_index"] = alphabet_to_index
        self.config["free_X"] = free_X
        self.config["free_Y"] = free_Y
        self.return_data = return_data
        self.flex_dp = run_flex_dp_fast if fast_mode else run_flex_dp
        self.reflen = len(ref)

    def align(self, read: str) -> AlignmentResult:
        """
        Align a read Y against the reference X provided at initialization.
        """
        flex_input = FlexInput(**self.config, Y=read)
        return self.flex_dp(
            flex_input,
            return_data=self.return_data,
        )
    
    def align_batch(self, reads: Iterable[str]) -> Iterable[AlignmentResult]:
        """
        Align a batch of reads against the reference X provided at initialization.
        """
        for read in reads:
            yield self.align(read)

    def simple_output(self, result: AlignmentResult, readlen: Optional[int] = None) -> dict:
        """
        Convert an AlignmentResult into a simple dict with score and CIGAR.
        """
        start_pos, cigar = alignment_to_cigar(
            result.path,
            lenX=self.reflen,
            lenY=readlen,
        )
        ## number of aligned bases in read
        aligned_bases = _op_length_total(cigar, {'M'})        
        return {
            "score": result.score,
            "aligned_bases": aligned_bases,
            "start_pos": start_pos,
            "cigar": cigar,
        }

    def align_simple(self, read: str) -> dict:
        """
        Align a read and return a simple dict with score and CIGAR.
        """
        result = self.align(read)
        ans = self.simple_output(result, readlen=len(read))
        return ans
    
    def align_batch_simple(self, reads: Iterable[str]) -> Iterable[dict]:
        """
        Align a batch of reads and return simple dicts with score and CIGAR.
        """
        for read in reads:
            result = self.align(read)
            ans = self.simple_output(result, readlen=len(read))
            yield ans
