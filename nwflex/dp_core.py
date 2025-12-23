"""
dp_core.py — NW-flex dynamic programming core

This module implements the generic NW-flex dynamic program:
a three-state Needleman-Wunsch/Gotoh alignment with row-wise
extra predecessors E(i).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray



# ---------------------------------------------------------------------------
# Input, DP state and output containers
# ---------------------------------------------------------------------------

@dataclass
class FlexInput:
    """
    Configuration for a single NW-flex DP run.

    Attributes
    ----------
    X, Y : str
        Reference (rows) and read (columns) sequences.

    score_matrix : (K, K) array
        Substitution score matrix σ over the alphabet Σ, indexed by
        alphabet_to_index[base] for both X and Y.

    gap_open, gap_extend : float
        Affine gap parameters g_s, g_e (typically negative).

    extra_predecessors : sequence of sequences of int
        Row-wise extra predecessor sets E(i).  For each row index
        i in 1..n, extra_predecessors[i] is an iterable of row
        indices r ∈ E(i) with 0 ≤ r < i-1.  The baseline predecessor
        i-1 is implicit and does *not* need to be included here.
        
        Additionally, extra_predecessors[n+1] is the *terminal predecessor
        set*: it specifies valid starting rows for traceback (at column m).
        If empty, traceback starts at row n (global alignment). If non-empty,
        it represents the set of allowed terminal rows excluding the baseline n.
        For semi-global (local in X), extra_predecessors[n+1] = [0..n-1].

    alphabet_to_index : mapping str -> int
        Maps each base/symbol to a row/column index in score_matrix.

    free_X: bool
        If True, perform semi-global alignment local in X (reference).
        Default is False (global in X).
    free_Y: bool
        If True, perform semi-global alignment local in Y (read).
        Default is False (global in Y).
    """

    X: str
    Y: str
    score_matrix: NDArray[np.floating]
    gap_open: float
    gap_extend: float
    extra_predecessors: Sequence[Sequence[int]]
    alphabet_to_index: Mapping[str, int]
    free_X: bool = False
    free_Y: bool = False

    def pair_score(self, i: int, j: int) -> float:
        """
        This class handles the difference in indexing between sequences and the DP.
        Treats sequence positions as 1-based but returning the 0-based DP answer.
        There is a realization of the DP algorithm where the 0-index corresponds to an 
        immovable start character that must be matched between X and Y.

        Return σ(X_i, Y_j) for DP indices i,j (1-based).

        Parameters
        ----------
        i, j : int
            DP row/column indices with 1 ≤ i ≤ len(X), 1 ≤ j ≤ len(Y).

        Returns
        -------
        float
            Substitution score σ(X_i, Y_j).
        """
        xi = self.alphabet_to_index[self.X[i - 1]]
        yj = self.alphabet_to_index[self.Y[j - 1]]
        return float(self.score_matrix[xi, yj])


@dataclass
class FlexData:
    """
    DP score and traceback arrays for NW-flex.

    Score layers
    ------------
    Yg, M, Xg : (n+1, m+1) arrays of float
        Standard three Gotoh states: Yg = gap in X (horizontal),
        M = match/mismatch, Xg = gap in Y (vertical).

    Traceback layers
    ----------------
    Yg_trace, M_trace, Xg_trace : (n+1, m+1) arrays of int
        For each cell, store the predecessor state as an integer
        0 = Yg, 1 = M, 2 = Xg.  Yg_trace[i,j] refers to state at
        (i, j-1); M_trace/Xg_trace refer to the state at the
        chosen predecessor row (see M_row/Xg_row below).

    Row backpointers
    ----------------
    M_row, Xg_row : (n+1, m+1) arrays of int
        For M and Xg, store the predecessor row index.  For the
        baseline i-1 predecessor we set M_row[i,j] = i-1, and
        likewise for Xg.  When improved via an extra predecessor
        r ∈ E(i), these entries are overwritten with r.

        Yg never changes rows (predecessor row is always i), so
        it does not need a separate row array.
    """

    Yg: NDArray[np.floating]
    M : NDArray[np.floating]
    Xg: NDArray[np.floating]

    Yg_trace: NDArray[np.integer]
    M_trace : NDArray[np.integer]
    Xg_trace: NDArray[np.integer]

    M_row : NDArray[np.integer]
    Xg_row: NDArray[np.integer]

@dataclass
class RowJump:
    """
    A jump in the reference (row index) induced by extra predecessors.

    In forward DP direction this corresponds to moving from row `from_row`
    to row `to_row` at column `col` via one or more extra-predecessor edges,
    in state `state` (0 = Yg, 1 = M, 2 = Xg).

    In traceback we detect a jump when the predecessor row for an M/Xg
    state is not i-1.
    
    For terminal jumps (when traceback starts at a row other than n due to
    EP[n+1]), the jump is from the virtual terminal row n+1 to the actual
    traceback start row r, at column m. The state indicates the Gotoh layer
    at the starting cell (r, m).
    """
    from_row: int  # predecessor row index (r), or n+1 for terminal jumps
    to_row: int    # current row index (i), or actual traceback start row for terminal jumps
    col: int       # DP column index j at which we land at to_row
    state: int     # 0 = Yg, 1 = M, 2 = Xg


@dataclass
class AlignmentResult:
    """
    Result of a single NW-flex alignment run.

    Attributes
    ----------
    score : float
        Final alignment score S_flex(X, Y) = max over states at (n, m).

    X_aln, Y_aln : str
        Aligned sequences (including gaps '-').

    path : list of (i, j, state)
        DP path from start to end, with state in {0,1,2} for (Yg,M,Xg).

    jumps : list of RowJump
        Row jumps induced by extra predecessors in the chosen path.

    data : FlexData or None
        Full DP tables and traceback information, if requested.
    """
    score: float
    X_aln: str
    Y_aln: str
    path: List[Tuple[int, int, int]]
    jumps: List[RowJump]
    data: Optional[FlexData] = None

    def to_tuple(self) -> Tuple[float, str, str, List[Tuple[int, int, int]], List[RowJump]] | Tuple[float, str, str, List[Tuple[int, int, int]], List[RowJump], FlexData]:
        """
        Convert to the tuple format used by the Python implementation.

        Returns
        -------
        tuple
            (score, X_aln, Y_aln, path, jumps)
        """
        if self.data is not None:
            return (self.score, self.X_aln, self.Y_aln, self.path, self.jumps, self.data)
        else:
            return (self.score, self.X_aln, self.Y_aln, self.path, self.jumps)
    
    def expanded_alignment(self, X, Y):
        """
        Return the full alignment with jumps as gaps.
        Recalculates from original X and Y.
        """
        jump_dict = {jump.from_row: jump for jump in self.jumps}
        X_expand = []
        Y_expand = []
        for i, j, s in self.path:
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

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_matrices(config: FlexInput) -> FlexData:
    """
    Allocate and initialize the NW-flex DP arrays.

    Global alignment boundary conditions: 
    the first row corresponds to gaps in X (Yg layer),
    the first column to gaps in Y (Xg layer), with an additional
    refinement of Xg(i,0) using E(i).
    """
    n, m = len(config.X), len(config.Y)
    gs, ge = config.gap_open, config.gap_extend
    EP = config.extra_predecessors

    # Scores
    Yg = np.full((n + 1, m + 1), -np.inf, dtype=float)
    M  = np.full((n + 1, m + 1), -np.inf, dtype=float)
    Xg = np.full((n + 1, m + 1), -np.inf, dtype=float)

    # Traceback (state IDs: 0=Yg, 1=M, 2=Xg)
    Yg_trace = np.full((n + 1, m + 1), -1, dtype=int)
    M_trace  = np.full((n + 1, m + 1), -1, dtype=int)
    Xg_trace = np.full((n + 1, m + 1), -1, dtype=int)

    # Row backpointers for M and Xg
    M_row  = np.full((n + 1, m + 1), -1, dtype=int)
    Xg_row = np.full((n + 1, m + 1), -1, dtype=int)

    # Default predecessor row is i-1 for M and Xg
    rows = np.arange(n + 1)
    M_row [rows, :] = (rows - 1)[:, None]
    Xg_row[rows, :] = (rows - 1)[:, None]

    # Initialization at (0,0)
    M[0, 0] = 0.0

    # First row: gaps in X (Yg layer)
    if config.free_Y:
        # Semi-global in Y: allow starting with gaps in X (Yg layer)
        for j in range(1, m + 1):
            Yg[0, j] = 0.0
            Yg_trace[0, j] = 0  # predecessor is from the left (0, j-1)
    else:
        # Global in Y: standard affine gap initialization                
        for j in range(1, m + 1):
            Yg      [0, j] = gs + (j - 1) * ge
            Yg_trace[0, j] = 0  # predecessor is from the left (0, j-1)

    # First column: gaps in Y (Xg layer), including EP refinement
    if config.free_X:
        # Semi-global in X: allow starting with gaps in Y (Xg layer)
        for i in range(1, n + 1):
            Xg[i, 0] = 0.0
            Xg_trace[i, 0] = 2
    else:
        for i in range(1, n + 1):
            # baseline predecessor i-1
            cand1 = M[i - 1, 0] + gs
            cand2 = Xg[i - 1, 0] + ge
            Xg[i, 0] = max(cand1, cand2)
            Xg_trace[i, 0] = 2  # predecessor is from above (i-1, 0)

            # refine with extra predecessors along the first column, if any
            for r in EP[i]:
                candX = max(M[r, 0] + gs, Xg[r, 0] + ge) # max is needed for the case r=0
                if candX > Xg[i, 0]:
                    Xg[i, 0] = candX
                    Xg_row[i, 0] = r

    return FlexData(
        Yg=Yg,
        M=M,
        Xg=Xg,
        Yg_trace=Yg_trace,
        M_trace=M_trace,
        Xg_trace=Xg_trace,
        M_row=M_row,
        Xg_row=Xg_row,
    )


# ---------------------------------------------------------------------------
# Per-cell updates
# ---------------------------------------------------------------------------

def baseline_update(config: FlexInput, data: FlexData, i: int, j: int) -> None:
    """
    Standard three-state Gotoh update at cell (i,j) using only row i-1.

    This fills Yg(i,j), M(i,j), Xg(i,j), their state traces, and sets
    the default predecessor row for M and Xg to i-1.
    """
    Yg, M, Xg = data.Yg, data.M, data.Xg
    Yg_t, M_t, Xg_t = data.Yg_trace, data.M_trace, data.Xg_trace
    M_row, Xg_row = data.M_row, data.Xg_row
    gs, ge = config.gap_open, config.gap_extend

    score = config.pair_score(i, j)

    # Yg: gap in X (move left, same row)
    y_candidates = (
        Yg[i, j - 1] + ge,  # extend gap in X
        M [i, j - 1] + gs,  # open from M
        Xg[i, j - 1] + gs,  # open from Xg
    )
    y_state    = int(np.argmax(y_candidates))
    Yg  [i, j] = y_candidates[y_state]
    Yg_t[i, j] = y_state

    # M: match/mismatch (diagonal from row i-1)
    m_candidates = (
        Yg[i - 1, j - 1],
        M [i - 1, j - 1],
        Xg[i - 1, j - 1],
    )
    m_state     = int(np.argmax(m_candidates))
    M    [i, j] = m_candidates[m_state] + score
    M_t  [i, j] = m_state
    M_row[i, j] = i - 1  # baseline predecessor row (already set, but for clarity)

    # Xg: gap in Y (move down, from row i-1)
    x_candidates = (
        Yg[i - 1, j] + gs,  # open from Yg
        M [i - 1, j] + gs,  # open from M
        Xg[i - 1, j] + ge,  # extend gap in Y
    )
    x_state      = int(np.argmax(x_candidates))
    Xg    [i, j] = x_candidates[x_state]
    Xg_t  [i, j] = x_state
    Xg_row[i, j] = i - 1  # baseline predecessor row (already set, but for clarity)


def refine_with_extra_predecessors(
    config: FlexInput,
    data: FlexData,
    i: int,
    j: int,
) -> None:
    """
    Refine M(i,j) and Xg(i,j) using extra predecessor rows E(i).

    For each r in E(i), we recompute M and Xg candidates using row r
    in place of row i-1, and keep the best.  
    Yg(i,j) is never modified by extra predecessors.
    """
    Yg, M, Xg = data.Yg, data.M, data.Xg
    M_t, Xg_t = data.M_trace, data.Xg_trace
    M_row, Xg_row = data.M_row, data.Xg_row
    gs, ge = config.gap_open, config.gap_extend

    score = config.pair_score(i, j)

    for r in config.extra_predecessors[i]:
        # M from row r
        m_candidates = (
            Yg[r, j - 1],
            M [r, j - 1],
            Xg[r, j - 1],
        )
        m_state = int(np.argmax(m_candidates))
        candM   = m_candidates[m_state] + score

        if candM > M[i, j]:
            M   [i, j] = candM
            M_t  [i, j] = m_state
            M_row[i, j] = r

        # Xg from row r
        x_candidates = (
            Yg[r, j] + gs,
            M [r, j] + gs,
            Xg[r, j] + ge,
        )        
        x_state = int(np.argmax(x_candidates))
        candX   = x_candidates[x_state]

        if candX > Xg[i, j]:
            Xg    [i, j] = candX
            Xg_t  [i, j] = x_state
            Xg_row[i, j] = r


# ---------------------------------------------------------------------------
# Traceback
# ---------------------------------------------------------------------------

def traceback_alignment(
    config: FlexInput,
    data: FlexData,
) -> Tuple[str, str, List[Tuple[int, int, int]], List[RowJump]]:
    """
    Recover one best flex alignment from the filled DP tables.

    The starting cell for traceback is determined by EP[n+1], the terminal
    predecessor set.  If EP[n+1] is empty, traceback starts at (n, m) as in
    global alignment.  Otherwise, we select the best (row, state) at column m
    among all rows in EP[n+1] ∪ {n} (n is the implicit baseline).

    Returns
    -------
    X_aln, Y_aln : str
        Aligned sequences (including gaps '-').
    best_score: float
        Best alignment score.
    path : list of (i, j, state)
        DP path from start to end, with state in {0,1,2} for (Yg,M,Xg).
    jumps : list of RowJump
        Row jumps induced by extra predecessors.  Each jump records a
        transition from from_row -> to_row at column col, in state M or Xg.
        If traceback starts at row r != n due to EP[n+1], this is recorded
        as a jump from n to r at column m (state = -1 indicates terminal jump).
    """
    Yg, M, Xg = data.Yg, data.M, data.Xg
    Yg_t, M_t, Xg_t = data.Yg_trace, data.M_trace, data.Xg_trace
    M_row, Xg_row = data.M_row, data.Xg_row

    n, m = len(config.X), len(config.Y)
    EP = config.extra_predecessors

    best_score = -np.inf
    best_col   = m
    best_row   = n
    best_state = 1  # default to M
    if config.free_Y:
        # Semi-global in Y: allow ending at any column in row n        
        for j in range(m + 1):
            for s, score in enumerate([Yg[n, j], M[n, j], Xg[n, j]]):
                if score > best_score:
                    best_score = score
                    best_col = j
                    best_state = s

    ## 
    best_col_score = best_score
    
    ## We either use semi-global in X or check the terminal predecessor set EP[n+1]
    if config.free_X:
        # Semi-global in X: allow ending at any row in column best_col
        for i in range(n + 1):
            for s, score in enumerate([Yg[i, m], M[i, m], Xg[i, m]]):
                if score > best_score:
                    best_score = score
                    best_row = i
                    best_state = s
    else:
        ## check terminal predecessor set EP[n+1], or 
        ## at minimum, the value at n, m.
        if len(EP) > n + 1 and EP[n + 1]:
            terminal_rows = list(EP[n + 1]) + [n]  # baseline n is implicit
        else:
            terminal_rows = [n]  # global alignment in X: only row n

        for r in terminal_rows:
            for s, score in enumerate([Yg[r, m], M[r, m], Xg[r, m]]):
                if score > best_score:
                    best_score = score
                    best_row = r
                    best_state = s

    ## we have checked local in X and Y and terminal predecessors if any
    if best_score == best_col_score:
        best_row = n  # for local in Y, default to row n
    else:
        best_col = m  # for local in X, default to column m
    
    i, j, state = best_row, best_col, best_state

    aln_X: List[str] = []
    aln_Y: List[str] = []
    path: List[Tuple[int, int, int]] = []
    jumps: List[RowJump] = []

    # Record terminal jump if we started at a row other than n
    # and were not in semi-global in X mode
    # The jump is from virtual row n+1 to the actual traceback start row
    if best_row != n:
        if not config.free_X:
            jumps.append(RowJump(
                from_row=int(n + 1),  # virtual terminal row
                to_row=int(best_row),
                col=int(m),
                state=int(best_state),  # state at the actual starting cell
            ))
        else:
            # if we are in semi-global in X, we need to put in gaps to reach row n
            for gap_i in range(n, best_row, -1):
                aln_X.append(config.X[gap_i - 1])
                aln_Y.append('-')
                path.append((gap_i, j, 2))  # Xg state
    elif best_col != m:
        # then we are in semi-global in Y, we need to put in gaps to reach column m
        for gap_j in range(m, best_col, -1):
            aln_X.append('-')
            aln_Y.append(config.Y[gap_j - 1])
            path.append((i, gap_j, 0))  # Yg state


    while i > 0 or j > 0:
        path.append((i, j, state))

        if state == 0:  # Yg: gap in X (move left, same row)
            prev_state = Yg_t[i, j]
            aln_X.append('-')
            aln_Y.append(config.Y[j - 1])
            j -= 1
            state = prev_state

        elif state == 1:  # M: match/mismatch (move diagonally, possibly jump rows)
            prev_row = M_row[i, j]
            prev_state = M_t[i, j]
            if prev_row != i - 1:
                jumps.append(RowJump(
                    from_row=int(prev_row),
                    to_row=int(i),
                    col=int(j),
                    state=1,
                ))
            aln_X.append(config.X[i - 1])
            aln_Y.append(config.Y[j - 1])
            i, j, state = prev_row, j - 1, prev_state

        else:  # state == 2, Xg: gap in Y (move up within column, possibly jump rows)
            prev_row = Xg_row[i, j]
            prev_state = Xg_t[i, j]
            if prev_row != i - 1:
                jumps.append(RowJump(
                    from_row=int(prev_row),
                    to_row=int(i),
                    col=int(j),
                    state=2,
                ))
            aln_X.append(config.X[i - 1])
            aln_Y.append('-')
            i, state = prev_row, prev_state

    aln_X.reverse()
    aln_Y.reverse()
    path.reverse()
    jumps.reverse()

    return ''.join(aln_X), ''.join(aln_Y), best_score, path, jumps


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_flex_dp(
    config: FlexInput,
    return_data: bool = False,
) -> AlignmentResult:
    """
    Run the NW-flex DP for a given FlexInput and return the best alignment.

    Parameters
    ----------
    config : FlexInput
        Sequences, scoring scheme, and extra predecessor sets E(i).
    return_data : bool, default False
        If True, also return the full DP/traceback arrays (FlexData).

    Returns
    -------
    AlignmentResult
        Contains:
          - score
          - X_aln, Y_aln
          - path (list[(i,j,state)])
          - jumps (list[RowJump])
          - data (FlexData or None)
    """
    n, m = len(config.X), len(config.Y)
    EP = config.extra_predecessors
    data = init_matrices(config)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            baseline_update(config, data, i, j)
            refine_with_extra_predecessors(config, data, i, j)

    X_aln, Y_aln, final_score, path, jumps = traceback_alignment(config, data)
    if return_data:
        return AlignmentResult(
            score=final_score,
            X_aln=X_aln,
            Y_aln=Y_aln,
            path=path,
            jumps=jumps,
            data=data,
        )
    else:
        return AlignmentResult(
            score=final_score,
            X_aln=X_aln,
            Y_aln=Y_aln,
            path=path,
            jumps=jumps,
            data=None,
        )

