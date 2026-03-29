# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
nwflex_dp.pyx — Cython core for NW-flex DP

This module exposes a low-level DP kernel that implements the NW-flex
recurrences from row-wise extra predecessor sets E(i). It is intended to
be called from a thin Python wrapper that:

  * encodes X and Y as integer codes (alphabet indices),
  * converts E(i) from a list-of-lists into interval arrays
    (ep_counts, ep_starts, ep_ends),
  * and then wraps the returned DP arrays into FlexData / AlignmentResult.

We keep this module focused on numeric work only: no string handling,
no Python-level EP structures.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.string cimport memset

ctypedef cnp.float32_t DTYPE_t
ctypedef cnp.int32_t   ITYPE_t


cdef class DPBuffers:
    """Pre-allocated buffers for NW-flex DP to enable reuse across calls."""
    cdef public int max_nrows, max_ncols

    # Score matrices (float64)
    cdef public object Yg, M, Xg

    # Trace matrices (int32)
    cdef public object Yg_tr, M_tr, Xg_tr

    # Row predecessor matrices (int32)
    cdef public object M_row, Xg_row

    # Path buffer (int32, shape max_path x 3)
    cdef public object path

    # CIGAR ops buffer (uint8, for single-char ops)
    cdef public object cigar_ops

    # Precomputed row_default for fast M_row/Xg_row init
    cdef public object row_default
    cdef ITYPE_t[:] row_default_view

    def __init__(self, int max_nrows, int max_ncols):
        self.max_nrows = max_nrows
        self.max_ncols = max_ncols

        # Score matrices
        self.Yg = np.empty((max_nrows, max_ncols), dtype=np.float32)
        self.M  = np.empty((max_nrows, max_ncols), dtype=np.float32)
        self.Xg = np.empty((max_nrows, max_ncols), dtype=np.float32)

        # Trace matrices
        self.Yg_tr = np.empty((max_nrows, max_ncols), dtype=np.int32)
        self.M_tr  = np.empty((max_nrows, max_ncols), dtype=np.int32)
        self.Xg_tr = np.empty((max_nrows, max_ncols), dtype=np.int32)

        # Row predecessor matrices
        self.M_row  = np.empty((max_nrows, max_ncols), dtype=np.int32)
        self.Xg_row = np.empty((max_nrows, max_ncols), dtype=np.int32)

        # Path buffer: max path length is max_nrows + max_ncols
        self.path = np.empty((max_nrows + max_ncols, 3), dtype=np.int32)

        # CIGAR ops buffer: one byte per op, max length is max_nrows + max_ncols
        self.cigar_ops = np.empty(max_nrows + max_ncols, dtype=np.uint8)

        # Precomputed row_default: row_default[i] = i - 1
        self.row_default = np.arange(max_nrows, dtype=np.int32) - 1
        self.row_default_view = self.row_default

    def ensure_size(self, int nrows, int ncols):
        """Reallocate if buffers are too small. Returns True if reallocated."""
        if nrows > self.max_nrows or ncols > self.max_ncols:
            self.__init__(max(nrows, self.max_nrows), max(ncols, self.max_ncols))
            return True
        return False


@cython.boundscheck(False)
@cython.wraparound(False)
def nwflex_dp_core(
    cnp.ndarray[ITYPE_t, ndim=1] X_codes,
    cnp.ndarray[ITYPE_t, ndim=1] Y_codes,
    cnp.ndarray[DTYPE_t, ndim=2] score_matrix,
    double gap_open,
    double gap_extend,
    cnp.ndarray[ITYPE_t, ndim=1] ep_counts,
    cnp.ndarray[ITYPE_t, ndim=2] ep_starts,
    cnp.ndarray[ITYPE_t, ndim=2] ep_ends,
    bint free_X = False,
    bint free_Y = False,
    bint return_path = False,
    bint return_matrices = True,
):
    """
    Low-level NW-flex DP core with row-wise extra predecessors encoded as intervals.

    This is a Cython translation of the Python dp_core:
      - Three Gotoh states per cell: Yg, M, Xg.
      - Standard baseline recurrences from row i-1.
      - For each row i, extra predecessors E(i) are given as intervals
        in ep_starts/ep_ends, and used to refine M(i,j) and Xg(i,j).

    Parameters
    ----------
    X_codes, Y_codes : int32[1D]
        Encoded sequences X and Y (alphabet indices).
    score_matrix : float64[2D]
        Substitution matrix sigma[a,b].
    gap_open, gap_extend : float
        Gap penalties g_s (open), g_e (extend).
    ep_counts : int32[1D], shape (n+1,)
        ep_counts[i] = number of intervals for DP row i.
    ep_starts, ep_ends : int32[2D], shape (n+1, max_intervals)
        For row i, intervals are [ep_starts[i,k], ep_ends[i,k]] (inclusive)
        for k = 0..ep_counts[i]-1.
    free_X, free_Y : bool
        If True, perform semi-global alignment local in X or Y, respectively.
    return_path : bool
        If True, perform traceback internally and return (score, path_array).
    return_matrices : bool
        If True, include DP and traceback matrices in the return value.

    Returns
    -------
    Yg, M, Xg : float64[2D]
        Score matrices for the three states.
    Yg_trace, M_trace, Xg_trace : int32[2D]
        Traceback state IDs (0=Yg, 1=M, 2=Xg).
    M_row, Xg_row : int32[2D]
        Traceback predecessor row indices for M and Xg.
    """
    cdef int n = X_codes.shape[0]
    cdef int m = Y_codes.shape[0]
    cdef int nrows = n + 1
    cdef int ncols = m + 1

    # Typed views onto inputs (for cheap access in loops)
    cdef ITYPE_t[:] Xv = X_codes
    cdef ITYPE_t[:] Yv = Y_codes
    cdef DTYPE_t[:, :] S = score_matrix

    cdef ITYPE_t[:] ep_c = ep_counts
    cdef ITYPE_t[:, :] ep_s = ep_starts
    cdef ITYPE_t[:, :] ep_e = ep_ends

    # "Minus infinity" sentinel for unusable cells
    cdef DTYPE_t NEG_INF = <DTYPE_t>(-1e300)

    # Allocate numpy arrays (these are returned to Python)
    cdef cnp.ndarray[DTYPE_t, ndim=2] Yg_np = np.empty((nrows, ncols), dtype=np.float32)
    cdef cnp.ndarray[DTYPE_t, ndim=2]  M_np = np.empty((nrows, ncols), dtype=np.float32)
    cdef cnp.ndarray[DTYPE_t, ndim=2] Xg_np = np.empty((nrows, ncols), dtype=np.float32)

    cdef cnp.ndarray[ITYPE_t, ndim=2] Yg_tr_np = np.empty((nrows, ncols), dtype=np.int32)
    cdef cnp.ndarray[ITYPE_t, ndim=2]  M_tr_np = np.empty((nrows, ncols), dtype=np.int32)
    cdef cnp.ndarray[ITYPE_t, ndim=2] Xg_tr_np = np.empty((nrows, ncols), dtype=np.int32)

    cdef cnp.ndarray[ITYPE_t, ndim=2]  M_row_np = np.empty((nrows, ncols), dtype=np.int32)
    cdef cnp.ndarray[ITYPE_t, ndim=2] Xg_row_np = np.empty((nrows, ncols), dtype=np.int32)

    # Memoryview aliases for fast C-style indexing inside loops
    cdef DTYPE_t[:, :] Yg = Yg_np
    cdef DTYPE_t[:, :]  M = M_np
    cdef DTYPE_t[:, :] Xg = Xg_np

    cdef ITYPE_t[:, :] Yg_tr = Yg_tr_np
    cdef ITYPE_t[:, :]  M_tr =  M_tr_np
    cdef ITYPE_t[:, :] Xg_tr = Xg_tr_np

    cdef ITYPE_t[:, :] M_row  = M_row_np
    cdef ITYPE_t[:, :] Xg_row = Xg_row_np

    # Loop indices and temporaries
    cdef int i, j, k, r
    cdef DTYPE_t gs = gap_open
    cdef DTYPE_t ge = gap_extend
    cdef DTYPE_t c0, c1, c2, best, score, candM, candX
    cdef ITYPE_t st
    cdef ITYPE_t xi, yj

    # Traceback variables
    cdef DTYPE_t best_score
    cdef DTYPE_t best_col_score
    cdef ITYPE_t best_row
    cdef ITYPE_t best_col
    cdef ITYPE_t best_state
    cdef ITYPE_t prev_row
    cdef ITYPE_t state
    cdef int max_path
    cdef int path_len
    cdef int left, right
    cdef ITYPE_t tmp0, tmp1, tmp2
    cdef cnp.ndarray[ITYPE_t, ndim=2] path_np
    cdef ITYPE_t[:, :] path
    cdef Py_ssize_t ep_len

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    # Fill all cells with NEG_INF and traces with -1
    for i in range(nrows):
        for j in range(ncols):
            Yg[i, j] = NEG_INF
            M[i, j]  = NEG_INF
            Xg[i, j] = NEG_INF
            Yg_tr[i, j] = -1
            M_tr[i, j]  = -1
            Xg_tr[i, j] = -1
            ## initialize M_row and Xg_row to default predecessor row i-1
            M_row[i, j] = i - 1
            Xg_row[i, j] = i - 1

    # Starting cell: only M(0,0) is 0
    M[0, 0] = 0.0

    # First row: gaps in X (horizontal, Yg state)
    # Check if free_Y (semi-global in Y)
    if free_Y:
        for j in range(1, ncols):
            Yg[0, j] = 0.0
            Yg_tr[0, j] = 0  # predecessor is Yg(0, j-1)
    else:
        # Yg(0, j) = g_s + (j-1)*g_e, j >= 1
        for j in range(1, ncols):
            Yg[0, j] = gs + (j - 1) * ge
            Yg_tr[0, j] = 0  # predecessor is Yg(0, j-1)

    # First column: gaps in Y (vertical, Xg state) with EP refinement
    # Check if free_X (semi-global in X)
    if free_X:
        for i in range(1, nrows):
            Xg[i, 0] = 0.0
            Xg_tr[i, 0] = 2  # predecessor is Xg(i-1, 0)
    else:
        # Baseline: Xg(i,0) from row i-1; then refine using E(i)
        for i in range(1, nrows):
            # baseline predecessor row i-1
            c1 = M[i - 1, 0] + gs
            c2 = Xg[i - 1, 0] + ge
            Xg_tr[i, 0] = 2  # default predecessor is (i-1,0)
            if c1 >= c2:
                Xg[i, 0] = c1            
            else:
                Xg[i, 0] = c2
            # Xg_row[i, 0] = i - 1 (default by initialization)

            # refine with extra predecessors on this column
            if ep_c[i] > 0:
                for k in range(ep_c[i]):
                    r = ep_s[i, k]
                    while r <= ep_e[i, k]:
                        c1 = M[r, 0] + gs
                        c2 = Xg[r, 0] + ge
                        if c1 >= c2:
                            candX = c1
                        else:
                            candX = c2
                        if candX > Xg[i, 0]:
                            Xg[i, 0] = candX
                            Xg_row[i, 0] = r
                        r += 1

    # ------------------------------------------------------------------
    # Main DP: rows 1..n, columns 1..m
    # ------------------------------------------------------------------
    for i in range(1, nrows):
        xi = Xv[i - 1]
        for j in range(1, ncols):
            yj = Yv[j - 1]
            score = S[xi, yj]

            # ---------------- Yg(i,j): gap in X (horizontal) ----------------
            # From (i, j-1), states Yg/M/Xg
            c0 = Yg[i, j - 1] + ge
            c1 = M [i, j - 1] + gs
            c2 = Xg[i, j - 1] + gs

            best = c0
            st = 0
            if c1 > best:
                best = c1
                st = 1
            if c2 > best:
                best = c2
                st = 2

            Yg[i, j] = best
            Yg_tr[i, j] = st

            # ---------------- M(i,j): match/mismatch ----------------
            # Baseline from row i-1, col j-1
            c0 = Yg[i - 1, j - 1]
            c1 = M [i - 1, j - 1]
            c2 = Xg[i - 1, j - 1]

            best = c0
            st = 0
            if c1 > best:
                best = c1
                st = 1
            if c2 > best:
                best = c2
                st = 2

            M[i, j] = best + score
            M_tr[i, j] = st
            #M_row[i, j] = i - 1  # predecessor row (default by initialization)

            # ---------------- Xg(i,j): gap in Y (vertical) ----------------
            # Baseline from row i-1, col j
            c0 = Yg[i - 1, j] + gs
            c1 = M [i - 1, j] + gs
            c2 = Xg[i - 1, j] + ge

            best = c0
            st = 0
            if c1 > best:
                best = c1
                st = 1
            if c2 > best:
                best = c2
                st = 2

            Xg[i, j] = best
            Xg_tr[i, j] = st
            #Xg_row[i, j] = i - 1  # predecessor row (default by initialization)

            # ---------------- EP refinement: rows r ∈ E(i) ----------------
            # For each extra predecessor row r, refine M(i,j) and Xg(i,j)
            if ep_c[i] > 0:
                for k in range(ep_c[i]):
                    r = ep_s[i, k]
                    while r <= ep_e[i, k]:
                        # M(i,j) candidate from row r, col j-1
                        c0 = Yg[r, j - 1]
                        c1 = M [r, j - 1]
                        c2 = Xg[r, j - 1]
                        best = c0
                        st = 0
                        if c1 > best:
                            best = c1
                            st = 1
                        if c2 > best:
                            best = c2
                            st = 2
                        candM = best + score
                        if candM > M[i, j]:
                            M[i, j] = candM
                            M_tr[i, j] = st
                            M_row[i, j] = r

                        # Xg(i,j) candidate from row r, col j
                        c0 = Yg[r, j] + gs
                        c1 = M [r, j] + gs
                        c2 = Xg[r, j] + ge
                        best = c0
                        st = 0
                        if c1 > best:
                            best = c1
                            st = 1
                        if c2 > best:
                            best = c2
                            st = 2
                        candX = best
                        if candX > Xg[i, j]:
                            Xg[i, j] = candX
                            Xg_tr[i, j] = st
                            Xg_row[i, j] = r

                        r += 1

    # ------------------------------------------------------------------
    # Optional traceback in Cython
    # ------------------------------------------------------------------
    if return_path:
        # Determine traceback start cell (mirrors dp_core.traceback_alignment)
        best_score = NEG_INF
        best_row = n
        best_col = m
        best_state = 1                  # default to M state
        best_col_score = best_score     # for free_Y case

        if free_Y:
            # semi-global in Y: allow ending at any column in row n
            for j in range(ncols):
                # check Yg, M, Xg at (n, j)
                c0 = Yg[n, j]
                if c0 > best_score:
                    best_score = c0
                    best_row = n
                    best_col = j
                    best_state = 0
                c1 = M[n, j]
                if c1 > best_score:
                    best_score = c1
                    best_row = n
                    best_col = j
                    best_state = 1
                c2 = Xg[n, j]
                if c2 > best_score:
                    best_score = c2
                    best_row = n
                    best_col = j
                    best_state = 2
            # end of semi-global in Y, store best column score
            best_col_score = best_score

        if free_X:
            # semi-global in X: allow ending at any row in column m
            for i in range(nrows):
                # as above, check Yg, M, Xg at (i, m)
                c0 = Yg[i, m]
                if c0 > best_score:
                    best_score = c0
                    best_row = i
                    best_col = m
                    best_state = 0
                c1 = M[i, m]
                if c1 > best_score:
                    best_score = c1
                    best_row = i
                    best_col = m
                    best_state = 1
                c2 = Xg[i, m]
                if c2 > best_score:
                    best_score = c2
                    best_row = i
                    best_col = m
                    best_state = 2
        else:
            # Global in X: include terminal predecessors if provided.
            # Check (n, m) first (global)
            c0 = Yg[n, m]
            if c0 > best_score:
                best_score = c0
                best_row = n
                best_col = m
                best_state = 0
            c1 = M[n, m]
            if c1 > best_score:
                best_score = c1
                best_row = n
                best_col = m
                best_state = 1
            c2 = Xg[n, m]
            if c2 > best_score:
                best_score = c2
                best_row = n
                best_col = m
                best_state = 2
            # Then check extra predecessors for row n+1
            ep_len = ep_c.shape[0]
            # check if there are extra predecessors for row n+1
            if ep_len > nrows and ep_c[nrows] > 0:
                ## for each extra predecessor, check (r, m)
                for k in range(ep_c[nrows]):
                    r = ep_s[nrows, k]
                    while r <= ep_e[nrows, k]:
                        c0 = Yg[r, m]
                        if c0 > best_score:
                            best_score = c0
                            best_row = r
                            best_col = m
                            best_state = 0
                        c1 = M[r, m]
                        if c1 > best_score:
                            best_score = c1
                            best_row = r
                            best_col = m
                            best_state = 1
                        c2 = Xg[r, m]
                        if c2 > best_score:
                            best_score = c2
                            best_row = r
                            best_col = m
                            best_state = 2
                        r += 1

        # If the best score is still the one found by the free_Y scan (row n),
        # we keep row n and its best column. Otherwise the best score came from
        # the free_X/terminal-row search, so we fix column m and keep that row.
        if best_score == best_col_score:
            best_row = n
        else:
            best_col = m

        # Traceback path (stored in reverse, then reversed in place)
        # Allocate max possible path length: n + m + 2
        max_path = n + m + 2
        path_np = np.empty((max_path, 3), dtype=np.int32)
        path = path_np
        path_len = 0

        i = best_row
        j = best_col
        state = best_state
        # record trailing gaps, as needed
        # NOTE: not sure we need to do this, all cases can be inferred.
        if best_row != n:
            # if we are here, we are traveling the last column
            # record the gaps against X if free_X, but not if a jump
            if free_X:
                for i in range(n, best_row, -1):
                    path[path_len, 0] = i
                    path[path_len, 1] = j
                    path[path_len, 2] = 2
                    path_len += 1
        elif best_col != m:
            # if we are here, we are traveling the last row
            # record the gaps against Y
            for j in range(m, best_col, -1):
                path[path_len, 0] = i
                path[path_len, 1] = j
                path[path_len, 2] = 0
                path_len += 1

        # and now begin the main traceback
        i = best_row
        j = best_col
        while i > 0 or j > 0:
            # record current entry
            path[path_len, 0] = i
            path[path_len, 1] = j
            path[path_len, 2] = state
            path_len += 1
            # step to predecessor
            if state == 0:
                # Yg: move left, X gets a gap.
                state = Yg_tr[i, j]
                j -= 1
            elif state == 1:
                # M: move diagonally, maybe jump in rows
                prev_row = M_row[i, j]
                state = M_tr[i, j]
                i = prev_row
                j -= 1
            else:
                # Xg: move up, Y gets a gap, maybe jump in rows
                prev_row = Xg_row[i, j]
                state = Xg_tr[i, j]
                i = prev_row

        # Reverse path in place
        left = 0
        right = path_len - 1
        while left < right:
            tmp0 = path[left, 0]
            tmp1 = path[left, 1]
            tmp2 = path[left, 2]

            path[left, 0] = path[right, 0]
            path[left, 1] = path[right, 1]
            path[left, 2] = path[right, 2]

            path[right, 0] = tmp0
            path[right, 1] = tmp1
            path[right, 2] = tmp2

            left += 1
            right -= 1

        if return_matrices:
            return (
                best_score,
                path_np[:path_len],
                Yg_np,
                M_np,
                Xg_np,
                Yg_tr_np,
                M_tr_np,
                Xg_tr_np,
                M_row_np,
                Xg_row_np,
            )
        else:
            return (
                best_score,
                path_np[:path_len],
            )

    return (
        Yg_np,
        M_np,
        Xg_np,
        Yg_tr_np,
        M_tr_np,
        Xg_tr_np,
        M_row_np,
        Xg_row_np,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def nwflex_dp_core_buffered(
    cnp.ndarray[ITYPE_t, ndim=1] X_codes,
    cnp.ndarray[ITYPE_t, ndim=1] Y_codes,
    cnp.ndarray[DTYPE_t, ndim=2] score_matrix,
    double gap_open,
    double gap_extend,
    cnp.ndarray[ITYPE_t, ndim=1] ep_counts,
    cnp.ndarray[ITYPE_t, ndim=2] ep_starts,
    cnp.ndarray[ITYPE_t, ndim=2] ep_ends,
    bint free_X,
    bint free_Y,
    DPBuffers buffers,
):
    """
    Buffered NW-flex DP core - reuses pre-allocated matrices.

    Same algorithm as nwflex_dp_core but uses buffers.Yg, buffers.M, etc.
    instead of allocating fresh arrays. Only initializes the nrows x ncols
    subregion needed for this alignment.

    Always performs traceback and returns (score, path_array.copy()).
    The copy is critical to prevent aliasing issues.

    Parameters
    ----------
    X_codes, Y_codes, score_matrix, gap_open, gap_extend,
    ep_counts, ep_starts, ep_ends, free_X, free_Y:
        Same as nwflex_dp_core.
    buffers : DPBuffers
        Pre-allocated buffer object. Must have max_nrows >= len(X)+1
        and max_ncols >= len(Y)+1.

    Returns
    -------
    tuple of (score: float, path_array: ndarray[int32, (path_len, 3)])
    """
    cdef int n = X_codes.shape[0]
    cdef int m = Y_codes.shape[0]
    cdef int nrows = n + 1
    cdef int ncols = m + 1

    # Ensure buffers are large enough (will reallocate if needed)
    buffers.ensure_size(nrows, ncols)

    # Typed views onto inputs (for cheap access in loops)
    cdef ITYPE_t[:] Xv = X_codes
    cdef ITYPE_t[:] Yv = Y_codes
    cdef DTYPE_t[:, :] S = score_matrix

    cdef ITYPE_t[:] ep_c = ep_counts
    cdef ITYPE_t[:, :] ep_s = ep_starts
    cdef ITYPE_t[:, :] ep_e = ep_ends

    # "Minus infinity" sentinel for unusable cells
    cdef DTYPE_t NEG_INF = <DTYPE_t>(-1e300)

    # Use pre-allocated arrays from buffers
    cdef cnp.ndarray[DTYPE_t, ndim=2] Yg_np = buffers.Yg
    cdef cnp.ndarray[DTYPE_t, ndim=2]  M_np = buffers.M
    cdef cnp.ndarray[DTYPE_t, ndim=2] Xg_np = buffers.Xg

    cdef cnp.ndarray[ITYPE_t, ndim=2] Yg_tr_np = buffers.Yg_tr
    cdef cnp.ndarray[ITYPE_t, ndim=2]  M_tr_np = buffers.M_tr
    cdef cnp.ndarray[ITYPE_t, ndim=2] Xg_tr_np = buffers.Xg_tr

    cdef cnp.ndarray[ITYPE_t, ndim=2]  M_row_np = buffers.M_row
    cdef cnp.ndarray[ITYPE_t, ndim=2] Xg_row_np = buffers.Xg_row

    cdef cnp.ndarray[ITYPE_t, ndim=2] path_np = buffers.path
    cdef ITYPE_t[:] row_default = buffers.row_default_view

    # Memoryview aliases for fast C-style indexing inside loops
    cdef DTYPE_t[:, :] Yg = Yg_np
    cdef DTYPE_t[:, :]  M = M_np
    cdef DTYPE_t[:, :] Xg = Xg_np

    cdef ITYPE_t[:, :] Yg_tr = Yg_tr_np
    cdef ITYPE_t[:, :]  M_tr =  M_tr_np
    cdef ITYPE_t[:, :] Xg_tr = Xg_tr_np

    cdef ITYPE_t[:, :] M_row  = M_row_np
    cdef ITYPE_t[:, :] Xg_row = Xg_row_np

    cdef ITYPE_t[:, :] path = path_np

    # Loop indices and temporaries
    cdef int i, j, k, r
    cdef DTYPE_t gs = gap_open
    cdef DTYPE_t ge = gap_extend
    cdef DTYPE_t c0, c1, c2, best, score, candM, candX
    cdef ITYPE_t st
    cdef ITYPE_t xi, yj

    # Traceback variables
    cdef DTYPE_t best_score
    cdef DTYPE_t best_col_score
    cdef ITYPE_t best_row
    cdef ITYPE_t best_col
    cdef ITYPE_t best_state
    cdef ITYPE_t prev_row
    cdef ITYPE_t state
    cdef int path_len
    cdef int left, right
    cdef ITYPE_t tmp0, tmp1, tmp2
    cdef Py_ssize_t ep_len

    # For memset
    cdef int trace_row_bytes = ncols * sizeof(ITYPE_t)

    # ------------------------------------------------------------------
    # Initialization (SUBREGION ONLY: rows 0..nrows-1, cols 0..ncols-1)
    # ------------------------------------------------------------------
    # Use memset for trace matrices: 0xFF = -1 for all bytes of int32
    for i in range(nrows):
        memset(&Yg_tr[i, 0], 0xFF, trace_row_bytes)
        memset(&M_tr[i, 0], 0xFF, trace_row_bytes)
        memset(&Xg_tr[i, 0], 0xFF, trace_row_bytes)

    # Initialize score matrices to NEG_INF and row predecessors
    for i in range(nrows):
        for j in range(ncols):
            Yg[i, j] = NEG_INF
            M[i, j]  = NEG_INF
            Xg[i, j] = NEG_INF
            M_row[i, j] = row_default[i]
            Xg_row[i, j] = row_default[i]

    # Starting cell: only M(0,0) is 0
    M[0, 0] = 0.0

    # First row: gaps in X (horizontal, Yg state)
    if free_Y:
        for j in range(1, ncols):
            Yg[0, j] = 0.0
            Yg_tr[0, j] = 0
    else:
        for j in range(1, ncols):
            Yg[0, j] = gs + (j - 1) * ge
            Yg_tr[0, j] = 0

    # First column: gaps in Y (vertical, Xg state) with EP refinement
    if free_X:
        for i in range(1, nrows):
            Xg[i, 0] = 0.0
            Xg_tr[i, 0] = 2
    else:
        for i in range(1, nrows):
            c1 = M[i - 1, 0] + gs
            c2 = Xg[i - 1, 0] + ge
            Xg_tr[i, 0] = 2
            if c1 >= c2:
                Xg[i, 0] = c1
            else:
                Xg[i, 0] = c2

            if ep_c[i] > 0:
                for k in range(ep_c[i]):
                    r = ep_s[i, k]
                    while r <= ep_e[i, k]:
                        c1 = M[r, 0] + gs
                        c2 = Xg[r, 0] + ge
                        if c1 >= c2:
                            candX = c1
                        else:
                            candX = c2
                        if candX > Xg[i, 0]:
                            Xg[i, 0] = candX
                            Xg_row[i, 0] = r
                        r += 1

    # ------------------------------------------------------------------
    # Main DP: rows 1..n, columns 1..m
    # ------------------------------------------------------------------
    for i in range(1, nrows):
        xi = Xv[i - 1]
        for j in range(1, ncols):
            yj = Yv[j - 1]
            score = S[xi, yj]

            # Yg(i,j): gap in X (horizontal)
            c0 = Yg[i, j - 1] + ge
            c1 = M [i, j - 1] + gs
            c2 = Xg[i, j - 1] + gs

            best = c0
            st = 0
            if c1 > best:
                best = c1
                st = 1
            if c2 > best:
                best = c2
                st = 2

            Yg[i, j] = best
            Yg_tr[i, j] = st

            # M(i,j): match/mismatch
            c0 = Yg[i - 1, j - 1]
            c1 = M [i - 1, j - 1]
            c2 = Xg[i - 1, j - 1]

            best = c0
            st = 0
            if c1 > best:
                best = c1
                st = 1
            if c2 > best:
                best = c2
                st = 2

            M[i, j] = best + score
            M_tr[i, j] = st

            # Xg(i,j): gap in Y (vertical)
            c0 = Yg[i - 1, j] + gs
            c1 = M [i - 1, j] + gs
            c2 = Xg[i - 1, j] + ge

            best = c0
            st = 0
            if c1 > best:
                best = c1
                st = 1
            if c2 > best:
                best = c2
                st = 2

            Xg[i, j] = best
            Xg_tr[i, j] = st

            # EP refinement: rows r ∈ E(i)
            if ep_c[i] > 0:
                for k in range(ep_c[i]):
                    r = ep_s[i, k]
                    while r <= ep_e[i, k]:
                        # M(i,j) candidate from row r, col j-1
                        c0 = Yg[r, j - 1]
                        c1 = M [r, j - 1]
                        c2 = Xg[r, j - 1]
                        best = c0
                        st = 0
                        if c1 > best:
                            best = c1
                            st = 1
                        if c2 > best:
                            best = c2
                            st = 2
                        candM = best + score
                        if candM > M[i, j]:
                            M[i, j] = candM
                            M_tr[i, j] = st
                            M_row[i, j] = r

                        # Xg(i,j) candidate from row r, col j
                        c0 = Yg[r, j] + gs
                        c1 = M [r, j] + gs
                        c2 = Xg[r, j] + ge
                        best = c0
                        st = 0
                        if c1 > best:
                            best = c1
                            st = 1
                        if c2 > best:
                            best = c2
                            st = 2
                        candX = best
                        if candX > Xg[i, j]:
                            Xg[i, j] = candX
                            Xg_tr[i, j] = st
                            Xg_row[i, j] = r

                        r += 1

    # ------------------------------------------------------------------
    # Traceback (always performed in buffered version)
    # ------------------------------------------------------------------
    best_score = NEG_INF
    best_row = n
    best_col = m
    best_state = 1
    best_col_score = best_score

    if free_Y:
        for j in range(ncols):
            c0 = Yg[n, j]
            if c0 > best_score:
                best_score = c0
                best_row = n
                best_col = j
                best_state = 0
            c1 = M[n, j]
            if c1 > best_score:
                best_score = c1
                best_row = n
                best_col = j
                best_state = 1
            c2 = Xg[n, j]
            if c2 > best_score:
                best_score = c2
                best_row = n
                best_col = j
                best_state = 2
        best_col_score = best_score

    if free_X:
        for i in range(nrows):
            c0 = Yg[i, m]
            if c0 > best_score:
                best_score = c0
                best_row = i
                best_col = m
                best_state = 0
            c1 = M[i, m]
            if c1 > best_score:
                best_score = c1
                best_row = i
                best_col = m
                best_state = 1
            c2 = Xg[i, m]
            if c2 > best_score:
                best_score = c2
                best_row = i
                best_col = m
                best_state = 2
    else:
        c0 = Yg[n, m]
        if c0 > best_score:
            best_score = c0
            best_row = n
            best_col = m
            best_state = 0
        c1 = M[n, m]
        if c1 > best_score:
            best_score = c1
            best_row = n
            best_col = m
            best_state = 1
        c2 = Xg[n, m]
        if c2 > best_score:
            best_score = c2
            best_row = n
            best_col = m
            best_state = 2
        # Check extra predecessors for row n+1
        ep_len = ep_c.shape[0]
        if ep_len > nrows and ep_c[nrows] > 0:
            for k in range(ep_c[nrows]):
                r = ep_s[nrows, k]
                while r <= ep_e[nrows, k]:
                    c0 = Yg[r, m]
                    if c0 > best_score:
                        best_score = c0
                        best_row = r
                        best_col = m
                        best_state = 0
                    c1 = M[r, m]
                    if c1 > best_score:
                        best_score = c1
                        best_row = r
                        best_col = m
                        best_state = 1
                    c2 = Xg[r, m]
                    if c2 > best_score:
                        best_score = c2
                        best_row = r
                        best_col = m
                        best_state = 2
                    r += 1

    if best_score == best_col_score:
        best_row = n
    else:
        best_col = m

    # Traceback path
    path_len = 0

    i = best_row
    j = best_col
    state = best_state
    # Record trailing gaps
    if best_row != n:
        if free_X:
            for i in range(n, best_row, -1):
                path[path_len, 0] = i
                path[path_len, 1] = j
                path[path_len, 2] = 2
                path_len += 1
    elif best_col != m:
        for j in range(m, best_col, -1):
            path[path_len, 0] = i
            path[path_len, 1] = j
            path[path_len, 2] = 0
            path_len += 1

    # Main traceback
    i = best_row
    j = best_col
    while i > 0 or j > 0:
        path[path_len, 0] = i
        path[path_len, 1] = j
        path[path_len, 2] = state
        path_len += 1
        if state == 0:
            state = Yg_tr[i, j]
            j -= 1
        elif state == 1:
            prev_row = M_row[i, j]
            state = M_tr[i, j]
            i = prev_row
            j -= 1
        else:
            prev_row = Xg_row[i, j]
            state = Xg_tr[i, j]
            i = prev_row

    # Reverse path in place
    left = 0
    right = path_len - 1
    while left < right:
        tmp0 = path[left, 0]
        tmp1 = path[left, 1]
        tmp2 = path[left, 2]

        path[left, 0] = path[right, 0]
        path[left, 1] = path[right, 1]
        path[left, 2] = path[right, 2]

        path[right, 0] = tmp0
        path[right, 1] = tmp1
        path[right, 2] = tmp2

        left += 1
        right -= 1

    # Return score and a COPY of the path array (critical for aliasing safety)
    return (best_score, path_np[:path_len].copy())


@cython.boundscheck(False)
@cython.wraparound(False)
def nwflex_dp_core_buffered_cigar(
    cnp.ndarray[ITYPE_t, ndim=1] X_codes,
    cnp.ndarray[ITYPE_t, ndim=1] Y_codes,
    cnp.ndarray[DTYPE_t, ndim=2] score_matrix,
    double gap_open,
    double gap_extend,
    cnp.ndarray[ITYPE_t, ndim=1] ep_counts,
    cnp.ndarray[ITYPE_t, ndim=2] ep_starts,
    cnp.ndarray[ITYPE_t, ndim=2] ep_ends,
    bint free_X,
    bint free_Y,
    DPBuffers buffers,
):
    """
    Buffered NW-flex DP core that returns CIGAR directly.

    Same algorithm as nwflex_dp_core_buffered but builds CIGAR string
    during traceback instead of returning a path array.

    Returns
    -------
    tuple of (score: float, start_pos: int, cigar: str)
    """
    cdef int n = X_codes.shape[0]
    cdef int m = Y_codes.shape[0]
    cdef int nrows = n + 1
    cdef int ncols = m + 1

    # Ensure buffers are large enough
    buffers.ensure_size(nrows, ncols)

    # Typed views onto inputs
    cdef ITYPE_t[:] Xv = X_codes
    cdef ITYPE_t[:] Yv = Y_codes
    cdef DTYPE_t[:, :] S = score_matrix
    cdef ITYPE_t[:] ep_c = ep_counts
    cdef ITYPE_t[:, :] ep_s = ep_starts
    cdef ITYPE_t[:, :] ep_e = ep_ends

    # Typed views onto buffer arrays
    cdef cnp.ndarray[DTYPE_t, ndim=2] Yg_np = buffers.Yg
    cdef cnp.ndarray[DTYPE_t, ndim=2] M_np = buffers.M
    cdef cnp.ndarray[DTYPE_t, ndim=2] Xg_np = buffers.Xg
    cdef cnp.ndarray[ITYPE_t, ndim=2] Yg_tr_np = buffers.Yg_tr
    cdef cnp.ndarray[ITYPE_t, ndim=2] M_tr_np = buffers.M_tr
    cdef cnp.ndarray[ITYPE_t, ndim=2] Xg_tr_np = buffers.Xg_tr
    cdef cnp.ndarray[ITYPE_t, ndim=2] M_row_np = buffers.M_row
    cdef cnp.ndarray[ITYPE_t, ndim=2] Xg_row_np = buffers.Xg_row
    cdef cnp.ndarray[ITYPE_t, ndim=2] path_np = buffers.path
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] cigar_ops_np = buffers.cigar_ops

    cdef DTYPE_t[:, :] Yg = Yg_np
    cdef DTYPE_t[:, :] M = M_np
    cdef DTYPE_t[:, :] Xg = Xg_np
    cdef ITYPE_t[:, :] Yg_tr = Yg_tr_np
    cdef ITYPE_t[:, :] M_tr = M_tr_np
    cdef ITYPE_t[:, :] Xg_tr = Xg_tr_np
    cdef ITYPE_t[:, :] M_row = M_row_np
    cdef ITYPE_t[:, :] Xg_row = Xg_row_np
    cdef ITYPE_t[:, :] path = path_np
    cdef cnp.uint8_t[:] cigar_ops = cigar_ops_np
    cdef ITYPE_t[:] row_def = buffers.row_default_view

    cdef double gs = gap_open
    cdef double ge = gap_extend
    cdef double NEG_INF = -1e100

    cdef int i, j, k, r, xi, yj, st, prev_row, ep_len
    cdef double score, c0, c1, c2, candX, best

    # Initialize row 0
    Yg[0, 0] = NEG_INF
    M[0, 0] = 0.0
    Xg[0, 0] = NEG_INF
    Yg_tr[0, 0] = 1
    M_tr[0, 0] = 1
    Xg_tr[0, 0] = 1
    M_row[0, 0] = -1
    Xg_row[0, 0] = -1

    for j in range(1, ncols):
        if free_Y:
            Yg[0, j] = 0.0
            M[0, j] = 0.0
            Xg[0, j] = NEG_INF
            Yg_tr[0, j] = 1
            M_tr[0, j] = 1
            Xg_tr[0, j] = 1
        else:
            c0 = Yg[0, j - 1] + ge
            c1 = M[0, j - 1] + gs
            Yg[0, j] = c0 if c0 >= c1 else c1
            Yg_tr[0, j] = 0 if c0 >= c1 else 1
            M[0, j] = NEG_INF
            Xg[0, j] = NEG_INF
            M_tr[0, j] = 1
            Xg_tr[0, j] = 1
        M_row[0, j] = -1
        Xg_row[0, j] = -1

    # Initialize column 0
    for i in range(1, nrows):
        Yg[i, 0] = NEG_INF
        M[i, 0] = NEG_INF
        Yg_tr[i, 0] = 1
        M_tr[i, 0] = 1
        M_row[i, 0] = row_def[i]
        Xg_row[i, 0] = row_def[i]

        if free_X:
            Xg[i, 0] = 0.0
            Xg_tr[i, 0] = 2
        else:
            c1 = M[i - 1, 0] + gs
            c2 = Xg[i - 1, 0] + ge
            Xg_tr[i, 0] = 2
            if c1 >= c2:
                Xg[i, 0] = c1
            else:
                Xg[i, 0] = c2

            if ep_c[i] > 0:
                for k in range(ep_c[i]):
                    r = ep_s[i, k]
                    while r <= ep_e[i, k]:
                        c1 = M[r, 0] + gs
                        c2 = Xg[r, 0] + ge
                        if c1 >= c2:
                            candX = c1
                        else:
                            candX = c2
                        if candX > Xg[i, 0]:
                            Xg[i, 0] = candX
                            Xg_row[i, 0] = r
                        r += 1

    # Main DP: rows 1..n, columns 1..m
    for i in range(1, nrows):
        xi = Xv[i - 1]
        for j in range(1, ncols):
            yj = Yv[j - 1]
            score = S[xi, yj]

            # Yg(i,j): gap in X (horizontal)
            c0 = Yg[i, j - 1] + ge
            c1 = M [i, j - 1] + gs
            c2 = Xg[i, j - 1] + gs

            best = c0
            st = 0
            if c1 > best:
                best = c1
                st = 1
            if c2 > best:
                best = c2
                st = 2

            Yg[i, j] = best
            Yg_tr[i, j] = st

            # M(i,j): match/mismatch
            c0 = Yg[i - 1, j - 1]
            c1 = M [i - 1, j - 1]
            c2 = Xg[i - 1, j - 1]

            best = c0
            st = 0
            if c1 > best:
                best = c1
                st = 1
            if c2 > best:
                best = c2
                st = 2

            M[i, j] = best + score
            M_tr[i, j] = st

            # Xg(i,j): gap in Y (vertical)
            c0 = Yg[i - 1, j] + gs
            c1 = M [i - 1, j] + gs
            c2 = Xg[i - 1, j] + ge

            best = c0
            st = 0
            if c1 > best:
                best = c1
                st = 1
            if c2 > best:
                best = c2
                st = 2

            Xg[i, j] = best
            Xg_tr[i, j] = st
            M_row[i, j] = row_def[i]
            Xg_row[i, j] = row_def[i]

            # Extra predecessors
            if ep_c[i] > 0:
                for k in range(ep_c[i]):
                    r = ep_s[i, k]
                    while r <= ep_e[i, k]:
                        # M(i,j) candidate from row r, col j-1
                        c0 = Yg[r, j - 1]
                        c1 = M [r, j - 1]
                        c2 = Xg[r, j - 1]
                        best = c0
                        st = 0
                        if c1 > best:
                            best = c1
                            st = 1
                        if c2 > best:
                            best = c2
                            st = 2
                        candM = best + score
                        if candM > M[i, j]:
                            M[i, j] = candM
                            M_tr[i, j] = st
                            M_row[i, j] = r

                        # Xg(i,j) candidate from row r, col j
                        c0 = Yg[r, j] + gs
                        c1 = M [r, j] + gs
                        c2 = Xg[r, j] + ge
                        best = c0
                        st = 0
                        if c1 > best:
                            best = c1
                            st = 1
                        if c2 > best:
                            best = c2
                            st = 2
                        candX = best
                        if candX > Xg[i, j]:
                            Xg[i, j] = candX
                            Xg_tr[i, j] = st
                            Xg_row[i, j] = r
                        r += 1

    # Find best ending cell
    cdef double best_score = NEG_INF
    cdef int best_row = n
    cdef int best_col = m
    cdef int best_state = 1
    cdef double best_col_score

    if free_Y:
        for j in range(ncols):
            c0 = Yg[n, j]
            if c0 > best_score:
                best_score = c0
                best_row = n
                best_col = j
                best_state = 0
            c1 = M[n, j]
            if c1 > best_score:
                best_score = c1
                best_row = n
                best_col = j
                best_state = 1
            c2 = Xg[n, j]
            if c2 > best_score:
                best_score = c2
                best_row = n
                best_col = j
                best_state = 2
        best_col_score = best_score

    if free_X:
        for i in range(nrows):
            c0 = Yg[i, m]
            if c0 > best_score:
                best_score = c0
                best_row = i
                best_col = m
                best_state = 0
            c1 = M[i, m]
            if c1 > best_score:
                best_score = c1
                best_row = i
                best_col = m
                best_state = 1
            c2 = Xg[i, m]
            if c2 > best_score:
                best_score = c2
                best_row = i
                best_col = m
                best_state = 2
    else:
        c0 = Yg[n, m]
        if c0 > best_score:
            best_score = c0
            best_row = n
            best_col = m
            best_state = 0
        c1 = M[n, m]
        if c1 > best_score:
            best_score = c1
            best_row = n
            best_col = m
            best_state = 1
        c2 = Xg[n, m]
        if c2 > best_score:
            best_score = c2
            best_row = n
            best_col = m
            best_state = 2
        # Check extra predecessors for row n+1
        ep_len = ep_c.shape[0]
        if ep_len > nrows and ep_c[nrows] > 0:
            for k in range(ep_c[nrows]):
                r = ep_s[nrows, k]
                while r <= ep_e[nrows, k]:
                    c0 = Yg[r, m]
                    if c0 > best_score:
                        best_score = c0
                        best_row = r
                        best_col = m
                        best_state = 0
                    c1 = M[r, m]
                    if c1 > best_score:
                        best_score = c1
                        best_row = r
                        best_col = m
                        best_state = 1
                    c2 = Xg[r, m]
                    if c2 > best_score:
                        best_score = c2
                        best_row = r
                        best_col = m
                        best_state = 2
                    r += 1

    if free_Y and best_score == best_col_score:
        best_row = n
    elif free_Y:
        best_col = m

    # Build path in reverse order (for CIGAR generation)
    cdef int path_len = 0
    i = best_row
    j = best_col
    cdef int state = best_state

    # Trailing gaps
    if best_row != n:
        if free_X:
            for i in range(n, best_row, -1):
                path[path_len, 0] = i
                path[path_len, 1] = j
                path[path_len, 2] = 2
                path_len += 1
    elif best_col != m:
        for j in range(m, best_col, -1):
            path[path_len, 0] = i
            path[path_len, 1] = j
            path[path_len, 2] = 0
            path_len += 1

    # Main traceback
    i = best_row
    j = best_col
    while i > 0 or j > 0:
        path[path_len, 0] = i
        path[path_len, 1] = j
        path[path_len, 2] = state
        path_len += 1
        if state == 0:
            state = Yg_tr[i, j]
            j -= 1
        elif state == 1:
            prev_row = M_row[i, j]
            state = M_tr[i, j]
            i = prev_row
            j -= 1
        else:
            prev_row = Xg_row[i, j]
            state = Xg_tr[i, j]
            i = prev_row

    # Reverse path in place
    cdef int left = 0
    cdef int right = path_len - 1
    cdef ITYPE_t tmp0, tmp1, tmp2
    while left < right:
        tmp0 = path[left, 0]
        tmp1 = path[left, 1]
        tmp2 = path[left, 2]
        path[left, 0] = path[right, 0]
        path[left, 1] = path[right, 1]
        path[left, 2] = path[right, 2]
        path[right, 0] = tmp0
        path[right, 1] = tmp1
        path[right, 2] = tmp2
        left += 1
        right -= 1

    # Now path is in forward order. Convert to CIGAR.
    # CIGAR ops: M=77, I=73, D=68, S=83, N=78 (ASCII codes)
    cdef int cigar_len = 0
    cdef int start_pos = -1
    cdef int pi = 0
    cdef int ci, cj, cstate
    cdef int skip

    for k in range(path_len):
        ci = path[k, 0]
        cj = path[k, 1]
        cstate = path[k, 2]

        # Skip entries at or before column 0 (leading reference gaps)
        if cj <= 0:
            pi = ci
            continue
        if ci == 0:
            cigar_ops[cigar_len] = 83  # 'S'
            cigar_len += 1
            continue
        if cj == m and cstate == 2:
            break
        if start_pos == -1:
            start_pos = ci

        # Handle row jumps (splicing)
        skip = ci - pi - 1
        if skip > 0:
            for r in range(skip):
                cigar_ops[cigar_len] = 78  # 'N'
                cigar_len += 1

        if cstate == 0:
            if ci == n:
                cigar_ops[cigar_len] = 83  # 'S' for trailing soft clip
            else:
                cigar_ops[cigar_len] = 73  # 'I'
            cigar_len += 1
        elif cstate == 1:
            cigar_ops[cigar_len] = 77  # 'M'
            cigar_len += 1
        else:
            cigar_ops[cigar_len] = 68  # 'D'
            cigar_len += 1

        pi = ci

    # Terminal contraction: if the path ended before row n,
    # the remaining rows were contracted via EP[n+1]
    if not free_X and pi < n:
        for r in range(n - pi):
            cigar_ops[cigar_len] = 78  # 'N'
            cigar_len += 1

    # Run-length encode CIGAR ops to string
    if cigar_len == 0:
        return (best_score, start_pos if start_pos >= 0 else -1, "")

    cdef list parts = []
    cdef int count = 1
    cdef cnp.uint8_t prev_op = cigar_ops[0]
    cdef cnp.uint8_t curr_op

    for k in range(1, cigar_len):
        curr_op = cigar_ops[k]
        if curr_op == prev_op:
            count += 1
        else:
            parts.append(f"{count}{chr(prev_op)}")
            prev_op = curr_op
            count = 1
    parts.append(f"{count}{chr(prev_op)}")

    return (best_score, start_pos, "".join(parts))
