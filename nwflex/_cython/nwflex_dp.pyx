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

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int32_t   ITYPE_t

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
    cdef cnp.ndarray[DTYPE_t, ndim=2] Yg_np = np.empty((nrows, ncols), dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=2]  M_np = np.empty((nrows, ncols), dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=2] Xg_np = np.empty((nrows, ncols), dtype=np.float64)

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
