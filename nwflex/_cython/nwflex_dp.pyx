# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
nwflex_dp2.pyx — Unified Cython core for NW-flex DP

Single kernel: nwflex_dp_core does DP fill + traceback + CIGAR in one call.
Replaces the three separate kernels in nwflex_dp.pyx.

Called from a thin Python wrapper that:
  * encodes X and Y as integer codes (alphabet indices),
  * converts E(i) from a list-of-lists into interval arrays
    (ep_counts, ep_starts, ep_ends).

Focused on numeric work only: no string handling beyond CIGAR construction,
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

    # Score matrices (float32)
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

        # CIGAR ops buffer: one byte per op, max length is 2 * (max_nrows + max_ncols)
        # Extra space for terminal N ops
        self.cigar_ops = np.empty(2 * (max_nrows + max_ncols), dtype=np.uint8)

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
    DPBuffers buffers = None,
    bint return_matrices = False,
):
    """
    Unified NW-flex DP core: fill + traceback + CIGAR.

    Parameters
    ----------
    X_codes, Y_codes : int32[1D]
        Encoded sequences (alphabet indices).
    score_matrix : float32[2D]
        Substitution matrix.
    gap_open, gap_extend : float
        Gap penalties.
    ep_counts : int32[1D]
        ep_counts[i] = number of EP intervals for row i.
    ep_starts, ep_ends : int32[2D]
        Interval bounds for each row's extra predecessors.
    free_X, free_Y : bool
        Semi-global flags.
    buffers : DPBuffers or None
        Pre-allocated buffers for reuse. If None, allocates internally.
    return_matrices : bool
        If True, include DP matrices in return value.

    Returns
    -------
    If return_matrices is False:
        (score, start_pos, cigar, path_array)
    If return_matrices is True:
        (score, start_pos, cigar, path_array,
         Yg, M, Xg, Yg_tr, M_tr, Xg_tr, M_row, Xg_row)
    """
    cdef int n = X_codes.shape[0]
    cdef int m = Y_codes.shape[0]
    cdef int nrows = n + 1
    cdef int ncols = m + 1

    # Allocate or reuse buffers
    cdef bint owns_buffers = buffers is None
    if owns_buffers:
        buffers = DPBuffers(nrows, ncols)
    else:
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
    cdef double score, c0, c1, c2, candM, candX, best

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    # Row 0
    Yg[0, 0] = NEG_INF
    M[0, 0] = 0.0
    Xg[0, 0] = NEG_INF
    Yg_tr[0, 0] = 0   # go left (safe direction for row 0)
    M_tr[0, 0] = 0
    Xg_tr[0, 0] = 0
    M_row[0, 0] = -1
    Xg_row[0, 0] = -1

    # First row: only Yg is reachable (horizontal gaps)
    # M and Xg are unreachable; all trace pointers point left (state 0 = Yg)
    for j in range(1, ncols):
        M[0, j] = NEG_INF
        Xg[0, j] = NEG_INF
        Yg_tr[0, j] = 0
        M_tr[0, j] = 0
        Xg_tr[0, j] = 0
        M_row[0, j] = -1
        Xg_row[0, j] = -1

        if free_Y:
            Yg[0, j] = 0.0
        else:
            c0 = Yg[0, j - 1] + ge
            c1 = M[0, j - 1] + gs
            Yg[0, j] = c0 if c0 >= c1 else c1
            Yg_tr[0, j] = 0 if c0 >= c1 else 1

    # First column: only Xg is reachable (vertical gaps)
    # M and Yg are unreachable; all trace pointers point up (state 2 = Xg)
    for i in range(1, nrows):
        Yg[i, 0] = NEG_INF
        M[i, 0] = NEG_INF
        Yg_tr[i, 0] = 2
        M_tr[i, 0] = 2
        Xg_tr[i, 0] = 2
        M_row[i, 0] = row_def[i]
        Xg_row[i, 0] = row_def[i]

        if free_X:
            Xg[i, 0] = 0.0
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

    # ------------------------------------------------------------------
    # Main DP fill: rows 1..n, columns 1..m
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

    # ------------------------------------------------------------------
    # Find best ending cell
    # ------------------------------------------------------------------
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
        # Global in X: check (n, m) first, then EP[n+1]
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

    # ------------------------------------------------------------------
    # Traceback
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # CIGAR generation from forward-order path
    # ------------------------------------------------------------------
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
    cdef str cigar_str
    if cigar_len == 0:
        cigar_str = ""
    else:
        parts = []
        count = 1
        prev_op = cigar_ops[0]
        for k in range(1, cigar_len):
            curr_op = cigar_ops[k]
            if curr_op == prev_op:
                count += 1
            else:
                parts.append(f"{count}{chr(prev_op)}")
                prev_op = curr_op
                count = 1
        parts.append(f"{count}{chr(prev_op)}")
        cigar_str = "".join(parts)

    # ------------------------------------------------------------------
    # Return results
    # ------------------------------------------------------------------
    cdef cnp.ndarray path_result = path_np[:path_len].copy()

    if start_pos < 0:
        start_pos = -1

    if return_matrices:
        return (
            best_score,
            start_pos,
            cigar_str,
            path_result,
            Yg_np[:nrows, :ncols].copy() if owns_buffers else Yg_np,
            M_np[:nrows, :ncols].copy() if owns_buffers else M_np,
            Xg_np[:nrows, :ncols].copy() if owns_buffers else Xg_np,
            Yg_tr_np[:nrows, :ncols].copy() if owns_buffers else Yg_tr_np,
            M_tr_np[:nrows, :ncols].copy() if owns_buffers else M_tr_np,
            Xg_tr_np[:nrows, :ncols].copy() if owns_buffers else Xg_tr_np,
            M_row_np[:nrows, :ncols].copy() if owns_buffers else M_row_np,
            Xg_row_np[:nrows, :ncols].copy() if owns_buffers else Xg_row_np,
        )

    return (best_score, start_pos, cigar_str, path_result)
