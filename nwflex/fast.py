"""
fast.py — Cython-backed NW-flex DP wrapper

Provides a drop-in replacement for the Python DP core:
    run_flex_dp_fast(config, return_data=False)

It:
  * encodes X and Y as integer codes using config.alphabet_to_index,
  * converts extra_predecessors[i] into interval arrays,
  * casts score_matrix to float32,
  * calls the unified Cython nwflex_dp_core,
  * wraps the results into AlignmentResult or (score, start_pos, cigar).
"""

import numpy as np

from .dp_core import FlexInput, FlexData, RowJump, AlignmentResult
from .ep_intervals import ep_to_intervals

try:
    from ._cython.nwflex_dp import nwflex_dp_core, DPBuffers
    CYTHON_AVAILABLE = True
except ImportError:
    nwflex_dp_core = None
    DPBuffers = None
    CYTHON_AVAILABLE = False

def _cython_not_available_error():
    """Raise a helpful error if Cython extension is not available."""
    raise ImportError(
        "The Cython extension 'nwflex._cython.nwflex_dp' is not available.\n"
        "This usually means the extension failed to compile during installation.\n\n"
        "To fix this:\n"
        "  1) Ensure a C compiler is installed (gcc, MSVC, etc.).\n"
        "  2) Reinstall nwflex with pip install -e . --force-reinstall\n\n"
        "Alternatively, use the pure-Python DP core via nwflex.dp_core import run_flex_dp"
    )

def _encode_sequence(seq: str, alphabet_to_index) -> np.ndarray:
    """
    Encode a sequence into int32 indices using alphabet_to_index.
    """
    codes = np.empty(len(seq), dtype=np.int32)
    for i, ch in enumerate(seq):
        codes[i] = alphabet_to_index[ch]
    return codes


def path_array_to_list(path_array: np.ndarray) -> list[tuple[int, int, int]]:
    """Convert a NumPy path array into a list of (i, j, state) tuples."""
    return [tuple(int(x) for x in row) for row in path_array]


def reconstruct_aligned_strings(
    X: str,
    Y: str,
    path_array: np.ndarray,
) -> tuple[str, str]:
    """Reconstruct aligned strings from a path array.

    Args:
        X: Reference sequence.
        Y: Read sequence.
        path_array: Array of (i, j, state) in forward order.

    Returns:
        Tuple of (X_aln, Y_aln) strings.
    """
    aln_X = []
    aln_Y = []
    for i, j, state in path_array:
        if state == 0:
            aln_X.append("-")
            aln_Y.append(Y[j - 1])
        elif state == 1:
            aln_X.append(X[i - 1])
            aln_Y.append(Y[j - 1])
        else:
            aln_X.append(X[i - 1])
            aln_Y.append("-")
    return "".join(aln_X), "".join(aln_Y)


def extract_jumps_from_path(
    path_array: np.ndarray,
    *,
    ref_len: int | None = None,
    free_X: bool = False,
) -> list[RowJump]:
    """Extract row jumps from a path array.

    Args:
        path_array: Array of (i, j, state) in forward order.
        ref_len: Reference length for terminal jump detection.
        free_X: If True, do not add terminal jump.

    Returns:
        List of RowJump objects.
    """
    jumps: list[RowJump] = []
    if path_array.size == 0:
        return jumps

    # Terminal jump (only for global-in-X with terminal predecessors)
    if ref_len is not None and not free_X:
        end_row = int(path_array[-1, 0])
        if end_row != ref_len:
            jumps.append(
                RowJump(
                    from_row=int(ref_len + 1),
                    to_row=int(end_row),
                    col=int(path_array[-1, 1]),
                    state=int(path_array[-1, 2]),
                )
            )

    # Internal jumps: row increases by more than 1 between steps
    prev_i, _, _ = path_array[0]
    for i, j, state in path_array[1:]:
        if i - prev_i > 1:
            jumps.append(
                RowJump(
                    from_row=int(prev_i),
                    to_row=int(i),
                    col=int(j),
                    state=int(state),
                )
            )
        prev_i = i

    return jumps


def run_flex_dp_fast(
    config: FlexInput,
    return_data: bool = False,
    return_cigar: bool = False,
) -> AlignmentResult:
    """
    Fast NW-flex DP using the unified Cython core.

    Parameters
    ----------
    config : FlexInput
        Sequences, scoring scheme, and extra predecessor sets E(i).
    return_data : bool, default False
        If True, attach the full FlexData object in the result.
    return_cigar : bool, default False
        If True, return (score, start_pos, cigar) instead of AlignmentResult.

    Returns
    -------
    AlignmentResult, or tuple (score, start_pos, cigar) if return_cigar=True.
    """
    if not CYTHON_AVAILABLE:
        _cython_not_available_error()

    X_codes = _encode_sequence(config.X, config.alphabet_to_index)
    Y_codes = _encode_sequence(config.Y, config.alphabet_to_index)
    score_matrix = np.ascontiguousarray(config.score_matrix, dtype=np.float32)
    ep_counts, ep_starts, ep_ends = ep_to_intervals(config.extra_predecessors)

    result = nwflex_dp_core(
        X_codes,
        Y_codes,
        score_matrix,
        config.gap_open,
        config.gap_extend,
        ep_counts,
        ep_starts,
        ep_ends,
        config.free_X,
        config.free_Y,
        return_matrices=return_data,
    )

    if return_data:
        score, start_pos, cigar, path_array, Yg, M, Xg, Yg_tr, M_tr, Xg_tr, M_row, Xg_row = result
        data = FlexData(
            Yg=Yg, M=M, Xg=Xg,
            Yg_trace=Yg_tr, M_trace=M_tr, Xg_trace=Xg_tr,
            M_row=M_row, Xg_row=Xg_row,
        )
    else:
        score, start_pos, cigar, path_array = result
        data = None

    if return_cigar:
        return score, start_pos, cigar

    X_aln, Y_aln = reconstruct_aligned_strings(config.X, config.Y, path_array)
    jumps = extract_jumps_from_path(
        path_array,
        ref_len=len(config.X),
        free_X=config.free_X,
    )
    path = path_array_to_list(path_array)

    return AlignmentResult(
        score=score,
        X_aln=X_aln,
        Y_aln=Y_aln,
        path=path,
        jumps=jumps,
        data=data,
    )
