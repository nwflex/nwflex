"""
fast.py — Cython-backed NW-flex DP wrapper

This module provides a drop-in replacement for the Python DP core:
    run_flex_dp_fast(config, return_data=False)

It:
  * encodes X and Y as integer codes using config.alphabet_to_index,
  * converts extra_predecessors[i] into interval arrays suitable for
    the Cython kernel,
  * calls nwflex_dp_core (Cython),
  * wraps the resulting DP arrays into FlexData,
  * reuses traceback_alignment to recover (X_aln, Y_aln, path, jumps),
  * returns an AlignmentResult with the same shape as the Python path.
"""

import numpy as np

from .dp_core import FlexInput, FlexData, traceback_alignment, AlignmentResult
from .ep_intervals import ep_to_intervals

try:
    from ._cython.nwflex_dp import nwflex_dp_core
    CYTHON_AVAILABLE = True
except ImportError:    
    nwflex_dp_core = None  # Placeholder to avoid NameError
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


def run_flex_dp_fast(
    config: FlexInput,
    return_data: bool = False,
) -> AlignmentResult:
    """
    Fast NW-flex DP using the Cython core.

    Parameters
    ----------
    config : FlexInput
        Sequences, scoring scheme, and extra predecessor sets E(i).
    return_data : bool, default False
        If True, attach the full FlexData object in the result.

    Returns
    -------
    AlignmentResult
        Same structure as the Python-based aligners return:
          - score
          - X_aln, Y_aln
          - jumps (list[RowJump])
          - data (FlexData or None)
          - path (list[(i,j,state)])
    """
    ## check cython availability
    if not CYTHON_AVAILABLE:
        _cython_not_available_error()
    # Encode sequences as integer codes
    X_codes = _encode_sequence(config.X, config.alphabet_to_index)
    Y_codes = _encode_sequence(config.Y, config.alphabet_to_index)

    # Convert EP list-of-lists -> interval arrays
    ep_counts, ep_starts, ep_ends = ep_to_intervals(config.extra_predecessors)

    # Call the Cython DP core
    (
        Yg,
        M,
        Xg,
        Yg_tr,
        M_tr,
        Xg_tr,
        M_row,
        Xg_row,
    ) = nwflex_dp_core(
        X_codes,
        Y_codes,
        config.score_matrix,
        config.gap_open,
        config.gap_extend,
        ep_counts,
        ep_starts,
        ep_ends,
        config.free_X,
        config.free_Y,
    )

    # Wrap into FlexData
    data = FlexData(
        Yg=Yg,
        M=M,
        Xg=Xg,
        Yg_trace=Yg_tr,
        M_trace=M_tr,
        Xg_trace=Xg_tr,
        M_row=M_row,
        Xg_row=Xg_row,
    )

    # Use the existing Python traceback logic
    X_aln, Y_aln, score, path, jumps = traceback_alignment(config, data)    

    return AlignmentResult(
        score=score,
        X_aln=X_aln,
        Y_aln=Y_aln,
        path=path,
        jumps=jumps,
        data=data if return_data else None,
        
    )
