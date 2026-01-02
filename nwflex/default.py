"""
default.py — Default parameters for NW-flex

Provides DNA alphabet and standard +5/-5/(-20,-1) scoring scheme
that is used throughout examples, notebooks and tests.
"""

import numpy as np

# DNA alphabet
BASES = np.array(["A", "C", "G", "T"])
ALPHABET_TO_INDEX = {b: i for i, b in enumerate(BASES)}

# Substitution matrix: +5 for match, -5 for mismatch
SCORE_MATRIX = np.full((4, 4), -5.0, dtype=float)
np.fill_diagonal(SCORE_MATRIX, 5.0)

## Affine gap penalties
GAP_OPEN = -20.0
GAP_EXTEND = -1.0

def align_params(*, semiglobal: bool = False) -> dict:
    """
    Bundle default scoring parameters into a dict for easy unpacking.

    Parameters:
        semiglobal (bool): If True, use semiglobal alignment settings.
    
    Usage:
        result = align_standard(X, Y, **align_params(semiglobal=True))"""
    return {
        "score_matrix": SCORE_MATRIX,
        "gap_open": GAP_OPEN,
        "gap_extend": GAP_EXTEND,
        "alphabet_to_index": ALPHABET_TO_INDEX,
        "free_X": semiglobal,
        "free_Y": semiglobal,
    }

def get_default_scoring():
    """
    Convenience helper returning the default scoring components:

        score_matrix, gap_open, gap_extend, alphabet_to_index
    """
    return (
        SCORE_MATRIX,
        GAP_OPEN,
        GAP_EXTEND,
        ALPHABET_TO_INDEX,
    )
