"""Tests for path helper utilities in nwflex.fast."""

import numpy as np

from nwflex.dp_core import FlexInput, run_flex_dp
from nwflex.ep_patterns import build_EP_single_block
from nwflex.fast import (
    extract_jumps_from_path,
    path_array_to_list,
    reconstruct_aligned_strings,
)


def test_path_helpers_match_traceback(scoring_params, rng, random_dna_factory):
    """Helper outputs match dp_core.traceback_alignment results."""
    nX = 15
    nY = 12
    X = random_dna_factory(nX, rng)
    Y = random_dna_factory(nY, rng)

    # Use e == nX to allow terminal predecessor set
    s, e = 5, nX
    EP = build_EP_single_block(nX, s, e)

    cfg = FlexInput(
        X=X,
        Y=Y,
        extra_predecessors=EP,
        **scoring_params,
    )

    py_result = run_flex_dp(cfg, return_data=False).to_tuple()
    py_score, py_X_aln, py_Y_aln, py_path, py_jumps = py_result[:5]

    path_array = np.asarray(py_path, dtype=np.int32)

    assert path_array_to_list(path_array) == py_path

    X_aln, Y_aln = reconstruct_aligned_strings(X, Y, path_array)
    assert X_aln == py_X_aln
    assert Y_aln == py_Y_aln

    jumps = extract_jumps_from_path(
        path_array,
        ref_len=len(X),
        free_X=cfg.free_X,
    )

    assert len(jumps) == len(py_jumps)
    for jp, jc in zip(py_jumps, jumps):
        assert jp.from_row == jc.from_row
        assert jp.to_row == jc.to_row
        assert jp.col == jc.col
        assert jp.state == jc.state
