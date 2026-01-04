"""Tests for the Cython traceback fast path."""

import numpy as np

from nwflex.dp_core import FlexInput
from nwflex.ep_patterns import build_EP_single_block, build_EP_standard
from nwflex.fast import run_flex_dp_fast, run_flex_dp_fast_path
from nwflex.aligners import RefAligner


def test_fast_path_matches_fast(scoring_params, rng, random_dna_factory):
    """Fast-path outputs match the existing Cython+Python traceback path."""
    nX = 20
    nY = 18
    X = random_dna_factory(nX, rng)
    Y = random_dna_factory(nY, rng)

    s, e = 5, 12
    EP = build_EP_single_block(nX, s, e)

    cfg = FlexInput(
        X=X,
        Y=Y,
        extra_predecessors=EP,
        **scoring_params,
    )

    ref_result = run_flex_dp_fast(cfg, return_data=False)
    fast_result = run_flex_dp_fast_path(cfg, return_data=False)

    assert ref_result.score == fast_result.score
    assert ref_result.X_aln == fast_result.X_aln
    assert ref_result.Y_aln == fast_result.Y_aln
    assert ref_result.path == fast_result.path
    assert len(ref_result.jumps) == len(fast_result.jumps)
    for jp, jc in zip(ref_result.jumps, fast_result.jumps):
        assert jp.from_row == jc.from_row
        assert jp.to_row == jc.to_row
        assert jp.col == jc.col
        assert jp.state == jc.state


def test_fast_path_matches_fast_with_data(scoring_params):
    """DP matrices are preserved when return_data=True."""
    X = "AACCCGTGT"
    Y = "AGTGT"
    s, e = 2, 5
    EP = build_EP_single_block(len(X), s, e)

    cfg = FlexInput(
        X=X,
        Y=Y,
        extra_predecessors=EP,
        **scoring_params,
    )

    ref_result = run_flex_dp_fast(cfg, return_data=True)
    fast_result = run_flex_dp_fast_path(cfg, return_data=True)

    assert np.allclose(ref_result.data.Yg, fast_result.data.Yg)
    assert np.allclose(ref_result.data.M, fast_result.data.M)
    assert np.allclose(ref_result.data.Xg, fast_result.data.Xg)
    assert np.array_equal(ref_result.data.Yg_trace, fast_result.data.Yg_trace)
    assert np.array_equal(ref_result.data.M_trace, fast_result.data.M_trace)
    assert np.array_equal(ref_result.data.Xg_trace, fast_result.data.Xg_trace)
    assert np.array_equal(ref_result.data.M_row, fast_result.data.M_row)
    assert np.array_equal(ref_result.data.Xg_row, fast_result.data.Xg_row)


def test_refaligner_fast_traceback_flag(scoring_params):
    """RefAligner fast_traceback toggles the new Cython path."""
    X = "ACGTACGTACGT"
    Y = "ACGTACGT"
    EP = build_EP_standard(len(X))

    aligner_fast = RefAligner(
        ref=X,
        extra_predecessors=EP,
        fast_mode=True,
        fast_traceback=True,
        **scoring_params,
    )
    aligner_slow = RefAligner(
        ref=X,
        extra_predecessors=EP,
        fast_mode=True,
        fast_traceback=False,
        **scoring_params,
    )

    res_fast = aligner_fast.align(Y)
    res_slow = aligner_slow.align(Y)

    assert res_fast.score == res_slow.score
    assert res_fast.X_aln == res_slow.X_aln
    assert res_fast.Y_aln == res_slow.Y_aln
    assert res_fast.path == res_slow.path
    assert len(res_fast.jumps) == len(res_slow.jumps)
