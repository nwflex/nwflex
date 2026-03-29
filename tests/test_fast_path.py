"""Tests for the unified Cython DP core and RefAligner."""

import numpy as np

from nwflex.dp_core import FlexInput
from nwflex.ep_patterns import build_EP_single_block, build_EP_standard
from nwflex.fast import run_flex_dp_fast
from nwflex.aligners import RefAligner, alignment_to_cigar


def test_fast_cigar_matches_alignment(scoring_params, rng, random_dna_factory):
    """CIGAR from return_cigar matches alignment_to_cigar on the full result."""
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

    full_result = run_flex_dp_fast(cfg)
    ref_start, ref_cigar = alignment_to_cigar(
        full_result.path, lenX=nX, lenY=nY,
    )

    score, start_pos, cigar = run_flex_dp_fast(cfg, return_cigar=True)

    assert full_result.score == score
    assert ref_start == start_pos
    assert ref_cigar == cigar


def test_fast_return_data(scoring_params):
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

    result = run_flex_dp_fast(cfg, return_data=True)

    # Data should be populated
    assert result.data is not None
    assert result.data.Yg is not None
    assert result.data.M is not None
    assert result.data.Xg is not None


def test_refaligner_matches_fast(scoring_params):
    """RefAligner.align_simple matches run_flex_dp_fast return_cigar."""
    X = "ACGTACGTACGT"
    Y = "ACGTACGT"
    EP = build_EP_standard(len(X))

    aligner = RefAligner(
        ref=X,
        extra_predecessors=EP,
        **scoring_params,
    )

    score, start_pos, cigar = run_flex_dp_fast(
        FlexInput(X=X, Y=Y, extra_predecessors=EP, **scoring_params),
        return_cigar=True,
    )

    simple = aligner.align_simple(Y)

    assert simple["score"] == score
    assert simple["start_pos"] == start_pos
    assert simple["cigar"] == cigar
