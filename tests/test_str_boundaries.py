"""Pathological STR regressions extracted from archived debug scripts.

Ported from nwflex_TNG (stress-test branch, commit 19f9ad2). Each test
protects a domain invariant documented in `tests/DOMAIN_GUIDE.md`:
terminal-EP population when `e == n`, no-`B` contraction CIGARs,
partial-phase exit, fast-path vs Python equivalence on the pathological
case, and interrupted-repeat phase preservation.
"""

from __future__ import annotations

import pytest

from nwflex.dp_core import FlexInput, run_flex_dp
from nwflex.aligners import align_STR_block, align_multi_STR, alignment_to_cigar
from nwflex.ep_patterns import build_EP_STR_phase, build_EP_multi_STR_phase
from nwflex.repeats import STRLocus
from nwflex.validation import check_alignment_validity

try:
    from nwflex.fast import CYTHON_AVAILABLE, run_flex_dp_fast
except ImportError:  # pragma: no cover - fallback only if module is unavailable
    CYTHON_AVAILABLE = False
    run_flex_dp_fast = None


def _assert_valid_alignment(result, scoring_params) -> None:
    """Assert a returned alignment is internally consistent."""
    valid, msg = check_alignment_validity(
        result,
        score_matrix=scoring_params["score_matrix"],
        gap_open=scoring_params["gap_open"],
        gap_extend=scoring_params["gap_extend"],
        alphabet_to_index=scoring_params["alphabet_to_index"],
    )
    assert valid, msg


class TestTerminalEPDerivedCases:
    """Regression tests for `e == n` terminal-EP handling."""

    def test_build_ep_str_phase_terminal_predecessors_when_b_absent(self):
        """Single-block STR builder must populate EP[n+1] when B is empty."""
        X = "GGG" + "AC" * 10
        s, e, k = 3, len(X), 2

        EP = build_EP_STR_phase(len(X), s, e, k)

        assert EP[len(X) + 1] == [3, 22]

    def test_build_ep_multi_str_phase_terminal_predecessors_for_last_block(self):
        """Multi-block STR builder must propagate terminal predecessors too."""
        X = "AC" * 17 + "C" + "CA" * 13
        blocks = [(0, 34, 2), (35, 61, 2)]

        EP = build_EP_multi_STR_phase(len(X), blocks)

        assert EP[len(X) + 1] == [35, 60]

    def test_terminal_no_b_contraction_is_perfect(self, scoring_params):
        """Minimal no-B contraction reproducer aligns with perfect score."""
        locus = STRLocus(A="GGG", R="AC", N=10, B="")
        Y = "GGG" + "AC" * 3

        result = align_STR_block(locus, Y, **scoring_params)

        assert result.score == len(Y) * 5.0
        assert alignment_to_cigar(result.path, lenX=len(locus.X), lenY=len(Y)) == (
            1,
            "3M14N6M",
        )
        _assert_valid_alignment(result, scoring_params)

    def test_terminal_no_b_partial_phase_alignment_is_perfect(self, scoring_params):
        """No-B reads that end mid-motif should still align perfectly."""
        locus = STRLocus(A="GGG", R="AC", N=10, B="")
        Y = "GGG" + "AC" * 3 + "A"

        result = align_STR_block(locus, Y, **scoring_params)

        assert result.score == len(Y) * 5.0
        assert alignment_to_cigar(result.path, lenX=len(locus.X), lenY=len(Y)) == (
            1,
            "3M12N7M1N",
        )
        assert any(jump.from_row == len(locus.X) + 1 for jump in result.jumps)
        _assert_valid_alignment(result, scoring_params)

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython fast path unavailable")
    def test_terminal_no_b_case_matches_fast_path(self, scoring_params):
        """Minimal terminal pathological case must match Python and fast paths."""
        locus = STRLocus(A="GGG", R="AC", N=10, B="")
        Y = "GGG" + "AC" * 3 + "A"
        EP = build_EP_STR_phase(len(locus.X), locus.s, locus.e, locus.k)
        cfg = FlexInput(
            X=locus.X,
            Y=Y,
            extra_predecessors=EP,
            score_matrix=scoring_params["score_matrix"],
            gap_open=scoring_params["gap_open"],
            gap_extend=scoring_params["gap_extend"],
            alphabet_to_index=scoring_params["alphabet_to_index"],
        )

        py_result = run_flex_dp(cfg, return_data=False)
        fast_result = run_flex_dp_fast(cfg, return_data=False)

        assert py_result.score == fast_result.score
        assert alignment_to_cigar(
            py_result.path, lenX=len(locus.X), lenY=len(Y)
        ) == alignment_to_cigar(
            fast_result.path, lenX=len(locus.X), lenY=len(Y)
        )
        py_jumps = {
            (jump.from_row, jump.to_row, jump.col, jump.state)
            for jump in py_result.jumps
        }
        fast_jumps = {
            (jump.from_row, jump.to_row, jump.col, jump.state)
            for jump in fast_result.jumps
        }
        assert py_jumps == fast_jumps


class TestInterruptedRepeatDerivedCases:
    """Regressions extracted from interrupted-repeat debug analysis."""

    SHORT_REF = "AC" * 6 + "C" + "CA" * 4
    EXPANDED_REF = "AC" * 17 + "C" + "CA" * 13
    WRONG_PHASE_READ = "AC" * 17 + "C" + "AC" * 13
    BLOCKS_SHORT = [(0, 12, 2), (13, 21, 2)]
    BLOCKS_LONG = [(0, 34, 2), (35, 61, 2)]

    def test_interrupted_repeat_phase_correct_contraction_is_perfect(
        self, scoring_params
    ):
        """Expanded interrupted repeat should contract cleanly to genomic form."""
        result = align_multi_STR(
            self.EXPANDED_REF,
            self.SHORT_REF,
            self.BLOCKS_LONG,
            **scoring_params,
        )

        assert result.score == len(self.SHORT_REF) * 5.0
        assert len(result.jumps) == 2
        assert [(jump.from_row, jump.to_row) for jump in result.jumps] == [
            (0, 23),
            (35, 54),
        ]
        _assert_valid_alignment(result, scoring_params)

    def test_interrupted_repeat_wrong_phase_scores_worse_than_phase_correct(
        self, scoring_params
    ):
        """The second block must preserve CA phase after the interruption."""
        phase_correct = align_multi_STR(
            self.EXPANDED_REF,
            self.EXPANDED_REF,
            self.BLOCKS_LONG,
            **scoring_params,
        )
        wrong_phase = align_multi_STR(
            self.EXPANDED_REF,
            self.WRONG_PHASE_READ,
            self.BLOCKS_LONG,
            **scoring_params,
        )

        assert phase_correct.score == len(self.EXPANDED_REF) * 5.0
        assert wrong_phase.score < phase_correct.score
        _assert_valid_alignment(wrong_phase, scoring_params)

    def test_interrupted_repeat_expansion_is_not_free(self, scoring_params):
        """A short interrupted-repeat reference must not give free expansion."""
        short_ref_result = align_multi_STR(
            self.SHORT_REF,
            self.EXPANDED_REF,
            self.BLOCKS_SHORT,
            **scoring_params,
        )
        expanded_ref_result = align_multi_STR(
            self.EXPANDED_REF,
            self.EXPANDED_REF,
            self.BLOCKS_LONG,
            **scoring_params,
        )

        assert expanded_ref_result.score == len(self.EXPANDED_REF) * 5.0
        assert short_ref_result.score < expanded_ref_result.score
        assert short_ref_result.score < len(self.EXPANDED_REF) * 5.0
        _assert_valid_alignment(short_ref_result, scoring_params)
