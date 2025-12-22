"""
test_str_phase.py — Tests for STR phase-restricted alignment

Verifies that align_STR_block produces valid alignments and that
the phase-restricted EP pattern behaves correctly.
"""

import pytest
import numpy as np

from nwflex.aligners import align_STR_block
from nwflex.validation import check_alignment_validity, nwg_global
from nwflex.repeats import STRLocus, phase_repeat

A = "GATTACA"
B = "ACATGAT"
R = "CAA"
N = 10
M = 4

strLocus = STRLocus(A=A, R=R, N=N, B=B)


class TestSTRPhaseBasics:
    """Basic tests for STR phase alignment."""

    def test_perfect_repeat_alignment(self, scoring_params):
        """Align a perfect repeat to itself."""        
        
        result = align_STR_block(
            strLocus=strLocus,            
            Y=strLocus.X,
            **scoring_params,
        )
        valid, msg = check_alignment_validity(result, **scoring_params)
        assert valid, msg

    def test_repeat_contraction(self, scoring_params):
        """Y has fewer repeats than X (contraction)."""
        Y = strLocus.build_locus_variant(a=0, b=0, M=4)                
        result = align_STR_block(
            strLocus=strLocus,            
            Y=Y,
            **scoring_params,
        )
        
        valid, msg = check_alignment_validity(result, **scoring_params)
        assert valid, msg


class TestPhaseRepeat:
    """Test the phase_repeat utility function."""

    def test_phase_zero(self):
        """Phase 0: repeat starts at boundary, complete copies only."""
        R = "CAG"
        M = 3  # Number of repeat units
        
        # a=0, b=0 means no partial motifs
        result = phase_repeat(R, a=0, b=0, M=M)
        expected = R * M  # "CAGCAGCAG"
        assert result == expected

    def test_phase_with_suffix(self):
        """Phase with entry suffix (partial start)."""
        R = "CAG"
        # a=2 means suffix of length 2: "AG"
        result = phase_repeat(R, a=2, b=0, M=2)
        expected = "AG" + "CAGCAG"  # suffix + 2 complete copies
        assert result == expected

    def test_phase_with_prefix(self):
        """Phase with exit prefix (partial end)."""
        R = "CAG"
        # b=1 means prefix of length 1: "C"
        result = phase_repeat(R, a=0, b=1, M=2)
        expected = "CAGCAG" + "C"  # 2 complete copies + prefix
        assert result == expected

    def test_phase_both_partial(self):
        """Phase with both entry suffix and exit prefix."""
        R = "ACT"
        # a=2 -> suffix "CT", b=1 -> prefix "A"
        result = phase_repeat(R, a=2, b=1, M=3)
        expected = "CT" + "ACTACTACT" + "A"
        assert result == expected


class TestSTRPhaseVsStandard:
    """Compare STR phase alignment to standard alignment behavior."""

    def test_no_worse_than_standard(self, scoring_params):
        """STR phase alignment should be >= standard alignment score."""
        Y = strLocus.build_locus_variant(a=1, b=2, M=4)                
        
        result_phase = align_STR_block(
            strLocus=strLocus,
            Y=Y,
            **scoring_params,
        )
        
        score_standard = nwg_global(strLocus.X, Y, **scoring_params)
        
        # Flex should find at least as good an alignment
        assert result_phase.score >= score_standard, (
            f"Phase score {result_phase.score} < standard {score_standard}"
        )

    def test_benefits_from_flexibility(self, scoring_params):
        """Cases where flex should outperform standard alignment."""
        # X has 10 repeats, Y has 4 - standard alignment must pay gaps
        Y = strLocus.build_locus_variant(a=0, b=0, M=4)
        
        result_phase = align_STR_block(
            strLocus=strLocus,
            Y=Y,
            **scoring_params,
        )
        score_standard = nwg_global(strLocus.X, Y, **scoring_params)
        
        # With contraction, flex can skip repeats without gap penalty
        # Should be at least as good as standard
        assert result_phase.score >= score_standard


class TestSTRPhaseAlignmentValidity:
    """Validate alignment strings from STR phase mode."""

    REPEAT_UNITS = ["AT", "CAG", "CAGCAG", "AAAG"]

    @pytest.mark.parametrize("R", REPEAT_UNITS)
    def test_various_repeats(self, R, scoring_params):
        """Test alignment validity with various repeat units."""
        strLocus = STRLocus(A=A, R=R, N=N, B=B)
        Y = strLocus.build_locus_variant(a=1, b=1, M=4)
        result = align_STR_block(
            strLocus,             
            Y=Y,
            **scoring_params,
        )            
        valid, msg = check_alignment_validity(result, **scoring_params)
        assert valid, f"Invalid alignment for R={R}: {msg}"

    def test_random_str_alignment(self, rng, scoring_params):
        """Random STR alignments produce valid results."""
        repeats = ["AT", "CAG", "AAAG", "TTTA"]
        
        for _ in range(10):
            R = repeats[rng.integers(0, len(repeats))]  # Pick random repeat unit
            k = len(R)
            n_ref = int(rng.integers(2, 8))
            n_read = int(rng.integers(2, 8))
            strLocus = STRLocus(A=A, R=R, N=n_ref, B=B)
            Y = strLocus.build_locus_variant(a=0, b=0, M=n_read)
            
            result = align_STR_block(
                strLocus=strLocus,
                Y=Y,
                **scoring_params,
            )
            valid, msg = check_alignment_validity(result, **scoring_params)
            assert valid, msg
