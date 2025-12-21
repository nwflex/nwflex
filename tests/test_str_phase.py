"""
test_str_phase.py — Tests for STR phase-restricted alignment

Verifies that align_STR_block produces valid alignments and that
the phase-restricted EP pattern behaves correctly.
"""

import pytest
import numpy as np

from nwflex.aligners import align_STR_block
from nwflex.validation import check_alignment_validity, nwg_global
from nwflex.repeats import phase_repeat


class TestSTRPhaseBasics:
    """Basic tests for STR phase alignment."""

    def test_perfect_repeat_alignment(self, scoring_params):
        """Align a perfect repeat to itself."""
        R = "CAG"
        k = len(R)
        num_repeats = 5
        X = R * num_repeats  # "CAGCAGCAGCAGCAG"
        Y = R * num_repeats
        
        result = align_STR_block(
            X=X,
            Y=Y,
            s=0,
            e=len(X),
            k=k,
            **scoring_params,
        )
        
        # Perfect match should have high score
        assert result.score > 0
        valid, msg = check_alignment_validity(result)
        assert valid, msg

    def test_repeat_expansion(self, scoring_params):
        """Y has more repeats than X (expansion)."""
        R = "CAG"
        k = len(R)
        X = R * 3  # Reference: 3 repeats
        Y = R * 5  # Read: 5 repeats
        
        result = align_STR_block(
            X=X,
            Y=Y,
            s=0,
            e=len(X),
            k=k,
            **scoring_params,
        )
        
        valid, msg = check_alignment_validity(result)
        assert valid, msg

    def test_repeat_contraction(self, scoring_params):
        """Y has fewer repeats than X (contraction)."""
        R = "CAG"
        k = len(R)
        X = R * 5  # Reference: 5 repeats
        Y = R * 3  # Read: 3 repeats
        
        result = align_STR_block(
            X=X,
            Y=Y,
            s=0,
            e=len(X),
            k=k,
            **scoring_params,
        )
        
        valid, msg = check_alignment_validity(result)
        assert valid, msg

    def test_flanking_regions(self, scoring_params):
        """STR with flanking sequences."""
        R = "AT"
        k = len(R)
        flank_L = "GGCC"
        flank_R = "CCGG"
        X = flank_L + R * 4 + flank_R
        Y = flank_L + R * 6 + flank_R  # Expansion
        
        s = len(flank_L)
        e = len(X) - len(flank_R)
        
        result = align_STR_block(
            X=X,
            Y=Y,
            s=s,
            e=e,
            k=k,
            **scoring_params,
        )
        
        valid, msg = check_alignment_validity(result)
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
        R = "CAG"
        k = len(R)
        X = R * 4
        Y = R * 3  # Contraction
        
        result_phase = align_STR_block(
            X=X,
            Y=Y,
            s=0,
            e=len(X),
            k=k,
            **scoring_params,
        )
        
        score_standard = nwg_global(X, Y, **scoring_params)
        
        # Flex should find at least as good an alignment
        assert result_phase.score >= score_standard, (
            f"Phase score {result_phase.score} < standard {score_standard}"
        )

    def test_benefits_from_flexibility(self, scoring_params):
        """Cases where flex should outperform standard alignment."""
        R = "CAG"
        k = len(R)
        # X has 5 repeats, Y has 3 - standard alignment must pay gaps
        X = R * 5
        Y = R * 3
        
        result_phase = align_STR_block(
            X=X,
            Y=Y,
            s=0,
            e=len(X),
            k=k,
            **scoring_params,
        )
        
        score_standard = nwg_global(X, Y, **scoring_params)
        
        # With contraction, flex can skip repeats without gap penalty
        # Should be at least as good as standard
        assert result_phase.score >= score_standard


class TestSTRPhaseAlignmentValidity:
    """Validate alignment strings from STR phase mode."""

    REPEAT_UNITS = ["AT", "CAG", "CAGCAG", "AAAG"]

    @pytest.mark.parametrize("R", REPEAT_UNITS)
    def test_various_repeats(self, R, scoring_params):
        """Test alignment validity with various repeat units."""
        k = len(R)
        X = R * 4
        Y = R * 5
        
        result = align_STR_block(
            X=X,
            Y=Y,
            s=0,
            e=len(X),
            k=k,
            **scoring_params,
        )
        
        valid, msg = check_alignment_validity(result)
        assert valid, f"Invalid alignment for R={R}: {msg}"

    def test_random_str_alignment(self, rng, scoring_params):
        """Random STR alignments produce valid results."""
        repeats = ["AT", "CAG", "AAAG", "TTTA"]
        
        for _ in range(10):
            R = repeats[rng.integers(0, len(repeats))]  # Pick random repeat unit
            k = len(R)
            n_ref = int(rng.integers(2, 8))
            n_read = int(rng.integers(2, 8))
            
            X = R * n_ref
            Y = R * n_read
            
            result = align_STR_block(
                X=X,
                Y=Y,
                s=0,
                e=len(X),
                k=k,
                **scoring_params,
            )
            
            valid, msg = check_alignment_validity(result)
            assert valid, msg
