"""
test_standard.py — Tests for standard NW-flex alignment (E(i) = empty)

Verifies that NW-flex with no extra predecessors matches the independent
Needleman-Wunsch/Gotoh baseline implementation.
"""

import pytest
import numpy as np

from nwflex.validation import check_standard_vs_nwg, nwg_global
from nwflex.aligners import align_standard


class TestStandardVsNWG:
    """Test that align_standard matches nwg_global on various inputs."""

    # Fixed test cases with known behavior
    FIXED_CASES = [
        # (X, Y, description)
        ("ACGT", "ACGT", "identical sequences"),
        ("AAAA", "TTTT", "all mismatches"),
        ("ACGTACGT", "ACGT", "X longer than Y"),
        ("ACGT", "ACGTACGT", "Y longer than X"),
        ("A", "A", "single base match"),
        ("A", "T", "single base mismatch"),
        ("ACGT", "AGT", "deletion in Y"),
        ("AGT", "ACGT", "insertion in Y"),
        ("GATTACA", "GCATGCU".replace("U", "T"), "classic example"),
        ("AAAAACGT", "ACGTAAAA", "shifted pattern"),
    ]

    @pytest.mark.parametrize("X,Y,desc", FIXED_CASES)
    def test_fixed_cases(self, X, Y, desc, scoring_params):
        """NW-flex standard mode matches nwg_global on fixed cases."""
        flex_score, nwg_score = check_standard_vs_nwg(X, Y, **scoring_params)
        assert flex_score == nwg_score, f"Mismatch on '{desc}': flex={flex_score}, nwg={nwg_score}"

    def test_random_sequences(self, rng, random_dna_factory, scoring_params):
        """NW-flex standard mode matches nwg_global on random sequences."""
        for length in [5, 10, 20, 50]:
            X = random_dna_factory(length, rng)
            Y = random_dna_factory(length, rng)
            flex_score, nwg_score = check_standard_vs_nwg(X, Y, **scoring_params)
            assert flex_score == nwg_score, f"Mismatch on random len={length}"

    def test_random_asymmetric(self, rng, random_dna_factory, scoring_params):
        """NW-flex handles asymmetric sequence lengths correctly."""
        for nX, nY in [(10, 20), (30, 15), (5, 50)]:
            X = random_dna_factory(nX, rng)
            Y = random_dna_factory(nY, rng)
            flex_score, nwg_score = check_standard_vs_nwg(X, Y, **scoring_params)
            assert flex_score == nwg_score, f"Mismatch on random nX={nX}, nY={nY}"


class TestAlignmentValidity:
    """Test that standard alignments produce valid alignment strings."""

    def test_alignment_lengths_match(self, scoring_params):
        """X_aln and Y_aln have the same length."""
        X, Y = "ACGTACGT", "ACGACG"
        result = align_standard(X, Y, **scoring_params)
        assert len(result.X_aln) == len(result.Y_aln)

    def test_no_double_gaps(self, scoring_params):
        """No position has gaps in both X_aln and Y_aln."""
        X, Y = "ACGTACGT", "ACGACG"
        result = align_standard(X, Y, **scoring_params)
        for i, (x, y) in enumerate(zip(result.X_aln, result.Y_aln)):
            assert not (x == '-' and y == '-'), f"Double gap at position {i}"

    def test_ungapped_recovers_original(self, scoring_params):
        """Removing gaps from alignment recovers original sequences."""
        X, Y = "ACGTACGT", "ACGACG"
        result = align_standard(X, Y, **scoring_params)
        X_recovered = result.X_aln.replace('-', '')
        Y_recovered = result.Y_aln.replace('-', '')
        assert X_recovered == X, f"X mismatch: {X_recovered} != {X}"
        assert Y_recovered == Y, f"Y mismatch: {Y_recovered} != {Y}"

    def test_random_alignment_validity(self, rng, random_dna_factory, scoring_params):
        """Random alignments produce valid strings."""
        for _ in range(10):
            X = random_dna_factory(20, rng)
            Y = random_dna_factory(20, rng)
            result = align_standard(X, Y, **scoring_params)
            
            assert len(result.X_aln) == len(result.Y_aln)
            for i, (x, y) in enumerate(zip(result.X_aln, result.Y_aln)):
                assert not (x == '-' and y == '-')
            assert result.X_aln.replace('-', '') == X
            assert result.Y_aln.replace('-', '') == Y
