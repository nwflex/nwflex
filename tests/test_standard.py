"""
test_standard.py — Tests for standard NW-flex alignment (E(i) = empty)

Verifies that NW-flex with no extra predecessors matches the independent
Needleman-Wunsch/Gotoh baseline implementation.
"""

import pytest

from nwflex.validation import check_standard_vs_nwg, check_alignment_validity
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

    def test_ungapped_recovers_original(self, scoring_params):
        """Removing gaps from alignment recovers original sequences."""
        X, Y = "ACGTACGT", "ACGACG"
        result = align_standard(X, Y, **scoring_params)
        X_recovered = result.X_aln.replace('-', '')
        Y_recovered = result.Y_aln.replace('-', '')
        assert X_recovered == X, f"X mismatch: {X_recovered} != {X}"
        assert Y_recovered == Y, f"Y mismatch: {Y_recovered} != {Y}"
    
    def test_alignment_validity_checks(self, scoring_params):
        """
        Verify that align_standard produces alignment results that are:
        1. The same length for X_aln and Y_aln,
        2. Free of double gaps at any position,
        3. Consistent between the reported score and the score recomputed from the alignment strings.

        This test uses check_alignment_validity to confirm all three properties.
        """
        X, Y = "ACGTACGT", "ACGACG"
        result = align_standard(X, Y, **scoring_params)
        valid, msg = check_alignment_validity(result, **scoring_params)
        assert valid, msg

    def test_random_alignment_validity(self, rng, random_dna_factory, scoring_params):
        """Random alignments produce valid strings."""
        for _ in range(20):
            X = random_dna_factory(20, rng)
            Y = random_dna_factory(20, rng)
            result = align_standard(X, Y, **scoring_params)
            valid, msg = check_alignment_validity(result, **scoring_params)
            assert result.X_aln.replace('-', '') == X
            assert result.Y_aln.replace('-', '') == Y
            assert valid, msg


class TestScoringSchemeVariants:
    """Test across different scoring schemes."""
    
    SCORING_VARIANTS = [
        ("default", -20.0, -1.0),
        ("smaller_gap_open", -10.0, -1.0),
        ("larger_gap_extend", -20.0, -3.0),
        ("balanced_gaps", -4.0, -4.0),
    ]
    
    @pytest.mark.parametrize("name,gap_open,gap_extend", SCORING_VARIANTS)
    def test_flex_matches_nwg_various_scoring(self, name, gap_open, gap_extend, 
                                               score_matrix, alphabet_to_index, rng, random_dna_factory):
        """NW-flex matches nwg_global across scoring schemes."""
        for _ in range(20):
            X = random_dna_factory(15, rng)
            Y = random_dna_factory(15, rng)
            flex_score, nwg_score = check_standard_vs_nwg(
                X, Y,
                score_matrix=score_matrix,
                gap_open=gap_open,
                gap_extend=gap_extend,
                alphabet_to_index=alphabet_to_index,
            )
            assert flex_score == nwg_score , f"Mismatch under {name} scoring"