"""
test_single_block.py — Tests for NW-flex single-block mode

Verifies that align_single_block matches the naive S_flex definition:
    S_flex(X, Y) = max_{Z* subset Z} NWG(A + Z* + B, Y)
where X = A·Z·B and Z* ranges over all contiguous substrings of Z.
"""

import pytest

from nwflex.validation import (
    check_single_block_case,
    check_AZB_case,
    check_alignment_validity,
    run_mutated_AZB_tests,
)
from nwflex.aligners import align_single_block


class TestSingleBlockVsNaive:
    """Test that align_single_block matches sflex_naive."""

    # Fixed test cases: (A, Z, B, Y, description)
    # Note: Z must be non-empty (s < e required by build_EP_single_block)
    FIXED_CASES = [
        # Basic cases
        ("AC", "GGG", "GT", "ACGT", "simple block with partial match"),
        ("A", "CG", "T", "AT", "skip entire block"),
        ("A", "CG", "T", "ACGT", "keep entire block"),
        ("A", "CGT", "A", "ACA", "partial block usage"),
        
        # Edge cases with non-empty Z
        ("A", "C", "T", "ACT", "minimal: single base block match"),
        ("A", "C", "T", "AT", "minimal: single base block skip"),
        ("AA", "CCCC", "AA", "AAAA", "skip long block"),
        ("AA", "CCCC", "AA", "AACCCCAA", "keep long block"),
        
        # Repeat-like patterns
        ("A", "TATA", "A", "ATATATA", "repeat block"),
        ("AC", "GTGT", "AC", "ACGTAC", "partial repeat"),
    ]

    @pytest.mark.parametrize("A,Z,B,Y,desc", FIXED_CASES)
    def test_fixed_cases(self, A, Z, B, Y, desc, scoring_params):
        """NW-flex single-block matches naive on fixed cases."""
        flex_score, naive_score = check_AZB_case(A, Z, B, Y, **scoring_params)
        assert flex_score == naive_score, f"Mismatch on '{desc}': flex={flex_score}, naive={naive_score}"

    def test_random_sequences(self, rng, random_dna_factory, scoring_params):
        """NW-flex single-block matches naive on random sequences."""
        for _ in range(20):
            # Random block structure
            len_A = rng.integers(0, 10)
            len_Z = rng.integers(1, 8)  # At least 1 to have flexibility
            len_B = rng.integers(0, 10)  # Can be empty (block at end)
            len_Y = rng.integers(5, 25)
            
            A = random_dna_factory(len_A, rng)
            Z = random_dna_factory(len_Z, rng)
            B = random_dna_factory(len_B, rng)
            Y = random_dna_factory(len_Y, rng)
            
            flex_score, naive_score = check_AZB_case(A, Z, B, Y, **scoring_params)
            assert flex_score == naive_score, f"Random mismatch: A={A}, Z={Z}, B={B}, Y={Y}"


class TestBlockBoundaries:
    """Test edge cases at block boundaries."""

    def test_block_at_start(self, scoring_params):
        """Flexible block at the start of X (A is empty)."""
        X = "ACGTTT"  # Z=ACGT at start, B=TT
        Y = "TT"
        s, e = 0, 4  # Block is X[0:4] = "ACGT"
        
        flex_score, naive_score = check_single_block_case(X, Y, s, e, **scoring_params)
        assert flex_score == naive_score

    def test_block_at_end(self, scoring_params):
        """Flexible block at the end of X (B is empty)."""
        X = "TTACGT"  # A=TT, Z=ACGT at end
        Y = "TTAC"    # Matches TT + AC (partial block)
        s, e = 2, 6   # Block is X[2:6] = "ACGT"
        
        flex_score, naive_score = check_single_block_case(X, Y, s, e, **scoring_params)
        assert flex_score == naive_score

    def test_block_in_middle(self, scoring_params):
        """Flexible block in the middle of X."""
        X = "AACCCGGG"  # A=AA, Z=CCC, B=GGG
        Y = "AAGGG"     # Skip the block
        s, e = 2, 5     # Block is X[2:5] = "CCC"
        
        flex_score, naive_score = check_single_block_case(X, Y, s, e, **scoring_params)
        assert flex_score == naive_score

    def test_small_block(self, scoring_params):
        """Small flexible block (2 bases)."""
        X = "AACGT"  # A=A, Z=AC, B=GT
        Y = "AGT"    # Skip AC block
        s, e = 1, 3  # Block is X[1:3] = "AC"
        
        flex_score, naive_score = check_single_block_case(X, Y, s, e, **scoring_params)
        assert flex_score == naive_score


class TestSingleBlockAlignmentValidity:
    """Test alignment validity for single-block mode."""

    def test_alignment_strings_valid(self, scoring_params):
        """Single-block alignments produce valid strings."""
        X = "ACGTACGT"
        Y = "ACGACG"
        s, e = 2, 6  # Block in the middle
        
        result = align_single_block(X, Y, s, e, **scoring_params)
        valid, msg = check_alignment_validity(result, **scoring_params)
        assert valid, msg

    def test_random_alignment_validity(self, rng, random_dna_factory, scoring_params):
        """Random single-block alignments are valid."""
        for _ in range(10):
            nX = rng.integers(10, 30)
            nY = rng.integers(10, 30)
            X = random_dna_factory(nX, rng)
            Y = random_dna_factory(nY, rng)
            
            # Random block within X
            s = rng.integers(0, nX // 2)
            e = rng.integers(s + 1, nX)
            
            result = align_single_block(X, Y, s, e, **scoring_params)
            valid, msg = check_alignment_validity(result, **scoring_params)
            assert valid, msg


class TestMutatedSequences:
    """Test with mutated sequences for broader coverage."""

    BASE_CASES = [
        ("AC", "GGG", "GT", "ACGGGGT", "three G block"),
        ("AA", "TATA", "AA", "AATATAA", "repeat pattern"),
        ("", "ACGT", "", "ACGT", "full flexibility"),
    ]

    def test_mutated_cases(self, scoring_params):
        """NW-flex matches naive on mutated versions of base cases."""
        results = run_mutated_AZB_tests(
            base_cases=self.BASE_CASES,
            num_mutations=5,
            sub_rate=0.1,
            indel_rate=0.05,
            seed=888,
            **scoring_params,
        )
        
        for r in results:
            assert r["match"], (
                f"Mismatch on mutated '{r['description']}': "
                f"flex={r['flex']}, naive={r['naive']}, Y_mut={r['Y_mutated']}"
            )
