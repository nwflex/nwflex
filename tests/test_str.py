"""
test_str.py — Tests for STR repeat utilities and STR alignment

Combines:
- Unit tests for repeats.py (phase_repeat, valid_phase_combinations, etc.)
- Integration tests for STR alignment (align_STR_block)
- Validation tests comparing against naive enumeration
"""

import pytest
import numpy as np

from nwflex.repeats import (
    phase_repeat,
    valid_phase_combinations,
    count_valid_combinations,
    infer_abM_from_jumps,
    STRLocus,
    CompoundSTRLocus,
)
from nwflex.aligners import align_STR_block
from nwflex.validation import check_alignment_validity, nwg_global

# ===========================================================================
# PART 1: Unit tests for repeat utilities
# ===========================================================================

class TestPhaseRepeat:
    """Test phase_repeat construction: Z* = suf(R,a) · R^M · pre(R,b)."""

    # Cases: (R, a, b, M, expected)
    CASES = [
        # No partial motifs (a=0, b=0)
        ("AT", 0, 0, 3, "ATATAT"),
        ("CAG", 0, 0, 2, "CAGCAG"),
        ("GATA", 0, 0, 1, "GATA"),
        
        # Entry phase only (suffix of R)
        ("AT", 1, 0, 2, "TATAT"),     # suf("AT",1)="T"
        ("CAG", 2, 0, 1, "AGCAG"),    # suf("CAG",2)="AG"
        ("CAG", 1, 0, 2, "GCAGCAG"),  # suf("CAG",1)="G"
        
        # Exit phase only (prefix of R)
        ("AT", 0, 1, 2, "ATATA"),     # pre("AT",1)="A"
        ("CAG", 0, 2, 1, "CAGCA"),    # pre("CAG",2)="CA"
        ("CAG", 0, 1, 2, "CAGCAGC"),  # pre("CAG",1)="C"
        
        # Both entry and exit phases
        ("ACT", 2, 1, 3, "CTACTACTACTA"),  # docstring example
        ("AT", 1, 1, 1, "TATA"),      # suf="T", R^1="AT", pre="A"
        ("CAG", 1, 2, 2, "GCAGCAGCA"),  # suf="G", R^2="CAGCAG", pre="CA"
        
        # Edge cases: M=0
        ("AT", 0, 0, 0, ""),          # Empty result
        ("AT", 1, 0, 0, "T"),         # Only suffix
        ("AT", 0, 1, 0, "A"),         # Only prefix
        ("AT", 1, 1, 0, "TA"),        # Both partials, no complete
        ("CAG", 2, 1, 0, "AGC"),      # suf="AG", pre="C"
        
        # Single character motif
        ("A", 0, 0, 5, "AAAAA"),
    ]

    @pytest.mark.parametrize("R,a,b,M,expected", CASES)
    def test_phase_repeat_cases(self, R, a, b, M, expected):
        """Verify phase_repeat produces correct strings."""
        result = phase_repeat(R, a, b, M)
        assert result == expected, f"phase_repeat({R!r}, {a}, {b}, {M}) = {result!r}, expected {expected!r}"

    def test_phase_repeat_length(self):
        """Verify output length is a + k*M + b."""
        R, a, b, M = "CAG", 2, 1, 4
        result = phase_repeat(R, a, b, M)
        expected_len = a + len(R) * M + b
        assert len(result) == expected_len


class TestValidPhaseCombinations:
    """Test valid_phase_combinations enumeration."""

    def test_count_matches_generator(self):
        """count_valid_combinations matches length of generator."""
        for N in range(1, 5):
            for k in range(1, 4):
                combos = list(valid_phase_combinations(N, k))
                count = count_valid_combinations(N, k)
                assert len(combos) == count, f"N={N}, k={k}: len={len(combos)}, count={count}"

    def test_constraint_satisfied(self):
        """All combinations satisfy M + 1_{a>0} + 1_{b>0} <= N."""
        for N in range(1, 5):
            for k in range(1, 4):
                for a, b, M in valid_phase_combinations(N, k):
                    overhead = (1 if a > 0 else 0) + (1 if b > 0 else 0)
                    assert M + overhead <= N, f"Constraint violated: a={a}, b={b}, M={M}, N={N}"

    def test_a_b_ranges(self):
        """a and b are in range [0, k-1]."""
        N, k = 4, 3
        for a, b, M in valid_phase_combinations(N, k):
            assert 0 <= a < k, f"a={a} out of range for k={k}"
            assert 0 <= b < k, f"b={b} out of range for k={k}"
            assert M >= 0, f"M={M} is negative"

    def test_specific_counts(self):
        """Verify specific known counts."""
        # N=1, k=2: can have M=0 or M=1 (no room for partials if M=1)
        # (0,0,0), (0,0,1), (0,1,0), (1,0,0) = 4 combinations
        assert count_valid_combinations(1, 2) == 4
        
        # N=2, k=2: more combinations possible
        count = count_valid_combinations(2, 2)
        assert count > 4  # Should be more than N=1

    def test_no_duplicates(self):
        """No duplicate combinations."""
        N, k = 3, 3
        combos = list(valid_phase_combinations(N, k))
        assert len(combos) == len(set(combos)), "Duplicate combinations found"


class TestSTRLocus:
    """Test STRLocus dataclass."""

    def test_basic_properties(self):
        """Test X, s, e, k, n properties."""
        locus = STRLocus(A="GAG", R="ACT", N=6, B="GTCA")
        
        assert locus.X == "GAG" + "ACT" * 6 + "GTCA"
        assert locus.s == 3  # len("GAG")
        assert locus.e == 3 + 3 * 6  # 21
        assert locus.k == 3  # len("ACT")
        assert locus.n == len(locus.X)

    def test_empty_flanks(self):
        """Test with empty A and/or B."""
        # Empty A
        locus = STRLocus(A="", R="AT", N=5, B="GGG")
        assert locus.s == 0
        assert locus.X == "AT" * 5 + "GGG"
        
        # Empty B
        locus = STRLocus(A="CCC", R="AT", N=5, B="")
        assert locus.e == locus.n
        assert locus.X == "CCC" + "AT" * 5
        
        # Both empty
        locus = STRLocus(A="", R="CAG", N=4, B="")
        assert locus.s == 0
        assert locus.e == locus.n
        assert locus.X == "CAG" * 4

    def test_build_locus_variant(self):
        """Test build_locus_variant method."""
        locus = STRLocus(A="GAG", R="ACT", N=6, B="GTCA")
        
        # a=2, b=1, M=3 → Z* = "CT" + "ACT"*3 + "A" = "CTACTACTACTA"
        read = locus.build_locus_variant(a=2, b=1, M=3)
        expected = "GAG" + "CTACTACTACTA" + "GTCA"
        assert read == expected
        
        # Full repeat (a=0, b=0, M=6)
        read = locus.build_locus_variant(a=0, b=0, M=6)
        assert read == locus.X
        
        # Empty Z* (a=0, b=0, M=0)
        read = locus.build_locus_variant(a=0, b=0, M=0)
        assert read == "GAG" + "GTCA"

    def test_valid_combinations(self):
        """Test valid_combinations iterator."""
        locus = STRLocus(A="AA", R="CAG", N=4, B="TT")
        
        combos = list(locus.valid_combinations())
        expected_count = count_valid_combinations(locus.N, locus.k)
        assert len(combos) == expected_count
        
        # All should satisfy constraint
        for a, b, M in combos:
            overhead = (1 if a > 0 else 0) + (1 if b > 0 else 0)
            assert M + overhead <= locus.N

    def test_repr(self):
        """Test string representation."""
        locus = STRLocus(A="A", R="TG", N=3, B="C")
        repr_str = repr(locus)
        assert "STRLocus" in repr_str
        assert "TG" in repr_str


class TestCompoundSTRLocus:
    """Test CompoundSTRLocus dataclass."""

    def test_basic_properties(self):
        """Test X and n properties."""
        locus = CompoundSTRLocus(
            A="AAA",
            blocks=[("AT", 3), ("CAG", 2)],
            B="GGG"
        )
        
        expected_X = "AAA" + "AT" * 3 + "CAG" * 2 + "GGG"
        assert locus.X == expected_X
        assert locus.n == len(expected_X)

    def test_single_block(self):
        """Compound with single block behaves like simple STR."""
        locus = CompoundSTRLocus(
            A="AA",
            blocks=[("TG", 4)],
            B="CC"
        )
        
        assert locus.X == "AA" + "TG" * 4 + "CC"

    def test_block_boundaries(self):
        """Test block_boundaries method."""
        locus = CompoundSTRLocus(
            A="AAA",  # len=3
            blocks=[("AT", 3), ("CAG", 2)],  # AT*3=6, CAG*2=6
            B="GGG"
        )
        
        boundaries = locus.block_boundaries()
        
        # First block: starts at 3 (after A), ends at 9, k=2
        assert boundaries[0] == (3, 9, 2)
        
        # Second block: starts at 9, ends at 15, k=3
        assert boundaries[1] == (9, 15, 3)

    def test_empty_flanks(self):
        """Test with empty flanks."""
        locus = CompoundSTRLocus(
            A="",
            blocks=[("AT", 2), ("GC", 2)],
            B=""
        )
        
        assert locus.X == "ATAT" + "GCGC"
        boundaries = locus.block_boundaries()
        assert boundaries[0] == (0, 4, 2)
        assert boundaries[1] == (4, 8, 2)

    def test_three_blocks(self):
        """Test with three repeat blocks."""
        locus = CompoundSTRLocus(
            A="X",
            blocks=[("A", 3), ("TG", 2), ("CAG", 1)],
            B="Y"
        )
        
        assert locus.X == "X" + "AAA" + "TGTG" + "CAG" + "Y"
        
        boundaries = locus.block_boundaries()
        assert len(boundaries) == 3
        assert boundaries[0] == (1, 4, 1)   # A*3
        assert boundaries[1] == (4, 8, 2)   # TG*2
        assert boundaries[2] == (8, 11, 3)  # CAG*1


# ===========================================================================
# PART 2: STR alignment validation tests
# ===========================================================================

class TestSTRPhaseValidation:
    """Validate STR alignment against naive enumeration of all (a,b,M) combinations."""
    
    # Test loci with different motifs and repeat counts
    TEST_LOCI = [
        ("GAG", "ACT", 4, "GTCA"),   # k=3, N=4
        ("AA", "TG", 5, "CC"),        # k=2, N=5
        ("TCA", "CAG", 3, "AC"),           # k=3, N=3
    ]
    
    @pytest.mark.parametrize("A,R,N,B", TEST_LOCI)
    def test_all_phase_combinations_score_correctly(self, A, R, N, B, scoring_params):
        """
        For each valid (a,b,M), verify NW-flex achieves perfect match score.
        """
        locus = STRLocus(A=A, R=R, N=N, B=B)
        match_score = scoring_params["score_matrix"][0, 0]  # diagonal = match
        
        for a, b, M in locus.valid_combinations():
            Y = locus.build_locus_variant(a=a, b=b, M=M)
            expected_score = len(Y) * match_score
            
            result = align_STR_block(locus, Y, **scoring_params)
            
            assert result.score == expected_score, (
                f"Score mismatch for (a={a}, b={b}, M={M}): "
                f"expected {expected_score}, got {result.score}"
            )
    
    @pytest.mark.parametrize("A,R,N,B", TEST_LOCI)
    def test_inferred_abM_matches_input(self, A, R, N, B, scoring_params):
        """
        Verify that (a,b,M) inferred from alignment jumps matches input.
        
        This checks that the traceback correctly identifies which phase
        combination was used in the optimal alignment.
        """
        locus = STRLocus(A=A, R=R, N=N, B=B)
        
        for a, b, M in locus.valid_combinations():
            Y = locus.build_locus_variant(a=a, b=b, M=M)
            
            result = align_STR_block(locus, Y, **scoring_params, return_data=True)
            a_inf, b_inf, M_inf = infer_abM_from_jumps(
                result.jumps, locus.s, locus.e, locus.k
            )
            
            assert (a_inf, b_inf, M_inf) == (a, b, M), (
                f"Phase inference mismatch: input ({a}, {b}, {M}), "
                f"inferred ({a_inf}, {b_inf}, {M_inf})"
            )


class TestSTRFlexVsNaive:
    """Compare STR flex alignment to naive enumeration baseline."""
    
    def sflex_str_naive(self, locus: STRLocus, Y: str, **scoring_params) -> float:
        """
        Naive STR flex baseline: enumerate all valid (a,b,M) and return max NWG score.
        
        This is analogous to sflex_naive for single-block, but restricted to
        phase-valid substrings of the repeat block.
        """
        best_score = -np.inf
        
        for a, b, M in locus.valid_combinations():
            # Build effective reference with this phase combination
            Zstar = phase_repeat(locus.R, a, b, M)
            X_eff = locus.A + Zstar + locus.B
            
            score = nwg_global(X_eff, Y, **scoring_params)
            best_score = max(best_score, score)
        
        return best_score
    
    def test_str_flex_matches_naive(self, rng, scoring_params):
        """STR flex alignment matches naive enumeration on random Y."""
        locus = STRLocus(A="GAG", R="ACT", N=5, B="GTCA")
        
        for _ in range(10):
            # Generate random Y (may not be a perfect locus variant)
            len_Y = rng.integers(10, 25)
            Y = "".join(rng.choice(list("ACGT"), size=len_Y))
            
            flex_result = align_STR_block(locus, Y, **scoring_params)
            naive_score = self.sflex_str_naive(locus, Y, **scoring_params)
            
            assert flex_result.score == naive_score, (
                f"STR flex vs naive mismatch: flex={flex_result.score}, naive={naive_score}"
            )


class TestSTRAlignmentValidity:
    """Validate alignment strings from STR mode."""
    
    def test_alignment_strings_valid(self, scoring_params):
        """STR alignments produce valid strings."""
        locus = STRLocus(A="GATTACA", R="CAA", N=10, B="ACATGAT")
        Y = locus.build_locus_variant(a=0, b=0, M=4)
        
        result = align_STR_block(locus, Y, **scoring_params)
        valid, msg = check_alignment_validity(result, **scoring_params)
        assert valid, msg

    def test_flex_score_geq_standard(self, scoring_params):
        """STR flex score should be >= standard NWG score."""
        locus = STRLocus(A="GATTACA", R="CAA", N=10, B="ACATGAT")
        Y = locus.build_locus_variant(a=1, b=2, M=4)
        
        result = align_STR_block(locus, Y, **scoring_params)
        standard_score = nwg_global(locus.X, Y, **scoring_params)
        
        assert result.score >= standard_score, (
            f"STR flex score {result.score} < standard NWG score {standard_score}"
        )