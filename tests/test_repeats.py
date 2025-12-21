"""
test_repeats.py — Tests for STR repeat utilities

Tests the functions in nwflex.repeats:
- phase_repeat: phased repeat string construction
- valid_phase_combinations: enumeration of valid (a, b, M) tuples
- count_valid_combinations: counting valid combinations
- infer_abM_from_jumps: phase inference from alignment jumps
- STRLocus: STR locus dataclass
- CompoundSTRLocus: compound STR locus dataclass
"""

import pytest
from nwflex.repeats import (
    phase_repeat,
    valid_phase_combinations,
    count_valid_combinations,
    infer_abM_from_jumps,
    STRLocus,
    CompoundSTRLocus,
)
from nwflex.dp_core import RowJump


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


class TestInferABMFromJumps:
    """Test phase inference from alignment row jumps."""

    def test_no_jumps_returns_none(self):
        """Empty jumps list returns (None, None, None)."""
        result = infer_abM_from_jumps([], s=5, e=15, k=3)
        assert result == (None, None, None)

    def test_missing_entry_jump(self):
        """Missing entry jump returns None."""
        # Only exit jump, no entry
        jumps = [RowJump(from_row=10, to_row=16, col=5, state=1)]
        result = infer_abM_from_jumps(jumps, s=5, e=15, k=3)
        assert result == (None, None, None)

    def test_missing_exit_jump(self):
        """Missing exit jump returns None."""
        # Only entry jump, no exit
        jumps = [RowJump(from_row=5, to_row=8, col=3, state=1)]
        result = infer_abM_from_jumps(jumps, s=5, e=15, k=3)
        assert result == (None, None, None)

    def test_valid_jumps_infer_phase(self):
        """Valid entry and exit jumps produce phase inference."""
        # Construct jumps that should give known phases
        s, e, k = 5, 14, 3  # Block from row 6 to 14, motif length 3
        
        # Entry from s=5 to row 8, exit from row 12 to e+1=15
        entry_jump = RowJump(from_row=5, to_row=8, col=3, state=1)
        exit_jump = RowJump(from_row=12, to_row=15, col=10, state=1)
        jumps = [entry_jump, exit_jump]
        
        a, b, M = infer_abM_from_jumps(jumps, s=s, e=e, k=k)
        
        # Should return valid integers (not None)
        assert a is not None
        assert b is not None
        assert M is not None
        assert isinstance(a, int)
        assert isinstance(b, int)
        assert isinstance(M, int)


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
