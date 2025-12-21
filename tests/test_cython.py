"""
test_cython.py — Tests for Cython-accelerated NW-flex implementation

Verifies that run_flex_dp_fast (Cython) produces identical results to
run_flex_dp (Python) for both standard and single-block modes.

Note: run_flex_dp returns a tuple (score, X_aln, Y_aln, path, jumps, [data])
      run_flex_dp_fast returns an AlignmentResult object

Note 2: uses of to_tuple() are to unpack AlignmentResult into tuple form.
        not robustly written, but backfilling for previous function signature.
"""

import pytest
import numpy as np

from nwflex.dp_core import FlexInput, FlexData, run_flex_dp
from nwflex.fast import run_flex_dp_fast
from nwflex.ep_patterns import build_EP_standard, build_EP_single_block


class TestCythonVsPython:
    """Test that Cython implementation matches Python exactly."""

    def test_standard_mode_score(self, rng, random_dna_factory, scoring_params):
        """Cython matches Python score in standard mode."""
        for length in [10, 25, 50]:
            X = random_dna_factory(length, rng)
            Y = random_dna_factory(length, rng)
            EP = build_EP_standard(len(X))
            
            cfg = FlexInput(
                X=X,
                Y=Y,
                extra_predecessors=EP,
                **scoring_params,
            )
            
            # Python returns tuple: (score, X_aln, Y_aln, path, jumps)
            py_result = run_flex_dp(cfg, return_data=False).to_tuple()
            py_score = py_result[0]
            
            # Cython returns AlignmentResult object
            cy_result = run_flex_dp_fast(cfg, return_data=False)
            
            assert py_score == cy_result.score, (
                f"Score mismatch at len={length}: py={py_score}, cy={cy_result.score}"
            )

    def test_standard_mode_alignment(self, rng, random_dna_factory, scoring_params):
        """Cython matches Python alignment strings in standard mode."""
        for _ in range(5):
            X = random_dna_factory(20, rng)
            Y = random_dna_factory(20, rng)
            EP = build_EP_standard(len(X))
            
            cfg = FlexInput(
                X=X,
                Y=Y,
                extra_predecessors=EP,
                **scoring_params,
            )
            
            py_result = run_flex_dp(cfg, return_data=False).to_tuple()
            py_score, py_X_aln, py_Y_aln = py_result[0], py_result[1], py_result[2]
            
            cy_result = run_flex_dp_fast(cfg, return_data=False)
            
            assert py_X_aln == cy_result.X_aln, "X_aln mismatch"
            assert py_Y_aln == cy_result.Y_aln, "Y_aln mismatch"

    def test_single_block_mode_score(self, rng, random_dna_factory, scoring_params):
        """Cython matches Python score in single-block mode."""
        for _ in range(10):
            nX = rng.integers(15, 40)
            nY = rng.integers(15, 40)
            X = random_dna_factory(nX, rng)
            Y = random_dna_factory(nY, rng)
            
            # Random block (ensure s < e)
            s = rng.integers(0, nX // 3)
            e = rng.integers(s + 2, min(s + 10, nX))
            EP = build_EP_single_block(nX, s, e)
            
            cfg = FlexInput(
                X=X,
                Y=Y,
                extra_predecessors=EP,
                **scoring_params,
            )
            
            py_result = run_flex_dp(cfg, return_data=False).to_tuple()
            py_score = py_result[0]
            
            cy_result = run_flex_dp_fast(cfg, return_data=False)
            
            assert py_score == cy_result.score, (
                f"Score mismatch: s={s}, e={e}, py={py_score}, cy={cy_result.score}"
            )

    def test_single_block_mode_alignment(self, rng, random_dna_factory, scoring_params):
        """Cython matches Python alignment in single-block mode."""
        for _ in range(5):
            nX = 25
            nY = 25
            X = random_dna_factory(nX, rng)
            Y = random_dna_factory(nY, rng)
            
            s, e = 5, 15
            EP = build_EP_single_block(nX, s, e)
            
            cfg = FlexInput(
                X=X,
                Y=Y,
                extra_predecessors=EP,
                **scoring_params,
            )
            
            py_result = run_flex_dp(cfg, return_data=False).to_tuple()
            py_X_aln, py_Y_aln = py_result[1], py_result[2]
            
            cy_result = run_flex_dp_fast(cfg, return_data=False)
            
            assert py_X_aln == cy_result.X_aln, "X_aln mismatch in single-block"
            assert py_Y_aln == cy_result.Y_aln, "Y_aln mismatch in single-block"


def _dp_tables_equivalent(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Check if two DP tables are functionally equivalent.
    
    Python uses float('-inf') which becomes np.inf, while Cython uses -1e300.
    Both represent "unreachable" states, so we treat them as equivalent.
    """
    # Threshold for "effectively -inf" (Cython uses -1e300)
    NEG_INF_THRESHOLD = -1e200
    
    # Create masks for "effectively -inf" values
    a_neginf = a < NEG_INF_THRESHOLD
    b_neginf = b < NEG_INF_THRESHOLD
    
    # Both should have -inf in the same places
    if not np.array_equal(a_neginf, b_neginf):
        return False
    
    # For non-inf values, they should be close
    finite_mask = ~a_neginf
    if np.any(finite_mask):
        if not np.allclose(a[finite_mask], b[finite_mask]):
            return False
    
    return True


class TestCythonDPTables:
    """Test that Cython DP tables match Python tables."""

    def test_dp_tables_match(self, scoring_params):
        """Full DP tables (Yg, M, Xg) are identical between implementations."""
        X = "ACGTACGT"
        Y = "ACGACG"
        EP = build_EP_standard(len(X))
        
        cfg = FlexInput(
            X=X,
            Y=Y,
            extra_predecessors=EP,
            **scoring_params,
        )
        
        # Python with return_data=True returns 6 elements
        py_result = run_flex_dp(cfg, return_data=True).to_tuple()
        py_data = py_result[5]  # data is 6th element
        
        cy_result = run_flex_dp_fast(cfg, return_data=True)
        
        # Compare DP tables (handles -inf vs -1e300)
        assert _dp_tables_equivalent(py_data.Yg, cy_result.data.Yg), "Yg tables differ"
        assert _dp_tables_equivalent(py_data.M, cy_result.data.M), "M tables differ"
        assert _dp_tables_equivalent(py_data.Xg, cy_result.data.Xg), "Xg tables differ"

    def test_dp_tables_single_block(self, scoring_params):
        """DP tables match in single-block mode."""
        X = "AACCCGTGT"
        Y = "AGTGT"
        s, e = 2, 5  # Block is "CCC"
        EP = build_EP_single_block(len(X), s, e)
        
        cfg = FlexInput(
            X=X,
            Y=Y,
            extra_predecessors=EP,
            **scoring_params,
        )
        
        py_result = run_flex_dp(cfg, return_data=True).to_tuple()
        py_data = py_result[5]
        
        cy_result = run_flex_dp_fast(cfg, return_data=True)
        
        assert _dp_tables_equivalent(py_data.Yg, cy_result.data.Yg), "Yg tables differ"
        assert _dp_tables_equivalent(py_data.M, cy_result.data.M), "M tables differ"
        assert _dp_tables_equivalent(py_data.Xg, cy_result.data.Xg), "Xg tables differ"


class TestCythonJumps:
    """Test that jump detection is consistent between implementations."""

    def test_jumps_match(self, rng, random_dna_factory, scoring_params):
        """Row jumps detected by Cython match those from Python."""
        for _ in range(5):
            nX = 20
            nY = 20
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
            
            py_result = run_flex_dp(cfg, return_data=False).to_tuple()
            py_jumps = py_result[4]  # jumps is 5th element
            
            cy_result = run_flex_dp_fast(cfg, return_data=False)
            
            # Compare jump lists
            assert len(py_jumps) == len(cy_result.jumps), "Jump count mismatch"
            for jp, jc in zip(py_jumps, cy_result.jumps):
                assert jp.from_row == jc.from_row, "Jump from_row mismatch"
                assert jp.to_row == jc.to_row, "Jump to_row mismatch"
                assert jp.col == jc.col, "Jump col mismatch"
