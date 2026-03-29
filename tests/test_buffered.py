"""
test_buffered.py — Tests for buffered Cython DP path

Verifies that the buffered CIGAR path (nwflex_dp_core_buffered_cigar via
RefAligner.align_simple) produces results consistent with the pure Python
reference (run_flex_dp) across standard, single-block, and STR modes.
"""

import pytest
import numpy as np

from nwflex.dp_core import FlexInput, run_flex_dp
from nwflex.aligners import RefAligner, alignment_to_cigar
from nwflex.ep_patterns import (
    build_EP_standard,
    build_EP_single_block,
    build_EP_STR_phase,
)


def _make_buffered_aligner(X, EP, scoring_params, **kwargs):
    """Create a RefAligner configured to use the buffered CIGAR path."""
    return RefAligner(
        ref=X,
        extra_predecessors=EP,
        fast_mode=True,
        fast_cigar_only=True,
        **scoring_params,
        **kwargs,
    )


def _python_score_and_cigar(X, Y, EP, scoring_params):
    """Run pure Python DP and return (score, start_pos, cigar)."""
    cfg = FlexInput(X=X, Y=Y, extra_predecessors=EP, **scoring_params)
    result = run_flex_dp(cfg, return_data=False)
    start_pos, cigar = alignment_to_cigar(result.path, lenX=len(X), lenY=len(Y))
    return result.score, start_pos, cigar


class TestBufferedVsPython:
    """Buffered CIGAR path matches pure Python for standard mode."""

    def test_standard_score(self, rng, random_dna_factory, scoring_params):
        """Scores match across various lengths."""
        for length in [10, 25, 50]:
            X = random_dna_factory(length, rng)
            Y = random_dna_factory(length, rng)
            EP = build_EP_standard(len(X))

            py_score, _, _ = _python_score_and_cigar(X, Y, EP, scoring_params)

            aligner = _make_buffered_aligner(X, EP, scoring_params)
            buf = aligner.align_simple(Y)

            assert py_score == pytest.approx(buf["score"], abs=0.01), (
                f"Score mismatch at len={length}: py={py_score}, buf={buf['score']}"
            )

    def test_standard_cigar(self, rng, random_dna_factory, scoring_params):
        """CIGAR and start_pos match in standard mode."""
        for _ in range(5):
            X = random_dna_factory(20, rng)
            Y = random_dna_factory(20, rng)
            EP = build_EP_standard(len(X))

            py_score, py_start, py_cigar = _python_score_and_cigar(
                X, Y, EP, scoring_params
            )

            aligner = _make_buffered_aligner(X, EP, scoring_params)
            buf = aligner.align_simple(Y)

            assert py_score == pytest.approx(buf["score"], abs=0.01)
            assert py_start == buf["start_pos"]
            assert py_cigar == buf["cigar"]

    def test_asymmetric_lengths(self, rng, random_dna_factory, scoring_params):
        """Works when X and Y have different lengths."""
        for nX, nY in [(30, 15), (15, 30), (50, 20), (10, 40)]:
            X = random_dna_factory(nX, rng)
            Y = random_dna_factory(nY, rng)
            EP = build_EP_standard(len(X))

            py_score, py_start, py_cigar = _python_score_and_cigar(
                X, Y, EP, scoring_params
            )

            aligner = _make_buffered_aligner(X, EP, scoring_params)
            buf = aligner.align_simple(Y)

            assert py_score == pytest.approx(buf["score"], abs=0.01)
            assert py_start == buf["start_pos"]
            assert py_cigar == buf["cigar"]


class TestBufferedSingleBlock:
    """Buffered CIGAR path matches pure Python for single-block mode."""

    def test_single_block_score(self, rng, random_dna_factory, scoring_params):
        """Scores match with a flexible block."""
        for _ in range(10):
            nX = rng.integers(15, 40)
            nY = rng.integers(15, 40)
            X = random_dna_factory(nX, rng)
            Y = random_dna_factory(nY, rng)

            s = rng.integers(0, nX // 3)
            e = rng.integers(s + 2, min(s + 10, nX))
            EP = build_EP_single_block(nX, s, e)

            py_score, _, _ = _python_score_and_cigar(X, Y, EP, scoring_params)

            aligner = _make_buffered_aligner(X, EP, scoring_params)
            buf = aligner.align_simple(Y)

            assert py_score == pytest.approx(buf["score"], abs=0.01), (
                f"Score mismatch: s={s}, e={e}, py={py_score}, buf={buf['score']}"
            )

    def test_single_block_cigar(self, rng, random_dna_factory, scoring_params):
        """CIGAR matches with a flexible block."""
        for _ in range(5):
            X = random_dna_factory(25, rng)
            Y = random_dna_factory(25, rng)

            s, e = 5, 15
            EP = build_EP_single_block(25, s, e)

            py_score, py_start, py_cigar = _python_score_and_cigar(
                X, Y, EP, scoring_params
            )

            aligner = _make_buffered_aligner(X, EP, scoring_params)
            buf = aligner.align_simple(Y)

            assert py_score == pytest.approx(buf["score"], abs=0.01)
            assert py_start == buf["start_pos"]
            assert py_cigar == buf["cigar"]

    def test_block_at_end(self, rng, random_dna_factory, scoring_params):
        """Block extends to end of reference (e == n, empty B flank)."""
        for _ in range(5):
            nX = rng.integers(15, 30)
            nY = rng.integers(10, 25)
            X = random_dna_factory(nX, rng)
            Y = random_dna_factory(nY, rng)

            s = rng.integers(2, nX // 2)
            e = nX  # empty B
            EP = build_EP_single_block(nX, s, e)

            py_score, py_start, py_cigar = _python_score_and_cigar(
                X, Y, EP, scoring_params
            )

            aligner = _make_buffered_aligner(X, EP, scoring_params)
            buf = aligner.align_simple(Y)

            assert py_score == pytest.approx(buf["score"], abs=0.01)
            assert py_start == buf["start_pos"]
            assert py_cigar == buf["cigar"]


class TestBufferedSTR:
    """Buffered CIGAR path matches pure Python for STR phase mode."""

    def test_str_phase_score(self, rng, random_dna_factory, scoring_params):
        """Scores match with STR phase-preserving pattern."""
        for _ in range(10):
            A = random_dna_factory(rng.integers(3, 8), rng)
            R = random_dna_factory(rng.integers(2, 5), rng)
            N = rng.integers(3, 8)
            B = random_dna_factory(rng.integers(3, 8), rng)
            X = A + R * N + B

            k = len(R)
            s = len(A)
            e = s + k * N

            Y = random_dna_factory(rng.integers(15, 40), rng)
            EP = build_EP_STR_phase(len(X), s, e, k)

            py_score, _, _ = _python_score_and_cigar(X, Y, EP, scoring_params)

            aligner = _make_buffered_aligner(X, EP, scoring_params)
            buf = aligner.align_simple(Y)

            assert py_score == pytest.approx(buf["score"], abs=0.01), (
                f"Score mismatch: R={R}, N={N}, py={py_score}, buf={buf['score']}"
            )

    def test_str_phase_cigar(self, scoring_params):
        """CIGAR matches for known STR locus configurations."""
        cases = [
            ("GAG", "ACT", 4, "GTCA"),
            ("AA", "TG", 5, "CC"),
            ("TCA", "CAG", 3, "AC"),
        ]
        for A, R, N, B in cases:
            X = A + R * N + B
            k = len(R)
            s = len(A)
            e = s + k * N

            # Align a variant with fewer repeats
            Y = A + R * (N - 2) + B
            EP = build_EP_STR_phase(len(X), s, e, k)

            py_score, py_start, py_cigar = _python_score_and_cigar(
                X, Y, EP, scoring_params
            )

            aligner = _make_buffered_aligner(X, EP, scoring_params)
            buf = aligner.align_simple(Y)

            assert py_score == pytest.approx(buf["score"], abs=0.01)
            assert py_start == buf["start_pos"]
            assert py_cigar == buf["cigar"]

    def test_str_block_at_end(self, scoring_params):
        """STR block extends to end of reference (e == n, empty B).

        Uses realistic reads (actual repeat variants) to test the terminal
        predecessor path. With real repeat content, the terminal cell
        scores are never tied, so CIGAR comparison is meaningful.
        """
        cases = [
            ("ACGT", "CA", 5, 3),   # 5 repeats in ref, 3 in read
            ("ACGT", "CA", 5, 2),   # 5 repeats in ref, 2 in read
            ("GGA", "CAG", 4, 2),   # trinucleotide, 4 vs 2
            ("TC", "AT", 6, 4),     # dinucleotide, 6 vs 4
        ]
        for A, R, N_ref, N_read in cases:
            X = A + R * N_ref
            Y = A + R * N_read

            k = len(R)
            s = len(A)
            e = len(X)
            assert e == len(X)  # confirm e == n

            EP = build_EP_STR_phase(len(X), s, e, k)

            py_score, py_start, py_cigar = _python_score_and_cigar(
                X, Y, EP, scoring_params
            )

            aligner = _make_buffered_aligner(X, EP, scoring_params)
            buf = aligner.align_simple(Y)

            assert py_score == pytest.approx(buf["score"], abs=0.01)
            assert py_start == buf["start_pos"]
            assert py_cigar == buf["cigar"]


class TestBufferedSemiglobal:
    """Buffered CIGAR path matches pure Python for semiglobal mode."""

    def test_semiglobal_score(self, rng, random_dna_factory, scoring_params):
        """Scores match in semiglobal (free_X, free_Y) mode.

        Only checks scores, not CIGARs — semiglobal with random sequences
        produces many tied-score optimal paths with different tie-breaking.
        """
        for _ in range(5):
            nX = rng.integers(20, 40)
            nY = rng.integers(10, 20)
            X = random_dna_factory(nX, rng)
            Y = random_dna_factory(nY, rng)
            EP = build_EP_standard(len(X))

            sg_params = {**scoring_params, "free_X": True, "free_Y": True}
            py_score, _, _ = _python_score_and_cigar(X, Y, EP, sg_params)

            aligner = _make_buffered_aligner(
                X, EP, scoring_params, free_X=True, free_Y=True
            )
            buf = aligner.align_simple(Y)

            assert py_score == pytest.approx(buf["score"], abs=0.01)


class TestBufferedBatch:
    """Batch alignment reuses buffers correctly across reads."""

    def test_batch_matches_individual(self, rng, random_dna_factory, scoring_params):
        """Batch results match individual align_simple calls."""
        X = random_dna_factory(30, rng)
        EP = build_EP_standard(len(X))
        reads = [random_dna_factory(rng.integers(15, 35), rng) for _ in range(20)]

        aligner = _make_buffered_aligner(X, EP, scoring_params)

        batch_results = list(aligner.align_batch_simple(reads))

        for read, batch_res in zip(reads, batch_results):
            py_score, py_start, py_cigar = _python_score_and_cigar(
                X, read, EP, scoring_params
            )
            assert py_score == pytest.approx(batch_res["score"], abs=0.01)
            assert py_start == batch_res["start_pos"]
            assert py_cigar == batch_res["cigar"]

    def test_batch_buffer_growth(self, rng, random_dna_factory, scoring_params):
        """Buffers grow correctly when reads exceed initial max_read_length."""
        X = random_dna_factory(30, rng)
        EP = build_EP_standard(len(X))

        # Start with tiny buffer, then send larger reads
        aligner = _make_buffered_aligner(
            X, EP, scoring_params, max_read_length=10
        )

        reads = [random_dna_factory(length, rng) for length in [5, 15, 30, 50]]

        for read in reads:
            py_score, py_start, py_cigar = _python_score_and_cigar(
                X, read, EP, scoring_params
            )
            buf = aligner.align_simple(read)

            assert py_score == pytest.approx(buf["score"], abs=0.01)
            assert py_start == buf["start_pos"]
            assert py_cigar == buf["cigar"]
