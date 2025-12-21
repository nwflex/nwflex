"""
conftest.py — Shared pytest fixtures for NW-flex test suite

Provides common scoring parameters, alphabet mappings, and random
number generators used across all test modules.
"""

import pytest
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Default scoring fixtures (match validation.py defaults)
# ---------------------------------------------------------------------------

@pytest.fixture
def default_bases() -> NDArray:
    """Default DNA alphabet."""
    return np.array(["A", "C", "G", "T"])


@pytest.fixture
def alphabet_to_index(default_bases) -> dict:
    """Mapping from base to matrix index."""
    return {b: i for i, b in enumerate(default_bases)}


@pytest.fixture
def score_matrix() -> NDArray[np.floating]:
    """Simple match/mismatch matrix: +5 on diagonal, -5 off-diagonal."""
    mat = np.full((4, 4), -5.0, dtype=float)
    np.fill_diagonal(mat, 5.0)
    return mat


@pytest.fixture
def gap_open() -> float:
    """Gap opening penalty."""
    return -20.0


@pytest.fixture
def gap_extend() -> float:
    """Gap extension penalty."""
    return -1.0


@pytest.fixture
def scoring_params(score_matrix, gap_open, gap_extend, alphabet_to_index):
    """Bundle all scoring parameters into a dict for easy unpacking."""
    return {
        "score_matrix": score_matrix,
        "gap_open": gap_open,
        "gap_extend": gap_extend,
        "alphabet_to_index": alphabet_to_index,
    }


# ---------------------------------------------------------------------------
# Random number generator fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(888)


@pytest.fixture
def rng_alt():
    """Alternative seed for diversity in randomized tests."""
    return np.random.default_rng(123)


# ---------------------------------------------------------------------------
# Sequence generation helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def random_dna_factory(default_bases):
    """Factory fixture returning a function to generate random DNA strings."""
    def _random_dna(length: int, rng: np.random.Generator) -> str:
        return "".join(rng.choice(default_bases, size=length))
    return _random_dna
