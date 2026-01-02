"""
conftest.py — Shared pytest fixtures for NW-flex test suite

Provides common scoring parameters, alphabet mappings, and random
number generators used across all test modules.
"""

import pytest
import numpy as np
from numpy.typing import NDArray
from nwflex import default


# ---------------------------------------------------------------------------
# Default scoring fixtures (match validation.py defaults)
# ---------------------------------------------------------------------------

@pytest.fixture
def default_bases() -> NDArray:
    """Default DNA alphabet."""
    return default.BASES


@pytest.fixture
def alphabet_to_index(default_bases) -> dict:
    """Mapping from base to matrix index."""
    return default.ALPHABET_TO_INDEX


@pytest.fixture
def score_matrix() -> NDArray[np.floating]:
    """Simple match/mismatch matrix: +5 on diagonal, -5 off-diagonal."""
    return default.SCORE_MATRIX


@pytest.fixture
def gap_open() -> float:
    """Gap opening penalty."""
    return default.GAP_OPEN


@pytest.fixture
def gap_extend() -> float:
    """Gap extension penalty."""
    return default.GAP_EXTEND


@pytest.fixture
def scoring_params(score_matrix, gap_open, gap_extend, alphabet_to_index):
    """Bundle all scoring parameters into a dict for easy unpacking."""
    return {
        "score_matrix": score_matrix,
        "gap_open": gap_open,
        "gap_extend": gap_extend,
        "alphabet_to_index": alphabet_to_index,
    }

@pytest.fixture
def align_semiglobal():
    """Return default parameters for semiglobal alignment."""
    return default.align_params(semiglobal=True)

@pytest.fixture
def align_global():
    """Return default parameters for global alignment."""
    return default.align_params(semiglobal=False)


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
