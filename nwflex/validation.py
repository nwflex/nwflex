"""
validation.py — independent baselines and regression tests for NW-flex

This module provides an independent implementation of standard
Needleman-Wunsch/Gotoh (nwg_global), a naive flex baseline
(sflex_naive), and small helpers for randomized regression tests.

The goals are:

  1. Verify that the NW-flex implementation with E(i) = empty
     matches a plain NW/Gotoh implementation on (X, Y).

  2. Verify that NW-flex with a single-block EP pattern returns
     the same flex score as the naive definition

         S_flex(X,Y) = max_{Z* subset Z} NWG(A·Z*·B, Y)

     obtained by enumerating all substrings Z* of the block Z
     and calling nwg_global on each effective reference A·Z*·B.

These checks are used in the Validation section of the paper and
in the accompanying notebooks. This module is independent of dp_core: 
nwg_global reimplements the Gotoh recurrences directly 
so that bugs in dp_core cannot mask each other during testing.
"""


from typing import Mapping, Optional, Tuple
import timeit

import numpy as np
from numpy.typing import NDArray

from .ep_patterns import build_EP_standard, build_EP_single_block
from .aligners import align_standard, align_single_block
from .dp_core import FlexInput, run_flex_dp
from .fast import run_flex_dp_fast
from . import default

# ---------------------------------------------------------------------------
# Default scoring and alphabet for tests and demos
# ---------------------------------------------------------------------------
# Use defaults from default.py
DEFAULT_BASES = default.BASES
DEFAULT_ALPHABET_TO_INDEX = default.ALPHABET_TO_INDEX
DEFAULT_SCORE_MATRIX = default.SCORE_MATRIX
DEFAULT_GAP_OPEN = default.GAP_OPEN
DEFAULT_GAP_EXTEND = default.GAP_EXTEND

def nwg_global(
    X: str,
    Y: str,
    score_matrix: NDArray[np.floating] = DEFAULT_SCORE_MATRIX,
    gap_open: float = DEFAULT_GAP_OPEN,
    gap_extend: float = DEFAULT_GAP_EXTEND,
    alphabet_to_index: Mapping[str, int] = DEFAULT_ALPHABET_TO_INDEX,
) -> float:
    """
    Standard global Needleman-Wunsch/Gotoh score NWG(X, Y).
    """
    n, m = len(X), len(Y)
    gs, ge = gap_open, gap_extend

    # three Gotoh layers
    Yg = np.full((n + 1, m + 1), -np.inf, dtype=float)
    M  = np.full((n + 1, m + 1), -np.inf, dtype=float)
    Xg = np.full((n + 1, m + 1), -np.inf, dtype=float)

    M[0, 0] = 0.0

    # first row: gaps in X
    for j in range(1, m + 1):
        Yg[0, j] = gs + (j - 1) * ge

    # first column: gaps in Y
    for i in range(1, n + 1):
        Xg[i, 0] = gs + (i - 1) * ge

    # main DP
    for i in range(1, n + 1):
        xi = alphabet_to_index[X[i - 1]]
        for j in range(1, m + 1):
            yj = alphabet_to_index[Y[j - 1]]
            score = float(score_matrix[xi, yj])

            # Yg: gap in X (horizontal)
            Yg[i, j] = max(
                Yg[i, j - 1] + ge,
                M [i, j - 1] + gs,
                Xg[i, j - 1] + gs,
            )

            # M: match/mismatch (diagonal)
            M[i, j] = max(
                Yg[i - 1, j - 1],
                M [i - 1, j - 1],
                Xg[i - 1, j - 1],
            ) + score

            # Xg: gap in Y (vertical)
            Xg[i, j] = max(
                Yg[i - 1, j] + gs,
                M [i - 1, j] + gs,
                Xg[i - 1, j] + ge,
            )
    return max(Yg[n, m], M[n, m], Xg[n, m])


def sflex_naive(
    X: str,
    Y: str,
    s: int,
    e: int,
    score_matrix: NDArray[np.floating] = DEFAULT_SCORE_MATRIX,
    gap_open: float = DEFAULT_GAP_OPEN,
    gap_extend: float = DEFAULT_GAP_EXTEND,
    alphabet_to_index: Mapping[str, int] = DEFAULT_ALPHABET_TO_INDEX,
) -> float:
    """
    Naive flex baseline for a single flexible block X = A·Z·B.

    Here we assume:
        A = X[0:s]
        Z = X[s:e]
        B = X[e:]

    and define the flex score as:

        S_flex(X, Y) = max_{Z* subset of Z} NWG(A + Z* + B, Y),

    where Z* ranges over all contiguous substrings of Z, including
    the empty substring.

    This function enumerates all substrings Z* = X[i:j] with
    s <= i <= j <= e and calls nwg_global on each effective
    reference A + Z* + B, returning the maximum score.
    """
    n = len(X)
    if not (0 <= s <= e <= n):
        raise ValueError(f"Invalid block indices: need 0 <= s <= e <= n, got s={s}, e={e}, n={n}")

    best = -np.inf
    A = X[:s]
    B = X[e:]

    for i in range(s, e + 1):
        for j in range(i, e + 1):
            Z_star = X[i:j]           # contiguous substring of Z
            X_prime = A + Z_star + B  # effective reference
            score = nwg_global(
                X_prime,
                Y,
                score_matrix,
                gap_open,
                gap_extend,
                alphabet_to_index,
            )
            if score > best:
                best = score

    return best


def check_standard_vs_nwg(
    X: str,
    Y: str,
    score_matrix: NDArray[np.floating] = DEFAULT_SCORE_MATRIX,
    gap_open: float = DEFAULT_GAP_OPEN,
    gap_extend: float = DEFAULT_GAP_EXTEND,
    alphabet_to_index: Mapping[str, int] = DEFAULT_ALPHABET_TO_INDEX,
) -> Tuple[float, float]:
    """
    Compare NW-flex in standard mode to the independent NWG baseline.

    Returns
    -------
    flex_score : float
        Score from align_standard (NW-flex with E(i) = empty).
    nwg_score : float
        Score from nwg_global.

    The caller can assert flex_score == nwg_score (or close) in tests.
    """
    res = align_standard(
        X=X,
        Y=Y,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        alphabet_to_index=alphabet_to_index,
        return_data=False,
    )
    flex_score = res.score

    nwg_score = nwg_global(
        X,
        Y,
        score_matrix,
        gap_open,
        gap_extend,
        alphabet_to_index,
    )
    return flex_score, nwg_score


def check_single_block_case(
    X: str,
    Y: str,
    s: int,
    e: int,
    score_matrix: NDArray[np.floating] = DEFAULT_SCORE_MATRIX,
    gap_open: float = DEFAULT_GAP_OPEN,
    gap_extend: float = DEFAULT_GAP_EXTEND,
    alphabet_to_index: Mapping[str, int] = DEFAULT_ALPHABET_TO_INDEX,
) -> Tuple[float, float]:
    """
    Compare NW-flex single-block mode to the naive flex baseline
    for one specific X, Y, and block (s, e).

    Returns
    -------
    flex_score : float
        Score from align_single_block (NW-flex).
    naive_score : float
        Naive S_flex(X, Y) from sflex_naive.

    The caller can assert flex_score == naive_score in tests.
    """
    # NW-flex single-block
    res = align_single_block(
        X=X,
        Y=Y,
        s=s,
        e=e,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        alphabet_to_index=alphabet_to_index,
        return_data=False,
    )
    flex_score = res.score

    # Naive flex baseline
    naive_score = sflex_naive(
        X=X,
        Y=Y,
        s=s,
        e=e,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        alphabet_to_index=alphabet_to_index,
    )

    return flex_score, naive_score

def check_standard_case(
    X: str,
    Y: str,
    score_matrix: NDArray[np.floating] = DEFAULT_SCORE_MATRIX,
    gap_open: float = DEFAULT_GAP_OPEN,
    gap_extend: float = DEFAULT_GAP_EXTEND,
    alphabet_to_index: Mapping[str, int] = DEFAULT_ALPHABET_TO_INDEX,
):
    """
    Convenience wrapper to compare standard NW-flex vs nwg_global
    using default scoring unless overridden.
    """
    return check_standard_vs_nwg(
        X=X,
        Y=Y,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        alphabet_to_index=alphabet_to_index,
    )


def check_AZB_case(
    A: str,
    Z: str,
    B: str,
    Y: str,
    score_matrix: NDArray[np.floating] = DEFAULT_SCORE_MATRIX,
    gap_open: float = DEFAULT_GAP_OPEN,
    gap_extend: float = DEFAULT_GAP_EXTEND,
    alphabet_to_index: Mapping[str, int] = DEFAULT_ALPHABET_TO_INDEX,
):
    """
    Convenience wrapper to compare NW-flex single-block vs naive S_flex
    for sequences specified as (A, Z, B, Y).

    Constructs X = A + Z + B and block indices
        s = len(A), e = len(A) + len(Z),
    then calls check_single_block_case with the given or default scoring.
    """
    X = A + Z + B
    s = len(A)
    e = len(A) + len(Z)
    return check_single_block_case(
        X=X,
        Y=Y,
        s=s,
        e=e,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        alphabet_to_index=alphabet_to_index,
    )


# ---------------------------------------------------------------------------
# Alignment validity helper
# ---------------------------------------------------------------------------

def check_alignment_validity(result, 
                             score_matrix, 
                             gap_open, 
                             gap_extend, 
                             alphabet_to_index) -> Tuple[bool, str]:
    """
    Check that an AlignmentResult has valid alignment strings.
    
    Verifies that X_aln and Y_aln:
    - Have the same length
    - Don't have simultaneous gaps at any position
    - The reported alignment score matches the score recomputed from the alignment strings,
       using the provided scoring parameters.
    
    Parameters
    ----------
    result : AlignmentResult
        Output from align_single_block or align_standard.
        
    Returns
    -------
    valid : bool
        True if the alignment strings pass all checks.
    message : str
        Description of what was checked or what failed.
    """
    X_aln = result.X_aln
    Y_aln = result.Y_aln

    computed_score = 0.0
    in_x_gap = False
    in_y_gap = False
    for i, (x, y) in enumerate(zip(X_aln, Y_aln)):
        if x == "-" and y == "-":
            return False, f"Double gap found in alignment at position {i}"
        elif x == "-":
            computed_score += gap_extend if in_x_gap else gap_open
            in_x_gap = True
            in_y_gap = False
        elif y == "-":
            computed_score += gap_extend if in_y_gap else gap_open
            in_y_gap = True
            in_x_gap = False
        else:
            # Match/mismatch
            xi = alphabet_to_index[x]
            yj = alphabet_to_index[y]
            computed_score += score_matrix[xi, yj]
            in_x_gap = False
            in_y_gap = False
    # Check score matches
    if not np.isclose(computed_score, result.score):
        return False, f"Score mismatch: computed {computed_score}, reported {result.score}"

    # Check same length
    if len(X_aln) != len(Y_aln):
        return False, f"Length mismatch: X_aln={len(X_aln)}, Y_aln={len(Y_aln)}"
    
    return True, f"Valid alignment of length {len(X_aln)}"


# ---------------------------------------------------------------------------
# Mutation helpers for testing
# ---------------------------------------------------------------------------

def mutate_sequence(
    seq: str,
    rng,
    sub_rate: float = 0.1,
    indel_rate: float = 0.05,
) -> str:
    """
    Apply random mutations to a DNA sequence.
    
    Parameters
    ----------
    seq : str
        Input DNA sequence.
    rng : numpy random generator
        Random number generator (e.g., np.random.default_rng(seed)).
    sub_rate : float
        Per-base substitution probability (default 0.1).
    indel_rate : float
        Per-base insertion/deletion probability (default 0.05).
    
    Returns
    -------
    str
        Mutated sequence.
    """
    bases = default.BASES
    result = []
    
    for base in seq:
        # Deletion
        if rng.random() < indel_rate:
            continue  # skip this base
        
        # Substitution
        if rng.random() < sub_rate:
            other_bases = [b for b in bases if b != base]
            base = rng.choice(other_bases)
        
        result.append(base)
        
        # Insertion (after current base)
        if rng.random() < indel_rate:
            result.append(rng.choice(bases))
    
    return "".join(result)


def run_mutated_AZB_tests(
    base_cases: list,
    num_mutations: int = 5,
    sub_rate: float = 0.1,
    indel_rate: float = 0.05,
    seed: int = 42,
    score_matrix: NDArray[np.floating] = DEFAULT_SCORE_MATRIX,
    gap_open: float = DEFAULT_GAP_OPEN,
    gap_extend: float = DEFAULT_GAP_EXTEND,
    alphabet_to_index: Mapping[str, int] = DEFAULT_ALPHABET_TO_INDEX,
) -> list:
    """
    Run validation tests on mutated versions of base cases.
    
    For each base case, generate multiple mutated Y sequences and verify
    that NW-flex matches the naive baseline on each.
    
    Parameters
    ----------
    base_cases : list of tuples
        Each tuple is (A, Z, B, Y_base, description).
    num_mutations : int
        Number of mutated Y sequences to generate per base case.
    sub_rate : float
        Per-base substitution probability.
    indel_rate : float
        Per-base insertion/deletion probability.
    seed : int
        Random seed for reproducibility.
    score_matrix, gap_open, gap_extend, alphabet_to_index : scoring params
        See other functions for defaults.
    
    Returns
    -------
    results : list of dicts
        Each dict contains: description, A, Z, B, Y, Y_mutated, flex, naive, match.
    """
    rng = np.random.default_rng(seed)
    results = []
    
    for A, Z, B, Y_base, desc in base_cases:
        for _ in range(num_mutations):
            # Mutate Y
            Y_mut = mutate_sequence(Y_base, rng, sub_rate, indel_rate)
            if len(Y_mut) == 0:
                Y_mut = Y_base[0]  # Ensure non-empty
            
            flex_score, naive_score = check_AZB_case(
                A, Z, B, Y_mut,
                score_matrix=score_matrix,
                gap_open=gap_open,
                gap_extend=gap_extend,
                alphabet_to_index=alphabet_to_index,
            )
            match = flex_score == naive_score
            
            results.append({
                "description": desc,
                "A": A,
                "Z": Z,
                "B": B,
                "Y": Y_base,
                "Y_mutated": Y_mut,
                "flex": flex_score,
                "naive": naive_score,
                "match": match,
            })
    
    return results


# ---------------------------------------------------------------------------
# Random sequence generation and benchmarking
# ---------------------------------------------------------------------------

def random_dna(length: int, rng: np.random.Generator) -> str:
    """
    Generate a random DNA string of a given length.
    
    Parameters
    ----------
    length : int
        Length of the DNA string to generate.
    rng : np.random.Generator
        NumPy random generator instance.
    
    Returns
    -------
    str
        Random DNA string of the specified length.
    """
    return "".join(rng.choice(DEFAULT_BASES, size=length))


def benchmark_python_vs_cython(
    nX: int,
    nY: int,
    rng: np.random.Generator,
    mode: str = "standard",
    block_fraction: float = 0.4,
    n_samples: int = 3,
    cython_multiplier: int = 100,
    score_matrix: Optional[NDArray[np.floating]] = None,
    gap_open: Optional[float] = None,
    gap_extend: Optional[float] = None,
    alphabet_to_index: Optional[Mapping[str, int]] = None,
) -> Tuple[float, float]:
    """
    Benchmark Python vs Cython NW-flex using timeit for accurate averaging.
    
    Parameters
    ----------
    nX, nY : int
        Sequence lengths.
    rng : np.random.Generator
        Random number generator.
    mode : str
        'standard' or 'single_block'.
    block_fraction : float
        Fraction of X to use as flexible block (for single_block mode).
    n_samples : int
        Number of timing samples for Python.
    cython_multiplier : int
        Run Cython this many more times than Python for accurate sub-ms timing.
    score_matrix : ndarray, optional
        Substitution matrix. Defaults to DEFAULT_SCORE_MATRIX.
    gap_open : float, optional
        Gap opening penalty. Defaults to DEFAULT_GAP_OPEN.
    gap_extend : float, optional
        Gap extension penalty. Defaults to DEFAULT_GAP_EXTEND.
    alphabet_to_index : dict, optional
        Mapping from characters to indices. Defaults to DEFAULT_ALPHABET_TO_INDEX.
    
    Returns
    -------
    (py_avg, cy_avg) : Tuple[float, float]
        Average time per call in seconds for Python and Cython implementations.
    """
    # Use defaults if not provided
    if score_matrix is None:
        score_matrix = DEFAULT_SCORE_MATRIX
    if gap_open is None:
        gap_open = DEFAULT_GAP_OPEN
    if gap_extend is None:
        gap_extend = DEFAULT_GAP_EXTEND
    if alphabet_to_index is None:
        alphabet_to_index = DEFAULT_ALPHABET_TO_INDEX
    
    X = random_dna(nX, rng)
    Y = random_dna(nY, rng)

    if mode == "standard":
        EP = build_EP_standard(nX)
    elif mode == "single_block":
        block_len = max(1, int(block_fraction * nX))
        s = (nX - block_len) // 2
        e = s + block_len
        EP = build_EP_single_block(nX, s, e)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    cfg = FlexInput(
        X=X,
        Y=Y,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        extra_predecessors=EP,
        alphabet_to_index=alphabet_to_index,
    )

    # Warm-up calls to avoid one-time overhead in timing
    run_flex_dp(cfg, return_data=False)
    run_flex_dp_fast(cfg, return_data=False)

    # Time Python
    timer_py = timeit.Timer(lambda: run_flex_dp(cfg, return_data=False))
    py_total = timer_py.timeit(number=n_samples)
    py_avg = py_total / n_samples

    # Time Cython (more iterations for sub-ms accuracy)
    timer_cy = timeit.Timer(lambda: run_flex_dp_fast(cfg, return_data=False))
    cy_total = timer_cy.timeit(number=n_samples * cython_multiplier)
    cy_avg = cy_total / (n_samples * cython_multiplier)

    return py_avg, cy_avg
