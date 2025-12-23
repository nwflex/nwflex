"""
repeats.py — STR repeat utilities for NW-flex
==============================================

This module provides helpers for working with phased short tandem repeats (STRs):

- Building phased repeat sequences Z* = suf(R,a) · R^M · pre(R,b)
- Enumerating valid phase combinations
- Inferring phase parameters from alignment jumps
- Representing STR loci as structured objects
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

from nwflex.dp_core import RowJump


# ============================================================================
# PHASE REPEAT CONSTRUCTION
# ============================================================================

def phase_repeat(R: str, a: int, b: int, M: int) -> str:
    """
    Construct a phased repeat block Z* = suf(R, a) · R^M · pre(R, b).

    This represents a repeat region that may start or end mid-motif:
    - suf(R, a) is the length-a suffix of R (entry phase)
    - R^M is M complete copies of the motif
    - pre(R, b) is the length-b prefix of R (exit phase)

    Parameters
    ----------
    R : str
        The repeat motif
    a : int
        Entry phase: length of suffix of R to prepend (0 ≤ a < |R|)
    b : int
        Exit phase: length of prefix of R to append (0 ≤ b < |R|)
    M : int
        Number of complete motif copies (M ≥ 0)

    Returns
    -------
    str
        The phased repeat sequence Z*

    Examples
    --------
    >>> phase_repeat("ACT", 2, 1, 3)
    'CTACTACTACTA'

    With R = "ACT":
    - suf(R, 2) = "CT"
    - R^3 = "ACTACTACT"
    - pre(R, 1) = "A"
    - Z* = "CT" + "ACTACTACT" + "A" = "CTACTACTACTA"
    """
    result = ""
    if a > 0:
        result += R[-a:]  # suffix of length a
    result += R * M       # M complete copies
    if b > 0:
        result += R[:b]   # prefix of length b
    return result


# ============================================================================
# PHASE ENUMERATION
# ============================================================================

def valid_phase_combinations(N: int, k: int) -> Iterator[Tuple[int, int, int]]:
    """
    Generate all valid (a, b, M) combinations satisfying the constraint
    M + 1_{a>0} + 1_{b>0} ≤ N.

    This constraint ensures that the phased repeat Z* fits within the
    reference repeat block R^N. Intuitively:
    - Each partial motif (non-zero a or b) "costs" one full copy
    - M complete copies cost M
    - Total must not exceed N

    Parameters
    ----------
    N : int
        Reference repeat count (number of full motif copies in X)
    k : int
        Motif length |R|

    Yields
    ------
    tuple of (int, int, int)
        Valid (a, b, M) combinations where:
        - a ∈ {0, 1, ..., k-1} is the entry phase (suffix length)
        - b ∈ {0, 1, ..., k-1} is the exit phase (prefix length)
        - M ≥ 0 is the number of complete motif copies

    Examples
    --------
    >>> list(valid_phase_combinations(N=2, k=3))
    [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), ...]
    """
    for a in range(k):
        for b in range(k):
            # Compute the "overhead" from partial repeats
            overhead = (1 if a > 0 else 0) + (1 if b > 0 else 0)
            # M can range from 0 to N - overhead
            max_M = N - overhead
            for M in range(max_M + 1):
                yield (a, b, M)


def count_valid_combinations(N: int, k: int) -> int:
    """
    Count the number of valid (a, b, M) combinations.

    Parameters
    ----------
    N : int
        Reference repeat count
    k : int
        Motif length |R|

    Returns
    -------
    int
        Number of valid combinations
    """
    return sum(1 for _ in valid_phase_combinations(N, k))


# ============================================================================
# PHASE INFERENCE FROM ALIGNMENT
# ============================================================================


def infer_abM_from_jumps(
    jumps: List[RowJump],
    s: int,
    e: int,
    k: int
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Infer (a, b, M) phase parameters from entry/exit row jumps.

    The NW-flex traceback records row jumps when the alignment uses an
    extra predecessor (EP) edge. For an STR locus:
    - Entry jump: from leader row s into the block (determines a)
    - Exit jump: from inside block to closer row e+1 (determines b)

    There are four cases:
    - No jumps: a=0, b=0, M=N (exact match)
    - Entry jump only: a is inferred, b=0, M is calculated
    - Exit jump only: a=0, b is inferred, M=N-1
    - Both jumps: a, b, M all inferred from jump positions

    Parameters
    ----------
    jumps : list of RowJump
        Row jumps from the alignment traceback
    s : int
        Leader row index (end of flank A, i.e., |A|)
    e : int
        End of repeat block Z (i.e., |A| + |R|*N)
    k : int
        Motif length |R|

    Returns
    -------
    tuple of (int or None, int or None, int or None)
        Inferred (a, b, M) values

    Notes
    -----
    The phase formulas derive from the structure of the EP pattern:
    - Entry at row i means we skip (i - s - 1) bases of the first motif
    - Exit at row i means we include ((i - s - 1) mod k + 1) bases of the last motif
    """
    N = (e - s) // k  # Reference repeat count

    # Entry: jump from leader s into the block
    entry_row = next(
        (j.to_row for j in jumps if j.from_row == s and j.to_row > s),
        None
    )
    # Exit: jump from inside block to closer e+1
    exit_row = next(
        (j.from_row for j in jumps if j.to_row == e + 1 and j.from_row > s),
        None
    )

    # Case 1: No jumps - exact match (a=0, b=0, M=N)
    if entry_row is None and exit_row is None:
        return 0, 0, N

    # Case 2: Entry jump only (b=0, exit at block end)
    if entry_row is not None and exit_row is None:
        a_inf = (s + 1 - entry_row) % k
        b_inf = 0
        Zstar_len = e - entry_row + 1
        M_inf = (Zstar_len - a_inf - b_inf) // k
        return a_inf, b_inf, M_inf

    # Case 3: Exit jump only (a=0, entry at block start)
    if entry_row is None and exit_row is not None:
        a_inf = 0
        b_inf = (exit_row - e) % k
        M_inf = N - 1
        return a_inf, b_inf, M_inf

    # Case 4: Both jumps
    if exit_row == s:
        return 0, 0, 0

    # Compute phases from row positions
    a_inf = (s + 1 - entry_row) % k
    b_inf = (exit_row - e) % k
    Zstar_len = exit_row - entry_row + 1
    M_inf = (Zstar_len - a_inf - b_inf) // k

    return a_inf, b_inf, M_inf


# ============================================================================
# STR LOCUS REPRESENTATION
# ============================================================================

@dataclass
class STRLocus:
    """
    Representation of an STR locus X = A · R^N · B.

    This dataclass encapsulates the structure of a short tandem repeat locus:
    - A: left flank sequence
    - R: repeat motif
    - N: number of motif copies in reference
    - B: right flank sequence

    It provides convenient properties for the derived quantities (s, e, k, X)
    and methods for building reads with specified phase parameters.

    Attributes
    ----------
    A : str
        Left flank sequence
    R : str
        Repeat motif
    N : int
        Number of motif copies in reference
    B : str
        Right flank sequence

    Examples
    --------
    >>> locus = STRLocus(A="GAG", R="ACT", N=6, B="GTCA")
    >>> locus.X
    'GAGACTACTACTACTACTACTGTCA'
    >>> locus.s, locus.e, locus.k
    (3, 21, 3)
    >>> locus.build_locus_variant(a=2, b=1, M=3)
    'GAGCTACTACTACTGTCA'
    """
    A: str
    R: str
    N: int
    B: str

    @property
    def X(self) -> str:
        """Full reference sequence X = A · R^N · B."""
        return self.A + self.R * self.N + self.B

    @property
    def k(self) -> int:
        """Motif length |R|."""
        return len(self.R)

    @property
    def s(self) -> int:
        """Leader row index (end of flank A, i.e., |A|)."""
        return len(self.A)

    @property
    def e(self) -> int:
        """End of repeat block Z (i.e., |A| + |R|*N)."""
        return len(self.A) + len(self.R) * self.N

    @property
    def n(self) -> int:
        """Length of reference sequence |X|."""
        return len(self.X)

    def build_locus_variant(self, a: int, b: int, M: int) -> str:
        """
        Construct a read Y = A · Z* · B with given phase parameters.

        Parameters
        ----------
        a : int
            Entry phase (suffix length of R)
        b : int
            Exit phase (prefix length of R)
        M : int
            Number of complete motif copies

        Returns
        -------
        str
            The read sequence Y = A · phase_repeat(R, a, b, M) · B
        """
        Zstar = phase_repeat(self.R, a, b, M)
        return self.A + Zstar + self.B

    def valid_combinations(self) -> Iterator[Tuple[int, int, int]]:
        """
        Iterate over all valid (a, b, M) combinations for this locus.

        Yields
        ------
        tuple of (int, int, int)
            Valid (a, b, M) combinations satisfying M + 1_{a>0} + 1_{b>0} ≤ N
        """
        return valid_phase_combinations(self.N, self.k)
    
    def jumps_to_phase(
        self,
        jumps: List[RowJump]
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Get the (a, b, M) phase parameters from entry/exit row jumps.

        Parameters
        ----------
        jumps : list of RowJump
            Row jumps from the alignment traceback

        Returns
        -------
        tuple of (int or None, int or None, int or None)
            Inferred (a, b, M) values, or (None, None, None) if jumps are
            missing or invalid
        """
        return infer_abM_from_jumps(jumps, self.s, self.e, self.k)

    def __repr__(self) -> str:
        return (
            f"STRLocus(A='{self.A}', R='{self.R}', N={self.N}, B='{self.B}')\n"
            f"  X = '{self.X}' (n={self.n})\n"
            f"  s={self.s}, e={self.e}, k={self.k}"
        )


# ============================================================================
# COMPLEX REPEAT STRUCTURES (PLACEHOLDER)
# ============================================================================

@dataclass
class CompoundSTRLocus:
    """
    Representation of a compound STR locus with multiple adjacent repeat blocks.

    A compound locus has the form:
        X = A · R1^N1 · R2^N2 · ... · B

    This is a placeholder for future extension to handle compound and
    interrupted repeats.

    Attributes
    ----------
    A : str
        Left flank sequence
    blocks : list of (str, int)
        List of (motif, count) pairs for each repeat block
    B : str
        Right flank sequence
    """
    A: str
    blocks: List[Tuple[str, int]]  # [(R1, N1), (R2, N2), ...]
    B: str

    @property
    def X(self) -> str:
        """Full reference sequence."""
        middle = "".join(R * N for R, N in self.blocks)
        return self.A + middle + self.B

    @property
    def n(self) -> int:
        """Length of reference sequence."""
        return len(self.X)

    def block_boundaries(self) -> List[Tuple[int, int, int]]:
        """
        Compute (s, e, k) for each repeat block.

        Returns
        -------
        list of (int, int, int)
            For each block i: (s_i, e_i, k_i) where
            - s_i is the start row (end of previous region)
            - e_i is the end row
            - k_i is the motif length
        """
        boundaries = []
        pos = len(self.A)
        for R, N in self.blocks:
            s_i = pos
            e_i = pos + len(R) * N
            k_i = len(R)
            boundaries.append((s_i, e_i, k_i))
            pos = e_i
        return boundaries
