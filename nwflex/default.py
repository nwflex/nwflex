"""
default.py — Default parameters for NW-flex

Provides the DNA alphabet and a registry of named scoring presets
(`PRESETS`) used throughout examples, notebooks and tests. The
`nwflex_default` preset (+5/-5 match/mismatch, affine -20/-1 gaps)
is the library default; additional presets (e.g. `bwa_mem`) let
callers select alternate scoring schemes for cross-tool comparison.

Use `get_preset(name)` for direct lookup, or `make_dna_score_matrix`
to build ad-hoc matrices.
"""

from dataclasses import dataclass

import numpy as np

# DNA alphabet
BASES = np.array(["A", "C", "G", "T"])
ALPHABET_TO_INDEX = {b: i for i, b in enumerate(BASES)}

# TODO (future): add IUPAC support alongside the strict 4-letter alphabet,
# e.g. BASES_IUPAC = np.array(["A", "C", "G", "T", "N"]), paired with an
# IUPAC-aware score-matrix factory and dedicated presets. Each ScoringPreset
# already carries its own `alphabet_to_index`, so this can be additive.


def make_dna_score_matrix(
    match: float,
    mismatch: float,
    alphabet: np.ndarray = BASES,
) -> np.ndarray:
    """Square match/mismatch matrix: `match` on the diagonal, `mismatch` elsewhere."""
    n = len(alphabet)
    M = np.full((n, n), mismatch, dtype=float)
    np.fill_diagonal(M, match)
    return M


@dataclass(frozen=True)
class ScoringPreset:
    """A named bundle of alignment scoring parameters."""
    name: str
    score_matrix: np.ndarray
    gap_open: float
    gap_extend: float
    alphabet_to_index: dict
    notes: str = ""


PRESETS: dict[str, ScoringPreset] = {
    "nwflex_default": ScoringPreset(
        name="nwflex_default",
        score_matrix=make_dna_score_matrix(5.0, -5.0),
        gap_open=-20.0,
        gap_extend=-1.0,
        alphabet_to_index=ALPHABET_TO_INDEX,
        notes="NW-flex default: +5/-5 match/mismatch, affine -20/-1 gaps.",
    ),
    "bwa_mem": ScoringPreset(
        name="bwa_mem",
        score_matrix=make_dna_score_matrix(1.0, -4.0),
        gap_open=-6.0,
        gap_extend=-1.0,
        alphabet_to_index=ALPHABET_TO_INDEX,
        notes="BWA-MEM defaults (-A 1 -B 4 -O 6 -E 1).",
    ),
}


def get_preset(name: str) -> ScoringPreset:
    """Look up a scoring preset by name. Unknown names raise KeyError listing the available presets."""
    if name not in PRESETS:
        raise KeyError(
            f"Unknown scoring preset {name!r}. Available: {sorted(PRESETS)}"
        )
    return PRESETS[name]


# Module-level constants alias the `nwflex_default` preset, so the registry
# is the single source of truth. Existing callers that import these names
# directly continue to work unchanged.
SCORE_MATRIX = PRESETS["nwflex_default"].score_matrix
GAP_OPEN     = PRESETS["nwflex_default"].gap_open
GAP_EXTEND   = PRESETS["nwflex_default"].gap_extend


def align_params(*, preset: str = "nwflex_default", semiglobal: bool = False) -> dict:
    """
    Bundle scoring parameters for the named preset into a dict for easy unpacking.

    Parameters:
        preset (str): Name of the scoring preset to use (see `PRESETS`).
        semiglobal (bool): If True, set `free_X`/`free_Y` for semiglobal alignment.

    Usage:
        result = align_standard(X, Y, **align_params(preset="bwa_mem", semiglobal=True))
    """
    p = get_preset(preset)
    return {
        "score_matrix": p.score_matrix,
        "gap_open": p.gap_open,
        "gap_extend": p.gap_extend,
        "alphabet_to_index": p.alphabet_to_index,
        "free_X": semiglobal,
        "free_Y": semiglobal,
    }


def get_default_scoring(preset: str = "nwflex_default"):
    """
    Return the scoring components of the named preset as a tuple:

        (score_matrix, gap_open, gap_extend, alphabet_to_index)
    """
    p = get_preset(preset)
    return (p.score_matrix, p.gap_open, p.gap_extend, p.alphabet_to_index)
