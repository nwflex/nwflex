# NW-flex Test Suite

This document describes the test suite for the NW-flex package.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_standard.py -v
```

## Test Modules Overview

| Module | Purpose |
|--------|---------|
| `test_standard.py` | Standard NW/Gotoh alignment (EP = ∅); includes alignment validity and scoring-variant tests|
| `test_single_block.py` | Single-block flex alignment (X = A·Z·B); compares to naive substring enumeration|
| `test_str.py` | STR repeat utilities + alignment tests. Contains enumeration of phase combinations, inference checks, and flex-vs-naive validations |
| `test_cython.py` | Cython vs Python equivalence and performance checks|

---

## `test_standard.py` — Standard NW/Gotoh alignment (EP = ∅)

Verifies that NW-flex with no extra predecessors matches an independent  
Needleman–Wunsch / Gotoh baseline.

**Tests included:**
- Fixed-case and randomized score comparisons
- Alignment validity checks that verify:
  - Equal aligned lengths
  - No double gaps
  - The reported score equals the score recomputed from the alignment strings
- Tests across multiple gap-penalty variants

---

## `test_single_block.py` — Single-Block Flex Alignment

Validates the core NW-flex guarantee: single-block flex score equals the naive maximum over all Z* substrings:

```
S_flex(X, Y) = max_{Z* ⊆ Z} NWG(A·Z*·B, Y)
```

**Tests included:**
- Fixed cases: Hand-crafted A·Z·B examples covering common substring-selection scenarios
- Boundary cases: Block at start/end/middle and small-block edge cases
- Randomized cases: Random A·Z·B lengths and sequences to cover diverse inputs
- Mutated reads: Substitutions and indels; verify NW-flex matches the naive baseline
- Alignment validity: Check aligned-length consistency, absence of double gaps, and agreement between reported and recomputed/naive scores

---

## `test_str.py` — STR repeat utilities and STR alignment

**Tests included:**
- `phase_repeat`, `valid_phase_combinations`, `count_valid_combinations`,  
  and other repeat utilities (unit tests)
- `STRLocus` and `CompoundSTRLocus` behaviors and helpers
- STR alignment validation:
  - Enumerates all valid `(a, b, M)` phase combinations for an `STRLocus`
  - Verifies NW-flex (phase-preserving EP) achieves the expected
    perfect-match score for perfect-locus reads
  - Checks that `(a, b, M)` can be inferred from row-jump traceback
- A naive STR baseline (enumeration over valid phase combinations) is used  
  to validate `align_STR_block` on random and constructed reads
---

## `test_cython.py` — Cython vs Python Equivalence

Ensures the Cython-accelerated implementation (`run_flex_dp_fast`) produces identical results to pure Python (`run_flex_dp`).

### TestCythonVsPython

| Test | Purpose |
|------|---------|
| `test_standard_mode_score` | Scores match (standard mode) |
| `test_standard_mode_alignment` | Aligned strings match |
| `test_single_block_mode_score` | Scores match (single-block) |
| `test_single_block_mode_alignment` | Aligned strings match |

### TestCythonDPTables

| Test | Purpose |
|------|---------|
| `test_dp_tables_match` | Full DP matrices (Yg, M, Xg) match |
| `test_dp_tables_single_block` | DP tables match in flex mode |

### TestCythonJumps

| Test | Purpose |
|------|---------|
| `test_jumps_match` | Row jumps detected identically |

---

## Fixtures (conftest.py)

### Scoring Parameters

| Fixture | Value | Description |
|---------|-------|-------------|
| `score_matrix` | +5/-5 | Match +5, mismatch -5 |
| `gap_open` | -20 | Gap opening penalty |
| `gap_extend` | -1 | Gap extension penalty |
| `alphabet_to_index` | A=0, C=1, G=2, T=3 | Base to index mapping |
| `scoring_params` | dict | Bundle of all above |

### Random Generators

| Fixture | Seed | Purpose |
|---------|------|---------|
| `rng` | 888 | Primary random generator |
| `rng_alt` | 123 | Alternative for diversity |

### Helpers

| Fixture | Purpose |
|---------|---------|
| `random_dna_factory` | Generate random DNA strings |

---