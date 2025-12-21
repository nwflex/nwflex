# NW-flex Test Suite

This document describes the test suite for the NW-flex package.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=nwflex --cov-report=term-missing

# Run specific test module
pytest tests/test_standard.py -v
```

## Test Modules Overview

| Module | Tests | Purpose |
|--------|-------|---------|
| `test_standard.py` | 18 | Standard NW/Gotoh alignment (EP = ∅) |
| `test_single_block.py` | 17 | Single-block flex alignment (X = A·Z·B) |
| `test_cython.py` | 7 | Cython vs Python equivalence |
| `test_str_phase.py` | 14 | STR phase-aware alignment |
| `test_repeats.py` | 38 | STR repeat utilities |

---

## `test_standard.py` — Standard NW/Gotoh Alignment

Verifies that NW-flex with empty EP (no extra predecessors) matches the independent Needleman-Wunsch/Gotoh baseline implementation in `validation.py`.

### TestStandardVsNWG

| Test | Purpose |
|------|---------|
| `test_fixed_cases[identical]` | Perfect match (no gaps/mismatches) |
| `test_fixed_cases[all mismatches]` | Complete mismatch |
| `test_fixed_cases[X longer]` | X longer than Y |
| `test_fixed_cases[Y longer]` | Y longer than X |
| `test_fixed_cases[single base match]` | Minimal case: 1bp match |
| `test_fixed_cases[single base mismatch]` | Minimal case: 1bp mismatch |
| `test_fixed_cases[deletion in Y]` | Gap in Y |
| `test_fixed_cases[insertion in Y]` | Extra base in Y |
| `test_fixed_cases[classic example]` | GATTACA vs GCATGCT |
| `test_fixed_cases[shifted pattern]` | Pattern at different positions |
| `test_random_sequences` | 20 random same-length pairs |
| `test_random_asymmetric` | 20 random different-length pairs |

### TestAlignmentValidity

| Test | Purpose |
|------|---------|
| `test_alignment_lengths_match` | X_aln and Y_aln same length |
| `test_no_double_gaps` | No gap-gap columns |
| `test_ungapped_recovers_original` | Removing gaps recovers input |
| `test_random_alignment_validity` | 10 random validity checks |

---

## `test_single_block.py` — Single-Block Flex Alignment

Validates the core NW-flex guarantee: single-block flex score equals the naive maximum over all Z* substrings:

```
S_flex(X, Y) = max_{Z* ⊆ Z} NWG(A·Z*·B, Y)
```

### TestSingleBlockVsNaive

| Test | Purpose |
|------|---------|
| `test_fixed_cases[simple block]` | Basic partial match |
| `test_fixed_cases[skip entire block]` | Optimal Z* = ∅ |
| `test_fixed_cases[keep entire block]` | Optimal Z* = Z |
| `test_fixed_cases[partial block]` | Optimal Z* is substring |
| `test_fixed_cases[minimal match]` | Single base block, matched |
| `test_fixed_cases[minimal skip]` | Single base block, skipped |
| `test_fixed_cases[skip long block]` | Long block skipped |
| `test_fixed_cases[keep long block]` | Long block kept |
| `test_fixed_cases[repeat block]` | Block with repeat pattern |
| `test_fixed_cases[partial repeat]` | Repeat block partial match |
| `test_random_sequences` | 20 random A·Z·B configurations |

### TestBlockBoundaries

| Test | Purpose |
|------|---------|
| `test_block_at_start` | A is empty (s=0) |
| `test_block_at_end` | B is empty (e=n), uses terminal EP |
| `test_block_in_middle` | Standard A·Z·B case |
| `test_small_block` | Minimal 2-base block |

### TestSingleBlockAlignmentValidity

| Test | Purpose |
|------|---------|
| `test_alignment_strings_valid` | Fixed case validity |
| `test_random_alignment_validity` | 10 random validity checks |

### TestMutatedSequences

| Test | Purpose |
|------|---------|
| `test_mutated_cases` | Flex matches naive on mutated Y |

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

## `test_str_phase.py` — STR Phase-Aware Alignment

Tests STR-specific alignment using `align_STR_block` with motif phase constraints.

### TestSTRPhaseBasics

| Test | Purpose |
|------|---------|
| `test_perfect_repeat_alignment` | Exact repeat match |
| `test_repeat_expansion` | Read has more repeats |
| `test_repeat_contraction` | Read has fewer repeats |
| `test_flanking_regions` | Flanks align correctly |

### TestPhaseRepeat

| Test | Purpose |
|------|---------|
| `test_phase_zero` | Phase 0 (aligned start) |
| `test_phase_with_suffix` | Phase with trailing partial |
| `test_phase_with_prefix` | Phase with leading partial |
| `test_phase_both_partial` | Both prefix and suffix partial |

### TestSTRPhaseVsStandard

| Test | Purpose |
|------|---------|
| `test_no_worse_than_standard` | STR flex ≥ standard score |
| `test_benefits_from_flexibility` | STR flex > standard on contractions |

### TestSTRPhaseAlignmentValidity

| Test | Purpose |
|------|---------|
| `test_various_repeats[AT]` | Dinucleotide validity |
| `test_various_repeats[CAG]` | Trinucleotide validity |
| `test_various_repeats[CAGCAG]` | Hexanucleotide validity |
| `test_various_repeats[AAAG]` | Tetranucleotide validity |
| `test_random_str_alignment` | 10 random STR alignments |

---

## `test_repeats.py` — STR Repeat Utilities

Tests utility functions in `nwflex.repeats` for working with phased short tandem repeats.

### TestPhaseRepeat

| Test | Purpose |
|------|---------|
| `test_phase_repeat_cases` | Parametrized: 18 cases of Z* construction |
| `test_phase_repeat_length` | Output length = a + k×M + b |

### TestValidPhaseCombinations

| Test | Purpose |
|------|---------|
| `test_count_matches_generator` | count_valid_combinations matches generator |
| `test_constraint_satisfied` | M + 1_{a>0} + 1_{b>0} ≤ N |
| `test_a_b_ranges` | a, b ∈ [0, k-1], M ≥ 0 |
| `test_specific_counts` | Known counts for small N, k |
| `test_no_duplicates` | No duplicate (a, b, M) tuples |

### TestInferABMFromJumps

| Test | Purpose |
|------|---------|
| `test_no_jumps_returns_none` | Empty jumps → (None, None, None) |
| `test_missing_entry_jump` | No entry jump → None |
| `test_missing_exit_jump` | No exit jump → None |
| `test_valid_jumps_infer_phase` | Valid jumps produce integers |

### TestSTRLocus

| Test | Purpose |
|------|---------|
| `test_basic_properties` | X, s, e, k, n properties |
| `test_empty_flanks` | Empty A and/or B |
| `test_locus_variant` | build_locus_variant(a, b, M) method |
| `test_valid_combinations` | valid_combinations iterator |
| `test_repr` | String representation |

### TestCompoundSTRLocus

| Test | Purpose |
|------|---------|
| `test_basic_properties` | X and n for compound locus |
| `test_single_block` | Single block case |
| `test_block_boundaries` | (s, e, k) for each block |
| `test_empty_flanks` | Empty A and B |
| `test_three_blocks` | Three adjacent repeat blocks |

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

## Adding New Tests

To add tests for a new module (e.g., `repeats.py`):

1. Create `tests/test_repeats.py`
2. Import fixtures from `conftest.py`
3. Use validation functions from `nwflex.validation` where available
4. Group related tests in classes with descriptive names
5. Use `@pytest.mark.parametrize` for multiple similar test cases

Example structure:

```python
import pytest
from nwflex.repeats import phase_repeat, STRLocus

class TestPhaseRepeat:
    """Test the phase_repeat utility function."""
    
    @pytest.mark.parametrize("R,a,b,M,expected", [
        ("AT", 0, 0, 3, "ATATAT"),
        ("AT", 1, 0, 3, "TATATA"),
        # ... more cases
    ])
    def test_phase_repeat_cases(self, R, a, b, M, expected):
        result = phase_repeat(R, a, b, M)
        assert result == expected


class TestSTRLocus:
    """Test the STRLocus dataclass."""
    
    def test_basic_construction(self):
        locus = STRLocus(...)
        assert locus.motif == ...
```
