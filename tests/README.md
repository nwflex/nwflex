# NW-flex Test Suite

This document describes the test suite for the NW-flex package.
The domain-knowledge reference for *what* the tests should be
protecting lives in [`DOMAIN_GUIDE.md`](DOMAIN_GUIDE.md); this file
is the *inventory*. See [`../TEST_BRANCH.md`](../TEST_BRANCH.md) for
the ongoing audit and rewrite plan.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test module
pytest tests/test_standard.py -v
```

## Test Modules Overview

The table below covers every `test_*.py` file currently in this
directory. Modules marked **(audit pending)** were drafted by an
LLM agent and have not yet been reviewed against
[`DOMAIN_GUIDE.md`](DOMAIN_GUIDE.md); their one-line description
reflects what the file appears to cover, not a vetted contract.
The full rewrite will replace the "Purpose" column with a "what
breaks if removed" column.

| Module | Purpose |
|--------|---------|
| `test_standard.py` | Standard NW/Gotoh alignment (EP = ∅); fixed and randomized score comparisons, alignment validity, scoring-variant coverage. |
| `test_single_block.py` | Single-block flex alignment (X = A·Z·B); compares to naive substring enumeration. |
| `test_str.py` | STR repeat utilities + alignment. Phase enumeration, `(a, b, M)` inference, flex-vs-naive validations for `STRLocus` and `CompoundSTRLocus`. |
| `test_str_boundaries.py` | STR boundary regimes: terminal EP when `e == n` (no B flank), partial-phase exit at terminal, Python-vs-fast equivalence on the terminal case, and interrupted-repeat phase preservation. Ported from `nwflex_TNG@19f9ad2`. |
| `test_cython.py` | Cython vs Python equivalence — scores, alignment strings, DP tables, and detected jumps. |
| `test_aligners.py` | CIGAR parsing/writing helpers, RLE ops, `align_single_block`, `align_standard`. **(audit pending)** |
| `test_buffered.py` | Buffered Cython DP path (`RefAligner.align_simple` via `nwflex_dp_core_buffered_cigar`) vs pure-Python `run_flex_dp` across standard, single-block, and STR modes. **(audit pending)** |
| `test_default.py` | Scoring preset registry (`PRESETS`, `ScoringPreset`, `align_params`, `make_dna_score_matrix`). **(audit pending)** |
| `test_fast_path.py` | Unified Cython DP core and `RefAligner` — CIGAR-only return, equivalence with full-result path. **(audit pending)** |
| `test_path_helpers.py` | Path utilities in `nwflex.fast` — `extract_jumps_from_path`, `path_array_to_list`, `reconstruct_aligned_strings`. **(audit pending)** |
| `test_simulation.py` | Simulation harness for notebook 07 (NW-flex vs BWA-MEM) — locus/haplotype/read construction, BWA wrappers, mirror frame, verdict helpers. **(audit pending)** |
| `test_trf.py` | TRF parsing, isolation annotation, and filtering against the `data/chr21_snippet_demo.dat` fixture. **(audit pending)** |

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

Validates the core NW-flex guarantee: single-block flex score equals
the naive maximum over all Z* substrings:

```
S_flex(X, Y) = max_{Z* ⊆ Z} NWG(A·Z*·B, Y)
```

**Tests included:**
- Fixed cases: hand-crafted A·Z·B examples covering common substring-selection scenarios
- Boundary cases: block at start/end/middle and small-block edge cases
- Randomized cases: random A·Z·B lengths and sequences
- Mutated reads: substitutions and indels; verify NW-flex matches the naive baseline
- Alignment validity: aligned-length consistency, absence of double gaps, agreement between reported and recomputed/naive scores

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

## `test_str_boundaries.py` — STR boundary regimes

Regressions for STR alignment in boundary geometries that the
centered, well-formed inputs in `test_str.py` never reach. Ported
from `nwflex_TNG` (stress-test branch, commit `19f9ad2`) and
extended for our codebase. See [`DOMAIN_GUIDE.md`](DOMAIN_GUIDE.md)
§ "Reference / EP boundary geometry" for the motivation.

### TestTerminalEPDerivedCases

`e == n` family — STR block flush against the reference end with no
B flank.

| Test | What breaks if removed |
|------|------------------------|
| `test_build_ep_str_phase_terminal_predecessors_when_b_absent` | Single-block builder can silently return an empty `EP[n+1]` again. |
| `test_build_ep_multi_str_phase_terminal_predecessors_for_last_block` | Multi-block terminal case can regress even if the single-block one stays correct. |
| `test_terminal_no_b_contraction_is_perfect` | Minimal no-B contraction CIGAR (`3M14N6M`) can drift unnoticed. |
| `test_terminal_no_b_partial_phase_alignment_is_perfect` | Partial-phase exit near the terminal stops being protected. |
| `test_terminal_no_b_case_matches_fast_path` | Python and fast paths can diverge on the pathological terminal case without anyone noticing. |

### TestInterruptedRepeatDerivedCases

Phase preservation across an interrupting base between two adjacent
STR blocks.

| Test | What breaks if removed |
|------|------------------------|
| `test_interrupted_repeat_phase_correct_contraction_is_perfect` | Multi-block phase-preserving contraction can break without an obvious single-block failure. |
| `test_interrupted_repeat_wrong_phase_scores_worse_than_phase_correct` | The aligner can start ignoring the interrupted-repeat phase contract. |
| `test_interrupted_repeat_expansion_is_not_free` | EP could accidentally start behaving like free expansion. |

---

## `test_cython.py` — Cython vs Python equivalence

Ensures the Cython-accelerated implementation (`run_flex_dp_fast`)
produces identical results to pure Python (`run_flex_dp`).

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

## Audit-pending modules

The modules below are documented here for inventory only. Their
detailed sections will be filled in as part of the audit
described in [`../TEST_BRANCH.md`](../TEST_BRANCH.md). Each audit
should:

1. Classify every assertion as *domain-anchored*,
   *implementation-anchored*, or *trivial*.
2. Rewrite implementation-anchored assertions against the
   invariants in [`DOMAIN_GUIDE.md`](DOMAIN_GUIDE.md).
3. Replace this stub section with a "what breaks if removed"
   table mirroring the `test_str_boundaries.py` section above.

- `test_aligners.py` — CIGAR helpers and high-level aligner wrappers.
- `test_buffered.py` — Buffered Cython CIGAR path equivalence.
- `test_default.py` — Scoring preset registry.
- `test_fast_path.py` — Unified Cython DP core and `RefAligner`.
- `test_path_helpers.py` — `nwflex.fast` path utilities.
- `test_simulation.py` — Notebook-07 simulation harness.
- `test_trf.py` — TRF parsing and filtering (currently has failures).

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
