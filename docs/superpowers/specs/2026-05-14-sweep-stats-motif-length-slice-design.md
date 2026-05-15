# Sweep summary stats: slice single-repeat to dinucleotides

**Date:** 2026-05-14
**Branch:** performance
**Status:** design — awaiting review

## Problem

The single-repeat sweep covers all panel loci, but the panel is
**400 / 1,080 / 5,420** loci at motif length 1 / 2 / 3. Every
summary statistic that pools across motif length is therefore
weighted **5.8% / 15.7% / 78.6%** — the headline "method recovers
X%" number is, in effect, the trinucleotide result. That weighting
is an artifact of the genome's motif inventory and panel
construction, not a design choice, and it is invisible in the
output.

Worked example (single-repeat, no SNV, score = truth): BWA-MEM at
standard parameters reads 0.314 pooled, but the per-stratum values
are 0.640 / 0.412 / 0.271 for L = 1 / 2 / 3 — the pooled number sits
on top of L = 3 because trinucleotides dominate.

The compound side has the same kind of stratum-size imbalance
(12 / 48 / 100 motif pairs per length-pair) but the strata *agree*
(all 0.76–0.87), so pooling barely moves it (0.848 → 0.849).

## Decision

Do **not** reweight or normalize. Instead, **slice the
single-repeat summary statistics and summary figures to motif
length 2 (dinucleotides)**:

- Dinucleotides are the canonical microsatellite class, so the
  slice reads as a deliberate scientific scope rather than a
  convenience.
- L = 2 is the middle stratum; its value (0.412) sits next to the
  length-equal-weighted value (0.441), so the slice is neither a
  flattering nor an unflattering pick.
- The figure pipeline already does this — `aggregate_per_locus_for_A.py`
  slices single-repeat to `motif_len = 2`.

The **compound side stays pooled/mixed** — untouched.

One motif-length view is **kept deliberately**, unfiltered, so the
L = 1 → L = 3 degradation trend (a real result a reviewer will ask
about) stays on record. This costs nothing: the two functions that
provide it already exist.

## Scope

### `scripts/build_stats_tables.py` — single-repeat only

Add a module constant `SINGLE_MOTIF_LEN = 2`. In `_maybe_single`,
derive `sr_di = sr_df[sr_df["motif_len"] == SINGLE_MOTIF_LEN]` and
route it to the tables that currently pool across motif length:

- `_single_recovery_by_n_snv` → `sr_di`
- `_single_strand_asymmetry` → `sr_di`
- `_single_per_locus_distribution` → `sr_di`
- `_nwflex_t_breakdown` → `sr_di`
- `_single_motif_length_breakdown` → **keep `sr_df` (unfiltered)** —
  this is the deliberate motif-length exhibit.

Update the captions of the sliced tables to state "dinucleotide
loci".

Compound functions (`_recovery_by_bridge`, `_bridge_breakpoint`,
`_strand_asymmetry`, `_motif_length_breakdown`,
`_compound_per_locus_distribution`) are **not touched**.

### `scripts/aggregate_results.py` — single-repeat only

Add the same `SINGLE_MOTIF_LEN = 2` constant. In `main()`, derive
`sr_di` and route it to the single-repeat summary figures that pool
across motif length:

- `_single_repeat_proportion_per_n` → `sr_di`
- `_single_repeat_proportion_snv_stack` → `sr_di`
- `_single_repeat_proportion_by_N` → `sr_di`
- `_single_repeat_proportion_by_motif_length` → **keep `sr_df`** —
  the deliberate trend figure.
- `_single_repeat_tidy_aggregate` → **keep `sr_df`** — it writes the
  committed `single_repeat_cross_locus_aggregate.csv` keyed *by*
  `motif_len` (already stratified, not pooled); downstream consumers
  slice it themselves.

Update the suptitles/subtitles of the sliced figures to state
"dinucleotide loci".

Compound functions are **not touched**.

### Text

State the "why L = 2" rationale once where the single-repeat
headline numbers are quoted (NB7 summary and/or `PERFORMANCE_BRANCH.md`).
Coordinate with the in-flight notebook work — this spec does not
edit notebooks.

## Data

The raw shards `build_stats_tables.py` and `aggregate_results.py`
consume are gitignored. They are archived at
`~/temp/nwflex_sweep_shards.zip` (201 MB) and unzip at the repo root
into the expected layout:

- `supplement/data/single_repeat/` — 144,879 single-repeat shards
- `supplement/data_full/compound/` — 10,620 compound shards
- `supplement/data/compound/` — 1,032 (slim/superseded compound)

Unzip before running the scripts.

## Non-goals

- No reweighting / normalization / equal-weighting helper.
- No new scripts, modules, or shared helpers.
- No change to the data source (scripts keep reading raw shards).
- No change to the compound path in either script.
- No change to `run_batch_sweep.py` or the sweep itself.
- No notebook edits in this change.

## Verification

After unzipping the shards:

1. `python scripts/build_stats_tables.py --config scripts/configs/small.yaml`
   — single-repeat tables reflect ~1,080 dinucleotide loci;
   `single_motif_length_breakdown` still shows L ∈ {1, 2, 3};
   compound tables (if compound shards present) unchanged.
2. `python scripts/aggregate_results.py --config scripts/configs/small.yaml`
   — single-repeat summary figures are dinucleotide-only and say so;
   `single__*__motifL_stack` still shows all three lengths;
   `single_repeat_cross_locus_aggregate.csv` still carries all
   motif lengths; compound figures unchanged.

## Open judgment call

Keeping `_single_motif_length_breakdown` /
`_single_repeat_proportion_by_motif_length` as the unfiltered trend
exhibit is recommended (it is free and pre-empts a reviewer
question). The alternative is to drop L = 1 / L = 3 entirely. Flag
for review.
