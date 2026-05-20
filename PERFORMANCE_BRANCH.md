# Performance comparison: NW-flex vs BWA-MEM

This branch adds a performance comparison between NW-flex and BWA-MEM
to the NW-flex repository. The comparison is delivered as two guided
walkthrough notebooks that simulate reads from real human STR loci,
align them with both methods under matched conditions, and report
where the methods agree and where they diverge.

We are combining the messy versions from parallel repos into a clean
version that lives in the main repo. Lessons learned will be presented
as though we already knew the answers.

## The Short Story

We compare BWA-MEM (the most widely used short-read aligner) to NW-flex
on simulated reads from real human STR loci. Reads are tiled across
simulated haplotypes whose repeat count differs from the reference.

**BWA-MEM at standard parameters** is mostly wrong on these reads:
soft-clipping at the boundary consumes part of the flank and the
repeat length is unmeasured. Under principled scoring (soft-clip
charged the same affine-gap penalty as an insertion of the same
length), the truth-shape alignment strictly outscores the soft-clip,
so this is a heuristic-miss failure rather than a score-landscape
one.

**Disabling soft-clipping** recovers many cells, but BWA's
seed-and-extend heuristic is direction-dependent: in a noticeable
fraction of cells the forward strand finds the truth and the
reverse-complement strand misses it (or vice versa). Running both
orientations recovers more reads, but the **principled test** is
whether the truth alignment is itself co-optimal in the score
landscape — that question generalizes uniformly across BWA fwd, BWA
rc, NW-flex fwd, and NW-flex rc.

**NW-flex is uniformly correct on length-only sweeps.** With a single
SNV adjacent to the repeat boundary, NW-flex still passes most cells.
A band at small lflank flips to *tied* (truth is co-optimal but
deterministic tie-break picked a different alignment). A small set
at extreme contractions flips to *outscored* — the EP pattern's
per-base-skip flexibility lets the aligner construct a wrong
alignment that strictly outscores the truth.

**Compound repeats** sharpen the result. The reference is built from
two adjacent motifs joined by a bridge of length $|M|$, and the
haplotype varies both counts independently. BWA-MEM fails by a
gradient that depends on motif similarity, bridge length, and
absolute counts. NW-flex is correct by construction — when the only
difference between haplotype and reference is the repeat counts, the
algorithm is guaranteed to find the optimum over the allowed counts.

The single-repeat story (length sweep and single SNV) lives in
Notebook 7. The compound-repeat story, which needs its own simulation
setup with the new $|M|$ bridge parameter, lives in Notebook 8.

## Two scoring conventions

Throughout the notebooks we are careful to distinguish two scores:

- **SW (Smith-Waterman / local-extension) score** — what BWA-MEM
  emits in its `AS:i:` tag. This is the maximum cumulative score
  *during* BWA's local extension; it can land at an interior cell
  when the optimal global path dips through an indel. SW score is
  defined only for the alignment BWA chose; the natural truth
  alignment has no SW counterpart.

- **NW (Needleman-Wunsch / global) score** — affine-gap walked over
  a full CIGAR with soft-clips charged the same affine-gap penalty
  as an insertion of the same length, computed via `score_alignment`
  in `simulation.core`. NW score is well-defined for any CIGAR
  (BWA's output, NW-flex's output, or a constructed truth alignment)
  and matches NW-flex's reported `RefAligner` score by construction
  on CIGARs without soft-clip ops. We use it whenever we *evaluate*
  an alignment.

SW and NW can disagree on BWA-MEM hits for two independent reasons:
under no-clip, BWA's `mem_chain2aln` keeps the local extension's
`max` even after switching to the global path, so e.g. `AS = 146` for
an alignment whose NW score is 138; under standard parameters a
soft-clipped tail adds another affine-gap charge on the NW side
(e.g. `AS = 146` for `146M4S`, NW = 136). We sidestep both by always
rescoring under NW.

## Verdict

We evaluate each (read, arm) cell along two independent axes:

1. **Length correctness** — does the chosen CIGAR recover the truth
   alignment's repeat length? The alignment must span the repeat
   (at least one reference base consumed on each side of the repeat
   interval) and its decoded repeat length must equal the truth's.
2. **Score relationship** — how does the chosen alignment's NW score
   compare to the truth alignment's NW score: $>$, $=$, or $<$?

A length-correct alignment necessarily matches the truth's score; the
score axis only carries information when the alignment is wrong on
length. Crossed against the correctness axis, the score relationship
distinguishes three failure modes:

- *score $<$ truth* — heuristic miss: the truth was reachable but
  the aligner left score on the table. Structurally impossible for
  NW-flex (its DP is exhaustive).
- *score $=$ truth* — co-optimal: the aligner could have picked
  truth; deterministic tie-break landed elsewhere. We treat this as
  a fair success.
- *score $>$ truth* — the score landscape itself prefers a wrong
  alignment.

We compute both axes per strand. Each read has a forward arm and a
reverse-complement arm, and **both BWA-MEM and NW-flex** get the
same treatment: BWA-MEM is run on the read and its reverse
complement; NW-flex is run on the original (read, reference) and on
the mirror frame (reverse-complement of both). The mirror frame is
built once during simulation setup so that all four configurations
(BWA fwd, BWA rc, NW-flex fwd, NW-flex rc) consume strand-equivalent
inputs. Per-strand verdicts can be combined with a `combine` policy
(`"best"` gives the arm every benefit of the doubt; `"worst"` makes
the weakest strand dominate); the heatmap also shows them split
diagonally so the strand asymmetry stays visible.

## Input panel of repeat loci

Our simulations use a panel of repeat loci sourced from the human
reference genome (hg38). The panel TSV has one row per locus, columns:

    pind, chr, start_38, stop_38, strand, type,
    lflank, rflank, ms_seq, ref_score_per_base

Panel lives at `data/hg38_motif_sample_K100.tsv`. A separate appendix
notebook (`notebooks/Appendix_TRF.ipynb`) documents the TRF run and
panel construction.

## New Validation Notebooks

The work splits across two notebooks. Notebook 7 builds the single-repeat
machinery and runs the length and SNV comparisons. Notebook 8 reuses
the verdict and scoring conventions but rebuilds the simulation around
a compound-repeat reference.

### Notebook 7 — single repeat (length and SNV)

#### Simulation setup

We construct the simulation in four stages.

The **locus** comes from the panel: a real flank pair, a real repeat
motif, and a chosen reference repeat count $N$. Together these give
the reference sequence $X = A \cdot R^N \cdot B$. We trim the genomic
flanks to a fixed length and a clean motif-edge boundary so the
boundary between flank and repeat is unambiguous.

The **haplotype** is the sequence we sample reads from. It uses the
same flanks as the reference but a different repeat count
$N + \Delta$, optionally with one or more SNVs. The argument that
follows lives in how reads of the haplotype line up against the
reference.

The **reads** are tiled across the haplotype at a fixed read length,
constrained to cover at least $K$ bases of each flank. Within that
constraint the read start determines how much left flank the read
covers — its *lflank extent*.

The **mirror frame** is a single derived object — locus and reads
both reverse-complemented — that supports running every aligner on
both orientations under matched conditions. Building it once at
setup means the alignment configurations downstream see strand
fairness as a property of the inputs, not a side step inside each
aligner.

#### Three alignment configurations

Each read is aligned against the locus reference under three
configurations:

1. **BWA-MEM at standard parameters** — soft-clipping allowed.
2. **BWA-MEM with soft-clip suppressed** (`-L 500`) — clipping never
   improves the score, isolating the soft-clip mechanism.
3. **NW-flex** with the STR-aware EP pattern from Notebook 4, against
   an extended reference of $3N$ repeat copies so the EP pattern can
   match haplotype counts both below and above $N$.

A closer-look section walks the chosen CIGARs on three illustrative
reads (a centered read and the two boundary reads), and an SW-vs-NW
score explainer documents the gap between BWA's reported SW score
and the recomputed NW score on boundary cells.

#### First comparison — length variation only

We run the simulation with the haplotype flanks unchanged and the
repeat count varying over $N + \Delta$ for a small range of $\Delta$.

NW-flex is uniformly **length-correct**. BWA-MEM at standard
parameters is mostly *missed* — the heuristic chose a soft-clip
alignment whose NW score (with soft-clip charged like an insertion)
sits below the truth-shape alignment's. A smaller band of cells
still flips to *outscored*, reflecting reads where BWA-MEM picks a
non-truth, non-clip alignment that beats truth on the score
landscape. The no-clip arm shows a stair-step: many length-correct
cells, but for negative-$\Delta$ reads at small lflank, BWA's
heuristic in one direction is a miss even though truth has a strictly
higher NW score (the other direction often finds it). The triangle
visualization makes the strand asymmetry direct.

#### Second comparison — a single SNV in the flank

We repeat the first comparison with one change: the haplotype carries
a single SNV in the left flank, two bases inside the boundary
(landing just outside the body but inside the read's flank context).
Locus, read tiling, and alignment configurations unchanged.

NW-flex remains length-correct where it has solid flank overhang. A
band of cells at lflank ∈ {2, 3, 4} flips to *tied* — the truth's
NW score equals the chosen alignment's score; the aligner's
tie-break landed differently. At $\Delta=-5$ (0-motif haplotype)
three cells flip to *outscored*: the EP pattern's per-base-skip
flexibility lets the aligner construct a wrong alignment that
strictly outscores the truth.

A diagnostic cell at the bottom of this section walks every NW-flex
non-length-correct cell, displays the chosen vs truth CIGARs side by
side, and confirms the tied/outscored split.

### Notebook 8 — compound repeat

Compound-repeat alignment introduces a new structural parameter and
needs its own simulation setup.

#### Compound-locus simulation

The reference is built from two adjacent repeat motifs joined by a
bridge of length $|M|$:
$X = A \cdot R_1^{N_1} \cdot M \cdot R_2^{N_2} \cdot B$. The flanks
$A$ and $B$ come from the panel as in Notebook 7. The bridge $M$ is
a fixed-length sequence drawn from real adjacent context so the
boundary between each repeat block and the bridge is clean. The
haplotype varies both counts independently:
$X' = A \cdot R_1^{N_1 + \Delta_1} \cdot M \cdot R_2^{N_2 + \Delta_2} \cdot B$.

Reads are tiled across the haplotype as in Notebook 7. The mirror
frame is built the same way.

#### Three alignment configurations on compound

The two BWA configurations carry over without change — BWA does not
need to know about the compound structure. NW-flex uses a
multi-block EP pattern built by `build_EP_multi_STR_phase`, which
enumerates the product of allowed counts in a single pass against a
$3N$-style extended compound reference.

#### Third comparison — the $(\Delta_1, \Delta_2)$ sweep

We sweep both deltas over a grid and produce a heatmap per arm. The
EP pattern for two repeat blocks is exhaustive over the grid, so
NW-flex is correct everywhere by construction. BWA-MEM is correct
on part of the grid; the size and shape of the failure region
depends on motif similarity and the absolute counts.

#### Bridge-length effect

We sweep $|M|$ over a small range and stack the resulting per-arm
heatmaps so the dependence of BWA's failure region on bridge length
is visible side by side. NW-flex remains correct by construction.

## Not included

- No changes to the NW-flex core algorithm.
- No external service or large-data dependency. The notebooks run on
  the committed panel TSV plus `bwa` and `samtools` on `PATH`; if
  those are missing, the BWA cells skip cleanly with an installation
  hint.

## Plan

The work on this branch falls into a few areas. We move between them
as the notebooks drive demand — the goal is for intermediate states
to run, not for any one area to be finished first.

**Notebook 7 — single repeat**
- [x] Outline, introduction, setup
- [x] Simulation setup — locus, haplotype, reads
- [x] Simulation setup — mirror frame (in the setup section)
- [x] Three alignment configurations (BWA std/no-clip + NW-flex, with
      closer-look and SW-vs-NW score explainer)
- [x] Verdict — two-axis (length, score) framing; lives as the
      `### Correctness rule` subsection of the alignment-configurations
      section, with the strand-inequivalence material in `### Mirror`
- [x] First comparison — length variation
- [x] Second comparison — SNV in flank (with the NW-flex tie/outscored
      diagnostic)
- [x] Summary

**Notebook 8 — compound repeat**
- [x] Outline, setup, scoring carry-through
- [x] Compound-locus simulation (with $|M|$ bridge parameter)
- [x] Compound haplotype and reads
- [x] Three alignment configurations on compound (multi-block EP)
- [x] Correctness rule and truth-alignment helpers for compound
- [x] $(\Delta_1, \Delta_2)$ sweep with per-arm heatmaps
- [x] Bridge-length effect grid
- [x] Verdict — `## Two-level correctness` (per-block length checks
      plus the score relationship)
- [x] Mirror frame for NW-flex symmetric treatment
- [x] Summary

**Package code** (lives in `nwflex/simulation/`, split into `core.py`,
`viz.py`, and `sweep.py`)
- [x] Default parameters for the score schemas in use
- [x] Panel loading and locus construction
- [x] Haplotype and read tiling
- [x] BWA-MEM wrappers — `align_bwa` (single-strand) and
      `align_bwa_both_strands` (both orientations)
- [x] CIGAR utilities — `parse_cigar`, `decode_z_bp`,
      `flank_bases_consumed`, `is_arm_correct`,
      `rc_to_forward_alignment`
- [x] Scoring helpers — `score_alignment`; `bwa_truth_cigar`,
      `nwflex_truth_cigar`
- [x] Verdict helpers — `alignment_state`, `alignment_state_multi`,
      `combine_states`, `state_to_glyph`
- [x] Visualization — `render_zoom`, `plot_correctness_heatmap`
- [x] Mirror frame — `build_mirror_frame`
- [x] Compound-repeat helpers (`CompoundLocus`, multi-block EP,
      `is_arm_correct_multi`, compound truth-CIGAR builder)
- [x] NW-flex harness — `align_nwflex` / `NwflexResult` in `core.py`,
      with strand handling driven by the mirror frame
- [x] Sweep harness — `sweep.py`: `SweepVariant`, `make_variant`,
      `sweep`, `pivot_for_heatmap`, the `BWAMethod` / `NWFlexMethod`
      method classes (plus `BWACompoundMethod` / `NWFlexCompoundMethod`),
      `wrap_methods_for_multizone_truth`, and `aggregate_per_cell`

**Data and tests**
- [x] Panel TSV in `data/`
- [x] TRF parsing, annotation, and filtering utilities
- [x] Tests covering the simulation modules — see `tests/test_simulation.py`

**Repo plumbing**
- [x] Add Notebook 7 and Notebook 8 to `notebooks/build_pdf.sh`
- [x] Note the `bwa` and `samtools` runtime requirements in the install
      docs

**Scripts** (`scripts/`)
- [x] Cross-locus batch sweep — `run_batch_sweep.py`, with
      `aggregate_results.py` / `aggregate_per_locus_for_A.py` for
      per-locus rollups and `sweep_viz.py` / `build_manuscript_figures.py`
      / `build_stats_tables.py` for figures and tables

**Later**
- [x] Appendix notebook documenting TRF and panel construction
      (`Appendix_TRF.ipynb` exists; polish deferred)

## Deferred follow-up work

These items came out of the pre-merge code review but were scoped out
of this PR.  Listed here so the next pass has a starting point.

### Compound pipeline test coverage

The single-zone code in `nwflex/simulation/` is well-covered
(`tests/test_simulation.py`, 139 tests).  The compound pipeline ships
with one test class (`TestIsArmCorrectMulti`) and is otherwise
untested.  A focused follow-up should mirror the single-zone coverage
for the compound surface.

Three tiers, in priority order:

**Tier 1 — Construction & invariants** (cheap, high-value; mirror
existing single-zone tests, each 20-60 lines, no real aligner needed):
- `TestBuildCompoundLocus` — verify `CompoundLocus` and
  `build_compound_locus_from_panel`: flank lengths, bridge insertion
  offset, ref sequence is $A \cdot R_1^{N_1} \cdot M \cdot R_2^{N_2}
  \cdot B$, zone1 / zone2 endpoints, motif lengths.
- `TestBuildCompoundHaplotype` — verify `build_compound_haplotype`
  varies both repeat counts independently, produces correct
  `hap_z1` / `hap_z2` intervals, total length matches the closed-form.
- `TestBuildCompoundMirrorFrame` — verify the RC handling swaps zone
  order, preserves the bridge as its own RC, and produces matching
  read mirroring.
- `TestAlignmentStateMulti` — P/T/M/D classification on multi-zone
  alignments, especially the case where one zone is correct and the
  other is not.

**Tier 2 — Round-trip / scoring correctness** (catches scoring drift):
- `TestBwaCompoundTruthCigar` — build a known compound truth CIGAR,
  rescore it with `score_alignment`, verify the score matches a
  cell-by-cell tally.  Cover zero-delta, single-block delta, and
  both-block delta.
- `TestNwflexCompoundTruthCigar` — same for the NW-flex truth path;
  confirm the EP-pattern path scores to the truth.

**Tier 3 — Sweep harness on compound** (mock-based, mirrors
`TestSweep`):
- `TestSweepCompound` — mock the alignment to return a known hit,
  verify `BWACompoundMethod.classify` and
  `NWFlexCompoundMethod.classify` produce the expected state.
- (Optional) `TestWrapMethodsForMultizoneTruth` — verify the helper
  composes a multi-zone method out of single-zone methods.

**Scope estimate**: Tier 1 ≈ 250 lines / 12 tests; Tier 2 ≈ 120 lines
/ 5 tests; Tier 3 ≈ 100 lines / 6 tests.  Total ≈ 470 lines, ≈ 23
tests.  Tiers 1 and 2 are the priority pair — the compound
construction and scoring is where the back-and-forth in commit
`6dbec36` ("fixed reverse complement errors in compound alignment")
landed.

### Other deferred items

- **NB7 `## Diagnostic — non-length-correct cells`.**  The plan
  described a diagnostic cell that walks every NW-flex non-length-
  correct cell and displays the chosen vs. truth CIGARs side by side.
  Not added in this PR.
- **NB8 per-read verdict table + SW-vs-NW worked example.**  NB7 has
  these; NB8 forwards to NB7 for the SW-vs-NW arithmetic.  A compound
  worked example would round out NB8.
- **NB7 / NB8 execution-count cleanup.**  Both notebooks have
  non-contiguous code-cell execution counts (NB7: `..., 18, 25, 20,
  26`; NB8: `..., 17, 23, 19`) left over from rerunning individual
  cells rather than _Kernel → Restart Kernel and Run All Cells_.
  Outputs are fresh and the supplement PDF builds either way, but the
  numbering will read odd to a careful reader.  Resolve before the
  next supplement-PDF build.
- **`scripts/sweep_viz.py` shim.**  After the heatmap consolidation
  the 1D-proportion plotters live in `nwflex.simulation.viz`; this
  file is now a thin re-export shim kept so existing `from sweep_viz
  import …` callers (`scripts/aggregate_results.py`) keep working.
  Optional: migrate the callers to import from `nwflex.simulation`
  directly and delete the shim.

### Completed during the code-review pass

- **Pre-merge code review** across notebooks, package code, scripts,
  and supplement.  Four parallel reviewers; structured findings;
  ~50 items triaged into Critical / Important / Minor and worked.
- **Heatmap function consolidation** — body duplication across the
  eight `plot_*_heatmap*` variants collapsed via shared internal
  helpers; public API unchanged; 11 representative renderings are
  byte-identical before / after; commit `aee0867`.
- **Data consolidation.** Retired the slim / `data_full/`
  bifurcation; everything lives under one canonical
  `supplement/data/` tree, fed by a pair of YAML configs
  (`single_repeat.yaml`, `compound.yaml`).
- **Stats-table writer.**  Underscore-escaping in column headers and
  integer formatting for count columns, so the emitted `.tex` files
  compile cleanly without `\usepackage{underscore}` and read as
  `5420` rather than `5420.000`.
- **CSV / TeX artifacts** untracked from the repo (`.gitignore` now
  carries the globs); every regenerable artifact is produced by the
  pipeline scripts on demand.
