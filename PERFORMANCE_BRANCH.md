# Performance comparison: NW-flex vs BWA-MEM

This branch adds a performance comparison between NW-flex and BWA-MEM
to the NW-flex repository. The comparison is delivered as a guided
walkthrough notebook that simulates reads from real human STR loci,
aligns them with both methods under matched conditions, and reports
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
repeat length is unmeasured — and the soft-clip alignment scores
*above* the truth-shape alignment under standard affine-gap, so this
is a real failure of the score landscape, not just tie-breaking.

**Disabling soft-clipping** recovers many cells, but BWA's
seed-and-extend heuristic is direction-dependent: in a noticeable
fraction of cells the forward strand finds the truth and the
reverse-complement strand misses it (or vice versa). Running both
orientations recovers more reads, but the **principled test** is
whether the truth alignment is itself co-optimal in the score
landscape — that question generalizes uniformly across BWA fwd, BWA
rc, and NW-flex.

**NW-flex is uniformly correct on length-only sweeps.** With a single
SNV adjacent to the repeat boundary, NW-flex still passes most cells.
A band at small lflank flips to *tied* (truth is co-optimal but
deterministic tie-break picked a different alignment). A small set
at extreme contractions flips to *outscored* — the EP pattern's
per-base-skip flexibility lets the aligner construct a wrong
alignment that strictly outscores the truth.

Compound repeats (val3, still in progress) sharpen the result: BWA-MEM
fails by a gradient that depends on motif similarity, intervening
sequence, and length. NW-flex is correct by construction — when the
only difference between haplotype and reference is the repeat counts,
the algorithm is guaranteed to find the optimum over the allowed
counts.

## Two scoring conventions

Throughout the notebook we are careful to distinguish two scores:

- **SW (Smith-Waterman / local-extension) score** — what BWA-MEM
  emits in its `AS:i:` tag. This is the maximum cumulative score
  *during* BWA's local extension; it can land at an interior cell
  when the optimal global path dips through an indel. SW score is
  defined only for the alignment BWA chose; the natural truth
  alignment has no SW counterpart.

- **NW (Needleman-Wunsch / global) score** — affine-gap walked over
  a full CIGAR with soft-clips treated as free, computed via
  `score_alignment` in `simulation.core`. NW score is well-defined
  for any CIGAR (BWA's output, NW-flex's output, or a constructed
  truth alignment) and matches NW-flex's reported `RefAligner` score
  by construction. We use it whenever we *evaluate* an alignment.

For boundary BWA reads under no-clip, SW and NW disagree by up to 8
points (BWA's `AS = 146` for an alignment whose NW score is 138).
This is a documented BWA-MEM choice (`mem_chain2aln` keeps the local
extension's `max` even after switching to the global path); we
sidestep it by always rescoring under NW.

## The four-state verdict

Each (read, arm) cell is classified into one of four states:

- **Pass (P)** — the aligner's chosen CIGAR recovers the truth.
- **Tied (T)** — chosen wrong, but its NW score equals the truth's NW
  score. The aligner *could* have picked truth; tie-break landed
  elsewhere. We treat tied as a co-optimal "fair" success.
- **Missed (M)** — chosen wrong, truth strictly outscores the chosen
  NW score. The aligner heuristically missed: truth was reachable
  but the aligner left score on the table. Structurally impossible
  for NW-flex (its DP is exhaustive).
- **Outscored (D)** — chosen wrong, chosen NW score strictly outscores
  truth. The score landscape genuinely prefers a wrong alignment.

For BWA arms (which run forward and reverse-complement strands),
each strand is classified independently and the cell carries both
states. The summary state combines them under a `combine` policy
(`"best"` gives BWA every benefit of the doubt, `"worst"` makes the
weakest strand dominate). NW-flex is single-arm.

## Input panel of repeat loci

Our simulations use a panel of repeat loci sourced from the human
reference genome (hg38). The panel TSV has one row per locus, columns:

    pind, chr, start_38, stop_38, strand, type,
    lflank, rflank, ms_seq, ref_score_per_base

Panel lives at `data/hg38_motif_sample_K100.tsv`.

A separate appendix notebook documenting the TRF run and panel
construction is deferred — there is code and instructions but it's
not strictly needed for the comparison.

## New Validation Notebook

The notebook is a guided walkthrough. We build the simulation
machinery once, set up the alignment under three configurations, and
then run the comparison three times, each time changing one thing
about the simulation.

### Simulation setup

We construct the simulation in three stages.

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

### Three alignment configurations

Each read is aligned against the locus reference under three
configurations:

1. **BWA-MEM at standard parameters** — soft-clipping allowed.
2. **BWA-MEM with soft-clip suppressed** (`-L 500`) — clipping never
   improves the score, isolating the soft-clip mechanism.
3. **NW-flex** with the STR-aware EP pattern from Notebook 4, against
   an extended reference of $3N$ repeat copies so the EP pattern can
   match haplotype counts both below and above $N$.

For BWA-MEM we align both orientations and classify each strand
independently. Smith-Waterman tie-breaking depends on DP cell
evaluation order, so the two strands can return different (equally
optimal) alignments; running both orientations removes the
order-of-evaluation artifact and surfaces direction-dependent
heuristic misses.

### Verdict and visualization

A read's verdict for each arm is the four-state classification above
(P / T / M / D). The heatmap renders each cell as two triangles split
along the anti-diagonal:

- lower-left triangle = forward strand
- upper-right triangle = reverse-complement strand

Cells where the strands agree render uniformly; cells where they
disagree show two colors split diagonally. Single-arm aligners
(NW-flex) render with both triangles the same color, looking like
ordinary heatmap squares. The Δ=0 column (haplotype = reference) is
outlined in black on every panel.

### First comparison — length variation only

We run the simulation with the haplotype flanks unchanged and the
repeat count varying over $N + \Delta$ for a small range of $\Delta$.

NW-flex is uniformly **Pass**. BWA-MEM at standard parameters is
mostly **Outscored** except for a thin Δ≈0 stripe — soft-clipping
wins under standard affine-gap. The no-clip arm shows a stair-step:
many Pass cells, but for negative-Δ reads at small lflank, BWA's
heuristic in one direction *Misses* the truth alignment even though
truth has a strictly higher NW score (the other direction often
finds it). The triangle visualization makes the strand asymmetry
direct.

### Second comparison — a single SNV in the flank

We repeat the first comparison with one change: the haplotype carries
a single SNV in the left flank, two bases inside the boundary
(landing just outside the body but inside the read's flank context).
Locus, read tiling, and alignment configurations unchanged.

The heatmap now reads differently. NW-flex remains Pass where it has
solid flank overhang. A band of cells at lflank ∈ {2, 3, 4} flips
to **Tied** — the truth's NW score equals the chosen alignment's
score; the aligner's tie-break landed differently. At Δ=-5
(0-motif haplotype) three cells flip to **Outscored**: the EP
pattern's per-base-skip flexibility lets the aligner construct a
wrong alignment that strictly outscores the truth.

A diagnostic cell at the bottom of this section walks every NW-flex
non-pass cell, displays the chosen vs truth CIGARs side by side, and
confirms the Tied/Outscored split.

### Third comparison — compound repeat (in progress)

The third comparison changes the locus structure. The reference is
built from two adjacent repeat motifs joined by a short interrupting
sequence,
$X = A \cdot R_1^{N_1} \cdot M \cdot R_2^{N_2} \cdot B$, and the
haplotype varies both counts independently. The reads, the alignment
methods, and the verdict carry over.

The EP pattern for two repeat blocks enumerates the product of
allowed counts in a single pass, so NW-flex is correct everywhere on
the $(\Delta_1, \Delta_2)$ grid by construction. BWA-MEM is correct
on part of the grid; the size and shape of the failure region
depends on motif similarity, the length of the interrupting
sequence, and the absolute counts.

## Not included

- No changes to the NW-flex core algorithm.
- No full benchmarking harness or command-line sweep tooling. The
  notebook generates a representative slice inline; larger-scale
  results are deferred.
- No external service or large-data dependency. The notebook runs on
  the committed panel TSV plus `bwa` and `samtools` on `PATH`; if
  those are missing, the BWA cells skip cleanly with an installation
  hint.

## Plan

The work on this branch falls into a few areas. We move between them
as the notebook drives demand — the goal is for intermediate states
to run, not for any one area to be finished first.

**Notebook**
- [x] Instantiate outline
- [x] Compose introduction with explanation and purpose
- [x] Setup cell, imports, scoring carry-through
- [x] Simulation setup — locus, haplotype, reads
- [x] Three alignment configurations (BWA-MEM std/no-clip + NW-flex,
      with closer-look render_zoom views, the SW-vs-NW score
      explainer, the correctness rule, and the
      inequivalence-of-orientations demo motivating both-strands BWA)
- [x] First comparison — length variation (four-state verdict,
      two-triangle heatmap)
- [x] Second comparison — SNV in flank (with the tie-test diagnostic
      that connects back to the inequivalence demo)
- [ ] Third comparison — compound repeat
- [ ] Summary

**Package code** (lives in `nwflex/simulation/`, split into `core.py`
and `viz.py`)
- [x] Load default parameters for different score schema
- [x] Panel loading and locus construction
- [x] Haplotype and read tiling
- [x] BWA-MEM wrappers — `align_bwa` (single-strand) and
      `align_bwa_both_strands` (both orientations)
- [x] CIGAR utilities — `parse_cigar`, `decode_z_bp`,
      `flank_bases_consumed`, `is_arm_correct`,
      `rc_to_forward_alignment`
- [x] Scoring helpers — `score_alignment` (NW score of any CIGAR);
      `bwa_truth_cigar`, `nwflex_truth_cigar` (natural truth-alignment
      builders for the locus / 3N references)
- [x] Verdict helpers — `alignment_state` (P/T/M/D classification);
      `bwa_state_both_strands` (per-strand + combined classification,
      with `combine={"best","worst"}`); `bwa_verdict_both_strands`
      (legacy boolean form, kept for backwards compatibility)
- [x] Visualization — `render_zoom` (per-alignment ASCII zoom);
      `plot_correctness_heatmap` (three-panel four-state two-triangle
      heatmap with shared legend showing forward / reverse split)
- [x] NW-flex setup stays inline in the notebook (3N STRLocus + EP
      pattern + RefAligner; no wrapper, by design)
- [ ] Compound-repeat helpers

**Data and tests**
- [x] Panel TSV in `data/`
- [x] Tests covering the new modules — 81 in `tests/test_simulation.py`
      (203 across the repo)

**Repo plumbing**
- [ ] Add the notebook to `notebooks/build_pdf.sh`
- [ ] Note the `bwa` and `samtools` runtime requirements in the install
      docs

**Later**
- [ ] Appendix notebook documenting TRF and panel construction
- [ ] Scripts for data generation at scale
