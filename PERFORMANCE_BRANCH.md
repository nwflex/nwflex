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
With BWA-MEM at standard parameters, soft-clipping consumes the flanks
near the repeat boundary and the repeat length is unmeasured.
Disabling soft-clipping recovers some of these cases but still loses
to NW-flex when a single SNV sits adjacent to the repeat boundary.
NW-flex remains correct in both settings as long as the read has any
flank overhang. Compound repeats sharpen the result: BWA-MEM fails by
a gradient that depends on motif similarity, intervening sequence, and
length. NW-flex is correct by construction — when the only difference
between haplotype and reference is the repeat counts, the algorithm is
guaranteed to find the optimum over the allowed counts.

## Input panel of repeat loci

Our simulations use a panel of repeat loci sourced from the human
reference genome (hg38). We assume the panel is provided as a TSV with
one row per locus, columns:

    pind, chr, start_38, stop_38, strand, type,
    lflank, rflank, ms_seq, ref_score_per_base

The panel lives at `data/hg38_motif_sample_K100.tsv`

We may opt to add a separate appendix notebook to document how to run
TRF and how the panel is built from a TRF run. There is code and instructions
but not strictly needed.

## New Validation Notebook

The notebook is a guided walkthrough. We build up the simulation
machinery once, set up the alignment under three configurations, and
then run the comparison three times, each time changing one
thing about the simulation.

### Simulation setup

We construct the simulation in three stages.

The **locus** comes from the panel: a real flank pair, a real repeat
motif, and a chosen reference repeat count $N$. Together these give
the reference sequence $X = A \cdot R^N \cdot B$. We trim the genomic
flanks to a fixed length and a clean motif-edge boundary so the
boundary between flank and repeat is unambiguous.

The **haplotype** is the sequence we will sample reads from. It uses
the same flanks as the reference but a different repeat count
$N + \Delta$. The argument that follows lives in how reads of the
haplotype line up against the reference.

The **reads** are tiled across the haplotype at a fixed read length,
constrained to cover at least $K$ bases of each flank. Within that
constraint we vary the read start so reads cover different amounts of
the left flank — we call this the read's *flank extent*.

### Three alignment configurations

Each read is aligned against the locus reference under three
configurations: BWA-MEM at standard parameters; BWA-MEM with the
soft-clip penalty raised so high that clipping never improves the
score; and NW-flex with the STR-aware EP pattern from Notebook 4.

For NW-flex we use an extended reference with $3N$ repeat copies, 
so the EP pattern can match haplotype counts both below and 
above $N$. For BWA-MEM we align the read in both orientations and
credit it if either run finds the truth — Smith-Waterman tie-breaking
depends on DP cell evaluation order, so running both orientations
removes that artifact.

A read is correct under a method if the method recovers the
repeat-region length encoded in the haplotype and has a non-trivial
left and right flank extent. We use CIGAR-based length decoders.

### First comparison — length variation only

We run the simulation above with the haplotype flanks left untouched
and the repeat count varying over $N + \Delta$ for a small range of
$\Delta$. We tabulate correctness for each method as a function of
flank extent and $\Delta$ and present the result as a heatmap with
one panel per method.

NW-flex is uniformly correct as long as the read has any flank
extent. BWA-MEM at standard parameters fails along the boundary
even though the haplotype differs from the reference only in the
repeat count. The no-clip arm recovers some of these cases but not
all.

### Second comparison — a single SNV in the flank

We repeat the first comparison with one change: the haplotype carries
a single SNV at a fixed position. We choose for our example a single
base change in the left flank, one base outside
the repeat boundary. The locus, the read tiling, and the alignment
configurations are unchanged.

The same heatmap now reads differently. NW-flex remains correct in
the regime where it has flank overhang, but not always. Sometimes,
the local sequence and variant have a higher value alignment. 
The no-clip arm — the one that recovered the easy cases above — now 
fails on the reads that cross the SNV.

### Third comparison — compound repeat

The third comparison changes the locus structure. The reference is
built from two adjacent repeat motifs joined by a short interrupting
sequence,
$X = A \cdot R_1^{N_1} \cdot M \cdot R_2^{N_2} \cdot B$, and the
haplotype varies both counts independently. The reads, the alignment
methods, and the correctness rule carry over.

The EP pattern for two repeat blocks enumerates the product of
allowed counts in a single pass, so NW-flex is correct everywhere on
the $(\Delta_1, \Delta_2)$ grid by construction. BWA-MEM is correct
on part of the grid; the size and shape of the failure region
depends on motif similarity, the length of the interrupting
sequence, and the absolute counts.

## Not included

- No changes to the NW-flex core algorithm. 
- No full benchmarking harness or command-line sweep tooling. Notebook 
  generates a representative slice inline; larger-scale results deferred.
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
      with closer-look render_zoom views, the correctness rule, and
      the inequivalence-of-orientations demo motivating both-strands BWA)
- [ ] First comparison — length variation
- [ ] Second comparison — SNV in flank
- [ ] Third comparison — compound repeat
- [ ] Summary

**Package code** (lives in `nwflex/simulation/`, a package split into
`core.py` and `viz.py`)
- [x] Load default parameters for different score schema
- [x] Panel loading and locus construction
- [x] Haplotype and read tiling
- [x] BWA-MEM wrappers — `align_bwa` (single-strand) and
      `align_bwa_both_strands` (both orientations)
- [x] CIGAR utilities — `parse_cigar`, `decode_z_bp`,
      `flank_bases_consumed`, `is_arm_correct`, `rc_to_forward_alignment`
- [x] Alignment visualization — `render_zoom` (in `simulation.viz`)
- [x] NW-flex setup stays inline in the notebook (3N STRLocus + EP
      pattern + RefAligner; no wrapper, by design)
- [ ] Compound-repeat helpers

**Data and tests**
- [x] Panel TSV in `data/`
- [x] Tests covering the new modules — 81 in `tests/test_simulation.py`

**Repo plumbing**
- [ ] Add the notebook to `notebooks/build_pdf.sh`
- [ ] Note the `bwa` and `samtools` runtime requirements in the install docs

**Later**
- [ ] Appendix notebook documenting TRF and panel construction
- [ ] Scripts for data generation at scale
