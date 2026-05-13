# NW-flex domain testing guide

This document captures the algorithmic invariants and boundary
regimes that the NW-flex test suite should encode. It is the
domain-knowledge reference that test authors (human or agent) should
consult before adding or rewriting tests: assertions should anchor
to the invariants here, not to whatever value the current
implementation happens to return.

The companion file in the repo root, `TEST_BRANCH.md`, explains *why*
the test suite is being rewritten and what is in scope on the current
branch. This file is intended to outlive that branch and become the
durable reference.

## Domain invariants the tests should encode

A non-exhaustive list, drawn from the notebooks and core modules.
These are the things that should fail loudly if the algorithm
regresses, regardless of which path produced the output:

- **Score consistency.** `score_alignment(CIGAR)` matches the
  aligner's reported score on every CIGAR without soft-clip ops,
  exactly. With soft-clips, the NW rescoring charges them as
  deletions of equal length.
- **Mirror-frame symmetry.** For any (locus, read) pair, NW-flex on
  the original inputs and NW-flex on the mirror frame produce the
  same NW score; per-strand outputs may differ via deterministic
  tie-break but cannot disagree on optimality.
- **Phase preservation.** Every row jump taken in an STR alignment
  corresponds to a valid `(a, b, M)` phase combination for the
  locus; the inferred `(a, b, M)` from traceback equals the one
  used to construct the read on perfect-locus inputs.
- **Single-block guarantee.**
  $S_\text{flex}(A \cdot Z \cdot B, Y) = \max_{Z^* \subseteq Z}
  \text{NWG}(A \cdot Z^* \cdot B, Y)$ — checked against the naive
  enumeration baseline.
- **Compound exhaustiveness.** Multi-block EP enumerates the full
  product of allowed counts; when haplotype counts lie inside the
  allowed range, NW-flex is correct by construction.
- **Cython equivalence.** `run_flex_dp_fast` and `run_flex_dp` agree
  on score, alignment strings, DP tables, and detected jumps.
- **Verdict axes.** `alignment_state` reports length-correctness and
  the score relationship ($<$, $=$, $>$ vs truth) consistently with
  the manual two-axis framing in Notebook 7.

## Edge cases worth explicit tests

These are the regimes where NW-flex has actually shipped bugs (see
recent fix commits on `main`: `11c70ab`, `56e69a8`, `5baef74`,
`3b8f6ea`, `59878b6`, `fc95192`, `2d5fd6e`). The pattern is the
same in each case: a boundary configuration that the centered,
well-formed test inputs never exercise.

**Reference / EP boundary geometry**

- **STR block flush against the reference end** (`e == n`, no B
  flank). `build_EP_STR_phase` needs to apply the phase-preserving
  exit pattern to `EP[n+1]` because the closer row `e+1` does not
  exist. Multi-block STR alignments where the final block has no
  trailing flank.
- **STR construction ending in the STR** — analogous failure mode in
  EP pattern generation; see `3b8f6ea`.
- **STR block flush against the reference start** (no A flank) —
  symmetric case; verify alongside the `e == n` case.
- **Zero-length flanks** generally — A or B empty.

**DP-table boundary cells**

- **Row 0 and column 0 trace pointers.** Cells on row 0 / column 0
  are unreachable but their trace pointers are still read during
  traceback. Row 0 must point left, column 0 must point up;
  otherwise `j` (or `i`) goes negative and the path buffer corrupts
  memory with boundscheck off. Trigger: any path that enters these
  cells during traceback, including soft-clip with bisulfite scoring.
- **`free_Y` mode and `M[0, j]`.** `M` is unreachable on row 0
  regardless of `free_Y`; the initialization must respect that.
- **`free_X = True` trace gutters** — the boundary trace values
  along the free axis must be set consistently, otherwise the
  buffered Cython path segfaults.
- **Semiglobal mode `Xg_tr` initialization** — wrong initial state
  caused a segfault; assert correctness across all four
  free_X/free_Y combinations.

**CIGAR / traceback**

- **Terminal contraction N ops in CIGAR.** Conversion from the path
  matrix to CIGAR must emit terminal contraction operations when
  the traceback ends inside a contracted block.
- **Terminal traceback tie-breaking.** When multiple terminal cells
  tie on score, prefer row `n` (full-reference consumption). Tests
  should construct a tied case and assert the choice deterministically.
- **`alignment_to_cigar` start position.** The reported start
  position must match where the alignment actually begins, not
  where the DP path was first nonzero (see `2d5fd6e`).

**EP refinement / candidate selection**

- **Stale-value bugs in EP candidate updates.** The fix in `5baef74`
  removed in-place updates that read stale values mid-pass for `M`
  and `Xg` candidate selection. Tests should construct EP patterns
  where the in-place vs out-of-place order would produce different
  scores, and assert the correct (out-of-place) result.

**Numeric / dtype**

- **float32 DP tables.** With the float32 switch in `3b8f6ea`,
  scores near the precision limit can diverge from the float64
  reference. Worth a regression test that scales scoring parameters
  up enough to expose any precision-sensitive path.

**Cross-module consistency**

- Cython vs Python equivalence at every boundary case above —
  scores, alignment strings, DP tables, jumps.
- Mirror-frame symmetry: NW score on `(X, Y)` and on the
  reverse-complement of both should agree.

## How to use this document

When adding or rewriting a test:

1. Identify which invariant or edge case the test is meant to
   protect, and cite it in a comment or docstring (e.g.
   "regression for `e == n` EP terminal predecessor — see
   `DOMAIN_GUIDE.md`").
2. Build the input to *trigger* the boundary regime, not to
   demonstrate the centered well-formed case. The centered case
   belongs in a smoke test; this file is about the rest.
3. Anchor the assertion to the invariant (a property that must hold
   for any correct implementation), not to a specific numeric value
   captured from the current run. Numeric snapshots are acceptable
   only when paired with an independent recomputation
   (`score_alignment`, naive baseline, mirror frame, etc.).

When a new bug is fixed in `nwflex/`, the fixing commit should add
both a regression test and an entry here describing the regime.
