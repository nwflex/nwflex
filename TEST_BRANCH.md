# Testing-suite enhancement branch

This branch enhances NW-flex's test suite. It is branched from
`performance` (not from `main`) and is intended to merge directly
into `main` once `performance` lands. Edits here should be scoped so
they do not complicate that merge — see "Merge considerations" below.

## Why this branch exists

Parts of the existing test suite were drafted by an LLM agent. Agent-
generated tests are a useful scaffold but typically have three
weaknesses that we need to address by hand:

1. **Missing domain knowledge.** Tests treat NW-flex as a generic
   string algorithm and miss invariants that only make sense once you
   know what the EP pattern is *supposed* to express: phase-preserving
   row jumps, the $3N$-extended
   reference convention, etc. Tests should encode
   these invariants, not just exercise the functions.
2. **Missed edge cases.** Tests tend to cover the centered, well-
   behaved case and skip the boundary regime where bugs actually
   live.
3. **Coverage gaps.** Whole modules (`fast.py`, `ep_intervals.py`,
   parts of `simulation/`) have only smoke tests or none at all.
   Coverage should be measured and steered, not assumed.

## Scope

In-scope:

- Reviewing and rewriting every test currently in `tests/` that was
  agent-drafted, anchoring assertions to the algorithmic invariants
  documented in the notebooks and in `PERFORMANCE_BRANCH.md`.
- Adding edge-case tests around STR boundaries, mirror-frame
  symmetry, compound-repeat phase enumeration, and BWA-vs-NW score
  rescoring.
- Wiring up a coverage report (`pytest --cov=nwflex`) and using it to
  identify and close gaps. We are not chasing 100% — we are closing
  *meaningful* gaps where the code path encodes a guarantee.
- Updating `tests/README.md` to match what the suite actually does
  after the rewrite.

Out of scope:

- No changes to `nwflex/` source. If a test reveals a real bug, file
  it and either xfail the test with a pointer or fix the bug on a
  separate branch that lands before this one.
- No new notebooks. The notebooks on `performance` are the source of
  truth for what the algorithm is *supposed* to do; tests cite them,
  they don't replace them.
- No CI changes in this branch beyond what's needed to surface the
  coverage report locally. CI wiring can land separately.

## Inventory of current test files

`tests/README.md` is **out of date** — it documents only four
modules and misses the rest. Bringing it back in sync with the
suite is part of this branch's deliverable.

Actual contents of `tests/` today:

- `conftest.py` — shared fixtures (scoring params, RNGs, DNA factory).
- `test_standard.py` — documented; NW/Gotoh baseline (EP = ∅).
- `test_single_block.py` — documented; single-block flex vs naive
  substring max.
- `test_str.py` — documented; STR utilities, `STRLocus` /
  `CompoundSTRLocus`, phase-preserving EP alignment.
- `test_cython.py` — documented; Cython vs Python equivalence.
- `test_aligners.py` — undocumented; audit first.
- `test_buffered.py` — undocumented; audit first.
- `test_default.py` — undocumented; audit first.
- `test_fast_path.py` — undocumented; audit first.
- `test_path_helpers.py` — undocumented; audit first.
- `test_simulation.py` — undocumented; covers the
  `nwflex/simulation/` modules introduced on `performance`.
- `test_trf.py` — undocumented; covers `nwflex/trf.py`.

The undocumented files are the most likely agent-drafted set and the
primary audit target. Each needs a pass: confirm what it actually
asserts, decide whether the assertions encode domain truth or just
whatever the implementation happened to return when the test was
written, and rewrite where needed.

## Domain invariants and edge cases

The algorithmic invariants the tests should encode, and the boundary
regimes where NW-flex has actually shipped bugs, are documented in
[tests/DOMAIN_GUIDE.md](tests/DOMAIN_GUIDE.md). That file is intended
to outlive this branch and become the durable domain-knowledge
reference for test authors. Test rewrites and new tests on this
branch should cite the relevant entry there.

## Merge considerations

This branch will eventually merge to `main` after `performance`
lands. Two practical rules to keep the merge clean:

1. **Touch tests, not sources.** Source-file conflicts with
   `performance` are the merge risk we want to avoid. Restrict edits
   to `tests/`, `tests/README.md`, and this file. If a real bug
   surfaces, capture it in an xfail with a pointer and let the fix
   land on `performance` (or on a dedicated fix branch).
2. **Stay current with `performance`.** Periodically rebase or merge
   `performance` into this branch so that tests are written against
   the latest simulation APIs (`build_mirror_frame`,
   `bwa_state_both_strands`, multi-block EP helpers). Tests written
   against an older API will silently rot.

## Plan

- [x] Port `tests/test_str_boundaries.py` from `nwflex_TNG`
      (stress-test branch, commit `19f9ad2`) — 8 tests covering the
      `e == n` terminal-EP family and interrupted-repeat phase
      semantics. Two assertions were updated for our codebase:
      expected CIGAR gained a trailing `1N` after the terminal-
      contraction CIGAR feature (`fc95192` on `main`), and the
      Python-vs-fast jump comparison switched from ordered list to
      set since jump-report order is implementation-defined.
- [x] Document the hand-trace-first principle and the "what breaks
      if removed" convention in `tests/DOMAIN_GUIDE.md`.
- [ ] Audit each agent-drafted test file and classify assertions as
      *domain-anchored*, *implementation-anchored*, or *trivial*.
- [ ] Rewrite implementation-anchored assertions against the
      invariants listed above.
- [ ] Add the missing edge cases.
- [ ] Wire `pytest --cov=nwflex` and record a baseline.
- [ ] Close coverage gaps where the missing path encodes a guarantee.
- [ ] Update `tests/README.md` to match the rewritten suite, using
      the "what breaks if removed" column convention.
