"""
Generic sweep harness for the NW-flex vs BWA-MEM comparison.

A sweep is the cartesian product of

    variant  x  read  x  method  x  orient

where the reference per ``(method, orient)`` is fixed across variants.
That fixed-reference property lets us flatten every variant's reads
into one list and run a single batched alignment per ``(method, orient)``.

The output is a tidy long-form DataFrame with one row per cell.
:func:`pivot_for_heatmap` reshapes it for
:func:`nwflex.simulation.viz.plot_correctness_heatmap`.
"""

from dataclasses import dataclass
from functools import reduce
from types import SimpleNamespace
from typing import Any, Iterable, List, Mapping, Sequence

import pandas as pd

from .core import (
    align_bwa,
    align_nwflex,
    alignment_state,
    alignment_state_multi,
    bwa_compound_truth_cigar,
    bwa_truth_cigar,
    combine_states,
    mirror_reads,
    nwflex_compound_truth_cigar,
    nwflex_truth_cigar,
    reverse_complement,
    score_alignment,
    tile_reads,
)


@dataclass
class SweepVariant:
    """One haplotype configuration evaluated by a sweep.

    Attributes
    ----------
    label : mapping
        Per-variant columns for the output rows (e.g., ``{"delta": -3}``
        or ``{"delta": -3, "snv": "T>A"}``).
    hap : object
        Haplotype-like object.  Must expose ``body_len``; methods may
        use it for ground-truth-CIGAR construction.
    reads : list
        Reads to evaluate against every method.  Order is preserved.
    """
    label: Mapping[str, Any]
    hap: Any
    reads: List[Any]


def make_variant(label, hap, *, read_len, k_min_flank,
                 target_lflanks=None, step=1):
    """Tile reads across ``hap`` and wrap the result in a
    :class:`SweepVariant`.

    Parameters
    ----------
    label : mapping
        Per-variant columns for the sweep output.
    hap : haplotype-like
        Passed straight through to :func:`tile_reads`.
    read_len, k_min_flank, step
        Tiling parameters; see :func:`tile_reads`.
    target_lflanks : iterable of int, optional
        If given, keep only reads whose ``lflank_extent`` is in this set.
    """
    reads = tile_reads(hap, read_len=read_len, k_min_flank=k_min_flank,
                       step=step)
    if target_lflanks is not None:
        target = set(target_lflanks)
        reads = [r for r in reads if r.lflank_extent in target]
    return SweepVariant(label=label, hap=hap, reads=reads)


def sweep(variants: Iterable[SweepVariant],
          methods: Iterable) -> pd.DataFrame:
    """
    Run every ``(variant, read, method, orient)`` cell and return a
    long-form DataFrame.

    Each ``method`` is expected to expose four attributes:

    - ``name`` — display label that lands in the ``method`` column.
    - ``run(reads, orient) -> list``
        Batched alignment over every read at once.  Must return one hit
        per input read, in input order.  The method is responsible for
        any orient-specific reference / mirror-read transformation.
    - ``truth(r, hap) -> float``
        NW score of the ground-truth alignment for this ``(read, hap)``
        pair.  Strand-symmetric, so it's computed once per cell and
        shared between the fwd and rc classifications.
    - ``classify(hit, r, truth_score, truth_z_bp, orient) -> str``
        State classification (``"P"``/``"T"``/``"M"``/``"D"``).

    Parameters
    ----------
    variants : iterable of SweepVariant
    methods : iterable

    Returns
    -------
    pandas.DataFrame
        Long-form: one row per cell, columns ``[*v.label.keys()],
        "lflank", "method", "orient", "state"``.
    """
    methods = list(methods)
    variants = list(variants)

    # 1. Flatten every variant's reads into one list, remembering each
    #    variant's slice so per-cell classification can walk back out.
    all_reads: List[Any] = []
    slices = []
    for v in variants:
        lo = len(all_reads)
        all_reads.extend(v.reads)
        slices.append((v, lo, lo + len(v.reads)))

    # 2. One batched alignment per (method, orient) across every read.
    hits = {
        (m.name, orient): m.run(all_reads, orient)
        for m in methods
        for orient in ("fwd", "rc")
    }

    # 3. Per-cell ground truth + classify.
    rows = []
    for v, lo, hi in slices:
        truth_z_bp = v.hap.body_len
        for i in range(lo, hi):
            r = all_reads[i]
            for m in methods:
                ts = m.truth(r, v.hap)
                for orient in ("fwd", "rc"):
                    state = m.classify(
                        hits[m.name, orient][i], r, ts,
                        truth_z_bp, orient,
                    )
                    rows.append({
                        **v.label,
                        "lflank": r.lflank_extent,
                        "method": m.name,
                        "orient": orient,
                        "state":  state,
                    })
    return pd.DataFrame(rows)


def pivot_for_heatmap(long_df: pd.DataFrame, *,
                      combine: str = "best") -> pd.DataFrame:
    """
    Reshape :func:`sweep` output for
    :func:`nwflex.simulation.viz.plot_correctness_heatmap`.

    The heatmap consumes a wide-by-orient table with columns
    ``arm``, ``fwd_state``, ``rc_state``, and a combined ``state``.
    This function

    - pivots ``orient`` from rows to columns (-> ``fwd_state`` /
      ``rc_state``),
    - adds a combined ``state`` column via :func:`combine_states`,
    - renames ``method`` -> ``arm`` to match the heatmap's vocabulary.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Output of :func:`sweep`.
    combine : {"best", "worst"}, default "best"
        Policy passed to :func:`combine_states`.
    """
    idx_cols = [c for c in long_df.columns if c not in {"orient", "state"}]
    wide = (long_df.pivot_table(
                index=idx_cols, columns="orient",
                values="state", aggfunc="first")
            .rename(columns={"fwd": "fwd_state", "rc": "rc_state"})
            .reset_index())
    wide["state"] = [
        combine_states(f, r, combine)
        for f, r in zip(wide["fwd_state"], wide["rc_state"])
    ]
    return wide.rename(columns={"method": "arm"})


def wrap_methods_for_multizone_truth(methods, variants):
    """Wrap a methods list so each ``classify`` call receives the variant
    haplotype's ``body_lens`` tuple instead of the scalar ``body_len``
    passed by :func:`sweep`.

    :func:`sweep` reads ``v.hap.body_len`` once per cell and threads it
    through every ``classify`` call as ``truth_z_bp``. For multi-zone
    haplotypes (e.g. :class:`CompoundHaplotype`) we need a per-block
    tuple instead. This wrapper rebinds each method's ``classify`` so it
    looks up the variant by read identity and reads ``body_lens`` off
    the haplotype directly, leaving ``run`` and ``truth`` untouched.
    """
    hap_for_read = {id(r): v.hap for v in variants for r in v.reads}
    wrapped = []
    for m in methods:
        base_classify = m.classify
        def make_classify(base):
            def classify(hit, r, truth_score, truth_z_bp, orient):
                hap = hap_for_read[id(r)]
                return base(hit, r, truth_score, hap.body_lens, orient)
            return classify
        wrapped.append(SimpleNamespace(
            name=m.name, run=m.run, truth=m.truth,
            classify=make_classify(base_classify),
        ))
    return wrapped


class BWAMethod:
    """BWA-MEM as a single-repeat sweep method.

    Aligns reads against the forward locus reference and against the
    mirror reference; classifies each hit using the scalar repeat zone
    and single ``truth_z_bp``.
    """

    def __init__(self, name, locus, rc_locus_X, rc_zone, *,
                 no_clip, score_kwargs):
        self.name = name
        self.locus = locus
        self.rc_locus_X = rc_locus_X
        self.rc_zone = rc_zone
        self.no_clip = no_clip
        self.score_kwargs = score_kwargs

    def run(self, reads, orient):
        if orient == "fwd":
            return align_bwa(self.locus.X, reads, no_clip=self.no_clip)
        return align_bwa(self.rc_locus_X, mirror_reads(reads),
                         no_clip=self.no_clip)

    def truth(self, read, hap):
        pos, cig = bwa_truth_cigar(read, hap, self.locus)
        return score_alignment(read.sequence, self.locus.X, pos, cig,
                               **self.score_kwargs)

    def classify(self, hit, read, truth_score, truth_z_bp, orient):
        if orient == "fwd":
            seq = read.sequence
            ref = self.locus.X
            zone = (self.locus.s, self.locus.e)
        else:
            seq = reverse_complement(read.sequence)
            ref = self.rc_locus_X
            zone = self.rc_zone
        chosen = (
            None if hit.cigar is None or hit.pos is None
            else score_alignment(seq, ref, hit.pos, hit.cigar,
                                 **self.score_kwargs)
        )
        return alignment_state(
            hit.cigar, hit.pos, chosen, truth_score, *zone, truth_z_bp,
            convention="bwa",
        )


def _to_dp_convention(score_kwargs: Mapping[str, Any]) -> dict:
    """Bridge stand-alone-open scoring (BWA / ``score_alignment``) to the
    NW-flex DP's subsumed-open convention.

    A length-L gap costs ``go + L*ge`` under convention A (stand-alone open,
    e.g. BWA's ``O + k*E``) and ``go + (L-1)*ge`` under convention B (the
    Gotoh recurrence the DP implements). Shifting ``gap_open`` by ``+ge``
    makes the DP charge the same effective per-gap cost as the convention-A
    scheme it was handed. See ``scripts/check_gap_conventions.py``.
    """
    return {**score_kwargs,
            "gap_open": score_kwargs["gap_open"] + score_kwargs["gap_extend"]}


class NWFlexMethod:
    """NW-flex as a single-repeat sweep method.

    Runs against the 3N-extended locus reference (and its mirror) with
    the STR-aware extra-predecessor pattern. NW-flex's hit score is the
    NW score by construction, so ``classify`` uses it directly without
    rescoring.

    ``score_kwargs`` is taken as stand-alone-open (convention A, the
    ``score_alignment`` / BWA convention). ``truth()`` calls
    ``score_alignment`` and uses it as-is; ``run()`` feeds the NW-flex DP
    (convention B) via :func:`_to_dp_convention`.
    """

    def __init__(self, name, nwflex_locus, rc_nwflex_X, rc_nwflex_zone,
                 ep_fwd, ep_rc, *, score_kwargs):
        self.name = name
        self.nwflex_locus = nwflex_locus
        self.rc_nwflex_X = rc_nwflex_X
        self.rc_nwflex_zone = rc_nwflex_zone
        self.ep_fwd = ep_fwd
        self.ep_rc = ep_rc
        self.score_kwargs = score_kwargs
        self._dp_score_kwargs = _to_dp_convention(score_kwargs)

    def run(self, reads, orient):
        if orient == "fwd":
            return align_nwflex(self.nwflex_locus.X, reads,
                                extra_predecessors=self.ep_fwd,
                                **self._dp_score_kwargs)
        return align_nwflex(self.rc_nwflex_X, mirror_reads(reads),
                            extra_predecessors=self.ep_rc,
                            **self._dp_score_kwargs)

    def truth(self, read, hap):
        pos, cig = nwflex_truth_cigar(read, hap, self.nwflex_locus)
        return score_alignment(read.sequence, self.nwflex_locus.X, pos, cig,
                               **self.score_kwargs)

    def classify(self, hit, read, truth_score, truth_z_bp, orient):
        zone = ((self.nwflex_locus.s, self.nwflex_locus.e) if orient == "fwd"
                else self.rc_nwflex_zone)
        return alignment_state(
            hit.cigar, hit.pos, float(hit.score), truth_score,
            *zone, truth_z_bp, convention="nwflex",
        )


class BWACompoundMethod:
    """BWA-MEM as a compound-repeat sweep method.

    Same shape as :class:`BWAMethod`, but uses the compound
    ground-truth-CIGAR builder and the multi-zone classifier. ``truth_z_bp`` here is the
    haplotype's ``body_lens`` tuple (one block length per repeat block);
    use :func:`wrap_methods_for_multizone_truth` to thread it through.
    """

    def __init__(self, name, compound, rc_X, rc_zones, *,
                 no_clip, score_kwargs):
        self.name = name
        self.compound = compound
        self.rc_X = rc_X
        self.rc_zones = rc_zones
        self.no_clip = no_clip
        self.score_kwargs = score_kwargs

    def run(self, reads, orient):
        if orient == "fwd":
            return align_bwa(self.compound.X, reads, no_clip=self.no_clip)
        return align_bwa(self.rc_X, mirror_reads(reads), no_clip=self.no_clip)

    def truth(self, read, hap):
        pos, cig = bwa_compound_truth_cigar(read, hap, self.compound)
        return score_alignment(read.sequence, self.compound.X, pos, cig,
                               **self.score_kwargs)

    def classify(self, hit, read, truth_score, truth_z_bps, orient):
        if orient == "fwd":
            seq = read.sequence
            ref = self.compound.X
            zones = self.compound.zones
        else:
            seq = reverse_complement(read.sequence)
            ref = self.rc_X
            zones = self.rc_zones
        chosen = (
            None if hit.cigar is None or hit.pos is None
            else score_alignment(seq, ref, hit.pos, hit.cigar,
                                 **self.score_kwargs)
        )
        return alignment_state_multi(
            hit.cigar, hit.pos, chosen, truth_score, zones, truth_z_bps,
            convention="bwa",
        )


class NWFlexCompoundMethod:
    """NW-flex as a compound-repeat sweep method.

    Aligns against the multi-block 3N-extended reference with a
    multi-STR extra-predecessor pattern (see
    :func:`build_EP_multi_STR_phase`). As with :class:`NWFlexMethod`,
    the hit's score is the NW score, so ``classify`` uses it directly.
    ``truth_z_bp`` is the haplotype's ``body_lens`` tuple; use
    :func:`wrap_methods_for_multizone_truth` to thread it through.

    ``score_kwargs`` is taken as stand-alone-open (convention A); the DP
    is fed the convention-B equivalent via :func:`_to_dp_convention`.
    ``truth()`` (which routes through ``score_alignment``) uses
    ``score_kwargs`` as-is.
    """

    def __init__(self, name, compound, rc_X_ext, rc_zones_ext,
                 ep_fwd, ep_rc, *, score_kwargs):
        self.name = name
        self.compound = compound
        self.rc_X_ext = rc_X_ext
        self.rc_zones_ext = rc_zones_ext
        self.ep_fwd = ep_fwd
        self.ep_rc = ep_rc
        self.score_kwargs = score_kwargs
        self._dp_score_kwargs = _to_dp_convention(score_kwargs)

    def run(self, reads, orient):
        if orient == "fwd":
            return align_nwflex(self.compound.X_ext, reads,
                                extra_predecessors=self.ep_fwd,
                                **self._dp_score_kwargs)
        return align_nwflex(self.rc_X_ext, mirror_reads(reads),
                            extra_predecessors=self.ep_rc,
                            **self._dp_score_kwargs)

    def truth(self, read, hap):
        pos, cig = nwflex_compound_truth_cigar(read, hap, self.compound)
        return score_alignment(read.sequence, self.compound.X_ext, pos, cig,
                               **self.score_kwargs)

    def classify(self, hit, read, truth_score, truth_z_bps, orient):
        zones = (self.compound.zones_ext if orient == "fwd"
                 else self.rc_zones_ext)
        return alignment_state_multi(
            hit.cigar, hit.pos, float(hit.score), truth_score,
            zones, truth_z_bps, convention="nwflex",
        )


def aggregate_per_cell(
    df: pd.DataFrame,
    groupby_cols: Sequence[str],
    *,
    strand_policy: str = "best",
    combine_policy: str = "best",
) -> pd.DataFrame:
    """Aggregate a wide-by-orient heatmap frame to one row per cell.

    For each group in ``groupby_cols`` (typically
    ``["arm", "delta1", "delta2"]`` or
    ``["bridge_len", "arm", "delta1", "delta2"]``), reduce ``fwd_state``
    and ``rc_state`` across reads under ``strand_policy``, then combine
    fwd/rc under ``combine_policy``.

    The output preserves ``groupby_cols`` and adds ``fwd_state``,
    ``rc_state``, ``state`` columns. Groups whose strand lists are empty
    are returned with all three columns ``None``.
    """
    rows = []
    for key, g in df.groupby(list(groupby_cols)):
        if not isinstance(key, tuple):
            key = (key,)
        fwds = g["fwd_state"].dropna().tolist()
        rcs  = g["rc_state"].dropna().tolist()
        if not fwds or not rcs:
            rows.append({
                **dict(zip(groupby_cols, key)),
                "fwd_state": None, "rc_state": None, "state": None,
            })
            continue
        f = reduce(lambda a, b: combine_states(a, b, strand_policy), fwds)
        r = reduce(lambda a, b: combine_states(a, b, strand_policy), rcs)
        rows.append({
            **dict(zip(groupby_cols, key)),
            "fwd_state": f, "rc_state": r,
            "state": combine_states(f, r, combine_policy),
        })
    return pd.DataFrame(rows)
