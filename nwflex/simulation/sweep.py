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
from typing import Any, Iterable, List, Mapping

import pandas as pd

from .core import combine_states


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
        use it for truth-CIGAR construction.
    reads : list
        Reads to evaluate against every method.  Order is preserved.
    """
    label: Mapping[str, Any]
    hap: Any
    reads: List[Any]


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
        NW score of the truth alignment for this ``(read, hap)`` pair.
        Strand-symmetric, so it's computed once per cell and shared
        between the fwd and rc classifications.
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

    # 3. Per-cell truth + classify.
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
