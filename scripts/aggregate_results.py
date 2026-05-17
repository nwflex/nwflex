"""Aggregate sweep shards into cross-locus figures + tidy CSVs.

Reads ``supplement/data/{single_repeat,compound}/*.csv`` written by
``scripts/run_batch_sweep.py``, then produces:

- **Compound cross-locus heatmaps** — one per ``(N1, N2)`` combination,
  using the existing
  :func:`nwflex.simulation.viz.plot_proportion_heatmap_2d_rows` with each
  bridge length as a row.  Each cell shows the proportion of motif
  pairs in which the chosen alignment has score = truth (states P+T),
  per strand.
- **Compound all-loci aggregate** — a single bridge-stacked heatmap
  pooling every motif pair × N-pair.
- **Single-repeat cross-locus heatmaps** — one per ``(N, snv_offset)``
  combination, using the local
  :func:`scripts.sweep_viz.plot_proportion_heatmap` on
  ``(Δ × lflank)`` axes.
- **Tidy aggregate CSVs** — per-cell fraction-by-state tables.

Usage::

    python scripts/aggregate_results.py --config scripts/configs/single_repeat.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from nwflex.simulation import (
    plot_proportion_heatmap_2d,
    plot_proportion_heatmap_2d_rows,
)

# Local 1-D continuous-color heatmap; kept out of the package until the
# in-flight viz refactor stabilizes.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_viz import (  # noqa: E402
    plot_proportion_heatmap,
    plot_proportion_heatmap_rows,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> Mapping[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _load_shards(data_dir: Path, kind: str) -> pd.DataFrame:
    files = sorted((data_dir / kind).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no {kind!r} shards in {data_dir / kind!s}"
        )
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _save_fig(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=144, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / name}.{{png,pdf}}")


_ARM_TITLES = {
    "BWA-std":     "BWA-MEM",
    "BWA-no-clip": "BWA-MEM (no clip)",
    "NW-flex":     "NW-flex",
}


# Manuscript-priority figure names (without extension).  After the full
# aggregate run writes every figure to ``figures_dir``, these are
# duplicated into the sibling ``<figures_dir>_priority`` folder so they
# are easy to find amongst the ~60 alternative stratifications.
_PRIORITY_FIGURE_NAMES = [
    "compound__all_loci__bridge_stack",
    "compound__N10_10__M3__motifLpair_stack",
]


def _compound_proportion_per_npair(
    df: pd.DataFrame, figures_dir: Path,
) -> None:
    """One bridge-stacked proportion heatmap per (N1, N2) combo,
    aggregating across the motif pairs in each cell."""
    for (N1, N2), sub in df.groupby(["N1", "N2"]):
        bridge_lengths = sorted(sub["bridge_len"].unique())
        deltas1 = sorted(sub["delta1"].unique())
        deltas2 = sorted(sub["delta2"].unique())
        n_pairs = sub.groupby("bridge_len")[
            ["pind1", "pind2"]
        ].apply(lambda g: g.drop_duplicates().shape[0]).iloc[0]
        fig = plot_proportion_heatmap_2d_rows(
            sub,
            deltas1=deltas1, deltas2=deltas2,
            row_values=bridge_lengths, row_col="bridge_len",
            arm_titles=_ARM_TITLES,
            row_label_fn=lambda m: f"|M| = {m} bp",
            suptitle=(f"Compound cross-locus fraction with score = truth "
                      f"— N₁={N1}, N₂={N2}"),
            subtitle=f"each cell aggregates {n_pairs} motif pair(s)",
            cbar_label="fraction of motif pairs with score = truth",
        )
        _save_fig(fig, figures_dir,
                  f"compound__N{N1:02d}_{N2:02d}__bridge_stack")


def _compound_proportion_all_loci(
    df: pd.DataFrame, figures_dir: Path,
) -> None:
    """Bridge-stacked heatmap pooling every motif pair at the symmetric
    N-pairs (``N1 == N2``); asymmetric N-pair loci are excluded so the
    pooled view does not commingle two distinct count regimes."""
    df = df[df["N1"] == df["N2"]]
    if df.empty:
        return
    bridge_lengths = sorted(df["bridge_len"].unique())
    deltas1 = sorted(df["delta1"].unique())
    deltas2 = sorted(df["delta2"].unique())
    n_loci = df[["pind1", "pind2", "N1", "N2"]].drop_duplicates().shape[0]
    fig = plot_proportion_heatmap_2d_rows(
        df,
        deltas1=deltas1, deltas2=deltas2,
        row_values=bridge_lengths, row_col="bridge_len",
        arm_titles=_ARM_TITLES,
        row_label_fn=lambda m: f"|M| = {m} bp",
        suptitle=("Compound cross-locus fraction with score = truth — "
                  "symmetric N-pairs"),
        subtitle=f"each cell aggregates {n_loci} (motif pair × N-pair) loci",
        cbar_label="fraction of loci with score = truth",
    )
    _save_fig(fig, figures_dir, "compound__all_loci__bridge_stack")


def _single_repeat_proportion_per_n(
    df: pd.DataFrame, figures_dir: Path,
) -> None:
    """One ``(Δ × lflank)`` proportion heatmap per ``(N, snv_offset)``
    combination, pooling all selected loci of that combination."""
    deltas  = sorted(df["delta"].unique())
    lflanks = sorted(df["lflank"].unique())
    for (N, snv_offset), sub in df.groupby(["N", "snv_offset"]):
        n_loci = sub["pind"].nunique()
        if snv_offset == -1:
            snv_label = "no SNV"
            snv_tag = "noSNV"
        else:
            snv_label = f"SNV @ {int(snv_offset) + 1}"
            snv_tag = f"SNV{int(snv_offset):+d}"
        fig = plot_proportion_heatmap(
            sub,
            deltas=deltas, lflanks=lflanks,
            arm_titles=_ARM_TITLES,
            suptitle=(f"Single-repeat cross-locus fraction with score = truth"
                      f" — N={N}, {snv_label}"),
            subtitle=f"each cell aggregates {n_loci} locus/loci",
            cbar_label="fraction of loci with score = truth",
        )
        _save_fig(fig, figures_dir,
                  f"single__N{int(N):02d}__{snv_tag}")


def _single_repeat_proportion_snv_stack(
    df: pd.DataFrame, figures_dir: Path,
) -> None:
    """SNV-offset-stacked heatmap (one row per SNV position) at a fixed
    N.  Mirrors NB7's first-and-second-comparison figure structure."""
    deltas  = sorted(df["delta"].unique())
    lflanks = sorted(df["lflank"].unique())
    for N, sub_n in df.groupby("N"):
        present = sub_n["snv_offset"].unique()
        offsets = sorted([o for o in present if o != -1])
        if -1 in present:
            offsets = offsets + [-1]
        rows = [(off, sub_n[sub_n["snv_offset"] == off]) for off in offsets]
        n_loci = sub_n["pind"].nunique()

        def _row_label(off):
            return "no SNV" if off == -1 else f"SNV @ {int(off) + 1}"

        fig = plot_proportion_heatmap_rows(
            rows,
            deltas=deltas, lflanks=lflanks,
            arm_titles=_ARM_TITLES,
            row_label_fn=_row_label,
            suptitle=(f"Single-repeat — fraction with score = truth, "
                      f"N={N}"),
            subtitle=f"each cell aggregates {n_loci} loci",
            cbar_label="fraction of loci with score = truth",
        )
        _save_fig(fig, figures_dir,
                  f"single__N{int(N):02d}__snv_stack")


def _single_repeat_proportion_by_N(
    df: pd.DataFrame, figures_dir: Path,
) -> None:
    """At each SNV offset, stack rows by ``N`` value so the variation in
    BWA's recovery as a function of repeat count is visible in a single
    figure."""
    deltas  = sorted(df["delta"].unique())
    lflanks = sorted(df["lflank"].unique())
    n_values = sorted(df["N"].unique())
    for snv_offset, sub in df.groupby("snv_offset"):
        rows = [(N, sub[sub["N"] == N]) for N in n_values]
        n_loci = sub["pind"].nunique()
        snv_label = "no SNV" if snv_offset == -1 else f"SNV @ {int(snv_offset) + 1}"
        snv_tag = "noSNV" if snv_offset == -1 else f"SNV{int(snv_offset):+d}"

        fig = plot_proportion_heatmap_rows(
            rows,
            deltas=deltas, lflanks=lflanks,
            arm_titles=_ARM_TITLES,
            row_label_fn=lambda N: f"N = {N}",
            suptitle=(f"Single-repeat — fraction with score = truth, "
                      f"by repeat count ({snv_label})"),
            subtitle=f"each cell aggregates {n_loci} loci",
            cbar_label="fraction of loci with score = truth",
        )
        _save_fig(fig, figures_dir, f"single__{snv_tag}__N_stack")


def _single_repeat_proportion_by_motif_length(
    df: pd.DataFrame, figures_dir: Path,
) -> None:
    """At each ``(N, snv_offset)``, stack rows by motif length.  Useful
    when motif length is the discriminator (e.g., 1-mers vs 3-mers
    behave differently under boundary SNVs)."""
    deltas  = sorted(df["delta"].unique())
    lflanks = sorted(df["lflank"].unique())
    motif_lengths = sorted(df["motif_len"].unique())
    for (N, snv_offset), sub in df.groupby(["N", "snv_offset"]):
        rows = [(L, sub[sub["motif_len"] == L]) for L in motif_lengths]
        snv_label = "no SNV" if snv_offset == -1 else f"SNV @ {int(snv_offset) + 1}"
        snv_tag = "noSNV" if snv_offset == -1 else f"SNV{int(snv_offset):+d}"

        fig = plot_proportion_heatmap_rows(
            rows,
            deltas=deltas, lflanks=lflanks,
            arm_titles=_ARM_TITLES,
            row_label_fn=lambda L: f"motif L = {L}",
            suptitle=(f"Single-repeat — fraction with score = truth, "
                      f"by motif length (N={int(N)}, {snv_label})"),
            subtitle=("each row aggregates the loci of one motif "
                      "length"),
            cbar_label="fraction of loci with score = truth",
        )
        _save_fig(fig, figures_dir,
                  f"single__N{int(N):02d}__{snv_tag}__motifL_stack")


def _compound_proportion_by_motif_length_pair(
    df: pd.DataFrame, figures_dir: Path,
) -> None:
    """At each ``(N1, N2)``, stack rows by motif-length pair (six
    combinations of L1, L2 in ``{1, 2, 3}``).  Shows how the BWA
    failure region depends on motif-length pair within a fixed N-pair
    configuration.

    For brevity, fixes the bridge length to a "boundary" value (the
    smallest tested).  A motif-length-pair × bridge cross would be a
    separate figure if needed.
    """
    deltas1 = sorted(df["delta1"].unique())
    deltas2 = sorted(df["delta2"].unique())
    bridge_lengths = sorted(df["bridge_len"].unique())

    for (N1, N2), sub_n in df.groupby(["N1", "N2"]):
        for bridge_len in bridge_lengths:
            sub = sub_n[sub_n["bridge_len"] == bridge_len].copy()
            if sub.empty:
                continue
            # Synthetic composite column so plot_proportion_heatmap_2d_rows
            # can filter by a single key.
            sub["_motifL_pair"] = list(
                zip(sub["motif1_len"], sub["motif2_len"])
            )
            length_pairs = sorted(
                sub[["motif1_len", "motif2_len"]]
                   .drop_duplicates()
                   .itertuples(index=False, name=None)
            )
            n_per_row = [
                sub[sub["_motifL_pair"] == (L1, L2)]
                   [["pind1", "pind2"]].drop_duplicates().shape[0]
                for (L1, L2) in length_pairs
            ]
            fig = plot_proportion_heatmap_2d_rows(
                sub,
                deltas1=deltas1, deltas2=deltas2,
                row_values=length_pairs, row_col="_motifL_pair",
                arm_titles=_ARM_TITLES,
                row_label_fn=lambda L: f"$(L_1, L_2)$ = {L}",
                suptitle=(f"Compound — fraction with score = truth, "
                          f"by motif-length pair "
                          f"(N₁={N1}, N₂={N2}, |M|={bridge_len})"),
                subtitle=(f"rows aggregate {min(n_per_row)}–"
                          f"{max(n_per_row)} motif pairs each"),
                cbar_label="fraction of motif pairs with score = truth",
            )
            _save_fig(fig, figures_dir,
                      f"compound__N{N1:02d}_{N2:02d}__M{bridge_len}"
                      f"__motifLpair_stack")


def _per_cell_fracs(g: pd.DataFrame) -> pd.Series:
    """Per-cell fraction-by-state for combined verdict + each strand."""
    out = {"n_loci": len(g)}
    # Combined-strand verdict (the cell-level "best of fwd/rc")
    for s in ("P", "T", "M", "D"):
        out[f"frac_{s}"] = (g["state"] == s).mean()
    out["frac_score_eq_truth"] = g["state"].isin(["P", "T"]).mean()
    # Per-strand verdicts
    for strand in ("fwd", "rc"):
        col = f"{strand}_state"
        for s in ("P", "T", "M", "D"):
            out[f"frac_{s}_{strand}"] = (g[col] == s).mean()
        out[f"frac_score_eq_truth_{strand}"] = g[col].isin(["P", "T"]).mean()
    return pd.Series(out)


def _single_repeat_tidy_aggregate(
    df: pd.DataFrame, data_dir: Path,
) -> None:
    """Per-cell fraction-by-state table for the single-repeat side,
    keyed by ``(N, snv_offset, motif_len, arm, delta, lflank)`` so a
    cross-locus view can be stratified by motif length."""
    keys = ["N", "snv_offset", "motif_len", "arm", "delta", "lflank"]
    agg = (df.groupby(keys, dropna=False)
             .apply(_per_cell_fracs, include_groups=False)
             .reset_index())
    out_path = data_dir / "single_repeat_cross_locus_aggregate.csv"
    agg.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")


def _compound_tidy_aggregate(
    df: pd.DataFrame, data_dir: Path,
) -> None:
    """Per-cell fraction-by-state table keyed by
    ``(N1, N2, bridge_len, motif1_len, motif2_len, arm, delta1, delta2)``
    so a cross-locus view can be stratified by either motif length."""
    keys = ["N1", "N2", "bridge_len", "motif1_len", "motif2_len",
            "arm", "delta1", "delta2"]
    agg = (df.groupby(keys, dropna=False)
             .apply(_per_cell_fracs, include_groups=False)
             .reset_index())
    out_path = data_dir / "compound_cross_locus_aggregate.csv"
    agg.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = (REPO_ROOT / config["output"]["data_dir"]).resolve()
    figures_dir = (REPO_ROOT / config["output"]["figures_dir"]).resolve()

    print(f"Reading shards from {data_dir}")

    # Compound
    try:
        cmp_df = _load_shards(data_dir, "compound")
        print(f"  compound:      {len(cmp_df)} rows from "
              f"{cmp_df[['pind1','pind2','N1','N2','bridge_len']].drop_duplicates().shape[0]} task shards")
    except FileNotFoundError:
        cmp_df = None
        print("  compound:      no shards (skipping compound aggregation)")

    # Single repeat
    try:
        sr_df = _load_shards(data_dir, "single_repeat")
        print(f"  single-repeat: {len(sr_df)} rows from "
              f"{sr_df[['pind','N','snv_offset']].drop_duplicates().shape[0]} task shards")
    except FileNotFoundError:
        sr_df = None
        print("  single-repeat: no shards (skipping single-repeat aggregation)")

    print(f"\nWriting figures to {figures_dir}")
    if cmp_df is not None:
        _compound_proportion_per_npair(cmp_df, figures_dir)
        _compound_proportion_all_loci(cmp_df, figures_dir)
    if sr_df is not None:
        _single_repeat_proportion_per_n(sr_df, figures_dir)
        _single_repeat_proportion_snv_stack(sr_df, figures_dir)
        _single_repeat_proportion_by_N(sr_df, figures_dir)
        _single_repeat_proportion_by_motif_length(sr_df, figures_dir)
    if cmp_df is not None:
        _compound_proportion_by_motif_length_pair(cmp_df, figures_dir)

    print(f"\nWriting tidy aggregate CSVs to {data_dir}")
    if cmp_df is not None:
        _compound_tidy_aggregate(cmp_df, data_dir)
    if sr_df is not None:
        _single_repeat_tidy_aggregate(sr_df, data_dir)

    priority_dir = figures_dir.parent / (figures_dir.name + "_priority")
    _copy_priority_figures(figures_dir, priority_dir)


def _copy_priority_figures(
    figures_dir: Path, priority_dir: Path,
) -> None:
    import shutil
    priority_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in _PRIORITY_FIGURE_NAMES:
        for ext in ("png", "pdf"):
            src = figures_dir / f"{name}.{ext}"
            if src.exists():
                shutil.copy2(src, priority_dir / src.name)
                n += 1
    print(f"\nCopied {n} priority files to {priority_dir}")


if __name__ == "__main__":
    main()
