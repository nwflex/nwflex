"""Descriptive statistics tables for the comprehensive sweep.

Reads the compound shards written by ``scripts/run_batch_sweep.py``
and produces a handful of focused tables, each emitted as both CSV
(for downstream analysis) and LaTeX (for paste-in supplement use):

- **recovery_by_bridge** — per-arm × bridge-length, the mean and
  quartile fractions of cells with ``state == "P"``, and the cell-pass
  rate (fraction of cells reaching ≥95% locus agreement).
- **bridge_breakpoint** — for each arm, the smallest ``|M|`` at which
  every cell crosses each of {50%, 80%, 95%, 99%} score-equals-truth
  agreement across loci.
- **strand_asymmetry** — per arm, the fraction of (locus, cell) pairs
  whose fwd verdict differs from the rc verdict (i.e. BWA's
  seed-and-extend direction-dependence rate).
- **motif_length_breakdown** — for compound: per-arm × motif-length-
  pair (L1, L2), the mean fraction of cells where score equals truth.

The single-repeat cross-locus tables are restricted to trinucleotide
(3-mer) loci — the primary focus of this analysis — so a pooled
headline number is not silently weighted by the panel's motif-length
composition (400 / 1,080 / 5,420 loci at length 1 / 2 / 3).
``single_motif_length_breakdown`` is the exception, kept unfiltered as
the deliberate motif-length view.

Usage::

    python scripts/build_stats_tables.py --config scripts/configs/main.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent

# Single-repeat cross-locus tables are reported on the trinucleotide
# slice — see the module docstring.  _single_motif_length_breakdown is
# the deliberate exception (kept unfiltered).
SINGLE_MOTIF_LEN = 3


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


def _load_compound_shards(data_dir: Path) -> pd.DataFrame:
    return _load_shards(data_dir, "compound")


def _write_table(df: pd.DataFrame, out_dir: Path, name: str,
                 *, caption: str, label: str,
                 float_format: str = "%.3f") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    tex_path = out_dir / f"{name}.tex"
    df.to_csv(csv_path, index=False)
    with tex_path.open("w") as f:
        f.write(df.to_latex(
            index=False,
            float_format=float_format,
            caption=caption,
            label=label,
            escape=False,
        ))
    print(f"  wrote {csv_path.relative_to(REPO_ROOT)} + "
          f"{tex_path.relative_to(REPO_ROOT)}")


def _recovery_by_bridge(df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm × |M|, fraction-of-cells-in-{P,T} across loci, plus
    quartiles and the cell-pass rate at the 95% threshold."""
    # First: per (arm, bridge_len, delta1, delta2), fraction of loci
    # where state in {P, T}.
    per_cell = (df.groupby(["arm", "bridge_len", "delta1", "delta2"])["state"]
                  .apply(lambda s: s.isin(["P", "T"]).mean())
                  .reset_index(name="frac_score_eq_truth"))

    def _summarize(g):
        return pd.Series({
            "n_cells": len(g),
            "mean_frac_score_eq_truth": g["frac_score_eq_truth"].mean(),
            "p25_frac_score_eq_truth":  g["frac_score_eq_truth"].quantile(0.25),
            "median_frac_score_eq_truth": g["frac_score_eq_truth"].quantile(0.50),
            "p75_frac_score_eq_truth":  g["frac_score_eq_truth"].quantile(0.75),
            "cells_at_100pct":  (g["frac_score_eq_truth"] >= 0.999).mean(),
            "cells_at_95pct":   (g["frac_score_eq_truth"] >= 0.95).mean(),
        })

    return (per_cell.groupby(["arm", "bridge_len"])
              .apply(_summarize, include_groups=False)
              .reset_index())


def _bridge_breakpoint(df: pd.DataFrame) -> pd.DataFrame:
    """For each arm, the smallest |M| at which the WORST cell crosses
    each agreement threshold (50%, 80%, 95%, 99%, 100%).  Returns NaN
    if no bridge length tested clears the threshold for that arm."""
    per_cell = (df.groupby(["arm", "bridge_len", "delta1", "delta2"])["state"]
                  .apply(lambda s: s.isin(["P", "T"]).mean())
                  .reset_index(name="frac"))
    bridges = sorted(per_cell["bridge_len"].unique())
    rows = []
    for arm, sub in per_cell.groupby("arm"):
        row = {"arm": arm}
        for thresh in (0.50, 0.80, 0.95, 0.99, 1.00):
            hit = None
            for m in bridges:
                cells = sub[sub["bridge_len"] == m]["frac"]
                if (cells >= thresh - 1e-9).all():
                    hit = m
                    break
            row[f"min_M_at_{int(thresh*100)}pct"] = (
                int(hit) if hit is not None else pd.NA
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _strand_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    """Per arm, fraction of (locus, cell) pairs whose fwd verdict
    differs from the rc verdict.  Excludes NaN rows.  A locus is the
    (pind1, pind2, N1, N2) tuple here."""
    rows = []
    for arm, sub in df.groupby("arm"):
        valid = sub.dropna(subset=["fwd_state", "rc_state"])
        n_total = len(valid)
        n_diff  = (valid["fwd_state"] != valid["rc_state"]).sum()
        n_fwd_only = ((valid["fwd_state"].isin(["P", "T"]))
                      & (~valid["rc_state"].isin(["P", "T"]))).sum()
        n_rc_only  = ((~valid["fwd_state"].isin(["P", "T"]))
                      & (valid["rc_state"].isin(["P", "T"]))).sum()
        rows.append({
            "arm": arm,
            "n_locus_cells": n_total,
            "frac_strands_disagree": n_diff / n_total if n_total else 0.0,
            "frac_fwd_only_correct": n_fwd_only / n_total if n_total else 0.0,
            "frac_rc_only_correct":  n_rc_only / n_total if n_total else 0.0,
        })
    return pd.DataFrame(rows)


def _motif_length_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per arm × (motif1_len, motif2_len), mean cell-pass rate
    across loci (averaged over all bridge lengths and Δ-cells)."""
    df = df.copy()
    df["score_eq_truth"] = df["state"].isin(["P", "T"]).astype(float)
    return (df.groupby(["arm", "motif1_len", "motif2_len"])["score_eq_truth"]
              .mean()
              .reset_index()
              .rename(columns={"score_eq_truth":
                               "mean_frac_score_eq_truth"}))


# ---------------------------------------------------------------------------
# Single-repeat tables
# ---------------------------------------------------------------------------

def _single_recovery_by_n_snv(df: pd.DataFrame) -> pd.DataFrame:
    """Per arm × N × snv_offset, summary of fraction of locus-cells
    with score equal to truth."""
    per_cell = (df.groupby(["arm", "N", "snv_offset", "delta", "lflank"])["state"]
                  .apply(lambda s: s.isin(["P", "T"]).mean())
                  .reset_index(name="frac_score_eq_truth"))

    def _summarize(g):
        return pd.Series({
            "n_cells": len(g),
            "mean_frac_score_eq_truth": g["frac_score_eq_truth"].mean(),
            "p25_frac_score_eq_truth":  g["frac_score_eq_truth"].quantile(0.25),
            "median_frac_score_eq_truth": g["frac_score_eq_truth"].quantile(0.50),
            "p75_frac_score_eq_truth":  g["frac_score_eq_truth"].quantile(0.75),
            "cells_at_100pct":  (g["frac_score_eq_truth"] >= 0.999).mean(),
            "cells_at_95pct":   (g["frac_score_eq_truth"] >= 0.95).mean(),
        })

    return (per_cell.groupby(["arm", "N", "snv_offset"])
              .apply(_summarize, include_groups=False)
              .reset_index())


def _single_strand_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    """Per arm × snv_offset, fraction of (locus, delta, lflank) cells
    whose fwd verdict differs from the rc verdict."""
    rows = []
    for (arm, snv), sub in df.groupby(["arm", "snv_offset"]):
        valid = sub.dropna(subset=["fwd_state", "rc_state"])
        n_total = len(valid)
        n_diff  = (valid["fwd_state"] != valid["rc_state"]).sum()
        n_fwd_only = ((valid["fwd_state"].isin(["P", "T"]))
                      & (~valid["rc_state"].isin(["P", "T"]))).sum()
        n_rc_only  = ((~valid["fwd_state"].isin(["P", "T"]))
                      & (valid["rc_state"].isin(["P", "T"]))).sum()
        rows.append({
            "arm": arm,
            "snv_offset": snv,
            "n_locus_cells": n_total,
            "frac_strands_disagree": n_diff / n_total if n_total else 0.0,
            "frac_fwd_only_correct": n_fwd_only / n_total if n_total else 0.0,
            "frac_rc_only_correct":  n_rc_only / n_total if n_total else 0.0,
        })
    return pd.DataFrame(rows)


def _single_motif_length_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per arm × motif_len × snv_offset, mean fraction of cells where
    score equals truth, averaged across loci of that motif length."""
    df = df.copy()
    df["score_eq_truth"] = df["state"].isin(["P", "T"]).astype(float)
    return (df.groupby(["arm", "motif_len", "snv_offset"])["score_eq_truth"]
              .mean()
              .reset_index()
              .rename(columns={"score_eq_truth":
                               "mean_frac_score_eq_truth"}))


def _single_per_locus_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per ``(arm, N, snv_offset)``, distribution of the per-locus
    fraction of cells in {P, T}, across selected loci.

    Each row aggregates ``n_loci`` per-locus values; quantiles describe
    the spread, and the ``frac_loci_at_*`` columns answer "what fraction
    of loci recover the truth on at least X% of cells?"
    """
    per_locus = (df.groupby(["arm", "N", "snv_offset", "pind"])["state"]
                   .apply(lambda s: s.isin(["P", "T"]).mean())
                   .reset_index(name="frac_cells_in_PT"))

    def _summarize(g):
        return pd.Series({
            "n_loci": len(g),
            "mean":   g["frac_cells_in_PT"].mean(),
            "p10":    g["frac_cells_in_PT"].quantile(0.10),
            "p25":    g["frac_cells_in_PT"].quantile(0.25),
            "median": g["frac_cells_in_PT"].quantile(0.50),
            "p75":    g["frac_cells_in_PT"].quantile(0.75),
            "p90":    g["frac_cells_in_PT"].quantile(0.90),
            "frac_loci_at_100pct": (g["frac_cells_in_PT"] >= 0.999).mean(),
            "frac_loci_at_95pct":  (g["frac_cells_in_PT"] >= 0.95).mean(),
            "frac_loci_at_50pct":  (g["frac_cells_in_PT"] >= 0.50).mean(),
        })

    return (per_locus.groupby(["arm", "N", "snv_offset"])
              .apply(_summarize, include_groups=False)
              .reset_index())


def _nwflex_t_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """For NW-flex single-repeat: per ``(N, snv_offset, delta, lflank)``,
    fraction of loci hitting ``T`` (tied with truth on score, wrong on
    length) or ``D`` (truth outscored — the unique NW-flex failure).
    Filtered to cells where at least one locus is non-P, so the table
    is short and surfaces just the problematic cells.
    """
    sub = df[df["arm"] == "NW-flex"].copy()
    sub["is_T"] = (sub["state"] == "T").astype(float)
    sub["is_D"] = (sub["state"] == "D").astype(float)
    agg = (sub.groupby(["N", "snv_offset", "delta", "lflank"])
              .agg(n_loci=("state", "count"),
                   frac_T=("is_T", "mean"),
                   frac_D=("is_D", "mean"))
              .reset_index())
    # Only surface cells where NW-flex fails on at least one locus.
    return agg[(agg["frac_T"] > 0) | (agg["frac_D"] > 0)].copy()


# ---------------------------------------------------------------------------
# Compound per-locus distribution
# ---------------------------------------------------------------------------

def _compound_per_locus_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per ``(arm, N1, N2, bridge_len)``, distribution of the per-locus
    fraction of (Δ1, Δ2) cells in {P, T}, across motif-pair loci."""
    per_locus = (df.groupby(["arm", "N1", "N2", "bridge_len",
                             "pind1", "pind2"])["state"]
                   .apply(lambda s: s.isin(["P", "T"]).mean())
                   .reset_index(name="frac_cells_in_PT"))

    def _summarize(g):
        return pd.Series({
            "n_loci": len(g),
            "mean":   g["frac_cells_in_PT"].mean(),
            "p10":    g["frac_cells_in_PT"].quantile(0.10),
            "p25":    g["frac_cells_in_PT"].quantile(0.25),
            "median": g["frac_cells_in_PT"].quantile(0.50),
            "p75":    g["frac_cells_in_PT"].quantile(0.75),
            "p90":    g["frac_cells_in_PT"].quantile(0.90),
            "frac_loci_at_100pct": (g["frac_cells_in_PT"] >= 0.999).mean(),
            "frac_loci_at_95pct":  (g["frac_cells_in_PT"] >= 0.95).mean(),
            "frac_loci_at_50pct":  (g["frac_cells_in_PT"] >= 0.50).mean(),
        })

    return (per_locus.groupby(["arm", "N1", "N2", "bridge_len"])
              .apply(_summarize, include_groups=False)
              .reset_index())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = (REPO_ROOT / config["output"]["data_dir"]).resolve()
    tables_dir = (REPO_ROOT / config["output"]["tables_dir"]).resolve()

    print(f"Reading shards from {data_dir}")
    try:
        df = _load_compound_shards(data_dir)
        n_loci = df[["pind1", "pind2", "N1", "N2"]].drop_duplicates().shape[0]
        print(f"  compound:      {len(df)} rows from {n_loci} (motif-pair × N-pair) loci")
    except FileNotFoundError:
        df = None
        print("  compound:      no shards (skipping compound tables)")

    try:
        sr_df = _load_shards(data_dir, "single_repeat")
        n_sr_loci = sr_df[["pind", "N", "snv_offset"]].drop_duplicates().shape[0]
        print(f"  single-repeat: {len(sr_df)} rows from "
              f"{n_sr_loci} (locus × N × SNV) loci")
    except FileNotFoundError:
        sr_df = None
        print("  single-repeat: no shards (skipping single-repeat tables)")

    print(f"\nWriting tables to {tables_dir}")

    if df is None:
        return _maybe_single(sr_df, tables_dir)

    recovery = _recovery_by_bridge(df)
    _write_table(
        recovery, tables_dir, "compound_recovery_by_bridge",
        caption=("Compound cross-locus recovery, per arm and bridge "
                 "length: mean and quartile fractions of loci with "
                 "score equal to truth, and the fraction of cells "
                 "reaching 95\\% / 100\\% agreement."),
        label="tab:compound-recovery-by-bridge",
    )

    breakpoint_ = _bridge_breakpoint(df)
    _write_table(
        breakpoint_, tables_dir, "compound_bridge_breakpoint",
        caption=("Smallest bridge length $|M|$ at which every "
                 "$(\\Delta_1, \\Delta_2)$ cell crosses the given "
                 "score-equals-truth agreement threshold."),
        label="tab:compound-bridge-breakpoint",
        float_format="%.0f",
    )

    asymmetry = _strand_asymmetry(df)
    _write_table(
        asymmetry, tables_dir, "compound_strand_asymmetry",
        caption=("Per-arm strand-asymmetry rate across all "
                 "(locus, cell) combinations: fraction whose forward "
                 "and reverse-complement verdicts disagree, broken "
                 "out by which strand carried truth alone."),
        label="tab:compound-strand-asymmetry",
    )

    motif_breakdown = _motif_length_breakdown(df)
    _write_table(
        motif_breakdown, tables_dir, "compound_motif_length_breakdown",
        caption=("Mean fraction of locus cells in which score equals "
                 "truth, stratified by motif-length pair $(L_1, L_2)$ "
                 "and aligner."),
        label="tab:compound-motif-length-breakdown",
    )

    per_locus_compound = _compound_per_locus_distribution(df)
    _write_table(
        per_locus_compound, tables_dir, "compound_per_locus_distribution",
        caption=("Per-locus distribution of the fraction of "
                 "$(\\Delta_1, \\Delta_2)$ cells where score equals "
                 "truth, taken across motif-pair loci.  "
                 "``frac\\_loci\\_at\\_*`` answers \"what fraction of "
                 "loci hit at least this success rate?\""),
        label="tab:compound-per-locus-distribution",
    )

    print("\nSummary preview:")
    print("\n[compound recovery by bridge]")
    print(recovery.to_string(index=False, float_format="%.3f"))
    print("\n[compound bridge breakpoint]")
    print(breakpoint_.to_string(index=False))
    print("\n[compound strand asymmetry]")
    print(asymmetry.to_string(index=False, float_format="%.3f"))

    _maybe_single(sr_df, tables_dir)


def _maybe_single(sr_df, tables_dir: Path) -> None:
    if sr_df is None:
        return

    # Tables display the SNV position 1-indexed from the repeat
    # boundary; the underlying CSV column is 0-indexed (k=0 means the
    # base immediately adjacent to the boundary).  Sentinel -1 ("no
    # SNV") is preserved.
    sr_df = sr_df.assign(
        snv_offset=sr_df["snv_offset"].where(
            sr_df["snv_offset"] == -1, sr_df["snv_offset"] + 1
        )
    )

    # The single-repeat panel is 400 / 1,080 / 5,420 loci at motif
    # length 1 / 2 / 3, so any statistic pooled across motif length is
    # ~79% trinucleotide by construction.  Rather than reweight, the
    # cross-locus tables below are reported on the trinucleotide slice.
    # _single_motif_length_breakdown
    # is the deliberate exception: it stays unfiltered as the
    # motif-length view.
    sr_di = sr_df[sr_df["motif_len"] == SINGLE_MOTIF_LEN]

    sr_recovery = _single_recovery_by_n_snv(sr_di)
    _write_table(
        sr_recovery, tables_dir, "single_recovery_by_N_snv",
        caption=("Single-repeat cross-locus recovery, per arm × $N$ × "
                 "SNV offset.  ``snv\\_offset = -1`` means no SNV.  "
                 "Restricted to trinucleotide (3-mer) loci."),
        label="tab:single-recovery-by-N-snv",
    )

    sr_asym = _single_strand_asymmetry(sr_di)
    _write_table(
        sr_asym, tables_dir, "single_strand_asymmetry",
        caption=("Single-repeat per-arm strand-asymmetry rate, broken "
                 "out by SNV offset.  Trinucleotide (3-mer) loci only."),
        label="tab:single-strand-asymmetry",
    )

    sr_motif = _single_motif_length_breakdown(sr_df)
    _write_table(
        sr_motif, tables_dir, "single_motif_length_breakdown",
        caption=("Mean fraction of locus cells in which score equals "
                 "truth, stratified by motif length and SNV offset.  "
                 "This table spans all motif lengths (1--3); the other "
                 "single-repeat tables are restricted to trinucleotides."),
        label="tab:single-motif-length-breakdown",
    )

    sr_per_locus = _single_per_locus_distribution(sr_di)
    _write_table(
        sr_per_locus, tables_dir, "single_per_locus_distribution",
        caption=("Per-locus distribution of the fraction of "
                 "$(\\Delta, \\text{lflank})$ cells where score equals "
                 "truth, across selected loci.  ``frac\\_loci\\_at\\_*"
                 "`` answers \"what fraction of loci hit at least this "
                 "success rate?\"  Trinucleotide (3-mer) loci only."),
        label="tab:single-per-locus-distribution",
    )

    nwflex_t = _nwflex_t_breakdown(sr_di)
    _write_table(
        nwflex_t, tables_dir, "single_nwflex_T_breakdown",
        caption=("NW-flex failures in the single-repeat sweep: each "
                 "row is one $(\\Delta, \\text{lflank})$ cell where at "
                 "least one locus did not land in state P, broken out "
                 "by SNV offset.  ``frac\\_T`` and ``frac\\_D`` are the "
                 "fraction of loci in the tied and dominated state "
                 "respectively.  Trinucleotide (3-mer) loci only."),
        label="tab:single-nwflex-T-breakdown",
    )

    print("\n[single-repeat recovery (NW-flex only, for brevity)]")
    print(sr_recovery[sr_recovery["arm"] == "NW-flex"]
          .to_string(index=False, float_format="%.3f"))
    print("\n[single-repeat strand asymmetry]")
    print(sr_asym.to_string(index=False, float_format="%.3f"))
    print("\n[NW-flex T-breakdown (first 20 problem cells)]")
    print(nwflex_t.head(20).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
