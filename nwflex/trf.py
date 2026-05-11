"""
trf.py — TRF (Tandem Repeats Finder) parsing and panel construction

Utilities for working with TRF output and turning it into a curated
locus panel suitable for simulation.

The expected workflow is:

1. Run TRF on a reference FASTA (typical params:
   ``trf <fasta> 2 5 5 80 10 20 100 -d -h``).
2. :func:`parse_trf_dat` — load the resulting ``.dat`` file into a
   :class:`pandas.DataFrame` with one row per detected repeat.
3. :func:`annotate_repeat_isolation` — annotate each repeat with its
   coverage status and distance to the nearest neighboring repeat.
4. :func:`filter_isolated_repeats` — drop repeats that overlap others
   or sit too close to a neighbor; optionally restrict by period size
   or "perfect" (single-base) mono-repeats.
5. The resulting DataFrame is the input to panel-building code that
   downstream simulation modules can consume.

Chromosome lengths needed by step 3 can be read from a FASTA's
``.fai`` index via :func:`read_chrom_lengths` (the index is generated
on demand by ``samtools faidx`` if missing).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ============================================================================
# TRF .dat PARSING
# ============================================================================

_TRF_COLUMNS = [
    "start", "end", "period_size", "copy_number", "consensus_size",
    "pct_matches", "pct_indels", "score",
    "pct_A", "pct_C", "pct_G", "pct_T", "entropy",
    "consensus_pattern", "repeat_sequence",
]
_TRF_INT_COLS = ["start", "end", "period_size", "consensus_size", "score"]
_TRF_FLOAT_COLS = [
    "copy_number", "pct_matches", "pct_indels",
    "pct_A", "pct_C", "pct_G", "pct_T", "entropy",
]


def parse_trf_dat(path) -> pd.DataFrame:
    """
    Parse a TRF ``.dat`` file into a DataFrame with one row per repeat.

    Coordinates are converted to 0-based half-open: ``start`` is 0-based
    (TRF emits 1-based starts), ``end`` is exclusive.

    Parameters
    ----------
    path : str or Path
        Path to the TRF ``.dat`` output file.

    Returns
    -------
    pandas.DataFrame
        Columns: ``chrom``, plus the 15 TRF fields.  Numeric columns
        are coerced to ``int`` / ``float``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is parseable but contains no repeat data lines.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"TRF .dat file not found: {path}\n"
            "Generate it with: trf <fasta> 2 5 5 80 10 20 100 -d -h"
        )
    rows = []
    current_chrom: Optional[str] = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Sequence:"):
                current_chrom = line.split()[1]
                continue
            if line[0].isdigit():
                fields = line.split()
                if len(fields) < 15:
                    continue
                row = {"chrom": current_chrom}
                for col, val in zip(_TRF_COLUMNS, fields[:15]):
                    row[col] = val
                rows.append(row)
    if not rows:
        raise ValueError(f"No data lines found in TRF .dat file: {path}")
    df = pd.DataFrame(rows)
    for col in _TRF_INT_COLS:
        df[col] = df[col].astype(int)
    for col in _TRF_FLOAT_COLS:
        df[col] = df[col].astype(float)
    df["start"] = df["start"] - 1
    return df


# ============================================================================
# REPEAT ISOLATION ANNOTATION
# ============================================================================

def _annotate_chrom(df: pd.DataFrame, chrom_len: int) -> pd.DataFrame:
    """
    Annotate one chromosome's repeats with coverage and neighbor distance.

    Adds four columns:

    - ``start_cover`` — number of repeat intervals covering the start
      coordinate (1 if the repeat's start is uncovered by any other).
    - ``end_cover`` — number of repeat intervals covering the end
      coordinate (0 if uncovered).
    - ``ldist`` — distance from this repeat's start to the nearest
      neighboring repeat's end on its left.
    - ``rdist`` — distance from this repeat's end to the nearest
      neighboring repeat's start on its right.
    """
    starts, ends = df["start"].values, df["end"].values
    interval_table = pd.concat([
        pd.DataFrame({"pos": starts, "is_start": True}),
        pd.DataFrame({"pos": ends, "is_start": False}),
    ]).sort_values(["pos", "is_start"], ignore_index=True)
    interval_table["cum_start"] = interval_table["is_start"].cumsum()
    interval_table["cum_end"] = (~interval_table["is_start"]).cumsum()
    interval_table["coverage"] = (
        interval_table["cum_start"] - interval_table["cum_end"]
    )
    coverage_at = interval_table.groupby("pos")["coverage"].last()
    df = df.copy()
    df["start_cover"] = df["start"].map(coverage_at).astype(int)
    df["end_cover"] = df["end"].map(coverage_at).astype(int)
    unique_ends = np.append(0, np.unique(ends))
    unique_starts = np.append(np.unique(starts), chrom_len)
    df["ldist"] = (
        starts
        - unique_ends[np.searchsorted(unique_ends, starts, side="right") - 1]
    )
    df["rdist"] = unique_starts[np.searchsorted(unique_starts, ends)] - ends
    return df


def annotate_repeat_isolation(
    df: pd.DataFrame, chrom_lengths: Dict[str, int]
) -> pd.DataFrame:
    """
    Annotate every repeat with isolation metrics, per-chromosome.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`parse_trf_dat`.
    chrom_lengths : dict[str, int]
        Mapping from chromosome name (matching the ``chrom`` column) to
        chromosome length in bp.

    Returns
    -------
    pandas.DataFrame
        ``df`` with the four isolation columns added (see :func:`_annotate_chrom`).

    Raises
    ------
    KeyError
        If any chromosome in ``df`` is missing from ``chrom_lengths``.
    """
    parts = []
    for chrom, group in df.groupby("chrom"):
        if chrom not in chrom_lengths:
            raise KeyError(f"Chromosome {chrom!r} not in chrom_lengths")
        parts.append(_annotate_chrom(group, chrom_lengths[chrom]))
    return pd.concat(parts, ignore_index=True)


# ============================================================================
# FILTERING
# ============================================================================

def filter_isolated_repeats(
    df: pd.DataFrame,
    *,
    min_dist: int = 100,
    max_period: Optional[int] = None,
    require_isolated: bool = True,
    perfect_only: bool = False,
) -> pd.DataFrame:
    """
    Filter a TRF DataFrame to a curated panel.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`annotate_repeat_isolation`.  Must have the
        isolation columns (``start_cover``, ``end_cover``, ``ldist``,
        ``rdist``).
    min_dist : int, default 100
        Minimum bp from each repeat boundary to the nearest neighbor.
    max_period : int or None, default None
        If set, drop repeats with ``period_size`` larger than this.
    require_isolated : bool, default True
        If True, keep only repeats whose start has no overlapping repeat
        and whose end has none either (``start_cover == 1`` and
        ``end_cover == 0``).
    perfect_only : bool, default False
        If True, keep only mono-repeats whose ``repeat_sequence`` is a
        single base (i.e., truly perfect ``A^n``, ``C^n``, etc.).
        Multi-base periods are unaffected.

    Returns
    -------
    pandas.DataFrame
        A copy of the filtered subset.
    """
    mask = pd.Series(True, index=df.index)
    if require_isolated:
        mask &= df["start_cover"] == 1
        mask &= df["end_cover"] == 0
    mask &= df["ldist"] >= min_dist
    mask &= df["rdist"] >= min_dist
    if max_period is not None:
        mask &= df["period_size"] <= max_period
    if perfect_only:
        is_mono = df["period_size"] == 1
        is_pure = df["repeat_sequence"].apply(lambda s: len(set(s.upper())) == 1)
        mask &= ~is_mono | is_pure
    return df.loc[mask].copy()


# ============================================================================
# CHROMOSOME LENGTHS FROM FASTA INDEX
# ============================================================================

def read_chrom_lengths(fasta_path) -> Dict[str, int]:
    """
    Read chromosome lengths from a FASTA's ``.fai`` index.

    If the ``.fai`` index does not exist, it is generated by running
    ``samtools faidx`` on the FASTA.

    Parameters
    ----------
    fasta_path : str or Path
        Path to the reference FASTA.

    Returns
    -------
    dict[str, int]
        Mapping from chromosome name to length in bp.

    Raises
    ------
    FileNotFoundError
        If ``fasta_path`` does not exist.
    subprocess.CalledProcessError
        If ``samtools faidx`` fails when invoked.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    fai_path = Path(str(fasta_path) + ".fai")
    if not fai_path.exists():
        subprocess.run(["samtools", "faidx", str(fasta_path)], check=True)
    lengths: Dict[str, int] = {}
    with open(fai_path) as fh:
        for line in fh:
            parts = line.split("\t")
            if len(parts) >= 2:
                lengths[parts[0]] = int(parts[1])
    return lengths
