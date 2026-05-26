"""Build a K-per-motif catalog from a repeat panel.

The output schema matches the input panel layout, so it loads through the same
panel-reading machinery as the larger source panel.

Boundary-clean rule:
    lflank[-1] != motif[-1]
    rflank[0]  != motif[0]

Example:
    python scripts/panels/build_motif_catalog.py \\
        --panel-tsv /data/safe/levy/genome/wgs-panels/wgs.16chr.iso_pure.panel.v2.tsv \\
        --K 100 \\
        --max-period 3 \\
        --output data/wgs.motif_catalog_K100.panel.v2.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--panel-tsv", type=Path, required=True)
    p.add_argument("--K", type=int, default=100,
                   help="Take up to this many loci per motif.")
    p.add_argument("--min-period", type=int, default=1)
    p.add_argument("--max-period", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {args.panel_tsv} ...", file=sys.stderr)
    df = pd.read_csv(args.panel_tsv, sep="\t", low_memory=False)
    print(f"  loaded {len(df):,} rows", file=sys.stderr)

    df["period"] = df["type"].str.len()
    df = df[df["period"].between(args.min_period, args.max_period)].copy()
    print(
        f"  after period {args.min_period}-{args.max_period}: {len(df):,}",
        file=sys.stderr,
    )

    clean = (
        (df["lflank"].str[-1] != df["type"].str[-1])
        & (df["rflank"].str[0] != df["type"].str[0])
    )
    df = df[clean].copy()
    print(f"  after boundary-clean filter: {len(df):,}", file=sys.stderr)

    df = df.sort_values(["chr", "start_38"]).reset_index(drop=True)
    out = df.groupby("type", observed=True, group_keys=False, sort=False).head(args.K)
    out = out.sort_values(["period", "type", "chr", "start_38"]).reset_index(drop=True)
    out["pind"] = range(len(out))
    out = out.drop(columns=["period"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, sep="\t", index=False)
    print(f"\nWrote {len(out):,} loci to {args.output}", file=sys.stderr)

    summary = (
        out.assign(period=out["type"].str.len())
        .groupby(["period", "type"])
        .size()
        .reset_index(name="n")
    )
    short = summary[summary["n"] < args.K]
    print(f"\nMotifs short of K={args.K}: {len(short)}", file=sys.stderr)
    if len(short):
        print(short.to_string(index=False), file=sys.stderr)
    print("\nPer-period totals:", file=sys.stderr)
    for period in sorted(summary["period"].unique()):
        sub = summary[summary["period"] == period]
        print(
            f"  period {period}: {len(sub)} motifs, {sub['n'].sum():,} loci",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
