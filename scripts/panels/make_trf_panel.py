"""Build a repeat panel from TRF .dat files.

This is the nwflex-native version of the historical
``stormflex/wgs/scripts/make_panel_fast.py`` builder. It parses TRF calls,
keeps isolated loci, extracts reference flanks, and writes the current panel
schema used by the simulation notebooks.

Example:
    python scripts/panels/make_trf_panel.py \\
        --chr chr1 chr2 chr21 chr22 \\
        --fasta genomes/hg38.chrYpsr.fa \\
        --trf-dir genomes/trf/chroms \\
        --require-pure \\
        --max-period 6 \\
        --max-tract-bp 1000 \\
        --output /data/safe/levy/genome/wgs-panels/wgs.16chr.iso_pure.panel.v2.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nwflex.trf import (
    annotate_repeat_isolation,
    build_panel_table,
    build_target_table,
    classify_isolated_repeats,
    parse_trf_dat,
    read_chrom_lengths,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--chr", type=str, nargs="+", required=True,
                   help="Chromosomes to include, e.g. chr1 chr2 chr21.")
    p.add_argument("--fasta", type=Path, required=True,
                   help="Reference FASTA. A .fai index is created if absent.")
    p.add_argument("--trf-dir", type=Path, required=True,
                   help="Directory containing <chrom>.fa.2.5.5.80.10.20.100.dat files.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-period", type=int, default=1)
    p.add_argument("--max-period", type=int, default=2)
    p.add_argument("--max-tract-bp", type=int, default=200)
    p.add_argument("--flank-size", type=int, default=500)
    p.add_argument("--require-pure", action="store_true", default=False,
                   help="Keep only iso_pure rows.")
    p.add_argument("--min-pct-matches", type=float, default=80,
                   help="Minimum TRF pct_matches when imperfect rows are allowed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    chrom_lengths = read_chrom_lengths(args.fasta)

    panels = []
    for chrom in args.chr:
        trf_dat = args.trf_dir / f"{chrom}.fa.2.5.5.80.10.20.100.dat"
        if not trf_dat.exists():
            print(f"{chrom}: missing {trf_dat}; skipping")
            continue

        print(f"{chrom}: parsing {trf_dat}")
        df = parse_trf_dat(trf_dat)
        print(f"  raw TRF rows: {len(df):,}")

        tract_len = df["end"] - df["start"]
        df = df[df["period_size"].between(args.min_period, args.max_period)].copy()
        df = df[tract_len.loc[df.index] <= args.max_tract_bp].copy()
        print(
            f"  after period {args.min_period}-{args.max_period} "
            f"and tract <= {args.max_tract_bp}: {len(df):,}"
        )

        df = annotate_repeat_isolation(df, chrom_lengths)
        df["class"] = classify_isolated_repeats(df)
        df = df[df["class"] != "drop"].copy()
        if args.require_pure:
            df = df[df["class"] == "iso_pure"].copy()
        else:
            df = df[df["pct_matches"] >= args.min_pct_matches].copy()
        print(f"  after isolation/purity filters: {len(df):,}")

        if len(df) == 0:
            continue

        targets = build_target_table(df, args.fasta, flank_size=args.flank_size)
        panels.append(build_panel_table(targets))

    if not panels:
        raise SystemExit("No panel rows built across requested chromosomes")

    out = pd.concat(panels, ignore_index=True)
    out = out.sort_values(["chr", "start_38"]).reset_index(drop=True)
    out["pind"] = range(len(out))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, sep="\t", index=False)
    print(f"\nWrote {len(out):,} loci to {args.output}")
    print("Top motif counts:")
    print(out["type"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
