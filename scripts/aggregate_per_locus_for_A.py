"""Per-locus aggregates for the A__aggregate_4panel figure.

The four A panels want to show locus-to-locus variation in correctness,
holding everything else as the per-locus average.  This script reads
the raw sweep shards once and writes four small per-locus CSVs that
the figure builder consumes.

Single-locus side (filtered to motif_len = 3):
  A1_per_pind__delta.csv     one row per (pind, arm, delta)
                              correctness avg over snv=-1, all N, all lflank
  A2_per_pind__lflank.csv    one row per (pind, arm, lflank)
                              correctness avg over snv=-1, all N, all delta
  A3_per_pind__snv.csv       one row per (pind, arm, snv_offset)
                              correctness avg over all N, all delta, all lflank

Compound side (filtered to motif1_len = 2, motif2_len = 3, N1 = N2 = 10):
  A4_per_pair__bridge.csv    one row per (pind1, pind2, arm, bridge_len)
                              correctness avg over all (delta1, delta2)

Each row's ``correct`` column is in [0, 1].
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
# Defaults; overridden in ``main`` via ``--config``.
SINGLE_SHARDS = REPO_ROOT / "supplement/data/single_repeat"
COMPOUND_SHARDS = REPO_ROOT / "supplement/data/compound"
OUT_DIR = REPO_ROOT / "supplement/data"
PANEL_PATH = REPO_ROOT / "data/hg38_motif_sample_K100.tsv"
DEFAULT_CONFIG = REPO_ROOT / "scripts/configs/single_repeat.yaml"

SINGLE_MOTIF_LEN = 3
COMPOUND_L1L2 = (2, 3)
COMPOUND_N = (10, 10)


def _filter_single_shards(target_pinds: set[int]) -> list[str]:
    """Shard files whose pind is in the target set."""
    files = sorted(glob.glob(str(SINGLE_SHARDS / "*.csv")))
    keep = []
    for f in files:
        # filename pattern: pind{ddddd}__N{NN}__SNV{+d}.csv
        name = Path(f).stem
        pind = int(name.split("__")[0].replace("pind", ""))
        if pind in target_pinds:
            keep.append(f)
    return keep


def _filter_compound_shards(panel: pd.DataFrame) -> list[str]:
    """Shard files matching motif1_len=2, motif2_len=3, N1=N2=10."""
    panel_by_pind = panel.set_index("pind")
    L1_pinds = set(panel.loc[panel["type"].str.len() == COMPOUND_L1L2[0],
                              "pind"].astype(int))
    L2_pinds = set(panel.loc[panel["type"].str.len() == COMPOUND_L1L2[1],
                              "pind"].astype(int))
    N1, N2 = COMPOUND_N
    files = sorted(glob.glob(str(COMPOUND_SHARDS / "*.csv")))
    keep = []
    for f in files:
        # filename pattern: pair{p1}_{p2}__N{N1}_{N2}__M{bridge}.csv
        name = Path(f).stem
        parts = name.split("__")
        pair_part = parts[0].replace("pair", "")
        p1, p2 = map(int, pair_part.split("_"))
        n_part = parts[1].replace("N", "")
        n1, n2 = map(int, n_part.split("_"))
        if n1 != N1 or n2 != N2:
            continue
        if p1 in L1_pinds and p2 in L2_pinds:
            keep.append(f)
    return keep


def _read_concat(files, usecols, label):
    t0 = time.time()
    dfs = []
    for i, f in enumerate(files):
        dfs.append(pd.read_csv(f, usecols=usecols))
        if (i + 1) % 5000 == 0:
            print(f"  [{label}] {i+1:>6d} / {len(files)}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return pd.concat(dfs, ignore_index=True)


def main():
    global SINGLE_SHARDS, COMPOUND_SHARDS, OUT_DIR, PANEL_PATH
    parser = argparse.ArgumentParser(
        description="Per-locus aggregates for the A__aggregate_4panel figure."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="YAML config; reads output.data_dir and panel keys.",
    )
    parser.add_argument("--single-dir", type=Path, default=None,
                        help="Override single-repeat shard dir.")
    parser.add_argument("--compound-dir", type=Path, default=None,
                        help="Override compound shard dir.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Override output dir for the A*.csv files.")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_dir = (REPO_ROOT / cfg["output"]["data_dir"]).resolve()
    SINGLE_SHARDS = args.single_dir or (data_dir / "single_repeat")
    COMPOUND_SHARDS = args.compound_dir or (data_dir / "compound")
    OUT_DIR = args.out_dir or data_dir
    PANEL_PATH = (REPO_ROOT / cfg["panel"]).resolve()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH, sep="\t")
    target_pinds = set(panel.loc[panel["type"].str.len() == SINGLE_MOTIF_LEN,
                                  "pind"].astype(int))
    print(f"single-locus: {len(target_pinds)} pinds with motif_len={SINGLE_MOTIF_LEN} in panel")

    # ----- single-locus side --------------------------------------------
    single_files = _filter_single_shards(target_pinds)
    print(f"single-locus: reading {len(single_files)} shards "
          f"(filtered from full set)...", flush=True)
    usecols = ["pind", "N", "snv_offset", "arm", "delta", "lflank",
               "fwd_state", "rc_state"]
    s_df = _read_concat(single_files, usecols, "single")
    # length-only correctness (state P), pooled over strands: each strand
    # contributes one observation.
    s_df = pd.concat([
        s_df.assign(correct=(s_df["fwd_state"] == "P").astype(float)),
        s_df.assign(correct=(s_df["rc_state"] == "P").astype(float)),
    ], ignore_index=True)
    print(f"single-locus: loaded {len(s_df):,} rows (x2 strands)", flush=True)

    # A1 — per (pind, arm, delta) over snv=-1, all N, all lflank
    a1 = (s_df[s_df["snv_offset"] == -1]
            .groupby(["pind", "arm", "delta"], dropna=False, sort=False)
            ["correct"].mean().reset_index())
    a1.to_csv(OUT_DIR / "A1_per_pind__delta.csv", index=False)
    print(f"  wrote A1: {len(a1):,} rows")

    # A2 — per (pind, arm, lflank) over snv=-1, all N, all delta
    a2 = (s_df[s_df["snv_offset"] == -1]
            .groupby(["pind", "arm", "lflank"], dropna=False, sort=False)
            ["correct"].mean().reset_index())
    a2.to_csv(OUT_DIR / "A2_per_pind__lflank.csv", index=False)
    print(f"  wrote A2: {len(a2):,} rows")

    # A3 — per (pind, arm, snv_offset) over all N, all delta, all lflank
    a3 = (s_df.groupby(["pind", "arm", "snv_offset"],
                       dropna=False, sort=False)
              ["correct"].mean().reset_index())
    a3.to_csv(OUT_DIR / "A3_per_pind__snv.csv", index=False)
    print(f"  wrote A3: {len(a3):,} rows")

    # ----- compound side ------------------------------------------------
    compound_files = _filter_compound_shards(panel)
    print(f"\ncompound: reading {len(compound_files)} shards "
          f"(L1={COMPOUND_L1L2[0]}, L2={COMPOUND_L1L2[1]}, "
          f"N1=N2=10)...", flush=True)
    usecols = ["pind1", "pind2", "bridge_len",
               "arm", "delta1", "delta2", "fwd_state", "rc_state"]
    c_df = _read_concat(compound_files, usecols, "compound")
    # exclude the (0,0) reference allele; length-only correctness, pooled strands
    c_df = c_df[~((c_df["delta1"] == 0) & (c_df["delta2"] == 0))]
    c_df = pd.concat([
        c_df.assign(correct=(c_df["fwd_state"] == "P").astype(float)),
        c_df.assign(correct=(c_df["rc_state"] == "P").astype(float)),
    ], ignore_index=True)
    print(f"compound: loaded {len(c_df):,} rows (x2 strands, (0,0) excluded)", flush=True)

    a4 = (c_df.groupby(["pind1", "pind2", "arm", "bridge_len"],
                        dropna=False, sort=False)
              ["correct"].mean().reset_index())
    a4.to_csv(OUT_DIR / "A4_per_pair__bridge.csv", index=False)
    print(f"  wrote A4: {len(a4):,} rows")


if __name__ == "__main__":
    main()
