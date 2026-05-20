"""Length-only (state == "P") proportion-correct by bridge length |M| for Table 1B.

Aggregates the compound per-read shards in
``supplement/data_perread/compound/`` (produced by run_compound_paper.py).
Recipe per reports/perf_sweep_spec.md:
  - per-read, full lflank, both-flanks-positive
  - EXCLUDE the (delta1, delta2) = (0, 0) cell (reference == haplotype)
  - pool forward and reverse-complement arms as independent observations
  - correct == state "P" (repeat length recovered)
Grouped by (bridge_len, arm).

Usage:
    python scripts/aggregate_compound_table_lengthonly.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARD_DIR = REPO_ROOT / "supplement" / "data_perread" / "compound"
OUT_CSV = REPO_ROOT / "supplement" / "figures" / "compound_proportion_by_bridge_lengthonly.csv"

USECOLS = ["delta1", "delta2", "lflank", "rflank", "arm",
           "fwd_state", "rc_state", "bridge_len"]
GROUP = ["bridge_len", "arm"]


def main():
    files = sorted(glob.glob(str(SHARD_DIR / "*.csv")))
    print(f"{len(files)} compound shards", flush=True)
    parts = []
    for p in files:
        df = pd.read_csv(p, usecols=USECOLS)
        df = df[(df["lflank"] > 0) & (df["rflank"] > 0)]
        df = df[~((df["delta1"] == 0) & (df["delta2"] == 0))]   # exclude (0,0)
        if df.empty:
            continue
        long = pd.concat([
            df[GROUP].assign(s=df["fwd_state"]),
            df[GROUP].assign(s=df["rc_state"]),
        ], ignore_index=True)
        long["P"] = (long["s"] == "P").astype(int)
        parts.append(long.groupby(GROUP, sort=False)
                      .agg(n_P=("P", "sum"), n_obs=("P", "size")))
    total = pd.concat(parts).groupby(level=GROUP).sum().reset_index()
    total["prop_P"] = total["n_P"] / total["n_obs"]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    total.sort_values(GROUP).to_csv(OUT_CSV, index=False)

    # comparison vs committed Table 1B
    table = {"BWA-std": [.21, .48, .69, .76, .80],
             "BWA-no-clip": [.23, .55, .79, .87, .92],
             "NW-flex": [1.0, 1.0, 1.0, 1.0, 1.0]}
    print(f"\nwrote {OUT_CSV}")
    print(f"\n{'arm':12} " + "  ".join(f"|M|={m}" for m in (1, 2, 3, 4, 5)))
    for arm in ("BWA-std", "BWA-no-clip", "NW-flex"):
        row = []
        for m in (1, 2, 3, 4, 5):
            r = total[(total.arm == arm) & (total.bridge_len == m)]
            row.append(f"{r.prop_P.iloc[0]:.3f}" if not r.empty else "  -  ")
        print(f"{arm:12} " + "    ".join(row) + f"   TABLE {table[arm]}")


if __name__ == "__main__":
    main()
