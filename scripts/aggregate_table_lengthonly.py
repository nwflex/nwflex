"""Length-only (state == "P") proportion-correct by Delta for Table 1A.

Re-aggregates the existing full-lflank per-read shards in
``supplement/data_full_lflank/single_repeat/`` WITHOUT re-running the
sweep.  Pools forward and reverse-complement arms as independent
observations (per reports/perf_sweep_spec.md), filters to
both-flanks-positive reads, and counts a strand-observation correct
when its state is "P" (length recovered).

For verification it also emits the combined-state P-only proportion and
the old (P or T) independent proportion, so we can see which recipe the
committed table actually used.

Usage:
    python scripts/aggregate_table_lengthonly.py --workers 48
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from multiprocessing import Pool

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARD_DIR = REPO_ROOT / "supplement" / "data_full_lflank" / "single_repeat"
OUT_CSV = REPO_ROOT / "supplement" / "figures_full_lflank" / "proportion_by_delta_lengthonly.csv"

USECOLS = ["delta", "lflank", "rflank", "arm",
           "fwd_state", "rc_state", "state", "motif_len", "snv_offset"]
GROUP = ["motif_len", "snv_offset", "arm", "delta"]


def _agg_chunk(paths: list[str]) -> pd.DataFrame:
    parts = []
    for p in paths:
        try:
            df = pd.read_csv(p, usecols=USECOLS)
        except Exception:
            continue
        df = df[(df["lflank"] > 0) & (df["rflank"] > 0)]
        if df.empty:
            continue
        # Independent strand observations: stack fwd_state and rc_state.
        long = pd.concat([
            df[GROUP].assign(s=df["fwd_state"]),
            df[GROUP].assign(s=df["rc_state"]),
        ], ignore_index=True)
        long["P_indep"] = (long["s"] == "P").astype(int)
        long["PT_indep"] = long["s"].isin(["P", "T"]).astype(int)
        g_indep = (long.groupby(GROUP, sort=False)
                   .agg(n_P_indep=("P_indep", "sum"),
                        n_PT_indep=("PT_indep", "sum"),
                        n_obs_indep=("P_indep", "size")))
        # Combined-state (one observation per read).
        df = df.copy()
        df["P_comb"] = (df["state"] == "P").astype(int)
        g_comb = (df.groupby(GROUP, sort=False)
                  .agg(n_P_comb=("P_comb", "sum"),
                       n_reads=("P_comb", "size")))
        parts.append(g_indep.join(g_comb))
    if not parts:
        return pd.DataFrame()
    return (pd.concat(parts).groupby(level=GROUP).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    files = sorted(glob.glob(str(SHARD_DIR / "*.csv")))
    print(f"{len(files)} shards in {SHARD_DIR}", flush=True)
    n = max(1, len(files) // (args.workers * 4))
    chunks = [files[i:i + n] for i in range(0, len(files), n)]
    print(f"{len(chunks)} chunks, {args.workers} workers", flush=True)

    with Pool(args.workers) as pool:
        results = pool.map(_agg_chunk, chunks)

    results = [r for r in results if not r.empty]
    total = pd.concat(results).groupby(level=GROUP).sum().reset_index()

    total["prop_P_indep"] = total["n_P_indep"] / total["n_obs_indep"]
    total["prop_PT_indep"] = total["n_PT_indep"] / total["n_obs_indep"]
    total["prop_P_comb"] = total["n_P_comb"] / total["n_reads"]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    total.sort_values(GROUP).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(total)} rows)", flush=True)


if __name__ == "__main__":
    main()
