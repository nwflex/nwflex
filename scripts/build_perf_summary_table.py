"""Build the performance summary table (Table 1) as a tidy CSV.

Reads per-read sweep shards, applies the length-only correctness recipe
(reports/perf_sweep_spec.md), and writes one long-format CSV of
correct-alignment rates.

Coverage modes:
  paper : Table 1A (tri, N=10, no-SNV + SNV@1) + Table 1B (di-tri).
  full  : all single motif lengths + SNV range, all compound pairs.

Recipe: correct = (state == "P") [length recovered]; forward and rc strands
pooled as independent observations; both-flanks-positive reads only; compound
excludes the (delta1, delta2) == (0, 0) cell.

Usage:
    python scripts/build_perf_summary_table.py --mode paper
    python scripts/build_perf_summary_table.py --mode full
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SINGLE_DIR = REPO_ROOT / "supplement" / "data_perread" / "single_repeat"
COMPOUND_DIR = REPO_ROOT / "supplement" / "data_perread" / "compound"

OUTPUT_COLUMNS = [
    "test", "aligner", "motif_len", "motif1_len", "motif2_len",
    "snv_offset", "delta", "bridge_len", "n_correct", "n_obs", "prop_correct",
]


def _both_flanks_positive(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only reads that span at least 1 bp of each flank."""
    return df[(df["lflank"] > 0) & (df["rflank"] > 0)]


def _pooled_strand_correct(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Stack fwd_state/rc_state as independent observations; correct = state 'P'.

    Returns group_cols + n_correct, n_obs, prop_correct.
    """
    long = pd.concat(
        [df[group_cols].assign(s=df["fwd_state"]),
         df[group_cols].assign(s=df["rc_state"])],
        ignore_index=True,
    )
    long["correct"] = (long["s"] == "P").astype(int)
    g = (long.groupby(group_cols, sort=False)
              .agg(n_correct=("correct", "sum"), n_obs=("correct", "size"))
              .reset_index())
    g["prop_correct"] = g["n_correct"] / g["n_obs"]
    return g


def single_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy single-repeat correct-rate by (motif_len, snv_offset, delta, arm)."""
    df = _both_flanks_positive(df)
    g = _pooled_strand_correct(df, ["motif_len", "snv_offset", "delta", "arm"])
    g = g.rename(columns={"arm": "aligner"})
    g["test"] = "single"
    return g


def compound_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy compound correct-rate by (motif1_len, motif2_len, bridge_len, arm),
    pooling the (delta1, delta2) grid and excluding the (0, 0) cell."""
    df = _both_flanks_positive(df)
    df = df[~((df["delta1"] == 0) & (df["delta2"] == 0))]
    g = _pooled_strand_correct(df, ["motif1_len", "motif2_len", "bridge_len", "arm"])
    g = g.rename(columns={"arm": "aligner"})
    g["test"] = "compound"
    return g


def _to_output_schema(single_g: pd.DataFrame, compound_g: pd.DataFrame) -> pd.DataFrame:
    """Combine single + compound tidy frames into the long OUTPUT_COLUMNS layout."""
    frames = [f for f in (single_g, compound_g) if f is not None and not f.empty]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    # Dimension columns are NA on the rows of the other test type; concat upcasts
    # them to float (delta -> -5.0). Use nullable Int64 so the committed CSV stays
    # integer-valued and re-loads cleanly for filtering.
    int_dims = ["motif_len", "motif1_len", "motif2_len", "snv_offset", "delta", "bridge_len"]
    for col in int_dims:
        out[col] = out[col].astype("Int64")
    return out[OUTPUT_COLUMNS]


def filter_paper(single_df: pd.DataFrame, compound_df: pd.DataFrame):
    """Restrict raw shards to the paper's Table 1 coverage:
    single = tri, N=10, no-SNV (offset -1) + SNV@1 (offset 0); compound = di-tri."""
    s = single_df[(single_df["motif_len"] == 3)
                  & (single_df["N"] == 10)
                  & (single_df["snv_offset"].isin([-1, 0]))]
    c = compound_df[(compound_df["motif1_len"] == 2)
                    & (compound_df["motif2_len"] == 3)]
    return s, c


SINGLE_USECOLS = ["delta", "lflank", "rflank", "arm", "fwd_state", "rc_state",
                  "state", "motif_len", "N", "snv_offset"]
COMPOUND_USECOLS = ["delta1", "delta2", "lflank", "rflank", "arm", "fwd_state",
                    "rc_state", "state", "motif1_len", "motif2_len", "bridge_len"]


def load_shards(directory, usecols: list[str]) -> pd.DataFrame:
    """Read and concat all *.csv shards in a directory (only `usecols`)."""
    paths = sorted(glob.glob(str(Path(directory) / "*.csv")))
    if not paths:
        raise FileNotFoundError(f"no shards in {directory}")
    return pd.concat((pd.read_csv(p, usecols=usecols) for p in paths),
                     ignore_index=True)


def build_table(mode: str, single_dir=SINGLE_DIR, compound_dir=COMPOUND_DIR) -> pd.DataFrame:
    """Aggregate shards into the long perf-summary table for the given mode."""
    single_df = load_shards(single_dir, SINGLE_USECOLS)
    compound_df = load_shards(compound_dir, COMPOUND_USECOLS)
    if mode == "paper":
        single_df, compound_df = filter_paper(single_df, compound_df)
    return _to_output_schema(single_table(single_df), compound_table(compound_df))


# Committed Table 1B values (prop correct by |M| = 1..5), per the manuscript.
PAPER_TABLE_1B = {
    "BWA-std":     [0.21, 0.48, 0.69, 0.76, 0.80],
    "BWA-no-clip": [0.23, 0.55, 0.79, 0.87, 0.92],
    "NW-flex":     [1.0, 1.0, 1.0, 1.0, 1.0],
}


def check_paper(out: pd.DataFrame, tol: float = 0.02) -> list[str]:
    """Return a list of mismatch messages vs the committed Table 1B (empty == OK)."""
    comp = out[out["test"] == "compound"]
    problems = []
    for aligner, expected in PAPER_TABLE_1B.items():
        for i, m in enumerate([1, 2, 3, 4, 5]):
            row = comp[(comp["aligner"] == aligner) & (comp["bridge_len"] == m)]
            got = float(row["prop_correct"].iloc[0]) if not row.empty else float("nan")
            if pd.isna(got) or abs(got - expected[i]) > tol:
                problems.append(f"{aligner} |M|={m}: got {got:.3f}, want {expected[i]:.3f}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the performance summary table.")
    ap.add_argument("--mode", choices=["paper", "full"], default="paper")
    ap.add_argument("--single-dir", type=Path, default=SINGLE_DIR)
    ap.add_argument("--compound-dir", type=Path, default=COMPOUND_DIR)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV (default: data/perf_summary[_full].csv).")
    args = ap.parse_args()

    out = build_table(args.mode, args.single_dir, args.compound_dir)
    out_path = args.out or (REPO_ROOT / "data" / (
        "perf_summary.csv" if args.mode == "paper" else "perf_summary_full.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} rows)")

    if args.mode == "paper":
        problems = check_paper(out)
        if problems:
            print("PAPER-MODE SELF-CHECK FAILED (Table 1B):")
            for p in problems:
                print("  " + p)
        else:
            print("paper-mode self-check OK: Table 1B matches within tol")


if __name__ == "__main__":
    main()
