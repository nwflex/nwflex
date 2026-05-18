"""Coverage-weighted "fraction correct" by Δ for the full-lflank resweep.

Loads shards from ``supplement/data_full_lflank/single_repeat/`` (N=10,
SNV slots {none, 1, 2, 5, 10}, full informative lflank range), filters
to both-flanks-positive reads, and plots one curve per arm in a
``(SNV slot × motif_len)`` small-multiples grid.

A read row counts as "correct" when its combined-strand ``state`` is
``P`` (score = truth, length correct) or ``T`` (score > truth — found a
better alignment but at the truth's length).  Reads are pooled across
loci within each ``(motif_len, snv_offset, arm, delta)`` cell.

Usage::

    python scripts/plot_full_lflank_proportion.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "supplement" / "data_full_lflank" / "single_repeat"
FIG_DIR   = REPO_ROOT / "supplement" / "figures_full_lflank"

ARMS = ["NW-flex", "BWA-no-clip", "BWA-std"]
ARM_COLOR = {
    "NW-flex":     "#08519c",
    "BWA-no-clip": "#d95f02",
    "BWA-std":     "#7570b3",
}
ARM_TITLE = {
    "NW-flex":     "NW-flex",
    "BWA-no-clip": "BWA-MEM (no clip)",
    "BWA-std":     "BWA-MEM",
}

# snv_offset is 0-indexed from the boundary; -1 sentinel = no SNV.
SNV_SLOTS = [-1, 0, 1, 4, 9]
SNV_LABEL = {-1: "no SNV", 0: "SNV @ 1", 1: "SNV @ 2",
             4: "SNV @ 5", 9: "SNV @ 10"}

MOTIF_LENS = [1, 2, 3]


_GROUP_FINE = ["motif_len", "snv_offset", "arm", "delta", "lflank"]
_USECOLS = ["delta", "lflank", "rflank", "arm", "state",
            "motif_len", "snv_offset"]


def _aggregate_one_shard(path: Path) -> pd.DataFrame:
    """Per-shard sum/count of (state ∈ {P, T}) at the
    ``(motif_len, snv_offset, arm, delta, lflank)`` grain so we can
    marginalize over either axis later."""
    df = pd.read_csv(path, usecols=_USECOLS)
    df = df[(df["lflank"] > 0) & (df["rflank"] > 0)]
    if df.empty:
        return pd.DataFrame(columns=[*_GROUP_FINE, "n_correct", "n_total"])
    df["correct"] = df["state"].isin(["P", "T"]).astype(int)
    return (df.groupby(_GROUP_FINE, as_index=False)
              .agg(n_correct=("correct", "sum"),
                   n_total=("correct", "size")))


def aggregate_all(data_dir: Path) -> pd.DataFrame:
    """Stream every shard through :func:`_aggregate_one_shard` and reduce
    sums/counts to one row per fine-grained cell."""
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no shards in {data_dir}")
    print(f"streaming {len(files)} shards from {data_dir} ...")

    chunks = []
    report_every = max(1, len(files) // 20)
    for i, path in enumerate(files, 1):
        chunks.append(_aggregate_one_shard(path))
        if i % report_every == 0 or i == len(files):
            print(f"  {i:,}/{len(files):,}")

    cat = pd.concat(chunks, ignore_index=True)
    return (cat.groupby(_GROUP_FINE, as_index=False)
               .agg(n_correct=("n_correct", "sum"),
                    n_total  =("n_total",   "sum")))


def marginalize(fine: pd.DataFrame, drop_col: str) -> pd.DataFrame:
    """Collapse the fine-grained frame by summing counts across ``drop_col``."""
    keys = [c for c in _GROUP_FINE if c != drop_col]
    m = (fine.groupby(keys, as_index=False)
             .agg(n_correct=("n_correct", "sum"),
                  n_total  =("n_total",   "sum")))
    m["prop"] = m["n_correct"] / m["n_total"]
    return m


def plot_grid(agg: pd.DataFrame, *, x_col: str, x_label: str,
              out_path: Path) -> None:
    """Small-multiples grid: rows = SNV slot, cols = motif_len,
    x = ``x_col``, y = coverage-weighted fraction P|T, one line per arm."""
    n_rows, n_cols = len(SNV_SLOTS), len(MOTIF_LENS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.0 * n_cols, 1.9 * n_rows),
        sharex=True, sharey=True,
    )

    for i, snv in enumerate(SNV_SLOTS):
        for j, L in enumerate(MOTIF_LENS):
            ax = axes[i, j]
            cell = agg[(agg["snv_offset"] == snv) & (agg["motif_len"] == L)]
            for arm in ARMS:
                line = cell[cell["arm"] == arm].sort_values(x_col)
                ax.plot(line[x_col], line["prop"],
                        marker="o", ms=3, lw=1.2,
                        color=ARM_COLOR[arm], label=ARM_TITLE[arm])
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3, lw=0.5)
            if i == 0:
                ax.set_title(f"motif_len = {L}", fontsize=10)
            if j == 0:
                ax.set_ylabel(SNV_LABEL[snv], fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel(x_label, fontsize=9)

    # Shared legend on top.
    handles = [plt.Line2D([0], [0], color=ARM_COLOR[a], marker="o", ms=4,
                          lw=1.2, label=ARM_TITLE[a]) for a in ARMS]
    fig.legend(handles=handles, loc="upper center",
               ncol=len(ARMS), frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(
        f"Coverage-weighted fraction correct (P or T) — "
        f"N=10, both flanks > 0  ·  x = {x_label}",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=144, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    fine = aggregate_all(DATA_DIR)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fine.to_csv(FIG_DIR / "proportion_fine.csv", index=False)
    print(f"wrote {FIG_DIR / 'proportion_fine.csv'}  ({len(fine)} rows)")

    by_delta = marginalize(fine, drop_col="lflank")
    by_lflank = marginalize(fine, drop_col="delta")
    by_delta.to_csv(FIG_DIR / "proportion_by_delta.csv", index=False)
    by_lflank.to_csv(FIG_DIR / "proportion_by_lflank.csv", index=False)

    plot_grid(by_delta, x_col="delta",
              x_label="Δ (haplotype repeat-count offset)",
              out_path=FIG_DIR / "proportion_by_delta.png")
    plot_grid(by_lflank, x_col="lflank",
              x_label="lflank (left-flank overhang, bp)",
              out_path=FIG_DIR / "proportion_by_lflank.png")


if __name__ == "__main__":
    main()
