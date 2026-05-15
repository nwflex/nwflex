"""Manuscript figures with the transposed layout.

Layout convention (all four figure families):
  rows top -> bottom: BWA-MEM, BWA-MEM (no clip), NW-flex
  columns: stratifier levels
  per panel: forward strand = filled rectangle, reverse-complement = inset circle
  color: fraction with score = truth, 0 (red) -> 1 (green)

Figures:
  1. single-repeat, no SNV, columns = N
  2. single-repeat, SNV stack at fixed N (one figure per N)
  3. compound, (mono, di) length pair, columns = |M|
  4. compound, BWA-MEM (no clip), grid = motif1_len x motif2_len (one figure per |M|)

Reads the pre-aggregated tidy CSVs written by aggregate_results.py.

Usage::

    python scripts/build_manuscript_figures.py --figs 1 2
    python scripts/build_manuscript_figures.py --figs 3 4   # after large_transposed lands
    python scripts/build_manuscript_figures.py --figs all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Patch, Rectangle

from nwflex.simulation.viz import (
    _CircleHandle,
    _PROPORTION_CMAP,
    _make_circle_handler,
    _proportion_color_fn,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
SINGLE_CSV = REPO_ROOT / "supplement/data/single_repeat_cross_locus_aggregate.csv"
COMPOUND_CSV_FULL = REPO_ROOT / "supplement/data_full/compound_cross_locus_aggregate.csv"
OUT_DIR = REPO_ROOT / "supplement/figures_priority"

ARM_ROWS = ["BWA-std", "BWA-no-clip", "NW-flex"]
ARM_LABELS = {
    "BWA-std": "BWA-MEM",
    "BWA-no-clip": "BWA-MEM\n(no-clip)",
    "NW-flex": "NW-flex",
}


# ---- pooling helpers ------------------------------------------------------

def _weighted_mean(g, col, w="n_loci"):
    return (g[col] * g[w]).sum() / g[w].sum()


def pool_single(df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    """Group by ``keep`` keys, weighted-average per-strand fractions."""
    g = df.groupby(keep, dropna=False, sort=False)
    pooled = g.apply(
        lambda d: pd.Series({
            "fwd": _weighted_mean(d, "frac_score_eq_truth_fwd"),
            "rc":  _weighted_mean(d, "frac_score_eq_truth_rc"),
            "n":   d["n_loci"].sum(),
        }),
        include_groups=False,
    ).reset_index()
    return pooled


# ---- panel drawing --------------------------------------------------------

def _draw_panel(ax, sub, *, x_col, y_col, x_vals, y_vals, color_of,
                cell_circle_radius=0.26, tick_fontsize=8):
    """Draw a heatmap panel with rectangle (fwd) + circle (rc) glyphs.

    ``sub`` is a per-cell DataFrame with columns ``x_col``, ``y_col``,
    ``fwd``, ``rc`` (one row per cell).
    """
    lookup = {(int(r[x_col]), int(r[y_col])): (r["fwd"], r["rc"])
              for _, r in sub.iterrows()}
    for y in y_vals:
        for x in x_vals:
            fwd_v, rc_v = lookup.get((x, y), (float("nan"), float("nan")))
            ax.add_patch(Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor=color_of(fwd_v), edgecolor="none",
            ))
            ax.add_patch(Circle(
                (x, y), cell_circle_radius,
                facecolor=color_of(rc_v),
                edgecolor="#222222", linewidth=0.6, zorder=4,
            ))
    ax.set_xlim(x_vals[0] - 0.5, x_vals[-1] + 0.5)
    ax.set_ylim(y_vals[0] - 0.5, y_vals[-1] + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(x_vals)
    ax.set_yticks(y_vals)
    ax.set_xticks(np.asarray(x_vals, dtype=float) - 0.5, minor=True)
    ax.set_yticks(np.asarray(y_vals, dtype=float) - 0.5, minor=True)
    ax.grid(which="minor", color="#bbbbbb", linewidth=0.5)
    ax.tick_params(which="major", labelsize=tick_fontsize, colors="#222222")
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)


# ---- figure assembler -----------------------------------------------------

def _make_grid_figure(
    *,
    n_rows: int, n_cols: int,
    row_labels: list[str],
    col_labels: list[str],
    panel_fn,                # callable(ax, row_index, col_index) -> None
    x_label: str,
    y_label: str,
    suptitle: str | None = None,
    subtitle: str | None = None,
    cell_size: float = 1.6,
    wspace: float = 0.04,
    hspace: float = 0.12,
):
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_size * n_cols + 2.0, cell_size * n_rows + 2.0),
        sharex=True, sharey=True,
        gridspec_kw={"wspace": wspace, "hspace": hspace},
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    # Reserve top space (only if there is a suptitle), right space for
    # arm row labels, and bottom space for the colorbar + strand legend.
    top_margin = 0.86 if (suptitle or subtitle) else 0.94
    fig.subplots_adjust(top=top_margin, bottom=0.18, left=0.09, right=0.84)
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            panel_fn(ax, r, c)
            if r == 0 and col_labels:
                ax.set_title(col_labels[c], fontsize=13,
                             fontweight="bold", color="#222222")
            if c == 0:
                # Y-axis name (the data axis) on every leftmost panel.
                ax.set_ylabel(y_label, fontsize=12, color="#222222")
            if r == n_rows - 1:
                ax.set_xlabel(x_label, fontsize=12, color="#222222")

    # Arm row labels: placed at the right edge of the figure,
    # vertically centered on each row.  Done after the loop so we can
    # read the final axes positions.
    if row_labels:
        for r in range(n_rows):
            bbox = axes[r, -1].get_position()
            y_center = 0.5 * (bbox.y0 + bbox.y1)
            fig.text(0.855, y_center, row_labels[r],
                     ha="left", va="center", rotation=0,
                     fontsize=13, fontweight="bold", color="#222222")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.97)
    if subtitle:
        fig.text(0.5, 0.92, subtitle, ha="center", fontsize=11,
                 style="italic", color="#444444")
    return fig, axes


def _add_colorbar_and_legend(fig, cmap, norm):
    """Place colorbar and the (forward, reverse) strand legend in the
    bottom margin reserved by ``_make_grid_figure``'s subplots_adjust."""
    cbar_ax = fig.add_axes([0.13, 0.07, 0.42, 0.020])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("correctness", fontsize=12, labelpad=4)
    cbar.ax.tick_params(labelsize=10)

    shape_color = "#999999"
    shape_handles = [
        Patch(facecolor=shape_color, edgecolor="#222222", linewidth=0.6),
        _CircleHandle(color=shape_color),
    ]
    fig.legend(
        handles=shape_handles, labels=["forward", "reverse"],
        handler_map={_CircleHandle: _make_circle_handler()},
        ncol=2, loc="lower right",
        bbox_to_anchor=(0.94, 0.04),
        frameon=True, fontsize=11,
        handlelength=1.8, handleheight=1.8,
        handletextpad=0.6, columnspacing=1.4,
    )


def _save(fig, name, subdir=None):
    d = OUT_DIR if subdir is None else OUT_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    out_png = d / f"{name}.png"
    fig.savefig(out_png, dpi=144, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


# ---- figures --------------------------------------------------------------

SINGLE_MOTIF_LEN = 3              # primary motif length for manuscript figs (trinucleotide)
SINGLE_MOTIF_LENGTHS = [1, 2, 3]  # generate one figure set per length, each to its own subdir
COMPOUND_L1L2 = (2, 3)   # filter compound aggregate to this (L1, L2)


def build_fig1(motif_len: int = SINGLE_MOTIF_LEN):
    """Single, no SNV; columns = N values; one motif length per call.

    Call once per entry in ``SINGLE_MOTIF_LENGTHS``; each version is
    saved to ``supplement/figures_priority/motif_L{motif_len}/``."""
    df = pd.read_csv(SINGLE_CSV)
    df = df[df["snv_offset"] == -1]
    df = df[df["motif_len"] == motif_len]
    pooled = pool_single(df, keep=["arm", "N", "delta", "lflank"])
    n_values = sorted(pooled["N"].unique())
    deltas = sorted(pooled["delta"].unique())
    lflanks = sorted(pooled["lflank"].unique())

    cmap = _PROPORTION_CMAP()
    norm = Normalize(0, 1)
    color_of = _proportion_color_fn(cmap, norm)

    def panel(ax, r, c):
        arm = ARM_ROWS[r]
        N = n_values[c]
        sub = pooled[(pooled["arm"] == arm) & (pooled["N"] == N)]
        _draw_panel(ax, sub,
                    x_col="delta", y_col="lflank",
                    x_vals=deltas, y_vals=lflanks,
                    color_of=color_of, tick_fontsize=7)

    fig, _ = _make_grid_figure(
        n_rows=3, n_cols=len(n_values),
        row_labels=[ARM_LABELS[a] for a in ARM_ROWS],
        col_labels=[f"N = {n}" for n in n_values],
        panel_fn=panel,
        x_label="Δ (Hap N − Ref N)",
        y_label="lflank extent",
        cell_size=2.4,
        wspace=0.068,
        hspace=0.075,
    )
    _add_colorbar_and_legend(fig, cmap, norm)
    _save(fig, "fig1__single_noSNV_byN", subdir=f"motif_L{motif_len}")


def build_fig2(N_value: int, motif_len: int = SINGLE_MOTIF_LEN):
    """Single, SNV stack at fixed N; columns = SNV positions (1-indexed).

    Columns show positions 1, 2, 5, 10 (bases from repeat boundary,
    1-indexed) = snv_offset values 0, 1, 4, 9 in the raw data.
    Call once per motif length; saved to motif_L{motif_len}/ subdir."""
    df = pd.read_csv(SINGLE_CSV)
    df = df[df["N"] == N_value]
    df = df[df["motif_len"] == motif_len]
    pooled = pool_single(df, keep=["arm", "snv_offset", "delta", "lflank"])
    snv_cols = [0, 1, 4, 9]
    snv_labels = ["SNV @ 1", "SNV @ 2", "SNV @ 5", "SNV @ 10"]
    deltas = sorted(pooled["delta"].unique())
    lflanks = sorted(pooled["lflank"].unique())

    cmap = _PROPORTION_CMAP()
    norm = Normalize(0, 1)
    color_of = _proportion_color_fn(cmap, norm)

    def panel(ax, r, c):
        arm = ARM_ROWS[r]
        snv = snv_cols[c]
        sub = pooled[(pooled["arm"] == arm) & (pooled["snv_offset"] == snv)]
        _draw_panel(ax, sub,
                    x_col="delta", y_col="lflank",
                    x_vals=deltas, y_vals=lflanks,
                    color_of=color_of, tick_fontsize=7)

    fig, _ = _make_grid_figure(
        n_rows=3, n_cols=len(snv_cols),
        row_labels=[ARM_LABELS[a] for a in ARM_ROWS],
        col_labels=snv_labels,
        panel_fn=panel,
        x_label="Δ (Hap N − Ref N)",
        y_label="lflank extent",
        cell_size=2.4,
        wspace=0.068,
        hspace=0.075,
    )
    _add_colorbar_and_legend(fig, cmap, norm)
    _save(fig, f"fig2__single_snv_stack_N{N_value:02d}", subdir=f"motif_L{motif_len}")


def _load_compound_for_monodi():
    df = pd.read_csv(COMPOUND_CSV_FULL)
    # (mono, di) and (di, mono) both, at N1=N2=10
    mask = (
        (df["N1"] == 10) & (df["N2"] == 10) &
        (
            ((df["motif1_len"] == 1) & (df["motif2_len"] == 2)) |
            ((df["motif1_len"] == 2) & (df["motif2_len"] == 1))
        )
    )
    return df[mask].copy()


def _pool_compound(df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    g = df.groupby(keep, dropna=False, sort=False)
    pooled = g.apply(
        lambda d: pd.Series({
            "fwd": _weighted_mean(d, "frac_score_eq_truth_fwd"),
            "rc":  _weighted_mean(d, "frac_score_eq_truth_rc"),
            "n":   d["n_loci"].sum(),
        }),
        include_groups=False,
    ).reset_index()
    return pooled


def build_fig3():
    """Compound (mono, di) bridge stack; columns = |M|."""
    df = _load_compound_for_monodi()
    if df.empty:
        print("  fig3: no (1,2)+(2,1) data yet — large_transposed pending")
        return
    pooled = _pool_compound(df, keep=["arm", "bridge_len", "delta1", "delta2"])
    bridges = [1, 2, 3, 5]
    deltas1 = sorted(pooled["delta1"].unique())
    deltas2 = sorted(pooled["delta2"].unique())

    cmap = _PROPORTION_CMAP()
    norm = Normalize(0, 1)
    color_of = _proportion_color_fn(cmap, norm)

    def panel(ax, r, c):
        arm = ARM_ROWS[r]
        M = bridges[c]
        sub = pooled[(pooled["arm"] == arm) & (pooled["bridge_len"] == M)]
        _draw_panel(ax, sub,
                    x_col="delta1", y_col="delta2",
                    x_vals=deltas1, y_vals=deltas2,
                    color_of=color_of, tick_fontsize=7)

    fig, _ = _make_grid_figure(
        n_rows=3, n_cols=len(bridges),
        row_labels=[ARM_LABELS[a] for a in ARM_ROWS],
        col_labels=[f"|M| = {M}" for M in bridges],
        panel_fn=panel,
        x_label="Δ₁",
        y_label="Δ₂",
        cell_size=2.4,
        wspace=0.068,
        hspace=0.068,
    )
    _add_colorbar_and_legend(fig, cmap, norm)
    _save(fig, "fig3__compound_monodi_bridge_stack")


def build_fig4(M: int):
    """Compound BWA-MEM (no clip) only — motif1_len × motif2_len grid at one |M|."""
    df = pd.read_csv(COMPOUND_CSV_FULL)
    df = df[(df["arm"] == "BWA-no-clip") &
            (df["bridge_len"] == M) &
            (df["N1"] == 10) & (df["N2"] == 10)]
    if df.empty:
        print(f"  fig4 (|M|={M}): no data")
        return
    pooled = _pool_compound(df, keep=["motif1_len", "motif2_len",
                                       "delta1", "delta2"])
    row_lens = [1, 2, 3]
    col_lens = [1, 2, 3]
    deltas1 = sorted(pooled["delta1"].unique())
    deltas2 = sorted(pooled["delta2"].unique())

    cmap = _PROPORTION_CMAP()
    norm = Normalize(0, 1)
    color_of = _proportion_color_fn(cmap, norm)

    def panel(ax, r, c):
        L2 = row_lens[r]; L1 = col_lens[c]
        sub = pooled[(pooled["motif1_len"] == L1) &
                     (pooled["motif2_len"] == L2)]
        _draw_panel(ax, sub,
                    x_col="delta1", y_col="delta2",
                    x_vals=deltas1, y_vals=deltas2,
                    color_of=color_of, tick_fontsize=7)

    name_for = {1: "mono", 2: "di", 3: "tri"}
    fig, _ = _make_grid_figure(
        n_rows=3, n_cols=3,
        row_labels=[f"R₂ = {name_for[L]}\n({L}-mer)" for L in row_lens],
        col_labels=[f"R₁ = {name_for[L]}\n({L}-mer)" for L in col_lens],
        panel_fn=panel,
        x_label="Δ₁",
        y_label="Δ₂",
        cell_size=2.4,
        wspace=0.068,
        hspace=0.068,
    )
    _add_colorbar_and_legend(fig, cmap, norm)
    _save(fig, f"fig4__compound_bwaNoClip_LpairxLpair_M{M}")


# ---- aggregate (single-axis) line plots ----------------------------------

_ARM_COLORS = {
    "BWA-std":     "#D55E00",   # orange-red
    "BWA-no-clip": "#0072B2",   # blue
    "NW-flex":     "#009E73",   # green
}


_BETA_RNG = np.random.default_rng(20260513)
_BETA_NSAMPLES = 20000


def _beta_mom_band(values, weights, q_low=0.16, q_high=0.84):
    """Beta-method-of-moments band for a sample of fractions in [0,1].

    Returns (mean, low, high).  q_low=0.16/q_high=0.84 mirror the
    coverage of ±1 SD under a normal but stay strictly within [0, 1].

    Quantiles are taken from a Monte-Carlo draw from Beta(α, β) (so we
    don't depend on scipy).  20k samples gives 16th/84th-percentile
    accuracy of roughly ±0.003.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    w_sum = w.sum()
    if w_sum <= 0:
        return float("nan"), float("nan"), float("nan")
    mu = (v * w).sum() / w_sum
    var = ((v - mu) ** 2 * w).sum() / w_sum
    # Boundary / degenerate cases: collapse the band to the mean.
    if not (0 < mu < 1) or var <= 1e-12:
        return mu, mu, mu
    # Beta MOM: solve mu = α/(α+β), var = αβ / [(α+β)² (α+β+1)]
    nu = mu * (1 - mu) / var - 1
    if nu <= 0:
        # Empirical variance too large for any Beta (more dispersed
        # than uniform).  Fall back to a clipped ±1 SD band.
        sd = var ** 0.5
        return mu, max(0.0, mu - sd), min(1.0, mu + sd)
    alpha = mu * nu
    beta_ = (1 - mu) * nu
    samples = _BETA_RNG.beta(alpha, beta_, size=_BETA_NSAMPLES)
    return mu, float(np.quantile(samples, q_low)), float(np.quantile(samples, q_high))


def _line_pool_by(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """Per (arm, x) cell: weighted mean of fwd+rc correctness, plus a
    Beta-MOM 16th/84th-percentile band (bounded in [0,1])."""
    df = df.copy()
    df["avg"] = 0.5 * (df["frac_score_eq_truth_fwd"]
                       + df["frac_score_eq_truth_rc"])
    g = df.groupby(["arm", x_col], dropna=False, sort=True)
    def _summ(d):
        mu, lo, hi = _beta_mom_band(d["avg"].values, d["n_loci"].values)
        return pd.Series({
            "mean": mu, "low": lo, "high": hi,
            "n":    d["n_loci"].sum(),
        })
    pooled = g.apply(_summ, include_groups=False).reset_index()
    return pooled


def _plot_arm_lines(ax, pooled, x_col, x_label, dodge=0.08,
                     show_legend=True, show_ylabel=True):
    """Line plot of mean correctness per arm with asymmetric capped
    error bars from a Beta-MOM 16th/84th-percentile band (bounded
    in [0, 1]), dodged horizontally so adjacent arms don't overlap."""
    offsets = {arm: (i - 1) * dodge for i, arm in enumerate(ARM_ROWS)}
    for arm in ARM_ROWS:
        sub = pooled[pooled["arm"] == arm].sort_values(x_col)
        x = sub[x_col].values.astype(float)
        m = sub["mean"].values
        lo = sub["low"].values
        hi = sub["high"].values
        yerr = np.vstack([np.maximum(m - lo, 0), np.maximum(hi - m, 0)])
        c = _ARM_COLORS[arm]
        ax.errorbar(
            x + offsets[arm], m, yerr=yerr,
            fmt="-o", color=c, ecolor=c,
            label=ARM_LABELS[arm],
            linewidth=2.2, markersize=6,
            capsize=3.5, capthick=1.2, elinewidth=1.2,
            alpha=0.95,
        )
    ax.set_xlabel(x_label, fontsize=14)
    if show_ylabel:
        ax.set_ylabel("correctness", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle=":", color="#bbbbbb", alpha=0.6)
    if show_legend:
        ax.legend(loc="best", fontsize=12, frameon=True)


def build_A1():
    """Correctness vs Δ, single-locus, no SNV, pooled over N + motif + lflank."""
    df = pd.read_csv(SINGLE_CSV)
    df = df[df["snv_offset"] == -1]
    pooled = _line_pool_by(df, "delta")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _plot_arm_lines(ax, pooled, "delta", "Δ (Hap N − Ref N)",
                    title="Single-locus correctness vs Δ "
                          "(no SNV; pooled over N, motif, lflank)")
    ax.axvline(0, color="#888888", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    _save(fig, "A1__corr_vs_delta_single_noSNV")


def build_A2(delta_fixed: int = -3):
    """Correctness vs lflank, single-locus, no SNV, at one Δ slice."""
    df = pd.read_csv(SINGLE_CSV)
    df = df[(df["snv_offset"] == -1) & (df["delta"] == delta_fixed)]
    pooled = _line_pool_by(df, "lflank")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _plot_arm_lines(ax, pooled, "lflank", "lflank extent (bases)",
                    title=f"Single-locus correctness vs lflank "
                          f"(no SNV, Δ = {delta_fixed}; pooled over N, motif)")
    fig.tight_layout()
    _save(fig, f"A2__corr_vs_lflank_single_noSNV_delta{delta_fixed:+d}")


def build_A3():
    """Correctness vs SNV offset, single-locus, pooled over N + motif + Δ + lflank."""
    df = pd.read_csv(SINGLE_CSV)
    pooled = _line_pool_by(df, "snv_offset")
    # Order: numeric ascending, then -1 (no SNV) last; remap x for plotting.
    nums = sorted([s for s in pooled["snv_offset"].unique() if s != -1])
    order = nums + ([-1] if -1 in pooled["snv_offset"].values else [])
    order_index = {v: i for i, v in enumerate(order)}
    pooled = pooled.copy()
    pooled["x_pos"] = pooled["snv_offset"].map(order_index)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm in ARM_ROWS:
        sub = pooled[pooled["arm"] == arm].sort_values("x_pos")
        x = sub["x_pos"].values
        m = sub["mean"].values
        sd = sub["sd"].values
        c = _ARM_COLORS[arm]
        ax.plot(x, m, "-o", color=c, label=ARM_LABELS[arm],
                linewidth=2.0, markersize=5)
        ax.fill_between(x, m - sd, m + sd, color=c, alpha=0.15, linewidth=0)
    ax.set_xticks(list(order_index.values()))
    ax.set_xticklabels([("none" if v == -1 else str(v + 1)) for v in order])
    ax.set_xlabel("SNV position in left flank (1 = repeat boundary)", fontsize=10)
    ax.set_ylabel("fraction with score = truth (fwd+rc avg)", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", color="#bbbbbb", alpha=0.7)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.set_title("Single-locus correctness vs SNV position "
                 "(pooled over N, motif, Δ, lflank)", fontsize=11)
    fig.tight_layout()
    _save(fig, "A3__corr_vs_snv_offset_single")


def _per_locus_summary(df: pd.DataFrame, x_col: str,
                        value_col: str = "correct") -> pd.DataFrame:
    """Per (arm, x_col): take the per-locus `value_col` (each row is one
    locus), compute mean and Beta-MOM 16/84% band across loci."""
    g = df.groupby(["arm", x_col], dropna=False, sort=True)
    def _summ(d):
        # Equal weights since each row is one locus.
        w = np.ones(len(d))
        mu, lo, hi = _beta_mom_band(d[value_col].values, w)
        return pd.Series({"mean": mu, "low": lo, "high": hi, "n": len(d)})
    return g.apply(_summ, include_groups=False).reset_index()


# Per-locus inputs are written by scripts/aggregate_per_locus_for_A.py.
A1_PER_LOCUS_CSV = REPO_ROOT / "supplement/data/A1_per_pind__delta.csv"
A2_PER_LOCUS_CSV = REPO_ROOT / "supplement/data/A2_per_pind__lflank.csv"
A3_PER_LOCUS_CSV = REPO_ROOT / "supplement/data/A3_per_pind__snv.csv"
A4_PER_LOCUS_CSV = REPO_ROOT / "supplement/data/A4_per_pair__bridge.csv"


def _build_single_panel(pool_df, x_col, x_label, *, outname,
                         show_legend=False, xticks=None,
                         xticklabels=None, axvline_at=None):
    """Render one aggregate line-plot panel as a self-contained PDF/PNG.
    Panel letters and the figure caption come from the LaTeX wrapper
    (``supplement/tex/aggregate_figure.tex``); the legend is shown on
    only one panel so the assembled figure has a single legend."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    fig.patch.set_facecolor("white")
    _plot_arm_lines(ax, pool_df, x_col, x_label,
                    show_ylabel=True, show_legend=show_legend)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if axvline_at is not None:
        ax.axvline(axvline_at, color="#888888",
                    linestyle="--", linewidth=0.7)
    fig.tight_layout()
    _save(fig, outname)


def build_A_panels():
    """Four standalone PDFs — one per panel — for LaTeX assembly via
    ``supplement/tex/aggregate_figure.tex``.  Only the first PDF
    carries the legend; the LaTeX wrapper labels the panels."""
    A1_pool = _per_locus_summary(pd.read_csv(A1_PER_LOCUS_CSV), "delta")
    A2_pool = _per_locus_summary(pd.read_csv(A2_PER_LOCUS_CSV), "lflank")
    A3_raw = pd.read_csv(A3_PER_LOCUS_CSV)
    A3_pool_raw = _per_locus_summary(A3_raw, "snv_offset")
    nums = sorted([s for s in A3_pool_raw["snv_offset"].unique() if s != -1])
    snv_order = nums + ([-1] if -1 in A3_pool_raw["snv_offset"].values else [])
    snv_pos = {v: i for i, v in enumerate(snv_order)}
    A3_pool = A3_pool_raw.copy()
    A3_pool["x_pos"] = A3_pool["snv_offset"].map(snv_pos)
    A4_pool = _per_locus_summary(pd.read_csv(A4_PER_LOCUS_CSV), "bridge_len")

    _build_single_panel(
        A1_pool, "delta", "Delta",
        outname="A__delta", axvline_at=0,
    )
    _build_single_panel(
        A2_pool, "lflank", "Flank Extent",
        outname="B__flank",
    )
    _build_single_panel(
        A3_pool, "x_pos", "SNV position",
        xticks=list(snv_pos.values()),
        xticklabels=[("none" if v == -1 else f"+{v}") for v in snv_order],
        outname="C__snv",
    )
    _build_single_panel(
        A4_pool, "bridge_len", "length of |M|",
        xticks=sorted(A4_pool["bridge_len"].unique()),
        outname="D__bridge", show_legend=True,
    )


def build_A4():
    """Correctness vs |M|, compound, N=10/10, pooled over motif pairs and (Δ1,Δ2)."""
    df = pd.read_csv(COMPOUND_CSV_FULL)
    df = df[(df["N1"] == 10) & (df["N2"] == 10)].copy()
    if df.empty:
        print("  A4: no compound data at N=10/10 yet")
        return
    df["avg"] = 0.5 * (df["frac_score_eq_truth_fwd"]
                       + df["frac_score_eq_truth_rc"])
    g = df.groupby(["arm", "bridge_len"], dropna=False, sort=True)
    pooled = g.apply(
        lambda d: pd.Series({
            "mean": (d["avg"] * d["n_loci"]).sum() / d["n_loci"].sum(),
            "sd":   d["avg"].std(ddof=0),
            "n":    d["n_loci"].sum(),
        }),
        include_groups=False,
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _plot_arm_lines(ax, pooled, "bridge_len", "|M| (bridge length, bp)",
                    title="Compound correctness vs bridge length "
                          "(N₁=N₂=10; pooled over motif pairs and Δ₁,Δ₂)")
    ax.set_xticks(sorted(pooled["bridge_len"].unique()))
    fig.tight_layout()
    _save(fig, "A4__corr_vs_bridge_compound")


# ---- driver ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figs", nargs="+", default=["all"],
                        help="Subset: 1, 2, 3, 4, or 'all'.")
    args = parser.parse_args()
    selected = set(args.figs)
    do_all = "all" in selected

    if do_all or "1" in selected:
        for L in SINGLE_MOTIF_LENGTHS:
            print(f"=== fig 1 (motif_len = {L}) ===")
            build_fig1(L)
    if do_all or "2" in selected:
        for L in SINGLE_MOTIF_LENGTHS:
            print(f"=== fig 2 (N = 10, motif_len = {L}) ===")
            build_fig2(10, L)
    if do_all or "3" in selected:
        print("=== fig 3 ===")
        build_fig3()
    if do_all or "4" in selected:
        for M in [1, 2, 3]:
            print(f"=== fig 4 (|M| = {M}) ===")
            build_fig4(M)
    if do_all or "A" in selected or "aggregate" in selected:
        print("=== A panels (four separate PDFs) ===")
        build_A_panels()


if __name__ == "__main__":
    main()
