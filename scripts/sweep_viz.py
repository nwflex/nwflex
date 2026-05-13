"""Plotting helpers for the comprehensive batch sweep.

These functions are kept separate from ``nwflex.simulation.viz`` until
the in-flight viz refactor stabilizes; once it does they should be
folded into the package next to ``plot_proportion_heatmap_2d``.

The single function exported here is :func:`plot_proportion_heatmap`
(plus its multi-row variant) — a 1-D continuous-color heatmap on
``(Δ × lflank)`` axes that mirrors the 2-D proportion plot exported
from the package.  Used by ``scripts/aggregate_results.py`` to render
single-repeat cross-locus figures.
"""
from __future__ import annotations

from typing import Iterable, Mapping

from nwflex.simulation.viz import (
    _CircleHandle,
    _PROPORTION_CMAP,
    _make_circle_handler,
    _proportion_color_fn,
)


def _proportion_value_fn_1d(cell_df):
    """``cell_value_fn`` for fraction-of-(P or T) per strand on a 1-D
    ``(delta, lflank)`` cell."""
    fwd = [s for s in cell_df["fwd_state"].tolist() if isinstance(s, str)]
    rc  = [s for s in cell_df["rc_state"].tolist()  if isinstance(s, str)]
    f = sum(1 for s in fwd if s in ("P", "T")) / len(fwd) if fwd else float("nan")
    r = sum(1 for s in rc  if s in ("P", "T")) / len(rc)  if rc  else float("nan")
    return f, r


def _draw_1d_grid_panel(ax, sub_df, *, deltas, lflanks,
                        cell_value_fn, color_fn, fontsize):
    """Draw one ``(Δ × lflank)`` panel using a pluggable cell-value
    function.  Mirrors ``viz._draw_state_panel``'s glyph convention
    (fwd Rectangle, rc Circle)."""
    import numpy as np
    from matplotlib.patches import Circle, Rectangle

    for L in lflanks:
        for D in deltas:
            cell = sub_df[(sub_df["delta"] == D) & (sub_df["lflank"] == L)]
            fwd_v, rc_v = cell_value_fn(cell)
            ax.add_patch(Rectangle(
                (D - 0.5, L - 0.5), 1, 1,
                facecolor=color_fn(fwd_v), edgecolor="none", linewidth=0,
            ))
            ax.add_patch(Circle(
                (D, L), 0.26,
                facecolor=color_fn(rc_v),
                edgecolor="#222222", linewidth=0.6, zorder=4,
            ))

    ax.set_xlim(deltas[0] - 0.5, deltas[-1] + 0.5)
    ax.set_ylim(lflanks[0] - 0.5, lflanks[-1] + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(deltas)
    ax.set_yticks(lflanks)
    ax.set_xticks(np.array(deltas, dtype=float) - 0.5, minor=True)
    ax.set_yticks(np.array(lflanks, dtype=float) - 0.5, minor=True)
    ax.grid(which="minor", color="#bbbbbb", linewidth=0.5)
    ax.tick_params(which="major", labelsize=fontsize, colors="#222222")
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    if 0 in deltas:
        ax.add_patch(Rectangle(
            (-0.5, lflanks[0] - 0.5),
            1, lflanks[-1] - lflanks[0] + 1,
            fill=False, edgecolor="black", linewidth=1.5, zorder=5,
        ))


def plot_proportion_heatmap(
    df,
    *,
    deltas: Iterable[int],
    lflanks: Iterable[int],
    arm_titles: Mapping[str, str],
    fontsize: int = 14,
    suptitle: str | None = None,
    subtitle: str | None = None,
    cbar_label: str = "fraction of reads with score = truth",
):
    """1-D analog of
    :func:`nwflex.simulation.viz.plot_proportion_heatmap_2d`.

    ``df`` must carry columns ``arm``, ``delta``, ``lflank``, and
    ``fwd_state`` + ``rc_state``.  Rows within a ``(delta, lflank,
    arm)`` cell are aggregated by the proportion-value function; for a
    cross-locus view, each row corresponds to one
    ``(locus, delta, lflank, arm)`` observation and the proportion is
    over loci.

    Returns the :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    deltas = list(deltas)
    lflanks = list(lflanks)
    arms = list(arm_titles.keys())
    n = len(arms)
    label_size = fontsize + 1
    title_size = fontsize + 3

    cmap = _PROPORTION_CMAP()
    norm = Normalize(vmin=0.0, vmax=1.0)
    color_of = _proportion_color_fn(cmap, norm)

    fig, axes = plt.subplots(
        1, n, figsize=(5.0 * n + 2.5, 6.6), sharey=True,
        gridspec_kw={"wspace": 0.06},
        subplot_kw={"facecolor": "white"},
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for c, arm in enumerate(arms):
        ax = axes[0, c]
        _draw_1d_grid_panel(
            ax, df[df["arm"] == arm],
            deltas=deltas, lflanks=lflanks,
            cell_value_fn=_proportion_value_fn_1d,
            color_fn=color_of, fontsize=fontsize,
        )
        ax.set_xlabel("Δ (Hap N $-$ Ref N)",
                      fontsize=label_size, color="#222222")
        ax.set_title(arm_titles[arm], fontsize=title_size,
                     color="#222222")
    axes[0, 0].set_ylabel("lflank extent",
                          fontsize=label_size, color="#222222")

    panel_top = 0.74 if suptitle else 0.80
    fig.subplots_adjust(left=0.07, right=0.98, top=panel_top, bottom=0.10)

    cbar_ax = fig.add_axes([0.10, panel_top + 0.055, 0.45, 0.025])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label(cbar_label, fontsize=fontsize, color="#222222")
    cbar.ax.tick_params(labelsize=fontsize - 1, colors="#222222")

    shape_color = "#999999"
    shape_handles = [
        Patch(facecolor=shape_color, edgecolor="#222222", linewidth=0.6),
        _CircleHandle(color=shape_color),
    ]
    fig.legend(
        handles=shape_handles, labels=["forward", "reverse"],
        handler_map={_CircleHandle: _make_circle_handler()},
        ncol=2, loc="lower left",
        bbox_to_anchor=(0.62, panel_top + 0.04),
        frameon=True, fontsize=fontsize,
        handlelength=1.9, handleheight=1.9,
        handletextpad=0.6, columnspacing=1.6,
        borderpad=0.7,
        facecolor="white", edgecolor="#444444", labelcolor="#222222",
    )

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=title_size + 1, y=0.985,
                     fontweight="bold", color="#222222")
    if subtitle is not None:
        sub_y = 0.95 if suptitle else 0.97
        fig.text(0.5, sub_y, subtitle, ha="center", va="top",
                 fontsize=label_size, style="italic", color="#444444")
    return fig


def plot_proportion_heatmap_rows(
    rows,
    *,
    deltas: Iterable[int],
    lflanks: Iterable[int],
    arm_titles: Mapping[str, str],
    row_label_fn,
    fontsize: int = 14,
    scale: float = 1.0,
    font_scale: float = 1.0,
    suptitle: str | None = None,
    subtitle: str | None = None,
    cbar_label: str = "fraction of reads with score = truth",
    cell_value_fn=_proportion_value_fn_1d,
):
    """Multi-row variant of :func:`plot_proportion_heatmap`.  ``rows``
    is a list of ``(key, df_subset)`` pairs.

    ``cell_value_fn`` defaults to the per-locus state-counting function,
    matching the original cross-locus behavior.  Pass a custom function
    to plot pre-aggregated fractions directly (one row per cell with
    ``frac_fwd`` / ``frac_rc`` columns)."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    deltas = list(deltas)
    lflanks = list(lflanks)
    arms = list(arm_titles.keys())
    rows = list(rows)
    n_rows = len(rows)
    n_cols = len(arms)
    label_size = (fontsize + 1) * font_scale
    title_size = (fontsize + 3) * font_scale
    fontsize = fontsize * font_scale

    cmap = _PROPORTION_CMAP()
    norm = Normalize(vmin=0.0, vmax=1.0)
    color_of = _proportion_color_fn(cmap, norm)

    figsize = (
        scale * (5.0 * n_cols + 0.8),
        scale * (5.0 * n_rows + 1.6),
    )
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.02, "hspace": 0.18},
        subplot_kw={"facecolor": "white"},
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for r, (key, df) in enumerate(rows):
        for c, arm in enumerate(arms):
            ax = axes[r, c]
            _draw_1d_grid_panel(
                ax, df[df["arm"] == arm],
                deltas=deltas, lflanks=lflanks,
                cell_value_fn=cell_value_fn,
                color_fn=color_of, fontsize=fontsize,
            )
            if r == 0:
                ax.set_title(arm_titles[arm], fontsize=title_size,
                             color="#222222")
            if r == n_rows - 1:
                ax.set_xlabel("Δ (Hap N $-$ Ref N)",
                              fontsize=label_size, color="#222222")
        axes[r, 0].set_ylabel(
            f"{row_label_fn(key)}\nlflank extent",
            fontsize=label_size, color="#222222",
        )

    panel_top = 0.84 if suptitle else 0.88
    fig.subplots_adjust(left=0.07, right=0.98, top=panel_top, bottom=0.05)

    cbar_ax = fig.add_axes([0.10, panel_top + 0.045, 0.45, 0.018])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label(cbar_label, fontsize=fontsize, color="#222222")
    cbar.ax.tick_params(labelsize=fontsize - 1, colors="#222222")

    shape_color = "#999999"
    shape_handles = [
        Patch(facecolor=shape_color, edgecolor="#222222", linewidth=0.6),
        _CircleHandle(color=shape_color),
    ]
    fig.legend(
        handles=shape_handles, labels=["forward", "reverse"],
        handler_map={_CircleHandle: _make_circle_handler()},
        ncol=2, loc="lower left",
        bbox_to_anchor=(0.62, panel_top + 0.04),
        frameon=True, fontsize=fontsize,
        handlelength=1.9, handleheight=1.9,
        handletextpad=0.6, columnspacing=1.6,
        borderpad=0.7,
        facecolor="white", edgecolor="#444444", labelcolor="#222222",
    )

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=title_size + 1, y=0.985,
                     fontweight="bold", color="#222222")
    if subtitle is not None:
        sub_y = 0.96 if suptitle else 0.985
        fig.text(0.5, sub_y, subtitle, ha="center", va="top",
                 fontsize=label_size, style="italic", color="#444444")
    return fig
