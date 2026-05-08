"""
viz.py — visualizations for the simulation harness.

- :func:`render_zoom` produces a column-aligned ASCII view of one
  alignment around a repeat-zone interval.
- :func:`plot_correctness_heatmap` produces the three-panel
  (Δ × lflank) correct/wrong heatmap used by the validation cells.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from .core import parse_cigar


def render_zoom(
    ref: str,
    read: str,
    pos: int,
    cigar: str,
    z_start: int,
    z_end: int,
    *,
    pad: int = 15,
) -> str:
    """
    Build a column-aligned ASCII zoom of one alignment around the
    repeat zone ``[z_start, z_end)``.

    Each CIGAR op contributes columns:

    - ``M``/``=``/``X`` — one ref base above one read base.
    - ``I`` — a ``-`` on the ref line above the inserted read base; a
      column marker ``I`` is emitted underneath.
    - ``D`` — a ``-`` on the read line; column marker ``D``.
    - ``N`` — a ``-`` on the read line; column marker ``N`` (skipped
      reference, distinct from a deletion).
    - ``S`` (soft-clip) — read bases shown bracketed (``[...]``) off
      the relevant end; not laid out as columns.

    Pipe characters (``|``) mark transitions between left flank, repeat
    zone, and right flank.  The view is trimmed to ``pad`` columns on
    each side of the zone.

    Parameters
    ----------
    ref : str
        Reference sequence the alignment is against.
    read : str
        Forward-strand read sequence (as supplied to BWA).
    pos : int
        1-based reference position of the first aligned read base
        (SAM ``POS``).
    cigar : str
        CIGAR string.
    z_start, z_end : int
        Half-open repeat interval in the reference (0-based).
    pad : int, default 15
        Columns to show on each side of the repeat zone.

    Returns
    -------
    str
        Multi-line block: a ``ref :`` line, a ``read:`` line, and (when
        any insertion or deletion appears) a third marker line.
    """
    ops = parse_cigar(cigar)
    ref_pos = pos - 1
    read_pos = 0

    # Peel soft-clips off the ends — they don't get column-aligned.
    left_clip = right_clip = ""
    if ops and ops[0][1] == "S":
        n = ops[0][0]
        left_clip = read[read_pos : read_pos + n]
        read_pos += n
        ops = ops[1:]
    if ops and ops[-1][1] == "S":
        n = ops[-1][0]
        right_clip = read[len(read) - n :]
        ops = ops[:-1]

    # Walk the rest of the CIGAR into per-column lists.
    cols_ref: list[str] = []
    cols_read: list[str] = []
    cols_zone: list[str] = []
    cols_op: list[str] = []

    def zone_of(p: int) -> str:
        if p < z_start:
            return "L"
        if p < z_end:
            return "Z"
        return "R"

    for length, op in ops:
        if op in ("M", "=", "X"):
            for _ in range(length):
                cols_ref.append(ref[ref_pos])
                cols_read.append(read[read_pos])
                cols_zone.append(zone_of(ref_pos))
                cols_op.append(op)
                ref_pos += 1
                read_pos += 1
        elif op == "I":
            # Insertion sits "between" ref bases; bin it with the next
            # ref position so that an I at ref_pos == z_start lands in
            # zone Z (matching the bwa boundary convention).
            for _ in range(length):
                cols_ref.append("-")
                cols_read.append(read[read_pos])
                cols_zone.append(zone_of(ref_pos))
                cols_op.append("I")
                read_pos += 1
        elif op in ("D", "N"):
            for _ in range(length):
                cols_ref.append(ref[ref_pos])
                cols_read.append("-")
                cols_zone.append(zone_of(ref_pos))
                cols_op.append(op)
                ref_pos += 1
        # H, P, S contribute no columns (S already peeled).

    # Choose the visible window: ``pad`` columns of context on each
    # side of the zone.  When the alignment touches no Z column (rare
    # for this notebook), show everything.
    z_cols = [i for i, z in enumerate(cols_zone) if z == "Z"]
    if z_cols:
        a = max(0, z_cols[0] - pad)
        b = min(len(cols_zone), z_cols[-1] + 1 + pad)
    else:
        a, b = 0, len(cols_zone)

    ref_buf: list[str] = []
    read_buf: list[str] = []
    mark_buf: list[str] = []

    if left_clip:
        pad_str = " " * (len(left_clip) + 2)
        ref_buf.append(pad_str)
        read_buf.append(f"[{left_clip}]")
        mark_buf.append(pad_str)

    prev_zone: str | None = None
    for i in range(a, b):
        z = cols_zone[i]
        if prev_zone is not None and z != prev_zone:
            ref_buf.append("|")
            read_buf.append("|")
            mark_buf.append(" ")
        ref_buf.append(cols_ref[i])
        read_buf.append(cols_read[i])
        op = cols_op[i]
        if op in ("I", "D", "N"):
            mark_buf.append(op)
        else:
            mark_buf.append(" ")
        prev_zone = z

    if right_clip:
        ref_buf.append("  ")
        read_buf.append(f"[{right_clip}]")
        mark_buf.append(" ")

    ref_line = "".join(ref_buf)
    read_line = "".join(read_buf)
    mark_line = "".join(mark_buf)

    out = [f"ref :  {ref_line}", f"read:  {read_line}"]
    if any(c != " " for c in mark_line):
        out.append(f"       {mark_line}")
    return "\n".join(out)


class _SplitCellHandle:
    """Sentinel for the split-cell legend entry."""

    def __init__(self, lower_color: str, upper_color: str):
        self.lower_color = lower_color
        self.upper_color = upper_color


def _make_split_cell_handler():
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.patches import Polygon as _Polygon

    class _SplitCellHandler(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            bl = (-xdescent, -ydescent)
            br = (-xdescent + width, -ydescent)
            tl = (-xdescent, -ydescent + height)
            tr = (-xdescent + width, -ydescent + height)
            lower = _Polygon(
                [bl, br, tl], facecolor=orig_handle.lower_color,
                edgecolor="#bbbbbb", linewidth=0.5, transform=trans,
            )
            upper = _Polygon(
                [tr, br, tl], facecolor=orig_handle.upper_color,
                edgecolor="#bbbbbb", linewidth=0.5, transform=trans,
            )
            return [lower, upper]

    return _SplitCellHandler()


def plot_correctness_heatmap(
    df,
    *,
    deltas: Iterable[int],
    lflanks: Iterable[int],
    arm_titles: Mapping[str, str],
    color_pass: str = "#08519c",        # dark blue
    color_tied: str = "#6baed6",        # light blue
    color_missed: str = "#ffffff",      # white
    color_outscored: str = "#c0392b",   # red
    color_nan: str = "#cccccc",
    fontsize: int = 14,
):
    """
    Three-panel four-state "two-triangle" heatmap for a
    (Δ × lflank × arm) sweep.

    Each cell is split diagonally (no edge between halves):

    - lower-left triangle = forward strand
    - upper-right triangle = reverse-complement strand

    Cells where the two strands classify the same render as a single
    flat color; cells where the strands disagree show two colors split
    along the cell's anti-diagonal.  For single-strand arms (e.g.
    NW-flex), set ``fwd_state == rc_state`` and the cell renders as
    one color.

    Each state's color:

    - ``"P"`` (pass)      → dark blue
    - ``"T"`` (tied)      → light blue
    - ``"M"`` (missed)    → white
    - ``"D"`` (outscored) → red
    - NaN (infeasible / missing) → grey

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form, with columns ``delta``, ``lflank``, ``arm`` and
        either ``fwd_state`` + ``rc_state``, or just ``state`` (in
        which case both triangles share that state).
    deltas, lflanks, arm_titles
        See module docstring; lflank axis runs high → top.
    fontsize : int
        Base font size in points.  Tick labels use ``fontsize``,
        axis labels use ``fontsize + 1``, panel titles use
        ``fontsize + 3``, legend uses ``fontsize``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon, Rectangle

    deltas = list(deltas)
    lflanks = list(lflanks)
    arms = list(arm_titles.keys())
    n = len(arms)

    label_size = fontsize + 1
    title_size = fontsize + 3

    state_to_color = {
        "P": color_pass,
        "T": color_tied,
        "M": color_missed,
        "D": color_outscored,
    }

    def color_of(state):
        if isinstance(state, str) and state in state_to_color:
            return state_to_color[state]
        return color_nan

    has_per_strand = "fwd_state" in df.columns and "rc_state" in df.columns

    fig, axes = plt.subplots(
        1, n, figsize=(5.0 * n + 2.5, 6.0), sharey=True,
        gridspec_kw={"wspace": 0.06},
    )
    if n == 1:
        axes = [axes]

    for ax, arm in zip(axes, arms):
        sub = df[df["arm"] == arm]
        if has_per_strand:
            fwd_grid = sub.pivot(index="lflank", columns="delta", values="fwd_state")
            rc_grid  = sub.pivot(index="lflank", columns="delta", values="rc_state")
        else:
            single = sub.pivot(index="lflank", columns="delta", values="state")
            fwd_grid = rc_grid = single
        fwd_grid = fwd_grid.reindex(index=lflanks, columns=deltas)
        rc_grid  = rc_grid.reindex(index=lflanks, columns=deltas)

        for li, L in enumerate(lflanks):
            for di, D in enumerate(deltas):
                fwd_c = color_of(fwd_grid.iat[li, di])
                rc_c  = color_of(rc_grid.iat[li, di])
                if fwd_c == rc_c:
                    # Strands agree — single rectangle, no diagonal seam.
                    ax.add_patch(Rectangle(
                        (D - 0.5, L - 0.5), 1, 1,
                        facecolor=fwd_c, edgecolor="none", linewidth=0,
                    ))
                else:
                    # Strands disagree — two triangles meeting on the
                    # anti-diagonal.
                    bl = (D - 0.5, L - 0.5)
                    br = (D + 0.5, L - 0.5)
                    tl = (D - 0.5, L + 0.5)
                    tr = (D + 0.5, L + 0.5)
                    ax.add_patch(Polygon(
                        [bl, br, tl], facecolor=fwd_c, edgecolor="none",
                        linewidth=0, antialiased=False,
                    ))
                    ax.add_patch(Polygon(
                        [tr, br, tl], facecolor=rc_c, edgecolor="none",
                        linewidth=0, antialiased=False,
                    ))

        ax.set_xlim(deltas[0] - 0.5, deltas[-1] + 0.5)
        ax.set_ylim(lflanks[0] - 0.5, lflanks[-1] + 0.5)
        ax.set_aspect("equal")
        ax.set_xticks(deltas)
        ax.set_yticks(lflanks)
        ax.set_xticks(np.array(deltas, dtype=float) - 0.5, minor=True)
        ax.set_yticks(np.array(lflanks, dtype=float) - 0.5, minor=True)
        ax.grid(which="minor", color="#bbbbbb", linewidth=0.5)
        ax.tick_params(which="major", labelsize=fontsize)
        ax.tick_params(which="minor", length=0)
        # Highlight the Δ=0 column (haplotype == reference) when present.
        if 0 in deltas:
            ax.add_patch(Rectangle(
                (-0.5, lflanks[0] - 0.5),
                1, lflanks[-1] - lflanks[0] + 1,
                fill=False, edgecolor="black", linewidth=1.5, zorder=5,
            ))
        ax.set_xlabel("Δ (Hap N $-$ Ref N)", fontsize=label_size)
        ax.set_title(arm_titles[arm], fontsize=title_size)
    axes[0].set_ylabel("lflank extent", fontsize=label_size)

    legend_handles = [
        Patch(facecolor=color_pass,      edgecolor="#bbbbbb", label="pass"),
        Patch(facecolor=color_tied,      edgecolor="#bbbbbb", label="tied"),
        Patch(facecolor=color_missed,    edgecolor="#bbbbbb", label="missed"),
        Patch(facecolor=color_outscored, edgecolor="#bbbbbb", label="outscored"),
        _SplitCellHandle(lower_color=color_outscored, upper_color=color_pass),
    ]
    legend_labels = [
        "pass", "tied", "missed", "outscored",
        "forward \\ reverse",
    ]
    fig.subplots_adjust(right=0.74)
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map={_SplitCellHandle: _make_split_cell_handler()},
        loc="center left",
        bbox_to_anchor=(0.76, 0.5),
        frameon=True,
        fontsize=fontsize,
        handlelength=1.6,
        handleheight=1.6,
        handletextpad=0.8,
        labelspacing=1.0,
        borderpad=0.8,
    )
    return fig
