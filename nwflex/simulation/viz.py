"""
viz.py — visualizations for the simulation harness.

- :func:`render_zoom` produces a column-aligned ASCII view of one
  alignment around a repeat-zone interval.
- :func:`plot_correctness_heatmap` produces the three-panel
  (Δ × lflank) correct/wrong heatmap used by the validation cells.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from .core import parse_cigar, reverse_complement


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


class _CircleHandle:
    """Sentinel for a circle-only legend entry."""

    def __init__(self, color: str):
        self.color = color


def _make_circle_handler():
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.patches import Circle as _Circle

    class _CircleHandler(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            r = 0.42 * min(width, height)
            return [_Circle(
                (-xdescent + width / 2, -ydescent + height / 2), r,
                facecolor=orig_handle.color, edgecolor="#222222",
                linewidth=0.6, transform=trans,
            )]

    return _CircleHandler()


def plot_layout_schematic(
    *,
    motif: str,
    ref_n: int,
    delta_example: int = 2,
    snv: Mapping | None = None,
    read_lflank_example: float | None = None,
    read_rflank_example: float | None = None,
    fontsize: int = 14,
    figsize: tuple = (13.0, 4.8),
    suptitle: str | None = None,
    subtitle: str | None = None,
    mirror: bool = False,
    flanks_aligned: bool = True,
    ax=None,
):
    """
    Standalone explainer figure for the simulation geometry: a
    Reference row on top, a Haplotype row carrying ``Δ`` extra motif
    copies (or missing copies when ``delta_example < 0``), and a Read
    row whose left overhang illustrates ``lflank extent``.

    Parameters
    ----------
    motif, ref_n
        Locus motif (e.g. ``"AAC"``) and reference repeat count.
    delta_example
        Number of extra (positive) or missing (negative) motif copies
        on the haplotype, used purely to illustrate the X axis.
    snv
        Optional ``{"offset_from_boundary": int}`` to render a small SNV
        marker on the haplotype's left flank.
    read_lflank_example, read_rflank_example
        Read overhang widths (axis units) shown in the READ row.  Pass
        ``None`` to use sensible defaults.
    fontsize, figsize
        Standard matplotlib knobs.
    suptitle, subtitle
        Optional headings drawn above the schematic.
    """
    import matplotlib.pyplot as plt

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    # Force a white figure background and white axes face so the
    # rendering looks the same in light and dark IDE themes — this
    # matters most for ``fig.suptitle`` text, which sits on the
    # figure background (not the axes) and otherwise reads black-on-
    # black in dark mode.
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    schematic_dict = {
        "motif": motif,
        "ref_n": int(ref_n),
        "delta_example": int(delta_example),
        "mirror": bool(mirror),
        "flanks_aligned": bool(flanks_aligned),
    }
    if snv is not None:
        schematic_dict["snv"] = dict(snv)
    if read_lflank_example is not None:
        schematic_dict["read_lflank_example"] = float(read_lflank_example)
    if read_rflank_example is not None:
        schematic_dict["read_rflank_example"] = float(read_rflank_example)

    _draw_schematic(ax, schematic=schematic_dict, fontsize=fontsize)

    title_size = fontsize + 3
    label_size = fontsize + 1
    if subtitle is None:
        bp = int(ref_n) * len(str(motif))
        sub_motif = reverse_complement(motif) if mirror else motif
        subtitle = (
            f"Repeat zone: $Z = (\\mathrm{{{sub_motif}}})^{{{ref_n}}}$ "
            f"= ${bp}$ bp"
        )

    if standalone:
        if suptitle is not None:
            fig.suptitle(suptitle, fontsize=title_size, fontweight="bold",
                         y=0.98, color="#222222")
        if subtitle:
            sub_y = 0.90 if suptitle is not None else 0.95
            fig.text(0.5, sub_y, subtitle, ha="center", va="top",
                     fontsize=label_size, style="italic", color="#444444")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88 if suptitle else 0.96))
    else:
        # Subplot mode: render the suptitle as the axes title, and the
        # subtitle as a small italic line just below it.  The caller
        # owns any figure-level suptitle.
        if suptitle is not None:
            ax.set_title(suptitle, fontsize=title_size, fontweight="bold",
                         color="#222222", pad=label_size + 6)
        if subtitle:
            ax.text(0.5, 1.0, subtitle, transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=label_size - 1, style="italic", color="#444444")
    return fig


def _draw_schematic(ax, *, schematic: Mapping, fontsize: int) -> None:
    """
    Draw the explainer banner: a Reference row on top, a Haplotype row
    in the middle showing ``Δ`` extra/missing motif copies, and a
    Read stack at the bottom illustrating the read-tiling at varying
    lflank/rflank balances.

    The Reference and Haplotype rows share flank x-positions: when
    ``delta > 0`` the haplotype's motif tiles are uniformly compressed
    so all hap_n copies fit the reference's repeat-zone width; when
    ``delta < 0`` ghost outlines fill the unoccupied portion.  Each
    read in the stack carries the same motif tiles (showing the repeat
    inside the read) plus its lflank / rflank slices.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    motif: str = schematic["motif"]
    ref_n: int = int(schematic["ref_n"])
    delta_ex: int = int(schematic.get("delta_example", 2))
    snv = schematic.get("snv")
    mirror: bool = bool(schematic.get("mirror", False))
    display_motif: str = reverse_complement(motif) if mirror else motif

    hap_n = ref_n + delta_ex
    if hap_n < 0:
        hap_n = 0

    flank_w = 10.0
    tile_w  = 1.4
    bar_h   = 0.7
    label_x = -0.8

    flanks_aligned = bool(schematic.get("flanks_aligned", True))
    ref_repeat_w = ref_n * tile_w
    if flanks_aligned:
        # Both rows span the same total width; haplotype tiles compress
        # when delta > 0, ghost outlines fill the gap when delta < 0.
        if hap_n > 0 and delta_ex >= 0:
            hap_tile_w = ref_repeat_w / hap_n
        else:
            hap_tile_w = tile_w
        hap_repeat_w = ref_repeat_w
    else:
        # Repeat zone scales with copy count; haplotype's right flank
        # shifts to make room (or in for delta < 0).
        hap_tile_w = tile_w
        hap_repeat_w = hap_n * tile_w

    x_max = flank_w * 2 + max(ref_repeat_w, hap_repeat_w) + 0.6
    # Mirror mode flips the figure horizontally; row labels would end
    # up on the visual right.  Counter that by anchoring them to the
    # right side of the data range so they land on the visual left
    # after ``ax.invert_xaxis()``.
    row_label_x = (x_max + 0.8) if mirror else label_x

    # Read stack: three reads with varying lflank / rflank balance.
    # Indices: 0 = bottom (heavy rflank), 2 = top (heavy lflank).
    read_lflanks = [flank_w * 0.20, flank_w * 0.45, flank_w * 0.70]
    read_rflanks = [flank_w * 0.70, flank_w * 0.45, flank_w * 0.20]
    n_reads = len(read_lflanks)
    read_gap = 0.30
    y_read_bot = 0.0
    read_ys = [y_read_bot + k * (bar_h + read_gap) for k in range(n_reads)]

    big_gap = 1.6
    y_hap = read_ys[-1] + bar_h + big_gap
    y_ref = y_hap + bar_h + big_gap

    flank_color  = "#e8e8e8"
    motif_color  = "#a6cee3"
    extra_color  = "#fdd49e"
    missing_edge = "#c0392b"
    read_outline = "#08519c"
    snv_color    = "#2ca02c"

    def _row_label(y, text):
        ax.text(row_label_x, y + bar_h / 2, text, ha="right", va="center",
                fontsize=fontsize + 2, fontweight="bold", color="#333333")

    def _flank_box(x, y, w, label):
        ax.add_patch(Rectangle((x, y), w, bar_h, facecolor=flank_color,
                               edgecolor="#888888", linewidth=0.7))
        ax.text(x + w / 2, y + bar_h / 2, label, ha="center", va="center",
                fontsize=fontsize - 3, color="#666666", style="italic")

    def _flank_slice(x, y, w):
        ax.add_patch(Rectangle((x, y), w, bar_h, facecolor=flank_color,
                               edgecolor="#888888", linewidth=0.5))

    def _motif_tile(x, y, color, w=tile_w, with_text=True):
        ax.add_patch(Rectangle((x, y), w, bar_h, facecolor=color,
                               edgecolor="#444444", linewidth=0.6))
        if with_text:
            ax.text(x + w / 2, y + bar_h / 2, display_motif,
                    ha="center", va="center",
                    fontsize=fontsize - 4, family="monospace", color="#222222")

    # --- Reference row ---------------------------------------------------
    _row_label(y_ref, "Reference")
    _flank_box(0, y_ref, flank_w, "left flank")
    for i in range(ref_n):
        _motif_tile(flank_w + i * tile_w, y_ref, motif_color, tile_w)
    _flank_box(flank_w + ref_repeat_w, y_ref, flank_w, "right flank")
    # A / Z / B labels above the Reference, matching library colors:
    # A=#2060a0 (blue), Z=#c06020 (orange), B=#7030a0 (purple).
    label_y = y_ref + bar_h + 0.20
    ax.text(flank_w / 2, label_y, "$A$",
            ha="center", va="bottom",
            fontsize=fontsize + 1, fontweight="bold", color="#2060a0")
    ax.text(flank_w + ref_repeat_w / 2, label_y,
            f"$Z = (\\mathrm{{{display_motif}}})^{{{ref_n}}}$",
            ha="center", va="bottom",
            fontsize=fontsize, color="#c06020")
    ax.text(flank_w + ref_repeat_w + flank_w / 2, label_y, "$B$",
            ha="center", va="bottom",
            fontsize=fontsize + 1, fontweight="bold", color="#7030a0")

    # --- Haplotype row ---------------------------------------------------
    _row_label(y_hap, "Haplotype")
    # Continuous haplotype outline so the shape stays whole even when a
    # delta < 0 leaves an empty gap between the motifs and the right flank.
    ax.add_patch(Rectangle((0, y_hap), 2 * flank_w + hap_repeat_w, bar_h,
                           facecolor="none", edgecolor="#888888",
                           linewidth=0.7, zorder=2))
    _flank_box(0, y_hap, flank_w, "left flank")
    shared = min(ref_n, hap_n)
    for i in range(shared):
        _motif_tile(flank_w + i * hap_tile_w, y_hap, motif_color, hap_tile_w)
    if delta_ex > 0:
        for i in range(ref_n, hap_n):
            _motif_tile(flank_w + i * hap_tile_w, y_hap, extra_color, hap_tile_w)
    # delta_ex < 0: the haplotype simply has fewer motif copies; we do
    # not draw any ghost outline or letter for the missing positions.
    _flank_box(flank_w + hap_repeat_w, y_hap, flank_w, "right flank")

    # Δ bracket above the haplotype.
    if delta_ex != 0:
        if delta_ex > 0:
            delta_x_a = flank_w + ref_n * hap_tile_w
            delta_x_b = flank_w + hap_n * hap_tile_w
        else:
            delta_x_a = flank_w + hap_n * tile_w
            delta_x_b = flank_w + ref_n * tile_w
        ax.annotate(
            "", xy=(delta_x_a, y_hap + bar_h + 0.10),
            xytext=(delta_x_b, y_hap + bar_h + 0.10),
            arrowprops=dict(arrowstyle="<->", color=missing_edge, lw=1.4),
        )
        ax.text((delta_x_a + delta_x_b) / 2, y_hap + bar_h + 0.30,
                f"$\\Delta = {delta_ex:+d}$ motif copies",
                ha="center", va="bottom",
                fontsize=fontsize - 1, color=missing_edge, fontweight="bold")

    # SNV (optional) — a green base-change bar in the haplotype's left
    # flank, mirrored into every read whose lflank extends over it, with
    # an arrow from the SNV label down onto the bar.
    if snv is not None:
        off = int(snv.get("offset_from_boundary", 2))
        snv_bar_w = 0.18
        # Position the bar so the grey gap to the repeat boundary equals
        # the bar's own width.
        snv_bar_x = flank_w - snv_bar_w - snv_bar_w
        snv_x = snv_bar_x + snv_bar_w / 2

        # Bar on the haplotype.
        ax.add_patch(Rectangle((snv_bar_x, y_hap), snv_bar_w, bar_h,
                               facecolor=snv_color, edgecolor="none",
                               zorder=5))
        # Bar on every read whose left edge is to the left of the SNV.
        for k in range(n_reads):
            rx0_k = flank_w - read_lflanks[k]
            if rx0_k < snv_x:
                ax.add_patch(Rectangle((snv_bar_x, read_ys[k]),
                                       snv_bar_w, bar_h,
                                       facecolor=snv_color, edgecolor="none",
                                       zorder=5))

        # Dimension-line bracket showing the SNV's distance from the
        # repeat boundary: a thin horizontal segment with tick marks at
        # each end, spanning from the bar's right edge to the boundary.
        snv_bar_right = snv_bar_x + snv_bar_w
        bracket_y = y_hap + bar_h + 0.15
        tick_h    = 0.10
        ax.plot([snv_bar_right, flank_w], [bracket_y, bracket_y],
                color=snv_color, lw=1.2, zorder=4)
        ax.plot([snv_bar_right, snv_bar_right],
                [bracket_y - tick_h / 2, bracket_y + tick_h / 2],
                color=snv_color, lw=1.2, zorder=4)
        ax.plot([flank_w, flank_w],
                [bracket_y - tick_h / 2, bracket_y + tick_h / 2],
                color=snv_color, lw=1.2, zorder=4)
        bracket_center = (snv_bar_right + flank_w) / 2
        ax.text(bracket_center, bracket_y + 0.12, f"SNV: {off} bp",
                ha="center", va="bottom",
                fontsize=fontsize - 1, color=snv_color, fontweight="bold")

    # --- Read stack ------------------------------------------------------
    stack_center = (read_ys[0] + read_ys[-1]) / 2
    _row_label(stack_center, "Reads")

    for k in range(n_reads):
        yk = read_ys[k]
        lflank_ex = read_lflanks[k]
        rflank_ex = read_rflanks[k]
        rx0 = flank_w - lflank_ex
        # Left flank slice.
        _flank_slice(rx0, yk, lflank_ex)
        # Motif tiles inside the read (no text — already labeled on Hap above).
        for i in range(hap_n):
            color = motif_color if i < ref_n else extra_color
            _motif_tile(flank_w + i * hap_tile_w, yk, color, hap_tile_w,
                        with_text=False)
        # Right flank slice.
        _flank_slice(flank_w + hap_repeat_w, yk, rflank_ex)
        # Outline around the whole read.
        rx1 = flank_w + hap_repeat_w + rflank_ex
        ax.add_patch(FancyBboxPatch(
            (rx0, yk), rx1 - rx0, bar_h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            facecolor="none", edgecolor=read_outline, linewidth=1.1,
            zorder=3,
        ))

    # lflank extent bracket on the topmost read (heaviest lflank).
    top_y       = read_ys[-1]
    top_lflank  = read_lflanks[-1]
    arrow_y     = top_y + bar_h + 0.15  # arrow nudged up ~2 px
    text_y      = top_y + bar_h + 0.30  # text stays where it was
    ax.annotate(
        "", xy=(flank_w - top_lflank, arrow_y),
        xytext=(flank_w, arrow_y),
        arrowprops=dict(arrowstyle="<->", color=read_outline, lw=1.4),
    )
    ax.text(flank_w - top_lflank / 2, text_y,
            "lflank extent",
            ha="center", va="bottom",
            fontsize=fontsize - 1, color=read_outline, fontweight="bold")

    # Frame.  In mirror mode, extend xlim on the right so the row
    # labels (anchored past x_max) land at the visual left after invert.
    if mirror:
        ax.set_xlim(label_x - 4.5, x_max + 5.0)
        ax.invert_xaxis()
    else:
        ax.set_xlim(label_x - 4.5, x_max)
    ax.set_ylim(-0.4, y_ref + bar_h + 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_correctness_heatmap(
    df,
    *,
    deltas: Iterable[int],
    lflanks: Iterable[int],
    arm_titles: Mapping[str, str],
    color_recovered: str = "#08519c",    # dark blue
    color_co_optimal: str = "#6baed6",   # light blue
    color_dominant: str = "#ffffff",     # white
    color_dominated: str = "#c0392b",    # red
    color_nan: str = "#cccccc",
    fontsize: int = 14,
    suptitle: str | None = None,
    subtitle: str | None = None,
):
    """
    Three-panel four-state heatmap for a (Δ × lflank × arm) sweep.

    Each cell carries two always-on symbols:

    - the **box** (filling the cell) is colored by the forward-strand
      verdict;
    - the **circle** at the cell's center is colored by the
      reverse-complement strand's verdict.

    For single-strand arms (e.g. NW-flex), set ``fwd_state == rc_state``
    and the box and circle render in the same color.

    State colors are labeled by a two-symbol code in the legend.  The
    first symbol is the alignment outcome (``✓`` correct, ``✗`` wrong);
    the second symbol is the chosen alignment's NW score relative to
    truth's NW score (``=`` tied, ``<`` chosen lower than truth, ``>``
    chosen higher than truth):

    - ``"P"`` ``✓ =`` → dark blue   (alignment correct; score equals truth)
    - ``"T"`` ``✗ =`` → light blue  (alignment wrong; score equals truth)
    - ``"M"`` ``✗ <`` → white       (alignment wrong; chosen scores below
      truth — the aligner's heuristic settled for less)
    - ``"D"`` ``✗ >`` → red         (alignment wrong; chosen scores above
      truth — the scoring landscape rejects truth)
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
    from matplotlib.patches import Circle, Patch, Rectangle

    deltas = list(deltas)
    lflanks = list(lflanks)
    arms = list(arm_titles.keys())
    n = len(arms)

    label_size = fontsize + 1
    title_size = fontsize + 3

    state_to_color = {
        "P": color_recovered,
        "T": color_co_optimal,
        "M": color_dominant,
        "D": color_dominated,
    }

    def color_of(state):
        if isinstance(state, str) and state in state_to_color:
            return state_to_color[state]
        return color_nan

    has_per_strand = "fwd_state" in df.columns and "rc_state" in df.columns

    fig, axes = plt.subplots(
        1, n, figsize=(5.0 * n + 2.5, 6.0), sharey=True,
        gridspec_kw={"wspace": 0.06},
        subplot_kw={"facecolor": "white"},
    )
    fig.patch.set_facecolor("white")
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
                # Box (always) = forward strand verdict.
                ax.add_patch(Rectangle(
                    (D - 0.5, L - 0.5), 1, 1,
                    facecolor=fwd_c, edgecolor="none", linewidth=0,
                ))
                # Circle (always) = reverse-complement strand verdict.
                ax.add_patch(Circle(
                    (D, L), 0.26,
                    facecolor=rc_c, edgecolor="#222222", linewidth=0.6,
                    zorder=4,
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
        # Highlight the Δ=0 column (haplotype == reference) when present.
        if 0 in deltas:
            ax.add_patch(Rectangle(
                (-0.5, lflanks[0] - 0.5),
                1, lflanks[-1] - lflanks[0] + 1,
                fill=False, edgecolor="black", linewidth=1.5, zorder=5,
            ))
        ax.set_xlabel("Δ (Hap N $-$ Ref N)", fontsize=label_size, color="#222222")
        ax.set_title(arm_titles[arm], fontsize=title_size, color="#222222")
    axes[0].set_ylabel("lflank extent", fontsize=label_size, color="#222222")

    shape_color = "#999999"
    blank = Patch(facecolor="none", edgecolor="none")
    # Single legend laid out as a 4-row × 2-col grid.  Matplotlib fills
    # column-by-column (col 1 top→bottom, then col 2), so entries are
    # ordered: 4 verdict-color rows (col 1), then 2 strand-shape rows
    # plus 2 blanks (col 2).
    interleaved_handles = [
        Patch(facecolor=color_recovered,  edgecolor="#bbbbbb"),
        Patch(facecolor=color_co_optimal, edgecolor="#bbbbbb"),
        Patch(facecolor=color_dominant,   edgecolor="#bbbbbb"),
        Patch(facecolor=color_dominated,  edgecolor="#bbbbbb"),
        Patch(facecolor=shape_color, edgecolor="#222222", linewidth=0.6),
        _CircleHandle(color=shape_color),
        blank,
        blank,
    ]
    interleaved_labels = [
        "✓ align, score = truth",
        "✗ align, score = truth",
        "✗ align, score < truth",
        "✗ align, score > truth",
        "forward",
        "reverse",
        "",
        "",
    ]

    panel_top = 0.86 if suptitle else 0.92
    fig.subplots_adjust(right=0.70, top=panel_top, bottom=0.10)

    legend_y = 0.5 * panel_top + 0.05
    fig.legend(
        handles=interleaved_handles,
        labels=interleaved_labels,
        handler_map={_CircleHandle: _make_circle_handler()},
        ncol=2,
        loc="center left",
        bbox_to_anchor=(0.72, legend_y),
        frameon=True,
        fontsize=fontsize,
        handlelength=1.6,
        handleheight=1.6,
        handletextpad=0.8,
        columnspacing=1.6,
        labelspacing=1.0,
        borderpad=0.8,
        facecolor="white",
        edgecolor="#444444",
        labelcolor="#222222",
    )

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=title_size + 1, y=0.97,
                     fontweight="bold", color="#222222")
    if subtitle is not None:
        sub_y = 0.93 if suptitle is not None else 0.95
        # Center on the panel area (not the whole figure, which includes legend).
        cx = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
        fig.text(cx, sub_y, subtitle, ha="center", va="top",
                 fontsize=label_size, style="italic", color="#444444")
    return fig
