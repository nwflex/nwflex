"""
Visualizations for the simulation harness.

- :func:`render_zoom` produces a column-aligned ASCII view of one
  alignment around a repeat-zone interval.
- :func:`project_alignment_to_ref` projects a CIGAR onto reference
  coordinates as a per-position read-base string.
- :func:`plot_correctness_heatmap` and :func:`plot_correctness_heatmap_rows`
  produce the (Δ × lflank) correct/wrong heatmaps used in the
  performance comparison.
"""

from __future__ import annotations

from functools import reduce
from typing import Iterable, Mapping, Sequence, Tuple

from .core import combine_states, parse_cigar, reverse_complement


# Internal sentinels for ``project_alignment_to_ref`` so the resulting
# string can round-trip into integer codes without colliding with any
# biological alphabet character.
_BG_SENTINEL = "\x01"
_GAP_SENTINEL = "\x02"


def project_alignment_to_ref(
    ref_len: int,
    pos_1based: int,
    cigar: str,
    read_seq: str,
    *,
    gap_char: str = "-",
    bg_char: str = " ",
) -> str:
    """
    Project an alignment onto reference coordinates.

    Walks ``cigar`` from ``pos_1based`` (1-based, as in SAM ``POS``) and
    writes the read base placed at each reference position.  The output
    is a string of length ``ref_len``:

    - ``M``/``=``/``X`` — the consumed read base at that ref position.
    - ``D``/``N`` — ``gap_char`` (ref position covered, no read base).
    - outside the alignment span, or covered by soft-clipped (``S``) or
      hard-clipped (``H``) ops — ``bg_char``.

    Insertion (``I``) and soft/hard clip ops consume read bases (or
    nothing) but no reference position, so they do not appear in the
    projected row.  This is the same projection convention as
    :func:`nwflex.aligners.get_aligned_bases`, just driven by
    (CIGAR, position) rather than expanded alignment strings, so
    pileups can mix BWA-MEM and NW-flex alignments uniformly.

    Parameters
    ----------
    ref_len : int
        Length of the forward reference the alignment is against.
    pos_1based : int
        1-based reference position of the first aligned read base.
    cigar : str
        CIGAR string for the alignment.
    read_seq : str
        Read DNA sequence (forward orientation relative to ``cigar``).
    gap_char, bg_char : str
        Single-character substitutes for deleted positions and
        outside-span positions, respectively.

    Returns
    -------
    str
        Length-``ref_len`` projection.
    """
    row = [bg_char] * ref_len
    ref_pos = pos_1based - 1
    read_pos = 0
    for length, op in parse_cigar(cigar):
        if op in ("M", "=", "X"):
            for _ in range(length):
                if 0 <= ref_pos < ref_len:
                    row[ref_pos] = read_seq[read_pos]
                ref_pos += 1
                read_pos += 1
        elif op in ("D", "N"):
            for _ in range(length):
                if 0 <= ref_pos < ref_len:
                    row[ref_pos] = gap_char
                ref_pos += 1
        elif op == "I":
            read_pos += length
        elif op == "S":
            read_pos += length
        elif op == "H":
            pass
        else:
            raise ValueError(f"unsupported CIGAR op: {op!r}")
    return "".join(row)



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
    # side of the zone.  When the alignment touches no Z column,
    # show everything.
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
    delta_range: Tuple[int, int] | None = None,
    snv_offset_range: Tuple[int, int] | None = None,
    lflank_range: Tuple[int, int] | None = None,
    fontsize: int = 14,
    figsize: tuple = (13.0, 4.8),
    suptitle: str | None = None,
    suptitle_loc: str = "center",
    subtitle: str | None = None,
    mirror: bool = False,
    flanks_aligned: bool = True,
    show_nwflex: bool = False,
    nwflex_factor: int = 3,
    ax=None,
):
    """
    Standalone explainer figure for the simulation geometry: a
    Reference row on top, a Haplotype row carrying ``Δ`` extra motif
    copies (or missing copies when ``delta_example < 0``), and a Read
    row whose left overhang illustrates ``lflank extent``.  When
    ``show_nwflex`` is True an extra row sits above Reference showing
    the NW-flex extended reference (``nwflex_factor * ref_n`` motif
    copies); that row is always rendered flanks-aligned so the extra
    copies do not blow out the figure width.

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
        "show_nwflex": bool(show_nwflex),
        "nwflex_factor": int(nwflex_factor),
    }
    if snv is not None:
        schematic_dict["snv"] = dict(snv)
    if read_lflank_example is not None:
        schematic_dict["read_lflank_example"] = float(read_lflank_example)
    if read_rflank_example is not None:
        schematic_dict["read_rflank_example"] = float(read_rflank_example)
    if delta_range is not None:
        schematic_dict["delta_range"] = tuple(int(v) for v in delta_range)
    if snv_offset_range is not None:
        schematic_dict["snv_offset_range"] = tuple(int(v) for v in snv_offset_range)
    if lflank_range is not None:
        schematic_dict["lflank_range"] = tuple(int(v) for v in lflank_range)

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
            ax.set_title(suptitle, loc=suptitle_loc,
                         fontsize=title_size, fontweight="bold",
                         color="#222222", pad=label_size + 6)
        if subtitle:
            _loc_x = {"left": 0.0, "center": 0.5, "right": 1.0}.get(
                suptitle_loc, 0.5)
            _loc_ha = {"left": "left", "center": "center",
                       "right": "right"}.get(suptitle_loc, "center")
            ax.text(_loc_x, 1.0, subtitle, transform=ax.transAxes,
                    ha=_loc_ha, va="bottom",
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
    show_nwflex = bool(schematic.get("show_nwflex", False))
    nwflex_factor = int(schematic.get("nwflex_factor", 3))
    nwflex_n = nwflex_factor * ref_n
    # NW-flex tiles always fit the same Z-zone width as the Reference,
    # regardless of the global flanks_aligned setting — otherwise the
    # 3N tiles would blow the figure width.
    nwflex_tile_w = ref_repeat_w / nwflex_n if nwflex_n > 0 else tile_w

    # Vertical stack (top to bottom): Reference, NW-flex ref (optional),
    # Haplotype, Reads.  Compute from bottom up.
    y_hap = read_ys[-1] + bar_h + big_gap
    if show_nwflex:
        y_nwflex = y_hap + bar_h + big_gap
        y_ref = y_nwflex + bar_h + big_gap
    else:
        y_nwflex = None
        y_ref = y_hap + bar_h + big_gap

    flank_color  = "#e8e8e8"
    motif_color  = "#a6cee3"
    nwflex_color = "#dceaf3"   # paler shade of motif_color; hints at "flexible"
    nwflex_edge  = "#5a9bc0"   # dashed-edge color for NW-flex tiles
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

    # --- NW-flex extended-reference row (below Reference) ----------------
    if show_nwflex:
        _row_label(y_nwflex, "NW-flex ref")
        _flank_box(0, y_nwflex, flank_w, "left flank")
        # Tiles render in a paler shade with dashed edges, hinting at the
        # EP-skip "extra edges": each tile boundary is a place the
        # aligner can step across without paying a per-base penalty.
        for i in range(nwflex_n):
            ax.add_patch(Rectangle(
                (flank_w + i * nwflex_tile_w, y_nwflex),
                nwflex_tile_w, bar_h,
                facecolor=nwflex_color, edgecolor=nwflex_edge,
                linewidth=0.8, linestyle=(0, (2.5, 1.5)),
            ))
        _flank_box(flank_w + ref_repeat_w, y_nwflex, flank_w, "right flank")
        # Italic Z' label above the row, matching the A/Z/B style on the
        # Reference row but simpler (one centered annotation).
        ax.text(flank_w + ref_repeat_w / 2, y_nwflex + bar_h + 0.20,
                f"$Z' = (\\mathrm{{{display_motif}}})^{{{nwflex_n}}}$",
                ha="center", va="bottom",
                fontsize=fontsize, color="#c06020")

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
        delta_range = schematic.get("delta_range")
        if delta_range is not None:
            lo, hi = delta_range
            delta_label = f"$\\Delta \\in [{lo:+d}, {hi:+d}]$ motif copies"
        else:
            delta_label = f"$\\Delta = {delta_ex:+d}$ motif copies"
        ax.text((delta_x_a + delta_x_b) / 2, y_hap + bar_h + 0.30,
                delta_label,
                ha="center", va="bottom",
                fontsize=fontsize - 1, color=missing_edge, fontweight="bold")

    # SNV (optional) — a green base-change bar in the haplotype's left
    # flank, mirrored into every read whose lflank extends over it, with
    # an arrow from the SNV label down onto the bar.
    if snv is not None:
        off = int(snv.get("offset_from_boundary", 2))
        snv_bar_w = 0.18
        # Place the bar `off` bar-widths into the left flank from the
        # repeat boundary so the visual distance scales with the SNV
        # offset parameter.
        snv_bar_x = flank_w - off * snv_bar_w - snv_bar_w
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
        snv_off_range = schematic.get("snv_offset_range")
        if snv_off_range is not None:
            lo, hi = snv_off_range
            snv_label = f"SNV: $\\in [{lo}, {hi}]$ bp"
        else:
            snv_label = f"SNV: {off} bp"
        ax.text(bracket_center, bracket_y + 0.12, snv_label,
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
    lflank_rng = schematic.get("lflank_range")
    if lflank_rng is not None:
        lo, hi = lflank_rng
        lflank_label = f"lflank extent $\\in [{lo}, {hi}]$"
    else:
        lflank_label = "lflank extent"
    ax.text(flank_w - top_lflank / 2, text_y,
            lflank_label,
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


def _format_grid_axes(ax, *, xs, ys, fontsize, highlight_zero_x=True,
                      highlight_zero_y=False):
    """Apply the shared (delta × <y-axis>) grid styling to ``ax``.

    Major ticks at ``xs``/``ys``, minor ticks at the cell boundaries,
    light grey grid lines, square aspect, dark spines.  Optionally
    overlays a 1.5 pt black rectangle highlighting the column at x=0
    (and/or the row at y=0) — the Δ=0 "no-change" reference.
    """
    import numpy as np
    from matplotlib.patches import Rectangle

    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    ax.set_ylim(ys[0] - 0.5, ys[-1] + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(xs)
    ax.set_yticks(ys)
    ax.set_xticks(np.array(xs, dtype=float) - 0.5, minor=True)
    ax.set_yticks(np.array(ys, dtype=float) - 0.5, minor=True)
    ax.grid(which="minor", color="#bbbbbb", linewidth=0.5)
    ax.tick_params(which="major", labelsize=fontsize, colors="#222222")
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    if highlight_zero_x and 0 in xs:
        ax.add_patch(Rectangle(
            (-0.5, ys[0] - 0.5),
            1, ys[-1] - ys[0] + 1,
            fill=False, edgecolor="black", linewidth=1.5, zorder=5,
        ))
    if highlight_zero_y and 0 in ys:
        ax.add_patch(Rectangle(
            (xs[0] - 0.5, -0.5),
            xs[-1] - xs[0] + 1, 1,
            fill=False, edgecolor="black", linewidth=1.5, zorder=5,
        ))


def _draw_glyphs(ax, *, xs, ys, fwd_at, rc_at, color_of):
    """Lay down fwd Rectangle + rc Circle at each (x, y) cell.

    ``fwd_at(xi, yi)`` and ``rc_at(xi, yi)`` return the per-strand
    value at index ``(xi, yi)`` into ``xs``/``ys``; ``color_of(value)``
    maps the value to a fill color.
    """
    from matplotlib.patches import Circle, Rectangle

    for yi, Y in enumerate(ys):
        for xi, X in enumerate(xs):
            ax.add_patch(Rectangle(
                (X - 0.5, Y - 0.5), 1, 1,
                facecolor=color_of(fwd_at(xi, yi)),
                edgecolor="none", linewidth=0,
            ))
            ax.add_patch(Circle(
                (X, Y), 0.26,
                facecolor=color_of(rc_at(xi, yi)),
                edgecolor="#222222", linewidth=0.6,
                zorder=4,
            ))


def _draw_1d_grid_panel(ax, sub_df, *, deltas, lflanks,
                        cell_value_fn, color_fn, fontsize):
    """Draw one ``(Δ × lflank)`` panel using a pluggable cell-value
    function — the 1-D analogue of :func:`_draw_2d_grid_panel`.

    Iteration order matches the 2-D version (outer X, inner Y), so a
    cross-cell aggregator passed in via ``cell_value_fn`` sees the same
    ordering as the panel rendering.
    """
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

    _format_grid_axes(ax, xs=deltas, ys=lflanks, fontsize=fontsize)


def _draw_state_panel(ax, sub_df, *, deltas, lflanks, color_of, fontsize):
    """Draw one (delta × lflank) panel into ``ax``.

    Renders a fwd Rectangle + rc Circle per cell, sets ticks/grid, and
    highlights the Δ=0 column.  Does not set the panel title or axis
    labels — the caller owns those.

    ``sub_df`` is the long-form subset for this panel.  It must carry
    either ``fwd_state`` + ``rc_state``, or a single ``state`` column
    (in which case both shapes share that state).
    """
    has_per_strand = "fwd_state" in sub_df.columns and "rc_state" in sub_df.columns
    if has_per_strand:
        fwd_grid = sub_df.pivot(index="lflank", columns="delta", values="fwd_state")
        rc_grid  = sub_df.pivot(index="lflank", columns="delta", values="rc_state")
    else:
        single = sub_df.pivot(index="lflank", columns="delta", values="state")
        fwd_grid = rc_grid = single
    fwd_grid = fwd_grid.reindex(index=lflanks, columns=deltas)
    rc_grid  = rc_grid.reindex(index=lflanks, columns=deltas)

    # Iteration order here (yi outer, xi inner) matches the original
    # _draw_state_panel for pixel-identical patch z-order.
    _draw_glyphs(
        ax, xs=deltas, ys=lflanks,
        fwd_at=lambda xi, yi: fwd_grid.iat[yi, xi],
        rc_at=lambda xi, yi: rc_grid.iat[yi, xi],
        color_of=color_of,
    )
    _format_grid_axes(ax, xs=deltas, ys=lflanks, fontsize=fontsize)


_STATE_COLOR_DEFAULTS = dict(
    color_recovered="#08519c",    # dark blue   — P
    color_co_optimal="#b7d8eb",   # light blue  — T
    color_dominant="#ebb7b7",     # light red   — M
    color_dominated="#c0392b",    # red         — D
    color_nan="#cccccc",          # grey        — missing/NaN
)


def _state_color_fn(*, color_recovered, color_co_optimal,
                    color_dominant, color_dominated, color_nan):
    table = {
        "P": color_recovered,
        "T": color_co_optimal,
        "M": color_dominant,
        "D": color_dominated,
    }
    def color_of(state):
        if isinstance(state, str) and state in table:
            return table[state]
        return color_nan
    return color_of


def plot_correctness_heatmap(
    df,
    *,
    deltas: Iterable[int],
    lflanks: Iterable[int],
    arm_titles: Mapping[str, str],
    color_recovered: str = "#08519c",    # dark blue
    color_co_optimal: str = "#b7d8eb",   # light blue
    color_dominant: str = "#ebb7b7",     # light red
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
    the ground-truth NW score (``=`` tied, ``<`` chosen lower than
    ground truth, ``>`` chosen higher than ground truth):

    - ``"P"`` ``✓ =`` → dark blue   (alignment correct; score equals ground truth)
    - ``"T"`` ``✗ =`` → light blue  (alignment wrong; score equals ground truth)
    - ``"M"`` ``✗ <`` → light red   (alignment wrong; chosen scores below
      ground truth — the aligner's heuristic settled for less)
    - ``"D"`` ``✗ >`` → red         (alignment wrong; chosen scores above
      ground truth — the scoring landscape rejects the ground truth)
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
    import matplotlib.pyplot as plt

    deltas = list(deltas)
    lflanks = list(lflanks)
    arms = list(arm_titles.keys())
    n = len(arms)

    label_size = fontsize + 1
    title_size = fontsize + 3

    color_of = _state_color_fn(
        color_recovered=color_recovered, color_co_optimal=color_co_optimal,
        color_dominant=color_dominant, color_dominated=color_dominated,
        color_nan=color_nan,
    )

    fig, axes = plt.subplots(
        1, n, figsize=(5.0 * n + 2.5, 6.0), sharey=True,
        gridspec_kw={"wspace": 0.06},
        subplot_kw={"facecolor": "white"},
    )
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]

    for ax, arm in zip(axes, arms):
        _draw_state_panel(
            ax, df[df["arm"] == arm],
            deltas=deltas, lflanks=lflanks,
            color_of=color_of, fontsize=fontsize,
        )
        ax.set_xlabel("Δ (Hap N $-$ Ref N)", fontsize=label_size, color="#222222")
        ax.set_title(arm_titles[arm], fontsize=title_size, color="#222222")
    axes[0].set_ylabel("lflank extent", fontsize=label_size, color="#222222")

    # Single legend laid out as a 4-row × 2-col grid.  Matplotlib fills
    # column-by-column (col 1 top→bottom, then col 2), so entries are
    # ordered: 4 verdict-color rows (col 1), then 2 strand-shape rows
    # plus 2 blanks (col 2).
    interleaved_handles, interleaved_labels = _legend_handles_states_strands(
        color_recovered, color_co_optimal, color_dominant, color_dominated,
        with_blanks=True,
    )

    panel_top = 0.86 if suptitle else 0.92
    panel_left, panel_right = 0.09, 0.70
    fig.subplots_adjust(left=panel_left, right=panel_right, top=panel_top, bottom=0.10)

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

    # Center the title block over the panel area, not the whole figure
    # (which includes the legend on the right) — otherwise suptitle and
    # subtitle drift apart and read as off-centered.
    cx = (panel_left + panel_right) / 2
    if suptitle is not None:
        fig.suptitle(suptitle, x=cx, fontsize=title_size + 1, y=0.97,
                     fontweight="bold", color="#222222")
    if subtitle is not None:
        sub_y = 0.91 if suptitle is not None else 0.95
        fig.text(cx, sub_y, subtitle, ha="center", va="top",
                 fontsize=label_size, style="italic", color="#444444")
    return fig


def _rows_top_margin(fig_h: float, has_suptitle: bool):
    """Figure-relative y coords for the title/legend block above panels.

    Translates a fixed absolute-inch top reservation into figure-relative
    coordinates so the forehead doesn't grow with ``n_rows``.  Used by
    every ``_rows`` heatmap variant.

    Layout (in inches from the top of the figure):

    - suptitle baseline at ~0.22 in
    - subtitle baseline at ~0.52 in (with suptitle) / 0.30 in (without)
    - legend bottom at ~1.85 in (legend extends upward from there)
    - panels start at ~2.2 in

    Returns a dict with ``panel_top``, ``panel_bottom``, ``suptitle_y``,
    ``subtitle_y``, ``legend_y`` — all in figure-relative coords.
    """
    top_in    = 2.2
    bottom_in = 0.5
    sup_in    = 0.22
    sub_in    = 0.52 if has_suptitle else 0.30
    leg_in    = 1.85
    return dict(
        panel_top    = 1.0 - top_in    / fig_h,
        panel_bottom = bottom_in       / fig_h,
        suptitle_y   = 1.0 - sup_in    / fig_h,
        subtitle_y   = 1.0 - sub_in    / fig_h,
        legend_y     = 1.0 - leg_in    / fig_h,
    )


def plot_correctness_heatmap_rows(
    rows,
    *,
    deltas: Iterable[int],
    lflanks: Iterable[int],
    arm_titles: Mapping[str, str],
    row_label_fn,
    color_recovered: str = "#08519c",
    color_co_optimal: str = "#b7d8eb",
    color_dominant: str = "#ebb7b7",
    color_dominated: str = "#c0392b",
    color_nan: str = "#cccccc",
    fontsize: int = 14,
    scale: float = 1.0,
    font_scale: float = 1.0,
    suptitle: str | None = None,
    subtitle: str | None = None,
):
    """
    Multi-row four-state heatmap.  Each row is one ``(key, df_subset)``
    pair; columns are the methods in ``arm_titles``.  Useful for sweeping
    an outer parameter (e.g. SNV position) while keeping each row's
    inner (Δ × lflank) panels comparable.

    Mirrors :func:`plot_correctness_heatmap` per panel (fwd Rectangle,
    rc Circle, Δ=0 column highlighted) and shares the same legend.

    Parameters
    ----------
    rows : list of (key, pandas.DataFrame)
        Each entry contributes one row of three panels.  The DataFrame
        must carry ``delta``, ``lflank``, ``arm`` and either
        ``fwd_state`` + ``rc_state`` or a single ``state``.
    deltas, lflanks, arm_titles
        Same as :func:`plot_correctness_heatmap`.
    row_label_fn : callable
        Maps the row's ``key`` to the string displayed as the row's
        y-axis label (e.g., ``lambda k: f"offset = {k} bp"``).
    scale : float
        Multiplier on the figure's physical size in inches.
    font_scale : float
        Multiplier on every font size (tick labels, axis labels, panel
        titles, suptitle, legend).  Independent of ``scale``, so figure
        size and text size can be tuned separately.
    suptitle, subtitle, fontsize, color_*
        Same as :func:`plot_correctness_heatmap`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    deltas = list(deltas)
    lflanks = list(lflanks)
    arms = list(arm_titles.keys())
    rows = list(rows)
    n_rows = len(rows)
    n_cols = len(arms)

    color_of = _state_color_fn(
        color_recovered=color_recovered, color_co_optimal=color_co_optimal,
        color_dominant=color_dominant, color_dominated=color_dominated,
        color_nan=color_nan,
    )

    fig, axes, label_size, title_size, fontsize, m = _build_rows_figure(
        n_rows=n_rows, n_cols=n_cols, fontsize=fontsize,
        scale=scale, font_scale=font_scale,
        suptitle=suptitle, subtitle=subtitle,
    )

    for r, (key, df) in enumerate(rows):
        for c, arm in enumerate(arms):
            ax = axes[r, c]
            _draw_state_panel(
                ax, df[df["arm"] == arm],
                deltas=deltas, lflanks=lflanks,
                color_of=color_of, fontsize=fontsize,
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

    legend_handles, legend_labels = _legend_handles_states_strands(
        color_recovered, color_co_optimal, color_dominant, color_dominated,
    )

    # Panels fill the width; legend sits below the title block, anchored
    # to the figure's left margin.  Two rows × ncol=3 (column-major):
    # col1 = score=ground truth (P,T), col2 = score≠ground truth (M,D),
    # col3 = strand shapes (fwd, rc).
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map={_CircleHandle: _make_circle_handler()},
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0.07, m["legend_y"]),
        frameon=True,
        fontsize=fontsize,
        handlelength=1.9, handleheight=1.9,
        handletextpad=0.6, columnspacing=1.9,
        labelspacing=0.8, borderpad=0.7,
        facecolor="white", edgecolor="#444444", labelcolor="#222222",
    )

    return fig


# ===========================================================================
# 2-D (Δ1 × Δ2) heatmaps — discrete state and continuous proportion
# ===========================================================================
#
# Same fwd-Rectangle / rc-Circle glyph convention as the 1-D heatmaps,
# but the panel axes are now (Δ1, Δ2) instead of (Δ, lflank).  Reads
# within each cell are aggregated into a per-strand summary by a
# pluggable ``cell_value_fn``; that value is then mapped to a color by
# ``color_fn``.  This lets the discrete (P/T/M/D state) and continuous
# (fraction-of-score=ground-truth) heatmaps share the panel-drawing core.
# ---------------------------------------------------------------------------


def _draw_2d_grid_panel(ax, sub_df, *, deltas1, deltas2,
                        cell_value_fn, color_fn, fontsize):
    """Draw one (Δ1 × Δ2) panel into ``ax``.

    ``cell_value_fn(cell_df) -> (fwd_value, rc_value)`` produces the
    per-strand summary for one cell.  ``color_fn(value) -> color``
    maps that summary to a color (handles None / NaN as the NaN color).
    """
    # Precompute per-cell (fwd, rc) values once so the inner iteration
    # below can reuse _draw_glyphs without re-filtering.  Iteration
    # order here (d1 outer, d2 inner) matches the original
    # _draw_2d_grid_panel for pixel-identical patch z-order.
    values = {}
    for d1 in deltas1:
        for d2 in deltas2:
            cell = sub_df[(sub_df["delta1"] == d1) & (sub_df["delta2"] == d2)]
            values[(d1, d2)] = cell_value_fn(cell)

    # _draw_glyphs iterates ys outer, xs inner; we want d1 outer, d2
    # inner to match the legacy z-order, so xs=deltas2 and ys=deltas1
    # is wrong (would flip axes).  Iterate explicitly to preserve both
    # axis assignment AND patch order.
    from matplotlib.patches import Circle, Rectangle
    for d1 in deltas1:
        for d2 in deltas2:
            fwd_v, rc_v = values[(d1, d2)]
            ax.add_patch(Rectangle(
                (d1 - 0.5, d2 - 0.5), 1, 1,
                facecolor=color_fn(fwd_v), edgecolor="none", linewidth=0,
            ))
            ax.add_patch(Circle(
                (d1, d2), 0.26,
                facecolor=color_fn(rc_v),
                edgecolor="#222222", linewidth=0.6, zorder=4,
            ))

    _format_grid_axes(ax, xs=deltas1, ys=deltas2, fontsize=fontsize,
                      highlight_zero_x=True, highlight_zero_y=True)


def _state_value_fn(combine_across_reads):
    """Return a ``cell_value_fn`` that reduces per-cell states under the
    given policy; returns ``(fwd, rc)`` strings (or ``None`` for empty)."""
    def value_fn(cell_df):
        if "fwd_state" in cell_df.columns and "rc_state" in cell_df.columns:
            fwds = [s for s in cell_df["fwd_state"].tolist()
                    if isinstance(s, str)]
            rcs  = [s for s in cell_df["rc_state"].tolist()
                    if isinstance(s, str)]
        else:
            single = [s for s in cell_df["state"].tolist()
                      if isinstance(s, str)]
            fwds = rcs = single
        f = (reduce(lambda a, b: combine_states(a, b, combine_across_reads),
                    fwds) if fwds else None)
        r = (reduce(lambda a, b: combine_states(a, b, combine_across_reads),
                    rcs) if rcs else None)
        return f, r
    return value_fn


def _proportion_value_fn(cell_df):
    """``cell_value_fn`` for fraction-of-(P or T) per strand."""
    fwd = [s for s in cell_df["fwd_state"].tolist() if isinstance(s, str)]
    rc  = [s for s in cell_df["rc_state"].tolist()  if isinstance(s, str)]
    f = sum(1 for s in fwd if s in ("P", "T")) / len(fwd) if fwd else float("nan")
    r = sum(1 for s in rc  if s in ("P", "T")) / len(rc)  if rc  else float("nan")
    return f, r


def _proportion_color_fn(cmap, norm, *, nan_color="#cccccc"):
    """Map a proportion in [0, 1] to ``cmap``'s color; NaN -> grey."""
    def color_of(v):
        if v is None:
            return nan_color
        try:
            if v != v:  # NaN check
                return nan_color
        except TypeError:
            return nan_color
        return cmap(norm(v))
    return color_of


def _PROPORTION_CMAP():
    """LinearSegmentedColormap anchored at the discrete palette's
    D-red (#c0392b) -> yellow (#ffffbf) -> P-blue (#08519c)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "score_eq_truth", ["#c0392b", "#ffffbf", "#08519c"],
    )


def _legend_handles_states_strands(color_recovered, color_co_optimal,
                                   color_dominant, color_dominated,
                                   *, with_blanks=False):
    """Build the standard 6-entry legend (4 states + fwd/rc shapes)
    matching :func:`plot_correctness_heatmap_rows`.

    When ``with_blanks=True`` two empty entries are appended so the
    legend lays out cleanly as a 4-row × 2-col grid (matplotlib fills
    column-by-column).
    """
    from matplotlib.patches import Patch
    shape_color = "#999999"
    handles = [
        Patch(facecolor=color_recovered,  edgecolor="#bbbbbb"),
        Patch(facecolor=color_co_optimal, edgecolor="#bbbbbb"),
        Patch(facecolor=color_dominant,   edgecolor="#bbbbbb"),
        Patch(facecolor=color_dominated,  edgecolor="#bbbbbb"),
        Patch(facecolor=shape_color, edgecolor="#222222", linewidth=0.6),
        _CircleHandle(color=shape_color),
    ]
    labels = [
        "✓ align, score = GT",
        "✗ align, score = GT",
        "✗ align, score < GT",
        "✗ align, score > GT",
        "forward",
        "reverse",
    ]
    if with_blanks:
        blank = Patch(facecolor="none", edgecolor="none")
        handles += [blank, blank]
        labels += ["", ""]
    return handles, labels


def _build_rows_figure(
    *, n_rows, n_cols, fontsize, scale, font_scale, suptitle, subtitle,
):
    """Common multi-row figure shell shared by every ``*_rows`` heatmap.

    Builds a (n_rows, n_cols) subplots grid using the standard
    ``5.0 * n_cols + 0.8`` × ``5.0 * n_rows + 1.6`` inch figsize
    (scaled by ``scale``), reserves the top margin in absolute inches
    via :func:`_rows_top_margin`, applies the standard left/right
    margins, and draws the fig-level suptitle/subtitle.

    Returns ``(fig, axes, label_size, title_size, fontsize_scaled, m)``.
    ``m`` is the margin dict from :func:`_rows_top_margin` so callers
    can place the legend / colorbar at ``m["legend_y"]``.
    """
    import matplotlib.pyplot as plt

    label_size = (fontsize + 1) * font_scale
    title_size = (fontsize + 3) * font_scale
    fontsize_scaled = fontsize * font_scale

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

    m = _rows_top_margin(figsize[1], suptitle is not None)
    fig.subplots_adjust(left=0.07, right=0.98,
                        top=m["panel_top"], bottom=m["panel_bottom"])

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=title_size + 1, y=m["suptitle_y"],
                     fontweight="bold", color="#222222")
    if subtitle is not None:
        fig.text(0.5, m["subtitle_y"], subtitle, ha="center", va="top",
                 fontsize=label_size, style="italic", color="#444444")

    return fig, axes, label_size, title_size, fontsize_scaled, m


def _build_single_row_2d_figure(
    *, n, fontsize, suptitle, subtitle,
    panel_left=0.06, panel_right=0.82,
):
    """Common single-row 2-D figure shell.

    Creates a (1, n) subplots grid sized to the standard 5.8-inch height,
    reserves the top strip in absolute inches for suptitle/subtitle and
    per-panel axis titles, applies the standard margins, and draws the
    fig-level suptitle/subtitle centered over the panel area.

    Returns ``(fig, axes_row, label_size, title_size, panel_top)`` —
    callers add their right-side block (legend or colorbar) and the
    per-panel content.  ``axes_row`` is a flat list of length ``n``.
    """
    import matplotlib.pyplot as plt

    label_size = fontsize + 1
    title_size = fontsize + 3

    fig, axes = plt.subplots(
        1, n, figsize=(5.0 * n + 3.0, 5.8),
        sharey=True,
        gridspec_kw={"wspace": 0.04},
        subplot_kw={"facecolor": "white"},
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    fig_h = 5.8
    top_in    = 1.30
    bottom_in = 0.50
    sup_in    = 0.20
    sub_in    = 0.65 if suptitle is not None else 0.30
    panel_top = 1.0 - top_in / fig_h
    fig.subplots_adjust(left=panel_left, right=panel_right,
                        top=panel_top, bottom=bottom_in / fig_h)

    # Center the title block over the panel area (left..right), not the
    # whole figure, so suptitle and subtitle stay aligned with each other.
    cx = (panel_left + panel_right) / 2
    if suptitle is not None:
        fig.suptitle(suptitle, x=cx, fontsize=title_size + 1,
                     y=1.0 - sup_in / fig_h,
                     fontweight="bold", color="#222222")
    if subtitle is not None:
        fig.text(cx, 1.0 - sub_in / fig_h, subtitle,
                 ha="center", va="top",
                 fontsize=label_size, style="italic", color="#444444")
    axes_row = [axes[0, c] for c in range(n)]
    return fig, axes_row, label_size, title_size, panel_top


def _legend_handles_shapes():
    """Build the 2-entry shape-only legend (fwd Rectangle + rc Circle)
    used by the proportion heatmaps."""
    from matplotlib.patches import Patch
    shape_color = "#999999"
    handles = [
        Patch(facecolor=shape_color, edgecolor="#222222", linewidth=0.6),
        _CircleHandle(color=shape_color),
    ]
    labels = ["forward", "reverse"]
    return handles, labels


def plot_correctness_heatmap_2d(
    df,
    *,
    deltas1: Iterable[int],
    deltas2: Iterable[int],
    arm_titles: Mapping[str, str],
    combine_across_reads: str = "best",
    color_recovered: str = "#08519c",
    color_co_optimal: str = "#b7d8eb",
    color_dominant: str = "#ebb7b7",
    color_dominated: str = "#c0392b",
    color_nan: str = "#cccccc",
    fontsize: int = 14,
    suptitle: str | None = None,
    subtitle: str | None = None,
):
    """Three-panel four-state heatmap for a (Δ1 × Δ2 × arm) sweep.

    Each cell carries the fwd Rectangle / rc Circle convention used by
    :func:`plot_correctness_heatmap`.  Reads in the cell are aggregated
    per strand under ``combine_across_reads`` (``"best"`` or ``"worst"``).
    The legend sits above the panels, matching
    :func:`plot_correctness_heatmap_rows`.

    ``df`` must carry columns ``arm``, ``delta1``, ``delta2``, and
    either ``fwd_state`` + ``rc_state`` or a single ``state``.
    """
    deltas1 = list(deltas1)
    deltas2 = list(deltas2)
    arms = list(arm_titles.keys())
    n = len(arms)

    color_of = _state_color_fn(
        color_recovered=color_recovered, color_co_optimal=color_co_optimal,
        color_dominant=color_dominant, color_dominated=color_dominated,
        color_nan=color_nan,
    )
    value_fn = _state_value_fn(combine_across_reads)

    fig, axes_row, label_size, title_size, _ = _build_single_row_2d_figure(
        n=n, fontsize=fontsize, suptitle=suptitle, subtitle=subtitle,
    )

    for c, arm in enumerate(arms):
        ax = axes_row[c]
        _draw_2d_grid_panel(
            ax, df[df["arm"] == arm],
            deltas1=deltas1, deltas2=deltas2,
            cell_value_fn=value_fn, color_fn=color_of, fontsize=fontsize,
        )
        ax.set_title(arm_titles[arm], fontsize=title_size, color="#222222")
        ax.set_xlabel(r"$\Delta_1$", fontsize=label_size, color="#222222")
    axes_row[0].set_ylabel(r"$\Delta_2$",
                            fontsize=label_size, color="#222222")

    # Single-row layout: legend goes in the right column so the top
    # strip is free for suptitle + subtitle without the legend crowding
    # them.  Top reservation is in absolute inches so the title strip
    # leaves room for both fig-level titles AND the per-panel axis
    # titles (``ax.set_title``) below them without overlap.
    legend_handles, legend_labels = _legend_handles_states_strands(
        color_recovered, color_co_optimal, color_dominant, color_dominated,
    )
    fig.legend(
        handles=legend_handles, labels=legend_labels,
        handler_map={_CircleHandle: _make_circle_handler()},
        ncol=1, loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        frameon=True, fontsize=fontsize,
        handlelength=1.9, handleheight=1.9,
        handletextpad=0.6,
        labelspacing=0.9, borderpad=0.7,
        facecolor="white", edgecolor="#444444", labelcolor="#222222",
    )

    return fig


def plot_correctness_heatmap_2d_rows(
    df,
    *,
    deltas1: Iterable[int],
    deltas2: Iterable[int],
    row_values: Sequence,
    row_col: str,
    arm_titles: Mapping[str, str],
    row_label_fn=None,
    combine_across_reads: str = "best",
    color_recovered: str = "#08519c",
    color_co_optimal: str = "#b7d8eb",
    color_dominant: str = "#ebb7b7",
    color_dominated: str = "#c0392b",
    color_nan: str = "#cccccc",
    fontsize: int = 14,
    scale: float = 1.0,
    suptitle: str | None = None,
    subtitle: str | None = None,
):
    """Multi-row 2-D heatmap.  One row per ``row_values`` entry (e.g.
    per bridge length), filtered from ``df`` on ``row_col``.  Shares the
    fwd-Rectangle / rc-Circle convention and the legend-above layout
    with :func:`plot_correctness_heatmap_2d`."""
    deltas1 = list(deltas1)
    deltas2 = list(deltas2)
    arms = list(arm_titles.keys())
    row_values = list(row_values)
    n_arms = len(arms)
    n_rows = len(row_values)

    color_of = _state_color_fn(
        color_recovered=color_recovered, color_co_optimal=color_co_optimal,
        color_dominant=color_dominant, color_dominated=color_dominated,
        color_nan=color_nan,
    )
    value_fn = _state_value_fn(combine_across_reads)

    fig, axes, label_size, title_size, _, m = _build_rows_figure(
        n_rows=n_rows, n_cols=n_arms, fontsize=fontsize,
        scale=scale, font_scale=1.0,
        suptitle=suptitle, subtitle=subtitle,
    )

    for r, key in enumerate(row_values):
        sub_row = df[df[row_col] == key]
        for c, arm in enumerate(arms):
            ax = axes[r, c]
            _draw_2d_grid_panel(
                ax, sub_row[sub_row["arm"] == arm],
                deltas1=deltas1, deltas2=deltas2,
                cell_value_fn=value_fn, color_fn=color_of,
                fontsize=fontsize,
            )
            if r == 0:
                ax.set_title(arm_titles[arm], fontsize=title_size,
                             color="#222222")
            if r == n_rows - 1:
                ax.set_xlabel(r"$\Delta_1$",
                              fontsize=label_size, color="#222222")
        row_label = (row_label_fn(key) if row_label_fn is not None
                     else f"{row_col} = {key}")
        axes[r, 0].set_ylabel(f"{row_label}\n$\\Delta_2$",
                              fontsize=label_size, color="#222222")

    legend_handles, legend_labels = _legend_handles_states_strands(
        color_recovered, color_co_optimal, color_dominant, color_dominated,
    )
    fig.legend(
        handles=legend_handles, labels=legend_labels,
        handler_map={_CircleHandle: _make_circle_handler()},
        ncol=3, loc="lower left",
        bbox_to_anchor=(0.07, m["legend_y"]),
        frameon=True, fontsize=fontsize,
        handlelength=1.9, handleheight=1.9,
        handletextpad=0.6, columnspacing=1.9,
        labelspacing=0.8, borderpad=0.7,
        facecolor="white", edgecolor="#444444", labelcolor="#222222",
    )

    return fig


def plot_proportion_heatmap_2d(
    df,
    *,
    deltas1: Iterable[int],
    deltas2: Iterable[int],
    arm_titles: Mapping[str, str],
    fontsize: int = 14,
    suptitle: str | None = None,
    subtitle: str | None = None,
    cbar_label: str = "P(score = ground truth)",
):
    """Continuous-color 2-D heatmap of the per-strand fraction of reads
    whose chosen alignment has score equal to ground truth (states P or T).

    Glyph convention matches :func:`plot_correctness_heatmap_2d`:
    Rectangle = fwd-strand proportion, Circle = rc-strand proportion.
    Horizontal colorbar sits above the panels with explicit ticks at
    0.0, 0.25, 0.5, 0.75, 1.0; a small fwd/rc shape legend sits beside.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    deltas1 = list(deltas1)
    deltas2 = list(deltas2)
    arms = list(arm_titles.keys())
    n = len(arms)

    cmap = _PROPORTION_CMAP()
    norm = Normalize(vmin=0.0, vmax=1.0)
    color_of = _proportion_color_fn(cmap, norm)

    fig, axes_row, label_size, title_size, _ = _build_single_row_2d_figure(
        n=n, fontsize=fontsize, suptitle=suptitle, subtitle=subtitle,
    )

    for c, arm in enumerate(arms):
        ax = axes_row[c]
        _draw_2d_grid_panel(
            ax, df[df["arm"] == arm],
            deltas1=deltas1, deltas2=deltas2,
            cell_value_fn=_proportion_value_fn,
            color_fn=color_of, fontsize=fontsize,
        )
        ax.set_title(arm_titles[arm], fontsize=title_size, color="#222222")
        ax.set_xlabel(r"$\Delta_1$", fontsize=label_size, color="#222222")
    axes_row[0].set_ylabel(r"$\Delta_2$",
                            fontsize=label_size, color="#222222")

    # Colorbar on the right.  Shortened (h=0.36, top at 0.70) so the
    # forward/reverse legend below has clearance; label moved to the
    # LHS of the bar so it cannot overrun the legend.
    cbar_ax = fig.add_axes([0.85, 0.34, 0.018, 0.36])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.yaxis.set_label_position("left")
    cbar.set_label(cbar_label, fontsize=fontsize, color="#222222",
                   labelpad=8)
    cbar.ax.tick_params(labelsize=fontsize - 1, colors="#222222")

    shape_handles, shape_labels = _legend_handles_shapes()
    fig.legend(
        handles=shape_handles, labels=shape_labels,
        handler_map={_CircleHandle: _make_circle_handler()},
        ncol=1, loc="lower left",
        bbox_to_anchor=(0.84, 0.06),
        frameon=True, fontsize=fontsize,
        handlelength=1.9, handleheight=1.9,
        handletextpad=0.6,
        labelspacing=0.9, borderpad=0.7,
        facecolor="white", edgecolor="#444444", labelcolor="#222222",
    )

    return fig


def plot_proportion_heatmap_2d_rows(
    df,
    *,
    deltas1: Iterable[int],
    deltas2: Iterable[int],
    row_values: Sequence,
    row_col: str,
    arm_titles: Mapping[str, str],
    row_label_fn=None,
    fontsize: int = 14,
    scale: float = 1.0,
    suptitle: str | None = None,
    subtitle: str | None = None,
    cbar_label: str = "P(score = ground truth)",
):
    """Multi-row variant of :func:`plot_proportion_heatmap_2d`."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    deltas1 = list(deltas1)
    deltas2 = list(deltas2)
    arms = list(arm_titles.keys())
    row_values = list(row_values)
    n_arms = len(arms)
    n_rows = len(row_values)

    cmap = _PROPORTION_CMAP()
    norm = Normalize(vmin=0.0, vmax=1.0)
    color_of = _proportion_color_fn(cmap, norm)

    fig, axes, label_size, title_size, _, m = _build_rows_figure(
        n_rows=n_rows, n_cols=n_arms, fontsize=fontsize,
        scale=scale, font_scale=1.0,
        suptitle=suptitle, subtitle=subtitle,
    )

    for r, key in enumerate(row_values):
        sub_row = df[df[row_col] == key]
        for c, arm in enumerate(arms):
            ax = axes[r, c]
            _draw_2d_grid_panel(
                ax, sub_row[sub_row["arm"] == arm],
                deltas1=deltas1, deltas2=deltas2,
                cell_value_fn=_proportion_value_fn,
                color_fn=color_of, fontsize=fontsize,
            )
            if r == 0:
                ax.set_title(arm_titles[arm], fontsize=title_size,
                             color="#222222")
            if r == n_rows - 1:
                ax.set_xlabel(r"$\Delta_1$",
                              fontsize=label_size, color="#222222")
        row_label = (row_label_fn(key) if row_label_fn is not None
                     else f"{row_col} = {key}")
        axes[r, 0].set_ylabel(f"{row_label}\n$\\Delta_2$",
                              fontsize=label_size, color="#222222")

    # Colorbar height is fixed in inches so it doesn't bloat with figure
    # height; ~0.9 in above panel_top (which sits at 2.2 in from top).
    fig_h = fig.get_figheight()
    cbar_height_in = 0.18
    cbar_bottom_in = 1.30
    cbar_y_frac    = 1.0 - cbar_bottom_in / fig_h
    cbar_h_frac    = cbar_height_in       / fig_h
    cbar_ax = fig.add_axes([0.10, cbar_y_frac, 0.45, cbar_h_frac])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label(cbar_label, fontsize=fontsize, color="#222222")
    cbar.ax.tick_params(labelsize=fontsize - 1, colors="#222222")

    shape_handles, shape_labels = _legend_handles_shapes()
    fig.legend(
        handles=shape_handles, labels=shape_labels,
        handler_map={_CircleHandle: _make_circle_handler()},
        ncol=2, loc="lower left",
        bbox_to_anchor=(0.62, cbar_y_frac),
        frameon=True, fontsize=fontsize,
        handlelength=1.9, handleheight=1.9,
        handletextpad=0.6, columnspacing=1.6,
        borderpad=0.7,
        facecolor="white", edgecolor="#444444", labelcolor="#222222",
    )

    return fig


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
    """1-D analog of :func:`plot_proportion_heatmap_2d`.

    ``df`` must carry columns ``arm``, ``delta``, ``lflank``, and
    ``fwd_state`` + ``rc_state``.  Rows within a ``(delta, lflank,
    arm)`` cell are aggregated by the proportion-value function; for a
    cross-locus view, each row corresponds to one
    ``(locus, delta, lflank, arm)`` observation and the proportion is
    over loci.

    Glyph convention matches :func:`plot_correctness_heatmap`:
    Rectangle = fwd-strand proportion, Circle = rc-strand proportion.
    Horizontal colorbar sits above the panels with explicit ticks at
    0.0, 0.25, 0.5, 0.75, 1.0; a small fwd/rc shape legend sits beside.

    Returns the :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

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
            cell_value_fn=_proportion_value_fn,
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

    shape_handles, shape_labels = _legend_handles_shapes()
    fig.legend(
        handles=shape_handles, labels=shape_labels,
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
    cell_value_fn=None,
):
    """Multi-row variant of :func:`plot_proportion_heatmap`.

    ``rows`` is a list of ``(key, df_subset)`` pairs.

    ``cell_value_fn`` defaults to the per-locus state-counting function,
    matching the original cross-locus behavior.  Pass a custom function
    to plot pre-aggregated fractions directly (one row per cell with
    ``frac_fwd`` / ``frac_rc`` columns).
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if cell_value_fn is None:
        cell_value_fn = _proportion_value_fn

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

    shape_handles, shape_labels = _legend_handles_shapes()
    fig.legend(
        handles=shape_handles, labels=shape_labels,
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


# ===========================================================================
# Compound-locus layout schematic
# ===========================================================================
#
# Two-block analogue of :func:`plot_layout_schematic`.  Shows a reference
# row (A · R1^N1 · M · R2^N2 · B), an optional NW-flex extended-reference
# row (R1^(f·N1) · M · R2^(f·N2)), and a haplotype row that compresses
# tiles when Δ > 0 and shows fewer tiles when Δ < 0.  Mirror mode flips
# the axis horizontally so left/right read consistently in rc coords.
# ---------------------------------------------------------------------------


def plot_compound_layout_schematic(
    *,
    motif1: str,
    motif2: str,
    ref_n1: int,
    ref_n2: int,
    bridge_len: int,
    delta1_example: int = 0,
    delta2_example: int = 0,
    delta1_range: Tuple[int, int] | None = None,
    delta2_range: Tuple[int, int] | None = None,
    bridge_len_range: Tuple[int, int] | None = None,
    lflank_range: Tuple[int, int] | None = None,
    nwflex_factor: int = 3,
    show_nwflex: bool = True,
    mirror: bool = False,
    ax=None,
    suptitle: str | None = None,
    subtitle: str | None = None,
    figsize: Tuple[float, float] = (12.0, 7.5),
    fontsize: int = 13,
):
    """Render the compound-locus layout schematic.

    Reference row: A · R1^N1 · M · R2^N2 · B (panel widths fixed).
    Optional NW-flex row: R1^(f·N1), R2^(f·N2) inside the same panel
    widths (so the extended-reference scale is visually compressed).
    Haplotype row: compresses tiles when Δ > 0, leaves the right side of
    the block panel empty when Δ < 0.  Δ1 / Δ2 brackets sit above the
    haplotype tiles.  Mirror mode flips the axis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    disp1 = reverse_complement(motif1) if mirror else motif1
    disp2 = reverse_complement(motif2) if mirror else motif2

    flank_w = 7.5
    tile_w = 1.05
    bar_h = 0.7
    bridge_w = max(0.7, 0.35 + 0.18 * bridge_len)

    n1_hap = max(ref_n1 + delta1_example, 0)
    n2_hap = max(ref_n2 + delta2_example, 0)

    block1_panel_w = ref_n1 * tile_w
    block2_panel_w = ref_n2 * tile_w
    x_end = 2 * flank_w + block1_panel_w + bridge_w + block2_panel_w

    # Read stack: three reads with varying lflank / rflank balance —
    # same pattern as NB7's plot_layout_schematic so the compound
    # schematic reads as another version of that figure.
    read_lflanks = [flank_w * 0.20, flank_w * 0.45, flank_w * 0.70]
    read_rflanks = [flank_w * 0.70, flank_w * 0.45, flank_w * 0.20]
    n_reads = len(read_lflanks)
    read_gap = 0.30
    y_read_bot = 0.0
    read_ys = [y_read_bot + k * (bar_h + read_gap) for k in range(n_reads)]

    big_gap = 1.6   # NB7's row spacing
    y_hap = read_ys[-1] + bar_h + big_gap
    if show_nwflex:
        y_nwf = y_hap + bar_h + big_gap
        y_ref = y_nwf + bar_h + big_gap
    else:
        y_nwf = None
        y_ref = y_hap + bar_h + big_gap

    flank_color = "#e8e8e8"
    motif_color_1 = "#a6cee3"
    motif_color_2 = "#b2df8a"
    extra_color = "#fdd49e"
    # Bridge gets a firebrick family — distinct from the orange
    # ``extra_color`` (+motif tiles), the blue / green motif colors,
    # and the purple used for the ``B`` flank label.
    bridge_color = "#e6b8b8"
    bridge_edge  = "firebrick"
    nwflex_color_1 = "#dceaf3"
    nwflex_color_2 = "#e6f1d8"
    nwflex_edge_1 = "#5a9bc0"
    nwflex_edge_2 = "#6aa845"
    missing_edge = "#c0392b"
    read_outline = "#08519c"

    # Row labels: anchor outside the panel in both orientations.  In
    # non-mirror, anchor at the left margin (data x < 0).  In mirror,
    # anchor at the right margin (data x > x_end); xlim is widened on
    # that side below so the text has room.
    row_label_x = (x_end + 0.8) if mirror else -0.4

    def _row_label(y, text):
        ax.text(row_label_x, y + bar_h / 2, text, ha="right", va="center",
                fontsize=fontsize + 1, fontweight="bold", color="#333333")

    def _flank(x, y, w, label):
        ax.add_patch(Rectangle((x, y), w, bar_h, facecolor=flank_color,
                               edgecolor="#888888", linewidth=0.7))
        if label:
            ax.text(x + w / 2, y + bar_h / 2, label,
                    ha="center", va="center",
                    fontsize=fontsize - 2, style="italic", color="#666666")

    def _flank_slice(x, y, w):
        ax.add_patch(Rectangle((x, y), w, bar_h, facecolor=flank_color,
                               edgecolor="#888888", linewidth=0.5))

    def _tile(x, y, w, color, text=None, dashed=False, edge_color=None):
        kw = dict(facecolor=color, edgecolor=edge_color or "#444444",
                  linewidth=0.6)
        if dashed:
            kw["linestyle"] = (0, (2.5, 1.5))
        ax.add_patch(Rectangle((x, y), w, bar_h, **kw))
        if text and w >= 0.45:
            ax.text(x + w / 2, y + bar_h / 2, text,
                    ha="center", va="center",
                    fontsize=fontsize - 5, family="monospace",
                    color="#222222")

    def _bridge_box(x, y, w, label):
        ax.add_patch(Rectangle((x, y), w, bar_h, facecolor=bridge_color,
                               edgecolor=bridge_edge, linewidth=0.8))
        if label:
            ax.text(x + w / 2, y + bar_h / 2, label,
                    ha="center", va="center",
                    fontsize=fontsize - 2, fontweight="bold",
                    color=bridge_edge)

    # Reference row
    _row_label(y_ref, "Reference")
    x0 = 0.0
    _flank(x0, y_ref, flank_w, "left flank")
    x0 += flank_w
    for i in range(ref_n1):
        _tile(x0 + i * tile_w, y_ref, tile_w, motif_color_1, disp1)
    x0 += block1_panel_w
    _bridge_box(x0, y_ref, bridge_w, "M")
    x0 += bridge_w
    for i in range(ref_n2):
        _tile(x0 + i * tile_w, y_ref, tile_w, motif_color_2, disp2)
    x0 += block2_panel_w
    _flank(x0, y_ref, flank_w, "right flank")

    # Above-the-line labels for every region, matching NB7's A/Z/B style:
    # A (left flank, blue, bold), R1, M, R2 (with bridge length), B (right
    # flank, purple, bold).
    label_y = y_ref + bar_h + 0.18
    ax.text(flank_w / 2, label_y, "$A$",
            ha="center", va="bottom",
            fontsize=fontsize + 1, fontweight="bold", color="#2060a0")
    ax.text(flank_w + block1_panel_w / 2, label_y,
            f"$R_1^{{{ref_n1}}}$", ha="center", va="bottom",
            fontsize=fontsize, color="#1f6090", fontweight="bold")
    if bridge_len_range is not None:
        lo, hi = bridge_len_range
        m_label = f"$M \\in [{lo}, {hi}]$ bp"
    else:
        m_label = f"$M$ ({bridge_len} bp)"
    ax.text(flank_w + block1_panel_w + bridge_w / 2, label_y,
            m_label, ha="center", va="bottom",
            fontsize=fontsize, color=bridge_edge, fontweight="bold")
    ax.text(flank_w + block1_panel_w + bridge_w + block2_panel_w / 2,
            label_y,
            f"$R_2^{{{ref_n2}}}$", ha="center", va="bottom",
            fontsize=fontsize, color="#3a7f1c", fontweight="bold")
    ax.text(flank_w + block1_panel_w + bridge_w + block2_panel_w + flank_w / 2,
            label_y, "$B$",
            ha="center", va="bottom",
            fontsize=fontsize + 1, fontweight="bold", color="#7030a0")

    # NW-flex row
    if show_nwflex:
        _row_label(y_nwf, "NW-flex ref")
        x0 = 0.0
        _flank(x0, y_nwf, flank_w, "left flank")
        x0 += flank_w
        n1e = nwflex_factor * ref_n1
        n2e = nwflex_factor * ref_n2
        t1e = block1_panel_w / n1e if n1e > 0 else tile_w
        for i in range(n1e):
            _tile(x0 + i * t1e, y_nwf, t1e, nwflex_color_1,
                  dashed=True, edge_color=nwflex_edge_1)
        x0 += block1_panel_w
        _bridge_box(x0, y_nwf, bridge_w, "M")
        x0 += bridge_w
        t2e = block2_panel_w / n2e if n2e > 0 else tile_w
        for i in range(n2e):
            _tile(x0 + i * t2e, y_nwf, t2e, nwflex_color_2,
                  dashed=True, edge_color=nwflex_edge_2)
        x0 += block2_panel_w
        _flank(x0, y_nwf, flank_w, "right flank")

        ax.text(flank_w + block1_panel_w / 2, y_nwf + bar_h + 0.18,
                f"$R_1^{{{n1e}}}$", ha="center", va="bottom",
                fontsize=fontsize - 1, color="#1f6090")
        ax.text(flank_w + block1_panel_w + bridge_w + block2_panel_w / 2,
                y_nwf + bar_h + 0.18,
                f"$R_2^{{{n2e}}}$", ha="center", va="bottom",
                fontsize=fontsize - 1, color="#3a7f1c")

    # Haplotype row
    _row_label(y_hap, "Haplotype")
    x0 = 0.0
    ax.add_patch(Rectangle(
        (0, y_hap),
        2 * flank_w + block1_panel_w + bridge_w + block2_panel_w,
        bar_h, facecolor="none", edgecolor="#888888", linewidth=0.7,
        zorder=2,
    ))
    _flank(x0, y_hap, flank_w, "left flank")
    x0 += flank_w
    if n1_hap > 0:
        if delta1_example > 0:
            t1 = block1_panel_w / n1_hap
            for i in range(ref_n1):
                _tile(x0 + i * t1, y_hap, t1, motif_color_1, disp1)
            for i in range(ref_n1, n1_hap):
                _tile(x0 + i * t1, y_hap, t1, extra_color, disp1)
        else:
            t1 = tile_w
            for i in range(n1_hap):
                _tile(x0 + i * t1, y_hap, t1, motif_color_1, disp1)
    x0 += block1_panel_w
    _bridge_box(x0, y_hap, bridge_w, "M")
    x0 += bridge_w
    if n2_hap > 0:
        if delta2_example > 0:
            t2 = block2_panel_w / n2_hap
            for i in range(ref_n2):
                _tile(x0 + i * t2, y_hap, t2, motif_color_2, disp2)
            for i in range(ref_n2, n2_hap):
                _tile(x0 + i * t2, y_hap, t2, extra_color, disp2)
        else:
            t2 = tile_w
            for i in range(n2_hap):
                _tile(x0 + i * t2, y_hap, t2, motif_color_2, disp2)
    x0 += block2_panel_w
    _flank(x0, y_hap, flank_w, "right flank")

    if delta1_example != 0 or delta1_range is not None:
        cx = flank_w + block1_panel_w / 2
        ax.annotate(
            "", xy=(cx - 0.5, y_hap + bar_h + 0.10),
            xytext=(cx + 0.5, y_hap + bar_h + 0.10),
            arrowprops=dict(arrowstyle="<->", color=missing_edge, lw=1.4),
        )
        if delta1_range is not None:
            lo, hi = delta1_range
            d1_label = f"$\\Delta_1 \\in [{lo:+d}, {hi:+d}]$"
        else:
            d1_label = f"$\\Delta_1 = {delta1_example:+d}$"
        ax.text(cx, y_hap + bar_h + 0.28,
                d1_label,
                ha="center", va="bottom",
                fontsize=fontsize - 1, color=missing_edge, fontweight="bold")
    if delta2_example != 0 or delta2_range is not None:
        cx = flank_w + block1_panel_w + bridge_w + block2_panel_w / 2
        ax.annotate(
            "", xy=(cx - 0.5, y_hap + bar_h + 0.10),
            xytext=(cx + 0.5, y_hap + bar_h + 0.10),
            arrowprops=dict(arrowstyle="<->", color=missing_edge, lw=1.4),
        )
        if delta2_range is not None:
            lo, hi = delta2_range
            d2_label = f"$\\Delta_2 \\in [{lo:+d}, {hi:+d}]$"
        else:
            d2_label = f"$\\Delta_2 = {delta2_example:+d}$"
        ax.text(cx, y_hap + bar_h + 0.28,
                d2_label,
                ha="center", va="bottom",
                fontsize=fontsize - 1, color=missing_edge, fontweight="bold")

    # --- Read stack -----------------------------------------------------
    stack_center = (read_ys[0] + read_ys[-1]) / 2
    _row_label(stack_center, "Reads")

    for k in range(n_reads):
        yk = read_ys[k]
        lflank_ex = read_lflanks[k]
        rflank_ex = read_rflanks[k]
        rx0 = flank_w - lflank_ex
        _flank_slice(rx0, yk, lflank_ex)
        # Block 1 tiles (no in-tile text — labeled on Hap above).
        x0 = flank_w
        if n1_hap > 0:
            if delta1_example > 0:
                t1 = block1_panel_w / n1_hap
                for i in range(ref_n1):
                    _tile(x0 + i * t1, yk, t1, motif_color_1)
                for i in range(ref_n1, n1_hap):
                    _tile(x0 + i * t1, yk, t1, extra_color)
            else:
                t1 = tile_w
                for i in range(n1_hap):
                    _tile(x0 + i * t1, yk, t1, motif_color_1)
        x0 += block1_panel_w
        _bridge_box(x0, yk, bridge_w, label=None)
        x0 += bridge_w
        if n2_hap > 0:
            if delta2_example > 0:
                t2 = block2_panel_w / n2_hap
                for i in range(ref_n2):
                    _tile(x0 + i * t2, yk, t2, motif_color_2)
                for i in range(ref_n2, n2_hap):
                    _tile(x0 + i * t2, yk, t2, extra_color)
            else:
                t2 = tile_w
                for i in range(n2_hap):
                    _tile(x0 + i * t2, yk, t2, motif_color_2)
        x0 += block2_panel_w
        _flank_slice(x0, yk, rflank_ex)
        rx1 = x0 + rflank_ex
        ax.add_patch(FancyBboxPatch(
            (rx0, yk), rx1 - rx0, bar_h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            facecolor="none", edgecolor=read_outline, linewidth=1.1,
            zorder=3,
        ))

    # lflank extent bracket on the topmost read (heaviest lflank).
    top_y = read_ys[-1]
    top_lflank = read_lflanks[-1]
    arrow_y = top_y + bar_h + 0.15
    text_y  = top_y + bar_h + 0.30
    ax.annotate(
        "", xy=(flank_w - top_lflank, arrow_y),
        xytext=(flank_w, arrow_y),
        arrowprops=dict(arrowstyle="<->", color=read_outline, lw=1.4),
    )
    if lflank_range is not None:
        lo, hi = lflank_range
        lflank_label = f"lflank extent $\\in [{lo}, {hi}]$"
    else:
        lflank_label = "lflank extent"
    ax.text(flank_w - top_lflank / 2, text_y,
            lflank_label,
            ha="center", va="bottom",
            fontsize=fontsize - 1, color=read_outline, fontweight="bold")

    # Frame.  Both modes extend the low-x side by 4.5 units for the
    # non-mirror row labels; mirror extends the high-x side by 5.0 for
    # the row labels after invert.
    if mirror:
        ax.set_xlim(-4.5, x_end + 5.0)
        ax.invert_xaxis()
    else:
        ax.set_xlim(-4.5, x_end + 0.5)
    ax.set_ylim(-0.4, y_ref + bar_h + 1.1)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    title_size = fontsize + 3
    label_size = fontsize + 1
    if standalone:
        if suptitle is not None:
            fig.suptitle(suptitle, fontsize=title_size, fontweight="bold",
                         y=0.98, color="#222222")
        if subtitle:
            sub_y = 0.91 if suptitle else 0.94
            fig.text(0.5, sub_y, subtitle, ha="center", va="top",
                     fontsize=label_size, style="italic", color="#444444")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90 if suptitle else 0.96))
    else:
        if suptitle is not None:
            ax.set_title(suptitle, fontsize=title_size, fontweight="bold",
                         color="#222222", pad=label_size + 6)
        if subtitle:
            ax.text(0.5, 1.0, subtitle, transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=label_size - 1, style="italic", color="#444444")
    return fig

