"""
viz.py — text visualizations for the simulation harness.

The single entry point is :func:`render_zoom`, which produces a
column-aligned ASCII view of a BWA-MEM (or NW-flex) alignment around
a repeat-zone interval.  Output is intentionally a plain multi-line
string so it composes well in notebooks and tests.
"""

from __future__ import annotations

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
