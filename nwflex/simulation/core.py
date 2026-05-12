"""
simulation.py — Simulation harness for notebook 07
(NW-flex vs BWA-MEM comparison).
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Tuple

from nwflex.repeats import STRLocus


# ---------------------------------------------------------------------------
# Flank window cleaning ("squeegee" off partial-motif edges)
# ---------------------------------------------------------------------------

def clean_flank_window(
    motif: str,
    panel_flank: str,
    flank_len: int,
    side: str,
    *,
    max_advance: int = 10,
) -> str:
    """
    Carve a clean ``flank_len``-bp window from a panel flank.

    The window's edge nearest the repeat must not match the
    corresponding motif base (otherwise the flank would partial-extend
    the motif); the window is slid by up to ``max_advance`` bp until
    that holds.

    Parameters
    ----------
    motif : str
        The repeat motif.
    panel_flank : str
        The full panel flank sequence.
    flank_len : int
        Length of the window to carve out.
    side : {"left", "right"}
        Which flank.  ``"left"`` anchors at the right edge of
        ``panel_flank`` and slides leftward; ``"right"`` mirrors.
    max_advance : int, default 10
        Maximum number of bp to slide before giving up.

    Returns
    -------
    str
        A flank window of length ``flank_len`` with a clean motif edge.
    """
    if side == "left":
        forbidden = motif[-1]
        for adv in range(max_advance + 1):
            end = len(panel_flank) - adv
            start = end - flank_len
            if start < 0:
                raise ValueError(
                    f"left flank too short: need {flank_len + adv} bp, "
                    f"have {len(panel_flank)}"
                )
            window = panel_flank[start:end]
            if window[-1] != forbidden:
                return window
        raise ValueError(
            f"left flank could not be cleaned within {max_advance} advances "
            f"(motif={motif!r})"
        )
    if side == "right":
        forbidden = motif[0]
        for adv in range(max_advance + 1):
            start = adv
            end = adv + flank_len
            if end > len(panel_flank):
                raise ValueError(
                    f"right flank too short: need {flank_len + adv} bp, "
                    f"have {len(panel_flank)}"
                )
            window = panel_flank[start:end]
            if window[0] != forbidden:
                return window
        raise ValueError(
            f"right flank could not be cleaned within {max_advance} advances "
            f"(motif={motif!r})"
        )
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


# ---------------------------------------------------------------------------
# Locus construction from a panel row
# ---------------------------------------------------------------------------

def build_locus_from_panel(
    motif: str,
    panel_lflank: str,
    panel_rflank: str,
    *,
    ref_n: int,
    flank_len: int,
    max_flank_advance: int = 10,
) -> STRLocus:
    """
    Build an :class:`~nwflex.repeats.STRLocus` from a panel locus.

    Cleans both flanks with :func:`clean_flank_window`, then assembles
    ``X = A · R^ref_n · B``.

    Parameters
    ----------
    motif : str
        Repeat motif.
    panel_lflank, panel_rflank : str
        Full panel flank sequences.
    ref_n : int
        Repeat count ``N`` for the assembled locus.
    flank_len : int
        Length of each flank window.
    max_flank_advance : int, default 10
        Forwarded to :func:`clean_flank_window`.

    Returns
    -------
    STRLocus
    """
    A = clean_flank_window(
        motif, panel_lflank, flank_len, "left",
        max_advance=max_flank_advance,
    )
    B = clean_flank_window(
        motif, panel_rflank, flank_len, "right",
        max_advance=max_flank_advance,
    )
    return STRLocus(A=A, R=motif, N=ref_n, B=B)


# ---------------------------------------------------------------------------
# Haplotype construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Haplotype:
    """
    A simulated haplotype: a sequence with a known flank/repeat-block
    structure and an optional single-base substitution.

    Attributes
    ----------
    sequence : str
        Full haplotype DNA sequence.
    flank_len : int
        Length of the left flank in ``sequence``.  The repeat block
        spans ``[flank_len, flank_len + body_len)``.
    body_len : int
        Length of the repeat block between the two flanks, in bp.
    snv_pos : int or None
        Absolute 0-based position of the SNV in ``sequence``, or
        ``None`` if no SNV was applied.
    snv_base : str or None
        Base placed at ``snv_pos``, or ``None``.
    """
    sequence: str
    flank_len: int
    body_len: int
    snv_pos: Optional[int] = None
    snv_base: Optional[str] = None


def build_haplotype(
    locus: STRLocus,
    delta: int,
    *,
    snv: Optional[Tuple[int, str]] = None,
) -> Haplotype:
    """
    Build a haplotype from ``locus``, perturbing the repeat count by
    ``delta`` and optionally substituting a single base.

    Parameters
    ----------
    locus : STRLocus
        Source locus.
    delta : int
        Repeat-count perturbation; the haplotype has ``locus.N + delta``
        motif copies.  Must satisfy ``locus.N + delta >= 0``.
    snv : (abs_pos, base) or None
        Optional single-base substitution at 0-based ``abs_pos`` within
        the haplotype sequence.

    Returns
    -------
    Haplotype
    """
    hap_n = locus.N + delta
    if hap_n < 0:
        raise ValueError(
            f"haplotype repeat count {hap_n} is negative "
            f"(locus.N={locus.N}, delta={delta})"
        )
    seq = locus.A + locus.R * hap_n + locus.B
    flank_len = len(locus.A)
    body_len = len(locus.R) * hap_n
    snv_pos: Optional[int] = None
    snv_base: Optional[str] = None
    if snv is not None:
        abs_pos, base = snv
        if abs_pos < 0 or abs_pos >= len(seq):
            raise ValueError(
                f"SNV abs_pos {abs_pos} out of range [0, {len(seq)})"
            )
        if len(base) != 1:
            raise ValueError(
                f"SNV base must be a single character, got {base!r}"
            )
        seq = seq[:abs_pos] + base + seq[abs_pos + 1:]
        snv_pos = abs_pos
        snv_base = base
    return Haplotype(
        sequence=seq,
        flank_len=flank_len,
        body_len=body_len,
        snv_pos=snv_pos,
        snv_base=snv_base,
    )


# ---------------------------------------------------------------------------
# Read tiling across a haplotype
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Read:
    """
    A single tiled read.

    Attributes
    ----------
    sequence : str
        Read DNA sequence of length ``read_len``.
    var_start : int
        0-based start position of the read in the source haplotype.
    lflank_extent : int
        Number of bp the read covers in the haplotype's left flank.
    rflank_extent : int
        Number of bp the read covers in the haplotype's right flank.
    """
    sequence: str
    var_start: int
    lflank_extent: int
    rflank_extent: int


def tile_reads(
    hap,
    read_len: int,
    k_min_flank: int,
    *,
    step: int = 1,
) -> List[Read]:
    """
    Tile reads of length ``read_len`` across ``hap``, requiring at least
    ``k_min_flank`` bp of flank context on both sides of the body.

    Parameters
    ----------
    hap : Haplotype
        Any object exposing ``sequence``, ``flank_len``, and
        ``body_len``.
    read_len : int
        Read length.
    k_min_flank : int
        Minimum bp of flank required on each side of the body.
    step : int, default 1
        Spacing between successive read starts.

    Returns
    -------
    list of Read
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    body_end = hap.flank_len + hap.body_len
    s_min = body_end + k_min_flank - read_len
    s_max = hap.flank_len - k_min_flank
    if s_max < s_min:
        raise ValueError(
            f"no reads fit: body_len={hap.body_len}, read_len={read_len}, "
            f"k_min_flank={k_min_flank}; require "
            f"read_len >= body_len + 2*k_min_flank "
            f"= {hap.body_len + 2 * k_min_flank}"
        )
    s_min = max(s_min, 0)
    s_max = min(s_max, len(hap.sequence) - read_len)
    reads: List[Read] = []
    for s in range(s_min, s_max + 1, step):
        seq = hap.sequence[s:s + read_len]
        lflank_extent = max(0, min(s + read_len, hap.flank_len) - s)
        rflank_extent = max(0, (s + read_len) - body_end)
        reads.append(Read(
            sequence=seq,
            var_start=s,
            lflank_extent=lflank_extent,
            rflank_extent=rflank_extent,
        ))
    return reads


# ---------------------------------------------------------------------------
# BWA-MEM single-strand alignment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BwaResult:
    """
    One read's outcome from a single ``bwa mem`` run.

    Attributes
    ----------
    is_unmapped : bool
        True if BWA reported the read unmapped (SAM flag 0x4).
    pos : int or None
        1-based reference position of the first aligned read base
        (the SAM ``POS`` field), or ``None`` when unmapped.
    cigar : str or None
        CIGAR string, or ``None`` when unmapped.
    score : int or None
        Alignment score from the SAM ``AS:i:`` tag, or ``None`` when
        unmapped.
    """
    is_unmapped: bool
    pos: Optional[int]
    cigar: Optional[str]
    score: Optional[int]


def _parse_sam_line(line: str) -> Optional[Tuple[str, BwaResult]]:
    """Parse one SAM line into ``(read_id, BwaResult)``.

    Returns ``None`` for headers and for secondary/supplementary records
    so the caller can ignore them.
    """
    if line.startswith("@"):
        return None
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 11:
        return None
    flag = int(fields[1])
    if flag & 0x100 or flag & 0x800:
        return None
    is_unmapped = bool(flag & 0x4)
    score: Optional[int] = None
    if not is_unmapped:
        for tag in fields[11:]:
            if tag.startswith("AS:i:"):
                score = int(tag[5:])
                break
    return fields[0], BwaResult(
        is_unmapped=is_unmapped,
        pos=None if is_unmapped else int(fields[3]),
        cigar=None if is_unmapped else fields[5],
        score=score,
    )


def align_bwa(
    reference: str,
    reads: List[Read],
    *,
    no_clip: bool,
) -> List[BwaResult]:
    """
    Run ``bwa index`` + ``bwa mem`` against ``reference`` for every read.

    Uses a temporary directory; nothing persists on disk.  Reads are
    written to a FASTQ with synthesized IDs ``r0, r1, ...`` so the
    returned list is index-aligned with ``reads``.

    Parameters
    ----------
    reference : str
        Reference sequence (single-locus FASTA contents).
    reads : list of Read
        Reads to align.
    no_clip : bool
        If True, pass ``-L 500`` to ``bwa mem`` so the soft-clip penalty
        is large enough that clipping never improves the score.

    Returns
    -------
    list of BwaResult
        Per-read alignment outcomes, in the same order as ``reads``.

    Raises
    ------
    FileNotFoundError
        If ``bwa`` is not on ``PATH``.
    """
    read_ids = [f"r{i}" for i in range(len(reads))]
    with tempfile.TemporaryDirectory(prefix="nwflex_sim_bwa_") as tmp:
        tmpdir = Path(tmp)
        ref_fa = tmpdir / "ref.fa"
        reads_fq = tmpdir / "reads.fq"

        # Write the reference FASTA.
        ref_fa.write_text(">locus\n" + reference + "\n")

        # Write reads as FASTQ with synthetic IDs (r0..rN) and a constant
        # 'I' quality string (BWA needs a quality field; the value is unused).
        with open(reads_fq, "w") as f:
            for rid, r in zip(read_ids, reads):
                f.write(f"@{rid}\n{r.sequence}\n+\n{'I' * len(r.sequence)}\n")

        # Index the reference.
        subprocess.run(
            ["bwa", "index", str(ref_fa)],
            check=True, capture_output=True,
        )

        # Align reads with bwa mem (-L 500 disables soft-clipping).
        cmd = ["bwa", "mem"]
        if no_clip:
            cmd.extend(["-L", "500"])
        cmd.extend([str(ref_fa), str(reads_fq)])
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Parse the SAM stream into one BwaResult per input read, in input order.
    parsed: dict = {}
    for line in proc.stdout.splitlines():
        item = _parse_sam_line(line)
        if item is not None:
            rid, result = item
            parsed[rid] = result
    return [parsed[rid] for rid in read_ids]


# ---------------------------------------------------------------------------
# Reverse-complement helpers and both-strands alignment
# ---------------------------------------------------------------------------

_COMP_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    """
    Return the reverse complement of a DNA sequence.

    ``A``/``T`` and ``C``/``G`` swap; ``N`` is preserved.  Lower-case
    input is preserved in case (``a`` → ``t``).

    Parameters
    ----------
    seq : str
        DNA sequence over ``{A, C, G, T, N}`` (case-insensitive).

    Returns
    -------
    str
        Reverse complement of ``seq``.
    """
    return seq.translate(_COMP_TABLE)[::-1]


def mirror_reads(reads: List[Read]) -> List[Read]:
    """
    Reverse-complement a list of reads for the mirror frame.

    Each returned read carries the reverse-complemented sequence with
    ``lflank_extent`` and ``rflank_extent`` swapped (since left and
    right flanks swap under rc).  ``var_start`` is carried through
    unchanged.

    Useful in per-haplotype sweeps where the mirror reference and zone
    were already built once at setup via :func:`build_mirror_frame` —
    each new haplotype only needs its reads flipped.

    Parameters
    ----------
    reads : list of Read
        Forward reads.

    Returns
    -------
    list of Read
        Mirror-frame reads.
    """
    return [
        Read(
            sequence=reverse_complement(r.sequence),
            var_start=r.var_start,
            lflank_extent=r.rflank_extent,
            rflank_extent=r.lflank_extent,
        )
        for r in reads
    ]


def build_mirror_frame(
    reference: str,
    reads: List[Read],
    zone: Tuple[int, int],
    *,
    extra_reference: Optional[str] = None,
    extra_zone: Optional[Tuple[int, int]] = None,
) -> Tuple:
    """
    Build the mirror (reverse-complement) frame of a simulation.

    The mirror frame is the reverse-complemented view of the forward
    simulation: the reference is reverse-complemented, each read is
    reverse-complemented (with its ``lflank_extent`` and
    ``rflank_extent`` swapped, since left and right flanks swap under
    rc), and the half-open repeat interval is mapped into rc-reference
    coordinates.

    Built into the simulation by design so every aligner can be
    evaluated on both orientations on equal footing.

    When ``extra_reference`` and ``extra_zone`` are provided, that
    second reference (e.g., NW-flex's extended ``A · R^{3N} · B``) and
    its zone are mirrored alongside the primary one and appended to
    the returned tuple.  The two extras must be provided together.

    Parameters
    ----------
    reference : str
        Forward reference sequence.
    reads : list of Read
        Reads tiled against the forward reference.
    zone : (int, int)
        Half-open repeat interval ``(z_start, z_end)`` in the forward
        reference (0-based).
    extra_reference : str, optional
        Second forward reference to mirror in the same call.
    extra_zone : (int, int), optional
        Half-open repeat interval in ``extra_reference``.

    Returns
    -------
    Without extras: ``(rc_reference, rc_reads, rc_zone)``.
    With extras: ``(rc_reference, rc_reads, rc_zone, rc_extra_reference,
    rc_extra_zone)``.

    ``rc_reads`` carry reverse-complemented sequences with
    ``lflank_extent`` / ``rflank_extent`` swapped; ``var_start`` is
    carried through unchanged.  The mirror zone for a reference of
    length ``n`` and forward zone ``(z_start, z_end)`` is
    ``(n - z_end, n - z_start)``.

    Raises
    ------
    ValueError
        If exactly one of ``extra_reference`` and ``extra_zone`` is
        provided.
    """
    if (extra_reference is None) != (extra_zone is None):
        raise ValueError(
            "extra_reference and extra_zone must be provided together "
            "(both or neither)"
        )

    z_start, z_end = zone
    n = len(reference)
    rc_reference = reverse_complement(reference)
    rc_reads = mirror_reads(reads)
    rc_zone = (n - z_end, n - z_start)

    if extra_reference is None:
        return rc_reference, rc_reads, rc_zone

    n_extra = len(extra_reference)
    rc_extra_reference = reverse_complement(extra_reference)
    e_start, e_end = extra_zone
    rc_extra_zone = (n_extra - e_end, n_extra - e_start)
    return (rc_reference, rc_reads, rc_zone,
            rc_extra_reference, rc_extra_zone)


@dataclass(frozen=True)
class BwaBothStrandsResult:
    """
    One read's outcome from running ``bwa mem`` on both orientations.

    The same read is aligned twice: once forward against ``reference``,
    once with both read and reference reverse-complemented.  Both
    per-strand results are reported; downstream correctness logic
    decides how to combine them.

    Attributes
    ----------
    fwd : BwaResult
        Result of aligning the read against ``reference``.
    rc : BwaResult
        Result of aligning ``reverse_complement(read)`` against
        ``reverse_complement(reference)``.
    """
    fwd: BwaResult
    rc: BwaResult


def align_bwa_both_strands(
    reference: str,
    reads: List[Read],
    *,
    no_clip: bool,
) -> List[BwaBothStrandsResult]:
    """
    Align every read in both orientations and return per-read results.

    Smith-Waterman returns one best alignment, not all of them; when
    several alignments tie, the choice depends on the order in which DP
    cells are evaluated.  Reverse-complementing both read and reference
    changes that order, so the two arms can return different (equally
    optimal) alignments.  Running both orientations exposes both.

    Parameters
    ----------
    reference : str
        Forward reference sequence.
    reads : list of Read
        Reads tiled against the forward reference.
    no_clip : bool
        Forwarded to :func:`align_bwa` for both arms.

    Returns
    -------
    list of BwaBothStrandsResult
        Per-read forward and reverse-complement outcomes, in the same
        order as ``reads``.
    """
    rc_reference = reverse_complement(reference)
    rc_reads = [
        Read(
            sequence=reverse_complement(r.sequence),
            var_start=r.var_start,
            lflank_extent=r.rflank_extent,
            rflank_extent=r.lflank_extent,
        )
        for r in reads
    ]
    fwd_results = align_bwa(reference, reads, no_clip=no_clip)
    rc_results = align_bwa(rc_reference, rc_reads, no_clip=no_clip)
    return [
        BwaBothStrandsResult(fwd=f, rc=b)
        for f, b in zip(fwd_results, rc_results)
    ]


# ---------------------------------------------------------------------------
# NW-flex harness
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NwflexResult:
    """
    One read's outcome from a single :func:`align_nwflex` run.

    Attributes
    ----------
    pos : int
        1-based reference position of the first aligned read base
        (SAM ``POS`` convention).  NW-flex always returns an alignment,
        so this is never ``None``.
    cigar : str
        CIGAR string for the alignment.
    score : float
        NW-flex's reported alignment score under the supplied scoring
        scheme.  Matches ``score_alignment`` on the same CIGAR by
        construction.
    """
    pos: int
    cigar: str
    score: float


def align_nwflex(
    reference: str,
    reads: List[Read],
    *,
    extra_predecessors: Any,
    score_matrix: Any,
    gap_open: float,
    gap_extend: float,
    alphabet_to_index: Any,
    max_read_length: Optional[int] = None,
) -> List[NwflexResult]:
    """
    Run NW-flex against ``reference`` for every read.

    Direction-agnostic by design: the harness takes a list of reads
    and a reference and returns one alignment per read, in the same
    order as ``reads``.  To evaluate both strands, call this twice —
    once on the forward ``(reads, reference, zone)`` pair and once on
    the mirror pair built by :func:`build_mirror_frame`.

    Builds a single :class:`~nwflex.aligners.RefAligner` against
    ``reference`` (so the cached DP buffers are reused across reads),
    aligns each read, and returns one :class:`NwflexResult` per input.

    Parameters
    ----------
    reference : str
        Reference sequence the aligner runs against (often the
        ``3N``-extended reference; the EP pattern must be built for
        this reference's zone).
    reads : list of Read
        Reads to align.
    extra_predecessors
        EP pattern for the STR-aware extension, as produced by
        ``nwflex.ep_patterns.build_EP_STR_phase`` for the matching
        ``(reference, zone, motif length)``.
    score_matrix, gap_open, gap_extend, alphabet_to_index
        Scoring scheme, forwarded to ``RefAligner``.
    max_read_length : int, optional
        DP-buffer size hint.  Defaults to ``max(len(r.sequence) for r
        in reads) + 5``.

    Returns
    -------
    list of NwflexResult
    """
    from nwflex.aligners import RefAligner

    if max_read_length is None:
        max_read_length = (
            max((len(r.sequence) for r in reads), default=0) + 5
        )

    aligner = RefAligner(
        ref=reference,
        extra_predecessors=extra_predecessors,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        alphabet_to_index=alphabet_to_index,
        free_X=True,
        free_Y=True,
        max_read_length=max_read_length,
    )

    out: List[NwflexResult] = []
    for r in reads:
        a = aligner.align_simple(r.sequence)
        out.append(NwflexResult(
            pos=int(a["start_pos"]),
            cigar=a["cigar"],
            score=float(a["score"]),
        ))
    return out


# ---------------------------------------------------------------------------
# CIGAR parsing and Z-region decoding
# ---------------------------------------------------------------------------

_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")


def parse_cigar(cigar: str) -> List[Tuple[int, str]]:
    """
    Tokenize a CIGAR string into a list of ``(length, op)`` tuples.

    Parameters
    ----------
    cigar : str
        CIGAR string, e.g. ``"5M2I3M"``.

    Returns
    -------
    list of (int, str)
        Sequence of ``(length, op)`` pairs in order.

    Raises
    ------
    ValueError
        If the input contains characters that cannot be reassembled to
        the original CIGAR (i.e., malformed input).
    """
    parts = [(int(n), op) for n, op in _CIGAR_RE.findall(cigar)]
    if "".join(f"{n}{op}" for n, op in parts) != cigar:
        raise ValueError(f"invalid CIGAR: {cigar!r}")
    return parts


def decode_z_bp(
    cigar: str,
    start_pos_1based: int,
    z_start: int,
    z_end: int,
    *,
    convention: str,
) -> int:
    """
    Count read bases the alignment placed inside ``[z_start, z_end)``.

    Walks the CIGAR from ``start_pos_1based`` (1-based reference, as in
    SAM ``POS``) and accumulates read bases whose reference cursor falls
    inside the half-open repeat interval.  Match-like ops (``M``, ``=``,
    ``X``) consume one reference position per base.  Deletion-like ops
    (``D``, ``N``) advance the reference cursor without contributing
    read bases.  Insertions consume zero reference positions but
    contribute read bases — the **boundary convention** controls whether
    an insertion sitting at the left edge of the interval counts as
    inside.

    Parameters
    ----------
    cigar : str
        CIGAR string for the alignment.
    start_pos_1based : int
        1-based reference position of the first aligned read base
        (the SAM ``POS`` field).
    z_start : int
        Left edge of the repeat interval (0-based, inclusive).
    z_end : int
        Right edge of the repeat interval (0-based, exclusive).
    convention : {"bwa", "nwflex"}
        Boundary convention for insertions:

        - ``"bwa"`` — an insertion at the left boundary is counted
          inside the repeat (cursor exactly at ``z_start``).
        - ``"nwflex"`` — an insertion at the left boundary is counted
          outside the repeat.

    Returns
    -------
    int
        Number of read bases placed inside ``[z_start, z_end)`` under
        the chosen convention.
    """
    if convention not in ("bwa", "nwflex"):
        raise ValueError(
            f"convention must be 'bwa' or 'nwflex', got {convention!r}"
        )
    ref_pos = start_pos_1based - 1
    count = 0
    for length, op in parse_cigar(cigar):
        if op in ("M", "=", "X"):
            for _ in range(length):
                if z_start <= ref_pos < z_end:
                    count += 1
                ref_pos += 1
        elif op == "I":
            if convention == "bwa":
                if z_start <= ref_pos <= z_end:
                    count += length
            else:  # nwflex
                if z_start < ref_pos <= z_end:
                    count += length
        elif op in ("D", "N"):
            ref_pos += length
        # S, H, P consume neither read bp nor reference position inside Z.
    return count


def flank_bases_consumed(
    cigar: str,
    start_pos_1based: int,
    z_start: int,
    z_end: int,
) -> Tuple[int, int]:
    """
    Reference bp the alignment consumes in the left and right flanks.

    Walks the CIGAR and counts reference positions reached by
    reference-consuming ops (``M``, ``=``, ``X``, ``D``, ``N``) that
    fall outside ``[z_start, z_end)``.  Soft-clip ``S`` and hard-clip
    ``H`` do not consume reference; insertions ``I`` do not consume
    reference.

    Parameters
    ----------
    cigar : str
        CIGAR string for the alignment.
    start_pos_1based : int
        1-based reference position of the first aligned read base.
    z_start, z_end : int
        Half-open repeat interval in the reference (0-based).

    Returns
    -------
    (int, int)
        ``(left, right)`` — reference bp consumed in the left flank
        (positions ``< z_start``) and the right flank (positions
        ``>= z_end``).
    """
    ref_pos = start_pos_1based - 1
    left = 0
    right = 0
    for length, op in parse_cigar(cigar):
        if op in ("M", "=", "X", "D", "N"):
            for _ in range(length):
                if ref_pos < z_start:
                    left += 1
                elif ref_pos >= z_end:
                    right += 1
                ref_pos += 1
    return left, right


def is_arm_correct(
    cigar: Optional[str],
    start_pos_1based: Optional[int],
    z_start: int,
    z_end: int,
    truth_z_bp: int,
    *,
    convention: str,
    min_flank: int = 1,
) -> bool:
    """
    Per-arm correctness verdict.

    A read is correct under an arm when both:

    1. ``decode_z_bp(cigar, ...) == truth_z_bp`` — the alignment placed
       exactly the truth number of read bp inside the repeat interval.
    2. ``flank_bases_consumed(cigar, ...)`` reports at least
       ``min_flank`` reference bp in each flank — the alignment spans
       the repeat.

    Returns ``False`` when ``cigar`` or ``start_pos_1based`` is
    ``None`` (unmapped read).

    Parameters
    ----------
    cigar : str or None
        CIGAR string, or ``None`` for an unmapped read.
    start_pos_1based : int or None
        1-based reference position, or ``None`` for an unmapped read.
    z_start, z_end : int
        Half-open repeat interval (0-based).
    truth_z_bp : int
        Ground-truth read bp inside the repeat interval.
    convention : {"bwa", "nwflex"}
        Boundary convention forwarded to :func:`decode_z_bp`.
    min_flank : int, default 1
        Minimum reference bp consumed in each flank.
    """
    if cigar is None or start_pos_1based is None:
        return False
    z_bp = decode_z_bp(
        cigar, start_pos_1based, z_start, z_end, convention=convention
    )
    if z_bp != truth_z_bp:
        return False
    left, right = flank_bases_consumed(cigar, start_pos_1based, z_start, z_end)
    return left >= min_flank and right >= min_flank


def rc_to_forward_alignment(
    rc_pos: int,
    rc_cigar: str,
    ref_length: int,
) -> Tuple[int, str]:
    """
    Express a reverse-complement-strand alignment in forward coordinates.

    BWA-MEM was run with both the read and the reference reverse-
    complemented; the returned hit ``(rc_pos, rc_cigar)`` indexes into
    the rc reference and walks the rc read.  This converts that result
    into an equivalent ``(pos, cigar)`` against the forward reference
    and forward read so the two strands can be rendered against the
    same sequences.

    The transformation is purely positional: CIGAR ops are reversed in
    order, and ``pos`` is shifted to the symmetric position in the
    forward reference.  No bases are read or rewritten; in particular,
    the same physical alignment is described, just expressed in
    forward-strand coordinates.

    Parameters
    ----------
    rc_pos : int
        1-based position in the rc reference where the rc-strand
        alignment starts.
    rc_cigar : str
        CIGAR string for the rc-strand alignment.
    ref_length : int
        Length of the (forward, equivalently rc) reference.

    Returns
    -------
    (int, str)
        Forward-strand 1-based position and forward-strand CIGAR.
    """
    ops = parse_cigar(rc_cigar)
    ref_consumed = sum(length for length, op in ops if op in "M=XDN")
    fwd_pos = ref_length - (rc_pos - 1) - ref_consumed + 1
    fwd_cigar = "".join(f"{length}{op}" for length, op in reversed(ops))
    return fwd_pos, fwd_cigar


def rc_cigar_to_forward(
    rc_pos: int,
    rc_cigar: str,
    ref_length: int,
) -> str:
    """
    Return the forward-strand CIGAR equivalent to an rc-strand alignment.

    Thin wrapper around :func:`rc_to_forward_alignment` that drops the
    position component, intended for side-by-side display of forward
    and mirror alignments in comparison tables (where only the CIGAR
    shape is shown).

    Parameters
    ----------
    rc_pos : int
        1-based position in the rc reference where the rc-strand
        alignment starts.
    rc_cigar : str
        CIGAR string for the rc-strand alignment.
    ref_length : int
        Length of the (forward, equivalently rc) reference.

    Returns
    -------
    str
        Forward-strand CIGAR — the same alignment, with op order
        reversed.
    """
    _, fwd_cigar = rc_to_forward_alignment(rc_pos, rc_cigar, ref_length)
    return fwd_cigar


def score_alignment(
    read: str,
    ref: str,
    pos_1based: int,
    cigar: str,
    *,
    score_matrix: Any,
    gap_open: float,
    gap_extend: float,
    alphabet_to_index: Any,
) -> float:
    """
    Score an alignment described by ``(pos_1based, cigar)`` against
    ``ref`` and ``read`` under the affine-gap scheme used elsewhere in
    this notebook (and by ``RefAligner``).

    This is a strict Needleman-Wunsch global score: every CIGAR op
    contributes, with no edge bonuses or position-dependent
    adjustments. It matches NW-flex's ``RefAligner.align_simple``
    score by construction on CIGARs without soft-clip ops (the
    standard ``free_X = free_Y = False`` configuration used by
    :func:`align_nwflex`); in semiglobal RefAligner mode, RefAligner
    emits ``S`` for its free edges and reports a score that does not
    charge them, while this function does — so the two diverge by the
    affine-gap cost of those ``S`` runs.  This function **does not**
    match BWA-MEM's reported SW ``AS:i:`` tag in general — BWA's AS
    is the local-extension maximum, which can land at an interior
    cell when the optimal global path dips through an indel.

    Per-base match/mismatch contributions come from ``score_matrix``
    (indexed via ``alphabet_to_index``).  A gap of length ``L`` costs
    ``gap_open + L * gap_extend`` (signs as supplied — typically both
    negative).  ``N`` (skipped reference) is free; ``S`` (soft-clip)
    is charged the same affine-gap penalty as a deletion of the same
    length — leading/terminal read bases that don't align should not
    score better than read bases that do align poorly; ``H`` and ``P``
    contribute nothing.

    Parameters
    ----------
    read : str
        Forward-strand read sequence the CIGAR was produced against.
    ref : str
        Reference sequence the CIGAR is anchored to.
    pos_1based : int
        1-based reference position of the first aligned read base
        (SAM ``POS``).
    cigar : str
        CIGAR string.

    Returns
    -------
    float
        Total alignment score under the supplied scoring scheme.
    """
    ops = parse_cigar(cigar)
    ref_pos = pos_1based - 1
    read_pos = 0
    score = 0.0
    for length, op in ops:
        if op in ("M", "=", "X"):
            for _ in range(length):
                score += score_matrix[
                    alphabet_to_index[ref[ref_pos]]
                ][alphabet_to_index[read[read_pos]]]
                ref_pos += 1
                read_pos += 1
        elif op == "I":
            score += gap_open + length * gap_extend
            read_pos += length
        elif op == "D":
            score += gap_open + length * gap_extend
            ref_pos += length
        elif op == "N":
            ref_pos += length
        elif op == "S":
            score += gap_open + length * gap_extend
            read_pos += length
        elif op in ("H", "P"):
            pass
        else:
            raise ValueError(f"unknown CIGAR op: {op!r}")
    return score


def bwa_truth_cigar(
    read,
    hap,
    locus: STRLocus,
) -> Tuple[int, str]:
    """
    Construct the natural truth alignment of ``read`` against the locus
    reference (suitable for BWA-MEM verdict comparison).

    The shape is ``L M  Δ·|R| {I,D}  (body_ref + Rf) M``: lflank matches,
    a single insertion or deletion at the lflank–zone boundary covering
    the haplotype/reference body-length difference, then body-and-rflank
    matches.  Mismatches (e.g. SNVs in the haplotype) sit inside the M
    runs and are accounted for when the CIGAR is scored against the
    actual sequences via :func:`score_alignment`.

    Parameters
    ----------
    read : Read
        Output of :func:`tile_reads`.
    hap : Haplotype
        Source haplotype (provides ``body_len``).
    locus : STRLocus
        Reference locus (provides flank length, motif, and ref ``N``).

    Returns
    -------
    (int, str)
        1-based start position and CIGAR.
    """
    L  = read.lflank_extent
    Rf = read.rflank_extent
    body_hap = hap.body_len
    body_ref = locus.k * locus.N
    pos = (locus.s - L) + 1
    if body_hap == body_ref:
        return pos, f"{L + body_hap + Rf}M"
    if body_hap > body_ref:
        return pos, f"{L}M{body_hap - body_ref}I{body_ref + Rf}M"
    return pos, f"{L}M{body_ref - body_hap}D{body_hap + Rf}M"


def nwflex_truth_cigar(
    read,
    hap,
    nwflex_locus: STRLocus,
) -> Tuple[int, str]:
    """
    Construct the natural truth alignment of ``read`` against the NW-flex
    extended (3N) reference: lflank matches, a free EP skip (``N`` op)
    covering the unused motifs, then (haplotype body + rflank) matches.

    The NW-flex reference is built with at least as many motifs as any
    haplotype in the sweep, so the skip is non-negative.

    Parameters
    ----------
    read : Read
    hap : Haplotype
    nwflex_locus : STRLocus
        Extended (e.g. 3N) reference locus.  Must satisfy
        ``nwflex_locus.k * nwflex_locus.N >= hap.body_len``.

    Returns
    -------
    (int, str)
        1-based start position and CIGAR.
    """
    L  = read.lflank_extent
    Rf = read.rflank_extent
    body_hap = hap.body_len
    body_ref = nwflex_locus.k * nwflex_locus.N
    skip = body_ref - body_hap
    if skip < 0:
        raise ValueError(
            f"hap body {body_hap} exceeds NW-flex ref body {body_ref}"
        )
    pos = (nwflex_locus.s - L) + 1
    if skip == 0:
        return pos, f"{L + body_hap + Rf}M"
    return pos, f"{L}M{skip}N{body_hap + Rf}M"


def alignment_state(
    cigar: Optional[str],
    pos_1based: Optional[int],
    chosen_score: Optional[float],
    truth_score: float,
    z_start: int,
    z_end: int,
    truth_z_bp: int,
    *,
    convention: str,
    min_flank: int = 1,
) -> str:
    """
    Classify a single alignment against the truth into one of four
    states.  Each state has a two-symbol code: the first symbol is the
    alignment outcome (``✓`` correct, ``✗`` wrong), the second is the
    chosen alignment's NW score relative to truth's NW score (``=``
    tied, ``<`` chosen below truth, ``>`` chosen above truth):

    - ``"P"`` (``✓ =``): the chosen CIGAR recovers the truth z-bp under
      :func:`is_arm_correct` — alignment correct, score trivially equal.
    - ``"T"`` (``✗ =``): alignment wrong, but ``chosen_score ==
      truth_score``.  The aligner *could* have picked truth; tie-break
      landed elsewhere.
    - ``"M"`` (``✗ <``): alignment wrong, ``chosen_score < truth_score``.
      The aligner's heuristic settled for less than truth.
    - ``"D"`` (``✗ >``): alignment wrong, ``chosen_score > truth_score``.
      The scoring landscape prefers a wrong alignment over truth.

    Unmapped reads (``cigar`` or ``pos_1based`` ``None``) classify as
    ``"D"``.
    """
    if cigar is None or pos_1based is None or chosen_score is None:
        return "D"
    if is_arm_correct(
        cigar, pos_1based, z_start, z_end, truth_z_bp,
        convention=convention, min_flank=min_flank,
    ):
        return "P"
    if chosen_score > truth_score:
        return "D"
    if chosen_score < truth_score:
        return "M"
    return "T"


_STATE_PRIORITY = {"P": 0, "T": 1, "M": 2, "D": 3}

_STATE_GLYPHS = {
    # length glyph (✓/✗)  +  score glyph (=, <, >) vs truth.
    "P": "✓  =",
    "T": "✗  =",
    "M": "✗  <",
    "D": "✗  >",
}


def state_to_glyph(state: str) -> str:
    """
    Format an :func:`alignment_state` classification as a two-glyph string.

    Each state has a length glyph (``✓`` if the alignment recovers the
    truth repeat length, ``✗`` otherwise) and a score glyph (``=`` for
    score equal to truth, ``<`` below truth, ``>`` above truth):

    - ``"P"`` → ``"✓  ="``
    - ``"T"`` → ``"✗  ="``
    - ``"M"`` → ``"✗  <"``
    - ``"D"`` → ``"✗  >"``
    """
    try:
        return _STATE_GLYPHS[state]
    except KeyError:
        raise ValueError(
            f"state must be one of P, T, M, D; got {state!r}"
        ) from None


def combine_states(state_a: str, state_b: str, policy: str = "best") -> str:
    """
    Combine two :func:`alignment_state` classifications into one.

    Priority order is ``P > T > M > D`` (lower priority number = better).

    - ``"best"`` returns whichever state is *better*: the cell counts as a
      pass when *either* strand passes.
    - ``"worst"`` returns whichever is *worse*: the cell counts as a pass
      only when *both* strands pass.
    """
    if policy not in ("best", "worst"):
        raise ValueError(f"policy must be 'best' or 'worst', got {policy!r}")
    a_pri = _STATE_PRIORITY[state_a]
    b_pri = _STATE_PRIORITY[state_b]
    if policy == "best":
        return state_a if a_pri <= b_pri else state_b
    return state_a if a_pri >= b_pri else state_b


# ===========================================================================
# Compound-repeat primitives
# ===========================================================================
#
# Two-block compound reference X = A · R1^N1 · M · R2^N2 · B and its
# extended counterpart with f·N1, f·N2 copies for NW-flex's EP pattern.
# Functions here mirror their single-repeat counterparts above
# (Locus / Haplotype / build_mirror_frame / is_arm_correct /
# bwa_truth_cigar / nwflex_truth_cigar / alignment_state).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompoundLocus:
    """Compound STR locus X = A · R1^N1 · M · R2^N2 · B with both the BWA
    reference (N1, N2) and the NW-flex extended reference (f·N1, f·N2)
    bundled in one object."""
    A: str
    R1: str
    N1: int
    M: str
    R2: str
    N2: int
    B: str
    nwflex_factor: int = 3

    @property
    def k1(self) -> int: return len(self.R1)
    @property
    def k2(self) -> int: return len(self.R2)
    @property
    def s1(self) -> int: return len(self.A)
    @property
    def e1(self) -> int: return self.s1 + self.k1 * self.N1
    @property
    def s2(self) -> int: return self.e1 + len(self.M)
    @property
    def e2(self) -> int: return self.s2 + self.k2 * self.N2
    @property
    def X(self) -> str:
        return (self.A + self.R1 * self.N1 + self.M
                + self.R2 * self.N2 + self.B)
    @property
    def zones(self) -> List[Tuple[int, int]]:
        return [(self.s1, self.e1), (self.s2, self.e2)]
    @property
    def X_ext(self) -> str:
        f = self.nwflex_factor
        return (self.A + self.R1 * (f * self.N1) + self.M
                + self.R2 * (f * self.N2) + self.B)
    @property
    def zones_ext(self) -> List[Tuple[int, int]]:
        f = self.nwflex_factor
        s1 = len(self.A)
        e1 = s1 + self.k1 * (f * self.N1)
        s2 = e1 + len(self.M)
        e2 = s2 + self.k2 * (f * self.N2)
        return [(s1, e1), (s2, e2)]


def build_compound_locus_from_panel(
    panel, *,
    motif1_len: int, motif2_len: int,
    ref_n1: int, ref_n2: int,
    bridge_n1: int, bridge_n2: int,
    flank_len: int,
    nwflex_factor: int = 3,
) -> CompoundLocus:
    """Pick two panel rows with distinct motif lengths; A and R1 come from
    row 1, B and R2 from row 2; M is composed from the interior edges of
    both panel flanks, each half cleaned against its adjacent motif."""
    row1 = panel[panel["type"].str.len() == motif1_len].iloc[0]
    row2 = panel[panel["type"].str.len() == motif2_len].iloc[0]
    R1, R2 = row1["type"], row2["type"]
    A = clean_flank_window(R1, row1["lflank"], flank_len, "left")
    B = clean_flank_window(R2, row2["rflank"], flank_len, "right")
    M_left = (clean_flank_window(R1, row1["rflank"], bridge_n1, "right")
              if bridge_n1 > 0 else "")
    M_right = (clean_flank_window(R2, row2["lflank"], bridge_n2, "left")
               if bridge_n2 > 0 else "")
    return CompoundLocus(
        A=A, R1=R1, N1=ref_n1,
        M=M_left + M_right,
        R2=R2, N2=ref_n2, B=B,
        nwflex_factor=nwflex_factor,
    )


@dataclass(frozen=True)
class CompoundHaplotype:
    """Compound haplotype, duck-typed for tile_reads via flank_len /
    body_len. The body spans both repeat blocks plus the bridge:
    body_len = k1·(N1+Δ1) + |M| + k2·(N2+Δ2)."""
    sequence: str
    flank_len: int
    body_len: int
    body_lens: Tuple[int, int]
    delta1: int
    delta2: int
    snv_pos: Optional[int] = None
    snv_base: Optional[str] = None


def build_compound_haplotype(
    compound: CompoundLocus, *, delta1: int, delta2: int,
    snv: Optional[Tuple[int, str]] = None,
) -> CompoundHaplotype:
    n1 = compound.N1 + delta1
    n2 = compound.N2 + delta2
    if n1 < 0 or n2 < 0:
        raise ValueError(
            f"compound haplotype counts must be >= 0: ({n1}, {n2})"
        )
    seq = (compound.A + compound.R1 * n1 + compound.M
           + compound.R2 * n2 + compound.B)
    body1 = compound.k1 * n1
    body2 = compound.k2 * n2
    body_len = body1 + len(compound.M) + body2
    snv_pos: Optional[int] = None
    snv_base: Optional[str] = None
    if snv is not None:
        abs_pos, base = snv
        if not (0 <= abs_pos < len(seq)):
            raise ValueError(
                f"SNV abs_pos {abs_pos} out of range [0, {len(seq)})"
            )
        if len(base) != 1:
            raise ValueError(f"SNV base must be a single character, got {base!r}")
        seq = seq[:abs_pos] + base + seq[abs_pos + 1:]
        snv_pos = abs_pos
        snv_base = base
    return CompoundHaplotype(
        sequence=seq, flank_len=len(compound.A), body_len=body_len,
        body_lens=(body1, body2), delta1=delta1, delta2=delta2,
        snv_pos=snv_pos, snv_base=snv_base,
    )


def build_compound_mirror_frame(compound: CompoundLocus, reads):
    """Compound analogue of build_mirror_frame: rc the BWA reference, the
    NW-flex extended reference, the reads, and both pairs of zones.
    Indexed zone-by-zone (``rc_zones[i]`` is the rc of forward
    ``zones[i]``) so a per-block truth comparison can keep the same
    ``i`` mapping in both frames."""
    rc_X = reverse_complement(compound.X)
    rc_X_ext = reverse_complement(compound.X_ext)
    rc_reads = mirror_reads(reads)
    n = len(compound.X)
    n_ext = len(compound.X_ext)
    rc_zones = [(n - e, n - s) for (s, e) in compound.zones]
    rc_zones_ext = [(n_ext - e, n_ext - s) for (s, e) in compound.zones_ext]
    return rc_X, rc_X_ext, rc_reads, rc_zones, rc_zones_ext


def is_arm_correct_multi(
    cigar, start_pos_1based, ref_zones, truth_z_bps, *,
    convention: str, min_flank: int = 1,
):
    """Return ``(per_block_correct, all_correct)``.

    Per-block check: ``decode_z_bp`` inside each zone must equal the
    corresponding truth bp. Outer span check: the alignment must consume
    at least ``min_flank`` reference bp outside both the leftmost and
    rightmost zones (so it actually spans both blocks).

    Outer endpoints are computed as ``min`` / ``max`` over zone starts
    / ends so the check is orientation-agnostic — fwd zones (listed in
    increasing ref-address order) and rc zones (where the rc reversal
    flips the address order while preserving logical ``[block1, block2]``
    listing) both resolve to the actual outer ref boundaries.
    """
    if cigar is None or start_pos_1based is None:
        return [False] * len(ref_zones), False
    per_block = []
    for (zs, ze), tbp in zip(ref_zones, truth_z_bps):
        zbp = decode_z_bp(cigar, start_pos_1based, zs, ze,
                          convention=convention)
        per_block.append(zbp == tbp)
    outer_s = min(z[0] for z in ref_zones)
    outer_e = max(z[1] for z in ref_zones)
    left, right = flank_bases_consumed(
        cigar, start_pos_1based, outer_s, outer_e,
    )
    spans = left >= min_flank and right >= min_flank
    return per_block, all(per_block) and spans


def _merge_m_ops(ops):
    """Coalesce adjacent M ops; drop zero-length ops."""
    merged = []
    for length, op in ops:
        if length == 0:
            continue
        if merged and merged[-1][1] == "M" and op == "M":
            merged[-1] = (merged[-1][0] + length, "M")
        else:
            merged.append((length, op))
    return merged


def bwa_compound_truth_cigar(read, hap, compound: CompoundLocus):
    """Compound BWA truth: gaps placed at the LEFT edge of each repeat
    block so the BWA boundary convention counts insertions as inside."""
    L = read.lflank_extent
    Rf = read.rflank_extent
    d1, d2 = hap.delta1, hap.delta2
    k1, k2 = compound.k1, compound.k2
    N1, N2 = compound.N1, compound.N2
    bridge = len(compound.M)
    pos = (compound.s1 - L) + 1
    ops = []
    if L > 0:
        ops.append((L, "M"))
    if d1 > 0:
        ops.append((d1 * k1, "I"))
        ops.append((N1 * k1, "M"))
    elif d1 < 0:
        ops.append((-d1 * k1, "D"))
        ops.append(((N1 + d1) * k1, "M"))
    else:
        ops.append((N1 * k1, "M"))
    if bridge > 0:
        ops.append((bridge, "M"))
    if d2 > 0:
        ops.append((d2 * k2, "I"))
        ops.append((N2 * k2, "M"))
    elif d2 < 0:
        ops.append((-d2 * k2, "D"))
        ops.append(((N2 + d2) * k2, "M"))
    else:
        ops.append((N2 * k2, "M"))
    if Rf > 0:
        ops.append((Rf, "M"))
    merged = _merge_m_ops(ops)
    cigar = "".join(f"{l}{o}" for l, o in merged)
    return pos, cigar


def nwflex_compound_truth_cigar(read, hap, compound: CompoundLocus):
    """Compound NW-flex truth against the extended (f·N1, f·N2)
    reference: free-skip past the unused motifs at the LEFT edge of each
    block (an N op), then walk the haplotype's block bases as M."""
    L = read.lflank_extent
    Rf = read.rflank_extent
    d1, d2 = hap.delta1, hap.delta2
    k1, k2 = compound.k1, compound.k2
    N1, N2 = compound.N1, compound.N2
    f = compound.nwflex_factor
    bridge = len(compound.M)
    pos = (compound.zones_ext[0][0] - L) + 1
    skip1 = (f * N1 - (N1 + d1)) * k1
    skip2 = (f * N2 - (N2 + d2)) * k2
    body1 = (N1 + d1) * k1
    body2 = (N2 + d2) * k2
    if skip1 < 0 or skip2 < 0:
        raise ValueError(
            f"hap counts exceed NW-flex extended ref: "
            f"({N1 + d1}, {N2 + d2}) vs ({f * N1}, {f * N2})"
        )
    ops = []
    if L > 0:      ops.append((L, "M"))
    if skip1 > 0:  ops.append((skip1, "N"))
    if body1 > 0:  ops.append((body1, "M"))
    if bridge > 0: ops.append((bridge, "M"))
    if skip2 > 0:  ops.append((skip2, "N"))
    if body2 > 0:  ops.append((body2, "M"))
    if Rf > 0:     ops.append((Rf, "M"))
    merged = _merge_m_ops(ops)
    cigar = "".join(f"{l}{o}" for l, o in merged)
    return pos, cigar


def alignment_state_multi(
    cigar, pos_1based, chosen_score, truth_score,
    ref_zones, truth_z_bps, *,
    convention: str, min_flank: int = 1,
) -> str:
    """Compound analogue of :func:`alignment_state`. Returns P / T / M / D
    using :func:`is_arm_correct_multi` for the length axis and the
    chosen-vs-truth score comparison for the score axis."""
    if cigar is None or pos_1based is None or chosen_score is None:
        return "D"
    _, ok = is_arm_correct_multi(
        cigar, pos_1based, ref_zones, truth_z_bps,
        convention=convention, min_flank=min_flank,
    )
    if ok:
        return "P"
    if chosen_score > truth_score:
        return "D"
    if chosen_score < truth_score:
        return "M"
    return "T"

