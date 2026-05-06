"""
simulation.py — Simulation harness for notebook 07
(NW-flex vs BWA-MEM comparison).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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
