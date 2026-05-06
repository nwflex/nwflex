"""
NW-flex simulation harness for notebook 07
(NW-flex vs BWA-MEM comparison).

Public API is re-exported here; users can import from
``nwflex.simulation`` directly, or reach into the submodules:

Submodules
----------
- ``simulation.core`` — locus and haplotype construction, read tiling,
  BWA-MEM wrappers, CIGAR decoding, per-arm correctness rule.
- ``simulation.viz`` — text-based alignment visualizations.

Example imports::

    from nwflex.simulation import build_haplotype, align_bwa
    from nwflex.simulation.viz import render_zoom
"""

from .core import (
    # Flank cleaning + locus construction
    clean_flank_window,
    build_locus_from_panel,
    # Haplotype + reads
    Haplotype,
    Read,
    build_haplotype,
    tile_reads,
    # BWA-MEM wrappers
    BwaResult,
    BwaBothStrandsResult,
    align_bwa,
    align_bwa_both_strands,
    reverse_complement,
    # CIGAR parsing + Z decoding
    parse_cigar,
    decode_z_bp,
    flank_bases_consumed,
    is_arm_correct,
    rc_to_forward_alignment,
    # Internal SAM parser (used by tests)
    _parse_sam_line,
)

from .viz import render_zoom


__all__ = [
    # Flanks + locus
    "clean_flank_window",
    "build_locus_from_panel",
    # Haplotype + reads
    "Haplotype",
    "Read",
    "build_haplotype",
    "tile_reads",
    # BWA wrappers
    "BwaResult",
    "BwaBothStrandsResult",
    "align_bwa",
    "align_bwa_both_strands",
    "reverse_complement",
    # CIGAR + Z
    "parse_cigar",
    "decode_z_bp",
    "flank_bases_consumed",
    "is_arm_correct",
    "rc_to_forward_alignment",
    # Viz
    "render_zoom",
]
