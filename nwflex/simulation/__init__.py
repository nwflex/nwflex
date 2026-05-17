"""
Simulation harness for the NW-flex vs BWA-MEM performance comparison.

Public API is re-exported here; users can import from
``nwflex.simulation`` directly, or reach into the submodules:

Submodules
----------
- ``simulation.core`` — locus and haplotype construction, read tiling,
  BWA-MEM and NW-flex wrappers, CIGAR decoding, per-arm correctness,
  mirror-frame strand handling, and scoring helpers.
- ``simulation.viz`` — ASCII and matplotlib alignment visualizations.
- ``simulation.sweep`` — parametric sweep harness with method classes
  for BWA-MEM and NW-flex (single and compound repeats).

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
    mirror_reads,
    build_mirror_frame,
    # NW-flex harness
    NwflexResult,
    align_nwflex,
    # CIGAR parsing + Z decoding
    parse_cigar,
    decode_z_bp,
    flank_bases_consumed,
    is_arm_correct,
    rc_to_forward_alignment,
    rc_cigar_to_forward,
    alignment_state,
    state_to_glyph,
    combine_states,
    score_alignment,
    bwa_truth_cigar,
    nwflex_truth_cigar,
    # Compound-repeat primitives
    CompoundLocus,
    build_compound_locus_from_panel,
    CompoundHaplotype,
    build_compound_haplotype,
    build_compound_mirror_frame,
    is_arm_correct_multi,
    bwa_compound_truth_cigar,
    nwflex_compound_truth_cigar,
    alignment_state_multi,
    # Internal SAM parser (used by tests)
    _parse_sam_line,
)

from .viz import (
    render_zoom,
    plot_correctness_heatmap,
    plot_correctness_heatmap_rows,
    plot_correctness_heatmap_2d,
    plot_correctness_heatmap_2d_rows,
    plot_proportion_heatmap_2d,
    plot_proportion_heatmap_2d_rows,
    plot_layout_schematic,
    plot_compound_layout_schematic,
    project_alignment_to_ref,
)

from .sweep import (
    SweepVariant,
    make_variant,
    sweep,
    pivot_for_heatmap,
    wrap_methods_for_multizone_truth,
    aggregate_per_cell,
    BWAMethod,
    NWFlexMethod,
    BWACompoundMethod,
    NWFlexCompoundMethod,
)


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
    "mirror_reads",
    "build_mirror_frame",
    "NwflexResult",
    "align_nwflex",
    # CIGAR + Z
    "parse_cigar",
    "decode_z_bp",
    "flank_bases_consumed",
    "is_arm_correct",
    "rc_to_forward_alignment",
    "rc_cigar_to_forward",
    "alignment_state",
    "state_to_glyph",
    "combine_states",
    "score_alignment",
    "bwa_truth_cigar",
    "nwflex_truth_cigar",
    # Compound-repeat primitives
    "CompoundLocus",
    "build_compound_locus_from_panel",
    "CompoundHaplotype",
    "build_compound_haplotype",
    "build_compound_mirror_frame",
    "is_arm_correct_multi",
    "bwa_compound_truth_cigar",
    "nwflex_compound_truth_cigar",
    "alignment_state_multi",
    # Viz
    "render_zoom",
    "plot_correctness_heatmap",
    "plot_correctness_heatmap_rows",
    "plot_correctness_heatmap_2d",
    "plot_correctness_heatmap_2d_rows",
    "plot_proportion_heatmap_2d",
    "plot_proportion_heatmap_2d_rows",
    "plot_layout_schematic",
    "plot_compound_layout_schematic",
    "project_alignment_to_ref",
    # Sweep harness
    "SweepVariant",
    "make_variant",
    "sweep",
    "pivot_for_heatmap",
    "wrap_methods_for_multizone_truth",
    "aggregate_per_cell",
    "BWAMethod",
    "NWFlexMethod",
    "BWACompoundMethod",
    "NWFlexCompoundMethod",
]
