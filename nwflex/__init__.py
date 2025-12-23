"""
nwflex: Needleman-Wunsch Flex alignment package.
"""

# =============================================================================
# CORE ALIGNMENT
# =============================================================================

from .aligners import (
    align_with_EP,
    align_standard,
    align_semiglobal,
    align_single_block,
    align_STR_block,
    align_multi_STR,
)

from .dp_core import (
    AlignmentResult,
    FlexInput,
    FlexData,
    RowJump,
    run_flex_dp,
)

from .fast import run_flex_dp_fast, CYTHON_AVAILABLE

from .ep_patterns import (
    build_EP_standard,
    build_EP_single_block,
    build_EP_STR_phase,
    build_EP_multi_STR_phase,
    build_EP_semiglobal,
    union_EP_pair,
    union_EP_sequence,
)


# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

from .validation import (
    get_default_scoring,
    nwg_global,
    sflex_naive,
    check_standard_vs_nwg,
    check_single_block_case,
)


# =============================================================================
# STR REPEAT UTILITIES
# =============================================================================

from .repeats import (
    phase_repeat,
    valid_phase_combinations,
    count_valid_combinations,
    infer_abM_from_jumps,
    STRLocus,
    CompoundSTRLocus,
)


# =============================================================================
# BASIC NW (EDUCATIONAL)
# =============================================================================

from .nw_basics import (
    NWStep,
    GotohResult,
    needleman_wunsch_steps,
    traceback_simple,
    nw_gotoh_matrix,
)

from .nw_basics_plot import (
    NWDagPlotter,
    NWDagStyle,
    draw_nw_dag,
    visualize_cell_update,
)


# =============================================================================
# PLOTTING
# =============================================================================

from .plot import (
    draw_affine_k33_dag,
    draw_flex_gadget,
    draw_flex_dag,
    plot_flex_matrices,
    plot_gotoh_matrices,
    plot_dag_with_ep_arcs,
    plot_ep_pattern,
    plot_score_system,
)


__all__ = [
    # Core alignment
    "AlignmentResult",
    "align_with_EP",
    "align_standard",
    "align_semiglobal",
    "align_single_block",
    "align_STR_block",
    "align_multi_STR",
    # DP core
    "CYTHON_AVAILABLE",
    "FlexInput",
    "FlexData",
    "RowJump",
    "run_flex_dp",
    "run_flex_dp_fast",
    # EP patterns
    "build_EP_standard",
    "build_EP_single_block",
    "build_EP_STR_phase",
    "build_EP_multi_STR_phase",
    "build_EP_semiglobal",
    "union_EP_pair",
    "union_EP_sequence",
    # Validation
    "get_default_scoring",
    "nwg_global",
    "sflex_naive",
    "check_standard_vs_nwg",
    "check_single_block_case",
    # STR repeat utilities
    "phase_repeat",
    "valid_phase_combinations",
    "count_valid_combinations",
    "infer_abM_from_jumps",
    "STRLocus",
    "CompoundSTRLocus",
    # Basic NW (educational)
    "NWStep",
    "GotohResult",
    "needleman_wunsch_steps",
    "traceback_simple",
    "nw_gotoh_matrix",
    "NWDagPlotter",
    "NWDagStyle",
    "draw_nw_dag",
    "visualize_cell_update",
    # Plotting
    "draw_affine_k33_dag",
    "draw_flex_gadget",
    "draw_flex_dag",
    "plot_flex_matrices",
    "plot_gotoh_matrices",
    "plot_dag_with_ep_arcs",
    "plot_ep_pattern",
    "plot_score_system",
]
