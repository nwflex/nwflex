"""
NW-flex plotting package.

This package provides visualization functions for the NW-flex algorithm.
Users can import from nwflex.plot as before, or access submodules directly.

Submodules:
    - plot.colors: Color constants for all visualization
    - plot.utils: Shared utility functions
    - plot.scoring: Scoring system visualization
    - plot.ep: EP pattern visualization
    - plot.dag: DAG visualization (K₃,₃ gadgets, flex gadgets, basic NW DAG)
    - plot.matrix: DP matrix heatmaps

Example imports:
    from nwflex.plot import plot_flex_matrices  # top-level re-export
    from nwflex.plot.matrix import plot_flex_matrices  # direct submodule
    from nwflex.plot.colors import NT_COLOR  # color constants
"""

# =============================================================================
# COLOR CONSTANTS (from colors)
# =============================================================================

from .colors import (
    # Nucleotide colors
    NT_COLOR,
    # Port/layer colors
    PORT_COLORS,
    # Edge colors for affine-gap DAG
    EDGE_COLORS,
    # Internal DAG edges (muted)
    DIAG_EDGE_COLORS,
    # EP edges
    EP_COLORS,
    # Region fills and labels
    REGION_FILL_COLORS,
    REGION_LABEL_COLORS,
    REGION_COLORS,  # combined dict
    # DAG path colors
    DAG_PATH_COLORS,
)


# =============================================================================
# UTILITY FUNCTIONS (from utils)
# =============================================================================

from .utils import (
    polar,
    draw_shortened_arrow,
    draw_highlighted_edge,
    draw_node_with_score,
)


# =============================================================================
# SCORING VISUALIZATION (from scoring)
# =============================================================================

from .scoring import plot_score_system


# =============================================================================
# EP PATTERN VISUALIZATION (from ep)
# =============================================================================

from .ep import (
    draw_ep_background,
    draw_matrix_regions,
    build_AZB_regions,
    plot_ep_pattern,
    summarize_ep_edges,
    plot_ep_comparison,
    print_ep_comparison_summary,
)


# =============================================================================
# DAG VISUALIZATION (from dag)
# =============================================================================

from .dag import (
    # Combined EP + DAG views
    plot_dag_with_ep_arcs,
    # K₃,₃ gadgets and flex DAG
    draw_affine_k33_dag,
    draw_flex_gadget,
    draw_flex_dag,
)


# =============================================================================
# DP MATRIX HEATMAPS (from matrix)
# =============================================================================

from .matrix import (
    plot_flex_matrices,
    plot_gotoh_matrices,
    visualize_alignment_matrices,
    # Publication-quality plotter
    plot_flex_matrices_publication,
    # Figure-level helpers
    draw_figure_region_backgrounds,
    draw_figure_leader_closer_boxes,
    draw_figure_row_highlights,
)

# Path utilities (from utils)
from .utils import (
    path_to_alignment,
    score_path,
)


# =============================================================================
# MODULE-LEVEL __all__ for explicit API
# =============================================================================

__all__ = [
    # Colors
    "NT_COLOR",
    "PORT_COLORS",
    "EDGE_COLORS",
    "DIAG_EDGE_COLORS",
    "EP_COLORS",
    "REGION_FILL_COLORS",
    "REGION_LABEL_COLORS",
    "REGION_COLORS",
    "DAG_PATH_COLORS",
    # Utils
    "polar",
    "draw_shortened_arrow",
    "draw_highlighted_edge",
    "draw_node_with_score",
    # Scoring
    "plot_score_system",
    # EP
    "draw_ep_background",
    "draw_matrix_regions",
    "build_AZB_regions",
    "plot_ep_pattern",
    "summarize_ep_edges",
    "plot_ep_comparison",
    "print_ep_comparison_summary",
    # DAG
    "plot_dag_with_ep_arcs",
    "draw_affine_k33_dag",
    "draw_flex_gadget",
    "draw_flex_dag",
    # Matrix
    "plot_flex_matrices",
    "plot_gotoh_matrices",
    "visualize_alignment_matrices",
    "plot_flex_matrices_publication",
    "draw_figure_region_backgrounds",
    "draw_figure_leader_closer_boxes",
    "draw_figure_row_highlights",
    "path_to_alignment",
    "score_path",
]
