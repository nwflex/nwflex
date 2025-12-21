"""
Color constants for NW-flex plotting.

This module defines all color schemes used across the plotting library.
"""

# =============================================================================
# NUCLEOTIDE COLORS
# =============================================================================

# Nucleotide colors (slightly lighter + more plain)
NT_COLOR = {
    "A": "#74AB86",  # soft green
    "C": "#6E93C0",  # soft blue
    "G": "#C19A5A",  # soft warm ochre
    "T": "#C26F6F",  # soft red
    "": "#000000",
}


# =============================================================================
# PORT/LAYER COLORS (for K_{3,3} gadgets)
# =============================================================================

PORT_COLORS = dict(
    H="#9BBF4C",  # brighter yellow-green (Yg layer)
    D="#4285C7",  # bright steel/cornflower blue (M layer)
    V="#A058C7",  # bright violet (Xg layer)
)


# =============================================================================
# EDGE COLORS
# =============================================================================

# Edge colors for affine-gap DAG
EDGE_COLORS = dict(
    match="#16946C",        # vivid teal green
    mismatch="#D45500",     # vivid orange-brown
    gap_extend="#396CB4",   # vivid indigo-blue
    gap_initiate="#E5A000", # vivid golden orange
    zero="#7A7A7A",         # neutral gray
)

# Internal DAG edges (muted but not pale)
DIAG_EDGE_COLORS = dict(
    diag_match="#4C9D77",     # mid muted teal
    diag_mismatch="#C66B2C",  # mid muted orange-brown
    diag_zero="#B0B0B0",      # light gray
    gap_initiate="#D3983A",   # mid muted gold
    extend="#5874B8",         # mid muted indigo
    gap="#5874B8",            # alias for extend
    zero="#B0B0B0",           # light gray
)


# =============================================================================
# EP (EXTRA PREDECESSOR) COLORS
# =============================================================================

# EP edges (quite prominent)
EP_COLORS = dict(
    leader="#1B5DAA",   # strong blue
    closer="#7533B5",   # strong purple
)


# =============================================================================
# REGION COLORS (for A·Z·B decomposition)
# =============================================================================

# Region background fills (slightly stronger contrast)
REGION_FILL_COLORS = dict(
    A="#C3D8F3",  # light cool blue
    Z="#F3D1AF",  # light peach
    B="#D6C4F0",  # light lavender
)

# Region label colors (to match stronger edges)
REGION_LABEL_COLORS = dict(
    A="#19467F",  # dark blue
    Z="#9C440F",  # dark warm orange-brown
    B="#6533A5",  # dark purple
)

# Combined region colors (for draw_matrix_regions)
REGION_COLORS = {
    'A': {'fill': REGION_FILL_COLORS['A'], 'label': REGION_LABEL_COLORS['A']},
    'Z': {'fill': REGION_FILL_COLORS['Z'], 'label': REGION_LABEL_COLORS['Z']},
    'Z*': {'fill': REGION_FILL_COLORS['Z'], 'label': REGION_LABEL_COLORS['Z']},
    'B': {'fill': REGION_FILL_COLORS['B'], 'label': REGION_LABEL_COLORS['B']},
}


# =============================================================================
# DAG PATH COLORS
# =============================================================================

DAG_PATH_COLORS = dict(
    match="#2CA02C",     # green
    mismatch="#C53030",  # red
    gap="#7F7F7F",       # gray
)

# =============================================================================
# HEATMAP COLORMAPS
# =============================================================================
HEATMAP_COLORMAPS = {
    'default': 'Reds',
    'diverging': 'RdBu_r',
}
