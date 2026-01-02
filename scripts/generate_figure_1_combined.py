#!/usr/bin/env python3
"""
generate_figure_1_combined.py — Main-text Figure 1 (2-row layout)

Creates the combined NW-flex figure with:
  - Row 1: Panel A (DAG with EP arcs) + Panel C (S_flex vs N scalar plateau)
  - Row 2: Panel B (three DP matrices: Yg, M, Xg)

Usage:
    python scripts/generate_figure_1_combined.py [--output PATH] [--dpi DPI] [--format FORMAT]

Output:
    tex/figure1_3panel.pdf (default)
"""

import argparse
import sys
from pathlib import Path

# Use Agg backend by default for headless generation
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib as mpl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from nwflex package
from nwflex.ep_patterns import build_EP_single_block
from nwflex.aligners import align_single_block, align_with_EP
from nwflex.default import get_default_scoring
from nwflex.repeats import phase_repeat
from nwflex.plot import (
    plot_dag_with_ep_arcs,
    plot_flex_matrices_publication,
    # Import default colors
    NT_COLOR,
    PORT_COLORS,
    EDGE_COLORS,
    EP_COLORS,
    REGION_FILL_COLORS,
    REGION_LABEL_COLORS,
    DAG_PATH_COLORS,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Fonts for main panels
FONTS = {
    'panel_label': {'size': 26, 'weight': 'bold'},
    'axis_label': {'size': 12},
    'tick': {'size': 10},
    'legend': {'size': 10},
}

# Fonts for Panel C (scaled up from original 8/6/8/9 to match other panels)
FONTS_C = {
    'axis_label': 12,
    'tick': 10,
    'legend': 10,
    'title': 12,
}

PLATEAU_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

FIGURE = {
    'width': 15.0,    # Full page width
    'height': 15.0,   # Taller for 2-row layout
    'dpi': 300,
    'format': 'pdf',
    'background': '#ffffff',
}

# =============================================================================
# UNIFIED VISUAL STYLE
# =============================================================================

STYLE = {
    # Panel A: Full DAG overview
    'panel_a': {
        'spacing': 1.0,
        'node_radius': 0.2,
        'dag_linewidth': 1.0,        # Thinner for overview
        'dag_mutation_scale': 6.0,   # Smaller arrowheads
        'ep_linewidth': 1.5,
        'ep_mutation_scale': 8.0,
        'sequence_fontsize': 12.0,   # Smaller for overview
        'region_fontsize': 14.0,
        'box_label_fontsize': 9.0,
        'legend_fontsize': 8.0,
    },
    # Panel B: DP matrix view
    'panel_b': {
        'marker_size': 16,
        'marker_color': "#02CA42",   # Accessible green for path too
        'marker_width': 3,         # Thicker edge for accessibility
        'jump_color': "#02CA42",     # Accessible green for jumps
        'jump_width': 3.5,           # Thicker for accessibility
        'region_linewidth': 3.0,     # Thicker region boxes for accessibility
        'region_label_fontsize': 14.0,
        # Region background margins (figure coordinates)
        'region_margin_left': 0.015,   # Extend left for region backgrounds
        'region_margin_right': 0.035,  # Extend right for region backgrounds
        'region_label_offset': 0.005,  # Distance from region left edge to label
        # Leader/closer box margins (figure coordinates) - separate from region
        'leader_closer_margin_left': 0.001,   # Left margin for leader/closer boxes
        'leader_closer_margin_right': 0.001,  # Right margin for leader/closer boxes
        # Subplot spacing
        'wgap': 0.08,                 # Gap between the 3 matrix subplots
        # Colormap for heatmaps
        'colormap': 'Reds',           # Diverging colormap (try: RdBu_r, BrBG, coolwarm)
        # Nucleotide label positions and size
        'nuc_label_x_offset': -0.04,  # Y-axis labels: distance left of axis
        'nuc_label_y_offset': 0, # X-axis labels: distance above axis (negative = above)
        'nuc_label_fontsize': 12,      # Font size for nucleotide labels
        # Title positioning
        'title_pad': 20,              # Padding between title and top of plot
        'title_fontsize': 16,         # Font size for subplot titles
        # Leader/closer label styling
        'leader_closer_fontsize': 9,  # Font size for leader/closer labels
        # Opacity controls
        'path_box_alpha': 1.0,        # Background path boxes on all panels
        'path_box_highlight_alpha': 1.0,  # Highlighted path box on active panel
        'leader_closer_box_alpha': 1.0,   # Leader/closer FancyBboxPatch
        'region_fill_alpha': 0.35,    # Opacity for region background fills
        'annot_fontsize': 9.0,          # Font size for cell annotations
    },
}


# =============================================================================
# PANEL C: S_flex vs N SCALAR PLATEAU
# =============================================================================

def draw_panel_C_scalar(ax):
    """Draw the scalar plateau plot: S_flex(X_N, Y) vs N."""
    score_matrix, gap_open, gap_extend, a2i = get_default_scoring()
    
    A, R, B = "GAG", "ACC", "GTCA"
    k = len(R)
    
    # Labels describing partial repeat content
    test_cases = [
        (0, 0, 3, r'$(\mathrm{ACC})_3$', 3),        # no partials
        (1, 0, 2, r'$\mathrm{C}(\mathrm{ACC})_2$', 3),       # left partial
        (0, 2, 2, r'$(\mathrm{ACC})_2\mathrm{AC}$', 3),     # right partial
        (2, 1, 2, r'$\mathrm{C}(\mathrm{ACC})_2\mathrm{AC}$', 4),     # both partials
    ]
    
    N_values = list(range(1, 6))
    markers = ['o', 's', '^', 'D']
    
    for idx, (a, b, M, label, repeat_content) in enumerate(test_cases):
        Y = A + phase_repeat(R, a, b, M) + B
        scores = []
        
        for N in N_values:
            X = A + R * N + B
            s = len(A)
            e = s + N * k
            n = len(X)
            EP = build_EP_single_block(n, s, e)
            result = align_with_EP(X, Y, score_matrix, gap_open, gap_extend, a2i, EP)
            scores.append(result.score)
        
        ax.plot(N_values, scores, color=PLATEAU_COLORS[idx], marker=markers[idx],
                markersize=6, linewidth=2, label=label)
    
    ax.set_xlabel('$N$', fontsize=FONTS_C['axis_label'])
    ax.set_ylabel('Score', fontsize=FONTS_C['axis_label'])
    ax.set_title('$S_{flex}(N)$ vs $N$', fontsize=FONTS_C['title'])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=FONTS_C['legend'], frameon=True,
              columnspacing=1.0, handletextpad=0.5)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONTS_C['tick'])


# =============================================================================
# MAIN FIGURE GENERATION
# =============================================================================

def generate_figure1_combined(output_path=None, dpi=None, fmt=None):
    """Generate the complete Figure 1 (2-row layout)."""
    if output_path is None:
        output_path = Path(__file__).parent.parent / 'tex' / 'figure1_3panel.pdf'
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dpi is None:
        dpi = FIGURE['dpi']
    if fmt is None:
        fmt = output_path.suffix[1:] if output_path.suffix else FIGURE['format']
    
    # Set fonttype for editable text in Illustrator
    mpl.rcParams['pdf.fonttype'] = 42
    
    # Example sequences shared by the DAG and matrix views (Panels A and B)
    A = "G"
    Z = "ACCACC"
    B = "TG"
    X = A + Z + B
    Y = A + Z[2:-1] + B
    
    s = len(A)           # leader row (last row of A)
    e = len(A) + len(Z)  # end of Z; closer row is e+1
    n = len(X)
    
    # Build EP pattern
    EP = build_EP_single_block(n, s, e)
    leaders = [s]  # For single block, leader is row s
    
    # Run alignment to get path and jumps
    score_matrix, gap_open, gap_extend, a2i = get_default_scoring()
    result = align_single_block(
        X=X, Y=Y, s=s, e=e,
        score_matrix=score_matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
        alphabet_to_index=a2i,
        return_data=True,
    )
    
    # =========================================================================
    # Setup figure and axes layout (2-row)
    # =========================================================================
    fig = plt.figure(figsize=(FIGURE['width'], FIGURE['height']))
    
    # Layout: 2 rows
    # Row 1: Panel A (DAG) + Panel C (S_flex vs N)
    # Row 2: Panel B (three DP matrices, spanning both columns)
    gs = gridspec.GridSpec(2, 2,
                           height_ratios=[1.2, 1.0],
                           width_ratios=[2.0, 1.0],
                           hspace=0.25,
                           wspace=0.15)
    
    # Get Panel A and B style config
    sty_a = STYLE['panel_a']    
    sty_b = STYLE['panel_b']
    
    # Row 1: Panel A (left) and Panel C (right)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_C = fig.add_subplot(gs[0, 1])
    
    # Make Panel C shorter while keeping Panel A the same height
    # Shrink from the bottom by adjusting the axes position
    pos_C = ax_C.get_position()
    shrink_factor = 0.6  # Panel C will be 60% of its original height
    new_height = pos_C.height * shrink_factor
    # Keep top aligned, shrink from bottom
    new_y0 = pos_C.y0 + (pos_C.height - (1.25* new_height))
    ax_C.set_position([pos_C.x0, new_y0, pos_C.width, new_height])
    
    # Row 2: Panel B spans both columns, with nested GridSpec for 3 matrices
    # Add left/right margins to Panel B
    panel_b_margin_left = 0.1  # fraction of figure width
    panel_b_margin_right = 0.05  # fraction of figure width
    gs_B = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :],
                                             wspace=sty_b['wgap'] )
    
    ax_B1 = fig.add_subplot(gs_B[0])  # Yg
    ax_B2 = fig.add_subplot(gs_B[1])  # M
    ax_B3 = fig.add_subplot(gs_B[2])  # Xg
    
    # Adjust Panel B axes to add left/right margins
    for ax_B in [ax_B1, ax_B2, ax_B3]:
        pos = ax_B.get_position()
        # Shrink width and shift to account for margins
        total_margin = panel_b_margin_left + panel_b_margin_right
        scale = 1.0 - total_margin
        new_x0 = panel_b_margin_left + (pos.x0 * scale)
        new_width = pos.width * scale
        ax_B.set_position([new_x0, pos.y0, new_width, pos.height])
    
    # =========================================================================
    # Panel A: DAG view with EP arcs and region highlighting
    # =========================================================================
    plot_dag_with_ep_arcs(
        EP=EP,
        leaders=leaders,
        X=X,
        Y=Y,
        s=s,
        e=e,
        ax=ax_A,
        # Styling from unified config
        spacing=sty_a['spacing'],
        node_radius=sty_a['node_radius'],
        dag_linewidth=sty_a['dag_linewidth'],
        dag_mutation_scale=sty_a['dag_mutation_scale'],
        ep_linewidth=sty_a['ep_linewidth'],
        ep_mutation_scale=sty_a['ep_mutation_scale'],
        sequence_fontsize=sty_a['sequence_fontsize'],
        region_fontsize=sty_a['region_fontsize'],
        box_label_fontsize=sty_a['box_label_fontsize'],
        legend_fontsize=sty_a['legend_fontsize'],
        # Suppress legend and title
        show_legend=False,
        title=None,
        # Use imported default colors from nwflex.plot
        match_color=DAG_PATH_COLORS['match'],
        mismatch_color=DAG_PATH_COLORS['mismatch'],
        gap_color=DAG_PATH_COLORS['gap'],
        # Region colors
        color_A=REGION_FILL_COLORS['A'],
        color_Z=REGION_FILL_COLORS['Z'],
        color_B=REGION_FILL_COLORS['B'],
        label_A=REGION_LABEL_COLORS['A'],
        label_Z=REGION_LABEL_COLORS['Z'],
        label_B=REGION_LABEL_COLORS['B'],
        # EP edge colors
        leader_edge_color=EP_COLORS['leader'],
        closer_edge_color=EP_COLORS['closer'],
        box_linewidth=2.0,
    )
    
    # Panel label for DAG view
    ax_A.text(-0.07, 0.97, 'A', transform=ax_A.transAxes,
              fontsize=FONTS['panel_label']['size'],
              fontweight=FONTS['panel_label']['weight'],
              va='top', ha='left')
    
    # Legend for DAG view (below the panel)
    legend_handles = [
        # --- Col 1: match/leader types ---
        Line2D([0], [0], color=EDGE_COLORS['match'], lw=2, marker='>',
               markersize=5, label='match'),
        Line2D([0], [0], color=EP_COLORS['leader'], lw=2, marker='>',
               markersize=5, label='EP leader'),
        # --- Col 2: mismatch/closer ---        
        Line2D([0], [0], color=EDGE_COLORS['mismatch'], lw=2, marker='>',
               markersize=5, label='mismatch'),        
        Line2D([0], [0], color=EP_COLORS['closer'], lw=2, marker='>',
               markersize=5, label='EP closer'),
        # --- Col 3: gap ---
        Line2D([0], [0], color='#7F7F7F', lw=2, marker='>',
               markersize=5, label='gap'),  # Gray for gap
        
    ]
    
    ax_A.legend(handles=legend_handles, loc='lower center',
                bbox_to_anchor=(0.5, -0.12),
                ncol=3,  # 3 columns forces row 1: 3 items, row 2: 2 items
                fontsize=9, frameon=True, fancybox=True,
                framealpha=0.95, edgecolor='#cccccc',
                labelspacing=0.6, borderpad=0.8, handletextpad=0.6,
                columnspacing=1.5)
    
    # =========================================================================
    # Panel C: S_flex vs N scalar plateau
    # =========================================================================
    draw_panel_C_scalar(ax_C)
    
    # Panel label for scalar plateau
    ax_C.text(-0.25, 1.2, 'C', transform=ax_C.transAxes,
              fontsize=FONTS['panel_label']['size'],
              fontweight=FONTS['panel_label']['weight'],
              va='top', ha='left')
    
    # =========================================================================
    # Panel B: Matrix view - DP matrices with path, jumps, and region backgrounds
    # =========================================================================
    # Panel label for matrix view
    ax_B1.text(-0.22, 1.1, 'B', transform=ax_B1.transAxes,
               fontsize=FONTS['panel_label']['size'],
               fontweight=FONTS['panel_label']['weight'],
               va='top', ha='right')    
    
    # Build A·Z·B regions for background fills (matching the DAG coloring)
    regions = [
        {
            'start': 1,
            'end': s,
            'label': 'A',
            'fill_color': REGION_FILL_COLORS['A'],
            'label_color': REGION_LABEL_COLORS['A'],
        },
        {
            'start': s + 1,
            'end': e,
            'label': 'Z',
            'fill_color': REGION_FILL_COLORS['Z'],
            'label_color': REGION_LABEL_COLORS['Z'],
        },
        {
            'start': e + 1,
            'end': n,
            'label': 'B',
            'fill_color': REGION_FILL_COLORS['B'],
            'label_color': REGION_LABEL_COLORS['B'],
        },
    ] if s >= 1 else []
    
    # Build Z* row highlight from the alignment jumps
    row_highlights = []
    if len(result.jumps) >= 2:
        entry_jump = result.jumps[0]  # entry into Z
        exit_jump = result.jumps[1]   # exit from Z
        z_star_start = entry_jump.to_row
        z_star_end = exit_jump.from_row
        
        row_highlights = [
            {
                'start': z_star_start,
                'end': z_star_end,
                'label': 'Z*',
                'color': "#D5B812",  # vivid golden orange
                'linestyle': '--',
                'label_position': 'middle',
                'fontsize': 14,  # Larger font for visibility
            },
        ]
    
    # Plot the matrices using the publication-quality plotter from nwflex.plot
    plot_flex_matrices_publication(
        fig=fig,
        axes=[ax_B1, ax_B2, ax_B3],
        result=result,
        X=X,
        Y=Y,
        s=s,
        e=e,
        regions=regions,
        row_highlights=row_highlights,
        # Heatmap styling
        colormap=sty_b['colormap'],
        # Path styling
        marker_color=sty_b['marker_color'],
        marker_width=sty_b['marker_width'],
        path_box_alpha=sty_b['path_box_alpha'],
        path_box_highlight_alpha=sty_b['path_box_highlight_alpha'],
        # Jump styling
        jump_color=sty_b['jump_color'],
        jump_width=sty_b['jump_width'],
        # Nucleotide labels
        nuc_label_x_offset=sty_b['nuc_label_x_offset'],
        nuc_label_y_offset=sty_b['nuc_label_y_offset'],
        nuc_label_fontsize=sty_b['nuc_label_fontsize'],
        # Title styling
        title_pad=sty_b['title_pad'],
        title_fontsize=sty_b['title_fontsize'],
        # Region backgrounds (figure-level)
        region_margin_left=sty_b['region_margin_left'],
        region_margin_right=sty_b['region_margin_right'],
        region_label_offset=sty_b['region_label_offset'],
        region_label_fontsize=sty_b['region_label_fontsize'],
        region_fill_alpha=sty_b['region_fill_alpha'],
        # Leader/closer boxes (figure-level)
        leader_closer_margin_left=sty_b['leader_closer_margin_left'],
        leader_closer_margin_right=sty_b['leader_closer_margin_right'],
        leader_closer_fontsize=sty_b['leader_closer_fontsize'],
        leader_closer_box_alpha=sty_b['leader_closer_box_alpha'],
        path_zorder=200,
        jump_zorder=190,
        path_color_bg=sty_b['jump_color'],
        path_width_factor_bg=1.,
        annot_fontsize=sty_b['annot_fontsize'],
    )
    

    # Save
    plt.savefig(output_path, dpi=dpi, format=fmt, bbox_inches='tight',
                facecolor=FIGURE['background'])
    print(f"Saved figure to: {output_path}")
    
    return fig


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Figure 1 for NW-flex paper (2-row layout)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: tex/figure1_3panel.pdf)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Resolution in DPI (default: 300)')
    parser.add_argument('--format', '-f', type=str, default=None,
                        choices=['pdf', 'png', 'svg', 'eps'],
                        help='Output format (default: pdf)')
    parser.add_argument('--show', action='store_true',
                        help='Display figure interactively')
    
    args = parser.parse_args()
    
    fig = generate_figure1_combined(output_path=args.output, dpi=args.dpi, fmt=args.format)
    
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
