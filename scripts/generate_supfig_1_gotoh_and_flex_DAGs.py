#!/usr/bin/env python3
"""
generate_supfig_1_gotoh_and_flex_DAGs.py — Supplementary Figure 1 (Gotoh vs NW-flex DAGs)

This script produces the supplementary figure comparing the Gotoh DAG and NW-flex DAG
(cropped 3×3) as a single output file for the gotoh_and_flex_DAGs panel.

Output (default):
  tex/gotoh_and_flex_DAGs.pdf
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
from matplotlib.patches import Rectangle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nwflex.plot import (
    draw_affine_k33_dag,
    draw_flex_dag,
    PORT_COLORS,
    EDGE_COLORS,
    EP_COLORS,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

FONTS = {
    'panel_label': {'size': 20, 'weight': 'bold'},
}

FIGURE = {
    'panel_B': {'width': 14, 'height': 5.0},
    'dpi': 300,
    'format': 'pdf',
    'background': '#ffffff',
}

STYLE = {
    'panel_b': {
        'spacing': 1.0,
        'gadget_radius': 0.425,
        'inner_radius_scale': 0.75,
        'edge_linewidth': 1.5,
        'internal_linewidth': 1.0,
        'mutation_scale': 6.0,
        'circle_linewidth': 1.5,
        'circle_alpha': 0.25,
        'edge_alpha': 1.0,
        'internal_alpha': 1.0,
        'port_marker_size': 6.0,
        'marker_edge_width': 1.2,
        'letter_fs': 14,
        'title_fs': 12,
        'legend_fs': 9,
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def add_fade_edges(ax, fade_width=0.15, color='white'):
    """Add fade-out gradient rectangles at right and bottom edges of axes."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = abs(ylim[1] - ylim[0])
    
    fade_x = x_range * fade_width
    fade_y = y_range * fade_width
    
    gradient_h = np.linspace(0, 1, 100).reshape(1, -1)
    right_extent = [xlim[1] - fade_x, xlim[1], min(ylim), max(ylim)]
    ax.imshow(gradient_h, extent=right_extent, aspect='auto',
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list('fade_r', [(1,1,1,0), (1,1,1,1)]),
              zorder=100, interpolation='bilinear')
    
    is_inverted = ylim[0] > ylim[1]
    if is_inverted:
        bottom_extent = [xlim[0], xlim[1], ylim[0] - fade_y, ylim[0]]
        gradient_v = np.linspace(1, 0, 100).reshape(-1, 1)
    else:
        bottom_extent = [xlim[0], xlim[1], ylim[0], ylim[0] + fade_y]
        gradient_v = np.linspace(1, 0, 100).reshape(-1, 1)
    
    ax.imshow(gradient_v, extent=bottom_extent, aspect='auto',
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list('fade_b', [(1,1,1,0), (1,1,1,1)]),
              zorder=100, interpolation='bilinear')


def get_panel_sequences():
    """Sequences used for the gadget comparison."""
    A = "G"
    Z = "ACCACC"
    B = "TG"
    X = A + Z + B
    Y = A + Z[2:-1] + B
    return X[:5], Y[:5]


# =============================================================================
# PANEL GENERATOR
# =============================================================================
max_nuc = 3
def generate_panel_B(output_path, dpi=None, fmt=None):
    """Generate supplementary figure 1 DAG comparison output."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dpi is None:
        dpi = FIGURE['dpi']
    if fmt is None:
        fmt = output_path.suffix[1:] if output_path.suffix else FIGURE['format']
    
    fig = plt.figure(figsize=(FIGURE['panel_B']['width'], FIGURE['panel_B']['height']))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.5], wspace=0.2)
    
    X_clip, Y_clip = get_panel_sequences()
    ax_B1 = fig.add_subplot(gs[0, 0])
    ax_B2 = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[0, 2])
    ax_legend.axis('off')
    
    # Shrink both subplots to 95% of their size
    for ax_b in [ax_B1, ax_B2]:
        pos = ax_b.get_position()
        new_width = pos.width * 0.95
        new_height = pos.height * 0.95
        new_x0 = pos.x0 + (pos.width - new_width) / 2
        new_y0 = pos.y0 + (pos.height - new_height) / 2
        ax_b.set_position([new_x0, new_y0, new_width, new_height])
    
    sty_b = STYLE['panel_b']
    spacing_b = sty_b['spacing']
    gadget_radius = sty_b['gadget_radius']
    margin = 0.7
    clip_limit = 1.75 * spacing_b + (1.5 * margin)
    
    draw_affine_k33_dag(
        seq1=X_clip,
        seq2=Y_clip,
        match=5,
        mismatch=-5,
        spacing=spacing_b,
        gadget_radius=gadget_radius,
        inner_radius_scale=sty_b['inner_radius_scale'],
        port_marker_size=sty_b['port_marker_size'],
        marker_edge_width=sty_b['marker_edge_width'],
        circle_alpha=sty_b['circle_alpha'],
        circle_linewidth=sty_b['circle_linewidth'],
        edge_linewidth=sty_b['edge_linewidth'],
        edge_alpha=sty_b['edge_alpha'],
        internal_linewidth=sty_b['internal_linewidth'],
        internal_alpha=sty_b['internal_alpha'],
        mutation_scale=sty_b['mutation_scale'],
        letter_fs=sty_b['letter_fs'],
        legend_fs=sty_b['legend_fs'],
        title="",
        show_legend=False,
        ax=ax_B1,
        max_nucleotide_x=max_nuc,
        max_nucleotide_y=max_nuc,
    )
    
    ax_B1.set_xlim(-margin, clip_limit)
    ax_B1.set_ylim(clip_limit, -margin)
    add_fade_edges(ax_B1, fade_width=0.12)
    border = Rectangle((-0.10, 0.0), 1.10, 1.05, transform=ax_B1.transAxes,
                        fill=False, edgecolor='#888888', linewidth=0.8,
                        clip_on=False, zorder=200)
    ax_B1.add_patch(border)
    ax_B1.text(0.40, 1.08, 'Gotoh DAG', transform=ax_B1.transAxes,
               fontsize=sty_b['title_fs'], fontweight='bold',
               va='bottom', ha='center')
    
    draw_flex_dag(
        seq1=X_clip,
        seq2=Y_clip,
        match=5,
        mismatch=-5,
        spacing=spacing_b,
        gadget_radius=gadget_radius,
        inner_radius_scale=sty_b['inner_radius_scale'],
        port_marker_size=sty_b['port_marker_size'],
        marker_edge_width=sty_b['marker_edge_width'],
        circle_alpha=sty_b['circle_alpha'],
        circle_linewidth=sty_b['circle_linewidth'],
        edge_linewidth=sty_b['edge_linewidth'],
        edge_alpha=sty_b['edge_alpha'],
        lw_internal=sty_b['internal_linewidth'],
        internal_alpha=sty_b['internal_alpha'],
        mutation_scale=sty_b['mutation_scale'],
        letter_fs=sty_b['letter_fs'],
        legend_fs=sty_b['legend_fs'],
        title="",
        show_legend=False,
        ax=ax_B2,
        max_nucleotide_x=max_nuc,
        max_nucleotide_y=max_nuc,
    )
    
    ax_B2.set_xlim(-margin, clip_limit)
    ax_B2.set_ylim(clip_limit, -margin)
    add_fade_edges(ax_B2, fade_width=0.12)
    border = Rectangle((-0.08, 0.0), 1.08, 1.05, transform=ax_B2.transAxes,
                        fill=False, edgecolor='#888888', linewidth=0.8,
                        clip_on=False, zorder=200)
    ax_B2.add_patch(border)
    ax_B2.text(0.40, 1.08, 'NW-flex DAG', transform=ax_B2.transAxes,
               fontsize=sty_b['title_fs'], fontweight='bold',
               va='bottom', ha='center')
    
    # Legend on the right (matches the main figure legend styling)
    legend_handles = [
        Line2D([0], [0], color=EDGE_COLORS['match'], lw=2, marker='>',
               markersize=5, label='match'),
        Line2D([0], [0], color=EDGE_COLORS['mismatch'], lw=2, marker='>',
               markersize=5, label='mismatch'),
        Line2D([0], [0], color=EDGE_COLORS['gap_extend'], lw=2, marker='>',
               markersize=5, label='gap extend'),
        Line2D([0], [0], color=EDGE_COLORS['gap_initiate'], lw=2, marker='>',
               markersize=5, label='gap initiate'),
        Line2D([0], [0], color=EDGE_COLORS['zero'], lw=2, marker='>',
               markersize=5, label='zero cost'),
        Line2D([0], [0], color='none', lw=0, label=''),        
        Line2D([0], [0], marker='o', markersize=7, linestyle='',
               markerfacecolor='gray', markeredgecolor='gray',
               markeredgewidth=1, label='in'),
        Line2D([0], [0], marker='s', markersize=7, linestyle='',
               markerfacecolor='white', markeredgecolor='gray',
               markeredgewidth=1, label='out'),
        Line2D([0], [0], marker='^', markersize=8, linestyle='',
               markerfacecolor='white', markeredgecolor='gray',
               markeredgewidth=1, label='max'),
        Line2D([0], [0], marker='o', markersize=7, linestyle='',
               markerfacecolor='white', markeredgecolor='gray',
               markeredgewidth=1, label='EP'),
        Line2D([0], [0], color='none', lw=0, label=''),
        Line2D([0], [0], marker='s', markersize=10, linestyle='',
               markerfacecolor=PORT_COLORS['H'], markeredgecolor=PORT_COLORS['H'],
               markeredgewidth=1, label='$Y_g$ layer'),
        Line2D([0], [0], marker='s', markersize=10, linestyle='',
               markerfacecolor=PORT_COLORS['D'], markeredgecolor=PORT_COLORS['D'],
               markeredgewidth=1, label='$M$ layer'),
        Line2D([0], [0], marker='s', markersize=10, linestyle='',
               markerfacecolor=PORT_COLORS['V'], markeredgecolor=PORT_COLORS['V'],
               markeredgewidth=1, label='$X_g$ layer'),
    ]
    
    ax_legend.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(-0.2, 0.5),
                     fontsize=10, 
                     frameon=True, fancybox=True,
                framealpha=0.95, edgecolor='#cccccc',
                labelspacing=0.6, borderpad=0.8, handletextpad=0.6,)
    
    plt.savefig(output_path, dpi=dpi, format=fmt, bbox_inches='tight',
                facecolor=FIGURE['background'])
    print(f"Saved supplementary figure 1 DAG comparison to: {output_path}")
    
    return fig


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate supplementary figure 1 gotoh_and_flex_DAGs output')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: tex/gotoh_and_flex_DAGs.pdf)')
    parser.add_argument('--dpi', type=int, default=None,
                        help='Resolution in DPI (default: 300)')
    parser.add_argument('--format', '-f', type=str, default=None,
                        choices=['pdf', 'png', 'svg', 'eps'],
                        help='Output format (default: pdf)')
    parser.add_argument('--show', action='store_true',
                        help='Display figure interactively')
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else Path(__file__).parent.parent / 'tex' / 'gotoh_and_flex_DAGs.pdf'
    fig = generate_panel_B(output_path=output_path, dpi=args.dpi, fmt=args.format)
    
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
