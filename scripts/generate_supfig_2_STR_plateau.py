#!/usr/bin/env python3
"""
generate_supfig_2_STR_plateau.py — Supplementary Figure 2 (STR plateau comparison)

This script produces the supplementary STR plateau figure (scalar plateau + phase
monotonicity) as a single output file for the STR_plateau panel.

Output (default):
  tex/STR_plateau_fig.pdf
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
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nwflex.ep_patterns import build_EP_single_block
from nwflex.aligners import align_with_EP
from nwflex.default import get_default_scoring
from nwflex.repeats import phase_repeat


# =============================================================================
# CONFIGURATION
# =============================================================================

FONTS_C = {
    'axis_label': 8,
    'tick': 6,
    'legend': 8,
    'title': 9,
}

PLATEAU_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

FIGURE = {
    'panel_C': {'width': 8.75, 'height': 2},
    'dpi': 300,
    'format': 'pdf',
    'background': '#ffffff',
}


# =============================================================================
# SUPFIG 2 HELPERS
# =============================================================================

def draw_panel_C_scalar(ax):
    """Draw the scalar plateau plot: S_flex(X_N, Y) vs N."""
    score_matrix, gap_open, gap_extend, a2i = get_default_scoring()
    
    A, R, B = "GAG", "ACT", "GTCA"
    k = len(R)
    
    # Labels describing partial repeat content
    test_cases = [
        (0, 0, 3, r'$(\mathrm{ACT})_3$', 3),        # no partials
        (1, 0, 2, r'$T(\mathrm{ACT})_2$', 3),       # left partial
        (0, 2, 2, r'$(\mathrm{ACT})_2\mathrm{AC}$', 3),     # right partial
        (2, 1, 2, r'$T(\mathrm{ACT})_2\mathrm{AC}$', 4),     # both partials
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
                markersize=4, linewidth=1.5, label=label)
    
    ax.set_xlabel('$N$', fontsize=FONTS_C['axis_label'])
    ax.set_ylabel('Score', fontsize=FONTS_C['axis_label'])
    ax.set_title('$S_{flex}(N)$ vs $N$', fontsize=FONTS_C['title'])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
              ncol=2, fontsize=FONTS_C['legend'], frameon=True,
              columnspacing=0.8, handletextpad=0.4)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONTS_C['tick'])


def get_phase_rows(s, e, k, r):
    """Get row indices for phase r within STR block [s+1, e]."""
    rows = []
    i = s + 1 + r
    while i <= e:
        rows.append(i)
        i += k
    return rows


_phase_data_cache = {}

def get_phase_alignment_data():
    """Get alignment data for phase monotonicity plots (computed once and cached)."""
    if 'data' not in _phase_data_cache:
        from nwflex.repeats import STRLocus
        from nwflex.ep_patterns import build_EP_STR_phase
        
        score_matrix, gap_open, gap_extend, a2i = get_default_scoring()
        
        locus = STRLocus(A="GAG", R="ACT", N=6, B="GTCA")
        a, b, M = 2, 1, 2
        Y = locus.build_locus_variant(a, b, M)
        print(locus.X)
        print(Y)
        
        EP = build_EP_STR_phase(locus.n, locus.s, locus.e, locus.k)
        
        result = align_with_EP(locus.X, Y, score_matrix, gap_open, gap_extend, a2i, EP,
                               return_data=True)
        
        _phase_data_cache['data'] = result.data
        _phase_data_cache['locus'] = locus
        _phase_data_cache['Y'] = Y
    
    return _phase_data_cache['data'], _phase_data_cache['locus'], _phase_data_cache['Y']


def draw_panel_C_phase(ax, phase_r, show_ylabel=True, show_legend=False):
    """
    Draw phase monotonicity for a single phase r (M layer only).
    Uses same example as notebook 4: N=6 reference, (a,b,M)=(2,1,2) read.
    """
    data, locus, Y = get_phase_alignment_data()
    s, e, k = locus.s, locus.e, locus.k
    M_mat = data.M
    
    rows_r = get_phase_rows(s, e, k, phase_r)
    n_repeats = len(rows_r)
    
    if n_repeats < 2:
        ax.text(0.5, 0.5, f'Phase {phase_r}\n(insufficient data)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=FONTS_C['axis_label'])
        ax.axis('off')
        return
    
    n_cols = M_mat.shape[1]
    cols_to_show = n_cols - 1 
    
    cmap = plt.cm.viridis_r
    n_vals = list(range(n_repeats))
    
    for col_idx, j in enumerate(range(1, cols_to_show + 1)):
        scores = [M_mat[i, j] for i in rows_r]
        color = cmap(col_idx / max(1, cols_to_show - 1))
        ax.plot(n_vals, scores, marker='o', markersize=3, linewidth=1.2,
                color=color, alpha=0.85)
    
    xlabels = [f'{n}' for n in n_vals]
    ax.set_xticks(n_vals)
    ax.set_xticklabels(xlabels, fontsize=FONTS_C['tick'])
    
    ax.set_xlabel('$n$', fontsize=FONTS_C['axis_label'])
    if show_ylabel:
        ax.set_ylabel('$M(i,j)$', fontsize=FONTS_C['axis_label'])
    ax.set_title(f'$r={phase_r}$', fontsize=FONTS_C['title'])
    ax.tick_params(labelsize=FONTS_C['tick'])
    ax.grid(True, alpha=0.3)


# =============================================================================
# SUPFIG 2 GENERATOR
# =============================================================================

def generate_panel_C(output_path, dpi=None, fmt=None, include_scalar=True):
    """Generate supplementary figure 2 STR plateau output.
    
    Args:
        output_path: Path for output file
        dpi: Resolution in DPI
        fmt: Output format (pdf, png, etc.)
        include_scalar: If True, include the scalar plateau plot (leftmost panel).
                        If False, only show the phase monotonicity plots.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dpi is None:
        dpi = FIGURE['dpi']
    if fmt is None:
        fmt = output_path.suffix[1:] if output_path.suffix else FIGURE['format']
    
    mpl.rcParams['pdf.fonttype'] = 42
    
    if include_scalar:
        # Full layout with scalar plot + 3 phase plots
        fig = plt.figure(figsize=(FIGURE['panel_C']['width'], FIGURE['panel_C']['height']))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1.5, 1, 1, 1], wspace=0.025)
        
        ax_C1 = fig.add_subplot(gs[0])
        ax_C2 = fig.add_subplot(gs[1])
        ax_C3 = fig.add_subplot(gs[2], sharey=ax_C2)
        ax_C4 = fig.add_subplot(gs[3], sharey=ax_C2)
        
        pos = ax_C1.get_position()
        ax_C1.set_position([pos.x0, pos.y0, pos.width * 0.7, pos.height])
        
        draw_panel_C_scalar(ax_C1)
    else:
        # Phase-only layout: just the 3 phase monotonicity plots
        fig_width = FIGURE['panel_C']['width'] * 0.6  # Narrower without scalar
        fig = plt.figure(figsize=(fig_width, FIGURE['panel_C']['height']))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.08)
        
        ax_C2 = fig.add_subplot(gs[0])
        ax_C3 = fig.add_subplot(gs[1], sharey=ax_C2)
        ax_C4 = fig.add_subplot(gs[2], sharey=ax_C2)
    
    draw_panel_C_phase(ax_C2, phase_r=0, show_ylabel=True, show_legend=False)
    draw_panel_C_phase(ax_C3, phase_r=1, show_ylabel=False, show_legend=False)
    draw_panel_C_phase(ax_C4, phase_r=2, show_ylabel=False, show_legend=False)
    
    plt.setp(ax_C3.get_yticklabels(), visible=False)
    plt.setp(ax_C4.get_yticklabels(), visible=False)
    ax_C3.tick_params(axis='y', length=0)
    ax_C4.tick_params(axis='y', length=0)
    
    data, locus, _ = get_phase_alignment_data()
    cols_to_show = data.M.shape[1] - 1

    cmap = plt.cm.viridis_r
    bounds = np.arange(0.5, cols_to_show + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Position colorbar based on layout
    if include_scalar:
        cbar_ax = fig.add_axes([0.385, -0.15, 0.50, 0.05])
    else:
        cbar_ax = fig.add_axes([0.15, -0.15, 0.70, 0.05])
    even_ticks = [j for j in range(1, cols_to_show + 1) if j % 2 == 0]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=even_ticks)
    cbar.ax.set_xticklabels([f'{j}' for j in even_ticks], fontsize=5)
    cbar.ax.tick_params(which='minor', length=0)  # Remove only minor ticks
    cbar.ax.set_xlabel('col $j$', fontsize=6, labelpad=1, loc='left')
    
    plt.savefig(output_path, dpi=dpi, format=fmt, bbox_inches='tight',
                facecolor=FIGURE['background'])
    print(f"Saved supplementary figure 2 STR plateau to: {output_path}")
    
    return fig


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate supplementary figure 2 STR_plateau output')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: tex/STR_plateau_fig.pdf)')
    parser.add_argument('--dpi', type=int, default=None,
                        help='Resolution in DPI (default: 300)')
    parser.add_argument('--format', '-f', type=str, default=None,
                        choices=['pdf', 'png', 'svg', 'eps'],
                        help='Output format (default: pdf)')
    parser.add_argument('--show', action='store_true',
                        help='Display figure interactively')
    parser.add_argument('--no-scalar', action='store_true',
                        help='Exclude the scalar plateau plot (leftmost panel)')
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else Path(__file__).parent.parent / 'tex' / 'STR_plateau_fig.pdf'
    fig = generate_panel_C(output_path=output_path, dpi=args.dpi, fmt=args.format,
                           include_scalar=not args.no_scalar)
    
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
