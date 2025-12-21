"""
EP (Extra Predecessor) pattern visualization for NW-flex.

This module contains functions for visualizing EP patterns, region shading,
and EP comparisons.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from typing import Dict, Optional, List, Tuple, Union

from .colors import (
    NT_COLOR, REGION_LABEL_COLORS, REGION_COLORS
)


# =============================================================================
# REGION BACKGROUND SHADING
# =============================================================================

def draw_ep_background(
    ax: plt.Axes,
    n: int,
    regions: Optional[List[dict]] = None,
    row_spacing: float = 1.0,
    node_height: float = 1.0,
    x_node: float = 1.0,
    node_width: float = 2.0,
    block_alpha: float = 0.3,
    region_fontsize: float = 20.0,
    region_fontweight: str = "bold",
    region_label_x_offset: float = -1.5,
    bg_x_left_margin: float = 0.8,
    bg_x_right_margin: float = 0.3,
) -> None:
    """
    Draw background region shading for EP pattern visualization.
    
    This function draws colored horizontal bands to indicate sequence regions
    (like A·Z·B decomposition) behind the EP DAG.
    
    Parameters
    ----------
    ax : matplotlib Axes
        Axes to draw on.
    n : int
        Number of sequence positions (rows 0..n).
    regions : list of dict, optional
        List of region specifications. Each dict has:
        - 'start': int - starting row (inclusive)
        - 'end': int - ending row (inclusive)
        - 'color': str - fill color
        - 'label': str, optional - label text
        - 'label_color': str, optional - label text color
    row_spacing : float
        Vertical spacing between rows.
    node_height : float
        Height of row nodes.
    x_node : float
        X position of nodes.
    node_width : float
        Width of row nodes.
    block_alpha : float
        Alpha transparency for region fills.
    region_fontsize : float
        Font size for region labels.
    region_fontweight : str
        Font weight for region labels.
    region_label_x_offset : float
        X offset for region labels (negative = left of nodes).
    bg_x_left_margin : float
        Left margin from node edge for background box.
    bg_x_right_margin : float
        Right margin from node edge for background box.
    """
    if regions is None:
        return
    
    def row_y(i):
        return i * row_spacing
    
    half_h = node_height / 2 + 0.1
    label_x = x_node - node_width / 2 + region_label_x_offset
    
    bg_x_left = x_node - node_width / 2 - bg_x_left_margin
    bg_x_right = x_node + node_width / 2 + bg_x_right_margin
    
    for region in regions:
        start = region['start']
        end = region['end']
        color = region['color']
        label = region.get('label')
        label_color = region.get('label_color', color)
        
        y_top = row_y(start) - half_h if start == 0 else row_y(start - 1) + half_h
        y_bottom = row_y(end) + half_h
        
        ax.fill_between([bg_x_left, bg_x_right], y_top, y_bottom,
                        color=color, alpha=block_alpha, zorder=0)
        
        if label:
            y_center = (y_top + y_bottom) / 2
            ax.text(label_x, y_center, label, fontsize=region_fontsize,
                    fontweight=region_fontweight, ha="right", va="center", color=label_color)


# =============================================================================
# DP MATRIX REGION ANNOTATION
# =============================================================================

def draw_matrix_regions(
    axes: Union[plt.Axes, List[plt.Axes]],
    regions: List[dict],
    label_side: str = 'right',
    linewidth: float = 2.5,
    linestyle: str = '-',
    label_fontsize: float = 14.0,
    label_fontweight: str = 'bold',
    label_offset: float = 0.15,
    zorder: int = 10,
) -> None:
    """
    Draw region outline boxes on DP matrix heatmap axes.
    
    Parameters
    ----------
    axes : Axes or list of Axes
        Matplotlib axes to draw on.
    regions : list of dict
        List of region specifications. Each dict has:
        - 'start': int - starting row (1-indexed)
        - 'end': int - ending row (1-indexed)
        - 'label': str - label text (e.g., 'A', 'Z*', 'B')
        - 'color': str, optional - outline color (defaults to light fill color)
        - 'label_color': str, optional - label text color (defaults to dark label color)
        - 'linestyle': str, optional - line style
        - 'label_position': str, optional - 'top', 'middle', or 'bottom'
    label_side : str
        Side for labels: 'left' or 'right'.
    linewidth : float
        Width of region outline.
    linestyle : str
        Line style for outline.
    label_fontsize : float
        Font size for region labels.
    label_fontweight : str
        Font weight for region labels.
    label_offset : float
        Distance from axis edge to label.
    zorder : int
        Z-order for drawing.
    """
    if hasattr(axes, '__iter__') and not isinstance(axes, plt.Axes):
        axes = list(axes)
    else:
        axes = [axes]
    
    for ax in axes:
        xlim = ax.get_xlim()
        
        for region in regions:
            start = region['start']
            end = region['end']
            label = region.get('label', '')
            
            # Edge color: use explicit 'color' or default to dark label color for visibility
            edge_color = region.get('color')
            if edge_color is None:
                edge_color = REGION_COLORS.get(label, {}).get('label', 'gray')
            
            # Label text color: use explicit 'label_color' or default to dark label color
            label_color = region.get('label_color')
            if label_color is None:
                label_color = REGION_COLORS.get(label, {}).get('label', edge_color)
            
            region_linestyle = region.get('linestyle', linestyle)
            
            rect = Rectangle(
                (xlim[0], start),
                xlim[1] - xlim[0],
                end - start + 1,
                linewidth=linewidth,
                edgecolor=edge_color,
                facecolor='none',
                linestyle=region_linestyle,
                zorder=zorder,
                clip_on=False,
            )
            ax.add_patch(rect)
            
            if label:
                label_position = region.get('label_position', 'middle')
                if label_position == 'top':
                    y_pos = start + 0.5
                elif label_position == 'bottom':
                    y_pos = end + 0.5
                else:
                    y_pos = start + (end - start + 1) / 2
                
                if label_side == 'right':
                    x_pos = xlim[1] + label_offset
                    ha = 'left'
                else:
                    x_pos = xlim[0] - label_offset
                    ha = 'right'
                
                ax.text(
                    x_pos, y_pos, label,
                    color=label_color,
                    ha=ha,
                    va='center',
                    fontsize=label_fontsize,
                    fontweight=label_fontweight,
                    clip_on=False,
                )


def build_AZB_regions(
    s: int,
    e: int,
    n: int,
    zstar_rows: Optional[Tuple[int, int]] = None,
    colors: Optional[dict] = None,
) -> List[dict]:
    """
    Build a standard A·Z·B (or A·Z*·B) region list for DP matrix annotation.
    
    Parameters
    ----------
    s : int
        Leader row index (last row of A, 0-indexed into X).
    e : int
        End of flexible block Z (e = s + len(Z)).
    n : int
        Length of X (total sequence length).
    zstar_rows : tuple of (int, int), optional
        If provided, label Z region as Z* with these bounds.
    colors : dict, optional
        Custom colors for regions.
    
    Returns
    -------
    list of dict
        Region specifications for draw_matrix_regions().
    """
    if colors is None:
        colors = {}
    
    regions = []
    
    if s >= 1:
        # Use dark label color for edge (accessible contrast), explicit label_color for text
        edge_A = colors.get('A', REGION_LABEL_COLORS['A'])
        label_A = colors.get('A_label', REGION_LABEL_COLORS['A'])
        regions.append({
            'start': 1,
            'end': s,
            'label': 'A',
            'color': edge_A,
            'label_color': label_A,
        })
    
    if zstar_rows is not None:
        zstart, zend = zstar_rows
        edge_Z = colors.get('Z*', colors.get('Z', REGION_LABEL_COLORS['Z']))
        label_Z = colors.get('Z*_label', colors.get('Z_label', REGION_LABEL_COLORS['Z']))
        regions.append({
            'start': zstart,
            'end': zend,
            'label': 'Z*',
            'color': edge_Z,
            'label_color': label_Z,
        })
    else:
        if e > s:
            edge_Z = colors.get('Z', REGION_LABEL_COLORS['Z'])
            label_Z = colors.get('Z_label', REGION_LABEL_COLORS['Z'])
            regions.append({
                'start': s + 1,
                'end': e,
                'label': 'Z',
                'color': edge_Z,
                'label_color': label_Z,
            })
    
    if e < n:
        edge_B = colors.get('B', REGION_LABEL_COLORS['B'])
        label_B = colors.get('B_label', REGION_LABEL_COLORS['B'])
        regions.append({
            'start': e + 1,
            'end': n,
            'label': 'B',
            'color': edge_B,
            'label_color': label_B,
        })
    
    return regions


# =============================================================================
# EP PATTERN VISUALIZATION
# =============================================================================

def plot_ep_pattern(
    EP: List[List[int]],
    leaders: List[int],
    title: Optional[str] = None,
    X: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    node_width: float = 2.0,
    node_height: float = 1.0,
    node_color: str = "white",
    node_edge_color: str = "black",
    node_linewidth: float = 1.0,
    source_color: str = "#f0f0f0",
    sink_color: str = "#f0f0f0",
    leader_box_linewidth: float = 1.5,
    closer_box_linewidth: float = 1.5,
    show_standard_edges: bool = True,
    std_edge_color: str = "0.6",
    std_edge_width: float = 1.0,
    std_edge_alpha: float = 0.8,
    leader_edge_color: str = "#1f77b4",
    closer_edge_color: str = "#7030a0",
    ep_linewidth: float = 1.8,
    ep_edge_alpha: float = 0.7,
    leader_curvature: float = 0.3,
    closer_curvature: float = 0.25,
    curvature_distance_scale: float = 0.0,
    ep_mutation_scale: float = 10.0,
    ep_arc_offset: float = 0.3,
    show_row_labels: bool = True,
    show_seq_labels: bool = True,
    sequence_fontsize: float = 18.0,
    sequence_fontweight: str = "bold",
    row_label_fontsize: float = 12.0,
    row_label_fontweight: str = "normal",
    title_fontsize: float = 14.0,
    row_annotations: Optional[Dict[int, str]] = None,
    show_box_labels: bool = True,
    box_label_fontsize: float = 12.0,
    box_label_fontweight: str = "bold",
    box_label_fontstyle: str = "italic",
    row_spacing: float = 1.0,
    figsize: Tuple[float, float] = (4, 8),
    nt_color_map: Optional[Dict[str, str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize an extra-predecessor (EP) pattern as a DAG-style row diagram.
    
    EP arcs are drawn on the RIGHT side of the figure, curving outward.
    
    Parameters
    ----------
    EP : list of lists
        Extra predecessor sets. EP[i] contains extra predecessors for row i.
    leaders : list of int
        Row indices that are leader rows.
    title : str, optional
        Title for the plot.
    X : str, optional
        Reference sequence.
    ax : matplotlib Axes, optional
        Axes to draw on.
    
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if nt_color_map is None:
        nt_color_map = NT_COLOR
    
    n = len(EP) - 1
    leader_set = set(leaders)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    def row_y(i):
        return i * row_spacing
    
    x_node = 1.0
    
    closer_rows = set()
    for i, ep_set in enumerate(EP):
        for pred in ep_set:
            if i == pred + 1:
                continue
            if pred not in leader_set:
                closer_rows.add(i)
    
    if show_standard_edges:
        for i in range(1, n + 1):
            y0 = row_y(i - 1) + node_height / 2 + 0.05
            y1 = row_y(i) - node_height / 2 - 0.05
            ax.annotate(
                "", xy=(x_node, y1), xytext=(x_node, y0),
                arrowprops=dict(
                    arrowstyle="->",
                    color=std_edge_color,
                    lw=std_edge_width,
                    alpha=std_edge_alpha,
                    shrinkA=0, shrinkB=0,
                ),
                zorder=1,
            )
    
    arc_x_start = x_node + node_width / 2 + ep_arc_offset
    
    for i, ep_set in enumerate(EP):
        for pred in ep_set:
            if i == pred + 1:
                continue
            
            if pred in leader_set:
                edge_color = leader_edge_color
                base_curvature = leader_curvature
            else:
                edge_color = closer_edge_color
                base_curvature = closer_curvature
            
            y0 = row_y(pred)
            y1 = row_y(i)
            
            dist = abs(i - pred)
            curvature = base_curvature * (1 + curvature_distance_scale * dist)
            
            arrow = FancyArrowPatch(
                (arc_x_start, y0),
                (arc_x_start, y1),
                connectionstyle=f"arc3,rad=-{curvature}",
                arrowstyle="-|>",
                mutation_scale=ep_mutation_scale,
                color=edge_color,
                linewidth=ep_linewidth,
                alpha=ep_edge_alpha,
                zorder=2,
            )
            ax.add_patch(arrow)
    
    for i in range(n + 1):
        y = row_y(i)
        if i == 0:
            fill_color = source_color
        elif i == n:
            fill_color = sink_color
        else:
            fill_color = node_color
        
        if i in leader_set or i in closer_rows:
            if i in leader_set:
                box_lw = leader_box_linewidth
                box_edge_color = leader_edge_color
                box_label = "leader"
            else:
                box_lw = closer_box_linewidth
                box_edge_color = closer_edge_color
                box_label = "closer"
            rect = FancyBboxPatch(
                (x_node - node_width / 2, y - node_height / 2),
                node_width,
                node_height,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=fill_color,
                edgecolor=box_edge_color,
                linewidth=box_lw,
                linestyle='--',
                zorder=3,
            )
            ax.add_patch(rect)
            if show_box_labels:
                box_right = x_node + node_width / 2
                ax.text(
                    box_right + 0.5, y, box_label,
                    ha="left", va="center",
                    fontsize=box_label_fontsize,
                    fontweight=box_label_fontweight,
                    fontstyle=box_label_fontstyle,
                    color=box_edge_color,
                    zorder=4,
                )
        else:
            rect = plt.Rectangle(
                (x_node - node_width / 2, y - node_height / 2),
                node_width,
                node_height,
                facecolor=fill_color,
                edgecolor=node_edge_color,
                linewidth=node_linewidth,
                zorder=3,
            )
            ax.add_patch(rect)
    
    if show_row_labels:
        if row_annotations is None:
            row_annotations = {}
        for i in range(n + 1):
            y = row_y(i)
            annotation = row_annotations.get(i, "")
            label = f"{i} {annotation}".strip() if annotation else str(i)
            ax.text(
                x_node, y, label,
                fontsize=row_label_fontsize,
                ha="center", va="center",
                fontweight=row_label_fontweight,
                zorder=4,
            )
    
    if show_seq_labels and X is not None:
        for i in range(1, min(n + 1, len(X) + 1)):
            y_mid = (row_y(i - 1) + row_y(i)) / 2
            base = X[i - 1] if i - 1 < len(X) else ""
            color = nt_color_map.get(base, "black")
            ax.text(
                x_node - node_width / 2 - 0.2, y_mid, base,
                fontsize=sequence_fontsize,
                ha="right", va="center",
                color=color,
                fontweight=sequence_fontweight,
            )
    
    max_arc_extent = max(leader_curvature, closer_curvature) * n * row_spacing * 0.5 + 0.8
    ax.set_xlim(x_node - node_width / 2 - 2.0,
                x_node + node_width / 2 + ep_arc_offset + max_arc_extent + 1.5)
    ax.set_ylim(row_y(n) + 0.6, row_y(0) - 0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=10)
    
    return fig, ax


def summarize_ep_edges(
    EP: List[List[int]],
    leaders: List[int],
    title: Optional[str] = None,
) -> str:
    """
    Generate a text summary of EP edges.
    
    Parameters
    ----------
    EP : list of list of int
        Extra predecessor pattern.
    leaders : list of int
        Leader row indices.
    title : str, optional
        Title for the summary.
    
    Returns
    -------
    str : Summary text.
    """
    n = len(EP) - 1
    leader_set = set(leaders)
    
    leader_edges = []
    closer_edges = []
    
    for i, ep_set in enumerate(EP):
        for pred in ep_set:
            if i == pred + 1:
                continue
            if pred in leader_set:
                leader_edges.append((pred, i))
            else:
                closer_edges.append((pred, i))
    
    lines = []
    if title:
        lines.append(f"=== {title} ===")
    else:
        lines.append(f"=== EP Summary (n={n}) ===")
    
    lines.append(f"Leader rows: {sorted(leaders) if leaders else 'none'}")
    
    lines.append("")
    if leader_edges:
        lines.append(f"Leader edges (blue, from leader rows) - {len(leader_edges)} total:")
        for src, tgt in sorted(leader_edges):
            lines.append(f"  row {src} → row {tgt}")
    else:
        lines.append("Leader edges: none")
    
    lines.append("")
    if closer_edges:
        lines.append(f"Closer edges (green, from non-leader rows) - {len(closer_edges)} total:")
        for src, tgt in sorted(closer_edges):
            lines.append(f"  row {src} → row {tgt}")
    else:
        lines.append("Closer edges: none")
    
    return "\n".join(lines)


def plot_ep_comparison(
    EP_list: List[List[List[int]]],
    leaders_list: List[List[int]],
    titles: List[str],
    X: Optional[str] = None,
    backgrounds: Optional[List[Optional[List[dict]]]] = None,
    row_annotations_list: Optional[List[Optional[Dict[int, str]]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show_legend: bool = True,
    legend_fontsize: float = 12.0,
    wspace: float = 0.3,
    **kwargs,
) -> plt.Figure:
    """
    Plot multiple EP patterns side by side for comparison.
    
    Parameters
    ----------
    EP_list : list of EP patterns
        Each EP pattern is a list of lists.
    leaders_list : list of list of int
        Leader row indices for each panel.
    titles : list of str
        Titles for each panel.
    X : str, optional
        Reference sequence.
    backgrounds : list of list of dict, optional
        Background region configs for each panel.
    row_annotations_list : list of dict, optional
        Row annotations for each panel.
    figsize : tuple, optional
        Figure size.
    show_legend : bool
        If True, add a legend below panels.
    legend_fontsize : float
        Font size for legend.
    wspace : float
        Horizontal space between panels.
    **kwargs : dict
        Additional arguments passed to plot_ep_pattern and draw_ep_background.
    
    Returns
    -------
    fig : matplotlib Figure
    """
    n_panels = len(EP_list)
    n_rows = len(EP_list[0]) - 1 if EP_list else 10
    
    if backgrounds is None:
        backgrounds = [None] * n_panels
    if row_annotations_list is None:
        row_annotations_list = [None] * n_panels
    
    if figsize is None:
        panel_width = 4.0
        panel_height = max(6, n_rows * 0.8 + 2)
        figsize = (panel_width * n_panels, panel_height)
    
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]
    
    bg_params = {'row_spacing', 'node_height', 'node_width', 'block_alpha',
                 'region_fontsize', 'region_fontweight', 'region_label_x_offset',
                 'bg_x_left_margin', 'bg_x_right_margin'}
    
    ep_params = {'node_width', 'node_height', 'node_color', 'node_edge_color', 
                 'node_linewidth', 'source_color', 'sink_color',
                 'leader_box_linewidth', 'closer_box_linewidth',
                 'show_standard_edges', 'std_edge_color', 'std_edge_width', 'std_edge_alpha',
                 'leader_edge_color', 'closer_edge_color', 'ep_linewidth', 'ep_edge_alpha',
                 'leader_curvature', 'closer_curvature', 'curvature_distance_scale',
                 'ep_mutation_scale', 'ep_arc_offset',
                 'show_row_labels', 'show_seq_labels', 
                 'sequence_fontsize', 'sequence_fontweight',
                 'row_label_fontsize', 'row_label_fontweight', 'title_fontsize',
                 'show_box_labels', 'box_label_fontsize', 'box_label_fontweight', 'box_label_fontstyle',
                 'row_spacing', 'figsize', 'nt_color_map'}
    
    for ax, EP, leaders, title, bg, annot in zip(
        axes, EP_list, leaders_list, titles, backgrounds, row_annotations_list
    ):
        n = len(EP) - 1
        if bg:
            draw_ep_background(ax, n, regions=bg, **{
                k: v for k, v in kwargs.items() if k in bg_params
            })
        
        plot_ep_pattern(EP, leaders=leaders, title=title, X=X, ax=ax,
                       row_annotations=annot, **{
                           k: v for k, v in kwargs.items() if k in ep_params
                       })
    
    if show_legend:
        leader_color = kwargs.get('leader_edge_color', '#1f77b4')
        closer_color = kwargs.get('closer_edge_color', '#7030a0')
        standard_color = kwargs.get('std_edge_color', '0.6')
        node_color = kwargs.get('node_color', 'white')
        legend_bbox_y = kwargs.pop('legend_bbox_y', 0.25)
        bottom_margin = kwargs.pop('bottom_margin', 0.12)
        
        row_patch = FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0",
                                    facecolor=node_color, edgecolor='black',
                                    linewidth=1.5)
        
        legend_elements = [
            row_patch,
            Line2D([0], [0], color=standard_color, linewidth=2, linestyle='-',
                   marker='>', markersize=6, markeredgecolor=standard_color),
            Line2D([0], [0], color=leader_color, linewidth=2.5, linestyle='-',
                   marker='>', markersize=6, markeredgecolor=leader_color),
            Line2D([0], [0], color=closer_color, linewidth=2.5, linestyle='-',
                   marker='>', markersize=6, markeredgecolor=closer_color),
        ]
        
        legend_labels = [
            'row $i$ of array',
            'Standard predecessor $(i-1 \\to i)$',
            'Extra predecessor (from leader)',
            'Extra predecessor (to closer)',
        ]
        
        fig.legend(handles=legend_elements, labels=legend_labels,
                   loc='lower center',
                   ncol=2, fontsize=legend_fontsize,
                   frameon=True, bbox_to_anchor=(0.5, legend_bbox_y),
                   fancybox=True, edgecolor='#cccccc', facecolor='white',
                   framealpha=1.0, borderpad=0.8, handletextpad=0.6,
                   columnspacing=1.5)
    
    plt.tight_layout()
    if show_legend:
        plt.subplots_adjust(wspace=wspace, bottom=bottom_margin)
    else:
        plt.subplots_adjust(wspace=wspace)
    
    return fig


def print_ep_comparison_summary(
    EP_list: List[List[List[int]]],
    leaders_list: List[List[int]],
    titles: List[str],
) -> None:
    """
    Print summaries for multiple EP patterns.
    """
    summaries = []
    for EP, leaders, title in zip(EP_list, leaders_list, titles):
        clean_title = title.replace('\n', ' ').replace('$', '').replace('\\varnothing', '∅')
        summary = summarize_ep_edges(EP, leaders=leaders, title=clean_title)
        summaries.append(summary)
    
    print("\n\n".join(summaries))
