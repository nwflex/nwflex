"""
DAG (Directed Acyclic Graph) visualization for NW-flex.

This module contains functions for visualizing alignment DAGs,
K_{3,3} gadgets, flex gadgets, and combined EP+DAG views.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch, RegularPolygon
from matplotlib.lines import Line2D
from typing import Dict, Optional, List, Tuple

from .colors import (
    NT_COLOR, PORT_COLORS, EDGE_COLORS
)
from .utils import polar


# =============================================================================
# COMBINED EP + DAG VISUALIZATION
# =============================================================================

def plot_dag_with_ep_arcs(
    EP: List[List[int]],
    leaders: List[int],
    X: str,
    Y: str,
    s: int,
    e: int,
    figsize: Tuple[float, float] = (10, 9),
    ax=None,
    color_A: str = "#d0e0ff",
    color_Z: str = "#ffe0d0",
    color_B: str = "#e0d0f0",
    label_A: str = "#2060a0",
    label_Z: str = "#c06020",
    label_B: str = "#7030a0",
    leader_edge_color: str = "#1f77b4",
    closer_edge_color: str = "#7030a0",
    match_color: str = "#2ca02c",
    mismatch_color: str = "#d62728",
    gap_color: str = "#7f7f7f",
    dag_edge_alpha: float = 0.25,
    ep_arc_alpha: float = 0.7,
    node_radius: float = 0.3,
    spacing: float = 1.0,
    leader_curvature: float = 0.3,
    closer_curvature: float = 0.25,
    curvature_distance_scale: float = 0.0,
    show_legend: bool = True,
    show_region_labels: bool = True,
    show_row_boxes: bool = True,
    title: Optional[str] = None,
    dag_linewidth: float = 1.5,
    dag_mutation_scale: float = 8.0,
    dag_shrink_factor: float = 1.1,
    ep_linewidth: float = 1.8,
    ep_mutation_scale: float = 10.0,
    ep_arc_offset: float = 0.5,
    region_fontsize: float = 20.0,
    region_fontweight: str = "bold",
    sequence_fontsize: float = 18.0,
    sequence_fontweight: str = "bold",
    box_label_fontsize: float = 11.0,
    box_label_fontweight: str = "bold",
    box_label_fontstyle: str = "italic",
    title_fontsize: float = 12.0,
    legend_fontsize: float = 12.0,
    box_linewidth: float = 3.0,
    region_alpha: float = 0.3,
    region_edgecolor: str = 'none',
    legend_bbox_to_anchor: Tuple[float, float] = (0.45, -0.18),
) -> plt.Figure:
    """
    Create a single-panel figure showing the alignment DAG with EP arcs on the right.
    
    Parameters
    ----------
    EP : list of list of int
        Extra predecessor pattern.
    leaders : list of int
        Leader row indices.
    X : str
        Reference sequence (rows).
    Y : str
        Read sequence (columns).
    s : int
        Start of flexible block (leader row index).
    e : int
        End of flexible block.
    figsize : tuple
        Figure size.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    
    Returns
    -------
    fig : matplotlib Figure
    """
    n = len(X)
    m = len(Y)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()
    
    def node_xy(i, j):
        return (j * spacing, i * spacing)
    
    # Draw region shading
    half_h = node_radius + 0.1
    region_x_left = -1.2
    region_x_right = m * spacing + 2.0
    region_label_x = -1.5
    
    if s >= 1:
        y_top = spacing * 0 + half_h
        y_bottom = spacing * s + half_h
        ax.fill_between([region_x_left, region_x_right], y_top, y_bottom, 
                        color=color_A, alpha=region_alpha, edgecolor=region_edgecolor, zorder=0)
        if show_region_labels:
            y_center = (y_top + y_bottom) / 2
            ax.text(region_label_x, y_center, "A", fontsize=region_fontsize, 
                   fontweight=region_fontweight, ha="center", va="center", color=label_A)
    
    if e > s:
        y_top = spacing * s + half_h
        y_bottom = spacing * e + half_h
        ax.fill_between([region_x_left, region_x_right], y_top, y_bottom,
                        color=color_Z, alpha=region_alpha, edgecolor=region_edgecolor, zorder=0)
        if show_region_labels:
            y_center = (y_top + y_bottom) / 2
            ax.text(region_label_x, y_center, "Z", fontsize=region_fontsize, 
                   fontweight=region_fontweight, ha="center", va="center", color=label_Z)
    
    if n > e:
        y_top = spacing * e + half_h
        y_bottom = spacing * n + half_h
        ax.fill_between([region_x_left, region_x_right], y_top, y_bottom,
                        color=color_B, alpha=region_alpha, edgecolor=region_edgecolor, zorder=0)
        if show_region_labels:
            y_center = (y_top + y_bottom) / 2
            ax.text(region_label_x, y_center, "B", fontsize=region_fontsize, 
                   fontweight=region_fontweight, ha="center", va="center", color=label_B)
    
    # Draw standard DAG edges
    shrink_dist = node_radius * dag_shrink_factor
    
    for i in range(n + 1):
        for j in range(m + 1):
            x0, y0 = node_xy(i, j)
            
            # Vertical edge
            if i < n:
                x1, y1 = node_xy(i + 1, j)
                arrow = FancyArrowPatch(
                    (x0, y0 + shrink_dist), (x1, y1 - shrink_dist),
                    arrowstyle="-|>",
                    mutation_scale=dag_mutation_scale,
                    color=gap_color,
                    linewidth=dag_linewidth,
                    zorder=1
                )
                ax.add_patch(arrow)
            
            # Horizontal edge
            if j < m:
                x1, y1 = node_xy(i, j + 1)
                arrow = FancyArrowPatch(
                    (x0 + shrink_dist, y0), (x1 - shrink_dist, y1),
                    arrowstyle="-|>",
                    mutation_scale=dag_mutation_scale,
                    color=gap_color,
                    linewidth=dag_linewidth,
                    zorder=1
                )
                ax.add_patch(arrow)
            
            # Diagonal edge
            if i < n and j < m:
                x1, y1 = node_xy(i + 1, j + 1)
                is_match = X[i] == Y[j]
                edge_color = match_color if is_match else mismatch_color
                diag_shrink = shrink_dist * 0.707
                arrow = FancyArrowPatch(
                    (x0 + diag_shrink, y0 + diag_shrink),
                    (x1 - diag_shrink, y1 - diag_shrink),
                    arrowstyle="-|>",
                    mutation_scale=dag_mutation_scale,
                    color=edge_color,
                    linewidth=dag_linewidth,
                    zorder=1
                )
                ax.add_patch(arrow)
    
    # Draw DAG nodes
    for i in range(n + 1):
        for j in range(m + 1):
            x, y = node_xy(i, j)
            circ = Circle(
                (x, y), radius=node_radius,
                facecolor='white', edgecolor='black', linewidth=0.8,
                zorder=3
            )
            ax.add_patch(circ)
    
    # Draw row boxes
    if show_row_boxes:        
        box_left = -0.4
        box_right = m * spacing + 0.4
        
        leader_row = s
        y_bottom_leader = spacing * s + half_h
        y_top_leader = y_bottom_leader - spacing
        box_height_leader = y_bottom_leader - y_top_leader
        
        box = FancyBboxPatch(
            (box_left, y_top_leader),
            box_right - box_left, box_height_leader,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor='none',
            edgecolor=leader_edge_color,
            linewidth=box_linewidth,
            linestyle='--',
            zorder=4
        )
        ax.add_patch(box)
        ax.text(box_right + 0.25, leader_row * spacing, "leader",
                ha="left", va="center", fontsize=box_label_fontsize,
                color=leader_edge_color, fontstyle=box_label_fontstyle, 
                fontweight=box_label_fontweight)
        
        closer_row = e + 1
        if closer_row <= n:
            y_top_closer = spacing * e + half_h
            y_bottom_closer = spacing * closer_row + half_h
            box_height_closer = y_bottom_closer - y_top_closer
            
            box = FancyBboxPatch(
                (box_left, y_top_closer),
                box_right - box_left, box_height_closer,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                facecolor='none',
                edgecolor=closer_edge_color,
                linewidth=box_linewidth,
                linestyle='--',
                zorder=4
            )
            ax.add_patch(box)
            ax.text(box_right + 0.25, closer_row * spacing, "closer",
                    ha="left", va="center", fontsize=box_label_fontsize,
                    color=closer_edge_color, fontstyle=box_label_fontstyle, 
                    fontweight=box_label_fontweight)
    
    # Draw EP arcs
    leader_set = set(leaders)
    arc_x_start = m * spacing + ep_arc_offset
    
    for target_row, ep_list in enumerate(EP):
        for source_row in ep_list:
            if source_row == target_row - 1:
                continue
            
            is_leader = source_row in leader_set
            edge_color = leader_edge_color if is_leader else closer_edge_color
            base_curvature = leader_curvature if is_leader else closer_curvature
            
            y0 = source_row * spacing
            y1 = target_row * spacing
            
            dist = abs(target_row - source_row)
            curvature = base_curvature * (1 + curvature_distance_scale * dist)
            
            arrow = FancyArrowPatch(
                (arc_x_start, y0), (arc_x_start, y1),
                connectionstyle=f"arc3,rad=-{curvature}",
                arrowstyle="-|>",
                mutation_scale=ep_mutation_scale,
                color=edge_color,
                linewidth=ep_linewidth,
                zorder=2
            )
            ax.add_patch(arrow)
    
    # Sequence labels
    for i, base in enumerate(X):
        y_pos = (i + 0.5) * spacing
        base_color = NT_COLOR.get(base, "black")
        ax.text(
            -0.55, y_pos, base,
            ha="right", va="center",
            fontsize=sequence_fontsize, fontweight=sequence_fontweight, color=base_color
        )
    
    for j, base in enumerate(Y):
        x_pos = (j + 0.5) * spacing
        base_color = NT_COLOR.get(base, "black")
        ax.text(
            x_pos, -0.5, base,
            ha="center", va="bottom",
            fontsize=sequence_fontsize, fontweight=sequence_fontweight, color=base_color
        )
    
    # Configure axes
    ax.set_xlim(-2.0, m * spacing + 3.5)
    ax.set_ylim(n * spacing + 0.8, -0.8)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=10)
    
    # Legend
    if show_legend:
        legend_elements = [
            Line2D([0], [0], color=match_color, linewidth=2,
                   marker='>', markersize=5, label='match'),            
            Line2D([0], [0], color=mismatch_color, linewidth=2,
                   marker='>', markersize=5, label='mismatch'),
            Line2D([0], [0], color=gap_color, linewidth=2,
                   marker='>', markersize=5, label='gap'),
            Line2D([0], [0], color=leader_edge_color, linewidth=2.5,
                   marker='>', markersize=6, label='EP (from leader)'),
            Line2D([0], [0], color=closer_edge_color, linewidth=2.5,
                   marker='>', markersize=6, label='EP (to closer)'),
            Line2D([], [], color='none', label=''),
        ]
        
        ax.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=2, fontsize=legend_fontsize,
            frameon=True, bbox_to_anchor=legend_bbox_to_anchor,
            fancybox=True, edgecolor='#cccccc', facecolor='white',
        )
    
    return fig


# =============================================================================
# AFFINE-GAP K_{3,3} DAG
# =============================================================================

def draw_affine_k33_dag(
    seq1: str,
    seq2: str,
    match: float = 5,
    mismatch: float = -5,
    spacing: float = 1.0,
    gadget_radius: float = 0.35,
    inner_radius_scale: float = 0.65,
    port_marker_size: float = 5.0,
    marker_edge_width: float = 1.0,
    circle_alpha: float = 0.5,
    circle_linewidth: float = 1.0,
    edge_linewidth: float = 1.0,
    edge_alpha: float = 0.9,
    internal_linewidth: float = 0.7,
    internal_alpha: float = 0.9,
    mutation_scale: float = 10.0,
    edge_shrink_factor: float = 0.15,
    title: Optional[str] = None,
    title_fs: int = 14,
    letter_fs: int = 10,
    legend_fs: int = 12,
    show_port_labels: bool = False,
    show_coord_labels: bool = False,
    show_diag_scores: bool = False,
    show_legend: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    nt_color_map: Optional[Dict[str, str]] = None,
    max_nucleotide_x: Optional[int] = None,
    max_nucleotide_y: Optional[int] = None,
) -> plt.Figure:
    """
    Draw a clean affine-gap DAG using K_{3,3} gadgets with arrow-style edges.

    Each cell (i,j) is shown as a dashed circle with six ports:
      - Incoming (white circles): H_in, D_in, V_in
      - Outgoing (colored squares): H_out, D_out, V_out

    Parameters
    ----------
    seq1, seq2 : str
        Sequences (seq1 = rows/X, seq2 = columns/Y).
    match, mismatch : float
        Substitution scores for diagonal edges.
    spacing : float
        Distance between adjacent cell centers.
    gadget_radius : float
        Radius of the dashed circle representing each K_{3,3} gadget.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """    
    if nt_color_map is None:
        nt_color_map = NT_COLOR

    if max_nucleotide_x is None:
        max_nucleotide_x = len(seq1)
    if max_nucleotide_y is None:
        max_nucleotide_y = len(seq2)

    C_edge = EDGE_COLORS
    C_port = PORT_COLORS

    m, n = len(seq1), len(seq2)

    if ax is None:
        if figsize is None:
            fig_w = max(6, (n + 2) * spacing * 1.2)
            fig_h = max(5, (m + 2) * spacing * 1.2)
            if show_legend:
                fig_h += 0.8
            figsize = (fig_w, fig_h)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if title is None:
        title = f"Affine-gap DAG ($K_{{3,3}}$ gadgets)\nX={seq1}  Y={seq2}"
    ax.set_title(title, fontsize=title_fs)

    centers: Dict[Tuple[int, int], Tuple[float, float]] = {}
    ports: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]] = {}

    port_angles = dict(
        H_in=180.0, D_in=225.0, V_in=270.0,
        H_out=0.0, D_out=45.0, V_out=90.0
    )

    # First pass: compute all centers and ports
    for i in range(m + 1):
        for j in range(n + 1):
            cx, cy = j * spacing, i * spacing
            centers[(i, j)] = (cx, cy)

            ri = inner_radius_scale * gadget_radius
            p = {name: polar(cx, cy, ri, deg) for name, deg in port_angles.items()}
            ports[(i, j)] = p

    # Draw solid circles
    for i in range(m + 1):
        for j in range(n + 1):
            cx, cy = centers[(i, j)]
            circle = plt.Circle(
                (cx, cy), gadget_radius, fill=False,
                linestyle="solid", linewidth=circle_linewidth,
                edgecolor="gray", alpha=circle_alpha
            )
            ax.add_patch(circle)

            if show_coord_labels:
                ax.text(
                    cx + gadget_radius * 0.8, cy - gadget_radius * 0.8,
                    f"({i},{j})", ha="left", va="top", fontsize=7, color="gray"
                )

    def _draw_arrow(p0, p1, color, lw, alpha, shrink_start=True, shrink_end=True,
                    shrink_factor=None, mut_scale=None, zorder=2,
                    head_width=0.4, head_length=0.6):
        x0, y0 = p0
        x1, y1 = p1
        v = np.array([x1 - x0, y1 - y0], dtype=float)
        L = np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0
        sf = shrink_factor if shrink_factor is not None else edge_shrink_factor
        shrink = sf * gadget_radius

        if shrink_start:
            start = (x0 + v[0] * (shrink / L), y0 + v[1] * (shrink / L))
        else:
            start = p0
        if shrink_end:
            end = (x1 - v[0] * (shrink / L), y1 - v[1] * (shrink / L))
        else:
            end = p1

        ms = mut_scale if mut_scale is not None else mutation_scale
        style = f"->,head_width={head_width},head_length={head_length}"
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle=style,
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            alpha=alpha,
            zorder=zorder
        )
        ax.add_patch(arrow)

    # Draw internal edges
    for i in range(m + 1):
        for j in range(n + 1):
            p = ports[(i, j)]
            
            valid_in = []
            if j > 0:
                valid_in.append(("H", p["H_in"]))
            if i > 0 and j > 0:
                valid_in.append(("D", p["D_in"]))
            elif i == 0 and j == 0:
                valid_in.append(("D", p["D_in"]))
            if i > 0:
                valid_in.append(("V", p["V_in"]))
            
            valid_out = []
            if j < n:
                valid_out.append(("H", p["H_out"]))
            if i < m and j < n:
                valid_out.append(("D", p["D_out"]))
            elif i == m and j == n:
                valid_out.append(("D", p["D_out"]))
            if i < m:
                valid_out.append(("V", p["V_out"]))

            for (iname, (ix, iy)) in valid_in:
                for (oname, (ox, oy)) in valid_out:
                    if oname == "D":
                        color = C_edge["zero"]
                    elif oname == "H":
                        color = C_edge["zero"] if iname == "H" else C_edge["gap_initiate"]
                    else:
                        color = C_edge["zero"] if iname == "V" else C_edge["gap_initiate"]

                    ax.plot([ix, ox], [iy, oy], color=color,
                            linewidth=internal_linewidth, alpha=internal_alpha,
                            zorder=1)

    # Draw inter-cell arrows
    for i in range(m + 1):
        for j in range(n + 1):
            if j + 1 <= n:
                p0 = ports[(i, j)]["H_out"]
                p1 = ports[(i, j + 1)]["H_in"]
                _draw_arrow(p0, p1, C_edge["gap_extend"], edge_linewidth, edge_alpha,
                            zorder=2)

            if i + 1 <= m:
                p0 = ports[(i, j)]["V_out"]
                p1 = ports[(i + 1, j)]["V_in"]
                _draw_arrow(p0, p1, C_edge["gap_extend"], edge_linewidth, edge_alpha,
                            zorder=2)

            if i + 1 <= m and j + 1 <= n:
                p0 = ports[(i, j)]["D_out"]
                p1 = ports[(i + 1, j + 1)]["D_in"]
                w = match if seq1[i] == seq2[j] else mismatch
                color = C_edge["match"] if w > 0 else C_edge["mismatch"]
                _draw_arrow(p0, p1, color, edge_linewidth, edge_alpha, zorder=2)

                if show_diag_scores:
                    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
                    ax.text(mx + 0.08, my - 0.08, str(int(w)),
                            fontsize=8, ha="center", va="center")

    # Draw port markers
    for i in range(m + 1):
        for j in range(n + 1):
            p = ports[(i, j)]

            for name in ["H_in", "D_in", "V_in"]:
                if name == "H_in" and j == 0:
                    continue
                if name == "D_in" and (i == 0 or j == 0) and not (i == 0 and j == 0):
                    continue
                if name == "V_in" and i == 0:
                    continue
                    
                px, py = p[name]
                layer = name[0]
                ax.plot(px, py, marker="o", markersize=port_marker_size,
                        markerfacecolor=C_port[layer], markeredgecolor=C_port[layer],
                        markeredgewidth=marker_edge_width, zorder=3)

            for name in ["H_out", "D_out", "V_out"]:
                if name == "H_out" and j == n:
                    continue
                if name == "D_out" and (i == m or j == n) and not (i == m and j == n):
                    continue
                if name == "V_out" and i == m:
                    continue
                    
                px, py = p[name]
                layer = name[0]
                ax.plot(px, py, marker="s", markersize=port_marker_size,
                        markerfacecolor="white", markeredgecolor=C_port[layer],
                        markeredgewidth=marker_edge_width, zorder=3)

            if show_port_labels:
                for name, (px, py) in p.items():
                    cx, cy = centers[(i, j)]
                    dx, dy = px - cx, py - cy
                    lx = px + dx * 0.4
                    ly = py + dy * 0.4
                    ax.text(lx, ly, name.replace("_", "\n"),
                            fontsize=6, ha="center", va="center", color="gray")

    # Sequence letters
    for idx, base in enumerate(seq1[:max_nucleotide_x], start=1):
        x_curr, y_curr = centers[(idx, 0)]
        x_prev, y_prev = centers[(idx - 1, 0)]
        mid_y = (y_curr + y_prev) / 2
        color = nt_color_map.get(base, "black")
        ax.text(x_curr - spacing * 0.5, mid_y, base,
                ha="right", va="center", fontsize=letter_fs,
                fontweight="bold", color=color)

    for idx, base in enumerate(seq2[:max_nucleotide_y], start=1):
        x_curr, y_curr = centers[(0, idx)]
        x_prev, y_prev = centers[(0, idx - 1)]
        mid_x = (x_curr + x_prev) / 2
        color = nt_color_map.get(base, "black")
        ax.text(mid_x, y_curr - spacing * 0.5, base,
                ha="center", va="bottom", fontsize=letter_fs,
                fontweight="bold", color=color)

    margin = spacing * 0.8
    ax.set_xlim(-margin, n * spacing + margin)
    ax.set_ylim(m * spacing + margin, -margin)

    if show_legend:
        h_match = Line2D([0], [0], color=C_edge["match"], lw=1, marker=">",
                         markersize=4, label="match")
        h_mismatch = Line2D([0], [0], color=C_edge["mismatch"], lw=1, marker=">",
                            markersize=4, label="mismatch")
        h_gap_extend = Line2D([0], [0], color=C_edge["gap_extend"], lw=1, marker=">",
                              markersize=4, label="gap-extend")
        h_gap_init = Line2D([0], [0], color=C_edge["gap_initiate"], lw=1, marker=">",
                            markersize=4, label="gap-initiate")
        h_zero = Line2D([0], [0], color=C_edge["zero"], lw=1, marker=">",
                        markersize=4, label="zero-cost")
        h_spacer = Line2D([0], [0], color="none", lw=0, label="")

        h_in_H = Line2D([0], [0], marker="o", markersize=6, linestyle="",
                        markerfacecolor=C_port["H"], markeredgecolor=C_port["H"],
                        markeredgewidth=1.2, label="in: Yg (H)")
        h_in_D = Line2D([0], [0], marker="o", markersize=6, linestyle="",
                        markerfacecolor=C_port["D"], markeredgecolor=C_port["D"],
                        markeredgewidth=1.2, label="in: M (D)")
        h_in_V = Line2D([0], [0], marker="o", markersize=6, linestyle="",
                        markerfacecolor=C_port["V"], markeredgecolor=C_port["V"],
                        markeredgewidth=1.2, label="in: Xg (V)")

        h_out_H = Line2D([0], [0], marker="s", markersize=6, linestyle="",
                         markerfacecolor="white", markeredgecolor=C_port["H"],
                         markeredgewidth=1.2, label="out: Yg (H)")
        h_out_D = Line2D([0], [0], marker="s", markersize=6, linestyle="",
                         markerfacecolor="white", markeredgecolor=C_port["D"],
                         markeredgewidth=1.2, label="out: M (D)")
        h_out_V = Line2D([0], [0], marker="s", markersize=6, linestyle="",
                         markerfacecolor="white", markeredgecolor=C_port["V"],
                         markeredgewidth=1.2, label="out: Xg (V)")

        legend_handles = [
            h_match, h_mismatch, h_gap_extend,
            h_gap_init, h_zero, h_spacer,
            h_in_H, h_in_D, h_in_V,
            h_out_H, h_out_D, h_out_V,
        ]

        legend = ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            fontsize=legend_fs,
            frameon=True,
            edgecolor="black",
            fancybox=False,
            borderpad=0.6,
        )
        legend.get_frame().set_linewidth(1.0)

    return fig


# =============================================================================
# FLEX GADGET VISUALIZATION
# =============================================================================

def draw_flex_gadget(
    ax=None,
    figsize=(6, 6),
    node_radius=0.3,
    inner_radius_scale=0.55,
    show_legend=True,
):
    """
    Draw a single NW-flex gadget with EP (extra predecessor) structure.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    figsize : tuple, optional
        Figure size if creating new figure.
    node_radius : float, optional
        Radius of the central node circle.
    inner_radius_scale : float, optional
        Scale factor for port distance from center.
    show_legend : bool, optional
        Whether to show the legend.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    C_port = PORT_COLORS
    C_edge = EDGE_COLORS
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    cx, cy = 0.5, 0.5
    
    central_node = Circle((cx, cy), node_radius, facecolor="white",
                          edgecolor="black", linewidth=1.5, zorder=0)
    ax.add_patch(central_node)
    
    port_angles_in = {"H": 180, "D": 225, "V": 270}
    port_angles_out = {"H": 0, "D": 45, "V": 90}
    max_angles = {"H": 180 - 22.5, "D": 225 - 22.5, "V": 270 - 22.5}
    ep_angles = {"H": 180 - 22.5, "D": 225 - 22.5, "V": 270 - 22.5}
    
    r_port = node_radius
    r_max = node_radius * 0.65
    r_ep = node_radius * 1.35
    
    port_size = 0.035
    max_size = 0.04
    ep_size = 0.03
    
    def port_pos(angle_deg, radius):
        return polar(cx, cy, radius, angle_deg)
    
    pos_in = {k: port_pos(v, r_port) for k, v in port_angles_in.items()}
    pos_out = {k: port_pos(v, r_port) for k, v in port_angles_out.items()}
    pos_max = {k: port_pos(v, r_max) for k, v in max_angles.items()}
    pos_ep = {k: port_pos(v, r_ep) for k, v in ep_angles.items()}
    
    def draw_edge(p1, p2, color, linestyle="-", alpha=0.8, linewidth=1.5):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=2)
    
    gold = C_edge["gap_initiate"]
    gray = C_edge["zero"]
    
    draw_edge(pos_in["H"], pos_out["H"], gray)
    draw_edge(pos_in["V"], pos_out["H"], gold)
    draw_edge(pos_in["D"], pos_out["H"], gold)
    
    for k in ["H", "V", "D"]:
        draw_edge(pos_in[k], pos_max[k], gray)
    
    for k in ["H", "V", "D"]:
        draw_edge(pos_ep[k], pos_max[k], gray)
    
    draw_edge(pos_max["H"], pos_out["V"], gold)
    draw_edge(pos_max["V"], pos_out["V"], gray)
    draw_edge(pos_max["D"], pos_out["V"], gold)
    
    draw_edge(pos_max["H"], pos_out["D"], gray)
    draw_edge(pos_max["V"], pos_out["D"], gray)
    draw_edge(pos_max["D"], pos_out["D"], gray)
    
    for k, (px, py) in pos_in.items():
        port = Circle((px, py), port_size, facecolor="white",
                      edgecolor=C_port[k], linewidth=1.5, zorder=10)
        ax.add_patch(port)
    
    for k, (px, py) in pos_out.items():
        port = plt.Rectangle((px - port_size, py - port_size), 
                             port_size * 2, port_size * 2,
                             facecolor=C_port[k], edgecolor=C_port[k],
                             linewidth=1.0, zorder=10)
        ax.add_patch(port)
    
    for k, (px, py) in pos_max.items():
        tri_patch = RegularPolygon((px, py), numVertices=3, radius=max_size,
                                   facecolor=C_port[k], edgecolor="black",
                                   linewidth=0.8, zorder=10)
        ax.add_patch(tri_patch)
    
    for k, (px, py) in pos_ep.items():
        port = Circle((px, py), ep_size, facecolor="white",
                      edgecolor=C_port[k], linewidth=1.5, linestyle="--", zorder=10)
        ax.add_patch(port)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1.1, -0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    
    if show_legend:
        h_spacer = Line2D([0], [0], marker="", linestyle="", label="")
        
        h_zero = Line2D([0], [0], color=C_edge["zero"], linewidth=2, label="zero-cost")
        h_gold = Line2D([0], [0], color=C_edge["gap_initiate"], linewidth=2, label="gap-initiate")
        
        h_in_H = Line2D([0], [0], marker="o", markersize=6, linestyle="",
                        markerfacecolor="white", markeredgecolor=C_port["H"],
                        markeredgewidth=1.5, label="in: Yg")
        h_in_D = Line2D([0], [0], marker="o", markersize=6, linestyle="",
                        markerfacecolor="white", markeredgecolor=C_port["D"],
                        markeredgewidth=1.5, label="in: M")
        h_in_V = Line2D([0], [0], marker="o", markersize=6, linestyle="",
                        markerfacecolor="white", markeredgecolor=C_port["V"],
                        markeredgewidth=1.5, label="in: Xg")
        
        h_ep_H = Line2D([0], [0], marker="o", markersize=6, linestyle="",
                        markerfacecolor="white", markeredgecolor=C_port["H"],
                        markeredgewidth=1.5, label="EP: Yg")
        h_ep_D = Line2D([0], [0], marker="o", markersize=6, linestyle="",
                        markerfacecolor="white", markeredgecolor=C_port["D"],
                        markeredgewidth=1.5, label="EP: M")
        h_ep_V = Line2D([0], [0], marker="o", markersize=6, linestyle="",
                        markerfacecolor="white", markeredgecolor=C_port["V"],
                        markeredgewidth=1.5, label="EP: Xg")
        
        h_max_H = Line2D([0], [0], marker="^", markersize=8, linestyle="",
                         markerfacecolor=C_port["H"], markeredgecolor="black",
                         markeredgewidth=0.8, label="max: Yg")
        h_max_D = Line2D([0], [0], marker="^", markersize=8, linestyle="",
                         markerfacecolor=C_port["D"], markeredgecolor="black",
                         markeredgewidth=0.8, label="max: M")
        h_max_V = Line2D([0], [0], marker="^", markersize=8, linestyle="",
                         markerfacecolor=C_port["V"], markeredgecolor="black",
                         markeredgewidth=0.8, label="max: Xg")
        
        h_out_H = Line2D([0], [0], marker="s", markersize=6, linestyle="",
                         markerfacecolor=C_port["H"], markeredgecolor=C_port["H"],
                         markeredgewidth=1.0, label="out: Yg")
        h_out_D = Line2D([0], [0], marker="s", markersize=6, linestyle="",
                         markerfacecolor=C_port["D"], markeredgecolor=C_port["D"],
                         markeredgewidth=1.0, label="out: M")
        h_out_V = Line2D([0], [0], marker="s", markersize=6, linestyle="",
                         markerfacecolor=C_port["V"], markeredgecolor=C_port["V"],
                         markeredgewidth=1.0, label="out: Xg")
        
        legend_handles = [
            h_zero,   h_in_H, h_ep_H, h_max_H, h_out_H,
            h_gold,   h_in_D, h_ep_D, h_max_D, h_out_D,
            h_spacer, h_in_V, h_ep_V, h_max_V, h_out_V,
        ]
        
        legend = ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=5,
            fontsize=9,
            frameon=True,
            edgecolor="black",
            fancybox=False,
            borderpad=0.6,
            columnspacing=0.8,
            handletextpad=0.4,
        )
        legend.get_frame().set_linewidth(1.0)
    
    fig.tight_layout()
    return fig


def draw_flex_dag(
    seq1: str,
    seq2: str,
    match: float = 5,
    mismatch: float = -5,
    spacing: float = 1.0,
    gadget_radius: float = 0.35,
    inner_radius_scale: float = 0.65,
    max_radius_scale: float = 0.35,
    port_marker_size: float = 5.0,
    marker_edge_width: float = 1.0,
    max_marker_size: float = 0.045,
    ep_marker_size: float = 4.0,
    ep_radius_scale: float = 1.0,
    circle_alpha: float = 0.5,
    circle_linewidth: float = 1.0,
    edge_linewidth: float = 1.0,
    edge_alpha: float = 0.9,
    lw_internal: float = 0.7,
    internal_alpha: float = 0.9,
    mutation_scale: float = 10.0,
    edge_shrink_factor: float = 0.15,
    show_legend: bool = True,
    show_internal: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    nt_color_map: Optional[Dict[str, str]] = None,
    ax: Optional[plt.Axes] = None,
    letter_fs: int = 12,
    title_fs: int = 14,
    legend_fs: int = 9,
    max_nucleotide_x: Optional[int] = None,
    max_nucleotide_y: Optional[int] = None,    
):
    """
    Draw a grid of NW-flex gadgets with EP (extra predecessor) structure.
    
    Parameters
    ----------
    seq1, seq2 : str
        Sequences (seq1 = rows/X, seq2 = columns/Y).
    match, mismatch : float
        Substitution scores.
    spacing : float
        Distance between gadget centers.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if max_nucleotide_x is None:
        max_nucleotide_x = len(seq1)
    if max_nucleotide_y is None:
        max_nucleotide_y = len(seq2)
    
    if nt_color_map is None:
        nt_color_map = NT_COLOR
    
    C_port = PORT_COLORS
    C_edge = EDGE_COLORS
    
    m, n = len(seq1), len(seq2)
    
    if ax is None:
        if figsize is None:
            fig_w = max(6, (n + 1) * spacing * 1.8)
            fig_h = max(5, (m + 1) * spacing * 1.8)
            figsize = (fig_w, fig_h)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    ax.set_aspect("equal")
    ax.axis("off")
    
    if title is None:
        title = f"NW-flex DAG (flex gadgets)\nX={seq1}  Y={seq2}"
    if title:
        ax.set_title(title, fontsize=title_fs)
    
    port_angles_in = {"H": 180, "D": 225, "V": 270}
    port_angles_out = {"H": 0, "D": 45, "V": 90}
    max_angles = {"H": 180 - 22.5, "D": 225 - 22.5, "V": 270 - 22.5}
    ep_angles = {"H": 180 - 22.5, "D": 225 - 22.5, "V": 270 - 22.5}
    
    r_port = gadget_radius * inner_radius_scale
    r_max = gadget_radius * max_radius_scale
    r_ep = gadget_radius * ep_radius_scale
    
    centers = {}
    ports_in = {}
    ports_out = {}
    ports_max = {}
    ports_ep = {}
    
    def polar_pos(cx, cy, radius, angle_deg):
        return polar(cx, cy, radius, angle_deg)
    
    def _draw_arrow(p0, p1, color, lw, alpha, shrink_start=True, shrink_end=True,
                    shrink_factor=None, mut_scale=None, zorder=2,
                    head_width=0.4, head_length=0.6):
        x0, y0 = p0
        x1, y1 = p1
        v = np.array([x1 - x0, y1 - y0], dtype=float)
        L = np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0
        sf = shrink_factor if shrink_factor is not None else edge_shrink_factor
        shrink = sf * gadget_radius

        if shrink_start:
            start = (x0 + v[0] * (shrink / L), y0 + v[1] * (shrink / L))
        else:
            start = p0
        if shrink_end:
            end = (x1 - v[0] * (shrink / L), y1 - v[1] * (shrink / L))
        else:
            end = p1

        ms = mut_scale if mut_scale is not None else mutation_scale
        style = f"->,head_width={head_width},head_length={head_length}"
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle=style,
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            alpha=alpha,
            zorder=zorder
        )
        ax.add_patch(arrow)
    
    for i in range(m + 1):
        for j in range(n + 1):
            cx, cy = j * spacing, i * spacing
            centers[(i, j)] = (cx, cy)
            
            circle = Circle((cx, cy), gadget_radius, fill=False,
                           linestyle="solid", linewidth=circle_linewidth,
                           edgecolor="gray", alpha=circle_alpha,
                           zorder=1)
            ax.add_patch(circle)
            
            p_in = {k: polar_pos(cx, cy, r_port, v) for k, v in port_angles_in.items()}
            p_out = {k: polar_pos(cx, cy, r_port, v) for k, v in port_angles_out.items()}
            p_max = {k: polar_pos(cx, cy, r_max, v) for k, v in max_angles.items()}
            p_ep = {k: polar_pos(cx, cy, r_ep, v) for k, v in ep_angles.items()}
            
            ports_in[(i, j)] = p_in
            ports_out[(i, j)] = p_out
            ports_max[(i, j)] = p_max
            ports_ep[(i, j)] = p_ep
            
            has_in = {"H": j > 0, "D": (i > 0 and j > 0) or (i == 0 and j == 0), "V": i > 0}
            has_out = {"H": j < n, "D": (i < m and j < n) or (i == m and j == n), "V": i < m}
            
            if show_internal:
                gold = C_edge["gap_initiate"]
                gray = C_edge["zero"]
                
                if has_out["H"]:
                    for k in ["H", "V", "D"]:
                        if has_in[k]:
                            color = gray if k == "H" else gold
                            ax.plot([p_in[k][0], p_out["H"][0]], 
                                   [p_in[k][1], p_out["H"][1]],
                                   color=color, linewidth=lw_internal, 
                                   alpha=internal_alpha, zorder=2)
                
                for k in ["H", "V", "D"]:
                    if has_in[k]:
                        ax.plot([p_in[k][0], p_max[k][0]], 
                               [p_in[k][1], p_max[k][1]],
                               color=gray, linewidth=lw_internal, 
                               alpha=internal_alpha, zorder=2)
                
                if i > 0:
                    for k in ["H", "V", "D"]:
                        if has_in[k]:
                            ax.plot([p_ep[k][0], p_max[k][0]], 
                                   [p_ep[k][1], p_max[k][1]],
                                   color=gray, linewidth=lw_internal, 
                                   alpha=internal_alpha, zorder=2)
                
                if has_out["V"]:
                    for k in ["H", "V", "D"]:
                        if has_in[k]:
                            color = gray if k == "V" else gold
                            ax.plot([p_max[k][0], p_out["V"][0]], 
                                   [p_max[k][1], p_out["V"][1]],
                                   color=color, linewidth=lw_internal, 
                                   alpha=internal_alpha, zorder=2)
                
                if has_out["D"]:
                    for k in ["H", "V", "D"]:
                        if has_in[k]:
                            ax.plot([p_max[k][0], p_out["D"][0]], 
                                   [p_max[k][1], p_out["D"][1]],
                                   color=gray, linewidth=lw_internal, 
                                   alpha=internal_alpha, zorder=2)
            
            for k, (px, py) in p_in.items():
                if has_in[k]:
                    ax.plot(px, py, marker="o", markersize=port_marker_size,
                           markerfacecolor=C_port[k], markeredgecolor=C_port[k],
                           markeredgewidth=marker_edge_width, zorder=10)
            
            for k, (px, py) in p_out.items():
                if has_out[k]:
                    ax.plot(px, py, marker="s", markersize=port_marker_size,
                           markerfacecolor="white", markeredgecolor=C_port[k],
                           markeredgewidth=marker_edge_width, zorder=10)
            
            for k, (px, py) in p_max.items():
                if has_in[k]:
                    tri_patch = RegularPolygon((px, py), numVertices=3, 
                                              radius=max_marker_size,
                                              orientation=np.pi,
                                              facecolor="white", edgecolor=C_port[k],
                                              linewidth=marker_edge_width, zorder=10)
                    ax.add_patch(tri_patch)
            
            if i > 0:
                for k, (px, py) in p_ep.items():
                    if has_in[k]:
                        ax.plot(px, py, marker="o", markersize=port_marker_size,
                               markerfacecolor="white", markeredgecolor=C_port[k],
                               markeredgewidth=marker_edge_width, zorder=10)
    
    for i in range(m + 1):
        for j in range(n + 1):
            if j + 1 <= n:
                p1 = ports_out[(i, j)]["H"]
                p2 = ports_in[(i, j + 1)]["H"]
                _draw_arrow(p1, p2, C_edge["gap_extend"], edge_linewidth, edge_alpha,
                           zorder=3)
            
            if i + 1 <= m:
                p1 = ports_out[(i, j)]["V"]
                p2 = ports_in[(i + 1, j)]["V"]
                _draw_arrow(p1, p2, C_edge["gap_extend"], edge_linewidth, edge_alpha,
                           zorder=3)
            
            if i + 1 <= m and j + 1 <= n:
                p1 = ports_out[(i, j)]["D"]
                p2 = ports_in[(i + 1, j + 1)]["D"]
                is_match = seq1[i] == seq2[j]
                color = C_edge["match"] if is_match else C_edge["mismatch"]
                _draw_arrow(p1, p2, color, edge_linewidth, edge_alpha, zorder=3)
    
    for j in range(max_nucleotide_y):
        x1, _ = centers[(0, j)]
        x2, _ = centers[(0, j + 1)]
        mx = (x1 + x2) / 2
        my = -gadget_radius - 0.15
        color = nt_color_map.get(seq2[j], "black")
        ax.text(mx, my, seq2[j], fontsize=letter_fs, ha="center", va="bottom",
               color=color, fontweight="bold")
    
    for i in range(max_nucleotide_x):
        _, y1 = centers[(i, 0)]
        _, y2 = centers[(i + 1, 0)]
        mx = -gadget_radius - 0.15
        my = (y1 + y2) / 2
        color = nt_color_map.get(seq1[i], "black")
        ax.text(mx, my, seq1[i], fontsize=letter_fs, ha="right", va="center",
               color=color, fontweight="bold")
    
    ax.set_xlim(-gadget_radius - 0.5, n * spacing + gadget_radius + 0.3)
    ax.set_ylim(-gadget_radius - 0.5, m * spacing + gadget_radius + 0.3)
    ax.invert_yaxis()
    
    if show_legend:
        h_spacer = Line2D([0], [0], marker="", linestyle="", label="")
        
        h_match = Line2D([0], [0], color=C_edge["match"], lw=1, marker=">",
                        markersize=4, label="match")
        h_mismatch = Line2D([0], [0], color=C_edge["mismatch"], lw=1, marker=">",
                           markersize=4, label="mismatch")
        h_extend = Line2D([0], [0], color=C_edge["gap_extend"], lw=1, marker=">",
                         markersize=4, label="gap-extend")
        h_init = Line2D([0], [0], color=C_edge["gap_initiate"], lw=1, marker=">",
                       markersize=4, label="gap-initiate")
        h_zero = Line2D([0], [0], color=C_edge["zero"], lw=1, marker=">",
                       markersize=4, label="zero-cost")
        
        h_in = Line2D([0], [0], marker="o", markersize=6, linestyle="",
                     markerfacecolor="gray", markeredgecolor="gray",
                     markeredgewidth=1, label="in-port")
        h_ep = Line2D([0], [0], marker="o", markersize=6, linestyle="",
                     markerfacecolor="white", markeredgecolor="gray",
                     markeredgewidth=1, label="EP-port")
        h_max = Line2D([0], [0], marker="^", markersize=8, linestyle="",
                      markerfacecolor="white", markeredgecolor="gray",
                      markeredgewidth=1, label="max")
        h_out = Line2D([0], [0], marker="s", markersize=6, linestyle="",
                      markerfacecolor="white", markeredgecolor="gray",
                      markeredgewidth=1, label="out-port")
        
        legend_handles = [
            h_match, h_mismatch, h_extend, h_init, h_zero,
            h_in, h_ep, h_max, h_out,
        ]
        
        legend = ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=5,
            fontsize=legend_fs,
            frameon=True,
            edgecolor="black",
            fancybox=False,
            borderpad=0.6,
        )
        legend.get_frame().set_linewidth(1.0)
    
    return fig

