"""
Shared utilities for NW-flex plotting functions.

This module contains helper functions used across multiple plotting modules:
- Coordinate transformations
- Arrow drawing helpers
- Legend building utilities
- Nucleotide label coloring
"""

import math
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle


# =============================================================================
# COORDINATE TRANSFORMATIONS
# =============================================================================

def polar(cx: float, cy: float, r: float, deg: float) -> Tuple[float, float]:
    """
    Convert polar to Cartesian coordinates.
    
    Parameters
    ----------
    cx, cy : float
        Center coordinates
    r : float
        Radius
    deg : float
        Angle in degrees
        
    Returns
    -------
    (x, y) : tuple of float
        Cartesian coordinates
    """
    rad = math.radians(deg)
    return (cx + r * math.cos(rad), cy + r * math.sin(rad))


# =============================================================================
# ARROW DRAWING HELPERS
# =============================================================================

def draw_shortened_arrow(
    ax: plt.Axes,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    color: str,
    node_radius: float,
    edge_shrink_factor: float = 1.2,
    mutation_scale: float = 10.0,
    linewidth: float = 1.0,
    alpha: float = 0.7,
    arrowstyle: str = "->",
    zorder: int = 2,
) -> FancyArrowPatch:
    """
    Draw an arrow between two points, shortened to avoid overlapping nodes.
    
    Parameters
    ----------
    ax : matplotlib Axes
        Axes to draw on
    p0 : tuple
        Start coordinates (x0, y0)
    p1 : tuple
        End coordinates (x1, y1)
    color : str
        Arrow color
    node_radius : float
        Radius of nodes (used to calculate shrink distance)
    edge_shrink_factor : float
        Multiplier for shrink distance (shrink = node_radius * edge_shrink_factor)
    mutation_scale : float
        Arrow head size
    linewidth : float
        Line width
    alpha : float
        Transparency
    arrowstyle : str
        Arrow style (default "->")
    zorder : int
        Drawing order
        
    Returns
    -------
    arrow : FancyArrowPatch
        The created arrow patch
    """
    x0, y0 = p0
    x1, y1 = p1
    v = np.array([x1 - x0, y1 - y0], dtype=float)
    L = np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0
    shrink = node_radius * edge_shrink_factor
    
    start = (x0 + v[0] * (shrink / L), y0 + v[1] * (shrink / L))
    end = (x1 - v[0] * (shrink / L), y1 - v[1] * (shrink / L))
    
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=arrowstyle,
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def draw_highlighted_edge(
    ax: plt.Axes,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    color: str,
    node_radius: float,
    edge_shrink_factor: float = 1.2,
    mutation_scale: float = 12.0,
    score: Optional[float] = None,
    score_fontsize: int = 8,
) -> None:
    """
    Draw a highlighted edge with optional score label at midpoint.
    
    Parameters
    ----------
    ax : matplotlib Axes
        Axes to draw on
    p0 : tuple
        Start coordinates
    p1 : tuple
        End coordinates
    color : str
        Edge color
    node_radius : float
        Node radius for shrink calculation
    edge_shrink_factor : float
        Shrink multiplier
    mutation_scale : float
        Arrow head size
    score : float, optional
        If provided, display at midpoint
    score_fontsize : int
        Font size for score label
    """
    x0, y0 = p0
    x1, y1 = p1
    v = np.array([x1 - x0, y1 - y0], dtype=float)
    L = np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0
    shrink = node_radius * edge_shrink_factor
    
    start = (x0 + v[0] * (shrink / L), y0 + v[1] * (shrink / L))
    end = (x1 - v[0] * (shrink / L), y1 - v[1] * (shrink / L))

    arrow = FancyArrowPatch(
        start, end, arrowstyle="->", mutation_scale=mutation_scale,
        linewidth=2.5, color=color, alpha=0.9
    )
    ax.add_patch(arrow)

    if score is not None:
        midx, midy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(midx, midy, f"{int(score)}", ha="center", va="center",
                fontsize=score_fontsize, color="black", fontweight="bold")


# =============================================================================
# NODE DRAWING HELPERS
# =============================================================================

def draw_node_with_score(
    ax: plt.Axes,
    x: float,
    y: float,
    score: float,
    node_radius: float = 0.25,
    facecolor: str = "#d0ffd0",
    edgecolor: str = "#00ff2f",
    linewidth: float = 2.0,
    alpha: float = 0.8,
    fontsize: int = 9,
    fontweight: str = "bold",
) -> None:
    """
    Draw a filled node with a score label.
    
    Parameters
    ----------
    ax : matplotlib Axes
        Axes to draw on
    x, y : float
        Node center coordinates
    score : float
        Score to display
    node_radius : float
        Node radius
    facecolor : str
        Fill color
    edgecolor : str
        Border color
    linewidth : float
        Border width
    alpha : float
        Transparency
    fontsize : int
        Font size for score
    fontweight : str
        Font weight for score
    """
    circ = Circle(
        (x, y), radius=node_radius * 0.9,
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, alpha=alpha
    )
    ax.add_patch(circ)
    ax.text(x, y, str(int(score)), ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight)


# =============================================================================
# PATH UTILITIES
# =============================================================================

def path_to_alignment(
    seq1: str,
    seq2: str,
    path: list,
) -> Tuple[str, str]:
    """
    Convert a DAG path to aligned sequences.

    Parameters
    ----------
    seq1, seq2 : str
        The original sequences.
    path : list of (i, j) tuples or (i, j, state) tuples
        Path through the DP grid from (0,0) to (n,m).

    Returns
    -------
    align1, align2 : str
        Aligned sequences with gap characters '-'.
    """
    align1, align2 = [], []
    for k in range(1, len(path)):
        # Handle both (i, j) and (i, j, state) tuples
        if len(path[k]) == 2:
            i_prev, j_prev = path[k - 1]
            i_curr, j_curr = path[k]
        else:
            i_prev, j_prev, _ = path[k - 1]
            i_curr, j_curr, _ = path[k]
            
        di, dj = i_curr - i_prev, j_curr - j_prev
        
        if di == 1 and dj == 1:
            # Diagonal: match/mismatch
            align1.append(seq1[i_prev])
            align2.append(seq2[j_prev])
        elif di == 1 and dj == 0:
            # Vertical: gap in seq2
            align1.append(seq1[i_prev])
            align2.append("-")
        elif di == 0 and dj == 1:
            # Horizontal: gap in seq1
            align1.append("-")
            align2.append(seq2[j_prev])
    return "".join(align1), "".join(align2)


def score_path(
    path: list,
    seq1: str,
    seq2: str,
    match: float,
    mismatch: float,
    gap: float,
) -> float:
    """
    Compute the total score of a path through the alignment DAG.

    Parameters
    ----------
    path : list of (i, j) tuples or (i, j, state) tuples
        Path through the DP grid.
    seq1, seq2 : str
        The sequences being aligned.
    match, mismatch, gap : float
        Scoring parameters.

    Returns
    -------
    score : float
        Total path score.
    """
    score = 0.0
    for k in range(1, len(path)):
        # Handle both (i, j) and (i, j, state) tuples
        if len(path[k]) == 2:
            i_prev, j_prev = path[k - 1]
            i_curr, j_curr = path[k]
        else:
            i_prev, j_prev, _ = path[k - 1]
            i_curr, j_curr, _ = path[k]
            
        di, dj = i_curr - i_prev, j_curr - j_prev
        
        if di == 1 and dj == 1:
            # Diagonal
            score += match if seq1[i_prev] == seq2[j_prev] else mismatch
        else:
            # Gap (vertical or horizontal)
            score += gap
    return score
