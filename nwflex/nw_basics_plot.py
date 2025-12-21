"""
Generalized NW DAG visualization for basic Needleman-Wunsch.

This module provides a unified backbone for all NW DAG visualizations:
- Drawing the grid with optional score values
- Rendering traceback edges (bold with values)
- Supporting multiple traceback edges
- Filling individual nodes with scores

Usage:
    plotter = NWDagPlotter("GATT", "GACT", match=5, mismatch=-5, gap=-3)
    fig, ax = plt.subplots()
    coords = plotter.draw_grid(ax, scores=F_matrix)
    plotter.draw_traceback_edges(ax, edges=[((0,0), (1,1)), ((1,1), (2,2))])
    plotter.fill_node(ax, 1, 1, score=5, style="update")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

from .plot.colors import NT_COLOR
from .plot.utils import path_to_alignment


@dataclass
class NWDagStyle:
    """Configuration for DAG drawing styles.
    
    Attributes
    ----------
    node_radius : float
        Radius of node circles.
    spacing : float
        Grid cell spacing.
    edge_shrink_factor : float
        Factor to shrink edges from node centers (prevents overlap with nodes).
    mutation_scale : float
        Arrow head size.
    edge_linewidth : float
        Line width for background edges.
    traceback_linewidth : float
        Line width for highlighted traceback edges.
    edge_alpha : float
        Alpha for background edges.
    traceback_alpha : float
        Alpha for traceback edges.
    match_color : str
        Color for match (diagonal) edges with positive score.
    mismatch_color : str
        Color for mismatch (diagonal) edges with negative score.
    gap_color : str
        Color for gap (horizontal/vertical) edges.
    empty_node_fc : str
        Facecolor for empty (unfilled) nodes.
    empty_node_ec : str
        Edgecolor for empty nodes.
    scored_node_fc : str
        Facecolor for nodes with computed scores.
    scored_node_ec : str
        Edgecolor for scored nodes.
    update_node_fc : str
        Facecolor for the cell being updated (gold highlight).
    update_node_ec : str
        Edgecolor for the cell being updated.
    nt_colors : dict
        Mapping of nucleotides to colors for axis labels.
    """
    node_radius: float = 0.25
    spacing: float = 1.0
    edge_shrink_factor: float = 1.2
    mutation_scale: float = 12.0
    edge_linewidth: float = 1.0
    traceback_linewidth: float = 2.5
    edge_alpha: float = 0.3
    traceback_alpha: float = 0.9
    
    # Edge colors
    match_color: str = "green"
    mismatch_color: str = "red"
    gap_color: str = "blue"
    
    # Node styles
    empty_node_fc: str = "white"
    empty_node_ec: str = "black"
    scored_node_fc: str = "white"  # No green fill by default
    scored_node_ec: str = "black"
    update_node_fc: str = "#ffffd0"  # Gold/yellow
    update_node_ec: str = "#ffaa00"  # Orange
    
    # Nucleotide colors
    nt_colors: Dict[str, str] = field(default_factory=lambda: NT_COLOR.copy())


class NWDagPlotter:
    """Generalized NW DAG visualization.
    
    This class provides a unified approach to drawing NW alignment DAGs,
    supporting:
    - Base grid drawing with optional score values
    - Bold traceback edges with score labels
    - Individual node highlighting (scored vs update styles)
    
    Parameters
    ----------
    seq1, seq2 : str
        Sequences to align (seq1 = rows/X, seq2 = columns/Y).
    match : float
        Match score (typically positive).
    mismatch : float
        Mismatch penalty (typically negative).
    gap : float
        Gap penalty (typically negative).
    style : NWDagStyle, optional
        Style configuration. If None, uses defaults.
    
    Examples
    --------
    >>> plotter = NWDagPlotter("GATT", "GACT", match=5, mismatch=-5, gap=-3)
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> coords = plotter.draw_grid(ax)
    >>> plotter.draw_traceback_edges(ax, [((0, 0), (1, 1))])
    >>> plt.show()
    """
    
    def __init__(
        self,
        seq1: str,
        seq2: str,
        match: float = 1,
        mismatch: float = -1,
        gap: float = -1,
        style: Optional[NWDagStyle] = None,
    ):
        self.seq1 = seq1
        self.seq2 = seq2
        self.m = len(seq1)
        self.n = len(seq2)
        self.match = match
        self.mismatch = mismatch
        self.gap = gap
        self.style = style or NWDagStyle()
        
        # Precompute node coordinates
        self._coords: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for i in range(self.m + 1):
            for j in range(self.n + 1):
                self._coords[(i, j)] = (j * self.style.spacing, i * self.style.spacing)
    
    @property
    def coords(self) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """Return the coordinate mapping (i, j) -> (x, y)."""
        return self._coords
    
    def _get_edge_color(self, from_coord: Tuple[int, int], to_coord: Tuple[int, int]) -> str:
        """Determine the edge color based on move type."""
        i1, j1 = from_coord
        i2, j2 = to_coord
        di, dj = i2 - i1, j2 - j1
        
        if di == 1 and dj == 1:
            # Diagonal move - check match/mismatch
            w = self.match if self.seq1[i1] == self.seq2[j1] else self.mismatch
            return self.style.match_color if w > 0 else self.style.mismatch_color
        else:
            # Gap move
            return self.style.gap_color
    
    def _get_edge_score(self, from_coord: Tuple[int, int], to_coord: Tuple[int, int]) -> float:
        """Compute the score contribution of an edge."""
        i1, j1 = from_coord
        i2, j2 = to_coord
        di, dj = i2 - i1, j2 - j1
        
        if di == 1 and dj == 1:
            # Diagonal move
            return self.match if self.seq1[i1] == self.seq2[j1] else self.mismatch
        else:
            # Gap move
            return self.gap
    
    def _draw_arrow(
        self,
        ax: plt.Axes,
        from_coord: Tuple[int, int],
        to_coord: Tuple[int, int],
        color: str,
        linewidth: float,
        alpha: float,
    ) -> None:
        """Draw a shortened arrow between two nodes."""
        x0, y0 = self._coords[from_coord]
        x1, y1 = self._coords[to_coord]
        
        v = np.array([x1 - x0, y1 - y0], dtype=float)
        L = np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0
        shrink = self.style.node_radius * self.style.edge_shrink_factor
        
        start = (x0 + v[0] * (shrink / L), y0 + v[1] * (shrink / L))
        end = (x1 - v[0] * (shrink / L), y1 - v[1] * (shrink / L))
        
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle="->",
            mutation_scale=self.style.mutation_scale,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
        )
        ax.add_patch(arrow)
    
    def draw_grid(
        self,
        ax: plt.Axes,
        scores: Optional[np.ndarray] = None,
        edge_alpha: Optional[float] = None,
        show_edges: bool = True,
    ) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """Draw the base DAG grid with optional score values in nodes.
        
        Parameters
        ----------
        ax : matplotlib Axes
            The axes to draw on.
        scores : ndarray, optional
            Score matrix F[i, j]. If provided, displays scores in nodes.
            Use np.nan for cells without scores yet.
        edge_alpha : float, optional
            Override the default edge alpha from style.
        show_edges : bool
            If True, draw background edges. Set False if only drawing nodes.
        
        Returns
        -------
        coords : dict
            Mapping (i, j) -> (x, y) plot coordinates.
        """
        style = self.style
        alpha = edge_alpha if edge_alpha is not None else style.edge_alpha
        
        # Draw edges (background)
        if show_edges:
            for i in range(self.m + 1):
                for j in range(self.n + 1):
                    # Vertical edge (gap in Y)
                    if i + 1 <= self.m:
                        self._draw_arrow(
                            ax, (i, j), (i + 1, j),
                            color=style.gap_color,
                            linewidth=style.edge_linewidth,
                            alpha=alpha,
                        )
                    # Horizontal edge (gap in X)
                    if j + 1 <= self.n:
                        self._draw_arrow(
                            ax, (i, j), (i, j + 1),
                            color=style.gap_color,
                            linewidth=style.edge_linewidth,
                            alpha=alpha,
                        )
                    # Diagonal edge (match/mismatch)
                    if i + 1 <= self.m and j + 1 <= self.n:
                        color = self._get_edge_color((i, j), (i + 1, j + 1))
                        self._draw_arrow(
                            ax, (i, j), (i + 1, j + 1),
                            color=color,
                            linewidth=style.edge_linewidth,
                            alpha=alpha,
                        )
        
        # Draw nodes
        for (i, j), (x, y) in self._coords.items():
            # Determine if we have a score for this cell
            has_score = (scores is not None and 
                        i < scores.shape[0] and j < scores.shape[1] and
                        np.isfinite(scores[i, j]))
            
            if has_score:
                fc = style.scored_node_fc
                ec = style.scored_node_ec
            else:
                fc = style.empty_node_fc
                ec = style.empty_node_ec
            
            circ = Circle(
                (x, y), radius=style.node_radius,
                facecolor=fc, edgecolor=ec, linewidth=1.0
            )
            ax.add_patch(circ)
            
            # Add score text if available
            if has_score:
                score_val = int(scores[i, j])
                ax.text(x, y, str(score_val), ha="center", va="center",
                       fontsize=9, fontweight="bold")
        
        # Label bases along axes
        nt_colors = style.nt_colors
        spacing = style.spacing
        
        # seq1 (X) along vertical axis (rows)
        for i, base in enumerate(self.seq1, start=1):
            x0, y0 = self._coords[(i, 0)]
            ax.text(
                x0 - 0.5 * spacing, y0 - 0.5 * spacing, base,
                ha="right", va="center",
                color=nt_colors.get(base, "black"),
                fontsize=10, fontweight="bold"
            )
        
        # seq2 (Y) along horizontal axis (columns)
        for j, base in enumerate(self.seq2, start=1):
            x0, y0 = self._coords[(0, j)]
            ax.text(
                x0 - 0.5 * spacing, y0 - 0.5 * spacing, base,
                ha="center", va="bottom",
                color=nt_colors.get(base, "black"),
                fontsize=10, fontweight="bold"
            )
        
        # Configure axes
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-spacing, (self.n + 1) * spacing)
        ax.set_ylim((self.m + 1) * spacing, -spacing)
        
        return self._coords
    
    def draw_traceback_edges(
        self,
        ax: plt.Axes,
        edges: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        show_scores: bool = True,
        linewidth: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> None:
        """Draw bold traceback edges with optional score labels.
        
        Parameters
        ----------
        ax : matplotlib Axes
            The axes to draw on.
        edges : list of ((i1, j1), (i2, j2)) tuples
            List of edges to highlight. Each edge is a tuple of
            (from_coord, to_coord).
        show_scores : bool
            If True, display the edge score at the midpoint.
        linewidth : float, optional
            Override the default traceback linewidth.
        alpha : float, optional
            Override the default traceback alpha.
        """
        style = self.style
        lw = linewidth if linewidth is not None else style.traceback_linewidth
        a = alpha if alpha is not None else style.traceback_alpha
        
        for from_coord, to_coord in edges:
            color = self._get_edge_color(from_coord, to_coord)
            
            # Draw the highlighted arrow
            self._draw_arrow(ax, from_coord, to_coord, color, lw, a)
            
            # Add score label at midpoint
            if show_scores:
                x0, y0 = self._coords[from_coord]
                x1, y1 = self._coords[to_coord]
                midx, midy = (x0 + x1) / 2, (y0 + y1) / 2
                if x0 == x1:
                    midx += 0.15  # Shift up for horizontal edges
                elif y0 == y1:
                    midy -= 0.15  # Shift right for vertical edges
                else:
                    midx += 0.1
                    midy -= 0.1  # Shift for diagonal edges
                score = self._get_edge_score(from_coord, to_coord)
                ax.text(
                    midx, midy, f"{int(score)}",
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color="black"
                )
    
    def fill_node(
        self,
        ax: plt.Axes,
        i: int,
        j: int,
        score: Optional[Union[int, float]] = None,
        style: str = "scored",
    ) -> None:
        """Fill a single node with optional score.
        
        Parameters
        ----------
        ax : matplotlib Axes
            The axes to draw on.
        i, j : int
            Node coordinates.
        score : int or float, optional
            Score to display in the node.
        style : str
            Node style: "scored" for computed cells, "update" for the
            cell being updated (gold highlight).
        """
        x, y = self._coords[(i, j)]
        s = self.style
        
        if style == "update":
            fc = s.update_node_fc
            ec = s.update_node_ec
        else:  # "scored"
            fc = s.scored_node_fc
            ec = s.scored_node_ec
        
        # Draw filled circle (slightly smaller to overlay on existing node)
        circ = Circle(
            (x, y), radius=s.node_radius,
            facecolor=fc, edgecolor=ec, linewidth=1, alpha=0.9
        )
        ax.add_patch(circ)
        
        # Add score text
        if score is not None:
            ax.text(x, y, str(int(score)), ha="center", va="center",
                   fontsize=10, fontweight="bold")
    
    def highlight_best_panel(self, ax: plt.Axes) -> None:
        """Add a gold border around the axes (for best candidate highlighting).
        
        Parameters
        ----------
        ax : matplotlib Axes
            The axes to highlight with a gold border.
        """
        for spine in ax.spines.values():
            spine.set_edgecolor("gold")
            spine.set_linewidth(3)
    
    def create_figure(
        self,
        scores: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Convenience method to create a figure and draw the grid.
        
        Parameters
        ----------
        scores : ndarray, optional
            Score matrix to display.
        title : str, optional
            Figure title.
        figsize : tuple
            Figure size.
        
        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        fig, ax = plt.subplots(figsize=figsize)
        self.draw_grid(ax, scores=scores)
        
        if title:
            ax.set_title(title, fontsize=12)
        else:
            ax.set_title(f"Alignment DAG for X='{self.seq1}', Y='{self.seq2}'")
        
        return fig, ax


@dataclass
class CellUpdateResult:
    """Result of a single cell update visualization.
    
    Attributes
    ----------
    i, j : int
        The cell coordinates that were updated.
    best_score : float
        The best score (max of all candidates).
    best_move : str
        The best move type: "left", "diag", or "above".
    predecessor : tuple
        The (i, j) coordinates of the predecessor cell.
    candidates : dict
        Dictionary mapping move names to (predecessor, score, edge_score).
    """
    i: int
    j: int
    best_score: float
    best_move: str
    predecessor: Tuple[int, int]
    candidates: Dict[str, Tuple[Tuple[int, int], float, float]]


def visualize_cell_update(
    plotter: NWDagPlotter,
    F: np.ndarray,
    i: int,
    j: int,
    traceback_edges: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
    figsize: Tuple[int, int] = (15, 5),
    show: bool = True,
) -> Dict[str, any]:
    """Visualize the three candidate moves for filling cell (i, j).
    
    Creates a 3-panel figure showing:
    - Left panel: coming from the left (horizontal gap)
    - Center panel: coming from diagonal (match/mismatch)
    - Right panel: coming from above (vertical gap)
    
    The best candidate panel gets a gold border.
    
    Parameters
    ----------
    plotter : NWDagPlotter
        The plotter instance with sequences and scoring parameters.
    F : ndarray
        Current DP score matrix. Must have boundary values filled.
        The function reads F[i-1, j], F[i-1, j-1], F[i, j-1] and 
        writes F[i, j] with the best score.
    i, j : int
        Target cell coordinates (must be >= 1).
    traceback_edges : list, optional
        List of previously computed traceback edges to display as bold.
        Each edge is ((i1, j1), (i2, j2)). The function will add the
        new best edge to this list in-place.
    figsize : tuple
        Figure size for the 3-panel plot.
    show : bool
        If True, call plt.show().
    
    Returns
    -------
    dict
        Contains keys: 'i', 'j', 'best_score', 'best_move', 'predecessor', 
        'candidates', 'F', 'traceback_edges'. The F matrix and traceback_edges
        list are updated in-place.
    
    Examples
    --------
    >>> plotter = NWDagPlotter("GATT", "GACT", match=5, mismatch=-5, gap=-3)
    >>> F = np.zeros((5, 5))
    >>> # Initialize boundaries
    >>> for k in range(5): F[k, 0] = k * (-3); F[0, k] = k * (-3)
    >>> edges = [((0,j), (0,j+1)) for j in range(4)] + [((k,0), (k+1,0)) for k in range(4)]
    >>> result = visualize_cell_update(plotter, F, 1, 1, traceback_edges=edges)
    >>> print(f"Best move: {result['best_move']}, score: {result['best_score']}")
    """
    if i < 1 or j < 1:
        raise ValueError(f"Cell ({i}, {j}) must have i >= 1 and j >= 1")
    
    # Initialize traceback_edges if not provided
    if traceback_edges is None:
        traceback_edges = []
    
    X, Y = plotter.seq1, plotter.seq2
    match, mismatch, gap = plotter.match, plotter.mismatch, plotter.gap
    
    # Read predecessor scores from F
    F_pred_left = F[i, j-1]      # left
    F_pred_diag = F[i-1, j-1]    # diagonal
    F_pred_above = F[i-1, j]     # above
    
    # Compute candidate scores
    is_match = X[i-1] == Y[j-1]
    diag_edge_score = match if is_match else mismatch
    
    score_from_left = F_pred_left + gap
    score_from_diag = F_pred_diag + diag_edge_score
    score_from_above = F_pred_above + gap
    
    # Determine the best
    scores_list = [score_from_left, score_from_diag, score_from_above]
    best_score = max(scores_list)
    
    # Map to move names
    move_info = [
        ("left", (i, j-1), score_from_left, gap),
        ("diag", (i-1, j-1), score_from_diag, diag_edge_score),
        ("above", (i-1, j), score_from_above, gap),
    ]
    
    # Find best move (prefer diagonal on tie for convention)
    if score_from_diag == best_score:
        best_move = "diag"
        best_pred = (i-1, j-1)
    elif score_from_left == best_score:
        best_move = "left"
        best_pred = (i, j-1)
    else:
        best_move = "above"
        best_pred = (i-1, j)
    
    # Update F in place
    F[i, j] = best_score
    
    # Add the new best edge to traceback_edges (in-place)
    new_edge = (best_pred, (i, j))
    traceback_edges.append(new_edge)
    
    # Build candidates dict for return
    candidates = {
        "left": ((i, j-1), score_from_left, gap),
        "diag": ((i-1, j-1), score_from_diag, diag_edge_score),
        "above": ((i-1, j), score_from_above, gap),
    }
    
    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    titles = [
        f"From left: F({i},{j-1}) + gap\n= {int(F_pred_left)} + ({int(gap)}) = {int(score_from_left)}",
        f"From diagonal: F({i-1},{j-1}) + {'match' if is_match else 'mismatch'}\n= {int(F_pred_diag)} + ({int(diag_edge_score)}) = {int(score_from_diag)}",
        f"From above: F({i-1},{j}) + gap\n= {int(F_pred_above)} + ({int(gap)}) = {int(score_from_above)}",
    ]
    
    panel_moves = [
        ((i, j-1), score_from_left),    # left
        ((i-1, j-1), score_from_diag),  # diagonal
        ((i-1, j), score_from_above),   # above
    ]
    
    # Previous edges (all except the one we just added)
    previous_edges = traceback_edges[:-1]
    
    for ax, title, (pred_coord, total_score) in zip(axes, titles, panel_moves):
        # Draw base grid with faded edges
        plotter.draw_grid(ax, edge_alpha=0.2)
        
        # Draw all previous traceback edges (bold, but no score labels to avoid clutter)
        if previous_edges:
            plotter.draw_traceback_edges(ax, previous_edges, show_scores=False)
        
        # Fill all computed cells (those with finite values in F, up to current)
        for ii in range(F.shape[0]):
            for jj in range(F.shape[1]):
                if (ii, jj) != (i, j) and np.isfinite(F[ii, jj]) and (ii < i or (ii == i and jj < j) or ii == 0 or jj == 0):
                    plotter.fill_node(ax, ii, jj, score=int(F[ii, jj]))
        
        # Draw the candidate edge (with score label)
        plotter.draw_traceback_edges(ax, [(pred_coord, (i, j))], show_scores=True)
        
        # Highlight the target cell with gold
        plotter.fill_node(ax, i, j, score=int(total_score), style="update")
        
        ax.set_title(title, fontsize=10)
        
        # Gold border on the best candidate
        if total_score == best_score:
            plotter.highlight_best_panel(ax)
    
    plt.tight_layout()
    if show:
        plt.show()
    
    # Return result as dict (works better with autoreload)
    return {
        "i": i, 
        "j": j,
        "best_score": best_score,
        "best_move": best_move,
        "predecessor": best_pred,
        "candidates": candidates,
        "F": F,  # Include reference to updated matrix
        "traceback_edges": traceback_edges,  # Include updated edge list
    }


def draw_nw_dag(
    seq1: str,
    seq2: str,
    match: float = 1,
    mismatch: float = -1,
    gap: float = -1,
    scores: Optional[np.ndarray] = None,
    traceback_edges: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
    update_cell: Optional[Tuple[int, int, Union[int, float]]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    style: Optional[NWDagStyle] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes, Dict[Tuple[int, int], Tuple[float, float]]]:
    """Convenience function to draw a complete NW DAG with all features.
    
    This is the main entry point for simple use cases.
    
    Parameters
    ----------
    seq1, seq2 : str
        Sequences to align.
    match, mismatch, gap : float
        Scoring parameters.
    scores : ndarray, optional
        Score matrix F[i, j] to display in nodes.
    traceback_edges : list, optional
        List of edges to highlight as traceback path.
    update_cell : tuple (i, j, score), optional
        Cell to highlight with gold "update" style.
    title : str, optional
        Figure title.
    figsize : tuple
        Figure size.
    style : NWDagStyle, optional
        Style configuration.
    show : bool
        If True, call plt.show().
    
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    coords : dict
        Coordinate mapping.
    
    Examples
    --------
    >>> # Simple blank DAG
    >>> fig, ax, coords = draw_nw_dag("GATT", "GACT", match=5, mismatch=-5, gap=-3)
    
    >>> # DAG with scores and traceback
    >>> F = np.array([[0, -3, -6], [-3, 5, 2], [-6, 2, 10]])
    >>> edges = [((0, 0), (1, 1)), ((1, 1), (2, 2))]
    >>> fig, ax, coords = draw_nw_dag("GA", "GA", scores=F, traceback_edges=edges)
    """
    plotter = NWDagPlotter(seq1, seq2, match, mismatch, gap, style)
    
    # Adjust edge alpha if we have traceback edges
    edge_alpha = 0.2 if traceback_edges else None
    
    fig, ax = plt.subplots(figsize=figsize)
    coords = plotter.draw_grid(ax, scores=scores, edge_alpha=edge_alpha)
    
    # Draw traceback edges
    if traceback_edges:
        plotter.draw_traceback_edges(ax, traceback_edges)
    
    # Highlight update cell
    if update_cell:
        i, j, score = update_cell
        plotter.fill_node(ax, i, j, score=score, style="update")
    
    # Set title
    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title(f"Alignment DAG for X='{seq1}', Y='{seq2}'")
    
    if show:
        plt.show()
    
    return fig, ax, coords


def plot_example_alignments(
    seq1: str,
    seq2: str,
    paths: List[List[Tuple[int, int]]],
    titles: List[str],
    match: float = 5,
    mismatch: float = -5,
    gap: float = -3,
    figsize: Tuple[float, float] = (15, 5),
    style: Optional[NWDagStyle] = None,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot multiple alignment paths through the NW DAG.
    
    Each path is shown in a separate panel with the alignment and score.
    
    Parameters
    ----------
    seq1, seq2 : str
        Sequences to align (seq1 = rows/X, seq2 = columns/Y).
    paths : list of list of (i, j) tuples
        Each path is a list of coordinates from (0,0) to (n,m).
    titles : list of str
        Title for each panel (should match length of paths).
    match, mismatch, gap : float
        Scoring parameters.
    figsize : tuple
        Figure size.
    style : NWDagStyle, optional
        Style configuration.
    show : bool
        If True, call plt.show().
    
    Returns
    -------
    fig : matplotlib Figure
    axes : array of matplotlib Axes
    
    Examples
    --------
    >>> path1 = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    >>> path2 = [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (4, 4)]
    >>> fig, axes = plot_example_alignments(
    ...     "GATT", "GACT",
    ...     paths=[path1, path2],
    ...     titles=["All diagonal", "Shifted"]
    ... )
    """
    plotter = NWDagPlotter(seq1, seq2, match, mismatch, gap, style)
    
    n_panels = len(paths)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]
    
    for ax, path, title in zip(axes, paths, titles):
        # Draw grid with faded edges
        plotter.draw_grid(ax, edge_alpha=0.2)
        
        # Convert path to edges and draw traceback
        edges = [(path[k], path[k+1]) for k in range(len(path) - 1)]
        plotter.draw_traceback_edges(ax, edges)
        
        # Fill path nodes with cumulative scores
        cumulative = 0
        for idx, (i, j) in enumerate(path):
            if idx > 0:
                cumulative += plotter._get_edge_score(path[idx-1], (i, j))
            plotter.fill_node(ax, i, j, score=cumulative)
        
        # Compute alignment string from path
        align1, align2 = path_to_alignment(seq1, seq2, path)
        
        ax.set_title(f"{title}\n{align1}\n{align2}\nScore = {cumulative}", font='monospace', fontsize=16)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, axes


def _fill_cell_silent(
    plotter: NWDagPlotter,
    F: np.ndarray,
    i: int,
    j: int,
    traceback_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> float:
    """Fill cell (i,j) without visualization, updating F and traceback_edges.
    
    Internal helper for run_dp_fill_demo.
    """
    X, Y = plotter.seq1, plotter.seq2
    match, mismatch, gap = plotter.match, plotter.mismatch, plotter.gap
    
    is_match = X[i-1] == Y[j-1]
    diag_edge_score = match if is_match else mismatch
    
    score_from_left = F[i, j-1] + gap
    score_from_diag = F[i-1, j-1] + diag_edge_score
    score_from_above = F[i-1, j] + gap
    
    best_score = max(score_from_left, score_from_diag, score_from_above)
    F[i, j] = best_score
    
    # Determine best predecessor (prefer diagonal > left > above for ties)
    if score_from_diag == best_score:
        best_pred = (i-1, j-1)
    elif score_from_left == best_score:
        best_pred = (i, j-1)
    else:
        best_pred = (i-1, j)
    
    traceback_edges.append((best_pred, (i, j)))
    return best_score


def run_dp_fill_demo(
    seq1: str,
    seq2: str,
    match: float = 5,
    mismatch: float = -5,
    gap: float = -3,
    show_first_n: int = 2,
    show_last_m: int = 2,
    style: Optional[NWDagStyle] = None,
) -> Dict:
    """Run and visualize the NW DP fill process.
    
    This function:
    1. Initializes the DP matrix F with boundary conditions
    2. Fills all interior cells in row-major order
    3. Visualizes the first N and last M cells being filled
    4. Returns the completed F matrix and traceback edges
    
    Parameters
    ----------
    seq1, seq2 : str
        Sequences to align (seq1 = rows/X, seq2 = columns/Y).
    match, mismatch, gap : float
        Scoring parameters.
    show_first_n : int
        Number of first interior cells to visualize.
    show_last_m : int
        Number of last interior cells to visualize.
    style : NWDagStyle, optional
        Style configuration.
    
    Returns
    -------
    dict with keys:
        'F': np.ndarray - the filled DP matrix
        'traceback_edges': list - edges from best predecessors
        'plotter': NWDagPlotter - the plotter instance (for further plotting)
        'final_score': float - optimal alignment score F[n,m]
    
    Examples
    --------
    >>> result = run_dp_fill_demo("GATT", "GACT", show_first_n=2, show_last_m=2)
    >>> print(f"Optimal score: {result['final_score']}")
    """
    plotter = NWDagPlotter(seq1, seq2, match, mismatch, gap, style)
    n, m = len(seq1), len(seq2)
    
    # Initialize F matrix with boundaries
    F = np.full((n + 1, m + 1), np.nan)
    F[0, 0] = 0
    for i in range(1, n + 1):
        F[i, 0] = i * gap
    for j in range(1, m + 1):
        F[0, j] = j * gap
    
    # Initialize traceback edges from boundary conditions
    traceback_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    traceback_edges += [((0, j), (0, j+1)) for j in range(m)]      # first row
    traceback_edges += [((i, 0), (i+1, 0)) for i in range(n)]      # first column
    
    # Build list of all interior cells in row-major order
    all_cells = [(i, j) for i in range(1, n + 1) for j in range(1, m + 1)]
    total_cells = len(all_cells)
    
    # Determine which cells to visualize
    cells_to_show = set()
    for idx in range(min(show_first_n, total_cells)):
        cells_to_show.add(all_cells[idx])
    for idx in range(max(0, total_cells - show_last_m), total_cells):
        cells_to_show.add(all_cells[idx])
    
    # Process all cells
    print(f"DP fill: visualizing first {show_first_n} and last {show_last_m} of {total_cells} interior cells\n")
    for idx, (i, j) in enumerate(all_cells):
        if (i, j) in cells_to_show:
            result = visualize_cell_update(plotter, F, i=i, j=j,
                                           traceback_edges=traceback_edges, show=True)
            print(f"Cell ({i}, {j}): best_move='{result['best_move']}', best_score={int(result['best_score'])}\n")
        else:
            _fill_cell_silent(plotter, F, i, j, traceback_edges)
            if idx == show_first_n:
                print(f"... (silently filling {total_cells - show_first_n - show_last_m} intermediate cells) ...\n")
    
    final_score = F[n, m]
    print(f"*** Final alignment score: F[{n},{m}] = {int(final_score)} ***")
    
    return {
        'F': F,
        'traceback_edges': traceback_edges,
        'plotter': plotter,
        'final_score': final_score,
    }


def plot_filled_dag(
    seq1: str,
    seq2: str,
    F: np.ndarray,
    traceback_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    match: float = 5,
    mismatch: float = -5,
    gap: float = -3,
    figsize: Tuple[float, float] = (7, 6),
    style: Optional[NWDagStyle] = None,
    show: bool = True,
    plotter: Optional[NWDagPlotter] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a fully filled NW DAG with all traceback edges.
    
    Parameters
    ----------
    seq1, seq2 : str
        Sequences to align.
    F : np.ndarray
        The filled DP matrix.
    traceback_edges : list
        List of (source, target) tuples for traceback edges.
    match, mismatch, gap : float
        Scoring parameters.
    figsize : tuple
        Figure size.
    style : NWDagStyle, optional
        Style configuration.
    show : bool
        If True, call plt.show().
    plotter : NWDagPlotter, optional
        Existing plotter to reuse (avoids recreating).
    
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    
    Examples
    --------
    >>> result = run_dp_fill_demo("GATT", "GACT")
    >>> fig, ax = plot_filled_dag("GATT", "GACT", result['F'], result['traceback_edges'])
    """
    if plotter is None:
        plotter = NWDagPlotter(seq1, seq2, match, mismatch, gap, style)
    n, m = len(seq1), len(seq2)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw grid with faded background edges
    plotter.draw_grid(ax, edge_alpha=0.2)
    
    # Draw all traceback edges (bold)
    plotter.draw_traceback_edges(ax, traceback_edges, show_scores=False)
    
    # Fill all nodes with their scores
    for i in range(n + 1):
        for j in range(m + 1):
            plotter.fill_node(ax, i, j, score=int(F[i, j]))
    
    # Highlight the final cell
    plotter.fill_node(ax, n, m, score=int(F[n, m]), style="update")
    
    ax.set_title(f"Fully filled NW DAG with traceback edges\nOptimal score: F[{n},{m}] = {int(F[n, m])}", 
                 fontsize=12)
    plt.tight_layout()
    
    if show:
        plt.show()
        print(f"\nFinal F matrix:")
        print(F.astype(int))
    
    return fig, ax


def _extract_optimal_path(
    traceback_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    n: int,
    m: int,
) -> List[Tuple[int, int]]:
    """Extract one optimal path from (0,0) to (n,m) using traceback edges.
    
    Internal helper for plot_optimal_path.
    """
    # Build a dict from target -> source for quick lookup
    edge_dict = {}
    for (src, tgt) in traceback_edges:
        if tgt not in edge_dict:
            edge_dict[tgt] = src
    
    # Trace back from (n, m) to (0, 0)
    path = [(n, m)]
    current = (n, m)
    while current != (0, 0):
        if current in edge_dict:
            current = edge_dict[current]
            path.append(current)
        else:
            break
    path.reverse()
    return path


def plot_optimal_path(
    seq1: str,
    seq2: str,
    F: np.ndarray,
    traceback_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    match: float = 5,
    mismatch: float = -5,
    gap: float = -3,
    figsize: Tuple[float, float] = (7, 6),
    style: Optional[NWDagStyle] = None,
    show: bool = True,
    plotter: Optional[NWDagPlotter] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Plot the optimal path through the NW DAG.
    
    Highlights the traceback path in red and shows the resulting alignment.
    
    Parameters
    ----------
    seq1, seq2 : str
        Sequences to align.
    F : np.ndarray
        The filled DP matrix.
    traceback_edges : list
        List of (source, target) tuples for traceback edges.
    match, mismatch, gap : float
        Scoring parameters.
    figsize : tuple
        Figure size.
    style : NWDagStyle, optional
        Style configuration.
    show : bool
        If True, call plt.show() and print alignment.
    plotter : NWDagPlotter, optional
        Existing plotter to reuse.
    
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    info : dict with keys:
        'optimal_path': list of (i,j) coordinates
        'alignment': tuple (aligned_seq1, aligned_seq2)
        'score': float
    
    Examples
    --------
    >>> result = run_dp_fill_demo("GATT", "GACT")
    >>> fig, ax, info = plot_optimal_path("GATT", "GACT", result['F'], result['traceback_edges'])
    """
    if plotter is None:
        plotter = NWDagPlotter(seq1, seq2, match, mismatch, gap, style)
    n, m = len(seq1), len(seq2)
    
    # Extract the optimal path
    optimal_path = _extract_optimal_path(traceback_edges, n, m)
    optimal_edges = [(optimal_path[k], optimal_path[k+1]) for k in range(len(optimal_path) - 1)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw grid with faded edges
    plotter.draw_grid(ax, edge_alpha=0.15)
    
    # Fill all nodes with their scores
    for i in range(n + 1):
        for j in range(m + 1):
            plotter.fill_node(ax, i, j, score=int(F[i, j]))
    
    # Draw optimal path edges as bold red arrows
    for (src, tgt) in optimal_edges:
        # Use the plotter's coordinate system
        x0, y0 = plotter._coords[src]
        x1, y1 = plotter._coords[tgt]
        
        # Compute direction and shrink from node centers
        v = np.array([x1 - x0, y1 - y0], dtype=float)
        L = np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0
        shrink = plotter.style.node_radius * plotter.style.edge_shrink_factor
        
        start = (x0 + v[0] * (shrink / L), y0 + v[1] * (shrink / L))
        end = (x1 - v[0] * (shrink / L), y1 - v[1] * (shrink / L))
        
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle='-|>',
            mutation_scale=12,
            linewidth=3,
            color='red',
            alpha=0.9,
            zorder=10
        )
        ax.add_patch(arrow)
    
    # Highlight nodes on the optimal path with red circles
    for (i, j) in optimal_path:
        x, y = plotter._coords[(i, j)]
        circle = Circle((x, y), 0.28, fill=False, linewidth=2.5, color='red', alpha=0.8, zorder=11)
        ax.add_patch(circle)
    
    ax.set_title(f"Optimal path (red) through the NW DAG\nAlignment score: {int(F[n, m])}", fontsize=12)
    plt.tight_layout()
    
    # Compute alignment from path
    aligned1, aligned2 = path_to_alignment(seq1, seq2, optimal_path)
    
    if show:
        plt.show()
        print("Optimal alignment recovered from traceback:")
        print(f"  X: {aligned1}")
        print(f"  Y: {aligned2}")
        match_line = ''.join('|' if a == b else ' ' if '-' in (a, b) else 'x' 
                             for a, b in zip(aligned1, aligned2))
        print(f"     {match_line}")
        print(f"     (|=match, x=mismatch, space=gap)")
    
    return fig, ax, {
        'optimal_path': optimal_path,
        'alignment': (aligned1, aligned2),
        'score': F[n, m],
    }


def plot_dp_initialization(
    seq1: str,
    seq2: str,
    match: float = 5,
    mismatch: float = -5,
    gap: float = -3,
    figsize: Tuple[float, float] = (15, 5),
    style: Optional[NWDagStyle] = None,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot the three initialization steps for NW dynamic programming.
    
    Shows three panels:
    1. Initialize F(0,0) = 0
    2. Fill first row: F(0,j) = j × gap
    3. Fill first column: F(i,0) = i × gap
    
    Parameters
    ----------
    seq1, seq2 : str
        Sequences to align (seq1 = rows/X, seq2 = columns/Y).
    match, mismatch, gap : float
        Scoring parameters.
    figsize : tuple
        Figure size.
    style : NWDagStyle, optional
        Style configuration.
    show : bool
        If True, call plt.show().
    
    Returns
    -------
    fig : matplotlib Figure
    axes : array of matplotlib Axes
    
    Examples
    --------
    >>> fig, axes = plot_dp_initialization("GATT", "GACT", gap=-3)
    """
    plotter = NWDagPlotter(seq1, seq2, match, mismatch, gap, style)
    n, m = len(seq1), len(seq2)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    titles = [
        "Step 1: Initialize F(0,0) = 0",
        "Step 2: Fill first row\nF(0,j) = j × gap",
        "Step 3: Fill first column\nF(i,0) = i × gap",
    ]
    
    for panel_idx, (ax, title) in enumerate(zip(axes, titles)):
        # Draw base grid with faded edges
        plotter.draw_grid(ax, edge_alpha=0.2)
        
        # Panel 1+: F[0,0] = 0
        plotter.fill_node(ax, 0, 0, score=0)
        
        # Panel 2+: First row with traceback edges
        if panel_idx >= 1:
            row_edges = [((0, j), (0, j+1)) for j in range(m)]
            plotter.draw_traceback_edges(ax, row_edges)
            for j in range(1, m + 1):
                plotter.fill_node(ax, 0, j, score=j * gap)
        
        # Panel 3: First column with traceback edges
        if panel_idx >= 2:
            col_edges = [((i, 0), (i+1, 0)) for i in range(n)]
            plotter.draw_traceback_edges(ax, col_edges)
            for i in range(1, n + 1):
                plotter.fill_node(ax, i, 0, score=i * gap)
        
        ax.set_title(title, fontsize=11)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, axes
