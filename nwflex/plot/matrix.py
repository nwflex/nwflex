"""
DP Matrix visualization for NW-flex.

This module contains functions for visualizing DP score matrices
as heatmaps with traceback paths overlaid.

Functions:
    - plot_flex_matrices: Original flex matrix plotter
    - plot_gotoh_matrices: Standard Gotoh matrix plotter  
    - plot_flex_matrices_publication: Publication-quality flex matrix plotter
    - draw_figure_region_backgrounds: Draw region bands at figure level
    - draw_figure_leader_closer_boxes: Draw leader/closer boxes at figure level
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
from typing import Dict, Optional, List, Tuple

from .colors import NT_COLOR, EP_COLORS, HEATMAP_COLORMAPS
from .ep import draw_matrix_regions

grid_color_map = HEATMAP_COLORMAPS['diverging']

def plot_flex_matrices(
    result,  # AlignmentResult
    X: str,
    Y: str,
    s: int,
    e: int,
    nt_color_map: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    marker_size: int = 18,
    marker_color: str = "#00ff2f",
    marker_width: int = 2,
    show_bg_path: bool = True,
    marker_bg_color: str = "black",
    show_jumps: bool = True,
    jump_color: str = "#ffcc00",
    jump_width: int = 3,
    jump_alpha: float = 0.9,
    marker_alpha: float = 0.9,
    marker_type: str = "s",
    regions: Optional[List[dict]] = None,
    region_label_side: str = 'right',
    region_linewidth: float = 2.5,
    region_label_fontsize: float = 14.0,
    colormap: str = grid_color_map,
    annotate: bool = True,
    tick_fontsize: float = 10.0,
) -> plt.Figure:
    """
    Plot the three NW-flex score layers (Yg, M, Xg) as heatmaps with
    optional overlay of the best DP path and block boundaries.

    Jumps (RowJump instances) are drawn as segments from the predecessor
    to the current cell. On the panel for the corresponding state they
    use jump_color; on other panels they are shown in gray.
    
    Parameters
    ----------
    result : AlignmentResult
        Result from align_single_block or similar, with return_data=True.
    X : str
        Reference sequence.
    Y : str
        Query sequence.
    s : int
        Leader row index (last row of flank A).
    e : int
        End of flexible block Z (e = s + len(Z)).
    regions : list of dict, optional
        Region specifications for drawing outline boxes. Each dict has:
        - 'start': int - starting row (1-indexed)
        - 'end': int - ending row (1-indexed)
        - 'label': str - label text (e.g., 'A', 'Z*', 'B')
        - 'color': str, optional - outline/label color
        Use build_AZB_regions() to generate standard A·Z·B regions.
    region_label_side : str
        Side for region labels: 'left' or 'right' (default: 'right').
    region_linewidth : float
        Width of region outline boxes (default: 2.5).
    region_label_fontsize : float
        Font size for region labels (default: 14.0).
    annotate : bool True
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object with three panels (Yg, M, Xg).
    
    Examples
    --------
    >>> res = align_single_block(X, Y, s, e, ..., return_data=True)
    >>> # Basic plot without region boxes
    >>> fig = plot_flex_matrices(res, X, Y, s, e)
    >>> 
    >>> # With A·Z·B region annotations
    >>> from nwflex.plot.ep import build_AZB_regions
    >>> regions = build_AZB_regions(s, e, len(X))
    >>> fig = plot_flex_matrices(res, X, Y, s, e, regions=regions)
    """
    if result.data is None:
        raise ValueError(
            "plot_flex_matrices requires result.data (FlexData). "
            "Run the aligner with return_data=True."
        )

    if nt_color_map is None:
        nt_color_map = NT_COLOR

    data = result.data
    Yg = data.Yg
    M  = data.M
    Xg = data.Xg

    # Panels: Yg, M, Xg
    matrices = [Yg, M, Xg]
    titles   = ["Yg (gap in X)", "M (match/mismatch)", "Xg (gap in Y)"]
    # State ids from dp_core: 0 = Yg, 1 = M, 2 = Xg
    state_for_panel = [0, 1, 2]

    # Common vmin/vmax ignoring -inf
    finite_vals = np.concatenate([mat[np.isfinite(mat)] for mat in matrices])
    vmin, vmax = finite_vals.min(), finite_vals.max()

    # Colormap with NaN (bad) as black
    cmap = sns.color_palette(colormap, as_cmap=True)
    cmap.set_bad(color="grey")

    xticklabels = [""] + list(Y)
    yticklabels = [""] + list(X)

    fig, axes = plt.subplots(
        1, 3, figsize=figsize, sharex=True, sharey=True, constrained_layout=False
    )

    # Plot score heatmaps
    for ax, mat, title in zip(axes, matrices, titles):
        mat_plot = mat.copy()
        mat_plot[~np.isfinite(mat_plot)] = np.nan

        sns.heatmap(
            mat_plot,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar=False,
            annot=annotate,
            fmt=".0f",
            xticklabels=xticklabels,
            yticklabels=yticklabels,
        )
        ax.set_title(title)
        ax.set_xlabel("Y (columns)")
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        ax.xaxis.set_label_position("top")

        # Color X-axis labels
        for tick, lab in zip(ax.get_xticklabels(), xticklabels):
            tick.set_rotation(0)
            tick.set_va("center")
            tick.set_color(nt_color_map.get(lab, "black"))
            tick.set_fontweight("bold")
            tick.set_fontsize(tick_fontsize)
        LEADER_COLOR = "#1f77b4"  # blue - for A (leader row s)
        CLOSER_COLOR = "#7030a0"  # purple - for B (closer row e+1)
        # Block boundaries
        ax.axhline(s,     color=LEADER_COLOR,  linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axhline(s + 1, color=LEADER_COLOR,  linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axhline(e + 1, color=CLOSER_COLOR, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axhline(e + 2, color=CLOSER_COLOR, linestyle="--", linewidth=1.5, alpha=0.7)

    # Y-axis labels on first subplot
    axes[0].set_ylabel("X (rows)")
    for tick, lab in zip(axes[0].get_yticklabels(), yticklabels):
        tick.set_rotation(0)
        tick.set_va("center")
        tick.set_color(nt_color_map.get(lab, "black"))
        tick.set_fontweight("bold")
        tick.set_fontsize(tick_fontsize)

    # Map state id -> axis
    state_to_ax = {state_id: ax for state_id, ax in zip(state_for_panel, axes)}

    # Path overlay
    path: Optional[List[Tuple[int, int, int]]] = getattr(result, "path", None)
    if path:
        for (i, j, s_state) in path:
            x = j + 0.5
            y = i + 0.5

            if show_bg_path:
                for ax in axes:
                    ax.plot(
                        x,
                        y,
                        marker=marker_type,
                        markersize=marker_size,
                        markeredgecolor=marker_bg_color,
                        markerfacecolor="none",
                        alpha=marker_alpha * marker_alpha,
                        markeredgewidth=marker_width,
                    )

            ax_state = state_to_ax.get(s_state)
            if ax_state is not None:
                ax_state.plot(
                    x,
                    y,
                    marker=marker_type,
                    markersize=marker_size,
                    markeredgecolor=marker_color,
                    markerfacecolor="none",
                    alpha=marker_alpha,
                    markeredgewidth=marker_width,
                )

    # Jump overlay on all panels
    if show_jumps and getattr(result, "jumps", None):
        for jmp in result.jumps:
            state_id = int(jmp.state)
            col = int(jmp.col)
            r0 = int(jmp.from_row)
            r1 = int(jmp.to_row)
            
            # Terminal jumps have from_row > to_row (from virtual n+1 to actual row r)
            is_terminal = r0 > r1

            if is_terminal:
                # Terminal jump: from virtual row n+1 to actual traceback start row
                x0 = col + 0.5
                x1 = col + 1.0
                y0 = r1 + 1.0  # actual traceback start
                y1 = r0  # virtual row (may be off grid)
            elif state_id == 1:
                # M: diagonal jump from (r0, col-1) to (r1, col)
                x0 = col - 0.5
                y0 = r0 + 1.
                x1 = col + 0.5
                y1 = r1
            elif state_id == 2:
                # Xg: vertical jump from (r0, col) to (r1, col)
                x0 = x1 = col + 0.5
                y0 = r0 + 1
                y1 = r1
            elif state_id == 0:
                # Yg: horizontal (but Yg jumps don't occur in standard NW-flex)
                continue
            else:
                continue

            for st, ax in zip(state_for_panel, axes):
                if is_terminal or st == state_id:
                    # Terminal jumps and matching state use standard jump styling
                    color = jump_color
                    alpha = jump_alpha
                    lw = jump_width
                else:
                    color = "black"
                    alpha = jump_alpha * jump_alpha
                    lw = max(1, jump_width - 1)

                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=color,
                    linewidth=lw,
                    alpha=alpha,
                )

    # Draw region annotations if provided
    if regions is not None:
        draw_matrix_regions(
            axes,
            regions,
            label_side=region_label_side,
            linewidth=region_linewidth,
            label_fontsize=region_label_fontsize,
        )

    fig.tight_layout()
    return fig


def plot_gotoh_matrices(
    result,  # AlignmentResult from nw_gotoh_matrix
    X: str,
    Y: str,
    nt_color_map: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    marker_size: int = 18,
    marker_color: str = "#00ff2f",
    marker_width: int = 2,
    show_bg_path: bool = True,
    marker_bg_color: str = "black",
    marker_style: str = 's',
    colormap: str = grid_color_map,
) -> plt.Figure:
    """
    Plot the three Gotoh score layers (Yg, M, Xg) as heatmaps with
    the traceback path overlaid.
    
    This uses the same visual style as plot_flex_matrices but works with
    the AlignmentResult from nw_gotoh_matrix (which has M, X, Y matrices
    instead of Yg, M, Xg in a FlexData object).
    
    Parameters
    ----------
    result : AlignmentResult
        Result from nw_gotoh_matrix with return_matrices=True
    X : str
        Reference sequence (seq1)
    Y : str
        Query sequence
    nt_color_map : dict, optional
        Mapping nucleotides to colors for axis labels
    figsize : tuple
        Figure size
    marker_size : int
        Size of path markers
    marker_color : str
        Color of main path markers
    marker_width : int
        Width of path marker edges
    show_bg_path : bool
        Show faded path markers on all panels
    marker_bg_color : str
        Color for background path markers
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    if nt_color_map is None:
        nt_color_map = NT_COLOR

    # nw_gotoh_matrix returns: M (match), X (gap in Y/seq2), Y (gap in X/seq1)
    # We want to display as: Yg (gap in X), M, Xg (gap in Y) to match flex convention
    # So: Yg = result.Y, M = result.M, Xg = result.X
    Yg = result.Y  # gap in seq1 (horizontal move) = gap in X
    M  = result.M  # match/mismatch
    Xg = result.X  # gap in seq2 (vertical move) = gap in Y

    # Panels: Yg, M, Xg
    matrices = [Yg, M, Xg]
    titles   = ["Yg (gap in X)", "M (match/mismatch)", "Xg (gap in Y)"]
    
    # State mapping: Gotoh uses 0=M, 1=X, 2=Y
    # We display panels as [Yg, M, Xg] = indices [0, 1, 2]
    # Gotoh state 0 (M) -> panel 1
    # Gotoh state 1 (X=Xg) -> panel 2  
    # Gotoh state 2 (Y=Yg) -> panel 0
    gotoh_state_to_panel = {0: 1, 1: 2, 2: 0}

    # Common vmin/vmax ignoring -inf
    finite_vals = np.concatenate([mat[np.isfinite(mat)] for mat in matrices])
    vmin, vmax = finite_vals.min(), finite_vals.max()

    # Colormap with NaN (bad) as black
    cmap = sns.color_palette(colormap, as_cmap=True)
    cmap.set_bad(color="grey")

    xticklabels = [""] + list(Y)
    yticklabels = [""] + list(X)

    fig, axes = plt.subplots(
        1, 3, figsize=figsize, sharex=True, sharey=True, constrained_layout=False
    )

    # Plot score heatmaps
    for ax, mat, title in zip(axes, matrices, titles):
        mat_plot = mat.copy()
        mat_plot[~np.isfinite(mat_plot)] = np.nan

        sns.heatmap(
            mat_plot,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar=False,
            annot=True,
            fmt=".0f",
            xticklabels=xticklabels,
            yticklabels=yticklabels,
        )
        ax.set_title(title)
        ax.set_xlabel("Y (columns)")
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        ax.xaxis.set_label_position("top")

        # Color X-axis labels
        for tick, lab in zip(ax.get_xticklabels(), xticklabels):
            tick.set_rotation(0)
            tick.set_va("center")
            tick.set_color(nt_color_map.get(lab, "black"))
            tick.set_fontweight("bold")

    # Y-axis labels on first subplot
    axes[0].set_ylabel("X (rows)")
    for tick, lab in zip(axes[0].get_yticklabels(), yticklabels):
        tick.set_rotation(0)
        tick.set_va("center")
        tick.set_color(nt_color_map.get(lab, "black"))
        tick.set_fontweight("bold")

    # Path overlay
    path = getattr(result, "best_path", None)
    if path:
        for (i, j, gotoh_state) in path:
            x = j + 0.5
            y = i + 0.5
            panel_idx = gotoh_state_to_panel.get(gotoh_state, 1)

            if show_bg_path:
                for ax in axes:
                    ax.plot(
                        x,
                        y,
                        marker=marker_style,
                        markersize=marker_size,
                        markeredgecolor=marker_bg_color,
                        markerfacecolor="none",
                        alpha=0.6,
                        markeredgewidth=marker_width,
                    )

            axes[panel_idx].plot(
                x,
                y,
                marker=marker_style,
                markersize=marker_size,
                markeredgecolor=marker_color,
                markerfacecolor="none",
                alpha=0.9,
                markeredgewidth=marker_width,
            )

    fig.tight_layout()
    return fig


def visualize_alignment_matrices(
    res,  # AlignmentResult
    X: str,
    Y: str,
    block_starts: Optional[List[int]] = None,
    block_ends: Optional[List[int]] = None,
    nt_color_map: Optional[Dict[str, str]] = None,
    figsize: tuple = (25, 10),
    marker_size: int = 20,
    marker_color: str = "#00ff2f",
    marker_width: int = 3,
    show_bg_path: bool = True,
    marker_bg_color: str = "grey",
    colormap: str = grid_color_map,
):
    """
    Visualize alignment matrices (M, X, Y, [Z], [J]) as heatmaps with the optimal path.

    Parameters
    ----------
    res : AlignmentResult
        Result from nw_freejumpup_matrix or nw_freeup_matrix
    X, Y : str
        Reference and query sequences
    block_starts : list of int, optional
        Start positions of flexible blocks (leader rows). If empty or None, no
        block boundaries are drawn.
    block_ends : list of int, optional
        End positions of flexible blocks (closer rows). Must have same length
        as block_starts.
    nt_color_map : dict, optional
        Mapping nucleotides to colors for axis labels
    figsize : tuple, optional
        Figure size (width, height)
    marker_size : int, optional
        Size of path markers
    marker_color : str, optional
        Color of main path markers (e.g., green)
    marker_width : int, optional
        Width of path marker edges
    show_bg_path : bool, optional
        Whether to show white circles (background path markers)
    marker_bg_color : str, optional
        Color of background path markers (e.g., white)
    """

    # Default nucleotide colors
    if nt_color_map is None:
        nt_color_map = NT_COLOR

    # --- Collect available matrices ---
    matrices = [res.M, res.X, res.Y]
    titles = ["M (match/mismatch)", "X (gap in Y)", "Y (gap in X)"]
    state_indices = [0, 1, 2]

    if hasattr(res, "Z") and res.Z is not None:
        matrices.append(res.Z)
        titles.append("Z (free-up)")
        state_indices.append(3)

    if hasattr(res, "J") and res.J is not None:
        matrices.append(res.J)
        titles.append("J (free-jump-up)")
        state_indices.append(4)

    n_plots = len(matrices)

    # --- Calculate global vmin/vmax ignoring -inf ---
    all_mats = np.stack(matrices)
    finite_vals = all_mats[np.isfinite(all_mats)]
    vmin, vmax = np.min(finite_vals), np.max(finite_vals)

    # --- Axis labels ---
    xticklabels = [""] + list(Y)
    yticklabels = [""] + list(X)

    # --- Subplots ---
    fig, axes = plt.subplots(
        1, n_plots, figsize=figsize, constrained_layout=False, sharex=True, sharey=True
    )

    # Ensure axes iterable
    if n_plots == 1:
        axes = [axes]

    # --- Plot heatmaps ---
    for ax, mat, title in zip(axes, matrices, titles):
        mat_plot = mat.copy()
        mat_plot[~np.isfinite(mat_plot)] = np.nan  # For seaborn to handle -inf
        sns.heatmap(
            mat_plot,
            ax=ax,
            cmap=colormap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar=False,
            annot=True,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            fmt=".0f",
        )
        ax.set_title(title)
        ax.set_xlabel("Y")
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        ax.xaxis.set_label_position("top")

        # Color x-axis labels
        for tick, lab in zip(ax.get_xticklabels(), xticklabels):
            tick.set_rotation(0)
            tick.set_va("center")
            tick.set_color(nt_color_map.get(lab, "black"))
            tick.set_fontweight("bold")
            tick.set_fontsize(12)

        # Add block boundaries (if any)
        if block_starts and block_ends:
            for bs in block_starts:
                ax.axhline(bs, color="blue", linestyle="--", linewidth=1.5, alpha=0.7)
            for be in block_ends:
                ax.axhline(be + 1, color="green", linestyle="--", linewidth=1.5, alpha=0.7)

    # Color y-ticks for first subplot
    axes[0].set_ylabel("X")
    for tick, lab in zip(axes[0].get_yticklabels(), yticklabels):
        tick.set_rotation(0)
        tick.set_va("center")
        tick.set_color(nt_color_map.get(lab, "black"))
        tick.set_fontweight("bold")
        tick.set_fontsize(12)

    # --- Path plotting ---
    state_to_ax = {state_idx: axes[i] for i, state_idx in enumerate(state_indices)}

    if getattr(res, "best_path", None):
        for (i, j, s) in res.best_path:
            if s in state_to_ax:
                ax = state_to_ax[s]
                x, y = j + 0.5, i + 0.5
                # Draw white background path on *all* axes
                if show_bg_path:
                    for ax in axes:
                        ax.plot(
                            x,
                            y,
                            marker="o",
                            markersize=marker_size,
                            markeredgecolor=marker_bg_color,
                            markerfacecolor="none",
                            alpha=0.7,
                            markeredgewidth=marker_width,
                        )

                # Draw main (green) path only on active state
                if s in state_to_ax:
                    ax = state_to_ax[s]
                    ax.plot(
                        x,
                        y,
                        marker="o",
                        markersize=marker_size,
                        markeredgecolor=marker_color,
                        markerfacecolor="none",
                        alpha=0.9,
                        markeredgewidth=marker_width,
                    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# FIGURE-LEVEL REGION AND LEADER/CLOSER DRAWING
# =============================================================================

def draw_figure_region_backgrounds(
    fig: plt.Figure,
    axes: List[plt.Axes],
    regions: List[dict],
    margin_left: float = 0.03,
    margin_right: float = 0.07,
    label_offset: float = 0.01,
    label_fontsize: float = 14.0,
    fill_alpha: float = 0.35,
    zorder: int = -20,
) -> None:
    """
    Draw region background bands at figure level, spanning all provided axes.
    
    This creates continuous horizontal bands that span across multiple matrix
    subplots and the gaps between them, similar to the DAG region backgrounds.
    
    Parameters
    ----------
    fig : matplotlib Figure
        Figure to draw on.
    axes : list of Axes
        List of axes that the regions should span.
    regions : list of dict
        Region specifications. Each dict has:
        - 'start': int - starting row (1-indexed)
        - 'end': int - ending row (1-indexed)  
        - 'label': str - label text (e.g., 'A', 'Z', 'B')
        - 'fill_color': str - background fill color
        - 'label_color': str - label text color
    margin_left : float
        Left margin in figure coordinates.
    margin_right : float
        Right margin in figure coordinates.
    label_offset : float
        Distance from left edge to label in figure coordinates.
    label_fontsize : float
        Font size for region labels.
    fill_alpha : float
        Alpha transparency for fills.
    zorder : int
        Z-order for drawing (negative = behind axes).
    """
    if not regions or not axes:
        return
    
    # Force draw to update positions
    fig.canvas.draw_idle()
    
    # Get bounding boxes in figure coordinates
    ax_left = axes[0]
    ax_right = axes[-1]
    bbox_left = ax_left.get_position()
    bbox_right = ax_right.get_position()
    
    # X range for backgrounds
    fig_x_left = bbox_left.x0 - margin_left
    fig_x_right = bbox_right.x1 + margin_right
    fig_width = fig_x_right - fig_x_left
    
    # Y mapping from data to figure coordinates
    ylim = ax_left.get_ylim()  # (n_rows, 0) - inverted
    y_data_top = min(ylim)
    y_data_bottom = max(ylim)
    data_height = y_data_bottom - y_data_top
    
    fig_y_bottom = bbox_left.y0
    fig_y_top = bbox_left.y1
    fig_height = fig_y_top - fig_y_bottom
    
    for region in regions:
        start = region['start']
        end = region['end']
        fill_color = region.get('fill_color', '#cccccc')
        label_color = region.get('label_color', 'black')
        label = region.get('label', '')
        
        # Convert data row coordinates to figure coordinates
        row_top_data = start
        row_bottom_data = end + 1
        
        frac_top = (row_top_data - y_data_top) / data_height
        frac_bottom = (row_bottom_data - y_data_top) / data_height
        
        fig_y_region_top = fig_y_top - frac_top * fig_height
        fig_y_region_bottom = fig_y_top - frac_bottom * fig_height
        
        # Draw rectangle at figure level
        rect = Rectangle(
            (fig_x_left, fig_y_region_bottom),
            fig_width,
            fig_y_region_top - fig_y_region_bottom,
            transform=fig.transFigure,
            facecolor=fill_color,
            edgecolor='none',
            alpha=fill_alpha,
            zorder=zorder,
        )
        fig.patches.append(rect)
        
        # Draw region label
        if label:
            label_x = fig_x_left - label_offset
            label_y = (fig_y_region_top + fig_y_region_bottom) / 2
            
            fig.text(
                label_x, label_y, label,
                color=label_color,
                fontsize=label_fontsize,
                fontweight='bold',
                ha='right',
                va='center',
                transform=fig.transFigure,
            )


def draw_figure_leader_closer_boxes(
    fig: plt.Figure,
    axes: List[plt.Axes],
    leader_row: int,
    closer_row: int,
    margin_left: float = 0.002,
    margin_right: float = 0.002,
    leader_color: Optional[str] = None,
    closer_color: Optional[str] = None,
    linewidth: float = 2.0,
    linestyle: str = '--',
    alpha: float = 1.0,
    label_fontsize: float = 8,
    show_labels: bool = True,
    zorder: int = 50,
) -> None:
    """
    Draw leader and closer FancyBboxPatch spanning all provided axes.
    
    Parameters
    ----------
    fig : matplotlib Figure
        Figure to draw on.
    axes : list of Axes
        List of axes that the boxes should span.
    leader_row : int
        Row index of the leader row.
    closer_row : int
        Row index of the closer row.
    margin_left : float
        Left margin in figure coordinates.
    margin_right : float
        Right margin in figure coordinates.
    leader_color : str, optional
        Color for leader box (defaults to EP_COLORS['leader']).
    closer_color : str, optional
        Color for closer box (defaults to EP_COLORS['closer']).
    linewidth : float
        Line width for boxes.
    linestyle : str
        Line style for boxes.
    alpha : float
        Alpha transparency.
    label_fontsize : float
        Font size for labels.
    show_labels : bool
        Whether to show 'leader' and 'closer' labels.
    zorder : int
        Z-order for drawing.
    """
    if leader_color is None:
        leader_color = EP_COLORS['leader']
    if closer_color is None:
        closer_color = EP_COLORS['closer']
    
    # Force draw to update positions
    fig.canvas.draw_idle()
    
    # Get bounding boxes
    ax_left = axes[0]
    ax_right = axes[-1]
    bbox_left = ax_left.get_position()
    bbox_right = ax_right.get_position()
    
    # X range for boxes
    lc_fig_x_left = bbox_left.x0 - margin_left
    lc_fig_x_right = bbox_right.x1 + margin_right
    lc_fig_width = lc_fig_x_right - lc_fig_x_left
    
    # Y mapping
    ylim = ax_left.get_ylim()
    y_data_top = min(ylim)
    y_data_bottom = max(ylim)
    data_height = y_data_bottom - y_data_top
    
    fig_y_bottom = bbox_left.y0
    fig_y_top = bbox_left.y1
    fig_height = fig_y_top - fig_y_bottom
    
    for row_idx, box_color, box_label in [(leader_row, leader_color, 'leader'),
                                           (closer_row, closer_color, 'closer')]:
        # Map row to figure coordinates
        row_top_data = row_idx
        row_bottom_data = row_idx + 1
        
        frac_top = (row_top_data - y_data_top) / data_height
        frac_bottom = (row_bottom_data - y_data_top) / data_height
        
        fig_y_row_top = fig_y_top - frac_top * fig_height
        fig_y_row_bottom = fig_y_top - frac_bottom * fig_height
        
        # Draw FancyBboxPatch
        fancy_box = FancyBboxPatch(
            (lc_fig_x_left, fig_y_row_bottom),
            lc_fig_width,
            fig_y_row_top - fig_y_row_bottom,
            boxstyle="round,pad=0,rounding_size=0.008",
            transform=fig.transFigure,
            facecolor='none',
            edgecolor=box_color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
            clip_on=False,
        )
        fig.patches.append(fancy_box)
        
        # Add label
        if show_labels:
            label_x = lc_fig_x_right + 0.01
            label_y = (fig_y_row_top + fig_y_row_bottom) / 2
            
            fig.text(
                label_x, label_y, box_label,
                color=box_color,
                fontsize=label_fontsize,
                fontweight='bold',
                fontstyle='italic',
                ha='left',
                va='center',
                transform=fig.transFigure,
            )


def draw_figure_row_highlights(
    fig: plt.Figure,
    axes: List[plt.Axes],
    highlights: List[dict],
    margin_left: float = 0.002,
    margin_right: float = 0.002,
    default_linewidth: float = 2.0,
    default_linestyle: str = ':',
    default_alpha: float = 1.0,
    label_fontsize: float = 8,
    label_side: str = 'right',
    zorder: int = 45,
) -> None:
    """
    Draw arbitrary row highlight boxes spanning all provided axes.
    
    This is a generalized version of draw_figure_leader_closer_boxes that
    supports arbitrary row spans with custom styling. Useful for highlighting
    Z* regions, selected substrings, or other row ranges.
    
    Parameters
    ----------
    fig : matplotlib Figure
        Figure to draw on.
    axes : list of Axes
        List of axes that the boxes should span.
    highlights : list of dict
        List of highlight specifications. Each dict has:
        - 'start': int - starting row (required)
        - 'end': int - ending row (required, can equal start for single row)
        - 'label': str - label text (optional)
        - 'color': str - box and label color (required)
        - 'linestyle': str - line style (optional, default ':')
        - 'linewidth': float - line width (optional)
        - 'fontsize': float - font size for this label (optional, overrides label_fontsize)
        - 'label_position': str - 'top', 'middle', or 'bottom' (optional, default 'middle')
    margin_left : float
        Left margin in figure coordinates.
    margin_right : float
        Right margin in figure coordinates.
    default_linewidth : float
        Default line width for boxes.
    default_linestyle : str
        Default line style for boxes.
    default_alpha : float
        Default alpha transparency.
    label_fontsize : float
        Font size for labels.
    label_side : str
        Side for labels: 'left' or 'right'.
    zorder : int
        Z-order for drawing.
    
    Examples
    --------
    >>> highlights = [
    ...     {'start': 2, 'end': 4, 'label': 'Z*', 'color': '#DAA520',
    ...      'linestyle': ':', 'label_position': 'top'},
    ... ]
    >>> draw_figure_row_highlights(fig, axes, highlights)
    """
    if not highlights or not axes:
        return
    
    # Force draw to update positions
    fig.canvas.draw_idle()
    
    # Get bounding boxes
    ax_left = axes[0]
    ax_right = axes[-1]
    bbox_left = ax_left.get_position()
    bbox_right = ax_right.get_position()
    
    # X range for boxes
    fig_x_left = bbox_left.x0 - margin_left
    fig_x_right = bbox_right.x1 + margin_right
    fig_width = fig_x_right - fig_x_left
    
    # Y mapping
    ylim = ax_left.get_ylim()
    y_data_top = min(ylim)
    y_data_bottom = max(ylim)
    data_height = y_data_bottom - y_data_top
    
    fig_y_bottom = bbox_left.y0
    fig_y_top = bbox_left.y1
    fig_height = fig_y_top - fig_y_bottom
    
    for highlight in highlights:
        start_row = highlight['start']
        end_row = highlight['end']
        color = highlight['color']
        label = highlight.get('label', '')
        linestyle = highlight.get('linestyle', default_linestyle)
        linewidth = highlight.get('linewidth', default_linewidth)
        alpha = highlight.get('alpha', default_alpha)
        label_position = highlight.get('label_position', 'middle')
        
        # Map rows to figure coordinates
        row_top_data = start_row
        row_bottom_data = end_row + 1
        
        frac_top = (row_top_data - y_data_top) / data_height
        frac_bottom = (row_bottom_data - y_data_top) / data_height
        
        fig_y_row_top = fig_y_top - frac_top * fig_height
        fig_y_row_bottom = fig_y_top - frac_bottom * fig_height
        
        # Draw FancyBboxPatch
        fancy_box = FancyBboxPatch(
            (fig_x_left, fig_y_row_bottom),
            fig_width,
            fig_y_row_top - fig_y_row_bottom,
            boxstyle="round,pad=0,rounding_size=0.008",
            transform=fig.transFigure,
            facecolor='none',
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
            clip_on=False,
        )
        fig.patches.append(fancy_box)
        
        # Add label
        if label:
            # Get per-highlight fontsize or use default
            fontsize = highlight.get('fontsize', label_fontsize)
            
            # Determine label Y position
            if label_position == 'top':
                label_y = fig_y_row_top
                va = 'bottom'
            elif label_position == 'bottom':
                label_y = fig_y_row_bottom
                va = 'top'
            else:  # middle
                label_y = (fig_y_row_top + fig_y_row_bottom) / 2
                va = 'center'
            
            # Determine label X position
            if label_side == 'left':
                label_x = fig_x_left - 0.01
                ha = 'right'
            else:  # right
                label_x = fig_x_right + 0.01
                ha = 'left'
            
            fig.text(
                label_x, label_y, label,
                color=color,
                fontsize=fontsize,
                fontweight='bold',
                fontstyle='italic',
                ha=ha,
                va=va,
                transform=fig.transFigure,
            )


# =============================================================================
# PUBLICATION-QUALITY MATRIX PLOTTER
# =============================================================================

def plot_flex_matrices_publication(
    fig: plt.Figure,
    axes: List[plt.Axes],
    result,  # AlignmentResult
    X: str,
    Y: str,
    s: int,
    e: int,
    regions: Optional[List[dict]] = None,
    # Heatmap styling
    colormap: str = 'Reds',
    annot_fontsize: float = 7,
    # Path styling
    marker_color: str = '#00C853',
    marker_width: float = 2.5,
    path_box_alpha: float = 0.4,
    path_box_highlight_alpha: float = 0.9,
    path_zorder: int = 10,
    path_color_bg: str = 'black',
    path_width_factor_bg: float = 0.6,
    # Jump styling
    jump_color: str = '#FFD600',
    jump_width: float = 3.5,
    jump_zorder: int = 9,
    # Nucleotide labels
    nuc_label_x_offset: float = -0.04,
    nuc_label_y_offset: float = 0.0,
    nuc_label_fontsize: float = 10,
    # Title styling
    title_pad: float = 20,
    title_fontsize: float = 10,
    # Region backgrounds (figure-level)
    region_margin_left: float = 0.03,
    region_margin_right: float = 0.07,
    region_label_offset: float = 0.01,
    region_label_fontsize: float = 14.0,
    region_fill_alpha: float = 0.35,
    # Leader/closer boxes (figure-level)
    leader_closer_margin_left: float = 0.002,
    leader_closer_margin_right: float = 0.002,
    leader_closer_fontsize: float = 8,
    leader_closer_box_alpha: float = 1.0,
    show_leader_closer: bool = True,
    # Row highlights (figure-level) - for Z*, X*, etc.
    row_highlights: Optional[List[dict]] = None,
    row_highlight_fontsize: float = 8,
    # Axis border
    show_axis_border: bool = True,
    border_color: str = '#888888',
    border_linewidth: float = 0.8,
    # Color mapping
    nt_color_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Draw publication-quality DP matrices on provided axes.
    
    This function is designed for embedding in compound figures. It draws
    the three Gotoh DP matrices (Yg, M, Xg) with:
    - Continuous region background bands spanning all matrices
    - Leader/closer FancyBboxPatch spanning all matrices
    - Rectangle-based path markers for better visibility
    - Nucleotide labels positioned using axis transforms
    
    Parameters
    ----------
    fig : matplotlib Figure
        Figure to draw on (needed for figure-level annotations).
    axes : list of 3 Axes
        Axes for [Yg, M, Xg] matrices.
    result : AlignmentResult
        Result from align_single_block with return_data=True.
    X : str
        Reference sequence.
    Y : str
        Query sequence.
    s : int
        Leader row index (last row of A region).
    e : int
        End of flexible block Z (e = s + len(Z)).
    regions : list of dict, optional
        Region specifications for background fills. Each dict has:
        - 'start': int - starting row (1-indexed)
        - 'end': int - ending row (1-indexed)
        - 'label': str - region label
        - 'fill_color': str - background color
        - 'label_color': str - label text color
    colormap : str
        Seaborn colormap name for heatmaps.
    marker_color : str
        Color for highlighted path boxes.
    marker_width : float
        Line width for path boxes.
    path_box_alpha : float
        Alpha for background path boxes.
    path_box_highlight_alpha : float
        Alpha for highlighted path boxes.
    jump_color : str
        Color for jump lines.
    jump_width : float
        Line width for jumps.
    nuc_label_x_offset : float
        X offset for Y-axis nucleotide labels (axes transform).
    nuc_label_y_offset : float
        Y offset for X-axis nucleotide labels (axes transform).
    nuc_label_fontsize : float
        Font size for nucleotide labels.
    title_pad : float
        Padding between title and plot.
    region_margin_left : float
        Left margin for region backgrounds (figure coords).
    region_margin_right : float
        Right margin for region backgrounds (figure coords).
    region_label_offset : float
        Distance from region edge to label (figure coords).
    region_label_fontsize : float
        Font size for region labels.
    region_fill_alpha : float
        Alpha for region fills.
    leader_closer_margin_left : float
        Left margin for leader/closer boxes (figure coords).
    leader_closer_margin_right : float
        Right margin for leader/closer boxes (figure coords).
    leader_closer_fontsize : float
        Font size for leader/closer labels.
    show_leader_closer : bool
        Whether to show leader/closer boxes.
    row_highlights : list of dict, optional
        Additional row highlight specifications (like Z*). Each dict has:
        - 'start': int - starting row (required)
        - 'end': int - ending row (required)
        - 'label': str - label text (optional)
        - 'color': str - box and label color (required)
        - 'linestyle': str - line style (optional, default ':')
        - 'label_position': str - 'top', 'middle', or 'bottom' (optional)
    row_highlight_fontsize : float
        Font size for row highlight labels.
    nt_color_map : dict, optional
        Mapping from nucleotide to color.
    
    Examples
    --------
    >>> # In a compound figure
    >>> fig = plt.figure(figsize=(10, 8))
    >>> gs = gridspec.GridSpec(2, 1)
    >>> gs_B = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1])
    >>> axes = [fig.add_subplot(gs_B[i]) for i in range(3)]
    >>> 
    >>> regions = [
    ...     {'start': 1, 'end': s, 'label': 'A', 
    ...      'fill_color': '#cce5ff', 'label_color': '#004085'},
    ...     {'start': s+1, 'end': e, 'label': 'Z',
    ...      'fill_color': '#d4edda', 'label_color': '#155724'},
    ...     {'start': e+1, 'end': n, 'label': 'B',
    ...      'fill_color': '#f8d7da', 'label_color': '#721c24'},
    ... ]
    >>> plot_flex_matrices_publication(fig, axes, result, X, Y, s, e, regions=regions)
    """
    if nt_color_map is None:
        nt_color_map = NT_COLOR
    
    if result.data is None:
        raise ValueError(
            "plot_flex_matrices_publication requires result.data (FlexData). "
            "Run the aligner with return_data=True."
        )
    
    data = result.data
    Yg = data.Yg
    M = data.M
    Xg = data.Xg
    
    matrices = [Yg, M, Xg]
    titles = ["$Y_g$ (gap in $X$)", 
              "$M$ (match/mismatch)", 
              "$X_g$ (gap in $Y$)"]
    state_for_panel = [0, 1, 2]
    
    # Common vmin/vmax ignoring -inf
    finite_vals = np.concatenate([mat[np.isfinite(mat)] for mat in matrices])
    vmin, vmax = finite_vals.min(), finite_vals.max()
    
    # Colormap with NaN as gray
    cmap = sns.color_palette(colormap, as_cmap=True)
    cmap.set_bad(color="grey")
    
    # Labels
    xticklabels = [""] + list(Y)
    yticklabels = [""] + list(X)
    
    # Make axes backgrounds transparent
    for ax in axes:
        ax.set_facecolor('none')
        ax.patch.set_alpha(0)
    
    # Draw heatmaps
    for ax, mat, title in zip(axes, matrices, titles):
        mat_plot = mat.copy()
        mat_plot[~np.isfinite(mat_plot)] = np.nan
        
        sns.heatmap(
            mat_plot,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar=False,
            annot=True,
            fmt=".0f",
            xticklabels=False,
            yticklabels=False,
            annot_kws={'fontsize': annot_fontsize},
        )
        ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(top=False, bottom=False, left=False, right=False,
                      labeltop=False, labelbottom=False, labelleft=False, labelright=False)
    
    # Y-axis nucleotide labels (on first axis only)
    ax_left = axes[0]
    for i, lab in enumerate(yticklabels):
        if lab:
            ax_left.text(nuc_label_x_offset, i + 0.5, lab, 
                        fontsize=nuc_label_fontsize, fontweight='bold',
                        color=nt_color_map.get(lab, 'black'),
                        ha='right', va='center',
                        transform=ax_left.get_yaxis_transform())
    
    # X-axis nucleotide labels (on all axes, at top)
    for ax in axes:
        for j, lab in enumerate(xticklabels):
            if lab:
                ax.text(j + 0.5, 1.0 + abs(nuc_label_y_offset), lab,
                       fontsize=nuc_label_fontsize, fontweight='bold',
                       color=nt_color_map.get(lab, 'black'),
                       ha='center', va='bottom',
                       transform=ax.get_xaxis_transform())
    
    # Draw axis borders
    if show_axis_border:
        for ax in axes:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            border = Rectangle(
                (xlim[0], min(ylim)), 
                xlim[1] - xlim[0], 
                abs(ylim[1] - ylim[0]),
                fill=False, edgecolor=border_color, linewidth=border_linewidth,
                clip_on=False, zorder=100)
            ax.add_patch(border)
    
    # Draw figure-level region backgrounds
    if regions:
        draw_figure_region_backgrounds(
            fig=fig,
            axes=axes,
            regions=regions,
            margin_left=region_margin_left,
            margin_right=region_margin_right,
            label_offset=region_label_offset,
            label_fontsize=region_label_fontsize,
            fill_alpha=region_fill_alpha,
        )
    
    # Draw leader/closer boxes
    if show_leader_closer:
        leader_row = s
        closer_row = e + 1
        
        draw_figure_leader_closer_boxes(
            fig=fig,
            axes=axes,
            leader_row=leader_row,
            closer_row=closer_row,
            margin_left=leader_closer_margin_left,
            margin_right=leader_closer_margin_right,
            label_fontsize=leader_closer_fontsize,
            alpha=leader_closer_box_alpha,
        )
    
    # Draw row highlights (Z*, X*, etc.)
    if row_highlights:
        draw_figure_row_highlights(
            fig=fig,
            axes=axes,
            highlights=row_highlights,
            margin_left=leader_closer_margin_left,
            margin_right=leader_closer_margin_right,
            label_fontsize=row_highlight_fontsize,
        )
    
    # Map state id -> axis
    state_to_ax = {state_id: ax for state_id, ax in zip(state_for_panel, axes)}
    
    # Path overlay with rectangles
    path = getattr(result, "path", None)
    if path:
        for (i, j, s_state) in path:
            # Background on all panels
            for ax in axes:
                rect = Rectangle(
                    (j, i), 1, 1,
                    fill=False,
                    edgecolor=path_color_bg,
                    linewidth=marker_width * path_width_factor_bg,
                    alpha=path_box_alpha,
                    zorder=path_zorder,
                )
                ax.add_patch(rect)
            
            # Highlighted on state's panel
            ax_state = state_to_ax.get(s_state)
            if ax_state is not None:
                rect = Rectangle(
                    (j, i), 1, 1,
                    fill=False,
                    edgecolor=marker_color,
                    linewidth=marker_width,
                    alpha=path_box_highlight_alpha,
                    zorder=path_zorder,
                )
                ax_state.add_patch(rect)
    
    # Jump overlay
    jumps = getattr(result, "jumps", None)
    if jumps:
        for jmp in jumps:
            state_id = int(jmp.state)
            col = int(jmp.col)
            r0 = int(jmp.from_row)
            r1 = int(jmp.to_row)
            
            is_terminal = r0 > r1
            
            if is_terminal:
                x0 = col + 0.5
                x1 = col + 1.0
                y0 = r1 + 1.0
                y1 = r0
            elif state_id == 1:
                x0 = col - 0.5
                y0 = r0 + 1.
                x1 = col + 0.5
                y1 = r1
            elif state_id == 2:
                x0 = x1 = col + 0.5
                y0 = r0 + 1
                y1 = r1
            else:
                continue
            
            for st, ax in zip(state_for_panel, axes):
                if is_terminal or st == state_id:
                    color = jump_color
                    alpha = path_box_highlight_alpha
                    lw = jump_width
                else:
                    color = path_color_bg
                    alpha = path_box_alpha
                    lw = jump_width * path_width_factor_bg
                
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
                       alpha=alpha, solid_capstyle='round', zorder=jump_zorder)

