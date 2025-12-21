"""
Scoring system visualization for NW-flex.

This module contains functions for visualizing alignment scoring parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

def plot_score_system(
    score_matrix: np.ndarray,
    gap_open: float,
    gap_extend: float,
    alphabet_to_index: Dict[str, int],
    figsize: Tuple[float, float] = (5, 3.5),
) -> plt.Figure:
    """
    Display the alignment scoring system as a heatmap of the substitution
    matrix and a panel showing the affine gap penalties.

    Parameters
    ----------
    score_matrix : ndarray
        Substitution matrix (4x4 for ACGT).
    gap_open : float
        Gap open penalty.
    gap_extend : float
        Gap extend penalty.
    alphabet_to_index : dict
        Mapping from alphabet characters to indices (e.g., {'A': 0, 'C': 1, ...}).
    figsize : tuple, optional
        Figure size (width, height).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    def tc(val, v, cmap):  # auto text color
        rgb = cmap((val + v) / (2 * v))
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        return "white" if lum < 0.5 else "black"

    s = np.asarray(score_matrix)
    labs = list(alphabet_to_index.keys())

    cmap = plt.cm.RdBu.copy()  # negative=red, positive=blue
    cmap.set_bad("white")
    v = np.max(np.abs(np.concatenate([s.ravel(), [gap_open, gap_extend]])))

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # ---------------- substitution ----------------
    ax[0].imshow(s, cmap=cmap, vmin=-v, vmax=v)
    for i in range(4):
        for j in range(4):
            val = s[i, j]
            ax[0].text(j, i, f"{val:.0f}", ha="center", va="center",
                       color=tc(val, v, cmap))

    ax[0].set_xticks(np.arange(4), labs)
    ax[0].set_yticks(np.arange(4), labs)
    ax[0].set_title("Substitution Matrix\ns(a, b)")
    ax[0].set_xlabel("Seq2 Base")
    ax[0].set_ylabel("Seq1 Base")
    for sp in ax[0].spines.values():
        sp.set_visible(False)

    # ---------------- gaps ----------------
    g = np.full((4, 4), np.nan)
    g[0, 0], g[1, 0] = gap_open, gap_extend

    ax[1].imshow(g, cmap=cmap, vmin=-v, vmax=v)
    ax[1].text(0, 0, f"{gap_open:.0f}", ha="center", va="center",
               color=tc(gap_open, v, cmap))
    ax[1].text(0, 1, f"{gap_extend:.0f}", ha="center", va="center",
               color=tc(gap_extend, v, cmap))

    ax[1].set_xticks([])
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels(["Gap Open", "Gap Extend"])
    ax[1].set_title("Affine Gaps")
    for sp in ax[1].spines.values():
        sp.set_visible(False)

    plt.tight_layout()
    return fig
