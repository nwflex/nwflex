"""Plotting helpers for the comprehensive batch sweep.

These functions previously lived here while the in-flight viz refactor
stabilized; they have since been folded into the package next to
``plot_proportion_heatmap_2d``.  This module is kept as a backwards
compatibility shim that re-exports the public names so existing
``from sweep_viz import ...`` callers keep working.
"""
from __future__ import annotations

from nwflex.simulation.viz import (
    plot_proportion_heatmap,
    plot_proportion_heatmap_rows,
    _proportion_value_fn as _proportion_value_fn_1d,
)

__all__ = [
    "plot_proportion_heatmap",
    "plot_proportion_heatmap_rows",
    "_proportion_value_fn_1d",
]
