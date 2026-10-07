"""Top-level timing-phase config, detector, and stacked-bar figure builder.

Shared by the Performance and Training tabs so a new cobre timing column is
registered once instead of drifting between two copies.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from cobre_bridge.ui.theme import PERFORMANCE_PHASE_COLORS

# The six non-overlapping top-level phases (sum ≈ iteration total). Any
# sub-component or aggregate-CPU column lives in performance_charts instead —
# see that module's docstring for the full timing column hierarchy.
TOP_LEVEL_PHASE_CONFIG: tuple[tuple[str, str, str], ...] = (
    ("forward_wall_ms", "Forward", PERFORMANCE_PHASE_COLORS["forward"]),
    ("backward_wall_ms", "Backward", PERFORMANCE_PHASE_COLORS["backward"]),
    ("cut_selection_ms", "Cut Selection", PERFORMANCE_PHASE_COLORS["lp_solve"]),
    ("lower_bound_ms", "Lower Bound Eval", "#14B8A6"),
    ("mpi_allreduce_ms", "MPI AllReduce", "#8B5CF6"),
    ("overhead_ms", "Other Overhead", PERFORMANCE_PHASE_COLORS["overhead"]),
)

TOP_LEVEL_PHASE_COLUMNS: tuple[str, ...] = tuple(
    col for col, _, _ in TOP_LEVEL_PHASE_CONFIG
)


def active_top_level_phases(timing: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return the config triples present in ``timing.columns``, in config order."""
    return [
        (col, label, color)
        for col, label, color in TOP_LEVEL_PHASE_CONFIG
        if col in timing.columns
    ]


def build_timing_stacked_figure(timing: pd.DataFrame) -> go.Figure | None:
    """Build the per-iteration stacked top-level-phase ``go.Bar`` traces.

    Traces only — callers apply their own title/layout. Returns ``None`` when
    *timing* is empty or has no recognised top-level phase column.
    """
    if timing.empty:
        return None

    phases = active_top_level_phases(timing)
    if not phases:
        return None

    # Superset fallback ported from the training tab: every real/tested
    # caller carries an `iteration` column, so this only changes the
    # untested no-column path from a KeyError to a positional index.
    iters = (
        timing["iteration"].tolist()
        if "iteration" in timing.columns
        else list(range(len(timing)))
    )

    fig = go.Figure()
    for col, label, color in phases:
        fig.add_trace(
            go.Bar(
                x=iters,
                y=timing[col].tolist(),
                name=label,
                marker_color=color,
                hovertemplate=f"{label}: %{{y:.0f}} ms<extra></extra>",
            )
        )
    return fig
