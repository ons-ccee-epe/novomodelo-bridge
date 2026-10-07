"""Thermal tab chart."""

from __future__ import annotations

import polars as pl

from novomodelo_bridge.comparators import analyze
from novomodelo_bridge.comparators.charts._shared import _aggregate_percentile_traces
from novomodelo_bridge.comparators.html_report import (
    COLOR_NEWAVE,
    COLOR_NOVOMODELO,
)
from novomodelo_bridge.comparators.model import ResultComparison
from novomodelo_bridge.ui.html.plotly import plotly_div as _plotly_div


def thermal_generation_chart(
    results: list[ResultComparison],
    pct_df: pl.DataFrame | None = None,
    reference_label: str = "NEWAVE",
) -> str:
    """Aggregate thermal generation comparison by stage."""
    thermal_data = [r for r in results if r.entity_type == "thermal"]
    if not thermal_data:
        return "<p>No thermal generation data.</p>"

    nw_by_stage, cb_by_stage, matched_ids = analyze.per_stage_sum_from_results(
        results, "thermal", ""
    )
    stages = sorted(set(nw_by_stage) | set(cb_by_stage))
    traces = _aggregate_percentile_traces(pct_df, "generation_mw", stages, matched_ids)
    traces.extend(
        [
            {
                "x": stages,
                "y": [nw_by_stage.get(s, 0) for s in stages],
                "name": reference_label,
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NEWAVE, "width": 2},
            },
            {
                "x": stages,
                "y": [cb_by_stage.get(s, 0) for s in stages],
                "name": "Novomodelo Mean",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NOVOMODELO, "width": 2},
            },
        ]
    )

    layout = {
        "title": "Thermal Generation",
        "xaxis": {"title": "Stage"},
        "yaxis": {"title": "MW"},
    }

    return _plotly_div(traces, layout)
