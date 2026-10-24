"""Convergence overlay chart: source-model vs Novomodelo lower/upper bounds."""

from __future__ import annotations

import polars as pl

from novomodelo_bridge.comparators.html_report import (
    COLOR_NEWAVE,
    COLOR_NOVOMODELO,
)
from novomodelo_bridge.ui.html.plotly import plotly_div as _plotly_div


def _bounds(conv: pl.DataFrame, column: str) -> dict[int, float]:
    """Per-iteration values of *column*, skipping iterations where it is null
    (a source-model bound that overflowed its print field)."""
    return {
        int(row["iteration"]): float(row[column])
        for row in conv.iter_rows(named=True)
        if row[column] is not None
    }


def _line(bound: dict[int, float], name: str, line: dict) -> dict:
    iters = sorted(bound)
    return {
        "x": iters,
        "y": [bound[i] for i in iters],
        "name": name,
        "type": "scatter",
        "mode": "lines",
        "line": line,
    }


def convergence_chart(
    nw_conv: pl.DataFrame,
    novomodelo_conv: pl.DataFrame,
    reference_label: str = "NEWAVE",
) -> str:
    """Convergence overlay: The source model vs Novomodelo lower/upper bounds.

    Accepts raw convergence DataFrames directly so it can show the source model data
    even when Novomodelo convergence is empty. Each bound is drawn over the
    iterations where it is present.
    """
    series = [
        (
            _bounds(nw_conv, "lower_bound"),
            f"{reference_label} ZINF",
            {"color": COLOR_NEWAVE},
        ),
        (
            _bounds(nw_conv, "upper_bound_mean"),
            f"{reference_label} ZSUP",
            {"color": COLOR_NEWAVE, "dash": "dash"},
        ),
        (
            _bounds(novomodelo_conv, "lower_bound"),
            "Novomodelo Lower",
            {"color": COLOR_NOVOMODELO},
        ),
        (
            _bounds(novomodelo_conv, "upper_bound_mean"),
            "Novomodelo Upper",
            {"color": COLOR_NOVOMODELO, "dash": "dash"},
        ),
    ]
    traces = [_line(bound, name, line) for bound, name, line in series if bound]
    if not traces:
        return "<p>No convergence data available.</p>"

    layout = {
        "title": f"Convergence: {reference_label} vs Novomodelo",
        "xaxis": {"title": "Iteration"},
        "yaxis": {"title": "Cost (R$)", "type": "log"},
    }

    return _plotly_div(traces, layout)
