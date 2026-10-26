"""Convergence overlay chart: source-model vs Novomodelo lower/upper bounds."""

from __future__ import annotations

import math

import polars as pl

from novomodelo_bridge.comparators.html_report import (
    COLOR_NEWAVE,
    COLOR_NOVOMODELO,
)
from novomodelo_bridge.ui.html.plotly import plotly_div as _plotly_div


def _bounds(conv: pl.DataFrame, column: str) -> dict[int, float]:
    """Per-iteration values of *column*, skipping iterations where it is null
    (a source-model bound that overflowed its print field) or the column is
    absent."""
    if column not in conv.columns:
        return {}
    return {
        int(row["iteration"]): float(row[column])
        for row in conv.iter_rows(named=True)
        if row[column] is not None
    }


def _running_min(bound: dict[int, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    best = math.inf
    for it in sorted(bound):
        best = min(best, bound[it])
        out[it] = best
    return out


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
    iterations where it is present. The source model's ``upper_bound_best``
    (its printed ZSUP, the lowest upper bound so far) is paired with Novomodelo's
    running minimum, never with Novomodelo's per-iteration ``upper_bound_mean``.
    """
    source_best = _bounds(nw_conv, "upper_bound_best")
    novomodelo_upper = _bounds(novomodelo_conv, "upper_bound_mean")
    series = [
        (
            _bounds(nw_conv, "lower_bound"),
            f"{reference_label} ZINF",
            {"color": COLOR_NEWAVE},
        ),
        (
            _bounds(nw_conv, "upper_bound_mean"),
            f"{reference_label} ZSUP ITER",
            {"color": COLOR_NEWAVE, "dash": "dash"},
        ),
        (
            source_best,
            f"{reference_label} ZSUP",
            {"color": COLOR_NEWAVE, "dash": "dot"},
        ),
        (
            _bounds(novomodelo_conv, "lower_bound"),
            "Novomodelo Lower",
            {"color": COLOR_NOVOMODELO},
        ),
        (
            novomodelo_upper,
            "Novomodelo Upper",
            {"color": COLOR_NOVOMODELO, "dash": "dash"},
        ),
        (
            _running_min(novomodelo_upper) if source_best else {},
            "Novomodelo Upper, running min",
            {"color": COLOR_NOVOMODELO, "dash": "dot"},
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
