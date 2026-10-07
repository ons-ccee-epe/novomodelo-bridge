"""Convergence overlay chart: source-model vs Cobre lower/upper bounds."""

from __future__ import annotations

import polars as pl

from cobre_bridge.comparators.html_report import (
    COLOR_COBRE,
    COLOR_NEWAVE,
)
from cobre_bridge.ui.html.plotly import plotly_div as _plotly_div


def convergence_chart(
    nw_conv: pl.DataFrame,
    cobre_conv: pl.DataFrame,
    reference_label: str = "NEWAVE",
) -> str:
    """Convergence overlay: The source model vs Cobre lower/upper bounds.

    Accepts raw convergence DataFrames directly so it can show the source model data
    even when Cobre convergence is empty.
    """
    lb_nw: dict[int, float] = {}
    ub_nw: dict[int, float] = {}
    lb_cb: dict[int, float] = {}
    ub_cb: dict[int, float] = {}

    if not nw_conv.is_empty():
        for row in nw_conv.iter_rows(named=True):
            it = int(row["iteration"])
            lb_nw[it] = float(row["lower_bound"])
            ub_nw[it] = float(row["upper_bound_mean"])

    if not cobre_conv.is_empty():
        for row in cobre_conv.iter_rows(named=True):
            it = int(row["iteration"])
            lb_cb[it] = float(row["lower_bound"])
            ub_cb[it] = float(row["upper_bound_mean"])

    iters = sorted(set(lb_nw) | set(lb_cb))
    if not iters:
        return "<p>No convergence data available.</p>"

    traces: list[dict] = []

    if lb_nw:
        nw_iters = sorted(lb_nw)
        traces.append(
            {
                "x": nw_iters,
                "y": [lb_nw[i] for i in nw_iters],
                "name": f"{reference_label} ZINF",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NEWAVE},
            }
        )
        traces.append(
            {
                "x": nw_iters,
                "y": [ub_nw.get(i) for i in nw_iters],
                "name": f"{reference_label} ZSUP",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NEWAVE, "dash": "dash"},
            }
        )

    if lb_cb:
        cb_iters = sorted(lb_cb)
        traces.append(
            {
                "x": cb_iters,
                "y": [lb_cb[i] for i in cb_iters],
                "name": "Cobre Lower",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_COBRE},
            }
        )
        traces.append(
            {
                "x": cb_iters,
                "y": [ub_cb.get(i) for i in cb_iters],
                "name": "Cobre Upper",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_COBRE, "dash": "dash"},
            }
        )

    layout = {
        "title": f"Convergence: {reference_label} vs Cobre",
        "xaxis": {"title": "Iteration"},
        "yaxis": {"title": "Cost (R$)", "type": "log"},
    }

    return _plotly_div(traces, layout)
