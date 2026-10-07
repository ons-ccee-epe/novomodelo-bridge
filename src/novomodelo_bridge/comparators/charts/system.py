"""System, REE, and per-bus energy-balance tab charts."""

from __future__ import annotations

import polars as pl

from cobre_bridge.comparators import analyze
from cobre_bridge.comparators.charts._shared import (
    _BAND_FILL,
    _BAND_LINE,
    _REAL_SUBMARKET_ORDER,
    _aggregate_percentile_traces,
)
from cobre_bridge.comparators.html_report import (
    COLOR_COBRE,
    COLOR_NEWAVE,
)
from cobre_bridge.comparators.model import ResultComparison
from cobre_bridge.ui.html.plotly import plotly_div as _plotly_div


def system_comparison_chart(
    results: list[ResultComparison],
    variable: str,
    title: str,
    pct_df: pl.DataFrame | None = None,
    reference_label: str = "NEWAVE",
) -> str:
    """Line chart comparing a system variable by stage with p10-p90 band."""
    bus_data = [r for r in results if r.entity_type == "bus" and r.variable == variable]
    if not bus_data:
        return f"<p>No {variable} data available.</p>"

    nw_by_stage, cb_by_stage, matched_ids = analyze.per_stage_sum_from_results(
        results, "bus", variable
    )
    stages = sorted(set(nw_by_stage) | set(cb_by_stage))
    traces = _aggregate_percentile_traces(pct_df, variable, stages, matched_ids)
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
                "name": "Cobre Mean",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_COBRE, "width": 2},
            },
        ]
    )

    layout = {
        "title": title,
        "xaxis": {"title": "Stage"},
        "yaxis": {"title": variable},
    }

    return _plotly_div(traces, layout)


def ree_energy_chart(
    results: list[ResultComparison],
    variable: str,
    title: str,
    reference_label: str = "NEWAVE",
) -> str:
    """Line chart comparing an REE energy variable by stage.

    Mirrors :func:`system_comparison_chart`'s aggregate-line
    shape (the source model's own value vs Cobre's, summed across every
    matched REE per stage), keyed on ``entity_type == "ree"`` instead of
    ``"bus"``. REE carries no Cobre percentile band --
    :class:`~cobre_bridge.comparators.model.PercentileData` has no ``ree``
    field -- so this omits the optional p10-p90 overlay entirely rather than
    fabricating one.
    """
    ree_data = [r for r in results if r.entity_type == "ree" and r.variable == variable]
    if not ree_data:
        return f"<p>No {variable} data available.</p>"

    nw_by_stage, cb_by_stage, _matched_ids = analyze.per_stage_sum_from_results(
        results, "ree", variable
    )
    stages = sorted(set(nw_by_stage) | set(cb_by_stage))
    traces = [
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
            "name": "Cobre Mean",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_COBRE, "width": 2},
        },
    ]

    layout = {
        "title": title,
        "xaxis": {"title": "Stage"},
        "yaxis": {"title": variable},
    }

    return _plotly_div(traces, layout)


_BALANCE_VARS: list[tuple[str, str, str, str]] = [
    # (display_label, newave_var, cobre_var, unit)
    ("Hydro Generation", "GHTOT", "hydro_gen_mw", "MW"),
    ("Thermal Generation", "GTERM", "thermal_gen_mw", "MW"),
    ("Net Load", "NET_LOAD", "net_load_mw", "MW"),
    ("Deficit", "DEFT", "deficit_mw", "MW"),
    ("Excess", "EXCESSO", "excess_mw", "MW"),
]


def system_per_bus_chart(
    results: list[ResultComparison],
    variable: str,
    title: str,
    pct_df: pl.DataFrame | None = None,
    reference_label: str = "NEWAVE",
) -> str:
    """2x2 faceted per-bus chart with p10-p90 bands.

    Restricted to the real submarket buses (:data:`_REAL_SUBMARKET_ORDER`) so
    the grid is a clean 2x2 — fictitious/transhipment nodes are never faceted.
    """
    bus_data = [r for r in results if r.entity_type == "bus" and r.variable == variable]
    if not bus_data:
        return f"<p>No {variable} data available.</p>"

    buses, pct_by_eid = analyze.bus_groups_and_pct(results, variable, pct_df)
    p10_col = f"{variable}_p10"
    p90_col = f"{variable}_p90"

    # Real submarkets only, fixed order for a clean 2x2 (fictitious buses excluded).
    ordered = [b for b in _REAL_SUBMARKET_ORDER if b in buses]
    if not ordered:
        return f"<p>No {variable} data available.</p>"

    # 2x2 grid using plotly subplots via xaxis/yaxis domains.
    n = len(ordered)
    ncols = 2
    nrows = (n + 1) // ncols
    traces: list[dict] = []
    layout: dict = {"title": title}
    first = True

    for idx, bus_name in enumerate(ordered):
        rows_list = buses[bus_name]
        row_i = idx // ncols
        col_i = idx % ncols
        ax_idx = idx + 1

        xa = f"x{ax_idx}" if ax_idx > 1 else "x"
        ya = f"y{ax_idx}" if ax_idx > 1 else "y"

        # Fixed-stride domains (0.52/0.47/0.44) — intentionally NOT facet_grid:
        # its gap formula yields different col_w/row_h and would drift this
        # chart's golden.
        x0 = col_i * 0.52
        x1 = x0 + 0.47
        y1 = 1.0 - row_i * 0.52
        y0 = y1 - 0.44

        xa_key = f"xaxis{ax_idx}" if ax_idx > 1 else "xaxis"
        ya_key = f"yaxis{ax_idx}" if ax_idx > 1 else "yaxis"
        layout[xa_key] = {
            "domain": [round(x0, 3), round(x1, 3)],
            "title": "Stage" if row_i == nrows - 1 else "",
            "anchor": ya,
        }
        layout[ya_key] = {
            "domain": [round(y0, 3), round(y1, 3)],
            "title": bus_name,
            "anchor": xa,
        }

        rows_sorted = sorted(rows_list, key=lambda r: r.stage)
        stages = [r.stage for r in rows_sorted]
        nw = [r.newave_value for r in rows_sorted]
        cb = [r.cobre_value for r in rows_sorted]

        # P10-P90 band for this bus.
        cobre_id = rows_sorted[0].cobre_id if rows_sorted else None
        bus_pct = pct_by_eid.get(cobre_id, {}) if cobre_id is not None else {}
        if bus_pct:
            p10 = [float(bus_pct.get(s, {}).get(p10_col, 0) or 0) for s in stages]
            p90 = [float(bus_pct.get(s, {}).get(p90_col, 0) or 0) for s in stages]
            traces.append(
                {
                    "x": stages + stages[::-1],
                    "y": p90 + p10[::-1],
                    "fill": "toself",
                    "fillcolor": _BAND_FILL,
                    "line": {"color": _BAND_LINE},
                    "name": "Cobre P10–P90",
                    "hoverinfo": "skip",
                    "type": "scatter",
                    "xaxis": xa,
                    "yaxis": ya,
                    "legendgroup": "band",
                    "showlegend": first,
                }
            )

        traces.append(
            {
                "x": stages,
                "y": nw,
                "name": reference_label,
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NEWAVE, "width": 2},
                "xaxis": xa,
                "yaxis": ya,
                "legendgroup": "nw",
                "showlegend": first,
            }
        )
        traces.append(
            {
                "x": stages,
                "y": cb,
                "name": "Cobre Mean",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_COBRE, "width": 2},
                "xaxis": xa,
                "yaxis": ya,
                "legendgroup": "cb",
                "showlegend": first,
            }
        )
        first = False

    return _plotly_div(traces, layout, height=nrows * 300 + 80)
