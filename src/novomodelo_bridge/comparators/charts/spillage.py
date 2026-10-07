"""System spillage energy chart (MWmes): source-model vs Cobre aggregate."""

from __future__ import annotations

import polars as pl

from cobre_bridge.comparators import analyze
from cobre_bridge.comparators.html_report import (
    COLOR_COBRE,
    COLOR_NEWAVE,
)
from cobre_bridge.comparators.model import ResultComparison
from cobre_bridge.ui.html.plotly import LEGEND_DEFAULTS as _LEGEND
from cobre_bridge.ui.html.plotly import MARGIN_DEFAULTS as _MARGIN
from cobre_bridge.ui.html.plotly import facet_grid
from cobre_bridge.ui.html.plotly import plotly_div as _plotly_div


def system_spillage_energy_chart(
    results: list[ResultComparison],
    cobre_spill_energy: pl.DataFrame,
    reference_label: str = "NEWAVE",
) -> str:
    """Three-panel chart of system spillage in MWmes.

    Each panel pairs a source-model trace (``VERTOT`` / ``VERTcont`` / ``VERTfio``)
    against the matching Cobre aggregate (``total_mw`` / ``reservoir_mw`` /
    ``rorov_mw``).  Both axes are stage-average MW (MWmes).
    """
    nw_rows = [r for r in results if r.entity_type == "system_spillage"]
    if not nw_rows and cobre_spill_energy.is_empty():
        return "<p>No system spillage data available.</p>"

    nw_lookup, cb_lookup = analyze.spillage_lookups(results, cobre_spill_energy)

    panels: list[tuple[str, str, str]] = [
        ("Total (VERTOT)", "spill_energy_total_mw", "VERTOT"),
        ("Reservoir cascades (VERTcont)", "spill_energy_reservoir_mw", "VERTcont"),
        ("Run-of-river (VERTfio)", "spill_energy_rorov_mw", "VERTfio"),
    ]
    # Vertically stacked: y in [0.7,1.0], [0.35,0.65], [0.0,0.3]
    facets = facet_grid(3, ncols=1, row_gap=0.05)

    traces: list[dict] = []
    layout: dict = {
        "title": "System Spillage Energy (MWmes)",
        "showlegend": True,
        "hovermode": "x unified",
        "legend": _LEGEND,
        "margin": _MARGIN,
    }

    for idx, (panel_title, var_key, nw_label) in enumerate(panels):
        ax_idx = idx + 1
        xa = f"x{ax_idx}" if ax_idx > 1 else "x"
        ya = f"y{ax_idx}" if ax_idx > 1 else "y"
        xa_key = f"xaxis{ax_idx}" if ax_idx > 1 else "xaxis"
        ya_key = f"yaxis{ax_idx}" if ax_idx > 1 else "yaxis"

        facet = facets[idx]
        layout[xa_key] = {
            "domain": facet.x_domain,
            "anchor": ya,
            "title": "Stage" if idx == len(panels) - 1 else "",
        }
        layout[ya_key] = {
            "domain": facet.y_domain,
            "anchor": xa,
            "title": panel_title,
        }

        all_stages = sorted(
            set(nw_lookup.get(var_key, {}).keys())
            | set(cb_lookup.get(var_key, {}).keys())
        )
        if not all_stages:
            continue

        nw_y = [nw_lookup.get(var_key, {}).get(s) for s in all_stages]
        cb_y = [cb_lookup.get(var_key, {}).get(s) for s in all_stages]

        first = idx == 0
        traces.append(
            {
                "x": all_stages,
                "y": nw_y,
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
                "x": all_stages,
                "y": cb_y,
                "name": "Cobre",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_COBRE, "width": 2},
                "xaxis": xa,
                "yaxis": ya,
                "legendgroup": "cb",
                "showlegend": first,
            }
        )

    if not traces:
        return "<p>No system spillage data available.</p>"

    return _plotly_div(traces, layout, height=720)
