"""Network (line-interchange) tab chart: source-model vs Cobre net flow, per line."""

from __future__ import annotations

import polars as pl

from cobre_bridge.comparators.charts._shared import (
    _BAND_FILL,
    _BAND_LINE,
)
from cobre_bridge.comparators.html_report import (
    COLOR_COBRE,
    COLOR_NEWAVE,
)
from cobre_bridge.comparators.model import ResultComparison
from cobre_bridge.core.tolerances import is_effectively_infinite
from cobre_bridge.ui.html.plotly import facet_grid
from cobre_bridge.ui.html.plotly import plotly_div as _plotly_div


def line_summary_chart(
    results: list[ResultComparison],
    line_pct: pl.DataFrame | None,
    line_bounds: pl.DataFrame | None,
    line_meta: list[dict],
    reference_label: str = "NEWAVE",
) -> str:
    """Per-line small-multiples comparing the source model vs Cobre net flow.

    One panel per aligned line. Each panel shows the Cobre P10–P90 band, Cobre median,
    and the source model mean, plus dashed upper/lower capacity bounds (direct /
    −reverse).
    """
    line_data = [r for r in results if r.entity_type == "line"]
    if not line_data:
        return "<p>No line interchange data available.</p>"

    # Group by cobre_id.
    by_line: dict[int, list[ResultComparison]] = {}
    for r in line_data:
        by_line.setdefault(r.cobre_id, []).append(r)
    ordered_ids = sorted(by_line.keys())

    # Build per-line p10/p90 lookups from percentile data.
    pct_by_lid: dict[int, dict[int, dict]] = {}
    if (
        line_pct is not None
        and not line_pct.is_empty()
        and {"net_flow_mw_p10", "net_flow_mw_p90"}.issubset(line_pct.columns)
    ):
        for r in line_pct.iter_rows(named=True):
            lid = int(r["entity_id"])
            sid = int(r["stage_id"])
            pct_by_lid.setdefault(lid, {})[sid] = r

    # Per-line stage-keyed capacity bounds.
    static_caps: dict[int, tuple[float, float]] = {}
    for lm in line_meta:
        cap = lm.get("capacity", {})
        static_caps[int(lm["id"])] = (
            float(cap.get("direct_mw", 0.0) or 0.0),
            float(cap.get("reverse_mw", 0.0) or 0.0),
        )
    stage_caps: dict[int, dict[int, tuple[float, float]]] = {}
    if line_bounds is not None and not line_bounds.is_empty():
        for row in line_bounds.iter_rows(named=True):
            stage_caps.setdefault(int(row["line_id"]), {})[int(row["stage_id"])] = (
                float(row.get("direct_mw", 0.0) or 0.0),
                float(row.get("reverse_mw", 0.0) or 0.0),
            )

    ncols = 2
    nrows = (len(ordered_ids) + ncols - 1) // ncols
    # Distribute panels evenly across the [0, 1] y-domain with a
    # constant gap between rows. The previous fixed-stride formula
    # produced negative y-domains for nrows ≥ 3 (rendering the bottom
    # rows outside the chart area) which made bottom panels disappear.
    panels = facet_grid(len(ordered_ids), ncols=2)

    traces: list[dict] = []
    layout: dict = {"title": "Net Line Flow (MW)"}
    first = True

    for idx, lid in enumerate(ordered_ids):
        rows_list = sorted(by_line[lid], key=lambda r: r.stage)
        panel = panels[idx]
        row_i = panel.row
        ax_idx = idx + 1
        xa = f"x{ax_idx}" if ax_idx > 1 else "x"
        ya = f"y{ax_idx}" if ax_idx > 1 else "y"

        xa_key = f"xaxis{ax_idx}" if ax_idx > 1 else "xaxis"
        ya_key = f"yaxis{ax_idx}" if ax_idx > 1 else "yaxis"
        layout[xa_key] = {
            "domain": panel.x_domain,
            "title": "Stage" if row_i == nrows - 1 else "",
            "anchor": ya,
        }
        layout[ya_key] = {
            "domain": panel.y_domain,
            "title": rows_list[0].entity_name if rows_list else f"line {lid}",
            "anchor": xa,
        }

        stages = [r.stage for r in rows_list]
        nw = [r.newave_value for r in rows_list]
        cb = [r.cobre_value for r in rows_list]

        # P10-P90 band.
        line_pct_map = pct_by_lid.get(lid, {})
        if line_pct_map:
            p10 = [
                float(line_pct_map.get(s, {}).get("net_flow_mw_p10", 0) or 0)
                for s in stages
            ]
            p90 = [
                float(line_pct_map.get(s, {}).get("net_flow_mw_p90", 0) or 0)
                for s in stages
            ]
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

        # Capacity bound lines. Skip when both directions are effectively infinite (the
        # source model big-M sentinel 99999 used for fictitious connections) — those
        # bounds dwarf real flows and compress every other trace to a flat strip at
        # zero.
        d_static, r_static = static_caps.get(lid, (0.0, 0.0))
        finite_upper: list[tuple[int, float]] = []
        finite_lower: list[tuple[int, float]] = []
        for s in stages:
            d_cap, r_cap = stage_caps.get(lid, {}).get(s, (d_static, r_static))
            if not is_effectively_infinite(d_cap):
                finite_upper.append((s, d_cap))
            if not is_effectively_infinite(r_cap):
                finite_lower.append((s, -r_cap))
        if finite_upper:
            traces.append(
                {
                    "x": [s for s, _ in finite_upper],
                    "y": [v for _, v in finite_upper],
                    "name": "Upper bound",
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"color": "#DC4C4C", "width": 1.5, "dash": "dash"},
                    "xaxis": xa,
                    "yaxis": ya,
                    "legendgroup": "bound",
                    "showlegend": first,
                }
            )
        if finite_lower:
            traces.append(
                {
                    "x": [s for s, _ in finite_lower],
                    "y": [v for _, v in finite_lower],
                    "name": "Lower bound",
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"color": "#DC4C4C", "width": 1.5, "dash": "dash"},
                    "xaxis": xa,
                    "yaxis": ya,
                    "legendgroup": "bound",
                    "showlegend": False,
                }
            )

        # The source model + Cobre mean.
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

    return _plotly_div(traces, layout, height=max(nrows * 300 + 80, 360))
