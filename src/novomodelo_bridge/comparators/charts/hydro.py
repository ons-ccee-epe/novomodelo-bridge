"""Hydro tab charts."""

from __future__ import annotations

from typing import cast

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
from cobre_bridge.ui.html.plotly import facet_grid
from cobre_bridge.ui.html.plotly import plotly_div as _plotly_div


def cobre_aggregate_chart(
    cobre_hydro: pl.DataFrame,
    variable: str,
    title: str,
    unit: str,
    pct_df: pl.DataFrame | None = None,
    *,
    nw_sin: pl.DataFrame | None = None,
    nw_variable: str | None = None,
    nw_factor: float = 1.0,
    nw_offset: int = 0,
    matched_ids: set[int] | None = None,
    reference_label: str = "NEWAVE",
) -> str:
    """System-aggregate chart for a Cobre per-hydro variable.

    Sums *variable* across all (or matched) hydros per stage to produce a system-total
    mean line.  Adds a p10-p90 band from ``pct_df`` and an optional the source model
    SIN-level line from ``nw_sin`` (multiplied by ``nw_factor`` for unit conversion).

    Parameters
    ----------
    cobre_hydro:
        Per-hydro Cobre means with ``entity_id``, ``stage_id``, and
        ``variable`` columns.
    variable:
        Column name in ``cobre_hydro`` and percentile prefix in ``pct_df``.
    title, unit:
        Chart title and y-axis unit label.
    pct_df:
        Per-hydro Cobre percentiles for the same variable.
    nw_sin:
        Long-format the source model SIN DataFrame (``newave_code``, ``stage``,
        ``variable``, ``value``) — typically read by ``read_medias_sin``.
    nw_variable:
        Variable name to filter in ``nw_sin`` (e.g. ``"EARMF"``).
    nw_factor:
        Multiplicative factor applied to the source model values for unit alignment
        (e.g. ``730`` to convert MWmes → MWh).
    nw_offset:
        Subtracted from the source model ``stage`` to align with Cobre ``stage_id`` (the
        source model columns are numbered from the study start month).
    matched_ids:
        Optional subset of Cobre hydro IDs to include — keeps the
        aggregate consistent with comparisons that only cover matched
        plants.
    """
    if cobre_hydro.is_empty() or variable not in cobre_hydro.columns:
        return f"<p>No {variable} data available.</p>"

    cobre_by_stage, nw_by_stage = analyze.cobre_sum_and_newave_sin(
        cobre_hydro,
        variable,
        nw_sin,
        nw_variable,
        nw_factor,
        nw_offset,
        matched_ids,
    )

    stages = sorted(set(cobre_by_stage) | set(nw_by_stage))

    traces = _aggregate_percentile_traces(pct_df, variable, stages, matched_ids)
    if nw_by_stage:
        traces.append(
            {
                "x": stages,
                "y": [nw_by_stage.get(s, 0) for s in stages],
                "name": f"{reference_label} SIN",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NEWAVE, "width": 2},
            }
        )
    traces.append(
        {
            "x": stages,
            "y": [cobre_by_stage.get(s, 0) for s in stages],
            "name": "Cobre Mean",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_COBRE, "width": 2},
        }
    )

    layout = {
        "title": title,
        "xaxis": {"title": "Stage"},
        "yaxis": {"title": f"{title} ({unit})"},
    }

    return _plotly_div(traces, layout)


def hydro_per_bus_chart(
    results: list[ResultComparison],
    variable: str,
    title: str,
    pct_df: pl.DataFrame | None,
    hydro_meta: dict[int, dict],
    bus_meta: dict[int, dict],
    reference_label: str = "NEWAVE",
) -> str:
    """Per-bus faceted hydro comparison for *variable*.

    Aggregates hydro-plant ResultComparison rows by the plant's owning bus (taken from
    ``hydro_meta[cobre_id]["bus_ids"]``, the hydro_bus_generation-partition-sourced
    label -- see ``analyze._bus_name_lookups``), then renders a small-multiples grid
    (one panel per non-fictitious bus, same layout convention as ``line_summary_chart``)
    with the source model + Cobre traces and an optional Cobre P10–P90 band summed
    across each bus's plants.

    Returns a short ``<p>`` fallback when the variable is absent on
    both sides or no plants can be mapped to buses.
    """
    hydro_data = [
        r for r in results if r.entity_type == "hydro" and r.variable == variable
    ]
    if not hydro_data:
        return f"<p>No hydro {variable} data.</p>"

    # Per-(bus, stage) the source model/Cobre sums (analyze owns the roll-up; the
    # bus-name resolution and NOFICT skip live there).
    per_bus = analyze.per_bus_sums_from_results(results, variable, hydro_meta, bus_meta)
    per_bus_nw: dict[str, dict[int, float]] = {
        bus_name: cast("dict[int, float]", agg["nw"])
        for bus_name, agg in per_bus.items()
    }
    per_bus_cb: dict[str, dict[int, float]] = {
        bus_name: cast("dict[int, float]", agg["cb"])
        for bus_name, agg in per_bus.items()
    }
    per_bus_ids: dict[str, set[int]] = {
        bus_name: cast("set[int]", agg["ids"]) for bus_name, agg in per_bus.items()
    }

    if not per_bus_nw:
        return f"<p>No hydro {variable} data mapped to buses.</p>"

    # Real submarkets only, fixed order for a clean 2x2 (fictitious buses excluded).
    ordered = [b for b in _REAL_SUBMARKET_ORDER if b in per_bus_nw]

    # Build aggregate-percentile lookups per bus (sum of plant p10/p90).
    per_bus_pct = analyze.per_bus_band_from_pct(pct_df, variable, per_bus_ids)

    ncols = 2
    nrows = (len(ordered) + ncols - 1) // ncols
    panels = facet_grid(len(ordered), ncols=2)

    traces: list[dict] = []
    layout: dict = {"title": title}
    first = True

    for idx, bus_name in enumerate(ordered):
        stages = sorted(set(per_bus_nw[bus_name]) | set(per_bus_cb[bus_name]))
        nw = [per_bus_nw[bus_name].get(s, 0.0) for s in stages]
        cb = [per_bus_cb[bus_name].get(s, 0.0) for s in stages]
        if not stages:
            continue

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
            "title": bus_name,
            "anchor": xa,
        }

        bus_pct = per_bus_pct.get(bus_name, {})
        if bus_pct:
            p10 = [bus_pct.get(s, (0.0, 0.0))[0] for s in stages]
            p90 = [bus_pct.get(s, (0.0, 0.0))[1] for s in stages]
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

    return _plotly_div(traces, layout, height=max(nrows * 300 + 80, 360))


def hydro_aggregate_chart(
    results: list[ResultComparison],
    variable: str,
    title: str,
    pct_df: pl.DataFrame | None = None,
    reference_label: str = "NEWAVE",
) -> str:
    """Aggregate hydro comparison by stage with optional p10-p90 band."""
    hydro_data = [
        r for r in results if r.entity_type == "hydro" and r.variable == variable
    ]
    if not hydro_data:
        return f"<p>No hydro {variable} data.</p>"

    nw_by_stage, cb_by_stage, matched_ids = analyze.per_stage_sum_from_results(
        results, "hydro", variable
    )
    stages = sorted(set(nw_by_stage) | set(cb_by_stage))

    # Band traces first (rendered behind the lines).
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


def _hydro_per_stage_sum(
    df: pl.DataFrame | None,
    variable: str,
    matched_ids: set[int] | None,
) -> dict[int, float]:
    """Sum *variable* across (matched) hydros per ``stage_id``.

    Returns an empty dict when the frame is missing/empty or the column is
    absent.  Used by the slack-aggregate / slack-per-bus chart helpers to
    collapse per-(entity_id, stage_id) frames into a per-stage SIN total.
    """
    return analyze.per_stage_sum_from_frame(df, variable, matched_ids)


def hydro_slack_aggregate_chart(
    cobre_hydro: pl.DataFrame,
    nw_slacks: pl.DataFrame | None,
    variable: str,
    title: str,
    pct_df: pl.DataFrame | None = None,
    matched_ids: set[int] | None = None,
    unit: str = "m³/s",
    reference_label: str = "NEWAVE",
) -> str:
    """SIN-total slack chart from per-(entity_id, stage_id) frames.

    Mirrors :func:`hydro_aggregate_chart` but reads both sides from per-hydro frames
    instead of ``ResultComparison`` rows — the four hydro slacks (water-withdrawal
    pos/neg + evaporation pos/neg) plus the Cobre-only inflow non-negativity slack don't
    go through the comparison pipeline, so the chart machinery has to consume Cobre's
    ``cobre_hydro_means`` columns and the source model ``nw_hydro_slacks`` frame (or
    ``None`` for slacks without a source-model counterpart) directly.
    """
    cobre_by_stage = _hydro_per_stage_sum(cobre_hydro, variable, matched_ids)
    nw_by_stage = _hydro_per_stage_sum(nw_slacks, variable, matched_ids)
    if not cobre_by_stage and not nw_by_stage:
        return f"<p>No {variable} data available.</p>"

    stages = sorted(set(cobre_by_stage) | set(nw_by_stage))
    traces = _aggregate_percentile_traces(pct_df, variable, stages, matched_ids)
    if nw_by_stage:
        traces.append(
            {
                "x": stages,
                "y": [nw_by_stage.get(s, 0) for s in stages],
                "name": reference_label,
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NEWAVE, "width": 2},
            }
        )
    traces.append(
        {
            "x": stages,
            "y": [cobre_by_stage.get(s, 0) for s in stages],
            "name": "Cobre Mean",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_COBRE, "width": 2},
        }
    )

    layout = {
        "title": title,
        "xaxis": {"title": "Stage"},
        "yaxis": {"title": f"{title} ({unit})"},
    }
    return _plotly_div(traces, layout)


def hydro_slack_per_bus_chart(
    cobre_hydro: pl.DataFrame,
    nw_slacks: pl.DataFrame | None,
    variable: str,
    title: str,
    pct_df: pl.DataFrame | None,
    hydro_meta: dict[int, dict],
    bus_meta: dict[int, dict],
    matched_ids: set[int] | None = None,
    reference_label: str = "NEWAVE",
) -> str:
    """Per-bus faceted slack chart from per-(entity_id, stage_id) frames.

    Parallel to :func:`hydro_per_bus_chart` for the slack variables that aren't surfaced
    through ``ResultComparison`` (no Cobre/the source model comparison row exists for
    them).  Plants are bucketed by their owning bus via
    ``hydro_meta[cobre_id]["bus_ids"]``; fictitious buses (``NOFICT*``) are excluded,
    matching the existing per-bus charts.  When *nw_slacks* is ``None`` or lacks the
    column, the source model trace is omitted (used for
    ``inflow_nonnegativity_slack_m3s`` which has no source-model counterpart).
    """
    if cobre_hydro.is_empty() or variable not in cobre_hydro.columns:
        return f"<p>No {variable} data available.</p>"

    # Frame-sourced per-(bus, stage) sums (analyze owns the roll-up and the
    # bus-name resolution / NOFICT skip). The band's bus ids come from the
    # Cobre frame, matching the legacy ``per_bus_cb, per_bus_ids`` pairing.
    cb_agg = analyze.per_bus_sums_from_frame(
        cobre_hydro, variable, matched_ids, hydro_meta, bus_meta
    )
    nw_agg = analyze.per_bus_sums_from_frame(
        nw_slacks, variable, matched_ids, hydro_meta, bus_meta
    )
    per_bus_cb = {
        bus_name: cast("dict[int, float]", a["sum"]) for bus_name, a in cb_agg.items()
    }
    per_bus_ids = {
        bus_name: cast("set[int]", a["ids"]) for bus_name, a in cb_agg.items()
    }
    per_bus_nw = {
        bus_name: cast("dict[int, float]", a["sum"]) for bus_name, a in nw_agg.items()
    }
    if not per_bus_cb and not per_bus_nw:
        return f"<p>No {variable} data mapped to buses.</p>"

    ordered = [b for b in _REAL_SUBMARKET_ORDER if b in per_bus_cb or b in per_bus_nw]

    per_bus_pct = analyze.per_bus_band_from_pct(pct_df, variable, per_bus_ids)

    ncols = 2
    nrows = (len(ordered) + ncols - 1) // ncols
    panels = facet_grid(len(ordered), ncols=2)

    traces: list[dict] = []
    layout: dict = {"title": title}
    first = True

    for idx, bus_name in enumerate(ordered):
        cb_map = per_bus_cb.get(bus_name, {})
        nw_map = per_bus_nw.get(bus_name, {})
        stages = sorted(set(cb_map) | set(nw_map))
        if not stages:
            continue

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
            "title": bus_name,
            "anchor": xa,
        }

        bus_pct = per_bus_pct.get(bus_name, {})
        if bus_pct:
            p10 = [bus_pct.get(s, (0.0, 0.0))[0] for s in stages]
            p90 = [bus_pct.get(s, (0.0, 0.0))[1] for s in stages]
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

        if nw_map:
            traces.append(
                {
                    "x": stages,
                    "y": [nw_map.get(s, 0.0) for s in stages],
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
                "y": [cb_map.get(s, 0.0) for s in stages],
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
