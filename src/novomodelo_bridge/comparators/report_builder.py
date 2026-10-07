"""Assemble the full HTML comparison report from results data.

Combines chart implementations and the HTML template into a single HTML file
that loads plotly.js from a CDN.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cobre_bridge.comparators.charts import (
    _BALANCE_VARS,
    cobre_aggregate_chart,
    constraints_comparison_chart,
    convergence_chart,
    cost_breakdown_chart,
    cost_breakdown_table,
    fpha_detail_chart,
    fpha_metrics_table,
    future_cost_chart,
    hydro_aggregate_chart,
    hydro_per_bus_chart,
    hydro_slack_aggregate_chart,
    hydro_slack_per_bus_chart,
    immediate_cost_chart,
    line_summary_chart,
    other_costs_chart,
    overview_metrics,
    performance_fwd_bwd_split_chart,
    performance_iteration_chart,
    performance_metric_cards,
    productivity_blocks_table,
    productivity_comparison_scatter,
    productivity_per_stage_chart,
    ree_energy_chart,
    system_comparison_chart,
    system_per_bus_chart,
    thermal_cost_chart,
    thermal_generation_chart,
)
from cobre_bridge.comparators.charts._shared import (
    _BAND_FILL,
    _BAND_LINE,
    _REAL_SUBMARKET_ORDER,
    _build_interactive_detail_html,
    _enrich_with_percentiles,
    _plant_max_reldiff_table,
)
from cobre_bridge.comparators.constraints import per_stage_bounds
from cobre_bridge.comparators.html_report import (
    COLOR_COBRE,
    COLOR_NEWAVE,
    build_comparison_html,
    chart_grid,
    section_title,
    wrap_chart,
)
from cobre_bridge.comparators.model import (
    ResultComparison,
    ResultsSummary,
    ResultVariableStats,
)
from cobre_bridge.core.summary_counts import footer_counts
from cobre_bridge.ui.html.plotly import plotly_div as _plotly_div

if TYPE_CHECKING:
    from cobre_bridge.comparators.dataset import ComparisonDataset


def _results_summary_from_dataset(dataset: ComparisonDataset) -> ResultsSummary:
    """Reconstruct a :class:`ResultsSummary` from the canonical dataset.

    Rebuilds the summary object that ``overview_metrics`` consumes straight from
    the already-computed ``dataset.summary`` rows (one
    :class:`ResultVariableStats` per variable) and the ``footer_counts`` metadata
    (``total`` + ``by_entity_type``), so no per-row statistic is recomputed. The
    per-variable ``mean_rel_diff`` / ``max_rel_diff`` fields aren't carried in
    ``SUMMARY_SCHEMA`` (and aren't read by ``overview_metrics``); they keep their
    dataclass defaults of ``0.0``.
    """
    total, by_entity_type = footer_counts(dataset.metadata)

    by_variable: dict[str, ResultVariableStats] = {}
    for row in dataset.summary.to_dicts():
        correlation = row["correlation"]
        by_variable[str(row["variable"])] = ResultVariableStats(
            count=int(row["count"]),
            mean_abs_diff=float(row["mean_abs_diff"]),
            max_abs_diff=float(row["max_abs_diff"]),
            within_tol_rate=float(row["within_tol_rate"]),
            mean_smape=float(row["mean_smape"]),
            max_smape=float(row["max_smape"]),
            correlation=float(correlation) if correlation is not None else None,
        )

    return ResultsSummary(
        total=total,
        by_entity_type=by_entity_type,
        by_variable=by_variable,
    )


def build_energy_balance_tab(
    nw_market: pl.DataFrame,
    bus_agg: pl.DataFrame,
    bus_meta: dict[int, dict],
    nw_bus_names: dict[int, str],
    *,
    nw_net_load: pl.DataFrame | None = None,
    reference_label: str = "NEWAVE",
) -> str:
    """Build per-bus energy balance charts with p10/p90 bands.

    One 2x2 faceted chart per variable, with the source model mean + Cobre p10/p50/p90.
    """
    if bus_agg.is_empty() and nw_market.is_empty():
        return "<p>No energy balance data available.</p>"

    nw_offset = 0
    if not nw_market.is_empty():
        nw_offset = int(nw_market["stage"].min())

    # Merge the source model net load into nw_market if available.
    if nw_net_load is not None and not nw_net_load.is_empty():
        nw_market = pl.concat([nw_market, nw_net_load], how="diagonal_relaxed")
        if nw_offset == 0:
            nw_offset = int(nw_net_load["stage"].min())

    # Build Cobre bus_id → name and the source model code → bus_id lookups.
    cobre_name_to_id: dict[str, int] = {
        m["name"].strip().upper(): eid for eid, m in bus_meta.items()
    }
    nw_code_to_name: dict[int, str] = {
        code: name.strip().upper() for code, name in nw_bus_names.items()
    }

    # Match the source model bus codes to Cobre bus IDs by name.
    matched: dict[int, tuple[int, str]] = {}  # nw_code → (cobre_bus_id, name)
    for nw_code, nw_name in nw_code_to_name.items():
        cid = cobre_name_to_id.get(nw_name)
        if cid is not None:
            matched[nw_code] = (cid, nw_name)

    # Real submarkets only, fixed order for a clean 2x2 (fictitious/transhipment
    # buses — NEWAVE NOFICT*, DECOMP FC/IV — are excluded, never faceted).
    ordered_buses = []
    for bname in _REAL_SUBMARKET_ORDER:
        for nw_code, (cid, name) in matched.items():
            if name == bname:
                ordered_buses.append((nw_code, cid, name))
                break

    if not ordered_buses:
        return "<p>No matching buses found.</p>"

    # Pre-index the source model data: {(nw_code, var_upper): {stage_0based: value}}
    nw_lookup: dict[tuple[int, str], dict[int, float]] = {}
    for row in nw_market.iter_rows(named=True):
        if row["value"] is None:
            continue
        code = int(row["newave_code"])
        stage = int(row["stage"]) - nw_offset
        var = str(row["variable"]).strip().upper()
        nw_lookup.setdefault((code, var), {})[stage] = float(row["value"])

    # Pre-index Cobre percentile data: {bus_id: {stage: row_dict}}
    cobre_lookup: dict[int, dict[int, dict]] = {}
    for row in bus_agg.iter_rows(named=True):
        bid = int(row["bus_id"])
        sid = int(row["stage_id"])
        cobre_lookup.setdefault(bid, {})[sid] = row

    parts: list[str] = []

    for display_label, nw_var, cb_var, unit in _BALANCE_VARS:
        p10_col = f"{cb_var}_p10"
        p50_col = f"{cb_var}_p50"
        p90_col = f"{cb_var}_p90"

        # Check if Cobre has this variable.
        has_cobre = not bus_agg.is_empty() and p50_col in bus_agg.columns
        has_newave = bool(nw_var)

        if not has_cobre and not has_newave:
            continue

        parts.append(section_title(display_label))
        charts: list[str] = []

        ncols = 2
        nrows = (len(ordered_buses) + 1) // ncols
        traces: list[dict] = []
        layout: dict = {"title": f"{display_label} ({unit})"}
        first = True

        for idx, (nw_code, cid, bname) in enumerate(ordered_buses):
            row_i = idx // ncols
            col_i = idx % ncols
            ax_idx = idx + 1

            xa = f"x{ax_idx}" if ax_idx > 1 else "x"
            ya = f"y{ax_idx}" if ax_idx > 1 else "y"

            # Fixed-stride domains (0.52/0.47/0.44) — intentionally NOT
            # facet_grid: its gap formula yields different col_w/row_h and
            # would drift this chart's golden.
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
                "title": bname,
                "anchor": xa,
            }

            # Determine stage range from Cobre data.
            bus_pct = cobre_lookup.get(cid, {})
            nw_data = nw_lookup.get((nw_code, nw_var), {}) if nw_var else {}
            all_stages = sorted(set(bus_pct.keys()) | set(nw_data.keys()))
            if not all_stages:
                continue

            # Cobre P10-P90 band.
            if has_cobre and bus_pct:
                p10 = [
                    float(bus_pct.get(s, {}).get(p10_col, 0) or 0) for s in all_stages
                ]
                p90 = [
                    float(bus_pct.get(s, {}).get(p90_col, 0) or 0) for s in all_stages
                ]
                p50 = [
                    float(bus_pct.get(s, {}).get(p50_col, 0) or 0) for s in all_stages
                ]
                traces.append(
                    {
                        "x": all_stages + all_stages[::-1],
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
                        "x": all_stages,
                        "y": p50,
                        "name": "Cobre Median",
                        "type": "scatter",
                        "mode": "lines",
                        "line": {"color": COLOR_COBRE, "width": 2},
                        "xaxis": xa,
                        "yaxis": ya,
                        "legendgroup": "cb",
                        "showlegend": first,
                    }
                )

            # The source model mean line.
            if has_newave and nw_data:
                nw_y = [nw_data.get(s, 0) for s in all_stages]
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

            first = False

        if traces:
            charts.append(
                wrap_chart(_plotly_div(traces, layout, height=nrows * 300 + 80))
            )
            parts.append(chart_grid(charts, single=True))

    if not parts:
        return "<p>No energy balance data available.</p>"
    return "\n".join(parts)


_HYDRO_VARIABLES = [
    ("storage_final_hm3", "Storage (hm³)"),
    ("generation_mw", "Generation (MW)"),
    ("turbined_m3s", "Turbined (m³/s)"),
    ("productivity_mw_per_m3s", "Productivity = Gen / Turbined (MW per m³/s)"),
    ("spillage_m3s", "Spillage (m³/s)"),
    ("outflow_m3s", "Total Outflow (m³/s)"),
    ("inflow_m3s", "Incremental Inflow (m³/s)"),
    ("total_inflow_m3s", "Total Inflow / QAFLUH (m³/s)"),
    ("evaporation_m3s", "Evaporation (m³/s)"),
    ("withdrawal_m3s", "Water Withdrawal (m³/s)"),
    ("water_value_per_hm3", "Water Value (R$/hm³)"),
]

# Cobre-only per-plant variables (no per-plant equivalent in the source model).
#
# Withdrawal-slack ``pos``/``neg`` labels follow the source model's convention,
# the *inverse* of Cobre's column-name convention: Cobre's
# ``water_withdrawal_violation_pos_m3s`` is the physical equivalent of the source
# model's ``VIOL_NEG_VRETIRUH`` and vice versa. ``_NW_HYDRO_SLACK_VARS`` in
# ``results.py`` is swapped to match, so each panel pairs the right Cobre column
# with the right source-model series under a source-model-style label.
# Evaporation slacks share the source model's convention, so no swap is needed.
_HYDRO_COBRE_ONLY_VARIABLES = [
    ("stored_energy_initial_mwh", "Stored Energy Initial (MWh)"),
    ("stored_energy_final_mwh", "Stored Energy Final (MWh)"),
    ("incremental_inflow_energy_mw", "Natural Inflow Energy (MW)"),
    ("water_withdrawal_violation_neg_m3s", "Withdrawal Slack Pos (m³/s)"),
    ("water_withdrawal_violation_pos_m3s", "Withdrawal Slack Neg (m³/s)"),
    ("inflow_nonnegativity_slack_m3s", "Inflow Non-Negativity Slack (m³/s)"),
]

# Per-comparison-variable bound mapping: which static / per-stage bound
# columns to overlay as dashed reference lines.  Each entry is
# ``(static_meta_key, per_stage_bound_col)`` for the lower and upper
# bound respectively; either side may be ``None`` (e.g. "Outflow" has
# a min but typically no max in cobre).  The dashboard renders one
# dashed line per non-null bound; per-stage values shadow the static
# value when both are present at a given stage.
_HYDRO_BOUND_OVERLAY: dict[str, dict[str, tuple[str | None, str | None]]] = {
    "storage_final_hm3": {
        "min": ("min_storage_hm3", "min_storage_hm3"),
        "max": ("max_storage_hm3", "max_storage_hm3"),
    },
    "generation_mw": {
        "min": ("min_generation_mw", "min_generation_mw"),
        "max": ("max_generation_mw", None),
    },
    "turbined_m3s": {
        "min": ("min_turbined_m3s", "min_turbined_m3s"),
        "max": ("max_turbined_m3s", "max_turbined_m3s"),
    },
    "outflow_m3s": {
        "min": ("min_outflow_m3s", "min_outflow_m3s"),
        "max": ("max_outflow_m3s", None),
    },
}


def build_hydro_detail_tab(
    results: list[ResultComparison],
    pct_df: pl.DataFrame | None = None,
    cobre_hydro: pl.DataFrame | None = None,
    cobre_hydro_meta: dict[int, dict] | None = None,
    cobre_hydro_per_stage_bounds: pl.DataFrame | None = None,
    nw_hydro_slacks: pl.DataFrame | None = None,
    reference_label: str = "NEWAVE",
) -> str:
    """Build interactive per-plant hydro detail with JS dropdown.

    Comparison variables (the source model + Cobre) are populated from ``results``.
    Cobre-only variables (EARM, ENA, plus the three operational slacks:
    withdrawal pos/neg and inflow non-negativity) are populated from
    ``cobre_hydro`` if provided — these display only the Cobre line and
    band.

    When ``cobre_hydro_meta`` is supplied, static reservoir / outflow /
    turbined / generation bounds are surfaced as dashed reference lines
    on the matching variable charts.  When
    ``cobre_hydro_per_stage_bounds`` is also supplied, any per-stage
    overrides from ``constraints/hydro_bounds.parquet`` replace the
    static value at the affected stages — matching what the LP
    actually saw.

    When ``nw_hydro_slacks`` is supplied, the source model ``VIOL_POS_VRETIRUH`` /
    ``VIOL_NEG_VRETIRUH`` series (converted to m³/s) are rendered as a source-model line
    on the two withdrawal-slack panels alongside the existing Cobre Mean + p10/p90 band.
    The inflow-non-negativity slack stays Cobre-only because the source model has no
    direct counterpart.
    """
    hydro_data = [r for r in results if r.entity_type == "hydro"]
    if not hydro_data:
        return "<p>No hydro data available.</p>"

    plants: dict[tuple[str, int], dict[str, dict[int, tuple[float, float]]]] = {}
    cobre_ids: dict[tuple[str, int], int] = {}
    for r in hydro_data:
        key = (r.entity_name, r.newave_code)
        plants.setdefault(key, {}).setdefault(r.variable, {})[r.stage] = (
            r.newave_value,
            r.cobre_value,
        )
        cobre_ids[key] = r.cobre_id

    if not plants:
        return "<p>No hydro data available.</p>"

    # Build cobre_id -> {var: {stage: value}} for cobre-only variables.
    cobre_only_lookup: dict[int, dict[str, dict[int, float]]] = {}
    # Per-(cobre_id, stage_id) Cobre LP gen-max for the dashed overlay trace on the
    # generation_mw chart. The source model GHMAX_FPHC trace was found to be unhelpful
    # in practice and is intentionally not surfaced — see report notes.
    gen_max_cb_lookup: dict[int, dict[int, float]] = {}
    cobre_only_vars = [v for v, _ in _HYDRO_COBRE_ONLY_VARIABLES]
    if cobre_hydro is not None and not cobre_hydro.is_empty():
        avail_vars = [v for v in cobre_only_vars if v in cobre_hydro.columns]
        has_cb_lp_max = "cobre_lp_gen_max_mw" in cobre_hydro.columns
        for row in cobre_hydro.iter_rows(named=True):
            eid = int(row["entity_id"])
            sid = int(row["stage_id"])
            entry = cobre_only_lookup.setdefault(eid, {})
            for v in avail_vars:
                val = row.get(v)
                if val is None:
                    continue
                entry.setdefault(v, {})[sid] = float(val)
            if has_cb_lp_max:
                val_cb = row.get("cobre_lp_gen_max_mw")
                if val_cb is not None:
                    gen_max_cb_lookup.setdefault(eid, {})[sid] = float(val_cb)

    # cobre_id -> bound_col -> {stage: value} for the per-stage overrides
    # supplied by ``hydro_bounds.parquet``.  Falls back to the empty dict
    # when the parquet is absent.
    per_stage_bounds_lookup: dict[int, dict[str, dict[int, float]]] = {}
    if (
        cobre_hydro_per_stage_bounds is not None
        and not cobre_hydro_per_stage_bounds.is_empty()
    ):
        for row in cobre_hydro_per_stage_bounds.iter_rows(named=True):
            eid = int(row["entity_id"])
            sid = int(row["stage_id"])
            entry_map = per_stage_bounds_lookup.setdefault(eid, {})
            for col, val in row.items():
                if col in ("entity_id", "stage_id") or val is None:
                    continue
                entry_map.setdefault(col, {})[sid] = float(val)

    static_meta = cobre_hydro_meta or {}

    # cobre_id -> {var: {stage_id: nw_value}} for the two withdrawal slacks. Drives the
    # source model line on the matching cobre-only chart panels.
    nw_slack_lookup: dict[int, dict[str, dict[int, float]]] = {}
    _NW_SLACK_VARS = (
        "water_withdrawal_violation_pos_m3s",
        "water_withdrawal_violation_neg_m3s",
    )
    if nw_hydro_slacks is not None and not nw_hydro_slacks.is_empty():
        avail_nw_slacks = [v for v in _NW_SLACK_VARS if v in nw_hydro_slacks.columns]
        for row in nw_hydro_slacks.iter_rows(named=True):
            eid = int(row["entity_id"])
            sid = int(row["stage_id"])
            entry_map = nw_slack_lookup.setdefault(eid, {})
            for v in avail_nw_slacks:
                val = row.get(v)
                if val is None:
                    continue
                entry_map.setdefault(v, {})[sid] = float(val)

    all_vars = _HYDRO_VARIABLES + _HYDRO_COBRE_ONLY_VARIABLES

    js_plants: dict[str, dict] = {}
    for (name, nw_code), var_data in sorted(plants.items()):
        pid = f"{nw_code}_{name}"
        cid = cobre_ids.get((name, nw_code), -1)
        entry: dict = {
            "name": name,
            "code": nw_code,
            "cobre_id": cid,
        }
        for var_key, _var_label in _HYDRO_VARIABLES:
            stage_data = var_data.get(var_key, {})
            stages = sorted(stage_data.keys())
            entry[f"{var_key}_stages"] = stages
            entry[f"{var_key}_nw"] = [stage_data[s][0] for s in stages]
            entry[f"{var_key}_cb"] = [stage_data[s][1] for s in stages]
        cobre_only = cobre_only_lookup.get(cid, {})
        nw_slacks_for_plant = nw_slack_lookup.get(cid, {})
        for var_key, _var_label in _HYDRO_COBRE_ONLY_VARIABLES:
            stage_data_co = cobre_only.get(var_key, {})
            stages = sorted(stage_data_co.keys())
            entry[f"{var_key}_stages"] = stages
            nw_stage_map = nw_slacks_for_plant.get(var_key)
            if nw_stage_map:
                entry[f"{var_key}_nw"] = [
                    round(nw_stage_map.get(s, 0.0), 2) for s in stages
                ]
            else:
                entry[f"{var_key}_nw"] = []
            entry[f"{var_key}_cb"] = [round(stage_data_co[s], 2) for s in stages]
        # Cobre LP gen_max overlay (dashed trace). Aligned to the
        # generation_mw stage grid populated above.
        gen_stages = entry.get("generation_mw_stages", [])
        cb_max_map = gen_max_cb_lookup.get(cid, {})
        if cb_max_map:
            entry["generation_mw_max_cb"] = [
                round(cb_max_map.get(s, 0.0), 2) for s in gen_stages
            ]

        # Bound overlays per variable.  Static values come from
        # ``hydros.json`` via cobre_hydro_meta; per-stage rows in
        # hydro_bounds.parquet shadow the static value at the matching
        # stages.  When the bound is structurally absent (e.g.
        # max_outflow), we skip emitting the array so the JS layer
        # doesn't draw a constant-zero dashed line.
        meta = static_meta.get(cid, {})
        per_stage_overrides = per_stage_bounds_lookup.get(cid, {})
        for var_key, sides in _HYDRO_BOUND_OVERLAY.items():
            var_stages = entry.get(f"{var_key}_stages", [])
            if not var_stages:
                continue
            for side, (static_key, ps_key) in sides.items():
                static_val = meta.get(static_key) if static_key is not None else None
                ps_overrides = per_stage_overrides.get(ps_key, {}) if ps_key else {}
                if static_val is None and not ps_overrides:
                    continue
                series = []
                any_value = False
                for s in var_stages:
                    v = ps_overrides.get(s)
                    if v is None and static_val is not None:
                        v = static_val
                    if v is None:
                        series.append(None)
                    else:
                        series.append(round(float(v), 4))
                        any_value = True
                if any_value:
                    entry[f"{var_key}_bound_{side}"] = series

        js_plants[pid] = entry

    _enrich_with_percentiles(js_plants, all_vars, pct_df)

    summary_table = _plant_max_reldiff_table(
        results, "hydro", _HYDRO_VARIABLES, reference_label
    )
    detail_html = _build_interactive_detail_html(
        js_plants,
        all_vars,
        "hydro",
        "Hydro Plant",
        reference_label,
    )
    if summary_table:
        summary_table = f'<div style="margin-bottom:32px">{summary_table}</div>'
    return summary_table + detail_html


def build_thermal_detail_tab(
    results: list[ResultComparison],
    pct_df: pl.DataFrame | None = None,
    reference_label: str = "NEWAVE",
) -> str:
    """Build interactive per-plant thermal detail with JS dropdown."""
    thermal_data = [r for r in results if r.entity_type == "thermal"]
    if not thermal_data:
        return "<p>No thermal data available.</p>"

    plants: dict[tuple[str, int], dict[str, dict[int, tuple[float, float]]]] = {}
    cobre_ids: dict[tuple[str, int], int] = {}
    for r in thermal_data:
        key = (r.entity_name, r.newave_code)
        plants.setdefault(key, {}).setdefault(r.variable, {})[r.stage] = (
            r.newave_value,
            r.cobre_value,
        )
        cobre_ids[key] = r.cobre_id

    if not plants:
        return "<p>No thermal data available.</p>"

    thermal_vars = [("generation_mw", "Generation (MW)")]

    js_plants: dict[str, dict] = {}
    for (name, nw_code), var_data in sorted(plants.items()):
        pid = f"{nw_code}_{name}"
        entry: dict = {
            "name": name,
            "code": nw_code,
            "cobre_id": cobre_ids.get((name, nw_code), -1),
        }
        for var_key, _var_label in thermal_vars:
            stage_data = var_data.get(var_key, {})
            stages = sorted(stage_data.keys())
            entry[f"{var_key}_stages"] = stages
            entry[f"{var_key}_nw"] = [stage_data[s][0] for s in stages]
            entry[f"{var_key}_cb"] = [stage_data[s][1] for s in stages]
        js_plants[pid] = entry

    _enrich_with_percentiles(js_plants, thermal_vars, pct_df)

    summary_table = _plant_max_reldiff_table(
        results, "thermal", thermal_vars, reference_label
    )
    detail_html = _build_interactive_detail_html(
        js_plants,
        thermal_vars,
        "thermal",
        "Thermal Plant",
        reference_label,
    )
    if summary_table:
        summary_table = f'<div style="margin-bottom:32px">{summary_table}</div>'
    return summary_table + detail_html


def build_comparison_report(
    dataset: ComparisonDataset, reference_label: str = "NEWAVE"
) -> str:
    """Build a complete HTML comparison report.

    Every tab sources its non-tidy inputs from ``dataset.render`` (see
    :class:`~cobre_bridge.comparators.dataset.RenderInputs`): each tab reads
    its own named, already-typed fields, and the chart functions that still
    take ``list[ResultComparison]`` directly read ``dataset.render.results``.

    Parameters
    ----------
    dataset:
        The canonical comparison dataset. Its ``render`` carries every render
        input (named per-tab fields plus the ``results`` list).
    reference_label:
        Display name for the reference series (trace names, chart titles,
        prose). Defaults to ``"NEWAVE"``; ``compare decomp`` passes ``"DECOMP"``.

    Returns
    -------
    str
        Complete HTML document string.
    """
    results = dataset.render.results
    # Reconstruct the overview summary from the dataset's already-computed
    # summary rows + footer counts rather than recomputing it from the raw
    # ``results`` list — the chart functions below still consume ``results``
    # directly, but ``overview_metrics`` reads only the reconstructed summary.
    summary = _results_summary_from_dataset(dataset)

    tab_contents: dict[str, str] = {}

    # --- Overview tab ---
    overview_parts: list[str] = []
    nw_costs = dataset.render.nw_costs
    cobre_costs = dataset.render.cobre_costs
    overview_parts.append(
        overview_metrics(summary, nw_costs, cobre_costs, reference_label)
    )
    overview_parts.append(section_title("Cost Breakdown"))
    overview_parts.append(
        chart_grid(
            [
                wrap_chart(
                    cost_breakdown_chart(nw_costs, cobre_costs, reference_label)
                ),
                wrap_chart(
                    cost_breakdown_table(nw_costs, cobre_costs, reference_label)
                ),
            ],
        )
    )
    overview_parts.append(section_title("Per-Stage Cost"))
    nw_sin = dataset.render.nw_sin
    cobre_stage_costs = dataset.render.cobre_stage_costs
    nw_offset = dataset.render.nw_offset
    # Two side-by-side charts — immediate and future cost have very
    # different scales (one is per-stage operating cost, the other is a
    # cumulative future expectation), so we don't share an axis.
    overview_parts.append(
        chart_grid(
            [
                wrap_chart(
                    immediate_cost_chart(
                        nw_sin, cobre_stage_costs, nw_offset, reference_label
                    )
                ),
                wrap_chart(
                    future_cost_chart(
                        nw_sin, cobre_stage_costs, nw_offset, reference_label
                    )
                ),
            ],
        )
    )
    # Thermal-only (CTERM, live on both sides) and the non-thermal remainder (COPER −
    # CTERM) — the latter goes negative for the source model in the post-study because
    # COPER is frozen at the last study value while CTERM stays live.
    overview_parts.append(
        chart_grid(
            [
                wrap_chart(
                    thermal_cost_chart(
                        nw_sin, cobre_stage_costs, nw_offset, reference_label
                    )
                ),
                wrap_chart(
                    other_costs_chart(
                        nw_sin, cobre_stage_costs, nw_offset, reference_label
                    )
                ),
            ],
        )
    )

    overview_parts.append(section_title("Convergence"))
    nw_conv = dataset.render.nw_convergence
    cb_conv = dataset.render.cobre_convergence
    overview_parts.append(
        chart_grid(
            [wrap_chart(convergence_chart(nw_conv, cb_conv, reference_label))],
            single=True,
        )
    )
    tab_contents["tab-overview"] = "\n".join(overview_parts)

    # --- System tab ---
    bus_pct = dataset.render.bus
    system_parts: list[str] = []
    system_parts.append(section_title("Spot Price by Bus"))
    system_parts.append(
        chart_grid(
            [
                wrap_chart(
                    system_per_bus_chart(
                        results, "spot_price", "CMO by Bus", bus_pct, reference_label
                    )
                )
            ],
            single=True,
        )
    )
    system_parts.append(section_title("Deficit"))
    system_parts.append(
        chart_grid(
            [
                wrap_chart(
                    system_comparison_chart(
                        results, "deficit_mw", "Deficit", bus_pct, reference_label
                    )
                )
            ],
            single=True,
        )
    )
    tab_contents["tab-system"] = "\n".join(system_parts)

    # --- Energy Balance tab ---
    balance_html = build_energy_balance_tab(
        dataset.render.nw_market,
        dataset.render.bus_aggregates,
        dataset.render.cobre_bus_meta,
        dataset.render.nw_bus_names,
        nw_net_load=dataset.render.nw_net_load,
        reference_label=reference_label,
    )
    balance_cobre_hydro_means = dataset.render.cobre_hydro_means
    balance_hydro = dataset.render.hydro
    balance_nw_sin = dataset.render.nw_sin
    balance_nw_offset = dataset.render.nw_offset
    energy_balance_extra: list[str] = []
    if not balance_cobre_hydro_means.is_empty():
        energy_balance_extra.append(section_title("System Energy (EARM / ENA)"))
        energy_balance_extra.append(
            chart_grid(
                [
                    wrap_chart(
                        cobre_aggregate_chart(
                            balance_cobre_hydro_means,
                            "stored_energy_final_mwh",
                            "System Stored Energy (EARM)",
                            "MWh",
                            balance_hydro,
                            nw_sin=balance_nw_sin,
                            nw_variable="EARMF",
                            nw_factor=730.0,
                            nw_offset=balance_nw_offset,
                            reference_label=reference_label,
                        )
                    ),
                    wrap_chart(
                        cobre_aggregate_chart(
                            balance_cobre_hydro_means,
                            "incremental_inflow_energy_mw",
                            "System Natural Inflow Energy (ENA)",
                            "MW",
                            balance_hydro,
                            nw_sin=balance_nw_sin,
                            nw_variable="ENA",
                            nw_factor=1.0,
                            nw_offset=balance_nw_offset,
                            reference_label=reference_label,
                        )
                    ),
                ]
            )
        )
    # --- REE energy rollup (additive; absent for `compare
    # newave` datasets, which never carry `entity_type == "ree"` rows) ---
    ree_results = [r for r in results if r.entity_type == "ree"]
    if ree_results:
        energy_balance_extra.append(section_title("REE Energy (ENA / EARM)"))
        energy_balance_extra.append(
            chart_grid(
                [
                    wrap_chart(
                        ree_energy_chart(
                            results,
                            "ena_mwmes",
                            "REE Natural Inflow Energy (ENA)",
                            reference_label,
                        )
                    ),
                    wrap_chart(
                        ree_energy_chart(
                            results,
                            "earm_final_mwmes",
                            "REE Stored Energy (EARM)",
                            reference_label,
                        )
                    ),
                ]
            )
        )
    tab_contents["tab-balance"] = balance_html + "\n" + "\n".join(energy_balance_extra)

    # --- Network tab ---
    line_pct = dataset.render.line
    line_bounds = dataset.render.line_bounds
    line_meta = dataset.render.line_meta
    network_parts: list[str] = []
    network_parts.append(section_title("Line Net Flow"))
    network_parts.append(
        chart_grid(
            [
                wrap_chart(
                    line_summary_chart(
                        results, line_pct, line_bounds, line_meta, reference_label
                    )
                )
            ],
            single=True,
        )
    )
    tab_contents["tab-network"] = "\n".join(network_parts)

    # --- Constraints tab --- Per-constraint LHS comparison: The source-model-side LHS
    # evaluated against MEDIAS-USIH / int*.out output, Cobre-side LHS as the mean across
    # scenarios and blocks from the simulation parquet. Bounds are taken from
    # constraints/generic_constraint_bounds.parquet's F3 sense-free `bound_lower`/
    # `bound_upper` endpoints via `per_stage_bounds` (block 0 preferred when blocks
    # disagree; the resolved `ResolvedBound.shape` — not a removed `sense` field —
    # drives the chart's direction label, see `constraints_comparison_chart`).
    gc_constraints = dataset.render.gc_constraints
    gc_bounds_df = dataset.render.gc_bounds
    gc_lhs_nw = dataset.render.gc_lhs_newave
    gc_lhs_cb = dataset.render.gc_lhs_cobre
    gc_max_stage = dataset.render.nw_max_stage
    bound_lookup = per_stage_bounds(gc_bounds_df, max_stage=gc_max_stage)
    if gc_max_stage is not None:
        gc_lhs_nw = (
            gc_lhs_nw.filter(pl.col("stage_id") <= gc_max_stage)
            if not gc_lhs_nw.is_empty()
            else gc_lhs_nw
        )
        gc_lhs_cb = (
            gc_lhs_cb.filter(pl.col("stage_id") <= gc_max_stage)
            if not gc_lhs_cb.is_empty()
            else gc_lhs_cb
        )
    constraints_parts: list[str] = []
    constraints_parts.append(section_title("Generic Constraints — LHS vs Bound"))
    constraints_parts.append(
        chart_grid(
            [
                wrap_chart(
                    constraints_comparison_chart(
                        gc_constraints,
                        gc_lhs_nw,
                        gc_lhs_cb,
                        bound_lookup,
                        reference_label,
                    )
                )
            ],
            single=True,
        )
    )
    tab_contents["tab-constraints"] = "\n".join(constraints_parts)

    # --- Hydro Operation tab ---
    hydro_pct = dataset.render.hydro
    cobre_hydro_means = dataset.render.cobre_hydro_means
    nw_sin = dataset.render.nw_sin
    nw_offset = dataset.render.nw_offset
    matched_hydro_ids = {r.cobre_id for r in results if r.entity_type == "hydro"}

    hydro_meta = dataset.render.cobre_hydro_meta
    bus_meta = dataset.render.cobre_bus_meta
    hydro_parts: list[str] = []
    for var, title in [
        ("storage_final_hm3", "Storage by Bus (hm³)"),
        ("generation_mw", "Hydro Generation by Bus (MW)"),
        ("spillage_m3s", "Spillage by Bus (m³/s)"),
        ("turbined_m3s", "Turbined by Bus (m³/s)"),
        ("inflow_m3s", "Inflow by Bus (m³/s)"),
        ("water_value_per_hm3", "Water Value by Bus (R$/hm³)"),
    ]:
        hydro_parts.append(section_title(title))
        hydro_parts.append(
            chart_grid(
                [
                    wrap_chart(
                        hydro_per_bus_chart(
                            results,
                            var,
                            title,
                            hydro_pct,
                            hydro_meta,
                            bus_meta,
                            reference_label,
                        )
                    )
                ],
                single=True,
            )
        )

    # System-level EARM and ENA (Cobre per-hydro aggregate vs the source model SIN). The
    # source model EARMF is in MWmes (mean MW over a month); convert to MWh via the
    # canonical 730 h/month factor used by the source model.  ENA is already in MW (mean
    # power) on both sides.
    hydro_parts.append(section_title("Aggregate Energy Variables"))
    energy_charts = [
        wrap_chart(
            cobre_aggregate_chart(
                cobre_hydro_means,
                "stored_energy_final_mwh",
                "Stored Energy (EARM)",
                "MWh",
                hydro_pct,
                nw_sin=nw_sin,
                nw_variable="EARMF",
                nw_factor=730.0,
                nw_offset=nw_offset,
                matched_ids=matched_hydro_ids or None,
                reference_label=reference_label,
            )
        ),
        wrap_chart(
            cobre_aggregate_chart(
                cobre_hydro_means,
                "incremental_inflow_energy_mw",
                "Natural Inflow Energy (ENA)",
                "MW",
                hydro_pct,
                nw_sin=nw_sin,
                nw_variable="ENA",
                nw_factor=1.0,
                nw_offset=nw_offset,
                matched_ids=matched_hydro_ids or None,
                reference_label=reference_label,
            )
        ),
    ]
    hydro_parts.append(chart_grid(energy_charts))

    # System-aggregate (SIN) totals for each hydro variable. Sums Cobre plant values per
    # stage and overlays the source model total. Mirrors the per-bus facet section but
    # collapses across buses — useful as a one-glance global view alongside the per-bus
    # disaggregation.
    hydro_parts.append(section_title("System Totals (SIN)"))
    aggregate_charts: list[str] = []
    for var, title in [
        ("storage_final_hm3", "Total Storage (hm³)"),
        ("generation_mw", "Hydro Generation (MW)"),
        ("spillage_m3s", "Total Spillage (m³/s)"),
        ("turbined_m3s", "Total Turbined (m³/s)"),
        ("inflow_m3s", "Total Inflow (m³/s)"),
        ("water_value_per_hm3", "Water Value (R$/hm³)"),
    ]:
        aggregate_charts.append(
            wrap_chart(
                hydro_aggregate_chart(results, var, title, hydro_pct, reference_label)
            )
        )
    hydro_parts.append(chart_grid(aggregate_charts))

    # Slack variables: same per-bus + SIN-total treatment as the operational
    # variables above, but driven by the per-(entity_id, stage_id) Cobre frame and the
    # source model slack frame (no ResultComparison rows exist for slacks).  The inflow
    # non-negativity slack has no source-model counterpart, so its the source model
    # source is passed as None — the chart still renders the Cobre Mean + p10/p90 band,
    # just without an overlaid the source model line.
    nw_hydro_slacks = dataset.render.nw_hydro_slacks
    # Withdrawal pos/neg are SWAPPED to follow the source model's sign convention; the
    # ``_NW_HYDRO_SLACK_VARS`` mapping in ``results.py`` is correspondingly swapped so
    # each panel pairs the right Cobre column with the right The source model series.
    # Evaporation pos/neg already share the source model's convention.
    slack_specs: list[tuple[str, str, bool]] = [
        ("water_withdrawal_violation_neg_m3s", "Withdrawal Slack Pos (m³/s)", True),
        ("water_withdrawal_violation_pos_m3s", "Withdrawal Slack Neg (m³/s)", True),
        ("evaporation_violation_pos_m3s", "Evaporation Slack Pos (m³/s)", True),
        ("evaporation_violation_neg_m3s", "Evaporation Slack Neg (m³/s)", True),
        ("inflow_nonnegativity_slack_m3s", "Inflow Non-Negativity Slack (m³/s)", False),
    ]
    for var, slack_title, has_newave in slack_specs:
        hydro_parts.append(section_title(slack_title + " by Bus"))
        hydro_parts.append(
            chart_grid(
                [
                    wrap_chart(
                        hydro_slack_per_bus_chart(
                            cobre_hydro_means,
                            nw_hydro_slacks if has_newave else None,
                            var,
                            slack_title + " by Bus",
                            hydro_pct,
                            hydro_meta,
                            bus_meta,
                            matched_ids=matched_hydro_ids or None,
                            reference_label=reference_label,
                        )
                    )
                ],
                single=True,
            )
        )

    hydro_parts.append(section_title("Hydro Slacks (SIN)"))
    slack_sin_charts = [
        wrap_chart(
            hydro_slack_aggregate_chart(
                cobre_hydro_means,
                nw_hydro_slacks if has_newave else None,
                var,
                slack_title,
                hydro_pct,
                matched_ids=matched_hydro_ids or None,
                reference_label=reference_label,
            )
        )
        for var, slack_title, has_newave in slack_specs
    ]
    hydro_parts.append(chart_grid(slack_sin_charts))

    tab_contents["tab-hydro"] = "\n".join(hydro_parts)

    # --- Hydro Plant Details tab ---
    tab_contents["tab-hydro-detail"] = build_hydro_detail_tab(
        results,
        hydro_pct,
        cobre_hydro_means,
        cobre_hydro_meta=hydro_meta,
        cobre_hydro_per_stage_bounds=dataset.render.cobre_hydro_per_stage_bounds,
        nw_hydro_slacks=nw_hydro_slacks,
        reference_label=reference_label,
    )

    # --- Thermal Operation tab ---
    thermal_pct = dataset.render.thermal
    thermal_parts: list[str] = []
    thermal_parts.append(section_title("Thermal Generation Comparison"))
    thermal_parts.append(
        chart_grid(
            [
                wrap_chart(
                    thermal_generation_chart(results, thermal_pct, reference_label)
                )
            ],
            single=True,
        )
    )
    tab_contents["tab-thermal"] = "\n".join(thermal_parts)

    # --- Thermal Plant Details tab ---
    tab_contents["tab-thermal-detail"] = build_thermal_detail_tab(
        results, thermal_pct, reference_label
    )

    # --- Productivity tab ---
    prod_df = dataset.render.productivity_detail
    per_stage_df = dataset.render.productivity_per_stage
    prod_parts: list[str] = []
    static_title = (
        "Static productivity — pmo vs cobre-bridge conversion "
        "(point / equivalent / accumulated)"
    )
    # the static (pmo-derived) and realized (per-stage) halves are
    # disjoint data sources -- a source with no pmo.dat (e.g. DECOMP) can
    # carry a non-empty ``per_stage_df`` while ``prod_df`` stays empty, and
    # vice versa -- so each half is gated on its OWN frame, independently.
    # Order is preserved exactly for the case both are non-empty (e.g.
    # NEWAVE, which always has both): static title -> scatter/no-data-note ->
    # realized title/description/chart -> building-blocks table.
    prod_parts.append(section_title(static_title))
    if prod_df.is_empty():
        prod_parts.append("<p>No productivity data available.</p>")
    else:
        prod_parts.append(
            chart_grid(
                [
                    wrap_chart(
                        productivity_comparison_scatter(
                            prod_df,
                            "point",
                            "Point — pmo altura_65 vs compute_productivity",
                            reference_label,
                        )
                    ),
                    wrap_chart(
                        productivity_comparison_scatter(
                            prod_df,
                            "equivalent",
                            "Equivalent — pmo vs stored_energy_productivity",
                            reference_label,
                        )
                    ),
                    wrap_chart(
                        productivity_comparison_scatter(
                            prod_df,
                            "accumulated",
                            "Accumulated — pmo vs cobre-bridge cascade",
                            reference_label,
                        )
                    ),
                ]
            )
        )

    if not per_stage_df.is_empty():
        prod_parts.append(section_title("Realized productivity across stages"))
        prod_parts.append(
            '<p style="color:#64748B;margin:-8px 0 12px">Productivity is constant'
            " within a stage but varies across stages, tracking the reservoir"
            f" head reached each stage — pick a reservoir to compare {reference_label}"
            " vs Cobre.</p>"
        )
        # Reuses the shared per-plant dropdown widget (same as the hydro/thermal
        # detail tabs), so every reservoir is selectable — not a fixed subset.
        prod_parts.append(productivity_per_stage_chart(per_stage_df, reference_label))

    if not prod_df.is_empty():
        prod_parts.append(section_title("Productivity Building Blocks"))
        prod_parts.append(
            chart_grid(
                [wrap_chart(productivity_blocks_table(prod_df, reference_label))],
                single=True,
            )
        )

    # --- Fitted production functions (FPHA) --- Present only when both sides
    # fitted FPHA hyperplanes; compares the production surfaces GH(V, Q) the two
    # solvers fit, on a shared grid (run-of-river plants reduce to a Q-curve).
    fpha_metrics = dataset.render.fpha_metrics
    if not fpha_metrics.is_empty():
        fpha_surface = dataset.render.fpha_surface
        fpha_spill = dataset.render.fpha_spill
        prod_parts.append(section_title("Fitted production functions (FPHA)"))
        prod_parts.append(
            '<p style="color:#64748B;margin:-8px 0 12px">Both solvers fit the'
            " hydro production surface GH(V, Q, S) as a set of hyperplanes; this"
            " compares the resulting surfaces at the fitting-grid nodes. Use the"
            f" {reference_label} / Cobre / Both / Difference buttons to isolate each"
            f" surface or their difference (Cobre − {reference_label}, MW); they"
            " nearly coincide at S = 0. Spillage (S) is shown separately at the max"
            " V/Q corner. Pick a plant and stage.</p>"
        )
        prod_parts.append(fpha_detail_chart(fpha_surface, fpha_spill, reference_label))
        prod_parts.append(section_title("FPHA surface fidelity by plant"))
        prod_parts.append(
            chart_grid(
                [wrap_chart(fpha_metrics_table(fpha_metrics, reference_label))],
                single=True,
            )
        )

    tab_contents["tab-productivity"] = "\n".join(prod_parts)

    # --- Performance tab ---
    nw_tim_iters = dataset.render.nw_tim_iterations
    nw_tim_stages = dataset.render.nw_tim_stages
    cb_train_secs = dataset.render.cobre_training_seconds
    cb_conv_perf = dataset.render.cobre_iteration_timing
    perf_parts: list[str] = []
    perf_parts.append(
        performance_metric_cards(nw_tim_stages, cb_train_secs, reference_label)
    )
    perf_parts.append(section_title("Time per Iteration"))
    perf_parts.append(
        chart_grid(
            [
                wrap_chart(
                    performance_iteration_chart(
                        nw_tim_iters, cb_conv_perf, reference_label
                    )
                )
            ],
            single=True,
        )
    )
    perf_parts.append(section_title("Forward / Backward Split"))
    perf_parts.append(
        chart_grid(
            [
                wrap_chart(
                    performance_fwd_bwd_split_chart(
                        nw_tim_iters, cb_conv_perf, reference_label
                    )
                )
            ],
            single=True,
        )
    )
    tab_contents["tab-performance"] = "\n".join(perf_parts)

    return build_comparison_html(
        title=f"Cobre vs {reference_label} Results Comparison",
        tab_contents=tab_contents,
    )
