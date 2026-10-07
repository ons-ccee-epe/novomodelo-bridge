"""v2 Plant Explorer tab — hydro and thermal sub-tabs.

Provides a master-detail split-pane layout for both hydro and thermal plants.
Left pane: searchable/sortable table.  Right pane: detail charts rendered
client-side via Plotly.react() using precomputed per-plant p10/p50/p90
percentile bands.  Sub-tab switching is handled by SUB_TAB_JS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from cobre_bridge.ui.html.document import (
    chart_grid,
    collapsible_section,
    escape_attr,
    escape_text,
    json_for_script,
    plant_explorer_table,
    section_title,
    sparkline_svg,
    wrap_chart,
)
from cobre_bridge.ui.html.js import PLANT_EXPLORER_JS, SUB_TAB_JS
from cobre_bridge.ui.html.plotly import stage_x_dates

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-plants"
TAB_LABEL = "Plant Explorer"
TAB_ORDER = 50
REQUIRED_JS: list[str] = [SUB_TAB_JS, PLANT_EXPLORER_JS]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Flow variables: block-hours weighted average
_FLOW_VARS: list[str] = [
    "generation_mw",
    "turbined_m3s",
    "spillage_m3s",
    "evaporation_m3s",
    # Operational slack columns surfaced as cobre-only series so the user can see at a
    # glance which constraints the LP had to relax.  These have no source-model
    # counterpart — they're plotted with the standard p10/p50/p90 band like any other
    # flow variable.
    "water_withdrawal_violation_pos_m3s",
    "water_withdrawal_violation_neg_m3s",
    "inflow_nonnegativity_slack_m3s",
]

# Stage-level variables: filter block_id == 0
_STAGE_VARS: list[str] = [
    "storage_final_hm3",
    "inflow_m3s",
    "water_value_per_hm3",
    # cobre HEAD energy columns: stage-level at block_id=0
    "stored_energy_final_mwh",
    "incremental_inflow_energy_mw",
    "equivalent_productivity_mw_per_m3s",
]


def _block_hours_dict(bh_df: pl.DataFrame) -> dict[tuple[int, int], float]:
    """Materialize the ``(stage_id, block_id) -> hours`` map from ``bh_df``.

    ``bh_df`` (columns ``stage_id``, ``block_id``, ``_bh``) is the same
    block-hours table the percentile aggregation joins on; this reuses it as the
    weighting basis for :func:`_weighted_lp_generation_bounds` so the bound and
    the generation series are aggregated over an identical block layout.
    """
    return {
        (int(r["stage_id"]), int(r["block_id"])): float(r["_bh"])
        for r in bh_df.iter_rows(named=True)
    }


def _weighted_lp_generation_bounds(
    lp_bounds: pd.DataFrame,
    entity_type_code: int,
    block_hours: dict[tuple[int, int], float],
) -> dict[int, dict[int, dict[int, float]]]:
    """``entity_id -> stage_id -> {bound_type_code -> block-hours-weighted bound}``.

    cobre's LP-bounds dump (``output/training/dictionaries/bounds.parquet``)
    records a generation bound per ``(entity, stage, bound_type_code)`` either
    as a single stage-level row (``block_id`` NULL, in force for every block)
    **or** as per-block override rows (``block_id`` 0..n-1) that replace the
    stage-level value for their own block. The simulated generation series the
    detail charts plot against these bounds is a block-hours weighted mean over
    the stage's blocks (see :func:`_compute_hydro_percentiles` /
    :func:`_compute_thermal_percentiles`), so the bound has to be aggregated the
    same way: the effective per-block bound (the override where present, else
    the stage-level value) weighted by block hours. Collapsing the per-block
    rows by ``bound_type_code`` alone — ignoring ``block_id`` — lets one block's
    value stand in for the whole stage and yields a bound the plotted
    generation appears to exceed (e.g. a must-run thermal fixed at 568/544/516
    MW across the heavy/medium/light blocks reads as a flat 516 MW cap while its
    534.6 MW hours-weighted mean sits above it).
    """
    result: dict[int, dict[int, dict[int, float]]] = {}
    if lp_bounds.empty:
        return result

    stage_blocks: dict[int, list[tuple[int, float]]] = {}
    for (sid, bid), hours in block_hours.items():
        stage_blocks.setdefault(sid, []).append((bid, hours))

    # entity_id -> stage_id -> btc -> (stage_level_value, {block_id: value})
    raw: dict[int, dict[int, dict[int, tuple[float | None, dict[int, float]]]]] = {}
    sub = lp_bounds[lp_bounds["entity_type_code"] == entity_type_code]
    # A frame without a block_id column (or with a NULL block_id) carries only
    # stage-level rows — every value is a base with no per-block override.
    has_block = "block_id" in sub.columns
    for row in sub.itertuples(index=False):
        eid = int(row.entity_id)
        sid = int(row.stage_id)
        btc = int(row.bound_type_code)
        btcs = raw.setdefault(eid, {}).setdefault(sid, {})
        base, overrides = btcs.get(btc, (None, {}))
        block_id = row.block_id if has_block else None
        if block_id is None or pd.isna(block_id):
            base = float(row.bound_value)
        else:
            overrides[int(block_id)] = float(row.bound_value)
        btcs[btc] = (base, overrides)

    for eid, stages in raw.items():
        for sid, btcs in stages.items():
            blocks = stage_blocks.get(sid, [])
            for btc, (base, overrides) in btcs.items():
                if not overrides:
                    # Only a stage-level row: the weighted mean is that value.
                    weighted = base
                elif blocks:
                    num = den = 0.0
                    for bid, hours in blocks:
                        eff = overrides.get(bid, base)
                        if eff is None:
                            continue
                        num += eff * hours
                        den += hours
                    weighted = num / den if den else base
                else:
                    # No block-hours metadata for the stage: fall back to an
                    # unweighted mean of the overrides rather than dropping them.
                    vals = list(overrides.values())
                    weighted = sum(vals) / len(vals)
                if weighted is not None:
                    result.setdefault(eid, {}).setdefault(sid, {})[btc] = weighted
    return result


def _compute_hydro_percentiles(
    hydros_lf: pl.LazyFrame,
    bh_df: pl.DataFrame,
) -> pl.DataFrame:
    """Compute per-plant p10/p50/p90 percentiles for all 7 variables.

    Flow variables (generation_mw, turbined_m3s, spillage_m3s, evaporation_m3s)
    are aggregated via block-hours weighted average before computing percentiles
    across scenarios.

    Stage-level variables (storage_final_hm3, inflow_m3s, water_value_per_hm3)
    are filtered to block_id == 0 before computing percentiles.

    Returns a single DataFrame with columns:
        hydro_id, stage_id, <var>_p10, <var>_p50, <var>_p90 for each var.
    """
    schema = hydros_lf.collect_schema()

    # --- Flow variables -------------------------------------------------------
    available_flow = [v for v in _FLOW_VARS if v in schema]
    flow_pcts: pl.DataFrame | None = None
    if available_flow:
        flow_pcts = (
            hydros_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
            .group_by(["scenario_id", "stage_id", "hydro_id"])
            .agg(
                [
                    (pl.col(m) * pl.col("_bh")).sum() / pl.col("_bh").sum()
                    for m in available_flow
                ]
            )
            .group_by(["stage_id", "hydro_id"])
            .agg(
                [
                    expr
                    for m in available_flow
                    for expr in [
                        pl.col(m)
                        .quantile(0.1, interpolation="linear")
                        .alias(f"{m}_p10"),
                        pl.col(m)
                        .quantile(0.5, interpolation="linear")
                        .alias(f"{m}_p50"),
                        pl.col(m)
                        .quantile(0.9, interpolation="linear")
                        .alias(f"{m}_p90"),
                    ]
                ]
            )
            .sort(["hydro_id", "stage_id"])
            .collect(engine="streaming")
        )

    # --- Stage-level variables ------------------------------------------------
    available_stage = [v for v in _STAGE_VARS if v in schema]
    stage_pcts: pl.DataFrame | None = None
    if available_stage:
        stage_pcts = (
            hydros_lf.filter(pl.col("block_id") == 0)
            .group_by(["scenario_id", "stage_id", "hydro_id"])
            .agg([pl.col(m).mean() for m in available_stage])
            .group_by(["stage_id", "hydro_id"])
            .agg(
                [
                    expr
                    for m in available_stage
                    for expr in [
                        pl.col(m)
                        .quantile(0.1, interpolation="linear")
                        .alias(f"{m}_p10"),
                        pl.col(m)
                        .quantile(0.5, interpolation="linear")
                        .alias(f"{m}_p50"),
                        pl.col(m)
                        .quantile(0.9, interpolation="linear")
                        .alias(f"{m}_p90"),
                    ]
                ]
            )
            .sort(["hydro_id", "stage_id"])
            .collect(engine="streaming")
        )

    # --- Merge ---------------------------------------------------------------
    if flow_pcts is not None and stage_pcts is not None:
        return flow_pcts.join(
            stage_pcts, on=["stage_id", "hydro_id"], how="full", coalesce=True
        )
    if flow_pcts is not None:
        return flow_pcts
    if stage_pcts is not None:
        return stage_pcts
    # Neither set of variables is available — return empty
    return pl.DataFrame(
        {
            "hydro_id": pl.Series([], dtype=pl.Int32),
            "stage_id": pl.Series([], dtype=pl.Int32),
        }
    )


def _build_hydro_json(
    hydro_meta: dict[int, dict],
    percentiles: pl.DataFrame,
    stages: list[int],
    stage_dates: dict[int, str],
    bus_names: dict[int, str],
    hydro_bounds: pd.DataFrame,
    lp_bounds: pd.DataFrame,
    block_hours: dict[tuple[int, int], float],
) -> tuple[dict[str, dict], list[str]]:
    """Build per-plant data dict for JS embedding.

    Returns ``(hydro_data_dict, xdates_list)`` where ``hydro_data_dict``
    maps string plant-id keys to dicts containing percentile arrays and
    optional bounds arrays.
    """
    xdates = stage_x_dates(stages, stage_dates)
    hydro_data: dict[str, dict] = {}

    # Pre-index percentiles by (hydro_id, stage_id)
    pct_index: dict[tuple[int, int], dict] = {}
    for row in percentiles.iter_rows(named=True):
        pct_index[(row["hydro_id"], row["stage_id"])] = row

    # Pre-process hydro_bounds once: {hydro_id: {stage_id: row}}
    hb_index: dict[int, dict[int, dict]] = {}
    if not hydro_bounds.empty:
        for _, row in hydro_bounds.iterrows():
            hid = int(row["hydro_id"])
            sid = int(row["stage_id"])
            hb_index.setdefault(hid, {})[sid] = row.to_dict()

    # Pre-process lp_bounds for hydro generation (entity_type_code == 0):
    # hid -> {stage_id -> {bt_code -> block-hours-weighted bound}}. Weighting
    # keeps the bound comparable to the block-hours weighted generation series
    # (see _weighted_lp_generation_bounds).
    lp_index = _weighted_lp_generation_bounds(lp_bounds, 0, block_hours)

    all_vars = _FLOW_VARS + _STAGE_VARS

    for hid, meta in sorted(hydro_meta.items()):
        name: str = meta.get("name", str(hid))
        bus_id: int = meta.get("bus_id", -1)
        bus: str = bus_names.get(bus_id, str(bus_id))
        max_gen_mw: float = float(meta.get("max_gen_mw", 0) or 0)
        vol_max: float = float(meta.get("vol_max", 0) or 0)

        entry: dict = {
            "name": name,
            "bus": bus,
            "max_gen_mw": round(max_gen_mw, 1),
            "vol_max": round(vol_max, 1),
        }

        # Percentile arrays for each variable
        for var in all_vars:
            short = _var_short(var)
            for sfx in ["p10", "p50", "p90"]:
                col = f"{var}_{sfx}"
                entry[f"{short}_{sfx}"] = [
                    round(float(pct_index.get((hid, s), {}).get(col, 0) or 0), 4)
                    for s in stages
                ]

        # Derived outflow = turbined + spillage (sum of p50 arrays for display)
        for sfx in ("p50", "p10", "p90"):
            turb = entry.get(f"turb_{sfx}", [])
            spill = entry.get(f"spill_{sfx}", [])
            entry[f"outflow_{sfx}"] = [
                round(float(t + sp), 4) for t, sp in zip(turb, spill)
            ]

        # hydro_bounds: storage and turbined/spillage/outflow min/max
        if hid in hb_index:
            hb_stages = hb_index[hid]
            bound_cols = [
                ("min_storage_hm3", "stor_min"),
                ("max_storage_hm3", "stor_max"),
                ("min_turbined_m3s", "turb_min"),
                ("max_turbined_m3s", "turb_max"),
                ("min_spillage_m3s", "spill_min"),
                ("max_spillage_m3s", "spill_max"),
                ("min_outflow_m3s", "outflow_min"),
                ("max_outflow_m3s", "outflow_max"),
            ]
            for col, key in bound_cols:
                entry[key] = [
                    round(float(hb_stages.get(s, {}).get(col, 0) or 0), 4)
                    for s in stages
                ]
        else:
            for key in [
                "stor_min",
                "stor_max",
                "turb_min",
                "turb_max",
                "spill_min",
                "spill_max",
                "outflow_min",
                "outflow_max",
            ]:
                entry[key] = []

        # lp_bounds: generation min/max (bound_type_code 6=min, 7=max)
        if hid in lp_index:
            lp_stages = lp_index[hid]
            entry["gen_min"] = [
                round(lp_stages.get(s, {}).get(6, 0.0), 4) for s in stages
            ]
            entry["gen_max"] = [
                round(lp_stages.get(s, {}).get(7, 0.0), 4) for s in stages
            ]
        else:
            entry["gen_min"] = []
            entry["gen_max"] = []

        hydro_data[str(hid)] = entry

    return hydro_data, xdates


def _var_short(var: str) -> str:
    """Map a column name to its short JSON key prefix."""
    _MAP: dict[str, str] = {
        "generation_mw": "gen",
        "turbined_m3s": "turb",
        "spillage_m3s": "spill",
        "evaporation_m3s": "evap",
        "storage_final_hm3": "stor",
        "inflow_m3s": "inflow",
        "water_value_per_hm3": "wv",
        "stored_energy_final_mwh": "earm",
        "incremental_inflow_energy_mw": "ena",
        "equivalent_productivity_mw_per_m3s": "rhoeq",
        "water_withdrawal_violation_pos_m3s": "wpos",
        "water_withdrawal_violation_neg_m3s": "wneg",
        "inflow_nonnegativity_slack_m3s": "innn",
    }
    return _MAP.get(var, var)


# ---------------------------------------------------------------------------
# Thermal helpers
# ---------------------------------------------------------------------------

_THERMAL_FLOW_VARS: list[str] = [
    "generation_mw",
    "generation_cost",
    "generation_mwh",
]

_THERMAL_VAR_SHORT: dict[str, str] = {
    "generation_mw": "gen",
    "generation_cost": "cost",
    "generation_mwh": "energy",
}


def _compute_thermal_percentiles(
    thermals_lf: pl.LazyFrame,
    bh_df: pl.DataFrame,
) -> pl.DataFrame:
    """Compute per-plant p10/p50/p90 percentiles for all 3 thermal variables.

    All thermal metrics are flow variables (generation_mw, generation_cost,
    generation_mwh) — aggregate via block-hours weighted average before
    computing percentiles across scenarios.

    Returns a DataFrame with columns:
        thermal_id, stage_id, <var>_p10, <var>_p50, <var>_p90 for each var.
    """
    schema = thermals_lf.collect_schema()
    available = [m for m in _THERMAL_FLOW_VARS if m in schema]

    if not available:
        return pl.DataFrame(
            {
                "thermal_id": pl.Series([], dtype=pl.Int32),
                "stage_id": pl.Series([], dtype=pl.Int32),
            }
        )

    return (
        thermals_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
        .group_by(["scenario_id", "stage_id", "thermal_id"])
        .agg(
            [(pl.col(m) * pl.col("_bh")).sum() / pl.col("_bh").sum() for m in available]
        )
        .group_by(["stage_id", "thermal_id"])
        .agg(
            [
                expr
                for m in available
                for expr in [
                    pl.col(m).quantile(0.1, interpolation="linear").alias(f"{m}_p10"),
                    pl.col(m).quantile(0.5, interpolation="linear").alias(f"{m}_p50"),
                    pl.col(m).quantile(0.9, interpolation="linear").alias(f"{m}_p90"),
                ]
            ]
        )
        .sort(["thermal_id", "stage_id"])
        .collect(engine="streaming")
    )


def _build_thermal_json(
    thermal_meta: dict[int, dict],
    percentiles: pl.DataFrame,
    stages: list[int],
    stage_dates: dict[int, str],
    bus_names: dict[int, str],
    lp_bounds: pd.DataFrame,
    block_hours: dict[tuple[int, int], float],
) -> tuple[dict[str, dict], list[str]]:
    """Build per-plant data dict for JS embedding (thermal).

    Returns ``(thermal_data_dict, xdates_list)`` where ``thermal_data_dict``
    maps string plant-id keys to dicts containing percentile arrays and
    optional LP bounds arrays.
    """
    xdates = stage_x_dates(stages, stage_dates)
    thermal_data: dict[str, dict] = {}

    # Pre-index percentiles by (thermal_id, stage_id)
    pct_index: dict[tuple[int, int], dict] = {}
    for row in percentiles.iter_rows(named=True):
        pct_index[(row["thermal_id"], row["stage_id"])] = row

    # Pre-process lp_bounds for thermals (entity_type_code == 1):
    # tid -> {stage_id -> {bt_code -> block-hours-weighted bound}}. Weighting
    # keeps the bound comparable to the block-hours weighted generation series
    # (see _weighted_lp_generation_bounds) — otherwise a must-run plant whose
    # per-block levels differ reads as violating its own cap.
    lp_index = _weighted_lp_generation_bounds(lp_bounds, 1, block_hours)

    for tid, meta in sorted(thermal_meta.items()):
        name: str = meta.get("name", str(tid))
        bus_id: int = meta.get("bus_id", -1)
        bus: str = bus_names.get(bus_id, str(bus_id))
        max_mw: float = float(meta.get("max_mw", 0) or 0)
        cost_per_mwh: float = float(meta.get("cost_per_mwh", 0) or 0)

        entry: dict = {
            "name": name,
            "bus": bus,
            "max_mw": round(max_mw, 1),
            "cost_per_mwh": round(cost_per_mwh, 2),
        }

        # Percentile arrays for each variable
        for var in _THERMAL_FLOW_VARS:
            k = _THERMAL_VAR_SHORT[var]
            for sfx in ["p10", "p50", "p90"]:
                col = f"{var}_{sfx}"
                entry[f"{k}_{sfx}"] = [
                    round(float(pct_index.get((tid, s), {}).get(col, 0) or 0), 4)
                    for s in stages
                ]

        # lp_bounds: generation min/max (bound_type_code 6=min, 7=max)
        if tid in lp_index:
            lp_stages = lp_index[tid]
            entry["gen_min"] = [
                round(lp_stages.get(s, {}).get(6, 0.0), 4) for s in stages
            ]
            entry["gen_max"] = [
                round(lp_stages.get(s, {}).get(7, 0.0), 4) for s in stages
            ]
        else:
            entry["gen_min"] = []
            entry["gen_max"] = []

        thermal_data[str(tid)] = entry

    return thermal_data, xdates


def build_thermal_explorer(
    thermals_lf: pl.LazyFrame,
    thermal_meta: dict[int, dict],
    bus_names: dict[int, str],
    stage_dates: dict[int, str],
    lp_bounds: pd.DataFrame,
    bh_df: pl.DataFrame,
) -> str:
    """Build the HTML for the thermal plant explorer sub-tab (data + JS only).

    Returns an HTML string with the explorer layout and embedded data/JS.
    Returns a no-data paragraph if ``thermal_meta`` is empty.

    Note: PLANT_EXPLORER_JS must NOT be emitted here; it is emitted once
    in ``render()`` to avoid duplicate JS function definitions.
    """
    if not thermal_meta:
        return "<p>No thermal plant data available.</p>"

    percentiles = _compute_thermal_percentiles(thermals_lf, bh_df)

    stages: list[int] = []
    if not percentiles.is_empty() and "stage_id" in percentiles.columns:
        stages = sorted(percentiles["stage_id"].unique().to_list())

    thermal_data, xdates = _build_thermal_json(
        thermal_meta=thermal_meta,
        percentiles=percentiles,
        stages=stages,
        stage_dates=stage_dates,
        bus_names=bus_names,
        lp_bounds=lp_bounds if lp_bounds is not None else pd.DataFrame(),
        block_hours=_block_hours_dict(bh_df),
    )

    options = sorted(thermal_data.items(), key=lambda x: x[1]["name"])
    if not options:
        return "<p>No thermal plant data available.</p>"

    # --- Table rows -----------------------------------------------------------
    table_rows: list[str] = []
    for tid, d in options:
        name: str = d.get("name", "")
        bus: str = d.get("bus", "")
        max_mw: float = d.get("max_mw", 0)
        cost_per_mwh: float = d.get("cost_per_mwh", 0)
        gen_p50: list[float] = d.get("gen_p50", [])
        gen_spark = (
            sparkline_svg(gen_p50, "#F5A623")
            if len(gen_p50) >= 2 and any(v != 0 for v in gen_p50)
            else ""
        )
        table_rows.append(
            f'<tr data-name="{escape_attr(name.lower())}" data-index="{tid}">'
            f'<td><input type="checkbox" class="compare-checkbox" data-id="{tid}"></td>'
            f"<td>{escape_text(name)}</td>"
            f"<td>{escape_text(bus)}</td>"
            f'<td data-sort-value="{max_mw:.0f}">{max_mw:.0f}</td>'
            f'<td data-sort-value="{cost_per_mwh:.2f}">{cost_per_mwh:.2f} R$/MWh</td>'
            f"<td>{gen_spark}</td>"
            f"</tr>"
        )

    rows_html = "".join(table_rows)
    columns: list[tuple[str, str]] = [
        ("Cmp", "none"),
        ("Name", "string"),
        ("Bus", "string"),
        ("Capacity (MW)", "number"),
        ("Cost (R$/MWh)", "number"),
        ("Gen", "none"),
    ]
    table_pane = (
        '<div class="explorer-table-pane" style="width:480px;min-width:320px;flex-shrink:0;">'
        + plant_explorer_table(
            "tt-tbody",
            "tt-search",
            columns,
            rows_html,
        )
        + "</div>"
    )

    # --- Detail pane ----------------------------------------------------------
    def _chart_div(div_id: str) -> str:
        return f'<div id="{div_id}" style="width:100%;height:350px;"></div>'

    generation_section = collapsible_section(
        "Generation",
        chart_grid([wrap_chart(_chart_div("tt-gen"))], single=True),
    )
    economics_section = collapsible_section(
        "Economics",
        chart_grid(
            [
                wrap_chart(_chart_div("tt-cost")),
                wrap_chart(_chart_div("tt-energy")),
            ]
        ),
    )

    band_toggle = (
        '<div class="explorer-band-toggle" style="padding:4px 8px;">'
        '<label><input type="checkbox" id="pe-band-toggle" checked> Show p10-p90</label>'
        "</div>"
    )

    detail_pane = (
        '<div id="tt-detail" class="explorer-detail-pane">'
        + band_toggle
        + generation_section
        + economics_section
        + "</div>"
    )

    explorer_html = (
        '<div class="explorer-container">' + table_pane + detail_pane + "</div>"
    )

    # --- Embedded JS (data + render functions only — PLANT_EXPLORER_JS emitted
    #     once in render()) ---------------------------------------------------
    data_json = json_for_script(thermal_data)
    dates_json = json_for_script(xdates)

    inline_js = (
        "\nvar TT = "
        + data_json
        + ";\n"
        + "var TT_DATES = "
        + dates_json
        + ";\n"
        + """
var _TC = {responsive: true};

function renderThermalDetail(containerId, d) {
  if (!d) { return; }
  var lbl = TT_DATES;

  var ttGenBand = plotlyBand(lbl, d.gen_p10, d.gen_p90, 'rgba(245,166,35,0.15)', 'P10\u2013P90');
  ttGenBand.visible = _peBandVisible;
  var genTraces = [
    ttGenBand,
    plotlyLine(lbl, d.gen_p50, '#F5A623', 'P50'),
    plotlyLine(lbl, d.gen_p10, '#F5A623', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.gen_p90, '#F5A623', 'P90', 1, 'dot'),
  ];
  if (d.gen_max && d.gen_max.length > 0 && d.gen_max.some(function(v) { return v > 0; })) {
    genTraces.push(plotlyRef(lbl, d.gen_max, '#DC4C4C', 'Gen Max (LP)'));
  }
  if (d.gen_min && d.gen_min.length > 0 && d.gen_min.some(function(v) { return v > 0; })) {
    genTraces.push(plotlyRef(lbl, d.gen_min, '#4A8B6F', 'Gen Min (LP)'));
  }
  Plotly.react('tt-gen', genTraces,
    plotlyLayout({title: d.name + ' \u2014 Generation (MW)', yaxis: {title: 'MW'}}), _TC);

  var ttCostBand = plotlyBand(lbl, d.cost_p10, d.cost_p90, 'rgba(220,76,76,0.12)', 'P10\u2013P90');
  ttCostBand.visible = _peBandVisible;
  Plotly.react('tt-cost', [
    ttCostBand,
    plotlyLine(lbl, d.cost_p50, '#DC4C4C', 'P50'),
    plotlyLine(lbl, d.cost_p10, '#DC4C4C', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.cost_p90, '#DC4C4C', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Dispatch Cost (R$)', yaxis: {title: 'R$'}}), _TC);

  var ttEnergyBand = plotlyBand(lbl, d.energy_p10, d.energy_p90, 'rgba(74,139,111,0.15)', 'P10\u2013P90');
  ttEnergyBand.visible = _peBandVisible;
  Plotly.react('tt-energy', [
    ttEnergyBand,
    plotlyLine(lbl, d.energy_p50, '#4A8B6F', 'P50'),
    plotlyLine(lbl, d.energy_p10, '#4A8B6F', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.energy_p90, '#4A8B6F', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Generation Energy (MWh)', yaxis: {title: 'MWh'}}), _TC);
}

var _TT_PALETTE = ['#2196F3', '#FF9800', '#4CAF50'];

function renderThermalComparison(entries, labels) {
  if (!entries || entries.length === 0) { return; }
  var lbl = labels || TT_DATES;

  function _buildTraces(p50key) {
    return entries.map(function(d, i) {
      var c = _TT_PALETTE[i % _TT_PALETTE.length];
      return plotlyLine(lbl, d[p50key] || [], c, d.name || ('Plant ' + i));
    });
  }

  Plotly.react('tt-gen', _buildTraces('gen_p50'),
    plotlyLayout({title: 'Generation (MW) \u2014 Comparison', yaxis: {title: 'MW'}}), _TC);
  Plotly.react('tt-cost', _buildTraces('cost_p50'),
    plotlyLayout({title: 'Dispatch Cost (R$) \u2014 Comparison', yaxis: {title: 'R$'}}), _TC);
  Plotly.react('tt-energy', _buildTraces('energy_p50'),
    plotlyLayout({title: 'Generation Energy (MWh) \u2014 Comparison', yaxis: {title: 'MWh'}}), _TC);
}

initPlantExplorer({
  tableId: 'tt-tbody',
  searchInputId: 'tt-search',
  detailContainerId: 'tt-detail',
  dataVar: 'TT',
  labelsVar: 'TT_DATES',
  renderDetail: renderThermalDetail
});

initComparisonMode({
  tableId: 'tt-tbody',
  dataVar: 'TT',
  labelsVar: 'TT_DATES',
  chartIds: ['tt-gen','tt-cost','tt-energy'],
  renderComparison: renderThermalComparison,
  renderDetail: renderThermalDetail,
  maxCompare: 3
});

syncHover(['tt-gen','tt-cost','tt-energy']);
"""
    )

    return explorer_html + "<script>\n" + inline_js + "\n</script>"


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_hydro_explorer(
    hydros_lf: pl.LazyFrame,
    hydro_meta: dict[int, dict],
    bus_names: dict[int, str],
    stage_dates: dict[int, str],
    bh_df: pl.DataFrame,
    hydro_bounds: pd.DataFrame,
    lp_bounds: pd.DataFrame,
) -> str:
    """Build the HTML for the hydro plant explorer sub-tab.

    Returns the complete HTML string (explorer layout + embedded JS + data).
    Returns a no-data paragraph if ``hydro_meta`` is empty.
    """
    if not hydro_meta:
        return "<p>No hydro plant data available.</p>"

    percentiles = _compute_hydro_percentiles(hydros_lf, bh_df)

    stages: list[int] = []
    if not percentiles.is_empty() and "stage_id" in percentiles.columns:
        stages = sorted(percentiles["stage_id"].unique().to_list())

    hydro_data, xdates = _build_hydro_json(
        hydro_meta=hydro_meta,
        percentiles=percentiles,
        stages=stages,
        stage_dates=stage_dates,
        bus_names=bus_names,
        hydro_bounds=hydro_bounds if hydro_bounds is not None else pd.DataFrame(),
        lp_bounds=lp_bounds if lp_bounds is not None else pd.DataFrame(),
        block_hours=_block_hours_dict(bh_df),
    )

    options = sorted(hydro_data.items(), key=lambda x: x[1]["name"])
    if not options:
        return "<p>No hydro plant data available.</p>"

    # --- Table rows -----------------------------------------------------------
    table_rows: list[str] = []
    for hid, d in options:
        name: str = d.get("name", "")
        bus: str = d.get("bus", "")
        max_gen_mw: float = d.get("max_gen_mw", 0)
        vol_max: float = d.get("vol_max", 0)
        gen_p50: list[float] = d.get("gen_p50", [])
        gen_spark = (
            sparkline_svg(gen_p50, "#2196F3")
            if len(gen_p50) >= 2 and any(v != 0 for v in gen_p50)
            else ""
        )
        table_rows.append(
            f'<tr data-name="{escape_attr(name.lower())}" data-index="{hid}">'
            f'<td><input type="checkbox" class="compare-checkbox" data-id="{hid}"></td>'
            f"<td>{escape_text(name)}</td>"
            f"<td>{escape_text(bus)}</td>"
            f'<td data-sort-value="{max_gen_mw:.0f}">{max_gen_mw:.0f}</td>'
            f'<td data-sort-value="{vol_max:.0f}">{vol_max:.0f}</td>'
            f"<td>{gen_spark}</td>"
            f"</tr>"
        )

    rows_html = "".join(table_rows)
    columns: list[tuple[str, str]] = [
        ("Cmp", "none"),
        ("Name", "string"),
        ("Bus", "string"),
        ("Max Gen (MW)", "number"),
        ("Vol Max (hm3)", "number"),
        ("Gen", "none"),
    ]
    table_pane = (
        '<div class="explorer-table-pane" style="width:480px;min-width:320px;flex-shrink:0;">'
        + plant_explorer_table(
            "hp-tbody",
            "hp-search",
            columns,
            rows_html,
        )
        + "</div>"
    )

    # --- Detail pane ----------------------------------------------------------
    def _chart_div(div_id: str) -> str:
        return f'<div id="{div_id}" style="width:100%;height:350px;"></div>'

    water_balance_section = collapsible_section(
        "Water Balance",
        chart_grid(
            [
                wrap_chart(_chart_div("hp-stor")),
                wrap_chart(_chart_div("hp-inflow")),
                wrap_chart(_chart_div("hp-spill")),
                wrap_chart(_chart_div("hp-evap")),
            ]
        ),
    )
    generation_section = collapsible_section(
        "Generation",
        chart_grid(
            [
                wrap_chart(_chart_div("hp-gen")),
                wrap_chart(_chart_div("hp-turb")),
                wrap_chart(_chart_div("hp-outflow")),
                wrap_chart(_chart_div("hp-wv")),
            ]
        ),
    )
    energy_section = collapsible_section(
        "Energy",
        chart_grid(
            [
                wrap_chart(_chart_div("hp-earm")),
                wrap_chart(_chart_div("hp-ena")),
                wrap_chart(_chart_div("hp-rhoeq")),
            ]
        ),
    )
    # Operational slack columns: water-withdrawal violation in both
    # directions (positive = over-withdrew, negative = under-withdrew)
    # plus the inflow non-negativity slack — useful to localize which
    # LP constraints were relaxed under stochastic noise.
    slacks_section = collapsible_section(
        "Operational Slacks",
        chart_grid(
            [
                wrap_chart(_chart_div("hp-wpos")),
                wrap_chart(_chart_div("hp-wneg")),
                wrap_chart(_chart_div("hp-innn")),
            ]
        ),
    )

    band_toggle = (
        '<div class="explorer-band-toggle" style="padding:4px 8px;">'
        '<label><input type="checkbox" id="pe-band-toggle" checked> Show p10-p90</label>'
        "</div>"
    )

    detail_pane = (
        '<div id="hp-detail" class="explorer-detail-pane">'
        + band_toggle
        + water_balance_section
        + generation_section
        + energy_section
        + slacks_section
        + "</div>"
    )

    explorer_html = (
        '<div class="explorer-container">' + table_pane + detail_pane + "</div>"
    )

    # --- Embedded JS (data + render functions only — PLANT_EXPLORER_JS emitted
    #     once in render()) ---------------------------------------------------
    data_json = json_for_script(hydro_data)
    dates_json = json_for_script(xdates)

    inline_js = (
        "\nvar HP = "
        + data_json
        + ";\n"
        + "var HP_DATES = "
        + dates_json
        + ";\n"
        + """
var _HC = {responsive: true};
var _peBandVisible = true;

document.addEventListener('change', function(e) {
  if (e.target.id !== 'pe-band-toggle') return;
  _peBandVisible = e.target.checked;
  var activeRow = document.querySelector('.explorer-row-selected');
  if (activeRow) activeRow.click();
});

function renderHydroDetail(containerId, d) {
  if (!d) { return; }
  var lbl = HP_DATES;

  // Storage (hm3) — stage-level
  var storBand = plotlyBand(lbl, d.stor_p10, d.stor_p90, 'rgba(33,150,243,0.15)', 'P10\u2013P90');
  storBand.visible = _peBandVisible;
  var storTraces = [
    storBand,
    plotlyLine(lbl, d.stor_p50, '#2196F3', 'P50'),
    plotlyLine(lbl, d.stor_p10, '#2196F3', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.stor_p90, '#2196F3', 'P90', 1, 'dot'),
  ];
  if (d.stor_min && d.stor_min.length > 0 && d.stor_min.some(function(v){return v>0;})) {
    storTraces.push(plotlyRef(lbl, d.stor_min, '#4A8B6F', 'Min Storage'));
  }
  if (d.stor_max && d.stor_max.length > 0 && d.stor_max.some(function(v){return v>0;})) {
    storTraces.push(plotlyRef(lbl, d.stor_max, '#DC4C4C', 'Max Storage'));
  }
  Plotly.react('hp-stor', storTraces,
    plotlyLayout({title: d.name + ' \u2014 Storage (hm\u00b3)', yaxis: {title: 'hm\u00b3'}}), _HC);

  // Inflow (m3/s)
  var inflowBand = plotlyBand(lbl, d.inflow_p10, d.inflow_p90, 'rgba(76,175,80,0.15)', 'P10\u2013P90');
  inflowBand.visible = _peBandVisible;
  Plotly.react('hp-inflow', [
    inflowBand,
    plotlyLine(lbl, d.inflow_p50, '#4CAF50', 'P50'),
    plotlyLine(lbl, d.inflow_p10, '#4CAF50', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.inflow_p90, '#4CAF50', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Inflow (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _HC);

  // Spillage (m3/s)
  var spillBand = plotlyBand(lbl, d.spill_p10, d.spill_p90, 'rgba(255,152,0,0.15)', 'P10\u2013P90');
  spillBand.visible = _peBandVisible;
  var spillTraces = [
    spillBand,
    plotlyLine(lbl, d.spill_p50, '#FF9800', 'P50'),
    plotlyLine(lbl, d.spill_p10, '#FF9800', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.spill_p90, '#FF9800', 'P90', 1, 'dot'),
  ];
  if (d.spill_min && d.spill_min.length > 0 && d.spill_min.some(function(v){return v>0;})) {
    spillTraces.push(plotlyRef(lbl, d.spill_min, '#4A8B6F', 'Min Spillage'));
  }
  if (d.spill_max && d.spill_max.length > 0 && d.spill_max.some(function(v){return v>0;})) {
    spillTraces.push(plotlyRef(lbl, d.spill_max, '#DC4C4C', 'Max Spillage'));
  }
  Plotly.react('hp-spill', spillTraces,
    plotlyLayout({title: 'Spillage (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _HC);

  // Evaporation (m3/s)
  var evapBand = plotlyBand(lbl, d.evap_p10, d.evap_p90, 'rgba(156,39,176,0.12)', 'P10\u2013P90');
  evapBand.visible = _peBandVisible;
  Plotly.react('hp-evap', [
    evapBand,
    plotlyLine(lbl, d.evap_p50, '#9C27B0', 'P50'),
    plotlyLine(lbl, d.evap_p10, '#9C27B0', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.evap_p90, '#9C27B0', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Evaporation (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _HC);

  // Generation (MW) — flow variable
  var genBand = plotlyBand(lbl, d.gen_p10, d.gen_p90, 'rgba(245,166,35,0.15)', 'P10\u2013P90');
  genBand.visible = _peBandVisible;
  var genTraces = [
    genBand,
    plotlyLine(lbl, d.gen_p50, '#F5A623', 'P50'),
    plotlyLine(lbl, d.gen_p10, '#F5A623', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.gen_p90, '#F5A623', 'P90', 1, 'dot'),
  ];
  if (d.gen_max && d.gen_max.length > 0 && d.gen_max.some(function(v){return v>0;})) {
    genTraces.push(plotlyRef(lbl, d.gen_max, '#DC4C4C', 'Gen Max (LP)'));
  }
  if (d.gen_min && d.gen_min.length > 0 && d.gen_min.some(function(v){return v>0;})) {
    genTraces.push(plotlyRef(lbl, d.gen_min, '#4A8B6F', 'Gen Min (LP)'));
  }
  Plotly.react('hp-gen', genTraces,
    plotlyLayout({title: d.name + ' \u2014 Generation (MW)', yaxis: {title: 'MW'}}), _HC);

  // Turbined (m3/s)
  var turbBand = plotlyBand(lbl, d.turb_p10, d.turb_p90, 'rgba(3,169,244,0.15)', 'P10\u2013P90');
  turbBand.visible = _peBandVisible;
  var turbTraces = [
    turbBand,
    plotlyLine(lbl, d.turb_p50, '#03A9F4', 'P50'),
    plotlyLine(lbl, d.turb_p10, '#03A9F4', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.turb_p90, '#03A9F4', 'P90', 1, 'dot'),
  ];
  if (d.turb_min && d.turb_min.length > 0 && d.turb_min.some(function(v){return v>0;})) {
    turbTraces.push(plotlyRef(lbl, d.turb_min, '#4A8B6F', 'Min Turbined'));
  }
  if (d.turb_max && d.turb_max.length > 0 && d.turb_max.some(function(v){return v>0;})) {
    turbTraces.push(plotlyRef(lbl, d.turb_max, '#DC4C4C', 'Max Turbined'));
  }
  Plotly.react('hp-turb', turbTraces,
    plotlyLayout({title: 'Turbined (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _HC);

  // Outflow (m3/s) — derived turbined + spillage
  var outflowBand = plotlyBand(lbl, d.outflow_p10, d.outflow_p90, 'rgba(0,188,212,0.15)', 'P10\u2013P90');
  outflowBand.visible = _peBandVisible;
  var outflowTraces = [
    outflowBand,
    plotlyLine(lbl, d.outflow_p50, '#00BCD4', 'P50'),
    plotlyLine(lbl, d.outflow_p10, '#00BCD4', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.outflow_p90, '#00BCD4', 'P90', 1, 'dot'),
  ];
  if (d.outflow_min && d.outflow_min.length > 0 && d.outflow_min.some(function(v){return v>0;})) {
    outflowTraces.push(plotlyRef(lbl, d.outflow_min, '#4A8B6F', 'Min Outflow'));
  }
  if (d.outflow_max && d.outflow_max.length > 0 && d.outflow_max.some(function(v){return v>0;})) {
    outflowTraces.push(plotlyRef(lbl, d.outflow_max, '#DC4C4C', 'Max Outflow'));
  }
  Plotly.react('hp-outflow', outflowTraces,
    plotlyLayout({title: 'Outflow (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _HC);

  // Water Value (R$/hm3) — stage-level
  var wvBand = plotlyBand(lbl, d.wv_p10, d.wv_p90, 'rgba(233,30,99,0.12)', 'P10\u2013P90');
  wvBand.visible = _peBandVisible;
  Plotly.react('hp-wv', [
    wvBand,
    plotlyLine(lbl, d.wv_p50, '#E91E63', 'P50'),
    plotlyLine(lbl, d.wv_p10, '#E91E63', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.wv_p90, '#E91E63', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Water Value (R$/hm\u00b3)', yaxis: {title: 'R$/hm\u00b3'}}), _HC);

  // Stored Energy (MWh) \u2014 EARM, stage-level. cobre HEAD column.
  if (d.earm_p50 && d.earm_p50.length > 0) {
    var earmBand = plotlyBand(lbl, d.earm_p10, d.earm_p90, 'rgba(63,81,181,0.15)', 'P10\u2013P90');
    earmBand.visible = _peBandVisible;
    Plotly.react('hp-earm', [
      earmBand,
      plotlyLine(lbl, d.earm_p50, '#3F51B5', 'P50'),
      plotlyLine(lbl, d.earm_p10, '#3F51B5', 'P10', 1, 'dot'),
      plotlyLine(lbl, d.earm_p90, '#3F51B5', 'P90', 1, 'dot'),
    ], plotlyLayout({title: 'Stored Energy (MWh)', yaxis: {title: 'MWh'}}), _HC);
  }

  // Inflow Energy (MW) \u2014 ENA, stage-level. cobre HEAD column.
  if (d.ena_p50 && d.ena_p50.length > 0) {
    var enaBand = plotlyBand(lbl, d.ena_p10, d.ena_p90, 'rgba(0,150,136,0.15)', 'P10\u2013P90');
    enaBand.visible = _peBandVisible;
    Plotly.react('hp-ena', [
      enaBand,
      plotlyLine(lbl, d.ena_p50, '#009688', 'P50'),
      plotlyLine(lbl, d.ena_p10, '#009688', 'P10', 1, 'dot'),
      plotlyLine(lbl, d.ena_p90, '#009688', 'P90', 1, 'dot'),
    ], plotlyLayout({title: 'Inflow Energy (MW)', yaxis: {title: 'MW'}}), _HC);
  }

  // Equivalent Productivity (MW/(m\u00b3/s)) \u2014 \u03c1_eq, stage-level. cobre HEAD column.
  if (d.rhoeq_p50 && d.rhoeq_p50.length > 0) {
    var rhoeqBand = plotlyBand(lbl, d.rhoeq_p10, d.rhoeq_p90, 'rgba(121,85,72,0.15)', 'P10\u2013P90');
    rhoeqBand.visible = _peBandVisible;
    Plotly.react('hp-rhoeq', [
      rhoeqBand,
      plotlyLine(lbl, d.rhoeq_p50, '#795548', 'P50'),
      plotlyLine(lbl, d.rhoeq_p10, '#795548', 'P10', 1, 'dot'),
      plotlyLine(lbl, d.rhoeq_p90, '#795548', 'P90', 1, 'dot'),
    ], plotlyLayout({title: 'Equivalent Productivity (MW/(m\u00b3/s))', yaxis: {title: 'MW/(m\u00b3/s)'}}), _HC);
  }

  // Operational slacks (m\u00b3/s) \u2014 each is silent when the LP never
  // touched it.  Plotted in the standard p10/p50/p90 idiom so the user
  // can see at a glance whether a violation was a tail event (band
  // wide, p50 near zero) or systematic (whole band non-zero).
  var wposBand = plotlyBand(lbl, d.wpos_p10, d.wpos_p90, 'rgba(244,67,54,0.15)', 'P10\u2013P90');
  wposBand.visible = _peBandVisible;
  Plotly.react('hp-wpos', [
    wposBand,
    plotlyLine(lbl, d.wpos_p50, '#F44336', 'P50'),
    plotlyLine(lbl, d.wpos_p10, '#F44336', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.wpos_p90, '#F44336', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Withdrawal Slack Pos (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _HC);

  var wnegBand = plotlyBand(lbl, d.wneg_p10, d.wneg_p90, 'rgba(244,67,54,0.15)', 'P10\u2013P90');
  wnegBand.visible = _peBandVisible;
  Plotly.react('hp-wneg', [
    wnegBand,
    plotlyLine(lbl, d.wneg_p50, '#F44336', 'P50'),
    plotlyLine(lbl, d.wneg_p10, '#F44336', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.wneg_p90, '#F44336', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Withdrawal Slack Neg (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _HC);

  var innnBand = plotlyBand(lbl, d.innn_p10, d.innn_p90, 'rgba(156,39,176,0.15)', 'P10\u2013P90');
  innnBand.visible = _peBandVisible;
  Plotly.react('hp-innn', [
    innnBand,
    plotlyLine(lbl, d.innn_p50, '#9C27B0', 'P50'),
    plotlyLine(lbl, d.innn_p10, '#9C27B0', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.innn_p90, '#9C27B0', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Inflow Non-Negativity Slack (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _HC);
}

var _HP_PALETTE = ['#2196F3', '#FF9800', '#4CAF50'];

function renderHydroComparison(entries, labels) {
  if (!entries || entries.length === 0) { return; }
  var lbl = labels || HP_DATES;

  function _buildTraces(p50key) {
    return entries.map(function(d, i) {
      var c = _HP_PALETTE[i % _HP_PALETTE.length];
      return plotlyLine(lbl, d[p50key] || [], c, d.name || ('Plant ' + i));
    });
  }

  Plotly.react('hp-stor', _buildTraces('stor_p50'),
    plotlyLayout({title: 'Storage (hm\u00b3) \u2014 Comparison', yaxis: {title: 'hm\u00b3'}}), _HC);
  Plotly.react('hp-inflow', _buildTraces('inflow_p50'),
    plotlyLayout({title: 'Inflow (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _HC);
  Plotly.react('hp-spill', _buildTraces('spill_p50'),
    plotlyLayout({title: 'Spillage (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _HC);
  Plotly.react('hp-evap', _buildTraces('evap_p50'),
    plotlyLayout({title: 'Evaporation (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _HC);
  Plotly.react('hp-gen', _buildTraces('gen_p50'),
    plotlyLayout({title: 'Generation (MW) \u2014 Comparison', yaxis: {title: 'MW'}}), _HC);
  Plotly.react('hp-turb', _buildTraces('turb_p50'),
    plotlyLayout({title: 'Turbined (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _HC);
  Plotly.react('hp-outflow', _buildTraces('outflow_p50'),
    plotlyLayout({title: 'Outflow (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _HC);
  Plotly.react('hp-wv', _buildTraces('wv_p50'),
    plotlyLayout({title: 'Water Value (R$/hm\u00b3) \u2014 Comparison', yaxis: {title: 'R$/hm\u00b3'}}), _HC);

  // Comparison mode for energy-domain charts; safe no-op if columns absent.
  if (entries.every(function(d){return d && d.earm_p50;})) {
    Plotly.react('hp-earm', _buildTraces('earm_p50'),
      plotlyLayout({title: 'Stored Energy (MWh) \u2014 Comparison', yaxis: {title: 'MWh'}}), _HC);
  }
  if (entries.every(function(d){return d && d.ena_p50;})) {
    Plotly.react('hp-ena', _buildTraces('ena_p50'),
      plotlyLayout({title: 'Inflow Energy (MW) \u2014 Comparison', yaxis: {title: 'MW'}}), _HC);
  }
  if (entries.every(function(d){return d && d.rhoeq_p50;})) {
    Plotly.react('hp-rhoeq', _buildTraces('rhoeq_p50'),
      plotlyLayout({title: 'Equivalent Productivity \u2014 Comparison', yaxis: {title: 'MW/(m\u00b3/s)'}}), _HC);
  }

  // Slack-variable comparison mode (silent no-op when every selected
  // plant has zero slack \u2014 the trace just shows a flat zero line).
  Plotly.react('hp-wpos', _buildTraces('wpos_p50'),
    plotlyLayout({title: 'Withdrawal Slack Pos (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _HC);
  Plotly.react('hp-wneg', _buildTraces('wneg_p50'),
    plotlyLayout({title: 'Withdrawal Slack Neg (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _HC);
  Plotly.react('hp-innn', _buildTraces('innn_p50'),
    plotlyLayout({title: 'Inflow Non-Negativity Slack (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _HC);
}

initPlantExplorer({
  tableId: 'hp-tbody',
  searchInputId: 'hp-search',
  detailContainerId: 'hp-detail',
  dataVar: 'HP',
  labelsVar: 'HP_DATES',
  renderDetail: renderHydroDetail
});

initComparisonMode({
  tableId: 'hp-tbody',
  dataVar: 'HP',
  labelsVar: 'HP_DATES',
  chartIds: ['hp-stor','hp-inflow','hp-spill','hp-evap','hp-gen','hp-turb','hp-outflow','hp-wv','hp-earm','hp-ena','hp-rhoeq','hp-wpos','hp-wneg','hp-innn'],
  renderComparison: renderHydroComparison,
  renderDetail: renderHydroDetail,
  maxCompare: 3
});

syncHover(['hp-stor','hp-inflow','hp-spill','hp-evap','hp-gen','hp-turb','hp-outflow','hp-wv','hp-earm','hp-ena','hp-rhoeq','hp-wpos','hp-wneg','hp-innn']);
"""
    )

    return explorer_html + "<script>\n" + inline_js + "\n</script>"


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:
    """Plant Explorer renders only when simulation data is available."""
    return data.simulation_available


def render(data: DashboardData) -> str:
    """Return the full HTML string for the Plant Explorer tab content area.

    Wraps the hydro and thermal sub-tabs in a sub-tab bar with switching.
    SUB_TAB_JS and PLANT_EXPLORER_JS are declared via REQUIRED_JS and emitted
    once in the document shell, not inline here.
    """
    hydro_content = build_hydro_explorer(
        hydros_lf=data.hydros_lf,
        hydro_meta=data.hydro_meta,
        bus_names=data.bus_names,
        stage_dates=data.stage_dates,
        bh_df=data.bh_df,
        hydro_bounds=data.hydro_bounds,
        lp_bounds=data.lp_bounds,
    )
    thermal_content = build_thermal_explorer(
        thermals_lf=data.thermals_lf,
        thermal_meta=data.thermal_meta,
        bus_names=data.bus_names,
        stage_dates=data.stage_dates,
        lp_bounds=data.lp_bounds,
        bh_df=data.bh_df,
    )

    sub_tab_bar = (
        '<div class="sub-tab-bar">'
        "<button class=\"sub-tab-btn active\" onclick=\"switchSubTab('plants-hydro', 'plants-explorer')\">Hydro</button>"
        "<button class=\"sub-tab-btn\" onclick=\"switchSubTab('plants-thermal', 'plants-explorer')\">Thermal</button>"
        "</div>"
    )

    hydro_panel = (
        '<div id="plants-hydro" class="sub-tab-panel" style="display:block;">'
        + hydro_content
        + "</div>"
    )
    thermal_panel = (
        '<div id="plants-thermal" class="sub-tab-panel" style="display:none;">'
        + thermal_content
        + "</div>"
    )

    group = (
        '<div data-subtab-group="plants-explorer">'
        + sub_tab_bar
        + hydro_panel
        + thermal_panel
        + "</div>"
    )

    return section_title("Plant Explorer") + group
