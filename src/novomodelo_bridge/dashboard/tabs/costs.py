"""v2 Costs tab — NPV financial analysis and temporal evolution of simulation costs.

Displays a 4-card NPV metrics row (total expected cost, immediate cost,
thermal cost, deficit cost), a horizontal stacked cost breakdown bar chart,
and a companion summary table with component/mean/std/p10/p90/percentage
columns.

This module also provides four temporal evolution sections:
  D. Cost Composition by Stage — stacked area, undiscounted.
  E. Cost Category Trends — line chart with p10-p90 bands per group.
  F. Spot Price by Bus — faceted mean+p50+band, block-hours weighted.
  G. Violation Costs — conditional horizontal bar chart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from cobre_bridge.dashboard.chart_helpers import (
    COST_GROUP_COLORS,
    COST_GROUPS,
    add_mean_p50_band,
    build_cost_table,
    chart_cost_bar,
    compute_cost_summary,
    compute_npv_costs,
    compute_percentiles,
    group_costs,
    make_chart_card,
)
from cobre_bridge.ui.html.document import (
    chart_grid,
    collapsible_section,
    json_for_script,
    metric_card,
    metrics_grid,
    section_title,
    wrap_chart,
)
from cobre_bridge.ui.html.plotly import (
    LEGEND_DEFAULTS as _LEGEND,
)
from cobre_bridge.ui.html.plotly import (
    MARGIN_DEFAULTS as _MARGIN,
)
from cobre_bridge.ui.html.plotly import (
    apply_stage_date_axis,
    apply_standard_layout,
    stage_x_dates,
    stage_x_labels,
)
from cobre_bridge.ui.theme import BUS_COLORS, COLORS

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

TAB_ID = "tab-costs"
TAB_LABEL = "Costs"
TAB_ORDER = 40

# Non-cost metadata columns that are excluded from cost aggregations.
# Must match chart_helpers._NON_COST_COLS to avoid double-counting.
_NON_COST_COLS: frozenset[str] = frozenset(
    {
        "scenario_id",
        "stage_id",
        "block_id",
        "total_cost",
        "immediate_cost",
        "future_cost",
        "discount_factor",
        "hydro_violation_cost",  # aggregate of 6 individual hydro violations
    }
)

# All known violation cost column names from the costs DataFrame spec.
# NOTE: hydro_violation_cost is excluded — it's an aggregate of the 6 below.
_VIOLATION_COLS: frozenset[str] = frozenset(
    {
        "storage_violation_cost",
        "filling_target_cost",
        "outflow_violation_below_cost",
        "outflow_violation_above_cost",
        "turbined_violation_cost",
        "generation_violation_cost",
        "evaporation_violation_cost",
        "withdrawal_violation_cost",
        "inflow_penalty_cost",
        "generic_violation_cost",
        "generic_constraint_violation_cost",
    }
)


def can_render(data: DashboardData) -> bool:
    """Costs tab renders only when simulation data is available."""
    return data.simulation_available


def _compute_npv_metric(data: DashboardData, column_name: str) -> float:
    """Compute the mean NPV of a single cost column across scenarios.

    Applies discount factors via :func:`compute_npv_costs`, groups by
    ``scenario_id``, sums *column_name* per scenario, then returns the mean
    across scenarios.

    Args:
        data: Dashboard data instance containing ``costs`` and
            ``discount_rate``.
        column_name: Name of the cost column to aggregate.

    Returns:
        Mean NPV value as a float.  Returns ``0.0`` when the column is
        absent, the DataFrame is empty, or any aggregation fails.
    """
    if data.costs.empty:
        return 0.0
    if column_name not in data.costs.columns:
        return 0.0

    try:
        discounted = compute_npv_costs(data.costs, data.discount_rate)
        if "scenario_id" not in discounted.columns:
            total = float(discounted[column_name].sum())
            return total
        per_scenario = discounted.groupby("scenario_id")[column_name].sum()
        return float(per_scenario.mean())
    except (ValueError, TypeError, KeyError):
        return 0.0


def _build_metrics_row(data: DashboardData) -> str:
    """Build the 4-card NPV metrics row HTML.

    Cards (in order):
    1. Expected Cost (NPV) — total NPV from ``compute_cost_summary``.
    2. Immediate Cost (NPV) — NPV of ``immediate_cost`` column.
    3. Thermal Cost (NPV) — NPV of ``thermal_cost`` column.
    4. Deficit Cost (NPV) — NPV of ``deficit_cost`` column.

    Args:
        data: Dashboard data instance.

    Returns:
        HTML string containing a ``<div class="metrics-grid">`` with 4 cards.
    """
    if data.costs.empty:
        na_card = metric_card(
            "N/A", "Expected Cost (NPV)", color=COST_GROUP_COLORS["Thermal"]
        )
        return metrics_grid([na_card] * 4)

    # Card 1: Total expected cost NPV via compute_cost_summary
    try:
        summary = compute_cost_summary(data.costs, data.discount_rate)
        total_npv_str = (
            f"{float(summary['mean'].sum()):,.0f}" if not summary.empty else "N/A"
        )
    except (ValueError, TypeError, KeyError):
        total_npv_str = "N/A"

    # Cards 2-4: per-column NPV metrics
    def _fmt(col: str) -> str:
        val = _compute_npv_metric(data, col)
        return f"{val:,.0f}" if col in data.costs.columns else "N/A"

    cards = [
        metric_card(
            total_npv_str,
            "Expected Cost (NPV)",
            color=COST_GROUP_COLORS["Thermal"],
        ),
        metric_card(
            _fmt("immediate_cost"),
            "Immediate Cost (NPV)",
            color=COLORS["future_cost"],
        ),
        metric_card(
            _fmt("thermal_cost"),
            "Thermal Cost (NPV)",
            color=COLORS["thermal"],
        ),
        metric_card(
            _fmt("deficit_cost"),
            "Deficit Cost (NPV)",
            color=COLORS["deficit"],
        ),
    ]
    return metrics_grid(cards)


def _render_npv_section(data: DashboardData) -> str:
    """Render the full NPV perspective section.

    Computes the cost summary, then renders:
    - Metrics row (4 cards)
    - Horizontal stacked bar chart card
    - Summary table card

    Args:
        data: Dashboard data instance.

    Returns:
        HTML string for the NPV section body (metrics grid + chart grid).
    """
    metrics_html = _build_metrics_row(data)

    if data.costs.empty:
        chart_html = wrap_chart("<p>No cost data available.</p>")
        table_html = wrap_chart("<p>No cost data available.</p>")
        return metrics_html + chart_grid([chart_html, table_html])

    summary = compute_cost_summary(data.costs, data.discount_rate)
    summary = summary[summary["mean"] > 0]
    if summary.empty:
        chart_html = wrap_chart("<p>No cost data available.</p>")
        table_html = wrap_chart("<p>No cost data available.</p>")
        return metrics_html + chart_grid([chart_html, table_html])

    bar_fig = chart_cost_bar(summary)
    bar_html = make_chart_card(bar_fig, "NPV Cost by Group", "v2-costs-bar-chart")
    table_html = wrap_chart(build_cost_table(summary))
    return metrics_html + chart_grid([bar_html, table_html])


# ---------------------------------------------------------------------------
# Section D — Cost Composition by Stage (stacked area, undiscounted)
# ---------------------------------------------------------------------------


def _render_cost_composition(data: DashboardData) -> str:
    """Render stacked area chart of undiscounted cost composition per stage.

    Groups individual cost component columns into logical categories using
    :func:`~cobre_bridge.dashboard.chart_helpers.group_costs`, computes the
    mean across scenarios per stage (summing blocks first), then builds a
    stacked area chart with one area per group.

    Args:
        data: Dashboard data instance.

    Returns:
        HTML string: a ``collapsible_section`` wrapping the chart card.
    """
    if data.costs.empty:
        return collapsible_section(
            "Cost Composition by Stage",
            "<p>No cost data available.</p>",
            section_id="v2-costs-composition-section",
            default_collapsed=False,
        )

    cost_cols = [c for c in data.costs.columns if c not in _NON_COST_COLS]
    if not cost_cols:
        return collapsible_section(
            "Cost Composition by Stage",
            "<p>No cost data available.</p>",
            section_id="v2-costs-composition-section",
            default_collapsed=False,
        )

    # Sum blocks per (scenario_id, stage_id), then group by cost categories
    by_stage_scen = (
        data.costs.groupby(["scenario_id", "stage_id"])[cost_cols].sum().reset_index()
    )
    grouped = group_costs(by_stage_scen, cost_cols)
    group_cols = [
        c for c in grouped.columns if c not in _NON_COST_COLS and c != "scenario_id"
    ]

    # Mean across scenarios per stage
    mean_by_stage = grouped.groupby("stage_id")[group_cols].mean().reset_index()
    stages = sorted(mean_by_stage["stage_id"].tolist())
    xdates = stage_x_dates(stages, data.stage_dates)
    mean_by_stage["_x"] = mean_by_stage["stage_id"].map(dict(zip(stages, xdates)))

    fig = go.Figure()
    for group_name in COST_GROUPS:
        if group_name not in mean_by_stage.columns:
            continue
        color = COST_GROUP_COLORS.get(group_name, "#6B7280")
        fig.add_trace(
            go.Scatter(
                x=mean_by_stage["_x"],
                y=mean_by_stage[group_name],
                name=group_name,
                stackgroup="costs",
                mode="lines",
                line=dict(color=color, width=0.5),
                fillcolor=color,
            )
        )
    # Include "Other" if it exists and has non-zero values
    if "Other" in mean_by_stage.columns and mean_by_stage["Other"].sum() > 0:
        if "Other" not in [t.name for t in fig.data]:
            fig.add_trace(
                go.Scatter(
                    x=mean_by_stage["_x"],
                    y=mean_by_stage["Other"],
                    name="Other",
                    stackgroup="costs",
                    mode="lines",
                    line=dict(color=COST_GROUP_COLORS["Other"], width=0.5),
                    fillcolor=COST_GROUP_COLORS["Other"],
                )
            )

    fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="Cost (R$)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=_MARGIN,
    )
    all_stages = sorted(data.stage_dates)
    apply_stage_date_axis(
        fig,
        stage_x_dates(all_stages, data.stage_dates),
        stage_x_labels(all_stages, data.stage_labels),
    )

    chart_html = make_chart_card(
        fig,
        "Cost Composition by Stage (undiscounted mean across scenarios)",
        "v2-costs-composition-chart",
        height=480,
    )
    return collapsible_section(
        "Cost Composition by Stage",
        chart_grid([chart_html], single=True),
        section_id="v2-costs-composition-section",
        default_collapsed=False,
    )


# ---------------------------------------------------------------------------
# Section D — interactive group-by composition section
# ---------------------------------------------------------------------------

# 28-color sequential palette for individual cost components.
_COMPONENT_PALETTE: list[str] = [
    "#F59E0B",
    "#EF4444",
    "#3B82F6",
    "#10B981",
    "#8B5CF6",
    "#6B7280",
    "#F97316",
    "#EC4899",
    "#14B8A6",
    "#84CC16",
    "#A78BFA",
    "#FB923C",
    "#22D3EE",
    "#4ADE80",
    "#FBBF24",
    "#60A5FA",
    "#F472B6",
    "#34D399",
    "#C084FC",
    "#FCA5A5",
    "#93C5FD",
    "#6EE7B7",
    "#DDD6FE",
    "#FDE68A",
    "#BAE6FD",
    "#BBF7D0",
    "#E9D5FF",
    "#FEF3C7",
]


def _build_composition_data(data: DashboardData) -> dict | None:
    """Precompute both Category and Component views for the cost composition chart.

    Computes:
    - ``category``: mean per stage for each of the 6 logical groups.
    - ``component``: mean per stage for each non-zero raw cost column (cleaned names).
    - ``total``: ``mean``, ``p10``, ``p90`` of total cost across scenarios per stage.
    - ``stages``: x-axis labels.
    - ``colors``: hex color per group/component key.

    Args:
        data: Dashboard data instance.

    Returns:
        Dict with keys ``category``, ``component``, ``total``, ``stages``,
        ``colors``, or ``None`` when the costs DataFrame is empty or has no
        cost columns.
    """
    if data.costs.empty:
        return None

    cost_cols = [c for c in data.costs.columns if c not in _NON_COST_COLS]
    if not cost_cols:
        return None

    # Sum blocks per (scenario_id, stage_id)
    by_stage_scen: pd.DataFrame = (
        data.costs.groupby(["scenario_id", "stage_id"])[cost_cols].sum().reset_index()
    )

    stages = sorted(by_stage_scen["stage_id"].unique().tolist())
    xlabels = stage_x_labels(stages, data.stage_labels)
    xdates = stage_x_dates(stages, data.stage_dates)

    # --- Category view ---
    grouped = group_costs(by_stage_scen, cost_cols)
    group_col_names = [
        c for c in grouped.columns if c not in _NON_COST_COLS and c != "scenario_id"
    ]
    mean_by_stage_cat = (
        grouped.groupby("stage_id")[group_col_names].mean().reset_index()
    )
    mean_by_stage_cat = mean_by_stage_cat.sort_values("stage_id")

    category: dict[str, list[float]] = {}
    cat_colors: dict[str, str] = {}
    for gname in group_col_names:
        vals = mean_by_stage_cat[gname].tolist()
        category[gname] = [round(float(v), 2) for v in vals]
        cat_colors[gname] = COST_GROUP_COLORS.get(gname, "#6B7280")

    # --- Component view (non-zero columns only, cleaned names) ---
    mean_by_stage_comp = (
        by_stage_scen.groupby("stage_id")[cost_cols].mean().reset_index()
    )
    mean_by_stage_comp = mean_by_stage_comp.sort_values("stage_id")

    component: dict[str, list[float]] = {}
    comp_colors: dict[str, str] = {}
    palette_idx = 0
    for raw_col in cost_cols:
        vals = mean_by_stage_comp[raw_col].tolist()
        if all(v == 0.0 for v in vals):
            continue
        label = raw_col.replace("_cost", "").replace("_", " ").title()
        component[label] = [round(float(v), 2) for v in vals]
        comp_colors[label] = _COMPONENT_PALETTE[palette_idx % len(_COMPONENT_PALETTE)]
        palette_idx += 1

    if not component and all(all(v == 0.0 for v in arr) for arr in category.values()):
        return None

    # --- Total cost: p10/mean/p90 across scenarios per stage ---
    by_stage_scen["_total"] = by_stage_scen[cost_cols].sum(axis=1)
    total_pct = (
        by_stage_scen.groupby("stage_id")["_total"]
        .agg(
            mean="mean",
            p10=lambda x: x.quantile(0.1),
            p90=lambda x: x.quantile(0.9),
        )
        .reset_index()
        .sort_values("stage_id")
    )

    total: dict[str, list[float]] = {
        "mean": [round(float(v), 2) for v in total_pct["mean"].tolist()],
        "p10": [round(float(v), 2) for v in total_pct["p10"].tolist()],
        "p90": [round(float(v), 2) for v in total_pct["p90"].tolist()],
    }

    colors: dict[str, str] = {**cat_colors, **comp_colors}

    return {
        "category": category,
        "component": component,
        "total": total,
        "stages": xdates,
        "stage_labels": xlabels,
        "colors": colors,
    }


def _build_composition_section(data: DashboardData) -> str:
    """Build the interactive Cost Composition by Stage section with group-by dropdown.

    Emits a ``<select id="costs-group-sel">`` dropdown, a
    ``<div id="costs-comp">`` Plotly target, and an inline ``<script>`` that
    embeds precomputed JSON and a ``updateCostsComposition()`` JS function.
    Falls back to the static "No cost data" message when data is unavailable.

    Args:
        data: Dashboard data instance.

    Returns:
        HTML string: a ``collapsible_section`` wrapping the interactive chart.
    """
    comp_data = _build_composition_data(data)

    if comp_data is None:
        return collapsible_section(
            "Cost Composition by Stage",
            "<p>No cost data available.</p>",
            section_id="v2-costs-composition-section",
            default_collapsed=False,
        )

    data_json = json_for_script(comp_data)

    content = (
        '<div style="margin-bottom:16px;">'
        '<label for="costs-group-sel" style="font-weight:600;margin-right:8px;">'
        "Group by:</label>"
        '<select id="costs-group-sel" onchange="updateCostsComposition()" '
        'style="padding:6px 10px;font-size:0.9rem;border-radius:4px;'
        'border:1px solid #ccc;">'
        '<option value="category">Category</option>'
        '<option value="component">Component</option>'
        "</select>"
        "</div>"
        '<div class="chart-grid-single">'
        '<div class="chart-card">'
        '<div id="costs-comp" style="width:100%;height:420px;"></div>'
        "</div>"
        "</div>"
        "<script>\n"
        "const COSTS_COMP_DATA = " + data_json + ";\n"
        r"""
function updateCostsComposition() {
  var sel = document.getElementById('costs-group-sel');
  var view = sel ? sel.value : 'category';
  var d = COSTS_COMP_DATA;
  var groups = d[view];
  var stages = d.stages;
  var colors = d.colors;
  var traces = [];

  // Stacked area traces (one per group/component)
  Object.keys(groups).forEach(function(name) {
    traces.push({
      x: stages, y: groups[name], name: name,
      stackgroup: 'costs', mode: 'lines',
      line: {color: colors[name] || '#6B7280', width: 0.5},
      fillcolor: colors[name] || '#6B7280',
      hovertemplate: '%{y:,.0f}<extra>' + name + '</extra>'
    });
  });

  // p10-p90 band (lower bound, invisible reference)
  traces.push({
    x: stages, y: d.total.p10,
    name: 'Total P10', showlegend: false, mode: 'lines',
    line: {width: 0}, hoverinfo: 'skip'
  });
  // p10-p90 band (upper bound, fills to p10)
  traces.push({
    x: stages, y: d.total.p90,
    name: 'Total P10\u2013P90', showlegend: true, mode: 'lines',
    line: {width: 0}, fill: 'tonexty',
    fillcolor: 'rgba(55,65,81,0.15)', hoverinfo: 'skip'
  });
  // Total mean line
  traces.push({
    x: stages, y: d.total.mean,
    name: 'Total Mean', mode: 'lines',
    line: {color: '#374151', width: 2, dash: 'dash'},
    hovertemplate: '%{y:,.0f}<extra>Total Mean</extra>'
  });

  var layout = {
    title: {text: 'Cost Composition by Stage (undiscounted mean)', font: {size: 13}, x: 0.02},
    xaxis: {title: 'Stage', type: 'date', tickmode: 'array', tickvals: d.stages, ticktext: d.stage_labels},
    yaxis: {title: 'Cost (R$)'},
    legend: {orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'center', x: 0.5, font: {size: 11}},
    margin: {l: 60, r: 20, t: 60, b: 50},
    template: 'plotly_white'
  };
  Plotly.react('costs-comp', traces, layout, {responsive: true});
}
document.addEventListener('DOMContentLoaded', function() {
  setTimeout(updateCostsComposition, 100);
});
""" + "</script>"
    )

    return collapsible_section(
        "Cost Composition by Stage",
        content,
        section_id="v2-costs-composition-section",
        default_collapsed=False,
    )


# ---------------------------------------------------------------------------
# Section E — Cost Category Trends (line chart with p10-p90 bands)
# ---------------------------------------------------------------------------


def _render_category_evolution(data: DashboardData) -> str:
    """Render line chart of cost category trends with p10-p90 bands per stage.

    For each cost group, computes per-(scenario_id, stage_id) sum (across
    blocks), then uses :func:`compute_percentiles` and
    :func:`add_mean_p50_band` to show the distribution across scenarios.

    Args:
        data: Dashboard data instance.

    Returns:
        HTML string: a ``collapsible_section`` wrapping the chart card.
    """
    if data.costs.empty:
        return collapsible_section(
            "Cost Category Trends",
            "<p>No cost data available.</p>",
            section_id="v2-costs-trends-section",
            default_collapsed=True,
        )

    cost_cols = [c for c in data.costs.columns if c not in _NON_COST_COLS]
    if not cost_cols:
        return collapsible_section(
            "Cost Category Trends",
            "<p>No cost data available.</p>",
            section_id="v2-costs-trends-section",
            default_collapsed=True,
        )

    # Sum blocks per (scenario_id, stage_id), then group by cost categories
    by_stage_scen = (
        data.costs.groupby(["scenario_id", "stage_id"])[cost_cols].sum().reset_index()
    )
    grouped = group_costs(by_stage_scen, cost_cols)
    group_cols = [
        c for c in grouped.columns if c not in _NON_COST_COLS and c != "scenario_id"
    ]

    stages = sorted(grouped["stage_id"].unique().tolist())
    xdates = stage_x_dates(stages, data.stage_dates)
    stage_to_x = dict(zip(stages, xdates))

    fig = go.Figure()
    for group_name in group_cols:
        color = COST_GROUP_COLORS.get(group_name, "#6B7280")
        sub = grouped[["scenario_id", "stage_id", group_name]].copy()
        pct = compute_percentiles(sub, ["stage_id"], group_name)
        if pct.empty:
            continue
        pct["_x"] = pct["stage_id"].map(stage_to_x)
        add_mean_p50_band(fig, pct, "_x", group_name, color)

    apply_standard_layout(
        fig,
        xaxis_title="Stage",
        yaxis_title="Cost (R$)",
    )
    all_stages = sorted(data.stage_dates)
    apply_stage_date_axis(
        fig,
        stage_x_dates(all_stages, data.stage_dates),
        stage_x_labels(all_stages, data.stage_labels),
    )

    chart_html = make_chart_card(
        fig,
        "Cost Category Trends (undiscounted, p10/mean/p50/p90 across scenarios)",
        "v2-costs-trends-chart",
        height=420,
    )
    return collapsible_section(
        "Cost Category Trends",
        chart_grid([chart_html], single=True),
        section_id="v2-costs-trends-section",
        default_collapsed=True,
    )


# ---------------------------------------------------------------------------
# Section F — Spot Price by Bus (faceted, block-hours weighted)
# ---------------------------------------------------------------------------


def _render_spot_price(data: DashboardData) -> str:
    """Render faceted spot price chart: one subplot per non-fictitious bus.

    Spot price is weighted by block-hours before computing percentiles:
    ``sum(spot_price * _bh) / sum(_bh)`` per ``(scenario_id, stage_id, bus_id)``.
    Then :func:`compute_percentiles` + :func:`add_mean_p50_band` per bus subplot.

    Args:
        data: Dashboard data instance.

    Returns:
        HTML string: a ``collapsible_section`` wrapping the chart card.
    """
    bus_ids = sorted(data.non_fictitious_bus_ids)

    # Guard: no buses or no buses_lf data
    if not bus_ids:
        return collapsible_section(
            "Spot Price by Bus",
            "<p>No spot price data.</p>",
            section_id="v2-costs-spot-price-section",
            default_collapsed=True,
        )

    try:
        weighted = (
            data.buses_lf.filter(pl.col("bus_id").is_in(bus_ids))
            .join(data.bh_df.lazy(), on=["stage_id", "block_id"])
            .with_columns((pl.col("spot_price") * pl.col("_bh")).alias("_sp_bh"))
            .group_by(["scenario_id", "stage_id", "bus_id"])
            .agg((pl.col("_sp_bh").sum() / pl.col("_bh").sum()).alias("w_spot"))
            .sort("stage_id")
            .collect(engine="streaming")
        )
    except (pl.exceptions.ColumnNotFoundError, KeyError, pl.exceptions.SchemaError):
        return collapsible_section(
            "Spot Price by Bus",
            "<p>No spot price data.</p>",
            section_id="v2-costs-spot-price-section",
            default_collapsed=True,
        )

    if weighted.is_empty():
        return collapsible_section(
            "Spot Price by Bus",
            "<p>No spot price data.</p>",
            section_id="v2-costs-spot-price-section",
            default_collapsed=True,
        )

    stages = sorted(weighted["stage_id"].unique().to_list())
    xdates = stage_x_dates(stages, data.stage_dates)
    stage_to_x = dict(zip(stages, xdates))

    n_buses = len(bus_ids)
    n_cols = 2
    n_rows = (n_buses + n_cols - 1) // n_cols
    subplot_titles = [data.bus_names.get(bid, str(bid)) for bid in bus_ids]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=max(0.06, 0.35 / max(n_rows, 1)),
        horizontal_spacing=0.10,
    )

    for idx, bus_id in enumerate(bus_ids):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        color = BUS_COLORS[idx % len(BUS_COLORS)]
        bus_name = data.bus_names.get(bus_id, str(bus_id))

        sub_pl = weighted.filter(pl.col("bus_id") == bus_id).select(
            ["scenario_id", "stage_id", "w_spot"]
        )
        if sub_pl.is_empty():
            continue
        sub = sub_pl.to_pandas()
        pct = compute_percentiles(sub, ["stage_id"], "w_spot")
        if pct.empty:
            continue
        pct["_x"] = pct["stage_id"].map(stage_to_x)
        add_mean_p50_band(fig, pct, "_x", bus_name, color, row=row, col=col)

    fig.update_layout(
        height=350 * n_rows + 60,
        legend=_LEGEND,
        margin=dict(l=60, r=30, t=80, b=50),
        hovermode="x unified",
    )
    for ax in fig.layout:
        if str(ax).startswith("yaxis"):
            fig.layout[ax].title = "R$/MWh"  # type: ignore[index]
    all_stages = sorted(data.stage_dates)
    apply_stage_date_axis(
        fig,
        stage_x_dates(all_stages, data.stage_dates),
        stage_x_labels(all_stages, data.stage_labels),
    )

    chart_html = make_chart_card(
        fig,
        "Spot Price by Bus (block-hours weighted, p10/mean/p50/p90)",
        "v2-costs-spot-price-chart",
        height=350 * n_rows + 60,
    )
    return collapsible_section(
        "Spot Price by Bus",
        chart_grid([chart_html], single=True),
        section_id="v2-costs-spot-price-section",
        default_collapsed=True,
    )


# ---------------------------------------------------------------------------
# Section G — Violation Costs (conditional)
# ---------------------------------------------------------------------------


def _chart_violation_timeline(data: DashboardData) -> go.Figure | None:
    """Line chart of violation cost per stage, one line per nonzero category.

    Computes the mean undiscounted violation cost per stage across scenarios
    (summing blocks within each scenario/stage first), then plots one
    :class:`~plotly.graph_objects.Scatter` trace per violation category that
    has a nonzero total.

    Args:
        data: Dashboard data instance.

    Returns:
        A :class:`~plotly.graph_objects.Figure` with one trace per nonzero
        violation category, or ``None`` when no data is available or all
        violation values are zero.
    """
    if data.costs.empty:
        return None

    if "stage_id" not in data.costs.columns:
        return None

    viol_cols = [
        c
        for c in data.costs.columns
        if c in _VIOLATION_COLS or c.endswith("_violation_cost")
    ]
    if not viol_cols:
        return None

    # Sum blocks per (scenario_id, stage_id), then mean across scenarios
    group_keys = [k for k in ("scenario_id", "stage_id") if k in data.costs.columns]
    by_stage = (
        data.costs.groupby(group_keys)[viol_cols]
        .sum()
        .reset_index()
        .groupby("stage_id")[viol_cols]
        .mean()
        .reset_index()
    )

    stages = sorted(by_stage["stage_id"].tolist())
    xdates = stage_x_dates(stages, data.stage_dates)

    fig = go.Figure()
    has_trace = False
    for col in viol_cols:
        total = by_stage[col].sum()
        if total <= 0:
            continue
        label = col.replace("_cost", "").replace("_", " ").title()
        fig.add_trace(
            go.Scatter(
                x=xdates,
                y=by_stage[col].tolist(),
                name=label,
                mode="lines",
            )
        )
        has_trace = True

    if not has_trace:
        return None

    fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="Violation Cost (R$)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=_MARGIN,
    )
    all_stages = sorted(data.stage_dates)
    apply_stage_date_axis(
        fig,
        stage_x_dates(all_stages, data.stage_dates),
        stage_x_labels(all_stages, data.stage_labels),
    )
    return fig


def _render_violations(data: DashboardData) -> str:
    """Render a horizontal bar chart of violation costs per category.

    Only rendered when at least one known violation cost column has non-zero
    values.  When all violation costs are zero, returns a ``<p>`` fallback.

    Violation cost columns are identified by intersecting ``data.costs.columns``
    with the known set :data:`_VIOLATION_COLS` plus any column whose name ends
    in ``"_violation_cost"``.

    Args:
        data: Dashboard data instance.

    Returns:
        HTML string: a ``collapsible_section`` wrapping the chart (or a
        ``<p>`` fallback) when data is missing or all violations are zero.
    """
    if data.costs.empty:
        return collapsible_section(
            "Violation Costs",
            "<p>No violation costs recorded.</p>",
            section_id="v2-costs-violations-section",
            default_collapsed=True,
        )

    # Identify violation columns present in the DataFrame
    viol_cols = [
        c
        for c in data.costs.columns
        if c in _VIOLATION_COLS or c.endswith("_violation_cost")
    ]

    if not viol_cols:
        return collapsible_section(
            "Violation Costs",
            "<p>No violation costs recorded.</p>",
            section_id="v2-costs-violations-section",
            default_collapsed=True,
        )

    # Check whether any violation column has non-zero values
    subset = data.costs[viol_cols]
    if subset.values.sum() == 0:
        return collapsible_section(
            "Violation Costs",
            "<p>No violation costs recorded.</p>",
            section_id="v2-costs-violations-section",
            default_collapsed=True,
        )

    # Compute mean total violation cost per category across all stages/blocks,
    # averaged across scenarios.
    total_per_scen: dict[str, float] = {}
    for col in viol_cols:
        if "scenario_id" in data.costs.columns:
            per_scen = data.costs.groupby("scenario_id")[col].sum()
            total_per_scen[col] = float(per_scen.mean())
        else:
            total_per_scen[col] = float(data.costs[col].sum())

    # Remove zero-valued categories
    total_per_scen = {k: v for k, v in total_per_scen.items() if v > 0}
    if not total_per_scen:
        return collapsible_section(
            "Violation Costs",
            "<p>No violation costs recorded.</p>",
            section_id="v2-costs-violations-section",
            default_collapsed=True,
        )

    # Sort descending
    sorted_items = sorted(total_per_scen.items(), key=lambda kv: kv[1], reverse=True)

    # Map column names to group labels and colors
    _col_to_group: dict[str, str] = {}
    for group_name, cols in COST_GROUPS.items():
        for col in cols:
            _col_to_group[col] = group_name

    categories = [
        _col_to_group.get(k, k.replace("_cost", "").replace("_", " ").title())
        for k, _ in sorted_items
    ]
    values = [v for _, v in sorted_items]
    colors = [COST_GROUP_COLORS.get(cat, "#6B7280") for cat in categories]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=categories,
            orientation="h",
            marker_color=colors,
        )
    )
    apply_standard_layout(
        fig,
        xaxis_title="Mean Cost (R$)",
        yaxis=dict(autorange="reversed"),
    )

    chart_html = make_chart_card(
        fig,
        "Violation Costs by Category (mean across scenarios)",
        "v2-costs-violations-chart",
        height=max(300, 40 * len(categories) + 120),
    )

    timeline_fig = _chart_violation_timeline(data)
    if timeline_fig is not None:
        timeline_html = make_chart_card(
            timeline_fig,
            "Violation Cost Timeline",
            "v2-costs-violations-timeline",
            height=max(300, 40 * len(categories) + 120),
        )
        content = chart_grid([chart_html, timeline_html])
    else:
        content = chart_grid([chart_html], single=True)

    return collapsible_section(
        "Violation Costs",
        content,
        section_id="v2-costs-violations-section",
        default_collapsed=True,
    )


# ---------------------------------------------------------------------------
# Public render
# ---------------------------------------------------------------------------


def render(data: DashboardData) -> str:
    """Return the full HTML string for the v2 Costs tab content area.

    Renders five sections in order:
    - NPV Cost Analysis (metrics + bar chart + table).
    - Cost Composition by Stage (stacked area, undiscounted) — Section D.
    - Cost Category Trends (line + bands per group) — Section E.
    - Spot Price by Bus (faceted subplots, block-hours weighted) — Section F.
    - Violation Costs (conditional horizontal bar) — Section G.

    Args:
        data: Dashboard data instance.

    Returns:
        HTML string for the entire tab content.
    """
    return (
        section_title("NPV Cost Analysis")
        + _render_npv_section(data)
        + _build_composition_section(data)
        + _render_spot_price(data)
        + _render_violations(data)
    )
