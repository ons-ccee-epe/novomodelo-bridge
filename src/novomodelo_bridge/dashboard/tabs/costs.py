"""Costs tab — NPV breakdown and the evolution of simulation costs over the stages.

Sections, in order:
  - NPV Cost Analysis: four NPV cards, and a bar chart and table of the NPV of
    each cost component.
  - Cost Composition by Stage: stacked areas of the undiscounted cost, by
    category or by component.
  - Spot Price by Bus: faceted mean+p50+band, block-hours weighted.
  - Violation Costs: the components of the Violations category, undiscounted.

Categories and components are those of
:data:`novomodelo_bridge.novomodelo.cost_categories.COST_CATEGORIES`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from novomodelo_bridge.dashboard.chart_helpers import (
    COST_GROUPS,
    OTHER_COSTS,
    CostLevel,
    add_mean_p50_band,
    build_cost_table,
    chart_cost_bar,
    compute_cost_summary,
    compute_percentiles,
    cost_color,
    cost_columns,
    group_costs,
    make_chart_card,
)
from novomodelo_bridge.novomodelo.cost_categories import VIOLATIONS
from novomodelo_bridge.ui.html.document import (
    chart_grid,
    collapsible_section,
    json_for_script,
    metric_card,
    metrics_grid,
    section_title,
    wrap_chart,
)
from novomodelo_bridge.ui.html.plotly import (
    LEGEND_DEFAULTS as _LEGEND,
)
from novomodelo_bridge.ui.html.plotly import (
    MARGIN_DEFAULTS as _MARGIN,
)
from novomodelo_bridge.ui.html.plotly import (
    apply_stage_date_axis,
    apply_standard_layout,
    stage_date_xaxis,
    stage_x_dates,
    stage_x_labels,
)
from novomodelo_bridge.ui.theme import BUS_COLORS, COST_TOTAL_COLOR

if TYPE_CHECKING:
    from novomodelo_bridge.dashboard.data import DashboardData

TAB_ID = "tab-costs"
TAB_LABEL = "Costs"
TAB_ORDER = 40


def can_render(data: DashboardData) -> bool:
    """Costs tab renders only when simulation data is available."""
    return data.simulation_available


def _summary_means(summary: pd.DataFrame) -> dict[str, float]:
    """Each entry's mean NPV in a :func:`compute_cost_summary` frame."""
    return {str(g): float(m) for g, m in zip(summary["group"], summary["mean"])}


def _build_metrics_row(data: DashboardData) -> str:
    """The four NPV cards: expected cost, then thermal, deficit and violations.

    Thermal and Deficit are those components, Violations the whole category.
    """
    labels = (
        "Expected Cost (NPV)",
        "Thermal (NPV)",
        "Deficit (NPV)",
        "Violations (NPV)",
    )
    colors = (
        COST_TOTAL_COLOR,
        cost_color("component", "Thermal"),
        cost_color("component", "Deficit"),
        cost_color("category", VIOLATIONS.label),
    )
    if data.costs.empty:
        return metrics_grid(
            [
                metric_card("N/A", label, color=color)
                for label, color in zip(labels, colors, strict=True)
            ]
        )

    components = _summary_means(compute_cost_summary(data.costs, "component"))
    categories = _summary_means(compute_cost_summary(data.costs, "category"))
    values = (
        sum(components.values()),
        components["Thermal"],
        components["Deficit"],
        categories[VIOLATIONS.label],
    )
    return metrics_grid(
        [
            metric_card(f"R$ {value:,.0f}", label, color=color)
            for value, label, color in zip(values, labels, colors, strict=True)
        ]
    )


def _render_npv_section(data: DashboardData) -> str:
    """The NPV cards, then the bar chart and table of each component's NPV."""
    metrics_html = _build_metrics_row(data)

    summary = compute_cost_summary(data.costs, "component")
    summary = summary[summary["mean"] > 0]
    if summary.empty:
        no_data = wrap_chart("<p>No cost data available.</p>")
        return metrics_html + chart_grid([no_data, no_data])

    bar_fig = chart_cost_bar(summary, "component")
    bar_html = make_chart_card(bar_fig, "NPV Cost by Component", "v2-costs-bar-chart")
    table_html = wrap_chart(build_cost_table(summary, "component"))
    return metrics_html + chart_grid([bar_html, table_html])


# ---------------------------------------------------------------------------
# Cost Composition by Stage — stacked areas, by category or by component
# ---------------------------------------------------------------------------


def _stage_means(
    by_stage_scen: pd.DataFrame, columns: list[str], level: CostLevel
) -> dict[str, list[float]]:
    """Each *level* entry's mean cost per stage across scenarios, for the
    entries that are non-zero in some stage, in taxonomy order."""
    grouped = group_costs(by_stage_scen, columns, level)
    labels = [*COST_GROUPS[level], OTHER_COSTS]
    means = grouped.groupby("stage_id")[labels].mean().sort_index()
    return {
        label: [round(float(v), 2) for v in means[label]]
        for label in labels
        if (means[label] != 0.0).any()
    }


def _build_composition_data(data: DashboardData) -> dict | None:
    """Precompute both views of the cost composition chart.

    Returns:
        A dict with ``category`` and ``component`` (each entry's mean cost
        per stage), ``total`` (``mean``, ``p10`` and ``p90`` of the total cost
        across scenarios per stage), ``stages`` (x-axis dates), ``xaxis`` (the
        :func:`stage_date_xaxis` over them) and ``colors`` (per view, each
        entry's colour).
        ``None`` when there is no cost, or every cost is zero.
    """
    if data.costs.empty:
        return None
    columns = cost_columns(data.costs)
    if not columns:
        return None

    # Sum blocks per (scenario_id, stage_id)
    by_stage_scen: pd.DataFrame = (
        data.costs.groupby(["scenario_id", "stage_id"])[columns].sum().reset_index()
    )
    views: dict[CostLevel, dict[str, list[float]]] = {
        "category": _stage_means(by_stage_scen, columns, "category"),
        "component": _stage_means(by_stage_scen, columns, "component"),
    }
    if not views["component"]:
        return None

    stages = sorted(by_stage_scen["stage_id"].unique().tolist())
    by_stage_scen["_total"] = by_stage_scen[columns].sum(axis=1)
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
        key: [round(float(v), 2) for v in total_pct[key].tolist()]
        for key in ("mean", "p10", "p90")
    }

    x_dates = stage_x_dates(stages, data.stage_dates)
    return {
        **views,
        "total": total,
        "stages": x_dates,
        "xaxis": stage_date_xaxis(x_dates, stage_x_labels(stages, data.stage_labels)),
        "colors": {
            level: {label: cost_color(level, label) for label in entries}
            for level, entries in views.items()
        },
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
        "const COSTS_COMP_LEGEND = " + json_for_script(_LEGEND) + ";\n"
        r"""
function updateCostsComposition() {
  var sel = document.getElementById('costs-group-sel');
  var view = sel ? sel.value : 'category';
  var d = COSTS_COMP_DATA;
  var groups = d[view];
  var stages = d.stages;
  var colors = d.colors[view];
  // The total's P10-P90 band goes under the stack, which would otherwise
  // show through it in a tinted colour; its P10 edge is redrawn on top.
  var traces = [{
    x: stages, y: d.total.p10,
    name: 'Total P10', showlegend: false, mode: 'lines',
    line: {width: 0}, hoverinfo: 'skip', legendgroup: 'band'
  }, {
    x: stages, y: d.total.p90,
    name: 'Total P10\u2013P90', mode: 'lines',
    line: {width: 0}, fill: 'tonexty',
    fillcolor: 'rgba(55,65,81,0.15)', hoverinfo: 'skip', legendgroup: 'band'
  }];

  Object.keys(groups).forEach(function(name) {
    traces.push({
      x: stages, y: groups[name], name: name,
      stackgroup: 'costs', mode: 'lines',
      line: {color: colors[name] || '#6B7280', width: 0.5},
      fillcolor: colors[name] || '#6B7280',
      hovertemplate: 'R$ %{y:,.0f}<extra>' + name + '</extra>'
    });
  });

  traces.push({
    x: stages, y: d.total.p10,
    name: 'Total P10', showlegend: false, mode: 'lines',
    line: {color: '#374151', width: 1, dash: 'dot'},
    hoverinfo: 'skip', legendgroup: 'band'
  });
  traces.push({
    x: stages, y: d.total.mean,
    name: 'Total Mean', mode: 'lines',
    line: {color: '#374151', width: 2, dash: 'dash'},
    hovertemplate: 'R$ %{y:,.0f}<extra>Total Mean</extra>'
  });

  var layout = {
    title: {text: 'Cost Composition by Stage (undiscounted, mean across scenarios)', font: {size: 13}, x: 0.02},
    xaxis: Object.assign({title: 'Stage'}, d.xaxis),
    yaxis: {title: 'Cost (R$)'},
    legend: COSTS_COMP_LEGEND,
    margin: {l: 60, r: 20, t: 60, b: 10},
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
# Spot Price by Bus (faceted, block-hours weighted)
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
            fig.layout[ax].title = "R$/MWh"
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
# Violation Costs — the Violations category's components, undiscounted
# ---------------------------------------------------------------------------


def _violation_costs(costs: pd.DataFrame) -> pd.DataFrame | None:
    """Each Violations component's undiscounted cost per (scenario, stage).

    Returns:
        ``scenario_id``, ``stage_id`` and one column per component that has a
        column in *costs*, named as the component; ``None`` when *costs* has
        no such column or lacks the scenario and stage keys.
    """
    if costs.empty or not {"scenario_id", "stage_id"} <= set(costs.columns):
        return None
    components = {
        comp.label: [c for c in comp.columns if c in costs.columns]
        for comp in VIOLATIONS.components
    }
    components = {label: cols for label, cols in components.items() if cols}
    if not components:
        return None

    by_stage_scen = costs.groupby(["scenario_id", "stage_id"])[
        [c for cols in components.values() for c in cols]
    ].sum()
    return pd.DataFrame(
        {label: by_stage_scen[cols].sum(axis=1) for label, cols in components.items()}
    ).reset_index()


#: Height of one violation-timeline panel, in pixels.
_VIOLATION_PANEL_HEIGHT = 130


def _chart_violation_timeline(data: DashboardData) -> go.Figure | None:
    """One panel per Violations component with a positive total: its mean
    cost per stage across scenarios. The panels share the stage axis and each
    has its own cost scale; the bar chart beside them compares the totals.

    Returns:
        The figure, or ``None`` when no component has a positive cost.
    """
    per_stage_scen = _violation_costs(data.costs)
    if per_stage_scen is None:
        return None

    labels = [c for c in per_stage_scen.columns if c not in ("scenario_id", "stage_id")]
    by_stage = per_stage_scen.groupby("stage_id")[labels].mean().sort_index()
    shown = [label for label in labels if by_stage[label].sum() > 0]
    if not shown:
        return None
    xdates = stage_x_dates(by_stage.index.tolist(), data.stage_dates)

    fig = make_subplots(
        rows=len(shown),
        cols=1,
        shared_xaxes=True,
        subplot_titles=shown,
        vertical_spacing=min(0.12, 0.5 / len(shown)),
    )
    for row, label in enumerate(shown, start=1):
        fig.add_trace(
            go.Scatter(
                x=xdates,
                y=by_stage[label].tolist(),
                name=label,
                mode="lines",
                line=dict(color=cost_color("component", label), width=2),
                showlegend=False,
                hovertemplate="R$ %{y:,.0f}",
            ),
            row=row,
            col=1,
        )
    fig.update_yaxes(tickprefix="R$ ", tickformat="~s", rangemode="tozero")
    fig.update_layout(margin=_MARGIN)
    all_stages = sorted(data.stage_dates)
    apply_stage_date_axis(
        fig,
        stage_x_dates(all_stages, data.stage_dates),
        stage_x_labels(all_stages, data.stage_labels),
    )
    return fig


def _render_violations(data: DashboardData) -> str:
    """The Violation Costs section, collapsed: a bar chart of each Violations
    component's mean total cost across scenarios above its stage timeline, or
    a ``<p>`` fallback when no component has a positive cost.
    """
    per_stage_scen = _violation_costs(data.costs)
    totals: dict[str, float] = {}
    if per_stage_scen is not None:
        labels = [
            c for c in per_stage_scen.columns if c not in ("scenario_id", "stage_id")
        ]
        means = per_stage_scen.groupby("scenario_id")[labels].sum().mean()
        totals = {label: float(v) for label, v in means.items() if v > 0}
    if not totals:
        return collapsible_section(
            "Violation Costs",
            "<p>No violation costs recorded.</p>",
            section_id="v2-costs-violations-section",
            default_collapsed=True,
        )

    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    components = [label for label, _ in ordered]
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in ordered],
            y=components,
            orientation="h",
            marker_color=[cost_color("component", label) for label in components],
            hovertemplate="%{y}: R$ %{x:,.0f}<extra></extra>",
        )
    )
    apply_standard_layout(
        fig,
        xaxis_title="Cost (R$)",
        yaxis=dict(autorange="reversed"),
    )

    height = max(300, 40 * len(components) + 120)
    chart_html = make_chart_card(
        fig,
        "Violation Costs by Component (undiscounted, mean across scenarios)",
        "v2-costs-violations-chart",
        height=height,
    )

    content = chart_grid([chart_html], single=True)
    timeline_fig = _chart_violation_timeline(data)
    if timeline_fig is not None:
        content += chart_grid(
            [
                make_chart_card(
                    timeline_fig,
                    "Violation Costs by Stage (undiscounted, mean across scenarios)",
                    "v2-costs-violations-timeline",
                    height=120 + _VIOLATION_PANEL_HEIGHT * len(timeline_fig.data),
                )
            ],
            single=True,
        )

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
    """Return the full HTML string for the Costs tab content area."""
    return (
        section_title("NPV Cost Analysis")
        + _render_npv_section(data)
        + _build_composition_section(data)
        + _render_spot_price(data)
        + _render_violations(data)
    )
