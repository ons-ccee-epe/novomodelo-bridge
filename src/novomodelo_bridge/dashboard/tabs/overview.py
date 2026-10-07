"""v2 Overview tab — executive summary landing page for the Novomodelo dashboard.

Displays run identity, run status, key metric cards, cost breakdown
(horizontal stacked bar + summary table), and two quick-look mini charts
(training convergence, generation mix).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import polars as pl

from novomodelo_bridge.dashboard.chart_helpers import (
    COST_GROUP_COLORS,
    build_cost_table,
    chart_cost_bar,
    compute_cost_summary,
    make_chart_card,
)
from novomodelo_bridge.dashboard.data import _stage_avg_mw
from novomodelo_bridge.novomodelo.readers import novomodelo_software_version
from novomodelo_bridge.ui.html.document import (
    chart_grid,
    metric_card,
    metrics_grid,
    section_title,
    wrap_chart,
)
from novomodelo_bridge.ui.html.plotly import (
    MARGIN_DEFAULTS,
    apply_stage_date_axis,
    apply_standard_layout,
    stage_x_dates,
    stage_x_labels,
)
from novomodelo_bridge.ui.theme import COLORS, GENERATION_COLORS

if TYPE_CHECKING:
    from novomodelo_bridge.dashboard.data import DashboardData

TAB_ID = "tab-overview"
TAB_LABEL = "Overview"
TAB_ORDER = 0


def _format_duration(elapsed_seconds: object) -> str:
    """Format elapsed seconds as a human-readable duration string.

    Args:
        elapsed_seconds: Numeric seconds value, or ``None``.

    Returns:
        A string like ``"4h 12min"``, or ``"N/A"`` when *elapsed_seconds* is
        ``None`` or cannot be converted to an integer.
    """
    if elapsed_seconds is None:
        return "N/A"
    try:
        total = int(elapsed_seconds)
    except (ValueError, TypeError):
        return "N/A"
    hours = total // 3600
    mins = (total % 3600) // 60
    return f"{hours}h {mins}min"


def _run_identity_strip(data: DashboardData) -> str:
    """Build the run identity row HTML."""
    version = novomodelo_software_version(data.training_metadata) or "N/A"
    discount_pct = data.discount_rate * 100.0

    run_date = data.training_metadata.get("started_at", "N/A")
    elapsed = data.training_metadata.get("duration_seconds")
    duration_str = _format_duration(elapsed)

    n_hydros = len(data.hydro_meta)
    n_thermals = len(data.thermal_meta)
    n_buses = len(data.non_fictitious_bus_ids)
    n_lines = len(data.line_meta)

    return (
        '<div class="run-identity-strip" style="'
        "display:flex;gap:2rem;flex-wrap:wrap;padding:0.75rem 1rem;"
        "background:var(--surface,#F8FAFC);border-radius:6px;"
        'margin-bottom:1rem;font-size:0.875rem;">'
        f"<span><strong>Case:</strong> {data.case_name}</span>"
        f"<span><strong>Scenarios:</strong> {data.n_scenarios}</span>"
        f"<span><strong>Stages:</strong> {data.n_stages}</span>"
        f"<span><strong>Discount Rate:</strong> {discount_pct:.1f}%</span>"
        f"<span><strong>Solver Version:</strong> {version}</span>"
        f"<span><strong>Run Date:</strong> {run_date}</span>"
        f"<span><strong>Duration:</strong> {duration_str}</span>"
        f"<span><strong>Hydros:</strong> {n_hydros}</span>"
        f"<span><strong>Thermals:</strong> {n_thermals}</span>"
        f"<span><strong>Buses:</strong> {n_buses}</span>"
        f"<span><strong>Lines:</strong> {n_lines}</span>"
        "</div>"
    )


def _run_status_strip(data: DashboardData) -> str:
    """Build the run status row HTML."""
    if not data.training_metadata:
        return (
            '<div class="run-status-strip" style="'
            "padding:0.5rem 1rem;background:#FEF3C7;border-radius:6px;"
            'margin-bottom:1rem;font-size:0.875rem;">'
            "No training metadata available"
            "</div>"
        )
    termination = data.training_metadata.get("convergence", {}).get(
        "termination_reason", "N/A"
    )
    n_iterations = len(data.conv)
    policy_states = data.policy_metadata.get("state_dimension", "N/A")

    if not data.conv.empty:
        lb_str = f"{float(data.conv.iloc[-1]['lower_bound']):,.2f}"
        active_cuts = int(data.conv.iloc[-1]["cuts_active"])
        total_cuts = int(data.conv["cuts_added"].sum())
        cuts_fields = (
            f"<span><strong>Lower bound:</strong> {lb_str}</span>"
            f"<span><strong>Total cuts:</strong> {total_cuts}</span>"
            f"<span><strong>Active cuts:</strong> {active_cuts}</span>"
        )
    else:
        cuts_fields = ""

    return (
        '<div class="run-status-strip" style="'
        "display:flex;gap:2rem;flex-wrap:wrap;padding:0.5rem 1rem;"
        "background:var(--surface,#F0FDF4);border-radius:6px;"
        'margin-bottom:1rem;font-size:0.875rem;">'
        f"<span><strong>Termination:</strong> {termination}</span>"
        f"<span><strong>Training Iterations:</strong> {n_iterations}</span>"
        + cuts_fields
        + f"<span><strong>Policy states:</strong> {policy_states}</span>"
        "</div>"
    )


def _compute_gen_gwh(lf: pl.LazyFrame) -> float:
    """Extract mean total generation GWh from a simulation LazyFrame.

    Groups by scenario_id, sums generation_mwh, then averages across
    scenarios.  Returns 0.0 when the LazyFrame is empty or the column
    is absent.
    """
    try:
        result = (
            lf.group_by("scenario_id")
            .agg(pl.col("generation_mwh").sum())
            .select(pl.col("generation_mwh").mean())
            .collect(engine="streaming")
        )
        if result.height == 0:
            return 0.0
        value = result["generation_mwh"][0]
        return float(value) / 1e3 if value is not None else 0.0
    except (ValueError, TypeError, KeyError):
        return 0.0


def _chart_training_mini(conv: pd.DataFrame) -> go.Figure | None:
    """Build a compact convergence mini-chart with lower bound, upper bound,
    and a ±1 std confidence band around the upper bound.

    Args:
        conv: Convergence DataFrame with columns ``iteration``,
            ``lower_bound``, ``upper_bound_mean``, and optionally
            ``upper_bound_std``.

    Returns:
        A :class:`plotly.graph_objects.Figure`, or ``None`` when *conv* is
        empty or the required columns are absent.
    """
    required = {"iteration", "lower_bound", "upper_bound_mean"}
    if conv.empty or not required.issubset(conv.columns):
        return None

    iters = conv["iteration"]
    ub_mean = conv["upper_bound_mean"]
    ub_color = COLORS["upper_bound"]

    fig = go.Figure()

    ub_legend_group = "upper_bound"

    # Upper bound confidence band (±1 std), toggles with the UB line
    if "upper_bound_std" in conv.columns:
        ub_std = conv["upper_bound_std"]
        ub_upper = ub_mean + ub_std
        ub_lower = ub_mean - ub_std
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=ub_upper,
                mode="lines",
                line={"width": 0},
                legendgroup=ub_legend_group,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=ub_lower,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=f"rgba({_hex_rgb(ub_color)},0.15)",
                legendgroup=ub_legend_group,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=iters,
            y=conv["lower_bound"],
            name="Lower Bound",
            mode="lines",
            line={"color": COLORS["lower_bound"], "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=ub_mean,
            name="Upper Bound",
            mode="lines",
            line={"color": ub_color, "width": 2},
            legendgroup=ub_legend_group,
        )
    )
    apply_standard_layout(
        fig,
        yaxis_title="Cost",
    )
    return fig


def _hex_rgb(hex_color: str) -> str:
    """Convert ``#RRGGBB`` to ``R,G,B`` for use in rgba() strings."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


def _chart_gen_mix(data: DashboardData) -> go.Figure | None:
    """Build a stacked area chart of mean generation MW per stage.

    Computes stage-average MW for hydro, thermal, and NCS using
    :func:`~novomodelo_bridge.dashboard.data._stage_avg_mw` and renders them
    as stacked area traces.

    Args:
        data: Full :class:`~novomodelo_bridge.dashboard.data.DashboardData` instance.

    Returns:
        A :class:`plotly.graph_objects.Figure`, or ``None`` when all three
        generation sources yield no data.
    """
    if not data.stage_hours:
        return None

    sources: list[tuple[str, pl.LazyFrame, str]] = [
        ("Hydro", data.hydros_lf, GENERATION_COLORS["hydro"]),
        ("Thermal", data.thermals_lf, GENERATION_COLORS["thermal"]),
        ("NCS", data.ncs_lf, GENERATION_COLORS["ncs"]),
    ]

    has_data = False
    fig = go.Figure()

    for label, lf, color in sources:
        try:
            stage_mw = _stage_avg_mw(lf, "generation_mwh", data.stage_hours)
        except (ValueError, TypeError, KeyError):
            continue
        if not stage_mw:
            continue

        stages = sorted(stage_mw.keys())
        xdates = stage_x_dates(stages, data.stage_dates)
        yvals = [stage_mw[s] for s in stages]
        fig.add_trace(
            go.Scatter(
                x=xdates,
                y=yvals,
                name=label,
                stackgroup="gen",
                mode="lines",
                line={"color": color},
            )
        )
        has_data = True

    if not has_data:
        return None

    fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="Average MW",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=MARGIN_DEFAULTS,
    )
    all_stages = sorted(data.stage_dates)
    apply_stage_date_axis(
        fig,
        stage_x_dates(all_stages, data.stage_dates),
        stage_x_labels(all_stages, data.stage_labels),
    )
    return fig


def _render_metric_cards(data: DashboardData) -> str:
    """Render the four key-metric cards."""
    # Expected cost (NPV)
    if data.costs.empty:
        cost_value_str = "N/A"
    else:
        summary = compute_cost_summary(data.costs, data.discount_rate)
        if summary.empty:
            cost_value_str = "N/A"
        else:
            total_npv = float(summary["mean"].sum())
            cost_value_str = f"{total_npv:,.0f}"

    hydro_gwh = _compute_gen_gwh(data.hydros_lf)
    thermal_gwh = _compute_gen_gwh(data.thermals_lf)
    ncs_gwh = _compute_gen_gwh(data.ncs_lf)

    cards = [
        metric_card(
            cost_value_str,
            "Expected Cost (NPV)",
            color=COST_GROUP_COLORS["Thermal"],
        ),
        metric_card(
            f"{hydro_gwh:,.0f} GWh",
            "Total Hydro GWh",
            color=GENERATION_COLORS["hydro"],
        ),
        metric_card(
            f"{thermal_gwh:,.0f} GWh",
            "Total Thermal GWh",
            color=GENERATION_COLORS["thermal"],
        ),
        metric_card(
            f"{ncs_gwh:,.0f} GWh",
            "Total NCS GWh",
            color=GENERATION_COLORS["ncs"],
        ),
    ]
    return section_title("Key Metrics") + metrics_grid(cards)


def _render_cost_breakdown(data: DashboardData) -> str:
    """Render the cost breakdown bar chart and summary table."""
    if data.costs.empty:
        return section_title("Cost Breakdown") + "<p>No cost data available.</p>"

    summary = compute_cost_summary(data.costs, data.discount_rate)
    summary = summary[summary["mean"] > 0]
    if summary.empty:
        return section_title("Cost Breakdown") + "<p>No cost data available.</p>"

    bar_fig = chart_cost_bar(summary)
    bar_html = make_chart_card(bar_fig, "NPV Cost by Group", "v2-cost-bar-chart")
    table_html = wrap_chart(build_cost_table(summary))
    return section_title("Cost Breakdown") + chart_grid([bar_html, table_html])


def _quick_look_cell(chart_html: str, deep_link_html: str) -> str:
    """Wrap a chart card and its deep-link in a single grid cell."""
    return f"<div>{chart_html}{deep_link_html}</div>"


def _render_quick_look_charts(data: DashboardData) -> str:
    """Render the quick-look mini charts."""
    link_style = 'style="font-size:0.8rem;margin-top:0.25rem;"'

    # Training Bounds mini
    training_fig = _chart_training_mini(data.conv)
    if training_fig is None:
        training_cell = wrap_chart("<p>No convergence data.</p>")
    else:
        chart = make_chart_card(
            training_fig,
            "Training Convergence",
            "v2-training-mini-chart",
            height=280,
        )
        link = (
            f"<p {link_style}>"
            '<a href="#" onclick="showTab(\'tab-training\','
            " document.querySelector('nav button:nth-child(2)'));"
            'return false;">Full training analysis \u2192</a>'
            "</p>"
        )
        training_cell = _quick_look_cell(chart, link)

    # Generation Mix mini
    gen_fig = _chart_gen_mix(data)
    if gen_fig is None:
        gen_cell = wrap_chart("<p>No generation data.</p>")
    else:
        chart = make_chart_card(
            gen_fig,
            "Generation Mix",
            "v2-gen-mix-chart",
            height=280,
        )
        link = (
            f"<p {link_style}>"
            '<a href="#" onclick="showTab(\'tab-energy-balance\','
            " document.querySelector('nav button:nth-child(4)'));"
            'return false;">Energy balance details \u2192</a>'
            "</p>"
        )
        gen_cell = _quick_look_cell(chart, link)

    return section_title("Quick Look") + chart_grid([training_cell, gen_cell])


def can_render(data: DashboardData) -> bool:
    """Overview tab renders only when simulation data is available."""
    return data.simulation_available


def render(data: DashboardData) -> str:
    """Return the full HTML string for the v2 Overview tab content area."""
    return (
        _run_identity_strip(data)
        + _run_status_strip(data)
        + _render_metric_cards(data)
        + _render_cost_breakdown(data)
        + _render_quick_look_charts(data)
    )
