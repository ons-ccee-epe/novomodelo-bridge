"""Shared chart builder helpers for the Cobre dashboard.

Provides reusable functions for building Plotly traces that follow the v2
dashboard convention: every line chart shows a mean solid line, a p50/median
dashed line, and a p10-p90 shaded band.  HTML card wrapping is also provided
here so that all tab modules share a single, consistent implementation.

Also provides NPV cost computation and cost category grouping helpers used
by the Overview and Costs tabs.
"""

from __future__ import annotations

import plotly.graph_objects as go

from cobre_bridge.cobre.cost_categories import (
    AGGREGATE_COST_COLUMNS,
    COST_PARTITION_COLUMNS,
)
from cobre_bridge.ui.html.document import wrap_chart
from cobre_bridge.ui.html.plotly import (
    LEGEND_DEFAULTS as _LEGEND,
)
from cobre_bridge.ui.html.plotly import (
    MARGIN_DEFAULTS as _MARGIN,
)
from cobre_bridge.ui.html.plotly import (
    add_mean_p50_band as add_mean_p50_band,
)
from cobre_bridge.ui.html.plotly import (
    fig_to_html,
)

try:
    import pandas as pd
except ImportError as _pd_err:  # pragma: no cover
    raise ImportError("pandas is required for chart_helpers") from _pd_err

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

#: Mapping from logical cost group name to known Cobre cost component columns.
#: The ``"Other"`` key is intentionally absent — it is computed dynamically in
#: :func:`group_costs` as all columns not claimed by the explicit groups.
COST_GROUPS: dict[str, list[str]] = {
    # Generation costs (anticipated = GNL forward-committed fuel, grouped here)
    "Thermal": ["thermal_cost", "anticipated_thermal_cost"],
    "Deficit": ["deficit_cost"],
    "Energy Excess": ["excess_cost"],
    "Spillage": ["spillage_cost"],
    "Turbined Reg.": ["turbined_cost"],
    "NCS Curtailment": ["curtailment_cost"],
    # Operational costs
    "Exchange": ["exchange_cost"],
    "Contract": ["contract_cost"],
    "Pumping": ["pumping_cost"],
    "Inflow Penalty": ["inflow_penalty_cost"],
    # Hydro violation costs (each broken out for visibility)
    # NOTE: hydro_violation_cost is an aggregate of the 6 costs below — excluded
    # via _NON_COST_COLS to avoid double-counting.
    "Outflow Min": ["outflow_violation_below_cost"],
    "Outflow Max": ["outflow_violation_above_cost"],
    "Turbining Bounds": ["turbined_violation_cost"],
    "Generation Bounds": ["generation_violation_cost"],
    "Storage Bounds": ["storage_violation_cost"],
    "Filling Target": ["filling_target_cost"],
    "Evaporation": ["evaporation_violation_cost"],
    "Water Withdrawal": ["withdrawal_violation_cost"],
    # Generic constraints
    "Generic Constraints": ["generic_violation_cost"],
    # Catch-all
    "Other": [],  # filled dynamically by group_costs
}

COST_GROUP_COLORS: dict[str, str] = {
    # Generation — warm palette
    "Thermal": "#D97706",
    "Deficit": "#DC2626",
    "Energy Excess": "#F59E0B",
    "Spillage": "#2563EB",
    # Regularisation — cool/teal palette to distinguish from violations
    "Turbined Reg.": "#0EA5E9",
    "NCS Curtailment": "#059669",
    # Operational — cool/neutral tones
    "Exchange": "#7C3AED",
    "Contract": "#DB2777",
    "Pumping": "#0891B2",
    "Inflow Penalty": "#EA580C",
    # Hydro violations — red-orange gradient (penalty costs)
    "Outflow Min": "#E11D48",
    "Outflow Max": "#F43F5E",
    "Turbining Bounds": "#FB923C",
    "Generation Bounds": "#F97316",
    "Storage Bounds": "#C2410C",
    "Filling Target": "#A16207",
    "Evaporation": "#9333EA",
    "Water Withdrawal": "#0284C7",
    # Generic constraints
    "Generic Constraints": "#6D28D9",
    # Catch-all
    "Other": "#6B7280",
}


def compute_percentiles(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    percentiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Group *df* and compute mean plus named percentile columns.

    Accepts either a :class:`pandas.DataFrame` or a :class:`polars.DataFrame`.
    Polars input is converted to pandas internally before aggregation.

    Args:
        df: Source data.  Must contain all columns in *group_cols* and
            *value_col*.
        group_cols: Column names to group by (e.g. ``["stage_id"]``).
        value_col: Name of the numeric column to aggregate.
        percentiles: Quantile levels.  Each value ``p`` produces a column
            named ``p{int(p * 100)}``, e.g. ``0.1`` -> ``"p10"``.

    Returns:
        A :class:`pandas.DataFrame` with columns
        ``[*group_cols, "mean", "p10", "p50", "p90"]`` (or whichever
        percentile names correspond to *percentiles*).  One row per group.
        Returns an empty :class:`pandas.DataFrame` with the expected columns
        when *df* is empty.
    """
    # Convert polars to pandas if necessary
    if pl is not None and isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    pct_col_names = [f"p{int(p * 100)}" for p in percentiles]
    expected_cols = group_cols + ["mean"] + pct_col_names

    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    grouped = df.groupby(group_cols, sort=True)[value_col]
    result = grouped.mean().rename("mean").to_frame()
    for p, col_name in zip(percentiles, pct_col_names):
        result[col_name] = grouped.quantile(p)
    return result.reset_index()


def stage_hours_weighted_mean(
    lf: pl.LazyFrame | pl.DataFrame,
    value_col: str,
    group_cols: list[str],
    stage_hours: dict[int, float] | pl.DataFrame,
) -> pl.DataFrame:
    """Collapse a per-stage rate across the horizon into one value per group.

    Weight a per-stage rate by stage hours; a bare ``.mean()`` gives a
    648-hour stage the same weight as a 168-hour stage.

    Args:
        lf: (Lazy)Frame carrying ``stage_id``, *value_col*, and every column
            in *group_cols*.
        value_col: Name of the per-stage rate column to collapse across
            stages (e.g. MW, m3/s) — never an already-summed energy column.
        group_cols: Columns identifying the groups the horizon is reduced to
            (e.g. ``["scenario_id", "line_id"]``); must not include
            ``stage_id``.
        stage_hours: Either ``{stage_id: hours}`` or a DataFrame with
            ``stage_id`` and ``_hours`` columns (e.g. summed from a
            block-hours frame).

    Returns:
        A :class:`polars.DataFrame` with ``[*group_cols, value_col]``, one
        row per group, holding ``Σ(value·stage_hours) / Σ(stage_hours)``.
        Empty (with the expected schema) when *lf* has no rows or the total
        stage hours are zero.
    """
    if pl is None:  # pragma: no cover
        raise ImportError("polars is required for stage_hours_weighted_mean")

    hours_df = (
        pl.DataFrame(
            {
                "stage_id": list(stage_hours.keys()),
                "_hours": list(stage_hours.values()),
            }
        )
        if isinstance(stage_hours, dict)
        else stage_hours
    )
    expected_cols = [*group_cols, value_col]

    if hours_df.height == 0 or hours_df["_hours"].sum() == 0:
        return pl.DataFrame(schema={c: pl.Float64 for c in expected_cols})

    lazy = lf.lazy() if isinstance(lf, pl.DataFrame) else lf
    return (
        lazy.join(hours_df.lazy(), on="stage_id")
        .group_by(group_cols)
        .agg(
            (
                (pl.col(value_col) * pl.col("_hours")).sum() / pl.col("_hours").sum()
            ).alias(value_col)
        )
        .collect(engine="streaming")
    )


def make_chart_card(
    fig: go.Figure | None,
    title: str,
    chart_id: str,
    height: int = 380,
) -> str:
    """Wrap a Plotly figure in a standard ``.chart-card`` HTML fragment.

    Applies default layout (template, margins, legend position) from
    :func:`~cobre_bridge.ui.html.plotly.fig_to_html`, sets the figure
    height, and embeds the result inside the ``.chart-card`` div with an
    expand button (via :func:`~cobre_bridge.ui.html.document.wrap_chart`).

    The output does **not** include a ``<script src="plotly.js">`` tag —
    callers are responsible for including Plotly exactly once in the outer
    HTML shell.

    Args:
        fig: The Plotly figure to embed.  Raises :class:`ValueError` when
            ``None``.
        title: Chart title applied to *fig* via
            :meth:`~plotly.graph_objects.Figure.update_layout`.
        chart_id: HTML ``id`` attribute placed on the inner chart container
            div for JavaScript targeting.
        height: Figure height in pixels (default 380).

    Returns:
        An HTML string: ``<div class="chart-card">...</div>``.

    Raises:
        ValueError: When *fig* is ``None``.
    """
    if fig is None:
        raise ValueError("Cannot build a chart card: the figure is missing.")

    defaults: dict = dict(
        title=dict(text=title, font=dict(size=13), x=0.02, xanchor="left"),
        height=height,
        margin=_MARGIN,
        template="plotly_white",
    )
    # Only apply default legend if the figure hasn't set a custom one
    if fig.layout.legend is None or fig.layout.legend.y is None:
        defaults["legend"] = _LEGEND
    fig.update_layout(**defaults)

    inner_html = f'<div id="{chart_id}">{fig_to_html(fig)}</div>'
    return wrap_chart(inner_html)


#: Non-cost metadata columns that are never treated as cost components: the
#: hive-partition/time keys plus the derived/aggregate cost roll-ups (the latter
#: single-sourced with the comparator via cost_categories to avoid drift).
_NON_COST_COLS: frozenset[str] = COST_PARTITION_COLUMNS | AGGREGATE_COST_COLUMNS


def compute_npv_costs(
    costs_df: pd.DataFrame,
    discount_rate: float,  # noqa: ARG001 — retained for API compat; unused
    stage_start: int = 0,  # noqa: ARG001 — retained for API compat; unused
) -> pd.DataFrame:
    """Apply per-stage NPV discount factors to cost component columns.

    Uses the ``discount_factor`` column written by cobre in
    ``simulation/costs/`` (the cumulative discount factor ``D_t`` that
    maps undiscounted stage-*t* costs to present value at stage 0).
    Cobre's simulation extraction stores raw per-stage component costs
    (``thermal_cost``, ``deficit_cost``, …) in undiscounted units; this
    helper multiplies each component by ``D_t`` to produce present-value
    costs. The stage aggregates (``total_cost``, ``immediate_cost``,
    ``future_cost``) carry LP-objective semantics and are left untouched.

    Args:
        costs_df: DataFrame containing a ``discount_factor`` column plus
            one or more numeric cost component columns.
        discount_rate: Unused. Retained so existing call sites do not break
            while the helper transitions off ad-hoc rate-based discounting.
        stage_start: Unused. Cobre's ``discount_factor`` column is already
            anchored at ``D_0 = 1.0``.

    Returns:
        A new :class:`pandas.DataFrame` with discounted cost values.  Input is
        never mutated.  Returns a copy of *costs_df* when it is empty or
        when ``discount_factor`` is missing (degrades to undiscounted).
    """
    if costs_df.empty:
        return costs_df.copy()

    result = costs_df.copy()
    if "discount_factor" not in result.columns:
        return result

    cost_cols = [c for c in result.columns if c not in _NON_COST_COLS]
    if not cost_cols:
        return result

    factors = result["discount_factor"].astype(float)
    for col in cost_cols:
        result[col] = result[col] * factors

    return result


def group_costs(
    costs_df: pd.DataFrame,
    cost_columns: list[str],
) -> pd.DataFrame:
    """Aggregate individual cost component columns into logical groups.

    Groups are defined by :data:`COST_GROUPS`.  Any column present in
    *cost_columns* that is not claimed by a named group is placed in an
    ``"Other"`` group.  Missing columns (present in :data:`COST_GROUPS` but
    absent from *cost_columns*) are silently skipped.

    Args:
        costs_df: DataFrame containing the individual cost component columns
            listed in *cost_columns*.
        cost_columns: Complete list of cost column names in *costs_df*.  Used
            to determine which columns fall into ``"Other"``.

    Returns:
        A new :class:`pandas.DataFrame` where the individual component columns
        are replaced by one column per group.  Non-cost columns (``scenario_id``,
        ``stage_id``, ``block_id``) are preserved unchanged.  Input is never
        mutated.  Returns a copy of *costs_df* when it is empty.
    """
    if costs_df.empty:
        return costs_df.copy()

    result = costs_df.copy()
    assigned: set[str] = set()

    for group_name, components in COST_GROUPS.items():
        if group_name == "Other":
            continue
        present = [c for c in components if c in cost_columns]
        assigned.update(present)
        result[group_name] = result[present].sum(axis=1) if present else 0.0

    other_cols = [c for c in cost_columns if c not in assigned]
    result["Other"] = result[other_cols].sum(axis=1) if other_cols else 0.0

    cols_to_drop = [c for c in cost_columns if c in result.columns]
    return result.drop(columns=cols_to_drop)


def compute_cost_summary(
    costs_df: pd.DataFrame,
    discount_rate: float,
) -> pd.DataFrame:
    """Compute a grouped NPV cost summary with statistics across scenarios.

    Chains three operations:

    1. Apply NPV discount factors via :func:`compute_npv_costs`.
    2. Sum discounted costs across all stages per scenario.
    3. Group columns via :func:`group_costs`.
    4. Aggregate across scenarios: mean, std, p5, p10, p90, p95.
    5. Add a ``pct`` column (percentage of total mean cost).

    Args:
        costs_df: DataFrame with ``scenario_id``, ``stage_id``, and cost
            component columns.
        discount_rate: Per-stage discount rate (e.g. ``0.12`` for 12%).

    Returns:
        A :class:`pandas.DataFrame` with columns
        ``["group", "mean", "std", "p5", "p10", "p90", "p95", "pct"]``, one
        row per cost group, sorted descending by ``mean``.  Returns an empty
        DataFrame with those columns when *costs_df* is empty.
    """
    summary_cols = ["group", "mean", "std", "p5", "p10", "p90", "p95", "pct"]

    if costs_df.empty:
        return pd.DataFrame(columns=summary_cols)

    cost_columns = [c for c in costs_df.columns if c not in _NON_COST_COLS]

    discounted = compute_npv_costs(costs_df, discount_rate)

    meta_cols_present = [c for c in ("scenario_id",) if c in discounted.columns]
    per_scenario = (
        discounted.groupby(meta_cols_present)[cost_columns].sum()
        if meta_cols_present
        else discounted[cost_columns]
    )

    grouped = group_costs(per_scenario.reset_index(), cost_columns)
    group_cols = [
        c for c in grouped.columns if c not in _NON_COST_COLS and c != "scenario_id"
    ]

    agg = grouped[group_cols].agg(
        [
            "mean",
            "std",
            lambda q: q.quantile(0.05),
            lambda q: q.quantile(0.1),
            lambda q: q.quantile(0.9),
            lambda q: q.quantile(0.95),
        ]
    )
    agg.index = ["mean", "std", "p5", "p10", "p90", "p95"]  # type: ignore[assignment]
    agg = agg.T.reset_index().rename(columns={"index": "group"})

    total_mean = agg["mean"].sum()
    agg["pct"] = (agg["mean"] / total_mean * 100.0) if total_mean != 0.0 else 0.0

    agg = agg.sort_values("mean", ascending=False).reset_index(drop=True)
    return agg[summary_cols]


def chart_cost_bar(summary_df: pd.DataFrame) -> go.Figure:
    """Build a vertical bar chart of NPV cost by group with p5–p95 error bars.

    One bar per cost group, sorted descending by mean value.  Each bar has
    asymmetric error bars showing the p5–p95 range across scenarios.  Groups
    with zero mean are excluded.

    Args:
        summary_df: DataFrame with columns
            ``["group", "mean", "p5", "p95", ...]`` as returned by
            :func:`~cobre_bridge.dashboard.chart_helpers.compute_cost_summary`.

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    import math

    nz = summary_df[summary_df["mean"] > 0].copy()
    if nz.empty:
        nz = summary_df.head(1)

    groups: list[str] = []
    means: list[float] = []
    colors: list[str] = []
    err_plus: list[float] = []
    err_minus: list[float] = []
    has_errors = False

    for _, row in nz.iterrows():
        group = str(row["group"])
        mean_val = float(row["mean"])
        groups.append(group)
        means.append(mean_val)
        colors.append(COST_GROUP_COLORS.get(group, "#6B7280"))

        ep, em = 0.0, 0.0
        if "p5" in row.index and "p95" in row.index:
            p5 = float(row["p5"])
            p95 = float(row["p95"])
            if not (math.isnan(p5) or math.isnan(p95)):
                ep = p95 - mean_val
                em = mean_val - p5
                has_errors = True
        err_plus.append(ep)
        err_minus.append(em)

    error_y: dict | None = None
    if has_errors:
        error_y = dict(
            type="data",
            array=err_plus,
            arrayminus=err_minus,
            visible=True,
        )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=groups,
            y=means,
            marker_color=colors,
            error_y=error_y,
            showlegend=False,
        )
    )
    fig.update_layout(
        yaxis_title="NPV Cost",
        xaxis_tickangle=-35,
        margin=_MARGIN,
    )
    return fig


def build_cost_table(summary_df: pd.DataFrame) -> str:
    """Return a ``<table class="data-table">`` HTML string from a cost summary.

    Args:
        summary_df: DataFrame with columns
            ``["group", "mean", "std", "p10", "p90", "pct"]`` as returned
            by :func:`~cobre_bridge.dashboard.chart_helpers.compute_cost_summary`.

    Returns:
        An HTML string containing a complete ``<table>`` element, or a
        fallback ``<p>`` when *summary_df* is empty.
    """
    if summary_df.empty:
        return "<p>No cost data available.</p>"

    headers = ("Group", "Mean", "Std", "P10", "P90", "% of Total")
    header_cells = "".join(f"<th>{h}</th>" for h in headers)
    rows = [
        f"<tr>"
        f"<td>{row['group']}</td>"
        f"<td>{row['mean']:,.0f}</td>"
        f"<td>{row['std']:,.0f}</td>"
        f"<td>{row['p10']:,.0f}</td>"
        f"<td>{row['p90']:,.0f}</td>"
        f"<td>{row['pct']:.1f}%</td>"
        f"</tr>"
        for _, row in summary_df.iterrows()
    ]

    return (
        '<table class="data-table" style="width:100%;border-collapse:collapse;">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )
