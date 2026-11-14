"""Shared chart builder helpers for the Novomodelo dashboard.

Provides reusable functions for building Plotly traces that follow the v2
dashboard convention: every line chart shows a mean solid line, a p50/median
dashed line, and a p10-p90 shaded band.  HTML card wrapping is also provided
here so that all tab modules share a single, consistent implementation.

Also provides NPV cost computation and cost category grouping helpers used
by the Overview and Costs tabs.
"""

from __future__ import annotations

import math
from typing import Literal

import pandas as pd
import plotly.graph_objects as go
import polars as pl

from novomodelo_bridge.novomodelo.cost_categories import (
    AGGREGATE_COST_COLUMNS,
    COST_CATEGORIES,
    COST_CATEGORY_OF_COMPONENT,
    COST_COMPONENT_BY_LABEL,
    COST_KEY_COLUMNS,
)
from novomodelo_bridge.ui.html.document import wrap_chart
from novomodelo_bridge.ui.html.plotly import (
    LEGEND_DEFAULTS as _LEGEND,
)
from novomodelo_bridge.ui.html.plotly import (
    MARGIN_DEFAULTS as _MARGIN,
)
from novomodelo_bridge.ui.html.plotly import (
    add_mean_p50_band as add_mean_p50_band,
)
from novomodelo_bridge.ui.html.plotly import (
    fig_to_html,
)
from novomodelo_bridge.ui.theme import (
    COST_CATEGORY_COLORS,
    COST_COMPONENT_COLORS,
    COST_OTHER_COLOR,
)


def compute_percentiles(
    df: pd.DataFrame | pl.DataFrame,
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
    if isinstance(df, pl.DataFrame):
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
    :func:`~novomodelo_bridge.ui.html.plotly.fig_to_html`, sets the figure
    height, and embeds the result inside the ``.chart-card`` div with an
    expand button (via :func:`~novomodelo_bridge.ui.html.document.wrap_chart`).

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


#: A cost breakdown level: the taxonomy's categories or its components.
CostLevel = Literal["category", "component"]

#: The entry that sums every cost column the taxonomy does not classify.
OTHER_COSTS = "Other"

#: The cost columns behind each category and each component, in taxonomy order.
COST_GROUPS: dict[CostLevel, dict[str, tuple[str, ...]]] = {
    "category": {cat.label: cat.columns for cat in COST_CATEGORIES},
    "component": {
        label: comp.columns for label, comp in COST_COMPONENT_BY_LABEL.items()
    },
}

_COST_COLORS: dict[CostLevel, dict[str, str]] = {
    "category": COST_CATEGORY_COLORS,
    "component": COST_COMPONENT_COLORS,
}

_NON_COST_COLS: frozenset[str] = COST_KEY_COLUMNS | AGGREGATE_COST_COLUMNS


def cost_columns(costs_df: pd.DataFrame) -> list[str]:
    """The cost component columns of *costs_df*: every column but the row keys
    and the aggregates, so a column the taxonomy does not know still counts,
    under :data:`OTHER_COSTS`.
    """
    return [c for c in costs_df.columns if c not in _NON_COST_COLS]


def cost_color(level: CostLevel, label: str) -> str:
    """The colour of the *level* entry *label*; unclassified costs share one."""
    return _COST_COLORS[level].get(label, COST_OTHER_COLOR)


def compute_npv_costs(costs_df: pd.DataFrame) -> pd.DataFrame:
    """Discount every cost component column to present value.

    Multiplies each component by the row's ``discount_factor``, the cumulative
    factor ``D_t`` Novomodelo writes in ``simulation/costs/`` (``D_0 = 1``); the
    components themselves are undiscounted stage-*t* money. The aggregates and
    row keys are left untouched.

    Returns:
        A new :class:`pandas.DataFrame`; the input is never mutated. A copy of
        *costs_df* when it is empty or has no ``discount_factor`` column
        (undiscounted).
    """
    result = costs_df.copy()
    if costs_df.empty or "discount_factor" not in result.columns:
        return result

    factors = result["discount_factor"].astype(float)
    for col in cost_columns(result):
        result[col] = result[col] * factors
    return result


def group_costs(
    costs_df: pd.DataFrame,
    columns: list[str],
    level: CostLevel,
) -> pd.DataFrame:
    """Replace the cost *columns* of *costs_df* by one column per *level* entry.

    Every entry of :data:`COST_GROUPS` gets a column (0.0 when none of its
    columns is present), and the *columns* no entry claims sum into
    :data:`OTHER_COSTS`. Other columns are kept; the input is never mutated.
    """
    if costs_df.empty:
        return costs_df.copy()

    result = costs_df.copy()
    assigned: set[str] = set()
    for label, group_columns in COST_GROUPS[level].items():
        present = [c for c in group_columns if c in columns]
        assigned.update(present)
        result[label] = result[present].sum(axis=1) if present else 0.0

    other = [c for c in columns if c not in assigned]
    result[OTHER_COSTS] = result[other].sum(axis=1) if other else 0.0
    return result.drop(columns=[c for c in columns if c in result.columns])


def compute_cost_summary(costs_df: pd.DataFrame, level: CostLevel) -> pd.DataFrame:
    """The NPV of each *level* entry, with its spread across scenarios.

    Discounts the components, sums each scenario's over the stages, groups them
    by *level*, then takes the mean, standard deviation, p10 and p90 across
    scenarios, and each entry's share of the total mean.

    Returns:
        Columns ``["group", "mean", "std", "p10", "p90", "pct"]``, one row per
        entry (:data:`OTHER_COSTS` included), sorted by descending ``mean``;
        empty with those columns when *costs_df* is empty.
    """
    summary_cols = ["group", "mean", "std", "p10", "p90", "pct"]
    if costs_df.empty:
        return pd.DataFrame(columns=summary_cols)

    columns = cost_columns(costs_df)
    discounted = compute_npv_costs(costs_df)
    per_scenario = (
        discounted.groupby("scenario_id")[columns].sum().reset_index()
        if "scenario_id" in discounted.columns
        else discounted[columns]
    )
    grouped = group_costs(per_scenario, columns, level)
    group_cols = [*COST_GROUPS[level], OTHER_COSTS]

    agg = grouped[group_cols].agg(
        ["mean", "std", lambda q: q.quantile(0.1), lambda q: q.quantile(0.9)]
    )
    agg.index = ["mean", "std", "p10", "p90"]
    agg = agg.T.reset_index().rename(columns={"index": "group"})

    total_mean = agg["mean"].sum()
    agg["pct"] = (agg["mean"] / total_mean * 100.0) if total_mean != 0.0 else 0.0

    agg = agg.sort_values("mean", ascending=False).reset_index(drop=True)
    return agg[summary_cols]


def chart_cost_bar(summary_df: pd.DataFrame, level: CostLevel) -> go.Figure:
    """Bar chart of the NPV of each *level* entry, with p10–p90 error bars.

    Bars follow *summary_df*'s order (descending mean, from
    :func:`compute_cost_summary`) and skip zero-mean entries, unless every
    entry is zero.
    """
    nz = summary_df[summary_df["mean"] > 0]
    if nz.empty:
        nz = summary_df.head(1)

    groups = [str(g) for g in nz["group"]]
    means = [float(m) for m in nz["mean"]]
    err_plus: list[float] = []
    err_minus: list[float] = []
    has_errors = False
    for mean_val, p10, p90 in zip(means, nz["p10"], nz["p90"], strict=True):
        if math.isnan(p10) or math.isnan(p90):
            err_plus.append(0.0)
            err_minus.append(0.0)
            continue
        err_plus.append(max(0.0, float(p90) - mean_val))
        err_minus.append(max(0.0, mean_val - float(p10)))
        has_errors = True

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=groups,
            y=means,
            marker_color=[cost_color(level, g) for g in groups],
            error_y=(
                dict(type="data", array=err_plus, arrayminus=err_minus, visible=True)
                if has_errors
                else None
            ),
            showlegend=False,
            hovertemplate="%{x}: R$ %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis_title="NPV Cost (R$)",
        xaxis_tickangle=-35,
        margin=_MARGIN,
    )
    return fig


def build_cost_table(summary_df: pd.DataFrame, level: CostLevel) -> str:
    """The cost summary as a ``<table class="data-table">``.

    A component table also names each component's category.

    Args:
        summary_df: A :func:`compute_cost_summary` frame of the same *level*.
        level: The breakdown level *summary_df* was computed at.

    Returns:
        The ``<table>`` HTML, or a ``<p>`` fallback when *summary_df* is empty.
    """
    if summary_df.empty:
        return "<p>No cost data available.</p>"

    by_component = level == "component"
    headers = (("Component", "Category") if by_component else ("Category",)) + (
        "Mean (R$)",
        "Std (R$)",
        "P10 (R$)",
        "P90 (R$)",
        "% of Total",
    )
    header_cells = "".join(f"<th>{h}</th>" for h in headers)

    def _label_cells(group: str) -> str:
        if not by_component:
            return f"<td>{group}</td>"
        category = COST_CATEGORY_OF_COMPONENT.get(group, OTHER_COSTS)
        return f"<td>{group}</td><td>{category}</td>"

    rows = [
        f"<tr>"
        f"{_label_cells(str(row['group']))}"
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
