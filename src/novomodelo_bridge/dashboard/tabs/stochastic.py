"""v2 Stochastic Model tab — inflow comparison, correlation, noise, and AR sections.

Displays six collapsible sections:
  A. System-Wide Inflow — hero chart of historical vs synthetic mean+std band.
  B. Inflow by Bus — 2x2 faceted subplots for the first 4 non-fictitious buses.
  C. Per-Hydro Explorer — dropdown selector showing one hydro at a time.
  D. Spatial Correlation — fitted vs empirical correlation heatmaps.
  E. Noise Diagnostics — noise histogram and stage-grouped box plot.
  F. AR Model Summary — AR order distribution and reduction reasons.

Only rendered when stochastic output is available (data.stochastic_available).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as ps
import polars as pl

from novomodelo_bridge.dashboard.chart_helpers import make_chart_card
from novomodelo_bridge.ui.html.document import (
    chart_grid,
    collapsible_section,
    escape_attr,
    escape_text,
    json_for_script,
    plant_explorer_table,
    wrap_chart,
)
from novomodelo_bridge.ui.html.js import PLANT_EXPLORER_JS
from novomodelo_bridge.ui.html.plotly import (
    LEGEND_DEFAULTS,
    MARGIN_DEFAULTS,
    apply_stage_date_axis,
    apply_standard_layout,
    stage_x_dates,
    stage_x_labels,
)
from novomodelo_bridge.ui.theme import COLORS, hex_to_rgba

if TYPE_CHECKING:
    from novomodelo_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-stochastic"
TAB_LABEL = "Stochastic Model"
TAB_ORDER = 10
REQUIRED_JS: list[str] = [PLANT_EXPLORER_JS]

_HIST_COLOR: str = COLORS["hydro"]  # #4A90B8 — historical traces
_SYNTH_COLOR: str = COLORS["thermal"]  # #F5A623 — synthetic traces

_NO_HIST = "<p>No historical inflow data.</p>"
_NO_DATA = "<p>No data.</p>"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _with_month(history: pd.DataFrame) -> pd.DataFrame:
    """Copy of *history* (one row per hydro and monthly ``start_date`` /
    ``end_date`` window) with each window's calendar ``month``."""
    df = history.copy()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["month"] = df["start_date"].dt.month
    return df


def _compute_historical_stats(data: DashboardData) -> pd.DataFrame:
    """Compute per-hydro per-stage mean and std from inflow_history.

    Historical inflow records predate the study horizon, so they cannot be
    mapped by absolute date.  Instead, each stage's calendar month is used
    to select matching historical records (e.g. stage 0 = March → all
    historical March values across all years).  This produces the seasonal
    mean and std that can be compared against the fitted synthetic stats.

    Args:
        data: Full DashboardData instance.

    Returns:
        DataFrame with columns ``hydro_id``, ``stage_id``, ``mean_m3s``,
        ``std_m3s``.  Returns an empty DataFrame with those columns when
        ``inflow_history`` is empty or ``stages_data`` has no stages.
    """
    empty = pd.DataFrame(columns=["hydro_id", "stage_id", "mean_m3s", "std_m3s"])

    hist = data.inflow_history
    if hist.empty:
        return empty

    stages = data.stages_data.get("stages", [])
    if not stages:
        return empty

    # Build month -> list of stage_ids mapping
    month_to_stages: dict[int, list[int]] = {}
    for s in stages:
        sd = s.get("start_date")
        if sd is None:
            continue
        month = pd.to_datetime(sd).month
        month_to_stages.setdefault(month, []).append(s["id"])

    if not month_to_stages:
        return empty

    df = _with_month(hist)

    # Compute historical mean/std per hydro per calendar month
    grp = df.groupby(["hydro_id", "month"])["value_m3s"]
    monthly = grp.mean().rename("mean_m3s").to_frame()
    monthly["std_m3s"] = grp.std()
    monthly = monthly.reset_index()

    # Expand each monthly stat to all stages that share that month
    rows: list[dict] = []
    for _, r in monthly.iterrows():
        for sid in month_to_stages.get(int(r["month"]), []):
            rows.append(
                {
                    "hydro_id": r["hydro_id"],
                    "stage_id": sid,
                    "mean_m3s": r["mean_m3s"],
                    "std_m3s": r["std_m3s"],
                }
            )

    if not rows:
        return empty

    result = pd.DataFrame(rows)
    return result[["hydro_id", "stage_id", "mean_m3s", "std_m3s"]]


def _compute_synthetic_stats(data: DashboardData) -> pd.DataFrame:
    """Compute per-hydro per-stage mean and std from simulation inflows.

    Uses ``incremental_inflow_m3s`` from the simulation hydro output, which
    contains the actual PAR-generated inflow realizations.  Mean and std are
    computed across scenarios for each ``(hydro_id, stage_id)`` pair.

    Returns:
        DataFrame with columns ``hydro_id``, ``stage_id``, ``mean_m3s``,
        ``std_m3s``.  Returns an empty DataFrame when the LazyFrame has
        no data or the column is absent.
    """
    empty = pd.DataFrame(columns=["hydro_id", "stage_id", "mean_m3s", "std_m3s"])
    if not data.simulation_available:
        return empty
    try:
        result = (
            data.hydros_lf.group_by(["hydro_id", "stage_id"])
            .agg(
                [
                    pl.col("incremental_inflow_m3s").mean().alias("mean_m3s"),
                    pl.col("incremental_inflow_m3s").std().alias("std_m3s"),
                ]
            )
            .sort(["hydro_id", "stage_id"])
            .collect(engine="streaming")
            .to_pandas()
        )
    except (ValueError, KeyError, pl.exceptions.PolarsError):
        return empty
    if result.empty:
        return empty
    return result[["hydro_id", "stage_id", "mean_m3s", "std_m3s"]]


def _aggregate_by_bus(
    stats_df: pd.DataFrame,
    hydro_bus_map: dict[int, int],
) -> dict[int, pd.DataFrame]:
    """Aggregate per-hydro stats into per-bus stats.

    For each bus: mean_m3s is the sum of all hydro means; std_m3s is the
    root-sum-of-squares of all hydro stds (treating NaN as 0).

    Args:
        stats_df: DataFrame with columns ``hydro_id``, ``stage_id``,
            ``mean_m3s``, ``std_m3s``.
        hydro_bus_map: Mapping ``hydro_id -> bus_id``.

    Returns:
        Dict mapping ``bus_id`` to a DataFrame with columns ``stage_id``,
        ``mean_m3s``, ``std_m3s``.  Returns empty dict when *stats_df* is empty
        or *hydro_bus_map* is empty.
    """
    if stats_df.empty or not hydro_bus_map:
        return {}

    df = stats_df.copy()
    df["bus_id"] = df["hydro_id"].map(hydro_bus_map)
    df = df.dropna(subset=["bus_id"])
    if df.empty:
        return {}
    df["bus_id"] = df["bus_id"].astype(int)
    df["std_sq"] = df["std_m3s"].fillna(0.0) ** 2

    bus_ids = sorted(df["bus_id"].unique())
    result: dict[int, pd.DataFrame] = {}
    for bus_id in bus_ids:
        sub = df[df["bus_id"] == bus_id]
        grp = sub.groupby("stage_id")
        mean_ser = grp["mean_m3s"].sum().rename("mean_m3s")
        std_ser = grp["std_sq"].apply(np.nansum).apply(np.sqrt).rename("std_m3s")
        bus_df = mean_ser.to_frame().join(std_ser).reset_index()
        result[bus_id] = bus_df[["stage_id", "mean_m3s", "std_m3s"]]
    return result


def _aggregate_system(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-hydro stats to a single system-wide series.

    Sum of means and root-sum-of-squares of stds per stage.

    Args:
        stats_df: DataFrame with columns ``hydro_id``, ``stage_id``,
            ``mean_m3s``, ``std_m3s``.

    Returns:
        DataFrame with columns ``stage_id``, ``mean_m3s``, ``std_m3s``.
        Returns an empty DataFrame with those columns when *stats_df* is empty.
    """
    empty = pd.DataFrame(columns=["stage_id", "mean_m3s", "std_m3s"])
    if stats_df.empty:
        return empty

    df = stats_df.copy()
    df["std_sq"] = df["std_m3s"].fillna(0.0) ** 2
    grp = df.groupby("stage_id")
    mean_ser = grp["mean_m3s"].sum().rename("mean_m3s")
    std_ser = grp["std_sq"].apply(np.nansum).apply(np.sqrt).rename("std_m3s")
    result = mean_ser.to_frame().join(std_ser).reset_index()
    return result[["stage_id", "mean_m3s", "std_m3s"]]


# ---------------------------------------------------------------------------
# Band trace helpers
# ---------------------------------------------------------------------------


def _add_mean_std_band(
    fig: go.Figure,
    stage_ids: list[int],
    mean_vals: list[float],
    std_vals: list[float],
    x_labels: list[str],
    name: str,
    color: str,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add mean line + mean±std shaded band traces to *fig*.

    Three traces are added:
    1. Lower bound ``mean - std`` (invisible, reference for fill).
    2. Upper bound ``mean + std`` filled to the lower bound.
    3. Mean as a solid line (plotted last so it sits on top of the band).

    Args:
        fig: Figure to mutate.
        stage_ids: Ordered stage IDs.
        mean_vals: Mean values aligned with *stage_ids*.
        std_vals: Std values aligned with *stage_ids*.
        x_labels: Human-readable labels aligned with *stage_ids*.
        name: Legend group name.
        color: Hex colour string.
        row: Subplot row (1-based), or ``None`` for non-subplot figures.
        col: Subplot column (1-based), or ``None`` for non-subplot figures.
    """
    subplot_kw: dict = {}
    if row is not None:
        subplot_kw["row"] = row
    if col is not None:
        subplot_kw["col"] = col

    lower = [m - s for m, s in zip(mean_vals, std_vals)]
    upper = [m + s for m, s in zip(mean_vals, std_vals)]
    fill_color = hex_to_rgba(color, 0.18)

    # Lower bound (invisible anchor for fill="tonexty")
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=lower,
            name=f"{name} −σ",
            legendgroup=name,
            showlegend=False,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
        ),
        **subplot_kw,
    )
    # Upper bound with fill back to lower
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=upper,
            name=f"{name} ±σ band",
            legendgroup=name,
            showlegend=False,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=fill_color,
            hoverinfo="skip",
        ),
        **subplot_kw,
    )
    # Mean solid line
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=mean_vals,
            name=name,
            legendgroup=name,
            mode="lines",
            line={"color": color, "width": 2},
            hovertemplate=(
                f"{name}<br>Stage: %{{x}}<br>Mean: %{{y:,.0f}} m³/s<extra></extra>"
            ),
        ),
        **subplot_kw,
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _chart_system_inflow(
    hist_system: pd.DataFrame,
    synth_system: pd.DataFrame,
    stage_labels: dict[int, str],
    stage_dates: dict[int, str],
) -> go.Figure:
    """Build the system-wide historical vs synthetic inflow figure.

    Args:
        hist_system: DataFrame with columns ``stage_id``, ``mean_m3s``,
            ``std_m3s`` (historical aggregated across all hydros).
        synth_system: DataFrame with columns ``stage_id``, ``mean_m3s``,
            ``std_m3s`` (synthetic from inflow_stats_stoch).
        stage_labels: Mapping ``stage_id -> label string``.

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    fig = go.Figure()

    if not hist_system.empty:
        h = hist_system.sort_values("stage_id")
        x_dates = stage_x_dates(h["stage_id"].tolist(), stage_dates)
        _add_mean_std_band(
            fig,
            h["stage_id"].tolist(),
            h["mean_m3s"].tolist(),
            h["std_m3s"].fillna(0.0).tolist(),
            x_dates,
            "Historical",
            _HIST_COLOR,
        )

    if not synth_system.empty:
        s = synth_system.sort_values("stage_id")
        x_dates = stage_x_dates(s["stage_id"].tolist(), stage_dates)
        _add_mean_std_band(
            fig,
            s["stage_id"].tolist(),
            s["mean_m3s"].tolist(),
            s["std_m3s"].fillna(0.0).tolist(),
            x_dates,
            "Synthetic",
            _SYNTH_COLOR,
        )

    fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="Total Inflow (m³/s)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=MARGIN_DEFAULTS,
        template="plotly_white",
    )
    _all = sorted(stage_dates)
    apply_stage_date_axis(
        fig,
        stage_x_dates(_all, stage_dates),
        stage_x_labels(_all, stage_labels),
    )
    return fig


def _chart_bus_facet(
    hist_by_bus: dict[int, pd.DataFrame],
    synth_by_bus: dict[int, pd.DataFrame],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    stage_dates: dict[int, str],
    bus_ids: list[int],
) -> go.Figure:
    """Build a 2x2 faceted subplot for up to 4 buses.

    Args:
        hist_by_bus: Dict ``bus_id -> DataFrame(stage_id, mean_m3s, std_m3s)``.
        synth_by_bus: Dict ``bus_id -> DataFrame(stage_id, mean_m3s, std_m3s)``.
        bus_names: Dict ``bus_id -> name string``.
        stage_labels: Mapping ``stage_id -> label string``.
        bus_ids: Ordered list of bus IDs to display (up to 4 used).

    Returns:
        A :class:`plotly.graph_objects.Figure` with 4 subplots arranged 2x2.
    """
    selected = bus_ids[:4]
    subplot_titles = [bus_names.get(bid, str(bid)) for bid in selected]
    # Pad to 4 for make_subplots
    while len(subplot_titles) < 4:
        subplot_titles.append("")

    fig = ps.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        vertical_spacing=0.22,
        horizontal_spacing=0.1,
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for idx, bus_id in enumerate(selected):
        row, col = positions[idx]
        h = hist_by_bus.get(bus_id, pd.DataFrame())
        s = synth_by_bus.get(bus_id, pd.DataFrame())

        if not h.empty:
            h = h.sort_values("stage_id")
            x_dates = stage_x_dates(h["stage_id"].tolist(), stage_dates)
            _add_mean_std_band(
                fig,
                h["stage_id"].tolist(),
                h["mean_m3s"].tolist(),
                h["std_m3s"].fillna(0.0).tolist(),
                x_dates,
                "Historical",
                _HIST_COLOR,
                row=row,
                col=col,
            )

        if not s.empty:
            s = s.sort_values("stage_id")
            x_dates = stage_x_dates(s["stage_id"].tolist(), stage_dates)
            _add_mean_std_band(
                fig,
                s["stage_id"].tolist(),
                s["mean_m3s"].tolist(),
                s["std_m3s"].fillna(0.0).tolist(),
                x_dates,
                "Synthetic",
                _SYNTH_COLOR,
                row=row,
                col=col,
            )

    # Suppress legend for all traces except the first occurrence of each group
    seen: set[str] = set()
    for trace in fig.data:
        lg = getattr(trace, "legendgroup", None)
        if lg and trace.showlegend is not False:
            if lg in seen:
                trace.showlegend = False
            else:
                seen.add(lg)

    fig.update_layout(
        legend=LEGEND_DEFAULTS,
        margin=dict(l=50, r=20, t=60, b=10),
        template="plotly_white",
    )
    _all = sorted(stage_dates)
    apply_stage_date_axis(
        fig,
        stage_x_dates(_all, stage_dates),
        stage_x_labels(_all, stage_labels),
    )
    return fig


def _chart_hydro_explorer(
    hist_per_hydro: dict[int, pd.DataFrame],
    synth_per_hydro: dict[int, pd.DataFrame],
    hydro_meta: dict[int, dict],
    stage_labels: dict[int, str],
    stage_dates: dict[int, str],
) -> go.Figure:
    """Build a per-hydro explorer figure with dropdown menu.

    All hydros' traces are added to the figure. A Plotly ``updatemenus``
    dropdown allows toggling visibility per hydro. The first hydro is visible
    by default; all others are hidden.

    Args:
        hist_per_hydro: Dict ``hydro_id -> DataFrame(stage_id, mean_m3s, std_m3s)``.
        synth_per_hydro: Dict ``hydro_id -> DataFrame(stage_id, mean_m3s, std_m3s)``.
        hydro_meta: Dict ``hydro_id -> {name: str, ...}``.
        stage_labels: Mapping ``stage_id -> label string``.

    Returns:
        A :class:`plotly.graph_objects.Figure` with updatemenus dropdown.
    """
    fig = go.Figure()

    # Ordered list of hydro_ids (sorted for deterministic order)
    hydro_ids = sorted(hydro_meta.keys())

    # Each hydro contributes up to 6 traces: 3 for hist (lower, upper, mean)
    # and 3 for synth.  Track the trace count per hydro to build visibility arrays.
    traces_per_hydro: list[int] = []

    for hydro_id in hydro_ids:
        trace_count_before = len(fig.data)

        h = hist_per_hydro.get(hydro_id, pd.DataFrame())
        if not h.empty:
            h = h.sort_values("stage_id")
            x_dates = stage_x_dates(h["stage_id"].tolist(), stage_dates)
            _add_mean_std_band(
                fig,
                h["stage_id"].tolist(),
                h["mean_m3s"].tolist(),
                h["std_m3s"].fillna(0.0).tolist(),
                x_dates,
                "Historical",
                _HIST_COLOR,
            )

        s = synth_per_hydro.get(hydro_id, pd.DataFrame())
        if not s.empty:
            s = s.sort_values("stage_id")
            x_dates = stage_x_dates(s["stage_id"].tolist(), stage_dates)
            _add_mean_std_band(
                fig,
                s["stage_id"].tolist(),
                s["mean_m3s"].tolist(),
                s["std_m3s"].fillna(0.0).tolist(),
                x_dates,
                "Synthetic",
                _SYNTH_COLOR,
            )

        trace_count_after = len(fig.data)
        traces_per_hydro.append(trace_count_after - trace_count_before)

    total_traces = len(fig.data)

    # Build visibility arrays per hydro and dropdown buttons
    buttons: list[dict] = []
    trace_offset = 0
    for i, hydro_id in enumerate(hydro_ids):
        n = traces_per_hydro[i]
        # Visibility array: True only for this hydro's traces
        visible: list[bool] = [False] * total_traces
        for j in range(n):
            visible[trace_offset + j] = True
        name = hydro_meta[hydro_id].get("name", str(hydro_id))
        buttons.append(
            {
                "method": "update",
                "label": name,
                "args": [
                    {"visible": visible},
                    {"title": f"Inflow — {name}"},
                ],
            }
        )
        trace_offset += n

    # Default: first hydro visible, rest hidden
    for trace_idx, trace in enumerate(fig.data):
        trace.visible = trace_idx < traces_per_hydro[0] if traces_per_hydro else False  # type: ignore[assignment]

    first_name = (
        hydro_meta[hydro_ids[0]].get("name", str(hydro_ids[0]))
        if hydro_ids
        else "No Data"
    )

    apply_standard_layout(
        fig,
        title=f"Inflow — {first_name}",
        xaxis_title="Stage",
        yaxis_title="Inflow (m³/s)",
        template="plotly_white",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "active": 0,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
                "type": "dropdown",
            }
        ],
    )
    _all = sorted(stage_dates)
    apply_stage_date_axis(
        fig,
        stage_x_dates(_all, stage_dates),
        stage_x_labels(_all, stage_labels),
    )
    return fig


# ---------------------------------------------------------------------------
# Per-hydro stats extraction from aggregated data
# ---------------------------------------------------------------------------


def _per_hydro_stats(
    stats_df: pd.DataFrame,
    hydro_ids: list[int],
) -> dict[int, pd.DataFrame]:
    """Split an aggregated stats DataFrame into per-hydro DataFrames.

    Args:
        stats_df: DataFrame with columns ``hydro_id``, ``stage_id``,
            ``mean_m3s``, ``std_m3s``.
        hydro_ids: List of hydro IDs to include.

    Returns:
        Dict ``hydro_id -> DataFrame(stage_id, mean_m3s, std_m3s)``.
    """
    if stats_df.empty:
        return {}
    result: dict[int, pd.DataFrame] = {}
    for hid in hydro_ids:
        sub = stats_df[stats_df["hydro_id"] == hid][["stage_id", "mean_m3s", "std_m3s"]]
        if not sub.empty:
            result[hid] = sub.reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Correlation and noise helpers
# ---------------------------------------------------------------------------

_MONTH_NAMES: list[str] = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

_REDUCTION_COLORS: list[str] = [
    "#4A90B8",
    "#F5A623",
    "#DC4C4C",
    "#4A8B6F",
    "#B87333",
    "#8B5E3C",
    "#607D8B",
    "#8B9298",
]


def _extract_fitted_correlation(
    corr_dict: dict,
) -> tuple[list[int], np.ndarray] | None:
    """Extract hydro_ids and correlation matrix from a correlation dict.

    Navigates ``corr_dict["profiles"]["default"]["correlation_groups"][0]``
    and returns the entity IDs and matrix.

    Args:
        corr_dict: The loaded ``correlation.json`` dict.

    Returns:
        A tuple ``(hydro_ids, matrix)`` where ``hydro_ids`` is a list of
        integer IDs and ``matrix`` is a 2-D :class:`numpy.ndarray`, or
        ``None`` if any required key is missing or the matrix is empty.
    """
    try:
        group = corr_dict["profiles"]["default"]["correlation_groups"][0]
        entities: list[dict] = group["entities"]
        matrix_raw: list[list[float]] = group["matrix"]
    except (KeyError, IndexError, TypeError):
        return None

    if not entities or not matrix_raw:
        return None

    hydro_ids = [int(e["id"]) for e in entities]
    matrix = np.array(matrix_raw, dtype=np.float64)
    return hydro_ids, matrix


def _compute_empirical_correlation(
    inflow_history: pd.DataFrame,
    hydro_ids: list[int],
) -> np.ndarray | None:
    """Compute an averaged empirical correlation matrix from inflow history.

    Groups ``inflow_history`` by calendar month, computes a per-hydro
    correlation matrix for each month, then returns the element-wise mean
    across the 12 monthly matrices.

    Args:
        inflow_history: DataFrame with columns ``hydro_id``, ``date``,
            ``value_m3s``.
        hydro_ids: Ordered list of hydro IDs to include (determines row/column
            order of the returned matrix).

    Returns:
        A 2-D :class:`numpy.ndarray` of shape ``(N, N)`` where ``N =
        len(hydro_ids)``, or ``None`` when the history is empty or there are
        fewer than 2 hydros after filtering.
    """
    if inflow_history.empty or len(hydro_ids) < 2:
        return None

    df = inflow_history[inflow_history["hydro_id"].isin(hydro_ids)]
    if df.empty:
        return None
    df = _with_month(df)

    n = len(hydro_ids)
    cumulative = np.zeros((n, n), dtype=np.float64)
    count = 0

    for _month in range(1, 13):
        sub = df[df["month"] == _month]
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index="start_date",
            columns="hydro_id",
            values="value_m3s",
            aggfunc="mean",
        )
        # Reindex to the canonical hydro_id order; missing hydros get NaN cols
        pivot = pivot.reindex(columns=hydro_ids)
        if pivot.shape[0] < 2:
            continue
        corr_m = pivot.corr(min_periods=10)
        corr_arr = corr_m.reindex(index=hydro_ids, columns=hydro_ids).values
        if not np.all(np.isnan(corr_arr)):
            cumulative += np.nan_to_num(corr_arr, nan=0.0)
            count += 1

    if count == 0:
        return None

    result = cumulative / count
    # Ensure diagonal is exactly 1.0 for non-NaN entries
    np.fill_diagonal(result, 1.0)
    return result


def _chart_correlation_heatmap(
    matrix: np.ndarray,
    hydro_ids: list[int],
    hydro_meta: dict[int, dict],
    title: str,
    max_display: int = 40,
) -> tuple[go.Figure, str]:
    """Build a correlation heatmap figure.

    Truncates to ``max_display x max_display`` and appends a truncation
    notice to the title when the matrix exceeds that size.

    Args:
        matrix: Square symmetric 2-D array.
        hydro_ids: Ordered list of hydro IDs matching the matrix rows/cols.
        hydro_meta: Mapping ``hydro_id -> {"name": str, ...}``.
        title: Chart title (may be extended with truncation note).
        max_display: Maximum number of rows/columns to display.

    Returns:
        A tuple ``(fig, resolved_title)`` where ``resolved_title`` includes
        the truncation note when the matrix was truncated.
    """
    total = len(hydro_ids)
    if total > max_display:
        display_ids = hydro_ids[:max_display]
        display_matrix = matrix[:max_display, :max_display]
        title = f"{title} (showing {max_display} of {total})"
    else:
        display_ids = hydro_ids
        display_matrix = matrix

    labels = [
        hydro_meta[hid]["name"] if hid in hydro_meta else str(hid)
        for hid in display_ids
    ]

    fig = go.Figure(
        go.Heatmap(
            z=display_matrix.tolist(),
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-1.0,
            zmax=1.0,
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
            colorbar={"title": "Corr"},
        )
    )
    fig.update_layout(
        xaxis_title="Hydro",
        yaxis_title="Hydro",
        margin=MARGIN_DEFAULTS,
        template="plotly_white",
    )
    return fig, title


def _chart_noise_histogram(noise_openings: pd.DataFrame) -> go.Figure:
    """Build a noise histogram overlaid with the N(0,1) reference curve.

    Args:
        noise_openings: DataFrame with a ``value`` column of noise samples.

    Returns:
        A :class:`plotly.graph_objects.Figure` with two traces: histogram and
        N(0,1) reference line.
    """
    vals = noise_openings["value"].dropna().values
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=vals,
            histnorm="probability density",
            nbinsx=60,
            name="Noise samples",
            marker_color=COLORS["hydro"],
            opacity=0.7,
            hovertemplate="Value: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
        )
    )
    x_ref = np.linspace(float(vals.min()), float(vals.max()), 300)
    y_ref = np.exp(-0.5 * x_ref**2) / np.sqrt(2 * np.pi)
    fig.add_trace(
        go.Scatter(
            x=x_ref,
            y=y_ref,
            name="N(0,1) reference",
            line={"color": COLORS["deficit"], "width": 2, "dash": "dash"},
            hovertemplate="x: %{x:.3f}<br>N(0,1): %{y:.4f}<extra></extra>",
        )
    )
    apply_standard_layout(
        fig,
        xaxis_title="Value",
        yaxis_title="Density",
        template="plotly_white",
    )
    return fig


def _chart_noise_boxplot_by_stage(
    noise_openings: pd.DataFrame,
    stage_labels: dict[int, str],
) -> go.Figure:
    """Build a box plot of noise values grouped by stage (all stages).

    One box per stage along the x-axis.  Each box aggregates across all hydros
    and opening indices for that stage.  Whiskers show the 5th–95th percentile.

    Args:
        noise_openings: DataFrame with columns ``stage_id`` and ``value``.
        stage_labels: Mapping ``stage_id -> label string``.

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    stage_ids = sorted(noise_openings["stage_id"].unique().tolist())

    # Pre-compute percentiles per stage for efficiency instead of
    # sending all 448K samples to Plotly
    rows: list[dict] = []
    for sid in stage_ids:
        vals = noise_openings.loc[noise_openings["stage_id"] == sid, "value"].dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "stage_id": sid,
                "median": float(vals.median()),
                "q1": float(vals.quantile(0.25)),
                "q3": float(vals.quantile(0.75)),
                "lowerfence": float(vals.quantile(0.05)),
                "upperfence": float(vals.quantile(0.95)),
            }
        )

    if not rows:
        return go.Figure()

    stats = pd.DataFrame(rows)
    xlabels = [stage_labels.get(int(s), str(s)) for s in stats["stage_id"]]

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=xlabels,
            median=stats["median"],
            q1=stats["q1"],
            q3=stats["q3"],
            lowerfence=stats["lowerfence"],
            upperfence=stats["upperfence"],
            marker_color=COLORS["hydro"],
            fillcolor="rgba(74,144,184,0.3)",
            showlegend=False,
            hovertemplate=(
                "Stage: %{x}<br>"
                "Median: %{median:.3f}<br>"
                "Q1: %{q1:.3f}  Q3: %{q3:.3f}<br>"
                "P5: %{lowerfence:.3f}  P95: %{upperfence:.3f}"
                "<extra></extra>"
            ),
        )
    )
    # Zero reference line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#94A3B8",
        line_width=1,
        annotation_text="0",
        annotation_position="bottom left",
    )
    fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="Noise Value",
        margin=MARGIN_DEFAULTS,
        template="plotly_white",
    )
    return fig


def _chart_ar_order_distribution(
    fitting_report: dict,
    hydro_meta: dict[int, dict],
) -> go.Figure:
    """Build a bar chart of selected AR order distribution across hydros.

    Hover text includes the count and list of hydro names for each order.

    Args:
        fitting_report: The loaded ``fitting_report.json`` dict with a
            ``"hydros"`` key mapping hydro IDs to per-hydro fitting data.
        hydro_meta: Mapping ``hydro_id -> {"name": str, ...}``.

    Returns:
        A :class:`plotly.graph_objects.Figure` with one
        :class:`~plotly.graph_objects.Bar`
        trace.
    """
    hydros_data: dict = fitting_report.get("hydros", {})
    order_names: dict[int, list[str]] = {}
    for hid_str, hydro_info in hydros_data.items():
        order = int(hydro_info.get("selected_order", 0))
        hid = int(hid_str)
        name = hydro_meta.get(hid, {}).get("name", str(hid))
        order_names.setdefault(order, []).append(name)

    if not order_names:
        return go.Figure()

    orders = list(range(min(min(order_names), 1), max(order_names) + 1))
    counts = [len(order_names.get(o, [])) for o in orders]

    # Build hover text with hydro names (truncate long lists)
    hover_texts: list[str] = []
    for o in orders:
        names = sorted(order_names.get(o, []))
        n = len(names)
        if n == 0:
            hover_texts.append(f"Order {o}: 0 hydros")
        elif n <= 8:
            names_str = ", ".join(names)
            hover_texts.append(f"Order {o}: {n} hydros<br>{names_str}")
        else:
            shown = ", ".join(names[:8])
            hover_texts.append(
                f"Order {o}: {n} hydros<br>{shown}<br>... and {n - 8} more"
            )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=orders,
            y=counts,
            marker_color=COLORS["hydro"],
            customdata=hover_texts,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Selected AR Order",
        yaxis_title="Number of Hydros",
        xaxis=dict(dtick=1),
        margin=MARGIN_DEFAULTS,
        template="plotly_white",
    )
    return fig


def _chart_order_reduction_reasons(fitting_report: dict) -> go.Figure:
    """Build a stacked bar chart of AR order reduction reasons by season.

    Args:
        fitting_report: The loaded ``fitting_report.json`` dict with a
            ``"hydros"`` key.

    Returns:
        A :class:`plotly.graph_objects.Figure` with one stacked bar trace per
        reduction reason.
    """
    hydros_data: dict = fitting_report.get("hydros", {})
    reason_season: dict[str, list[int]] = {}
    for hydro_info in hydros_data.values():
        contrib = hydro_info.get("contribution_reductions", [])
        for season_id, reasons in enumerate(contrib):
            if season_id >= 12:
                break
            if not isinstance(reasons, list):
                continue
            for reason in reasons:
                if reason not in reason_season:
                    reason_season[reason] = [0] * 12
                reason_season[reason][season_id] += 1

    fig = go.Figure()
    for i, (reason, counts) in enumerate(sorted(reason_season.items())):
        fig.add_trace(
            go.Bar(
                x=_MONTH_NAMES,
                y=counts,
                name=reason,
                marker_color=_REDUCTION_COLORS[i % len(_REDUCTION_COLORS)],
                hovertemplate=f"{reason}<br>%{{x}}: %{{y}} reductions<extra></extra>",
            )
        )
    apply_standard_layout(
        fig,
        xaxis_title="Season (Month)",
        yaxis_title="Count of Reductions",
        barmode="stack",
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_system_inflow_hero(data: DashboardData) -> str:
    """Render the system-wide historical vs synthetic inflow hero chart."""
    if data.inflow_history.empty:
        return collapsible_section(
            "System-Wide Inflow",
            _NO_HIST,
            section_id="v2-stoch-section-a",
        )

    hist_stats = _compute_historical_stats(data)
    hist_system = _aggregate_system(hist_stats)

    # Synthetic system-level aggregation from simulation inflows
    synth_stats = _compute_synthetic_stats(data)
    synth_system = _aggregate_system(synth_stats)

    fig = _chart_system_inflow(
        hist_system, synth_system, data.stage_labels, data.stage_dates
    )
    content = make_chart_card(
        fig, "System-Wide Inflow (Historical vs Synthetic)", "v2-stoch-system-inflow"
    )

    return collapsible_section(
        "System-Wide Inflow",
        content,
        section_id="v2-stoch-section-a",
    )


def _render_inflow_by_bus(data: DashboardData) -> str:
    """Render the inflow-by-bus 2x2 faceted subplot section."""
    if data.inflow_history.empty:
        return collapsible_section(
            "Inflow by Bus",
            _NO_HIST,
            section_id="v2-stoch-section-b",
        )

    if not data.hydro_bus_map:
        return collapsible_section(
            "Inflow by Bus",
            "<p>No bus mapping available.</p>",
            section_id="v2-stoch-section-b",
        )

    hist_stats = _compute_historical_stats(data)
    hist_by_bus = _aggregate_by_bus(hist_stats, data.hydro_bus_map)

    if not hist_by_bus:
        return collapsible_section(
            "Inflow by Bus",
            "<p>No bus mapping available.</p>",
            section_id="v2-stoch-section-b",
        )

    synth_stats = _compute_synthetic_stats(data)
    synth_by_bus = _aggregate_by_bus(synth_stats, data.hydro_bus_map)

    # Use non_fictitious_bus_ids intersected with buses that have hydros
    hydro_bus_ids = set(data.hydro_bus_map.values())
    non_fict = [bid for bid in data.non_fictitious_bus_ids if bid in hydro_bus_ids]
    # Fallback: all buses with hydros (sorted) if non_fictitious_bus_ids is empty
    if not non_fict:
        non_fict = sorted(hydro_bus_ids)

    bus_ids_to_show = non_fict[:4]
    if not bus_ids_to_show:
        return collapsible_section(
            "Inflow by Bus",
            "<p>No bus mapping available.</p>",
            section_id="v2-stoch-section-b",
        )

    fig = _chart_bus_facet(
        hist_by_bus,
        synth_by_bus,
        data.bus_names,
        data.stage_labels,
        data.stage_dates,
        bus_ids_to_show,
    )
    content = make_chart_card(
        fig, "Inflow by Bus (Historical vs Synthetic)", "v2-stoch-bus-facet", height=600
    )

    return collapsible_section(
        "Inflow by Bus",
        content,
        section_id="v2-stoch-section-b",
    )


def _render_per_hydro_inflow_explorer(data: DashboardData) -> str:
    """Render the per-hydro inflow explorer with plant explorer table."""
    if not data.hydro_meta:
        return collapsible_section(
            "Per-Hydro Inflow Explorer",
            "<p>No hydro data.</p>",
            section_id="v2-stoch-section-c",
        )

    if data.inflow_history.empty:
        return collapsible_section(
            "Per-Hydro Inflow Explorer",
            _NO_HIST,
            section_id="v2-stoch-section-c",
        )

    hist_stats = _compute_historical_stats(data)
    hydro_ids = sorted(data.hydro_meta.keys())
    hist_per_hydro = _per_hydro_stats(hist_stats, hydro_ids)

    synth_stats = _compute_synthetic_stats(data)
    synth_per_hydro = _per_hydro_stats(synth_stats, hydro_ids)

    # Serialize per-hydro data for JS-driven rendering
    hydro_json: dict[str, dict] = {}
    for hid in hydro_ids:
        meta = data.hydro_meta[hid]
        entry: dict = {"name": meta.get("name", str(hid))}

        h = hist_per_hydro.get(hid, pd.DataFrame())
        if not h.empty:
            h = h.sort_values("stage_id")
            entry["hist_stages"] = h["stage_id"].tolist()
            entry["hist_labels"] = stage_x_labels(
                h["stage_id"].tolist(), data.stage_labels
            )
            entry["hist_dates"] = stage_x_dates(
                h["stage_id"].tolist(), data.stage_dates
            )
            entry["hist_mean"] = h["mean_m3s"].tolist()
            entry["hist_std"] = h["std_m3s"].fillna(0.0).tolist()

        s = synth_per_hydro.get(hid, pd.DataFrame())
        if not s.empty:
            s = s.sort_values("stage_id")
            entry["synth_stages"] = s["stage_id"].tolist()
            entry["synth_labels"] = stage_x_labels(
                s["stage_id"].tolist(), data.stage_labels
            )
            entry["synth_dates"] = stage_x_dates(
                s["stage_id"].tolist(), data.stage_dates
            )
            entry["synth_mean"] = s["mean_m3s"].tolist()
            entry["synth_std"] = s["std_m3s"].fillna(0.0).tolist()

        hydro_json[str(hid)] = entry

    # Build table rows
    table_rows: list[str] = []
    for hid in hydro_ids:
        meta = data.hydro_meta[hid]
        name = meta.get("name", str(hid))
        bus_id = meta.get("bus_id", "")
        bus_name = data.bus_names.get(bus_id, str(bus_id)) if bus_id != "" else ""
        table_rows.append(
            f'<tr data-name="{escape_attr(name.lower())}" data-index="{hid}">'
            f"<td>{escape_text(name)}</td>"
            f"<td>{escape_text(bus_name)}</td>"
            f"</tr>"
        )

    columns: list[tuple[str, str]] = [
        ("Name", "string"),
        ("Bus", "string"),
    ]
    table_pane = (
        '<div class="explorer-table-pane"'
        ' style="width:300px;min-width:200px;flex-shrink:0;'
        'overflow-y:auto;max-height:500px;">'
        + plant_explorer_table(
            "stoch-hydro-tbody",
            "stoch-hydro-search",
            columns,
            "".join(table_rows),
        )
        + "</div>"
    )

    chart_div = wrap_chart('<div id="stoch-hydro-chart" style="width:100%;"></div>')
    detail_pane = (
        '<div class="explorer-detail-pane"'
        ' style="flex:1;min-width:0;"'
        ' id="stoch-hydro-detail">' + chart_div + "</div>"
    )

    explorer_html = (
        '<div class="explorer-container">' + table_pane + detail_pane + "</div>"
    )

    # JS: data + render function + init call
    hist_color = _HIST_COLOR
    synth_color = _SYNTH_COLOR
    js = f"""
<script>
window._stochHydroData = {json_for_script(hydro_json)};
function _renderStochHydro(containerId, entry) {{
    var chartDiv = document.getElementById('stoch-hydro-chart');
    if (!chartDiv) return;
    var chartHeight = 500;
    var traces = [];
    if (entry.hist_mean) {{
        var hUpper = entry.hist_mean.map(function(m,i){{ return m + entry.hist_std[i]; }});
        var hLower = entry.hist_mean.map(function(m,i){{ return m - entry.hist_std[i]; }});
        traces.push({{x:entry.hist_dates, y:hUpper, mode:'lines',
            line:{{width:0}}, showlegend:false, hoverinfo:'skip',
            legendgroup:'hist'}});
        traces.push({{x:entry.hist_dates, y:hLower, mode:'lines',
            line:{{width:0}}, fill:'tonexty',
            fillcolor:'rgba(74,144,184,0.15)', showlegend:false,
            hoverinfo:'skip', legendgroup:'hist'}});
        traces.push({{x:entry.hist_dates, y:entry.hist_mean, mode:'lines',
            name:'Historical', line:{{color:'{hist_color}',width:2}},
            legendgroup:'hist'}});
    }}
    if (entry.synth_mean) {{
        var sUpper = entry.synth_mean.map(function(m,i){{ return m + entry.synth_std[i]; }});
        var sLower = entry.synth_mean.map(function(m,i){{ return m - entry.synth_std[i]; }});
        traces.push({{x:entry.synth_dates, y:sUpper, mode:'lines',
            line:{{width:0}}, showlegend:false, hoverinfo:'skip',
            legendgroup:'synth'}});
        traces.push({{x:entry.synth_dates, y:sLower, mode:'lines',
            line:{{width:0}}, fill:'tonexty',
            fillcolor:'rgba(245,158,11,0.15)', showlegend:false,
            hoverinfo:'skip', legendgroup:'synth'}});
        traces.push({{x:entry.synth_dates, y:entry.synth_mean, mode:'lines',
            name:'Synthetic', line:{{color:'{synth_color}',width:2}},
            legendgroup:'synth'}});
    }}
    var layout = {{
        height: chartHeight,
        title: {{text: 'Inflow — ' + entry.name, font:{{size:13}},
                 x:0.02, xanchor:'left'}},
        xaxis: {{title:'Stage', type:'date', tickmode:'array',
                 tickvals:(entry.hist_dates||entry.synth_dates),
                 ticktext:(entry.hist_labels||entry.synth_labels)}},
        yaxis: {{title:'Inflow (m³/s)'}},
        legend: {{orientation:'h', yanchor:'bottom', y:1.02,
                  xanchor:'center', x:0.5, font:{{size:11}}}},
        hovermode: 'x unified',
        margin: {{l:60, r:30, t:60, b:60}},
        template: 'plotly_white'
    }};
    Plotly.newPlot(chartDiv, traces, layout, {{responsive:true}});
}}
document.addEventListener('DOMContentLoaded', function() {{
    initPlantExplorer({{
        tableId: 'stoch-hydro-tbody',
        searchInputId: 'stoch-hydro-search',
        detailContainerId: 'stoch-hydro-detail',
        dataVar: '_stochHydroData',
        renderDetail: _renderStochHydro
    }});
}});
</script>
"""

    content = explorer_html + js
    return collapsible_section(
        "Per-Hydro Inflow Explorer",
        content,
        section_id="v2-stoch-section-c",
    )


def _render_spatial_correlation(data: DashboardData) -> str:
    """Render the spatial-correlation section: historical vs synthetic.

    Side-by-side heatmaps with a season dropdown.  Historical correlation is
    computed per calendar month from ``inflow_history``.  Synthetic correlation
    comes from the seasonal profiles in ``correlation.json``.
    """
    if not data.correlation:
        return collapsible_section(
            "Spatial Correlation",
            "<p>No correlation data available.</p>",
            section_id="v2-stoch-section-d",
        )

    # Extract entity IDs from the default profile
    extracted = _extract_fitted_correlation(data.correlation)
    if extracted is None:
        return collapsible_section(
            "Spatial Correlation",
            "<p>No correlation matrix in data.</p>",
            section_id="v2-stoch-section-d",
        )

    hydro_ids_orig, _ = extracted

    # Sort hydros by bus for block structure visibility
    bus_map = data.hydro_bus_map
    sorted_ids = sorted(hydro_ids_orig, key=lambda h: (bus_map.get(h, 999), h))
    n = len(sorted_ids)

    # Build reorder index: original position -> sorted position
    orig_index = {h: i for i, h in enumerate(hydro_ids_orig)}
    reorder = [orig_index[h] for h in sorted_ids]

    # Hover labels (name) — no tick labels on axes
    hover_labels = [
        data.hydro_meta[h]["name"] if h in data.hydro_meta else str(h)
        for h in sorted_ids
    ]

    # Compute bus separator positions and bus labels for annotation
    bus_boundaries: list[dict] = []
    prev_bus: int | None = None
    bus_start = 0
    for i, hid in enumerate(sorted_ids):
        bid = bus_map.get(hid, -1)
        if bid != prev_bus:
            if prev_bus is not None:
                bus_name = data.bus_names.get(prev_bus, str(prev_bus))
                bus_boundaries.append({"pos": i, "label": bus_name, "start": bus_start})
            bus_start = i
            prev_bus = bid
    # Last bus
    if prev_bus is not None:
        bus_name = data.bus_names.get(prev_bus, str(prev_bus))
        bus_boundaries.append({"pos": n, "label": bus_name, "start": bus_start})

    def _reorder_matrix(
        mat: np.ndarray,
        idx: list[int],
    ) -> list[list[float]]:
        """Reorder a matrix by the given index permutation.

        Correlations are rounded to 4 decimals: the heatmap hover shows ``.3f``
        and the colour scale resolves far less, so full float64 reprs (~18
        chars each) are pure payload bloat — rounding cuts this block ~3x.
        """
        return np.round(mat[np.ix_(idx, idx)], 4).tolist()

    # Build per-season synthetic matrices from correlation.json profiles
    synthetic_by_season: dict[str, list[list[float]]] = {}
    profiles = data.correlation.get("profiles", {})
    for month_idx in range(12):
        profile_key = f"season_{month_idx:02d}"
        profile = profiles.get(profile_key)
        if profile is None:
            continue
        try:
            mat = np.array(profile["correlation_groups"][0]["matrix"], dtype=np.float64)
            synthetic_by_season[_MONTH_NAMES[month_idx]] = _reorder_matrix(mat, reorder)
        except (KeyError, IndexError, TypeError):
            continue

    if not synthetic_by_season:
        _, default_mat = extracted
        synthetic_by_season["All"] = _reorder_matrix(default_mat, reorder)

    # Build per-month historical correlation matrices
    historical_by_season: dict[str, list[list[float]]] = {}
    if not data.inflow_history.empty and len(sorted_ids) >= 2:
        df = _with_month(
            data.inflow_history[data.inflow_history["hydro_id"].isin(sorted_ids)]
        )

        for month_idx in range(12):
            month_num = month_idx + 1
            sub = df[df["month"] == month_num]
            if sub.empty:
                continue
            pivot = sub.pivot_table(
                index="start_date",
                columns="hydro_id",
                values="value_m3s",
                aggfunc="mean",
            )
            pivot = pivot.reindex(columns=sorted_ids)
            if pivot.shape[0] < 2:
                continue
            corr_m = pivot.corr(min_periods=10)
            corr_arr = corr_m.reindex(
                index=sorted_ids, columns=sorted_ids
            ).values.copy()
            np.fill_diagonal(corr_arr, 1.0)
            # Round to 4 decimals to match the synthetic matrices (see
            # ``_reorder_matrix``) — keeps the heatmap's .3f hover exact while
            # avoiding full float64 reprs in the embedded payload.
            historical_by_season[_MONTH_NAMES[month_idx]] = np.round(
                np.nan_to_num(corr_arr, nan=0.0), 4
            ).tolist()

    # Determine available seasons
    available = [
        m
        for m in _MONTH_NAMES
        if m in synthetic_by_season and m in historical_by_season
    ]
    if not available:
        available = list(synthetic_by_season.keys())

    # Serialize data for JS
    corr_data = {
        "labels": hover_labels,
        "seasons": available,
        "synthetic": synthetic_by_season,
        "historical": historical_by_season,
        "busBoundaries": bus_boundaries,
        "n": n,
    }

    # Build HTML
    options_html = "".join(f'<option value="{s}">{s}</option>' for s in available)
    dropdown = (
        '<select id="stoch-corr-season" style="'
        "padding:4px 8px;border-radius:4px;border:1px solid #ccc;"
        'font-size:0.875rem;margin-bottom:12px;">' + options_html + "</select>"
    )
    selector_row = (
        '<div style="display:flex;align-items:center;gap:8px;'
        'margin-bottom:8px;">'
        "<span><strong>Season:</strong></span>" + dropdown + "</div>"
    )

    hist_card = wrap_chart('<div id="stoch-corr-hist" style="width:100%;"></div>')
    synth_card = wrap_chart('<div id="stoch-corr-synth" style="width:100%;"></div>')
    charts = chart_grid([hist_card, synth_card])

    js = f"""
<script>
(function() {{
    var _cd = {json_for_script(corr_data)};
    // Build bus separator shapes and annotations
    var _shapes = [], _annotations = [];
    _cd.busBoundaries.forEach(function(b) {{
        // Vertical line
        _shapes.push({{
            type:'line', x0:b.pos-0.5, x1:b.pos-0.5,
            y0:-0.5, y1:_cd.n-0.5,
            line:{{color:'#334155',width:1.5}}
        }});
        // Horizontal line
        _shapes.push({{
            type:'line', x0:-0.5, x1:_cd.n-0.5,
            y0:b.pos-0.5, y1:b.pos-0.5,
            line:{{color:'#334155',width:1.5}}
        }});
        // Bus label (on x-axis side, rotated)
        var mid = (b.start + b.pos) / 2;
        _annotations.push({{
            x: mid, y: _cd.n + 2, text: b.label,
            showarrow: false, font:{{size:10,color:'#334155'}},
            xref:'x', yref:'y'
        }});
    }});
    // Numeric tick positions (no labels — hover provides names)
    var _tickvals = [];
    for (var i = 0; i < _cd.n; i += 10) _tickvals.push(i);
    function _makeHeatmap(divId, matrix, title) {{
        var el = document.getElementById(divId);
        if (!matrix) {{
            el.innerHTML = '<p style="padding:1rem;">No data for this season.</p>';
            return;
        }}
        var trace = {{
            z: matrix, type:'heatmap', colorscale:'RdBu',
            zmid:0, zmin:-1, zmax:1,
            x: _cd.labels, y: _cd.labels,
            hovertemplate: '%{{y}} vs %{{x}}: %{{z:.3f}}<extra></extra>',
            colorbar:{{title:'Corr',len:0.8}},
            showscale: (divId === 'stoch-corr-synth')
        }};
        var layout = {{
            title:{{text:title, font:{{size:13}}, x:0.02, xanchor:'left'}},
            xaxis:{{showticklabels:false, showgrid:false}},
            yaxis:{{showticklabels:false, showgrid:false, autorange:'reversed'}},
            shapes: _shapes.slice(),
            annotations: _annotations.slice(),
            margin:{{l:20, r:40, t:40, b:40}},
            template:'plotly_white',
            height: Math.max(500, _cd.n * 4 + 80)
        }};
        Plotly.newPlot(divId, [trace], layout, {{responsive:true}});
    }}
    function _updateCorr() {{
        var s = document.getElementById('stoch-corr-season').value;
        _makeHeatmap('stoch-corr-hist', _cd.historical[s], 'Historical — '+s);
        _makeHeatmap('stoch-corr-synth', _cd.synthetic[s], 'Synthetic — '+s);
    }}
    document.addEventListener('DOMContentLoaded', function() {{
        var sel = document.getElementById('stoch-corr-season');
        if (sel) {{ sel.addEventListener('change', _updateCorr); _updateCorr(); }}
    }});
}})();
</script>
"""

    content = selector_row + charts + js
    return collapsible_section(
        "Spatial Correlation — Historical vs Synthetic",
        content,
        section_id="v2-stoch-section-d",
    )


def _render_noise_diagnostics(data: DashboardData) -> str:
    """Render the noise-diagnostics box plot across all stages."""
    if data.noise_openings.empty:
        return collapsible_section(
            "Noise Diagnostics",
            "<p>No noise data.</p>",
            section_id="v2-stoch-section-e",
        )

    box_fig = _chart_noise_boxplot_by_stage(data.noise_openings, data.stage_labels)
    box_html = make_chart_card(
        box_fig,
        "Noise Distribution by Stage",
        "v2-stoch-noise-boxplot",
    )

    content = chart_grid([box_html], single=True)
    return collapsible_section(
        "Noise Diagnostics",
        content,
        section_id="v2-stoch-section-e",
    )


def _render_ar_model_summary(data: DashboardData) -> str:
    """Render the AR model summary (order distribution histogram)."""
    if not data.fitting_report.get("hydros"):
        return collapsible_section(
            "AR Model Summary",
            "<p>No AR model was fitted for this case.</p>",
            section_id="v2-stoch-section-f",
            default_collapsed=True,
        )

    order_fig = _chart_ar_order_distribution(data.fitting_report, data.hydro_meta)
    order_html = make_chart_card(
        order_fig,
        "AR Order Distribution",
        "v2-stoch-ar-order",
    )

    content = chart_grid([order_html], single=True)
    return collapsible_section(
        "AR Model Summary",
        content,
        section_id="v2-stoch-section-f",
        default_collapsed=True,
    )


# ---------------------------------------------------------------------------
# Tab interface (TabModule protocol)
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:
    """Return True when stochastic model output is available."""
    return data.stochastic_available


def render(data: DashboardData) -> str:
    """Return the full HTML string for the v2 Stochastic Model tab content area."""
    sections = [
        _render_system_inflow_hero(data),
        _render_inflow_by_bus(data),
        _render_per_hydro_inflow_explorer(data),
        _render_spatial_correlation(data),
        _render_noise_diagnostics(data),
        _render_ar_model_summary(data),
    ]
    return "".join(sections)
