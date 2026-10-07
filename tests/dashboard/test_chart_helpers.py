"""Unit tests for novomodelo_bridge.dashboard.chart_helpers.

Covers compute_percentiles, stage_hours_weighted_mean, add_mean_p50_band,
make_chart_card, compute_npv_costs, group_costs, and compute_cost_summary.
"""

from __future__ import annotations

import dataclasses
import math
import re

import pandas as pd
import plotly.graph_objects as go
import polars as pl
import pytest

from novomodelo_bridge.comparators import charts as _cmp_charts
from novomodelo_bridge.comparators import report_builder
from novomodelo_bridge.comparators.model import PercentileData, ResultComparison
from novomodelo_bridge.dashboard.chart_helpers import (
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
    stage_hours_weighted_mean,
)
from tests.golden_utils import assert_html_golden

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def five_scenarios_three_stages() -> pd.DataFrame:
    """Return a pandas DataFrame with 5 scenarios x 3 stages."""
    rows = [
        {"stage_id": s, "scenario_id": sc, "value": float(s * 10 + sc)}
        for s in range(1, 4)
        for sc in range(1, 6)
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def percentile_df() -> pd.DataFrame:
    """Return a percentile DataFrame with 10 rows suitable for add_mean_p50_band."""
    return pd.DataFrame(
        {
            "stage_id": list(range(1, 11)),
            "mean": [float(i) for i in range(10)],
            "p10": [float(i) * 0.8 for i in range(10)],
            "p50": [float(i) * 0.95 for i in range(10)],
            "p90": [float(i) * 1.2 for i in range(10)],
        }
    )


# ---------------------------------------------------------------------------
# compute_percentiles — pandas input
# ---------------------------------------------------------------------------


def test_compute_percentiles_pandas_basic(
    five_scenarios_three_stages: pd.DataFrame,
) -> None:
    """Five scenarios x three stages produces 3 rows with the expected columns."""
    result = compute_percentiles(five_scenarios_three_stages, ["stage_id"], "value")

    assert result.shape[0] == 3
    assert list(result.columns) == ["stage_id", "mean", "p10", "p50", "p90"]


def test_compute_percentiles_pandas_values(
    five_scenarios_three_stages: pd.DataFrame,
) -> None:
    """Computed mean equals the arithmetic mean of the five scenarios per stage."""
    result = compute_percentiles(five_scenarios_three_stages, ["stage_id"], "value")
    # For stage 1: values are 11, 12, 13, 14, 15 -> mean = 13.0
    stage1 = result[result["stage_id"] == 1]
    assert stage1["mean"].iloc[0] == pytest.approx(13.0)


# ---------------------------------------------------------------------------
# compute_percentiles — polars input
# ---------------------------------------------------------------------------


def test_compute_percentiles_polars_input(
    five_scenarios_three_stages: pd.DataFrame,
) -> None:
    """Polars DataFrame input is converted and returns correct pandas output."""
    polars_df = pl.from_pandas(five_scenarios_three_stages)

    result = compute_percentiles(polars_df, ["stage_id"], "value")

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 3
    assert list(result.columns) == ["stage_id", "mean", "p10", "p50", "p90"]
    # P50 for stage 2 using values: 21, 22, 23, 24, 25 -> median = 23.0
    stage2 = result[result["stage_id"] == 2]
    assert stage2["p50"].iloc[0] == pytest.approx(23.0)


# ---------------------------------------------------------------------------
# compute_percentiles — empty input
# ---------------------------------------------------------------------------


def test_compute_percentiles_empty() -> None:
    """Empty DataFrame returns empty result with the expected column names."""
    empty_df = pd.DataFrame(columns=["stage_id", "scenario_id", "value"])

    result = compute_percentiles(empty_df, ["stage_id"], "value")

    assert result.empty
    assert list(result.columns) == ["stage_id", "mean", "p10", "p50", "p90"]


# ---------------------------------------------------------------------------
# compute_percentiles — custom percentiles
# ---------------------------------------------------------------------------


def test_compute_percentiles_custom_percentiles(
    five_scenarios_three_stages: pd.DataFrame,
) -> None:
    """Custom percentile tuple produces correctly named columns."""
    result = compute_percentiles(
        five_scenarios_three_stages,
        ["stage_id"],
        "value",
        percentiles=(0.05, 0.5, 0.95),
    )

    assert list(result.columns) == ["stage_id", "mean", "p5", "p50", "p95"]
    assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# stage_hours_weighted_mean
# ---------------------------------------------------------------------------


def test_stage_hours_weighted_mean_weights_by_stage_hours() -> None:
    """A 648-hour stage outweighs a 168-hour stage, unlike a bare .mean().

    stage 1: value=100, hours=168; stage 2: value=200, hours=648.
    Weighted: (100*168 + 200*648) / (168 + 648) ~= 179.4, not the
    unweighted mean of 150.0.
    """
    lf = pl.DataFrame(
        {"stage_id": [1, 2], "line_id": [1, 1], "value": [100.0, 200.0]}
    ).lazy()

    result = stage_hours_weighted_mean(lf, "value", ["line_id"], {1: 168.0, 2: 648.0})

    assert result.height == 1
    weighted = result["value"][0]
    assert weighted == pytest.approx(179.41176, rel=1e-4)
    assert weighted != pytest.approx(150.0)


def test_stage_hours_weighted_mean_uniform_duration_matches_bare_mean() -> None:
    """Equal stage hours reduce the weighted mean to the plain mean (no-op case)."""
    lf = pl.DataFrame(
        {"stage_id": [1, 2], "line_id": [1, 1], "value": [100.0, 200.0]}
    ).lazy()

    result = stage_hours_weighted_mean(lf, "value", ["line_id"], {1: 720.0, 2: 720.0})

    assert result["value"][0] == pytest.approx(150.0)


def test_stage_hours_weighted_mean_stage_hours_as_dataframe() -> None:
    """A stage-hours DataFrame (e.g. summed from a block-hours frame) works
    the same as the ``{stage_id: hours}`` dict form."""
    lf = pl.DataFrame(
        {"stage_id": [1, 2], "line_id": [1, 1], "value": [100.0, 200.0]}
    ).lazy()
    stage_hours_df = pl.DataFrame({"stage_id": [1, 2], "_hours": [168.0, 648.0]})

    result = stage_hours_weighted_mean(lf, "value", ["line_id"], stage_hours_df)

    assert result["value"][0] == pytest.approx(179.41176, rel=1e-4)


def test_stage_hours_weighted_mean_multiple_groups() -> None:
    """Each group in group_cols is weighted independently."""
    lf = pl.DataFrame(
        {
            "stage_id": [1, 2, 1, 2],
            "line_id": [1, 1, 2, 2],
            "value": [100.0, 200.0, 10.0, 20.0],
        }
    ).lazy()

    result = stage_hours_weighted_mean(lf, "value", ["line_id"], {1: 168.0, 2: 648.0})

    by_line = dict(zip(result["line_id"].to_list(), result["value"].to_list()))
    assert by_line[1] == pytest.approx(179.41176, rel=1e-4)
    assert by_line[2] == pytest.approx(17.941176, rel=1e-4)


def test_stage_hours_weighted_mean_empty_frame_returns_empty() -> None:
    """An empty input frame is a no-op, not an error."""
    lf = pl.DataFrame(
        schema={"stage_id": pl.Int64, "line_id": pl.Int64, "value": pl.Float64}
    ).lazy()

    result = stage_hours_weighted_mean(lf, "value", ["line_id"], {1: 168.0, 2: 648.0})

    assert result.height == 0
    assert list(result.columns) == ["line_id", "value"]


def test_stage_hours_weighted_mean_zero_total_hours_returns_empty() -> None:
    """Sigma(stage_hours) == 0 is a no-op rather than a division by zero."""
    lf = pl.DataFrame(
        {"stage_id": [1, 2], "line_id": [1, 1], "value": [100.0, 200.0]}
    ).lazy()

    result = stage_hours_weighted_mean(lf, "value", ["line_id"], {1: 0.0, 2: 0.0})

    assert result.height == 0
    assert list(result.columns) == ["line_id", "value"]


# ---------------------------------------------------------------------------
# add_mean_p50_band
# ---------------------------------------------------------------------------


def test_add_mean_p50_band_traces(percentile_df: pd.DataFrame) -> None:
    """Three traces are added: mean (solid), p50 (dashed), and p10-p90 band."""
    fig = go.Figure()
    result = add_mean_p50_band(fig, percentile_df, "stage_id", "Hydro", "#3B82F6")

    # 1 mean + 1 p50 + 2 band traces (p10 lower + p90 upper)
    assert len(result.data) == 4

    mean_trace = result.data[0]
    assert mean_trace.mode == "lines"
    assert mean_trace.line.dash is None or mean_trace.line.dash == "solid"
    assert mean_trace.line.width == 2
    assert mean_trace.name == "Hydro"

    p50_trace = result.data[1]
    assert p50_trace.line.dash == "dash"
    assert p50_trace.line.width == 1.5
    assert p50_trace.opacity == pytest.approx(0.7)

    # Band: p10 lower (no fill) then p90 upper (fill="tonexty")
    p10_trace = result.data[2]
    assert p10_trace.line.width == 0

    p90_trace = result.data[3]
    assert p90_trace.fill == "tonexty"
    assert "rgba" in p90_trace.fillcolor


def test_add_mean_p50_band_returns_figure(percentile_df: pd.DataFrame) -> None:
    """The function returns the same figure object (chaining support)."""
    fig = go.Figure()
    returned = add_mean_p50_band(fig, percentile_df, "stage_id", "Hydro", "#3B82F6")
    assert returned is fig


def test_add_mean_p50_band_is_the_promoted_plotly_helpers_function() -> None:
    """chart_helpers re-exports the helper promoted to ui.html.plotly rather
    than defining its own copy — the two names must be the same object."""
    from novomodelo_bridge.ui.html.plotly import add_mean_p50_band as _promoted

    assert add_mean_p50_band is _promoted


def test_add_mean_p50_band_empty_df() -> None:
    """Empty DataFrame causes no traces to be added."""
    empty_df = pd.DataFrame(columns=["stage_id", "mean", "p10", "p50", "p90"])
    fig = go.Figure()
    result = add_mean_p50_band(fig, empty_df, "stage_id", "Hydro", "#3B82F6")

    assert len(result.data) == 0


def test_add_mean_p50_band_show_band_false(percentile_df: pd.DataFrame) -> None:
    """When show_band=False, only mean and p50 traces are added."""
    fig = go.Figure()
    add_mean_p50_band(
        fig, percentile_df, "stage_id", "Hydro", "#3B82F6", show_band=False
    )
    assert len(fig.data) == 2


def test_add_mean_p50_band_show_p50_false(percentile_df: pd.DataFrame) -> None:
    """When show_p50=False, only mean and band traces are added."""
    fig = go.Figure()
    add_mean_p50_band(
        fig, percentile_df, "stage_id", "Hydro", "#3B82F6", show_p50=False
    )
    # mean + p10 lower + p90 upper = 3
    assert len(fig.data) == 3


# ---------------------------------------------------------------------------
# make_chart_card
# ---------------------------------------------------------------------------


def test_make_chart_card_html() -> None:
    """Output contains the chart-card class and figure HTML."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))

    html = make_chart_card(fig, "Test Chart", "chart-test-001")

    assert "chart-card" in html
    assert "chart-test-001" in html
    # fig_to_html produces a div element
    assert "<div" in html


def test_make_chart_card_none_raises() -> None:
    """Passing None as fig raises ValueError."""
    with pytest.raises(ValueError, match="the figure is missing"):
        make_chart_card(None, "Title", "chart-id")


def test_make_chart_card_contains_expand_button() -> None:
    """The expand button SVG (from wrap_chart) is present in the output."""
    fig = go.Figure()
    html = make_chart_card(fig, "My Chart", "chart-abc", height=300)

    assert "expand-btn" in html


def test_make_chart_card_no_plotlyjs() -> None:
    """The output must not embed the plotly.js script tag."""
    fig = go.Figure()
    html = make_chart_card(fig, "Chart", "chart-xyz")

    assert "cdn.plot.ly" not in html
    assert "plotly.min.js" not in html
    assert "include_plotlyjs" not in html


# ---------------------------------------------------------------------------
# compute_npv_costs
# ---------------------------------------------------------------------------


def test_compute_npv_costs_basic() -> None:
    """Cost columns are multiplied by the parquet's ``discount_factor`` column."""
    df = pd.DataFrame(
        {
            "stage_id": [0, 1, 2],
            "scenario_id": [0, 0, 0],
            "discount_factor": [1.0, 1 / 1.1, 1 / 1.21],
            "thermal_generation_cost": [100.0, 100.0, 100.0],
        }
    )
    result = compute_npv_costs(df, discount_rate=0.10)

    assert result["thermal_generation_cost"].iloc[0] == pytest.approx(100.0)
    assert result["thermal_generation_cost"].iloc[1] == pytest.approx(
        100.0 / 1.1, rel=1e-4
    )
    assert result["thermal_generation_cost"].iloc[2] == pytest.approx(
        100.0 / (1.1**2), rel=1e-4
    )


def test_compute_npv_costs_unit_discount_factor_leaves_values_unchanged() -> None:
    """When ``discount_factor`` is 1.0 for every row costs are unchanged."""
    df = pd.DataFrame(
        {
            "stage_id": [0, 1, 2],
            "scenario_id": [0, 0, 0],
            "discount_factor": [1.0, 1.0, 1.0],
            "thermal_generation_cost": [50.0, 75.0, 100.0],
        }
    )
    result = compute_npv_costs(df, discount_rate=0.0)

    pd.testing.assert_series_equal(
        result["thermal_generation_cost"],
        df["thermal_generation_cost"],
    )


def test_compute_npv_costs_missing_discount_factor_returns_unmodified_copy() -> None:
    """Without ``discount_factor`` the helper returns a copy, costs untouched."""
    df = pd.DataFrame(
        {
            "stage_id": [0, 1, 2],
            "scenario_id": [0, 0, 0],
            "thermal_generation_cost": [100.0, 100.0, 100.0],
        }
    )
    result = compute_npv_costs(df, discount_rate=0.10)

    pd.testing.assert_frame_equal(result, df)


def test_compute_npv_costs_empty() -> None:
    """Empty DataFrame input returns an empty DataFrame (no error)."""
    empty = pd.DataFrame(
        columns=[
            "stage_id",
            "scenario_id",
            "discount_factor",
            "thermal_generation_cost",
        ]
    )
    result = compute_npv_costs(empty, discount_rate=0.10)

    assert result.empty
    assert list(result.columns) == list(empty.columns)


def test_compute_npv_costs_does_not_mutate() -> None:
    """Input DataFrame is not modified in place."""
    df = pd.DataFrame(
        {
            "stage_id": [0, 1],
            "discount_factor": [1.0, 0.9],
            "thermal_generation_cost": [100.0, 100.0],
        }
    )
    original_values = df["thermal_generation_cost"].tolist()
    compute_npv_costs(df, discount_rate=0.10)

    assert df["thermal_generation_cost"].tolist() == original_values


def test_compute_npv_costs_metadata_cols_unchanged() -> None:
    """scenario_id, stage_id, block_id, and discount_factor are never discounted."""
    df = pd.DataFrame(
        {
            "stage_id": [0, 1, 2],
            "scenario_id": [7, 7, 7],
            "block_id": [1, 1, 1],
            "discount_factor": [1.0, 0.9, 0.8],
            "thermal_generation_cost": [100.0, 100.0, 100.0],
        }
    )
    result = compute_npv_costs(df, discount_rate=0.10)

    assert result["scenario_id"].tolist() == [7, 7, 7]
    assert result["stage_id"].tolist() == [0, 1, 2]
    assert result["block_id"].tolist() == [1, 1, 1]
    assert result["discount_factor"].tolist() == [1.0, 0.9, 0.8]


# ---------------------------------------------------------------------------
# group_costs
# ---------------------------------------------------------------------------


def test_group_costs_known_groups() -> None:
    """Thermal and Deficit columns (as named in COST_GROUPS) sum into their groups.

    COST_GROUPS maps "Thermal" -> ["thermal_cost"] and "Deficit" ->
    ["deficit_cost"].  The test therefore uses those exact column names.
    """
    df = pd.DataFrame(
        {
            "stage_id": [0],
            "scenario_id": [0],
            "thermal_cost": [60.0],
            "deficit_cost": [20.0],
        }
    )
    cost_cols = [
        "thermal_cost",
        "deficit_cost",
    ]
    result = group_costs(df, cost_cols)

    assert result["Thermal"].iloc[0] == pytest.approx(60.0)
    assert result["Deficit"].iloc[0] == pytest.approx(20.0)
    # Component columns should be gone
    assert "thermal_cost" not in result.columns
    assert "deficit_cost" not in result.columns


def test_group_costs_other_column() -> None:
    """Unrecognized cost columns are accumulated in the 'Other' group."""
    df = pd.DataFrame(
        {
            "stage_id": [0],
            "scenario_id": [0],
            "future_cost": [999.0],
        }
    )
    cost_cols = ["future_cost"]
    result = group_costs(df, cost_cols)

    assert result["Other"].iloc[0] == pytest.approx(999.0)
    assert "future_cost" not in result.columns


def test_group_costs_missing_columns() -> None:
    """Missing columns in COST_GROUPS components do not raise an error.

    COST_GROUPS["Thermal"] maps to ["thermal_cost"].  When only that column
    is present the group is populated correctly.  No component of any other
    group being absent must not raise an error.
    """
    df = pd.DataFrame(
        {
            "stage_id": [0],
            "scenario_id": [0],
            "thermal_cost": [80.0],
        }
    )
    cost_cols = ["thermal_cost"]
    result = group_costs(df, cost_cols)

    # Thermal group contains only the present column.
    assert result["Thermal"].iloc[0] == pytest.approx(80.0)
    # Other should be 0.0 since nothing is unassigned.
    assert result["Other"].iloc[0] == pytest.approx(0.0)


def test_group_costs_empty() -> None:
    """Empty DataFrame returns an empty DataFrame without error."""
    empty = pd.DataFrame(columns=["stage_id", "scenario_id", "thermal_generation_cost"])
    result = group_costs(empty, ["thermal_generation_cost"])

    assert result.empty


def test_group_costs_preserves_metadata() -> None:
    """Non-cost columns (scenario_id, stage_id) are preserved in output."""
    df = pd.DataFrame(
        {
            "stage_id": [0, 1],
            "scenario_id": [3, 3],
            "spillage_cost": [5.0, 10.0],
        }
    )
    result = group_costs(df, ["spillage_cost"])

    assert "stage_id" in result.columns
    assert "scenario_id" in result.columns
    assert result["scenario_id"].tolist() == [3, 3]


# ---------------------------------------------------------------------------
# compute_cost_summary
# ---------------------------------------------------------------------------


def _make_costs_df(n_scenarios: int = 100, n_stages: int = 10) -> pd.DataFrame:
    """Helper: build a costs DataFrame with 100 scenarios x 10 stages."""
    rows = []
    for sc in range(n_scenarios):
        for st in range(n_stages):
            rows.append(
                {
                    "scenario_id": sc,
                    "stage_id": st,
                    "discount_factor": 1 / (1.12**st),
                    "thermal_generation_cost": 50.0 + sc * 0.1,
                    "deficit_cost_depth_1": 20.0 + sc * 0.05,
                    "spillage_cost": 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_cost_summary_shape() -> None:
    """Output has exactly the expected columns and one row per cost group."""
    df = _make_costs_df()
    result = compute_cost_summary(df, 0.12)

    assert list(result.columns) == [
        "group",
        "mean",
        "std",
        "p5",
        "p10",
        "p90",
        "p95",
        "pct",
    ]
    # Groups present: Thermal, Deficit, Spillage, NCS, Violations, Other
    assert len(result) == len(COST_GROUPS)


def test_cost_summary_pct_sums_to_100() -> None:
    """The 'pct' column sums to approximately 100.0 across all groups."""
    df = _make_costs_df()
    result = compute_cost_summary(df, 0.12)

    assert result["pct"].sum() == pytest.approx(100.0, rel=1e-4)


def test_cost_summary_sorted_descending() -> None:
    """Rows are sorted by mean in descending order."""
    df = _make_costs_df()
    result = compute_cost_summary(df, 0.12)

    means = result["mean"].tolist()
    assert means == sorted(means, reverse=True)


def test_cost_summary_empty() -> None:
    """Empty input returns an empty DataFrame with the expected columns."""
    empty = pd.DataFrame(columns=["scenario_id", "stage_id", "thermal_generation_cost"])
    result = compute_cost_summary(empty, 0.12)

    assert result.empty
    assert list(result.columns) == [
        "group",
        "mean",
        "std",
        "p5",
        "p10",
        "p90",
        "p95",
        "pct",
    ]


# ---------------------------------------------------------------------------
# compute_cost_summary — p5 and p95 columns
# ---------------------------------------------------------------------------


def test_cost_summary_has_p5_and_p95_columns() -> None:
    """compute_cost_summary must return columns 'p5' and 'p95'."""
    df = _make_costs_df()
    result = compute_cost_summary(df, 0.12)

    assert "p5" in result.columns
    assert "p95" in result.columns


def test_cost_summary_p5_le_p10_le_mean_le_p90_le_p95() -> None:
    """For every row: p5 <= p10 <= mean <= p90 <= p95."""
    df = _make_costs_df()
    result = compute_cost_summary(df, 0.12)

    # Only check rows with positive mean (zero-cost groups have equal percentiles)
    non_zero = result[result["mean"] > 0]
    for _, row in non_zero.iterrows():
        assert row["p5"] <= row["p10"] + 1e-9, f"p5 > p10 for group {row['group']}"
        assert row["p10"] <= row["mean"] + 1e-9, f"p10 > mean for group {row['group']}"
        assert row["mean"] <= row["p90"] + 1e-9, f"mean > p90 for group {row['group']}"
        assert row["p90"] <= row["p95"] + 1e-9, f"p90 > p95 for group {row['group']}"


# ---------------------------------------------------------------------------
# COST_GROUP_COLORS completeness
# ---------------------------------------------------------------------------


def test_cost_group_colors_keys() -> None:
    """Every key in COST_GROUPS has a corresponding colour in COST_GROUP_COLORS."""
    for group_name in COST_GROUPS:
        assert group_name in COST_GROUP_COLORS, (
            f"COST_GROUP_COLORS is missing an entry for group '{group_name}'"
        )


# ---------------------------------------------------------------------------
# build_cost_table
# ---------------------------------------------------------------------------


def test_build_cost_table_contains_table_and_thermal() -> None:
    """build_cost_table must return HTML containing <table and group names."""
    df = _make_costs_df()
    summary = compute_cost_summary(df, 0.12)
    html = build_cost_table(summary)

    assert "<table" in html
    assert "Thermal" in html


def test_build_cost_table_contains_data_table_class() -> None:
    """build_cost_table must return HTML with class 'data-table'."""
    df = _make_costs_df()
    summary = compute_cost_summary(df, 0.12)
    html = build_cost_table(summary)
    assert 'class="data-table"' in html


def test_build_cost_table_has_tbody_with_rows() -> None:
    """build_cost_table must include a <tbody> with at least one <tr>."""
    df = _make_costs_df()
    summary = compute_cost_summary(df, 0.12)
    html = build_cost_table(summary)
    assert "<tbody>" in html
    assert "<tr>" in html


def test_build_cost_table_empty_df_returns_placeholder() -> None:
    """build_cost_table on an empty DataFrame must return the <p> fallback."""
    html = build_cost_table(pd.DataFrame())
    assert "<table" not in html
    assert "No cost data" in html


# ---------------------------------------------------------------------------
# chart_cost_bar
# ---------------------------------------------------------------------------


def test_chart_cost_bar_returns_figure() -> None:
    """chart_cost_bar must return a plotly Figure."""
    df = _make_costs_df()
    summary = compute_cost_summary(df, 0.12)
    fig = chart_cost_bar(summary)
    assert isinstance(fig, go.Figure)


def test_chart_cost_bar_has_vertical_bar_traces() -> None:
    """chart_cost_bar must produce at least one vertical Bar trace."""
    df = _make_costs_df()
    summary = compute_cost_summary(df, 0.12)
    fig = chart_cost_bar(summary)
    bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
    assert len(bar_traces) >= 1
    # Bars are vertical: orientation is None (default) or "v", never "h"
    for trace in bar_traces:
        assert trace.orientation != "h"


def test_chart_cost_bar_error_bars_p5_p95() -> None:
    """chart_cost_bar must set error_y with the p5-p95 range on each bar trace.

    Given p5=800, mean=1000, p95=1200, the trace must have
    error_y.array=[200] and error_y.arrayminus=[200].
    """
    summary = pd.DataFrame(
        {
            "group": ["Thermal"],
            "mean": [1000.0],
            "std": [100.0],
            "p5": [800.0],
            "p10": [850.0],
            "p90": [1150.0],
            "p95": [1200.0],
            "pct": [100.0],
        }
    )
    fig = chart_cost_bar(summary)

    bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
    assert len(bar_traces) == 1
    trace = bar_traces[0]
    assert trace.error_y is not None
    assert trace.error_y.visible is True
    assert trace.error_y.array == (200.0,)
    assert trace.error_y.arrayminus == (200.0,)


def test_chart_cost_bar_error_bars_omitted_when_nan() -> None:
    """chart_cost_bar must omit error_y when p5 or p95 is NaN."""
    summary = pd.DataFrame(
        {
            "group": ["Thermal"],
            "mean": [1000.0],
            "std": [0.0],
            "p5": [math.nan],
            "p10": [math.nan],
            "p90": [math.nan],
            "p95": [math.nan],
            "pct": [100.0],
        }
    )
    fig = chart_cost_bar(summary)

    bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
    assert len(bar_traces) == 1
    assert bar_traces[0].error_y is None or bar_traces[0].error_y.visible is not True


# ---------------------------------------------------------------------------
# comparators.charts golden-string parity
#
# These tests guard that re-pointing the per-stage / percentile-band
# aggregation in ``novomodelo_bridge.comparators.charts`` onto the analyze-layer
# primitives (``analyze.aggregate_percentile_band`` /
# ``per_stage_sum_from_results`` / ``per_stage_sum_from_frame``) leaves the
# rendered HTML character-for-character identical.
#
# The golden files under ``tests/golden/`` were captured from the LEGACY
# (pre-re-point) ``charts.py`` on the fixtures below. The only non-deterministic
# part of the output is the random ``chart-<hex>`` div id emitted by
# ``plotly_div`` (a fresh uuid per render, unrelated to the payload); it is
# normalised away by ``_strip_chart_id`` before comparison so the assertion
# tests the numeric/structural payload only. Regenerate via
# ``scripts/regen-goldens.sh``.
# ---------------------------------------------------------------------------


def _rc(
    entity_type: str,
    name: str,
    novomodelo_id: int,
    stage: int,
    variable: str,
    nw: float,
    cb: float,
) -> ResultComparison:
    abs_diff = abs(nw - cb)
    rel = abs_diff / abs(nw) if abs(nw) > 1e-10 else None
    return ResultComparison(
        entity_type=entity_type,
        entity_name=name,
        newave_code=novomodelo_id + 10,
        novomodelo_id=novomodelo_id,
        stage=stage,
        variable=variable,
        newave_value=nw,
        novomodelo_value=cb,
        abs_diff=abs_diff,
        rel_diff=rel,
    )


@pytest.fixture()
def parity_results() -> list[ResultComparison]:
    """Hydro/thermal/bus comparison rows spanning two stages."""
    return [
        _rc("hydro", "H1", 0, 1, "storage_final_hm3", 100.0, 90.0),
        _rc("hydro", "H2", 1, 1, "storage_final_hm3", 200.0, 180.0),
        _rc("hydro", "H1", 0, 2, "storage_final_hm3", 110.0, 95.0),
        _rc("hydro", "H2", 1, 2, "storage_final_hm3", 210.0, 185.0),
        _rc("thermal", "T1", 5, 1, "generation_mw", 50.0, 45.0),
        _rc("thermal", "T2", 6, 1, "generation_mw", 60.0, 55.0),
        _rc("thermal", "T1", 5, 2, "generation_mw", 52.0, 47.0),
        _rc("bus", "B1", 3, 1, "marginal_cost", 70.0, 65.0),
        _rc("bus", "B2", 4, 1, "marginal_cost", 80.0, 75.0),
        _rc("bus", "B1", 3, 2, "marginal_cost", 72.0, 67.0),
    ]


@pytest.fixture()
def hydro_pct() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 1],
            "stage_id": [1, 1, 2, 2],
            "storage_final_hm3_p10": [80.0, 160.0, 85.0, 165.0],
            "storage_final_hm3_p50": [90.0, 180.0, 95.0, 185.0],
            "storage_final_hm3_p90": [100.0, 200.0, 105.0, 205.0],
        }
    )


@pytest.fixture()
def thermal_pct() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [5, 6, 5],
            "stage_id": [1, 1, 2],
            "generation_mw_p10": [40.0, 50.0, 42.0],
            "generation_mw_p50": [45.0, 55.0, 47.0],
            "generation_mw_p90": [50.0, 60.0, 52.0],
        }
    )


@pytest.fixture()
def bus_pct() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [3, 4, 3],
            "stage_id": [1, 1, 2],
            "marginal_cost_p10": [60.0, 70.0, 62.0],
            "marginal_cost_p50": [65.0, 75.0, 67.0],
            "marginal_cost_p90": [70.0, 80.0, 72.0],
        }
    )


def test_thermal_generation_chart_html_matches_golden(
    parity_results: list[ResultComparison],
    thermal_pct: pl.DataFrame,
) -> None:
    html = _cmp_charts.thermal_generation_chart(parity_results, thermal_pct)
    assert_html_golden(html, "thermal_generation_chart.html")


def test_hydro_aggregate_chart_html_matches_golden(
    parity_results: list[ResultComparison],
    hydro_pct: pl.DataFrame,
) -> None:
    html = _cmp_charts.hydro_aggregate_chart(
        parity_results, "storage_final_hm3", "Hydro Storage", hydro_pct
    )
    assert_html_golden(html, "hydro_aggregate_chart.html")


def test_system_comparison_chart_html_matches_golden(
    parity_results: list[ResultComparison],
    bus_pct: pl.DataFrame,
) -> None:
    html = _cmp_charts.system_comparison_chart(
        parity_results, "marginal_cost", "Bus Marginal Cost", bus_pct
    )
    assert_html_golden(html, "system_comparison_chart.html")


# ---------------------------------------------------------------------------
# per-bus roll-up + per-plant percentile golden-string parity
#
# Guards that re-pointing the per-bus aggregation (hydro_per_bus_chart,
# hydro_slack_per_bus_chart) and the per-plant percentile extraction
# (_enrich_with_percentiles, used by the hydro/thermal detail tabs) onto the
# analyze-layer functions leaves the rendered HTML character-for-character
# identical. The golden files were captured from the LEGACY (pre-re-point)
# charts.py on the fixtures below; the random chart-<hex> div id is normalised
# away by _strip_chart_id before comparison.
# ---------------------------------------------------------------------------


@pytest.fixture()
def per_bus_results() -> list[ResultComparison]:
    """Hydro/thermal rows over two stages and two non-fictitious buses.

    Plant id 2 (H3) is owned by a NOFICT bus so the roll-up must drop it.
    """
    return [
        _rc("hydro", "H1", 0, 1, "storage_final_hm3", 100.0, 90.0),
        _rc("hydro", "H2", 1, 1, "storage_final_hm3", 200.0, 180.0),
        _rc("hydro", "H1", 0, 2, "storage_final_hm3", 110.0, 95.0),
        _rc("hydro", "H2", 1, 2, "storage_final_hm3", 210.0, 185.0),
        _rc("hydro", "H3", 2, 1, "storage_final_hm3", 50.0, 48.0),
        _rc("hydro", "H3", 2, 2, "storage_final_hm3", 55.0, 52.0),
        _rc("hydro", "H1", 0, 1, "generation_mw", 30.0, 28.0),
        _rc("hydro", "H1", 0, 2, "generation_mw", 32.0, 29.0),
        _rc("thermal", "T1", 5, 1, "generation_mw", 50.0, 45.0),
        _rc("thermal", "T2", 6, 1, "generation_mw", 60.0, 55.0),
        _rc("thermal", "T1", 5, 2, "generation_mw", 52.0, 47.0),
    ]


@pytest.fixture()
def per_bus_hydro_meta() -> dict[int, dict]:
    return {
        0: {"bus_ids": {100}},
        1: {"bus_ids": {101}},
        2: {"bus_ids": {199}},  # NOFICT
    }


@pytest.fixture()
def per_bus_bus_meta() -> dict[int, dict]:
    return {
        100: {"name": "SUDESTE"},
        101: {"name": "SUL"},
        199: {"name": "NOFICT1"},
    }


@pytest.fixture()
def per_bus_hydro_pct() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 1],
            "stage_id": [1, 1, 2, 2],
            "storage_final_hm3_p10": [80.0, 160.0, 85.0, 165.0],
            "storage_final_hm3_p50": [90.0, 180.0, 95.0, 185.0],
            "storage_final_hm3_p90": [100.0, 200.0, 105.0, 205.0],
            "generation_mw_p10": [25.0, 0.0, 26.0, 0.0],
            "generation_mw_p50": [28.0, 0.0, 29.0, 0.0],
            "generation_mw_p90": [31.0, 0.0, 33.0, 0.0],
        }
    )


@pytest.fixture()
def slack_novomodelo_hydro() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 1, 2],
            "stage_id": [1, 1, 2, 2, 1],
            "water_withdrawal_violation_pos_m3s": [5.0, 3.0, 6.0, 4.0, 9.0],
        }
    )


@pytest.fixture()
def slack_nw() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 1],
            "stage_id": [1, 1, 2, 2],
            "water_withdrawal_violation_pos_m3s": [4.5, 2.5, 5.5, 3.5],
        }
    )


@pytest.fixture()
def slack_pct() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 1],
            "stage_id": [1, 1, 2, 2],
            "water_withdrawal_violation_pos_m3s_p10": [3.0, 2.0, 4.0, 3.0],
            "water_withdrawal_violation_pos_m3s_p90": [7.0, 5.0, 8.0, 6.0],
        }
    )


@pytest.fixture()
def detail_novomodelo_hydro() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [0, 0, 1, 1],
            "stage_id": [1, 2, 1, 2],
            "stored_energy_initial_mwh": [1000.0, 1100.0, 2000.0, 2100.0],
            "stored_energy_final_mwh": [900.0, 1000.0, 1900.0, 2000.0],
            "water_withdrawal_violation_pos_m3s": [5.0, 6.0, 3.0, 4.0],
        }
    )


# read_novomodelo_hydro_metadata no longer carries a plant
# "bus_id"; the per-bus roll-up now sources the plant->bus label
# from "bus_ids" (see analyze._bus_name_lookups), merged onto hydro_meta by
# the results-comparison orchestrator from the hydro_bus_generation
# partition. ``per_bus_hydro_meta`` uses the "bus_ids" shape,
# one bus per plant (0->100, 1->101, 2->199), so every plant here
# still resolves to exactly the bus it did under the legacy "bus_id" shape --
# a value-preserving migration that renders the byte-identical golden HTML.
def test_hydro_per_bus_chart_html_matches_golden(
    per_bus_results: list[ResultComparison],
    per_bus_hydro_pct: pl.DataFrame,
    per_bus_hydro_meta: dict[int, dict],
    per_bus_bus_meta: dict[int, dict],
) -> None:
    html = _cmp_charts.hydro_per_bus_chart(
        per_bus_results,
        "storage_final_hm3",
        "Hydro Storage by Bus",
        per_bus_hydro_pct,
        per_bus_hydro_meta,
        per_bus_bus_meta,
    )
    assert_html_golden(html, "hydro_per_bus_chart.html")


def test_hydro_slack_per_bus_chart_html_matches_golden(
    slack_novomodelo_hydro: pl.DataFrame,
    slack_nw: pl.DataFrame,
    slack_pct: pl.DataFrame,
    per_bus_hydro_meta: dict[int, dict],
    per_bus_bus_meta: dict[int, dict],
) -> None:
    html = _cmp_charts.hydro_slack_per_bus_chart(
        slack_novomodelo_hydro,
        slack_nw,
        "water_withdrawal_violation_pos_m3s",
        "Withdrawal Slack by Bus",
        slack_pct,
        per_bus_hydro_meta,
        per_bus_bus_meta,
        {0, 1, 2},
    )
    assert_html_golden(html, "hydro_slack_per_bus_chart.html")


# ---------------------------------------------------------------------------
# synthetic two-bus plant coverage at the chart-rendering
# level -- the analyze-layer exclusion/diagnostic behaviour is unit-tested
# directly in test_analyze.py; these confirm the chart builders that consume
# it render the single-bus plant's panel and simply omit the ambiguous one,
# rather than crashing or double-counting it into both bus panels.
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_bus_hydro_meta() -> dict[int, dict]:
    return {
        0: {"bus_ids": {100}},
        9: {"bus_ids": {100, 101}},  # synthetic two-bus plant.
    }


@pytest.fixture()
def two_bus_bus_meta() -> dict[int, dict]:
    return {100: {"name": "SUDESTE"}, 101: {"name": "SUL"}}


def test_hydro_per_bus_chart_excludes_multi_bus_plant(
    two_bus_hydro_meta: dict[int, dict],
    two_bus_bus_meta: dict[int, dict],
) -> None:
    results = [
        _rc("hydro", "H1", 0, 1, "storage_final_hm3", 100.0, 90.0),
        _rc("hydro", "H9", 9, 1, "storage_final_hm3", 500.0, 480.0),
    ]
    html = _cmp_charts.hydro_per_bus_chart(
        results,
        "storage_final_hm3",
        "Hydro Storage by Bus",
        None,
        two_bus_hydro_meta,
        two_bus_bus_meta,
    )
    # SUDESTE (plant 0) renders; SUL never appears since plant 9's value was
    # excluded rather than collapsed into it or double-counted across both.
    assert "SUDESTE" in html
    assert "SUL" not in html


def test_hydro_slack_per_bus_chart_excludes_multi_bus_plant(
    two_bus_hydro_meta: dict[int, dict],
    two_bus_bus_meta: dict[int, dict],
) -> None:
    novomodelo_hydro = pl.DataFrame(
        {
            "entity_id": [0, 9],
            "stage_id": [1, 1],
            "water_withdrawal_violation_pos_m3s": [5.0, 50.0],
        }
    )
    html = _cmp_charts.hydro_slack_per_bus_chart(
        novomodelo_hydro,
        None,
        "water_withdrawal_violation_pos_m3s",
        "Withdrawal Slack by Bus",
        None,
        two_bus_hydro_meta,
        two_bus_bus_meta,
        {0, 9},
    )
    assert "SUDESTE" in html
    assert "SUL" not in html


@pytest.fixture()
def line_summary_results() -> list[ResultComparison]:
    """Three lines over two stages -> 3 panels -> multi-row 2-col grid."""
    rows: list[ResultComparison] = []
    specs = [
        (0, "L1", [(1, 100.0, 95.0), (2, 110.0, 104.0)]),
        (1, "L2", [(1, -50.0, -48.0), (2, -55.0, -52.0)]),
        (2, "L3", [(1, 20.0, 18.0), (2, 25.0, 23.0)]),
    ]
    for cid, name, pts in specs:
        for stage, nw, cb in pts:
            rows.append(_rc("line", name, cid, stage, "net_flow_mw", nw, cb))
    return rows


@pytest.fixture()
def line_summary_pct() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [0, 0, 1, 1, 2, 2],
            "stage_id": [1, 2, 1, 2, 1, 2],
            "net_flow_mw_p10": [80.0, 90.0, -60.0, -65.0, 15.0, 20.0],
            "net_flow_mw_p90": [120.0, 130.0, -40.0, -45.0, 25.0, 30.0],
        }
    )


@pytest.fixture()
def line_summary_bounds() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "line_id": [0, 0, 1, 1, 2, 2],
            "stage_id": [1, 2, 1, 2, 1, 2],
            "direct_mw": [200.0, 200.0, 150.0, 150.0, 100.0, 100.0],
            "reverse_mw": [180.0, 180.0, 120.0, 120.0, 90.0, 90.0],
        }
    )


@pytest.fixture()
def line_summary_meta() -> list[dict]:
    return [
        {"id": 0, "capacity": {"direct_mw": 200.0, "reverse_mw": 180.0}},
        {"id": 1, "capacity": {"direct_mw": 150.0, "reverse_mw": 120.0}},
        {"id": 2, "capacity": {"direct_mw": 100.0, "reverse_mw": 90.0}},
    ]


def test_line_summary_chart_html_matches_golden(
    line_summary_results: list[ResultComparison],
    line_summary_pct: pl.DataFrame,
    line_summary_bounds: pl.DataFrame,
    line_summary_meta: list[dict],
) -> None:
    html = _cmp_charts.line_summary_chart(
        line_summary_results,
        line_summary_pct,
        line_summary_bounds,
        line_summary_meta,
    )
    assert_html_golden(html, "line_summary_chart.html")
    # Anti-silent-blank guard: the overlay's two capacity
    # traces must actually be present, not just byte-match an empty chart.
    assert "Upper bound" in html
    assert "Lower bound" in html


def test_build_hydro_detail_tab_html_matches_golden(
    per_bus_results: list[ResultComparison],
    per_bus_hydro_pct: pl.DataFrame,
    detail_novomodelo_hydro: pl.DataFrame,
) -> None:
    html = report_builder.build_hydro_detail_tab(
        per_bus_results,
        per_bus_hydro_pct,
        detail_novomodelo_hydro,
    )
    assert_html_golden(html, "build_hydro_detail_tab.html")


def test_build_thermal_detail_tab_html_matches_golden(
    per_bus_results: list[ResultComparison],
    thermal_pct: pl.DataFrame,
) -> None:
    html = report_builder.build_thermal_detail_tab(per_bus_results, thermal_pct)
    assert_html_golden(html, "build_thermal_detail_tab.html")


# ---------------------------------------------------------------------------
# system/network draw-only golden-string parity
#
# Guards that re-pointing the Novomodelo-sum + the source-model-SIN fold
# (novomodelo_aggregate_chart), the per-bus grouping + per-eid percentile lookup
# (system_per_bus_chart), and the spillage nw/cb lookups (system_spillage_energy_chart)
# onto the analyze-layer functions (novomodelo_sum_and_newave_sin / bus_groups_and_pct /
# spillage_lookups) leaves the rendered HTML character-for-character identical. The
# golden files were captured from the LEGACY (pre-re-point) charts.py on the fixtures
# below; the random chart-<hex> div id is normalised away by _strip_chart_id.
# ---------------------------------------------------------------------------


@pytest.fixture()
def agg_novomodelo_hydro() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 1],
            "stage_id": [1, 1, 2, 2],
            "stored_energy_final_mwh": [900.0, 1900.0, 1000.0, 2000.0],
        }
    )


@pytest.fixture()
def agg_pct() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 1],
            "stage_id": [1, 1, 2, 2],
            "stored_energy_final_mwh_p10": [850.0, 1850.0, 950.0, 1950.0],
            "stored_energy_final_mwh_p90": [950.0, 1950.0, 1050.0, 2050.0],
        }
    )


@pytest.fixture()
def agg_nw_sin() -> pl.DataFrame:
    # The middle row exercises the strip()/upper() filter on " earmf ".
    return pl.DataFrame(
        {
            "newave_code": [0, 0, 0],
            "stage": [2, 3, 4],
            "variable": ["EARMF", " earmf ", "EARMF"],
            "value": [3.0, 4.0, 5.0],
        }
    )


def test_novomodelo_aggregate_chart_html_matches_golden(
    agg_novomodelo_hydro: pl.DataFrame,
    agg_pct: pl.DataFrame,
    agg_nw_sin: pl.DataFrame,
) -> None:
    html = _cmp_charts.novomodelo_aggregate_chart(
        agg_novomodelo_hydro,
        "stored_energy_final_mwh",
        "Stored Energy SIN",
        "MWh",
        agg_pct,
        nw_sin=agg_nw_sin,
        nw_variable="EARMF",
        nw_factor=730.0,
        nw_offset=1,
        matched_ids=None,
    )
    assert_html_golden(html, "novomodelo_aggregate_chart.html")


@pytest.fixture()
def per_bus_system_results() -> list[ResultComparison]:
    return [
        _rc("bus", "SUDESTE", 0, 1, "deficit_mw", 10.0, 9.0),
        _rc("bus", "SUDESTE", 0, 2, "deficit_mw", 12.0, 11.0),
        _rc("bus", "SUL", 1, 1, "deficit_mw", 5.0, 4.0),
        _rc("bus", "SUL", 1, 2, "deficit_mw", 6.0, 5.0),
        _rc("bus", "NORTE", 2, 1, "deficit_mw", 3.0, 2.0),
        _rc("bus", "NORTE", 2, 2, "deficit_mw", 4.0, 3.0),
    ]


@pytest.fixture()
def per_bus_system_pct() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [0, 0, 1, 1, 2, 2],
            "stage_id": [1, 2, 1, 2, 1, 2],
            "deficit_mw_p10": [8.0, 10.0, 4.0, 5.0, 2.0, 3.0],
            "deficit_mw_p90": [11.0, 13.0, 6.0, 7.0, 4.0, 5.0],
        }
    )


def test_system_per_bus_chart_html_matches_golden(
    per_bus_system_results: list[ResultComparison],
    per_bus_system_pct: pl.DataFrame,
) -> None:
    html = _cmp_charts.system_per_bus_chart(
        per_bus_system_results, "deficit_mw", "Bus Deficit", per_bus_system_pct
    )
    assert_html_golden(html, "system_per_bus_chart.html")


@pytest.fixture()
def spill_results() -> list[ResultComparison]:
    return [
        _rc("system_spillage", "SIN", 0, 1, "VERTOT", 100.0, 95.0),
        _rc("system_spillage", "SIN", 0, 2, "VERTOT", 110.0, 105.0),
        _rc("system_spillage", "SIN", 0, 1, "VERTcont", 60.0, 58.0),
        _rc("system_spillage", "SIN", 0, 2, "VERTcont", 65.0, 63.0),
        _rc("system_spillage", "SIN", 0, 1, "VERTfio", 40.0, 37.0),
        _rc("system_spillage", "SIN", 0, 2, "VERTfio", 45.0, 42.0),
    ]


@pytest.fixture()
def novomodelo_spill_energy() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "stage_id": [1, 2],
            "total_mw": [96.0, 106.0],
            "reservoir_mw": [58.0, 63.0],
            "rorov_mw": [38.0, 43.0],
        }
    )


def test_system_spillage_energy_chart_html_matches_golden(
    spill_results: list[ResultComparison],
    novomodelo_spill_energy: pl.DataFrame,
) -> None:
    html = _cmp_charts.system_spillage_energy_chart(
        spill_results, novomodelo_spill_energy
    )
    assert_html_golden(html, "system_spillage_energy_chart.html")


# ---------------------------------------------------------------------------
# build_comparison_report dataset-seam golden parity
#
# Guards that re-pointing ``build_comparison_report`` onto the
# ``ComparisonDataset`` seam (it now takes the dataset plus explicit
# ``results``/``pct`` keyword params) leaves the rendered HTML
# character-for-character identical to the pre-change output.
#
# The golden ``tests/golden/build_comparison_report_full.html`` was captured
# from the LEGACY ``build_comparison_report(results, pctiles)`` call on the
# fixtures below; the random ``chart-<hex>`` div id and the plotly.js CDN
# version are normalised by ``_strip_chart_id`` before comparison.
# ---------------------------------------------------------------------------


def _report_fixture_results() -> list[ResultComparison]:
    """Representative comparison rows across entity types (golden fixture)."""
    return [
        ResultComparison(
            entity_type="hydro",
            entity_name="PLANT_A",
            newave_code=1,
            novomodelo_id=0,
            stage=0,
            variable="generation_mw",
            newave_value=1200.0,
            novomodelo_value=1195.0,
            abs_diff=5.0,
            rel_diff=0.004,
        ),
        ResultComparison(
            entity_type="hydro",
            entity_name="PLANT_A",
            newave_code=1,
            novomodelo_id=0,
            stage=1,
            variable="storage_final_hm3",
            newave_value=4500.0,
            novomodelo_value=4510.0,
            abs_diff=10.0,
            rel_diff=0.002,
        ),
        ResultComparison(
            entity_type="thermal",
            entity_name="GAS_A",
            newave_code=10,
            novomodelo_id=0,
            stage=0,
            variable="generation_mw",
            newave_value=300.0,
            novomodelo_value=298.0,
            abs_diff=2.0,
            rel_diff=0.007,
        ),
        ResultComparison(
            entity_type="bus",
            entity_name="SE",
            newave_code=1,
            novomodelo_id=0,
            stage=0,
            variable="spot_price",
            newave_value=150.0,
            novomodelo_value=152.0,
            abs_diff=2.0,
            rel_diff=0.013,
        ),
        ResultComparison(
            entity_type="convergence",
            entity_name="iteration_1",
            newave_code=1,
            novomodelo_id=1,
            stage=1,
            variable="lower_bound",
            newave_value=50000.0,
            novomodelo_value=50100.0,
            abs_diff=100.0,
            rel_diff=0.002,
        ),
    ]


def _report_fixture_pct() -> PercentileData:
    """PercentileData with non-empty hydro/thermal frames (golden fixture)."""
    hydro_df = pl.DataFrame(
        {
            "entity_id": [0, 0, 0],
            "stage_id": [0, 1, 2],
            "generation_mw_p10": [1000.0, 1050.0, 1100.0],
            "generation_mw_p50": [1200.0, 1250.0, 1300.0],
            "generation_mw_p90": [1400.0, 1450.0, 1500.0],
            "storage_final_hm3_p10": [4000.0, 4100.0, 4200.0],
            "storage_final_hm3_p50": [4500.0, 4550.0, 4600.0],
            "storage_final_hm3_p90": [5000.0, 5050.0, 5100.0],
        }
    )
    thermal_df = pl.DataFrame(
        {
            "entity_id": [0, 0, 0],
            "stage_id": [0, 1, 2],
            "generation_mw_p10": [250.0, 260.0, 270.0],
            "generation_mw_p50": [300.0, 310.0, 320.0],
            "generation_mw_p90": [350.0, 360.0, 370.0],
        }
    )
    return PercentileData(hydro=hydro_df, thermal=thermal_df)


def test_build_comparison_report_dataset_golden() -> None:
    """The dataset seam renders byte-identically to the legacy signature."""
    from novomodelo_bridge.comparators.analyze import build_results_dataset
    from novomodelo_bridge.comparators.report_builder import build_comparison_report

    results = _report_fixture_results()
    pct = _report_fixture_pct()

    dataset = build_results_dataset(results, pct, 0.05)
    html = build_comparison_report(dataset)

    assert_html_golden(html, "build_comparison_report_full.html")


# ---------------------------------------------------------------------------
# per-tab metadata-drain golden parity
#
# Guards that re-pointing the Overview / System / Energy-Balance / Network tab
# blocks of ``build_comparison_report`` to read their frame/dict/list/int args
# from ``dataset.metadata`` named keys (instead of the monolithic ``pct``
# object) leaves each tab's rendered HTML character-for-character identical. The
# golden files were captured from the LEGACY (pre-re-point) report_builder on
# the shared ``_report_fixture_results`` / ``_report_fixture_pct`` fixtures; the
# random ``chart-<hex>`` div id is normalised away by ``_strip_chart_id``.
# ---------------------------------------------------------------------------


def _extract_tab_content(html: str, tab_id: str) -> str:
    """Return the inner HTML of the ``<section id="{tab_id}">`` (== tab_contents)."""
    pat = re.compile(
        r'<section id="' + re.escape(tab_id) + r'" class="tab-content[^"]*">\n'
        r"(.*?)\n</section>",
        re.DOTALL,
    )
    match = pat.search(html)
    assert match is not None, f"tab section {tab_id} not found in report HTML"
    return match.group(1)


@pytest.mark.parametrize(
    ("tab_id", "golden_name"),
    [
        ("tab-overview", "report_tab_overview.html"),
        ("tab-system", "report_tab_system.html"),
        ("tab-balance", "report_tab_balance.html"),
        ("tab-network", "report_tab_network.html"),
    ],
)
def test_report_tab_matches_golden(tab_id: str, golden_name: str) -> None:
    """Each migrated tab renders byte-identically after the metadata drain."""
    from novomodelo_bridge.comparators.analyze import build_results_dataset
    from novomodelo_bridge.comparators.report_builder import build_comparison_report

    results = _report_fixture_results()
    pct = _report_fixture_pct()

    dataset = build_results_dataset(results, pct, 0.05)
    html = build_comparison_report(dataset)
    content = _extract_tab_content(html, tab_id)

    assert_html_golden(content, golden_name)


# ---------------------------------------------------------------------------
# Hydro Operation / Hydro Details metadata-drain golden parity
#
# Guards that re-pointing the Hydro Operation and Hydro Details tab blocks of
# ``build_comparison_report`` to read their frame/dict/int args from
# ``dataset.metadata`` named keys (instead of the monolithic ``pct`` object)
# leaves each tab's rendered HTML character-for-character identical. The golden
# files were captured from the LEGACY (pre-re-point) report_builder on the
# shared ``_report_fixture_results`` / ``_report_fixture_pct`` fixtures; the
# random ``chart-<hex>`` div id is normalised away by ``_strip_chart_id``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tab_id", "golden_name"),
    [
        ("tab-hydro", "report_tab_hydro.html"),
        ("tab-hydro-detail", "report_tab_hydro_detail.html"),
    ],
)
def test_report_hydro_tab_matches_golden(tab_id: str, golden_name: str) -> None:
    """Each hydro tab renders byte-identically after the metadata drain."""
    from novomodelo_bridge.comparators.analyze import build_results_dataset
    from novomodelo_bridge.comparators.report_builder import build_comparison_report

    results = _report_fixture_results()
    pct = _report_fixture_pct()

    dataset = build_results_dataset(results, pct, 0.05)
    html = build_comparison_report(dataset)
    content = _extract_tab_content(html, tab_id)

    assert_html_golden(content, golden_name)


# ---------------------------------------------------------------------------
# Thermal Operation / Thermal Details / Productivity drain parity
#
# Guards that re-pointing the Thermal Operation, Thermal Details and
# Productivity tab blocks of ``build_comparison_report`` to read their frame
# args from ``dataset.metadata`` named keys (``thermal`` /
# ``productivity_detail``) instead of the monolithic ``pct`` object leaves each
# tab's rendered HTML character-for-character identical. The golden files were
# captured from the LEGACY (pre-re-point) report_builder on the shared
# ``_report_fixture_results`` / ``_report_fixture_pct`` fixtures; the random
# ``chart-<hex>`` div id is normalised away by ``_strip_chart_id``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tab_id", "golden_name"),
    [
        ("tab-thermal", "report_tab_thermal.html"),
        ("tab-thermal-detail", "report_tab_thermal_detail.html"),
        ("tab-productivity", "report_tab_productivity.html"),
    ],
)
def test_report_thermal_productivity_tab_matches_golden(
    tab_id: str, golden_name: str
) -> None:
    """Each thermal/productivity tab renders byte-identically after the drain."""
    from novomodelo_bridge.comparators.analyze import build_results_dataset
    from novomodelo_bridge.comparators.report_builder import build_comparison_report

    results = _report_fixture_results()
    pct = _report_fixture_pct()

    dataset = build_results_dataset(results, pct, 0.05)
    html = build_comparison_report(dataset)
    content = _extract_tab_content(html, tab_id)

    assert_html_golden(content, golden_name)


# ---------------------------------------------------------------------------
# Constraints / Performance metadata-drain golden parity
#
# Guards that re-pointing the Constraints and Performance tab blocks of
# ``build_comparison_report`` to read their frame/list/int/float/dict args from
# ``dataset.metadata`` named keys (``gc_constraints`` / ``gc_bounds`` /
# ``gc_lhs_newave`` / ``gc_lhs_novomodelo`` / ``nw_max_stage`` / ``nw_tim_iterations``
# / ``nw_tim_stages`` / ``novomodelo_training_seconds`` / ``novomodelo_iteration_timing``)
# instead of the monolithic ``pct`` object leaves each tab's rendered HTML
# character-for-character identical. The goldens were captured from the LEGACY
# (pre-re-point) report_builder on the ``_constraints_perf_fixture_pct`` fixture
# below; the random ``chart-<hex>`` div id is normalised by ``_strip_chart_id``.
# ---------------------------------------------------------------------------


def _constraints_perf_fixture_pct() -> PercentileData:
    """PercentileData with non-empty constraint + performance frames."""
    # F3, sense-free: both constraints are ``>=`` (lower-only endpoint), matching
    # the pre-F3 fixture's sense so the golden HTML label stays unchanged.
    gc_constraints = [
        {"id": 0, "name": "RE_1"},
        {"id": 1, "name": "AGRINT_1"},
    ]
    gc_bounds = pl.DataFrame(
        {
            "constraint_id": [0, 0, 1, 1],
            "stage_id": [0, 1, 0, 1],
            "block_id": [0, 0, 0, 0],
            "bound_lower": [500.0, 520.0, 300.0, 310.0],
            "bound_upper": [None, None, None, None],
        },
        schema={
            "constraint_id": pl.Int64,
            "stage_id": pl.Int64,
            "block_id": pl.Int64,
            "bound_lower": pl.Float64,
            "bound_upper": pl.Float64,
        },
    )
    gc_lhs_newave = pl.DataFrame(
        {
            "constraint_id": [0, 0, 1, 1],
            "stage_id": [0, 1, 0, 1],
            "lhs_value": [510.0, 525.0, 305.0, 312.0],
        }
    )
    gc_lhs_novomodelo = pl.DataFrame(
        {
            "constraint_id": [0, 0, 1, 1],
            "stage_id": [0, 1, 0, 1],
            "lhs_value": [508.0, 523.0, 304.0, 311.0],
        }
    )
    nw_tim_iterations = pl.DataFrame(
        {
            "iteration": [1, 2, 3],
            "forward_seconds": [10.0, 9.0, 8.5],
            "backward_seconds": [20.0, 18.0, 17.0],
            "total_seconds": [30.0, 27.0, 25.5],
        }
    )
    novomodelo_iteration_timing = pl.DataFrame(
        {
            "iteration": [1, 2, 3],
            "time_total_ms": [25000.0, 24000.0, 23000.0],
            "time_forward_ms": [10000.0, 9500.0, 9000.0],
            "time_backward_ms": [15000.0, 14500.0, 14000.0],
        }
    )
    return PercentileData(
        gc_constraints=gc_constraints,
        gc_bounds=gc_bounds,
        gc_lhs_newave=gc_lhs_newave,
        gc_lhs_novomodelo=gc_lhs_novomodelo,
        nw_max_stage=1,
        nw_tim_iterations=nw_tim_iterations,
        nw_tim_stages={"Tempo Total": 120.0, "Calculo da Politica": 90.0},
        novomodelo_training_seconds=72.0,
        novomodelo_iteration_timing=novomodelo_iteration_timing,
    )


@pytest.mark.parametrize(
    ("tab_id", "golden_name"),
    [
        ("tab-constraints", "report_tab_constraints.html"),
        ("tab-performance", "report_tab_performance.html"),
    ],
)
def test_report_constraints_performance_tab_matches_golden(
    tab_id: str, golden_name: str
) -> None:
    """Constraints/Performance tabs render byte-identically after the drain."""
    from novomodelo_bridge.comparators.analyze import build_results_dataset
    from novomodelo_bridge.comparators.report_builder import build_comparison_report

    results = _report_fixture_results()
    pct = _constraints_perf_fixture_pct()

    dataset = build_results_dataset(results, pct, 0.05)
    html = build_comparison_report(dataset)
    content = _extract_tab_content(html, tab_id)

    assert_html_golden(content, golden_name)


def test_report_productivity_tab_empty_detail_renders_fallback() -> None:
    """Empty ``productivity_detail`` renders the literal no-data fallback."""
    from novomodelo_bridge.comparators.analyze import build_results_dataset
    from novomodelo_bridge.comparators.report_builder import build_comparison_report

    results = _report_fixture_results()
    # The shared fixture leaves ``productivity_detail`` empty by default.
    pct = _report_fixture_pct()
    assert pct.productivity_detail.is_empty()

    dataset = build_results_dataset(results, pct, 0.05)
    html = build_comparison_report(dataset)
    content = _extract_tab_content(html, "tab-productivity")

    assert "No productivity data available." in content


def test_build_comparison_report_empty_dataset_has_all_tabs() -> None:
    """An empty dataset/pct renders all 11 tab ids without raising."""
    from novomodelo_bridge.comparators.analyze import build_results_dataset
    from novomodelo_bridge.comparators.html_report import COMPARISON_TABS
    from novomodelo_bridge.comparators.report_builder import build_comparison_report

    dataset = build_results_dataset([], PercentileData(), 0.05)
    html = build_comparison_report(dataset)

    for tab_id, _ in COMPARISON_TABS:
        assert f'id="{tab_id}"' in html


# ---------------------------------------------------------------------------
# facet_grid subplot-domain helper
#
# These tests pin the `facet_grid` / `FacetPanel` helper in
# `novomodelo_bridge.ui.html.plotly` to the legacy gap-based subplot-domain
# arithmetic hand-copied across `comparators.charts`. They assert the exact
# domain pairs for the representative call sites (the 2-column grids, the
# single-column spillage stack, and the unclamped performance stack including
# the intentional `-0.0`), re-derive the legacy formula inline as an oracle for
# `n` in range(1, 8), and confirm `FacetPanel` is immutable. This is a pure-add
# guard for re-pointing the chart functions onto `facet_grid`.
# ---------------------------------------------------------------------------


def test_facet_grid_default_2col_n4() -> None:
    """facet_grid(4) reproduces the 2-column grid domains panel-for-panel."""
    from novomodelo_bridge.ui.html.plotly import facet_grid

    panels = facet_grid(4)
    assert len(panels) == 4
    assert [(p.x_domain, p.y_domain) for p in panels] == [
        ([0.0, 0.475], [0.53, 1.0]),
        ([0.525, 1.0], [0.53, 1.0]),
        ([0.0, 0.475], [0.0, 0.47]),
        ([0.525, 1.0], [0.0, 0.47]),
    ]


def test_facet_grid_single_col_spillage() -> None:
    """facet_grid(3, ncols=1, row_gap=0.05) matches the spillage layout."""
    from novomodelo_bridge.ui.html.plotly import facet_grid

    panels = facet_grid(3, ncols=1, row_gap=0.05)
    assert all(p.x_domain == [0.0, 1.0] for p in panels)
    assert [p.y_domain for p in panels] == [
        [0.7, 1.0],
        [0.35, 0.65],
        [0.0, 0.3],
    ]


def test_facet_grid_single_col_unclamped_preserves_neg_zero() -> None:
    """The unclamped performance stack preserves the legacy -0.0 y-domain low."""
    from novomodelo_bridge.ui.html.plotly import facet_grid

    panels = facet_grid(2, ncols=1, row_gap=0.10, min_row_h=0.0)
    assert panels[0].y_domain == [0.55, 1.0]
    assert panels[0].x_domain == [0.0, 1.0]
    assert panels[1].y_domain == [-0.0, 0.45]
    assert panels[1].x_domain == [0.0, 1.0]
    # -0.0 == 0.0 is True, so confirm the SIGN explicitly via copysign.
    assert math.copysign(1, panels[1].y_domain[0]) == -1.0


@pytest.mark.parametrize("n", range(1, 8))
def test_facet_grid_matches_inline_formula(n: int) -> None:
    """facet_grid(n) matches the legacy inline formula panel-by-panel."""
    from novomodelo_bridge.ui.html.plotly import facet_grid

    ncols = 2
    row_gap = 0.06
    col_gap = 0.05
    nrows = (n + ncols - 1) // ncols
    row_h = max((1.0 - row_gap * (nrows - 1)) / nrows, 0.001)
    col_w = (1.0 - col_gap * (ncols - 1)) / ncols

    panels = facet_grid(n)
    assert len(panels) == n
    for idx, panel in enumerate(panels):
        row_i = idx // ncols
        col_i = idx % ncols
        x0 = col_i * (col_w + col_gap)
        x1 = x0 + col_w
        y1 = 1.0 - row_i * (row_h + row_gap)
        y0 = y1 - row_h
        assert panel.x_domain == [round(x0, 3), round(x1, 3)]
        assert panel.y_domain == [round(y0, 3), round(y1, 3)]


def test_facet_grid_zero_returns_empty() -> None:
    """facet_grid(0) returns [] per its contract and does not raise."""
    from novomodelo_bridge.ui.html.plotly import facet_grid

    assert facet_grid(0) == []


def test_facet_panel_is_frozen() -> None:
    """FacetPanel is immutable: assigning a field raises FrozenInstanceError."""
    from novomodelo_bridge.ui.html.plotly import FacetPanel

    panel = FacetPanel(row=0, col=0, x_domain=[0.0, 1.0], y_domain=[0.0, 1.0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        panel.row = 1  # type: ignore[misc]
