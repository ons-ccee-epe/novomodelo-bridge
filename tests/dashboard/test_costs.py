"""Unit tests for novomodelo_bridge.dashboard.tabs.costs.

Covers module constants, can_render, _compute_npv_metric, _build_metrics_row,
and the full render() path including the empty-costs degradation branch.

Also covers: _render_cost_composition, _render_category_evolution,
_render_spot_price, _render_violations, and the extended render() output.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import plotly.graph_objects as go
import polars as pl

import novomodelo_bridge.dashboard.tabs.costs as costs_mod
from novomodelo_bridge.dashboard.tabs.costs import (
    _build_composition_data,
    _build_composition_section,
    _build_metrics_row,
    _chart_violation_timeline,
    _compute_npv_metric,
    _render_category_evolution,
    _render_cost_composition,
    _render_spot_price,
    _render_violations,
    can_render,
    render,
)

# ---------------------------------------------------------------------------
# Helpers / data factories
# ---------------------------------------------------------------------------


def _make_costs_df(
    n_scenarios: int = 2,
    n_stages: int = 3,
    thermal_cost: float = 1000.0,
    deficit_cost: float = 50.0,
    *,
    stage_discount: float = 1.0,
) -> pd.DataFrame:
    """Return a minimal costs DataFrame with ``n_scenarios`` and ``n_stages``.

    ``stage_discount`` is the per-stage one-step factor used to populate the
    ``discount_factor`` column (``stage_discount ** stage_id``). The default
    of 1.0 produces undiscounted present values.
    """
    rows = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage_id": stage_id,
                    "discount_factor": stage_discount**stage_id,
                    "thermal_cost": thermal_cost,
                    "deficit_cost": deficit_cost,
                    "immediate_cost": thermal_cost + deficit_cost,
                    "total_cost": thermal_cost + deficit_cost,
                    "future_cost": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _make_mock_data(
    *,
    costs: pd.DataFrame | None = None,
    discount_rate: float = 0.0,
    n_scenarios: int = 2,
    n_stages: int = 3,
) -> MagicMock:
    """Build a minimal MagicMock that satisfies the DashboardData interface.

    Sets proper defaults for all fields accessed by ``render()`` (including
    the extended sections) to avoid MagicMock auto-chaining on polars
    LazyFrames, which causes OOM.
    """
    data = MagicMock()
    data.costs = costs if costs is not None else _make_costs_df(n_scenarios, n_stages)
    data.discount_rate = discount_rate
    data.n_scenarios = n_scenarios
    data.n_stages = n_stages
    data.stage_labels = {i: f"Stage {i}" for i in range(n_stages)}
    data.stage_dates = {i: f"2024-01-{i + 1:02d}" for i in range(n_stages)}
    data.non_fictitious_bus_ids = [0, 1]
    data.bus_names = {0: "Bus A", 1: "Bus B"}
    # Provide real empty polars objects to prevent MagicMock auto-chaining OOM
    data.buses_lf = pl.LazyFrame()
    data.bh_df = pl.DataFrame(
        [{"stage_id": s, "block_id": 0, "_bh": 730.0} for s in range(n_stages)]
    )
    return data


# ---------------------------------------------------------------------------
# test_tab_constants
# ---------------------------------------------------------------------------


def test_tab_constants() -> None:
    """Module-level constants must match their expected values exactly."""
    assert costs_mod.TAB_ID == "tab-costs"
    assert costs_mod.TAB_LABEL == "Costs"
    assert costs_mod.TAB_ORDER == 40


# ---------------------------------------------------------------------------
# test_can_render
# ---------------------------------------------------------------------------


def test_can_render_returns_true_when_simulation_available() -> None:
    """can_render must return True when simulation data is present."""
    data = _make_mock_data()
    data.simulation_available = True
    assert can_render(data) is True


def test_can_render_returns_false_when_simulation_missing() -> None:
    """can_render must return False on training-only cases."""
    data = _make_mock_data()
    data.simulation_available = False
    assert can_render(data) is False


# ---------------------------------------------------------------------------
# test__compute_npv_metric
# ---------------------------------------------------------------------------


def test_compute_npv_metric_undiscounted_returns_mean_per_scenario() -> None:
    """With discount_rate=0.0 and thermal_cost summing to 6000 across 2 scenarios,
    _compute_npv_metric must return 3000.0 (mean across scenarios).

    Each scenario has 3 stages, each with thermal_cost=1000.0, so the per-scenario
    sum is 3000.0. With 2 scenarios the mean is also 3000.0.
    """
    costs = _make_costs_df(n_scenarios=2, n_stages=3, thermal_cost=1000.0)
    data = _make_mock_data(costs=costs, discount_rate=0.0)
    result = _compute_npv_metric(data, "thermal_cost")
    assert abs(result - 3000.0) < 1e-6


def test_compute_npv_metric_discounted_less_than_undiscounted() -> None:
    """When novomodelo's ``discount_factor`` column is <1 on later stages the NPV
    is lower than the undiscounted per-scenario sum."""
    undiscounted = _compute_npv_metric(
        _make_mock_data(
            costs=_make_costs_df(n_scenarios=2, n_stages=3, thermal_cost=1000.0)
        ),
        "thermal_cost",
    )
    discounted = _compute_npv_metric(
        _make_mock_data(
            costs=_make_costs_df(
                n_scenarios=2, n_stages=3, thermal_cost=1000.0, stage_discount=0.9
            )
        ),
        "thermal_cost",
    )
    assert discounted < undiscounted


def test_compute_npv_metric_missing_column_returns_zero() -> None:
    """When the requested column is absent, _compute_npv_metric must return 0.0."""
    costs = _make_costs_df()
    data = _make_mock_data(costs=costs)
    result = _compute_npv_metric(data, "nonexistent_column")
    assert result == 0.0


def test_compute_npv_metric_empty_dataframe_returns_zero() -> None:
    """When costs is empty, _compute_npv_metric must return 0.0."""
    data = _make_mock_data(costs=pd.DataFrame())
    result = _compute_npv_metric(data, "thermal_cost")
    assert result == 0.0


# ---------------------------------------------------------------------------
# test__build_metrics_row
# ---------------------------------------------------------------------------


def test_build_metrics_row_produces_four_metric_cards() -> None:
    """_build_metrics_row must produce HTML with 'metric-card' at least 4 times."""
    data = _make_mock_data()
    html = _build_metrics_row(data)
    assert html.count("metric-card") >= 4


def test_build_metrics_row_empty_costs_still_four_cards() -> None:
    """_build_metrics_row with empty costs must still produce 4 metric-card elements."""
    data = _make_mock_data(costs=pd.DataFrame())
    html = _build_metrics_row(data)
    assert html.count("metric-card") >= 4


def test_build_metrics_row_contains_expected_cost_label() -> None:
    """_build_metrics_row must include the 'Expected Cost (NPV)' label."""
    data = _make_mock_data()
    html = _build_metrics_row(data)
    assert "Expected Cost (NPV)" in html


def test_build_metrics_row_contains_thermal_and_deficit_labels() -> None:
    """_build_metrics_row must include Thermal Cost and Deficit Cost labels."""
    data = _make_mock_data()
    html = _build_metrics_row(data)
    assert "Thermal Cost (NPV)" in html
    assert "Deficit Cost (NPV)" in html


# ---------------------------------------------------------------------------
# test_render
# ---------------------------------------------------------------------------


def test_render_with_valid_costs_contains_metric_card_four_times() -> None:
    """render() with valid costs must contain 'metric-card' at least 4 times."""
    data = _make_mock_data()
    html = render(data)
    assert html.count("metric-card") >= 4


def test_render_with_valid_costs_contains_data_table() -> None:
    """render() with valid costs must produce HTML with a data-table."""
    data = _make_mock_data()
    html = render(data)
    assert 'class="data-table"' in html


def test_render_with_valid_costs_has_table_and_tbody() -> None:
    """render() HTML must contain a <table with a <tbody> and at least one <tr>."""
    data = _make_mock_data()
    html = render(data)
    assert "<table" in html
    assert "<tbody>" in html
    assert "<tr>" in html


def test_render_with_empty_costs_contains_no_cost_data() -> None:
    """render() with empty costs must produce 'No cost data' fallback text."""
    data = _make_mock_data(costs=pd.DataFrame())
    html = render(data)
    assert "No cost data" in html


def test_render_contains_section_title() -> None:
    """render() must include the 'NPV Cost Analysis' section title."""
    data = _make_mock_data()
    html = render(data)
    assert "NPV Cost Analysis" in html


# ---------------------------------------------------------------------------
# Helpers for the extended-section tests
# ---------------------------------------------------------------------------


def _make_costs_df_with_violations(
    n_scenarios: int = 2,
    n_stages: int = 3,
    thermal_cost: float = 1000.0,
    generic_violation_cost: float = 0.0,
) -> pd.DataFrame:
    """Return a costs DataFrame that includes a violation cost column."""
    rows = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage_id": stage_id,
                    "block_id": 0,
                    "thermal_cost": thermal_cost,
                    "deficit_cost": 50.0,
                    "immediate_cost": thermal_cost + 50.0,
                    "total_cost": thermal_cost + 50.0,
                    "future_cost": 0.0,
                    "generic_violation_cost": generic_violation_cost,
                }
            )
    return pd.DataFrame(rows)


def _make_mock_data_full(
    *,
    costs: pd.DataFrame | None = None,
    non_fictitious_bus_ids: list[int] | None = None,
    bus_names: dict[int, str] | None = None,
    buses_lf: pl.LazyFrame | None = None,
    bh_df: pl.DataFrame | None = None,
    discount_rate: float = 0.0,
    n_scenarios: int = 2,
    n_stages: int = 3,
) -> MagicMock:
    """Build a MagicMock satisfying the full DashboardData interface."""
    data = MagicMock()
    data.costs = costs if costs is not None else _make_costs_df(n_scenarios, n_stages)
    data.discount_rate = discount_rate
    data.n_scenarios = n_scenarios
    data.n_stages = n_stages
    data.stage_labels = {i: f"Stage {i}" for i in range(n_stages)}
    data.stage_dates = {i: f"2024-01-{i + 1:02d}" for i in range(n_stages)}
    data.non_fictitious_bus_ids = (
        non_fictitious_bus_ids if non_fictitious_bus_ids is not None else [0, 1]
    )
    data.bus_names = bus_names if bus_names is not None else {0: "Bus A", 1: "Bus B"}

    if buses_lf is not None:
        data.buses_lf = buses_lf
    else:
        # Build a minimal buses_lf with spot_price
        rows_list = []
        for scen in range(n_scenarios):
            for stage in range(n_stages):
                for bus_id in data.non_fictitious_bus_ids or [0, 1]:
                    rows_list.append(
                        {
                            "scenario_id": scen,
                            "stage_id": stage,
                            "block_id": 0,
                            "bus_id": bus_id,
                            "spot_price": 100.0 + scen * 10 + stage * 5,
                        }
                    )
        data.buses_lf = pl.DataFrame(rows_list).lazy()

    if bh_df is not None:
        data.bh_df = bh_df
    else:
        # Build a minimal bh_df with one block per stage
        bh_rows = [
            {"stage_id": s, "block_id": 0, "_bh": 730.0} for s in range(n_stages)
        ]
        data.bh_df = pl.DataFrame(bh_rows)

    return data


# ---------------------------------------------------------------------------
# test__render_cost_composition (Section D)
# ---------------------------------------------------------------------------


def test_render_cost_composition_contains_collapsible_section() -> None:
    """_render_cost_composition must return HTML with collapsible-section class."""
    data = _make_mock_data_full()
    html = _render_cost_composition(data)
    assert "collapsible-section" in html


def test_render_cost_composition_contains_section_title() -> None:
    """_render_cost_composition must include the 'Cost Composition by Stage' title."""
    data = _make_mock_data_full()
    html = _render_cost_composition(data)
    assert "Cost Composition by Stage" in html


def test_render_cost_composition_with_empty_costs_returns_fallback() -> None:
    """_render_cost_composition with empty costs must return fallback text."""
    data = _make_mock_data_full(costs=pd.DataFrame())
    html = _render_cost_composition(data)
    assert "No cost data" in html


def test_render_cost_composition_is_not_default_collapsed() -> None:
    """_render_cost_composition must start expanded (no 'default-collapsed' class)."""
    data = _make_mock_data_full()
    html = _render_cost_composition(data)
    # default_collapsed=False means the section_class is 'collapsible-section'
    # (not 'collapsible-section default-collapsed')
    assert "default-collapsed" not in html


def test_render_cost_composition_contains_chart_card() -> None:
    """_render_cost_composition must wrap the chart in a chart-card."""
    data = _make_mock_data_full()
    html = _render_cost_composition(data)
    assert "chart-card" in html


# ---------------------------------------------------------------------------
# test__render_category_evolution (Section E)
# ---------------------------------------------------------------------------


def test_render_category_evolution_contains_collapsible_section() -> None:
    """_render_category_evolution must return HTML with collapsible-section class."""
    data = _make_mock_data_full()
    html = _render_category_evolution(data)
    assert "collapsible-section" in html


def test_render_category_evolution_contains_section_title() -> None:
    """_render_category_evolution must include the 'Cost Category Trends' title."""
    data = _make_mock_data_full()
    html = _render_category_evolution(data)
    assert "Cost Category Trends" in html


def test_render_category_evolution_with_empty_costs_returns_fallback() -> None:
    """_render_category_evolution with empty costs must return fallback text."""
    data = _make_mock_data_full(costs=pd.DataFrame())
    html = _render_category_evolution(data)
    assert "No cost data" in html


def test_render_category_evolution_is_default_collapsed() -> None:
    """_render_category_evolution must start collapsed.

    The collapsible_section helper marks a collapsed section with the CSS
    class 'collapsed-title' on the title div and 'collapsed' on the content
    div — not the string 'default-collapsed'.
    """
    data = _make_mock_data_full()
    html = _render_category_evolution(data)
    assert "collapsed-title" in html


def test_render_category_evolution_contains_chart_card() -> None:
    """_render_category_evolution must wrap the chart in a chart-card."""
    data = _make_mock_data_full()
    html = _render_category_evolution(data)
    assert "chart-card" in html


# ---------------------------------------------------------------------------
# test__render_spot_price (Section F)
# ---------------------------------------------------------------------------


def test_render_spot_price_contains_collapsible_section() -> None:
    """_render_spot_price must return HTML with collapsible-section class."""
    data = _make_mock_data_full(non_fictitious_bus_ids=[0, 1, 2, 3])
    data.bus_names = {0: "Bus A", 1: "Bus B", 2: "Bus C", 3: "Bus D"}
    # Rebuild buses_lf for 4 buses
    rows_list = []
    for scen in range(2):
        for stage in range(3):
            for bus_id in [0, 1, 2, 3]:
                rows_list.append(
                    {
                        "scenario_id": scen,
                        "stage_id": stage,
                        "block_id": 0,
                        "bus_id": bus_id,
                        "spot_price": 100.0 + scen * 10,
                    }
                )
    data.buses_lf = pl.DataFrame(rows_list).lazy()
    data.stage_labels = {0: "Stage 0", 1: "Stage 1", 2: "Stage 2"}
    data.stage_dates = {0: "2024-01-01", 1: "2024-01-02", 2: "2024-01-03"}
    html = _render_spot_price(data)
    assert "collapsible-section" in html


def test_render_spot_price_subplot_titles_match_bus_count() -> None:
    """_render_spot_price with 4 non-fictitious buses must produce 4 subplot titles.

    The number of subplot titles in the figure equals the number of
    non-fictitious buses.
    """
    bus_ids = [0, 1, 2, 3]
    bus_names = {0: "Alpha", 1: "Beta", 2: "Gamma", 3: "Delta"}
    rows_list = []
    for scen in range(2):
        for stage in range(3):
            for bus_id in bus_ids:
                rows_list.append(
                    {
                        "scenario_id": scen,
                        "stage_id": stage,
                        "block_id": 0,
                        "bus_id": bus_id,
                        "spot_price": 80.0 + scen * 5 + stage * 2,
                    }
                )
    buses_lf = pl.DataFrame(rows_list).lazy()
    bh_df = pl.DataFrame(
        [{"stage_id": s, "block_id": 0, "_bh": 730.0} for s in range(3)]
    )

    data = _make_mock_data_full(
        non_fictitious_bus_ids=bus_ids,
        bus_names=bus_names,
        buses_lf=buses_lf,
        bh_df=bh_df,
    )
    data.stage_labels = {0: "S0", 1: "S1", 2: "S2"}
    data.stage_dates = {0: "2024-01-01", 1: "2024-01-02", 2: "2024-01-03"}

    html = _render_spot_price(data)
    # Each bus name should appear in the HTML as a subplot title annotation
    for name in bus_names.values():
        assert name in html


def test_render_spot_price_with_empty_buses_lf_returns_fallback() -> None:
    """_render_spot_price with empty buses_lf must return 'No spot price data.'."""
    data = _make_mock_data_full(
        non_fictitious_bus_ids=[0, 1],
        buses_lf=pl.LazyFrame(),
    )
    html = _render_spot_price(data)
    assert "No spot price data" in html


def test_render_spot_price_with_no_bus_ids_returns_fallback() -> None:
    """_render_spot_price with empty non_fictitious_bus_ids returns fallback."""
    data = _make_mock_data_full(non_fictitious_bus_ids=[])
    html = _render_spot_price(data)
    assert "No spot price data" in html


def test_render_spot_price_is_default_collapsed() -> None:
    """_render_spot_price must start collapsed.

    The collapsible_section helper marks a collapsed section with the CSS
    class 'collapsed-title' on the title div and 'collapsed' on the content
    div — not the string 'default-collapsed'.
    """
    data = _make_mock_data_full()
    html = _render_spot_price(data)
    assert "collapsed-title" in html


# ---------------------------------------------------------------------------
# test__render_violations (Section G)
# ---------------------------------------------------------------------------


def test_render_violations_zero_costs_returns_no_violation_text() -> None:
    """_render_violations with all-zero violation costs must return 'No violation
    costs'."""
    costs = _make_costs_df_with_violations(generic_violation_cost=0.0)
    data = _make_mock_data_full(costs=costs)
    html = _render_violations(data)
    assert "No violation costs" in html


def test_render_violations_nonzero_costs_returns_collapsible_section() -> None:
    """_render_violations with non-zero generic_violation_cost returns
    collapsible-section.

    The result contains 'collapsible-section' and a bar chart.
    """
    costs = _make_costs_df_with_violations(generic_violation_cost=500.0)
    data = _make_mock_data_full(costs=costs)
    html = _render_violations(data)
    assert "collapsible-section" in html
    # Bar chart is embedded in the chart-card
    assert "chart-card" in html


def test_render_violations_nonzero_contains_section_title() -> None:
    """_render_violations with violations must include the 'Violation Costs' title."""
    costs = _make_costs_df_with_violations(generic_violation_cost=200.0)
    data = _make_mock_data_full(costs=costs)
    html = _render_violations(data)
    assert "Violation Costs" in html


def test_render_violations_empty_costs_returns_fallback() -> None:
    """_render_violations with empty costs DataFrame must return fallback text."""
    data = _make_mock_data_full(costs=pd.DataFrame())
    html = _render_violations(data)
    assert "No violation costs" in html


def test_render_violations_no_violation_columns_returns_fallback() -> None:
    """_render_violations when costs has no violation columns must return fallback."""
    # costs has only thermal_cost and deficit_cost — no violation columns
    costs = _make_costs_df(n_scenarios=2, n_stages=3)
    data = _make_mock_data_full(costs=costs)
    html = _render_violations(data)
    assert "No violation costs" in html


def test_render_violations_is_default_collapsed() -> None:
    """_render_violations must start collapsed regardless of content.

    The collapsible_section helper marks a collapsed section with the CSS
    class 'collapsed-title' on the title div and 'collapsed' on the content
    div — not the string 'default-collapsed'.
    """
    costs = _make_costs_df_with_violations(generic_violation_cost=100.0)
    data = _make_mock_data_full(costs=costs)
    html = _render_violations(data)
    assert "collapsed-title" in html


# ---------------------------------------------------------------------------
# test_render — extended
# ---------------------------------------------------------------------------


def test_render_contains_cost_composition_section() -> None:
    """render() with non-zero thermal_cost across 3 stages must include
    'Cost Composition by Stage'."""
    costs = _make_costs_df(n_scenarios=2, n_stages=3, thermal_cost=1000.0)
    data = _make_mock_data_full(costs=costs)
    html = render(data)
    assert "Cost Composition by Stage" in html


def test_render_contains_at_least_three_collapsible_sections() -> None:
    """render() must contain at least 3 collapsible-section elements.

    Composition, category trends, and spot price sections are always present
    (violation section may be absent).
    """
    costs = _make_costs_df(n_scenarios=2, n_stages=3, thermal_cost=1000.0)
    data = _make_mock_data_full(costs=costs)
    html = render(data)
    assert html.count("collapsible-section") >= 3


def test_render_includes_npv_section() -> None:
    """render() must still include the NPV Cost Analysis section."""
    data = _make_mock_data_full()
    html = render(data)
    assert "NPV Cost Analysis" in html


def test_render_includes_all_temporal_sections() -> None:
    """render() with violation costs must include the three temporal sections.

    render() renders: Cost Composition by Stage (via _build_composition_section),
    Spot Price by Bus (via _render_spot_price), and Violation Costs (via
    _render_violations). Cost Category Trends (_render_category_evolution) was
    removed from render() and is only accessible as a standalone function.
    """
    costs = _make_costs_df_with_violations(
        n_scenarios=2, n_stages=3, thermal_cost=1000.0, generic_violation_cost=50.0
    )
    data = _make_mock_data_full(costs=costs)
    html = render(data)
    assert "Cost Composition by Stage" in html
    assert "Spot Price by Bus" in html
    assert "Violation Costs" in html


# ---------------------------------------------------------------------------
# test__chart_violation_timeline
# ---------------------------------------------------------------------------


def _make_costs_df_with_storage_violation(
    n_scenarios: int = 2,
    n_stages: int = 3,
    storage_violation_cost: float = 10.0,
) -> pd.DataFrame:
    """Return a costs DataFrame with a storage_violation_cost column.

    Values are set to *storage_violation_cost* for all rows so that per-stage
    sums are always positive when the argument is nonzero.
    """
    rows = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage_id": stage_id,
                    "block_id": 0,
                    "thermal_cost": 1000.0,
                    "storage_violation_cost": storage_violation_cost,
                }
            )
    return pd.DataFrame(rows)


def test_chart_violation_timeline_returns_figure_with_nonzero_data() -> None:
    """_chart_violation_timeline must return a Figure when violation data is present."""
    costs = _make_costs_df_with_storage_violation(storage_violation_cost=10.0)
    data = _make_mock_data_full(costs=costs)
    fig = _chart_violation_timeline(data)
    assert fig is not None
    assert isinstance(fig, go.Figure)


def test_chart_violation_timeline_has_scatter_trace_for_storage_violation() -> None:
    """_chart_violation_timeline must include a Scatter trace for storage_violation.

    Acceptance criterion: trace name contains 'Storage Violation' (cleaned label).
    """
    costs = _make_costs_df_with_storage_violation(storage_violation_cost=10.0)
    data = _make_mock_data_full(costs=costs)
    fig = _chart_violation_timeline(data)
    assert fig is not None
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(scatter_traces) >= 1
    trace_names = [t.name for t in scatter_traces]
    # Cleaned label: "_cost" removed, "_" -> " ", then title-cased
    assert any("Storage Violation" in name for name in trace_names)


def test_chart_violation_timeline_x_axis_uses_dates_with_label_ticks() -> None:
    """_chart_violation_timeline plots on a proportional date x-axis.

    The trace x-values are each stage's ISO ``start_date`` (strings, never raw
    stage_id integers), the axis is a ``type="date"`` axis so stages are spaced
    by real calendar distance, and the human-readable stage labels are carried
    as the axis tick text.
    """
    costs = _make_costs_df_with_storage_violation(
        n_scenarios=2, n_stages=3, storage_violation_cost=20.0
    )
    data = _make_mock_data_full(costs=costs, n_stages=3)
    # stage_dates maps 0->"2024-01-01", 1->"2024-01-02", 2->"2024-01-03";
    # stage_labels maps 0->"Stage 0", 1->"Stage 1", 2->"Stage 2".
    fig = _chart_violation_timeline(data)
    assert fig is not None
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(scatter_traces) >= 1
    x_vals = list(scatter_traces[0].x)
    # x-values are ISO date strings, never raw stage_id integers.
    assert all(isinstance(v, str) for v in x_vals)
    assert any(v in ("2024-01-01", "2024-01-02", "2024-01-03") for v in x_vals)
    # Proportional date axis, with the stage labels as tick text.
    assert fig.layout.xaxis.type == "date"
    assert set(fig.layout.xaxis.ticktext) <= {"Stage 0", "Stage 1", "Stage 2"}


def test_chart_violation_timeline_returns_none_when_all_zero() -> None:
    """_chart_violation_timeline must return None when all violation costs are zero."""
    costs = _make_costs_df_with_storage_violation(storage_violation_cost=0.0)
    data = _make_mock_data_full(costs=costs)
    fig = _chart_violation_timeline(data)
    assert fig is None


def test_chart_violation_timeline_returns_none_for_empty_costs() -> None:
    """_chart_violation_timeline must return None when costs DataFrame is empty."""
    data = _make_mock_data_full(costs=pd.DataFrame())
    fig = _chart_violation_timeline(data)
    assert fig is None


def test_chart_violation_timeline_returns_none_when_no_violation_columns() -> None:
    """_chart_violation_timeline must return None when no violation columns exist."""
    # costs only has thermal_cost (no violation column)
    costs = _make_costs_df(n_scenarios=2, n_stages=3, thermal_cost=1000.0)
    data = _make_mock_data_full(costs=costs)
    fig = _chart_violation_timeline(data)
    assert fig is None


# ---------------------------------------------------------------------------
# test__render_violations extended
# ---------------------------------------------------------------------------


def test_render_violations_with_nonzero_data_contains_two_chart_cards() -> None:
    """_render_violations with nonzero violation data must contain two chart-card divs.

    The Violations section contains a 2-column grid with bar chart (left) +
    timeline (right).
    """
    costs = _make_costs_df_with_storage_violation(
        n_scenarios=2, n_stages=3, storage_violation_cost=10.0
    )
    data = _make_mock_data_full(costs=costs)
    html = _render_violations(data)
    assert html.count("chart-card") >= 2


def test_render_violations_timeline_absent_when_all_violation_zero() -> None:
    """_render_violations with all-zero violations must render only the bar fallback.

    When all violations are zero, _render_violations returns the 'No violation costs'
    message — the timeline chart is not rendered.
    """
    costs = _make_costs_df_with_storage_violation(storage_violation_cost=0.0)
    data = _make_mock_data_full(costs=costs)
    html = _render_violations(data)
    assert "No violation costs" in html
    assert "chart-card" not in html


# ---------------------------------------------------------------------------
# test__build_composition_data
# ---------------------------------------------------------------------------


def _make_composition_costs_df(
    n_scenarios: int = 2,
    n_stages: int = 3,
) -> pd.DataFrame:
    """Return a small costs DataFrame with 3 cost columns for composition tests.

    Columns: thermal_generation_cost (non-zero), spillage_cost (non-zero),
    ncs_generation_cost (all zero).
    """
    rows = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage_id": stage_id,
                    "block_id": 0,
                    "thermal_generation_cost": 1000.0
                    + scenario_id * 100
                    + stage_id * 50,
                    "spillage_cost": 200.0 + scenario_id * 20 + stage_id * 10,
                    "ncs_generation_cost": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_build_composition_data_keys() -> None:
    """_build_composition_data must return a dict with required top-level keys.

    Keys are category, component, total, stages, colors — plus ``stage_labels``
    (the tick text paired with the ISO date positions in ``stages`` for the
    proportional date x-axis).
    """
    costs = _make_composition_costs_df()
    data = _make_mock_data_full(costs=costs, n_scenarios=2, n_stages=3)
    result = _build_composition_data(data)
    assert result is not None
    assert set(result.keys()) == {
        "category",
        "component",
        "total",
        "stages",
        "stage_labels",
        "colors",
    }


def test_build_composition_data_total_stats() -> None:
    """total.mean, total.p10, total.p90 must all have length == number of stages
    and satisfy p10 <= mean <= p90 for each stage."""
    costs = _make_composition_costs_df(n_scenarios=2, n_stages=3)
    data = _make_mock_data_full(costs=costs, n_scenarios=2, n_stages=3)
    result = _build_composition_data(data)
    assert result is not None
    total = result["total"]
    n = len(result["stages"])
    assert len(total["mean"]) == n
    assert len(total["p10"]) == n
    assert len(total["p90"]) == n
    for i in range(n):
        assert total["p10"][i] <= total["mean"][i] + 1e-6
        assert total["mean"][i] <= total["p90"][i] + 1e-6


def test_build_composition_data_empty() -> None:
    """_build_composition_data must return None for an empty costs DataFrame."""
    data = _make_mock_data_full(costs=pd.DataFrame())
    result = _build_composition_data(data)
    assert result is None


def test_build_composition_data_zero_cols_excluded() -> None:
    """Zero-valued columns must not appear in component.

    ncs_generation_cost is all-zero and must be absent from component keys.
    """
    costs = _make_composition_costs_df()
    data = _make_mock_data_full(costs=costs, n_scenarios=2, n_stages=3)
    result = _build_composition_data(data)
    assert result is not None
    # The cleaned label for "ncs_generation_cost" -> "Ncs Generation"
    assert "Ncs Generation" not in result["component"]
    # Non-zero columns must be present
    assert len(result["component"]) >= 1


def test_build_composition_section_html() -> None:
    """_build_composition_section must emit costs-group-sel, costs-comp,
    and COSTS_COMP_DATA in the HTML output."""
    costs = _make_composition_costs_df()
    data = _make_mock_data_full(costs=costs, n_scenarios=2, n_stages=3)
    html = _build_composition_section(data)
    assert "costs-group-sel" in html
    assert "costs-comp" in html
    assert "COSTS_COMP_DATA" in html
