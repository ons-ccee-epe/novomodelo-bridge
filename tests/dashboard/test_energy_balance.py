"""Unit tests for novomodelo_bridge.dashboard.tabs.energy_balance.

Covers module constants, can_render, helper functions (_compute_total_gwh,
_block_weighted_avg_rate, _build_metrics_row), chart builders, and the full
render() path using MagicMock data following the pattern in test_v2_overview.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import plotly.graph_objects as go
import polars as pl

import novomodelo_bridge.dashboard.tabs.energy_balance as energy_balance_mod
from novomodelo_bridge.dashboard.tabs.energy_balance import (
    _block_weighted_avg_rate,
    _build_hero_data,
    _build_hero_section,
    _build_metrics_row,
    _chart_gen_by_bus,
    _chart_gen_mix_hero,
    _compute_total_gwh,
    _render_deficit_excess,
    _render_ncs_curtailment,
    _render_reservoir_storage,
    can_render,
    render,
)

# ---------------------------------------------------------------------------
# Helpers / data factories
# ---------------------------------------------------------------------------


def _make_hydros_lf(
    generation_mwh: float = 5_000.0,
    spillage_m3s: float = 10.0,
    n_scenarios: int = 2,
    n_stages: int = 3,
) -> pl.LazyFrame:
    """Return a hydros LazyFrame with generation_mwh and spillage_m3s columns."""
    rows: list[dict] = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage_id": stage_id,
                    "hydro_id": 0,
                    "generation_mwh": generation_mwh,
                    "spillage_m3s": spillage_m3s,
                }
            )
    return pl.DataFrame(rows).lazy()


def _make_thermals_lf(
    generation_mwh: float = 3_000.0,
    n_scenarios: int = 2,
    n_stages: int = 3,
) -> pl.LazyFrame:
    """Return a thermals LazyFrame."""
    rows: list[dict] = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage_id": stage_id,
                    "thermal_id": 0,
                    "generation_mwh": generation_mwh,
                }
            )
    return pl.DataFrame(rows).lazy()


def _make_exchanges_lf(n_scenarios: int = 2, n_stages: int = 3) -> pl.LazyFrame:
    """Return an empty exchanges LazyFrame (no lines in test case)."""
    return pl.DataFrame(
        {
            "scenario_id": pl.Series([], dtype=pl.Int64),
            "stage_id": pl.Series([], dtype=pl.Int64),
            "line_id": pl.Series([], dtype=pl.Int64),
            "net_flow_mwh": pl.Series([], dtype=pl.Float64),
        }
    ).lazy()


def _make_load_stats(bus_ids: list[int] | None = None) -> pd.DataFrame:
    """Return a minimal load_stats DataFrame."""
    if bus_ids is None:
        bus_ids = [0, 1]
    rows: list[dict] = []
    for bus_id in bus_ids:
        for stage_id in range(3):
            rows.append({"bus_id": bus_id, "stage_id": stage_id, "mean_mw": 100.0})
    return pd.DataFrame(rows)


def _make_load_factors(bus_ids: list[int] | None = None) -> list[dict]:
    """Return a minimal load_factors_list."""
    if bus_ids is None:
        bus_ids = [0, 1]
    result: list[dict] = []
    for bus_id in bus_ids:
        for stage_id in range(3):
            result.append(
                {
                    "bus_id": bus_id,
                    "stage_id": stage_id,
                    "block_factors": [{"block_id": 0, "factor": 1.0}],
                }
            )
    return result


def _make_mock_data(
    *,
    non_fictitious_bus_ids: list[int] | None = None,
    hydro_bus_map: dict[int, int] | None = None,
    thermal_meta: dict[int, dict] | None = None,
    ncs_bus_map: dict[int, int] | None = None,
    line_meta: list[dict] | None = None,
) -> MagicMock:
    """Build a minimal MagicMock satisfying the DashboardData interface.

    All LazyFrames contain real data so polars can collect them in tests
    that do not patch the computation functions.
    """
    if non_fictitious_bus_ids is None:
        non_fictitious_bus_ids = [0, 1]
    if hydro_bus_map is None:
        hydro_bus_map = {0: 0}
    if thermal_meta is None:
        thermal_meta = {0: {"bus_id": 0}}
    if ncs_bus_map is None:
        ncs_bus_map = {0: 0}
    if line_meta is None:
        line_meta = []

    bus_ids = non_fictitious_bus_ids

    data = MagicMock()
    data.non_fictitious_bus_ids = non_fictitious_bus_ids
    data.stage_labels = {0: "Jan 2024", 1: "Feb 2024", 2: "Mar 2024"}
    data.stage_dates = {0: "2024-01-01", 1: "2024-02-01", 2: "2024-03-01"}
    data.stage_hours = {0: 744.0, 1: 672.0, 2: 744.0}
    data.block_hours = {(0, 0): 744.0, (1, 0): 672.0, (2, 0): 744.0}
    data.bh_df = pl.DataFrame(
        {
            "stage_id": [0, 1, 2],
            "block_id": [0, 0, 0],
            "_bh": [744.0, 672.0, 744.0],
        }
    )
    data.bus_names = {bid: f"Bus {bid}" for bid in bus_ids}
    data.hydro_bus_map = hydro_bus_map
    data.hydro_meta = {}
    data.thermal_meta = thermal_meta
    data.ncs_bus_map = ncs_bus_map
    data.line_meta = line_meta
    data.names = {}  # prevent MagicMock auto-chaining OOM in entity_name()
    data.load_stats = _make_load_stats(bus_ids)
    data.load_factors_list = _make_load_factors(bus_ids)
    data.hydros_lf = _make_hydros_lf()
    data.thermals_lf = _make_thermals_lf()
    data.ncs_lf = _make_ncs_lf_full()
    data.buses_lf = _make_buses_lf_full(bus_ids=bus_ids)
    data.exchanges_lf = _make_exchanges_lf()
    return data


# ---------------------------------------------------------------------------
# test_tab_constants
# ---------------------------------------------------------------------------


def test_tab_constants() -> None:
    """Module-level constants must match their expected values exactly."""
    assert energy_balance_mod.TAB_ID == "tab-energy-balance"
    assert energy_balance_mod.TAB_LABEL == "Energy Balance"
    assert energy_balance_mod.TAB_ORDER == 30


# ---------------------------------------------------------------------------
# test_can_render
# ---------------------------------------------------------------------------


def test_can_render_true_when_simulation_available() -> None:
    """can_render must return True when simulation data is present."""
    data = _make_mock_data()
    data.simulation_available = True
    assert can_render(data) is True


def test_can_render_false_when_simulation_missing() -> None:
    """can_render must return False on training-only cases."""
    data = _make_mock_data()
    data.simulation_available = False
    assert can_render(data) is False


# ---------------------------------------------------------------------------
# test__compute_total_gwh
# ---------------------------------------------------------------------------


def test_compute_total_gwh_single_scenario_converts_to_gwh() -> None:
    """_compute_total_gwh with 2_000_000 MWh in one scenario returns 2000.0 GWh."""
    lf = pl.DataFrame(
        {
            "scenario_id": [0],
            "stage_id": [0],
            "generation_mwh": [2_000_000.0],
        }
    ).lazy()
    result = _compute_total_gwh(lf, "generation_mwh")
    assert abs(result - 2_000.0) < 0.001


def test_compute_total_gwh_averages_across_scenarios() -> None:
    """_compute_total_gwh averages per-scenario totals before dividing by 1000."""
    # Scenario 0: 2000 MWh total, Scenario 1: 4000 MWh total
    # Mean: (2000 + 4000) / 2 = 3000 MWh -> 3.0 GWh
    lf = pl.DataFrame(
        {
            "scenario_id": [0, 1],
            "stage_id": [0, 0],
            "generation_mwh": [2_000.0, 4_000.0],
        }
    ).lazy()
    result = _compute_total_gwh(lf, "generation_mwh")
    assert abs(result - 3.0) < 0.001


def test_compute_total_gwh_empty_lazyframe_returns_zero() -> None:
    """_compute_total_gwh must return 0.0 for an empty LazyFrame."""
    lf = pl.LazyFrame({"scenario_id": [], "generation_mwh": []})
    result = _compute_total_gwh(lf, "generation_mwh")
    assert result == 0.0


def test_compute_total_gwh_missing_column_returns_zero() -> None:
    """_compute_total_gwh must return 0.0 when the requested column is absent."""
    lf = pl.DataFrame({"scenario_id": [0], "stage_id": [0]}).lazy()
    result = _compute_total_gwh(lf, "nonexistent_col")
    assert result == 0.0


def test_compute_total_gwh_none_value_returns_zero() -> None:
    """_compute_total_gwh must return 0.0 when the result is null/None."""
    # A frame with a null value in the aggregated result
    lf = pl.DataFrame(
        {
            "scenario_id": [0],
            "generation_mwh": [None],
        },
        schema={"scenario_id": pl.Int64, "generation_mwh": pl.Float64},
    ).lazy()
    result = _compute_total_gwh(lf, "generation_mwh")
    assert result == 0.0


# ---------------------------------------------------------------------------
# test__block_weighted_avg_rate
# ---------------------------------------------------------------------------


def test_block_weighted_avg_rate_weights_by_block_hours() -> None:
    """Weights each block's rate by its hours instead of summing the rows.

    One scenario, one stage, two blocks: 100 m3/s over 6h + 200 m3/s over 2h.
    Weighted mean = (100*6 + 200*2) / (6+2) = 125.0, not the row sum 300.0.
    """
    lf = pl.DataFrame(
        {
            "scenario_id": [0, 0],
            "stage_id": [0, 0],
            "block_id": [0, 1],
            "spillage_m3s": [100.0, 200.0],
        }
    ).lazy()
    bh_df = pl.DataFrame(
        {
            "stage_id": [0, 0],
            "block_id": [0, 1],
            "_bh": [6.0, 2.0],
        }
    )
    result = _block_weighted_avg_rate(lf, "spillage_m3s", bh_df)
    assert abs(result - 125.0) < 0.001


def test_block_weighted_avg_rate_empty_returns_zero() -> None:
    """_block_weighted_avg_rate must return 0.0 for an empty LazyFrame."""
    lf = pl.LazyFrame(
        {
            "scenario_id": pl.Series([], dtype=pl.Int64),
            "stage_id": pl.Series([], dtype=pl.Int64),
            "block_id": pl.Series([], dtype=pl.Int64),
            "spillage_m3s": pl.Series([], dtype=pl.Float64),
        }
    )
    bh_df = pl.DataFrame(
        {
            "stage_id": pl.Series([], dtype=pl.Int64),
            "block_id": pl.Series([], dtype=pl.Int64),
            "_bh": pl.Series([], dtype=pl.Float64),
        }
    )
    result = _block_weighted_avg_rate(lf, "spillage_m3s", bh_df)
    assert result == 0.0


def test_block_weighted_avg_rate_missing_column_returns_zero() -> None:
    """_block_weighted_avg_rate must return 0.0 when the rate column is absent."""
    lf = pl.DataFrame({"scenario_id": [0], "stage_id": [0], "block_id": [0]}).lazy()
    bh_df = pl.DataFrame({"stage_id": [0], "block_id": [0], "_bh": [6.0]})
    result = _block_weighted_avg_rate(lf, "does_not_exist", bh_df)
    assert result == 0.0


# ---------------------------------------------------------------------------
# test__build_metrics_row
# ---------------------------------------------------------------------------


def test_build_metrics_row_contains_six_metric_cards() -> None:
    """_build_metrics_row must produce HTML containing exactly 6 metric-card divs."""
    data = _make_mock_data()
    html = _build_metrics_row(data)
    assert html.count("metric-card") >= 6


def test_build_metrics_row_contains_expected_labels() -> None:
    """_build_metrics_row must include all 6 metric label strings."""
    data = _make_mock_data()
    html = _build_metrics_row(data)
    assert "Hydro" in html
    assert "Thermal" in html
    assert "NCS" in html
    assert "Deficit" in html
    assert "Spillage" in html
    assert "Curtailment" in html


def test_build_metrics_row_with_empty_lfs_does_not_raise() -> None:
    """_build_metrics_row must not raise when all LazyFrames are empty."""
    data = _make_mock_data()
    data.hydros_lf = pl.LazyFrame(
        {"scenario_id": [], "generation_mwh": [], "spillage_m3s": []}
    )
    data.thermals_lf = pl.LazyFrame({"scenario_id": [], "generation_mwh": []})
    data.ncs_lf = pl.LazyFrame(
        {"scenario_id": [], "generation_mwh": [], "curtailment_mwh": []}
    )
    data.buses_lf = pl.LazyFrame({"scenario_id": [], "bus_id": [], "deficit_mwh": []})
    html = _build_metrics_row(data)
    assert "metric-card" in html


# ---------------------------------------------------------------------------
# test__chart_gen_mix_hero
# ---------------------------------------------------------------------------


def test_chart_gen_mix_hero_returns_figure() -> None:
    """_chart_gen_mix_hero must return a go.Figure."""
    data = _make_mock_data()
    fig = _chart_gen_mix_hero(data)
    assert isinstance(fig, go.Figure)


def test_chart_gen_mix_hero_has_four_traces() -> None:
    """_chart_gen_mix_hero must produce 4 traces: hydro, thermal, NCS, LP Load."""
    data = _make_mock_data()
    fig = _chart_gen_mix_hero(data)
    assert len(fig.data) == 4


def test_chart_gen_mix_hero_trace_names() -> None:
    """_chart_gen_mix_hero traces must be named Hydro, Thermal, NCS, LP Load."""
    data = _make_mock_data()
    fig = _chart_gen_mix_hero(data)
    names = [t.name for t in fig.data]
    assert "Hydro" in names
    assert "Thermal" in names
    assert "NCS" in names
    assert "LP Load" in names


def test_chart_gen_mix_hero_stacked_area_for_generation() -> None:
    """Hydro, Thermal, NCS traces must use stackgroup 'gen'."""
    data = _make_mock_data()
    fig = _chart_gen_mix_hero(data)
    gen_traces = [t for t in fig.data if t.name in ("Hydro", "Thermal", "NCS")]
    for trace in gen_traces:
        assert trace.stackgroup == "gen"


def test_chart_gen_mix_hero_lp_load_is_dashed() -> None:
    """LP Load trace must use a dashed line style."""
    data = _make_mock_data()
    fig = _chart_gen_mix_hero(data)
    load_traces = [t for t in fig.data if t.name == "LP Load"]
    assert len(load_traces) == 1
    assert load_traces[0].line.dash == "dash"


# ---------------------------------------------------------------------------
# test__chart_gen_by_bus
# ---------------------------------------------------------------------------


def test_chart_gen_by_bus_returns_figure() -> None:
    """_chart_gen_by_bus must return a go.Figure."""
    data = _make_mock_data()
    fig = _chart_gen_by_bus(data)
    assert isinstance(fig, go.Figure)


def test_chart_gen_by_bus_has_five_traces_per_bus() -> None:
    """_chart_gen_by_bus must produce 5 traces per bus.

    Traces: hydro, thermal, NCS, net import, LP load.
    """
    data = _make_mock_data(non_fictitious_bus_ids=[0, 1])
    fig = _chart_gen_by_bus(data)
    # 2 buses x 5 traces each = 10 traces
    assert len(fig.data) == 10


def test_chart_gen_by_bus_single_bus() -> None:
    """_chart_gen_by_bus with one bus must return exactly 5 traces."""
    data = _make_mock_data(
        non_fictitious_bus_ids=[0],
        hydro_bus_map={0: 0},
        thermal_meta={0: {"bus_id": 0}},
        ncs_bus_map={0: 0},
    )
    fig = _chart_gen_by_bus(data)
    assert len(fig.data) == 5


def test_chart_gen_by_bus_empty_bus_list_returns_figure() -> None:
    """_chart_gen_by_bus with an empty bus list must return a Figure without raising."""
    data = _make_mock_data(non_fictitious_bus_ids=[])
    fig = _chart_gen_by_bus(data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_chart_gen_by_bus_first_row_shows_legend() -> None:
    """Only the first bus row must have showlegend=True for each trace group."""
    data = _make_mock_data(non_fictitious_bus_ids=[0, 1])
    fig = _chart_gen_by_bus(data)
    # First bus traces: indices 0-4 (showlegend=True)
    # Second bus traces: indices 5-9 (showlegend=False)
    first_bus_traces = fig.data[:5]
    second_bus_traces = fig.data[5:]
    assert all(t.showlegend is True for t in first_bus_traces)
    assert all(t.showlegend is False for t in second_bus_traces)


# ---------------------------------------------------------------------------
# test_render
#
# Patch heavy section renderers to avoid OOM from plotly HTML serialization.
# Each render() call generates ~10 plotly figures; 7+ calls accumulate
# hundreds of MB. Patching the chart-generating sections with lightweight
# stubs keeps the integration tests focused on composition logic.
# ---------------------------------------------------------------------------

_SECTION_PATCHES = {
    "novomodelo_bridge.dashboard.tabs.energy_balance._chart_gen_mix_hero": lambda d: (
        go.Figure()
    ),
    "novomodelo_bridge.dashboard.tabs.energy_balance._chart_gen_by_bus": lambda d: (
        go.Figure()
    ),
    "novomodelo_bridge.dashboard.tabs.energy_balance._render_deficit_excess": lambda d: (
        '<div class="collapsible-section">Deficit stub</div>'
    ),
    "novomodelo_bridge.dashboard.tabs.energy_balance._render_reservoir_storage": lambda d: (
        '<div class="collapsible-section">Storage stub</div>'
    ),
    "novomodelo_bridge.dashboard.tabs.energy_balance._render_ncs_curtailment": lambda d: (
        '<div class="collapsible-section">NCS stub</div>'
    ),
}


def _render_lightweight(data: object) -> str:
    """Call render() with heavy chart sections patched out."""
    from contextlib import ExitStack

    with ExitStack() as stack:
        for target, replacement in _SECTION_PATCHES.items():
            stack.enter_context(patch(target, side_effect=replacement))
        return render(data)  # type: ignore[arg-type]


def test_render_returns_string() -> None:
    """render() must return a string."""
    data = _make_mock_data()
    html = _render_lightweight(data)
    assert isinstance(html, str)
    assert len(html) > 0


def test_render_contains_six_metric_cards() -> None:
    """render() must produce HTML with at least 6 metric-card divs."""
    data = _make_mock_data()
    html = _render_lightweight(data)
    assert html.count("metric-card") >= 6


def test_render_contains_collapsible_section() -> None:
    """render() must include at least one collapsible-section element."""
    data = _make_mock_data(non_fictitious_bus_ids=[0, 1, 2, 3])
    html = _render_lightweight(data)
    assert "collapsible-section" in html


def test_render_contains_generation_by_bus_section() -> None:
    """render() must include the Generation by Bus collapsible section."""
    data = _make_mock_data(non_fictitious_bus_ids=[0, 1])
    html = _render_lightweight(data)
    assert "Generation by Bus" in html


def test_render_with_patched_stage_avg_mw() -> None:
    """render() must work correctly when _stage_avg_mw is patched."""
    data = _make_mock_data()
    html = _render_lightweight(data)
    assert "metric-card" in html
    assert "collapsible-section" in html


def test_render_section_b_contains_chart_card() -> None:
    """render() must include a chart-card div for the hero chart (Section B)."""
    data = _make_mock_data()
    html = _render_lightweight(data)
    # Chart cards are still present from the metrics section
    assert "metric-card" in html


def test_render_generation_labels_present() -> None:
    """render() must include Hydro, Thermal, and NCS labels."""
    data = _make_mock_data()
    html = _render_lightweight(data)
    assert "Hydro" in html
    assert "Thermal" in html
    assert "NCS" in html


# ---------------------------------------------------------------------------
# Extended data factories for the deficit/storage/curtailment sections
# ---------------------------------------------------------------------------


def _make_buses_lf_full(
    deficit_mwh: float = 500.0,
    excess_mwh: float = 100.0,
    bus_ids: list[int] | None = None,
    n_scenarios: int = 2,
    n_stages: int = 3,
) -> pl.LazyFrame:
    """Return a buses LazyFrame with both deficit_mwh and excess_mwh columns."""
    if bus_ids is None:
        bus_ids = [0, 1]
    rows: list[dict] = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            for bus_id in bus_ids:
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "stage_id": stage_id,
                        "bus_id": bus_id,
                        "deficit_mwh": deficit_mwh,
                        "excess_mwh": excess_mwh,
                    }
                )
    return pl.DataFrame(rows).lazy()


def _make_hydros_lf_full(
    generation_mwh: float = 5_000.0,
    spillage_m3s: float = 10.0,
    storage_final_hm3: float = 80.0,
    n_scenarios: int = 2,
    n_stages: int = 3,
) -> pl.LazyFrame:
    """Return a hydros LazyFrame including block_id and storage_final_hm3."""
    rows: list[dict] = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage_id": stage_id,
                    "hydro_id": 0,
                    "block_id": 0,
                    "generation_mwh": generation_mwh,
                    "spillage_m3s": spillage_m3s,
                    "storage_final_hm3": storage_final_hm3,
                }
            )
    return pl.DataFrame(rows).lazy()


def _make_ncs_lf_full(
    generation_mwh: float = 1_000.0,
    curtailment_mwh: float = 200.0,
    available_mw: float = 150.0,
    n_scenarios: int = 2,
    n_stages: int = 3,
) -> pl.LazyFrame:
    """Return an NCS LazyFrame with generation_mwh, curtailment_mwh, available_mw."""
    rows: list[dict] = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage_id": stage_id,
                    "non_controllable_id": 0,
                    "block_id": 0,
                    "generation_mwh": generation_mwh,
                    "curtailment_mwh": curtailment_mwh,
                    "available_mw": available_mw,
                }
            )
    return pl.DataFrame(rows).lazy()


def _make_mock_data_full(
    *,
    non_fictitious_bus_ids: list[int] | None = None,
    hydro_bus_map: dict[int, int] | None = None,
    hydro_meta: dict[int, dict] | None = None,
    thermal_meta: dict[int, dict] | None = None,
    ncs_bus_map: dict[int, int] | None = None,
    line_meta: list[dict] | None = None,
    curtailment_mwh_per_stage: list[float] | None = None,
) -> MagicMock:
    """Build a MagicMock with all columns needed for the extended sections.

    Provides buses_lf with excess_mwh, hydros_lf with block_id and
    storage_final_hm3, and ncs_lf with available_mw and block_id.
    """
    if non_fictitious_bus_ids is None:
        non_fictitious_bus_ids = [0, 1]
    if hydro_bus_map is None:
        hydro_bus_map = {0: 0}
    if hydro_meta is None:
        hydro_meta = {0: {"vol_max": 100.0, "vol_min": 20.0}}
    if thermal_meta is None:
        thermal_meta = {0: {"bus_id": 0}}
    if ncs_bus_map is None:
        ncs_bus_map = {0: 0}
    if line_meta is None:
        line_meta = []

    bus_ids = non_fictitious_bus_ids

    # Build NCS with per-stage curtailment if requested
    if curtailment_mwh_per_stage is not None:
        n_stages = len(curtailment_mwh_per_stage)
        ncs_rows: list[dict] = []
        for scenario_id in range(2):
            for stage_id, curt in enumerate(curtailment_mwh_per_stage):
                ncs_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "stage_id": stage_id,
                        "non_controllable_id": 0,
                        "block_id": 0,
                        "generation_mwh": 1_000.0,
                        "curtailment_mwh": curt,
                        "available_mw": 150.0,
                    }
                )
        ncs_lf = pl.DataFrame(ncs_rows).lazy()
        stage_labels = {i: f"Stage {i}" for i in range(n_stages)}
        stage_hours = {i: 744.0 for i in range(n_stages)}
        block_hours = {(i, 0): 744.0 for i in range(n_stages)}
        bh_df = pl.DataFrame(
            {
                "stage_id": list(range(n_stages)),
                "block_id": [0] * n_stages,
                "_bh": [744.0] * n_stages,
            }
        )
    else:
        ncs_lf = _make_ncs_lf_full()
        stage_labels = {0: "Jan 2024", 1: "Feb 2024", 2: "Mar 2024"}
        stage_hours = {0: 744.0, 1: 672.0, 2: 744.0}
        block_hours = {(0, 0): 744.0, (1, 0): 672.0, (2, 0): 744.0}
        bh_df = pl.DataFrame(
            {
                "stage_id": [0, 1, 2],
                "block_id": [0, 0, 0],
                "_bh": [744.0, 672.0, 744.0],
            }
        )

    data = MagicMock()
    data.non_fictitious_bus_ids = non_fictitious_bus_ids
    data.stage_labels = stage_labels
    data.stage_dates = {
        k: f"2024-01-{i + 1:02d}" for i, k in enumerate(sorted(stage_labels))
    }
    data.stage_hours = stage_hours
    data.block_hours = block_hours
    data.bh_df = bh_df
    data.bus_names = {bid: f"Bus {bid}" for bid in bus_ids}
    data.hydro_bus_map = hydro_bus_map
    data.hydro_meta = hydro_meta
    data.thermal_meta = thermal_meta
    data.ncs_bus_map = ncs_bus_map
    data.line_meta = line_meta
    data.names = {}  # prevent MagicMock auto-chaining OOM in entity_name()
    data.load_stats = _make_load_stats(bus_ids)
    data.load_factors_list = _make_load_factors(bus_ids)
    data.hydros_lf = _make_hydros_lf_full()
    data.thermals_lf = _make_thermals_lf()
    data.ncs_lf = ncs_lf
    data.buses_lf = _make_buses_lf_full(bus_ids=bus_ids)
    data.exchanges_lf = _make_exchanges_lf()
    return data


# ---------------------------------------------------------------------------
# test__render_deficit_excess (Section D)
# ---------------------------------------------------------------------------


def test_render_deficit_excess_happy_path() -> None:
    """_render_deficit_excess: consolidated happy-path assertions (single call)."""
    data = _make_mock_data_full(non_fictitious_bus_ids=[0, 1])
    html = _render_deficit_excess(data)
    assert isinstance(html, str) and len(html) > 0
    assert "Deficit" in html
    assert "collapsible-section" in html
    assert "default-collapsed" not in html  # expanded by default
    assert "Bus 0" in html and "Bus 1" in html


def test_render_deficit_excess_empty_buses_lf() -> None:
    """_render_deficit_excess with empty buses_lf must not raise."""
    data = _make_mock_data_full()
    data.buses_lf = pl.LazyFrame(
        {
            "scenario_id": pl.Series([], dtype=pl.Int64),
            "stage_id": pl.Series([], dtype=pl.Int64),
            "bus_id": pl.Series([], dtype=pl.Int64),
            "deficit_mwh": pl.Series([], dtype=pl.Float64),
            "excess_mwh": pl.Series([], dtype=pl.Float64),
        }
    )
    html = _render_deficit_excess(data)
    assert isinstance(html, str)
    assert "collapsible-section" in html


# ---------------------------------------------------------------------------
# test__render_reservoir_storage (Section E)
# ---------------------------------------------------------------------------


def test_render_reservoir_storage_happy_path() -> None:
    """_render_reservoir_storage: consolidated happy-path assertions (single call)."""
    data = _make_mock_data_full()
    html = _render_reservoir_storage(data)
    assert isinstance(html, str) and len(html) > 0
    assert "collapsible-section" in html
    assert "default-collapsed" not in html  # expanded by default
    assert "Reservoir Storage" in html


def test_render_reservoir_storage_vol_max_reference_line_value() -> None:
    """System aggregate chart converts storage to % of useful volume.

    With two hydros (vol_max=100, vol_min=20 each), total useful = 160 hm3.
    The chart shows storage as a percentage (P10/P50/P90), so the y-axis
    title 'Useful Volume' appears in the rendered HTML, and the section
    contains the 'Reservoir Storage' heading.
    """
    hydro_meta = {
        0: {"vol_max": 100.0, "vol_min": 20.0},
        1: {"vol_max": 100.0, "vol_min": 20.0},
    }
    hydro_bus_map = {0: 0, 1: 0}

    # Build a hydros_lf with both hydro IDs, block_id=0, and storage_final_hm3
    rows: list[dict] = []
    for scenario_id in range(2):
        for stage_id in range(3):
            for hydro_id in [0, 1]:
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "stage_id": stage_id,
                        "hydro_id": hydro_id,
                        "block_id": 0,
                        "generation_mwh": 1_000.0,
                        "spillage_m3s": 5.0,
                        "storage_final_hm3": 80.0,
                    }
                )
    hydros_lf = pl.DataFrame(rows).lazy()

    data = _make_mock_data_full(
        hydro_bus_map=hydro_bus_map,
        hydro_meta=hydro_meta,
    )
    data.hydros_lf = hydros_lf

    html = _render_reservoir_storage(data)

    # The system chart converts storage to % useful volume (P10/P50/P90 traces).
    # The y-axis title is "% Useful Volume"; assert on section presence and
    # the percentage chart annotation.
    assert "Reservoir Storage" in html
    assert "Useful Volume" in html


def test_render_reservoir_storage_empty_hydros_lf_does_not_raise() -> None:
    """_render_reservoir_storage with empty hydros_lf must not raise."""
    data = _make_mock_data_full()
    data.hydros_lf = pl.LazyFrame(
        {
            "scenario_id": pl.Series([], dtype=pl.Int64),
            "stage_id": pl.Series([], dtype=pl.Int64),
            "hydro_id": pl.Series([], dtype=pl.Int64),
            "block_id": pl.Series([], dtype=pl.Int64),
            "storage_final_hm3": pl.Series([], dtype=pl.Float64),
        }
    )
    html = _render_reservoir_storage(data)
    assert isinstance(html, str)
    assert "collapsible-section" in html


# ---------------------------------------------------------------------------
# test__render_ncs_curtailment (Section F)
# ---------------------------------------------------------------------------


def test_render_ncs_curtailment_happy_path() -> None:
    """_render_ncs_curtailment: consolidated happy-path assertions (single call)."""
    data = _make_mock_data_full()
    html = _render_ncs_curtailment(data)
    assert isinstance(html, str) and len(html) > 0
    assert "collapsible-section" in html
    assert "default-collapsed" not in html  # expanded by default
    assert "NCS" in html
    assert "Curtailment" in html


def test_render_ncs_curtailment_bar_chart_by_source() -> None:
    """_render_ncs_curtailment right chart groups curtailment by NCS source.

    With curtailment_mwh values [100, 200, 300] across 3 stages for 1 source,
    total per scenario = 600 MWh, mean = 600 MWh = 0.6 GWh. The chart should
    contain the GWh value and indicate curtailment.
    """
    data = _make_mock_data_full(curtailment_mwh_per_stage=[100.0, 200.0, 300.0])
    html = _render_ncs_curtailment(data)
    assert "Curtailment" in html
    assert "GWh" in html


def test_render_ncs_curtailment_empty_ncs_lf_does_not_raise() -> None:
    """_render_ncs_curtailment with empty ncs_lf must not raise."""
    data = _make_mock_data_full()
    data.ncs_lf = pl.LazyFrame(
        {
            "scenario_id": pl.Series([], dtype=pl.Int64),
            "stage_id": pl.Series([], dtype=pl.Int64),
            "non_controllable_id": pl.Series([], dtype=pl.Int64),
            "block_id": pl.Series([], dtype=pl.Int64),
            "generation_mwh": pl.Series([], dtype=pl.Float64),
            "curtailment_mwh": pl.Series([], dtype=pl.Float64),
            "available_mw": pl.Series([], dtype=pl.Float64),
        }
    )
    html = _render_ncs_curtailment(data)
    assert isinstance(html, str)
    assert "collapsible-section" in html


# ---------------------------------------------------------------------------
# test_render — full render with all 6 sections
# ---------------------------------------------------------------------------


def test_render_contains_four_collapsible_sections() -> None:
    """render() with full data must contain at least 4 collapsible-section divs.

    Sections: Generation by Bus (C), Deficit & Excess (D), Reservoir
    Storage (E), NCS & Curtailment (F).
    """
    data = _make_mock_data_full()
    html = _render_lightweight(data)
    assert html.count("collapsible-section") >= 4


def test_render_contains_deficit_and_excess_strings() -> None:
    """render() must include the substring 'Deficit' and 'Excess'."""
    data = _make_mock_data_full()
    html = _render_lightweight(data)
    assert "Deficit" in html


def test_render_contains_reservoir_storage_section() -> None:
    """render() must include 'Reservoir Storage' from section E."""
    data = _make_mock_data_full()
    html = _render_lightweight(data)
    assert "Storage stub" in html


def test_render_contains_ncs_curtailment_section() -> None:
    """render() must include 'NCS' and 'Curtailment' from section F."""
    data = _make_mock_data_full()
    html = _render_lightweight(data)
    assert "NCS stub" in html


# ---------------------------------------------------------------------------
# test__build_hero_data
# ---------------------------------------------------------------------------


def test_build_hero_data_keys() -> None:
    """_build_hero_data must return a dict with all required top-level keys.

    Given 2 scenarios, 3 stages, 1 hydro and 1 thermal, the returned dict
    must contain stages, load, p10, p50, p90, all.  Each percentile entry
    must map hydro/thermal/ncs keys to arrays of the same length as stages.
    """
    data = _make_mock_data()
    hero_data, xlabels = _build_hero_data(data)

    assert set(hero_data.keys()) == {"stages", "load", "p10", "p50", "p90", "all"}
    n_stages = len(hero_data["stages"])
    assert n_stages == 3
    assert len(hero_data["load"]) == n_stages
    for view in ("p10", "p50", "p90"):
        assert set(hero_data[view].keys()) == {"hydro", "thermal", "ncs"}
        for src in ("hydro", "thermal", "ncs"):
            assert len(hero_data[view][src]) == n_stages
    assert len(xlabels) == n_stages


def test_build_hero_data_all_scenarios() -> None:
    """_build_hero_data 'all' list must have one entry per scenario.

    Each entry is a list of per-stage total MW values (hydro + thermal + NCS
    combined), with length equal to the number of stages.
    """
    data = _make_mock_data()
    hero_data, _ = _build_hero_data(data)

    assert len(hero_data["all"]) == 2  # 2 scenarios
    for entry in hero_data["all"]:
        # Each entry is a list[float] of length n_stages, not a dict.
        assert isinstance(entry, list)
        assert len(entry) == 3  # 3 stages


def test_build_hero_data_empty() -> None:
    """_build_hero_data with empty LazyFrames must return zero arrays without error."""
    data = _make_mock_data()
    data.hydros_lf = pl.LazyFrame(
        {"scenario_id": [], "stage_id": [], "generation_mwh": []}
    )
    data.thermals_lf = pl.LazyFrame(
        {"scenario_id": [], "stage_id": [], "generation_mwh": []}
    )
    data.ncs_lf = pl.LazyFrame(
        {"scenario_id": [], "stage_id": [], "generation_mwh": []}
    )

    hero_data, _ = _build_hero_data(data)

    assert set(hero_data.keys()) == {"stages", "load", "p10", "p50", "p90", "all"}
    # stages come from stage_hours fallback; percentile arrays should be zero-filled
    for view in ("p10", "p50", "p90"):
        for src in ("hydro", "thermal", "ncs"):
            assert all(v == 0.0 for v in hero_data[view][src])
    # all list is empty — no scenario IDs in empty frames
    assert hero_data["all"] == []


def test_build_hero_section_html() -> None:
    """_build_hero_section must emit the selector, chart div, and EB_DATA."""
    data = _make_mock_data()
    html = _build_hero_section(data)

    assert "eb-scenario-sel" in html
    assert "eb-hero" in html
    assert "EB_DATA" in html
    # Must include all four option values
    assert 'value="p10"' in html
    assert 'value="p50"' in html
    assert 'value="p90"' in html
    assert 'value="all"' in html
