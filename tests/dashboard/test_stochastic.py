"""Tests for the v2 Stochastic Model tab module.

Covers:
- _compute_historical_stats: stage assignment from dates
- _aggregate_by_bus: mean sum and root-sum-square std
- _aggregate_system: total system aggregation
- _extract_fitted_correlation: navigation of correlation dict
- _compute_empirical_correlation: averaged monthly correlation
- _chart_ar_order_distribution: bar chart trace count and y-sum
- _chart_noise_histogram: trace count (histogram + N(0,1))
- _chart_noise_boxplot_by_stage: 12-stage limit
- can_render: returns data.stochastic_available
- render: section titles present, fallbacks on empty data
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from novomodelo_bridge.dashboard.tabs.stochastic import (
    _aggregate_by_bus,
    _aggregate_system,
    _chart_ar_order_distribution,
    _chart_noise_boxplot_by_stage,
    _chart_noise_histogram,
    _compute_empirical_correlation,
    _compute_historical_stats,
    _extract_fitted_correlation,
    can_render,
    render,
)
from novomodelo_bridge.ui.html.css import dashboard_css

# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------


def _make_stages_data(
    stage_defs: list[tuple[int, str]],
) -> dict:
    """Build a stages_data dict from (stage_id, start_date_str) pairs."""
    stages = [
        {
            "id": sid,
            "start_date": start,
            "blocks": [{"id": 0, "hours": 720.0}],
        }
        for sid, start in stage_defs
    ]
    return {"stages": stages}


def _make_inflow_history(
    rows: list[tuple[int, str, float]],
) -> pd.DataFrame:
    """Build a windowed inflow_history DataFrame (the converted case's
    ``scenarios/inflow_history.parquet`` shape) from (hydro_id, month_start,
    value_m3s) rows, one calendar-month window per row."""
    hydro_ids, starts, values = zip(*rows) if rows else ([], [], [])
    start = pd.to_datetime(list(starts))
    return pd.DataFrame(
        {
            "hydro_id": list(hydro_ids),
            "start_date": start.date,
            "end_date": (start + pd.offsets.MonthBegin(1)).date,
            "value_m3s": list(values),
        }
    )


def _make_data(
    *,
    inflow_history: pd.DataFrame | None = None,
    inflow_stats_stoch: pd.DataFrame | None = None,
    stages_data: dict | None = None,
    hydro_bus_map: dict[int, int] | None = None,
    hydro_meta: dict[int, dict] | None = None,
    bus_names: dict[int, str] | None = None,
    non_fictitious_bus_ids: list[int] | None = None,
    stage_labels: dict[int, str] | None = None,
    stage_dates: dict[int, str] | None = None,
    stochastic_available: bool = True,
    noise_openings: pd.DataFrame | None = None,
    fitting_report: dict | None = None,
    correlation: dict | None = None,
) -> Any:
    """Build a minimal DashboardData-like mock for v2_stochastic tests."""
    data = MagicMock()
    data.stochastic_available = stochastic_available
    data.inflow_history = (
        inflow_history
        if inflow_history is not None
        else pd.DataFrame(columns=["hydro_id", "start_date", "end_date", "value_m3s"])
    )
    data.inflow_stats_stoch = (
        inflow_stats_stoch
        if inflow_stats_stoch is not None
        else pd.DataFrame(columns=["hydro_id", "stage_id", "mean_m3s", "std_m3s"])
    )
    data.stages_data = stages_data if stages_data is not None else {"stages": []}
    data.hydro_bus_map = hydro_bus_map if hydro_bus_map is not None else {}
    data.hydro_meta = hydro_meta if hydro_meta is not None else {}
    data.bus_names = bus_names if bus_names is not None else {}
    data.non_fictitious_bus_ids = (
        non_fictitious_bus_ids if non_fictitious_bus_ids is not None else []
    )
    data.stage_labels = stage_labels if stage_labels is not None else {}
    data.stage_dates = stage_dates if stage_dates is not None else {}
    data.noise_openings = (
        noise_openings
        if noise_openings is not None
        else pd.DataFrame(
            columns=["stage_id", "opening_index", "entity_index", "value"]
        )
    )
    data.fitting_report = fitting_report if fitting_report is not None else {}
    data.correlation = correlation if correlation is not None else {}
    return data


# ---------------------------------------------------------------------------
# _compute_historical_stats
# ---------------------------------------------------------------------------


class TestComputeHistoricalStats:
    """Unit tests for _compute_historical_stats."""

    def test_assigns_dates_to_correct_stages(self) -> None:
        """Dates in Jan 2024 -> stage 0, dates in Feb 2024 -> stage 1."""
        stages_data = _make_stages_data([(0, "2024-01-01"), (1, "2024-02-01")])
        inflow_history = _make_inflow_history(
            [
                (0, "2024-01-10", 100.0),
                (0, "2024-01-20", 200.0),
                (0, "2024-02-05", 300.0),
                (0, "2024-02-15", 400.0),
            ]
        )
        data = _make_data(
            inflow_history=inflow_history,
            stages_data=stages_data,
        )

        result = _compute_historical_stats(data)

        assert set(result["stage_id"].tolist()) == {0, 1}
        row_s0 = result[result["stage_id"] == 0].iloc[0]
        row_s1 = result[result["stage_id"] == 1].iloc[0]
        assert row_s0["mean_m3s"] == pytest.approx(150.0)  # mean(100, 200)
        assert row_s1["mean_m3s"] == pytest.approx(350.0)  # mean(300, 400)

    def test_std_computed_correctly(self) -> None:
        """Std of two values matches expected formula."""
        stages_data = _make_stages_data([(0, "2024-01-01")])
        inflow_history = _make_inflow_history(
            [
                (0, "2024-01-05", 100.0),
                (0, "2024-01-15", 300.0),
            ]
        )
        data = _make_data(
            inflow_history=inflow_history,
            stages_data=stages_data,
        )

        result = _compute_historical_stats(data)

        # Sample std of [100, 300] = sqrt(((100-200)^2 + (300-200)^2) / 1) = 141.42...
        assert result.iloc[0]["std_m3s"] == pytest.approx(
            pd.Series([100.0, 300.0]).std(), rel=1e-6
        )

    def test_multiple_hydros(self) -> None:
        """Each hydro is aggregated independently per stage."""
        stages_data = _make_stages_data([(0, "2024-01-01"), (1, "2024-02-01")])
        inflow_history = _make_inflow_history(
            [
                (0, "2024-01-10", 100.0),
                (1, "2024-01-10", 500.0),
                (0, "2024-02-10", 200.0),
                (1, "2024-02-10", 600.0),
            ]
        )
        data = _make_data(
            inflow_history=inflow_history,
            stages_data=stages_data,
        )

        result = _compute_historical_stats(data)

        hydros = sorted(result["hydro_id"].unique().tolist())
        assert hydros == [0, 1]

        h0_s0 = result[(result["hydro_id"] == 0) & (result["stage_id"] == 0)]
        assert h0_s0.iloc[0]["mean_m3s"] == pytest.approx(100.0)

        h1_s1 = result[(result["hydro_id"] == 1) & (result["stage_id"] == 1)]
        assert h1_s1.iloc[0]["mean_m3s"] == pytest.approx(600.0)

    def test_empty_inflow_history_returns_empty_dataframe(self) -> None:
        """Empty history yields empty result with correct columns."""
        stages_data = _make_stages_data([(0, "2024-01-01")])
        data = _make_data(stages_data=stages_data)

        result = _compute_historical_stats(data)

        assert result.empty
        assert list(result.columns) == ["hydro_id", "stage_id", "mean_m3s", "std_m3s"]

    def test_empty_stages_data_returns_empty_dataframe(self) -> None:
        """No stages means no assignment, returns empty result."""
        inflow_history = _make_inflow_history([(0, "2024-01-10", 100.0)])
        data = _make_data(
            inflow_history=inflow_history,
            stages_data={"stages": []},
        )

        result = _compute_historical_stats(data)

        assert result.empty

    def test_output_columns_are_correct(self) -> None:
        """Result always has exactly the four expected columns."""
        stages_data = _make_stages_data([(0, "2024-01-01")])
        inflow_history = _make_inflow_history([(0, "2024-01-10", 100.0)])
        data = _make_data(
            inflow_history=inflow_history,
            stages_data=stages_data,
        )

        result = _compute_historical_stats(data)

        assert list(result.columns) == ["hydro_id", "stage_id", "mean_m3s", "std_m3s"]

    def test_unsorted_stages_are_handled_correctly(self) -> None:
        """Stages given in reverse order are still assigned correctly."""
        # stages_data with id order reversed
        stages_data = {
            "stages": [
                {
                    "id": 1,
                    "start_date": "2024-02-01",
                    "blocks": [{"id": 0, "hours": 672}],
                },
                {
                    "id": 0,
                    "start_date": "2024-01-01",
                    "blocks": [{"id": 0, "hours": 744}],
                },
            ]
        }
        inflow_history = _make_inflow_history(
            [
                (0, "2024-01-15", 100.0),
                (0, "2024-02-15", 200.0),
            ]
        )
        data = _make_data(
            inflow_history=inflow_history,
            stages_data=stages_data,
        )

        result = _compute_historical_stats(data)

        assert set(result["stage_id"].tolist()) == {0, 1}
        s0 = result[result["stage_id"] == 0].iloc[0]["mean_m3s"]
        s1 = result[result["stage_id"] == 1].iloc[0]["mean_m3s"]
        assert s0 == pytest.approx(100.0)
        assert s1 == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# _aggregate_by_bus
# ---------------------------------------------------------------------------


class TestAggregateByBus:
    """Unit tests for _aggregate_by_bus."""

    def test_mean_is_sum_per_bus(self) -> None:
        """Mean for a bus is the sum of all constituent hydro means."""
        stats_df = pd.DataFrame(
            {
                "hydro_id": [0, 1, 2, 3],
                "stage_id": [0, 0, 0, 0],
                "mean_m3s": [100.0, 200.0, 300.0, 400.0],
                "std_m3s": [10.0, 20.0, 30.0, 40.0],
            }
        )
        hydro_bus_map = {0: 10, 1: 10, 2: 20, 3: 20}

        result = _aggregate_by_bus(stats_df, hydro_bus_map)

        assert 10 in result
        assert 20 in result
        assert result[10].iloc[0]["mean_m3s"] == pytest.approx(300.0)  # 100+200
        assert result[20].iloc[0]["mean_m3s"] == pytest.approx(700.0)  # 300+400

    def test_std_is_root_sum_of_squares(self) -> None:
        """Std for a bus is sqrt(sum of squared stds of its hydros)."""
        stats_df = pd.DataFrame(
            {
                "hydro_id": [0, 1],
                "stage_id": [0, 0],
                "mean_m3s": [100.0, 200.0],
                "std_m3s": [3.0, 4.0],  # RSS = sqrt(9 + 16) = 5
            }
        )
        hydro_bus_map = {0: 10, 1: 10}

        result = _aggregate_by_bus(stats_df, hydro_bus_map)

        assert result[10].iloc[0]["std_m3s"] == pytest.approx(5.0)

    def test_nan_std_treated_as_zero(self) -> None:
        """NaN std values do not propagate to the aggregated result."""
        stats_df = pd.DataFrame(
            {
                "hydro_id": [0, 1],
                "stage_id": [0, 0],
                "mean_m3s": [100.0, 200.0],
                "std_m3s": [float("nan"), 4.0],
            }
        )
        hydro_bus_map = {0: 10, 1: 10}

        result = _aggregate_by_bus(stats_df, hydro_bus_map)

        assert result[10].iloc[0]["std_m3s"] == pytest.approx(4.0)

    def test_multiple_stages(self) -> None:
        """Aggregation handles multiple stages independently."""
        stats_df = pd.DataFrame(
            {
                "hydro_id": [0, 1, 0, 1],
                "stage_id": [0, 0, 1, 1],
                "mean_m3s": [100.0, 200.0, 150.0, 250.0],
                "std_m3s": [10.0, 20.0, 15.0, 25.0],
            }
        )
        hydro_bus_map = {0: 10, 1: 10}

        result = _aggregate_by_bus(stats_df, hydro_bus_map)

        assert len(result[10]) == 2
        s0 = result[10][result[10]["stage_id"] == 0].iloc[0]
        s1 = result[10][result[10]["stage_id"] == 1].iloc[0]
        assert s0["mean_m3s"] == pytest.approx(300.0)
        assert s1["mean_m3s"] == pytest.approx(400.0)

    def test_empty_stats_returns_empty_dict(self) -> None:
        """Empty input yields empty dict."""
        stats_df = pd.DataFrame(columns=["hydro_id", "stage_id", "mean_m3s", "std_m3s"])
        result = _aggregate_by_bus(stats_df, {0: 10})

        assert result == {}

    def test_empty_hydro_bus_map_returns_empty_dict(self) -> None:
        """No bus mapping yields empty dict."""
        stats_df = pd.DataFrame(
            {
                "hydro_id": [0],
                "stage_id": [0],
                "mean_m3s": [100.0],
                "std_m3s": [10.0],
            }
        )
        result = _aggregate_by_bus(stats_df, {})

        assert result == {}

    def test_output_columns_correct(self) -> None:
        """Each bus DataFrame has stage_id, mean_m3s, std_m3s columns."""
        stats_df = pd.DataFrame(
            {
                "hydro_id": [0],
                "stage_id": [0],
                "mean_m3s": [100.0],
                "std_m3s": [10.0],
            }
        )
        result = _aggregate_by_bus(stats_df, {0: 10})

        assert list(result[10].columns) == ["stage_id", "mean_m3s", "std_m3s"]


# ---------------------------------------------------------------------------
# _aggregate_system
# ---------------------------------------------------------------------------


class TestAggregateSystem:
    """Unit tests for _aggregate_system."""

    def test_mean_is_sum_across_hydros(self) -> None:
        """System mean per stage is the sum of all hydro means."""
        stats_df = pd.DataFrame(
            {
                "hydro_id": [0, 1, 2],
                "stage_id": [0, 0, 0],
                "mean_m3s": [100.0, 200.0, 300.0],
                "std_m3s": [10.0, 20.0, 30.0],
            }
        )

        result = _aggregate_system(stats_df)

        assert len(result) == 1
        assert result.iloc[0]["mean_m3s"] == pytest.approx(600.0)

    def test_std_is_root_sum_of_squares(self) -> None:
        """System std is RSS of all hydro stds."""
        stats_df = pd.DataFrame(
            {
                "hydro_id": [0, 1],
                "stage_id": [0, 0],
                "mean_m3s": [100.0, 200.0],
                "std_m3s": [3.0, 4.0],  # RSS = 5
            }
        )

        result = _aggregate_system(stats_df)

        assert result.iloc[0]["std_m3s"] == pytest.approx(5.0)

    def test_multiple_stages(self) -> None:
        """Each stage is aggregated independently."""
        stats_df = pd.DataFrame(
            {
                "hydro_id": [0, 1, 0, 1],
                "stage_id": [0, 0, 1, 1],
                "mean_m3s": [100.0, 200.0, 150.0, 250.0],
                "std_m3s": [0.0, 0.0, 0.0, 0.0],
            }
        )

        result = _aggregate_system(stats_df)

        assert len(result) == 2
        s0 = result[result["stage_id"] == 0].iloc[0]
        s1 = result[result["stage_id"] == 1].iloc[0]
        assert s0["mean_m3s"] == pytest.approx(300.0)
        assert s1["mean_m3s"] == pytest.approx(400.0)

    def test_nan_std_treated_as_zero(self) -> None:
        """NaN stds contribute 0 to the RSS."""
        stats_df = pd.DataFrame(
            {
                "hydro_id": [0, 1],
                "stage_id": [0, 0],
                "mean_m3s": [100.0, 200.0],
                "std_m3s": [float("nan"), 5.0],
            }
        )

        result = _aggregate_system(stats_df)

        assert result.iloc[0]["std_m3s"] == pytest.approx(5.0)

    def test_empty_returns_empty_dataframe_with_correct_columns(self) -> None:
        """Empty input returns empty DataFrame with expected columns."""
        stats_df = pd.DataFrame(columns=["hydro_id", "stage_id", "mean_m3s", "std_m3s"])

        result = _aggregate_system(stats_df)

        assert result.empty
        assert list(result.columns) == ["stage_id", "mean_m3s", "std_m3s"]

    def test_output_columns_correct(self) -> None:
        """Result has stage_id, mean_m3s, std_m3s columns."""
        stats_df = pd.DataFrame(
            {
                "hydro_id": [0],
                "stage_id": [0],
                "mean_m3s": [100.0],
                "std_m3s": [10.0],
            }
        )

        result = _aggregate_system(stats_df)

        assert list(result.columns) == ["stage_id", "mean_m3s", "std_m3s"]


# ---------------------------------------------------------------------------
# can_render
# ---------------------------------------------------------------------------


class TestCanRender:
    """Unit tests for can_render."""

    def test_returns_false_when_stochastic_unavailable(self) -> None:
        data = _make_data(stochastic_available=False)
        assert can_render(data) is False

    def test_returns_true_when_stochastic_available(self) -> None:
        data = _make_data(stochastic_available=True)
        assert can_render(data) is True


# ---------------------------------------------------------------------------
# render — section titles
# ---------------------------------------------------------------------------


class TestRenderSectionTitles:
    """Integration tests for render() checking section structure."""

    def _make_full_data(self) -> Any:
        """Build a data mock with all non-empty fields for a full render."""
        stages_data = _make_stages_data([(0, "2024-01-01"), (1, "2024-02-01")])
        inflow_history = _make_inflow_history(
            [
                (0, "2024-01-10", 100.0),
                (0, "2024-01-20", 200.0),
                (1, "2024-01-10", 300.0),
                (0, "2024-02-10", 150.0),
                (1, "2024-02-10", 350.0),
            ]
        )
        inflow_stats_stoch = pd.DataFrame(
            {
                "hydro_id": [0, 0, 1, 1],
                "stage_id": [0, 1, 0, 1],
                "mean_m3s": [120.0, 160.0, 320.0, 360.0],
                "std_m3s": [12.0, 16.0, 32.0, 36.0],
            }
        )
        noise_openings = pd.DataFrame(
            {
                "stage_id": [0] * 50 + [1] * 50,
                "opening_index": list(range(50)) * 2,
                "entity_index": [0] * 100,
                "value": list(np.random.default_rng(42).standard_normal(100)),
            }
        )
        fitting_report = {
            "hydros": {
                "0": {
                    "selected_order": 1,
                    "coefficients": [],
                    "contribution_reductions": [[] for _ in range(12)],
                },
                "1": {
                    "selected_order": 2,
                    "coefficients": [],
                    "contribution_reductions": [["low_corr"] for _ in range(12)],
                },
            }
        }
        corr_matrix = [[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]]
        correlation = {
            "method": "cholesky",
            "profiles": {
                "default": {
                    "correlation_groups": [
                        {
                            "name": "group1",
                            "entities": [
                                {"type": "inflow", "id": 0},
                                {"type": "inflow", "id": 1},
                                {"type": "inflow", "id": 2},
                            ],
                            "matrix": corr_matrix,
                        }
                    ]
                }
            },
            "schedule": [],
        }
        return _make_data(
            inflow_history=inflow_history,
            inflow_stats_stoch=inflow_stats_stoch,
            stages_data=stages_data,
            hydro_bus_map={0: 10, 1: 20},
            hydro_meta={
                0: {"name": "Hydro A", "bus_id": 10},
                1: {"name": "Hydro B", "bus_id": 20},
                2: {"name": "Hydro C", "bus_id": 10},
            },
            bus_names={10: "Bus SE", 20: "Bus S"},
            non_fictitious_bus_ids=[10, 20],
            stage_labels={0: "Jan 2024", 1: "Feb 2024"},
            stage_dates={0: "2024-01-01", 1: "2024-02-01"},
            stochastic_available=True,
            noise_openings=noise_openings,
            fitting_report=fitting_report,
            correlation=correlation,
        )

    def test_render_contains_all_six_section_divs(self) -> None:
        """render() output contains all six collapsible section divs."""
        data = self._make_full_data()

        html = render(data)

        # Each collapsible section wraps content in div class="collapsible-section"
        assert html.count('class="collapsible-section"') == 6

    def test_render_contains_system_wide_inflow_title(self) -> None:
        data = self._make_full_data()

        html = render(data)

        assert "System-Wide Inflow" in html

    def test_render_contains_inflow_by_bus_title(self) -> None:
        data = self._make_full_data()

        html = render(data)

        assert "Inflow by Bus" in html

    def test_render_contains_per_hydro_explorer_title(self) -> None:
        data = self._make_full_data()

        html = render(data)

        assert "Per-Hydro Inflow Explorer" in html

    def test_render_returns_string(self) -> None:
        data = self._make_full_data()

        html = render(data)

        assert isinstance(html, str)
        assert len(html) > 0

    def test_explorer_wrapper_uses_responsive_container_class(self) -> None:
        """The per-hydro explorer wrapper carries the class that stacks its
        panes below 1024px, not the undefined `.explorer-layout` class."""
        data = self._make_full_data()

        html = render(data)

        assert 'class="explorer-container"' in html
        assert "explorer-layout" not in html
        css = dashboard_css()
        assert "@media (max-width: 1023px)" in css
        media_block = css.split("@media (max-width: 1023px)", 1)[1]
        assert ".explorer-container" in media_block
        assert "flex-direction: column" in media_block


# ---------------------------------------------------------------------------
# render — empty inflow_history fallbacks
# ---------------------------------------------------------------------------


class TestRenderEmptyHistoryFallbacks:
    """Tests for render() fallback behavior when inflow_history is empty."""

    def test_all_three_sections_contain_no_historical_inflow_data(self) -> None:
        """All three sections show fallback text when inflow_history is empty."""
        data = _make_data(
            stages_data=_make_stages_data([(0, "2024-01-01")]),
            hydro_bus_map={0: 10},
            hydro_meta={0: {"name": "H0", "bus_id": 10}},
            non_fictitious_bus_ids=[10],
        )

        html = render(data)

        assert html.count("No historical inflow data") == 3

    def test_no_plotly_chart_divs_when_history_empty(self) -> None:
        """No Plotly chart divs in output when inflow_history is empty."""
        data = _make_data(
            stages_data=_make_stages_data([(0, "2024-01-01")]),
            hydro_bus_map={0: 10},
            hydro_meta={0: {"name": "H0", "bus_id": 10}},
            non_fictitious_bus_ids=[10],
        )

        html = render(data)

        # Plotly divs contain class="plotly-graph-div"
        assert "plotly-graph-div" not in html


# ---------------------------------------------------------------------------
# render — empty hydro_meta fallback (Section C)
# ---------------------------------------------------------------------------


class TestRenderEmptyHydroMeta:
    """Tests for Section C fallback when hydro_meta is empty."""

    def test_section_c_shows_no_hydro_data_when_hydro_meta_empty(self) -> None:
        stages_data = _make_stages_data([(0, "2024-01-01")])
        inflow_history = _make_inflow_history([(0, "2024-01-10", 100.0)])
        data = _make_data(
            inflow_history=inflow_history,
            stages_data=stages_data,
            hydro_meta={},
            hydro_bus_map={0: 10},
            non_fictitious_bus_ids=[10],
        )

        html = render(data)

        assert "No hydro data" in html


# ---------------------------------------------------------------------------
# render — empty hydro_bus_map fallback (Section B)
# ---------------------------------------------------------------------------


class TestRenderEmptyBusMap:
    """Tests for Section B fallback when hydro_bus_map is empty."""

    def test_section_b_shows_no_bus_mapping_when_bus_map_empty(self) -> None:
        stages_data = _make_stages_data([(0, "2024-01-01")])
        inflow_history = _make_inflow_history([(0, "2024-01-10", 100.0)])
        data = _make_data(
            inflow_history=inflow_history,
            stages_data=stages_data,
            hydro_bus_map={},
            hydro_meta={0: {"name": "H0", "bus_id": 10}},
            non_fictitious_bus_ids=[],
        )

        html = render(data)

        assert "No bus mapping available" in html


# ---------------------------------------------------------------------------
# render — synthetic-only (empty inflow_stats_stoch)
# ---------------------------------------------------------------------------


class TestRenderNoSyntheticData:
    """render() works when inflow_stats_stoch is empty (historical-only chart)."""

    def test_renders_without_error_when_synth_empty(self) -> None:
        stages_data = _make_stages_data([(0, "2024-01-01"), (1, "2024-02-01")])
        inflow_history = _make_inflow_history(
            [
                (0, "2024-01-10", 100.0),
                (0, "2024-02-10", 200.0),
            ]
        )
        data = _make_data(
            inflow_history=inflow_history,
            inflow_stats_stoch=pd.DataFrame(
                columns=["hydro_id", "stage_id", "mean_m3s", "std_m3s"]
            ),
            stages_data=stages_data,
            hydro_bus_map={0: 10},
            hydro_meta={0: {"name": "Hydro A", "bus_id": 10}},
            bus_names={10: "Bus SE"},
            non_fictitious_bus_ids=[10],
            stage_labels={0: "Jan 2024", 1: "Feb 2024"},
            stage_dates={0: "2024-01-01", 1: "2024-02-01"},
        )

        html = render(data)

        assert "System-Wide Inflow" in html
        assert "plotly-graph-div" in html


# ---------------------------------------------------------------------------
# _extract_fitted_correlation
# ---------------------------------------------------------------------------


def _make_corr_dict(
    hydro_ids: list[int],
    matrix: list[list[float]],
) -> dict:
    """Build a minimal correlation dict with one group."""
    entities = [{"type": "inflow", "id": hid} for hid in hydro_ids]
    return {
        "method": "cholesky",
        "profiles": {
            "default": {
                "correlation_groups": [
                    {
                        "name": "group1",
                        "entities": entities,
                        "matrix": matrix,
                    }
                ]
            }
        },
        "schedule": [],
    }


class TestExtractFittedCorrelation:
    """Unit tests for _extract_fitted_correlation."""

    def test_returns_correct_hydro_ids_and_matrix_shape(self) -> None:
        """3 entities with 3x3 matrix returns (hydro_ids, 3x3 array)."""
        matrix = [[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]]
        corr = _make_corr_dict([10, 20, 30], matrix)

        result = _extract_fitted_correlation(corr)

        assert result is not None
        hydro_ids, arr = result
        assert hydro_ids == [10, 20, 30]
        assert arr.shape == (3, 3)
        assert arr[0, 1] == pytest.approx(0.5)

    def test_empty_dict_returns_none(self) -> None:
        """Empty dict returns None."""
        assert _extract_fitted_correlation({}) is None

    def test_missing_profiles_key_returns_none(self) -> None:
        """Dict without 'profiles' returns None."""
        assert _extract_fitted_correlation({"method": "cholesky"}) is None

    def test_empty_entities_returns_none(self) -> None:
        """Group with empty entities list returns None."""
        corr = _make_corr_dict([], [])
        assert _extract_fitted_correlation(corr) is None

    def test_entity_ids_are_integers(self) -> None:
        """Returned hydro_ids are Python ints, not floats or strings."""
        matrix = [[1.0, 0.8], [0.8, 1.0]]
        corr = _make_corr_dict([5, 7], matrix)

        result = _extract_fitted_correlation(corr)

        assert result is not None
        hydro_ids, _ = result
        assert all(isinstance(hid, int) for hid in hydro_ids)


# ---------------------------------------------------------------------------
# _compute_empirical_correlation
# ---------------------------------------------------------------------------


class TestComputeEmpiricalCorrelation:
    """Unit tests for _compute_empirical_correlation."""

    def _make_history(self, n_hydros: int = 3, n_years: int = 15) -> pd.DataFrame:
        """Synthetic windowed inflow history: one calendar-month window per
        hydro over *n_years* years, enough observations per month for the
        ``min_periods=10`` requirement in ``_compute_empirical_correlation``.
        """
        rng = np.random.default_rng(99)
        rows: list[dict] = []
        for year in range(2000, 2000 + n_years):
            for month in range(1, 13):
                start = pd.Timestamp(year=year, month=month, day=1)
                for hid in range(n_hydros):
                    rows.append(
                        {
                            "hydro_id": hid,
                            "start_date": start.date(),
                            "end_date": (start + pd.offsets.MonthBegin(1)).date(),
                            "value_m3s": float(rng.uniform(10, 1000)),
                        }
                    )
        return pd.DataFrame(rows)

    def test_returns_symmetric_matrix(self) -> None:
        """Result matrix is symmetric."""
        history = self._make_history(3)
        hydro_ids = [0, 1, 2]

        result = _compute_empirical_correlation(history, hydro_ids)

        assert result is not None
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result, result.T, atol=1e-10)

    def test_diagonal_elements_are_one(self) -> None:
        """Diagonal elements of the result are all 1.0."""
        history = self._make_history(3)
        hydro_ids = [0, 1, 2]

        result = _compute_empirical_correlation(history, hydro_ids)

        assert result is not None
        np.testing.assert_allclose(np.diag(result), np.ones(3), atol=1e-10)

    def test_empty_history_returns_none(self) -> None:
        """Empty inflow_history returns None."""
        empty = pd.DataFrame(
            columns=["hydro_id", "start_date", "end_date", "value_m3s"]
        )
        assert _compute_empirical_correlation(empty, [0, 1, 2]) is None

    def test_single_hydro_returns_none(self) -> None:
        """Single hydro in hydro_ids returns None (need >= 2)."""
        history = self._make_history(1)
        assert _compute_empirical_correlation(history, [0]) is None

    def test_values_in_range(self) -> None:
        """All values in the result are in [-1, 1]."""
        history = self._make_history(3)
        hydro_ids = [0, 1, 2]

        result = _compute_empirical_correlation(history, hydro_ids)

        assert result is not None
        assert np.all(result >= -1.0 - 1e-10)
        assert np.all(result <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# _chart_ar_order_distribution
# ---------------------------------------------------------------------------


class TestChartArOrderDistribution:
    """Unit tests for _chart_ar_order_distribution.

    The current implementation signature is:
        _chart_ar_order_distribution(fitting_report, hydro_meta)

    The x-axis covers orders from 1 through max_order, plus an order-0 bin
    only when some hydro selects order 0.
    """

    # Minimal hydro_meta for tests that don't care about names.
    _EMPTY_HYDRO_META: dict[int, dict] = {}

    def test_bar_trace_y_values_sum_to_hydro_count(self) -> None:
        """Bar trace y-values sum equals the number of hydros in the report."""
        fitting_report = {
            "hydros": {
                "0": {
                    "selected_order": 1,
                    "coefficients": [],
                    "contribution_reductions": [],
                },
                "1": {
                    "selected_order": 2,
                    "coefficients": [],
                    "contribution_reductions": [],
                },
                "2": {
                    "selected_order": 2,
                    "coefficients": [],
                    "contribution_reductions": [],
                },
                "3": {
                    "selected_order": 3,
                    "coefficients": [],
                    "contribution_reductions": [],
                },
                "4": {
                    "selected_order": 1,
                    "coefficients": [],
                    "contribution_reductions": [],
                },
            }
        }

        fig = _chart_ar_order_distribution(fitting_report, self._EMPTY_HYDRO_META)

        assert len(fig.data) == 1
        total = sum(int(v) for v in fig.data[0].y)
        assert total == 5

    def test_bar_chart_x_covers_all_orders(self) -> None:
        """x-axis covers 1 through max_order (no order-0 bin)."""
        fitting_report = {
            "hydros": {
                "0": {
                    "selected_order": 3,
                    "coefficients": [],
                    "contribution_reductions": [],
                },
            }
        }

        fig = _chart_ar_order_distribution(fitting_report, self._EMPTY_HYDRO_META)

        assert list(fig.data[0].x) == [1, 2, 3]

    def test_order_zero_gets_its_own_bin(self) -> None:
        fitting_report = {
            "hydros": {
                "0": {"selected_order": 0},
                "1": {"selected_order": 2},
            }
        }

        fig = _chart_ar_order_distribution(fitting_report, self._EMPTY_HYDRO_META)

        assert list(fig.data[0].x) == [0, 1, 2]
        assert list(fig.data[0].y) == [1, 0, 1]

    def test_order_counts_correct(self) -> None:
        """Counts match the manually expected distribution.

        Orders 1, 2, 3 have counts 2, 2, 1 (order-0 bin is not emitted).
        """
        fitting_report = {
            "hydros": {
                "0": {
                    "selected_order": 1,
                    "coefficients": [],
                    "contribution_reductions": [],
                },
                "1": {
                    "selected_order": 2,
                    "coefficients": [],
                    "contribution_reductions": [],
                },
                "2": {
                    "selected_order": 2,
                    "coefficients": [],
                    "contribution_reductions": [],
                },
                "3": {
                    "selected_order": 3,
                    "coefficients": [],
                    "contribution_reductions": [],
                },
                "4": {
                    "selected_order": 1,
                    "coefficients": [],
                    "contribution_reductions": [],
                },
            }
        }

        fig = _chart_ar_order_distribution(fitting_report, self._EMPTY_HYDRO_META)

        # x=[1,2,3] -> counts [2,2,1]
        assert list(fig.data[0].y) == [2, 2, 1]


# ---------------------------------------------------------------------------
# _chart_noise_histogram
# ---------------------------------------------------------------------------


class TestChartNoiseHistogram:
    """Unit tests for _chart_noise_histogram."""

    def _make_noise_df(self, n: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        return pd.DataFrame(
            {
                "stage_id": [0] * n,
                "opening_index": list(range(n)),
                "entity_index": [0] * n,
                "value": rng.standard_normal(n).tolist(),
            }
        )

    def test_figure_has_two_traces(self) -> None:
        """Figure contains exactly 2 traces: histogram + N(0,1) reference."""
        df = self._make_noise_df(100)

        fig = _chart_noise_histogram(df)

        assert len(fig.data) == 2

    def test_first_trace_is_histogram(self) -> None:
        """First trace is a Histogram."""
        import plotly.graph_objects as go

        df = self._make_noise_df(100)
        fig = _chart_noise_histogram(df)

        assert isinstance(fig.data[0], go.Histogram)

    def test_second_trace_is_scatter(self) -> None:
        """Second trace is a Scatter (N(0,1) reference curve)."""
        import plotly.graph_objects as go

        df = self._make_noise_df(100)
        fig = _chart_noise_histogram(df)

        assert isinstance(fig.data[1], go.Scatter)


# ---------------------------------------------------------------------------
# _chart_noise_boxplot_by_stage
# ---------------------------------------------------------------------------


class TestChartNoiseBoxplotByStage:
    """Unit tests for _chart_noise_boxplot_by_stage.

    The current implementation uses a single vectorised Box trace where the
    ``x`` array contains one label per stage and the pre-computed stats are
    passed via ``median``, ``q1``, ``q3``, ``lowerfence``, and
    ``upperfence`` arrays.  All stages therefore appear in one trace object.
    """

    def _make_noise_df_stages(
        self, n_stages: int, n_per_stage: int = 20
    ) -> pd.DataFrame:
        rng = np.random.default_rng(13)
        stage_ids = []
        values = []
        for sid in range(n_stages):
            stage_ids.extend([sid] * n_per_stage)
            values.extend(rng.standard_normal(n_per_stage).tolist())
        return pd.DataFrame(
            {
                "stage_id": stage_ids,
                "opening_index": list(range(n_stages * n_per_stage)),
                "entity_index": [0] * (n_stages * n_per_stage),
                "value": values,
            }
        )

    def test_limits_to_12_stages_when_more_provided(self) -> None:
        """When 15 stages are present, the single Box trace has all 15 x-values.

        The current implementation renders all stages without capping at 12.
        """
        df = self._make_noise_df_stages(15)

        fig = _chart_noise_boxplot_by_stage(df, {})

        # One vectorised Box trace; x has one entry per stage.
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 15

    def test_uses_all_stages_when_fewer_than_12(self) -> None:
        """When only 5 stages are present, the Box trace x array has 5 entries."""
        df = self._make_noise_df_stages(5)

        fig = _chart_noise_boxplot_by_stage(df, {})

        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 5

    def test_stage_labels_used_as_trace_names(self) -> None:
        """Stage labels from stage_labels dict appear as the x-axis values."""
        df = self._make_noise_df_stages(3)
        stage_labels = {0: "Jan 2024", 1: "Feb 2024", 2: "Mar 2024"}

        fig = _chart_noise_boxplot_by_stage(df, stage_labels)

        assert list(fig.data[0].x) == ["Jan 2024", "Feb 2024", "Mar 2024"]


# ---------------------------------------------------------------------------
# render — integration tests for sections D, E, F
# ---------------------------------------------------------------------------


class TestRenderSectionsDEF:
    """Integration tests verifying sections D, E, F render correctly."""

    def test_all_six_sections_present_on_full_data(self) -> None:
        """Full data produces HTML with exactly 6 collapsible sections."""
        stages_data = _make_stages_data([(0, "2024-01-01"), (1, "2024-02-01")])
        inflow_history = _make_inflow_history(
            [(0, "2024-01-10", 100.0), (0, "2024-02-10", 200.0)]
        )
        rng = np.random.default_rng(0)
        noise_openings = pd.DataFrame(
            {
                "stage_id": [0] * 60 + [1] * 60,
                "opening_index": list(range(60)) * 2,
                "entity_index": [0] * 120,
                "value": rng.standard_normal(120).tolist(),
            }
        )
        fitting_report = {
            "hydros": {
                "0": {
                    "selected_order": 1,
                    "coefficients": [],
                    "contribution_reductions": [[] for _ in range(12)],
                },
            }
        }
        corr_matrix = [[1.0, 0.6], [0.6, 1.0]]
        correlation = {
            "method": "cholesky",
            "profiles": {
                "default": {
                    "correlation_groups": [
                        {
                            "name": "g",
                            "entities": [
                                {"type": "inflow", "id": 0},
                                {"type": "inflow", "id": 1},
                            ],
                            "matrix": corr_matrix,
                        }
                    ]
                }
            },
            "schedule": [],
        }
        data = _make_data(
            inflow_history=inflow_history,
            stages_data=stages_data,
            hydro_bus_map={0: 10, 1: 10},
            hydro_meta={
                0: {"name": "HA", "bus_id": 10},
                1: {"name": "HB", "bus_id": 10},
            },
            bus_names={10: "BusSE"},
            non_fictitious_bus_ids=[10],
            stage_labels={0: "Jan 2024", 1: "Feb 2024"},
            stage_dates={0: "2024-01-01", 1: "2024-02-01"},
            noise_openings=noise_openings,
            fitting_report=fitting_report,
            correlation=correlation,
        )

        html = render(data)

        assert html.count('class="collapsible-section"') == 6

    def test_section_d_empty_correlation_shows_fallback(self) -> None:
        """Empty correlation dict shows fallback in D; sections E and F still render."""
        rng = np.random.default_rng(1)
        noise_openings = pd.DataFrame(
            {
                "stage_id": [0] * 30,
                "opening_index": list(range(30)),
                "entity_index": [0] * 30,
                "value": rng.standard_normal(30).tolist(),
            }
        )
        fitting_report = {
            "hydros": {
                "0": {
                    "selected_order": 2,
                    "coefficients": [],
                    "contribution_reductions": [[] for _ in range(12)],
                },
            }
        }
        data = _make_data(
            correlation={},
            noise_openings=noise_openings,
            fitting_report=fitting_report,
        )

        html = render(data)

        assert "No correlation data available" in html
        assert "Noise Diagnostics" in html
        assert "AR Order Distribution" in html

    def test_section_e_empty_noise_shows_fallback(self) -> None:
        """Empty noise_openings shows fallback in section E."""
        data = _make_data(
            noise_openings=pd.DataFrame(
                columns=["stage_id", "opening_index", "entity_index", "value"]
            ),
            fitting_report={
                "hydros": {
                    "0": {
                        "selected_order": 1,
                        "coefficients": [],
                        "contribution_reductions": [],
                    }
                }
            },
        )

        html = render(data)

        assert "No noise data" in html

    @pytest.mark.parametrize("fitting_report", [{}, {"hydros": {}}])
    def test_section_f_without_fitted_hydros_shows_fallback(
        self, fitting_report: dict
    ) -> None:
        """A missing report, or one with no fitted hydro (no AR estimation),
        shows the fallback instead of an empty chart."""
        html = render(_make_data(fitting_report=fitting_report))

        assert "No AR model was fitted for this case" in html
        assert "v2-stoch-ar-order" not in html
