"""Unit tests for novomodelo_bridge.dashboard.tabs.overview.

Covers module constants, can_render, helper functions, and the full
render() path including the empty-costs degradation branch.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import plotly.graph_objects as go
import polars as pl

import novomodelo_bridge.dashboard.tabs.overview as overview_mod
from novomodelo_bridge.dashboard.tabs.overview import (
    _chart_training_mini,
    _compute_gen_gwh,
    _format_duration,
    _run_identity_strip,
    _run_status_strip,
    can_render,
    render,
)

# ---------------------------------------------------------------------------
# Helpers / data factories
# ---------------------------------------------------------------------------


def _make_costs_df() -> pd.DataFrame:
    """Return a minimal costs DataFrame with two scenarios and three stages."""
    rows = []
    for scenario_id in range(2):
        for stage_id in range(3):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage_id": stage_id,
                    "thermal_generation_cost": float(1000 * (stage_id + 1)),
                    "deficit_cost_depth_1": float(50 * (stage_id + 1)),
                }
            )
    return pd.DataFrame(rows)


def _make_conv_df(n: int = 50) -> pd.DataFrame:
    """Return a minimal convergence DataFrame with *n* rows."""
    return pd.DataFrame(
        {
            "iteration": list(range(1, n + 1)),
            "lower_bound": [float(i * 100) for i in range(n)],
            "upper_bound_mean": [float(i * 110) for i in range(n)],
            "upper_bound_std": [float(5) for _ in range(n)],
            "gap_percent": [float(10 - i * 0.2) for i in range(n)],
            "cuts_active": [int(i * 10) for i in range(n)],
            "cuts_added": [5 for _ in range(n)],
        }
    )


def _make_mock_lf(gwh_value: float = 5000.0) -> pl.LazyFrame:
    """Return a LazyFrame with a single scenario/stage yielding *gwh_value* MWh."""
    return pl.DataFrame(
        {
            "scenario_id": [0],
            "stage_id": [0],
            "generation_mwh": [gwh_value],
        }
    ).lazy()


def _make_training_metadata(
    training_manifest: dict | None,
    metadata: dict | None,
) -> dict:
    """Build a training_metadata dict from legacy test parameters.

    The implementation reads from ``data.training_metadata`` using keys
    ``novomodelo_version``, ``started_at``, ``duration_seconds``, and a nested
    ``convergence`` sub-dict with ``termination_reason``.

    Legacy test helpers passed ``training_manifest`` and ``metadata``
    separately.  This function unifies them into the shape the
    implementation now expects.
    """
    result: dict = {}
    if metadata:
        # metadata may carry {"version": "1.0"} -> map to software_version
        if "version" in metadata:
            result["software_version"] = metadata["version"]

    if training_manifest:
        if "elapsed_seconds" in training_manifest:
            result["duration_seconds"] = training_manifest["elapsed_seconds"]
        if "start_time" in training_manifest:
            result["started_at"] = training_manifest["start_time"]
        if "termination_reason" in training_manifest:
            result.setdefault("convergence", {})["termination_reason"] = (
                training_manifest["termination_reason"]
            )

    return result


def _make_policy_metadata(training_manifest: dict | None) -> dict:
    """Build a policy_metadata dict from the legacy training_manifest."""
    result: dict = {}
    if training_manifest and "policy_states" in training_manifest:
        result["state_dimension"] = training_manifest["policy_states"]
    return result


def _make_mock_data(
    *,
    costs: pd.DataFrame | None = None,
    conv: pd.DataFrame | None = None,
    training_manifest: dict | None = None,
    metadata: dict | None = None,
    case_name: str = "test-case",
    n_scenarios: int = 100,
    n_stages: int = 60,
    discount_rate: float = 0.12,
    hydro_meta: dict | None = None,
    thermal_meta: dict | None = None,
    non_fictitious_bus_ids: list | None = None,
    line_meta: list | None = None,
) -> MagicMock:
    """Build a minimal MagicMock that satisfies the DashboardData interface."""
    data = MagicMock()
    data.case_name = case_name
    data.n_scenarios = n_scenarios
    data.n_stages = n_stages
    data.discount_rate = discount_rate
    data.training_metadata = _make_training_metadata(training_manifest, metadata)
    data.policy_metadata = _make_policy_metadata(training_manifest)
    data.conv = conv if conv is not None else _make_conv_df()
    data.costs = costs if costs is not None else _make_costs_df()
    data.stage_labels = {0: "Jan 2024", 1: "Feb 2024", 2: "Mar 2024"}
    data.stage_dates = {0: "2024-01-01", 1: "2024-02-01", 2: "2024-03-01"}
    data.stage_hours = {0: 744.0, 1: 672.0, 2: 744.0}
    data.hydros_lf = _make_mock_lf(12_000.0)
    data.thermals_lf = _make_mock_lf(8_000.0)
    data.ncs_lf = _make_mock_lf(3_000.0)
    data.hydro_meta = hydro_meta if hydro_meta is not None else {0: {}, 1: {}, 2: {}}
    data.thermal_meta = thermal_meta if thermal_meta is not None else {0: {}, 1: {}}
    data.non_fictitious_bus_ids = (
        non_fictitious_bus_ids if non_fictitious_bus_ids is not None else [0, 1, 2, 3]
    )
    data.line_meta = line_meta if line_meta is not None else [{}, {}]
    return data


# ---------------------------------------------------------------------------
# test_tab_constants
# ---------------------------------------------------------------------------


def test_tab_constants() -> None:
    """Module-level constants must match their expected values exactly."""
    assert overview_mod.TAB_ID == "tab-overview"
    assert overview_mod.TAB_LABEL == "Overview"
    assert overview_mod.TAB_ORDER == 0


# ---------------------------------------------------------------------------
# test_can_render_returns_true
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
# test_run_identity_strip
# ---------------------------------------------------------------------------


def test_run_identity_strip_contains_expected_values() -> None:
    """_run_identity_strip must embed case_name, scenario count, stage count,
    discount rate as percentage, and solver version."""
    data = _make_mock_data(
        case_name="test",
        n_scenarios=100,
        n_stages=60,
        discount_rate=0.12,
        metadata={"version": "1.0"},
    )
    html = _run_identity_strip(data)

    assert "test" in html
    assert "100" in html
    assert "60" in html
    # 0.12 -> 12.0%
    assert "12" in html
    assert "1.0" in html


def test_run_identity_strip_missing_version_shows_na() -> None:
    """When metadata has no 'version' key, 'N/A' must appear in the strip."""
    data = _make_mock_data(metadata={})
    html = _run_identity_strip(data)
    assert "N/A" in html


# ---------------------------------------------------------------------------
# test_run_status_strip
# ---------------------------------------------------------------------------


def test_run_status_strip_with_manifest() -> None:
    """When training_manifest contains termination_reason, both the reason
    and the iteration count must appear in the returned HTML."""
    conv = _make_conv_df(50)
    data = _make_mock_data(
        training_manifest={"termination_reason": "gap_tolerance"},
        conv=conv,
    )
    html = _run_status_strip(data)

    assert "gap_tolerance" in html
    assert "50" in html


def test_run_status_strip_empty_manifest() -> None:
    """When training_metadata is empty, the fallback message must appear."""
    data = _make_mock_data(training_manifest={})
    html = _run_status_strip(data)
    assert "No training metadata available" in html


# ---------------------------------------------------------------------------
# test_format_duration
# ---------------------------------------------------------------------------


def test_format_duration_full_hours_and_minutes() -> None:
    """_format_duration must format seconds as hours and minutes."""
    assert _format_duration(15120) == "4h 12min"


def test_format_duration_zero_seconds() -> None:
    """_format_duration(0) must return '0h 0min'."""
    assert _format_duration(0) == "0h 0min"


def test_format_duration_none_returns_na() -> None:
    """_format_duration(None) must return 'N/A'."""
    assert _format_duration(None) == "N/A"


def test_format_duration_non_numeric_returns_na() -> None:
    """_format_duration with a non-numeric value must return 'N/A'."""
    assert _format_duration("not-a-number") == "N/A"


# ---------------------------------------------------------------------------
# test_run_identity_strip — new fields
# ---------------------------------------------------------------------------


def test_run_identity_strip_hydros_count() -> None:
    """_run_identity_strip must include the hydro count from hydro_meta."""
    data = _make_mock_data(hydro_meta={0: {}, 1: {}, 2: {}})
    html = _run_identity_strip(data)
    assert "Hydros:</strong> 3" in html


def test_run_identity_strip_thermals_count() -> None:
    """_run_identity_strip must include the thermal count from thermal_meta."""
    data = _make_mock_data(thermal_meta={0: {}, 1: {}})
    html = _run_identity_strip(data)
    assert "Thermals:</strong> 2" in html


def test_run_identity_strip_buses_count() -> None:
    """_run_identity_strip must include the bus count from non_fictitious_bus_ids."""
    data = _make_mock_data(non_fictitious_bus_ids=[10, 20, 30, 40, 50])
    html = _run_identity_strip(data)
    assert "Buses:</strong> 5" in html


def test_run_identity_strip_lines_count() -> None:
    """_run_identity_strip must include the line count from line_meta."""
    data = _make_mock_data(line_meta=[{}, {}, {}])
    html = _run_identity_strip(data)
    assert "Lines:</strong> 3" in html


def test_run_identity_strip_duration_formatted() -> None:
    """_run_identity_strip must show a formatted duration when elapsed_seconds is set.

    15120 seconds == 4h 12min.
    """
    data = _make_mock_data(
        training_manifest={"elapsed_seconds": 15120, "termination_reason": "converged"}
    )
    html = _run_identity_strip(data)
    assert "4h 12min" in html


def test_run_identity_strip_duration_na_when_missing() -> None:
    """_run_identity_strip must show 'N/A' for duration when elapsed_seconds absent."""
    data = _make_mock_data(training_manifest={"termination_reason": "converged"})
    html = _run_identity_strip(data)
    assert "Duration:</strong> N/A" in html


def test_run_identity_strip_run_date_shown() -> None:
    """_run_identity_strip must include the run date from training_manifest."""
    data = _make_mock_data(training_manifest={"start_time": "2024-06-01T08:00:00"})
    html = _run_identity_strip(data)
    assert "2024-06-01T08:00:00" in html


def test_run_identity_strip_run_date_na_when_manifest_empty() -> None:
    """_run_identity_strip must show 'N/A' for run date when manifest is empty."""
    data = _make_mock_data(training_manifest={})
    html = _run_identity_strip(data)
    assert "Run Date:</strong> N/A" in html


# ---------------------------------------------------------------------------
# test_run_status_strip — new fields
# ---------------------------------------------------------------------------


def test_run_status_strip_lower_bound_and_cuts() -> None:
    """_run_status_strip must show lower bound and active cuts from conv."""
    conv = pd.DataFrame(
        {
            "iteration": [1, 2],
            "lower_bound": [5000.0, 5430.0],
            "upper_bound_mean": [6000.0, 5800.0],
            "cuts_active": [600, 702],
            "cuts_added": [50, 52],
        }
    )
    data = _make_mock_data(
        training_manifest={"termination_reason": "gap_tolerance"},
        conv=conv,
    )
    html = _run_status_strip(data)

    assert "Lower bound:" in html
    assert "Active cuts:" in html
    assert "702" in html


def test_run_status_strip_policy_states_shown() -> None:
    """_run_status_strip must include policy_states from training_manifest."""
    conv = _make_conv_df(5)
    data = _make_mock_data(
        training_manifest={
            "termination_reason": "gap_tolerance",
            "policy_states": 12345,
        },
        conv=conv,
    )
    html = _run_status_strip(data)
    assert "Policy states:" in html
    assert "12345" in html


def test_run_status_strip_policy_states_na_when_missing() -> None:
    """_run_status_strip must show 'N/A' for policy_states when key is absent."""
    conv = _make_conv_df(5)
    data = _make_mock_data(
        training_manifest={"termination_reason": "converged"},
        conv=conv,
    )
    html = _run_status_strip(data)
    assert "Policy states:</strong> N/A" in html


def test_run_status_strip_no_cuts_fields_when_conv_empty() -> None:
    """_run_status_strip must omit lower bound / cuts fields when conv is empty."""
    data = _make_mock_data(
        training_manifest={"termination_reason": "converged"},
        conv=pd.DataFrame(),
    )
    html = _run_status_strip(data)
    assert "Lower bound:" not in html
    assert "Active cuts:" not in html
    assert "Total cuts:" not in html


# ---------------------------------------------------------------------------
# test_compute_gen_gwh
# ---------------------------------------------------------------------------


def test_compute_gen_gwh_returns_gwh() -> None:
    """_compute_gen_gwh must convert MWh sum to GWh (divide by 1e3)."""
    lf = pl.DataFrame(
        {
            "scenario_id": [0, 1],
            "stage_id": [0, 0],
            "generation_mwh": [2000.0, 4000.0],
        }
    ).lazy()
    # Mean across scenarios: (2000 + 4000) / 2 = 3000 MWh -> 3.0 GWh
    result = _compute_gen_gwh(lf)
    assert abs(result - 3.0) < 0.001


def test_compute_gen_gwh_empty_lazyframe_returns_zero() -> None:
    """_compute_gen_gwh must return 0.0 for an empty LazyFrame."""
    lf = pl.LazyFrame({"scenario_id": [], "generation_mwh": []})
    result = _compute_gen_gwh(lf)
    assert result == 0.0


# ---------------------------------------------------------------------------
# test_chart_training_mini
# ---------------------------------------------------------------------------


def test_chart_training_mini_returns_figure_for_valid_conv() -> None:
    """_chart_training_mini must return a Figure when conv has required columns.

    When upper_bound_std is present the implementation adds two invisible band
    traces (fill=tonexty) in addition to the lower_bound and upper_bound_mean
    lines, for a total of 4 traces.
    """
    conv = _make_conv_df(10)
    fig = _chart_training_mini(conv)
    assert isinstance(fig, go.Figure)
    # 2 band traces (upper/lower of ±1-std confidence band) + lower_bound +
    # upper_bound_mean
    assert len(fig.data) == 4


def test_chart_training_mini_returns_none_for_empty_conv() -> None:
    """_chart_training_mini must return None for an empty DataFrame."""
    result = _chart_training_mini(pd.DataFrame())
    assert result is None


def test_chart_training_mini_returns_none_for_missing_columns() -> None:
    """_chart_training_mini must return None when required columns are absent."""
    conv = pd.DataFrame({"iteration": [1, 2], "lower_bound": [100.0, 110.0]})
    result = _chart_training_mini(conv)
    assert result is None


# ---------------------------------------------------------------------------
# test_render_with_full_data
# ---------------------------------------------------------------------------


def test_render_with_full_data_contains_required_substrings() -> None:
    """render() on a fully-populated mock must contain all required substrings."""
    data = _make_mock_data()
    # Patch _stage_avg_mw so it doesn't try to execute the LazyFrame
    stage_mw = {0: 100.0, 1: 110.0, 2: 105.0}
    with patch(
        "novomodelo_bridge.dashboard.tabs.overview._stage_avg_mw",
        return_value=stage_mw,
    ):
        html = render(data)

    assert "Expected Cost" in html
    assert "Hydro" in html
    assert "Thermal" in html
    assert "NCS" in html
    assert "Cost Breakdown" in html
    assert "Training Convergence" in html


def test_render_termination_reason_appears_in_output() -> None:
    """render() must surface the termination_reason from training_manifest."""
    data = _make_mock_data(
        training_manifest={"termination_reason": "gap_tolerance"},
    )
    with patch(
        "novomodelo_bridge.dashboard.tabs.overview._stage_avg_mw",
        return_value={0: 100.0},
    ):
        html = render(data)

    assert "gap_tolerance" in html


# ---------------------------------------------------------------------------
# test_render_with_empty_costs
# ---------------------------------------------------------------------------


def test_render_with_empty_costs_does_not_raise_and_contains_placeholder() -> None:
    """render() with empty costs must not raise and must include placeholder text."""
    data = _make_mock_data(costs=pd.DataFrame())

    with patch(
        "novomodelo_bridge.dashboard.tabs.overview._stage_avg_mw",
        return_value={0: 100.0},
    ):
        html = render(data)

    assert "No cost data" in html


def test_render_with_empty_conv_shows_no_convergence_placeholder() -> None:
    """render() with empty conv must emit the no-convergence placeholder."""
    data = _make_mock_data(conv=pd.DataFrame())

    with patch(
        "novomodelo_bridge.dashboard.tabs.overview._stage_avg_mw",
        return_value={0: 100.0},
    ):
        html = render(data)

    assert "No convergence data" in html
