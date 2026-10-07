"""Unit tests for novomodelo_bridge.dashboard.tabs.training (the Training tab)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from novomodelo_bridge.dashboard.tabs.training import (
    TAB_ID,
    TAB_LABEL,
    TAB_ORDER,
    _build_metrics_row,
    _chart_convergence_hero,
    _chart_cut_activity_heatmap,
    can_render,
    render,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conv(n: int = 10) -> pd.DataFrame:
    """Synthetic convergence DataFrame with *n* rows."""
    iterations = list(range(1, n + 1))
    lower_bound = [100.0 + i * 5.0 for i in range(n)]
    upper_bound_mean = [200.0 - i * 3.0 for i in range(n)]
    upper_bound_std = [10.0 - i * 0.5 for i in range(n)]
    gap_percent = [
        (ub - lb) / ub * 100 for lb, ub in zip(lower_bound, upper_bound_mean)
    ]
    cuts_active = [50 + i * 10 for i in range(n)]
    cuts_added = [5 + i % 3 for i in range(n)]
    return pd.DataFrame(
        {
            "iteration": iterations,
            "lower_bound": lower_bound,
            "upper_bound_mean": upper_bound_mean,
            "upper_bound_std": upper_bound_std,
            "gap_percent": gap_percent,
            "cuts_active": cuts_active,
            "cuts_added": cuts_added,
        }
    )


def _make_cut_selection(n_stages: int = 5, n_iters: int = 10) -> pd.DataFrame:
    """Synthetic cut_selection DataFrame."""
    rows = []
    for stage in range(1, n_stages + 1):
        for it in range(1, n_iters + 1):
            rows.append(
                {
                    "stage": stage,
                    "iteration": it,
                    "cuts_active_after": 50 + stage * 2,
                    "cuts_deactivated": stage % 3,
                }
            )
    return pd.DataFrame(rows)


def _make_timing(n: int = 10) -> pd.DataFrame:
    """Synthetic timing DataFrame matching novomodelo v0.5.0 schema."""
    iterations = list(range(1, n + 1))
    return pd.DataFrame(
        {
            "iteration": iterations,
            "forward_wall_ms": [100.0 + i for i in range(n)],
            "fwd_setup_ms": [5.0 + i for i in range(n)],
            "backward_wall_ms": [150.0 + i for i in range(n)],
            "lower_bound_ms": [10.0 + i for i in range(n)],
            "cut_selection_ms": [40.0 + i for i in range(n)],
            "overhead_ms": [10.0 + i for i in range(n)],
        }
    )


def _make_mock_data(
    conv: pd.DataFrame | None = None,
    cut_selection: pd.DataFrame | None = None,
    timing: pd.DataFrame | None = None,
    training_manifest: dict | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics DashboardData.

    The ``training_manifest`` parameter accepts a dict that may use either
    ``"elapsed_seconds"`` or ``"duration_seconds"`` as the timing key.  The
    implementation reads ``data.training_metadata.get("duration_seconds")``,
    so both key names are normalised here.
    """
    data = MagicMock()
    data.conv = conv if conv is not None else _make_conv()
    data.cut_selection = (
        cut_selection if cut_selection is not None else _make_cut_selection()
    )
    data.timing = timing if timing is not None else _make_timing()

    # Normalise the training manifest: the implementation reads
    # data.training_metadata.get("duration_seconds").  Map "elapsed_seconds"
    # -> "duration_seconds" if present so legacy test data still works.
    # When training_manifest is None, supply an empty dict so that
    # .get("duration_seconds") returns None -> "N/A" display.
    if training_manifest is None:
        data.training_metadata = {}
    else:
        normalised: dict = {}
        if "elapsed_seconds" in training_manifest:
            normalised["duration_seconds"] = training_manifest["elapsed_seconds"]
        normalised.update(
            {k: v for k, v in training_manifest.items() if k != "elapsed_seconds"}
        )
        data.training_metadata = normalised

    data.stage_labels = {i: f"Stage {i}" for i in range(1, 20)}
    return data


# ---------------------------------------------------------------------------
# test_tab_constants
# ---------------------------------------------------------------------------


def test_tab_constants() -> None:
    """TAB_ID, TAB_LABEL, and TAB_ORDER must match the spec."""
    assert TAB_ID == "tab-training"
    assert TAB_LABEL == "Training"
    assert TAB_ORDER == 20


# ---------------------------------------------------------------------------
# test_can_render_returns_true
# ---------------------------------------------------------------------------


def test_can_render_returns_true() -> None:
    """can_render() must always return True regardless of data."""
    data = _make_mock_data()
    assert can_render(data) is True


# ---------------------------------------------------------------------------
# test_build_metrics_row  (wireframe-aligned cards)
# ---------------------------------------------------------------------------


def test_build_metrics_row_contains_new_labels() -> None:
    """_build_metrics_row() must include the 5 wireframe labels."""
    data = _make_mock_data(
        conv=_make_conv(10),
        training_manifest={"elapsed_seconds": 15120},
    )
    html = _build_metrics_row(data)
    assert "Iterations" in html
    assert "Training Time" in html
    assert "Final Lower Bound" in html
    assert "Total Cuts" in html
    assert "Active Cuts" in html


def test_build_metrics_row_omits_removed_labels() -> None:
    """_build_metrics_row() must NOT contain Upper Bound, Gap, or Termination."""
    data = _make_mock_data(
        conv=_make_conv(10),
        training_manifest={"elapsed_seconds": 15120},
    )
    html = _build_metrics_row(data)
    assert "Upper Bound" not in html
    assert "Final Gap" not in html
    assert "Termination" not in html


def test_build_metrics_row_elapsed_seconds_formatted() -> None:
    """15120 seconds must render as '4h 12min'."""
    data = _make_mock_data(
        conv=_make_conv(5),
        training_manifest={"elapsed_seconds": 15120},
    )
    html = _build_metrics_row(data)
    assert "4h 12min" in html


def test_build_metrics_row_missing_elapsed_shows_na() -> None:
    """When training_manifest has no 'elapsed_seconds', Training Time shows 'N/A'."""
    data = _make_mock_data(training_manifest={})
    html = _build_metrics_row(data)
    assert "N/A" in html


def test_build_metrics_row_cut_totals() -> None:
    """cuts_added=[100,150,200] total=450; last cuts_active=350 must appear."""
    conv = pd.DataFrame(
        {
            "iteration": [1, 2, 3],
            "lower_bound": [100.0, 110.0, 120.0],
            "upper_bound_mean": [200.0, 190.0, 180.0],
            "upper_bound_std": [5.0, 5.0, 5.0],
            "gap_percent": [50.0, 42.1, 33.3],
            "cuts_active": [300, 320, 350],
            "cuts_added": [100, 150, 200],
        }
    )
    data = _make_mock_data(conv=conv, training_manifest={})
    html = _build_metrics_row(data)
    assert "450" in html
    assert "350" in html


# ---------------------------------------------------------------------------
# test_chart_convergence_hero_has_dropdown
# ---------------------------------------------------------------------------


def test_chart_convergence_hero_has_dropdown() -> None:
    """The hero convergence figure must have an updatemenus dropdown with 4 buttons."""
    conv = _make_conv(20)
    fig = _chart_convergence_hero(conv)
    assert fig.layout.updatemenus is not None
    assert len(fig.layout.updatemenus) > 0
    buttons = fig.layout.updatemenus[0].buttons
    assert len(buttons) == 4


def test_chart_convergence_hero_button_labels() -> None:
    """Dropdown buttons must be labeled: All, Last 200, Last 100, Last 50."""
    conv = _make_conv(20)
    fig = _chart_convergence_hero(conv)
    labels = [b.label for b in fig.layout.updatemenus[0].buttons]
    assert labels == ["All", "Last 200", "Last 100", "Last 50"]


def test_chart_convergence_hero_null_upper_bound_std_collapses_band() -> None:
    """novomodelo 0.14 leaves ``upper_bound_std`` NULL for an exact (enumerated)
    upper bound; the ±std band must collapse to the mean line rather than
    render NaN y-values that break the trace."""  # noqa: E501
    import math

    conv = _make_conv(6)
    conv["upper_bound_std"] = float("nan")  # exact-bound: no statistical std
    fig = _chart_convergence_hero(conv)
    band_traces = [t for t in fig.data if t.name in ("UB -std", "UB ±std Band")]
    assert band_traces, "expected the ±std band traces to be present"
    for trace in band_traces:
        ys = [y for y in trace.y if y is not None]
        assert ys and not any(math.isnan(y) for y in ys)
    # With a zero-width band each edge equals the mean upper bound.
    means = conv["upper_bound_mean"].tolist()
    for trace in band_traces:
        assert list(trace.y) == means


# ---------------------------------------------------------------------------
# test_render_with_empty_cut_selection
# ---------------------------------------------------------------------------


def test_render_with_empty_cut_selection() -> None:
    """render() must not raise and must contain 'No cut selection data' when
    cut_selection is empty."""  # noqa: E501
    data = _make_mock_data(
        conv=_make_conv(10),
        cut_selection=pd.DataFrame(),
        timing=_make_timing(10),
    )
    html = render(data)
    assert "No cut selection data" in html


# ---------------------------------------------------------------------------
# test_render_with_empty_timing
# ---------------------------------------------------------------------------


def test_render_with_empty_timing() -> None:
    """render() must not raise and must contain 'No timing data' when timing is
    empty."""  # noqa: E501
    data = _make_mock_data(
        conv=_make_conv(10),
        cut_selection=_make_cut_selection(),
        timing=pd.DataFrame(),
    )
    html = render(data)
    assert "No timing data" in html


# ---------------------------------------------------------------------------
# test_chart_cut_activity_heatmap_empty
# ---------------------------------------------------------------------------


def test_chart_cut_activity_heatmap_empty() -> None:
    """_chart_cut_activity_heatmap() with empty DataFrame returns placeholder HTML."""
    html = _chart_cut_activity_heatmap(pd.DataFrame(), {})
    assert "No cut selection data" in html


# ---------------------------------------------------------------------------
# Additional render content checks
# ---------------------------------------------------------------------------


def test_render_contains_required_sections() -> None:
    """render() HTML must contain the structural sections required by the spec.

    The hero chart is titled "Training Bounds" (the convergence hero); the cut
    pool chart is titled "Cut Pool Evolution".  Both contribute visible text in
    the rendered HTML.
    """
    data = _make_mock_data(conv=_make_conv(15))
    html = render(data)
    assert "Lower Bound" in html
    assert "Training Bounds" in html
    assert "Cut Pool" in html


def test_render_empty_conv_returns_placeholder() -> None:
    """render() with empty conv must return a safe placeholder string."""
    data = _make_mock_data(conv=pd.DataFrame())
    html = render(data)
    assert "No convergence data" in html


def test_render_full_data_no_exception() -> None:
    """render() with all data present must complete without raising."""
    data = _make_mock_data(
        conv=_make_conv(20),
        cut_selection=_make_cut_selection(6, 20),
        timing=_make_timing(20),
        training_manifest={"termination_reason": "convergence"},
    )
    html = render(data)
    assert len(html) > 0


def test_build_metrics_row_none_training_manifest_shows_na() -> None:
    """When training_manifest is None, Training Time shows 'N/A'."""
    data = _make_mock_data(training_manifest=None)
    html = _build_metrics_row(data)
    assert "N/A" in html


@pytest.mark.parametrize("n_iters", [50, 201, 400])
def test_chart_cut_activity_heatmap_sampling(n_iters: int) -> None:
    """Heatmap must succeed for both small and large iteration counts."""
    cs = _make_cut_selection(n_stages=4, n_iters=n_iters)
    html = _chart_cut_activity_heatmap(cs, {})
    # Should produce plotly HTML, not the placeholder
    assert "No cut selection data" not in html
    assert len(html) > 0


# ---------------------------------------------------------------------------
# Gap % dual-axis on convergence hero
# ---------------------------------------------------------------------------


def test_chart_convergence_hero_has_gap_trace_on_secondary_y() -> None:
    """Hero chart must include a 'Gap %' trace on secondary_y.

    Condition: gap_percent column is present and has real values.
    """
    conv = _make_conv(10)
    fig = _chart_convergence_hero(conv)
    gap_traces = [t for t in fig.data if t.name == "Gap %"]
    assert len(gap_traces) == 1, "Expected exactly one 'Gap %' trace"
    # make_subplots assigns secondary traces to yaxis2
    assert gap_traces[0].yaxis == "y2"


def test_chart_convergence_hero_gap_trace_style() -> None:
    """The 'Gap %' trace must be a dashed copper line."""
    conv = _make_conv(10)
    fig = _chart_convergence_hero(conv)
    gap_trace = next(t for t in fig.data if t.name == "Gap %")
    assert gap_trace.line.dash == "dash"
    assert "#B87333" in gap_trace.line.color


def test_chart_convergence_hero_yaxis2_title() -> None:
    """The secondary y-axis (yaxis2) title must be 'Gap (%)'."""
    conv = _make_conv(10)
    fig = _chart_convergence_hero(conv)
    assert fig.layout.yaxis2.title.text == "Gap (%)"


def test_chart_convergence_hero_no_gap_trace_when_column_missing() -> None:
    """Hero chart must NOT add a 'Gap %' trace when gap_percent column is absent."""
    conv = _make_conv(10).drop(columns=["gap_percent"])
    fig = _chart_convergence_hero(conv)
    gap_traces = [t for t in fig.data if t.name == "Gap %"]
    assert len(gap_traces) == 0


def test_chart_convergence_hero_no_gap_trace_when_all_nan() -> None:
    """Hero chart must NOT add a 'Gap %' trace when all gap_percent values are NaN."""
    import numpy as np

    conv = _make_conv(10)
    conv["gap_percent"] = np.nan
    fig = _chart_convergence_hero(conv)
    gap_traces = [t for t in fig.data if t.name == "Gap %"]
    assert len(gap_traces) == 0


def test_render_does_not_contain_standalone_gap_chart_title() -> None:
    """render() output must not contain the old standalone gap chart title."""
    data = _make_mock_data(conv=_make_conv(10))
    html = render(data)
    assert "Convergence Gap (%) per Iteration" not in html
