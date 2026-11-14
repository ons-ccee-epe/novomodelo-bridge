"""Unit tests for novomodelo_bridge.dashboard.tabs.constraints.

Constraint dicts here carry no ``sense`` key and ``gc_bounds`` fixtures
carry the ``bound_lower``/``bound_upper`` endpoint pair, matching what the
constraint-bounds writers emit. ``render()`` runs on MagicMock data with
real polars/pandas objects for the fields read as frames.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest

import novomodelo_bridge.dashboard.tabs.constraints as tab_constraints
from novomodelo_bridge.core.generic_constraint_format import sense_to_interval
from novomodelo_bridge.dashboard.tabs.constraints import (
    _build_constraint_lhs_data,
    _build_lhs_section,
    _build_metrics_row,
    _violation_zones,
    can_render,
    render,
)
from novomodelo_bridge.dashboard.tabs.constraints_utils import (
    bound_value_column,
    build_constraints_summary_table,
    derive_constraint_shape,
)
from novomodelo_bridge.novomodelo.constraint_expr import BLOCK_HOURS_SCHEMA

# ---------------------------------------------------------------------------
# Helpers / data factories
# ---------------------------------------------------------------------------


def _make_constraints(n: int = 2, ctype: str = "VminOP") -> list[dict]:
    """Return a list of minimal constraint dicts (sense-free shape)."""
    return [
        {
            "id": i,
            "name": f"{ctype}_constraint_{i}",
            "expression": f"hydro_storage({i})",
            "slack": {"enabled": False},
        }
        for i in range(n)
    ]


def _make_gc_bounds(
    constraint_ids: list[int],
    n_stages: int = 2,
    sense: str = ">=",
    value: float = 100.0,
) -> pd.DataFrame:
    """Return a minimal gc_bounds DataFrame in the interval endpoint shape.

    *sense*/*value* describe a ``(sense, bound)`` pair;
    :func:`sense_to_interval` maps it to the
    ``bound_lower``/``bound_upper`` endpoints actually stored on disk.
    """
    lower, upper = sense_to_interval(sense, value)
    rows = []
    for cid in constraint_ids:
        for stage_id in range(n_stages):
            rows.append(
                {
                    "constraint_id": cid,
                    "stage_id": stage_id,
                    "block_id": float("nan"),
                    "bound_lower": lower,
                    "bound_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def _make_gc_bounds_range(
    constraint_ids: list[int],
    n_stages: int,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    """Return a gc_bounds DataFrame with a genuine two-sided ("range") endpoint
    pair — both ``bound_lower`` and ``bound_upper`` populated and distinct, the
    shape DECOMP's RE/HQ/HV families emit for an ``L <= expr <= U`` band
    (``sense_to_interval`` has no single-sense equivalent for this, unlike
    :func:`_make_gc_bounds`)."""
    rows = []
    for cid in constraint_ids:
        for stage_id in range(n_stages):
            rows.append(
                {
                    "constraint_id": cid,
                    "stage_id": stage_id,
                    "block_id": float("nan"),
                    "bound_lower": lower,
                    "bound_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def _make_gc_violations(
    rows: list[tuple[int, int, int, int, float]] | None = None,
) -> pd.DataFrame:
    """A ``violations/generic`` frame from ``(constraint_id, scenario_id,
    stage_id, block_id, slack_value)`` rows; empty when *rows* is None."""
    return pd.DataFrame(
        rows or [],
        columns=["constraint_id", "scenario_id", "stage_id", "block_id", "slack_value"],
    )


def _make_costs(
    n_scenarios: int = 1,
    n_stages: int = 2,
    include_violation_cost: bool = False,
    violation_cost: float = 0.0,
) -> pd.DataFrame:
    """Return a minimal costs DataFrame."""
    rows = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            row: dict = {
                "scenario_id": scenario_id,
                "stage_id": stage_id,
                "thermal_cost": 500.0,
            }
            if include_violation_cost:
                row["generic_violation_cost"] = violation_cost
            rows.append(row)
    return pd.DataFrame(rows)


def _make_lhs_df(
    constraint_ids: list[int],
    n_scenarios: int = 2,
    n_stages: int = 3,
    lhs_value: float = 120.0,
) -> pd.DataFrame:
    """Return a small LHS DataFrame with uniform lhs_value across all rows."""
    rows = []
    for cid in constraint_ids:
        for sid in range(n_scenarios):
            for stg in range(n_stages):
                rows.append(
                    {
                        "constraint_id": cid,
                        "scenario_id": sid,
                        "stage_id": stg,
                        "block_id": 0,
                        "lhs_value": lhs_value,
                    }
                )
    return pd.DataFrame(rows)


def _make_mock_data(
    *,
    constraints: list[dict] | None = None,
    gc_bounds: pd.DataFrame | None = None,
    gc_violations: pd.DataFrame | None = None,
    costs: pd.DataFrame | None = None,
    n_stages: int = 2,
) -> MagicMock:
    """Build a minimal MagicMock satisfying the DashboardData interface.

    Sets real polars LazyFrames for ``lhs_frames`` to prevent MagicMock
    auto-chaining from causing OOM.
    """
    data = MagicMock()
    data.gc_constraints = (
        constraints if constraints is not None else _make_constraints()
    )
    data.gc_bounds = (
        gc_bounds
        if gc_bounds is not None
        else _make_gc_bounds([c["id"] for c in data.gc_constraints], n_stages)
    )
    data.gc_violations = (
        gc_violations if gc_violations is not None else _make_gc_violations()
    )
    data.costs = costs if costs is not None else _make_costs(n_stages=n_stages)
    data.stage_labels = {i: f"Stage {i}" for i in range(n_stages)}
    data.stage_dates = {i: f"2026-{i + 1:02d}-01" for i in range(n_stages)}
    data.names = {}
    data.stages_data = {}
    data.block_hours = {(i, 0): 730.0 for i in range(n_stages)}
    # Real empty LazyFrames prevent MagicMock auto-chaining OOM
    data.lhs_frames = {"hydros": pl.LazyFrame()}
    data.simulation_available = True
    return data


# ---------------------------------------------------------------------------
# test_tab_constants
# ---------------------------------------------------------------------------


def test_tab_constants() -> None:
    """Module-level constants must match their expected values exactly."""
    assert tab_constraints.TAB_ID == "tab-constraints"
    assert tab_constraints.TAB_LABEL == "Constraints"
    assert tab_constraints.TAB_ORDER == 80


# ---------------------------------------------------------------------------
# test_can_render
# ---------------------------------------------------------------------------


def test_can_render_false_when_no_constraints() -> None:
    """can_render must return False when gc_constraints is empty."""
    data = _make_mock_data(constraints=[])
    assert can_render(data) is False


def test_can_render_true_when_constraints_present() -> None:
    """can_render must return True when gc_constraints contains 1+ dicts."""
    data = _make_mock_data(constraints=_make_constraints(n=1))
    assert can_render(data) is True


def test_can_render_true_with_multiple_constraints() -> None:
    """can_render must return True when gc_constraints contains multiple dicts."""
    data = _make_mock_data(constraints=_make_constraints(n=3))
    assert can_render(data) is True


def test_can_render_false_without_simulation() -> None:
    """can_render must return False when simulation data is unavailable."""
    data = _make_mock_data(constraints=_make_constraints(n=1))
    data.simulation_available = False
    assert can_render(data) is False


# ---------------------------------------------------------------------------
# test__build_metrics_row
# ---------------------------------------------------------------------------


def test_build_metrics_row_produces_metrics_grid() -> None:
    """_build_metrics_row must produce HTML with 'metrics-grid' class."""
    data = _make_mock_data()
    html = _build_metrics_row(data)
    assert "metrics-grid" in html


def test_build_metrics_row_produces_four_metric_cards() -> None:
    """_build_metrics_row must produce HTML with 'metric-card' at least 4 times."""
    data = _make_mock_data()
    html = _build_metrics_row(data)
    assert html.count("metric-card") >= 4


def test_build_metrics_row_shows_constraint_count() -> None:
    """_build_metrics_row must display the correct total constraint count."""
    constraints = _make_constraints(n=3)
    data = _make_mock_data(constraints=constraints)
    html = _build_metrics_row(data)
    assert ">3<" in html


def test_build_metrics_row_with_violations() -> None:
    """_build_metrics_row must show violated count > 0 when violations present.

    When one constraint has a nonzero slack_value, the violated count must be 1.
    """
    constraints = _make_constraints(n=2)
    gc_violations = _make_gc_violations([(constraints[0]["id"], 0, 0, 0, 5.0)])
    data = _make_mock_data(constraints=constraints, gc_violations=gc_violations)
    html = _build_metrics_row(data)
    # 1 violated constraint should appear as ">1<"
    assert ">1<" in html
    assert "Constraints with Violations" in html


def test_build_metrics_row_no_violation_cost_column() -> None:
    """_build_metrics_row must show R$ 0 when violation cost column is absent."""
    costs = _make_costs(include_violation_cost=False)
    data = _make_mock_data(costs=costs)
    html = _build_metrics_row(data)
    assert "R$ 0" in html
    assert "Generic Constraints (NPV)" in html


def test_build_metrics_row_violation_cost_is_mean_over_scenarios() -> None:
    """The card is the expected cost per scenario, as on the Costs tab, not
    the sum over every scenario."""
    # 2 stages x 100.0 = 200.0 per scenario, in each of the 2 scenarios.
    costs = _make_costs(
        n_scenarios=2,
        n_stages=2,
        include_violation_cost=True,
        violation_cost=100.0,
    )
    data = _make_mock_data(costs=costs)
    html = _build_metrics_row(data)
    assert "R$ 200" in html
    assert "R$ 400" not in html


def test_build_metrics_row_violation_cost_is_discounted() -> None:
    """The card is a present value: stage 1 is weighted by its discount factor."""
    costs = _make_costs(
        n_scenarios=2,
        n_stages=2,
        include_violation_cost=True,
        violation_cost=100.0,
    )
    costs["discount_factor"] = costs["stage_id"].map({0: 1.0, 1: 0.5})
    data = _make_mock_data(costs=costs)
    html = _build_metrics_row(data)
    assert "R$ 150" in html


def test_build_metrics_row_shows_active_types() -> None:
    """_build_metrics_row must include unique type prefixes from constraint names."""
    constraints = [
        {
            "id": 0,
            "name": "VminOP_c1",
            "expression": "hydro_storage(0)",
            "slack": {},
        },
        {
            "id": 1,
            "name": "RE_c2",
            "expression": "hydro_storage(1)",
            "slack": {},
        },
    ]
    data = _make_mock_data(constraints=constraints)
    html = _build_metrics_row(data)
    assert "VminOP" in html
    assert "RE" in html
    assert "Active Types" in html


def test_build_metrics_row_empty_violations_shows_zero_violated() -> None:
    """_build_metrics_row must show 0 violated constraints when violations is empty."""
    data = _make_mock_data(gc_violations=_make_gc_violations())
    html = _build_metrics_row(data)
    assert ">0<" in html


# ---------------------------------------------------------------------------
# test__violation_zones
# ---------------------------------------------------------------------------


def test_violation_zones_merge_contiguous_violated_stages() -> None:
    assert _violation_zones([0, 2, 1, 0, 0, 3]) == [
        {"start": 1, "end": 2},
        {"start": 5, "end": 5},
    ]


def test_violation_zones_empty_without_violations() -> None:
    assert _violation_zones([0, 0, 0]) == []
    assert _violation_zones([]) == []


# ---------------------------------------------------------------------------
# test__build_constraint_lhs_data
# ---------------------------------------------------------------------------


def _hours(n_stages: int, blocks: dict[int, float] | None = None) -> pl.DataFrame:
    """Every stage with *blocks* (block id to hours; one 730 h block by default)."""
    blocks = blocks or {0: 730.0}
    return pl.DataFrame(
        {
            "stage_id": [s for s in range(n_stages) for _ in blocks],
            "block_id": list(blocks) * n_stages,
            "hours": list(blocks.values()) * n_stages,
        },
        schema=BLOCK_HOURS_SCHEMA,
    )


def _lhs_data(
    constraints: list[dict],
    lhs_df: pd.DataFrame,
    gc_bounds: pd.DataFrame,
    *,
    n_stages: int,
    gc_violations: pd.DataFrame | None = None,
    block_hours: pl.DataFrame | None = None,
    block_names: dict[int, str] | None = None,
) -> dict:
    return _build_constraint_lhs_data(
        constraints,
        lhs_df,
        gc_bounds,
        gc_violations if gc_violations is not None else _make_gc_violations(),
        block_hours if block_hours is not None else _hours(n_stages),
        block_names or {},
        {i: f"S{i}" for i in range(n_stages)},
    )


def _make_block_gc_bounds(
    cid: int, n_stages: int, upper: dict[int, float]
) -> pd.DataFrame:
    """Per-block ceiling rows: block id to its upper bound, at every stage."""
    return pd.DataFrame(
        [
            {
                "constraint_id": cid,
                "stage_id": s,
                "block_id": b,
                "bound_lower": None,
                "bound_upper": u,
            }
            for s in range(n_stages)
            for b, u in upper.items()
        ]
    )


_LHS_KEYS = ("lhs_p10", "lhs_p50", "lhs_p90", "bound_lower", "bound_upper")
_VIEW_KEYS = {
    "lhs_p10",
    "lhs_p50",
    "lhs_p90",
    "bound_lower",
    "bound_upper",
    "violated",
    "violations",
}


def test_build_constraint_lhs_data_structure() -> None:
    constraints = [{"id": 0, "name": "VminOP_c0"}, {"id": 1, "name": "RE_c1"}]
    lhs_df = _make_lhs_df([0, 1], n_scenarios=2, n_stages=3, lhs_value=120.0)
    gc_bounds = pd.concat(
        [
            _make_gc_bounds([0], n_stages=3, sense=">="),
            _make_gc_bounds([1], n_stages=3, sense="<="),
        ],
        ignore_index=True,
    )

    result = _lhs_data(constraints, lhs_df, gc_bounds, n_stages=3)

    assert result["stages"] == [0, 1, 2]
    assert result["xlabels"] == ["S0", "S1", "S2"]
    assert result["n_scenarios"] == 2
    for cid_str in ("0", "1"):
        entry = result["constraints"][cid_str]
        assert set(entry) == {"name", "sense", "evaluated", "stage", "blocks"}
        assert _VIEW_KEYS <= set(entry["stage"])
        assert all(len(entry["stage"][k]) == 3 for k in _LHS_KEYS)
        # Stage-level rows have no per-block view.
        assert entry["blocks"] == []


def test_build_constraint_lhs_data_uniform_lhs_gives_equal_percentiles() -> None:
    lhs_df = _make_lhs_df([0], n_scenarios=3, n_stages=3, lhs_value=50.0)

    result = _lhs_data(
        [{"id": 0, "name": "VminOP_c0"}],
        lhs_df,
        _make_gc_bounds([0], n_stages=3),
        n_stages=3,
    )
    view = result["constraints"]["0"]["stage"]

    for key in ("lhs_p10", "lhs_p50", "lhs_p90"):
        assert view[key] == pytest.approx([50.0, 50.0, 50.0])


def test_build_constraint_lhs_data_stage_view_weights_blocks_by_hours() -> None:
    """The stage view is the hours-weighted mean of the blocks, LHS and bound."""
    lhs_df = pd.DataFrame(
        {
            "constraint_id": [0, 0],
            "scenario_id": [0, 0],
            "stage_id": [0, 0],
            "block_id": [0, 1],
            "lhs_value": [10.0, 30.0],
        }
    )
    gc_bounds = _make_block_gc_bounds(0, n_stages=1, upper={0: 40.0, 1: 20.0})

    entry = _lhs_data(
        [{"id": 0, "name": "RE_c0"}],
        lhs_df,
        gc_bounds,
        n_stages=1,
        block_hours=_hours(1, {0: 100.0, 1: 300.0}),
    )["constraints"]["0"]

    assert entry["stage"]["lhs_p50"] == pytest.approx([25.0])
    assert entry["stage"]["bound_upper"] == pytest.approx([25.0])
    assert entry["stage"]["bound_lower"] == [None]


def test_build_constraint_lhs_data_block_views_hold_each_blocks_own_bound() -> None:
    lhs_df = pd.DataFrame(
        {
            "constraint_id": [0, 0],
            "scenario_id": [0, 0],
            "stage_id": [0, 0],
            "block_id": [0, 1],
            "lhs_value": [10.0, 30.0],
        }
    )
    gc_bounds = _make_block_gc_bounds(0, n_stages=1, upper={0: 40.0, 1: 20.0})

    blocks = _lhs_data(
        [{"id": 0, "name": "RE_c0"}],
        lhs_df,
        gc_bounds,
        n_stages=1,
        block_hours=_hours(1, {0: 100.0, 1: 300.0}),
        block_names={0: "Heavy"},
    )["constraints"]["0"]["blocks"]

    assert [b["label"] for b in blocks] == ["Block 0 · Heavy", "Block 1"]
    assert [b["lhs_p50"] for b in blocks] == [[10.0], [30.0]]
    assert [b["bound_upper"] for b in blocks] == [[40.0], [20.0]]


def test_build_constraint_lhs_data_violations_come_from_the_slacks() -> None:
    """A stage is flagged where some scenario has a nonzero slack, whatever
    the percentile band shows, and the count is of distinct scenarios."""
    lhs_df = _make_lhs_df([0], n_scenarios=3, n_stages=3, lhs_value=120.0)
    gc_violations = _make_gc_violations(
        [(0, 0, 1, 0, 5.0), (0, 2, 1, 0, 1.0), (0, 1, 2, 0, 1e-9)]
    )

    view = _lhs_data(
        [{"id": 0, "name": "VminOP_c0"}],
        lhs_df,
        _make_gc_bounds([0], n_stages=3, value=100.0),
        n_stages=3,
        gc_violations=gc_violations,
    )["constraints"]["0"]["stage"]

    assert view["violated"] == [0, 2, 0]
    assert view["violations"] == [{"start": 1, "end": 1}]


def test_build_constraint_lhs_data_block_violations_stay_on_their_block() -> None:
    lhs_df = pd.DataFrame(
        {
            "constraint_id": [0, 0],
            "scenario_id": [0, 0],
            "stage_id": [0, 0],
            "block_id": [0, 1],
            "lhs_value": [50.0, 30.0],
        }
    )

    entry = _lhs_data(
        [{"id": 0, "name": "RE_c0"}],
        lhs_df,
        _make_block_gc_bounds(0, n_stages=1, upper={0: 40.0, 1: 40.0}),
        n_stages=1,
        gc_violations=_make_gc_violations([(0, 0, 0, 0, 10.0)]),
        block_hours=_hours(1, {0: 100.0, 1: 300.0}),
    )["constraints"]["0"]

    assert entry["stage"]["violated"] == [1]
    assert [b["violated"] for b in entry["blocks"]] == [[1], [0]]


def test_build_constraint_lhs_data_unevaluated_lhs_reads_none_not_zero() -> None:
    lhs_df = pd.DataFrame(
        columns=["constraint_id", "scenario_id", "stage_id", "block_id", "lhs_value"]
    )

    entry = _lhs_data(
        [{"id": 0, "name": "VminOP_c0"}],
        lhs_df,
        _make_gc_bounds([0], n_stages=3),
        n_stages=3,
    )["constraints"]["0"]

    assert entry["evaluated"] is False
    assert entry["stage"]["lhs_p50"] == [None, None, None]
    assert entry["stage"]["bound_lower"] == [100.0, 100.0, 100.0]


def test_build_constraint_lhs_data_missing_bounds_gives_none_bound() -> None:
    lhs_df = _make_lhs_df([99], n_scenarios=2, n_stages=3)
    gc_bounds = pd.DataFrame(
        columns=["constraint_id", "stage_id", "block_id", "bound_lower", "bound_upper"]
    )

    entry = _lhs_data(
        [{"id": 99, "name": "VminOP_c99"}], lhs_df, gc_bounds, n_stages=3
    )["constraints"]["99"]

    assert entry["stage"]["bound_lower"] == [None, None, None]
    assert entry["stage"]["bound_upper"] == [None, None, None]
    assert entry["stage"]["violations"] == []
    # Falls back to the historical `c.get("sense", "<=")` default shape.
    assert entry["sense"] == "<="


# ---------------------------------------------------------------------------
# Derived sense + bound value parity with a legacy (sense, bound) case
# ---------------------------------------------------------------------------


def test_build_constraint_lhs_data_derives_ge_from_lower_endpoint() -> None:
    """A VminOP-style floor (lower-only) renders like a legacy ``sense=">="``."""
    lhs_df = _make_lhs_df([0], n_scenarios=2, n_stages=2, lhs_value=120.0)
    gc_bounds = _make_gc_bounds([0], n_stages=2, sense=">=", value=500.0)

    entry = _lhs_data([{"id": 0, "name": "VminOP_c0"}], lhs_df, gc_bounds, n_stages=2)[
        "constraints"
    ]["0"]

    assert entry["sense"] == ">="
    assert entry["stage"]["bound_lower"] == [500.0, 500.0]
    assert entry["stage"]["bound_upper"] == [None, None]


def test_build_constraint_lhs_data_derives_le_from_upper_endpoint() -> None:
    """An RE/AGRINT-style ceiling (upper-only) renders like a legacy ``"<="``."""
    lhs_df = _make_lhs_df([1], n_scenarios=2, n_stages=2, lhs_value=120.0)
    gc_bounds = _make_gc_bounds([1], n_stages=2, sense="<=", value=200.0)

    entry = _lhs_data([{"id": 1, "name": "RE_c1"}], lhs_df, gc_bounds, n_stages=2)[
        "constraints"
    ]["1"]

    assert entry["sense"] == "<="
    assert entry["stage"]["bound_upper"] == [200.0, 200.0]


def test_build_constraint_lhs_data_range_exposes_floor_and_ceiling() -> None:
    """A two-sided ``L <= expr <= U`` band (DECOMP RE/HQ/HV) exposes both
    endpoints, not just the ceiling."""
    lhs_df = _make_lhs_df([0], n_scenarios=2, n_stages=3, lhs_value=50.0)
    gc_bounds = _make_gc_bounds_range([0], n_stages=3, lower=10.0, upper=90.0)

    entry = _lhs_data([{"id": 0, "name": "HQ_c0"}], lhs_df, gc_bounds, n_stages=3)[
        "constraints"
    ]["0"]

    assert entry["sense"] == "range"
    assert entry["stage"]["bound_lower"] == [10.0, 10.0, 10.0]
    assert entry["stage"]["bound_upper"] == [90.0, 90.0, 90.0]


class TestDeriveConstraintShapeAndBoundValueColumn:
    """Unit coverage for the shared helpers constraints.py/constraints_utils.py
    now use in place of the removed ``c["sense"]``/single ``bound`` column."""

    def test_derives_ge_from_lower_only(self) -> None:
        rows = pd.DataFrame(
            {
                "constraint_id": [0],
                "bound_lower": [500.0],
                "bound_upper": [float("nan")],
            }
        )
        assert derive_constraint_shape(rows) == ">="
        assert bound_value_column(">=") == "bound_lower"

    def test_derives_le_from_upper_only(self) -> None:
        rows = pd.DataFrame(
            {
                "constraint_id": [0],
                "bound_lower": [float("nan")],
                "bound_upper": [200.0],
            }
        )
        assert derive_constraint_shape(rows) == "<="
        assert bound_value_column("<=") == "bound_upper"

    def test_derives_eq_from_both_equal(self) -> None:
        rows = pd.DataFrame(
            {
                "constraint_id": [0],
                "bound_lower": [300.0],
                "bound_upper": [300.0],
            }
        )
        assert derive_constraint_shape(rows) == "=="
        assert bound_value_column("==") == "bound_lower"

    def test_derives_range_from_distinct_endpoints(self) -> None:
        """A genuine two-sided band (distinct ``bound_lower``/``bound_upper``)
        is a live DECOMP RE/HQ/HV path, not a theoretical shape."""
        rows = pd.DataFrame(
            {
                "constraint_id": [0],
                "bound_lower": [10.0],
                "bound_upper": [90.0],
            }
        )
        assert derive_constraint_shape(rows) == "range"
        assert bound_value_column("range") == "bound_upper"

    def test_defaults_to_le_when_empty(self) -> None:
        rows = pd.DataFrame(columns=["constraint_id", "bound_lower", "bound_upper"])
        assert derive_constraint_shape(rows) == "<="


# ---------------------------------------------------------------------------
# test__build_lhs_section
# ---------------------------------------------------------------------------


def _section(data: MagicMock, lhs_df: pd.DataFrame, n_stages: int) -> str:
    return _build_lhs_section(data, lhs_df, _hours(n_stages))


def test_build_lhs_section_html_contains_required_elements() -> None:
    constraints = _make_constraints(n=2)
    lhs_df = _make_lhs_df([0, 1], n_scenarios=2, n_stages=3)
    data = _make_mock_data(constraints=constraints, n_stages=3)

    html = _section(data, lhs_df, 3)

    assert "gc-constraint-sel" in html
    assert "gc-lhs-chart" in html
    assert "GC_LHS_DATA" in html


def _select(html: str, select_id: str) -> str:
    """The ``<select id="select_id">`` element of *html*."""
    match = re.search(rf'<select id="{select_id}".*?</select>', html, re.DOTALL)
    assert match is not None
    return match.group(0)


def test_build_lhs_section_offers_stage_and_block_views() -> None:
    data = _make_mock_data(constraints=_make_constraints(n=1), n_stages=2)

    html = _section(data, _make_lhs_df([0], n_stages=2), 2)

    select = _select(html, "gc-view-sel")
    assert re.findall(r'<option value="(\w+)"', select) == ["stage", "block"]
    assert "Stage (hours-weighted)" in select


def test_build_lhs_section_has_one_constraint_option_per_constraint() -> None:
    constraints = _make_constraints(n=3)
    data = _make_mock_data(constraints=constraints, n_stages=2)

    html = _section(data, _make_lhs_df([0, 1, 2], n_stages=2), 2)

    select = _select(html, "gc-constraint-sel")
    assert select.count("<option") == 3


def test_build_lhs_section_escapes_constraint_names() -> None:
    constraints = [{"id": 0, "name": "RE_<b>"}]
    data = _make_mock_data(constraints=constraints, n_stages=2)

    html = _section(data, _make_lhs_df([0], n_stages=2), 2)

    assert "RE_&lt;b&gt;" in html
    assert '<option value="0">RE_<b>' not in html


def test_build_lhs_section_range_data_and_lower_bound_trace_present() -> None:
    """A two-sided range's endpoints reach the embedded blob, and the script
    draws a second, lower-bound line for a distinct floor."""
    constraints = [{"id": 0, "name": "HQ_c0"}]
    gc_bounds = _make_gc_bounds_range([0], n_stages=2, lower=10.0, upper=90.0)
    data = _make_mock_data(constraints=constraints, gc_bounds=gc_bounds, n_stages=2)

    html = _section(data, _make_lhs_df([0], n_stages=2, lhs_value=50.0), 2)

    assert '"bound_lower":[10.0,10.0]' in html
    assert '"bound_upper":[90.0,90.0]' in html
    assert "distinctLower" in html
    assert "Lower Bound" in html


# ---------------------------------------------------------------------------
# test_type_filter_dropdown
# ---------------------------------------------------------------------------


def _type_filter_values(html: str) -> list[str]:
    return re.findall(r'<option value="([^"]*)"', _select(html, "gc-type-filter"))


def test_type_filter_offers_the_families_present_in_first_seen_order() -> None:
    constraints = [
        {"id": 0, "name": "RHE_1", "slack": {"enabled": False}},
        {"id": 1, "name": "RHV_2", "slack": {"enabled": False}},
        {"id": 2, "name": "RHE_3", "slack": {"enabled": False}},
    ]
    data = _make_mock_data(constraints=constraints)

    with _render_with_stubs_ctx(data) as html:
        assert _type_filter_values(html) == ["All", "RHE", "RHV"]


def test_type_filter_has_no_family_absent_from_the_case() -> None:
    data = _make_mock_data(constraints=_make_constraints(n=2, ctype="VminOP"))

    with _render_with_stubs_ctx(data) as html:
        assert _type_filter_values(html) == ["All", "VminOP"]


# ---------------------------------------------------------------------------
# test_no_three_separate_sections
# ---------------------------------------------------------------------------


def test_no_three_separate_old_section_titles() -> None:
    """render() must NOT contain the old three-section titles from Section C."""
    data = _make_mock_data()

    with _render_with_stubs_ctx(data) as html:
        assert "VminOP: Stored Energy vs Minimum" not in html
        assert "Electric Constraints (RE)" not in html
        assert "Exchange Group Constraints (AGRINT)" not in html


# ---------------------------------------------------------------------------
# test_render (existing + updated)
# ---------------------------------------------------------------------------

_PATCH_EVAL = "novomodelo_bridge.dashboard.tabs.constraints.evaluate_lhs"

_STUB_LHS_DF = pd.DataFrame(
    columns=["constraint_id", "scenario_id", "stage_id", "block_id", "lhs_value"]
)


class _render_with_stubs_ctx:
    """Context manager that calls render() with LazyFrame-dependent functions
    patched."""

    def __init__(self, data: MagicMock) -> None:
        self._data = data
        self._html: str = ""

    def __enter__(self) -> str:
        from unittest.mock import patch

        with patch(_PATCH_EVAL, return_value=_STUB_LHS_DF):
            self._html = render(self._data)
        return self._html

    def __exit__(self, *_: object) -> None:
        pass


def _render_with_stubs(data: MagicMock) -> str:
    """Call render() with LazyFrame-dependent functions patched to stubs."""
    from unittest.mock import patch

    with patch(_PATCH_EVAL, return_value=_STUB_LHS_DF):
        return render(data)


def test_render_produces_expected_sections() -> None:
    """render() with minimal data must contain expected HTML substrings."""
    constraints = _make_constraints(n=2, ctype="VminOP")
    gc_bounds = _make_gc_bounds([c["id"] for c in constraints])
    gc_violations = _make_gc_violations()
    costs = _make_costs(include_violation_cost=False)
    data = _make_mock_data(
        constraints=constraints,
        gc_bounds=gc_bounds,
        gc_violations=gc_violations,
        costs=costs,
    )
    html = _render_with_stubs(data)
    assert "Constraint Summary" in html
    assert "VminOP" in html
    assert "Violation" in html
    assert "metrics-grid" in html


def test_render_contains_section_titles() -> None:
    """render() must contain the current section titles (A, B, C only)."""
    data = _make_mock_data()
    html = _render_with_stubs(data)
    assert "Constraint Summary" in html
    assert "LHS vs Bound" in html


def test_render_contains_chart_grid() -> None:
    """render() must contain chart-grid elements."""
    data = _make_mock_data()
    html = _render_with_stubs(data)
    assert "chart-grid" in html


def test_render_contains_lhs_vs_bound_section() -> None:
    """render() must contain the new unified 'LHS vs Bound' section title."""
    data = _make_mock_data()
    html = _render_with_stubs(data)
    assert "LHS vs Bound" in html


def test_render_contains_gc_lhs_chart_div() -> None:
    """render() must contain the gc-lhs-chart div for the JS chart."""
    data = _make_mock_data()
    html = _render_with_stubs(data)
    assert "gc-lhs-chart" in html


def test_render_returns_string() -> None:
    """render() must return a str."""
    data = _make_mock_data()
    result = _render_with_stubs(data)
    assert isinstance(result, str)


def test_render_evaluates_lhs_from_the_resolved_view() -> None:
    """render() evaluates the LHS from novomodelo's resolved coefficients."""
    from unittest.mock import patch

    data = _make_mock_data()
    with patch(_PATCH_EVAL, return_value=_STUB_LHS_DF) as mock_eval:
        render(data)

    resolved, frames, block_hours = mock_eval.call_args.args
    assert resolved is data.gc_resolved
    assert frames is data.lhs_frames
    assert block_hours.to_dicts() == [
        {"stage_id": 0, "block_id": 0, "hours": 730.0},
        {"stage_id": 1, "block_id": 0, "hours": 730.0},
    ]


# ---------------------------------------------------------------------------
# build_constraints_summary_table derives its "Sense"
# column + "Bound Range" from the interval endpoints
# ---------------------------------------------------------------------------


class TestSummaryTableDerivesSenseFromBounds:
    """``build_constraints_summary_table`` shows the same Sense + Bound Range
    a legacy (sense, bound) case shows, sourced from ``bound_lower``/``bound_upper``."""

    @staticmethod
    def _violations() -> pd.DataFrame:
        return pd.DataFrame(columns=["constraint_id", "slack_value"])

    def test_ge_constraint_shows_ge_and_lower_bound_range(self) -> None:
        constraints = [{"id": 0, "name": "VminOP_c0", "slack": {"enabled": False}}]
        gc_bounds = _make_gc_bounds([0], n_stages=2, sense=">=", value=500.0)

        html = build_constraints_summary_table(
            constraints, gc_bounds, self._violations()
        )

        assert "<code>&gt;=</code>" in html
        assert "500.0" in html

    def test_le_constraint_shows_le_and_upper_bound_range(self) -> None:
        constraints = [{"id": 1, "name": "RE_c1", "slack": {"enabled": False}}]
        gc_bounds = _make_gc_bounds([1], n_stages=2, sense="<=", value=200.0)

        html = build_constraints_summary_table(
            constraints, gc_bounds, self._violations()
        )

        assert "<code>&lt;=</code>" in html
        assert "200.0" in html

    def test_no_bounds_falls_back_to_le_default(self) -> None:
        constraints = [{"id": 2, "name": "AGRINT_c2", "slack": {"enabled": False}}]
        gc_bounds = pd.DataFrame(
            columns=[
                "constraint_id",
                "stage_id",
                "block_id",
                "bound_lower",
                "bound_upper",
            ]
        )

        html = build_constraints_summary_table(
            constraints, gc_bounds, self._violations()
        )

        assert "<code>&lt;=</code>" in html

    def test_penalty_shows_how_it_is_charged(self) -> None:
        constraints = [
            {
                "id": 0,
                "name": "VminOP_c0",
                "slack": {
                    "enabled": True,
                    "penalty": 1488.0,
                    "penalty_basis": "per_unit",
                },
            },
            {
                "id": 1,
                "name": "RE_c1",
                "slack": {
                    "enabled": True,
                    "penalty": 500.0,
                    "penalty_basis": "per_hour",
                },
            },
        ]
        gc_bounds = _make_gc_bounds([0, 1], n_stages=1, sense=">=", value=1.0)

        html = build_constraints_summary_table(
            constraints, gc_bounds, self._violations()
        )

        assert "1,488 per unit" in html
        assert "500 per hour" in html


# ---------------------------------------------------------------------------
# dashboard/data.py's generic-constraint bounds loader reads the
# bound_lower/bound_upper columns straight through
# ---------------------------------------------------------------------------


class TestLoadGenericConstraintsF3Shape:
    """``load_generic_constraints`` takes the bounds from the run's echo — the
    ``bound_lower``/``bound_upper`` pair — with no ``bound`` column."""

    def test_loads_f3_bounds_from_the_echo(self, tmp_path: Path) -> None:
        from novomodelo_bridge.dashboard.data import load_generic_constraints
        from tests.conftest import write_generic_constraint_echo

        constraints_dir = tmp_path / "constraints"
        constraints_dir.mkdir()
        (constraints_dir / "generic_constraints.json").write_text(
            '{"constraints": [{"id": 0, "name": "HV_1", '
            '"expression": "hydro_storage(0)", "slack": {"enabled": false}}]}',
            encoding="utf-8",
        )
        write_generic_constraint_echo(
            tmp_path / "output",
            [(0, "HV_1", 0, 0, "hydro_storage(id=0)", 1.0, 500.0, None)],
        )

        result = load_generic_constraints(tmp_path)

        assert "sense" not in result.constraints[0]
        assert "bound" not in result.bounds.columns
        assert result.bounds["bound_lower"].tolist() == [500.0]
        assert math.isnan(result.bounds["bound_upper"].iloc[0])

    def test_missing_files_give_empty_defaults(self, tmp_path: Path) -> None:
        from novomodelo_bridge.dashboard.data import load_generic_constraints

        result = load_generic_constraints(tmp_path)

        assert result.constraints == []
        assert result.bounds.empty


# ---------------------------------------------------------------------------
# grep guard: no dashboard/report reader accesses a removed `sense` key
# or a single `bound` column
# ---------------------------------------------------------------------------

_BOUND_ACCESS_RE = re.compile(
    r'\.col\(\s*["\']bound["\']\s*\)|\[\s*["\']bound["\']\s*\]|\.get\(\s*["\']bound["\']'
)
_SENSE_ACCESS_RE = re.compile(
    r'\.col\(\s*["\']sense["\']\s*\)|\[\s*["\']sense["\']\s*\]|\.get\(\s*["\']sense["\']'
)


class TestNoSenseOrSingleBoundColumnRemainsInDashboardOrReport:
    """Mirrors ``TestNoSenseOrSingleBoundColumnRemainsInComparators``
    in the comparators compare tests, for the dashboard/report readers:
    matches only genuine column/dict *access* patterns (``.col("bound")``,
    ``row["bound"]``, ``.get("bound"``) — column *names* that merely contain
    "bound" as a substring (``bound_lower``, ``bound_upper``) are not
    matches.
    """

    @pytest.mark.parametrize(
        "relative_path",
        [
            "src/novomodelo_bridge/dashboard/tabs/constraints.py",
            "src/novomodelo_bridge/dashboard/tabs/constraints_utils.py",
            "src/novomodelo_bridge/dashboard/data.py",
            "src/novomodelo_bridge/comparators/report_builder.py",
        ],
    )
    def test_module_has_no_sense_or_bound_column_access(
        self, relative_path: str
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        assert not _BOUND_ACCESS_RE.search(text), (
            f"{relative_path} still accesses a single `bound` column"
        )
        assert not _SENSE_ACCESS_RE.search(text), (
            f"{relative_path} still accesses a `sense` key"
        )
