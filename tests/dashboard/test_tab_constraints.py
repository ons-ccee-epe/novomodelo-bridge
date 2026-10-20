"""Unit tests for novomodelo_bridge.dashboard.tabs.constraints.

Covers module constants, can_render, _build_metrics_row, the new
_compute_violation_zones, _build_constraint_lhs_data, _build_lhs_section,
_add_type_filter_and_row_attrs helpers, and the full render() path using
MagicMock data with real polars/pandas objects for fields that get accessed
as LazyFrames/DataFrames.

Also covers the sense-free shape: constraint
dicts here carry no ``sense`` key and ``gc_bounds`` fixtures carry the
``bound_lower``/``bound_upper`` endpoint pair instead of a single ``bound``
column, matching what the constraint-bounds writers emit.
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
    _compute_violation_zones,
    can_render,
    render,
)
from novomodelo_bridge.dashboard.tabs.constraints_utils import (
    bound_value_column,
    build_constraints_summary_table,
    derive_constraint_shape,
)

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
    constraint_ids: list[int] | None = None,
    slack_value: float = 0.0,
) -> pd.DataFrame:
    """Return a minimal gc_violations DataFrame.

    When *constraint_ids* is None, returns an empty DataFrame.
    """
    if constraint_ids is None:
        return pd.DataFrame(columns=["constraint_id", "slack_value"])
    rows = [
        {"constraint_id": cid, "slack_value": slack_value} for cid in constraint_ids
    ]
    return pd.DataFrame(rows)


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

    Sets real polars LazyFrames for hydros_lf and exchanges_lf to prevent
    MagicMock auto-chaining from causing OOM.
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
    # Real empty LazyFrames prevent MagicMock auto-chaining OOM
    data.hydros_lf = pl.LazyFrame()
    data.exchanges_lf = pl.LazyFrame()
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
    gc_violations = _make_gc_violations(
        constraint_ids=[constraints[0]["id"]], slack_value=5.0
    )
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
    assert "Total Violation Cost" in html


def test_build_metrics_row_with_violation_cost() -> None:
    """_build_metrics_row must display the summed violation cost when column present."""
    # 2 scenarios x 2 stages x 100.0 = 400.0 total
    costs = _make_costs(
        n_scenarios=2,
        n_stages=2,
        include_violation_cost=True,
        violation_cost=100.0,
    )
    data = _make_mock_data(costs=costs)
    html = _build_metrics_row(data)
    assert "R$ 400" in html


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
    data = _make_mock_data(gc_violations=_make_gc_violations(constraint_ids=None))
    html = _build_metrics_row(data)
    assert ">0<" in html


# ---------------------------------------------------------------------------
# test__compute_violation_zones
# ---------------------------------------------------------------------------


def test_compute_violation_zones_ge_single_dip() -> None:
    """Floor-only (">="-style): violation at stage 1 where p10=80 < bound_lower=85."""
    p10 = [100.0, 80.0, 90.0]
    p90 = [110.0, 95.0, 105.0]
    bound_lower = [85.0, 85.0, 85.0]
    bound_upper: list[float | None] = [None, None, None]
    result = _compute_violation_zones(p10, p90, bound_lower, bound_upper)
    assert result == [{"start": 1, "end": 1}]


def test_compute_violation_zones_le_single_spike() -> None:
    """Ceiling-only ("<="-style): violation at stage 1 where p90=120 > bound_upper=110."""
    p10 = [90.0, 90.0, 90.0]
    p90 = [100.0, 120.0, 90.0]
    bound_lower: list[float | None] = [None, None, None]
    bound_upper = [110.0, 110.0, 110.0]
    result = _compute_violation_zones(p10, p90, bound_lower, bound_upper)
    assert result == [{"start": 1, "end": 1}]


def test_compute_violation_zones_no_violations() -> None:
    """No violations when band never crosses bound."""
    p10 = [100.0, 100.0, 100.0]
    p90 = [110.0, 110.0, 110.0]
    bound_lower = [85.0, 85.0, 85.0]
    bound_upper: list[float | None] = [None, None, None]
    result = _compute_violation_zones(p10, p90, bound_lower, bound_upper)
    assert result == []


def test_compute_violation_zones_missing_bound_skipped() -> None:
    """Stages with no bound at all (both endpoints None) must not be violations."""
    p10 = [80.0, 80.0, 80.0]
    p90 = [90.0, 90.0, 90.0]
    bound_lower: list[float | None] = [85.0, None, 85.0]
    bound_upper: list[float | None] = [None, None, None]
    result = _compute_violation_zones(p10, p90, bound_lower, bound_upper)
    # Stage 0 violated, missing bound at 1 breaks the zone, stage 2 violated → two zones
    assert result == [{"start": 0, "end": 0}, {"start": 2, "end": 2}]


def test_compute_violation_zones_contiguous_range() -> None:
    """Floor-only: contiguous violation across stages 0–2 produces a single interval."""
    p10 = [70.0, 70.0, 70.0]
    p90 = [80.0, 80.0, 80.0]
    bound_lower = [85.0, 85.0, 85.0]
    bound_upper: list[float | None] = [None, None, None]
    result = _compute_violation_zones(p10, p90, bound_lower, bound_upper)
    assert result == [{"start": 0, "end": 2}]


def test_compute_violation_zones_empty_inputs() -> None:
    """Empty input lists must return empty result without error."""
    result = _compute_violation_zones([], [], [], [])
    assert result == []


def test_compute_violation_zones_two_sided_range_flags_both_directions() -> None:
    """A genuine distinct-endpoint range (``bound_lower`` != ``bound_upper``) flags
    a below-floor breach AND an above-ceiling breach independently, in one call —
    not a single-sense ``>=``/``<=`` branch."""
    p10 = [5.0, 50.0, 50.0]
    p90 = [5.0, 50.0, 95.0]
    bound_lower = [10.0, 10.0, 10.0]
    bound_upper = [90.0, 90.0, 90.0]
    result = _compute_violation_zones(p10, p90, bound_lower, bound_upper)
    assert result == [{"start": 0, "end": 0}, {"start": 2, "end": 2}]


# ---------------------------------------------------------------------------
# test__build_constraint_lhs_data
# ---------------------------------------------------------------------------


def test_build_constraint_lhs_data_structure() -> None:
    """Returned dict must have stages, xlabels, and constraints with expected keys."""
    constraints = [
        {"id": 0, "name": "VminOP_c0"},
        {"id": 1, "name": "RE_c1"},
    ]
    lhs_df = _make_lhs_df([0, 1], n_scenarios=2, n_stages=3, lhs_value=120.0)
    gc_bounds = pd.concat(
        [
            _make_gc_bounds([0], n_stages=3, sense=">="),
            _make_gc_bounds([1], n_stages=3, sense="<="),
        ],
        ignore_index=True,
    )
    stage_labels = {0: "Jan", 1: "Feb", 2: "Mar"}

    result = _build_constraint_lhs_data(constraints, lhs_df, gc_bounds, stage_labels)

    assert "stages" in result
    assert "xlabels" in result
    assert "constraints" in result
    assert result["stages"] == [0, 1, 2]
    assert result["xlabels"] == ["Jan", "Feb", "Mar"]

    for cid_str in ("0", "1"):
        entry = result["constraints"][cid_str]
        for key in (
            "name",
            "sense",
            "lhs_p10",
            "lhs_p50",
            "lhs_p90",
            "bound",
            "violations",
        ):
            assert key in entry, f"Missing key '{key}' in constraint {cid_str}"
        assert len(entry["lhs_p10"]) == 3
        assert len(entry["lhs_p50"]) == 3
        assert len(entry["lhs_p90"]) == 3
        assert len(entry["bound"]) == 3


def test_build_constraint_lhs_data_uniform_lhs_gives_equal_percentiles() -> None:
    """Uniform LHS across scenarios must yield p10 == p50 == p90 at every stage."""
    constraints = [{"id": 0, "name": "VminOP_c0"}]
    lhs_df = _make_lhs_df([0], n_scenarios=3, n_stages=3, lhs_value=50.0)
    gc_bounds = _make_gc_bounds([0], n_stages=3)
    stage_labels = {0: "S0", 1: "S1", 2: "S2"}

    result = _build_constraint_lhs_data(constraints, lhs_df, gc_bounds, stage_labels)
    entry = result["constraints"]["0"]

    for i in range(3):
        assert math.isclose(entry["lhs_p10"][i], 50.0, abs_tol=1e-3)
        assert math.isclose(entry["lhs_p50"][i], 50.0, abs_tol=1e-3)
        assert math.isclose(entry["lhs_p90"][i], 50.0, abs_tol=1e-3)


def test_build_constraint_lhs_data_violations_populated_for_ge() -> None:
    """>=: violations list is non-empty when p10 < bound at some stages."""
    constraints = [{"id": 0, "name": "VminOP_c0"}]
    # lhs_value=80 < bound=100 → p10 will be 80, bound is 100 → violation everywhere
    lhs_df = _make_lhs_df([0], n_scenarios=2, n_stages=3, lhs_value=80.0)
    gc_bounds = _make_gc_bounds([0], n_stages=3)  # bound=100.0
    stage_labels = {0: "S0", 1: "S1", 2: "S2"}

    result = _build_constraint_lhs_data(constraints, lhs_df, gc_bounds, stage_labels)
    violations = result["constraints"]["0"]["violations"]

    assert len(violations) > 0
    assert violations[0]["start"] == 0


def test_build_constraint_lhs_data_empty_lhs_gives_zeros() -> None:
    """Empty lhs_df must produce zero arrays for p10/p50/p90."""
    constraints = [{"id": 0, "name": "VminOP_c0"}]
    lhs_df = pd.DataFrame(
        columns=["constraint_id", "scenario_id", "stage_id", "block_id", "lhs_value"]
    )
    gc_bounds = _make_gc_bounds([0], n_stages=3)
    stage_labels = {0: "S0", 1: "S1", 2: "S2"}

    result = _build_constraint_lhs_data(constraints, lhs_df, gc_bounds, stage_labels)
    entry = result["constraints"]["0"]

    assert entry["lhs_p10"] == [0.0, 0.0, 0.0]
    assert entry["lhs_p50"] == [0.0, 0.0, 0.0]
    assert entry["lhs_p90"] == [0.0, 0.0, 0.0]


def test_build_constraint_lhs_data_missing_bounds_gives_none_bound() -> None:
    """No gc_bounds rows for a constraint → bound array is all None."""
    constraints = [{"id": 99, "name": "VminOP_c99"}]
    lhs_df = _make_lhs_df([99], n_scenarios=2, n_stages=3)
    gc_bounds = pd.DataFrame(
        columns=["constraint_id", "stage_id", "block_id", "bound_lower", "bound_upper"]
    )
    stage_labels = {0: "S0", 1: "S1", 2: "S2"}

    result = _build_constraint_lhs_data(constraints, lhs_df, gc_bounds, stage_labels)
    entry = result["constraints"]["99"]

    assert all(v is None for v in entry["bound"])
    assert entry["violations"] == []
    # Falls back to the historical `c.get("sense", "<=")` default shape.
    assert entry["sense"] == "<="


# ---------------------------------------------------------------------------
# Derived sense + bound value parity with a legacy (sense, bound) case
# ---------------------------------------------------------------------------


def test_build_constraint_lhs_data_derives_ge_from_lower_endpoint() -> None:
    """A VminOP-style floor (lower-only) renders exactly like a legacy
    ``sense=">="`` case: same ``">="`` label, same numeric bound."""
    constraints = [{"id": 0, "name": "VminOP_c0"}]  # sense-free
    lhs_df = _make_lhs_df([0], n_scenarios=2, n_stages=2, lhs_value=120.0)
    gc_bounds = _make_gc_bounds([0], n_stages=2, sense=">=", value=500.0)
    stage_labels = {0: "S0", 1: "S1"}

    result = _build_constraint_lhs_data(constraints, lhs_df, gc_bounds, stage_labels)
    entry = result["constraints"]["0"]

    assert entry["sense"] == ">="
    assert entry["bound"] == [500.0, 500.0]


def test_build_constraint_lhs_data_derives_le_from_upper_endpoint() -> None:
    """An RE/AGRINT-style ceiling (upper-only) renders exactly like a
    legacy ``sense="<="`` case: same ``"<="`` label, same numeric bound."""
    constraints = [{"id": 1, "name": "RE_c1"}]  # sense-free
    lhs_df = _make_lhs_df([1], n_scenarios=2, n_stages=2, lhs_value=120.0)
    gc_bounds = _make_gc_bounds([1], n_stages=2, sense="<=", value=200.0)
    stage_labels = {0: "S0", 1: "S1"}

    result = _build_constraint_lhs_data(constraints, lhs_df, gc_bounds, stage_labels)
    entry = result["constraints"]["1"]

    assert entry["sense"] == "<="
    assert entry["bound"] == [200.0, 200.0]


# ---------------------------------------------------------------------------
# A genuine distinct-endpoint "range" band
# (DECOMP RE/HQ/HV) must render both the floor and the ceiling and
# violation-test both directions, instead of silently dropping the floor.
# ---------------------------------------------------------------------------


def test_build_constraint_lhs_data_range_flags_floor_and_ceiling() -> None:
    """A two-sided ``L <= expr <= U`` band (distinct ``bound_lower``/
    ``bound_upper``) must expose both endpoints — not just the ceiling — and
    flag a below-floor breach and an above-ceiling breach independently."""
    constraints = [{"id": 0, "name": "HQ_c0"}]
    # 2 identical scenarios per stage so p10 == p50 == p90 == the stage's LHS
    # value, isolating the floor/ceiling test from percentile spread.
    lhs_values = {0: 5.0, 1: 50.0, 2: 95.0}
    rows = [
        {
            "constraint_id": 0,
            "scenario_id": sid,
            "stage_id": stage,
            "block_id": 0,
            "lhs_value": val,
        }
        for stage, val in lhs_values.items()
        for sid in range(2)
    ]
    lhs_df = pd.DataFrame(rows)
    gc_bounds = _make_gc_bounds_range([0], n_stages=3, lower=10.0, upper=90.0)
    stage_labels = {0: "S0", 1: "S1", 2: "S2"}

    result = _build_constraint_lhs_data(constraints, lhs_df, gc_bounds, stage_labels)
    entry = result["constraints"]["0"]

    assert entry["sense"] == "range"
    # Both endpoints are exposed at every stage — the floor is no longer
    # silently dropped in favour of the ceiling.
    assert entry["bound_lower"] == [10.0, 10.0, 10.0]
    assert entry["bound_upper"] == [90.0, 90.0, 90.0]
    # Legacy single-value field keeps the ceiling, matching the pre-fix value.
    assert entry["bound"] == [90.0, 90.0, 90.0]

    # Stage 0 (LHS=5, below the 10.0 floor) and stage 2 (LHS=95, above the
    # 90.0 ceiling) must both be flagged; stage 1 (LHS=50, inside the band)
    # must not be.
    violated_stages = {
        i for v in entry["violations"] for i in range(v["start"], v["end"] + 1)
    }
    assert violated_stages == {0, 2}


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


def test_build_lhs_section_html_contains_required_elements() -> None:
    """_build_lhs_section must contain gc-constraint-sel, gc-lhs-chart, GC_LHS_DATA."""
    constraints = _make_constraints(n=2)
    gc_bounds = _make_gc_bounds([0, 1], n_stages=3)
    lhs_df = _make_lhs_df([0, 1], n_scenarios=2, n_stages=3)
    data = _make_mock_data(constraints=constraints, gc_bounds=gc_bounds, n_stages=3)
    data.gc_bounds = gc_bounds

    html = _build_lhs_section(data, lhs_df)

    assert "gc-constraint-sel" in html
    assert "gc-lhs-chart" in html
    assert "GC_LHS_DATA" in html


def test_build_lhs_section_html_has_one_option_per_constraint() -> None:
    """_build_lhs_section must emit one <option> per constraint."""
    constraints = _make_constraints(n=3)
    lhs_df = _make_lhs_df([0, 1, 2], n_scenarios=2, n_stages=2)
    data = _make_mock_data(constraints=constraints, n_stages=2)

    html = _build_lhs_section(data, lhs_df)

    assert html.count("<option") == 3


def test_build_lhs_section_json_blob_has_violations_key() -> None:
    """GC_LHS_DATA JSON embedded in the section must include 'violations' key."""
    constraints = _make_constraints(n=1)
    lhs_df = _make_lhs_df([0], n_scenarios=2, n_stages=2)
    data = _make_mock_data(constraints=constraints, n_stages=2)

    html = _build_lhs_section(data, lhs_df)

    assert '"violations"' in html


def test_build_lhs_section_range_data_and_floor_trace_logic_present() -> None:
    """A genuine two-sided range constraint's ``bound_lower``/``bound_upper``
    reach the embedded ``GC_LHS_DATA`` blob, and the JS floor-trace logic
    that draws the second ("Bound Floor") line is present in the script."""
    constraints = [{"id": 0, "name": "HQ_c0"}]
    lhs_df = _make_lhs_df([0], n_scenarios=2, n_stages=2, lhs_value=50.0)
    gc_bounds = _make_gc_bounds_range([0], n_stages=2, lower=10.0, upper=90.0)
    data = _make_mock_data(constraints=constraints, gc_bounds=gc_bounds, n_stages=2)

    html = _build_lhs_section(data, lhs_df)

    assert '"bound_lower":[10.0,10.0]' in html
    assert '"bound_upper":[90.0,90.0]' in html
    assert "hasDistinctFloor" in html
    assert "Bound Floor" in html


# ---------------------------------------------------------------------------
# test_type_filter_dropdown
# ---------------------------------------------------------------------------


def test_type_filter_dropdown_present_in_render() -> None:
    """render() HTML must contain gc-type-filter with 4 options."""
    data = _make_mock_data()

    with _render_with_stubs_ctx(data) as html:
        assert "gc-type-filter" in html
        assert (
            html.count("<option") >= 4
        )  # All, VminOP, RE, AGRINT + constraint options


def test_type_filter_has_all_four_options() -> None:
    """gc-type-filter must have options: All, VminOP, RE, AGRINT."""
    data = _make_mock_data()

    with _render_with_stubs_ctx(data) as html:
        # Find the filter select block
        assert 'value="All"' in html
        assert 'value="VminOP"' in html
        assert 'value="RE"' in html
        assert 'value="AGRINT"' in html


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
    gc_violations = _make_gc_violations(constraint_ids=None)
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

    mock_eval.assert_called_once_with(
        data.gc_resolved, data.hydros_lf, data.exchanges_lf
    )


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
            '"expression": "hydro_storage(0)", "slack": {"enabled": false}}]}'
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
