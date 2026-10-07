"""Unit tests for novomodelo_bridge.dashboard.tabs.constraints.

Covers module constants, can_render, _build_metrics_row, the new
_compute_violation_zones, _build_constraint_lhs_data, _build_lhs_section,
_add_type_filter_and_row_attrs helpers, and the full render() path using
MagicMock data with real polars/pandas objects for fields that get accessed
as LazyFrames/DataFrames.

Also covers the F3 sense-free migration: constraint
dicts here carry no ``sense`` key and ``gc_bounds`` fixtures carry the F3
``bound_lower``/``bound_upper`` endpoint pair instead of a single ``bound``
column, matching what the current constraint-bounds writers now emit.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
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
    """Return a list of minimal constraint dicts (F3 sense-free shape)."""
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
    """Return a minimal gc_bounds DataFrame in the F3 endpoint shape.

    *sense*/*value* describe the pre-F3 ``(sense, bound)`` pair this fixture
    is migrated from; :func:`sense_to_interval` maps it to the F3
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
    # A real, non-existent Path so render()'s load_rho_acum_overrides() call
    # degrades to {} instead of running json.load() against a MagicMock.
    data.case_dir = Path("nonexistent-novomodelo-bridge-dashboard-case")
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
    the two-sided test this fix adds in place of the single-sense ``>=``/``<=``
    branch (the F3 range-mishandling case this fixes)."""
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
# Derived sense + bound value parity with a pre-F3 case
# ---------------------------------------------------------------------------


def test_build_constraint_lhs_data_derives_ge_from_lower_endpoint() -> None:
    """A VminOP-style floor (F3 lower-only) renders exactly like a pre-F3
    ``sense=">="`` case: same ``">="`` label, same numeric bound."""
    constraints = [{"id": 0, "name": "VminOP_c0"}]  # sense-free (F3)
    lhs_df = _make_lhs_df([0], n_scenarios=2, n_stages=2, lhs_value=120.0)
    gc_bounds = _make_gc_bounds([0], n_stages=2, sense=">=", value=500.0)
    stage_labels = {0: "S0", 1: "S1"}

    result = _build_constraint_lhs_data(constraints, lhs_df, gc_bounds, stage_labels)
    entry = result["constraints"]["0"]

    assert entry["sense"] == ">="
    assert entry["bound"] == [500.0, 500.0]


def test_build_constraint_lhs_data_derives_le_from_upper_endpoint() -> None:
    """An RE/AGRINT-style ceiling (F3 upper-only) renders exactly like a
    pre-F3 ``sense="<="`` case: same ``"<="`` label, same numeric bound."""
    constraints = [{"id": 1, "name": "RE_c1"}]  # sense-free (F3)
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

_PATCH_EVAL = (
    "novomodelo_bridge.dashboard.tabs.constraints.evaluate_constraint_expressions"
)

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


def test_render_wires_rho_acum_overrides_into_evaluate_constraint_expressions() -> None:
    """render() loads case_dir's rho_acum overrides and forwards them to
    evaluate_constraint_expressions -- the dashboard's half of the LHS-scale
    fix (constraints_compare/decomp_results carry the compare-side half)."""
    from unittest.mock import patch

    data = _make_mock_data()
    sentinel_overrides = {0: {0: 2.19}}
    with (
        patch(
            "novomodelo_bridge.dashboard.tabs.constraints.load_rho_acum_overrides",
            return_value=sentinel_overrides,
        ) as mock_load,
        patch(_PATCH_EVAL, return_value=_STUB_LHS_DF) as mock_eval,
    ):
        render(data)

    mock_load.assert_called_once_with(data.case_dir)
    assert mock_eval.call_args.args[-1] == sentinel_overrides


# ---------------------------------------------------------------------------
# Parser tests — @name sigil + literal coefficient handling
# ---------------------------------------------------------------------------


class TestParseExpression:
    """``parse_expression`` recognises both legacy literal coefficients and the novomodelo
    HEAD ``@name`` sigil."""

    def _parse(self, expr: str) -> list[tuple]:
        from novomodelo_bridge.novomodelo.constraint_expr import parse_expression

        return parse_expression(expr)

    def test_legacy_literal_coefficient(self) -> None:
        terms = self._parse("5.68 * hydro_storage(78)")
        assert terms == [(5.68, None, "hydro_storage", 78)]

    def test_implicit_unit_coefficient(self) -> None:
        terms = self._parse("hydro_generation(145)")
        assert terms == [(1.0, None, "hydro_generation", 145)]

    def test_unary_minus_implicit(self) -> None:
        terms = self._parse("- line_exchange(4)")
        assert terms == [(-1.0, None, "line_exchange", 4)]

    def test_at_name_implicit_unit(self) -> None:
        """``@rho_acum_h78 * hydro_storage(78)`` carries an implicit 1.0 literal."""
        terms = self._parse("@rho_acum_h78 * hydro_storage(78)")
        assert terms == [(1.0, "rho_acum_h78", "hydro_storage", 78)]

    def test_hydro_storage_final_normalised_to_hydro_storage(self) -> None:
        """The converter emits ``hydro_storage_final``; it is the end-of-stage volume
        Sᴷ, identical to the legacy ``hydro_storage`` and normalised to it so
        downstream storage-only detection and column lookup stay unchanged."""
        terms = self._parse("@rho_acum_h3 * hydro_storage_final(3)")
        assert terms == [(1.0, "rho_acum_h3", "hydro_storage", 3)]

    def test_hydro_storage_final_and_legacy_alias_coexist(self) -> None:
        """A mix of the new and legacy names both collapse to ``hydro_storage``."""
        terms = self._parse("hydro_storage_final(0) + hydro_storage(1)")
        assert terms == [
            (1.0, None, "hydro_storage", 0),
            (1.0, None, "hydro_storage", 1),
        ]

    def test_at_name_with_literal_scale(self) -> None:
        """``0.5 * @rho_eq_h12 * hydro_generation(12)`` keeps both."""
        terms = self._parse("0.5 * @rho_eq_h12 * hydro_generation(12)")
        assert terms == [(0.5, "rho_eq_h12", "hydro_generation", 12)]

    def test_at_name_with_unary_minus(self) -> None:
        terms = self._parse("- @rho_eq_h12 * hydro_generation(12)")
        assert terms == [(-1.0, "rho_eq_h12", "hydro_generation", 12)]

    def test_mixed_terms_in_one_expression(self) -> None:
        """Sum of @name and literal terms parse together."""
        terms = self._parse(
            "@rho_acum_h0 * hydro_storage(0) + 2.5 * hydro_storage(1) - hydro_storage(2)"
        )
        assert terms == [
            (1.0, "rho_acum_h0", "hydro_storage", 0),
            (2.5, None, "hydro_storage", 1),
            (-1.0, None, "hydro_storage", 2),
        ]

    def test_unknown_param_name_still_parses(self) -> None:
        """Unrecognised ``@name`` parses; evaluator decides how to handle it."""
        terms = self._parse("@some_other_thing * hydro_generation(5)")
        assert terms == [(1.0, "some_other_thing", "hydro_generation", 5)]


class TestResolveParamToColumn:
    """``resolve_param_to_column`` maps novomodelo-bridge's per-hydro names to simulation
    columns."""

    def _resolve(self, name: str):
        from novomodelo_bridge.novomodelo.constraint_expr import resolve_param_to_column

        return resolve_param_to_column(name)

    def test_rho_acum(self) -> None:
        assert self._resolve("rho_acum_h78") == (
            "accumulated_productivity_mw_per_m3s",
            78,
        )

    def test_rho_eq(self) -> None:
        assert self._resolve("rho_eq_h12") == ("equivalent_productivity_mw_per_m3s", 12)

    def test_unknown(self) -> None:
        assert self._resolve("rho_something_else_h0") is None

    def test_no_hydro_suffix(self) -> None:
        assert self._resolve("rho_acum") is None


# ---------------------------------------------------------------------------
# evaluate_constraint_expressions — @name resolves against simulation column
# ---------------------------------------------------------------------------


def _make_hydros_lf(
    *,
    with_productivity: bool = True,
) -> pl.LazyFrame:
    """Tiny 2-stage 1-scenario synthetic hydros parquet shape."""
    cols = {
        "scenario_id": pl.Series([0, 0, 0, 0], dtype=pl.Int64),
        "stage_id": pl.Series([0, 0, 1, 1], dtype=pl.Int32),
        "block_id": pl.Series([0, 0, 0, 0], dtype=pl.Int32),
        "hydro_id": pl.Series([0, 1, 0, 1], dtype=pl.Int32),
        "storage_final_hm3": pl.Series([100.0, 200.0, 150.0, 250.0], dtype=pl.Float64),
        "generation_mw": pl.Series([10.0, 20.0, 30.0, 40.0], dtype=pl.Float64),
    }
    if with_productivity:
        cols["accumulated_productivity_mw_per_m3s"] = pl.Series(
            [2.0, 3.0, 2.0, 3.0], dtype=pl.Float64
        )
        cols["equivalent_productivity_mw_per_m3s"] = pl.Series(
            [0.5, 0.7, 0.5, 0.7], dtype=pl.Float64
        )
    return pl.DataFrame(cols).lazy()


def _empty_exchanges_lf() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "scenario_id": pl.Series([], dtype=pl.Int64),
            "stage_id": pl.Series([], dtype=pl.Int32),
            "block_id": pl.Series([], dtype=pl.Int32),
            "line_id": pl.Series([], dtype=pl.Int32),
            "net_flow_mw": pl.Series([], dtype=pl.Float64),
        }
    ).lazy()


class TestEvaluateAtName:
    """``evaluate_constraint_expressions`` resolves ``@name`` against simulation
    columns."""

    def _evaluate(self, expression: str, *, with_productivity: bool = True):
        from novomodelo_bridge.novomodelo.constraint_expr import (
            evaluate_constraint_expressions,
        )

        constraints = [
            {
                "id": 0,
                "name": "test",
                "expression": expression,
                "sense": ">=",
                "slack": {"enabled": False},
            }
        ]
        return evaluate_constraint_expressions(
            constraints,
            _make_hydros_lf(with_productivity=with_productivity),
            _empty_exchanges_lf(),
        )

    def test_at_rho_acum_multiplies_by_column(self) -> None:
        """``@rho_acum_h0 * hydro_storage(0)`` at stage 0 = 2.0 * 100.0 = 200.0."""
        df = self._evaluate("@rho_acum_h0 * hydro_storage(0)")
        # storage-only constraint -> block_id = 0
        s0 = df[(df["stage_id"] == 0) & (df["scenario_id"] == 0)]
        assert len(s0) == 1
        assert s0["lhs_value"].iloc[0] == 200.0

        s1 = df[(df["stage_id"] == 1) & (df["scenario_id"] == 0)]
        assert s1["lhs_value"].iloc[0] == 300.0  # 2.0 * 150.0

    def test_at_rho_eq_with_generation(self) -> None:
        """``@rho_eq_h1 * hydro_generation(1)`` at stage 0 = 0.7 * 20.0 = 14.0."""
        df = self._evaluate("@rho_eq_h1 * hydro_generation(1)")
        s0 = df[(df["stage_id"] == 0) & (df["scenario_id"] == 0)]
        assert s0["lhs_value"].iloc[0] == 14.0

    def test_literal_only_still_works(self) -> None:
        """Backward-compat: ``5.0 * hydro_storage(0)`` evaluates to 5.0 * storage."""
        df = self._evaluate("5.0 * hydro_storage(0)")
        s0 = df[(df["stage_id"] == 0) & (df["scenario_id"] == 0)]
        assert s0["lhs_value"].iloc[0] == 500.0

    def test_literal_times_at_name(self) -> None:
        """``0.5 * @rho_acum_h0 * hydro_storage(0)`` = 0.5 * 2.0 * 100.0 = 100.0."""
        df = self._evaluate("0.5 * @rho_acum_h0 * hydro_storage(0)")
        s0 = df[(df["stage_id"] == 0) & (df["scenario_id"] == 0)]
        assert s0["lhs_value"].iloc[0] == 100.0

    def test_at_name_falls_back_to_zero_when_column_missing(self) -> None:
        """If the productivity column isn't in the parquet, contribution is 0."""
        df = self._evaluate("@rho_acum_h0 * hydro_storage(0)", with_productivity=False)
        s0 = df[(df["stage_id"] == 0) & (df["scenario_id"] == 0)]
        assert s0["lhs_value"].iloc[0] == 0.0


class TestEvaluateAtNameRhoAcumOverride:
    """``rho_acum_overrides`` replaces the default productivity column at
    ``@rho_acum_h{id}`` -- the mechanism VminOP/RHE need to match the LP's
    own resolution instead of the simulation's default point productivity
    (see :func:`novomodelo_bridge.novomodelo.constraint_expr.load_rho_acum_overrides`)."""

    def _evaluate(
        self,
        expression: str,
        rho_acum_overrides: dict[int, dict[int, float]] | None,
    ):
        from novomodelo_bridge.novomodelo.constraint_expr import (
            evaluate_constraint_expressions,
        )

        constraints = [
            {
                "id": 0,
                "name": "test",
                "expression": expression,
                "sense": ">=",
                "slack": {"enabled": False},
            }
        ]
        return evaluate_constraint_expressions(
            constraints,
            _make_hydros_lf(),
            _empty_exchanges_lf(),
            rho_acum_overrides,
        )

    def test_override_wins_over_default_column(self) -> None:
        """Sim default at (hydro 0, stage 0) is 2.0 (see _make_hydros_lf); a
        distinct override (5.68, mirroring the real VminOP/RHE gap) must win:
        LHS = 5.68 * 100.0, not 2.0 * 100.0."""
        df = self._evaluate("@rho_acum_h0 * hydro_storage(0)", {0: {0: 5.68, 1: 5.9}})
        s0 = df[(df["stage_id"] == 0) & (df["scenario_id"] == 0)]
        assert s0["lhs_value"].iloc[0] == 5.68 * 100.0
        assert s0["lhs_value"].iloc[0] != 2.0 * 100.0

    def test_missing_stage_in_override_falls_back_to_default_column(self) -> None:
        """An override dict that only covers stage 0 leaves stage 1 on the
        sim's default column."""
        df = self._evaluate("@rho_acum_h0 * hydro_storage(0)", {0: {0: 5.68}})
        s1 = df[(df["stage_id"] == 1) & (df["scenario_id"] == 0)]
        assert s1["lhs_value"].iloc[0] == 2.0 * 150.0  # default column at stage 1

    def test_override_for_other_hydro_does_not_affect_this_one(self) -> None:
        """Overrides are keyed per hydro id; an override for hydro 1 must not
        leak into hydro 0's term."""
        df = self._evaluate("@rho_acum_h0 * hydro_storage(0)", {1: {0: 9.99}})
        s0 = df[(df["stage_id"] == 0) & (df["scenario_id"] == 0)]
        assert s0["lhs_value"].iloc[0] == 2.0 * 100.0  # unchanged default

    def test_none_overrides_preserves_prior_behaviour(self) -> None:
        """``rho_acum_overrides=None`` (the default) is identical to the
        pre-override evaluator."""
        df = self._evaluate("@rho_acum_h0 * hydro_storage(0)", None)
        s0 = df[(df["stage_id"] == 0) & (df["scenario_id"] == 0)]
        assert s0["lhs_value"].iloc[0] == 2.0 * 100.0

    def test_override_lets_a_satisfied_stage_clear_its_own_bound(self) -> None:
        """The regression guarded here: the *default* column can put the
        LHS on the wrong side of its own bound; the LP-faithful override
        clears it. Default LHS = 2.0*100=200 < bound=300 (falsely violated);
        override LHS = 5.68*100=568 >= bound=300 (actually satisfied)."""
        bound = 300.0
        default_df = self._evaluate("@rho_acum_h0 * hydro_storage(0)", None)
        override_df = self._evaluate(
            "@rho_acum_h0 * hydro_storage(0)", {0: {0: 5.68, 1: 5.9}}
        )
        default_lhs = default_df[
            (default_df["stage_id"] == 0) & (default_df["scenario_id"] == 0)
        ]["lhs_value"].iloc[0]
        override_lhs = override_df[
            (override_df["stage_id"] == 0) & (override_df["scenario_id"] == 0)
        ]["lhs_value"].iloc[0]

        assert default_lhs < bound
        assert override_lhs >= bound


def _make_multiblock_hydros_lf() -> pl.LazyFrame:
    """1 scenario, 2 stages, 2 blocks. Storage lives at block 0; generation
    spans both blocks. A storage-only constraint must collapse to one row per
    (scenario, stage) at block 0 — a per-block fan-out would duplicate it.
    """
    return pl.DataFrame(
        {
            "scenario_id": pl.Series([0] * 8, dtype=pl.Int64),
            "stage_id": pl.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=pl.Int32),
            "block_id": pl.Series([0, 0, 1, 1, 0, 0, 1, 1], dtype=pl.Int32),
            "hydro_id": pl.Series([0, 1, 0, 1, 0, 1, 0, 1], dtype=pl.Int32),
            "storage_final_hm3": pl.Series(
                [100.0, 200.0, 100.0, 200.0, 150.0, 250.0, 150.0, 250.0],
                dtype=pl.Float64,
            ),
            "generation_mw": pl.Series(
                [10.0, 20.0, 11.0, 21.0, 30.0, 40.0, 31.0, 41.0], dtype=pl.Float64
            ),
            "accumulated_productivity_mw_per_m3s": pl.Series(
                [2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0], dtype=pl.Float64
            ),
        }
    ).lazy()


class TestStorageOnlyFastPath:
    """Storage-only constraints emit exactly one row per (scenario, stage) at
    block_id=0 — even with multiple blocks present in the simulation output.

    Regression for the dead ``storage_only`` fast path: the variable-type set
    was built from the term's param_name slot instead of its variable-type
    slot, so the branch was unreachable and storage-only constraints were
    fanned out across every block. Correctness was only recovered downstream
    because consumers re-averaged across blocks.
    """

    def _evaluate(self, expression: str):
        from novomodelo_bridge.novomodelo.constraint_expr import (
            evaluate_constraint_expressions,
        )

        constraints = [
            {
                "id": 0,
                "name": "test",
                "expression": expression,
                "sense": ">=",
                "slack": {"enabled": False},
            }
        ]
        return evaluate_constraint_expressions(
            constraints,
            _make_multiblock_hydros_lf(),
            _empty_exchanges_lf(),
        )

    def test_storage_only_collapses_to_one_row_per_stage(self) -> None:
        df = self._evaluate("@rho_acum_h0 * hydro_storage(0)")
        # Two stages, one scenario -> exactly two rows, not 4 (block fan-out).
        assert len(df) == 2
        assert set(df["block_id"].unique()) == {0}
        by_stage = df.set_index("stage_id")["lhs_value"]
        assert by_stage[0] == 200.0  # 2.0 * 100.0
        assert by_stage[1] == 300.0  # 2.0 * 150.0

    def test_mixed_constraint_still_fans_out_per_block(self) -> None:
        """A constraint touching generation keeps the per-block grid."""
        df = self._evaluate("@rho_acum_h0 * hydro_storage(0) + hydro_generation(0)")
        # Two stages x two blocks -> 4 rows.
        assert len(df) == 4
        assert set(df["block_id"].unique()) == {0, 1}


# ---------------------------------------------------------------------------
# build_constraints_summary_table derives its "Sense"
# column + "Bound Range" from F3 endpoints
# ---------------------------------------------------------------------------


class TestSummaryTableDerivesSenseFromBounds:
    """``build_constraints_summary_table`` shows the same Sense + Bound Range
    a pre-F3 case showed, now sourced from ``bound_lower``/``bound_upper``."""

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


# ---------------------------------------------------------------------------
# dashboard/data.py's generic-constraint bounds loader reads the F3
# bound_lower/bound_upper columns straight through
# ---------------------------------------------------------------------------


class TestLoadGenericConstraintsF3Shape:
    """``load_generic_constraints`` reads whatever the converter wrote — the
    F3 ``bound_lower``/``bound_upper`` pair — with no ``bound`` column."""

    def test_loads_f3_bounds_parquet_verbatim(self, tmp_path: Path) -> None:
        from novomodelo_bridge.dashboard.data import load_generic_constraints

        constraints_dir = tmp_path / "constraints"
        constraints_dir.mkdir()
        (constraints_dir / "generic_constraints.json").write_text(
            '{"constraints": [{"id": 0, "name": "VminOP_c0", '
            '"expression": "hydro_storage(0)", "slack": {"enabled": false}}]}'
        )
        pq.write_table(
            pa.table(
                {
                    "constraint_id": pa.array([0], type=pa.int32()),
                    "stage_id": pa.array([0], type=pa.int32()),
                    "block_id": pa.array([0], type=pa.int32()),
                    "bound_lower": pa.array([500.0], type=pa.float64()),
                    "bound_upper": pa.array([None], type=pa.float64()),
                }
            ),
            constraints_dir / "generic_constraint_bounds.parquet",
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
