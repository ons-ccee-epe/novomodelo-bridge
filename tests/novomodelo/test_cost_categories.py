"""Drift guards for the canonical Cobre cost taxonomy (PRES-05).

Both the dashboard cost chart (``COST_GROUPS``) and the comparator cost breakdown
(``_COST_MAP`` + ``read_cobre_cost_breakdown``) must classify exactly the same set
of Cobre cost-component columns. These tests fail if either stack drifts (the bug
PRES-05 fixed: the comparator silently dropped ``contract_cost`` and the dashboard
lumped ``excess_cost`` into "Other").
"""

from __future__ import annotations

from cobre_bridge.cobre.cost_categories import (
    AGGREGATE_COST_COLUMNS,
    COBRE_COST_COMPONENT_COLUMNS,
    COST_PARTITION_COLUMNS,
)


def test_component_and_aggregate_columns_are_disjoint() -> None:
    assert not (set(COBRE_COST_COMPONENT_COLUMNS) & AGGREGATE_COST_COLUMNS)
    assert not (set(COBRE_COST_COMPONENT_COLUMNS) & COST_PARTITION_COLUMNS)
    assert len(COBRE_COST_COMPONENT_COLUMNS) == len(set(COBRE_COST_COMPONENT_COLUMNS))


def test_dashboard_cost_groups_cover_exactly_the_canonical_columns() -> None:
    from cobre_bridge.dashboard.chart_helpers import COST_GROUPS

    mapped = {col for cols in COST_GROUPS.values() for col in cols}
    assert mapped == set(COBRE_COST_COMPONENT_COLUMNS), (
        "dashboard COST_GROUPS drifted from the canonical cost-component set: "
        f"missing={set(COBRE_COST_COMPONENT_COLUMNS) - mapped}, "
        f"extra={mapped - set(COBRE_COST_COMPONENT_COLUMNS)}"
    )


def test_dashboard_non_cost_cols_use_shared_aggregate_set() -> None:
    from cobre_bridge.dashboard.chart_helpers import _NON_COST_COLS

    assert _NON_COST_COLS == COST_PARTITION_COLUMNS | AGGREGATE_COST_COLUMNS


def test_comparator_cost_map_covers_exactly_the_canonical_columns() -> None:
    from cobre_bridge.comparators.charts import _COST_MAP

    mapped = {col for _, _, cobre_cols, _ in _COST_MAP for col in cobre_cols}
    assert mapped == set(COBRE_COST_COMPONENT_COLUMNS), (
        "comparator _COST_MAP drifted from the canonical cost-component set: "
        f"missing={set(COBRE_COST_COMPONENT_COLUMNS) - mapped}, "
        f"extra={mapped - set(COBRE_COST_COMPONENT_COLUMNS)}"
    )


def test_comparator_exclusion_set_is_the_shared_aggregate_set() -> None:
    from cobre_bridge.comparators.charts import _COBRE_NON_COST_KEYS

    assert _COBRE_NON_COST_KEYS == AGGREGATE_COST_COLUMNS
