"""Tier-1 tests for the shared generic-constraint builder + id allocator.

Pure Python: no synthetic deck, no ``example/`` read, no ``import novomodelo``.
Covers the ``GENERIC_BOUNDS_SCHEMA`` drift-guard invariant, every
``GenericConstraintBuilder`` method (``add``/``add_constraint``/
``add_bound_row``/``result``), and the seeded/threaded ``ConstraintIdAllocator``.
"""

from __future__ import annotations

from novomodelo_bridge.core.generic_constraint_builder import (
    GENERIC_BOUNDS_SCHEMA,
    ConstraintIdAllocator,
    GenericConstraintBuilder,
)
from novomodelo_bridge.core.generic_constraint_format import GENERIC_BOUNDS_COLUMNS


def test_generic_bounds_schema_matches_shared_columns() -> None:
    """The builder's bounds schema stays in lockstep with the F3 column list."""
    assert GENERIC_BOUNDS_SCHEMA.names == list(GENERIC_BOUNDS_COLUMNS)


def test_add_two_sided_slot_yields_one_id_with_both_endpoints() -> None:
    """A single two-sided slot produces one constraint id and one bound row."""
    builder = GenericConstraintBuilder(ConstraintIdAllocator())

    cid = builder.add(
        name="RE_1",
        description="RE generic constraint 1",
        expression="hydro_generation(0)",
        slack={"enabled": True, "penalty": 1000.0},
        slots=[(0, None, 10.0, 20.0)],
    )

    assert cid == 0
    result = builder.result()
    assert result is not None
    assert result.constraints == [
        {
            "id": 0,
            "name": "RE_1",
            "description": "RE generic constraint 1",
            "expression": "hydro_generation(0)",
            "slack": {"enabled": True, "penalty": 1000.0},
        }
    ]
    assert result.bounds.num_rows == 1
    row = result.bounds.to_pylist()[0]
    assert row == {
        "constraint_id": 0,
        "stage_id": 0,
        "block_id": None,
        "bound_lower": 10.0,
        "bound_upper": 20.0,
    }


def test_add_constraint_and_bound_row_append_in_call_order() -> None:
    """``add_constraint`` + ``add_bound_row`` calls append rows in call order."""
    builder = GenericConstraintBuilder(ConstraintIdAllocator())

    cid = builder.add_constraint(
        name="AGRINT_0",
        description="AGRINT constraint 0",
        expression="hydro_outflow(1)",
        slack={"enabled": False},
    )
    builder.add_bound_row(cid, stage_id=0, block_id=None, lower=5.0, upper=None)
    builder.add_bound_row(cid, stage_id=1, block_id=None, lower=7.0, upper=None)

    result = builder.result()
    assert result is not None
    assert result.constraints == [
        {
            "id": cid,
            "name": "AGRINT_0",
            "description": "AGRINT constraint 0",
            "expression": "hydro_outflow(1)",
            "slack": {"enabled": False},
        }
    ]
    rows = result.bounds.to_pylist()
    assert [r["stage_id"] for r in rows] == [0, 1]
    assert [r["bound_lower"] for r in rows] == [5.0, 7.0]


def test_add_with_all_unbounded_slots_returns_none_and_adds_no_id() -> None:
    """No slot bounded on either side -> no constraint, no id consumed."""
    allocator = ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)

    cid = builder.add(
        name="HQ_1",
        description="HQ generic constraint 1",
        expression="hydro_turbined(2)",
        slack={"enabled": True, "penalty": 100.0},
        slots=[(0, None, None, None), (1, None, 1e21, -1e21)],
    )

    assert cid is None
    assert builder.result() is None
    assert allocator.next_id == 0


def test_constraint_id_allocator_seeded_start_and_advance() -> None:
    """A seeded allocator starts at its seed and advances/no-ops correctly."""
    alloc = ConstraintIdAllocator(7)
    assert alloc.next_id == 7
    alloc.advance(0)
    assert alloc.next_id == 7
    alloc.advance(2)
    assert alloc.next_id == 9


def test_constraint_id_allocator_advances_contiguously() -> None:
    """The allocator hands out contiguous, non-overlapping ID ranges."""
    alloc = ConstraintIdAllocator()
    assert alloc.next_id == 0
    alloc.advance(3)  # VminOP used IDs 0,1,2
    assert alloc.next_id == 3  # electric starts here
    alloc.advance(0)  # electric produced nothing
    assert alloc.next_id == 3  # AGRINT still starts at 3
    alloc.advance(2)
    assert alloc.next_id == 5
