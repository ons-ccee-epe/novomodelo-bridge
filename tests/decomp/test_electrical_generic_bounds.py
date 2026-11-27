"""Tests for the ``generic_constraint_bounds.parquet`` rows of the generic
constraints the electrical restrictions emit (tier 1, no ``novomodelo``
import)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import pyarrow as pa
import pytest

from novomodelo_bridge.core.electrical_expression import MalformedElectricalExpression
from novomodelo_bridge.core.generic_constraint_builder import GENERIC_BOUNDS_SCHEMA
from novomodelo_bridge.decomp.converters.electrical_constraints import (
    EmittedConstraint,
    build_generic_bounds,
    build_generic_outputs,
    emitted_constraints,
)
from novomodelo_bridge.decomp.converters.electrical_translate import EntityTables
from novomodelo_bridge.decomp.converters.libs_electrical import (
    ElectricalRestriction,
    LibsElectricalModel,
    PeriodPatamarOverride,
)
from tests.decomp.conftest import weekly_calendar

_CALENDAR = [replace(stage, block_hours=(84.0, 84.0)) for stage in weekly_calendar(3)]

_EVERY_STAGE_UPPER = {(stage, None): (None, 500.0) for stage in range(3)}

type Limits = Mapping[tuple[int, int | None], tuple[float | None, float | None]]
type Row = tuple[int, int, int | None, float | None, float | None]


def _formula(code: int, limits: Limits) -> ElectricalRestriction:
    return ElectricalRestriction(
        code=code,
        lhs="ger_usih(285)+ger_usih(287)",
        operator=None,
        rhs=None,
        is_formula=True,
        limits=limits,
        overrides=(),
    )


def _model(*restrictions: ElectricalRestriction) -> LibsElectricalModel:
    return LibsElectricalModel(
        expressions={},
        aliases={},
        restrictions={restriction.code: restriction for restriction in restrictions},
        rules={},
    )


def _bounds(model: LibsElectricalModel) -> pa.Table:
    return build_generic_bounds(
        model, _CALENDAR, emitted_constraints(model, _CALENDAR, taken=set())
    )


def _rows(table: pa.Table) -> list[Row]:
    return [
        (
            row["constraint_id"],
            row["stage_id"],
            row["block_id"],
            row["bound_lower"],
            row["bound_upper"],
        )
        for row in table.to_pylist()
    ]


def test_an_every_block_limit_is_one_row_and_block_limits_are_explicit() -> None:
    model = _model(
        _formula(601, _EVERY_STAGE_UPPER),
        _formula(602, {(0, 1): (-100.0, 100.0), (0, 2): (-100.0, 100.0)}),
    )

    table = _bounds(model)

    assert table.schema.equals(GENERIC_BOUNDS_SCHEMA)
    assert _rows(table) == [
        (601, 0, None, None, 500.0),
        (601, 1, None, None, 500.0),
        (601, 2, None, None, 500.0),
        (602, 0, 0, -100.0, 100.0),
        (602, 0, 1, -100.0, 100.0),
    ]


def test_a_block_limit_makes_the_stage_explicit() -> None:
    model = _model(
        _formula(603, {(0, None): (None, 500.0), (0, 2): (10.0, 20.0)}),
        _formula(604, {(0, None): (None, 500.0), (0, 2): (None, None)}),
    )

    assert _rows(_bounds(model)) == [
        (603, 0, 0, None, 500.0),
        (603, 0, 1, 10.0, 20.0),
        (604, 0, 0, None, 500.0),
    ]


def test_an_inequality_holds_a_row_with_null_endpoints_in_each_cell() -> None:
    inequality = ElectricalRestriction(
        code=405,
        lhs="ger_usih(285)",
        operator="<=",
        rhs="100",
        is_formula=False,
        limits={},
        overrides=(PeriodPatamarOverride(1, 1, 2, "ger_usih(285)", "<=", "80"),),
        stage_start=0,
        stage_end=1,
    )
    model = _model(inequality)
    emitted = emitted_constraints(model, _CALENDAR, taken=set())
    ids = {name: constraint_id for constraint_id, name, _, _ in emitted[405]}

    rows = _rows(build_generic_bounds(model, _CALENDAR, emitted))

    first, second = ids["constraint_405_1"], ids["constraint_405_2"]
    assert [row for row in rows if row[0] == first] == [
        (first, 0, None, None, None),
        (first, 1, 0, None, None),
    ]
    assert [row for row in rows if row[0] == second] == [(second, 1, 1, None, None)]
    assert len(rows) == 3
    assert all(row[1] != 2 for row in rows)


def test_every_record_has_rows_and_every_row_has_a_record() -> None:
    model = _model(_formula(421, {}), _formula(601, _EVERY_STAGE_UPPER))

    outputs = build_generic_outputs(
        model,
        _CALENDAR,
        EntityTables(
            code_map={"hydros": {285: 0, 287: 1}}, machine_sets={}, contract_kinds={}
        ),
        {},
        reserved=frozenset(),
        taken=set(),
        default_penalty=51000.0,
    )

    bounded = set(outputs.bounds.column("constraint_id").to_pylist())
    assert bounded == {record["id"] for record in outputs.constraints} == {601}
    assert outputs.bounds.schema.equals(GENERIC_BOUNDS_SCHEMA)


def _reversed_constraint(constraint: EmittedConstraint) -> EmittedConstraint:
    constraint_id, name, key, cells = constraint
    return (constraint_id, name, key, tuple(reversed(cells)))


def test_bounds_do_not_depend_on_insertion_order() -> None:
    first: Limits = {
        (2, None): (1.0, 2.0),
        (0, 2): (3.0, None),
        (0, None): (None, 4.0),
        (0, 1): (5.0, 6.0),
    }
    second: Limits = {(1, None): (7.0, None), (0, None): (None, 8.0)}
    forward_model = _model(_formula(405, first), _formula(402, second))
    backward_model = _model(
        _formula(402, dict(reversed(list(second.items())))),
        _formula(405, dict(reversed(list(first.items())))),
    )
    emitted = emitted_constraints(backward_model, _CALENDAR, taken=set())
    backward_emitted: dict[int, Sequence[EmittedConstraint]] = {
        code: [_reversed_constraint(constraint) for constraint in constraints]
        for code, constraints in reversed(list(emitted.items()))
    }

    forward = _bounds(forward_model)
    backward = build_generic_bounds(backward_model, _CALENDAR, backward_emitted)

    assert forward.equals(backward)
    assert _rows(forward) == [
        (402, 0, None, None, 8.0),
        (402, 1, None, 7.0, None),
        (405, 0, 0, 5.0, 6.0),
        (405, 0, 1, 3.0, None),
        (405, 2, None, 1.0, 2.0),
    ]


def test_values_are_written_as_read() -> None:
    lower, upper = 0.1 + 0.2, 1 / 3
    limits: Limits = {(0, None): (lower, upper), (1, 1): (lower, upper)}

    rows = _rows(_bounds(_model(_formula(405, limits))))

    assert rows == [(405, 0, None, lower, upper), (405, 1, 0, lower, upper)]


@pytest.mark.parametrize(
    "limits",
    [
        {(1, None): (1.0, 2.0), (1, 3): (1.0, 2.0)},
        {(1, None): (1.0, 2.0), (1, 3): (None, None)},
        {(1, 3): (1.0, 2.0)},
    ],
    ids=["bounded", "open", "without-cells"],
)
def test_a_limit_patamar_outside_the_stage_blocks_fails(limits: Limits) -> None:
    with pytest.raises(
        MalformedElectricalExpression,
        match=r"^restriction 405 limits stage 1: patamar 3 outside 1\.\.2$",
    ):
        _bounds(_model(_formula(405, limits)))
