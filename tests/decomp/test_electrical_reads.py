"""Tests for the exact read-cell analysis of the electrical constraints
(tier 1, no ``novomodelo`` import)."""

from __future__ import annotations

import re
import string
from collections.abc import Mapping
from dataclasses import replace

import pytest

from novomodelo_bridge.core.electrical_expression import MalformedElectricalExpression
from novomodelo_bridge.decomp.converters.electrical_constraints import read_cells
from novomodelo_bridge.decomp.converters.libs_electrical import (
    ElectricalAlias,
    ElectricalExpression,
    ElectricalRestriction,
    FormulaOverride,
    LibsElectricalModel,
    PeriodPatamarOverride,
)
from tests.decomp.conftest import weekly_calendar

_CALENDAR = [replace(stage, block_hours=(84.0, 84.0)) for stage in weekly_calendar(3)]

_STUDY = frozenset((stage, block) for stage in range(3) for block in range(2))

_LHS = "ger_usih(285)"


def _inequality(
    code: int,
    rhs: str | None,
    *,
    overrides: tuple[PeriodPatamarOverride, ...] = (),
    stage_start: int | None = None,
    stage_end: int | None = None,
    habilita: int | None = None,
) -> ElectricalRestriction:
    """``ger_usih(285) <= rhs``, or no relation of its own when ``rhs`` is None."""
    return ElectricalRestriction(
        code=code,
        lhs=None if rhs is None else _LHS,
        operator=None if rhs is None else "<=",
        rhs=rhs,
        is_formula=False,
        limits={},
        overrides=overrides,
        stage_start=stage_start,
        stage_end=stage_end,
        habilita=habilita,
    )


def _override(
    stage_start: int,
    stage_end: int,
    rhs: str,
    *,
    patamar: int | None = None,
    lhs: str = _LHS,
) -> PeriodPatamarOverride:
    return PeriodPatamarOverride(stage_start, stage_end, patamar, lhs, "<=", rhs)


def _formula(
    code: int,
    lhs: str,
    *,
    limits: Mapping[tuple[int, int | None], tuple[float | None, float | None]]
    | None = None,
    formula_overrides: tuple[FormulaOverride, ...] = (),
    stage_start: int | None = None,
    stage_end: int | None = None,
    habilita: int | None = None,
) -> ElectricalRestriction:
    return ElectricalRestriction(
        code=code,
        lhs=lhs,
        operator=None,
        rhs=None,
        is_formula=True,
        limits=limits or {},
        overrides=(),
        stage_start=stage_start,
        stage_end=stage_end,
        habilita=habilita,
        formula_overrides=formula_overrides,
    )


def _model(
    *restrictions: ElectricalRestriction,
    expressions: Mapping[str, str] | None = None,
    rules: Mapping[int, str] | None = None,
) -> LibsElectricalModel:
    """Every one-letter name and ``R1`` that is not an expression is a declared
    alias."""
    expressions = expressions or {}
    aliases = [*string.ascii_uppercase, "R1"]
    return LibsElectricalModel(
        expressions={
            name: ElectricalExpression(code, name, formula)
            for code, (name, formula) in enumerate(expressions.items(), start=1)
        },
        aliases={
            name: ElectricalAlias(code, name, {})
            for code, name in enumerate(aliases, start=1)
            if name not in expressions
        },
        restrictions={r.code: r for r in restrictions},
        rules=dict(rules or {}),
    )


def _reversed[K, V](mapping: Mapping[K, V]) -> dict[K, V]:
    return dict(reversed(list(mapping.items())))


def _cells(*cells: tuple[int, int]) -> frozenset[tuple[int, int]]:
    return frozenset(cells)


def _restriction_405(habilita: int | None = None) -> ElectricalRestriction:
    return _formula(
        405,
        "ger_usih(285) + D",
        limits={
            (0, None): (None, 6800.0),
            (1, None): (100.0, 200.0),
            (1, 2): (None, None),
        },
        formula_overrides=(FormulaOverride(0, 0, 1, "ger_usih(285) + F"),),
        habilita=habilita,
    )


def _restriction_702() -> ElectricalRestriction:
    return _inequality(
        702,
        "A",
        overrides=(_override(1, 1, "B"), _override(1, 1, "C", patamar=2)),
    )


def test_an_override_past_the_horizon_reads_nothing_there() -> None:
    model = _model(
        _inequality(
            701, "A", overrides=(_override(1, 2, "B"),), stage_start=0, stage_end=1
        )
    )

    reads = read_cells(model, _CALENDAR)

    assert reads[("alias", "A")] == _cells((0, 0), (0, 1))
    assert reads[("alias", "B")] == _cells((1, 0), (1, 1))


def test_a_block_override_wins_over_an_every_block_override_and_the_base() -> None:
    reads = read_cells(_model(_restriction_702()), _CALENDAR)

    assert reads[("alias", "A")] == _cells((0, 0), (0, 1), (2, 0), (2, 1))
    assert reads[("alias", "B")] == _cells((1, 0))
    assert reads[("alias", "C")] == _cells((1, 1))


def test_a_formula_reads_only_where_a_limit_row_bounds_it() -> None:
    model = _model(_restriction_405(), _formula(406, "ger_usih(287) + E"))

    reads = read_cells(model, _CALENDAR)

    assert reads[("alias", "D")] == _cells((0, 1), (1, 0))
    assert reads[("alias", "F")] == _cells((0, 0))
    assert reads[("alias", "E")] == frozenset()


def test_a_rule_condition_reads_only_where_its_restriction_holds_a_row() -> None:
    model = _model(_restriction_405(habilita=502), rules={502: "G > 1", 503: "H > 1"})

    reads = read_cells(model, _CALENDAR)

    assert reads[("alias", "G")] == _cells((0, 0), (0, 1), (1, 0))
    assert ("alias", "H") not in reads


def test_a_reference_reads_in_the_cells_of_the_relation_that_cites_it() -> None:
    model = _model(
        _formula(701, "ger_usih(285) + K", stage_start=0, stage_end=0),
        ElectricalRestriction(
            code=801,
            lhs="re(701)",
            operator="<=",
            rhs="10",
            is_formula=False,
            limits={},
            overrides=(),
        ),
    )

    reads = read_cells(model, _CALENDAR)

    assert reads[("reference", 701)] == _STUDY
    assert reads[("alias", "K")] == _STUDY


def test_cells_held_without_an_every_block_row_or_a_base_relation_are_read() -> None:
    model = _model(
        _formula(407, "ger_usih(285) + L", limits={(1, 2): (0.0, 10.0)}),
        _inequality(408, None, overrides=(_override(0, 2, "M"),), habilita=504),
        rules={504: "N > 0"},
    )

    reads = read_cells(model, _CALENDAR)

    assert reads[("alias", "L")] == _cells((1, 1))
    assert reads[("alias", "M")] == _STUDY
    assert reads[("alias", "N")] == _STUDY


def test_every_branch_expression_and_data_term_is_read_and_cycles_terminate() -> None:
    model = _model(
        _inequality(901, "se(R1 > 0, X, carga_ande() + peq_SE_PCHgd_SE)"),
        expressions={"X": "Y + P", "Y": "X + Q"},
    )

    reads = read_cells(model, _CALENDAR)

    for key in (
        ("alias", "R1"),
        ("alias", "P"),
        ("alias", "Q"),
        ("expression", "X"),
        ("expression", "Y"),
        ("data", "carga_ande"),
        ("data", "peq_SE_PCHgd_SE"),
    ):
        assert reads[key] == _STUDY, key


def test_unreferenced_names_are_absent() -> None:
    model = _model(
        _inequality(703, None, overrides=(_override(1, 1, "T"),)),
        expressions={"Z": "S"},
    )

    reads = read_cells(model, _CALENDAR)

    assert ("expression", "Z") not in reads
    assert ("alias", "S") not in reads
    assert reads[("alias", "T")] == _cells((1, 0), (1, 1))


def test_a_cell_two_overrides_govern_fails() -> None:
    model = _model(
        _inequality(703, None, overrides=(_override(1, 1, "T"), _override(1, 1, "U")))
    )
    message = (
        "restriction 703: stage 1 block 0 is covered by 2 overrides of the same "
        "precedence"
    )

    with pytest.raises(MalformedElectricalExpression, match=f"^{re.escape(message)}$"):
        read_cells(model, _CALENDAR)


def test_reads_do_not_depend_on_insertion_order() -> None:
    model = _model(
        _restriction_702(),
        _restriction_405(habilita=502),
        _inequality(703, None, overrides=(_override(1, 1, "T"), _override(2, 2, "U"))),
        _inequality(901, "X + W"),
        expressions={"X": "Y + P", "Y": "Q"},
        rules={502: "G > 1", 503: "H > 1"},
    )
    shuffled = LibsElectricalModel(
        expressions=_reversed(model.expressions),
        aliases=_reversed(model.aliases),
        restrictions={
            code: replace(restriction, overrides=restriction.overrides[::-1])
            for code, restriction in _reversed(model.restrictions).items()
        },
        rules=_reversed(model.rules),
    )

    assert read_cells(shuffled, _CALENDAR) == read_cells(model, _CALENDAR)


def test_an_override_patamar_beyond_a_stage_governs_no_cell_there() -> None:
    model = _model(_inequality(704, "A", overrides=(_override(0, 2, "B", patamar=3),)))

    reads = read_cells(model, _CALENDAR)

    assert reads[("alias", "A")] == _STUDY
    assert reads[("alias", "B")] == frozenset()


def test_an_inequality_override_on_a_formula_free_restriction_reads_its_rhs_and_lhs() -> (
    None
):
    model = _model(
        _inequality(
            705, "A", overrides=(_override(1, 1, "W", lhs="ger_usih(285) + V"),)
        )
    )

    reads = read_cells(model, _CALENDAR)

    assert reads[("alias", "A")] == _cells((0, 0), (0, 1), (2, 0), (2, 1))
    assert reads[("alias", "V")] == _cells((1, 0), (1, 1))
    assert reads[("alias", "W")] == _cells((1, 0), (1, 1))


def test_a_called_data_term_reads_the_declared_name_that_shadows_it() -> None:
    model = replace(
        _model(_inequality(706, "carga_ande()")),
        aliases={"carga_ande": ElectricalAlias(1, "carga_ande", {})},
    )

    reads = read_cells(model, _CALENDAR)

    assert reads == {("alias", "carga_ande"): _STUDY}


def test_an_invalid_reference_reads_nothing() -> None:
    model = _model(
        _inequality(707, "re(A) + re(999) + re(708) + re(7.0)"),
        _inequality(708, None, overrides=(_override(0, 2, "B"),)),
    )

    reads = read_cells(model, _CALENDAR)

    assert set(reads) == {("alias", "B")}
