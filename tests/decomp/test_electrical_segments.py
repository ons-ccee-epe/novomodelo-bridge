"""Tests for the split of an electrical restriction into segments of one
relation and one violation treatment, for the ids and names of the generic
constraints the segments emit, and for the slack each treatment becomes (tier
1, no ``novomodelo`` import)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.electrical_expression import MalformedElectricalExpression
from novomodelo_bridge.decomp.converters.constraints import big_m_penalty
from novomodelo_bridge.decomp.converters.electrical_constraints import (
    EmittedConstraint,
    Treatment,
    _governing,
    emitted_constraints,
    report_ignored_hard_costs,
    segment_slack,
    split_restriction,
)
from novomodelo_bridge.decomp.converters.libs_electrical import (
    ElectricalRestriction,
    FormulaOverride,
    LibsElectricalModel,
    PeriodPatamarOverride,
    ViolationRow,
)
from tests.decomp.conftest import weekly_calendar

_CALENDAR = [replace(stage, block_hours=(84.0, 84.0)) for stage in weekly_calendar(3)]

_LHS = "ger_usih(285)"

_SOFT = ("soft", None)


def _inequality(
    code: int,
    rhs: str,
    *,
    overrides: tuple[PeriodPatamarOverride, ...] = (),
    violations: tuple[ViolationRow, ...] = (),
) -> ElectricalRestriction:
    """``ger_usih(285) <= rhs``."""
    return ElectricalRestriction(
        code=code,
        lhs=_LHS,
        operator="<=",
        rhs=rhs,
        is_formula=False,
        limits={},
        overrides=overrides,
        violations=violations,
    )


def _override(
    stage_start: int, stage_end: int, rhs: str, *, patamar: int | None = None
) -> PeriodPatamarOverride:
    return PeriodPatamarOverride(stage_start, stage_end, patamar, _LHS, "<=", rhs)


def _formula(
    code: int,
    lhs: str,
    *,
    limits: Mapping[tuple[int, int | None], tuple[float | None, float | None]]
    | None = None,
    formula_overrides: tuple[FormulaOverride, ...] = (),
    violations: tuple[ViolationRow, ...] = (),
) -> ElectricalRestriction:
    return ElectricalRestriction(
        code=code,
        lhs=lhs,
        operator=None,
        rhs=None,
        is_formula=True,
        limits=limits or {},
        overrides=(),
        violations=violations,
        formula_overrides=formula_overrides,
    )


def _at_most(rhs: str) -> tuple[str, str, str]:
    return (_LHS, "<=", rhs)


def test_overrides_split_a_restriction_by_block_then_stage() -> None:
    restriction = _inequality(
        703,
        "100",
        overrides=(_override(1, 1, "50", patamar=2), _override(1, 2, "70")),
    )

    segments = split_restriction(restriction, _CALENDAR)

    assert list(segments.items()) == [
        ((_at_most("100"), _SOFT), ((0, 0), (0, 1))),
        ((_at_most("70"), _SOFT), ((1, 0), (2, 0), (2, 1))),
        ((_at_most("50"), _SOFT), ((1, 1),)),
    ]


def test_a_treatment_range_splits_a_restriction_by_stage() -> None:
    restriction = _inequality(
        703,
        "100",
        violations=(
            ViolationRow(None, None, "soft", 2000.0),
            ViolationRow(1, 1, "hard", 5.0),
        ),
    )

    segments = split_restriction(restriction, _CALENDAR)

    assert segments == {
        (_at_most("100"), ("soft", 2000.0)): ((0, 0), (0, 1), (2, 0), (2, 1)),
        (_at_most("100"), ("hard", None)): ((1, 0), (1, 1)),
    }


@pytest.mark.parametrize(
    ("restriction", "expected"),
    [
        pytest.param(
            _formula(405, "ger_usih(285) + D", limits={(0, None): (None, 6800.0)}),
            {(("ger_usih(285) + D",), _SOFT): ((0, 0), (0, 1))},
            id="formula-bounded-in-one-stage",
        ),
        pytest.param(_formula(406, "ger_usih(287) + E"), {}, id="formula-unbounded"),
        pytest.param(
            _inequality(703, "100"),
            {
                (_at_most("100"), _SOFT): (
                    (0, 0),
                    (0, 1),
                    (1, 0),
                    (1, 1),
                    (2, 0),
                    (2, 1),
                )
            },
            id="inequality",
        ),
    ],
)
def test_the_segments_partition_the_cells_that_hold_a_row(
    restriction: ElectricalRestriction, expected: dict
) -> None:
    segments = split_restriction(restriction, _CALENDAR)

    cells = [cell for segment in segments.values() for cell in segment]
    assert segments == expected
    assert len(cells) == len(set(cells))
    assert set(cells) == set(_governing(restriction, _CALENDAR))


@pytest.mark.parametrize(
    ("restriction", "message"),
    [
        pytest.param(
            _inequality(
                703, "100", overrides=(_override(1, 1, "50"), _override(0, 1, "70"))
            ),
            "restriction 703: stage 1 block 0 is covered by 2 overrides of the same "
            "precedence",
            id="two-every-block-overrides",
        ),
        pytest.param(
            _inequality(
                703,
                "100",
                violations=(
                    ViolationRow(None, None, "soft", 2000.0),
                    ViolationRow(None, None, "hard", None),
                ),
            ),
            "restriction 703: the whole horizon is given more than one violation "
            "treatment",
            id="two-whole-horizon-treatments",
        ),
        pytest.param(
            _inequality(
                703,
                "100",
                violations=(
                    ViolationRow(0, 1, "soft", 2000.0),
                    ViolationRow(1, 2, "hard", None),
                ),
            ),
            "restriction 703: stage 1 is covered by more than one violation "
            "treatment range",
            id="overlapping-treatment-ranges",
        ),
        pytest.param(
            _inequality(703, "100", overrides=(_override(0, 2, "50", patamar=3),)),
            "restriction 703 override 1: patamar 3 outside 1..2",
            id="override-patamar-beyond-the-blocks",
        ),
        pytest.param(
            _formula(
                406,
                "ger_usih(287) + E",
                formula_overrides=(FormulaOverride(0, 0, 3, "ger_usih(287)"),),
            ),
            "restriction 406 override 1: patamar 3 outside 1..2",
            id="patamar-checked-without-a-cell",
        ),
        pytest.param(
            _formula(
                406,
                "ger_usih(287) + E",
                violations=(
                    ViolationRow(0, 2, "soft", 2000.0),
                    ViolationRow(2, 2, "hard", None),
                ),
            ),
            "restriction 406: stage 2 is covered by more than one violation "
            "treatment range",
            id="treatments-checked-without-a-cell",
        ),
    ],
)
def test_a_conflict_between_rows_stops_the_split(
    restriction: ElectricalRestriction, message: str
) -> None:
    with pytest.raises(MalformedElectricalExpression, match=f"^{re.escape(message)}$"):
        split_restriction(restriction, _CALENDAR)


def test_the_split_does_not_depend_on_insertion_order() -> None:
    restriction = _inequality(
        703,
        "100",
        overrides=(_override(1, 1, "50", patamar=2), _override(1, 2, "70")),
        violations=(
            ViolationRow(None, None, "soft", 2000.0),
            ViolationRow(0, 0, "hard", None),
            ViolationRow(2, 2, "soft", 10.0),
        ),
    )
    reversed_rows = replace(
        restriction,
        overrides=restriction.overrides[::-1],
        violations=restriction.violations[::-1],
    )

    assert list(split_restriction(reversed_rows, _CALENDAR).items()) == list(
        split_restriction(restriction, _CALENDAR).items()
    )


def _model(*restrictions: ElectricalRestriction) -> LibsElectricalModel:
    return LibsElectricalModel(
        expressions={},
        aliases={},
        restrictions={restriction.code: restriction for restriction in restrictions},
        rules={},
    )


def _three_segments(code: int) -> ElectricalRestriction:
    return _inequality(
        code,
        "100",
        overrides=(_override(1, 1, "50", patamar=2), _override(1, 2, "70")),
    )


def _two_segments(code: int) -> ElectricalRestriction:
    return _inequality(code, "100", violations=(ViolationRow(1, 1, "hard", None),))


def _identities(emitted: Sequence[EmittedConstraint]) -> list[tuple[int, str]]:
    return [(constraint_id, name) for constraint_id, name, _, _ in emitted]


def test_a_single_segment_keeps_the_deck_code() -> None:
    model = _model(_inequality(401, "100"))

    result = emitted_constraints(model, _CALENDAR, taken={0, 1, 2})

    assert _identities(result[401]) == [(401, "constraint_401")]


def test_split_restrictions_take_ids_above_every_id_of_the_case() -> None:
    model = _model(_inequality(401, "100"), _three_segments(405), _two_segments(409))

    result = emitted_constraints(model, _CALENDAR, taken={0, 1, 2})

    assert {code: _identities(emitted) for code, emitted in result.items()} == {
        401: [(401, "constraint_401")],
        405: [
            (410, "constraint_405_1"),
            (411, "constraint_405_2"),
            (412, "constraint_405_3"),
        ],
        409: [(413, "constraint_409_1"), (414, "constraint_409_2")],
    }
    assert [(key, cells) for _, _, key, cells in result[405]] == list(
        split_restriction(model.restrictions[405], _CALENDAR).items()
    )


def test_a_code_another_constraint_uses_is_renumbered() -> None:
    model = _model(_inequality(601, "100"))

    result = emitted_constraints(model, _CALENDAR, taken={0, 1, 601})

    assert _identities(result[601]) == [(602, "constraint_601")]


def test_a_restriction_without_cells_has_no_constraint() -> None:
    model = _model(_formula(421, "ger_usih(287) + E"), _two_segments(405))

    result = emitted_constraints(model, _CALENDAR, taken=set())

    assert list(result) == [405, 421]
    assert result[421] == ()
    assert _identities(result[405]) == [
        (422, "constraint_405_1"),
        (423, "constraint_405_2"),
    ]


def test_identities_do_not_depend_on_insertion_order() -> None:
    model = _model(
        _inequality(401, "100"),
        _three_segments(405),
        _two_segments(409),
        _formula(421, "ger_usih(287) + E"),
        _inequality(601, "100"),
    )
    reversed_model = replace(
        model, restrictions=dict(reversed(list(model.restrictions.items())))
    )

    assert list(
        emitted_constraints(
            reversed_model, _CALENDAR, taken=dict.fromkeys((601, 1, 0)).keys()
        ).items()
    ) == list(emitted_constraints(model, _CALENDAR, taken={0, 1, 601}).items())


def test_a_treatment_becomes_the_generic_slack() -> None:
    default_penalty = big_m_penalty(5100.0)

    slacks = [
        segment_slack(treatment, default_penalty, context="restriction 405")
        for treatment in (("hard", None), ("soft", 2000.0), ("soft", None))
    ]

    assert slacks == [
        {"enabled": False},
        {"enabled": True, "penalty": 2000.0, "penalty_basis": "per_hour"},
        {"enabled": True, "penalty": 51000.0, "penalty_basis": "per_hour"},
    ]
    assert list(slacks[1]) == ["enabled", "penalty", "penalty_basis"]


_COST_MESSAGE = (
    "restriction 405: the cost of a soft violation treatment is finite and above "
    "0, got {}"
)


@pytest.mark.parametrize(
    ("treatment", "message"),
    [
        pytest.param(("soft", 0.0), _COST_MESSAGE.format(0.0), id="zero-cost"),
        pytest.param(("soft", -1.0), _COST_MESSAGE.format(-1.0), id="negative-cost"),
        pytest.param(
            ("soft", float("inf")), _COST_MESSAGE.format("inf"), id="infinite-cost"
        ),
        pytest.param(
            ("soft", None),
            "restriction 405: no violation cost is given and the deck declares no "
            "deficit cost to derive the default from",
            id="no-default-penalty",
        ),
    ],
)
def test_a_soft_treatment_needs_a_positive_cost(
    treatment: Treatment, message: str
) -> None:
    with pytest.raises(MalformedElectricalExpression, match=f"^{re.escape(message)}$"):
        segment_slack(treatment, 0.0, context="restriction 405")

    assert segment_slack(("hard", None), 0.0, context="restriction 405") == {
        "enabled": False
    }


def test_a_hard_treatment_with_a_cost_is_reported_once() -> None:
    without_cost = _inequality(
        413, "100", violations=(ViolationRow(None, None, "hard", None),)
    )
    model = _model(
        _inequality(409, "100", violations=(ViolationRow(1, 2, "hard", 7.5),)),
        without_cost,
        _inequality(405, "100", violations=(ViolationRow(None, None, "hard", 5.0),)),
    )

    with dx.collect() as sink:
        report_ignored_hard_costs(model)
    with dx.collect() as silent:
        report_ignored_hard_costs(_model(without_cost))

    [diagnostic] = sink
    assert diagnostic.code == "decomp-electrical-hard-treatment-cost-ignored"
    assert diagnostic.severity is dx.Severity.INFO
    assert diagnostic.category == "Special constraints"
    assert diagnostic.title.endswith(" (2)")
    assert diagnostic.table is not None
    assert diagnostic.table.columns == ["Code", "Stages", "Cost"]
    assert [list(row) for row in diagnostic.table.rows] == [
        [405, "all", 5.0],
        [409, "1 to 2", 7.5],
    ]
    assert diagnostic.table.caption == "Stage ids are 0-based."
    assert silent == []
