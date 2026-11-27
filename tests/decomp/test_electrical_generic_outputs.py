"""Tests for the generic constraints, named expressions and parameter entries
the electrical cards convert to: idecomp-shaped frames ->
``read_libs_electrical`` -> ``build_generic_outputs`` (tier 1, no
``novomodelo`` import)."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.electrical_expression import MalformedElectricalExpression
from novomodelo_bridge.core.errors import UnsupportedSourceError
from novomodelo_bridge.decomp.converters.electrical_constraints import (
    GenericOutputs,
    build_generic_outputs,
    emitted_constraints,
    report_split_restrictions,
)
from novomodelo_bridge.decomp.converters.electrical_translate import EntityTables
from novomodelo_bridge.decomp.converters.libs_electrical import (
    LibsElectricalModel,
    read_libs_electrical,
)
from tests.decomp.conftest import (
    SHORT_FORM_ACCESSORS,
    alias,
    alias_valor,
    expressao,
    habilita,
    horizonte,
    horizonte_data,
    re_formula,
    re_formula_data_pat,
    re_formula_per_pat,
    re_inequacao,
    re_inequacao_per_pat,
    re_lim_formula,
    re_lim_formula_data,
    regra,
    spelled,
    tratamento_violacao,
    tratamento_violacao_periodo,
    weekly_calendar,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence, Set

    import pandas as pd

    from novomodelo_bridge.decomp.temporal import OperativeStage

    type Frames = dict[str, pd.DataFrame]

_CALENDAR = [replace(stage, block_hours=(84.0, 84.0)) for stage in weekly_calendar(3)]

_SPLIT = "decomp-electrical-restrictions-split"

# No source code equals the id it maps to, so an untranslated code cannot pass.
_TABLES = EntityTables(
    code_map={
        "hydros": {6: 0, 251: 1},
        "thermals": {12: 3},
        "submarkets": {1: 0, 4: 1},
    },
    machine_sets={},
    contract_kinds={},
)

_DEFAULT_PENALTY = 51000.0

_DEFAULT_SLACK = {
    "enabled": True,
    "penalty": _DEFAULT_PENALTY,
    "penalty_basis": "per_hour",
}

_SOURCE_TERM = re.compile(r"ger_usih|demanda|\b(?:se|re)\(")


def _model(
    frames: Frames,
    *,
    short: bool = False,
    calendar: Sequence[OperativeStage] = _CALENDAR,
) -> LibsElectricalModel:
    model = read_libs_electrical(spelled(frames, short=short), calendar)
    assert model is not None
    return model


def _build(
    model: LibsElectricalModel, *, taken: Set[int] = frozenset()
) -> GenericOutputs:
    # The cards cite no data term, so no data term has values.
    return build_generic_outputs(
        model,
        _CALENDAR,
        _TABLES,
        {},
        reserved=frozenset(),
        taken=taken,
        default_penalty=_DEFAULT_PENALTY,
    )


def _outputs(frames: Frames, *, taken: Set[int] = frozenset()) -> GenericOutputs:
    return _build(_model(frames), taken=taken)


def _leaves(node: object) -> Iterator[object]:
    match node:
        case dict():
            for value in node.values():
                yield from _leaves(value)
        case list():
            for item in node:
                yield from _leaves(item)
        case _:
            yield node


def _every_card() -> Frames:
    """Cards of every kind, each short-form card among them: 401 splits by a
    formula override and by a treatment range, 402 is bounded only through
    its date cards, 403 is an inequality under rule 201, 404 is declared only
    by a period row and reads ``re(402)``."""
    return {
        "expressao_eletrica": expressao(
            (1, "north_exchange", "ener_interc(1,4) - ener_interc(4,1)"),
            (2, "margins", "MARGIN + FLOOR"),
        ),
        "alias_eletrico": alias((1, "MARGIN"), (2, "FLOOR")),
        "alias_eletrico_valor_periodo_patamar": alias_valor(
            (1, 1, 3, None, 300.0), (2, 1, 3, None, 10.0)
        ),
        "restricao_eletrica_formula": re_formula(
            (401, "ger_usih(6) + margins"), (402, "ger_usit(12)")
        ),
        "restricao_eletrica_horizonte_periodo": horizonte((401, 1, 3)),
        "restricao_eletrica_horizonte_data": horizonte_data(
            (402, "2026-07-11", "2026-07-24")
        ),
        "restricao_eletrica_limite_formula_periodo_patamar": re_lim_formula(
            (401, 1, 3, None, -1e31, 6300.0), (401, 2, 2, 1, 10.0, 20.0)
        ),
        "restricao_eletrica_limites_formula_data_patamar": re_lim_formula_data(
            (402, "2026-07-11", "2026-07-17", 1, 10.0, 1e31)
        ),
        "restricao_eletrica_formula_periodo_patamar": re_formula_per_pat(
            (401, 2, 3, 2, "2*ger_usih(6)  + 1")
        ),
        "restricao_eletrica_formula_data_patamar": re_formula_data_pat(
            (402, "2026-07-11", "2026-07-17", 1, "ger_usit(12) + ger_usih(251)")
        ),
        "restricao_eletrica_inequacao": re_inequacao(
            (
                403,
                "ger_usih(251) + north_exchange",
                ">=",
                "se(demanda_sin <= 82000, 0.05, 0.1)*val_demanda(1)",
            )
        ),
        "restricao_eletrica_inequacao_periodo_patamar": re_inequacao_per_pat(
            (404, 1, 3, 1, "re(402) + north_exchange", "<=", "14700")
        ),
        "restricao_eletrica_regra_ativacao": regra(
            (201, "demanda_sin > 82000 & demanda(4) <= 900"),
            (202, "demanda_sin <= 82000"),
        ),
        "restricao_eletrica_habilita": habilita((403, 201)),
        "restricao_eletrica_tratamento_violacao": tratamento_violacao(
            (402, "hard", 1234.5)
        ),
        "restricao_eletrica_tratamento_violacao_periodo": tratamento_violacao_periodo(
            (401, 3, 3, "soft", 900.0)
        ),
    }


def test_records_carry_only_the_generic_fields() -> None:
    outputs = _outputs(
        {
            "expressao_eletrica": expressao(
                (1, "north_exchange", "ener_interc(1,4) - ener_interc(4,1)")
            ),
            "restricao_eletrica_inequacao": re_inequacao(
                (101, "ger_usih(6) + north_exchange", "<=", "0.1*val_demanda_sin")
            ),
            "restricao_eletrica_formula": re_formula((102, "ger_usit(12)")),
            "restricao_eletrica_limite_formula_periodo_patamar": re_lim_formula(
                (102, 1, 3, None, -1e31, 1500.0)
            ),
            "restricao_eletrica_regra_ativacao": regra(
                (201, "demanda_sin > 500 & demanda(1) <= 900")
            ),
            "restricao_eletrica_habilita": habilita((101, 201)),
        }
    )

    records = {record["id"]: record for record in outputs.constraints}
    assert list(records) == [101, 102]
    assert list(records[101]) == ["id", "name", "expression", "active_when", "slack"]
    assert list(records[102]) == ["id", "name", "expression", "slack"]
    assert records[101] == {
        "id": 101,
        "name": "constraint_101",
        "expression": "hydro_generation(0) + @north_exchange "
        "<= 0.1 * (bus_load_value(0) + bus_load_value(1))",
        "active_when": "(bus_load_value(0) + bus_load_value(1)) > 500 "
        "& bus_load_value(0) <= 900",
        "slack": _DEFAULT_SLACK,
    }
    assert outputs.expressions == [
        {
            "name": "north_exchange",
            "expression": "line_flow(source_bus=0, target_bus=1) "
            "- line_flow(source_bus=1, target_bus=0)",
        }
    ]
    assert [list(record) for record in outputs.expressions] == [["name", "expression"]]
    assert None not in list(_leaves([outputs.constraints, outputs.expressions]))


def test_a_split_restriction_is_one_record_per_segment() -> None:
    outputs = _outputs(
        {
            "restricao_eletrica_inequacao_periodo_patamar": re_inequacao_per_pat(
                (449, 1, 2, None, "ger_usih(6)", "<=", "100"),
                (449, 3, 3, None, "ger_usih(6)", "<=", "80"),
            ),
            "restricao_eletrica_regra_ativacao": regra((201, "demanda_sin > 82000")),
            "restricao_eletrica_habilita": habilita((449, 201)),
        },
        taken={0},
    )

    assert [
        (record["id"], record["name"], record["expression"])
        for record in outputs.constraints
    ] == [
        (450, "constraint_449_1", "hydro_generation(0) <= 100"),
        (451, "constraint_449_2", "hydro_generation(0) <= 80"),
    ]
    assert [record["active_when"] for record in outputs.constraints] == [
        "(bus_load_value(0) + bus_load_value(1)) > 82000"
    ] * 2


def test_the_treatment_is_the_slack_of_each_record() -> None:
    outputs = _outputs(
        {
            "restricao_eletrica_inequacao": re_inequacao(
                (405, "ger_usih(6)", "<=", "100"),
                (409, "ger_usih(251)", ">=", "10"),
                (413, "ger_usih(6) - ger_usih(251)", "==", "0"),
            ),
            "restricao_eletrica_tratamento_violacao": tratamento_violacao(
                (405, "soft", 2000.0)
            ),
            "restricao_eletrica_tratamento_violacao_periodo": (
                tratamento_violacao_periodo((405, 2, 2, "hard", float("nan")))
            ),
        }
    )

    records = {record["name"]: record for record in outputs.constraints}
    split = [records["constraint_405_1"], records["constraint_405_2"]]
    assert split[0]["expression"] == split[1]["expression"]
    assert [record["slack"] for record in split] == [
        {"enabled": True, "penalty": 2000.0, "penalty_basis": "per_hour"},
        {"enabled": False},
    ]
    assert records["constraint_409"]["slack"] == _DEFAULT_SLACK
    assert " == " in records["constraint_413"]["expression"]
    assert [record["id"] for record in outputs.constraints] == [409, 413, 414, 415]


def test_short_form_cards_build_the_same_outputs() -> None:
    frames = _every_card()
    assert set(SHORT_FORM_ACCESSORS) <= set(frames)

    long_outputs = _build(_model(frames))
    short_outputs = _build(_model(frames, short=True))

    assert short_outputs == long_outputs
    assert long_outputs.constraints
    assert long_outputs.expressions
    assert long_outputs.parameters


def _reversed[K, V](mapping: Mapping[K, V]) -> dict[K, V]:
    return dict(reversed(list(mapping.items())))


def test_the_outputs_do_not_depend_on_insertion_order() -> None:
    forward = _model(_every_card())
    backward = replace(
        forward,
        expressions=_reversed(forward.expressions),
        aliases=_reversed(forward.aliases),
        restrictions=_reversed(forward.restrictions),
        rules=_reversed(forward.rules),
    )
    assert list(backward.expressions) != list(forward.expressions)
    assert list(backward.aliases) != list(forward.aliases)
    assert list(backward.restrictions) != list(forward.restrictions)
    assert list(backward.rules) != list(forward.rules)

    # 403 is taken, so it is renumbered among the split restriction's ids.
    assert _build(backward, taken=dict.fromkeys((403, 0)).keys()) == _build(
        forward, taken={0, 403}
    )


def test_an_undeclared_activation_rule_fails() -> None:
    frames = {
        "restricao_eletrica_inequacao": re_inequacao((101, "ger_usih(6)", "<=", "100")),
        "restricao_eletrica_regra_ativacao": regra((201, "demanda_sin > 82000")),
        "restricao_eletrica_habilita": habilita((101, 999)),
    }
    message = (
        "restriction 101 is assigned activation rule 999, which the file does not "
        "declare"
    )

    with pytest.raises(MalformedElectricalExpression, match=f"^{re.escape(message)}$"):
        _outputs(frames)


@pytest.mark.parametrize(
    ("frames", "cycle"),
    [
        pytest.param(
            {
                "restricao_eletrica_formula": re_formula((1, "re(1) + ger_usih(6)")),
                "restricao_eletrica_limite_formula_periodo_patamar": re_lim_formula(
                    (1, 1, 3, None, -1e31, 100.0)
                ),
            },
            "restriction_1_formula -> restriction_1_formula",
            id="a-formula-reading-itself",
        ),
        pytest.param(
            {
                "expressao_eletrica": expressao(
                    (1, "E", "F + ger_usih(6)"), (2, "F", "E + 1")
                ),
                "restricao_eletrica_inequacao": re_inequacao((101, "E", "<=", "100")),
            },
            "E -> F -> E",
            id="two-expressions",
        ),
        pytest.param(
            {
                "expressao_eletrica": expressao((1, "E", "re(1) + 1")),
                "restricao_eletrica_formula": re_formula((1, "E + ger_usih(6)")),
                "restricao_eletrica_limite_formula_periodo_patamar": re_lim_formula(
                    (1, 1, 3, None, -1e31, 100.0)
                ),
            },
            "E -> restriction_1_formula -> E",
            id="an-expression-and-a-formula",
        ),
        pytest.param(
            {
                "restricao_eletrica_formula": re_formula(
                    (1, "re(2) + ger_usih(6)"), (2, "re(1)")
                ),
                "restricao_eletrica_limite_formula_periodo_patamar": re_lim_formula(
                    (1, 1, 3, None, -1e31, 100.0)
                ),
            },
            "restriction_1_formula -> restriction_2_formula -> restriction_1_formula",
            id="two-formulas",
        ),
    ],
)
def test_a_cycle_fails_naming_it(frames: Frames, cycle: str) -> None:
    message = f"electrical expressions form a cycle: {cycle}"

    with pytest.raises(MalformedElectricalExpression, match=f"^{re.escape(message)}$"):
        _outputs(frames)


def test_a_name_reached_by_two_routes_converts_without_a_false_cycle() -> None:
    outputs = _outputs(
        {
            "expressao_eletrica": expressao((1, "E", "re(1) * 2")),
            "restricao_eletrica_formula": re_formula(
                (1, "ger_usih(6)"), (2, "re(1) + 1")
            ),
            "restricao_eletrica_inequacao": re_inequacao((3, "re(2) + E", "<=", "100")),
        }
    )

    assert [record["expression"] for record in outputs.constraints] == [
        "@restriction_2_formula + @E <= 100"
    ]
    assert outputs.expressions == [
        {"name": "E", "expression": "@restriction_1_formula * 2"},
        {"name": "restriction_1_formula", "expression": "hydro_generation(0)"},
        {"name": "restriction_2_formula", "expression": "@restriction_1_formula + 1"},
    ]


def test_declared_expressions_come_before_the_reference_formulas() -> None:
    outputs = _outputs(
        {
            "expressao_eletrica": expressao((1, "zeta", "ger_usih(6) * 2")),
            "restricao_eletrica_formula": re_formula((5, "ger_usih(251)")),
            "restricao_eletrica_inequacao": re_inequacao(
                (7, "re(5) + zeta", "<=", "100")
            ),
        }
    )

    assert [record["name"] for record in outputs.expressions] == [
        "zeta",
        "restriction_5_formula",
    ]


def test_a_reached_data_only_expression_is_a_parameter_not_an_expression() -> None:
    outputs = _outputs(
        {
            "expressao_eletrica": expressao(
                (1, "north_exchange", "ener_interc(1,4) - ener_interc(4,1)"),
                (2, "margins", "MARGIN + 1"),
            ),
            "alias_eletrico": alias((1, "MARGIN")),
            "alias_eletrico_valor_periodo_patamar": alias_valor((1, 1, 3, None, 300.0)),
            "restricao_eletrica_inequacao": re_inequacao(
                (101, "ger_usih(6) + north_exchange", "<=", "margins")
            ),
        }
    )

    assert [record["expression"] for record in outputs.constraints] == [
        "hydro_generation(0) + @north_exchange <= @margins"
    ]
    assert [record["name"] for record in outputs.expressions] == ["north_exchange"]
    assert [(entry["name"], entry["kind"]) for entry in outputs.parameters] == [
        ("MARGIN", "per_stage_block"),
        ("margins", "expression"),
    ]


@pytest.mark.parametrize(
    ("frames", "label"),
    [
        pytest.param(
            {
                "expressao_eletrica": expressao((1, "E", "ger_usih(99) + 1")),
                "restricao_eletrica_inequacao": re_inequacao((101, "E", "<=", "100")),
            },
            "expression E: ",
            id="expression",
        ),
        pytest.param(
            {
                "restricao_eletrica_inequacao": re_inequacao(
                    (101, "ger_usih(6)", "<=", "100")
                ),
                "restricao_eletrica_regra_ativacao": regra((201, "demanda(9) > 0")),
                "restricao_eletrica_habilita": habilita((101, 201)),
            },
            "activation rule 201: ",
            id="activation-rule",
        ),
    ],
)
def test_a_translation_error_names_the_text_it_is_in(
    frames: Frames, label: str
) -> None:
    with pytest.raises(UnsupportedSourceError, match=f"^{re.escape(label)}"):
        _outputs(frames)


def test_a_restriction_without_cells_keeps_its_formula_expression() -> None:
    outputs = _outputs(
        {
            "restricao_eletrica_formula": re_formula(
                (421, "ger_usih(6) + ger_usih(251)")
            ),
            "restricao_eletrica_inequacao": re_inequacao((430, "re(421)", "<=", "100")),
        }
    )

    assert [record["name"] for record in outputs.constraints] == ["constraint_430"]
    assert outputs.constraints[0]["expression"] == "@restriction_421_formula <= 100"
    assert outputs.expressions == [
        {
            "name": "restriction_421_formula",
            "expression": "hydro_generation(0) + hydro_generation(1)",
        }
    ]


def test_no_source_vocabulary_is_emitted() -> None:
    frames = _every_card()
    deck = " ".join(
        value
        for frame in frames.values()
        for value in frame.to_numpy().ravel()
        if isinstance(value, str)
    )
    assert {match.group() for match in _SOURCE_TERM.finditer(deck)} == {
        "ger_usih",
        "demanda",
        "se(",
        "re(",
    }

    outputs = _outputs(frames)

    emitted = [
        leaf
        for leaf in _leaves([outputs.constraints, outputs.expressions])
        if isinstance(leaf, str)
    ]
    assert "@restriction_402_formula" in " ".join(emitted)
    assert [text for text in emitted if _SOURCE_TERM.search(text)] == []


def _split_and_left_out() -> Frames:
    """401 keeps its code, 449 splits into stage 0 and stage 1, and the
    formula 421 has no limit, so it holds no row."""
    return {
        "restricao_eletrica_formula": re_formula((421, "ger_usih(6)")),
        "restricao_eletrica_inequacao": re_inequacao((401, "ger_usih(6)", "<=", "100")),
        "restricao_eletrica_inequacao_periodo_patamar": re_inequacao_per_pat(
            (449, 1, 1, None, "ger_usih(251)", "<=", "100"),
            (449, 2, 2, None, "ger_usih(251)", "<=", "80"),
        ),
        "restricao_eletrica_horizonte_periodo": horizonte((449, 1, 2)),
    }


def _rows(sink: list[dx.Diagnostic], code: str) -> list[list[object]]:
    """The table rows of the one diagnostic *code* in *sink*."""
    [diagnostic] = [d for d in sink if d.code == code]
    assert diagnostic.table is not None
    return [list(row) for row in diagnostic.table.rows]


def test_split_and_left_out_restrictions_are_listed_once() -> None:
    emitted = emitted_constraints(
        _model(_split_and_left_out()), _CALENDAR, taken=frozenset()
    )

    with dx.collect() as sink:
        report_split_restrictions(emitted)

    [diagnostic] = sink
    assert diagnostic.code == _SPLIT
    assert diagnostic.severity is dx.Severity.INFO
    assert diagnostic.category == "Special constraints"
    assert diagnostic.title.endswith(" (2)")
    assert diagnostic.table is not None
    assert diagnostic.table.columns == [
        "Code",
        "Constraint ID",
        "Constraint name",
        "Cells",
    ]
    assert diagnostic.table.caption == "Stage and block ids are 0-based."
    assert _rows(sink, _SPLIT) == [
        [421, "", "", "none"],
        [449, 450, "constraint_449_1", "stage 0 block 0, stage 0 block 1"],
        [449, 451, "constraint_449_2", "stage 1 block 0, stage 1 block 1"],
    ]


def test_a_renumbered_restriction_is_listed() -> None:
    calendar = [
        replace(stage, block_hours=(84.0, 84.0)) for stage in weekly_calendar(2)
    ]
    model = _model(
        {"restricao_eletrica_inequacao": re_inequacao((601, "ger_usih(6)", "<=", "1"))},
        calendar=calendar,
    )

    with dx.collect() as sink:
        report_split_restrictions(emitted_constraints(model, calendar, taken={601}))

    assert _rows(sink, _SPLIT) == [
        [
            601,
            602,
            "constraint_601",
            "stage 0 block 0, stage 0 block 1, stage 1 block 0, stage 1 block 1",
        ]
    ]


def test_no_diagnostic_when_every_restriction_keeps_its_code() -> None:
    model = _model(
        {
            "restricao_eletrica_inequacao": re_inequacao(
                (101, "ger_usih(6)", "<=", "100")
            ),
            "restricao_eletrica_formula": re_formula((102, "ger_usit(12)")),
            "restricao_eletrica_limite_formula_periodo_patamar": re_lim_formula(
                (102, 1, 3, None, -1e31, 1500.0)
            ),
        }
    )

    with dx.collect() as sink:
        outputs = _build(model)

    assert [record["id"] for record in outputs.constraints] == [101, 102]
    assert _SPLIT not in [diagnostic.code for diagnostic in sink]


def test_the_outputs_report_the_split_restrictions() -> None:
    with dx.collect() as sink:
        _outputs(_split_and_left_out())

    assert [diagnostic.code for diagnostic in sink].count(_SPLIT) == 1


def test_names_read_only_by_left_out_restrictions_are_not_emitted() -> None:
    outputs = _outputs(
        {
            "expressao_eletrica": expressao(
                (1, "E", "ger_usih(251) + A"), (2, "F", "ger_usih(6) + B")
            ),
            "alias_eletrico": alias((1, "A"), (2, "B")),
            "alias_eletrico_valor_periodo_patamar": alias_valor(
                (1, 1, 3, None, 5.0), (2, 1, 3, None, 7.0)
            ),
            "restricao_eletrica_formula": re_formula((421, "E")),
            "restricao_eletrica_inequacao": re_inequacao((401, "F", "<=", "100")),
        }
    )

    assert [record["name"] for record in outputs.constraints] == ["constraint_401"]
    assert outputs.expressions == [
        {"name": "F", "expression": "hydro_generation(0) + @B"}
    ]
    assert [entry["name"] for entry in outputs.parameters] == ["B"]


def test_a_pruned_name_is_in_neither_the_filled_nor_the_omitted_table() -> None:
    model = _model(
        {
            "alias_eletrico": alias((1, "A"), (2, "B"), (3, "C")),
            "alias_eletrico_valor_periodo_patamar": alias_valor(
                (1, 1, 1, None, 5.0), (2, 1, 1, None, 7.0), (3, 1, 3, None, 1.0)
            ),
            "restricao_eletrica_formula": re_formula((421, "ger_usih(251) + A")),
            "restricao_eletrica_inequacao": re_inequacao(
                (401, "ger_usih(6) + B", "<=", "100")
            ),
            "restricao_eletrica_horizonte_periodo": horizonte((401, 1, 1)),
        }
    )

    with dx.collect() as sink:
        outputs = _build(model)

    assert [entry["name"] for entry in outputs.parameters] == ["B"]
    assert _rows(sink, "decomp-electrical-unread-cells-filled") == [
        [
            "B",
            "alias",
            "stage 1 block 0, stage 1 block 1, stage 2 block 0, stage 2 block 1",
        ]
    ]
    assert _rows(sink, "decomp-electrical-unread-omitted") == [["alias", "C"]]
