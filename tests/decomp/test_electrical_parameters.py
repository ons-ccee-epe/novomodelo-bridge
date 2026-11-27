"""Tests for the generic parameters the electrical constraints emit: reached
aliases and data terms as ``per_stage_block`` entries, data-only expressions as
``expression`` entries (tier 1, no ``novomodelo`` import)."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.decomp.converters.electrical_constraints import (
    DataAliases,
    build_electrical_parameters,
    build_generic_outputs,
    is_parameter_expression,
    read_cells,
)
from novomodelo_bridge.decomp.converters.electrical_translate import (
    Translator,
    build_name_table,
    entity_tables,
)
from novomodelo_bridge.decomp.converters.libs_electrical import (
    ElectricalAlias,
    ElectricalExpression,
    ElectricalRestriction,
    LibsElectricalModel,
    PeriodPatamarOverride,
)
from novomodelo_bridge.novomodelo.scalar_parameters import (
    append_scalar_parameters,
    build_scalar_parameters,
)
from tests.decomp.conftest import weekly_calendar

if TYPE_CHECKING:
    from collections.abc import Mapping

_CALENDAR = [replace(stage, block_hours=(84.0, 84.0)) for stage in weekly_calendar(3)]

_STUDY = [(stage, block) for stage in range(3) for block in range(2)]

_TABLES = entity_tables({"hydros": {285: 20, 287: 21}, "submarkets": {1: 0}}, {}, [])

_LHS = "ger_usih(285)"

_FILLED = "decomp-electrical-unread-cells-filled"
_OMITTED = "decomp-electrical-unread-omitted"

AliasValues = dict[tuple[int, int | None], float]

#: A value for every block of every study stage.
_EVERY_STAGE: AliasValues = {(0, None): 1.0, (1, None): 1.0, (2, None): 1.0}

#: Alias ``A`` of the coverage criteria: block 1 of stage 1 overrides the stage.
_PARTIAL_A: AliasValues = {(0, None): 1.0, (1, None): 1.0, (1, 2): 2.0}


def _inequality(
    code: int,
    rhs: str,
    *,
    overrides: tuple[PeriodPatamarOverride, ...] = (),
    stage_start: int | None = None,
    stage_end: int | None = None,
    lhs: str = _LHS,
) -> ElectricalRestriction:
    return ElectricalRestriction(
        code=code,
        lhs=lhs,
        operator="<=",
        rhs=rhs,
        is_formula=False,
        limits={},
        overrides=overrides,
        stage_start=stage_start,
        stage_end=stage_end,
    )


def _override(stage_start: int, stage_end: int, rhs: str) -> PeriodPatamarOverride:
    return PeriodPatamarOverride(stage_start, stage_end, None, _LHS, "<=", rhs)


def _formula(
    code: int,
    lhs: str,
    *,
    stage_start: int | None = None,
    stage_end: int | None = None,
) -> ElectricalRestriction:
    """A formula restriction without ``limits``: it holds no row of its own."""
    return ElectricalRestriction(
        code=code,
        lhs=lhs,
        operator=None,
        rhs=None,
        is_formula=True,
        limits={},
        overrides=(),
        stage_start=stage_start,
        stage_end=stage_end,
    )


def _model(
    *restrictions: ElectricalRestriction,
    aliases: Mapping[str, AliasValues] | None = None,
    expressions: Mapping[str, str] | None = None,
) -> LibsElectricalModel:
    """Codes follow the mapping order, from 1."""
    return LibsElectricalModel(
        expressions={
            name: ElectricalExpression(code, name, formula)
            for code, (name, formula) in enumerate((expressions or {}).items(), 1)
        },
        aliases={
            name: ElectricalAlias(code, name, values)
            for code, (name, values) in enumerate((aliases or {}).items(), 1)
        },
        restrictions={r.code: r for r in restrictions},
        rules={},
    )


def _build(
    model: LibsElectricalModel, data_terms: DataAliases | None = None
) -> tuple[list[dict], list[dx.Diagnostic]]:
    """The entries and only the diagnostics :func:`build_electrical_parameters`
    emits."""
    names = build_name_table(model, reserved=frozenset())
    translator = Translator(model, _TABLES, names)
    reads = read_cells(model, _CALENDAR)
    with dx.collect() as sink:
        entries = build_electrical_parameters(
            model,
            _CALENDAR,
            translator,
            names,
            data_terms or {},
            reads,
            kept=reads.keys(),
        )
    return entries, sink


def _block_values(values: Mapping[tuple[int, int], float]) -> list[list[float]]:
    return [[stage, block, values.get((stage, block), 0.0)] for stage, block in _STUDY]


def _rows(sink: list[dx.Diagnostic], code: str) -> list[list[object]]:
    """The table rows of the one diagnostic *code* in *sink*."""
    [diagnostic] = [d for d in sink if d.code == code]
    assert diagnostic.table is not None
    return [list(row) for row in diagnostic.table.rows]


def _entry(entries: list[dict], name: str) -> dict:
    [entry] = [e for e in entries if e["name"] == name]
    return entry


def test_a_partial_alias_read_only_where_it_has_values_is_filled_and_warned() -> None:
    model = _model(
        _inequality(701, "A", stage_start=0, stage_end=1), aliases={"A": _PARTIAL_A}
    )

    entries, sink = _build(model)

    assert entries == [
        {
            "name": "A",
            "kind": "per_stage_block",
            "block_values": [
                [0, 0, 1.0],
                [0, 1, 1.0],
                [1, 0, 1.0],
                [1, 1, 2.0],
                [2, 0, 0.0],
                [2, 1, 0.0],
            ],
        }
    ]
    assert list(entries[0]) == ["name", "kind", "block_values"]
    [warning] = sink
    assert warning.code == _FILLED
    assert warning.severity is dx.Severity.INFO
    assert warning.category == "Special constraints"
    assert warning.table is not None
    assert warning.table.columns == ["Parameter", "Kind", "Filled cells"]
    assert _rows(sink, _FILLED) == [["A", "alias", "stage 2 block 0, stage 2 block 1"]]


def test_an_alias_missing_in_a_read_cell_fails_naming_the_missing_read_cells() -> None:
    model = _model(
        _inequality(701, "A", stage_start=0, stage_end=2), aliases={"A": _PARTIAL_A}
    )

    with pytest.raises(ValueError, match=r"\bA\b") as error:
        _build(model)

    message = str(error.value)
    assert "stage 2 block 0, stage 2 block 1" in message
    assert "stage 1" not in message


def test_an_alias_missing_only_where_an_override_wins_is_accepted() -> None:
    model = _model(
        _inequality(702, "A", overrides=(_override(1, 1, "B"),)),
        aliases={"A": {(0, None): 1.0, (2, None): 3.0}, "B": _EVERY_STAGE},
    )

    entries, sink = _build(model)

    assert _entry(entries, "A")["block_values"] == _block_values(
        {(0, 0): 1.0, (0, 1): 1.0, (2, 0): 3.0, (2, 1): 3.0}
    )
    assert _entry(entries, "B")["block_values"] == _block_values(
        dict.fromkeys(_STUDY, 1.0)
    )
    assert [d.code for d in sink] == [_FILLED]
    assert _rows(sink, _FILLED) == [["A", "alias", "stage 1 block 0, stage 1 block 1"]]


def test_a_reachable_alias_that_no_cell_reads_is_emitted_filled() -> None:
    model = _model(_formula(406, "ger_usih(287) + E"), aliases={"E": {(0, None): 4.0}})

    entries, sink = _build(model)

    assert entries == [
        {
            "name": "E",
            "kind": "per_stage_block",
            "block_values": _block_values({(0, 0): 4.0, (0, 1): 4.0}),
        }
    ]
    assert _rows(sink, _FILLED) == [
        [
            "E",
            "alias",
            "stage 1 block 0, stage 1 block 1, stage 2 block 0, stage 2 block 1",
        ]
    ]


def test_a_block_value_does_not_cover_the_other_blocks_of_its_stage() -> None:
    model = _model(
        _inequality(701, "A", stage_start=0, stage_end=1),
        aliases={"A": {(0, None): 1.0, (1, 2): 2.0}},
    )

    with pytest.raises(ValueError, match="stage 1 block 0") as error:
        _build(model)

    assert "stage 1 block 1" not in str(error.value)


def test_an_alias_read_through_a_reference_is_covered_where_the_citing_relation_reads() -> (
    None
):
    model = _model(
        _formula(701, "ger_usih(285) + K", stage_start=0, stage_end=0),
        _inequality(801, "10", lhs="re(701)"),
        aliases={"K": {(0, None): 1.0}},
    )

    with pytest.raises(ValueError, match=r"\bK\b") as error:
        _build(model)

    assert "stage 1 block 0, stage 1 block 1, stage 2 block 0, stage 2 block 1" in str(
        error.value
    )
    assert "stage 0" not in str(error.value)


def test_unreachable_aliases_and_expressions_are_omitted_and_reported() -> None:
    model = _model(
        _inequality(701, "A"),
        aliases={"A": _EVERY_STAGE, "B": _EVERY_STAGE},
        expressions={"Z": "B + 1", "W": "val_demanda(1)"},
    )

    entries, sink = _build(model)

    assert [entry["name"] for entry in entries] == ["A"]
    [info] = sink
    assert info.code == _OMITTED
    assert info.severity is dx.Severity.INFO
    assert info.category == "Special constraints"
    assert info.table is not None
    assert info.table.columns == ["Kind", "Name"]
    assert _rows(sink, _OMITTED) == [
        ["alias", "B"],
        ["expression", "Z"],
        ["expression", "W"],
    ]


def test_an_unassigned_activation_rule_is_listed_once_as_omitted() -> None:
    model = replace(
        _model(replace(_inequality(701, "10"), habilita=501), _inequality(702, "20")),
        rules={501: "demanda_sin > 0", 502: "demanda_sin > 1"},
    )

    with dx.collect() as sink:
        build_generic_outputs(
            model,
            _CALENDAR,
            _TABLES,
            {},
            reserved=frozenset(),
            taken=frozenset(),
            default_penalty=1000.0,
        )

    assert _rows(sink, _OMITTED) == [["activation rule", "502"]]


def test_only_data_expressions_over_parameters_become_parameters() -> None:
    model = _model(
        _inequality(801, "RNE + CAP + LOADX"),
        aliases={"C_AM": _EVERY_STAGE},
        expressions={
            "FNESE": "ger_usih(285)",
            "RNE": "FNESE + C_AM",
            "CAP": "0.5*(C_AM + carga_ande)",
            "LOADX": "0.05*(val_demanda(1) + val_demanda(2))",
        },
    )

    entries, sink = _build(model, {"carga_ande": dict.fromkeys(_STUDY, 7.0)})

    assert [e for e in entries if e["kind"] == "expression"] == [
        {
            "name": "CAP",
            "kind": "expression",
            "expression": "0.5 * (@C_AM + @carga_ande)",
        }
    ]
    assert list(entries[-1]) == ["name", "kind", "expression"]
    assert [(e["name"], e["kind"]) for e in entries] == [
        ("C_AM", "per_stage_block"),
        ("carga_ande", "per_stage_block"),
        ("CAP", "expression"),
    ]
    assert is_parameter_expression("CAP", model)
    for name in ("RNE", "FNESE", "LOADX"):
        assert not is_parameter_expression(name, model), name
    assert sink == []


@pytest.mark.parametrize(
    ("formula", "is_parameter"),
    [
        ("-(1 + 2) / 3 * A - +4", True),
        ("A + carga_ande() + peq_SE_PCHgd_SE()", True),
        ("A + demanda_sin", False),
        ("A + val_demanda_sin", False),
        ("A + carga_ande(1)", False),
        ("A + A()", False),
        ("se(A > 0, A, 1)", False),
        ("re(1)", False),
        ("disp_usih(285)", False),
        ("A + UNKNOWN", False),
    ],
)
def test_only_arithmetic_over_data_names_is_a_parameter(
    formula: str, is_parameter: bool
) -> None:
    model = _model(aliases={"A": _EVERY_STAGE}, expressions={"X": formula})

    assert is_parameter_expression("X", model) is is_parameter


def test_a_data_term_needs_a_source_only_where_it_is_read() -> None:
    read = _model(_inequality(801, "peq_SE_PCHgd_SE"))
    series = {cell: 10.0 + index for index, cell in enumerate(_STUDY)}

    with pytest.raises(ValueError, match=r"\bpeq_SE_PCHgd_SE\b"):
        _build(read)

    sourced, sink = _build(
        read, {"peq_SE_PCHgd_SE": series, "peq_N_PCH_NE": dict.fromkeys(_STUDY, 1.0)}
    )
    assert sourced == [
        {
            "name": "peq_SE_PCHgd_SE",
            "kind": "per_stage_block",
            "block_values": _block_values(series),
        }
    ]
    assert sink == []

    unread = _model(
        _inequality(
            801,
            "10",
            overrides=(_override(1, 2, "peq_SE_PCHgd_SE"),),
            stage_start=0,
            stage_end=0,
        )
    )
    filled, sink = _build(unread)
    assert filled == [
        {
            "name": "peq_SE_PCHgd_SE",
            "kind": "per_stage_block",
            "block_values": _block_values({}),
        }
    ]
    assert _rows(sink, _FILLED) == [
        [
            "peq_SE_PCHgd_SE",
            "data",
            ", ".join(f"stage {s} block {b}" for s, b in _STUDY),
        ]
    ]


def test_append_keeps_existing_ids_and_numbers_the_new_entries() -> None:
    doc = build_scalar_parameters([4], per_stage={"share": [0.5, 0.5, 0.5]})
    original = copy.deepcopy(doc)
    entries = [
        {"name": "A", "kind": "per_stage_block", "block_values": [[0, 0, 1.0]]},
        {"name": "CAP", "kind": "expression", "expression": "2 * @A"},
    ]

    appended = append_scalar_parameters(doc, entries)

    assert doc == original
    assert appended["$schema"] == original["$schema"]
    assert appended["scalar_parameters"][:8] == original["scalar_parameters"]
    assert appended["scalar_parameters"][8:] == [
        {"id": 8, **entries[0]},
        {"id": 9, **entries[1]},
    ]
    assert [list(entry)[0] for entry in appended["scalar_parameters"]] == ["id"] * 10
    assert [entry["id"] for entry in appended["scalar_parameters"]] == list(range(10))


def test_a_block_value_wins_over_the_stage_value() -> None:
    model = _model(
        _inequality(701, "A"),
        aliases={
            "A": {
                (0, 1): 5.0,
                (0, None): 1.0,
                (1, None): 2.0,
                (2, None): 3.0,
                (2, 2): 7.0,
            }
        },
    )

    entries, sink = _build(model)

    assert entries[0]["block_values"] == [
        [0, 0, 5.0],
        [0, 1, 1.0],
        [1, 0, 2.0],
        [1, 1, 2.0],
        [2, 0, 3.0],
        [2, 1, 7.0],
    ]
    assert sink == []


def test_an_alias_patamar_outside_the_stage_blocks_fails() -> None:
    values: AliasValues = {**_EVERY_STAGE, (1, 3): 2.0}

    with pytest.raises(ValueError, match=r"alias A stage 1: patamar 3 outside 1\.\.2"):
        _build(_model(_inequality(701, "A"), aliases={"A": values}))

    entries, sink = _build(
        _model(_inequality(701, "B"), aliases={"A": values, "B": _EVERY_STAGE})
    )
    assert [entry["name"] for entry in entries] == ["B"]
    assert _rows(sink, _OMITTED) == [["alias", "A"]]


def test_an_expression_cycle_fails_naming_the_cycle() -> None:
    model = _model(
        _inequality(701, "X"),
        aliases={"A": _EVERY_STAGE},
        expressions={"X": "Y + A", "Y": "ger_usih(285) * X"},
    )

    with pytest.raises(ValueError, match="X -> Y -> X"):
        is_parameter_expression("X", model)
    with pytest.raises(ValueError, match="Y -> X -> Y"):
        is_parameter_expression("Y", model)
    with pytest.raises(ValueError, match="cycle: X -> Y -> X"):
        _build(model)


def test_a_shadowed_alias_is_not_reported_as_omitted() -> None:
    model = _model(
        _inequality(701, "X"),
        aliases={"X": _EVERY_STAGE, "Y": _EVERY_STAGE},
        expressions={"X": "2"},
    )

    entries, sink = _build(model)

    assert entries == [{"name": "X", "kind": "expression", "expression": "2"}]
    assert _rows(sink, _OMITTED) == [["alias", "Y"]]


def _reversed[K, V](mapping: Mapping[K, V]) -> dict[K, V]:
    return dict(reversed(list(mapping.items())))


def test_parameters_do_not_depend_on_input_insertion_order() -> None:
    model = _model(
        _inequality(701, "x + CAP + peq_B_SE", stage_start=0, stage_end=0),
        _inequality(702, "A + LOADX + peq_A_SE"),
        aliases={
            "x": {(0, None): 1.0},
            "A": {(0, 2): 9.0, **_EVERY_STAGE},
            "D": _EVERY_STAGE,
            "C": _EVERY_STAGE,
        },
        expressions={
            "CAP": "2 * A + carga_ande",
            "LOADX": "val_demanda(1)",
            "Z": "1",
            "W": "2",
        },
    )
    data_terms = {
        "peq_B_SE": dict.fromkeys(_STUDY, 3.0),
        "carga_ande": {(0, 0): 4.0, (0, 1): 4.0},
        "peq_A_SE": dict.fromkeys(_STUDY, 5.0),
    }
    shuffled = LibsElectricalModel(
        expressions=_reversed(model.expressions),
        aliases={
            name: replace(alias, values=_reversed(alias.values))
            for name, alias in _reversed(model.aliases).items()
        },
        restrictions=_reversed(model.restrictions),
        rules={},
    )

    forward = _build(model, data_terms)
    backward = _build(shuffled, _reversed(data_terms))

    assert backward == forward
    entries, sink = forward
    assert [entry["name"] for entry in entries] == [
        "x",
        "A",
        "carga_ande",
        "peq_A_SE",
        "peq_B_SE",
        "CAP",
    ]
    assert _entry(entries, "A")["block_values"][1] == [0, 1, 9.0]
    unread = "stage 1 block 0, stage 1 block 1, stage 2 block 0, stage 2 block 1"
    assert _rows(sink, _FILLED) == [
        ["carga_ande", "data", unread],
        ["x", "alias", unread],
    ]
    assert _rows(sink, _OMITTED) == [
        ["alias", "D"],
        ["alias", "C"],
        ["expression", "Z"],
        ["expression", "W"],
    ]
