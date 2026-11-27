"""Tier-1 tests for the translation of the source model's electrical texts into
novomodelo's neutral vocabulary, ids and names."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.electrical_expression import MalformedElectricalExpression
from novomodelo_bridge.core.errors import UnsupportedSourceError
from novomodelo_bridge.decomp.converters.contracts import Contract
from novomodelo_bridge.decomp.converters.electrical_constraints import (
    GenericOutputs,
    build_code_map,
    build_generic_outputs,
)
from novomodelo_bridge.decomp.converters.electrical_translate import (
    EntityTables,
    NameTable,
    Translator,
    build_name_table,
    entity_tables,
    translate_call,
)
from novomodelo_bridge.decomp.converters.libs_electrical import (
    ElectricalAlias,
    ElectricalExpression,
    ElectricalRestriction,
    FormulaOverride,
    LibsElectricalModel,
    PeriodPatamarOverride,
)
from novomodelo_bridge.decomp.converters.libs_electrical_parser import (
    Call,
    Keyword,
    Number,
    parse_expression,
    render,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from tests.decomp.conftest import weekly_calendar

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_CONTEXT = "restriction 405 formula"


def _contract(contract_id: int, kind: str, numero: int) -> Contract:
    return Contract(
        id=contract_id,
        kind=kind,
        numero=numero,
        name=f"C{numero}",
        bus_code=1,
        stages=[],
    )


_TABLES = entity_tables(
    {
        "hydros": {261: 14, 66: 3, 285: 20, 287: 21},
        "thermals": {7: 2, 90: 5},
        "ncs_pee": {11: 0},
        "submarkets": {1: 0, 3: 2, 6: 4},
        "contracts": {4: 1, 8: 2},
    },
    {66: {1: 0, 2: 1}},
    [_contract(1, "import", 4), _contract(2, "export", 8)],
)


def _source_call(text: str) -> Call:
    node = parse_expression(text)
    assert isinstance(node, Call)
    return node


def _translate(
    text: str, tables: EntityTables = _TABLES, *, in_condition: bool = True
) -> str:
    return render(
        translate_call(
            _source_call(text), tables, context=_CONTEXT, in_condition=in_condition
        )
    )


_SYSTEM_LOAD = "(bus_load_value(0) + bus_load_value(2) + bus_load_value(4))"


@pytest.mark.parametrize(
    ("source", "neutral"),
    [
        ("ger_usih(261)", "hydro_generation(14)"),
        ("ger_usih(261,2)", "hydro_generation(14, 2)"),
        ("ger_usit(90)", "thermal_generation(5)"),
        ("ger_pee(11)", "non_controllable_generation(0)"),
        ("disp_usih(261)", "hydro_availability(14)"),
        ("demanda(6)", "bus_load_value(4)"),
        ("val_demanda(1,1)", "bus_load_value(0, 1)"),
        ("demanda_sin()", _SYSTEM_LOAD),
        (
            "val_demanda_sin(1)",
            "(bus_load_value(0, 1) + bus_load_value(2, 1) + bus_load_value(4, 1))",
        ),
        ("ener_comerc(4)", "contract_import(1)"),
        ("ener_comerc(8)", "contract_export(2)"),
        ("ener_interc(3,1)", "line_flow(source_bus=2, target_bus=0)"),
        ("ger_conjh(66,2)", "hydro_generation(3, unit_group=1)"),
        ("ger_conjh(66,1,2)", "hydro_generation(3, 2, unit_group=0)"),
    ],
)
def test_each_source_term_becomes_its_neutral_term(source: str, neutral: str) -> None:
    assert _translate(source) == neutral


def test_pair_and_unit_group_arguments_are_keyword_nodes() -> None:
    line = translate_call(
        _source_call("ener_interc(3,1)"), _TABLES, context=_CONTEXT, in_condition=False
    )
    group = translate_call(
        _source_call("ger_conjh(66,1,2)"), _TABLES, context=_CONTEXT, in_condition=False
    )

    assert line == Call(
        "line_flow",
        (Keyword("source_bus", Number("2")), Keyword("target_bus", Number("0"))),
    )
    assert group == Call(
        "hydro_generation",
        (Number("3"), Number("2"), Keyword("unit_group", Number("0"))),
    )


@pytest.mark.parametrize(
    ("source", "neutral"),
    [
        ("ger_usih(261, 1E0)", "hydro_generation(14, 1E0)"),
        ("ener_comerc(8, 0)", "contract_export(2, 0)"),
        (
            "val_demanda_sin(03)",
            "(bus_load_value(0, 03) + bus_load_value(2, 03) + bus_load_value(4, 03))",
        ),
        ("ger_conjh(66, 2, 1.0)", "hydro_generation(3, 1.0, unit_group=1)"),
    ],
)
def test_block_arguments_pass_verbatim(source: str, neutral: str) -> None:
    assert _translate(source, in_condition=False) == neutral


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("ger_usih(999)", "ger_usih(999): hydro code 999 is not in the converted case"),
        (
            "ener_comerc(5)",
            "ener_comerc(5): contract number 5 is not in the converted case; a "
            "number declared by more than one contract cannot be translated",
        ),
        (
            "ger_conjh(66,3)",
            "ger_conjh(66, 3): hydro plant 66 has no machine set 3 in the "
            "converted case",
        ),
        (
            "ger_conjh(261,1)",
            "ger_conjh(261, 1): hydro plant 261 has no machine set 1 in the "
            "converted case",
        ),
        (
            "ger_conjh(67,1)",
            "ger_conjh(67, 1): hydro code 67 is not in the converted case",
        ),
        ("ger_usit(8)", "ger_usit(8): thermal code 8 is not in the converted case"),
        (
            "ger_pee(12)",
            "ger_pee(12): renewable park code 12 is not in the converted case",
        ),
        ("demanda(2)", "demanda(2): submarket code 2 is not in the converted case"),
        (
            "val_demanda(2)",
            "val_demanda(2): submarket code 2 is not in the converted case",
        ),
        (
            "ener_interc(3, 9)",
            "ener_interc(3, 9): submarket code 9 is not in the converted case",
        ),
    ],
)
def test_unknown_codes_fail_naming_the_context_and_the_call(
    source: str, message: str
) -> None:
    with pytest.raises(UnsupportedSourceError) as caught:
        _translate(source)

    assert str(caught.value) == f"{_CONTEXT}: {message}"


@pytest.mark.parametrize(
    ("source", "form"),
    [
        ("ener_interc(1)", "ener_interc(from, to)"),
        ("ener_interc(1,3,2)", "ener_interc(from, to)"),
        ("ger_conjh(66)", "ger_conjh(plant, machine_set[, block])"),
        ("ger_conjh(66,1,2,3)", "ger_conjh(plant, machine_set[, block])"),
        ("ger_conjh(66,x)", "ger_conjh(plant, machine_set[, block])"),
        ("ger_usih(1.5)", "ger_usih(code[, block])"),
        ("ger_usih(x)", "ger_usih(code[, block])"),
        ("ger_usih(-1)", "ger_usih(code[, block])"),
        ("ger_usih(1E2)", "ger_usih(code[, block])"),
        ("ger_usih()", "ger_usih(code[, block])"),
        ("ger_usih(261,1,2)", "ger_usih(code[, block])"),
        ("ger_usih(261,k)", "ger_usih(code[, block])"),
        ("ger_usih(261,-1)", "ger_usih(code[, block])"),
        ("ener_comerc()", "ener_comerc(contract[, block])"),
        ("demanda_sin(1,2)", "demanda_sin([block])"),
        ("demanda_sin(k)", "demanda_sin([block])"),
        ("val_demanda_sin(1,2)", "val_demanda_sin([block])"),
        ("val_demanda_sin(k)", "val_demanda_sin([block])"),
        ("demanda(1.5)", "demanda(code[, block])"),
        ("val_demanda(x)", "val_demanda(code[, block])"),
    ],
)
def test_malformed_calls_fail(source: str, form: str) -> None:
    with pytest.raises(MalformedElectricalExpression) as caught:
        _translate(source)

    message = str(caught.value)
    assert message.startswith(f"{_CONTEXT}: {render(_source_call(source))}: ")
    assert message.endswith(f"; expected {form}")


@pytest.mark.parametrize("source", ["foo(1)", "hydro_generation(14)"])
def test_an_unknown_function_is_malformed(source: str) -> None:
    name = _source_call(source).name

    with pytest.raises(MalformedElectricalExpression) as caught:
        _translate(source)

    assert str(caught.value).startswith(f"{_CONTEXT}: {source}: ")
    assert repr(name) in str(caught.value)


_ID_MAP = DecompIdMap(
    bus_codes=(1, 2),
    bus_names=("SE", "S"),
    hydro_codes=(66, 285),
    thermal_codes=(7,),
)


def test_gnl_thermal_is_a_thermal_generation() -> None:
    tables = entity_tables(build_code_map(_ID_MAP, {86: 5}, {}, []), {}, [])

    assert _translate("ger_usit(86)", tables) == "thermal_generation(5)"
    assert _translate("ger_usit(7)", tables) == "thermal_generation(0)"


def test_submarket_code_6_is_the_transhipment_bus() -> None:
    tables = entity_tables(build_code_map(_ID_MAP, {}, {}, []), {}, [])

    assert _ID_MAP.transhipment_bus_id == 2
    assert _translate("demanda(6)", tables) == "bus_load_value(2)"
    assert (
        _translate("val_demanda_sin()", tables)
        == "(bus_load_value(0) + bus_load_value(1) + bus_load_value(2))"
    )
    assert (
        _translate("ener_interc(6,1)", tables)
        == "line_flow(source_bus=2, target_bus=0)"
    )


def _model(
    *,
    expressions: Sequence[tuple[str, str]] = (),
    aliases: Sequence[str] = (),
    restrictions: Sequence[ElectricalRestriction] = (),
    rules: Sequence[tuple[int, str]] = (),
) -> LibsElectricalModel:
    return LibsElectricalModel(
        expressions={
            name: ElectricalExpression(code=code, name=name, formula=formula)
            for code, (name, formula) in enumerate(expressions, start=1)
        },
        aliases={
            name: ElectricalAlias(code=code, name=name, values={})
            for code, name in enumerate(aliases, start=1)
        },
        restrictions={restriction.code: restriction for restriction in restrictions},
        rules=dict(rules),
    )


def _restriction(
    code: int,
    lhs: str | None,
    operator: str | None = None,
    rhs: str | None = None,
    *,
    overrides: Sequence[PeriodPatamarOverride] = (),
    formula_overrides: Sequence[FormulaOverride] = (),
) -> ElectricalRestriction:
    return ElectricalRestriction(
        code=code,
        lhs=lhs,
        operator=operator,
        rhs=rhs,
        is_formula=operator is None and not overrides,
        limits={},
        overrides=tuple(overrides),
        formula_overrides=tuple(formula_overrides),
    )


def _names_and_warnings(
    model: LibsElectricalModel, reserved: frozenset[str] = frozenset()
) -> tuple[NameTable, list[dx.Diagnostic]]:
    with dx.collect() as sink:
        names = build_name_table(model, reserved=reserved)
    return names, sink


def _translator(
    model: LibsElectricalModel, reserved: frozenset[str] = frozenset()
) -> Translator:
    return Translator(model, _TABLES, build_name_table(model, reserved=reserved))


_REFERRING_LHS = "val_demanda(1) - re(701) - FNESE"
_REFERRING_RHS = "se(demanda_sin <= 82000, carga_ande + peq_SE_PCHgd_SE, C_AM_cMMGD)"
_REFERRING_MODEL = _model(
    expressions=[("FNESE", "ger_usih(285) - ger_usih(287)")],
    aliases=["C_AM_cMMGD"],
    restrictions=[
        _restriction(701, "ger_usih(285) + ger_usih(287)"),
        _restriction(801, _REFERRING_LHS, "<=", _REFERRING_RHS),
    ],
)


def test_names_references_and_conditionals_translate() -> None:
    with dx.collect() as sink:
        translator = _translator(_REFERRING_MODEL)

    assert (
        translator.relation(_REFERRING_LHS, "<=", _REFERRING_RHS, context=_CONTEXT)
        == "bus_load_value(0) - @restriction_701_formula - @FNESE <= "
        f"if({_SYSTEM_LOAD} <= 82000, @carga_ande + @peq_SE_PCHgd_SE, @C_AM_cMMGD)"
    )
    assert translator.names.reference(701) == "restriction_701_formula"
    assert (
        translator.formula("ger_usih(285) + ger_usih(287)", context=_CONTEXT)
        == "hydro_generation(20) + hydro_generation(21)"
    )
    assert sink == []


def test_sanitization_is_deterministic_and_warned() -> None:
    reserved = frozenset({"rho_eq_h0"})
    forward = _names_and_warnings(
        _model(expressions=[("A-B", "1"), ("A_B", "2")], aliases=["rho_eq_h0"]),
        reserved,
    )
    backward = _names_and_warnings(
        _model(expressions=[("A_B", "2"), ("A-B", "1")], aliases=["rho_eq_h0"]),
        reserved,
    )

    assert forward == backward
    names, [warning] = forward
    assert names.alias("rho_eq_h0") == "rho_eq_h0_2"
    assert names.expression("A-B") == "A_B_2"
    assert names.expression("A_B") == "A_B"
    assert warning.code == "decomp-electrical-name-sanitized"
    assert warning.severity is dx.Severity.INFO
    assert warning.category == "Special constraints"
    assert warning.table is not None
    assert warning.table.columns == ["Name", "Kind", "Emitted as", "Reason"]
    assert warning.table.rows == [
        ["A-B", "expression", "A_B_2", "invalid identifier"],
        ["rho_eq_h0", "alias", "rho_eq_h0_2", "collides with rho_eq_h0"],
    ]


def test_sanitized_names_take_suffixes_by_kind_then_by_name() -> None:
    model = _model(
        expressions=[("A.B", "1"), ("A-B", "1")],
        aliases=["peq-A-SE", "A+B"],
        restrictions=[_restriction(1, "peq_A_SE")],
    )

    names, [warning] = _names_and_warnings(model, frozenset({"A_B", "peq_A_SE"}))

    assert warning.table is not None
    assert warning.table.rows == [
        ["A-B", "expression", "A_B_2", "invalid identifier"],
        ["A.B", "expression", "A_B_3", "invalid identifier"],
        ["A+B", "alias", "A_B_4", "invalid identifier"],
        ["peq-A-SE", "alias", "peq_A_SE_2", "invalid identifier"],
        ["peq_A_SE", "data term", "peq_A_SE_3", "collides with peq_A_SE"],
    ]
    assert names.data("peq_A_SE") == "peq_A_SE_3"


@pytest.mark.parametrize(
    ("name", "emitted"),
    [("1x", "_1x"), ("x y", "x_y"), ("²x", "_²x"), ("C.A-M", "C_A_M")],
)
def test_an_invalid_name_becomes_an_identifier(name: str, emitted: str) -> None:
    names, [warning] = _names_and_warnings(_model(aliases=[name]))

    assert names.alias(name) == emitted
    assert warning.code == "decomp-electrical-name-sanitized"


def test_valid_names_pass_verbatim_without_a_warning() -> None:
    model = _model(
        expressions=[("Ação", "1"), ("_x1", "2")],
        aliases=["Δp", "C_AM_cMMGD"],
        restrictions=[_restriction(1, "peq_N_PCHgd_SE + carga_ande")],
    )

    names, warnings = _names_and_warnings(model, frozenset({"rho_eq_h0"}))

    assert [names.expression("Ação"), names.expression("_x1")] == ["Ação", "_x1"]
    assert [names.alias("Δp"), names.alias("C_AM_cMMGD")] == ["Δp", "C_AM_cMMGD"]
    assert [names.data("peq_N_PCHgd_SE"), names.data("carga_ande")] == [
        "peq_N_PCHgd_SE",
        "carga_ande",
    ]
    assert warnings == []


def test_a_deck_name_colliding_with_a_generated_name_is_sanitized() -> None:
    model = _model(
        expressions=[("restriction_701_formula", "1")],
        restrictions=[
            _restriction(701, "ger_usih(285)"),
            _restriction(801, "re(701) + restriction_701_formula"),
        ],
    )

    with dx.collect() as sink:
        translator = _translator(model)

    assert (
        translator.formula("re(701) + restriction_701_formula", context=_CONTEXT)
        == "@restriction_701_formula + @restriction_701_formula_2"
    )
    [warning] = sink
    assert warning.table is not None
    assert warning.table.rows == [
        [
            "restriction_701_formula",
            "expression",
            "restriction_701_formula_2",
            "collides with restriction_701_formula",
        ]
    ]


def test_a_generated_name_another_conversion_declares_fails() -> None:
    with pytest.raises(ValueError, match=r"\['restriction_701_formula'\]"):
        build_name_table(
            _REFERRING_MODEL, reserved=frozenset({"restriction_701_formula"})
        )


def test_every_text_of_the_model_is_read_for_citations() -> None:
    model = _model(
        expressions=[("E", "re(1) + peq_E_SE")],
        restrictions=[
            *(_restriction(code, "1") for code in range(1, 6)),
            _restriction(6, "ger_usih(285)", "<=", "re(2) + peq_R_SE"),
            _restriction(
                7,
                None,
                overrides=[
                    PeriodPatamarOverride(0, 0, None, "re(3)", "<=", "peq_O_SE")
                ],
            ),
            _restriction(
                8,
                "1",
                formula_overrides=[FormulaOverride(0, 0, 1, "re(4) + peq_F_SE")],
            ),
        ],
        rules=[(1, "re(5) > peq_C_SE")],
    )

    names, _ = _names_and_warnings(model)

    assert [names.reference(code) for code in range(1, 6)] == [
        f"restriction_{code}_formula" for code in range(1, 6)
    ]
    assert [names.data(f"peq_{series}_SE") for series in "EROFC"] == [
        f"peq_{series}_SE" for series in "EROFC"
    ]


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (
            _model(expressions=[("E", "1 +")]),
            "expression E: electrical expression '1 +', position 3: expected an "
            "operand but the text ends",
        ),
        (
            _model(restrictions=[_restriction(9, "1", "<=", "x @ y")]),
            "restriction 9: electrical expression 'x @ y', position 2: unexpected "
            "character '@'",
        ),
        (
            _model(rules=[(3, "demanda_sin")]),
            "activation rule 3: electrical expression 'demanda_sin', position 11: "
            "expected a comparison operator (<, <=, >, >=, ==) but the text ends",
        ),
    ],
)
def test_a_text_that_does_not_parse_names_where_it_is(
    model: LibsElectricalModel, message: str
) -> None:
    with pytest.raises(ValueError) as caught:
        build_name_table(model, reserved=frozenset())

    assert str(caught.value) == message


_OVERRIDE_ONLY_MODEL = _model(
    restrictions=[
        _restriction(
            402,
            None,
            overrides=[PeriodPatamarOverride(0, 0, None, "ger_usih(285)", "<=", "1")],
        )
    ]
)

_NOT_A_REFERENCE = "expected re(code), one non-negative integer restriction code"


@pytest.mark.parametrize(
    ("translate", "error", "message"),
    [
        (
            lambda t: t.formula("ZZTOP + ger_usih(285)", context=_CONTEXT),
            MalformedElectricalExpression,
            "electrical expression 'ZZTOP + ger_usih(285)': unknown name 'ZZTOP'; "
            "it is not a declared expression or alias, a load term or a data term",
        ),
        (
            lambda t: t.condition("foo(1) > 2", context=_CONTEXT),
            MalformedElectricalExpression,
            "electrical expression 'foo(1) > 2': foo(1): unknown function 'foo'",
        ),
        (
            lambda t: t.formula("re(999)", context=_CONTEXT),
            MalformedElectricalExpression,
            "electrical expression 're(999)': re(999): restriction 999 is not declared",
        ),
        (
            lambda t: t.formula("re(402)", context=_CONTEXT),
            MalformedElectricalExpression,
            "electrical expression 're(402)': re(402): restriction 402 has no "
            "formula of its own; it is declared only by per-period rows",
        ),
        (
            lambda t: t.formula("re(4.5)", context=_CONTEXT),
            MalformedElectricalExpression,
            f"electrical expression 're(4.5)': re(4.5): {_NOT_A_REFERENCE}",
        ),
        (
            lambda t: t.formula("re(402, 1)", context=_CONTEXT),
            MalformedElectricalExpression,
            f"electrical expression 're(402, 1)': re(402, 1): {_NOT_A_REFERENCE}",
        ),
        (
            lambda t: t.formula("carga_ande(1)", context=_CONTEXT),
            MalformedElectricalExpression,
            "electrical expression 'carga_ande(1)': carga_ande(1): data term "
            "carga_ande takes no arguments",
        ),
        (
            lambda t: t.formula("2 * peq_A_SE(1)", context=_CONTEXT),
            MalformedElectricalExpression,
            "electrical expression '2 * peq_A_SE(1)': peq_A_SE(1): data term "
            "peq_A_SE takes no arguments",
        ),
        (
            lambda t: t.relation("x", "=", "1", context=_CONTEXT),
            MalformedElectricalExpression,
            "operator '=' is not one of <=, >=, ==",
        ),
        (
            lambda t: t.formula("(1", context=_CONTEXT),
            MalformedElectricalExpression,
            "electrical expression '(1', position 0: '(' is never closed",
        ),
    ],
)
def test_untranslatable_constructs_fail_naming_the_context(
    translate: Callable[[Translator], str], error: type[ValueError], message: str
) -> None:
    with pytest.raises(ValueError) as caught:
        translate(_translator(_OVERRIDE_ONLY_MODEL))

    assert type(caught.value) is error
    assert str(caught.value) == f"{_CONTEXT}: {message}"


def test_conditions_keep_their_connectives() -> None:
    rule = "82000 < demanda_sin & demanda_sin <=95000 | val_demanda(3) > 1"

    translated = _translator(_model(rules=[(1, rule)])).condition(
        rule, context="activation rule 1"
    )

    assert (
        translated == f"82000 < {_SYSTEM_LOAD} & {_SYSTEM_LOAD} <= 95000 | "
        "bus_load_value(2) > 1"
    )


def test_bare_load_names_are_the_load_terms() -> None:
    translator = _translator(_model())

    assert (
        translator.condition(
            "demanda_sin - 2 * val_demanda_sin > 0", context="activation rule 1"
        )
        == f"{_SYSTEM_LOAD} - 2 * {_SYSTEM_LOAD} > 0"
    )


def test_a_declared_name_shadows_a_load_term() -> None:
    model = _model(expressions=[("demanda_sin", "1")], aliases=["val_demanda_sin"])

    assert (
        _translator(model).formula("demanda_sin - val_demanda_sin", context=_CONTEXT)
        == "@demanda_sin - @val_demanda_sin"
    )


@pytest.mark.parametrize(
    ("submarkets", "source", "neutral"),
    [
        ({1: 0, 4: 1}, "val_demanda_sin()", "(bus_load_value(0) + bus_load_value(1))"),
        (
            {9: 0, 1: 2, 6: 1},
            "val_demanda_sin()",
            "(bus_load_value(0) + bus_load_value(1) + bus_load_value(2))",
        ),
        ({1: 0}, "val_demanda_sin()", "bus_load_value(0)"),
        ({1: 0}, "val_demanda_sin(2)", "bus_load_value(0, 2)"),
        (
            {1: 0, 4: 1},
            "val_demanda_sin(2)",
            "(bus_load_value(0, 2) + bus_load_value(1, 2))",
        ),
        (
            {1: 0, 4: 1, 6: 1},
            "val_demanda_sin()",
            "(bus_load_value(0) + bus_load_value(1))",
        ),
    ],
)
def test_system_load_is_the_sum_of_every_bus(
    submarkets: dict[int, int], source: str, neutral: str
) -> None:
    tables = entity_tables({"submarkets": submarkets}, {}, [])

    assert _translate(source, tables, in_condition=False) == neutral


def test_load_terms_in_a_condition_read_the_value() -> None:
    rule = (
        "demanda_sin > 1 & demanda_sin(2) < 3 | "
        "demanda(6) <= val_demanda_sin & demanda(1, 2) > 0"
    )
    translator = _translator(_model(rules=[(1, rule)]))

    assert translator.condition(rule, context="activation rule 1") == (
        f"{_SYSTEM_LOAD} > 1 & (bus_load_value(0, 2) + bus_load_value(2, 2) + "
        f"bus_load_value(4, 2)) < 3 | bus_load_value(4) <= {_SYSTEM_LOAD} & "
        "bus_load_value(0, 2) > 0"
    )
    assert (
        translator.formula("se(demanda(3) > demanda_sin, 1, 0)", context=_CONTEXT)
        == f"if(bus_load_value(2) > {_SYSTEM_LOAD}, 1, 0)"
    )
    assert (
        translator.relation(
            "ger_usih(285)",
            "<=",
            "se(se(demanda(1) > 0, demanda_sin, 0) > 5, 800, 600)",
            context=_CONTEXT,
        )
        == f"hydro_generation(20) <= if(if(bus_load_value(0) > 0, {_SYSTEM_LOAD}, "
        "0) > 5, 800, 600)"
    )


_LP_LOAD = (
    "the load as an LP term is not supported; write val_demanda or "
    "val_demanda_sin to read the load as data"
)

_LOAD_EXPRESSION_READ_ONLY_BY_A_RULE = _model(
    expressions=[("E", "demanda(1)")],
    restrictions=[replace(_restriction(801, "ger_usih(285)", "<=", "100"), habilita=1)],
    rules=[(1, "E > 1")],
)


def _generic_outputs(model: LibsElectricalModel) -> GenericOutputs:
    with dx.collect():
        return build_generic_outputs(
            model,
            weekly_calendar(1),
            _TABLES,
            {},
            reserved=frozenset(),
            taken=frozenset(),
            default_penalty=1.0,
        )


@pytest.mark.parametrize(
    ("translate", "message"),
    [
        (
            lambda t: t.formula("2 * demanda(1)", context="restriction 405"),
            "restriction 405: electrical expression '2 * demanda(1)': "
            f"demanda(1): {_LP_LOAD}",
        ),
        (
            lambda t: t.formula("demanda_sin - 1", context="restriction 405"),
            "restriction 405: electrical expression 'demanda_sin - 1': "
            f"demanda_sin: {_LP_LOAD}",
        ),
        (
            lambda t: t.formula("demanda(9)", context="restriction 405"),
            f"restriction 405: electrical expression 'demanda(9)': demanda(9): "
            f"{_LP_LOAD}",
        ),
        (
            lambda t: t.relation(
                "ger_usih(285)", "<=", "0.1*demanda_sin()", context="restriction 405"
            ),
            "restriction 405: electrical expression '0.1*demanda_sin()': "
            f"demanda_sin(): {_LP_LOAD}",
        ),
        (
            lambda t: t.relation("demanda(1,2)", ">=", "0", context="restriction 405"),
            "restriction 405: electrical expression 'demanda(1,2)': "
            f"demanda(1, 2): {_LP_LOAD}",
        ),
        (
            lambda t: t.formula(
                "se(val_demanda_sin > 1, demanda(1), 0)", context="restriction 405"
            ),
            "restriction 405: electrical expression 'se(val_demanda_sin > 1, "
            f"demanda(1), 0)': demanda(1): {_LP_LOAD}",
        ),
        (
            lambda t: t.relation(
                "ger_usih(285)",
                "<=",
                "se(val_demanda(1) > 1, 0, demanda_sin)",
                context="restriction 405",
            ),
            "restriction 405: electrical expression 'se(val_demanda(1) > 1, 0, "
            f"demanda_sin)': demanda_sin: {_LP_LOAD}",
        ),
        (
            lambda _: _generic_outputs(_LOAD_EXPRESSION_READ_ONLY_BY_A_RULE),
            f"expression E: electrical expression 'demanda(1)': demanda(1): {_LP_LOAD}",
        ),
    ],
)
def test_load_as_an_lp_term_is_a_conversion_error(
    translate: Callable[[Translator], object], message: str
) -> None:
    with pytest.raises(UnsupportedSourceError) as caught:
        translate(_translator(_model()))

    assert str(caught.value) == message


def test_val_demanda_sin_is_the_sum_in_an_expression() -> None:
    translator = _translator(_model())

    assert translator.formula(
        "0.1*val_demanda_sin - val_demanda_sin(2)", context=_CONTEXT
    ) == (
        f"0.1 * {_SYSTEM_LOAD} - (bus_load_value(0, 2) + bus_load_value(2, 2) + "
        "bus_load_value(4, 2))"
    )
    assert (
        translator.relation(
            "ger_usih(285)",
            "<=",
            "se(val_demanda_sin > 9, val_demanda_sin(), 0)",
            context=_CONTEXT,
        )
        == f"hydro_generation(20) <= if({_SYSTEM_LOAD} > 9, {_SYSTEM_LOAD}, 0)"
    )


def test_a_declared_expression_shadows_an_alias_of_the_same_name() -> None:
    model = _model(expressions=[("X", "ger_usih(285)")], aliases=["X", "Y"])

    names, [warning] = _names_and_warnings(model)

    assert Translator(model, _TABLES, names).formula("X + Y", context=_CONTEXT) == (
        "@X + @Y"
    )
    with pytest.raises(KeyError):
        names.alias("X")
    assert warning.code == "decomp-electrical-alias-shadowed"
    assert warning.severity is dx.Severity.INFO
    assert warning.category == "Special constraints"
    assert warning.table is not None
    assert warning.table.columns == ["Name"]
    assert warning.table.rows == [["X"]]
    assert "read the expression" in warning.summary


def test_a_declared_name_shadows_a_data_term() -> None:
    model = _model(
        expressions=[("peq_A_SE", "1")],
        aliases=["carga_ande"],
        restrictions=[_restriction(1, "carga_ande + peq_A_SE + peq_B_SE")],
    )

    names, [warning] = _names_and_warnings(model)

    assert (
        Translator(model, _TABLES, names).formula(
            "carga_ande() + peq_A_SE + peq_B_SE", context=_CONTEXT
        )
        == "@carga_ande + @peq_A_SE + @peq_B_SE"
    )
    assert names.data("peq_B_SE") == "peq_B_SE"
    for shadowed in ("carga_ande", "peq_A_SE"):
        with pytest.raises(KeyError):
            names.data(shadowed)
    assert warning.code == "decomp-electrical-data-alias-shadowed"
    assert warning.severity is dx.Severity.INFO
    assert warning.category == "Special constraints"
    assert warning.table is not None
    assert warning.table.columns == ["Name", "Declared as"]
    assert warning.table.rows == [["carga_ande", "alias"], ["peq_A_SE", "expression"]]
    assert "parameter" in warning.summary


def test_bare_and_called_data_terms_read_the_same_parameter() -> None:
    model = _model(restrictions=[_restriction(1, "carga_ande() + peq_A_SE")])

    translator = _translator(model, frozenset({"carga_ande"}))

    assert (
        translator.formula(
            "carga_ande() - carga_ande + peq_A_SE() - peq_A_SE", context=_CONTEXT
        )
        == "@carga_ande_2 - @carga_ande_2 + @peq_A_SE - @peq_A_SE"
    )
