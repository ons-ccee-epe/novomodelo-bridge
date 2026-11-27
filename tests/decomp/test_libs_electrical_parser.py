"""Tier-1 tests for the source-model electrical DSL parser."""

from __future__ import annotations

import re
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from novomodelo_bridge.core.electrical_expression import MalformedElectricalExpression
from novomodelo_bridge.decomp.converters.libs_electrical import (
    ElectricalAlias,
    ElectricalExpression,
    LibsElectricalModel,
)
from novomodelo_bridge.decomp.converters.libs_electrical_parser import (
    Binary,
    Call,
    Group,
    Keyword,
    Name,
    NameKind,
    Node,
    Number,
    Param,
    Unary,
    classify_name,
    iter_nodes,
    parse_condition,
    parse_expression,
    render,
    split_small_plant,
)

if TYPE_CHECKING:
    from collections.abc import Callable

#: Independent of the parser's own tokenizer: numbers (exponent included),
#: names, two-character comparisons, then any other single character.
_TOKEN_RE = re.compile(r"\d+\.?\d*(?:[eE][+-]?\d+)?|\w+|[<>=]=|\S")


def _call(name: str, *args: Node) -> Call:
    return Call(name, args)


def _model(
    expressions: tuple[str, ...], aliases: tuple[str, ...]
) -> LibsElectricalModel:
    return LibsElectricalModel(
        expressions={
            name: ElectricalExpression(code=code, name=name, formula="ger_usih(1)")
            for code, name in enumerate(expressions)
        },
        aliases={
            name: ElectricalAlias(code=code, name=name, values={})
            for code, name in enumerate(aliases)
        },
        restrictions={},
        rules={},
    )


def test_parse_keeps_the_group_and_does_not_distribute() -> None:
    tree = parse_expression("0.05*(val_demanda(1)+val_demanda(2))")

    assert tree == Binary(
        "*",
        Number("0.05"),
        Group(
            Binary(
                "+",
                _call("val_demanda", Number("1")),
                _call("val_demanda", Number("2")),
            )
        ),
    )
    assert render(tree) == "0.05 * (val_demanda(1) + val_demanda(2))"


def test_se_parses_its_condition_and_both_branches() -> None:
    text = (
        "se(demanda_sin > 50000 & demanda(1) <= 9E3, ger_usih(1) + ger_usih(2), -1E+31)"
    )

    tree = parse_expression(text)

    assert tree == _call(
        "se",
        Binary(
            "&",
            Binary(">", Name("demanda_sin"), Number("50000")),
            Binary("<=", _call("demanda", Number("1")), Number("9E3")),
        ),
        Binary("+", _call("ger_usih", Number("1")), _call("ger_usih", Number("2"))),
        Unary("-", Number("1E+31")),
    )
    assert render(tree) == text


@pytest.mark.parametrize(
    "text",
    [
        "re(405)",
        "FNESE + C_AM_cMMGD",
        "1E2 + EXPNE - 2*E1",
        "- ger_usih(1)",
        "a / 2",
        "peq_N_PCHgd_SE",
        "ger_conjh(66,1)",
        "0.05*(val_demanda(1)+val_demanda(2))",
        "disp_usih(261) - ger_usih(261) + 0.05*carga_ande",
        "11000 - 0.04*val_demanda(3) + 0.06*ger_pee(11)",
        "se(demanda_sin > 50000 & demanda(1) <= 9E3, ger_usih(1) + ger_usih(2), -1E+31)",
        "se(demanda(1)>100|demanda_sin<=82000,ger_usih(1),se(MMGDSECO==0,0,1))",
        "-(ener_interc(3,1) - ener_interc(1,3)) * 0.5",
        "carga_ande() + +2.5e-3",
        "ger_usih( 1 , 2 )  -  ger_usit(3)/ 2",
        "((a)) - -b",
    ],
)
def test_render_round_trips_the_token_sequence(text: str) -> None:
    tree = parse_expression(text)
    rendered = render(tree)

    assert parse_expression(rendered) == tree
    assert _TOKEN_RE.findall(rendered) == _TOKEN_RE.findall(text)


@pytest.mark.parametrize(
    "text",
    [
        "82000 < demanda_sin & demanda_sin <=95000 | val_demanda(3) > 1",
        "demanda(3)+demanda(4) >= 17000",
        "(demanda(1) - 2) * 3 == -MMGDSECO",
    ],
)
def test_condition_render_round_trips_the_token_sequence(text: str) -> None:
    tree = parse_condition(text)
    rendered = render(tree)

    assert parse_condition(rendered) == tree
    assert _TOKEN_RE.findall(rendered) == _TOKEN_RE.findall(text)


@pytest.mark.parametrize(
    ("parse", "text", "position", "error"),
    [
        (parse_expression, "ger_usih(261", 8, "'(' is never closed"),
        (
            parse_expression,
            "ger_usih(1) +",
            13,
            "expected an operand but the text ends",
        ),
        (parse_condition, "a <> b", 3, "expected an operand, found '>'"),
        (parse_condition, "a != b", 2, "unexpected character '!'"),
        (parse_expression, "se(a > 1, b)", 0, "exactly 3 arguments"),
        (
            parse_expression,
            "ger_usih(1) >= 3",
            12,
            "only allowed at the top level of a condition",
        ),
        (parse_expression, "", 0, "expected an operand but the text ends"),
        (parse_expression, "ger_usih(1))", 11, "')' has no matching '('"),
        (
            parse_expression,
            "ger_usih(1) 2",
            12,
            "expected the end of the text, found '2'",
        ),
        (parse_expression, "* ger_usih(1)", 0, "expected an operand, found '*'"),
        (parse_expression, "1 + 1.2.3", 4, "malformed number"),
        (parse_expression, "2E + 1", 0, "malformed number"),
        (parse_condition, "a = b", 2, "unexpected character '='"),
        (parse_expression, "a; b", 1, "unexpected character ';'"),
        (parse_expression, "f(1,)", 4, "expected an operand, found ')'"),
        (parse_expression, "f(,1)", 2, "expected an operand, found ','"),
        (parse_condition, "a < b < c", 6, "relates exactly two expressions"),
        (
            parse_condition,
            "se(a < b < c, 1, 0) > 0",
            9,
            "relates exactly two expressions",
        ),
        (parse_condition, "demanda_sin", 11, "expected a comparison operator"),
        (
            parse_condition,
            "demanda_sin & demanda(1) > 0",
            12,
            "expected a comparison operator",
        ),
        (
            parse_condition,
            "(demanda(1) > 0 & demanda(2) < 5)",
            12,
            "only allowed at the top level",
        ),
        (parse_condition, "f(a & b) > 1", 4, "expected ',' or ')', found '&'"),
        (
            parse_expression,
            "demanda(1) & demanda(2)",
            11,
            "expected the end of the text",
        ),
        (
            parse_expression,
            "se()",
            0,
            "exactly 3 arguments (a condition, the value when it holds, the value otherwise), found 0",
        ),
        (parse_expression, "1 + se(a > 1, b, c, d)", 4, "found 4"),
        (parse_expression, "se(a, b, c)", 4, "expected a comparison operator"),
        (parse_expression, "se(a > 1, b > 2, c)", 12, "only allowed at the top level"),
    ],
)
def test_syntax_errors_name_the_text_and_position(
    parse: Callable[[str], Node], text: str, position: int, error: str
) -> None:
    with pytest.raises(MalformedElectricalExpression) as exc_info:
        parse(text)

    message = str(exc_info.value)
    assert message.startswith(f"electrical expression {text!r}, position {position}: ")
    assert error in message


@pytest.mark.parametrize(
    "text", ["1E2", "1e2", "1e+2", "1E+2", "1.5E3", "2.5e-3", "1E+31", "0.05", "7."]
)
def test_numbers_keep_their_literal_text(text: str) -> None:
    tree = parse_expression(text)

    assert tree == Number(text)
    assert render(tree) == text


def test_exponent_number_beside_names_starting_with_e() -> None:
    assert parse_expression("1E2 + EXPNE - 2*E1") == Binary(
        "-",
        Binary("+", Number("1E2"), Name("EXPNE")),
        Binary("*", Number("2"), Name("E1")),
    )


@pytest.mark.parametrize(
    "name", ["E1", "_x1", "geração_mmgd", "ÁREA_SUL", "peq_N_PCHgd_SE"]
)
def test_names_follow_the_identifier_rule(name: str) -> None:
    assert parse_expression(name) == Name(name)


def test_call_arguments_are_expressions() -> None:
    assert parse_expression("re(405)") == _call("re", Number("405"))
    assert parse_expression("carga_ande()") == _call("carga_ande")
    assert parse_expression("f (a + 1, -2, g())") == _call(
        "f",
        Binary("+", Name("a"), Number("1")),
        Unary("-", Number("2")),
        _call("g"),
    )


def test_operators_bind_by_precedence_and_chains_fold_left() -> None:
    a, b, c = Name("a"), Name("b"), Name("c")

    assert parse_expression("-a * b + c / 2") == Binary(
        "+",
        Binary("*", Unary("-", a), b),
        Binary("/", c, Number("2")),
    )
    assert parse_expression("a - b - c") == Binary("-", Binary("-", a, b), c)
    assert parse_expression("a / b * c") == Binary("*", Binary("/", a, b), c)

    def at_most(name: Name, bound: str) -> Binary:
        return Binary("<=", name, Number(bound))

    assert parse_condition("a<=1 | b<=2 & c<=3") == Binary(
        "|", at_most(a, "1"), Binary("&", at_most(b, "2"), at_most(c, "3"))
    )
    assert parse_condition("a<=1 & b<=2 | c<=3") == Binary(
        "|", Binary("&", at_most(a, "1"), at_most(b, "2")), at_most(c, "3")
    )
    assert parse_condition("a<=1 & b<=2 & c<=3") == Binary(
        "&", Binary("&", at_most(a, "1"), at_most(b, "2")), at_most(c, "3")
    )


def test_se_keeps_nested_calls_commas_and_parentheses() -> None:
    tree = parse_expression("se(f(1,2) > (g(3) + 1), h(4, 5), se(x == 0, (1 + 2), 0))")

    assert tree == _call(
        "se",
        Binary(
            ">",
            _call("f", Number("1"), Number("2")),
            Group(Binary("+", _call("g", Number("3")), Number("1"))),
        ),
        _call("h", Number("4"), Number("5")),
        _call(
            "se",
            Binary("==", Name("x"), Number("0")),
            Group(Binary("+", Number("1"), Number("2"))),
            Number("0"),
        ),
    )


@pytest.mark.parametrize("op", ["<=", ">=", "==", "<", ">"])
@pytest.mark.parametrize("spacing", ["", " "])
def test_condition_operators_match_longest(op: str, spacing: str) -> None:
    tree = parse_condition(f"demanda_sin{spacing}{op}{spacing}50000")

    assert tree == Binary(op, Name("demanda_sin"), Number("50000"))


def test_render_prints_translated_nodes() -> None:
    tree = Binary(
        "-",
        Param("FNESE"),
        Call("hydro_generation", (Number("3"), Keyword("bus", Param("b")))),
    )

    assert render(tree) == "@FNESE - hydro_generation(3, bus=@b)"
    assert (
        render(Unary("-", Group(Call("bus_load_value", (Number("0"),)))))
        == "-(bus_load_value(0))"
    )


def test_iter_nodes_visits_every_node_in_preorder() -> None:
    first = _call("f", Number("1"))
    sign = Unary("-", Name("g"))
    total = Binary("+", first, sign)
    group = Group(total)
    tree = Binary("*", Number("0.05"), group)
    assert parse_expression("0.05*(f(1) + -g)") == tree

    assert list(iter_nodes(tree)) == [
        tree,
        Number("0.05"),
        group,
        total,
        first,
        Number("1"),
        sign,
        Name("g"),
    ]

    keyword = Keyword("bus", Number("7"))
    call = Call("f", (Param("p"), keyword))
    assert list(iter_nodes(call)) == [call, Param("p"), keyword, Number("7")]


def test_classify_name_follows_the_source_order() -> None:
    model = _model(
        expressions=("BOTH", "peq_N_SE"),
        aliases=("BOTH", "MMGDSECO", "val_demanda_sin"),
    )
    expected = {
        "BOTH": NameKind.EXPRESSION,
        "MMGDSECO": NameKind.ALIAS,
        "val_demanda_sin": NameKind.ALIAS,
        "peq_N_SE": NameKind.EXPRESSION,
        "demanda_sin": NameKind.LOAD,
        "carga_ande": NameKind.DATA,
        "peq_x_SE": NameKind.SMALL_PLANT,
        "peq_SE": NameKind.UNRECOGNIZED,
        "ZZTOP": NameKind.UNRECOGNIZED,
    }

    assert {name: classify_name(name, model) for name in expected} == expected
    assert classify_name("val_demanda_sin", _model((), ())) is NameKind.LOAD


@pytest.mark.parametrize(
    ("name", "parts"),
    [
        ("peq_N_PCHgd_SE", ("N_PCHgd", "SE")),
        ("peq_SE_PCHgd_SECO", ("SE_PCHgd", "SECO")),
        ("peq_x_SE", ("x", "SE")),
        ("peq_SE", None),
        ("peq__SE", None),
        ("peq_x_", None),
        ("x_peq_SE", None),
    ],
)
def test_split_small_plant_splits_at_the_last_underscore(
    name: str, parts: tuple[str, str] | None
) -> None:
    assert split_small_plant(name) == parts


def test_parser_loads_neither_pandas_nor_the_reader() -> None:
    probe = (
        "import sys\n"
        "import novomodelo_bridge.decomp.converters.libs_electrical_parser\n"
        "print(sorted({'pandas', 'novomodelo_bridge.decomp.converters.libs_electrical'}"
        " & set(sys.modules)))"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"
