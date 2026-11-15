"""The electrical-expression grammar NEWAVE and DECOMP share: the parser against
a minimal scope, term rendering, and the formula-bound fold.

DECOMP's LIBS model as a scope is covered in ``tests/decomp/test_libs_electrical.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from novomodelo_bridge.core.electrical_expression import (
    CONST_TOKEN,
    GNL_TOKEN,
    PEQ_TOKEN,
    SE_TOKEN,
    Bucket,
    MalformedElectricalExpression,
    ParsedTerm,
    UnrecognizedElectricalToken,
    UnsupportedElectricalExpression,
    classify_term,
    classify_terms,
    evaluate_terms,
    fold_formula_bound,
    parse_linear_expression,
    term_text,
)


@dataclass(frozen=True)
class _Scope:
    formulas: dict[int, str] = field(default_factory=dict)
    gnl_thermal_codes: frozenset[int] = frozenset()

    def expression_formula(self, name: str) -> str | None:
        return None

    def is_alias(self, name: str) -> bool:
        return False

    def restriction_formula(self, code: int) -> str | None:
        return self.formulas.get(code)


def _terms(text: str, scope: _Scope | None = None) -> list[tuple[float, str, tuple]]:
    return [
        (term.coefficient, term.token, term.args)
        for term in parse_linear_expression(text, scope or _Scope())
    ]


def test_a_coefficient_written_against_its_term_multiplies_it() -> None:
    assert _terms("1.0ener_interc(2,1) + 0.5ger_usih(66)") == [
        (1.0, "ener_interc", (2, 1)),
        (0.5, "ger_usih", (66,)),
    ]


def test_a_coefficient_before_a_space_or_a_group_multiplies_it() -> None:
    assert _terms("2 ger_usih(1) - 0.5(ger_usih(2) - ger_usih(3))") == [
        (2.0, "ger_usih", (1,)),
        (-0.5, "ger_usih", (2,)),
        (0.5, "ger_usih", (3,)),
    ]


def test_a_number_followed_by_an_operator_is_a_constant() -> None:
    assert _terms("ger_usih(1) + 100 - 1E+3") == [
        (1.0, "ger_usih", (1,)),
        (100.0, CONST_TOKEN, ()),
        (-1000.0, CONST_TOKEN, ()),
    ]


def test_re_inlines_the_restriction_formula_under_its_coefficient() -> None:
    scope = _Scope(formulas={7: "ger_usih(1) + demanda(2)"})

    assert _terms("ger_usit(4) + 2*re(7)", scope) == [
        (1.0, "ger_usit", (4,)),
        (2.0, "ger_usih", (1,)),
        (2.0, "demanda", (2,)),
    ]


def test_re_to_a_restriction_the_scope_lacks_names_it() -> None:
    with pytest.raises(
        MalformedElectricalExpression,
        match=r"unknown restriction 9 in electrical expression 're\(9\)'",
    ):
        parse_linear_expression("re(9)", _Scope())


@dataclass(frozen=True)
class _ScopeWithoutAFormula(_Scope):
    def restriction_formula(self, code: int) -> str | None:
        raise MalformedElectricalExpression(f"restriction {code} has no formula")


def test_re_to_a_restriction_with_no_formula_is_malformed_naming_the_expression() -> (
    None
):
    with pytest.raises(MalformedElectricalExpression) as err:
        parse_linear_expression("2 * re(3)", _ScopeWithoutAFormula())

    assert str(err.value) == (
        "restriction 3 has no formula in electrical expression '2 * re(3)'"
    )


def test_a_cycle_through_re_is_refused() -> None:
    scope = _Scope(formulas={1: "re(2)", 2: "ger_usih(5) + re(1)"})

    with pytest.raises(
        MalformedElectricalExpression, match="cyclic electrical expression reference"
    ):
        parse_linear_expression("re(1)", scope)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ener_interc(1)", "ener_interc\\(...\\) expects 2 arguments"),
        ("ger_usih(1, 2)", "ger_usih\\(...\\) expects one argument"),
        ("demanda()", "demanda\\(...\\) expects one argument"),
        ("re(1, 2)", "re\\(...\\) expects exactly one restriction id"),
        ("se(demanda(1) > 0, 1)", "se\\(...\\) expects exactly 3 arguments"),
    ],
)
def test_a_function_given_the_wrong_number_of_arguments_is_refused(
    text: str, expected: str
) -> None:
    with pytest.raises(MalformedElectricalExpression, match=expected):
        parse_linear_expression(text, _Scope())


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ger_usih(1) )", "unexpected trailing content"),
        ("ger_usih(1) * ger_usih(2)", "unexpected trailing content '\\* ger_usih"),
        ("ger_usih(1) / 0", "division by zero"),
        ("(ger_usih(1)", "expected '\\)' to close '\\('"),
        ("ger_usih(1) +", "expected a term but reached the end"),
        ("ger_usih(1) + #", "malformed token at '#'"),
        ("ger_usih(1.5)", "expected an integer argument for 'ger_usih'"),
        ("se(demanda(1) > 0, 1, 2", "unbalanced parentheses in se"),
    ],
)
def test_text_the_grammar_rejects_is_malformed(text: str, expected: str) -> None:
    with pytest.raises(MalformedElectricalExpression, match=expected):
        parse_linear_expression(text, _Scope())


@pytest.mark.parametrize(
    ("text", "constant"),
    [
        ("se (demanda_sin > 0, 50.0, 0) * 2 + 50.0 * 2.0", "2"),
        ("ger_usih(1) * 2", "2"),
        ("ger_usih(1) / 4", "4"),
        ("50.0 * 2.0", "2.0"),
        ("ger_usih(3) + (ger_usih(1) - ger_usih(2)) * -0.5", "-0.5"),
    ],
)
def test_a_constant_factor_after_its_term_is_not_read_yet(
    text: str, constant: str
) -> None:
    with pytest.raises(UnsupportedElectricalExpression) as err:
        parse_linear_expression(text, _Scope())

    assert not isinstance(err.value, MalformedElectricalExpression)
    assert f"by the constant {constant} written after it" in str(err.value)


def test_an_unknown_function_names_only_the_functions_a_formula_may_call() -> None:
    (term,) = parse_linear_expression("ger_usina(66)", _Scope())

    with pytest.raises(MalformedElectricalExpression) as err:
        classify_term(term)

    message = str(err.value)
    assert message.startswith("unknown function ger_usina(...)")
    assert "ger_usih" in message
    assert "re" in message.split(", ")
    for internal in ("__", "alias", "peq"):
        assert internal not in message


def test_an_undeclared_bare_name_is_its_own_error() -> None:
    with pytest.raises(UnrecognizedElectricalToken) as err:
        parse_linear_expression("ger_usih(1) + FLUXO", _Scope())

    assert err.value.name == "FLUXO"


def test_a_gnl_thermal_generation_is_input_data() -> None:
    (term,) = parse_linear_expression(
        "ger_usit(3)", _Scope(gnl_thermal_codes=frozenset({3}))
    )

    assert term.token == GNL_TOKEN
    assert classify_terms([term])[Bucket.B] == [term]


@pytest.mark.parametrize(
    ("term", "text"),
    [
        (ParsedTerm(0.5, "ener_interc", (2, 1)), "ener_interc(2, 1)"),
        (ParsedTerm(1.0, GNL_TOKEN, (3,)), "ger_usit(3)"),
        (ParsedTerm(1.0, "demanda_sin"), "demanda_sin"),
        (ParsedTerm(-250.0, CONST_TOKEN), "-250"),
        (
            ParsedTerm(1.0, PEQ_TOKEN, pq_name="N_PCHgd", pq_submarket="NE"),
            "peq_N_PCHgd_NE",
        ),
        (
            ParsedTerm(
                1.0,
                SE_TOKEN,
                condition="demanda(1) > 10",
                branch_true="1",
                branch_false="0",
            ),
            "se(demanda(1) > 10, 1, 0)",
        ),
    ],
)
def test_term_text_writes_the_term_as_a_formula_does(
    term: ParsedTerm, text: str
) -> None:
    assert term_text(term) == text


def test_fold_moves_the_data_terms_onto_both_limits() -> None:
    partition = classify_terms(
        parse_linear_expression("ger_usih(1) - demanda(2) + 100", _Scope())
    )

    def loads(term: ParsedTerm) -> float:
        return 1.0 if term.token == CONST_TOKEN else 40.0

    folded = fold_formula_bound(partition, 0.0, 500.0, loads)

    assert [(t.coefficient, t.token) for t in folded.terms] == [(1.0, "ger_usih")]
    assert (folded.lower, folded.upper) == (-60.0, 440.0)


def test_fold_keeps_an_unbounded_side_unbounded() -> None:
    partition = classify_terms(parse_linear_expression("ger_usih(1) + 100", _Scope()))

    folded = fold_formula_bound(partition, None, 500.0, lambda _term: 1.0)

    assert (folded.lower, folded.upper) == (None, 400.0)


def test_evaluate_terms_reads_a_constant_from_its_coefficient() -> None:
    terms = [ParsedTerm(100.0, CONST_TOKEN), ParsedTerm(2.0, "demanda", (1,))]

    assert evaluate_terms(terms, lambda _term: 0.0) == 100.0
