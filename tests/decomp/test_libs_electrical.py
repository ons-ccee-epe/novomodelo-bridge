"""Tier-1 tests for the LIBs-era electrical special-constraint reader (E9-1).

Synthetic-fixture only — a fake ``Restricoes`` returns pandas DataFrames shaped
like idecomp's ``df=True`` accessors, so these run in CI with no real deck.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.libs_electrical import (
    ActivationRule,
    AssembledBound,
    AvailablePower,
    AvailablePowerSource,
    Bucket,
    Comparison,
    DataContext,
    ElectricalAlias,
    ElectricalExpression,
    ElectricalRestriction,
    LibsElectricalModel,
    ParsedTerm,
    PeriodPatamarOverride,
    UnrecognizedElectricalToken,
    ViolationTreatment,
    _effective_inequacao_sides,
    _fold_formula_bound,
    _fold_inequacao_bound,
    _fold_reserve_disp_usih,
    _UnresolvableBucketBTerm,
    _UnresolvableDispUsih,
    active_cells,
    assemble_bound,
    build_available_power,
    build_data_context,
    classify_term,
    classify_terms,
    evaluate_rule,
    evaluate_se,
    is_always_active,
    parse_activation_rule,
    parse_linear_expression,
    read_carga_ande,
    read_libs_electrical,
    resolve_disp_usih,
)
from novomodelo_bridge.decomp.group_bounds import GroupBoundEntry
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import OperativeStage
from tests.conftest import make_decomp_case


class _StubRestricoes:
    """Return a preset DataFrame (or ``None``) for each ``Restricoes`` accessor."""

    def __init__(self, **frames: pd.DataFrame) -> None:
        self._frames = frames

    def __getattr__(self, name: str):  # noqa: ANN204 - test double
        if name.startswith("_"):
            raise AttributeError(name)

        def accessor(df: bool = True) -> pd.DataFrame | None:
            return self._frames.get(name)

        return accessor


def _stage(index: int) -> OperativeStage:
    return OperativeStage(
        index=index,
        start_date=date(2026, 7, 4),
        end_date=date(2026, 7, 11),
        season_id=6,
        block_hours=(168.0,),
    )


def _calendar(n_stages: int) -> list[OperativeStage]:
    return [_stage(i) for i in range(n_stages)]


def _expressao(*rows: tuple) -> pd.DataFrame:
    """EXPRESSAO-ELETRICA rows: (codigo_expressao, identificador_expressao, formula)."""
    return pd.DataFrame(
        rows, columns=["codigo_expressao", "identificador_expressao", "formula"]
    )


def _alias(*rows: tuple) -> pd.DataFrame:
    """ALIAS-ELETRICO rows: (codigo_alias, identificador_alias)."""
    return pd.DataFrame(rows, columns=["codigo_alias", "identificador_alias"])


def _alias_valor(*rows: tuple) -> pd.DataFrame:
    """ALIAS-ELETRICO-VALOR-PERIODO-PATAMAR rows: (codigo_alias, estagio_inicio,
    estagio_fim, patamar, valor)."""
    return pd.DataFrame(
        rows,
        columns=["codigo_alias", "estagio_inicio", "estagio_fim", "patamar", "valor"],
    )


def _re_formula(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-FORMULA rows: (codigo_restricao, formula)."""
    return pd.DataFrame(rows, columns=["codigo_restricao", "formula"])


def _re_lim_formula(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR rows: (codigo_restricao,
    estagio_inicio, estagio_fim, patamar, limite_inferior, limite_superior)."""
    return pd.DataFrame(
        rows,
        columns=[
            "codigo_restricao",
            "estagio_inicio",
            "estagio_fim",
            "patamar",
            "limite_inferior",
            "limite_superior",
        ],
    )


def _re_inequacao(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-INEQUACAO rows: (codigo_restricao, formula, operador,
    formula_limite)."""
    return pd.DataFrame(
        rows, columns=["codigo_restricao", "formula", "operador", "formula_limite"]
    )


def _re_inequacao_per_pat(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR rows: (codigo_restricao,
    estagio_inicio, estagio_fim, patamar, formula, operador, formula_limite)."""
    return pd.DataFrame(
        rows,
        columns=[
            "codigo_restricao",
            "estagio_inicio",
            "estagio_fim",
            "patamar",
            "formula",
            "operador",
            "formula_limite",
        ],
    )


def _horizonte(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-HORIZONTE-PERIODO rows: (codigo_restricao,
    estagio_inicio, estagio_fim)."""
    return pd.DataFrame(
        rows, columns=["codigo_restricao", "estagio_inicio", "estagio_fim"]
    )


def _habilita(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-HABILITA rows: (codigo_restricao, codigo_regra_ativacao)."""
    return pd.DataFrame(rows, columns=["codigo_restricao", "codigo_regra_ativacao"])


def _regra(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-REGRA-ATIVACAO rows: (codigo_regra_ativacao,
    regra_ativacao)."""
    return pd.DataFrame(rows, columns=["codigo_regra_ativacao", "regra_ativacao"])


def _tratamento_violacao(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO rows: (codigo_restricao,
    custo_violacao)."""
    return pd.DataFrame(rows, columns=["codigo_restricao", "custo_violacao"])


def test_expressions_indexed_by_name() -> None:
    stub = _StubRestricoes(
        expressao_eletrica=_expressao(
            (1, "FNESE", "ener_interc(3,1) - ener_interc(1,3)"),
            (2, "RNE", "ger_usih(10)"),
        ),
    )
    model = read_libs_electrical(stub, _calendar(6))
    assert model is not None
    assert model.expressions["FNESE"].code == 1
    assert model.expressions["FNESE"].formula == "ener_interc(3,1) - ener_interc(1,3)"
    assert model.expressions["RNE"].code == 2
    assert model.expressions["RNE"].formula == "ger_usih(10)"


def test_aliases_with_period_patamar_values_including_na_patamar() -> None:
    stub = _StubRestricoes(
        alias_eletrico=_alias((1, "MMGDSIN")),
        alias_eletrico_valor_periodo_patamar=_alias_valor(
            (1, 1, 2, 1.0, 100.0),
            (1, 3, 3, float("nan"), 250.0),
        ),
    )
    model = read_libs_electrical(stub, _calendar(6))
    assert model is not None
    alias = model.aliases["MMGDSIN"]
    assert alias.code == 1
    # a two-stage range (estagio 1..2) expands to both 0-based stage indices
    assert alias.values[(0, 1)] == 100.0
    assert alias.values[(1, 1)] == 100.0
    # NA patamar -> None ("all blocks")
    assert alias.values[(2, None)] == 250.0


def test_formula_restriction_two_sided_limits_with_sentinel() -> None:
    stub = _StubRestricoes(
        restricao_eletrica_formula=_re_formula(
            (401, "ger_usih(285)+ger_usih(287)"),
        ),
        restricao_eletrica_horizonte_periodo=_horizonte((401, 1, 6)),
        restricao_eletrica_limite_formula_periodo_patamar=_re_lim_formula(
            (401, 1, 6, float("nan"), -1e31, 6300.0),
        ),
    )
    model = read_libs_electrical(stub, _calendar(6))
    assert model is not None
    restriction = model.restrictions[401]
    assert restriction.is_formula is True
    assert restriction.lhs == "ger_usih(285)+ger_usih(287)"
    assert restriction.operator is None
    assert restriction.rhs is None
    assert restriction.stage_start == 0
    assert restriction.stage_end == 5
    # -1E+31 sentinel normalizes to None; the upper side is verbatim
    assert restriction.limits[(0, None)] == (None, 6300.0)
    assert restriction.limits[(5, None)] == (None, 6300.0)


def test_inequacao_restriction_verbatim_lhs_operator_rhs() -> None:
    stub = _StubRestricoes(
        restricao_eletrica_inequacao=_re_inequacao(
            (402, "disp_usih(261) - ger_usih(261)", ">=", "0.05*(val_demanda(1))"),
        ),
    )
    model = read_libs_electrical(stub, _calendar(6))
    assert model is not None
    restriction = model.restrictions[402]
    assert restriction.is_formula is False
    assert restriction.lhs == "disp_usih(261) - ger_usih(261)"
    assert restriction.operator == ">="
    assert restriction.rhs == "0.05*(val_demanda(1))"


def test_inequacao_period_patamar_overrides_read_verbatim() -> None:
    stub = _StubRestricoes(
        restricao_eletrica_inequacao=_re_inequacao(
            (406, "EXPNE", "<=", "14700"),
        ),
        restricao_eletrica_inequacao_periodo_patamar=_re_inequacao_per_pat(
            (406, 1, 6, 1, "EXPNE", "<=", "14700"),
        ),
    )
    model = read_libs_electrical(stub, _calendar(6))
    assert model is not None
    restriction = model.restrictions[406]
    assert len(restriction.overrides) == 1
    override = restriction.overrides[0]
    assert override.stage_start == 0
    assert override.stage_end == 5
    assert override.patamar == 1
    assert override.lhs == "EXPNE"
    assert override.operator == "<="
    assert override.rhs == "14700"


def test_restriction_habilita_rule_and_violation_joined() -> None:
    stub = _StubRestricoes(
        restricao_eletrica_inequacao=_re_inequacao(
            (501, "demanda(1)", ">=", "1000"),
        ),
        restricao_eletrica_habilita=_habilita((501, 10)),
        restricao_eletrica_regra_ativacao=_regra((10, "demanda(1) > 5000")),
        restricao_eletrica_tratamento_violacao=_tratamento_violacao((501, 1234.5)),
    )
    model = read_libs_electrical(stub, _calendar(6))
    assert model is not None
    restriction = model.restrictions[501]
    assert restriction.habilita == 10
    assert model.rules[10] == "demanda(1) > 5000"
    assert restriction.violation == ViolationTreatment(enabled=True, penalty=1234.5)


def test_horizonte_period_out_of_range_raises_value_error() -> None:
    stub = _StubRestricoes(
        restricao_eletrica_inequacao=_re_inequacao(
            (99, "ger_usih(1)", ">=", "0"),
        ),
        restricao_eletrica_horizonte_periodo=_horizonte((99, 1, 7)),
    )
    with pytest.raises(ValueError) as exc_info:
        read_libs_electrical(stub, _calendar(6))
    message = str(exc_info.value)
    assert "7" in message
    assert "6" in message


def test_no_electrical_cards_returns_none() -> None:
    stub = _StubRestricoes()
    assert read_libs_electrical(stub, _calendar(6)) is None


# ---------------------------------------------------------------------------
# parse_linear_expression
# ---------------------------------------------------------------------------


def _model(
    expressions: dict[str, str] | None = None,
    aliases: tuple[str, ...] = (),
    restrictions: dict[int, str] | None = None,
    alias_values: dict[str, dict[tuple[int, int | None], float]] | None = None,
) -> LibsElectricalModel:
    """A minimal hand-built model for parser tests: named expressions keyed by
    ``{name: formula}``, bare alias names (their per-(stage,patamar) values are
    irrelevant to the parser, which never resolves them, but relevant to
    ``build_data_context`` -- supply them via *alias_values*,
    keyed by alias name), and restrictions keyed by ``{code: lhs}`` (used only
    to exercise ``re(R)`` expansion)."""
    return LibsElectricalModel(
        expressions={
            name: ElectricalExpression(code=i, name=name, formula=formula)
            for i, (name, formula) in enumerate((expressions or {}).items())
        },
        aliases={
            name: ElectricalAlias(
                code=i, name=name, values=(alias_values or {}).get(name, {})
            )
            for i, name in enumerate(aliases)
        },
        restrictions={
            code: ElectricalRestriction(
                code=code,
                lhs=lhs,
                operator=None,
                rhs=None,
                is_formula=True,
                limits={},
                overrides=(),
            )
            for code, lhs in (restrictions or {}).items()
        },
        rules={},
    )


def test_sign_distribution_binary_minus() -> None:
    model = _model()
    terms = parse_linear_expression("disp_usih(261) - ger_usih(261)", model)
    assert terms == [
        ParsedTerm(coefficient=1.0, token="disp_usih", args=(261,)),
        ParsedTerm(coefficient=-1.0, token="ger_usih", args=(261,)),
    ]


def test_sign_distribution_leading_unary_minus() -> None:
    model = _model()
    terms = parse_linear_expression("-ger_usih(5) + ger_usih(6)", model)
    assert terms == [
        ParsedTerm(coefficient=-1.0, token="ger_usih", args=(5,)),
        ParsedTerm(coefficient=1.0, token="ger_usih", args=(6,)),
    ]


def test_scalar_distributed_over_parenthesized_sum() -> None:
    model = _model()
    terms = parse_linear_expression("0.05*(val_demanda(1) + val_demanda(2))", model)
    assert terms == [
        ParsedTerm(coefficient=0.05, token="val_demanda", args=(1,)),
        ParsedTerm(coefficient=0.05, token="val_demanda", args=(2,)),
    ]


def test_standalone_constant_and_explicit_coefficient_terms() -> None:
    model = _model()
    terms = parse_linear_expression(
        "11000 - 0.04*val_demanda(3) + 0.06*ger_pee(11)", model
    )
    assert terms == [
        ParsedTerm(coefficient=11000.0, token="__const__"),
        ParsedTerm(coefficient=-0.04, token="val_demanda", args=(3,)),
        ParsedTerm(coefficient=0.06, token="ger_pee", args=(11,)),
    ]


def test_named_expression_inlined_no_reference_token_remains() -> None:
    model = _model(expressions={"RNE": "ener_interc(3,1) - ener_interc(1,3)"})
    terms = parse_linear_expression("RNE", model)
    assert terms == [
        ParsedTerm(coefficient=1.0, token="ener_interc", args=(3, 1)),
        ParsedTerm(coefficient=-1.0, token="ener_interc", args=(1, 3)),
    ]
    assert all(term.token != "RNE" for term in terms)


def test_named_expression_inlining_multiplies_through_enclosing_coefficient() -> None:
    model = _model(expressions={"RNE": "ener_interc(3,1) - ener_interc(1,3)"})
    terms = parse_linear_expression("0.5*RNE", model)
    assert terms == [
        ParsedTerm(coefficient=0.5, token="ener_interc", args=(3, 1)),
        ParsedTerm(coefficient=-0.5, token="ener_interc", args=(1, 3)),
    ]


def test_nested_named_expression_inlining() -> None:
    model = _model(
        expressions={
            "OUTER": "2*INNER - ger_usih(1)",
            "INNER": "ger_usih(2) + ger_usih(3)",
        }
    )
    terms = parse_linear_expression("OUTER", model)
    assert terms == [
        ParsedTerm(coefficient=2.0, token="ger_usih", args=(2,)),
        ParsedTerm(coefficient=2.0, token="ger_usih", args=(3,)),
        ParsedTerm(coefficient=-1.0, token="ger_usih", args=(1,)),
    ]


def test_re_reference_inlines_restriction_formula() -> None:
    model = _model(restrictions={401: "ger_usih(285) + ger_usih(287)"})
    terms = parse_linear_expression("re(401)", model)
    assert terms == [
        ParsedTerm(coefficient=1.0, token="ger_usih", args=(285,)),
        ParsedTerm(coefficient=1.0, token="ger_usih", args=(287,)),
    ]


def test_alias_reference_kept_as_single_unexpanded_token() -> None:
    model = _model(aliases=("MMGDSECO",))
    terms = parse_linear_expression("MMGDSECO", model)
    assert terms == [
        ParsedTerm(coefficient=1.0, token="alias", alias_name="MMGDSECO"),
    ]


def test_alias_reference_not_a_numeric_value() -> None:
    model = _model(aliases=("MMGDSECO",))
    terms = parse_linear_expression("MMGDSECO", model)
    assert len(terms) == 1
    assert terms[0].token != "__const__"
    assert terms[0].alias_name == "MMGDSECO"


def test_builtin_bare_tokens_kept_unexpanded() -> None:
    model = _model()
    for name in ("demanda_sin", "val_demanda_sin", "carga_ande"):
        terms = parse_linear_expression(name, model)
        assert terms == [ParsedTerm(coefficient=1.0, token=name)]


def test_cyclic_named_expression_reference_raises_value_error() -> None:
    model = _model(expressions={"A": "B", "B": "A"})
    with pytest.raises(ValueError) as exc_info:
        parse_linear_expression("A", model)
    message = str(exc_info.value)
    assert "A" in message
    assert "B" in message


def test_unbalanced_parentheses_raises_value_error() -> None:
    model = _model()
    with pytest.raises(ValueError, match="ger_usih") as exc_info:
        parse_linear_expression("ger_usih(261", model)
    # malformed syntax stays a plain ValueError, never the
    # well-formed-but-undeclared-identifier subclass.
    assert not isinstance(exc_info.value, UnrecognizedElectricalToken)


def test_non_numeric_function_argument_raises_value_error() -> None:
    model = _model()
    with pytest.raises(ValueError, match="abc") as exc_info:
        parse_linear_expression("ger_usih(abc)", model)
    assert not isinstance(exc_info.value, UnrecognizedElectricalToken)


def test_trailing_operator_with_no_operand_raises_plain_value_error() -> None:
    # a trailing '+' with no following operand is
    # malformed syntax, not an unrecognized identifier -- stays fail-loud as
    # a plain ValueError.
    model = _model()
    with pytest.raises(ValueError) as exc_info:
        parse_linear_expression("ger_usih(1) +", model)
    assert not isinstance(exc_info.value, UnrecognizedElectricalToken)


def test_unrecognized_bare_identifier_raises_unrecognized_electrical_token() -> None:
    # a well-formed but undeclared identifier raises the
    # dedicated subclass -- still an instance of ValueError, so any existing
    # pytest.raises(ValueError) contract keeps passing.
    model = _model()
    with pytest.raises(UnrecognizedElectricalToken, match="ZZTOP") as exc_info:
        parse_linear_expression("ZZTOP", model)
    assert isinstance(exc_info.value, ValueError)


def test_unrecognized_token_message_names_identifier_and_expression() -> None:
    # the motivating example -- an undeclared
    # peq_*gd_* MMGD token alongside a resolvable ger_usih(...) term. The
    # message must name both the offending identifier and the enclosing
    # expression, so a diagnostic built from it is actionable.
    model = _model()
    text = "peq_N_PCHgd_N + ger_usih(1)"
    with pytest.raises(UnrecognizedElectricalToken) as exc_info:
        parse_linear_expression(text, model)
    assert isinstance(exc_info.value, ValueError)
    message = str(exc_info.value)
    assert "peq_N_PCHgd_N" in message
    assert text in message


def test_se_preserved_as_one_structural_term() -> None:
    model = _model()
    terms = parse_linear_expression("se(demanda_sin > 50000, ger_usih(1), 0)", model)
    assert len(terms) == 1
    term = terms[0]
    assert term.token == "se"
    assert term.coefficient == 1.0
    assert term.args == ()
    assert term.condition == "demanda_sin > 50000"
    assert term.branch_true == "ger_usih(1)"
    assert term.branch_false == "0"


def test_se_condition_is_never_evaluated() -> None:
    # A condition referencing an undeclared identifier would raise if the
    # parser tried to evaluate/parse it as its own linear expression; it must
    # not, since se(...)'s cond is opaque, unparsed text.
    model = _model()
    terms = parse_linear_expression(
        "se(totally_unparseable !! syntax, ger_usih(1), ger_usih(2))", model
    )
    assert terms[0].condition == "totally_unparseable !! syntax"


def test_se_with_nested_parens_and_commas_in_branches() -> None:
    model = _model()
    terms = parse_linear_expression(
        "se(demanda(1) > 100, ger_usih(1) + ger_usih(2), ger_usih(3))", model
    )
    term = terms[0]
    assert term.condition == "demanda(1) > 100"
    assert term.branch_true == "ger_usih(1) + ger_usih(2)"
    assert term.branch_false == "ger_usih(3)"


# ---------------------------------------------------------------------------
# classify_term / classify_terms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    ["ger_usih", "ger_usit", "ger_pee", "ger_conjh", "ener_interc", "ener_comerc"],
)
def test_classify_term_bucket_a_decision_tokens(token: str) -> None:
    term = ParsedTerm(coefficient=1.0, token=token, args=(1,))
    assert classify_term(term) == Bucket.A


@pytest.mark.parametrize(
    "term",
    [
        ParsedTerm(coefficient=1.0, token="demanda", args=(1,)),
        ParsedTerm(coefficient=1.0, token="val_demanda", args=(1,)),
        ParsedTerm(coefficient=1.0, token="demanda_sin"),
        ParsedTerm(coefficient=1.0, token="val_demanda_sin"),
        ParsedTerm(coefficient=1.0, token="carga_ande"),
        ParsedTerm(coefficient=1.0, token="alias", alias_name="MMGDSECO"),
        ParsedTerm(coefficient=100.0, token="__const__"),
        ParsedTerm(
            coefficient=1.0,
            token="se",
            condition="demanda_sin > 50000",
            branch_true="ger_usih(1)",
            branch_false="0",
        ),
    ],
)
def test_classify_term_bucket_b_input_data_tokens(term: ParsedTerm) -> None:
    assert classify_term(term) == Bucket.B


def test_classify_term_bucket_c_disp_usih() -> None:
    term = ParsedTerm(coefficient=1.0, token="disp_usih", args=(261,))
    assert classify_term(term) == Bucket.C


def test_classify_term_unrecognized_token_raises_value_error() -> None:
    term = ParsedTerm(coefficient=1.0, token="ger_bogus", args=(1,))
    with pytest.raises(ValueError, match="ger_bogus"):
        classify_term(term)


def test_classify_terms_partitions_mixed_expression_into_a_b_c() -> None:
    model = _model()
    terms = parse_linear_expression(
        "disp_usih(261) - ger_usih(261) + 0.05*carga_ande", model
    )
    partition = classify_terms(terms)
    assert len(partition[Bucket.A]) == 1
    assert len(partition[Bucket.B]) == 1
    assert len(partition[Bucket.C]) == 1
    assert partition[Bucket.A][0].token == "ger_usih"
    assert partition[Bucket.B][0].token == "carga_ande"
    assert partition[Bucket.C][0].token == "disp_usih"


def test_classify_terms_present_empty_lists_for_absent_buckets() -> None:
    partition = classify_terms(
        [ParsedTerm(coefficient=1.0, token="ger_usih", args=(1,))]
    )
    assert partition[Bucket.A] == [
        ParsedTerm(coefficient=1.0, token="ger_usih", args=(1,))
    ]
    assert partition[Bucket.B] == []
    assert partition[Bucket.C] == []


def test_classify_terms_order_preserving_within_bucket() -> None:
    terms = [
        ParsedTerm(coefficient=1.0, token="ger_usih", args=(1,)),
        ParsedTerm(coefficient=1.0, token="ger_usit", args=(2,)),
        ParsedTerm(coefficient=1.0, token="ger_pee", args=(3,)),
    ]
    partition = classify_terms(terms)
    assert [term.args for term in partition[Bucket.A]] == [(1,), (2,), (3,)]


# ---------------------------------------------------------------------------
# parse_activation_rule / evaluate_rule / is_always_active
# ---------------------------------------------------------------------------


def _dict_context(values: Mapping[str, float]) -> DataContext:
    """A trivial dict-backed :data:`DataContext` test double.

    Keys are a bare token's own name (``"demanda_sin"``), or
    ``"token(args)"`` for a parameterized token (``"demanda(1)"``), or the
    alias's own name for an unresolved ``ALIAS-ELETRICO`` reference. A bare
    numeric constant always resolves to ``1.0`` (its value already lives in
    the term's own ``coefficient`` — see :data:`DataContext`), so it is never
    looked up in *values*.
    """

    def resolve(term: ParsedTerm) -> float:
        if term.token == "__const__":
            return 1.0
        if term.token == "alias":
            assert term.alias_name is not None
            return values[term.alias_name]
        key = (
            f"{term.token}({','.join(str(arg) for arg in term.args)})"
            if term.args
            else term.token
        )
        return values[key]

    return resolve


@pytest.mark.parametrize("op", ["<", "<=", ">", ">=", "==", "!="])
def test_parse_activation_rule_recognizes_each_operator_longest_match(
    op: str,
) -> None:
    model = _model()
    rule = parse_activation_rule(f"demanda_sin {op} 50000", model)
    assert len(rule.comparisons) == 1
    assert rule.comparisons[0].op == op


def test_parse_activation_rule_single_comparison() -> None:
    model = _model()
    rule = parse_activation_rule("demanda_sin >= 50000", model)
    assert rule == ActivationRule(
        comparisons=(
            Comparison(
                left=[ParsedTerm(coefficient=1.0, token="demanda_sin")],
                op=">=",
                right=[ParsedTerm(coefficient=50000.0, token="__const__")],
            ),
        )
    )


def test_parse_activation_rule_alias_and_sum_operands() -> None:
    model = _model(aliases=("MMGDSECO",))
    rule = parse_activation_rule("demanda(3)+demanda(4) < MMGDSECO", model)
    assert len(rule.comparisons) == 1
    comparison = rule.comparisons[0]
    assert comparison.op == "<"
    assert comparison.left == [
        ParsedTerm(coefficient=1.0, token="demanda", args=(3,)),
        ParsedTerm(coefficient=1.0, token="demanda", args=(4,)),
    ]
    assert comparison.right == [
        ParsedTerm(coefficient=1.0, token="alias", alias_name="MMGDSECO"),
    ]


def test_parse_activation_rule_multi_conjunct_splits_on_top_level_ampersand() -> None:
    model = _model()
    rule = parse_activation_rule("demanda_sin >= 50000 & demanda(1) < 1000", model)
    assert len(rule.comparisons) == 2
    first, second = rule.comparisons
    assert first == Comparison(
        left=[ParsedTerm(coefficient=1.0, token="demanda_sin")],
        op=">=",
        right=[ParsedTerm(coefficient=50000.0, token="__const__")],
    )
    assert second == Comparison(
        left=[ParsedTerm(coefficient=1.0, token="demanda", args=(1,))],
        op="<",
        right=[ParsedTerm(coefficient=1000.0, token="__const__")],
    )


def test_parse_activation_rule_bucket_a_operand_raises_value_error() -> None:
    model = _model()
    with pytest.raises(ValueError, match="ger_usih"):
        parse_activation_rule("ger_usih(5) >= 100", model)


def test_parse_activation_rule_conjunct_without_operator_raises_value_error() -> None:
    model = _model()
    with pytest.raises(ValueError, match="no comparison operator"):
        parse_activation_rule("demanda_sin 50000", model)


def test_parse_activation_rule_unrecognized_operator_raises_value_error() -> None:
    model = _model()
    with pytest.raises(ValueError, match="unrecognized comparison operator"):
        parse_activation_rule("demanda_sin <> 50000", model)


def test_parse_activation_rule_unbalanced_parens_raises_value_error() -> None:
    model = _model()
    with pytest.raises(ValueError, match="unbalanced parentheses"):
        parse_activation_rule("demanda(3 >= 5", model)


def test_evaluate_rule_and_of_comparisons_true() -> None:
    model = _model()
    rule = parse_activation_rule("demanda_sin >= 50000 & demanda(1) < 1000", model)
    context = _dict_context({"demanda_sin": 60000.0, "demanda(1)": 900.0})
    assert evaluate_rule(rule, context) is True


def test_evaluate_rule_and_of_comparisons_false_when_one_fails() -> None:
    model = _model()
    rule = parse_activation_rule("demanda_sin >= 50000 & demanda(1) < 1000", model)
    context = _dict_context({"demanda_sin": 60000.0, "demanda(1)": 1500.0})
    assert evaluate_rule(rule, context) is False


def test_evaluate_rule_empty_conjunction_is_always_true() -> None:
    rule = ActivationRule(comparisons=())
    assert evaluate_rule(rule, _dict_context({})) is True


def test_is_always_active_true_for_no_habilita_restriction() -> None:
    assert is_always_active(ActivationRule(comparisons=())) is True


def test_is_always_active_false_for_a_parsed_rule() -> None:
    model = _model()
    rule = parse_activation_rule("demanda_sin >= 50000", model)
    assert is_always_active(rule) is False


# ---------------------------------------------------------------------------
# read_carga_ande
# ---------------------------------------------------------------------------


class _StubDadger:
    """Return a preset ``ri`` DataFrame (or ``None``) — the module's first
    ``Dadger`` test double."""

    def __init__(self, ri: pd.DataFrame | None) -> None:
        self._ri = ri

    def ri(self, df: bool = True) -> pd.DataFrame | None:
        return self._ri


def _ri(*rows: tuple, n_patamares: int) -> pd.DataFrame:
    """``RI`` rows shaped exactly as idecomp's ``df=True`` accessor expands
    them: ``(codigo_usina, estagio, codigo_submercado, carga_ande_1, ...,
    carga_ande_n)``."""
    columns = ["codigo_usina", "estagio", "codigo_submercado"] + [
        f"carga_ande_{k}" for k in range(1, n_patamares + 1)
    ]
    return pd.DataFrame(rows, columns=columns)


def _ri_stage(index: int, *, n_blocks: int) -> OperativeStage:
    return OperativeStage(
        index=index,
        start_date=date(2026, 7, 4),
        end_date=date(2026, 7, 11),
        season_id=6,
        block_hours=tuple(24.0 for _ in range(n_blocks)),
    )


def _ri_calendar(n_stages: int, *, n_blocks: int) -> list[OperativeStage]:
    return [_ri_stage(i, n_blocks=n_blocks) for i in range(n_stages)]


def test_carga_ande_single_stage_inherited_forward_across_calendar() -> None:
    stub = _StubDadger(_ri((66, 1, 1, 3505.0, 3332.0, 2817.5), n_patamares=3))
    calendar = _ri_calendar(6, n_blocks=3)
    result = read_carga_ande(stub, calendar)
    assert set(result) == {0, 1, 2, 3, 4, 5}
    for values in result.values():
        assert values == [3505.0, 3332.0, 2817.5]


def test_carga_ande_multi_stage_sparse_inheritance() -> None:
    stub = _StubDadger(
        _ri((66, 1, 1, 100.0, 100.0), (66, 3, 1, 200.0, 200.0), n_patamares=2)
    )
    calendar = _ri_calendar(4, n_blocks=2)
    result = read_carga_ande(stub, calendar)
    assert result[0] == [100.0, 100.0]
    assert result[1] == [100.0, 100.0]
    assert result[2] == [200.0, 200.0]
    assert result[3] == [200.0, 200.0]


def test_carga_ande_no_ri_register_returns_empty_dict() -> None:
    stub = _StubDadger(None)
    assert read_carga_ande(stub, _ri_calendar(6, n_blocks=3)) == {}


def test_carga_ande_block_count_mismatch_raises_value_error() -> None:
    stub = _StubDadger(_ri((66, 2, 1, 111.0, 222.0), n_patamares=2))
    calendar = _ri_calendar(4, n_blocks=3)
    with pytest.raises(ValueError) as exc_info:
        read_carga_ande(stub, calendar)
    message = str(exc_info.value)
    # estagio 2 -> 0-based stage index 1
    assert "1" in message
    assert "2" in message
    assert "3" in message


def test_carga_ande_missing_estagio_1_raises_value_error() -> None:
    stub = _StubDadger(_ri((66, 3, 1, 200.0, 200.0), n_patamares=2))
    calendar = _ri_calendar(4, n_blocks=2)
    with pytest.raises(ValueError, match="estágio 1"):
        read_carga_ande(stub, calendar)


# ---------------------------------------------------------------------------
# build_data_context / assemble_bound / evaluate_se
# ---------------------------------------------------------------------------


class _StubElectricalDadger:
    """Return a preset ``dp``/``ri`` DataFrame (or ``None``) for each
    ``Dadger`` accessor ``build_data_context`` reads -- mirrors
    ``_StubRestricoes``'s generic-accessor shape."""

    def __init__(self, **frames: pd.DataFrame) -> None:
        self._frames = frames

    def __getattr__(self, name: str):  # noqa: ANN204 - test double
        if name.startswith("_"):
            raise AttributeError(name)

        def accessor(df: bool = True) -> pd.DataFrame | None:
            return self._frames.get(name)

        return accessor


def _case(
    dadger: _StubElectricalDadger, calendar: Sequence[OperativeStage]
) -> DecompCase:
    return make_decomp_case(Path("unused"), dadger=dadger, calendar=calendar)


def _dp(*rows: tuple, n_patamares: int) -> pd.DataFrame:
    """``DP`` rows shaped as idecomp's ``df=True`` accessor: (codigo_submercado,
    estagio, numero_patamares, carga_1, ..., carga_n)."""
    columns = ["codigo_submercado", "estagio", "numero_patamares"] + [
        f"carga_{k}" for k in range(1, n_patamares + 1)
    ]
    return pd.DataFrame(rows, columns=columns)


def test_build_data_context_demanda_sin_and_demanda_submarket() -> None:
    id_map = DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "S"))
    calendar = _ri_calendar(1, n_blocks=2)
    dadger = _StubElectricalDadger(
        dp=_dp(
            (1, 1, 2, 1000.0, 900.0),
            (2, 1, 2, 500.0, 400.0),
            n_patamares=2,
        )
    )
    ctx = build_data_context(_case(dadger, calendar), id_map, model=_model())(0, 0)
    assert ctx(ParsedTerm(coefficient=1.0, token="demanda_sin")) == 1500.0
    assert ctx(ParsedTerm(coefficient=1.0, token="demanda", args=(1,))) == 1000.0


def test_build_data_context_carga_ande_resolves_the_series() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    calendar = _ri_calendar(1, n_blocks=2)
    dadger = _StubElectricalDadger(
        dp=_dp((1, 1, 2, 0.0, 0.0), n_patamares=2),
        ri=_ri((66, 1, 1, 50.0, 60.0), n_patamares=2),
    )
    ctx0 = build_data_context(_case(dadger, calendar), id_map, model=_model())(0, 0)
    ctx1 = build_data_context(_case(dadger, calendar), id_map, model=_model())(0, 1)
    term = ParsedTerm(coefficient=1.0, token="carga_ande")
    assert ctx0(term) == 50.0
    assert ctx1(term) == 60.0


def test_build_data_context_alias_stage_inherited_with_na_patamar_fallback() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    calendar = _ri_calendar(3, n_blocks=2)
    dadger = _StubElectricalDadger(dp=_dp((1, 1, 2, 0.0, 0.0), n_patamares=2))
    model = _model(
        aliases=("MMGDSIN",),
        alias_values={"MMGDSIN": {(0, None): 100.0, (2, 1): 500.0}},
    )
    factory = build_data_context(_case(dadger, calendar), id_map, model=model)
    term = ParsedTerm(coefficient=1.0, token="alias", alias_name="MMGDSIN")
    # stage 0: no exact patamar entry, NA (all-blocks) covers both blocks.
    assert factory(0, 0)(term) == 100.0
    assert factory(0, 1)(term) == 100.0
    # stage 1: nothing declared -- inherits stage 0's NA entry.
    assert factory(1, 0)(term) == 100.0
    # stage 2: an exact patamar-1 entry overrides the inherited NA default.
    assert factory(2, 0)(term) == 500.0


def test_build_data_context_constant_resolves_to_one_not_its_value() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    calendar = _ri_calendar(1, n_blocks=1)
    dadger = _StubElectricalDadger(dp=_dp((1, 1, 1, 0.0), n_patamares=1))
    ctx = build_data_context(_case(dadger, calendar), id_map, model=_model())(0, 0)
    # the constant's own value lives in
    # the term's coefficient; ctx must return 1.0, never the value itself.
    assert ctx(ParsedTerm(coefficient=42.0, token="__const__")) == 1.0


def test_build_data_context_unknown_submarket_raises_unresolvable() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    calendar = _ri_calendar(1, n_blocks=1)
    dadger = _StubElectricalDadger(dp=_dp((1, 1, 1, 0.0), n_patamares=1))
    ctx = build_data_context(_case(dadger, calendar), id_map, model=_model())(0, 0)
    with pytest.raises(_UnresolvableBucketBTerm, match="99"):
        ctx(ParsedTerm(coefficient=1.0, token="demanda", args=(99,)))


def test_build_data_context_alias_with_no_value_raises_unresolvable() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    calendar = _ri_calendar(1, n_blocks=1)
    dadger = _StubElectricalDadger(dp=_dp((1, 1, 1, 0.0), n_patamares=1))
    model = _model(aliases=("MMGDSIN",))  # no values declared at all
    ctx = build_data_context(_case(dadger, calendar), id_map, model=model)(0, 0)
    with pytest.raises(_UnresolvableBucketBTerm, match="MMGDSIN"):
        ctx(ParsedTerm(coefficient=1.0, token="alias", alias_name="MMGDSIN"))


def _inequacao(
    code: int,
    lhs: str,
    operator: str,
    rhs: str,
    overrides: tuple[PeriodPatamarOverride, ...] = (),
) -> ElectricalRestriction:
    return ElectricalRestriction(
        code=code,
        lhs=lhs,
        operator=operator,
        rhs=rhs,
        is_formula=False,
        limits={},
        overrides=overrides,
    )


def _formula(
    code: int, lhs: str, limits: dict[tuple[int, int | None], tuple]
) -> ElectricalRestriction:
    return ElectricalRestriction(
        code=code,
        lhs=lhs,
        operator=None,
        rhs=None,
        is_formula=True,
        limits=limits,
        overrides=(),
    )


def test_assemble_bound_inequacao_upper_only_from_rhs_bucket_b() -> None:
    restriction = _inequacao(285, "ger_usih(285)", "<=", "11000 - 0.04*val_demanda(3)")
    ctx = _dict_context({"val_demanda(3)": 5000.0})
    assert assemble_bound(restriction, _model(), 0, 0, ctx) == AssembledBound(
        terms=(ParsedTerm(coefficient=1.0, token="ger_usih", args=(285,)),),
        lower=None,
        upper=10800.0,
    )


def test_assemble_bound_formula_two_sided_sentinel_maps_to_none() -> None:
    restriction = _formula(
        401, "ger_usih(285)+ger_usih(287)", {(0, None): (None, 6300.0)}
    )
    assert assemble_bound(
        restriction, _model(), 0, 0, _dict_context({})
    ) == AssembledBound(
        terms=(
            ParsedTerm(coefficient=1.0, token="ger_usih", args=(285,)),
            ParsedTerm(coefficient=1.0, token="ger_usih", args=(287,)),
        ),
        lower=None,
        upper=6300.0,
    )


def test_assemble_bound_negates_a_lhs_bucket_b_term_sign_discipline() -> None:
    # a bucket-B term on the LHS must be
    # negated when it moves to the bound side, not just summed in verbatim.
    # ger_usih(5) + val_demanda(1) >= 0, val_demanda(1) = 300 <=>
    # ger_usih(5) >= -300.
    restriction = _inequacao(42, "ger_usih(5) + val_demanda(1)", ">=", "0")
    ctx = _dict_context({"val_demanda(1)": 300.0})
    assert assemble_bound(restriction, _model(), 0, 0, ctx) == AssembledBound(
        terms=(ParsedTerm(coefficient=1.0, token="ger_usih", args=(5,)),),
        lower=-300.0,
        upper=None,
    )


def test_assemble_bound_moves_rhs_bucket_a_term_to_lhs_negated() -> None:
    # a plain INEQUACAO carrying a bucket-A term on the RHS (spec §2b's
    # +0.06*ger_pee(11)) must surface it on AssembledBound.terms with its
    # sign flipped -- reading restriction.rhs verbatim would reproduce the
    # A1 sign hazard for this shape too.
    restriction = _inequacao(403, "ger_usih(1)", "<=", "1000 + 0.06*ger_pee(11)")
    result = assemble_bound(restriction, _model(), 0, 0, _dict_context({}))
    assert result == AssembledBound(
        terms=(
            ParsedTerm(coefficient=1.0, token="ger_usih", args=(1,)),
            ParsedTerm(coefficient=-0.06, token="ger_pee", args=(11,)),
        ),
        lower=None,
        upper=1000.0,
    )


def test_assemble_bound_reserve_single_plant_sign_proof() -> None:
    # The A1 fix: disp_usih(1) - ger_usih(1) >= R, A_h(1) = 1000,
    # R = 300 -> ger_usih(1) <= 700, with a POSITIVE surviving coefficient --
    # a generation cap, never an inverted lower bound.
    restriction = _inequacao(407, "disp_usih(1) - ger_usih(1)", ">=", "300")
    a_h = AvailablePower(overlay={(1, 0): 1000.0}, rated_envelope={})
    result = assemble_bound(restriction, _model(), 0, 0, _dict_context({}), a_h)
    assert result == AssembledBound(
        terms=(ParsedTerm(coefficient=1.0, token="ger_usih", args=(1,)),),
        lower=None,
        upper=700.0,
    )


def test_assemble_bound_drops_and_warns_on_unresolvable_carga_ande() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    calendar = _ri_calendar(1, n_blocks=1)
    dadger = _StubElectricalDadger(dp=_dp((1, 1, 1, 1000.0), n_patamares=1))
    model = _model()
    cell_ctx = build_data_context(_case(dadger, calendar), id_map, model=model)(0, 0)
    restriction = _inequacao(999, "carga_ande", ">=", "100")

    with dx.collect() as sink:
        result = assemble_bound(restriction, model, 0, 0, cell_ctx)

    assert result is None
    assert len(sink) == 1
    assert sink[0].severity is dx.Severity.WARNING
    assert "carga_ande" in sink[0].summary
    assert "999" in sink[0].summary


# ---------------------------------------------------------------------------
# INEQUACAO-PERIODO-PATAMAR override precedence
# ---------------------------------------------------------------------------


def test_effective_inequacao_sides_no_matching_override_returns_base() -> None:
    restriction = _inequacao(500, "ger_usih(1)", "<=", "5000")
    assert _effective_inequacao_sides(restriction, 0, 1) == (
        "ger_usih(1)",
        "<=",
        "5000",
    )


def test_effective_inequacao_sides_later_declared_override_wins_tie() -> None:
    # Requirement 3: two rows tie on specificity (both exact-patamar 1) for
    # the same cell -- the later-declared one wins.
    first = PeriodPatamarOverride(0, 3, 1, "ger_usih(1)", "<=", "6000")
    second = PeriodPatamarOverride(0, 3, 1, "ger_usih(1)", "<=", "7000")
    restriction = _inequacao(
        501, "ger_usih(1)", "<=", "5000", overrides=(first, second)
    )
    assert _effective_inequacao_sides(restriction, 1, 1) == (
        "ger_usih(1)",
        "<=",
        "7000",
    )


def test_assemble_bound_period_patamar_override_applies_only_to_matching_cell() -> None:
    # base X<=5000, override for stages 1-2 patamar 1 (X<=6000). At
    # (stage 1, block 0 = patamar 1) the override matches; at (stage 1,
    # block 1 = patamar 2) no override matches this patamar, so the base
    # constant applies.
    override = PeriodPatamarOverride(1, 2, 1, "ger_usih(9)", "<=", "6000")
    restriction = _inequacao(502, "ger_usih(9)", "<=", "5000", overrides=(override,))
    model = _model()
    ctx = _dict_context({})

    overridden = assemble_bound(restriction, model, 1, 0, ctx)
    base_applies = assemble_bound(restriction, model, 1, 1, ctx)

    assert overridden is not None
    assert overridden.upper == 6000.0
    assert base_applies is not None
    assert base_applies.upper == 5000.0


def test_assemble_bound_exact_patamar_override_beats_na_patamar_override() -> None:
    # an exact-patamar override (patamar=1, X<=6000) and an NA-patamar
    # override (patamar=None, X<=5500) both cover the same stage -- the
    # exact-patamar one wins at patamar 1.
    exact = PeriodPatamarOverride(0, 5, 1, "ger_usih(9)", "<=", "6000")
    na = PeriodPatamarOverride(0, 5, None, "ger_usih(9)", "<=", "5500")
    restriction = _inequacao(503, "ger_usih(9)", "<=", "5000", overrides=(na, exact))
    model = _model()
    ctx = _dict_context({})

    result = assemble_bound(restriction, model, 2, 0, ctx)

    assert result is not None
    assert result.upper == 6000.0


def test_assemble_bound_period_patamar_override_flips_operator() -> None:
    # base X>=100, override X<=900 -- the returned interval must
    # reflect the override's operator (upper=900, lower=None), not the
    # base's ">=" (which would have set lower=100, upper=None).
    override = PeriodPatamarOverride(0, 1, None, "ger_usih(9)", "<=", "900")
    restriction = _inequacao(504, "ger_usih(9)", ">=", "100", overrides=(override,))
    model = _model()
    ctx = _dict_context({})

    result = assemble_bound(restriction, model, 0, 0, ctx)

    assert result is not None
    assert result.upper == 900.0
    assert result.lower is None


def test_assemble_bound_empty_overrides_matches_pre_ticket_base_fold() -> None:
    # an empty overrides tuple must be byte-identical to folding
    # restriction.lhs/.operator/.rhs verbatim (the pre-override behavior),
    # for a cell nowhere near any override range.
    restriction = _inequacao(505, "ger_usih(9)", "<=", "1000 - 0.02*val_demanda(1)")
    ctx = _dict_context({"val_demanda(1)": 400.0})
    model = _model()
    lhs_partition = classify_terms(parse_linear_expression(restriction.lhs, model))
    assert restriction.rhs is not None
    rhs_partition = classify_terms(parse_linear_expression(restriction.rhs, model))
    expected = _fold_inequacao_bound(restriction, ctx, lhs_partition, rhs_partition)

    assert assemble_bound(restriction, model, 3, 2, ctx) == expected


def test_evaluate_se_folds_a_pure_bucket_b_selected_branch() -> None:
    model = _model()
    term = parse_linear_expression("se(demanda_sin > 1000, val_demanda(1), 0)", model)[
        0
    ]
    ctx = _dict_context({"demanda_sin": 1500.0, "val_demanda(1)": 300.0})
    assert evaluate_se(term, model, 0, 0, ctx) == 300.0


def test_evaluate_se_raises_on_decision_bearing_selected_branch() -> None:
    # a se(...) selected branch carrying
    # a bucket-A/-C term cannot be folded to a float without silently
    # dropping that term from the LP -- fail loud instead.
    model = _model()
    term = parse_linear_expression("se(demanda_sin > 1000, ger_usih(1), 0)", model)[0]
    ctx = _dict_context({"demanda_sin": 1500.0})
    with pytest.raises(ValueError, match="ger_usih"):
        evaluate_se(term, model, 0, 0, ctx)


# ---------------------------------------------------------------------------
# active_cells
# ---------------------------------------------------------------------------


def _restriction_with_horizon(
    code: int,
    *,
    stage_start: int | None,
    stage_end: int | None,
    habilita: int | None,
) -> ElectricalRestriction:
    """A minimal INEQUACAO restriction carrying only the fields
    ``active_cells`` reads: ``habilita``/horizonte. ``lhs``/``operator``/
    ``rhs`` are irrelevant placeholders -- ``active_cells`` never parses
    them, only the activation rule text."""
    return ElectricalRestriction(
        code=code,
        lhs="ger_usih(1)",
        operator=">=",
        rhs="0",
        is_formula=False,
        limits={},
        overrides=(),
        stage_start=stage_start,
        stage_end=stage_end,
        habilita=habilita,
    )


def _demanda_sin_context_factory(
    values: Mapping[tuple[int, int], float],
) -> Callable[[int, int], DataContext]:
    """A per-cell :data:`DataContext` factory over a
    dict of ``demanda_sin`` values keyed by ``(stage_index, block_index)``;
    a cell absent from *values* defaults to ``0.0``."""

    def factory(stage_index: int, block_index: int) -> DataContext:
        return _dict_context(
            {"demanda_sin": values.get((stage_index, block_index), 0.0)}
        )

    return factory


def test_active_cells_band_gated_returns_single_cell() -> None:
    model = LibsElectricalModel(
        expressions={},
        aliases={},
        restrictions={},
        rules={10: "demanda_sin >= 50000"},
    )
    restriction = _restriction_with_horizon(
        701, stage_start=0, stage_end=1, habilita=10
    )
    calendar = _ri_calendar(2, n_blocks=2)
    factory = _demanda_sin_context_factory(
        {
            (0, 0): 60000.0,
            (0, 1): 40000.0,
            (1, 0): 40000.0,
            (1, 1): 40000.0,
        }
    )
    assert active_cells(restriction, model, factory, calendar) == {(0, 0)}


def test_active_cells_no_habilita_active_in_every_horizon_cell() -> None:
    model = LibsElectricalModel(expressions={}, aliases={}, restrictions={}, rules={})
    restriction = _restriction_with_horizon(
        702, stage_start=0, stage_end=1, habilita=None
    )
    calendar = _ri_calendar(2, n_blocks=2)
    # No cell data at all -- an always-active rule must never even consult
    # the context factory.
    factory = _demanda_sin_context_factory({})
    assert active_cells(restriction, model, factory, calendar) == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }


def test_active_cells_no_horizon_declared_defaults_to_full_calendar() -> None:
    model = LibsElectricalModel(expressions={}, aliases={}, restrictions={}, rules={})
    restriction = _restriction_with_horizon(
        705, stage_start=None, stage_end=None, habilita=None
    )
    calendar = _ri_calendar(2, n_blocks=2)
    factory = _demanda_sin_context_factory({})
    assert active_cells(restriction, model, factory, calendar) == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }


def test_active_cells_inactive_everywhere_returns_empty_and_emits_info() -> None:
    model = LibsElectricalModel(
        expressions={},
        aliases={},
        restrictions={},
        rules={10: "demanda_sin >= 50000"},
    )
    restriction = _restriction_with_horizon(
        703, stage_start=0, stage_end=1, habilita=10
    )
    calendar = _ri_calendar(2, n_blocks=2)
    # Every cell defaults to 0.0 -- the rule is false everywhere in-horizon.
    factory = _demanda_sin_context_factory({})

    with dx.collect() as sink:
        result = active_cells(restriction, model, factory, calendar)

    assert result == set()
    assert len(sink) == 1
    assert sink[0].severity is dx.Severity.INFO
    assert "703" in sink[0].summary


def test_active_cells_dangling_habilita_raises_value_error_naming_both() -> None:
    model = LibsElectricalModel(expressions={}, aliases={}, restrictions={}, rules={})
    restriction = _restriction_with_horizon(
        704, stage_start=0, stage_end=1, habilita=999
    )
    calendar = _ri_calendar(2, n_blocks=2)
    factory = _demanda_sin_context_factory({})

    with pytest.raises(ValueError) as exc_info:
        active_cells(restriction, model, factory, calendar)
    message = str(exc_info.value)
    assert "704" in message
    assert "999" in message


# ---------------------------------------------------------------------------
# AvailablePower / resolve_disp_usih / build_available_power
# ---------------------------------------------------------------------------


def test_resolve_disp_usih_returns_overlay_sum_when_present() -> None:
    a_h = AvailablePower(overlay={(261, 0): 1000.0}, rated_envelope={261: 500.0})
    term = ParsedTerm(coefficient=1.0, token="disp_usih", args=(261,))
    assert resolve_disp_usih(term, 0, a_h) == 1000.0


def test_resolve_disp_usih_returns_rated_envelope_fallback_when_absent() -> None:
    a_h = AvailablePower(overlay={}, rated_envelope={261: 1500.0})
    term = ParsedTerm(coefficient=1.0, token="disp_usih", args=(261,))
    assert resolve_disp_usih(term, 0, a_h) == 1500.0


def test_available_power_resolve_raises_disp_usih_unresolvable_from_both_tiers() -> (
    None
):
    a_h = AvailablePower(overlay={}, rated_envelope={})
    with pytest.raises(_UnresolvableDispUsih, match="999"):
        a_h.resolve(999, 0)


def test_build_available_power_for_disp_usih_sums_overlay_and_falls_back() -> None:
    # code 261 -> hydro_id 0 (two unit-groups summed at stage 0); code 262 ->
    # hydro_id 1, with only a max_turbined_m3s overlay entry (no generation
    # override) -- it must fall back to the (mocked) rated envelope, never
    # read the un-derated envelope as the primary source.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(261, 262))
    overlay = {
        (0, 0, 0): GroupBoundEntry(max_generation_mw=900.0),
        (0, 1, 0): GroupBoundEntry(max_generation_mw=100.0),
        (1, 0, 0): GroupBoundEntry(max_turbined_m3s=50.0),
    }
    hidr = pd.DataFrame(index=[261, 262])

    case = make_decomp_case(Path("unused"), hidr=hidr)
    with patch(
        "novomodelo_bridge.decomp.converters.libs_electrical._rated_envelope",
        return_value=(0.0, 1500.0),
    ):
        a_h = build_available_power(case, id_map, overlay=overlay, effective=object())

    assert a_h.overlay == {(261, 0): 1000.0}
    assert a_h.rated_envelope == {261: 1500.0, 262: 1500.0}
    assert a_h.resolve(261, 0) == (1000.0, AvailablePowerSource.OVERLAY)
    assert a_h.resolve(262, 0) == (1500.0, AvailablePowerSource.RATED_ENVELOPE)


# ---------------------------------------------------------------------------
# assemble_bound bucket-C disp_usih reserve -> gen-cap fold
# ---------------------------------------------------------------------------

_RESERVE_LHS = "disp_usih(261) + disp_usih(262) - ger_usih(261) - ger_usih(262)"


def test_assemble_bound_disp_usih_reserve_rewrites_to_gen_cap() -> None:
    # disp_usih(261)+disp_usih(262)-ger_usih(261)-ger_usih(262) >= 500,
    # A_h = {261: 1000, 262: 800} -> ger_usih(261)+ger_usih(262) <= 1300, with
    # both surviving ger_usih terms carrying a POSITIVE coefficient (the A1
    # fix: restriction.lhs itself carries them negative).
    restriction = _inequacao(402, _RESERVE_LHS, ">=", "500")
    a_h = AvailablePower(overlay={(261, 0): 1000.0, (262, 0): 800.0}, rated_envelope={})
    result = assemble_bound(restriction, _model(), 0, 0, _dict_context({}), a_h)
    assert result == AssembledBound(
        terms=(
            ParsedTerm(coefficient=1.0, token="ger_usih", args=(261,)),
            ParsedTerm(coefficient=1.0, token="ger_usih", args=(262,)),
        ),
        lower=None,
        upper=1300.0,
    )


def test_assemble_bound_disp_usih_emits_one_fidelity_diagnostic_per_substitution() -> (
    None
):
    restriction = _inequacao(402, _RESERVE_LHS, ">=", "500")
    a_h = AvailablePower(overlay={(261, 0): 1000.0, (262, 0): 800.0}, rated_envelope={})

    with dx.collect() as sink:
        result = assemble_bound(restriction, _model(), 0, 0, _dict_context({}), a_h)

    assert result is not None
    assert result.lower is None
    assert result.upper == 1300.0
    assert len(sink) == 2
    for diagnostic in sink:
        assert diagnostic.severity is dx.Severity.INFO

    first, second = sink
    assert "261" in first.summary
    assert "1000" in first.summary
    assert "stage 0" in first.summary
    assert "262" in second.summary
    assert "800" in second.summary
    assert "stage 0" in second.summary


def test_assemble_bound_disp_usih_uses_warning_severity_for_rated_envelope_fallback() -> (
    None
):
    restriction = _inequacao(285, "disp_usih(285) - ger_usih(285)", ">=", "0")
    a_h = AvailablePower(overlay={}, rated_envelope={285: 400.0})

    with dx.collect() as sink:
        assemble_bound(restriction, _model(), 0, 0, _dict_context({}), a_h)

    assert len(sink) == 1
    assert sink[0].severity is dx.Severity.WARNING
    assert "285" in sink[0].summary
    assert "400" in sink[0].summary


def test_assemble_bound_disp_usih_unresolvable_plant_drops_and_warns() -> None:
    restriction = _inequacao(403, "disp_usih(999) - ger_usih(999)", ">=", "100")
    a_h = AvailablePower(overlay={}, rated_envelope={})

    with dx.collect() as sink:
        result = assemble_bound(restriction, _model(), 0, 0, _dict_context({}), a_h)

    assert result is None
    assert len(sink) == 1
    assert sink[0].severity is dx.Severity.WARNING
    assert "999" in sink[0].summary
    assert "403" in sink[0].summary


def test_assemble_bound_disp_usih_without_available_power_raises_value_error() -> None:
    restriction = _inequacao(404, "disp_usih(1) - ger_usih(1)", ">=", "0")
    with pytest.raises(ValueError, match="AvailablePower"):
        assemble_bound(restriction, _model(), 0, 0, _dict_context({}))


def test_assemble_bound_disp_usih_on_rhs_raises_value_error() -> None:
    # CRITICAL pitfall guard: the reserve pattern requires disp_usih paired
    # with ger_usih on the LHS -- a disp_usih on the RHS is outside it.
    restriction = _inequacao(405, "ger_usih(1)", ">=", "disp_usih(1)")
    a_h = AvailablePower(overlay={(1, 0): 100.0}, rated_envelope={})
    with pytest.raises(ValueError, match="RHS"):
        assemble_bound(restriction, _model(), 0, 0, _dict_context({}), a_h)


def test_assemble_bound_disp_usih_unpaired_ger_usih_raises_value_error() -> None:
    # CRITICAL pitfall guard: a ger_usih term with no matching disp_usih is
    # outside the documented reserve pattern.
    restriction = _inequacao(406, "disp_usih(1) - ger_usih(1) + ger_usih(2)", ">=", "0")
    a_h = AvailablePower(overlay={(1, 0): 500.0}, rated_envelope={})
    with pytest.raises(ValueError, match="ger_usih\\(2\\)"):
        assemble_bound(restriction, _model(), 0, 0, _dict_context({}), a_h)


def test_assemble_bound_disp_usih_mismatched_coefficient_raises_value_error() -> None:
    # CRITICAL pitfall guard: the paired ger_usih coefficient must be the
    # exact opposite of disp_usih's -- a magnitude mismatch is outside the
    # documented sign structure, not silently rewritten.
    restriction = _inequacao(410, "disp_usih(1) - 2*ger_usih(1)", ">=", "0")
    a_h = AvailablePower(overlay={}, rated_envelope={})
    with pytest.raises(ValueError, match="disp_usih\\(1\\)"):
        assemble_bound(restriction, _model(), 0, 0, _dict_context({}), a_h)


def test_assemble_bound_disp_usih_non_ger_usih_decision_term_raises_value_error() -> (
    None
):
    # CRITICAL pitfall guard: the documented pattern only pairs disp_usih
    # with ger_usih -- a different bucket-A token alongside it is outside
    # the pattern, not silently folded.
    restriction = _inequacao(411, "disp_usih(1) - ger_usit(5)", ">=", "0")
    a_h = AvailablePower(overlay={(1, 0): 100.0}, rated_envelope={})
    with pytest.raises(ValueError, match="ger_usit"):
        assemble_bound(restriction, _model(), 0, 0, _dict_context({}), a_h)


def test_assemble_bound_formula_with_disp_usih_raises_value_error() -> None:
    # The documented reserve->gen-cap rewrite only covers the INEQUACAO
    # shape; a FORMULA restriction carrying disp_usih is out of pattern.
    restriction = _formula(420, "disp_usih(1)", {(0, None): (None, 100.0)})
    with pytest.raises(ValueError, match="FORMULA"):
        assemble_bound(restriction, _model(), 0, 0, _dict_context({}))


# ---------------------------------------------------------------------------
# the three folds share the harmonized pre-computed-partitions
# signature -- none re-parses restriction.lhs/.rhs a second time
# ---------------------------------------------------------------------------


def test_fold_formula_bound_never_reparses_lhs() -> None:
    # A malformed lhs would raise if _fold_formula_bound parsed it itself;
    # it must instead use only the pre-computed lhs_partition, exactly as
    # assemble_bound's own parse of "ger_usih(285)" would have classified it.
    restriction = _formula(
        421, "not a valid !! expression", {(0, None): (None, 6300.0)}
    )
    lhs_partition = classify_terms(parse_linear_expression("ger_usih(285)", _model()))
    result = _fold_formula_bound(restriction, 0, 1, _dict_context({}), lhs_partition)
    assert result == AssembledBound(
        terms=(ParsedTerm(coefficient=1.0, token="ger_usih", args=(285,)),),
        lower=None,
        upper=6300.0,
    )


def test_fold_inequacao_bound_never_reparses_lhs_or_rhs() -> None:
    restriction = _inequacao(
        422, "not a valid !! expression", "<=", "also not valid !!"
    )
    lhs_partition = classify_terms(parse_linear_expression("ger_usih(1)", _model()))
    rhs_partition = classify_terms(parse_linear_expression("500", _model()))
    result = _fold_inequacao_bound(
        restriction, _dict_context({}), lhs_partition, rhs_partition
    )
    assert result == AssembledBound(
        terms=(ParsedTerm(coefficient=1.0, token="ger_usih", args=(1,)),),
        lower=None,
        upper=500.0,
    )


def test_fold_reserve_disp_usih_takes_precomputed_partitions() -> None:
    # Pins the shared signature shape against the other two folds, now
    # that all three match.
    restriction = _inequacao(423, "disp_usih(1) - ger_usih(1)", ">=", "300")
    lhs_partition = classify_terms(
        parse_linear_expression("disp_usih(1) - ger_usih(1)", _model())
    )
    rhs_partition = classify_terms(parse_linear_expression("300", _model()))
    a_h = AvailablePower(overlay={(1, 0): 1000.0}, rated_envelope={})
    result = _fold_reserve_disp_usih(
        restriction, 0, a_h, _dict_context({}), lhs_partition, rhs_partition
    )
    assert result == AssembledBound(
        terms=(ParsedTerm(coefficient=1.0, token="ger_usih", args=(1,)),),
        lower=None,
        upper=700.0,
    )
