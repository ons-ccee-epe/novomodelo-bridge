"""Tier-1 tests for the LIBs-era electrical special-constraint reader.

Synthetic-fixture only — a fake ``Restricoes`` returns pandas DataFrames shaped
like idecomp's ``df=True`` accessors, so these run in CI with no real deck.
"""

from __future__ import annotations

import re
from datetime import date

import pandas as pd
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.decomp.converters.libs_electrical import (
    FormulaOverride,
    PeriodPatamarOverride,
    ViolationRow,
    _stage_of_date,
    _stage_range_from_dates,
    read_carga_ande,
    read_libs_electrical,
)
from novomodelo_bridge.decomp.temporal import OperativeStage
from tests.decomp.conftest import (
    SHORT_FORM_ACCESSORS,
    StubRestricoes,
    alias,
    alias_valor,
    calendar,
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


def test_expressions_indexed_by_name() -> None:
    stub = StubRestricoes(
        expressao_eletrica=expressao(
            (1, "FNESE", "ener_interc(3,1) - ener_interc(1,3)"),
            (2, "RNE", "ger_usih(10)"),
        ),
    )
    model = read_libs_electrical(stub, calendar(6))
    assert model is not None
    assert model.expressions["FNESE"].code == 1
    assert model.expressions["FNESE"].formula == "ener_interc(3,1) - ener_interc(1,3)"
    assert model.expressions["RNE"].code == 2
    assert model.expressions["RNE"].formula == "ger_usih(10)"


def test_aliases_with_period_patamar_values_including_na_patamar() -> None:
    stub = StubRestricoes(
        alias_eletrico=alias((1, "MMGDSIN")),
        alias_eletrico_valor_periodo_patamar=alias_valor(
            (1, 1, 2, 1.0, 100.0),
            (1, 3, 3, float("nan"), 250.0),
        ),
    )
    model = read_libs_electrical(stub, calendar(6))
    assert model is not None
    mmgdsin = model.aliases["MMGDSIN"]
    assert mmgdsin.code == 1
    # a two-stage range (estagio 1..2) expands to both 0-based stage indices
    assert mmgdsin.values[(0, 1)] == 100.0
    assert mmgdsin.values[(1, 1)] == 100.0
    # NA patamar -> None ("all blocks")
    assert mmgdsin.values[(2, None)] == 250.0


def test_formula_restriction_two_sided_limits_with_sentinel() -> None:
    stub = StubRestricoes(
        restricao_eletrica_formula=re_formula(
            (401, "ger_usih(285)+ger_usih(287)"),
        ),
        restricao_eletrica_horizonte_periodo=horizonte((401, 1, 6)),
        restricao_eletrica_limite_formula_periodo_patamar=re_lim_formula(
            (401, 1, 6, float("nan"), -1e31, 6300.0),
        ),
    )
    model = read_libs_electrical(stub, calendar(6))
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
    stub = StubRestricoes(
        restricao_eletrica_inequacao=re_inequacao(
            (402, "disp_usih(261) - ger_usih(261)", ">=", "0.05*(val_demanda(1))"),
        ),
    )
    model = read_libs_electrical(stub, calendar(6))
    assert model is not None
    restriction = model.restrictions[402]
    assert restriction.is_formula is False
    assert restriction.lhs == "disp_usih(261) - ger_usih(261)"
    assert restriction.operator == ">="
    assert restriction.rhs == "0.05*(val_demanda(1))"


def test_inequacao_period_patamar_overrides_read_verbatim() -> None:
    stub = StubRestricoes(
        restricao_eletrica_inequacao=re_inequacao(
            (406, "EXPNE", "<=", "14700"),
        ),
        restricao_eletrica_inequacao_periodo_patamar=re_inequacao_per_pat(
            (406, 1, 6, 1, "EXPNE", "<=", "14700"),
        ),
    )
    model = read_libs_electrical(stub, calendar(6))
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
    stub = StubRestricoes(
        restricao_eletrica_inequacao=re_inequacao(
            (501, "demanda(1)", ">=", "1000"),
        ),
        restricao_eletrica_habilita=habilita((501, 10)),
        restricao_eletrica_regra_ativacao=regra((10, "demanda(1) > 5000")),
        restricao_eletrica_tratamento_violacao=tratamento_violacao(
            (501, "hard", 1234.5)
        ),
    )
    model = read_libs_electrical(stub, calendar(6))
    assert model is not None
    restriction = model.restrictions[501]
    assert restriction.habilita == 10
    assert model.rules[10] == "demanda(1) > 5000"
    assert restriction.violations == (ViolationRow(None, None, "hard", 1234.5),)


def test_violation_period_row_maps_stages_kind_and_blank_cost() -> None:
    stub = StubRestricoes(
        restricao_eletrica_inequacao=re_inequacao((501, "demanda(1)", ">=", "1000")),
        restricao_eletrica_tratamento_violacao_periodo=tratamento_violacao_periodo(
            (501, 2, 4, "Soft", float("nan")),
        ),
    )
    model = read_libs_electrical(stub, calendar(6))
    assert model is not None
    assert model.restrictions[501].violations == (ViolationRow(1, 3, "soft", None),)


def test_violation_rows_whole_horizon_first_then_period_in_file_order() -> None:
    stub = StubRestricoes(
        restricao_eletrica_formula=re_formula((401, "ger_usih(285)")),
        restricao_eletrica_inequacao=re_inequacao((501, "demanda(1)", ">=", "1000")),
        restricao_eletrica_tratamento_violacao_periodo=tratamento_violacao_periodo(
            (501, 3, 3, "hard", 10.0),
            (401, 1, 6, "soft", 5.0),
            (501, 1, 2, "soft", float("nan")),
        ),
        restricao_eletrica_tratamento_violacao=tratamento_violacao(
            (501, "hard", 1234.5),
            (401, "soft", float("nan")),
            (501, "soft", 99.0),
        ),
    )
    model = read_libs_electrical(stub, calendar(6))
    assert model is not None
    assert model.restrictions[501].violations == (
        ViolationRow(None, None, "hard", 1234.5),
        ViolationRow(None, None, "soft", 99.0),
        ViolationRow(2, 2, "hard", 10.0),
        ViolationRow(0, 1, "soft", None),
    )
    assert model.restrictions[401].violations == (
        ViolationRow(None, None, "soft", None),
        ViolationRow(0, 5, "soft", 5.0),
    )


def test_violation_rows_of_unknown_restriction_are_not_attached() -> None:
    stub = StubRestricoes(
        restricao_eletrica_inequacao=re_inequacao((501, "demanda(1)", ">=", "1000")),
        restricao_eletrica_tratamento_violacao=tratamento_violacao((777, "hard", 1.0)),
    )
    model = read_libs_electrical(stub, calendar(6))
    assert model is not None
    assert 777 not in model.restrictions
    assert model.restrictions[501].violations == ()


def test_restriction_without_violation_rows_has_empty_violations() -> None:
    stub = StubRestricoes(
        restricao_eletrica_inequacao=re_inequacao((501, "demanda(1)", ">=", "1000")),
    )
    model = read_libs_electrical(stub, calendar(6))
    assert model is not None
    restriction = model.restrictions[501]
    assert restriction.violations == ()


def test_violation_rows_attach_to_period_only_inequacao_restriction() -> None:
    stub = StubRestricoes(
        restricao_eletrica_inequacao_periodo_patamar=re_inequacao_per_pat(
            (406, 1, 6, 1, "EXPNE", "<=", "14700"),
        ),
        restricao_eletrica_tratamento_violacao=tratamento_violacao((406, "hard", 7.0)),
    )
    model = read_libs_electrical(stub, calendar(6))
    assert model is not None
    assert model.restrictions[406].violations == (
        ViolationRow(None, None, "hard", 7.0),
    )


@pytest.mark.parametrize("tipo", ["teste", float("nan")])
def test_violation_unknown_tipo_violacao_raises(tipo: object) -> None:
    stub = StubRestricoes(
        restricao_eletrica_inequacao=re_inequacao((501, "demanda(1)", ">=", "1000")),
        restricao_eletrica_tratamento_violacao=tratamento_violacao((501, tipo, 1.0)),
    )
    with pytest.raises(ValueError, match="unknown tipo_violacao") as exc_info:
        read_libs_electrical(stub, calendar(6))
    assert "501" in str(exc_info.value)


def test_violation_period_unknown_tipo_violacao_raises() -> None:
    stub = StubRestricoes(
        restricao_eletrica_inequacao=re_inequacao((501, "demanda(1)", ">=", "1000")),
        restricao_eletrica_tratamento_violacao_periodo=tratamento_violacao_periodo(
            (501, 1, 2, "teste", 1.0),
        ),
    )
    with pytest.raises(ValueError, match="unknown tipo_violacao") as exc_info:
        read_libs_electrical(stub, calendar(6))
    assert "501" in str(exc_info.value)


def test_violation_period_outside_calendar_raises() -> None:
    stub = StubRestricoes(
        restricao_eletrica_inequacao=re_inequacao((501, "demanda(1)", ">=", "1000")),
        restricao_eletrica_tratamento_violacao_periodo=tratamento_violacao_periodo(
            (501, 4, 9, "hard", 1.0),
        ),
    )
    with pytest.raises(ValueError, match="outside the calendar") as exc_info:
        read_libs_electrical(stub, calendar(6))
    assert "501" in str(exc_info.value)


def test_horizonte_period_out_of_range_raises_value_error() -> None:
    stub = StubRestricoes(
        restricao_eletrica_inequacao=re_inequacao(
            (99, "ger_usih(1)", ">=", "0"),
        ),
        restricao_eletrica_horizonte_periodo=horizonte((99, 1, 7)),
    )
    with pytest.raises(ValueError) as exc_info:
        read_libs_electrical(stub, calendar(6))
    message = str(exc_info.value)
    assert "7" in message
    assert "6" in message


def test_no_electrical_cards_returns_none() -> None:
    stub = StubRestricoes()
    assert read_libs_electrical(stub, calendar(6)) is None


def _every_card_with_a_short_form() -> dict[str, pd.DataFrame]:
    return {
        "expressao_eletrica": expressao((1, "FNESE", "ener_interc(3,1)")),
        "alias_eletrico": alias((1, "MMGDSIN")),
        "alias_eletrico_valor_periodo_patamar": alias_valor((1, 1, 2, 1.0, 100.0)),
        "restricao_eletrica_formula": re_formula(
            (401, "ger_usih(285)"), (403, "ger_usih(287)")
        ),
        "restricao_eletrica_horizonte_periodo": horizonte((401, 1, 6)),
        "restricao_eletrica_horizonte_data": horizonte_data(
            (403, "2026-07-11", "2026-07-24")
        ),
        "restricao_eletrica_limite_formula_periodo_patamar": re_lim_formula(
            (401, 1, 6, float("nan"), -1e31, 6300.0)
        ),
        "restricao_eletrica_limites_formula_data_patamar": re_lim_formula_data(
            (403, "2026-07-04", "2026-07-11", 2, 10.0, 1e31)
        ),
        "restricao_eletrica_formula_periodo_patamar": re_formula_per_pat(
            (401, 1, 6, None, "2*ger_usih(285)")
        ),
        "restricao_eletrica_formula_data_patamar": re_formula_data_pat(
            (403, "2026-07-11", "2026-07-24", 2, "ger_usih(287)")
        ),
        "restricao_eletrica_inequacao": re_inequacao(
            (402, "disp_usih(261)", ">=", "0.05*(val_demanda(1))")
        ),
        "restricao_eletrica_habilita": habilita((402, 10)),
        "restricao_eletrica_regra_ativacao": regra((10, "demanda(1) > 5000")),
        "restricao_eletrica_tratamento_violacao": tratamento_violacao(
            (402, "hard", 1234.5)
        ),
        "restricao_eletrica_tratamento_violacao_periodo": tratamento_violacao_periodo(
            (401, 2, 4, "soft", float("nan"))
        ),
    }


def test_short_form_cards_read_as_their_long_form_equivalents() -> None:
    frames = _every_card_with_a_short_form()
    assert set(SHORT_FORM_ACCESSORS) <= set(frames)
    long_model = read_libs_electrical(spelled(frames, short=False), weekly_calendar(6))
    short_model = read_libs_electrical(spelled(frames, short=True), weekly_calendar(6))
    assert short_model is not None
    assert short_model == long_model
    assert short_model.aliases["MMGDSIN"].values == {(0, 1): 100.0, (1, 1): 100.0}
    assert short_model.rules == {10: "demanda(1) > 5000"}
    assert short_model.restrictions[401].stage_end == 5
    assert short_model.restrictions[401].limits[(0, None)] == (None, 6300.0)
    assert short_model.restrictions[403].stage_start == 1
    assert short_model.restrictions[403].limits == {
        (0, 2): (10.0, None),
        (1, 2): (10.0, None),
    }
    assert short_model.restrictions[401].violations == (
        ViolationRow(1, 3, "soft", None),
    )
    assert short_model.restrictions[401].formula_overrides == (
        FormulaOverride(0, 5, None, "2*ger_usih(285)"),
    )
    assert short_model.restrictions[403].formula_overrides == (
        FormulaOverride(1, 2, 2, "ger_usih(287)"),
    )
    assert short_model.restrictions[402].habilita == 10
    assert short_model.restrictions[402].violations == (
        ViolationRow(None, None, "hard", 1234.5),
    )


@pytest.mark.filterwarnings("error")
def test_long_and_short_form_restrictions_are_read_together() -> None:
    stub = StubRestricoes(
        restricao_eletrica_formula=re_formula((401, "ger_usih(285)")),
        re=re_formula((402, "ger_usih(287)")),
        restricao_eletrica_limite_formula_periodo_patamar=re_lim_formula(
            (401, 1, 6, float("nan"), -1e31, 6300.0)
        ),
        re_lim_form_per_pat=re_lim_formula((402, 1, 6, 1, 0.0, 100.0)),
    )
    model = read_libs_electrical(stub, calendar(6))
    assert model is not None
    assert sorted(model.restrictions) == [401, 402]
    assert model.restrictions[401].limits[(0, None)] == (None, 6300.0)
    assert model.restrictions[402].limits[(0, 1)] == (0.0, 100.0)


def test_short_form_only_deck_is_not_empty() -> None:
    stub = StubRestricoes(re=re_formula((401, "ger_usih(285)")))
    model = read_libs_electrical(stub, calendar(6))
    assert model is not None
    assert model.restrictions[401].lhs == "ger_usih(285)"


@pytest.mark.filterwarnings("error")
def test_empty_short_form_frame_is_ignored() -> None:
    long_only = StubRestricoes(restricao_eletrica_formula=re_formula((401, "A")))
    with_empty_alias = StubRestricoes(
        restricao_eletrica_formula=re_formula((401, "A")),
        re=re_formula(),
    )
    assert read_libs_electrical(with_empty_alias, calendar(6)) == read_libs_electrical(
        long_only, calendar(6)
    )


def test_code_in_both_spellings_takes_the_short_form_row() -> None:
    stub = StubRestricoes(
        restricao_eletrica_formula=re_formula((401, "A")),
        re=re_formula((401, "B")),
    )
    model = read_libs_electrical(stub, calendar(6))
    assert model is not None
    assert model.restrictions[401].lhs == "B"


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
    stages = _ri_calendar(6, n_blocks=3)
    result = read_carga_ande(stub, stages)
    assert set(result) == {0, 1, 2, 3, 4, 5}
    for values in result.values():
        assert values == [3505.0, 3332.0, 2817.5]


def test_carga_ande_multi_stage_sparse_inheritance() -> None:
    stub = _StubDadger(
        _ri((66, 1, 1, 100.0, 100.0), (66, 3, 1, 200.0, 200.0), n_patamares=2)
    )
    stages = _ri_calendar(4, n_blocks=2)
    result = read_carga_ande(stub, stages)
    assert result[0] == [100.0, 100.0]
    assert result[1] == [100.0, 100.0]
    assert result[2] == [200.0, 200.0]
    assert result[3] == [200.0, 200.0]


def test_carga_ande_no_ri_register_returns_empty_dict() -> None:
    stub = _StubDadger(None)
    assert read_carga_ande(stub, _ri_calendar(6, n_blocks=3)) == {}


def test_carga_ande_block_count_mismatch_raises_value_error() -> None:
    stub = _StubDadger(_ri((66, 2, 1, 111.0, 222.0), n_patamares=2))
    stages = _ri_calendar(4, n_blocks=3)
    with pytest.raises(ValueError) as exc_info:
        read_carga_ande(stub, stages)
    message = str(exc_info.value)
    # estagio 2 -> 0-based stage index 1
    assert "1" in message
    assert "2" in message
    assert "3" in message


def test_carga_ande_missing_estagio_1_raises_value_error() -> None:
    stub = _StubDadger(_ri((66, 3, 1, 200.0, 200.0), n_patamares=2))
    stages = _ri_calendar(4, n_blocks=2)
    with pytest.raises(ValueError, match="estágio 1"):
        read_carga_ande(stub, stages)


# ---------------------------------------------------------------------------
# inequalities declared only by RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR
# ---------------------------------------------------------------------------


def test_read_inequacao_declared_only_by_period_rows() -> None:
    stub = StubRestricoes(
        restricao_eletrica_inequacao_periodo_patamar=re_inequacao_per_pat(
            (449, 1, 7, None, "FSILE", "<=", "0.45*(C_AM_cMMGD - MMGD_AMexp)"),
        ),
        restricao_eletrica_horizonte_periodo=horizonte((449, 1, 7)),
        restricao_eletrica_habilita=habilita((449, 524)),
    )
    model = read_libs_electrical(stub, calendar(7))
    assert model is not None
    restriction = model.restrictions[449]
    assert restriction.is_formula is False
    assert (restriction.lhs, restriction.operator, restriction.rhs) == (
        None,
        None,
        None,
    )
    assert restriction.overrides == (
        PeriodPatamarOverride(
            0, 6, None, "FSILE", "<=", "0.45*(C_AM_cMMGD - MMGD_AMexp)"
        ),
    )
    assert (restriction.stage_start, restriction.stage_end) == (0, 6)
    assert restriction.habilita == 524


def _unused_period_rows(sink: list[dx.Diagnostic]) -> list[list[object]]:
    """The rows of the one period-rows-unused diagnostic in *sink*."""
    [unused] = [d for d in sink if d.code == "decomp-electrical-period-rows-unused"]
    assert unused.severity is dx.Severity.WARNING
    assert unused.table is not None
    assert unused.table.columns == ["Card", "Restriction"]
    return [list(row) for row in unused.table.rows]


def test_read_period_rows_not_matching_the_card_shape_are_reported() -> None:
    stub = StubRestricoes(
        restricao_eletrica_formula=re_formula((401, "ger_usih(285)")),
        restricao_eletrica_inequacao_periodo_patamar=re_inequacao_per_pat(
            (401, 1, 1, None, "ger_usih(285)", "<=", "10"),
            (449, 1, 1, None, "ger_usih(1)", "<=", "10"),
        ),
        restricao_eletrica_limite_formula_periodo_patamar=re_lim_formula(
            (449, 1, 1, None, -1e31, 5.0),
        ),
    )
    with dx.collect() as sink:
        model = read_libs_electrical(stub, calendar(1))
    assert model is not None
    assert model.restrictions[401].overrides == ()
    assert _unused_period_rows(sink) == [
        ["RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR", 401],
        ["RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR/-DATA-PATAMAR", 449],
    ]


# ---------------------------------------------------------------------------
# date-indexed horizonte and limit cards
# ---------------------------------------------------------------------------


#: Stages starting 2026-07-04, 2026-07-11 and 2026-07-18; the last ends 2026-07-25
#: (exclusive).
_THREE_WEEKS = weekly_calendar(3)


def _stage_range(start: object, end: object) -> tuple[int, int]:
    return _stage_range_from_dates(start, end, _THREE_WEEKS, "ctx", card="CARD")


def test_date_range_end_is_inclusive() -> None:
    assert _stage_range("2026-07-06", "2026-07-11") == (0, 1)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2026-07-11 00:00", "2026-07-24", (1, 2)),
        ("2026-07-04", "2026-07-04", (0, 0)),
        ("2026-07-04", "2026-07-24 23:59", (0, 2)),
        ("2026-07-10 23:59", "2026-07-11 08:30", (0, 1)),
    ],
    ids=["start-on-first-day", "single-day", "whole-horizon", "time-of-day-ignored"],
)
def test_date_range_runs_from_the_stage_of_its_start_to_the_stage_of_its_end(
    start: str, end: str, expected: tuple[int, int]
) -> None:
    assert _stage_range(start, end) == expected


@pytest.mark.parametrize("day", ["2026-07-03", "2026-07-25"])
def test_date_outside_the_calendar_raises_with_the_calendar_span(day: str) -> None:
    message = rf"ctx: data {day} outside the calendar \(2026-07-04\.\.2026-07-25\)"
    with pytest.raises(ValueError, match=message):
        _stage_of_date(day, _THREE_WEEKS, "ctx")


def test_date_range_inverted_inside_one_stage_raises() -> None:
    with pytest.raises(ValueError, match="after data_fim") as exc_info:
        _stage_range("2026-07-06", "2026-07-05")
    assert "CARD" in str(exc_info.value)


def test_blank_date_raises_naming_the_restriction() -> None:
    stub = StubRestricoes(
        restricao_eletrica_formula=re_formula((401, "ger_usih(285)")),
        restricao_eletrica_horizonte_data=horizonte_data((401, None, "2026-07-24")),
    )
    with pytest.raises(ValueError, match="restriction 401 horizonte: date is blank"):
        read_libs_electrical(stub, _THREE_WEEKS)


def test_horizonte_date_past_the_calendar_raises_naming_the_restriction() -> None:
    stub = StubRestricoes(
        restricao_eletrica_formula=re_formula((401, "ger_usih(285)")),
        restricao_eletrica_horizonte_data=horizonte_data(
            (401, "2026-07-11", "2026-07-25")
        ),
    )
    with pytest.raises(
        ValueError,
        match="restriction 401 horizonte: data 2026-07-25 outside the calendar",
    ):
        read_libs_electrical(stub, _THREE_WEEKS)


def _date_deck(*, short: bool) -> StubRestricoes:
    horizonte_accessor, limits_accessor = (
        ("re_horiz_data", "re_lim_form_data_pat")
        if short
        else (
            "restricao_eletrica_horizonte_data",
            "restricao_eletrica_limites_formula_data_patamar",
        )
    )
    return StubRestricoes(
        restricao_eletrica_formula=re_formula((401, "ger_usih(285)")),
        **{
            horizonte_accessor: horizonte_data((401, "2026-07-11", "2026-07-24")),
            limits_accessor: re_lim_formula_data(
                (401, "2026-07-04", "2026-07-11", 2, 10.0, 1e31)
            ),
        },
    )


def test_date_cards_read_as_the_period_cards_of_the_same_deck() -> None:
    period_deck = StubRestricoes(
        restricao_eletrica_formula=re_formula((401, "ger_usih(285)")),
        restricao_eletrica_horizonte_periodo=horizonte((401, 2, 3)),
        restricao_eletrica_limite_formula_periodo_patamar=re_lim_formula(
            (401, 1, 2, 2, 10.0, 1e31)
        ),
    )
    model = read_libs_electrical(_date_deck(short=False), _THREE_WEEKS)
    assert model is not None
    restriction = model.restrictions[401]
    assert (restriction.stage_start, restriction.stage_end) == (1, 2)
    assert restriction.limits == {(0, 2): (10.0, None), (1, 2): (10.0, None)}
    assert model == read_libs_electrical(period_deck, _THREE_WEEKS)


def test_short_form_date_cards_read_as_their_long_form_equivalents() -> None:
    long_model = read_libs_electrical(_date_deck(short=False), _THREE_WEEKS)
    short_model = read_libs_electrical(_date_deck(short=True), _THREE_WEEKS)
    assert short_model is not None
    assert short_model == long_model
    assert short_model.restrictions[401].stage_start == 1


def test_date_horizonte_row_follows_the_period_row_of_the_same_restriction() -> None:
    stub = StubRestricoes(
        restricao_eletrica_formula=re_formula((401, "ger_usih(285)")),
        restricao_eletrica_horizonte_periodo=horizonte((401, 1, 3)),
        restricao_eletrica_horizonte_data=horizonte_data(
            (401, "2026-07-11", "2026-07-24")
        ),
    )
    model = read_libs_electrical(stub, _THREE_WEEKS)
    assert model is not None
    restriction = model.restrictions[401]
    assert (restriction.stage_start, restriction.stage_end) == (1, 2)


def test_date_limit_row_follows_the_period_row_of_the_same_cell() -> None:
    stub = StubRestricoes(
        restricao_eletrica_formula=re_formula((401, "ger_usih(285)")),
        restricao_eletrica_limite_formula_periodo_patamar=re_lim_formula(
            (401, 1, 3, 1, 1.0, 2.0)
        ),
        restricao_eletrica_limites_formula_data_patamar=re_lim_formula_data(
            (401, "2026-07-11", "2026-07-11", 1, 5.0, 6.0)
        ),
    )
    model = read_libs_electrical(stub, _THREE_WEEKS)
    assert model is not None
    assert model.restrictions[401].limits == {
        (0, 1): (1.0, 2.0),
        (1, 1): (5.0, 6.0),
        (2, 1): (1.0, 2.0),
    }


def test_limit_rows_by_date_on_a_non_formula_restriction_are_reported() -> None:
    stub = StubRestricoes(
        restricao_eletrica_inequacao=re_inequacao((402, "A", "<=", "0")),
        restricao_eletrica_limites_formula_data_patamar=re_lim_formula_data(
            (402, "2026-07-04", "2026-07-11", None, -1e31, 5.0)
        ),
    )
    with dx.collect() as sink:
        model = read_libs_electrical(stub, _THREE_WEEKS)
    assert model is not None
    assert _unused_period_rows(sink) == [
        ["RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR/-DATA-PATAMAR", 402]
    ]


# ---------------------------------------------------------------------------
# formula cards by período and by date
# ---------------------------------------------------------------------------


def test_formula_overrides_read_period_rows_then_date_rows_verbatim() -> None:
    stub = StubRestricoes(
        restricao_eletrica_formula=re_formula((401, "ger_usih(261)")),
        restricao_eletrica_formula_periodo_patamar=re_formula_per_pat(
            (401, 1, 2, None, "2*ger_usih(261)  + 1")
        ),
        restricao_eletrica_formula_data_patamar=re_formula_data_pat(
            (401, "2026-07-11", "2026-07-18", 2, "ger_usih(261)")
        ),
    )
    model = read_libs_electrical(stub, _THREE_WEEKS)
    assert model is not None
    restriction = model.restrictions[401]
    assert restriction.formula_overrides == (
        FormulaOverride(0, 1, None, "2*ger_usih(261)  + 1"),
        FormulaOverride(1, 2, 2, "ger_usih(261)"),
    )
    assert restriction.overrides == ()
    assert restriction.lhs == "ger_usih(261)"


def test_formula_override_rows_keep_file_order_within_a_card() -> None:
    stub = StubRestricoes(
        restricao_eletrica_formula=re_formula((401, "A")),
        restricao_eletrica_formula_periodo_patamar=re_formula_per_pat(
            (401, 3, 3, 1, "C"),
            (401, 1, 1, 1, "B"),
        ),
    )
    model = read_libs_electrical(stub, _THREE_WEEKS)
    assert model is not None
    assert [o.formula for o in model.restrictions[401].formula_overrides] == ["C", "B"]


def test_formula_overrides_alone_declare_a_formula_restriction() -> None:
    stub = StubRestricoes(
        re_per_pat=re_formula_per_pat((402, 1, 1, 1, "ger_usih(261)")),
        restricao_eletrica_limite_formula_periodo_patamar=re_lim_formula(
            (402, 1, 1, 1, 0.0, 100.0)
        ),
    )
    with dx.collect() as sink:
        model = read_libs_electrical(stub, calendar(1))
    assert model is not None
    restriction = model.restrictions[402]
    assert restriction.is_formula is True
    assert restriction.lhs is None
    assert restriction.limits == {(0, 1): (0.0, 100.0)}
    assert restriction.formula_overrides == (FormulaOverride(0, 0, 1, "ger_usih(261)"),)
    assert not [d for d in sink if d.code == "decomp-electrical-period-rows-unused"]


def test_formula_override_rows_alone_make_a_non_empty_model() -> None:
    stub = StubRestricoes(
        restricao_eletrica_formula_data_patamar=re_formula_data_pat(
            (402, "2026-07-04", "2026-07-04", None, "A")
        )
    )
    model = read_libs_electrical(stub, _THREE_WEEKS)
    assert model is not None
    assert model.restrictions[402].formula_overrides == (
        FormulaOverride(0, 0, None, "A"),
    )


def test_formula_override_rows_on_an_inequacao_restriction_are_reported() -> None:
    stub = StubRestricoes(
        restricao_eletrica_inequacao=re_inequacao((403, "A", "<=", "0")),
        restricao_eletrica_formula_periodo_patamar=re_formula_per_pat(
            (403, 1, 1, None, "B")
        ),
    )
    with dx.collect() as sink:
        model = read_libs_electrical(stub, calendar(1))
    assert model is not None
    assert model.restrictions[403].formula_overrides == ()
    assert _unused_period_rows(sink) == [
        ["RESTRICAO-ELETRICA-FORMULA-PERIODO-PATAMAR/-DATA-PATAMAR", 403]
    ]


def test_inequality_rows_beside_formula_override_rows_are_reported() -> None:
    stub = StubRestricoes(
        restricao_eletrica_formula_periodo_patamar=re_formula_per_pat(
            (404, 1, 1, None, "B")
        ),
        restricao_eletrica_inequacao_periodo_patamar=re_inequacao_per_pat(
            (404, 1, 1, None, "A", "<=", "0")
        ),
    )
    with dx.collect() as sink:
        model = read_libs_electrical(stub, calendar(1))
    assert model is not None
    restriction = model.restrictions[404]
    assert restriction.is_formula is True
    assert restriction.overrides == ()
    assert _unused_period_rows(sink) == [
        ["RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR", 404]
    ]


def test_formula_override_period_past_the_calendar_names_the_restriction() -> None:
    stub = StubRestricoes(
        restricao_eletrica_formula=re_formula((401, "A")),
        restricao_eletrica_formula_periodo_patamar=re_formula_per_pat(
            (401, 2, 4, 1, "B")
        ),
    )
    with pytest.raises(
        ValueError,
        match="restriction 401 formula override: período 4 outside the calendar",
    ):
        read_libs_electrical(stub, _THREE_WEEKS)


def test_formula_override_date_past_the_calendar_names_the_restriction() -> None:
    stub = StubRestricoes(
        restricao_eletrica_formula=re_formula((401, "A")),
        restricao_eletrica_formula_data_patamar=re_formula_data_pat(
            (401, "2026-07-18", "2026-07-25", 1, "B")
        ),
    )
    with pytest.raises(
        ValueError,
        match="restriction 401 formula override: data 2026-07-25 outside the calendar",
    ):
        read_libs_electrical(stub, _THREE_WEEKS)


# ---------------------------------------------------------------------------
# a stage or date range that ends before it starts
# ---------------------------------------------------------------------------

_INVERTED_RANGES = {
    "alias value": (
        {
            "alias_eletrico": alias((7, "MMGDSIN")),
            "alias_eletrico_valor_periodo_patamar": alias_valor((7, 3, 1, 1, 1.0)),
        },
        "alias 7 value (ALIAS-ELETRICO-VALOR-PERIODO-PATAMAR): "
        "estagio_inicio 3 after estagio_fim 1",
    ),
    "horizonte by period": (
        {
            "restricao_eletrica_formula": re_formula((401, "A")),
            "restricao_eletrica_horizonte_periodo": horizonte((401, 3, 1)),
        },
        "restriction 401 horizonte (RESTRICAO-ELETRICA-HORIZONTE-PERIODO): "
        "estagio_inicio 3 after estagio_fim 1",
    ),
    "horizonte by date": (
        {
            "restricao_eletrica_formula": re_formula((401, "A")),
            "restricao_eletrica_horizonte_data": horizonte_data(
                (401, "2026-07-18", "2026-07-04")
            ),
        },
        "restriction 401 horizonte (RESTRICAO-ELETRICA-HORIZONTE-DATA): "
        "data_inicio 2026-07-18 after data_fim 2026-07-04",
    ),
    "limits by period": (
        {
            "restricao_eletrica_formula": re_formula((401, "A")),
            "restricao_eletrica_limite_formula_periodo_patamar": re_lim_formula(
                (401, 3, 1, 1, 0.0, 1.0)
            ),
        },
        "restriction 401 limits (RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR): "
        "estagio_inicio 3 after estagio_fim 1",
    ),
    "limits by date": (
        {
            "restricao_eletrica_formula": re_formula((401, "A")),
            "restricao_eletrica_limites_formula_data_patamar": re_lim_formula_data(
                (401, "2026-07-18", "2026-07-04", 1, 0.0, 1.0)
            ),
        },
        "restriction 401 limits (RESTRICAO-ELETRICA-LIMITES-FORMULA-DATA-PATAMAR): "
        "data_inicio 2026-07-18 after data_fim 2026-07-04",
    ),
    "override with patamar": (
        {
            "restricao_eletrica_inequacao_periodo_patamar": re_inequacao_per_pat(
                (406, 3, 1, 1, "A", "<=", "0")
            ),
        },
        "restriction 406 inequacao override "
        "(RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR): "
        "estagio_inicio 3 after estagio_fim 1",
    ),
    "override without patamar": (
        {
            "restricao_eletrica_inequacao_periodo_patamar": re_inequacao_per_pat(
                (406, 3, 1, None, "A", "<=", "0")
            ),
        },
        "restriction 406 inequacao override "
        "(RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR): "
        "estagio_inicio 3 after estagio_fim 1",
    ),
    "formula override by period": (
        {
            "restricao_eletrica_formula": re_formula((401, "A")),
            "restricao_eletrica_formula_periodo_patamar": re_formula_per_pat(
                (401, 3, 1, 1, "B")
            ),
        },
        "restriction 401 formula override "
        "(RESTRICAO-ELETRICA-FORMULA-PERIODO-PATAMAR): "
        "estagio_inicio 3 after estagio_fim 1",
    ),
    "formula override by date": (
        {
            "restricao_eletrica_formula": re_formula((401, "A")),
            "restricao_eletrica_formula_data_patamar": re_formula_data_pat(
                (401, "2026-07-18", "2026-07-04", None, "B")
            ),
        },
        "restriction 401 formula override "
        "(RESTRICAO-ELETRICA-FORMULA-DATA-PATAMAR): "
        "data_inicio 2026-07-18 after data_fim 2026-07-04",
    ),
    "violation period": (
        {
            "restricao_eletrica_inequacao": re_inequacao((501, "A", ">=", "0")),
            "restricao_eletrica_tratamento_violacao_periodo": (
                tratamento_violacao_periodo((501, 3, 1, "hard", 1.0))
            ),
        },
        "restriction 501 violation period "
        "(RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO-PERIODO): "
        "estagio_inicio 3 after estagio_fim 1",
    ),
}


@pytest.mark.parametrize("card", _INVERTED_RANGES)
def test_inverted_range_raises_naming_the_card_and_the_code(card: str) -> None:
    frames, message = _INVERTED_RANGES[card]
    with pytest.raises(ValueError, match=re.escape(message)):
        read_libs_electrical(StubRestricoes(**frames), _THREE_WEEKS)
