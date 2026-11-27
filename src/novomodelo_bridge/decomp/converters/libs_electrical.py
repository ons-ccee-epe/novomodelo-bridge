"""Read the LIBs-era electrical special-constraint file into a structured model.

A LIBs-era source-model deck moves its richer electrical special constraints
out of the classic ``dadger`` ``RE``/``LU``/``FU``/``FT``/``FI`` registers into
``lib_restricao-eletrica-especial.csv``, read by idecomp's
``idecomp.libs.restricoes.Restricoes``. This module reads the file's
cards — named expressions (``EXPRESSAO-ELETRICA``),
data aliases (``ALIAS-ELETRICO`` + its per-(período,patamar) values), and
restrictions in either the ``RESTRICAO-ELETRICA-FORMULA`` (two-sided
per-(período,patamar) or per-(date,patamar) limit) or
``RESTRICAO-ELETRICA-INEQUACAO`` (``lhs op rhs``, with per-(período,patamar)
overrides) shape, plus each restriction's active horizon (by período or by
date), activation-rule binding, and violation-treatment rows — into an
immutable :class:`LibsElectricalModel`. Each short-form alias (``RE``,
``RE-*``, ``ALIAS-ELET*``) is read as its long-form card. The formula cards
by período and by date (``RESTRICAO-ELETRICA-FORMULA-PERIODO-PATAMAR``,
``RESTRICAO-ELETRICA-FORMULA-DATA-PATAMAR``) are read as formula overrides of
a FORMULA restriction.

A date-indexed range maps to the stages from the one that contains its start
date to the one that contains its end date, the end date inclusive. A date or
período outside the calendar raises ``ValueError`` naming the restriction or
alias code (and the date); so does any stage or date range that ends before it
starts, which also names the card.

idecomp returns every ``formula``/``formula_limite``/``limite_inferior``/
``limite_superior`` DSL cell as an opaque string or number; the reader stores
it **verbatim** and neither parses nor evaluates it.

:func:`read_carga_ande` and :func:`ande_load` read the ``RI`` register's
``carga_ande`` (the ANDE / Itaipu-import load), which is not part of the model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import pandas as pd

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.core.pandas_utils import as_float, as_frame, as_int
from novomodelo_bridge.decomp.converters.hydro import _ITAIPU_CODE

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import date

    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)

#: DECOMP's unbounded-side sentinel on two-sided limit columns is exactly
#: ``-1E+31``/``+1E+31``; anything at or beyond this magnitude means "no bound
#: on this side" and is normalized to ``None``.
_UNBOUNDED_SENTINEL = 1e30


@dataclass(frozen=True)
class ElectricalExpression:
    """A named, reusable linear sub-expression (``EXPRESSAO-ELETRICA``).

    ``formula`` is the raw DSL string verbatim; references to this expression
    inside other formulas stay unexpanded.
    """

    code: int
    name: str
    formula: str


@dataclass(frozen=True)
class ElectricalAlias:
    """A data alias (``ALIAS-ELETRICO``) and its per-(stage,patamar) values.

    ``values`` is keyed by ``(stage_index, patamar)`` — the 0-based stage
    index (période mapped against the calendar) and the 1-based patamar, or
    ``patamar=None`` meaning "all blocks" (the source row's ``NA`` patamar).
    """

    code: int
    name: str
    values: Mapping[tuple[int, int | None], float]


@dataclass(frozen=True)
class PeriodPatamarOverride:
    """One ``RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR`` row.

    Overrides the parent restriction's constant ``lhs op rhs`` for its own
    ``[stage_start, stage_end]`` range and ``patamar`` (``None`` = all
    blocks). All three sides are the raw DSL strings verbatim; the reader
    applies no precedence between an override and the constant sides.
    """

    stage_start: int
    stage_end: int
    patamar: int | None
    lhs: str
    operator: str
    rhs: str


@dataclass(frozen=True)
class FormulaOverride:
    """One ``RESTRICAO-ELETRICA-FORMULA-PERIODO-PATAMAR`` row, or one
    ``-FORMULA-DATA-PATAMAR`` row with its dates mapped to stages.

    Replaces the parent FORMULA restriction's formula for its own
    ``[stage_start, stage_end]`` range and ``patamar`` (``None`` = all blocks).
    ``formula`` is the raw DSL string verbatim; the reader applies no precedence
    between overrides or against the restriction's own formula.
    """

    stage_start: int
    stage_end: int
    patamar: int | None
    formula: str


@dataclass(frozen=True)
class ViolationRow:
    """One violation-treatment card line, exactly as the source declares it.

    ``stage_start``/``stage_end`` are 0-based inclusive stage indices from a
    ``RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO-PERIODO`` row; both ``None`` for a
    ``RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO`` row, which spans the whole
    horizon. ``kind`` is ``"hard"`` or ``"soft"``; ``cost`` is ``None`` when the
    line leaves ``custo_violacao`` blank.
    """

    stage_start: int | None
    stage_end: int | None
    kind: str
    cost: float | None


@dataclass(frozen=True)
class ElectricalRestriction:
    """One electrical special constraint, unifying the FORMULA and INEQUACAO
    long-form card shapes.

    ``lhs``/``operator``/``rhs`` are the raw DSL strings verbatim (never
    parsed here). A FORMULA-form restriction (``is_formula`` is ``True``) has
    no ``operator``/``rhs`` — its per-(stage,patamar) two-sided limit lives in
    ``limits`` instead, keyed by ``(stage_index, patamar)`` with
    ``patamar=None`` meaning "all blocks" and an absent side normalized from
    DECOMP's ``±1E+31`` sentinel to ``None``. An INEQUACAO-form restriction
    carries the constant ``lhs op rhs`` plus any per-(stage-range,patamar)
    ``overrides``; one declared only by
    ``RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR`` rows has ``lhs``,
    ``operator`` and ``rhs`` all ``None`` and is active only in the cells its
    ``overrides`` cover. A FORMULA-form restriction may also carry
    ``formula_overrides``; one declared only by those rows plus its ``limits``
    has ``lhs`` ``None``. ``stage_start``/``stage_end`` come from
    ``RESTRICAO-ELETRICA-HORIZONTE-PERIODO`` or ``-HORIZONTE-DATA`` (``None``
    when absent);
    ``habilita`` is the bound ``REGRA-ATIVACAO`` id — look its text up in
    :attr:`LibsElectricalModel.rules`. ``violations`` holds every
    violation-treatment line of the restriction as :class:`ViolationRow`: the
    whole-horizon ``TRATAMENTO-VIOLACAO`` lines first, then the
    ``TRATAMENTO-VIOLACAO-PERIODO`` lines, each in file order.
    """

    code: int
    lhs: str | None
    operator: str | None
    rhs: str | None
    is_formula: bool
    limits: Mapping[tuple[int, int | None], tuple[float | None, float | None]]
    overrides: tuple[PeriodPatamarOverride, ...]
    stage_start: int | None = None
    stage_end: int | None = None
    habilita: int | None = None
    violations: tuple[ViolationRow, ...] = ()
    formula_overrides: tuple[FormulaOverride, ...] = ()


@dataclass(frozen=True)
class LibsElectricalModel:
    """The full long-form electrical-constraint model, read from one deck's
    ``Restricoes``.

    ``expressions``/``aliases`` are keyed by their string identifier
    (``identificador_expressao``/``identificador_alias``); ``restrictions``
    by ``codigo_restricao``; ``rules`` (the raw ``regra_ativacao`` text) by
    ``codigo_regra_ativacao``.
    """

    expressions: Mapping[str, ElectricalExpression]
    aliases: Mapping[str, ElectricalAlias]
    restrictions: Mapping[int, ElectricalRestriction]
    rules: Mapping[int, str]


class _ElectricalCards(Protocol):
    def expressao_eletrica(self, *, df: bool) -> object: ...

    def alias_eletrico_valor_periodo_patamar(self, *, df: bool) -> object: ...

    def alias_eletrico(self, *, df: bool) -> object: ...

    def restricao_eletrica_horizonte_periodo(self, *, df: bool) -> object: ...

    def restricao_eletrica_horizonte_data(self, *, df: bool) -> object: ...

    def restricao_eletrica_limite_formula_periodo_patamar(
        self, *, df: bool
    ) -> object: ...

    def restricao_eletrica_limites_formula_data_patamar(
        self, *, df: bool
    ) -> object: ...

    def restricao_eletrica_inequacao_periodo_patamar(self, *, df: bool) -> object: ...

    def restricao_eletrica_habilita(self, *, df: bool) -> object: ...

    def restricao_eletrica_regra_ativacao(self, *, df: bool) -> object: ...

    def restricao_eletrica_tratamento_violacao(self, *, df: bool) -> object: ...

    def restricao_eletrica_tratamento_violacao_periodo(self, *, df: bool) -> object: ...

    def restricao_eletrica_formula(self, *, df: bool) -> object: ...

    def restricao_eletrica_inequacao(self, *, df: bool) -> object: ...

    def restricao_eletrica_formula_periodo_patamar(self, *, df: bool) -> object: ...

    def restricao_eletrica_formula_data_patamar(self, *, df: bool) -> object: ...

    def re(self, *, df: bool) -> object: ...

    def re_horiz_per(self, *, df: bool) -> object: ...

    def re_lim_form_per_pat(self, *, df: bool) -> object: ...

    def re_horiz_data(self, *, df: bool) -> object: ...

    def re_lim_form_data_pat(self, *, df: bool) -> object: ...

    def alias_elet(self, *, df: bool) -> object: ...

    def alias_elet_val_per_pat(self, *, df: bool) -> object: ...

    def re_regra_ativa(self, *, df: bool) -> object: ...

    def re_habilita(self, *, df: bool) -> object: ...

    def re_trat_viol(self, *, df: bool) -> object: ...

    def re_trat_viol_per(self, *, df: bool) -> object: ...

    def re_per_pat(self, *, df: bool) -> object: ...

    def re_data_pat(self, *, df: bool) -> object: ...


#: Long-form accessor -> its short-form alias, which idecomp exposes with the
#: same columns. :func:`_df` reads the long card first and the alias after it, so
#: a code declared in both spellings resolves to the alias row wherever a
#: reader dict keeps the last row.
_SHORT_FORM_ALIAS: Mapping[str, str] = {
    "restricao_eletrica_formula": "re",
    "restricao_eletrica_horizonte_periodo": "re_horiz_per",
    "restricao_eletrica_limite_formula_periodo_patamar": "re_lim_form_per_pat",
    "restricao_eletrica_horizonte_data": "re_horiz_data",
    "restricao_eletrica_limites_formula_data_patamar": "re_lim_form_data_pat",
    "alias_eletrico": "alias_elet",
    "alias_eletrico_valor_periodo_patamar": "alias_elet_val_per_pat",
    "restricao_eletrica_regra_ativacao": "re_regra_ativa",
    "restricao_eletrica_habilita": "re_habilita",
    "restricao_eletrica_tratamento_violacao": "re_trat_viol",
    "restricao_eletrica_tratamento_violacao_periodo": "re_trat_viol_per",
    "restricao_eletrica_formula_periodo_patamar": "re_per_pat",
    "restricao_eletrica_formula_data_patamar": "re_data_pat",
}


def _df(source: _ElectricalCards, attr: str) -> pd.DataFrame | None:
    """Read one card (long accessor, then its short-form alias) as a DataFrame,
    treating an absent card as empty."""
    names = (attr, _SHORT_FORM_ALIAS[attr]) if attr in _SHORT_FORM_ALIAS else (attr,)
    present = [
        frame
        for frame in (getattr(source, name)(df=True) for name in names)
        if frame is not None and not getattr(frame, "empty", True)
    ]
    return pd.concat(present, ignore_index=True) if present else None


def _opt_int(value: object) -> int | None:
    """A cell as ``int``, or ``None`` when blank (``NA``)."""
    return None if pd.isna(value) else as_int(value)


def _opt_float(value: object) -> float | None:
    """A cell as ``float``, or ``None`` when blank (``NA``)."""
    return None if pd.isna(value) else as_float(value)


def _opt_bound(value: object) -> float | None:
    """A two-sided limit cell as ``float``, or ``None`` when blank or at/beyond
    DECOMP's ``±1E+31`` unbounded sentinel."""
    if pd.isna(value):
        return None
    parsed = as_float(value)
    return None if abs(parsed) >= _UNBOUNDED_SENTINEL else parsed


def _stage_index(estagio: int, n_stages: int, context: str) -> int:
    """Map a 1-based período to its 0-based stage index against the calendar.

    Mirrors ``network._ia_dense``'s bounds-check shape.
    """
    if not 1 <= estagio <= n_stages:
        raise ValueError(
            f"{context}: período {estagio} outside the calendar (1..{n_stages})"
        )
    return estagio - 1


def _as_day(value: object, context: str) -> date:
    """A date cell as a ``date``, dropping any time of day. idecomp reads a blank
    or unparseable date as ``NaT``, which is rejected here."""
    stamp = pd.Timestamp(value)  # ty: ignore[invalid-argument-type] cell typed as a scalar union
    if not isinstance(stamp, pd.Timestamp):
        raise ValueError(f"{context}: date is blank or not a date")
    return stamp.date()


def _stage_of_date(
    value: object, calendar: Sequence[OperativeStage], context: str
) -> int:
    """Map a date to the 0-based index of the stage that contains it.

    A stage's ``end_date`` is exclusive, and a time of day on *value* is ignored.
    """
    day = _as_day(value, context)
    for index, stage in enumerate(calendar):
        if stage.start_date <= day < stage.end_date:
            return index
    raise ValueError(
        f"{context}: data {day} outside the calendar "
        f"({calendar[0].start_date}..{calendar[-1].end_date})"
    )


def _require_ordered[T: (int, date)](
    start: T, end: T, *, fields: tuple[str, str], card: str, context: str
) -> None:
    """The one check that a stage or date range does not end before it starts;
    the error names *card* and carries the restriction or alias code in *context*."""
    if start > end:
        raise ValueError(
            f"{context} ({card}): {fields[0]} {start} after {fields[1]} {end}"
        )


def _stage_range_from_dates(
    start: object,
    end: object,
    calendar: Sequence[OperativeStage],
    context: str,
    *,
    card: str,
) -> tuple[int, int]:
    """Map a date range to its inclusive 0-based stage range.

    The range runs from the stage that contains *start* to the stage that
    contains *end*; the end date is inclusive, so an *end* on the first day of a
    stage includes that stage.
    """
    first, last = _as_day(start, context), _as_day(end, context)
    stages = (
        _stage_of_date(first, calendar, context),
        _stage_of_date(last, calendar, context),
    )
    _require_ordered(
        first, last, fields=("data_inicio", "data_fim"), card=card, context=context
    )
    return stages


def _row_stage_range(
    row: pd.Series,
    calendar: Sequence[OperativeStage],
    *,
    card: str,
    context: str,
) -> tuple[int, int]:
    """Map a card row's inclusive range to 0-based stage indices: from
    ``data_inicio``/``data_fim`` on a date-indexed card, else from the 1-based
    ``estagio_inicio``/``estagio_fim``. Every stage mapping of a card row goes
    through here, so a range that ends before it starts is rejected for every card."""
    if "data_inicio" in row.index:
        return _stage_range_from_dates(
            row["data_inicio"], row["data_fim"], calendar, context, card=card
        )
    first, last = int(row["estagio_inicio"]), int(row["estagio_fim"])
    stages = (
        _stage_index(first, len(calendar), context),
        _stage_index(last, len(calendar), context),
    )
    _require_ordered(
        first,
        last,
        fields=("estagio_inicio", "estagio_fim"),
        card=card,
        context=context,
    )
    return stages


def _read_expressions(restricoes: _ElectricalCards) -> dict[str, ElectricalExpression]:
    """Read ``EXPRESSAO-ELETRICA`` into ``{name: ElectricalExpression}``."""
    frame = _df(restricoes, "expressao_eletrica")
    if frame is None:
        return {}
    expressions: dict[str, ElectricalExpression] = {}
    for _, row in frame.iterrows():
        name = str(row["identificador_expressao"]).strip()
        expressions[name] = ElectricalExpression(
            code=int(row["codigo_expressao"]),
            name=name,
            formula=str(row["formula"]),
        )
    return expressions


def _read_alias_values(
    restricoes: _ElectricalCards,
    calendar: Sequence[OperativeStage],
) -> dict[int, dict[tuple[int, int | None], float]]:
    """Read ``ALIAS-ELETRICO-VALOR-PERIODO-PATAMAR`` into ``{codigo_alias:
    {(stage_index, patamar): valor}}``, expanding each row's stage range."""
    frame = _df(restricoes, "alias_eletrico_valor_periodo_patamar")
    if frame is None:
        return {}
    values_by_code: dict[int, dict[tuple[int, int | None], float]] = {}
    for _, row in frame.iterrows():
        code = int(row["codigo_alias"])
        start, end = _row_stage_range(
            row,
            calendar,
            card="ALIAS-ELETRICO-VALOR-PERIODO-PATAMAR",
            context=f"alias {code} value",
        )
        patamar = _opt_int(row["patamar"])
        value = float(row["valor"])
        cell = values_by_code.setdefault(code, {})
        for stage in range(start, end + 1):
            cell[(stage, patamar)] = value
    return values_by_code


def _read_aliases(
    restricoes: _ElectricalCards,
    calendar: Sequence[OperativeStage],
) -> dict[str, ElectricalAlias]:
    """Read ``ALIAS-ELETRICO`` joined to its per-(stage,patamar) values."""
    declared = _df(restricoes, "alias_eletrico")
    if declared is None:
        return {}
    values_by_code = _read_alias_values(restricoes, calendar)
    aliases: dict[str, ElectricalAlias] = {}
    for _, row in declared.iterrows():
        code = int(row["codigo_alias"])
        name = str(row["identificador_alias"]).strip()
        aliases[name] = ElectricalAlias(
            code=code,
            name=name,
            values=values_by_code.get(code, {}),
        )
    return aliases


def _read_horizonte(
    restricoes: _ElectricalCards,
    calendar: Sequence[OperativeStage],
) -> dict[int, tuple[int, int]]:
    """Read ``RESTRICAO-ELETRICA-HORIZONTE-PERIODO``, then ``-HORIZONTE-DATA``, into
    ``{codigo_restricao: (stage_start, stage_end)}``; the last row of a code wins."""
    horizonte: dict[int, tuple[int, int]] = {}
    for accessor, card in (
        (
            "restricao_eletrica_horizonte_periodo",
            "RESTRICAO-ELETRICA-HORIZONTE-PERIODO",
        ),
        ("restricao_eletrica_horizonte_data", "RESTRICAO-ELETRICA-HORIZONTE-DATA"),
    ):
        frame = _df(restricoes, accessor)
        if frame is None:
            continue
        for _, row in frame.iterrows():
            code = int(row["codigo_restricao"])
            horizonte[code] = _row_stage_range(
                row, calendar, card=card, context=f"restriction {code} horizonte"
            )
    return horizonte


def _read_formula_limits(
    restricoes: _ElectricalCards,
    calendar: Sequence[OperativeStage],
) -> dict[int, dict[tuple[int, int | None], tuple[float | None, float | None]]]:
    """Read ``RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR``, then
    ``-LIMITES-FORMULA-DATA-PATAMAR``, into ``{codigo_restricao: {(stage_index,
    patamar): (lower, upper)}}``, expanding each row's stage range (the last row
    of a cell wins) and normalizing the ``±1E+31`` sentinel to ``None``."""
    limits: dict[
        int, dict[tuple[int, int | None], tuple[float | None, float | None]]
    ] = {}
    for accessor, card in (
        (
            "restricao_eletrica_limite_formula_periodo_patamar",
            "RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR",
        ),
        (
            "restricao_eletrica_limites_formula_data_patamar",
            "RESTRICAO-ELETRICA-LIMITES-FORMULA-DATA-PATAMAR",
        ),
    ):
        frame = _df(restricoes, accessor)
        if frame is None:
            continue
        for _, row in frame.iterrows():
            code = int(row["codigo_restricao"])
            start, end = _row_stage_range(
                row, calendar, card=card, context=f"restriction {code} limits"
            )
            patamar = _opt_int(row["patamar"])
            bounds = (
                _opt_bound(row["limite_inferior"]),
                _opt_bound(row["limite_superior"]),
            )
            cell = limits.setdefault(code, {})
            for stage in range(start, end + 1):
                cell[(stage, patamar)] = bounds
    return limits


def _read_inequacao_overrides(
    restricoes: _ElectricalCards,
    calendar: Sequence[OperativeStage],
) -> dict[int, tuple[PeriodPatamarOverride, ...]]:
    """Read ``RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR`` into
    ``{codigo_restricao: (PeriodPatamarOverride, ...)}``."""
    frame = _df(restricoes, "restricao_eletrica_inequacao_periodo_patamar")
    if frame is None:
        return {}
    overrides: dict[int, list[PeriodPatamarOverride]] = {}
    for _, row in frame.iterrows():
        code = int(row["codigo_restricao"])
        start, end = _row_stage_range(
            row,
            calendar,
            card="RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR",
            context=f"restriction {code} inequacao override",
        )
        overrides.setdefault(code, []).append(
            PeriodPatamarOverride(
                stage_start=start,
                stage_end=end,
                patamar=_opt_int(row["patamar"]),
                lhs=str(row["formula"]),
                operator=str(row["operador"]),
                rhs=str(row["formula_limite"]),
            )
        )
    return {code: tuple(rows) for code, rows in overrides.items()}


def _read_formula_overrides(
    restricoes: _ElectricalCards,
    calendar: Sequence[OperativeStage],
) -> dict[int, tuple[FormulaOverride, ...]]:
    """Read ``RESTRICAO-ELETRICA-FORMULA-PERIODO-PATAMAR``, then
    ``-FORMULA-DATA-PATAMAR``, into ``{codigo_restricao: (FormulaOverride, ...)}``,
    each card in file order."""
    overrides: dict[int, list[FormulaOverride]] = {}
    for accessor, card in (
        (
            "restricao_eletrica_formula_periodo_patamar",
            "RESTRICAO-ELETRICA-FORMULA-PERIODO-PATAMAR",
        ),
        (
            "restricao_eletrica_formula_data_patamar",
            "RESTRICAO-ELETRICA-FORMULA-DATA-PATAMAR",
        ),
    ):
        frame = _df(restricoes, accessor)
        if frame is None:
            continue
        for _, row in frame.iterrows():
            code = int(row["codigo_restricao"])
            start, end = _row_stage_range(
                row,
                calendar,
                card=card,
                context=f"restriction {code} formula override",
            )
            overrides.setdefault(code, []).append(
                FormulaOverride(
                    stage_start=start,
                    stage_end=end,
                    patamar=_opt_int(row["patamar"]),
                    formula=str(row["formula"]),
                )
            )
    return {code: tuple(rows) for code, rows in overrides.items()}


def _read_habilita(restricoes: _ElectricalCards) -> dict[int, int]:
    """Read ``RESTRICAO-ELETRICA-HABILITA`` into ``{codigo_restricao:
    codigo_regra_ativacao}``."""
    frame = _df(restricoes, "restricao_eletrica_habilita")
    if frame is None:
        return {}
    return {
        int(row["codigo_restricao"]): int(row["codigo_regra_ativacao"])
        for _, row in frame.iterrows()
    }


def _read_rules(restricoes: _ElectricalCards) -> dict[int, str]:
    """Read ``RESTRICAO-ELETRICA-REGRA-ATIVACAO`` into
    ``{codigo_regra_ativacao: regra_ativacao}`` (the raw boolean-rule text)."""
    frame = _df(restricoes, "restricao_eletrica_regra_ativacao")
    if frame is None:
        return {}
    return {
        int(row["codigo_regra_ativacao"]): str(row["regra_ativacao"])
        for _, row in frame.iterrows()
    }


def _violation_kind(code: int, raw: object) -> str:
    """*raw* ``tipo_violacao`` normalized to ``"hard"`` or ``"soft"``."""
    kind = str(raw).strip().lower()
    if kind not in ("hard", "soft"):
        raise ValueError(
            f"restriction {code}: unknown tipo_violacao {raw!r}; expected hard or soft"
        )
    return kind


def _read_violation_rows(
    restricoes: _ElectricalCards,
    calendar: Sequence[OperativeStage],
) -> dict[int, tuple[ViolationRow, ...]]:
    """Read ``RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO`` and its ``-PERIODO``
    variant into ``{codigo_restricao: (ViolationRow, ...)}``: the
    whole-horizon rows first, then the per-period rows, each in file order."""
    rows: dict[int, list[ViolationRow]] = {}
    base = _df(restricoes, "restricao_eletrica_tratamento_violacao")
    if base is not None:
        for _, row in base.iterrows():
            code = int(row["codigo_restricao"])
            rows.setdefault(code, []).append(
                ViolationRow(
                    stage_start=None,
                    stage_end=None,
                    kind=_violation_kind(code, row["tipo_violacao"]),
                    cost=_opt_float(row["custo_violacao"]),
                )
            )
    per_period = _df(restricoes, "restricao_eletrica_tratamento_violacao_periodo")
    if per_period is not None:
        for _, row in per_period.iterrows():
            code = int(row["codigo_restricao"])
            start, end = _row_stage_range(
                row,
                calendar,
                card="RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO-PERIODO",
                context=f"restriction {code} violation period",
            )
            rows.setdefault(code, []).append(
                ViolationRow(
                    stage_start=start,
                    stage_end=end,
                    kind=_violation_kind(code, row["tipo_violacao"]),
                    cost=_opt_float(row["custo_violacao"]),
                )
            )
    return {code: tuple(code_rows) for code, code_rows in rows.items()}


def _read_restrictions(
    restricoes: _ElectricalCards,
    calendar: Sequence[OperativeStage],
) -> dict[int, ElectricalRestriction]:
    """Join the FORMULA and INEQUACAO restriction cards to their per-(period or
    date, patamar) limit/override rows, horizonte, habilita, and violation rows
    by ``codigo_restricao``.

    A code with formula-override rows but no FORMULA or INEQUACAO card is a
    FORMULA restriction declared by those rows and its limits; a code with
    ``RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR`` rows but no FORMULA,
    INEQUACAO or formula-override rows is an INEQUACAO restriction declared by
    those rows alone. Rows that cannot apply to any restriction (inequality rows
    on a FORMULA code, formula-override rows on an INEQUACAO code, limit rows on
    a non-FORMULA code) are reported.
    """
    formula = _df(restricoes, "restricao_eletrica_formula")
    inequacao = _df(restricoes, "restricao_eletrica_inequacao")
    overrides = _read_inequacao_overrides(restricoes, calendar)
    formula_overrides = _read_formula_overrides(restricoes, calendar)
    if (
        formula is None
        and inequacao is None
        and not overrides
        and not formula_overrides
    ):
        return {}

    horizonte = _read_horizonte(restricoes, calendar)
    limits = _read_formula_limits(restricoes, calendar)
    habilita = _read_habilita(restricoes)
    violation_rows = _read_violation_rows(restricoes, calendar)

    restrictions: dict[int, ElectricalRestriction] = {}

    if formula is not None:
        for _, row in formula.iterrows():
            code = int(row["codigo_restricao"])
            stage_start, stage_end = horizonte.get(code, (None, None))
            restrictions[code] = ElectricalRestriction(
                code=code,
                lhs=str(row["formula"]),
                operator=None,
                rhs=None,
                is_formula=True,
                limits=limits.get(code, {}),
                overrides=(),
                stage_start=stage_start,
                stage_end=stage_end,
                habilita=habilita.get(code),
                violations=violation_rows.get(code, ()),
                formula_overrides=formula_overrides.get(code, ()),
            )

    if inequacao is not None:
        for _, row in inequacao.iterrows():
            code = int(row["codigo_restricao"])
            stage_start, stage_end = horizonte.get(code, (None, None))
            restrictions[code] = ElectricalRestriction(
                code=code,
                lhs=str(row["formula"]),
                operator=str(row["operador"]),
                rhs=str(row["formula_limite"]),
                is_formula=False,
                limits={},
                overrides=overrides.get(code, ()),
                stage_start=stage_start,
                stage_end=stage_end,
                habilita=habilita.get(code),
                violations=violation_rows.get(code, ()),
            )

    for code, formula_rows in formula_overrides.items():
        if code in restrictions:
            continue
        stage_start, stage_end = horizonte.get(code, (None, None))
        restrictions[code] = ElectricalRestriction(
            code=code,
            lhs=None,
            operator=None,
            rhs=None,
            is_formula=True,
            limits=limits.get(code, {}),
            overrides=(),
            stage_start=stage_start,
            stage_end=stage_end,
            habilita=habilita.get(code),
            violations=violation_rows.get(code, ()),
            formula_overrides=formula_rows,
        )

    for code, rows in overrides.items():
        if code in restrictions:
            continue
        stage_start, stage_end = horizonte.get(code, (None, None))
        restrictions[code] = ElectricalRestriction(
            code=code,
            lhs=None,
            operator=None,
            rhs=None,
            is_formula=False,
            limits={},
            overrides=rows,
            stage_start=stage_start,
            stage_end=stage_end,
            habilita=habilita.get(code),
            violations=violation_rows.get(code, ()),
        )

    _emit_unused_period_rows(
        [
            ("RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR", code)
            for code in sorted(c for c in overrides if restrictions[c].is_formula)
        ]
        + [
            ("RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR/-DATA-PATAMAR", code)
            for code in sorted(
                c
                for c in limits
                if c not in restrictions or not restrictions[c].is_formula
            )
        ]
        + [
            ("RESTRICAO-ELETRICA-FORMULA-PERIODO-PATAMAR/-DATA-PATAMAR", code)
            for code in sorted(
                c for c in formula_overrides if not restrictions[c].is_formula
            )
        ]
    )
    return restrictions


def _emit_unused_period_rows(rows: Sequence[tuple[str, int]]) -> None:
    if not rows:
        return
    emit(
        Diagnostic(
            code="decomp-electrical-period-rows-unused",
            severity=Severity.WARNING,
            category="Special constraints",
            title=f"Electrical period rows not applied ({len(rows)})",
            summary=(
                "The listed period rows of the LIBS electrical file do not fit "
                "their restriction: inequality period rows apply only to "
                "inequality restrictions, and formula and limit rows, by period "
                "or by date, only to formula restrictions. The conversion "
                "ignores them, so the case runs without what they set."
            ),
            table=DiagnosticTable(
                columns=["Card", "Restriction"], rows=[list(row) for row in rows]
            ),
            remediation=(
                "Move each listed row to the card that fits its restriction, or "
                "remove it."
            ),
        ),
        logger=_LOG,
    )


def read_libs_electrical(
    restricoes: _ElectricalCards,
    calendar: Sequence[OperativeStage],
) -> LibsElectricalModel | None:
    """Read every electrical card into a structured model.

    Joins the ``RESTRICAO-ELETRICA-FORMULA``/``-INEQUACAO`` restrictions (and
    the inequalities declared by ``-INEQUACAO-PERIODO-PATAMAR`` rows alone, and
    the formulas declared by ``-FORMULA-PERIODO-PATAMAR``/``-DATA-PATAMAR`` rows
    alone) to their per-(period or date,patamar) limit/override rows, horizonte,
    habilita, rule, and violation rows by ``codigo_restricao``, and maps every
    período (1-based) and every date onto 0-based stage indices against
    *calendar*.

    Returns ``None`` when the deck carries no electrical card at all (no
    expressions, aliases, restrictions, or rules), so callers can no-op cleanly
    instead of handling an empty-but-present model.

    Raises
    ------
    ValueError
        When a período or date falls outside *calendar* (naming the restriction
        or alias code), or a stage or date range ends before it starts (naming
        the card as well).
    """
    expressions = _read_expressions(restricoes)
    aliases = _read_aliases(restricoes, calendar)
    restrictions = _read_restrictions(restricoes, calendar)
    rules = _read_rules(restricoes)

    if not expressions and not aliases and not restrictions and not rules:
        return None

    return LibsElectricalModel(
        expressions=expressions,
        aliases=aliases,
        restrictions=restrictions,
        rules=rules,
    )


#: idecomp's ``df=True`` accessor expands a per-patamar list column into
#: ``carga_ande_1``, ``carga_ande_2``, ... (one per patamar slot, mirroring
#: ``DP``'s ``carga_1``/``IA``'s ``limite_de_para_1`` convention); a row's own
#: slots beyond its declared patamar count read back as ``NaN``. This pattern
#: discovers however many patamar-slot columns the installed idecomp emits,
#: rather than hardcoding a slot count.
_CARGA_ANDE_COLUMN = re.compile(r"^carga_ande_(\d+)$")


class _RiRegisters(Protocol):
    def ri(self, *, df: bool) -> object: ...


def read_carga_ande(
    dadger: _RiRegisters,
    calendar: Sequence[OperativeStage],
) -> dict[int, list[float]]:
    """Read the ``RI`` (restrição de Itaipu) register's ``carga_ande``
    per-patamar list into a dense, forward-filled per-(stage,block) series.

    ``carga_ande`` (the ANDE / Itaipu-import load) is a per-(estágio,
    patamar) list on the deck's ``RI`` register. Each row's list is trimmed
    of any trailing ``NaN`` (idecomp's fixed-width patamar slots beyond the
    row's own declared patamar count) and the resulting length validated
    against the calendar's own per-stage block count. Declared stages
    forward-fill across *calendar* (estágio 1 mandatory as the inheritance
    base, mirroring ``network._ia_dense``'s idiom) — a deck that declares
    only a handful of stages still yields one entry per calendar stage.

    Returns ``{}`` when the deck carries no ``RI`` register at all (a deck
    with no Itaipu import); callers then place no ``IV`` load.

    Raises
    ------
    ValueError
        When a declared row's ``estagio`` falls outside the calendar
        (:func:`_stage_index`), when a row's own patamar-value count does not
        match the calendar's block count for that stage (naming the stage
        index and both counts), or when the deck carries an ``RI`` row but
        none at estágio 1 (no inheritance base).
    """
    ri = as_frame(dadger.ri(df=True))
    if ri is None or ri.empty:
        return {}

    n_stages = len(calendar)
    indexed_columns: list[tuple[int, str]] = []
    for column in ri.columns:
        match = _CARGA_ANDE_COLUMN.match(column)
        if match is not None:
            indexed_columns.append((int(match.group(1)), column))
    patamar_columns = [column for _, column in sorted(indexed_columns)]

    declared: dict[int, list[float]] = {}
    for _, row in ri.iterrows():
        stage_index = _stage_index(int(row["estagio"]), n_stages, "RI carga_ande")
        values = [row[column] for column in patamar_columns]
        while values and pd.isna(values[-1]):
            values.pop()
        n_blocks = len(calendar[stage_index].block_hours)
        if len(values) != n_blocks:
            raise ValueError(
                f"RI carga_ande stage {stage_index}: {len(values)} patamares "
                f"declared vs {n_blocks} blocks in the calendar"
            )
        declared[stage_index] = [float(value) for value in values]

    if 0 not in declared:
        raise ValueError(
            "RI carga_ande: no row declares estágio 1; sparse-stage "
            "inheritance has no base"
        )

    per_stage: list[list[float]] = []
    for stage in calendar:
        per_stage.append(
            declared.get(stage.index, per_stage[-1] if per_stage else declared[0])
        )
    return {
        stage.index: values for stage, values in zip(calendar, per_stage, strict=True)
    }


def ande_load(case: DecompCase) -> tuple[int, dict[int, list[float]]] | None:
    """The ANDE load and the submarket that carries it.

    Returns ``(codigo_submercado, {stage_index: per-block MW})``: the
    :func:`read_carga_ande` series on Itaipu's own ``hidr.dat`` submarket.
    ``None`` when the deck does not operate Itaipu or carries no ``RI``
    register.
    """
    if _ITAIPU_CODE not in case.id_map.operated_hydro_codes:
        return None
    carga_ande = read_carga_ande(case.dadger, case.calendar)
    if not carga_ande:
        return None
    return int(case.hidr.loc[_ITAIPU_CODE, "submercado"]), carga_ande
