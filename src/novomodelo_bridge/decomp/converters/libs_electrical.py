"""Read the LIBs-era electrical special-constraint file into a structured model,
and fold its parsed formulas into per-(stage,block) bounds.

A LIBs-era source-model deck moves its richer electrical special constraints
out of the classic ``dadger`` ``RE``/``LU``/``FU``/``FT``/``FI`` registers into
``lib_restricao-eletrica-especial.csv``, read by idecomp's
``idecomp.libs.restricoes.Restricoes``. This module reads the file's
period-keyed long-form cards — named expressions (``EXPRESSAO-ELETRICA``),
data aliases (``ALIAS-ELETRICO`` + its per-(período,patamar) values), and
restrictions in either the ``RESTRICAO-ELETRICA-FORMULA`` (two-sided
per-(período,patamar) limit) or ``RESTRICAO-ELETRICA-INEQUACAO`` (``lhs op
rhs``, with per-(período,patamar) overrides) shape, plus each restriction's
active horizon, activation-rule binding, and violation treatment — into an
immutable :class:`LibsElectricalModel`.

idecomp returns every ``formula``/``formula_limite``/``limite_inferior``/
``limite_superior`` DSL cell as an opaque string or number; the reader stores
it **verbatim** and does not evaluate it.
:mod:`novomodelo_bridge.core.electrical_expression`, the grammar NEWAVE's
``restricao-eletrica.csv`` shares, parses one such string into flat terms and
sorts them into the A/B/C buckets; the model is the scope it parses in, so
``EXPRESSAO-ELETRICA`` names and ``re(R)`` references expand to their formulas
and ``ALIAS-ELETRICO`` names stay unresolved for a later resolver.
The activation-rule engine (:class:`ActivationRule`, :class:`Comparison`,
:func:`parse_activation_rule`, :func:`evaluate_rule`) builds on this model
and reuses this same parser for its comparison operands. This module only
reads the period-keyed long-form cards; the short-form ``RE``/``RE-*`` and
the date-indexed ``-HORIZONTE-DATA``/``-FORMULA-DATA-PATAMAR`` variants are
out of scope (kept on the detect+warn fallback).

:func:`build_data_context` resolves the activation engine's :data:`DataContext`
interface against the source model's per-(stage,block) demand/``carga_ande``/
alias data; :func:`assemble_bound` folds a restriction's bucket-B (input-data)
terms, from both sides and sign-aware, into its per-(stage,block) numeric
bound, returning the surviving bucket-A (decision) terms as a sign-canonical
:class:`AssembledBound` — verbatim for a FORMULA restriction, but with any
RHS bucket-A term moved onto the LHS (negated) for a plain INEQUACAO, so a
caller never has to re-derive which side/sign a term ended up on;
:func:`evaluate_se` resolves a structural ``se(cond, X, Y)`` term the same
way, reusing the activation-rule engine for ``cond``.

Bucket C (``disp_usih``, available power) has no novomodelo decision-variable
counterpart, so it never stays structural: :func:`build_available_power`
builds the maintenance-aware per-(plant,stage) :class:`AvailablePower`
lookup and :func:`resolve_disp_usih` reads one plant's value from it;
:func:`assemble_bound` folds every ``disp_usih`` substitution into the same
numeric bound as bucket B, realizing the documented reserve->gen-cap
rewrite (``Σ_h(disp_usih(h) − ger_usih(h)) >= R`` -> ``Σ_h ger_usih(h) <=
Σ_h A_h − R``) so only bucket-A terms ever survive structurally.

:func:`active_cells` ties the activation-rule engine to the restriction's
horizonte and the per-(stage,block) data-context factory, producing
the exact set of ``(stage_index, block_index)`` cells a restriction is
active in — the set the emitter uses to decide which cells
get a bound row at all, as distinct from what that bound's value is.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import pandas as pd

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.core.electrical_expression import (
    ALIAS_TOKEN,
    CONST_TOKEN,
    GNL_TOKEN,
    PEQ_TOKEN,
    SE_TOKEN,
    AssembledBound,
    Bucket,
    DataContext,
    MalformedElectricalExpression,
    ParsedTerm,
    classify_term,
    classify_terms,
    evaluate_terms,
    fold_formula_bound,
    negate,
    parse_linear_expression,
    term_text,
)
from novomodelo_bridge.core.errors import UnsupportedSourceError
from novomodelo_bridge.core.generic_constraint_format import sense_to_interval
from novomodelo_bridge.core.pandas_utils import as_float, as_frame, as_int
from novomodelo_bridge.decomp.converters.anticipated import gnl_block_commitments
from novomodelo_bridge.decomp.converters.hydro import (
    _ITAIPU_CODE,
    plant_available_power,
)
from novomodelo_bridge.decomp.converters.ncs import pq_block_generation
from novomodelo_bridge.decomp.load import _per_stage_block_loads

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.converters.anticipated import GnlCommitmentModel
    from novomodelo_bridge.decomp.group_bounds import GroupBoundEntry
    from novomodelo_bridge.decomp.id_map import DecompIdMap
    from novomodelo_bridge.decomp.temporal import OperativeStage

#: DECOMP's unbounded-side sentinel on two-sided limit columns is exactly
#: ``-1E+31``/``+1E+31``; anything at or beyond this magnitude means "no bound
#: on this side" and is normalized to ``None``.
_UNBOUNDED_SENTINEL = 1e30


@dataclass(frozen=True)
class ElectricalExpression:
    """A named, reusable linear sub-expression (``EXPRESSAO-ELETRICA``).

    ``formula`` is the raw DSL string verbatim — expanding a reference to this
    expression inside another formula is the linear-expression parser's job,
    not this reader's.
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
    blocks). All three sides are the raw DSL strings verbatim; the source
    model's precedence rule (période-patamar overrides the constant) is
    applied by :func:`_effective_inequacao_sides`, consumed from
    :func:`assemble_bound` — not here.
    """

    stage_start: int
    stage_end: int
    patamar: int | None
    lhs: str
    operator: str
    rhs: str


@dataclass(frozen=True)
class ViolationTreatment:
    """One ``RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO`` row.

    Maps to novomodelo's generic-constraint ``slack{enabled, penalty}`` shape:
    ``enabled`` is ``True`` whenever the restriction carries an explicit
    treatment row (``penalty`` is that row's ``custo_violacao``, or ``None``
    when the row leaves it blank). Resolving the classic BIG-M default for a
    restriction with **no** treatment row at all is not this reader's job —
    this reader leaves :attr:`ElectricalRestriction.violation` as ``None``
    in that case.
    """

    enabled: bool
    penalty: float | None


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
    ``overrides`` cover. ``stage_start``/``stage_end`` come from
    ``RESTRICAO-ELETRICA-HORIZONTE-PERIODO`` (``None`` when absent);
    ``habilita`` is the bound ``REGRA-ATIVACAO`` id — look its text up in
    :attr:`LibsElectricalModel.rules`; ``violation`` is the raw
    ``TRATAMENTO-VIOLACAO`` row, or ``None`` when the restriction carries
    none.
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
    violation: ViolationTreatment | None = None


@dataclass(frozen=True)
class LibsElectricalModel:
    """The full period-keyed long-form electrical-constraint model, read from
    one deck's ``Restricoes``.

    ``expressions``/``aliases`` are keyed by their string identifier
    (``identificador_expressao``/``identificador_alias``); ``restrictions``
    by ``codigo_restricao``; ``rules`` (the raw ``regra_ativacao`` text) by
    ``codigo_regra_ativacao``. ``gnl_thermal_codes`` are the deck's ``dadgnl``
    GNL plants, whose ``ger_usit`` the parser reads as input data.

    The model is the :class:`~novomodelo_bridge.core.electrical_expression.
    ExpressionScope` its formulas are parsed in.
    """

    expressions: Mapping[str, ElectricalExpression]
    aliases: Mapping[str, ElectricalAlias]
    restrictions: Mapping[int, ElectricalRestriction]
    rules: Mapping[int, str]
    gnl_thermal_codes: frozenset[int] = frozenset()

    def expression_formula(self, name: str) -> str | None:
        expression = self.expressions.get(name)
        return None if expression is None else expression.formula

    def is_alias(self, name: str) -> bool:
        return name in self.aliases

    def restriction_formula(self, code: int) -> str | None:
        restriction = self.restrictions.get(code)
        if restriction is None:
            return None
        if restriction.lhs is None:
            raise ValueError(
                f"re({code}) references restriction {code}, which is declared "
                "only by period rows and has no single formula"
            )
        return restriction.lhs


class _ElectricalCards(Protocol):
    def expressao_eletrica(self, *, df: bool) -> object: ...

    def alias_eletrico_valor_periodo_patamar(self, *, df: bool) -> object: ...

    def alias_eletrico(self, *, df: bool) -> object: ...

    def restricao_eletrica_horizonte_periodo(self, *, df: bool) -> object: ...

    def restricao_eletrica_limite_formula_periodo_patamar(
        self, *, df: bool
    ) -> object: ...

    def restricao_eletrica_inequacao_periodo_patamar(self, *, df: bool) -> object: ...

    def restricao_eletrica_habilita(self, *, df: bool) -> object: ...

    def restricao_eletrica_regra_ativacao(self, *, df: bool) -> object: ...

    def restricao_eletrica_tratamento_violacao(self, *, df: bool) -> object: ...

    def restricao_eletrica_formula(self, *, df: bool) -> object: ...

    def restricao_eletrica_inequacao(self, *, df: bool) -> object: ...


def _df(source: _ElectricalCards, attr: str) -> pd.DataFrame | None:
    """Read one accessor as a DataFrame, treating an absent card as empty."""
    frame = getattr(source, attr)(df=True)
    if frame is None or getattr(frame, "empty", True):
        return None
    return frame


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
    n_stages = len(calendar)
    values_by_code: dict[int, dict[tuple[int, int | None], float]] = {}
    for _, row in frame.iterrows():
        code = int(row["codigo_alias"])
        context = f"alias {code} value"
        start = _stage_index(int(row["estagio_inicio"]), n_stages, context)
        end = _stage_index(int(row["estagio_fim"]), n_stages, context)
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
    """Read ``RESTRICAO-ELETRICA-HORIZONTE-PERIODO`` into ``{codigo_restricao:
    (stage_start, stage_end)}``."""
    frame = _df(restricoes, "restricao_eletrica_horizonte_periodo")
    if frame is None:
        return {}
    n_stages = len(calendar)
    horizonte: dict[int, tuple[int, int]] = {}
    for _, row in frame.iterrows():
        code = int(row["codigo_restricao"])
        context = f"restriction {code} horizonte"
        start = _stage_index(int(row["estagio_inicio"]), n_stages, context)
        end = _stage_index(int(row["estagio_fim"]), n_stages, context)
        horizonte[code] = (start, end)
    return horizonte


def _read_formula_limits(
    restricoes: _ElectricalCards,
    calendar: Sequence[OperativeStage],
) -> dict[int, dict[tuple[int, int | None], tuple[float | None, float | None]]]:
    """Read ``RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR`` into
    ``{codigo_restricao: {(stage_index, patamar): (lower, upper)}}``,
    expanding each row's stage range and normalizing the ``±1E+31`` sentinel
    to ``None``."""
    frame = _df(restricoes, "restricao_eletrica_limite_formula_periodo_patamar")
    if frame is None:
        return {}
    n_stages = len(calendar)
    limits: dict[
        int, dict[tuple[int, int | None], tuple[float | None, float | None]]
    ] = {}
    for _, row in frame.iterrows():
        code = int(row["codigo_restricao"])
        context = f"restriction {code} limits"
        start = _stage_index(int(row["estagio_inicio"]), n_stages, context)
        end = _stage_index(int(row["estagio_fim"]), n_stages, context)
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
    n_stages = len(calendar)
    overrides: dict[int, list[PeriodPatamarOverride]] = {}
    for _, row in frame.iterrows():
        code = int(row["codigo_restricao"])
        context = f"restriction {code} inequacao override"
        start = _stage_index(int(row["estagio_inicio"]), n_stages, context)
        end = _stage_index(int(row["estagio_fim"]), n_stages, context)
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


def _read_violations(restricoes: _ElectricalCards) -> dict[int, ViolationTreatment]:
    """Read ``RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO`` into
    ``{codigo_restricao: ViolationTreatment}``. Absent entirely, or absent for
    a given restriction, is not an error — it simply yields no entry."""
    frame = _df(restricoes, "restricao_eletrica_tratamento_violacao")
    if frame is None:
        return {}
    return {
        int(row["codigo_restricao"]): ViolationTreatment(
            enabled=True,
            penalty=_opt_float(row["custo_violacao"]),
        )
        for _, row in frame.iterrows()
    }


def _read_restrictions(
    restricoes: _ElectricalCards,
    calendar: Sequence[OperativeStage],
) -> dict[int, ElectricalRestriction]:
    """Join the FORMULA and INEQUACAO restriction cards to their per-(period,
    patamar) limit/override rows, horizonte, habilita, and violation rows by
    ``codigo_restricao``.

    A code with ``RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR`` rows but no
    FORMULA or INEQUACAO card is an INEQUACAO restriction declared by those
    rows alone. Period rows that cannot apply to any restriction (inequality
    rows on a FORMULA code, limit rows on a non-FORMULA code) are reported.
    """
    formula = _df(restricoes, "restricao_eletrica_formula")
    inequacao = _df(restricoes, "restricao_eletrica_inequacao")
    overrides = _read_inequacao_overrides(restricoes, calendar)
    if formula is None and inequacao is None and not overrides:
        return {}

    horizonte = _read_horizonte(restricoes, calendar)
    limits = _read_formula_limits(restricoes, calendar)
    habilita = _read_habilita(restricoes)
    violations = _read_violations(restricoes)

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
                violation=violations.get(code),
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
                violation=violations.get(code),
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
            violation=violations.get(code),
        )

    _emit_unused_period_rows(
        [
            ("RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR", code)
            for code in sorted(c for c in overrides if restrictions[c].is_formula)
        ]
        + [
            ("RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR", code)
            for code in sorted(
                c
                for c in limits
                if c not in restrictions or not restrictions[c].is_formula
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
                "inequality restrictions, and limit period rows only to formula "
                "restrictions. The conversion ignores them, so the case runs "
                "without the limits they set."
            ),
            table=DiagnosticTable(
                columns=["Card", "Restriction"], rows=[list(row) for row in rows]
            ),
            remediation=(
                "Move each listed row to the card that fits its restriction, or "
                "remove it."
            ),
        )
    )


def read_libs_electrical(
    restricoes: _ElectricalCards,
    calendar: Sequence[OperativeStage],
    *,
    gnl_thermal_codes: frozenset[int] = frozenset(),
) -> LibsElectricalModel | None:
    """Read every period-keyed long-form electrical card into a structured model.

    Joins the ``RESTRICAO-ELETRICA-FORMULA``/``-INEQUACAO`` restrictions (and
    the inequalities declared by ``-INEQUACAO-PERIODO-PATAMAR`` rows alone) to
    their per-(period,patamar) limit/override rows, horizonte, habilita, rule,
    and violation rows by ``codigo_restricao``, and maps every période (1-based)
    onto its 0-based stage index against *calendar*.

    Returns ``None`` when the deck carries no electrical long-form card at all
    (no expressions, aliases, restrictions, or rules), so callers can no-op
    cleanly instead of handling an empty-but-present model.
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
        gnl_thermal_codes=gnl_thermal_codes,
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
    with no Itaipu import); callers then place no ``IV`` load and resolve any
    ``carga_ande`` reference to a fail-loud diagnostic.

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
    if _ITAIPU_CODE not in case.id_map.hydro_codes:
        return None
    carga_ande = read_carga_ande(case.dadger, case.calendar)
    if not carga_ande:
        return None
    return int(case.hidr.loc[_ITAIPU_CODE, "submercado"]), carga_ande


#: Region names accepted as a ``peq_`` subsystem suffix besides the ``SB``
#: mnemonics, mapped to the mnemonic they stand for.
_PEQ_SUBMARKET_ALIASES = {"SECO": "SE", "SUL": "S"}


#: The activation-rule grammar's six comparison operators,
#: longest-match: the two-character operators must be tried before their
#: single-character prefix (``<=``/``>=`` before ``<``/``>``).
_COMPARISON_TWO_CHAR_OPS = frozenset({"<=", ">=", "==", "!="})
_COMPARISON_ONE_CHAR_OPS = frozenset({"<", ">"})
#: Every character that can start or continue a comparison operator — used to
#: recognize a malformed operator (e.g. ``<>``) fast, without first parsing
#: an operand.
_COMPARISON_OP_CHARS = frozenset("<>=!")

#: The closed six-operator -> predicate map that :func:`evaluate_rule` applies
#: to each :class:`Comparison`'s two evaluated sides.
_COMPARISON_FUNCS: Mapping[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass(frozen=True)
class Comparison:
    """One parsed ``operand <op> operand`` comparison from an activation
    rule's ``&``-conjunction.

    ``left``/``right`` are the two operand sides, each a flat list of
    :class:`ParsedTerm` from :func:`parse_linear_expression`; every term on
    either side classifies as :attr:`Bucket.B` (:func:`parse_activation_rule`
    enforces this — an activation rule gates on input data, never a novomodelo
    decision term). ``op`` is one of the six recognized comparison
    operators, verbatim.
    """

    left: list[ParsedTerm]
    op: str
    right: list[ParsedTerm]


@dataclass(frozen=True)
class ActivationRule:
    """A parsed ``RESTRICAO-ELETRICA-REGRA-ATIVACAO`` boolean rule: an AND
    (the grammar's top-level ``&``) of :class:`Comparison`\\ s.

    An empty ``comparisons`` tuple models "always active" — see
    :func:`is_always_active` — for a restriction with no ``HABILITA``
    binding at all, which never has rule text to parse.
    """

    comparisons: tuple[Comparison, ...]


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split *text* on every top-level (parenthesis-depth-0) occurrence of
    *sep*.

    The observed activation-rule grammar never lets a parenthesized group
    span across a top-level ``&`` — this only affects which ``&`` occurrences
    are split points (one inside a parenthesized sub-expression is not); it
    is *not* violated by a normal nested ``&`` inside parentheses, which
    simply is not a split point.

    Raises
    ------
    MalformedElectricalExpression
        When *text*'s parentheses are unbalanced (a stray closing paren, or
        one left open at the end) — this would mean some paren group is not
        self-contained the way the grammar requires.
    """
    parts: list[str] = []
    depth = 0
    start = 0
    for pos, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise MalformedElectricalExpression(
                    f"unbalanced parentheses in activation rule {text!r}"
                )
        elif char == sep and depth == 0:
            parts.append(text[start:pos])
            start = pos + 1
    if depth != 0:
        raise MalformedElectricalExpression(
            f"unbalanced parentheses in activation rule {text!r}"
        )
    parts.append(text[start:])
    return parts


def _split_comparison(conjunct: str) -> tuple[str, str, str]:
    """Split one activation-rule conjunct into ``(left_text, op, right_text)``
    at its top-level comparison operator, longest-match (the two-character
    operators ``<=``/``>=``/``==``/``!=`` win over their single-character
    prefix ``<``/``>``).

    Scans outside of any parenthesized group (an operand may itself contain
    parentheses, e.g. ``demanda(3)+demanda(4)``) for the first character in
    :data:`_COMPARISON_OP_CHARS`.

    Raises
    ------
    MalformedElectricalExpression
        When the conjunct carries no comparison operator at all, or an
        operator-like character sequence that is not one of the six
        recognized operators (e.g. ``<>``, a lone ``=``) — both name the
        conjunct verbatim.
    """
    depth = 0
    pos = 0
    length = len(conjunct)
    while pos < length:
        char = conjunct[pos]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and char in _COMPARISON_OP_CHARS:
            two_char = conjunct[pos : pos + 2]
            if two_char in _COMPARISON_TWO_CHAR_OPS:
                return conjunct[:pos], two_char, conjunct[pos + 2 :]
            next_char = conjunct[pos + 1 : pos + 2]
            if next_char in _COMPARISON_OP_CHARS:
                raise MalformedElectricalExpression(
                    f"unrecognized comparison operator {two_char!r} in "
                    f"activation-rule conjunct {conjunct!r}"
                )
            if char not in _COMPARISON_ONE_CHAR_OPS:
                raise MalformedElectricalExpression(
                    f"unrecognized comparison operator {char!r} in "
                    f"activation-rule conjunct {conjunct!r}"
                )
            return conjunct[:pos], char, conjunct[pos + 1 :]
        pos += 1
    raise MalformedElectricalExpression(
        f"no comparison operator found in activation-rule conjunct {conjunct!r} "
        "(recognized operators: <, <=, >, >=, ==, !=)"
    )


def _require_bucket_b(terms: Sequence[ParsedTerm], conjunct: str) -> None:
    """Raise :class:`MalformedElectricalExpression` naming the offending term
    when any *terms* is not bucket B — an activation-rule operand is always
    input data, never a novomodelo decision term (bucket A) or the
    available-power bucket-C term."""
    for term in terms:
        if classify_term(term) is not Bucket.B:
            raise MalformedElectricalExpression(
                f"activation-rule operand {term_text(term)} in conjunct "
                f"{conjunct!r} is not input data; an activation rule reads "
                "only input data, never a decision"
            )


def parse_activation_rule(text: str, model: LibsElectricalModel) -> ActivationRule:
    """Parse one ``RESTRICAO-ELETRICA-REGRA-ATIVACAO`` boolean-rule string
    into an :class:`ActivationRule`.

    Splits *text* on its top-level ``&`` conjunctions (:func:`_split_top_level`),
    splits each conjunct on its comparison operator, longest-match
    (:func:`_split_comparison`), and parses each side with
    :func:`parse_linear_expression` — the comparison operands are the same
    input-data linear expressions that function already parses. Every
    resulting operand term must classify as :attr:`Bucket.B`
    (:func:`classify_term`); an activation rule gates on input data, never a
    novomodelo decision term.

    This is the closed six-operator + ``&`` grammar verified against every
    rule in the source model's decks — it has no ``|``
    (disjunction) and no ``se`` literal at the rule-text level, so neither is
    supported here; a deck that needs either is out of scope, not a silent
    extension.

    Raises
    ------
    MalformedElectricalExpression
        Propagated from :func:`_split_top_level` on unbalanced parentheses,
        from :func:`_split_comparison` on a missing/unrecognized operator,
        from an operand's own parse, or raised here when an operand term does
        not classify as bucket B (naming the offending term).
    """
    comparisons = tuple(
        _parse_comparison(conjunct, model) for conjunct in _split_top_level(text, "&")
    )
    return ActivationRule(comparisons=comparisons)


def _parse_comparison(conjunct: str, model: LibsElectricalModel) -> Comparison:
    """Parse one ``&``-delimited conjunct into a :class:`Comparison`,
    enforcing the bucket-B operand guard on both sides."""
    left_text, op, right_text = _split_comparison(conjunct)
    left = parse_linear_expression(left_text, model)
    right = parse_linear_expression(right_text, model)
    _require_bucket_b(left, conjunct)
    _require_bucket_b(right, conjunct)
    return Comparison(left=left, op=op, right=right)


def evaluate_rule(rule: ActivationRule, context: DataContext) -> bool:
    """Evaluate one parsed activation rule against a per-(stage,block)
    :data:`DataContext`.

    Returns the AND of every :class:`Comparison`: each side sums to
    ``Σ coefficient * context(term)`` over its operand terms
    (:func:`evaluate_terms`), and the six comparison operators map onto the
    matching predicate (:data:`_COMPARISON_FUNCS`). An empty
    ``rule.comparisons`` — :func:`is_always_active` — evaluates to ``True``,
    the vacuous AND.

    Pure and deterministic given *context*: this engine never resolves a
    bucket-B token's value itself, so the same rule can be evaluated against
    any (stage, block) cell just by supplying a different *context*.
    """
    return all(
        _COMPARISON_FUNCS[comparison.op](
            evaluate_terms(comparison.left, context),
            evaluate_terms(comparison.right, context),
        )
        for comparison in rule.comparisons
    )


def is_always_active(rule: ActivationRule) -> bool:
    """Whether *rule* is the vacuous always-active rule.

    A restriction with no ``HABILITA`` binding has no rule text to parse at
    all — model that as the empty-conjunction :class:`ActivationRule`
    (``comparisons=()``), which is always active, subject only to the
    restriction's own ``horizonte``.
    """
    return len(rule.comparisons) == 0


# ---------------------------------------------------------------------------
# bucket-B substitution + per-(stage,block) bound assembly
# ---------------------------------------------------------------------------


class _UnresolvableBucketBTerm(Exception):
    """Internal control-flow signal: one bucket-B term's data lookup failed
    at a (stage, block) cell — a ``carga_ande``
    reference with no ``RI`` register, a ``demanda``/``val_demanda``
    reference to an unknown submarket, or an ``alias`` reference with no
    value at or before the cell.

    Deliberately its own type: :func:`evaluate_se`'s refusal of a malformed or
    decision-bearing ``se(...)`` branch must propagate (fail loud), never be
    caught and turned into a drop. :func:`assemble_bound` is the only place that
    catches this signal, converting it into a ``Severity.WARNING``
    :class:`Diagnostic` and dropping the whole constraint (skip-not-partial)
    rather than emitting a partial bound.
    """


def _resolve_alias_value(
    alias: ElectricalAlias, stage_index: int, patamar: int
) -> float | None:
    """Resolve one ``ALIAS-ELETRICO``'s value at ``(stage_index, patamar)``.

    At the current stage, an exact ``patamar`` match wins over the ``NA``
    (all-blocks, ``patamar=None``) entry; when the current stage carries
    neither, the value is inherited from the closest earlier stage that
    declares one (checked with the same exact-then-``NA`` precedence) —
    mirroring ``constraint_registers._forward_fill_bounds``'s "most recently
    declared at or before" idiom, generalized with the patamar/``NA``
    precedence applied at each candidate stage.

    Returns ``None`` when *alias* declares no value at or before
    *stage_index* at all.
    """
    for candidate in range(stage_index, -1, -1):
        if (candidate, patamar) in alias.values:
            return alias.values[(candidate, patamar)]
        if (candidate, None) in alias.values:
            return alias.values[(candidate, None)]
    return None


def _block_load(
    demand: Mapping[tuple[int, int], list[float]],
    bus_id: int,
    stage_index: int,
    block_index: int,
) -> float:
    """One bus's block load at ``(stage_index, block_index)``, or ``0.0``
    when the deck's ``DP`` declares no row for that (bus, stage) — mirrors
    ``load.convert_load_stats``'s "a bus absent from ``DP`` carries zero
    load" rule."""
    block_loads = demand.get((bus_id, stage_index))
    return 0.0 if block_loads is None else block_loads[block_index]


def _submarket_bus_id(id_map: DecompIdMap, term: ParsedTerm) -> int:
    """Map a ``demanda``/``val_demanda`` term's submarket code
    (``term.args[0]``) to its 0-based bus id.

    Raises
    ------
    _UnresolvableBucketBTerm
        When the code is not a declared subsystem (:meth:`DecompIdMap.
        bus_id` raises ``KeyError``) — the "unknown submarket" case.
    """
    code = term.args[0]
    try:
        return id_map.bus_id(code)
    except KeyError as err:
        raise _UnresolvableBucketBTerm(
            f"{term.token}({code}) names an undeclared submarket"
        ) from err


def _peq_block_generation(
    pq_generation: Mapping[tuple[int, str], Sequence[Sequence[float]]],
    id_map: DecompIdMap,
    term: ParsedTerm,
    stage_index: int,
    block_index: int,
) -> float:
    """A ``peq_<name>_<sub>`` term's ``PQ`` series generation (MW) at one cell.

    The suffix is an ``SB`` mnemonic or a :data:`_PEQ_SUBMARKET_ALIASES`
    region name, compared case-insensitively; the series name must match a
    ``PQ`` record of that subsystem exactly.

    Raises
    ------
    _UnresolvableBucketBTerm
        When the suffix names no declared subsystem, or that subsystem has no
        ``PQ`` series of that name.
    """
    assert term.pq_name is not None and term.pq_submarket is not None
    reference = term_text(term)
    mnemonics = [name.strip().upper() for name in id_map.bus_names]
    suffix = term.pq_submarket.upper()
    if suffix not in mnemonics:
        suffix = _PEQ_SUBMARKET_ALIASES.get(suffix, suffix)
    if suffix not in mnemonics:
        raise _UnresolvableBucketBTerm(
            f"{reference} names the undeclared subsystem {term.pq_submarket}"
        )
    sub_code = id_map.bus_codes[mnemonics.index(suffix)]
    series = pq_generation.get((sub_code, term.pq_name))
    if series is None:
        raise _UnresolvableBucketBTerm(
            f"{reference} names no PQ series {term.pq_name} of subsystem {sub_code}"
        )
    return series[stage_index][block_index]


def _gnl_block_generation(
    gnl_generation: Mapping[int, Sequence[Sequence[float] | None]],
    term: ParsedTerm,
    stage_index: int,
    block_index: int,
) -> float:
    """A GNL plant's ``ger_usit`` at one cell: its ``gl`` commanded generation (MW).

    Raises
    ------
    UnsupportedSourceError
        When the plant's generation is not commanded at the cell's stage
        (no ``gl`` week starts in it): the term would be a decision the
        abatement cannot represent, so the conversion aborts.
    """
    code = term.args[0]
    per_stage = gnl_generation.get(code)
    blocks = per_stage[stage_index] if per_stage is not None else None
    if blocks is None:
        raise UnsupportedSourceError(
            f"ger_usit({code}) names a GNL plant whose generation no dadgnl GL "
            f"record commands in período {stage_index + 1}; the conversion "
            "folds a GNL plant's generation into the bound only where GL "
            "commands it"
        )
    return blocks[block_index]


def build_data_context(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    model: LibsElectricalModel,
    gnl_model: GnlCommitmentModel | None = None,
) -> Callable[[int, int], DataContext]:
    """Build the per-(stage,block) :data:`DataContext` factory that drives
    :func:`assemble_bound` and the activation gate.

    :data:`DataContext` (a ``Callable[[ParsedTerm], float]``) is
    defined for exactly *one* ``(stage_index, block_index)`` cell — see its
    own docstring — and *build_data_context*'s own signature carries no such
    cell, so it returns a **factory**: call the returned closure with a cell
    to get back the :data:`DataContext` scoped to it, ready to hand to
    :func:`assemble_bound`/:func:`evaluate_rule`. The submarket-demand map
    (:func:`~novomodelo_bridge.decomp.load._per_stage_block_loads`) and the
    ``carga_ande`` series (:func:`read_carga_ande`) are each read once and
    captured — along with *model* — in the returned closure, rather than
    re-read per cell.

    Resolves every bucket-B token:

    - ``demanda(X)``/``val_demanda(X)`` — submarket ``X``'s (the deck's
      ``codigo_submercado``) block load, ``0.0`` when ``DP`` declares no row
      for that (bus, stage);
    - ``demanda_sin``/``val_demanda_sin`` — the block load summed over every
      declared submarket;
    - ``carga_ande`` — the series from :func:`read_carga_ande`;
    - a ``peq_<name>_<sub>`` token — that ``PQ`` series' block generation
      (:func:`_peq_block_generation`);
    - a GNL plant's ``ger_usit`` — its commanded ``gl`` generation at the
      cell (:func:`_gnl_block_generation`), read from *gnl_model*;
    - an ``alias`` token — the model alias's value
      (:func:`_resolve_alias_value`: stage-inherited, ``NA``-patamar applies
      to all blocks);
    - the numeric-constant token — ``1.0``, never the constant's own value:
      the value already lives in
      ``term.coefficient``, which :func:`evaluate_terms` applies directly;
    - a ``se(cond, X, Y)`` conditional — :func:`evaluate_se`.

    Raises
    ------
    _UnresolvableBucketBTerm
        From the returned per-cell context, when a ``demanda``/
        ``val_demanda`` term names an unknown submarket, a ``carga_ande``
        term is referenced but the deck declares no ``RI`` register (or no
        series for that stage), or an ``alias`` term has no value at or
        before the cell. :func:`assemble_bound` is the sole catcher —
        turning this into a ``Severity.WARNING`` diagnostic and dropping the
        constraint (skip-not-partial), never a silently wrong bound.
    UnsupportedSourceError
        From the returned per-cell context, when a GNL plant's ``ger_usit``
        is not commanded at the cell (:func:`_gnl_block_generation`).
    """
    calendar = case.calendar
    demand = _per_stage_block_loads(case.dadger, id_map, calendar)
    carga_ande_series = read_carga_ande(case.dadger, calendar)
    pq_generation = pq_block_generation(case.dadger, calendar)
    gnl_generation = (
        gnl_block_commitments(gnl_model, calendar) if gnl_model is not None else {}
    )
    n_submarkets = len(id_map.bus_codes)

    # TRACKED NOVOMODELO-GAP (C36): the case has no load term in a generic
    # constraint, so the demand is folded into the bound and a bare
    # demanda/demanda_sin multiplier is not added to the CMO as the source
    # model does; decomp-electrical-demand-in-cmo lists them.
    def context_for_cell(stage_index: int, block_index: int) -> DataContext:
        def resolve(term: ParsedTerm) -> float:
            if term.token == CONST_TOKEN:
                return 1.0
            if term.token in ("demanda", "val_demanda"):
                bus_id = _submarket_bus_id(id_map, term)
                return _block_load(demand, bus_id, stage_index, block_index)
            if term.token in ("demanda_sin", "val_demanda_sin"):
                return sum(
                    _block_load(demand, bus_id, stage_index, block_index)
                    for bus_id in range(n_submarkets)
                )
            if term.token == "carga_ande":
                values = carga_ande_series.get(stage_index)
                if values is None:
                    raise _UnresolvableBucketBTerm(
                        "carga_ande has no RI series on this stage"
                    )
                return values[block_index]
            if term.token == ALIAS_TOKEN:
                assert term.alias_name is not None  # parser invariant
                alias = model.aliases[term.alias_name]
                patamar = block_index + 1
                value = _resolve_alias_value(alias, stage_index, patamar)
                if value is None:
                    raise _UnresolvableBucketBTerm(
                        f"alias {term.alias_name} has no value on this block "
                        "of this stage or an earlier one"
                    )
                return value
            if term.token == PEQ_TOKEN:
                return _peq_block_generation(
                    pq_generation, id_map, term, stage_index, block_index
                )
            if term.token == GNL_TOKEN:
                return _gnl_block_generation(
                    gnl_generation, term, stage_index, block_index
                )
            if term.token == SE_TOKEN:
                return evaluate_se(term, model, stage_index, block_index, resolve)
            raise ValueError(
                f"Cannot resolve electrical-constraint data term {term.token!r}."
            )

        return resolve

    return context_for_cell


def evaluate_se(
    term: ParsedTerm,
    model: LibsElectricalModel,
    stage_index: int,
    block_index: int,
    ctx: DataContext,
) -> float:
    """Evaluate one structural ``se(cond, X, Y)`` term to a float for one
    (stage, block) cell.

    ``cond`` is parsed and evaluated by reusing the activation-rule
    engine (:func:`parse_activation_rule` + :func:`evaluate_rule`) — the
    same comparison grammar an activation rule's own condition uses —
    selecting :attr:`ParsedTerm.branch_true` when it evaluates ``True``,
    else :attr:`ParsedTerm.branch_false`.

    Both branches are parsed (:func:`parse_linear_expression`) and classified
    (:func:`classify_terms`) before either is selected, so a malformed branch
    fails the cell even where the other one is taken. A selected branch is
    only safe to fold to a single float when it is
    entirely bucket B (input data): folding a branch that carries a
    bucket-A (novomodelo decision) or bucket-C (``disp_usih``) term would fold
    that term into a plain number and silently drop it from the LP. ``se``
    does not appear on the target deck (gating is via
    ``REGRA-ATIVACAO``, not inline ``se()``), so this guard is defensive
    hardening, not a gap in coverage — keeping a decision-bearing branch's
    bucket-A/-C terms structural (folding only its bucket-B terms, the way
    :func:`assemble_bound` does for a whole restriction) is out of scope
    here; this fails loud instead.

    Raises
    ------
    MalformedElectricalExpression
        When the condition or either branch is malformed.
    UnsupportedSourceError
        When the selected branch carries any non-bucket-B term (naming it and
        the cell).
    ValueError
        When *term* is not a ``se`` term or is missing its condition/branch
        text (a caller contract violation — the parser always populates all
        three for a ``se`` token).
    """
    if term.token != SE_TOKEN:
        raise ValueError(f"evaluate_se: expected a 'se' term, got {term.token!r}")
    if term.condition is None or term.branch_true is None or term.branch_false is None:
        raise ValueError(
            f"evaluate_se: 'se' term is missing condition/branches: {term!r}"
        )

    rule = parse_activation_rule(term.condition, model)
    true_partition = classify_terms(parse_linear_expression(term.branch_true, model))
    false_partition = classify_terms(parse_linear_expression(term.branch_false, model))
    selected_text, partition = (
        (term.branch_true, true_partition)
        if evaluate_rule(rule, ctx)
        else (term.branch_false, false_partition)
    )
    decision_terms = partition[Bucket.A] + partition[Bucket.C]
    if decision_terms:
        offender = decision_terms[0]
        raise UnsupportedSourceError(
            f"in período {stage_index + 1}, patamar {block_index + 1}, se(...) "
            f"takes the branch {selected_text!r}, which carries the decision "
            f"term {term_text(offender)}; the conversion folds a se(...) "
            "branch into the bound only when it reads input data alone"
        )
    return evaluate_terms(partition[Bucket.B], ctx)


_FORMULA_LIMITS_CARD = "RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR"
_RESERVE_PATTERN_ONLY = (
    "the conversion rewrites disp_usih only in the reserve pattern "
    "disp_usih(h) - ger_usih(h) >= R"
)


def _fold_formula_bound(
    restriction: ElectricalRestriction,
    stage_index: int,
    patamar: int,
    ctx: DataContext,
    lhs_partition: Mapping[Bucket, list[ParsedTerm]],
) -> AssembledBound:
    """Fold a FORMULA restriction's LHS bucket-B sum into its per-(stage,
    patamar) two-sided limit, keeping its LHS bucket-A terms verbatim as the
    surviving structural expression — a FORMULA restriction is already in
    novomodelo's single-LHS ``lower <= expr <= upper`` shape, so no sign flip is
    ever needed here.

    ``restriction.limits`` is looked up at an exact ``(stage_index,
    patamar)`` match, falling back to the ``NA`` (all-blocks,
    ``patamar=None``) entry for that stage. The limit's two endpoints are
    already numeric (DECOMP's ``±1E+31`` sentinel already normalized to
    ``None`` by the reader — see :func:`_opt_bound`), and
    :func:`~novomodelo_bridge.core.electrical_expression.fold_formula_bound`
    moves the LHS's bucket-B sum onto them.

    *lhs_partition* is *restriction.lhs* already parsed and classified
    (:func:`assemble_bound` computes it once and passes it to every fold —
    this helper never re-parses ``restriction.lhs`` itself).

    Raises
    ------
    UnsupportedSourceError
        When *restriction* declares no limit at all for this active cell.
    """
    limit = restriction.limits.get((stage_index, patamar))
    if limit is None:
        limit = restriction.limits.get((stage_index, None))
    if limit is None:
        raise UnsupportedSourceError(
            f"no {_FORMULA_LIMITS_CARD} row covers período {stage_index + 1}, "
            f"patamar {patamar}, where the restriction is active, and the "
            "conversion does not know how DECOMP bounds it there",
            field=_FORMULA_LIMITS_CARD,
        )
    lower, upper = limit
    return fold_formula_bound(lhs_partition, lower, upper, ctx)


def _effective_inequacao_sides(
    restriction: ElectricalRestriction, stage_index: int, patamar: int
) -> tuple[str, str, str] | None:
    """The ``(lhs, operator, rhs)`` DSL strings in effect for one INEQUACAO
    restriction's ``(stage_index, patamar)`` cell (precedence:
    ``période-patamar > data-patamar > constante``), or ``None`` when no
    override covers the cell and the restriction has no constant form.

    Scans :attr:`ElectricalRestriction.overrides` for rows whose
    ``[stage_start, stage_end]`` range contains *stage_index*: an exact
    ``patamar == patamar`` match wins over a ``patamar is None`` (all-blocks)
    match — mirroring :func:`_resolve_alias_value`'s exact-then-``NA``
    idiom — and when neither matches, the restriction's own constant ``lhs
    op rhs`` applies. When two rows tie on specificity for this cell (two
    exact matches, or two ``NA`` matches), the later-declared one wins:
    :attr:`ElectricalRestriction.overrides` preserves the reader's row
    order, and this scan keeps overwriting its per-tier candidate as it
    walks forward, so the last match at each tier is always the one kept.

    The source model's third precedence tier, data-patamar overrides, never
    appears on this deck's cards (only période-patamar and the constant do)
    and is not synthesized here.
    """
    exact: PeriodPatamarOverride | None = None
    na: PeriodPatamarOverride | None = None
    for override in restriction.overrides:
        if not override.stage_start <= stage_index <= override.stage_end:
            continue
        if override.patamar == patamar:
            exact = override
        elif override.patamar is None:
            na = override
    chosen = exact if exact is not None else na
    if chosen is not None:
        return chosen.lhs, chosen.operator, chosen.rhs
    if restriction.lhs is None:
        return None
    assert restriction.operator is not None  # INEQUACAO invariant
    assert restriction.rhs is not None  # INEQUACAO invariant
    return restriction.lhs, restriction.operator, restriction.rhs


def _fold_inequacao_bound(
    restriction: ElectricalRestriction,
    ctx: DataContext,
    lhs_partition: Mapping[Bucket, list[ParsedTerm]],
    rhs_partition: Mapping[Bucket, list[ParsedTerm]],
) -> AssembledBound:
    """Fold an INEQUACAO restriction's bucket-B terms — both sides,
    sign-aware — into its operator's ``(lower, upper)`` interval, and move any
    RHS bucket-A term onto the LHS, sign-flipped, so the surviving expression
    is novomodelo's single-LHS form (the sign hazard's plain-INEQUACAO half:
    ``lhs op rhs`` with a bucket-A term on the RHS must not be read as if it
    were already on the LHS).

    Sums each side's bucket-B contribution (:func:`evaluate_terms`) and
    combines them with bucket-A normalized onto the LHS:
    ``bound = Σ_{B∩RHS} − Σ_{B∩LHS}``. :func:`~novomodelo_bridge.
    generic_constraint_format.sense_to_interval` then maps *restriction*'s
    operator onto the ``(lower, upper)`` shape (``>=`` lower-only, ``<=``
    upper-only, ``==`` both endpoints equal to *bound*).

    *lhs_partition*/*rhs_partition* are already-parsed-and-classified sides
    (:func:`assemble_bound` computes them once and passes them to every
    fold — this helper never re-parses either side itself); *restriction*'s
    own ``operator`` must match whichever sides fed *lhs_partition*/
    *rhs_partition* — for an INEQUACAO restriction that carries
    ``RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR`` overrides
    (:attr:`ElectricalRestriction.overrides`), that is the override
    precedence :func:`assemble_bound` already resolved via
    :func:`_effective_inequacao_sides`, not necessarily the restriction's
    own constant sides.
    """
    assert restriction.operator is not None  # INEQUACAO invariant
    lhs_b_sum = evaluate_terms(lhs_partition[Bucket.B], ctx)
    rhs_b_sum = evaluate_terms(rhs_partition[Bucket.B], ctx)
    lower, upper = sense_to_interval(restriction.operator, rhs_b_sum - lhs_b_sum)
    terms = tuple(lhs_partition[Bucket.A]) + tuple(
        negate(term) for term in rhs_partition[Bucket.A]
    )
    return AssembledBound(terms=terms, lower=lower, upper=upper)


def assemble_bound(
    restriction: ElectricalRestriction,
    model: LibsElectricalModel,
    stage_index: int,
    block_index: int,
    ctx: DataContext,
    a_h: AvailablePower | None = None,
) -> AssembledBound | None:
    """Fold *restriction*'s bucket-B (input-data) and bucket-C (``disp_usih``,
    available power) terms into its per-(stage,block) numeric bound, leaving
    only its bucket-A (decision) terms structural — and sign-canonical for
    novomodelo — in the returned :class:`AssembledBound` (the bucket-C
    fold is the reserve->gen-cap rewrite — see this module's header).

    Classifies *restriction.lhs* (and, for a FORMULA restriction, only
    *.lhs* — it has no ``.rhs``) exactly once via :func:`classify_terms`,
    then dispatches the resulting partitions to one of three folds, none of
    which re-parses either side a second time: a FORMULA restriction's
    per-(stage,patamar) two-sided limit is reduced by its LHS's bucket-B
    sum, its LHS bucket-A terms kept verbatim (:func:`_fold_formula_bound`)
    — it never carries a ``disp_usih`` term (the documented reserve pattern
    is INEQUACAO-shaped only; a FORMULA restriction with a bucket-C term
    raises, below). An INEQUACAO restriction classifies its *effective*
    ``(lhs, operator, rhs)`` instead — :func:`_effective_inequacao_sides`
    resolves any ``RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR`` override
    in effect at this cell (precedence: exact-patamar >
    NA-patamar > the restriction's own constant sides) before classification
    ever runs, so a cell with no matching override classifies the constant
    sides verbatim and one with a matching override classifies that
    override's sides instead. With no bucket-C term, the effective sides
    fold sign-aware into the effective operator's ``(lower, upper)``
    interval, moving any RHS bucket-A term onto the LHS negated
    (:func:`_fold_inequacao_bound`);
    with a ``disp_usih`` term (on either side), they fold via
    :func:`_fold_reserve_disp_usih` instead, which requires *a_h* and
    negates the surviving ``ger_usih`` terms to match its operator flip —
    both dispatch/guard branches are unchanged by the override; only the
    sides they classify are.

    *ctx* must already be scoped to this ``(stage_index, block_index)`` cell
    — the shape :func:`build_data_context`'s returned factory produces.
    *stage_index*/*block_index* are used here to select *restriction*'s own
    per-cell limit/patamar, not to re-scope *ctx*. *a_h* is required exactly
    when *restriction* carries a bucket-C term; a restriction with none never
    touches it, so a caller with no bucket-C term passes nothing.

    Returns ``None`` — after emitting one ``Severity.WARNING``
    :class:`Diagnostic` naming the unresolvable term and the cell — when a
    bucket-B term cannot be resolved (:class:`_UnresolvableBucketBTerm` from
    *ctx*) or a bucket-C ``disp_usih`` term's available power cannot be
    resolved (:class:`_UnresolvableDispUsih` from *a_h*, Requirement 5): the
    whole constraint is dropped rather than emitted with a partial bound
    (skip-not-partial), mirroring ``constraints.py``'s resolver diagnostics.

    Raises
    ------
    UnsupportedSourceError
        When a FORMULA restriction carries no limit at all for this cell,
        when a FORMULA restriction carries a bucket-C term (out of the
        documented pattern's scope), when a restriction uses ``disp_usih``
        outside the documented reserve sign structure
        (:func:`_fold_reserve_disp_usih`'s CRITICAL-pitfall guard), or from
        the cell's data context.
    ValueError
        When an INEQUACAO restriction carries a bucket-C term but *a_h* is
        ``None``, or propagated from :func:`~novomodelo_bridge.core.
        generic_constraint_format.sense_to_interval` on an unrecognized
        INEQUACAO operator — none of these is the resolver-unresolvable case
        this function otherwise catches.
    """
    patamar = block_index + 1
    try:
        if restriction.is_formula:
            assert restriction.lhs is not None  # FORMULA invariant
            lhs_partition = classify_terms(
                parse_linear_expression(restriction.lhs, model)
            )
            if lhs_partition[Bucket.C]:
                raise UnsupportedSourceError(
                    f"its RESTRICAO-ELETRICA-FORMULA carries a disp_usih term; "
                    f"{_RESERVE_PATTERN_ONLY} of a RESTRICAO-ELETRICA-INEQUACAO"
                )
            return _fold_formula_bound(
                restriction, stage_index, patamar, ctx, lhs_partition
            )

        sides = _effective_inequacao_sides(restriction, stage_index, patamar)
        if sides is None:
            raise ValueError(
                f"restriction {restriction.code} has no inequality in effect at "
                f"stage {stage_index}, patamar {patamar}; active_cells excludes "
                "such cells"
            )
        lhs, op, rhs = sides
        effective = replace(restriction, lhs=lhs, operator=op, rhs=rhs)
        lhs_partition = classify_terms(parse_linear_expression(lhs, model))
        rhs_partition = classify_terms(parse_linear_expression(rhs, model))
        if lhs_partition[Bucket.C] or rhs_partition[Bucket.C]:
            if a_h is None:
                raise ValueError(
                    f"restriction {restriction.code} carries a disp_usih "
                    "term but no AvailablePower lookup was supplied"
                )
            return _fold_reserve_disp_usih(
                effective, stage_index, a_h, ctx, lhs_partition, rhs_partition
            )
        return _fold_inequacao_bound(effective, ctx, lhs_partition, rhs_partition)
    except _UnresolvableBucketBTerm as err:
        emit(
            _unresolved_cell_diagnostic(
                "decomp-electrical-bucket-b-unresolved",
                "a data term the conversion cannot evaluate",
                [restriction.code, stage_index, block_index, str(err)],
            )
        )
        return None
    except _UnresolvableDispUsih as err:
        emit(
            _unresolved_cell_diagnostic(
                "decomp-electrical-disp-usih-unresolved",
                "a disp_usih term whose available power the conversion cannot compute",
                [restriction.code, stage_index, block_index, str(err)],
            )
        )
        return None


#: The title of each diagnostic a restriction's unresolved cell is reported under.
_UNRESOLVED_CELL_TITLES = {
    "decomp-electrical-bucket-b-unresolved": (
        "Electrical restrictions with an unresolved data term"
    ),
    "decomp-electrical-disp-usih-unresolved": (
        "Electrical restrictions with an unresolved disp_usih term"
    ),
}


def _unresolved_cell_diagnostic(code: str, what: str, row: list[object]) -> Diagnostic:
    """WARNING for one restriction left out over one unresolved cell, as a
    one-row table that :func:`~novomodelo_bridge.core.diagnostics.merge_tables`
    joins across restrictions."""
    return Diagnostic(
        code=code,
        severity=Severity.WARNING,
        category="Special constraints",
        title=_UNRESOLVED_CELL_TITLES[code],
        summary=(
            f"The listed electrical restrictions of the LIBS file have {what} on "
            "the listed stage and block. The conversion leaves each restriction "
            "out, so the case runs without these limits."
        ),
        table=DiagnosticTable(
            columns=["Restriction", "Stage", "Block", "Reason"],
            rows=[row],
            justify=["right", "right", "right", "left"],
            caption="Stage and block ids are 0-based.",
        ),
    )


# ---------------------------------------------------------------------------
# per-(stage,block) activation-gate evaluation
# ---------------------------------------------------------------------------


def _restriction_horizon(
    restriction: ElectricalRestriction, calendar: Sequence[OperativeStage]
) -> tuple[int, int]:
    """*restriction*'s horizonte stage range, defaulting to the full
    calendar when either endpoint is ``None`` (no
    ``RESTRICAO-ELETRICA-HORIZONTE-PERIODO`` row for it) — mirrors "no
    ``HABILITA`` -> always active" (:func:`is_always_active`): an absent
    horizonte is "no restriction on which stages", not "inactive
    everywhere"."""
    start = restriction.stage_start if restriction.stage_start is not None else 0
    end = (
        restriction.stage_end
        if restriction.stage_end is not None
        else len(calendar) - 1
    )
    return start, end


def active_cells(
    restriction: ElectricalRestriction,
    model: LibsElectricalModel,
    context_factory: Callable[[int, int], DataContext],
    calendar: Sequence[OperativeStage],
) -> set[tuple[int, int]]:
    """The set of ``(stage_index, block_index)`` cells where *restriction* is
    active: its horizonte stage range intersected with the cells where its
    activation rule evaluates ``True`` and, for an inequality declared only by
    period rows, with the cells those rows cover.

    Resolves *restriction*'s rule once — from its ``habilita`` binding
    (looked up in ``model.rules`` and parsed with
    :func:`parse_activation_rule`), or the vacuous always-active rule
    (:func:`is_always_active`) when ``habilita`` is ``None`` — then evaluates
    it **per (stage, block)** cell (OQ-5): every block of every in-horizonte
    stage is checked independently via :func:`evaluate_rule`, so a
    demand-band rule that holds only in the heavy-load block activates only
    that block, never the light blocks of the same stage. This
    per-(stage,block) granularity is the faithful default here; confirming
    that no restriction in the source model instead needs coarser stage-only
    (all-or-no-block) granularity is a separate census's concern, not this
    function's.

    *context_factory* is :func:`build_data_context`'s return
    value — a per-cell **factory**, not a single :data:`DataContext` — since
    a :data:`DataContext` is scoped to exactly one ``(stage_index,
    block_index)`` cell (see its own docstring) and this function evaluates
    the rule at every in-horizonte cell independently. For each candidate
    cell this calls ``context_factory(stage_index, block_index)`` to obtain
    that cell's :data:`DataContext`, then hands it to :func:`evaluate_rule`.
    This keeps *active_cells* pure given *context_factory*: no deck I/O
    beyond whatever *context_factory*'s own closure already reads once
    (requirement 5) — and when the rule is the always-active one,
    *context_factory* is never even called, since every in-horizonte cell is
    active regardless of its data.

    When the rule evaluates ``False`` in **every** in-horizonte cell, the
    returned set is empty and this function itself emits one
    ``Severity.INFO`` :class:`~novomodelo_bridge.core.diagnostics.Diagnostic` naming
    *restriction* (mirroring ``constraint_registers``'s dropped-restriction
    skip logging, as a diagnostic here rather than a bare log record).
    Raising it here keeps *active_cells* self-contained and directly
    testable (via :func:`~novomodelo_bridge.core.diagnostics.collect`) — the caller
    simply adds no bound rows for an empty set and does not need to emit
    this diagnostic itself.

    The activation rule's ``==``/``!=`` comparisons
    (:func:`evaluate_rule`/``_COMPARISON_FUNCS``) are exact float equality,
    with no tolerance — an intentional, faithful choice already made at the
    rule engine and left unchanged here; a census confirming no real rule
    gates on ``==``/``!=`` over a computed demand sum is a separate concern,
    not a reason to relax the comparison here.

    Raises
    ------
    MalformedElectricalExpression
        When *restriction*'s ``habilita`` names a ``regra_id`` absent from
        ``model.rules`` — a dangling activation reference must fail loud,
        naming the missing rule id — or when its rule is malformed.
    """
    if restriction.habilita is None:
        rule = ActivationRule(comparisons=())
    else:
        rule_text = model.rules.get(restriction.habilita)
        if rule_text is None:
            raise MalformedElectricalExpression(
                f"RESTRICAO-ELETRICA-HABILITA names activation rule "
                f"{restriction.habilita}, which no RESTRICAO-ELETRICA-REGRA-ATIVACAO "
                "record declares"
            )
        rule = parse_activation_rule(rule_text, model)

    always_active = is_always_active(rule)
    stage_start, stage_end = _restriction_horizon(restriction, calendar)

    cells: set[tuple[int, int]] = set()
    for stage_index in range(stage_start, stage_end + 1):
        n_blocks = len(calendar[stage_index].block_hours)
        for block_index in range(n_blocks):
            if (
                restriction.lhs is None
                and _effective_inequacao_sides(
                    restriction, stage_index, block_index + 1
                )
                is None
            ):
                continue
            if always_active or evaluate_rule(
                rule, context_factory(stage_index, block_index)
            ):
                cells.add((stage_index, block_index))

    if not cells:
        emit(
            Diagnostic(
                code="decomp-electrical-restriction-inactive",
                severity=Severity.INFO,
                category="Special constraints",
                title="Electrical restrictions never active",
                summary=(
                    "The activation rule of each listed electrical restriction is "
                    "false on every stage and block of its horizon, so, as in "
                    "DECOMP, it binds nowhere and the case leaves it out."
                ),
                table=DiagnosticTable(
                    columns=["Restriction", "First stage", "Last stage"],
                    rows=[[restriction.code, stage_start, stage_end]],
                    justify=["right", "right", "right"],
                    caption="Stage ids are 0-based.",
                ),
            )
        )
    return cells


# ---------------------------------------------------------------------------
# bucket-C ``disp_usih`` reserve -> gen-cap reformulation
# ---------------------------------------------------------------------------


class _UnresolvableDispUsih(Exception):
    """Internal control-flow signal: one bucket-C ``disp_usih`` term's
    available power could not be resolved (Requirement 5) — the plant is
    absent from the :class:`AvailablePower` lookup.

    Deliberately not a :class:`ValueError`, mirroring
    :class:`_UnresolvableBucketBTerm`: :func:`assemble_bound` is the sole
    catcher, converting this into a ``Severity.WARNING`` :class:`Diagnostic`
    and dropping the whole constraint (skip-not-partial) rather than
    emitting a partial bound.
    """


@dataclass(frozen=True)
class AvailablePower:
    """Per-(plant code, stage_index) maintenance-aware available power
    (``A_h``) — the reference-head constant this module's header pins for
    the ``disp_usih`` reformulation (OQ-2, approved for a weekly/monthly
    reserve margin).

    ``overlay`` is keyed by the plant's deck code, never its ``hydro_id``,
    since :func:`resolve_disp_usih` resolves a term by the code it was parsed
    with.
    """

    overlay: Mapping[tuple[int, int], float]

    def resolve(self, code: int, stage_index: int) -> float:
        """*code*'s ``A_h`` at *stage_index*.

        Raises
        ------
        _UnresolvableDispUsih
            When *code* has no available power at *stage_index*
            (Requirement 5).
        """
        value = self.overlay.get((code, stage_index))
        if value is None:
            raise _UnresolvableDispUsih(
                f"plant {code} has no available power on this stage"
            )
        return value


def build_available_power(
    id_map: DecompIdMap,
    *,
    overlay: Mapping[tuple[int, int, int], GroupBoundEntry],
) -> AvailablePower:
    """Build the :class:`AvailablePower` lookup (Requirement 1) from
    :func:`~novomodelo_bridge.decomp.converters.hydro.bounds.
    convert_hydro_group_availability`'s per-(hydro_id, hydro_unit_group_id,
    stage_id) *overlay*: each plant's
    :func:`~novomodelo_bridge.decomp.converters.hydro.bounds.
    plant_available_power`, re-keyed from ``hydro_id`` onto the plant's deck
    code.
    """
    code_by_hydro_id = {id_map.hydro_id(code): code for code in id_map.hydro_codes}
    return AvailablePower(
        overlay={
            (code_by_hydro_id[hydro_id], stage_id): total
            for (hydro_id, stage_id), total in plant_available_power(overlay).items()
        }
    )


def resolve_disp_usih(term: ParsedTerm, stage_index: int, a_h: AvailablePower) -> float:
    """*term*'s (a bucket-C ``disp_usih`` term) plant's ``A_h`` at
    *stage_index* (Requirement 2), as :func:`build_available_power` summed it.

    Raises
    ------
    _UnresolvableDispUsih
        Propagated from :meth:`AvailablePower.resolve`.
    """
    return a_h.resolve(term.args[0], stage_index)


def _flip_operator(sense: str) -> str:
    """The sense-flipped counterpart of one INEQUACAO comparison operator.

    Multiplying an inequality by ``-1`` swaps ``>=``/``<=`` and leaves
    ``==`` unchanged — exactly the operator-side half of the reserve->gen-cap
    rewrite (:func:`_fold_reserve_disp_usih`).

    Raises
    ------
    ValueError
        When *sense* is not one of ``">="``, ``"<="``, ``"=="`` — mirroring
        :func:`~novomodelo_bridge.core.generic_constraint_format.sense_to_interval`.
    """
    if sense == ">=":
        return "<="
    if sense == "<=":
        return ">="
    if sense == "==":
        return "=="
    raise ValueError(f"_flip_operator: unknown operator {sense!r}")


DISP_USIH_SUBSTITUTED = "decomp-electrical-disp-usih-substituted"


def _emit_disp_usih_substitution(
    restriction_code: int,
    plant_code: int,
    stage_index: int,
    a_h_value: float,
) -> None:
    """Emit the INFO row for one ``disp_usih -> A_h`` substitution, naming the
    plant code, the stage, and the ``A_h`` value used."""
    emit(
        Diagnostic(
            code=DISP_USIH_SUBSTITUTED,
            severity=Severity.INFO,
            category="Special constraints",
            title="Available power held constant in reserve restrictions",
            summary=(
                "The listed electrical restrictions bound a plant's reserve "
                "through disp_usih, its available power. The conversion takes "
                "that power on each listed stage as a constant, at the "
                "reference head and after maintenance, so each restriction "
                "caps the plant's generation."
            ),
            table=DiagnosticTable(
                columns=["Restriction", "Plant code", "Stage", "Available power (MW)"],
                rows=[[restriction_code, plant_code, stage_index, a_h_value]],
                justify=["right", "right", "right", "right"],
                caption="Stage ids are 0-based.",
            ),
        )
    )


def _fold_reserve_disp_usih(
    restriction: ElectricalRestriction,
    stage_index: int,
    a_h: AvailablePower,
    ctx: DataContext,
    lhs_partition: Mapping[Bucket, list[ParsedTerm]],
    rhs_partition: Mapping[Bucket, list[ParsedTerm]],
) -> AssembledBound:
    """Fold an INEQUACAO restriction's bucket-C ``disp_usih`` terms into its
    numeric bound, realizing the documented reserve->gen-cap rewrite (this
    module's header):

    .. code-block:: text

        Σ_h (disp_usih(h) − ger_usih(h)) >= R
            -> Σ_h ger_usih(h) <= Σ_h A_h − R

    Every ``disp_usih(h)`` term (coefficient ``c``) on the LHS must be
    paired with a ``ger_usih(h)`` term also on the LHS carrying the exact
    opposite coefficient (``-c``) — the documented reserve SIGN structure
    (this module's header). Substituting
    ``disp_usih(h) -> A_h`` (:func:`resolve_disp_usih`) and multiplying the
    whole relation by ``-1`` — realized on the returned
    :class:`AssembledBound` by negating every surviving ``ger_usih`` term
    (this is the sign fix: a caller reading ``restriction.lhs`` verbatim would
    keep its original negative coefficient, silently inverting the bound) —
    flips the operator's sense; any bucket-B terms on either side fold into
    the same bound.

    Raises
    ------
    UnsupportedSourceError
        When *restriction* uses ``disp_usih`` outside this documented
        pattern: a bucket-C or bucket-A term on the RHS, a bucket-A term on
        the LHS that is not ``ger_usih``, more than one ``ger_usih`` term
        for the same plant, a ``disp_usih(h)`` with no paired ``ger_usih(h)``
        of the exact opposite coefficient, or a ``ger_usih(h)`` with no
        paired ``disp_usih(h)`` — the CRITICAL pitfall: fail loud rather
        than silently mis-rewrite a shape the pattern
        does not cover.
    _UnresolvableDispUsih
        Propagated from :func:`resolve_disp_usih` when a paired plant's
        ``A_h`` cannot be resolved (Requirement 5) — the caller
        (:func:`assemble_bound`) is the sole catcher.
    """
    assert restriction.operator is not None  # INEQUACAO invariant

    if rhs_partition[Bucket.C] or rhs_partition[Bucket.A]:
        raise UnsupportedSourceError(
            "the right-hand side carries a decision or disp_usih term; "
            f"{_RESERVE_PATTERN_ONLY}"
        )

    disp_terms = lhs_partition[Bucket.C]
    ger_terms = lhs_partition[Bucket.A]
    non_ger_terms = [term for term in ger_terms if term.token != "ger_usih"]
    if non_ger_terms:
        raise UnsupportedSourceError(
            f"the decision term {term_text(non_ger_terms[0])} sits beside "
            f"disp_usih; {_RESERVE_PATTERN_ONLY}"
        )

    ger_by_plant = {term.args[0]: term for term in ger_terms}
    if len(ger_by_plant) != len(ger_terms):
        raise UnsupportedSourceError(
            "two ger_usih terms beside disp_usih name the same plant; "
            f"{_RESERVE_PATTERN_ONLY}"
        )

    reserve_sum = 0.0
    for disp_term in disp_terms:
        code = disp_term.args[0]
        ger_term = ger_by_plant.pop(code, None)
        if ger_term is None or ger_term.coefficient != -disp_term.coefficient:
            raise UnsupportedSourceError(
                f"disp_usih({code}) has no ger_usih({code}) of the opposite "
                f"coefficient; {_RESERVE_PATTERN_ONLY}"
            )
        a_h_value = a_h.resolve(code, stage_index)
        reserve_sum += disp_term.coefficient * a_h_value
        _emit_disp_usih_substitution(restriction.code, code, stage_index, a_h_value)

    if ger_by_plant:
        leftover_code = next(iter(ger_by_plant))
        raise UnsupportedSourceError(
            f"ger_usih({leftover_code}) has no disp_usih({leftover_code}) "
            f"beside it; {_RESERVE_PATTERN_ONLY}"
        )

    lhs_b_sum = evaluate_terms(lhs_partition[Bucket.B], ctx)
    rhs_b_sum = evaluate_terms(rhs_partition[Bucket.B], ctx)
    bound_value = reserve_sum + lhs_b_sum - rhs_b_sum
    lower, upper = sense_to_interval(_flip_operator(restriction.operator), bound_value)
    return AssembledBound(
        terms=tuple(negate(term) for term in ger_terms), lower=lower, upper=upper
    )
