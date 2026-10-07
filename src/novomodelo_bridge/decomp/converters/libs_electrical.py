"""Read the LIBs-era electrical special-constraint file into a structured model,
and parse its linear-expression DSL into flat terms.

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
it **verbatim** and does not evaluate it. :func:`parse_linear_expression` turns
one such string into a flat list of :class:`ParsedTerm`, expanding
``EXPRESSAO-ELETRICA`` named-expression references and ``re(R)`` restriction
references (multiplying through the enclosing coefficient) while keeping
``ALIAS-ELETRICO`` references and ``se(cond, X, Y)`` conditionals unresolved
for a later resolver; :func:`classify_term`/:func:`classify_terms` then sort
each parsed term into the source model's A/B/C bucket trichotomy (spec §0).
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

Bucket C (``disp_usih``, available power) has no cobre decision-variable
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
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

from cobre_bridge.core.diagnostics import Diagnostic, Severity, emit
from cobre_bridge.core.generic_constraint_format import sense_to_interval
from cobre_bridge.decomp.converters.hydro import _rated_envelope
from cobre_bridge.decomp.load import _per_stage_block_loads

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from idecomp.decomp import Dadger
    from idecomp.libs.restricoes import Restricoes

    from cobre_bridge.decomp.case import DecompCase
    from cobre_bridge.decomp.converters.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.group_bounds import GroupBoundEntry
    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage

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

    Maps to cobre's generic-constraint ``slack{enabled, penalty}`` shape:
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
    ``overrides``. ``stage_start``/``stage_end`` come from
    ``RESTRICAO-ELETRICA-HORIZONTE-PERIODO`` (``None`` when absent);
    ``habilita`` is the bound ``REGRA-ATIVACAO`` id — look its text up in
    :attr:`LibsElectricalModel.rules`; ``violation`` is the raw
    ``TRATAMENTO-VIOLACAO`` row, or ``None`` when the restriction carries
    none.
    """

    code: int
    lhs: str
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
    ``codigo_regra_ativacao``.
    """

    expressions: Mapping[str, ElectricalExpression]
    aliases: Mapping[str, ElectricalAlias]
    restrictions: Mapping[int, ElectricalRestriction]
    rules: Mapping[int, str]


def _df(source: Restricoes, attr: str) -> pd.DataFrame | None:
    """Read one accessor as a DataFrame, treating an absent card as empty."""
    frame = getattr(source, attr)(df=True)
    if frame is None or getattr(frame, "empty", True):
        return None
    return frame


def _opt_int(value: object) -> int | None:
    """A cell as ``int``, or ``None`` when blank (``NA``)."""
    return None if pd.isna(value) else int(value)  # type: ignore[call-overload]


def _opt_float(value: object) -> float | None:
    """A cell as ``float``, or ``None`` when blank (``NA``)."""
    return None if pd.isna(value) else float(value)  # type: ignore[arg-type]


def _opt_bound(value: object) -> float | None:
    """A two-sided limit cell as ``float``, or ``None`` when blank or at/beyond
    DECOMP's ``±1E+31`` unbounded sentinel."""
    if pd.isna(value):
        return None
    parsed = float(value)  # type: ignore[arg-type]
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


def _read_expressions(restricoes: Restricoes) -> dict[str, ElectricalExpression]:
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
    restricoes: Restricoes,
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
    restricoes: Restricoes,
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
    restricoes: Restricoes,
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
    restricoes: Restricoes,
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
    restricoes: Restricoes,
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


def _read_habilita(restricoes: Restricoes) -> dict[int, int]:
    """Read ``RESTRICAO-ELETRICA-HABILITA`` into ``{codigo_restricao:
    codigo_regra_ativacao}``."""
    frame = _df(restricoes, "restricao_eletrica_habilita")
    if frame is None:
        return {}
    return {
        int(row["codigo_restricao"]): int(row["codigo_regra_ativacao"])
        for _, row in frame.iterrows()
    }


def _read_rules(restricoes: Restricoes) -> dict[int, str]:
    """Read ``RESTRICAO-ELETRICA-REGRA-ATIVACAO`` into
    ``{codigo_regra_ativacao: regra_ativacao}`` (the raw boolean-rule text)."""
    frame = _df(restricoes, "restricao_eletrica_regra_ativacao")
    if frame is None:
        return {}
    return {
        int(row["codigo_regra_ativacao"]): str(row["regra_ativacao"])
        for _, row in frame.iterrows()
    }


def _read_violations(restricoes: Restricoes) -> dict[int, ViolationTreatment]:
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
    restricoes: Restricoes,
    calendar: Sequence[OperativeStage],
) -> dict[int, ElectricalRestriction]:
    """Join the FORMULA and INEQUACAO restriction cards to their per-(period,
    patamar) limit/override rows, horizonte, habilita, and violation rows by
    ``codigo_restricao``."""
    formula = _df(restricoes, "restricao_eletrica_formula")
    inequacao = _df(restricoes, "restricao_eletrica_inequacao")
    if formula is None and inequacao is None:
        return {}

    horizonte = _read_horizonte(restricoes, calendar)
    limits = _read_formula_limits(restricoes, calendar)
    overrides = _read_inequacao_overrides(restricoes, calendar)
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

    return restrictions


def read_libs_electrical(
    restricoes: Restricoes,
    calendar: Sequence[OperativeStage],
) -> LibsElectricalModel | None:
    """Read every period-keyed long-form electrical card into a structured model.

    Joins the ``RESTRICAO-ELETRICA-FORMULA``/``-INEQUACAO`` restrictions to
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
    )


#: idecomp's ``df=True`` accessor expands a per-patamar list column into
#: ``carga_ande_1``, ``carga_ande_2``, ... (one per patamar slot, mirroring
#: ``DP``'s ``carga_1``/``IA``'s ``limite_de_para_1`` convention); a row's own
#: slots beyond its declared patamar count read back as ``NaN``. This pattern
#: discovers however many patamar-slot columns the installed idecomp emits,
#: rather than hardcoding a slot count.
_CARGA_ANDE_COLUMN = re.compile(r"^carga_ande_(\d+)$")


def read_carga_ande(
    dadger: Dadger,
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
    ri = dadger.ri(df=True)
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


#: Built-in bare (argument-less) input-data tokens (spec §0/§3). A bare name not
#: declared by ``EXPRESSAO-ELETRICA`` or ``ALIAS-ELETRICO``, and not in this set,
#: is an unrecognized identifier.
_BUILTIN_BARE_TOKENS = frozenset({"demanda_sin", "val_demanda_sin", "carga_ande"})

#: Reserved token for a bare numeric constant; its value lives in
#: :attr:`ParsedTerm.coefficient`.
_CONST_TOKEN = "__const__"

#: Reserved token for an unresolved ``ALIAS-ELETRICO`` reference; the alias's
#: own name lives in :attr:`ParsedTerm.alias_name`.
_ALIAS_TOKEN = "alias"

#: Reserved token for a structural ``se(cond, X, Y)`` conditional.
_SE_TOKEN = "se"

_NUMBER_RE = re.compile(r"\d+\.?\d*")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class ParsedTerm:
    """One atomic, sign-and-coefficient-distributed term of a parsed linear
    expression (see :func:`parse_linear_expression`).

    ``token`` identifies the term's kind:

    - a function-call name (``ger_usih``, ``ener_interc``, ``disp_usih``, …)
      with its integer arguments in ``args``;
    - the reserved :data:`_CONST_TOKEN` (``"__const__"``) for a bare numeric
      constant — its value is in ``coefficient``, ``args`` is empty;
    - the reserved :data:`_ALIAS_TOKEN` (``"alias"``) for an unresolved
      ``ALIAS-ELETRICO`` reference — ``args`` is empty and the alias's own
      name is in ``alias_name``; its numeric value is resolved
      per-(stage,block) later, never here;
    - a built-in bare token (``demanda_sin`` / ``val_demanda_sin`` /
      ``carga_ande``), ``args`` empty;
    - the reserved :data:`_SE_TOKEN` (``"se"``) for a ``se(cond, X, Y)``
      conditional, kept structural: ``condition``, ``branch_true``, and
      ``branch_false`` carry the raw, unparsed sub-expression text (evaluated
      later, never here; ``cond`` is not evaluated by this parser).

    A named-expression (``EXPRESSAO-ELETRICA``) or ``re(R)`` reference never
    survives to a ``ParsedTerm`` — the parser inlines it into its own terms,
    multiplying the enclosing coefficient through.
    """

    coefficient: float
    token: str
    args: tuple[int, ...] = ()
    alias_name: str | None = None
    condition: str | None = None
    branch_true: str | None = None
    branch_false: str | None = None


@dataclass(frozen=True)
class AssembledBound:
    """The result of folding one restriction's bucket-B/-C terms into a
    per-(stage,block) numeric bound (:func:`assemble_bound`), sign-canonical
    for cobre.

    ``terms`` is the surviving bucket-A (decision) expression, already in
    cobre's single-LHS form: ``lower <= Σ terms <= upper``. Each of
    :func:`assemble_bound`'s three folds gets there differently — a FORMULA
    restriction's LHS bucket-A terms verbatim, a plain INEQUACAO's LHS
    bucket-A terms plus its RHS bucket-A terms moved over and sign-flipped,
    a ``disp_usih`` reserve's ``ger_usih`` terms sign-flipped to match the
    reserve->gen-cap rewrite's operator flip — so a caller reading ``terms``
    never has to know which shape produced it, and never reproduces the A1
    sign hazard of reading ``restriction.lhs``/``.rhs`` verbatim instead.
    """

    terms: tuple[ParsedTerm, ...]
    lower: float | None
    upper: float | None


def _negate(term: ParsedTerm) -> ParsedTerm:
    """*term* with its coefficient sign flipped, every other field verbatim."""
    return replace(term, coefficient=-term.coefficient)


class UnrecognizedElectricalToken(ValueError):
    """A well-formed but undeclared bare identifier (spec §3): neither a
    declared ``EXPRESSAO-ELETRICA`` named expression, a declared
    ``ALIAS-ELETRICO`` data alias, nor a member of
    :data:`_BUILTIN_BARE_TOKENS`.

    Raised only from :meth:`_ExpressionParser._parse_bare_name`'s
    undeclared-identifier branch — never from a malformed-DSL
    raise (unbalanced parentheses, a non-numeric coefficient/argument,
    unexpected trailing content), which stays a plain :class:`ValueError`
    and remains fail-loud. A :class:`ValueError` subclass so an existing
    ``pytest.raises(ValueError)`` contract still holds, but distinct so the
    emitter's per-restriction skip-not-partial boundary
    (``libs_electrical_emit.emit_libs_electrical_generics``) can contain
    exactly this "well-formed but undeclared identifier" case to the one
    restriction that references it, without catching (and so silently
    masking) a genuine malformed-DSL bug as a routine drop.
    """


class _ExpressionParser:
    """Character-position recursive-descent parser for one DSL expression string.

    Threads a running numeric ``scale`` — the product of every sign and scalar
    coefficient enclosing the current position — down to each atomic term, so a
    coefficient distributed over a parenthesized sum (``0.05*(a + b)``) lands on
    every term inside it, and a named-expression/``re(R)`` reference inlines its
    own terms multiplied by the scale in effect where the reference occurred.
    """

    def __init__(
        self, text: str, model: LibsElectricalModel, seen: frozenset[str]
    ) -> None:
        self._text = text
        self._model = model
        self._seen = seen
        self._pos = 0

    def parse(self) -> list[ParsedTerm]:
        terms = self._parse_expr(1.0)
        self._skip_ws()
        if self._pos != len(self._text):
            raise ValueError(
                f"unexpected trailing content {self._text[self._pos :]!r} in "
                f"electrical expression {self._text!r}"
            )
        return terms

    def _skip_ws(self) -> None:
        while self._pos < len(self._text) and self._text[self._pos].isspace():
            self._pos += 1

    def _peek(self) -> str:
        self._skip_ws()
        return self._text[self._pos] if self._pos < len(self._text) else ""

    def _consume(self, char: str, context: str) -> None:
        self._skip_ws()
        if self._pos >= len(self._text) or self._text[self._pos] != char:
            raise ValueError(
                f"expected {char!r} {context} in electrical expression {self._text!r}"
            )
        self._pos += 1

    def _parse_expr(self, scale: float) -> list[ParsedTerm]:
        """``['+'|'-'] term (('+'|'-') term)*`` — a signed sum of terms."""
        terms: list[ParsedTerm] = []
        sign = 1.0
        if self._peek() in ("+", "-"):
            sign = -1.0 if self._text[self._pos] == "-" else 1.0
            self._pos += 1
        terms.extend(self._parse_term(scale * sign))
        while self._peek() in ("+", "-"):
            sign = -1.0 if self._text[self._pos] == "-" else 1.0
            self._pos += 1
            terms.extend(self._parse_term(scale * sign))
        return terms

    def _parse_term(self, scale: float) -> list[ParsedTerm]:
        """``[NUMBER '*'] factor`` — a bare number alone is a constant term."""
        self._skip_ws()
        number = _NUMBER_RE.match(self._text, self._pos)
        if number is None:
            return self._parse_factor(scale)
        probe = number.end()
        while probe < len(self._text) and self._text[probe].isspace():
            probe += 1
        if probe < len(self._text) and self._text[probe] == "*":
            self._pos = probe + 1
            return self._parse_factor(scale * float(number.group()))
        self._pos = number.end()
        return [
            ParsedTerm(coefficient=scale * float(number.group()), token=_CONST_TOKEN)
        ]

    def _parse_factor(self, scale: float) -> list[ParsedTerm]:
        """``'(' expr ')' | IDENT '(' args ')' | IDENT`` — a group or an atom."""
        self._skip_ws()
        if self._pos >= len(self._text):
            raise ValueError(
                "expected a term but reached the end of electrical expression "
                f"{self._text!r}"
            )
        if self._text[self._pos] == "(":
            self._pos += 1
            terms = self._parse_expr(scale)
            self._consume(")", "to close '('")
            return terms
        ident = _IDENT_RE.match(self._text, self._pos)
        if ident is None:
            raise ValueError(
                f"malformed token at {self._text[self._pos :]!r} in electrical "
                f"expression {self._text!r}"
            )
        name = ident.group()
        self._pos = ident.end()
        if self._peek() == "(":
            return self._parse_function_call(name, scale)
        return self._parse_bare_name(name, scale)

    def _parse_function_call(self, name: str, scale: float) -> list[ParsedTerm]:
        if name == _SE_TOKEN:
            return self._parse_se(scale)
        if name == "re":
            return self._parse_re(scale)
        args = self._parse_int_args(name)
        return [ParsedTerm(coefficient=scale, token=name, args=args)]

    def _parse_int_args(self, fn_name: str) -> tuple[int, ...]:
        self._consume("(", f"after {fn_name!r}")
        args: list[int] = []
        if self._peek() == ")":
            self._pos += 1
            return tuple(args)
        while True:
            self._skip_ws()
            number = _NUMBER_RE.match(self._text, self._pos)
            if number is None or "." in number.group():
                raise ValueError(
                    f"expected an integer argument for {fn_name!r}(...) at "
                    f"{self._text[self._pos :]!r} in electrical expression "
                    f"{self._text!r}"
                )
            args.append(int(number.group()))
            self._pos = number.end()
            if self._peek() == ",":
                self._pos += 1
                continue
            break
        self._consume(")", f"to close {fn_name!r}(...)")
        return tuple(args)

    def _parse_re(self, scale: float) -> list[ParsedTerm]:
        """``re(R)`` — a reference to restriction ``R``'s formula, inlined."""
        args = self._parse_int_args("re")
        if len(args) != 1:
            raise ValueError(
                f"re(...) expects exactly one restriction id, got {args!r} in "
                f"electrical expression {self._text!r}"
            )
        code = args[0]
        restriction = self._model.restrictions.get(code)
        if restriction is None:
            raise ValueError(
                f"re({code}) references unknown restriction {code} in "
                f"electrical expression {self._text!r}"
            )
        return self._expand_reference(f"re({code})", restriction.lhs, scale)

    def _parse_se(self, scale: float) -> list[ParsedTerm]:
        """``se(cond, X, Y)`` — one structural term; ``cond`` is never evaluated
        here, so its raw text (and both branches') is captured by a
        depth-aware scan for the top-level commas rather than by tokenizing it."""
        self._consume("(", "after 'se'")
        args_start = self._pos
        comma_positions: list[int] = []
        depth = 0
        pos = self._pos
        while True:
            if pos >= len(self._text):
                raise ValueError(
                    "unbalanced parentheses in se(...) in electrical expression "
                    f"{self._text!r}"
                )
            char = self._text[pos]
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    break
                depth -= 1
            elif char == "," and depth == 0:
                comma_positions.append(pos)
            pos += 1
        close_pos = pos
        if len(comma_positions) != 2:
            raise ValueError(
                "se(...) expects exactly 3 arguments (condition, true-branch, "
                f"false-branch) in electrical expression {self._text!r}"
            )
        first_comma, second_comma = comma_positions
        condition = self._text[args_start:first_comma].strip()
        branch_true = self._text[first_comma + 1 : second_comma].strip()
        branch_false = self._text[second_comma + 1 : close_pos].strip()
        self._pos = close_pos + 1
        return [
            ParsedTerm(
                coefficient=scale,
                token=_SE_TOKEN,
                condition=condition,
                branch_true=branch_true,
                branch_false=branch_false,
            )
        ]

    def _parse_bare_name(self, name: str, scale: float) -> list[ParsedTerm]:
        """Disambiguate a bare name per spec §3: a named expression, else a data
        alias, else a built-in bare token — else an unrecognized identifier."""
        if name in self._model.expressions:
            return self._expand_reference(
                name, self._model.expressions[name].formula, scale
            )
        if name in self._model.aliases:
            return [ParsedTerm(coefficient=scale, token=_ALIAS_TOKEN, alias_name=name)]
        if name in _BUILTIN_BARE_TOKENS:
            return [ParsedTerm(coefficient=scale, token=name)]
        raise UnrecognizedElectricalToken(
            f"unrecognized identifier {name!r} in electrical expression "
            f"{self._text!r} (not a declared named expression, data alias, or "
            "built-in bare token)"
        )

    def _expand_reference(
        self, key: str, formula: str, scale: float
    ) -> list[ParsedTerm]:
        """Recursively parse *formula* (a named expression's or a restriction's
        LHS), multiplying every resulting term's coefficient by *scale*, and
        guard the recursion against a cycle through *key*."""
        if key in self._seen:
            cycle = " -> ".join(sorted({*self._seen, key}))
            raise ValueError(f"cyclic electrical expression reference: {cycle}")
        sub_terms = parse_linear_expression(
            formula, self._model, _seen=self._seen | {key}
        )
        return [
            ParsedTerm(
                coefficient=scale * term.coefficient,
                token=term.token,
                args=term.args,
                alias_name=term.alias_name,
                condition=term.condition,
                branch_true=term.branch_true,
                branch_false=term.branch_false,
            )
            for term in sub_terms
        ]


def parse_linear_expression(
    text: str,
    model: LibsElectricalModel,
    *,
    _seen: frozenset[str] = frozenset(),
) -> list[ParsedTerm]:
    """Parse one electrical linear-expression DSL string into flat terms.

    Distributes every sign and scalar coefficient onto each atomic term
    (including across a parenthesized sum) and recursively inlines
    ``EXPRESSAO-ELETRICA`` named-expression references and ``re(R)``
    restriction references, multiplying through the enclosing coefficient —
    ``_seen`` (the names/``re(R)`` keys expanded on the current recursion path)
    guards that inlining against a cycle, raising ``ValueError`` naming it. An
    ``ALIAS-ELETRICO`` reference is kept as a single unresolved
    :data:`_ALIAS_TOKEN` term (its value is resolved per-(stage,block) later,
    never here); a ``se(cond, X, Y)`` conditional is kept as a single
    structural term with its condition and both branches unevaluated.

    This is the source model's linear-expression grammar, not cobre's: the
    parser never normalizes the relation (no RHS→LHS move, no same-token
    merge) — it preserves the author's side and term multiplicity verbatim;
    cobre owns relational normalization.

    Raises
    ------
    UnrecognizedElectricalToken
        On a bare identifier that is well-formed but undeclared — neither a
        named expression, a data alias, nor a built-in bare token (a
        :class:`ValueError` subclass; the per-restriction
        skip-not-partial signal — see its own docstring).
    ValueError
        On a cyclic named-expression/``re(R)`` reference, or a malformed
        token (unbalanced parentheses, a non-numeric coefficient/argument,
        or unexpected trailing content) — genuinely malformed DSL, which
        stays fail-loud and is never caught as a routine drop.
    """
    return _ExpressionParser(text, model, _seen).parse()


class Bucket(Enum):
    """Which of the source model's spec-§0 A/B/C trichotomy a parsed term
    belongs to.

    - **A** — a cobre decision term: stays a structural LP term.
    - **B** — an input-data term: the bridge evaluates it to a number and
      folds it into the bound.
    - **C** — ``disp_usih``: cobre has no available-power ``VariableRef``, so
      the bridge resolves it to a per-stage constant that rewrites the
      constraint form instead of emitting it as a structural LP term.
    """

    A = "A"
    B = "B"
    C = "C"


#: Bucket A (spec §0) — cobre decision tokens that stay structural LP terms.
#: ``re(X)`` is not listed here: the parser already inlines a restriction
#: reference into its own terms (multiplying through the enclosing
#: coefficient), so no :class:`ParsedTerm` ever carries a bare ``"re"`` token.
_BUCKET_A_TOKENS = frozenset(
    {"ger_usih", "ger_usit", "ger_pee", "ger_conjh", "ener_interc", "ener_comerc"}
)

#: Bucket B (spec §0) — input-data tokens the bridge evaluates to a number and
#: folds into the bound: the built-in demand/load tokens, plus the parser's
#: three reserved structural tokens (a bare numeric constant, an unresolved
#: ``ALIAS-ELETRICO`` reference, and a ``se(cond, X, Y)`` conditional).
_BUCKET_B_TOKENS = frozenset(
    {
        "demanda",
        "val_demanda",
        "demanda_sin",
        "val_demanda_sin",
        "carga_ande",
        _CONST_TOKEN,
        _ALIAS_TOKEN,
        _SE_TOKEN,
    }
)

#: Bucket C (spec §0) — ``disp_usih`` is the sole member (see :class:`Bucket`).
_BUCKET_C_TOKENS = frozenset({"disp_usih"})

#: The closed token -> :class:`Bucket` map (spec §0), the sole source of truth
#: for :func:`classify_term`. Mirrors
#: ``constraint_registers._BOUNDS_AXIS``'s module-level membership-map shape.
_TOKEN_BUCKET: dict[str, Bucket] = {
    **dict.fromkeys(_BUCKET_A_TOKENS, Bucket.A),
    **dict.fromkeys(_BUCKET_B_TOKENS, Bucket.B),
    **dict.fromkeys(_BUCKET_C_TOKENS, Bucket.C),
}


def classify_term(term: ParsedTerm) -> Bucket:
    """Classify one parsed term into its spec-§0 A/B/C bucket.

    Decides purely from ``term.token`` against the closed set in
    :data:`_TOKEN_BUCKET` — never a model lookup, since the parser already
    resolved every name to a concrete token (an alias to :data:`_ALIAS_TOKEN`,
    a bare numeric constant to :data:`_CONST_TOKEN`, a conditional to
    :data:`_SE_TOKEN`, a named-expression/``re(R)`` reference inlined away).

    Raises
    ------
    ValueError
        When ``term.token`` is not a recognized token — fail loud rather than
        silently drop or mis-bucket a term, which would corrupt either the LP
        structure (bucket A/C) or the bound (bucket B).
    """
    bucket = _TOKEN_BUCKET.get(term.token)
    if bucket is None:
        recognized = ", ".join(sorted(_TOKEN_BUCKET))
        raise ValueError(
            f"unrecognized electrical-expression token {term.token!r}; "
            f"recognized tokens are: {recognized}"
        )
    return bucket


def classify_terms(terms: Sequence[ParsedTerm]) -> dict[Bucket, list[ParsedTerm]]:
    """Partition a term sequence into its three spec-§0 buckets.

    Every :class:`Bucket` key is present (an empty list when no term
    classified into it), and each bucket's list preserves the input order —
    callers get a stable, total partition.

    Raises
    ------
    ValueError
        Propagated from :func:`classify_term` on the first unrecognized token.
    """
    partition: dict[Bucket, list[ParsedTerm]] = {bucket: [] for bucket in Bucket}
    for term in terms:
        partition[classify_term(term)].append(term)
    return partition


#: The activation-rule grammar's six comparison operators (spec §4a),
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
    enforces this — an activation rule gates on input data, never a cobre
    decision term). ``op`` is one of the six recognized comparison operators
    (spec §4a), verbatim.
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


#: Resolves one bucket-B operand term to its numeric value for one
#: (stage, block) cell — a built-in bare token (``demanda_sin``), a
#: parameterized one (``demanda(3)``), an unresolved ``ALIAS-ELETRICO``
#: reference (``term.alias_name``), or a ``se(cond, X, Y)`` conditional.
#:
#: A bare numeric constant (``term.token == "__const__"``) is the one
#: exception: its value already lives in the term's own ``coefficient``
#: (see :class:`ParsedTerm`), so a resolver **must** return ``1.0`` for it —
#: :func:`evaluate_rule`'s ``coefficient * context(term)`` product would
#: otherwise double-apply the constant.
#:
#: This is the source model's per-(stage, block) demand/alias resolver
#: interface, kept abstract so the engine stays a pure function; the real
#: resolver is :func:`build_data_context`.
type DataContext = Callable[[ParsedTerm], float]


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
    ValueError
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
                raise ValueError(f"unbalanced parentheses in activation rule {text!r}")
        elif char == sep and depth == 0:
            parts.append(text[start:pos])
            start = pos + 1
    if depth != 0:
        raise ValueError(f"unbalanced parentheses in activation rule {text!r}")
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
    ValueError
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
                raise ValueError(
                    f"unrecognized comparison operator {two_char!r} in "
                    f"activation-rule conjunct {conjunct!r}"
                )
            if char not in _COMPARISON_ONE_CHAR_OPS:
                raise ValueError(
                    f"unrecognized comparison operator {char!r} in "
                    f"activation-rule conjunct {conjunct!r}"
                )
            return conjunct[:pos], char, conjunct[pos + 1 :]
        pos += 1
    raise ValueError(
        f"no comparison operator found in activation-rule conjunct {conjunct!r} "
        "(recognized operators: <, <=, >, >=, ==, !=)"
    )


def _require_bucket_b(terms: Sequence[ParsedTerm], conjunct: str) -> None:
    """Raise ``ValueError`` naming the offending token when any *terms* is
    not bucket B — an activation-rule operand is always input data (spec
    §4a), never a cobre decision term (bucket A) or the available-power
    bucket-C term."""
    for term in terms:
        if classify_term(term) is not Bucket.B:
            raise ValueError(
                f"activation-rule operand {term.token!r} in conjunct "
                f"{conjunct!r} is not input data (bucket B); an activation "
                "rule must gate on input data, never a decision term"
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
    cobre decision term.

    This is the closed six-operator + ``&`` grammar verified against every
    rule in the source model's decks (spec §4a) — it has no ``|``
    (disjunction) and no ``se`` literal at the rule-text level, so neither is
    supported here; a deck that needs either is out of scope, not a silent
    extension.

    Raises
    ------
    ValueError
        Propagated from :func:`_split_top_level` on unbalanced parentheses,
        from :func:`_split_comparison` on a missing/unrecognized operator, or
        raised here when an operand term does not classify as bucket B
        (naming the offending token).
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


def _evaluate_side(terms: Sequence[ParsedTerm], context: DataContext) -> float:
    """Sum one :class:`Comparison` side's ``coefficient * context(term)``
    over its operand terms.

    A bare numeric constant (``term.token == _CONST_TOKEN``) is read
    straight from ``term.coefficient`` rather than through *context*: a
    conforming context already returns ``1.0`` for it (the DataContext
    convention), so this changes nothing for one, but it closes
    the handshake so a non-conforming context can never silently double- or
    zero-apply a constant.
    """
    return sum(
        term.coefficient
        if term.token == _CONST_TOKEN
        else term.coefficient * context(term)
        for term in terms
    )


def evaluate_rule(rule: ActivationRule, context: DataContext) -> bool:
    """Evaluate one parsed activation rule against a per-(stage,block)
    :data:`DataContext`.

    Returns the AND of every :class:`Comparison`: each side sums to
    ``Σ coefficient * context(term)`` over its operand terms
    (:func:`_evaluate_side`), and the six comparison operators map onto the
    matching predicate (:data:`_COMPARISON_FUNCS`). An empty
    ``rule.comparisons`` — :func:`is_always_active` — evaluates to ``True``,
    the vacuous AND.

    Pure and deterministic given *context*: this engine never resolves a
    bucket-B token's value itself, so the same rule can be evaluated against
    any (stage, block) cell just by supplying a different *context*.
    """
    return all(
        _COMPARISON_FUNCS[comparison.op](
            _evaluate_side(comparison.left, context),
            _evaluate_side(comparison.right, context),
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
    at a (stage, block) cell (spec §4c requirement 4) — a ``carga_ande``
    reference with no ``RI`` register, a ``demanda``/``val_demanda``
    reference to an unknown submarket, or an ``alias`` reference with no
    value at or before the cell.

    Deliberately not a :class:`ValueError`: :func:`evaluate_se` raises a
    genuine ``ValueError`` for an unrecognized/decision-bearing ``se(...)``
    shape, and that failure must propagate (fail loud), never be caught and
    turned into a drop. :func:`assemble_bound` is the only place that
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
        bus_id` raises ``KeyError``) — spec §4c requirement 4's "unknown
        submarket" case.
    """
    code = term.args[0]
    try:
        return id_map.bus_id(code)
    except KeyError as err:
        raise _UnresolvableBucketBTerm(
            f"{term.token}({code}) references unknown submarket {code}"
        ) from err


def build_data_context(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    model: LibsElectricalModel,
) -> Callable[[int, int], DataContext]:
    """Build the per-(stage,block) :data:`DataContext` factory that drives
    :func:`assemble_bound` and the activation gate.

    :data:`DataContext` (a ``Callable[[ParsedTerm], float]``) is
    defined for exactly *one* ``(stage_index, block_index)`` cell — see its
    own docstring — and *build_data_context*'s own signature carries no such
    cell, so it returns a **factory**: call the returned closure with a cell
    to get back the :data:`DataContext` scoped to it, ready to hand to
    :func:`assemble_bound`/:func:`evaluate_rule`. The submarket-demand map
    (:func:`~cobre_bridge.decomp.load._per_stage_block_loads`) and the
    ``carga_ande`` series (:func:`read_carga_ande`) are each read once and
    captured — along with *model* — in the returned closure, rather than
    re-read per cell.

    Resolves every bucket-B token (spec §4c):

    - ``demanda(X)``/``val_demanda(X)`` — submarket ``X``'s (the deck's
      ``codigo_submercado``) block load, ``0.0`` when ``DP`` declares no row
      for that (bus, stage);
    - ``demanda_sin``/``val_demanda_sin`` — the block load summed over every
      declared submarket;
    - ``carga_ande`` — the series from :func:`read_carga_ande`;
    - an ``alias`` token — the model alias's value
      (:func:`_resolve_alias_value`: stage-inherited, ``NA``-patamar applies
      to all blocks);
    - the numeric-constant token — ``1.0``, never the constant's own value:
      the value already lives in
      ``term.coefficient``, which :func:`_evaluate_side` applies directly;
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
    """
    calendar = case.calendar
    demand = _per_stage_block_loads(case.dadger, id_map, calendar)
    carga_ande_series = read_carga_ande(case.dadger, calendar)
    n_submarkets = len(id_map.bus_codes)

    def context_for_cell(stage_index: int, block_index: int) -> DataContext:
        def resolve(term: ParsedTerm) -> float:
            if term.token == _CONST_TOKEN:
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
                        "carga_ande referenced but the deck declares no RI "
                        f"register (or no series for stage {stage_index})"
                    )
                return values[block_index]
            if term.token == _ALIAS_TOKEN:
                assert term.alias_name is not None  # parser invariant
                alias = model.aliases[term.alias_name]
                patamar = block_index + 1
                value = _resolve_alias_value(alias, stage_index, patamar)
                if value is None:
                    raise _UnresolvableBucketBTerm(
                        f"alias {term.alias_name!r} has no value at stage "
                        f"{stage_index} (patamar {patamar}) or any earlier "
                        "stage"
                    )
                return value
            if term.token == _SE_TOKEN:
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

    The selected branch is then parsed (:func:`parse_linear_expression`) and
    classified (:func:`classify_terms`). A selected branch is only safe to
    fold to a single float when it is
    entirely bucket B (input data): folding a branch that carries a
    bucket-A (cobre decision) or bucket-C (``disp_usih``) term would fold
    that term into a plain number and silently drop it from the LP. ``se``
    does not appear on the target deck (spec §2b: gating is via
    ``REGRA-ATIVACAO``, not inline ``se()``), so this guard is defensive
    hardening, not a gap in coverage — keeping a decision-bearing branch's
    bucket-A/-C terms structural (folding only its bucket-B terms, the way
    :func:`assemble_bound` does for a whole restriction) is out of scope
    here; this fails loud instead.

    Raises
    ------
    ValueError
        When *term* is not a ``se`` term or is missing its condition/branch
        text (a caller contract violation — the parser always populates all
        three for a ``se`` token), or when the selected branch carries any
        non-bucket-B term (naming it and the cell).
    """
    if term.token != _SE_TOKEN:
        raise ValueError(f"evaluate_se: expected a 'se' term, got {term.token!r}")
    if term.condition is None or term.branch_true is None or term.branch_false is None:
        raise ValueError(
            f"evaluate_se: 'se' term is missing condition/branches: {term!r}"
        )

    rule = parse_activation_rule(term.condition, model)
    selected_text = term.branch_true if evaluate_rule(rule, ctx) else term.branch_false
    branch_terms = parse_linear_expression(selected_text, model)
    partition = classify_terms(branch_terms)
    decision_terms = partition[Bucket.A] + partition[Bucket.C]
    if decision_terms:
        offender = decision_terms[0]
        raise ValueError(
            f"evaluate_se: at stage {stage_index}, block {block_index}, "
            f"se(...) selected branch {selected_text!r} carries a decision "
            f"term ({offender.token!r}, bucket {classify_term(offender).name}); "
            "folding a decision-bearing branch to a number would silently "
            "drop it from the LP — structural handling of a decision-bearing "
            "se branch is out of scope here (fail loud instead)"
        )
    return _evaluate_side(partition[Bucket.B], ctx)


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
    cobre's single-LHS ``lower <= expr <= upper`` shape, so no sign flip is
    ever needed here (spec §1).

    ``restriction.limits`` is looked up at an exact ``(stage_index,
    patamar)`` match, falling back to the ``NA`` (all-blocks,
    ``patamar=None``) entry for that stage. The limit's two endpoints are
    already numeric (DECOMP's ``±1E+31`` sentinel already normalized to
    ``None`` by the reader — see :func:`_opt_bound`); moving the LHS's
    bucket-B sum onto them keeps its bucket-A/-C terms untouched: ``lower <=
    lhs_A + lhs_B <= upper`` becomes ``lower - lhs_B <= lhs_A <= upper -
    lhs_B``.

    *lhs_partition* is *restriction.lhs* already parsed and classified
    (:func:`assemble_bound` computes it once and passes it to every fold —
    this helper never re-parses ``restriction.lhs`` itself).

    Raises
    ------
    ValueError
        When *restriction* declares no limit at all for this cell (a
        caller-contract violation — *stage_index* should always fall inside
        the restriction's active horizon).
    """
    limit = restriction.limits.get((stage_index, patamar))
    if limit is None:
        limit = restriction.limits.get((stage_index, None))
    if limit is None:
        raise ValueError(
            f"restriction {restriction.code}: no FORMULA limit declared for "
            f"stage {stage_index}, patamar {patamar}"
        )
    lower, upper = limit
    lhs_b_sum = _evaluate_side(lhs_partition[Bucket.B], ctx)
    return AssembledBound(
        terms=tuple(lhs_partition[Bucket.A]),
        lower=None if lower is None else lower - lhs_b_sum,
        upper=None if upper is None else upper - lhs_b_sum,
    )


def _effective_inequacao_sides(
    restriction: ElectricalRestriction, stage_index: int, patamar: int
) -> tuple[str, str, str]:
    """The ``(lhs, operator, rhs)`` DSL strings in effect for one INEQUACAO
    restriction's ``(stage_index, patamar)`` cell (spec §3's precedence:
    ``période-patamar > data-patamar > constante``).

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
    assert restriction.operator is not None  # INEQUACAO invariant
    assert restriction.rhs is not None  # INEQUACAO invariant
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
    if chosen is None:
        return restriction.lhs, restriction.operator, restriction.rhs
    return chosen.lhs, chosen.operator, chosen.rhs


def _fold_inequacao_bound(
    restriction: ElectricalRestriction,
    ctx: DataContext,
    lhs_partition: Mapping[Bucket, list[ParsedTerm]],
    rhs_partition: Mapping[Bucket, list[ParsedTerm]],
) -> AssembledBound:
    """Fold an INEQUACAO restriction's bucket-B terms — both sides,
    sign-aware — into its operator's F3 interval, and move any RHS bucket-A
    term onto the LHS, sign-flipped, so the surviving expression is cobre's
    single-LHS form (spec §1 — the A1 sign hazard's plain-INEQUACAO half:
    ``lhs op rhs`` with a bucket-A term on the RHS must not be read as if it
    were already on the LHS).

    Sums each side's bucket-B contribution (:func:`_evaluate_side`) and
    combines them with bucket-A normalized onto the LHS:
    ``bound = Σ_{B∩RHS} − Σ_{B∩LHS}``. :func:`~cobre_bridge.
    generic_constraint_format.sense_to_interval` then maps *restriction*'s
    operator onto the F3 ``(lower, upper)`` shape (``>=`` lower-only, ``<=``
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
    lhs_b_sum = _evaluate_side(lhs_partition[Bucket.B], ctx)
    rhs_b_sum = _evaluate_side(rhs_partition[Bucket.B], ctx)
    lower, upper = sense_to_interval(restriction.operator, rhs_b_sum - lhs_b_sum)
    terms = tuple(lhs_partition[Bucket.A]) + tuple(
        _negate(term) for term in rhs_partition[Bucket.A]
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
    cobre — in the returned :class:`AssembledBound` (spec §4c; the bucket-C
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
    in effect at this cell (spec §3's precedence: exact-patamar >
    NA-patamar > the restriction's own constant sides) before classification
    ever runs, so a cell with no matching override classifies the constant
    sides verbatim and one with a matching override classifies that
    override's sides instead. With no bucket-C term, the effective sides
    fold sign-aware into the effective operator's F3 interval, moving any
    RHS bucket-A term onto the LHS negated (:func:`_fold_inequacao_bound`);
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
    ValueError
        When a FORMULA restriction carries no limit at all for this cell,
        when a FORMULA restriction carries a bucket-C term (out of the
        documented pattern's scope), when an INEQUACAO restriction carries a
        bucket-C term but *a_h* is ``None``, when a restriction uses
        ``disp_usih`` outside the documented reserve sign structure
        (:func:`_fold_reserve_disp_usih`'s CRITICAL-pitfall guard), or
        propagated from :func:`~cobre_bridge.core.generic_constraint_format.
        sense_to_interval` on an unrecognized INEQUACAO operator — none of
        these is the resolver-unresolvable case this function otherwise
        catches.
    """
    patamar = block_index + 1
    try:
        if restriction.is_formula:
            lhs_partition = classify_terms(
                parse_linear_expression(restriction.lhs, model)
            )
            if lhs_partition[Bucket.C]:
                raise ValueError(
                    f"restriction {restriction.code}: a FORMULA restriction "
                    "carries a disp_usih term; the reserve->gen-cap rewrite "
                    "only covers the documented INEQUACAO "
                    "reserve pattern"
                )
            return _fold_formula_bound(
                restriction, stage_index, patamar, ctx, lhs_partition
            )

        lhs, op, rhs = _effective_inequacao_sides(restriction, stage_index, patamar)
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
            Diagnostic(
                code="decomp-electrical-bucket-b-unresolved",
                severity=Severity.WARNING,
                category="Special constraints",
                title="Electrical restriction input-data term could not be resolved",
                summary=(
                    f"Electrical restriction {restriction.code} at stage "
                    f"{stage_index}, block {block_index} could not resolve "
                    f"an input-data term ({err}); the constraint is skipped "
                    "rather than emitted with a partial bound."
                ),
                remediation=(
                    "Check the source model's data feeding this restriction "
                    f"(restriction {restriction.code}, stage {stage_index}, "
                    f"block {block_index}); the constraint is skipped."
                ),
            )
        )
        return None
    except _UnresolvableDispUsih as err:
        emit(
            Diagnostic(
                code="decomp-electrical-disp-usih-unresolved",
                severity=Severity.WARNING,
                category="Special constraints",
                title="disp_usih available power could not be resolved",
                summary=(
                    f"Electrical restriction {restriction.code} at stage "
                    f"{stage_index}, block {block_index} could not resolve a "
                    f"disp_usih available-power term ({err}); the "
                    "constraint is skipped rather than emitted with a "
                    "partial bound."
                ),
                remediation=(
                    "Check the source model's cadastro/maintenance data for "
                    f"the plant named in restriction {restriction.code}'s "
                    "disp_usih term; the constraint is skipped."
                ),
            )
        )
        return None


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
    activation rule evaluates ``True``.

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
    ``Severity.INFO`` :class:`~cobre_bridge.core.diagnostics.Diagnostic` naming
    *restriction* (mirroring ``constraint_registers``'s dropped-restriction
    skip logging, as a diagnostic here rather than a bare log record).
    Raising it here keeps *active_cells* self-contained and directly
    testable (via :func:`~cobre_bridge.core.diagnostics.collect`) — the caller
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
    ValueError
        When *restriction*'s ``habilita`` names a ``regra_id`` absent from
        ``model.rules`` — a dangling activation reference must fail loud,
        naming both the restriction and the missing rule id.
    """
    if restriction.habilita is None:
        rule = ActivationRule(comparisons=())
    else:
        rule_text = model.rules.get(restriction.habilita)
        if rule_text is None:
            raise ValueError(
                f"restriction {restriction.code}: HABILITA names activation "
                f"rule {restriction.habilita}, which is absent from the "
                "model's rules"
            )
        rule = parse_activation_rule(rule_text, model)

    always_active = is_always_active(rule)
    stage_start, stage_end = _restriction_horizon(restriction, calendar)

    cells: set[tuple[int, int]] = set()
    for stage_index in range(stage_start, stage_end + 1):
        n_blocks = len(calendar[stage_index].block_hours)
        for block_index in range(n_blocks):
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
                title="Electrical restriction inactive across its horizon",
                summary=(
                    f"Electrical restriction {restriction.code} evaluated "
                    "false in every in-horizon (stage, block) cell within "
                    f"stages {stage_start}-{stage_end}; no bound is emitted "
                    "for it."
                ),
                remediation=(
                    "Check the source model's activation rule and demand "
                    f"data feeding restriction {restriction.code}; the "
                    "restriction carries no active cell and is dropped."
                ),
            )
        )
    return cells


# ---------------------------------------------------------------------------
# bucket-C ``disp_usih`` reserve -> gen-cap reformulation
# ---------------------------------------------------------------------------


class AvailablePowerSource(Enum):
    """Which tier resolved one plant/stage's ``disp_usih`` available power
    (``A_h``), per Requirement 1's PRIMARY-vs-FALLBACK OQ-2 decision (this
    module's header): the maintenance-aware availability overlay (the
    PRIMARY source :func:`build_available_power` sums), or the un-derated
    rated envelope (the FALLBACK — used only where the overlay carries no
    entry, never as the primary source)."""

    OVERLAY = "overlay"
    RATED_ENVELOPE = "rated_envelope"


class _UnresolvableDispUsih(Exception):
    """Internal control-flow signal: one bucket-C ``disp_usih`` term's
    available power could not be resolved from *either*
    :class:`AvailablePower` tier (Requirement 5) — the plant is absent from
    both the availability overlay and the rated-envelope fallback.

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

    ``overlay`` is the PRIMARY source, keyed by the plant's deck code (never
    its ``hydro_id``, since :func:`resolve_disp_usih` resolves a term by the
    code it was parsed with): ``convert_hydro_group_availability``'s
    per-(hydro, group, stage) ``availability_mw`` (``installed × MP × FD``,
    hydraulic-capped), summed across a plant's unit-groups per stage.
    ``rated_envelope`` is the FALLBACK ONLY, keyed by deck code: the
    un-derated ``_rated_envelope`` ``max_generation`` for a plant/stage the
    overlay carries no entry for — never read as the primary source
    (Requirement 1's pitfall).
    """

    overlay: Mapping[tuple[int, int], float]
    rated_envelope: Mapping[int, float]

    def resolve(
        self, code: int, stage_index: int
    ) -> tuple[float, AvailablePowerSource]:
        """*code*'s ``A_h`` at *stage_index*, and which tier produced it.

        Tries the maintenance-aware overlay first (the PRIMARY source);
        falls back to the un-derated rated envelope only when the overlay
        carries no entry for this exact ``(code, stage_index)`` — never the
        other way around.

        Raises
        ------
        _UnresolvableDispUsih
            When *code* is absent from *both* tiers (Requirement 5).
        """
        overlay_value = self.overlay.get((code, stage_index))
        if overlay_value is not None:
            return overlay_value, AvailablePowerSource.OVERLAY
        fallback_value = self.rated_envelope.get(code)
        if fallback_value is not None:
            return fallback_value, AvailablePowerSource.RATED_ENVELOPE
        raise _UnresolvableDispUsih(
            f"plant {code} has no available power at stage {stage_index} "
            "(absent from both the availability overlay and the rated "
            "envelope)"
        )


def build_available_power(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    overlay: Mapping[tuple[int, int, int], GroupBoundEntry],
    effective: EffectiveCadastro,
) -> AvailablePower:
    """Build the :class:`AvailablePower` lookup (Requirement 1) from
    :func:`~cobre_bridge.decomp.converters.hydro.bounds.
    convert_hydro_group_availability`'s raw per-(hydro_id,
    hydro_unit_group_id, stage_id) *overlay*.

    Sums each entry's ``max_generation_mw`` across a plant's unit-groups per
    stage, re-keying the ``hydro_id`` it is stored under onto the plant's own
    deck code (:meth:`~cobre_bridge.decomp.id_map.DecompIdMap.hydro_id`'s
    inverse) — :func:`resolve_disp_usih` resolves a parsed ``disp_usih(code)``
    term by that same deck code, never by ``hydro_id``. A group with no entry
    at a given stage contributes nothing to the sum: *overlay* is sparse by
    construction (only stores a value where the maintenance-derated figure
    falls below that group's own declared envelope — spec's B8 "only where it
    differs" convention), so a stage/group absent from it means "no further
    restriction beyond the rated envelope", not "zero" — which is exactly
    what the :meth:`AvailablePower.resolve` fallback then supplies.

    Precomputes the FALLBACK tier for every declared hydro plant
    (:attr:`~cobre_bridge.decomp.id_map.DecompIdMap.hydro_codes`) via
    :func:`~cobre_bridge.decomp.converters.hydro.bounds._rated_envelope`'s
    ``max_generation`` — the un-derated rated envelope — so
    :meth:`AvailablePower.resolve` never needs *hidr*/*effective* itself at
    resolve time.
    """
    hidr = case.hidr
    code_by_hydro_id = {id_map.hydro_id(code): code for code in id_map.hydro_codes}

    overlay_sums: dict[tuple[int, int], float] = {}
    for (hydro_id, _group_id, stage_id), entry in overlay.items():
        if entry.max_generation_mw is None:
            continue
        code = code_by_hydro_id.get(hydro_id)
        if code is None:
            continue
        key = (code, stage_id)
        overlay_sums[key] = overlay_sums.get(key, 0.0) + float(
            entry.max_generation_mw  # type: ignore[arg-type]
        )

    rated_envelope = {
        code: _rated_envelope(hidr.loc[code], code, effective)[1]
        for code in id_map.hydro_codes
    }

    return AvailablePower(overlay=overlay_sums, rated_envelope=rated_envelope)


def resolve_disp_usih(term: ParsedTerm, stage_index: int, a_h: AvailablePower) -> float:
    """*term*'s (a bucket-C ``disp_usih`` term) plant's ``A_h`` at
    *stage_index* (Requirement 2) — the value :func:`build_available_power`
    resolved: the per-plant availability-overlay sum when present, else the
    rated-envelope fallback.

    Raises
    ------
    _UnresolvableDispUsih
        Propagated from :meth:`AvailablePower.resolve` when the plant is
        absent from *both* tiers.
    """
    value, _source = a_h.resolve(term.args[0], stage_index)
    return value


def _flip_operator(sense: str) -> str:
    """The sense-flipped counterpart of one INEQUACAO comparison operator.

    Multiplying an inequality by ``-1`` swaps ``>=``/``<=`` and leaves
    ``==`` unchanged — exactly the operator-side half of the reserve->gen-cap
    rewrite (:func:`_fold_reserve_disp_usih`).

    Raises
    ------
    ValueError
        When *sense* is not one of ``">="``, ``"<="``, ``"=="`` — mirroring
        :func:`~cobre_bridge.core.generic_constraint_format.sense_to_interval`.
    """
    if sense == ">=":
        return "<="
    if sense == "<=":
        return ">="
    if sense == "==":
        return "=="
    raise ValueError(f"_flip_operator: unknown operator {sense!r}")


def _emit_disp_usih_substitution(
    restriction_code: int,
    plant_code: int,
    stage_index: int,
    a_h_value: float,
    source: AvailablePowerSource,
) -> None:
    """Emit the spec-§4b fidelity diagnostic for one ``disp_usih -> A_h``
    substitution (Requirement 3), naming the plant code, the stage, the
    ``A_h`` value used, and which :class:`AvailablePowerSource` tier
    produced it (Requirement 1's "log which was used").

    ``Severity.INFO`` for the maintenance-aware overlay (the intended
    primary path); ``Severity.WARNING`` for the un-derated rated-envelope
    fallback — a second approximation stacked on top of the reference-head
    one, worth flagging more loudly than the primary path.
    """
    severity = (
        Severity.INFO if source is AvailablePowerSource.OVERLAY else Severity.WARNING
    )
    source_label = (
        "the maintenance-aware availability overlay"
        if source is AvailablePowerSource.OVERLAY
        else "the un-derated rated-envelope fallback"
    )
    emit(
        Diagnostic(
            code="decomp-electrical-disp-usih-substituted",
            severity=severity,
            category="Special constraints",
            title="disp_usih resolved to a reference-head available-power constant",
            summary=(
                f"Restriction {restriction_code}: disp_usih({plant_code}) at "
                f"stage {stage_index} substituted A_h={a_h_value:g} MW, from "
                f"{source_label}."
            ),
            remediation=(
                "This reformulates the reserve constraint into a generation "
                "cap at a fixed reference-head available-power value; "
                "check the source model's "
                f"maintenance/availability data for plant {plant_code} if "
                "this value seems stale."
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
    (this is the A1 fix: a caller reading ``restriction.lhs`` verbatim would
    keep its original negative coefficient, silently inverting the bound) —
    flips the operator's sense; any bucket-B terms on either side fold into
    the same bound.

    Raises
    ------
    ValueError
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
        ``A_h`` cannot be resolved from either :class:`AvailablePower` tier
        (Requirement 5) — the caller (:func:`assemble_bound`) is the sole
        catcher.
    """
    assert restriction.operator is not None  # INEQUACAO invariant

    if rhs_partition[Bucket.C] or rhs_partition[Bucket.A]:
        raise ValueError(
            f"restriction {restriction.code}: the disp_usih reserve pattern "
            "requires its ger_usih/disp_usih pair on the LHS; the RHS "
            "carries a decision or disp_usih term"
        )

    disp_terms = lhs_partition[Bucket.C]
    ger_terms = lhs_partition[Bucket.A]
    non_ger_terms = [term for term in ger_terms if term.token != "ger_usih"]
    if non_ger_terms:
        raise ValueError(
            f"restriction {restriction.code}: the disp_usih reserve pattern "
            "requires every paired decision term to be ger_usih; found "
            f"{non_ger_terms[0].token!r}"
        )

    ger_by_plant = {term.args[0]: term for term in ger_terms}
    if len(ger_by_plant) != len(ger_terms):
        raise ValueError(
            f"restriction {restriction.code}: the disp_usih reserve pattern "
            "requires at most one ger_usih term per plant"
        )

    reserve_sum = 0.0
    for disp_term in disp_terms:
        code = disp_term.args[0]
        ger_term = ger_by_plant.pop(code, None)
        if ger_term is None or ger_term.coefficient != -disp_term.coefficient:
            raise ValueError(
                f"restriction {restriction.code}: disp_usih({code}) is not "
                "paired with a ger_usih term of the exact opposite "
                "coefficient — outside the documented reserve pattern"
            )
        a_h_value, source = a_h.resolve(code, stage_index)
        reserve_sum += disp_term.coefficient * a_h_value
        _emit_disp_usih_substitution(
            restriction.code, code, stage_index, a_h_value, source
        )

    if ger_by_plant:
        leftover_code = next(iter(ger_by_plant))
        raise ValueError(
            f"restriction {restriction.code}: ger_usih({leftover_code}) has "
            "no paired disp_usih term — outside the documented reserve "
            "pattern"
        )

    lhs_b_sum = _evaluate_side(lhs_partition[Bucket.B], ctx)
    rhs_b_sum = _evaluate_side(rhs_partition[Bucket.B], ctx)
    bound_value = reserve_sum + lhs_b_sum - rhs_b_sum
    lower, upper = sense_to_interval(_flip_operator(restriction.operator), bound_value)
    return AssembledBound(
        terms=tuple(_negate(term) for term in ger_terms), lower=lower, upper=upper
    )
