"""Turn a resolved electrical restriction's surviving bucket-A terms into a
novomodelo expression string.

``libs_electrical.assemble_bound`` folds a restriction's bucket-B/-C terms
into a numeric bound and returns its surviving bucket-A (novomodelo decision)
terms as a sign-canonical :class:`~novomodelo_bridge.core.electrical_expression.
AssembledBound` — a source-model :class:`~novomodelo_bridge.core.
electrical_expression.ParsedTerm` sequence, not yet a novomodelo token. This module
owns that last mile: :func:`_novomodelo_token` maps one such term to its novomodelo
``VariableRef`` string, and :func:`build_electrical_expression`
joins a whole :class:`AssembledBound`'s terms into one expression via
:func:`~novomodelo_bridge.core.generic_constraint_format.render_expression`,
the renderer every generic-constraint emitter uses.

This mirrors the codebase's existing read/emit split
(``constraint_registers.py`` reads/classifies, ``constraints.py`` emits):
``libs_electrical.py`` stays novomodelo-agnostic (its ``ParsedTerm``s carry the
source model's own token vocabulary — ``ger_usih``, ``ener_interc``, …),
and this module is where that vocabulary turns into novomodelo's.

:func:`emit_libs_electrical_generics` is the other half of that
last mile: for each restriction, it drives ``active_cells``/``assemble_bound``
over every active ``(stage, block)`` cell, renders the cell-invariant
expression once via :func:`build_electrical_expression`, and synthesizes a
per-restriction :class:`~novomodelo_bridge.decomp.constraint_registers.
ConstraintRecord` that feeds the **existing**
:class:`~novomodelo_bridge.core.generic_constraint_builder.GenericConstraintBuilder` —
never a new emitter. The pipeline wiring that builds *id_map*/*ncs_id_by_pee_code*/
*conjh_bus_by_code_group*/*line_map* from a real deck lives in the pipeline
caller — this module only ever consumes those maps, never
builds them (except :mod:`novomodelo_bridge.decomp.converters.ncs`'s own
``build_pee_ncs_id_map``, which lives next to the ``ncs_id`` assignment it
mirrors).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    collect,
    emit,
    merge_tables,
)
from novomodelo_bridge.core.electrical_expression import (
    CMO_DEMAND_TOKENS,
    GNL_TOKEN,
    SE_TOKEN,
    AssembledBound,
    DataContext,
    MalformedElectricalExpression,
    ParsedTerm,
    UnrecognizedElectricalToken,
    UnsupportedElectricalExpression,
    demand_in_cmo_diagnostic,
    term_text,
)
from novomodelo_bridge.core.errors import FieldParseError, UnsupportedSourceError
from novomodelo_bridge.core.generic_constraint_builder import (
    ConstraintIdAllocator,
    GenericConstraintBuilder,
    soft_slack,
)
from novomodelo_bridge.core.generic_constraint_format import render_expression
from novomodelo_bridge.core.outside_terms import OutsideTerm
from novomodelo_bridge.decomp.constraint_registers import ConstraintRecord, StageBounds
from novomodelo_bridge.decomp.converters.anticipated import (
    gnl_patamar_values_diagnostic,
)
from novomodelo_bridge.decomp.converters.constraints import (
    Unresolved,
    _hydro_generation_token,
    slots_from_record,
)
from novomodelo_bridge.decomp.converters.libs_electrical import (
    _INEQUACAO_CARD,
    _INEQUACAO_PERIOD_CARD,
    DISP_USIH_SUBSTITUTED,
    _effective_inequacao_sides,
    active_cells,
    assemble_bound,
    evaluate_se,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from novomodelo_bridge.decomp.converters.constraints import GenericConstraintResult
    from novomodelo_bridge.decomp.converters.libs_electrical import (
        AvailablePower,
        ElectricalRestriction,
        LibsElectricalModel,
    )
    from novomodelo_bridge.decomp.id_map import DecompIdMap
    from novomodelo_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)


_OUTSIDE_CASE = "decomp-electrical-bucket-a-unresolved"
_ENER_COMERC = "decomp-electrical-ener-comerc-deferred"
_COUPLING_ONLY = "decomp-electrical-term-coupling-only-plant"
_TERM_NO_LINE = "decomp-electrical-term-interchange-no-line"

_TERM_FINDINGS: dict[str, tuple[str, str, list[str]]] = {
    _OUTSIDE_CASE: (
        "Electrical restrictions on entities outside the case",
        "The listed electrical restrictions in the LIBS file name a plant, park "
        "or line the case does not have. The conversion leaves each restriction "
        "out, so the case runs without these limits.",
        ["Restriction", "Term", "Missing"],
    ),
    _ENER_COMERC: (
        "Electrical restrictions on commercial energy",
        "The listed electrical restrictions in the LIBS file name ener_comerc, "
        "the commercial energy between two submarkets, which the case has no "
        "variable for. The conversion leaves each restriction out, so the case "
        "runs without these limits.",
        ["Restriction", "Term"],
    ),
}


#: ``(cause, title, summary, missing)`` of each kind of term left out of a
#: kept restriction; *missing* names the entity when no other term is left.
_LEFT_OUT_FINDINGS: dict[str, tuple[OutsideTerm, str, str, str]] = {
    _COUPLING_ONLY: (
        OutsideTerm.NOT_GENERATING,
        "Electrical-restriction terms on plants DECOMP does not operate left out",
        "The listed electrical restrictions in the LIBS file name a plant "
        "whose UH record in dadger.rvN gives no initial volume or is marked "
        "NW: DECOMP does not operate the plant, so the term adds nothing "
        "there either. The conversion leaves the term out and keeps the "
        "rest of the restriction.",
        "hydro plant",
    ),
    _TERM_NO_LINE: (
        OutsideTerm.NO_LINE,
        "Electrical-restriction terms on submarkets without interchange left out",
        "The listed electrical restrictions in the LIBS file have an "
        "ener_interc term between two submarkets no IA record of dadger.rvN "
        "joins, so no flow runs between them. The conversion leaves the "
        "term out and keeps the rest of the restriction.",
        "line",
    ),
}


def _outside_case(term: ParsedTerm, missing: str) -> Unresolved:
    return Unresolved(_OUTSIDE_CASE, (term_text(term), missing), OutsideTerm.ABSENT)


def _left_out(code: str, term: ParsedTerm) -> Unresolved:
    return Unresolved(code, (term_text(term),), _LEFT_OUT_FINDINGS[code][0])


def _emit_unresolved_term(restriction: int, unresolved: Unresolved) -> None:
    """Emit the one-row WARNING for *restriction*, left out over the term
    *unresolved* describes."""
    title, summary, columns = _TERM_FINDINGS[unresolved.code]
    emit(
        Diagnostic(
            code=unresolved.code,
            severity=Severity.WARNING,
            category="Special constraints",
            title=title,
            summary=summary,
            table=DiagnosticTable(
                columns=columns, rows=[[restriction, *unresolved.cells]]
            ),
        ),
        logger=_LOG,
    )


def _emit_left_out_terms(restriction: int, left_out: Sequence[Unresolved]) -> None:
    """Emit one diagnostic per kind of term left out of the kept
    *restriction*, at its cause's severity."""
    for code, (cause, title, summary, _) in _LEFT_OUT_FINDINGS.items():
        found = [u for u in left_out if u.code == code]
        if not found:
            continue
        emit(
            Diagnostic(
                code=code,
                severity=cause.severity,
                category="Special constraints",
                title=title,
                summary=summary,
                table=DiagnosticTable(
                    columns=["Restriction", "Term"],
                    rows=[[restriction, *u.cells] for u in found],
                ),
            ),
            logger=_LOG,
        )


def _emit_unrecognized_token(restriction: int, name: str) -> None:
    """Emit the one-row ``decomp-electrical-unrecognized-token`` WARNING for
    *restriction*, left out over the undeclared *name*."""
    emit(
        Diagnostic(
            code="decomp-electrical-unrecognized-token",
            severity=Severity.WARNING,
            category="Special constraints",
            title="Electrical restrictions with an undeclared name",
            summary=(
                "The listed electrical restrictions name an expression or alias "
                "the LIBS file does not declare. The conversion leaves each "
                "restriction out, so the case runs without these limits."
            ),
            table=DiagnosticTable(
                columns=["Restriction", "Name"], rows=[[restriction, name]]
            ),
            remediation=(
                "Declare each listed name in the LIBS electrical file, or "
                "correct it in the restriction."
            ),
        ),
        logger=_LOG,
    )


def _resolve_interc_bus(code: int, id_map: DecompIdMap) -> int:
    """Resolve one ``ener_interc`` operand submarket code to a bus id.

    Returns ``id_map.bus_id(code)`` when *code* is a declared SB submarket
    code; otherwise returns ``id_map.transhipment_bus_id``. The IV
    transshipment bus is the ONLY converter-created bus without an SB code
    (:class:`~novomodelo_bridge.decomp.id_map.DecompIdMap`'s own invariant), so an
    operand code absent from ``bus_codes`` can only name it — never a
    hardcoded code value.

    This never raises: the caller's two-probe ``line_map`` lookup is what
    keeps the fallback safe — a genuinely-unknown code that is NOT the IV
    node resolves here but still has no matching line, so it still drops its
    whole restriction.
    """
    if code in id_map.bus_codes:
        return id_map.bus_id(code)
    return id_map.transhipment_bus_id


def _novomodelo_token(
    term: ParsedTerm,
    id_map: DecompIdMap,
    ncs_id_by_pee_code: Mapping[int, int],
    conjh_bus_by_code_group: Mapping[tuple[int, int], int],
    line_map: Mapping[tuple[int, int], int],
) -> str | Unresolved:
    """The novomodelo ``VariableRef`` expression token for one bucket-A term.

    - ``ger_usih(h)`` -> ``hydro_generation(id_map.hydro_id(h))``.
    - ``ger_usit(t)`` -> ``thermal_generation(id_map.thermal_id(t))``.
    - ``ger_pee(p)`` -> ``non_controllable_generation(ncs_id_by_pee_code[p])``
      — *p* is the source model's ``codigo_pee``, never a novomodelo ncs id
      directly; it always goes through *ncs_id_by_pee_code*.
    - ``ger_conjh(x, y)`` -> ``hydro_generation(id_map.hydro_id(x),
      bus=conjh_bus_by_code_group[(x, y)])`` — the frequency-split
      generation selector (Itaipu ``(66, 1)``/``(66, 2)`` -> its IV/SE bus).
    - ``ener_interc(a, b)`` -> a two-probe direct/reverse line lookup on
      *line_map*, keyed by ``(_resolve_interc_bus(a, id_map),
      _resolve_interc_bus(b, id_map))`` — the same direct/reverse convention
      as ``constraints.resolve_fi_term``, but keyed on the numeric submarket
      codes ``ener_interc`` carries, never submarket names. An operand code
      that is not a declared SB code resolves to the IV transshipment bus
      (:func:`_resolve_interc_bus`); the line-lookup miss path still guards
      against a genuinely-unknown code with no matching transshipment line.
    - ``ener_comerc(c)`` -> no novomodelo variable; always
      :class:`~novomodelo_bridge.decomp.converters.constraints.Unresolved`.

    Returns an ``Unresolved`` naming the finding when *term*'s plant,
    thermal, park, bus, or line is absent from its map, or when *term* is
    ``ener_comerc``. Two of them leave out only the term, which is zero in
    DECOMP too: a ``ger_usih`` or ``ger_conjh`` on a plant of
    ``id_map.idle_hydro_codes``, which generates nothing, and an
    ``ener_interc`` between two SB submarkets no line joins. An operand that
    fell back to the IV bus may be an undeclared code, so its missing line
    still leaves the whole restriction out.

    Raises
    ------
    ValueError
        When ``term.token`` is not one of the six bucket-A tokens — a
        caller-contract violation, since :class:`~novomodelo_bridge.core.
        electrical_expression.AssembledBound.terms` only ever carries bucket-A
        terms.
    """
    if term.token == "ger_usih":
        code = term.args[0]
        if code in id_map.idle_hydro_codes:
            return _left_out(_COUPLING_ONLY, term)
        try:
            hydro_id = id_map.hydro_id(code)
        except KeyError:
            return _outside_case(term, "hydro plant")
        return _hydro_generation_token(hydro_id)

    if term.token == "ger_usit":
        code = term.args[0]
        try:
            thermal_id = id_map.thermal_id(code)
        except KeyError:
            return _outside_case(term, "thermal")
        return f"thermal_generation({thermal_id})"

    if term.token == "ger_pee":
        code = term.args[0]
        try:
            ncs_id = ncs_id_by_pee_code[code]
        except KeyError:
            return _outside_case(term, "renewable park")
        return f"non_controllable_generation({ncs_id})"

    if term.token == "ger_conjh":
        code, group = term.args
        if code in id_map.idle_hydro_codes:
            return _left_out(_COUPLING_ONLY, term)
        try:
            hydro_id = id_map.hydro_id(code)
        except KeyError:
            return _outside_case(term, "hydro plant")
        try:
            bus_id = conjh_bus_by_code_group[(code, group)]
        except KeyError:
            return _outside_case(term, "unit group bus")
        return _hydro_generation_token(hydro_id, bus_id)

    if term.token == "ener_interc":
        submarket_a, submarket_b = term.args
        bus_a = _resolve_interc_bus(submarket_a, id_map)
        bus_b = _resolve_interc_bus(submarket_b, id_map)
        if (bus_a, bus_b) in line_map:
            return f"line_direct({line_map[(bus_a, bus_b)]})"
        if (bus_b, bus_a) in line_map:
            return f"line_reverse({line_map[(bus_b, bus_a)]})"
        if submarket_a in id_map.bus_codes and submarket_b in id_map.bus_codes:
            return _left_out(_TERM_NO_LINE, term)
        return _outside_case(term, "line")

    if term.token == "ener_comerc":
        return Unresolved(_ENER_COMERC, (term_text(term),))

    raise ValueError(f"unrecognized decision token {term.token!r}")


def build_electrical_expression(
    assembled: AssembledBound,
    id_map: DecompIdMap,
    ncs_id_by_pee_code: Mapping[int, int],
    conjh_bus_by_code_group: Mapping[tuple[int, int], int],
    line_map: Mapping[tuple[int, int], int],
    left_out: list[Unresolved],
) -> str | Unresolved | None:
    """Render *assembled*'s sign-canonical bucket-A terms into one novomodelo
    expression string.

    Maps every term via :func:`_novomodelo_token` to ``(term.coefficient,
    token)`` and joins the result with :func:`~novomodelo_bridge.core.
    generic_constraint_format.render_expression`. The first term that drops
    its constraint (:attr:`Unresolved.drops_constraint`) drops the **whole**
    expression, skip-not-partial: never a partial string built from only the
    resolvable terms. A term that is zero in DECOMP too is left out of the
    sum and its finding added to *left_out*; when no other term is left, the
    restriction is unresolved under ``decomp-electrical-bucket-a-unresolved``
    on the first of them. ``None`` when *assembled* has no decision term.
    """
    resolved: list[tuple[float, str]] = []
    zero: list[Unresolved] = []
    for term in assembled.terms:
        token = _novomodelo_token(
            term, id_map, ncs_id_by_pee_code, conjh_bus_by_code_group, line_map
        )
        if isinstance(token, str):
            resolved.append((term.coefficient, token))
        elif token.drops_constraint:
            return token
        else:
            zero.append(token)
    if zero and not resolved:
        first = zero[0]
        return Unresolved(
            _OUTSIDE_CASE, (*first.cells, _LEFT_OUT_FINDINGS[first.code][3])
        )
    left_out.extend(zero)
    return render_expression(resolved)


# ---------------------------------------------------------------------------
# Emit resolved restrictions through the shared GenericConstraintBuilder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibsElectricalResult:
    """The outcome of emitting every resolved electrical restriction as one
    novomodelo generic constraint.

    ``generic`` is the surviving :class:`~novomodelo_bridge.decomp.converters.constraints.
    GenericConstraintResult` (``None`` when no restriction survives).
    ``converted_codes`` is every restriction code that was emitted.
    ``deferred`` maps each drop reason to the restriction codes dropped for
    it — ``"inactive"`` (``active_cells`` found no active cell across the
    restriction's horizon; it already emitted the one INFO diagnostic for
    this), ``"unresolved-bucket-bc"`` (``assemble_bound`` could not resolve
    a bucket-B/-C term for some active cell; already WARNED),
    ``"unresolved-bucket-a"`` (``build_electrical_expression`` could not
    resolve a bucket-A decision term; already WARNED), and
    ``"unrecognized-token"`` (the restriction's formula references a
    well-formed but undeclared identifier —
    :class:`~novomodelo_bridge.core.electrical_expression.UnrecognizedElectricalToken`,
    already WARNED). All four keys are always present, empty
    when nothing was dropped for that reason — this is the census data
    the census diagnostic renders.
    """

    generic: GenericConstraintResult | None
    converted_codes: tuple[int, ...]
    deferred: Mapping[str, tuple[int, ...]]


def _synthesize_stage_bounds(
    assembled_by_cell: Mapping[tuple[int, int], AssembledBound],
    calendar: Sequence[OperativeStage],
) -> dict[int, StageBounds]:
    """Group *assembled_by_cell* by stage into full-length per-block
    :class:`StageBounds`, carrying each active cell's ``lower``/``upper`` at
    its own block index and ``None`` at every inactive block of that stage.

    Each stage's tuples are exactly ``len(calendar[stage].block_hours)``
    long — never shorter — since ``constraints.slots_from_record`` clamps a
    short tuple to the stage's real block count, which would silently drop
    a trailing active block instead of emitting its bound row (spec
    pitfall). A stage with no active cell at all is never a key here —
    *assembled_by_cell* only ever carries active cells.
    """
    by_stage: dict[int, list[tuple[int, AssembledBound]]] = {}
    for (stage_index, block_index), assembled in assembled_by_cell.items():
        by_stage.setdefault(stage_index, []).append((block_index, assembled))

    bounds: dict[int, StageBounds] = {}
    for stage_index, entries in by_stage.items():
        n_blocks = len(calendar[stage_index].block_hours)
        lower: list[float | None] = [None] * n_blocks
        upper: list[float | None] = [None] * n_blocks
        for block_index, assembled in entries:
            lower[block_index] = assembled.lower
            upper[block_index] = assembled.upper
        bounds[stage_index] = StageBounds(lower=tuple(lower), upper=tuple(upper))
    return bounds


def _resolve_violation_penalty(
    restriction: ElectricalRestriction, big_m: float
) -> float:
    """The slack penalty for one restriction's generic constraint.

    ``restriction.violation.penalty`` overrides *big_m* exactly when a
    ``TRATAMENTO-VIOLACAO`` row is present and its penalty is a positive
    number; otherwise the classic BIG-M fallback applies. Never inspects
    ``violation.enabled`` — a present :class:`ViolationTreatment` is always
    enabled (its own docstring), so the check would be redundant.
    """
    violation = restriction.violation
    if (
        violation is not None
        and violation.penalty is not None
        and violation.penalty > 0
    ):
        return violation.penalty
    return big_m


def emit_libs_electrical_generics(
    model: LibsElectricalModel,
    id_map: DecompIdMap,
    context_factory: Callable[[int, int], DataContext],
    a_h: AvailablePower,
    calendar: Sequence[OperativeStage],
    ncs_id_by_pee_code: Mapping[int, int],
    conjh_bus_by_code_group: Mapping[tuple[int, int], int],
    line_map: Mapping[tuple[int, int], int],
    big_m: float,
    allocator: ConstraintIdAllocator | None = None,
    *,
    source: Path,
) -> LibsElectricalResult:
    """Emit every resolved electrical restriction as one novomodelo generic
    constraint, feeding the **existing** ``GenericConstraintBuilder`` the
    other special-constraint emitters use — never a new emitter.

    Iterates ``sorted(model.restrictions)`` (deterministic order). For each
    restriction:

    - ``active_cells`` resolves the exact set of active
      ``(stage, block)`` cells. An empty set means ``active_cells`` already
      emitted the one INFO diagnostic; this records ``deferred["inactive"]``
      and never emits a second one.
    - ``assemble_bound`` folds every active cell's bucket-
      B/-C terms into its numeric bound, leaving only the bucket-A decision
      terms structural. A single unresolvable cell (``assemble_bound``
      returning ``None``, already WARNED) drops the **whole** restriction —
      ``deferred["unresolved-bucket-bc"]`` — skip-not-partial.
    - :func:`build_electrical_expression` renders one cell's
      surviving terms into the novomodelo expression string exactly once, since
      the structural expression is cell-invariant by construction; every
      other active cell's terms are asserted equal to it — a mismatch is a
      resolver bug and raises ``ValueError`` naming the restriction (fail
      loud, never silently pick one). An unresolved bucket-A token (already
      WARNED) drops the whole restriction — ``deferred["unresolved-bucket-a"]``;
      a term on a coupling-only plant is only left out of the sum, and a kept
      restriction's such terms are reported in one INFO. So are the stages
      where a kept restriction reads a GNL plant's generation that differs
      by patamar.
    - A well-formed but undeclared identifier anywhere in the restriction's
      (expanded) formula OR its activation rule (:class:`~novomodelo_bridge.
      core.electrical_expression.UnrecognizedElectricalToken`, raised from
      inside ``active_cells``'s, ``assemble_bound``'s, or
      ``build_electrical_expression``'s own ``parse_linear_expression``
      call) drops **only this restriction** —
      ``deferred["unrecognized-token"]`` — WARNed here and skipped, never
      aborting the rest of the deck. Malformed text
      (:class:`~novomodelo_bridge.core.electrical_expression.
      MalformedElectricalExpression`), a dangling ``HABILITA`` included,
      stops the conversion as a :class:`~novomodelo_bridge.core.errors.
      FieldParseError` naming *source* (the LIBS electrical file), the
      restriction and the card: ``RESTRICAO-ELETRICA-REGRA-ATIVACAO`` for the
      activation rule, else the card that declares the failing cell's
      formula, or the formula's limits card when an active cell has no
      limit. A valid restriction the conversion does not convert yet, a
      formula the grammar does not read yet
      (:class:`~novomodelo_bridge.core.electrical_expression.
      UnsupportedElectricalExpression`) included, stops it as an
      :class:`~novomodelo_bridge.core.errors.UnsupportedSourceError` naming
      the same file, restriction and card, or the period rows when its
      decision terms differ between cells.
    - The surviving expression and per-active-cell bound values feed a
      synthesized :class:`~novomodelo_bridge.decomp.constraint_registers.
      ConstraintRecord` (``family="LIBS_ELEC"``, ``per_block=True``,
      ``terms=()`` — ``slots_from_record`` reads only ``record.bounds``/
      ``record.per_block``), named ``LIBS_ELEC_{code}``.

    The slack penalty is *big_m* unless the restriction's own
    ``TRATAMENTO-VIOLACAO`` penalty overrides it (:func:`
    _resolve_violation_penalty`) — slack stays enabled either way,
    so a negative generation-cap bound (heavy maintenance shrinking
    available power below the reserve requirement) stays a priced
    violation, never a hard LP infeasibility.

    Reads no deck: every input — *id_map*, *context_factory*, *a_h*,
    *calendar*, the three id maps, *big_m* — arrives from the caller.
    *allocator* defaults to a fresh :class:`ConstraintIdAllocator` (ids from
    0) when omitted; the pipeline threads one shared instance across every
    emitter.
    """
    found: list[Diagnostic] = []
    try:
        with collect() as found:
            result = _emit_restrictions(
                model,
                id_map,
                context_factory,
                a_h,
                calendar,
                ncs_id_by_pee_code,
                conjh_bus_by_code_group,
                line_map,
                big_m,
                allocator or ConstraintIdAllocator(),
                source,
            )
    finally:
        for diagnostic in merge_tables(found):
            emit(diagnostic, logger=_LOG)
    return result


def _emit_restrictions(
    model: LibsElectricalModel,
    id_map: DecompIdMap,
    context_factory: Callable[[int, int], DataContext],
    a_h: AvailablePower,
    calendar: Sequence[OperativeStage],
    ncs_id_by_pee_code: Mapping[int, int],
    conjh_bus_by_code_group: Mapping[tuple[int, int], int],
    line_map: Mapping[tuple[int, int], int],
    big_m: float,
    allocator: ConstraintIdAllocator,
    source: Path,
) -> LibsElectricalResult:
    """The body of :func:`emit_libs_electrical_generics`, reporting each
    restriction's findings as it meets them, and the available power it held
    constant only for a restriction it keeps."""
    builder = GenericConstraintBuilder(allocator)
    converted: list[int] = []
    demand_in_cmo: list[tuple[int, str]] = []
    gnl_folds: list[tuple[int, int, int, Sequence[float]]] = []
    deferred: dict[str, list[int]] = {
        "inactive": [],
        "unresolved-bucket-bc": [],
        "unresolved-bucket-a": [],
        "unrecognized-token": [],
    }

    for code in sorted(model.restrictions):
        kept = False
        local: list[Diagnostic] = []
        try:
            with collect() as local:
                kept = _emit_restriction(
                    code,
                    model.restrictions[code],
                    model,
                    id_map,
                    context_factory,
                    a_h,
                    calendar,
                    ncs_id_by_pee_code,
                    conjh_bus_by_code_group,
                    line_map,
                    big_m,
                    builder,
                    deferred,
                    demand_in_cmo,
                    gnl_folds,
                    source,
                )
        finally:
            for diagnostic in local:
                if kept or diagnostic.code != DISP_USIH_SUBSTITUTED:
                    emit(diagnostic, logger=_LOG)
        if kept:
            converted.append(code)

    if demand_in_cmo:
        emit(
            demand_in_cmo_diagnostic(
                "decomp-electrical-demand-in-cmo",
                demand_in_cmo,
                source="the RESTRICAO-ELETRICA-ESPECIAL file",
            ),
            logger=_LOG,
        )
    patamar_values = gnl_patamar_values_diagnostic(
        "decomp-electrical-gnl-patamar-values",
        gnl_folds,
        noun="electrical restrictions",
        column="Restriction",
    )
    if patamar_values is not None:
        emit(patamar_values, logger=_LOG)
    return LibsElectricalResult(
        generic=builder.result(),
        converted_codes=tuple(converted),
        deferred={reason: tuple(codes) for reason, codes in deferred.items()},
    )


_RULE_CARD = "RESTRICAO-ELETRICA-REGRA-ATIVACAO"


def _cell_text(cell: tuple[int, int]) -> str:
    """A ``(stage_index, block_index)`` cell as the deck numbers it."""
    stage_index, block_index = cell
    return f"período {stage_index + 1}, patamar {block_index + 1}"


def _cell_card(
    restriction: ElectricalRestriction, stage_index: int, block_index: int
) -> str:
    """The LIBS card that declares *restriction*'s formula at one cell: an
    inequality's period row where one covers the cell, else its own card."""
    if restriction.is_formula:
        return "RESTRICAO-ELETRICA-FORMULA"
    sides = _effective_inequacao_sides(restriction, stage_index, block_index + 1)
    if sides == (restriction.lhs, restriction.operator, restriction.rhs):
        return _INEQUACAO_CARD
    return _INEQUACAO_PERIOD_CARD


def _emit_restriction(
    code: int,
    restriction: ElectricalRestriction,
    model: LibsElectricalModel,
    id_map: DecompIdMap,
    context_factory: Callable[[int, int], DataContext],
    a_h: AvailablePower,
    calendar: Sequence[OperativeStage],
    ncs_id_by_pee_code: Mapping[int, int],
    conjh_bus_by_code_group: Mapping[tuple[int, int], int],
    line_map: Mapping[tuple[int, int], int],
    big_m: float,
    builder: GenericConstraintBuilder,
    deferred: dict[str, list[int]],
    demand_in_cmo: list[tuple[int, str]],
    gnl_folds: list[tuple[int, int, int, Sequence[float]]],
    source: Path,
) -> bool:
    """Add restriction *code* to *builder* and return ``True``, or record on
    *deferred* why it is dropped and return ``False``.

    A kept restriction's bound terms in :data:`CMO_DEMAND_TOKENS` are added to
    *demand_in_cmo*, and each GNL plant's generation it reads to *gnl_folds*
    as ``(code, thermal code, stage, MW per patamar)``."""
    demand_terms: set[str] = set()
    gnl_reads: dict[tuple[int, int], ParsedTerm] = {}
    card = _RULE_CARD
    left_out: list[Unresolved] = []
    try:
        cells = active_cells(restriction, model, context_factory, calendar)
        if not cells:
            deferred["inactive"].append(code)
            return False

        assembled_by_cell: dict[tuple[int, int], AssembledBound] = {}
        for stage_index, block_index in sorted(cells):
            card = _cell_card(restriction, stage_index, block_index)
            assembled = assemble_bound(
                restriction,
                model,
                stage_index,
                block_index,
                _recording_demand_terms(
                    _recording_gnl_reads(
                        context_factory(stage_index, block_index),
                        model,
                        stage_index,
                        gnl_reads,
                    ),
                    demand_terms,
                ),
                a_h,
            )
            if assembled is None:
                deferred["unresolved-bucket-bc"].append(code)
                return False
            assembled_by_cell[(stage_index, block_index)] = assembled

        reference_cell, reference_assembled = next(iter(assembled_by_cell.items()))
        for cell, assembled in assembled_by_cell.items():
            if assembled.terms != reference_assembled.terms:
                raise UnsupportedSourceError(
                    f"its decision terms in {_cell_text(reference_cell)} differ "
                    f"from those in {_cell_text(cell)}; the conversion carries "
                    "one set of decision terms per restriction",
                    field=_INEQUACAO_PERIOD_CARD,
                )

        expression = build_electrical_expression(
            reference_assembled,
            id_map,
            ncs_id_by_pee_code,
            conjh_bus_by_code_group,
            line_map,
            left_out,
        )
    except UnrecognizedElectricalToken as err:
        _emit_unrecognized_token(code, err.name)
        deferred["unrecognized-token"].append(code)
        return False
    except (MalformedElectricalExpression, FieldParseError) as err:
        if isinstance(err, FieldParseError) and err.path is not None:
            # A field of another file, such as an uncommanded GNL plant's dadgnl GL.
            raise FieldParseError(
                f"{source.name}: restriction {code}: {err}",
                path=err.path,
                field=err.field,
            ) from err
        field = err.field if isinstance(err, FieldParseError) else None
        raise FieldParseError(
            f"{source.name}: restriction {code} is malformed: {err}",
            path=str(source),
            field=field or card,
        ) from err
    except (UnsupportedElectricalExpression, UnsupportedSourceError) as err:
        field = err.field if isinstance(err, UnsupportedSourceError) else None
        raise UnsupportedSourceError(
            f"{source.name}: restriction {code} is not converted yet: {err}",
            path=str(source),
            field=field or card,
        ) from err

    if isinstance(expression, Unresolved):
        _emit_unresolved_term(code, expression)
    if expression is None or isinstance(expression, Unresolved):
        deferred["unresolved-bucket-a"].append(code)
        return False

    bounds = _synthesize_stage_bounds(assembled_by_cell, calendar)
    stages = sorted(bounds)
    record = ConstraintRecord(
        family="LIBS_ELEC",
        constraint_id=code,
        stage_start=stages[0],
        stage_end=stages[-1],
        terms=(),
        bounds=bounds,
        per_block=True,
    )
    builder.add(
        name=f"LIBS_ELEC_{code}",
        description=f"LIBs electrical special constraint {code}",
        expression=expression,
        slack=soft_slack(_resolve_violation_penalty(restriction, big_m), "per_hour"),
        slots=slots_from_record(record, calendar),
    )
    demand_in_cmo.extend((code, term) for term in demand_terms)
    gnl_folds.extend(
        (
            code,
            thermal_code,
            stage_index,
            [
                context_factory(stage_index, block_index)(term)
                for block_index in range(len(calendar[stage_index].block_hours))
            ],
        )
        for (thermal_code, stage_index), term in sorted(gnl_reads.items())
    )
    _emit_left_out_terms(code, left_out)
    return True


def _recording_demand_terms(ctx: DataContext, seen: set[str]) -> DataContext:
    """*ctx*, adding to *seen* each :data:`CMO_DEMAND_TOKENS` term it values."""

    def resolve(term: ParsedTerm) -> float:
        if term.token in CMO_DEMAND_TOKENS:
            seen.add(term_text(term))
        return ctx(term)

    return resolve


def _recording_gnl_reads(
    ctx: DataContext,
    model: LibsElectricalModel,
    stage_index: int,
    seen: dict[tuple[int, int], ParsedTerm],
) -> DataContext:
    """*ctx*, adding to *seen*, under ``(thermal code, stage_index)``, each
    GNL plant's ``ger_usit`` term it values, inside a ``se(...)`` too."""

    def resolve(term: ParsedTerm) -> float:
        if term.token == SE_TOKEN:
            return evaluate_se(term, model, resolve)
        if term.token == GNL_TOKEN and term.coefficient != 0.0:
            seen.setdefault((term.args[0], stage_index), term)
        return ctx(term)

    return resolve
