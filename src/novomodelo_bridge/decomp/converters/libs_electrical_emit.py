"""Turn a resolved electrical restriction's surviving bucket-A terms into a
cobre expression string.

``libs_electrical.assemble_bound`` folds a restriction's bucket-B/-C terms
into a numeric bound and returns its surviving bucket-A (cobre decision)
terms as a sign-canonical :class:`~cobre_bridge.decomp.converters.libs_electrical.
AssembledBound` — a source-model :class:`~cobre_bridge.decomp.
libs_electrical.ParsedTerm` sequence, not yet a cobre token. This module
owns that last mile: :func:`_cobre_token` maps one such term to its cobre
``VariableRef`` string (spec §2), and :func:`build_electrical_expression`
joins a whole :class:`AssembledBound`'s terms into one expression via
``constraints._format_expression`` — the same coefficient-formatting rules
the E1–E7 emitter already uses.

This mirrors the codebase's existing read/emit split
(``constraint_registers.py`` reads/classifies, ``constraints.py`` emits):
``libs_electrical.py`` stays cobre-agnostic (its ``ParsedTerm``s carry the
source model's own token vocabulary — ``ger_usih``, ``ener_interc``, …),
and this module is where that vocabulary turns into cobre's.

:func:`emit_libs_electrical_generics` is the other half of that
last mile: for each restriction, it drives ``active_cells``/``assemble_bound``
over every active ``(stage, block)`` cell, renders the cell-invariant
expression once via :func:`build_electrical_expression`, and synthesizes a
per-restriction :class:`~cobre_bridge.decomp.constraint_registers.
ConstraintRecord` that feeds the **existing** E1–E7
:class:`~cobre_bridge.core.generic_constraint_builder.GenericConstraintBuilder` —
never a new emitter. The pipeline wiring that builds *id_map*/*ncs_id_by_pee_code*/
*conjh_bus_by_code_group*/*line_map* from a real deck lives in the pipeline
caller — this module only ever consumes those maps, never
builds them (except :mod:`cobre_bridge.decomp.converters.ncs`'s own
``build_pee_ncs_id_map``, which lives next to the ``ncs_id`` assignment it
mirrors).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cobre_bridge.core.diagnostics import Diagnostic, Severity, emit
from cobre_bridge.core.generic_constraint_builder import (
    ConstraintIdAllocator,
    GenericConstraintBuilder,
)
from cobre_bridge.decomp.constraint_registers import ConstraintRecord, StageBounds
from cobre_bridge.decomp.converters.constraints import (
    _format_expression,
    _hydro_generation_token,
    slots_from_record,
)
from cobre_bridge.decomp.converters.libs_electrical import (
    UnrecognizedElectricalToken,
    active_cells,
    assemble_bound,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from cobre_bridge.decomp.converters.constraints import GenericConstraintResult
    from cobre_bridge.decomp.converters.libs_electrical import (
        AssembledBound,
        AvailablePower,
        DataContext,
        ElectricalRestriction,
        LibsElectricalModel,
        ParsedTerm,
    )
    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage


def _emit_bucket_a_unresolved(term: ParsedTerm, *, map_name: str) -> None:
    """Emit the shared ``decomp-electrical-bucket-a-unresolved`` WARNING for
    one bucket-A term whose plant/thermal/park/line is absent from *map_name*."""
    emit(
        Diagnostic(
            code="decomp-electrical-bucket-a-unresolved",
            severity=Severity.WARNING,
            category="Special constraints",
            title="Electrical restriction decision term could not be resolved",
            summary=(
                f"Electrical restriction decision term {term.token}"
                f"{list(term.args)} could not be resolved via {map_name}; "
                "the constraint is skipped rather than emitted with a "
                "partial expression."
            ),
            remediation=(
                "Check the source model's data feeding "
                f"{term.token}{list(term.args)}; the constraint is skipped."
            ),
        )
    )


def _emit_ener_comerc_deferred(term: ParsedTerm) -> None:
    """Emit the shared ``decomp-electrical-ener-comerc-deferred`` WARNING.

    ``ener_comerc`` (bilateral commercial energy) does not appear on the
    target deck (spec §2) and has no cobre ``VariableRef`` counterpart yet;
    a restriction carrying one is skipped rather than guessed at.
    """
    emit(
        Diagnostic(
            code="decomp-electrical-ener-comerc-deferred",
            severity=Severity.WARNING,
            category="Special constraints",
            title="ener_comerc term deferred",
            summary=(
                f"Electrical restriction decision term ener_comerc"
                f"{list(term.args)} has no equivalent variable in the target "
                "model yet; the constraint is skipped."
            ),
            remediation=(
                "ener_comerc does not appear on the target deck; no action "
                "needed to convert."
            ),
        )
    )


def _emit_unrecognized_token(code: int, err: UnrecognizedElectricalToken) -> None:
    """Emit the shared ``decomp-electrical-unrecognized-token`` WARNING for
    one restriction whose (expanded) formula references a well-formed but
    undeclared identifier (skip-not-partial containment)."""
    emit(
        Diagnostic(
            code="decomp-electrical-unrecognized-token",
            severity=Severity.WARNING,
            category="Special constraints",
            title="Electrical restriction references an unrecognized token",
            summary=(
                f"Electrical restriction {code} references an unrecognized "
                f"token ({err}); the constraint is skipped rather than "
                "aborting the whole conversion."
            ),
            remediation=(
                "Check the source model's declared expressions/aliases for "
                f"restriction {code} (see the summary for the offending "
                "identifier); the constraint is skipped until it is "
                "modeled."
            ),
        )
    )


def _resolve_interc_bus(code: int, id_map: DecompIdMap) -> int:
    """Resolve one ``ener_interc`` operand submarket code to a bus id.

    Returns ``id_map.bus_id(code)`` when *code* is a declared SB submarket
    code; otherwise returns ``id_map.transhipment_bus_id``. The IV
    transshipment bus is the ONLY converter-created bus without an SB code
    (:class:`~cobre_bridge.decomp.id_map.DecompIdMap`'s own invariant), so an
    operand code absent from ``bus_codes`` can only name it — never a
    hardcoded code value.

    This never raises: the caller's two-probe ``line_map`` lookup is what
    keeps the fallback safe — a genuinely-unknown code that is NOT the IV
    node resolves here but still has no matching line, so it still drops via
    the existing skip-not-partial path.
    """
    if code in id_map.bus_codes:
        return id_map.bus_id(code)
    return id_map.transhipment_bus_id


def _cobre_token(
    term: ParsedTerm,
    id_map: DecompIdMap,
    ncs_id_by_pee_code: Mapping[int, int],
    conjh_bus_by_code_group: Mapping[tuple[int, int], int],
    line_map: Mapping[tuple[int, int], int],
) -> str | None:
    """The cobre ``VariableRef`` expression token for one bucket-A term (spec §2).

    - ``ger_usih(h)`` -> ``hydro_generation(id_map.hydro_id(h))``.
    - ``ger_usit(t)`` -> ``thermal_generation(id_map.thermal_id(t))``.
    - ``ger_pee(p)`` -> ``non_controllable_generation(ncs_id_by_pee_code[p])``
      — *p* is the source model's ``codigo_pee``, never a cobre ncs id
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
    - ``ener_comerc(c)`` -> not on the target deck; always ``None`` (deferred).

    Returns ``None`` — after emitting one ``Severity.WARNING``
    :class:`~cobre_bridge.core.diagnostics.Diagnostic` — when *term*'s plant,
    thermal, park, bus, or line is absent from its map (a caught
    ``KeyError``, never propagated) or when *term* is ``ener_comerc``.

    Raises
    ------
    ValueError
        When ``term.token`` is not one of the six bucket-A tokens — a
        caller-contract violation, since :class:`~cobre_bridge.decomp.
        libs_electrical.AssembledBound.terms` only ever carries bucket-A
        terms.
    """
    if term.token == "ger_usih":
        code = term.args[0]
        try:
            hydro_id = id_map.hydro_id(code)
        except KeyError:
            _emit_bucket_a_unresolved(term, map_name="the hydro id map")
            return None
        return _hydro_generation_token(hydro_id)

    if term.token == "ger_usit":
        code = term.args[0]
        try:
            thermal_id = id_map.thermal_id(code)
        except KeyError:
            _emit_bucket_a_unresolved(term, map_name="the thermal id map")
            return None
        return f"thermal_generation({thermal_id})"

    if term.token == "ger_pee":
        code = term.args[0]
        try:
            ncs_id = ncs_id_by_pee_code[code]
        except KeyError:
            _emit_bucket_a_unresolved(term, map_name="the renewable-park ncs id map")
            return None
        return f"non_controllable_generation({ncs_id})"

    if term.token == "ger_conjh":
        code, group = term.args
        try:
            hydro_id = id_map.hydro_id(code)
        except KeyError:
            _emit_bucket_a_unresolved(term, map_name="the hydro id map")
            return None
        try:
            bus_id = conjh_bus_by_code_group[(code, group)]
        except KeyError:
            _emit_bucket_a_unresolved(
                term, map_name="the ger_conjh frequency-split bus map"
            )
            return None
        return _hydro_generation_token(hydro_id, bus_id)

    if term.token == "ener_interc":
        submarket_a, submarket_b = term.args
        bus_a = _resolve_interc_bus(submarket_a, id_map)
        bus_b = _resolve_interc_bus(submarket_b, id_map)
        if (bus_a, bus_b) in line_map:
            return f"line_direct({line_map[(bus_a, bus_b)]})"
        if (bus_b, bus_a) in line_map:
            return f"line_reverse({line_map[(bus_b, bus_a)]})"
        _emit_bucket_a_unresolved(term, map_name="the FI line map")
        return None

    if term.token == "ener_comerc":
        _emit_ener_comerc_deferred(term)
        return None

    raise ValueError(f"_cobre_token: unrecognized bucket-A token {term.token!r}")


def build_electrical_expression(
    assembled: AssembledBound,
    id_map: DecompIdMap,
    ncs_id_by_pee_code: Mapping[int, int],
    conjh_bus_by_code_group: Mapping[tuple[int, int], int],
    line_map: Mapping[tuple[int, int], int],
) -> str | None:
    """Render *assembled*'s sign-canonical bucket-A terms into one cobre
    expression string (spec §4).

    Maps every term via :func:`_cobre_token` to ``(term.coefficient,
    token)`` and joins the result with ``constraints._format_expression``
    (the same coefficient-formatting rules — unit coefficients render bare,
    fractional ones as ``{coeff} * {token}`` — the E1–E7 emitter already
    uses). Any single unresolved token (:func:`_cobre_token` returning
    ``None``, having already emitted its own WARNING) drops the **whole**
    expression — ``None``, skip-not-partial — never a partial string built
    from only the resolvable terms.
    """
    resolved: list[tuple[float, str]] = []
    for term in assembled.terms:
        token = _cobre_token(
            term, id_map, ncs_id_by_pee_code, conjh_bus_by_code_group, line_map
        )
        if token is None:
            return None
        resolved.append((term.coefficient, token))
    return _format_expression(resolved)


# ---------------------------------------------------------------------------
# Emit resolved restrictions through the E1–E7 GenericConstraintBuilder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibsElectricalResult:
    """The outcome of emitting every resolved electrical restriction as one
    cobre generic constraint (spec §5).

    ``generic`` is the surviving :class:`~cobre_bridge.decomp.converters.constraints.
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
    :class:`~cobre_bridge.decomp.converters.libs_electrical.UnrecognizedElectricalToken`,
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
    """The slack penalty for one restriction's generic constraint (spec §4d).

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
) -> LibsElectricalResult:
    """Emit every resolved electrical restriction as one cobre generic
    constraint, feeding the **existing** E1–E7 ``GenericConstraintBuilder`` —
    never a new emitter (spec §5).

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
      surviving terms into the cobre expression string exactly once, since
      the structural expression is cell-invariant by construction; every
      other active cell's terms are asserted equal to it — a mismatch is a
      resolver bug and raises ``ValueError`` naming the restriction (fail
      loud, never silently pick one). An unresolved bucket-A token (already
      WARNED) drops the whole restriction — ``deferred["unresolved-bucket-a"]``.
    - A well-formed but undeclared identifier anywhere in the restriction's
      (expanded) formula OR its activation rule (:class:`~cobre_bridge.
      decomp.converters.libs_electrical.UnrecognizedElectricalToken`, raised from
      inside ``active_cells``'s, ``assemble_bound``'s, or
      ``build_electrical_expression``'s own ``parse_linear_expression``
      call) drops **only this restriction** —
      ``deferred["unrecognized-token"]`` — WARNed here and skipped, never
      aborting the rest of the deck. A malformed-DSL ``ValueError`` (not
      this subclass) — including ``active_cells``' dangling-``HABILITA``
      guard — is NOT caught here and still propagates, fail-loud.
    - The surviving expression and per-active-cell bound values feed a
      synthesized :class:`~cobre_bridge.decomp.constraint_registers.
      ConstraintRecord` (``family="LIBS_ELEC"``, ``per_block=True``,
      ``terms=()`` — ``slots_from_record`` reads only ``record.bounds``/
      ``record.per_block``), named ``LIBS_ELEC_{code}``.

    The slack penalty is *big_m* unless the restriction's own
    ``TRATAMENTO-VIOLACAO`` penalty overrides it (:func:`
    _resolve_violation_penalty`, spec §4d) — slack stays enabled either way,
    so a negative generation-cap bound (heavy maintenance shrinking
    available power below the reserve requirement) stays a priced
    violation, never a hard LP infeasibility (A3).

    Reads no deck: every input — *id_map*, *context_factory*, *a_h*,
    *calendar*, the three id maps, *big_m* — arrives from the caller.
    *allocator* defaults to a fresh :class:`ConstraintIdAllocator` (ids from
    0) when omitted; the pipeline threads one shared instance across every
    emitter.
    """
    allocator = allocator or ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)
    converted: list[int] = []
    deferred: dict[str, list[int]] = {
        "inactive": [],
        "unresolved-bucket-bc": [],
        "unresolved-bucket-a": [],
        "unrecognized-token": [],
    }

    for code in sorted(model.restrictions):
        restriction = model.restrictions[code]
        try:
            cells = active_cells(restriction, model, context_factory, calendar)
            if not cells:
                deferred["inactive"].append(code)
                continue

            assembled_by_cell: dict[tuple[int, int], AssembledBound] = {}
            bucket_bc_unresolved = False
            for stage_index, block_index in sorted(cells):
                assembled = assemble_bound(
                    restriction,
                    model,
                    stage_index,
                    block_index,
                    context_factory(stage_index, block_index),
                    a_h,
                )
                if assembled is None:
                    bucket_bc_unresolved = True
                    break
                assembled_by_cell[(stage_index, block_index)] = assembled
            if bucket_bc_unresolved:
                deferred["unresolved-bucket-bc"].append(code)
                continue

            reference_cell, reference_assembled = next(iter(assembled_by_cell.items()))
            for cell, assembled in assembled_by_cell.items():
                if assembled.terms != reference_assembled.terms:
                    raise ValueError(
                        f"restriction {code}: AssembledBound.terms differ "
                        f"between active cell {reference_cell} and {cell} — "
                        "the structural expression must be cell-invariant"
                    )

            expression = build_electrical_expression(
                reference_assembled,
                id_map,
                ncs_id_by_pee_code,
                conjh_bus_by_code_group,
                line_map,
            )
        except UnrecognizedElectricalToken as err:
            _emit_unrecognized_token(code, err)
            deferred["unrecognized-token"].append(code)
            continue

        if expression is None:
            deferred["unresolved-bucket-a"].append(code)
            continue

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
            slack={
                "enabled": True,
                "penalty": _resolve_violation_penalty(restriction, big_m),
            },
            slots=slots_from_record(record, calendar),
        )
        converted.append(code)

    return LibsElectricalResult(
        generic=builder.result(),
        converted_codes=tuple(converted),
        deferred={reason: tuple(codes) for reason, codes in deferred.items()},
    )
