"""Foundation for the source model's generic-constraint emitter.

The special-constraint reader (``constraint_registers.py``) splits every
special constraint into ones that lower to a plain entity bound
(``single_term_bounds.py``) and everything else — multi-term participation,
an unbounded variable (spillage, diversion, pumping), a non-unit
coefficient, or a whole-cascade energy sum — which needs novomodelo's
**generic** constraint wire format instead: a flat ``expression`` string, a
``slack{enabled, penalty}`` pair, and a companion per-
``(constraint_id, stage_id, block_id)`` bounds table carrying novomodelo's F3
sense-free interval endpoints (``bound_lower``/``bound_upper`` — see
:mod:`novomodelo_bridge.core.generic_constraint_format`) instead of a ``sense``/
``bound`` pair.

This module stands up that shared scaffolding: the per-term expression-token
dispatch (``_variable_token``), the ``FI``-interchange line resolver
(``build_fi_line_map``/``resolve_fi_term``), the coefficient-string formatter
(``_format_expression``), the ``BIG_M`` slack-penalty helper, and
``slots_from_record`` (the per-record (stage, block) slot enumeration feeding
the shared :class:`~novomodelo_bridge.core.generic_constraint_builder.
GenericConstraintBuilder`), plus the multi-term ``RE`` emitter
(``emit_re_generics``) and the ``RHQ``/``RHV`` emitter (
``emit_rhq_rhv_generics``) built on top of it. This module mirrors — but
never imports — the sibling ``converters/constraints.py`` emitter's
``_parse_formula`` pattern; both tracks share one
``GenericConstraintBuilder``/``GenericConstraintResult``
(:mod:`novomodelo_bridge.core.generic_constraint_builder`).

GNL pre-processing (feature spec section 2.1/G4 -- abating commanded
generation, aborting on uncommanded) is out of scope here: ``emit_re_generics``
simply skips a constraint whose ``thermal_generation`` term names a thermal
outside ``id_map.thermal_codes`` (an anticipation-track/GNL thermal), with a
``WARNING``, deferring coordination with the GNL-ring track.

``RHV``'s volume tipos (``VDEF``/``VDES``/``VBOM``) are deferred rather than
lowered: their hm³→flow conversion needs a per-stage coefficient (block
hours vary per stage) that a stage-invariant novomodelo expression cannot carry,
so ``emit_rhq_rhv_generics`` detects one and skips the constraint with a
``WARNING`` instead of guessing a coefficient (see the E5 scalar-parameter
plumbing).

``HE`` (RHE stored-energy) constraints are emitted by
``emit_rhe_generics``: unlike the three families above, it
owns its own productivity reconstruction (the shared
``productivity.integrated_productivity``/``compute_productivity``
primitives, branched on the DECOMP reservoir predicate and driven over the
operated-cascade walk this module imports as
:func:`_downstream_operated`) rather than reading a bounds axis off the
register directly — the RHS is a
whole-REE stored-energy sum, not a per-plant limit, so it needs the
per-stage accumulated productivity to convert reservoir storage to energy.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, NamedTuple

from novomodelo_bridge.core.diagnostics import Diagnostic, Severity, emit
from novomodelo_bridge.core.generic_constraint_builder import (
    ConstraintIdAllocator,
    GenericConstraintBuilder,
    GenericConstraintResult,
    is_bounded,
)
from novomodelo_bridge.core.productivity import (
    compute_productivity,
    integrated_productivity,
)
from novomodelo_bridge.decomp.constraint_registers import StageBounds
from novomodelo_bridge.decomp.converters.cadastro import (
    effective_storage_range,
    is_reservoir,
)
from novomodelo_bridge.decomp.converters.hydro import _downstream_operated
from novomodelo_bridge.decomp.converters.scalar_parameters import rho_acum_name

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from novomodelo_bridge.core.generic_constraint_builder import Slot
    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.constraint_registers import (
        ConstraintCensus,
        ConstraintRecord,
        ConstraintTerm,
    )
    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
    from novomodelo_bridge.decomp.id_map import DecompIdMap
    from novomodelo_bridge.decomp.temporal import OperativeStage


#: The source model's ``num_max_iteracoes``-style deficit multiplier: the
#: slack penalty on a generic constraint is this many times the worst bus
#: deficit cost, so violating the constraint is always far more expensive
#: than shedding load but never numerically unbounded.
_BIG_M_DEFICIT_FACTOR = 10.0


def big_m_penalty(max_deficit: float) -> float:
    """The slack penalty for a generic constraint: ``10 x`` the worst deficit cost.

    The pipeline computes ``max_deficit`` from the deck's per-bus deficit
    costs and passes the result here — this module never reads the deck
    itself.
    """
    return _BIG_M_DEFICIT_FACTOR * max_deficit


def _hydro_generation_token(hydro_id: int, bus_id: int | None = None) -> str:
    """The novomodelo ``hydro_generation`` token, optionally split by frequency bus.

    ``bus_id`` selects the v0.14 frequency-split generation selector
    (``hydro_generation(id, bus=bus_id)``); omitted, it is the plant's whole
    generation (``hydro_generation(id)``). Frequency-split resolution itself
    is deferred to the multi-term RE emitter.
    """
    if bus_id is not None:
        return f"hydro_generation({hydro_id}, bus={bus_id})"
    return f"hydro_generation({hydro_id})"


#: Flow/volume/generation variables that map straight through
#: ``id_map.hydro_id`` to a single-argument novomodelo hydro token. Kept explicit
#: and symmetric with the reader's ``constraint_registers._BOUNDS_AXIS`` and
#: ``single_term_bounds._HQ_AXIS_BY_VARIABLE`` variable taxonomies.
_HYDRO_TOKEN_FN: dict[str, str] = {
    "QDEF": "hydro_outflow",
    "QTUR": "hydro_turbined",
    "QVER": "hydro_spillage",
    "QDES": "hydro_diversion",
    "VARM": "hydro_storage",
}

#: Variables ``_variable_token`` refuses to resolve itself, mapped to the
#: mechanism that resolves them instead.
_DEFERRED_VARIABLES: dict[str, str] = {
    "interchange": "the interchange (FI line) resolution path",
    "VDEF": "the hours-weighted flow-domain path",
    "VDES": "the hours-weighted flow-domain path",
    "VBOM": "the hours-weighted flow-domain path",
}


def _variable_token(
    term: ConstraintTerm,
    id_map: DecompIdMap,
    pumping_station_ids: Mapping[int, int],
) -> str:
    """The novomodelo expression token for one constraint term.

    Dispatches on ``term.variable``: ``generation``/``thermal_generation``
    map to the whole-plant generation tokens; ``QDEF``/``QTUR``/``QVER``/
    ``QDES``/``VARM`` map through :data:`_HYDRO_TOKEN_FN` and
    ``id_map.hydro_id``; ``QBOM`` maps through *pumping_station_ids*.

    Raises
    ------
    ValueError
        For ``interchange`` (resolved by ``resolve_fi_term``, which needs
        the line map), the volume tipos ``VDEF``/``VDES``/``VBOM``
        (resolved by the hours-weighted flow-domain path), or any other
        unrecognised variable — fail-loud, never a silent skip
        that would build a malformed expression.
    KeyError
        Propagated from ``id_map.hydro_id``/``id_map.thermal_id`` for an
        unknown plant/thermal code, or from *pumping_station_ids* for an
        unknown ``QBOM`` code.
    """
    variable = term.variable
    if variable == "generation":
        return _hydro_generation_token(id_map.hydro_id(term.code))
    if variable == "thermal_generation":
        return f"thermal_generation({id_map.thermal_id(term.code)})"
    if variable == "QBOM":
        return f"pumping_flow({pumping_station_ids[term.code]})"
    token_fn = _HYDRO_TOKEN_FN.get(variable)
    if token_fn is not None:
        return f"{token_fn}({id_map.hydro_id(term.code)})"
    owner = _DEFERRED_VARIABLES.get(variable)
    if owner is not None:
        raise ValueError(
            f"Cannot build a constraint term: {variable!r} is resolved by "
            f"{owner}, not this dispatcher."
        )
    raise ValueError(f"Cannot build a constraint term: unknown variable {variable!r}.")


def _line_int_field(line: Mapping[str, object], field: str) -> int:
    """Extract *field* from a converted line dict, validating it is an ``int``.

    ``convert_lines``'s own ``lines.json`` list hands back untyped
    ``dict[str, object]`` entries; this narrows the three id fields with an
    explicit runtime check (mirroring ``fcf.mapper._slot_int``) rather than a
    bare ``int()`` cast, so a malformed line entry fails loudly here instead
    of silently corrupting the map.
    """
    value = line[field]
    if not isinstance(value, int):
        raise TypeError(
            f"line field {field!r} is {type(value).__name__}, not int: {value!r}"
        )
    return value


def build_fi_line_map(
    lines: Sequence[Mapping[str, object]],
) -> dict[tuple[int, int], int]:
    """Map each converted line's declared ``(source, target)`` bus pair to its id.

    ``lines`` is the ``convert_lines`` ``lines.json`` ``lines`` list — one
    entry per ``IA`` line, in its ``IA``-declared orientation (**not**
    canonicalized). Keying on that orientation, rather than a
    sorted/canonical pair, is what lets :func:`resolve_fi_term` tell an
    ``FI`` term's direct direction from its reverse via a two-probe lookup.
    """
    return {
        (
            _line_int_field(line, "source_bus_id"),
            _line_int_field(line, "target_bus_id"),
        ): _line_int_field(line, "id")
        for line in lines
    }


def _emit_fi_no_line(term: ConstraintTerm, *, reason: str) -> None:
    """Emit the shared ``decomp-fi-no-line`` WARNING for an unresolved FI term."""
    emit(
        Diagnostic(
            code="decomp-fi-no-line",
            severity=Severity.WARNING,
            category="Special constraints",
            title="FI interchange term has no matching line",
            summary=(
                f"RE constraint FI interchange term ({term.submarket_de!r} -> "
                f"{term.submarket_para!r}) could not be resolved to a novomodelo "
                f"line: {reason}."
            ),
            remediation=(
                "Check that the pair is declared as an IA line; the "
                "constraint carrying this FI term is skipped."
            ),
        )
    )


def resolve_fi_term(
    term: ConstraintTerm,
    id_map: DecompIdMap,
    line_map: Mapping[tuple[int, int], int],
) -> str | None:
    """Resolve one ``FI`` interchange term to a directional novomodelo line token.

    novomodelo has no submarket-pair variable, only per-line directional
    ``line_direct(id)``/``line_reverse(id)`` variables, so — mirroring the
    source model precedent's ``ener_interc`` resolution
    (``converters/constraints.py::_parse_formula``) — the term's direction is
    encoded by *which* directional variable is chosen, never by negating the
    coefficient (a negated coefficient would dilute the bound when the LP
    routes the non-canonical direction).

    Resolves both submarket names to bus ids via ``id_map.bus_id_by_name``
    (the reconciled ``SB`` name -> id map, which also handles the fictitious
    ``IV`` transhipment submarket) and probes *line_map* in both
    orientations, without canonicalizing the pair: ``(de, para)`` matching a
    line's declared ``source_bus_id``/``target_bus_id`` returns
    ``"line_direct(id)"`` (the term's orientation matches the line's); the
    reverse match returns ``"line_reverse(id)"``.

    Returns ``None`` — after emitting one ``Severity.WARNING``
    ``"decomp-fi-no-line"`` diagnostic — when either submarket name is
    unknown (a caught ``KeyError``) or no line joins the pair in either
    orientation. Raises ``ValueError`` if *term* is missing a submarket name
    (a caller contract violation: a genuine ``FI`` term, as produced by the
    reader, always carries both). Otherwise never raises: the caller drops
    the whole constraint on a ``None`` rather than build a partial
    expression.
    """
    if term.submarket_de is None or term.submarket_para is None:
        raise ValueError(
            "Cannot resolve an interchange term: it must carry both "
            f"submarket_de and submarket_para, got {term!r}."
        )
    try:
        de = id_map.bus_id_by_name(term.submarket_de)
        para = id_map.bus_id_by_name(term.submarket_para)
    except KeyError as exc:
        reason = str(exc.args[0]) if exc.args else str(exc)
        _emit_fi_no_line(term, reason=reason)
        return None
    if (de, para) in line_map:
        return f"line_direct({line_map[(de, para)]})"
    if (para, de) in line_map:
        return f"line_reverse({line_map[(para, de)]})"
    _emit_fi_no_line(
        term,
        reason=f"no IA line joins {term.submarket_de!r} and {term.submarket_para!r}",
    )
    return None


def _format_expression(terms: Sequence[tuple[float, str]]) -> str | None:
    """Join ``(coefficient, token)`` pairs into a novomodelo expression string.

    Mirrors ``converters/constraints.py::_parse_formula``'s coefficient-
    formatting rules: a unit (``|coeff| == 1.0``) term is the bare token
    (or ``- token`` when negative); a fractional term is ``{abs_coeff} *
    {token}`` (or ``- {abs_coeff} * {token}`` when negative). The first term
    omits the leading ``+``; every later term uses ``+``/``-`` as a binary
    operator. Returns ``None`` for an empty term list.
    """
    if not terms:
        return None
    parts: list[str] = []
    for i, (coeff, token) in enumerate(terms):
        abs_coeff = abs(coeff)
        is_negative = coeff < 0.0
        body = token if abs_coeff == 1.0 else f"{abs_coeff} * {token}"
        if i == 0:
            parts.append(f"- {body}" if is_negative else body)
        else:
            parts.append(f"- {body}" if is_negative else f"+ {body}")
    return " ".join(parts)


def slots_from_record(
    record: ConstraintRecord, calendar: Sequence[OperativeStage]
) -> list[Slot]:
    """Enumerate the exact (stage, block) slots *record*'s bounds cover.

    For a per-block record, one slot per
    ``range(min(len(stage_bounds.lower), len(calendar[stage].block_hours)))``
    block per declared stage (mirroring
    ``single_term_bounds._per_block_contributions``'s clamp — a bounded slot
    beyond the stage's real block count is dropped, never emitted as an
    orphan row); for a stage-level record (``record.per_block is False``),
    one slot per declared stage with ``block_id=None``. Feeds
    :meth:`~novomodelo_bridge.core.generic_constraint_builder.GenericConstraintBuilder.add`,
    which itself filters to slots bounded on either side.
    """
    slots: list[Slot] = []
    for stage_index, stage_bounds in record.bounds.items():
        if record.per_block:
            n_slots = min(
                len(stage_bounds.lower), len(calendar[stage_index].block_hours)
            )
            slots.extend(
                (stage_index, b, stage_bounds.lower[b], stage_bounds.upper[b])
                for b in range(n_slots)
            )
        else:
            slots.append(
                (stage_index, None, stage_bounds.lower[0], stage_bounds.upper[0])
            )
    return slots


def _emit_re_frequency_split_deferred(term: ConstraintTerm) -> None:
    """Emit the shared ``decomp-re-frequency-split-deferred`` WARNING.

    Resolving a frequency-split ``FU`` term (Itaipu 50/60 Hz) needs a
    frequency-to-bus map this conversion does not build; the caller drops
    the whole constraint rather than guess the bus.
    """
    emit(
        Diagnostic(
            code="decomp-re-frequency-split-deferred",
            severity=Severity.WARNING,
            category="Special constraints",
            title="RE frequency-split generation term deferred",
            summary=(
                "RE generic constraint carries a hydro generation term "
                f"(plant code {term.code}) split by frequency "
                f"({term.frequency} Hz), which needs a frequency-to-bus map "
                "this conversion does not build; the constraint is skipped."
            ),
            remediation=(
                "Frequency-split (RI-to-bus) resolution is deferred; no "
                "action needed to convert."
            ),
        )
    )


def _emit_re_thermal_not_normal(term: ConstraintTerm) -> None:
    """Emit the shared ``decomp-re-thermal-not-normal`` WARNING, naming the code."""
    emit(
        Diagnostic(
            code="decomp-re-thermal-not-normal",
            severity=Severity.WARNING,
            category="Special constraints",
            title="RE thermal generation term is not a normal thermal",
            summary=(
                "RE generic constraint carries a thermal generation term "
                f"for thermal code {term.code}, which is not declared as a "
                "normal thermal (an anticipation-track/GNL thermal); the "
                "constraint is skipped."
            ),
            remediation=(
                "GNL coordination for RE constraints is deferred; no action "
                "needed to convert."
            ),
        )
    )


def _resolve_re_terms(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    line_map: Mapping[tuple[int, int], int],
) -> list[tuple[float, str]] | None:
    """Resolve every term of one generic RE record to ``(coefficient, token)`` pairs.

    Dispatches on ``term.variable``:

    - ``generation`` with ``frequency is not None`` — a frequency-split
      term with no RI-to-bus map available — emits
      :func:`_emit_re_frequency_split_deferred` and drops the whole record.
    - ``generation`` with ``frequency is None`` — resolved via
      ``_variable_token``.
    - ``thermal_generation`` whose ``code`` is absent from
      ``id_map.thermal_codes`` — a GNL/anticipation-track thermal not yet a
      normal thermal entity — emits :func:`_emit_re_thermal_not_normal` and
      drops the whole record; the membership check runs *before*
      ``_variable_token`` so a GNL code never reaches ``id_map.thermal_id``
      (which would raise ``KeyError``). Otherwise resolved via
      ``_variable_token``.
    - ``interchange`` — resolved via ``resolve_fi_term``; a ``None`` result
      (the resolver already warned) drops the whole record.

    Any single unresolved term drops the **whole** constraint
    (skip-not-partial) — never a partial expression. Returns the ordered
    ``(coefficient, token)`` list when every term resolves.
    """
    resolved: list[tuple[float, str]] = []
    for term in record.terms:
        if term.variable == "generation":
            if term.frequency is not None:
                _emit_re_frequency_split_deferred(term)
                return None
            resolved.append((term.coefficient, _variable_token(term, id_map, {})))
        elif term.variable == "thermal_generation":
            if term.code not in id_map.thermal_codes:
                _emit_re_thermal_not_normal(term)
                return None
            resolved.append((term.coefficient, _variable_token(term, id_map, {})))
        elif term.variable == "interchange":
            token = resolve_fi_term(term, id_map, line_map)
            if token is None:
                return None
            resolved.append((term.coefficient, token))
        else:
            raise ValueError(
                f"_resolve_re_terms: unexpected RE term variable {term.variable!r}"
            )
    return resolved


def emit_re_generics(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    census: ConstraintCensus,
    line_map: Mapping[tuple[int, int], int],
    big_m: float,
    allocator: ConstraintIdAllocator | None = None,
) -> GenericConstraintResult | None:
    """Emit every generic ``RE`` constraint (multi-term hydro/thermal/interchange).

    Iterates ``census.to_generic`` filtered to ``record.family == "RE"``,
    resolves each record's terms via :func:`_resolve_re_terms`, and feeds a
    :class:`~novomodelo_bridge.core.generic_constraint_builder.GenericConstraintBuilder`;
    a record whose terms cannot all be resolved is dropped entirely
    (skip-not-partial, one diagnostic per skipped constraint — see
    :func:`_resolve_re_terms`). Single-hydro-generation RE records never
    reach here — they lower to a plant bound in ``single_term_bounds.py``
    (``lowers_to_bound``). ``HQ``/``HV`` records also present in
    ``census.to_generic`` are :func:`emit_rhq_rhv_generics`'s, not this
    function's; ``HE`` records are unhandled (deferred).

    Reads no ``dadger``; *big_m* and *line_map* arrive from the caller (E7).
    *allocator* defaults to a fresh :class:`ConstraintIdAllocator` (ids from
    0) when omitted; the pipeline threads one shared instance across every
    emitter. Returns ``builder.result()`` — ``None`` when no RE generic
    constraint survives.
    """
    calendar = case.calendar
    allocator = allocator or ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)
    for record in census.to_generic:
        if record.family != "RE":
            continue
        terms = _resolve_re_terms(record, id_map, line_map)
        if terms is None:
            continue
        expression = _format_expression(terms)
        if expression is None:
            continue
        builder.add(
            name=f"RE_{record.constraint_id}",
            description=f"RE generic constraint {record.constraint_id}",
            expression=expression,
            slack={"enabled": True, "penalty": big_m},
            slots=slots_from_record(record, calendar),
        )
    return builder.result()


#: RHV ``CV.tipo`` volume variables whose hm³→flow conversion needs a
#: per-stage coefficient (block hours vary per stage) that a stage-invariant
#: novomodelo expression cannot carry — deferred to the E5 scalar-parameter
#: plumbing rather than emitted with a guessed coefficient.
_RHV_VOLUME_TIPOS = frozenset({"VDEF", "VDES", "VBOM"})


def _emit_rhq_qbom_no_station(record: ConstraintRecord, code: int) -> None:
    """Emit the shared ``decomp-rhq-qbom-no-station`` WARNING, naming the code."""
    emit(
        Diagnostic(
            code="decomp-rhq-qbom-no-station",
            severity=Severity.WARNING,
            category="Special constraints",
            title="RHQ QBOM term has no matching pumping station",
            summary=(
                f"HQ constraint {record.constraint_id} carries a QBOM term "
                f"for plant code {code}, which has no matching pumping "
                "station; the constraint is skipped."
            ),
            remediation=(
                f"HQ constraint {record.constraint_id} is skipped; check "
                f"that plant code {code} is declared as a pumping station."
            ),
        )
    )


def _resolve_rhq_terms(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    pumping_station_ids: Mapping[int, int],
) -> list[tuple[float, str]] | None:
    """Resolve every term of one generic RHQ record to ``(coefficient, token)`` pairs.

    Every flow variable resolves via :func:`_variable_token`. A ``QBOM`` term
    whose ``code`` has no matching entry in *pumping_station_ids* — it is a
    pumping-station code, never a hydro code, so it cannot fall through to
    ``id_map.hydro_id`` — emits :func:`_emit_rhq_qbom_no_station` and drops
    the whole record (skip-not-partial), checked *before* the
    ``_variable_token`` call so a missing station never raises a raw
    ``KeyError``. Returns the ordered ``(coefficient, token)`` list when
    every term resolves.
    """
    resolved: list[tuple[float, str]] = []
    for term in record.terms:
        if term.variable == "QBOM" and term.code not in pumping_station_ids:
            _emit_rhq_qbom_no_station(record, term.code)
            return None
        resolved.append(
            (term.coefficient, _variable_token(term, id_map, pumping_station_ids))
        )
    return resolved


def _emit_rhv_varm_uncadastred(record: ConstraintRecord, code: int) -> None:
    """Emit the shared ``decomp-rhv-varm-plant-uncadastred`` WARNING, naming the code.

    Mirrors ``single_term_bounds._hv_storage_contributions``'s single-``VARM``
    guard, reusing the same diagnostic code for the multi-``VARM`` case.
    """
    emit(
        Diagnostic(
            code="decomp-rhv-varm-plant-uncadastred",
            severity=Severity.WARNING,
            category="Special constraints",
            title="RHV VARM constraint references an uncadastred plant",
            summary=(
                f"HV constraint {record.constraint_id} bounds the stored "
                f"volume of plant code {code}, which has no cadastro row; "
                "the constraint cannot be lowered to a generic constraint."
            ),
            remediation=(
                f"HV constraint {record.constraint_id} is skipped; check "
                f"that plant code {code} is cadastred and operated."
            ),
        )
    )


def _emit_rhv_volume_tipo_deferred(record: ConstraintRecord, tipo: str) -> None:
    """Emit the shared ``decomp-rhv-volume-tipo-deferred`` WARNING, naming the tipo."""
    emit(
        Diagnostic(
            code="decomp-rhv-volume-tipo-deferred",
            severity=Severity.WARNING,
            category="Special constraints",
            title="RHV volume tipo term deferred",
            summary=(
                f"HV constraint {record.constraint_id} carries a {tipo} "
                "volume term; its hm3-to-flow form needs a per-stage "
                "coefficient (block hours vary per stage) that a "
                "stage-invariant novomodelo expression cannot carry, so the "
                "constraint is skipped rather than emitted with a guessed "
                "coefficient."
            ),
            remediation=(
                "This volume-based RHV limit needs per-stage coefficients that "
                "are not yet supported, so the constraint is skipped; no action "
                "needed to convert."
            ),
        )
    )


def _offset_if_bounded(value: float | None, offset: float) -> float | None:
    """Add *offset* to *value* iff it is a real bound; otherwise pass it through.

    Mirrors :func:`~novomodelo_bridge.core.generic_constraint_builder.is_bounded`'s
    sentinel check, phrased so mypy narrows *value* to ``float`` on the
    addition: a ``None`` or ``±1e21`` side never gets an offset added to it.
    """
    if value is None or not is_bounded(value):
        return value
    return value + offset


def _resolve_hv_varm(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    effective: EffectiveCadastro,
    calendar: Sequence[OperativeStage],
) -> tuple[list[tuple[float, str]], Mapping[int, StageBounds]] | None:
    """Resolve one generic multi-``VARM`` RHV record's expression and floor-adjusted
    bounds.

    Every term resolves via :func:`_variable_token` to a ``hydro_storage``
    token; a term whose ``code`` is absent from ``effective.base.index``
    emits :func:`_emit_rhv_varm_uncadastred` and drops the whole record
    (skip-not-partial) before any bounds are adjusted.

    The source model's ``LV`` limits are relative to each plant's own useful
    volume, so the absolute RHS is the **per-term additive floor**: for each
    declared stage, ``offset = sum(cᵢ * effective_storage_range(effective,
    codeᵢ, stage)[0] for term i)``, added to whichever side of the record's
    stage-level ``StageBounds`` is actually bounded
    (:func:`~novomodelo_bridge.core.generic_constraint_builder.is_bounded`) —
    a ``±1e21``/``None`` side is left untouched, never offset. *calendar* is
    accepted for signature symmetry with the per-block resolvers (mirroring
    ``single_term_bounds._hv_storage_contributions``); ``VARM`` is
    stage-level, so it is not read here.

    Returns the ``(coefficient, token)`` list and the floor-adjusted
    ``{stage_index: StageBounds}`` map when every term resolves.
    """
    resolved: list[tuple[float, str]] = []
    coded_terms: list[tuple[float, int]] = []
    for term in record.terms:
        if term.code not in effective.base.index:
            _emit_rhv_varm_uncadastred(record, term.code)
            return None
        resolved.append((term.coefficient, _variable_token(term, id_map, {})))
        coded_terms.append((term.coefficient, term.code))

    adjusted: dict[int, StageBounds] = {}
    for stage_index, stage_bounds in record.bounds.items():
        offset = sum(
            coefficient * effective_storage_range(effective, code, stage_index)[0]
            for coefficient, code in coded_terms
        )
        adjusted[stage_index] = StageBounds(
            lower=(_offset_if_bounded(stage_bounds.lower[0], offset),),
            upper=(_offset_if_bounded(stage_bounds.upper[0], offset),),
        )
    return resolved, adjusted


def emit_rhq_rhv_generics(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    census: ConstraintCensus,
    pumping_station_ids: Mapping[int, int],
    effective: EffectiveCadastro,
    big_m: float,
    allocator: ConstraintIdAllocator | None = None,
) -> GenericConstraintResult | None:
    """Emit every generic ``RHQ``/``RHV`` constraint (flow mixes and multi-``VARM``).

    Iterates ``census.to_generic`` filtered to ``record.family in ("HQ",
    "HV")``:

    - ``"HQ"`` (per-block flow terms) resolves via :func:`_resolve_rhq_terms`
      and feeds a two-sided ``HQ_<id>`` constraint.
    - ``"HV"`` carrying any volume-tipo term (``VDEF``/``VDES``/``VBOM``)
      emits :func:`_emit_rhv_volume_tipo_deferred` and is skipped (deferred
      to the E5 scalar-parameter plumbing — see the module docstring).
    - ``"HV"`` all-``VARM`` (stage-level) resolves via
      :func:`_resolve_hv_varm` and feeds a two-sided ``HV_<id>`` constraint
      on ``dataclasses.replace(record, bounds=adjusted)`` — the floor-
      adjusted bounds, never the reader's original record (a frozen
      dataclass is never mutated in place).

    Any record whose terms cannot all be resolved is dropped entirely
    (skip-not-partial, one diagnostic per skipped constraint). Single-term
    RHQ/RHV records on a bounded variable never reach here — they lower to
    an entity bound in ``single_term_bounds.py`` (``lowers_to_bound``).

    Reads no ``dadger``; *pumping_station_ids*, *effective*, and *big_m*
    arrive from the caller (E7). *allocator* defaults to a fresh
    :class:`ConstraintIdAllocator` (ids from 0) when omitted; the pipeline
    threads one shared instance across every emitter. Returns
    ``builder.result()`` — ``None`` when no RHQ/RHV generic constraint
    survives.
    """
    calendar = case.calendar
    allocator = allocator or ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)
    for record in census.to_generic:
        if record.family not in ("HQ", "HV"):
            continue

        if record.family == "HQ":
            rhq_terms = _resolve_rhq_terms(record, id_map, pumping_station_ids)
            if rhq_terms is None:
                continue
            expression = _format_expression(rhq_terms)
            if expression is None:
                continue
            builder.add(
                name=f"HQ_{record.constraint_id}",
                description=f"RHQ generic constraint {record.constraint_id}",
                expression=expression,
                slack={"enabled": True, "penalty": big_m},
                slots=slots_from_record(record, calendar),
            )
            continue

        volume_tipo = next(
            (t.variable for t in record.terms if t.variable in _RHV_VOLUME_TIPOS),
            None,
        )
        if volume_tipo is not None:
            _emit_rhv_volume_tipo_deferred(record, volume_tipo)
            continue

        hv_resolved = _resolve_hv_varm(record, id_map, effective, calendar)
        if hv_resolved is None:
            continue
        hv_terms, adjusted_bounds = hv_resolved
        expression = _format_expression(hv_terms)
        if expression is None:
            continue
        adjusted_record = dataclasses.replace(record, bounds=adjusted_bounds)
        builder.add(
            name=f"HV_{record.constraint_id}",
            description=f"RHV generic constraint {record.constraint_id}",
            expression=expression,
            slack={"enabled": True, "penalty": big_m},
            slots=slots_from_record(adjusted_record, calendar),
        )
    return builder.result()


#: The slack penalty (R$/MWh) an RHE constraint falls back to when its own
#: ``valor_penalidade`` is missing or non-positive — mirrors the source
#: model's VminOP ``penalty <= 0 -> 1000.0`` fallback
#: (``converters/constraints.py::convert_vminop_constraints``).
_RHE_DEFAULT_PENALTY = 1000.0

#: Unit-conversion constants for the DECOMP stage-hours energy factor
#: (:func:`_rhe_energy_factor`): seconds per hour and cubic meters per cubic
#: hectometer.
_SECONDS_PER_HOUR = 3600.0
_M3_PER_HM3 = 1_000_000.0


class RheResult(NamedTuple):
    """Result of :func:`emit_rhe_generics`.

    ``result`` is the usual :class:`GenericConstraintResult` (``None`` when
    no RHE constraint survives). ``rho_acum_overrides`` maps every novomodelo
    hydro id *actually referenced* by a surviving RHE expression to its
    per-stage integrated ρ_acum in MWmês/hm³ — the override contract
    ``decomp.converters.scalar_parameters.build_decomp_scalar_parameters`` consumes so
    the LP's ``@rho_acum_h{id}`` coefficient matches the RHS this emitter
    computes. Empty when ``result`` is ``None``.
    """

    result: GenericConstraintResult | None
    rho_acum_overrides: dict[int, list[float]]


def _is_stored_energy_reservoir(effective: EffectiveCadastro, code: int) -> bool:
    """True iff DECOMP counts plant *code*'s storage in a REE's stored energy.

    A reservoir under the DECOMP predicate (:func:`~novomodelo_bridge.decomp.
    converters.cadastro.is_reservoir`: ``"M"`` or ``"S"``) with usable
    storage (``volume_maximo > volume_minimo``), read off the *base*
    cadastro. Run-of-river (``"D"``) plants are excluded even when their
    accumulated cascade productivity is positive.
    """
    if not is_reservoir(effective, code):
        return False
    vol_min = float(effective.base.loc[code, "volume_minimo"])
    vol_max = float(effective.base.loc[code, "volume_maximo"])
    return vol_max - vol_min > 0.0


def _rhe_energy_factor(stage: OperativeStage) -> float:
    """The DECOMP stage-hours ``ρ_acum·hm³ -> MWmês`` divisor for *stage*.

    ``_SECONDS_PER_HOUR * stage.total_hours / _M3_PER_HM3``, built from the
    stage's *real* hours (168 for a weekly stage, ~730 for a monthly one) —
    **never** a fixed month length and **never** the source model's
    calendar-month ``_vminop_energy_factor``.

    For a **percentage** limit (``tipo_limite == 2``) the factor cancels
    between the RHE expression's LHS (the ``@rho_acum_h{id}`` coefficient) and
    this module's RHS (``ρ_acum_energy·volume_maximo``), so there it only sets
    the slack's energy units (and thus the effective R$/MWh penalty). For an
    **absolute** limit (``tipo_limite == 1``) it does **not** cancel: it is a
    load-bearing ``MW/(m³/s) -> MWmês/hm³`` conversion that fixes the binding
    storage threshold (``storage_bind = limite·factor / Σ acc``), so it must
    not be dropped.
    """
    return _SECONDS_PER_HOUR * stage.total_hours / _M3_PER_HM3


def _per_stage_own_integrated_rho(
    effective: EffectiveCadastro, code: int, n_stages: int
) -> list[float]:
    """Per-stage own stored-energy productivity (MW/(m³/s)) for plant *code*.

    Builds a per-stage ``hidr``-shaped row from *effective* — copying
    ``effective.base.loc[code]`` and overwriting the five
    ``a{i}_volume_cota`` coefficients
    (:meth:`~novomodelo_bridge.decomp.converters.cadastro.effective.
    EffectiveCadastro.cota_polynomial`)
    and ``canal_fuga_medio``/``volume_minimo``/``volume_maximo``/
    ``volume_referencia`` (:meth:`~novomodelo_bridge.decomp.converters.cadastro.
    effective.EffectiveCadastro.value`) — then evaluates the volume-integrated
    EARM ρ (:func:`~novomodelo_bridge.core.productivity.integrated_productivity`)
    for a DECOMP reservoir (``"M"`` or ``"S"``) or the point ρ at
    ``volume_referencia``
    (:func:`~novomodelo_bridge.core.productivity.compute_productivity`) for a
    run-of-river ``"D"`` plant. This deliberately differs from the source
    model's ``stored_energy_productivity``, which integrates only ``"M"``.
    A plant with no per-stage override on any of these falls through to the
    base row at every stage, so this collapses to a stage-invariant series —
    the common case.
    """
    reservoir = is_reservoir(effective, code)
    values: list[float] = []
    for stage_index in range(n_stages):
        hreg = effective.base.loc[code].copy()
        coeffs = effective.cota_polynomial(code, stage_index)
        for i in range(5):
            hreg[f"a{i}_volume_cota"] = coeffs[i]
        hreg["canal_fuga_medio"] = effective.value(
            code, "canal_fuga_medio", stage_index
        )
        hreg["volume_minimo"] = effective.value(code, "volume_minimo", stage_index)
        hreg["volume_maximo"] = effective.value(code, "volume_maximo", stage_index)
        hreg["volume_referencia"] = effective.value(
            code, "volume_referencia", stage_index
        )
        values.append(
            integrated_productivity(hreg) if reservoir else compute_productivity(hreg)
        )
    return values


def _per_stage_rho_acum_energy(
    effective: EffectiveCadastro,
    operated: set[int],
    calendar: Sequence[OperativeStage],
) -> dict[int, list[float]]:
    """Per-stage cascade-summed ρ_acum (MWmês/hm³) for every operated hydro.

    For each stage, builds the stage-representative operated-cascade
    topology (``_downstream_operated(effective, code, operated,
    stage_index=...)``, imported from
    :mod:`novomodelo_bridge.decomp.converters.hydro`) and
    topologically accumulates ``acc[code] = own[code][s] + (acc[downstream]
    if downstream is not None else 0.0)`` — a memoized DAG walk mirroring
    the source model's ``_cascade_sum``/``compute_per_stage_acc_productivities`` —
    over **every** operated hydro, not only REE members (a member's ρ_acum
    includes downstream own values from non-member plants). Each stage's
    accumulated sum is then divided by :func:`_rhe_energy_factor` to convert
    MW/(m³/s) to MWmês/hm³.
    """
    n_stages = len(calendar)
    own: dict[int, list[float]] = {
        code: _per_stage_own_integrated_rho(effective, code, n_stages)
        for code in operated
    }
    result: dict[int, list[float]] = {code: [0.0] * n_stages for code in operated}
    for stage in calendar:
        s = stage.index
        downstream_at_s = {
            code: _downstream_operated(effective, code, operated, stage_index=s)
            for code in operated
        }
        acc: dict[int, float] = {}

        def _accumulate(code: int) -> float:
            if code in acc:
                return acc[code]
            downstream = downstream_at_s.get(code)
            downstream_acc = _accumulate(downstream) if downstream is not None else 0.0
            acc[code] = own[code][s] + downstream_acc
            return acc[code]

        for code in operated:
            _accumulate(code)
        factor = _rhe_energy_factor(stage)
        for code in operated:
            result[code][s] = acc[code] / factor
    return result


def _emit_rhe_no_storage_plants(record: ConstraintRecord) -> None:
    """Emit the shared ``decomp-rhe-no-storage-plants`` WARNING.

    Mirrors ``_emit_rhv_varm_uncadastred``'s skip-not-partial pattern: an HE
    record whose REE(s) yield no stored-energy reservoir at all contributes
    no constraint (not a partial one).
    """
    emit(
        Diagnostic(
            code="decomp-rhe-no-storage-plants",
            severity=Severity.WARNING,
            category="Special constraints",
            title="RHE constraint's REE has no stored-energy reservoir",
            summary=(
                f"HE constraint {record.constraint_id} bounds the stored "
                "energy of a REE with no monthly-regulating ('M') "
                "reservoir carrying usable storage; the constraint cannot "
                "be lowered to a generic constraint."
            ),
            remediation=(
                f"HE constraint {record.constraint_id} is skipped; check "
                "that its REE(s) have at least one cadastred stored-energy "
                "reservoir mapped via the UH register."
            ),
        )
    )


def _emit_rhe_default_penalty(record: ConstraintRecord, penalty: float | None) -> None:
    """Emit the shared ``decomp-rhe-default-penalty`` WARNING, naming the record."""
    emit(
        Diagnostic(
            code="decomp-rhe-default-penalty",
            severity=Severity.WARNING,
            category="Special constraints",
            title="RHE constraint has no positive penalty",
            summary=(
                f"HE constraint {record.constraint_id} declares a "
                f"valor_penalidade of {penalty!r} (must be > 0); using the "
                f"default {_RHE_DEFAULT_PENALTY}."
            ),
            remediation=(
                "Check the HE register's valor_penalidade column for "
                f"constraint {record.constraint_id}."
            ),
        )
    )


def _emit_rhe_unknown_tipo_limite(
    record: ConstraintRecord, tipo_limite: int | None
) -> None:
    """Emit the shared ``decomp-rhe-unknown-tipo-limite`` WARNING, naming the record."""
    emit(
        Diagnostic(
            code="decomp-rhe-unknown-tipo-limite",
            severity=Severity.WARNING,
            category="Special constraints",
            title="RHE constraint has an unrecognized tipo_limite",
            summary=(
                f"HE constraint {record.constraint_id} declares "
                f"tipo_limite={tipo_limite!r} (expected 1 or 2); treating "
                "the limit as absolute MWmes."
            ),
            remediation=(
                "Check the HE register's tipo_limite column for constraint "
                f"{record.constraint_id}."
            ),
        )
    )


def emit_rhe_generics(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    census: ConstraintCensus,
    effective: EffectiveCadastro,
    hydro_to_ree: Mapping[int, int],
    allocator: ConstraintIdAllocator | None = None,
) -> RheResult:
    """Emit every generic ``HE`` (RHE stored-energy) constraint.

    Iterates ``census.to_generic`` filtered to ``record.family == "HE"``.
    Each record's ``energy`` terms name a REE (``term.code``) with a CM ±1
    sign (``term.coefficient``); the participating plants for that REE are
    its *hydro_to_ree* members filtered to stored-energy reservoirs
    (:func:`_is_stored_energy_reservoir`) and to codes resolvable via
    ``id_map.hydro_id`` (a member absent from the id map is skipped, not
    fatal). A record whose terms yield **no** participating reservoir at
    all is skipped whole (:func:`_emit_rhe_no_storage_plants`,
    skip-not-partial) — never a partial constraint.

    The expression sums ``@rho_acum_h{id} * hydro_storage(id)`` over every
    participating reservoir, signed by its REE's CM coefficient
    (:func:`_format_expression`); the RHS is absolute stored energy
    directly — **no floor offset** (unlike the E4 RHV emitter's ``VARM``
    floor: ``hydro_storage`` is already the absolute stage-final storage) —
    computed per the record's ``tipo_limite``: ``1`` treats ``limite`` as an
    absolute MWmês bound directly; ``2`` treats it as a percentage of the
    participating reservoirs' summed ``ρ_acum_energy · volume_maximo`` (the
    manual §2.4 form — deliberately *not* the source model's VminOP
    ``pct·useful + dead`` form); any other value is treated as absolute
    with a WARNING. The slack penalty is ``record.he_meta.valor_penalidade``
    when positive, else :data:`_RHE_DEFAULT_PENALTY` with a WARNING
    (mirroring the source model's ``penalty <= 0 -> 1000.0`` fallback) — RHE is soft
    per its own penalty, never ``BIG_M``.

    Reads no ``dadger``; *hydro_to_ree* arrives from the caller (E7).
    *allocator* defaults to a fresh :class:`ConstraintIdAllocator` (ids from
    0) when omitted; the pipeline threads one shared instance across every
    emitter. Returns :class:`RheResult`: ``result`` is ``builder.result()``
    (``None`` when no RHE constraint survives); ``rho_acum_overrides`` maps
    every referenced novomodelo hydro id to its per-stage ρ_acum (MWmês/hm³) —
    unreferenced hydros keep novomodelo's ``computed`` default, mirroring the
    source model's emitter ``all_referenced_ids`` gate.
    """
    calendar = case.calendar
    operated = set(id_map.hydro_codes)
    rho_acum_energy = _per_stage_rho_acum_energy(effective, operated, calendar)

    ree_to_hydros: dict[int, list[int]] = {}
    for code, ree_code in hydro_to_ree.items():
        ree_to_hydros.setdefault(ree_code, []).append(code)

    allocator = allocator or ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)
    all_referenced_ids: set[int] = set()

    for record in census.to_generic:
        if record.family != "HE":
            continue

        expr_terms: list[tuple[float, str]] = []
        participating_codes: list[int] = []
        record_referenced_ids: set[int] = set()

        for term in record.terms:
            for code in sorted(ree_to_hydros.get(term.code, [])):
                if not _is_stored_energy_reservoir(effective, code):
                    continue
                try:
                    hid = id_map.hydro_id(code)
                except KeyError:
                    continue
                expr_terms.append(
                    (
                        term.coefficient,
                        f"@{rho_acum_name(hid)} * hydro_storage({hid})",
                    )
                )
                participating_codes.append(code)
                record_referenced_ids.add(hid)

        expression = _format_expression(expr_terms)
        if expression is None:
            _emit_rhe_no_storage_plants(record)
            continue

        penalty = (
            record.he_meta.valor_penalidade if record.he_meta is not None else None
        )
        if penalty is None or penalty <= 0.0:
            _emit_rhe_default_penalty(record, penalty)
            penalty = _RHE_DEFAULT_PENALTY

        tipo_limite = record.tipo_limite
        if tipo_limite not in (1, 2):
            _emit_rhe_unknown_tipo_limite(record, tipo_limite)

        new_bounds: dict[int, StageBounds] = {}
        for stage_index, stage_bounds in record.bounds.items():
            limite = stage_bounds.lower[0]
            if limite is None:
                new_bounds[stage_index] = StageBounds(lower=(None,), upper=(None,))
                continue
            if tipo_limite == 2:
                total_energy = sum(
                    rho_acum_energy[code][stage_index]
                    * effective.value(code, "volume_maximo", stage_index)
                    for code in participating_codes
                )
                rhs = (limite / 100.0) * total_energy
            else:
                rhs = limite
            new_bounds[stage_index] = StageBounds(lower=(rhs,), upper=(None,))

        adjusted_record = dataclasses.replace(record, bounds=new_bounds)
        builder.add(
            name=f"RHE_{record.constraint_id}",
            description=f"RHE stored-energy constraint {record.constraint_id}",
            expression=expression,
            slack={"enabled": True, "penalty": penalty},
            slots=slots_from_record(adjusted_record, calendar),
        )
        all_referenced_ids.update(record_referenced_ids)

    result = builder.result()
    if result is None:
        return RheResult(None, {})

    rho_acum_overrides = {
        hid: list(rho_acum_energy[id_map.hydro_codes[hid]])
        for hid in sorted(all_referenced_ids)
    }
    return RheResult(result, rho_acum_overrides)
