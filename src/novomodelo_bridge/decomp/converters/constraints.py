"""Foundation for the source model's generic-constraint emitter.

The special-constraint reader (``constraint_registers.py``) splits every
special constraint into ones that lower to a plain entity bound
(``single_term_bounds.py``) and everything else — multi-term participation,
an unbounded variable (spillage, diversion, pumping), a non-unit
coefficient, or a whole-cascade energy sum — which needs novomodelo's
**generic** constraint wire format instead: a flat ``expression`` string, a
``slack{enabled, penalty}`` pair, and a companion per-
``(constraint_id, stage_id, block_id)`` bounds table carrying novomodelo's
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

GNL pre-processing (abating commanded generation, aborting on uncommanded)
is out of scope here: ``emit_re_generics`` simply skips a constraint whose
``thermal_generation`` term names a thermal outside ``id_map.thermal_codes``
(an anticipation-track/GNL thermal), with a ``WARNING``, deferring
coordination with the GNL-ring track.

A coefficient that changes over a constraint's stages, or a term that joins
after its first stage, reads a ``per_stage`` parameter
(:func:`coefficient_parameter_name`); a constant one stays a literal.

``RHV``'s volume tipos (``VDEF``/``VDES``/``VBOM``) bound a volume over the
whole stage in hm³ (each block's flow times its duration, summed), not a
per-block flow. TRACKED NOVOMODELO-GAP (C19): novomodelo's generic constraints have no
stage-volume variable, and block-pinned flow terms would repeat the stage row
once per block, so ``emit_rhq_rhv_generics`` skips such a constraint with a
``WARNING``.

``HE`` (RHE stored-energy) constraints are emitted by
``emit_rhe_generics``: the LHS weights each reservoir's useful volume by
novomodelo's integrated accumulated productivity in MWh/hm³
(:mod:`novomodelo_bridge.core.stored_energy`), so the constraint is a whole-REE
stored-energy bound in MWh rather than a per-plant limit read off the
register. A cascade the source model splits between two REEs is weighted per
REE instead (:func:`_rhe_ree_terms`).
"""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.core.generic_constraint_builder import (
    ConstraintIdAllocator,
    GenericConstraintBuilder,
    GenericConstraintResult,
    is_bounded,
    soft_slack,
)
from novomodelo_bridge.core.stored_energy import (
    StoredEnergyShare,
    cascade_share,
    cut_cascade_share,
    downstream_share,
    share_parameters,
    stored_energy_floor,
    stored_energy_lhs,
)
from novomodelo_bridge.decomp.constraint_registers import StageBounds
from novomodelo_bridge.decomp.converters.cadastro import (
    effective_storage_range,
    is_reservoir,
    storage_envelope,
)
from novomodelo_bridge.decomp.converters.hydro import _downstream_operated

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pyarrow as pa

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


_LOG = logging.getLogger(__name__)

#: A term's weight in an expression: a literal coefficient, or the ``@name``
#: of the per-stage parameter carrying a stage-varying one.
Weight = float | str

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
    "VDEF": "the RHV volume-tipo skip",
    "VDES": "the RHV volume-tipo skip",
    "VBOM": "the RHV volume-tipo skip",
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
        (skipped by ``emit_rhq_rhv_generics``), or any other
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
            f"Cannot build a constraint term: {variable!r} is handled by "
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


def _format_expression(terms: Sequence[tuple[Weight, str]]) -> str | None:
    """Join ``(weight, token)`` pairs into a novomodelo expression string.

    Mirrors ``converters/constraints.py::_parse_formula``'s coefficient-
    formatting rules: a unit (``|coeff| == 1.0``) term is the bare token
    (or ``- token`` when negative); a fractional term is ``{abs_coeff} *
    {token}`` (or ``- {abs_coeff} * {token}`` when negative). A parameter
    weight is ``@{name} * {token}``, its sign carried by the parameter's
    values. The first term omits the leading ``+``; every later term uses
    ``+``/``-`` as a binary operator. Returns ``None`` for an empty term list.
    """
    if not terms:
        return None
    parts: list[str] = []
    for i, (coeff, token) in enumerate(terms):
        if isinstance(coeff, str):
            body = f"@{coeff} * {token}"
            parts.append(body if i == 0 else f"+ {body}")
            continue
        abs_coeff = abs(coeff)
        is_negative = coeff < 0.0
        body = token if abs_coeff == 1.0 else f"{abs_coeff} * {token}"
        if i == 0:
            parts.append(f"- {body}" if is_negative else body)
        else:
            parts.append(f"- {body}" if is_negative else f"+ {body}")
    return " ".join(parts)


def coefficient_parameter_name(record: ConstraintRecord, term_index: int) -> str:
    """``@name`` of the per-stage coefficient of term *term_index* of *record*."""
    return f"coef_{record.family.lower()}{record.constraint_id}_t{term_index}"


def _weighted_terms(
    record: ConstraintRecord, tokens: Sequence[str], n_stages: int
) -> tuple[list[tuple[Weight, str]], dict[str, list[float]]]:
    """Pair each term's token with its weight, and the per-stage parameters.

    A constant coefficient is a literal weight; a coefficient that changes
    over the constraint's stages (or a term that joins after its first stage)
    reads a ``per_stage`` parameter holding one value per study stage.
    """
    weighted: list[tuple[Weight, str]] = []
    parameters: dict[str, list[float]] = {}
    for index, (term, token) in enumerate(zip(record.terms, tokens, strict=True)):
        if not term.schedule:
            weighted.append((term.coefficient, token))
            continue
        name = coefficient_parameter_name(record, index)
        parameters[name] = [term.coefficient_at(stage) for stage in range(n_stages)]
        weighted.append((name, token))
    return weighted, parameters


class DecompGenericResult(NamedTuple):
    """Generic constraints of one emitter with the parameters they read.

    ``constraints`` and ``bounds`` are as in
    :class:`~novomodelo_bridge.core.generic_constraint_builder.GenericConstraintResult`.
    ``per_stage_parameters`` maps each ``@name`` to one value per study stage
    and ``expression_parameters`` maps each ``@name`` to its body, both for
    ``build_decomp_scalar_parameters``.
    """

    constraints: list[dict]
    bounds: pa.Table
    per_stage_parameters: dict[str, list[float]]
    expression_parameters: dict[str, str]


def _with_parameters(
    generic: GenericConstraintResult | None,
    per_stage_parameters: dict[str, list[float]],
    expression_parameters: dict[str, str] | None = None,
) -> DecompGenericResult | None:
    if generic is None:
        return None
    return DecompGenericResult(
        generic.constraints,
        generic.bounds,
        per_stage_parameters,
        expression_parameters or {},
    )


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
) -> list[str] | None:
    """Resolve every term of one generic RE record to its expression token.

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
    (skip-not-partial) — never a partial expression. Returns the tokens in
    term order when every term resolves.
    """
    resolved: list[str] = []
    for term in record.terms:
        if term.variable == "generation":
            if term.frequency is not None:
                _emit_re_frequency_split_deferred(term)
                return None
            resolved.append(_variable_token(term, id_map, {}))
        elif term.variable == "thermal_generation":
            if term.code not in id_map.thermal_codes:
                _emit_re_thermal_not_normal(term)
                return None
            resolved.append(_variable_token(term, id_map, {}))
        elif term.variable == "interchange":
            token = resolve_fi_term(term, id_map, line_map)
            if token is None:
                return None
            resolved.append(token)
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
) -> DecompGenericResult | None:
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

    Reads no ``dadger``; *big_m* and *line_map* arrive from the caller.
    *allocator* defaults to a fresh :class:`ConstraintIdAllocator` (ids from
    0) when omitted; the pipeline threads one shared instance across every
    emitter. Returns ``None`` when no RE generic constraint survives.
    """
    calendar = case.calendar
    allocator = allocator or ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)
    per_stage_parameters: dict[str, list[float]] = {}
    for record in census.to_generic:
        if record.family != "RE":
            continue
        tokens = _resolve_re_terms(record, id_map, line_map)
        if tokens is None:
            continue
        terms, parameters = _weighted_terms(record, tokens, len(calendar))
        expression = _format_expression(terms)
        if expression is None:
            continue
        added = builder.add(
            name=f"RE_{record.constraint_id}",
            description=f"RE generic constraint {record.constraint_id}",
            expression=expression,
            slack=soft_slack(big_m, "per_hour"),
            slots=slots_from_record(record, calendar),
        )
        if added is not None:
            per_stage_parameters.update(parameters)
    return _with_parameters(builder.result(), per_stage_parameters)


#: RHV ``CV.tipo`` stage-volume variables (hm³), skipped under C19.
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
) -> list[str] | None:
    """Resolve every term of one generic RHQ record to its expression token.

    Every flow variable resolves via :func:`_variable_token`. A ``QBOM`` term
    whose ``code`` has no matching entry in *pumping_station_ids* — it is a
    pumping-station code, never a hydro code, so it cannot fall through to
    ``id_map.hydro_id`` — emits :func:`_emit_rhq_qbom_no_station` and drops
    the whole record (skip-not-partial), checked *before* the
    ``_variable_token`` call so a missing station never raises a raw
    ``KeyError``. Returns the tokens in term order when every term resolves.
    """
    resolved: list[str] = []
    for term in record.terms:
        if term.variable == "QBOM" and term.code not in pumping_station_ids:
            _emit_rhq_qbom_no_station(record, term.code)
            return None
        resolved.append(_variable_token(term, id_map, pumping_station_ids))
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
                "term, a volume over the whole stage in hm3; novomodelo generic "
                "constraints have no stage-volume variable yet, so the "
                "constraint is skipped."
            ),
            remediation=(
                "This volume-based RHV limit is not converted until novomodelo "
                "supports stage volumes in generic constraints; no action "
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
) -> tuple[list[str], Mapping[int, StageBounds]] | None:
    """Resolve one generic multi-``VARM`` RHV record's expression and floor-adjusted
    bounds.

    Every term resolves via :func:`_variable_token` to a ``hydro_storage``
    token; a term whose ``code`` is absent from ``effective.base.index``
    emits :func:`_emit_rhv_varm_uncadastred` and drops the whole record
    (skip-not-partial) before any bounds are adjusted.

    The source model's ``LV`` limits are relative to each plant's own useful
    volume, so the absolute RHS is the **per-term additive floor**: for each
    declared stage, ``offset = sum(cᵢ(stage) * effective_storage_range(
    effective, codeᵢ, stage)[0] for term i)``, added to whichever side of the
    record's
    stage-level ``StageBounds`` is actually bounded
    (:func:`~novomodelo_bridge.core.generic_constraint_builder.is_bounded`) —
    a ``±1e21``/``None`` side is left untouched, never offset. *calendar* is
    accepted for signature symmetry with the per-block resolvers (mirroring
    ``single_term_bounds._hv_storage_contributions``); ``VARM`` is
    stage-level, so it is not read here.

    Returns the tokens in term order and the floor-adjusted
    ``{stage_index: StageBounds}`` map when every term resolves.
    """
    resolved: list[str] = []
    for term in record.terms:
        if term.code not in effective.base.index:
            _emit_rhv_varm_uncadastred(record, term.code)
            return None
        resolved.append(_variable_token(term, id_map, {}))

    adjusted: dict[int, StageBounds] = {}
    for stage_index, stage_bounds in record.bounds.items():
        offset = sum(
            term.coefficient_at(stage_index)
            * effective_storage_range(effective, term.code, stage_index)[0]
            for term in record.terms
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
) -> DecompGenericResult | None:
    """Emit every generic ``RHQ``/``RHV`` constraint (flow mixes and multi-``VARM``).

    Iterates ``census.to_generic`` filtered to ``record.family in ("HQ",
    "HV")``:

    - ``"HQ"`` (per-block flow terms) resolves via :func:`_resolve_rhq_terms`
      and feeds a two-sided ``HQ_<id>`` constraint.
    - ``"HV"`` carrying any volume-tipo term (``VDEF``/``VDES``/``VBOM``)
      emits :func:`_emit_rhv_volume_tipo_deferred` and is skipped (see the
      module docstring).
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
    arrive from the caller. *allocator* defaults to a fresh
    :class:`ConstraintIdAllocator` (ids from 0) when omitted; the pipeline
    threads one shared instance across every emitter. Returns ``None`` when
    no RHQ/RHV generic constraint survives.
    """
    calendar = case.calendar
    allocator = allocator or ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)
    per_stage_parameters: dict[str, list[float]] = {}
    for record in census.to_generic:
        if record.family not in ("HQ", "HV"):
            continue

        if record.family == "HQ":
            rhq_tokens = _resolve_rhq_terms(record, id_map, pumping_station_ids)
            if rhq_tokens is None:
                continue
            rhq_terms, parameters = _weighted_terms(record, rhq_tokens, len(calendar))
            expression = _format_expression(rhq_terms)
            if expression is None:
                continue
            added = builder.add(
                name=f"HQ_{record.constraint_id}",
                description=f"RHQ generic constraint {record.constraint_id}",
                expression=expression,
                slack=soft_slack(big_m, "per_hour"),
                slots=slots_from_record(record, calendar),
            )
            if added is not None:
                per_stage_parameters.update(parameters)
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
        hv_tokens, adjusted_bounds = hv_resolved
        hv_terms, parameters = _weighted_terms(record, hv_tokens, len(calendar))
        expression = _format_expression(hv_terms)
        if expression is None:
            continue
        adjusted_record = dataclasses.replace(record, bounds=adjusted_bounds)
        added = builder.add(
            name=f"HV_{record.constraint_id}",
            description=f"RHV generic constraint {record.constraint_id}",
            expression=expression,
            # per_hour on this hm³ row, as on the MW and m³/s rows: big_m is
            # sized as a price per MWh, so per_unit would make it a weak
            # penalty here.
            slack=soft_slack(big_m, "per_hour"),
            slots=slots_from_record(adjusted_record, calendar),
        )
        if added is not None:
            per_stage_parameters.update(parameters)
    return _with_parameters(builder.result(), per_stage_parameters)


#: The slack penalty (R$/MWh) an RHE constraint falls back to when its own
#: ``valor_penalidade`` is missing or non-positive — mirrors the source
#: model's VminOP ``penalty <= 0 -> 1000.0`` fallback
#: (``converters/constraints.py::convert_vminop_constraints``).
_RHE_DEFAULT_PENALTY = 1000.0


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


@dataclass(frozen=True)
class _RheCoupling:
    """One reservoir whose cascade the source model splits between two REEs."""

    plant_code: int
    ree_code: int
    cut_code: int
    fict_code: int | None
    fict_ree_code: int | None
    downstream_code: int | None


def _fictitious_twins(
    effective: EffectiveCadastro, operated: set[int]
) -> dict[int, int]:
    """Map each operated plant to the fictitious twin sharing its ``hidr`` posto.

    The source model keeps, outside the configuration, a non-generating
    (``produtibilidade_especifica == 0``) ``hidr.dat`` plant on a real plant's
    posto to account the energy below a REE cut — the same structural rule as
    the NEWAVE track's ``fictitious_codes``. Read off the *base* cadastro.
    """
    base = effective.base
    real_by_posto: dict[int, int] = {}
    for code in operated:
        if code in base.index:
            real_by_posto.setdefault(int(base.loc[code, "posto"]), code)
    twins: dict[int, int] = {}
    for code in base.index:
        code = int(code)
        if code in operated or float(base.loc[code, "produtibilidade_especifica"]):
            continue
        real = real_by_posto.get(int(base.loc[code, "posto"]))
        if real is not None:
            twins.setdefault(real, code)
    return twins


def _rhe_ree_terms(
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    hydro_to_ree: Mapping[int, int],
) -> tuple[dict[int, list[StoredEnergyShare]], list[_RheCoupling]]:
    """Build every REE's RHE shares, splitting the cascades the REEs couple.

    The source model cuts a reservoir's cascade at the first plant whose
    downstream lies in another REE **and** that has a fictitious twin
    (:func:`_fictitious_twins`), and accounts the water below the cut in the
    twin's downstream REE; ``hydros.json`` keeps the physical link, so novomodelo's
    whole-cascade parameters would over-count the reservoir's REE. A reservoir
    whose cascade reaches such a cut gets a
    :func:`~novomodelo_bridge.core.stored_energy.cut_cascade_share` in its own REE
    and, when it has a fictitious twin, a
    :func:`~novomodelo_bridge.core.stored_energy.downstream_share` in the REE of
    the first operated plant on the twin's cascade. Both use the reservoir's
    useful volume (its ``hydros.json`` storage envelope). A cascade that
    crosses a REE boundary without a twin at the crossing (e.g. into ITAIPU)
    is not cut. Every other reservoir gets a
    :func:`~novomodelo_bridge.core.stored_energy.cascade_share`. Returns the shares
    by REE code (in plant-code order) and the split reservoirs.
    """
    operated = set(id_map.hydro_codes)
    twins = _fictitious_twins(effective, operated)
    downstream = {
        code: _downstream_operated(effective, code, operated) for code in operated
    }

    def _cut_at(code: int, ree_code: int) -> bool:
        below = downstream[code]
        return (
            code in twins and below is not None and hydro_to_ree.get(below) != ree_code
        )

    terms: dict[int, list[StoredEnergyShare]] = defaultdict(list)
    couplings: list[_RheCoupling] = []
    for code, ree_code in sorted(hydro_to_ree.items()):
        if code not in operated or not _is_stored_energy_reservoir(effective, code):
            continue
        hid = id_map.hydro_id(code)
        min_storage, max_storage = storage_envelope(effective, code)
        useful = max_storage - min_storage

        chain = [code]
        while not _cut_at(chain[-1], ree_code):
            below = downstream[chain[-1]]
            if below is None or below in chain:
                chain = []
                break
            chain.append(below)

        if not chain:
            terms[ree_code].append(cascade_share(hid))
            continue

        terms[ree_code].append(
            cut_cascade_share(hid, useful, [id_map.hydro_id(c) for c in chain])
        )

        twin = twins.get(code)
        twin_downstream = (
            _downstream_operated(effective, twin, operated)
            if twin is not None
            else None
        )
        twin_ree = hydro_to_ree.get(twin_downstream) if twin_downstream else None
        lands = (
            twin is not None
            and twin_ree is not None
            and twin_ree != ree_code
            and _is_stored_energy_reservoir(effective, twin)
        )
        if lands:
            terms[twin_ree].append(
                downstream_share(hid, useful, id_map.hydro_id(twin_downstream))
            )
        couplings.append(
            _RheCoupling(
                plant_code=code,
                ree_code=ree_code,
                cut_code=chain[-1],
                fict_code=twin if lands else None,
                fict_ree_code=twin_ree if lands else None,
                downstream_code=twin_downstream if lands else None,
            )
        )
    return terms, couplings


def _emit_rhe_hydraulic_coupling(couplings: Sequence[_RheCoupling]) -> None:
    """Emit the ``decomp-rhe-ree-hydraulic-coupling`` INFO listing split reservoirs."""
    emit(
        Diagnostic(
            code="decomp-rhe-ree-hydraulic-coupling",
            severity=Severity.INFO,
            category="Special constraints",
            title=f"RHE split across coupled REEs ({len(couplings)} plant(s))",
            summary=(
                f"{len(couplings)} reservoir(s) drain into another REE; their "
                "stored energy counts in their own REE only down to the coupling "
                "plant, and the rest counts in the downstream REE."
            ),
            table=DiagnosticTable(
                columns=[
                    "Plant",
                    "REE",
                    "Coupling plant",
                    "Fictitious twin",
                    "Downstream REE",
                    "Downstream plant",
                ],
                rows=[
                    [
                        c.plant_code,
                        c.ree_code,
                        c.cut_code,
                        c.fict_code if c.fict_code is not None else "-",
                        c.fict_ree_code if c.fict_ree_code is not None else "-",
                        c.downstream_code if c.downstream_code is not None else "-",
                    ]
                    for c in couplings
                ],
                justify=["right"] * 6,
            ),
        )
    )


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
                "the limit as absolute MWmes (stage-average MW)."
            ),
            remediation=(
                "Check the HE register's tipo_limite column for constraint "
                f"{record.constraint_id}."
            ),
        )
    )


def rhe_pct_name(constraint_id: int) -> str:
    """Canonical ``@name`` for the per-stage limit fraction of one RHE record."""
    return f"pct_rhe{constraint_id}"


def emit_rhe_generics(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    census: ConstraintCensus,
    effective: EffectiveCadastro,
    hydro_to_ree: Mapping[int, int],
    allocator: ConstraintIdAllocator | None = None,
) -> DecompGenericResult | None:
    """Emit every generic ``HE`` (RHE stored-energy) constraint.

    Iterates ``census.to_generic`` filtered to ``record.family == "HE"``.
    Each record's ``energy`` terms name a REE (``term.code``) with a CM ±1
    sign (``term.coefficient``); the participating shares for that REE are
    the ones :func:`_rhe_ree_terms` builds over its stored-energy reservoirs
    (:func:`_is_stored_energy_reservoir`) operated in ``id_map``. A record
    whose terms yield **no** participating reservoir at all is skipped whole
    (:func:`_emit_rhe_no_storage_plants`, skip-not-partial) — never a partial
    constraint.

    Each record is one constraint over its stages. The LHS is the signed
    useful stored energy in MWh, the stored energy the source model's
    ``%EAMAX`` refers to (novomodelo moves the dead-volume term onto the bound).
    The bound follows the record's ``tipo_limite``:

    - ``2`` — ``limite`` percent of the shares' maximum stored energy,
      ``lhs >= @pct_rhe{id} * (Σ ±@emax)`` with a per-stage fraction
      parameter (``0.0`` where the record sets no limit) and a zero parquet
      bound;
    - ``1`` — ``limite`` is an absolute stage-average MW (MWmês) bound,
      converted to MWh with the stage's hours into the parquet bound; any
      other value is treated the same way, with a WARNING.

    The slack penalty is ``record.he_meta.valor_penalidade`` (R$/MWh) when
    positive, else :data:`_RHE_DEFAULT_PENALTY` with a WARNING (mirroring
    the source model's ``penalty <= 0 -> 1000.0`` fallback) — RHE is soft
    per its own penalty, never ``BIG_M``. It is ``per_unit``, a price per MWh
    of violation on weekly and monthly stages alike.

    Reads no ``dadger``; *hydro_to_ree* arrives from the caller.
    *allocator* defaults to a fresh :class:`ConstraintIdAllocator` (ids from
    0) when omitted; the pipeline threads one shared instance across every
    emitter. Returns ``None`` when no RHE constraint survives.
    """
    calendar = case.calendar

    ree_terms, couplings = _rhe_ree_terms(effective, id_map, hydro_to_ree)

    allocator = allocator or ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)
    per_stage_parameters: dict[str, list[float]] = {}
    expression_parameters: dict[str, str] = {}

    for record in census.to_generic:
        if record.family != "HE":
            continue

        shares = [
            (term.coefficient, share)
            for term in record.terms
            for share in ree_terms.get(term.code, [])
        ]
        if not shares:
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

        limits = {
            stage_index: stage_bounds.lower[0]
            for stage_index, stage_bounds in sorted(record.bounds.items())
            if stage_bounds.lower[0] is not None
        }
        record_parameters: dict[str, list[float]] = {}
        if tipo_limite == 2:
            pct_name = rhe_pct_name(record.constraint_id)
            expression = stored_energy_floor(shares, pct_name)
            lowers = dict.fromkeys(limits, 0.0)
            record_parameters[pct_name] = [
                limits.get(stage_index, 0.0) / 100.0
                for stage_index in range(len(calendar))
            ]
        else:
            expression = stored_energy_lhs(shares)
            lowers = {
                stage_index: limite * calendar[stage_index].total_hours
                for stage_index, limite in limits.items()
            }

        added = builder.add(
            name=f"RHE_{record.constraint_id}",
            description=f"RHE stored-energy constraint {record.constraint_id}",
            expression=expression,
            slack=soft_slack(penalty, "per_unit"),
            slots=slots_from_record(
                dataclasses.replace(
                    record,
                    bounds={
                        stage_index: StageBounds(lower=(lower,), upper=(None,))
                        for stage_index, lower in lowers.items()
                    },
                ),
                calendar,
            ),
        )
        if added is not None:
            per_stage_parameters.update(record_parameters)
            expression_parameters.update(share_parameters(s for _, s in shares))

    if couplings:
        _emit_rhe_hydraulic_coupling(couplings)
    return _with_parameters(
        builder.result(), per_stage_parameters, expression_parameters
    )
