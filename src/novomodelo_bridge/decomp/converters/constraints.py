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
(``build_fi_line_map``/``resolve_fi_term``), the ``FE``-contract resolver
(``build_contract_token_map``/``resolve_fe_term``), the ``BIG_M``
slack-penalty helper, and ``slots_from_record`` (the per-record (stage,
block) slot enumeration feeding the shared :class:`~novomodelo_bridge.core.
generic_constraint_builder.GenericConstraintBuilder`), plus the multi-term
``RE`` emitter (``emit_re_generics``) and the ``RHQ``/``RHV`` emitter
(``emit_rhq_rhv_generics``) built on top of it. ``emit_itaipu_ande_generic``
holds Itaipu's 50 Hz generation at or above the ``RI`` ANDE load.

An ``RE`` ``FT`` term on a GNL thermal of ``dadgnl`` never reaches these
emitters: :func:`fold_commanded_gnl_terms` moves its commanded generation into
the constraint's limits first, and
:func:`~novomodelo_bridge.decomp.constraint_registers.drop_unknown_entities`
leaves out, with a ``WARNING``, a constraint on any other thermal outside
``id_map.thermal_codes``.

A coefficient that changes over a constraint's stages, or a term that joins
after its first stage, reads a ``per_stage`` parameter
(:func:`coefficient_parameter_name`); a constant one stays a literal.

``RHV``'s volume tipos (``VDEF``/``VDES``/``VBOM``) bound a volume over the
whole stage in hm³ (each block's flow times its duration, summed), not a
per-block flow. They map to novomodelo's stage-volume variables, which admit
only stage-level amounts beside them, so every ``HV`` row stays stage-level.

``HE`` (RHE stored-energy) constraints are emitted by
``emit_rhe_generics``: the LHS weights each reservoir's useful volume by its
energy-chain productivity in MWh/hm³ (:mod:`novomodelo_bridge.decomp.converters.
rhe_shares`), so the constraint is a whole-REE stored-energy bound in MWh
rather than a per-plant limit read off the register.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
    format_stage_ranges,
)
from novomodelo_bridge.core.errors import FieldParseError, UnsupportedSourceError
from novomodelo_bridge.core.generic_constraint_builder import (
    ConstraintIdAllocator,
    GenericConstraintBuilder,
    GenericConstraintResult,
    is_bounded,
    soft_slack,
)
from novomodelo_bridge.core.generic_constraint_format import Weight, render_expression
from novomodelo_bridge.core.outside_terms import OutsideTerm
from novomodelo_bridge.core.stored_energy import (
    share_parameters,
    stored_energy_floor,
    stored_energy_lhs,
)
from novomodelo_bridge.decomp.constraint_registers import StageBounds, lowers_to_bound
from novomodelo_bridge.decomp.converters.anticipated import (
    commanded_gnl_generation,
    gnl_patamar_values_diagnostic,
)
from novomodelo_bridge.decomp.converters.cadastro import effective_storage_range
from novomodelo_bridge.decomp.converters.rhe_shares import (
    build_rhe_shares,
    ez_cap_unmodeled_stages,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    import pyarrow as pa

    from novomodelo_bridge.core.generic_constraint_builder import Slot
    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.constraint_registers import (
        ConstraintCensus,
        ConstraintRecord,
        ConstraintTerm,
    )
    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
    from novomodelo_bridge.decomp.converters.contracts import Contract
    from novomodelo_bridge.decomp.converters.rhe_shares import RheCoupling
    from novomodelo_bridge.decomp.id_map import DecompIdMap
    from novomodelo_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)


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

#: RHV stage-volume tipos (hm³ over the stage); they have no entity-bound
#: axis, so they stay out of :data:`_HYDRO_TOKEN_FN`.
_HYDRO_VOLUME_TOKEN_FN: dict[str, str] = {
    "VDEF": "hydro_outflow_volume",
    "VDES": "hydro_diversion_volume",
}

#: Variables ``_variable_token`` refuses to resolve itself, mapped to the
#: mechanism that resolves them instead.
_DEFERRED_VARIABLES: dict[str, str] = {
    "interchange": "the interchange (FI line) resolution path",
    "contract": "the FE contract resolution path",
}


def _variable_token(
    term: ConstraintTerm,
    id_map: DecompIdMap,
    pumping_station_ids: Mapping[int, int],
) -> str:
    """The novomodelo expression token for one constraint term.

    Dispatches on ``term.variable``: ``generation``/``thermal_generation``
    map to the whole-plant generation tokens; ``QDEF``/``QTUR``/``QVER``/
    ``QDES``/``VARM`` map through :data:`_HYDRO_TOKEN_FN` and ``VDEF``/
    ``VDES`` through :data:`_HYDRO_VOLUME_TOKEN_FN`, both via
    ``id_map.hydro_id``; ``QBOM`` and ``VBOM`` map through
    *pumping_station_ids*.

    Raises
    ------
    ValueError
        For ``interchange`` (resolved by ``resolve_fi_term``, which needs
        the line map), ``contract`` (resolved by ``resolve_fe_term``, which
        needs the contract list), or any other unrecognised variable —
        fail-loud, never a silent skip that would build a malformed
        expression.
    KeyError
        Propagated from ``id_map.hydro_id``/``id_map.thermal_id`` for an
        unknown plant/thermal code, or from *pumping_station_ids* for an
        unknown ``QBOM``/``VBOM`` code.
    """
    variable = term.variable
    if variable == "generation":
        return _hydro_generation_token(id_map.hydro_id(term.code))
    if variable == "thermal_generation":
        return f"thermal_generation({id_map.thermal_id(term.code)})"
    if variable == "QBOM":
        return f"pumping_flow({pumping_station_ids[term.code]})"
    if variable == "VBOM":
        return f"pumping_volume({pumping_station_ids[term.code]})"
    token_fn = _HYDRO_TOKEN_FN.get(variable) or _HYDRO_VOLUME_TOKEN_FN.get(variable)
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


class Unresolved(NamedTuple):
    """Why a term has no token: a diagnostic code, the term's cells in that
    diagnostic's table after the constraint id, and the cause of a term on an
    entity outside the case.

    The term leaves its whole constraint out unless its *cause* leaves out
    only the term (:attr:`OutsideTerm.drops_constraint`).
    """

    code: str
    cells: tuple[object, ...]
    cause: OutsideTerm | None = None

    @property
    def drops_constraint(self) -> bool:
        return self.cause is None or self.cause.drops_constraint


_FI_NO_LINE = "decomp-fi-no-line"
_RE_TERM_NO_LINE = "decomp-re-term-interchange-no-line"
_FE_NO_CONTRACT = "decomp-fe-no-contract"
_FREQUENCY_SPLIT = "decomp-re-frequency-split-deferred"


def resolve_fi_term(
    term: ConstraintTerm,
    id_map: DecompIdMap,
    line_map: Mapping[tuple[int, int], int],
) -> str | Unresolved:
    """Resolve one ``FI`` interchange term to a directional novomodelo line token.

    novomodelo has no submarket-pair variable, only per-line directional
    ``line_direct(id)``/``line_reverse(id)`` variables, so — mirroring the
    source model precedent's ``ener_interc`` resolution
    (``newave/converters/constraints.py::_decision_variable``) — the term's direction
    is encoded by *which* directional variable is chosen, never by negating the
    coefficient (a negated coefficient would dilute the bound when the LP
    routes the non-canonical direction).

    Resolves both submarket names to bus ids via ``id_map.bus_id_by_name``
    (the reconciled ``SB`` name -> id map, which also handles the fictitious
    ``IV`` transhipment submarket) and probes *line_map* in both
    orientations, without canonicalizing the pair: ``(de, para)`` matching a
    line's declared ``source_bus_id``/``target_bus_id`` returns
    ``"line_direct(id)"`` (the term's orientation matches the line's); the
    reverse match returns ``"line_reverse(id)"``.

    Returns an :class:`Unresolved` under ``decomp-fi-no-line`` when either
    submarket name is unknown, which leaves the whole constraint out, and
    under ``decomp-re-term-interchange-no-line`` when no line joins the pair
    in either orientation: no flow runs between them, so only the term is
    left out. Raises ``ValueError`` if *term* is missing a submarket name (a
    caller contract violation: a genuine ``FI`` term, as produced by the
    reader, always carries both).
    """
    if term.submarket_de is None or term.submarket_para is None:
        raise ValueError(
            "Cannot resolve an interchange term: it must carry both "
            f"submarket_de and submarket_para, got {term!r}."
        )
    pair = (term.submarket_de, term.submarket_para)
    try:
        de = id_map.bus_id_by_name(term.submarket_de)
        para = id_map.bus_id_by_name(term.submarket_para)
    except KeyError:
        return Unresolved(
            _FI_NO_LINE, (*pair, "a submarket has no SB record"), OutsideTerm.ABSENT
        )
    if (de, para) in line_map:
        return f"line_direct({line_map[(de, para)]})"
    if (para, de) in line_map:
        return f"line_reverse({line_map[(para, de)]})"
    return Unresolved(_RE_TERM_NO_LINE, pair, OutsideTerm.NO_LINE)


def build_contract_token_map(
    contracts: Sequence[Contract],
) -> dict[tuple[int, int], str | None]:
    """Map each ``CI``/``CE`` contract's ``(numero, submarket code)`` to its token.

    The source model identifies a contract by its number within its
    submarket, whether it imports or exports, so an ``FE`` term names it by
    that pair alone; the token's ``contract_import``/``contract_export``
    variant comes from the contract's register. A pair declared by more than
    one contract (which the source model forbids) maps to ``None`` so the
    resolver can report the ambiguity instead of guessing.
    """
    tokens: dict[tuple[int, int], str | None] = {}
    for contract in contracts:
        key = (contract.numero, contract.bus_code)
        token = f"contract_{contract.kind}({contract.id})"
        tokens[key] = None if key in tokens else token
    return tokens


def resolve_fe_term(
    term: ConstraintTerm,
    contract_tokens: Mapping[tuple[int, int], str | None],
) -> str | Unresolved:
    """Resolve one ``FE`` contract term to its novomodelo contract token.

    The coefficient is used as declared for both contract kinds: the source
    model adds an import and an export to the constraint the same way, and a
    deck flips a contract's sign through the coefficient.

    Returns an :class:`Unresolved` under ``decomp-fe-no-contract`` when no
    converted contract, or more than one, carries the term's ``(numero,
    submarket)`` pair; the caller then drops the whole constraint. Raises
    ``ValueError`` if *term* carries no submarket (a caller contract
    violation: the reader always sets it on an ``FE`` term).
    """
    if term.submarket is None:
        raise ValueError(
            f"Cannot resolve a contract term: it must carry a submarket, got {term!r}."
        )
    key = (term.code, term.submarket)
    if key not in contract_tokens:
        return Unresolved(_FE_NO_CONTRACT, (*key, "no CI or CE record"))
    token = contract_tokens[key]
    if token is None:
        return Unresolved(_FE_NO_CONTRACT, (*key, "both a CI and a CE record"))
    return token


def coefficient_parameter_name(record: ConstraintRecord, term_index: int) -> str:
    """``@name`` of the per-stage coefficient of term *term_index* of *record*."""
    return f"coef_{record.family.lower()}{record.constraint_id}_t{term_index}"


def _weighted_terms(
    record: ConstraintRecord, tokens: Sequence[str | None], n_stages: int
) -> tuple[list[tuple[Weight, str]], dict[str, list[float]]]:
    """Pair each term's token with its weight, and the per-stage parameters;
    a term whose token is ``None`` is left out.

    A constant coefficient is a literal weight; a coefficient that changes
    over the constraint's stages (or a term that joins after its first stage)
    reads a ``per_stage`` parameter holding one value per study stage.
    """
    weighted: list[tuple[Weight, str]] = []
    parameters: dict[str, list[float]] = {}
    for index, (term, token) in enumerate(zip(record.terms, tokens, strict=True)):
        if token is None:
            continue
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


def _resolve_re_terms(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    line_map: Mapping[tuple[int, int], int],
    contract_tokens: Mapping[tuple[int, int], str | None],
    left_out: list[Unresolved],
) -> list[str | None] | Unresolved:
    """Resolve every term of one generic RE record to its expression token.

    Dispatches on ``term.variable``: a ``generation`` term split by
    frequency (Itaipu 50/60 Hz) needs a frequency-to-bus map this conversion
    does not build, so it leaves the record unresolved under
    ``decomp-re-frequency-split-deferred``; any other ``generation`` or
    ``thermal_generation`` term resolves via ``_variable_token``;
    ``interchange`` resolves via :func:`resolve_fi_term` and ``contract`` via
    :func:`resolve_fe_term`.

    The first term that drops its constraint (:attr:`Unresolved.drops_constraint`)
    leaves the **whole** constraint out (skip-not-partial) — never a partial
    expression. Returns the tokens in term order otherwise, ``None`` for each
    term left out of the sum, whose finding is added to *left_out*; a record
    with every term left out is unresolved under ``decomp-fi-no-line``.
    """
    resolved: list[str | None] = []
    zero: list[Unresolved] = []
    for term in record.terms:
        token: str | Unresolved
        if term.variable == "generation" and term.frequency is not None:
            token = Unresolved(_FREQUENCY_SPLIT, (term.code, f"{term.frequency:g}"))
        elif term.variable in ("generation", "thermal_generation"):
            token = _variable_token(term, id_map, {})
        elif term.variable == "interchange":
            token = resolve_fi_term(term, id_map, line_map)
        elif term.variable == "contract":
            token = resolve_fe_term(term, contract_tokens)
        else:
            raise ValueError(
                f"_resolve_re_terms: unexpected RE term variable {term.variable!r}"
            )
        if isinstance(token, Unresolved):
            if token.drops_constraint:
                return token
            zero.append(token)
            resolved.append(None)
            continue
        resolved.append(token)
    if zero and len(zero) == len(resolved):
        return Unresolved(_FI_NO_LINE, (*zero[0].cells, "no IA record joins them"))
    left_out.extend(zero)
    return resolved


#: ``(title, summary, columns)`` of each diagnostic an RE term leaves its
#: constraint out under; the table leads with the constraint id.
_RE_UNRESOLVED: dict[str, tuple[str, str, list[str]]] = {
    _FI_NO_LINE: (
        "RE constraints with an FI term on no line",
        "The listed RE constraints have an FI term on a submarket no SB record "
        "declares, or only FI terms between submarkets no IA record joins.",
        ["Constraint", "From", "To", "Reason"],
    ),
    _FE_NO_CONTRACT: (
        "RE constraints with an FE term on no contract",
        "The listed RE constraints have an FE term whose contract number and "
        "submarket do not name exactly one CI or CE record of dadger.rvN.",
        ["Constraint", "Contract", "Submarket", "Reason"],
    ),
    _FREQUENCY_SPLIT: (
        "RE constraints on one frequency of a plant",
        "The listed RE constraints have an FU term on one frequency of a plant "
        "split by frequency, such as Itaipu at 50 or 60 Hz, among other terms, "
        "and the conversion does not map a frequency to its bus.",
        ["Constraint", "Plant code", "Frequency (Hz)"],
    ),
}

_RE_UNRESOLVED_REMEDIATION = {
    _FI_NO_LINE: (
        "Declare the listed submarket pairs in SB and IA records of dadger.rvN, "
        "or remove the FI terms."
    ),
    _FE_NO_CONTRACT: (
        "Name a contract declared in one CI or CE record of the term's "
        "submarket in each listed FE term."
    ),
}


def _emit_re_unresolved(rows_by_code: Mapping[str, Sequence[list[object]]]) -> None:
    """Emit one WARNING per kind of term that left RE constraints out."""
    for code, (title, summary, columns) in _RE_UNRESOLVED.items():
        rows = rows_by_code.get(code)
        if not rows:
            continue
        emit(
            Diagnostic(
                code=code,
                severity=Severity.WARNING,
                category="Special constraints",
                title=f"{title} ({len(rows)})",
                summary=(
                    f"{summary} The conversion leaves each constraint out, so "
                    "the case runs without the limits DECOMP applies."
                ),
                table=DiagnosticTable(columns=columns, rows=list(rows)),
                remediation=_RE_UNRESOLVED_REMEDIATION.get(code),
            ),
            logger=_LOG,
        )


def _emit_re_terms_left_out(rows: Sequence[list[object]]) -> None:
    """Emit the note on the ``FI`` terms left out of kept RE constraints."""
    if not rows:
        return
    emit(
        Diagnostic(
            code=_RE_TERM_NO_LINE,
            severity=OutsideTerm.NO_LINE.severity,
            category="Special constraints",
            title=f"RE terms on submarkets without interchange left out ({len(rows)})",
            summary=(
                "The listed RE constraints have an FI term between two "
                "submarkets no IA record of dadger.rvN joins, so no flow runs "
                "between them. The conversion leaves the term out and keeps "
                "the rest of the constraint."
            ),
            table=DiagnosticTable(
                columns=["Constraint", "From", "To"], rows=list(rows)
            ),
        ),
        logger=_LOG,
    )


def emit_re_generics(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    census: ConstraintCensus,
    line_map: Mapping[tuple[int, int], int],
    contract_tokens: Mapping[tuple[int, int], str | None],
    big_m: float,
    allocator: ConstraintIdAllocator | None = None,
) -> DecompGenericResult | None:
    """Emit every generic ``RE`` constraint (hydro/thermal/interchange/contract terms).

    Iterates ``census.to_generic`` filtered to ``record.family == "RE"``,
    resolves each record's terms via :func:`_resolve_re_terms`, and feeds a
    :class:`~novomodelo_bridge.core.generic_constraint_builder.GenericConstraintBuilder`;
    a record with a term that cannot be resolved is dropped entirely
    (skip-not-partial, one WARNING per kind of unresolved term), and an
    ``FI`` term on no line is left out of a record kept (one INFO) — see
    :func:`_resolve_re_terms`. *census* is screened by
    :func:`~novomodelo_bridge.decomp.constraint_registers.drop_unknown_entities`,
    so every ``FT`` thermal is a CT thermal. Single-hydro-generation RE records never
    reach here — they lower to a plant bound in ``single_term_bounds.py``
    (``lowers_to_bound``). ``HQ``/``HV`` records also present in
    ``census.to_generic`` are :func:`emit_rhq_rhv_generics`'s, not this
    function's; ``HE`` records are unhandled (deferred).

    Reads no ``dadger``; *big_m*, *line_map* and *contract_tokens*
    (:func:`build_contract_token_map`) arrive from the caller.
    *allocator* defaults to a fresh :class:`ConstraintIdAllocator` (ids from
    0) when omitted; the pipeline threads one shared instance across every
    emitter. Returns ``None`` when no RE generic constraint survives.
    """
    calendar = case.calendar
    allocator = allocator or ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)
    per_stage_parameters: dict[str, list[float]] = {}
    unresolved: dict[str, list[list[object]]] = defaultdict(list)
    left_out_rows: list[list[object]] = []
    for record in census.to_generic:
        if record.family != "RE":
            continue
        left_out: list[Unresolved] = []
        tokens = _resolve_re_terms(record, id_map, line_map, contract_tokens, left_out)
        if isinstance(tokens, Unresolved):
            unresolved[tokens.code].append([record.constraint_id, *tokens.cells])
            continue
        terms, parameters = _weighted_terms(record, tokens, len(calendar))
        expression = render_expression(terms)
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
            left_out_rows.extend([record.constraint_id, *u.cells] for u in left_out)
    _emit_re_unresolved(unresolved)
    _emit_re_terms_left_out(left_out_rows)
    return _with_parameters(builder.result(), per_stage_parameters)


def _offset_if_bounded(value: float | None, offset: float) -> float | None:
    """Add *offset* to *value* iff it is a real bound; otherwise pass it through.

    Mirrors :func:`~novomodelo_bridge.core.generic_constraint_builder.is_bounded`'s
    sentinel check, phrased so mypy narrows *value* to ``float`` on the
    addition: a ``None`` or ``±1e21`` side never gets an offset added to it.
    """
    if value is None or not is_bounded(value):
        return value
    return value + offset


def fold_commanded_gnl_terms(
    census: ConstraintCensus,
    gnl_generation: Mapping[int, Sequence[Sequence[float] | None]],
    *,
    gnl_submarkets: Mapping[int, int],
    ct_submarkets: Mapping[int, Collection[int]],
    dadger_path: str,
    dadgnl_path: str | None,
) -> tuple[ConstraintCensus, list[Diagnostic]]:
    """*census* with every ``RE`` ``FT`` term on a GNL plant moved into its
    constraint's limits, as the source model does.

    *gnl_generation* is
    :func:`~novomodelo_bridge.decomp.converters.anticipated.gnl_block_commitments`.
    At each stage and block, the term's coefficient times the plant's
    commanded ``gl`` generation is subtracted from both limits, and the term
    leaves the sum. The rest of the constraint is lowered anew, so one left
    with a single unit term becomes an entity bound. A constraint left with
    no term bounds no case variable: it is left out and reported in one
    INFO. ``by_family`` keeps every record read.

    *gnl_submarkets* gives each GNL plant's subsystem and *ct_submarkets*
    the subsystems of each ``CT`` plant number, for a GNL plant whose number
    a ``CT`` record also gives.

    Raises
    ------
    FieldParseError
        When a GNL plant's generation is not commanded at a stage its term is
        in (:func:`~novomodelo_bridge.decomp.converters.anticipated.commanded_gnl_generation`),
        where the source model stops too, or when a ``CT`` record gives the
        plant's number in the plant's own subsystem, which DECOMP does not
        allow.
    UnsupportedSourceError
        When a ``CT`` record gives the plant's number in another subsystem:
        the conversion tells thermals apart by number alone, so the term
        cannot be told apart as a decision or a commanded generation.
    """
    folded: dict[int, ConstraintRecord | None] = {}
    rows: list[list[object]] = []
    for record in (*census.to_bounds, *census.to_generic):
        if record.family != "RE":
            continue
        gnl_terms = [
            term
            for term in record.terms
            if term.variable == "thermal_generation" and term.code in gnl_generation
        ]
        if not gnl_terms:
            continue
        for term in gnl_terms:
            _refuse_shared_thermal_number(
                record.constraint_id,
                term.code,
                gnl_submarkets,
                ct_submarkets,
                dadger_path=dadger_path,
                dadgnl_path=dadgnl_path,
            )
        bounds = {
            stage_index: _without_commanded_generation(
                record.constraint_id,
                stage_index,
                stage_bounds,
                gnl_terms,
                gnl_generation,
                dadgnl_path,
            )
            for stage_index, stage_bounds in record.bounds.items()
        }
        kept = tuple(term for term in record.terms if term not in gnl_terms)
        if kept:
            folded[record.constraint_id] = dataclasses.replace(
                record,
                terms=kept,
                bounds=bounds,
                gnl_folds=tuple(
                    (term.code, stage_index)
                    for stage_index in record.bounds
                    for term in gnl_terms
                    if term.coefficient_at(stage_index) != 0.0
                ),
            )
        else:
            folded[record.constraint_id] = None
            rows.extend([record.constraint_id, term.code] for term in gnl_terms)
    if not folded:
        return census, []
    to_bounds: list[ConstraintRecord] = []
    to_generic: list[ConstraintRecord] = []
    for bucket, records in (
        (to_bounds, census.to_bounds),
        (to_generic, census.to_generic),
    ):
        for record in records:
            if record.family != "RE" or record.constraint_id not in folded:
                bucket.append(record)
                continue
            kept_record = folded[record.constraint_id]
            if kept_record is not None:
                target = to_bounds if lowers_to_bound(kept_record) else to_generic
                target.append(kept_record)
    screened = dataclasses.replace(
        census, to_bounds=tuple(to_bounds), to_generic=tuple(to_generic)
    )
    return screened, [_gnl_terms_only_diagnostic(rows)] if rows else []


def re_gnl_patamar_diagnostic(
    census: ConstraintCensus,
    gnl_generation: Mapping[int, Sequence[Sequence[float] | None]],
) -> Diagnostic | None:
    """The INFO for the ``RE`` constraints *census* keeps that
    :func:`fold_commanded_gnl_terms` folded with a GNL generation of
    *gnl_generation* differing by patamar
    (:func:`~novomodelo_bridge.decomp.converters.anticipated.gnl_patamar_values_diagnostic`)."""
    reads: list[tuple[int, int, int, Sequence[float]]] = []
    for record in (*census.to_bounds, *census.to_generic):
        for thermal_code, stage_index in record.gnl_folds:
            blocks = gnl_generation[thermal_code][stage_index]
            assert blocks is not None  # the fold stops on an uncommanded stage
            reads.append((record.constraint_id, thermal_code, stage_index, blocks))
    return gnl_patamar_values_diagnostic(
        "decomp-re-gnl-patamar-values",
        reads,
        noun="RE constraints",
        column="Constraint",
    )


def _refuse_shared_thermal_number(
    constraint_id: int,
    code: int,
    gnl_submarkets: Mapping[int, int],
    ct_submarkets: Mapping[int, Collection[int]],
    *,
    dadger_path: str,
    dadgnl_path: str | None,
) -> None:
    """Stop on the ``FT`` term of ``RE`` *constraint_id* on GNL plant *code*
    when a ``CT`` record gives the same number.

    Raises
    ------
    FieldParseError
        On the ``TG`` field of *dadgnl_path*, when the ``CT`` record is in
        the plant's own subsystem: DECOMP requires distinct numbers for two
        thermals of one subsystem, even across ``CT`` and ``TG`` records.
    UnsupportedSourceError
        On the ``RE/FT`` field of *dadger_path*, when it is in another one.
    """
    ct_subsystems = ct_submarkets.get(code)
    if ct_subsystems is None:
        return
    subsystem = gnl_submarkets[code]
    if subsystem in ct_subsystems:
        raise FieldParseError(
            f"thermal {code}, which the FT term of RE {constraint_id} is on, "
            f"is the number of both a CT record in dadger and a TG record in "
            f"dadgnl in subsystem {subsystem}; DECOMP requires distinct "
            "numbers for two thermals of one subsystem, so give the GNL "
            "thermal a number no CT record of its subsystem uses",
            path=dadgnl_path,
            field="TG",
        )
    others = sorted(ct_subsystems)
    raise UnsupportedSourceError(
        f"RE {constraint_id} has an FT term on thermal {code}, the number of "
        f"a TG record in dadgnl in subsystem {subsystem} and of a CT record "
        f"in dadger in subsystem{'s' if len(others) > 1 else ''} "
        f"{', '.join(str(s) for s in others)}; such a term is not converted "
        "yet, since the conversion tells thermals apart by number alone",
        path=dadger_path,
        field="RE/FT",
    )


def _without_commanded_generation(
    constraint_id: int,
    stage_index: int,
    stage_bounds: StageBounds,
    gnl_terms: Sequence[ConstraintTerm],
    gnl_generation: Mapping[int, Sequence[Sequence[float] | None]],
    dadgnl_path: str | None,
) -> StageBounds:
    """*stage_bounds* less each GNL term's coefficient times its plant's
    commanded MW, per block; a term absent at the stage reads no commitment."""
    offsets = [0.0] * len(stage_bounds.lower)
    for term in gnl_terms:
        coefficient = term.coefficient_at(stage_index)
        if coefficient == 0.0:
            continue
        blocks = commanded_gnl_generation(
            gnl_generation,
            term.code,
            stage_index,
            reference=f"the FT term of RE {constraint_id} on thermal {term.code}",
            dadgnl_path=dadgnl_path,
        )
        for block, mw in enumerate(blocks[: len(offsets)]):
            offsets[block] += coefficient * mw
    return StageBounds(
        lower=tuple(
            _offset_if_bounded(value, -offset)
            for value, offset in zip(stage_bounds.lower, offsets, strict=True)
        ),
        upper=tuple(
            _offset_if_bounded(value, -offset)
            for value, offset in zip(stage_bounds.upper, offsets, strict=True)
        ),
    )


def _gnl_terms_only_diagnostic(rows: list[list[object]]) -> Diagnostic:
    return Diagnostic(
        code="decomp-re-gnl-terms-only",
        severity=Severity.INFO,
        category="Special constraints",
        title=f"RE constraints only on GNL thermals left out ({len(rows)})",
        summary=(
            "The listed RE constraints have FT terms only on GNL thermals of "
            "dadgnl.rvN. As in DECOMP, the generation their GL records command "
            "is moved into the limits, which leaves no case variable to bound, "
            "so the conversion leaves each constraint out."
        ),
        table=DiagnosticTable(
            columns=["Constraint", "Thermal code"],
            rows=sorted(rows),
            justify=["right", "right"],
        ),
    )


def _resolve_hv_terms(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    effective: EffectiveCadastro,
    pumping_station_ids: Mapping[int, int],
) -> tuple[list[str], Mapping[int, StageBounds]]:
    """Resolve one generic RHV record's expression and floor-adjusted bounds.

    Every term resolves via :func:`_variable_token`.

    The source model's ``LV`` limits are relative to each ``VARM`` plant's
    own useful volume, so the absolute RHS is the **per-term additive
    floor**: for each declared stage, ``offset = sum(cᵢ(stage) *
    effective_storage_range(effective, codeᵢ, stage)[0] for VARM term i)``,
    added to whichever side of the record's stage-level ``StageBounds`` is
    actually bounded
    (:func:`~novomodelo_bridge.core.generic_constraint_builder.is_bounded`) —
    a ``±1e21``/``None`` side is left untouched, never offset. The volume
    tipos ``VDEF``/``VDES``/``VBOM`` carry no floor.

    Returns the tokens in term order and the floor-adjusted
    ``{stage_index: StageBounds}`` map.
    """
    resolved = [
        _variable_token(term, id_map, pumping_station_ids) for term in record.terms
    ]
    storage_terms = [term for term in record.terms if term.variable == "VARM"]
    adjusted: dict[int, StageBounds] = {}
    for stage_index, stage_bounds in record.bounds.items():
        offset = sum(
            term.coefficient_at(stage_index)
            * effective_storage_range(effective, term.code, stage_index)[0]
            for term in storage_terms
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
    """Emit every generic ``RHQ``/``RHV`` constraint (flow and volume mixes).

    Iterates ``census.to_generic`` filtered to ``record.family in ("HQ",
    "HV")``:

    - ``"HQ"`` (per-block flow terms) resolves via :func:`_variable_token`
      and feeds a two-sided ``HQ_<id>`` constraint.
    - ``"HV"`` (stage-level ``VARM`` and volume-tipo terms) resolves via
      :func:`_resolve_hv_terms` and feeds a two-sided ``HV_<id>`` constraint
      on ``dataclasses.replace(record, bounds=adjusted)`` — the floor-
      adjusted bounds, never the reader's original record (a frozen
      dataclass is never mutated in place).

    *census* is screened by
    :func:`~novomodelo_bridge.decomp.constraint_registers.drop_unknown_entities`,
    so every ``QBOM``/``VBOM`` term names a pumping station and every ``VARM``
    term an operated plant. Single-term RHQ/RHV records on a bounded variable
    never reach here — they lower to an entity bound in
    ``single_term_bounds.py`` (``lowers_to_bound``).

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
            rhq_tokens = [
                _variable_token(term, id_map, pumping_station_ids)
                for term in record.terms
            ]
            rhq_terms, parameters = _weighted_terms(record, rhq_tokens, len(calendar))
            expression = render_expression(rhq_terms)
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

        hv_tokens, adjusted_bounds = _resolve_hv_terms(
            record, id_map, effective, pumping_station_ids
        )
        hv_terms, parameters = _weighted_terms(record, hv_tokens, len(calendar))
        expression = render_expression(hv_terms)
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


def _emit_rhe_hydraulic_coupling(
    couplings: Sequence[RheCoupling], names: Mapping[int, str]
) -> None:
    """Emit the ``decomp-rhe-ree-hydraulic-coupling`` INFO listing cut reservoirs."""
    emit(
        Diagnostic(
            code="decomp-rhe-ree-hydraulic-coupling",
            severity=Severity.INFO,
            category="Special constraints",
            title=f"RHE stored energy shared by coupled REEs ({len(couplings)})",
            summary=(
                "The listed reservoirs have an EZ record and drain into another "
                "REE. As in DECOMP, their stored energy counts in their own REE "
                "down to the coupling plant, and up to the EZ share of their "
                "useful volume counts in the downstream REE."
            ),
            table=DiagnosticTable(
                columns=[
                    "Code",
                    "Name",
                    "REE",
                    "Coupling code",
                    "EZ (%)",
                    "Downstream REE",
                    "Downstream code",
                ],
                rows=[
                    [
                        c.plant_code,
                        names.get(c.plant_code, ""),
                        c.ree_code,
                        c.cut_code,
                        f"{c.ez_percent:g}",
                        c.downstream_ree_code,
                        c.downstream_code,
                    ]
                    for c in couplings
                ],
                justify=["right", "left", "right", "right", "right", "right", "right"],
            ),
        ),
        logger=_LOG,
    )


def _emit_rhe_findings(
    *,
    no_storage: Sequence[list[object]],
    default_penalty: Sequence[list[object]],
    unknown_tipo: Sequence[list[object]],
    ez_cap_unmodeled: Sequence[list[object]],
) -> None:
    """Emit one WARNING per kind of RHE finding, each listing its constraints."""
    if no_storage:
        emit(
            Diagnostic(
                code="decomp-rhe-no-storage-plants",
                severity=Severity.WARNING,
                category="Special constraints",
                title=f"HE constraints on REEs without storage ({len(no_storage)})",
                summary=(
                    "The listed HE constraints bound the stored energy of REEs "
                    "with no operated reservoir that stores water. The "
                    "conversion leaves each constraint out, so the case runs "
                    "without these limits."
                ),
                table=DiagnosticTable(
                    columns=["Constraint", "REE"], rows=list(no_storage)
                ),
                remediation=(
                    "Map each listed REE to an operated reservoir through its "
                    "UH records in dadger.rvN, or remove the HE constraint."
                ),
            ),
            logger=_LOG,
        )
    if default_penalty:
        emit(
            Diagnostic(
                code="decomp-rhe-default-penalty",
                severity=Severity.WARNING,
                category="Special constraints",
                title=(
                    f"HE constraints without a positive penalty ({len(default_penalty)})"
                ),
                summary=(
                    "The listed HE records give a penalty that is missing or not "
                    "positive, so the conversion penalizes their violation at "
                    f"{_RHE_DEFAULT_PENALTY:g} R$/MWh instead."
                ),
                table=DiagnosticTable(
                    columns=["Constraint", "Declared penalty", "Applied penalty"],
                    rows=list(default_penalty),
                    justify=["right", "right", "right"],
                ),
                remediation="Give each listed HE record a positive penalty in dadger.rvN.",
            ),
            logger=_LOG,
        )
    if unknown_tipo:
        emit(
            Diagnostic(
                code="decomp-rhe-unknown-tipo-limite",
                severity=Severity.WARNING,
                category="Special constraints",
                title=f"HE constraints with an unknown limit type ({len(unknown_tipo)})",
                summary=(
                    "The listed HE records give a limit type other than 1 "
                    "(MWmês) or 2 (percent of the maximum stored energy), so the "
                    "conversion reads each limit as MWmês."
                ),
                table=DiagnosticTable(
                    columns=["Constraint", "Limit type"],
                    rows=list(unknown_tipo),
                    justify=["right", "right"],
                ),
                remediation="Give each listed HE record a limit type of 1 or 2 in dadger.rvN.",
            ),
            logger=_LOG,
        )
    if ez_cap_unmodeled:
        emit(
            Diagnostic(
                code="decomp-rhe-ez-cap-unmodeled",
                severity=Severity.WARNING,
                category="Special constraints",
                title=f"HE constraints looser than in DECOMP ({len(ez_cap_unmodeled)})",
                summary=(
                    "Each listed HE constraint counts a reservoir whose EZ record "
                    "caps the volume it stores in the REE. The converted "
                    "constraint counts the whole volume, so on the listed stages "
                    "it is met more easily than in DECOMP while that reservoir "
                    "is above its cap."
                ),
                table=DiagnosticTable(
                    columns=["Constraint", "Stages"],
                    rows=list(ez_cap_unmodeled),
                    caption="Stage ids are 0-based.",
                ),
            ),
            logger=_LOG,
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
    ez_fractions: Mapping[int, float],
    big_m: float,
    allocator: ConstraintIdAllocator | None = None,
) -> DecompGenericResult | None:
    """Emit every generic ``HE`` (RHE stored-energy) constraint.

    Iterates ``census.to_generic`` filtered to ``record.family == "HE"``.
    Each record's ``energy`` terms name a REE (``term.code``) with a CM ±1
    sign (``term.coefficient``); the participating shares for that REE are
    the ones :func:`~novomodelo_bridge.decomp.converters.rhe_shares.
    build_rhe_shares` builds over its stored-energy reservoirs operated in
    ``id_map``, with *ez_fractions* from the deck's ``EZ`` records. A record
    whose terms yield **no** participating reservoir at all is skipped whole
    (skip-not-partial, listed under ``decomp-rhe-no-storage-plants``) — never
    a partial constraint.

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

    The slack follows the record's violation flag (``HE`` field 9). At ``0``
    or blank the source model treats a violation as an infeasibility, so the
    slack is priced at *big_m*, as for every other hard constraint. At ``1``
    a violation is a priced resource: the penalty is
    ``record.he_meta.valor_penalidade`` (R$/MWh) when positive, else
    :data:`_RHE_DEFAULT_PENALTY` with a WARNING (mirroring the source model's
    ``penalty <= 0 -> 1000.0`` fallback). Either way it is ``per_unit``, a
    price per MWh of violation on weekly and monthly stages alike.

    Reads no ``dadger``; *hydro_to_ree* and *ez_fractions* arrive from the
    caller. *allocator* defaults to a fresh :class:`ConstraintIdAllocator`
    (ids from 0) when omitted; the pipeline threads one shared instance
    across every emitter. Returns ``None`` when no RHE constraint survives.
    """
    calendar = case.calendar

    rhe_shares = build_rhe_shares(effective, id_map, hydro_to_ree, ez_fractions)

    allocator = allocator or ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)
    per_stage_parameters: dict[str, list[float]] = {}
    expression_parameters: dict[str, str] = {}
    no_storage: list[list[object]] = []
    default_penalty: list[list[object]] = []
    unknown_tipo: list[list[object]] = []
    ez_cap_unmodeled: list[list[object]] = []

    for record in census.to_generic:
        if record.family != "HE":
            continue

        shares = [
            (term.coefficient, share)
            for term in record.terms
            for share in rhe_shares.by_ree.get(term.code, [])
        ]
        if not shares:
            rees = ", ".join(str(term.code) for term in record.terms)
            no_storage.append([record.constraint_id, rees])
            continue

        if record.he_meta is None or record.he_meta.violation_is_infeasible:
            penalty = big_m
        else:
            penalty = record.he_meta.valor_penalidade
            if penalty is None or penalty <= 0.0:
                declared = "missing" if penalty is None else f"{penalty:g}"
                default_penalty.append(
                    [record.constraint_id, declared, f"{_RHE_DEFAULT_PENALTY:g}"]
                )
                penalty = _RHE_DEFAULT_PENALTY

        tipo_limite = record.tipo_limite
        if tipo_limite not in (1, 2):
            declared_tipo = "missing" if tipo_limite is None else tipo_limite
            unknown_tipo.append([record.constraint_id, declared_tipo])

        limits = {
            stage_index: stage_bounds.lower[0]
            for stage_index, stage_bounds in sorted(record.bounds.items())
            if stage_bounds.lower[0] is not None
        }
        energy_shares = [(sign, rhe.share) for sign, rhe in shares]
        record_parameters: dict[str, list[float]] = {}
        if tipo_limite == 2:
            pct_name = rhe_pct_name(record.constraint_id)
            expression = stored_energy_floor(energy_shares, pct_name)
            lowers = dict.fromkeys(limits, 0.0)
            record_parameters[pct_name] = [
                limits.get(stage_index, 0.0) / 100.0
                for stage_index in range(len(calendar))
            ]
            floors = {
                stage_index: limite
                / 100.0
                * sum(sign * rhe.max_energy_mwh(stage_index) for sign, rhe in shares)
                for stage_index, limite in limits.items()
            }
        else:
            expression = stored_energy_lhs(energy_shares)
            lowers = {
                stage_index: limite * calendar[stage_index].total_hours
                for stage_index, limite in limits.items()
            }
            floors = lowers

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
        if added is None:
            continue
        # TRACKED NOVOMODELO-GAP (C23): a generic constraint cannot cap one volume
        # term at a constant, min(V, EZ·Vu); the share counts the whole volume
        # and the warning names the stages where that can loosen the floor.
        unmodeled = ez_cap_unmodeled_stages(shares, floors)
        if unmodeled:
            stages = format_stage_ranges(unmodeled)
            ez_cap_unmodeled.append([record.constraint_id, stages])
        per_stage_parameters.update(record_parameters)
        for _, rhe in shares:
            per_stage_parameters.update(
                {name: list(values) for name, values in rhe.constants.items()}
            )
        expression_parameters.update(share_parameters(s for _, s in energy_shares))

    if rhe_shares.couplings:
        _emit_rhe_hydraulic_coupling(rhe_shares.couplings, effective.plant_names())
    _emit_rhe_findings(
        no_storage=no_storage,
        default_penalty=default_penalty,
        unknown_tipo=unknown_tipo,
        ez_cap_unmodeled=ez_cap_unmodeled,
    )
    return _with_parameters(
        builder.result(), per_stage_parameters, expression_parameters
    )


def emit_itaipu_ande_generic(
    itaipu_id: int,
    ande_bus_id: int,
    carga_ande: Mapping[int, Sequence[float]],
    *,
    big_m: float,
    allocator: ConstraintIdAllocator,
) -> GenericConstraintResult | None:
    """Itaipu's generation on the ANDE bus held at or above the ANDE load.

    The source model subtracts the ``RI`` ``carga_ande`` load from the 50 Hz
    generation it delivers to the submarket, so the 50 Hz group alone serves
    it. The case adds the load to the bus the 50 Hz group sits on
    (*ande_bus_id*), where any plant could serve it, so this constraint keeps
    the group's generation there at or above the load in every stage and block
    with a positive load, soft at *big_m* like the ``RE`` constraints.

    *carga_ande* is ``{stage_index: [MW per block]}``, as
    :func:`~novomodelo_bridge.decomp.converters.libs_electrical.read_carga_ande`
    returns it. Returns ``None`` when no block carries a load.
    """
    builder = GenericConstraintBuilder(allocator)
    builder.add(
        name="RI_ANDE",
        description="Itaipu 50 Hz generation serving the RI ANDE load",
        expression=_hydro_generation_token(itaipu_id, ande_bus_id),
        slack=soft_slack(big_m, "per_hour"),
        slots=[
            (stage_id, block_id, load, None)
            for stage_id, loads in sorted(carga_ande.items())
            for block_id, load in enumerate(loads)
            if load > 0.0
        ],
    )
    return builder.result()
