"""Lower single-term, unit-coefficient special constraints to entity bounds.

``constraint_registers.read_constraints`` splits every special constraint
into ``to_bounds`` (single-term, ``|coefficient|==1``, bounded-variable) and
``to_generic``. This module turns each ``to_bounds`` record into
:class:`~novomodelo_bridge.decomp.bounds_accumulator.BoundContribution`\\ s that
``bounds_accumulator.resolve``/``build_bound_tables`` can later fold into the
novomodelo bound parquet rows. It produces contributions only — it does not write
parquet, call ``resolve``/``build_bound_tables``, or touch the pipeline.

``single_term_bound_contributions`` is the stable public seam: it dispatches
each record on its ``family`` (``"RE"`` and ``"HQ"``) and raises
on anything else, so an unhandled family fails loud instead of silently
dropping a bound. An ``"HV"`` record is skipped there and lowered by
:func:`single_term_hv_contributions`: ``VARM`` to absolute storage, the volume
tipos (``VDEF``/``VDES``/``VBOM``) to absolute hm3 contributions, unshifted.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from novomodelo_bridge.core.diagnostics import Diagnostic, Severity, emit
from novomodelo_bridge.core.generic_constraint_builder import is_bounded
from novomodelo_bridge.decomp.bounds_accumulator import BoundContribution
from novomodelo_bridge.decomp.constraint_registers import sided_bounds
from novomodelo_bridge.decomp.converters.cadastro import effective_storage_range
from novomodelo_bridge.decomp.converters.constraints import (
    hv_terms_convertible,
    offset_if_bounded,
)
from novomodelo_bridge.decomp.converters.hydro import split_plant_group_id

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.constraint_registers import (
        ConstraintCensus,
        ConstraintRecord,
    )
    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
    from novomodelo_bridge.decomp.id_map import DecompIdMap
    from novomodelo_bridge.decomp.temporal import OperativeStage


#: RHQ ``CQ.tipo`` flow -> novomodelo hydro bound axis, mirroring the reader's
#: ``constraint_registers._BOUNDS_AXIS`` QDEF/QTUR/QDES/QVER entries. Keep the
#: two mappings consistent for a *hydro* flow axis; a future hydro flow
#: ``tipo`` gaining a bound axis updates both. ``QBOM`` is the one asymmetry:
#: it lives in ``_BOUNDS_AXIS`` but is a *pumping*-entity
#: axis, not a hydro one, so it is dispatched to
#: :func:`_qbom_pumping_contributions` instead of living in this mapping.
_HQ_AXIS_BY_VARIABLE: dict[str, str] = {
    "QDEF": "outflow",
    "QTUR": "turbined",
    "QDES": "diversion",
    "QVER": "spillage",
}

#: RHV ``CV.tipo`` -> the ``(family, axis)`` its single-term bound lands on;
#: the ``*_volume`` axes are hm3 over the stage.
_HV_AXIS_BY_VARIABLE: dict[str, tuple[str, str]] = {
    "VARM": ("hydro", "storage"),
    "VDEF": ("hydro", "outflow_volume"),
    "VDES": ("hydro", "diversion_volume"),
    "VBOM": ("pumping", "volume"),
}

#: Contributor label of the physical-range clamp, so a conflict names it.
_STORAGE_RANGE = "storage-range"

#: Contributor label of the ``>= 0`` clamp a negative volume upper trips on.
_NON_NEGATIVE_VOLUME = "non-negative volume"


@dataclasses.dataclass(frozen=True)
class HydroCapacities:
    """One hydro's declared ``generation`` envelope from ``hydros.json``.

    ``max_generation_mw``/``max_turbined_m3s`` are read the same way novomodelo
    rule 43 (``emission_checks.check_hydro_bounds_no_raising``) reads
    ``hydros.json``'s ``generation`` block — the exact scalars a single-term
    bound's upper-bound clamp compares against on each of the two rule-43-
    guarded axes.
    """

    max_generation_mw: float
    max_turbined_m3s: float


#: Relative tolerance below which a source ceiling sitting above the declared
#: capacity is treated as float-representation noise (~8x float32 epsilon). The
#: bound is still clamped so novomodelo rule 43 holds, but no diagnostic is emitted:
#: a float32 round-trip (e.g. an 11000 MW RE ceiling vs a 10999.998 MW declared
#: capacity, ~1.4e-7 relative) is not a real cross-source inconsistency.
_CLAMP_REPORT_REL_TOL = 1e-6


def _clamp_upper_to_capacity(
    contributions: list[BoundContribution], cap: float
) -> tuple[list[BoundContribution], list[float]]:
    """Clamp every contribution's ``upper`` to *cap*, ``lower`` untouched.

    Shared by the RE ``generation`` clamp and the HQ ``QTUR``/``turbined``
    clamp — both guard a novomodelo rule-43 axis the same way: a contribution
    whose upper is already at or below *cap* passes through unchanged;
    one above it is replaced (:func:`dataclasses.replace`) with *cap*.

    A ceiling above *cap* is **always** clamped (novomodelo rule 43 rejects any
    upper above the declared capacity, even by a float ULP), but it is only
    returned in the reported ``ceilings`` list — the caller's diagnostic
    trigger — when it exceeds *cap* by more than ``_CLAMP_REPORT_REL_TOL``
    relative. So a genuine cross-source mismatch (a looser nameplate vs.
    head-derated capacity, MW-scale) is surfaced while float-representation
    noise clamps silently. Returns the clamped contributions alongside those
    materially-lowered ceilings (empty when nothing needed reporting).
    """
    report_threshold = cap * (1.0 + _CLAMP_REPORT_REL_TOL)
    clamped: list[BoundContribution] = []
    ceilings: list[float] = []
    for contribution in contributions:
        if contribution.upper is None or contribution.upper <= cap:
            clamped.append(contribution)
            continue
        if contribution.upper > report_threshold:
            ceilings.append(contribution.upper)
        clamped.append(dataclasses.replace(contribution, upper=cap))
    return clamped, ceilings


def _per_block_contributions(
    record: ConstraintRecord,
    family: str,
    entity_id: int,
    axis: str,
    contributor: str,
    calendar: Sequence[OperativeStage],
) -> list[BoundContribution]:
    """The shared per-block emission loop for a single-term entity bound.

    Iterates every declared stage in ``record.bounds`` and, within it, every
    block up to the stage's *real* block count
    (``len(calendar[stage_index].block_hours)``) — never the raw
    ``StageBounds`` slot count, which LU carries up to 5 slots wide
    regardless of the stage's actual block count. A ``block_id`` at or past
    ``len(stage_bounds.lower)`` is skipped (no declared slot for it), and a
    block whose sided bounds are both ``None`` emits no contribution.
    """
    contributions: list[BoundContribution] = []
    term = record.terms[0]
    for stage_index, stage_bounds in record.bounds.items():
        coefficient = term.coefficient_at(stage_index)
        n_blocks = min(len(calendar[stage_index].block_hours), len(stage_bounds.lower))
        for block_id in range(n_blocks):
            entity_lower, entity_upper = sided_bounds(
                stage_bounds.lower[block_id],
                stage_bounds.upper[block_id],
                coefficient,
            )
            if entity_lower is None and entity_upper is None:
                continue
            contributions.append(
                BoundContribution(
                    family=family,
                    entity_id=entity_id,
                    stage_id=stage_index,
                    block_id=block_id,
                    axis=axis,
                    lower=entity_lower,
                    upper=entity_upper,
                    contributor=contributor,
                )
            )
    return contributions


def _re_generation_contributions(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    hydro_capacities: Mapping[int, HydroCapacities],
) -> list[BoundContribution]:
    """One RE single-hydro-generation constraint -> hydro ``generation`` bounds.

    The RE ceiling and the plant's own declared ``max_generation_mw`` (in
    *hydro_capacities*, keyed the same way novomodelo rule 43 —
    ``emission_checks.check_hydro_bounds_no_raising`` — reads it from
    ``hydros.json``) are two independent sources that can disagree: on both
    real decks probed (a monthly and a weekly deck), BELO MONTE carries an
    RE ceiling of 11000 MW above its own declared, head-derated capacity.
    novomodelo rule 43 rejects any bound-table upper above an entity's declared
    capacity, so every contribution's ``upper`` is clamped
    (:func:`_clamp_upper_to_capacity`) to ``min(ceiling, capacity)`` — the
    declared capacity is authoritative, and a looser RE ceiling becomes
    non-binding, LP-neutral. Only ``upper`` is clamped; ``lower`` passes
    through unchanged (rule 43 only guards a raised ceiling). A
    ``decomp-re-generation-clamped`` diagnostic is emitted once per record
    whenever the clamp actually changes at least one contribution's upper —
    a ceiling above capacity is a real cross-source inconsistency worth
    surfacing, not silently absorbing.
    """
    code = record.terms[0].code
    hydro_id = id_map.hydro_id(code)
    cap = hydro_capacities[hydro_id].max_generation_mw
    contributions = _per_block_contributions(
        record, "hydro", hydro_id, "generation", f"RE_{record.constraint_id}", calendar
    )
    clamped_contributions, clamp_ceilings = _clamp_upper_to_capacity(contributions, cap)
    _emit_re_generation_clamped(
        f"Hydro {hydro_id}", code, [record.constraint_id], clamp_ceilings, cap
    )
    return clamped_contributions


def _emit_re_generation_clamped(
    subject: str,
    code: int,
    constraint_ids: Sequence[int],
    ceilings: list[float],
    cap: float,
) -> None:
    """Report RE generation ceilings clamped to *subject*'s declared capacity;
    a no-op when *ceilings* is empty."""
    if not ceilings:
        return
    constraints = (
        f"constraint {constraint_ids[0]}"
        if len(constraint_ids) == 1
        else f"constraints {', '.join(str(c) for c in constraint_ids)}"
    )
    emit(
        Diagnostic(
            code="decomp-re-generation-clamped",
            severity=Severity.WARNING,
            category="Special constraints",
            title="RE generation ceiling clamped to declared capacity",
            summary=(
                f"{subject} (source code {code}) carries an RE "
                f"generation ceiling ({constraints}) "
                f"of {max(ceilings)} MW across "
                f"{len(ceilings)} stage/block cell(s), above its "
                f"declared max_generation_mw of {cap} MW; the emitted "
                "upper bound is clamped down to the declared capacity."
            ),
            remediation=(
                "The RE ceiling is looser than the declared capacity, so "
                "the clamp is LP-neutral. Reconcile the "
                "two sources (e.g. nameplate vs. head-derated capacity) "
                "if this mismatch is unexpected."
            ),
        )
    )


def split_plant_group_contributions(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    census: ConstraintCensus,
    group_capacities: Mapping[tuple[int, int], float],
) -> dict[int, list[BoundContribution]]:
    """Frequency-tagged single-term RE records -> unit-group ``generation`` bounds.

    An ``FU`` term's frequency names one half of the split plant, so its bound
    lands on that half's unit group (family ``hydro_unit_group``), never on
    the plant: a plant-level bound would cap the sum of both halves at one
    half's limit. Each upper is clamped to the group's declared
    ``max_generation_mw`` in *group_capacities*, keyed ``(hydro_id,
    group_id)``, since novomodelo rejects a group bound above it.

    Returns the contributions keyed by hydro id, because a group id is scoped
    within its plant.

    Raises
    ------
    ValueError
        From :func:`~novomodelo_bridge.decomp.converters.hydro.split_plant_group_id`
        when a tagged term names a plant or frequency with no unit group.
    KeyError
        When *group_capacities* has no entry for the resolved group.
    """
    calendar = case.calendar
    by_hydro: dict[int, list[BoundContribution]] = {}
    clamped: dict[tuple[int, int, int], tuple[list[int], list[float]]] = {}
    for record in census.to_bounds:
        if not _is_frequency_tagged_generation(record):
            continue
        term = record.terms[0]
        assert term.frequency is not None
        hydro_id = id_map.hydro_id(term.code)
        group_id = split_plant_group_id(term.code, term.frequency)
        contributions, ceilings = _clamp_upper_to_capacity(
            _per_block_contributions(
                record,
                "hydro_unit_group",
                group_id,
                "generation",
                f"RE_{record.constraint_id}",
                calendar,
            ),
            group_capacities[(hydro_id, group_id)],
        )
        by_hydro.setdefault(hydro_id, []).extend(contributions)
        if ceilings:
            ids, all_ceilings = clamped.setdefault(
                (hydro_id, group_id, term.code), ([], [])
            )
            ids.append(record.constraint_id)
            all_ceilings.extend(ceilings)
    for (hydro_id, group_id, code), (ids, ceilings) in sorted(clamped.items()):
        _emit_re_generation_clamped(
            f"Unit group {group_id} of hydro {hydro_id}",
            code,
            ids,
            ceilings,
            group_capacities[(hydro_id, group_id)],
        )
    return by_hydro


def _is_frequency_tagged_generation(record: ConstraintRecord) -> bool:
    return (
        record.family == "RE"
        and record.terms[0].variable == "generation"
        and record.terms[0].frequency is not None
    )


def _ft_thermal_contributions(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> list[BoundContribution]:
    """One RE single-thermal-generation constraint -> thermal ``generation`` bounds.

    Mirrors :func:`_re_generation_contributions`, but the term's ``code`` is a
    thermal ``codigo_usina`` (an ``FT`` term), so it resolves via
    ``id_map.thermal_id`` rather than ``id_map.hydro_id``.

    Unlike the hydro path, this does **not** clamp its ``upper`` to a
    declared thermal capacity: both real decks probed (rv3 and
    decomp-set-24-rv0) carry zero ``RE``->``thermal_generation`` records —
    ``FT`` never appears as a single-term-bound RE term on either — so there
    is no real ceiling-above-capacity case to clamp, and adding one here
    would be speculative code with nothing to exercise it. Revisit if a
    future deck is found to declare one.
    """
    thermal_id = id_map.thermal_id(record.terms[0].code)
    return _per_block_contributions(
        record,
        "thermal",
        thermal_id,
        "generation",
        f"RE_{record.constraint_id}",
        calendar,
    )


def _hq_flow_contributions(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    hydro_capacities: Mapping[int, HydroCapacities],
) -> list[BoundContribution]:
    """One RHQ single-flow constraint -> a hydro flow-band bound.

    Routes by ``_HQ_AXIS_BY_VARIABLE[term.variable]``: ``QDEF`` lowers to
    ``outflow``, ``QTUR`` to ``turbined``, ``QDES`` (diverted flow) to
    ``diversion``, and ``QVER`` (spilled flow) to ``spillage`` — a family-
    agnostic dict lookup, not a hardcoded branch, so a QDEF and a QDES
    constraint (or any other pair of these four) on the same plant land on
    different axes and both survive (the accumulator keys on axis).

    ``QTUR``'s emitted ``turbined`` upper is clamped
    (:func:`_clamp_upper_to_capacity`) to the plant's declared
    ``max_turbined_m3s`` (in *hydro_capacities*) — mirroring
    :func:`_re_generation_contributions`'s ``max_generation_mw`` clamp for
    the identical reason: ``max_turbined_m3s`` is head-derated too, so a
    source ceiling can exceed it just as an RE ceiling can exceed
    ``max_generation_mw`` (rv3's hydro 17 sits 0.1% below its own declared
    value — close enough that a slightly looser source ceiling would trip
    novomodelo rule 43, which guards both columns). Only ``QTUR``/``turbined`` is
    clamped: ``QDEF``/``QDES``/``QVER`` (outflow/diversion/spillage) are not
    rule-43 axes, so they pass through :func:`_per_block_contributions`
    unmodified. A ``decomp-qtur-turbined-clamped`` diagnostic is emitted
    once per record whenever the clamp actually changes at least one
    contribution's upper.
    """
    term = record.terms[0]
    axis = _HQ_AXIS_BY_VARIABLE[term.variable]
    hydro_id = id_map.hydro_id(term.code)
    contributions = _per_block_contributions(
        record, "hydro", hydro_id, axis, f"HQ_{record.constraint_id}", calendar
    )
    if axis != "turbined":
        return contributions

    cap = hydro_capacities[hydro_id].max_turbined_m3s
    clamped_contributions, clamp_ceilings = _clamp_upper_to_capacity(contributions, cap)

    if clamp_ceilings:
        emit(
            Diagnostic(
                code="decomp-qtur-turbined-clamped",
                severity=Severity.WARNING,
                category="Special constraints",
                title="QTUR turbined ceiling clamped to declared capacity",
                summary=(
                    f"Hydro {hydro_id} (source code {term.code}) carries a "
                    f"QTUR turbined ceiling (constraint {record.constraint_id}) "
                    f"of {max(clamp_ceilings)} m3/s across "
                    f"{len(clamp_ceilings)} stage/block cell(s), above its "
                    f"declared max_turbined_m3s of {cap} m3/s; the emitted "
                    "upper bound is clamped down to the declared capacity."
                ),
                remediation=(
                    "The QTUR ceiling is looser than the plant's declared "
                    "capacity, so the clamp is LP-neutral. Reconcile the "
                    "two sources (e.g. nameplate vs. head-derated capacity) "
                    "if this mismatch is unexpected."
                ),
            )
        )
    return clamped_contributions


def _qbom_pumping_contributions(
    record: ConstraintRecord,
    pumping_station_ids: Mapping[int, int],
    calendar: Sequence[OperativeStage],
) -> list[BoundContribution]:
    """One RHQ single ``QBOM`` constraint -> pumping ``flow`` bounds.

    A ``QBOM`` term's ``code`` is a pumping-station ``codigo_usina``, not a
    hydro code, so it resolves through *pumping_station_ids* rather than
    ``id_map.hydro_id``. A code absent from *pumping_station_ids* emits a
    ``WARNING`` diagnostic and is skipped entirely — skip-not-partial,
    mirroring ``constraints.py::_emit_pumping_term_no_station`` — rather than
    raising a raw ``KeyError`` or falling back to a generic constraint.
    """
    code = record.terms[0].code
    if code not in pumping_station_ids:
        emit(
            Diagnostic(
                code="decomp-rhq-qbom-no-station",
                severity=Severity.WARNING,
                category="Special constraints",
                title="RHQ QBOM term has no matching pumping station",
                summary=(
                    f"HQ constraint {record.constraint_id} carries a QBOM "
                    f"term for plant code {code}, which has no matching "
                    "pumping station; the constraint is skipped."
                ),
                remediation=(
                    f"HQ constraint {record.constraint_id} is skipped; check "
                    f"that plant code {code} is declared as a pumping "
                    "station."
                ),
            )
        )
        return []
    return _per_block_contributions(
        record,
        "pumping",
        pumping_station_ids[code],
        "flow",
        f"HQ_{record.constraint_id}",
        calendar,
    )


def single_term_bound_contributions(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    census: ConstraintCensus,
    pumping_station_ids: Mapping[int, int],
    hydro_capacities: Mapping[int, HydroCapacities],
) -> list[BoundContribution]:
    """Lower every ``census.to_bounds`` record to its entity bound contributions.

    Dispatches on ``record.family``: ``"RE"`` lowers to a ``generation`` bound
    on either the hydro (``FU``/``generation``) or the thermal
    (``FT``/``thermal_generation``) entity named by its single term, except a
    frequency-tagged ``FU`` record, which bounds a unit group and is lowered
    by :func:`split_plant_group_contributions` instead — any
    other bounded RE variable raises ``ValueError`` naming it. ``"HQ"``
    further dispatches on its single term's variable: ``QBOM`` lowers to a
    pumping ``flow`` bound via :func:`_qbom_pumping_contributions` (resolved
    through *pumping_station_ids*, never ``id_map.hydro_id`` — its ``code``
    is a pumping-station code, not a hydro one); everything else
    (``QDEF``/``QTUR``/``QDES``/``QVER``) lowers to a hydro
    ``outflow``/``turbined``/``diversion``/``spillage`` bound via
    :func:`_hq_flow_contributions`. An ``"HV"`` record is skipped, lowered by
    :func:`single_term_hv_contributions`; any other family raises
    ``ValueError`` naming it. *pumping_station_ids* feeds only the ``HQ``
    ``QBOM`` path — required, not defaulted, so a forgotten wiring fails loud instead of
    silently dropping every QBOM bound; *hydro_capacities* feeds the ``FU``/
    ``generation`` RE path's ``max_generation_mw`` clamp (see
    :func:`_re_generation_contributions`) and the ``HQ``/``QTUR``/
    ``turbined`` path's ``max_turbined_m3s`` clamp (see
    :func:`_hq_flow_contributions`) — both novomodelo rule-43-guarded axes, both
    required-not-defaulted for the same fail-loud reason: a hydro id the map
    does not cover is a wiring bug, not a data gap, so it raises ``KeyError``
    rather than skipping the clamp. Neither the ``QDEF``/``QDES``/``QVER``
    HQ paths nor the ``FT``/``thermal_generation`` RE path read it.

    Raises
    ------
    KeyError
        Propagated from ``id_map.hydro_id``/``id_map.thermal_id`` when an
        RE/HQ term names a code the id map does not know (a real
        reader/id-map mismatch), or from *hydro_capacities* when it has no
        entry for a hydro id the id map just resolved (a wiring mismatch
        between the two maps' sources).
    ValueError
        When a ``to_bounds`` record carries a family other than ``"RE"``/
        ``"HQ"``/``"HV"``, or an ``"RE"`` record whose single term's variable
        is neither ``"generation"`` nor ``"thermal_generation"``.
    """
    calendar = case.calendar
    contributions: list[BoundContribution] = []
    for record in census.to_bounds:
        if _is_frequency_tagged_generation(record) or record.family == "HV":
            continue
        if record.family == "RE":
            variable = record.terms[0].variable
            if variable == "generation":
                contributions.extend(
                    _re_generation_contributions(
                        record, id_map, calendar, hydro_capacities
                    )
                )
            elif variable == "thermal_generation":
                contributions.extend(
                    _ft_thermal_contributions(record, id_map, calendar)
                )
            else:
                raise ValueError(
                    "single_term_bound_contributions: unhandled RE bounded "
                    f"variable {variable!r}"
                )
        elif record.family == "HQ":
            if record.terms[0].variable == "QBOM":
                contributions.extend(
                    _qbom_pumping_contributions(record, pumping_station_ids, calendar)
                )
            else:
                contributions.extend(
                    _hq_flow_contributions(record, id_map, calendar, hydro_capacities)
                )
        else:
            raise ValueError(
                f"single_term_bound_contributions: unhandled family {record.family!r}"
            )
    return contributions


def _stage_sides(
    record: ConstraintRecord,
) -> Iterator[tuple[int, float | None, float | None]]:
    """Each stage of a single-term, one-slot *record* with a bounded side, as
    ``(stage, lower, upper)`` on the entity after the coefficient-sign map."""
    term = record.terms[0]
    for stage, bounds in record.bounds.items():
        lower, upper = sided_bounds(
            bounds.lower[0], bounds.upper[0], term.coefficient_at(stage)
        )
        if is_bounded(lower) or is_bounded(upper):
            yield stage, lower, upper


def single_term_hv_contributions(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    census: ConstraintCensus,
    effective: EffectiveCadastro,
    pumping_station_ids: Mapping[int, int],
    diversion_hydro_ids: frozenset[int],
) -> list[BoundContribution]:
    """Lower every ``"HV"`` record of ``census.to_bounds`` to stage-level contributions.

    A ``VARM`` ``LV`` limit is relative to the plant's useful volume above the
    stage floor, so each bounded side becomes absolute hm³ by adding the
    stage's effective floor (:func:`offset_if_bounded`: a ``None`` or
    ``±1e21`` side is never shifted). A second ``storage-range`` contribution
    ``[floor, ceiling]`` on the same stage lets the accumulator clamp a looser
    side to the physical range and write both storage sides.

    A volume tipo (``VDEF``/``VDES``/``VBOM``) is already absolute hm³ over
    the stage and is never shifted. Its volume is non-negative: a negative
    lower is dropped, a ``±0.0`` side is written as ``0.0``, and a negative
    upper adds a ``non-negative volume`` contribution ``[0.0, None]`` that the
    accumulator raises on at resolve. ``VBOM`` resolves its entity through
    *pumping_station_ids*, the other tipos through ``id_map.hydro_id``.

    A stage whose two sides are unbounded emits nothing; a record
    :func:`hv_terms_convertible` refuses is skipped whole, the guard having
    emitted the diagnostic. An HV side no stage range meets is left to the
    accumulator, which raises on resolve. *case* keeps the signature of the
    sibling lowerings; a stage-level bound reads no calendar.

    Raises
    ------
    ValueError
        When a convertible record's variable has no entry in
        ``_HV_AXIS_BY_VARIABLE`` (a wiring bug, not a data gap).
    """
    contributions: list[BoundContribution] = []
    for record in census.to_bounds:
        if record.family != "HV" or not hv_terms_convertible(
            record, id_map, effective, pumping_station_ids, diversion_hydro_ids
        ):
            continue
        term = record.terms[0]
        if term.variable not in _HV_AXIS_BY_VARIABLE:
            raise ValueError(
                f"single_term_hv_contributions: unhandled HV variable {term.variable!r}"
            )
        family, axis = _HV_AXIS_BY_VARIABLE[term.variable]
        entity_id = (
            pumping_station_ids[term.code]
            if family == "pumping"
            else id_map.hydro_id(term.code)
        )
        for stage, lower, upper in _stage_sides(record):
            if term.variable == "VARM":
                floor, ceiling = effective_storage_range(effective, term.code, stage)
                lower, upper = (
                    offset_if_bounded(lower, floor),
                    offset_if_bounded(upper, floor),
                )
            else:
                if is_bounded(lower) and lower < 0.0:
                    lower = None
                # `+ 0.0` rewrites a `-0.0` side (a -1 coefficient negates 0.0) as 0.0.
                lower, upper = (
                    offset_if_bounded(lower, 0.0),
                    offset_if_bounded(upper, 0.0),
                )
            limit = BoundContribution(
                family=family,
                entity_id=entity_id,
                stage_id=stage,
                block_id=None,
                axis=axis,
                lower=lower,
                upper=upper,
                contributor=f"HV_{record.constraint_id}",
            )
            contributions.append(limit)
            if term.variable == "VARM":
                contributions.append(
                    dataclasses.replace(
                        limit, lower=floor, upper=ceiling, contributor=_STORAGE_RANGE
                    )
                )
            elif is_bounded(upper) and upper < 0.0:
                contributions.append(
                    dataclasses.replace(
                        limit,
                        lower=0.0,
                        upper=None,
                        contributor=_NON_NEGATIVE_VOLUME,
                    )
                )
    return contributions
