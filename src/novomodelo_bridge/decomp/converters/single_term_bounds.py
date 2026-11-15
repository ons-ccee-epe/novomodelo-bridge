"""Lower single-term, unit-coefficient special constraints to entity bounds.

``constraint_registers.read_constraints`` splits every special constraint
into ``to_bounds`` (single-term, ``|coefficient|==1``, bounded-variable) and
``to_generic``. This module turns each ``to_bounds`` record into
:class:`~novomodelo_bridge.decomp.bounds_accumulator.BoundContribution`\\ s that
``bounds_accumulator.resolve``/``build_bound_tables`` can later fold into the
novomodelo bound parquet rows. It produces contributions only — it does not write
parquet, call ``resolve``/``build_bound_tables``, or touch the pipeline.

``single_term_bound_contributions`` is the stable public seam: it dispatches
each record on its ``family`` (``"RE"``, ``"HQ"``, and ``"HV"``) and raises
on anything else, so an unhandled family fails loud instead of silently
dropping a bound.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
    format_stage_ranges,
)
from novomodelo_bridge.core.tolerances import beyond_written_precision
from novomodelo_bridge.decomp.bounds_accumulator import BoundContribution
from novomodelo_bridge.decomp.constraint_registers import sided_bounds
from novomodelo_bridge.decomp.converters.cadastro import effective_storage_range
from novomodelo_bridge.decomp.converters.hydro import split_plant_group_id

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.constraint_registers import (
        ConstraintCensus,
        ConstraintRecord,
    )
    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
    from novomodelo_bridge.decomp.id_map import DecompIdMap
    from novomodelo_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)

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


def _clamp_upper_to_capacity(
    contributions: list[BoundContribution], cap: float
) -> tuple[list[BoundContribution], list[BoundContribution]]:
    """Clamp every unit-group contribution's ``upper`` to the group's declared
    *cap*, ``lower`` untouched, since novomodelo rejects a group bound above it.

    Returns the clamped contributions and the originals whose ceiling sat past
    the deck's written precision above *cap*
    (:func:`~novomodelo_bridge.core.tolerances.beyond_written_precision`), the
    ones the note reports; a ceiling nearer *cap* is clamped silently.
    """
    clamped: list[BoundContribution] = []
    reported: list[BoundContribution] = []
    for contribution in contributions:
        if contribution.upper is None or contribution.upper <= cap:
            clamped.append(contribution)
            continue
        if beyond_written_precision(contribution.upper, cap):
            reported.append(contribution)
        clamped.append(dataclasses.replace(contribution, upper=cap))
    return clamped, reported


def _ceiling_clamp_row(
    code: int,
    name: str,
    constraint_ids: Sequence[int],
    reported: Sequence[BoundContribution],
    cap: float,
) -> list[object]:
    """One ``[code, name, constraints, stages, deck ceiling, capacity]`` row of
    the unit-group clamp table; the deck ceiling is the highest one clamped."""
    return [
        code,
        name,
        ", ".join(str(c) for c in constraint_ids),
        format_stage_ranges(c.stage_id for c in reported),
        max(c.upper for c in reported if c.upper is not None),
        cap,
    ]


def _unit_group_clamp_diagnostic(rows: list[list[object]]) -> Diagnostic:
    """INFO for RE ceilings above a unit group's declared capacity, capped at
    it: the group cannot exceed its capacity, so no result changes."""
    return Diagnostic(
        code="decomp-re-unit-group-generation-clamped",
        severity=Severity.INFO,
        category="Special constraints",
        title=f"RE ceilings above unit-group capacity capped ({len(rows)})",
        summary=(
            "The listed RE unit-group generation ceilings sit above the "
            "capacity they bound and were capped at it. That capacity is "
            "already a hard limit, so the cap leaves the results unchanged."
        ),
        table=DiagnosticTable(
            columns=[
                "Code",
                "Name",
                "Constraints",
                "Stages",
                "Deck ceiling (MW)",
                "Capacity (MW)",
            ],
            rows=rows,
            justify=["right", "left", "left", "left", "right", "right"],
            caption="Stage ids are 0-based; the deck ceiling shown is the highest.",
        ),
    )


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
        n_blocks = len(calendar[stage_index].block_hours)
        for block_id in range(n_blocks):
            if block_id >= len(stage_bounds.lower):
                continue
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
) -> list[BoundContribution]:
    """One RE single-hydro-generation constraint -> hydro ``generation`` bounds.

    The ceiling can sit above the plant's declared ``max_generation_mw``
    (BELO MONTE's RE writes 11000 MW over 10999.9984 MW of units); the
    pipeline holds the merged bound to that capacity afterwards
    (:func:`~novomodelo_bridge.core.emission_checks.clamp_hydro_bounds_to_declared`).
    """
    hydro_id = id_map.hydro_id(record.terms[0].code)
    return _per_block_contributions(
        record, "hydro", hydro_id, "generation", f"RE_{record.constraint_id}", calendar
    )


def split_plant_group_contributions(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    census: ConstraintCensus,
    group_capacities: Mapping[tuple[int, int], float],
    plant_names: Mapping[int, str],
) -> dict[int, list[BoundContribution]]:
    """Frequency-tagged single-term RE records -> unit-group ``generation`` bounds.

    An ``FU`` term's frequency names one half of the split plant, so its bound
    lands on that half's unit group (family ``hydro_unit_group``), never on
    the plant: a plant-level bound would cap the sum of both halves at one
    half's limit. Each upper is clamped to the group's declared
    ``max_generation_mw`` in *group_capacities*, keyed ``(hydro_id,
    group_id)``, since novomodelo rejects a group bound above it; one
    ``decomp-re-unit-group-generation-clamped`` note names, by *plant_names*,
    every group whose ceiling was capped past the deck's written precision.

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
    clamped: dict[
        tuple[int, int, int, float], tuple[list[int], list[BoundContribution]]
    ] = {}
    for record in census.to_bounds:
        if not _is_frequency_tagged_generation(record):
            continue
        term = record.terms[0]
        assert term.frequency is not None
        hydro_id = id_map.hydro_id(term.code)
        group_id = split_plant_group_id(term.code, term.frequency)
        contributions, reported = _clamp_upper_to_capacity(
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
        if reported:
            ids, all_reported = clamped.setdefault(
                (hydro_id, group_id, term.code, term.frequency), ([], [])
            )
            ids.append(record.constraint_id)
            all_reported.extend(reported)
    if clamped:
        rows = [
            _ceiling_clamp_row(
                code,
                f"{plant_names[hydro_id]} ({frequency:g} Hz units)",
                ids,
                reported,
                group_capacities[(hydro_id, group_id)],
            )
            for (hydro_id, group_id, code, frequency), (ids, reported) in sorted(
                clamped.items()
            )
        ]
        emit(_unit_group_clamp_diagnostic(rows), logger=_LOG)
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

    Unlike a hydro ceiling, a thermal one is not held to a declared capacity:
    both real decks probed (rv3 and decomp-set-24-rv0) carry zero
    ``RE``->``thermal_generation`` records — ``FT`` never appears as a
    single-term-bound RE term on either — so there is no real
    ceiling-above-capacity case to clamp, and adding one would be speculative
    code with nothing to exercise it. Revisit if a future deck is found to
    declare one.
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
) -> list[BoundContribution]:
    """One RHQ single-flow constraint -> a hydro flow-band bound.

    Routes by ``_HQ_AXIS_BY_VARIABLE[term.variable]``: ``QDEF`` lowers to
    ``outflow``, ``QTUR`` to ``turbined``, ``QDES`` (diverted flow) to
    ``diversion``, and ``QVER`` (spilled flow) to ``spillage`` — a family-
    agnostic dict lookup, not a hardcoded branch, so a QDEF and a QDES
    constraint (or any other pair of these four) on the same plant land on
    different axes and both survive (the accumulator keys on axis). Like an
    RE ceiling, a ``QTUR`` ceiling above the plant's declared
    ``max_turbined_m3s`` is held to it after the merge.
    """
    term = record.terms[0]
    hydro_id = id_map.hydro_id(term.code)
    return _per_block_contributions(
        record,
        "hydro",
        hydro_id,
        _HQ_AXIS_BY_VARIABLE[term.variable],
        f"HQ_{record.constraint_id}",
        calendar,
    )


def _qbom_pumping_contributions(
    record: ConstraintRecord,
    pumping_station_ids: Mapping[int, int],
    calendar: Sequence[OperativeStage],
) -> list[BoundContribution]:
    """One RHQ single ``QBOM`` constraint -> pumping ``flow`` bounds.

    A ``QBOM`` term's ``code`` is a pumping-station ``codigo_usina``, not a
    hydro code, so it resolves through *pumping_station_ids* rather than
    ``id_map.hydro_id``; the census screening
    (:func:`~novomodelo_bridge.decomp.constraint_registers.drop_unknown_entities`)
    has already left out a code with no station.
    """
    code = record.terms[0].code
    return _per_block_contributions(
        record,
        "pumping",
        pumping_station_ids[code],
        "flow",
        f"HQ_{record.constraint_id}",
        calendar,
    )


def _hv_storage_contributions(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    effective: EffectiveCadastro,
) -> list[BoundContribution]:
    """One RHV single ``VARM`` constraint -> hydro ``storage`` bounds.

    Storage is a stage-level axis (unlike RE/RHQ's per-block axes), so every
    contribution carries ``block_id=None`` — the accumulator raises on a
    ``block_id`` reaching the ``storage`` axis, so this does not route through
    :func:`_per_block_contributions`.

    The source model's ``LV`` limits are relative to the plant's useful
    volume; novomodelo's ``min/max_storage_hm3`` are absolute. The sign map
    (:func:`sided_bounds`) runs first, then each surviving side is added to
    the per-stage effective floor (:func:`~novomodelo_bridge.decomp.converters.
    cadastro.effective.effective_storage_range`'s floor element, which
    honours per-stage ``AC VOLMIN``/``VOLMAX`` overrides and the
    run-of-river ``D`` collapse) to reach the absolute bound.

    The census screening
    (:func:`~novomodelo_bridge.decomp.constraint_registers.drop_unknown_entities`)
    has already left out a plant the case does not operate.
    """
    code = record.terms[0].code
    hydro_id = id_map.hydro_id(code)
    term = record.terms[0]
    contributions: list[BoundContribution] = []
    for stage_index, stage_bounds in record.bounds.items():
        rel_lower, rel_upper = sided_bounds(
            stage_bounds.lower[0],
            stage_bounds.upper[0],
            term.coefficient_at(stage_index),
        )
        floor = effective_storage_range(effective, code, stage_index)[0]
        abs_lower = None if rel_lower is None else floor + rel_lower
        abs_upper = None if rel_upper is None else floor + rel_upper
        if abs_lower is None and abs_upper is None:
            continue
        contributions.append(
            BoundContribution(
                family="hydro",
                entity_id=hydro_id,
                stage_id=stage_index,
                block_id=None,
                axis="storage",
                lower=abs_lower,
                upper=abs_upper,
                contributor=f"HV_{record.constraint_id}",
            )
        )
    return contributions


def single_term_bound_contributions(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    census: ConstraintCensus,
    pumping_station_ids: Mapping[int, int],
    effective: EffectiveCadastro,
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
    :func:`_hq_flow_contributions`. ``"HV"`` lowers to hydro ``storage``
    bounds. Any other family raises ``ValueError`` naming it.
    *pumping_station_ids* feeds only the ``HQ`` ``QBOM`` path — required, not
    defaulted, so a forgotten wiring fails loud instead of silently dropping
    every QBOM bound; *effective* feeds only the ``HV`` handler's per-stage
    floor and cadastro guard.

    Raises
    ------
    KeyError
        Propagated from ``id_map.hydro_id``/``id_map.thermal_id`` when an
        RE/HQ/HV term names a code the id map does not know (a real
        reader/id-map mismatch).
    ValueError
        When a ``to_bounds`` record carries a family other than ``"RE"``/
        ``"HQ"``/``"HV"``, or an ``"RE"`` record whose single term's variable
        is neither ``"generation"`` nor ``"thermal_generation"``.
    """
    calendar = case.calendar
    contributions: list[BoundContribution] = []
    for record in census.to_bounds:
        if _is_frequency_tagged_generation(record):
            continue
        if record.family == "RE":
            variable = record.terms[0].variable
            if variable == "generation":
                contributions.extend(
                    _re_generation_contributions(record, id_map, calendar)
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
                contributions.extend(_hq_flow_contributions(record, id_map, calendar))
        elif record.family == "HV":
            contributions.extend(
                _hv_storage_contributions(record, id_map, calendar, effective)
            )
        else:
            raise ValueError(
                f"single_term_bound_contributions: unhandled family {record.family!r}"
            )
    return contributions
