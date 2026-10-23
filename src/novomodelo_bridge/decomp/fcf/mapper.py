"""Manifest-to-manifest mapper for the source model's boundary cuts.

The bootstrap stage (``fcf/bootstrap.py``) reads back novomodelo's terminal
``entity_manifest`` — the target case's per-slot state-vector layout. This
module maps each of the source model's boundary cuts
(:class:`~novomodelo_bridge.decomp.fcf.cortes.BoundaryCuts`) onto that layout. It is
deck-free: the importer resolves which converted hydro takes which cut plant's
coefficients (:class:`HydroCoupling`), the constant inflow terms to fold, and
the GNL delivery calendar (:class:`GnlRingPlan`).

Units. The source writes every cut term per mean month
(:data:`~novomodelo_bridge.core.units.MEAN_MONTH_HOURS`): the source model evaluates
a cut as ``MEAN_MONTH_HOURS × (RHS + Σ PIVARM·(V − Vmin) + Σ PIAFL·Q_hm3)`` plus
the GNL terms, which are already ``$/MWh`` and take no month factor. novomodelo's
storage state is the total volume in hm³ and its inflow-lag state a flow in
m³/s, so:

- intercept ``× MEAN_MONTH_HOURS``;
- storage ``× MEAN_MONTH_HOURS``, with ``MEAN_MONTH_HOURS·PIVARM·Vmin`` taken
  off the intercept, and only on a plant the importer marks as a reservoir —
  the source model gives a run-of-river plant no storage term;
- inflow lag ``× MEAN_MONTH_HOURS × C_M3S2HM3`` (the hm³ a 1 m³/s flow carries
  over a mean month);
- GNL on a ring slot ``Σ_p pi_gnl[s, p, lag] · h_slot · f_p``: ``h_slot`` is the
  slot's own delivery hours in novomodelo's calendar and ``f_p`` the source model's
  patamar share of the delivery month, so one MW committed on the slot is
  priced as energy spread over the source's patamares.

A cut plant no converted hydro takes is dropped, never folded into a
neighbour, and recorded in :class:`MappingResult.dropped`; a GNL term with no
live ring slot is recorded in :class:`MappingResult.gnl_dropped`.
``HydroTransitBucket`` slots stay at an explicit ``0.0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

from novomodelo_bridge.core.units import C_M3S2HM3, MEAN_MONTH_HOURS

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from novomodelo_bridge.decomp.fcf.bootstrap import TerminalManifest
    from novomodelo_bridge.decomp.fcf.cortes import BoundaryCuts

#: novomodelo `policy.fbs` entity_type codes (confirmed against `policy_export.rs`).
_HYDRO_STORAGE = 0
_HYDRO_INFLOW_LAG = 1
_ANTICIPATED_THERMAL_STATE = 2
_HYDRO_TRANSIT_BUCKET = 3

#: `HydroStorage`'s `subindex` is always 0 (policy.fbs: one slot per plant).
_STORAGE_SUBINDEX = 0

#: novomodelo's `ENTITY_SLOT_DATE_SENTINEL` (`i32::MIN`), the "no date" value of every
#: per-slot date field; an undated `AnticipatedThermalState` slot is never a GNL
#: target.
_SLOT_DATE_SENTINEL = -2147483648

_INFLOW_LAG_FACTOR = MEAN_MONTH_HOURS * C_M3S2HM3


def month_key(year: int, month: int) -> int:
    """A calendar month as one ordered integer, ``year * 12 + month - 1``."""
    return year * 12 + month - 1


@dataclass(frozen=True)
class HydroCoupling:
    """One converted hydro taking one cut plant's coefficients.

    ``plant_index`` is the position in ``cuts.header.plant_codes``; several
    hydros may share one (the components of a source complex, or plants the
    source model couples through a downstream plant). ``storage_floor_hm3`` is
    set only when the hydro takes the storage term: the cut prices
    ``V - storage_floor_hm3`` while novomodelo's state is the total volume ``V``.
    """

    hydro_id: int
    plant_index: int
    storage_floor_hm3: float | None = None


@dataclass(frozen=True)
class MappedCut:
    """One source cut mapped onto the target's state-vector layout.

    ``coefficients`` is aligned to ``TerminalManifest.state_dimension``, every
    slot explicit. ``intercept`` is the scaled source ``rhs`` plus every
    constant the mapper folds in (storage floors, inflow-lag terms novomodelo cannot
    carry, already-commanded GNL deliveries). ``cut_id``, ``iteration``,
    ``forward_pass_index`` and ``is_active`` are the source record's, verbatim.
    """

    intercept: float
    coefficients: tuple[float, ...]
    cut_id: int
    iteration: int
    forward_pass_index: int
    is_active: bool
    #: Inflow-lag terms keyed by novomodelo hydro id (depth 1..N), used when the
    #: manifest carries no ``HydroInflowLag`` slots (``map_boundary_cuts``'s
    #: ``inflow_lag_depth``): novomodelo's ``write_policy_checkpoint`` reserves the
    #: canonical slots and places these. Empty when the slots are already in
    #: ``coefficients`` or the boundary prices no inflow-lag state.
    inflow_lag_coefficients: Mapping[int, tuple[float, ...]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class DroppedTerm:
    """A cut plant no converted hydro takes.

    ``beta`` is its ``pi_varm`` in the first cut record, a representative value
    for the diagnostics layer. ``inflow_folded`` is true when its inflow-lag
    terms were folded into the intercept from the deck's inflow history.
    """

    plant_code: int
    beta: float
    inflow_folded: bool = False


@dataclass(frozen=True)
class GnlThermalTarget:
    """One GNL thermal on the anticipated ring.

    ``thermal_id`` is the novomodelo ring ``entity_id``; ``submercado`` is the
    1-based source submercado index of the ``pi_gnl`` layout
    (``cortes.py::_gnl_columns``).
    """

    thermal_id: int
    submercado: int


@dataclass(frozen=True)
class GnlDeliveryMonth:
    """How the cut prices GNL delivered in one calendar month.

    ``lag`` is the 1-based ``pi_gnl`` lag axis (delivery month minus the cut's
    month); ``patamar_fractions`` the source model's duration share of each
    patamar in that month, in patamar order.
    """

    lag: int
    patamar_fractions: tuple[float, ...]


@dataclass(frozen=True)
class GnlCommandedDelivery:
    """An already-commanded post-horizon delivery, a constant to the cut.

    ``block_mw`` is the commanded MW per patamar; ``hours`` the delivery span.
    """

    submercado: int
    month: int
    block_mw: tuple[float, ...]
    hours: float


@dataclass(frozen=True)
class GnlRingPlan:
    """The deck's GNL delivery calendar, resolved by the importer.

    ``months`` is keyed by :func:`month_key` and holds only the months the
    cut's lag axis prices. ``week_months`` maps an operative-week start
    (``YYYYMMDD``) to the month the source model books that week to, which can
    differ from the calendar month of its start; a slot not listed falls back
    to its start's calendar month.
    """

    targets: tuple[GnlThermalTarget, ...]
    months: Mapping[int, GnlDeliveryMonth]
    week_months: Mapping[int, int] = field(default_factory=dict)
    commanded: tuple[GnlCommandedDelivery, ...] = ()


@dataclass(frozen=True)
class GnlDroppedTerm:
    """A GNL term that reached no ring slot, never folded onto a neighbour.

    ``thermal_id`` is ``None`` for a source submercado no target claims, whose
    ``coefficient`` is its summed ``pi_gnl`` at ``lag`` in the first active
    record that carries one. A target-side drop names the thermal and carries
    ``coefficient`` ``0.0``; ``lag`` is ``None`` when the slot's month has no
    lag on the cut's axis.
    """

    thermal_id: int | None
    submercado: int
    lag: int | None
    coefficient: float
    reason: str


@dataclass(frozen=True)
class MappingResult:
    """The mapped cuts plus every dropped source term."""

    cuts: tuple[MappedCut, ...]
    dropped: tuple[DroppedTerm, ...]
    gnl_dropped: tuple[GnlDroppedTerm, ...] = ()


def _default_lag_slot_of(depth: int) -> int:
    """Identity calendar-month lag mapping: source lag depth d -> subindex d-1."""
    return depth - 1


def _slot_int(slot: Mapping[str, object], field: str) -> int:
    """Extract `field` from a manifest slot dict, validating it is an `int`.

    novomodelo's `load_policy` hands each slot back as an untyped dict; a malformed
    slot fails here instead of corrupting a downstream index lookup.
    """
    value = slot[field]
    if not isinstance(value, int):
        raise TypeError(
            f"manifest slot field {field!r} is {type(value).__name__}, not "
            f"int: {value!r}"
        )
    return value


def _index_manifest(manifest: TerminalManifest) -> dict[tuple[int, int, int], int]:
    """Index the target manifest by (entity_type, entity_id, subindex) -> position."""
    return {
        (
            _slot_int(slot, "entity_type"),
            _slot_int(slot, "entity_id"),
            _slot_int(slot, "subindex"),
        ): position
        for position, slot in enumerate(manifest.entity_manifest)
    }


def _yyyymmdd(value: int) -> date:
    year, rest = divmod(value, 10_000)
    month, day = divmod(rest, 100)
    return date(year, month, day)


def _dated_ring_slots(
    manifest: TerminalManifest,
) -> dict[int, tuple[tuple[int, int, int], ...]]:
    """``{thermal id: ((interval_start, interval_end, position), ...)}``.

    Dated `AnticipatedThermalState` slots only; `_index_manifest` cannot serve
    here because it drops the dates.
    """
    by_thermal: dict[int, list[tuple[int, int, int]]] = {}
    for position, slot in enumerate(manifest.entity_manifest):
        if _slot_int(slot, "entity_type") != _ANTICIPATED_THERMAL_STATE:
            continue
        interval_start = _slot_int(slot, "interval_start")
        if interval_start == _SLOT_DATE_SENTINEL:
            continue
        by_thermal.setdefault(_slot_int(slot, "entity_id"), []).append(
            (interval_start, _slot_int(slot, "interval_end"), position)
        )
    return {thermal_id: tuple(slots) for thermal_id, slots in by_thermal.items()}


def _lag_subindex_bound(slot_positions: Mapping[tuple[int, int, int], int]) -> int:
    """One past the max `HydroInflowLag` subindex present, or 0 if absent."""
    lag_subindices = [
        subindex
        for entity_type, _entity_id, subindex in slot_positions
        if entity_type == _HYDRO_INFLOW_LAG
    ]
    return max(lag_subindices) + 1 if lag_subindices else 0


def _validated_lag_subindices(
    lag_slot_of: Callable[[int], int], lag_bound: int
) -> tuple[int, ...]:
    """Resolve `lag_slot_of(d)` for calendar-month depth `d` in 1..12.

    Bound-checked up front against `lag_bound` (skipped when 0, no
    `HydroInflowLag` family) so a mis-injected `lag_slot_of` fails with a
    `ValueError` naming the depth.
    """
    subindices = tuple(lag_slot_of(depth) for depth in range(1, 13))
    if lag_bound == 0:
        return subindices
    for depth, subindex in enumerate(subindices, start=1):
        if not 0 <= subindex < lag_bound:
            raise ValueError(
                f"lag_slot_of({depth}) = {subindex} is out of range for the "
                f"manifest's {lag_bound} HydroInflowLag slot(s)"
            )
    return subindices


class _GnlLayout:
    """The flat ``pi_gnl`` column layout, sbm-major × patamar × lag."""

    def __init__(self, cuts: BoundaryCuts) -> None:
        self.n_patamares = cuts.header.n_patamares
        self.lag_max = cuts.header.lag_maximo_gnl
        width = len(cuts.records[0].pi_gnl) if cuts.records else 0
        block = self.n_patamares * self.lag_max
        if block == 0 or width == 0:
            self.n_submercados = 0
            return
        if width % block != 0:
            raise ValueError(
                f"pi_gnl width {width} is not a multiple of n_patamares "
                f"({self.n_patamares}) * lag_maximo_gnl ({self.lag_max}) = {block}"
            )
        self.n_submercados = width // block

    def column(self, submercado: int, patamar: int, lag: int) -> int:
        """Flat column for 1-based (submercado, patamar, lag)."""
        return ((submercado - 1) * self.n_patamares + (patamar - 1)) * self.lag_max + (
            lag - 1
        )

    def weights(
        self, submercado: int, month: GnlDeliveryMonth, block_hours: Sequence[float]
    ) -> tuple[tuple[int, float], ...]:
        """``(column, weight)`` per patamar, ``weight = hours_p · f_p``."""
        return tuple(
            (self.column(submercado, patamar, month.lag), hours * fraction)
            for patamar, (hours, fraction) in enumerate(
                zip(block_hours, month.patamar_fractions, strict=True), start=1
            )
        )


def _resolve_gnl(
    cuts: BoundaryCuts,
    manifest: TerminalManifest,
    gnl_plan: GnlRingPlan | None,
) -> tuple[
    dict[int, tuple[tuple[int, float], ...]],
    tuple[tuple[int, float], ...],
    tuple[GnlDroppedTerm, ...],
]:
    """Resolve the GNL ring once: per-slot weights, commanded weights, drops.

    Returns ``({}, (), ())`` when there is no plan, no record, or no GNL axis.
    """
    if gnl_plan is None or not cuts.records:
        return {}, (), ()
    layout = _GnlLayout(cuts)
    if layout.n_submercados == 0:
        return {}, (), ()

    def in_range(submercado: int) -> bool:
        return 1 <= submercado <= layout.n_submercados

    ring = _dated_ring_slots(manifest)
    resolved: dict[int, tuple[tuple[int, float], ...]] = {}
    dropped: list[GnlDroppedTerm] = []
    for target in gnl_plan.targets:
        if not in_range(target.submercado):
            dropped.append(
                GnlDroppedTerm(
                    target.thermal_id,
                    target.submercado,
                    None,
                    0.0,
                    "submercado out of the pi_gnl range",
                )
            )
            continue
        slots = ring.get(target.thermal_id, ())
        if not slots:
            dropped.append(
                GnlDroppedTerm(
                    target.thermal_id,
                    target.submercado,
                    None,
                    0.0,
                    "no dated ring slot for thermal",
                )
            )
            continue
        unpriced = 0
        for interval_start, interval_end, position in slots:
            start = _yyyymmdd(interval_start)
            key = gnl_plan.week_months.get(
                interval_start, month_key(start.year, start.month)
            )
            month = gnl_plan.months.get(key)
            if month is None:
                unpriced += 1
                continue
            hours = (_yyyymmdd(interval_end) - start).days * 24.0
            # TRACKED NOVOMODELO-GAP (C24): the ring holds one MW per slot, so the
            # source's per-patamar prices collapse onto it, hours-weighted.
            resolved[position] = layout.weights(
                target.submercado, month, (hours,) * layout.n_patamares
            )
        if unpriced:
            dropped.append(
                GnlDroppedTerm(
                    target.thermal_id,
                    target.submercado,
                    None,
                    0.0,
                    f"{unpriced} ring slot(s) deliver in a month the cut's "
                    "GNL lags do not price",
                )
            )

    commanded: list[tuple[int, float]] = []
    for delivery in gnl_plan.commanded:
        month = gnl_plan.months.get(delivery.month)
        if month is None or not in_range(delivery.submercado):
            continue
        commanded.extend(
            (column, weight * mw)
            for (column, weight), mw in zip(
                layout.weights(
                    delivery.submercado,
                    month,
                    (delivery.hours,) * layout.n_patamares,
                ),
                delivery.block_mw,
                strict=True,
            )
        )

    targeted = {target.submercado for target in gnl_plan.targets}
    active = tuple(record for record in cuts.records if record.is_active)
    for submercado in range(1, layout.n_submercados + 1):
        if submercado in targeted:
            continue
        for lag in range(1, layout.lag_max + 1):
            columns = tuple(
                layout.column(submercado, patamar, lag)
                for patamar in range(1, layout.n_patamares + 1)
            )
            first = next(
                (r for r in active if any(r.pi_gnl[c] != 0.0 for c in columns)),
                None,
            )
            if first is not None:
                dropped.append(
                    GnlDroppedTerm(
                        None,
                        submercado,
                        lag,
                        math.fsum(first.pi_gnl[c] for c in columns),
                        "no GNL thermal in submercado",
                    )
                )
    return resolved, tuple(commanded), tuple(dropped)


def map_boundary_cuts(
    cuts: BoundaryCuts,
    manifest: TerminalManifest,
    couplings: Sequence[HydroCoupling],
    *,
    lag_slot_of: Callable[[int], int] = _default_lag_slot_of,
    inflow_lag_depth: int = 0,
    inflow_lag_offsets: Mapping[int, Sequence[float]] | None = None,
    source_only_inflows: Mapping[int, Sequence[float]] | None = None,
    gnl_plan: GnlRingPlan | None = None,
) -> MappingResult:
    """Map every cut in `cuts.records` onto `manifest`'s state-vector layout.

    Each coupling places its cut plant's storage term (when it carries a
    floor) on the hydro's ``HydroStorage`` slot and the inflow-lag terms on its
    ``HydroInflowLag`` slots via `lag_slot_of`, or into
    `MappedCut.inflow_lag_coefficients` (depths ``1..inflow_lag_depth``) when the
    manifest has none. Every record maps, active or not.

    `inflow_lag_offsets` (``{hydro_id: per-depth m³/s}``) is the part of the
    source's lag value novomodelo's state will not carry for that hydro; it folds as
    ``Σ_d coef_d · offset_d`` over the depths actually placed.
    `source_only_inflows` (``{plant_index: per-depth m³/s}``) folds the inflow
    terms of a cut plant no hydro takes the same way. Both carry all 12
    depths.

    Raises
    ------
    ValueError
        If the manifest has no ``HydroStorage`` slot at all, a coupling with a
        storage floor names a hydro without one, `lag_slot_of` is out of range,
        or the ``pi_gnl`` width does not fit the header's layout.
    """
    slot_positions = _index_manifest(manifest)
    if not any(entity_type == _HYDRO_STORAGE for entity_type, _, _ in slot_positions):
        raise ValueError(
            "target manifest has no HydroStorage slots at all; a real "
            "terminal-manifest read must carry at least the storage family"
        )
    lag_bound = _lag_subindex_bound(slot_positions)
    lag_subindices = _validated_lag_subindices(lag_slot_of, lag_bound)
    offsets = inflow_lag_offsets or {}
    source_only = source_only_inflows or {}

    storage_positions: dict[int, int] = {}
    for coupling in couplings:
        if coupling.storage_floor_hm3 is None:
            continue
        position = slot_positions.get(
            (_HYDRO_STORAGE, coupling.hydro_id, _STORAGE_SUBINDEX)
        )
        if position is None:
            raise ValueError(
                f"hydro {coupling.hydro_id} takes a storage term but the "
                "target manifest has no HydroStorage slot for it"
            )
        storage_positions[coupling.hydro_id] = position

    coupled = {coupling.plant_index for coupling in couplings}
    representative = cuts.records[0].pi_varm if cuts.records else ()
    dropped = tuple(
        DroppedTerm(
            plant_code=code,
            beta=representative[index] if representative else 0.0,
            inflow_folded=index in source_only,
        )
        for index, code in enumerate(cuts.header.plant_codes)
        if index not in coupled
    )

    gnl_slots, gnl_commanded, gnl_dropped = _resolve_gnl(cuts, manifest, gnl_plan)

    mapped_cuts: list[MappedCut] = []
    for record in cuts.records:
        coefficients = [0.0] * manifest.state_dimension
        inflow_lag_coefficients: dict[int, tuple[float, ...]] = {}
        folds: list[float] = []
        for coupling in couplings:
            if coupling.storage_floor_hm3 is not None:
                storage = record.pi_varm[coupling.plant_index] * MEAN_MONTH_HOURS
                coefficients[storage_positions[coupling.hydro_id]] = storage
                folds.append(-storage * coupling.storage_floor_hm3)
            lags = tuple(
                value * _INFLOW_LAG_FACTOR
                for value in record.pi_qafl[coupling.plant_index]
            )
            offset = offsets.get(coupling.hydro_id)
            if lag_bound > 0:
                for depth_index, subindex in enumerate(lag_subindices):
                    position = slot_positions.get(
                        (_HYDRO_INFLOW_LAG, coupling.hydro_id, subindex)
                    )
                    if position is None:
                        continue
                    coefficients[position] = lags[depth_index]
                    if offset is not None:
                        folds.append(lags[depth_index] * offset[depth_index])
            elif inflow_lag_depth > 0:
                inflow_lag_coefficients[coupling.hydro_id] = lags[:inflow_lag_depth]
                if offset is not None:
                    folds.extend(lags[d] * offset[d] for d in range(inflow_lag_depth))
        for plant_index, values in source_only.items():
            folds.extend(
                coefficient * _INFLOW_LAG_FACTOR * value
                for coefficient, value in zip(
                    record.pi_qafl[plant_index], values, strict=True
                )
            )
        for position, weights in gnl_slots.items():
            coefficients[position] = math.fsum(
                record.pi_gnl[column] * weight for column, weight in weights
            )
        folds.extend(record.pi_gnl[column] * weight for column, weight in gnl_commanded)

        mapped_cuts.append(
            MappedCut(
                intercept=record.rhs * MEAN_MONTH_HOURS + math.fsum(folds),
                coefficients=tuple(coefficients),
                cut_id=record.cut_id,
                iteration=record.iteration,
                forward_pass_index=record.forward_pass_index,
                is_active=record.is_active,
                inflow_lag_coefficients=inflow_lag_coefficients,
            )
        )

    return MappingResult(
        cuts=tuple(mapped_cuts), dropped=dropped, gnl_dropped=gnl_dropped
    )
