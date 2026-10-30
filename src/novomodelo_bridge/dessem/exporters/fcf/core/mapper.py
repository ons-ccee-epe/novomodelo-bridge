"""Novomodelo-to-DECOMP mapping rules, one named function per premise.

Each premise lives in exactly one function so it stays individually testable,
documented and revertible. Nothing here performs I/O.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from novomodelo_bridge.core.policy_dates import (
    UNDATED,
    date_from_yyyymmdd,
    hours_between,
)
from novomodelo_bridge.core.tolerances import floats_differ
from novomodelo_bridge.dessem.exporters.fcf.core.layout import (
    GNL_EXPORT_MIDDLE_OFFSET,
    cortdeco_gnl_offset,
)

if TYPE_CHECKING:
    from novomodelo_bridge.dessem.exporters.fcf.core.inputs_reader import (
        BusInfo,
        HydroInfo,
        StageInfo,
        ThermalInfo,
    )
    from novomodelo_bridge.dessem.exporters.fcf.core.layout import CutBlockOffsets
    from novomodelo_bridge.dessem.exporters.fcf.core.reader import (
        EntitySlotRecord,
        PolicyManifest,
        StageCutPool,
    )

_LOG = logging.getLogger(__name__)

# Premise P2: DECOMP works in 10^3 R$ while Novomodelo works in R$.
DECOMP_COST_DIVISOR = 1000.0

# Premise P9: Day-count basis 365.25.
DAYS_PER_YEAR = 365.25

# Premise P6: Bus 5 is the interconnection node, not a submarket.
EXCLUDED_BUS_ID = 5
EXCLUDED_BUS_NAME = "IV"
SUBMARKET_COUNT = 5

# Entity type codes
ENTITY_TYPE_HYDRO_STORAGE = 0
ENTITY_TYPE_HYDRO_INFLOW_LAG = 1
ENTITY_TYPE_ANTICIPATED_THERMAL = 2
ENTITY_TYPE_HYDRO_TRANSIT_BUCKET = 3


class MappingError(Exception):
    """Raised when an input violates a mapping rule."""


@dataclass(frozen=True)
class SlotIndex:
    """Entity-manifest positions grouped by axis, ascending within each group."""

    storage: tuple[int, ...]
    inflow_lag: tuple[int, ...]
    anticipated_thermal: tuple[int, ...]
    transit_bucket: tuple[int, ...]


@dataclass(frozen=True)
class InflowLagAudit:
    """Counted record of inflow-lag coefficients premise P8 drops."""

    per_pool: tuple[tuple[int, int], ...]
    total: int


@dataclass(frozen=True)
class GnlDropAudit:
    """Counted record of anticipated-thermal coefficients premise P14 drops."""

    per_pool: tuple[tuple[int, int], ...]
    total: int


def index_slots(slots: Sequence[EntitySlotRecord]) -> SlotIndex:
    """Group slot positions by axis."""
    storage: list[int] = []
    inflow_lag: list[int] = []
    anticipated_thermal: list[int] = []
    transit_bucket: list[int] = []

    for position, slot in enumerate(slots):
        if slot.entity_type == ENTITY_TYPE_HYDRO_STORAGE:
            storage.append(position)
        elif slot.entity_type == ENTITY_TYPE_HYDRO_INFLOW_LAG:
            inflow_lag.append(position)
        elif slot.entity_type == ENTITY_TYPE_ANTICIPATED_THERMAL:
            anticipated_thermal.append(position)
        elif slot.entity_type == ENTITY_TYPE_HYDRO_TRANSIT_BUCKET:
            transit_bucket.append(position)

    return SlotIndex(
        storage=tuple(storage),
        inflow_lag=tuple(inflow_lag),
        anticipated_thermal=tuple(anticipated_thermal),
        transit_bucket=tuple(transit_bucket),
    )


# ---------------------------------------------------------------------------
# Premise P2: Cost unit conversion
# ---------------------------------------------------------------------------


def to_decomp_cost(value: float) -> float:
    """Convert one Novomodelo currency value to DECOMP's 10^3 R$ (premise P2)."""
    return value / DECOMP_COST_DIVISOR


def to_decomp_costs(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a coefficient vector to DECOMP's 10^3 R$ (premise P2)."""
    return np.asarray(values, dtype=np.float64) / DECOMP_COST_DIVISOR


# ---------------------------------------------------------------------------
# Premise P4: GNL sign convention
# ---------------------------------------------------------------------------


def negate_gnl_array(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flip a GNL coefficient vector's sign (premise P4)."""
    return -np.asarray(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Premise P6: Submarket mapping
# ---------------------------------------------------------------------------


def submarket_for_bus(bus_id: int) -> int:
    """Map a Novomodelo bus id to a DECOMP submarket number (premise P6)."""
    if bus_id == EXCLUDED_BUS_ID:
        raise MappingError(
            f"bus {bus_id} is the interconnection node and has no submarket"
        )
    if not 0 <= bus_id < EXCLUDED_BUS_ID:
        raise MappingError(
            f"bus {bus_id} is outside the {SUBMARKET_COUNT} submarket buses"
        )
    return bus_id + 1


def select_submarket_buses(buses: Sequence[BusInfo]) -> tuple[BusInfo, ...]:
    """Return buses that become DECOMP submarkets, excluding interconnection."""
    excluded = [bus for bus in buses if bus.id == EXCLUDED_BUS_ID]
    if len(excluded) != 1:
        raise MappingError(
            f"expected exactly one bus with id {EXCLUDED_BUS_ID}, found {len(excluded)}"
        )
    if excluded[0].name != EXCLUDED_BUS_NAME:
        raise MappingError(
            f"bus {EXCLUDED_BUS_ID} is named {excluded[0].name!r}, "
            f"expected {EXCLUDED_BUS_NAME!r}"
        )
    selected = tuple(bus for bus in buses if bus.id != EXCLUDED_BUS_ID)
    if len(selected) != SUBMARKET_COUNT:
        raise MappingError(
            f"expected {SUBMARKET_COUNT} submarket buses, found {len(selected)}"
        )
    return selected


# ---------------------------------------------------------------------------
# Premise P7: Cut-building pools
# ---------------------------------------------------------------------------


def cut_building_pools(
    pools: Mapping[int, StageCutPool], expected_count: int | None = None
) -> tuple[int, ...]:
    """Return pools that actually built cuts, ascending (finding F8)."""
    if not pools:
        return ()
    selected = tuple(
        sorted(
            pool_id
            for pool_id, pool in pools.items()
            if pool.populated_count > pool.warm_start_count
        )
    )
    expected = len(pools) - 1 if expected_count is None else expected_count
    if len(selected) != expected:
        _LOG.warning(
            "cut-building pools by populated_count > warm_start_count is %s "
            "(%d pool(s)), but n_pools - 1 is %d",
            list(selected),
            len(selected),
            expected,
        )
    return selected


# ---------------------------------------------------------------------------
# Premise P8: Inflow lag audit
# ---------------------------------------------------------------------------


def inflow_lag_drop_audit(pools: Mapping[int, StageCutPool]) -> InflowLagAudit:
    """Count the inflow-lag coefficients premise P8 drops, per pool."""
    per_pool = []
    for pool_id in sorted(pools):
        pool = pools[pool_id]
        lag_slots = len(index_slots(pool.slots).inflow_lag)
        per_pool.append((pool_id, lag_slots * len(pool.pieces)))

    total = sum(count for _, count in per_pool)
    if total:
        _LOG.warning(
            "premise P8 drops %d inflow-lag coefficients: %s",
            total,
            ", ".join(
                f"pool {pool_id}={count}" for pool_id, count in per_pool if count
            ),
        )
    else:
        _LOG.info("premise P8 drops no inflow-lag coefficients on this case")
    return InflowLagAudit(per_pool=tuple(per_pool), total=total)


# ---------------------------------------------------------------------------
# Premise P14: GNL drop audit
# ---------------------------------------------------------------------------


def gnl_drop_audit(
    pools: Mapping[int, StageCutPool],
    trunk_pool_ids: Sequence[int],
    deliveries: GnlDeliveries,
) -> GnlDropAudit:
    """Count the GNL coefficients the written cuts leave out, per cut-building pool."""
    delivered = sum(position is not None for position in deliveries.positions)
    per_pool = []
    for node, pool_id in enumerate(trunk_pool_ids):
        pool = pools[pool_id]
        written_cuts = sum(
            piece.slot_index >= pool.warm_start_count for piece in pool.pieces
        )
        gnl_slots = len(index_slots(pool.slots).anticipated_thermal)
        written_slots = delivered if node == 0 else 0
        per_pool.append((pool_id, (gnl_slots - written_slots) * written_cuts))

    total = sum(count for _, count in per_pool)
    if total:
        _LOG.warning(
            "premise P14 drops %d anticipated-thermal coefficients: %s",
            total,
            ", ".join(
                f"pool {pool_id}={count}" for pool_id, count in per_pool if count
            ),
        )
    else:
        _LOG.info("premise P14 drops no anticipated-thermal coefficients on this case")
    return GnlDropAudit(per_pool=tuple(per_pool), total=total)


# ---------------------------------------------------------------------------
# Premise P9: Discount factors
# ---------------------------------------------------------------------------


def parse_start_date(stage: StageInfo) -> date:
    """Parse a stage's start_date as ISO-8601."""
    try:
        return date.fromisoformat(stage.start_date)
    except ValueError as exc:
        raise MappingError(
            f"stage {stage.id} start_date {stage.start_date!r} is not ISO-8601"
        ) from exc


def discount_factors(
    stages: Sequence[StageInfo], annual_rate: float
) -> tuple[float, ...]:
    """Per-stage discount factors, duration-proportional on 365.25-day year."""
    if not stages:
        return ()
    if annual_rate <= -1.0:
        raise MappingError(f"annual discount rate must exceed -1, got {annual_rate}")

    origin = parse_start_date(stages[0])
    factors = []
    for stage in stages:
        elapsed_days = (parse_start_date(stage) - origin).days
        if elapsed_days < 0:
            raise MappingError(
                f"stage {stage.id} starts {abs(elapsed_days)} days before stage 0"
            )
        factors.append((1.0 + annual_rate) ** (-elapsed_days / DAYS_PER_YEAR))
    return tuple(factors)


# ---------------------------------------------------------------------------
# Premise P15: GNL lag months
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceGnlThermal:
    """A GNL plant as the source deck declares it: `TG` code and name, `NL` lag."""

    code: int
    name: str
    lag_months: int | None


def gnl_lag_months(
    gnl_thermals: Sequence[ThermalInfo], source: Sequence[SourceGnlThermal]
) -> tuple[int, ...]:
    """Each anticipated thermal's lag in whole months, from the deck's `NL` register.

    *gnl_thermals* ascend by novomodelo id and *source* by deck code; they pair by
    position because `convert decomp` numbers the GNL plants in ascending code
    order.

    Raises:
        MappingError: the plant counts or names differ, or a plant has no `NL` lag.
    """
    if len(gnl_thermals) != len(source):
        raise MappingError(
            f"the deck declares {len(source)} GNL plants but the case holds "
            f"{len(gnl_thermals)} anticipated thermals; export from the deck the "
            "case was converted from"
        )
    lags: list[int] = []
    for thermal, plant in zip(gnl_thermals, source, strict=True):
        if thermal.name != plant.name:
            raise MappingError(
                f"anticipated thermal {thermal.id} is {thermal.name!r} in the case "
                f"but GNL plant {plant.code} in the deck is {plant.name!r}; export "
                "from the deck the case was converted from"
            )
        if plant.lag_months is None:
            raise MappingError(
                f"GNL plant {plant.code} ({plant.name}) has no NL record in the "
                "deck's dadgnl file, so its dispatch-anticipation lag in months "
                "is unknown"
            )
        lags.append(plant.lag_months)
    _LOG.info("premise P15: GNL lag months %s from the deck's NL records", lags)
    return tuple(lags)


# ---------------------------------------------------------------------------
# Premise P14: GNL delivery selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GnlDeliveries:
    """The stage whose delivery is priced and each GNL thermal's slot for it.

    `positions` holds one slot position per GNL thermal, in the thermals' order,
    `None` where the pool has no slot for that delivery. `stage` is `None` only
    when the case has no GNL thermal.
    """

    stage: StageInfo | None
    positions: tuple[int | None, ...]

    @property
    def block_hours(self) -> tuple[float, ...]:
        """The stage's block hours, `()` when there is no stage."""
        if self.stage is None:
            return ()
        return tuple(block.hours for block in self.stage.blocks)

    @property
    def hours(self) -> float:
        """The stage's total hours, the divisor of a GNL coefficient."""
        return math.fsum(self.block_hours)


def select_gnl_deliveries(
    pool: StageCutPool,
    gnl_thermals: Sequence[ThermalInfo],
    stages: Sequence[StageInfo],
) -> GnlDeliveries:
    """Each GNL thermal's slot for the delivery of the stage after the first.

    *pool* is the first cut-building pool. A slot is found by its
    `interval_start`, never by its rotating ring index, which names a different
    delivery in every pool.

    Raises:
        MappingError: the pool is not at stage 0, the case has no stage after the
            first, a thermal has two slots for the delivery, a slot's interval
            hours differ from the stage's block hours, or a candidate's dates are
            not valid.
    """
    if not gnl_thermals:
        return GnlDeliveries(stage=None, positions=())
    if pool.graph_stage_id != 0:
        raise MappingError(
            f"the first cut-building pool is at stage {pool.graph_stage_id}, but "
            "GNL coefficients are exported only from a pool at stage 0, whose "
            "next stage holds the delivery they price"
        )
    target = pool.graph_stage_id + 1
    if target >= len(stages):
        raise MappingError(
            f"the case has {len(stages)} stage(s) but holds {len(gnl_thermals)} GNL "
            "thermals, whose coefficients price the delivery of the stage after "
            "the first"
        )

    stage = stages[target]
    start = parse_start_date(stage)
    stage_hours = math.fsum(block.hours for block in stage.blocks)
    positions: list[int | None] = []
    for thermal in gnl_thermals:
        matches: list[int] = []
        for position, slot in enumerate(pool.slots):
            if (
                slot.entity_type != ENTITY_TYPE_ANTICIPATED_THERMAL
                or slot.entity_id != thermal.id
                or slot.interval_start == UNDATED
            ):
                continue
            try:
                slot_start = date_from_yyyymmdd(slot.interval_start)
            except ValueError as exc:
                raise MappingError(
                    f"GNL thermal {thermal.id} ({thermal.name}) has a delivery slot "
                    f"starting {slot.interval_start}, which is not a YYYYMMDD date"
                ) from exc
            if slot_start == start:
                matches.append(position)

        if len(matches) > 1:
            raise MappingError(
                f"GNL thermal {thermal.id} ({thermal.name}) has {len(matches)} "
                f"slots delivering from {start}; exactly one is expected, so the "
                "coefficient to export is ambiguous"
            )
        if not matches:
            positions.append(None)
            continue

        slot = pool.slots[matches[0]]
        try:
            end = date_from_yyyymmdd(slot.interval_end)
        except ValueError as exc:
            raise MappingError(
                f"GNL thermal {thermal.id} ({thermal.name}) has a slot delivering "
                f"from {start} whose end {slot.interval_end} is not a YYYYMMDD date"
            ) from exc
        slot_hours = hours_between(start, end)
        if floats_differ(slot_hours, stage_hours):
            raise MappingError(
                f"GNL thermal {thermal.id} ({thermal.name}) delivers from {start} "
                f"to {end} ({slot_hours:g} h) but the blocks of stage {stage.id} "
                f"sum to {stage_hours:g} h; the coefficient is divided by the "
                "stage's hours, so both must agree"
            )
        positions.append(matches[0])

    return GnlDeliveries(stage=stage, positions=tuple(positions))


def gnl_block(
    deliveries: GnlDeliveries,
    coefficients: NDArray[np.float64],
    offsets: CutBlockOffsets,
) -> NDArray[np.float64]:
    """One cut's `pi_gnl` block, each GNL coefficient in R$/MWh on every load block.

    Divides the R$/MW coefficient by the delivery stage's hours; a multiply by the
    reciprocal is 1 ulp off. The sign is kept: the writer negates it (premise P4).

    Raises:
        MappingError: the positions do not match `offsets.n_sbm_gnl`, a position is
            outside *coefficients*, or the delivery stage has no hours.
    """
    if len(deliveries.positions) != offsets.n_sbm_gnl:
        raise MappingError(
            f"{len(deliveries.positions)} GNL delivery positions but the cut layout "
            f"holds {offsets.n_sbm_gnl} GNL entries"
        )
    block = np.zeros(offsets.ncoef - offsets.pi_gnl)
    hours = deliveries.hours
    for entry, position in enumerate(deliveries.positions):
        if position is None:
            continue
        if not 0 <= position < len(coefficients):
            raise MappingError(
                f"GNL entry {entry} points at coefficient {position} but the "
                f"coefficient vector has {len(coefficients)} elements"
            )
        if hours <= 0.0:
            raise MappingError(
                f"GNL entry {entry} is priced on a delivery stage with {hours:g} "
                "hours, which cannot convert R$/MW to R$/MWh"
            )
        value = coefficients[position] / hours
        # TRACKED NOVOMODELO-GAP (C24): novomodelo prices one value per delivery, not per load block.
        for patamar in range(offsets.n_patamares):
            index = cortdeco_gnl_offset(
                offsets, entry, GNL_EXPORT_MIDDLE_OFFSET, patamar
            )
            block[index - offsets.pi_gnl] = value
    return block


# ---------------------------------------------------------------------------
# Tree traversal
# ---------------------------------------------------------------------------


def tree_indices(manifest: PolicyManifest) -> tuple[int, ...]:
    """DECOMP's 1-based parent index per node, from the policy graph edges."""
    node_ids = [node.id for node in manifest.nodes]
    if node_ids != list(range(len(node_ids))):
        raise MappingError(
            f"node ids must be contiguous 0..{len(node_ids) - 1}; got {node_ids[:8]}..."
        )
    known = set(node_ids)

    parents: dict[int, int] = {}
    for edge in manifest.edges:
        if edge.source_id not in known:
            raise MappingError(
                f"edge {edge.source_id} -> {edge.target_id} names unknown source"
            )
        if edge.target_id not in known:
            raise MappingError(
                f"edge {edge.source_id} -> {edge.target_id} names unknown target"
            )
        if edge.target_id in parents:
            raise MappingError(
                f"node {edge.target_id} has more than one parent "
                f"({parents[edge.target_id]} and {edge.source_id})"
            )
        parents[edge.target_id] = edge.source_id

    roots = [node_id for node_id in node_ids if node_id not in parents]
    if len(roots) != 1:
        raise MappingError(f"expected exactly one root node, found {len(roots)}")
    return tuple(parents.get(node_id, node_id) + 1 for node_id in node_ids)


def first_node_per_stage(manifest: PolicyManifest) -> tuple[int, ...]:
    """DECOMP's 1-based first-node id per stage, ordered by stage_id ascending."""
    node_ids = [node.id for node in manifest.nodes]
    if node_ids != list(range(len(node_ids))):
        raise MappingError(
            f"node ids must be contiguous 0..{len(node_ids) - 1}; got {node_ids[:8]}..."
        )

    import itertools

    stage_sequence = [node.stage_id for node in manifest.nodes]
    for prev_stage, next_stage in itertools.pairwise(stage_sequence):
        if next_stage < prev_stage:
            raise MappingError(
                f"stage ids must be non-decreasing in node-id order; "
                f"got {stage_sequence[:8]}..."
            )

    first_by_stage: dict[int, int] = {}
    for node in manifest.nodes:
        current = first_by_stage.get(node.stage_id)
        if current is None or node.id < current:
            first_by_stage[node.stage_id] = node.id

    observed = sorted(first_by_stage)
    expected = list(range(manifest.num_stages))
    if observed != expected:
        raise MappingError(
            f"stage ids must be contiguous 0..{manifest.num_stages - 1}; got {observed}"
        )
    return tuple(first_by_stage[stage_id] + 1 for stage_id in observed)


# ---------------------------------------------------------------------------
# Hydro codes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceHydro:
    """A hydro plant as the source deck names it: `hidr.dat` code and name."""

    code: int
    name: str


def match_source_hydros(
    hydros: Sequence[HydroInfo], source: Mapping[int, SourceHydro]
) -> None:
    """Refuse a deck other than the one the case was converted from.

    *source* is keyed by novomodelo hydro id. The deck must operate exactly the
    case's hydros, each under the name the case gives it.

    Raises:
        MappingError: the plant sets or any plant name differ.
    """
    if {hydro.id for hydro in hydros} != source.keys():
        raise MappingError(
            f"the deck operates {len(source)} hydro plants but the case holds "
            f"{len(hydros)}; export from the deck the case was converted from"
        )
    renamed = [
        (hydro, source[hydro.id])
        for hydro in sorted(hydros, key=lambda hydro: hydro.id)
        if hydro.name != source[hydro.id].name
    ]
    if renamed:
        hydro, plant = renamed[0]
        raise MappingError(
            f"{len(renamed)} hydro plant names differ between the case and the "
            f"deck, first hydro {hydro.id}: {hydro.name!r} in the case, "
            f"{plant.name!r} (plant {plant.code}) in the deck; export from the "
            "deck the case was converted from"
        )


def storage_plant_ids(slots: Sequence[EntitySlotRecord]) -> tuple[int, ...]:
    """The hydro ids of a pool's storage axes, in slot order.

    Slot order is novomodelo's plant order, not ascending id order, and it orders
    every per-plant `mapcut` record and each cut's `pi_varm` block.

    Raises:
        MappingError: storage is not one block at the head of the slots.
    """
    positions = index_slots(slots).storage
    if positions != tuple(range(len(positions))):
        raise MappingError(
            f"storage occupies slot positions {positions[:4]}... "
            f"({len(positions)} slots) instead of leading the slots as one block"
        )
    return tuple(slots[position].entity_id for position in positions)


def plant_codes(
    plant_ids: Sequence[int], source: Mapping[int, SourceHydro]
) -> tuple[int, ...]:
    """The `mapcut` plant-code record: each storage axis's deck plant code.

    Raises:
        MappingError: the storage axes and the deck's plants differ.
    """
    if sorted(plant_ids) != sorted(source):
        missing = sorted(set(source) - set(plant_ids))
        unknown = sorted(set(plant_ids) - set(source))
        raise MappingError(
            f"the cuts carry storage for {len(plant_ids)} hydros but the deck "
            f"operates {len(source)} (deck hydro ids without storage: "
            f"{missing[:10]}; storage hydro ids not in the deck: {unknown[:10]})"
        )
    return tuple(source[hydro_id].code for hydro_id in plant_ids)


def downstream_positions(
    hydros: Sequence[HydroInfo], plant_ids: Sequence[int]
) -> tuple[int, ...]:
    """The `mapcut` downstream-plant record, one entry per plant of *plant_ids*.

    Each entry is the downstream plant's 1-based position in *plant_ids*, the
    order of the plant-code record (0 = no downstream plant), never its plant
    code: DECOMP writes positions into this record, and a code reads as a
    different plant wherever code and position differ.

    Raises:
        MappingError: a plant names a downstream id outside *plant_ids*.
    """
    by_id = {hydro.id: hydro for hydro in hydros}
    position_of = {hydro_id: position for position, hydro_id in enumerate(plant_ids, 1)}
    positions = []
    for hydro_id in plant_ids:
        hydro = by_id[hydro_id]
        if hydro.downstream_id is None:
            positions.append(0)
        elif hydro.downstream_id in position_of:
            positions.append(position_of[hydro.downstream_id])
        else:
            raise MappingError(
                f"hydro {hydro.id} ({hydro.name}) names downstream hydro "
                f"{hydro.downstream_id}, which the case does not hold"
            )
    return tuple(positions)
