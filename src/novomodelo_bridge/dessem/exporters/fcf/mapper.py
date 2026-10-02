"""Novomodelo-to-DECOMP mapping rules, one named function per premise.

Each premise lives in exactly one function so it stays individually testable,
documented and revertible. Nothing here performs I/O except `load_hydro_codes`.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from novomodelo_bridge.dessem.exporters.fcf.inputs_reader import (
        BlockInfo,
        BusInfo,
        StageInfo,
        ThermalInfo,
    )
    from novomodelo_bridge.dessem.exporters.fcf.reader import (
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

# Storage slots carry this sentinel in place of a delivery date
DELIVERY_DATE_SENTINEL = -2147483648

# Premise P15: Evidenced GNL lead time mapping
EVIDENCED_GNL_LEAD_TIME_HOURS = 1608.0
EVIDENCED_GNL_LAG_MESES = 2


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
class DeliveryGroup:
    """A GNL delivery month and its rank within one pool's own set of months."""

    delivery_date: int
    pool_rank: int


@dataclass(frozen=True)
class InflowLagAudit:
    """Counted record of inflow-lag coefficients premise P8 drops."""

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


def has_delivery_date(slot: EntitySlotRecord) -> bool:
    """Whether the slot carries a real delivery date."""
    return slot.delivery_date > 0


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


def negate_gnl(value: float) -> float:
    """Flip a GNL coefficient's sign (premise P4)."""
    return -value


def negate_gnl_array(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flip a GNL coefficient vector's sign (premise P4)."""
    return -np.asarray(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Premise P5: GNL block weights
# ---------------------------------------------------------------------------


def gnl_block_weights(blocks: Sequence[BlockInfo]) -> tuple[float, ...]:
    """Hours-weighted partition of a stage across its load blocks (premise P5)."""
    if not blocks:
        raise MappingError("cannot weight an empty block list")
    for block in blocks:
        if not math.isfinite(block.hours) or block.hours < 0.0:
            raise MappingError(
                f"block {block.id} ({block.name}) has hours {block.hours!r}; "
                f"block hours must be finite and non-negative"
            )
    total = sum(block.hours for block in blocks)
    if total <= 0.0:
        raise MappingError(f"block hours must sum to a positive value, got {total}")
    return tuple(block.hours / total for block in blocks)


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
# Premise P15: GNL lead time validation
# ---------------------------------------------------------------------------


def assert_gnl_lead_time_is_evidenced(gnl_thermals: Sequence[ThermalInfo]) -> None:
    """Refuse a GNL lead time the lag-month mapping has no evidence for."""
    for thermal in gnl_thermals:
        if thermal.lead_time_hours != EVIDENCED_GNL_LEAD_TIME_HOURS:
            raise MappingError(
                f"thermal {thermal.id} ({thermal.name}) declares lead_time_hours="
                f"{thermal.lead_time_hours!r}, but the lead-time-to-lag-month "
                f"mapping is evidenced only at {EVIDENCED_GNL_LEAD_TIME_HOURS} h"
            )
    _LOG.info(
        "premise P15 holds: every GNL plant declares lead_time_hours=%s",
        EVIDENCED_GNL_LEAD_TIME_HOURS,
    )


# ---------------------------------------------------------------------------
# Delivery date grouping
# ---------------------------------------------------------------------------


def delivery_slot_map(
    slots: Sequence[EntitySlotRecord],
) -> dict[int, DeliveryGroup]:
    """Group anticipated-thermal positions by delivery month, per pool."""
    positions = index_slots(slots).anticipated_thermal
    if not positions:
        return {}

    for position in positions:
        if not has_delivery_date(slots[position]):
            raise MappingError(
                f"anticipated-thermal slot at position {position} "
                f"carries no delivery date"
            )

    dates = sorted({slots[position].delivery_date for position in positions})
    ranked = {delivery_date: rank for rank, delivery_date in enumerate(dates, start=1)}
    return {
        position: DeliveryGroup(
            delivery_date=slots[position].delivery_date,
            pool_rank=ranked[slots[position].delivery_date],
        )
        for position in positions
    }


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


def _reject_repeated_keys(
    path: Path,
) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]:
    """A json object hook that refuses a repeated key."""

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise MappingError(f"{path} repeats key {key!r}")
            seen.add(key)
        return dict(pairs)

    return hook


def load_hydro_codes(path: Path) -> tuple[int, ...]:
    """Read the Novomodelo-position-to-DECOMP-plant-code map in positional order."""
    if not path.is_file():
        raise MappingError(f"hydro code map not found: {path}")
    try:
        raw: Any = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_repeated_keys(path),
        )
    except json.JSONDecodeError as exc:
        raise MappingError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise MappingError(f"{path} must hold a JSON object, got {type(raw).__name__}")
    if not raw:
        raise MappingError(f"{path} holds no entries")

    positions: dict[int, int] = {}
    for key, value in raw.items():
        try:
            position = int(key)
        except ValueError as exc:
            raise MappingError(
                f"{path} key {key!r} is not an integer position"
            ) from exc
        if key != str(position):
            raise MappingError(
                f"{path} key {key!r} is not the canonical form of position "
                f"{position}; keys must be exactly '0'..'N-1'"
            )
        if type(value) is not int:
            raise MappingError(
                f"{path} entry {key!r} must map to an int, got {type(value).__name__}"
            )
        positions[position] = value

    expected = set(range(len(positions)))
    missing = sorted(expected - positions.keys())
    if missing:
        raise MappingError(
            f"{path} is not contiguous from 0: missing position(s) {missing[:10]}"
        )
    codes = tuple(positions[position] for position in sorted(positions))
    repeated = sorted(code for code, count in Counter(codes).items() if count > 1)
    if repeated:
        raise MappingError(
            f"{path} maps more than one position to DECOMP plant code(s) "
            f"{repeated[:10]}"
        )
    return codes


def hydro_code_for_position(codes: Sequence[int], position: int) -> int:
    """Return the DECOMP plant code for a Novomodelo entity-manifest position."""
    if not 0 <= position < len(codes):
        raise MappingError(
            f"hydro position {position} is outside the {len(codes)}-entry code map"
        )
    return codes[position]
