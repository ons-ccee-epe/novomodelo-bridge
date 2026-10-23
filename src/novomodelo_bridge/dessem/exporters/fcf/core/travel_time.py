"""Premise P3: water travel time, from Novomodelo's transit buckets to DECOMP's release lags.

DECOMP states water in transit by the release it came from: plant `u`'s
defluence over stage `t - lag`, in hm³, for `lag` in `0..max_lag-1` at a node
of stage `t`. Novomodelo states it by when it arrives: one HydroTransitBucket per
downstream plant and dated arrival window, also in hm³. A release spread
evenly over its stage arrives `hours` later, evenly over an equally long span,
so its coefficient is the bucket coefficients weighted by the share of that
span each bucket's window holds.

Times are hours from the study start. Before it, release periods are as long
as stage 0, the span `convert decomp` tiles `past_defluences` with.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from novomodelo_bridge.dessem.exporters.fcf.core.mapper import MappingError, index_slots

if TYPE_CHECKING:
    from novomodelo_bridge.dessem.exporters.fcf.core.inputs_reader import (
        HydroInfo,
        StageInfo,
    )
    from novomodelo_bridge.dessem.exporters.fcf.core.mapper import SourceHydro
    from novomodelo_bridge.dessem.exporters.fcf.core.reader import EntitySlotRecord

_LOG = logging.getLogger(__name__)

_HOURS_PER_DAY = 24
_COVERAGE_TOLERANCE_HOURS = 1e-9

Window = tuple[float, float]


@dataclass(frozen=True)
class TravelTimePlant:
    """An upstream plant whose defluence reaches `downstream_id` `hours` later.

    `arrival_shares[stage][lag]` is the share of the plant's defluence over
    stage `stage - lag` that arrives within `stage`, for `lag` up to the last
    one with a non-zero share: the values of `mapcut` regs 7/8.
    """

    hydro_id: int
    code: int
    downstream_id: int
    hours: int
    arrival_shares: tuple[tuple[float, ...], ...]

    @property
    def stage_lags(self) -> tuple[int, ...]:
        """The plant's reg 6 entries: its last lag per stage."""
        return tuple(len(shares) - 1 for shares in self.arrival_shares)


def max_lag(plants: Sequence[TravelTimePlant]) -> int:
    """`mapcut`'s `max_lag`, also the release lags each cut carries per plant."""
    return max((lag for plant in plants for lag in plant.stage_lags), default=0)


def _hours_since(origin: date, day: date) -> float:
    return float((day - origin).days * _HOURS_PER_DAY)


def _yyyymmdd(value: int) -> date:
    return date(value // 10000, value // 100 % 100, value % 100)


def stage_windows(stages: Sequence[StageInfo]) -> tuple[Window, ...]:
    """Each stage's `[start, end)` in hours from the study start.

    Raises:
        MappingError: the stages are not contiguous.
    """
    origin = date.fromisoformat(stages[0].start_date)
    windows = tuple(
        (
            _hours_since(origin, date.fromisoformat(stage.start_date)),
            _hours_since(origin, date.fromisoformat(stage.end_date)),
        )
        for stage in stages
    )
    for index in range(1, len(windows)):
        if windows[index][0] != windows[index - 1][1]:
            raise MappingError(
                f"stage {stages[index].id} starts {windows[index][0]} h after the "
                f"study start but the stage before it ends at "
                f"{windows[index - 1][1]} h; release lags need contiguous stages"
            )
    return windows


def release_window(windows: Sequence[Window], stage: int) -> Window:
    """The release period of *stage*; negative stages precede the study."""
    if stage >= 0:
        return windows[stage]
    width = windows[0][1] - windows[0][0]
    return (width * stage, width * (stage + 1))


def _overlap(first: Window, second: Window) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _arrival(release: Window, hours: int) -> Window:
    return (release[0] + hours, release[1] + hours)


def _stage_arrival_shares(
    windows: Sequence[Window], stage: int, hours: int
) -> tuple[float, ...]:
    """Shares of each lagged release arriving within *stage*, to the last non-zero.

    Release periods tile time, so their arrival spans do too: the last lag
    before the arrivals end ahead of *stage* is the last with a share.
    """
    shares: list[float] = []
    lag = 0
    while True:
        release = release_window(windows, stage - lag)
        arrival = _arrival(release, hours)
        if arrival[1] <= windows[stage][0]:
            break
        shares.append(_overlap(arrival, windows[stage]) / (release[1] - release[0]))
        lag += 1
    return tuple(shares)


def travel_time_plants(
    hydros: Sequence[HydroInfo],
    source: Mapping[int, SourceHydro],
    deck_order: Sequence[int],
    stages: Sequence[StageInfo],
) -> tuple[TravelTimePlant, ...]:
    """The plants with a travel time, ordered like their codes in *deck_order*.

    *deck_order* is the deck's `VI` record order, which numbers DECOMP's
    travel-time plants; ascending code order is not it.

    Raises:
        MappingError: a travel time is not a whole number of hours, its plant
            has no downstream plant or no `VI` record in the deck.
    """
    delayed = [
        (hydro, travel_hours)
        for hydro in hydros
        if (travel_hours := hydro.travel_time_hours)
    ]
    if not delayed:
        return ()
    windows = stage_windows(stages)
    plants: list[TravelTimePlant] = []
    for hydro, travel_hours in delayed:
        hours = int(travel_hours)
        if hours != travel_hours:
            raise MappingError(
                f"hydro {hydro.id} ({hydro.name}) has a travel time of "
                f"{travel_hours} h, but DECOMP records whole hours"
            )
        if hydro.downstream_id is None:
            raise MappingError(
                f"hydro {hydro.id} ({hydro.name}) has a travel time but no "
                "downstream plant for its water to reach"
            )
        plants.append(
            TravelTimePlant(
                hydro_id=hydro.id,
                code=source[hydro.id].code,
                downstream_id=hydro.downstream_id,
                hours=hours,
                arrival_shares=tuple(
                    _stage_arrival_shares(windows, stage, hours)
                    for stage in range(len(windows))
                ),
            )
        )
    rank = {code: position for position, code in enumerate(deck_order)}
    unlisted = sorted(plant.code for plant in plants if plant.code not in rank)
    if unlisted:
        raise MappingError(
            f"plants {unlisted} have a travel time in the case but no VI record "
            "in the deck; export from the deck the case was converted from"
        )
    plants.sort(key=lambda plant: rank[plant.code])
    _LOG.info(
        "premise P3: travel time for plants %s, max_lag %d",
        [(plant.code, plant.hours) for plant in plants],
        max_lag(plants),
    )
    return tuple(plants)


def _bucket_windows(
    slots: Sequence[EntitySlotRecord], origin: date, pool_id: int
) -> dict[int, list[tuple[int, Window]]]:
    """Each downstream plant's transit-bucket positions and arrival windows."""
    by_plant: dict[int, list[tuple[int, Window]]] = {}
    for position in index_slots(slots).transit_bucket:
        slot = slots[position]
        window = (
            _hours_since(origin, _yyyymmdd(slot.interval_start)),
            _hours_since(origin, _yyyymmdd(slot.interval_end)),
        )
        by_plant.setdefault(slot.entity_id, []).append((position, window))
    for plant_id, buckets in by_plant.items():
        ordered = sorted(window for _, window in buckets)
        for earlier, later in itertools.pairwise(ordered):
            if later[0] < earlier[1]:
                raise MappingError(
                    f"pool {pool_id}'s transit buckets for hydro {plant_id} "
                    f"overlap: {earlier} and {later} h"
                )
    return by_plant


def release_lag_weights(
    slots: Sequence[EntitySlotRecord],
    plants: Sequence[TravelTimePlant],
    stages: Sequence[StageInfo],
    stage: int,
    pool_id: int,
) -> NDArray[np.float64]:
    """The map from one pool's cut coefficients to its `pi_qdefp` block.

    Row `lag * len(plants) + u` holds, per slot, the share of plant `u`'s
    defluence over stage `stage - lag` that the slot's bucket receives, so
    `weights @ coefficients` prices each release in R$/hm³. The block is
    lag-major with plants inner, as DECOMP lays it out.

    Raises:
        MappingError: a release still in transit at the end of *stage* falls
            outside the pool's buckets or past `max_lag`, or a bucket belongs
            to no travel-time plant's downstream plant.
    """
    origin = date.fromisoformat(stages[0].start_date)
    buckets = _bucket_windows(slots, origin, pool_id)
    downstream = {plant.downstream_id for plant in plants}
    orphans = sorted(set(buckets) - downstream)
    if orphans:
        raise MappingError(
            f"pool {pool_id} carries transit buckets for hydros {orphans}, which "
            "no plant with a travel time flows into"
        )

    lags = max_lag(plants)
    weights = np.zeros((lags * len(plants), len(slots)), dtype=np.float64)
    if not plants:
        return weights
    windows = stage_windows(stages)
    stage_end = windows[stage][1]
    for index, plant in enumerate(plants):
        for lag in range(lags + 1):
            release = release_window(windows, stage - lag)
            arrival = _arrival(release, plant.hours)
            in_transit = (max(arrival[0], stage_end), arrival[1])
            transit_hours = in_transit[1] - in_transit[0]
            if transit_hours <= 0.0:
                continue
            if lag == lags:
                raise MappingError(
                    f"plant {plant.code}'s defluence {lag} stages before stage "
                    f"{stage} is still in transit at its end, beyond max_lag {lags}"
                )
            covered = 0.0
            for position, window in buckets.get(plant.downstream_id, ()):
                share = _overlap(in_transit, window)
                weights[lag * len(plants) + index, position] = share / (
                    release[1] - release[0]
                )
                covered += share
            if abs(covered - transit_hours) > _COVERAGE_TOLERANCE_HOURS:
                raise MappingError(
                    f"plant {plant.code}'s defluence {lag} stages before stage "
                    f"{stage} arrives over {in_transit} h, but pool {pool_id}'s "
                    f"transit buckets for hydro {plant.downstream_id} cover "
                    f"{covered} of its {transit_hours} h"
                )
    return weights
