"""Tests for dessem/exporters/fcf/core/travel_time.py.

The arrival shares and the release-lag layout are the values the reference
DECOMP wrote to its own `mapcut` and `cortdeco` for a weekly stage of 168 h
followed by a monthly stage of 648 h, with travel times of 360 h and 100 h.
"""

from __future__ import annotations

import numpy as np
import pytest

from novomodelo_bridge.dessem.exporters.fcf.core.inputs_reader import (
    BlockInfo,
    HydroInfo,
    StageInfo,
)
from novomodelo_bridge.dessem.exporters.fcf.core.mapper import (
    ENTITY_TYPE_HYDRO_STORAGE,
    ENTITY_TYPE_HYDRO_TRANSIT_BUCKET,
    MappingError,
    SourceHydro,
)
from novomodelo_bridge.dessem.exporters.fcf.core.reader import UNDATED, EntitySlotRecord
from novomodelo_bridge.dessem.exporters.fcf.core.travel_time import (
    TravelTimePlant,
    max_lag,
    release_lag_weights,
    stage_windows,
    travel_time_plants,
)

_SOURCE = {
    0: SourceHydro(code=156, name="TRES MARIAS"),
    1: SourceHydro(code=162, name="QUEIMADO"),
    2: SourceHydro(code=169, name="SOBRADINHO"),
}


def _stage(stage_id: int, start: str, end: str) -> StageInfo:
    return StageInfo(
        id=stage_id,
        start_date=start,
        end_date=end,
        blocks=(BlockInfo(id=0, name="unico", hours=168.0),),
        storage_state=True,
        inflow_lags_state=False,
    )


_STAGES = (
    _stage(0, "2026-03-21", "2026-03-28"),
    _stage(1, "2026-03-28", "2026-04-24"),
)


def _hydro(hydro_id: int, downstream_id: int | None, hours: float | None) -> HydroInfo:
    return HydroInfo(
        id=hydro_id,
        name=_SOURCE[hydro_id].name,
        downstream_id=downstream_id,
        min_storage_hm3=0.0,
        max_storage_hm3=100.0,
        travel_time_hours=hours,
    )


def _hydros(
    upstream_hours: float | None = 360.0, upstream_downstream: int | None = 2
) -> tuple[HydroInfo, ...]:
    return (
        _hydro(0, upstream_downstream, upstream_hours),
        _hydro(1, 2, 100.0),
        _hydro(2, None, None),
    )


def _storage(hydro_id: int) -> EntitySlotRecord:
    return EntitySlotRecord(
        entity_type=ENTITY_TYPE_HYDRO_STORAGE,
        entity_id=hydro_id,
        subindex=0,
        was_active=True,
        reference_date=UNDATED,
        interval_start=UNDATED,
        interval_end=UNDATED,
    )


def _bucket(hydro_id: int, start: int, end: int) -> EntitySlotRecord:
    return EntitySlotRecord(
        entity_type=ENTITY_TYPE_HYDRO_TRANSIT_BUCKET,
        entity_id=hydro_id,
        subindex=0,
        was_active=True,
        reference_date=UNDATED,
        interval_start=start,
        interval_end=end,
    )


_STORAGE_SLOTS = (_storage(0), _storage(1), _storage(2))
_STAGE_0_BUCKETS = (
    _bucket(2, 20260321, 20260328),
    _bucket(2, 20260328, 20260404),
    _bucket(2, 20260404, 20260501),
)


class TestTravelTimePlants:
    def test_arrival_shares_match_the_reference_decomp(self) -> None:
        plants = travel_time_plants(_hydros(), _SOURCE, (156, 162), _STAGES)

        assert [plant.code for plant in plants] == [156, 162]
        long, short = plants
        assert long.arrival_shares[0] == pytest.approx((0.0, 0.0, 6 / 7, 1 / 7))
        assert long.arrival_shares[1] == pytest.approx((4 / 9, 1.0, 1.0, 1 / 7))
        assert short.arrival_shares[0] == pytest.approx((0.4047619, 0.5952381))
        assert short.arrival_shares[1] == pytest.approx((0.84567901, 0.5952381))
        assert (long.stage_lags, short.stage_lags, max_lag(plants)) == (
            (3, 3),
            (1, 1),
            3,
        )

    def test_plants_follow_the_vi_record_order_not_ascending_code(self) -> None:
        plants = travel_time_plants(_hydros(), _SOURCE, (162, 156), _STAGES)

        assert [plant.code for plant in plants] == [162, 156]

    def test_case_without_travel_time_has_no_plants(self) -> None:
        hydros = (_hydro(0, 2, None), _hydro(1, 2, 0.0), _hydro(2, None, None))

        assert travel_time_plants(hydros, _SOURCE, (), ()) == ()
        assert max_lag(()) == 0

    def test_plant_without_a_vi_record_is_refused(self) -> None:
        with pytest.raises(MappingError, match=r"plants \[162\] .* no VI record"):
            travel_time_plants(_hydros(), _SOURCE, (156,), _STAGES)

    def test_fractional_hours_are_refused(self) -> None:
        with pytest.raises(MappingError, match="whole hours"):
            travel_time_plants(_hydros(360.5), _SOURCE, (156, 162), _STAGES)

    def test_travel_time_without_a_downstream_plant_is_refused(self) -> None:
        with pytest.raises(MappingError, match="no downstream plant"):
            travel_time_plants(
                _hydros(upstream_downstream=None), _SOURCE, (156, 162), _STAGES
            )


def test_stages_with_a_gap_are_refused() -> None:
    stages = (_STAGES[0], _stage(1, "2026-03-29", "2026-04-24"))

    with pytest.raises(MappingError, match="contiguous stages"):
        stage_windows(stages)


class TestReleaseLagWeights:
    @pytest.fixture
    def plants(self) -> tuple[TravelTimePlant, ...]:
        return travel_time_plants(_hydros(), _SOURCE, (156, 162), _STAGES)

    def test_rows_are_lag_major_and_spread_each_release_over_its_buckets(
        self, plants: tuple[TravelTimePlant, ...]
    ) -> None:
        slots = _STORAGE_SLOTS + _STAGE_0_BUCKETS

        weights = release_lag_weights(slots, plants, _STAGES, 0, pool_id=0)

        expected = np.zeros((3 * 2, len(slots)))
        expected[0, 5] = 1.0
        expected[1, 4] = 100 / 168
        expected[2, 4:6] = (6 / 7, 1 / 7)
        expected[4, 4] = 1 / 7
        np.testing.assert_allclose(weights, expected)

    def test_no_plants_give_an_empty_block(self) -> None:
        weights = release_lag_weights(_STORAGE_SLOTS, (), _STAGES, 0, pool_id=0)

        assert weights.shape == (0, len(_STORAGE_SLOTS))

    def test_water_outside_every_bucket_is_refused(
        self, plants: tuple[TravelTimePlant, ...]
    ) -> None:
        slots = _STORAGE_SLOTS + _STAGE_0_BUCKETS[:2]

        with pytest.raises(MappingError, match="cover 0.0 of its 168.0 h"):
            release_lag_weights(slots, plants, _STAGES, 0, pool_id=0)

    def test_bucket_no_plant_flows_into_is_refused(
        self, plants: tuple[TravelTimePlant, ...]
    ) -> None:
        slots = _STORAGE_SLOTS + _STAGE_0_BUCKETS + (_bucket(0, 20260328, 20260404),)

        with pytest.raises(MappingError, match=r"hydros \[0\], which no plant"):
            release_lag_weights(slots, plants, _STAGES, 0, pool_id=0)

    def test_overlapping_buckets_are_refused(
        self, plants: tuple[TravelTimePlant, ...]
    ) -> None:
        slots = _STORAGE_SLOTS + _STAGE_0_BUCKETS + (_bucket(2, 20260401, 20260410),)

        with pytest.raises(MappingError, match="overlap"):
            release_lag_weights(slots, plants, _STAGES, 0, pool_id=0)

    def test_water_in_transit_past_max_lag_is_refused(self) -> None:
        plant = TravelTimePlant(
            hydro_id=0, code=156, downstream_id=2, hours=360, arrival_shares=((1.0,),)
        )
        slots = _STORAGE_SLOTS + _STAGE_0_BUCKETS

        with pytest.raises(MappingError, match="beyond max_lag 0"):
            release_lag_weights(slots, (plant,), _STAGES[:1], 0, pool_id=0)
