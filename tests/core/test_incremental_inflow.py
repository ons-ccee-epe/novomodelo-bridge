"""Tests for the natural-to-incremental inflow attribution shared by both
conversion tracks."""

from __future__ import annotations

import numpy as np
import pytest

from novomodelo_bridge.core.incremental_inflow import (
    ARTIFICIAL_GAUGE,
    artificial_pass_through,
    incremental_by_plant,
)


def _series(*values: float) -> np.ndarray:
    return np.array(values, dtype=float)


class TestIncrementalByPlant:
    def test_plants_sharing_a_station_subtract_only_their_own_upstream(
        self,
    ) -> None:
        natural = {1: _series(10.0), 2: _series(30.0), 5: _series(100.0)}
        posto_of = {10: 1, 20: 2, 30: 5, 40: 5}
        upstream_of = {30: [10], 40: [20]}
        out = incremental_by_plant(natural, posto_of, upstream_of, [30, 40])
        assert out[30] == pytest.approx(_series(90.0))
        assert out[40] == pytest.approx(_series(70.0))

    def test_upstream_station_shared_by_two_plants_is_subtracted_once(
        self,
    ) -> None:
        natural = {1: _series(10.0), 5: _series(100.0)}
        posto_of = {10: 1, 11: 1, 30: 5}
        out = incremental_by_plant(natural, posto_of, {30: [10, 11]}, [30])
        assert out[30] == pytest.approx(_series(90.0))

    def test_upstream_plant_on_the_same_station_is_not_subtracted(self) -> None:
        natural = {5: _series(100.0)}
        posto_of = {10: 5, 30: 5}
        out = incremental_by_plant(natural, posto_of, {30: [10]}, [30])
        assert out[30] == pytest.approx(_series(100.0))

    def test_artificial_plant_is_zero_whatever_its_station_stores(self) -> None:
        natural = {1: _series(10.0, 20.0), ARTIFICIAL_GAUGE: _series(5.0, 5.0)}
        posto_of = {10: 1, 30: ARTIFICIAL_GAUGE}
        out = incremental_by_plant(natural, posto_of, {30: [10]}, [30])
        assert out[30] == pytest.approx(_series(0.0, 0.0))

    def test_artificial_plant_is_zero_without_a_stored_series(self) -> None:
        natural = {1: _series(10.0, 20.0)}
        out = incremental_by_plant(natural, {30: ARTIFICIAL_GAUGE}, {}, [30])
        assert out[30] == pytest.approx(_series(0.0, 0.0))

    def test_artificial_upstream_is_not_subtracted(self) -> None:
        natural = {ARTIFICIAL_GAUGE: _series(5.0), 5: _series(100.0)}
        posto_of = {10: ARTIFICIAL_GAUGE, 30: 5}
        out = incremental_by_plant(natural, posto_of, {30: [10]}, [30])
        assert out[30] == pytest.approx(_series(100.0))

    def test_zero_gauges_are_pinned_to_zero(self) -> None:
        natural = {1: _series(10.0), 7: _series(0.0)}
        posto_of = {10: 1, 30: 7}
        out = incremental_by_plant(natural, posto_of, {30: [10]}, [30], zero_gauges={7})
        assert out[30] == pytest.approx(_series(0.0))

    def test_plant_whose_station_has_no_series_is_left_out(self) -> None:
        out = incremental_by_plant({1: _series(10.0)}, {30: 9}, {}, [30])
        assert out == {}

    def test_series_of_different_lengths_are_rejected(self) -> None:
        natural = {1: _series(10.0), 2: _series(10.0, 20.0)}
        with pytest.raises(ValueError, match="different lengths"):
            incremental_by_plant(natural, {10: 1}, {}, [10])


class TestArtificialPassThrough:
    def test_gauged_plant_below_an_artificial_plant_with_gauged_upstream(
        self,
    ) -> None:
        posto_of = {10: 1, 20: ARTIFICIAL_GAUGE, 30: 5}
        upstream_of = {20: [10], 30: [20]}
        assert artificial_pass_through(posto_of, upstream_of, [10, 20, 30]) == [
            (30, 20)
        ]

    def test_gauged_ancestor_found_through_a_chain_of_artificial_plants(
        self,
    ) -> None:
        posto_of = {10: 1, 20: ARTIFICIAL_GAUGE, 21: ARTIFICIAL_GAUGE, 30: 5}
        upstream_of = {21: [10], 20: [21], 30: [20]}
        assert artificial_pass_through(posto_of, upstream_of, [30]) == [(30, 20)]

    def test_artificial_headwater_is_not_reported(self) -> None:
        posto_of = {20: ARTIFICIAL_GAUGE, 30: 5}
        assert artificial_pass_through(posto_of, {30: [20]}, [20, 30]) == []

    def test_artificial_plant_below_another_is_not_reported(self) -> None:
        posto_of = {10: 1, 20: ARTIFICIAL_GAUGE, 21: ARTIFICIAL_GAUGE}
        upstream_of = {20: [10], 21: [20]}
        assert artificial_pass_through(posto_of, upstream_of, [21]) == []
