"""Unit tests for Itaipu's per-frequency ``RI`` generation limits: the floors
(``convert_itaipu_frequency_min_generation``), the ceilings
(``convert_itaipu_frequency_max_generation``) and their merge into the
per-group availability (``itaipu_frequency_group_values``).

Tier 1 — pure Python, no optional dependency and no ``example/`` deck. The
``RI`` register is faked as a small pandas frame; the calendar is the same
synthetic-calendar idiom ``test_decomp_group_bounds.py`` uses.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from novomodelo_bridge.decomp.converters.hydro import (
    convert_itaipu_frequency_max_generation,
    convert_itaipu_frequency_min_generation,
    itaipu_frequency_group_values,
)
from novomodelo_bridge.decomp.group_bounds import GroupBoundEntry
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import build_operative_calendar
from tests.conftest import make_decomp_case

_ITAIPU_CODE = 66


def _calendar():
    """A 3-stage calendar, 3 blocks per stage."""
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


def _id_map(*, with_itaipu: bool = True):
    hydro_codes = (10, _ITAIPU_CODE) if with_itaipu else (10,)
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=hydro_codes)


def _ri_frame(rows: list[dict]) -> pd.DataFrame:
    """A minimal ``RI`` frame with the five fixed-width patamar slots per
    frequency, defaulting the unspecified slots to NaN (idecomp's shape)."""
    columns = ["estagio"]
    for prefix in (
        "geracao_minima_50_hz",
        "geracao_minima_60_hz",
        "geracao_maxima_50_hz",
        "geracao_maxima_60_hz",
    ):
        columns += [f"{prefix}_{k}" for k in range(1, 6)]
    return pd.DataFrame(rows, columns=columns)


def _dadger(ri: pd.DataFrame | None):
    return SimpleNamespace(ri=lambda df=True: ri)


def _case(ri: pd.DataFrame | None):
    return make_decomp_case(Path("unused"), dadger=_dadger(ri), calendar=_calendar())


def _itaipu_id() -> int:
    return _id_map().hydro_id(_ITAIPU_CODE)


class TestConvertItaipuFrequencyMinGeneration:
    def test_reads_both_frequencies_forward_filled(self) -> None:
        ri = _ri_frame(
            [
                {
                    "estagio": 1,
                    "geracao_minima_50_hz_1": 4267.2,
                    "geracao_minima_50_hz_2": 4073.2,
                    "geracao_minima_50_hz_3": 3491.7,
                    "geracao_minima_60_hz_1": 2000.0,
                    "geracao_minima_60_hz_2": 2000.0,
                    "geracao_minima_60_hz_3": 2000.0,
                },
                {
                    "estagio": 3,
                    "geracao_minima_50_hz_1": 3818.2,
                    "geracao_minima_50_hz_2": 3645.2,
                    "geracao_minima_50_hz_3": 3130.7,
                    "geracao_minima_60_hz_1": 1800.0,
                    "geracao_minima_60_hz_2": 1800.0,
                    "geracao_minima_60_hz_3": 1800.0,
                },
            ]
        )
        result = convert_itaipu_frequency_min_generation(_case(ri), _id_map())
        hid = _itaipu_id()
        # group 0 = 50 Hz: stage 0 explicit, stage 1 forward-filled, stage 2 explicit
        assert result[(hid, 0, 0)] == [4267.2, 4073.2, 3491.7]
        assert result[(hid, 0, 1)] == [4267.2, 4073.2, 3491.7]
        assert result[(hid, 0, 2)] == [3818.2, 3645.2, 3130.7]
        # group 1 = 60 Hz
        assert result[(hid, 1, 0)] == [2000.0, 2000.0, 2000.0]
        assert result[(hid, 1, 1)] == [2000.0, 2000.0, 2000.0]
        assert result[(hid, 1, 2)] == [1800.0, 1800.0, 1800.0]

    def test_no_ri_register_returns_empty(self) -> None:
        assert convert_itaipu_frequency_min_generation(_case(None), _id_map()) == {}

    def test_itaipu_not_operated_returns_empty(self) -> None:
        ri = _ri_frame(
            [
                {
                    "estagio": 1,
                    "geracao_minima_50_hz_1": 4267.2,
                    "geracao_minima_50_hz_2": 4073.2,
                    "geracao_minima_50_hz_3": 3491.7,
                    "geracao_minima_60_hz_1": 2000.0,
                    "geracao_minima_60_hz_2": 2000.0,
                    "geracao_minima_60_hz_3": 2000.0,
                }
            ]
        )
        assert (
            convert_itaipu_frequency_min_generation(
                _case(ri), _id_map(with_itaipu=False)
            )
            == {}
        )

    def test_block_count_mismatch_raises(self) -> None:
        # only two patamar values for a 3-block stage
        ri = _ri_frame(
            [
                {
                    "estagio": 1,
                    "geracao_minima_50_hz_1": 4267.2,
                    "geracao_minima_50_hz_2": 4073.2,
                    "geracao_minima_60_hz_1": 2000.0,
                    "geracao_minima_60_hz_2": 2000.0,
                }
            ]
        )
        with pytest.raises(ValueError, match="2 patamares"):
            convert_itaipu_frequency_min_generation(_case(ri), _id_map())

    def test_missing_estagio_one_base_raises(self) -> None:
        ri = _ri_frame(
            [
                {
                    "estagio": 3,
                    "geracao_minima_50_hz_1": 3818.2,
                    "geracao_minima_50_hz_2": 3645.2,
                    "geracao_minima_50_hz_3": 3130.7,
                    "geracao_minima_60_hz_1": 1800.0,
                    "geracao_minima_60_hz_2": 1800.0,
                    "geracao_minima_60_hz_3": 1800.0,
                }
            ]
        )
        with pytest.raises(ValueError, match="no row declares estágio 1"):
            convert_itaipu_frequency_min_generation(_case(ri), _id_map())

    def test_estagio_outside_calendar_raises(self) -> None:
        ri = _ri_frame(
            [
                {
                    "estagio": 1,
                    "geracao_minima_50_hz_1": 4267.2,
                    "geracao_minima_50_hz_2": 4073.2,
                    "geracao_minima_50_hz_3": 3491.7,
                    "geracao_minima_60_hz_1": 2000.0,
                    "geracao_minima_60_hz_2": 2000.0,
                    "geracao_minima_60_hz_3": 2000.0,
                },
                {
                    "estagio": 9,
                    "geracao_minima_50_hz_1": 3818.2,
                    "geracao_minima_50_hz_2": 3645.2,
                    "geracao_minima_50_hz_3": 3130.7,
                    "geracao_minima_60_hz_1": 1800.0,
                    "geracao_minima_60_hz_2": 1800.0,
                    "geracao_minima_60_hz_3": 1800.0,
                },
            ]
        )
        with pytest.raises(ValueError, match="outside the calendar"):
            convert_itaipu_frequency_min_generation(_case(ri), _id_map())


def _ri_with_maxima() -> pd.DataFrame:
    """Stage 1 declares a 50 Hz maximum below 7000 MW on its first block and
    stage 3 a 60 Hz maximum below it on its second; stage 2 inherits stage 1."""
    floors = {
        "geracao_minima_50_hz_1": 4267.2,
        "geracao_minima_50_hz_2": 4073.2,
        "geracao_minima_50_hz_3": 3491.7,
        "geracao_minima_60_hz_1": 2000.0,
        "geracao_minima_60_hz_2": 2000.0,
        "geracao_minima_60_hz_3": 2000.0,
    }
    return _ri_frame(
        [
            {
                "estagio": 1,
                **floors,
                "geracao_maxima_50_hz_1": 6400.0,
                "geracao_maxima_50_hz_2": 7000.0,
                "geracao_maxima_50_hz_3": 7000.0,
                "geracao_maxima_60_hz_1": 7000.0,
                "geracao_maxima_60_hz_2": 7000.0,
                "geracao_maxima_60_hz_3": 7000.0,
            },
            {
                "estagio": 3,
                **floors,
                "geracao_maxima_50_hz_1": 7000.0,
                "geracao_maxima_50_hz_2": 7000.0,
                "geracao_maxima_50_hz_3": 7000.0,
                "geracao_maxima_60_hz_1": 7000.0,
                "geracao_maxima_60_hz_2": 6000.0,
                "geracao_maxima_60_hz_3": 7000.0,
            },
        ]
    )


def _availability(mw: float) -> dict[tuple[int, int, int], GroupBoundEntry]:
    """Every group of plant 10 and of Itaipu at *mw* available power on all
    three stages."""
    id_map = _id_map()
    keys = [(id_map.hydro_id(10), 0)] + [(_itaipu_id(), group) for group in (0, 1)]
    return {
        (hydro_id, group, stage): GroupBoundEntry(
            max_generation_mw=mw, max_turbined_m3s=100.0
        )
        for hydro_id, group in keys
        for stage in range(3)
    }


class TestConvertItaipuFrequencyMaxGeneration:
    def test_reads_both_frequencies_forward_filled(self) -> None:
        result = convert_itaipu_frequency_max_generation(
            _case(_ri_with_maxima()), _id_map()
        )
        hid = _itaipu_id()
        assert result[(hid, 0, 0)] == [6400.0, 7000.0, 7000.0]
        assert result[(hid, 0, 1)] == [6400.0, 7000.0, 7000.0]
        assert result[(hid, 0, 2)] == [7000.0, 7000.0, 7000.0]
        assert result[(hid, 1, 0)] == [7000.0, 7000.0, 7000.0]
        assert result[(hid, 1, 2)] == [7000.0, 6000.0, 7000.0]

    def test_no_ri_register_returns_empty(self) -> None:
        assert convert_itaipu_frequency_max_generation(_case(None), _id_map()) == {}


class TestItaipuFrequencyGroupValues:
    def test_ceiling_is_the_lower_of_available_power_and_ri_maximum_per_block(
        self,
    ) -> None:
        values = itaipu_frequency_group_values(
            _case(_ri_with_maxima()), _id_map(), _availability(6500.0)
        )
        hid = _itaipu_id()
        assert values[(hid, 0, 0)].max_generation_mw == [6400.0, 6500.0, 6500.0]
        assert values[(hid, 1, 2)].max_generation_mw == [6500.0, 6000.0, 6500.0]

    def test_ri_maximum_above_available_power_keeps_a_stage_level_ceiling(
        self,
    ) -> None:
        values = itaipu_frequency_group_values(
            _case(_ri_with_maxima()), _id_map(), _availability(6000.0)
        )
        hid = _itaipu_id()
        assert values[(hid, 0, 0)].max_generation_mw == 6000.0
        assert values[(hid, 1, 2)].max_generation_mw == 6000.0

    def test_floors_land_on_min_generation_and_keep_the_turbined_limit(
        self,
    ) -> None:
        values = itaipu_frequency_group_values(
            _case(_ri_with_maxima()), _id_map(), _availability(6500.0)
        )
        entry = values[(_itaipu_id(), 0, 1)]
        assert entry.min_generation_mw == [4267.2, 4073.2, 3491.7]
        assert entry.max_turbined_m3s == 100.0

    def test_other_plants_pass_through_unchanged(self) -> None:
        availability = _availability(6500.0)
        values = itaipu_frequency_group_values(
            _case(_ri_with_maxima()), _id_map(), availability
        )
        plant = _id_map().hydro_id(10)
        assert all(
            values[(plant, 0, s)] == availability[(plant, 0, s)] for s in range(3)
        )

    def test_no_ri_register_returns_the_availability_unchanged(self) -> None:
        availability = _availability(6500.0)
        assert itaipu_frequency_group_values(_case(None), _id_map(), availability) == (
            availability
        )

    def test_per_block_available_power_raises_type_error(self) -> None:
        availability = _availability(6500.0)
        key = (_itaipu_id(), 0, 0)
        availability[key] = GroupBoundEntry(max_generation_mw=[6500.0] * 3)
        with pytest.raises(TypeError, match="stage-level float"):
            itaipu_frequency_group_values(
                _case(_ri_with_maxima()), _id_map(), availability
            )
