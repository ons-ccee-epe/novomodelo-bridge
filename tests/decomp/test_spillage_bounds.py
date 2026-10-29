"""Tests ``convert_spillage_upper_bounds`` (``UH`` field 9 spill ceiling)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from novomodelo_bridge.core.generic_constraint_builder import UNBOUNDED
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.bounds import convert_spillage_upper_bounds
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import OperativeStage, build_operative_calendar
from tests.conftest import make_decomp_case


class _StubDadger:
    """Minimal ``Dadger``-shaped stub keying record accessors by dataframe."""

    def __init__(self, **frames: pd.DataFrame) -> None:
        self._frames = frames

    def __getattr__(self, name: str):
        if name in self._frames:
            frame = self._frames[name]
            return lambda df=False, **kwargs: frame  # noqa: ARG005
        raise AttributeError(name)


class TestUhSpillCeiling:
    """Four plants: 1 finite ceiling; 2 at the ``1e21`` default; 3 blank
    ceiling; 4 coupling-only (no ``volume_inicial``). Calendar: five stages."""

    _ID_MAP = DecompIdMap(
        bus_codes=(1,),
        bus_names=("SE",),
        hydro_codes=(1, 2, 3, 4),
    )

    def _calendar(self) -> list[OperativeStage]:
        weekly = [[168.0]] * 4
        aggregate = [[744.0]]
        return build_operative_calendar(date(2026, 7, 4), weekly + aggregate)

    def _dadger(self) -> _StubDadger:
        uh = pd.DataFrame(
            [
                {
                    "codigo_usina": 1,
                    "volume_inicial": 50.0,
                    "limite_superior_vertimento": 1200.0,
                },
                {
                    "codigo_usina": 2,
                    "volume_inicial": 50.0,
                    "limite_superior_vertimento": UNBOUNDED,
                },
                {
                    "codigo_usina": 3,
                    "volume_inicial": 50.0,
                    "limite_superior_vertimento": None,
                },
                {
                    "codigo_usina": 4,
                    "volume_inicial": None,
                    "limite_superior_vertimento": 500.0,
                },
            ]
        )
        return _StubDadger(uh=uh)

    def _case(self, calendar: Sequence[OperativeStage]) -> DecompCase:
        return make_decomp_case(
            Path("unused"), dadger=self._dadger(), calendar=calendar
        )

    def _by_entity(self, contributions):
        for c in contributions:
            assert c.axis == "spillage"
            assert c.block_id is None
            assert c.lower is None
        return {c.entity_id for c in contributions}

    def test_finite_ceiling_emits_one_per_stage(self) -> None:
        calendar = self._calendar()
        contributions = convert_spillage_upper_bounds(
            self._case(calendar), self._ID_MAP
        )

        self._by_entity(contributions)
        plant1_id = self._ID_MAP.hydro_id(1)
        rows = [c for c in contributions if c.entity_id == plant1_id]
        assert {c.stage_id for c in rows} == {s.index for s in calendar}
        assert all(c.upper == 1200.0 for c in rows)

    def test_blank_ceiling_and_coupling_only_emit_nothing(self) -> None:
        calendar = self._calendar()
        contributions = convert_spillage_upper_bounds(
            self._case(calendar), self._ID_MAP
        )

        entities = self._by_entity(contributions)
        assert self._ID_MAP.hydro_id(1) in entities
        assert self._ID_MAP.hydro_id(3) not in entities  # blank ceiling
        assert self._ID_MAP.hydro_id(4) not in entities  # coupling-only

    def test_sentinel_default_passes_through_as_unbounded_upper(self) -> None:
        calendar = self._calendar()
        contributions = convert_spillage_upper_bounds(
            self._case(calendar), self._ID_MAP
        )
        plant2_rows = [
            c for c in contributions if c.entity_id == self._ID_MAP.hydro_id(2)
        ]
        assert all(c.upper is not None and c.upper >= UNBOUNDED for c in plant2_rows)

    def test_missing_column_emits_nothing(self) -> None:
        # UH frame without the field-9 column (older idecomp): no raise.
        calendar = self._calendar()
        uh = pd.DataFrame([{"codigo_usina": 1, "volume_inicial": 50.0}])
        case = make_decomp_case(
            Path("unused"), dadger=_StubDadger(uh=uh), calendar=calendar
        )
        assert convert_spillage_upper_bounds(case, self._ID_MAP) == []
