"""Tier-1 tests for the contribution-native bound converters.

Synthetic ``_StubDadger``/``EffectiveCadastro``/``ConstraintCensus``/
``OperativeStage`` fixtures only — no deck, no ``example/`` read, no
``import novomodelo`` (mirrors ``test_single_term_bounds.py``'s fixture
style). Covers:

- ``convert_hydro_bounds``/``convert_storage_bounds`` return
  ``list[BoundContribution]``; ``convert_thermal_bounds`` returns a
  ``ThermalBounds`` pair (contributions + cost side-table) — no converter
  returns a ``pa.Table`` bound table.
- ``RQ``/``UH`` outflow floors are stage-level (``block_id = None``): the
  ``RQ`` register carries one percentage per stage, carried forward past its
  columns. The replace-vs-intersect discipline (A.1/A.3) applies to the
  thermal ``CT`` axis: a block-uniform stage contributes one base
  contribution, a non-uniform stage per-block contributions only, never
  both.
- an RQ minimum-outflow contribution and an RHQ ``QDEF`` outflow
  contribution on the same ``(hydro, stage, block)`` cell intersect to one
  row via ``bounds_accumulator.resolve``.
- ``network.pumping_station_id_map`` and ``convert_pumping_stations``
  agree on every ``UE`` row's id (single authority).
- a small uniform / no-override synthetic deck resolves to the exact
  expected row set, including a cell fanning two axes (outflow + storage)
  from two different contributors into one row.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from novomodelo_bridge.decomp.bounds_accumulator import (
    BoundContribution,
    build_bound_tables,
    resolve,
)
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.constraint_registers import (
    ConstraintCensus,
    ConstraintRecord,
    ConstraintTerm,
    StageBounds,
)
from novomodelo_bridge.decomp.converters.bounds import (
    convert_hydro_bounds,
    convert_storage_bounds,
)
from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
from novomodelo_bridge.decomp.converters.network import (
    convert_pumping_stations,
    pumping_station_id_map,
)
from novomodelo_bridge.decomp.converters.single_term_bounds import (
    single_term_bound_contributions,
)
from novomodelo_bridge.decomp.converters.thermal import (
    ThermalBounds,
    convert_thermal_bounds,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.pipeline import _row_group_contributions
from novomodelo_bridge.decomp.temporal import OperativeStage
from tests.conftest import make_decomp_case


class _StubDadger:
    """Minimal ``Dadger``-shaped stub keying record accessors by dataframe
    (mirrors ``test_decomp_rq_bounds.py``/``test_decomp_idecomp112.py``)."""

    def __init__(self, **frames: pd.DataFrame) -> None:
        self._frames = frames

    def __getattr__(self, name: str):
        if name in self._frames:
            frame = self._frames[name]
            return lambda df=False, **kwargs: frame  # noqa: ARG005
        raise AttributeError(name)


def _stage(index: int, block_hours: tuple[float, ...]) -> OperativeStage:
    return OperativeStage(
        index=index,
        start_date=date(2026, 7, 4),
        end_date=date(2026, 7, 11),
        season_id=6,
        block_hours=block_hours,
    )


def _hidr_frame(code: int, **columns: float) -> pd.DataFrame:
    df = pd.DataFrame({code: columns}).T
    df.index.name = "codigo_usina"
    return df


def _effective(hidr: pd.DataFrame, n_stages: int = 1) -> EffectiveCadastro:
    return EffectiveCadastro(base=hidr, n_stages=n_stages, stage_varying={})


def _case(dadger: _StubDadger, calendar: list[OperativeStage]) -> DecompCase:
    return make_decomp_case(Path("unused"), dadger=dadger, calendar=calendar)


def _rq_dadger(pct_stages: list[float], *, code: int = 1, ree: int = 1) -> _StubDadger:
    # ``pct_stages[k]`` is the RQ percentage for stage ``k`` (``vazao_{k+1}``).
    uh = pd.DataFrame(
        [
            {
                "codigo_usina": code,
                "codigo_ree": ree,
                "volume_inicial": 50.0,
                "vazao_defluente_minima": None,
            }
        ]
    )
    rq_row: dict[str, object] = {"codigo_ree": ree}
    for i, value in enumerate(pct_stages, start=1):
        rq_row[f"vazao_{i}"] = value
    return _StubDadger(uh=uh, rq=pd.DataFrame([rq_row]))


def _ct_dadger(
    disp_blocks: list[float],
    inflex_blocks: list[float],
    *,
    code: int = 1,
    cvu: float = 10.0,
) -> _StubDadger:
    row: dict[str, object] = {
        "codigo_usina": code,
        "estagio": 1,
        "nome_usina": "T1",
        "codigo_submercado": 1,
    }
    for i, value in enumerate(disp_blocks, start=1):
        row[f"disponibilidade_{i}"] = value
    for i, value in enumerate(inflex_blocks, start=1):
        row[f"inflexibilidade_{i}"] = value
    for i in range(1, len(disp_blocks) + 1):
        row[f"cvu_{i}"] = cvu
    return _StubDadger(ct=pd.DataFrame([row]))


class TestContributionNativeReturnTypes:
    """No converter returns a ``pa.Table`` bound table."""

    def test_convert_hydro_bounds_returns_contribution_list(self) -> None:
        calendar = [_stage(0, (10.0,))]
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
        effective = _effective(_hidr_frame(1, vazao_minima_historica=40.0))

        contributions = convert_hydro_bounds(
            _case(_rq_dadger([50.0]), calendar), id_map, effective=effective
        )

        assert isinstance(contributions, list)
        assert all(isinstance(c, BoundContribution) for c in contributions)

    def test_convert_storage_bounds_returns_contribution_list(self) -> None:
        calendar = [_stage(0, (10.0,))]
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
        effective = EffectiveCadastro(
            base=_hidr_frame(1, volume_minimo=20.0, volume_maximo=100.0),
            n_stages=1,
            stage_varying={(1, "volume_maximo"): (250.0,)},
        )

        contributions = convert_storage_bounds(
            _case(_StubDadger(), calendar), id_map, effective=effective
        )

        assert isinstance(contributions, list)
        assert all(isinstance(c, BoundContribution) for c in contributions)
        assert all(c.axis == "storage" and c.block_id is None for c in contributions)

    def test_convert_thermal_bounds_returns_contributions_plus_cost_table(
        self,
    ) -> None:
        calendar = [_stage(0, (10.0, 20.0))]
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), thermal_codes=(1,))

        bounds = convert_thermal_bounds(
            _case(_ct_dadger([100.0, 100.0], [0.0, 0.0]), calendar), id_map
        )

        assert isinstance(bounds, ThermalBounds)
        assert isinstance(bounds.generation, list)
        assert all(isinstance(c, BoundContribution) for c in bounds.generation)
        assert bounds.cost.column_names == [
            "thermal_id",
            "stage_id",
            "block_id",
            "cost_per_mwh",
        ]


class TestReplaceVsIntersectDiscipline:
    """``RQ``/``UH`` outflow floors are stage-level (``block_id = None``);
    ``RQ`` carries one percentage per stage. Thermal (``CT``) is per-block,
    so a non-uniform CT stage contributes per-block-only (A.1/A.3's critical
    trap for the thermal axis)."""

    def test_hydro_rq_single_stage_emits_stage_level_contribution(self) -> None:
        calendar = [_stage(0, (10.0, 10.0))]
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
        effective = _effective(_hidr_frame(1, vazao_minima_historica=40.0))

        contributions = convert_hydro_bounds(
            _case(_rq_dadger([50.0]), calendar), id_map, effective=effective
        )

        assert len(contributions) == 1
        contribution = contributions[0]
        assert contribution.block_id is None
        assert contribution.axis == "outflow"
        assert contribution.lower == pytest.approx(20.0)  # 50% of 40 m3/s
        assert contribution.upper is None

    def test_hydro_rq_columns_are_per_stage(self) -> None:
        # Two RQ columns (100, 0) over a two-stage calendar map to two stages:
        # stage 0 gets 100% (40 m3/s, one stage-level row), stage 1 gets 0%
        # (dropped by the non-positive gate).
        calendar = [_stage(0, (10.0, 20.0)), _stage(1, (10.0, 20.0))]
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
        effective = _effective(_hidr_frame(1, vazao_minima_historica=40.0), n_stages=2)

        contributions = convert_hydro_bounds(
            _case(_rq_dadger([100.0, 0.0]), calendar), id_map, effective=effective
        )

        assert len(contributions) == 1
        contribution = contributions[0]
        assert contribution.stage_id == 0
        assert contribution.block_id is None
        assert contribution.axis == "outflow"
        assert contribution.lower == pytest.approx(40.0)

    def test_hydro_rq_carries_last_stage_percentage_forward(self) -> None:
        # One RQ column (50) over a two-stage calendar: stage 0 uses the
        # declared 50%, stage 1 (past the register's columns) carries the
        # last declared percentage forward — both stage-level.
        calendar = [_stage(0, (168.0,)), _stage(1, (168.0,))]
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
        effective = _effective(_hidr_frame(1, vazao_minima_historica=40.0), n_stages=2)

        contributions = convert_hydro_bounds(
            _case(_rq_dadger([50.0]), calendar), id_map, effective=effective
        )

        assert len(contributions) == 2
        assert all(c.block_id is None and c.axis == "outflow" for c in contributions)
        by_stage = {c.stage_id: c.lower for c in contributions}
        assert by_stage == {0: pytest.approx(20.0), 1: pytest.approx(20.0)}

    def test_thermal_uniform_stage_emits_single_base_contribution(self) -> None:
        calendar = [_stage(0, (10.0, 20.0))]
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), thermal_codes=(1,))

        bounds = convert_thermal_bounds(
            _case(_ct_dadger([100.0, 100.0], [0.0, 0.0], cvu=10.0), calendar), id_map
        )

        assert len(bounds.generation) == 1
        contribution = bounds.generation[0]
        assert contribution.block_id is None
        assert contribution.axis == "generation"
        assert contribution.lower == 0.0
        assert contribution.upper == 100.0
        # The cost side-table is unaffected by the generation split — always
        # one base-row-only entry per (thermal, stage).
        assert bounds.cost.num_rows == 1
        assert bounds.cost["cost_per_mwh"].to_pylist() == [10.0]

    def test_thermal_nonuniform_stage_emits_per_block_only_no_base(self) -> None:
        calendar = [_stage(0, (10.0, 20.0))]
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), thermal_codes=(1,))

        bounds = convert_thermal_bounds(
            _case(_ct_dadger([100.0, 50.0], [0.0, 0.0], cvu=10.0), calendar), id_map
        )

        assert len(bounds.generation) == 2
        assert all(c.block_id is not None for c in bounds.generation)
        by_block = {c.block_id: c.upper for c in bounds.generation}
        assert by_block == {0: 100.0, 1: 50.0}
        # cost still rides its own base-row-only side table, independent of
        # the generation axis's own base-vs-per-block split.
        assert bounds.cost.num_rows == 1


class TestCollisionIntersection:
    """An RQ minimum-outflow contribution and an RHQ ``QDEF`` outflow
    contribution on the same ``(hydro, stage, block)`` cell intersect to
    exactly one row with the max-lower value — no duplicate cell."""

    def test_collision_intersects(self) -> None:
        calendar = [_stage(0, (168.0,))]
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
        effective = _effective(_hidr_frame(1, vazao_minima_historica=40.0))

        rq_contribs = convert_hydro_bounds(
            _case(_rq_dadger([50.0]), calendar), id_map, effective=effective
        )
        assert rq_contribs == [
            BoundContribution(
                family="hydro",
                entity_id=0,
                stage_id=0,
                block_id=None,
                axis="outflow",
                lower=20.0,
                upper=None,
                contributor="RQ",
            )
        ]

        qdef_record = ConstraintRecord(
            family="HQ",
            constraint_id=200,
            stage_start=0,
            stage_end=0,
            terms=(ConstraintTerm(code=1, coefficient=1.0, variable="QDEF"),),
            bounds={0: StageBounds(lower=(30.0,), upper=(None,))},
            per_block=True,
        )
        census = ConstraintCensus(
            by_family={"HQ": (qdef_record,)}, to_bounds=(qdef_record,)
        )
        rhq_contribs = single_term_bound_contributions(
            _case(_StubDadger(), calendar),
            id_map,
            census=census,
            pumping_station_ids={},
            effective=effective,
            hydro_capacities={},
        )
        assert len(rhq_contribs) == 1

        rows = resolve([*rq_contribs, *rhq_contribs], {0: 1})

        assert len(rows) == 1
        row = rows[0]
        assert row.family == "hydro"
        assert row.entity_id == 0
        assert row.stage_id == 0
        assert row.block_id == 0
        assert row.axis == "outflow"
        assert row.lower == pytest.approx(30.0)  # max(20.0 RQ, 30.0 RHQ)
        assert row.upper is None

    def test_line_direct_axis_collision_intersects_to_min_upper(self) -> None:
        """A colliding pair on the ``line`` family's ``direct`` axis (now
        routed through the accumulator) intersects to the
        min-of-uppers -- an upper-only axis has no lower side to raise."""
        contribs = [
            BoundContribution(
                family="line",
                entity_id=0,
                stage_id=0,
                block_id=None,
                axis="direct",
                lower=None,
                upper=500.0,
                contributor="IA",
            ),
            BoundContribution(
                family="line",
                entity_id=0,
                stage_id=0,
                block_id=None,
                axis="direct",
                lower=None,
                upper=300.0,
                contributor="RE_12",
            ),
        ]

        rows = resolve(contribs, {0: 1})

        assert len(rows) == 1
        row = rows[0]
        assert row.family == "line"
        assert row.entity_id == 0
        assert row.stage_id == 0
        assert row.block_id is None
        assert row.axis == "direct"
        assert row.lower is None
        assert row.upper == pytest.approx(300.0)  # min(500.0, 300.0)


class TestRowGroupContributionsAsymmetricBaseOnlyColumn:
    """``hydro_unit_group`` bounds can carry one axis side that varies per
    block (``min_generation_mw``) alongside a sibling side that never does
    (``max_generation_mw`` -- a base-only ceiling). ``_row_group_contributions``
    must feed the base-only side into ``resolve()`` as its own base
    contribution so it folds onto every materialized block row, rather than
    being dropped along with the (block-varying) side's now-redundant base
    average."""

    def test_base_only_max_survives_onto_every_block_row(self) -> None:
        rows = [
            {
                "hydro_unit_group_id": 0,
                "stage_id": 0,
                "block_id": None,
                "min_turbined_m3s": None,
                "max_turbined_m3s": None,
                "min_generation_mw": 6300.0,
                "max_generation_mw": 7000.0,
            },
            {
                "hydro_unit_group_id": 0,
                "stage_id": 0,
                "block_id": 0,
                "min_turbined_m3s": None,
                "max_turbined_m3s": None,
                "min_generation_mw": 6503.0,
                "max_generation_mw": None,
            },
            {
                "hydro_unit_group_id": 0,
                "stage_id": 0,
                "block_id": 1,
                "min_turbined_m3s": None,
                "max_turbined_m3s": None,
                "min_generation_mw": 6139.0,
                "max_generation_mw": None,
            },
        ]

        contribs = _row_group_contributions(
            rows,
            family="hydro_unit_group",
            id_column="hydro_unit_group_id",
            axes=("turbined", "generation"),
            contributor="group_bounds",
        )

        resolved = resolve(contribs, {0: 2})
        generation_rows = {r.block_id: r for r in resolved if r.axis == "generation"}

        assert set(generation_rows) == {0, 1}
        assert all(row.block_id is not None for row in generation_rows.values())
        assert generation_rows[0].lower == pytest.approx(6503.0)
        assert generation_rows[0].upper == pytest.approx(7000.0)
        assert generation_rows[1].lower == pytest.approx(6139.0)
        assert generation_rows[1].upper == pytest.approx(7000.0)
        # turbined has no value on either the base or the block rows --
        # it must resolve to nothing, never a spurious empty row.
        assert not any(row.axis == "turbined" for row in resolved)


class TestWaterWithdrawalBaseOnlyAxis:
    """``("hydro", "water_withdrawal")`` is registered ``block_eligible =
    False`` — a hydro with a withdrawal value and no per-block
    bound on the same (hydro, stage) must resolve to exactly the base row,
    never a fabricated per-block row, even when the stage carries multiple
    blocks. A deck that declares no withdrawal at all must leave
    ``hydro_bounds`` untouched rather than resolving an empty contribution
    list."""

    def test_withdrawal_only_resolves_to_base_row_no_fabricated_blocks(
        self,
    ) -> None:
        contribs = [
            BoundContribution(
                family="hydro",
                entity_id=0,
                stage_id=0,
                block_id=None,
                axis="water_withdrawal",
                lower=12.5,
                upper=None,
                contributor="convert_irrigation_withdrawal",
            )
        ]

        rows = resolve(contribs, {0: 3})

        assert len(rows) == 1
        row = rows[0]
        assert row.family == "hydro"
        assert row.entity_id == 0
        assert row.stage_id == 0
        assert row.block_id is None
        assert row.axis == "water_withdrawal"
        assert row.lower == pytest.approx(12.5)
        assert row.upper is None

    def test_no_withdrawal_rows_is_a_noop(self) -> None:
        """No accumulator resolution happens when the deck declares no
        irrigation withdrawal -- ``hydro_bounds`` returns unchanged."""
        import pyarrow as pa

        from novomodelo_bridge.decomp.pipeline import _attach_water_withdrawal

        hydro_bounds = pa.table(
            {
                "hydro_id": pa.array([1], pa.int32()),
                "stage_id": pa.array([0], pa.int32()),
                "block_id": pa.array([None], pa.int32()),
            }
        )
        assert _attach_water_withdrawal(hydro_bounds, None, {0: 1}) is hydro_bounds


class TestPumpingStationIdMapSingleAuthority:
    """``pumping_station_id_map`` and ``convert_pumping_stations``
    agree on every ``UE`` row's id."""

    def test_ids_agree_for_every_ue_row(self) -> None:
        ue = pd.DataFrame(
            [
                {
                    "codigo_usina": 30,
                    "codigo_usina_jusante": 1,
                    "codigo_usina_montante": 2,
                    "codigo_submercado": 1,
                    "nome_usina": "UE30",
                    "taxa_consumo": 0.1,
                    "vazao_minima_bombeavel": 0.0,
                    "vazao_maxima_bombeavel": 50.0,
                },
                {
                    "codigo_usina": 10,
                    "codigo_usina_jusante": 1,
                    "codigo_usina_montante": 2,
                    "codigo_submercado": 1,
                    "nome_usina": "UE10",
                    "taxa_consumo": 0.2,
                    "vazao_minima_bombeavel": 0.0,
                    "vazao_maxima_bombeavel": 80.0,
                },
            ]
        )
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1, 2))
        dadger = _StubDadger(ue=ue)

        station_ids = pumping_station_id_map(dadger)
        # Position in codigo_usina-sorted order: 10 -> 0, 30 -> 1.
        assert station_ids == {10: 0, 30: 1}

        doc = convert_pumping_stations(_case(dadger, [_stage(0, (10.0,))]), id_map)
        stations_by_name = {s["name"]: s for s in doc["pumping_stations"]}
        for name, code in (("UE30", 30), ("UE10", 10)):
            assert stations_by_name[name]["id"] == station_ids[code]


class TestByteIdenticalRegression:
    """A uniform / no-override synthetic deck resolves to exactly the
    expected row set — including one cell fanning two axes (outflow +
    storage) from two different contributors into one row, never a
    duplicate."""

    def test_uniform_no_override_deck_resolves_to_the_expected_rows(self) -> None:
        calendar = [_stage(0, (168.0,)), _stage(1, (168.0,))]
        id_map = DecompIdMap(
            bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,), thermal_codes=(1,)
        )
        hidr = _hidr_frame(
            1,
            vazao_minima_historica=40.0,
            volume_minimo=20.0,
            volume_maximo=100.0,
        )
        effective = EffectiveCadastro(
            base=hidr,
            n_stages=2,
            # Stage 1 (0-based) widens volume_maximo past the envelope
            # (100.0 -> the widened value becomes the new envelope ceiling,
            # per storage_envelope/effective_storage_range's own Rule A), so
            # the *earlier* stage 0 differs from the resulting envelope and
            # contributes an override; stage 1 equals it and contributes
            # nothing on its own for storage.
            stage_varying={(1, "volume_maximo"): (100.0, 250.0)},
        )

        hydro_contribs = [
            *convert_hydro_bounds(
                _case(_rq_dadger([50.0]), calendar), id_map, effective=effective
            ),
            *convert_storage_bounds(
                _case(_StubDadger(), calendar), id_map, effective=effective
            ),
        ]
        thermal_bounds = convert_thermal_bounds(
            _case(_ct_dadger([100.0, 100.0], [0.0, 0.0], cvu=10.0), calendar), id_map
        )

        block_counts = {stage.index: len(stage.block_hours) for stage in calendar}
        tables = build_bound_tables(
            resolve([*hydro_contribs, *thermal_bounds.generation], block_counts)
        )

        hydro_rows = {
            (row["hydro_id"], row["stage_id"], row["block_id"]): row
            for row in tables.hydro.to_pylist()
        }
        assert set(hydro_rows) == {(0, 0, None), (0, 1, None)}
        # Stage 0: the RQ outflow contribution AND the storage override
        # (narrower than the widened envelope) fan into the SAME one row.
        stage0 = hydro_rows[(0, 0, None)]
        assert stage0["min_outflow_m3s"] == pytest.approx(20.0)
        assert stage0["min_storage_hm3"] == pytest.approx(20.0)
        assert stage0["max_storage_hm3"] == pytest.approx(100.0)
        # Stage 1: only the RQ outflow contribution — its own storage range
        # now equals the (widened) envelope, so it inherits it and
        # contributes no storage row.
        stage1 = hydro_rows[(0, 1, None)]
        assert stage1["min_outflow_m3s"] == pytest.approx(20.0)
        assert stage1["min_storage_hm3"] is None
        assert stage1["max_storage_hm3"] is None

        thermal_rows = {
            (row["thermal_id"], row["stage_id"], row["block_id"]): row
            for row in tables.thermal.to_pylist()
        }
        assert set(thermal_rows) == {(0, 0, None), (0, 1, None)}
        for row in thermal_rows.values():
            assert row["min_generation_mw"] == 0.0
            assert row["max_generation_mw"] == 100.0

        assert thermal_bounds.cost.num_rows == 2
        assert thermal_bounds.cost["cost_per_mwh"].to_pylist() == [10.0, 10.0]
