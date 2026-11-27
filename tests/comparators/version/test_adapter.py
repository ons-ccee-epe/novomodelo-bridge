"""Tests for ``comparators.version.adapter``: the encoding of a baseline into the
reference render slots, over literal frames."""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.analyze import build_results_dataset
from novomodelo_bridge.comparators.charts import _COST_MAP, _resolve_cost_categories
from novomodelo_bridge.comparators.model import PercentileData
from novomodelo_bridge.comparators.version.adapter import (
    PARCEL_OF_COST_COLUMN,
    BaselineFrames,
    BaselineSlots,
    encode_baseline,
)
from novomodelo_bridge.comparators.version.alignment import AlignedPair
from novomodelo_bridge.comparators.version.sides import SideInputs
from novomodelo_bridge.core.units import MEAN_MONTH_HOURS
from novomodelo_bridge.novomodelo.cost_categories import (
    NOVOMODELO_COST_COMPONENT_COLUMNS,
)

_COST_COLUMNS = (*NOVOMODELO_COST_COMPONENT_COLUMNS, "immediate_cost", "future_cost")
_BALANCE_VARIABLES = (
    "hydro_gen_mw",
    "thermal_gen_mw",
    "deficit_mw",
    "excess_mw",
    "net_load_mw",
)
_DECISION_BASIS_LABELS = ("TERMICA SEM GNL", "GNL COMANDADO")

_SIN_SCHEMA = {"stage": pl.Int64, "variable": pl.Utf8, "value": pl.Float64}
_BUS_SCHEMA = {"newave_code": pl.Int64, **_SIN_SCHEMA}
_SLACK_COLUMNS = (
    "water_withdrawal_violation_pos_m3s",
    "water_withdrawal_violation_neg_m3s",
    "evaporation_violation_pos_m3s",
    "evaporation_violation_neg_m3s",
)
_SLACK_SCHEMA = {
    "entity_id": pl.Int64,
    "stage_id": pl.Int64,
    **dict.fromkeys(_SLACK_COLUMNS, pl.Float64),
}
_TIMING_SCHEMA = {"iteration": pl.Int64, "total_seconds": pl.Float64}


def _wide(
    keys: Mapping[str, Sequence[int]], **values: Sequence[float | None]
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            **{name: pl.Series(ids, dtype=pl.Int64) for name, ids in keys.items()},
            **{
                name: pl.Series(column, dtype=pl.Float64)
                for name, column in values.items()
            },
        }
    )


def _baseline(hydros: Sequence[dict] = (), buses: Sequence[dict] = ()) -> SideInputs:
    return SideInputs(
        side="baseline",
        output_dir=Path("unused"),
        version=None,
        hydros=list(hydros),
        buses=list(buses),
        thermals=[],
        lines=[],
        risk_aversion={},
        partitions={},
    )


def _encode(
    frames: BaselineFrames | None = None,
    *,
    pairs: Sequence[AlignedPair] = (),
    cost_columns: Collection[str] = _COST_COLUMNS,
    balance_variables: Collection[str] = _BALANCE_VARIABLES,
    baseline: SideInputs | None = None,
) -> BaselineSlots:
    return encode_baseline(
        frames or BaselineFrames(),
        baseline or _baseline(),
        pairs,
        cost_columns=cost_columns,
        balance_variables=balance_variables,
    )


def _by_stage(frame: pl.DataFrame) -> dict[tuple[int, str], float]:
    return {(row[0], row[1]): row[2] for row in frame.iter_rows()}


def _by_bus(frame: pl.DataFrame) -> list[tuple[int, int, str, float]]:
    return list(frame.iter_rows())


class TestParcelOfCostColumn:
    def test_the_keys_are_the_cost_component_columns(self) -> None:
        assert set(PARCEL_OF_COST_COLUMN) == set(NOVOMODELO_COST_COMPONENT_COLUMNS)

    @pytest.mark.parametrize(
        ("label", "sources", "columns"),
        [
            (label, sources, columns)
            for label, sources, columns, _ in _COST_MAP
            if columns
        ],
    )
    def test_the_columns_of_a_row_share_one_of_its_source_labels(
        self, label: str, sources: list[str], columns: list[str]
    ) -> None:
        parcels = {PARCEL_OF_COST_COLUMN[column] for column in columns}

        assert len(parcels) == 1, label
        assert parcels <= set(sources), label

    def test_no_column_takes_a_decision_basis_label(self) -> None:
        assert set(PARCEL_OF_COST_COLUMN.values()).isdisjoint(_DECISION_BASIS_LABELS)

    def test_the_novomodelo_only_parcels_are_spelled_like_their_columns(self) -> None:
        columns = ["storage_violation_cost", "inflow_penalty_cost", "pumping_cost"]

        assert [PARCEL_OF_COST_COLUMN[column] for column in columns] == columns

    def test_the_map_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            PARCEL_OF_COST_COLUMN["thermal_cost"] = "X"  # ty: ignore[invalid-assignment] the write is the assertion


class TestCosts:
    def test_every_component_column_lands_on_its_parcel_label(self) -> None:
        breakdown = {
            column: float(position)
            for position, column in enumerate(NOVOMODELO_COST_COMPONENT_COLUMNS, 1)
        }
        expected: dict[str, float] = {}
        for column, value in breakdown.items():
            label = PARCEL_OF_COST_COLUMN[column]
            expected[label] = expected.get(label, 0.0) + value

        slots = _encode(BaselineFrames(cost_breakdown=breakdown))

        assert slots["nw_costs"] == expected
        assert set(slots["nw_costs"]) == set(PARCEL_OF_COST_COLUMN.values())

    def test_the_thermal_columns_sum_under_the_delivery_basis_label(self) -> None:
        breakdown = {
            "thermal_cost": 5.0e6,
            "anticipated_thermal_cost": 1.0e6,
            "storage_violation_cost": 2.0e3,
            "deficit_cost": 3.0e3,
        }

        nw_costs = _encode(BaselineFrames(cost_breakdown=breakdown))["nw_costs"]

        assert nw_costs == {
            "GERACAO TERMICA": 6.0e6,
            "storage_violation_cost": 2.0e3,
            "DEFICIT": 3.0e3,
        }

    def test_every_category_has_equal_sums_on_the_chart(self) -> None:
        breakdown = dict.fromkeys(NOVOMODELO_COST_COMPONENT_COLUMNS, 1.0e3)
        breakdown["thermal_cost"] = 5.0e6
        breakdown["anticipated_thermal_cost"] = 1.0e6

        nw_costs = _encode(BaselineFrames(cost_breakdown=breakdown))["nw_costs"]
        categories = _resolve_cost_categories(nw_costs, breakdown)

        assert categories
        assert all(reference == candidate for _, reference, candidate, _ in categories)
        assert {label for label, *_ in categories} <= {row[0] for row in _COST_MAP}

    def test_a_cost_column_that_is_not_cleared_fills_no_parcel(self) -> None:
        breakdown = {"thermal_cost": 5.0e6, "deficit_cost": 3.0e3}
        cleared = [column for column in _COST_COLUMNS if column != "deficit_cost"]

        nw_costs = _encode(
            BaselineFrames(cost_breakdown=breakdown), cost_columns=cleared
        )["nw_costs"]

        assert nw_costs == {"GERACAO TERMICA": 5.0e6}

    def test_values_stay_in_reais(self) -> None:
        nw_costs = _encode(BaselineFrames(cost_breakdown={"deficit_cost": 3.0e9}))[
            "nw_costs"
        ]

        assert nw_costs == {"DEFICIT": 3.0e9}

    def test_a_cleared_column_without_a_parcel_raises_naming_it(self) -> None:
        frames = BaselineFrames(cost_breakdown={"thermal_cost": 1.0, "total_cost": 9.0})

        with pytest.raises(ValueError, match="total_cost"):
            _encode(frames, cost_columns=[*_COST_COLUMNS, "total_cost"])

    def test_a_column_without_a_parcel_that_is_not_cleared_is_left_out(self) -> None:
        frames = BaselineFrames(cost_breakdown={"thermal_cost": 1.0, "total_cost": 9.0})

        assert _encode(frames)["nw_costs"] == {"GERACAO TERMICA": 1.0}


class TestSystemSeries:
    def _frames(self) -> BaselineFrames:
        return BaselineFrames(
            stage_costs=_wide(
                {"stage_id": [1]},
                immediate_cost=[2.0e9],
                future_cost=[4.0e9],
                thermal_cost_total=[5.0e8],
            ),
            hydro_means=_wide(
                {"entity_id": [1, 2], "stage_id": [1, 1]},
                stored_energy_final_mwh=[3.0e6, 1.0e6],
                incremental_inflow_energy_mw=[50.0, 70.0],
            ),
        )

    def test_each_variable_carries_its_unit(self) -> None:
        nw_sin = _encode(self._frames())["nw_sin"]

        series = _by_stage(nw_sin)
        assert series[(1, "COPER")] == pytest.approx(2000.0)
        assert series[(1, "CUSTO_FUTURO")] == pytest.approx(4000.0)
        assert series[(1, "CTERM")] == pytest.approx(500.0)
        assert series[(1, "EARMF")] == pytest.approx(4.0e6 / 730.5)
        assert series[(1, "EARMF")] * MEAN_MONTH_HOURS == pytest.approx(4.0e6)
        assert series[(1, "ENA")] == pytest.approx(120.0)
        assert len(series) == 5

    def test_the_frame_is_typed(self) -> None:
        assert _encode(self._frames())["nw_sin"].schema == _SIN_SCHEMA

    def test_energy_sums_every_plant_whatever_the_pairs(self) -> None:
        pairs = [AlignedPair("One", 1, 101)]

        series = _by_stage(_encode(self._frames(), pairs=pairs)["nw_sin"])

        assert series[(1, "ENA")] == pytest.approx(120.0)
        assert series[(1, "EARMF")] == pytest.approx(4.0e6 / MEAN_MONTH_HOURS)

    def test_the_rows_follow_the_stages(self) -> None:
        frames = BaselineFrames(
            stage_costs=_wide({"stage_id": [0, 1]}, immediate_cost=[1.0e6, 3.0e6]),
            hydro_means=_wide(
                {"entity_id": [1, 2, 1, 2], "stage_id": [0, 0, 1, 1]},
                incremental_inflow_energy_mw=[1.0, 2.0, 4.0, 8.0],
            ),
        )

        series = _by_stage(_encode(frames)["nw_sin"])

        assert series == {
            (0, "COPER"): pytest.approx(1.0),
            (1, "COPER"): pytest.approx(3.0),
            (0, "ENA"): pytest.approx(3.0),
            (1, "ENA"): pytest.approx(12.0),
        }

    @pytest.mark.parametrize(
        ("not_cleared", "variable"),
        [
            ("immediate_cost", "COPER"),
            ("future_cost", "CUSTO_FUTURO"),
            ("thermal_cost", "CTERM"),
        ],
    )
    def test_a_cost_variable_that_is_not_cleared_builds_no_row(
        self, not_cleared: str, variable: str
    ) -> None:
        cleared = [column for column in _COST_COLUMNS if column != not_cleared]

        series = _by_stage(_encode(self._frames(), cost_columns=cleared)["nw_sin"])

        assert (1, variable) not in series
        assert len(series) == 4

    def test_a_null_stage_cost_builds_no_row_and_no_zero(self) -> None:
        frames = BaselineFrames(
            stage_costs=_wide({"stage_id": [0, 1]}, immediate_cost=[1.0e6, None])
        )

        series = _by_stage(_encode(frames)["nw_sin"])

        assert series == {(0, "COPER"): pytest.approx(1.0)}

    def test_a_column_the_frames_lack_builds_no_variable(self) -> None:
        frames = BaselineFrames(
            stage_costs=_wide({"stage_id": [1]}, immediate_cost=[2.0e9]),
            hydro_means=_wide(
                {"entity_id": [1], "stage_id": [1]},
                stored_energy_final_mwh=[3.0e6],
            ),
        )

        series = _by_stage(_encode(frames)["nw_sin"])

        assert set(series) == {(1, "COPER"), (1, "EARMF")}

    def test_a_stage_with_a_null_plant_has_no_energy_total(self) -> None:
        frames = BaselineFrames(
            hydro_means=_wide(
                {"entity_id": [1, 2, 1, 2], "stage_id": [0, 0, 1, 1]},
                stored_energy_final_mwh=[3.0e6, None, 3.0e6, 1.0e6],
                incremental_inflow_energy_mw=[None, None, 1.0, 2.0],
            )
        )

        series = _by_stage(_encode(frames)["nw_sin"])

        assert set(series) == {(1, "EARMF"), (1, "ENA")}

    def test_a_hydro_frame_without_energy_columns_builds_no_row(self) -> None:
        frames = BaselineFrames(
            hydro_means=_wide({"entity_id": [1], "stage_id": [1]}, generation_mw=[1.0])
        )

        nw_sin = _encode(frames)["nw_sin"]

        assert nw_sin.is_empty()
        assert nw_sin.schema == _SIN_SCHEMA


class TestBusSeries:
    def _frames(self) -> BaselineFrames:
        return BaselineFrames(
            bus_aggregates=_wide(
                {"bus_id": [7], "stage_id": [0]},
                hydro_gen_mw_p10=[1.0],
                hydro_gen_mw_p50=[400.0],
                hydro_gen_mw_p90=[9.0],
                thermal_gen_mw_p50=[150.0],
                deficit_mw_p50=[0.0],
                excess_mw_p50=[12.0],
                net_load_mw_p50=[900.0],
            )
        )

    def test_a_cleared_variable_reads_its_median_column(self) -> None:
        balance = [name for name in _BALANCE_VARIABLES if name != "excess_mw"]

        slots = _encode(self._frames(), balance_variables=balance)

        assert _by_bus(slots["nw_market"]) == [
            (7, 0, "GHTOT", 400.0),
            (7, 0, "GTERM", 150.0),
            (7, 0, "DEFT", 0.0),
        ]
        assert _by_bus(slots["nw_net_load"]) == [(7, 0, "NET_LOAD", 900.0)]

    def test_the_excess_variable_is_read_when_cleared(self) -> None:
        market = _by_bus(_encode(self._frames())["nw_market"])

        assert (7, 0, "EXCESSO", 12.0) in market

    def test_the_frames_are_typed(self) -> None:
        slots = _encode(self._frames())

        assert slots["nw_market"].schema == _BUS_SCHEMA
        assert slots["nw_net_load"].schema == _BUS_SCHEMA

    def test_a_variable_that_is_not_cleared_leaves_its_frame_empty_and_typed(
        self,
    ) -> None:
        slots = _encode(self._frames(), balance_variables=["net_load_mw"])

        assert slots["nw_market"].is_empty()
        assert slots["nw_market"].schema == _BUS_SCHEMA
        assert _by_bus(slots["nw_net_load"]) == [(7, 0, "NET_LOAD", 900.0)]

    def test_a_column_the_frame_lacks_builds_no_variable(self) -> None:
        frames = BaselineFrames(
            bus_aggregates=_wide(
                {"bus_id": [7], "stage_id": [0]}, thermal_gen_mw_p50=[150.0]
            )
        )

        slots = _encode(frames)

        assert _by_bus(slots["nw_market"]) == [(7, 0, "GTERM", 150.0)]
        assert slots["nw_net_load"].is_empty()

    def test_a_null_value_builds_no_row_and_no_zero(self) -> None:
        frames = BaselineFrames(
            bus_aggregates=_wide(
                {"bus_id": [7, 8], "stage_id": [0, 0]},
                hydro_gen_mw_p50=[None, 5.0],
            )
        )

        assert _by_bus(_encode(frames)["nw_market"]) == [(8, 0, "GHTOT", 5.0)]


class TestNames:
    def test_each_map_is_keyed_by_the_baseline_id(self) -> None:
        baseline = _baseline(
            hydros=[{"id": 3, "name": "Furnas"}, {"id": 4, "name": "Itaipu"}],
            buses=[{"id": 0, "name": "SUDESTE"}],
        )

        slots = _encode(baseline=baseline)

        assert slots["nw_hydro_names"] == {3: "Furnas", 4: "Itaipu"}
        assert slots["nw_bus_names"] == {0: "SUDESTE"}


class TestHydroSlacks:
    def _frames(self) -> BaselineFrames:
        slacks = {
            name: [float(10 * position + stage) for stage in (0, 1, 0, 1)]
            for position, name in enumerate(_SLACK_COLUMNS, 1)
        }
        return BaselineFrames(
            hydro_means=_wide(
                {"entity_id": [3, 3, 5, 5], "stage_id": [0, 1, 0, 1]},
                generation_mw=[1.0, 2.0, 3.0, 4.0],
                **slacks,
            )
        )

    def test_a_paired_plant_carries_the_candidate_id(self) -> None:
        slacks = _encode(self._frames(), pairs=[AlignedPair("Furnas", 3, 103)])[
            "nw_hydro_slacks"
        ]

        assert slacks.schema == _SLACK_SCHEMA
        assert slacks.rows() == [
            (103, 0, 10.0, 20.0, 30.0, 40.0),
            (103, 1, 11.0, 21.0, 31.0, 41.0),
        ]

    def test_an_unpaired_plant_has_no_row(self) -> None:
        slacks = _encode(self._frames(), pairs=[AlignedPair("Furnas", 3, 103)])[
            "nw_hydro_slacks"
        ]

        assert 5 not in slacks["entity_id"].to_list()

    def test_each_plant_takes_its_own_pair(self) -> None:
        pairs = [AlignedPair("Furnas", 3, 103), AlignedPair("Itaipu", 5, 7)]

        slacks = _encode(self._frames(), pairs=pairs)["nw_hydro_slacks"]

        assert slacks["entity_id"].to_list() == [103, 103, 7, 7]
        assert slacks["stage_id"].to_list() == [0, 1, 0, 1]

    def test_no_pair_leaves_the_frame_empty_and_typed(self) -> None:
        slacks = _encode(self._frames())["nw_hydro_slacks"]

        assert slacks.is_empty()
        assert slacks.schema == _SLACK_SCHEMA

    def test_a_null_slack_stays_null(self) -> None:
        frames = BaselineFrames(
            hydro_means=_wide(
                {"entity_id": [3], "stage_id": [0]},
                **{name: [None] for name in _SLACK_COLUMNS},
            )
        )

        slacks = _encode(frames, pairs=[AlignedPair("Furnas", 3, 103)])[
            "nw_hydro_slacks"
        ]

        assert slacks.rows() == [(103, 0, None, None, None, None)]

    def test_a_slack_column_the_frame_lacks_is_null_never_zero(self) -> None:
        frames = BaselineFrames(
            hydro_means=_wide(
                {"entity_id": [3], "stage_id": [0]},
                evaporation_violation_pos_m3s=[0.5],
            )
        )

        slacks = _encode(frames, pairs=[AlignedPair("Furnas", 3, 103)])[
            "nw_hydro_slacks"
        ]

        assert slacks.schema == _SLACK_SCHEMA
        assert slacks.rows() == [(103, 0, None, None, 0.5, None)]

    def test_an_empty_hydro_frame_gives_an_empty_typed_frame(self) -> None:
        slacks = _encode(pairs=[AlignedPair("Furnas", 3, 103)])["nw_hydro_slacks"]

        assert slacks.is_empty()
        assert slacks.schema == _SLACK_SCHEMA


class TestTiming:
    def _timing(
        self,
        total: Sequence[float | None],
        backward: Sequence[float | None],
        forward: Sequence[float | None],
    ) -> pl.DataFrame:
        return _wide(
            {"iteration": range(1, len(total) + 1)},
            time_total_ms=total,
            time_backward_ms=backward,
            time_forward_ms=forward,
        )

    def test_milliseconds_become_seconds_and_the_split_is_kept(self) -> None:
        frames = BaselineFrames(
            iteration_timing=self._timing([300.0], [200.0], [100.0]),
            training_seconds=0.9,
        )

        slots = _encode(frames)

        assert slots["nw_tim_iterations"].columns == [
            "iteration",
            "total_seconds",
            "backward_seconds",
            "forward_seconds",
        ]
        assert slots["nw_tim_iterations"].rows() == [
            (1, pytest.approx(0.3), pytest.approx(0.2), pytest.approx(0.1))
        ]
        assert slots["nw_tim_iterations"].schema["iteration"] == pl.Int64
        assert slots["nw_tim_stages"] == {"Calculo da Politica": 0.9}

    @pytest.mark.parametrize("null_in", ["backward", "forward"])
    def test_a_null_split_column_drops_the_split_from_the_frame(
        self, null_in: str
    ) -> None:
        backward = [200.0, None if null_in == "backward" else 220.0]
        forward = [100.0, None if null_in == "forward" else 110.0]
        frames = BaselineFrames(
            iteration_timing=self._timing([300.0, 330.0], backward, forward)
        )

        timing = _encode(frames)["nw_tim_iterations"]

        assert timing.columns == ["iteration", "total_seconds"]
        assert timing["total_seconds"].to_list() == pytest.approx([0.3, 0.33])

    def test_a_split_column_the_frame_lacks_drops_the_split(self) -> None:
        frames = BaselineFrames(
            iteration_timing=_wide(
                {"iteration": [1]}, time_total_ms=[300.0], time_forward_ms=[100.0]
            )
        )

        timing = _encode(frames)["nw_tim_iterations"]

        assert timing.columns == ["iteration", "total_seconds"]

    def test_a_null_total_builds_no_row(self) -> None:
        frames = BaselineFrames(
            iteration_timing=self._timing(
                [300.0, None, 330.0], [200.0, 5.0, 220.0], [100.0, 5.0, 110.0]
            )
        )

        timing = _encode(frames)["nw_tim_iterations"]

        assert timing["iteration"].to_list() == [1, 3]
        assert timing["total_seconds"].to_list() == pytest.approx([0.3, 0.33])
        assert not timing["backward_seconds"].has_nulls()

    def test_a_split_that_is_null_only_where_the_total_is_null_is_kept(self) -> None:
        frames = BaselineFrames(
            iteration_timing=self._timing([300.0, None], [200.0, None], [100.0, None])
        )

        timing = _encode(frames)["nw_tim_iterations"]

        assert timing.columns == [
            "iteration",
            "total_seconds",
            "backward_seconds",
            "forward_seconds",
        ]
        assert timing["iteration"].to_list() == [1]

    def test_unknown_training_seconds_builds_no_stage_entry(self) -> None:
        assert _encode(BaselineFrames(training_seconds=0.0))["nw_tim_stages"] == {}

    def test_the_total_row_is_never_written(self) -> None:
        stages = _encode(BaselineFrames(training_seconds=12.5))["nw_tim_stages"]

        assert stages == {"Calculo da Politica": 12.5}


class TestEmptyBaseline:
    def test_every_slot_is_empty_and_typed_without_an_exception(self) -> None:
        baseline = _baseline(
            hydros=[{"id": 1, "name": "Furnas"}], buses=[{"id": 0, "name": "SE"}]
        )

        slots = _encode(baseline=baseline)

        assert slots["nw_convergence"].is_empty()
        assert slots["nw_costs"] == {}
        assert slots["nw_tim_stages"] == {}
        for key, schema in (
            ("nw_sin", _SIN_SCHEMA),
            ("nw_market", _BUS_SCHEMA),
            ("nw_net_load", _BUS_SCHEMA),
            ("nw_hydro_slacks", _SLACK_SCHEMA),
            ("nw_tim_iterations", _TIMING_SCHEMA),
        ):
            assert slots[key].is_empty(), key
            assert slots[key].schema == schema, key

    def test_the_names_still_come_from_the_side(self) -> None:
        baseline = _baseline(
            hydros=[{"id": 1, "name": "Furnas"}], buses=[{"id": 0, "name": "SE"}]
        )

        slots = _encode(baseline=baseline)

        assert slots["nw_hydro_names"] == {1: "Furnas"}
        assert slots["nw_bus_names"] == {0: "SE"}


class TestSlots:
    def _full_frames(self) -> BaselineFrames:
        return BaselineFrames(
            convergence=pl.DataFrame(
                {"iteration": [1], "lower_bound": [1.0], "upper_bound": [2.0]}
            ),
            cost_breakdown={"thermal_cost": 5.0e6, "deficit_cost": 3.0e3},
            stage_costs=_wide(
                {"stage_id": [0]},
                immediate_cost=[2.0e9],
                future_cost=[4.0e9],
                thermal_cost_total=[5.0e8],
            ),
            hydro_means=_wide(
                {"entity_id": [3], "stage_id": [0]},
                stored_energy_final_mwh=[3.0e6],
                incremental_inflow_energy_mw=[50.0],
                **{name: [0.5] for name in _SLACK_COLUMNS},
            ),
            bus_aggregates=_wide(
                {"bus_id": [7], "stage_id": [0]},
                hydro_gen_mw_p50=[400.0],
                net_load_mw_p50=[900.0],
            ),
            iteration_timing=_wide(
                {"iteration": [1]},
                time_total_ms=[300.0],
                time_backward_ms=[200.0],
                time_forward_ms=[100.0],
            ),
            training_seconds=0.9,
        )

    def test_the_keys_are_ten_fields_of_percentile_data(self) -> None:
        fields = {field.name for field in dataclasses.fields(PercentileData)}
        slots = _encode(self._full_frames())

        assert set(BaselineSlots.__annotations__) == set(slots)
        assert len(slots) == 10
        assert set(slots) <= fields

    def test_the_slots_build_a_valid_dataset(self) -> None:
        baseline = _baseline(
            hydros=[{"id": 3, "name": "Furnas"}], buses=[{"id": 7, "name": "SE"}]
        )
        slots = _encode(
            self._full_frames(),
            pairs=[AlignedPair("Furnas", 3, 103)],
            baseline=baseline,
        )

        dataset = build_results_dataset([], PercentileData(**slots), 1e-2)

        dataset.validate()
        assert dataset.render.nw_costs == slots["nw_costs"]
        assert dataset.render.nw_sin.height == 5

    def test_the_convergence_frame_passes_through_unchanged(self) -> None:
        frames = self._full_frames()

        convergence = _encode(frames)["nw_convergence"]

        assert convergence is frames.convergence
        assert convergence.columns == ["iteration", "lower_bound", "upper_bound"]

    def test_the_frames_are_not_mutated(self) -> None:
        frames = self._full_frames()
        before = {
            name: getattr(frames, name).clone()
            for name in (
                "convergence",
                "stage_costs",
                "hydro_means",
                "bus_aggregates",
                "iteration_timing",
            )
        }
        breakdown = dict(frames.cost_breakdown)

        _encode(frames, pairs=[AlignedPair("Furnas", 3, 103)])

        for name, frame in before.items():
            assert getattr(frames, name).equals(frame), name
        assert dict(frames.cost_breakdown) == breakdown

    def test_the_frames_record_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            BaselineFrames().training_seconds = 1.0  # ty: ignore[invalid-assignment] the write is the assertion
