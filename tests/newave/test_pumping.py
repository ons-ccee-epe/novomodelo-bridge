"""Tests for the pumping station converter."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, PropertyMock

import pandas as pd
import pytest

from novomodelo_bridge.core.errors import SourceFileError
from novomodelo_bridge.newave.converters.pumping import (
    PumpingInputs,
    convert_pumping_bounds,
    convert_pumping_stations,
    pumping_station_id_map,
    read_pumping_inputs,
)


def _mock_case(
    *,
    switch_on: bool = True,
    stations_df: pd.DataFrame | None = None,
    submercado_df: pd.DataFrame | None = None,
    confhd_codes: set[int] | None = None,
    limits_df: pd.DataFrame | None = None,
) -> MagicMock:
    """Build a minimal mock NewaveCase for pumping tests."""
    case = MagicMock()

    # switches.bombeamento.on
    case.switches.bombeamento.on = switch_on

    # estacao_bombeamento reader
    if stations_df is not None:
        reader = MagicMock()
        reader.estacao_bombeamento.return_value = stations_df
        reader.estacao_bombeamento_limites_periodo_patamar.return_value = limits_df
        case.estacao_bombeamento = reader
    else:
        case.estacao_bombeamento = None

    # estacao_bombeamento_submercado reader
    if submercado_df is not None:
        submercado_reader = MagicMock()
        submercado_reader.estacao_bombeamento_submercado.return_value = submercado_df
        case.estacao_bombeamento_submercado = submercado_reader
    else:
        case.estacao_bombeamento_submercado = None

    case.hydro_plants = pd.DataFrame({"codigo_usina": sorted(confhd_codes or ())})

    return case


class TestReadPumpingInputs:
    """Tests for read_pumping_inputs function."""

    def test_none_when_switch_off(self) -> None:
        case = _mock_case(switch_on=False)
        assert read_pumping_inputs(case) is None

    def test_none_when_file_missing(self) -> None:
        case = _mock_case(switch_on=True, stations_df=None)
        assert read_pumping_inputs(case) is None

    def test_none_when_stations_empty(self) -> None:
        case = _mock_case(switch_on=True, stations_df=pd.DataFrame())
        assert read_pumping_inputs(case) is None

    def test_raises_on_missing_submercado_reader(self) -> None:
        stations_df = pd.DataFrame(
            [
                {
                    "codigo_estacao": 1,
                    "nome_estacao": "STATION1",
                    "codigo_usina_origem": 10,
                    "codigo_usina_destino": 20,
                    "consumo_estacao": 1.5,
                    "bombeamento_maximo": 100.0,
                }
            ]
        )
        case = _mock_case(
            switch_on=True,
            stations_df=stations_df,
            submercado_df=None,
            confhd_codes={10, 20},
        )
        with pytest.raises(SourceFileError) as exc_info:
            read_pumping_inputs(case)
        assert "submercado file missing" in str(exc_info.value)

    def test_raises_on_missing_submercado_row(self) -> None:
        stations_df = pd.DataFrame(
            [
                {
                    "codigo_estacao": 1,
                    "nome_estacao": "STATION1",
                    "codigo_usina_origem": 10,
                    "codigo_usina_destino": 20,
                    "consumo_estacao": 1.5,
                    "bombeamento_maximo": 100.0,
                }
            ]
        )
        submercado_df = pd.DataFrame(
            [{"codigo_estacao": 99, "codigo_submercado": 1}]  # wrong station
        )
        case = _mock_case(
            switch_on=True,
            stations_df=stations_df,
            submercado_df=submercado_df,
            confhd_codes={10, 20},
        )
        with pytest.raises(SourceFileError) as exc_info:
            read_pumping_inputs(case)
        assert "no submercado assignment" in str(exc_info.value)

    def test_raises_on_duplicate_submercado(self) -> None:
        stations_df = pd.DataFrame(
            [
                {
                    "codigo_estacao": 1,
                    "nome_estacao": "STATION1",
                    "codigo_usina_origem": 10,
                    "codigo_usina_destino": 20,
                    "consumo_estacao": 1.5,
                    "bombeamento_maximo": 100.0,
                }
            ]
        )
        submercado_df = pd.DataFrame(
            [
                {"codigo_estacao": 1, "codigo_submercado": 1},
                {"codigo_estacao": 1, "codigo_submercado": 2},  # duplicate
            ]
        )
        case = _mock_case(
            switch_on=True,
            stations_df=stations_df,
            submercado_df=submercado_df,
            confhd_codes={10, 20},
        )
        with pytest.raises(SourceFileError) as exc_info:
            read_pumping_inputs(case)
        assert "multiple submercado assignments" in str(exc_info.value)

    def test_raises_on_invalid_hydro_code(self) -> None:
        stations_df = pd.DataFrame(
            [
                {
                    "codigo_estacao": 1,
                    "nome_estacao": "STATION1",
                    "codigo_usina_origem": 10,
                    "codigo_usina_destino": 999,  # not in confhd
                    "consumo_estacao": 1.5,
                    "bombeamento_maximo": 100.0,
                }
            ]
        )
        submercado_df = pd.DataFrame([{"codigo_estacao": 1, "codigo_submercado": 1}])
        case = _mock_case(
            switch_on=True,
            stations_df=stations_df,
            submercado_df=submercado_df,
            confhd_codes={10},  # missing 999
        )
        with pytest.raises(SourceFileError) as exc_info:
            read_pumping_inputs(case)
        assert "unknown hydro 999" in str(exc_info.value)

    def test_joins_submercado_correctly(self) -> None:
        stations_df = pd.DataFrame(
            [
                {
                    "codigo_estacao": 1,
                    "nome_estacao": "STATION1",
                    "codigo_usina_origem": 10,
                    "codigo_usina_destino": 20,
                    "consumo_estacao": 1.5,
                    "bombeamento_maximo": 100.0,
                },
                {
                    "codigo_estacao": 2,
                    "nome_estacao": "STATION2",
                    "codigo_usina_origem": 30,
                    "codigo_usina_destino": 40,
                    "consumo_estacao": 2.0,
                    "bombeamento_maximo": 200.0,
                },
            ]
        )
        submercado_df = pd.DataFrame(
            [
                {"codigo_estacao": 1, "codigo_submercado": 1},
                {"codigo_estacao": 2, "codigo_submercado": 2},
            ]
        )
        case = _mock_case(
            switch_on=True,
            stations_df=stations_df,
            submercado_df=submercado_df,
            confhd_codes={10, 20, 30, 40},
        )
        inputs = read_pumping_inputs(case)
        assert inputs is not None
        assert len(inputs.stations) == 2
        assert "codigo_submercado" in inputs.stations.columns
        row1 = inputs.stations[inputs.stations["codigo_estacao"] == 1].iloc[0]
        assert row1["codigo_submercado"] == 1


class TestPumpingStationIdMap:
    """Tests for pumping_station_id_map function."""

    def test_returns_sorted_0_based_mapping(self) -> None:
        stations_df = pd.DataFrame(
            [
                {"codigo_estacao": 30},
                {"codigo_estacao": 10},
                {"codigo_estacao": 20},
            ]
        )
        inputs = PumpingInputs(stations=stations_df, limits=pd.DataFrame())
        id_map = pumping_station_id_map(inputs)
        assert id_map == {10: 0, 20: 1, 30: 2}


class TestConvertPumpingStations:
    """Tests for convert_pumping_stations function."""

    def test_empty_when_inputs_none(self) -> None:
        case = MagicMock()
        id_map = MagicMock()
        result = convert_pumping_stations(case, id_map, None, {})
        assert result["pumping_stations"] == []
        assert "$schema" in result

    def test_builds_station_dict_correctly(self) -> None:
        case = MagicMock()
        case.horizon.start_year = 2025
        case.horizon.start_month = 1
        type(case).pumping_related_codes = PropertyMock(
            return_value=frozenset({10, 20})
        )

        id_map = MagicMock()
        id_map.bus_id.return_value = 0
        id_map.hydro_id.side_effect = lambda x: x - 10  # 10->0, 20->10

        stations_df = pd.DataFrame(
            [
                {
                    "codigo_estacao": 1,
                    "nome_estacao": "  STATION1  ",
                    "codigo_usina_origem": 10,
                    "codigo_usina_destino": 20,
                    "codigo_submercado": 1,
                    "consumo_estacao": 1.5,
                    "bombeamento_maximo": 100.0,
                }
            ]
        )
        inputs = PumpingInputs(stations=stations_df, limits=pd.DataFrame())
        station_ids = {1: 0}

        result = convert_pumping_stations(case, id_map, inputs, station_ids)
        assert len(result["pumping_stations"]) == 1

        station = result["pumping_stations"][0]
        assert station["id"] == 0
        assert station["name"] == "STATION1"
        assert station["operational_start_date"] == "2025-01-01"
        assert station["bus_id"] == 0
        assert station["source_hydro_id"] == 0
        assert station["destination_hydro_id"] == 10
        assert station["consumption_mw_per_m3s"] == 1.5
        assert station["flow"]["min_m3s"] == 0.0
        assert station["flow"]["max_m3s"] == 100.0


class TestConvertPumpingBounds:
    """Tests for convert_pumping_bounds function."""

    def _mock_case_with_horizon(
        self, start: date, total_stages: int, n_blocks: int = 3
    ) -> tuple[MagicMock, list[date]]:
        """Create a mock case with a real-ish horizon for bounds tests."""
        case = MagicMock()
        case.horizon.start_year = start.year
        case.horizon.start_month = start.month
        case.patamar.numero_patamares = n_blocks

        # Build stage dates starting from start month
        stage_dates = []
        y, m = start.year, start.month
        for _ in range(total_stages):
            stage_dates.append(date(y, m, 1))
            m += 1
            if m > 12:
                m = 1
                y += 1

        return case, stage_dates

    def test_none_when_inputs_none(self) -> None:
        case = MagicMock()
        result = convert_pumping_bounds(case, None, {})
        assert result is None

    def test_none_when_limits_empty(self) -> None:
        case = MagicMock()
        inputs = PumpingInputs(stations=pd.DataFrame(), limits=pd.DataFrame())
        result = convert_pumping_bounds(case, inputs, {})
        assert result is None

    def test_handles_closed_period(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Stage dates: Jan, Feb, Mar 2025
        stage_dates = [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1)]
        monkeypatch.setattr(
            "novomodelo_bridge.newave.converters.pumping.stage_dates_for",
            lambda _: stage_dates,
        )

        case = MagicMock()
        case.patamar.numero_patamares = 3

        limits_df = pd.DataFrame(
            [
                {
                    "codigo_estacao": 1,
                    "patamar": 1,
                    "data_inicio": date(2025, 1, 1),
                    "data_fim": date(2025, 2, 1),  # ends in Feb
                    "limite_inferior": 10.0,
                    "limite_superior": 100.0,
                }
            ]
        )
        inputs = PumpingInputs(stations=pd.DataFrame(), limits=limits_df)
        station_ids = {1: 0}

        result = convert_pumping_bounds(case, inputs, station_ids)
        assert result is not None
        # Should only cover Jan and Feb (stages 0 and 1)
        assert result.num_rows == 2
        assert result.column("stage_id").to_pylist() == [0, 1]

    def test_handles_open_ended_period(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stage_dates = [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1)]
        monkeypatch.setattr(
            "novomodelo_bridge.newave.converters.pumping.stage_dates_for",
            lambda _: stage_dates,
        )

        case = MagicMock()
        case.patamar.numero_patamares = 3

        limits_df = pd.DataFrame(
            [
                {
                    "codigo_estacao": 1,
                    "patamar": 1,
                    "data_inicio": date(2025, 1, 1),
                    "data_fim": None,  # open-ended
                    "limite_inferior": 10.0,
                    "limite_superior": 100.0,
                }
            ]
        )
        inputs = PumpingInputs(stations=pd.DataFrame(), limits=limits_df)
        station_ids = {1: 0}

        result = convert_pumping_bounds(case, inputs, station_ids)
        assert result is not None
        # Should cover all stages (0, 1, 2)
        assert result.num_rows == 3

    def test_correct_block_id_mapping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stage_dates = [date(2025, 1, 1)]
        monkeypatch.setattr(
            "novomodelo_bridge.newave.converters.pumping.stage_dates_for",
            lambda _: stage_dates,
        )

        case = MagicMock()
        case.patamar.numero_patamares = 3

        # patamar 2 -> block_id 1
        limits_df = pd.DataFrame(
            [
                {
                    "codigo_estacao": 1,
                    "patamar": 2,
                    "data_inicio": date(2025, 1, 1),
                    "data_fim": None,
                    "limite_inferior": 10.0,
                    "limite_superior": 100.0,
                }
            ]
        )
        inputs = PumpingInputs(stations=pd.DataFrame(), limits=limits_df)
        station_ids = {1: 0}

        result = convert_pumping_bounds(case, inputs, station_ids)
        assert result is not None
        assert result.column("block_id").to_pylist() == [1]

    def test_sorted_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stage_dates = [date(2025, 1, 1), date(2025, 2, 1)]
        monkeypatch.setattr(
            "novomodelo_bridge.newave.converters.pumping.stage_dates_for",
            lambda _: stage_dates,
        )

        case = MagicMock()
        case.patamar.numero_patamares = 3

        # Multiple stations, multiple stages, multiple blocks
        limits_df = pd.DataFrame(
            [
                {
                    "codigo_estacao": 2,
                    "patamar": 2,
                    "data_inicio": date(2025, 1, 1),
                    "data_fim": None,
                    "limite_inferior": 10.0,
                    "limite_superior": 100.0,
                },
                {
                    "codigo_estacao": 1,
                    "patamar": 1,
                    "data_inicio": date(2025, 1, 1),
                    "data_fim": None,
                    "limite_inferior": 5.0,
                    "limite_superior": 50.0,
                },
            ]
        )
        inputs = PumpingInputs(stations=pd.DataFrame(), limits=limits_df)
        station_ids = {1: 0, 2: 1}

        result = convert_pumping_bounds(case, inputs, station_ids)
        assert result is not None
        # Should be sorted by (station_id, stage_id, block_id)
        stations = result.column("pumping_station_id").to_pylist()

        # station 0 rows come before station 1 rows
        assert stations[:2] == [0, 0]  # station 1 -> id 0
        assert stations[2:] == [1, 1]  # station 2 -> id 1

    def test_raises_on_invalid_patamar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stage_dates = [date(2025, 1, 1)]
        monkeypatch.setattr(
            "novomodelo_bridge.newave.converters.pumping.stage_dates_for",
            lambda _: stage_dates,
        )

        case = MagicMock()
        case.patamar.numero_patamares = 3

        limits_df = pd.DataFrame(
            [
                {
                    "codigo_estacao": 1,
                    "patamar": 5,  # invalid: n_blocks is 3
                    "data_inicio": date(2025, 1, 1),
                    "data_fim": None,
                    "limite_inferior": 10.0,
                    "limite_superior": 100.0,
                }
            ]
        )
        inputs = PumpingInputs(stations=pd.DataFrame(), limits=limits_df)
        station_ids = {1: 0}

        with pytest.raises(SourceFileError) as exc_info:
            convert_pumping_bounds(case, inputs, station_ids)
        assert "invalid patamar 5" in str(exc_info.value)
