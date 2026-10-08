"""Tests for the windowed inflow emission module (Novomodelo >= 0.13 shapes)."""

from __future__ import annotations

import datetime
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from novomodelo_bridge.core.inflow_windows import (
    format_observation_windows,
    month_window,
    previous_months,
)
from novomodelo_bridge.newave.converters.inflow_windows import (
    INFLOW_HISTORY_WINDOW_SCHEMA,
    convert_inflow_history_windows,
    convert_recent_observation_windows,
)
from novomodelo_bridge.newave.id_map import NewaveIdMap
from tests.conftest import make_case, make_nw_files

# ---------------------------------------------------------------------------
# Local mock builders (mirroring test_temporal_stochastic_conversion patterns)
# ---------------------------------------------------------------------------


def _make_vazoes_mock(
    num_years: int = 3,
    postos: list[int] | None = None,
    start_year: int = 2020,
) -> MagicMock:
    """Synthetic historical-record mock with ``num_years`` of monthly data."""
    if postos is None:
        postos = [1, 2]
    rng = np.random.default_rng(7)
    rows = []
    for year in range(start_year, start_year + num_years):
        for month in range(1, 13):
            row: dict = {"data": datetime.datetime(year, month, 1)}
            for posto in postos:
                row[posto] = float(rng.uniform(50.0, 500.0))
            rows.append(row)
    mock = MagicMock()
    mock.vazoes = pd.DataFrame(rows)
    return mock


def _make_vazpast_mock(postos: list[int], months: list[int] | None = None) -> MagicMock:
    """Synthetic tendency mock: one row per (posto, calendar month)."""
    rng = np.random.default_rng(21)
    rows = []
    for posto in postos:
        for mes in months or list(range(1, 13)):
            rows.append(
                {
                    "codigo_usina": posto,
                    "nome_usina": f"PLANT_{posto}",
                    "mes": mes,
                    "valor": float(rng.uniform(50.0, 500.0)),
                }
            )
    mock = MagicMock()
    mock.tendencia = pd.DataFrame(rows)
    return mock


def _make_confhd_mock(
    posto_to_code: dict[int, int],
    jusante: dict[int, int] | None = None,
) -> MagicMock:
    """Confhd mock; ``jusante`` maps hydro code -> downstream hydro code."""
    rows = [
        {
            "posto": posto,
            "codigo_usina": code,
            "nome_usina": f"PLANT_{code}",
            "usina_existente": "EX",
            "codigo_usina_jusante": (jusante or {}).get(code, 0),
        }
        for posto, code in posto_to_code.items()
    ]
    mock = MagicMock()
    mock.usinas = pd.DataFrame(rows)
    return mock


def _make_dger_mock(
    mes_inicio_estudo: int = 9,
    ano_inicio_estudo: int = 2024,
    ano_inicial_historico: int = 2020,
) -> MagicMock:
    mock = MagicMock()
    mock.mes_inicio_estudo = mes_inicio_estudo
    mock.ano_inicio_estudo = ano_inicio_estudo
    mock.ano_inicial_historico = ano_inicial_historico
    mock.num_anos_estudo = 5
    mock.num_anos_pos_estudo = 0
    return mock


def _history_case(tmp_path: Path, **kwargs):
    (tmp_path / "vazoes.dat").touch()
    return make_case(
        tmp_path,
        confhd=kwargs.get("confhd", _make_confhd_mock({1: 1, 2: 2})),
        dger=kwargs.get("dger", _make_dger_mock()),
    )


def _tendency_case(tmp_path: Path, **kwargs):
    vazpast_path = tmp_path / "vazpast.dat"
    vazpast_path.touch()
    return make_case(
        make_nw_files(tmp_path, vazpast=vazpast_path),
        confhd=kwargs.get("confhd", _make_confhd_mock({1: 1, 2: 2})),
        dger=kwargs.get("dger", _make_dger_mock()),
        vazpast=kwargs.get("vazpast", _make_vazpast_mock(postos=[1, 2])),
    )


_ID_MAP = NewaveIdMap(subsystem_ids=[], hydro_codes=[1, 2], thermal_codes=[])


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------


class TestMonthWindow:
    def test_regular_month(self) -> None:
        assert month_window(2024, 9) == (date(2024, 9, 1), date(2024, 10, 1))

    def test_december_wraps_year(self) -> None:
        assert month_window(2024, 12) == (date(2024, 12, 1), date(2025, 1, 1))


class TestPreviousMonths:
    def test_twelve_back_from_september(self) -> None:
        months = previous_months(2024, 9, 12)
        assert len(months) == 12
        assert months[0] == (2023, 9)
        assert months[-1] == (2024, 8)

    def test_january_wraps_year(self) -> None:
        assert previous_months(2024, 1, 2) == [(2023, 11), (2023, 12)]


class TestFormatObservationWindows:
    def test_iso_shape_and_order_preserved(self) -> None:
        rows = [
            (3, date(2024, 7, 1), date(2024, 8, 1), 123.0),
            (1, date(2024, 8, 1), date(2024, 9, 1), 45.5),
        ]
        out = format_observation_windows(rows)
        assert out == [
            {
                "hydro_id": 3,
                "start_date": "2024-07-01",
                "end_date": "2024-08-01",
                "value_m3s": 123.0,
            },
            {
                "hydro_id": 1,
                "start_date": "2024-08-01",
                "end_date": "2024-09-01",
                "value_m3s": 45.5,
            },
        ]


# ---------------------------------------------------------------------------
# Windowed history
# ---------------------------------------------------------------------------


class TestConvertInflowHistoryWindows:
    @patch("novomodelo_bridge.newave.converters.stochastic.Vazoes")
    def test_schema_is_the_windowed_layout(self, mock_vazoes_cls, tmp_path) -> None:
        mock_vazoes_cls.read.return_value = _make_vazoes_mock()
        table = convert_inflow_history_windows(_history_case(tmp_path), _ID_MAP)
        assert table.schema.equals(INFLOW_HISTORY_WINDOW_SCHEMA)

    @patch("novomodelo_bridge.newave.converters.stochastic.Vazoes")
    def test_windows_are_calendar_months(self, mock_vazoes_cls, tmp_path) -> None:
        mock_vazoes_cls.read.return_value = _make_vazoes_mock()
        table = convert_inflow_history_windows(_history_case(tmp_path), _ID_MAP)
        df = table.to_pandas()
        for row in df.itertuples():
            start, end = month_window(row.start_date.year, row.start_date.month)
            assert row.start_date == start
            assert row.end_date == end

    @patch("novomodelo_bridge.newave.converters.stochastic.Vazoes")
    def test_contiguous_per_hydro_and_truncated_before_study(
        self, mock_vazoes_cls, tmp_path
    ) -> None:
        # 5 years of record (2020-2024) against a Sep-2024 study start: the
        # cutoff truncates the record to 56 months, before the record's end.
        mock_vazoes_cls.read.return_value = _make_vazoes_mock(num_years=5)
        table = convert_inflow_history_windows(_history_case(tmp_path), _ID_MAP)
        df = table.to_pandas()
        for _, group in df.groupby("hydro_id"):
            group = group.sort_values("start_date").reset_index(drop=True)
            assert len(group) == (2024 - 2020) * 12 + 8
            assert group.loc[0, "start_date"] == date(2020, 1, 1)
            assert group.iloc[-1]["end_date"] == date(2024, 9, 1)
            starts = group["start_date"].tolist()
            ends = group["end_date"].tolist()
            assert starts[1:] == ends[:-1]

    @patch("novomodelo_bridge.newave.converters.stochastic.Vazoes")
    def test_each_window_carries_its_month_of_record(
        self, mock_vazoes_cls, tmp_path
    ) -> None:
        """A window's value is the record's value for the month it spans.

        The two plants here have no cascade link, so incremental equals
        natural and the emitted value is the gauge reading itself.
        """
        record = _make_vazoes_mock()
        mock_vazoes_cls.read.return_value = record
        windowed = convert_inflow_history_windows(
            _history_case(tmp_path), _ID_MAP
        ).to_pandas()
        source = record.vazoes.set_index("data")

        for hydro_id, posto in ((0, 1), (1, 2)):
            rows = windowed[windowed["hydro_id"] == hydro_id]
            for row in rows.itertuples():
                expected = source.loc[
                    datetime.datetime(row.start_date.year, row.start_date.month, 1),
                    posto,
                ]
                assert row.value_m3s == expected


# ---------------------------------------------------------------------------
# Conditioning windows (tendency)
# ---------------------------------------------------------------------------


class TestConvertRecentObservationWindows:
    def test_empty_when_tendency_absent(self, tmp_path: Path) -> None:
        assert convert_recent_observation_windows(make_case(tmp_path), _ID_MAP) == []

    def test_twelve_contiguous_windows_ending_at_study_start(
        self, tmp_path: Path
    ) -> None:
        case = _tendency_case(tmp_path)
        result = convert_recent_observation_windows(case, _ID_MAP)
        by_hydro: dict[int, list[dict]] = {}
        for entry in result:
            by_hydro.setdefault(entry["hydro_id"], []).append(entry)
        assert set(by_hydro) == {0, 1}
        for windows in by_hydro.values():
            assert len(windows) == 12
            assert windows[0]["start_date"] == "2023-09-01"
            assert windows[-1]["end_date"] == "2024-09-01"
            for prev, curr in zip(windows, windows[1:]):
                assert curr["start_date"] == prev["end_date"]

    def test_each_window_carries_its_calendar_month_of_tendency(
        self, tmp_path: Path
    ) -> None:
        """Values follow the calendar month a window spans, oldest first.

        The tendency is keyed by calendar month, so a Sep-2024 study start
        maps the first window (Sep 2023) to month 9 and the last (Aug 2024)
        to month 8 — the ordering a positional lag list encodes backwards.
        """
        tendency = _make_vazpast_mock(postos=[1, 2])
        case = _tendency_case(tmp_path, vazpast=tendency)
        windows = convert_recent_observation_windows(case, _ID_MAP)
        source = tendency.tendencia.set_index(["codigo_usina", "mes"])["valor"]

        for hydro_id, posto in ((0, 1), (1, 2)):
            entries = [w for w in windows if w["hydro_id"] == hydro_id]
            assert [w["start_date"][:7] for w in entries[:2]] == [
                "2023-09",
                "2023-10",
            ]
            for entry in entries:
                month = int(entry["start_date"][5:7])
                assert entry["value_m3s"] == source[(posto, month)]

    def test_missing_month_is_omitted_not_zero_filled(self, tmp_path: Path) -> None:
        # Tendency carries only months 1..11: the missing month (12) must be
        # absent from the windows rather than fabricated as a zero inflow.
        case = _tendency_case(
            tmp_path,
            vazpast=_make_vazpast_mock(postos=[1, 2], months=list(range(1, 12))),
        )
        windows = convert_recent_observation_windows(case, _ID_MAP)
        by_hydro: dict[int, list[dict]] = {}
        for entry in windows:
            by_hydro.setdefault(entry["hydro_id"], []).append(entry)
        for hydro_windows in by_hydro.values():
            assert len(hydro_windows) == 11
            assert all(w["start_date"] != "2023-12-01" for w in hydro_windows)
            assert all(w["value_m3s"] != 0.0 for w in hydro_windows)

    def test_cascade_subtracts_upstream_tendency(self, tmp_path: Path) -> None:
        # Plant 1 (posto 1) flows into plant 2 (posto 2): plant 2's windows
        # carry natural(2) − natural(1).
        vazpast = _make_vazpast_mock(postos=[1, 2])
        tend = vazpast.tendencia
        case = _tendency_case(
            tmp_path,
            confhd=_make_confhd_mock({1: 1, 2: 2}, jusante={1: 2}),
            vazpast=vazpast,
        )
        windows = convert_recent_observation_windows(case, _ID_MAP)
        natural = {
            (int(r["codigo_usina"]), int(r["mes"])): float(r["valor"])
            for _, r in tend.iterrows()
        }
        downstream = [e for e in windows if e["hydro_id"] == 1]
        assert downstream
        for entry in downstream:
            month = int(entry["start_date"][5:7])
            expected = natural[(2, month)] - natural[(1, month)]
            assert abs(entry["value_m3s"] - expected) < 1e-12
