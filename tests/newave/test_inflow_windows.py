"""Tests for the windowed inflow emission module (Novomodelo >= 0.13 shapes)."""

from __future__ import annotations

import datetime
import struct
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from novomodelo_bridge.core.diagnostics import Severity, collect
from novomodelo_bridge.core.errors import FieldParseError
from novomodelo_bridge.core.inflow_windows import (
    format_observation_windows,
    month_window,
    previous_months,
)
from novomodelo_bridge.newave.converters.inflow_windows import (
    INFLOW_HISTORY_WINDOW_SCHEMA,
    _inflow_cascade,
    _posto_count,
    _read_vazoes,
    _vazpast_incremental,
    convert_inflow_history_windows,
    convert_recent_observation_windows,
    natural_monthly_means,
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
                    "codigo_posto": posto,
                    "nome_posto": f"POSTO_{posto}",
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
    @patch("novomodelo_bridge.newave.converters.inflow_windows.Vazoes")
    def test_schema_is_the_windowed_layout(self, mock_vazoes_cls, tmp_path) -> None:
        mock_vazoes_cls.read.return_value = _make_vazoes_mock()
        table = convert_inflow_history_windows(_history_case(tmp_path), _ID_MAP)
        assert table.schema.equals(INFLOW_HISTORY_WINDOW_SCHEMA)

    @patch("novomodelo_bridge.newave.converters.inflow_windows.Vazoes")
    def test_windows_are_calendar_months(self, mock_vazoes_cls, tmp_path) -> None:
        mock_vazoes_cls.read.return_value = _make_vazoes_mock()
        table = convert_inflow_history_windows(_history_case(tmp_path), _ID_MAP)
        df = table.to_pandas()
        for row in df.itertuples():
            start, end = month_window(row.start_date.year, row.start_date.month)
            assert row.start_date == start
            assert row.end_date == end

    @patch("novomodelo_bridge.newave.converters.inflow_windows.Vazoes")
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

    @patch("novomodelo_bridge.newave.converters.inflow_windows.Vazoes")
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


class TestNaturalMonthlyMeans:
    @patch("novomodelo_bridge.newave.converters.inflow_windows.Vazoes")
    def test_each_calendar_month_averages_the_record_before_the_study(
        self, mock_vazoes_cls, tmp_path
    ) -> None:
        # 5 years of record (2020-2024) against a Sep-2024 study start:
        # January-August average five years, September-December four.
        record = _make_vazoes_mock(num_years=5)
        mock_vazoes_cls.read.return_value = record
        means = natural_monthly_means(_history_case(tmp_path))

        source = record.vazoes.set_index("data")
        study_start = datetime.datetime(2024, 9, 1)
        for posto in (1, 2):
            before = source.loc[source.index < study_start, posto]
            expected = [
                before[before.index.month == month].mean() for month in range(1, 13)
            ]
            assert means[posto] == expected


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
        source = tendency.tendencia.set_index(["codigo_posto", "mes"])["valor"]

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
            (int(r["codigo_posto"]), int(r["mes"])): float(r["valor"])
            for _, r in tend.iterrows()
        }
        downstream = [e for e in windows if e["hydro_id"] == 1]
        assert downstream
        for entry in downstream:
            month = int(entry["start_date"][5:7])
            expected = natural[(2, month)] - natural[(1, month)]
            assert abs(entry["value_m3s"] - expected) < 1e-12


# ---------------------------------------------------------------------------
# Artificial gauge (posto 300)
# ---------------------------------------------------------------------------

# Plant 1 (posto 1) flows into plant 3; plants 2 and 3 share the artificial
# gauge and flow into plant 4 (posto 4).
_ARTIFICIAL_CASCADE = ((1, 1, 3), (2, 300, 4), (3, 300, 4), (4, 4, 0))
_ARTIFICIAL_ID_MAP = NewaveIdMap(
    subsystem_ids=[], hydro_codes=[1, 2, 3, 4], thermal_codes=[]
)


def _confhd_rows(rows: tuple[tuple[int, int, int], ...]) -> MagicMock:
    """Confhd mock from ``(code, posto, downstream code)`` rows."""
    mock = MagicMock()
    mock.usinas = pd.DataFrame(
        [
            {
                "posto": posto,
                "codigo_usina": code,
                "nome_usina": f"PLANT_{code}",
                "usina_existente": "EX",
                "codigo_usina_jusante": downstream,
            }
            for code, posto, downstream in rows
        ]
    )
    return mock


def _artificial_record(*, stored_on_artificial: bool) -> MagicMock:
    record = _make_vazoes_mock(postos=[1, 4, 300])
    if not stored_on_artificial:
        record.vazoes[300] = 0.0
    return record


class TestArtificialGauge:
    @patch("novomodelo_bridge.newave.converters.inflow_windows.Vazoes")
    def test_every_plant_sharing_the_artificial_gauge_gets_a_zero_history(
        self, mock_vazoes_cls, tmp_path
    ) -> None:
        mock_vazoes_cls.read.return_value = _artificial_record(
            stored_on_artificial=True
        )
        case = _history_case(tmp_path, confhd=_confhd_rows(_ARTIFICIAL_CASCADE))
        df = convert_inflow_history_windows(case, _ARTIFICIAL_ID_MAP).to_pandas()
        assert sorted(df["hydro_id"].unique()) == [0, 1, 2, 3]
        for hydro_id in (1, 2):
            assert (df[df["hydro_id"] == hydro_id]["value_m3s"] == 0.0).all()

    @patch("novomodelo_bridge.newave.converters.inflow_windows.Vazoes")
    def test_gauged_plants_subtract_only_gauged_direct_upstream(
        self, mock_vazoes_cls, tmp_path
    ) -> None:
        record = _artificial_record(stored_on_artificial=True)
        mock_vazoes_cls.read.return_value = record
        case = _history_case(tmp_path, confhd=_confhd_rows(_ARTIFICIAL_CASCADE))
        df = convert_inflow_history_windows(case, _ARTIFICIAL_ID_MAP).to_pandas()
        source = record.vazoes.set_index("data")
        for hydro_id, posto in ((0, 1), (3, 4)):
            for row in df[df["hydro_id"] == hydro_id].itertuples():
                month = datetime.datetime(row.start_date.year, row.start_date.month, 1)
                assert row.value_m3s == source.loc[month, posto]

    @patch("novomodelo_bridge.newave.converters.inflow_windows.Vazoes")
    def test_artificial_plants_and_double_count_are_reported(
        self, mock_vazoes_cls, tmp_path
    ) -> None:
        mock_vazoes_cls.read.return_value = _artificial_record(
            stored_on_artificial=False
        )
        case = _history_case(tmp_path, confhd=_confhd_rows(_ARTIFICIAL_CASCADE))
        with collect() as diagnostics:
            convert_inflow_history_windows(case, _ARTIFICIAL_ID_MAP)
        by_code = {d.code: d for d in diagnostics}
        assert set(by_code) == {
            "artificial-gauge-plants",
            "artificial-gauge-upstream-double-count",
        }
        listed = by_code["artificial-gauge-plants"].table
        assert listed is not None
        assert [row[0] for row in listed.rows] == [2, 3]
        pairs = by_code["artificial-gauge-upstream-double-count"].table
        assert pairs is not None
        assert [(row[0], row[2]) for row in pairs.rows] == [(4, 3)]

    @patch("novomodelo_bridge.newave.converters.inflow_windows.Vazoes")
    def test_values_stored_on_the_artificial_gauge_are_reported_ignored(
        self, mock_vazoes_cls, tmp_path
    ) -> None:
        mock_vazoes_cls.read.return_value = _artificial_record(
            stored_on_artificial=True
        )
        case = _history_case(tmp_path, confhd=_confhd_rows(_ARTIFICIAL_CASCADE))
        with collect() as diagnostics:
            convert_inflow_history_windows(case, _ARTIFICIAL_ID_MAP)
        assert "artificial-gauge-data-ignored" in {d.code for d in diagnostics}

    def test_artificial_plants_get_zero_tendency_windows(self, tmp_path) -> None:
        vazpast = _make_vazpast_mock(postos=[1, 4])
        case = _tendency_case(
            tmp_path, confhd=_confhd_rows(_ARTIFICIAL_CASCADE), vazpast=vazpast
        )
        windows = convert_recent_observation_windows(case, _ARTIFICIAL_ID_MAP)
        natural = vazpast.tendencia.set_index(["codigo_posto", "mes"])["valor"]
        by_hydro: dict[int, list[dict]] = {}
        for entry in windows:
            by_hydro.setdefault(entry["hydro_id"], []).append(entry)
        assert sorted(by_hydro) == [0, 1, 2, 3]
        for hydro_id in (1, 2):
            assert len(by_hydro[hydro_id]) == 12
            assert all(w["value_m3s"] == 0.0 for w in by_hydro[hydro_id])
        for entry in by_hydro[3]:
            assert entry["value_m3s"] == natural[(4, int(entry["start_date"][5:7]))]


# ---------------------------------------------------------------------------
# Tests: _inflow_cascade — NE/NC cascade bypass
# ---------------------------------------------------------------------------


def _confhd_row(
    code: int,
    posto: int,
    downstream: int,
    status: str = "EX",
) -> dict[str, object]:
    return {
        "codigo_usina": code,
        "nome_usina": f"PLANT_{code}",
        "posto": posto,
        "codigo_usina_jusante": downstream,
        "usina_existente": status,
    }


class TestInflowCascadeExpansionPlant:
    """An ``EE`` plant is in service, so it is a real inflow node. Walking
    through it instead credits its whole natural inflow to the plant below,
    inflating that plant's incremental series."""

    def test_ee_plant_between_two_ex_plants_is_its_own_node(self) -> None:
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 2, "EX"),
                _confhd_row(2, 200, 3, "EE"),
                _confhd_row(3, 310, 0, "EX"),
            ]
        )
        cascade = _inflow_cascade(confhd)
        assert cascade.upstream_of == {2: [1], 3: [2]}
        assert cascade.posto_of == {1: 100, 2: 200, 3: 310}


class TestInflowCascadeNonExistingBypass:
    """The cascade walks through NE/NC plants so it stays connected. Without
    this, the downstream EX plant's incremental inflow fails to subtract the
    upstream EX plant's natural inflow."""

    def test_nc_plant_between_two_ex_plants_keeps_the_link(self) -> None:
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 2, "EX"),
                _confhd_row(2, 200, 3, "NC"),
                _confhd_row(3, 310, 0, "EX"),
            ]
        )
        assert _inflow_cascade(confhd).upstream_of == {3: [1]}

    def test_ne_plant_between_two_ex_plants_keeps_the_link(self) -> None:
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 2, "EX"),
                _confhd_row(2, 200, 3, "NE"),
                _confhd_row(3, 310, 0, "EX"),
            ]
        )
        assert _inflow_cascade(confhd).upstream_of == {3: [1]}

    def test_consecutive_absent_plants_collapse_to_single_link(self) -> None:
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 2, "EX"),
                _confhd_row(2, 200, 3, "NC"),
                _confhd_row(3, 310, 4, "NE"),
                _confhd_row(4, 400, 0, "EX"),
            ]
        )
        cascade = _inflow_cascade(confhd)
        assert cascade.upstream_of == {4: [1]}
        assert set(cascade.posto_of) == {1, 4}

    def test_absent_at_chain_end_yields_no_link(self) -> None:
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 2, "EX"),
                _confhd_row(2, 200, 0, "NC"),
            ]
        )
        assert _inflow_cascade(confhd).upstream_of == {}

    def test_direct_ex_to_ex_link_preserved(self) -> None:
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 2, "EX"),
                _confhd_row(2, 200, 0, "EX"),
            ]
        )
        assert _inflow_cascade(confhd).upstream_of == {2: [1]}


class TestInflowCascadeFillingAdmission:
    """An admitted ``NE``-with-filling plant *receives* inflow, so it is a
    real node: an upstream plant links **to** the filling plant instead of
    walking through it. The JURUENA case (code 309, posto 226, ``NE``) is the
    live exemplar."""

    def test_filling_plant_is_a_node(self) -> None:
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 309, "EX"),
                _confhd_row(309, 226, 0, "NE"),
            ]
        )
        cascade = _inflow_cascade(confhd, filling_codes={309})
        assert cascade.posto_of[309] == 226
        assert cascade.upstream_of == {309: [1]}

    def test_without_filling_codes_the_ne_plant_is_walked_through(self) -> None:
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 309, "EX"),
                _confhd_row(309, 226, 0, "NE"),
            ]
        )
        cascade = _inflow_cascade(confhd)
        assert cascade == _inflow_cascade(confhd, filling_codes=set())
        assert 309 not in cascade.posto_of
        assert cascade.upstream_of == {}

    def test_filling_plant_with_nan_posto_is_skipped(self) -> None:
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 309, "EX"),
                {
                    "codigo_usina": 309,
                    "nome_usina": "JURUENA",
                    "posto": float("nan"),
                    "codigo_usina_jusante": 0,
                    "usina_existente": "NE",
                },
            ]
        )
        cascade = _inflow_cascade(confhd, filling_codes={309})
        assert 309 not in cascade.posto_of
        assert cascade.upstream_of == {}


# ---------------------------------------------------------------------------
# Tests: vazoes.dat width
# ---------------------------------------------------------------------------


def _write_vazoes(tmp_path: Path, num_bytes: int) -> Path:
    """A vazoes.dat of the given byte size; only its size matters."""
    path = tmp_path / "vazoes.dat"
    path.write_bytes(b"\x00" * num_bytes)
    return path


def _count(path: Path) -> int:
    return _posto_count(path, hist_start_year=1931, study_start_year=2026)


class TestPostoCount:
    @pytest.mark.parametrize(
        ("num_values", "expected"),
        [
            # One year at 320; not even a whole number of months at 600.
            (320 * 12, 320),
            (600 * 12, 600),
            # 1931-2026 at 600 is also a whole number of years at 320, but
            # one that would run to 2110.
            (600 * 12 * 96, 600),
        ],
    )
    def test_the_only_fitting_width_is_used_silently(
        self, tmp_path, num_values, expected
    ) -> None:
        with collect() as collected:
            assert _count(_write_vazoes(tmp_path, 4 * num_values)) == expected
        assert collected == []

    def test_ambiguous_size_assumes_320_and_warns(self, tmp_path) -> None:
        # 15 years at 320 postos is also 8 years at 600, both ending by 2026.
        with collect() as collected:
            count = _count(_write_vazoes(tmp_path, 4 * 320 * 12 * 15))

        assert count == 320
        assert len(collected) == 1
        diag = collected[0]
        assert diag.code == "vazoes-posto-count-ambiguous"
        assert diag.severity is Severity.WARNING
        assert diag.category == "Historical inflows"
        assert "320" in diag.summary
        assert "600" in diag.summary

    @pytest.mark.parametrize(
        ("num_bytes", "message"),
        [
            (4 * 320 * 12 + 2, "4-byte values"),
            (4 * 1000, "320 or 600"),
            # 13 months at 320 postos: the historical record ends in December.
            (4 * 320 * 13, "320 or 600"),
            # 100 years at 320 postos from 1931 runs to 2030.
            (4 * 320 * 12 * 100, "to 2030 at 320 postos"),
        ],
    )
    def test_size_that_fits_no_width_is_rejected(
        self, tmp_path, num_bytes, message
    ) -> None:
        with pytest.raises(FieldParseError, match=message):
            _count(_write_vazoes(tmp_path, num_bytes))

    def test_empty_file_falls_back_to_the_default(self, tmp_path) -> None:
        # Emptiness is the callers' finding, not the width derivation's.
        with collect() as collected:
            assert _count(_write_vazoes(tmp_path, 0)) == 320
        assert collected == []


def test_six_hundred_posto_matrix_is_laid_out_at_full_width(tmp_path) -> None:
    """Through the real reader: a width it ignores, rejects or lays out
    transposed would mis-assign every posto's series. The 320-posto default is
    covered end to end by the mini deck.
    """
    path = tmp_path / "vazoes.dat"
    # Sequential values, unlike ``_write_vazoes``' zeros: the row-major
    # assertions below are what pin the width.
    path.write_bytes(b"".join(struct.pack("<i", v) for v in range(600 * 12)))

    df = _read_vazoes(
        path, _make_dger_mock(ano_inicio_estudo=2026, ano_inicial_historico=1931)
    ).vazoes

    assert df.shape == (12, 600)
    assert df.iloc[0, 0] == 0
    assert df.iloc[0, -1] == 599
    assert df.iloc[1, 0] == 600


def test_unreadable_inflow_tendency_gives_no_recent_inflows(tmp_path) -> None:
    case = make_case(make_nw_files(tmp_path, vazpast=tmp_path / "vazpast.dat"))

    with (
        patch("novomodelo_bridge.newave.case.Vazpast.read", side_effect=ValueError),
        collect() as found,
    ):
        assert _vazpast_incremental(case, _ID_MAP) == {}

    assert [d.code for d in found] == ["vazpast-unreadable"]
