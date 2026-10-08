"""Tests for the DECOMP PQ → NCS converters."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.ncs import (
    convert_ncs_factors,
    convert_ncs_stats,
    convert_non_controllable_sources,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import build_operative_calendar
from tests.conftest import make_decomp_case

_ID_MAP = DecompIdMap(
    bus_codes=(1, 2, 3, 4, 11),
    bus_names=("SE", "S", "NE", "N", "FC"),
)


def _calendar():
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


class _StubDadger:
    def __init__(self, pq: pd.DataFrame | None) -> None:
        self._pq = pq

    def pq(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._pq


def _case(dadger: _StubDadger) -> DecompCase:
    return make_decomp_case(Path("unused"), dadger=dadger, calendar=_calendar())


def _pq_frame() -> pd.DataFrame:
    rows = [
        # Series declaring stages 1 and 3 (stage 2 inherits stage 1).
        {
            "codigo_submercado": 1,
            "estagio": 1,
            "nome": "SECO_PCH",
            "geracao_1": 1475.0,
            "geracao_2": 1424.0,
            "geracao_3": 1392.0,
        },
        {
            "codigo_submercado": 1,
            "estagio": 3,
            "nome": "SECO_PCH",
            "geracao_1": 1319.0,
            "geracao_2": 1270.0,
            "geracao_3": 1233.0,
        },
        # Series declaring only stage 1 (everything inherits).
        {
            "codigo_submercado": 2,
            "estagio": 1,
            "nome": "S_PCT",
            "geracao_1": 100.0,
            "geracao_2": 200.0,
            "geracao_3": 300.0,
        },
        # Zero-generation series: stats mean 0, no factors entries.
        {
            "codigo_submercado": 4,
            "estagio": 1,
            "nome": "N_EOL",
            "geracao_1": 0.0,
            "geracao_2": 0.0,
            "geracao_3": 0.0,
        },
    ]
    return pd.DataFrame(rows)


class TestRegistry:
    def test_entries_sorted_and_must_run(self) -> None:
        doc = convert_non_controllable_sources(_case(_StubDadger(_pq_frame())), _ID_MAP)
        entries = doc["non_controllable_sources"]
        assert [e["id"] for e in entries] == [0, 1, 2]
        assert [e["name"] for e in entries] == ["SECO_PCH_1", "S_PCT_2", "N_EOL_4"]
        assert [e["bus_id"] for e in entries] == [0, 1, 3]
        assert all(e["allow_curtailment"] is False for e in entries)
        assert entries[0]["max_generation_mw"] == 1475.0
        assert entries[0]["operational_start_date"] == "2026-07-18"

    def test_empty_pq_gives_empty_registry(self) -> None:
        doc = convert_non_controllable_sources(_case(_StubDadger(None)), _ID_MAP)
        assert doc["non_controllable_sources"] == []

    def test_missing_stage_one_raises(self) -> None:
        pq = _pq_frame()
        pq.loc[pq["nome"] == "S_PCT", "estagio"] = 2
        with pytest.raises(ValueError, match="stage 1"):
            convert_non_controllable_sources(_case(_StubDadger(pq)), _ID_MAP)

    def test_stage_outside_calendar_raises(self) -> None:
        pq = _pq_frame()
        pq.loc[0, "estagio"] = 9
        with pytest.raises(ValueError, match="outside the calendar"):
            convert_non_controllable_sources(_case(_StubDadger(pq)), _ID_MAP)


class TestStats:
    def test_availability_fraction_with_inheritance(self) -> None:
        stats = convert_ncs_stats(_case(_StubDadger(_pq_frame())), _ID_MAP).to_pandas()
        assert len(stats) == 3 * 3
        assert set(stats["std"]) == {0.0}

        pch = stats[stats["ncs_id"] == 0].set_index("stage_id")["mean"]
        stage0_mean = (1475.0 * 15 + 1424.0 * 64 + 1392.0 * 89) / 168.0
        stage2_mean = (1319.0 * 63 + 1270.0 * 280 + 1233.0 * 401) / 744.0
        assert pch[0] == pytest.approx(stage0_mean / 1475.0)
        assert pch[1] == pytest.approx(stage0_mean / 1475.0)  # inherited
        assert pch[2] == pytest.approx(stage2_mean / 1475.0)

        assert set(stats[stats["ncs_id"] == 2]["mean"]) == {0.0}


class TestFactors:
    def test_invariant_and_zero_mean_omission(self) -> None:
        calendar = _calendar()
        doc = convert_ncs_factors(_case(_StubDadger(_pq_frame())), _ID_MAP)
        entries = doc["non_controllable_factors"]
        assert {e["ncs_id"] for e in entries} == {0, 1}
        for entry in entries:
            stage = calendar[entry["stage_id"]]
            weighted = sum(
                bf["factor"] * stage.block_hours[bf["block_id"]]
                for bf in entry["block_factors"]
            )
            assert weighted == pytest.approx(stage.total_hours, rel=1e-12)


class _StubRenovaveis:
    def __init__(
        self, cad: pd.DataFrame, subm: pd.DataFrame, ger: pd.DataFrame
    ) -> None:
        self._cad, self._subm, self._ger = cad, subm, ger

    def pee_cad(self, df: bool = False) -> pd.DataFrame:  # noqa: ARG002
        return self._cad

    def pee_subm(self, df: bool = False) -> pd.DataFrame:  # noqa: ARG002
        return self._subm

    def pee_ger_per_pat_cen(self, df: bool = False) -> pd.DataFrame:  # noqa: ARG002
        return self._ger


def test_pee_series_deterministic_typo_uses_modal_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """DECOMP renewables are deterministic: a lone per-scenario outlier (deck
    typo) is collapsed to the majority (modal) value and warned, not crashed
    on, and never resolved to the outlier."""
    import logging

    from novomodelo_bridge.decomp.converters.ncs import _pee_series

    cad = pd.DataFrame([{"codigo_pee": 9, "nome_pee": "PARK9"}])
    subm = pd.DataFrame([{"codigo_pee": 9, "codigo_submercado": 1}])
    # stage 1, three patamares, three scenarios; patamar 1 carries a single-
    # scenario typo (77.9 among 77.07s), patamares 2/3 are clean.
    ger_rows = []
    for pat, clean in ((1, 77.07), (2, 50.0), (3, 60.0)):
        for cen in (1, 2, 3):
            g = 77.9 if (pat == 1 and cen == 3) else clean
            ger_rows.append(
                {
                    "codigo_pee": 9,
                    "estagio": 1,
                    "patamar": pat,
                    "cenario": cen,
                    "geracao": g,
                }
            )
    renov = _StubRenovaveis(cad, subm, pd.DataFrame(ger_rows))

    with caplog.at_level(
        logging.WARNING, logger="novomodelo_bridge.decomp.converters.ncs"
    ):
        series = _pee_series(renov, _ID_MAP, _calendar(), 0)

    assert len(series) == 1
    # block 0 (patamar 1) resolves to the modal 77.07, NOT the 77.9 outlier.
    assert series[0].per_stage_blocks[0] == (77.07, 50.0, 60.0)
    assert any("modal value" in r.message for r in caplog.records)
