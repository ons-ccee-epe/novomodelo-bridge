"""Tests for the expected inflows read off the source model's inflow tree
(``novomodelo_bridge.decomp.scenarios``). Tier-1: a synthetic inflow-file double."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from idecomp.decomp import Vazoes

from novomodelo_bridge.core.diagnostics import Diagnostic, Severity, collect
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.cadastro import (
    EffectiveCadastro,
    build_effective_cadastro,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.scenarios import (
    convert_external_inflows,
    expected_incremental_inflows,
)
from novomodelo_bridge.decomp.temporal import build_operative_calendar
from tests.conftest import make_decomp_case


class _StubVazoes:
    """Two weekly forecasts and a two-scenario terminal fan for postos 11 and 12."""

    previsoes = pd.DataFrame(
        [
            {"estagio": 1, "11": 100.0, "12": -5.0},
            {"estagio": 2, "11": 110.0, "12": 20.0},
        ]
    )
    cenarios_gerados = pd.DataFrame(
        [
            {"estagio": 3, "cenario": 1, "11": 200.0, "12": 40.0},
            {"estagio": 3, "cenario": 2, "11": 400.0, "12": 80.0},
        ]
    )
    previsoes_com_postos_artificiais = None
    cenarios_calculados_com_postos_artificiais = None


def _hidr_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            1: {"posto": 11, "codigo_usina_jusante": 2, "desvio": 0},
            2: {"posto": 12, "codigo_usina_jusante": 0, "desvio": 0},
        }
    ).T
    df.index.name = "codigo_usina"
    return df


def test_expected_incremental_inflows_weight_the_fan_by_its_probabilities() -> None:
    calendar = build_operative_calendar(
        date(2026, 7, 18), [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    )
    hidr = _hidr_frame()
    inflows = expected_incremental_inflows(
        _StubVazoes(),
        EffectiveCadastro(base=hidr, n_stages=3, stage_varying={}),
        DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1, 2)),
        calendar,
        [0.75, 0.25],
    )
    assert inflows[(1, 0)] == 100.0
    assert inflows[(2, 0)] == -5.0
    assert inflows[(1, 1)] == 110.0
    assert inflows[(1, 2)] == pytest.approx(0.75 * 200.0 + 0.25 * 400.0)
    assert inflows[(2, 2)] == pytest.approx(0.75 * 40.0 + 0.25 * 80.0)


_SPLIT_WARNING = "decomp-nw-inflow-split-fallback"
_SPLIT_KEYS = (
    {"estagio": 1},
    {"estagio": 2},
    {"estagio": 3, "cenario": 1},
    {"estagio": 3, "cenario": 2},
)
_NW_COUPLING = (30.0, 40.0, 50.0, 60.0)
_ABSORBER_COUPLING = (20.0, 25.0, 10.0, 15.0)
_NO_DEVIATION = (0.0, 0.0, 0.0, 0.0)


def _split_calendar():
    return build_operative_calendar(
        date(2026, 7, 18), [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    )


def _split_hidr() -> pd.DataFrame:
    """Plants 1 and 4 (postos 11 and 14) drain to plant 2 (posto 12); plants 3
    and 5 (postos 13 and 15) drain to the sink."""
    df = pd.DataFrame(
        {
            1: {"nome_usina": "NWONE", "posto": 11, "codigo_usina_jusante": 2},
            2: {"nome_usina": "ABSORBER", "posto": 12, "codigo_usina_jusante": 0},
            3: {"nome_usina": "OTHER", "posto": 13, "codigo_usina_jusante": 0},
            4: {"nome_usina": "NWTWO", "posto": 14, "codigo_usina_jusante": 2},
            5: {"nome_usina": "NWSINK", "posto": 15, "codigo_usina_jusante": 0},
        }
    ).T
    df.index.name = "codigo_usina"
    return df


def _split_effective() -> EffectiveCadastro:
    return EffectiveCadastro(
        base=_split_hidr(), n_stages=len(_split_calendar()), stage_varying={}
    )


def _split_id_map(*hydro_codes: int, dormant: tuple[int, ...]) -> DecompIdMap:
    return DecompIdMap(
        bus_codes=(1,),
        bus_names=("SE",),
        hydro_codes=hydro_codes,
        dormant_hydro_codes=dormant,
    )


def _split_vazoes(
    *,
    nw: Mapping[str, Sequence[float]] | None = None,
    gaps: Sequence[float] = _NO_DEVIATION,
    nw_operation: Sequence[float] = _NO_DEVIATION,
) -> SimpleNamespace:
    """Four paired rows: two weekly forecasts and a two-scenario terminal fan.

    *nw* gives each NW posto's coupling series (default: posto 11). The
    operation tables hold the whole group's flow on posto 12, plus the row's
    entry of *gaps*, and *nw_operation* on each NW posto.
    """
    nw = {"11": _NW_COUPLING} if nw is None else nw
    coupling: list[dict] = []
    operation: list[dict] = []
    for i, key in enumerate(_SPLIT_KEYS):
        shares = {posto: series[i] for posto, series in nw.items()}
        coupling.append({**key, **shares, "12": _ABSORBER_COUPLING[i], "13": 7.0})
        operation.append(
            {
                **key,
                **dict.fromkeys(nw, nw_operation[i]),
                "12": sum(shares.values()) + _ABSORBER_COUPLING[i] + gaps[i],
                "13": 7.0,
            }
        )
    return SimpleNamespace(
        previsoes=pd.DataFrame(operation[:2]),
        cenarios_gerados=pd.DataFrame(operation[2:]),
        previsoes_com_postos_artificiais=pd.DataFrame(coupling[:2]),
        cenarios_calculados_com_postos_artificiais=pd.DataFrame(coupling[2:]),
    )


def _series(table: pd.DataFrame, hydro_id: int) -> list[float]:
    rows = table[table["hydro_id"] == hydro_id].sort_values(["stage_id", "scenario_id"])
    return rows["value_m3s"].tolist()


def _convert(
    vazoes: SimpleNamespace, id_map: DecompIdMap
) -> tuple[pd.DataFrame, list[Diagnostic]]:
    case = make_decomp_case(Path("unused"), calendar=_split_calendar())
    with collect() as collected:
        table = convert_external_inflows(
            case, id_map, vazoes=vazoes, effective=_split_effective()
        ).to_pandas()
    return table, [d for d in collected if d.code == _SPLIT_WARNING]


class TestNwInflowSplit:
    def test_decomp_mini_estrela_takes_its_coupling_series_and_salto_the_rest(
        self, decomp_mini_case: DecompCase
    ) -> None:
        id_map = DecompIdMap.from_dadger(decomp_mini_case.dadger)
        effective, _ = build_effective_cadastro(
            decomp_mini_case.dadger, decomp_mini_case.hidr, decomp_mini_case.calendar
        )
        vazoes = Vazoes.read(str(decomp_mini_case.files.vazoes))
        estrela, salto = 260, 262

        with collect() as collected:
            table = convert_external_inflows(
                decomp_mini_case, id_map, vazoes=vazoes, effective=effective
            ).to_pandas()

        assert _series(table, id_map.hydro_id(estrela)) == [194, 156, 140, 106, 151]
        assert _series(table, id_map.hydro_id(salto)) == [65, 52, 47, 35, 50]
        assert vazoes.previsoes is not None
        assert vazoes.cenarios_gerados is not None
        operation_rows = [
            *vazoes.previsoes.sort_values("estagio").to_dict("records"),
            *vazoes.cenarios_gerados.sort_values(["estagio", "cenario"]).to_dict(
                "records"
            ),
        ]
        others = [c for c in id_map.hydro_codes if c not in (estrela, salto)]
        assert others
        for code in others:
            posto = str(effective.inflow_gauge(code, 0))
            assert _series(table, id_map.hydro_id(code)) == [
                float(row[posto]) for row in operation_rows
            ]
        assert not [d for d in collected if d.code == _SPLIT_WARNING]

    def test_a_gap_of_exactly_one_m3s_passes_the_guard(self) -> None:
        table, warnings = _convert(
            _split_vazoes(gaps=(1.0, -1.0, 0.0, 1.0)),
            _split_id_map(1, 2, 3, dormant=(1,)),
        )

        assert _series(table, 0) == [30.0, 40.0, 50.0, 60.0]
        assert _series(table, 1) == [21.0, 24.0, 10.0, 16.0]
        assert _series(table, 2) == [7.0] * 4
        assert warnings == []

    def test_a_gap_above_one_m3s_falls_back_for_the_whole_group_with_one_warning(
        self,
    ) -> None:
        table, warnings = _convert(
            _split_vazoes(gaps=(0.0, 0.0, 1.5, 0.0)),
            _split_id_map(1, 2, 3, dormant=(1,)),
        )

        assert _series(table, 0) == [0.0] * 4
        assert _series(table, 1) == [50.0, 65.0, 61.5, 75.0]
        [warning] = warnings
        assert warning.severity is Severity.WARNING
        assert warning.table is not None
        assert warning.table.rows == [[2, "ABSORBER", "1", 1.5, 0.0]]
        text = " ".join([warning.summary, *warning.notes])
        for leak in ("docs/", "plans/", "~/git", "ticket-", "epic-", "_"):
            assert leak not in text

    def test_an_nw_operation_value_of_exactly_one_m3s_passes_the_guard(self) -> None:
        table, warnings = _convert(
            _split_vazoes(nw_operation=(1.0, 1.0, 1.0, 1.0)),
            _split_id_map(1, 2, 3, dormant=(1,)),
        )

        assert _series(table, 0) == [30.0, 40.0, 50.0, 60.0]
        assert _series(table, 1) == [20.0, 25.0, 10.0, 15.0]
        assert warnings == []

    def test_an_nw_operation_value_above_one_m3s_falls_back_for_the_whole_group_with_one_warning(
        self,
    ) -> None:
        table, warnings = _convert(
            _split_vazoes(nw_operation=(0.0, 1.5, 0.0, 0.0)),
            _split_id_map(1, 2, 3, dormant=(1,)),
        )

        assert _series(table, 0) == [0.0, 1.5, 0.0, 0.0]
        assert _series(table, 1) == [50.0, 65.0, 60.0, 75.0]
        [warning] = warnings
        assert warning.table is not None
        assert warning.table.rows == [[2, "ABSORBER", "1", 0.0, 1.5]]

    def test_an_nw_plant_without_an_operated_plant_downstream_keeps_its_operation_series(
        self,
    ) -> None:
        vazoes = _split_vazoes()
        vazoes.previsoes["15"] = 9.0
        vazoes.cenarios_gerados["15"] = 9.0

        table, warnings = _convert(vazoes, _split_id_map(1, 2, 3, 5, dormant=(1, 5)))

        assert _series(table, 0) == [30.0, 40.0, 50.0, 60.0]
        assert _series(table, 3) == [9.0] * 4
        [warning] = warnings
        assert warning.table is not None
        assert warning.table.rows == [[5, "NWSINK", "5", "-", "-"]]

    def test_an_nw_plant_without_the_coupling_tables_raises(self) -> None:
        id_map = _split_id_map(1, 2, 3, dormant=(1,))
        for missing in (
            "previsoes_com_postos_artificiais",
            "cenarios_calculados_com_postos_artificiais",
        ):
            vazoes = _split_vazoes()
            setattr(vazoes, missing, None)

            with pytest.raises(ValueError, match=rf"plants? 1\b.*{missing}"):
                _convert(vazoes, id_map)

    def test_an_operation_row_without_its_coupling_pair_raises(self) -> None:
        id_map = _split_id_map(1, 2, 3, dormant=(1,))
        weekly = _split_vazoes()
        weekly.previsoes_com_postos_artificiais = (
            weekly.previsoes_com_postos_artificiais.iloc[:1]
        )
        monthly = _split_vazoes()
        monthly.cenarios_calculados_com_postos_artificiais = (
            monthly.cenarios_calculados_com_postos_artificiais.iloc[:1]
        )

        with pytest.raises(ValueError, match=r"estagio 2\b"):
            _convert(weekly, id_map)
        with pytest.raises(ValueError, match=r"estagio 3, cenario 2\b"):
            _convert(monthly, id_map)

    def test_two_nw_plants_above_one_absorber_split_as_one_group(self) -> None:
        nw = {"11": _NW_COUPLING, "14": (3.0, 4.0, 5.0, 6.0)}

        table, warnings = _convert(
            _split_vazoes(nw=nw), _split_id_map(1, 2, 3, 4, dormant=(1, 4))
        )

        assert _series(table, 0) == [30.0, 40.0, 50.0, 60.0]
        assert _series(table, 3) == [3.0, 4.0, 5.0, 6.0]
        assert _series(table, 1) == [20.0, 25.0, 10.0, 15.0]
        assert warnings == []

    def test_the_expected_inflows_take_the_split_and_emit_no_warning(self) -> None:
        calendar = _split_calendar()
        id_map = _split_id_map(1, 2, 3, dormant=(1,))
        effective = _split_effective()

        with collect() as collected:
            passed = expected_incremental_inflows(
                _split_vazoes(), effective, id_map, calendar, [0.75, 0.25]
            )
            fallen_back = expected_incremental_inflows(
                _split_vazoes(gaps=(0.0, 0.0, 1.5, 0.0)),
                effective,
                id_map,
                calendar,
                [0.75, 0.25],
            )

        assert passed[(1, 0)] == 30.0
        assert passed[(2, 0)] == 20.0
        assert passed[(1, 2)] == pytest.approx(0.75 * 50.0 + 0.25 * 60.0)
        assert passed[(2, 2)] == pytest.approx(0.75 * 10.0 + 0.25 * 15.0)
        assert fallen_back[(1, 0)] == 0.0
        assert fallen_back[(2, 2)] == pytest.approx(0.75 * 61.5 + 0.25 * 75.0)
        assert collected == []
