"""Tests for the expected inflows read off the source model's inflow tree
(``novomodelo_bridge.decomp.scenarios``). Tier-1: a synthetic inflow-file double."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.scenarios import expected_incremental_inflows
from novomodelo_bridge.decomp.temporal import build_operative_calendar


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
