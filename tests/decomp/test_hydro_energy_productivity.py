"""Tests for the ``hydro_energy_productivity.parquet`` table of the DECOMP track.

Tier-1: a synthetic ``hidr`` frame inside an ``EffectiveCadastro``, no deck.
"""

from __future__ import annotations

import pandas as pd
import pytest

from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
from novomodelo_bridge.decomp.converters.hydro import convert_hydro_energy_productivity
from novomodelo_bridge.decomp.id_map import DecompIdMap

# h(v) = 200 + 0.2·v over [100, 500]: 220 m at vmin, 300 m at vmax, mean 260 m.
_COTA = (200.0, 0.2, 0.0, 0.0, 0.0)
_CF = 20.0
_RHO_ESP = 0.01
_MEAN_HEAD_RHO = _RHO_ESP * (260.0 - _CF)


def _row(tipo_regulacao: str, volume_referencia: float = 300.0) -> dict:
    row: dict = {
        "nome_usina": "PLANT",
        "volume_minimo": 100.0,
        "volume_maximo": 500.0,
        "volume_referencia": volume_referencia,
        "canal_fuga_medio": _CF,
        "produtibilidade_especifica": _RHO_ESP,
        "tipo_perda": 0,
        "perdas": 0.0,
        "tipo_regulacao": tipo_regulacao,
    }
    for i, a in enumerate(_COTA):
        row[f"a{i}_volume_cota"] = a
    return row


def _effective(
    rows: dict[int, dict],
    n_stages: int = 3,
    stage_varying: dict[tuple[int, str], tuple[float, ...]] | None = None,
) -> EffectiveCadastro:
    hidr = pd.DataFrame.from_dict(rows, orient="index")
    return EffectiveCadastro(
        base=hidr, n_stages=n_stages, stage_varying=stage_varying or {}
    )


def _id_map(codes: tuple[int, ...]) -> DecompIdMap:
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=codes)


def test_reservoirs_integrate_and_run_of_river_takes_the_point_head() -> None:
    """ "M" and "S" average the head over [vmin, vmax]; "D" reads it at
    volume_referencia."""
    effective = _effective({1: _row("M"), 2: _row("S"), 3: _row("D", 450.0)})
    id_map = _id_map((1, 2, 3))

    rows = convert_hydro_energy_productivity(effective, id_map).to_pylist()

    assert [r["stage_id"] for r in rows] == [None, None, None]
    integrated = {
        r["hydro_id"]: r["integrated_equivalent_productivity_mw_per_m3s"] for r in rows
    }
    assert integrated[id_map.hydro_id(1)] == pytest.approx(_MEAN_HEAD_RHO)
    assert integrated[id_map.hydro_id(2)] == pytest.approx(_MEAN_HEAD_RHO)
    point = _RHO_ESP * (200.0 + 0.2 * 450.0 - _CF)
    assert integrated[id_map.hydro_id(3)] == pytest.approx(point)


def test_fpha_plant_keeps_its_integrated_value_without_a_rho_eq() -> None:
    """An FPHA plant's ρ_eq is NULL; its stored-energy ρ is still supplied, and
    a constant-productivity plant's ρ_eq takes the head at its initial volume."""
    effective = _effective({1: _row("M"), 2: _row("M")})
    id_map = _id_map((1, 2))

    rows = convert_hydro_energy_productivity(
        effective, id_map, {1: 460.0, 2: 460.0}, fpha_codes={2}
    ).to_pylist()

    by_id = {r["hydro_id"]: r for r in rows}
    constant = by_id[id_map.hydro_id(1)]
    fpha = by_id[id_map.hydro_id(2)]
    assert constant["equivalent_productivity_mw_per_m3s"] == pytest.approx(
        _RHO_ESP * (200.0 + 0.2 * 460.0 - _CF)
    )
    assert fpha["equivalent_productivity_mw_per_m3s"] is None
    for row in (constant, fpha):
        assert row["integrated_equivalent_productivity_mw_per_m3s"] == pytest.approx(
            _MEAN_HEAD_RHO
        )


def test_stage_varying_override_moves_the_value_to_per_stage_rows() -> None:
    """An AC JUSMED from stage 2 on leaves the default row's integrated value
    NULL and writes one integrated row per stage; ρ_eq stays on the default."""
    effective = _effective(
        {1: _row("M")},
        stage_varying={(1, "canal_fuga_medio"): (_CF, _CF, 30.0)},
    )
    id_map = _id_map((1,))

    rows = convert_hydro_energy_productivity(effective, id_map).to_pylist()

    default = next(r for r in rows if r["stage_id"] is None)
    assert default["equivalent_productivity_mw_per_m3s"] is not None
    assert default["integrated_equivalent_productivity_mw_per_m3s"] is None
    per_stage = {r["stage_id"]: r for r in rows if r["stage_id"] is not None}
    assert sorted(per_stage) == [0, 1, 2]
    assert all(
        r["equivalent_productivity_mw_per_m3s"] is None for r in per_stage.values()
    )
    values = [
        per_stage[s]["integrated_equivalent_productivity_mw_per_m3s"] for s in range(3)
    ]
    assert values == pytest.approx(
        [_MEAN_HEAD_RHO, _MEAN_HEAD_RHO, _RHO_ESP * (260.0 - 30.0)]
    )


def test_unread_override_columns_are_null() -> None:
    """``reference_outflow_m3s`` and the ρ_esp override are never written."""
    effective = _effective({1: _row("M")})

    table = convert_hydro_energy_productivity(effective, _id_map((1,)))

    for column in ("reference_outflow_m3s", "specific_productivity_mw_per_m3s_per_m"):
        assert table[column].to_pylist() == [None]
