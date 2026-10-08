"""Tests for collapsing run-of-river ('D') storage to a reference volume.

Under the DECOMP reservoir predicate a reservoir is ``tipo_regulacao in
("M", "S")``; only ``tipo_regulacao == "D"`` is run-of-river and cannot
accumulate water across stages. Before this fix, every storage consumer
(the entity ``reservoir`` block, the per-stage storage-bounds emitter, the
initial-storage %) fed the ``hidr`` registry's ``(volume_minimo,
volume_maximo)`` band straight through for a ``D`` plant, emitting phantom
weekly storage the plant does not have. :func:`~novomodelo_bridge.decomp.
cadastro.effective_storage_range` collapses that band to a single point
(``volume_referencia``) for a ``D`` plant only; every storage consumer now
routes through it, while productivity keeps reading the full range.

Tier-1 only: synthetic ``_StubDadger`` doubles and a synthetic
``EffectiveCadastro``/calendar, no real deck.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from novomodelo_bridge.decomp.converters.bounds import convert_storage_bounds
from novomodelo_bridge.decomp.converters.cadastro import (
    EffectiveCadastro,
    effective_storage_range,
    storage_envelope,
)
from novomodelo_bridge.decomp.converters.hydro import (
    convert_energy_productivity,
    convert_hydros,
    convert_initial_storage,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import OperativeStage
from tests.conftest import make_decomp_case


def _plant_row(
    *,
    name: str = "PLANT",
    vmin: float = 100.0,
    vmax: float = 500.0,
    vol_ref: float | None = 300.0,
    tipo_regulacao: str = "D",
    submercado: int = 1,
    jusante: int = 0,
    a0: float = 300.0,
    a1: float = 0.0,
    cf: float = 20.0,
    rho_esp: float = 0.009,
) -> dict:
    """One synthetic ``hidr`` row — every column the tests below touch."""
    return {
        "nome_usina": name,
        "submercado": submercado,
        "codigo_usina_jusante": jusante,
        "volume_minimo": vmin,
        "volume_maximo": vmax,
        "volume_referencia": vol_ref,
        "tipo_regulacao": tipo_regulacao,
        "numero_conjuntos_maquinas": 1,
        "maquinas_conjunto_1": 2,
        "vazao_nominal_conjunto_1": 100.0,
        "potencia_nominal_conjunto_1": 50.0,
        "teif": 0.0,
        "ip": 0.0,
        "a0_volume_cota": a0,
        "a1_volume_cota": a1,
        "a2_volume_cota": 0.0,
        "a3_volume_cota": 0.0,
        "a4_volume_cota": 0.0,
        "canal_fuga_medio": cf,
        "produtibilidade_especifica": rho_esp,
        "tipo_perda": 0,
        "perdas": 0.0,
    }


def _hidr_frame(rows: dict[int, dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).T
    df.index.name = "codigo_usina"
    return df


def _effective(hidr: pd.DataFrame, n_stages: int = 1) -> EffectiveCadastro:
    """No-override view of *hidr*: every accessor falls through to the base."""
    return EffectiveCadastro(base=hidr, n_stages=n_stages, stage_varying={})


def _calendar(n_stages: int) -> list[OperativeStage]:
    """Placeholder stages — only their count is used by the storage-bounds
    emitter, mirroring ``test_decomp_storage_bounds.py``'s own fixture.
    """
    return [
        OperativeStage(
            index=i,
            start_date=date(2026, 7, 4),
            end_date=date(2026, 7, 11),
            season_id=6,
            block_hours=(168.0,),
        )
        for i in range(n_stages)
    ]


class _StubDadger:
    """A ``Dadger`` double exposing only ``uh``/``ac``/``mp``/``fd``."""

    def __init__(self, uh: pd.DataFrame) -> None:
        self._uh = uh

    def uh(self, df: bool = False) -> pd.DataFrame:  # noqa: ARG002
        return self._uh

    def ac(  # noqa: ARG002
        self,
        codigo_usina: int | None = None,
        modificacao: type | None = None,
        df: bool = False,
    ) -> pd.DataFrame:
        # No AC ALTEFE (or any other) overrides on this synthetic stub.
        return pd.DataFrame()

    def mp(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return None

    def fd(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return None


def _uh_frame(
    codes: list[int], pct_by_code: dict[int, float] | None = None
) -> pd.DataFrame:
    pct_by_code = pct_by_code or {}
    return pd.DataFrame(
        [
            {
                "codigo_usina": code,
                "volume_inicial": pct_by_code.get(code, 50.0),
                "vazao_defluente_minima": None,
                "volume_morto_inicial": None,
            }
            for code in codes
        ]
    )


_ID_MAP_SINGLE = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))


# ---------------------------------------------------------------------------
# A D plant's storage collapses to vol_ref.
# ---------------------------------------------------------------------------


def test_d_plant_storage_collapses_to_vol_ref() -> None:
    """An operated ``D`` plant with a non-collapsed hidr range (100..500)
    collapses to its ``volume_referencia`` (300) in both the entity
    ``reservoir`` block and the per-stage storage-bounds emitter (zero
    override rows — its per-stage range already equals its envelope).
    """
    hidr = _hidr_frame(
        {1: _plant_row(vmin=100.0, vmax=500.0, vol_ref=300.0, tipo_regulacao="D")}
    )
    effective = _effective(hidr, n_stages=2)

    case = make_decomp_case(
        Path("unused"),
        dadger=_StubDadger(_uh_frame([1])),
        hidr=hidr,
        calendar=_calendar(2),
    )
    doc = convert_hydros(case, _ID_MAP_SINGLE, effective=effective)
    assert doc["hydros"][0]["reservoir"] == {
        "min_storage_hm3": 300.0,
        "max_storage_hm3": 300.0,
    }

    contributions = convert_storage_bounds(case, _ID_MAP_SINGLE, effective=effective)
    assert contributions == []


# ---------------------------------------------------------------------------
# M and S plants are unaffected (only 'D' collapses).
# ---------------------------------------------------------------------------


def test_m_and_s_plants_keep_full_range() -> None:
    """Under the DECOMP predicate, a reservoir is ``tipo_regulacao in ("M",
    "S")`` — neither collapses, even though both carry a ``volume_referencia``
    that a naive predicate might mistake for a collapse target.
    """
    hidr = _hidr_frame(
        {
            1: _plant_row(
                name="M_PLANT",
                vmin=100.0,
                vmax=500.0,
                vol_ref=300.0,
                tipo_regulacao="M",
            ),
            2: _plant_row(
                name="S_PLANT", vmin=50.0, vmax=250.0, vol_ref=150.0, tipo_regulacao="S"
            ),
        }
    )
    effective = _effective(hidr, n_stages=1)

    assert effective_storage_range(effective, 1, 0) == (100.0, 500.0)
    assert effective_storage_range(effective, 2, 0) == (50.0, 250.0)
    assert storage_envelope(effective, 1) == (100.0, 500.0)
    assert storage_envelope(effective, 2) == (50.0, 250.0)


# ---------------------------------------------------------------------------
# Initial storage for a D plant equals vol_ref, independent of UH %.
# ---------------------------------------------------------------------------


def test_d_plant_initial_storage_is_vol_ref() -> None:
    """A ``D`` plant's initial reservoir volume is always ``vol_ref`` — the
    collapsed ``v_min == v_max`` range zeroes out the ``UH`` percentage term
    regardless of its declared value.
    """
    hidr = _hidr_frame(
        {1: _plant_row(vmin=100.0, vmax=500.0, vol_ref=300.0, tipo_regulacao="D")}
    )
    effective = _effective(hidr, n_stages=1)

    for pct in (0.0, 42.0, 100.0):
        case = make_decomp_case(
            Path("unused"), dadger=_StubDadger(_uh_frame([1], {1: pct})), hidr=hidr
        )
        storage = convert_initial_storage(case, _ID_MAP_SINGLE, effective=effective)
        assert storage[0]["value_hm3"] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Productivity is NOT routed through the collapse.
# ---------------------------------------------------------------------------


def test_d_plant_productivity_uses_full_range() -> None:
    """A ``D`` plant's equivalent productivity uses the full-range mean cota
    (``volume_minimo``..``volume_maximo``), not the storage collapse. The
    linear cota polynomial (``cota(V) = V``) and a ``vol_ref`` deliberately
    off the range midpoint make the two readings diverge if productivity
    were wrongly routed through :func:`effective_storage_range`: the correct
    full-range mean is 300 (midpoint of 100..500); the wrong collapsed
    reading would be 200 (``vol_ref``).
    """
    hidr = _hidr_frame(
        {
            1: _plant_row(
                vmin=100.0,
                vmax=500.0,
                vol_ref=200.0,
                tipo_regulacao="D",
                a0=0.0,
                a1=1.0,
                cf=20.0,
                rho_esp=0.01,
            )
        }
    )
    effective = _effective(hidr, n_stages=1)

    table = convert_energy_productivity(effective, _ID_MAP_SINGLE).to_pandas()
    # mean_cota([100, 500]) = 300; h_net = 300 - 20 = 280; rho_eq = 0.01 * 280.
    assert table["equivalent_productivity_mw_per_m3s"].iloc[0] == pytest.approx(
        0.01 * 280.0
    )


# ---------------------------------------------------------------------------
# The collapse diagnostic counts only D plants whose hidr range
# actually carried volume_minimo != volume_maximo.
# ---------------------------------------------------------------------------


def test_collapse_diagnostic_counts_d_plants(caplog: pytest.LogCaptureFixture) -> None:
    """Two operated ``D`` plants with non-collapsed hidr ranges are reported
    in one INFO log; a third ``D`` plant whose hidr range is already
    collapsed (``v_min == v_max``) contributes nothing to change and is not
    counted.
    """
    hidr = _hidr_frame(
        {
            1: _plant_row(name="D1", vmin=100.0, vmax=500.0, vol_ref=300.0),
            2: _plant_row(name="D2", vmin=50.0, vmax=250.0, vol_ref=150.0),
            3: _plant_row(name="D3", vmin=10.0, vmax=10.0, vol_ref=10.0),
        }
    )
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1, 2, 3))
    effective = _effective(hidr, n_stages=1)

    case = make_decomp_case(
        Path("unused"),
        dadger=_StubDadger(_uh_frame([1, 2, 3])),
        hidr=hidr,
        calendar=_calendar(1),
    )
    with caplog.at_level(
        logging.INFO, logger="novomodelo_bridge.decomp.converters.hydro.entity"
    ):
        convert_hydros(case, id_map, effective=effective)

    assert "collapsed 2 run-of-river" in caplog.text


# ---------------------------------------------------------------------------
# A missing/zero vol_ref falls back to volume_minimo, never a
# zero-width range at zero.
# ---------------------------------------------------------------------------


def test_missing_vol_ref_falls_back_to_vmin() -> None:
    """A ``D`` plant whose ``volume_referencia`` is ``NaN``/``0`` falls back
    to ``(volume_minimo, volume_minimo)`` — never a zero-width range at
    ``0.0``.
    """
    hidr_nan = _hidr_frame(
        {1: _plant_row(vmin=120.0, vmax=480.0, vol_ref=None, tipo_regulacao="D")}
    )
    assert effective_storage_range(_effective(hidr_nan), 1, 0) == (120.0, 120.0)

    hidr_zero = _hidr_frame(
        {1: _plant_row(vmin=90.0, vmax=300.0, vol_ref=0.0, tipo_regulacao="D")}
    )
    assert effective_storage_range(_effective(hidr_zero), 1, 0) == (90.0, 90.0)

    hidr_negative = _hidr_frame(
        {1: _plant_row(vmin=70.0, vmax=200.0, vol_ref=-5.0, tipo_regulacao="D")}
    )
    assert effective_storage_range(_effective(hidr_negative), 1, 0) == (70.0, 70.0)
