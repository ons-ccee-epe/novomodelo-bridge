"""Tests for head-aware ``max_turbined`` on the DECOMP path.

Covers the DECOMP-local head-corrected engolimento formula
(``min(Σ n·q_nom·(h_op/h_nom)**k_turb, Σ n·p_nom/ρ_eq)``) that
``convert_hydros``/``convert_hydro_group_availability`` now apply to
``max_turbined_m3s`` — the affinity correction, the installed-power cap, the
un-derated (no TEIF/IP) and un-branched (no ``tipo_regulacao``) departures
from the source-model side's shared helper, the per-stage overlay, the
Itaipu per-conjunto split, and the ``AC ALTEFE`` tracked gap.
``max_generation_mw`` is untouched throughout (asserted explicitly where
relevant) — only ``max_turbined_m3s`` becomes head-aware.

Every fixture here shares one head configuration (flat forebay cota 100 m,
tailrace 20 m, ``produtibilidade_especifica`` 0.01) unless a test overrides
it, which reduces to a net head — and hence an operating head
``h_op = ρ_eq / ρ_esp`` — of exactly ``_H_OP = 80.0`` m; see ``_plant_row``.

Tier-1 only: synthetic ``_FakeDadger`` doubles and
``build_operative_calendar``, no real deck under ``example/``, no
``import novomodelo`` at module scope.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from idecomp.decomp.modelos.dadger import ACALTEFE

from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
from novomodelo_bridge.decomp.converters.hydro import (
    convert_hydro_group_availability,
    convert_hydros,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import build_operative_calendar
from tests.conftest import make_decomp_case

#: ``h_op = ρ_eq / ρ_esp`` for every fixture below that uses the default
#: head inputs (flat cota 100, tailrace 20, ρ_esp 0.01, no hydraulic loss):
#: ``h_net = 100 - 20 = 80``; ``ρ_eq = 0.01 x 80 = 0.8``;
#: ``h_op = ρ_eq / ρ_esp = 80``.
_H_OP = 80.0


def _plant_row(
    *,
    name: str = "PLANT",
    submercado: int = 1,
    jusante: int = 0,
    vmin: float = 100.0,
    vmax: float = 500.0,
    cota: tuple[float, float, float, float, float] = (100.0, 0.0, 0.0, 0.0, 0.0),
    cf: float = 20.0,
    rho_esp: float = 0.01,
    tipo_perda: int = 0,
    perdas: float = 0.0,
    tipo_regulacao: str = "D",
    conjuntos: list[tuple[int, float, float, float]] | None = None,
    tipo_turbina: int = 1,
    teif: float = 0.0,
    ip: float = 0.0,
) -> dict:
    """One synthetic ``hidr`` row with explicit per-conjunto nominal heads.

    *conjuntos* is a list of ``(n_machines, q_nom, p_nom, h_nom)`` tuples,
    one per 1-based conjunto index (default: a single 2-machine, 100 m3/s,
    50 MW, 80 m conjunto — a no-op affinity ratio). The default head inputs
    (*cota*/*cf*/*rho_esp*, with *tipo_perda*/*perdas* both zero) reduce to
    ``h_op == _H_OP == 80.0`` m for every stage.
    """
    if conjuntos is None:
        conjuntos = [(2, 100.0, 50.0, 80.0)]
    row: dict = {
        "nome_usina": name,
        "submercado": submercado,
        "codigo_usina_jusante": jusante,
        "desvio": 0,
        "volume_minimo": vmin,
        "volume_maximo": vmax,
        "numero_conjuntos_maquinas": len(conjuntos),
        "teif": teif,
        "ip": ip,
        "canal_fuga_medio": cf,
        "produtibilidade_especifica": rho_esp,
        "tipo_perda": tipo_perda,
        "perdas": perdas,
        "tipo_turbina": tipo_turbina,
        "tipo_regulacao": tipo_regulacao,
    }
    for i, a in enumerate(cota):
        row[f"a{i}_volume_cota"] = a
    for idx, (n, q_nom, p_nom, h_nom) in enumerate(conjuntos, start=1):
        row[f"maquinas_conjunto_{idx}"] = n
        row[f"vazao_nominal_conjunto_{idx}"] = q_nom
        row[f"potencia_nominal_conjunto_{idx}"] = p_nom
        row[f"queda_nominal_conjunto_{idx}"] = h_nom
    return row


def _hidr_frame(rows: dict[int, dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).T
    df.index.name = "codigo_usina"
    return df


def _uh_frame(codes: list[int], volume_inicial: float = 50.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_usina": code,
                "volume_inicial": volume_inicial,
                "vazao_defluente_minima": None,
                "volume_morto_inicial": None,
            }
            for code in codes
        ]
    )


def _calendar():
    """Three stages (two weekly + the aggregated final month), the same
    fixture shape ``test_decomp_hydro_thermal.py`` uses."""
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


def _case(dadger: object, hidr: pd.DataFrame) -> DecompCase:
    """A ``DecompCase`` pre-filled with *dadger*/*hidr* and ``_calendar()`` —
    the calendar's own stage-0 start date (2026-07-18) matches the raw
    ``start_date`` every test here used to pass directly."""
    return make_decomp_case(
        Path("unused"), dadger=dadger, hidr=hidr, calendar=_calendar()
    )


class _FakeDadger:
    """A ``Dadger`` double covering only what ``convert_hydros``/
    ``convert_hydro_group_availability`` need here: ``uh`` (the operated
    registrations), ``ac`` (dispatching only ``ACALTEFE``, ``None``
    otherwise — no other ``AC`` type is read by either consumer under
    test), and ``mp``/``fd`` (``None`` by default, i.e. no register at all
    — every plant defaults to a maintenance/availability factor of
    ``1.0`` — pass frames to exercise the split-plant path).
    """

    def __init__(
        self,
        uh: pd.DataFrame,
        altefe: pd.DataFrame | None = None,
        mp: pd.DataFrame | None = None,
        fd: pd.DataFrame | None = None,
    ) -> None:
        self._uh = uh
        self._altefe = altefe
        self._mp = mp
        self._fd = fd

    def uh(self, df: bool = False) -> pd.DataFrame:  # noqa: ARG002
        return self._uh

    def mp(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._mp

    def fd(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._fd

    def ac(
        self,
        codigo_usina: int | None = None,  # noqa: ARG002
        modificacao: type | None = None,
        df: bool = False,  # noqa: ARG002
    ) -> pd.DataFrame | None:
        if modificacao is ACALTEFE:
            return self._altefe
        return None


def test_affinity_lowers_runofriver_max_turbined() -> None:
    """A run-of-river single-conjunto plant whose nominal head (320 m)
    is well above the operating head (80 m) has its rated flow derated by
    the Francis affinity ratio ``(h_op / h_nom) ** 0.5`` — the power cap
    (``Σ n·p_nom / ρ_eq`` = 1000/0.8 = 1250) does not bind."""
    hidr = _hidr_frame(
        {1: _plant_row(conjuntos=[(1, 100.0, 1000.0, 320.0)], tipo_turbina=1)}
    )
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    effective = EffectiveCadastro(base=hidr, n_stages=1, stage_varying={})
    dadger = _FakeDadger(uh=_uh_frame([1]))

    doc = convert_hydros(_case(dadger, hidr), id_map, effective=effective)
    gen = doc["hydros"][0]["generation"]
    expected = 100.0 * (_H_OP / 320.0) ** 0.5
    assert gen["max_turbined_m3s"] == pytest.approx(expected)
    assert gen["max_turbined_m3s"] < 100.0


def test_power_cap_binds_below_affinity() -> None:
    """Nominal head equal to the operating head (ratio 1, no affinity
    reduction) but a low installed power relative to ρ_eq makes the power
    cap ``Σ n·p_nom / ρ_eq`` the binding term; ``max_generation_mw`` stays
    the unchanged rated power."""
    hidr = _hidr_frame(
        {1: _plant_row(conjuntos=[(1, 100.0, 40.0, 80.0)], tipo_turbina=1)}
    )
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    effective = EffectiveCadastro(base=hidr, n_stages=1, stage_varying={})
    dadger = _FakeDadger(uh=_uh_frame([1]))

    doc = convert_hydros(_case(dadger, hidr), id_map, effective=effective)
    gen = doc["hydros"][0]["generation"]
    assert gen["max_turbined_m3s"] == pytest.approx(40.0 / 0.8)
    assert gen["max_generation_mw"] == pytest.approx(40.0)


def test_max_generation_rated_and_no_availability_derating() -> None:
    """``max_turbined_m3s`` carries no TEIF/IP factor — a plant with
    non-zero ``teif``/``ip`` yields the identical head-corrected value as
    one with both zero — and ``max_generation_mw`` is the unchanged rated
    power in either case."""
    conjuntos = [(2, 50.0, 40.0, 100.0)]
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    dadger = _FakeDadger(uh=_uh_frame([1]))

    hidr_clean = _hidr_frame(
        {1: _plant_row(conjuntos=conjuntos, tipo_turbina=1, teif=0.0, ip=0.0)}
    )
    hidr_derated = _hidr_frame(
        {1: _plant_row(conjuntos=conjuntos, tipo_turbina=1, teif=30.0, ip=20.0)}
    )
    effective_clean = EffectiveCadastro(base=hidr_clean, n_stages=1, stage_varying={})
    effective_derated = EffectiveCadastro(
        base=hidr_derated, n_stages=1, stage_varying={}
    )

    doc_clean = convert_hydros(
        _case(dadger, hidr_clean), id_map, effective=effective_clean
    )
    doc_derated = convert_hydros(
        _case(dadger, hidr_derated), id_map, effective=effective_derated
    )
    gen_clean = doc_clean["hydros"][0]["generation"]
    gen_derated = doc_derated["hydros"][0]["generation"]

    expected = 100.0 * (_H_OP / 100.0) ** 0.5
    assert gen_clean["max_turbined_m3s"] == pytest.approx(expected)
    assert gen_derated["max_turbined_m3s"] == pytest.approx(
        gen_clean["max_turbined_m3s"]
    )
    assert gen_clean["max_generation_mw"] == pytest.approx(80.0)
    assert gen_derated["max_generation_mw"] == pytest.approx(80.0)


def test_per_stage_overlay_is_head_corrected() -> None:
    """A mid-horizon tailrace rise (``canal_fuga_medio`` 20 -> 60 m at
    the final stage) drops ρ_eq/``h_op`` there, lowering the head-corrected
    flow below the envelope — the ``hydro_unit_group_bounds`` overlay's
    ``max_turbined_m3s`` at that stage is the head-corrected value, not the
    (stage-invariant) rated ``q_g``. The unchanged earlier stages, whose
    head-corrected flow equals the envelope exactly, get no overlay row at
    all (availability, with no MP/FD derate here, sits at the rated envelope,
    so neither column drops below it)."""
    hidr = _hidr_frame(
        {1: _plant_row(conjuntos=[(1, 100.0, 1000.0, 160.0)], tipo_turbina=1)}
    )
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    calendar = _calendar()
    effective = EffectiveCadastro(
        base=hidr,
        n_stages=len(calendar),
        stage_varying={(1, "canal_fuga_medio"): (20.0, 20.0, 60.0)},
    )
    dadger = _FakeDadger(uh=_uh_frame([1]))

    values = convert_hydro_group_availability(
        _case(dadger, hidr), id_map, effective=effective
    )
    hydro_id = id_map.hydro_id(1)

    expected_stage2 = 100.0 * (40.0 / 160.0) ** 0.5
    entry2 = values[(hydro_id, 0, 2)]
    assert entry2.max_turbined_m3s == pytest.approx(expected_stage2)

    assert (hydro_id, 0, 0) not in values


@pytest.mark.parametrize(
    ("conjuntos", "tipo_turbina", "expected", "rated"),
    [
        pytest.param(
            [(1, 100.0, 1000.0, 320.0)],
            1,
            100.0 * (_H_OP / 320.0) ** 0.5,
            100.0,
            id="francis-single-affinity-binds",
        ),
        pytest.param(
            # 2560 = 80 x 32; 32 == 2**5, so (80/2560)**0.2 == (1/32)**0.2 == 0.5.
            [(1, 100.0, 1000.0, 2560.0)],
            2,
            100.0 * (_H_OP / 2560.0) ** 0.2,
            100.0,
            id="kaplan-single-affinity-binds",
        ),
        pytest.param(
            [(1, 100.0, 40.0, 80.0)],
            1,
            40.0 / 0.8,
            100.0,
            id="francis-single-power-cap-binds",
        ),
        pytest.param(
            [(1, 60.0, 600.0, 240.0), (1, 40.0, 400.0, 320.0)],
            1,
            60.0 * (_H_OP / 240.0) ** 0.5 + 40.0 * (_H_OP / 320.0) ** 0.5,
            100.0,
            id="francis-multi-conjunto-affinity-binds",
        ),
    ],
)
def test_reservoir_and_le_rated_invariant(
    conjuntos: list[tuple[int, float, float, float]],
    tipo_turbina: int,
    expected: float,
    rated: float,
) -> None:
    """A reservoir (``M``) plant across a fixture matrix (Francis/
    Kaplan turbine, affinity- vs power-cap-binding, single- vs
    multi-conjunto) matches its hand-computed full-range-head result, and
    every case's head-corrected flow sits at or below the rated flow."""
    hidr = _hidr_frame(
        {
            1: _plant_row(
                tipo_regulacao="M",
                conjuntos=conjuntos,
                tipo_turbina=tipo_turbina,
            )
        }
    )
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    effective = EffectiveCadastro(base=hidr, n_stages=1, stage_varying={})
    dadger = _FakeDadger(uh=_uh_frame([1]))

    doc = convert_hydros(_case(dadger, hidr), id_map, effective=effective)
    max_turbined = doc["hydros"][0]["generation"]["max_turbined_m3s"]
    assert max_turbined == pytest.approx(expected)
    assert max_turbined <= rated + 1e-9


def test_acaltefe_present_warns_and_proceeds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An ``AC ALTEFE`` override is logged as a tracked gap (idecomp
    exposes no value accessor for it) and conversion proceeds with the base
    ``hidr`` nominal head — the emitted ``max_turbined_m3s`` is unaffected
    by the override's mere presence."""
    hidr = _hidr_frame(
        {1: _plant_row(conjuntos=[(1, 100.0, 1000.0, 320.0)], tipo_turbina=1)}
    )
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    effective = EffectiveCadastro(base=hidr, n_stages=1, stage_varying={})
    altefe_rows = pd.DataFrame([{"codigo_usina": 1}])
    dadger = _FakeDadger(uh=_uh_frame([1]), altefe=altefe_rows)

    with caplog.at_level(
        logging.WARNING, logger="novomodelo_bridge.decomp.converters.hydro.entity"
    ):
        doc = convert_hydros(_case(dadger, hidr), id_map, effective=effective)

    assert any("ALTEFE" in r.message for r in caplog.records)
    gen = doc["hydros"][0]["generation"]
    expected = 100.0 * (_H_OP / 320.0) ** 0.5
    assert gen["max_turbined_m3s"] == pytest.approx(expected)


def test_itaipu_split_groups_use_conjunto_head_corrected_envelope() -> None:
    """Each conjunto-backed group's ``max_turbined_m3s`` is its *own*
    conjunto's head-corrected envelope (a per-group power cap, not
    plant-wide), and the entity's ``max_turbined_m3s`` is their sum —
    ``max_generation_mw`` stays each group's unchanged rated installed
    power."""
    hidr = _hidr_frame(
        {
            66: _plant_row(
                name="ITAIPU",
                tipo_regulacao="S",
                conjuntos=[
                    (1, 100.0, 1000.0, 320.0),
                    (1, 80.0, 1000.0, 160.0),
                ],
                tipo_turbina=1,
            )
        }
    )
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(66,))
    effective = EffectiveCadastro(base=hidr, n_stages=1, stage_varying={})
    mp = pd.DataFrame(
        [
            {"codigo_usina": 66, "frequencia": 50.0},
            {"codigo_usina": 66, "frequencia": 60.0},
        ]
    )
    fd = mp.copy()
    dadger = _FakeDadger(uh=_uh_frame([66]), mp=mp, fd=fd)

    doc = convert_hydros(_case(dadger, hidr), id_map, effective=effective)
    itaipu = doc["hydros"][0]
    groups = {g["id"]: g for g in itaipu["unit_groups"]}

    expected_group0 = 100.0 * (_H_OP / 320.0) ** 0.5
    expected_group1 = 80.0 * (_H_OP / 160.0) ** 0.5
    assert groups[0]["max_turbined_m3s"] == pytest.approx(expected_group0)
    assert groups[1]["max_turbined_m3s"] == pytest.approx(expected_group1)
    assert groups[0]["max_generation_mw"] == pytest.approx(1000.0)
    assert groups[1]["max_generation_mw"] == pytest.approx(1000.0)

    gen = itaipu["generation"]
    assert gen["max_turbined_m3s"] == pytest.approx(expected_group0 + expected_group1)
    assert gen["max_generation_mw"] == pytest.approx(2000.0)
