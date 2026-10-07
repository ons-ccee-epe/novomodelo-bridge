"""Tests for the head/productivity ``AC`` override family.

Covers the three scalar mnemonics (``PROESP``/``PERHID``/``JUSMED``, flowing
through the existing scalar-override machinery), the ``COTVOL`` polynomial
(the fifth, multi-row-per-plant shape, via ``_read_polynomial_overrides``),
and the resulting per-stage, full-range equivalent productivity
(``_equivalent_productivity_mw_per_m3s``) both ``convert_energy_
productivity`` and ``convert_hydro_group_availability`` consume. Tier-1
only: synthetic ``_FakeDadger``/``_StubDadger`` doubles and a synthetic
``build_operative_calendar``, no real deck.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from idecomp.decomp.modelos.dadger import ACCOTVOL, ACJUSMED, ACPERHID, ACPROESP

from novomodelo_bridge.core.productivity import apply_hydraulic_loss, mean_cota
from novomodelo_bridge.decomp.converters.cadastro import (
    EffectiveCadastro,
    OutOfHorizon,
    _read_polynomial_overrides,
    build_effective_cadastro,
)
from novomodelo_bridge.decomp.converters.hydro import (
    convert_energy_productivity,
    convert_hydro_group_availability,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import build_operative_calendar
from tests.conftest import make_decomp_case

if TYPE_CHECKING:
    from novomodelo_bridge.decomp.temporal import OperativeStage


class _FakeDadger:
    """Return a preset ``AC`` frame (or ``None``) dispatched by ``modificacao`` class.

    Mirrors ``test_decomp_cadastro.py``'s double of the same name — kept
    local rather than shared so this module stays a self-contained Tier-1
    unit, matching the sibling cadastro test files' own convention.
    """

    def __init__(self, frames: dict[type, pd.DataFrame | None]) -> None:
        self._frames = frames

    def ac(
        self,
        codigo_usina: int | None = None,  # noqa: ARG002
        modificacao: type | None = None,
        df: bool = True,  # noqa: ARG002
    ) -> pd.DataFrame | None:
        if modificacao is None:
            return None
        return self._frames.get(modificacao)


class _StubDadger:
    """A ``Dadger`` double exposing only ``mp``/``fd``, both empty — every
    plant here defaults to a maintenance/availability factor of ``1.0`` so
    the B8 hydraulic ceiling is the only thing under test.
    """

    def mp(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return None

    def fd(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return None


def _calendar() -> list[OperativeStage]:
    """Stages 0, 1 = July weekly stages; stage 2 = the August monthly stage."""
    stage_block_hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), stage_block_hours)


def _plant_row(
    *,
    vmin: float = 100.0,
    vmax: float = 500.0,
    cota: tuple[float, float, float, float, float] = (300.0, 0.0, 0.0, 0.0, 0.0),
    cf: float = 20.0,
    rho_esp: float = 0.01,
    tipo_perda: int = 0,
    perdas: float = 0.0,
    tipo_regulacao: str = "M",
    n_conjuntos: int = 1,
    n_machines: int = 2,
    q_nom: float = 50.0,
    p_nom: float = 200.0,
    submercado: int = 1,
    jusante: int = 0,
    name: str = "PLANT",
) -> dict:
    """One synthetic ``hidr`` row — every column the tests below touch."""
    row: dict = {
        "nome_usina": name,
        "submercado": submercado,
        "codigo_usina_jusante": jusante,
        "desvio": 0,
        "volume_minimo": vmin,
        "volume_maximo": vmax,
        "numero_conjuntos_maquinas": n_conjuntos,
        "maquinas_conjunto_1": n_machines,
        "vazao_nominal_conjunto_1": q_nom,
        "potencia_nominal_conjunto_1": p_nom,
        "teif": 0.0,
        "ip": 0.0,
        "canal_fuga_medio": cf,
        "produtibilidade_especifica": rho_esp,
        "tipo_perda": tipo_perda,
        "perdas": perdas,
        "tipo_regulacao": tipo_regulacao,
    }
    for i, a in enumerate(cota):
        row[f"a{i}_volume_cota"] = a
    return row


def _hidr_frame(rows: dict[int, dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).T
    df.index.name = "codigo_usina"
    return df


# ---------------------------------------------------------------------------
# The three scalar head/productivity mnemonics.
# ---------------------------------------------------------------------------


def test_proesp_override_applies_all_stages() -> None:
    """A blank-date ``AC PROESP`` row is permanent — it applies from
    stage 0 forward, so ``effective.value(code,
    "produtibilidade_especifica", s)`` equals the override at every stage.
    """
    hidr = _hidr_frame({1: _plant_row(rho_esp=0.009)})
    dadger = _FakeDadger(
        {
            ACPROESP: pd.DataFrame(
                [
                    {
                        "codigo_usina": 1,
                        "produtibilidade": 0.015,
                        "mes": "",
                        "semana": float("nan"),
                        "ano": float("nan"),
                    }
                ]
            )
        }
    )
    effective, report = build_effective_cadastro(dadger, hidr, _calendar())
    assert [effective.value(1, "produtibilidade_especifica", s) for s in range(3)] == [
        0.015,
        0.015,
        0.015,
    ]
    assert report.applied["produtibilidade_especifica"] == 1


def test_perhid_override_applies_all_stages() -> None:
    """``AC PERHID`` maps to ``perdas``."""
    hidr = _hidr_frame({1: _plant_row(perdas=0.0)})
    dadger = _FakeDadger(
        {
            ACPERHID: pd.DataFrame(
                [
                    {
                        "codigo_usina": 1,
                        "coeficiente": 3.5,
                        "mes": "",
                        "semana": float("nan"),
                        "ano": float("nan"),
                    }
                ]
            )
        }
    )
    effective, report = build_effective_cadastro(dadger, hidr, _calendar())
    assert [effective.value(1, "perdas", s) for s in range(3)] == [3.5, 3.5, 3.5]
    assert report.applied["perdas"] == 1


def test_jusmed_override_applies_all_stages() -> None:
    """``AC JUSMED`` maps to ``canal_fuga_medio``."""
    hidr = _hidr_frame({1: _plant_row(cf=20.0)})
    dadger = _FakeDadger(
        {
            ACJUSMED: pd.DataFrame(
                [
                    {
                        "codigo_usina": 1,
                        "cota": 18.5,
                        "mes": "",
                        "semana": float("nan"),
                        "ano": float("nan"),
                    }
                ]
            )
        }
    )
    effective, report = build_effective_cadastro(dadger, hidr, _calendar())
    assert [effective.value(1, "canal_fuga_medio", s) for s in range(3)] == [
        18.5,
        18.5,
        18.5,
    ]
    assert report.applied["canal_fuga_medio"] == 1


# ---------------------------------------------------------------------------
# The fifth shape, AC COTVOL (multi-row-per-plant polynomial).
# ---------------------------------------------------------------------------


def test_cotvol_polynomial_forward_fills() -> None:
    """``ACCOTVOL`` ``ordem`` 1..5 rows (the raw 1-based coefficient
    index idecomp surfaces; the reader normalises it to a 0-based tuple slot)
    dated ``AGO semana 1`` (the 3-stage calendar's index-2 stage) replace the
    plant's forebay-cota polynomial from that stage forward; a missing
    ``ordem`` in the override group defaults its coefficient to ``0.0``; the
    earlier stages still read the base ``a{0..4}_volume_cota``.
    """
    hidr = _hidr_frame({1: _plant_row(cota=(100.0, 1.0, 0.0, 0.0, 0.0))})
    override_rows = [
        {
            "codigo_usina": 1,
            "ordem": 1,  # a0
            "coeficiente": 50.0,
            "mes": "AGO",
            "semana": 1.0,
            "ano": 2026.0,
        },
        {
            "codigo_usina": 1,
            "ordem": 2,  # a1
            "coeficiente": 2.0,
            "mes": "AGO",
            "semana": 1.0,
            "ano": 2026.0,
        },
        # ordem 3..5 (a2..a4) omitted -> default to 0.0.
    ]
    dadger = _FakeDadger({ACCOTVOL: pd.DataFrame(override_rows)})
    effective, report = build_effective_cadastro(dadger, hidr, _calendar())

    assert effective.cota_polynomial(1, 0) == (100.0, 1.0, 0.0, 0.0, 0.0)
    assert effective.cota_polynomial(1, 1) == (100.0, 1.0, 0.0, 0.0, 0.0)
    assert effective.cota_polynomial(1, 2) == (50.0, 2.0, 0.0, 0.0, 0.0)
    assert report.applied["cota_volume"] == 1


def test_cotvol_single_constant_override_lands_in_a0() -> None:
    """Regression: a run-of-river plant whose ``ACCOTVOL`` override is a
    single constant forebay level (only ``ordem`` 1 present, the rest 0)
    must land that level in the a0 constant term, not a1. The real deck
    encodes fixed-level plants (JIRAU/BELO MONTE) exactly this way; treating
    the 1-based ``ordem`` as a 0-based slot turned a 90 m level into
    ``90·volume`` and inflated the head ~1000x.
    """
    hidr = _hidr_frame({1: _plant_row(cota=(66.35, 0.0207, -8e-6, 0.0, 0.0))})
    override_rows = [
        {
            "codigo_usina": 1,
            "ordem": ordem,
            "coeficiente": coeff,
            "mes": "AGO",
            "semana": 1.0,
            "ano": 2026.0,
        }
        for ordem, coeff in ((1, 90.0), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0))
    ]
    dadger = _FakeDadger({ACCOTVOL: pd.DataFrame(override_rows)})
    effective, _ = build_effective_cadastro(dadger, hidr, _calendar())

    # Constant polynomial: cota == 90.0 m at every volume, not 90·V.
    assert effective.cota_polynomial(1, 2) == (90.0, 0.0, 0.0, 0.0, 0.0)


def test_cotvol_ordem_out_of_range_raises() -> None:
    """A coefficient ``ordem`` outside the 1..5 range is a malformed register
    and is rejected, never silently dropped or mis-slotted.
    """
    rows = [
        {
            "codigo_usina": 1,
            "ordem": 0,  # 0 is out of the 1-based 1..5 range
            "coeficiente": 90.0,
            "mes": "AGO",
            "semana": 1.0,
            "ano": 2026.0,
        }
    ]
    dadger = _FakeDadger({ACCOTVOL: pd.DataFrame(rows)})
    with pytest.raises(ValueError, match="outside the expected 1..5 range"):
        _read_polynomial_overrides(dadger, _calendar(), ACCOTVOL, "cota_volume")


def test_cotvol_out_of_horizon_reported_not_dropped() -> None:
    """An ``ACCOTVOL`` group dated past the calendar horizon is reported —
    one entry per input row, never silently dropped.
    """
    rows = [
        {
            "codigo_usina": 1,
            "ordem": ordem,
            "coeficiente": float(ordem),
            "mes": "OUT",
            "semana": None,
            "ano": 2026.0,
        }
        for ordem in range(1, 6)
    ]
    dadger = _FakeDadger({ACCOTVOL: pd.DataFrame(rows)})
    records, out_of_horizon = _read_polynomial_overrides(
        dadger, _calendar(), ACCOTVOL, "cota_volume"
    )
    assert records == {}
    assert out_of_horizon == [OutOfHorizon(1, "cota_volume", 10, 2026)] * 5


# ---------------------------------------------------------------------------
# Per-stage rho_eq no longer caps generation (availability owns the
# generation cap; the head-corrected turbined-flow cap owns the physical
# hydraulic limit).
# ---------------------------------------------------------------------------


def test_per_stage_rho_eq_does_not_cap_generation() -> None:
    """A plant's ``PROESP`` doubles from stage 1 onward, so its per-stage
    ρ_eq doubles too — but that no longer lowers ``max_generation_mw``. The
    generation cap is the installed × MP × FD availability (head-independent),
    and the physical hydraulic limit binds through the head-corrected
    turbined-flow cap instead. With no MP/FD derate here the availability
    equals the rated envelope, so no lowered generation overlay is emitted at
    any stage.
    """
    hidr = _hidr_frame(
        {
            1: _plant_row(
                vmin=100.0,
                vmax=500.0,
                cota=(100.0, 0.0, 0.0, 0.0, 0.0),
                cf=20.0,
                rho_esp=0.01,
                n_conjuntos=1,
                n_machines=1,
                q_nom=100.0,
                p_nom=1000.0,
            )
        }
    )
    effective = EffectiveCadastro(
        base=hidr,
        n_stages=3,
        stage_varying={(1, "produtibilidade_especifica"): (0.01, 0.02, 0.02)},
    )
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))

    case = make_decomp_case(
        Path("unused"), dadger=_StubDadger(), hidr=hidr, calendar=_calendar()
    )
    values = convert_hydro_group_availability(case, id_map, effective=effective)
    hydro_id = id_map.hydro_id(1)

    for stage_index in range(3):
        entry = values.get((hydro_id, 0, stage_index))
        assert entry is None or entry.max_generation_mw is None


# ---------------------------------------------------------------------------
# Full-range head, no tipo_regulacao branch.
# ---------------------------------------------------------------------------


def test_rho_eq_uses_full_range_for_all_classes() -> None:
    """One ``M``, one ``S``, and one ``D`` plant, identical head inputs
    and a non-flat cota polynomial, all reduce to the SAME hand-computed
    full-range-mean formula — no ``tipo_regulacao``/``V_65`` branch.
    """
    coeffs = (100.0, 0.1, -0.0001, 0.0, 0.0)
    v_min, v_max, cf, rho_esp = 100.0, 500.0, 20.0, 0.009
    rows = {
        1: _plant_row(
            vmin=v_min,
            vmax=v_max,
            cota=coeffs,
            cf=cf,
            rho_esp=rho_esp,
            tipo_regulacao="M",
        ),
        2: _plant_row(
            vmin=v_min,
            vmax=v_max,
            cota=coeffs,
            cf=cf,
            rho_esp=rho_esp,
            tipo_regulacao="S",
        ),
        3: _plant_row(
            vmin=v_min,
            vmax=v_max,
            cota=coeffs,
            cf=cf,
            rho_esp=rho_esp,
            tipo_regulacao="D",
        ),
    }
    hidr = _hidr_frame(rows)
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1, 2, 3))
    effective = EffectiveCadastro(base=hidr, n_stages=1, stage_varying={})

    def antideriv(v: float) -> float:
        a0, a1, a2, a3, a4 = coeffs
        return (
            a0 * v
            + a1 * v * v / 2.0
            + a2 * v**3 / 3.0
            + a3 * v**4 / 4.0
            + a4 * v**5 / 5.0
        )

    mean_cota = (antideriv(v_max) - antideriv(v_min)) / (v_max - v_min)
    expected = rho_esp * (mean_cota - cf)

    table = convert_energy_productivity(effective, id_map)
    values_by_id = dict(
        zip(
            table["hydro_id"].to_pylist(),
            table["equivalent_productivity_mw_per_m3s"].to_pylist(),
            strict=True,
        )
    )
    for code in (1, 2, 3):
        assert values_by_id[id_map.hydro_id(code)] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Byte-identical when no head/productivity override is present.
# ---------------------------------------------------------------------------


def test_no_override_is_byte_identical() -> None:
    """With no head/productivity override at all, the emitted rho_eq
    equals the base-registry formula bit-for-bit — reconstructed
    here independently via the same shared coeffs-keyed primitives
    (``mean_cota``, ``apply_hydraulic_loss``) the production code now calls.
    The head no longer feeds a generation cap, so the B8 overlay emits no
    lowered ``max_generation_mw`` (availability, at the rated envelope here,
    owns that column).
    """
    hidr = _hidr_frame(
        {
            1: _plant_row(
                vmin=100.0,
                vmax=500.0,
                cota=(300.0, 0.0, 0.0, 0.0, 0.0),
                cf=20.0,
                rho_esp=0.01,
                n_conjuntos=1,
                n_machines=2,
                q_nom=50.0,
                p_nom=200.0,
            )
        }
    )
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    effective = EffectiveCadastro(base=hidr, n_stages=3, stage_varying={})

    # The base-registry formula, reconstructed verbatim off the base row.
    hreg = hidr.loc[1]
    v_min = float(hreg["volume_minimo"])
    v_max = float(hreg["volume_maximo"])
    rho_esp = float(hreg.get("produtibilidade_especifica", 0.0) or 0.0)
    cf = float(hreg.get("canal_fuga_medio", 0.0) or 0.0)
    tipo_perda = int(hreg.get("tipo_perda", 0) or 0)
    perdas = float(hreg.get("perdas", 0.0) or 0.0)
    coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]
    h_gross = mean_cota(coeffs, v_min, v_max) - cf
    h_net = max(apply_hydraulic_loss(h_gross, tipo_perda, perdas), 0.0)
    expected_rho_eq = rho_esp * h_net

    table = convert_energy_productivity(effective, id_map).to_pandas()
    assert table["equivalent_productivity_mw_per_m3s"].iloc[0] == expected_rho_eq

    case = make_decomp_case(
        Path("unused"), dadger=_StubDadger(), hidr=hidr, calendar=_calendar()
    )
    values = convert_hydro_group_availability(case, id_map, effective=effective)
    hydro_id = id_map.hydro_id(1)
    for stage_index in range(3):
        entry = values.get((hydro_id, 0, stage_index))
        assert entry is None or entry.max_generation_mw is None


# ---------------------------------------------------------------------------
# All-zero cota coefficients (the shared-core guard gained by routing
# through productivity.equivalent_productivity_from_coeffs).
# ---------------------------------------------------------------------------


def test_all_zero_coeffs_plant_returns_zero_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A plant with no forebay curve at all (all-zero ``a0..a4_volume_cota``)
    yields ``ρ_eq == 0.0`` and logs a WARNING, rather than the pre-shared-core
    behaviour of silently evaluating a degenerate polynomial and clamping a
    meaningless head to zero.
    """
    hidr = _hidr_frame({1: _plant_row(cota=(0.0, 0.0, 0.0, 0.0, 0.0))})
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    effective = EffectiveCadastro(base=hidr, n_stages=1, stage_varying={})

    with caplog.at_level(logging.WARNING, logger="novomodelo_bridge.core.productivity"):
        table = convert_energy_productivity(effective, id_map).to_pandas()

    assert table["equivalent_productivity_mw_per_m3s"].iloc[0] == 0.0
    assert any(
        record.levelno == logging.WARNING and "zero" in record.getMessage().lower()
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# FPHA anchor: generation productivity takes head at the initial volume.
# ---------------------------------------------------------------------------


def test_generation_productivity_anchors_at_initial_volume() -> None:
    """The generation ρ_eq takes head at the plant's initial volume (the source
    model's FPHA fit anchor, manual §3.4.6.4), not the full-range mean. For a
    reservoir starting high on a rising cota curve, that is a higher head — and
    a higher productivity — than the mid-range mean would give.
    """
    vmin, vmax = 100.0, 500.0
    # Rising cota: 220 m at vmin, 300 m at vmax; mean over the range = 260 m.
    coeffs = (200.0, 0.2, 0.0, 0.0, 0.0)
    cf, rho_esp = 20.0, 0.01
    hidr = _hidr_frame(
        {1: _plant_row(vmin=vmin, vmax=vmax, cota=coeffs, cf=cf, rho_esp=rho_esp)}
    )
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    effective = EffectiveCadastro(base=hidr, n_stages=1, stage_varying={})
    hydro_id = id_map.hydro_id(1)

    v_initial = 460.0  # 90% full
    expected_init = rho_esp * ((coeffs[0] + coeffs[1] * v_initial) - cf)

    def antideriv(v: float) -> float:
        return coeffs[0] * v + coeffs[1] * v * v / 2.0

    mean_cota = (antideriv(vmax) - antideriv(vmin)) / (vmax - vmin)
    expected_mean = rho_esp * (mean_cota - cf)

    anchored = (
        convert_energy_productivity(effective, id_map, {1: v_initial})
        .to_pandas()
        .set_index("hydro_id")["equivalent_productivity_mw_per_m3s"][hydro_id]
    )
    # No map -> full-range mean fallback (the engolimento/legacy anchor).
    fallback = (
        convert_energy_productivity(effective, id_map)
        .to_pandas()
        .set_index("hydro_id")["equivalent_productivity_mw_per_m3s"][hydro_id]
    )

    assert anchored == pytest.approx(expected_init)
    assert fallback == pytest.approx(expected_mean)
    assert anchored > fallback  # rising cota + high fill => higher than the mean
