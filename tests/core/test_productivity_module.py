"""Tests for the extracted productivity domain module."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from novomodelo_bridge.core.productivity import (
    KTURB_BY_TIPO_TURBINA,
    compute_productivity,
    equivalent_productivity,
    equivalent_productivity_from_coeffs,
    evaluate_cota,
    fpha_efficiency,
    integrated_productivity,
    mean_cota,
    stored_energy_productivity,
)

# novomodelo's phi = K * eta * q * h_net with K = g/1000 (mirrors the module's
# private _GRAVITY_MW_FACTOR — pinned here rather than imported, per the D2
# unit contract in fpha_efficiency's docstring).
_K = 9.81e-3


def _hreg(**overrides: object) -> pd.Series:
    defaults: dict = {
        "nome_usina": "TEST",
        "produtibilidade_especifica": 0.009,
        "volume_minimo": 100.0,
        "volume_maximo": 1000.0,
        "volume_referencia": 500.0,
        "canal_fuga_medio": 250.0,
        "tipo_regulacao": "M",
        "tipo_perda": 1,
        "perdas": 5.0,
        "a0_volume_cota": 300.0,
        "a1_volume_cota": 0.1,
        "a2_volume_cota": 0.0,
        "a3_volume_cota": 0.0,
        "a4_volume_cota": 0.0,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


def test_hydro_aliases_point_to_public_functions() -> None:
    from novomodelo_bridge.newave.converters.hydro import (
        _compute_integrated_productivity,
        _compute_productivity,
        _equivalent_productivity,
    )

    assert _compute_productivity is compute_productivity
    assert _equivalent_productivity is equivalent_productivity
    assert _compute_integrated_productivity is integrated_productivity


def test_constraints_and_fict_cascade_import_public_names() -> None:
    # Cross-boundary callers must no longer reach into hydro privates.
    import novomodelo_bridge.newave.converters.constraints as c
    import novomodelo_bridge.newave.converters.fict_cascade as fc

    assert c.compute_productivity is compute_productivity
    assert c.stored_energy_productivity is stored_energy_productivity
    assert fc.compute_productivity is compute_productivity


def test_cmont_override_pins_forebay() -> None:
    # head = cmont - cfuga = 400 - 250 = 150; ·0.95 loss; ·0.009 ρ_esp.
    hreg = _hreg()
    assert compute_productivity(hreg, cmont_override=400.0) == pytest.approx(
        0.009 * (400.0 - 250.0) * 0.95
    )


def test_equivalent_vs_integrated_agree_on_linear_polynomial() -> None:
    # For a purely linear h(V) the mean head equals the value at the midpoint;
    # equivalent_productivity and integrated_productivity use the same integral.
    hreg = _hreg(tipo_perda=0)
    assert equivalent_productivity(hreg) == pytest.approx(integrated_productivity(hreg))


def test_equivalent_productivity_from_coeffs_all_zero_guard_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="novomodelo_bridge.core.productivity"):
        result = equivalent_productivity_from_coeffs(
            [0.0, 0.0, 0.0, 0.0, 0.0],
            volume_min_hm3=0.0,
            volume_max_hm3=100.0,
            rho_esp=0.01,
            canal_fuga_m=10.0,
            tipo_perda=0,
            perdas=0.0,
            plant_name="Z",
        )

    assert result == 0.0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings
    assert any(getattr(record, "plant", None) == "Z" for record in warnings)


def test_equivalent_productivity_from_coeffs_negative_head_clamped_to_zero() -> None:
    # coeffs=[10,0,0,0,0] -> constant cota=10; canal_fuga=50 puts the tailrace
    # above the forebay, so the unclamped net head would be -40.
    result = equivalent_productivity_from_coeffs(
        [10.0, 0.0, 0.0, 0.0, 0.0],
        volume_min_hm3=0.0,
        volume_max_hm3=100.0,
        rho_esp=0.01,
        canal_fuga_m=50.0,
        tipo_perda=0,
        perdas=0.0,
        plant_name="NEG",
    )
    assert result == 0.0
    assert result >= 0.0


def test_mean_cota_degenerate_range_equals_evaluate_cota() -> None:
    coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert mean_cota(coeffs, 5.0, 5.0) == evaluate_cota(coeffs, 5.0)


def test_mean_cota_linear_coeffs_equals_evaluate_cota_at_midpoint() -> None:
    coeffs = [10.0, 0.5, 0.0, 0.0, 0.0]
    v_lo, v_hi = 20.0, 80.0
    midpoint = (v_lo + v_hi) / 2.0
    assert mean_cota(coeffs, v_lo, v_hi) == pytest.approx(
        evaluate_cota(coeffs, midpoint)
    )


def test_equivalent_productivity_agrees_with_coeffs_core_on_positive_head() -> None:
    # Row adapter and coeffs core must agree exactly for a positive-head plant
    # (the clamp/guard the core adds are no-ops here).
    hreg = _hreg()
    coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]
    core_value = equivalent_productivity_from_coeffs(
        coeffs,
        volume_min_hm3=float(hreg["volume_minimo"]),
        volume_max_hm3=float(hreg["volume_maximo"]),
        rho_esp=float(hreg["produtibilidade_especifica"]),
        canal_fuga_m=float(hreg["canal_fuga_medio"]),
        tipo_perda=int(hreg["tipo_perda"]),
        perdas=float(hreg["perdas"]),
        plant_name=str(hreg["nome_usina"]),
    )
    assert core_value > 0.0
    assert equivalent_productivity(hreg) == pytest.approx(core_value)


class TestFphaEfficiency:
    def test_realistic_rho_esp_round_trips_to_rho_esp_over_k(self) -> None:
        rho_esp = 0.00892
        eta = fpha_efficiency(rho_esp, "USINA")
        assert eta == pytest.approx(rho_esp / _K)
        assert 0.0 < eta <= 1.0

    def test_unphysical_rho_esp_clamped_to_one(self) -> None:
        # rho_esp >> K would imply eta > 1; novomodelo requires (0, 1].
        assert fpha_efficiency(0.9, "USINA") == 1.0

    def test_clamp_warns_naming_the_plant(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(
            logging.WARNING, logger="novomodelo_bridge.core.productivity"
        ):
            fpha_efficiency(0.9, "CLAMPED PLANT")
        assert any("CLAMPED PLANT" in record.getMessage() for record in caplog.records)


class TestKturbByTipoTurbina:
    def test_maps_known_turbine_types(self) -> None:
        # 1 = Francis, 2 = Kaplan, 3 = Pelton; 0 = unspecified -> Francis.
        assert KTURB_BY_TIPO_TURBINA[0] == 0.5
        assert KTURB_BY_TIPO_TURBINA[1] == 0.5
        assert KTURB_BY_TIPO_TURBINA[2] == 0.2
        assert KTURB_BY_TIPO_TURBINA[3] == 0.5

    def test_francis_and_pelton_share_the_square_root_exponent(self) -> None:
        assert KTURB_BY_TIPO_TURBINA[1] == KTURB_BY_TIPO_TURBINA[3]
