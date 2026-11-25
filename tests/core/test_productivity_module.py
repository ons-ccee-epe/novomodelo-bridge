"""Tests for the productivity domain module."""

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
    forebay_curve_missing_diagnostic,
    fpha_efficiency,
    fpha_efficiency_capped_diagnostic,
    fpha_efficiency_capped_row,
    fpha_spillage_fit_max_m3s,
    has_forebay_curve,
    hydraulic_generation_mw,
    integrated_productivity,
    mean_cota,
    stored_energy_productivity,
)

# novomodelo's phi = K * eta * q * h_net with K = g/1000 (mirrors the module's
# private _GRAVITY_MW_FACTOR — pinned here rather than imported, per the
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


def test_constraints_and_fict_cascade_import_public_names() -> None:
    # Cross-boundary callers import the public names, not hydro privates.
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


def test_equivalent_productivity_from_coeffs_all_zero_is_zero_without_a_log(
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
        )

    assert result == 0.0
    assert caplog.records == []


def test_forebay_curve_is_missing_only_when_every_coefficient_is_zero() -> None:
    assert has_forebay_curve([0.0, 0.0, 0.0, 0.0, 0.0]) is False
    assert has_forebay_curve([0.0, 0.0, 1e-9, 0.0, 0.0]) is True


def test_forebay_curve_missing_diagnostic_counts_the_listed_plants() -> None:
    diagnostic = forebay_curve_missing_diagnostic([[1, "A", "0-3"], [2, "B", "5"]])
    assert diagnostic.code == "hydro-forebay-curve-missing"
    assert diagnostic.title == "Plants without a forebay curve (2)"
    assert diagnostic.table is not None
    assert diagnostic.table.columns == ["Code", "Name", "Stages"]


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
    )
    assert result == 0.0
    assert result >= 0.0


def test_mean_cota_degenerate_range_equals_evaluate_cota() -> None:
    coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert mean_cota(coeffs, 5.0, 5.0) == evaluate_cota(coeffs, 5.0)


def test_mean_cota_rounding_wide_range_equals_evaluate_cota() -> None:
    # A real deck's forebay quartic over a range one ulp wide (a reservoir
    # simulated at its floor): the antiderivative difference cancels there.
    coeffs = [594.02, 0.35164, -2.1575e-3, 7.3589e-6, -9.5971e-9]
    v_lo = 228.27000427246094
    v_hi = 228.27000427246097
    assert v_hi > v_lo
    assert mean_cota(coeffs, v_lo, v_hi) == pytest.approx(
        evaluate_cota(coeffs, v_lo), abs=1e-9
    )


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
    )
    assert core_value > 0.0
    assert equivalent_productivity(hreg) == pytest.approx(core_value)


class TestFphaEfficiency:
    def test_realistic_rho_esp_round_trips_to_rho_esp_over_k(self) -> None:
        rho_esp = 0.00892
        eta = fpha_efficiency(rho_esp)
        assert eta == pytest.approx(rho_esp / _K)
        assert 0.0 < eta <= 1.0

    def test_unphysical_rho_esp_clamped_to_one(self) -> None:
        # rho_esp >> K would imply eta > 1; novomodelo requires (0, 1].
        assert fpha_efficiency(0.9) == 1.0

    def test_capped_row_names_the_plant_and_its_implied_efficiency(self) -> None:
        assert fpha_efficiency_capped_row(7, "CAPPED", 0.9) == [
            7,
            "CAPPED",
            round(0.9 / _K, 4),
        ]

    def test_realistic_rho_esp_has_no_capped_row(self) -> None:
        assert fpha_efficiency_capped_row(7, "FINE", 0.00892) is None

    def test_capped_diagnostic_is_a_warning_with_one_row_per_plant(self) -> None:
        rows = [[7, "A", 91.7], [9, "B", 1.02]]
        diagnostic = fpha_efficiency_capped_diagnostic(rows)
        assert diagnostic.code == "fpha-efficiency-capped"
        assert diagnostic.severity.name == "WARNING"
        assert diagnostic.table is not None
        assert diagnostic.table.rows == rows


class TestFphaSpillageFitMax:
    def test_is_twice_the_largest_monthly_mean(self) -> None:
        means = [900.0, 21141.0, 18000.0] + [500.0] * 9
        assert fpha_spillage_fit_max_m3s(288, means) == 42282.0

    def test_artificial_gauge_has_none_whatever_its_means(self) -> None:
        assert fpha_spillage_fit_max_m3s(300, [100.0] * 12) is None

    def test_gauge_without_a_positive_mean_has_none(self) -> None:
        assert fpha_spillage_fit_max_m3s(12, [0.0] * 12) is None
        assert fpha_spillage_fit_max_m3s(12, []) is None


class TestKturbByTipoTurbina:
    def test_maps_known_turbine_types(self) -> None:
        # 1 = Francis, 2 = Kaplan, 3 = Pelton; 0 = unspecified -> Francis.
        assert KTURB_BY_TIPO_TURBINA[0] == 0.5
        assert KTURB_BY_TIPO_TURBINA[1] == 0.5
        assert KTURB_BY_TIPO_TURBINA[2] == 0.2
        assert KTURB_BY_TIPO_TURBINA[3] == 0.5

    def test_francis_and_pelton_share_the_square_root_exponent(self) -> None:
        assert KTURB_BY_TIPO_TURBINA[1] == KTURB_BY_TIPO_TURBINA[3]


class TestHydraulicGeneration:
    def test_specific_productivity_times_flow_times_net_head(self) -> None:
        generation = hydraulic_generation_mw(0.009, 100.0, 50.0, 1000.0)
        assert generation == pytest.approx(45.0)

    def test_capped_at_the_generation_limit(self) -> None:
        assert hydraulic_generation_mw(0.009, 100.0, 50.0, 40.0) == 40.0

    def test_non_positive_net_head_generates_nothing(self) -> None:
        assert hydraulic_generation_mw(0.009, 100.0, -3.0, 1000.0) == 0.0


class TestIntegratedProductivity:
    """The EARM-flavored *integrated* productivity used by VminOP.

    The closed-form integral for ``h(V) = a0 + a1·V`` over [vmin, vmax] is
    ``a0 + a1·(vmin+vmax)/2`` — the polynomial average over the range.
    """

    @staticmethod
    def _plant(**overrides: object) -> pd.Series:
        return _hreg(
            produtibilidade_especifica=0.01,
            canal_fuga_medio=200.0,
            tipo_perda=0,
            perdas=0.0,
            **overrides,
        )

    def test_closed_form_average_for_linear_polynomial(self) -> None:
        # h(V) = 300 + 0.1·V over [100, 1000] averages 300 + 0.1·550 = 355;
        # drop = 355 - cf=200 = 155; ρ = 0.01 · 155 = 1.55.
        assert integrated_productivity(self._plant()) == pytest.approx(
            0.01 * (300.0 + 0.1 * 550.0 - 200.0)
        )

    def test_canal_fuga_override(self) -> None:
        result = integrated_productivity(self._plant(), canal_fuga_override=180.0)
        # Same average h=355, new drop = 355 - 180 = 175.
        assert result == pytest.approx(0.01 * (355.0 - 180.0))

    def test_cmont_override_collapses_to_constant_drop(self) -> None:
        # CMONT supplies the upstream level directly; the polynomial is
        # bypassed. drop = cmont - canal_fuga_medio = 360 - 200 = 160.
        result = integrated_productivity(self._plant(), cmont_override=360.0)
        assert result == pytest.approx(0.01 * (360.0 - 200.0))

    def test_run_of_river_uses_point_value(self) -> None:
        # Collapse the plant to run-of-river by setting vmax = vmin.
        result = integrated_productivity(self._plant(volume_maximo=100.0))
        # h(100) = 300 + 0.1·100 = 310. Drop = 310 - 200 = 110.
        assert result == pytest.approx(0.01 * (310.0 - 200.0))

    def test_quadratic_polynomial_uses_correct_antiderivative(self) -> None:
        """For h(V) = a0 + a1·V + a2·V² the integrated average over
        [vmin, vmax] is a0 + a1·(vmin+vmax)/2 + a2·(vmax³−vmin³)/(3·(vmax−vmin)).
        Verify the closed-form matches that formula on a quadratic case."""
        vmin, vmax = 100.0, 1000.0
        a0, a1, a2 = 300.0, 0.1, 1e-5
        expected_avg_h = (
            a0
            + a1 * (vmin + vmax) / 2.0
            + a2 * (vmax**3 - vmin**3) / (3.0 * (vmax - vmin))
        )
        result = integrated_productivity(self._plant(a2_volume_cota=a2))
        assert result == pytest.approx(0.01 * (expected_avg_h - 200.0))


class TestComputeProductivity:
    """Unit tests for the ``compute_productivity`` helper."""

    def test_monthly_regulated_linear_polynomial(self) -> None:
        """tipo_regulacao='M': poly evaluated at 65% useful storage (the source model
        ``produtibilidade_altura_65`` convention)."""
        hreg = _hreg(
            **{
                "tipo_regulacao": "M",
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
                "volume_minimo": 100.0,
                "volume_maximo": 1000.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 1,
                "perdas": 5.0,  # 5%
                "produtibilidade_especifica": 0.009,
            }
        )
        # 65% of useful storage: v_65 = 100 + 0.65 * (1000 - 100) = 685.0
        # poly(685) = 300 + 0.1 * 685 = 368.5
        # net_drop = 368.5 - 250.0 = 118.5
        # adjusted_drop = 118.5 * (1 - 5.0/100) = 112.575
        # result = 0.009 * 112.575 = 1.013175
        v_65_height = 300.0 + 0.1 * (100.0 + 0.65 * (1000.0 - 100.0))
        expected = 0.009 * (1.0 - 5.0 / 100.0) * (v_65_height - 250.0)
        result = compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_run_of_river_point_evaluation(self) -> None:
        """tipo_regulacao='D': evaluates poly at volume_referencia."""
        hreg = _hreg(
            **{
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 1,
                "perdas": 5.0,  # 5%
                "produtibilidade_especifica": 0.009,
            }
        )
        # poly(500) = 300 + 0.1*500 = 350.0
        # net_drop = 350.0 - 250.0 = 100.0
        # adjusted_drop = 100.0 * (1 - 5.0/100) = 95.0
        # result = 0.009 * 95.0 = 0.855
        expected = 0.009 * (1.0 - 5.0 / 100.0) * (350.0 - 250.0)
        result = compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_multiplicative_loss(self) -> None:
        """tipo_perda=1: adjusted_drop = net_drop * (1 - perdas/100)."""
        hreg = _hreg(
            **{
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 1,
                "perdas": 10.0,  # 10%
                "produtibilidade_especifica": 0.009,
            }
        )
        # net_drop = (300 + 50) - 250 = 100.0
        # adjusted = 100.0 * (1 - 10.0/100) = 90.0
        expected = 0.009 * 90.0
        result = compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_additive_loss(self) -> None:
        """tipo_perda=2: adjusted_drop = net_drop - perdas."""
        hreg = _hreg(
            **{
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 2,
                "perdas": 3.5,
                "produtibilidade_especifica": 0.009,
            }
        )
        # net_drop = 350.0 - 250.0 = 100.0
        # adjusted = 100.0 - 3.5 = 96.5
        expected = 0.009 * 96.5
        result = compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_no_loss(self) -> None:
        """tipo_perda=0 (or unknown): no loss applied, adjusted_drop = net_drop."""
        hreg = _hreg(
            **{
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 0,
                "perdas": 99.0,
                "produtibilidade_especifica": 0.009,
            }
        )
        # tipo_perda=0 -> no loss applied, perdas value ignored
        # net_drop = 350.0 - 250.0 = 100.0
        expected = 0.009 * 100.0
        result = compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_equal_volumes_fallback(self) -> None:
        """tipo_regulacao='M' with vmin == vmax: v_65 collapses to that point."""
        hreg = _hreg(
            **{
                "tipo_regulacao": "M",
                "volume_minimo": 500.0,
                "volume_maximo": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 1,
                "perdas": 0.0,
                "produtibilidade_especifica": 0.009,
            }
        )
        # vmin == vmax: v_65 = 500.0; poly(500) = 350.0; net_drop = 100.0
        expected = 0.009 * 100.0
        result = compute_productivity(hreg)
        assert result == pytest.approx(expected)


class TestComputeProductivityOverrides:
    """Unit tests for ``compute_productivity`` with canal_fuga/cmont overrides."""

    def test_canal_fuga_override_replaces_base(self) -> None:
        """canal_fuga_override replaces canal_fuga_medio in the net drop calc."""
        hreg = _hreg(
            **{
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 0,
                "perdas": 0.0,
                "produtibilidade_especifica": 0.009,
            }
        )
        # poly(500) = 300 + 50 = 350
        # With override canal_fuga=260: net_drop = 350 - 260 = 90
        base = compute_productivity(hreg)  # uses 250 -> drop 100
        overridden = compute_productivity(hreg, canal_fuga_override=260.0)
        assert base == pytest.approx(0.009 * 100.0)
        assert overridden == pytest.approx(0.009 * 90.0)

    def test_cmont_override_replaces_polynomial_height(self) -> None:
        """cmont_override bypasses the polynomial and uses the supplied height."""
        hreg = _hreg(
            **{
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 0,
                "perdas": 0.0,
                "produtibilidade_especifica": 0.009,
            }
        )
        # cmont=380 overrides polynomial; net_drop = 380 - 250 = 130
        result = compute_productivity(hreg, cmont_override=380.0)
        assert result == pytest.approx(0.009 * 130.0)

    def test_both_overrides_together(self) -> None:
        """canal_fuga_override and cmont_override can both be active."""
        hreg = _hreg(
            **{
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 0,
                "perdas": 0.0,
                "produtibilidade_especifica": 0.009,
            }
        )
        # cmont=400, canal_fuga=260 -> net_drop = 400 - 260 = 140
        result = compute_productivity(
            hreg, canal_fuga_override=260.0, cmont_override=400.0
        )
        assert result == pytest.approx(0.009 * 140.0)

    def test_no_overrides_matches_original_behaviour(self) -> None:
        """With no overrides, M-plant ρ comes from poly evaluated at 65% storage."""
        hreg = _hreg(
            **{
                "tipo_regulacao": "M",
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "volume_minimo": 100.0,
                "volume_maximo": 1000.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 1,
                "perdas": 5.0,  # 5%
                "produtibilidade_especifica": 0.009,
            }
        )
        # v_65 = 100 + 0.65 * 900 = 685; poly(685) = 368.5; net_drop = 118.5
        v_65_height = 300.0 + 0.1 * (100.0 + 0.65 * (1000.0 - 100.0))
        expected = 0.009 * (1.0 - 5.0 / 100.0) * (v_65_height - 250.0)
        assert compute_productivity(hreg) == pytest.approx(expected)


class TestEquivalentProductivity:
    """Unit tests for ``equivalent_productivity`` (the source model PRODT)."""

    def test_linear_polynomial_uses_mean_head(self) -> None:
        """h(v)=a0+a1·v → mean head over [vmin,vmax] = a0 + a1·(vmin+vmax)/2."""
        hreg = _hreg()  # a0=300, a1=0.1, vmin=100, vmax=1000, cfuga=250
        # mean head = 300 + 0.1·550 = 355 ; net = 105 ; ·0.95 loss ; ·0.009 pesp
        assert equivalent_productivity(hreg) == pytest.approx(
            0.009 * (355.0 - 250.0) * 0.95
        )

    def test_differs_from_point_reference(self) -> None:
        """PRODT (mean over range) ≠ the 65%-volume point productivity."""
        # Non-linear forebay curve so the average over [vmin,vmax] differs from
        # the value at the 65% reference point.
        hreg = _hreg(**{"a2_volume_cota": 1e-4})
        assert equivalent_productivity(hreg) != pytest.approx(
            compute_productivity(hreg)
        )

    def test_run_of_river_uses_point_head(self) -> None:
        """Vmax == Vmin → head evaluated at Vmin (no integral)."""
        hreg = _hreg(**{"volume_minimo": 500.0, "volume_maximo": 500.0})
        # h(500) = 300 + 0.1·500 = 350 ; net 100 ; ·0.95 ; ·0.009
        assert equivalent_productivity(hreg) == pytest.approx(
            0.009 * (350.0 - 250.0) * 0.95
        )

    def test_canal_fuga_override(self) -> None:
        hreg = _hreg()
        assert equivalent_productivity(
            hreg, canal_fuga_override=300.0
        ) == pytest.approx(0.009 * (355.0 - 300.0) * 0.95)

    def test_cmont_override_pins_forebay(self) -> None:
        """CMONT pins the upstream level → head = cmont − cfuga (no integral)."""
        hreg = _hreg()
        assert equivalent_productivity(hreg, cmont_override=400.0) == pytest.approx(
            0.009 * (400.0 - 250.0) * 0.95
        )

    def test_zero_polynomial_returns_zero(self) -> None:
        hreg = _hreg(**{f"a{i}_volume_cota": 0.0 for i in range(5)})
        assert equivalent_productivity(hreg) == 0.0
