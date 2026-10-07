"""Unit tests for the VminOP, electric, and AGRINT generic constraints converters."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity, finalize_diagnostics
from novomodelo_bridge.core.generic_constraint_builder import (
    GENERIC_BOUNDS_SCHEMA,
    ConstraintIdAllocator,
)
from novomodelo_bridge.core.generic_constraint_format import GENERIC_BOUNDS_COLUMNS
from novomodelo_bridge.core.units import C_M3S2HM3
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.converters.constraints import (
    _curve_seasonalizes,
    _ElectricTermSkip,
    _is_stored_energy_reservoir,
    _parse_formula,
    _parse_re_dat,
    _vminop_energy_factor,
    _warn_if_non_fixa_penalization,
    compute_accumulated_integrated_productivities,
    compute_accumulated_productivities,
    convert_agrint_constraints,
    convert_electric_constraints,
    convert_vminop_constraints,
)
from novomodelo_bridge.newave.id_map import NewaveIdMap
from novomodelo_bridge.novomodelo.scalar_parameters import build_scalar_parameters
from tests.conftest import make_case, make_nw_files

# The remediation/summary/title/notes strings constraints.py emits reach a
# pip-installed user with no repo checkout — none may leak a repo-internal
# reference (mirrors test_decomp_fcf_capability.py's own marker scan, plus the
# two extra markers the migration calls out).
_REPO_INTERNAL_LEAKS = (
    "docs/",
    "plans/",
    "~/git",
    "feat/",
    "ticket-",
    "epic-",
    "src/",
    ".py",
)


def _assert_no_repo_internal_leaks(collected: list[dx.Diagnostic]) -> None:
    for diag in collected:
        strings = [diag.title, diag.summary, *diag.notes]
        if diag.remediation is not None:
            strings.append(diag.remediation)
        for s in strings:
            for leak in _REPO_INTERNAL_LEAKS:
                assert leak not in s, f"diagnostic {diag.code!r} leaks {leak!r}: {s!r}"


def _make_cadastro() -> pd.DataFrame:
    """Three-plant cascade: 1 -> 2 -> 3 (3 is the sink).

    Plant 3 has downstream=None (sea).  All have tipo_regulacao="D" and
    simple linear volume_cota polynomials for deterministic productivity.
    """
    data = {
        "nome_usina": ["PLANT_A", "PLANT_B", "PLANT_C"],
        "produtibilidade_especifica": [0.01, 0.02, 0.03],
        "volume_minimo": [100.0, 200.0, 300.0],
        "volume_maximo": [1000.0, 2000.0, 3000.0],
        "volume_referencia": [500.0, 1000.0, 1500.0],
        "canal_fuga_medio": [200.0, 300.0, 400.0],
        "tipo_perda": [0, 0, 0],
        "perdas": [0.0, 0.0, 0.0],
        "tipo_regulacao": ["D", "D", "D"],
        "a0_volume_cota": [300.0, 400.0, 500.0],
        "a1_volume_cota": [0.1, 0.05, 0.02],
        "a2_volume_cota": [0.0, 0.0, 0.0],
        "a3_volume_cota": [0.0, 0.0, 0.0],
        "a4_volume_cota": [0.0, 0.0, 0.0],
    }
    return pd.DataFrame(data, index=pd.Index([1, 2, 3], name="codigo_usina"))


def _make_confhd_df() -> pd.DataFrame:
    """Confhd with 3 plants: 1->2, 2->3, 3->None, all REE=1."""
    return pd.DataFrame(
        {
            "codigo_usina": [1, 2, 3],
            "nome_usina": ["PLANT_A", "PLANT_B", "PLANT_C"],
            "usina_existente": ["EX", "EX", "EX"],
            "codigo_usina_jusante": [2, 3, 0],
            "ree": [1, 1, 1],
            "posto": [1, 2, 3],
            "volume_inicial_percentual": [50.0, 50.0, 50.0],
        }
    )


class TestStoredEnergyReservoirFilter:
    """VminOP includes only the source model's EARM plant set: monthly-regulating
    reservoirs with usable storage (matching pmo.dat's
    ``produtibilidade_acumulada_calculo_earm``)."""

    @staticmethod
    def _cad(tipo: str, vmin: float, vmax: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "tipo_regulacao": [tipo],
                "volume_minimo": [vmin],
                "volume_maximo": [vmax],
            },
            index=pd.Index([42], name="codigo_usina"),
        )

    def test_monthly_reservoir_with_useful_volume_included(self) -> None:
        assert _is_stored_energy_reservoir(self._cad("M", 100.0, 200.0), 42) is True

    def test_run_of_river_excluded_even_with_useful_volume(self) -> None:
        # JIRAU is 'D' with a large operative volume range but is run-of-river.
        assert _is_stored_energy_reservoir(self._cad("D", 1249.8, 2746.7), 42) is False

    def test_special_regime_excluded_even_with_useful_volume(self) -> None:
        # ITAIPU is 'S' with ~1709 hm3 of useful volume yet not in EARM.
        assert (
            _is_stored_energy_reservoir(self._cad("S", 27695.2, 29403.9), 42) is False
        )

    def test_monthly_without_useful_volume_excluded(self) -> None:
        assert _is_stored_energy_reservoir(self._cad("M", 100.0, 100.0), 42) is False

    def test_unknown_code_excluded(self) -> None:
        assert _is_stored_energy_reservoir(self._cad("M", 100.0, 200.0), 999) is False


class TestNonFixaPenalizationWarning:
    """curva.dat TIPO DE PENALIZACAO = 0 (FIXA) matches novomodelo-bridge's VminOP
    modelling and needs no warning; a non-FIXA (iterative/variable) penalization is
    not reproduced, so the conversion warns that a VminOP-penalty difference is
    expected."""

    def test_fixa_does_not_warn(self, caplog) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            fired = _warn_if_non_fixa_penalization([0, 11, 1])
        assert fired is False
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_fixa_logs_info_confirmation(self, caplog) -> None:
        import logging

        with caplog.at_level(logging.INFO):
            _warn_if_non_fixa_penalization([0, 11, 1])
        assert "FIXA" in caplog.text
        assert "matches" in caplog.text

    def test_non_fixa_emits_warning(self, caplog) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            fired = _warn_if_non_fixa_penalization([1, 11, 1])
        assert fired is True
        assert "non-FIXA" in caplog.text

    def test_none_does_not_warn(self) -> None:
        assert _warn_if_non_fixa_penalization(None) is False

    def test_empty_does_not_warn(self) -> None:
        assert _warn_if_non_fixa_penalization([]) is False

    def test_malformed_first_field_does_not_warn(self) -> None:
        assert _warn_if_non_fixa_penalization(["x"]) is False

    def test_non_fixa_emits_structured_warning_diagnostic(self) -> None:
        with dx.collect() as collected:
            fired = _warn_if_non_fixa_penalization([1, 11, 1])

        assert fired is True
        assert len(collected) == 1
        diag = collected[0]
        assert diag.code == "vminop-penalization-not-fixa"
        assert diag.severity is Severity.WARNING
        assert diag.category == "Special constraints"
        assert "non-FIXA" in diag.summary

    def test_fixa_emits_structured_info_diagnostic(self) -> None:
        with dx.collect() as collected:
            fired = _warn_if_non_fixa_penalization([0, 11, 1])

        assert fired is False
        assert len(collected) == 1
        diag = collected[0]
        assert diag.code == "vminop-penalization-fixa"
        assert diag.severity is Severity.INFO
        assert diag.category == "Special constraints"
        assert "FIXA" in diag.summary
        assert "matches" in diag.summary

    def test_non_fixa_no_sink_fallback_logs_one_warning(self, caplog) -> None:
        """With no active collect() sink, emit() degrades to a single logging
        record — the pre-migration caplog contract keeps working."""
        import logging

        with caplog.at_level(logging.WARNING):
            _warn_if_non_fixa_penalization([1, 11, 1])

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "non-FIXA" in warnings[0].message


class TestCurveSeasonalizes:
    """curva.dat's third penalization field selects post-study seasonalization
    of the security curve; absence/garbage defaults to freeze (manual p.32-33)."""

    def test_flag_set_seasonalizes(self) -> None:
        assert _curve_seasonalizes([0, 11, 1]) is True

    def test_flag_clear_freezes(self) -> None:
        assert _curve_seasonalizes([0, 11, 0]) is False

    def test_missing_third_field_freezes(self) -> None:
        assert _curve_seasonalizes([0, 11]) is False

    def test_none_values_freeze(self) -> None:
        assert _curve_seasonalizes([None, None, None]) is False

    def test_none_and_empty_freeze(self) -> None:
        assert _curve_seasonalizes(None) is False
        assert _curve_seasonalizes([]) is False


class TestAccumulatedProductivity:
    def test_sink_plant_has_own_productivity_only(self) -> None:
        cadastro = _make_cadastro()
        confhd_df = _make_confhd_df()
        acc = compute_accumulated_productivities(cadastro, confhd_df)

        # Plant 3 is the sink (downstream=None).
        # Its own productivity: 0.03 * (500 + 0.02*1500 - 400) = 0.03 * 130 = 3.9
        own_3 = 0.03 * ((500.0 + 0.02 * 1500.0) - 400.0)
        assert acc[3] == pytest.approx(own_3)

    def test_middle_plant_accumulates_downstream(self) -> None:
        cadastro = _make_cadastro()
        confhd_df = _make_confhd_df()
        acc = compute_accumulated_productivities(cadastro, confhd_df)

        own_3 = 0.03 * ((500.0 + 0.02 * 1500.0) - 400.0)
        own_2 = 0.02 * ((400.0 + 0.05 * 1000.0) - 300.0)
        assert acc[2] == pytest.approx(own_2 + own_3)

    def test_upstream_plant_accumulates_full_chain(self) -> None:
        cadastro = _make_cadastro()
        confhd_df = _make_confhd_df()
        acc = compute_accumulated_productivities(cadastro, confhd_df)

        own_3 = 0.03 * ((500.0 + 0.02 * 1500.0) - 400.0)
        own_2 = 0.02 * ((400.0 + 0.05 * 1000.0) - 300.0)
        own_1 = 0.01 * ((300.0 + 0.1 * 500.0) - 200.0)
        assert acc[1] == pytest.approx(own_1 + own_2 + own_3)

    def test_independent_plants_no_accumulation(self) -> None:
        """Two plants with no cascade link each have only own productivity."""
        cadastro = _make_cadastro()
        confhd_df = _make_confhd_df().copy()
        # Break the cascade: all plants have downstream=0
        confhd_df["codigo_usina_jusante"] = [0, 0, 0]

        acc = compute_accumulated_productivities(cadastro, confhd_df)

        own_1 = 0.01 * ((300.0 + 0.1 * 500.0) - 200.0)
        own_2 = 0.02 * ((400.0 + 0.05 * 1000.0) - 300.0)
        own_3 = 0.03 * ((500.0 + 0.02 * 1500.0) - 400.0)
        assert acc[1] == pytest.approx(own_1)
        assert acc[2] == pytest.approx(own_2)
        assert acc[3] == pytest.approx(own_3)


class TestIntegratedProductivity:
    """Tests for the EARM-flavored *integrated* productivity used by VminOP.

    The closed-form integral for ``h(V) = a0 + a1·V`` over [vmin, vmax] is
    ``a0 + a1·(vmin+vmax)/2`` — the polynomial average over the range.
    """

    def test_closed_form_average_for_linear_polynomial(self) -> None:
        from novomodelo_bridge.newave.converters.hydro import (
            _compute_integrated_productivity,
        )

        cadastro = _make_cadastro()
        # PLANT_A: h(V) = 300 + 0.1·V. vmin=100, vmax=1000. Average h over
        # [100, 1000] = 300 + 0.1·550 = 355. Drop = 355 - cf=200 = 155.
        # ρ = ρ_esp · 155 = 0.01 · 155 = 1.55.
        result = _compute_integrated_productivity(cadastro.loc[1])
        assert result == pytest.approx(0.01 * (300.0 + 0.1 * 550.0 - 200.0))

    def test_canal_fuga_override(self) -> None:
        from novomodelo_bridge.newave.converters.hydro import (
            _compute_integrated_productivity,
        )

        cadastro = _make_cadastro()
        result = _compute_integrated_productivity(
            cadastro.loc[1], canal_fuga_override=180.0
        )
        # Same average h=355, new drop = 355 - 180 = 175.
        assert result == pytest.approx(0.01 * (355.0 - 180.0))

    def test_cmont_override_collapses_to_constant_drop(self) -> None:
        from novomodelo_bridge.newave.converters.hydro import (
            _compute_integrated_productivity,
        )

        cadastro = _make_cadastro()
        # CMONT supplies the upstream level directly; the polynomial is
        # bypassed. drop = cmont - canal_fuga_medio = 360 - 200 = 160.
        result = _compute_integrated_productivity(cadastro.loc[1], cmont_override=360.0)
        assert result == pytest.approx(0.01 * (360.0 - 200.0))

    def test_run_of_river_uses_point_value(self) -> None:
        from novomodelo_bridge.newave.converters.hydro import (
            _compute_integrated_productivity,
        )

        cadastro = _make_cadastro().copy()
        # Collapse PLANT_A to run-of-river by setting vmax = vmin.
        cadastro.loc[1, "volume_maximo"] = cadastro.loc[1, "volume_minimo"]
        result = _compute_integrated_productivity(cadastro.loc[1])
        # h(100) = 300 + 0.1·100 = 310. Drop = 310 - 200 = 110.
        assert result == pytest.approx(0.01 * (310.0 - 200.0))

    def test_quadratic_polynomial_uses_correct_antiderivative(self) -> None:
        """For h(V) = a0 + a1·V + a2·V² the integrated average over
        [vmin, vmax] is a0 + a1·(vmin+vmax)/2 + a2·(vmax³−vmin³)/(3·(vmax−vmin)).
        Verify the closed-form matches that formula on a quadratic case."""
        from novomodelo_bridge.newave.converters.hydro import (
            _compute_integrated_productivity,
        )

        cadastro = _make_cadastro().copy()
        cadastro.loc[1, "a2_volume_cota"] = 1e-5  # tiny quadratic term
        vmin, vmax = 100.0, 1000.0
        a0, a1, a2 = 300.0, 0.1, 1e-5
        expected_avg_h = (
            a0
            + a1 * (vmin + vmax) / 2.0
            + a2 * (vmax**3 - vmin**3) / (3.0 * (vmax - vmin))
        )
        result = _compute_integrated_productivity(cadastro.loc[1])
        assert result == pytest.approx(0.01 * (expected_avg_h - 200.0))


class TestAccumulatedIntegratedProductivities:
    """Cascade sum of integrated ρ — what VminOP uses for the rho_acum_h{id}
    override and the per-stage RHS."""

    def test_cascade_uses_integral_for_monthly_reservoir(self) -> None:
        # Monthly-regulating ('M') reservoirs use the volmin→volmax integral (the source
        # model's produtibilidade_equivalente_volmin_volmax), NOT h at
        # volume_referencia.
        cadastro = _make_cadastro()
        cadastro["tipo_regulacao"] = "M"
        confhd_df = _make_confhd_df()
        acc = compute_accumulated_integrated_productivities(cadastro, confhd_df)
        # PLANT_A: 0.01 · (300 + 0.1·550 − 200) = 0.01 · 155 = 1.55
        # PLANT_B: 0.02 · (400 + 0.05·(200+2000)/2 − 300) = 0.02 · 155 = 3.1
        # PLANT_C: 0.03 · (500 + 0.02·(300+3000)/2 − 400) = 0.03 · 133 = 3.99
        own_a = 0.01 * (300.0 + 0.1 * 550.0 - 200.0)
        own_b = 0.02 * (400.0 + 0.05 * (200.0 + 2000.0) / 2.0 - 300.0)
        own_c = 0.03 * (500.0 + 0.02 * (300.0 + 3000.0) / 2.0 - 400.0)
        assert acc[3] == pytest.approx(own_c)
        assert acc[2] == pytest.approx(own_b + own_c)
        assert acc[1] == pytest.approx(own_a + own_b + own_c)

    def test_cascade_uses_point_at_vref_for_run_of_river(self) -> None:
        # Run-of-river ('D') plants use the point productivity at volume_referencia,
        # matching the source model's EARM convention for them.
        cadastro = _make_cadastro()  # all tipo_regulacao == "D"
        confhd_df = _make_confhd_df()
        acc = compute_accumulated_integrated_productivities(cadastro, confhd_df)
        # PLANT_A @vref=500: 0.01 · (300 + 0.1·500 − 200) = 0.01 · 150 = 1.5
        # PLANT_B @vref=1000: 0.02 · (400 + 0.05·1000 − 300) = 0.02 · 150 = 3.0
        # PLANT_C @vref=1500: 0.03 · (500 + 0.02·1500 − 400) = 0.03 · 130 = 3.9
        own_a = 0.01 * (300.0 + 0.1 * 500.0 - 200.0)
        own_b = 0.02 * (400.0 + 0.05 * 1000.0 - 300.0)
        own_c = 0.03 * (500.0 + 0.02 * 1500.0 - 400.0)
        assert acc[3] == pytest.approx(own_c)
        assert acc[2] == pytest.approx(own_b + own_c)
        assert acc[1] == pytest.approx(own_a + own_b + own_c)


class TestBuildScalarParametersOverride:
    """build_scalar_parameters switches rho_acum_h{id} to kind=per_stage
    only for hydros present in the override map."""

    def test_no_overrides_emits_only_computed(self) -> None:
        d = build_scalar_parameters([0, 1])
        # 4 entries (rho_eq + rho_acum) × 2 plants; all computed.
        assert len(d["scalar_parameters"]) == 4
        assert all(e["kind"] == "computed" for e in d["scalar_parameters"])
        names = [e["name"] for e in d["scalar_parameters"]]
        assert names == ["rho_eq_h0", "rho_acum_h0", "rho_eq_h1", "rho_acum_h1"]

    def test_override_switches_only_rho_acum_to_per_stage(self) -> None:
        d = build_scalar_parameters(
            [0, 1, 2], rho_acum_per_stage_overrides={1: [1.5, 1.6, 1.7]}
        )
        by_name = {e["name"]: e for e in d["scalar_parameters"]}
        # Only plant 1's rho_acum becomes per_stage; everything else is computed.
        assert by_name["rho_acum_h1"]["kind"] == "per_stage"
        assert by_name["rho_acum_h1"]["values"] == [
            [0, 1.5],
            [1, 1.6],
            [2, 1.7],
        ]
        assert "computed_spec" not in by_name["rho_acum_h1"]
        # rho_eq_h1 is still computed — the override is only for ρ_acum.
        assert by_name["rho_eq_h1"]["kind"] == "computed"
        assert by_name["rho_eq_h1"]["computed_spec"]["tag"] == "equivalent_productivity"
        # Plants without overrides stay computed.
        assert by_name["rho_acum_h0"]["kind"] == "computed"
        assert by_name["rho_acum_h2"]["kind"] == "computed"

    def test_overrides_preserve_unique_ids(self) -> None:
        d = build_scalar_parameters(
            [0, 1], rho_acum_per_stage_overrides={0: [2.0], 1: [3.0]}
        )
        ids = [e["id"] for e in d["scalar_parameters"]]
        assert ids == sorted(set(ids))


class TestPerStageAccumulatedProductivities:
    """Per-stage cascade-sum of own ρ_eq for the VminOP RHS bound."""

    def test_no_overrides_yields_flat_lists(self) -> None:
        """Without CFUGA/CMONT inputs every plant has a constant ρ_acum across
        stages."""
        from novomodelo_bridge.newave.converters.constraints import (
            compute_per_stage_acc_productivities,
        )

        confhd_df = _make_confhd_df()
        # Three plants, two stages, all own ρ_eq constant in time.
        per_stage_own = {1: [0.5, 0.5], 2: [0.7, 0.7], 3: [0.9, 0.9]}
        acc = compute_per_stage_acc_productivities(confhd_df, per_stage_own)

        # Plant 3 is the sink → ρ_acum(3) = own(3) at every stage.
        assert acc[3] == pytest.approx([0.9, 0.9])
        # Plant 2 → ρ_acum(2) = own(2) + ρ_acum(3) = 1.6
        assert acc[2] == pytest.approx([1.6, 1.6])
        # Plant 1 → ρ_acum(1) = own(1) + ρ_acum(2) = 2.1
        assert acc[1] == pytest.approx([2.1, 2.1])

    def test_override_on_downstream_propagates_upstream_only_after_stage(
        self,
    ) -> None:
        """A CFUGA-style change in plant 3's own ρ_eq at stage 1 shifts every
        upstream plant's ρ_acum from that stage on. Stage 0 stays unchanged.
        """
        from novomodelo_bridge.newave.converters.constraints import (
            compute_per_stage_acc_productivities,
        )

        confhd_df = _make_confhd_df()
        # Plant 3 ρ_eq drops from 0.9 to 0.6 starting at stage 1.
        per_stage_own = {1: [0.5, 0.5], 2: [0.7, 0.7], 3: [0.9, 0.6]}
        acc = compute_per_stage_acc_productivities(confhd_df, per_stage_own)

        # Stage 0: unchanged baselines.
        assert acc[3][0] == pytest.approx(0.9)
        assert acc[2][0] == pytest.approx(1.6)
        assert acc[1][0] == pytest.approx(2.1)
        # Stage 1: ρ_acum shifts everywhere in plant 3's upstream cone.
        assert acc[3][1] == pytest.approx(0.6)
        assert acc[2][1] == pytest.approx(0.7 + 0.6)
        assert acc[1][1] == pytest.approx(0.5 + 0.7 + 0.6)

    def test_sibling_branch_unaffected_by_override(self) -> None:
        """Sibling branches stay untouched when a leaf in another branch shifts."""
        from novomodelo_bridge.newave.converters.constraints import (
            compute_per_stage_acc_productivities,
        )

        # Two independent branches: 1→2 (sink), 3→4 (sink). 1 and 3 are headwaters.
        confhd_df = pd.DataFrame(
            {
                "codigo_usina": [1, 2, 3, 4],
                "nome_usina": ["A", "B", "C", "D"],
                "usina_existente": ["EX", "EX", "EX", "EX"],
                "codigo_usina_jusante": [2, 0, 4, 0],
                "ree": [1, 1, 1, 1],
                "posto": [1, 2, 3, 4],
                "volume_inicial_percentual": [50.0, 50.0, 50.0, 50.0],
            }
        )
        # Branch 1→2: constant. Branch 3→4: plant 4 ρ_eq drops at stage 1.
        per_stage_own = {1: [0.5, 0.5], 2: [0.7, 0.7], 3: [0.4, 0.4], 4: [0.8, 0.5]}
        acc = compute_per_stage_acc_productivities(confhd_df, per_stage_own)

        # Branch 1→2 unchanged across stages.
        assert acc[1] == pytest.approx([1.2, 1.2])
        assert acc[2] == pytest.approx([0.7, 0.7])
        # Branch 3→4 shifts at stage 1.
        assert acc[3][0] == pytest.approx(1.2)
        assert acc[3][1] == pytest.approx(0.9)


class TestConvertVminopConstraints:
    def test_returns_none_when_curva_absent(self, tmp_path) -> None:
        case = make_case(tmp_path, curva=None)
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[], thermal_codes=[])
        assert convert_vminop_constraints(case, id_map) is None

    def test_returns_none_when_curva_aversao_zero(self, tmp_path) -> None:
        """dger.dat curva_aversao=0 means the source model disabled the risk-aversion
        curve; novomodelo-bridge must skip VminOP constraints even when curva.dat is present
        on disk."""
        mock_dger = MagicMock()
        mock_dger.curva_aversao = 0
        # curva.dat is present (non-None reader), but curva_aversao=0 disables it.
        case = make_case(
            make_nw_files(tmp_path, curva=tmp_path / "curva.dat"),
            curva=MagicMock(),
            dger=mock_dger,
        )
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[], thermal_codes=[])

        assert convert_vminop_constraints(case, id_map) is None


def _make_vminop_no_hydro_case(tmp_path: Path) -> tuple[NewaveCase, NewaveIdMap]:
    """A VminOP case where REE 1 has a curva.dat entry but zero hydro plants."""
    from datetime import datetime as _dt

    dger = MagicMock()
    dger.mes_inicio_estudo = 1
    dger.ano_inicio_estudo = 2020
    dger.num_anos_estudo = 1
    dger.num_anos_pos_estudo = 0
    dger.curva_aversao = 1

    curva = MagicMock()
    curva.curva_seguranca = pd.DataFrame(
        {"codigo_ree": [1], "data": [_dt(2020, 1, 1)], "valor": [50.0]}
    )
    curva.configuracoes_penalizacao = [0, 11, 1]
    curva.custos_penalidades = None

    confhd = MagicMock()
    confhd.usinas = pd.DataFrame(
        columns=[
            "codigo_usina",
            "nome_usina",
            "usina_existente",
            "codigo_usina_jusante",
            "ree",
            "posto",
            "volume_inicial_percentual",
        ]
    )

    hidr = MagicMock()
    hidr.cadastro = pd.DataFrame()

    ree = MagicMock()
    ree.rees = pd.DataFrame({"codigo": [1], "nome": ["SUDESTE"]})

    case = make_case(
        tmp_path, dger=dger, curva=curva, confhd=confhd, hidr=hidr, ree=ree
    )
    id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[], thermal_codes=[])
    return case, id_map


class TestConvertVminopConstraintsEmission:
    """Emission-shape coverage for the two loop-accumulated VminOP diagnostics."""

    def test_ree_with_no_hydro_plants_emits_not_expressible(
        self, tmp_path: Path
    ) -> None:
        case, id_map = _make_vminop_no_hydro_case(tmp_path)

        with dx.collect() as collected:
            result = convert_vminop_constraints(case, id_map)

        assert result is None  # no constraint survives -> builder.result() is None
        by_code = {d.code: d for d in collected}
        assert set(by_code) == {
            "vminop-penalization-fixa",
            "vminop-ree-not-expressible",
        }
        diag = by_code["vminop-ree-not-expressible"]
        assert diag.severity is Severity.WARNING
        assert diag.category == "Special constraints"
        assert diag.table is not None
        assert diag.table.columns == ["REE", "Name", "Reason"]
        assert len(diag.table.rows) == 1
        assert diag.table.rows[0][0] == 1
        assert diag.table.rows[0][1] == "SUDESTE"

        _assert_no_repo_internal_leaks(collected)

    def test_nonpositive_penalty_emits_clamp_diagnostic(self, tmp_path: Path) -> None:
        from datetime import datetime as _dt

        cadastro = _make_cadastro()
        cadastro["tipo_regulacao"] = "M"  # stored-energy reservoirs (EARM set)
        confhd_df = _make_confhd_df()

        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2020
        dger.num_anos_estudo = 1
        dger.num_anos_pos_estudo = 0
        dger.curva_aversao = 1

        curva = MagicMock()
        curva.curva_seguranca = pd.DataFrame(
            {"codigo_ree": [1], "data": [_dt(2020, 1, 1)], "valor": [50.0]}
        )
        curva.configuracoes_penalizacao = [0, 11, 1]
        curva.custos_penalidades = pd.DataFrame(
            {"codigo_ree": [1], "penalidade": [-5.0]}
        )

        confhd = MagicMock()
        confhd.usinas = confhd_df

        hidr = MagicMock()
        hidr.cadastro = cadastro

        ree = MagicMock()
        ree.rees = pd.DataFrame({"codigo": [1], "nome": ["SUDESTE"]})

        case = make_case(
            tmp_path, dger=dger, curva=curva, confhd=confhd, hidr=hidr, ree=ree
        )
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1, 2, 3], thermal_codes=[])

        with dx.collect() as collected:
            result = convert_vminop_constraints(case, id_map)

        assert result is not None
        by_code = {d.code: d for d in collected}
        assert "vminop-penalty-nonpositive" in by_code
        diag = by_code["vminop-penalty-nonpositive"]
        assert diag.severity is Severity.WARNING
        assert diag.category == "Special constraints"
        assert diag.table is not None
        assert diag.table.columns == ["REE", "Name", "Declared", "Applied"]
        assert diag.table.rows == [[1, "SUDESTE", -5.0, 1000.0]]

        _assert_no_repo_internal_leaks(collected)

        _assert_no_repo_internal_leaks(collected)


# ---------------------------------------------------------------------------
# _parse_formula unit tests
# ---------------------------------------------------------------------------


class TestParseFormula:
    """Unit tests for the formula parser used by electric constraints."""

    def _id_map(self) -> NewaveIdMap:
        # hydro codes 66, 204, 261, 284, 285, 287 with deterministic IDs
        return NewaveIdMap(
            subsystem_ids=[1, 2, 3, 4, 11],
            hydro_codes=[66, 204, 261, 284, 285, 287],
            thermal_codes=[],
        )

    def _line_map(self) -> dict[tuple[int, int], int]:
        # Mirrors the real data: canonical pairs sorted
        # (1,2)=0, (1,3)=1, (1,4)=2, (1,11)=3, (3,11)=4, (4,11)=5
        return {
            (1, 2): 0,
            (1, 3): 1,
            (1, 4): 2,
            (1, 11): 3,
            (3, 11): 4,
            (4, 11): 5,
        }

    def test_simple_ger_usih(self) -> None:
        """Single hydro generation term with implicit coefficient."""
        id_map = self._id_map()
        line_map = self._line_map()
        result = _parse_formula(
            "ger_usih(285) + ger_usih(287)", id_map, line_map, constraint_code=1
        )
        assert result is not None
        assert "hydro_generation(" in result
        # Both plants must appear; no coefficient prefix for coeff=1.0
        assert "* hydro_generation" not in result

    def test_explicit_coefficient_on_ger_usih(self) -> None:
        """Coefficient of 0.5 must appear in the expression."""
        id_map = self._id_map()
        line_map = self._line_map()
        result = _parse_formula("0.5ger_usih(66)", id_map, line_map, constraint_code=1)
        assert result is not None
        assert "0.5 * hydro_generation(" in result

    def test_ener_interc_canonical_direction(self) -> None:
        """ener_interc(A,B) with A<B (canonical) emits ``line_direct``.

        The source model's directional interchange is non-negative, so the converter
        chooses between ``line_direct`` and ``line_reverse`` based on the flow
        direction; the literal coefficient stays positive.
        """
        id_map = self._id_map()
        line_map = self._line_map()
        result = _parse_formula(
            "1.0ener_interc(1,2)", id_map, line_map, constraint_code=1
        )
        assert result is not None
        assert "line_direct(0)" in result
        assert "line_reverse" not in result
        assert "line_exchange" not in result

    def test_ener_interc_reversed_direction(self) -> None:
        """ener_interc(B,A) with B>A (reverse of canonical) emits ``line_reverse``."""
        id_map = self._id_map()
        line_map = self._line_map()
        result = _parse_formula(
            "1.0ener_interc(2,1)", id_map, line_map, constraint_code=1
        )
        assert result is not None
        assert "line_reverse(0)" in result
        # No legacy signed line_exchange in the new output.
        assert "line_exchange" not in result
        # Coefficient stays positive — direction is encoded by the variable.
        assert "- " not in result

    def test_unknown_hydro_code_skipped(self) -> None:
        """Unknown hydro code drops the term (and, when recorded, is captured
        as a ``hydro-unmapped`` skip naming the constraint and the code)."""
        id_map = self._id_map()
        line_map = self._line_map()
        skipped: list[_ElectricTermSkip] = []
        result = _parse_formula(
            "ger_usih(9999)", id_map, line_map, constraint_code=7, skipped=skipped
        )
        assert result is None  # all terms dropped => None
        assert len(skipped) == 1
        assert skipped[0].constraint_code == 7
        assert skipped[0].kind == "hydro-unmapped"
        assert skipped[0].source == "formula"
        assert skipped[0].hydro_code == 9999

    def test_unknown_hydro_code_not_recorded_by_default(self) -> None:
        """The default ``skipped=None`` records nothing — the seam that keeps
        a direct unit-test call unchanged."""
        id_map = self._id_map()
        line_map = self._line_map()
        result = _parse_formula("ger_usih(9999)", id_map, line_map, constraint_code=1)
        assert result is None

    def test_ener_interc_malformed_args_recorded(self) -> None:
        """``ener_interc`` with fewer than 2 args is dropped and recorded as
        ``malformed``, carrying the raw argument fragment."""
        id_map = self._id_map()
        line_map = self._line_map()
        skipped: list[_ElectricTermSkip] = []
        result = _parse_formula(
            "ener_interc(1)", id_map, line_map, constraint_code=3, skipped=skipped
        )
        assert result is None
        assert len(skipped) == 1
        assert skipped[0].kind == "malformed"
        assert skipped[0].raw == "1"

    def test_ener_interc_no_line_recorded(self) -> None:
        """An interchange pair with no matching novomodelo line is dropped and
        recorded as ``no-line``, naming both subsystems."""
        id_map = self._id_map()
        line_map = self._line_map()
        skipped: list[_ElectricTermSkip] = []
        result = _parse_formula(
            "ener_interc(2,3)", id_map, line_map, constraint_code=5, skipped=skipped
        )
        assert result is None
        assert len(skipped) == 1
        assert skipped[0].kind == "no-line"
        assert skipped[0].sys_from == 2
        assert skipped[0].sys_to == 3

    def test_mixed_formula(self) -> None:
        """Mixed ener_interc and ger_usih terms are both translated."""
        id_map = self._id_map()
        line_map = self._line_map()
        result = _parse_formula(
            "1.0ener_interc(2,1) + 0.5ger_usih(66)",
            id_map,
            line_map,
            constraint_code=1,
        )
        assert result is not None
        # ener_interc(2,1) → reverse direction → line_reverse
        assert "line_reverse(" in result
        assert "hydro_generation(" in result


# ---------------------------------------------------------------------------
# convert_electric_constraints integration tests
# ---------------------------------------------------------------------------


def _make_electric_re_case(tmp_path: Path) -> tuple[NewaveCase, NewaveIdMap]:
    """Return a ``(case, id_map)`` pair for a single-code electric constraint:
    ``indices.csv`` -> ``restricao-eletrica.csv`` declaring a two-sided
    ``[50, 200]`` limit on ``ger_usih(10)`` over 2020, with ``dger``/``sistema``
    mocked to a 1-year 2020 study carrying no exchange/deficit data.
    """
    indices = tmp_path / "indices.csv"
    indices.write_text(
        "RESTRICAO-ELETRICA-ESPECIAL;Descricao;restricao-eletrica.csv\n",
        encoding="latin-1",
    )
    re_path = tmp_path / "restricao-eletrica.csv"
    re_path.write_text(
        "RE;1;1.0ger_usih(10)\n"
        "RE-HORIZ-PER;1;2020/01;2020/12\n"
        "RE-LIM-FORM-PER-PAT;1;2020/01;2020/12;1;50.;200.\n",
        encoding="latin-1",
    )

    dger = MagicMock()
    dger.mes_inicio_estudo = 1
    dger.ano_inicio_estudo = 2020
    dger.num_anos_estudo = 1
    dger.num_anos_pos_estudo = 0

    sistema = MagicMock()
    sistema.limites_intercambio = None
    sistema.custo_deficit = None

    case = make_case(tmp_path, dger=dger, sistema=sistema, re_dat=None)
    id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[10], thermal_codes=[])
    return case, id_map


class TestConvertElectricConstraints:
    def test_switched_off_csv_is_ignored_and_reported(self, tmp_path: Path) -> None:
        case, id_map = _make_electric_re_case(tmp_path)
        case.dger.restricoes_eletricas_especiais = 0

        with dx.collect() as collected:
            assert convert_electric_constraints(case, id_map) is None

        assert [d.code for d in collected] == ["dger-switch-off"]
        assert "restricao-eletrica.csv" in collected[0].title

    def test_switched_off_re_dat_is_ignored_and_reported(self, tmp_path: Path) -> None:
        """With ``RESTRICOES ELETRICAS = 0`` a present re.dat is never parsed."""
        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2020
        dger.num_anos_estudo = 1
        dger.num_anos_pos_estudo = 0
        dger.restricoes_eletricas = 0
        case = make_case(make_nw_files(tmp_path, re_dat=tmp_path / "re.dat"), dger=dger)
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[10], thermal_codes=[])

        with dx.collect() as collected:
            assert convert_electric_constraints(case, id_map) is None

        assert [d.code for d in collected] == ["dger-switch-off"]
        assert "re.dat" in collected[0].title

    def test_returns_none_when_no_indices_csv(self, tmp_path: Path) -> None:
        """Return None gracefully when indices.csv is absent."""
        case = make_case(tmp_path, re_dat=None)
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[], thermal_codes=[])
        assert convert_electric_constraints(case, id_map) is None

    def test_returns_none_when_re_file_missing(self, tmp_path: Path) -> None:
        """Return None when indices.csv exists but points to a missing file."""
        indices = tmp_path / "indices.csv"
        indices.write_text(
            "RESTRICAO-ELETRICA-ESPECIAL; ; does-not-exist.csv\n", encoding="latin-1"
        )

        case = make_case(tmp_path, re_dat=None)
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[], thermal_codes=[])
        assert convert_electric_constraints(case, id_map) is None

    def test_two_sided_bound_yields_two_single_sided_f3_constraints(
        self, tmp_path: Path
    ) -> None:
        """A constraint code with both a lower and upper limit produces two
        F3 constraint objects sharing one expression/name (the reader's
        ``lim_inf``/``lim_sup`` are independent declarations, not one band —
        unlike the DECOMP ``_GenericBuilder``, which does collapse a genuine
        band into one id). Each keeps today's row semantics: one endpoint
        populated, the other null."""
        case, id_map = _make_electric_re_case(tmp_path)

        result = convert_electric_constraints(case, id_map)

        assert result is not None
        constraints, bounds = result
        assert len(constraints) == 2
        for c in constraints:
            assert set(c) == {"id", "name", "description", "expression", "slack"}
            assert c["name"] == "RE_1"

        rows = bounds.to_pylist()
        by_cid: dict[int, list[dict]] = {}
        for row in rows:
            by_cid.setdefault(row["constraint_id"], []).append(row)
        assert len(by_cid) == 2

        lower_rows = [
            r for rs in by_cid.values() for r in rs if r["bound_lower"] is not None
        ]
        upper_rows = [
            r for rs in by_cid.values() for r in rs if r["bound_upper"] is not None
        ]
        # Every row is single-sided: the opposite endpoint is null.
        assert all(r["bound_upper"] is None for r in lower_rows)
        assert all(r["bound_lower"] is None for r in upper_rows)
        assert {r["bound_lower"] for r in lower_rows} == {50.0}
        assert {r["bound_upper"] for r in upper_rows} == {200.0}
        # The two ids partition the rows: the lower-only id never carries an
        # upper-only row and vice versa.
        lower_ids = {r["constraint_id"] for r in lower_rows}
        upper_ids = {r["constraint_id"] for r in upper_rows}
        assert lower_ids != upper_ids
        assert lower_ids | upper_ids == set(by_cid)

    def test_bounds_schema_matches_canonical_generic_bounds_schema(
        self, tmp_path: Path
    ) -> None:
        """NEWAVE's emitted bounds table now shares the canonical
        ``GENERIC_BOUNDS_SCHEMA`` (``constraint_id``/``stage_id`` non-nullable)
        instead of the legacy bare-``pa.table`` schema (all-nullable) — both
        tracks route through the same ``GenericConstraintBuilder``, so NEWAVE
        now matches DECOMP's schema strictness. This is a deliberate,
        data-safe tightening (neither column is ever actually null on either
        track), pinned here so it stays intentional rather than drifting."""
        case, id_map = _make_electric_re_case(tmp_path)

        result = convert_electric_constraints(case, id_map)

        assert result is not None
        assert result.bounds.schema.equals(GENERIC_BOUNDS_SCHEMA)


def _valid_ree_reader():
    """A REE.DAT reader with a clean individualizado cutoff (no diagnostic)."""
    ree = MagicMock()
    ree.rees = pd.DataFrame(
        {
            "codigo": [1],
            "nome": ["SUDESTE"],
            "mes_fim_individualizado": [12],
            "ano_fim_individualizado": [2020],
        }
    )
    return ree


class TestConvertElectricConstraintsEmission:
    """Emission-shape coverage for convert_electric_constraints' diagnostics."""

    def test_hydro_unmapped_from_formula_and_re_dat_sources(
        self, tmp_path: Path
    ) -> None:
        """One hydro-unmapped term from a formula and one from an
        RE.DAT-only constraint both land in the same diagnostic, with the
        Source column distinguishing them."""
        indices = tmp_path / "indices.csv"
        indices.write_text(
            "RESTRICAO-ELETRICA-ESPECIAL;Descricao;restricao-eletrica.csv\n",
            encoding="latin-1",
        )
        re_path = tmp_path / "restricao-eletrica.csv"
        re_path.write_text(
            "RE;1;1.0ger_usih(10) + 1.0ger_usih(9999)\n"
            "RE-HORIZ-PER;1;2020/01;2020/12\n"
            "RE-LIM-FORM-PER-PAT;1;2020/01;2020/12;1;50.;200.\n",
            encoding="latin-1",
        )

        re_dat_reader = MagicMock()
        re_dat_reader.usinas_conjuntos = pd.DataFrame(
            {"conjunto": [2, 2], "codigo_usina": [10, 9998]}
        )
        re_dat_reader.restricoes = pd.DataFrame(
            {
                "conjunto": [2],
                "mes_inicio": [1],
                "ano_inicio": [2020],
                "mes_fim": [12],
                "ano_fim": [2020],
                "patamar": [0],
                "restricao": [500.0],
            }
        )

        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2020
        dger.num_anos_estudo = 1
        dger.num_anos_pos_estudo = 0

        sistema = MagicMock()
        sistema.limites_intercambio = None
        sistema.custo_deficit = None

        files = make_nw_files(tmp_path, re_dat=tmp_path / "RE.DAT")
        case = make_case(
            files,
            dger=dger,
            sistema=sistema,
            re_dat=re_dat_reader,
            ree=_valid_ree_reader(),
        )
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[10], thermal_codes=[])

        with dx.collect() as collected:
            result = convert_electric_constraints(case, id_map)

        assert result is not None
        by_code = {d.code: d for d in collected}
        diag = by_code["electric-constraint-hydro-unmapped"]
        assert diag.severity is Severity.WARNING
        assert diag.category == "Special constraints"
        assert diag.table is not None
        assert diag.table.columns == ["Constraint", "Hydro code", "Source"]
        rows = {(r[1], r[2]) for r in diag.table.rows}
        assert rows == {(9999, "formula"), (9998, "RE.DAT")}

        _assert_no_repo_internal_leaks(collected)

    def test_interchange_no_line_recorded(self, tmp_path: Path) -> None:
        indices = tmp_path / "indices.csv"
        indices.write_text(
            "RESTRICAO-ELETRICA-ESPECIAL;Descricao;restricao-eletrica.csv\n",
            encoding="latin-1",
        )
        re_path = tmp_path / "restricao-eletrica.csv"
        re_path.write_text(
            "RE;1;1.0ger_usih(10) + 1.0ener_interc(2,3)\n"
            "RE-HORIZ-PER;1;2020/01;2020/12\n"
            "RE-LIM-FORM-PER-PAT;1;2020/01;2020/12;1;50.;200.\n",
            encoding="latin-1",
        )

        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2020
        dger.num_anos_estudo = 1
        dger.num_anos_pos_estudo = 0

        sistema = MagicMock()
        sistema.limites_intercambio = None
        sistema.custo_deficit = None

        case = make_case(
            tmp_path,
            dger=dger,
            sistema=sistema,
            re_dat=None,
            ree=_valid_ree_reader(),
        )
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[10], thermal_codes=[])

        with dx.collect() as collected:
            result = convert_electric_constraints(case, id_map)

        assert result is not None
        by_code = {d.code: d for d in collected}
        diag = by_code["electric-constraint-interchange-no-line"]
        assert diag.table is not None
        assert diag.table.columns == ["Constraint", "From", "To"]
        assert diag.table.rows == [[1, 2, 3]]
        assert diag.remediation is not None

        _assert_no_repo_internal_leaks(collected)

    def test_malformed_term_recorded(self, tmp_path: Path) -> None:
        indices = tmp_path / "indices.csv"
        indices.write_text(
            "RESTRICAO-ELETRICA-ESPECIAL;Descricao;restricao-eletrica.csv\n",
            encoding="latin-1",
        )
        re_path = tmp_path / "restricao-eletrica.csv"
        re_path.write_text(
            "RE;1;1.0ger_usih(10) + 1.0ener_interc(2)\n"
            "RE-HORIZ-PER;1;2020/01;2020/12\n"
            "RE-LIM-FORM-PER-PAT;1;2020/01;2020/12;1;50.;200.\n",
            encoding="latin-1",
        )

        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2020
        dger.num_anos_estudo = 1
        dger.num_anos_pos_estudo = 0

        sistema = MagicMock()
        sistema.limites_intercambio = None
        sistema.custo_deficit = None

        case = make_case(
            tmp_path,
            dger=dger,
            sistema=sistema,
            re_dat=None,
            ree=_valid_ree_reader(),
        )
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[10], thermal_codes=[])

        with dx.collect() as collected:
            result = convert_electric_constraints(case, id_map)

        assert result is not None
        by_code = {d.code: d for d in collected}
        diag = by_code["electric-constraint-malformed-term"]
        assert diag.table is not None
        assert diag.table.columns == ["Constraint", "Term"]
        assert diag.table.rows == [[1, "2"]]

        _assert_no_repo_internal_leaks(collected)

    def test_constraint_skipped_no_valid_terms_and_no_bound_data(
        self, tmp_path: Path
    ) -> None:
        """Two constraint codes hit the two ``electric-constraint-skipped``
        reasons: code 1 has only an unmapped hydro term (no valid terms);
        code 2 has a valid expression but a horizon carrying no bound rows."""
        indices = tmp_path / "indices.csv"
        indices.write_text(
            "RESTRICAO-ELETRICA-ESPECIAL;Descricao;restricao-eletrica.csv\n",
            encoding="latin-1",
        )
        re_path = tmp_path / "restricao-eletrica.csv"
        re_path.write_text(
            "RE;1;1.0ger_usih(9999)\n"
            "RE;2;1.0ger_usih(10)\n"
            "RE-HORIZ-PER;1;2020/01;2020/12\n"
            "RE-HORIZ-PER;2;2020/01;2020/12\n"
            "RE-LIM-FORM-PER-PAT;1;2020/01;2020/12;1;50.;200.\n",
            encoding="latin-1",
        )

        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2020
        dger.num_anos_estudo = 1
        dger.num_anos_pos_estudo = 0

        sistema = MagicMock()
        sistema.limites_intercambio = None
        sistema.custo_deficit = None

        case = make_case(
            tmp_path,
            dger=dger,
            sistema=sistema,
            re_dat=None,
            ree=_valid_ree_reader(),
        )
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[10], thermal_codes=[])

        with dx.collect() as collected:
            result = convert_electric_constraints(case, id_map)

        assert result is None  # both codes dropped -> no surviving constraint
        by_code = {d.code: d for d in collected}
        diag = by_code["electric-constraint-skipped"]
        assert diag.table is not None
        assert diag.table.columns == ["Constraint", "Reason"]
        reasons = {r[0]: r[1] for r in diag.table.rows}
        assert reasons[1] == "no valid terms in expression"
        assert reasons[2] == "no bound data"

        _assert_no_repo_internal_leaks(collected)

    def test_electric_penalty_unreadable_records_exception_text(
        self, tmp_path: Path
    ) -> None:
        class _BadPenalid:
            @property
            def penalidades(self) -> pd.DataFrame:
                raise ValueError("corrupt PENALID.DAT")

        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2020
        dger.num_anos_estudo = 1
        dger.num_anos_pos_estudo = 0

        sistema = MagicMock()
        sistema.limites_intercambio = None
        sistema.custo_deficit = None

        files = make_nw_files(tmp_path, penalid=tmp_path / "penalid.dat")
        indices = tmp_path / "indices.csv"
        indices.write_text(
            "RESTRICAO-ELETRICA-ESPECIAL;Descricao;restricao-eletrica.csv\n",
            encoding="latin-1",
        )
        re_path = tmp_path / "restricao-eletrica.csv"
        re_path.write_text(
            "RE;1;1.0ger_usih(10)\n"
            "RE-HORIZ-PER;1;2020/01;2020/12\n"
            "RE-LIM-FORM-PER-PAT;1;2020/01;2020/12;1;50.;200.\n",
            encoding="latin-1",
        )

        case = make_case(
            files,
            dger=dger,
            sistema=sistema,
            re_dat=None,
            penalid=_BadPenalid(),
            ree=_valid_ree_reader(),
        )
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[10], thermal_codes=[])

        with dx.collect() as collected:
            result = convert_electric_constraints(case, id_map)

        assert result is not None
        by_code = {d.code: d for d in collected}
        diag = by_code["electric-penalty-unreadable"]
        assert diag.severity is Severity.WARNING
        assert diag.table is None
        assert diag.notes == ["corrupt PENALID.DAT"]

        _assert_no_repo_internal_leaks(collected)

    def test_re_dat_unreadable_records_diagnostic(self, tmp_path: Path) -> None:
        indices = tmp_path / "indices.csv"
        indices.write_text(
            "RESTRICAO-ELETRICA-ESPECIAL;Descricao;restricao-eletrica.csv\n",
            encoding="latin-1",
        )
        re_path = tmp_path / "restricao-eletrica.csv"
        re_path.write_text(
            "RE;1;1.0ger_usih(10)\n"
            "RE-HORIZ-PER;1;2020/01;2020/12\n"
            "RE-LIM-FORM-PER-PAT;1;2020/01;2020/12;1;50.;200.\n",
            encoding="latin-1",
        )

        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2020
        dger.num_anos_estudo = 1
        dger.num_anos_pos_estudo = 0

        sistema = MagicMock()
        sistema.limites_intercambio = None
        sistema.custo_deficit = None

        files = make_nw_files(tmp_path, re_dat=tmp_path / "RE.DAT")
        case = NewaveCase(files=files)
        case.__dict__["dger"] = dger
        case.__dict__["sistema"] = sistema
        case.__dict__["ree"] = _valid_ree_reader()
        case.__dict__["hidr"] = MagicMock(cadastro=pd.DataFrame())
        # ``case.re_dat`` is a cached_property; deleting the pre-seeded slot and
        # patching the class descriptor to raise reproduces a genuine read
        # failure without touching disk.
        with patch.object(
            NewaveCase, "re_dat", property(lambda self: (_ for _ in ()).throw(OSError))
        ):
            id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[10], thermal_codes=[])

            with dx.collect() as collected:
                result = convert_electric_constraints(case, id_map)

        assert result is not None
        by_code = {d.code: d for d in collected}
        diag = by_code["re-dat-unreadable"]
        assert diag.severity is Severity.WARNING
        assert "skipped" in diag.summary

        _assert_no_repo_internal_leaks(collected)

    def test_individualizado_cutoff_unknown_no_entries(self, tmp_path: Path) -> None:
        indices = tmp_path / "indices.csv"
        indices.write_text(
            "RESTRICAO-ELETRICA-ESPECIAL;Descricao;restricao-eletrica.csv\n",
            encoding="latin-1",
        )
        re_path = tmp_path / "restricao-eletrica.csv"
        re_path.write_text(
            "RE;1;1.0ger_usih(10)\n"
            "RE-HORIZ-PER;1;2020/01;2020/12\n"
            "RE-LIM-FORM-PER-PAT;1;2020/01;2020/12;1;50.;200.\n",
            encoding="latin-1",
        )

        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2020
        dger.num_anos_estudo = 1
        dger.num_anos_pos_estudo = 0

        sistema = MagicMock()
        sistema.limites_intercambio = None
        sistema.custo_deficit = None

        empty_ree = MagicMock()
        empty_ree.rees = pd.DataFrame()

        case = make_case(
            tmp_path, dger=dger, sistema=sistema, re_dat=None, ree=empty_ree
        )
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[10], thermal_codes=[])

        with dx.collect() as collected:
            result = convert_electric_constraints(case, id_map)

        assert result is not None
        by_code = {d.code: d for d in collected}
        diag = by_code["individualizado-cutoff-unknown"]
        assert diag.severity is Severity.WARNING
        assert diag.table is None

        _assert_no_repo_internal_leaks(collected)


# ---------------------------------------------------------------------------
# convert_agrint_constraints tests
# ---------------------------------------------------------------------------


def _make_minimal_case(
    tmp_path: Path,
    *,
    agrint: Path | None = None,
    dger: object | None = None,
):
    """Return a ``NewaveCase`` with minimal inputs for AGRINT tests.

    ``patamar`` points to a missing path so ``case.patamar`` reads it via the
    real ``Patamar.read`` and yields ``numero_patamares=None`` (→ 1 patamar),
    matching the legacy ``NewaveFiles`` behaviour. A parsed ``dger`` reader can
    be pre-injected so ``case.dger`` returns it without touching disk.
    """
    files = make_nw_files(tmp_path, agrint=agrint)
    parsed: dict[str, object] = {}
    if dger is not None:
        parsed["dger"] = dger
    return make_case(files, **parsed)


_AGRINT_CONTENT = """\
AGRUPAMENTOS DE INTERCAMBIO
 # AG A   B   COEF
 XXX XXX XXX XX.XXXX
   1   1   3  1.0000
   2   3   1  1.0000
 999
LIMITES POR GRUPO
  # AG MI ANOI MF ANOF LIM_P1  LIM_P2  LIM_P3
 XXX  XX XXXX XX XXXX XXXXXX. XXXXXX. XXXXXX.
   1   1 2020  2 2020  10000.  10000.  10000.
   2   1 2020  1 2020   5000.   5000.   5000.
 999
"""


def _make_dger_mock_for_agrint():
    """Dger mock for study starting Jan 2020, 1 year."""
    dger = MagicMock()
    dger.mes_inicio_estudo = 1
    dger.ano_inicio_estudo = 2020
    dger.num_anos_estudo = 1
    dger.num_anos_pos_estudo = 0
    return dger


class TestConvertAgrintConstraints:
    def test_returns_none_when_agrint_absent(self, tmp_path: Path) -> None:
        """Returns None when agrint path is None."""
        case = _make_minimal_case(tmp_path, agrint=None)
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])
        assert convert_agrint_constraints(case, id_map) is None

    def test_switched_off_agrint_is_ignored_and_reported(self, tmp_path: Path) -> None:
        """``AGRUPAMENTO LIVRE = 0`` leaves a present agrint.dat out, as the
        source model does, with one INFO diagnostic naming the switch."""
        agrint_path = tmp_path / "agrint.dat"
        agrint_path.write_text(_AGRINT_CONTENT, encoding="latin-1")
        dger = _make_dger_mock_for_agrint()
        dger.agrupamento_livre = 0
        case = _make_minimal_case(tmp_path, agrint=agrint_path, dger=dger)
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])

        with dx.collect() as collected:
            assert convert_agrint_constraints(case, id_map) is None

        assert [d.code for d in collected] == ["dger-switch-off"]
        assert collected[0].severity is Severity.INFO
        assert "agrint.dat" in collected[0].title

    def test_produces_constraints_from_agrint_dat(self, tmp_path: Path) -> None:
        """Parses a minimal AGRINT file and produces one constraint per group."""
        agrint_path = tmp_path / "agrint.dat"
        agrint_path.write_text(_AGRINT_CONTENT, encoding="latin-1")
        (tmp_path / "dger.dat").touch()

        case = _make_minimal_case(
            tmp_path, agrint=agrint_path, dger=_make_dger_mock_for_agrint()
        )
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])

        # Line map: canonical (1,3) -> line_id=0
        fake_line_map = {(1, 3): 0}

        with patch(
            "novomodelo_bridge.newave.converters.constraints._build_line_id_map",
            return_value=fake_line_map,
        ):
            result = convert_agrint_constraints(case, id_map)

        assert result is not None
        constraints, bounds_table = result
        assert len(constraints) == 2

    def test_constraint_has_no_sense_key(self, tmp_path: Path) -> None:
        """AGRINT constraint objects have exactly novomodelo's F3 sense-free keys."""
        agrint_path = tmp_path / "agrint.dat"
        agrint_path.write_text(_AGRINT_CONTENT, encoding="latin-1")
        (tmp_path / "dger.dat").touch()

        case = _make_minimal_case(
            tmp_path, agrint=agrint_path, dger=_make_dger_mock_for_agrint()
        )
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])

        with patch(
            "novomodelo_bridge.newave.converters.constraints._build_line_id_map",
            return_value={(1, 3): 0},
        ):
            result = convert_agrint_constraints(case, id_map)

        assert result is not None
        for c in result[0]:
            assert set(c) == {"id", "name", "description", "expression", "slack"}
            assert c["slack"]["enabled"] is False

    def test_allocator_offset_applied(self, tmp_path: Path) -> None:
        """A non-default allocator's starting id is added to all constraint IDs."""
        agrint_path = tmp_path / "agrint.dat"
        agrint_path.write_text(_AGRINT_CONTENT, encoding="latin-1")
        (tmp_path / "dger.dat").touch()

        case = _make_minimal_case(
            tmp_path, agrint=agrint_path, dger=_make_dger_mock_for_agrint()
        )
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])

        with patch(
            "novomodelo_bridge.newave.converters.constraints._build_line_id_map",
            return_value={(1, 3): 0},
        ):
            result_0 = convert_agrint_constraints(case, id_map)
            result_5 = convert_agrint_constraints(
                case, id_map, allocator=ConstraintIdAllocator(5)
            )

        assert result_0 is not None and result_5 is not None
        ids_0 = [c["id"] for c in result_0[0]]
        ids_5 = [c["id"] for c in result_5[0]]
        assert ids_5 == [i + 5 for i in ids_0]

    def test_bounds_table_schema(self, tmp_path: Path) -> None:
        """Bounds table has the F3 schema (bound_lower/bound_upper, no bound)."""
        agrint_path = tmp_path / "agrint.dat"
        agrint_path.write_text(_AGRINT_CONTENT, encoding="latin-1")
        (tmp_path / "dger.dat").touch()

        case = _make_minimal_case(
            tmp_path, agrint=agrint_path, dger=_make_dger_mock_for_agrint()
        )
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])

        with patch(
            "novomodelo_bridge.newave.converters.constraints._build_line_id_map",
            return_value={(1, 3): 0},
        ):
            result = convert_agrint_constraints(case, id_map)

        assert result is not None
        _, bounds_table = result
        assert isinstance(bounds_table, pa.Table)
        assert set(bounds_table.schema.names) == set(GENERIC_BOUNDS_COLUMNS)

    def test_post_study_freezes_at_last_study_value(self, tmp_path: Path) -> None:
        """Post-study AGRINT limits freeze at the last study stage value and ignore
        future-dated agrint.dat entries (the source model convention — the pmo.dat
        "LIMITES DOS AGRUPAMENTOS DE INTERCAMBIO" POS row is flat at the last study
        December value)."""
        content = (
            "AGRUPAMENTOS DE INTERCAMBIO\n"
            " #AG A   B   COEF\n"
            " XXX XXX XXX XX.XXXX\n"
            "   1   1   3  1.0000\n"
            " 999\n"
            "LIMITES POR GRUPO\n"
            "  #AG MI ANOI MF ANOF LIM_P1  LIM_P2  LIM_P3\n"
            " XXX  XX XXXX XX XXXX XXXXXX. XXXXXX. XXXXXX.\n"
            "   1   1 2020 12 2020  10000.  10000.  10000.\n"
            "   1   1 2021 12 2021  20000.  20000.  20000.\n"
            " 999\n"
        )
        agrint_path = tmp_path / "agrint.dat"
        agrint_path.write_text(content, encoding="latin-1")
        (tmp_path / "dger.dat").touch()

        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2020
        dger.num_anos_estudo = 1  # study_months = 12 (Jan–Dec 2020)
        dger.num_anos_pos_estudo = 1  # post-study 2021 → stages 12–23

        case = _make_minimal_case(tmp_path, agrint=agrint_path, dger=dger)
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])

        with patch(
            "novomodelo_bridge.newave.converters.constraints._build_line_id_map",
            return_value={(1, 3): 0},
        ):
            result = convert_agrint_constraints(case, id_map)

        assert result is not None
        _, bounds = result
        # Every AGRINT constraint is a `<=` ceiling, so its value lives in
        # `bound_upper`; `bound_lower` is always null.
        b0 = (
            bounds.to_pandas()
            .query("block_id == 0")
            .set_index("stage_id")["bound_upper"]
        )
        # Study (0–11): 10000.
        assert b0[0] == 10000.0
        assert b0[11] == 10000.0
        # Post-study (12–23): frozen at 10000, NOT the 2021 entry's 20000.
        for s in range(12, 24):
            assert b0[s] == 10000.0

    def test_multi_term_group_with_mixed_directions(self, tmp_path: Path) -> None:
        """Mixed direction terms (typical of NOFICT1 hubs) each pick the right
        variable.

        This pins the source model Group 4 case (``Interc(11→1) + Interc(11→3) ≤ 8000``)
        — both terms target a fictitious bus from subsystems with smaller codes, so
        canonically the lines are (1,11) and (3,11) but the directional flow we care
        about is the reverse one on each. The converter must emit
        ``line_reverse(line_1_11) + line_reverse(line_3_11)``, not signed
        ``-line_exchange`` terms, otherwise the constraint silently allows extra flow
        when one of the lines runs in its canonical (direct) direction.
        """
        agrint_path = tmp_path / "agrint.dat"
        agrint_path.write_text(
            "AGRUPAMENTOS DE INTERCAMBIO\n"
            " #AG A   B   COEF\n"
            " XXX XXX XXX XX.XXXX\n"
            "   4  11   1  1.0000\n"
            "   4  11   3  1.0000\n"
            " 999\n"
            "LIMITES POR GRUPO\n"
            "  #AG MI ANOI MF ANOF LIM_P1  LIM_P2  LIM_P3\n"
            " XXX  XX XXXX XX XXXX XXXXXX. XXXXXX. XXXXXX.\n"
            "   4   1 2020 12 2020   8000.   8000.   8000.\n"
            " 999\n",
            encoding="latin-1",
        )
        (tmp_path / "dger.dat").touch()

        case = _make_minimal_case(
            tmp_path, agrint=agrint_path, dger=_make_dger_mock_for_agrint()
        )
        id_map = NewaveIdMap(subsystem_ids=[1, 3, 11], hydro_codes=[], thermal_codes=[])

        with patch(
            "novomodelo_bridge.newave.converters.constraints._build_line_id_map",
            return_value={(1, 11): 3, (3, 11): 4},
        ):
            result = convert_agrint_constraints(case, id_map)

        assert result is not None
        constraints, _ = result
        assert len(constraints) == 1
        expr = constraints[0]["expression"]
        # Both terms are reverse direction on their canonical lines.
        assert "line_reverse(3)" in expr
        assert "line_reverse(4)" in expr
        # No signed line_exchange anywhere in the output.
        assert "line_exchange" not in expr
        # Coefficient stays positive — direction is encoded by variable.
        assert not expr.lstrip().startswith("-")

    def test_reversed_direction_uses_line_reverse(self, tmp_path: Path) -> None:
        """Flow A->B where A>B emits ``line_reverse``; canonical uses ``line_direct``.

        The source model's ``Interc(A→B)`` is a non-negative directional flow, so the
        converter picks ``line_direct`` (canonical A<B) or ``line_reverse`` (reversed
        A>B) instead of a signed ``line_exchange`` with a negated coefficient.  This
        keeps the constraint tight at fictitious hubs where one flow can be in its
        non-canonical direction.
        """
        agrint_path = tmp_path / "agrint.dat"
        agrint_path.write_text(_AGRINT_CONTENT, encoding="latin-1")
        (tmp_path / "dger.dat").touch()

        case = _make_minimal_case(
            tmp_path, agrint=agrint_path, dger=_make_dger_mock_for_agrint()
        )
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])

        with patch(
            "novomodelo_bridge.newave.converters.constraints._build_line_id_map",
            return_value={(1, 3): 0},
        ):
            result = convert_agrint_constraints(case, id_map)

        assert result is not None
        # Group 1: flow(1->3), canonical (1,3) => line_direct, no leading '-'.
        c1 = result[0][0]
        assert c1["expression"].startswith("line_direct(0)")
        assert "line_exchange" not in c1["expression"]

        # Group 2: flow(3->1), reverse of canonical => line_reverse with a
        # positive coefficient (no leading '-').
        c2 = result[0][1]
        assert c2["expression"].startswith("line_reverse(0)")
        assert "line_exchange" not in c2["expression"]
        assert not c2["expression"].startswith("-")


class TestConvertAgrintConstraintsEmission:
    """Emission-shape coverage for convert_agrint_constraints' diagnostics."""

    def test_no_line_and_empty_group_both_recorded(self, tmp_path: Path) -> None:
        """Group 1 has one resolvable term and one unresolvable term (dropped,
        recorded); group 2 has only an unresolvable term (the whole group is
        dropped, recorded separately)."""
        content = (
            "AGRUPAMENTOS DE INTERCAMBIO\n"
            " #AG A   B   COEF\n"
            " XXX XXX XXX XX.XXXX\n"
            "   1   1   3  1.0000\n"
            "   1   2   9  1.0000\n"
            "   2   5   6  1.0000\n"
            " 999\n"
            "LIMITES POR GRUPO\n"
            "  #AG MI ANOI MF ANOF LIM_P1  LIM_P2  LIM_P3\n"
            " XXX  XX XXXX XX XXXX XXXXXX. XXXXXX. XXXXXX.\n"
            "   1   1 2020 12 2020  10000.  10000.  10000.\n"
            "   2   1 2020 12 2020   5000.   5000.   5000.\n"
            " 999\n"
        )
        agrint_path = tmp_path / "agrint.dat"
        agrint_path.write_text(content, encoding="latin-1")
        (tmp_path / "dger.dat").touch()

        case = _make_minimal_case(
            tmp_path, agrint=agrint_path, dger=_make_dger_mock_for_agrint()
        )
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])

        with patch(
            "novomodelo_bridge.newave.converters.constraints._build_line_id_map",
            return_value={(1, 3): 0},
        ):
            with dx.collect() as collected:
                result = convert_agrint_constraints(case, id_map)

        assert result is not None  # group 1 still yields a constraint
        by_code = {d.code: d for d in collected}

        no_line = by_code["agrint-interchange-no-line"]
        assert no_line.severity is Severity.WARNING
        assert no_line.category == "Special constraints"
        assert no_line.table is not None
        assert no_line.table.columns == ["Group", "From", "To"]
        # Group 2's only term is also unresolvable, so it is recorded here too
        # (per-term), in addition to the whole-group record below.
        assert no_line.table.rows == [[1, 2, 9], [2, 5, 6]]

        empty_group = by_code["agrint-group-empty"]
        assert empty_group.table is not None
        assert empty_group.table.columns == ["Group"]
        assert empty_group.table.rows == [[2]]

        _assert_no_repo_internal_leaks(collected)


class TestNewaveCrossEmitterAllocatorContiguity:
    """Threading one ``ConstraintIdAllocator`` across emitter calls (as the
    pipeline does) keeps ids contiguous, closing the old per-emitter
    ``start_id`` arithmetic's gap-on-skip risk."""

    def test_electric_then_agrint_share_contiguous_ids(self, tmp_path: Path) -> None:
        electric_case, electric_id_map = _make_electric_re_case(tmp_path)

        agrint_dir = tmp_path / "agrint_case"
        agrint_dir.mkdir()
        agrint_path = agrint_dir / "agrint.dat"
        agrint_path.write_text(_AGRINT_CONTENT, encoding="latin-1")
        (agrint_dir / "dger.dat").touch()
        agrint_case = _make_minimal_case(
            agrint_dir, agrint=agrint_path, dger=_make_dger_mock_for_agrint()
        )
        agrint_id_map = NewaveIdMap(
            subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[]
        )

        allocator = ConstraintIdAllocator()
        with patch(
            "novomodelo_bridge.newave.converters.constraints._build_line_id_map",
            return_value={(1, 3): 0},
        ):
            electric_result = convert_electric_constraints(
                electric_case, electric_id_map, allocator=allocator
            )
            agrint_result = convert_agrint_constraints(
                agrint_case, agrint_id_map, allocator=allocator
            )

        assert electric_result is not None and agrint_result is not None
        electric_ids = [c["id"] for c in electric_result.constraints]
        agrint_ids = [c["id"] for c in agrint_result.constraints]
        # The two-sided bound still yields two single-sided ids under the
        # shared allocator (Req 3's byte-preservation is unaffected by Req 5's
        # id-threading change).
        assert len(electric_ids) == 2
        assert min(agrint_ids) == max(electric_ids) + 1


class TestConstraintResultTypes:
    """The constraint converters return named tuples (named + index access)."""

    def test_vminop_result_named_and_tuple_access(self) -> None:
        from novomodelo_bridge.newave.converters.constraints import VminopResult

        bounds = pa.table({"constraint_id": [0]})
        r = VminopResult({"constraints": [1, 2]}, bounds, [3, 4], {5: [1.0]})
        # Named access (new).
        assert r.constraints_dict == {"constraints": [1, 2]}
        assert r.bounds is bounds
        assert r.referenced_hydro_ids == [3, 4]
        assert r.rho_acum_overrides == {5: [1.0]}
        # Legacy index / destructure still works (backward compatible).
        assert r[2] == [3, 4]
        cdict, b, ids, overrides = r
        assert cdict == {"constraints": [1, 2]}
        assert ids == [3, 4]

    def test_generic_constraint_result_named_and_tuple_access(self) -> None:
        from novomodelo_bridge.newave.converters.constraints import (
            GenericConstraintResult,
        )

        bounds = pa.table({"constraint_id": [0]})
        r = GenericConstraintResult([{"id": 0}], bounds)
        assert r.constraints == [{"id": 0}]
        assert r.bounds is bounds
        constraints, b = r  # legacy unpack
        assert constraints == [{"id": 0}]


class TestVminopEnergyFactor:
    """Guard the VminOP unit conversion: ρ_acum·hm³ → MWmonth must use each
    stage's *real* month length, not the source model's fixed 730 h.

    Regression for the security-curve bug: leaving the VminOP LHS in
    ρ_acum·hm³ (≈ 2.628× true MWmonth) while novomodelo prices the slack
    ``× block_hours`` made the effective curve-violation penalty exceed the
    deficit cost, so novomodelo deficited instead of drawing reservoirs down.
    """

    def test_factor_matches_real_month_hours(self) -> None:
        # Study starting Sep 2024 (the example horizon).
        #   stage 0 = Sep 2024 (30 d = 720 h)
        #   stage 3 = Dec 2024 (31 d = 744 h)
        #   stage 5 = Feb 2025 (28 d = 672 h)
        assert _vminop_energy_factor(2024, 9, 0) == pytest.approx(720 * 3600 / 1e6)
        assert _vminop_energy_factor(2024, 9, 3) == pytest.approx(744 * 3600 / 1e6)
        assert _vminop_energy_factor(2024, 9, 5) == pytest.approx(672 * 3600 / 1e6)

    def test_factor_differs_from_fixed_constant_off_730(self) -> None:
        # Short/long months must NOT use the fixed 730 h constant — that was the
        # bug. February (672 h) is ~8% below C_M3S2HM3.
        feb = _vminop_energy_factor(2024, 9, 5)
        assert feb != pytest.approx(C_M3S2HM3)
        assert feb / C_M3S2HM3 == pytest.approx(672 / 730)

    def test_factor_equals_constant_for_730h_month(self) -> None:
        # A hypothetical 730 h month (the source model's convention) reproduces
        # C_M3S2HM3. No real calendar month is 730 h, so assert the relationship holds
        # via the month-hours ratio instead (April = 30 d = 720 h).
        apr = _vminop_energy_factor(2025, 4, 0)
        assert apr == pytest.approx(C_M3S2HM3 * 720 / 730)


class TestConstraintsNoLongerBridgesLegacyWarning:
    """After the migration, constraints.py has no ``_LOG.warning(...)`` call
    left to be swept into a generic ``legacy-warning`` panel (OQ3); the
    residual bridge in :func:`finalize_diagnostics` still wraps an unrelated
    captured legacy string, so that fallback path stays intact for any
    not-yet-migrated module."""

    def test_vminop_skip_under_collect_carries_no_legacy_warning(
        self, tmp_path: Path
    ) -> None:
        case, id_map = _make_vminop_no_hydro_case(tmp_path)

        with dx.collect() as collected:
            convert_vminop_constraints(case, id_map)

        assert not any(d.code == "legacy-warning" for d in collected)

    def test_finalize_diagnostics_still_wraps_an_unrelated_legacy_string(
        self,
    ) -> None:
        legacy_messages = ["some other warning"]
        result = finalize_diagnostics([], legacy_messages)

        assert len(result) == 1
        assert result[0].code == "legacy-warning"
        assert result[0].summary == "some other warning"


# ---------------------------------------------------------------------------
# Tests: electric constraint validity windows
# ---------------------------------------------------------------------------

_RE_COLUMNS = (
    "conjunto",
    "mes_inicio",
    "ano_inicio",
    "mes_fim",
    "ano_fim",
    "patamar",
    "restricao",
)


def test_re_dat_bound_covers_only_its_declared_window(tmp_path: Path) -> None:
    # A patamar-0 row over 2020 plus a narrower patamar-2 row (March-May),
    # inside a three-year horizon.
    reader = MagicMock()
    reader.usinas_conjuntos = pd.DataFrame({"conjunto": [1], "codigo_usina": [10]})
    reader.restricoes = pd.DataFrame(
        [(1, 1, 2020, 12, 2020, 0, 500.0), (1, 3, 2020, 5, 2020, 2, 300.0)],
        columns=list(_RE_COLUMNS),
    )
    files = make_nw_files(tmp_path, re_dat=tmp_path / "RE.DAT")

    _, bounds = _parse_re_dat(make_case(files, re_dat=reader), 2020, 1, 36, 3)
    stage_bounds = bounds[1]

    assert set(stage_bounds) == {(sid, b) for sid in range(12) for b in range(3)}
    assert [stage_bounds[(2, b)] for b in range(3)] == [500.0, 300.0, 500.0]
    assert [stage_bounds[(5, b)] for b in range(3)] == [500.0, 500.0, 500.0]


def test_restricao_eletrica_bounds_stop_at_the_declared_end(tmp_path: Path) -> None:
    # Declared for the first quarter of a three-year horizon whose
    # individualizado period ends in December 2020.
    (tmp_path / "indices.csv").write_text(
        "RESTRICAO-ELETRICA-ESPECIAL;Descricao;restricao-eletrica.csv\n",
        encoding="latin-1",
    )
    (tmp_path / "restricao-eletrica.csv").write_text(
        "RE;1;1.0ger_usih(10)\n"
        "RE-HORIZ-PER;1;2020/01;2020/03\n"
        "RE-LIM-FORM-PER-PAT;1;2020/01;2020/03;1;50.;200.\n",
        encoding="latin-1",
    )
    dger = MagicMock(
        mes_inicio_estudo=1,
        ano_inicio_estudo=2020,
        num_anos_estudo=3,
        num_anos_pos_estudo=0,
    )
    sistema = MagicMock(limites_intercambio=None, custo_deficit=None)
    case = make_case(
        make_nw_files(tmp_path), dger=dger, sistema=sistema, ree=_valid_ree_reader()
    )
    id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[10], thermal_codes=[])

    result = convert_electric_constraints(case, id_map)

    assert result is not None
    df = result[1].to_pandas()
    assert set(df[df["bound_upper"].notna()]["stage_id"]) == {0, 1, 2}
    assert set(df[df["bound_lower"].notna()]["stage_id"]) == {0, 1, 2}
