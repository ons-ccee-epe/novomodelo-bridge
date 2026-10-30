"""Unit tests for the VminOP, electric, and AGRINT generic constraints converters."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity, finalize_diagnostics
from novomodelo_bridge.core.errors import FieldParseError, SourceFileError
from novomodelo_bridge.core.generic_constraint_builder import (
    GENERIC_BOUNDS_SCHEMA,
    ConstraintIdAllocator,
)
from novomodelo_bridge.core.generic_constraint_format import GENERIC_BOUNDS_COLUMNS
from novomodelo_bridge.core.libs_index import UNREAD_LINES_CODE
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.converters.constraints import (
    _cascade_sum,
    _curve_seasonalizes,
    _ElectricTermSkip,
    _is_stored_energy_reservoir,
    _parse_formula,
    _parse_re_dat,
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
        acc = compute_accumulated_productivities(
            cadastro, confhd_df, fictitious=frozenset()
        )

        # Plant 3 is the sink (downstream=None).
        # Its own productivity: 0.03 * (500 + 0.02*1500 - 400) = 0.03 * 130 = 3.9
        own_3 = 0.03 * ((500.0 + 0.02 * 1500.0) - 400.0)
        assert acc[3] == pytest.approx(own_3)

    def test_middle_plant_accumulates_downstream(self) -> None:
        cadastro = _make_cadastro()
        confhd_df = _make_confhd_df()
        acc = compute_accumulated_productivities(
            cadastro, confhd_df, fictitious=frozenset()
        )

        own_3 = 0.03 * ((500.0 + 0.02 * 1500.0) - 400.0)
        own_2 = 0.02 * ((400.0 + 0.05 * 1000.0) - 300.0)
        assert acc[2] == pytest.approx(own_2 + own_3)

    def test_upstream_plant_accumulates_full_chain(self) -> None:
        cadastro = _make_cadastro()
        confhd_df = _make_confhd_df()
        acc = compute_accumulated_productivities(
            cadastro, confhd_df, fictitious=frozenset()
        )

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

        acc = compute_accumulated_productivities(
            cadastro, confhd_df, fictitious=frozenset()
        )

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
        acc = compute_accumulated_integrated_productivities(
            cadastro, confhd_df, fictitious=frozenset()
        )
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
        acc = compute_accumulated_integrated_productivities(
            cadastro, confhd_df, fictitious=frozenset()
        )
        # PLANT_A @vref=500: 0.01 · (300 + 0.1·500 − 200) = 0.01 · 150 = 1.5
        # PLANT_B @vref=1000: 0.02 · (400 + 0.05·1000 − 300) = 0.02 · 150 = 3.0
        # PLANT_C @vref=1500: 0.03 · (500 + 0.02·1500 − 400) = 0.03 · 130 = 3.9
        own_a = 0.01 * (300.0 + 0.1 * 500.0 - 200.0)
        own_b = 0.02 * (400.0 + 0.05 * 1000.0 - 300.0)
        own_c = 0.03 * (500.0 + 0.02 * 1500.0 - 400.0)
        assert acc[3] == pytest.approx(own_c)
        assert acc[2] == pytest.approx(own_b + own_c)
        assert acc[1] == pytest.approx(own_a + own_b + own_c)


class TestBuildScalarParameters:
    """build_scalar_parameters declares the per-hydro computed family for every
    hydro, then the caller's per_stage and expression entries."""

    def test_declares_the_computed_family_per_hydro(self) -> None:
        d = build_scalar_parameters([1, 0, 1])
        entries = d["scalar_parameters"]
        assert [e["name"] for e in entries] == [
            "rho_eq_h0",
            "rho_acum_h0",
            "rho_eq_mwh_h0",
            "rho_acum_mwh_h0",
            "earm_max_mwh_h0",
            "min_storage_h0",
            "max_storage_h0",
            "rho_eq_h1",
            "rho_acum_h1",
            "rho_eq_mwh_h1",
            "rho_acum_mwh_h1",
            "earm_max_mwh_h1",
            "min_storage_h1",
            "max_storage_h1",
        ]
        assert all(e["kind"] == "computed" for e in entries)
        assert [e["computed_spec"]["tag"] for e in entries[:7]] == [
            "equivalent_productivity",
            "accumulated_productivity",
            "integrated_equivalent_productivity_mwh_per_hm3",
            "integrated_accumulated_productivity_mwh_per_hm3",
            "max_stored_energy_mwh",
            "min_storage",
            "max_storage",
        ]
        assert {e["computed_spec"]["hydro_id"] for e in entries[7:]} == {1}

    def test_per_stage_entries_follow_the_computed_family(self) -> None:
        d = build_scalar_parameters([0], per_stage={"pct_vminop_ree1": [0.2, 0.25]})
        entries = d["scalar_parameters"]
        assert entries[-1] == {
            "id": 7,
            "name": "pct_vminop_ree1",
            "kind": "per_stage",
            "values": [[0, 0.2], [1, 0.25]],
        }

    def test_expression_entries_follow_the_per_stage_entries(self) -> None:
        d = build_scalar_parameters(
            [0],
            per_stage={"p": [1.0]},
            expressions={"rho_chain_mwh_h0": "@rho_eq_mwh_h0 + @p"},
        )
        entries = d["scalar_parameters"]
        assert [e["kind"] for e in entries[-2:]] == ["per_stage", "expression"]
        assert entries[-1] == {
            "id": 8,
            "name": "rho_chain_mwh_h0",
            "kind": "expression",
            "expression": "@rho_eq_mwh_h0 + @p",
        }

    def test_ids_are_contiguous_from_zero(self) -> None:
        d = build_scalar_parameters(
            [0, 1], per_stage={"p": [1.0]}, expressions={"e": "2.0 * @p"}
        )
        ids = [e["id"] for e in d["scalar_parameters"]]
        assert ids == list(range(len(ids)))


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


_TWO_SIDED_RE = (
    "RE;1;1.0ger_usih(10)\n"
    "RE-HORIZ-PER;1;2020/01;2020/12\n"
    "RE-LIM-FORM-PER-PAT;1;2020/01;2020/12;1;50.;200.\n"
)


def _make_electric_re_case(
    tmp_path: Path, re_text: str = _TWO_SIDED_RE
) -> tuple[NewaveCase, NewaveIdMap]:
    """Return a ``(case, id_map)`` pair whose ``restricao-eletrica.csv`` is
    *re_text*, by default a two-sided ``[50, 200]`` limit on ``ger_usih(10)``
    over 2020, with ``dger``/``sistema`` mocked to a 1-year 2020 study carrying
    no exchange/deficit data.
    """
    re_path = tmp_path / "restricao-eletrica.csv"
    re_path.write_text(re_text, encoding="latin-1")

    dger = MagicMock()
    dger.mes_inicio_estudo = 1
    dger.ano_inicio_estudo = 2020
    dger.num_anos_estudo = 1
    dger.num_anos_pos_estudo = 0

    sistema = MagicMock()
    sistema.limites_intercambio = None
    sistema.custo_deficit = None

    case = make_case(
        make_nw_files(tmp_path, restricao_eletrica=re_path),
        dger=dger,
        sistema=sistema,
        re_dat=None,
    )
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

    def test_returns_none_when_the_deck_declares_no_csv(self, tmp_path: Path) -> None:
        case = make_case(tmp_path, re_dat=None)
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[], thermal_codes=[])
        assert convert_electric_constraints(case, id_map) is None

    def test_two_sided_bound_yields_two_single_sided_f3_constraints(
        self, tmp_path: Path
    ) -> None:
        """A constraint code with both a lower and upper limit produces two
        sense-free constraint objects sharing one expression/name (the reader's
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

    @pytest.mark.parametrize(
        ("limits", "lower", "upper"),
        [
            ("1;;200.", set(), {200.0}),
            ("1;50.;", {50.0}, set()),
            ("1;50.", {50.0}, set()),
            ("1;-1.1e30;200.", set(), {200.0}),
        ],
        ids=["blank-lower", "blank-upper", "missing-upper", "lower-sentinel"],
    )
    def test_a_blank_or_sentinel_limit_is_unbounded(
        self, tmp_path: Path, limits: str, lower: set[float], upper: set[float]
    ) -> None:
        case, id_map = _make_electric_re_case(
            tmp_path,
            f"RE;1;1.0ger_usih(10)\nRE-LIM-FORM-PER-PAT;1;2020/01;2020/12;{limits}\n",
        )

        result = convert_electric_constraints(case, id_map)

        assert result is not None
        assert len(result.constraints) == 1
        rows = result.bounds.to_pylist()
        assert {r["bound_lower"] for r in rows if r["bound_lower"] is not None} == lower
        assert {r["bound_upper"] for r in rows if r["bound_upper"] is not None} == upper

    def test_a_conditional_limit_is_rejected(self, tmp_path: Path) -> None:
        case, id_map = _make_electric_re_case(
            tmp_path,
            "RE;1;1.0ger_usih(10)\n"
            "RE-LIM-FORM-PER-PAT;1;2020/01;2020/12;1;"
            "se(demanda(1) < 1000, 10000, 8500);200.\n",
        )

        with pytest.raises(SourceFileError, match="which is not a number") as err:
            convert_electric_constraints(case, id_map)
        assert "se(demanda(1) < 1000, 10000, 8500)" in str(err.value)
        assert err.value.field == "RE-LIM-FORM-PER-PAT"

    def test_a_period_that_is_not_a_date_is_rejected(self, tmp_path: Path) -> None:
        case, id_map = _make_electric_re_case(
            tmp_path,
            _TWO_SIDED_RE.replace("RE-HORIZ-PER;1;2020/01", "RE-HORIZ-PER;1;1"),
        )

        with pytest.raises(
            SourceFileError, match=r"RE-HORIZ-PER line without a readable start period"
        ):
            convert_electric_constraints(case, id_map)

    def test_lines_the_reader_keeps_as_text_are_warned(self, tmp_path: Path) -> None:
        case, id_map = _make_electric_re_case(
            tmp_path,
            _TWO_SIDED_RE
            + "re;2;1.0ger_usih(10)\nRESTRICAO-ELETRICA-FORMULA;3;ger_usih(10)\n",
        )

        with dx.collect() as collected:
            result = convert_electric_constraints(case, id_map)

        assert result is not None
        assert {c["name"] for c in result.constraints} == {"RE_1"}
        [unread] = [d for d in collected if d.code == UNREAD_LINES_CODE]
        assert unread.table is not None
        assert unread.table.rows == [
            ["re;2;1.0ger_usih(10)"],
            ["RESTRICAO-ELETRICA-FORMULA;3;ger_usih(10)"],
        ]


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

        files = make_nw_files(
            tmp_path, re_dat=tmp_path / "RE.DAT", restricao_eletrica=re_path
        )
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
            make_nw_files(tmp_path, restricao_eletrica=re_path),
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
            make_nw_files(tmp_path, restricao_eletrica=re_path),
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
            make_nw_files(tmp_path, restricao_eletrica=re_path),
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

        re_path = tmp_path / "restricao-eletrica.csv"
        files = make_nw_files(
            tmp_path, penalid=tmp_path / "penalid.dat", restricao_eletrica=re_path
        )
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

        files = make_nw_files(
            tmp_path, re_dat=tmp_path / "RE.DAT", restricao_eletrica=re_path
        )
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
            make_nw_files(tmp_path, restricao_eletrica=re_path),
            dger=dger,
            sistema=sistema,
            re_dat=None,
            ree=empty_ree,
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
        """AGRINT constraint objects have exactly novomodelo's sense-free keys."""
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
        """Bounds table has the interval schema (bound_lower/bound_upper, no bound)."""
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


def _three_block_agrint_case(tmp_path: Path, limit_line: str) -> NewaveCase:
    """A 3-block, one-year case whose single AGRINT group 1 (1 -> 3) has one
    limit record, *limit_line*; sistema.dat lists the pair so the group has a line."""
    agrint_path = tmp_path / "agrint.dat"
    agrint_path.write_text(
        "AGRUPAMENTOS DE INTERCAMBIO\n"
        " #AG A   B   COEF\n"
        " XXX XXX XXX XX.XXXX\n"
        "   1   1   3  1.0000\n"
        " 999\n"
        "LIMITES POR GRUPO\n"
        "  #AG MI ANOI MF ANOF LIM_P1  LIM_P2  LIM_P3\n"
        " XXX  XX XXXX XX XXXX XXXXXX. XXXXXX. XXXXXX.\n"
        f"{limit_line}\n"
        " 999\n",
        encoding="latin-1",
    )
    sistema = MagicMock()
    sistema.limites_intercambio = pd.DataFrame(
        [
            {
                "submercado_de": 1,
                "submercado_para": 3,
                "flag": 0,
                "sentido": 0,
                "data": datetime(2020, 1, 1),
                "valor": 1000.0,
            }
        ]
    )
    return make_case(
        make_nw_files(tmp_path, agrint=agrint_path),
        dger=_make_dger_mock_for_agrint(),
        patamar=MagicMock(numero_patamares=3),
        sistema=sistema,
    )


class TestAgrintBlankLimitColumns:
    def test_blank_block_repeats_the_previous_block_limit(self, tmp_path: Path) -> None:
        case = _three_block_agrint_case(
            tmp_path, "   1   1 2020 12 2020  10000.   9000."
        )
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])

        result = convert_agrint_constraints(case, id_map)

        assert result is not None
        bounds = result.bounds.to_pandas()
        first_stage = bounds[bounds["stage_id"] == 0].sort_values("block_id")
        assert list(first_stage["bound_upper"]) == [10000.0, 9000.0, 9000.0]

    def test_blank_first_block_is_a_parse_error(self, tmp_path: Path) -> None:
        case = _three_block_agrint_case(
            tmp_path, "   1   1 2020 12 2020           9000.   8000."
        )
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])

        with pytest.raises(FieldParseError, match="group 1 has no first block limit"):
            convert_agrint_constraints(case, id_map)


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
        r = VminopResult({"constraints": [1, 2]}, bounds, [3, 4], {"p": [1.0]})
        assert r.constraints_dict == {"constraints": [1, 2]}
        assert r.bounds is bounds
        assert r.referenced_hydro_ids == [3, 4]
        assert r.per_stage_parameters == {"p": [1.0]}
        assert r[2] == [3, 4]

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


def _make_vminop_case(
    tmp_path: Path,
    *,
    curve: list[tuple[datetime, float]],
    penalty: float = 1488.0,
    num_anos_pos_estudo: int = 0,
    seasonalize: bool = False,
) -> tuple[NewaveCase, NewaveIdMap]:
    """REE 1 (SUDESTE) over the three-plant monthly-reservoir cascade."""
    cadastro = _make_cadastro()
    cadastro["tipo_regulacao"] = "M"

    dger = MagicMock()
    dger.mes_inicio_estudo = 1
    dger.ano_inicio_estudo = 2020
    dger.num_anos_estudo = 1
    dger.num_anos_pos_estudo = num_anos_pos_estudo
    dger.curva_aversao = 1

    curva = MagicMock()
    curva.curva_seguranca = pd.DataFrame(
        {
            "codigo_ree": [1] * len(curve),
            "data": [d for d, _ in curve],
            "valor": [v for _, v in curve],
        }
    )
    curva.configuracoes_penalizacao = [0, 11, 1 if seasonalize else 0]
    curva.custos_penalidades = pd.DataFrame(
        {"codigo_ree": [1], "penalidade": [penalty]}
    )

    confhd = MagicMock()
    confhd.usinas = _make_confhd_df()
    hidr = MagicMock()
    hidr.cadastro = cadastro
    ree = MagicMock()
    ree.rees = pd.DataFrame({"codigo": [1], "nome": ["SUDESTE"]})

    case = make_case(
        tmp_path, dger=dger, curva=curva, confhd=confhd, hidr=hidr, ree=ree
    )
    id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1, 2, 3], thermal_codes=[])
    return case, id_map


class TestVminopStoredEnergy:
    """VminOP bounds useful stored energy (MWh) with novomodelo's MWh parameters,
    one constraint per REE whose per_unit slack is a price per MWh."""

    def test_expression_bounds_useful_energy_by_curve_fraction(
        self, tmp_path: Path
    ) -> None:
        case, id_map = _make_vminop_case(tmp_path, curve=[(datetime(2020, 1, 1), 30.0)])

        result = convert_vminop_constraints(case, id_map)

        assert result is not None
        constraint = result.constraints_dict["constraints"][0]
        assert constraint["expression"] == (
            "@rho_acum_mwh_h0 * hydro_useful_volume_final(0)"
            " + @rho_acum_mwh_h1 * hydro_useful_volume_final(1)"
            " + @rho_acum_mwh_h2 * hydro_useful_volume_final(2)"
            " >= @pct_vminop_ree1 * (@earm_max_mwh_h0 + @earm_max_mwh_h1"
            " + @earm_max_mwh_h2)"
        )
        assert result.referenced_hydro_ids == [0, 1, 2]

    def test_one_constraint_per_ree_bounded_on_every_curve_stage(
        self, tmp_path: Path
    ) -> None:
        case, id_map = _make_vminop_case(
            tmp_path,
            curve=[(datetime(2020, m, 1), 20.0 + m) for m in range(1, 13)],
        )

        result = convert_vminop_constraints(case, id_map)

        assert result is not None
        [constraint] = result.constraints_dict["constraints"]
        assert constraint["name"] == "VminOP_SUDESTE"
        rows = result.bounds.to_pylist()
        assert [(r["constraint_id"], r["stage_id"]) for r in rows] == [
            (constraint["id"], s) for s in range(12)
        ]
        assert all(r["bound_lower"] == 0.0 for r in rows)
        assert all(r["bound_upper"] is None and r["block_id"] is None for r in rows)

    def test_penalty_is_a_per_unit_price_per_mwh(self, tmp_path: Path) -> None:
        case, id_map = _make_vminop_case(
            tmp_path,
            curve=[(datetime(2020, m, 1), 20.0) for m in (1, 2)],
            penalty=1488.0,
        )

        result = convert_vminop_constraints(case, id_map)

        assert result is not None
        [constraint] = result.constraints_dict["constraints"]
        assert constraint["slack"] == {
            "enabled": True,
            "penalty": 1488.0,
            "penalty_basis": "per_unit",
        }

    def test_curve_fraction_parameter_covers_every_stage(self, tmp_path: Path) -> None:
        case, id_map = _make_vminop_case(
            tmp_path,
            curve=[(datetime(2020, 1, 1), 30.0), (datetime(2020, 12, 1), 45.0)],
            num_anos_pos_estudo=1,
        )

        result = convert_vminop_constraints(case, id_map)

        assert result is not None
        values = result.per_stage_parameters["pct_vminop_ree1"]
        assert len(values) == 24
        # Percent becomes a fraction; uncovered in-study months stay 0 (no
        # constraint there); the post-study tail freezes at December.
        assert values[0] == pytest.approx(0.30)
        assert values[1] == 0.0
        assert values[11] == pytest.approx(0.45)
        assert values[12:] == pytest.approx([0.45] * 12)
        stages = [r["stage_id"] for r in result.bounds.to_pylist()]
        assert stages == [0, 11, *range(12, 24)]

    def test_seasonalized_tail_repeats_the_study_year(self, tmp_path: Path) -> None:
        case, id_map = _make_vminop_case(
            tmp_path,
            curve=[(datetime(2020, m, 1), float(m)) for m in range(1, 13)],
            num_anos_pos_estudo=1,
            seasonalize=True,
        )

        result = convert_vminop_constraints(case, id_map)

        assert result is not None
        values = result.per_stage_parameters["pct_vminop_ree1"]
        assert values[12:] == pytest.approx([m / 100.0 for m in range(1, 13)])


def _make_coupled_vminop_case(tmp_path: Path) -> tuple[NewaveCase, NewaveIdMap]:
    """REE 1 (UP -> CUT, terminal) coupled to REE 2 through FICT.UP/FICT.CUT -> DOWN.

    FICT.UP stores less than its twin UP; the downstream REE's bound still
    uses UP's useful volume, since the twin does not operate.
    """
    codes = [1, 2, 3, 4, 5]
    cadastro = pd.DataFrame(
        {
            "nome_usina": ["UP", "CUT", "FICT.UP", "FICT.CUT", "DOWN"],
            "produtibilidade_especifica": [0.01, 0.02, 0.0, 0.0, 0.03],
            "volume_minimo": [100.0, 200.0, 100.0, 200.0, 300.0],
            "volume_maximo": [1000.0, 2000.0, 600.0, 2000.0, 3000.0],
            "volume_referencia": [500.0, 1000.0, 500.0, 1000.0, 1500.0],
            "canal_fuga_medio": [200.0, 300.0, 200.0, 300.0, 400.0],
            "tipo_perda": [0] * 5,
            "perdas": [0.0] * 5,
            "tipo_regulacao": ["M"] * 5,
            "a0_volume_cota": [300.0, 400.0, 300.0, 400.0, 500.0],
            "a1_volume_cota": [0.1, 0.05, 0.1, 0.05, 0.02],
            "a2_volume_cota": [0.0] * 5,
            "a3_volume_cota": [0.0] * 5,
            "a4_volume_cota": [0.0] * 5,
        },
        index=pd.Index(codes, name="codigo_usina"),
    )
    confhd_df = pd.DataFrame(
        {
            "codigo_usina": codes,
            "nome_usina": list(cadastro["nome_usina"]),
            "usina_existente": ["EX"] * 5,
            "codigo_usina_jusante": [2, 0, 4, 5, 0],
            "ree": [1, 1, 2, 2, 2],
            "posto": [1, 2, 1, 2, 5],
            "volume_inicial_percentual": [50.0] * 5,
        }
    )

    dger = MagicMock()
    dger.mes_inicio_estudo = 1
    dger.ano_inicio_estudo = 2020
    dger.num_anos_estudo = 1
    dger.num_anos_pos_estudo = 0
    dger.curva_aversao = 1

    curva = MagicMock()
    curva.curva_seguranca = pd.DataFrame(
        {
            "codigo_ree": [1, 2],
            "data": [datetime(2020, 1, 1)] * 2,
            "valor": [30.0, 40.0],
        }
    )
    curva.configuracoes_penalizacao = [0, 11, 0]
    curva.custos_penalidades = pd.DataFrame(
        {"codigo_ree": [1, 2], "penalidade": [1488.0, 1488.0]}
    )

    confhd = MagicMock()
    confhd.usinas = confhd_df
    hidr = MagicMock()
    hidr.cadastro = cadastro
    ree = MagicMock()
    ree.rees = pd.DataFrame({"codigo": [1, 2], "nome": ["SUDESTE", "NORDESTE"]})

    case = make_case(
        tmp_path, dger=dger, curva=curva, confhd=confhd, hidr=hidr, ree=ree
    )
    id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1, 2, 5], thermal_codes=[])
    return case, id_map


class TestVminopReeCoupling:
    """A reservoir cascade the source model cuts between two REEs (e.g. TRES
    MARIAS -> FICT.TRES MA -> SOBRADINHO) counts whole in the reservoir's own
    REE, as the source model's stored energy does."""

    def _convert(self, tmp_path: Path):
        case, id_map = _make_coupled_vminop_case(tmp_path)
        result = convert_vminop_constraints(case, id_map)
        assert result is not None
        return {c["name"]: c for c in result.constraints_dict["constraints"]}, result

    def test_own_ree_counts_the_whole_cascade_below_the_cut(
        self, tmp_path: Path
    ) -> None:
        by_name, _ = self._convert(tmp_path)

        assert by_name["VminOP_SUDESTE"]["expression"] == (
            "@rho_acum_mwh_h0 * hydro_useful_volume_final(0)"
            " + @rho_acum_mwh_h1 * hydro_useful_volume_final(1)"
            " >= @pct_vminop_ree1 * (@earm_max_mwh_h0 + @earm_max_mwh_h1)"
        )

    def test_downstream_ree_counts_only_its_own_reservoirs(
        self, tmp_path: Path
    ) -> None:
        by_name, result = self._convert(tmp_path)

        assert by_name["VminOP_NORDESTE"]["expression"] == (
            "@rho_acum_mwh_h2 * hydro_useful_volume_final(2)"
            " >= @pct_vminop_ree2 * @earm_max_mwh_h2"
        )
        assert result.referenced_hydro_ids == [0, 1, 2]

    def test_only_the_curve_fraction_is_a_per_stage_parameter(
        self, tmp_path: Path
    ) -> None:
        _, result = self._convert(tmp_path)

        assert set(result.per_stage_parameters) == {
            "pct_vminop_ree1",
            "pct_vminop_ree2",
        }


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
    re_path = tmp_path / "restricao-eletrica.csv"
    re_path.write_text(
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
        make_nw_files(tmp_path, restricao_eletrica=re_path),
        dger=dger,
        sistema=sistema,
        ree=_valid_ree_reader(),
    )
    id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[10], thermal_codes=[])

    result = convert_electric_constraints(case, id_map)

    assert result is not None
    df = result[1].to_pandas()
    assert set(df[df["bound_upper"].notna()]["stage_id"]) == {0, 1, 2}
    assert set(df[df["bound_lower"].notna()]["stage_id"]) == {0, 1, 2}


# ---------------------------------------------------------------------------
# _cascade_sum cycle detection
# ---------------------------------------------------------------------------


def test_cascade_sum_detects_cycle() -> None:
    """A cycle in the downstream map raises BridgeError with the path."""
    from novomodelo_bridge.core.errors import BridgeError

    # Crafted cycle: 1 → 2 → 1
    downstream_map: dict[int, int | None] = {1: 2, 2: 1}
    values = {1: 10.0, 2: 20.0}

    with pytest.raises(BridgeError, match=r"Cascade cycle detected.*1.*2.*1"):
        _cascade_sum(downstream_map, values)


def test_cascade_sum_cycle_path_is_ordered_and_excludes_the_approach() -> None:
    """The message names the cycle in flow order, without the plant above it."""
    from novomodelo_bridge.core.errors import BridgeError

    downstream_map: dict[int, int | None] = {5: 1, 1: 2, 2: 3, 3: 1}

    with pytest.raises(BridgeError, match=r"detected: 1 → 2 → 3 → 1\."):
        _cascade_sum(downstream_map, {})


def test_cascade_sum_no_cycle() -> None:
    """Normal cascade without cycle works correctly."""
    # Linear cascade: 1 → 2 → 3 → None (sea sink)
    downstream_map: dict[int, int | None] = {1: 2, 2: 3, 3: None}
    values = {1: 10.0, 2: 20.0, 3: 30.0}

    result = _cascade_sum(downstream_map, values)

    # result[3] = 30, result[2] = 20 + 30 = 50, result[1] = 10 + 50 = 60
    assert result == {1: 60.0, 2: 50.0, 3: 30.0}
