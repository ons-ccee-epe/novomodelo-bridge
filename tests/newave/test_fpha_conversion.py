"""Tests for FPHA emission in the hydro converters.

FPHA is gated on the source model's ``dger`` ``funcao_producao_uhe == 0``
(:attr:`NewaveCase.fpha_enabled`). When on, reservoir plants with storage swing are
emitted as ``model: "fpha"`` so novomodelo fits the production function; when off, every
plant stays on the constant-productivity path (regression-preserving).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.errors import SourceFileError
from novomodelo_bridge.core.productivity import fpha_efficiency
from novomodelo_bridge.newave.converters.hydro import (
    _fpha_computed_config,
    _is_fpha_eligible,
    _parse_fpha_plane_reduction,
    convert_hydro_energy_productivity,
    convert_hydros,
    convert_production_models,
    fpha_eligible_codes,
)
from novomodelo_bridge.newave.id_map import NewaveIdMap
from tests.conftest import make_case, make_nw_files, modif_mock
from tests.newave.conftest import (
    _make_confhd_df,
    _make_ee_exph_mock,
    _make_hidr_cadastro,
    _make_prod_model_dger_mock,
    _make_ree_df,
)

# novomodelo's phi = K * eta * q * h_net with K = g/1000.
_K = 9.81e-3
# A physically realistic specific productivity -> eta in (0, 1].
_REALISTIC_RHO_ESP = 0.00892


def _reservoir_row(
    *,
    vmin: float = 100.0,
    vmax: float = 1000.0,
    rho_esp: float = _REALISTIC_RHO_ESP,
    a1: float = 0.1,
    maquinas: int = 2,
    potencia: float = 50.0,
    vazao: float = 100.0,
) -> pd.Series:
    return pd.Series(
        {
            "volume_minimo": vmin,
            "volume_maximo": vmax,
            "a0_volume_cota": 300.0,
            "a1_volume_cota": a1,
            "a2_volume_cota": 0.0,
            "a3_volume_cota": 0.0,
            "a4_volume_cota": 0.0,
            "produtibilidade_especifica": rho_esp,
            "numero_conjuntos_maquinas": 1,
            "maquinas_conjunto_1": maquinas,
            "potencia_nominal_conjunto_1": potencia,
            "vazao_nominal_conjunto_1": vazao,
        }
    )


class TestIsFphaEligible:
    def test_reservoir_with_storage_swing_is_eligible(self) -> None:
        assert _is_fpha_eligible(_reservoir_row()) is True

    def test_run_of_river_zero_storage_is_eligible(self) -> None:
        # Single-volume plant (vmax == vmin): novomodelo fits it via the single-volume
        # FPHA path (γ_V = 0), so it IS eligible given a poly + ρ_esp.
        assert _is_fpha_eligible(_reservoir_row(vmin=50.0, vmax=50.0)) is True

    def test_degenerate_polynomial_not_eligible(self) -> None:
        # All volume_cota coefficients zero -> no forebay curve to fit.
        row = _reservoir_row()
        for i in range(5):
            row[f"a{i}_volume_cota"] = 0.0
        assert _is_fpha_eligible(row) is False

    def test_zero_rho_esp_not_eligible(self) -> None:
        assert _is_fpha_eligible(_reservoir_row(rho_esp=0.0)) is False

    def test_zero_machines_not_eligible(self) -> None:
        assert _is_fpha_eligible(_reservoir_row(maquinas=0)) is False

    def test_zero_rated_power_not_eligible(self) -> None:
        assert _is_fpha_eligible(_reservoir_row(potencia=0.0)) is False

    def test_zero_rated_flow_not_eligible(self) -> None:
        assert _is_fpha_eligible(_reservoir_row(vazao=0.0)) is False


class TestFphaEfficiency:
    def test_realistic_rho_esp_maps_to_fraction(self) -> None:
        eta = fpha_efficiency(_REALISTIC_RHO_ESP, "USINA")
        assert eta == pytest.approx(_REALISTIC_RHO_ESP / _K)
        assert 0.0 < eta <= 1.0

    def test_unphysical_rho_esp_clamped_to_one(self) -> None:
        # rho_esp >> K would imply eta > 1; novomodelo requires (0, 1].
        assert fpha_efficiency(0.9, "USINA") == 1.0


class TestFphaComputedConfig:
    def _row(self, reg: str) -> pd.Series:
        return pd.Series(
            {
                "tipo_regulacao": reg,
                "volume_minimo": 430.0,
                "volume_maximo": 1781.0,
                "volume_referencia": 900.0,
            }
        )

    def test_monthly_reservoir_uses_full_storage_range(self) -> None:
        cfg = _fpha_computed_config(self._row("M"))
        assert cfg == {
            "source": "computed",
            "fitting_window": {"volume_min_hm3": 430.0, "volume_max_hm3": 1781.0},
        }

    def test_daily_plant_is_single_volume_at_vref(self) -> None:
        cfg = _fpha_computed_config(self._row("D"))
        assert cfg["fitting_window"] == {
            "volume_min_hm3": 900.0,
            "volume_max_hm3": 900.0,
        }

    def test_run_of_river_plant_is_single_volume_at_vref(self) -> None:
        cfg = _fpha_computed_config(self._row("S"))
        assert cfg["fitting_window"] == {
            "volume_min_hm3": 900.0,
            "volume_max_hm3": 900.0,
        }

    def test_turbined_fit_range_when_given(self) -> None:
        cfg = _fpha_computed_config(self._row("D"), turbined_fit_max_m3s=812.5)
        assert cfg["turbined_fit_max_m3s"] == 812.5
        assert "turbined_fit_max_m3s" not in _fpha_computed_config(self._row("D"))


class TestParseFphaPlaneReduction:
    def _case(self, tmp_path: Path, text: str | None):
        overrides: dict = {}
        if text is not None:
            path = tmp_path / "tratamento-fpha.csv"
            path.write_text(text, encoding="utf-8")
            overrides["tratamento_fpha"] = path
        return make_case(make_nw_files(tmp_path, **overrides))

    def test_angle_method(self, tmp_path: Path) -> None:
        case = self._case(
            tmp_path, "HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO; 1.0\n"
        )
        assert _parse_fpha_plane_reduction(case) == {
            "method": "angle",
            "tolerance_deg": 1.0,
        }

    def test_distance_method(self, tmp_path: Path) -> None:
        case = self._case(
            tmp_path,
            "HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-DISTANCIA-PADRAO; 0.002\n",
        )
        result = _parse_fpha_plane_reduction(case)
        assert result is not None
        assert result["method"] == "distance"
        assert result["tolerance_pct"] == 0.002
        assert result["n_samples"] >= 1

    def test_commented_lines_ignored(self, tmp_path: Path) -> None:
        case = self._case(
            tmp_path,
            "&HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-DISTANCIA-PADRAO; 0.002\n"
            "HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO; 1.0\n",
        )
        assert _parse_fpha_plane_reduction(case) == {
            "method": "angle",
            "tolerance_deg": 1.0,
        }

    def test_no_file_returns_none(self, tmp_path: Path) -> None:
        assert _parse_fpha_plane_reduction(self._case(tmp_path, None)) is None


class TestFphaConverters:
    """Plant 1 = reservoir (FPHA-eligible); plant 2 = non-FPHA.

    Plant 2 has its volume→cota polynomial zeroed (no forebay curve), so it
    stays on the constant-productivity path — the remaining way a plant is
    non-FPHA now that run-of-river plants are eligible.
    """

    @pytest.fixture(autouse=True)
    def natural_means(self):
        with patch(
            "novomodelo_bridge.newave.converters.hydro.productivity.natural_monthly_means",
            return_value={},
        ) as mock:
            yield mock

    def _id_map(self) -> NewaveIdMap:
        return NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])

    def _case(
        self,
        tmp_path: Path,
        *,
        fpha: bool = True,
        tratamento: str | None = None,
        volref: dict[int, dict[int, float]] | None = None,
        modif: MagicMock | None = None,
        confhd: pd.DataFrame | None = None,
        exph: MagicMock | None = None,
        plant_1: dict[str, object] | None = None,
        plant_2: dict[str, object] | None = None,
        polinjus: MagicMock | None = None,
        volumes_referencia: MagicMock | None = None,
    ):
        cadastro = _make_hidr_cadastro().copy()
        # Plant 1: realistic specific productivity -> FPHA.
        cadastro.loc[1, "produtibilidade_especifica"] = _REALISTIC_RHO_ESP
        cadastro.loc[1, "queda_nominal_conjunto_1"] = 300.0
        for column, value in (plant_1 or {}).items():
            cadastro.loc[1, column] = value
        # Plant 2: zero the volume->cota polynomial (no forebay) -> non-FPHA.
        for i in range(5):
            cadastro.loc[2, f"a{i}_volume_cota"] = 0.0
        cadastro.loc[2, "produtibilidade_especifica"] = 0.0088
        for column, value in (plant_2 or {}).items():
            cadastro.loc[2, column] = value

        mock_hidr = MagicMock()
        mock_hidr.cadastro = cadastro
        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df() if confhd is None else confhd
        mock_ree = MagicMock()
        mock_ree.rees = _make_ree_df()
        dger = _make_prod_model_dger_mock()
        # dger.dat: =0 FPHA, =1 LINEAR (constant productivity).
        dger.funcao_producao_uhe = 0 if fpha else 1

        overrides: dict = {}
        if tratamento is not None:
            path = tmp_path / "tratamento-fpha.csv"
            path.write_text(tratamento, encoding="utf-8")
            overrides["tratamento_fpha"] = path
        if modif is not None:
            overrides["modif"] = tmp_path / "modif.dat"

        parsed: dict = {
            "hidr": mock_hidr,
            "confhd": mock_confhd,
            "ree": mock_ree,
            "dger": dger,
            "exph": exph,
        }
        if modif is not None:
            parsed["modif"] = modif
        if polinjus is not None:
            parsed["polinjus"] = polinjus
        if volumes_referencia is not None:
            parsed["volumes_referencia"] = volumes_referencia
        if volref is not None:
            rows = [
                {"codigo_usina": code, "mes": month, "valor": value}
                for code, months in volref.items()
                for month, value in months.items()
            ]
            mock_vs = MagicMock()
            mock_vs.volumes = pd.DataFrame(rows)
            parsed["volref_saz"] = mock_vs

        return make_case(make_nw_files(tmp_path, **overrides), **parsed)

    def test_eligible_codes_only_reservoir(self, tmp_path: Path) -> None:
        assert fpha_eligible_codes(self._case(tmp_path)) == {1}

    def test_eligible_codes_empty_when_fpha_off(self, tmp_path: Path) -> None:
        assert fpha_eligible_codes(self._case(tmp_path, fpha=False)) == set()

    def test_modif_nummaq_zero_excludes_and_reports(self, tmp_path: Path) -> None:
        nummaq_rec = MagicMock()
        type(nummaq_rec).__name__ = "NUMMAQ"
        nummaq_rec.conjunto = 1
        nummaq_rec.numero_maquinas = 0
        modif = modif_mock((1, [nummaq_rec]))

        with dx.collect() as collected:
            codes = fpha_eligible_codes(self._case(tmp_path, modif=modif))

        assert codes == set()
        [diagnostic] = [d for d in collected if d.code == "fpha-zero-capacity"]
        assert diagnostic.severity is Severity.INFO
        assert diagnostic.table is not None
        assert [row[:2] for row in diagnostic.table.rows] == [["USINA_A", 1]]

    def test_expanding_plant_is_eligible_on_its_declared_configuration(
        self, tmp_path: Path
    ) -> None:
        # MODIF leaves plant 1 with no machine at the study start; the two exph
        # entries give it the 400 MW hydros.json declares, so it stays FPHA.
        nummaq_rec = MagicMock()
        type(nummaq_rec).__name__ = "NUMMAQ"
        nummaq_rec.conjunto = 1
        nummaq_rec.numero_maquinas = 0
        modif = modif_mock((1, [nummaq_rec]))
        confhd = _make_confhd_df()
        confhd["usina_existente"] = ["EE", "EX"]
        case = self._case(
            tmp_path,
            modif=modif,
            confhd=confhd,
            exph=_make_ee_exph_mock(entry="2025-07-01", code=1, conjunto=1),
        )

        with dx.collect() as collected:
            codes = fpha_eligible_codes(case)

        assert codes == {1}
        assert [d for d in collected if d.code == "fpha-zero-capacity"] == []
        hydros = convert_hydros(case, self._id_map())["hydros"]
        assert hydros[0]["generation"]["max_generation_mw"] == pytest.approx(400.0)

    def test_hydros_reservoir_is_fpha_with_efficiency(self, tmp_path: Path) -> None:
        hydros = convert_hydros(self._case(tmp_path), self._id_map())["hydros"]
        reservoir = next(h for h in hydros if h["id"] == 0)
        assert reservoir["generation"]["model"] == "fpha"
        assert reservoir["efficiency"] == {
            "type": "constant",
            "value": pytest.approx(_REALISTIC_RHO_ESP / _K),
        }
        # The fpha path also emits the mandatory mirror unit group.
        assert len(reservoir["unit_groups"]) == 1

    def test_hydros_non_fpha_plant_stays_constant(self, tmp_path: Path) -> None:
        hydros = convert_hydros(self._case(tmp_path), self._id_map())["hydros"]
        non_fpha = next(h for h in hydros if h["id"] == 1)
        assert non_fpha["generation"]["model"] == "constant_productivity"
        assert non_fpha["efficiency"] is None
        # The constant-productivity path also emits the mirror unit group.
        assert len(non_fpha["unit_groups"]) == 1

    def test_hydros_fpha_off_all_constant(self, tmp_path: Path) -> None:
        hydros = convert_hydros(self._case(tmp_path, fpha=False), self._id_map())[
            "hydros"
        ]
        assert all(h["generation"]["model"] == "constant_productivity" for h in hydros)
        assert all(h["efficiency"] is None for h in hydros)

    def test_production_models_reservoir_fpha(self, tmp_path: Path) -> None:
        result = convert_production_models(self._case(tmp_path), self._id_map())
        models = {
            m["hydro_id"]: m["stage_ranges"][0] for m in result["production_models"]
        }
        assert models[0]["model"] == "fpha"
        # Plant 1 is tipo_regulacao "M" (vmin=100, vmax=1000) -> multi-volume window.
        assert models[0]["fpha_config"]["fitting_window"] == {
            "volume_min_hm3": 100.0,
            "volume_max_hm3": 1000.0,
        }
        # No volumes-referencia.csv: the initial storage, 50 % of [100, 1000].
        assert models[0]["reference_volume"] == {"volume_hm3": 550.0}
        assert "productivity_mw_per_m3s" not in models[0]
        assert models[1]["model"] == "constant_productivity"

    def test_production_models_spillage_range_from_the_posto_natural_means(
        self, tmp_path: Path, natural_means: MagicMock
    ) -> None:
        natural_means.return_value = {1: [400.0] * 11 + [900.0], 2: [50.0] * 12}
        result = convert_production_models(self._case(tmp_path), self._id_map())
        models = {
            m["hydro_id"]: m["stage_ranges"][0] for m in result["production_models"]
        }
        assert models[0]["fpha_config"]["spillage_fit_max_m3s"] == 1800.0
        assert "fpha_config" not in models[1]

    def test_production_models_daily_plant_turbined_range_at_its_held_volume(
        self, tmp_path: Path
    ) -> None:
        # Held at 550 hm³: 300 + 0.1 · 550 - 50 = 305 m net head. Each of the 4
        # machines swallows 222.2 · (305 / 300)^0.5 m³/s, but its 200 MW
        # generator caps it at 200 / (ρ_esp · 305).
        case = self._case(
            tmp_path, plant_1={"tipo_regulacao": "D", "queda_nominal_conjunto_1": 300.0}
        )
        result = convert_production_models(case, self._id_map())
        fpha = result["production_models"][0]["stage_ranges"][0]
        per_machine = min(
            222.2 * (305.0 / 300.0) ** 0.5, 200.0 / (_REALISTIC_RHO_ESP * 305.0)
        )
        assert fpha["fpha_config"]["turbined_fit_max_m3s"] == pytest.approx(
            4 * per_machine
        )

    @staticmethod
    def _polinjus(families: dict[float, tuple[float, float]]) -> MagicMock:
        """Plant 1's tailrace families, ``{downstream reference level: (a0, a1)}``,
        each one segment ``a0 + a1 · outflow``."""
        polinjus = MagicMock()
        polinjus.hidreletrica_curvajusante.return_value = pd.DataFrame(
            {
                "codigo_usina": [1] * len(families),
                "indice_familia": list(range(1, len(families) + 1)),
                "nivel_montante_referencia": list(families),
            }
        )
        polinjus.hidreletrica_curvajusante_polinomio_segmento.return_value = (
            pd.DataFrame(
                {
                    "codigo_usina": [1] * len(families),
                    "indice_familia": list(range(1, len(families) + 1)),
                    "indice_polinomio": [1] * len(families),
                    "limite_inferior_vazao_jusante": [0.0] * len(families),
                    "limite_superior_vazao_jusante": [1.0e6] * len(families),
                    "coeficiente_a0": [a0 for a0, _ in families.values()],
                    "coeficiente_a1": [a1 for _, a1 in families.values()],
                    "coeficiente_a2": [0.0] * len(families),
                    "coeficiente_a3": [0.0] * len(families),
                    "coeficiente_a4": [0.0] * len(families),
                }
            )
        )
        return polinjus

    def _turbined_fit_max(self, case) -> float | None:
        result = convert_production_models(case, self._id_map())
        config = result["production_models"][0]["stage_ranges"][0]["fpha_config"]
        return config.get("turbined_fit_max_m3s")

    def test_production_models_turbined_range_settles_on_the_tailrace_curve(
        self, tmp_path: Path
    ) -> None:
        # The 800 MW generators never bind, so each turbine swallows
        # 222.2 · (h / 300)^0.5 with the tailrace rising 0.01 m per m³/s.
        case = self._case(
            tmp_path,
            plant_1={
                "tipo_regulacao": "D",
                "queda_nominal_conjunto_1": 300.0,
                "potencia_nominal_conjunto_1": 800.0,
            },
            polinjus=self._polinjus({float("nan"): (50.0, 0.01)}),
        )
        flow = self._turbined_fit_max(case)
        assert flow is not None
        assert flow == pytest.approx(
            4 * 222.2 * ((355.0 - (50.0 + 0.01 * flow)) / 300.0) ** 0.5, abs=1e-2
        )

    def _downstream_case(
        self, tmp_path: Path, *, hidr_downstream: object, confhd_downstream: object
    ):
        """Plant 1 ('D', held at 550 hm³) over families at 40 m and 50 m; plant 2
        ('M') sits at 300 + 0.1 · 50 = 305 m, halfway between them."""
        confhd = _make_confhd_df()
        confhd["codigo_usina_jusante"] = [confhd_downstream, pd.NA]
        return self._case(
            tmp_path,
            confhd=confhd,
            plant_1={
                "tipo_regulacao": "D",
                "potencia_nominal_conjunto_1": 800.0,
                "codigo_usina_jusante": hidr_downstream,
            },
            plant_2={
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "queda_nominal_conjunto_1": 300.0,
                "queda_nominal_conjunto_2": 300.0,
            },
            polinjus=self._polinjus({300.0: (40.0, 0.0), 310.0: (50.0, 0.0)}),
        )

    def test_production_models_turbined_range_reads_the_hidr_downstream_at_its_minimum(
        self, tmp_path: Path
    ) -> None:
        # A 45 m tailrace under the 355 m forebay: a 310 m head.
        case = self._downstream_case(
            tmp_path, hidr_downstream=2, confhd_downstream=pd.NA
        )
        assert self._turbined_fit_max(case) == pytest.approx(
            4 * 222.2 * (310.0 / 300.0) ** 0.5
        )

    @pytest.mark.parametrize("hidr_downstream", [pd.NA, 0, 99])
    def test_production_models_turbined_range_lowest_family_without_a_listed_hidr_downstream(
        self, tmp_path: Path, hidr_downstream: object
    ) -> None:
        # Not the confhd link: with no hidr downstream in confhd the lowest
        # family (40 m) gives a 315 m head.
        case = self._downstream_case(
            tmp_path, hidr_downstream=hidr_downstream, confhd_downstream=2
        )
        assert self._turbined_fit_max(case) == pytest.approx(
            4 * 222.2 * (315.0 / 300.0) ** 0.5
        )

    def test_production_models_unsettled_turbined_range_is_reported(
        self, tmp_path: Path
    ) -> None:
        # A tailrace rising 1 m per m³/s makes the flow alternate between two
        # values instead of settling.
        case = self._case(
            tmp_path,
            plant_1={"tipo_regulacao": "D", "queda_nominal_conjunto_1": 300.0},
            polinjus=self._polinjus({float("nan"): (50.0, 1.0)}),
        )
        with dx.collect() as collected:
            convert_production_models(case, self._id_map())
        [diagnostic] = [
            d for d in collected if d.code == "fpha-turbined-fit-range-unconverged"
        ]
        assert diagnostic.severity is Severity.WARNING
        assert diagnostic.table is not None
        assert diagnostic.table.rows == [[1, "USINA_A"]]

    def test_production_models_monthly_reservoir_turbined_range_at_its_reference_volume(
        self, tmp_path: Path
    ) -> None:
        # Reference 1000 hm³: a 400 - 50 = 350 m turbine head. The generator head
        # is the mean forebay over [100, 1000], 355 m, down to 50 m: 305 m, under
        # which the 800 MW generators do not bind.
        case = self._case(
            tmp_path,
            plant_1={"potencia_nominal_conjunto_1": 800.0},
            volumes_referencia=self._volumes_referencia(
                [(1, "2025-01-01", "2029-12-01", 1000.0)]
            ),
        )
        assert self._turbined_fit_max(case) == pytest.approx(
            4 * 222.2 * (350.0 / 300.0) ** 0.5
        )

    def test_production_models_monthly_reservoir_generator_head_is_the_mean_forebay(
        self, tmp_path: Path
    ) -> None:
        # 200 MW generators bind at 200 / (ρ_esp · 305) under the mean forebay,
        # whatever the reference volume.
        case = self._case(
            tmp_path,
            volumes_referencia=self._volumes_referencia(
                [(1, "2025-01-01", "2029-12-01", 1000.0)]
            ),
        )
        assert self._turbined_fit_max(case) == pytest.approx(
            4 * 200.0 / (_REALISTIC_RHO_ESP * 305.0)
        )

    def test_production_models_expanding_plant_turbined_range_on_its_declared_machines(
        self, tmp_path: Path
    ) -> None:
        # MODIF leaves no machine at the study start; the two exph machines are
        # the configuration hydros.json declares.
        nummaq_rec = MagicMock()
        type(nummaq_rec).__name__ = "NUMMAQ"
        nummaq_rec.conjunto = 1
        nummaq_rec.numero_maquinas = 0
        confhd = _make_confhd_df()
        confhd["usina_existente"] = ["EE", "EX"]
        case = self._case(
            tmp_path,
            modif=modif_mock((1, [nummaq_rec])),
            confhd=confhd,
            exph=_make_ee_exph_mock(entry="2025-07-01", code=1, conjunto=1),
            plant_1={"tipo_regulacao": "D", "queda_nominal_conjunto_1": 300.0},
        )
        result = convert_production_models(case, self._id_map())
        fpha = result["production_models"][0]["stage_ranges"][0]
        per_machine = min(
            222.2 * (305.0 / 300.0) ** 0.5, 200.0 / (_REALISTIC_RHO_ESP * 305.0)
        )
        assert fpha["fpha_config"]["turbined_fit_max_m3s"] == pytest.approx(
            2 * per_machine
        )

    def test_production_models_artificial_gauge_keeps_novomodelo_default_range(
        self, tmp_path: Path, natural_means: MagicMock
    ) -> None:
        confhd = _make_confhd_df()
        confhd.loc[0, "posto"] = 300
        natural_means.return_value = {300: [100.0] * 12}
        result = convert_production_models(
            self._case(tmp_path, confhd=confhd), self._id_map()
        )
        (fpha,) = [
            m["stage_ranges"][0]
            for m in result["production_models"]
            if m["stage_ranges"][0]["model"] == "fpha"
        ]
        assert "spillage_fit_max_m3s" not in fpha["fpha_config"]

    def test_production_models_plane_reduction_emitted(self, tmp_path: Path) -> None:
        result = convert_production_models(
            self._case(
                tmp_path,
                tratamento="HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO; 1.0\n",
            ),
            self._id_map(),
        )
        assert result["fpha_plane_reduction"] == {
            "method": "angle",
            "tolerance_deg": 1.0,
        }

    def test_production_models_no_plane_reduction_when_fpha_off(
        self, tmp_path: Path
    ) -> None:
        result = convert_production_models(
            self._case(
                tmp_path,
                fpha=False,
                tratamento="HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO; 1.0\n",
            ),
            self._id_map(),
        )
        assert "fpha_plane_reduction" not in result
        assert all(
            m["stage_ranges"][0]["model"] == "constant_productivity"
            for m in result["production_models"]
        )

    def test_energy_productivity_includes_fpha_plants(self, tmp_path: Path) -> None:
        # FPHA plants must keep an equivalent_productivity row: novomodelo's
        # energy-conversion build resolves their ρ_eq from this parquet override
        # (build_energy_and_templates feeds the VHA-geometry derivation an empty
        # map, so the parquet is the only working source). Excluding them makes
        # novomodelo fail at load with "cannot derive ρ_eq".
        table = convert_hydro_energy_productivity(self._case(tmp_path), self._id_map())
        rows = {r["hydro_id"]: r for r in table.to_pylist()}
        assert 0 in rows  # FPHA reservoir is NOT excluded
        assert rows[0]["equivalent_productivity_mw_per_m3s"] is not None
        assert rows[1]["equivalent_productivity_mw_per_m3s"] is not None

    @staticmethod
    def _volumes_referencia(
        rows: list[tuple[int, str, str, float]], tipo: int | None = 1
    ) -> MagicMock:
        """A ``volumes-referencia.csv`` reader: ``(code, first month, last month,
        hm³)`` rows and the ``VOLUME-REFERENCIAL-TIPO-PADRAO`` value (``None``
        for no record)."""
        reader = MagicMock()
        reader.volume_referencial_tipo_padrao.return_value = (
            pd.DataFrame()
            if tipo is None
            else pd.DataFrame({"tipo_referencia": [tipo]})
        )
        reader.volume_referencial_periodo.return_value = pd.DataFrame(
            {
                "codigo_usina": [row[0] for row in rows],
                "data_inicio": pd.to_datetime([row[1] for row in rows]),
                "data_fim": pd.to_datetime([row[2] for row in rows]),
                "volume_referencia": [row[3] for row in rows],
            }
        )
        return reader

    @staticmethod
    def _monthly_rows(
        code: int, volume: Callable[[int, int], float]
    ) -> list[tuple[int, str, str, float]]:
        """One row per month of the 2025-2029 study, ``volume(year, month)``."""
        return [
            (
                code,
                f"{year}-{month:02d}-01",
                f"{year}-{month:02d}-01",
                volume(year, month),
            )
            for year in range(2025, 2030)
            for month in range(1, 13)
        ]

    def _models(self, case) -> dict[int, dict]:
        return {
            m["hydro_id"]: m
            for m in convert_production_models(case, self._id_map())[
                "production_models"
            ]
        }

    def test_calendar_periodic_reference_is_seasonal(self, tmp_path: Path) -> None:
        case = self._case(
            tmp_path,
            plant_1={"potencia_nominal_conjunto_1": 800.0},
            volumes_referencia=self._volumes_referencia(
                self._monthly_rows(1, lambda _year, month: 100.0 + 50.0 * month)
            ),
        )
        entry = self._models(case)[0]
        assert entry["selection_mode"] == "seasonal"
        assert entry["default_model"] == "constant_productivity"
        seasons = {s["season_id"]: s for s in entry["seasons"]}
        assert sorted(seasons) == list(range(12))
        assert seasons[0]["reference_volume"] == {"volume_hm3": 150.0}
        assert seasons[11]["reference_volume"] == {"volume_hm3": 700.0}
        assert all(s["model"] == "fpha" for s in entry["seasons"])
        # The turbined fit range follows each season's reference volume.
        fit = [s["fpha_config"]["turbined_fit_max_m3s"] for s in entry["seasons"]]
        assert fit == sorted(fit)
        assert fit[0] < fit[-1]

    def test_non_periodic_reference_merges_runs_of_equal_stages(
        self, tmp_path: Path
    ) -> None:
        # 2025 at 400 hm³, every later stage at 800 hm³.
        case = self._case(
            tmp_path,
            volumes_referencia=self._volumes_referencia(
                [
                    (1, "2025-01-01", "2025-12-01", 400.0),
                    (1, "2026-01-01", "2029-12-01", 800.0),
                ]
            ),
        )
        entry = self._models(case)[0]
        assert entry["selection_mode"] == "stage_ranges"
        ranges = entry["stage_ranges"]
        assert [(r["start_stage_id"], r["end_stage_id"]) for r in ranges] == [
            (0, 11),
            (12, None),
        ]
        assert [r["reference_volume"] for r in ranges] == [
            {"volume_hm3": 400.0},
            {"volume_hm3": 800.0},
        ]

    def test_plant_without_rows_reads_its_hidr_reference_volume(
        self, tmp_path: Path
    ) -> None:
        case = self._case(
            tmp_path,
            plant_1={"volume_referencia": 700.0},
            volumes_referencia=self._volumes_referencia(
                [(2, "2025-01-01", "2029-12-01", 300.0)]
            ),
        )
        models = self._models(case)
        assert models[0]["stage_ranges"][0]["reference_volume"] == {"volume_hm3": 700.0}
        assert models[1]["stage_ranges"][0]["reference_volume"] == {"volume_hm3": 300.0}

    @pytest.mark.parametrize("tipo", [0, None])
    def test_rows_off_read_the_initial_storage(
        self, tmp_path: Path, tipo: int | None
    ) -> None:
        case = self._case(
            tmp_path,
            volumes_referencia=self._volumes_referencia(
                [(1, "2025-01-01", "2029-12-01", 900.0)], tipo=tipo
            ),
        )
        with dx.collect() as collected:
            models = self._models(case)
        # 50 % of [100, 1000] and 75 % of [50, 500].
        assert models[0]["stage_ranges"][0]["reference_volume"] == {"volume_hm3": 550.0}
        assert models[1]["stage_ranges"][0]["reference_volume"] == {"volume_hm3": 387.5}
        [diagnostic] = [
            d for d in collected if d.code == "fpha-reference-initial-storage"
        ]
        assert diagnostic.severity is Severity.INFO

    def test_daily_plant_ignores_its_rows(self, tmp_path: Path) -> None:
        case = self._case(
            tmp_path,
            plant_1={"tipo_regulacao": "D"},
            volumes_referencia=self._volumes_referencia(
                [(1, "2025-01-01", "2029-12-01", 900.0)]
            ),
        )
        entry = self._models(case)[0]["stage_ranges"][0]
        assert entry["reference_volume"] == {"volume_hm3": 550.0}

    def test_volref_saz_does_not_move_the_reference_volume(
        self, tmp_path: Path
    ) -> None:
        case = self._case(tmp_path, volref={1: {1: 200.0, 7: 5000.0}})
        entry = self._models(case)[0]
        assert entry["selection_mode"] == "stage_ranges"
        assert entry["stage_ranges"][0]["reference_volume"] == {"volume_hm3": 550.0}

    def test_plant_outside_the_fpha_declares_its_reference_volume(
        self, tmp_path: Path
    ) -> None:
        # Plant 2 is the backwater reference of an upstream FPHA plant.
        [entry] = self._models(self._case(tmp_path))[1]["stage_ranges"]
        assert entry["model"] == "constant_productivity"
        assert "fpha_config" not in entry
        assert entry["reference_volume"] == {"volume_hm3": 387.5}

    @pytest.mark.parametrize(
        ("tipo", "rows", "message"),
        [
            (2, [(1, "2025-01-01", "2029-12-01", 900.0)], "TIPO-PADRAO"),
            (1, [(1, "2025-01-01", "2029-11-01", 900.0)], "2029/12"),
            (
                1,
                [
                    (1, "2025-01-01", "2029-12-01", 900.0),
                    (1, "2027-03-01", "2027-03-01", 900.0),
                ],
                "2027/03",
            ),
        ],
        ids=["unknown-type", "uncovered-stage", "overlapping-rows"],
    )
    def test_rows_the_source_model_rejects_are_rejected(
        self,
        tmp_path: Path,
        tipo: int,
        rows: list[tuple[int, str, str, float]],
        message: str,
    ) -> None:
        case = self._case(
            tmp_path, volumes_referencia=self._volumes_referencia(rows, tipo=tipo)
        )
        with pytest.raises(SourceFileError, match=message):
            convert_production_models(case, self._id_map())

    def test_volref_not_consulted_when_fpha_off(self, tmp_path: Path) -> None:
        case = self._case(tmp_path, fpha=False, volref={1: {1: 200.0}})
        models = convert_production_models(case, self._id_map())["production_models"]
        # Linear case: volref ignored, every plant stays a single constant range.
        assert all(m["selection_mode"] == "stage_ranges" for m in models)
        assert all(
            m["stage_ranges"][0]["model"] == "constant_productivity" for m in models
        )
