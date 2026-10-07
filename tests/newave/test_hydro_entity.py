"""Unit tests for the source model hydro entity converter."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pandas as pd
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.newave.id_map import NewaveIdMap
from tests.conftest import hydro_with_group
from tests.newave.conftest import (
    _ee_expansion_case,
    _hydro_case,
    _make_ee_confhd_df,
    _make_ee_exph_mock,
    _make_hidr_cadastro,
    _make_hydro_dger_mock,
    _make_ne_cadastro,
    _make_ne_confhd_df,
    _make_ne_exph_mock,
    _make_penalid_df,
    _ne_filling_case,
    _ne_filling_id_map,
)


class TestConvertHydros:
    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1, 2],
            thermal_codes=[],
        )

    def test_potefe_reaches_the_declared_max_generation(self, tmp_path) -> None:
        """A permanent POTEFE override travels from modif.dat into hydros.json.

        Pins the wiring, not the arithmetic: every cadastro consumer reads the
        MODIF-corrected registry, so a converter that reached for
        ``case.hidr.cadastro`` directly would emit the registry power instead.
        """
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        potefe_rec = MagicMock()
        type(potefe_rec).__name__ = "POTEFE"
        potefe_rec.potencia = 100.0
        potefe_rec.conjunto = 1

        usina_rec = MagicMock()
        usina_rec.codigo = 1
        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.side_effect = lambda code: (
            [potefe_rec] if code == 1 else []
        )

        with dx.collect():
            base = convert_hydros(_hydro_case(tmp_path), self._make_id_map())["hydros"]
            overridden = convert_hydros(
                _hydro_case(tmp_path, modif=mock_modif), self._make_id_map()
            )["hydros"]

        # Plant 1 runs 4 machines in conjunto 1, declared at 200 MW each.
        assert base[0]["generation"]["max_generation_mw"] == pytest.approx(800.0)
        assert overridden[0]["generation"]["max_generation_mw"] == pytest.approx(400.0)
        assert overridden[1]["generation"]["max_generation_mw"] == pytest.approx(
            base[1]["generation"]["max_generation_mw"]
        )

    def test_returns_hydros_key(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        assert "hydros" in result

    def test_hydro_count_matches_existing_plants(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        assert len(result["hydros"]) == 2

    def test_min_outflow_switch_zeroes_every_plant(self, tmp_path) -> None:
        """``DESCONSIDERA VAZMIN = 1`` writes no static outflow floor, as the
        source model applies none, with one INFO diagnostic."""
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        cadastro = _make_hidr_cadastro()
        cadastro["vazao_minima_historica"] = [30.0, 0.0]
        on = _hydro_case(tmp_path, cadastro=cadastro)
        floors = {
            h["name"]: h["outflow"]["min_outflow_m3s"]
            for h in convert_hydros(on, self._make_id_map())["hydros"]
        }
        assert max(floors.values()) == pytest.approx(30.0)

        dger = _make_hydro_dger_mock()
        dger.desconsidera_vazao_minima = 1
        off = _hydro_case(tmp_path, cadastro=cadastro, dger=dger)
        with dx.collect() as collected:
            result = convert_hydros(off, self._make_id_map())
        assert all(h["outflow"]["min_outflow_m3s"] == 0.0 for h in result["hydros"])
        assert [d.code for d in collected] == ["dger-switch-off"]
        assert collected[0].severity is Severity.INFO

    def test_hydro_ids_are_zero_based_and_sorted(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        ids = [h["id"] for h in result["hydros"]]
        assert ids == sorted(ids)
        assert ids[0] == 0

    def test_hydro_has_required_fields(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        for h in result["hydros"]:
            assert "id" in h
            assert "name" in h
            assert "operational_start_date" in h
            assert "bus_id" not in h
            assert "reservoir" in h
            assert "min_storage_hm3" in h["reservoir"]
            assert "max_storage_hm3" in h["reservoir"]
            assert "outflow" in h
            assert "generation" in h
            assert h["generation"]["model"] == "constant_productivity"
            assert len(h["unit_groups"]) == 1
            group = h["unit_groups"][0]
            assert group.keys() == {
                "id",
                "name",
                "bus_id",
                "min_generation_mw",
                "max_generation_mw",
                "min_turbined_m3s",
                "max_turbined_m3s",
            }

    def test_run_of_river_S_storage_collapsed_to_volref(self, tmp_path) -> None:
        """``tipo_regulacao='S'`` (fio-d'água) collapses storage to the
        reference volume.

        The source model treats 'S' plants as run-of-river with no usable buffer:
        they operate at ``volume_referencia`` every stage, spilling the
        turbine-excess inflow. The converter must pin min==max==volume_referencia
        (not volume_minimo) so novomodelo doesn't store and shift that surplus across
        stages, matching the daily-regulation ('D') collapse.
        """
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        # USINA_A: Vmin 100, Vmax 1000, Vref 550.
        cadastro = _make_hidr_cadastro()
        cadastro.loc[1, "tipo_regulacao"] = "S"
        case = _hydro_case(tmp_path, cadastro=cadastro)

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        assert hydro_a["reservoir"]["min_storage_hm3"] == 550.0
        assert hydro_a["reservoir"]["max_storage_hm3"] == 550.0
        # 'M' plant unchanged (keeps its full range).
        hydro_b = next(h for h in result["hydros"] if h["name"] == "USINA_B")
        assert hydro_b["reservoir"]["min_storage_hm3"] == 50.0
        assert hydro_b["reservoir"]["max_storage_hm3"] == 500.0

    def test_run_of_river_S_without_volref_keeps_full_range(self, tmp_path) -> None:
        """When ``volume_referencia`` is absent/NaN, an 'S' plant is NOT collapsed.

        Without a reference volume there is no defined point to freeze the range
        at, so the converter leaves ``[volume_minimo, volume_maximo]`` untouched
        rather than guessing (e.g. pinning to Vmin)."""
        import numpy as np

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        cadastro = _make_hidr_cadastro()
        cadastro.loc[1, "tipo_regulacao"] = "S"  # USINA_A: Vmin 100, Vmax 1000
        cadastro.loc[1, "volume_referencia"] = np.nan
        case = _hydro_case(tmp_path, cadastro=cadastro)

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        assert hydro_a["reservoir"]["min_storage_hm3"] == 100.0
        assert hydro_a["reservoir"]["max_storage_hm3"] == 1000.0

    def test_cascade_downstream_linkage(self, tmp_path) -> None:
        """Plant 2 (code=2) is downstream of plant 1 (code=1)."""
        case = _hydro_case(tmp_path)
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        # USINA_A (code=1, novomodelo id=0) has no downstream.
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        assert hydro_a["downstream_id"] is None
        # USINA_B (code=2, novomodelo id=1) is downstream of USINA_A (novomodelo id=0).
        hydro_b = next(h for h in result["hydros"] if h["name"] == "USINA_B")
        assert hydro_b["downstream_id"] == 0

    def test_bus_id_matches_ree_subsystem(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        for h in result["hydros"]:
            # Both plants are in REE 1 -> subsystem 1 -> bus 0. bus_id now
            # lives on the mirror unit group, not top-level.
            assert h["unit_groups"][0]["bus_id"] == 0

    def test_generation_values_match_machine_sets(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        gen = hydro_a["generation"]
        # USINA_A: 1 set, 4 machines, 200 MW each, flow 222.2 each.
        assert gen["max_generation_mw"] == pytest.approx(4 * 200.0)
        assert gen["max_turbined_m3s"] == pytest.approx(4 * 222.2)
        # On novomodelo HEAD productivity lives in hydro_production_models.json,
        # not in the hydros.json generation block. ρ_esp surfaces as a
        # top-level optional field for novomodelo's energy-conversion pipeline.
        assert "productivity_mw_per_m3s" not in gen
        assert hydro_a["specific_productivity_mw_per_m3s_per_m"] == pytest.approx(0.9)

    def test_unit_group_mirrors_generation_on_every_plant(self, tmp_path) -> None:
        """novomodelo rule 41: the mirror group's four bounds equal the plant's own
        ``generation`` bounds verbatim, for every plant (not just one)."""
        case = _hydro_case(tmp_path)
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        for h in result["hydros"]:
            gen = h["generation"]
            group = h["unit_groups"][0]
            assert group["min_generation_mw"] == pytest.approx(gen["min_generation_mw"])
            assert group["max_generation_mw"] == pytest.approx(gen["max_generation_mw"])
            assert group["min_turbined_m3s"] == pytest.approx(gen["min_turbined_m3s"])
            assert group["max_turbined_m3s"] == pytest.approx(gen["max_turbined_m3s"])

    def test_output_never_regresses_to_the_0_12_shape(self, tmp_path) -> None:
        """The real converter output has no top-level ``bus_id`` and a
        non-empty ``unit_groups`` on every plant — the shape novomodelo 0.13's
        ``hydros.schema.json`` requires (decisions 13/14), never the 0.12
        shape asserted in ``TestLegacyHydroShapeRejectedBy013``."""
        case = _hydro_case(tmp_path)
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        for h in result["hydros"]:
            assert "bus_id" not in h
            assert h["unit_groups"]

    def test_schema_key_present(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        assert "$schema" in result
        assert "hydros.schema.json" in result["$schema"]

    def test_hydro_code_absent_in_hidr_raises_value_error(self, tmp_path) -> None:
        # Set up mocks but make the cadastro empty (no plants).
        case = _hydro_case(tmp_path, cadastro=pd.DataFrame())

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        with pytest.raises(ValueError, match="not found in hidr.dat"):
            convert_hydros(case, id_map)

    def test_hydraulic_losses_factor(self, tmp_path) -> None:
        """tipo_perda=1 and perdas=5.0 (%) -> hydraulic_losses factor dict."""
        cadastro = _make_hidr_cadastro().copy()
        cadastro["tipo_perda"] = 1
        cadastro["perdas"] = 5.0  # 5% — stored as percentage in hidr.dat

        case = _hydro_case(tmp_path, cadastro=cadastro)

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        for h in result["hydros"]:
            assert h["hydraulic_losses"] == {
                "type": "factor",
                "value": pytest.approx(0.05),  # 5% / 100 = 0.05
            }

    def test_hydraulic_losses_constant(self, tmp_path) -> None:
        """tipo_perda=2 and perdas=3.5 -> hydraulic_losses constant dict."""
        cadastro = _make_hidr_cadastro().copy()
        cadastro["tipo_perda"] = 2
        cadastro["perdas"] = 3.5

        case = _hydro_case(tmp_path, cadastro=cadastro)

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        for h in result["hydros"]:
            assert h["hydraulic_losses"] == {
                "type": "constant",
                "value_m": pytest.approx(3.5),
            }

    def test_hydraulic_losses_none_when_zero(self, tmp_path) -> None:
        """perdas=0.0 produces hydraulic_losses=None regardless of tipo_perda."""
        cadastro = _make_hidr_cadastro().copy()
        cadastro["tipo_perda"] = 1
        cadastro["perdas"] = 0.0

        case = _hydro_case(tmp_path, cadastro=cadastro)

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        for h in result["hydros"]:
            assert h["hydraulic_losses"] is None

    def test_evaporation_reference_volumes_from_volref_saz(self, tmp_path) -> None:
        """Plant with seasonal volref → reference_volumes_hm3 emitted as
        ``vmin + volref_saz[m]`` per calendar month."""
        # Only USINA_A (code=1) gets a non-zero seasonal row.
        # vmin_A=100, vmax_A=1000 → useful volumes 50..600 all inside the range.
        volref_df = pd.DataFrame(
            {
                "codigo_usina": [1] * 12,
                "nome_usina": ["USINA_A"] * 12,
                "mes": list(range(1, 13)),
                "valor": [50.0 * m for m in range(1, 13)],
            }
        )
        case = _hydro_case(
            tmp_path,
            volref_volumes=volref_df,
            volref_saz=tmp_path / "volref_saz.dat",
        )

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        hydro_b = next(h for h in result["hydros"] if h["name"] == "USINA_B")

        # USINA_A: reference_volumes_hm3 = vmin_A + useful → 150, 200, ..., 700.
        assert hydro_a["evaporation"] is not None
        assert hydro_a["evaporation"]["coefficients_mm"] == [1.5] * 12
        assert hydro_a["evaporation"]["reference_volumes_hm3"] == [
            100.0 + 50.0 * m for m in range(1, 13)
        ]
        # USINA_B has no row in volref_saz → reference_volumes_hm3 omitted.
        assert "reference_volumes_hm3" not in hydro_b["evaporation"]

    def test_evaporation_reference_volumes_absent_for_all_zero_row(
        self, tmp_path
    ) -> None:
        """All-zero volref_saz row is the source model's sentinel; novomodelo falls back to
        its mid-storage default, so reference_volumes_hm3 is NOT emitted."""
        volref_df = pd.DataFrame(
            {
                "codigo_usina": [1] * 12 + [2] * 12,
                "nome_usina": ["USINA_A"] * 12 + ["USINA_B"] * 12,
                "mes": list(range(1, 13)) * 2,
                "valor": [0.0] * 24,
            }
        )
        case = _hydro_case(
            tmp_path,
            volref_volumes=volref_df,
            volref_saz=tmp_path / "volref_saz.dat",
        )

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        for h in result["hydros"]:
            assert h["evaporation"] is not None
            assert "reference_volumes_hm3" not in h["evaporation"]

    def test_evaporation_reference_volumes_clamp_into_reservoir_range(
        self, tmp_path
    ) -> None:
        """Useful volumes larger than (vmax-vmin) get clamped to vmax — the
        novomodelo schema requires every reference volume in [min_storage,
        max_storage], so we never emit a value outside the reservoir
        bounds even when volref_saz has out-of-range data."""
        # USINA_A has useful=[100,200,...,1200]. Useful range is 900 so
        # values 1000, 1100, 1200 exceed vmax (1000). Expect clamping to vmax.
        volref_df = pd.DataFrame(
            {
                "codigo_usina": [1] * 12,
                "nome_usina": ["USINA_A"] * 12,
                "mes": list(range(1, 13)),
                "valor": [100.0 * m for m in range(1, 13)],
            }
        )
        case = _hydro_case(
            tmp_path,
            volref_volumes=volref_df,
            volref_saz=tmp_path / "volref_saz.dat",
        )

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        ref_volumes = hydro_a["evaporation"]["reference_volumes_hm3"]
        assert len(ref_volumes) == 12
        # m=1..9 → vmin + 100*m = 200..1000, all <= vmax=1000.
        # m=10..12 → would be 1100, 1200, 1300 → clamped to 1000.
        for v in ref_volumes:
            assert 100.0 <= v <= 1000.0
        # Last three months hit the cap.
        assert ref_volumes[-3:] == [1000.0, 1000.0, 1000.0]

    def test_teif_ip_derates_turbined_not_generation(self, tmp_path) -> None:
        """TEIF/IP availability derates ``max_turbined`` but NOT ``max_generation``.

        ``max_generation`` is the rated installed power ``Σ n·p_nom`` — the source
        model's FPHA ``GHmax`` (verified TUCURUI 7445, QUEBRA QUEIX 120), independent of
        the production function and *not* availability-derated. ``max_turbined`` is the
        head-corrected engolimento, which an unavailable unit cannot pass, so it carries
        the ``0.95 * 0.97`` availability factor (the source model's operational cap —
        verified QUEBRA QUEIX 113.01 m³/s == head-corrected engolimento).
        """
        cadastro = _make_hidr_cadastro().copy()
        # Override only USINA_A (code=1) with non-zero TEIF/IP.
        cadastro.loc[1, "teif"] = 5.0
        cadastro.loc[1, "ip"] = 3.0

        case = _hydro_case(tmp_path, cadastro=cadastro)

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        # max_generation = rated nameplate (4 machines * 200 MW = 800 MW), NOT
        # availability-derated: The source model's GHmax is the installed capacity.
        assert hydro_a["generation"]["max_generation_mw"] == pytest.approx(800.0)
        # max_turbined IS derated — an unavailable unit can't pass water either,
        # so the head-corrected engolimento carries the availability factor.
        assert hydro_a["generation"]["max_turbined_m3s"] == pytest.approx(
            4 * 222.2 * 0.95 * 0.97
        )
        # min_generation_mw must NOT be derated (it is zero here)
        assert hydro_a["generation"]["min_generation_mw"] == pytest.approx(0.0)

    def test_zero_teif_ip_no_derating(self, tmp_path) -> None:
        """TEIF=0% and IP=0% leaves max_generation_mw unchanged."""
        case = _hydro_case(tmp_path)
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        # teif=0, ip=0 -> factor = 1.0 -> no change from nominal 800 MW
        assert hydro_a["generation"]["max_generation_mw"] == pytest.approx(800.0)

    def test_nan_teif_treated_as_zero(self, tmp_path) -> None:
        """NaN teif is treated as 0.0 — no derating, no error."""
        cadastro = _make_hidr_cadastro().copy()
        cadastro.loc[1, "teif"] = float("nan")
        cadastro.loc[1, "ip"] = float("nan")

        case = _hydro_case(tmp_path, cadastro=cadastro)

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        # NaN treated as 0 -> factor = 1.0 -> no change from nominal 800 MW
        assert hydro_a["generation"]["max_generation_mw"] == pytest.approx(800.0)

    # --- FILLING phase emission -------------------------------

    def test_ne_plant_emits_filling_block(self, tmp_path) -> None:
        """An admitted NE plant emits its entry stage + filling contract.

        JURUENA (code 309) fills Oct 2024 (start_sid=1) and enters Nov 2024
        (entry_sid=2); the single-stage rate is ``2.93 / ζ_Oct`` with
        ``ζ_Oct = 744 * 3600 / 1e6 = 2.6784``.
        """
        case = _ne_filling_case(tmp_path)
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, _ne_filling_id_map())
        juruena = next(h for h in result["hydros"] if h["name"] == "JURUENA")
        assert juruena["entry_stage_id"] == 2
        # operational_start_date is the month filling completes and the plant enters
        # service: Oct-2024 start + 1-month duracao ⇒ Nov 2024.
        assert juruena["operational_start_date"] == "2024-11-01"
        assert juruena["exit_stage_id"] is None
        assert juruena["filling"] == {
            "start_stage_id": 1,
            "filling_min_rate_m3s": pytest.approx(2.93 / 2.6784, rel=1e-6),
        }
        # Reservoir still collapses to the 'S' single point (no new branch).
        assert juruena["reservoir"]["min_storage_hm3"] == pytest.approx(2.93)
        assert juruena["reservoir"]["max_storage_hm3"] == pytest.approx(2.93)

    def test_ex_plants_keep_none_filling(self, tmp_path) -> None:
        """EX plants in the same case keep entry/exit/filling all None."""
        case = _ne_filling_case(tmp_path)
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, _ne_filling_id_map())
        for name in ("USINA_A", "USINA_B"):
            hydro = next(h for h in result["hydros"] if h["name"] == name)
            assert hydro["entry_stage_id"] is None
            assert hydro["exit_stage_id"] is None
            assert hydro["filling"] is None
            # EX plants are in service since the historical record (Jan 1, 1931).
            assert hydro["operational_start_date"] == "1931-01-01"

    def test_ne_plant_zero_duration_omits_filling(self, tmp_path) -> None:
        """``duracao_enchimento == 0`` ⇒ entry == start, no filling block.

        novomodelo rejects ``start_stage_id >= entry_stage_id``, so the empty window
        emits ``entry_stage_id`` only and keeps ``filling`` None.
        """
        case = _ne_filling_case(tmp_path, duracao=0)
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, _ne_filling_id_map())
        juruena = next(h for h in result["hydros"] if h["name"] == "JURUENA")
        # Oct-2024 start under Sep-2024 horizon ⇒ start_sid == 1; duracao 0 ⇒
        # entry_sid == start_sid == 1.
        assert juruena["entry_stage_id"] == 1
        assert juruena["filling"] is None

    def test_ne_plant_filling_past_horizon_no_crash(self, tmp_path) -> None:
        """An NE plant whose filling completes past the horizon converts cleanly.

        Under a SHORT 3-stage horizon (Oct–Dec 2024) JURUENA's Oct-2024 filling
        start maps to ``start_sid == 0`` and a 6-month ``duracao`` pushes the
        entry to ``entry_sid == 6 > total_stages (3)`` — a valid case:
        the plant fills but never operates within the study, yet is still emitted.
        The rate is summed only over the in-horizon stages (the clamp in the
        caller), so ``convert_hydros`` must NOT raise ``IndexError`` indexing the
        per-stage date list, and the true unclamped ``entry_stage_id`` is emitted.
        """
        case = _hydro_case(
            tmp_path,
            cadastro=_make_ne_cadastro(),
            confhd=_make_ne_confhd_df(),
            # 3-stage horizon: Oct 2024 start, 1 study year, 0 post ⇒
            # study_months = (13 - 10) = 3; stages 0=Oct, 1=Nov, 2=Dec 2024.
            dger=_make_hydro_dger_mock(
                start_year=2024, start_month=10, num_anos=1, num_anos_pos=0
            ),
            # duracao 6: Oct-2024 start ⇒ start_sid == 0, entry_sid == 6 > 3.
            exph=_make_ne_exph_mock(duracao=6, volume_morto=0.0),
        )
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        # Must not raise (the IndexError this fix guards against).
        result = convert_hydros(case, _ne_filling_id_map())
        juruena = next(h for h in result["hydros"] if h["name"] == "JURUENA")

        # True, unclamped filling-complete stage is emitted (≥ total_stages == 3).
        assert juruena["entry_stage_id"] == 6
        assert juruena["entry_stage_id"] >= case.horizon.total_stages

        # entry_sid (6) > start_sid (0) ⇒ a filling block is emitted; its rate is
        # finite and non-negative (summed only over the in-horizon stages 0..2).
        filling = juruena["filling"]
        assert filling is not None
        rate = filling["filling_min_rate_m3s"]
        assert math.isfinite(rate)
        assert rate >= 0.0

    def test_ne_plant_emits_filling_diagnostic(self, tmp_path) -> None:
        """A NE filling plant emits exactly one INFO ``ne-filling-plant`` finding.

        Run under a ``collect()`` sink (as the pipeline does) so the diagnostic
        is captured instead of logged.
        """
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        with dx.collect() as collected:
            convert_hydros(_ne_filling_case(tmp_path), _ne_filling_id_map())

        filling = [d for d in collected if d.code == "ne-filling-plant"]
        assert len(filling) == 1
        assert filling[0].severity is Severity.INFO

    def test_ne_filling_diagnostic_table_row(self, tmp_path) -> None:
        """JURUENA's row carries its code, window 1→2, vol. morto 0.0, and ramp."""
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        with dx.collect() as collected:
            convert_hydros(_ne_filling_case(tmp_path), _ne_filling_id_map())

        diag = next(d for d in collected if d.code == "ne-filling-plant")
        assert diag.table is not None
        rows = [dict(zip(diag.table.columns, row)) for row in diag.table.rows]

        juruena = next(r for r in rows if r["Code"] == 309)
        assert juruena["Fill start"] == 1
        assert juruena["Operates from"] == 2
        assert juruena["Vol. morto %"] == "0.0"
        assert "stage 4" in juruena["Unit ramp"]

    def test_ex_only_emits_no_filling_diagnostic(self, tmp_path) -> None:
        """An EX-only case emits no ``ne-filling-plant`` diagnostic."""
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        with dx.collect() as collected:
            convert_hydros(_hydro_case(tmp_path), self._make_id_map())

        assert not any(d.code == "ne-filling-plant" for d in collected)


# ---------------------------------------------------------------------------
# convert_hydros integration tests
# ---------------------------------------------------------------------------


class TestConvertHydrosGhmin:
    """Integration tests for GHMIN handling.

    GHMIN values are now emitted per-stage in ``hydro_bounds.parquet``
    and the static ``hydros.json:generation.min_generation_mw`` is
    always 0.0.  These tests pin both halves of that contract.
    """

    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1, 2],
            thermal_codes=[],
        )

    def test_static_min_generation_is_zero_when_ghmin_present(self, tmp_path) -> None:
        """The static field is always zero — GHMIN goes elsewhere."""
        import datetime

        ghmin_df = pd.DataFrame(
            {
                "codigo_usina": [1],
                "data": [datetime.datetime(2025, 1, 1)],
                "patamar": [0],
                "geracao": [99.9],
            }
        )
        mock_ghmin_obj = MagicMock()
        mock_ghmin_obj.geracoes = ghmin_df

        # convert_hydros never reads GHMIN (the per-stage values live in
        # hydro_bounds.parquet); the GHMIN mock is supplied to mirror a case
        # where the file is present, and the static field must still be 0.
        case = _hydro_case(tmp_path, ghmin=mock_ghmin_obj)

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())

        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        assert hydro_a["generation"]["min_generation_mw"] == pytest.approx(0.0)

    def test_static_min_generation_is_zero_when_ghmin_absent(self, tmp_path) -> None:
        """With no GHMIN.DAT, static min_generation_mw is still 0."""
        case = _hydro_case(tmp_path)

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        assert hydro_a["generation"]["min_generation_mw"] == pytest.approx(0.0)


class TestConvertHydrosDownstreamFict:
    """Downstream reference to a fictitious plant must produce downstream_id=None."""

    def test_downstream_to_fict_is_none(self, tmp_path) -> None:
        """Plant with a fictitious downstream gets downstream_id=None.

        USINA_A (code=1) has codigo_usina_jusante=2, which is FICT.SERRA M.
        Because FICT.SERRA M is absent from id_map, the KeyError catch in
        hydro.py must produce downstream_id=None for USINA_A.
        """
        # Build a confhd DataFrame where plant 1 points downstream to a
        # fictitious plant (code=2) that is NOT present in the id_map.
        confhd_df = pd.DataFrame(
            {
                "codigo_usina": [1],
                "nome_usina": ["USINA_A"],
                "posto": [1],
                "codigo_usina_jusante": [2],  # points to the absent fict. plant
                "ree": [1],
                "volume_inicial_percentual": [50.0],
                "usina_existente": ["EX"],
                "usina_modificada": [0],
            }
        )

        # Hidr.cadastro for plant 1 only.
        cadastro = _make_hidr_cadastro().iloc[:1].copy()

        case = _hydro_case(tmp_path, cadastro=cadastro, confhd=confhd_df)

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        # id_map has only plant 1; plant 2 (fictitious) is absent.
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        result = convert_hydros(case, id_map)

        assert len(result["hydros"]) == 1
        assert result["hydros"][0]["downstream_id"] is None

    def test_terminal_plant_with_matching_fict_resolves_through_chain(
        self, tmp_path
    ) -> None:
        """A real plant with confhd jusante=0 and a name-matched FICT must
        wire to the next real plant via the FICT chain.

        Topology:
            USINA_A (code=1, jusante=0)          ← physically terminal in confhd
            FICT.USINA (code=2, jusante=3)       ← carries the energy cascade
            USINA_B (code=3, jusante=0)          ← real downstream

        After the FICT-cascade fix, USINA_A's downstream_id must point to
        USINA_B (novomodelo id=1), not None as in the pre-fix behavior.  The
        7-char name match is ``USINA A`` (after the FICT. prefix) matching
        ``USINA_A``'s first-7-char key — pure prefix equality.
        """
        confhd_df = pd.DataFrame(
            {
                "codigo_usina": [1, 2, 3],
                "nome_usina": ["USINA_A", "FICT.USINA_A", "USINA_B"],
                # FICT.USINA_A shares USINA_A's posto (1) — the structural
                # fictitious-twin signature that drives the posto-based Rule 3.
                "posto": [1, 1, 3],
                "codigo_usina_jusante": [0, 3, 0],
                "ree": [1, 1, 1],
                "volume_inicial_percentual": [50.0, 50.0, 50.0],
                "usina_existente": ["EX", "EX", "EX"],
                "usina_modificada": [0, 0, 0],
            }
        )

        cadastro = _make_hidr_cadastro().copy()
        # _make_hidr_cadastro has plants 1 and 2.  Make plant 2 the fictitious
        # twin (ρ=0, sharing plant 1's posto) and add plant 3 as a second real
        # plant cloned from plant 1.
        plant3 = cadastro.iloc[0:1].copy()
        plant3.index = [3]
        cadastro = pd.concat([cadastro, plant3])
        # Zero out the twin's specific productivity so it is classified
        # fictitious (ρ=0 sharing a generating posto).
        cadastro.loc[2, "produtibilidade_especifica"] = 0.0

        case = _hydro_case(tmp_path, cadastro=cadastro, confhd=confhd_df)

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 3], thermal_codes=[])
        result = convert_hydros(case, id_map)

        assert len(result["hydros"]) == 2
        by_code = {h["name"]: h for h in result["hydros"]}
        usina_a = by_code["USINA_A"]
        usina_b = by_code["USINA_B"]
        # USINA_A must wire to USINA_B via the FICT chain.
        assert usina_a["downstream_id"] == usina_b["id"], (
            f"Expected USINA_A.downstream_id == {usina_b['id']}, "
            f"got {usina_a['downstream_id']}"
        )
        # USINA_B remains terminal.
        assert usina_b["downstream_id"] is None


class TestConvertHydrosPenalid:
    """Integration tests for PENALID.DAT -> hydro penalties in convert_hydros."""

    def test_penalid_present_still_leaves_penalties_none(self, tmp_path) -> None:
        """Per-plant penalties are always None; PENALID is handled globally."""
        # convert_hydros does not read PENALID (per-plant overrides were removed;
        # the PENALID mock is supplied only to mirror a case where the file is
        # present). The case carries it for parity with production.
        mock_penalid = MagicMock()
        mock_penalid.penalidades = _make_penalid_df()
        case = _hydro_case(tmp_path, penalid=mock_penalid)

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        result = convert_hydros(case, id_map)

        # Per-plant penalty overrides were removed: PENALID values are
        # converted once via system-average productivity in penalties.json.
        for hydro in result["hydros"]:
            assert hydro["penalties"] is None, (
                f"Plant '{hydro['name']}' should have penalties=None "
                "(per-plant overrides removed; handled globally)"
            )

    def test_missing_penalid_leaves_penalties_none(self, tmp_path) -> None:
        """When PENALID.DAT is absent, every hydro entry has penalties=None."""
        case = _hydro_case(tmp_path)

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        result = convert_hydros(case, id_map)

        for hydro in result["hydros"]:
            assert hydro["penalties"] is None, (
                f"Plant '{hydro['name']}' should have penalties=None "
                "when PENALID.DAT is absent"
            )

    def test_different_rees_still_get_none_penalties(self, tmp_path) -> None:
        """Plants in different REEs still get penalties=None (global handling)."""
        # Two plants in different REEs: plant 1 in REE 1, plant 2 in REE 2.
        confhd_df = pd.DataFrame(
            {
                "codigo_usina": [1, 2],
                "nome_usina": ["USINA_A", "USINA_B"],
                "posto": [1, 2],
                "codigo_usina_jusante": [pd.NA, pd.NA],
                "ree": [1, 2],
                "volume_inicial_percentual": [50.0, 75.0],
                "usina_existente": ["EX", "EX"],
                "usina_modificada": [0, 0],
            }
        )

        # REE table: REE 1 -> subsystem 1, REE 2 -> subsystem 1.
        ree_df = pd.DataFrame(
            {"codigo": [1, 2], "nome": ["SE", "S"], "submercado": [1, 1]}
        )

        case = _hydro_case(tmp_path, confhd=confhd_df, rees=ree_df)

        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        result = convert_hydros(case, id_map)

        for hydro in result["hydros"]:
            assert hydro["penalties"] is None


class TestBuildMirrorUnitGroup:
    """Unit tests for ``build_mirror_unit_group``."""

    def test_returns_exactly_seven_keys(self) -> None:
        """The returned dict has exactly the seven ``RawUnitGroup`` keys."""
        from novomodelo_bridge.core.hydro_units import build_mirror_unit_group

        group = build_mirror_unit_group(
            name="PLANT",
            bus_id=3,
            min_generation_mw=0.0,
            max_generation_mw=100.0,
            min_turbined_m3s=0.0,
            max_turbined_m3s=50.0,
        )

        assert set(group.keys()) == {
            "id",
            "name",
            "bus_id",
            "min_generation_mw",
            "max_generation_mw",
            "min_turbined_m3s",
            "max_turbined_m3s",
        }

    def test_id_is_zero_and_name_unchanged(self) -> None:
        """``id`` is always 0; ``name`` passes through verbatim."""
        from novomodelo_bridge.core.hydro_units import build_mirror_unit_group

        group = build_mirror_unit_group(
            name="M. DE MORAES",
            bus_id=1,
            min_generation_mw=0.0,
            max_generation_mw=10.0,
            min_turbined_m3s=0.0,
            max_turbined_m3s=5.0,
        )

        assert group["id"] == 0
        assert group["name"] == "M. DE MORAES"

    def test_bounds_pass_through_verbatim(self) -> None:
        """All four bounds are returned unchanged, including a 0.0 minimum."""
        from novomodelo_bridge.core.hydro_units import build_mirror_unit_group

        group = build_mirror_unit_group(
            name="PLANT",
            bus_id=2,
            min_generation_mw=12.5,
            max_generation_mw=1400.0,
            min_turbined_m3s=0.0,
            max_turbined_m3s=980.3,
        )

        assert group["min_generation_mw"] == pytest.approx(12.5)
        assert group["max_generation_mw"] == pytest.approx(1400.0)
        assert group["min_turbined_m3s"] == pytest.approx(0.0)
        assert group["max_turbined_m3s"] == pytest.approx(980.3)

    def test_mirror_invariant_group_maxima_sum_to_plant_maximum(self) -> None:
        """Nested under a plant, ``sum(group maxima) == plant maximum`` holds
        for both ``max_turbined_m3s`` and ``max_generation_mw`` — the novomodelo
        rule-41 invariant a single mirror group satisfies by construction."""
        from novomodelo_bridge.core.hydro_units import build_mirror_unit_group

        plant_max_generation_mw = 1400.0
        plant_max_turbined_m3s = 980.3
        plant = {
            "name": "PLANT",
            "max_generation_mw": plant_max_generation_mw,
            "max_turbined_m3s": plant_max_turbined_m3s,
            "unit_groups": [
                build_mirror_unit_group(
                    name="PLANT",
                    bus_id=2,
                    min_generation_mw=12.5,
                    max_generation_mw=plant_max_generation_mw,
                    min_turbined_m3s=0.0,
                    max_turbined_m3s=plant_max_turbined_m3s,
                )
            ],
        }

        assert sum(
            g["max_generation_mw"] for g in plant["unit_groups"]
        ) == pytest.approx(plant["max_generation_mw"])
        assert sum(
            g["max_turbined_m3s"] for g in plant["unit_groups"]
        ) == pytest.approx(plant["max_turbined_m3s"])

    def test_keyword_only_signature_enforced(self) -> None:
        """A positional call raises ``TypeError``."""
        from novomodelo_bridge.core.hydro_units import build_mirror_unit_group

        with pytest.raises(TypeError):
            build_mirror_unit_group("PLANT", 2, 0.0, 100.0, 0.0, 50.0)


class TestLegacyHydroShapeRejectedBy013:
    """Document the exact shape 0.12 produced — a hydro dict
    with a top-level ``bus_id`` and no ``unit_groups`` — and that no converter
    emits it anymore.

    novomodelo 0.13's ``hydros.schema.json`` rejects that shape on load:
    ``additionalProperties: false`` denies the stray top-level ``bus_id``
    (decision 14 → §7.8), and ``unit_groups`` is a required, non-empty array
    (decision 13 → §7.6). This test asserts the *shape difference* only — it
    does not load either dict into novomodelo or a jsonschema validator (E4 owns
    real schema loads).
    """

    def test_0_12_shape_has_top_level_bus_id_and_no_unit_groups(self) -> None:
        """The exact shape 0.12 emitted for a hydro plant."""
        legacy_0_12_hydro = {
            "id": 0,
            "name": "PLANT",
            "bus_id": 3,
            "generation": {
                "min_generation_mw": 0.0,
                "max_generation_mw": 50.0,
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 100.0,
            },
        }

        assert "bus_id" in legacy_0_12_hydro
        assert "unit_groups" not in legacy_0_12_hydro

    def test_0_13_shape_has_neither(self) -> None:
        """The shape every converter emits today — via the shared test helper,
        so this pins the helper to the same invariant the real converters
        satisfy (no top-level ``bus_id``, one mirror ``unit_groups`` entry).
        ``TestConvertHydros.test_output_never_regresses_to_the_0_12_shape``
        pins the same invariant against the real converter output."""
        modern_hydro = hydro_with_group(0, bus_id=3)

        assert "bus_id" not in modern_hydro
        assert len(modern_hydro["unit_groups"]) == 1
        assert modern_hydro["unit_groups"][0]["bus_id"] == 3


class TestExpansionPlantAdmission:
    """An ``EE`` plant operates from stage 0 at its study-start configuration and
    reaches the registry configuration as its ``exph`` machines enter."""

    def _id_map(self) -> NewaveIdMap:
        return NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])

    def _hydros(self, case) -> list[dict]:
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        return convert_hydros(case, self._id_map())["hydros"]

    def _plant_b(self, case) -> dict:
        return next(h for h in self._hydros(case) if h["name"] == "USINA_B")

    def test_ee_plant_is_converted(self, tmp_path) -> None:
        hydros = self._hydros(_ee_expansion_case(tmp_path))
        assert {h["name"] for h in hydros} == {"USINA_A", "USINA_B"}

    def test_declares_the_registry_configuration_not_the_start_one(
        self, tmp_path
    ) -> None:
        # 3 x 150 (conjunto 1) + 2 x 120 (conjunto 2, entering) = 690 MW. A
        # declaration left at the 450 MW start configuration would cap the plant
        # there for every stage past the ramp, which carries no per-stage row.
        ee = self._plant_b(_ee_expansion_case(tmp_path))
        assert ee["generation"]["max_generation_mw"] == pytest.approx(690.0)
        group = ee["unit_groups"][0]
        assert group["max_generation_mw"] == ee["generation"]["max_generation_mw"]

    def test_ramp_reports_start_and_full_capacity(self, tmp_path) -> None:
        with dx.collect() as collected:
            self._hydros(_ee_expansion_case(tmp_path))
        diags = [d for d in collected if d.code == "ee-expansion-ramped"]
        assert len(diags) == 1
        assert diags[0].severity is Severity.INFO
        assert diags[0].table is not None
        assert diags[0].table.rows[0] == ["USINA_B", 2, "450.0", "690.0", 2, 6]

    def test_ee_plant_without_exph_reports_nothing(self, tmp_path) -> None:
        case = _hydro_case(tmp_path, confhd=_make_ee_confhd_df())
        with dx.collect() as collected:
            self._hydros(case)
        assert [d for d in collected if d.code == "ee-expansion-ramped"] == []

    def test_dated_entry_without_conjunto_is_not_a_ramp(self, tmp_path) -> None:
        # A dated exph row with no machine group places no machine: the plant
        # converts at its registry configuration instead of crashing.
        exph = _make_ee_exph_mock()
        exph.expansoes = exph.expansoes.iloc[:1].assign(
            conjunto_maquina_entrada=float("nan")
        )
        case = _hydro_case(tmp_path, confhd=_make_ee_confhd_df(), exph=exph)
        with dx.collect() as collected:
            ee = self._plant_b(case)
        assert ee["generation"]["max_generation_mw"] == pytest.approx(690.0)
        assert [d for d in collected if d.code == "ee-expansion-ramped"] == []
