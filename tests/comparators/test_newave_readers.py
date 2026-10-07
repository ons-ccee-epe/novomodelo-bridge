"""Parse/degrade branch coverage for every public reader in ``newave_readers``.

Each reader has two branches per its module contract: the input file is
present and parses (typed non-empty result), or the input is absent (typed
empty frame / ``None`` plus a ``_LOG.warning``, never a raised exception).
The fixture case under ``fixtures/newave_results/`` is small hand-authored
synthetic data — one entity or two, at most a few stages per file — built
directly from each ``inewave`` reader's expected layout, not a slice of any
real deck.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.newave.readers import (
    read_fpha_grid,
    read_fpha_planes,
    read_medias_hydro,
    read_medias_market,
    read_medias_ree,
    read_medias_sin,
    read_medias_system,
    read_medias_thermal,
    read_newave_net_load,
    read_newave_net_load_nwlistop,
    read_newave_tim_iterations,
    read_newave_tim_stages,
    read_nwlistop_intercambio,
    read_pmo_convergence,
    read_pmo_cost_breakdown,
    read_pmo_productivity_detail,
)
from tests.conftest import make_nw_files

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "newave_results"


# ---------------------------------------------------------------------------
# MEDIAS CSV readers
# ---------------------------------------------------------------------------


class TestMediasHydro:
    def test_parses_ghiduh_and_varmuh_per_plant_stage(self) -> None:
        df = read_medias_hydro(_FIXTURE_DIR)
        assert not df.is_empty()
        assert set(df["newave_code"].unique().to_list()) == {10, 20}
        assert set(df["variable"].unique().to_list()) == {"GHIDUH", "VARMUH"}
        row = df.filter(
            (pl.col("newave_code") == 10)
            & (pl.col("stage") == 9)
            & (pl.col("variable") == "GHIDUH")
        )
        assert row["value"].item() == 120.5

    def test_degrades_to_empty_frame_on_missing_file(
        self, tmp_path: Path, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING):
            df = read_medias_hydro(tmp_path)
        assert df.is_empty()
        assert "MEDIAS-USIH.CSV" in caplog.text


class TestMediasThermal:
    def test_parses_gter_per_plant_stage(self) -> None:
        df = read_medias_thermal(_FIXTURE_DIR)
        assert not df.is_empty()
        assert df["newave_code"].unique().to_list() == [1]
        assert df["variable"].unique().to_list() == ["GTER"]

    def test_degrades_to_empty_frame_on_missing_file(
        self, tmp_path: Path, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING):
            df = read_medias_thermal(tmp_path)
        assert df.is_empty()
        assert "MEDIAS-USIT.CSV" in caplog.text


class TestMediasSystem:
    def test_parses_cmo_and_deft_per_submarket_stage(self) -> None:
        df = read_medias_system(_FIXTURE_DIR)
        assert not df.is_empty()
        assert set(df["variable"].unique().to_list()) == {"CMO", "DEFT"}

    def test_degrades_to_empty_frame_on_missing_file(
        self, tmp_path: Path, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING):
            df = read_medias_system(tmp_path)
        assert df.is_empty()
        assert "MEDIAS-MERC.CSV" in caplog.text


class TestMediasMarket:
    def test_parses_all_market_variables(self) -> None:
        df = read_medias_market(_FIXTURE_DIR)
        assert not df.is_empty()
        assert set(df["variable"].unique().to_list()) == {"CMO", "DEFT"}

    def test_degrades_to_empty_frame_on_missing_file(self, tmp_path: Path) -> None:
        assert read_medias_market(tmp_path).is_empty()


class TestMediasSin:
    def test_parses_earmf_and_ena(self) -> None:
        df = read_medias_sin(_FIXTURE_DIR)
        assert not df.is_empty()
        assert df["newave_code"].unique().to_list() == [0]
        assert set(df["variable"].unique().to_list()) == {"EARMF", "ENA"}

    def test_degrades_to_empty_frame_on_missing_file(self, tmp_path: Path) -> None:
        assert read_medias_sin(tmp_path).is_empty()


class TestMediasRee:
    def test_parses_earmf_per_ree_stage(self) -> None:
        df = read_medias_ree(_FIXTURE_DIR)
        assert not df.is_empty()
        assert df["newave_code"].unique().to_list() == [1]
        assert df["variable"].unique().to_list() == ["EARMF"]

    def test_degrades_to_empty_frame_on_missing_file(self, tmp_path: Path) -> None:
        assert read_medias_ree(tmp_path).is_empty()


# ---------------------------------------------------------------------------
# pmo.dat readers
# ---------------------------------------------------------------------------


class TestPmoConvergence:
    def test_parses_iteration_table_via_own_regex(self) -> None:
        df = read_pmo_convergence(_FIXTURE_DIR)
        assert df.to_dicts() == [
            {"iteration": 1, "lower_bound": 5.551e8, "upper_bound_mean": 5.803e8},
            {"iteration": 2, "lower_bound": 5.602e8, "upper_bound_mean": 5.855e8},
        ]

    def test_degrades_to_empty_frame_on_missing_pmo(
        self, tmp_path: Path, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING):
            df = read_pmo_convergence(tmp_path)
        assert df.is_empty()
        assert "pmo.dat not found" in caplog.text


class TestPmoProductivityDetail:
    def test_parses_head_dependent_productivities(self) -> None:
        df = read_pmo_productivity_detail(_FIXTURE_DIR)
        assert set(df["plant_name"].to_list()) == {"BATALHA", "FUNIL-GRANDE"}
        batalha = df.filter(pl.col("plant_name") == "BATALHA").row(0, named=True)
        assert batalha["equivalent"] == 0.3323
        assert batalha["altura_min"] == 0.2535
        assert batalha["accumulated_earm"] == 5.3539
        # FUNIL-GRANDE carries no reservoir-productivity row in the fixture,
        # so its head-dependent columns stay null rather than crashing.
        funil = df.filter(pl.col("plant_name") == "FUNIL-GRANDE").row(0, named=True)
        assert funil["altura_min"] is None

    def test_degrades_to_empty_frame_on_missing_pmo(self, tmp_path: Path) -> None:
        assert read_pmo_productivity_detail(tmp_path).is_empty()


class TestPmoCostBreakdown:
    def test_parses_nonzero_categories_converted_to_reais(self) -> None:
        result = read_pmo_cost_breakdown(_FIXTURE_DIR)
        assert result == {"GERACAO TERMICA": 21887910000.0}

    def test_degrades_to_empty_dict_on_missing_pmo(self, tmp_path: Path) -> None:
        assert read_pmo_cost_breakdown(tmp_path) == {}


# ---------------------------------------------------------------------------
# NWLISTOP intercambio / net load
# ---------------------------------------------------------------------------


class TestNwlistopIntercambio:
    def test_parses_directional_pair_total_row(self) -> None:
        df = read_nwlistop_intercambio(_FIXTURE_DIR)
        assert not df.is_empty()
        assert df["from_submarket_code"].unique().to_list() == [1]
        assert df["to_submarket_code"].unique().to_list() == [2]
        assert df["variable"].unique().to_list() == ["INTERC"]

    def test_degrades_to_empty_frame_when_no_int_files(self, tmp_path: Path) -> None:
        assert read_nwlistop_intercambio(tmp_path).is_empty()


class TestNewaveNetLoadNwlistop:
    def test_parses_mercl_file_full_horizon(self) -> None:
        df = read_newave_net_load_nwlistop(_FIXTURE_DIR)
        assert not df.is_empty()
        assert df["newave_code"].unique().to_list() == [1]
        assert df["variable"].unique().to_list() == ["NET_LOAD"]
        assert df.height == 12

    def test_degrades_to_empty_frame_when_no_mercl_files(self, tmp_path: Path) -> None:
        assert read_newave_net_load_nwlistop(tmp_path).is_empty()


class TestNewaveNetLoad:
    def test_prefers_nwlistop_full_horizon_over_sistema_reconstruction(self) -> None:
        df = read_newave_net_load(make_nw_files(_FIXTURE_DIR))
        assert not df.is_empty()
        assert df.height == 12  # the mercl001.out full-horizon reader, not sistema

    @pytest.mark.parametrize("extension", ["dat", "eas"])
    def test_falls_back_to_sistema_reconstruction_with_c_adic(
        self, tmp_path: Path, extension: str
    ) -> None:
        # No mercl*.out present -> falls back to the subsystem + c_adic files,
        # under whatever names the case's arquivos gives them.
        sistema = tmp_path / f"sistema.{extension}"
        c_adic = tmp_path / f"c_adic.{extension}"
        sistema.write_bytes((_FIXTURE_DIR / "sistema.dat").read_bytes())
        c_adic.write_bytes((_FIXTURE_DIR / "c_adic.dat").read_bytes())
        df = read_newave_net_load(
            make_nw_files(tmp_path, sistema=sistema, c_adic=c_adic)
        )
        assert not df.is_empty()
        assert df["variable"].unique().to_list() == ["NET_LOAD"]
        # net_load = mercado_energia (38124.0) + c_adic must-take (18.0)
        # - geracao_usinas_nao_simuladas (1503.0), all submarket 1 / month 7.
        july = df.filter(pl.col("stage") == 7)
        assert july["value"].item() == 38124.0 + 18.0 - 1503.0

    def test_degrades_to_empty_frame_on_missing_sistema(
        self, tmp_path: Path, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING):
            df = read_newave_net_load(make_nw_files(tmp_path))
        assert df.is_empty()
        assert "sistema.dat not found" in caplog.text


# ---------------------------------------------------------------------------
# newave.tim readers
# ---------------------------------------------------------------------------


class TestNewaveTimIterations:
    def test_parses_backward_forward_total_per_iteration(self) -> None:
        df = read_newave_tim_iterations(_FIXTURE_DIR)
        assert df.to_dicts() == [
            {
                "iteration": 1,
                "backward_seconds": 46.0,
                "forward_seconds": 14.0,
                "total_seconds": 60.0,
            },
            {
                "iteration": 2,
                "backward_seconds": 40.0,
                "forward_seconds": 13.0,
                "total_seconds": 53.0,
            },
        ]

    def test_degrades_to_empty_frame_on_missing_tim(
        self, tmp_path: Path, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING):
            df = read_newave_tim_iterations(tmp_path)
        assert df.is_empty()
        assert "newave.tim not found" in caplog.text


class TestNewaveTimStages:
    def test_parses_named_stage_durations(self) -> None:
        result = read_newave_tim_stages(_FIXTURE_DIR)
        assert result["Tempo Total"] == 3 * 3600 + 26 * 60 + 7
        assert result["Calculo da Politica"] == 2 * 3600 + 59 * 60 + 38

    def test_degrades_to_empty_dict_on_missing_tim(self, tmp_path: Path) -> None:
        assert read_newave_tim_stages(tmp_path) == {}


# ---------------------------------------------------------------------------
# FPHA reports
# ---------------------------------------------------------------------------


class TestFphaPlanes:
    def test_parses_production_hyperplane_coefficients(self) -> None:
        df = read_fpha_planes(_FIXTURE_DIR)
        assert df is not None
        assert set(df["newave_code"].to_list()) == {4, 20}
        row = df.filter(pl.col("newave_code") == 4).row(0, named=True)
        assert row["gamma_q"] == 0.35607775

    def test_degrades_to_none_on_missing_report(self, tmp_path: Path) -> None:
        assert read_fpha_planes(tmp_path) is None


class TestFphaGrid:
    def test_parses_fitting_grid_domain(self) -> None:
        df = read_fpha_grid(_FIXTURE_DIR)
        assert df is not None
        assert set(df["newave_code"].to_list()) == {4, 20}
        row = df.filter(pl.col("newave_code") == 4).row(0, named=True)
        assert row["v_min_hm3"] == 265.9
        assert row["gh_max_mw"] == 180.0

    def test_degrades_to_none_on_missing_report(self, tmp_path: Path) -> None:
        assert read_fpha_grid(tmp_path) is None
