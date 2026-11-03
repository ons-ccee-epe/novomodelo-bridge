"""Parse/degrade branch coverage for every public reader in ``newave_readers``.

Each reader has two branches per its module contract: the input file is
present and parses (typed non-empty result), or the input is absent (typed
empty frame / ``None`` plus a ``_LOG.warning``, never a raised exception).
The fixture case under ``fixtures/newave_results/`` is small hand-authored
synthetic data — one entity or two, at most a few stages per file — built
directly from each ``inewave`` reader's expected layout, not a slice of any
real deck. The nwlistop plant tables under
``fixtures/newave_results/nwlistop_plants/`` are the first two years of one
plant's and one submarket's tables from a September 2024 NEWAVE run with a
historical final simulation.
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
    read_medias_thermal,
    read_newave_net_load,
    read_newave_net_load_nwlistop,
    read_newave_tim_iterations,
    read_newave_tim_stages,
    read_nwlistop_block_cmo,
    read_nwlistop_hydro_blocks,
    read_nwlistop_hydro_storage,
    read_nwlistop_intercambio,
    read_pmo_convergence,
    read_pmo_cost_breakdown,
    read_pmo_productivity_detail,
)
from tests.conftest import make_nw_files

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "newave_results"
_PLANT_TABLES = _FIXTURE_DIR / "nwlistop_plants"


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

    @pytest.mark.parametrize("end", [",", ""], ids=["trailing-comma", "no-comma"])
    def test_reads_the_last_stage_whether_or_not_a_line_ends_in_a_comma(
        self, tmp_path: Path, end: str
    ) -> None:
        (tmp_path / "MEDIAS-USIH.CSV").write_text(
            f"USIH_ext,     VAR,       9,      10{end}\n"
            f"       1,  GHIDUH,   120.5,   121.0{end}\n",
            encoding="utf-8",
        )

        df = read_medias_hydro(tmp_path)

        assert df.sort("stage")["value"].to_list() == [120.5, 121.0]

    def test_a_value_printed_as_a_dash_has_no_row(self, tmp_path: Path) -> None:
        (tmp_path / "MEDIAS-USIH.CSV").write_text(
            "USIH_ext,     VAR,       9,      10,\n"
            "       1,  GHIDUH,   120.5,   121.0,\n"
            "       1, VIOL_NEG_EVAP,       -      ,     2.5,\n",
            encoding="utf-8",
        )

        df = read_medias_hydro(tmp_path)

        assert df.filter(pl.col("variable") == "VIOL_NEG_EVAP").select(
            "stage", "value"
        ).rows() == [(10, 2.5)]

    def test_degrades_to_empty_frame_on_missing_file(
        self, tmp_path: Path, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING):
            df = read_medias_hydro(tmp_path)
        assert df.is_empty()
        assert "MEDIAS-USIH.CSV" in caplog.text


class TestMediasThermal:
    def test_parses_the_generation_variables_per_plant_stage(self) -> None:
        df = read_medias_thermal(_FIXTURE_DIR)
        assert df["newave_code"].unique().to_list() == [1]
        assert set(df["variable"].unique().to_list()) == {"GTMIN", "GTERM", "GTERMTOT"}

    def test_degrades_to_empty_frame_on_missing_file(
        self, tmp_path: Path, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING):
            df = read_medias_thermal(tmp_path)
        assert df.is_empty()
        assert "MEDIAS-USIT.CSV" in caplog.text


class TestMediasMarket:
    def test_parses_cmo_and_deft_per_submarket_stage(self) -> None:
        df = read_medias_market(_FIXTURE_DIR)
        assert not df.is_empty()
        assert set(df["variable"].unique().to_list()) == {"CMO", "DEFT"}

    def test_the_code_0_system_rows_are_not_read(self, tmp_path: Path) -> None:
        (tmp_path / "MEDIAS-MERC.CSV").write_text(
            "SBM_ext,VAR,       9,      10,\n"
            "       0,COPER,  1000.0,  1100.0,\n"
            "       1,CMO,   150.3,   152.1,\n",
            encoding="utf-8",
        )

        df = read_medias_market(tmp_path)

        assert df["newave_code"].unique().to_list() == [1]

    def test_degrades_to_empty_frame_on_missing_file(
        self, tmp_path: Path, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING):
            df = read_medias_market(tmp_path)
        assert df.is_empty()
        assert "MEDIAS-MERC.CSV" in caplog.text


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


_PMO_CONVERGENCE_HEADER = [
    "    ITER               LIM.INF.                   ZINF               LIM.SUP."
    "                   ZSUP                  DZINF             ZSUP ITER.",
    "                        ($10^6)                ($10^6)                ($10^6)"
    "                ($10^6)                    (%)                ($10^6)",
    "   X----X" + "----------------------X" * 6,
]


def _write_pmo(directory: Path, rows: list[str]) -> Path:
    path = directory / "pmo.dat"
    path.write_text(
        "\n".join([*_PMO_CONVERGENCE_HEADER, *rows, ""]), encoding="latin-1"
    )
    return path


class TestPmoConvergence:
    def test_reads_the_row_carrying_zsup_iter_of_each_iteration(self) -> None:
        """The fixture prints three rows per iteration, as a run with several
        forward series does; the first two are confidence-interval rows."""
        df = read_pmo_convergence(_FIXTURE_DIR)
        assert df.to_dicts() == [
            {
                "iteration": 1,
                "lower_bound": pytest.approx(555.1e6),
                "upper_bound_mean": pytest.approx(580.4e6),
                "upper_bound_best": pytest.approx(580.4e6),
            },
            {
                "iteration": 2,
                "lower_bound": pytest.approx(560.2e6),
                "upper_bound_mean": pytest.approx(586.0e6),
                "upper_bound_best": pytest.approx(580.4e6),
            },
        ]

    def test_reads_two_rows_per_iteration_whose_numbers_overrun_the_columns(
        self, tmp_path: Path
    ) -> None:
        _write_pmo(
            tmp_path,
            [
                "      1              4016521.0626873286                 214.9923938220"
                "             4016521.0626873286             4016521.0626873286"
                "                                         000h00min02.000s",
                "      1              4016521.0626873286                 214.9923938220"
                "             4016521.0626873286             4016521.0626873286"
                "                -                     4016521.0626873286   ",
                " " * 61 + "2025 100.0  631.6 100.0 1550.2   0.0    0.0 100.0   67.9",
            ],
        )

        row = read_pmo_convergence(tmp_path).row(0, named=True)

        assert row == {
            "iteration": 1,
            "lower_bound": pytest.approx(214.9923938220e6),
            "upper_bound_mean": pytest.approx(4016521.0626873286e6),
            "upper_bound_best": pytest.approx(4016521.0626873286e6),
        }

    def test_a_rewritten_pmo_is_read_again(self, tmp_path: Path) -> None:
        pmo = tmp_path / "pmo.dat"
        pmo.write_bytes((_FIXTURE_DIR / "pmo.dat").read_bytes())
        assert read_pmo_convergence(tmp_path)["iteration"].to_list() == [1, 2]

        lines = pmo.read_text(encoding="latin-1").splitlines()
        first_of_iteration_2 = next(
            i for i, line in enumerate(lines) if line.startswith("      2")
        )
        pmo.write_text(
            "\n".join([*lines[:first_of_iteration_2], ""]), encoding="latin-1"
        )

        assert read_pmo_convergence(tmp_path)["iteration"].to_list() == [1]

    def test_degrades_to_empty_frame_on_missing_pmo(
        self, tmp_path: Path, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING):
            df = read_pmo_convergence(tmp_path)
        assert df.is_empty()
        assert "pmo.dat not found" in caplog.text

    def test_degrades_to_empty_frame_without_a_convergence_table(
        self, tmp_path: Path, caplog
    ) -> None:
        (tmp_path / "pmo.dat").write_text(
            " CEPEL   MODELO ESTRATEGICO DE GERACAO HIDROTERMICA\n", encoding="latin-1"
        )
        with caplog.at_level(logging.WARNING):
            df = read_pmo_convergence(tmp_path)
        assert df.is_empty()
        assert "No convergence table" in caplog.text


def test_one_comparison_parses_pmo_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inewave.newave import Pmo

    (tmp_path / "pmo.dat").write_bytes((_FIXTURE_DIR / "pmo.dat").read_bytes())
    parsed: list[str] = []
    read = Pmo.read
    monkeypatch.setattr(Pmo, "read", lambda path: parsed.append(path) or read(path))

    read_pmo_convergence(tmp_path)
    read_pmo_productivity_detail(tmp_path)
    read_pmo_cost_breakdown(tmp_path)

    assert parsed == [str(tmp_path / "pmo.dat")]


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


# ---------------------------------------------------------------------------
# nwlistop plant tables
# ---------------------------------------------------------------------------


class TestNwlistopHydroBlocks:
    def test_keys_each_block_from_the_study_start(self) -> None:
        df = read_nwlistop_hydro_blocks(_PLANT_TABLES, 2024, 9)

        assert set(df["newave_code"]) == {6}
        assert (df["stage"].min(), df["stage"].max()) == (0, 15)
        assert set(df["block"]) == {0, 1, 2}
        assert set(df["scenario"]) == {0}
        row = df.filter((pl.col("stage") == 0) & (pl.col("block") == 1)).row(
            0, named=True
        )
        assert row["generation_mwmes"] == 216.56
        assert row["turbined_m3s"] == 941.7
        assert row["spillage_m3s"] == 0.0
        assert row["exact_generation_mwmes"] == 209.74

    def test_a_later_study_start_drops_the_earlier_months(self) -> None:
        df = read_nwlistop_hydro_blocks(_PLANT_TABLES, 2025, 1)

        row = df.filter((pl.col("stage") == 0) & (pl.col("block") == 0)).row(
            0, named=True
        )
        assert row["generation_mwmes"] == 231.83
        assert df["stage"].max() == 11

    def test_without_the_exact_table_its_column_is_null(self, tmp_path: Path) -> None:
        for name in ("ghiduh006.out", "qturuh006.out", "qvertuh006.out"):
            (tmp_path / name).write_bytes((_PLANT_TABLES / name).read_bytes())

        df = read_nwlistop_hydro_blocks(tmp_path, 2024, 9)

        assert df.height == 48
        assert df["exact_generation_mwmes"].is_null().all()

    def test_degrades_to_empty_without_plant_tables(self, tmp_path: Path) -> None:
        df = read_nwlistop_hydro_blocks(tmp_path, 2024, 9)

        assert df.is_empty()
        assert "exact_generation_mwmes" in df.columns


class TestNwlistopHydroStorage:
    def test_reads_the_final_useful_storage_per_stage(self) -> None:
        df = read_nwlistop_hydro_storage(_PLANT_TABLES, 2024, 9)

        assert df.filter(pl.col("stage") == 0)["useful_storage_final_hm3"].item() == (
            7890.92
        )
        assert df.filter(pl.col("stage") == 4)["useful_storage_final_hm3"].item() == (
            10809.7
        )

    def test_degrades_to_empty_without_tables(self, tmp_path: Path) -> None:
        assert read_nwlistop_hydro_storage(tmp_path, 2024, 9).is_empty()


class TestNwlistopBlockCmo:
    def test_reads_each_submarket_block(self) -> None:
        df = read_nwlistop_block_cmo(_PLANT_TABLES, 2024, 9)

        assert set(df["submarket_code"]) == {1}
        row = df.filter((pl.col("stage") == 0) & (pl.col("block") == 2)).row(
            0, named=True
        )
        assert row["cmo"] == 920.85

    def test_degrades_to_empty_without_tables(self, tmp_path: Path) -> None:
        assert read_nwlistop_block_cmo(tmp_path, 2024, 9).is_empty()
