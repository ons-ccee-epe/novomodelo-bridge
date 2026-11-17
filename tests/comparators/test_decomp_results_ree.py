"""REE energy rollup tests for ``comparators.decomp.results``.

Covers Novomodelo's REE energies under DECOMP's definition, the DECOMP-side
per-REE ENA/EARM frame, the full REE result-comparison rollup, the Balance
tab's REE rows in ``build_decomp_dataset``, and the REE energy chart. The
definition itself is tested in ``test_decomp_ree_energy.py``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.charts import ree_energy_chart
from novomodelo_bridge.comparators.decomp.results import (
    _decomp_ree_frame,
    _novomodelo_ree_energies,
    _ree_result_comparisons,
    build_decomp_dataset,
)
from novomodelo_bridge.comparators.model import ResultComparison
from novomodelo_bridge.comparators.report_builder import build_comparison_report
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.units import MEAN_MONTH_HOURS
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.novomodelo.scenario_weights import ScenarioWeights
from tests.comparators.conftest import (
    _aligned_fixture,
    _extract_tab_content,
    _patch_aligned_frames,
    _patch_ree_sources,
    _patch_shared_case,
    _ree_block_fixture,
    _ree_dec_oper_ree_fixture,
    _ree_id_map,
    _ree_membership_fixture,
    _ree_rule_fixture,
    _unmapped_codes,
)
from tests.conftest import make_decomp_case

_RESULTS = "novomodelo_bridge.comparators.decomp.results"


def _ree_case(directory: Path, id_map: DecompIdMap | None = None) -> DecompCase:
    return make_decomp_case(directory, id_map=id_map or _ree_id_map())


def _patch_block_rows(monkeypatch: pytest.MonkeyPatch, rows: pl.DataFrame) -> None:
    monkeypatch.setattr(
        f"{_RESULTS}.novomodelo_readers.read_novomodelo_hydro_block_operation",
        lambda *_a, **_k: rows,
    )


class TestNovomodeloReeEnergies:
    """``_novomodelo_ree_energies``: Novomodelo's per-scenario plant storage
    and inflow valued with the rule, then weighted over the scenarios."""

    def test_values_each_stage_once_from_its_block_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_block_rows(monkeypatch, _ree_block_fixture())

        out = _novomodelo_ree_energies(
            _ree_rule_fixture(), {10: 0, 20: 1}, tmp_path, {}
        )

        assert out.to_dicts() == [
            {
                "entity_id": 100,
                "stage_id": 0,
                "earm_mwmes": pytest.approx(1000.0),
                "ena_mwmes": pytest.approx(150.0),
            }
        ]

    def test_scenarios_are_weighted_by_their_probability(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        base = _ree_block_fixture()
        doubled = base.with_columns(
            pl.lit(1, dtype=pl.Int64).alias("scenario_id"),
            pl.col("storage_final_hm3") * 2.0,
            pl.col("incremental_inflow_m3s") * 2.0,
        )
        _patch_block_rows(monkeypatch, pl.concat([base, doubled]))
        monkeypatch.setattr(
            f"{_RESULTS}.novomodelo_readers.read_novomodelo_scenario_weights",
            lambda *_a, **_k: ScenarioWeights.from_probabilities([0, 1], [0.25, 0.75]),
        )

        row = _novomodelo_ree_energies(
            _ree_rule_fixture(), {10: 0, 20: 1}, tmp_path, {}
        ).row(0, named=True)

        assert row["earm_mwmes"] == pytest.approx(0.25 * 1000.0 + 0.75 * 2000.0)
        assert row["ena_mwmes"] == pytest.approx(0.25 * 150.0 + 0.75 * 300.0)

    def test_a_hydro_outside_the_operated_map_is_left_out(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        stray = (
            _ree_block_fixture()
            .head(1)
            .with_columns(pl.lit(9, dtype=pl.Int64).alias("hydro_id"))
        )
        _patch_block_rows(monkeypatch, pl.concat([_ree_block_fixture(), stray]))

        row = _novomodelo_ree_energies(
            _ree_rule_fixture(), {10: 0, 20: 1}, tmp_path, {}
        ).row(0, named=True)

        assert row["earm_mwmes"] == pytest.approx(1000.0)

    def test_a_run_without_hydro_rows_has_no_ree_energy(self, tmp_path: Path) -> None:
        out = _novomodelo_ree_energies(
            _ree_rule_fixture(), {10: 0, 20: 1}, tmp_path, {}
        )

        assert out.is_empty()


class TestDecompReeFrame:
    """``_decomp_ree_frame``: DECOMP-side per-(codigo_ree, stage) ENA/EARM,
    scenario-averaged, plus the REE display-name lookup."""

    def test_scenario_means_and_rebases_stage(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_ree",
            lambda *_a, **_k: _ree_dec_oper_ree_fixture(),
        )

        frame, names = _decomp_ree_frame(tmp_path)

        assert frame.height == 1
        row = frame.row(0, named=True)
        assert row["entity_id"] == 100
        assert row["stage_id"] == 0
        assert row["ena_MWmes"] == pytest.approx(145.0)
        assert row["earm_final_MWmes"] == pytest.approx(1010.0)
        assert names == {100: "SUDESTE"}

    def test_missing_table_raises(self, tmp_path: Path) -> None:
        """``read_dec_oper_ree`` unmocked against a bare ``tmp_path`` raises
        -- the caller (``_ree_result_comparisons``) is what degrades this to
        an absent REE section, not this helper."""
        with pytest.raises(FileNotFoundError):
            _decomp_ree_frame(tmp_path)


class TestReeResultComparisons:
    """``_ree_result_comparisons``: the full REE rollup -- the scenario-averaged
    DECOMP side, Novomodelo's energies under DECOMP's definition, and the
    never-silently-dropped unmapped-plant diagnostic."""

    def test_without_stage_hours_ena_spans_one_month(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_ree_sources(monkeypatch)

        results, unmapped = _ree_result_comparisons(
            _ree_case(tmp_path), tmp_path, tmp_path
        )

        assert unmapped == []
        by_variable = {r.variable: r for r in results}
        assert set(by_variable) == {"ena_mwmes", "earm_final_mwmes"}

        ena = by_variable["ena_mwmes"]
        assert ena.entity_type == "ree"
        assert ena.entity_name == "SUDESTE"
        assert ena.newave_code == 100
        assert ena.novomodelo_id == 100
        assert ena.stage == 0
        assert ena.newave_value == pytest.approx(145.0)
        assert ena.novomodelo_value == pytest.approx(150.0)
        assert ena.abs_diff == pytest.approx(5.0)

        earm = by_variable["earm_final_mwmes"]
        assert earm.newave_value == pytest.approx(1010.0)
        assert earm.novomodelo_value == pytest.approx(1000.0)

    def test_stage_hours_scale_ena_to_the_mean_month(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_ree_sources(monkeypatch)

        results, _ = _ree_result_comparisons(
            _ree_case(tmp_path),
            tmp_path,
            tmp_path,
            stage_hours={0: MEAN_MONTH_HOURS / 2},
        )

        by_variable = {r.variable: r for r in results}
        assert by_variable["ena_mwmes"].novomodelo_value == pytest.approx(75.0)
        assert by_variable["earm_final_mwmes"].novomodelo_value == pytest.approx(1000.0)

    def test_plant_absent_from_membership_excluded_and_recorded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A hydro code absent from the membership table is left out of every
        REE energy and its code is recorded via the diagnostics path, not
        silently dropped."""
        id_map = DecompIdMap(
            bus_codes=(1,), bus_names=("SE",), hydro_codes=(10, 20, 30)
        )
        _patch_ree_sources(monkeypatch)

        with dx.collect() as collected:
            results, unmapped = _ree_result_comparisons(
                _ree_case(tmp_path, id_map), tmp_path, tmp_path
            )

        assert unmapped == [30]
        assert len(collected) == 1
        assert collected[0].code == "ree-membership-plant-unmapped"
        assert collected[0].table is not None
        assert collected[0].table.rows == [[30]]
        by_variable = {r.variable: r for r in results}
        assert by_variable["ena_mwmes"].novomodelo_value == pytest.approx(150.0)

    def test_a_dormant_plant_is_left_out_without_a_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        id_map = DecompIdMap(
            bus_codes=(1,),
            bus_names=("SE",),
            hydro_codes=(10, 20, 30),
            dormant_hydro_codes=(30,),
        )
        _patch_ree_sources(monkeypatch)

        with dx.collect() as collected:
            results, unmapped = _ree_result_comparisons(
                _ree_case(tmp_path, id_map), tmp_path, tmp_path
            )

        assert unmapped == []
        assert collected == []
        by_variable = {r.variable: r for r in results}
        assert by_variable["ena_mwmes"].novomodelo_value == pytest.approx(150.0)

    def test_no_membership_table_degrades_to_no_rows_and_notes_the_missing_relato(
        self, tmp_path: Path
    ) -> None:
        """A missing relato (``read_relato_membership`` raising against a
        bare ``tmp_path``) is an unavailable REE section, not a per-plant
        gap: one missing-file note and an empty result."""
        with dx.collect() as collected:
            results, unmapped = _ree_result_comparisons(
                _ree_case(tmp_path), tmp_path, tmp_path
            )

        assert results == []
        assert unmapped == []
        assert [(d.code, d.title) for d in collected] == [
            ("report-file-missing", "relato.rvN not found")
        ]

    def test_unreadable_membership_table_is_reported_and_degrades_to_no_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise ValueError("relato.rv0 has no uhes_rees_submercados table")

        monkeypatch.setattr(f"{_RESULTS}.read_relato_membership", _boom)

        with dx.collect() as collected:
            results, unmapped = _ree_result_comparisons(
                _ree_case(tmp_path), tmp_path, tmp_path
            )

        assert results == []
        assert unmapped == []
        assert [(d.code, d.title) for d in collected] == [
            ("report-file-unreadable", "relato.rvN could not be read")
        ]

    def test_no_dec_oper_ree_table_is_noted_and_degrades_to_no_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            f"{_RESULTS}.read_relato_membership",
            lambda *_a, **_k: _ree_membership_fixture(),
        )
        # ``read_dec_oper_ree`` left unmocked -> raises FileNotFoundError.

        with dx.collect() as collected:
            results, unmapped = _ree_result_comparisons(
                _ree_case(tmp_path), tmp_path, tmp_path
            )

        assert results == []
        assert unmapped == []
        assert [(d.code, d.title) for d in collected] == [
            ("report-file-missing", "dec_oper_ree.csv not found")
        ]

    def test_unreadable_dec_oper_ree_table_is_reported_and_degrades_to_no_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise ValueError("dec_oper_ree.csv parsed empty")

        monkeypatch.setattr(
            f"{_RESULTS}.read_relato_membership",
            lambda *_a, **_k: _ree_membership_fixture(),
        )
        monkeypatch.setattr(f"{_RESULTS}.read_dec_oper_ree", _boom)

        with dx.collect() as collected:
            results, unmapped = _ree_result_comparisons(
                _ree_case(tmp_path), tmp_path, tmp_path
            )

        assert results == []
        assert unmapped == []
        assert [(d.code, d.title) for d in collected] == [
            ("report-file-unreadable", "dec_oper_ree.csv could not be read")
        ]

    def test_an_energy_chain_outside_hidr_is_reported_and_degrades_to_no_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> None:
            raise ValueError(
                "plant 10's AC JUSENA names plant 77, which is neither in the "
                "configuration nor in hidr.dat"
            )

        _patch_ree_sources(monkeypatch)
        monkeypatch.setattr(f"{_RESULTS}._ree_energy_rule", _boom)

        with dx.collect() as collected:
            results, unmapped = _ree_result_comparisons(
                _ree_case(tmp_path), tmp_path, tmp_path
            )

        assert results == []
        assert unmapped == []
        assert [(d.code, d.title) for d in collected] == [
            ("report-file-unreadable", "dadger.rvN could not be read")
        ]


class TestBuildDecompDatasetRee:
    """Fills ``results`` with ``entity_type="ree"`` rows and
    ``dataset.metadata["unmapped"]["ree"]``."""

    def test_no_deck_no_ree_rows_and_empty_unmapped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The shared case's ``id_map`` is a degenerate, empty (but valid)
        ``DecompIdMap`` against a bare ``tmp_path`` (via
        ``_patch_aligned_frames``'s default) -> no REE rollup, empty
        ``unmapped["ree"]``, no exception, and no REE section in the report."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        ree_rows = dataset.tidy.filter(pl.col("entity_type") == "ree")
        assert ree_rows.is_empty()
        assert _unmapped_codes(dataset, "ree") == []
        html = build_comparison_report(dataset)  # must not raise
        assert "REE Energy" not in html

    def test_both_sides_present_tidy_carries_ree_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``tidy`` has ``entity_type=="ree"`` rows for ``ena_mwmes`` and
        ``earm_final_mwmes``, with ``source`` in {"newave", "novomodelo"}."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_shared_case(monkeypatch, id_map=_ree_id_map())
        _patch_ree_sources(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        ree_rows = dataset.tidy.filter(pl.col("entity_type") == "ree")
        assert set(ree_rows["variable"].unique().to_list()) == {
            "ena_mwmes",
            "earm_final_mwmes",
        }
        assert set(ree_rows["source"].unique().to_list()) == {"newave", "novomodelo"}
        assert _unmapped_codes(dataset, "ree") == []

    def test_report_ree_section_present_for_decomp_dataset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``build_comparison_report(dataset)`` renders a non-empty REE
        energy section for a DECOMP dataset."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_shared_case(monkeypatch, id_map=_ree_id_map())
        _patch_ree_sources(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        assert "REE Energy" in html
        balance_tab = _extract_tab_content(html, "tab-balance")
        assert "No ena_mwmes data available." not in balance_tab
        assert "No earm_final_mwmes data available." not in balance_tab
        assert "Plotly.newPlot" in balance_tab


class TestReeEnergyChart:
    """``charts.ree_energy_chart``: mirrors ``system_comparison_chart``'s
    aggregate-line shape, keyed on ``entity_type == "ree"``."""

    def _results(self) -> list[ResultComparison]:
        return [
            ResultComparison(
                entity_type="ree",
                entity_name="SUDESTE",
                newave_code=100,
                novomodelo_id=100,
                stage=0,
                variable="ena_mwmes",
                newave_value=145.0,
                novomodelo_value=150.0,
                abs_diff=5.0,
                rel_diff=5.0 / 145.0,
            ),
            ResultComparison(
                entity_type="ree",
                entity_name="SUL",
                newave_code=200,
                novomodelo_id=200,
                stage=0,
                variable="ena_mwmes",
                newave_value=50.0,
                novomodelo_value=48.0,
                abs_diff=2.0,
                rel_diff=2.0 / 50.0,
            ),
        ]

    def test_no_matching_rows_renders_placeholder(self) -> None:
        html = ree_energy_chart([], "ena_mwmes", "REE ENA")
        assert "No ena_mwmes data available." in html

    def test_sums_across_matched_rees_per_stage(self) -> None:
        html = ree_energy_chart(self._results(), "ena_mwmes", "REE ENA")

        assert "Plotly.newPlot" in html
        assert "195" in html  # 145 + 50 == 195 (newave aggregate)
        assert "198" in html  # 150 + 48 == 198 (novomodelo aggregate)

    def test_ignores_rows_of_a_different_variable(self) -> None:
        html = ree_energy_chart(self._results(), "earm_final_mwmes", "REE EARM")
        assert "No earm_final_mwmes data available." in html
