"""Productivity tab tests for ``comparators.decomp.results``.

Covers per-(plant, stage) realized productivity derivation and the
Productivity tab's realized-per-stage half of ``build_decomp_dataset``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.decomp.results import (
    _PRODUCTIVITY_TURBINED_EPS,
    _AlignedDecompFrames,
    _hydro_productivity_results,
    build_decomp_dataset,
)
from novomodelo_bridge.comparators.model import ResultComparison
from novomodelo_bridge.comparators.report_builder import build_comparison_report
from tests.comparators.conftest import (
    _aligned_fixture,
    _extract_tab_content,
    _patch_aligned_frames,
)


class TestHydroProductivityResults:
    """``_hydro_productivity_results`` derives per-(plant, stage) realized
    productivity = generation / turbined from the E1 hydro
    ``ResultComparison`` rows, mirroring
    ``novomodelo_bridge.comparators.newave.results``'s own ``_compare_hydros``
    productivity derivation (same ratio, same zero-guard)."""

    @staticmethod
    def _hydro_pair(
        *,
        nw_gen: float,
        nw_turb: float,
        cb_gen: float,
        cb_turb: float,
        code: int = 10,
        novomodelo_id: int = 0,
        stage: int = 0,
        name: str = "ALPHA",
    ) -> list[ResultComparison]:
        """One plant/stage's generation_mw + turbined_m3s rows -- the shape
        ``_result_comparisons`` already emits for hydro entities."""
        return [
            ResultComparison(
                entity_type="hydro",
                entity_name=name,
                newave_code=code,
                novomodelo_id=novomodelo_id,
                stage=stage,
                variable="generation_mw",
                newave_value=nw_gen,
                novomodelo_value=cb_gen,
                abs_diff=abs(nw_gen - cb_gen),
                rel_diff=None,
            ),
            ResultComparison(
                entity_type="hydro",
                entity_name=name,
                newave_code=code,
                novomodelo_id=novomodelo_id,
                stage=stage,
                variable="turbined_m3s",
                newave_value=nw_turb,
                novomodelo_value=cb_turb,
                abs_diff=abs(nw_turb - cb_turb),
                rel_diff=None,
            ),
        ]

    def test_ratio_math(self) -> None:
        """geracao_MW=100, vazao_turbinada_m3s=50 -> newave_value 2.0."""
        rows = self._hydro_pair(nw_gen=100.0, nw_turb=50.0, cb_gen=90.0, cb_turb=45.0)

        productivity = _hydro_productivity_results(rows)

        assert len(productivity) == 1
        row = productivity[0]
        assert row.entity_type == "hydro"
        assert row.variable == "productivity_mw_per_m3s"
        assert row.newave_value == pytest.approx(2.0)
        assert row.novomodelo_value == pytest.approx(2.0)

    def test_zero_guard_drops_the_row_when_source_turbined_is_zero(self) -> None:
        """vazao_turbinada_m3s=0 -> no non-null newave_value -- the row
        is DROPPED entirely (never null-kept)."""
        rows = self._hydro_pair(nw_gen=100.0, nw_turb=0.0, cb_gen=90.0, cb_turb=45.0)

        assert _hydro_productivity_results(rows) == []

    def test_zero_guard_drops_the_row_when_novomodelo_turbined_is_zero(self) -> None:
        """Either side's turbined flow at zero drops the row -- not just the
        source model's."""
        rows = self._hydro_pair(nw_gen=100.0, nw_turb=50.0, cb_gen=90.0, cb_turb=0.0)

        assert _hydro_productivity_results(rows) == []

    def test_zero_guard_uses_the_eps_floor_not_only_exact_zero(self) -> None:
        """Below the eps floor (not just exactly zero) is also dropped -- an
        undefined 0/0, not a genuinely tiny but real ratio."""
        rows = self._hydro_pair(
            nw_gen=100.0,
            nw_turb=_PRODUCTIVITY_TURBINED_EPS / 2,
            cb_gen=90.0,
            cb_turb=45.0,
        )

        assert _hydro_productivity_results(rows) == []

    def test_row_dropped_when_the_matching_variable_is_absent(self) -> None:
        """Only a generation_mw row, no turbined_m3s row for that key --
        never an exception, just no productivity row (missing turbined
        flow)."""
        rows = [
            ResultComparison(
                entity_type="hydro",
                entity_name="ALPHA",
                newave_code=10,
                novomodelo_id=0,
                stage=0,
                variable="generation_mw",
                newave_value=100.0,
                novomodelo_value=90.0,
                abs_diff=10.0,
                rel_diff=0.1,
            )
        ]

        assert _hydro_productivity_results(rows) == []

    def test_non_hydro_rows_are_ignored(self) -> None:
        rows = [
            ResultComparison(
                entity_type="thermal",
                entity_name="GAS_A",
                newave_code=1,
                novomodelo_id=0,
                stage=0,
                variable="generation_mw",
                newave_value=100.0,
                novomodelo_value=90.0,
                abs_diff=10.0,
                rel_diff=0.1,
            )
        ]

        assert _hydro_productivity_results(rows) == []

    def test_two_plants_sorted_by_code_then_stage(self) -> None:
        rows = self._hydro_pair(
            nw_gen=100.0,
            nw_turb=50.0,
            cb_gen=90.0,
            cb_turb=45.0,
            code=20,
            novomodelo_id=1,
            stage=1,
            name="BETA",
        ) + self._hydro_pair(
            nw_gen=60.0,
            nw_turb=30.0,
            cb_gen=55.0,
            cb_turb=25.0,
            code=10,
            novomodelo_id=0,
            stage=0,
            name="ALPHA",
        )

        productivity = _hydro_productivity_results(rows)

        assert [(r.newave_code, r.stage) for r in productivity] == [(10, 0), (20, 1)]


def _productivity_aligned_fixture() -> _AlignedDecompFrames:
    """Two hydro plants, one stage: plant A (code 10) exercises the ratio
    math (100/50 -> 2.0 on the source side); plant B (code 20)'s source
    turbined is 0 (zero-guard drops it, never null-keeps it)."""
    source_hydro = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "newave_code": [10, 20],
            "stage_id": [0, 0],
            "geracao_MW": [100.0, 60.0],
            "vazao_turbinada_m3s": [50.0, 0.0],
            "vazao_vertida_m3s": [0.0, 0.0],
            "vazao_defluente_m3s": [50.0, 0.0],
            "volume_util_final_hm3": [500.0, 300.0],
        }
    )
    novomodelo_hydro = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [0, 0],
            "generation_mw": [105.0, 60.0],
            "turbined_m3s": [50.0, 40.0],
            "spillage_m3s": [0.0, 0.0],
            "outflow_m3s": [50.0, 40.0],
            "useful_storage_hm3": [480.0, 300.0],
        }
    )
    return dataclasses.replace(
        _aligned_fixture(),
        source_hydro=source_hydro,
        novomodelo_hydro=novomodelo_hydro,
        hydro_names={0: "ALPHA", 1: "BETA"},
    )


class TestBuildDecompDatasetProductivity:
    """Fills the Productivity tab's realized per-stage half
    (``dataset.render.productivity_per_stage``) and leaves the static
    pmo-derived half (``productivity_detail``) empty -- DECOMP ships no
    pmo.dat ([ASSUMPTION] option a, no fabricated static comparison)."""

    def test_productivity_per_stage_has_the_six_columns(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _productivity_aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        per_stage = dataset.render.productivity_per_stage
        assert isinstance(per_stage, pl.DataFrame)
        assert set(per_stage.columns) == {
            "plant_name",
            "newave_code",
            "novomodelo_id",
            "stage",
            "newave_value",
            "novomodelo_value",
        }

    def test_ratio_math_matches_dec_oper_usih_generation_over_turbined(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """geracao_MW=100, vazao_turbinada_m3s=50 -> newave_value 2.0,
        end to end from ``build_decomp_dataset``."""
        _patch_aligned_frames(monkeypatch, _productivity_aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        per_stage = dataset.render.productivity_per_stage
        alpha = per_stage.filter(pl.col("newave_code") == 10)
        assert alpha["newave_value"].to_list() == [pytest.approx(2.0)]
        assert alpha["novomodelo_value"].to_list() == [pytest.approx(2.1)]

    def test_zero_turbined_plant_emits_no_non_null_newave_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Plant B's source turbined is 0 -> the zero-guard holds (the
        row is DROPPED entirely, so no non-null -- or null -- value for it
        survives into the frame)."""
        _patch_aligned_frames(monkeypatch, _productivity_aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        per_stage = dataset.render.productivity_per_stage
        assert per_stage.filter(pl.col("newave_code") == 20).is_empty()
        assert per_stage["newave_value"].null_count() == 0

    def test_productivity_detail_stays_empty_no_pmo_fabrication(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _productivity_aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.render.productivity_detail.is_empty()

    def test_report_renders_both_the_realized_title_and_the_static_no_data_note(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """On the SAME DECOMP dataset: with
        ``productivity_detail`` empty AND ``productivity_per_stage``
        non-empty, the decoupled report_builder gate renders
        BOTH the realized-productivity section and the static section's "No
        productivity data available" note."""
        _patch_aligned_frames(monkeypatch, _productivity_aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        productivity_tab = _extract_tab_content(html, "tab-productivity")
        assert "Realized productivity across stages" in productivity_tab
        assert "No productivity data available." in productivity_tab
        # The static-only Building Blocks table has no DECOMP data to render.
        assert "Productivity Building Blocks" not in productivity_tab
