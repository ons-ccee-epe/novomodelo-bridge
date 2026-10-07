"""REE energy rollup tests for ``comparators.decomp.results``.

Covers the REE membership map, Novomodelo-side and DECOMP-side per-REE
ENA/EARM sums, the full REE result-comparison rollup, the Balance tab's REE
rows in ``build_decomp_dataset``, and the REE energy chart.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.charts import ree_energy_chart
from novomodelo_bridge.comparators.decomp.results import (
    _EARM_MWH_TO_MWMES,
    _decomp_ree_frame,
    _novomodelo_ree_sums,
    _ree_membership_map,
    _ree_result_comparisons,
    build_decomp_dataset,
)
from novomodelo_bridge.comparators.model import ResultComparison
from novomodelo_bridge.comparators.report_builder import build_comparison_report
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.decomp.id_map import DecompIdMap
from tests.comparators.conftest import (
    _aligned_fixture,
    _extract_tab_content,
    _patch_aligned_frames,
    _patch_ree_sources,
    _patch_shared_case,
    _ree_aligned_fixture,
    _ree_dec_oper_ree_fixture,
    _ree_id_map,
    _ree_membership_fixture,
)


def _ree_novomodelo_hydro_fixture() -> pl.DataFrame:
    """Two Novomodelo hydro plants (ids 0, 1), one stage: ENA sums to 150.0 MW,
    EARM sums to 730000.0 MWh -- exactly ``1000.0 * _EARM_MWH_TO_MWMES``, so
    the MWh -> MWmes reconciliation lands on a round number."""
    return pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [0, 0],
            "incremental_inflow_energy_mw": [90.0, 60.0],
            "stored_energy_final_mwh": [400000.0, 330000.0],
        }
    )


class TestReeMembershipMap:
    """``_ree_membership_map``: ``{novomodelo_hydro_id: codigo_ree}`` via
    membership, restricted to the operated hydro codes."""

    def test_maps_novomodelo_ids_to_codigo_ree(self) -> None:
        ree_by_novomodelo_id, unmapped = _ree_membership_map(
            _ree_membership_fixture(), {10: 0, 20: 1}
        )
        assert ree_by_novomodelo_id == {0: 100, 1: 100}
        assert unmapped == []

    def test_reports_unmapped_hydro_codes_instead_of_dropping_silently(self) -> None:
        membership = pl.DataFrame({"codigo_usina": [10], "codigo_ree": [100]})

        ree_by_novomodelo_id, unmapped = _ree_membership_map(membership, {10: 0, 99: 5})

        assert ree_by_novomodelo_id == {0: 100}
        assert unmapped == [99]

    def test_empty_membership_excludes_every_hydro_code(self) -> None:
        ree_by_novomodelo_id, unmapped = _ree_membership_map(
            pl.DataFrame(), {10: 0, 20: 1}
        )

        assert ree_by_novomodelo_id == {}
        assert unmapped == [10, 20]


class TestNovomodeloReeSums:
    """``_novomodelo_ree_sums``: membership-weighted per-(codigo_ree, stage) sum
    of Novomodelo hydro ENA/EARM."""

    def test_sums_ena_and_earm_across_member_plants(self) -> None:
        out = _novomodelo_ree_sums(_ree_novomodelo_hydro_fixture(), {0: 100, 1: 100})

        assert out.height == 1
        row = out.row(0, named=True)
        assert row["entity_id"] == 100
        assert row["stage_id"] == 0
        assert row["ena_mw"] == pytest.approx(150.0)
        assert row["earm_mwh"] == pytest.approx(730000.0)

    def test_novomodelo_id_absent_from_membership_excluded_from_sum(self) -> None:
        novomodelo_hydro = pl.DataFrame(
            {
                "entity_id": [0, 9],
                "stage_id": [0, 0],
                "incremental_inflow_energy_mw": [90.0, 999.0],
                "stored_energy_final_mwh": [400000.0, 999.0],
            }
        )

        out = _novomodelo_ree_sums(novomodelo_hydro, {0: 100})

        assert out.height == 1
        row = out.row(0, named=True)
        assert row["ena_mw"] == pytest.approx(90.0)
        assert row["earm_mwh"] == pytest.approx(400000.0)

    def test_empty_novomodelo_hydro_yields_empty_frame(self) -> None:
        assert _novomodelo_ree_sums(pl.DataFrame(), {0: 100}).is_empty()

    def test_empty_membership_map_yields_empty_frame(self) -> None:
        assert _novomodelo_ree_sums(_ree_novomodelo_hydro_fixture(), {}).is_empty()

    def test_missing_energy_columns_degrades_to_empty_instead_of_raising(self) -> None:
        """A ``novomodelo_hydro`` frame that carries no ENA/EARM columns at all --
        e.g. the trimmed ``_aligned_fixture()`` shape other fixtures
        use -- must degrade gracefully rather than raising a Polars
        ``ColumnNotFoundError``."""
        novomodelo_hydro = pl.DataFrame({"entity_id": [0], "stage_id": [0]})

        assert _novomodelo_ree_sums(novomodelo_hydro, {0: 100}).is_empty()


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
    """``_ree_result_comparisons``: the full REE rollup -- membership map,
    scenario-averaged DECOMP side, membership-weighted Novomodelo side, the EARM
    MWh -> MWmes reconciliation, and the never-silently-dropped
    unmapped-plant diagnostic."""

    def test_without_stage_hours_ena_is_unscaled_earm_is_divided_by_730(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # With no stage_hours lookup, ENA falls back to the raw stage-mean MW
        # (factor 1.0) -- the backward-compatible default.
        _patch_ree_sources(monkeypatch)

        results, unmapped = _ree_result_comparisons(
            tmp_path, _ree_novomodelo_hydro_fixture(), _ree_id_map()
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
        assert ena.novomodelo_value == pytest.approx(150.0)  # unscaled fallback
        assert ena.abs_diff == pytest.approx(5.0)

        earm = by_variable["earm_final_mwmes"]
        assert earm.novomodelo_value == pytest.approx(730000.0 / _EARM_MWH_TO_MWMES)

    def test_stage_hours_convert_ena_rate_to_mwmes_energy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # novomodelo's ena_mw is an average-MW rate; with stage_hours it is converted
        # to MW-month energy via stage_hours/730 (mirroring EARM's ÷730), so it
        # is comparable to DECOMP's ena_MWmes. Half a month (365h) halves it.
        _patch_ree_sources(monkeypatch)

        results, _ = _ree_result_comparisons(
            tmp_path,
            _ree_novomodelo_hydro_fixture(),
            _ree_id_map(),
            stage_hours={0: _EARM_MWH_TO_MWMES / 2},  # 365 h == 0.5 month
        )

        ena = {r.variable: r for r in results}["ena_mwmes"]
        # 150.0 MW (rate) × (365 / 730) == 75.0 MWmês.
        assert ena.novomodelo_value == pytest.approx(75.0)
        assert ena.newave_value == pytest.approx(145.0)

    def test_none_id_map_returns_no_rows_and_no_unmapped(self, tmp_path: Path) -> None:
        results, unmapped = _ree_result_comparisons(
            tmp_path, _ree_novomodelo_hydro_fixture(), None
        )

        assert results == []
        assert unmapped == []

    def test_plant_absent_from_membership_excluded_and_recorded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A hydro code absent from the membership table is excluded
        from every REE sum and its code is recorded via the diagnostics
        path, not silently dropped."""
        # The id map declares a THIRD hydro code (30) the membership fixture
        # never lists.
        id_map = DecompIdMap(
            bus_codes=(1,), bus_names=("SE",), hydro_codes=(10, 20, 30)
        )
        _patch_ree_sources(monkeypatch)

        with dx.collect() as collected:
            results, unmapped = _ree_result_comparisons(
                tmp_path, _ree_novomodelo_hydro_fixture(), id_map
            )

        assert unmapped == [30]
        assert len(collected) == 1
        assert collected[0].code == "ree-membership-plant-unmapped"
        assert "30" in " ".join(str(n) for n in collected[0].notes)
        # The unmapped plant's novomodelo id (2) was never a REE member -- the
        # sums are unaffected (still exactly the two-plant fixture's totals).
        by_variable = {r.variable: r for r in results}
        assert by_variable["ena_mwmes"].novomodelo_value == pytest.approx(150.0)

    def test_no_membership_table_degrades_to_no_rows_no_diagnostic(
        self, tmp_path: Path
    ) -> None:
        """A missing relato (``read_relato_membership`` raising against a
        bare ``tmp_path``) is a genuinely unavailable REE section, not a
        per-plant gap -- no diagnostic, just an empty result."""
        with dx.collect() as collected:
            results, unmapped = _ree_result_comparisons(
                tmp_path, _ree_novomodelo_hydro_fixture(), _ree_id_map()
            )

        assert results == []
        assert unmapped == []
        assert collected == []

    def test_no_dec_oper_ree_table_degrades_to_no_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_membership",
            lambda *_a, **_k: _ree_membership_fixture(),
        )
        # ``read_dec_oper_ree`` left unmocked -> raises FileNotFoundError.

        results, unmapped = _ree_result_comparisons(
            tmp_path, _ree_novomodelo_hydro_fixture(), _ree_id_map()
        )

        assert results == []
        assert unmapped == []


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
        assert dataset.metadata["unmapped"]["ree"] == []
        html = build_comparison_report(dataset)  # must not raise
        assert "REE Energy" not in html

    def test_both_sides_present_tidy_carries_ree_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``tidy`` has ``entity_type=="ree"`` rows for ``ena_mwmes`` and
        ``earm_final_mwmes``, with ``source`` in {"newave", "novomodelo"}."""
        _patch_aligned_frames(monkeypatch, _ree_aligned_fixture())
        _patch_shared_case(monkeypatch, id_map=_ree_id_map())
        _patch_ree_sources(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        ree_rows = dataset.tidy.filter(pl.col("entity_type") == "ree")
        assert set(ree_rows["variable"].unique().to_list()) == {
            "ena_mwmes",
            "earm_final_mwmes",
        }
        assert set(ree_rows["source"].unique().to_list()) == {"newave", "novomodelo"}
        assert dataset.metadata["unmapped"]["ree"] == []

    def test_report_ree_section_present_for_decomp_dataset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``build_comparison_report(dataset)`` renders a non-empty REE
        energy section for a DECOMP dataset."""
        _patch_aligned_frames(monkeypatch, _ree_aligned_fixture())
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
