"""Hydro/Thermal detail tab tests for ``comparators.decomp.results``.

Second carve out of the legacy ``test_decomp_results_compare.py`` mega file:
the hydro bus-id merge helper and the Hydro/Thermal detail tabs'
``build_decomp_dataset`` rows, plus the hydro slack-aggregate chart's
robustness guard on decomp-shaped input. The remaining concern bands
(network, energy balance, costs, performance, productivity, FPHA, REE,
evaporation, constraints, CLI) stay in the mega file pending their own carve.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.charts import hydro_slack_aggregate_chart
from novomodelo_bridge.comparators.dataset import _metadata_to_json
from novomodelo_bridge.comparators.decomp.results import (
    _merge_hydro_bus_ids,
    build_decomp_dataset,
)
from novomodelo_bridge.comparators.report_builder import build_comparison_report
from tests.comparators.conftest import (
    _aligned_fixture,
    _extract_tab_content,
    _patch_aligned_frames,
)


def _hydro_percentiles_fixture() -> pl.DataFrame:
    """Novomodelo p10/p50/p90 for the two ``_aligned_fixture`` hydro entities."""
    return pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [0, 0],
            "generation_mw_p10": [100.0, 55.0],
            "generation_mw_p50": [110.0, 60.0],
            "generation_mw_p90": [120.0, 65.0],
            "storage_final_hm3_p10": [470.0, 290.0],
            "storage_final_hm3_p50": [480.0, 300.0],
            "storage_final_hm3_p90": [490.0, 310.0],
        }
    )


def _hydro_metadata_fixture() -> dict[int, dict]:
    return {
        0: {"name": "A", "min_storage_hm3": 20.0},
        1: {"name": "B", "min_storage_hm3": 10.0},
    }


def _hydro_bus_labels_fixture() -> dict[int, frozenset[int]]:
    return {0: frozenset({0}), 1: frozenset({0})}


def _hydro_per_stage_bounds_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [0, 0],
            "min_storage_hm3": [20.0, 10.0],
            "max_storage_hm3": [1000.0, 600.0],
        }
    )


def _patch_hydro_detail_readers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    percentiles: pl.DataFrame | None = None,
    metadata: dict[int, dict] | None = None,
    bus_labels: dict[int, frozenset[int]] | None = None,
    per_stage_bounds: pl.DataFrame | None = None,
) -> None:
    """Stub the four novomodelo readers, each defaulting to empty --
    matching how a Novomodelo run with no hydro percentile/metadata output
    (e.g. the deterministic 2-node tree) degrades in production."""
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
        "read_novomodelo_hydro_percentiles",
        lambda *_a, **_k: pl.DataFrame() if percentiles is None else percentiles,
    )
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
        "read_novomodelo_hydro_metadata",
        lambda *_a, **_k: {} if metadata is None else metadata,
    )
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
        "read_novomodelo_hydro_bus_labels",
        lambda *_a, **_k: {} if bus_labels is None else bus_labels,
    )
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
        "read_novomodelo_hydro_per_stage_bounds",
        lambda *_a, **_k: (
            pl.DataFrame() if per_stage_bounds is None else per_stage_bounds
        ),
    )


class TestMergeHydroBusIds:
    """The ``bus_ids`` merge helper -- the per-bus hydro charts
    KeyError without it (see ``analyze._bus_name_lookups``)."""

    def test_injects_bus_ids_as_a_sorted_list_not_a_frozenset(self) -> None:
        meta = {0: {"name": "A"}, 1: {"name": "B"}}
        labels = {0: frozenset({9, 7})}

        merged = _merge_hydro_bus_ids(meta, labels)

        # A JSON-native sorted list, not the reader's frozenset: bus_ids lands
        # in the JSON-serialized metadata side-table, which rejects a frozenset.
        assert merged[0]["bus_ids"] == [7, 9]
        assert isinstance(merged[0]["bus_ids"], list)
        assert merged[1]["bus_ids"] == []

    def test_merged_bus_ids_serialize_into_the_metadata_side_table(self) -> None:
        merged = _merge_hydro_bus_ids({0: {"name": "A"}}, {0: frozenset({3})})

        # Regression: a frozenset here raised ``TypeError`` in
        # ``_metadata_to_json`` at compare-artifact export time.
        _metadata_to_json({"novomodelo_hydro_meta": merged})

    def test_does_not_mutate_the_inputs(self) -> None:
        meta = {0: {"name": "A"}}
        labels = {0: frozenset({7})}

        merged = _merge_hydro_bus_ids(meta, labels)
        merged[0]["name"] = "changed"
        merged[0]["bus_ids"] = frozenset({99})

        assert meta[0] == {"name": "A"}
        assert labels[0] == frozenset({7})

    def test_every_plant_gets_a_bus_ids_key_even_with_no_labels(self) -> None:
        meta = {5: {"name": "C"}}

        merged = _merge_hydro_bus_ids(meta, {})

        assert merged[5]["bus_ids"] == []


class TestBuildDecompDatasetHydroDetail:
    """The Hydro Operation + Hydro Plant Details tabs' four
    remaining ``PercentileData`` fields (``hydro``, ``novomodelo_hydro_meta``,
    ``novomodelo_hydro_per_stage_bounds``, ``nw_hydro_slacks``)."""

    def test_hydro_percentiles_populate_metadata_when_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(
            monkeypatch, percentiles=_hydro_percentiles_fixture()
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        hydro_pct = dataset.render.hydro
        assert isinstance(hydro_pct, pl.DataFrame)
        assert not hydro_pct.is_empty()
        assert "generation_mw_p50" in hydro_pct.columns
        assert "storage_final_hm3_p50" in hydro_pct.columns

    def test_hydro_percentiles_stay_empty_when_novomodelo_output_lacks_them(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No percentile mock (deterministic-tree low-N, master-plan caveat
        1): ``read_novomodelo_hydro_percentiles`` degrades to its own empty-frame
        default and the dataset must not fabricate a spread."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        hydro_pct = dataset.render.hydro
        assert isinstance(hydro_pct, pl.DataFrame)
        assert hydro_pct.is_empty()

    def test_novomodelo_hydro_meta_entries_carry_bus_ids(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(
            monkeypatch,
            metadata=_hydro_metadata_fixture(),
            bus_labels=_hydro_bus_labels_fixture(),
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        novomodelo_hydro_meta = dataset.render.novomodelo_hydro_meta
        assert set(novomodelo_hydro_meta) == {0, 1}
        for entry in novomodelo_hydro_meta.values():
            assert "bus_ids" in entry
        assert novomodelo_hydro_meta[0]["bus_ids"] == [0]
        assert novomodelo_hydro_meta[1]["bus_ids"] == [0]
        # Plant physics from ``read_novomodelo_hydro_metadata`` survive the merge.
        assert novomodelo_hydro_meta[0]["name"] == "A"

    def test_novomodelo_hydro_meta_bus_ids_empty_when_plant_has_no_label(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(
            monkeypatch, metadata=_hydro_metadata_fixture(), bus_labels={}
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        novomodelo_hydro_meta = dataset.render.novomodelo_hydro_meta
        assert novomodelo_hydro_meta[0]["bus_ids"] == []
        assert novomodelo_hydro_meta[1]["bus_ids"] == []

    def test_hydro_per_stage_bounds_populate_metadata_verbatim(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(
            monkeypatch, per_stage_bounds=_hydro_per_stage_bounds_fixture()
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        bounds = dataset.render.novomodelo_hydro_per_stage_bounds
        assert bounds["max_storage_hm3"].to_list() == [1000.0, 600.0]

    def test_nw_hydro_slacks_is_empty_decomp_has_no_slack_table(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        nw_hydro_slacks = dataset.render.nw_hydro_slacks
        assert nw_hydro_slacks is None or nw_hydro_slacks.is_empty()

    def test_hydro_operation_and_detail_tabs_render_with_metadata_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(
            monkeypatch,
            percentiles=_hydro_percentiles_fixture(),
            metadata=_hydro_metadata_fixture(),
            bus_labels=_hydro_bus_labels_fixture(),
            per_stage_bounds=_hydro_per_stage_bounds_fixture(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_bus_metadata",
            lambda *_a, **_k: {0: {"name": "SUDESTE"}},
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        assert "Storage by Bus (hm³)" in html
        assert 'id="tab-hydro-detail"' in html
        hydro_tab = _extract_tab_content(html, "tab-hydro")
        assert "No hydro storage_final_hm3 data mapped to buses." not in hydro_tab
        assert "Plotly.newPlot" in html

    def test_both_hydro_tabs_render_without_exception_when_percentiles_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No novomodelo hydro percentiles/metadata/bus-labels at all -- both tabs
        must degrade gracefully (fallback panels), never raise."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)  # must not raise

        assert "Storage by Bus (hm³)" in html
        assert 'id="tab-hydro-detail"' in html

    def test_sin_slack_charts_render_novomodelo_only_no_newave_trace(self) -> None:
        """``nw_hydro_slacks`` empty (or ``None``) -> the SIN-total slack
        chart renders the Novomodelo Mean trace with no ``NEWAVE`` trace -- the
        documented ``has_newave=False`` degrade path -- exercised directly
        on the shared, unmodified chart function."""
        novomodelo_hydro = pl.DataFrame(
            {
                "entity_id": [0],
                "stage_id": [0],
                "water_withdrawal_violation_neg_m3s": [1.5],
            }
        )

        html = hydro_slack_aggregate_chart(
            novomodelo_hydro,
            None,
            "water_withdrawal_violation_neg_m3s",
            "Withdrawal Slack Pos (m³/s)",
        )

        assert '"name":"Novomodelo Mean"' in html
        assert '"name":"NEWAVE"' not in html


def _thermal_percentiles_fixture() -> pl.DataFrame:
    """Novomodelo p10/p50/p90 for the one ``_aligned_fixture`` thermal entity."""
    return pl.DataFrame(
        {
            "entity_id": [0],
            "stage_id": [0],
            "generation_mw_p10": [24.0],
            "generation_mw_p50": [28.0],
            "generation_mw_p90": [32.0],
        }
    )


def _patch_thermal_percentiles(
    monkeypatch: pytest.MonkeyPatch, percentiles: pl.DataFrame | None = None
) -> None:
    """Stub the novomodelo thermal-percentile reader -- defaults to
    empty, matching how a Novomodelo run with no thermal percentile output
    (e.g. the deterministic 2-node tree) degrades in production."""
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
        "read_novomodelo_thermal_percentiles",
        lambda *_a, **_k: pl.DataFrame() if percentiles is None else percentiles,
    )


class TestBuildDecompDatasetThermalDetail:
    """Fills ``PercentileData.thermal`` -- the disjoint thermal
    counterpart to the hydro percentile band -- for the shared
    Thermal Operation and Thermal Plant Details tabs."""

    def test_thermal_percentiles_populate_metadata_when_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_thermal_percentiles(monkeypatch, _thermal_percentiles_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        thermal_pct = dataset.render.thermal
        assert isinstance(thermal_pct, pl.DataFrame)
        assert not thermal_pct.is_empty()
        assert "generation_mw_p10" in thermal_pct.columns
        assert "generation_mw_p50" in thermal_pct.columns
        assert "generation_mw_p90" in thermal_pct.columns

    def test_thermal_percentiles_stay_empty_when_novomodelo_output_lacks_them(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No percentile mock (deterministic-tree low-N, master-plan caveat
        1): ``read_novomodelo_thermal_percentiles`` degrades to its own
        empty-frame default and the dataset must not fabricate a spread."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_thermal_percentiles(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        thermal_pct = dataset.render.thermal
        assert isinstance(thermal_pct, pl.DataFrame)
        assert thermal_pct.is_empty()

    def test_thermal_tidy_rows_carry_generation_mw_and_known_sources(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """E1 already emits the thermal ``ResultComparison`` rows -- the
        thermal-detail build must not re-emit or re-map them."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        thermal_rows = dataset.tidy.filter(pl.col("entity_type") == "thermal")
        assert not thermal_rows.is_empty()
        assert set(thermal_rows["variable"].to_list()) == {"generation_mw"}
        assert set(thermal_rows["source"].to_list()) == {"newave", "novomodelo"}

    def test_thermal_operation_and_detail_tabs_render_with_metadata_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_thermal_percentiles(monkeypatch, _thermal_percentiles_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        assert "Thermal Generation Comparison" in html
        assert 'id="tab-thermal-detail"' in html
        assert 'id="thermal-select"' in html
        assert "Plotly.newPlot" in html

    def test_both_thermal_tabs_render_without_exception_when_percentiles_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No novomodelo thermal percentiles at all -- both tabs must degrade
        gracefully (Novomodelo-only band suppressed), never raise."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_thermal_percentiles(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)  # must not raise

        assert "Thermal Generation Comparison" in html
        assert 'id="tab-thermal-detail"' in html
