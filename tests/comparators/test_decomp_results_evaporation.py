"""Evaporation comparison tests for ``comparators.decomp.results``.

Covers hm³ -> m³/s stage-hours conversion, Novomodelo stage-hours lookup, the
source-model evaporated-volume side, the full evaporation result-comparison
reconciliation, and the Hydro Plant Detail tab's ``evaporation_m3s`` rows in
``build_decomp_dataset``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.decomp.results import (
    _HM3_TO_M3S_HOUR_FACTOR,
    _AlignedDecompFrames,
    _evap_side,
    _evaporation_result_comparisons,
    _hm3_to_m3s,
    _novomodelo_stage_hours,
    build_decomp_dataset,
)
from novomodelo_bridge.comparators.report_builder import build_comparison_report
from novomodelo_bridge.core import diagnostics as dx
from tests.comparators.conftest import (
    _aligned_fixture,
    _extract_tab_content,
    _patch_aligned_frames,
    _patch_shared_case,
    _ree_id_map,
)


def _write_stages_json(case_dir: Path, stage_hours: dict[int, list[float]]) -> Path:
    """Write a minimal ``stages.json`` -- *stage_hours* maps
    ``stage_id -> [block_hours, ...]``. Mirrors ``test_novomodelo_readers.py``'s
    own ``_write_stages_json`` helper shape, duplicated locally rather than
    imported so this file keeps no cross-test-module dependency."""
    data = {
        "stages": [
            {
                "id": stage_id,
                "blocks": [
                    {"id": bid, "hours": hours} for bid, hours in enumerate(block_hours)
                ],
            }
            for stage_id, block_hours in stage_hours.items()
        ]
    }
    path = case_dir / "stages.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestHm3ToM3s:
    """`_hm3_to_m3s`: hm³ (stage volume) -> m³/s (mean flow), driven by the
    stage's own hours."""

    def test_matches_the_fixed_monthly_factor_at_730_hours(self) -> None:
        """1 m³/s sustained over a 730h month deposits exactly
        ``core.units.C_M3S2HM3`` (2.628 hm³); converting that volume
        back at 730h must recover 1.0 m³/s exactly."""
        assert _hm3_to_m3s(2.628, 730.0) == pytest.approx(1.0)

    def test_sub_monthly_stage_uses_its_own_hours_not_730(self) -> None:
        """A 168h (weekly) DECOMP stage: 1 m³/s deposits
        168 * 3600 / 1e6 = 0.6048 hm³ over that stage -- converting back at
        the stage's own 168h must recover 1.0 m³/s, not the ~0.23 m³/s a
        (wrong) fixed 730h monthly divisor would yield."""
        assert _hm3_to_m3s(0.6048, 168.0) == pytest.approx(1.0)
        assert _hm3_to_m3s(0.6048, 730.0) == pytest.approx(168.0 / 730.0)

    def test_hour_factor_matches_the_converter_side_monthly_constant(self) -> None:
        """`_HM3_TO_M3S_HOUR_FACTOR * 730` reproduces
        ``core.units.C_M3S2HM3`` (2.628) -- the same physical
        relationship, generalized from the fixed monthly constant to any
        stage's own hours."""
        assert _HM3_TO_M3S_HOUR_FACTOR * 730.0 == pytest.approx(2.628)


class TestNovomodeloStageHours:
    """`_novomodelo_stage_hours`: per-stage total hours from the Novomodelo case's own
    ``stages.json``, via `novomodelo_readers._load_block_hours`."""

    def test_sums_block_hours_per_stage(self, tmp_path: Path) -> None:
        _write_stages_json(tmp_path, {0: [24.0, 144.0], 1: [168.0]})

        hours = _novomodelo_stage_hours(tmp_path)

        assert hours == {0: pytest.approx(168.0), 1: pytest.approx(168.0)}

    def test_no_stages_json_returns_empty_dict(self, tmp_path: Path) -> None:
        assert _novomodelo_stage_hours(tmp_path) == {}


def _evap_dec_oper_evap_fixture() -> pl.DataFrame:
    """One stage (``estagio=1``), two nodes, two plants (codes 10, 20 --
    matching ``_ree_id_map()``'s hydro codes): the scenario mean is
    ``1.2 hm³`` for plant 10 and ``2.2 hm³`` for plant 20, deliberately not
    round numbers so the hm³ -> m³/s conversion is hand-checkable rather than
    trivially exact."""
    return pl.DataFrame(
        {
            "estagio": [1, 1, 1, 1],
            "no": [1, 2, 1, 2],
            "cenario": [1, 1, 1, 1],
            "codigo_usina": [10, 10, 20, 20],
            "nome_usina": ["A", "A", "B", "B"],
            "evaporacao_calculada_hm3": [1.0, 1.4, 2.0, 2.4],
        }
    )


def _evap_aligned_fixture() -> _AlignedDecompFrames:
    """``_aligned_fixture()`` with its ``novomodelo_hydro`` extended to carry
    ``evaporation_m3s`` -- the base fixture only carries E1's
    ``_HYDRO_VARIABLES`` columns. Plant 0 (code 10) = 2.5 m³/s, plant 1
    (code 20) = 3.0 m³/s."""
    base = _aligned_fixture()
    return dataclasses.replace(
        base,
        novomodelo_hydro=base.novomodelo_hydro.with_columns(
            pl.Series("evaporation_m3s", [2.5, 3.0])
        ),
    )


def _patch_evap_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage_hours: dict[int, list[float]] | None = None,
) -> None:
    """Wire ``read_dec_oper_evap`` (outside ``_read_aligned_frames``) to
    :func:`_evap_dec_oper_evap_fixture`, and ``_novomodelo_stage_hours`` to a
    fixed one-stage 168h lookup unless *stage_hours* overrides it."""
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.read_dec_oper_evap",
        lambda *_a, **_k: _evap_dec_oper_evap_fixture(),
    )
    hours = {0: 168.0} if stage_hours is None else stage_hours
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results._novomodelo_stage_hours",
        lambda *_a, **_k: hours,
    )


class TestEvapSide:
    """`_evap_side`: the source model's per-(hydro, stage) evaporated volume,
    scenario-averaged and mapped onto Novomodelo ids -- mirrors `_hydro_side`'s
    own fold exactly."""

    def test_scenario_means_and_maps_to_novomodelo_ids(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_evap",
            lambda *_a, **_k: _evap_dec_oper_evap_fixture(),
        )

        frame, unmapped = _evap_side(tmp_path, {10: 0, 20: 1})

        assert unmapped == []
        by_id = {
            int(row["entity_id"]): row["evaporacao_calculada_hm3"]
            for row in frame.iter_rows(named=True)
        }
        assert by_id == {0: pytest.approx(1.2), 1: pytest.approx(2.2)}
        assert frame["stage_id"].unique().to_list() == [0]

    def test_unmapped_code_is_reported_not_dropped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_evap",
            lambda *_a, **_k: _evap_dec_oper_evap_fixture(),
        )

        frame, unmapped = _evap_side(tmp_path, {10: 0})  # code 20 unmapped

        assert unmapped == [20]
        assert frame["entity_id"].to_list() == [0]

    def test_missing_table_raises(self, tmp_path: Path) -> None:
        """``read_dec_oper_evap`` unmocked against a bare ``tmp_path`` raises
        -- the caller (``_evaporation_result_comparisons``) is what degrades
        this to an absent evaporation section, not this helper."""
        with pytest.raises(FileNotFoundError):
            _evap_side(tmp_path, {10: 0})


class TestEvaporationResultComparisons:
    """`_evaporation_result_comparisons`: the full evaporation reconciliation
    -- scenario-averaged source-model volume converted to m³/s via the
    stage's own hours, joined against Novomodelo's ``evaporation_m3s``, with a
    one-sided plant excluded from the pairing and counted rather than
    silently dropped."""

    def test_paired_rows_use_the_stage_hours_conversion(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_evap_sources(monkeypatch)

        results, one_sided = _evaporation_result_comparisons(
            tmp_path,
            tmp_path,
            _evap_aligned_fixture().novomodelo_hydro,
            _ree_id_map(),
            {0: "A", 1: "B"},
        )

        assert one_sided == []
        by_id = {r.novomodelo_id: r for r in results}
        assert set(by_id) == {0, 1}

        for r in results:
            assert r.entity_type == "hydro"
            assert r.variable == "evaporation_m3s"

        # Plant 0 (code 10): scenario-mean 1.2 hm³ over a 168h stage.
        assert by_id[0].newave_code == 10
        assert by_id[0].newave_value == pytest.approx(_hm3_to_m3s(1.2, 168.0))
        assert by_id[0].novomodelo_value == pytest.approx(2.5)
        # Plant 1 (code 20): scenario-mean 2.2 hm³ over the same stage.
        assert by_id[1].newave_code == 20
        assert by_id[1].newave_value == pytest.approx(_hm3_to_m3s(2.2, 168.0))
        assert by_id[1].novomodelo_value == pytest.approx(3.0)

    def test_plant_present_only_on_novomodelo_side_excluded_and_counted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A hydro present in Novomodelo's ``evaporation_m3s`` but absent from
        the source model's own evaporation table is excluded from the paired
        comparison and counted."""
        # Source model reports evaporation for plant 10 (novomodelo id 0) only.
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_evap",
            lambda *_a, **_k: _evap_dec_oper_evap_fixture().filter(
                pl.col("codigo_usina") == 10
            ),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results._novomodelo_stage_hours",
            lambda *_a, **_k: {0: 168.0},
        )

        with dx.collect() as collected:
            results, one_sided = _evaporation_result_comparisons(
                tmp_path,
                tmp_path,
                _evap_aligned_fixture().novomodelo_hydro,  # carries ids 0 AND 1
                _ree_id_map(),
                {0: "A", 1: "B"},
            )

        assert [r.novomodelo_id for r in results] == [0]
        assert one_sided == [1]
        assert len(collected) == 1
        assert collected[0].code == "evaporation-plant-one-sided"
        assert "1" in " ".join(str(n) for n in collected[0].notes)

    def test_plant_present_only_on_source_side_excluded_and_counted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """AC (reverse direction): a hydro present in the source model's own
        evaporation table but absent from Novomodelo's ``evaporation_m3s`` is
        excluded from the paired comparison and counted."""
        _patch_evap_sources(monkeypatch)
        # Novomodelo reports evaporation_m3s for plant 0 (code 10) only.
        novomodelo_hydro = _evap_aligned_fixture().novomodelo_hydro.filter(
            pl.col("entity_id") == 0
        )

        with dx.collect() as collected:
            results, one_sided = _evaporation_result_comparisons(
                tmp_path, tmp_path, novomodelo_hydro, _ree_id_map(), {0: "A", 1: "B"}
            )

        assert [r.novomodelo_id for r in results] == [0]
        assert one_sided == [1]
        assert len(collected) == 1
        assert collected[0].code == "evaporation-plant-one-sided"

    def test_none_id_map_returns_no_rows_and_no_unmapped(self, tmp_path: Path) -> None:
        results, one_sided = _evaporation_result_comparisons(
            tmp_path, tmp_path, _evap_aligned_fixture().novomodelo_hydro, None, {}
        )

        assert results == []
        assert one_sided == []

    def test_no_source_evaporation_table_degrades_to_no_rows(
        self, tmp_path: Path
    ) -> None:
        """``read_dec_oper_evap`` left unmocked -> raises ``FileNotFoundError``
        -- a genuinely unavailable section, not a per-plant gap."""
        with dx.collect() as collected:
            results, one_sided = _evaporation_result_comparisons(
                tmp_path,
                tmp_path,
                _evap_aligned_fixture().novomodelo_hydro,
                _ree_id_map(),
                {0: "A", 1: "B"},
            )

        assert results == []
        assert one_sided == []
        assert collected == []

    def test_no_evaporation_m3s_column_on_novomodelo_side_degrades_to_no_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_evap_sources(monkeypatch)

        results, one_sided = _evaporation_result_comparisons(
            tmp_path,
            tmp_path,
            _aligned_fixture().novomodelo_hydro,  # no evaporation_m3s column
            _ree_id_map(),
            {0: "A", 1: "B"},
        )

        assert results == []
        assert one_sided == []

    def test_no_stage_hours_available_degrades_to_no_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No ``stages.json`` reconciliation denominator -- degrades
        gracefully rather than raising or fabricating a divisor."""
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_evap",
            lambda *_a, **_k: _evap_dec_oper_evap_fixture(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results._novomodelo_stage_hours",
            lambda *_a, **_k: {},
        )

        results, one_sided = _evaporation_result_comparisons(
            tmp_path,
            tmp_path,
            _evap_aligned_fixture().novomodelo_hydro,
            _ree_id_map(),
            {0: "A", 1: "B"},
        )

        assert results == []
        assert one_sided == []


class TestBuildDecompDatasetEvaporation:
    """Fills ``results`` with per-(hydro, stage)
    ``evaporation_m3s`` rows and ``dataset.metadata["unmapped"]["evaporation"]``."""

    def test_no_deck_no_evaporation_rows_and_empty_unmapped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The shared case's ``id_map`` is a degenerate, empty (but valid)
        ``DecompIdMap`` against a bare ``tmp_path`` (via
        ``_patch_aligned_frames``'s default) -> no evaporation rollup, empty
        ``unmapped["evaporation"]``, no exception."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        evap_rows = dataset.tidy.filter(
            (pl.col("entity_type") == "hydro")
            & (pl.col("variable") == "evaporation_m3s")
        )
        assert evap_rows.is_empty()
        assert dataset.metadata["unmapped"]["evaporation"] == []
        build_comparison_report(dataset)  # must not raise

    def test_both_sides_present_tidy_carries_evaporation_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``tidy`` has ``entity_type=="hydro"``/
        ``variable=="evaporation_m3s"`` rows with ``source`` in
        {"newave", "novomodelo"}, and ``dataset.summary`` includes the
        ``evaporation_m3s`` variable."""
        _patch_aligned_frames(monkeypatch, _evap_aligned_fixture())
        _patch_shared_case(monkeypatch, id_map=_ree_id_map())
        _patch_evap_sources(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        evap_rows = dataset.tidy.filter(
            (pl.col("entity_type") == "hydro")
            & (pl.col("variable") == "evaporation_m3s")
        )
        assert set(evap_rows["source"].unique().to_list()) == {"newave", "novomodelo"}
        assert evap_rows.height == 4  # 2 plants * 2 sources
        assert dataset.metadata["unmapped"]["evaporation"] == []

        summary_vars = set(dataset.summary["variable"].to_list())
        assert "evaporation_m3s" in summary_vars

    def test_one_sided_plant_counted_in_dataset_metadata(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _evap_aligned_fixture())
        _patch_shared_case(monkeypatch, id_map=_ree_id_map())
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_evap",
            lambda *_a, **_k: _evap_dec_oper_evap_fixture().filter(
                pl.col("codigo_usina") == 10
            ),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results._novomodelo_stage_hours",
            lambda *_a, **_k: {0: 168.0},
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.metadata["unmapped"]["evaporation"] == [1]

    def test_hydro_plant_detail_tab_carries_evaporation_content(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``build_comparison_report(dataset)`` renders the Hydro Plant
        Details tab's ``evaporation_m3s`` panel content -- the exact token
        ``report_builder._HYDRO_VARIABLES`` already wires into that tab, so no
        new chart is required."""
        _patch_aligned_frames(monkeypatch, _evap_aligned_fixture())
        _patch_shared_case(monkeypatch, id_map=_ree_id_map())
        _patch_evap_sources(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        assert 'id="tab-hydro-detail"' in html
        detail_tab = _extract_tab_content(html, "tab-hydro-detail")
        # The per-plant JSON payload's evaporation_m3s series is non-empty
        # for the one stage both fixture plants carry it at (compact JSON,
        # no spaces -- see ``ui.html.json_for_script``).
        assert '"evaporation_m3s_stages":[0]' in detail_tab
