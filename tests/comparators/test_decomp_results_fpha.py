"""Operating-point comparison wiring in ``comparators.decomp.results``.

``build_decomp_dataset`` hands the frames of
:func:`~novomodelo_bridge.comparators.decomp.operating_points.compare_operating_points`
to the report: the productivity split, the fit inputs and slices, the tie
share, the FPHA surface metrics, and the hydro rows averaged over the points
that are not ties.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.decomp.operating_points import (
    FPHA_FIT_INPUTS_SCHEMA,
    FPHA_SLICE_SCHEMA,
    MASKED_HYDRO_SCHEMA,
    TIE_SHARE_SCHEMA,
    OperatingPointComparison,
)
from novomodelo_bridge.comparators.decomp.results import build_decomp_dataset
from novomodelo_bridge.comparators.fpha import (
    FPHA_METRICS_SCHEMA,
    FPHA_SPILL_SCHEMA,
    FPHA_SPLIT_SCHEMA,
    FPHA_SURFACE_SCHEMA,
)
from novomodelo_bridge.comparators.report_builder import build_comparison_report
from tests.comparators.conftest import _aligned_fixture, _patch_aligned_frames

_SPLIT_TITLE = "Why the realized productivities differ"
_SLICE_TITLE = "Production function at the operating point"
_SURFACE_TITLE = "Fitted production functions (FPHA)"
_MASKED_NOTE = "averaged over the points that are not ties"


def _row(schema: dict[str, type[pl.DataType]], **values: object) -> pl.DataFrame:
    return pl.DataFrame([{c: values.get(c) for c in schema}], schema=schema)


def _comparison(*, with_planes: bool = True) -> OperatingPointComparison:
    """Plant A (Novomodelo id 0) in stage 0, a fifth of its points tied."""
    keys = {"novomodelo_id": 0, "plant_name": "A", "stage": 0}
    split = _row(
        FPHA_SPLIT_SCHEMA,
        **keys,
        tie_share=0.2,
        q_source_m3s=80.0,
        gh_source_mw=120.0,
        rho_source=1.5,
        rho_novomodelo=1.41,
        d_total=-0.06,
        d_model=0.01,
        d_operating_point=-0.0693,
    )
    fit_inputs = _row(
        FPHA_FIT_INPUTS_SCHEMA, **keys, source_v_hm3=500.0, source_q_m3s=80.0
    )
    slices = pl.DataFrame(
        [
            (0, 0, curve, q, 1.5 * q)
            for curve in ("source_planes", "exact")
            for q in (0.0, 100.0)
        ],
        schema=FPHA_SLICE_SCHEMA,
        orient="row",
    )
    metrics = _row(
        FPHA_METRICS_SCHEMA, **keys, n_planes_novomodelo=1, nmae=0.05, bias=0.05
    )
    if not with_planes:
        split, fit_inputs, slices, metrics = (
            pl.DataFrame(schema=frame.schema)
            for frame in (split, fit_inputs, slices, metrics)
        )
    return OperatingPointComparison(
        split=split,
        fit_inputs=fit_inputs,
        slices=slices,
        metrics=metrics,
        surface=pl.DataFrame(schema=FPHA_SURFACE_SCHEMA),
        spill=pl.DataFrame(schema=FPHA_SPILL_SCHEMA),
        tie_share=_row(
            TIE_SHARE_SCHEMA, stage=0, source_tie_share=0.2, novomodelo_tie_share=0.1
        ),
        masked_hydro=_row(
            MASKED_HYDRO_SCHEMA,
            entity_type="hydro",
            entity_name="A",
            newave_code=10,
            novomodelo_id=0,
            stage=0,
            variable="generation_mw",
            newave_value=100.0,
            novomodelo_value=100.0,
            abs_diff=0.0,
            rel_diff=0.0,
        ),
    )


def _patch_comparison(
    monkeypatch: pytest.MonkeyPatch, comparison: OperatingPointComparison | None
) -> list[tuple[object, ...]]:
    calls: list[tuple[object, ...]] = []

    def _fake(*args: object) -> OperatingPointComparison | None:
        calls.append(args)
        return comparison

    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.compare_operating_points", _fake
    )
    return calls


class TestBuildDecompDatasetOperatingPoints:
    def test_the_comparison_receives_the_runs_and_the_plant_names(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        calls = _patch_comparison(monkeypatch, None)
        decomp_dir, novomodelo_dir = tmp_path / "decomp", tmp_path / "novomodelo"

        build_decomp_dataset(decomp_dir, novomodelo_dir)

        ((source, _case, novomodelo, _probabilities, names),) = calls
        assert (source, novomodelo) == (decomp_dir, novomodelo_dir)
        assert names == _aligned_fixture().hydro_names

    def test_no_comparison_leaves_every_section_out(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_comparison(monkeypatch, None)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.render.fpha_split.is_empty()
        assert dataset.render.tie_share.is_empty()
        assert dataset.render.hydro_masked.is_empty()
        html = build_comparison_report(dataset)
        for text in (_SPLIT_TITLE, _SLICE_TITLE, _SURFACE_TITLE, _MASKED_NOTE):
            assert text not in html

    def test_comparison_frames_reach_the_productivity_tab(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_comparison(monkeypatch, _comparison())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.render.fpha_split.height == 1
        assert dataset.render.fpha_metrics.height == 1
        html = build_comparison_report(dataset)
        for text in (_SPLIT_TITLE, _SLICE_TITLE, _SURFACE_TITLE):
            assert text in html
        # No dense surface on this track: the surface widget shows its
        # placeholder under the fidelity table.
        assert "No production-function (FPHA) data available." in html

    def test_ties_without_planes_render_only_the_tie_share(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_comparison(monkeypatch, _comparison(with_planes=False))

        html = build_comparison_report(build_decomp_dataset(tmp_path, tmp_path))

        assert "Tie points" in html
        for text in (_SPLIT_TITLE, _SLICE_TITLE, _SURFACE_TITLE):
            assert text not in html

    def test_masked_rows_rank_the_hydro_detail_tab(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_comparison(monkeypatch, _comparison())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.render.hydro_masked.height == 1
        assert _MASKED_NOTE in build_comparison_report(dataset)
