"""FPHA fit-fidelity tests for ``comparators.decomp.results``.

Covers the "Fitted production functions (FPHA)" fallback-(b) metrics --
Novomodelo's fitted envelope evaluated at the source model's realized operating
points, compared to the source model's own realized generation -- and the
Productivity tab's FPHA metadata in ``build_decomp_dataset``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.decomp.results import (
    _fpha_metrics,
    build_decomp_dataset,
)
from novomodelo_bridge.comparators.report_builder import build_comparison_report
from novomodelo_bridge.decomp.id_map import DecompIdMap
from tests.comparators.conftest import (
    _aligned_fixture,
    _patch_aligned_frames,
    _patch_shared_case,
)


def _fpha_id_map() -> DecompIdMap:
    """One hydro plant (code 10 -> novomodelo id 0), matching `_aligned_fixture`'s
    own hydro code/id/name so the SAME `_patch_aligned_frames` fixture can
    back both the E1 result rows and the FPHA metrics in the same
    ``build_decomp_dataset`` test."""
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(10, 20))


def _fpha_novomodelo_planes_fixture() -> pl.DataFrame:
    """One fitted Novomodelo plane for (hydro_id=0, stage_id=0): ``GH = q``
    (``kappa=1``, every other coefficient zero except ``gamma_q``) --
    deliberately trivial so the envelope's value at any point is just its
    own ``q_m3s``, matching `novomodelo_readers.read_novomodelo_fpha_planes`'s own
    ``hydro_id``/``stage_id``/``gamma_0``/``gamma_v``/``gamma_q``/
    ``gamma_s``/``kappa`` schema."""
    return pl.DataFrame(
        {
            "hydro_id": [0],
            "stage_id": [0],
            "gamma_0": [0.0],
            "gamma_v": [0.0],
            "gamma_q": [1.0],
            "gamma_s": [0.0],
            "kappa": [1.0],
        }
    )


def _fpha_deviations_fixture() -> pl.DataFrame:
    """One realized source-model operating point for hydro code 10, stage 1
    (``estagio``, 1-based): ``vazao_turbinada_m3s=80`` (so Novomodelo's envelope
    evaluates to 80 MW under the trivial plane above) against the source
    model's own LP-consumed ``geracao_hidraulica_fpha=76`` -- a deliberate,
    exact 4 MW gap so nmae/bias/max_abs_dev/gh_max_ratio are hand-checkable.
    Matches `read_dec_desvfpha`'s real column set (a subset sufficient for
    `_fpha_metrics`)."""
    return pl.DataFrame(
        {
            "codigo_usina": [10],
            "estagio": [1],
            "volume_total_hm3": [500.0],
            "vazao_turbinada_m3s": [80.0],
            "vazao_vertida_m3s": [0.0],
            "geracao_hidraulica_fpha": [76.0],
        }
    )


def _patch_fpha_planes_and_deviations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire `build_decomp_dataset`'s three sources -- outside
    `_read_aligned_frames` -- to the fixtures above: Novomodelo's planes reader,
    the deck id map (the shared case's ``id_map``, reused verbatim by
    `_fpha_metrics` rather than rebuilt), and the source model's own
    deviation table. ``read_eco_fpha``/``read_dec_estatfpha`` are left
    unmocked -- they raise `FileNotFoundError` against a bare `tmp_path`,
    exercising `_fpha_metrics`'s own graceful degrade for those two optional
    sources (``n_v`` stays null; the deck-wide summary is only logged).
    """
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.novomodelo_readers.read_novomodelo_fpha_planes",
        lambda *_a, **_k: _fpha_novomodelo_planes_fixture(),
    )
    _patch_shared_case(monkeypatch, id_map=_fpha_id_map())
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.read_dec_desvfpha",
        lambda *_a, **_k: _fpha_deviations_fixture(),
    )


class TestFphaMetrics:
    """`_fpha_metrics`: fallback (b)'s per-(hydro, stage) fit-fidelity table
    -- Novomodelo's own fitted envelope evaluated at the source model's realized
    operating points, compared to the source model's own realized
    ``geracao_hidraulica_fpha``."""

    def test_both_sides_present_computes_real_cross_solver_metrics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_fpha_planes",
            lambda *_a, **_k: _fpha_novomodelo_planes_fixture(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_desvfpha",
            lambda *_a, **_k: _fpha_deviations_fixture(),
        )

        metrics = _fpha_metrics(tmp_path, tmp_path, _fpha_id_map(), {0: "A", 1: "B"})

        assert metrics is not None
        assert metrics.height == 1
        row = metrics.row(0, named=True)
        assert row["novomodelo_id"] == 0
        assert row["plant_name"] == "A"
        assert row["stage"] == 0
        # No declared FPHA reader counts the source model's own planes.
        assert row["n_planes_newave"] is None
        assert row["n_planes_novomodelo"] == 1
        # `read_eco_fpha` is unmocked -> raises against a bare tmp_path ->
        # degrades to null, never fabricated.
        assert row["n_v"] is None
        assert row["nmae"] == pytest.approx(4.0 / 76.0)
        assert row["bias"] == pytest.approx(4.0 / 76.0)
        assert row["max_abs_dev"] == pytest.approx(4.0)
        assert row["gh_max_ratio"] == pytest.approx(80.0 / 76.0)

    def test_novomodelo_has_no_planes_returns_none(self, tmp_path: Path) -> None:
        """`read_novomodelo_fpha_planes` naturally returns `None` against a bare
        Novomodelo output dir (no ``hydro_models/fpha_hyperplanes.parquet``)."""
        assert _fpha_metrics(tmp_path, tmp_path, _fpha_id_map(), {}) is None

    def test_no_id_map_returns_none_even_with_novomodelo_planes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_fpha_planes",
            lambda *_a, **_k: _fpha_novomodelo_planes_fixture(),
        )
        assert _fpha_metrics(tmp_path, tmp_path, None, {}) is None

    def test_missing_source_deviation_table_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_fpha_planes",
            lambda *_a, **_k: _fpha_novomodelo_planes_fixture(),
        )
        # `read_dec_desvfpha` left unmocked -> raises FileNotFoundError.
        assert _fpha_metrics(tmp_path, tmp_path, _fpha_id_map(), {}) is None

    def test_unresolvable_hydro_code_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A deviation table whose plant code the id map cannot resolve at
        all (no hydro codes declared) never reaches the Novomodelo planes it
        would otherwise match."""
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_fpha_planes",
            lambda *_a, **_k: _fpha_novomodelo_planes_fixture(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_desvfpha",
            lambda *_a, **_k: _fpha_deviations_fixture(),
        )
        no_hydros = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
        assert _fpha_metrics(tmp_path, tmp_path, no_hydros, {}) is None


class TestBuildDecompDatasetFpha:
    """Fills ``dataset.render.fpha_metrics``;
    ``fpha_surface``/``fpha_spill`` always stay `None` (fallback (b))."""

    def test_novomodelo_has_no_planes_fpha_metrics_absent_no_section(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``read_novomodelo_fpha_planes`` returns ``None`` (the default on a
        bare Novomodelo output dir) -> no FPHA metadata, the report omits the
        section entirely, no exception."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.metadata.get("fpha_metrics") is None
        html = build_comparison_report(dataset)  # must not raise
        assert "Fitted production functions (FPHA)" not in html

    def test_both_sides_fitted_planes_fpha_metrics_populated_section_renders(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Mocked frames on both sides -> a non-empty ``fpha_metrics``
        with the required columns, and the report renders the FPHA section
        title."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_fpha_planes_and_deviations(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        fpha_metrics = dataset.render.fpha_metrics
        assert isinstance(fpha_metrics, pl.DataFrame)
        assert not fpha_metrics.is_empty()
        for column in ("novomodelo_id", "plant_name", "nmae", "bias"):
            assert column in fpha_metrics.columns

        html = build_comparison_report(dataset)  # must not raise
        assert "Fitted production functions (FPHA)" in html

    def test_fpha_surface_and_spill_always_stay_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Fallback (b): never a dense surface, even once ``fpha_metrics``
        is populated -- the section renders the metrics table but the
        surface/spill widget falls back to its own "No production-function
        (FPHA) data available" placeholder."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_fpha_planes_and_deviations(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.metadata.get("fpha_surface") is None
        assert dataset.metadata.get("fpha_spill") is None
        html = build_comparison_report(dataset)  # must not raise
        assert "No production-function (FPHA) data available." in html

    def test_report_renders_without_exception_present_and_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``build_comparison_report`` raises no exception whether the
        FPHA section is present or omitted."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        dataset_absent = build_decomp_dataset(tmp_path, tmp_path)
        build_comparison_report(dataset_absent)  # must not raise

        _patch_fpha_planes_and_deviations(monkeypatch)
        dataset_present = build_decomp_dataset(tmp_path, tmp_path)
        build_comparison_report(dataset_present)  # must not raise
