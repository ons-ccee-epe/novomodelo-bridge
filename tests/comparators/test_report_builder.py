"""Tests for the reference-label indirection.

``build_comparison_report`` and ``print_results_summary_from_dataset`` hard-coded
the reference series' display label as the literal string ``"NEWAVE"`` across
~70 sites in ``charts.py`` / ``report_builder.py`` / ``report.py``. The indirection
threads an additive ``reference_label: str = "NEWAVE"`` keyword through every one
of those sites (explicit parameter passing, no module-level render state, no
post-render string rewrite) so a future DECOMP report can pass
``reference_label="DECOMP"`` instead of mislabeling the DECOMP series as
"NEWAVE".

This module:

* GOLDEN-guards that the default (no ``reference_label`` argument) renders
  byte-identical HTML / terminal output to the baseline behaviour, by reusing
  the exact fixture/golden pair ``tests/dashboard/test_chart_helpers.py`` already uses for
  ``build_comparison_report``'s dataset-seam golden test
  (``tests/golden/build_comparison_report_full.html``) — the ground truth
  captured before the reference_label change.
* Proves every one of the ~70 sites was threaded (no site missed) by asserting
  that ``reference_label="DECOMP"`` leaves zero "NEWAVE" substrings in the
  rendered HTML / printed terminal text, while the Novomodelo-side labels are
  unchanged.

No real NEWAVE/DECOMP case data is read (Tier 1 — CI, no ``example/``); the
dataset is built from the same hermetic in-memory fixtures
``test_chart_helpers.py`` uses for its own golden test.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest
from rich.console import Console
from typer.testing import CliRunner

from novomodelo_bridge.comparators.analyze import build_results_dataset
from novomodelo_bridge.comparators.dataset import ComparisonDataset
from novomodelo_bridge.comparators.model import PercentileData, ResultComparison
from novomodelo_bridge.comparators.report_builder import build_comparison_report
from novomodelo_bridge.comparators.verdict import build_compare_verdict
from novomodelo_bridge.ui.compare_summary import print_results_summary_from_dataset
from tests.comparators.conftest import _extract_tab_content
from tests.dashboard.test_chart_helpers import (
    _report_fixture_pct,
    _report_fixture_results,
)
from tests.golden_utils import assert_html_golden

_NW_DIR = Path("/fake/nw")
_NOVOMODELO_DIR = Path("/fake/novomodelo")


def _capture(func: Callable[..., None], *args: object, **kwargs: object) -> str:
    """Run a stdout-writing printer and return the captured text."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()


def _build_dataset() -> ComparisonDataset:
    """Build the same hermetic dataset used by test_chart_helpers's own golden test."""
    return build_results_dataset(_report_fixture_results(), _report_fixture_pct(), 0.05)


# ---------------------------------------------------------------------------
# GOLDEN: build_comparison_report(dataset) at the default label is
# byte-identical to the baseline output.
# ---------------------------------------------------------------------------


def test_build_comparison_report_default_label_matches_golden() -> None:
    """No ``reference_label`` argument renders byte-identical to the baseline HTML.

    Reuses the exact golden file
    ``tests/golden/build_comparison_report_full.html`` that
    ``tests.dashboard.test_chart_helpers.test_build_comparison_report_dataset_golden``
    already guards — captured from the baseline ``charts.py`` /
    ``report_builder.py`` — to prove the additive ``reference_label`` parameter
    changes nothing at its ``"NEWAVE"`` default.
    """
    html = build_comparison_report(_build_dataset())

    assert_html_golden(html, "build_comparison_report_full.html")


def test_build_comparison_report_explicit_newave_label_matches_golden() -> None:
    """Explicitly passing ``reference_label="NEWAVE"`` matches the default."""
    html = build_comparison_report(_build_dataset(), reference_label="NEWAVE")

    assert_html_golden(html, "build_comparison_report_full.html")


# ---------------------------------------------------------------------------
# DECOMP label: every reference-series trace name / chart title / report
# title / prose string reads "DECOMP" and zero "NEWAVE" substrings remain,
# proving no site was missed. The Novomodelo-side labels are unchanged.
# ---------------------------------------------------------------------------


def test_build_comparison_report_decomp_label_has_no_newave_substring() -> None:
    """``reference_label="DECOMP"`` relabels every site; none was left behind.

    A single missed site among the ~70 (a trace name, chart title, hover
    template, or prose string still hard-coding "NEWAVE") would surface here as
    a leftover "NEWAVE" substring in the rendered report.
    """
    html = build_comparison_report(_build_dataset(), reference_label="DECOMP")

    assert "NEWAVE" not in html
    assert "DECOMP" in html


def test_build_comparison_report_decomp_label_novomodelo_side_unchanged() -> None:
    """The Novomodelo-side series labels are untouched by the reference_label swap."""
    html_default = build_comparison_report(_build_dataset())
    html_decomp = build_comparison_report(_build_dataset(), reference_label="DECOMP")

    assert "Novomodelo" in html_default
    assert "Novomodelo" in html_decomp
    # "Novomodelo Mean" is the most common reference-series-adjacent trace name;
    # confirm it is present verbatim (untouched) in both renders.
    assert "Novomodelo Mean" in html_default
    assert "Novomodelo Mean" in html_decomp


def test_build_comparison_report_decomp_label_report_title() -> None:
    """The document title and report header read the DECOMP label."""
    html = build_comparison_report(_build_dataset(), reference_label="DECOMP")

    assert "Novomodelo vs DECOMP Results Comparison" in html
    assert "Novomodelo vs NEWAVE Results Comparison" not in html


def test_build_comparison_report_default_label_report_title() -> None:
    """The document title and report header keep the NEWAVE label by default."""
    html = build_comparison_report(_build_dataset())

    assert "Novomodelo vs NEWAVE Results Comparison" in html


# ---------------------------------------------------------------------------
# print_results_summary_from_dataset: default unchanged; "DECOMP" relabels.
# ---------------------------------------------------------------------------


def test_print_results_summary_default_label_unchanged() -> None:
    """No ``reference_label`` argument prints the default "NEWAVE" header/labels."""
    dataset = _build_dataset()

    text = _capture(
        print_results_summary_from_dataset,
        dataset,
        _NW_DIR,
        _NOVOMODELO_DIR,
        verdict=build_compare_verdict(dataset),
    )

    assert "Novomodelo vs NEWAVE Results Comparison" in text
    assert f"NEWAVE case:  {_NW_DIR}" in text
    assert f"Novomodelo output: {_NOVOMODELO_DIR}" in text


def test_print_results_summary_explicit_newave_label_matches_default() -> None:
    """Explicitly passing ``reference_label="NEWAVE"`` matches the default text."""
    dataset = _build_dataset()

    verdict = build_compare_verdict(dataset)
    text_default = _capture(
        print_results_summary_from_dataset,
        dataset,
        _NW_DIR,
        _NOVOMODELO_DIR,
        verdict=verdict,
    )
    text_explicit = _capture(
        print_results_summary_from_dataset,
        dataset,
        _NW_DIR,
        _NOVOMODELO_DIR,
        verdict=verdict,
        reference_label="NEWAVE",
    )

    assert text_default == text_explicit


def test_print_results_summary_decomp_label_relabels_header() -> None:
    """``reference_label="DECOMP"`` relabels the header/labels; no "NEWAVE" remains."""
    dataset = _build_dataset()

    text = _capture(
        print_results_summary_from_dataset,
        dataset,
        _NW_DIR,
        _NOVOMODELO_DIR,
        verdict=build_compare_verdict(dataset),
        reference_label="DECOMP",
    )

    assert "Novomodelo vs DECOMP Results Comparison" in text
    assert f"DECOMP case:  {_NW_DIR}" in text
    assert f"Novomodelo output: {_NOVOMODELO_DIR}" in text
    assert "NEWAVE" not in text


def test_print_results_summary_decomp_label_keeps_newave_dir_param_name() -> None:
    """``newave_dir`` stays the parameter name; only its printed LABEL changes.

    Calling with the ``newave_dir=`` keyword must still work (the
    parameter was NOT renamed to ``reference_dir`` or similar) — only the
    printed prefix in front of the path derives from ``reference_label``.
    """
    dataset = _build_dataset()

    text = _capture(
        print_results_summary_from_dataset,
        dataset,
        newave_dir=_NW_DIR,
        novomodelo_output_dir=_NOVOMODELO_DIR,
        verdict=build_compare_verdict(dataset),
        reference_label="DECOMP",
    )

    assert f"DECOMP case:  {_NW_DIR}" in text


# ---------------------------------------------------------------------------
# console threading: --no-color / --quiet through the CLI
# ---------------------------------------------------------------------------


def _patch_compare_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the shared NEWAVE case/alignment loaders for the CLI.

    ``read_novomodelo_lines`` is left unpatched: its consumer
    (``build_entity_alignment``) is mocked above, and the empty ``tmp_path``
    novomodelo dir yields ``[]`` from the real reader anyway.
    """
    from tests.conftest import make_nw_files

    # ``.files`` must be a real ``NewaveFiles`` dataclass (not a further
    # MagicMock attribute) — ``hash_input_files`` reflects over it via
    # ``dataclasses.fields``, which raises on a non-dataclass. The paths
    # need not exist: a missing file degrades to a ``None`` hash/size.
    monkeypatch.setattr(
        "novomodelo_bridge.newave.case.NewaveCase.from_directory",
        classmethod(
            lambda cls, _dir: MagicMock(
                id_map=MagicMock(), files=make_nw_files(Path("nw"))
            )
        ),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.newave.alignment.build_entity_alignment",
        lambda *a, **k: MagicMock(),
    )


def _one_row_results_dataset() -> ComparisonDataset:
    """A one-row results dataset built through the shared assembly kernel."""
    results = [
        ResultComparison(
            entity_type="hydro",
            entity_name="ITAIPU",
            newave_code=10,
            novomodelo_id=0,
            stage=0,
            variable="generation_mw",
            newave_value=100.0,
            novomodelo_value=110.0,
            abs_diff=10.0,
            rel_diff=0.1,
        ),
    ]
    return build_results_dataset(results, PercentileData(), 1e-2)


def _invoke_compare_newave(
    argv_tail: list[str], monkeypatch: pytest.MonkeyPatch, novomodelo_dir: Path
) -> object:
    """Invoke ``compare newave`` through the real Typer app via ``CliRunner``."""
    from novomodelo_bridge.cli import app

    _patch_compare_context(monkeypatch)
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.newave.results.compare_results",
        lambda **_k: _one_row_results_dataset(),
    )
    argv = [
        "compare",
        "newave",
        str(novomodelo_dir.parent / "nw"),
        str(novomodelo_dir),
        *argv_tail,
    ]
    return CliRunner().invoke(app, argv)


def test_compare_newave_no_color_summary_has_no_ansi_escapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dumb_terminal: None
) -> None:
    """``--no-color`` threads through the results-summary renderer.

    Under ``CliRunner`` the captured stream is never a TTY, so Rich withholds
    ANSI regardless of ``--no-color`` — an ANSI-absence assertion alone is
    tautological and can never fail. Spy on the ``console`` actually handed to
    :func:`print_results_summary_from_dataset` instead, and assert it was
    built with ``no_color=True`` only when the flag is passed (``False``
    without it), so the test genuinely fails if the flag stops threading.
    """
    monkeypatch.delenv("NO_COLOR", raising=False)
    novomodelo_dir_flagged = tmp_path / "novomodelo_no_color"
    novomodelo_dir_flagged.mkdir()
    novomodelo_dir_plain = tmp_path / "novomodelo_plain"
    novomodelo_dir_plain.mkdir()

    captured: list[Console] = []
    original: Callable[..., None] = print_results_summary_from_dataset

    def _spy(*args: object, **kwargs: object) -> None:
        console = kwargs["console"]
        assert isinstance(console, Console)
        captured.append(console)
        original(*args, **kwargs)

    monkeypatch.setattr(
        "novomodelo_bridge.ui.compare_summary.print_results_summary_from_dataset", _spy
    )

    no_color_result = _invoke_compare_newave(
        ["--no-color"], monkeypatch, novomodelo_dir_flagged
    )
    plain_result = _invoke_compare_newave([], monkeypatch, novomodelo_dir_plain)

    assert no_color_result.exit_code == 0
    assert plain_result.exit_code == 0
    assert "Novomodelo vs NEWAVE Results Comparison" in no_color_result.stdout
    assert "\x1b[" not in no_color_result.stdout
    assert [c.no_color for c in captured] == [True, False]


def test_compare_newave_quiet_suppresses_summary_but_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--quiet`` suppresses the summary render; artifacts are still written."""
    novomodelo_dir = tmp_path / "novomodelo"
    novomodelo_dir.mkdir()

    result = _invoke_compare_newave(["--quiet"], monkeypatch, novomodelo_dir)

    assert result.exit_code == 0
    assert "Novomodelo vs NEWAVE Results Comparison" not in result.stdout
    assert "generation_mw" not in result.stdout
    artifacts_dir = novomodelo_dir / "comparison_artifacts"
    assert artifacts_dir.exists()
    assert any(artifacts_dir.iterdir())


class TestReportBuilderProductivityGateDecoupling:
    """report_builder decouples its single ``if prod_df.is_empty():
    ... else: ...`` Productivity-tab gate into two independent gates (one per
    frame) so DECOMP's realized-only shape can render. This guards the
    NEWAVE-shaped case -- both frames populated, which is what NEWAVE always
    has -- still renders every section in the exact original order/content,
    i.e. ``compare newave`` is unaffected by the decoupling."""

    @staticmethod
    def _both_frames_dataset() -> ComparisonDataset:
        from novomodelo_bridge.comparators.analyze import (
            _PRODUCTIVITY_DETAIL_SCHEMA,
            build_results_dataset,
        )
        from novomodelo_bridge.comparators.model import PercentileData

        results = [
            ResultComparison(
                entity_type="hydro",
                entity_name="ALPHA",
                newave_code=1,
                novomodelo_id=0,
                stage=0,
                variable="productivity_mw_per_m3s",
                newave_value=0.78,
                novomodelo_value=0.80,
                abs_diff=0.02,
                rel_diff=0.026,
            )
        ]
        prod_detail = pl.DataFrame(
            {
                "plant_name": ["ALPHA"],
                "newave_code": [1],
                "novomodelo_id": [0],
                "nw_altura_min": [0.69],
                "nw_altura_65": [0.81],
                "nw_altura_max": [0.85],
                "nw_equivalent": [0.7865],
                "nw_accumulated_earm": [5.35],
                "nw_specific_productivity": [0.009],
                "nw_tailwater_m": [672.0],
                "nw_losses_m": [0.8],
                "nw_vmin_hm3": [100.0],
                "nw_vmax_hm3": [500.0],
                "cb_point": [0.811],
                "cb_equivalent": [0.7860],
                "cb_accumulated": [5.349],
                "cb_specific_productivity": [0.009],
                "cb_tailwater_m": [672.0],
                "cb_losses_m": [0.8],
                "cb_vmin_hm3": [100.0],
                "cb_vmax_hm3": [500.0],
            },
            schema=_PRODUCTIVITY_DETAIL_SCHEMA,
        )
        pct = PercentileData(productivity_detail=prod_detail)
        return build_results_dataset(results, pct, 0.05)

    def test_both_populated_renders_all_three_sections_in_the_original_order(
        self,
    ) -> None:
        dataset = self._both_frames_dataset()

        html = build_comparison_report(dataset)
        productivity_tab = _extract_tab_content(html, "tab-productivity")

        static_idx = productivity_tab.index(
            "Static productivity — pmo vs novomodelo-bridge conversion"
        )
        realized_idx = productivity_tab.index("Realized productivity across stages")
        blocks_idx = productivity_tab.index("Productivity Building Blocks")
        assert static_idx < realized_idx < blocks_idx
        assert "No productivity data available." not in productivity_tab

    def test_only_detail_populated_renders_static_only_no_realized_section(
        self,
    ) -> None:
        """The mirror case: ``prod_df`` non-empty, ``per_stage_df`` empty --
        the realized section must NOT render. Proves the two gates are
        independent, not still coupled to one another."""
        from novomodelo_bridge.comparators.analyze import build_results_dataset
        from novomodelo_bridge.comparators.model import PercentileData

        detail_only = self._both_frames_dataset()
        pct = PercentileData(productivity_detail=detail_only.render.productivity_detail)
        dataset = build_results_dataset([], pct, 0.05)

        html = build_comparison_report(dataset)
        productivity_tab = _extract_tab_content(html, "tab-productivity")

        assert "Productivity Building Blocks" in productivity_tab
        assert "Realized productivity across stages" not in productivity_tab


class TestReportBuilderReeSectionByteIdentityGuard:
    """The REE section is additive and must leave
    ``compare newave`` untouched -- it renders only when ``entity_type ==
    "ree"`` rows exist, which a NEWAVE-shaped dataset never carries."""

    def test_newave_shaped_dataset_has_no_ree_section(self) -> None:
        from novomodelo_bridge.comparators.analyze import build_results_dataset
        from novomodelo_bridge.comparators.model import PercentileData

        results = [
            ResultComparison(
                entity_type="hydro",
                entity_name="CAMARGOS",
                newave_code=1,
                novomodelo_id=0,
                stage=0,
                variable="generation_mw",
                newave_value=100.0,
                novomodelo_value=98.0,
                abs_diff=2.0,
                rel_diff=0.02,
            ),
            ResultComparison(
                entity_type="bus",
                entity_name="SE",
                newave_code=1,
                novomodelo_id=0,
                stage=0,
                variable="deficit_mw",
                newave_value=0.0,
                novomodelo_value=0.0,
                abs_diff=0.0,
                rel_diff=None,
            ),
        ]
        dataset = build_results_dataset(results, PercentileData(), 0.05)
        assert dataset.tidy.filter(pl.col("entity_type") == "ree").is_empty()

        html = build_comparison_report(dataset)

        assert "REE Energy" not in html
