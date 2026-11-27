"""Golden pins of the version comparison report.

The pair is built by ``build_solved_case`` with ``fpha_planes=False``: the builder's
generation does not lie on its own production planes, so a pair with planes would
pin an off-plane model term as a property of the report. The golden therefore holds
the unavailable notices of the operating-point split and the fitted surface, and
their numeric tests live in ``test_operating_points.py`` and ``test_production.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest

from novomodelo_bridge.comparators.version.report import (
    build_version_report,
    version_report_tabs,
)
from novomodelo_bridge.comparators.version.results import build_version_dataset
from novomodelo_bridge.core import diagnostics as dx
from tests import golden_utils
from tests.golden_utils import assert_html_golden
from tests.novomodelo_output_builders import SolvedCaseSpec, build_solved_case

_TOLERANCE = 0.05
_BASELINE = SolvedCaseSpec(
    load_stats=True,
    software_version="0.20.0",
    hydros=("FURNAS", "ITAIPU", "TUCURUI"),
    fpha_planes=False,
)
_CANDIDATE = SolvedCaseSpec(
    load_stats=True,
    id_offset=100,
    thermals=("GAS_A", "GAS_B"),
    fpha_planes=False,
    scale=1.1,
)


class _Rendered(NamedTuple):
    overview: str
    document: str


def _render(tmp_path: Path, name: str) -> _Rendered:
    baseline = build_solved_case(tmp_path / name / "baseline", _BASELINE)
    candidate = build_solved_case(tmp_path / name / "candidate", _CANDIDATE)
    with dx.collect():
        dataset = build_version_dataset(baseline, candidate, tolerance=_TOLERANCE)
        texts = (
            version_report_tabs(dataset)["tab-overview"],
            build_version_report(dataset),
        )
    masked = [
        text.replace(str(baseline), "<BASELINE_OUTPUT>").replace(
            str(candidate), "<CANDIDATE_OUTPUT>"
        )
        for text in texts
    ]
    assert all(str(tmp_path) not in text for text in masked)
    return _Rendered(*masked)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> _Rendered:
    return _render(tmp_path_factory.mktemp("golden"), "pair")


def test_overview_tab_matches_golden(rendered: _Rendered) -> None:
    assert_html_golden(rendered.overview, "version_report_tab_overview.html")


def test_full_report_matches_golden(rendered: _Rendered) -> None:
    assert_html_golden(rendered.document, "version_comparison_report_full.html")


def test_two_builds_in_different_directories_render_equal_documents(
    tmp_path: Path,
) -> None:
    first = _render(tmp_path, "first")
    second = _render(tmp_path, "second")

    assert golden_utils._strip_chart_id(first.overview) == golden_utils._strip_chart_id(
        second.overview
    )
    assert golden_utils._strip_chart_id(first.document) == golden_utils._strip_chart_id(
        second.document
    )
