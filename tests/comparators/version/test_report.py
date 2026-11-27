"""Tests of ``comparators.version.report``.

The fragments a version comparison adds are checked on hand-built facts records,
with the shared tab builder replaced by one marker string per tab, so each
assertion sees only what this module wrote. The integration cases build pairs of
``build_solved_case`` trees, run them through ``build_version_dataset`` and render
the real shared tabs; a render is compared with another through ``_strip_chart_id``
because every chart gets a fresh random id.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple, cast

import pytest

from novomodelo_bridge.comparators.analyze import build_results_dataset
from novomodelo_bridge.comparators.dataset import ComparisonDataset
from novomodelo_bridge.comparators.html_report import COMPARISON_TABS
from novomodelo_bridge.comparators.model import PercentileData
from novomodelo_bridge.comparators.report_builder import build_comparison_tabs
from novomodelo_bridge.comparators.version import report
from novomodelo_bridge.comparators.version.convergence import UB_COMPARABILITY
from novomodelo_bridge.comparators.version.facts import (
    METADATA_KEY,
    VersionFacts,
    facts_of,
)
from novomodelo_bridge.comparators.version.report import (
    SECTION_TABS,
    build_version_report,
    version_report_tabs,
)
from novomodelo_bridge.comparators.version.requirements import REQUIREMENTS
from novomodelo_bridge.comparators.version.results import build_version_dataset
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.novomodelo.case_io import case_dir_for
from tests.golden_utils import _strip_chart_id
from tests.novomodelo_output_builders import (
    SolvedCaseSpec,
    build_solved_case,
    drop_column,
    drop_partition,
)

_TOLERANCE = 0.05
_SPEC = SolvedCaseSpec(load_stats=True)
_CANDIDATE_OFFSET = 100
_TAB_IDS = [tab_id for tab_id, _ in COMPARISON_TABS]
_OVERVIEW = "tab-overview"

_NOTICE_TITLE = '<div class="section-title">Results unavailable</div>'
_ONLY_IN_TITLE = "Entities found in one output only"
_GAP_PARAGRAPH = re.compile(r'<p style="color:#374151;margin:0 0 12px">(.*?)</p>')
_NOTICE_BLOCK = re.compile(r"Results unavailable</div>.*?</table>", re.DOTALL)
_ONLY_IN_BLOCK = re.compile(rf"{_ONLY_IN_TITLE}</div>.*?</table>", re.DOTALL)
_ROW = re.compile(r"<tr><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td></tr>")

_SOURCE_MODEL_WORDS = (
    r"\b(?:ZINF|ZSUP|COPER|CTERM|CUSTO_FUTURO|CMO|SIN|EARMF?|ENA|QAFLUH|NEWAVE|"
    r"DECOMP|STAND_IN|[Pp]olicy|[Ss]olvers?|either model)\b"
)
_BASELINE_LABEL = "Novomodelo 0.20.0"
_CANDIDATE_LABEL = "Novomodelo 0.21.0"
_RISING = "Novomodelo 0.20.0 1.50%, Novomodelo 0.21.0 2.25%, difference +0.75 pp"
_FALLING = "Novomodelo 0.20.0 2.25%, Novomodelo 0.21.0 1.50%, difference -0.75 pp"
_NO_GAP = "Novomodelo 0.20.0 —, Novomodelo 0.21.0 —, difference —"
_ONE_GAP = "Novomodelo 0.20.0 —, Novomodelo 0.21.0 2.25%, difference —"

_TABS_OF_SECTION = {
    "costs": {"tab-overview"},
    "bus": {"tab-system"},
    "bus-net-import": {"tab-system"},
    "energy-balance": {"tab-balance"},
    "line": {"tab-network"},
    "line-limits": {"tab-network"},
    "generic-constraints": {"tab-constraints"},
    "hydro": {"tab-hydro", "tab-hydro-detail"},
    "system-spillage": {"tab-hydro"},
    "hydro-bounds": {"tab-hydro-detail"},
    "thermal": {"tab-thermal", "tab-thermal-detail"},
    "production-function": {"tab-productivity"},
    "operating-points": {"tab-productivity"},
}

_GAP_CASES = [
    ("comparable", 1.5, 2.25, 0.75, _RISING),
    ("bounds-missing", None, None, None, _NO_GAP),
    ("kind-missing", 1.5, 2.25, 0.75, _RISING),
    ("kind-differs", 2.25, 1.5, -0.75, _FALLING),
    ("risk-averse", 1.5, 2.25, 0.75, _RISING),
    ("risk-unknown", 1.5, 2.25, 0.75, _RISING),
    ("bound-nonpositive", None, 2.25, None, _ONE_GAP),
]

_BASE = VersionFacts(
    baseline_output_dir="base/output",
    candidate_output_dir="cand/output",
    baseline_version="0.20.0",
    candidate_version="0.21.0",
    baseline_label="Novomodelo 0.20.0",
    candidate_label="Novomodelo 0.21.0",
    only_in_baseline={},
    only_in_candidate={},
    unavailable=[],
    upper_bound_comparable=True,
    upper_bound_comparability="comparable",
    baseline_upper_bound_kind="statistical",
    candidate_upper_bound_kind="statistical",
    baseline_gap_percent=1.5,
    candidate_gap_percent=2.25,
    gap_delta_pp=0.75,
)


def _facts(**changes: Any) -> VersionFacts:
    return replace(_BASE, **changes)


def _item(
    section: str, column: str | None = None, side: str = "both"
) -> dict[str, str | None]:
    return {"section": section, "column": column, "side": side}


def _dataset_with(facts: VersionFacts) -> ComparisonDataset:
    dataset = build_results_dataset([], PercentileData(), _TOLERANCE)
    dataset.metadata[METADATA_KEY] = facts.to_dict()
    return dataset


def _marker(tab_id: str) -> str:
    return f"<shared {tab_id}>"


def _tabs_of(facts: VersionFacts) -> dict[str, str]:
    return version_report_tabs(_dataset_with(facts))


def _notice_rows(tab: str) -> list[tuple[str, str, str]]:
    block = _NOTICE_BLOCK.search(tab)
    assert block is not None
    return _ROW.findall(block.group(0))


def _only_in_rows(tab: str) -> list[tuple[str, str, str]]:
    block = _ONLY_IN_BLOCK.search(tab)
    assert block is not None
    return _ROW.findall(block.group(0))


def _gap_text(tab: str) -> str:
    found = _GAP_PARAGRAPH.search(tab)
    assert found is not None
    return found.group(1)


@pytest.fixture
def shared_tabs(monkeypatch: pytest.MonkeyPatch) -> None:
    def stub(dataset: ComparisonDataset, reference_label: str) -> dict[str, str]:
        return {tab_id: _marker(tab_id) for tab_id in _TAB_IDS}

    monkeypatch.setattr(report, "build_comparison_tabs", stub)


class TestSectionTabs:
    def test_the_table_names_every_section_of_the_comparison_and_no_other(
        self,
    ) -> None:
        assert set(SECTION_TABS) == set(REQUIREMENTS)

    def test_every_tab_of_the_table_is_a_tab_of_the_report(self) -> None:
        shown = {tab_id for tabs in SECTION_TABS.values() for tab_id in tabs}

        assert shown <= set(_TAB_IDS)
        assert all(SECTION_TABS.values())

    def test_the_table_maps_each_section_to_the_tabs_that_show_it(self) -> None:
        assert {s: set(tabs) for s, tabs in SECTION_TABS.items()} == _TABS_OF_SECTION

    @pytest.mark.parametrize("section", sorted(_TABS_OF_SECTION))
    @pytest.mark.usefixtures("shared_tabs")
    def test_a_section_notice_lands_on_the_tabs_of_its_section_only(
        self, section: str
    ) -> None:
        tabs = _tabs_of(_facts(unavailable=[_item(section)]))

        noticed = {tab_id for tab_id, html in tabs.items() if _NOTICE_TITLE in html}
        assert noticed == _TABS_OF_SECTION[section]


@pytest.mark.usefixtures("shared_tabs")
class TestNotice:
    def test_a_whole_section_is_one_row_with_a_dash_in_the_column_cell(self) -> None:
        tabs = _tabs_of(_facts(unavailable=[_item("generic-constraints")]))

        assert tabs["tab-constraints"].startswith(_NOTICE_TITLE)
        assert _notice_rows(tabs["tab-constraints"]) == [
            ("generic-constraints", "—", "both")
        ]
        assert tabs["tab-constraints"].endswith(f"\n{_marker('tab-constraints')}")

    def test_a_column_item_shows_the_column_name(self) -> None:
        tabs = _tabs_of(
            _facts(unavailable=[_item("thermal", "generation_mw", "candidate")])
        )

        for tab_id in ("tab-thermal", "tab-thermal-detail"):
            assert _notice_rows(tabs[tab_id]) == [
                ("thermal", "generation_mw", _CANDIDATE_LABEL)
            ]

    def test_the_rows_follow_the_order_of_the_record(self) -> None:
        items = [
            _item("hydro", None, "baseline"),
            _item("hydro", "inflow_m3s", "candidate"),
            _item("system-spillage", None, "both"),
        ]

        tabs = _tabs_of(_facts(unavailable=items))

        assert _notice_rows(tabs["tab-hydro"]) == [
            ("hydro", "—", _BASELINE_LABEL),
            ("hydro", "inflow_m3s", _CANDIDATE_LABEL),
            ("system-spillage", "—", "both"),
        ]
        assert _notice_rows(tabs["tab-hydro-detail"]) == [
            ("hydro", "—", _BASELINE_LABEL),
            ("hydro", "inflow_m3s", _CANDIDATE_LABEL),
        ]

    def test_without_an_item_every_tab_but_the_overview_is_the_shared_one(
        self,
    ) -> None:
        tabs = _tabs_of(_facts())

        assert all("Results unavailable" not in html for html in tabs.values())
        for tab_id in set(_TAB_IDS) - {_OVERVIEW}:
            assert tabs[tab_id] == _marker(tab_id)

    def test_the_overview_notice_comes_before_the_gap_and_the_shared_content(
        self,
    ) -> None:
        tab = _tabs_of(_facts(unavailable=[_item("costs", None, "baseline")]))[
            _OVERVIEW
        ]

        assert tab.startswith(_NOTICE_TITLE)
        assert (
            tab.index(_NOTICE_TITLE)
            < tab.index("Optimality gap")
            < tab.index(_marker(_OVERVIEW))
        )

    def test_a_section_outside_the_table_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            _tabs_of(_facts(unavailable=[_item("not-a-section")]))

    def test_a_dataset_without_a_facts_record_raises_type_error(self) -> None:
        dataset = build_results_dataset([], PercentileData(), _TOLERANCE)

        with pytest.raises(TypeError):
            version_report_tabs(dataset)


@pytest.mark.usefixtures("shared_tabs")
class TestGapLine:
    def test_both_gaps_and_their_difference_open_the_overview(self) -> None:
        tab = _tabs_of(_facts())[_OVERVIEW]

        assert tab == "\n".join(
            (
                '<p style="color:#374151;margin:0 0 12px">Optimality gap: '
                "Novomodelo 0.20.0 1.50%, Novomodelo 0.21.0 2.25%, "
                "difference +0.75 pp.</p>",
                _marker(_OVERVIEW),
            )
        )

    def test_a_negative_difference_keeps_its_sign(self) -> None:
        text = _gap_text(_tabs_of(_facts(gap_delta_pp=-0.5))[_OVERVIEW])

        assert text.endswith("difference -0.50 pp.")

    def test_the_cases_cover_every_comparability_value(self) -> None:
        assert {case[0] for case in _GAP_CASES} == set(UB_COMPARABILITY)

    @pytest.mark.parametrize(
        ("comparability", "baseline", "candidate", "delta", "figures"), _GAP_CASES
    )
    def test_the_line_prints_each_figure_and_why_a_gap_is_not_judged(
        self,
        comparability: str,
        baseline: float | None,
        candidate: float | None,
        delta: float | None,
        figures: str,
    ) -> None:
        facts = _facts(
            baseline_gap_percent=baseline,
            candidate_gap_percent=candidate,
            gap_delta_pp=delta,
            upper_bound_comparable=comparability == "comparable",
            upper_bound_comparability=comparability,
        )

        text = _gap_text(_tabs_of(facts)[_OVERVIEW])

        verdict = (
            "" if comparability == "comparable" else f" Not judged ({comparability})."
        )
        assert text == f"Optimality gap: {figures}.{verdict}"

    def test_only_the_overview_holds_the_gap(self) -> None:
        tabs = _tabs_of(_facts())

        assert [t for t, html in tabs.items() if "Optimality gap" in html] == [
            _OVERVIEW
        ]


@pytest.mark.usefixtures("shared_tabs")
class TestOnlyInTable:
    def test_rows_are_sorted_by_entity_type_then_name_then_side(self) -> None:
        facts = _facts(
            only_in_baseline={"thermal": ["Z"], "hydro": ["B", "A"], "bus": []},
            only_in_candidate={"thermal": [], "hydro": ["C"], "line": ["L"]},
        )

        tab = _tabs_of(facts)[_OVERVIEW]

        assert _only_in_rows(tab) == [
            ("hydro", "A", _BASELINE_LABEL),
            ("hydro", "B", _BASELINE_LABEL),
            ("hydro", "C", _CANDIDATE_LABEL),
            ("line", "L", _CANDIDATE_LABEL),
            ("thermal", "Z", _BASELINE_LABEL),
        ]

    def test_the_block_follows_the_shared_content(self) -> None:
        tab = _tabs_of(_facts(only_in_candidate={"hydro": ["A"]}))[_OVERVIEW]

        assert tab.index(_marker(_OVERVIEW)) < tab.index(_ONLY_IN_TITLE)
        assert tab.count(_ONLY_IN_TITLE) == 1

    def test_the_table_has_the_columns_of_the_finding(self) -> None:
        tab = _tabs_of(_facts(only_in_candidate={"hydro": ["A"]}))[_OVERVIEW]

        assert (
            "<thead><tr><th>Entity type</th><th>Name</th><th>Only in</th></tr></thead>"
            in tab
        )

    def test_no_name_on_either_side_appends_nothing(self) -> None:
        facts = _facts(only_in_baseline={"hydro": []}, only_in_candidate={"hydro": []})

        tab = _tabs_of(facts)[_OVERVIEW]

        assert _ONLY_IN_TITLE not in tab
        assert tab.endswith(_marker(_OVERVIEW))

    def test_only_the_overview_holds_the_table(self) -> None:
        tabs = _tabs_of(_facts(only_in_baseline={"hydro": ["A"]}))

        assert [t for t, html in tabs.items() if _ONLY_IN_TITLE in html] == [_OVERVIEW]


@pytest.mark.usefixtures("shared_tabs")
class TestEscaping:
    def test_an_entity_name_is_escaped(self) -> None:
        tab = _tabs_of(_facts(only_in_baseline={"hydro": ["<b>X&Y</b>"]}))[_OVERVIEW]

        assert "&lt;b&gt;X&amp;Y&lt;/b&gt;" in tab
        assert "<b>X&Y</b>" not in tab

    def test_a_column_name_is_escaped(self) -> None:
        facts = _facts(unavailable=[_item("thermal", "<i>&</i>", "candidate")])

        tab = _tabs_of(facts)["tab-thermal"]

        assert "&lt;i&gt;&amp;&lt;/i&gt;" in tab
        assert "<i>&</i>" not in tab

    def test_both_labels_are_escaped_in_the_gap_line(self) -> None:
        facts = _facts(baseline_label="<u>&", candidate_label="<s>&")

        text = _gap_text(_tabs_of(facts)[_OVERVIEW])

        assert text.startswith(
            "Optimality gap: &lt;u&gt;&amp; 1.50%, &lt;s&gt;&amp; 2.25%,"
        )

    def test_a_comparability_value_is_escaped(self) -> None:
        facts = _facts(upper_bound_comparable=False, upper_bound_comparability="<x>")

        text = _gap_text(_tabs_of(facts)[_OVERVIEW])

        assert text.endswith(" Not judged (&lt;x&gt;).")

    def test_the_title_is_escaped_by_the_document(self) -> None:
        facts = _facts(baseline_label="<u>&", candidate_label="<s>&")

        document = build_version_report(_dataset_with(facts))

        assert "<title>&lt;s&gt;&amp; vs &lt;u&gt;&amp; Results Comparison</title>" in (
            document
        )


@pytest.mark.usefixtures("shared_tabs")
class TestPurity:
    def test_the_tabs_come_back_keyed_and_ordered_as_the_shared_builder_gives_them(
        self,
    ) -> None:
        tabs = _tabs_of(_facts(unavailable=[_item("costs")]))

        assert list(tabs) == _TAB_IDS

    def test_no_diagnostic_is_emitted(self) -> None:
        dataset = _dataset_with(
            _facts(
                only_in_baseline={"hydro": ["A"]},
                unavailable=[_item("thermal", None, "candidate")],
            )
        )

        with dx.collect() as sink:
            version_report_tabs(dataset)
            build_version_report(dataset)

        assert sink == []

    def test_the_facts_record_and_the_metadata_are_left_untouched(self) -> None:
        dataset = _dataset_with(
            _facts(
                only_in_baseline={"hydro": ["A"]},
                unavailable=[_item("thermal", "generation_mw", "candidate")],
            )
        )
        before = copy.deepcopy(dataset.metadata)

        version_report_tabs(dataset)
        build_version_report(dataset)

        assert dataset.metadata == before


class _Case(NamedTuple):
    dataset: ComparisonDataset
    tabs: dict[str, str]
    shared: dict[str, str]
    document: str
    root: Path


def _build(root: Path, name: str, **changes: Any) -> Path:
    return build_solved_case(root / name, replace(_SPEC, **changes))


def _candidate(root: Path, **changes: Any) -> Path:
    return _build(root, "candidate", id_offset=_CANDIDATE_OFFSET, **changes)


def _shared_tabs(dataset: ComparisonDataset) -> dict[str, str]:
    """The shared tabs with both sides named by their labels, as the report
    draws them."""
    facts = facts_of(dataset)
    return {
        tab_id: report._novomodelo_names(content)
        .replace("Novomodelo", facts.candidate_label)
        .replace(report._BASELINE_STAND_IN, facts.baseline_label)
        for tab_id, content in build_comparison_tabs(
            dataset, report._BASELINE_STAND_IN
        ).items()
    }


def _render(root: Path, baseline: Path, candidate: Path) -> _Case:
    with dx.collect():
        dataset = build_version_dataset(baseline, candidate, tolerance=_TOLERANCE)
        return _Case(
            dataset,
            version_report_tabs(dataset),
            _shared_tabs(dataset),
            build_version_report(dataset),
            root,
        )


def _pair(
    root: Path,
    baseline: Callable[[Path], Path],
    candidate: Callable[[Path], Path],
) -> _Case:
    return _render(root, baseline(root), candidate(root))


def _starts_with_the_notice(tab: str) -> bool:
    return tab.startswith(_NOTICE_TITLE)


@pytest.fixture(scope="module")
def equal_pair(tmp_path_factory: pytest.TempPathFactory) -> _Case:
    return _pair(
        tmp_path_factory.mktemp("equal"),
        lambda root: _build(root, "baseline", software_version="0.20.0"),
        _candidate,
    )


@pytest.fixture(scope="module")
def one_sided_pair(tmp_path_factory: pytest.TempPathFactory) -> _Case:
    return _pair(
        tmp_path_factory.mktemp("one_sided"),
        lambda root: _build(root, "baseline", hydros=("FURNAS", "ITAIPU", "TUCURUI")),
        lambda root: _candidate(root, thermals=("GAS_A", "GAS_B")),
    )


@pytest.fixture(scope="module")
def thermal_simulation_dropped(tmp_path_factory: pytest.TempPathFactory) -> _Case:
    def candidate(root: Path) -> Path:
        output = _candidate(root)
        drop_partition(output, "simulation/thermals")
        return output

    return _pair(
        tmp_path_factory.mktemp("no_thermal_simulation"),
        lambda root: _build(root, "baseline"),
        candidate,
    )


@pytest.fixture(scope="module")
def malformed_thermals(tmp_path_factory: pytest.TempPathFactory) -> _Case:
    def candidate(root: Path) -> Path:
        output = _candidate(root)
        (case_dir_for(output) / "system" / "thermals.json").write_text(
            '{"thermals": [{"id": 1}]}', encoding="utf-8"
        )
        return output

    return _pair(
        tmp_path_factory.mktemp("malformed_thermals"),
        lambda root: _build(root, "baseline"),
        candidate,
    )


@pytest.fixture(scope="module")
def same_output_without_exchanges(
    tmp_path_factory: pytest.TempPathFactory,
) -> _Case:
    root = tmp_path_factory.mktemp("same_output")
    output = _build(root, "case")
    drop_partition(output, "simulation/exchanges")
    return _render(root, output, output)


@pytest.fixture(scope="module")
def candidate_without_a_column(tmp_path_factory: pytest.TempPathFactory) -> _Case:
    def candidate(root: Path) -> Path:
        output = _candidate(root)
        drop_column(output, "simulation/thermals", "generation_mw")
        return output

    return _pair(
        tmp_path_factory.mktemp("no_column"),
        lambda root: _build(root, "baseline"),
        candidate,
    )


_FIXTURES = [
    "equal_pair",
    "one_sided_pair",
    "thermal_simulation_dropped",
    "malformed_thermals",
    "same_output_without_exchanges",
    "candidate_without_a_column",
]


class TestEqualPair:
    def test_the_document_is_titled_with_both_labels(self, equal_pair: _Case) -> None:
        assert (
            "<title>Novomodelo 0.21.0 vs Novomodelo 0.20.0 Results Comparison</title>"
        ) in equal_pair.document

    def test_each_side_is_named_by_its_label_in_the_charts(
        self, equal_pair: _Case
    ) -> None:
        document = equal_pair.document

        names = set(re.findall(r'"name":\s*"([^"]*)"', document))

        assert {"Novomodelo 0.21.0 Mean", "Novomodelo 0.20.0"} <= names
        assert "Novomodelo Mean" not in names
        assert "Novomodelo 0.21.0 0.20.0" not in document

    def test_the_tabs_are_the_eleven_ids_of_the_report_in_order(
        self, equal_pair: _Case
    ) -> None:
        assert list(equal_pair.tabs) == _TAB_IDS

    def test_a_tab_without_an_item_is_the_shared_tab_for_the_baseline_label(
        self, equal_pair: _Case
    ) -> None:
        assert _strip_chart_id(equal_pair.tabs["tab-system"]) == _strip_chart_id(
            equal_pair.shared["tab-system"]
        )

    def test_the_overview_opens_with_both_gaps_and_judges_them(
        self, equal_pair: _Case
    ) -> None:
        tab = equal_pair.tabs[_OVERVIEW]

        assert tab.startswith("<p")
        assert _gap_text(tab) == (
            "Optimality gap: Novomodelo 0.20.0 28.57%, Novomodelo 0.21.0 28.57%, "
            "difference +0.00 pp."
        )
        assert "Not judged" not in _gap_text(tab)

    def test_no_entity_is_found_in_one_output_only(self, equal_pair: _Case) -> None:
        assert _ONLY_IN_TITLE not in equal_pair.tabs[_OVERVIEW]

    @pytest.mark.parametrize(
        ("tab_id", "section"),
        [
            ("tab-constraints", "generic-constraints"),
            ("tab-hydro-detail", "hydro-bounds"),
            ("tab-network", "line-limits"),
        ],
    )
    def test_a_section_the_outputs_lack_opens_its_tab_with_a_notice(
        self, equal_pair: _Case, tab_id: str, section: str
    ) -> None:
        tab = equal_pair.tabs[tab_id]

        assert _starts_with_the_notice(tab)
        assert _notice_rows(tab) == [(section, "—", "both")]

    def test_a_tab_whose_sections_are_all_available_holds_no_notice(
        self, equal_pair: _Case
    ) -> None:
        assert "Results unavailable" not in equal_pair.tabs["tab-thermal"]

    def test_the_document_is_a_whole_page_of_eleven_tabs(
        self, equal_pair: _Case
    ) -> None:
        assert equal_pair.document.startswith("<!DOCTYPE html>")
        assert equal_pair.document.count('<section id="tab-') == 11


class TestGapOfTheOverview:
    def test_upper_bounds_of_different_kinds_are_shown_and_not_judged(
        self, tmp_path: Path
    ) -> None:
        case = _render(
            tmp_path,
            _build(tmp_path, "baseline"),
            _candidate(tmp_path, upper_bound_kind="exact"),
        )

        text = _gap_text(case.tabs[_OVERVIEW])

        assert text.endswith(" Not judged (kind-differs).")

    def test_a_baseline_without_bounds_shows_dashes_and_is_not_judged(
        self, tmp_path: Path
    ) -> None:
        baseline = _build(tmp_path, "baseline", software_version="0.20.0")
        drop_partition(baseline, "simulation")
        drop_partition(baseline, "training/convergence.parquet")

        case = _render(tmp_path, baseline, _candidate(tmp_path))

        text = _gap_text(case.tabs[_OVERVIEW])
        assert "Novomodelo 0.20.0 —" in text
        assert "difference —" in text
        assert text.endswith(" Not judged (bounds-missing).")


class TestOneSidedPair:
    def test_the_table_closes_the_overview_after_the_shared_content(
        self, one_sided_pair: _Case
    ) -> None:
        tab = one_sided_pair.tabs[_OVERVIEW]

        assert tab.count(_ONLY_IN_TITLE) == 1
        before = tab.partition(f'<div class="section-title">{_ONLY_IN_TITLE}</div>')[0]
        assert (
            _strip_chart_id(before)
            .rstrip("\n")
            .endswith(_strip_chart_id(one_sided_pair.shared[_OVERVIEW]))
        )

    def test_the_rows_name_the_entity_and_the_output_that_holds_it(
        self, one_sided_pair: _Case
    ) -> None:
        assert _only_in_rows(one_sided_pair.tabs[_OVERVIEW]) == [
            ("hydro", "TUCURUI", "baseline"),
            ("thermal", "GAS_B", "candidate"),
        ]

    def test_a_name_in_the_record_reaches_the_page_escaped(
        self, one_sided_pair: _Case
    ) -> None:
        dataset = one_sided_pair.dataset
        stored = facts_of(dataset).to_dict()
        stored["only_in_baseline"]["hydro"] = ["<b>X&Y</b>"]
        patched = replace(dataset, metadata={**dataset.metadata, METADATA_KEY: stored})

        with dx.collect():
            tab = version_report_tabs(patched)[_OVERVIEW]

        assert "&lt;b&gt;X&amp;Y&lt;/b&gt;" in tab
        assert "<b>X&Y</b>" not in tab


class TestUnavailableSections:
    @pytest.mark.parametrize("tab_id", ["tab-thermal", "tab-thermal-detail"])
    def test_a_missing_thermal_simulation_opens_the_thermal_tabs_with_a_notice(
        self, thermal_simulation_dropped: _Case, tab_id: str
    ) -> None:
        tab = thermal_simulation_dropped.tabs[tab_id]

        assert _starts_with_the_notice(tab)
        assert _notice_rows(tab) == [("thermal", "—", "candidate")]

    @pytest.mark.parametrize(
        ("tab_id", "section"),
        [
            ("tab-thermal", "thermal"),
            ("tab-thermal-detail", "thermal"),
            ("tab-balance", "energy-balance"),
        ],
    )
    def test_an_unusable_thermal_list_opens_every_tab_that_needs_it_with_a_notice(
        self, malformed_thermals: _Case, tab_id: str, section: str
    ) -> None:
        tab = malformed_thermals.tabs[tab_id]

        assert _starts_with_the_notice(tab)
        assert _notice_rows(tab) == [(section, "—", "candidate")]

    @pytest.mark.parametrize(
        ("tab_id", "section"),
        [("tab-network", "line"), ("tab-system", "bus-net-import")],
    )
    def test_an_item_missing_from_the_one_directory_is_a_row_for_both_sides(
        self, same_output_without_exchanges: _Case, tab_id: str, section: str
    ) -> None:
        tab = same_output_without_exchanges.tabs[tab_id]

        assert (section, "—", "both") in _notice_rows(tab)

    @pytest.mark.parametrize("tab_id", ["tab-thermal", "tab-thermal-detail"])
    def test_a_dropped_column_is_a_row_naming_the_column_on_its_tabs(
        self, candidate_without_a_column: _Case, tab_id: str
    ) -> None:
        tab = candidate_without_a_column.tabs[tab_id]

        assert _notice_rows(tab) == [("thermal", "generation_mw", "candidate")]


@pytest.mark.parametrize("name", _FIXTURES)
class TestEveryFixture:
    def test_a_tab_the_record_does_not_touch_is_the_shared_tab(
        self, name: str, request: pytest.FixtureRequest
    ) -> None:
        case: _Case = request.getfixturevalue(name)
        facts = facts_of(case.dataset)
        touched = {
            tab_id
            for item in facts.unavailable
            for tab_id in SECTION_TABS[cast("str", item["section"])]
        } | {_OVERVIEW}

        for tab_id in set(_TAB_IDS) - touched:
            assert _strip_chart_id(case.tabs[tab_id]) == _strip_chart_id(
                case.shared[tab_id]
            )

    def test_every_tab_holds_the_shared_content_whole(
        self, name: str, request: pytest.FixtureRequest
    ) -> None:
        case: _Case = request.getfixturevalue(name)

        for tab_id in _TAB_IDS:
            tab = _strip_chart_id(case.tabs[tab_id])
            shared = _strip_chart_id(case.shared[tab_id])
            assert shared in tab
            if tab_id != _OVERVIEW:
                assert tab.endswith(shared)

    def test_the_document_names_nothing_the_way_a_source_model_does(
        self, name: str, request: pytest.FixtureRequest
    ) -> None:
        case: _Case = request.getfixturevalue(name)
        visible = re.sub(r"//[^\n]*", "", case.document)

        found = re.findall(_SOURCE_MODEL_WORDS, visible)

        assert found == []

    def test_the_document_holds_neither_an_output_directory_nor_the_risk_key(
        self, name: str, request: pytest.FixtureRequest
    ) -> None:
        case: _Case = request.getfixturevalue(name)

        assert str(case.root) not in case.document
        assert "risk_measures_differ" not in case.document
