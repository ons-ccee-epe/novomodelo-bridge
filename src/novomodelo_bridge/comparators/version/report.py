"""The report of a version comparison: the shared tabs plus what a comparison adds.

The shared builder draws the eleven tabs and knows nothing of versions, so this
module leaves its tabs as they are and only prepends to them and appends to them,
reading every fact from the record ``facts_of`` returns: a notice at the top of
each tab that lost a section, both optimality gaps at the top of the Overview tab
and the entities found in one output only at its end. Everything here is
presentation; the strings it prints from the record are printed verbatim.

Both public functions run inside the caller's ``diagnostics.collect()`` sink,
because the shared charts emit Diagnostics; this module emits none itself.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import cast

from novomodelo_bridge.comparators.dataset import ComparisonDataset
from novomodelo_bridge.comparators.html_report import (
    build_comparison_html,
    chart_grid,
    section_title,
    wrap_chart,
)
from novomodelo_bridge.comparators.report_builder import build_comparison_tabs
from novomodelo_bridge.comparators.version.facts import VersionFacts, facts_of
from novomodelo_bridge.comparators.version.labels import SIDES
from novomodelo_bridge.comparators.version.requirements import (
    BUS,
    BUS_NET_IMPORT,
    COSTS,
    ENERGY_BALANCE,
    GENERIC_CONSTRAINTS,
    HYDRO,
    HYDRO_BOUNDS,
    LINE,
    LINE_LIMITS,
    OPERATING_POINTS,
    PRODUCTION_FUNCTION,
    SYSTEM_SPILLAGE,
    THERMAL,
)
from novomodelo_bridge.ui.html.document import escape_text

_BASELINE, _CANDIDATE = SIDES

_OVERVIEW = "tab-overview"

# What the shared charts call the side they compare against the reference.
_CHART_CANDIDATE = "Novomodelo"
# Stands in for the baseline label while the shared charts draw, so a baseline
# label holding ``Novomodelo`` is not replaced; word characters only.
_BASELINE_STAND_IN = "BASELINE_LABEL_STAND_IN"

_S = _BASELINE_STAND_IN
_MINUS = ("−", "\\u2212")
#: The source-model names the shared charts give the reference side, each with
#: the Novomodelo name the charts give the other side; longest first, applied
#: before the labels are put in.
_NOVOMODELO_NAMES: tuple[tuple[str, str], ...] = (
    *(
        pair
        for minus in _MINUS
        for pair in (
            (f"{_S} COPER {minus} CTERM", f"{_S} immediate {minus} thermal"),
            (f"COPER {minus} CTERM", f"immediate {minus} thermal"),
        )
    ),
    *(
        (f"\u2014 {_S} {name} vs", f"\u2014 {_S} vs")
        for name in ("COPER", "CUSTO_FUTURO", "CTERM")
    ),
    *(
        (f"\\u2014 {_S} {name} vs", f"\\u2014 {_S} vs")
        for name in ("COPER", "CUSTO_FUTURO", "CTERM")
    ),
    (f"{_S} COPER", f"{_S} immediate_cost"),
    (f"{_S} CUSTO_FUTURO", f"{_S} future_cost"),
    (f"{_S} CTERM", f"{_S} thermal (incl. anticipated)"),
    (f"{_S} ZSUP ITER", f"{_S} Upper"),
    (f"{_S} ZSUP", f"{_S} Upper (lowest so far)"),
    (f"{_S} ZINF", f"{_S} Lower"),
    (f"{_S} SIN", f"{_S} Mean"),
    (f'"{_S} LHS"', f'"{_S} LHS (mean)"'),
    (f"zero {_S} CMO", f"zero {_S} spot price"),
    (f"{_S} Policy Training", f"{_S} Training"),
    (f"{_S} policy ÷", f"{_S} training ÷"),
    (f"{_S} policy \\u00f7", f"{_S} training \\u00f7"),
    ("No COPER/CTERM data available.", "No immediate or thermal cost data available."),
    ("CMO by Bus", "Spot Price by Bus"),
    ("Total Inflow / QAFLUH", "Total Inflow"),
    (" (EARM / ENA)", ""),
    (" (ENA / EARM)", ""),
    (" (EARM)", ""),
    (" (ENA)", ""),
    (" (SIN)", ""),
    ("Both solvers fit", "Both runs fit"),
    ("in either model", "in either run"),
)


def _novomodelo_names(content: str) -> str:
    for source, novomodelo in _NOVOMODELO_NAMES:
        content = content.replace(source, novomodelo)
    return content


_DASH = "—"

#: The tab ids that show the notice of each section, by section name.
SECTION_TABS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        COSTS: (_OVERVIEW,),
        BUS: ("tab-system",),
        BUS_NET_IMPORT: ("tab-system",),
        ENERGY_BALANCE: ("tab-balance",),
        LINE: ("tab-network",),
        LINE_LIMITS: ("tab-network",),
        GENERIC_CONSTRAINTS: ("tab-constraints",),
        HYDRO: ("tab-hydro", "tab-hydro-detail"),
        SYSTEM_SPILLAGE: ("tab-hydro",),
        HYDRO_BOUNDS: ("tab-hydro-detail",),
        THERMAL: ("tab-thermal", "tab-thermal-detail"),
        PRODUCTION_FUNCTION: ("tab-productivity",),
        OPERATING_POINTS: ("tab-productivity",),
    }
)

_Unavailable = dict[str, str | None]


def _table(columns: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    head = "".join(f"<th>{column}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape_text(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    table = (
        '<table class="cost-breakdown-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )
    return chart_grid([wrap_chart(table)], single=True)


def _side_names(facts: VersionFacts) -> dict[str, str]:
    return {
        _BASELINE: facts.baseline_label,
        _CANDIDATE: facts.candidate_label,
        "both": "both",
    }


def _notice(items: Sequence[_Unavailable], facts: VersionFacts) -> str:
    if not items:
        return ""
    names = _side_names(facts)
    rows = [
        (
            item["section"],
            _DASH if item["column"] is None else item["column"],
            names[cast("str", item["side"])],
        )
        for item in items
    ]
    return "\n".join(
        (
            section_title("Results unavailable"),
            '<p style="color:#64748B;margin:-8px 0 12px">The comparison leaves '
            f"these out of this tab because {escape_text(facts.baseline_label)}, "
            f"{escape_text(facts.candidate_label)} or both lack them.</p>",
            _table(("Section", "Column", "Missing in"), rows),
        )
    )


def _figure(value: float | None, spec: str, unit: str) -> str:
    return _DASH if value is None else f"{value:{spec}}{unit}"


def _gap_line(facts: VersionFacts) -> str:
    verdict = (
        ""
        if facts.upper_bound_comparable
        else f" Not judged ({escape_text(facts.upper_bound_comparability)})."
    )
    return (
        '<p style="color:#374151;margin:0 0 12px">Optimality gap: '
        f"{escape_text(facts.baseline_label)} "
        f"{_figure(facts.baseline_gap_percent, '.2f', '%')}, "
        f"{escape_text(facts.candidate_label)} "
        f"{_figure(facts.candidate_gap_percent, '.2f', '%')}, "
        f"difference {_figure(facts.gap_delta_pp, '+.2f', ' pp')}.{verdict}</p>"
    )


def _only_in_table(facts: VersionFacts) -> str:
    side_names = _side_names(facts)
    rows = sorted(
        (entity_type, name, side_names[side])
        for side, only_in in (
            (_BASELINE, facts.only_in_baseline),
            (_CANDIDATE, facts.only_in_candidate),
        )
        for entity_type, names in only_in.items()
        for name in names
    )
    if not rows:
        return ""
    return "\n".join(
        (
            section_title("Entities found in one output only"),
            _table(("Entity type", "Name", "Only in"), rows),
        )
    )


def _unavailable_by_tab(facts: VersionFacts) -> dict[str, list[_Unavailable]]:
    by_tab: dict[str, list[_Unavailable]] = defaultdict(list)
    for item in facts.unavailable:
        for tab_id in SECTION_TABS[cast("str", item["section"])]:
            by_tab[tab_id].append(item)
    return by_tab


def version_report_tabs(dataset: ComparisonDataset) -> dict[str, str]:
    """The eleven tabs of the shared report for the baseline label, plus what a
    version comparison adds, keyed and ordered as ``build_comparison_tabs`` gives
    them.

    A tab holding an unavailable item of the facts record starts with a notice
    listing its items, in the record's order. The Overview tab then holds the
    optimality-gap paragraph, the shared content and, when either output holds an
    entity the other lacks, the table of those entities. Any other tab is the
    shared builder's string.

    Raises:
        TypeError: *dataset* holds no facts record.
        KeyError: an unavailable item names a section outside
            :data:`SECTION_TABS`.
    """
    facts = facts_of(dataset)
    unavailable = _unavailable_by_tab(facts)
    tabs = {
        tab_id: _novomodelo_names(content)
        .replace(_CHART_CANDIDATE, facts.candidate_label)
        .replace(_BASELINE_STAND_IN, facts.baseline_label)
        for tab_id, content in build_comparison_tabs(
            dataset, _BASELINE_STAND_IN
        ).items()
    }
    result = {}
    for tab_id, content in tabs.items():
        notice = _notice(unavailable.get(tab_id, []), facts)
        if tab_id == _OVERVIEW:
            parts = (notice, _gap_line(facts), content, _only_in_table(facts))
        else:
            parts = (notice, content)
        result[tab_id] = "\n".join(part for part in parts if part)
    return result


def build_version_report(dataset: ComparisonDataset) -> str:
    """The complete HTML document of a version comparison, titled
    ``<candidate label> vs <baseline label> Results Comparison``.

    Raises:
        TypeError: *dataset* holds no facts record.
        KeyError: an unavailable item names a section outside
            :data:`SECTION_TABS`.
    """
    facts = facts_of(dataset)
    return build_comparison_html(
        title=f"{facts.candidate_label} vs {facts.baseline_label} Results Comparison",
        tab_contents=version_report_tabs(dataset),
    )
