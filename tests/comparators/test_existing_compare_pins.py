"""Pins on what ``compare newave`` and ``compare decomp`` produce that no other test holds.

The goldens under ``tests/golden/`` pin the report tabs of the NEWAVE label only. These
tests pin the console summary text, the DECOMP-label report, the artifact file set and
the key order of ``comparison.json``, the cost tables for cost components the source
model has no parcel for, and the order of the agreement label tables. Each expected
value is a literal captured from a run, except the artifact texts, which are compared
with the expression that produces each. A pin is recaptured only by a deliberate change
to those commands, never to make another change pass.

Tier 1: the dataset is built from the in-memory fixtures of
``tests/dashboard/test_chart_helpers.py`` and nothing outside ``tmp_path`` is written.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from rich.console import Console

from novomodelo_bridge.comparators.analyze import build_results_dataset
from novomodelo_bridge.comparators.charts import _resolve_cost_categories
from novomodelo_bridge.comparators.charts.costs import (
    cost_breakdown_table,
    net_of_penalties_table,
    training_costs,
)
from novomodelo_bridge.comparators.dataset import ComparisonDataset
from novomodelo_bridge.comparators.export import write_artifacts
from novomodelo_bridge.comparators.manifest import ComparisonManifest
from novomodelo_bridge.comparators.report_builder import build_comparison_report
from novomodelo_bridge.comparators.verdict import build_compare_verdict
from novomodelo_bridge.core.agreement import ENTITY_GROUPS, VARIABLE_LABELS
from novomodelo_bridge.ui.compare_summary import print_results_summary_from_dataset
from tests.dashboard.test_chart_helpers import (
    _report_fixture_pct,
    _report_fixture_results,
)
from tests.golden_utils import _strip_chart_id

_NW_DIR = Path("/fake/nw")
_NOVOMODELO_DIR = Path("/fake/novomodelo")

_NW_COSTS = {"GERACAO TERMICA": 3.0e9, "DEFICIT": 1.0e8, "VIOLACAO VZMIN": 2.0e7}
_NOVOMODELO_COSTS = {
    "thermal_cost": 2.9e9,
    "deficit_cost": 1.2e8,
    "outflow_violation_below_cost": 1.0e7,
    "storage_violation_cost": 5.0e6,
    "inflow_penalty_cost": 4.0e6,
    "pumping_cost": 3.0e6,
}

_CONSOLE_SUMMARY_NEWAVE = (
    "✓ 4 of 4 variables close (gap ≤ 5%)\n"
    "\n"
    "Novomodelo vs NEWAVE Results Comparison\n"
    "========================================================================================\n"
    f"NEWAVE case:  {_NW_DIR}\n"
    f"Novomodelo output: {_NOVOMODELO_DIR}\n"
    "                                                         \n"
    "  Variable                Points    Gap   r   Agreement  \n"
    " ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ \n"
    "  Buses                                                  \n"
    "    Spot price (R$/MWh)        1   1.3%   —   close      \n"
    "  Hydro plants                                           \n"
    "    Storage (hm³)              1   0.2%   —   close      \n"
    "    Generation (MW)            1   0.4%   —   close      \n"
    "  Thermal plants                                         \n"
    "    Generation (MW)            1   0.7%   —   close      \n"
    "                                                         \n"
    "Gap = Σ|Novomodelo − NEWAVE| / Σ|NEWAVE| over every stage and entity: close up to 5%, far beyond 20%.\n"
    "\n"
    "                                                          \n"
    "  Bounds and cost (10⁹ R$)   NEWAVE   Novomodelo      Δ%  \n"
    " ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ \n"
    "  Final lower bound          50.000       50.000   +0.0%  \n"
    "  Simulated mean cost         3.120        3.042   -2.5%  \n"
    "                                                          \n"
    "\n"
    "Summary: 5 comparisons across 4 entity types (1 bus, 1 convergence, 2 hydro, 1 thermal)\n"
    "\n"
)

_CONSOLE_SUMMARY_DECOMP = (
    "✓ 4 of 4 variables close (gap ≤ 5%)\n"
    "\n"
    "Novomodelo vs DECOMP Results Comparison\n"
    "========================================================================================\n"
    f"DECOMP case:  {_NW_DIR}\n"
    f"Novomodelo output: {_NOVOMODELO_DIR}\n"
    "                                                         \n"
    "  Variable                Points    Gap   r   Agreement  \n"
    " ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ \n"
    "  Buses                                                  \n"
    "    Spot price (R$/MWh)        1   1.3%   —   close      \n"
    "  Hydro plants                                           \n"
    "    Storage (hm³)              1   0.2%   —   close      \n"
    "    Generation (MW)            1   0.4%   —   close      \n"
    "  Thermal plants                                         \n"
    "    Generation (MW)            1   0.7%   —   close      \n"
    "                                                         \n"
    "Gap = Σ|Novomodelo − DECOMP| / Σ|DECOMP| over every stage and entity: close up to 5%, far beyond 20%.\n"
    "\n"
    "                                                          \n"
    "  Bounds and cost (10⁹ R$)   DECOMP   Novomodelo      Δ%  \n"
    " ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ \n"
    "  Final lower bound          50.000       50.000   +0.0%  \n"
    "  Simulated mean cost         3.120        3.042   -2.5%  \n"
    "                                                          \n"
    "\n"
    "Summary: 5 comparisons across 4 entity types (1 bus, 1 convergence, 2 hydro, 1 thermal)\n"
    "\n"
)

_REPORT_DECOMP_SHA256 = (
    "71ec671f702ac1e762f4dfd726bae021ec89d190cefbdc1f9a967d26f2a12c56"
)
_REPORT_TITLE_NEWAVE = "Novomodelo vs NEWAVE Results Comparison"
_REPORT_TITLE_DECOMP = "Novomodelo vs DECOMP Results Comparison"

_COST_BREAKDOWN_TABLE = (
    '<table class="cost-breakdown-table"><caption>NPV by Component (10⁹ R$)</caption>'
    '<thead><tr><th class="cb-cat">Component</th><th class="cb-num">NEWAVE</th><th class="cb-num">Novomodelo</th><th class="cb-num">Δ</th><th class="cb-num">Δ%</th></tr>'
    "</thead>"
    '<tbody><tr><td class="cb-cat"><span class="cb-swatch" style="background:#eda100"></span>Thermal</td><td class="cb-num">3.000</td><td class="cb-num">2.900</td><td class="cb-num cb-diff-neg">-0.100</td><td class="cb-num cb-diff-neg">-3.3%</td></tr>'
    '<tr><td class="cb-cat"><span class="cb-swatch" style="background:#e34948"></span>Deficit</td><td class="cb-num">0.100</td><td class="cb-num">0.120</td><td class="cb-num cb-diff-pos">0.020</td><td class="cb-num cb-diff-pos">+20.0%</td></tr>'
    '<tr><td class="cb-cat"><span class="cb-swatch" style="background:#aaaaff"></span>Outflow Min</td><td class="cb-num">0.020</td><td class="cb-num">0.010</td><td class="cb-num">-0.010</td><td class="cb-num">-50.0%</td></tr>'
    '<tr><td class="cb-cat"><span class="cb-swatch" style="background:#614ed4"></span>Storage Bounds</td><td class="cb-num">0.000</td><td class="cb-num">0.005</td><td class="cb-num">0.005</td><td class="cb-num">—</td></tr>'
    '<tr><td class="cb-cat"><span class="cb-swatch" style="background:#18004e"></span>Inflow Penalty</td><td class="cb-num">0.000</td><td class="cb-num">0.004</td><td class="cb-num">0.004</td><td class="cb-num">—</td></tr>'
    '<tr><td class="cb-cat"><span class="cb-swatch" style="background:#2a78d6"></span>Pumping</td><td class="cb-num">0.000</td><td class="cb-num">0.003</td><td class="cb-num">0.003</td><td class="cb-num">—</td></tr>'
    '<tr class="cb-total"><td class="cb-cat">Total</td><td class="cb-num">3.120</td><td class="cb-num">3.042</td><td class="cb-num">-0.078</td><td class="cb-num">-2.5%</td></tr>'
    "</tbody></table>"
)

_NET_OF_PENALTIES_TABLE = (
    '<table class="cost-breakdown-table"><caption>Lower bound net of constraint-violation penalties (10⁹ R$) <span class="cb-caption-note">— penalties: Outflow Min, Storage Bounds, Inflow Penalty</span></caption>'
    '<thead><tr><th class="cb-cat">Bound</th><th class="cb-num">NEWAVE</th><th class="cb-num">Novomodelo</th><th class="cb-num">Δ</th><th class="cb-num">Δ%</th></tr>'
    "</thead>"
    '<tbody><tr><td class="cb-cat">Final lower bound</td><td class="cb-num">50.000</td><td class="cb-num">50.000</td><td class="cb-num">0.000</td><td class="cb-num">+0.0%</td></tr>'
    '<tr><td class="cb-cat">Expected violation penalties</td><td class="cb-num">0.020</td><td class="cb-num">0.019</td><td class="cb-num">-0.001</td><td class="cb-num">-5.0%</td></tr>'
    '<tr class="cb-total"><td class="cb-cat">Lower bound net of penalties</td><td class="cb-num">49.980</td><td class="cb-num">49.981</td><td class="cb-num">0.001</td><td class="cb-num">+0.0%</td></tr>'
    "</tbody></table>"
)

_COST_CATEGORY_ROWS = [
    ("Thermal", 3000000000.0, 2900000000.0),
    ("Deficit", 100000000.0, 120000000.0),
    ("Outflow Min", 20000000.0, 10000000.0),
    ("Storage Bounds", 0, 5000000.0),
    ("Inflow Penalty", 0, 4000000.0),
    ("Pumping", 0, 3000000.0),
]

# (formats, sorted directory listing, manifest.artifacts); metadata.json is a
# side effect of the parquet export and is not a manifest artifact.
_ARTIFACT_CASES: list[tuple[list[str], list[str], list[str]]] = [
    (
        ["parquet", "json", "csv"],
        [
            "comparison.csv",
            "comparison.json",
            "comparison.parquet",
            "metadata.json",
            "summary.csv",
            "summary.json",
            "summary.parquet",
            "top_divergences.json",
        ],
        [
            "comparison.csv",
            "comparison.json",
            "comparison.parquet",
            "summary.csv",
            "summary.json",
            "summary.parquet",
            "top_divergences.json",
        ],
    ),
    (
        ["json"],
        ["comparison.json", "summary.json", "top_divergences.json"],
        ["comparison.json", "summary.json", "top_divergences.json"],
    ),
    (
        ["csv"],
        ["comparison.csv", "comparison.json", "summary.csv"],
        ["comparison.csv", "comparison.json", "summary.csv"],
    ),
    ([], ["comparison.json"], ["comparison.json"]),
]
_ARTIFACT_CASE_IDS = ["parquet-json-csv", "json", "csv", "no-format"]

_COMPARISON_JSON_KEYS = [
    "command",
    "source_dir",
    "novomodelo_output_dir",
    "tolerance",
    "bridge_version",
    "git_sha",
    "timestamp",
    "novomodelo_version",
    "newave_version",
    "artifacts",
    "top_divergences",
    "input_files",
    "diagnostics_summary",
    "diagnostics",
]

_ENTITY_GROUP_PREFIX = [
    ("bus", "Buses"),
    ("line", "Lines"),
    ("ree", "Energy reservoirs (REE)"),
    ("system_spillage", "System spillage"),
    ("hydro", "Hydro plants"),
    ("thermal", "Thermal plants"),
]

_VARIABLE_LABEL_PREFIX = [
    ("spot_price", "Spot price (R$/MWh)"),
    ("deficit_mw", "Deficit (MW)"),
    ("net_import_mw", "Net import (MW)"),
    ("net_flow_mw", "Net flow (MW)"),
    ("tied_net_flow_mw", "Net flow on tied routes (MW)"),
    ("shared_plant_net_import_mw", "Net import of a bus sharing a plant (MW)"),
    ("shared_plant_net_flow_mw", "Net flow between shared-plant buses (MW)"),
    ("earm_final_mwmes", "Stored energy, end of stage (MWmês)"),
    ("ena_mwmes", "Natural inflow energy (MWmês)"),
    ("spill_energy_reservoir_mw", "Spilled energy, reservoirs (MW)"),
    ("spill_energy_rorov_mw", "Spilled energy, run-of-river (MW)"),
    ("spill_energy_total_mw", "Spilled energy, total (MW)"),
    ("storage_final_hm3", "Storage (hm³)"),
    ("generation_mw", "Generation (MW)"),
    ("turbined_m3s", "Turbined flow (m³/s)"),
    ("productivity_mw_per_m3s", "Productivity (MW per m³/s)"),
    ("spillage_m3s", "Spillage (m³/s)"),
    ("outflow_m3s", "Outflow (m³/s)"),
    ("inflow_m3s", "Incremental inflow (m³/s)"),
    ("total_inflow_m3s", "Total inflow (m³/s)"),
    ("evaporation_m3s", "Evaporation (m³/s)"),
    ("withdrawal_m3s", "Water withdrawal (m³/s)"),
    ("water_value_per_hm3", "Water value (R$/hm³)"),
    ("lower_bound", "Lower bound (R$)"),
    ("upper_bound", "Upper bound, mean (R$)"),
]


def _dataset() -> ComparisonDataset:
    return build_results_dataset(_report_fixture_results(), _report_fixture_pct(), 0.05)


def _convergence() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "iteration": [1, 2],
            "lower_bound": [40e9, 50e9],
            "upper_bound": [60e9, 55e9],
        }
    )


def _console_text(reference_label: str | None = None) -> str:
    """The summary text; ``None`` leaves ``reference_label`` at its default.

    ``legacy_windows=False``: on a console without VT support Rich draws the table
    rule with ``SIMPLE`` in place of ``SIMPLE_HEAVY``, so the text would differ by OS.
    """
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        width=120,
        color_system=None,
        force_terminal=False,
        legacy_windows=False,
    )
    dataset = _dataset()
    convergence = _convergence()
    label: dict[str, Any] = (
        {} if reference_label is None else {"reference_label": reference_label}
    )
    print_results_summary_from_dataset(
        dataset,
        _NW_DIR,
        _NOVOMODELO_DIR,
        verdict=build_compare_verdict(dataset),
        training=training_costs(convergence, convergence, _NW_COSTS, _NOVOMODELO_COSTS),
        console=console,
        verbose=True,
        **label,
    )
    return buffer.getvalue()


def _titles(html: str) -> list[str]:
    return re.findall(r"<title>(.*?)</title>", html)


def _write(
    tmp_path: Path, dataset: ComparisonDataset, formats: list[str]
) -> tuple[Path, ComparisonManifest]:
    out_dir = tmp_path / "a"
    manifest = write_artifacts(
        dataset,
        command="compare newave",
        source_dir=tmp_path / "nw",
        novomodelo_output_dir=tmp_path / "out",
        tolerance=0.05,
        out_dir=out_dir,
        formats=formats,
    )
    return out_dir, manifest


def test_console_summary_text_newave() -> None:
    assert _console_text() == _CONSOLE_SUMMARY_NEWAVE


def test_console_summary_text_decomp() -> None:
    assert _console_text("DECOMP") == _CONSOLE_SUMMARY_DECOMP


def test_report_html_decomp_label_sha256() -> None:
    html = build_comparison_report(_dataset(), reference_label="DECOMP")

    digest = hashlib.sha256(_strip_chart_id(html).encode("utf-8")).hexdigest()

    assert digest == _REPORT_DECOMP_SHA256


def test_report_title_element_for_both_labels() -> None:
    default = build_comparison_report(_dataset())
    decomp = build_comparison_report(_dataset(), reference_label="DECOMP")

    assert _titles(default) == [_REPORT_TITLE_NEWAVE]
    assert _titles(decomp) == [_REPORT_TITLE_DECOMP]


@pytest.mark.parametrize(
    ("formats", "listing", "artifacts"), _ARTIFACT_CASES, ids=_ARTIFACT_CASE_IDS
)
def test_artifact_listing_and_manifest_artifacts(
    tmp_path: Path, formats: list[str], listing: list[str], artifacts: list[str]
) -> None:
    out_dir, manifest = _write(tmp_path, _dataset(), formats)

    assert sorted(p.name for p in out_dir.iterdir()) == listing
    assert manifest.artifacts == artifacts


@pytest.mark.parametrize(
    ("formats", "listing", "artifacts"), _ARTIFACT_CASES, ids=_ARTIFACT_CASE_IDS
)
def test_comparison_json_key_order_and_artifacts_value(
    tmp_path: Path, formats: list[str], listing: list[str], artifacts: list[str]
) -> None:
    out_dir, _ = _write(tmp_path, _dataset(), formats)

    written = json.loads((out_dir / "comparison.json").read_text(encoding="utf-8"))

    assert list(written) == _COMPARISON_JSON_KEYS
    assert written["artifacts"] == artifacts


@pytest.mark.parametrize(
    ("formats", "listing", "artifacts"), _ARTIFACT_CASES, ids=_ARTIFACT_CASE_IDS
)
def test_artifact_texts_match_the_expression_that_produces_each(
    tmp_path: Path, formats: list[str], listing: list[str], artifacts: list[str]
) -> None:
    dataset = _dataset()
    produced = {
        "summary.json": json.dumps(dataset.summary.to_dicts(), indent=2),
        "top_divergences.json": json.dumps(
            dataset.metadata.get("top_divergences", []), indent=2
        ),
        "comparison.csv": dataset.tidy.write_csv(),
        "summary.csv": dataset.summary.write_csv(),
    }
    out_dir, _ = _write(tmp_path, dataset, formats)

    for name in sorted(produced.keys() & set(listing)):
        assert (out_dir / name).read_text(encoding="utf-8") == produced[name], name


def test_cost_tables_for_source_without_novomodelo_only_parcels() -> None:
    convergence = _convergence()

    assert (
        cost_breakdown_table(_NW_COSTS, _NOVOMODELO_COSTS, "NEWAVE", {})
        == _COST_BREAKDOWN_TABLE
    )
    assert (
        net_of_penalties_table(
            convergence, convergence, _NW_COSTS, _NOVOMODELO_COSTS, "NEWAVE"
        )
        == _NET_OF_PENALTIES_TABLE
    )
    assert [
        (label, source, novomodelo)
        for label, source, novomodelo, _ in _resolve_cost_categories(
            _NW_COSTS, _NOVOMODELO_COSTS
        )
    ] == _COST_CATEGORY_ROWS


def test_entity_group_labels_keep_their_order_prefix() -> None:
    assert list(ENTITY_GROUPS.items())[: len(_ENTITY_GROUP_PREFIX)] == (
        _ENTITY_GROUP_PREFIX
    )


def test_variable_labels_keep_their_order_prefix() -> None:
    assert list(VARIABLE_LABELS.items())[: len(_VARIABLE_LABEL_PREFIX)] == (
        _VARIABLE_LABEL_PREFIX
    )
