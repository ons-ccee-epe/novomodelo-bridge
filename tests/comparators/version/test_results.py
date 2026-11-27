"""Tests of ``comparators.version.results``.

Every case is a pair of ``build_solved_case`` trees, damaged where the case needs
it, and runs through the whole builder; nothing is mocked. Two identical runs are
the reference: every judged difference is exactly zero, so any drift a case adds
shows up as the one thing that changed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl
import pytest

from novomodelo_bridge.comparators.dataset import ComparisonDataset
from novomodelo_bridge.comparators.report_builder import build_comparison_tabs
from novomodelo_bridge.comparators.verdict import build_compare_verdict, compare_status
from novomodelo_bridge.comparators.version.facts import (
    METADATA_KEY,
    VersionFacts,
    facts_of,
)
from novomodelo_bridge.comparators.version.report import version_report_tabs
from novomodelo_bridge.comparators.version.requirements import (
    OPTIONAL_PARTITIONS,
    REQUIREMENTS,
)
from novomodelo_bridge.comparators.version.results import (
    CONVERGENCE_BOUNDS_KEY,
    build_version_dataset,
)
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.errors import NovomodeloOutputError, SourceFileError
from novomodelo_bridge.novomodelo.case_io import case_dir_for
from novomodelo_bridge.novomodelo.readers import NovomodeloReadError
from tests.conftest import write_generic_constraint_echo
from tests.novomodelo_output_builders import (
    SolvedCaseSpec,
    build_solved_case,
    corrupt_file,
    drop_column,
    drop_partition,
)

_TOLERANCE = 0.05
_SPEC = SolvedCaseSpec(load_stats=True)
_CANDIDATE_OFFSET = 100

_BOUNDS_SCHEMA = {
    "side": pl.Utf8,
    "iteration": pl.Int64,
    "lower_bound": pl.Float64,
    "upper_bound": pl.Float64,
    "upper_bound_std": pl.Float64,
    "upper_bound_kind": pl.Utf8,
}
_FINDING_CODES = (
    "same-output-compared",
    "entity-one-sided",
    "entity-name-duplicated",
    "stage-sets-differ",
    "section-unavailable",
    "generic-constraint-differs",
    "upper-bound-not-comparable",
    "iteration-counts-differ",
)
_FIXTURE_UNAVAILABLE = [
    {"section": "generic-constraints", "column": None, "side": "both"},
    {"section": "hydro-bounds", "column": None, "side": "both"},
    {"section": "line-limits", "column": None, "side": "both"},
]
_SCENARIO_SUMMARY = "simulation/scenario_summary.parquet"
_HYDROS = "simulation/hydros"
_THERMALS = "simulation/thermals"
_BUSES = "simulation/buses"
_EXCHANGES = "simulation/exchanges"
_COSTS = "simulation/costs"

# (section, kind, path under the output, column): one column of a required
# partition, of an optional one the section lists, and of a variable's source.
_DRIFT = [
    ("hydro", "required", _HYDROS, "hydro_id"),
    ("hydro", "optional", _SCENARIO_SUMMARY, "probability"),
    ("hydro", "variable", _HYDROS, "evaporation_m3s"),
    (
        "system-spillage",
        "required",
        _HYDROS,
        "equivalent_productivity_mw_per_m3s",
    ),
    ("system-spillage", "optional", _SCENARIO_SUMMARY, "probability"),
    ("thermal", "required", _THERMALS, "thermal_id"),
    ("thermal", "optional", _SCENARIO_SUMMARY, "probability"),
    ("thermal", "variable", _THERMALS, "generation_mw"),
    ("bus", "required", _BUSES, "bus_id"),
    ("bus", "optional", _EXCHANGES, "line_id"),
    ("bus", "variable", _BUSES, "spot_price"),
    ("bus-net-import", "required", _EXCHANGES, "net_flow_mw"),
    ("bus-net-import", "optional", _SCENARIO_SUMMARY, "probability"),
    ("line", "required", _EXCHANGES, "line_id"),
    ("line", "optional", _SCENARIO_SUMMARY, "probability"),
    ("line", "variable", _EXCHANGES, "net_flow_mw"),
    (
        "energy-balance",
        "required",
        "simulation/hydro_bus_generation",
        "generation_mwh",
    ),
    ("energy-balance", "optional", _THERMALS, "thermal_id"),
    ("energy-balance", "variable", _BUSES, "excess_mw"),
    ("costs", "required", _COSTS, "discount_factor"),
    ("costs", "optional", _SCENARIO_SUMMARY, "probability"),
    ("costs", "variable", _COSTS, "thermal_cost"),
    (
        "production-function",
        "required",
        "hydro_models/fpha_hyperplanes.parquet",
        "gamma_0",
    ),
    (
        "operating-points",
        "required",
        _HYDROS,
        "accumulated_productivity_mw_per_m3s",
    ),
]


class _Run(NamedTuple):
    dataset: ComparisonDataset
    sink: list[dx.Diagnostic]


def _compare(baseline: Path, candidate: Path) -> _Run:
    with dx.collect() as sink:
        dataset = build_version_dataset(baseline, candidate, tolerance=_TOLERANCE)
    return _Run(dataset, sink)


def _build(root: Path, name: str, spec: SolvedCaseSpec = _SPEC, **changes: Any) -> Path:
    return build_solved_case(root / name, replace(spec, **changes))


def _candidate(root: Path, **changes: Any) -> Path:
    return _build(root, "candidate", id_offset=_CANDIDATE_OFFSET, **changes)


def _codes(sink: list[dx.Diagnostic]) -> list[str]:
    return [diagnostic.code for diagnostic in sink]


def _diagnostic(sink: list[dx.Diagnostic], code: str) -> dx.Diagnostic:
    (found,) = [diagnostic for diagnostic in sink if diagnostic.code == code]
    return found


def _table_rows(sink: list[dx.Diagnostic], code: str) -> Sequence[Sequence[object]]:
    table = _diagnostic(sink, code).table
    assert table is not None
    return table.rows


def _unavailable_items(
    facts: VersionFacts,
) -> set[tuple[str | None, str | None, str | None]]:
    return {
        (item["section"], item["column"], item["side"]) for item in facts.unavailable
    }


def _entity_types(dataset: ComparisonDataset) -> set[str]:
    return {row.entity_type for row in dataset.render.results}


def _variables(dataset: ComparisonDataset, entity_type: str) -> set[str]:
    return {
        row.variable for row in dataset.render.results if row.entity_type == entity_type
    }


def _write_constraint(
    output: Path, *, constraint_id: int, hydro: int, coefficient: float = 1.0
) -> None:
    path = case_dir_for(output) / "constraints" / "generic_constraints.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "constraints": [
                    {
                        "id": constraint_id,
                        "name": "RE_1",
                        "description": "RE_1 of the case",
                        "expression": f"hydro_storage_final(id={hydro})",
                        "slack": {"enabled": True},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    write_generic_constraint_echo(
        output,
        [
            (
                constraint_id,
                "RE_1",
                stage,
                None,
                f"hydro_storage_final(id={hydro}, block=all)",
                coefficient,
                100.0,
                None,
            )
            for stage in (0, 1)
        ],
    )


def _write_case_bounds(
    output: Path, *, entity_id: int, withdrawal: tuple[float, float] = (1.0, 2.0)
) -> None:
    constraints = case_dir_for(output) / "constraints"
    constraints.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "hydro_id": [entity_id, entity_id],
            "stage_id": [0, 1],
            "min_storage_hm3": [10.0, 10.0],
            "water_withdrawal_m3s": list(withdrawal),
        },
        schema={
            "hydro_id": pl.Int32,
            "stage_id": pl.Int32,
            "min_storage_hm3": pl.Float64,
            "water_withdrawal_m3s": pl.Float64,
        },
    ).write_parquet(constraints / "hydro_bounds.parquet")
    pl.DataFrame(
        {
            "line_id": [entity_id, entity_id],
            "stage_id": [0, 1],
            "block_id": [None, None],
            "direct_mw": [1000.0, 1000.0],
            "reverse_mw": [900.0, 900.0],
        },
        schema={
            "line_id": pl.Int32,
            "stage_id": pl.Int32,
            "block_id": pl.Int32,
            "direct_mw": pl.Float64,
            "reverse_mw": pl.Float64,
        },
    ).write_parquet(constraints / "line_bounds.parquet")


@pytest.fixture(scope="module")
def baseline_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_solved_case(tmp_path_factory.mktemp("baseline"), _SPEC)


@pytest.fixture(scope="module")
def equal_runs(tmp_path_factory: pytest.TempPathFactory, baseline_output: Path) -> _Run:
    candidate = build_solved_case(
        tmp_path_factory.mktemp("candidate"),
        replace(_SPEC, id_offset=_CANDIDATE_OFFSET),
    )
    return _compare(baseline_output, candidate)


class TestEqualRuns:
    def test_the_dataset_validates_and_the_verdict_is_ok(
        self, equal_runs: _Run
    ) -> None:
        equal_runs.dataset.validate()

        assert compare_status(equal_runs.dataset) == "ok"

    def test_every_judged_row_differs_by_exactly_zero(self, equal_runs: _Run) -> None:
        rows = equal_runs.dataset.render.results

        assert rows
        assert all(row.abs_diff == 0.0 for row in rows)

    def test_the_rows_cover_every_entity_type_of_the_fixture(
        self, equal_runs: _Run
    ) -> None:
        assert _entity_types(equal_runs.dataset) >= {
            "hydro",
            "thermal",
            "bus",
            "line",
            "system_spillage",
            "final_bounds",
            "convergence",
        }

    def test_the_unavailable_items_are_the_three_sections_the_fixture_lacks(
        self, equal_runs: _Run
    ) -> None:
        assert facts_of(equal_runs.dataset).unavailable == _FIXTURE_UNAVAILABLE

    def test_rows_pair_entities_by_name_whatever_their_ids(
        self, equal_runs: _Run
    ) -> None:
        hydro = [
            row
            for row in equal_runs.dataset.render.results
            if row.entity_type == "hydro"
        ]

        assert {
            (row.entity_name, row.newave_code, row.novomodelo_id) for row in hydro
        } == {
            ("FURNAS", 0, 100),
            ("ITAIPU", 1, 101),
        }

    def test_no_finding_but_the_unavailable_sections_is_emitted(
        self, equal_runs: _Run
    ) -> None:
        findings = [code for code in _codes(equal_runs.sink) if code in _FINDING_CODES]

        assert findings == ["section-unavailable"]

    def test_every_reader_diagnostic_carries_the_side_of_its_output(
        self, equal_runs: _Run
    ) -> None:
        readers = [d for d in equal_runs.sink if d.code not in _FINDING_CODES]

        assert readers
        assert all(
            d.summary.startswith(("Baseline output: ", "Candidate output: "))
            for d in readers
        )

    def test_the_findings_come_after_every_reader_diagnostic(
        self, equal_runs: _Run
    ) -> None:
        codes = _codes(equal_runs.sink)

        first = next(i for i, code in enumerate(codes) if code in _FINDING_CODES)
        assert first > 0
        assert all(code in _FINDING_CODES for code in codes[first:])

    def test_the_facts_record_is_stored_as_a_dict_under_the_metadata_key(
        self, equal_runs: _Run
    ) -> None:
        dataset = equal_runs.dataset

        assert dataset.metadata[METADATA_KEY] == facts_of(dataset).to_dict()
        assert isinstance(dataset.metadata[METADATA_KEY], dict)

    def test_the_facts_record_shares_no_container_with_the_metadata(
        self, equal_runs: _Run
    ) -> None:
        dataset = equal_runs.dataset
        stored = json.dumps(dataset.metadata[METADATA_KEY])

        facts_of(dataset).unavailable.append(
            {"section": "x", "column": None, "side": "both"}
        )

        assert json.dumps(dataset.metadata[METADATA_KEY]) == stored

    def test_the_facts_hold_the_directories_as_typed_and_both_labels(
        self, equal_runs: _Run, baseline_output: Path
    ) -> None:
        facts = facts_of(equal_runs.dataset)

        assert facts.baseline_output_dir == str(baseline_output)
        assert facts.baseline_label == baseline_output.parent.name
        assert facts.candidate_label == Path(facts.candidate_output_dir).parent.name
        assert facts.upper_bound_comparability == "comparable"

    def test_the_bounds_frame_has_the_declared_schema_and_the_baseline_first(
        self, equal_runs: _Run
    ) -> None:
        bounds = equal_runs.dataset.metadata[CONVERGENCE_BOUNDS_KEY]

        assert isinstance(bounds, pl.DataFrame)
        assert bounds.schema == _BOUNDS_SCHEMA
        assert bounds["side"].to_list() == ["baseline"] * 3 + ["candidate"] * 3
        assert bounds["iteration"].to_list() == [1, 2, 3, 1, 2, 3]
        assert bounds["upper_bound_kind"].to_list() == ["statistical"] * 6

    def test_the_dataset_serializes(self, equal_runs: _Run, tmp_path: Path) -> None:
        equal_runs.dataset.to_dir(tmp_path / "out")

        stored = json.loads((tmp_path / "out" / "metadata.json").read_text("utf-8"))
        assert stored[CONVERGENCE_BOUNDS_KEY]["__frame__"] == "polars"
        assert VersionFacts.from_dict(stored[METADATA_KEY]) == facts_of(
            equal_runs.dataset
        )

    def test_the_baseline_fills_the_nw_slots_and_the_candidate_the_other_inputs(
        self, equal_runs: _Run
    ) -> None:
        render = equal_runs.dataset.render

        assert equal_runs.dataset.metadata["nw_hydro_names"] == {
            0: "FURNAS",
            1: "ITAIPU",
        }
        assert render.nw_bus_names == {0: "SUDESTE", 1: "SUL"}
        assert set(render.novomodelo_hydro_meta) == {100, 101}
        assert set(render.novomodelo_bus_meta) == {100, 101}
        assert render.novomodelo_bus_meta[100] == {"name": "SUDESTE", "has_load": True}
        assert render.novomodelo_hydro_meta[100]["bus_ids"] == [100]
        assert set(render.hydro["entity_id"]) == {100, 101}
        assert set(render.thermal["entity_id"]) == {100}
        assert set(render.line["entity_id"]) == {100}
        assert [line["id"] for line in render.line_meta] == [100]

    def test_the_stage_offset_is_zero_and_the_last_stage_is_the_largest_common_one(
        self, equal_runs: _Run
    ) -> None:
        render = equal_runs.dataset.render

        assert render.nw_offset == 0
        assert render.nw_max_stage == 1

    def test_the_baseline_encodes_into_its_slots_in_the_reference_vocabulary(
        self, equal_runs: _Run
    ) -> None:
        render = equal_runs.dataset.render

        assert not render.nw_convergence.is_empty()
        assert not render.nw_sin.is_empty()
        assert not render.nw_market.is_empty()
        assert "GERACAO TERMICA" in render.nw_costs
        assert not render.nw_hydro_slacks.is_empty()
        assert not render.nw_tim_iterations.is_empty()
        assert render.nw_tim_stages

    def test_the_candidate_costs_and_stage_costs_are_read(
        self, equal_runs: _Run
    ) -> None:
        render = equal_runs.dataset.render

        assert "thermal_cost" in render.novomodelo_costs
        assert render.novomodelo_stage_costs["immediate_cost"].null_count() == 0
        assert render.novomodelo_training_seconds == pytest.approx(0.9)

    def test_the_fitted_surface_and_operating_points_are_filled(
        self, equal_runs: _Run
    ) -> None:
        render = equal_runs.dataset.render

        assert not render.fpha_metrics.is_empty()
        assert not render.fpha_split.is_empty()
        assert not render.tie_share.is_empty()
        assert not render.hydro_masked.is_empty()


class TestReport:
    def test_the_report_renders_eleven_tabs(self, equal_runs: _Run) -> None:
        dataset = equal_runs.dataset
        label = facts_of(dataset).baseline_label

        tabs = build_comparison_tabs(dataset, label)

        assert len(tabs) == 11
        assert "Fitted production functions (FPHA)" in tabs["tab-productivity"]
        assert "Why the realized productivities differ" in tabs["tab-productivity"]
        assert label in tabs["tab-overview"]

    def test_a_dataset_with_every_optional_section_unavailable_renders(
        self, tmp_path: Path
    ) -> None:
        outputs = [
            _build(tmp_path, name, id_offset=offset, fpha_planes=False)
            for name, offset in (("baseline", 0), ("candidate", _CANDIDATE_OFFSET))
        ]
        for output in outputs:
            drop_partition(output, "simulation")

        run = _compare(*outputs)

        facts = facts_of(run.dataset)
        assert len(build_comparison_tabs(run.dataset, facts.baseline_label)) == 11
        assert _entity_types(run.dataset) == {"final_bounds", "convergence"}
        unavailable = {item["section"] for item in facts.unavailable}
        assert unavailable == set(REQUIREMENTS)

    def test_a_generic_constraint_on_both_sides_draws_a_panel(
        self, tmp_path: Path
    ) -> None:
        baseline = _build(tmp_path, "baseline")
        candidate = _candidate(tmp_path)
        _write_constraint(baseline, constraint_id=0, hydro=0)
        _write_constraint(candidate, constraint_id=5, hydro=_CANDIDATE_OFFSET)

        run = _compare(baseline, candidate)

        facts = facts_of(run.dataset)
        assert "generic-constraints" not in {
            item["section"] for item in facts.unavailable
        }
        assert [c["name"] for c in run.dataset.render.gc_constraints] == ["RE_1"]
        assert set(run.dataset.render.gc_lhs_newave["constraint_id"]) == {5}
        assert set(run.dataset.render.gc_lhs_novomodelo["constraint_id"]) == {5}
        assert "generic-constraint-differs" not in _codes(run.sink)
        tabs = build_comparison_tabs(run.dataset, facts.baseline_label)
        assert "RE_1" in tabs["tab-constraints"]

    def test_a_constraint_defined_differently_is_reported_once(
        self, tmp_path: Path
    ) -> None:
        baseline = _build(tmp_path, "baseline")
        candidate = _candidate(tmp_path)
        _write_constraint(baseline, constraint_id=0, hydro=0)
        _write_constraint(
            candidate, constraint_id=5, hydro=_CANDIDATE_OFFSET, coefficient=2.0
        )

        run = _compare(baseline, candidate)

        assert _table_rows(run.sink, "generic-constraint-differs") == [
            ["RE_1", "terms"]
        ]
        assert facts_of(run.dataset).only_in_candidate["generic_constraint"] == []


class TestSameDirectory:
    def test_the_same_directory_twice_compares_equal_and_says_so(
        self, baseline_output: Path
    ) -> None:
        run = _compare(baseline_output, baseline_output)

        assert all(row.abs_diff == 0.0 for row in run.dataset.render.results)
        assert compare_status(run.dataset) == "ok"
        assert _codes(run.sink).count("same-output-compared") == 1
        facts = facts_of(run.dataset)
        assert facts.baseline_output_dir == facts.candidate_output_dir
        assert facts.baseline_label != facts.candidate_label

    def test_an_item_the_one_directory_lacks_is_one_row_for_both_sides(
        self, tmp_path: Path
    ) -> None:
        output = _build(tmp_path, "case")
        drop_partition(output, _THERMALS)

        run = _compare(output, output)

        thermal = {"section": "thermal", "column": None, "side": "both"}
        assert facts_of(run.dataset).unavailable.count(thermal) == 1
        assert all(row.abs_diff == 0.0 for row in run.dataset.render.results)
        assert _codes(run.sink).count("same-output-compared") == 1
        assert _codes(run.sink).count("section-unavailable") == 1
        assert "thermal" not in _entity_types(run.dataset)


class TestUnusableEntityLists:
    def test_a_thermal_list_without_a_name_makes_its_sections_unavailable(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(tmp_path, hydros=("FURNAS", "ITAIPU", "TUCURUI"))
        (case_dir_for(candidate) / "system" / "thermals.json").write_text(
            json.dumps({"thermals": [{"id": 1}]}), encoding="utf-8"
        )

        run = _compare(baseline_output, candidate)

        facts = facts_of(run.dataset)
        assert ("thermal", None, "candidate") in _unavailable_items(facts)
        assert ("energy-balance", None, "candidate") in _unavailable_items(facts)
        assert "thermal" not in facts.only_in_baseline
        assert "thermal" not in facts.only_in_candidate
        assert "thermal" not in _entity_types(run.dataset)
        assert _table_rows(run.sink, "entity-one-sided") == [
            ["hydro", "TUCURUI", "candidate"]
        ]

    def test_the_unreadable_list_is_reported_once_with_its_side(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(tmp_path)
        (case_dir_for(candidate) / "system" / "thermals.json").write_text(
            json.dumps({"thermals": [{"id": 1}]}), encoding="utf-8"
        )

        run = _compare(baseline_output, candidate)

        warnings = [d for d in run.sink if d.code == "report-file-unreadable"]
        assert len(warnings) == 1
        assert warnings[0].summary.startswith("Candidate output: ")

    def test_a_malformed_line_list_leaves_out_every_section_that_needs_it(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(tmp_path)
        (case_dir_for(candidate) / "system" / "lines.json").write_text(
            "{not json", encoding="utf-8"
        )

        run = _compare(baseline_output, candidate)

        facts = facts_of(run.dataset)
        for section in ("bus-net-import", "line"):
            assert (section, None, "candidate") in _unavailable_items(facts)
        assert ("line-limits", None, "both") in _unavailable_items(facts)
        assert "line" not in facts.only_in_baseline
        assert "line" not in facts.only_in_candidate
        assert "line" not in _entity_types(run.dataset)
        assert "thermal" in _entity_types(run.dataset)
        assert run.dataset.render.line_meta == []

    @pytest.mark.parametrize("side", ["baseline", "candidate"])
    def test_a_malformed_line_list_is_reported_once_with_its_side(
        self, tmp_path: Path, side: str
    ) -> None:
        baseline = _build(tmp_path, "baseline")
        candidate = _candidate(tmp_path)
        damaged = {"baseline": baseline, "candidate": candidate}[side]
        (case_dir_for(damaged) / "system" / "lines.json").write_text(
            "{not json", encoding="utf-8"
        )

        run = _compare(baseline, candidate)

        warnings = [d for d in run.sink if d.code == "report-file-unreadable"]
        assert [d.summary.split(": ")[0] for d in warnings] == [
            f"{side.title()} output"
        ]
        assert {"hydro", "thermal", "bus"} <= _entity_types(run.dataset)
        assert "spot_price" in _variables(run.dataset, "bus")
        assert "net_import_mw" not in _variables(run.dataset, "bus")

    def test_a_malformed_list_on_the_baseline_is_tagged_with_the_baseline(
        self, tmp_path: Path
    ) -> None:
        baseline = _build(tmp_path, "baseline")
        (case_dir_for(baseline) / "system" / "lines.json").write_text(
            "{not json", encoding="utf-8"
        )

        run = _compare(baseline, _candidate(tmp_path))

        warnings = [d for d in run.sink if d.code == "report-file-unreadable"]
        assert warnings
        assert {d.summary.split(":")[0] for d in warnings} == {"Baseline output"}
        assert ("line", None, "baseline") in _unavailable_items(facts_of(run.dataset))

    def test_a_constraint_list_with_a_blank_name_leaves_out_the_constraints(
        self, tmp_path: Path
    ) -> None:
        baseline = _build(tmp_path, "baseline")
        candidate = _candidate(tmp_path)
        _write_constraint(baseline, constraint_id=0, hydro=0)
        _write_constraint(candidate, constraint_id=5, hydro=_CANDIDATE_OFFSET)
        (
            case_dir_for(candidate) / "constraints" / "generic_constraints.json"
        ).write_text(
            json.dumps({"constraints": [{"id": 5, "name": "  ", "expression": "x"}]}),
            encoding="utf-8",
        )

        run = _compare(baseline, candidate)

        facts = facts_of(run.dataset)
        assert ("generic-constraints", None, "candidate") in _unavailable_items(facts)
        assert "generic_constraint" not in facts.only_in_baseline
        assert "generic_constraint" not in facts.only_in_candidate
        assert run.dataset.render.gc_constraints == []
        unreadable = [d for d in run.sink if d.code == "report-file-unreadable"]
        assert len(unreadable) == 1
        assert unreadable[0].summary.startswith("Candidate output: ")


class TestDrift:
    @pytest.mark.parametrize(
        ("section", "kind", "path", "column"),
        _DRIFT,
        ids=[f"{section}-{kind}" for section, kind, _, _ in _DRIFT],
    )
    def test_a_column_dropped_from_the_candidate_is_reported_and_never_raises(
        self,
        baseline_output: Path,
        tmp_path: Path,
        section: str,
        kind: str,
        path: str,
        column: str,
    ) -> None:
        candidate = _candidate(tmp_path)
        drop_column(candidate, path, column)

        run = _compare(baseline_output, candidate)

        item = {"section": section, "column": column, "side": "candidate"}
        assert item in facts_of(run.dataset).unavailable
        run.dataset.validate()

    def test_the_drift_cases_cover_every_available_section_of_the_fixture(self) -> None:
        available = set(REQUIREMENTS) - {
            "generic-constraints",
            "hydro-bounds",
            "line-limits",
        }

        assert {section for section, *_ in _DRIFT} == available

    def test_a_listed_optional_column_has_a_case_wherever_the_section_lists_one(
        self,
    ) -> None:
        listing = {
            name for name, partitions in OPTIONAL_PARTITIONS.items() if partitions
        }
        covered = {section for section, kind, *_ in _DRIFT if kind == "optional"}

        assert covered == listing - {
            "generic-constraints",
            "hydro-bounds",
        }

    def test_a_dropped_variable_leaves_only_that_variable_out(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(tmp_path)
        drop_column(candidate, _EXCHANGES, "net_flow_mw")

        run = _compare(baseline_output, candidate)

        assert "line" not in _entity_types(run.dataset)
        assert "hydro" in _entity_types(run.dataset)
        assert run.dataset.render.line.is_empty()
        assert run.dataset.render.line_meta == []

    def test_a_partition_missing_on_one_side_gives_a_whole_section_row(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(tmp_path)
        drop_partition(candidate, _THERMALS)

        run = _compare(baseline_output, candidate)

        assert ("thermal", None, "candidate") in _unavailable_items(
            facts_of(run.dataset)
        )
        assert run.dataset.render.thermal.is_empty()
        assert "thermal" not in _entity_types(run.dataset)


class TestRenderInputsOfUnavailableSections:
    def test_a_cost_component_that_is_not_cleared_stays_out_of_both_cost_slots(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(tmp_path)
        drop_column(candidate, _COSTS, "deficit_cost")

        run = _compare(baseline_output, candidate)

        render = run.dataset.render
        assert "deficit_cost" not in render.novomodelo_costs
        assert render.novomodelo_costs
        stage_costs = render.novomodelo_stage_costs
        for column in ("immediate_cost", "future_cost", "thermal_cost_total"):
            assert stage_costs[column].null_count() == 0
        assert {"COPER", "CUSTO_FUTURO", "CTERM"} <= set(render.nw_sin["variable"])

    def test_a_section_that_is_unavailable_keeps_its_default_inputs(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(tmp_path)
        drop_partition(candidate, _COSTS)
        drop_partition(candidate, "simulation/hydro_bus_generation")

        run = _compare(baseline_output, candidate)

        render = run.dataset.render
        assert render.novomodelo_costs == {}
        assert render.nw_costs == {}
        assert render.novomodelo_stage_costs.is_empty()
        assert render.bus_aggregates.is_empty()
        assert render.nw_market.is_empty()
        assert render.novomodelo_hydro_per_stage_bounds.is_empty()
        assert render.line_bounds.is_empty()
        assert "bus_ids" not in render.novomodelo_hydro_meta[_CANDIDATE_OFFSET]

    def test_a_bus_section_that_is_unavailable_falls_back_to_the_bus_list(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(tmp_path)
        drop_partition(candidate, _BUSES)

        run = _compare(baseline_output, candidate)

        assert run.dataset.render.novomodelo_bus_meta == {
            100: {"name": "SUDESTE", "has_load": False},
            101: {"name": "SUL", "has_load": False},
        }
        assert run.dataset.render.bus.is_empty()

    def test_a_hydro_section_without_a_cleared_variable_still_reads_its_frame(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(tmp_path)
        for column in (
            "storage_final_hm3",
            "generation_mw",
            "turbined_m3s",
            "spillage_m3s",
            "inflow_m3s",
            "water_value_per_hm3",
            "evaporation_m3s",
            "outflow_m3s",
        ):
            drop_column(candidate, _HYDROS, column)

        run = _compare(baseline_output, candidate)

        assert "hydro" not in _entity_types(run.dataset)
        assert set(run.dataset.render.novomodelo_hydro_means["entity_id"]) == {100, 101}
        assert not run.dataset.render.nw_hydro_slacks.is_empty()

    def test_the_masked_hydro_rows_are_dropped_when_the_hydro_section_is_unavailable(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(tmp_path)
        drop_column(candidate, _SCENARIO_SUMMARY, "probability")

        run = _compare(baseline_output, candidate)

        assert ("hydro", "probability", "candidate") in _unavailable_items(
            facts_of(run.dataset)
        )
        assert "hydro" not in _entity_types(run.dataset)
        assert run.dataset.render.hydro_masked.is_empty()
        assert run.dataset.render.hydro_masked.columns
        assert not run.dataset.render.fpha_split.is_empty()
        assert not run.dataset.render.tie_share.is_empty()


class TestBoundsSections:
    def test_bounds_files_on_both_sides_leave_only_the_generic_constraints_unavailable(
        self, tmp_path: Path
    ) -> None:
        baseline = _build(tmp_path, "baseline")
        candidate = _candidate(tmp_path)
        _write_case_bounds(baseline, entity_id=0)
        _write_case_bounds(candidate, entity_id=_CANDIDATE_OFFSET)

        run = _compare(baseline, candidate)

        assert facts_of(run.dataset).unavailable == [
            {"section": "generic-constraints", "column": None, "side": "both"}
        ]

    def test_the_candidates_bounds_reach_the_render_inputs_with_its_ids_and_values(
        self, tmp_path: Path
    ) -> None:
        baseline = _build(tmp_path, "baseline")
        candidate = _candidate(tmp_path)
        _write_case_bounds(baseline, entity_id=0)
        _write_case_bounds(candidate, entity_id=_CANDIDATE_OFFSET)

        render = _compare(baseline, candidate).dataset.render

        hydro_bounds = render.novomodelo_hydro_per_stage_bounds
        assert hydro_bounds.height == 2
        assert set(hydro_bounds["entity_id"]) == {_CANDIDATE_OFFSET}
        assert set(hydro_bounds["min_storage_hm3"]) == {10.0}
        assert render.line_bounds.height == 2
        assert set(render.line_bounds["line_id"]) == {_CANDIDATE_OFFSET}
        assert set(render.line_bounds["direct_mw"]) == {1000.0}
        assert set(render.line_bounds["reverse_mw"]) == {900.0}

    def test_the_withdrawal_is_compared_stage_by_stage_for_the_hydro_it_names(
        self, tmp_path: Path
    ) -> None:
        baseline = _build(tmp_path, "baseline")
        candidate = _candidate(tmp_path)
        _write_case_bounds(baseline, entity_id=0)
        _write_case_bounds(
            candidate, entity_id=_CANDIDATE_OFFSET, withdrawal=(1.0, 3.0)
        )

        dataset = _compare(baseline, candidate).dataset

        assert "withdrawal_m3s" in _variables(dataset, "hydro")
        rows = {
            row.stage: row
            for row in dataset.render.results
            if row.variable == "withdrawal_m3s" and row.entity_name == "FURNAS"
        }
        assert set(rows) == {0, 1}
        assert (
            rows[0].newave_value,
            rows[0].novomodelo_value,
            rows[0].abs_diff,
        ) == (1.0, 1.0, 0.0)
        assert (
            rows[1].newave_value,
            rows[1].novomodelo_value,
            rows[1].abs_diff,
        ) == (2.0, 3.0, 1.0)

    def test_the_hydro_detail_and_network_tabs_lose_their_notice_with_the_files(
        self, tmp_path: Path, equal_runs: _Run
    ) -> None:
        baseline = _build(tmp_path, "baseline")
        candidate = _candidate(tmp_path)
        _write_case_bounds(baseline, entity_id=0)
        _write_case_bounds(candidate, entity_id=_CANDIDATE_OFFSET)

        with_files = version_report_tabs(_compare(baseline, candidate).dataset)
        without_files = version_report_tabs(equal_runs.dataset)

        for tab in ("tab-hydro-detail", "tab-network"):
            assert "Results unavailable" not in with_files[tab]
            assert "Results unavailable" in without_files[tab]

    def test_bounds_files_on_the_baseline_only_are_reported_missing_in_the_candidate(
        self, tmp_path: Path
    ) -> None:
        baseline = _build(tmp_path, "baseline")
        candidate = _candidate(tmp_path)
        _write_case_bounds(baseline, entity_id=0)

        run = _compare(baseline, candidate)

        unavailable = _unavailable_items(facts_of(run.dataset))
        assert ("hydro-bounds", None, "candidate") in unavailable
        assert ("line-limits", None, "candidate") in unavailable
        assert run.dataset.render.novomodelo_hydro_per_stage_bounds.is_empty()
        assert run.dataset.render.line_bounds.is_empty()


class TestStatus:
    def test_a_scaled_candidate_is_a_mismatch_with_a_row_that_is_not_close(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(tmp_path, scale=1.1)

        run = _compare(baseline_output, candidate)

        assert compare_status(run.dataset) == "mismatch"
        verdict = build_compare_verdict(run.dataset)
        assert verdict.close < verdict.total

    def test_a_baseline_without_simulation_or_convergence_has_no_judged_row(
        self, tmp_path: Path
    ) -> None:
        baseline = _build(tmp_path, "baseline")
        drop_partition(baseline, "simulation")
        drop_partition(baseline, "training/convergence.parquet")

        run = _compare(baseline, _candidate(tmp_path))

        assert compare_status(run.dataset) == "no-comparable-rows"
        assert facts_of(run.dataset).upper_bound_comparability == "bounds-missing"
        assert not run.dataset.render.results

    def test_a_baseline_without_simulation_still_judges_the_final_lower_bound(
        self, tmp_path: Path
    ) -> None:
        baseline = _build(tmp_path, "baseline")
        drop_partition(baseline, "simulation")

        run = _compare(baseline, _candidate(tmp_path))

        assert compare_status(run.dataset) == "ok"
        assert "final_lower_bound" in _variables(run.dataset, "final_bounds")
        assert "final_bound_ratio" in _variables(run.dataset, "final_bounds")
        assert _entity_types(run.dataset) == {"final_bounds", "convergence"}


class TestBoundsFrame:
    def test_an_exact_upper_bound_keeps_a_null_standard_deviation_column_typed(
        self, tmp_path: Path
    ) -> None:
        baseline = _build(tmp_path, "baseline", upper_bound_kind="exact")
        candidate = _candidate(tmp_path, upper_bound_kind="exact")

        run = _compare(baseline, candidate)

        bounds = run.dataset.metadata[CONVERGENCE_BOUNDS_KEY]
        assert isinstance(bounds, pl.DataFrame)
        assert bounds.schema == _BOUNDS_SCHEMA
        assert bounds.height == 6
        assert bounds["upper_bound_std"].null_count() == 6
        assert set(bounds["upper_bound_kind"]) == {"exact"}

    def test_a_side_without_convergence_data_has_no_row_in_the_frame(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(tmp_path)
        drop_partition(candidate, "training/convergence.parquet")

        run = _compare(baseline_output, candidate)

        bounds = run.dataset.metadata[CONVERGENCE_BOUNDS_KEY]
        assert isinstance(bounds, pl.DataFrame)
        assert bounds.schema == _BOUNDS_SCHEMA
        assert bounds["side"].to_list() == ["baseline"] * 3
        assert facts_of(run.dataset).upper_bound_comparability == "bounds-missing"
        assert "final_bounds" not in _entity_types(run.dataset)
        assert "convergence" not in _entity_types(run.dataset)

    def test_both_sides_without_convergence_data_give_the_typed_empty_frame(
        self, tmp_path: Path
    ) -> None:
        outputs = [
            _build(tmp_path, name, id_offset=offset)
            for name, offset in (("baseline", 0), ("candidate", _CANDIDATE_OFFSET))
        ]
        for output in outputs:
            drop_partition(output, "training/convergence.parquet")

        run = _compare(*outputs)

        bounds = run.dataset.metadata[CONVERGENCE_BOUNDS_KEY]
        assert isinstance(bounds, pl.DataFrame)
        assert bounds.schema == _BOUNDS_SCHEMA
        assert bounds.is_empty()
        run.dataset.to_dir(tmp_path / "out")

    def test_the_frame_keeps_each_sides_own_iterations(self, tmp_path: Path) -> None:
        baseline = _build(tmp_path, "baseline", iterations=2)
        candidate = _candidate(tmp_path, iterations=4)

        run = _compare(baseline, candidate)

        bounds = run.dataset.metadata[CONVERGENCE_BOUNDS_KEY]
        assert isinstance(bounds, pl.DataFrame)
        assert bounds["side"].to_list() == ["baseline"] * 2 + ["candidate"] * 4
        assert bounds["iteration"].to_list() == [1, 2, 1, 2, 3, 4]
        assert _variables(run.dataset, "convergence") == {
            "lower_bound",
            "upper_bound",
        }
        names = {
            row.entity_name
            for row in run.dataset.render.results
            if row.entity_type == "convergence"
        }
        assert names == {"iteration_1", "iteration_2"}


class TestEntitiesAndStages:
    def test_an_entity_on_one_side_only_is_listed_and_never_compared(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(
            tmp_path,
            hydros=("FURNAS", "ITAIPU", "TUCURUI"),
            thermals=("GAS_A", "GAS_B"),
            lines=(("SE-S", "SUDESTE", "SUL"), ("SE-S2", "SUDESTE", "SUL")),
        )

        run = _compare(baseline_output, candidate)

        facts = facts_of(run.dataset)
        assert facts.only_in_baseline == {
            "hydro": [],
            "bus": [],
            "thermal": [],
            "line": [],
        }
        assert facts.only_in_candidate == {
            "hydro": ["TUCURUI"],
            "bus": [],
            "thermal": ["GAS_B"],
            "line": ["SE-S2"],
        }
        assert _table_rows(run.sink, "entity-one-sided") == [
            ["hydro", "TUCURUI", "candidate"],
            ["line", "SE-S2", "candidate"],
            ["thermal", "GAS_B", "candidate"],
        ]
        names = {row.entity_name for row in run.dataset.render.results}
        assert not names & {"TUCURUI", "GAS_B", "SE-S2"}

    def test_a_name_used_twice_is_reported_and_left_out_of_the_rows(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(tmp_path, hydros=("FURNAS", "ITAIPU", " itaipu "))

        run = _compare(baseline_output, candidate)

        assert _table_rows(run.sink, "entity-name-duplicated") == [
            ["hydro", "ITAIPU, itaipu", "candidate"]
        ]
        hydro = {
            row.entity_name
            for row in run.dataset.render.results
            if row.entity_type == "hydro"
        }
        assert hydro == {"FURNAS"}

    def test_stage_counts_that_differ_compare_the_common_stages_only(
        self, tmp_path: Path
    ) -> None:
        baseline = _build(tmp_path, "baseline", stages=3)
        candidate = _candidate(tmp_path, stages=2)

        run = _compare(baseline, candidate)

        assert _table_rows(run.sink, "stage-sets-differ") == [[2, "baseline"]]
        stages = {
            row.stage
            for row in run.dataset.render.results
            if row.entity_type == "hydro"
        }
        assert stages == {0, 1}
        assert run.dataset.render.nw_max_stage == 1

    def test_a_side_without_stages_has_no_operation_row_and_reports_the_stage_sets(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(tmp_path)
        (case_dir_for(candidate) / "stages.json").unlink()

        run = _compare(baseline_output, candidate)

        assert _entity_types(run.dataset) <= {"final_bounds", "convergence"}
        assert _table_rows(run.sink, "stage-sets-differ") == [
            [0, "baseline"],
            [1, "baseline"],
        ]
        assert run.dataset.render.nw_max_stage is None


class TestFailureOrder:
    def test_the_baseline_directory_is_checked_before_the_candidate_directory(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(SourceFileError, match="baseline"), dx.collect():
            build_version_dataset(
                tmp_path / "no-baseline",
                tmp_path / "no-candidate",
                tolerance=_TOLERANCE,
            )

    def test_the_candidate_directory_is_checked_before_the_baseline_is_read(
        self, tmp_path: Path
    ) -> None:
        not_an_output = tmp_path / "empty"
        not_an_output.mkdir()

        with pytest.raises(SourceFileError, match="candidate"), dx.collect():
            build_version_dataset(
                not_an_output, tmp_path / "no-candidate", tolerance=_TOLERANCE
            )

    def test_the_baseline_is_read_before_the_candidate(self, tmp_path: Path) -> None:
        not_an_output = tmp_path / "empty"
        not_an_output.mkdir()
        candidate = _candidate(tmp_path)
        (case_dir_for(candidate) / "system" / "hydros.json").unlink()

        with (
            pytest.raises(NovomodeloOutputError, match="baseline") as raised,
            dx.collect(),
        ):
            build_version_dataset(not_an_output, candidate, tolerance=_TOLERANCE)

        assert type(raised.value) is NovomodeloOutputError

    def test_a_candidate_without_buses_fails_after_a_valid_baseline(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _candidate(tmp_path)
        (case_dir_for(candidate) / "system" / "buses.json").unlink()

        with pytest.raises(SourceFileError, match="buses.json"), dx.collect():
            build_version_dataset(baseline_output, candidate, tolerance=_TOLERANCE)

    def test_a_corrupt_candidate_partition_fails_with_the_baseline_diagnostics_kept(
        self, tmp_path: Path
    ) -> None:
        baseline = _build(tmp_path, "baseline")
        (case_dir_for(baseline) / "system" / "lines.json").write_text(
            "{not json", encoding="utf-8"
        )
        candidate = _candidate(tmp_path)
        corrupt_file(candidate, _HYDROS)

        with pytest.raises(NovomodeloReadError), dx.collect() as sink:
            build_version_dataset(baseline, candidate, tolerance=_TOLERANCE)

        assert any(
            d.code == "report-file-unreadable"
            and d.summary.startswith("Baseline output: ")
            for d in sink
        )


class TestFindings:
    def test_each_finding_is_emitted_once_in_a_run_that_triggers_all_but_one(
        self, tmp_path: Path
    ) -> None:
        baseline = _build(
            tmp_path,
            "baseline",
            hydros=("FURNAS", "ITAIPU", "itaipu"),
            stages=3,
            iterations=3,
        )
        candidate = _candidate(
            tmp_path,
            hydros=("FURNAS", "TUCURUI"),
            stages=2,
            iterations=4,
            upper_bound_kind="exact",
            fpha_planes=False,
        )
        _write_constraint(baseline, constraint_id=0, hydro=0)
        _write_constraint(
            candidate, constraint_id=5, hydro=_CANDIDATE_OFFSET, coefficient=2.0
        )

        run = _compare(baseline, candidate)

        counts = {code: _codes(run.sink).count(code) for code in _FINDING_CODES}
        assert counts == {
            "same-output-compared": 0,
            "entity-one-sided": 1,
            "entity-name-duplicated": 1,
            "stage-sets-differ": 1,
            "section-unavailable": 1,
            "generic-constraint-differs": 1,
            "upper-bound-not-comparable": 1,
            "iteration-counts-differ": 1,
        }

    def test_the_findings_follow_the_order_of_the_report(self, tmp_path: Path) -> None:
        baseline = _build(tmp_path, "baseline", stages=3, iterations=3)
        candidate = _candidate(tmp_path, stages=2, iterations=4, thermals=("GAS_B",))

        run = _compare(baseline, candidate)

        emitted = [code for code in _codes(run.sink) if code in _FINDING_CODES]
        assert emitted == [
            "entity-one-sided",
            "stage-sets-differ",
            "section-unavailable",
            "iteration-counts-differ",
        ]
