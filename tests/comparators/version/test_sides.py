"""Tests of ``comparators.version.sides`` over ``build_solved_case`` trees damaged
with the builder's helpers, never mocks."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from novomodelo_bridge.comparators.version.sides import (
    SideInputs,
    read_side,
    require_output_dir,
    side_tagged,
)
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.errors import (
    FieldParseError,
    NovomodeloOutputError,
    SourceFileError,
)
from novomodelo_bridge.novomodelo.readers import NovomodeloReadError
from tests.novomodelo_output_builders import (
    SolvedCaseSpec,
    build_solved_case,
    corrupt_file,
    drop_partition,
)

_LOG = logging.getLogger(__name__)

_THERMALS_EFFECT = "the comparison leaves out the thermal plants of this output."
_LINES_EFFECT = "the comparison leaves out the lines of this output."


def _read(
    output: Path, side: str = "baseline"
) -> tuple[SideInputs, list[dx.Diagnostic]]:
    with dx.collect() as sink:
        inputs = read_side(output, side)
    return inputs, sink


def _write_system(output: Path, filename: str, text: str) -> None:
    (output.parent / "system" / filename).write_text(text, encoding="utf-8")


def _loud(sink: list[dx.Diagnostic]) -> list[dx.Diagnostic]:
    return [d for d in sink if d.severity is not dx.Severity.INFO]


def _emit(summary: str = "The file could not be read.") -> None:
    dx.emit(_diagnostic(summary), logger=_LOG)


def _diagnostic(summary: str = "The file could not be read.") -> dx.Diagnostic:
    return dx.Diagnostic(
        code="report-file-unreadable",
        severity=dx.Severity.WARNING,
        category="Case and result files",
        title="x.json could not be read",
        summary=summary,
        table=dx.DiagnosticTable(columns=["File"], rows=[["x.json"]]),
        notes=["Path: x.json"],
        remediation="Check x.json.",
    )


class TestCleanSide:
    def test_a_clean_side_reads_every_field(self, tmp_path: Path) -> None:
        output = build_solved_case(
            tmp_path / "case", SolvedCaseSpec(software_version="0.20.0")
        )

        inputs, sink = _read(output, "baseline")

        assert inputs.side == "baseline"
        assert inputs.output_dir == output
        assert inputs.version == "0.20.0"
        assert [h["name"] for h in inputs.hydros] == ["FURNAS", "ITAIPU"]
        assert [b["name"] for b in inputs.buses] == ["SUDESTE", "SUL"]
        assert inputs.thermals is not None
        assert [t["name"] for t in inputs.thermals] == ["GAS_A"]
        assert inputs.lines is not None
        assert [line["name"] for line in inputs.lines] == ["SE-S"]
        assert inputs.risk_aversion == {0: 0.0, 1: 0.0}
        assert isinstance(inputs.partitions["simulation/hydros"], frozenset)
        assert inputs.partitions["scenarios/load_seasonal_stats.parquet"] is None
        assert inputs.usable_entity_lists == {"thermal": True, "line": True}
        assert _loud(sink) == []

    def test_the_candidate_is_read_by_the_same_function(self, tmp_path: Path) -> None:
        output = build_solved_case(tmp_path / "case")

        baseline, _ = _read(output, "baseline")
        candidate, _ = _read(output, "candidate")

        assert candidate.side == "candidate"
        assert replace(candidate, side="baseline") == baseline

    def test_a_version_the_output_does_not_record_is_unknown(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(
            tmp_path / "jan-26", SolvedCaseSpec(software_version=None)
        )

        inputs, _ = _read(output)

        assert inputs.version is None

    def test_a_version_under_the_legacy_key_is_unknown(self, tmp_path: Path) -> None:
        output = build_solved_case(
            tmp_path / "case",
            SolvedCaseSpec(software_version="0.17.0", version_key="cobre_version"),
        )

        inputs, _ = _read(output)

        assert inputs.version is None

    def test_an_unreadable_metadata_file_is_tagged_and_leaves_the_version_unknown(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        (output / "training" / "metadata.json").write_text("not json", encoding="utf-8")

        inputs, sink = _read(output, "candidate")

        assert inputs.version is None
        assert [d.summary.split(" could not")[0] for d in _loud(sink)] == [
            "Candidate output: output/training/metadata.json"
        ]


class TestFailureOrder:
    def test_a_missing_directory_raises_source_file_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "nowhere"

        with pytest.raises(SourceFileError) as raised:
            read_side(missing, "candidate")

        assert str(raised.value) == (
            f"The candidate output directory was not found: {missing}."
        )
        assert raised.value.path == str(missing)

    def test_a_file_is_not_a_directory(self, tmp_path: Path) -> None:
        file = tmp_path / "output"
        file.write_text("", encoding="utf-8")

        with pytest.raises(SourceFileError):
            require_output_dir(file, "baseline")

    def test_an_existing_directory_passes_the_directory_check(
        self, tmp_path: Path
    ) -> None:
        require_output_dir(tmp_path, "baseline")

    def test_an_absent_hydros_file_raises_source_file_error(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        (output.parent / "system" / "hydros.json").unlink()

        with pytest.raises(SourceFileError):
            read_side(output, "baseline")

    def test_a_malformed_buses_file_raises_field_parse_error(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        _write_system(output, "buses.json", "not json")

        with pytest.raises(FieldParseError):
            read_side(output, "baseline")

    def test_a_nameless_hydro_raises_field_parse_error(self, tmp_path: Path) -> None:
        output = build_solved_case(tmp_path / "case")
        _write_system(output, "hydros.json", '{"hydros": [{"id": 0}]}')

        with pytest.raises(FieldParseError) as raised:
            read_side(output, "baseline")

        assert raised.value.field == "name"

    def test_an_empty_directory_is_not_a_novomodelo_output(
        self, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(NovomodeloOutputError) as raised:
            read_side(empty, "baseline")

        assert raised.type is NovomodeloOutputError
        assert str(raised.value) == (
            f"The baseline directory {empty} holds neither training/ nor "
            "simulation/, so it is not a Novomodelo output."
        )
        assert raised.value.path == str(empty)

    def test_valid_system_files_do_not_make_a_directory_an_output(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        drop_partition(output, "training")
        drop_partition(output, "simulation")

        with pytest.raises(NovomodeloOutputError) as raised:
            read_side(output, "candidate")

        assert raised.type is NovomodeloOutputError

    def test_the_output_check_comes_before_the_required_names(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        drop_partition(output, "training")
        drop_partition(output, "simulation")
        (output.parent / "system" / "hydros.json").unlink()
        _write_system(output, "buses.json", "not json")

        with pytest.raises(NovomodeloOutputError) as raised:
            read_side(output, "baseline")

        assert raised.type is NovomodeloOutputError

    def test_hydros_are_checked_before_buses(self, tmp_path: Path) -> None:
        output = build_solved_case(tmp_path / "case")
        (output.parent / "system" / "hydros.json").unlink()
        _write_system(output, "buses.json", "not json")

        with pytest.raises(SourceFileError):
            read_side(output, "baseline")

    def test_a_malformed_hydros_file_is_reported_before_an_absent_buses_file(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        _write_system(output, "hydros.json", "not json")
        (output.parent / "system" / "buses.json").unlink()

        with pytest.raises(FieldParseError):
            read_side(output, "baseline")

    def test_the_required_names_are_checked_before_the_probe(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        corrupt_file(output, "training/convergence.parquet")
        _write_system(output, "buses.json", "not json")

        with pytest.raises(FieldParseError):
            read_side(output, "baseline")

    def test_a_present_partition_the_probe_cannot_read_raises_read_error(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        corrupt_file(output, "training/convergence.parquet")

        with pytest.raises(NovomodeloReadError):
            read_side(output, "baseline")


class TestOptionalEntityLists:
    @pytest.mark.parametrize(
        ("key", "effect"),
        [("thermals", _THERMALS_EFFECT), ("lines", _LINES_EFFECT)],
    )
    @pytest.mark.parametrize(
        "template",
        [
            pytest.param("not json", id="unparseable"),
            pytest.param('{{"other": []}}', id="no-list"),
            pytest.param('{{"{key}": [{{"id": 0}}]}}', id="nameless-entry"),
        ],
    )
    def test_an_unreadable_list_is_none_with_one_warning(
        self, tmp_path: Path, key: str, effect: str, template: str
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        _write_system(output, f"{key}.json", template.format(key=key))

        inputs, sink = _read(output)

        assert getattr(inputs, key) is None
        assert inputs.usable_entity_lists == {
            "thermal": key != "thermals",
            "line": key != "lines",
        }
        [warning] = _loud(sink)
        assert warning.code == "report-file-unreadable"
        assert warning.severity is dx.Severity.WARNING
        assert warning.summary == (
            f"Baseline output: system/{key}.json could not be read, so {effect}"
        )

    def test_an_absent_thermals_file_is_a_case_without_thermals(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        (output.parent / "system" / "thermals.json").unlink()

        inputs, sink = _read(output)

        assert inputs.thermals == []
        assert inputs.usable_entity_lists["thermal"] is True
        assert _loud(sink) == []

    def test_an_absent_lines_file_is_a_case_without_lines(self, tmp_path: Path) -> None:
        output = build_solved_case(tmp_path / "case")
        (output.parent / "system" / "lines.json").unlink()

        inputs, sink = _read(output)

        assert inputs.lines == []
        assert inputs.usable_entity_lists["line"] is True
        assert _loud(sink) == []

    def test_an_empty_list_is_usable(self, tmp_path: Path) -> None:
        output = build_solved_case(
            tmp_path / "case", SolvedCaseSpec(thermals=(), lines=())
        )

        inputs, sink = _read(output)

        assert (inputs.thermals, inputs.lines) == ([], [])
        assert inputs.usable_entity_lists == {"thermal": True, "line": True}
        assert _loud(sink) == []

    def test_each_unreadable_file_gets_its_own_warning(self, tmp_path: Path) -> None:
        output = build_solved_case(tmp_path / "case")
        _write_system(output, "thermals.json", "not json")
        _write_system(output, "lines.json", "not json")

        inputs, sink = _read(output)

        assert (inputs.thermals, inputs.lines) == (None, None)
        assert inputs.usable_entity_lists == {"thermal": False, "line": False}
        assert [d.summary.split(" could not")[0] for d in _loud(sink)] == [
            "Baseline output: system/thermals.json",
            "Baseline output: system/lines.json",
        ]

    def test_an_unusable_list_is_the_only_one_reported_unusable(
        self, tmp_path: Path
    ) -> None:
        inputs, _ = _read(build_solved_case(tmp_path / "case"))

        assert replace(inputs, thermals=None).usable_entity_lists == {
            "thermal": False,
            "line": True,
        }
        assert replace(inputs, lines=None).usable_entity_lists == {
            "thermal": True,
            "line": False,
        }
        assert replace(inputs, thermals=[], lines=[]).usable_entity_lists == {
            "thermal": True,
            "line": True,
        }


class TestRiskAversion:
    def test_a_map_is_returned_as_read(self, tmp_path: Path) -> None:
        spec = SolvedCaseSpec(risk_measure={"cvar": {"alpha": 0.95, "lambda": 0.25}})

        inputs, sink = _read(build_solved_case(tmp_path / "case", spec))

        assert inputs.risk_aversion == {0: 0.25, 1: 0.25}
        assert _loud(sink) == []

    def test_an_absent_risk_measure_is_risk_neutral(self, tmp_path: Path) -> None:
        output = build_solved_case(tmp_path / "case")
        stages_path = output.parent / "stages.json"
        stages = json.loads(stages_path.read_text(encoding="utf-8"))
        for stage in stages["stages"]:
            del stage["risk_measure"]
        stages_path.write_text(json.dumps(stages), encoding="utf-8")

        inputs, sink = _read(output)

        assert inputs.risk_aversion == {0: 0.0, 1: 0.0}
        assert _loud(sink) == []

    def test_an_absent_stages_file_is_the_empty_map_not_none(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        (output.parent / "stages.json").unlink()

        inputs, sink = _read(output)

        assert inputs.risk_aversion == {}
        assert _loud(sink) == []

    @pytest.mark.parametrize(
        "risk_measure",
        [
            pytest.param({"cvar": {"alpha": 0.25}}, id="cvar-without-lambda"),
            pytest.param("cvar", id="cvar-as-a-string"),
        ],
    )
    def test_an_unknown_risk_measure_is_none_with_a_tagged_warning(
        self, tmp_path: Path, risk_measure: str | dict[str, object]
    ) -> None:
        output = build_solved_case(
            tmp_path / "case", SolvedCaseSpec(risk_measure=risk_measure)
        )

        inputs, sink = _read(output, "candidate")

        assert inputs.risk_aversion is None
        [warning] = _loud(sink)
        assert warning.code == "report-file-unreadable"
        assert warning.summary.startswith("Candidate output: stages.json could not")


class TestSideTags:
    def test_the_same_finding_on_both_sides_survives_finalization(
        self, tmp_path: Path
    ) -> None:
        baseline_output = build_solved_case(tmp_path / "base")
        candidate_output = build_solved_case(tmp_path / "cand")
        _write_system(baseline_output, "thermals.json", "not json")
        _write_system(candidate_output, "thermals.json", "not json")

        with dx.collect() as sink:
            baseline = read_side(baseline_output, "baseline")
            candidate = read_side(candidate_output, "candidate")
        finalized = dx.finalize_diagnostics(sink)

        assert baseline.thermals is None
        assert candidate.thermals is None
        assert baseline.usable_entity_lists == {"thermal": False, "line": True}
        assert candidate.usable_entity_lists == {"thermal": False, "line": True}
        assert [d.code for d in finalized] == ["report-file-unreadable"] * 2
        assert [d.summary.split(":")[0] for d in finalized] == [
            "Baseline output",
            "Candidate output",
        ]

    def test_only_the_summary_changes(self) -> None:
        original = _diagnostic()

        with dx.collect() as outer:
            with side_tagged("candidate"):
                dx.emit(original, logger=_LOG)

        assert outer == [
            replace(original, summary="Candidate output: The file could not be read.")
        ]

    def test_captured_diagnostics_keep_their_order(self) -> None:
        with dx.collect() as outer:
            with side_tagged("baseline"):
                for summary in ("One.", "Two.", "Three."):
                    _emit(summary)

        assert [d.summary for d in outer] == [
            "Baseline output: One.",
            "Baseline output: Two.",
            "Baseline output: Three.",
        ]

    def test_a_diagnostic_emitted_before_an_exception_is_still_re_emitted(
        self,
    ) -> None:
        with dx.collect() as outer:
            with pytest.raises(RuntimeError):
                with side_tagged("baseline"):
                    _emit()
                    raise RuntimeError("stop")

        assert [d.summary for d in outer] == [
            "Baseline output: The file could not be read."
        ]

    def test_a_read_error_after_a_warning_keeps_the_tagged_warning(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        _write_system(output, "thermals.json", "not json")
        corrupt_file(output, "training/convergence.parquet")

        with dx.collect() as sink:
            with pytest.raises(NovomodeloReadError):
                read_side(output, "candidate")

        assert [d.summary.split(" could not")[0] for d in sink] == [
            "Candidate output: system/thermals.json"
        ]

    def test_a_diagnostic_outside_the_block_is_not_tagged(self) -> None:
        with dx.collect() as outer:
            _emit("Before.")
            with side_tagged("baseline"):
                pass
            _emit("After.")

        assert [d.summary for d in outer] == ["Before.", "After."]
