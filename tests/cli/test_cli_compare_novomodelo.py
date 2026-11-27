"""Tests of ``cli.compare_version``, the ``compare novomodelo`` handler.

The handler is called directly with a ``CompareArgs``. The failure mappings run
with the dataset builder replaced by a function that emits one note and raises;
every other case runs a pair of ``build_solved_case`` trees through the whole
handler. The classes from ``TestExitOrder`` on run the registered command line
over the same trees, in process through ``CliRunner`` and, for the three exit
codes, as a real process. Tier-1: nothing here imports ``novomodelo``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

import pytest
import typer
from typer.testing import CliRunner

from novomodelo_bridge.cli import app
from novomodelo_bridge.cli.args import CompareArgs
from novomodelo_bridge.cli.compare_version import (
    ARTIFACTS_DIRNAME,
    _run_novomodelo_comparison,
)
from novomodelo_bridge.comparators.version.convergence import UB_COMPARABILITY
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.agreement import DEFAULT_TOLERANCE
from novomodelo_bridge.core.errors import (
    BridgeError,
    FieldParseError,
    NovomodeloOutputError,
    NovomodeloPartitionMissingError,
    OutputWriteError,
    SourceFileError,
)
from novomodelo_bridge.novomodelo.readers import NovomodeloReadError
from tests.cli.conftest import _run_cli_subprocess
from tests.novomodelo_output_builders import (
    SolvedCaseSpec,
    build_solved_case,
    corrupt_file,
    drop_partition,
)

_LOG = logging.getLogger(__name__)

_BUILDER = "novomodelo_bridge.comparators.version.results.build_version_dataset"
_REPORT = "novomodelo_bridge.comparators.version.report.build_version_report"
_ENVIRONMENT = (
    "NOVOMODELO_BRIDGE_OUT_DIR",
    "NOVOMODELO_BRIDGE_FORMAT",
    "NOVOMODELO_BRIDGE_RESULTS_TOLERANCE",
)
_SPEC = SolvedCaseSpec(load_stats=True)
_CANDIDATE_OFFSET = 100
_BASELINE_SPEC = replace(_SPEC, software_version="0.20.0")
_CANDIDATE_SPEC = replace(_SPEC, id_offset=_CANDIDATE_OFFSET)
_TITLE = "Novomodelo 0.21.0 vs Novomodelo 0.20.0 Results Comparison"
_MANIFEST = "comparison.json"
_ENVELOPE_KEYS = ["schema_version", "command", "status", "summary", "diagnostics"]
_SUMMARY_KEYS = [
    "tolerance",
    "total",
    "close",
    "moderate",
    "far",
    "far_variables",
    "baseline_version",
    "candidate_version",
    "only_in_baseline",
    "only_in_candidate",
    "upper_bound_comparable",
    "baseline_gap_percent",
    "candidate_gap_percent",
    "gap_delta_pp",
]
_MANIFEST_KEYS = [
    "manifest_version",
    "command",
    "baseline_output_dir",
    "candidate_output_dir",
    "baseline_version",
    "candidate_version",
    "baseline_label",
    "candidate_label",
    "only_in_baseline",
    "only_in_candidate",
    "unavailable",
    "upper_bound_comparable",
    "upper_bound_comparability",
    "baseline_upper_bound_kind",
    "candidate_upper_bound_kind",
    "baseline_gap_percent",
    "candidate_gap_percent",
    "gap_delta_pp",
    "tolerance",
    "bridge_version",
    "git_sha",
    "timestamp",
    "artifacts",
    "top_divergences",
    "diagnostics_summary",
    "diagnostics",
]
_COMPARABILITY = (
    "comparable",
    "bounds-missing",
    "kind-missing",
    "kind-differs",
    "risk-averse",
    "risk-unknown",
    "bound-nonpositive",
)
_ARTIFACTS = "version_comparison_artifacts"
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
_NOTE = dx.Diagnostic(
    code="reader-note",
    severity=dx.Severity.INFO,
    category="Compare data",
    title="Read before the failure",
    summary="A note emitted before the builder stopped.",
)
_FAILURES = [
    pytest.param(SourceFileError("directory missing"), 1, id="source-file"),
    pytest.param(FieldParseError("hydros.json malformed"), 1, id="field-parse"),
    pytest.param(NovomodeloOutputError("not an output"), 2, id="output"),
    pytest.param(NovomodeloReadError("file unreadable"), 2, id="read"),
    pytest.param(
        NovomodeloPartitionMissingError("partition missing"), 2, id="partition-missing"
    ),
]


class _Pair(NamedTuple):
    baseline: Path
    candidate: Path


@pytest.fixture(autouse=True)
def _isolated_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run from an empty directory with no setting in the environment, so no
    config file or variable changes what the handler resolves."""
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def _pair(tmp_path: Path, **candidate_changes: Any) -> _Pair:
    baseline = build_solved_case(tmp_path / "baseline", _BASELINE_SPEC)
    candidate = build_solved_case(
        tmp_path / "candidate", replace(_CANDIDATE_SPEC, **candidate_changes)
    )
    return _Pair(baseline, candidate)


@pytest.fixture
def pair(tmp_path: Path) -> _Pair:
    return _pair(tmp_path)


@pytest.fixture
def unbuilt(tmp_path: Path) -> _Pair:
    """Two directories no builder reads, for the cases that replace it."""
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    return _Pair(tmp_path / "baseline", candidate)


def _args(outputs: _Pair, **changes: Any) -> CompareArgs:
    args = CompareArgs(
        json_output=False,
        verbose=0,
        log_file=None,
        no_color=True,
        quiet=False,
        source_dir=outputs.baseline,
        novomodelo_output_dir=outputs.candidate,
        tolerance=None,
        format=None,
        out_dir=None,
    )
    return replace(args, **changes)


def _fails_with(error: Exception) -> Callable[..., NoReturn]:
    def build(*_args: object, **_kwargs: object) -> NoReturn:
        dx.emit(_NOTE, logger=_LOG)
        raise error

    return build


def _names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def _finding_codes(document: dict[str, Any]) -> list[str]:
    return [
        diagnostic["code"]
        for diagnostic in document["diagnostics"]
        if diagnostic["code"] in _FINDING_CODES
    ]


def _manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / _MANIFEST).read_text(encoding="utf-8"))


class TestFailureMapping:
    @pytest.mark.parametrize(("error", "code"), _FAILURES)
    def test_under_json_one_error_envelope_follows_the_note_and_the_exit_code_maps(
        self,
        error: Exception,
        code: int,
        unbuilt: _Pair,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(_BUILDER, _fails_with(error))

        with pytest.raises(typer.Exit) as raised:
            _run_novomodelo_comparison(_args(unbuilt, json_output=True))

        assert raised.value.exit_code == code
        document = json.loads(capsys.readouterr().out)
        assert document["command"] == "compare novomodelo"
        assert document["status"] == "error"
        titles = [diagnostic["title"] for diagnostic in document["diagnostics"]]
        assert titles == [_NOTE.title, "Comparison failed"]
        assert not (unbuilt.candidate / ARTIFACTS_DIRNAME).exists()

    @pytest.mark.parametrize(("error", "code"), _FAILURES)
    def test_without_json_the_failure_is_rendered_on_stderr_only(
        self,
        error: Exception,
        code: int,
        unbuilt: _Pair,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(_BUILDER, _fails_with(error))

        with pytest.raises(typer.Exit) as raised:
            _run_novomodelo_comparison(_args(unbuilt))

        assert raised.value.exit_code == code
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Comparison failed" in captured.err

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("invariant"),
            TypeError("invariant"),
            BridgeError("unmapped"),
            OutputWriteError("unmapped"),
        ],
        ids=["value-error", "type-error", "bridge-error", "output-write-error"],
    )
    def test_an_exception_outside_the_mapping_propagates(
        self, error: Exception, unbuilt: _Pair, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_BUILDER, _fails_with(error))

        with pytest.raises(Exception) as raised:
            _run_novomodelo_comparison(_args(unbuilt, json_output=True))

        assert raised.value is error

    def test_a_bad_format_exits_2_before_the_builder_runs(
        self,
        unbuilt: _Pair,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def never_called(*_args: object, **_kwargs: object) -> NoReturn:
            pytest.fail("the builder ran")

        monkeypatch.setattr(_BUILDER, never_called)

        with pytest.raises(typer.Exit) as raised:
            _run_novomodelo_comparison(
                _args(unbuilt, format=["bogus"], json_output=True)
            )

        assert raised.value.exit_code == 2
        document = json.loads(capsys.readouterr().out)
        assert document["command"] == "compare novomodelo"
        assert document["status"] == "error"


class TestRun:
    def test_the_default_run_writes_the_data_artifacts_and_prints_the_summary(
        self, pair: _Pair, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run_novomodelo_comparison(_args(pair))

        out = capsys.readouterr().out
        assert _names(pair.candidate / ARTIFACTS_DIRNAME) == [
            "comparison.json",
            "comparison.parquet",
            "metadata.json",
            "summary.json",
            "summary.parquet",
            "top_divergences.json",
        ]
        assert _TITLE in out
        assert "Artifacts written to" in out
        assert not (pair.candidate / "comparison_artifacts").exists()

    def test_under_json_stdout_is_one_envelope_whose_diagnostics_match_the_manifest(
        self, pair: _Pair, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run_novomodelo_comparison(_args(pair, json_output=True))

        document = json.loads(capsys.readouterr().out)
        assert list(document) == _ENVELOPE_KEYS
        assert document["command"] == "compare novomodelo"
        assert document["status"] == "ok"
        assert list(document["summary"]) == _SUMMARY_KEYS
        assert _finding_codes(document).count("section-unavailable") == 1
        manifest = _manifest(pair.candidate / ARTIFACTS_DIRNAME)
        assert manifest["diagnostics"] == document["diagnostics"]

    def test_each_finding_code_appears_once_when_an_entity_is_one_sided(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        extra = _pair(tmp_path, hydros=("FURNAS", "ITAIPU", "NEWPLANT"))

        _run_novomodelo_comparison(_args(extra, json_output=True))

        codes = _finding_codes(json.loads(capsys.readouterr().out))
        assert "entity-one-sided" in codes
        assert len(codes) == len(set(codes))

    def test_a_scaled_candidate_is_a_mismatch_and_the_run_still_returns(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        scaled = _pair(tmp_path, scale=1.1)

        _run_novomodelo_comparison(_args(scaled, json_output=True))

        assert json.loads(capsys.readouterr().out)["status"] == "mismatch"

    def test_quiet_drops_the_summary_and_keeps_the_status_line(
        self, pair: _Pair, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run_novomodelo_comparison(_args(pair, quiet=True))

        out = capsys.readouterr().out
        assert "Results Comparison" not in out
        assert "Artifacts written to" in out

    @pytest.mark.parametrize(
        ("tolerance", "recorded"), [(None, DEFAULT_TOLERANCE), (0.25, 0.25)]
    )
    def test_the_resolved_tolerance_reaches_the_dataset_and_the_manifest(
        self,
        tolerance: float | None,
        recorded: float,
        pair: _Pair,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _run_novomodelo_comparison(_args(pair, json_output=True, tolerance=tolerance))

        document = json.loads(capsys.readouterr().out)
        assert document["summary"]["tolerance"] == recorded
        assert _manifest(pair.candidate / ARTIFACTS_DIRNAME)["tolerance"] == recorded


class TestFormats:
    def test_html_writes_the_report_and_the_manifest_alone(self, pair: _Pair) -> None:
        _run_novomodelo_comparison(_args(pair, format=["html"]))

        artifacts = pair.candidate / ARTIFACTS_DIRNAME
        assert _names(artifacts) == [_MANIFEST, "report.html"]
        report = (artifacts / "report.html").read_text(encoding="utf-8")
        assert f"<title>{_TITLE}</title>" in report
        assert _manifest(artifacts)["artifacts"] == [_MANIFEST]

    def test_csv_writes_the_tables_and_the_manifest(self, pair: _Pair) -> None:
        _run_novomodelo_comparison(_args(pair, format=["csv"]))

        artifacts = pair.candidate / ARTIFACTS_DIRNAME
        assert _names(artifacts) == ["comparison.csv", _MANIFEST, "summary.csv"]

    def test_the_report_advisory_goes_to_stderr_under_json(
        self, pair: _Pair, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run_novomodelo_comparison(_args(pair, json_output=True, format=["html"]))

        captured = capsys.readouterr()
        assert json.loads(captured.out)["status"] == "ok"
        assert "HTML report written to" in captured.err

    def test_diagnostics_emitted_while_the_report_is_built_reach_the_envelope(
        self,
        pair: _Pair,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def build_report(_dataset: object) -> str:
            dx.emit(_NOTE, logger=_LOG)
            return "<html></html>"

        monkeypatch.setattr(_REPORT, build_report)

        _run_novomodelo_comparison(_args(pair, json_output=True, format=["html"]))

        document = json.loads(capsys.readouterr().out)
        titles = [diagnostic["title"] for diagnostic in document["diagnostics"]]
        assert _NOTE.title in titles
        manifest = _manifest(pair.candidate / ARTIFACTS_DIRNAME)
        assert manifest["diagnostics"] == document["diagnostics"]


class TestOutDir:
    def test_an_explicit_out_dir_receives_the_files_and_the_default_is_not_created(
        self, pair: _Pair, tmp_path: Path
    ) -> None:
        target = tmp_path / "elsewhere"

        _run_novomodelo_comparison(_args(pair, out_dir=target, format=["json"]))

        assert _MANIFEST in _names(target)
        assert not (pair.candidate / ARTIFACTS_DIRNAME).exists()

    def test_a_write_failure_warns_on_stderr_and_the_run_returns(
        self, pair: _Pair, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("", encoding="utf-8")

        _run_novomodelo_comparison(
            _args(pair, out_dir=blocker / "artifacts", format=["json"])
        )

        captured = capsys.readouterr()
        assert "Warning: failed to write artifacts:" in captured.err
        assert "Artifacts written to" not in captured.out

    def test_a_write_failure_under_json_still_emits_the_envelope(
        self, pair: _Pair, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("", encoding="utf-8")

        _run_novomodelo_comparison(
            _args(
                pair, out_dir=blocker / "artifacts", format=["json"], json_output=True
            )
        )

        captured = capsys.readouterr()
        assert "Warning: failed to write artifacts:" in captured.err
        assert json.loads(captured.out)["status"] == "ok"


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"stdout holds the non-finite constant {name}")


def _invoke(*argv: Path | str) -> tuple[int, dict[str, Any]]:
    result = CliRunner().invoke(
        app, ["compare", "novomodelo", *map(str, argv), "--json"]
    )
    document = json.loads(result.stdout, parse_constant=_reject_constant)
    assert list(document) == _ENVELOPE_KEYS
    assert document["command"] == "compare novomodelo"
    return result.exit_code, document


def _all_codes(document: dict[str, Any]) -> list[str]:
    return [diagnostic["code"] for diagnostic in document["diagnostics"]]


def _keys_at_any_depth(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for item in value.values() for key in _keys_at_any_depth(item)
        }
    if isinstance(value, list):
        return {key for item in value for key in _keys_at_any_depth(item)}
    return set()


_valid = build_solved_case


def _missing(case_dir: Path, _spec: SolvedCaseSpec) -> Path:
    return case_dir / "output"


def _empty(case_dir: Path, _spec: SolvedCaseSpec) -> Path:
    output = case_dir / "output"
    output.mkdir(parents=True)
    return output


def _without_hydros(case_dir: Path, spec: SolvedCaseSpec) -> Path:
    output = build_solved_case(case_dir, spec)
    (case_dir / "system" / "hydros.json").unlink()
    return output


def _with_malformed_buses(case_dir: Path, spec: SolvedCaseSpec) -> Path:
    output = build_solved_case(case_dir, spec)
    (case_dir / "system" / "buses.json").write_text("{", encoding="utf-8")
    return output


def _with_corrupt_costs(case_dir: Path, spec: SolvedCaseSpec) -> Path:
    output = build_solved_case(case_dir, spec)
    corrupt_file(output, "simulation/costs")
    return output


_MISSING_FILE = "source-file-missing"
_UNREADABLE = "novomodelo-output-unreadable"
_MALFORMED = "source-field-parse"


def _case(
    case_id: str,
    baseline: Callable[[Path, SolvedCaseSpec], Path],
    candidate: Callable[[Path, SolvedCaseSpec], Path],
    exit_code: int,
    code: str | None = None,
    named: str | None = None,
    options: tuple[str, ...] = (),
) -> Any:
    return pytest.param(
        baseline, candidate, options, exit_code, code, named, id=case_id
    )


_EXIT_ORDER = [
    _case("bad-format", _missing, _missing, 2, options=("--format", "bogus")),
    _case("baseline-dir-missing", _missing, _missing, 1, _MISSING_FILE, "baseline"),
    _case("candidate-dir-missing", _valid, _missing, 1, _MISSING_FILE, "candidate"),
    _case("not-an-output-baseline", _empty, _valid, 2, _UNREADABLE),
    _case("not-an-output-candidate", _valid, _empty, 2, _UNREADABLE),
    _case("no-hydros-baseline", _without_hydros, _valid, 1, _MISSING_FILE),
    _case("no-hydros-candidate", _valid, _without_hydros, 1, _MISSING_FILE),
    _case("bad-buses-baseline", _with_malformed_buses, _valid, 1, _MALFORMED),
    _case("bad-buses-candidate", _valid, _with_malformed_buses, 1, _MALFORMED),
    _case("corrupt-costs-baseline", _with_corrupt_costs, _valid, 2, _UNREADABLE),
    _case("corrupt-costs-candidate", _valid, _with_corrupt_costs, 2, _UNREADABLE),
    _case("order-floor-before-names", _empty, _without_hydros, 2, _UNREADABLE),
    _case("order-baseline-first", _without_hydros, _empty, 1, _MISSING_FILE),
    _case("order-dirs-before-reads", _with_corrupt_costs, _missing, 1, _MISSING_FILE),
]


class TestExitOrder:
    @pytest.mark.parametrize(
        ("baseline_state", "candidate_state", "options", "exit_code", "code", "named"),
        _EXIT_ORDER,
    )
    def test_each_step_fails_with_its_exit_code_and_one_error_envelope(
        self,
        baseline_state: Callable[[Path, SolvedCaseSpec], Path],
        candidate_state: Callable[[Path, SolvedCaseSpec], Path],
        options: tuple[str, ...],
        exit_code: int,
        code: str | None,
        named: str | None,
        tmp_path: Path,
    ) -> None:
        baseline = baseline_state(tmp_path / "first", _BASELINE_SPEC)
        candidate = candidate_state(tmp_path / "second", _CANDIDATE_SPEC)

        returned, document = _invoke(baseline, candidate, *options)

        assert returned == exit_code
        assert document["status"] == "error"
        assert document["summary"] == {}
        if code is not None:
            last = document["diagnostics"][-1]
            assert last["code"] == code
            assert named is None or named in last["summary"]

    def test_a_missing_argument_is_a_usage_error_naming_the_command(self) -> None:
        returned, document = _invoke()

        assert returned == 2
        assert document["status"] == "error"
        assert document["summary"] == {}
        assert document["diagnostics"][-1]["code"] == "usage-error"


class TestSummaryAndStatus:
    def test_the_default_pair_is_ok_with_the_fourteen_keys_and_finite_gaps(
        self, pair: _Pair
    ) -> None:
        returned, document = _invoke(pair.baseline, pair.candidate)

        assert returned == 0
        assert document["status"] == "ok"
        summary = document["summary"]
        assert list(summary) == _SUMMARY_KEYS
        for key in ("baseline_gap_percent", "candidate_gap_percent", "gap_delta_pp"):
            assert summary[key] is None or isinstance(summary[key], float)

    def test_a_scaled_candidate_is_a_mismatch_and_exits_0(self, tmp_path: Path) -> None:
        scaled = _pair(tmp_path, scale=1.1)

        returned, document = _invoke(scaled.baseline, scaled.candidate)

        assert returned == 0
        assert document["status"] == "mismatch"

    def test_outputs_without_simulation_still_compare_their_final_bounds(
        self, pair: _Pair
    ) -> None:
        drop_partition(pair.baseline, "simulation")
        drop_partition(pair.candidate, "simulation")

        returned, document = _invoke(pair.baseline, pair.candidate)

        assert returned == 0
        assert document["status"] in {"ok", "mismatch"}
        assert document["summary"]["total"] >= 1

    def test_a_baseline_without_simulation_and_convergence_compares_nothing(
        self, pair: _Pair
    ) -> None:
        drop_partition(pair.baseline, "simulation")
        drop_partition(pair.baseline, "training/convergence.parquet")

        returned, document = _invoke(pair.baseline, pair.candidate)

        assert returned == 0
        assert document["status"] == "no-comparable-rows"
        assert document["summary"]["baseline_gap_percent"] is None

    def test_quiet_under_json_still_prints_one_document(self, pair: _Pair) -> None:
        returned, document = _invoke(pair.baseline, pair.candidate, "--quiet")

        assert returned == 0
        assert document["status"] == "ok"

    def test_without_json_the_title_is_on_stdout_and_the_notes_on_stderr(
        self, pair: _Pair, dumb_terminal: None
    ) -> None:
        result = CliRunner().invoke(
            app, ["compare", "novomodelo", str(pair.baseline), str(pair.candidate)]
        )

        assert result.exit_code == 0
        assert _TITLE in result.stdout
        assert "Results unavailable" in result.stderr

    def test_chosen_labels_name_the_runs_in_the_terminal_and_the_report(
        self, pair: _Pair, dumb_terminal: None, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"

        result = CliRunner().invoke(
            app,
            [
                "compare",
                "novomodelo",
                str(pair.baseline),
                str(pair.candidate),
                "--baseline-label",
                "pré-inversão",
                "--candidate-label",
                "LPPs completas",
                "--format",
                "console,html",
                "--out-dir",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0
        title = "LPPs completas vs pré-inversão Results Comparison"
        assert title in result.stdout
        [header] = [
            line for line in result.stdout.splitlines() if "Bounds and cost" in line
        ]
        assert "pré-inversão" in header
        assert "LPPs completas" in header
        assert "Novomodelo" not in header
        report = (out_dir / "report.html").read_text(encoding="utf-8")
        assert f"<title>{title}</title>" in report
        manifest = json.loads((out_dir / "comparison.json").read_text("utf-8"))
        assert manifest["baseline_label"] == "pré-inversão"
        assert manifest["candidate_label"] == "LPPs completas"


class TestArtifacts:
    def test_the_default_directory_is_new_and_an_existing_one_is_left_alone(
        self, pair: _Pair
    ) -> None:
        marker = pair.candidate / "comparison_artifacts" / "marker.txt"
        marker.parent.mkdir()
        marker.write_text("keep", encoding="utf-8")
        baseline_names = _names(pair.baseline)

        returned, _document = _invoke(pair.baseline, pair.candidate)

        assert returned == 0
        assert _names(marker.parent) == ["marker.txt"]
        assert marker.read_text(encoding="utf-8") == "keep"
        manifest = _manifest(pair.candidate / _ARTIFACTS)
        assert list(manifest) == _MANIFEST_KEYS
        assert manifest["manifest_version"] == 1
        assert _names(pair.baseline) == baseline_names

    def test_out_dir_with_all_formats_receives_the_report_and_the_tables(
        self, pair: _Pair, tmp_path: Path
    ) -> None:
        target = tmp_path / "art"

        returned, _document = _invoke(
            pair.baseline, pair.candidate, "--out-dir", target, "--format", "all"
        )

        assert returned == 0
        assert {"report.html", "comparison.csv"} <= set(_names(target))
        assert not (pair.candidate / _ARTIFACTS).exists()
        assert not (pair.baseline / _ARTIFACTS).exists()


class TestSameOutput:
    def test_one_directory_twice_is_named_once_and_a_missing_item_is_one_both_row(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(tmp_path / "solo", _SPEC)
        drop_partition(output, "simulation/thermals")

        returned, document = _invoke(output, output)

        assert returned == 0
        codes = _all_codes(document)
        assert codes.count("same-output-compared") == 1
        assert not any("in-one-output" in code for code in codes)
        (unavailable,) = [
            diagnostic
            for diagnostic in document["diagnostics"]
            if diagnostic["code"] == "section-unavailable"
        ]
        rows = unavailable["table"]["rows"]
        assert [row for row in rows if row[0] == "thermal"] == [
            ["thermal", "—", "both"]
        ]

    def test_two_builds_of_one_version_are_two_outputs_of_that_version(
        self, tmp_path: Path
    ) -> None:
        first = build_solved_case(tmp_path / "first", _SPEC)
        second = build_solved_case(tmp_path / "second", _SPEC)

        returned, document = _invoke(first, second)

        assert returned == 0
        assert "same-output-compared" not in _all_codes(document)
        summary = document["summary"]
        assert summary["baseline_version"] == summary["candidate_version"] == "0.21.0"


class TestDriftAndRisk:
    def test_a_thermal_list_without_names_is_unavailable_and_never_one_sided(
        self, tmp_path: Path
    ) -> None:
        outputs = _pair(tmp_path, hydros=("FURNAS", "ITAIPU", "NEWPLANT"))
        (outputs.candidate.parent / "system" / "thermals.json").write_text(
            json.dumps({"thermals": [{"id": 1}]}), encoding="utf-8"
        )

        returned, document = _invoke(outputs.baseline, outputs.candidate)

        assert returned == 0
        manifest = _manifest(outputs.candidate / _ARTIFACTS)
        assert {"section": "thermal", "column": None, "side": "candidate"} in manifest[
            "unavailable"
        ]
        assert "thermal" not in document["summary"]["only_in_baseline"]
        assert "thermal" not in document["summary"]["only_in_candidate"]
        one_sided = [
            row
            for diagnostic in document["diagnostics"]
            if diagnostic["code"] == "entity-one-sided"
            for row in diagnostic["table"]["rows"]
        ]
        assert ["hydro", "NEWPLANT", "candidate"] in one_sided
        assert not [row for row in one_sided if row[0] == "thermal"]

    def test_a_string_risk_measure_is_unknown_so_the_bounds_are_not_comparable(
        self, tmp_path: Path
    ) -> None:
        outputs = _pair(tmp_path, risk_measure="cvar")

        returned, document = _invoke(outputs.baseline, outputs.candidate)

        assert returned == 0
        assert document["summary"]["upper_bound_comparable"] is False
        manifest = _manifest(outputs.candidate / _ARTIFACTS)
        assert manifest["upper_bound_comparability"] == "risk-unknown"

    def test_differing_bound_kinds_name_the_objectives_and_never_the_risk_flag(
        self, tmp_path: Path
    ) -> None:
        outputs = _pair(
            tmp_path,
            upper_bound_kind="exact",
            risk_measure={"cvar": {"alpha": 0.95, "lambda": 0.25}},
        )

        returned, document = _invoke(outputs.baseline, outputs.candidate)

        assert returned == 0
        manifest = _manifest(outputs.candidate / _ARTIFACTS)
        assert manifest["upper_bound_comparability"] == "kind-differs"
        (finding,) = [
            diagnostic
            for diagnostic in document["diagnostics"]
            if diagnostic["code"] == "upper-bound-not-comparable"
        ]
        assert "bound different objectives" in finding["summary"]
        keys = _keys_at_any_depth(manifest) | _keys_at_any_depth(document["summary"])
        assert {"upper_bound_comparability", "section"} <= keys
        assert "risk_measures_differ" not in keys


class TestPins:
    def test_the_comparability_literal_is_the_set_the_comparison_persists(
        self,
    ) -> None:
        assert len(set(_COMPARABILITY)) == len(_COMPARABILITY)
        assert set(_COMPARABILITY) == set(UB_COMPARABILITY)

    def test_the_artifacts_directory_literal_is_the_handlers_constant(self) -> None:
        assert _ARTIFACTS == ARTIFACTS_DIRNAME

    def test_a_pair_that_differs_everywhere_raises_six_pinned_finding_codes_once(
        self, tmp_path: Path
    ) -> None:
        baseline = build_solved_case(
            tmp_path / "first",
            replace(
                _BASELINE_SPEC,
                hydros=("FURNAS", "FURNAS", "ITAIPU"),
                stages=3,
                iterations=4,
            ),
        )
        candidate = build_solved_case(
            tmp_path / "second",
            replace(
                _CANDIDATE_SPEC,
                hydros=("FURNAS", "NEWPLANT"),
                iterations=2,
                upper_bound_kind="exact",
            ),
        )

        returned, document = _invoke(baseline, candidate)

        assert returned == 0
        codes = [
            diagnostic["code"]
            for diagnostic in document["diagnostics"]
            if diagnostic["category"] == "Compare data"
        ]
        assert sorted(codes) == [
            "entity-name-duplicated",
            "entity-one-sided",
            "iteration-counts-differ",
            "section-unavailable",
            "stage-sets-differ",
            "upper-bound-not-comparable",
        ]
        assert set(codes) <= set(_FINDING_CODES)


class TestProcess:
    @pytest.mark.parametrize(
        ("baseline_state", "returncode", "status"),
        [
            pytest.param(_valid, 0, "ok", id="compared"),
            pytest.param(_missing, 1, "error", id="directory-missing"),
            pytest.param(_empty, 2, "error", id="not-an-output"),
        ],
    )
    def test_the_process_exits_0_1_or_2_and_prints_one_document_under_json(
        self,
        baseline_state: Callable[[Path, SolvedCaseSpec], Path],
        returncode: int,
        status: str,
        tmp_path: Path,
    ) -> None:
        baseline = baseline_state(tmp_path / "first", _BASELINE_SPEC)
        candidate = build_solved_case(tmp_path / "second", _CANDIDATE_SPEC)

        result = _run_cli_subprocess(
            "compare", "novomodelo", str(baseline), str(candidate), "--json"
        )

        assert result.returncode == returncode
        document = json.loads(result.stdout)
        assert list(document) == _ENVELOPE_KEYS
        assert document["command"] == "compare novomodelo"
        assert document["status"] == status
