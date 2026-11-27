"""Tests of ``comparators.version.manifest``.

The hand-built instance covers the record on its own; every other case runs two
``build_solved_case`` trees through ``build_version_dataset`` and writes the real
artifacts to ``tmp_path``. Nothing is mocked.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import pytest

import novomodelo_bridge
from novomodelo_bridge.comparators.dataset import ComparisonDataset
from novomodelo_bridge.comparators.manifest import ComparisonManifest
from novomodelo_bridge.comparators.version.facts import (
    METADATA_KEY,
    VersionFacts,
    facts_of,
)
from novomodelo_bridge.comparators.version.manifest import (
    MANIFEST_VERSION,
    VersionComparisonManifest,
    write_version_artifacts,
)
from novomodelo_bridge.comparators.version.results import build_version_dataset
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.git import git_sha
from tests.novomodelo_output_builders import (
    SolvedCaseSpec,
    build_solved_case,
    drop_partition,
)

_TOLERANCE = 0.05
_COMMAND = "compare novomodelo"
_SPEC = SolvedCaseSpec(load_stats=True)
_CANDIDATE_OFFSET = 100
_MANIFEST_FILE = "comparison.json"
_ALL_FORMATS = ["csv", "json", "parquet"]

_KEYS = [
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
_FACT_KEYS = [field.name for field in fields(VersionFacts)]
_ABSENT_KEYS = [
    "risk_measures_differ",
    "source_dir",
    "novomodelo_output_dir",
    "newave_version",
    "input_files",
]
_VARYING_KEYS = ["timestamp", "git_sha", "bridge_version"]
_DIRECTORY_PLACEHOLDER = "<root>"


def _manifest(**changes: Any) -> VersionComparisonManifest:
    values: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "command": _COMMAND,
        "baseline_output_dir": "base",
        "candidate_output_dir": "cand",
        "baseline_version": "0.20.0",
        "candidate_version": None,
        "baseline_label": "v0.20.0 (baseline)",
        "candidate_label": "cand",
        "only_in_baseline": {"hydro": ["A"]},
        "only_in_candidate": {"hydro": []},
        "unavailable": [{"section": "line-limits", "column": None, "side": "both"}],
        "upper_bound_comparable": False,
        "upper_bound_comparability": "bounds-missing",
        "baseline_upper_bound_kind": None,
        "candidate_upper_bound_kind": "statistical",
        "baseline_gap_percent": None,
        "candidate_gap_percent": 1.5,
        "gap_delta_pp": None,
        "tolerance": _TOLERANCE,
        "bridge_version": "0.0.0",
        "git_sha": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "artifacts": [_MANIFEST_FILE],
        "top_divergences": [{"variable": "generation_mw", "abs_diff": 1.0}],
        "diagnostics_summary": {"info": 1},
        "diagnostics": [{"code": "x"}],
    }
    return VersionComparisonManifest(**{**values, **changes})


def _build(root: Path, name: str, **changes: Any) -> Path:
    return build_solved_case(root / name, replace(_SPEC, **changes))


def _dataset(baseline: Path, candidate: Path) -> ComparisonDataset:
    with dx.collect():
        return build_version_dataset(baseline, candidate, tolerance=_TOLERANCE)


def _write(
    dataset: ComparisonDataset,
    out_dir: Path,
    formats: list[str] | None = None,
    **kwargs: Any,
) -> VersionComparisonManifest:
    return write_version_artifacts(
        dataset,
        command=_COMMAND,
        tolerance=_TOLERANCE,
        out_dir=out_dir,
        formats=[] if formats is None else formats,
        **kwargs,
    )


def _read(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / _MANIFEST_FILE).read_text(encoding="utf-8"))


def _normalized(document: dict[str, Any], root: Path) -> dict[str, Any]:
    for key in ("baseline_output_dir", "candidate_output_dir"):
        document[key] = document[key].replace(str(root), _DIRECTORY_PLACEHOLDER)
    for key in _VARYING_KEYS:
        del document[key]
    return document


def _reject_constant(name: str) -> NoReturn:
    raise AssertionError(f"comparison.json holds the non-JSON constant {name}")


@pytest.fixture(scope="module")
def baseline_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_solved_case(tmp_path_factory.mktemp("baseline"), _SPEC)


@pytest.fixture(scope="module")
def candidate_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_solved_case(
        tmp_path_factory.mktemp("candidate"),
        replace(_SPEC, id_offset=_CANDIDATE_OFFSET),
    )


@pytest.fixture(scope="module")
def equal_runs(baseline_output: Path, candidate_output: Path) -> ComparisonDataset:
    return _dataset(baseline_output, candidate_output)


class TestRecordShape:
    def test_the_fields_are_the_twenty_six_keys_in_order(self) -> None:
        assert [field.name for field in fields(VersionComparisonManifest)] == _KEYS

    def test_the_facts_block_is_the_fields_of_the_facts_record(self) -> None:
        assert _KEYS[2:18] == _FACT_KEYS

    def test_the_manifest_version_is_one(self) -> None:
        assert MANIFEST_VERSION == 1

    def test_the_json_keys_follow_the_field_order(self, tmp_path: Path) -> None:
        _manifest().to_json(tmp_path / _MANIFEST_FILE)

        assert list(_read(tmp_path)) == _KEYS


class TestToJson:
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("key", ["baseline_gap_percent", "tolerance"])
    def test_a_non_finite_number_raises_and_leaves_no_file(
        self, tmp_path: Path, key: str, value: float
    ) -> None:
        path = tmp_path / "nan.json"

        with pytest.raises(ValueError, match="not JSON compliant"):
            _manifest(**{key: value}).to_json(path)

        assert not path.exists()

    def test_the_parent_directories_are_created(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / _MANIFEST_FILE

        _manifest().to_json(path)

        assert path.is_file()

    def test_a_round_trip_returns_an_equal_manifest(self, tmp_path: Path) -> None:
        manifest = _manifest()
        path = tmp_path / _MANIFEST_FILE

        manifest.to_json(path)

        assert VersionComparisonManifest.from_json(path) == manifest

    @pytest.mark.parametrize("key", _KEYS)
    def test_a_file_lacking_a_key_raises_instead_of_filling_it(
        self, tmp_path: Path, key: str
    ) -> None:
        path = tmp_path / _MANIFEST_FILE
        _manifest().to_json(path)
        data = _read(tmp_path)
        del data[key]
        path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(TypeError, match=key):
            VersionComparisonManifest.from_json(path)

    def test_a_missing_file_is_named_by_the_version_label(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Version comparison manifest"):
            VersionComparisonManifest.from_json(tmp_path / _MANIFEST_FILE)


class TestEqualRuns:
    def test_an_empty_format_list_writes_only_the_manifest(
        self, equal_runs: ComparisonDataset, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"

        _write(equal_runs, out_dir, [])

        written = _read(out_dir)
        assert sorted(path.name for path in out_dir.iterdir()) == [_MANIFEST_FILE]
        assert list(written) == _KEYS
        assert written["manifest_version"] == 1
        assert written["command"] == _COMMAND
        assert written["artifacts"] == [_MANIFEST_FILE]
        assert written["diagnostics_summary"] == {}
        assert written["diagnostics"] == []

    def test_the_three_formats_list_their_files_and_not_the_metadata_side_file(
        self, equal_runs: ComparisonDataset, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"

        manifest = _write(equal_runs, out_dir, _ALL_FORMATS)

        expected = [
            "comparison.csv",
            "comparison.json",
            "comparison.parquet",
            "summary.csv",
            "summary.json",
            "summary.parquet",
            "top_divergences.json",
        ]
        assert manifest.artifacts == expected
        assert _read(out_dir)["artifacts"] == expected
        assert sorted(path.name for path in out_dir.iterdir()) == sorted(
            [*expected, "metadata.json"]
        )

    def test_the_facts_block_and_the_top_divergences_are_copied_unchanged(
        self, equal_runs: ComparisonDataset, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"

        _write(equal_runs, out_dir, _ALL_FORMATS)

        written = _read(out_dir)
        facts = facts_of(equal_runs).to_dict()
        assert {key: written[key] for key in _FACT_KEYS} == facts
        assert written["top_divergences"] == equal_runs.metadata["top_divergences"]

    def test_the_directories_are_the_strings_the_caller_typed(
        self,
        equal_runs: ComparisonDataset,
        baseline_output: Path,
        candidate_output: Path,
        tmp_path: Path,
    ) -> None:
        _write(equal_runs, tmp_path)

        written = _read(tmp_path)
        assert written["baseline_output_dir"] == str(baseline_output)
        assert written["candidate_output_dir"] == str(candidate_output)

    def test_the_arguments_and_the_provenance_values_are_recorded(
        self, equal_runs: ComparisonDataset, tmp_path: Path
    ) -> None:
        before = datetime.now(tz=UTC)

        manifest = _write(equal_runs, tmp_path)

        assert manifest.tolerance == _TOLERANCE
        assert manifest.bridge_version == novomodelo_bridge.__version__
        assert manifest.git_sha == git_sha()
        stamp = datetime.fromisoformat(manifest.timestamp)
        assert before <= stamp <= datetime.now(tz=UTC)

    def test_the_returned_manifest_is_the_one_written(
        self, equal_runs: ComparisonDataset, tmp_path: Path
    ) -> None:
        manifest = _write(equal_runs, tmp_path, _ALL_FORMATS)

        assert VersionComparisonManifest.from_json(tmp_path / _MANIFEST_FILE) == (
            manifest
        )

    @pytest.mark.parametrize("key", _ABSENT_KEYS)
    def test_the_record_has_no_key_of_the_source_case_manifest(
        self, equal_runs: ComparisonDataset, tmp_path: Path, key: str
    ) -> None:
        _write(equal_runs, tmp_path)

        assert key not in _read(tmp_path)

    def test_two_runs_over_the_same_pair_write_equal_documents(
        self, tmp_path: Path
    ) -> None:
        documents = []
        for name in ("run-1", "run-2"):
            root = tmp_path / name
            dataset = _dataset(
                _build(root, "baseline"),
                _build(root, "candidate", id_offset=_CANDIDATE_OFFSET),
            )
            _write(dataset, root / "out", _ALL_FORMATS)
            documents.append(_normalized(_read(root / "out"), root))

        assert documents[0] == documents[1]


class TestDiagnostics:
    def test_the_summary_counts_by_severity_and_the_list_keeps_the_given_order(
        self, equal_runs: ComparisonDataset, tmp_path: Path
    ) -> None:
        warning = dx.Diagnostic(
            code="first-code",
            severity=dx.Severity.WARNING,
            category="Compare data",
            title="First",
            summary="A warning.",
        )
        info = dx.Diagnostic(
            code="second-code",
            severity=dx.Severity.INFO,
            category="Compare data",
            title="Second",
            summary="A note.",
        )

        manifest = _write(equal_runs, tmp_path, diagnostics=[warning, info])

        expected = [warning.to_dict(), info.to_dict()]
        written = _read(tmp_path)
        assert written["diagnostics_summary"] == {"warning": 1, "info": 1}
        assert written["diagnostics"] == expected
        assert manifest.diagnostics_summary == {"warning": 1, "info": 1}
        assert manifest.diagnostics == expected


class TestDamagedPairs:
    def test_a_baseline_without_bounds_writes_null_gaps_and_no_json_constant(
        self, candidate_output: Path, tmp_path: Path
    ) -> None:
        baseline = _build(tmp_path, "baseline")
        drop_partition(baseline, "simulation")
        drop_partition(baseline, "training/convergence.parquet")
        dataset = _dataset(baseline, candidate_output)
        out_dir = tmp_path / "out"

        _write(dataset, out_dir, ["json"])

        text = (out_dir / _MANIFEST_FILE).read_text(encoding="utf-8")
        written = json.loads(text, parse_constant=_reject_constant)
        assert written["baseline_gap_percent"] is None
        assert written["gap_delta_pp"] is None
        assert written["upper_bound_comparability"] == "bounds-missing"

    def test_a_candidate_without_thermals_lists_the_item_as_the_facts_record_does(
        self, baseline_output: Path, tmp_path: Path
    ) -> None:
        candidate = _build(tmp_path, "candidate", id_offset=_CANDIDATE_OFFSET)
        drop_partition(candidate, "simulation/thermals")
        dataset = _dataset(baseline_output, candidate)

        _write(dataset, tmp_path / "out")

        written = _read(tmp_path / "out")
        assert written["unavailable"] == facts_of(dataset).unavailable
        assert {"section": "thermal", "column": None, "side": "candidate"} in (
            written["unavailable"]
        )


class TestFailures:
    def test_an_unknown_format_raises_and_writes_no_manifest(
        self, equal_runs: ComparisonDataset, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"

        with pytest.raises(ValueError, match="bogus"):
            _write(equal_runs, out_dir, ["bogus"])

        assert not (out_dir / _MANIFEST_FILE).exists()

    def test_a_dataset_without_facts_raises_and_leaves_the_directory_untouched(
        self, equal_runs: ComparisonDataset, tmp_path: Path
    ) -> None:
        metadata = {
            key: value
            for key, value in equal_runs.metadata.items()
            if key != METADATA_KEY
        }
        dataset = ComparisonDataset(
            tidy=equal_runs.tidy,
            summary=equal_runs.summary,
            metadata=metadata,
            render=equal_runs.render,
        )
        out_dir = tmp_path / "out"

        with pytest.raises(TypeError, match=METADATA_KEY):
            _write(dataset, out_dir, _ALL_FORMATS)

        assert not out_dir.exists()

    def test_the_source_case_reader_cannot_read_the_file(
        self, equal_runs: ComparisonDataset, tmp_path: Path
    ) -> None:
        _write(equal_runs, tmp_path)

        with pytest.raises(TypeError, match="source_dir"):
            ComparisonManifest.from_json(tmp_path / _MANIFEST_FILE)
