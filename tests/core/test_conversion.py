"""Tests for the conversion runner both tracks share (``core/conversion.py``)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core import emission_checks
from novomodelo_bridge.core.conversion import (
    ClearedArtifacts,
    ConversionReport,
    run_conversion,
)
from novomodelo_bridge.core.errors import EmissionCheckError, OutputExistsError

_ARTIFACTS = ClearedArtifacts(subdirs=("system",), files=("config.json",))


def _diagnostic(code: str, severity: dx.Severity) -> dx.Diagnostic:
    return dx.Diagnostic(
        code=code, severity=severity, category="Test", title=code, summary=code
    )


def _write_outputs(dst: Path) -> None:
    (dst / "system").mkdir(parents=True)
    (dst / "config.json").write_text("{}", encoding="utf-8")


def test_run_conversion_folds_findings_into_the_report_once_each(
    tmp_path: Path,
) -> None:
    def convert() -> ConversionReport:
        dx.emit(_diagnostic("early", dx.Severity.WARNING))
        dx.emit(_diagnostic("note", dx.Severity.INFO))
        dx.emit(_diagnostic("early", dx.Severity.WARNING))
        return ConversionReport()

    report = run_conversion(convert, tmp_path, _ARTIFACTS, force=False, dry_run=False)

    assert [d.code for d in report.diagnostics] == ["early", "note"]
    assert report.warnings == ["early"]


def test_a_log_warning_is_not_a_finding(tmp_path: Path) -> None:
    def convert() -> ConversionReport:
        logging.getLogger("novomodelo_bridge.test").warning("log text")
        return ConversionReport()

    report = run_conversion(convert, tmp_path, _ARTIFACTS, force=False, dry_run=False)

    assert report.diagnostics == []
    assert report.warnings == []


def test_failed_emission_check_reaches_the_caller_with_its_findings(
    tmp_path: Path,
) -> None:
    def convert() -> ConversionReport:
        _write_outputs(tmp_path)
        dx.emit(_diagnostic("early", dx.Severity.WARNING))
        dx.emit(_diagnostic("early", dx.Severity.WARNING))
        emission_checks.run_and_gate(
            lambda: dx.emit(_diagnostic("check-failed", dx.Severity.ERROR))
        )
        return ConversionReport()

    with (
        dx.collect() as outer,
        pytest.raises(EmissionCheckError, match="check-failed"),
    ):
        run_conversion(convert, tmp_path, _ARTIFACTS, force=False, dry_run=False)

    assert [d.code for d in outer] == ["early", "check-failed"]
    assert not (tmp_path / "system").exists()
    assert not (tmp_path / "config.json").exists()


def test_dry_run_failure_leaves_dst_untouched(tmp_path: Path) -> None:
    _write_outputs(tmp_path)

    def convert() -> ConversionReport:
        raise RuntimeError("boom")

    with dx.collect(), pytest.raises(RuntimeError, match="boom"):
        run_conversion(convert, tmp_path, _ARTIFACTS, force=True, dry_run=True)

    assert (tmp_path / "system").is_dir()
    assert (tmp_path / "config.json").is_file()


def _unreachable() -> ConversionReport:
    raise AssertionError("convert ran")


@pytest.mark.parametrize("dry_run", [False, True])
def test_a_dst_that_holds_files_is_refused_and_left_as_it_was(
    tmp_path: Path, dry_run: bool
) -> None:
    _write_outputs(tmp_path)

    with pytest.raises(OutputExistsError, match="already holds files"):
        run_conversion(_unreachable, tmp_path, _ARTIFACTS, force=False, dry_run=dry_run)

    assert (tmp_path / "system").is_dir()
    assert (tmp_path / "config.json").is_file()


def test_force_clears_the_artifacts_but_not_other_files(tmp_path: Path) -> None:
    _write_outputs(tmp_path)
    (tmp_path / "notes.txt").write_text("mine", encoding="utf-8")
    left_at_start: list[str] = []

    def convert() -> ConversionReport:
        left_at_start.extend(sorted(p.name for p in tmp_path.iterdir()))
        return ConversionReport()

    run_conversion(convert, tmp_path, _ARTIFACTS, force=True, dry_run=False)

    assert left_at_start == ["notes.txt"]


def test_force_on_a_dry_run_clears_nothing(tmp_path: Path) -> None:
    _write_outputs(tmp_path)

    run_conversion(ConversionReport, tmp_path, _ARTIFACTS, force=True, dry_run=True)

    assert (tmp_path / "system").is_dir()
    assert (tmp_path / "config.json").is_file()


def test_a_failure_removes_the_directories_the_run_created(tmp_path: Path) -> None:
    dst = tmp_path / "cases" / "case"

    def convert() -> ConversionReport:
        _write_outputs(dst)
        raise RuntimeError("boom")

    with dx.collect(), pytest.raises(RuntimeError, match="boom"):
        run_conversion(convert, dst, _ARTIFACTS, force=False, dry_run=False)

    assert list(tmp_path.iterdir()) == []


def test_a_failure_keeps_a_dst_that_existed(tmp_path: Path) -> None:
    dst = tmp_path / "case"
    dst.mkdir()

    def convert() -> ConversionReport:
        _write_outputs(dst)
        raise RuntimeError("boom")

    with dx.collect(), pytest.raises(RuntimeError, match="boom"):
        run_conversion(convert, dst, _ARTIFACTS, force=False, dry_run=False)

    assert dst.is_dir()
    assert list(dst.iterdir()) == []


def test_a_failure_keeps_a_created_dst_holding_files_it_does_not_own(
    tmp_path: Path,
) -> None:
    dst = tmp_path / "case"

    def convert() -> ConversionReport:
        _write_outputs(dst)
        (dst / "notes.txt").write_text("mine", encoding="utf-8")
        raise RuntimeError("boom")

    with dx.collect(), pytest.raises(RuntimeError, match="boom"):
        run_conversion(convert, dst, _ARTIFACTS, force=False, dry_run=False)

    assert [p.name for p in dst.iterdir()] == ["notes.txt"]


def test_a_dry_run_creates_no_directory(tmp_path: Path) -> None:
    dst = tmp_path / "case"

    run_conversion(ConversionReport, dst, _ARTIFACTS, force=False, dry_run=True)

    assert not dst.exists()
