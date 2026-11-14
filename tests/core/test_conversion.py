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
from novomodelo_bridge.core.errors import EmissionCheckError

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

    report = run_conversion(convert, tmp_path, _ARTIFACTS, dry_run=False)

    assert [d.code for d in report.diagnostics] == ["early", "note"]
    assert report.warnings == ["early"]


def test_a_log_warning_is_not_a_finding(tmp_path: Path) -> None:
    def convert() -> ConversionReport:
        logging.getLogger("novomodelo_bridge.test").warning("log text")
        return ConversionReport()

    report = run_conversion(convert, tmp_path, _ARTIFACTS, dry_run=False)

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
        run_conversion(convert, tmp_path, _ARTIFACTS, dry_run=False)

    assert [d.code for d in outer] == ["early", "check-failed"]
    assert not (tmp_path / "system").exists()
    assert not (tmp_path / "config.json").exists()


def test_dry_run_failure_leaves_dst_untouched(tmp_path: Path) -> None:
    _write_outputs(tmp_path)

    def convert() -> ConversionReport:
        raise RuntimeError("boom")

    with dx.collect(), pytest.raises(RuntimeError, match="boom"):
        run_conversion(convert, tmp_path, _ARTIFACTS, dry_run=True)

    assert (tmp_path / "system").is_dir()
    assert (tmp_path / "config.json").is_file()
