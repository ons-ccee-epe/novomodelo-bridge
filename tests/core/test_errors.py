"""Unit tests for the typed-failure model (``novomodelo_bridge.core.errors``)."""

from __future__ import annotations

from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.errors import (
    BridgeError,
    FieldParseError,
    NovomodeloOutputError,
    NovomodeloPartitionMissingError,
    SourceFileError,
    diagnostic_from_exception,
)


class TestExceptionHierarchy:
    def test_source_file_error_is_filenotfounderror(self) -> None:
        assert isinstance(SourceFileError("x"), FileNotFoundError)

    def test_source_file_error_stores_location_attrs(self) -> None:
        exc = SourceFileError("caso.dat not found", path="/case", field="caso.dat")
        assert exc.path == "/case"
        assert exc.field == "caso.dat"
        assert str(exc) == "caso.dat not found"

    def test_novomodelo_partition_missing_error_is_bridge_error(self) -> None:
        exc = NovomodeloPartitionMissingError("partition not found", path="/out/sim/x")
        assert isinstance(exc, BridgeError)
        assert exc.path == "/out/sim/x"
        assert str(exc) == "partition not found"


class TestDiagnosticFromException:
    def test_diagnostic_from_source_file_error_code_and_severity(self) -> None:
        exc = SourceFileError("caso.dat not found", path="/case")
        diag = diagnostic_from_exception(exc, context="Conversion")
        assert diag.code == "source-file-missing"
        assert diag.severity is Severity.ERROR
        assert diag.category == "Conversion failure"
        assert diag.title == "Conversion failed"
        assert diag.summary == "caso.dat not found"
        assert any("/case" in note for note in diag.notes)
        assert diag.remediation is not None

    def test_diagnostic_from_unknown_exception_falls_back(self) -> None:
        diag = diagnostic_from_exception(ValueError("boom"), context="Conversion")
        assert diag.code == "unexpected-error"
        assert diag.severity is Severity.ERROR
        assert diag.summary == "boom"
        assert diag.remediation is None

    def test_diagnostic_from_plain_bridge_error_uses_the_base_fallback(self) -> None:
        diag = diagnostic_from_exception(BridgeError("boom"), context="Conversion")
        assert diag.code == "conversion-error"
        assert diag.category == "Conversion failure"
        assert diag.severity is Severity.ERROR
        assert diag.remediation is None
        assert diag.summary == "boom"

    def test_diagnostic_notes_list_location_attrs(self) -> None:
        exc = FieldParseError(
            "bad value",
            path="/case/dger.dat",
            field="dger.dat line 96",
            row=12,
        )
        diag = diagnostic_from_exception(exc, context="Conversion")
        assert "file: /case/dger.dat" in diag.notes
        assert "field: dger.dat line 96" in diag.notes
        assert "row: 12" in diag.notes

    def test_diagnostic_notes_omit_absent_location_attrs(self) -> None:
        exc = FieldParseError("bad value", field="dger.dat line 96")
        diag = diagnostic_from_exception(exc, context="Conversion")
        assert diag.notes == ["field: dger.dat line 96"]
        assert not any(note.startswith("file:") for note in diag.notes)
        assert not any(note.startswith("row:") for note in diag.notes)

    def test_novomodelo_output_error_category(self) -> None:
        exc = NovomodeloOutputError("bad parquet", path="/out/bounds.parquet")
        diag = diagnostic_from_exception(exc, context="Comparison")
        assert diag.code == "novomodelo-output-unreadable"
        assert diag.category == "Comparison failure"
        assert "file: /out/bounds.parquet" in diag.notes

    def test_category_and_title_follow_the_failing_operation(self) -> None:
        exc = NovomodeloOutputError("no simulation output found")
        diag = diagnostic_from_exception(exc, context="Dashboard")
        assert diag.category == "Dashboard failure"
        assert diag.title == "Dashboard failed"

    def test_novomodelo_partition_missing_error_category(self) -> None:
        exc = NovomodeloPartitionMissingError(
            "Novomodelo output partition not found: /out/simulation/hydro_bus_generation. "
            "novomodelo >= 0.13.0",
            path="/out/simulation/hydro_bus_generation",
        )
        diag = diagnostic_from_exception(exc, context="Comparison")
        assert diag.code == "novomodelo-partition-missing"
        assert diag.category == "Comparison failure"
        assert "file: /out/simulation/hydro_bus_generation" in diag.notes


def test_no_source_string_carries_the_remediation_arrow() -> None:
    """Remediation hints are plain sentences: the console adds the ``→``
    marker, so a literal starting with it renders as ``→ →``."""
    import ast
    from pathlib import Path

    import novomodelo_bridge

    package = Path(novomodelo_bridge.__file__).parent
    offenders = [
        f"{path.relative_to(package)}:{node.lineno}"
        for path in sorted(package.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("→")
        and path.name != "console.py"
    ]
    assert offenders == []
