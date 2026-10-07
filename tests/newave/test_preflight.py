"""Unit tests for the pure preflight validation engine (``novomodelo_bridge.newave.preflight``)."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.preflight import PreflightVerdict
from novomodelo_bridge.newave import preflight
from novomodelo_bridge.newave.files import NewaveFiles
from novomodelo_bridge.newave.preflight import run_preflight
from tests.conftest import make_nw_files

# Optional field names derived the same way the module does, so the "all present"
# fixture and the assertions stay in sync with the discovery dataclass.
_OPTIONAL_FIELDS = [
    f.name
    for f in fields(NewaveFiles)
    if f.name != "directory" and "None" in str(f.type)
]


def _all_optionals_present(tmp_path: Path) -> dict[str, Path]:
    """Every optional input resolved to a path under *tmp_path* (none absent)."""
    return {name: tmp_path / f"{name}.dat" for name in _OPTIONAL_FIELDS}


class TestRunPreflight:
    def test_missing_caso_yields_will_not_convert(self, tmp_path: Path) -> None:
        # An empty directory has no caso.dat: real discovery raises a
        # SourceFileError before any binary is parsed.
        result = run_preflight(tmp_path)

        assert result.verdict is PreflightVerdict.WILL_NOT_CONVERT
        error_codes = [
            diag.code for diag in result.diagnostics if diag.severity is Severity.ERROR
        ]
        assert "source-file-missing" in error_codes
        assert any(not check.passed for check in result.checks)

    def test_absent_optional_yields_ok_with_info_advisory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # All optionals present except cvar (left None): discovery succeeds, the
        # single absent optional is advisory-only and does not block or warn.
        present = _all_optionals_present(tmp_path)
        present["cvar"] = None  # type: ignore[assignment]
        files = make_nw_files(tmp_path, **present)
        monkeypatch.setattr(preflight.NewaveFiles, "from_directory", lambda _src: files)

        result = run_preflight(tmp_path)

        assert result.verdict is PreflightVerdict.OK
        info_codes = [
            diag.code for diag in result.diagnostics if diag.severity is Severity.INFO
        ]
        warning_codes = [
            diag.code
            for diag in result.diagnostics
            if diag.severity is Severity.WARNING
        ]
        assert "optional-file-absent" in info_codes
        assert "optional-file-absent" not in warning_codes
        assert all(check.passed for check in result.checks)

    def test_complete_case_yields_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        files = make_nw_files(tmp_path, **_all_optionals_present(tmp_path))
        monkeypatch.setattr(preflight.NewaveFiles, "from_directory", lambda _src: files)

        result = run_preflight(tmp_path)

        assert result.verdict is PreflightVerdict.OK
        assert result.diagnostics == []
        assert all(check.passed for check in result.checks)

    def test_preflight_writes_no_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        files = make_nw_files(tmp_path, **_all_optionals_present(tmp_path))
        monkeypatch.setattr(preflight.NewaveFiles, "from_directory", lambda _src: files)
        before = sorted(p.name for p in tmp_path.iterdir())

        run_preflight(tmp_path)

        after = sorted(p.name for p in tmp_path.iterdir())
        assert after == before

    def test_missing_caso_writes_no_files(self, tmp_path: Path) -> None:
        before = sorted(p.name for p in tmp_path.iterdir())

        run_preflight(tmp_path)

        after = sorted(p.name for p in tmp_path.iterdir())
        assert after == before


class TestOptionalInputAdvisory:
    def test_optional_input_advisory_reports_every_absent_optional(
        self, tmp_path: Path
    ) -> None:
        files = make_nw_files(tmp_path)  # every optional field left None

        checks, diagnostics = preflight.optional_input_advisory(files)

        assert len(checks) == len(_OPTIONAL_FIELDS)
        assert len(diagnostics) == len(_OPTIONAL_FIELDS)
        assert all(diag.severity is Severity.INFO for diag in diagnostics)
        assert all(diag.code == "optional-file-absent" for diag in diagnostics)
        assert all(
            check.detail == "absent (optional; conversion proceeds)" for check in checks
        )

    def test_optional_input_advisory_skips_present_optionals(
        self, tmp_path: Path
    ) -> None:
        files = make_nw_files(tmp_path, **_all_optionals_present(tmp_path))

        assert preflight.optional_input_advisory(files) == ([], [])


class TestSwitchAdvisory:
    def _switches(self, **values: int):
        from unittest.mock import MagicMock

        from novomodelo_bridge.newave.switches import DgerSwitches

        dger = MagicMock()
        for field, value in values.items():
            setattr(dger, field, value)
        return DgerSwitches.from_dger(dger)

    def test_present_input_switched_off_yields_info_and_passing_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        files = make_nw_files(tmp_path, ghmin=tmp_path / "ghmin.dat")
        monkeypatch.setattr(
            preflight,
            "_read_switches",
            lambda _files: self._switches(considera_ghmin=0),
        )

        checks, diagnostics = preflight._switch_advisory(files)

        assert [c.label for c in checks] == ["dger.dat switches"]
        assert (
            checks[0].passed and checks[0].detail == "1 present input(s) switched off"
        )
        assert [d.code for d in diagnostics] == ["dger-switch-off"]
        assert diagnostics[0].severity is Severity.INFO
        assert "ghmin.dat" in diagnostics[0].title

    def test_nothing_switched_off_yields_one_passing_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        files = make_nw_files(tmp_path, ghmin=tmp_path / "ghmin.dat")
        monkeypatch.setattr(
            preflight, "_read_switches", lambda _files: self._switches()
        )

        checks, diagnostics = preflight._switch_advisory(files)

        assert diagnostics == []
        assert checks[0].passed
        assert checks[0].detail == "every present optional input is switched on"

    def test_unreadable_dger_is_a_failed_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(_files):
            raise ValueError("bad dger")

        monkeypatch.setattr(preflight, "_read_switches", boom)

        checks, diagnostics = preflight._switch_advisory(make_nw_files(tmp_path))

        assert [c.label for c in checks] == ["dger.dat readable"]
        assert not checks[0].passed
        assert diagnostics and diagnostics[0].severity is Severity.ERROR

    def test_run_preflight_stays_ok_with_a_switched_off_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An INFO advisory never turns the verdict."""
        present = _all_optionals_present(tmp_path)
        files = make_nw_files(tmp_path, **present)
        monkeypatch.setattr(preflight.NewaveFiles, "from_directory", lambda _src: files)
        monkeypatch.setattr(
            preflight,
            "_read_switches",
            lambda _files: self._switches(agrupamento_livre=0),
        )

        result = run_preflight(tmp_path)

        assert result.verdict is PreflightVerdict.OK
        assert [d.code for d in result.diagnostics] == ["dger-switch-off"]
