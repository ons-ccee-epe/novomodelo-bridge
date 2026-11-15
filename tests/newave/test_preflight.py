"""Unit tests for the pure preflight validation engine (``novomodelo_bridge.newave.preflight``)."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.libs_index import UNREAD_LINES_CODE
from novomodelo_bridge.core.preflight import PreflightResult, PreflightVerdict
from novomodelo_bridge.newave import preflight
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.files import NewaveFiles
from novomodelo_bridge.newave.preflight import run_preflight
from novomodelo_bridge.newave.switches import DgerSwitches
from tests.conftest import make_nw_files

# Optional field names derived the same way the module does, so the "all present"
# fixture and the assertions stay in sync with the discovery dataclass.
_OPTIONAL_FIELDS = [
    f.name
    for f in fields(NewaveFiles)
    if f.name != "directory" and "None" in str(f.type)
]


def _all_optionals_present(tmp_path: Path) -> dict[str, Path | None]:
    """Every optional input resolved to a path under *tmp_path* (none absent).

    The electrical-constraint file, the one optional input whose contents
    preflight reads, is written empty.
    """
    (tmp_path / "restricao_eletrica.dat").touch()
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
        present["cvar"] = None
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

    def test_tab_in_a_fixed_column_file_yields_warnings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        files = make_nw_files(tmp_path, **_all_optionals_present(tmp_path))
        files.clast.write_text("    1 A\n\t2 B\n", encoding="utf-8")
        monkeypatch.setattr(preflight.NewaveFiles, "from_directory", lambda _src: files)

        result = run_preflight(tmp_path)

        assert result.verdict is PreflightVerdict.WARNINGS
        assert [d.code for d in result.diagnostics] == ["source-tab-character"]
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
        [advisory] = diagnostics
        assert advisory.severity is Severity.INFO
        assert advisory.code == "optional-file-absent"
        assert advisory.table is not None
        assert len(advisory.table.rows) == len(_OPTIONAL_FIELDS)
        assert all(
            check.detail == "absent (optional; conversion proceeds)" for check in checks
        )

    def test_optional_input_advisory_skips_present_optionals(
        self, tmp_path: Path
    ) -> None:
        files = make_nw_files(tmp_path, **_all_optionals_present(tmp_path))

        assert preflight.optional_input_advisory(files) == ([], [])


class TestSwitchAdvisory:
    @staticmethod
    def _switches(**values: int) -> DgerSwitches:
        dger = MagicMock()
        for field, value in values.items():
            setattr(dger, field, value)
        return DgerSwitches.from_dger(dger)

    def test_present_input_switched_off_yields_info_and_passing_check(
        self, tmp_path: Path
    ) -> None:
        files = make_nw_files(tmp_path, ghmin=tmp_path / "ghmin.dat")

        checks, diagnostics = preflight._switch_advisory(
            files, self._switches(considera_ghmin=0)
        )

        assert [c.label for c in checks] == ["dger.dat switches"]
        assert checks[0].passed and checks[0].detail == "1 present input switched off"
        assert [d.code for d in diagnostics] == ["dger-switch-off"]
        assert diagnostics[0].severity is Severity.INFO
        assert "ghmin.dat" in diagnostics[0].title

    def test_nothing_switched_off_yields_one_passing_check(
        self, tmp_path: Path
    ) -> None:
        files = make_nw_files(tmp_path, ghmin=tmp_path / "ghmin.dat")

        checks, diagnostics = preflight._switch_advisory(files, self._switches())

        assert diagnostics == []
        assert checks[0].passed
        assert checks[0].detail == "every present optional input is switched on"

    def test_unreadable_dger_is_a_failed_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(_case: NewaveCase) -> DgerSwitches:
            raise ValueError("bad dger")

        files = make_nw_files(tmp_path)
        monkeypatch.setattr(preflight.NewaveFiles, "from_directory", lambda _src: files)
        monkeypatch.setattr(NewaveCase, "switches", property(boom))

        result = run_preflight(tmp_path)

        [readable] = [c for c in result.checks if c.label == "dger.dat readable"]
        assert not readable.passed
        assert any(d.severity is Severity.ERROR for d in result.diagnostics)
        assert result.verdict is PreflightVerdict.WILL_NOT_CONVERT

    def test_run_preflight_stays_ok_with_a_switched_off_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An INFO advisory never turns the verdict."""
        present = _all_optionals_present(tmp_path)
        files = make_nw_files(tmp_path, **present)
        monkeypatch.setattr(preflight.NewaveFiles, "from_directory", lambda _src: files)
        switches = self._switches(agrupamento_livre=0)
        monkeypatch.setattr(NewaveCase, "switches", property(lambda _case: switches))

        result = run_preflight(tmp_path)

        assert result.verdict is PreflightVerdict.OK
        assert [d.code for d in result.diagnostics] == ["dger-switch-off"]


class TestElectricFileAdvisory:
    """``check newave`` reads the electrical-constraint file as ``convert
    newave`` does, when ``dger.dat`` lets the conversion read it."""

    def _run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        text: str,
        switches: DgerSwitches | None = None,
    ) -> PreflightResult:
        libs_file = tmp_path / "restricao-eletrica.csv"
        libs_file.write_text(text, encoding="utf-8")
        files = make_nw_files(tmp_path, restricao_eletrica=libs_file)
        monkeypatch.setattr(preflight.NewaveFiles, "from_directory", lambda _src: files)
        if switches is not None:
            monkeypatch.setattr(
                NewaveCase, "switches", property(lambda _case: switches)
            )
        return run_preflight(tmp_path)

    def test_lines_the_reader_keeps_as_text_are_warned_as_convert_does(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._run(
            tmp_path,
            monkeypatch,
            "& comentario\n\nre;2;1.0ger_usih(10)\n"
            "RESTRICAO-ELETRICA-FORMULA;3;ger_usih(10)\n",
        )

        [unread] = [d for d in result.diagnostics if d.code == UNREAD_LINES_CODE]
        assert unread.severity is Severity.WARNING
        assert unread.table is not None
        assert unread.table.rows == [
            ["re;2;1.0ger_usih(10)"],
            ["RESTRICAO-ELETRICA-FORMULA;3;ger_usih(10)"],
        ]
        assert result.verdict is PreflightVerdict.WARNINGS

    def test_a_file_whose_lines_are_all_read_raises_no_unread_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._run(tmp_path, monkeypatch, "& comentario\n")

        assert UNREAD_LINES_CODE not in {d.code for d in result.diagnostics}
        assert result.verdict is PreflightVerdict.OK

    def test_a_file_switched_off_in_dger_is_not_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._run(
            tmp_path,
            monkeypatch,
            "re;2;1.0ger_usih(10)\n",
            TestSwitchAdvisory._switches(restricoes_eletricas_especiais=0),
        )

        assert "dger-switch-off" in {d.code for d in result.diagnostics}
        assert UNREAD_LINES_CODE not in {d.code for d in result.diagnostics}
        assert result.verdict is PreflightVerdict.OK

    def test_an_unreadable_file_fails_the_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(_case: NewaveCase) -> None:
            raise ValueError("restriction 7 is malformed")

        monkeypatch.setattr(NewaveCase, "restricao_eletrica", property(boom))

        result = self._run(tmp_path, monkeypatch, "")

        [readable] = [
            c
            for c in result.checks
            if c.label == "LIBS electrical restrictions readable"
        ]
        assert readable.passed is False
        assert readable.detail == "restriction 7 is malformed"
        assert result.verdict is PreflightVerdict.WILL_NOT_CONVERT
