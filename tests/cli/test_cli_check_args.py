"""Unit tests for the check-family CLI args migration.

Tier 1 — pure Python, imports no novomodelo. Locks in the migration of
``_run_check``/``_run_decomp_check`` from ``SimpleNamespace`` to the typed
``CheckArgs``, and the ``--no-color``-aware console split between the
checklist (stdout) and diagnostics (stderr) it now threads through
:func:`novomodelo_bridge.ui.console.render_checklist`.
"""

from __future__ import annotations

import inspect
import json
import sys
import typing
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner, Result

from novomodelo_bridge.cli import app
from novomodelo_bridge.cli.args import CheckArgs
from novomodelo_bridge.cli.check import _run_check, _run_decomp_check
from novomodelo_bridge.core.diagnostics import Diagnostic, Severity
from novomodelo_bridge.core.preflight import (
    CheckItem,
    PreflightResult,
    PreflightVerdict,
)
from tests.cli.conftest import _run_cli_subprocess
from tests.conftest import _make_fake_newave_dir


def _invoke(argv: list[str]) -> Result:
    return CliRunner().invoke(app, argv)


def _spy_render_checklist(monkeypatch: pytest.MonkeyPatch) -> dict[str, Console]:
    """Patch ``render_checklist`` to record the consoles it renders through,
    then delegate to the real implementation so rendering still happens.

    ``CliRunner`` never presents a TTY, so an ANSI-absence assertion on the
    captured output cannot distinguish ``--no-color`` actually threading from
    Rich's own non-TTY auto-detection; the returned dict lets a test assert
    directly on the ``Console.no_color`` the CLI built.
    """
    import importlib

    from novomodelo_bridge.cli.check import render_checklist as original

    captured: dict[str, Console] = {}

    def _spy(*args: object, **kwargs: object) -> None:
        console = kwargs["console"]
        diagnostics_console = kwargs["diagnostics_console"]
        assert isinstance(console, Console)
        assert isinstance(diagnostics_console, Console)
        captured["console"] = console
        captured["diagnostics_console"] = diagnostics_console
        original(*args, **kwargs)

    # `render_checklist` is patched on its DEFINING module (mock discipline):
    # the handler resolves the name off `cli.check`'s own module globals, so a
    # spy set anywhere else (e.g. `cli.app`, which only re-imports the handler)
    # never intercepts the call.
    cli_module = importlib.import_module("novomodelo_bridge.cli.check")
    monkeypatch.setattr(cli_module, "render_checklist", _spy)
    return captured


def _warnings_result() -> PreflightResult:
    """A WARNINGS verdict carrying one diagnostic, so both the checklist and
    the diagnostics panel render — exercising both consoles at once."""
    return PreflightResult(
        verdict=PreflightVerdict.WARNINGS,
        diagnostics=[
            Diagnostic(
                code="optional-file-absent",
                severity=Severity.WARNING,
                category="Preflight",
                title="Optional input absent",
                summary="Optional input 'modif' was not found.",
            )
        ],
        checks=[
            CheckItem(label="Required files present", passed=True),
            CheckItem(
                label="Optional: modif",
                passed=True,
                detail="absent (will use defaults)",
            ),
        ],
    )


class TestCheckHandlerSignatures:
    """Both check handlers are typed on ``CheckArgs``, not ``SimpleNamespace``."""

    @pytest.mark.parametrize("func", [_run_check, _run_decomp_check])
    def test_args_parameter_is_annotated_check_args(
        self, func: typing.Callable[..., None]
    ) -> None:
        hints = typing.get_type_hints(func)
        assert hints["args"] is CheckArgs

    @pytest.mark.parametrize("func", [_run_check, _run_decomp_check])
    def test_does_not_reference_simple_namespace(
        self, func: typing.Callable[..., None]
    ) -> None:
        assert "SimpleNamespace" not in inspect.getsource(func)


class TestCheckNewaveJsonShape:
    def test_json_emits_expected_command_status_and_checks_shape(
        self, tmp_path: Path
    ) -> None:
        with patch(
            "novomodelo_bridge.newave.preflight.run_preflight",
            return_value=_warnings_result(),
        ):
            result = _invoke(["check", "newave", str(tmp_path / "case"), "--json"])

        assert result.exit_code == 1  # WARNINGS -> 1
        document = json.loads(result.stdout)
        assert document["command"] == "check newave"
        assert document["status"] == "warnings"
        assert document["summary"]["checks"] == [
            {"label": "Required files present", "passed": True, "detail": None},
            {
                "label": "Optional: modif",
                "passed": True,
                "detail": "absent (will use defaults)",
            },
        ]
        assert document["diagnostics"][0]["code"] == "optional-file-absent"
        # No Rich checklist leaked onto either stream under --json.
        assert "Ready with warnings" not in result.stdout
        assert result.stderr == ""


class TestCheckNewaveSwitchAdvisoryJson:
    def test_switch_off_advisory_reaches_the_json_envelope_as_info(
        self, tmp_path: Path
    ) -> None:
        from novomodelo_bridge.newave.switches import (
            DgerSwitches,
            switch_off_diagnostic,
        )

        dger = MagicMock()
        dger.considera_ghmin = 0
        switch = DgerSwitches.from_dger(dger).ghmin
        result_obj = PreflightResult(
            verdict=PreflightVerdict.OK,
            diagnostics=[switch_off_diagnostic(switch)],
            checks=[
                CheckItem(label="Required files present", passed=True),
                CheckItem(
                    label="dger.dat switches",
                    passed=True,
                    detail="1 present input(s) switched off",
                ),
            ],
        )
        with patch(
            "novomodelo_bridge.newave.preflight.run_preflight", return_value=result_obj
        ):
            result = _invoke(["check", "newave", str(tmp_path / "case"), "--json"])

        assert result.exit_code == 0
        document = json.loads(result.stdout)
        assert document["status"] == "ok"
        assert document["summary"]["checks"][1]["label"] == "dger.dat switches"
        diag = document["diagnostics"][0]
        assert diag["code"] == "dger-switch-off"
        assert diag["severity"] == "info"
        assert "CONSIDERA GHMIN = 0" in diag["title"]


class TestCheckNewaveNoColor:
    def test_no_color_produces_ansi_free_output_on_both_streams(
        self, tmp_path: Path, dumb_terminal: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _spy_render_checklist(monkeypatch)

        with patch(
            "novomodelo_bridge.newave.preflight.run_preflight",
            return_value=_warnings_result(),
        ):
            result = _invoke(["check", "newave", str(tmp_path / "case"), "--no-color"])

        assert (
            result.exit_code == 1
        )  # the verdict's mapped code, unaffected by --no-color
        assert "\x1b[" not in result.stdout
        assert "\x1b[" not in result.stderr
        # Rendering still happened on both streams (headline on stdout,
        # diagnostics delegated to stderr), just without ANSI escapes.
        assert "Ready with warnings" in result.stdout
        assert "Optional input absent" in result.stderr
        # Hermetic guard: the streams were actually built with no_color=True,
        # not merely non-TTY-quiet (see _spy_render_checklist).
        assert captured["console"].no_color is True
        assert captured["diagnostics_console"].no_color is True


class TestCheckDecompJsonShape:
    """Mirrors TestCheckNewaveJsonShape for ``_run_decomp_check`` — the two
    handlers build the identical verdict envelope off the same PreflightResult
    shape, so the twin-track contract (``.claude/rules/bridge.md`` §1) applies."""

    def test_json_emits_expected_command_status_and_checks_shape(
        self, tmp_path: Path
    ) -> None:
        with patch(
            "novomodelo_bridge.decomp.preflight.run_decomp_preflight",
            return_value=_warnings_result(),
        ):
            result = _invoke(["check", "decomp", str(tmp_path / "case"), "--json"])

        assert result.exit_code == 1  # WARNINGS -> 1
        document = json.loads(result.stdout)
        assert document["command"] == "check decomp"
        assert document["status"] == "warnings"
        assert document["summary"]["checks"] == [
            {"label": "Required files present", "passed": True, "detail": None},
            {
                "label": "Optional: modif",
                "passed": True,
                "detail": "absent (will use defaults)",
            },
        ]
        assert document["diagnostics"][0]["code"] == "optional-file-absent"
        # No Rich checklist leaked onto either stream under --json.
        assert "Ready with warnings" not in result.stdout
        assert result.stderr == ""


class TestCheckDecompNoColor:
    """Mirrors TestCheckNewaveNoColor for ``_run_decomp_check``."""

    def test_no_color_produces_ansi_free_output_on_both_streams(
        self, tmp_path: Path, dumb_terminal: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _spy_render_checklist(monkeypatch)

        with patch(
            "novomodelo_bridge.decomp.preflight.run_decomp_preflight",
            return_value=_warnings_result(),
        ):
            result = _invoke(["check", "decomp", str(tmp_path / "case"), "--no-color"])

        assert (
            result.exit_code == 1
        )  # the verdict's mapped code, unaffected by --no-color
        assert "\x1b[" not in result.stdout
        assert "\x1b[" not in result.stderr
        # Rendering still happened on both streams (headline on stdout,
        # diagnostics delegated to stderr), just without ANSI escapes.
        assert "Ready with warnings" in result.stdout
        assert "Optional input absent" in result.stderr
        # Hermetic guard: the streams were actually built with no_color=True,
        # not merely non-TTY-quiet (see _spy_render_checklist).
        assert captured["console"].no_color is True
        assert captured["diagnostics_console"].no_color is True


class TestCheckCommand:
    """The ``check newave`` preflight command (exit 0/1/2 + --json)."""

    def _invoke_main(
        self,
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[int, str, str]:
        """Run cli.main() in-process, capturing stdout/stderr and exit code."""
        import io

        from novomodelo_bridge import cli

        monkeypatch.setattr(sys, "argv", ["novomodelo-bridge", *argv])

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        exit_code = 0

        with patch("sys.stdout", stdout_buf), patch("sys.stderr", stderr_buf):
            try:
                cli.main()
            except SystemExit as exc:
                exit_code = int(exc.code) if exc.code is not None else 0

        return exit_code, stdout_buf.getvalue(), stderr_buf.getvalue()

    @staticmethod
    def _result(verdict: object) -> object:
        """Build a small ``PreflightResult`` with the given verdict.

        Warnings carry one ``WARNING`` diagnostic so the JSON/headline paths see a
        realistic payload; the verdict itself is taken verbatim by the handler and
        renderer (never recomputed from the checks/diagnostics here).
        """
        from novomodelo_bridge.core.diagnostics import Diagnostic, Severity
        from novomodelo_bridge.core.preflight import (
            CheckItem,
            PreflightResult,
            PreflightVerdict,
        )

        if verdict is PreflightVerdict.WILL_NOT_CONVERT:
            return PreflightResult(
                verdict=PreflightVerdict.WILL_NOT_CONVERT,
                diagnostics=[
                    Diagnostic(
                        code="source-file-error",
                        severity=Severity.ERROR,
                        category="Preflight",
                        title="Required input missing",
                        summary="caso.dat not found",
                    )
                ],
                checks=[
                    CheckItem(
                        label="File discovery (caso.dat → arquivos.dat)",
                        passed=False,
                        detail="caso.dat not found",
                    )
                ],
            )
        if verdict is PreflightVerdict.WARNINGS:
            return PreflightResult(
                verdict=PreflightVerdict.WARNINGS,
                diagnostics=[
                    Diagnostic(
                        code="optional-file-absent",
                        severity=Severity.WARNING,
                        category="Preflight",
                        title="Optional input absent",
                        summary="Optional input 'modif' was not found.",
                    )
                ],
                checks=[
                    CheckItem(label="Required files present", passed=True),
                    CheckItem(
                        label="Optional: modif",
                        passed=True,
                        detail="absent (will use defaults)",
                    ),
                ],
            )
        return PreflightResult(
            verdict=PreflightVerdict.OK,
            diagnostics=[],
            checks=[CheckItem(label="Required files present", passed=True)],
        )

    # -- Unit tests ---------------------------------------------------------

    def test_check_verdict_shape(self) -> None:
        """The check ``summary`` helper feeds the unified envelope (checks nested)."""
        from novomodelo_bridge.cli.verdict import build_verdict, check_summary
        from novomodelo_bridge.core.preflight import PreflightVerdict

        result = self._result(PreflightVerdict.WILL_NOT_CONVERT)
        summary = check_summary(
            [
                {"label": c.label, "passed": c.passed, "detail": c.detail}
                for c in result.checks
            ]
        )
        doc = build_verdict(
            "check newave", result.verdict.value, summary, result.diagnostics
        )

        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["schema_version"] == 1
        assert doc["command"] == "check newave"
        assert doc["status"] == "will-not-convert"
        # The checklist moves UNDER summary.
        assert "checks" not in doc
        assert doc["summary"]["checks"] == [  # type: ignore[index]
            {
                "label": "File discovery (caso.dat → arquivos.dat)",
                "passed": False,
                "detail": "caso.dat not found",
            }
        ]
        assert doc["diagnostics"][0]["severity"] == "error"  # type: ignore[index]

    def test_verdict_to_exit_code_mapping(self) -> None:
        """The 0/1/2 mapping is exactly OK/WARNINGS/WILL_NOT_CONVERT (2 = severe)."""
        from novomodelo_bridge.cli.check import _VERDICT_EXIT_CODE
        from novomodelo_bridge.core.preflight import PreflightVerdict

        assert _VERDICT_EXIT_CODE == {
            PreflightVerdict.OK: 0,
            PreflightVerdict.WARNINGS: 1,
            PreflightVerdict.WILL_NOT_CONVERT: 2,
        }

    # -- Integration tests (in-process, patched run_preflight) --------------

    def test_check_ok_exits_0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from novomodelo_bridge.core.preflight import PreflightVerdict

        result = self._result(PreflightVerdict.OK)
        with patch(
            "novomodelo_bridge.newave.preflight.run_preflight", return_value=result
        ):
            code, stdout, _ = self._invoke_main(
                ["check", "newave", str(tmp_path / "case")],
                monkeypatch,
            )

        assert code == 0
        assert "✓ Ready to convert" in stdout

    def test_check_warnings_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from novomodelo_bridge.core.preflight import PreflightVerdict

        result = self._result(PreflightVerdict.WARNINGS)
        with patch(
            "novomodelo_bridge.newave.preflight.run_preflight", return_value=result
        ):
            code, stdout, _ = self._invoke_main(
                ["check", "newave", str(tmp_path / "case")],
                monkeypatch,
            )

        assert code == 1
        assert "Ready with warnings" in stdout

    def test_check_will_not_convert_exits_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from novomodelo_bridge.core.preflight import PreflightVerdict

        result = self._result(PreflightVerdict.WILL_NOT_CONVERT)
        with patch(
            "novomodelo_bridge.newave.preflight.run_preflight", return_value=result
        ):
            code, stdout, _ = self._invoke_main(
                ["check", "newave", str(tmp_path / "case")],
                monkeypatch,
            )

        assert code == 2
        assert "✖ Will not convert" in stdout

    def test_check_json_emits_stdout_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--json`` on a WILL_NOT_CONVERT result emits JSON to stdout; exit 2."""
        from novomodelo_bridge.core.preflight import PreflightVerdict

        result = self._result(PreflightVerdict.WILL_NOT_CONVERT)
        with patch(
            "novomodelo_bridge.newave.preflight.run_preflight", return_value=result
        ):
            code, stdout, stderr = self._invoke_main(
                ["check", "newave", str(tmp_path / "case"), "--json"],
                monkeypatch,
            )

        assert code == 2
        doc = json.loads(stdout)
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["schema_version"] == 1
        assert doc["command"] == "check newave"
        assert doc["status"] == "will-not-convert"
        # The checklist lives under summary now, not at the top level.
        assert "checks" not in doc
        assert doc["summary"]["checks"][0]["passed"] is False
        # No Rich checklist leaked onto either stream.
        assert "✖ Will not convert" not in stdout
        assert stderr == ""

    def test_check_writes_no_files_under_src(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``check`` must not write anything under the source directory."""
        from novomodelo_bridge.core.preflight import PreflightVerdict

        src = _make_fake_newave_dir(tmp_path)
        before = sorted(p.name for p in src.iterdir())

        result = self._result(PreflightVerdict.OK)
        with patch(
            "novomodelo_bridge.newave.preflight.run_preflight", return_value=result
        ):
            code, _, _ = self._invoke_main(
                ["check", "newave", str(src)],
                monkeypatch,
            )

        assert code == 0
        assert sorted(p.name for p in src.iterdir()) == before

    # -- E2E test (real discovery failure via subprocess) -------------------

    def test_check_missing_caso_subprocess_exits_2(self, tmp_path: Path) -> None:
        """A real discovery failure (no caso.dat) exits 2 with the ✖ headline."""
        result = _run_cli_subprocess("check", "newave", str(tmp_path / "nonexistent"))

        assert result.returncode == 2
        combined = result.stdout + result.stderr
        assert "✖ Will not convert" in combined
