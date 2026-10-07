"""Tier-1 CLI-UX tests for ``dashboard``.

Locks in the ``dashboard`` migration from ``SimpleNamespace`` to the typed
``DashboardArgs``: the no-simulation-output early exit now routes through
``_fail`` (a ``--json`` error envelope instead of empty stdout),
the two human status lines are suppressed under ``--quiet`` while
the dashboard file is still written, and ``--no-color`` reaches every
console ``_run_dashboard`` builds. Imports no novomodelo; the dashboard build
itself is stubbed so no real case data is required.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import typing
import webbrowser
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console
from typer.testing import CliRunner, Result

from novomodelo_bridge.cli import app
from novomodelo_bridge.cli.args import DashboardArgs
from novomodelo_bridge.cli.dashboard import _run_dashboard


def _invoke(argv: list[str]) -> Result:
    return CliRunner().invoke(app, argv)


def _run_cli_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the novomodelo-bridge entry point as a real subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "novomodelo_bridge.cli", *args],
        capture_output=True,
        text=True,
    )


def _make_case_dir(tmp_path: Path) -> Path:
    """Create a case dir whose ``output/simulation`` path exists."""
    case_dir = tmp_path / "case"
    (case_dir / "output" / "simulation").mkdir(parents=True)
    return case_dir


def _stub_build_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``build_dashboard`` to write a tiny file at the output path.

    ``_run_dashboard`` imports ``build_dashboard`` lazily inside its own body,
    so the patch target is the defining module, not ``novomodelo_bridge.cli.dashboard``.
    """

    def _fake_build(_case_dir: Path, output_path: Path) -> None:
        output_path.write_text("x", encoding="utf-8")

    monkeypatch.setattr("novomodelo_bridge.dashboard.build_dashboard", _fake_build)


def _spy_print_status(monkeypatch: pytest.MonkeyPatch) -> list[Console]:
    """Patch ``print_status`` to record the consoles it renders through, then
    delegate to the real implementation so the status lines still print.

    ``CliRunner`` never presents a TTY, so an ANSI-absence assertion on the
    captured output cannot distinguish ``--no-color`` actually threading from
    Rich's own non-TTY auto-detection; the returned list lets a test assert
    directly on the ``Console.no_color`` the CLI built for each status line.
    """
    import importlib

    from novomodelo_bridge.cli.dashboard import print_status as original

    captured: list[Console] = []

    def _spy(*args: object, **kwargs: object) -> None:
        console = kwargs["console"]
        assert isinstance(console, Console)
        captured.append(console)
        original(*args, **kwargs)

    # `print_status` is patched on its DEFINING module (mock discipline): the
    # handler resolves the name off `cli.dashboard`'s own module globals, so a
    # spy set anywhere else (e.g. `cli.app`, which only re-imports the handler)
    # never intercepts the call.
    cli_module = importlib.import_module("novomodelo_bridge.cli.dashboard")
    monkeypatch.setattr(cli_module, "print_status", _spy)
    return captured


class TestDashboardHandlerSignature:
    """``_run_dashboard`` is typed on ``DashboardArgs``, not ``SimpleNamespace``."""

    def test_args_parameter_is_annotated_dashboard_args(self) -> None:
        hints = typing.get_type_hints(_run_dashboard)
        assert hints["args"] is DashboardArgs

    def test_does_not_reference_simple_namespace(self) -> None:
        assert "SimpleNamespace" not in inspect.getsource(_run_dashboard)


class TestDashboardNoSimulationJson:
    def test_json_emits_one_error_envelope_and_exits_1(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case"
        case_dir.mkdir()

        result = _run_cli_subprocess("dashboard", str(case_dir), "--json")

        assert result.returncode == 1
        document = json.loads(result.stdout)
        assert list(document.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert document["command"] == "dashboard"
        assert document["status"] == "error"
        assert len(document["diagnostics"]) == 1
        assert "no simulation output found" in document["diagnostics"][0]["summary"]


class TestDashboardNoSimulationPlain:
    def test_without_json_stdout_stays_empty_and_exits_1(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case"
        case_dir.mkdir()

        result = _run_cli_subprocess("dashboard", str(case_dir))

        assert result.returncode == 1
        assert result.stdout == ""
        assert "no simulation output found" in result.stderr
        assert "Dashboard notes: 1 error(s)" in result.stderr
        assert "Conversion notes" not in result.stderr
        assert "Dashboard failure" in result.stderr
        assert "Comparison failure" not in result.stderr
        assert "→ → " not in result.stderr


class TestDashboardQuiet:
    def test_quiet_suppresses_status_lines_but_still_writes_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = _make_case_dir(tmp_path)
        _stub_build_dashboard(monkeypatch)
        output_path = case_dir / "dashboard.html"

        result = _invoke(["dashboard", str(case_dir), "--quiet"])

        assert result.exit_code == 0
        assert "Building dashboard from" not in result.stdout
        assert "Dashboard written to" not in result.stdout
        assert output_path.read_text(encoding="utf-8") == "x"


class TestDashboardNoColor:
    def test_no_color_yields_ansi_free_status_lines(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        dumb_terminal: None,
    ) -> None:
        """``--no-color`` threads through the status-line renderer.

        Under ``CliRunner`` the captured stream is never a TTY, so Rich
        withholds ANSI regardless of ``--no-color`` — an ANSI-absence
        assertion alone is tautological. Spy on the consoles ``print_status``
        actually renders through and assert they were built with
        ``no_color=True``, so the test genuinely fails if the flag stops
        threading.
        """
        case_dir = _make_case_dir(tmp_path)
        _stub_build_dashboard(monkeypatch)
        captured = _spy_print_status(monkeypatch)

        result = _invoke(["dashboard", str(case_dir), "--no-color"])

        assert result.exit_code == 0
        assert "Building dashboard from" in result.stdout
        assert "Dashboard written to" in result.stdout
        assert "\x1b[" not in result.stdout
        assert "\x1b[" not in result.stderr
        # Hermetic guard: both status-line consoles were actually built with
        # no_color=True, not merely non-TTY-quiet (see _spy_print_status).
        assert len(captured) == 2
        assert all(console.no_color is True for console in captured)


class TestDashboardOpen:
    """``dashboard --open`` launches the written HTML in a browser.

    Runs ``dashboard`` in-process via ``cli.main`` with the real dashboard build
    stubbed (``novomodelo_bridge.dashboard.build_dashboard``) so it only writes a tiny
    file at the output path — enough for the ``output_path.stat()`` size line to
    succeed without building a real dashboard. ``webbrowser.open`` is patched at
    its ``novomodelo_bridge.cli.dashboard`` import site so no actual browser is launched.
    """

    def _invoke_main(
        self,
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[int, str, str]:
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
    def _make_case_dir(tmp_path: Path) -> Path:
        """Create a case dir whose ``output/simulation`` path exists."""
        case_dir = tmp_path / "case"
        (case_dir / "output" / "simulation").mkdir(parents=True)
        return case_dir

    @staticmethod
    def _stub_build_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub ``build_dashboard`` to write a tiny file at the output path.

        The real build is skipped; a 1-byte file keeps the ``stat()`` size line
        in ``_run_dashboard`` from raising.
        """

        def _fake_build(_case_dir: Path, output_path: Path) -> None:
            output_path.write_text("x", encoding="utf-8")

        monkeypatch.setattr("novomodelo_bridge.dashboard.build_dashboard", _fake_build)

    def test_dashboard_open_calls_webbrowser_with_file_uri(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)
        expected_uri = (case_dir / "dashboard.html").resolve().as_uri()

        with patch(
            "novomodelo_bridge.cli.dashboard.webbrowser.open", return_value=True
        ) as mock_open:
            exit_code, _stdout, _stderr = self._invoke_main(
                ["dashboard", str(case_dir), "--open"], monkeypatch
            )

        assert exit_code == 0
        mock_open.assert_called_once_with(expected_uri)

    def test_dashboard_without_open_does_not_call_webbrowser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)

        with patch("novomodelo_bridge.cli.dashboard.webbrowser.open") as mock_open:
            exit_code, _stdout, _stderr = self._invoke_main(
                ["dashboard", str(case_dir)], monkeypatch
            )

        assert exit_code == 0
        mock_open.assert_not_called()

    def test_dashboard_open_swallows_webbrowser_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)

        with patch(
            "novomodelo_bridge.cli.dashboard.webbrowser.open",
            side_effect=webbrowser.Error("no browser"),
        ):
            exit_code, _stdout, stderr = self._invoke_main(
                ["dashboard", str(case_dir), "--open"], monkeypatch
            )

        assert exit_code == 0
        assert "could not open a browser" in stderr

    def test_dashboard_open_swallows_false_return(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)

        with patch(
            "novomodelo_bridge.cli.dashboard.webbrowser.open", return_value=False
        ):
            exit_code, _stdout, stderr = self._invoke_main(
                ["dashboard", str(case_dir), "--open"], monkeypatch
            )

        assert exit_code == 0
        assert "could not open a browser" in stderr


class TestDashboardJson:
    """``dashboard --json`` emits the unified verdict envelope.

    Reuses ``TestDashboardOpen``'s in-process driver and build stub: the real
    dashboard build is replaced by a stub that writes a tiny file at the output
    path (so ``output_path.stat()`` succeeds), the CLI runs via ``cli.main`` with
    ``sys.stdout``/``sys.stderr`` captured, and the stdout is parsed as JSON. The
    file is STILL built under ``--json``; only the two Rich status lines are
    suppressed.
    """

    def _invoke_main(
        self,
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[int, str, str]:
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
    def _make_case_dir(tmp_path: Path) -> Path:
        """Create a case dir whose ``output/simulation`` path exists."""
        case_dir = tmp_path / "case"
        (case_dir / "output" / "simulation").mkdir(parents=True)
        return case_dir

    @staticmethod
    def _stub_build_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub ``build_dashboard`` to write a tiny file at the output path."""

        def _fake_build(_case_dir: Path, output_path: Path) -> None:
            output_path.write_text("x", encoding="utf-8")

        monkeypatch.setattr("novomodelo_bridge.dashboard.build_dashboard", _fake_build)

    def test_dashboard_json_success_shape_and_exit_0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)

        exit_code, stdout, _stderr = self._invoke_main(
            ["dashboard", str(case_dir), "--json"], monkeypatch
        )

        assert exit_code == 0
        document = json.loads(stdout)
        assert list(document.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert document["command"] == "dashboard"
        assert document["status"] == "ok"
        assert document["summary"]["output"].endswith("dashboard.html")
        assert isinstance(document["summary"]["size_kb"], (int, float))
        assert document["diagnostics"] == []

    def test_dashboard_json_suppresses_status_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)

        exit_code, stdout, _stderr = self._invoke_main(
            ["dashboard", str(case_dir), "--json"], monkeypatch
        )

        assert exit_code == 0
        assert "Building dashboard from" not in stdout
        assert "Dashboard written to" not in stdout
        # stdout parses as exactly one JSON object.
        assert json.loads(stdout)["command"] == "dashboard"

    def test_dashboard_json_no_simulation_exits_1_emits_error_envelope(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A case dir WITHOUT output/simulation fires the exit-1 guard FIRST, so no
        # build happens; the guard now routes through ``_fail``, so stdout carries
        # one error envelope instead of staying empty.
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        self._stub_build_dashboard(monkeypatch)

        exit_code, stdout, _stderr = self._invoke_main(
            ["dashboard", str(case_dir), "--json"], monkeypatch
        )

        assert exit_code == 1
        document = json.loads(stdout)
        assert document["command"] == "dashboard"
        assert document["status"] == "error"
        assert "no simulation output found" in document["diagnostics"][0]["summary"]

    def test_dashboard_json_open_advisory_stays_on_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)

        with patch(
            "novomodelo_bridge.cli.dashboard.webbrowser.open", return_value=False
        ):
            exit_code, stdout, stderr = self._invoke_main(
                ["dashboard", str(case_dir), "--json", "--open"], monkeypatch
            )

        assert exit_code == 0
        # stdout is exactly one verdict — no browser advisory leaked onto it.
        document = json.loads(stdout)
        assert document["command"] == "dashboard"
        assert "could not open a browser" not in stdout
        assert "could not open a browser" in stderr

    def test_dashboard_json_custom_output_path_in_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)
        out_path = tmp_path / "OUT.html"

        exit_code, stdout, _stderr = self._invoke_main(
            ["dashboard", str(case_dir), "-o", str(out_path), "--json"], monkeypatch
        )

        assert exit_code == 0
        document = json.loads(stdout)
        assert document["summary"]["output"] == str(out_path)
