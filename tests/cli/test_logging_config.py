"""Tests for cli/logging_config.py."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import _make_fake_newave_dir


class TestVerbosityAndLogFile:
    """Graduated ``-v/-vv`` verbosity and the shared ``--log-file``.

    Drives ``cli.main`` in-process (NOT the Typer ``CliRunner``) so the ``main``
    ``finally`` teardown that removes the ``--log-file`` ``FileHandler`` actually
    runs, and so the progress gate sees a real (patchable) ``sys.stderr.isatty``.
    """

    @staticmethod
    def _invoke_main(
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
        *,
        tty: bool = False,
    ) -> tuple[int, str, str]:
        """Run ``cli.main()`` in-process, capturing stdout/stderr and exit code.

        When *tty* is ``True`` the captured stderr buffer reports
        ``isatty() == True`` so the live-progress gate takes its interactive branch
        (otherwise a ``StringIO`` buffer is never a TTY and progress is always off).
        """
        import io

        from novomodelo_bridge import cli

        monkeypatch.setattr(sys, "argv", ["novomodelo-bridge", *argv])

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        if tty:
            stderr_buf.isatty = lambda: True  # type: ignore[method-assign]
        exit_code = 0

        with patch("sys.stdout", stdout_buf), patch("sys.stderr", stderr_buf):
            try:
                cli.main()
            except SystemExit as exc:
                exit_code = int(exc.code) if exc.code is not None else 0

        return exit_code, stdout_buf.getvalue(), stderr_buf.getvalue()

    @staticmethod
    def _file_handlers() -> list[logging.FileHandler]:
        """Return the ``FileHandler``s currently attached to the package logger."""
        pkg = logging.getLogger("novomodelo_bridge")
        return [h for h in pkg.handlers if isinstance(h, logging.FileHandler)]

    def test_configure_logging_levels(self) -> None:
        """The count maps 0 → warnings-only, 1 → INFO, 2 → DEBUG."""
        from novomodelo_bridge.cli.app import _NULL_HANDLER, _configure_logging

        pkg = logging.getLogger("novomodelo_bridge")

        _configure_logging(2, None)
        assert pkg.getEffectiveLevel() == logging.DEBUG

        _configure_logging(1, None)
        assert pkg.getEffectiveLevel() == logging.INFO

        _configure_logging(0, None)
        assert pkg.propagate is False
        assert _NULL_HANDLER in pkg.handlers

    def test_log_file_captures_debug_and_handler_removed_after_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--log-file`` writes the full DEBUG trace; the handler is gone after."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        log_path = tmp_path / "run.log"

        def _fake_convert(*_args: object, **_kwargs: object) -> ConversionReport:
            logging.getLogger("novomodelo_bridge.newave.pipeline").debug(
                "converting widgets"
            )
            return ConversionReport(hydro_count=1, stage_count=12)

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            side_effect=_fake_convert,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--log-file", str(log_path)],
                monkeypatch,
            )

        assert code == 0
        assert log_path.exists()
        contents = log_path.read_text(encoding="utf-8")
        assert "DEBUG novomodelo_bridge.newave.pipeline: converting widgets" in contents
        # The FileHandler was removed + closed in main()'s finally.
        assert self._file_handlers() == []

    def test_log_file_creates_missing_parent_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        log_path = tmp_path / "logs" / "nested" / "run.log"

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=ConversionReport(hydro_count=1, stage_count=12),
        ):
            code, _stdout, _stderr = self._invoke_main(
                [
                    "convert",
                    "newave",
                    str(src),
                    str(tmp_path / "dst"),
                    "--log-file",
                    str(log_path),
                ],
                monkeypatch,
            )

        assert code == 0
        assert log_path.exists()
        assert self._file_handlers() == []

    def test_unwritable_log_file_is_an_option_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dumb_terminal: None
    ) -> None:
        """A path that cannot be opened (here, an existing directory) fails as a
        ``--log-file`` usage error, not a traceback."""
        src = _make_fake_newave_dir(tmp_path)

        code, _stdout, stderr = self._invoke_main(
            [
                "convert",
                "newave",
                str(src),
                str(tmp_path / "dst"),
                "--log-file",
                str(tmp_path),
            ],
            monkeypatch,
        )

        assert code == 2
        assert "--log-file" in stderr
        assert "Traceback" not in stderr
        assert self._file_handlers() == []

    def test_consecutive_log_file_runs_leave_no_handler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two ``--log-file`` runs leave zero ``FileHandler``s (no leak)."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        report = ConversionReport(hydro_count=1, stage_count=12)

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=report,
        ):
            for run in ("a", "b"):
                code, _stdout, _stderr = self._invoke_main(
                    [
                        "convert",
                        "newave",
                        str(src),
                        str(tmp_path / f"dst_{run}"),
                        "--log-file",
                        str(tmp_path / f"run_{run}.log"),
                    ],
                    monkeypatch,
                )
                assert code == 0

        assert self._file_handlers() == []

    def test_log_file_without_verbose_keeps_progress_on_tty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On a TTY, ``--log-file`` with no ``-v`` still shows live progress.

        Spies on ``ui.console._progress_enabled`` to assert it is consulted with
        ``verbose=False`` (progress NOT suppressed) and returns ``True``.
        """
        from novomodelo_bridge import ui
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        log_path = tmp_path / "run.log"
        report = ConversionReport(hydro_count=1, stage_count=12)

        real_progress_enabled = ui.console._progress_enabled
        # Record (verbose-flag, gate-result) pairs as they happen during the run,
        # so the TTY-dependent result is captured inside the patched context.
        calls: list[tuple[bool, bool]] = []

        def _spy(*, verbose: bool, quiet: bool) -> bool:
            result = real_progress_enabled(verbose=verbose, quiet=quiet)
            calls.append((verbose, result))
            return result

        with (
            patch(
                "novomodelo_bridge.newave.pipeline.convert_newave_case",
                return_value=report,
            ),
            patch.object(ui.console, "_progress_enabled", _spy),
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--log-file", str(log_path)],
                monkeypatch,
                tty=True,
            )

        assert code == 0
        # conversion_progress consulted the gate with verbose=False (not suppressed)
        # and, on the simulated TTY, the gate returned True (progress shown).
        assert calls == [(False, True)]

    def test_vv_threads_int_and_suppresses_progress(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``dashboard -vv`` threads the int (2) and suppresses the spinner.

        Spies on ``_configure_logging`` (asserting it sees ``verbose=2``) and on
        ``ui.console._progress_enabled`` (asserting it is consulted with
        ``verbose=True`` — the ``> 0`` conversion of the count 2 — and returns
        ``False`` so the spinner is suppressed even on a TTY).
        """
        import importlib

        from novomodelo_bridge import ui

        cli = importlib.import_module("novomodelo_bridge.cli.app")

        case_dir = tmp_path / "case"
        (case_dir / "output" / "simulation").mkdir(parents=True)

        def _fake_build(_case_dir: Path, output_path: Path) -> None:
            output_path.write_text("x", encoding="utf-8")

        monkeypatch.setattr("novomodelo_bridge.dashboard.build_dashboard", _fake_build)

        real_configure = cli._configure_logging
        configure_calls: list[int] = []

        def _configure_spy(verbose: int, log_file: Path | None) -> None:
            configure_calls.append(verbose)
            real_configure(verbose, log_file)

        progress_calls: list[bool] = []

        def _progress_spy(*, verbose: bool, quiet: bool) -> bool:
            progress_calls.append(verbose)
            return False

        with (
            patch.object(cli, "_configure_logging", _configure_spy),
            patch.object(ui.console, "_progress_enabled", _progress_spy),
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["dashboard", str(case_dir), "-vv"],
                monkeypatch,
                tty=True,
            )

        assert code == 0
        # The int count threaded through unchanged.
        assert configure_calls == [2]
        # The spinner consulted the gate with verbose=True (count 2 → > 0) and the
        # gate (here forced False) suppressed it.
        assert progress_calls == [True]

    def test_vv_accepted_via_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Smoke: ``-vv`` is accepted on the real CLI and exits cleanly (code 0)."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        report = ConversionReport(hydro_count=1, stage_count=12)

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "-vv"],
                monkeypatch,
            )

        assert code == 0
        assert "1 hydros" in stdout
