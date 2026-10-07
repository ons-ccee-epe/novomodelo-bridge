"""Tests for cli.py: the Typer app surface."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from tests.conftest import _make_fake_newave_dir


class TestTyperApp:
    """Typer-app behaviours via the idiomatic CliRunner invocation path."""

    @staticmethod
    def _invoke(argv: list[str]):
        from typer.testing import CliRunner

        from novomodelo_bridge.cli import app

        return CliRunner().invoke(app, argv)

    def test_version_exit_zero(self) -> None:
        result = self._invoke(["--version"])
        assert result.exit_code == 0
        assert "novomodelo-bridge" in result.stdout

    def test_help_lists_subcommands(self, dumb_terminal: None) -> None:
        result = self._invoke(["--help"])
        assert result.exit_code == 0
        assert "convert" in result.stdout
        assert "compare" in result.stdout
        assert "dashboard" in result.stdout

    def test_help_exposes_shell_completion(self, dumb_terminal: None) -> None:
        result = self._invoke(["--help"])
        assert "install-completion" in result.stdout

    def test_convert_missing_subcommand_exits_two(self) -> None:
        assert self._invoke(["convert"]).exit_code == 2

    def test_compare_missing_subcommand_exits_two(self) -> None:
        assert self._invoke(["compare"]).exit_code == 2

    def test_convert_newave_happy_path(self, tmp_path: Path) -> None:
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "out"
        report = ConversionReport(hydro_count=7, stage_count=12)
        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case", return_value=report
        ):
            result = self._invoke(["convert", "newave", str(src), str(dst)])
        assert result.exit_code == 0
        assert "7 hydros" in result.stdout

    @staticmethod
    def _json_help_text(stdout: str) -> str:
        """Reassemble the ``--json`` help text, undoing Rich's per-table wrap.

        The Options box wraps the help column at a width set by the longest
        option name/metavar in that *specific* command, so the same help
        string breaks across a different number of lines per command; compare
        the reflowed words, not the raw wrapped lines.
        """
        words: list[str] = []
        capturing = False
        for raw in stdout.splitlines():
            content = raw.strip().strip("│").strip()
            if content.startswith("--json"):
                capturing = True
                words.extend(content.split()[1:])
                continue
            if capturing:
                if content == "" or content.startswith(("--", "╰")):
                    break
                words.extend(content.split())
        return " ".join(words)

    def test_json_help_line_identical_across_commands(
        self, dumb_terminal: None
    ) -> None:
        """The ``--json`` help text converged to one canonical string (``_JsonOpt``)."""
        canonical = (
            "Emit a single machine-readable JSON verdict to stdout and "
            "suppress the human-readable (Rich) output."
        )
        commands = [
            ["convert", "newave", "--help"],
            ["convert", "decomp", "--help"],
            ["compare", "newave", "--help"],
            ["compare", "decomp", "--help"],
            ["check", "newave", "--help"],
            ["check", "decomp", "--help"],
            ["dashboard", "--help"],
        ]
        texts: dict[str, str] = {}
        for argv in commands:
            result = self._invoke(argv)
            assert result.exit_code == 0
            texts[" ".join(argv[:-1])] = self._json_help_text(result.stdout)
        assert len(set(texts.values())) == 1, texts
        assert next(iter(texts.values())) == canonical
