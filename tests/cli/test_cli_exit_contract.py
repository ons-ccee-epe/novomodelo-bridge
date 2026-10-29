"""Exit-code and ``--json`` envelope contract, asserted across every command.

``.claude/rules/bridge.md`` (section 6) and the README promise that, under
``--json``, every failure prints exactly one verdict envelope on stdout and that
an error status and a non-zero exit code travel together. These tests hold the
twin tracks to the same codes for the same failure (a missing source exits the
same on NEWAVE and DECOMP) and hold every command, found by walking the real
command tree, to the usage-error contract. Tier-1: nothing needs ``novomodelo``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import typer.main
from typer.core import TyperGroup
from typer.testing import CliRunner

from novomodelo_bridge.cli import app

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

ENVELOPE_KEYS = ["schema_version", "command", "status", "summary", "diagnostics"]


def _run(argv: list[str]) -> Any:
    return CliRunner().invoke(app, argv)


def _envelope(result: Any) -> dict[str, Any]:
    """Stdout must be one JSON document with exactly the five envelope keys."""
    document = json.loads(result.stdout)
    assert list(document) == ENVELOPE_KEYS
    return document


def _leaf_commands() -> Iterator[list[str]]:
    def walk(command: Any, path: list[str]) -> Iterator[list[str]]:
        if isinstance(command, TyperGroup):
            for name, child in command.commands.items():
                yield from walk(child, [*path, name])
        else:
            yield path

    yield from walk(typer.main.get_command(app), [])


LEAF_COMMANDS = sorted(_leaf_commands())


@pytest.fixture
def missing(tmp_path: Path) -> Path:
    return tmp_path / "does-not-exist"


@pytest.fixture
def empty(tmp_path: Path) -> Path:
    directory = tmp_path / "empty"
    directory.mkdir()
    return directory


class TestUsageErrors:
    @pytest.mark.parametrize("path", LEAF_COMMANDS, ids=" ".join)
    def test_every_command_envelopes_a_usage_error_under_json(
        self, path: list[str]
    ) -> None:
        result = _run([*path, "--no-such-option", "--json"])

        assert result.exit_code == 2
        document = _envelope(result)
        assert document["command"] == " ".join(path)
        assert document["status"] == "error"
        assert document["summary"] == {}
        (diagnostic,) = document["diagnostics"]
        assert diagnostic["code"] == "usage-error"
        assert diagnostic["severity"] == "error"
        assert "--no-such-option" in diagnostic["summary"]
        assert result.stderr == ""

    @pytest.mark.parametrize("path", LEAF_COMMANDS, ids=" ".join)
    def test_every_command_leaves_a_usage_error_to_click_without_json(
        self, path: list[str]
    ) -> None:
        result = _run([*path, "--no-such-option"])

        assert result.exit_code == 2
        assert result.stdout == ""
        assert "--no-such-option" in result.stderr

    @pytest.mark.parametrize(
        ("argv", "command", "fragment"),
        [
            (["convert", "newave", "--json"], "convert newave", "Missing argument"),
            (["convert", "decomp", "--json"], "convert decomp", "Missing argument"),
            (["compare", "newave", "--json"], "compare newave", "Missing argument"),
            (["compare", "decomp", "--json"], "compare decomp", "Missing argument"),
            (["check", "newave", "--json"], "check newave", "Missing argument"),
            (["check", "decomp", "--json"], "check decomp", "Missing argument"),
            (["dashboard", "--json"], "dashboard", "Missing argument"),
            (["export", "dessem", "--json"], "export dessem", "Missing argument"),
            (["convert", "bogus", "--json"], "convert", "No such command"),
            (["bogus", "--json"], "novomodelo-bridge", "No such command"),
            (["--bogus", "convert", "--json"], "novomodelo-bridge", "No such option"),
        ],
        ids=lambda value: value if isinstance(value, str) else " ".join(value),
    )
    def test_the_command_names_where_the_command_line_was_rejected(
        self, argv: list[str], command: str, fragment: str
    ) -> None:
        result = _run(argv)

        assert result.exit_code == 2
        document = _envelope(result)
        assert document["command"] == command
        assert fragment in document["diagnostics"][0]["summary"]

    @pytest.mark.parametrize("track", ["newave", "decomp"])
    def test_an_invalid_option_value_is_a_usage_error_on_both_tracks(
        self, track: str, missing: Path, empty: Path
    ) -> None:
        result = _run(
            ["compare", track, str(missing), str(empty), "--tolerance", "abc", "--json"]
        )

        assert result.exit_code == 2
        document = _envelope(result)
        assert document["command"] == f"compare {track}"
        assert document["diagnostics"][0]["code"] == "usage-error"

    def test_help_is_not_a_failure_under_json(self) -> None:
        result = _run(["convert", "newave", "--help", "--json"])

        assert result.exit_code == 0
        assert "Usage" in result.stdout

    def test_version_is_not_a_failure(self) -> None:
        assert _run(["--version"]).exit_code == 0


class TestTwinTrackParity:
    """The same failure exits the same way on NEWAVE and DECOMP."""

    @pytest.mark.parametrize("track", ["newave", "decomp"])
    def test_convert_missing_source_exits_one(
        self, track: str, missing: Path, tmp_path: Path
    ) -> None:
        result = _run(["convert", track, str(missing), str(tmp_path / "out"), "--json"])

        assert result.exit_code == 1
        document = _envelope(result)
        assert (document["command"], document["status"]) == (
            f"convert {track}",
            "error",
        )

    @pytest.mark.parametrize("track", ["newave", "decomp"])
    def test_compare_missing_source_exits_one(
        self, track: str, missing: Path, empty: Path
    ) -> None:
        result = _run(["compare", track, str(missing), str(empty), "--json"])

        assert result.exit_code == 1
        document = _envelope(result)
        assert (document["command"], document["status"]) == (
            f"compare {track}",
            "error",
        )

    @pytest.mark.parametrize("track", ["newave", "decomp"])
    def test_compare_source_without_a_deck_exits_one(
        self, track: str, empty: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "output"
        output.mkdir()

        result = _run(["compare", track, str(empty), str(output), "--json"])

        assert result.exit_code == 1
        document = _envelope(result)
        assert document["status"] == "error"
        assert document["diagnostics"][0]["code"] == "source-file-missing"

    @pytest.mark.parametrize("track", ["newave", "decomp"])
    def test_check_missing_source_exits_two_will_not_convert(
        self, track: str, missing: Path
    ) -> None:
        result = _run(["check", track, str(missing), "--json"])

        assert result.exit_code == 2
        document = _envelope(result)
        assert (document["command"], document["status"]) == (
            f"check {track}",
            "will-not-convert",
        )

    @pytest.mark.parametrize("track", ["newave", "decomp"])
    def test_compare_unreadable_novomodelo_output_exits_two(
        self,
        track: str,
        newave_mini_deck: Path,
        decomp_mini_deck: Path,
        empty: Path,
    ) -> None:
        deck = {"newave": newave_mini_deck, "decomp": decomp_mini_deck}[track]

        result = _run(["compare", track, str(deck), str(empty), "--json"])

        assert result.exit_code == 2
        document = _envelope(result)
        assert (document["command"], document["status"]) == (
            f"compare {track}",
            "error",
        )

    def test_the_tracks_report_the_same_code_for_the_same_failure(
        self, missing: Path, empty: Path, tmp_path: Path
    ) -> None:
        def outcome(track: str) -> tuple[int, str, str]:
            result = _run(["compare", track, str(missing), str(empty), "--json"])
            document = _envelope(result)
            return (
                result.exit_code,
                document["status"],
                document["diagnostics"][0]["code"],
            )

        assert outcome("newave") == outcome("decomp")


class TestOtherFamilies:
    def test_dashboard_missing_case_exits_one(self, missing: Path) -> None:
        result = _run(["dashboard", str(missing), "--json"])

        assert result.exit_code == 1
        document = _envelope(result)
        assert (document["command"], document["status"]) == ("dashboard", "error")

    def test_export_dessem_missing_deck_exits_one(
        self, missing: Path, empty: Path
    ) -> None:
        result = _run(["export", "dessem", str(missing), str(empty), "--json"])

        assert result.exit_code == 1
        document = _envelope(result)
        assert (document["command"], document["status"]) == ("export dessem", "error")

    def test_export_dessem_missing_case_exits_one(
        self, decomp_mini_deck: Path, missing: Path
    ) -> None:
        result = _run(
            ["export", "dessem", str(decomp_mini_deck), str(missing), "--json"]
        )

        assert result.exit_code == 1
        document = _envelope(result)
        assert (document["command"], document["status"]) == ("export dessem", "error")
