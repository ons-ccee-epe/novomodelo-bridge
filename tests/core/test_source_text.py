"""Tests for the tab-character check on the source decks' fixed-column files."""

from __future__ import annotations

from pathlib import Path

from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.source_text import tab_character_diagnostic, tab_lines


def _write(path: Path, text: str) -> Path:
    path.write_bytes(text.encode("latin-1"))
    return path


def test_tab_lines_numbers_lines_from_one_per_file_with_tabs(tmp_path: Path) -> None:
    clean = _write(tmp_path / "conft.dat", "    1 A\r\n    2 B\r\n")
    tabbed = _write(
        tmp_path / "clast.dat",
        "    1 A\r\n\t2 B\r\n    3 C\r\n    4\tD\r\n    5 E\t\r\n",
    )

    assert tab_lines([clean, tabbed]) == {tabbed: [2, 4, 5]}


def test_tab_lines_skips_lines_opening_with_the_comment_prefix(tmp_path: Path) -> None:
    dadger = _write(tmp_path / "dadger.rv0", "&\tfree text\nTE  title\t\n")

    assert tab_lines([dadger], comment_prefix="&") == {dadger: [2]}
    assert tab_lines([dadger]) == {dadger: [1, 2]}


def test_tab_lines_skips_a_file_it_cannot_read(tmp_path: Path) -> None:
    assert tab_lines([tmp_path / "absent.dat"]) == {}


def test_tab_character_diagnostic_is_none_without_tabs() -> None:
    assert tab_character_diagnostic({}) is None


def test_tab_character_diagnostic_lists_each_file_with_its_line_ranges(
    tmp_path: Path,
) -> None:
    diagnostic = tab_character_diagnostic(
        {tmp_path / "clast.dat": [3, 7, 8], tmp_path / "term.dat": [1]}
    )

    assert diagnostic is not None
    assert diagnostic.code == "source-tab-character"
    assert diagnostic.severity is Severity.WARNING
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [["clast.dat", "3, 7-8"], ["term.dat", "1"]]


def test_tab_character_remediation_has_no_repo_internal_references(
    tmp_path: Path,
) -> None:
    diagnostic = tab_character_diagnostic({tmp_path / "clast.dat": [1]})

    assert diagnostic is not None
    text = f"{diagnostic.summary} {diagnostic.remediation}"
    for leak in ("docs/", "plans/", "~/git", "ticket-", "epic-", ".py"):
        assert leak not in text
