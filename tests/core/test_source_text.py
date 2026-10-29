"""Tests for reading the source decks' text files and for the tab check."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.source_text import (
    read_source_text,
    tab_character_diagnostic,
)


def _write(path: Path, text: str) -> Path:
    path.write_bytes(text.encode("latin-1"))
    return path


def _rows(
    paths: list[Path], *, comment_prefix: str | None = None
) -> Sequence[Sequence[object]]:
    diagnostic = tab_character_diagnostic(paths, comment_prefix=comment_prefix)
    assert diagnostic is not None and diagnostic.table is not None
    return diagnostic.table.rows


def test_tab_character_diagnostic_lists_each_file_with_its_line_ranges(
    tmp_path: Path,
) -> None:
    clean = _write(tmp_path / "conft.dat", "    1 A\r\n    2 B\r\n")
    clast = _write(
        tmp_path / "clast.dat",
        "    1 A\r\n\t2 B\r\n    3 C\r\n    4\tD\r\n    5 E\t\r\n",
    )
    term = _write(tmp_path / "term.dat", "\t1\r\n")

    diagnostic = tab_character_diagnostic([clean, clast, term])

    assert diagnostic is not None
    assert diagnostic.code == "source-tab-character"
    assert diagnostic.severity is Severity.WARNING
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [["clast.dat", "2, 4-5"], ["term.dat", "1"]]


def test_tab_character_diagnostic_skips_comment_lines_even_when_indented(
    tmp_path: Path,
) -> None:
    dadger = _write(tmp_path / "dadger.rv0", "&\tfree text\n  &\tnote\nTE  title\t\n")

    assert _rows([dadger], comment_prefix="&") == [["dadger.rv0", "3"]]
    assert _rows([dadger]) == [["dadger.rv0", "1-3"]]


def test_tab_character_diagnostic_is_none_without_tabs(tmp_path: Path) -> None:
    clean = _write(tmp_path / "conft.dat", "    1 A\n")

    assert tab_character_diagnostic([clean, tmp_path / "absent.dat"]) is None


def test_tab_character_remediation_has_no_repo_internal_references(
    tmp_path: Path,
) -> None:
    diagnostic = tab_character_diagnostic([_write(tmp_path / "clast.dat", "\t1\n")])

    assert diagnostic is not None
    text = f"{diagnostic.summary} {diagnostic.remediation}"
    for leak in ("docs/", "plans/", "~/git", "ticket-", "epic-", ".py"):
        assert leak not in text


def test_read_source_text_decodes_utf8_accents(tmp_path: Path) -> None:
    path = tmp_path / "restricao-eletrica.csv"
    path.write_bytes("& Restrição de geração UHE Jacuí\n".encode())

    assert read_source_text(path) == "& Restrição de geração UHE Jacuí\n"


def test_read_source_text_drops_a_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "indices.csv"
    path.write_bytes(b"\xef\xbb\xbfRESTRICAO-ELETRICA-ESPECIAL; ; re.csv\n")

    assert read_source_text(path).startswith("RESTRICAO-ELETRICA-ESPECIAL")


def test_read_source_text_falls_back_to_latin1_for_a_non_utf8_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "renovaveis.csv"
    path.write_bytes("& Descrição colunas: parque eólico\n".encode("latin-1"))

    assert read_source_text(path) == "& Descrição colunas: parque eólico\n"
