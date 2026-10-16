"""Reading the source decks' text files, and the tab check on fixed-column ones."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    format_stage_ranges,
)


def read_source_text(path: Path) -> str:
    """*path*'s text in the encoding it was written in: UTF-8 when it is valid
    UTF-8 (a BOM is dropped), else latin-1.

    Decks mix encodings file by file, so the choice is never made once per deck.
    """
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Not UTF-8 means a single-byte code page; latin-1 decodes every byte.
        return data.decode("latin-1")


def tab_character_diagnostic(
    paths: Sequence[Path], *, comment_prefix: str | None = None
) -> Diagnostic | None:
    """WARNING naming every line of *paths* that holds a tab, or ``None``.

    Args:
        comment_prefix: lines opening with it, after any indentation, are
            skipped as free text the readers ignore.
    """
    found = _tab_lines(paths, comment_prefix=comment_prefix)
    if not found:
        return None
    return Diagnostic(
        code="source-tab-character",
        severity=Severity.WARNING,
        category="Source files",
        title="Tab characters in fixed-column files",
        summary=(
            f"{len(found)} input file(s) contain tab characters. These files are "
            "read by fixed columns, so a tab shifts every field after it on its "
            "line, and a data line can read with wrong or missing values."
        ),
        table=DiagnosticTable(
            columns=["file", "lines"],
            rows=[
                [path.name, format_stage_ranges(lines)] for path, lines in found.items()
            ],
        ),
        remediation=(
            "Replace each tab with the spaces that put the line's fields back in "
            "their columns."
        ),
    )


def _tab_lines(
    paths: Sequence[Path], *, comment_prefix: str | None
) -> dict[Path, list[int]]:
    """The 1-based numbers of the lines that contain a tab, per file that has any."""
    prefix = comment_prefix.encode() if comment_prefix is not None else None
    found: dict[Path, list[int]] = {}
    for path in paths:
        try:
            lines = path.read_bytes().split(b"\n")
        except OSError:
            # The conversion's own read reports an unreadable file better.
            continue
        numbers = [
            number
            for number, line in enumerate(lines, start=1)
            if b"\t" in line
            and not (prefix is not None and line.lstrip().startswith(prefix))
        ]
        if numbers:
            found[path] = numbers
    return found
