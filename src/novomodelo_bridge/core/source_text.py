"""The tab-character check on the source decks' fixed-column text files.

The source-model readers slice each line at fixed columns, so a tab counts as
one column where the layout expects several spaces: the fields after it shift,
and the line reads with wrong or missing values, or is not recognised as a
record at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    format_stage_ranges,
)


def tab_lines(
    paths: Sequence[Path], *, comment_prefix: str | None = None
) -> dict[Path, list[int]]:
    """The 1-based numbers of the lines that contain a tab, per file that has any.

    Args:
        comment_prefix: lines starting with it are skipped, as free text the
            readers ignore.
    """
    prefix = comment_prefix.encode() if comment_prefix is not None else None
    found: dict[Path, list[int]] = {}
    for path in paths:
        try:
            lines = path.read_bytes().split(b"\n")
        except OSError:
            # An unreadable file fails the conversion's own read with a better
            # message; this advisory has nothing to add.
            continue
        numbers = [
            number
            for number, line in enumerate(lines, start=1)
            if b"\t" in line and not (prefix is not None and line.startswith(prefix))
        ]
        if numbers:
            found[path] = numbers
    return found


def tab_character_diagnostic(
    found: Mapping[Path, Sequence[int]],
) -> Diagnostic | None:
    """WARNING naming every line in *found* (from :func:`tab_lines`), or ``None``."""
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
            "line, and the line reads with wrong or missing values."
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
