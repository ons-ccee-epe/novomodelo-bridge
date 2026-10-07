"""Shared filesystem path-resolution helpers.

Both the source-model and DECOMP readers resolve files inside a case
directory without regard to case (the tools that write these decks are
inconsistent about filename casing across platforms). This is the single
public home for that lookup, so neither reader reaches into the other's
private helper (``.claude/rules/bridge.md`` §1).
"""

from __future__ import annotations

from pathlib import Path


def find_case_insensitive(directory: Path, filename: str) -> Path | None:
    """Find a file in *directory* ignoring case, or return None."""
    lower = filename.lower()
    try:
        for entry in directory.iterdir():
            if entry.is_file() and entry.name.lower() == lower:
                return entry
    except OSError:
        pass
    return None
