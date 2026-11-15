"""Tests for the preflight advisories both tracks share (``core/preflight.py``)."""

from __future__ import annotations

from pathlib import Path

from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.libs_index import (
    HIDRELETRICA_CURVAJUSANTE,
    UNINDEXED_FILES_CODE,
    unindexed_files_diagnostic,
)
from novomodelo_bridge.core.preflight import (
    tab_character_advisory,
    unindexed_libs_advisory,
)


def test_tab_character_advisory_with_a_tab_passes_and_warns(tmp_path: Path) -> None:
    clast = tmp_path / "clast.dat"
    clast.write_text("    1 A\n\t2 B\n", encoding="utf-8")

    checks, diagnostics = tab_character_advisory([clast, tmp_path / "term.dat"])

    assert [(c.passed, c.detail) for c in checks] == [(True, "found; see the warning")]
    assert [(d.code, d.severity) for d in diagnostics] == [
        ("source-tab-character", Severity.WARNING)
    ]


def test_tab_character_advisory_without_tabs_is_one_passing_check(
    tmp_path: Path,
) -> None:
    conft = tmp_path / "conft.dat"
    conft.write_text("    1 A\n", encoding="utf-8")

    checks, diagnostics = tab_character_advisory([conft])

    assert [(c.passed, c.detail) for c in checks] == [(True, "none found")]
    assert diagnostics == []


def test_unindexed_libs_advisory_points_at_the_note() -> None:
    note = unindexed_files_diagnostic(
        [("polinjus.csv", HIDRELETRICA_CURVAJUSANTE)], remediation="Declare it."
    )
    assert note is not None

    checks, diagnostics = unindexed_libs_advisory(note)

    assert [(c.passed, c.detail) for c in checks] == [
        (True, "some ignored; see the note")
    ]
    assert [(d.code, d.severity) for d in diagnostics] == [
        (UNINDEXED_FILES_CODE, Severity.INFO)
    ]
