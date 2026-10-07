"""Tests for the shared case-insensitive filename resolver."""

from __future__ import annotations

from pathlib import Path

from novomodelo_bridge.core.paths import find_case_insensitive


def test_find_case_insensitive_mixed_case_hit_returns_path(tmp_path: Path) -> None:
    target = tmp_path / "CaSo.DAT"
    target.write_text("data", encoding="utf-8")

    assert find_case_insensitive(tmp_path, "caso.dat") == target


def test_find_case_insensitive_miss_returns_none(tmp_path: Path) -> None:
    assert find_case_insensitive(tmp_path, "missing.dat") is None


def test_find_case_insensitive_non_directory_returns_none(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x", encoding="utf-8")

    assert find_case_insensitive(not_a_dir, "anything.dat") is None
