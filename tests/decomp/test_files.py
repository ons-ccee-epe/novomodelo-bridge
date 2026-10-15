"""Tests for ``DecompFiles``, the resolved input files of one deck revision."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import make_decomp_files


def test_fixed_column_files_are_the_register_files_present(tmp_path: Path) -> None:
    without_gnl = make_decomp_files(tmp_path, renovaveis=tmp_path / "renovaveis.rv0")
    with_gnl = make_decomp_files(tmp_path, dadgnl=tmp_path / "dadgnl.rv0")

    assert without_gnl.fixed_column_files == [tmp_path / "dadger"]
    assert with_gnl.fixed_column_files == [tmp_path / "dadger", tmp_path / "dadgnl.rv0"]
