"""Tests for dessem/exporters/fcf/source.py, on the committed DECOMP mini-deck."""

from __future__ import annotations

from pathlib import Path

import pytest

from novomodelo_bridge.core.errors import SourceFileError
from novomodelo_bridge.dessem.exporters.fcf.core.inputs_reader import read_case_inputs
from novomodelo_bridge.dessem.exporters.fcf.core.mapper import match_source_hydros
from novomodelo_bridge.dessem.exporters.fcf.source import read_source_deck


def test_revision_comes_from_caso_dat(decomp_mini_deck: Path) -> None:
    deck = read_source_deck(decomp_mini_deck)

    assert deck.revision == (decomp_mini_deck / "caso.dat").read_text().split()[0]


def test_hydros_are_keyed_by_dense_id_in_ascending_code_order(
    decomp_mini_deck: Path,
) -> None:
    deck = read_source_deck(decomp_mini_deck)

    assert list(deck.hydros) == list(range(len(deck.hydros)))
    codes = [plant.code for plant in deck.hydros.values()]
    assert codes == sorted(codes)


def test_deck_matches_the_case_convert_writes_from_it(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    from novomodelo_bridge.decomp.pipeline import convert_decomp_case

    case_dir = tmp_path / "case"
    convert_decomp_case(decomp_mini_deck, case_dir, force=True)

    match_source_hydros(
        read_case_inputs(case_dir).hydros, read_source_deck(decomp_mini_deck).hydros
    )


def test_directory_without_caso_dat_raises(tmp_path: Path) -> None:
    with pytest.raises(SourceFileError, match="caso.dat"):
        read_source_deck(tmp_path)
