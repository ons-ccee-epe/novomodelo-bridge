"""Tests for dessem/exporters/fcf/source.py, on the committed DECOMP mini-deck."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from novomodelo_bridge.core.errors import SourceFileError
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.anticipated import (
    GnlCommitmentModel,
    GnlThermal,
)
from novomodelo_bridge.dessem.exporters.fcf.core.inputs_reader import read_case_inputs
from novomodelo_bridge.dessem.exporters.fcf.core.mapper import (
    SourceGnlThermal,
    match_source_hydros,
)
from novomodelo_bridge.dessem.exporters.fcf.core.travel_time import travel_time_plants
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


def test_travel_time_plants_of_the_converted_case_follow_the_vi_records(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    from novomodelo_bridge.decomp.pipeline import convert_decomp_case

    case_dir = tmp_path / "case"
    convert_decomp_case(decomp_mini_deck, case_dir, force=True)
    inputs = read_case_inputs(case_dir)
    deck = read_source_deck(decomp_mini_deck)

    plants = travel_time_plants(
        inputs.hydros, deck.hydros, deck.travel_time_codes, inputs.stages
    )

    assert deck.travel_time_codes == (156, 162)
    assert [(plant.code, plant.hours) for plant in plants] == [(156, 360), (162, 360)]


def test_travel_time_codes_keep_the_deck_order_over_ascending_code(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    deck_dir = tmp_path / "deck"
    shutil.copytree(decomp_mini_deck, deck_dir)
    dadger = next(deck_dir.glob("dadger.*"))
    lines = dadger.read_text(encoding="latin-1").splitlines(keepends=True)
    first, second = (
        index for index, line in enumerate(lines) if line.startswith("VI ")
    )
    lines[first], lines[second] = lines[second], lines[first]
    dadger.write_text("".join(lines), encoding="latin-1")

    assert read_source_deck(deck_dir).travel_time_codes == (162, 156)


def test_deck_without_gnl_dispatch_has_no_gnl_plants(decomp_mini_deck: Path) -> None:
    assert read_source_deck(decomp_mini_deck).gnl_thermals == ()


def test_gnl_plants_carry_their_nl_lag_in_code_order(decomp_mini_deck: Path) -> None:
    model = GnlCommitmentModel(
        thermals=(
            GnlThermal(code=86, name="SANTA CRUZ", submarket_code=1),
            GnlThermal(code=224, name="PSERGIPE I", submarket_code=3),
        ),
        commitments={},
        weeks_per_month={},
        nl_lag_months={86: 2},
        tg=pd.DataFrame(),
    )
    with (
        patch.object(DecompCase, "dadgnl", new=object()),
        patch(
            "novomodelo_bridge.decomp.converters.anticipated.read_gnl_model",
            return_value=model,
        ),
    ):
        deck = read_source_deck(decomp_mini_deck)

    assert deck.gnl_thermals == (
        SourceGnlThermal(code=86, name="SANTA CRUZ", lag_months=2),
        SourceGnlThermal(code=224, name="PSERGIPE I", lag_months=None),
    )


def test_directory_without_caso_dat_raises(tmp_path: Path) -> None:
    with pytest.raises(SourceFileError, match="caso.dat"):
        read_source_deck(tmp_path)
