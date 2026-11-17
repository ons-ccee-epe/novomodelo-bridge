"""Tests for decomp/case_identity.py, the deck-against-case hydro identity check."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from novomodelo_bridge.core.errors import CaseIdentityError, FieldParseError
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.case_identity import (
    SourceHydro,
    deck_hydros,
    match_case_hydros,
    require_registered_hydros,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from tests.conftest import make_decomp_case


def _case(*hydro_ids: int) -> dict[int, str]:
    return {i: f"UHE_{i}" for i in hydro_ids}


def _deck(*hydro_ids: int) -> dict[int, SourceHydro]:
    return {i: SourceHydro(code=100 + i, name=f"UHE_{i}") for i in hydro_ids}


class TestMatchCaseHydros:
    """The deck must be the one the case was converted from."""

    def test_same_plants_and_names_pass(self) -> None:
        match_case_hydros(_case(0, 1), _deck(0, 1))

    def test_different_plant_set_raises(self) -> None:
        with pytest.raises(
            CaseIdentityError, match="converts 3 hydro plants but the case holds 2"
        ):
            match_case_hydros(_case(0, 1), _deck(0, 1, 2))

    def test_renamed_plant_raises(self) -> None:
        deck = _deck(0, 1) | {1: SourceHydro(code=101, name="OTHER")}
        with pytest.raises(CaseIdentityError, match="'UHE_1' in the case, 'OTHER'"):
            match_case_hydros(_case(0, 1), deck)


class TestRegisteredHydros:
    """`hidr.dat` must hold every converted plant, the dormant ones included."""

    @staticmethod
    def _case(tmp_path: Path, *, registered: tuple[int, ...]) -> DecompCase:
        id_map = DecompIdMap(
            bus_codes=(1,),
            bus_names=("SE",),
            hydro_codes=(10, 20, 30),
            dormant_hydro_codes=(20,),
        )
        hidr = pd.DataFrame(
            {"nome_usina": [f"PLANT {code} " for code in registered]},
            index=pd.Index(registered, name="codigo_usina"),
        )
        return make_decomp_case(tmp_path, id_map=id_map, hidr=hidr)

    def test_every_converted_plant_registered_passes(self, tmp_path: Path) -> None:
        case = self._case(tmp_path, registered=(10, 20, 30))

        require_registered_hydros(case)

        assert deck_hydros(case) == {
            0: SourceHydro(code=10, name="PLANT 10"),
            1: SourceHydro(code=20, name="PLANT 20"),
            2: SourceHydro(code=30, name="PLANT 30"),
        }

    @pytest.mark.parametrize("check", [require_registered_hydros, deck_hydros])
    def test_a_dormant_plant_missing_from_hidr_raises(
        self, tmp_path: Path, check: Callable[[DecompCase], object]
    ) -> None:
        case = self._case(tmp_path, registered=(10, 30))

        with pytest.raises(
            FieldParseError, match="converted plant 20 is not in the hydro registry"
        ):
            check(case)
