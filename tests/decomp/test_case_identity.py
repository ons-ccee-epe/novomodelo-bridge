"""Tests for decomp/case_identity.py, the deck-against-case hydro identity check."""

from __future__ import annotations

import pytest

from novomodelo_bridge.core.errors import CaseIdentityError
from novomodelo_bridge.decomp.case_identity import SourceHydro, match_case_hydros


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
            CaseIdentityError, match="operates 3 hydro plants but the case holds 2"
        ):
            match_case_hydros(_case(0, 1), _deck(0, 1, 2))

    def test_renamed_plant_raises(self) -> None:
        deck = _deck(0, 1) | {1: SourceHydro(code=101, name="OTHER")}
        with pytest.raises(CaseIdentityError, match="'UHE_1' in the case, 'OTHER'"):
            match_case_hydros(_case(0, 1), deck)
