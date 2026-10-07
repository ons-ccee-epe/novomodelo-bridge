"""Tests for ``novomodelo_bridge.newave.converters.fict_cascade.resolve_cascade``.

Focuses on cascade rewiring rules — in particular, that NE/NC plants in
the middle of a cascade are bypassed transparently so upstream real
plants keep a valid downstream link.
"""

from __future__ import annotations

import pandas as pd
import pytest

from novomodelo_bridge.newave.converters.fict_cascade import resolve_cascade


def _confhd_row(
    code: int,
    name: str,
    downstream: int,
    status: str = "EX",
) -> dict[str, object]:
    """Build a minimal confhd row for tests."""
    return {
        "codigo_usina": code,
        "nome_usina": name,
        "codigo_usina_jusante": downstream,
        "usina_existente": status,
    }


def _make_confhd(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


@pytest.fixture
def empty_cadastro() -> pd.DataFrame:
    """A cadastro with no entries — FICT ρ_eq falls back to zero."""
    return pd.DataFrame()


class TestNonExistingPlantBypass:
    """NE/NC plants mid-cascade must be bypassed so the graph stays connected."""

    def test_nc_plant_between_two_real_rewires_to_downstream(
        self,
        empty_cadastro: pd.DataFrame,
    ) -> None:
        # A (EX) -> B (NC) -> C (EX); A should rewire to C.
        confhd = _make_confhd(
            [
                _confhd_row(1, "A", 2, "EX"),
                _confhd_row(2, "B", 3, "NC"),
                _confhd_row(3, "C", 0, "EX"),
            ]
        )
        res = resolve_cascade(confhd, empty_cadastro)
        assert res[1].downstream_code == 3
        assert res[1].fict_chain == ()
        assert res[1].fict_rho_sum == 0.0
        # C has no downstream — terminal.
        assert res[3].downstream_code is None

    def test_ne_plant_between_two_real_rewires_to_downstream(
        self,
        empty_cadastro: pd.DataFrame,
    ) -> None:
        # A (EX) -> B (NE) -> C (EX); A should rewire to C.
        confhd = _make_confhd(
            [
                _confhd_row(1, "A", 2, "EX"),
                _confhd_row(2, "B", 3, "NE"),
                _confhd_row(3, "C", 0, "EX"),
            ]
        )
        res = resolve_cascade(confhd, empty_cadastro)
        assert res[1].downstream_code == 3

    def test_consecutive_absent_plants_are_all_skipped(
        self,
        empty_cadastro: pd.DataFrame,
    ) -> None:
        # A (EX) -> B (NC) -> C (NE) -> D (EX); A should rewire straight to D.
        confhd = _make_confhd(
            [
                _confhd_row(1, "A", 2, "EX"),
                _confhd_row(2, "B", 3, "NC"),
                _confhd_row(3, "C", 4, "NE"),
                _confhd_row(4, "D", 0, "EX"),
            ]
        )
        res = resolve_cascade(confhd, empty_cadastro)
        assert res[1].downstream_code == 4

    def test_absent_plant_at_chain_end_terminates_upstream(
        self,
        empty_cadastro: pd.DataFrame,
    ) -> None:
        # A (EX) -> B (NC) -> 0 (terminal); A becomes terminal too.
        confhd = _make_confhd(
            [
                _confhd_row(1, "A", 2, "EX"),
                _confhd_row(2, "B", 0, "NC"),
            ]
        )
        res = resolve_cascade(confhd, empty_cadastro)
        assert res[1].downstream_code is None

    def test_top_of_cascade_absent_plant_is_not_resolved(
        self,
        empty_cadastro: pd.DataFrame,
    ) -> None:
        # B (NC) has no upstream and is not EX, so it gets no resolution row.
        confhd = _make_confhd(
            [
                _confhd_row(1, "A", 0, "EX"),
                _confhd_row(2, "B", 0, "NC"),
            ]
        )
        res = resolve_cascade(confhd, empty_cadastro)
        assert 2 not in res
        assert res[1].downstream_code is None

    def test_absent_plant_does_not_contribute_rho(
        self,
        empty_cadastro: pd.DataFrame,
    ) -> None:
        # Even when traversing through an NE/NC plant, fict_rho_sum stays
        # at zero — out-of-LP plants have no productivity contribution.
        confhd = _make_confhd(
            [
                _confhd_row(1, "A", 2, "EX"),
                _confhd_row(2, "B", 3, "NC"),
                _confhd_row(3, "C", 0, "EX"),
            ]
        )
        res = resolve_cascade(confhd, empty_cadastro)
        assert res[1].fict_rho_sum == 0.0
        assert res[1].fict_chain == ()


class TestExistingBehaviorPreserved:
    """Rules 1, 2, 3 still work as before when no NE/NC plants are involved."""

    def test_direct_real_downstream_unaffected(
        self,
        empty_cadastro: pd.DataFrame,
    ) -> None:
        confhd = _make_confhd(
            [
                _confhd_row(1, "A", 2, "EX"),
                _confhd_row(2, "B", 0, "EX"),
            ]
        )
        res = resolve_cascade(confhd, empty_cadastro)
        assert res[1].downstream_code == 2
        assert res[1].fict_chain == ()

    def test_fict_plant_still_walked_through(
        self,
        empty_cadastro: pd.DataFrame,
    ) -> None:
        confhd = _make_confhd(
            [
                _confhd_row(1, "A", 99, "EX"),
                _confhd_row(99, "FICT.A", 2, "EX"),
                _confhd_row(2, "B", 0, "EX"),
            ]
        )
        res = resolve_cascade(confhd, empty_cadastro, fictitious={99})
        assert res[1].downstream_code == 2
        assert res[1].fict_chain == (99,)


class TestFillingPlantAdmission:
    """``NE``-with-filling plants are real novomodelo nodes, not pass-throughs."""

    def test_cascade_filling_plant_is_real_downstream(
        self,
        empty_cadastro: pd.DataFrame,
    ) -> None:
        # A (EX) -> 309 (NE-with-filling) -> C (EX). With 309 promoted to a
        # real node, A's downstream stops at 309 instead of walking through it.
        confhd = _make_confhd(
            [
                _confhd_row(1, "A", 309, "EX"),
                _confhd_row(309, "JURUENA", 3, "NE"),
                _confhd_row(3, "C", 0, "EX"),
            ]
        )
        res = resolve_cascade(confhd, empty_cadastro, filling_codes={309})
        assert res[1].downstream_code == 309
        assert res[1].fict_chain == ()
        assert res[1].fict_rho_sum == 0.0

    def test_cascade_without_filling_codes_unchanged(
        self,
        empty_cadastro: pd.DataFrame,
    ) -> None:
        # Same topology, but no filling set: 309 is an ordinary NE plant and is
        # walked through, so A rewires to the next EX plant (3) — today's
        # behaviour, i.e. not 309.
        confhd = _make_confhd(
            [
                _confhd_row(1, "A", 309, "EX"),
                _confhd_row(309, "JURUENA", 3, "NE"),
                _confhd_row(3, "C", 0, "EX"),
            ]
        )
        res = resolve_cascade(confhd, empty_cadastro)
        assert res[1].downstream_code == 3
        assert res[1].downstream_code != 309
        # 309 stays absent — no resolution entry of its own.
        assert 309 not in res

    def test_cascade_filling_plant_has_own_entry(
        self,
        empty_cadastro: pd.DataFrame,
    ) -> None:
        # A promoted filling plant gets its own resolution row, with its
        # downstream resolved per the usual rules (here, terminal at the sea).
        confhd = _make_confhd(
            [
                _confhd_row(1, "A", 309, "EX"),
                _confhd_row(309, "JURUENA", 0, "NE"),
            ]
        )
        res = resolve_cascade(confhd, empty_cadastro, filling_codes={309})
        assert 309 in res
        assert res[309].downstream_code is None
        assert res[309].fict_chain == ()
        assert res[1].downstream_code == 309


class TestMixedFictAndAbsent:
    """Combinations of FICT and NE/NC plants in the same chain."""

    def test_fict_followed_by_absent_walks_through_both(
        self,
        empty_cadastro: pd.DataFrame,
    ) -> None:
        # A (EX) -> F (FICT.EX) -> B (NC) -> C (EX); A->C, F in fict_chain.
        confhd = _make_confhd(
            [
                _confhd_row(1, "A", 99, "EX"),
                _confhd_row(99, "FICT.A", 2, "EX"),
                _confhd_row(2, "B", 3, "NC"),
                _confhd_row(3, "C", 0, "EX"),
            ]
        )
        res = resolve_cascade(confhd, empty_cadastro, fictitious={99})
        assert res[1].downstream_code == 3
        assert res[1].fict_chain == (99,)

    def test_absent_followed_by_fict_walks_through_both(
        self,
        empty_cadastro: pd.DataFrame,
    ) -> None:
        # A (EX) -> B (NC) -> F (FICT.X) -> C (EX); A->C, F in fict_chain.
        confhd = _make_confhd(
            [
                _confhd_row(1, "A", 2, "EX"),
                _confhd_row(2, "B", 99, "NC"),
                _confhd_row(99, "FICT.X", 3, "EX"),
                _confhd_row(3, "C", 0, "EX"),
            ]
        )
        res = resolve_cascade(confhd, empty_cadastro, fictitious={99})
        assert res[1].downstream_code == 3
        assert res[1].fict_chain == (99,)


class TestExpansionPlantIsRealNode:
    """An ``EE`` plant is in service, so the cascade stops at it instead of
    rewiring its upstream to the plant below — which would drop it from the
    water balance while the source model still routes through it."""

    def test_ee_plant_between_two_real_is_not_bypassed(
        self,
        empty_cadastro: pd.DataFrame,
    ) -> None:
        # A (EX) -> B (EE) -> C (EX); A keeps B, B keeps C.
        confhd = _make_confhd(
            [
                _confhd_row(1, "A", 2, "EX"),
                _confhd_row(2, "B", 3, "EE"),
                _confhd_row(3, "C", 0, "EX"),
            ]
        )
        res = resolve_cascade(confhd, empty_cadastro)
        assert res[1].downstream_code == 2
        assert res[2].downstream_code == 3
        assert res[3].downstream_code is None
