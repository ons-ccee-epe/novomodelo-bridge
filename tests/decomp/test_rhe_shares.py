"""Tier-1 tests for the RHE stored-energy share helpers.

A synthetic ``EZ`` frame double, hand-built shares and a synthetic cadastro
for the energy chain; the share builder itself is exercised through the
emitter in ``test_rhe.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from novomodelo_bridge.core.stored_energy import cascade_share
from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
from novomodelo_bridge.decomp.converters.rhe_shares import (
    RheShare,
    energy_chain_next,
    ez_cap_unmodeled_stages,
    read_ez_fractions,
)
from tests.decomp.test_rhe import _hidr_frame, _hydro_row


class _EzDadger:
    def __init__(self, frame: pd.DataFrame | None) -> None:
        self._frame = frame

    def ez(self, df: bool = False) -> pd.DataFrame | None:
        return self._frame


def _ez_frame(rows: list[tuple[int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["codigo_usina", "volume"])


def test_read_ez_fractions_converts_percent_to_fraction() -> None:
    dadger = _EzDadger(_ez_frame([(251, 55.0), (257, 100.0)]))

    assert read_ez_fractions(dadger) == {251: pytest.approx(0.55), 257: 1.0}


def test_read_ez_fractions_without_records_is_empty() -> None:
    assert read_ez_fractions(_EzDadger(None)) == {}
    assert read_ez_fractions(_EzDadger(_ez_frame([]))) == {}


def test_read_ez_fractions_rejects_a_percentage_above_100() -> None:
    with pytest.raises(ValueError, match="plant 251 gives 120.0%"):
        read_ez_fractions(_EzDadger(_ez_frame([(251, 120.0)])))


def _share(hydro_id: int, rho: float, useful: float, ez: float = 1.0) -> RheShare:
    return RheShare(cascade_share(hydro_id), (rho,), useful, ez)


def test_ez_cap_unmodeled_stages_without_a_capped_share_is_empty() -> None:
    shares = [(1.0, _share(0, 1.0, 100.0))]

    assert ez_cap_unmodeled_stages(shares, {0: 1e9}) == []


def test_ez_cap_unmodeled_stages_names_stages_whose_floor_exceeds_the_cap() -> None:
    shares = [(1.0, _share(0, 2.0, 100.0, ez=0.5)), (1.0, _share(1, 1.0, 1000.0))]

    # The capped share stores at most 2.0 * 100 * 0.5 = 100 MWh.
    assert ez_cap_unmodeled_stages(shares, {0: 100.0}) == []
    assert ez_cap_unmodeled_stages(shares, {0: 100.5}) == [0]


def test_ez_cap_unmodeled_stages_never_certifies_a_negative_share() -> None:
    shares = [(1.0, _share(0, 2.0, 100.0, ez=0.5)), (-1.0, _share(1, 1.0, 10.0))]

    assert ez_cap_unmodeled_stages(shares, {0: 0.0}) == [0]


def _chain_cadastro(
    energy_links: dict[int, tuple[int | None, ...]],
) -> EffectiveCadastro:
    """UP (1) -> GAP (3, not operated) -> DOWN (5); AGG (9, not operated)."""
    return EffectiveCadastro(
        base=_hidr_frame(
            {
                1: _hydro_row(tipo_regulacao="M", codigo_usina_jusante=3),
                3: _hydro_row(tipo_regulacao="D", codigo_usina_jusante=5),
                5: _hydro_row(tipo_regulacao="M", codigo_usina_jusante=0),
                9: _hydro_row(tipo_regulacao="D", codigo_usina_jusante=5),
            }
        ),
        n_stages=1,
        stage_varying={},
        energy_links=energy_links,
    )


def test_energy_chain_next_without_a_link_is_the_next_operated_plant() -> None:
    effective = _chain_cadastro({})

    assert energy_chain_next(effective, 1, {1, 5}) == 5
    assert energy_chain_next(effective, 5, {1, 5}) is None


def test_energy_chain_next_follows_a_link_to_a_plant_outside_the_configuration() -> (
    None
):
    assert energy_chain_next(_chain_cadastro({1: (9,)}), 1, {1, 5}) == 9


def test_energy_chain_next_link_to_the_sink_ends_the_chain() -> None:
    assert energy_chain_next(_chain_cadastro({1: (0,)}), 1, {1, 5}) is None


def test_energy_chain_next_rejects_a_link_outside_hidr() -> None:
    with pytest.raises(ValueError, match="plant 1's AC JUSENA names plant 77"):
        energy_chain_next(_chain_cadastro({1: (77,)}), 1, {1, 5})
