"""Tier-1 tests for the RHE stored-energy share helpers.

A synthetic ``EZ`` frame double and hand-built shares only; the share builder
itself is exercised through the emitter in ``test_rhe.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from novomodelo_bridge.core.stored_energy import cascade_share
from novomodelo_bridge.decomp.converters.rhe_shares import (
    RheShare,
    ez_cap_unmodeled_stages,
    read_ez_fractions,
)


class _EzDadger:
    def __init__(self, frame: pd.DataFrame | None) -> None:
        self._frame = frame

    def ez(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
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
