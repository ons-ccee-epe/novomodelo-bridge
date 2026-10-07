"""Tests for the boundary-FCF long-term-mean inflow incrementalization
(``decomp/inflow_mlt.py``): the ``mu`` the cut RHS fold subtracts so the loaded
cut prices the raw inflow-lag state as the deviation from the seasonal mean.

Tier 1 — pure Python; ``_incremental_context`` (the posto/parent topology, its
own module's concern) is patched so these tests pin the natural→incremental
subtraction and the lag-depth alignment.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.inflow_mlt import (
    build_incremental_mlt,
    coupling_lag_means,
)


def _id_map(hydro_codes: tuple[int, ...]) -> DecompIdMap:
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=hydro_codes)


def _mlt(columns: dict[str, list[float]]) -> pd.DataFrame:
    """A synthetic ``mlt.dat`` frame: ``mes`` 1..12 plus per-posto columns."""
    return pd.DataFrame({"mes": list(range(1, 13)), **columns})


def _build(mlt: pd.DataFrame, context) -> dict[int, dict[int, float]]:
    with patch(
        "novomodelo_bridge.decomp.inflow_mlt._incremental_context", return_value=context
    ):
        return build_incremental_mlt(mlt, object(), _id_map(tuple(context[0])))


class TestBuildIncrementalMlt:
    def test_subtracts_upstream_posto(self) -> None:
        # Plant 10 (posto 1) is upstream of plant 20 (posto 2). The downstream
        # increment is its natural minus the upstream station's natural.
        mlt = _mlt({"1": [100.0] * 12, "2": [170.0] * 12})
        context = ({10: "1", 20: "2"}, {10: [], 20: [10]})
        out = _build(mlt, context)
        assert out[0][1] == pytest.approx(100.0)  # headwater: natural == incr
        assert out[1][1] == pytest.approx(70.0)  # 170 - 100

    def test_zero_posto_pinned_to_zero(self) -> None:
        # A diversion-fed plant on an artificial zero-natural station has zero
        # incremental inflow — never a spurious negative from subtracting a real
        # upstream station off the zero base.
        mlt = _mlt({"1": [100.0] * 12, "300": [0.0] * 12})
        context = ({10: "1", 20: "300"}, {10: [], 20: [10]})
        out = _build(mlt, context)
        assert all(value == 0.0 for value in out[1].values())

    def test_shared_upstream_posto_subtracted_once(self) -> None:
        # Two upstream plants share one station; the downstream increment
        # subtracts that station's natural exactly once (posto-space dedup).
        mlt = _mlt({"1": [100.0] * 12, "2": [250.0] * 12})
        context = ({10: "1", 11: "1", 20: "2"}, {10: [], 11: [], 20: [10, 11]})
        out = _build(mlt, context)
        assert out[2][1] == pytest.approx(150.0)  # 250 - 100 (not 250 - 200)

    def test_self_loop_upstream_skipped(self) -> None:
        # A plant sharing its upstream sibling's station forms no posto edge
        # (self-loop), so no subtraction happens.
        mlt = _mlt({"1": [100.0] * 12})
        context = ({10: "1", 20: "1"}, {10: [], 20: [10]})
        out = _build(mlt, context)
        assert out[1][1] == pytest.approx(100.0)  # unchanged (no self-subtract)

    def test_missing_posto_omitted(self) -> None:
        # A plant whose station is absent from the MLT is left out entirely
        # (its fold defaults to mu = 0 downstream).
        mlt = _mlt({"1": [100.0] * 12})
        context = ({10: "1", 20: "999"}, {10: [], 20: []})
        out = _build(mlt, context)
        assert set(out) == {0}  # only plant 10 (hydro_id 0)

    def test_genuine_negative_increment_preserved(self) -> None:
        # A net-evaporative reach (downstream natural below upstream) yields a
        # real negative increment — not clamped.
        mlt = _mlt({"1": [100.0] * 12, "2": [78.0] * 12})
        context = ({10: "1", 20: "2"}, {10: [], 20: [10]})
        out = _build(mlt, context)
        assert out[1][1] == pytest.approx(-22.0)


class TestCouplingLagMeans:
    def test_aligns_depth_to_calendar_month(self) -> None:
        # At an April (month 4) coupling: depth 1 -> March (3), depth 2 -> Feb
        # (2), …, depth 12 -> April prev year (4).
        incremental = {0: {month: float(month) for month in range(1, 13)}}
        aligned = coupling_lag_means(incremental, coupling_month=4)
        assert aligned[0] == (
            3.0,  # depth 1 -> March
            2.0,  # depth 2 -> Feb
            1.0,  # depth 3 -> Jan
            12.0,  # depth 4 -> Dec
            11.0,
            10.0,
            9.0,
            8.0,
            7.0,
            6.0,
            5.0,  # depth 11 -> May
            4.0,  # depth 12 -> April (prev year)
        )

    def test_wraps_january_coupling(self) -> None:
        # A January (month 1) coupling: depth 1 -> Dec (12), depth 12 -> Jan.
        incremental = {0: {month: float(month) for month in range(1, 13)}}
        aligned = coupling_lag_means(incremental, coupling_month=1)
        assert aligned[0][0] == 12.0  # depth 1 -> December
        assert aligned[0][11] == 1.0  # depth 12 -> January (prev year)
