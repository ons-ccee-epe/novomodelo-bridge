"""Tests for ``comparators.operating_points``, the track-neutral core both
tracks' operating-point builders compose; each track's conventions are
tested with its builder."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from novomodelo_bridge.comparators.operating_points import (
    PLANE_SCHEMA,
    FitWindow,
    Hydraulics,
    OperatingPointComparison,
    StagePhysics,
    head,
    operating_point_fields,
    source_envelope,
    window_gh_max,
)
from novomodelo_bridge.core.tailrace import TailraceFamily, TailraceSegment

_LINEAR_TAILRACE = (
    TailraceFamily(
        downstream_reference_level_m=math.nan,
        segments=(
            TailraceSegment(
                outflow_max_m3s=math.inf, coefficients=(100.0, 0.01, 0.0, 0.0, 0.0)
            ),
        ),
    ),
)
_PHYSICS = StagePhysics(
    cota=(200.0, 0.0, 0.0, 0.0, 0.0),
    rho_esp=0.009,
    losses=0.0,
    max_generation_mw=math.inf,
)


def _hydraulics(*, spill_in_tailrace: bool) -> Hydraulics:
    return Hydraulics(
        families=_LINEAR_TAILRACE,
        spill_in_tailrace=spill_in_tailrace,
        tipo_perda=0,
        volume_floor_hm3=50.0,
    )


class TestHead:
    def test_spill_raises_the_tailrace_only_where_it_reaches_it(self) -> None:
        def _tailrace(spill_in_tailrace: bool) -> float:
            return head(
                _hydraulics(spill_in_tailrace=spill_in_tailrace),
                _PHYSICS,
                v=60.0,
                q=100.0,
                s=400.0,
                lateral_m3s=0.0,
                downstream_level_m=None,
            ).tailrace_m

        assert _tailrace(False) == pytest.approx(101.0)
        assert _tailrace(True) == pytest.approx(105.0)

    def test_generation_is_capped_at_the_maximum(self) -> None:
        capped = StagePhysics(
            cota=_PHYSICS.cota, rho_esp=0.009, losses=0.0, max_generation_mw=50.0
        )

        point = head(
            _hydraulics(spill_in_tailrace=False),
            capped,
            v=60.0,
            q=100.0,
            s=0.0,
            lateral_m3s=0.0,
            downstream_level_m=None,
        )

        assert point.net_head_m == pytest.approx(99.0)
        assert point.generation_mw == 50.0


class TestSourceEnvelope:
    def test_is_the_lowest_plane_with_volume_above_the_floor(self) -> None:
        planes = pl.DataFrame(
            {
                "gamma_0": [0.0, 30.0],
                "gamma_v": [1.0, 0.0],
                "gamma_q": [0.5, 0.2],
                "gamma_s": [0.0, 0.0],
                "gamma_lat": [0.0, 0.0],
                "fator_correcao": [0.9, 0.9],
            },
            schema=PLANE_SCHEMA,
        )

        envelope = source_envelope(
            planes,
            v_useful=np.array([10.0]),
            q=np.array([100.0]),
            s=np.array([0.0]),
            lateral=np.array([0.0]),
        )

        assert envelope.tolist() == pytest.approx([0.9 * 50.0])


class TestWindowGhMax:
    def test_a_missing_window_or_cap_is_unbounded(self) -> None:
        windows = {
            (1, 0): FitWindow(100.0, 200.0, 50.0, "5 × 5", 80.0),
            (1, 1): FitWindow(100.0, 200.0, 50.0, "5 × 5", None),
        }

        assert window_gh_max(windows, 1, 0) == 80.0
        assert window_gh_max(windows, 1, 1) == math.inf
        assert window_gh_max(windows, 2, 0) == math.inf


class TestOperatingPointFields:
    def test_empty_surfaces_are_none_and_point_frames_pass_through(self) -> None:
        fields = operating_point_fields(OperatingPointComparison())

        assert fields["fpha_metrics"] is None
        assert fields["fpha_surface"] is None
        assert fields["fpha_spill"] is None
        assert isinstance(fields["fpha_split"], pl.DataFrame)
        assert isinstance(fields["hydro_masked"], pl.DataFrame)

    def test_no_comparison_fills_nothing(self) -> None:
        assert operating_point_fields(None) == {}
