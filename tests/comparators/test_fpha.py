"""Tests for the shared FPHA envelope math (``comparators/fpha.py``).

Covers the two entry points -- ``dense_grid`` (numpy, whole-meshgrid) and
``point_cloud`` (polars, scattered samples) -- and the three shared schemas
both the NEWAVE and DECOMP comparison tracks import.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from novomodelo_bridge.comparators.fpha import (
    FPHA_METRICS_SCHEMA,
    FPHA_SPILL_SCHEMA,
    FPHA_SURFACE_SCHEMA,
    dense_grid,
    point_cloud,
)

# --------------------------------------------------------------------------- #
# dense_grid                                                                   #
# --------------------------------------------------------------------------- #


def test_dense_grid_takes_min_over_planes() -> None:
    # Plane A: GH = q ; plane B: GH = 10 (constant). Envelope = min(q, 10).
    gamma_0 = np.array([0.0, 10.0])
    gamma_v = np.array([0.0, 0.0])
    gamma_q = np.array([1.0, 0.0])
    gamma_s = np.array([0.0, 0.0])
    mult = np.array([1.0, 1.0])
    q = np.array([5.0, 20.0])
    zeros = np.zeros_like(q)

    out = dense_grid(gamma_0, gamma_v, gamma_q, gamma_s, mult, zeros, q, zeros)
    assert out.tolist() == [5.0, 10.0]


def test_dense_grid_applies_volume_offset() -> None:
    # GH = 0.1 * useful_volume, useful = v - offset. At v=110, offset=10 -> 10.0.
    out = dense_grid(
        np.array([0.0]),
        np.array([0.1]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([110.0]),
        np.array([0.0]),
        np.array([0.0]),
        volume_offset=10.0,
    )
    assert out[0] == pytest.approx(10.0)


def test_dense_grid_default_volume_offset_is_zero() -> None:
    # No offset passed -> the volume coefficient multiplies absolute volume.
    out = dense_grid(
        np.array([0.0]),
        np.array([0.1]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([110.0]),
        np.array([0.0]),
        np.array([0.0]),
    )
    assert out[0] == pytest.approx(11.0)


# --------------------------------------------------------------------------- #
# point_cloud                                                                  #
# --------------------------------------------------------------------------- #


def _planes(rows: list[dict[str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "hydro_id": pl.Int64,
            "stage_id": pl.Int64,
            "gamma_0": pl.Float64,
            "gamma_v": pl.Float64,
            "gamma_q": pl.Float64,
            "gamma_s": pl.Float64,
            "kappa": pl.Float64,
        },
    )


def _points(rows: list[dict[str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "_point_id": pl.Int64,
            "novomodelo_id": pl.Int64,
            "stage": pl.Int64,
            "v_hm3": pl.Float64,
            "q_m3s": pl.Float64,
            "s_m3s": pl.Float64,
        },
    )


def test_point_cloud_takes_min_over_planes() -> None:
    # Plane A: GH = q ; plane B: GH = 10 (constant). Envelope = min(q, 10).
    planes = _planes(
        [
            {
                "hydro_id": 0,
                "stage_id": 0,
                "gamma_0": 0.0,
                "gamma_v": 0.0,
                "gamma_q": 1.0,
                "gamma_s": 0.0,
                "kappa": 1.0,
            },
            {
                "hydro_id": 0,
                "stage_id": 0,
                "gamma_0": 10.0,
                "gamma_v": 0.0,
                "gamma_q": 0.0,
                "gamma_s": 0.0,
                "kappa": 1.0,
            },
        ]
    )
    points = _points(
        [
            {
                "_point_id": 0,
                "novomodelo_id": 0,
                "stage": 0,
                "v_hm3": 0.0,
                "q_m3s": 5.0,
                "s_m3s": 0.0,
            },
            {
                "_point_id": 1,
                "novomodelo_id": 0,
                "stage": 0,
                "v_hm3": 0.0,
                "q_m3s": 20.0,
                "s_m3s": 0.0,
            },
        ]
    )

    out = point_cloud(planes, points).sort("_point_id")
    assert out["novomodelo_gh_mw"].to_list() == [5.0, 10.0]


def test_point_cloud_applies_volume_offset() -> None:
    # GH = 0.1 * useful_volume, useful = v - offset. At v=110, offset=10 -> 10.0.
    planes = _planes(
        [
            {
                "hydro_id": 0,
                "stage_id": 0,
                "gamma_0": 0.0,
                "gamma_v": 0.1,
                "gamma_q": 0.0,
                "gamma_s": 0.0,
                "kappa": 1.0,
            }
        ]
    )
    points = _points(
        [
            {
                "_point_id": 0,
                "novomodelo_id": 0,
                "stage": 0,
                "v_hm3": 110.0,
                "q_m3s": 0.0,
                "s_m3s": 0.0,
            }
        ]
    )

    out = point_cloud(planes, points, volume_offset=10.0)
    assert out["novomodelo_gh_mw"].to_list() == pytest.approx([10.0])


def test_point_cloud_default_volume_offset_is_zero() -> None:
    # No offset passed -> the volume coefficient multiplies absolute volume,
    # matching the DECOMP caller's convention.
    planes = _planes(
        [
            {
                "hydro_id": 0,
                "stage_id": 0,
                "gamma_0": 0.0,
                "gamma_v": 0.1,
                "gamma_q": 0.0,
                "gamma_s": 0.0,
                "kappa": 1.0,
            }
        ]
    )
    points = _points(
        [
            {
                "_point_id": 0,
                "novomodelo_id": 0,
                "stage": 0,
                "v_hm3": 110.0,
                "q_m3s": 0.0,
                "s_m3s": 0.0,
            }
        ]
    )

    out = point_cloud(planes, points)
    assert out["novomodelo_gh_mw"].to_list() == pytest.approx([11.0])


def test_point_cloud_point_with_no_matching_plane_drops_out() -> None:
    planes = _planes(
        [
            {
                "hydro_id": 0,
                "stage_id": 0,
                "gamma_0": 1.0,
                "gamma_v": 0.0,
                "gamma_q": 0.0,
                "gamma_s": 0.0,
                "kappa": 1.0,
            }
        ]
    )
    points = _points(
        [
            {
                "_point_id": 0,
                "novomodelo_id": 0,
                "stage": 0,
                "v_hm3": 0.0,
                "q_m3s": 0.0,
                "s_m3s": 0.0,
            },
            # No (hydro_id=1, stage_id=0) plane exists.
            {
                "_point_id": 1,
                "novomodelo_id": 1,
                "stage": 0,
                "v_hm3": 0.0,
                "q_m3s": 0.0,
                "s_m3s": 0.0,
            },
        ]
    )

    out = point_cloud(planes, points)
    assert out["_point_id"].to_list() == [0]


def test_point_cloud_returns_empty_typed_frame_when_no_planes_match() -> None:
    planes = _planes(
        [
            {
                "hydro_id": 9,
                "stage_id": 0,
                "gamma_0": 1.0,
                "gamma_v": 0.0,
                "gamma_q": 0.0,
                "gamma_s": 0.0,
                "kappa": 1.0,
            }
        ]
    )
    points = _points(
        [
            {
                "_point_id": 0,
                "novomodelo_id": 0,
                "stage": 0,
                "v_hm3": 0.0,
                "q_m3s": 0.0,
                "s_m3s": 0.0,
            }
        ]
    )

    out = point_cloud(planes, points)
    assert out.is_empty()
    assert out.schema == {"_point_id": pl.Int64, "novomodelo_gh_mw": pl.Float64}


# --------------------------------------------------------------------------- #
# Schemas                                                                      #
# --------------------------------------------------------------------------- #


def test_fpha_metrics_schema_columns() -> None:
    assert list(FPHA_METRICS_SCHEMA) == [
        "novomodelo_id",
        "plant_name",
        "stage",
        "n_planes_newave",
        "n_planes_novomodelo",
        "n_v",
        "nmae",
        "bias",
        "max_abs_dev",
        "gh_max_ratio",
    ]


def test_fpha_surface_schema_columns() -> None:
    assert list(FPHA_SURFACE_SCHEMA) == [
        "novomodelo_id",
        "plant_name",
        "stage",
        "v_hm3",
        "q_m3s",
        "source",
        "gh_mw",
    ]


def test_fpha_spill_schema_columns() -> None:
    assert list(FPHA_SPILL_SCHEMA) == [
        "novomodelo_id",
        "plant_name",
        "stage",
        "s_m3s",
        "source",
        "gh_mw",
    ]
