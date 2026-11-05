"""Tests for the shared FPHA math (``comparators/fpha.py``).

Covers the envelope evaluator ``dense_grid``, Novomodelo's plane read-back
``novomodelo_plane_arrays``, the operating-point ``productivity_split``, and the
shared schemas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
import pytest

from novomodelo_bridge.comparators.fpha import (
    FPHA_METRICS_SCHEMA,
    FPHA_SPILL_SCHEMA,
    FPHA_SPLIT_SCHEMA,
    FPHA_SURFACE_SCHEMA,
    dense_grid,
    novomodelo_plane_arrays,
    productivity_split,
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
# novomodelo_plane_arrays                                                           #
# --------------------------------------------------------------------------- #


def test_novomodelo_plane_arrays_scales_only_the_intercept_by_kappa() -> None:
    planes = pl.DataFrame(
        {
            "gamma_0": [10.0],
            "gamma_v": [0.2],
            "gamma_q": [0.5],
            "gamma_s": [-0.1],
            "kappa": [0.9],
        }
    )
    gamma_0, gamma_v, gamma_q, gamma_s, mult = novomodelo_plane_arrays(planes)
    assert gamma_0.tolist() == pytest.approx([9.0])
    assert gamma_v.tolist() == [0.2]
    assert gamma_q.tolist() == [0.5]
    assert gamma_s.tolist() == [-0.1]
    assert mult.tolist() == [1.0]


# --------------------------------------------------------------------------- #
# productivity_split                                                           #
# --------------------------------------------------------------------------- #


def _split_points(
    rows: Sequence[Mapping[str, object]], *, planes: bool
) -> pl.DataFrame:
    schema: dict[str, type[pl.DataType]] = {
        "novomodelo_id": pl.Int64,
        "stage": pl.Int64,
        "weight": pl.Float64,
        "q_m3s": pl.Float64,
        "gh_mw": pl.Float64,
        "exact_mw": pl.Float64,
        "tie": pl.Boolean,
    }
    if planes:
        schema |= {"source_planes_mw": pl.Float64, "novomodelo_planes_mw": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


def test_productivity_split_composes_model_and_operating_point() -> None:
    # Source: realized 0.5 MW per m³/s, its planes 0.5, Novomodelo's planes 0.55,
    # exact 0.5. Novomodelo realizes 0.605 at its own points.
    source = _split_points(
        [
            {
                "novomodelo_id": 1,
                "stage": 0,
                "weight": 1.0,
                "q_m3s": 100.0,
                "gh_mw": 50.0,
                "source_planes_mw": 50.0,
                "novomodelo_planes_mw": 55.0,
                "exact_mw": 50.0,
                "tie": False,
            }
        ],
        planes=True,
    )
    novomodelo = _split_points(
        [
            {
                "novomodelo_id": 1,
                "stage": 0,
                "weight": 1.0,
                "q_m3s": 100.0,
                "gh_mw": 60.5,
                "exact_mw": 55.0,
                "tie": False,
            }
        ],
        planes=False,
    )
    row = productivity_split(source, novomodelo).row(0, named=True)
    assert row["d_model"] == pytest.approx(0.10)
    assert row["d_operating_point"] == pytest.approx(0.10)
    assert row["d_total"] == pytest.approx(0.21)
    assert (1 + row["d_model"]) * (1 + row["d_operating_point"]) - 1 == (
        pytest.approx(row["d_total"])
    )
    assert row["fit_error_source"] == pytest.approx(0.0)
    assert row["fit_error_novomodelo"] == pytest.approx(0.10)
    assert row["rho_exact_at_novomodelo"] == pytest.approx(0.55)
    assert row["q_source_m3s"] == pytest.approx(100.0)
    assert row["gh_source_mw"] == pytest.approx(50.0)
    assert row["tie_share"] == pytest.approx(0.0)
    assert list(productivity_split(source, novomodelo).columns) == list(
        FPHA_SPLIT_SCHEMA
    )


def test_productivity_split_leaves_ties_out_and_reports_their_share() -> None:
    base = {
        "novomodelo_id": 1,
        "stage": 0,
        "q_m3s": 100.0,
        "source_planes_mw": 50.0,
        "novomodelo_planes_mw": 50.0,
        "exact_mw": 50.0,
    }
    source = _split_points(
        [
            {**base, "weight": 3.0, "gh_mw": 50.0, "tie": False},
            # A tie point with a wildly different realized generation.
            {**base, "weight": 1.0, "gh_mw": 5.0, "tie": True},
        ],
        planes=True,
    )
    novomodelo = _split_points(
        [
            {
                "novomodelo_id": 1,
                "stage": 0,
                "weight": 1.0,
                "q_m3s": 100.0,
                "gh_mw": 50.0,
                "exact_mw": 50.0,
                "tie": False,
            }
        ],
        planes=False,
    )
    row = productivity_split(source, novomodelo).row(0, named=True)
    assert row["rho_source"] == pytest.approx(0.5)
    assert row["tie_share"] == pytest.approx(0.25)
    assert row["d_total"] == pytest.approx(0.0)


def test_productivity_split_nulls_a_stage_that_turbines_nothing() -> None:
    zero = {
        "novomodelo_id": 1,
        "stage": 0,
        "weight": 1.0,
        "q_m3s": 0.0,
        "gh_mw": 0.0,
        "exact_mw": 0.0,
        "tie": False,
    }
    source = _split_points(
        [{**zero, "source_planes_mw": 0.0, "novomodelo_planes_mw": 0.0}], planes=True
    )
    novomodelo = _split_points([zero], planes=False)
    row = productivity_split(source, novomodelo).row(0, named=True)
    assert row["rho_source"] is None
    assert row["d_total"] is None


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
