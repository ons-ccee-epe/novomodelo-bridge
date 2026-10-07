"""Shared min-over-planes production-surface (FPHA) envelope math.

Both the source model and Cobre fit a piecewise-linear production surface
GH(V, Q, S) as a set of hyperplanes; the LP consumes the lower envelope --
the minimum across planes -- of ``multiplier * (gamma_0 + gamma_v * (v -
volume_offset) + gamma_q * q + gamma_s * s)``. :func:`dense_grid` evaluates
that envelope with numpy over a whole meshgrid at once (the NEWAVE track,
which has fitted planes on both sides and compares a dense (V, Q) surface).
:func:`point_cloud` evaluates the identical contract with polars over a
scattered set of operating points (the DECOMP track, which has only Cobre's
fitted planes plus the source model's realized trajectory -- no source-side
plane coefficients and no fitting grid to build a dense surface from).

The two tracks' fidelity metrics (``nmae``/``bias`` in
:data:`FPHA_METRICS_SCHEMA`) normalize to different denominators, and this is
not a discrepancy to unify: NEWAVE normalizes to the fitted grid-corner max
(the source model's own envelope value at the max-V/max-Q grid corner);
DECOMP normalizes to the realized-trajectory max (the largest generation the
source model actually realized over the simulated operating points), because
it has no fitted grid to take a corner from in the first place. The two
``nmae``/``bias`` values share column names but are **not cross-track
comparable** -- each is only meaningful measured against its own track's
denominator.
"""

from __future__ import annotations

import numpy as np
import polars as pl

#: Output schema of the per-(plant, stage) FPHA fidelity metrics frame (the
#: comparison conclusion: how close Cobre's fitted surface is to the source
#: model's, normalized to the plant's max generation).
FPHA_METRICS_SCHEMA: dict[str, type[pl.DataType]] = {
    "cobre_id": pl.Int64,
    "plant_name": pl.Utf8,
    "stage": pl.Int64,
    "n_planes_newave": pl.Int64,
    "n_planes_cobre": pl.Int64,
    "n_v": pl.Int64,
    "nmae": pl.Float64,
    "bias": pl.Float64,
    "max_abs_dev": pl.Float64,
    "gh_max_ratio": pl.Float64,
}

#: Output schema of the dense (V, Q) production-surface frame (the render
#: substrate for the heatmaps; one row per grid point per source).
FPHA_SURFACE_SCHEMA: dict[str, type[pl.DataType]] = {
    "cobre_id": pl.Int64,
    "plant_name": pl.Utf8,
    "stage": pl.Int64,
    "v_hm3": pl.Float64,
    "q_m3s": pl.Float64,
    "source": pl.Utf8,
    "gh_mw": pl.Float64,
}

#: Output schema of the spillage-slice frame (GH vs spill at the max V/Q corner).
FPHA_SPILL_SCHEMA: dict[str, type[pl.DataType]] = {
    "cobre_id": pl.Int64,
    "plant_name": pl.Utf8,
    "stage": pl.Int64,
    "s_m3s": pl.Float64,
    "source": pl.Utf8,
    "gh_mw": pl.Float64,
}


def dense_grid(
    gamma_0: np.ndarray,
    gamma_v: np.ndarray,
    gamma_q: np.ndarray,
    gamma_s: np.ndarray,
    multiplier: np.ndarray,
    v: np.ndarray,
    q: np.ndarray,
    s: np.ndarray,
    *,
    volume_offset: float = 0.0,
) -> np.ndarray:
    """Evaluate a min-over-planes FPHA envelope on a grid of operating points.

    Each plane contributes ``multiplier * (gamma_0 + gamma_v * (v -
    volume_offset) + gamma_q * q + gamma_s * s)``; the envelope is the minimum
    across planes — the value the operating model's LP consumes. ``v``, ``q``,
    ``s`` are broadcast together, so passing a meshgrid evaluates a whole
    surface in one call.

    ``volume_offset`` subtracts the plant minimum storage so a useful-volume
    coefficient is applied to absolute volume; pass ``0.0`` for a coefficient
    that already multiplies absolute volume. ``multiplier`` is the per-plane
    correction (the source model's ``fator_correcao``, Cobre's ``kappa``).

    Args:
        gamma_0: Per-plane constant term, shape ``(P,)``.
        gamma_v: Per-plane volume coefficient, shape ``(P,)``.
        gamma_q: Per-plane turbined-flow coefficient, shape ``(P,)``.
        gamma_s: Per-plane spilled-flow coefficient, shape ``(P,)``.
        multiplier: Per-plane scalar multiplier, shape ``(P,)``.
        v: Volume coordinate(s), broadcastable with ``q`` and ``s``.
        q: Turbined-flow coordinate(s), broadcastable with ``v`` and ``s``.
        s: Spilled-flow coordinate(s), broadcastable with ``v`` and ``q``.
        volume_offset: Storage subtracted from ``v`` before applying ``gamma_v``.

    Returns:
        The envelope generation, shaped like the broadcast of ``v``/``q``/``s``.
    """
    shape = np.broadcast_shapes(np.shape(v), np.shape(q), np.shape(s))
    vv = np.broadcast_to(v, shape).reshape(-1)
    qq = np.broadcast_to(q, shape).reshape(-1)
    ss = np.broadcast_to(s, shape).reshape(-1)
    useful_v = vv - volume_offset
    planes = multiplier[:, None] * (
        gamma_0[:, None]
        + gamma_v[:, None] * useful_v[None, :]
        + gamma_q[:, None] * qq[None, :]
        + gamma_s[:, None] * ss[None, :]
    )
    return planes.min(axis=0).reshape(shape)


def point_cloud(
    planes: pl.DataFrame, points: pl.DataFrame, *, volume_offset: float = 0.0
) -> pl.DataFrame:
    """Evaluate a min-over-planes FPHA envelope at a scattered set of
    (hydro, stage, v, q, s) operating points.

    ``points`` carries one row per sample, with columns ``_point_id`` (a
    stable row identity), ``cobre_id``, ``stage``, ``v_hm3``, ``q_m3s``,
    ``s_m3s``. ``planes`` is a Cobre-shaped ``hydro_id``/``stage_id``/
    ``gamma_0``/``gamma_v``/``gamma_q``/``gamma_s``/``kappa`` frame (multiple
    plane rows per (hydro_id, stage_id)) — the polars counterpart of
    :func:`dense_grid`'s ``multiplier * (gamma_0 + gamma_v * (v -
    volume_offset) + gamma_q * q + gamma_s * s)`` contract, expressed as a
    join + group-by rather than a numpy broadcast since this evaluates a
    scattered point cloud, not a dense grid.

    ``volume_offset`` follows :func:`dense_grid`'s convention (subtract the
    plant's minimum storage before applying ``gamma_v``); pass ``0.0`` (the
    default) when ``gamma_v`` already multiplies absolute volume.

    Returns one row per ``_point_id`` with the envelope value in
    ``cobre_gh_mw``; a point whose (hydro, stage) has no matching plane rows
    drops out of the join rather than null-keeping.
    """
    keyed_planes = planes.rename({"hydro_id": "cobre_id", "stage_id": "stage"}).cast(
        {"cobre_id": pl.Int64, "stage": pl.Int64}
    )
    joined = points.join(keyed_planes, on=["cobre_id", "stage"], how="inner")
    if joined.is_empty():
        return pl.DataFrame(schema={"_point_id": pl.Int64, "cobre_gh_mw": pl.Float64})
    return (
        joined.with_columns(
            (
                pl.col("kappa")
                * (
                    pl.col("gamma_0")
                    + pl.col("gamma_v") * (pl.col("v_hm3") - volume_offset)
                    + pl.col("gamma_q") * pl.col("q_m3s")
                    + pl.col("gamma_s") * pl.col("s_m3s")
                )
            ).alias("plane_value")
        )
        .group_by("_point_id")
        .agg(pl.col("plane_value").min().alias("cobre_gh_mw"))
    )
