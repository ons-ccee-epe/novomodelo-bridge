"""Shared min-over-planes production-surface (FPHA) envelope math.

Both the source model and Novomodelo fit a piecewise-linear production surface
GH(V, Q, S) as a set of hyperplanes; the LP consumes the lower envelope --
the minimum across planes. :func:`dense_grid` evaluates that envelope with
numpy over any broadcastable set of points: a whole meshgrid (the surface
comparison) or a scattered set of operating points. The source models scale
the whole plane by their correction factor; Novomodelo scales the intercept only
(:func:`novomodelo_plane_arrays`).

:func:`productivity_split` splits the gap between the two models' realized
productivity into a part due to the fitted surfaces and a part due to where
each model operates.
"""

from __future__ import annotations

import numpy as np
import polars as pl

#: Output schema of the per-(plant, stage) FPHA fidelity metrics frame (the
#: comparison conclusion: how close Novomodelo's fitted surface is to the source
#: model's, normalized to the plant's max generation).
FPHA_METRICS_SCHEMA: dict[str, type[pl.DataType]] = {
    "novomodelo_id": pl.Int64,
    "plant_name": pl.Utf8,
    "stage": pl.Int64,
    "n_planes_newave": pl.Int64,
    "n_planes_novomodelo": pl.Int64,
    "n_v": pl.Int64,
    "nmae": pl.Float64,
    "bias": pl.Float64,
    "max_abs_dev": pl.Float64,
    "gh_max_ratio": pl.Float64,
}

#: Output schema of the dense (V, Q) production-surface frame (the render
#: substrate for the heatmaps; one row per grid point per source).
FPHA_SURFACE_SCHEMA: dict[str, type[pl.DataType]] = {
    "novomodelo_id": pl.Int64,
    "plant_name": pl.Utf8,
    "stage": pl.Int64,
    "v_hm3": pl.Float64,
    "q_m3s": pl.Float64,
    "source": pl.Utf8,
    "gh_mw": pl.Float64,
}

#: Output schema of the spillage-slice frame (GH vs spill at the max V/Q corner).
FPHA_SPILL_SCHEMA: dict[str, type[pl.DataType]] = {
    "novomodelo_id": pl.Int64,
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
    correction (the source model's ``fator_correcao``, Novomodelo's ``kappa``).

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


def novomodelo_plane_arrays(
    planes: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Novomodelo's planes as :func:`dense_grid` arguments.

    Novomodelo's ``kappa`` scales the intercept only, so it is folded into
    ``gamma_0`` and the per-plane multiplier is one.
    """
    kappa = planes.get_column("kappa").to_numpy()
    return (
        planes.get_column("gamma_0").to_numpy() * kappa,
        planes.get_column("gamma_v").to_numpy(),
        planes.get_column("gamma_q").to_numpy(),
        planes.get_column("gamma_s").to_numpy(),
        np.ones_like(kappa),
    )


#: Output schema of :func:`productivity_split`, plus ``plant_name``.
FPHA_SPLIT_SCHEMA: dict[str, type[pl.DataType]] = {
    "novomodelo_id": pl.Int64,
    "plant_name": pl.Utf8,
    "stage": pl.Int64,
    "tie_share": pl.Float64,
    "q_source_m3s": pl.Float64,
    "gh_source_mw": pl.Float64,
    "rho_source": pl.Float64,
    "rho_novomodelo": pl.Float64,
    "rho_novomodelo_planes_at_source": pl.Float64,
    "rho_source_planes_at_source": pl.Float64,
    "rho_exact_at_source": pl.Float64,
    "rho_exact_at_novomodelo": pl.Float64,
    "d_total": pl.Float64,
    "d_model": pl.Float64,
    "d_operating_point": pl.Float64,
    "fit_error_source": pl.Float64,
    "fit_error_novomodelo": pl.Float64,
}

#: Turbined flow (weighted m³/s) below which a productivity is 0/0.
_SPLIT_TURBINED_EPS: float = 1.0e-6


def productivity_split(
    source_points: pl.DataFrame, novomodelo_points: pl.DataFrame
) -> pl.DataFrame:
    """Split each (plant, stage) realized-productivity gap at the operating points.

    *source_points* holds one row per source-model operating point with
    ``novomodelo_id``, ``stage``, ``weight`` (probability × hours), ``q_m3s``,
    ``gh_mw`` (realized), ``source_planes_mw`` and ``novomodelo_planes_mw`` (each
    model's envelope there), ``exact_mw`` and ``tie``; *novomodelo_points* the same
    for Novomodelo without the plane columns. Each productivity is a weighted
    ``Σ GH / Σ Q`` over the points that are not ties:

    - ``d_total = ρ_novomodelo / ρ_source − 1 = (1 + d_model)(1 + d_operating_point) − 1``;
    - ``d_model = ρ_novomodelo_planes_at_source / ρ_source − 1``, Novomodelo's surface
      against the source model's realized generation at the source's points;
    - ``d_operating_point = ρ_novomodelo / ρ_novomodelo_planes_at_source − 1``, where
      Novomodelo operates against where the source operates, under Novomodelo's surface;
    - ``fit_error_* = ρ_*_planes_at_source / ρ_exact_at_source − 1``, each
      surface against the exact production function at the source's points.

    ``tie_share`` is the weight share of the source's points that are ties;
    ``q_source_m3s`` and ``gh_source_mw`` are the source's weighted mean
    turbined flow and generation over the others. A productivity whose points
    turbine nothing is null.
    """

    def _rho(column: str) -> pl.Expr:
        q = (pl.col("weight") * pl.col("q_m3s")).filter(~pl.col("tie")).sum()
        g = (pl.col("weight") * pl.col(column)).filter(~pl.col("tie")).sum()
        return pl.when(q > _SPLIT_TURBINED_EPS).then(g / q).otherwise(None)

    def _untied_mean(column: str) -> pl.Expr:
        w = pl.col("weight").filter(~pl.col("tie")).sum()
        total = (pl.col("weight") * pl.col(column)).filter(~pl.col("tie")).sum()
        return pl.when(w > 0.0).then(total / w).otherwise(None)

    def _gap(numerator: str, denominator: str) -> pl.Expr:
        return (
            pl.when(pl.col(denominator) > 0.0)
            .then(pl.col(numerator) / pl.col(denominator) - 1.0)
            .otherwise(None)
        )

    keys = ["novomodelo_id", "stage"]
    source = source_points.group_by(keys).agg(
        (pl.col("weight").filter(pl.col("tie")).sum() / pl.col("weight").sum()).alias(
            "tie_share"
        ),
        _untied_mean("q_m3s").alias("q_source_m3s"),
        _untied_mean("gh_mw").alias("gh_source_mw"),
        _rho("gh_mw").alias("rho_source"),
        _rho("novomodelo_planes_mw").alias("rho_novomodelo_planes_at_source"),
        _rho("source_planes_mw").alias("rho_source_planes_at_source"),
        _rho("exact_mw").alias("rho_exact_at_source"),
    )
    novomodelo = novomodelo_points.group_by(keys).agg(
        _rho("gh_mw").alias("rho_novomodelo"),
        _rho("exact_mw").alias("rho_exact_at_novomodelo"),
    )
    return (
        source.join(novomodelo, on=keys, how="left")
        .with_columns(
            _gap("rho_novomodelo", "rho_source").alias("d_total"),
            _gap("rho_novomodelo_planes_at_source", "rho_source").alias("d_model"),
            _gap("rho_novomodelo", "rho_novomodelo_planes_at_source").alias(
                "d_operating_point"
            ),
            _gap("rho_source_planes_at_source", "rho_exact_at_source").alias(
                "fit_error_source"
            ),
            _gap("rho_novomodelo_planes_at_source", "rho_exact_at_source").alias(
                "fit_error_novomodelo"
            ),
        )
        .with_columns(pl.lit(None, dtype=pl.Utf8).alias("plant_name"))
        .select(list(FPHA_SPLIT_SCHEMA))
        .cast(FPHA_SPLIT_SCHEMA)
        .sort(keys)
    )
