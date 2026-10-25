"""Both models compared at the per-block operating points each one reached.

The source model's planes, Novomodelo's planes and the plant's exact production
function are evaluated at both models' operating points, so the gap between
the two realized productivities splits into a part due to the fitted surfaces
and a part due to where each model operates
(:func:`~novomodelo_bridge.comparators.fpha.productivity_split`). Each track reads
its source model's points and production model into the shapes this module
takes (:mod:`novomodelo_bridge.comparators.decomp.operating_points`,
:mod:`novomodelo_bridge.comparators.newave.operating_points`).

A point frame holds one row per plant and :data:`POINT_KEYS`: ``code`` (the
source plant code), ``novomodelo_id``, ``weight`` (probability × hours), ``v_hm3``
(the storage the production function reads), ``q_m3s``, ``s_m3s``,
``outflow_m3s``, ``gh_mw`` and ``worthless``. Storage, flows and generation
are each model's own; the source model's FPHA storage is its stage mean, and
Novomodelo's is the mean of its initial and final storage.

The exact production function is ``min(ρ_esp · Q · h_net, G_max)``
(:func:`head`). The forebay comes from the cota polynomial at the point's
storage. The tailrace is read at ``outflow_factor · (Q + S) + Q_lat + gauge
share``, with spill left out where it does not reach the tailrace, and
interpolated between families at the downstream plant's forebay at the same
operating point.

A point is a tie when the plant's energy or stored water is worth nothing in
either model: the source model's submarket CMO, the spot price at every bus
the plant feeds, or Novomodelo's water value in $/MWh rounds to zero at the source
models' two-decimal print resolution. Ties are left out of the split and of
the masked per-plant ranking, and the tie share is reported next to both.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from novomodelo_bridge.comparators.fpha import (
    FPHA_METRICS_SCHEMA,
    FPHA_SPILL_SCHEMA,
    FPHA_SPLIT_SCHEMA,
    FPHA_SURFACE_SCHEMA,
    dense_grid,
    novomodelo_plane_arrays,
    productivity_split,
)
from novomodelo_bridge.core.productivity import (
    apply_hydraulic_loss,
    evaluate_cota,
    hydraulic_generation_mw,
)
from novomodelo_bridge.core.tailrace import TailraceFamily, tailrace_level_m
from novomodelo_bridge.core.units import HM3_TO_MWH_PER_RHO
from novomodelo_bridge.novomodelo import readers as novomodelo_readers

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

_LOG = logging.getLogger(__name__)

#: The source models print prices with two decimals.
ZERO_PRICE: float = 0.005

POINT_KEYS: tuple[str, ...] = ("stage", "scenario", "block")

_TURBINED_EPS: float = 1.0e-6

_SLICE_POINTS = 25

#: Per-(plant, stage) fit inputs, and each model's mean operating point
#: (weighted over every point) with the head components there.
#: Novomodelo fits turbined flow up to the config's ``turbined_fit_max_m3s``, else
#: the plant's static limit (``novomodelo_q_max_m3s``); ``novomodelo_q_stage_max_m3s`` is
#: the stage's limit.
#: ``*_lateral_m3s`` is the flow reaching the tailrace besides the plant's own,
#: ``*_downstream_level_m`` the downstream forebay.
#: ``exact_check_max_mw`` is the largest gap between the exact production
#: function and the source model's own printed exact production at its printed
#: points; null when the run did not print it.
FPHA_FIT_INPUTS_SCHEMA: dict[str, type[pl.DataType]] = {
    "novomodelo_id": pl.Int64,
    "plant_name": pl.Utf8,
    "stage": pl.Int64,
    "source_v_min_hm3": pl.Float64,
    "source_v_max_hm3": pl.Float64,
    "novomodelo_v_min_hm3": pl.Float64,
    "novomodelo_v_max_hm3": pl.Float64,
    "source_q_max_m3s": pl.Float64,
    "novomodelo_q_max_m3s": pl.Float64,
    "novomodelo_q_stage_max_m3s": pl.Float64,
    "novomodelo_spill_fit_max_m3s": pl.Float64,
    "source_grid": pl.Utf8,
    "n_planes_source": pl.Int64,
    "n_planes_novomodelo": pl.Int64,
    "source_gh_max_mw": pl.Float64,
    "exact_check_max_mw": pl.Float64,
    "source_v_hm3": pl.Float64,
    "novomodelo_v_hm3": pl.Float64,
    "source_q_m3s": pl.Float64,
    "novomodelo_q_m3s": pl.Float64,
    "source_s_m3s": pl.Float64,
    "novomodelo_s_m3s": pl.Float64,
    "source_gh_mw": pl.Float64,
    "novomodelo_gh_mw": pl.Float64,
    "source_lateral_m3s": pl.Float64,
    "novomodelo_lateral_m3s": pl.Float64,
    "source_downstream_level_m": pl.Float64,
    "novomodelo_downstream_level_m": pl.Float64,
    "source_forebay_m": pl.Float64,
    "novomodelo_forebay_m": pl.Float64,
    "source_tailrace_m": pl.Float64,
    "novomodelo_tailrace_m": pl.Float64,
    "source_loss_m": pl.Float64,
    "novomodelo_loss_m": pl.Float64,
    "source_net_head_m": pl.Float64,
    "novomodelo_net_head_m": pl.Float64,
    "source_rho_exact": pl.Float64,
    "novomodelo_rho_exact": pl.Float64,
}

#: GH along a turbined-flow sweep per (plant, stage). ``source_planes``,
#: ``novomodelo_planes`` and ``exact`` are taken at the source model's mean
#: storage, spill and lateral flow; ``novomodelo_planes_own`` at Novomodelo's own mean
#: storage and spill.
FPHA_SLICE_SCHEMA: dict[str, type[pl.DataType]] = {
    "novomodelo_id": pl.Int64,
    "stage": pl.Int64,
    "curve": pl.Utf8,
    "q_m3s": pl.Float64,
    "gh_mw": pl.Float64,
}

#: Per-stage weight share of each model's points that are ties.
TIE_SHARE_SCHEMA: dict[str, type[pl.DataType]] = {
    "stage": pl.Int64,
    "source_tie_share": pl.Float64,
    "novomodelo_tie_share": pl.Float64,
}

#: :class:`~novomodelo_bridge.comparators.model.ResultComparison` fields, one row
#: per (plant, stage, variable), over the points that are not ties.
MASKED_HYDRO_SCHEMA: dict[str, type[pl.DataType]] = {
    "entity_type": pl.Utf8,
    "entity_name": pl.Utf8,
    "newave_code": pl.Int64,
    "novomodelo_id": pl.Int64,
    "stage": pl.Int64,
    "variable": pl.Utf8,
    "newave_value": pl.Float64,
    "novomodelo_value": pl.Float64,
    "abs_diff": pl.Float64,
    "rel_diff": pl.Float64,
}

#: The source model's planes per (plant, stage), each bounding ``GH <=
#: fator_correcao · (gamma_0 + gamma_v · (V − volume_floor) + gamma_q · Q +
#: gamma_s · S + gamma_lat · Q_lat)``.
PLANE_SCHEMA: dict[str, type[pl.DataType]] = {
    "gamma_0": pl.Float64,
    "gamma_v": pl.Float64,
    "gamma_q": pl.Float64,
    "gamma_s": pl.Float64,
    "gamma_lat": pl.Float64,
    "fator_correcao": pl.Float64,
}

#: ``Q_lat`` of ``code`` sums ``factor`` times each influencer's turbined plus
#: spilled flow at the same point.
LINK_SCHEMA: dict[str, type[pl.DataType]] = {
    "code": pl.Int64,
    "influencer": pl.Int64,
    "factor": pl.Float64,
}

_MASKED_FLOWS = {
    "generation_mw": "gh_mw",
    "turbined_m3s": "q_m3s",
    "spillage_m3s": "s_m3s",
    "outflow_m3s": "outflow_m3s",
}


@dataclass(frozen=True)
class OperatingPointComparison:
    """The frames the Productivity tab draws; each is empty when its inputs
    are missing."""

    split: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=FPHA_SPLIT_SCHEMA)
    )
    fit_inputs: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=FPHA_FIT_INPUTS_SCHEMA)
    )
    slices: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=FPHA_SLICE_SCHEMA)
    )
    metrics: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=FPHA_METRICS_SCHEMA)
    )
    surface: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=FPHA_SURFACE_SCHEMA)
    )
    spill: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=FPHA_SPILL_SCHEMA)
    )
    tie_share: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=TIE_SHARE_SCHEMA)
    )
    masked_hydro: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=MASKED_HYDRO_SCHEMA)
    )


@dataclass(frozen=True)
class Hydraulics:
    """A plant's tailrace inputs to the exact production function.

    ``volume_floor_hm3`` is the storage the source planes' volume coefficient
    is measured above.
    """

    families: tuple[TailraceFamily, ...]
    spill_in_tailrace: bool
    tipo_perda: int
    volume_floor_hm3: float
    outflow_factor: float = 1.0
    gauge_share_m3s: float = 0.0


@dataclass(frozen=True)
class StagePhysics:
    """A plant's head and generation inputs at one stage."""

    cota: tuple[float, ...]
    rho_esp: float
    losses: float
    max_generation_mw: float


@dataclass(frozen=True)
class Head:
    """The head components and exact generation at one operating point."""

    forebay_m: float
    tailrace_m: float
    net_head_m: float
    generation_mw: float


@dataclass(frozen=True)
class FitWindow:
    """The source model's fit window for a plant and stage."""

    v_min_hm3: float | None
    v_max_hm3: float | None
    q_max_m3s: float | None
    grid: str | None
    gh_max_mw: float | None


@dataclass(frozen=True)
class SourceProductionModel:
    """The source model's production model at the (plant, stage) pairs both
    models fitted.

    ``physics`` names the pairs; ``planes`` (:data:`PLANE_SCHEMA`) and
    ``downstream`` (the plant downstream, ``None`` for none) cover each, and
    ``hydraulics`` each pair's plant. ``cota`` returns any operated plant's
    cota polynomial at a stage, for the downstream plant's forebay.
    """

    planes: Mapping[tuple[int, int], pl.DataFrame]
    windows: Mapping[tuple[int, int], FitWindow]
    hydraulics: Mapping[int, Hydraulics]
    physics: Mapping[tuple[int, int], StagePhysics]
    downstream: Mapping[tuple[int, int], int | None]
    cota: Callable[[int, int], Sequence[float]]
    links: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=LINK_SCHEMA)
    )


def window_gh_max(
    windows: Mapping[tuple[int, int], FitWindow], code: int, stage: int
) -> float:
    """The generation cap of *code*'s fit window at *stage*, unbounded when
    the source model reports none."""
    window = windows.get((code, stage))
    if window is None or window.gh_max_mw is None:
        return math.inf
    return float(window.gh_max_mw)


def operating_point_fields(
    comparison: OperatingPointComparison | None,
) -> dict[str, pl.DataFrame | None]:
    """The :class:`~novomodelo_bridge.comparators.model.PercentileData` fields
    *comparison* fills; the FPHA surface frames are ``None`` when either model
    fitted no planes."""
    if comparison is None:
        return {}

    def _or_none(frame: pl.DataFrame) -> pl.DataFrame | None:
        return None if frame.is_empty() else frame

    return {
        "fpha_metrics": _or_none(comparison.metrics),
        "fpha_surface": _or_none(comparison.surface),
        "fpha_spill": _or_none(comparison.spill),
        "fpha_split": comparison.split,
        "fpha_fit_inputs": comparison.fit_inputs,
        "fpha_slices": comparison.slices,
        "tie_share": comparison.tie_share,
        "hydro_masked": comparison.masked_hydro,
    }


def weighted_mean(column: str) -> pl.Expr:
    """The ``weight``-weighted mean of *column*."""
    return (pl.col("weight") * pl.col(column)).sum() / pl.col("weight").sum()


def novomodelo_points(
    novomodelo_output_dir: Path, novomodelo_to_code: Mapping[int, int]
) -> pl.DataFrame:
    """Novomodelo's per-block operating points, every hydro *novomodelo_to_code* maps to
    a source plant code; empty without per-block hydro operation."""
    hydro = novomodelo_readers.read_novomodelo_hydro_block_operation(
        novomodelo_output_dir
    )
    if hydro.is_empty():
        return pl.DataFrame()
    points = hydro.select(
        pl.col("hydro_id")
        .replace_strict(dict(novomodelo_to_code), default=None, return_dtype=pl.Int64)
        .alias("code"),
        pl.col("hydro_id").alias("novomodelo_id"),
        pl.col("stage_id").alias("stage"),
        pl.col("scenario_id").alias("scenario"),
        pl.col("block_id").alias("block"),
        (pl.col("hours") / pl.col("scenario_id").n_unique().over("stage_id")).alias(
            "weight"
        ),
        ((pl.col("storage_initial_hm3") + pl.col("storage_final_hm3")) / 2.0).alias(
            "v_hm3"
        ),
        pl.col("turbined_m3s").fill_null(0.0).alias("q_m3s"),
        pl.col("spillage_m3s").fill_null(0.0).alias("s_m3s"),
        "outflow_m3s",
        pl.col("generation_mw").fill_null(0.0).alias("gh_mw"),
        pl.when(pl.col("accumulated_productivity_mw_per_m3s") > 0.0)
        .then(
            (
                pl.col("water_value_per_hm3")
                / (pl.col("accumulated_productivity_mw_per_m3s") * HM3_TO_MWH_PER_RHO)
            ).abs()
            < ZERO_PRICE
        )
        .otherwise(False)
        .fill_null(False)
        .alias("_water_worthless"),
    ).filter(pl.col("code").is_not_null())
    return points.join(
        _novomodelo_worthless_energy(novomodelo_output_dir),
        on=["novomodelo_id", *POINT_KEYS],
        how="left",
    ).select(
        pl.exclude("_water_worthless", "_energy_worthless"),
        (
            pl.col("_water_worthless") | pl.col("_energy_worthless").fill_null(False)
        ).alias("worthless"),
    )


def _novomodelo_worthless_energy(novomodelo_output_dir: Path) -> pl.DataFrame:
    """Whether the spot price is zero at every bus a hydro feeds, per point."""
    prices = novomodelo_readers.read_novomodelo_bus_block_prices(novomodelo_output_dir)
    labels = novomodelo_readers.read_novomodelo_hydro_bus_labels(novomodelo_output_dir)
    buses = pl.DataFrame(
        [(hydro_id, bus) for hydro_id, ids in labels.items() for bus in ids],
        schema={"novomodelo_id": pl.Int64, "bus_id": pl.Int64},
        orient="row",
    )
    return (
        buses.join(
            prices.select(
                pl.col("scenario_id").alias("scenario"),
                pl.col("stage_id").alias("stage"),
                pl.col("block_id").alias("block"),
                "bus_id",
                (pl.col("spot_price").abs() < ZERO_PRICE).alias("_zero"),
            ),
            on="bus_id",
        )
        .group_by("novomodelo_id", *POINT_KEYS)
        .agg(pl.col("_zero").all().alias("_energy_worthless"))
    )


def _scenario_alignment(source: pl.DataFrame, novomodelo: pl.DataFrame) -> pl.DataFrame:
    """``(stage, scenario, source_scenario)``: each Novomodelo scenario's source node.

    A single-node stage takes every Novomodelo scenario, and a fan with as many
    scenarios as Novomodelo's pairs them in order. Any other stage is left out and
    logged; its points are flagged on each model's own prices only.
    """
    source_scenarios = dict(
        source.group_by("stage").agg(pl.col("scenario").unique().sort()).iter_rows()
    )
    novomodelo_scenarios = dict(
        novomodelo.group_by("stage").agg(pl.col("scenario").unique().sort()).iter_rows()
    )
    rows: list[tuple[int, int, int]] = []
    for stage, nodes in sorted(source_scenarios.items()):
        scenarios = novomodelo_scenarios.get(stage, [])
        if len(nodes) == 1:
            rows.extend((stage, s, nodes[0]) for s in scenarios)
        elif len(nodes) == len(scenarios):
            rows.extend((stage, s, n) for s, n in zip(scenarios, nodes, strict=True))
        else:
            _LOG.info(
                "Stage %d: %d source-model scenarios against %d Novomodelo scenarios; "
                "its tie points are flagged on each model's own prices.",
                stage + 1,
                len(nodes),
                len(scenarios),
            )
    return pl.DataFrame(
        rows,
        schema={"stage": pl.Int64, "scenario": pl.Int64, "source_scenario": pl.Int64},
        orient="row",
    )


def flag_ties(
    source: pl.DataFrame, novomodelo: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add a ``tie`` column to both models' points.

    A Novomodelo point is a tie when it or its source node is worthless; a source
    point when it or any Novomodelo scenario aligned to it is.
    """
    plant_keys = ["novomodelo_id", *POINT_KEYS]
    joint = (
        novomodelo.select(*plant_keys, "worthless")
        .join(_scenario_alignment(source, novomodelo), on=["stage", "scenario"])
        .join(
            source.select(
                "novomodelo_id",
                "stage",
                pl.col("scenario").alias("source_scenario"),
                "block",
                pl.col("worthless").alias("_source_worthless"),
            ),
            on=["novomodelo_id", "stage", "source_scenario", "block"],
            how="left",
        )
        .with_columns(
            (pl.col("worthless") | pl.col("_source_worthless").fill_null(False)).alias(
                "tie"
            )
        )
    )
    novomodelo = novomodelo.join(
        joint.select(*plant_keys, "tie"), on=plant_keys, how="left"
    ).with_columns(pl.col("tie").fill_null(pl.col("worthless")))
    source_ties = joint.group_by(
        "novomodelo_id", "stage", pl.col("source_scenario").alias("scenario"), "block"
    ).agg(pl.col("tie").any().alias("_any"))
    source = (
        source.join(source_ties, on=plant_keys, how="left")
        .with_columns(
            (pl.col("worthless") | pl.col("_any").fill_null(False)).alias("tie")
        )
        .drop("_any")
    )
    return source, novomodelo


def _masked_means(points: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    return (
        points.filter(~pl.col("tie"))
        .group_by("novomodelo_id", "stage")
        .agg(pl.col("code").first(), *[weighted_mean(c).alias(c) for c in columns])
    )


def masked_hydro(
    source: pl.DataFrame, novomodelo: pl.DataFrame, names: Mapping[int, str]
) -> pl.DataFrame:
    """Per-(plant, stage) flows and productivity over the points that are not ties."""
    columns = list(_MASKED_FLOWS.values())
    joined = _masked_means(source, columns).join(
        _masked_means(novomodelo, columns).drop("code"),
        on=["novomodelo_id", "stage"],
        suffix="_novomodelo",
    )
    if joined.is_empty():
        return pl.DataFrame(schema=MASKED_HYDRO_SCHEMA)
    long = [
        joined.select(
            "novomodelo_id",
            "stage",
            "code",
            pl.lit(variable).alias("variable"),
            pl.col(column).alias("newave_value"),
            pl.col(f"{column}_novomodelo").alias("novomodelo_value"),
        )
        for variable, column in _MASKED_FLOWS.items()
    ]
    long.append(
        joined.filter(
            (pl.col("q_m3s") > _TURBINED_EPS)
            & (pl.col("q_m3s_novomodelo") > _TURBINED_EPS)
        ).select(
            "novomodelo_id",
            "stage",
            "code",
            pl.lit("productivity_mw_per_m3s").alias("variable"),
            (pl.col("gh_mw") / pl.col("q_m3s")).alias("newave_value"),
            (pl.col("gh_mw_novomodelo") / pl.col("q_m3s_novomodelo")).alias(
                "novomodelo_value"
            ),
        )
    )
    return (
        pl.concat(long)
        .drop_nulls(["newave_value", "novomodelo_value"])
        .with_columns(
            pl.lit("hydro").alias("entity_type"),
            pl.col("novomodelo_id")
            .replace_strict(dict(names), default="", return_dtype=pl.Utf8)
            .alias("entity_name"),
            pl.col("code").alias("newave_code"),
            (pl.col("newave_value") - pl.col("novomodelo_value"))
            .abs()
            .alias("abs_diff"),
        )
        .with_columns(
            pl.when(pl.col("newave_value").abs() > 1e-10)
            .then(pl.col("abs_diff") / pl.col("newave_value").abs())
            .otherwise(None)
            .alias("rel_diff")
        )
        .select(list(MASKED_HYDRO_SCHEMA))
        .cast(MASKED_HYDRO_SCHEMA)
        .sort("novomodelo_id", "stage", "variable")
    )


def tie_share(source: pl.DataFrame, novomodelo: pl.DataFrame) -> pl.DataFrame:
    """Per-stage weight share of each model's points that are ties."""

    def _share(points: pl.DataFrame, alias: str) -> pl.DataFrame:
        return points.group_by("stage").agg(
            (
                pl.col("weight").filter(pl.col("tie")).sum() / pl.col("weight").sum()
            ).alias(alias)
        )

    return (
        _share(source, "source_tie_share")
        .join(
            _share(novomodelo, "novomodelo_tie_share"),
            on="stage",
            how="full",
            coalesce=True,
        )
        .select(list(TIE_SHARE_SCHEMA))
        .cast(TIE_SHARE_SCHEMA)
        .sort("stage")
    )


# --- FPHA: both surfaces and the exact function at the operating points ---


def head(
    hydraulics: Hydraulics,
    physics: StagePhysics,
    *,
    v: float,
    q: float,
    s: float,
    lateral_m3s: float,
    downstream_level_m: float | None,
) -> Head:
    """The exact production function's head and generation at one point."""
    forebay = evaluate_cota(physics.cota, v)
    tail_flow = (
        hydraulics.outflow_factor * (q + (s if hydraulics.spill_in_tailrace else 0.0))
        + lateral_m3s
        + hydraulics.gauge_share_m3s
    )
    tailrace = tailrace_level_m(hydraulics.families, tail_flow, downstream_level_m)
    net = apply_hydraulic_loss(
        forebay - tailrace, hydraulics.tipo_perda, physics.losses
    )
    return Head(
        forebay_m=forebay,
        tailrace_m=tailrace,
        net_head_m=net,
        generation_mw=hydraulic_generation_mw(
            physics.rho_esp, q, net, physics.max_generation_mw
        ),
    )


def evaluate_points(points: pl.DataFrame, model: SourceProductionModel) -> pl.DataFrame:
    """*points* at the pairs *model* covers, with ``lateral_m3s``, the
    downstream plant's storage and forebay at the same point, and
    ``exact_mw``.

    The other plants' rows stay in *points* only to supply the lateral flow
    and the downstream storage.
    """
    downstream = pl.DataFrame(
        [
            (code, stage, model.downstream[(code, stage)])
            for code, stage in model.physics
        ],
        schema={"code": pl.Int64, "stage": pl.Int64, "downstream": pl.Int64},
        orient="row",
    )
    frame = _with_context(points, model.links, downstream)
    cota_cache: dict[tuple[int, int], tuple[float, ...]] = {}

    def _downstream_level(
        code: int | None, stage: int, v: float | None
    ) -> float | None:
        if code is None or v is None:
            return None
        key = (code, stage)
        if key not in cota_cache:
            cota_cache[key] = tuple(model.cota(code, stage))
        return evaluate_cota(cota_cache[key], v)

    levels: list[float | None] = []
    exact: list[float] = []
    for code, stage, v, q, s, lateral, down, down_v in frame.select(
        "code",
        "stage",
        "v_hm3",
        "q_m3s",
        "s_m3s",
        "lateral_m3s",
        "downstream",
        "downstream_v_hm3",
    ).iter_rows():
        level = _downstream_level(down, stage, down_v)
        levels.append(level)
        exact.append(
            head(
                model.hydraulics[code],
                model.physics[(code, stage)],
                v=v,
                q=q,
                s=s,
                lateral_m3s=lateral,
                downstream_level_m=level,
            ).generation_mw
        )
    return frame.with_columns(
        pl.Series("downstream_level_m", levels, dtype=pl.Float64),
        pl.Series("exact_mw", exact, dtype=pl.Float64),
    )


def _with_context(
    points: pl.DataFrame, links: pl.DataFrame, downstream: pl.DataFrame
) -> pl.DataFrame:
    """The FPHA plants' points, with the lateral flow and the downstream
    plant's storage at the same point."""
    release = points.select(
        *POINT_KEYS,
        pl.col("code").alias("influencer"),
        (pl.col("q_m3s") + pl.col("s_m3s")).alias("_release"),
    )
    lateral = (
        links.join(release, on="influencer")
        .group_by("code", *POINT_KEYS)
        .agg((pl.col("factor") * pl.col("_release")).sum().alias("lateral_m3s"))
    )
    storage = points.select(
        *POINT_KEYS,
        pl.col("code").alias("downstream"),
        pl.col("v_hm3").alias("downstream_v_hm3"),
    )
    return (
        points.join(downstream, on=["code", "stage"])
        .join(lateral, on=["code", *POINT_KEYS], how="left")
        .join(storage, on=["downstream", *POINT_KEYS], how="left")
        .with_columns(pl.col("lateral_m3s").fill_null(0.0))
    )


def source_envelope(
    planes: pl.DataFrame,
    v_useful: np.ndarray,
    q: np.ndarray,
    s: np.ndarray,
    lateral: np.ndarray,
) -> np.ndarray:
    """The source model's lower envelope over *planes* (:data:`PLANE_SCHEMA`)
    at each point, *v_useful* measured above the plant's volume floor."""

    def _col(name: str) -> np.ndarray:
        return planes.get_column(name).to_numpy()[:, None]

    values = _col("fator_correcao") * (
        _col("gamma_0")
        + _col("gamma_v") * v_useful[None, :]
        + _col("gamma_q") * q[None, :]
        + _col("gamma_s") * s[None, :]
        + _col("gamma_lat") * lateral[None, :]
    )
    return values.min(axis=0)


def with_envelopes(
    points: pl.DataFrame,
    model: SourceProductionModel,
    novomodelo_planes: Mapping[tuple[int, int], pl.DataFrame],
) -> pl.DataFrame:
    """Both models' envelopes at the evaluated source points;
    *novomodelo_planes* is keyed by ``(novomodelo_id, stage)``."""
    parts: list[pl.DataFrame] = []
    for (code, novomodelo_id, stage), group in points.partition_by(
        ["code", "novomodelo_id", "stage"], as_dict=True
    ).items():
        v = group.get_column("v_hm3").to_numpy()
        q = group.get_column("q_m3s").to_numpy()
        s = group.get_column("s_m3s").to_numpy()
        lateral = group.get_column("lateral_m3s").to_numpy()
        source_mw = source_envelope(
            model.planes[(code, stage)],
            v - model.hydraulics[code].volume_floor_hm3,
            q,
            s,
            lateral,
        )
        novomodelo_mw = dense_grid(
            *novomodelo_plane_arrays(novomodelo_planes[(novomodelo_id, stage)]), v, q, s
        )
        parts.append(
            group.with_columns(
                pl.Series("source_planes_mw", source_mw, dtype=pl.Float64),
                pl.Series("novomodelo_planes_mw", novomodelo_mw, dtype=pl.Float64),
            )
        )
    return pl.concat(parts)


def productivity_frames(
    source_eval: pl.DataFrame,
    novomodelo_eval: pl.DataFrame,
    model: SourceProductionModel,
    novomodelo_planes: Mapping[tuple[int, int], pl.DataFrame],
    novomodelo_output_dir: Path,
    exact_check: Mapping[tuple[int, int], float],
    names: Mapping[int, str],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """``(split, fit_inputs, slices)`` from the source points with their
    envelopes (:func:`with_envelopes`) and Novomodelo's evaluated points.

    *exact_check* maps ``(code, stage)`` to ``exact_check_max_mw``.
    """
    split = (
        productivity_split(source_eval, novomodelo_eval)
        .with_columns(
            pl.col("novomodelo_id")
            .replace_strict(dict(names), default=None, return_dtype=pl.Utf8)
            .alias("plant_name")
        )
        .select(list(FPHA_SPLIT_SCHEMA))
    )
    means = _mean_points(source_eval, "source").join(
        _mean_points(novomodelo_eval, "novomodelo").drop("code"),
        on=["novomodelo_id", "stage"],
        how="left",
    )
    fit_inputs, slices = _fit_inputs_and_slices(
        means, model, novomodelo_planes, novomodelo_output_dir, exact_check, names
    )
    return split, fit_inputs, slices


def _mean_points(points: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Each (plant, stage)'s weighted mean operating point over every point."""
    return points.group_by("novomodelo_id", "stage").agg(
        pl.col("code").first(),
        *[
            weighted_mean(c).alias(f"{prefix}_{c}")
            for c in ("v_hm3", "q_m3s", "s_m3s", "gh_mw", "lateral_m3s")
        ],
        (
            (pl.col("weight") * pl.col("downstream_level_m")).sum()
            / pl.col("weight").filter(pl.col("downstream_level_m").is_not_null()).sum()
        ).alias(f"{prefix}_downstream_level_m"),
        pl.col("q_m3s").max().alias(f"{prefix}_q_peak_m3s"),
    )


def _fit_inputs_and_slices(
    means: pl.DataFrame,
    model: SourceProductionModel,
    novomodelo_planes_by_key: Mapping[tuple[int, int], pl.DataFrame],
    novomodelo_output_dir: Path,
    exact_check: Mapping[tuple[int, int], float],
    names: Mapping[int, str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    novomodelo_windows = {
        (int(r["hydro_id"]), int(r["stage_id"])): r
        for r in novomodelo_readers.read_novomodelo_fpha_fit_config(
            novomodelo_output_dir
        ).iter_rows(named=True)
    }
    static_q_max, stage_q_max = _novomodelo_turbined_limits(novomodelo_output_dir)
    no_window = FitWindow(None, None, None, None, None)

    rows: list[dict[str, object]] = []
    curves: list[tuple[int, int, str, float, float]] = []
    for m in means.sort("novomodelo_id", "stage").iter_rows(named=True):
        code, novomodelo_id, stage = (
            int(m["code"]),
            int(m["novomodelo_id"]),
            int(m["stage"]),
        )
        hyd, phys = model.hydraulics[code], model.physics[(code, stage)]
        window = model.windows.get((code, stage), no_window)
        novomodelo_window = novomodelo_windows.get((novomodelo_id, stage), {})
        source_planes = model.planes[(code, stage)]
        novomodelo_planes = novomodelo_planes_by_key[(novomodelo_id, stage)]
        static_limit = static_q_max.get(novomodelo_id)
        fit_q_max = novomodelo_window.get("turbined_fit_max_m3s")
        novomodelo_q_max = static_limit if fit_q_max is None else fit_q_max
        novomodelo_q_stage_max = stage_q_max.get((novomodelo_id, stage), static_limit)

        def _at_mean(prefix: str, q: float | None = None) -> Head:
            level = m[f"{prefix}_downstream_level_m"]
            return head(
                hyd,
                phys,
                v=m[f"{prefix}_v_hm3"],
                q=m[f"{prefix}_q_m3s"] if q is None else q,
                s=m[f"{prefix}_s_m3s"],
                lateral_m3s=m[f"{prefix}_lateral_m3s"],
                downstream_level_m=None
                if level is None or math.isnan(level)
                else level,
            )

        heads = {"source": _at_mean("source")}
        if m["novomodelo_v_hm3"] is not None:
            heads["novomodelo"] = _at_mean("novomodelo")
        row: dict[str, object] = {
            "novomodelo_id": novomodelo_id,
            "plant_name": names.get(novomodelo_id),
            "stage": stage,
            "source_v_min_hm3": window.v_min_hm3,
            "source_v_max_hm3": window.v_max_hm3,
            "novomodelo_v_min_hm3": novomodelo_window.get("volume_min_hm3"),
            "novomodelo_v_max_hm3": novomodelo_window.get("volume_max_hm3"),
            "source_q_max_m3s": window.q_max_m3s,
            "novomodelo_q_max_m3s": novomodelo_q_max,
            "novomodelo_q_stage_max_m3s": novomodelo_q_stage_max,
            "novomodelo_spill_fit_max_m3s": novomodelo_window.get(
                "spillage_fit_max_m3s"
            ),
            "source_grid": window.grid,
            "n_planes_source": source_planes.height,
            "n_planes_novomodelo": novomodelo_planes.height,
            "source_gh_max_mw": window.gh_max_mw,
            "exact_check_max_mw": exact_check.get((code, stage)),
        }
        for prefix in ("source", "novomodelo"):
            point_head = heads.get(prefix)
            row[f"{prefix}_v_hm3"] = m[f"{prefix}_v_hm3"]
            row[f"{prefix}_q_m3s"] = m[f"{prefix}_q_m3s"]
            row[f"{prefix}_s_m3s"] = m[f"{prefix}_s_m3s"]
            row[f"{prefix}_gh_mw"] = m[f"{prefix}_gh_mw"]
            row[f"{prefix}_lateral_m3s"] = (
                None
                if m[f"{prefix}_lateral_m3s"] is None
                else m[f"{prefix}_lateral_m3s"] + hyd.gauge_share_m3s
            )
            level = m[f"{prefix}_downstream_level_m"]
            row[f"{prefix}_downstream_level_m"] = (
                None if level is None or math.isnan(level) else level
            )
            row[f"{prefix}_forebay_m"] = (
                None if point_head is None else point_head.forebay_m
            )
            row[f"{prefix}_tailrace_m"] = (
                None if point_head is None else point_head.tailrace_m
            )
            row[f"{prefix}_loss_m"] = (
                None
                if point_head is None
                else point_head.forebay_m
                - point_head.tailrace_m
                - point_head.net_head_m
            )
            row[f"{prefix}_net_head_m"] = (
                None if point_head is None else point_head.net_head_m
            )
            row[f"{prefix}_rho_exact"] = (
                None if point_head is None else phys.rho_esp * point_head.net_head_m
            )
        rows.append(row)

        q_top = max(
            float(v)
            for v in (
                window.q_max_m3s,
                novomodelo_q_max,
                m["source_q_peak_m3s"],
                m["novomodelo_q_peak_m3s"],
            )
            if v is not None
        )
        q_axis = np.linspace(0.0, q_top, _SLICE_POINTS)
        v_src = np.full_like(q_axis, m["source_v_hm3"])
        s_src = np.full_like(q_axis, m["source_s_m3s"])
        sweeps = {
            "source_planes": source_envelope(
                source_planes,
                v_src - hyd.volume_floor_hm3,
                q_axis,
                s_src,
                np.full_like(q_axis, m["source_lateral_m3s"]),
            ),
            "novomodelo_planes": dense_grid(
                *novomodelo_plane_arrays(novomodelo_planes), v_src, q_axis, s_src
            ),
            "exact": np.array(
                [_at_mean("source", float(q)).generation_mw for q in q_axis]
            ),
        }
        if m["novomodelo_v_hm3"] is not None:
            sweeps["novomodelo_planes_own"] = dense_grid(
                *novomodelo_plane_arrays(novomodelo_planes),
                np.full_like(q_axis, m["novomodelo_v_hm3"]),
                q_axis,
                np.full_like(q_axis, m["novomodelo_s_m3s"]),
            )
        for curve, gh in sweeps.items():
            curves.extend(
                (novomodelo_id, stage, curve, float(q), float(g))
                for q, g in zip(q_axis, gh, strict=True)
            )

    fit_inputs = pl.DataFrame(rows, schema=FPHA_FIT_INPUTS_SCHEMA)
    slices = pl.DataFrame(curves, schema=FPHA_SLICE_SCHEMA, orient="row")
    return fit_inputs, slices


def _novomodelo_turbined_limits(
    novomodelo_output_dir: Path,
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """Novomodelo's turbined-flow limits: ``({hydro: static}, {(hydro, stage):
    per-stage bound})``, the per-stage bound overriding the static one."""
    static = {
        hydro_id: float(meta["max_turbined_m3s"])
        for hydro_id, meta in novomodelo_readers.read_novomodelo_hydro_metadata(
            novomodelo_output_dir
        ).items()
    }
    per_stage: dict[tuple[int, int], float] = {}
    bounds = novomodelo_readers.read_novomodelo_hydro_per_stage_bounds(
        novomodelo_output_dir
    )
    if "max_turbined_m3s" in bounds.columns:
        for hydro_id, stage, value in bounds.select(
            "entity_id", "stage_id", "max_turbined_m3s"
        ).iter_rows():
            if value is not None:
                per_stage[(int(hydro_id), int(stage))] = float(value)
    return static, per_stage
