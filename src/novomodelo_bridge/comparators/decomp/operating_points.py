"""Both models compared at the per-block operating points each one reached.

The source model's planes, Novomodelo's planes and the plant's exact production
function are evaluated at both models' operating points, so the gap between
the two realized productivities splits into a part due to the fitted surfaces
and a part due to where each model operates
(:func:`~novomodelo_bridge.comparators.fpha.productivity_split`).

The source model's conventions, each checked against its own printed
deviation table (``dec_desvfpha``):

- Its planes read ``GH <= fc · (rhs + cv · V_u + cq · Q + cs · S + cl ·
  Q_lat)``, ``V_u`` measured above the ``hidr.dat`` minimum and ``Q_lat`` the
  ``VU`` share of the influencing plants' turbined plus spilled flow in the
  same block. The ``VA`` gauge share is folded into ``rhs``.
- Its FPHA storage is the stage's mean, the effective minimum plus the mean of
  the initial and final useful volumes. Novomodelo's is the mean of its initial and
  final storage.
- The exact production function is ``min(ρ_esp · Q · h_net, G_max)``. The
  forebay comes from the effective cota polynomial. The tailrace is read at
  ``VL · (Q + S) + Q_lat + VA share``, with spill left out where ``hidr.dat``
  says it does not reach the tailrace, and interpolated between families at
  the downstream plant's forebay at the same operating point. ``G_max`` is the
  ``eco_fpha_`` generation window. The ``VA`` share is the factor times the
  gauge plant's first-stage incremental inflow, held over the horizon as the
  source model prints it.

A point is a tie when the plant's energy or stored water is worth nothing in
either model: the submarket CMO, the spot price at every bus the plant feeds,
or Novomodelo's water value in $/MWh rounds to zero at the source model's
two-decimal print resolution. The source model prints no water value. Ties
are left out of the split and of the masked per-plant ranking, and the tie
share is reported next to both.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from novomodelo_bridge.comparators.analyze import build_fpha_comparison
from novomodelo_bridge.comparators.decomp.readers import (
    read_avl_cortesfpha,
    read_dec_desvfpha,
    read_dec_oper_sist,
    read_dec_oper_usih,
    read_eco_fpha,
)
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
from novomodelo_bridge.decomp.converters.cadastro import (
    EffectiveCadastro,
    build_effective_cadastro,
    effective_storage_range,
)
from novomodelo_bridge.decomp.converters.hydro.turbine_limit import tailrace_families
from novomodelo_bridge.novomodelo import readers as novomodelo_readers

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from novomodelo_bridge.decomp.case import DecompCase

_LOG = logging.getLogger(__name__)

#: The source model prints prices with two decimals.
_ZERO_PRICE: float = 0.005

_TURBINED_EPS: float = 1.0e-6

_SLICE_POINTS = 25

_POINT_KEYS = ["stage", "scenario", "block"]

#: Per-(plant, stage) fit inputs, and each model's mean operating point
#: (weighted over every point) with the head components there.
#: Novomodelo fits turbined flow up to the config's ``turbined_fit_max_m3s``, else
#: the plant's static limit (``novomodelo_q_max_m3s``); ``novomodelo_q_stage_max_m3s`` is
#: the stage's limit.
#: ``*_lateral_m3s`` is the flow reaching the tailrace besides the plant's own
#: (``VU`` and ``VA``), ``*_downstream_level_m`` the downstream forebay.
#: ``exact_check_max_mw`` is the largest gap between the exact production
#: function and the source model's own printed FPH (``dec_desvfpha``) at its
#: printed points; null when the run did not print that table.
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

_MASKED_FLOWS = {
    "generation_mw": "gh_mw",
    "turbined_m3s": "q_m3s",
    "spillage_m3s": "s_m3s",
    "outflow_m3s": "outflow_m3s",
}


@dataclass(frozen=True)
class OperatingPointComparison:
    """The frames :func:`compare_operating_points` builds."""

    split: pl.DataFrame
    fit_inputs: pl.DataFrame
    slices: pl.DataFrame
    metrics: pl.DataFrame
    surface: pl.DataFrame
    spill: pl.DataFrame
    tie_share: pl.DataFrame
    masked_hydro: pl.DataFrame


@dataclass(frozen=True)
class _Hydraulics:
    families: tuple[TailraceFamily, ...]
    spill_in_tailrace: bool
    outflow_factor: float
    gauge_share_m3s: float
    hidr_min_storage_hm3: float
    tipo_perda: int


@dataclass(frozen=True)
class _StagePhysics:
    cota: tuple[float, ...]
    rho_esp: float
    losses: float
    max_generation_mw: float


@dataclass(frozen=True)
class _Head:
    forebay_m: float
    tailrace_m: float
    net_head_m: float
    generation_mw: float


def compare_operating_points(
    decomp_dir: Path,
    case: DecompCase,
    novomodelo_output_dir: Path,
    probabilities: pl.DataFrame,
    names: Mapping[int, str],
) -> OperatingPointComparison | None:
    """Compare both models at their per-block operating points.

    *probabilities* is the source model's ``(estagio, cenario,
    probabilidade)`` tree; when empty, each stage's scenarios weigh equally.
    *names* maps Novomodelo hydro ids to plant names. ``None`` when either run has
    no per-block hydro operation. The FPHA frames are empty when either model
    fitted no planes.
    """
    try:
        usih = read_dec_oper_usih(decomp_dir)
        sist = read_dec_oper_sist(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model per-block operation: %s", exc)
        return None
    eff, _ = build_effective_cadastro(case.dadger, case.hidr, case.calendar)
    source = _source_points(usih, sist, case, eff, probabilities)
    novomodelo = _novomodelo_points(novomodelo_output_dir, case)
    if source.is_empty() or novomodelo.is_empty():
        return None
    source, novomodelo = _flag_ties(source, novomodelo)
    split, fit_inputs, slices, metrics, surface, spill = _fpha_frames(
        decomp_dir,
        case,
        eff,
        novomodelo_output_dir,
        source,
        novomodelo,
        _first_stage_incremental(usih),
        names,
    )
    return OperatingPointComparison(
        split=split,
        fit_inputs=fit_inputs,
        slices=slices,
        metrics=metrics,
        surface=surface,
        spill=spill,
        tie_share=_tie_share(source, novomodelo),
        masked_hydro=_masked_hydro(source, novomodelo, names),
    )


def _wmean(column: str) -> pl.Expr:
    return (pl.col("weight") * pl.col(column)).sum() / pl.col("weight").sum()


def _source_points(
    usih: pl.DataFrame,
    sist: pl.DataFrame,
    case: DecompCase,
    eff: EffectiveCadastro,
    probabilities: pl.DataFrame,
) -> pl.DataFrame:
    """The source model's per-block operating points, every operated plant."""
    code_to_novomodelo = {c: case.id_map.hydro_id(c) for c in case.id_map.hydro_codes}
    points = (
        usih.filter(pl.col("patamar").is_not_null())
        .with_columns(
            pl.col("codigo_usina").cast(pl.Int64).alias("code"),
            (pl.col("estagio").cast(pl.Int64) - 1).alias("stage"),
            pl.col("cenario").cast(pl.Int64).alias("scenario"),
            (pl.col("patamar").cast(pl.Int64) - 1).alias("block"),
        )
        .with_columns(
            pl.col("code")
            .replace_strict(code_to_novomodelo, default=None, return_dtype=pl.Int64)
            .alias("novomodelo_id")
        )
        .filter(pl.col("novomodelo_id").is_not_null())
    )
    if points.is_empty():
        return pl.DataFrame()
    if probabilities.is_empty():
        points = points.with_columns(
            (1.0 / pl.col("scenario").n_unique().over("stage")).alias("probability")
        )
    else:
        points = points.join(
            probabilities.select(
                (pl.col("estagio").cast(pl.Int64) - 1).alias("stage"),
                pl.col("cenario").cast(pl.Int64).alias("scenario"),
                pl.col("probabilidade").cast(pl.Float64).alias("probability"),
            ),
            on=["stage", "scenario"],
            how="left",
        ).with_columns(pl.col("probability").fill_null(1.0))
    floors = pl.DataFrame(
        [
            (code, stage, effective_storage_range(eff, code, stage)[0])
            for code, stage in points.select("code", "stage").unique().iter_rows()
        ],
        schema={"code": pl.Int64, "stage": pl.Int64, "floor_hm3": pl.Float64},
        orient="row",
    )
    cmo = sist.filter(pl.col("patamar").is_not_null()).select(
        (pl.col("estagio").cast(pl.Int64) - 1).alias("stage"),
        pl.col("cenario").cast(pl.Int64).alias("scenario"),
        (pl.col("patamar").cast(pl.Int64) - 1).alias("block"),
        pl.col("nome_submercado").str.strip_chars().alias("submarket"),
        pl.col("cmo").cast(pl.Float64).alias("price"),
    )
    return (
        points.join(floors, on=["code", "stage"], how="left")
        .with_columns(pl.col("nome_submercado").str.strip_chars().alias("submarket"))
        .join(cmo, on=[*_POINT_KEYS, "submarket"], how="left")
        .select(
            "code",
            "novomodelo_id",
            *_POINT_KEYS,
            (pl.col("probability") * pl.col("duracao").cast(pl.Float64)).alias(
                "weight"
            ),
            (
                pl.col("floor_hm3")
                + (
                    pl.col("volume_util_inicial_hm3").fill_null(0.0)
                    + pl.col("volume_util_final_hm3").fill_null(0.0)
                )
                / 2.0
            ).alias("v_hm3"),
            pl.col("vazao_turbinada_m3s")
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("q_m3s"),
            pl.col("vazao_vertida_m3s").cast(pl.Float64).fill_null(0.0).alias("s_m3s"),
            pl.col("vazao_defluente_m3s").cast(pl.Float64).alias("outflow_m3s"),
            pl.col("geracao_MW").cast(pl.Float64).fill_null(0.0).alias("gh_mw"),
            (pl.col("price").abs() < _ZERO_PRICE).fill_null(False).alias("worthless"),
        )
    )


def _first_stage_incremental(usih: pl.DataFrame) -> dict[int, float]:
    """``{code: incremental inflow}`` of the first stage's aggregate rows."""
    rows = usih.filter(
        pl.col("patamar").is_null()
        & (pl.col("estagio") == pl.col("estagio").min())
        & (pl.col("cenario") == pl.col("cenario").min())
    )
    return {
        int(code): float(value)
        for code, value in rows.select(
            "codigo_usina", "vazao_incremental_m3s"
        ).iter_rows()
        if value is not None
    }


def _novomodelo_points(novomodelo_output_dir: Path, case: DecompCase) -> pl.DataFrame:
    """Novomodelo's per-block operating points, every hydro the id map knows."""
    hydro = novomodelo_readers.read_novomodelo_hydro_block_operation(
        novomodelo_output_dir
    )
    if hydro.is_empty():
        return pl.DataFrame()
    novomodelo_to_code = dict(enumerate(case.id_map.hydro_codes))
    points = hydro.select(
        pl.col("hydro_id")
        .replace_strict(novomodelo_to_code, default=None, return_dtype=pl.Int64)
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
            < _ZERO_PRICE
        )
        .otherwise(False)
        .fill_null(False)
        .alias("_water_worthless"),
    ).filter(pl.col("code").is_not_null())
    return points.join(
        _novomodelo_worthless_energy(novomodelo_output_dir),
        on=["novomodelo_id", *_POINT_KEYS],
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
                (pl.col("spot_price").abs() < _ZERO_PRICE).alias("_zero"),
            ),
            on="bus_id",
        )
        .group_by("novomodelo_id", *_POINT_KEYS)
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


def _flag_ties(
    source: pl.DataFrame, novomodelo: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add a ``tie`` column to both models' points.

    A Novomodelo point is a tie when it or its source node is worthless; a source
    point when it or any Novomodelo scenario aligned to it is.
    """
    plant_keys = ["novomodelo_id", *_POINT_KEYS]
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
        .agg(pl.col("code").first(), *[_wmean(c).alias(c) for c in columns])
    )


def _masked_hydro(
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


def _tie_share(source: pl.DataFrame, novomodelo: pl.DataFrame) -> pl.DataFrame:
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


def _fpha_frames(
    decomp_dir: Path,
    case: DecompCase,
    eff: EffectiveCadastro,
    novomodelo_output_dir: Path,
    source: pl.DataFrame,
    novomodelo: pl.DataFrame,
    first_incremental: Mapping[int, float],
    names: Mapping[int, str],
) -> tuple[pl.DataFrame, ...]:
    """``(split, fit_inputs, slices, metrics, surface, spill)``."""
    empty = (
        pl.DataFrame(schema=FPHA_SPLIT_SCHEMA),
        pl.DataFrame(schema=FPHA_FIT_INPUTS_SCHEMA),
        pl.DataFrame(schema=FPHA_SLICE_SCHEMA),
        pl.DataFrame(schema=FPHA_METRICS_SCHEMA),
        pl.DataFrame(schema=FPHA_SURFACE_SCHEMA),
        pl.DataFrame(schema=FPHA_SPILL_SCHEMA),
    )
    try:
        source_planes = read_avl_cortesfpha(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model FPHA planes: %s", exc)
        return empty
    novomodelo_planes = novomodelo_readers.read_novomodelo_fpha_planes(
        novomodelo_output_dir
    )
    if (
        novomodelo_planes is None
        or novomodelo_planes.is_empty()
        or source_planes.is_empty()
    ):
        return empty
    try:
        eco = read_eco_fpha(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model FPHA fit windows (eco_fpha_): %s", exc)
        eco = None

    code_to_novomodelo = {c: case.id_map.hydro_id(c) for c in case.id_map.hydro_codes}
    source_planes = source_planes.with_columns(
        pl.col("codigo_usina").alias("code"),
        (pl.col("estagio") - 1).alias("stage"),
    )
    source_by_key = {
        (int(k[0]), int(k[1])): v
        for k, v in source_planes.partition_by(["code", "stage"], as_dict=True).items()
    }
    novomodelo_by_key = {
        (int(k[0]), int(k[1])): v
        for k, v in novomodelo_planes.partition_by(
            ["hydro_id", "stage_id"], as_dict=True
        ).items()
    }
    pairs = {
        (code, stage)
        for code, stage in source_by_key
        if code in code_to_novomodelo
        and (code_to_novomodelo[code], stage) in novomodelo_by_key
    }
    hydraulics, links = _plant_hydraulics(
        case, {code for code, _ in pairs}, first_incremental
    )
    missing = sorted({code for code, _ in pairs if code not in hydraulics})
    if missing:
        _LOG.info(
            "No tailrace curve for plants %s; their FPHA is left out of the "
            "operating-point comparison.",
            missing,
        )
    pairs = {(code, stage) for code, stage in pairs if code in hydraulics}
    if not pairs:
        return empty

    gh_max = (
        {}
        if eco is None
        else {
            (int(c), int(e) - 1): float(g)
            for c, e, g in eco.select(
                "codigo_usina", "estagio", "geracao_maxima"
            ).iter_rows()
            if g is not None
        }
    )
    physics: dict[tuple[int, int], _StagePhysics] = {}
    for code, stage in pairs:
        physics[(code, stage)] = _StagePhysics(
            cota=tuple(eff.cota_polynomial(code, stage)),
            rho_esp=eff.value(code, "produtibilidade_especifica", stage),
            losses=eff.value(code, "perdas", stage),
            max_generation_mw=gh_max.get((code, stage), math.inf),
        )
    downstream = pl.DataFrame(
        [
            (code, stage, eff.downstream_plant(code, stage) or None)
            for code, stage in sorted(pairs)
        ],
        schema={"code": pl.Int64, "stage": pl.Int64, "downstream": pl.Int64},
        orient="row",
    )
    cota_cache: dict[tuple[int, int], tuple[float, ...]] = {}

    def _downstream_level(
        code: int | None, stage: int, v: float | None
    ) -> float | None:
        if code is None or v is None:
            return None
        key = (code, stage)
        if key not in cota_cache:
            cota_cache[key] = tuple(eff.cota_polynomial(code, stage))
        return evaluate_cota(cota_cache[key], v)

    def _evaluate(points: pl.DataFrame) -> pl.DataFrame:
        frame = _with_context(points, links, downstream)
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
                _head(
                    hydraulics[code],
                    physics[(code, stage)],
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

    source_eval = _with_envelopes(
        _evaluate(source),
        source_by_key,
        novomodelo_by_key,
        code_to_novomodelo,
        hydraulics,
    )
    novomodelo_eval = _evaluate(novomodelo)

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
        means,
        eco,
        novomodelo_output_dir,
        source_by_key,
        novomodelo_by_key,
        hydraulics,
        physics,
        _exact_check(decomp_dir, eff, hydraulics, physics),
        names,
    )
    metrics, surface, spill = build_fpha_comparison(
        _folded_source_planes(source_planes, source_eval, pairs),
        None if eco is None else _source_grid(eco, hydraulics),
        novomodelo_planes,
        code_to_novomodelo,
        names,
    )
    return split, fit_inputs, slices, metrics, surface, spill


def _plant_hydraulics(
    case: DecompCase, codes: set[int], first_incremental: Mapping[int, float]
) -> tuple[dict[int, _Hydraulics], pl.DataFrame]:
    """Each plant's exact-function inputs, and the ``VU`` links
    ``(code, influencer, factor)``.

    A ``VA`` gauge with no operated plant on it contributes nothing and is
    logged.
    """
    hidr = case.hidr
    families = tailrace_families(case, case.id_map)
    dadger = case.dadger
    vl = dadger.vl(df=True)
    vu = dadger.vu(df=True)
    va = dadger.va(df=True)
    outflow_factor: dict[int, float] = (
        {}
        if vl is None
        else {
            int(c): float(f)
            for c, f in zip(
                vl["codigo_usina_influenciada"],
                vl["fator_impacto_defluencia"],
                strict=True,
            )
        }
    )
    links = pl.DataFrame(
        []
        if vu is None
        else [
            (int(t), int(i), float(f))
            for t, i, f in zip(
                vu["codigo_usina_influenciada"],
                vu["codigo_usina_influenciadora"],
                vu["fator_impacto_defluencia"],
                strict=True,
            )
        ],
        schema={"code": pl.Int64, "influencer": pl.Int64, "factor": pl.Float64},
        orient="row",
    )
    operated = set(case.id_map.hydro_codes)
    plant_of_gauge = {
        int(p): int(c) for c, p in hidr["posto"].items() if int(c) in operated
    }
    gauge_share: dict[int, float] = defaultdict(float)
    if va is not None:
        for gauge, target, factor in zip(
            va["codigo_posto_influenciador"],
            va["codigo_usina_influenciada"],
            va["fator_impacto_incremental"],
            strict=True,
        ):
            plant = plant_of_gauge.get(int(gauge))
            if plant is None or plant not in first_incremental:
                _LOG.info(
                    "VA register of plant %d: no operated plant on gauge %d; its "
                    "share of the tail flow is left out of the exact production "
                    "function.",
                    int(target),
                    int(gauge),
                )
                continue
            gauge_share[int(target)] += float(factor) * first_incremental[plant]
    hydraulics = {
        code: _Hydraulics(
            families=families[code],
            spill_in_tailrace=bool(
                int(hidr.loc[code, "influencia_vertimento_canal_fuga"])
            ),
            outflow_factor=outflow_factor.get(code, 1.0),
            gauge_share_m3s=gauge_share.get(code, 0.0),
            hidr_min_storage_hm3=float(hidr.loc[code, "volume_minimo"]),
            tipo_perda=int(hidr.loc[code, "tipo_perda"]),
        )
        for code in codes
        if code in families
    }
    return hydraulics, links


def _with_context(
    points: pl.DataFrame, links: pl.DataFrame, downstream: pl.DataFrame
) -> pl.DataFrame:
    """The FPHA plants' points, with the ``VU`` lateral flow and the downstream
    plant's storage at the same point."""
    release = points.select(
        *_POINT_KEYS,
        pl.col("code").alias("influencer"),
        (pl.col("q_m3s") + pl.col("s_m3s")).alias("_release"),
    )
    lateral = (
        links.join(release, on="influencer")
        .group_by("code", *_POINT_KEYS)
        .agg((pl.col("factor") * pl.col("_release")).sum().alias("lateral_m3s"))
    )
    storage = points.select(
        *_POINT_KEYS,
        pl.col("code").alias("downstream"),
        pl.col("v_hm3").alias("downstream_v_hm3"),
    )
    return (
        points.join(downstream, on=["code", "stage"])
        .join(lateral, on=["code", *_POINT_KEYS], how="left")
        .join(storage, on=["downstream", *_POINT_KEYS], how="left")
        .with_columns(pl.col("lateral_m3s").fill_null(0.0))
    )


def _head(
    hydraulics: _Hydraulics,
    physics: _StagePhysics,
    *,
    v: float,
    q: float,
    s: float,
    lateral_m3s: float,
    downstream_level_m: float | None,
) -> _Head:
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
    return _Head(
        forebay_m=forebay,
        tailrace_m=tailrace,
        net_head_m=net,
        generation_mw=hydraulic_generation_mw(
            physics.rho_esp, q, net, physics.max_generation_mw
        ),
    )


def _exact_check(
    decomp_dir: Path,
    eff: EffectiveCadastro,
    hydraulics: Mapping[int, _Hydraulics],
    physics: Mapping[tuple[int, int], _StagePhysics],
) -> dict[tuple[int, int], float]:
    """``{(code, stage): max |exact − printed FPH|}`` over the source model's
    printed deviation points, with its printed storage, flows, ``VU`` lateral
    flow and the downstream plant's printed forebay; empty without the table.
    """
    try:
        printed = read_dec_desvfpha(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model FPHA deviation table: %s", exc)
        return {}
    keys = ["estagio", "no", "patamar"]
    forebay = {
        (code, *key): level
        for code, *key, level in printed.select(
            "codigo_usina", *keys, "altura_montante"
        ).iter_rows()
    }
    worst: dict[tuple[int, int], float] = {}
    for code, estagio, no, patamar, v, q, s, lateral, fph in printed.select(
        "codigo_usina",
        *keys,
        "volume_total_hm3",
        "vazao_turbinada_m3s",
        "vazao_vertida_m3s",
        "vazao_lateral_usina_m3s",
        "geracao_hidraulica_fph",
    ).iter_rows():
        key = (code, estagio - 1)
        if key not in physics or fph is None:
            continue
        downstream = eff.downstream_plant(code, estagio - 1)
        head = _head(
            hydraulics[code],
            physics[key],
            v=v,
            q=q,
            s=s,
            lateral_m3s=lateral or 0.0,
            downstream_level_m=forebay.get((downstream, estagio, no, patamar)),
        )
        worst[key] = max(worst.get(key, 0.0), abs(head.generation_mw - fph))
    return worst


def _source_envelope(
    planes: pl.DataFrame,
    v_useful: np.ndarray,
    q: np.ndarray,
    s: np.ndarray,
    lateral: np.ndarray,
) -> np.ndarray:
    def _col(name: str) -> np.ndarray:
        return planes.get_column(name).to_numpy()[:, None]

    values = _col("fator_correcao") * (
        _col("rhs")
        + _col("coeficiente_volume_util") * v_useful[None, :]
        + _col("coeficiente_vazao_turbinada") * q[None, :]
        + _col("coeficiente_vazao_vertida") * s[None, :]
        + _col("coeficiente_vazao_lateral") * lateral[None, :]
    )
    return values.min(axis=0)


def _with_envelopes(
    points: pl.DataFrame,
    source_by_key: Mapping[tuple[int, int], pl.DataFrame],
    novomodelo_by_key: Mapping[tuple[int, int], pl.DataFrame],
    code_to_novomodelo: Mapping[int, int],
    hydraulics: Mapping[int, _Hydraulics],
) -> pl.DataFrame:
    """Both models' envelopes at the source model's points."""
    parts: list[pl.DataFrame] = []
    for (code, stage), group in points.partition_by(
        ["code", "stage"], as_dict=True
    ).items():
        v = group.get_column("v_hm3").to_numpy()
        q = group.get_column("q_m3s").to_numpy()
        s = group.get_column("s_m3s").to_numpy()
        lateral = group.get_column("lateral_m3s").to_numpy()
        source_mw = _source_envelope(
            source_by_key[(code, stage)],
            v - hydraulics[code].hidr_min_storage_hm3,
            q,
            s,
            lateral,
        )
        novomodelo_mw = dense_grid(
            *novomodelo_plane_arrays(
                novomodelo_by_key[(code_to_novomodelo[code], stage)]
            ),
            v,
            q,
            s,
        )
        parts.append(
            group.with_columns(
                pl.Series("source_planes_mw", source_mw, dtype=pl.Float64),
                pl.Series("novomodelo_planes_mw", novomodelo_mw, dtype=pl.Float64),
            )
        )
    return pl.concat(parts)


def _mean_points(points: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Each (plant, stage)'s weighted mean operating point over every point."""
    return points.group_by("novomodelo_id", "stage").agg(
        pl.col("code").first(),
        *[
            _wmean(c).alias(f"{prefix}_{c}")
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
    eco: pl.DataFrame | None,
    novomodelo_output_dir: Path,
    source_by_key: Mapping[tuple[int, int], pl.DataFrame],
    novomodelo_by_key: Mapping[tuple[int, int], pl.DataFrame],
    hydraulics: Mapping[int, _Hydraulics],
    physics: Mapping[tuple[int, int], _StagePhysics],
    exact_check: Mapping[tuple[int, int], float],
    names: Mapping[int, str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    windows = (
        {}
        if eco is None
        else {
            (int(r["codigo_usina"]), int(r["estagio"]) - 1): r
            for r in eco.iter_rows(named=True)
        }
    )
    novomodelo_windows = {
        (int(r["hydro_id"]), int(r["stage_id"])): r
        for r in novomodelo_readers.read_novomodelo_fpha_fit_config(
            novomodelo_output_dir
        ).iter_rows(named=True)
    }
    static_q_max, stage_q_max = _novomodelo_turbined_limits(novomodelo_output_dir)

    rows: list[dict[str, object]] = []
    curves: list[tuple[int, int, str, float, float]] = []
    for m in means.sort("novomodelo_id", "stage").iter_rows(named=True):
        code, novomodelo_id, stage = (
            int(m["code"]),
            int(m["novomodelo_id"]),
            int(m["stage"]),
        )
        hyd, phys = hydraulics[code], physics[(code, stage)]
        eco_row = windows.get((code, stage), {})
        novomodelo_window = novomodelo_windows.get((novomodelo_id, stage), {})
        novomodelo_planes = novomodelo_by_key[(novomodelo_id, stage)]
        static_limit = static_q_max.get(novomodelo_id)
        fit_q_max = novomodelo_window.get("turbined_fit_max_m3s")
        novomodelo_q_max = static_limit if fit_q_max is None else fit_q_max
        novomodelo_q_stage_max = stage_q_max.get((novomodelo_id, stage), static_limit)

        def _at_mean(prefix: str, q: float | None = None) -> _Head:
            level = m[f"{prefix}_downstream_level_m"]
            return _head(
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
            "source_v_min_hm3": eco_row.get("volume_armazenado_minimo"),
            "source_v_max_hm3": eco_row.get("volume_armazenado_maximo"),
            "novomodelo_v_min_hm3": novomodelo_window.get("volume_min_hm3"),
            "novomodelo_v_max_hm3": novomodelo_window.get("volume_max_hm3"),
            "source_q_max_m3s": eco_row.get("vazao_turbinada_maxima"),
            "novomodelo_q_max_m3s": novomodelo_q_max,
            "novomodelo_q_stage_max_m3s": novomodelo_q_stage_max,
            "novomodelo_spill_fit_max_m3s": novomodelo_window.get(
                "spillage_fit_max_m3s"
            ),
            "source_grid": (
                f"{int(eco_row['numero_pontos_volume_armazenado'])}"
                f" × {int(eco_row['numero_pontos_vazao_turbinada'])}"
                if eco_row
                else None
            ),
            "n_planes_source": source_by_key[(code, stage)].height,
            "n_planes_novomodelo": novomodelo_planes.height,
            "source_gh_max_mw": eco_row.get("geracao_maxima"),
            "exact_check_max_mw": exact_check.get((code, stage)),
        }
        for prefix in ("source", "novomodelo"):
            head = heads.get(prefix)
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
            row[f"{prefix}_forebay_m"] = None if head is None else head.forebay_m
            row[f"{prefix}_tailrace_m"] = None if head is None else head.tailrace_m
            row[f"{prefix}_loss_m"] = (
                None
                if head is None
                else head.forebay_m - head.tailrace_m - head.net_head_m
            )
            row[f"{prefix}_net_head_m"] = None if head is None else head.net_head_m
            row[f"{prefix}_rho_exact"] = (
                None if head is None else phys.rho_esp * head.net_head_m
            )
        rows.append(row)

        q_top = max(
            float(v)
            for v in (
                eco_row.get("vazao_turbinada_maxima"),
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
            "source_planes": _source_envelope(
                source_by_key[(code, stage)],
                v_src - hyd.hidr_min_storage_hm3,
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


def _folded_source_planes(
    planes: pl.DataFrame, source_eval: pl.DataFrame, pairs: set[tuple[int, int]]
) -> pl.DataFrame:
    """The source model's planes in the shared surface comparison's shape, the
    lateral term folded into the intercept at the stage's mean lateral flow."""
    lateral = source_eval.group_by("code", "stage").agg(
        _wmean("lateral_m3s").alias("_lateral")
    )
    keep = pl.DataFrame(
        sorted(pairs), schema={"code": pl.Int64, "stage": pl.Int64}, orient="row"
    )
    return (
        planes.join(keep, on=["code", "stage"])
        .join(lateral, on=["code", "stage"], how="left")
        .select(
            pl.col("code").alias("newave_code"),
            pl.col("estagio").alias("periodo"),
            (
                pl.col("rhs")
                + pl.col("coeficiente_vazao_lateral")
                * pl.col("_lateral").fill_null(0.0)
            ).alias("gamma_0"),
            pl.col("coeficiente_volume_util").alias("gamma_v"),
            pl.col("coeficiente_vazao_turbinada").alias("gamma_q"),
            pl.col("coeficiente_vazao_vertida").alias("gamma_s"),
            "fator_correcao",
        )
    )


def _source_grid(
    eco: pl.DataFrame, hydraulics: Mapping[int, _Hydraulics]
) -> pl.DataFrame:
    """The source model's fit windows in the shared surface comparison's shape."""
    offsets = pl.DataFrame(
        [(code, h.hidr_min_storage_hm3) for code, h in hydraulics.items()],
        schema={"newave_code": pl.Int64, "volume_offset_hm3": pl.Float64},
        orient="row",
    )
    return eco.select(
        pl.col("codigo_usina").alias("newave_code"),
        pl.col("estagio").alias("periodo"),
        pl.col("volume_armazenado_minimo").alias("v_min_hm3"),
        pl.col("volume_armazenado_maximo").alias("v_max_hm3"),
        pl.col("vazao_turbinada_minima").alias("q_min_m3s"),
        pl.col("vazao_turbinada_maxima").alias("q_max_m3s"),
        pl.col("numero_pontos_volume_armazenado").cast(pl.Int64).alias("n_v"),
        pl.col("numero_pontos_vazao_turbinada").cast(pl.Int64).alias("n_q"),
    ).join(offsets, on="newave_code")
