"""Resolved generic constraints and their LHS evaluation (presentation-free).

novomodelo writes the generic constraints it actually built into
``output/generic_constraints/resolved_echo.parquet``: one row per LHS term per
active ``(stage, block)`` row, with the term's coefficient and the row's bounds
exactly as the LP received them — every ``@name`` parameter and arithmetic
over parameters already resolved, and repeated references to one variable
merged into one term. This module turns that echo into the comparison-ready
view both the dashboard and the comparator render, so neither re-derives a
coefficient from the expression text.

Useful-volume terms are shown as useful volume. novomodelo resolves
``hydro_useful_volume_final`` to the absolute storage column and moves the
dead-volume term ``coefficient × min_storage`` onto the bound. The view keeps
the useful volume on the LHS and removes that term from the bound, so a VminOP
row reads as useful stored energy (MWh) against the curve fraction of its
maximum.

It lives at the package top level so neither consumer imports across the
comparator↔UI boundary.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import NamedTuple

import pandas as pd
import polars as pl

from novomodelo_bridge.core.units import HM3_PER_M3S_HOUR

_LOG = logging.getLogger(__name__)

_ENTITY_RE = re.compile(r"\bid=(\d+)")
_BLOCK_RE = re.compile(r"\bblock=(\d+)")
_BUS_RE = re.compile(r"\bbus=(\d+)")

#: Stage-level storage terms: evaluated on the block-0 simulation row.
STORAGE_KINDS = frozenset(
    {"hydro_storage", "hydro_storage_final", "hydro_useful_volume_final"}
)
USEFUL_VOLUME_KINDS = frozenset({"hydro_useful_volume_final"})


class _Source(NamedTuple):
    """Where a term's value lives in the simulation output."""

    entity: str
    id_col: str
    value_col: str


#: Per-block terms: the simulation column each kind reads at its block.
_BLOCK_SOURCES: dict[str, _Source] = {
    "hydro_generation": _Source("hydros", "hydro_id", "generation_mw"),
    "hydro_turbined": _Source("hydros", "hydro_id", "turbined_m3s"),
    "hydro_spillage": _Source("hydros", "hydro_id", "spillage_m3s"),
    "hydro_outflow": _Source("hydros", "hydro_id", "outflow_m3s"),
    "hydro_diversion": _Source("hydros", "hydro_id", "diverted_outflow_m3s"),
    "thermal_generation": _Source("thermals", "thermal_id", "generation_mw"),
    "line_exchange": _Source("exchanges", "line_id", "net_flow_mw"),
    "line_direct": _Source("exchanges", "line_id", "direct_flow_mw"),
    "line_reverse": _Source("exchanges", "line_id", "reverse_flow_mw"),
    "pumping_flow": _Source(
        "pumping_stations", "pumping_station_id", "pumped_flow_m3s"
    ),
}

#: A term naming one bus cell of a split plant (``bus=k``) reads that cell.
_BUS_CELL_SOURCES: dict[str, _Source] = {
    "hydro_generation": _Source("hydro_bus_generation", "hydro_id", "generation_mw"),
    "hydro_turbined": _Source("hydro_bus_generation", "hydro_id", "turbined_m3s"),
}

#: Stage volumes (hm³): their flow's per-block m³/s weighted by block hours.
_VOLUME_SOURCES: dict[str, _Source] = {
    "hydro_outflow_volume": _BLOCK_SOURCES["hydro_outflow"],
    "hydro_diversion_volume": _BLOCK_SOURCES["hydro_diversion"],
    "pumping_volume": _BLOCK_SOURCES["pumping_flow"],
}

#: Every term kind the LHS evaluator can value from simulation output.
SUPPORTED_KINDS = STORAGE_KINDS | frozenset(_BLOCK_SOURCES) | frozenset(_VOLUME_SOURCES)

#: The simulation output directories the LHS evaluator reads.
LHS_ENTITIES: tuple[str, ...] = tuple(
    dict.fromkeys(
        ["hydros"]
        + [s.entity for s in _BLOCK_SOURCES.values()]
        + [s.entity for s in _BUS_CELL_SOURCES.values()]
    )
)

TERMS_SCHEMA = pl.Schema(
    {
        "constraint_id": pl.Int32,
        "stage_id": pl.Int32,
        "block_id": pl.Int32,
        "term_index": pl.Int32,
        "variable_kind": pl.String,
        "entity_id": pl.Int32,
        "term_block_id": pl.Int32,
        "bus_id": pl.Int32,
        "coefficient": pl.Float64,
    }
)
BOUNDS_SCHEMA = pl.Schema(
    {
        "constraint_id": pl.Int32,
        "stage_id": pl.Int32,
        "block_id": pl.Int32,
        "bound_lower": pl.Float64,
        "bound_upper": pl.Float64,
    }
)
BLOCK_HOURS_SCHEMA = pl.Schema(
    {"stage_id": pl.Int32, "block_id": pl.Int32, "hours": pl.Float64}
)
_LHS_COLUMNS = ["constraint_id", "scenario_id", "stage_id", "block_id", "lhs_value"]
_ROW_KEYS = ["constraint_id", "stage_id", "block_id"]
_CONTRIBUTION_SCHEMA = pl.Schema(
    {
        "constraint_id": pl.Int32,
        "stage_id": pl.Int32,
        "block_id": pl.Int32,
        "term_index": pl.Int32,
        "scenario_id": pl.Int64,
        "out_block": pl.Int32,
        "value": pl.Float64,
    }
)


class ResolvedGenericConstraints(NamedTuple):
    """The comparison-ready view of a case's generic constraints.

    ``constraints`` holds the declared constraint dicts; ``terms`` and
    ``bounds`` are keyed by their ids; ``min_storage`` is the per-hydro
    ``min_storage_hm3`` the useful-volume terms subtract.
    """

    constraints: list[dict]
    terms: pl.DataFrame
    bounds: pl.DataFrame
    min_storage: dict[int, float]


def resolve_generic_constraints(
    constraints: list[dict],
    echo: pl.DataFrame | None,
    min_storage: dict[int, float],
) -> ResolvedGenericConstraints:
    """Build the comparison-ready view from the declared constraints and the echo.

    With no echo the view carries the constraints but no terms or bounds.
    """
    if echo is None or echo.is_empty():
        return ResolvedGenericConstraints(
            constraints,
            pl.DataFrame(schema=TERMS_SCHEMA),
            pl.DataFrame(schema=BOUNDS_SCHEMA),
            min_storage,
        )

    rows = echo.filter(pl.col("coefficient").is_not_null()).with_columns(
        pl.col("constraint_id").cast(pl.Int32),
        pl.col("variable")
        .str.extract(_ENTITY_RE.pattern, 1)
        .cast(pl.Int32)
        .alias("entity_id"),
        pl.col("variable")
        .str.extract(_BLOCK_RE.pattern, 1)
        .cast(pl.Int32)
        .alias("term_block_id"),
        pl.col("variable")
        .str.extract(_BUS_RE.pattern, 1)
        .cast(pl.Int32)
        .alias("bus_id"),
    )
    terms = rows.select(list(TERMS_SCHEMA)).cast(TERMS_SCHEMA)

    vmin = pl.DataFrame(
        {
            "entity_id": list(min_storage),
            "vmin": list(min_storage.values()),
        },
        schema={"entity_id": pl.Int32, "vmin": pl.Float64},
    )
    dead = (
        terms.filter(pl.col("variable_kind").is_in(list(USEFUL_VOLUME_KINDS)))
        .join(vmin, on="entity_id", how="left")
        .group_by(_ROW_KEYS)
        .agg(
            (pl.col("coefficient") * pl.col("vmin").fill_null(0.0)).sum().alias("dead")
        )
    )
    bounds = (
        echo.with_columns(pl.col("constraint_id").cast(pl.Int32))
        .select([*_ROW_KEYS, "bound_lower", "bound_upper"])
        .unique(subset=_ROW_KEYS, keep="first")
        .join(dead, on=_ROW_KEYS, how="left", nulls_equal=True)
        .with_columns(
            (pl.col("bound_lower") - pl.col("dead").fill_null(0.0)),
            (pl.col("bound_upper") - pl.col("dead").fill_null(0.0)),
        )
        .select(list(BOUNDS_SCHEMA))
        .cast(BOUNDS_SCHEMA)
        .sort(_ROW_KEYS)
    )
    return ResolvedGenericConstraints(constraints, terms, bounds, min_storage)


def unsupported_constraint_ids(
    terms: pl.DataFrame, supported: frozenset[str] = SUPPORTED_KINDS
) -> set[int]:
    """Ids of constraints with a term kind outside *supported*."""
    return set(
        terms.filter(~pl.col("variable_kind").is_in(list(supported)))[
            "constraint_id"
        ].to_list()
    )


def required_entities(terms: pl.DataFrame) -> set[str]:
    """The simulation output directories *terms* read from."""
    out: set[str] = set()
    for kind, bus in terms.select("variable_kind", "bus_id").unique().iter_rows():
        if kind in STORAGE_KINDS:
            out.add("hydros")
        elif kind in _VOLUME_SOURCES:
            out.add(_VOLUME_SOURCES[kind].entity)
        elif bus is not None and kind in _BUS_CELL_SOURCES:
            out.add(_BUS_CELL_SOURCES[kind].entity)
        elif kind in _BLOCK_SOURCES:
            out.add(_BLOCK_SOURCES[kind].entity)
    return out


def evaluate_lhs(
    resolved: ResolvedGenericConstraints,
    frames: Mapping[str, pl.LazyFrame],
    block_hours: pl.DataFrame,
) -> pd.DataFrame:
    """Evaluate every shown constraint's LHS from Novomodelo simulation output.

    *frames* maps a simulation output directory (see :data:`LHS_ENTITIES`) to
    its scan; *block_hours* (:data:`BLOCK_HOURS_SCHEMA`) turns a stage-volume
    term's per-block m³/s into hm³. Storage terms take the stage-final
    ``storage_final_hm3`` of the block-0 row (less ``min_storage_hm3`` for a
    useful-volume term); per-block terms take their block's column (see
    :data:`_BLOCK_SOURCES`), a bus-cell term its cell's. An echo row with only
    stage-level terms yields one row per (scenario, stage) with
    ``block_id = 0``; any other one row per (scenario, stage, block).

    A constraint with a term kind outside :data:`SUPPORTED_KINDS`, or whose
    simulation output is absent, is skipped whole, and a row whose terms are
    not all found in the output is dropped: the LHS is never a partial sum.

    Returns a DataFrame with columns ``constraint_id``, ``scenario_id``,
    ``stage_id``, ``block_id``, ``lhs_value``.
    """
    terms = resolved.terms
    available = {
        entity
        for entity, lf in frames.items()
        if entity in LHS_ENTITIES and lf.collect_schema().names()
    }
    no_output = {
        int(cid)
        for (cid,), group in terms.group_by("constraint_id")
        if not required_entities(group) <= available
    }
    skipped = unsupported_constraint_ids(terms) | no_output
    if skipped:
        _LOG.info(
            "%d generic constraint(s) carry a term the simulation output cannot "
            "value; their LHS is not evaluated.",
            len(skipped),
        )
        terms = terms.filter(~pl.col("constraint_id").is_in(list(skipped)))
    if terms.is_empty():
        return pd.DataFrame(columns=_LHS_COLUMNS)

    kind = pl.col("variable_kind")
    bus_cell = pl.col("bus_id").is_not_null() & kind.is_in(list(_BUS_CELL_SOURCES))
    parts = [
        _storage_contributions(
            terms.filter(kind.is_in(list(STORAGE_KINDS))),
            frames.get("hydros", pl.LazyFrame()),
            resolved,
        ),
        _volume_contributions(
            terms.filter(kind.is_in(list(_VOLUME_SOURCES))), frames, block_hours
        ),
        _block_contributions(
            terms.filter(kind.is_in(list(_BLOCK_SOURCES)) & ~bus_cell),
            frames,
            _BLOCK_SOURCES,
        ),
        _block_contributions(terms.filter(bus_cell), frames, _BUS_CELL_SOURCES),
    ]
    found = [p.cast(_CONTRIBUTION_SCHEMA) for p in parts if not p.is_empty()]
    if not found:
        return pd.DataFrame(columns=_LHS_COLUMNS)
    contributions = pl.concat(found, how="vertical")

    # A stage-level contribution joins each block the same echo row is
    # evaluated at; a row with none is one stage-level row at block 0.
    group = ["constraint_id", "stage_id", "block_id", "scenario_id"]
    per_block = contributions.filter(pl.col("out_block").is_not_null())
    stage_level = contributions.filter(pl.col("out_block").is_null()).drop("out_block")
    blocks = per_block.select(*group, "out_block").unique()
    rows = pl.concat(
        [
            per_block,
            stage_level.join(blocks, on=group, how="inner", nulls_equal=True).select(
                per_block.columns
            ),
            stage_level.join(blocks, on=group, how="anti", nulls_equal=True)
            .with_columns(pl.lit(0, dtype=pl.Int32).alias("out_block"))
            .select(per_block.columns),
        ]
    )

    expected = terms.group_by(_ROW_KEYS).agg(
        pl.col("term_index").n_unique().alias("n_terms")
    )
    lhs = (
        rows.group_by([*group, "out_block"])
        .agg(
            pl.col("value").sum().alias("lhs_value"),
            pl.col("term_index").n_unique().alias("n_found"),
        )
        .join(expected, on=_ROW_KEYS, how="inner", nulls_equal=True)
    )
    incomplete = lhs.filter(pl.col("n_found") < pl.col("n_terms"))
    if not incomplete.is_empty():
        _LOG.info(
            "%d generic constraint(s) reference an entity missing from the "
            "simulation output; those rows are not evaluated.",
            incomplete["constraint_id"].n_unique(),
        )
    out = (
        lhs.filter(pl.col("n_found") == pl.col("n_terms"))
        .with_columns(pl.col("out_block").alias("block_id"))
        .select(_LHS_COLUMNS)
        .sort(["constraint_id", "scenario_id", "stage_id", "block_id"])
    )
    return out.to_pandas()


def _storage_contributions(
    terms: pl.DataFrame,
    hydros_lf: pl.LazyFrame,
    resolved: ResolvedGenericConstraints,
) -> pl.DataFrame:
    """``coefficient × storage`` per term and scenario, at the echo row's block."""
    if terms.is_empty():
        return pl.DataFrame(schema=_CONTRIBUTION_SCHEMA)
    ids = terms["entity_id"].unique().to_list()
    storage = (
        hydros_lf.filter((pl.col("block_id") == 0) & pl.col("hydro_id").is_in(ids))
        .select(
            pl.col("scenario_id"),
            pl.col("stage_id").cast(pl.Int32),
            pl.col("hydro_id").cast(pl.Int32).alias("entity_id"),
            pl.col("storage_final_hm3"),
        )
        .collect(engine="streaming")
    )
    vmin = pl.DataFrame(
        {
            "entity_id": list(resolved.min_storage),
            "vmin": list(resolved.min_storage.values()),
        },
        schema={"entity_id": pl.Int32, "vmin": pl.Float64},
    )
    useful = pl.col("variable_kind").is_in(list(USEFUL_VOLUME_KINDS))
    return (
        terms.join(storage, on=["stage_id", "entity_id"], how="inner")
        .join(vmin, on="entity_id", how="left")
        .with_columns(
            (
                pl.col("coefficient")
                * (
                    pl.col("storage_final_hm3")
                    - pl.when(useful).then(pl.col("vmin").fill_null(0.0)).otherwise(0.0)
                )
            ).alias("value"),
            pl.col("block_id").alias("out_block"),
        )
        .select(list(_CONTRIBUTION_SCHEMA))
    )


def _scan_values(
    terms: pl.DataFrame, lf: pl.LazyFrame, source: _Source, *, by_bus: bool
) -> pl.DataFrame:
    """The per-(scenario, stage, block, entity) values *terms* read from *lf*."""
    columns = [
        pl.col("scenario_id"),
        pl.col("stage_id").cast(pl.Int32),
        pl.col("block_id").cast(pl.Int32).alias("value_block_id"),
        pl.col(source.id_col).cast(pl.Int32).alias("entity_id"),
        pl.col(source.value_col).cast(pl.Float64).alias("raw"),
    ]
    if by_bus:
        columns.append(pl.col("bus_id").cast(pl.Int32))
    return (
        lf.filter(pl.col(source.id_col).is_in(terms["entity_id"].unique().to_list()))
        .select(columns)
        .collect(engine="streaming")
    )


def _volume_contributions(
    terms: pl.DataFrame,
    frames: Mapping[str, pl.LazyFrame],
    block_hours: pl.DataFrame,
) -> pl.DataFrame:
    """``coefficient × Σ_blocks flow × hours`` (hm³) per term and scenario."""
    parts: list[pl.DataFrame] = []
    hours = block_hours.select(
        pl.col("stage_id").cast(pl.Int32),
        pl.col("block_id").cast(pl.Int32).alias("value_block_id"),
        pl.col("hours").cast(pl.Float64),
    )
    for kind, source in _VOLUME_SOURCES.items():
        kind_terms = terms.filter(pl.col("variable_kind") == kind)
        if kind_terms.is_empty():
            continue
        volumes = (
            _scan_values(kind_terms, frames[source.entity], source, by_bus=False)
            .join(hours, on=["stage_id", "value_block_id"], how="inner")
            .group_by(["scenario_id", "stage_id", "entity_id"])
            .agg(
                (pl.col("raw").fill_null(0.0) * pl.col("hours") * HM3_PER_M3S_HOUR)
                .sum()
                .alias("raw")
            )
        )
        parts.append(
            kind_terms.join(volumes, on=["stage_id", "entity_id"], how="inner")
            .with_columns(
                (pl.col("coefficient") * pl.col("raw")).alias("value"),
                pl.col("block_id").alias("out_block"),
            )
            .select(list(_CONTRIBUTION_SCHEMA))
        )
    if not parts:
        return pl.DataFrame(schema=_CONTRIBUTION_SCHEMA)
    return pl.concat(parts, how="vertical_relaxed")


def _block_contributions(
    terms: pl.DataFrame,
    frames: Mapping[str, pl.LazyFrame],
    sources: Mapping[str, _Source],
) -> pl.DataFrame:
    """``coefficient × per-block value`` per term, scenario and evaluated block.

    A term reads its pinned block, else its echo row's block, else (a
    stage-level row) every block. It belongs to its echo row's block; a pinned
    term of a stage-level row stays stage-level.
    """
    parts: list[pl.DataFrame] = []
    by_bus = sources is _BUS_CELL_SOURCES
    for kind, source in sources.items():
        kind_terms = terms.filter(pl.col("variable_kind") == kind)
        if kind_terms.is_empty():
            continue
        values = _scan_values(kind_terms, frames[source.entity], source, by_bus=by_bus)
        keys = ["stage_id", "entity_id", *(["bus_id"] if by_bus else [])]
        read_block = pl.coalesce(pl.col("term_block_id"), pl.col("block_id"))
        out_block = (
            pl.when(pl.col("block_id").is_not_null())
            .then(pl.col("block_id"))
            .when(pl.col("term_block_id").is_not_null())
            .then(pl.lit(None, dtype=pl.Int32))
            .otherwise(pl.col("value_block_id"))
        )
        parts.append(
            kind_terms.join(values, on=keys, how="inner")
            .filter(read_block.is_null() | (read_block == pl.col("value_block_id")))
            .with_columns(
                (pl.col("coefficient") * pl.col("raw").fill_null(0.0)).alias("value"),
                out_block.alias("out_block"),
            )
            .select(list(_CONTRIBUTION_SCHEMA))
        )
    if not parts:
        return pl.DataFrame(schema=_CONTRIBUTION_SCHEMA)
    return pl.concat(parts, how="vertical_relaxed")


def stage_weighted_lhs(lhs: pd.DataFrame, block_hours: pl.DataFrame) -> pl.DataFrame:
    """Hours-weighted stage LHS per (constraint, scenario, stage).

    Each scenario's stage value is ``Σ hours_b × lhs_b / Σ hours_b`` over the
    blocks *lhs* holds for that stage, so a stage-level row reads as itself.
    """
    schema = {
        "constraint_id": pl.Int32,
        "scenario_id": pl.Int64,
        "stage_id": pl.Int32,
        "lhs_value": pl.Float64,
    }
    if lhs.empty:
        return pl.DataFrame(schema=schema)
    return (
        pl.from_pandas(lhs[_LHS_COLUMNS])
        .cast({"constraint_id": pl.Int32, "stage_id": pl.Int32, "block_id": pl.Int32})
        .join(_hours(block_hours), on=["stage_id", "block_id"], how="inner")
        .group_by(["constraint_id", "scenario_id", "stage_id"])
        .agg(
            (
                (pl.col("lhs_value") * pl.col("hours")).sum() / pl.col("hours").sum()
            ).alias("lhs_value")
        )
        .cast(schema)
        .sort(["constraint_id", "scenario_id", "stage_id"])
    )


def stage_weighted_bounds(
    bounds: pl.DataFrame, block_hours: pl.DataFrame
) -> pl.DataFrame:
    """Hours-weighted stage bounds per (constraint, stage), ``block_id`` null.

    The hours-weighted stage LHS of a row family bounded block by block is
    bounded by the hours-weighted block bounds. An endpoint some block of the
    stage leaves open bounds nothing at stage level, so it stays null. A
    stage-level row keeps its bounds.
    """
    if bounds.is_empty():
        return pl.DataFrame(schema=BOUNDS_SCHEMA)
    stage_rows = bounds.filter(pl.col("block_id").is_null())
    block_rows = bounds.filter(pl.col("block_id").is_not_null()).join(
        _hours(block_hours), on=["stage_id", "block_id"], how="inner"
    )

    def weighted(col: str) -> pl.Expr:
        return (
            pl.when(pl.col(col).is_not_null().all())
            .then((pl.col(col) * pl.col("hours")).sum() / pl.col("hours").sum())
            .otherwise(None)
            .alias(col)
        )

    weighted_rows = (
        block_rows.group_by(["constraint_id", "stage_id"])
        .agg(weighted("bound_lower"), weighted("bound_upper"))
        .with_columns(pl.lit(None, dtype=pl.Int32).alias("block_id"))
    )
    return (
        pl.concat(
            [
                stage_rows.select(list(BOUNDS_SCHEMA)).cast(BOUNDS_SCHEMA),
                weighted_rows.select(list(BOUNDS_SCHEMA)).cast(BOUNDS_SCHEMA),
            ]
        )
        .unique(subset=["constraint_id", "stage_id"], keep="first", maintain_order=True)
        .sort(["constraint_id", "stage_id"])
    )


def _hours(block_hours: pl.DataFrame) -> pl.DataFrame:
    return block_hours.select(list(BLOCK_HOURS_SCHEMA)).cast(BLOCK_HOURS_SCHEMA)
