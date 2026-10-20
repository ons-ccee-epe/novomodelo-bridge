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
from typing import NamedTuple

import pandas as pd
import polars as pl

_LOG = logging.getLogger(__name__)

_ENTITY_RE = re.compile(r"\bid=(\d+)")
_BLOCK_RE = re.compile(r"\bblock=(\d+)")

#: Stage-level storage terms: evaluated on the block-0 simulation row.
STORAGE_KINDS = frozenset(
    {"hydro_storage", "hydro_storage_final", "hydro_useful_volume_final"}
)
USEFUL_VOLUME_KINDS = frozenset({"hydro_useful_volume_final"})

#: Per-block terms: ``kind -> (simulation entity, id column, value column)``.
_BLOCK_KINDS: dict[str, tuple[str, str, str]] = {
    "hydro_generation": ("hydros", "hydro_id", "generation_mw"),
    "line_exchange": ("exchanges", "line_id", "net_flow_mw"),
    "line_direct": ("exchanges", "line_id", "direct_flow_mw"),
    "line_reverse": ("exchanges", "line_id", "reverse_flow_mw"),
}

#: Every term kind the LHS evaluators can value from simulation output.
SUPPORTED_KINDS = STORAGE_KINDS | frozenset(_BLOCK_KINDS)

TERMS_SCHEMA: dict[str, pl.DataType] = {
    "constraint_id": pl.Int32,
    "stage_id": pl.Int32,
    "block_id": pl.Int32,
    "term_index": pl.Int32,
    "variable_kind": pl.String,
    "entity_id": pl.Int32,
    "term_block_id": pl.Int32,
    "coefficient": pl.Float64,
}
BOUNDS_SCHEMA: dict[str, pl.DataType] = {
    "constraint_id": pl.Int32,
    "stage_id": pl.Int32,
    "block_id": pl.Int32,
    "bound_lower": pl.Float64,
    "bound_upper": pl.Float64,
}
_LHS_COLUMNS = ["constraint_id", "scenario_id", "stage_id", "block_id", "lhs_value"]


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
        .group_by(["constraint_id", "stage_id", "block_id"])
        .agg(
            (pl.col("coefficient") * pl.col("vmin").fill_null(0.0)).sum().alias("dead")
        )
    )
    keys = ["constraint_id", "stage_id", "block_id"]
    bounds = (
        echo.with_columns(pl.col("constraint_id").cast(pl.Int32))
        .select([*keys, "bound_lower", "bound_upper"])
        .unique(subset=keys, keep="first")
        .join(dead, on=keys, how="left", nulls_equal=True)
        .with_columns(
            (pl.col("bound_lower") - pl.col("dead").fill_null(0.0)),
            (pl.col("bound_upper") - pl.col("dead").fill_null(0.0)),
        )
        .select(list(BOUNDS_SCHEMA))
        .cast(BOUNDS_SCHEMA)
        .sort(keys)
    )
    return ResolvedGenericConstraints(constraints, terms, bounds, min_storage)


def unsupported_constraint_ids(terms: pl.DataFrame) -> set[int]:
    """Ids of constraints with a term kind no evaluator can value."""
    return set(
        terms.filter(~pl.col("variable_kind").is_in(list(SUPPORTED_KINDS)))[
            "constraint_id"
        ].to_list()
    )


def evaluate_lhs(
    resolved: ResolvedGenericConstraints,
    hydros_lf: pl.LazyFrame,
    exchanges_lf: pl.LazyFrame,
) -> pd.DataFrame:
    """Evaluate every shown constraint's LHS from Novomodelo simulation output.

    Storage terms take the stage-final ``storage_final_hm3`` of the block-0 row
    (less ``min_storage_hm3`` for a useful-volume term); per-block terms take
    their block's ``generation_mw`` / ``net_flow_mw`` / ``direct_flow_mw`` /
    ``reverse_flow_mw``. A stage-level row with only storage terms yields one
    row per (scenario, stage) with ``block_id = 0``; any other one row per
    (scenario, stage, block). A constraint carrying a term kind outside
    :data:`SUPPORTED_KINDS` is skipped whole rather than evaluated partially.

    Returns a DataFrame with columns ``constraint_id``, ``scenario_id``,
    ``stage_id``, ``block_id``, ``lhs_value``.
    """
    terms = resolved.terms
    skipped = unsupported_constraint_ids(terms)
    if skipped:
        _LOG.info(
            "%d generic constraint(s) carry a term the simulation output cannot "
            "value; their LHS is not evaluated.",
            len(skipped),
        )
        terms = terms.filter(~pl.col("constraint_id").is_in(list(skipped)))
    if terms.is_empty():
        return pd.DataFrame(columns=_LHS_COLUMNS)

    storage_terms = terms.filter(pl.col("variable_kind").is_in(list(STORAGE_KINDS)))
    block_terms = terms.filter(~pl.col("variable_kind").is_in(list(STORAGE_KINDS)))

    parts: list[pl.DataFrame] = []
    if not storage_terms.is_empty():
        parts.append(_storage_contributions(storage_terms, hydros_lf, resolved))
    if not block_terms.is_empty():
        parts.append(_block_contributions(block_terms, hydros_lf, exchanges_lf))
    contributions = pl.concat(parts, how="vertical_relaxed")

    has_block_term = set(block_terms["constraint_id"].unique().to_list())
    per_block = contributions.filter(pl.col("block_id").is_not_null())
    stage_level = contributions.filter(pl.col("block_id").is_null())

    # Storage-only constraints: one stage-level row at block 0.
    storage_only = (
        stage_level.filter(~pl.col("constraint_id").is_in(list(has_block_term)))
        .group_by(["constraint_id", "scenario_id", "stage_id"])
        .agg(pl.col("value").sum().alias("lhs_value"))
        .with_columns(pl.lit(0, dtype=pl.Int32).alias("block_id"))
    )
    # Mixed constraints: storage terms broadcast onto each evaluated block.
    blocks = per_block.select(
        "constraint_id", "scenario_id", "stage_id", "block_id"
    ).unique()
    broadcast = (
        stage_level.filter(pl.col("constraint_id").is_in(list(has_block_term)))
        .join(blocks, on=["constraint_id", "scenario_id", "stage_id"], how="inner")
        .with_columns(pl.col("block_id_right").alias("block_id"))
        .drop("block_id_right")
    )
    mixed = (
        pl.concat([per_block, broadcast.select(per_block.columns)])
        .group_by(["constraint_id", "scenario_id", "stage_id", "block_id"])
        .agg(pl.col("value").sum().alias("lhs_value"))
    )
    out = pl.concat(
        [storage_only.select(_LHS_COLUMNS), mixed.select(_LHS_COLUMNS)],
        how="vertical_relaxed",
    ).sort(["constraint_id", "scenario_id", "stage_id", "block_id"])
    return out.to_pandas()


def _storage_contributions(
    terms: pl.DataFrame,
    hydros_lf: pl.LazyFrame,
    resolved: ResolvedGenericConstraints,
) -> pl.DataFrame:
    """``coefficient × storage`` per (constraint, scenario, stage, row block).

    The block is the echo row's own (null for a stage-level row).
    """
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
        )
        .select("constraint_id", "scenario_id", "stage_id", "block_id", "value")
    )


def _block_contributions(
    terms: pl.DataFrame,
    hydros_lf: pl.LazyFrame,
    exchanges_lf: pl.LazyFrame,
) -> pl.DataFrame:
    """``coefficient × per-block value`` per (constraint, scenario, stage, block)."""
    frames = {"hydros": hydros_lf, "exchanges": exchanges_lf}
    parts: list[pl.DataFrame] = []
    for kind, (entity, id_col, value_col) in _BLOCK_KINDS.items():
        kind_terms = terms.filter(pl.col("variable_kind") == kind)
        if kind_terms.is_empty():
            continue
        values = (
            frames[entity]
            .filter(pl.col(id_col).is_in(kind_terms["entity_id"].unique().to_list()))
            .select(
                pl.col("scenario_id"),
                pl.col("stage_id").cast(pl.Int32),
                pl.col("block_id").cast(pl.Int32).alias("value_block_id"),
                pl.col(id_col).cast(pl.Int32).alias("entity_id"),
                pl.col(value_col).alias("raw"),
            )
            .collect(engine="streaming")
        )
        # A term pinned to a block reads that block; otherwise the row's own
        # block, or every block for a stage-level row.
        wanted = pl.coalesce(pl.col("term_block_id"), pl.col("block_id"))
        parts.append(
            kind_terms.join(values, on=["stage_id", "entity_id"], how="inner")
            .filter(wanted.is_null() | (wanted == pl.col("value_block_id")))
            .with_columns(
                (pl.col("coefficient") * pl.col("raw").fill_null(0.0)).alias("value"),
                pl.col("value_block_id").alias("block_id"),
            )
            .select("constraint_id", "scenario_id", "stage_id", "block_id", "value")
        )
    if not parts:
        return pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "scenario_id": pl.Int64,
                "stage_id": pl.Int32,
                "block_id": pl.Int32,
                "value": pl.Float64,
            }
        )
    return pl.concat(parts, how="vertical_relaxed")
