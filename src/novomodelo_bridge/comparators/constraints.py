"""Shared novomodelo-side generic-constraint loaders, LHS evaluator, and bound resolution."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import polars as pl

from novomodelo_bridge.core.generic_constraint_format import shape_from_bounds
from novomodelo_bridge.novomodelo import readers as novomodelo_readers
from novomodelo_bridge.novomodelo.constraint_expr import (
    ResolvedGenericConstraints,
    evaluate_lhs,
    required_entities,
    resolve_generic_constraints,
    stage_weighted_bounds,
    stage_weighted_lhs,
)
from novomodelo_bridge.novomodelo.readers import scan_simulation_entity

_LOG = logging.getLogger(__name__)


def load_generic_constraints(novomodelo_input_dir: Path) -> list[dict]:
    """Load constraint definitions from ``constraints/generic_constraints.json``.

    The objects are sense-free (no ``sense`` key); direction is derived
    from the companion bounds table's endpoints when a label is needed (see
    :func:`per_stage_bounds` / :func:`novomodelo_bridge.core.generic_constraint_format.
    shape_from_bounds`). Returns an empty list when the file is missing or
    malformed.
    """
    path = novomodelo_input_dir / "constraints" / "generic_constraints.json"
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("generic_constraints.json could not be parsed: %s", exc)
        return []
    return list(data.get("constraints", []))


def load_resolved_generic_constraints(
    novomodelo_case_dir: Path, novomodelo_output_dir: Path
) -> ResolvedGenericConstraints:
    """The case's generic constraints with the coefficients and bounds novomodelo built.

    See :func:`novomodelo_bridge.novomodelo.constraint_expr.resolve_generic_constraints`;
    terms and bounds are empty when the run wrote no
    ``generic_constraints/resolved_echo.parquet``.
    """
    min_storage = {
        hid: float(meta.get("min_storage_hm3") or 0.0)
        for hid, meta in novomodelo_readers.read_novomodelo_hydro_metadata(
            novomodelo_output_dir
        ).items()
    }
    return resolve_generic_constraints(
        load_generic_constraints(novomodelo_case_dir),
        novomodelo_readers.read_novomodelo_generic_constraint_echo(
            novomodelo_output_dir
        ),
        min_storage,
    )


_LHS_SCHEMA = pl.Schema(
    {
        "constraint_id": pl.Int32,
        "stage_id": pl.Int32,
        "lhs_value": pl.Float64,
    }
)


def evaluate_lhs_novomodelo(
    resolved: ResolvedGenericConstraints,
    novomodelo_output_dir: Path,
) -> pl.DataFrame:
    """Each constraint's hours-weighted stage LHS from Novomodelo simulation outputs.

    Evaluates the per-block LHS with
    :func:`novomodelo_bridge.novomodelo.constraint_expr.evaluate_lhs`, weights
    each scenario's blocks by their hours into one stage value (the
    source models report block-collapsed, duration-weighted stage values),
    and takes the expectation across scenarios per (constraint, stage).

    Returns
    -------
    polars.DataFrame
        Columns: ``constraint_id`` (Int32), ``stage_id`` (Int32),
        ``lhs_value`` (Float64).  Empty when no simulation data is
        available or no constraint has an evaluable term.
    """
    empty = pl.DataFrame(schema=_LHS_SCHEMA)
    if resolved.terms.is_empty():
        return empty

    # A present-but-corrupt parquet raises NovomodeloReadError.
    frames = {
        entity: lf
        for entity in sorted(required_entities(resolved.terms))
        if (lf := scan_simulation_entity(novomodelo_output_dir, entity)) is not None
    }
    block_hours = novomodelo_readers.read_novomodelo_block_hours(novomodelo_output_dir)
    if block_hours.is_empty():
        _LOG.warning(
            "No block hours in %s; generic-constraint LHS cannot be "
            "hours-weighted and is not compared.",
            novomodelo_output_dir,
        )
        return empty
    lhs_pd: pd.DataFrame = evaluate_lhs(resolved, frames, block_hours)
    if lhs_pd.empty:
        return empty

    weights = novomodelo_readers.read_novomodelo_scenario_weights(novomodelo_output_dir)
    return (
        weights.attach(stage_weighted_lhs(lhs_pd, block_hours))
        .group_by(["constraint_id", "stage_id"])
        .agg(weights.mean("lhs_value"))
        .cast(_LHS_SCHEMA)
        .sort(["constraint_id", "stage_id"])
    )


def stage_bounds_novomodelo(
    resolved: ResolvedGenericConstraints, novomodelo_output_dir: Path
) -> pl.DataFrame:
    """The hours-weighted stage bounds matching :func:`evaluate_lhs_novomodelo`.

    See :func:`novomodelo_bridge.novomodelo.constraint_expr.stage_weighted_bounds`.
    """
    return stage_weighted_bounds(
        resolved.bounds,
        novomodelo_readers.read_novomodelo_block_hours(novomodelo_output_dir),
    )


class ResolvedBound(NamedTuple):
    """A per-(constraint, stage) interval bound reduced to one comparison-ready limit.

    ``value`` is the endpoint appropriate to the row's evaluated direction —
    the number the numeric comparison and the chart plot, identical to what
    a legacy single ``bound`` column held. ``shape`` is the direction label
    (``">="``/``"<="``/``"=="``/``"range"``) from
    :func:`~novomodelo_bridge.core.generic_constraint_format.shape_from_bounds`, for
    display only.
    """

    value: float
    shape: str


def _resolve_bound(lower: float | None, upper: float | None) -> ResolvedBound:
    """Resolve one interval bound row to a single comparison-ready limit + shape.

    ``>=`` (lower-only) reads ``lower``; ``<=`` (upper-only) reads ``upper``;
    ``==`` (both equal) reads either (they agree) — this matches the legacy
    single ``bound`` column value exactly for every constraint family this
    comparator evaluates today (RE/AGRINT/VminOP are all single-sided). A
    genuine two-sided ``range`` has no producer feeding this comparator; the
    upper (ceiling) endpoint is used as the documented fallback so a future
    band constraint degrades to one visualized limit rather than raising.
    """
    shape = shape_from_bounds(lower, upper)
    if shape in ("<=", "range"):
        if upper is None:  # pragma: no cover - guaranteed by shape_from_bounds
            raise AssertionError(f"shape {shape!r} implies a non-null upper bound")
        return ResolvedBound(upper, shape)
    if lower is None:  # pragma: no cover - guaranteed by shape_from_bounds
        raise AssertionError(f"shape {shape!r} implies a non-null lower bound")
    return ResolvedBound(lower, shape)


def per_stage_bounds(
    gc_bounds: pl.DataFrame,
    max_stage: int | None = None,
) -> dict[int, dict[int, ResolvedBound]]:
    """Reduce stage bounds to one :class:`ResolvedBound` per (constraint, stage).

    *gc_bounds* holds one row per (constraint, stage), as
    :func:`stage_bounds_novomodelo` returns it. Returns
    ``{constraint_id: {stage_id: ResolvedBound}}``; a stage with neither
    endpoint is left out, and stages beyond ``max_stage`` are dropped when
    ``max_stage`` is supplied.
    """
    out: dict[int, dict[int, ResolvedBound]] = {}
    if gc_bounds.is_empty():
        return out
    df = gc_bounds
    if max_stage is not None:
        df = df.filter(pl.col("stage_id") <= max_stage)
    df = df.filter(
        pl.col("bound_lower").is_not_null() | pl.col("bound_upper").is_not_null()
    )
    for row in df.sort(["constraint_id", "stage_id"]).iter_rows(named=True):
        per_stage = out.setdefault(int(row["constraint_id"]), {})
        per_stage.setdefault(
            int(row["stage_id"]), _resolve_bound(row["bound_lower"], row["bound_upper"])
        )
    return out
