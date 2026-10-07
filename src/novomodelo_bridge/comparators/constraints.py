"""Shared cobre-side generic-constraint loaders, LHS evaluator, and bound resolution."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import polars as pl

from cobre_bridge.cobre.constraint_expr import evaluate_constraint_expressions
from cobre_bridge.cobre.readers import scan_simulation_entity
from cobre_bridge.core.generic_constraint_format import shape_from_bounds

_LOG = logging.getLogger(__name__)


def load_generic_constraints(cobre_input_dir: Path) -> list[dict]:
    """Load constraint definitions from ``constraints/generic_constraints.json``.

    The F3 objects are sense-free (no ``sense`` key); direction is derived
    from the companion bounds table's endpoints when a label is needed (see
    :func:`per_stage_bounds` / :func:`cobre_bridge.core.generic_constraint_format.
    shape_from_bounds`). Returns an empty list when the file is missing or
    malformed.
    """
    path = cobre_input_dir / "constraints" / "generic_constraints.json"
    if not path.exists():
        return []
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("generic_constraints.json could not be parsed: %s", exc)
        return []
    return list(data.get("constraints", []))


def load_generic_constraint_bounds(cobre_input_dir: Path) -> pl.DataFrame:
    """Load bound table from ``constraints/generic_constraint_bounds.parquet``.

    F3 shape: nullable ``bound_lower``/``bound_upper`` endpoints, no single
    ``bound`` column — even in the missing-file fallback schema below.
    """
    path = cobre_input_dir / "constraints" / "generic_constraint_bounds.parquet"
    if not path.exists():
        return pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "block_id": pl.Int32,
                "bound_lower": pl.Float64,
                "bound_upper": pl.Float64,
            }
        )
    return pl.read_parquet(path)


def evaluate_lhs_cobre(
    constraints: list[dict],
    cobre_output_dir: Path,
    rho_acum_overrides: dict[int, dict[int, float]] | None = None,
) -> pl.DataFrame:
    """Evaluate each constraint's LHS from Cobre simulation outputs.

    Uses the shared
    :func:`cobre_bridge.cobre.constraint_expr.evaluate_constraint_expressions`
    (which returns one row per (constraint, scenario, stage, block)) and
    collapses to mean across scenarios and blocks per (constraint, stage).

    ``rho_acum_overrides`` (typically :func:`cobre_bridge.cobre.constraint_expr.
    load_rho_acum_overrides` against the converted case dir) is forwarded
    verbatim so a ``@rho_acum_h{id}``-scaled constraint (VminOP, RHE)
    resolves against the LP's actual per-stage coefficient rather than the
    simulation's default point-productivity column — see
    :func:`cobre_bridge.cobre.constraint_expr.evaluate_constraint_expressions`.

    Returns
    -------
    polars.DataFrame
        Columns: ``constraint_id`` (Int32), ``stage_id`` (Int32),
        ``lhs_value`` (Float64).  Empty when no simulation data is
        available or no constraints reference the simulation entities.
    """
    if not constraints:
        return pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "lhs_value": pl.Float64,
            }
        )

    # Scan with the comparator's own simulation reader (which already takes the
    # ``output/`` directory directly), instead of reaching into the dashboard's
    # case-dir-based scanner via a synthetic ``cobre_output_dir.parent``. A
    # present-but-corrupt parquet raises CobreReadError.
    # NB: ``lf or pl.LazyFrame()`` would evaluate ``bool(lf)``, which polars
    # rejects ("truth value of a LazyFrame is ambiguous") — use explicit None
    # checks.
    empty = pl.DataFrame(
        schema={
            "constraint_id": pl.Int32,
            "stage_id": pl.Int32,
            "lhs_value": pl.Float64,
        }
    )

    hydros_lf = scan_simulation_entity(cobre_output_dir, "hydros")
    if hydros_lf is None:
        # No hydro simulation → no operation data to evaluate the LHS against.
        return empty
    exchanges_lf = scan_simulation_entity(cobre_output_dir, "exchanges")
    if exchanges_lf is None:
        exchanges_lf = pl.LazyFrame()

    lhs_pd: pd.DataFrame = evaluate_constraint_expressions(
        constraints, hydros_lf, exchanges_lf, rho_acum_overrides
    )
    if lhs_pd.empty:
        return pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "lhs_value": pl.Float64,
            }
        )

    # Collapse to mean across scenarios and blocks per (constraint, stage).
    grouped = lhs_pd.groupby(["constraint_id", "stage_id"], as_index=False)
    means = pd.DataFrame(grouped["lhs_value"].mean())
    return pl.from_pandas(means).with_columns(
        pl.col("constraint_id").cast(pl.Int32),
        pl.col("stage_id").cast(pl.Int32),
        pl.col("lhs_value").cast(pl.Float64),
    )


class ResolvedBound(NamedTuple):
    """A per-(constraint, stage) F3 bound reduced to one comparison-ready limit.

    ``value`` is the endpoint appropriate to the row's evaluated direction —
    the number the numeric comparison and the chart plot, identical to what
    the pre-F3 single ``bound`` column held. ``shape`` is the direction label
    (``">="``/``"<="``/``"=="``/``"range"``) from
    :func:`~cobre_bridge.core.generic_constraint_format.shape_from_bounds`, for
    display only.
    """

    value: float
    shape: str


def _resolve_bound(lower: float | None, upper: float | None) -> ResolvedBound:
    """Resolve one F3 bound row to a single comparison-ready limit + shape.

    ``>=`` (lower-only) reads ``lower``; ``<=`` (upper-only) reads ``upper``;
    ``==`` (both equal) reads either (they agree) — this matches the pre-F3
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
    """Reduce per-(stage, block) bounds to one value per (constraint, stage).

    When per-block bounds disagree (rare; AGRINT/RE bounds are usually
    block-invariant), the block_id=0 value is preferred.  Returns
    ``{constraint_id: {stage_id: ResolvedBound}}``; stages beyond
    ``max_stage`` are dropped when ``max_stage`` is supplied.
    """
    out: dict[int, dict[int, ResolvedBound]] = {}
    if gc_bounds.is_empty():
        return out
    df = gc_bounds
    if max_stage is not None:
        df = df.filter(pl.col("stage_id") <= max_stage)
    for cid, group in df.group_by("constraint_id"):
        cid_int = int(cid[0])
        per_stage: dict[int, ResolvedBound] = {}
        if "block_id" in group.columns:
            preferred = group.filter(pl.col("block_id") == 0)
            if preferred.is_empty():
                preferred = group
        else:
            preferred = group
        for row in preferred.iter_rows(named=True):
            stage = int(row["stage_id"])
            if stage not in per_stage:
                per_stage[stage] = _resolve_bound(
                    row["bound_lower"], row["bound_upper"]
                )
        out[cid_int] = per_stage
    return out
