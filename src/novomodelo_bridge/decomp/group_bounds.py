"""``constraints/hydro_unit_group_bounds.parquet`` — per-group bound overlay.

Every hydro declares a mandatory, stage-invariant ``unit_groups[]`` array
(``build_mirror_unit_group``, ``core/hydro_units.py``). cobre 0.13 adds a
stage-varying (optionally per-block) overlay for it — this table — letting
any of a group's four declared bounds be *lowered* per stage/block.

Verified against cobre's ``hydro_unit_group_bounds`` schema:

======================= ========== ============
Column                  Arrow type Required?
======================= ========== ============
``hydro_id``            ``int32``  required
``hydro_unit_group_id`` ``int32``  required
``stage_id``            ``int32``  required
``min_turbined_m3s``    ``float64`` optional
``max_turbined_m3s``    ``float64`` optional
``min_generation_mw``   ``float64`` optional
``max_generation_mw``   ``float64`` optional
``block_id``            ``int32``  optional / nullable
======================= ========== ============

All four bound columns are block-eligible (see the reader module docstring) —
unlike the plant-axis ``hydro_bounds``, this family has no block-ineligible
column (contrast ``thermal_bounds.cost_per_mwh``), so cobre rule 37 never
touches it.

**The overlay is id-addressed, not position-addressed.** cobre matches an
override row to a group by the group's own declared ``id`` (cobre builds
``group_position[(hydro.id, group.id)]`` and looks each row up by
``(row.hydro_id, row.hydro_unit_group_id)``), and it sorts both
``unit_groups[]`` and the bound rows by their full key on load
(``entities/hydro.rs::sort_unit_groups``). Producer emission
order is therefore never load-bearing; the only producer obligation is that
``hydro_unit_group_id`` names the target group's declared ``id`` and that
group ids are unique within a plant (cobre rule 39).

This module is **value-fed, not deck-deriving** — it computes nothing about
availability, unit maintenance, or Itaipu's split. :func:`_empty` is what
the DECOMP pipeline calls until real per-group values are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cobre_bridge.decomp.case import DecompCase
    from cobre_bridge.decomp.temporal import OperativeStage

_HYDRO_UNIT_GROUP_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("hydro_unit_group_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("min_turbined_m3s", pa.float64(), nullable=True),
        pa.field("max_turbined_m3s", pa.float64(), nullable=True),
        pa.field("min_generation_mw", pa.float64(), nullable=True),
        pa.field("max_generation_mw", pa.float64(), nullable=True),
    ]
)

#: The four bound columns, in schema order — every one is block-eligible.
_BOUND_COLUMNS: tuple[str, ...] = (
    "min_turbined_m3s",
    "max_turbined_m3s",
    "min_generation_mw",
    "max_generation_mw",
)


def _empty() -> pa.Table:
    return pa.table(
        {
            "hydro_id": pa.array([], type=pa.int32()),
            "hydro_unit_group_id": pa.array([], type=pa.int32()),
            "stage_id": pa.array([], type=pa.int32()),
            "block_id": pa.array([], type=pa.int32()),
            "min_turbined_m3s": pa.array([], type=pa.float64()),
            "max_turbined_m3s": pa.array([], type=pa.float64()),
            "min_generation_mw": pa.array([], type=pa.float64()),
            "max_generation_mw": pa.array([], type=pa.float64()),
        },
        schema=_HYDRO_UNIT_GROUP_BOUNDS_SCHEMA,
    )


@dataclass(frozen=True)
class GroupBoundEntry:
    """One ``(hydro_id, hydro_unit_group_id, stage_id)``'s bound overrides.

    Each of the four schema columns is independently optional (``None`` ⇒
    no override for that column at this key — the row stays sparse, not
    zero). A populated column is either:

    - a bare ``float`` — a stage-level value, emitted on the base row only
      (no per-block override rows for this column), or
    - a ``Sequence[float]``, one entry per the stage's declared block count
      — folded into the base row as the hours-weighted stage mean (the same
      fold ``convert_thermal_bounds``/``convert_hydro_bounds`` use), plus
      sparse per-block override rows wherever the stage's per-block values
      are not all equal.
    """

    min_turbined_m3s: float | Sequence[float] | None = None
    max_turbined_m3s: float | Sequence[float] | None = None
    min_generation_mw: float | Sequence[float] | None = None
    max_generation_mw: float | Sequence[float] | None = None


def _hours_weighted(values: Sequence[float], stage: OperativeStage) -> float:
    """Hours-weighted stage mean of a per-block value list — the same fold
    ``convert_thermal_bounds``/``convert_hydro_bounds`` use for their own
    base row (``decomp/converters/thermal.py``, ``decomp/converters/bounds.py``).

    Raises
    ------
    ValueError
        Via ``zip(..., strict=True)``, if *values* has more or fewer
        entries than the stage's declared block count — the same
        length-mismatch contract the sibling emitters enforce.
    """
    return (
        sum(v * h for v, h in zip(values, stage.block_hours, strict=True))
        / stage.total_hours
    )


def convert_hydro_unit_group_bounds(
    case: DecompCase,
    *,
    values: Mapping[tuple[int, int, int], GroupBoundEntry],
) -> pa.Table:
    """Build ``hydro_unit_group_bounds`` from a hand-fed value mapping.

    *values* keys the four-column overlay by ``(hydro_id,
    hydro_unit_group_id, stage_id)``; *calendar* supplies each stage's
    declared block count for the block-sparsity fold. This function computes
    no availability, maintenance, or Itaipu-specific values itself — 026 (or
    a test) is responsible for what goes into *values*. An empty mapping
    returns :func:`_empty`, which is what the pipeline calls until 026 is
    wired.

    Emission follows the base-plus-sparse-per-block convention verbatim
    (``convert_thermal_bounds``, ``convert_hydro_bounds``, ``convert_lines``):
    every populated ``(hydro, group, stage)`` gets one base row (``block_id
    = None``); if any of its columns supplied a per-block breakdown and that
    breakdown is not block-uniform, one additional override row per declared
    block follows, carrying each per-block-supplied column's exact value at
    that block (a ``0.0`` value is a real bound and is emitted; a column
    with no per-block breakdown stays ``None`` on the override rows). Row
    order (base before its overrides, keys sorted) is a source-code
    convention for readability only — cobre re-sorts on load, so it is never
    load-bearing (see the module docstring's id-addressed note).

    Raises
    ------
    ValueError
        If an entry's *stage_id* is not in *calendar*, or a per-block
        breakdown's length disagrees with that stage's declared block count
        (via :func:`_hours_weighted`'s ``zip(..., strict=True)``).
    """
    if not values:
        return _empty()

    stages_by_id = {stage.index: stage for stage in case.calendar}

    hydro_ids: list[int] = []
    group_ids: list[int] = []
    stage_ids: list[int] = []
    columns: dict[str, list[float | None]] = {c: [] for c in _BOUND_COLUMNS}
    block_ids: list[int | None] = []

    for (hydro_id, group_id, stage_id), entry in sorted(values.items()):
        stage = stages_by_id.get(stage_id)
        if stage is None:
            raise ValueError(
                f"hydro {hydro_id} group {group_id}: stage {stage_id} is not "
                "in the calendar"
            )

        base: dict[str, float | None] = {}
        block_breakdowns: dict[str, list[float]] = {}
        for column in _BOUND_COLUMNS:
            raw = getattr(entry, column)
            if raw is None:
                base[column] = None
            elif isinstance(raw, int | float):
                base[column] = float(raw)
            else:
                block_values = [float(v) for v in raw]
                base[column] = _hours_weighted(block_values, stage)
                block_breakdowns[column] = block_values

        if all(value is None for value in base.values()):
            continue

        hydro_ids.append(hydro_id)
        group_ids.append(group_id)
        stage_ids.append(stage_id)
        for column in _BOUND_COLUMNS:
            columns[column].append(base[column])
        block_ids.append(None)

        if not block_breakdowns:
            continue
        uniform = all(
            all(v == breakdown[0] for v in breakdown)
            for breakdown in block_breakdowns.values()
        )
        if uniform:
            continue

        n_blocks = len(stage.block_hours)
        for b in range(n_blocks):
            hydro_ids.append(hydro_id)
            group_ids.append(group_id)
            stage_ids.append(stage_id)
            for column in _BOUND_COLUMNS:
                breakdown = block_breakdowns.get(column)
                columns[column].append(breakdown[b] if breakdown is not None else None)
            block_ids.append(b)

    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "hydro_unit_group_id": pa.array(group_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "block_id": pa.array(block_ids, type=pa.int32()),
            "min_turbined_m3s": pa.array(
                columns["min_turbined_m3s"], type=pa.float64()
            ),
            "max_turbined_m3s": pa.array(
                columns["max_turbined_m3s"], type=pa.float64()
            ),
            "min_generation_mw": pa.array(
                columns["min_generation_mw"], type=pa.float64()
            ),
            "max_generation_mw": pa.array(
                columns["max_generation_mw"], type=pa.float64()
            ),
        },
        schema=_HYDRO_UNIT_GROUP_BOUNDS_SCHEMA,
    )
