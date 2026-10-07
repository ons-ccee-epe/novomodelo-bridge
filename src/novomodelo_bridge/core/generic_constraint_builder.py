"""Shared generic-constraint builder + id allocator for both conversion tracks.

The NEWAVE track (``newave/converters/constraints.py``) and the DECOMP track
(``decomp/converters/constraints.py``) each assemble cobre F3 generic constraints
(``generic_constraint_format.py``'s sense-free interval shape). This module is
the single model-agnostic home both tracks use for that stateful assembly —
:class:`ConstraintIdAllocator`, :class:`GenericConstraintBuilder`,
:class:`GenericConstraintResult`, :data:`GENERIC_BOUNDS_SCHEMA`, and the F3
slot/sentinel helpers (:data:`UNBOUNDED`, :func:`is_bounded`,
:func:`slot_endpoints`) — built on the pure mapping leaf
:mod:`cobre_bridge.core.generic_constraint_format`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import pyarrow as pa

from cobre_bridge.core.generic_constraint_format import (
    GENERIC_BOUNDS_COLUMNS,
    sense_to_interval,
)

#: One `(stage_id, block_id, raw_lower, raw_upper)` slot a caller offers to
#: :meth:`GenericConstraintBuilder.add`; `block_id` is `None` for a
#: stage-level (not per-block) slot.
Slot = tuple[int, int | None, float | None, float | None]


class ConstraintIdAllocator:
    """Hands out contiguous, non-overlapping generic-constraint ID ranges.

    VminOP, electric, and AGRINT constraints share one 0-based ID space. Each
    converter is given the next free start ID; after it returns, ``advance(n)``
    reserves the ``n`` IDs it produced so the next converter starts past them.
    This replaces the by-hand offset arithmetic
    (``start_id = vminop_count + electric_count``) that silently corrupts IDs if
    a term is missed.
    """

    def __init__(self, next_id: int = 0) -> None:
        self.next_id = next_id

    def advance(self, count: int) -> None:
        self.next_id += count


class GenericConstraintResult(NamedTuple):
    """Result of a generic-constraint emitter.

    ``constraints`` is the list of constraint dicts (``{"id", "name",
    "description", "expression", "slack"}`` — cobre's F3 sense-free shape,
    see :mod:`cobre_bridge.core.generic_constraint_format`); ``bounds`` is the
    per-``(constraint_id, stage_id, block_id)`` bounds table honouring
    :data:`GENERIC_BOUNDS_SCHEMA` (nullable ``bound_lower``/``bound_upper``
    endpoints, no single ``bound``).
    """

    constraints: list[dict]
    bounds: pa.Table


#: Schema for the generic-constraint bounds table (F3 shape: see
#: :data:`~cobre_bridge.core.generic_constraint_format.GENERIC_BOUNDS_COLUMNS`).
#: ``block_id`` is nullable: ``None`` means "all blocks" (a stage-level
#: constraint, or a per-block one whose bound applies uniformly).
#: ``bound_lower``/``bound_upper`` are both nullable; the null-pattern
#: encodes direction (lower-only, upper-only, or a genuine band with both
#: populated) instead of a ``sense`` label.
GENERIC_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("constraint_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("bound_lower", pa.float64(), nullable=True),
        pa.field("bound_upper", pa.float64(), nullable=True),
    ]
)
# Keep this schema's field names in lockstep with the shared F3 column list —
# a silent drift here would desync this module from the shared mapping helper.
# A plain `assert` is stripped under `python -O`, which would silently drop
# this load-bearing drift guard, so it is an explicit raise instead.
if GENERIC_BOUNDS_SCHEMA.names != list(GENERIC_BOUNDS_COLUMNS):
    raise RuntimeError(
        "generic_constraint_builder.py: GENERIC_BOUNDS_SCHEMA field names "
        f"{GENERIC_BOUNDS_SCHEMA.names!r} have drifted from the shared "
        f"GENERIC_BOUNDS_COLUMNS {list(GENERIC_BOUNDS_COLUMNS)!r} — update "
        "one to match the other."
    )

#: The unbounded sentinel: a bound whose magnitude is at or past this value
#: carries no real limit. Single source across both conversion tracks —
#: ``decomp/bounds_accumulator.py`` imports :func:`is_bounded` (built on this
#: constant) rather than redeclaring the sentinel.
UNBOUNDED = 1e21


def is_bounded(value: float | None) -> bool:
    """True iff *value* is a real (non-``None``, non-sentinel) bound."""
    return value is not None and abs(value) < UNBOUNDED


def slot_endpoints(
    lower: float | None, upper: float | None
) -> tuple[float | None, float | None]:
    """Resolve one (stage, block) slot's raw (lower, upper) to F3 endpoints.

    Each bounded side is independent — this is not a single ``sense``, it is
    the union of an optional ``">="`` floor and an optional ``"<="``
    ceiling — so a genuinely two-sided slot keeps both endpoints. Each
    bounded side is fed through :func:`sense_to_interval` for its own
    direction, keeping only the half of that call's result the side
    actually populates; an unbounded (``None``/``±1e21``) side stays
    ``None``.
    """
    bound_lower: float | None = None
    bound_upper: float | None = None
    if is_bounded(lower):
        bound_lower = sense_to_interval(">=", lower)[0]
    if is_bounded(upper):
        bound_upper = sense_to_interval("<=", upper)[1]
    return bound_lower, bound_upper


class GenericConstraintBuilder:
    """Assembles generic constraints sharing one 0-based id space via *allocator*.

    Every id comes from *allocator* (advanced as each constraint is added), so
    no caller does manual id arithmetic. :meth:`add` is the high-level
    convenience over :meth:`add_constraint` + :meth:`add_bound_row`.
    """

    def __init__(self, allocator: ConstraintIdAllocator) -> None:
        self._allocator = allocator
        self._constraints: list[dict] = []
        self._bound_cids: list[int] = []
        self._bound_stages: list[int] = []
        self._bound_blocks: list[int | None] = []
        self._bound_lowers: list[float | None] = []
        self._bound_uppers: list[float | None] = []

    def add_constraint(
        self,
        *,
        name: str,
        description: str,
        expression: str,
        slack: Mapping[str, object],
    ) -> int:
        """Append one constraint dict and return its allocated id."""
        constraint_id = self._allocator.next_id
        self._constraints.append(
            {
                "id": constraint_id,
                "name": name,
                "description": description,
                "expression": expression,
                "slack": slack,
            }
        )
        self._allocator.advance(1)
        return constraint_id

    def add_bound_row(
        self,
        constraint_id: int,
        *,
        stage_id: int,
        block_id: int | None,
        lower: float | None,
        upper: float | None,
    ) -> None:
        """Append one bound row for *constraint_id* at ``(stage_id, block_id)``."""
        bound_lower, bound_upper = slot_endpoints(lower, upper)
        self._bound_cids.append(constraint_id)
        self._bound_stages.append(stage_id)
        self._bound_blocks.append(block_id)
        self._bound_lowers.append(bound_lower)
        self._bound_uppers.append(bound_upper)

    def add(
        self,
        *,
        name: str,
        description: str,
        expression: str,
        slack: Mapping[str, object],
        slots: Sequence[Slot],
    ) -> int | None:
        """Add one two-sided generic constraint from *slots*, or nothing.

        Filters *slots* to those bounded on either side; adds nothing (not
        even an id) when none are bounded — mirroring the "no orphan
        constraint" gate a cobre constraint with no companion bound row would
        violate. Otherwise emits one constraint plus one bound row per
        bounded slot, in order, and returns the constraint's id.
        """
        bounded_slots = [
            slot for slot in slots if is_bounded(slot[2]) or is_bounded(slot[3])
        ]
        if not bounded_slots:
            return None

        constraint_id = self.add_constraint(
            name=name, description=description, expression=expression, slack=slack
        )
        for stage_id, block_id, lower, upper in bounded_slots:
            self.add_bound_row(
                constraint_id,
                stage_id=stage_id,
                block_id=block_id,
                lower=lower,
                upper=upper,
            )
        return constraint_id

    def result(self) -> GenericConstraintResult | None:
        """Package everything added so far, or ``None`` if nothing was added."""
        if not self._constraints:
            return None
        bounds = pa.table(
            {
                "constraint_id": pa.array(self._bound_cids, type=pa.int32()),
                "stage_id": pa.array(self._bound_stages, type=pa.int32()),
                "block_id": pa.array(self._bound_blocks, type=pa.int32()),
                "bound_lower": pa.array(self._bound_lowers, type=pa.float64()),
                "bound_upper": pa.array(self._bound_uppers, type=pa.float64()),
            },
            schema=GENERIC_BOUNDS_SCHEMA,
        )
        return GenericConstraintResult(
            constraints=list(self._constraints), bounds=bounds
        )
