"""The one bound-table merge: full-outer join plus explicit precedence.

Both conversion tracks assemble a bound table by folding an overlay's extra
columns/rows onto a base table, keyed on some subset of ``(entity_id,
stage_id, block_id)``. This is the single shared implementation for that
merge, replacing three call sites' independent ad-hoc ``coalesce``/null
treatments with one explicit precedence rule (``.claude/rules/bridge.md``
§1).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import polars as pl


def merge_bound_tables(
    base: pl.DataFrame,
    overlay: pl.DataFrame,
    *,
    on: Sequence[str],
    precedence: Literal["base", "overlay"],
) -> pl.DataFrame:
    """Full-outer-merge *base* and *overlay* on *on* into one bound table.

    Rows present on either side survive. A non-key column present in both
    frames resolves to *precedence*'s value, falling back to the other
    side only where *precedence*'s side is null for that row (including
    the structural null a row present on only one side leaves on the
    other) — the one rule this helper enforces in place of a per-caller
    ``coalesce``. Column order is deterministic: *on*, then *base*'s
    remaining columns, then *overlay*'s columns not already covered.

    Raises
    ------
    ValueError
        When a name in *on* is absent from *base* or from *overlay*.
    """
    missing = sorted(
        {name for name in on if name not in base.columns or name not in overlay.columns}
    )
    if missing:
        raise ValueError(
            f"join key(s) {missing} not present in both base and overlay frames"
        )

    key = list(on)
    merged = base.join(overlay, on=key, how="full", coalesce=True)

    shared = [c for c in base.columns if c in overlay.columns and c not in key]
    for column in shared:
        overlay_column = f"{column}_right"
        winner, loser = (
            (column, overlay_column)
            if precedence == "base"
            else (overlay_column, column)
        )
        merged = merged.with_columns(pl.coalesce([winner, loser]).alias(column)).drop(
            overlay_column
        )

    column_order = [
        *key,
        *[c for c in base.columns if c not in key],
        *[c for c in overlay.columns if c not in key and c not in shared],
    ]
    return merged.select(column_order)
