"""Tier-1 tests for the one shared bound-table merge.

Pure polars — imports no ``novomodelo`` and reads no ``example/`` deck.
"""

from __future__ import annotations

import polars as pl
import pytest

from novomodelo_bridge.core.bound_merge import merge_bound_tables


def test_base_only_rows_survive() -> None:
    base = pl.DataFrame({"hydro_id": [1, 2], "stage_id": [0, 0], "min": [1.0, 2.0]})
    overlay = pl.DataFrame({"hydro_id": [1], "stage_id": [0], "max": [10.0]})

    result = merge_bound_tables(
        base, overlay, on=["hydro_id", "stage_id"], precedence="overlay"
    )

    row = result.filter(pl.col("hydro_id") == 2).row(0, named=True)
    assert row["min"] == 2.0
    assert row["max"] is None


def test_overlay_only_rows_survive() -> None:
    base = pl.DataFrame({"hydro_id": [1], "stage_id": [0], "min": [1.0]})
    overlay = pl.DataFrame(
        {"hydro_id": [1, 2], "stage_id": [0, 0], "max": [10.0, 20.0]}
    )

    result = merge_bound_tables(
        base, overlay, on=["hydro_id", "stage_id"], precedence="overlay"
    )

    row = result.filter(pl.col("hydro_id") == 2).row(0, named=True)
    assert row["min"] is None
    assert row["max"] == 20.0
    assert result.height == 2


def test_overlapping_key_precedence_overlay_wins() -> None:
    base = pl.DataFrame({"hydro_id": [1], "stage_id": [0], "value": [1.0]})
    overlay = pl.DataFrame({"hydro_id": [1], "stage_id": [0], "value": [99.0]})

    result = merge_bound_tables(
        base, overlay, on=["hydro_id", "stage_id"], precedence="overlay"
    )

    assert result.height == 1
    assert result.row(0, named=True)["value"] == 99.0


def test_overlapping_key_precedence_base_wins() -> None:
    base = pl.DataFrame({"hydro_id": [1], "stage_id": [0], "value": [1.0]})
    overlay = pl.DataFrame({"hydro_id": [1], "stage_id": [0], "value": [99.0]})

    result = merge_bound_tables(
        base, overlay, on=["hydro_id", "stage_id"], precedence="base"
    )

    assert result.height == 1
    assert result.row(0, named=True)["value"] == 1.0


def test_outer_union_row_count_on_overlapping_keys() -> None:
    base = pl.DataFrame({"hydro_id": [1, 2], "stage_id": [0, 0], "value": [1.0, 2.0]})
    overlay = pl.DataFrame(
        {"hydro_id": [2, 3], "stage_id": [0, 0], "value": [99.0, 3.0]}
    )

    result = merge_bound_tables(
        base, overlay, on=["hydro_id", "stage_id"], precedence="overlay"
    )

    # base-only {1}, overlapping {2}, overlay-only {3} -> 3 rows, not 4.
    assert result.height == 3


def test_disjoint_columns_both_preserved_with_deterministic_order() -> None:
    base = pl.DataFrame({"hydro_id": [1], "stage_id": [0], "min_flow": [1.0]})
    overlay = pl.DataFrame({"hydro_id": [1], "stage_id": [0], "max_flow": [9.0]})

    result = merge_bound_tables(
        base, overlay, on=["hydro_id", "stage_id"], precedence="overlay"
    )

    assert result.columns == ["hydro_id", "stage_id", "min_flow", "max_flow"]
    row = result.row(0, named=True)
    assert row["min_flow"] == 1.0
    assert row["max_flow"] == 9.0


def test_missing_join_key_raises_value_error() -> None:
    base = pl.DataFrame({"hydro_id": [1], "stage_id": [0], "value": [1.0]})
    overlay = pl.DataFrame({"hydro_id": [1], "stage_id": [0], "value": [2.0]})

    with pytest.raises(ValueError, match="missing_col"):
        merge_bound_tables(base, overlay, on=["missing_col"], precedence="base")


def test_precedence_side_null_falls_back_to_present_side() -> None:
    # The coalesce fallback contract: when a row exists on only one input, a
    # shared non-key column resolves to the present side even if precedence
    # points at the (structurally null) absent side.
    base = pl.DataFrame({"hydro_id": [1, 2], "stage_id": [0, 0], "value": [1.0, 2.0]})
    overlay = pl.DataFrame({"hydro_id": [2], "stage_id": [0], "value": [99.0]})

    result = merge_bound_tables(
        base, overlay, on=["hydro_id", "stage_id"], precedence="overlay"
    )

    by_hydro = {r["hydro_id"]: r["value"] for r in result.iter_rows(named=True)}
    # hydro_id=1 is base-only; precedence='overlay' has no such row, so 'value'
    # falls back to base's 1.0 (not null).
    assert by_hydro[1] == 1.0
    # hydro_id=2 exists on both; precedence='overlay' wins -> 99.0.
    assert by_hydro[2] == 99.0
