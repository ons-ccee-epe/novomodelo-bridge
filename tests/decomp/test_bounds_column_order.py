"""Column order of the DECOMP ``constraints/`` bounds parquets.

``block_id`` is an *indexing* column (which patamar a row applies to), so it
belongs with the other index columns — ``entity_id``, ``stage_id`` — ahead of
the value columns it indexes, matching the ``(constraint_id, stage_id,
block_id)`` order the generic-constraint bounds already use. These tests pin
that layout so it cannot silently drift back to a trailing ``block_id``.

The schema constant governs the written column order (each writer passes it as
``pa.table(..., schema=...)``), so asserting on the schema names asserts on the
parquet layout. The hydro/thermal/pumping *bound*
schemas live on the E2 accumulator (``bounds_accumulator``), not on the
individual converters — ``bounds.py``/``thermal.py``/``network.py`` emit
``BoundContribution`` lists the accumulator fans into these tables, and
``thermal.py`` keeps only its own ``cost_per_mwh`` side-table schema (cost is
not a bound axis, so it never reaches the accumulator).
"""

from __future__ import annotations

from novomodelo_bridge.decomp.bounds_accumulator import (
    HYDRO_BOUNDS_SCHEMA,
    PUMPING_BOUNDS_SCHEMA,
    THERMAL_BOUNDS_SCHEMA,
)
from novomodelo_bridge.decomp.converters.network import _LINE_BOUNDS_SCHEMA
from novomodelo_bridge.decomp.converters.thermal import _THERMAL_COST_SCHEMA
from novomodelo_bridge.decomp.group_bounds import _HYDRO_UNIT_GROUP_BOUNDS_SCHEMA


def test_hydro_bounds_index_columns_precede_values() -> None:
    assert HYDRO_BOUNDS_SCHEMA.names == [
        "hydro_id",
        "stage_id",
        "block_id",
        "min_outflow_m3s",
        "max_outflow_m3s",
        "min_turbined_m3s",
        "max_turbined_m3s",
        "min_generation_mw",
        "max_generation_mw",
        "min_storage_hm3",
        "max_storage_hm3",
        "min_diversion_m3s",
        "max_diversion_m3s",
        "min_spillage_m3s",
        "max_spillage_m3s",
    ]


def test_thermal_bounds_index_columns_precede_values() -> None:
    assert THERMAL_BOUNDS_SCHEMA.names == [
        "thermal_id",
        "stage_id",
        "block_id",
        "min_generation_mw",
        "max_generation_mw",
    ]


def test_thermal_cost_side_table_index_columns_precede_values() -> None:
    """``cost_per_mwh`` rides alongside the accumulator, not through it
    (it is not a registered bound axis) — this pins its own side-table
    schema, which the pipeline rejoins onto the resolved thermal bounds."""
    assert _THERMAL_COST_SCHEMA.names == [
        "thermal_id",
        "stage_id",
        "block_id",
        "cost_per_mwh",
    ]


def test_pumping_bounds_index_columns_precede_values() -> None:
    assert PUMPING_BOUNDS_SCHEMA.names == [
        "pumping_station_id",
        "stage_id",
        "block_id",
        "min_m3s",
        "max_m3s",
    ]


def test_line_bounds_index_columns_precede_values() -> None:
    assert _LINE_BOUNDS_SCHEMA.names == [
        "line_id",
        "stage_id",
        "block_id",
        "direct_mw",
        "reverse_mw",
    ]


def test_group_bounds_index_columns_precede_values() -> None:
    assert _HYDRO_UNIT_GROUP_BOUNDS_SCHEMA.names == [
        "hydro_id",
        "hydro_unit_group_id",
        "stage_id",
        "block_id",
        "min_turbined_m3s",
        "max_turbined_m3s",
        "min_generation_mw",
        "max_generation_mw",
    ]
