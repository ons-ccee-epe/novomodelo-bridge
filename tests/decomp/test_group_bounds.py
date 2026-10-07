"""Unit tests for ``novomodelo_bridge.decomp.group_bounds`` and the
``BoundFamily.group_column`` extension.

``hydro_unit_group_bounds`` is a genuinely new table with no deck fixture to
read it from yet — every test here works against hand-built, in-memory
tables, matching ``test_emission_checks.py``'s own style and
``test_rq_bounds.py``'s synthetic-calendar fixture.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pyarrow as pa
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.emission_checks import (
    BoundFamily,
    check_bound_block_id_range,
    check_bound_row_uniqueness,
)
from novomodelo_bridge.decomp.group_bounds import (
    _HYDRO_UNIT_GROUP_BOUNDS_SCHEMA,
    GroupBoundEntry,
    _empty,
    convert_hydro_unit_group_bounds,
)
from novomodelo_bridge.decomp.temporal import build_operative_calendar
from tests.conftest import make_decomp_case


def _calendar():
    """A valid 3-stage operative calendar, 3 blocks per stage — the same
    fixture shape ``test_decomp_rq_bounds.py`` uses for its own synthetic
    cases."""
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


def _stages(*block_counts: int) -> dict:
    """A minimal ``stages.json``-shaped dict declaring *block_counts[i]*
    blocks for stage *i* — same helper ``test_emission_checks.py`` uses."""
    return {
        "stages": [
            {"id": i, "blocks": [{"id": b} for b in range(count)]}
            for i, count in enumerate(block_counts)
        ]
    }


# ---------------------------------------------------------------------------
# Criterion 1 — schema + one base row per populated (hydro, group, stage)
# ---------------------------------------------------------------------------


class TestSingleGroupBaseRows:
    def test_schema_and_one_base_row_per_populated_stage(self) -> None:
        calendar = _calendar()
        values = {
            (0, 0, 0): GroupBoundEntry(max_generation_mw=100.0),
            (0, 0, 1): GroupBoundEntry(max_generation_mw=110.0),
            (0, 0, 2): GroupBoundEntry(max_generation_mw=120.0),
        }

        case = make_decomp_case(Path("unused"), calendar=calendar)
        table = convert_hydro_unit_group_bounds(case, values=values)

        assert table.schema == _HYDRO_UNIT_GROUP_BOUNDS_SCHEMA
        assert table.num_rows == 3
        assert table["hydro_id"].to_pylist() == [0, 0, 0]
        assert table["hydro_unit_group_id"].to_pylist() == [0, 0, 0]
        assert table["stage_id"].to_pylist() == [0, 1, 2]
        assert table["block_id"].to_pylist() == [None, None, None]
        assert table["max_generation_mw"].to_pylist() == [100.0, 110.0, 120.0]
        assert table["min_turbined_m3s"].to_pylist() == [None, None, None]
        assert table["max_turbined_m3s"].to_pylist() == [None, None, None]
        assert table["min_generation_mw"].to_pylist() == [None, None, None]

    def test_unpopulated_key_emits_no_row(self) -> None:
        """A ``GroupBoundEntry`` with every column ``None`` is sparse: it
        contributes no row, not a row of nulls."""
        calendar = _calendar()
        values = {(0, 0, 0): GroupBoundEntry()}

        case = make_decomp_case(Path("unused"), calendar=calendar)
        table = convert_hydro_unit_group_bounds(case, values=values)

        assert table.num_rows == 0

    def test_empty_mapping_returns_empty_schema_correct_table(self) -> None:
        case = make_decomp_case(Path("unused"), calendar=_calendar())
        table = convert_hydro_unit_group_bounds(case, values={})

        assert table.schema == _HYDRO_UNIT_GROUP_BOUNDS_SCHEMA
        assert table.num_rows == 0
        assert table.equals(_empty())


# ---------------------------------------------------------------------------
# Criterion 2 — base-plus-sparse-per-block
# ---------------------------------------------------------------------------


class TestBasePlusSparsePerBlock:
    def test_nonuniform_stage_gets_base_plus_n_blocks_overrides(self) -> None:
        calendar = _calendar()
        values = {
            (0, 0, 0): GroupBoundEntry(max_generation_mw=[100.0, 100.0, 0.0]),
        }

        case = make_decomp_case(Path("unused"), calendar=calendar)
        table = convert_hydro_unit_group_bounds(case, values=values)

        assert table.num_rows == 4  # 1 base row + 3 per-block override rows
        assert table["block_id"].to_pylist() == [None, 0, 1, 2]
        max_gen = table["max_generation_mw"].to_pylist()
        assert max_gen[1:] == [100.0, 100.0, 0.0]  # including the 0.0 block

        stage0 = calendar[0]
        expected_base = (
            sum(
                v * h
                for v, h in zip([100.0, 100.0, 0.0], stage0.block_hours, strict=True)
            )
            / stage0.total_hours
        )
        assert max_gen[0] == pytest.approx(expected_base)

        # Columns with no per-block breakdown at this key stay None even on
        # the override rows.
        assert table["min_turbined_m3s"].to_pylist() == [None, None, None, None]

    def test_uniform_stage_gets_base_row_only(self) -> None:
        calendar = _calendar()
        values = {
            (0, 0, 0): GroupBoundEntry(max_generation_mw=[50.0, 50.0, 50.0]),
        }

        case = make_decomp_case(Path("unused"), calendar=calendar)
        table = convert_hydro_unit_group_bounds(case, values=values)

        assert table.num_rows == 1
        assert table["block_id"].to_pylist() == [None]
        assert table["max_generation_mw"].to_pylist() == [50.0]

    def test_per_block_length_mismatch_raises(self) -> None:
        calendar = _calendar()
        values = {
            # stage 0 declares 3 blocks; only 2 supplied.
            (0, 0, 0): GroupBoundEntry(max_generation_mw=[100.0, 100.0]),
        }

        case = make_decomp_case(Path("unused"), calendar=calendar)
        with pytest.raises(ValueError, match=r"zip\(\)"):
            convert_hydro_unit_group_bounds(case, values=values)


# ---------------------------------------------------------------------------
# Criterion 3 — two groups of one plant do not collide; a real duplicate
# within one group is still caught
# ---------------------------------------------------------------------------


class TestTwoGroupNonCollision:
    def test_two_groups_same_key_are_not_a_duplicate(self) -> None:
        table = pa.table(
            {
                "hydro_id": pa.array([0, 0], type=pa.int32()),
                "hydro_unit_group_id": pa.array([0, 1], type=pa.int32()),
                "stage_id": pa.array([0, 0], type=pa.int32()),
                "block_id": pa.array([None, None], type=pa.int32()),
                "max_generation_mw": pa.array([100.0, 200.0], type=pa.float64()),
            }
        )
        families = [
            BoundFamily(
                "HydroUnitGroup",
                "hydro_id",
                table,
                group_column="hydro_unit_group_id",
            )
        ]

        with dx.collect() as collected:
            check_bound_row_uniqueness(families)

        assert collected == []

    def test_real_duplicate_within_same_group_is_caught(self) -> None:
        table = pa.table(
            {
                "hydro_id": pa.array([0, 0], type=pa.int32()),
                "hydro_unit_group_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 0], type=pa.int32()),
                "block_id": pa.array([None, None], type=pa.int32()),
                "max_generation_mw": pa.array([100.0, 200.0], type=pa.float64()),
            }
        )
        families = [
            BoundFamily(
                "HydroUnitGroup",
                "hydro_id",
                table,
                group_column="hydro_unit_group_id",
            )
        ]

        with dx.collect() as collected:
            check_bound_row_uniqueness(families)

        errors = [d for d in collected if d.severity is Severity.ERROR]
        assert len(errors) == 1
        assert errors[0].code == "bound-row-duplicate"


# ---------------------------------------------------------------------------
# Criterion 4 — the three existing families are unchanged (group_column=None)
# ---------------------------------------------------------------------------


class TestExistingFamilyNonRegression:
    def test_uniqueness_finding_unchanged_for_group_column_none(self) -> None:
        table = pa.table(
            {
                "hydro_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 0], type=pa.int32()),
                "max_turbined_m3s": pa.array([10.0, 20.0], type=pa.float64()),
            }
        )
        families = [BoundFamily("Hydro", "hydro_id", table)]

        with dx.collect() as collected:
            check_bound_row_uniqueness(families)

        errors = [d for d in collected if d.severity is Severity.ERROR]
        assert len(errors) == 1
        assert errors[0].table is not None
        assert errors[0].table.columns == [
            "Family",
            "Entity ID",
            "Stage",
            "Block",
            "Column",
        ]
        assert errors[0].table.rows[0] == [
            "Hydro",
            0,
            0,
            "stage-wide",
            "max_turbined_m3s",
        ]

    def test_block_id_range_finding_unchanged_for_group_column_none(self) -> None:
        stages = _stages(3, 2)  # stage 0 -> 3 blocks, stage 1 -> 2 blocks
        table = pa.table(
            {
                "line_id": pa.array([1], type=pa.int32()),
                "stage_id": pa.array([1], type=pa.int32()),
                "block_id": pa.array([2], type=pa.int32()),  # stage 1 has 0..1
                "direct_mw": pa.array([10.0], type=pa.float64()),
            }
        )
        families = [BoundFamily("Line", "line_id", table)]

        with dx.collect() as collected:
            check_bound_block_id_range(stages, families)

        errors = [d for d in collected if d.severity is Severity.ERROR]
        assert len(errors) == 1
        assert errors[0].table is not None
        assert errors[0].table.columns == [
            "Family",
            "Entity ID",
            "Stage",
            "block_id",
            "Declared blocks",
        ]
        assert errors[0].table.rows[0] == ["Line", 1, 1, 2, 2]

    def test_thermal_fixture_still_passes_clean(self) -> None:
        stages = _stages(1)
        table = pa.table(
            {
                "thermal_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([0], type=pa.int32()),
                "max_generation_mw": pa.array([10.0], type=pa.float64()),
            }
        )
        families = [BoundFamily("Thermal", "thermal_id", table)]

        with dx.collect() as collected:
            check_bound_row_uniqueness(families)
            check_bound_block_id_range(stages, families)

        assert collected == []


# ---------------------------------------------------------------------------
# Criterion 5 — block-id range with group_column set names the offending
# (hydro, group, stage, block_id) row
# ---------------------------------------------------------------------------


class TestGroupBlockIdRange:
    def test_out_of_range_block_id_names_hydro_group_stage_block(self) -> None:
        stages = _stages(3, 2)  # stage 0 -> 3 blocks, stage 1 -> 2 blocks
        table = pa.table(
            {
                "hydro_id": pa.array([5], type=pa.int32()),
                "hydro_unit_group_id": pa.array([1], type=pa.int32()),
                "stage_id": pa.array([1], type=pa.int32()),
                "block_id": pa.array([2], type=pa.int32()),  # stage 1 has 0..1
                "max_generation_mw": pa.array([10.0], type=pa.float64()),
            }
        )
        family = BoundFamily(
            "HydroUnitGroup",
            "hydro_id",
            table,
            group_column="hydro_unit_group_id",
        )

        with dx.collect() as collected:
            check_bound_block_id_range(stages, [family])

        errors = [d for d in collected if d.severity is Severity.ERROR]
        assert len(errors) == 1
        diag = errors[0]
        assert diag.code == "bound-block-id-out-of-range"
        assert diag.table is not None
        assert diag.table.columns == [
            "Family",
            "Entity ID",
            "Group ID",
            "Stage",
            "block_id",
            "Declared blocks",
        ]
        assert diag.table.rows[0] == ["HydroUnitGroup", 5, 1, 1, 2, 2]

    def test_in_range_block_id_with_group_column_passes(self) -> None:
        stages = _stages(3, 2)
        table = pa.table(
            {
                "hydro_id": pa.array([5, 5]),
                "hydro_unit_group_id": pa.array([0, 1]),
                "stage_id": pa.array([0, 1]),
                "block_id": pa.array([2, 1]),
                "max_generation_mw": pa.array([10.0, 20.0]),
            },
            schema=pa.schema(
                [
                    ("hydro_id", pa.int32()),
                    ("hydro_unit_group_id", pa.int32()),
                    ("stage_id", pa.int32()),
                    ("block_id", pa.int32()),
                    ("max_generation_mw", pa.float64()),
                ]
            ),
        )
        family = BoundFamily(
            "HydroUnitGroup",
            "hydro_id",
            table,
            group_column="hydro_unit_group_id",
        )

        with dx.collect() as collected:
            check_bound_block_id_range(stages, [family])

        assert collected == []
