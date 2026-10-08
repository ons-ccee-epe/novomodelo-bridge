"""Unit tests for novomodelo_bridge.core.emission_checks.

Each rule gets a positive test (synthetic violation caught) and a negative
test (the legal shape passes), run against hand-built artifacts — this module
never imports either pipeline, matching emission_checks.py's own contract.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.emission_checks import (
    BoundFamily,
    EmissionCheckError,
    check_block_id_not_on_anticipated_thermal,
    check_bound_block_id_range,
    check_bound_row_uniqueness,
    check_group_bound_envelope,
    check_hydro_bounds_no_raising,
    check_unit_group_envelope,
    clamp_hydro_bounds_to_declared,
    run_and_gate,
)


def _hydros(*entries: dict) -> dict:
    return {"hydros": list(entries)}


def _hydro(
    hydro_id: int,
    *,
    max_turbined_m3s: float,
    max_generation_mw: float,
    groups: list[tuple[float, float]] | None = None,
) -> dict:
    """One ``hydros.json`` entry with a declared envelope and unit groups.

    ``groups`` is a list of ``(max_turbined_m3s, max_generation_mw)`` pairs;
    defaults to a single mirror group matching the plant's own declaration
    (holds rule 41 by construction, as ``build_mirror_unit_group`` does).
    """
    if groups is None:
        groups = [(max_turbined_m3s, max_generation_mw)]
    return {
        "id": hydro_id,
        "name": f"Hydro {hydro_id}",
        "generation": {
            "max_turbined_m3s": max_turbined_m3s,
            "max_generation_mw": max_generation_mw,
        },
        "unit_groups": [
            {
                "id": i,
                "max_turbined_m3s": turbined,
                "max_generation_mw": generation,
            }
            for i, (turbined, generation) in enumerate(groups)
        ],
    }


def _thermals(*entries: dict) -> dict:
    return {"thermals": list(entries)}


def _thermal(thermal_id: int, *, anticipated: bool = False) -> dict:
    """One ``thermals.json`` entry, optionally declaring ``anticipated_config``."""
    return {
        "id": thermal_id,
        "name": f"Thermal {thermal_id}",
        "anticipated_config": {"lead_stages": 1} if anticipated else None,
    }


def _stages(*block_counts: int) -> dict:
    return {
        "stages": [
            {"id": i, "blocks": [{"id": b} for b in range(count)]}
            for i, count in enumerate(block_counts)
        ]
    }


# ---------------------------------------------------------------------------
# Rule 43 — check_hydro_bounds_no_raising
# ---------------------------------------------------------------------------


class TestHydroBoundsNoRaising:
    def test_hydro_bounds_none_is_not_applicable(self) -> None:
        hydros = _hydros(_hydro(0, max_turbined_m3s=100.0, max_generation_mw=50.0))

        with dx.collect() as collected:
            check_hydro_bounds_no_raising(hydros, None)

        assert len(collected) == 1
        assert collected[0].code == "hydro-bounds-raising-not-applicable"
        assert collected[0].severity is Severity.INFO

    def test_decomp_shaped_table_without_guarded_columns_is_not_applicable(
        self,
    ) -> None:
        """DECOMP's hydro_bounds carries only min_outflow_m3s."""
        hydros = _hydros(_hydro(0, max_turbined_m3s=100.0, max_generation_mw=50.0))
        hydro_bounds = pa.table(
            {
                "hydro_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([0], type=pa.int32()),
                "min_outflow_m3s": pa.array([5.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            check_hydro_bounds_no_raising(hydros, hydro_bounds)

        assert len(collected) == 1
        assert collected[0].code == "hydro-bounds-raising-not-applicable"
        assert collected[0].severity is Severity.INFO
        assert not any(d.severity is Severity.ERROR for d in collected)

    def test_row_raising_declared_turbined_cap_is_caught(self) -> None:
        hydros = _hydros(_hydro(0, max_turbined_m3s=100.0, max_generation_mw=50.0))
        hydro_bounds = pa.table(
            {
                "hydro_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([3], type=pa.int32()),
                "max_turbined_m3s": pa.array([150.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            check_hydro_bounds_no_raising(hydros, hydro_bounds)

        errors = [d for d in collected if d.severity is Severity.ERROR]
        assert len(errors) == 1
        diag = errors[0]
        assert diag.code == "hydro-bounds-raises-declared-capacity"
        assert diag.table is not None
        row = diag.table.rows[0]
        # entity, column, stage, declared, offending — all named.
        assert row[0] == 0
        assert row[1] == "max_turbined_m3s"
        assert "3" in row[2]
        assert row[3] == 100.0
        assert row[4] == 150.0

    def test_row_at_or_below_declared_cap_passes(self) -> None:
        hydros = _hydros(_hydro(0, max_turbined_m3s=100.0, max_generation_mw=50.0))
        hydro_bounds = pa.table(
            {
                "hydro_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "max_turbined_m3s": pa.array([100.0, 80.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            check_hydro_bounds_no_raising(hydros, hydro_bounds)

        assert collected == []

    def test_generation_column_checked_independently_of_turbined(self) -> None:
        hydros = _hydros(_hydro(0, max_turbined_m3s=100.0, max_generation_mw=50.0))
        hydro_bounds = pa.table(
            {
                "hydro_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([0], type=pa.int32()),
                "max_turbined_m3s": pa.array([90.0], type=pa.float64()),
                "max_generation_mw": pa.array([75.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            check_hydro_bounds_no_raising(hydros, hydro_bounds)

        errors = [d for d in collected if d.severity is Severity.ERROR]
        assert len(errors) == 1
        assert errors[0].table is not None
        assert [row[1] for row in errors[0].table.rows] == ["max_generation_mw"]


# ---------------------------------------------------------------------------
# Rule 43 repair — clamp_hydro_bounds_to_declared
# ---------------------------------------------------------------------------


def _hydro_with_envelope(
    hydro_id: int,
    *,
    turbined: tuple[float, float] | None = None,
    generation: tuple[float, float] | None = None,
    outflow: tuple[float, float | None] | None = None,
) -> dict:
    """A ``hydros.json`` entry declaring ``(min, max)`` for any of the clampable
    variables — ``generation.min/max_turbined_m3s``, ``generation.min/max_
    generation_mw``, ``outflow.min/max_outflow_m3s`` (max may be ``None``)."""
    entry: dict = {"id": hydro_id, "name": f"Hydro {hydro_id}"}
    gen: dict[str, float] = {}
    if turbined is not None:
        gen["min_turbined_m3s"], gen["max_turbined_m3s"] = turbined
    if generation is not None:
        gen["min_generation_mw"], gen["max_generation_mw"] = generation
    if gen:
        entry["generation"] = gen
    if outflow is not None:
        entry["outflow"] = {
            "min_outflow_m3s": outflow[0],
            "max_outflow_m3s": outflow[1],
        }
    return entry


class TestClampHydroBoundsToDeclared:
    """``clamp_hydro_bounds_to_declared`` ceils/floors each per-stage MAX bound
    into the plant's declared envelope and WARNs about what it clamped, so
    novomodelo rule 43 (and its outflow analogue) holds without raising the
    declaration to fit a per-stage override."""

    def test_turbined_above_declared_max_is_clamped_and_warned(self) -> None:
        hydros = _hydros(_hydro_with_envelope(0, turbined=(0.0, 100.0)))
        hydro_bounds = pa.table(
            {
                "hydro_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 1], type=pa.int32()),
                # stage 0 raises above declared (TURBMAXT-style), stage 1 is fine.
                "max_turbined_m3s": pa.array([150.0, 80.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            result = clamp_hydro_bounds_to_declared(hydros, hydro_bounds)

        # The raised value is clamped to the declaration; the in-envelope one is
        # untouched.
        assert result["max_turbined_m3s"].to_pylist() == [100.0, 80.0]

        warnings = [d for d in collected if d.severity is Severity.WARNING]
        assert len(warnings) == 1
        diag = warnings[0]
        assert diag.code == "hydro-bounds-clamped-to-declared-capacity"
        assert diag.table is not None
        row = diag.table.rows[0]
        assert row[0] == 0  # hydro id
        assert row[1] == "max_turbined_m3s"  # column
        assert "0" in row[2]  # stage
        assert row[3] == 100.0  # declared bound clamped to
        assert row[4] == 150.0  # original value

        # The clamped table passes the rule-43 error check.
        with dx.collect() as after:
            check_hydro_bounds_no_raising(hydros, result)
        assert not any(d.severity is Severity.ERROR for d in after)

    def test_outflow_vazmaxt_above_declared_max_is_clamped(self) -> None:
        hydros = _hydros(_hydro_with_envelope(7, outflow=(50.0, 500.0)))
        hydro_bounds = pa.table(
            {
                "hydro_id": pa.array([7], type=pa.int32()),
                "stage_id": pa.array([2], type=pa.int32()),
                "max_outflow_m3s": pa.array([600.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            result = clamp_hydro_bounds_to_declared(hydros, hydro_bounds)

        assert result["max_outflow_m3s"].to_pylist() == [500.0]
        assert any(d.severity is Severity.WARNING for d in collected)

    def test_null_declared_max_leaves_bound_untouched(self) -> None:
        # Run-of-river plant: no outflow ceiling declared -> nothing to clamp.
        hydros = _hydros(_hydro_with_envelope(1, outflow=(131.0, None)))
        hydro_bounds = pa.table(
            {
                "hydro_id": pa.array([1], type=pa.int32()),
                "stage_id": pa.array([0], type=pa.int32()),
                "max_outflow_m3s": pa.array([9999.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            result = clamp_hydro_bounds_to_declared(hydros, hydro_bounds)

        assert result is hydro_bounds  # unchanged object
        assert collected == []

    def test_value_below_declared_min_is_floored(self) -> None:
        # A MAX bound driven below the plant's own floor (max < min is
        # infeasible) is raised to the declared min — the "floor" half.
        hydros = _hydros(_hydro_with_envelope(4, outflow=(131.0, 800.0)))
        hydro_bounds = pa.table(
            {
                "hydro_id": pa.array([4], type=pa.int32()),
                "stage_id": pa.array([0], type=pa.int32()),
                "max_outflow_m3s": pa.array([100.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            result = clamp_hydro_bounds_to_declared(hydros, hydro_bounds)

        assert result["max_outflow_m3s"].to_pylist() == [131.0]
        assert any(d.severity is Severity.WARNING for d in collected)

    def test_within_envelope_is_untouched(self) -> None:
        hydros = _hydros(_hydro_with_envelope(0, turbined=(0.0, 100.0)))
        hydro_bounds = pa.table(
            {
                "hydro_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([0], type=pa.int32()),
                "max_turbined_m3s": pa.array([100.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            result = clamp_hydro_bounds_to_declared(hydros, hydro_bounds)

        assert result is hydro_bounds
        assert collected == []

    def test_none_table_returns_none(self) -> None:
        hydros = _hydros(_hydro_with_envelope(0, turbined=(0.0, 100.0)))
        assert clamp_hydro_bounds_to_declared(hydros, None) is None


# ---------------------------------------------------------------------------
# Rule 41 — check_unit_group_envelope
# ---------------------------------------------------------------------------


class TestUnitGroupEnvelope:
    def test_single_mirror_group_holds_by_construction(self) -> None:
        hydros = _hydros(_hydro(0, max_turbined_m3s=100.0, max_generation_mw=50.0))

        with dx.collect() as collected:
            check_unit_group_envelope(hydros)

        assert collected == []

    def test_group_sum_exceeding_declared_cap_is_caught(self) -> None:
        hydros = _hydros(
            _hydro(
                0,
                max_turbined_m3s=100.0,
                max_generation_mw=50.0,
                groups=[(60.0, 30.0), (60.0, 30.0)],  # sums to 120 / 60
            )
        )

        with dx.collect() as collected:
            check_unit_group_envelope(hydros)

        errors = [d for d in collected if d.severity is Severity.ERROR]
        assert len(errors) == 1
        diag = errors[0]
        assert diag.code == "hydro-unit-group-envelope-exceeded"
        assert diag.table is not None
        by_column = {row[1]: row for row in diag.table.rows}
        assert by_column["max_turbined_m3s"][2] == 100.0  # declared
        assert by_column["max_turbined_m3s"][3] == 120.0  # group sum
        assert by_column["max_generation_mw"][2] == 50.0
        assert by_column["max_generation_mw"][3] == 60.0

    def test_value_inside_relative_tolerance_passes(self) -> None:
        """Rule 41's tolerance mirrors novomodelo's ENVELOPE_TOLERANCE = 1e-9 * max(|v|, 1.0)."""
        declared = 100.0
        tolerance = 1e-9 * max(abs(declared), 1.0)
        hydros = _hydros(
            _hydro(
                0,
                max_turbined_m3s=declared,
                max_generation_mw=50.0,
                groups=[(declared + tolerance * 0.5, 50.0)],
            )
        )

        with dx.collect() as collected:
            check_unit_group_envelope(hydros)

        assert collected == []

    def test_value_outside_relative_tolerance_fails(self) -> None:
        declared = 100.0
        tolerance = 1e-9 * max(abs(declared), 1.0)
        hydros = _hydros(
            _hydro(
                0,
                max_turbined_m3s=declared,
                max_generation_mw=50.0,
                groups=[(declared + tolerance * 10, 50.0)],
            )
        )

        with dx.collect() as collected:
            check_unit_group_envelope(hydros)

        errors = [d for d in collected if d.severity is Severity.ERROR]
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# Rule 45 — check_group_bound_envelope
# ---------------------------------------------------------------------------


class TestGroupBoundEnvelope:
    def test_none_table_passes_without_a_finding(self) -> None:
        hydros = _hydros(_hydro(0, max_turbined_m3s=100.0, max_generation_mw=50.0))

        with dx.collect() as collected:
            check_group_bound_envelope(hydros, None)

        assert collected == []

    def test_empty_table_passes_without_a_finding(self) -> None:
        hydros = _hydros(_hydro(0, max_turbined_m3s=100.0, max_generation_mw=50.0))
        group_bounds = pa.table(
            {
                "hydro_id": pa.array([], type=pa.int32()),
                "hydro_unit_group_id": pa.array([], type=pa.int32()),
                "stage_id": pa.array([], type=pa.int32()),
                "max_generation_mw": pa.array([], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            check_group_bound_envelope(hydros, group_bounds)

        assert collected == []

    def test_row_raising_the_groups_own_declared_max_is_caught(self) -> None:
        hydros = _hydros(
            _hydro(
                0,
                max_turbined_m3s=100.0,
                max_generation_mw=50.0,
                groups=[(60.0, 30.0), (40.0, 20.0)],
            )
        )
        group_bounds = pa.table(
            {
                "hydro_id": pa.array([0], type=pa.int32()),
                "hydro_unit_group_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([2], type=pa.int32()),
                "max_generation_mw": pa.array([35.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            check_group_bound_envelope(hydros, group_bounds)

        errors = [d for d in collected if d.severity is Severity.ERROR]
        assert len(errors) == 1
        diag = errors[0]
        assert diag.code == "hydro-group-bounds-raises-declared-capacity"
        assert diag.table is not None
        row = diag.table.rows[0]
        # hydro, group, column, stages, declared, offending — all named.
        assert row[0] == 0
        assert row[1] == 0
        assert row[2] == "max_generation_mw"
        assert "2" in row[3]
        assert row[4] == 30.0
        assert row[5] == 35.0

    def test_row_at_or_below_the_groups_declared_max_passes(self) -> None:
        hydros = _hydros(
            _hydro(
                0,
                max_turbined_m3s=100.0,
                max_generation_mw=50.0,
                groups=[(60.0, 30.0), (40.0, 20.0)],
            )
        )
        group_bounds = pa.table(
            {
                "hydro_id": pa.array([0, 0], type=pa.int32()),
                "hydro_unit_group_id": pa.array([0, 1], type=pa.int32()),
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "max_generation_mw": pa.array([30.0, 15.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            check_group_bound_envelope(hydros, group_bounds)

        assert collected == []

    def test_a_groups_row_is_checked_against_its_own_declared_max_only(self) -> None:
        """The row is checked against *its own* group's declared max, not
        another group of the same plant — this is what distinguishes rule 45
        from rule 43 (:class:`TestHydroBoundsNoRaising`, the plant-envelope
        mirror)."""
        hydros = _hydros(
            _hydro(
                0,
                max_turbined_m3s=100.0,
                max_generation_mw=50.0,
                groups=[(60.0, 30.0), (40.0, 20.0)],
            )
        )
        group_bounds = pa.table(
            {
                "hydro_id": pa.array([0, 0], type=pa.int32()),
                "hydro_unit_group_id": pa.array([0, 1], type=pa.int32()),
                "stage_id": pa.array([0, 0], type=pa.int32()),
                # Group 0's row (25) is under its own declared 30 — legal
                # even though it exceeds group 1's declared 20.
                "max_generation_mw": pa.array([25.0, 20.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            check_group_bound_envelope(hydros, group_bounds)

        assert collected == []

    def test_unknown_group_id_is_skipped_without_a_finding(self) -> None:
        """Referential existence (does ``hydro_unit_group_id`` name a
        declared group) is out of scope for this rule — covered-by-
        construction while ``id = 0`` is the only group the bridge ever
        emits."""
        hydros = _hydros(_hydro(0, max_turbined_m3s=100.0, max_generation_mw=50.0))
        group_bounds = pa.table(
            {
                "hydro_id": pa.array([0], type=pa.int32()),
                "hydro_unit_group_id": pa.array([7], type=pa.int32()),
                "stage_id": pa.array([0], type=pa.int32()),
                "max_generation_mw": pa.array([9999.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            check_group_bound_envelope(hydros, group_bounds)

        assert collected == []

    def test_value_inside_relative_tolerance_passes(self) -> None:
        declared = 100.0
        tolerance = 1e-9 * max(abs(declared), 1.0)
        hydros = _hydros(
            _hydro(
                0,
                max_turbined_m3s=100.0,
                max_generation_mw=declared,
                groups=[(100.0, declared)],
            )
        )
        group_bounds = pa.table(
            {
                "hydro_id": pa.array([0], type=pa.int32()),
                "hydro_unit_group_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([0], type=pa.int32()),
                "max_generation_mw": pa.array(
                    [declared + tolerance * 0.5], type=pa.float64()
                ),
            }
        )

        with dx.collect() as collected:
            check_group_bound_envelope(hydros, group_bounds)

        assert collected == []

    def test_value_outside_relative_tolerance_fails(self) -> None:
        declared = 100.0
        tolerance = 1e-9 * max(abs(declared), 1.0)
        hydros = _hydros(
            _hydro(
                0,
                max_turbined_m3s=100.0,
                max_generation_mw=declared,
                groups=[(100.0, declared)],
            )
        )
        group_bounds = pa.table(
            {
                "hydro_id": pa.array([0], type=pa.int32()),
                "hydro_unit_group_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([0], type=pa.int32()),
                "max_generation_mw": pa.array(
                    [declared + tolerance * 10], type=pa.float64()
                ),
            }
        )

        with dx.collect() as collected:
            check_group_bound_envelope(hydros, group_bounds)

        errors = [d for d in collected if d.severity is Severity.ERROR]
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# Rule 36 — check_bound_row_uniqueness
# ---------------------------------------------------------------------------


class TestBoundRowUniqueness:
    def test_duplicate_stage_wide_column_is_caught(self) -> None:
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
        assert errors[0].code == "bound-row-duplicate"
        assert errors[0].table is not None
        row = errors[0].table.rows[0]
        assert row[0] == "Hydro"
        assert row[1] == 0
        assert row[2] == 0
        assert row[4] == "max_turbined_m3s"

    def test_stage_wide_row_and_per_block_rows_coexist(self) -> None:
        """A None-block base row plus Some(b) per-block rows on the same
        (entity, stage, column) is legal usage (the line_bounds
        convention), not a duplicate."""
        table = pa.table(
            {
                "line_id": pa.array([0, 0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 0, 0], type=pa.int32()),
                "block_id": pa.array([None, 0, 1], type=pa.int32()),
                "direct_mw": pa.array([100.0, 80.0, 100.0], type=pa.float64()),
            }
        )
        families = [BoundFamily("Line", "line_id", table)]

        with dx.collect() as collected:
            check_bound_row_uniqueness(families)

        assert collected == []

    def test_disjoint_columns_on_same_key_are_not_duplicates(self) -> None:
        table = pa.table(
            {
                "hydro_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([0], type=pa.int32()),
                "min_turbined_m3s": pa.array([1.0], type=pa.float64()),
                "max_turbined_m3s": pa.array([2.0], type=pa.float64()),
            }
        )
        families = [BoundFamily("Hydro", "hydro_id", table)]

        with dx.collect() as collected:
            check_bound_row_uniqueness(families)

        assert collected == []

    def test_absent_family_table_is_skipped_without_a_finding(self) -> None:
        families = [BoundFamily("Hydro", "hydro_id", None)]

        with dx.collect() as collected:
            check_bound_row_uniqueness(families)

        assert collected == []


# ---------------------------------------------------------------------------
# block_id range — check_bound_block_id_range
# ---------------------------------------------------------------------------


class TestBoundBlockIdRange:
    def test_out_of_range_block_id_is_caught(self) -> None:
        stages = _stages(3, 2)  # stage 0 -> 3 blocks, stage 1 -> 2 blocks
        table = pa.table(
            {
                "line_id": pa.array([1], type=pa.int32()),
                "stage_id": pa.array([1], type=pa.int32()),
                "block_id": pa.array([2], type=pa.int32()),  # stage 1 has 0..2
                "direct_mw": pa.array([10.0], type=pa.float64()),
            }
        )
        families = [BoundFamily("Line", "line_id", table)]

        with dx.collect() as collected:
            check_bound_block_id_range(stages, families)

        errors = [d for d in collected if d.severity is Severity.ERROR]
        assert len(errors) == 1
        diag = errors[0]
        assert diag.code == "bound-block-id-out-of-range"
        assert diag.table is not None
        row = diag.table.rows[0]
        assert row[0] == "Line"
        assert row[1] == 1  # entity
        assert row[2] == 1  # stage
        assert row[3] == 2  # offending block_id
        assert row[4] == 2  # declared block count

    def test_in_range_block_id_passes(self) -> None:
        stages = _stages(3, 2)
        table = pa.table(
            {
                "line_id": pa.array([1, 2], type=pa.int32()),
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "block_id": pa.array([2, 1], type=pa.int32()),
                "direct_mw": pa.array([10.0, 20.0], type=pa.float64()),
            }
        )
        families = [BoundFamily("Line", "line_id", table)]

        with dx.collect() as collected:
            check_bound_block_id_range(stages, families)

        assert collected == []

    def test_unknown_stage_id_is_skipped_without_a_finding(self) -> None:
        stages = _stages(3)
        table = pa.table(
            {
                "line_id": pa.array([1], type=pa.int32()),
                "stage_id": pa.array([99], type=pa.int32()),
                "block_id": pa.array([0], type=pa.int32()),
                "direct_mw": pa.array([10.0], type=pa.float64()),
            }
        )
        families = [BoundFamily("Line", "line_id", table)]

        with dx.collect() as collected:
            check_bound_block_id_range(stages, families)

        assert collected == []

    def test_family_without_block_id_column_is_skipped(self) -> None:
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
            check_bound_block_id_range(stages, families)

        assert collected == []


# ---------------------------------------------------------------------------
# Rule 38 — check_block_id_not_on_anticipated_thermal
# ---------------------------------------------------------------------------


class TestBlockIdNotOnAnticipatedThermal:
    def test_block_id_on_anticipated_thermal_is_caught(self) -> None:
        thermals = _thermals(_thermal(0, anticipated=True))
        thermal_bounds = pa.table(
            {
                "thermal_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([2], type=pa.int32()),
                "min_generation_mw": pa.array([10.0], type=pa.float64()),
                "max_generation_mw": pa.array([20.0], type=pa.float64()),
                "cost_per_mwh": pa.array([None], type=pa.float64()),
                "block_id": pa.array([1], type=pa.int32()),
            }
        )

        with dx.collect() as collected:
            check_block_id_not_on_anticipated_thermal(thermals, thermal_bounds)

        assert len(collected) == 1
        diag = collected[0]
        assert diag.severity is Severity.ERROR
        assert diag.code == "thermal-bound-block-id-on-anticipated"
        assert diag.table is not None
        row = diag.table.rows[0]
        assert row[0] == 0  # thermal id
        assert row[1] == 2  # stage
        assert row[2] == 1  # block_id

    def test_block_id_without_anticipated_config_passes(self) -> None:
        """Same table shape, but no thermal declares anticipated_config."""
        thermals = _thermals(_thermal(0, anticipated=False))
        thermal_bounds = pa.table(
            {
                "thermal_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([2], type=pa.int32()),
                "min_generation_mw": pa.array([10.0], type=pa.float64()),
                "max_generation_mw": pa.array([20.0], type=pa.float64()),
                "cost_per_mwh": pa.array([None], type=pa.float64()),
                "block_id": pa.array([1], type=pa.int32()),
            }
        )

        with dx.collect() as collected:
            check_block_id_not_on_anticipated_thermal(thermals, thermal_bounds)

        assert collected == []

    def test_anticipated_thermal_with_no_block_id_rows_passes(self) -> None:
        """An anticipated thermal is fine as long as its own rows stay
        stage-level; another thermal's block_id row is not its problem."""
        thermals = _thermals(_thermal(0, anticipated=True), _thermal(1))
        thermal_bounds = pa.table(
            {
                "thermal_id": pa.array([0, 1], type=pa.int32()),
                "stage_id": pa.array([0, 0], type=pa.int32()),
                "min_generation_mw": pa.array([10.0, 5.0], type=pa.float64()),
                "max_generation_mw": pa.array([20.0, 15.0], type=pa.float64()),
                "cost_per_mwh": pa.array([5.0, None], type=pa.float64()),
                "block_id": pa.array([None, 0], type=pa.int32()),
            }
        )

        with dx.collect() as collected:
            check_block_id_not_on_anticipated_thermal(thermals, thermal_bounds)

        assert collected == []

    def test_absent_block_id_column_is_not_applicable(self) -> None:
        thermals = _thermals(_thermal(0, anticipated=True))
        thermal_bounds = pa.table(
            {
                "thermal_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([0], type=pa.int32()),
                "min_generation_mw": pa.array([10.0], type=pa.float64()),
                "max_generation_mw": pa.array([20.0], type=pa.float64()),
                "cost_per_mwh": pa.array([5.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            check_block_id_not_on_anticipated_thermal(thermals, thermal_bounds)

        assert collected == []

    def test_absent_thermal_bounds_table_is_not_applicable(self) -> None:
        thermals = _thermals(_thermal(0, anticipated=True))

        with dx.collect() as collected:
            check_block_id_not_on_anticipated_thermal(thermals, None)

        assert collected == []


# ---------------------------------------------------------------------------
# run_and_gate
# ---------------------------------------------------------------------------


class TestRunAndGate:
    def test_error_diagnostic_raises_emission_check_error_naming_the_code(
        self,
    ) -> None:
        def run_checks() -> None:
            dx.emit(
                dx.Diagnostic(
                    code="synthetic-error",
                    severity=Severity.ERROR,
                    category="Emission self-checks",
                    title="Synthetic error",
                    summary="a synthetic ERROR finding for run_and_gate",
                )
            )

        with pytest.raises(EmissionCheckError) as exc_info:
            run_and_gate(run_checks)

        assert "synthetic-error" in str(exc_info.value)

    def test_warning_and_info_only_returns_none(self) -> None:
        def run_checks() -> None:
            dx.emit(
                dx.Diagnostic(
                    code="synthetic-warning",
                    severity=Severity.WARNING,
                    category="Emission self-checks",
                    title="Synthetic warning",
                    summary="a synthetic WARNING finding for run_and_gate",
                )
            )
            dx.emit(
                dx.Diagnostic(
                    code="synthetic-info",
                    severity=Severity.INFO,
                    category="Emission self-checks",
                    title="Synthetic info",
                    summary="a synthetic INFO finding for run_and_gate",
                )
            )

        assert run_and_gate(run_checks) is None


# ---------------------------------------------------------------------------
# run_and_gate over the NEWAVE pipeline's four-check closure
# ---------------------------------------------------------------------------


class TestNewaveEmissionGate:
    """Mirrors pipeline.py's ``_run_checks`` closure order: rule 43, rule 41,
    rule 36, then the block_id-range rule."""

    def _run_checks(
        self,
        hydros: dict,
        stages: dict,
        bound_families: list[BoundFamily],
        hydro_bounds: pa.Table | None,
    ) -> None:
        check_hydro_bounds_no_raising(hydros, hydro_bounds)
        check_unit_group_envelope(hydros)
        check_bound_row_uniqueness(bound_families)
        check_bound_block_id_range(stages, bound_families)

    def test_duplicate_bound_row_raises_emission_check_error(self) -> None:
        hydros = _hydros(_hydro(0, max_turbined_m3s=100.0, max_generation_mw=50.0))
        stages = _stages(2)
        hydro_bounds = pa.table(
            {
                "hydro_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 0], type=pa.int32()),
                "max_turbined_m3s": pa.array([10.0, 20.0], type=pa.float64()),
            }
        )
        bound_families = [BoundFamily("Hydro", "hydro_id", hydro_bounds)]

        with pytest.raises(EmissionCheckError) as exc_info:
            run_and_gate(
                lambda: self._run_checks(hydros, stages, bound_families, hydro_bounds)
            )

        assert "bound-row-duplicate" in str(exc_info.value)

    def test_clean_bound_set_returns_none(self) -> None:
        hydros = _hydros(_hydro(0, max_turbined_m3s=100.0, max_generation_mw=50.0))
        stages = _stages(2)
        hydro_bounds = pa.table(
            {
                "hydro_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([0], type=pa.int32()),
                "max_turbined_m3s": pa.array([10.0], type=pa.float64()),
            }
        )
        bound_families = [BoundFamily("Hydro", "hydro_id", hydro_bounds)]

        result = run_and_gate(
            lambda: self._run_checks(hydros, stages, bound_families, hydro_bounds)
        )

        assert result is None
