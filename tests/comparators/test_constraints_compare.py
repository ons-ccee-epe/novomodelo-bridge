"""Branch coverage for ``comparators.constraints`` and ``comparators.newave.constraints``.

Uses the committed converted-Novomodelo-input fixture (``generic_constraints.json``)
and a NEWAVE ``MEDIAS-USIH.CSV`` result under ``fixtures/constraints_compare/``,
plus an in-test echo, simulation parquet, ``hydros.json`` and ``stages.json``
(built under ``tmp_path`` rather than committed) to exercise the resolved-view
loader, the bound-resolution helpers and both ``evaluate_lhs_*`` entry points.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from novomodelo_bridge.comparators.constraints import (
    _resolve_bound,
    evaluate_lhs_novomodelo,
    load_generic_constraints,
    load_resolved_generic_constraints,
    per_stage_bounds,
)
from novomodelo_bridge.comparators.newave.alignment import EntityAlignment
from novomodelo_bridge.comparators.newave.constraints import evaluate_lhs_newave
from novomodelo_bridge.comparators.newave.readers import read_medias_hydro
from novomodelo_bridge.newave.id_map import NewaveIdMap
from tests.conftest import write_generic_constraint_echo

_NOVOMODELO_INPUT_DIR = (
    Path(__file__).parent.parent
    / "fixtures"
    / "constraints_compare"
    / "novomodelo_input"
)
_NEWAVE_DIR = (
    Path(__file__).parent.parent / "fixtures" / "constraints_compare" / "newave"
)

# ``fixtures/constraints_compare/novomodelo_input/constraints/generic_constraints.json``:
# constraint 0 = non-VminOP generation sum (RE-style, "<=" bound); constraints
# 1 and 2 = the stage-0 and stage-1 members of one VminOP family, shown merged
# under id 1.
_NON_VMINOP_ID = 0
_VMINOP_ID = 1

_EMPTY_LINE_MEANS = pl.DataFrame(
    schema={
        "from_submarket_code": pl.Int64,
        "to_submarket_code": pl.Int64,
        "from_name": pl.Utf8,
        "to_name": pl.Utf8,
        "stage": pl.Int64,
        "variable": pl.Utf8,
        "value": pl.Float64,
    }
)


def _id_map() -> NewaveIdMap:
    # hydro_codes sorted ascending -> novomodelo id: 10 -> 0, 20 -> 1.
    return NewaveIdMap(subsystem_ids=[1], hydro_codes=[10, 20], thermal_codes=[])


def _write_case(tmp_path: Path) -> tuple[Path, Path]:
    """A converted case (fixture constraints, hydros, stages) plus its run output.

    The echo carries novomodelo's coefficients: 1.0 on each generation term, the
    integrated ρ (2.0 / 2.5 MWh/hm³) on the useful-volume term, whose bound holds
    the ``2.0 · 100`` / ``2.5 · 100`` dead-volume energy.
    """
    case_dir = tmp_path / "case"
    output_dir = case_dir / "output"
    shutil.copytree(_NOVOMODELO_INPUT_DIR / "constraints", case_dir / "constraints")
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "system" / "hydros.json").write_text(
        json.dumps(
            {
                "hydros": [
                    {"id": 0, "reservoir": {"min_storage_hm3": 100.0}},
                    {"id": 1, "reservoir": {"min_storage_hm3": 50.0}},
                ]
            }
        )
    )
    (case_dir / "stages.json").write_text(
        json.dumps(
            {
                "stages": [
                    {"id": 0, "blocks": [{"id": 0, "hours": 744.0}]},
                    {"id": 1, "blocks": [{"id": 0, "hours": 696.0}]},
                ]
            }
        )
    )
    useful = "hydro_useful_volume_final(id=0, block=all)"
    write_generic_constraint_echo(
        output_dir,
        [
            (
                0,
                "RE_generation_sum",
                0,
                0,
                "hydro_generation(id=0, bus=all, block=0)",
                1.0,
                None,
                200.0,
            ),
            (
                0,
                "RE_generation_sum",
                0,
                0,
                "hydro_generation(id=1, bus=all, block=0)",
                1.0,
                None,
                200.0,
            ),
            (
                0,
                "RE_generation_sum",
                1,
                0,
                "hydro_generation(id=0, bus=all, block=0)",
                1.0,
                None,
                210.0,
            ),
            (
                0,
                "RE_generation_sum",
                1,
                0,
                "hydro_generation(id=1, bus=all, block=0)",
                1.0,
                None,
                210.0,
            ),
            (1, "VminOP_SUDESTE", 0, None, useful, 2.0, 300.0, None),
            (1, "VminOP_SUDESTE", 1, None, useful, 2.5, 400.0, None),
        ],
    )
    sim_dir = output_dir / "simulation" / "hydros" / "scenario_id=0"
    sim_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "stage_id": pa.array([0, 0, 1, 1], type=pa.int32()),
                "block_id": pa.array([0, 0, 0, 0], type=pa.int32()),
                "hydro_id": pa.array([0, 1, 0, 1], type=pa.int32()),
                "storage_final_hm3": pa.array([500.0, 300.0, 480.0, 290.0]),
                "generation_mw": pa.array([95.0, 88.0, 97.0, 90.0]),
            }
        ),
        sim_dir / "data.parquet",
    )
    return case_dir, output_dir


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


class TestLoadGenericConstraints:
    def test_parses_sense_free_constraints(self) -> None:
        constraints = load_generic_constraints(_NOVOMODELO_INPUT_DIR)
        assert {c["id"] for c in constraints} == {0, 1}
        assert all("sense" not in c for c in constraints)

    def test_degrades_to_empty_list_on_missing_file(self, tmp_path: Path) -> None:
        assert load_generic_constraints(tmp_path) == []

    def test_degrades_to_empty_list_and_warns_on_malformed_json(
        self, tmp_path: Path, caplog
    ) -> None:
        path = tmp_path / "constraints" / "generic_constraints.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not valid json")

        with caplog.at_level(logging.WARNING):
            constraints = load_generic_constraints(tmp_path)

        assert constraints == []
        assert "could not be parsed" in caplog.text


class TestLoadResolvedGenericConstraints:
    def test_shows_each_stage_vminop_bound_as_useful_energy(
        self, tmp_path: Path
    ) -> None:
        case_dir, output_dir = _write_case(tmp_path)

        resolved = load_resolved_generic_constraints(case_dir, output_dir)

        assert [(c["id"], c["name"]) for c in resolved.constraints] == [
            (_NON_VMINOP_ID, "RE_generation_sum"),
            (_VMINOP_ID, "VminOP_SUDESTE"),
        ]
        vminop = resolved.bounds.filter(pl.col("constraint_id") == _VMINOP_ID)
        assert vminop["bound_lower"].to_list() == [300.0 - 200.0, 400.0 - 250.0]

    def test_no_echo_leaves_no_bounds(self, tmp_path: Path) -> None:
        shutil.copytree(_NOVOMODELO_INPUT_DIR / "constraints", tmp_path / "constraints")

        resolved = load_resolved_generic_constraints(tmp_path, tmp_path / "output")

        assert len(resolved.constraints) == 2
        assert resolved.bounds.is_empty()


# ---------------------------------------------------------------------------
# Bound resolution
# ---------------------------------------------------------------------------


class TestResolveBound:
    def test_lower_only_resolves_to_ge(self) -> None:
        resolved = _resolve_bound(100.0, None)
        assert resolved.value == 100.0
        assert resolved.shape == ">="

    def test_upper_only_resolves_to_le(self) -> None:
        resolved = _resolve_bound(None, 200.0)
        assert resolved.value == 200.0
        assert resolved.shape == "<="

    def test_equal_endpoints_resolve_to_eq(self) -> None:
        resolved = _resolve_bound(300.0, 300.0)
        assert resolved.value == 300.0
        assert resolved.shape == "=="

    def test_distinct_endpoints_resolve_to_range_using_upper(self) -> None:
        resolved = _resolve_bound(100.0, 200.0)
        assert resolved.value == 200.0
        assert resolved.shape == "range"


class TestPerStageBounds:
    def test_resolves_both_constraints(self, tmp_path: Path) -> None:
        bounds = load_resolved_generic_constraints(*_write_case(tmp_path)).bounds
        resolved = per_stage_bounds(bounds)
        assert resolved[_NON_VMINOP_ID][0].shape == "<="
        assert resolved[_NON_VMINOP_ID][0].value == 200.0
        assert resolved[_VMINOP_ID][0].shape == ">="
        assert resolved[_VMINOP_ID][0].value == 100.0

    def test_max_stage_drops_later_stages(self, tmp_path: Path) -> None:
        bounds = load_resolved_generic_constraints(*_write_case(tmp_path)).bounds
        resolved = per_stage_bounds(bounds, max_stage=0)
        assert set(resolved[_NON_VMINOP_ID]) == {0}

    def test_empty_bounds_yields_empty_dict(self) -> None:
        empty = pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "block_id": pl.Int32,
                "bound_lower": pl.Float64,
                "bound_upper": pl.Float64,
            }
        )
        assert per_stage_bounds(empty) == {}


# ---------------------------------------------------------------------------
# evaluate_lhs_newave / evaluate_lhs_novomodelo
# ---------------------------------------------------------------------------


class TestEvaluateLhsNewave:
    def _lhs(self, tmp_path: Path) -> dict[tuple[int, int], float]:
        resolved = load_resolved_generic_constraints(*_write_case(tmp_path))
        lhs = evaluate_lhs_newave(
            resolved,
            read_medias_hydro(_NEWAVE_DIR),
            _EMPTY_LINE_MEANS,
            EntityAlignment(),
            _id_map(),
            1,
        )
        return {
            (r["constraint_id"], r["stage_id"]): r["lhs_value"]
            for r in lhs.iter_rows(named=True)
        }

    def test_evaluates_non_vminop_generation_sum(self, tmp_path: Path) -> None:
        rows = self._lhs(tmp_path)
        # GHIDUH(hydro 10) + GHIDUH(hydro 20) per stage: 100+90, 102+92.
        assert rows[(_NON_VMINOP_ID, 0)] == 190.0
        assert rows[(_NON_VMINOP_ID, 1)] == 194.0

    def test_vminop_weights_varmuh_by_novomodelo_coefficient(
        self, tmp_path: Path
    ) -> None:
        rows = self._lhs(tmp_path)
        # novomodelo's integrated ρ × VARMUH(hydro 10), the useful stored volume.
        assert rows[(_VMINOP_ID, 0)] == pytest.approx(2.0 * 40.0)
        assert rows[(_VMINOP_ID, 1)] == pytest.approx(2.5 * 42.0)

    def test_degrades_to_empty_frame_without_echo(self, tmp_path: Path) -> None:
        shutil.copytree(_NOVOMODELO_INPUT_DIR / "constraints", tmp_path / "constraints")
        resolved = load_resolved_generic_constraints(tmp_path, tmp_path / "output")
        lhs = evaluate_lhs_newave(
            resolved,
            read_medias_hydro(_NEWAVE_DIR),
            _EMPTY_LINE_MEANS,
            EntityAlignment(),
            _id_map(),
            1,
        )
        assert lhs.is_empty()


class TestEvaluateLhsNovomodelo:
    def test_evaluates_both_constraints_from_simulation(self, tmp_path: Path) -> None:
        case_dir, output_dir = _write_case(tmp_path)
        resolved = load_resolved_generic_constraints(case_dir, output_dir)

        lhs = evaluate_lhs_novomodelo(resolved, output_dir)

        rows = {
            (r["constraint_id"], r["stage_id"]): r["lhs_value"]
            for r in lhs.iter_rows(named=True)
        }
        # constraint 0: generation_mw(hydro 0) + generation_mw(hydro 1).
        assert rows[(_NON_VMINOP_ID, 0)] == 95.0 + 88.0
        assert rows[(_NON_VMINOP_ID, 1)] == 97.0 + 90.0
        # VminOP: novomodelo's integrated ρ × (storage_final − min_storage), useful energy.
        assert rows[(_VMINOP_ID, 0)] == 2.0 * (500.0 - 100.0)
        assert rows[(_VMINOP_ID, 1)] == 2.5 * (480.0 - 100.0)

    def test_degrades_to_empty_frame_when_no_hydro_simulation(
        self, tmp_path: Path, caplog
    ) -> None:
        case_dir, output_dir = _write_case(tmp_path)
        resolved = load_resolved_generic_constraints(case_dir, output_dir)
        shutil.rmtree(output_dir / "simulation")

        with caplog.at_level(logging.WARNING):
            lhs = evaluate_lhs_novomodelo(resolved, output_dir)

        assert lhs.is_empty()
        assert "Simulation directory not found" in caplog.text
