"""Branch coverage for ``comparators.constraints`` and ``comparators.newave.constraints``.

Uses the committed converted-Novomodelo-input fixture (``generic_constraints.json``
+ ``generic_constraint_bounds.parquet``) and a NEWAVE ``MEDIAS-USIH.CSV``
result under ``fixtures/constraints_compare/``, plus a small in-test pyarrow
simulation parquet and in-test ``generic_parameters.json`` / ``hydros.json``
(built under ``tmp_path`` rather than committed, since they are needed only
by ``apply_vminop_useful_energy``) to exercise the bound-resolution helpers,
the two loader pairs behind the VminOP useful-energy rewrite, and both
``evaluate_lhs_*`` entry points.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from novomodelo_bridge.comparators.constraints import (
    _resolve_bound,
    evaluate_lhs_novomodelo,
    load_generic_constraint_bounds,
    load_generic_constraints,
    per_stage_bounds,
)
from novomodelo_bridge.comparators.newave.alignment import EntityAlignment
from novomodelo_bridge.comparators.newave.constraints import (
    _load_hydro_min_storage,
    apply_vminop_useful_energy,
    evaluate_lhs_newave,
)
from novomodelo_bridge.comparators.newave.readers import read_medias_hydro
from novomodelo_bridge.newave.id_map import NewaveIdMap
from novomodelo_bridge.novomodelo.constraint_expr import (
    load_rho_acum_overrides,
    scales_storage_by_rho_acum,
)

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
# constraint 0 = non-VminOP generation sum (RE-style, "<=" bound); constraint
# 1 = VminOP storage constraint scaled by ``@rho_acum_h0`` (">=" bound). The
# companion bounds parquet carries 2 stages (0, 1) for each.
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


def _write_sim_hydros_parquet(output_dir: Path) -> None:
    """Small in-test simulation parquet: 2 hydros x 2 stages x 1 block x 1
    scenario, under the hive-partitioned layout ``scan_simulation_entity``
    expects."""
    sim_dir = output_dir / "simulation" / "hydros" / "scenario_id=0"
    sim_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "stage_id": pa.array([0, 0, 1, 1], type=pa.int32()),
            "block_id": pa.array([0, 0, 0, 0], type=pa.int32()),
            "hydro_id": pa.array([0, 1, 0, 1], type=pa.int32()),
            "storage_final_hm3": pa.array(
                [500.0, 300.0, 480.0, 290.0], type=pa.float64()
            ),
            "generation_mw": pa.array([95.0, 88.0, 97.0, 90.0], type=pa.float64()),
            "accumulated_productivity_mw_per_m3s": pa.array(
                [0.5, 0.4, 0.52, 0.42], type=pa.float64()
            ),
        }
    )
    pq.write_table(table, sim_dir / "data.parquet")


def _write_generic_parameters(novomodelo_case_dir: Path) -> None:
    path = novomodelo_case_dir / "constraints" / "generic_parameters.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "scalar_parameters": [
                    {
                        "name": "rho_acum_h0",
                        "kind": "per_stage",
                        "values": [[0, 0.5], [1, 0.52]],
                    }
                ]
            }
        )
    )


def _write_hydros_json(novomodelo_case_dir: Path) -> None:
    path = novomodelo_case_dir / "system" / "hydros.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "hydros": [
                    {"id": 0, "reservoir": {"min_storage_hm3": 100.0}},
                    {"id": 1, "reservoir": {"min_storage_hm3": 50.0}},
                ]
            }
        )
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


class TestLoadGenericConstraints:
    def test_parses_sense_free_constraints(self) -> None:
        constraints = load_generic_constraints(_NOVOMODELO_INPUT_DIR)
        assert {c["id"] for c in constraints} == {_NON_VMINOP_ID, _VMINOP_ID}
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


class TestLoadGenericConstraintBounds:
    def test_parses_f3_bound_endpoints(self) -> None:
        df = load_generic_constraint_bounds(_NOVOMODELO_INPUT_DIR)
        assert "bound" not in df.columns
        assert set(df["constraint_id"].to_list()) == {_NON_VMINOP_ID, _VMINOP_ID}

    def test_degrades_to_empty_typed_frame_on_missing_file(
        self, tmp_path: Path
    ) -> None:
        df = load_generic_constraint_bounds(tmp_path)
        assert df.is_empty()
        assert set(df.columns) == {
            "constraint_id",
            "stage_id",
            "block_id",
            "bound_lower",
            "bound_upper",
        }


class TestLoadRhoAcumOverrides:
    def test_parses_per_stage_values(self, tmp_path: Path) -> None:
        _write_generic_parameters(tmp_path)
        rho = load_rho_acum_overrides(tmp_path)
        assert rho == {0: {0: 0.5, 1: 0.52}}

    def test_degrades_to_empty_dict_on_missing_file(self, tmp_path: Path) -> None:
        assert load_rho_acum_overrides(tmp_path) == {}


class TestLoadHydroMinStorage:
    def test_parses_min_storage_per_hydro(self, tmp_path: Path) -> None:
        _write_hydros_json(tmp_path)
        assert _load_hydro_min_storage(tmp_path) == {0: 100.0, 1: 50.0}

    def test_degrades_to_empty_dict_on_missing_file(self, tmp_path: Path) -> None:
        assert _load_hydro_min_storage(tmp_path) == {}


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
    def test_resolves_both_constraints_from_fixture(self) -> None:
        bounds = load_generic_constraint_bounds(_NOVOMODELO_INPUT_DIR)
        resolved = per_stage_bounds(bounds)
        assert resolved[_NON_VMINOP_ID][0].shape == "<="
        assert resolved[_NON_VMINOP_ID][0].value == 200.0
        assert resolved[_VMINOP_ID][0].shape == ">="
        assert resolved[_VMINOP_ID][0].value == 100.0

    def test_max_stage_drops_later_stages(self) -> None:
        bounds = load_generic_constraint_bounds(_NOVOMODELO_INPUT_DIR)
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


class TestScalesStorageByRhoAcum:
    def test_true_for_rho_acum_scaled_storage(self) -> None:
        constraints = load_generic_constraints(_NOVOMODELO_INPUT_DIR)
        vminop = next(c for c in constraints if c["id"] == _VMINOP_ID)
        assert scales_storage_by_rho_acum(vminop) is True

    def test_false_for_plain_generation_sum(self) -> None:
        constraints = load_generic_constraints(_NOVOMODELO_INPUT_DIR)
        non_vminop = next(c for c in constraints if c["id"] == _NON_VMINOP_ID)
        assert scales_storage_by_rho_acum(non_vminop) is False


# ---------------------------------------------------------------------------
# evaluate_lhs_newave / evaluate_lhs_novomodelo
# ---------------------------------------------------------------------------


class TestEvaluateLhsNewave:
    def test_evaluates_non_vminop_generation_sum(self) -> None:
        constraints = load_generic_constraints(_NOVOMODELO_INPUT_DIR)
        nw_hydro = read_medias_hydro(_NEWAVE_DIR)
        lhs = evaluate_lhs_newave(
            constraints, nw_hydro, _EMPTY_LINE_MEANS, EntityAlignment(), _id_map(), 1
        )
        rows = {r["stage_id"]: r["lhs_value"] for r in lhs.iter_rows(named=True)}
        # GHIDUH(hydro 10) + GHIDUH(hydro 20) per stage: 100+90, 102+92.
        assert rows == {0: 190.0, 1: 194.0}

    def test_vminop_rho_param_constraint_is_skipped(self) -> None:
        """@rho_acum params have no source-model-side productivity handy, so
        the generic evaluator skips those stages entirely (no zero row)."""
        constraints = load_generic_constraints(_NOVOMODELO_INPUT_DIR)
        nw_hydro = read_medias_hydro(_NEWAVE_DIR)
        lhs = evaluate_lhs_newave(
            constraints, nw_hydro, _EMPTY_LINE_MEANS, EntityAlignment(), _id_map(), 1
        )
        assert _VMINOP_ID not in lhs["constraint_id"].to_list()

    def test_degrades_to_empty_frame_on_no_constraints(self) -> None:
        nw_hydro = read_medias_hydro(_NEWAVE_DIR)
        lhs = evaluate_lhs_newave(
            [], nw_hydro, _EMPTY_LINE_MEANS, EntityAlignment(), _id_map(), 1
        )
        assert lhs.is_empty()


class TestEvaluateLhsNovomodelo:
    def test_evaluates_both_constraints_from_simulation(self, tmp_path: Path) -> None:
        _write_sim_hydros_parquet(tmp_path)
        constraints = load_generic_constraints(_NOVOMODELO_INPUT_DIR)

        lhs = evaluate_lhs_novomodelo(constraints, tmp_path)

        rows = {
            (r["constraint_id"], r["stage_id"]): r["lhs_value"]
            for r in lhs.iter_rows(named=True)
        }
        # constraint 0: generation_mw(hydro 0) + generation_mw(hydro 1).
        assert rows[(_NON_VMINOP_ID, 0)] == 95.0 + 88.0
        assert rows[(_NON_VMINOP_ID, 1)] == 97.0 + 90.0
        # constraint 1, no rho_acum_overrides given: falls back to the
        # simulation's default accumulated_productivity_mw_per_m3s column.
        assert rows[(_VMINOP_ID, 0)] == 0.5 * 500.0
        assert rows[(_VMINOP_ID, 1)] == 0.52 * 480.0

    def test_degrades_to_empty_frame_on_no_constraints(self, tmp_path: Path) -> None:
        _write_sim_hydros_parquet(tmp_path)
        assert evaluate_lhs_novomodelo([], tmp_path).is_empty()

    def test_degrades_to_empty_frame_when_no_hydro_simulation(
        self, tmp_path: Path, caplog
    ) -> None:
        constraints = load_generic_constraints(_NOVOMODELO_INPUT_DIR)
        with caplog.at_level(logging.WARNING):
            lhs = evaluate_lhs_novomodelo(constraints, tmp_path)
        assert lhs.is_empty()
        assert "Simulation directory not found" in caplog.text


class TestEvaluateLhsNovomodeloRhoAcumOverride:
    """Regression for the dashboard/DECOMP-compare LHS-scale bug: without
    ``rho_acum_overrides``, a ``@rho_acum_h{id}``-scaled constraint (VminOP,
    RHE) silently resolves against the simulation's *default* productivity
    column, which sits on a different scale than the LP's own per-stage
    override -- so the evaluated LHS is incomparable to its own bound."""

    def test_override_replaces_default_productivity_column(
        self, tmp_path: Path
    ) -> None:
        _write_sim_hydros_parquet(tmp_path)
        constraints = load_generic_constraints(_NOVOMODELO_INPUT_DIR)
        # Deliberately different from the sim default (0.5 at stage 0, see
        # _write_sim_hydros_parquet) -- mirrors the real VminOP/RHE gap
        # between novomodelo's computed default and the LP's per_stage override.
        overrides = {0: {0: 2.19, 1: 2.5}}

        default_lhs = evaluate_lhs_novomodelo(constraints, tmp_path)
        override_lhs = evaluate_lhs_novomodelo(constraints, tmp_path, overrides)

        def _row(df: pl.DataFrame, cid: int, stage: int) -> float:
            sub = df.filter(
                (pl.col("constraint_id") == cid) & (pl.col("stage_id") == stage)
            )
            return float(sub["lhs_value"][0])

        assert _row(default_lhs, _VMINOP_ID, 0) == 0.5 * 500.0
        assert _row(override_lhs, _VMINOP_ID, 0) == 2.19 * 500.0
        assert _row(override_lhs, _VMINOP_ID, 1) == 2.5 * 480.0
        assert _row(override_lhs, _VMINOP_ID, 0) != _row(default_lhs, _VMINOP_ID, 0)

        # A constraint with no @rho_acum term is unaffected by the override.
        assert _row(override_lhs, _NON_VMINOP_ID, 0) == _row(
            default_lhs, _NON_VMINOP_ID, 0
        )

    def test_override_can_flip_a_falsely_violated_stage_to_satisfied(
        self, tmp_path: Path
    ) -> None:
        """A wrong (too-low) default productivity can make a satisfied
        constraint LOOK violated; the LP-faithful override must not."""
        _write_sim_hydros_parquet(tmp_path)
        constraint = [
            {
                "id": 0,
                "name": "RHE_1",
                "expression": "@rho_acum_h0 * hydro_storage(0)",
            }
        ]
        # Between 0.5*500=250 (default) and 2.19*500=1095 (override).
        bound_lower = 600.0

        default_lhs = evaluate_lhs_novomodelo(constraint, tmp_path)
        override_lhs = evaluate_lhs_novomodelo(
            constraint, tmp_path, {0: {0: 2.19, 1: 2.5}}
        )

        default_value = float(
            default_lhs.filter(pl.col("stage_id") == 0)["lhs_value"][0]
        )
        override_value = float(
            override_lhs.filter(pl.col("stage_id") == 0)["lhs_value"][0]
        )

        assert default_value < bound_lower  # falsely appears violated
        assert override_value >= bound_lower  # LP-faithful: actually satisfied


# ---------------------------------------------------------------------------
# apply_vminop_useful_energy
# ---------------------------------------------------------------------------


class TestApplyVminopUsefulEnergy:
    def test_rewrites_vminop_rows_to_useful_energy(self, tmp_path: Path) -> None:
        novomodelo_case_dir = tmp_path / "case"
        novomodelo_output_dir = tmp_path / "case" / "output"
        _write_generic_parameters(novomodelo_case_dir)
        _write_hydros_json(novomodelo_case_dir)
        _write_sim_hydros_parquet(novomodelo_output_dir)

        constraints = load_generic_constraints(_NOVOMODELO_INPUT_DIR)
        gc_bounds = load_generic_constraint_bounds(_NOVOMODELO_INPUT_DIR)
        nw_hydro = read_medias_hydro(_NEWAVE_DIR)
        id_map = _id_map()
        gc_lhs_nw = evaluate_lhs_newave(
            constraints, nw_hydro, _EMPTY_LINE_MEANS, EntityAlignment(), id_map, 1
        )
        gc_lhs_cb = evaluate_lhs_novomodelo(constraints, novomodelo_output_dir)

        new_bounds, new_lhs_nw, new_lhs_cb = apply_vminop_useful_energy(
            constraints,
            gc_bounds,
            gc_lhs_nw,
            gc_lhs_cb,
            novomodelo_case_dir,
            novomodelo_output_dir,
            nw_hydro,
            id_map,
            nw_offset=1,
        )

        # bound_lower -= rho_acum(stage) * min_storage_hm3(hydro 0): 100-50=50.
        vminop_bounds = new_bounds.filter(pl.col("constraint_id") == _VMINOP_ID)
        assert sorted(vminop_bounds["bound_lower"].to_list()) == [50.0, 58.0]
        # Non-VminOP rows pass through with their original bound untouched.
        non_vminop_bounds = new_bounds.filter(pl.col("constraint_id") == _NON_VMINOP_ID)
        assert sorted(non_vminop_bounds["bound_upper"].to_list()) == [200.0, 210.0]

        # novomodelo LHS = rho_acum(stage) * (storage_final - min_storage).
        cb_vminop = {
            r["stage_id"]: r["lhs_value"]
            for r in new_lhs_cb.filter(pl.col("constraint_id") == _VMINOP_ID).iter_rows(
                named=True
            )
        }
        assert cb_vminop == {0: 0.5 * (500.0 - 100.0), 1: 0.52 * (480.0 - 100.0)}

        # NEWAVE LHS (newly added; the generic evaluator skips @rho_acum
        # rows) = rho_acum(stage) * VARMUH(hydro 10, stage).
        nw_vminop = {
            r["stage_id"]: r["lhs_value"]
            for r in new_lhs_nw.filter(pl.col("constraint_id") == _VMINOP_ID).iter_rows(
                named=True
            )
        }
        assert nw_vminop == {0: 0.5 * 40.0, 1: 0.52 * 42.0}

    def test_no_vminop_constraints_returns_inputs_unchanged(
        self, tmp_path: Path
    ) -> None:
        non_vminop = [{"id": 0, "expression": "hydro_generation(0)"}]
        bounds = pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "block_id": pl.Int32,
                "bound_lower": pl.Float64,
                "bound_upper": pl.Float64,
            }
        )
        lhs_nw = bounds.select("constraint_id", "stage_id")
        lhs_cb = bounds.select("constraint_id", "stage_id")

        out_bounds, out_nw, out_cb = apply_vminop_useful_energy(
            non_vminop, bounds, lhs_nw, lhs_cb, tmp_path, tmp_path, lhs_nw, _id_map(), 1
        )

        assert out_bounds is bounds
        assert out_nw is lhs_nw
        assert out_cb is lhs_cb

    def test_missing_rho_or_vmin_or_sim_degrades_with_warning(
        self, tmp_path: Path, caplog
    ) -> None:
        constraints = load_generic_constraints(_NOVOMODELO_INPUT_DIR)
        gc_bounds = load_generic_constraint_bounds(_NOVOMODELO_INPUT_DIR)
        nw_hydro = read_medias_hydro(_NEWAVE_DIR)
        id_map = _id_map()
        # No generic_parameters.json / hydros.json / simulation written under
        # tmp_path -> rho/vmin/sim are all missing.
        gc_lhs_nw = evaluate_lhs_newave(
            constraints, nw_hydro, _EMPTY_LINE_MEANS, EntityAlignment(), id_map, 1
        )
        gc_lhs_cb = evaluate_lhs_novomodelo(constraints, tmp_path)

        with caplog.at_level(logging.WARNING):
            out_bounds, out_nw, out_cb = apply_vminop_useful_energy(
                constraints,
                gc_bounds,
                gc_lhs_nw,
                gc_lhs_cb,
                tmp_path,
                tmp_path,
                nw_hydro,
                id_map,
                nw_offset=1,
            )

        assert out_bounds is gc_bounds
        assert out_nw is gc_lhs_nw
        assert out_cb is gc_lhs_cb
        assert "VminOP useful-energy rewrite skipped" in caplog.text
