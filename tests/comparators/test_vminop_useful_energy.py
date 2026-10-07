"""Tests for the VminOP useful-energy rewrite in the results comparator.

``apply_vminop_useful_energy`` re-expresses VminOP generic-constraint rows as
*useful* stored energy (MWmonth) so they compare like-for-like — all three on the
**linear** stored-energy convention the security-curve constraint binds on:

- novomodelo LHS  = Σ override ρ_acum(stage) · (storage_final − Vmin)
- the source model LHS = Σ override ρ_acum(stage) · VARMUH(plant, stage)   (MEDIAS-USIH)
- bound      = original stored-bound − dead-energy (Σ ρ_acum · Vmin)

The source model LHS is deliberately the *linear* per-plant ``VARMUH`` reconstruction,
not the per-REE ``EARMF`` (which the source model reports as the nonlinear ∫ρ dv
energy).

RE / AGRINT constraints (no ``@rho_acum``) must pass through untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from novomodelo_bridge.comparators.newave.constraints import apply_vminop_useful_energy
from novomodelo_bridge.newave.id_map import NewaveIdMap
from novomodelo_bridge.novomodelo.constraint_expr import scales_storage_by_rho_acum

_GC_SCHEMA = {
    "constraint_id": pl.Int32,
    "stage_id": pl.Int32,
    "lhs_value": pl.Float64,
}


def _build_case(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal Novomodelo case: one hydro (id 0), Vmin=100, ρ_acum=2.

    Storage trajectory: 600 hm³ at stage 0, 400 hm³ at stage 1.
    Returns ``(case_dir, output_dir)``.
    """
    case = tmp_path / "case"
    (case / "system").mkdir(parents=True)
    (case / "constraints").mkdir(parents=True)
    out = case / "output"
    sim = out / "simulation" / "hydros" / "scenario_id=0000"
    sim.mkdir(parents=True)

    (case / "constraints" / "generic_parameters.json").write_text(
        json.dumps(
            {
                "scalar_parameters": [
                    {
                        "id": 0,
                        "name": "rho_acum_h0",
                        "kind": "per_stage",
                        "values": [[0, 2.0], [1, 2.0]],
                    }
                ]
            }
        )
    )
    (case / "system" / "hydros.json").write_text(
        json.dumps(
            {
                "hydros": [
                    {
                        "id": 0,
                        "reservoir": {
                            "min_storage_hm3": 100.0,
                            "max_storage_hm3": 1000.0,
                        },
                    }
                ]
            }
        )
    )
    pq.write_table(
        pa.table(
            {
                "hydro_id": [0, 0],
                "stage_id": [0, 1],
                "block_id": [0, 0],
                "storage_final_hm3": [600.0, 400.0],
            }
        ),
        sim / "data.parquet",
    )
    return case, out


def _vminop_constraint() -> dict:
    return {
        "id": 0,
        "name": "VminOP_X",
        "expression": "@rho_acum_h0 * hydro_storage(0)",
        "description": "Minimum stored energy for REE 1 (X)",
    }


def _bounds() -> pl.DataFrame:
    # VminOP is always a lower-bound (``>=``) constraint: the F3 endpoint pair
    # carries the stored value in ``bound_lower``, ``bound_upper`` null.
    return pl.DataFrame(
        {
            "constraint_id": [0, 0],
            "stage_id": [0, 1],
            "block_id": [0, 0],
            "bound_lower": [1000.0, 800.0],
            "bound_upper": [None, None],
        },
        schema={
            "constraint_id": pl.Int32,
            "stage_id": pl.Int32,
            "block_id": pl.Int32,
            "bound_lower": pl.Float64,
            "bound_upper": pl.Float64,
        },
    )


def _nw_hydro() -> pl.DataFrame:
    # VARMUH (useful stored volume, hm³ above Vmin) for plant code 1 (→ novomodelo id 0) at
    # MEDIAS stages 9 (=stage 0) and 10 (=stage 1).  With ρ_acum=2 the linear the source
    # model LHS is 2·475=950 and 2·290=580.
    return pl.DataFrame(
        {
            "newave_code": [1, 1],
            "stage": [9, 10],
            "variable": ["VARMUH", "VARMUH"],
            "value": [475.0, 290.0],
        }
    )


def _id_map() -> NewaveIdMap:
    # The source model plant code 1 → novomodelo hydro id 0 (enumerate order).
    return NewaveIdMap(subsystem_ids=[], hydro_codes=[1], thermal_codes=[])


def _cell(df: pl.DataFrame, cid: int, stage: int, col: str) -> float:
    sub = df.filter((pl.col("constraint_id") == cid) & (pl.col("stage_id") == stage))
    return float(sub[col][0])


def test_scales_storage_by_rho_acum_partitions_by_rho_acum() -> None:
    vminop = _vminop_constraint()
    re_constraint = {
        "id": 1,
        "name": "RE_1",
        "expression": "hydro_generation(0) + hydro_generation(1)",
    }
    assert scales_storage_by_rho_acum(vminop) is True
    assert scales_storage_by_rho_acum(re_constraint) is False


def test_novomodelo_lhs_is_useful_energy(tmp_path: Path) -> None:
    case, out = _build_case(tmp_path)
    nw_empty = pl.DataFrame(schema=_GC_SCHEMA)
    _, _, cb = apply_vminop_useful_energy(
        [_vminop_constraint()],
        _bounds(),
        nw_empty,
        nw_empty,
        case,
        out,
        _nw_hydro(),
        _id_map(),
        9,
    )
    # ρ·(storage − Vmin): stage 0 → 2·(600−100)=1000; stage 1 → 2·(400−100)=600.
    assert _cell(cb, 0, 0, "lhs_value") == 1000.0
    assert _cell(cb, 0, 1, "lhs_value") == 600.0


def test_source_model_lhs_is_linear_varmuh_energy(tmp_path: Path) -> None:
    # The source model LHS = Σ ρ_acum(stage) · VARMUH = 2·475=950 and 2·290=580 — the
    # linear stored energy the curve binds on, NOT the per-REE EARMF.
    case, out = _build_case(tmp_path)
    nw_empty = pl.DataFrame(schema=_GC_SCHEMA)
    _, nw, _ = apply_vminop_useful_energy(
        [_vminop_constraint()],
        _bounds(),
        nw_empty,
        nw_empty,
        case,
        out,
        _nw_hydro(),
        _id_map(),
        9,
    )
    assert _cell(nw, 0, 0, "lhs_value") == 950.0
    assert _cell(nw, 0, 1, "lhs_value") == 580.0


def test_bound_has_dead_energy_removed(tmp_path: Path) -> None:
    case, out = _build_case(tmp_path)
    nw_empty = pl.DataFrame(schema=_GC_SCHEMA)
    bounds_out, _, _ = apply_vminop_useful_energy(
        [_vminop_constraint()],
        _bounds(),
        nw_empty,
        nw_empty,
        case,
        out,
        _nw_hydro(),
        _id_map(),
        9,
    )
    # dead = ρ·Vmin = 2·100 = 200; useful bound = stored bound − dead.
    assert _cell(bounds_out, 0, 0, "bound_lower") == 800.0  # 1000 − 200
    assert _cell(bounds_out, 0, 1, "bound_lower") == 600.0  # 800 − 200
    # VminOP is single-sided (>=); the upper endpoint stays null.
    sub = bounds_out.filter(pl.col("constraint_id") == 0)
    assert sub["bound_upper"].is_null().all()


def test_re_agrint_rows_pass_through_untouched(tmp_path: Path) -> None:
    case, out = _build_case(tmp_path)
    re_constraint = {
        "id": 7,
        "name": "RE_1",
        "expression": "hydro_generation(0)",
        "description": "Electric restriction",
    }
    # RE/AGRINT constraints are ceilings (``<=``): upper-only endpoint.
    bounds = pl.DataFrame(
        {
            "constraint_id": [7],
            "stage_id": [0],
            "block_id": [0],
            "bound_lower": [None],
            "bound_upper": [500.0],
        },
        schema={
            "constraint_id": pl.Int32,
            "stage_id": pl.Int32,
            "block_id": pl.Int32,
            "bound_lower": pl.Float64,
            "bound_upper": pl.Float64,
        },
    )
    cb_in = pl.DataFrame(
        {"constraint_id": [7], "stage_id": [0], "lhs_value": [123.0]},
        schema=_GC_SCHEMA,
    )
    nw_in = pl.DataFrame(schema=_GC_SCHEMA)
    bounds_out, nw_out, cb_out = apply_vminop_useful_energy(
        [re_constraint], bounds, nw_in, cb_in, case, out, _nw_hydro(), _id_map(), 9
    )
    # No VminOP constraint → everything returned unchanged.
    assert _cell(bounds_out, 7, 0, "bound_upper") == 500.0
    assert _cell(cb_out, 7, 0, "lhs_value") == 123.0
    assert nw_out.is_empty()


def test_missing_inputs_degrade_gracefully(tmp_path: Path) -> None:
    # No generic_parameters.json / hydros.json → original frames returned as-is.
    empty_case = tmp_path / "empty"
    (empty_case / "system").mkdir(parents=True)
    out = empty_case / "output"
    bounds = _bounds()
    cb_in = pl.DataFrame(
        {"constraint_id": [0], "stage_id": [0], "lhs_value": [42.0]},
        schema=_GC_SCHEMA,
    )
    nw_in = pl.DataFrame(schema=_GC_SCHEMA)
    bounds_out, nw_out, cb_out = apply_vminop_useful_energy(
        [_vminop_constraint()],
        bounds,
        nw_in,
        cb_in,
        empty_case,
        out,
        _nw_hydro(),
        _id_map(),
        9,
    )
    assert cb_out.equals(cb_in)
    assert nw_out.equals(nw_in)
    assert bounds_out.equals(bounds)
