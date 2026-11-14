"""Tests for ``novomodelo_bridge.novomodelo.constraint_expr``: the resolved generic-constraint
view built from novomodelo's echo, and the LHS evaluated against simulation output.

Tier-1: synthetic echo frames and in-memory simulation frames only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from novomodelo_bridge.novomodelo.constraint_expr import (
    BLOCK_HOURS_SCHEMA,
    BOUNDS_SCHEMA,
    evaluate_lhs,
    resolve_generic_constraints,
    stage_weighted_bounds,
    stage_weighted_lhs,
)
from novomodelo_bridge.novomodelo.readers import read_novomodelo_generic_constraint_echo
from tests.conftest import EchoTerm, write_generic_constraint_echo


def _echo(tmp_path: Path, terms: list[EchoTerm]) -> pl.DataFrame:
    write_generic_constraint_echo(tmp_path, terms)
    echo = read_novomodelo_generic_constraint_echo(tmp_path)
    assert echo is not None
    return echo


def _vminop(cid: int) -> dict:
    return {
        "id": cid,
        "name": "VminOP_SUDESTE",
        "description": "Minimum stored energy for REE 1 (SUDESTE)",
        "expression": "...",
        "slack": {"enabled": True, "penalty": 1488.0, "penalty_basis": "per_unit"},
    }


_HYDROS = pl.LazyFrame(
    {
        "scenario_id": [0, 0, 0, 0],
        "stage_id": [0, 0, 1, 1],
        "block_id": [0, 1, 0, 1],
        "hydro_id": [0, 0, 0, 0],
        "storage_final_hm3": [500.0, 500.0, 480.0, 480.0],
        "generation_mw": [95.0, 85.0, 97.0, 87.0],
    }
)


_HOURS = pl.DataFrame(
    {
        "stage_id": [0, 0, 1, 1],
        "block_id": [0, 1, 0, 1],
        "hours": [100.0, 300.0, 100.0, 300.0],
    },
    schema=BLOCK_HOURS_SCHEMA,
)


def _lhs(resolved, **frames: pl.LazyFrame):
    return evaluate_lhs(resolved, {"hydros": _HYDROS, **frames}, _HOURS)


class TestResolveGenericConstraints:
    def test_one_constraint_carries_each_stage_coefficient_and_bound(
        self, tmp_path: Path
    ) -> None:
        echo = _echo(
            tmp_path,
            [
                (
                    3,
                    "VminOP_SUDESTE",
                    0,
                    None,
                    "hydro_useful_volume_final(id=0, block=all)",
                    2.0,
                    900.0,
                    None,
                ),
                (
                    3,
                    "VminOP_SUDESTE",
                    1,
                    None,
                    "hydro_useful_volume_final(id=0, block=all)",
                    2.5,
                    950.0,
                    None,
                ),
            ],
        )

        resolved = resolve_generic_constraints([_vminop(3)], echo, {0: 100.0})

        assert resolved.constraints == [_vminop(3)]
        assert resolved.terms["coefficient"].to_list() == [2.0, 2.5]
        assert resolved.bounds["bound_lower"].to_list() == [
            900.0 - 2.0 * 100.0,
            950.0 - 2.5 * 100.0,
        ]

    def test_useful_volume_dead_term_leaves_the_bound(self, tmp_path: Path) -> None:
        echo = _echo(
            tmp_path,
            [
                (
                    0,
                    "VminOP_SUDESTE",
                    0,
                    None,
                    "hydro_useful_volume_final(id=0, block=all)",
                    2.0,
                    900.0,
                    None,
                ),
                (
                    0,
                    "VminOP_SUDESTE",
                    0,
                    None,
                    "hydro_useful_volume_final(id=1, block=all)",
                    3.0,
                    900.0,
                    None,
                ),
            ],
        )

        resolved = resolve_generic_constraints([_vminop(0)], echo, {0: 100.0, 1: 50.0})

        # novomodelo's bound carries 2·100 + 3·50 of dead-volume energy.
        assert resolved.bounds["bound_lower"].to_list() == [900.0 - 350.0]

    def test_absolute_storage_bound_is_kept(self, tmp_path: Path) -> None:
        echo = _echo(
            tmp_path,
            [
                (
                    0,
                    "RHE_7",
                    0,
                    None,
                    "hydro_storage_final(id=0, block=all)",
                    2.0,
                    900.0,
                    None,
                )
            ],
        )

        resolved = resolve_generic_constraints(
            [{"id": 0, "name": "RHE_7"}], echo, {0: 100.0}
        )

        assert resolved.bounds["bound_lower"].to_list() == [900.0]

    def test_no_echo_keeps_constraints_without_terms(self) -> None:
        resolved = resolve_generic_constraints([_vminop(0)], None, {})

        assert [c["name"] for c in resolved.constraints] == ["VminOP_SUDESTE"]
        assert resolved.terms.is_empty()
        assert resolved.bounds.is_empty()


class TestEvaluateLhs:
    def test_useful_volume_subtracts_min_storage(self, tmp_path: Path) -> None:
        echo = _echo(
            tmp_path,
            [
                (
                    0,
                    "VminOP_SUDESTE",
                    0,
                    None,
                    "hydro_useful_volume_final(id=0, block=all)",
                    2.0,
                    0.0,
                    None,
                ),
                (
                    0,
                    "VminOP_SUDESTE",
                    1,
                    None,
                    "hydro_useful_volume_final(id=0, block=all)",
                    2.5,
                    0.0,
                    None,
                ),
            ],
        )
        resolved = resolve_generic_constraints([_vminop(0)], echo, {0: 100.0})

        lhs = _lhs(resolved)

        assert lhs[["stage_id", "block_id", "lhs_value"]].values.tolist() == [
            [0, 0, 2.0 * 400.0],
            [1, 0, 2.5 * 380.0],
        ]
        assert set(lhs["constraint_id"]) == {0}

    def test_per_block_terms_broadcast_storage(self, tmp_path: Path) -> None:
        echo = _echo(
            tmp_path,
            [
                (
                    0,
                    "RE_1",
                    0,
                    0,
                    "hydro_generation(id=0, bus=all, block=0)",
                    1.0,
                    None,
                    200.0,
                ),
                (0, "RE_1", 0, 0, "hydro_storage(id=0)", 0.5, None, 200.0),
                (
                    0,
                    "RE_1",
                    0,
                    1,
                    "hydro_generation(id=0, bus=all, block=1)",
                    1.0,
                    None,
                    200.0,
                ),
                (0, "RE_1", 0, 1, "hydro_storage(id=0)", 0.5, None, 200.0),
            ],
        )
        resolved = resolve_generic_constraints(
            [{"id": 0, "name": "RE_1"}], echo, {0: 100.0}
        )

        lhs = _lhs(resolved)

        assert lhs[["block_id", "lhs_value"]].values.tolist() == [
            [0, 95.0 + 0.5 * 500.0],
            [1, 85.0 + 0.5 * 500.0],
        ]

    def test_constraint_with_an_unsupported_term_is_skipped(
        self, tmp_path: Path
    ) -> None:
        echo = _echo(
            tmp_path,
            [
                (0, "RE_1", 0, 0, "bus_deficit(id=0, block=0)", 1.0, None, 10.0),
                (
                    0,
                    "RE_1",
                    0,
                    0,
                    "hydro_generation(id=0, bus=all, block=0)",
                    1.0,
                    None,
                    10.0,
                ),
            ],
        )
        resolved = resolve_generic_constraints([{"id": 0, "name": "RE_1"}], echo, {})

        assert _lhs(resolved).empty

    def test_constraint_without_its_simulation_output_is_skipped(
        self, tmp_path: Path
    ) -> None:
        echo = _echo(
            tmp_path,
            [(0, "RE_1", 0, 0, "thermal_generation(id=4, block=0)", 1.0, None, 10.0)],
        )
        resolved = resolve_generic_constraints([{"id": 0, "name": "RE_1"}], echo, {})

        assert _lhs(resolved, thermals=pl.LazyFrame()).empty

    def test_row_with_a_term_missing_from_the_output_is_not_a_partial_sum(
        self, tmp_path: Path
    ) -> None:
        echo = _echo(
            tmp_path,
            [
                (
                    0,
                    "RE_1",
                    0,
                    0,
                    "hydro_generation(id=0, bus=all, block=0)",
                    1.0,
                    None,
                    10.0,
                ),
                (
                    0,
                    "RE_1",
                    0,
                    0,
                    "hydro_generation(id=9, bus=all, block=0)",
                    1.0,
                    None,
                    10.0,
                ),
            ],
        )
        resolved = resolve_generic_constraints([{"id": 0, "name": "RE_1"}], echo, {})

        assert _lhs(resolved).empty

    def test_thermal_and_flow_terms_read_their_block(self, tmp_path: Path) -> None:
        echo = _echo(
            tmp_path,
            [
                (0, "RE_1", 0, 1, "thermal_generation(id=4, block=1)", 2.0, None, 1.0),
                (0, "RE_1", 0, 1, "hydro_outflow(id=0, block=1)", 1.0, None, 1.0),
                (0, "RE_1", 0, 1, "hydro_diversion(id=0, block=1)", -1.0, None, 1.0),
            ],
        )
        resolved = resolve_generic_constraints([{"id": 0, "name": "RE_1"}], echo, {})
        hydros = _HYDROS.with_columns(
            pl.Series("outflow_m3s", [10.0, 20.0, 30.0, 40.0]),
            pl.Series("diverted_outflow_m3s", [1.0, 2.0, 3.0, 4.0]),
        )
        thermals = pl.LazyFrame(
            {
                "scenario_id": [0, 0],
                "stage_id": [0, 0],
                "block_id": [0, 1],
                "thermal_id": [4, 4],
                "generation_mw": [50.0, 60.0],
            }
        )

        lhs = evaluate_lhs(resolved, {"hydros": hydros, "thermals": thermals}, _HOURS)

        assert lhs[["block_id", "lhs_value"]].values.tolist() == [
            [1, 2.0 * 60.0 + 20.0 - 2.0]
        ]

    def test_pinned_term_counts_on_its_rows_block(self, tmp_path: Path) -> None:
        echo = _echo(
            tmp_path,
            [
                (
                    0,
                    "RE_1",
                    0,
                    0,
                    "hydro_generation(id=0, bus=all, block=1)",
                    1.0,
                    None,
                    10.0,
                ),
                (
                    0,
                    "RE_1",
                    0,
                    0,
                    "hydro_generation(id=0, bus=all, block=0)",
                    -1.0,
                    None,
                    10.0,
                ),
            ],
        )
        resolved = resolve_generic_constraints([{"id": 0, "name": "RE_1"}], echo, {})

        lhs = _lhs(resolved)

        # Block 1's generation minus block 0's, on block 0's row.
        assert lhs[["block_id", "lhs_value"]].values.tolist() == [[0, 85.0 - 95.0]]

    def test_stage_volume_weights_each_block_by_its_hours(self, tmp_path: Path) -> None:
        echo = _echo(
            tmp_path,
            [(0, "RHV_1", 0, None, "hydro_outflow_volume(id=0)", 1.0, None, 1.0)],
        )
        resolved = resolve_generic_constraints([{"id": 0, "name": "RHV_1"}], echo, {})
        hydros = _HYDROS.with_columns(
            pl.Series("outflow_m3s", [10.0, 20.0, 30.0, 40.0])
        )

        lhs = evaluate_lhs(resolved, {"hydros": hydros}, _HOURS)

        assert lhs[["stage_id", "block_id"]].values.tolist() == [[0, 0]]
        assert lhs["lhs_value"].iloc[0] == pytest.approx(
            (10.0 * 100.0 + 20.0 * 300.0) * 3600.0 / 1e6
        )

    def test_bus_cell_term_reads_its_cell(self, tmp_path: Path) -> None:
        echo = _echo(
            tmp_path,
            [
                (
                    0,
                    "RE_1",
                    0,
                    0,
                    "hydro_generation(id=0, bus=2, block=0)",
                    1.0,
                    None,
                    10.0,
                )
            ],
        )
        resolved = resolve_generic_constraints([{"id": 0, "name": "RE_1"}], echo, {})
        cells = pl.LazyFrame(
            {
                "scenario_id": [0, 0],
                "stage_id": [0, 0],
                "block_id": [0, 0],
                "hydro_id": [0, 0],
                "bus_id": [1, 2],
                "turbined_m3s": [1.0, 1.0],
                "generation_mw": [60.0, 35.0],
            }
        )

        lhs = _lhs(resolved, hydro_bus_generation=cells)

        assert lhs["lhs_value"].tolist() == [35.0]


class TestStageWeighting:
    def test_lhs_weights_blocks_by_their_hours(self) -> None:
        lhs = pd.DataFrame(
            {
                "constraint_id": [0, 0],
                "scenario_id": [0, 0],
                "stage_id": [0, 0],
                "block_id": [0, 1],
                "lhs_value": [10.0, 30.0],
            }
        )

        out = stage_weighted_lhs(lhs, _HOURS)

        assert out["lhs_value"].to_list() == [(10.0 * 100.0 + 30.0 * 300.0) / 400.0]

    def test_bounds_weight_blocks_and_keep_open_endpoints_open(self) -> None:
        bounds = pl.DataFrame(
            {
                "constraint_id": [0, 0, 1],
                "stage_id": [0, 0, 1],
                "block_id": [0, 1, None],
                "bound_lower": [None, 5.0, 2.0],
                "bound_upper": [40.0, 20.0, None],
            },
            schema=BOUNDS_SCHEMA,
        )

        out = stage_weighted_bounds(bounds, _HOURS)

        assert out.to_dicts() == [
            {
                "constraint_id": 0,
                "stage_id": 0,
                "block_id": None,
                "bound_lower": None,
                "bound_upper": (40.0 * 100.0 + 20.0 * 300.0) / 400.0,
            },
            {
                "constraint_id": 1,
                "stage_id": 1,
                "block_id": None,
                "bound_lower": 2.0,
                "bound_upper": None,
            },
        ]
