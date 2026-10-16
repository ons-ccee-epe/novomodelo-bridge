"""Tests for ``novomodelo_bridge.novomodelo.constraint_expr``: the resolved generic-constraint
view built from novomodelo's echo, and the LHS evaluated against simulation output.

Tier-1: synthetic echo frames and in-memory simulation frames only.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.novomodelo.constraint_expr import (
    evaluate_lhs,
    resolve_generic_constraints,
)
from novomodelo_bridge.novomodelo.readers import read_novomodelo_generic_constraint_echo
from tests.conftest import EchoTerm, write_generic_constraint_echo


def _echo(tmp_path: Path, terms: list[EchoTerm]) -> pl.DataFrame:
    write_generic_constraint_echo(tmp_path, terms)
    echo = read_novomodelo_generic_constraint_echo(tmp_path)
    assert echo is not None
    return echo


def _vminop(stage: int, cid: int) -> dict:
    return {
        "id": cid,
        "name": f"VminOP_SUDESTE_s{stage}",
        "description": f"Minimum stored energy for REE 1 (SUDESTE), stage {stage}",
        "expression": "...",
        "slack": {"enabled": True, "penalty": 1488.0 / [744.0, 696.0][stage]},
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


class TestResolveGenericConstraints:
    def test_stage_split_family_is_merged_with_its_penalty_in_r_per_mwh(
        self, tmp_path: Path
    ) -> None:
        echo = _echo(
            tmp_path,
            [
                (
                    3,
                    "VminOP_SUDESTE_s0",
                    0,
                    None,
                    "hydro_useful_volume_final(id=0, block=all)",
                    2.0,
                    900.0,
                    None,
                ),
                (
                    4,
                    "VminOP_SUDESTE_s1",
                    1,
                    None,
                    "hydro_useful_volume_final(id=0, block=all)",
                    2.5,
                    950.0,
                    None,
                ),
            ],
        )

        resolved = resolve_generic_constraints(
            [_vminop(0, 3), _vminop(1, 4)], echo, {0: 100.0}, {0: 744.0, 1: 696.0}
        )

        [shown] = resolved.constraints
        assert (shown["id"], shown["name"]) == (3, "VminOP_SUDESTE")
        assert shown["description"] == "Minimum stored energy for REE 1 (SUDESTE)"
        assert shown["slack"]["penalty"] == pytest.approx(1488.0)
        assert resolved.display_ids == {3: 3, 4: 3}
        assert set(resolved.terms["constraint_id"].to_list()) == {3}

    def test_useful_volume_dead_term_leaves_the_bound(self, tmp_path: Path) -> None:
        echo = _echo(
            tmp_path,
            [
                (
                    0,
                    "VminOP_SUDESTE_s0",
                    0,
                    None,
                    "hydro_useful_volume_final(id=0, block=all)",
                    2.0,
                    900.0,
                    None,
                ),
                (
                    0,
                    "VminOP_SUDESTE_s0",
                    0,
                    None,
                    "hydro_useful_volume_final(id=1, block=all)",
                    3.0,
                    900.0,
                    None,
                ),
            ],
        )

        resolved = resolve_generic_constraints(
            [_vminop(0, 0)], echo, {0: 100.0, 1: 50.0}, {}
        )

        # novomodelo's bound carries 2·100 + 3·50 of dead-volume energy.
        assert resolved.bounds["bound_lower"].to_list() == [900.0 - 350.0]

    def test_absolute_storage_bound_is_kept(self, tmp_path: Path) -> None:
        echo = _echo(
            tmp_path,
            [
                (
                    0,
                    "RHE_7_s0",
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
            [{"id": 0, "name": "RHE_7_s0", "slack": {"penalty": 1.0}}],
            echo,
            {0: 100.0},
            {0: 168.0},
        )

        assert resolved.constraints[0]["name"] == "RHE_7"
        assert resolved.bounds["bound_lower"].to_list() == [900.0]

    def test_no_echo_keeps_constraints_without_terms(self) -> None:
        resolved = resolve_generic_constraints([_vminop(0, 0)], None, {}, {})

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
                    "VminOP_SUDESTE_s0",
                    0,
                    None,
                    "hydro_useful_volume_final(id=0, block=all)",
                    2.0,
                    0.0,
                    None,
                ),
                (
                    1,
                    "VminOP_SUDESTE_s1",
                    1,
                    None,
                    "hydro_useful_volume_final(id=0, block=all)",
                    2.5,
                    0.0,
                    None,
                ),
            ],
        )
        resolved = resolve_generic_constraints(
            [_vminop(0, 0), _vminop(1, 1)], echo, {0: 100.0}, {}
        )

        lhs = evaluate_lhs(resolved, _HYDROS, pl.LazyFrame())

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
            [{"id": 0, "name": "RE_1"}], echo, {0: 100.0}, {}
        )

        lhs = evaluate_lhs(resolved, _HYDROS, pl.LazyFrame())

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
                (
                    0,
                    "HQ_1",
                    0,
                    0,
                    "hydro_turbined(id=0, bus=all, block=0)",
                    1.0,
                    None,
                    10.0,
                ),
                (
                    0,
                    "HQ_1",
                    0,
                    0,
                    "hydro_generation(id=0, bus=all, block=0)",
                    1.0,
                    None,
                    10.0,
                ),
            ],
        )
        resolved = resolve_generic_constraints(
            [{"id": 0, "name": "HQ_1"}], echo, {}, {}
        )

        assert evaluate_lhs(resolved, _HYDROS, pl.LazyFrame()).empty
