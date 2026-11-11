"""Tier-1 tests for the RHV slack price and its tie to ``penalties.json``."""

from __future__ import annotations

import pytest

from novomodelo_bridge.core.penalties import storage_violation_cost
from novomodelo_bridge.decomp.converters.config import (
    convert_penalties,
    rhv_slack_penalty,
)


@pytest.mark.parametrize("productivities", [[0.5, 2.0, 1.0], [0.0, 0.8, 0.8]])
def test_rhv_slack_penalty_matches_penalties_json_storage_violation_cost(
    productivities: list[float],
) -> None:
    penalties = convert_penalties(3000.0, productivities)

    assert (
        rhv_slack_penalty(3000.0, productivities)
        == penalties["hydro"]["storage_violation_below_cost"]
    )


def test_rhv_slack_penalty_empty_productivities_uses_unit_rho_max() -> None:
    assert rhv_slack_penalty(100.0, []) == storage_violation_cost(
        max_deficit_cost=100.0, rho_max_acum=1.0
    )
