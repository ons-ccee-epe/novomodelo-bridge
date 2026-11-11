"""Tier-1 tests for the storage-floor violation price and its use in the hydro penalty block."""

from __future__ import annotations

import pytest

from novomodelo_bridge.core.penalties import hydro_penalty_costs, storage_violation_cost
from novomodelo_bridge.core.units import HM3_TO_MWH_PER_RHO


def test_storage_violation_cost_known_inputs_returns_deficit_tier_energy_equivalent() -> (
    None
):
    cost = storage_violation_cost(max_deficit_cost=100.0, rho_max_acum=2.0)

    assert cost == pytest.approx(10.0 * 100.0 * 2.0 * HM3_TO_MWH_PER_RHO)


@pytest.mark.parametrize(
    ("rho_avg", "rho_max_acum", "max_deficit_cost"),
    [(1.0, 1.0, 100.0), (0.8, 2.5, 3000.0)],
)
def test_hydro_penalty_costs_without_volmin_uses_storage_violation_cost(
    rho_avg: float, rho_max_acum: float, max_deficit_cost: float
) -> None:
    costs = hydro_penalty_costs(
        rho_avg=rho_avg,
        rho_max_acum=rho_max_acum,
        penalid_costs={},
        max_deficit_cost=max_deficit_cost,
    )

    assert costs["storage_violation_below_cost"] == storage_violation_cost(
        max_deficit_cost=max_deficit_cost, rho_max_acum=rho_max_acum
    )


def test_hydro_penalty_costs_with_volmin_keeps_penalid_override() -> None:
    costs = hydro_penalty_costs(
        rho_avg=0.8,
        rho_max_acum=2.5,
        penalid_costs={"VOLMIN": 50.0},
        max_deficit_cost=3000.0,
    )

    assert costs["storage_violation_below_cost"] == 50.0 * 2.5 * HM3_TO_MWH_PER_RHO
