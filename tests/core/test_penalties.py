"""Tier-1 tests for ``hydro_penalty_costs`` and its ``flow_micro_costs`` seam."""

from __future__ import annotations

from novomodelo_bridge.core.penalties import FlowMicroCosts, hydro_penalty_costs

_MICRO_KEYS = ("spillage_cost", "turbined_cost", "diversion_cost")


def _costs(flow_micro_costs: FlowMicroCosts | None = None) -> dict[str, float]:
    return hydro_penalty_costs(
        rho_avg=0.634,
        rho_max_acum=0.9,
        penalid_costs={},
        max_deficit_cost=8291.25,
        flow_micro_costs=flow_micro_costs,
    )


def test_omitted_flow_micro_costs_keep_the_rho_scaled_values() -> None:
    omitted = hydro_penalty_costs(
        rho_avg=0.634, rho_max_acum=0.9, penalid_costs={}, max_deficit_cost=8291.25
    )

    assert omitted == _costs(None)
    assert omitted["spillage_cost"] == 0.000300 * 0.634
    assert omitted["turbined_cost"] == 0.000333 * 0.634
    assert omitted["diversion_cost"] == 0.000300 * 0.634


def test_flow_micro_costs_replace_only_the_three_micro_keys() -> None:
    default = _costs()
    flat = _costs(FlowMicroCosts(1.0, 2.0, 3.0))

    assert (flat["spillage_cost"], flat["turbined_cost"], flat["diversion_cost"]) == (
        1.0,
        2.0,
        3.0,
    )
    assert {k: v for k, v in flat.items() if k not in _MICRO_KEYS} == {
        k: v for k, v in default.items() if k not in _MICRO_KEYS
    }


def test_hydro_block_key_order_is_unchanged() -> None:
    expected = [
        "spillage_cost",
        "turbined_cost",
        "diversion_cost",
        "storage_violation_below_cost",
        "filling_target_violation_cost",
        "turbined_violation_below_cost",
        "outflow_violation_below_cost",
        "outflow_violation_above_cost",
        "generation_violation_below_cost",
        "evaporation_violation_cost",
        "water_withdrawal_violation_cost",
        "inflow_nonnegativity_cost",
    ]

    assert list(_costs()) == expected
    assert list(_costs(FlowMicroCosts(1.0, 2.0, 3.0))) == expected
