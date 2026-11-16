"""Tier-1 tests for the RHV slack price and its tie to ``penalties.json``."""

from __future__ import annotations

import pytest

from novomodelo_bridge.core.penalties import storage_violation_cost
from novomodelo_bridge.core.units import HM3_TO_MWH_PER_RHO
from novomodelo_bridge.decomp.converters.config import (
    convert_penalties,
    rhv_slack_penalty,
    with_outflow_volume_violation_cost,
)

_OUTFLOW_VOLUME_KEYS = (
    "outflow_volume_violation_below_cost",
    "outflow_volume_violation_above_cost",
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


@pytest.mark.parametrize(
    ("deficit_cost", "productivities"),
    [(3000.0, [0.5, 2.0, 1.0]), (1000.0, [0.0, 0.8, 0.8]), (100.0, [])],
)
def test_inflow_slack_price_matches_the_rhv_slack_price_plus_one_m3s_hour(
    deficit_cost: float, productivities: list[float]
) -> None:
    """Inflow price in R$/(m³/s)·h, scaled to R$/hm³ by ``HM3_TO_MWH_PER_RHO``."""
    inflow = convert_penalties(deficit_cost, productivities)["hydro"][
        "inflow_nonnegativity_cost"
    ]
    assert inflow * HM3_TO_MWH_PER_RHO == pytest.approx(
        rhv_slack_penalty(deficit_cost, productivities) + HM3_TO_MWH_PER_RHO
    )


def test_with_outflow_volume_violation_cost_prices_both_keys_without_mutating() -> None:
    doc = convert_penalties(1000.0, [1.0, 2.0])

    priced = with_outflow_volume_violation_cost(doc, 42.0)

    assert {key: priced["hydro"][key] for key in _OUTFLOW_VOLUME_KEYS} == {
        key: 42.0 for key in _OUTFLOW_VOLUME_KEYS
    }
    assert {
        key: value
        for key, value in priced["hydro"].items()
        if key not in _OUTFLOW_VOLUME_KEYS
    } == doc["hydro"]
    assert {key: value for key, value in priced.items() if key != "hydro"} == {
        key: value for key, value in doc.items() if key != "hydro"
    }
    assert not set(_OUTFLOW_VOLUME_KEYS) & doc["hydro"].keys()
