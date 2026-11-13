"""DECOMP micro-penalty defaults held in source units and their conversion to Novomodelo units."""

from __future__ import annotations

from typing import TYPE_CHECKING

from novomodelo_bridge.core.penalties import FlowMicroCosts
from novomodelo_bridge.core.units import HM3_PER_M3S_HOUR

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

SPILLAGE_COST_PER_HM3 = 0.1  # R$/hm³
TURBINED_COST_PER_HM3 = 0.11  # R$/hm³
DIVERSION_COST_PER_HM3 = SPILLAGE_COST_PER_HM3  # R$/hm³
EXCHANGE_COST = 1e-4  # R$/MWh
FICTITIOUS_EXCHANGE_FACTOR = 0.5


def flow_micro_costs() -> FlowMicroCosts:
    """The per-hm³ costs (R$/hm³) as Novomodelo's R$/(m³/s·h) flow micro-penalties."""
    return FlowMicroCosts(
        spillage_cost=SPILLAGE_COST_PER_HM3 * HM3_PER_M3S_HOUR,
        turbined_cost=TURBINED_COST_PER_HM3 * HM3_PER_M3S_HOUR,
        diversion_cost=DIVERSION_COST_PER_HM3 * HM3_PER_M3S_HOUR,
    )


def with_fictitious_exchange_cost(
    lines_doc: dict,
    fictitious_bus_ids: AbstractSet[int],
) -> dict:
    """Set half of ``EXCHANGE_COST`` as ``exchange_cost`` on each line touching a fictitious bus.

    The source model halves the interchange penalty on those links so a
    real -> fictitious -> real route costs the same as one direct real link.
    A line with both ends fictitious still carries half, not a quarter; other
    lines carry no per-line key and fall back to ``penalties.json``.
    """
    return {
        **lines_doc,
        "lines": [
            {**line, "exchange_cost": EXCHANGE_COST * FICTITIOUS_EXCHANGE_FACTOR}
            if line["source_bus_id"] in fictitious_bus_ids
            or line["target_bus_id"] in fictitious_bus_ids
            else line
            for line in lines_doc["lines"]
        ],
    }
