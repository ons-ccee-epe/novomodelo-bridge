"""How closely Novomodelo agrees with the source model on a compared variable.

The ``compare`` console table, its ``--json`` status and the HTML report's
per-plant tables all judge agreement the same way. The *gap* of a set of
paired values is ``Σ|Novomodelo − reference| / Σ|reference|``: the mean
absolute difference relative to the mean reference magnitude, so a point
whose reference is near zero cannot inflate it. A gap up to the compare
tolerance is :attr:`Band.CLOSE`, up to :data:`FAR_FACTOR` times it
:attr:`Band.MODERATE`, and beyond that :attr:`Band.FAR`.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

#: The ``--tolerance`` a compare uses when none is configured: the close band.
DEFAULT_TOLERANCE: float = 0.05

#: A gap beyond this multiple of the tolerance is far.
FAR_FACTOR: float = 4.0

#: Below this sum the reference counts as zero.
_ZERO_REFERENCE: float = 1e-9

#: Below this mean absolute difference per point, so does the difference.
_ZERO_DIFFERENCE: float = 1e-6


class Band(StrEnum):
    """How a variable's gap compares with the tolerance."""

    CLOSE = "close"
    MODERATE = "moderate"
    FAR = "far"


def gap(pairs: Iterable[tuple[float, float]], floor: float = 0.0) -> float | None:
    """``Σ|novomodelo − reference| / max(Σ|reference|, floor)`` over
    ``(reference, novomodelo)`` pairs.

    A *floor* judges a small entity against a typical one's size. ``0.0``
    when both sides are zero everywhere; ``None`` when only the reference
    and the floor are, since no relative gap exists then.
    """
    count = 0
    total_reference = 0.0
    total_difference = 0.0
    for reference, novomodelo in pairs:
        count += 1
        total_reference += abs(reference)
        total_difference += abs(novomodelo - reference)
    denominator = max(total_reference, floor)
    if denominator > _ZERO_REFERENCE:
        return total_difference / denominator
    return 0.0 if total_difference <= _ZERO_DIFFERENCE * count else None


def band(value: float | None, tolerance: float) -> Band:
    """The band of a :func:`gap` under *tolerance*; an undefined gap is far."""
    if value is None or value > FAR_FACTOR * tolerance:
        return Band.FAR
    return Band.CLOSE if value <= tolerance else Band.MODERATE


def format_gap(value: float | None) -> str:
    """A gap as a short percentage: ``0.6%``, ``5%``, ``46%``, ``>999%``; ``∞``
    when undefined."""
    if value is None:
        return "∞"
    percent = value * 100.0
    if percent == 0.0:
        return "0%"
    if percent < 0.1:
        return "<0.1%"
    if percent < 9.95:
        return f"{percent:.1f}".removesuffix(".0") + "%"
    if percent < 999.5:
        return f"{percent:.0f}%"
    return ">999%"


#: Entity types in the order the console groups them, with their headings.
ENTITY_GROUPS: dict[str, str] = {
    "bus": "Buses",
    "line": "Lines",
    "ree": "Energy reservoirs (REE)",
    "system_spillage": "System spillage",
    "hydro": "Hydro plants",
    "thermal": "Thermal plants",
}

#: Display label, with unit, of each compared variable.
VARIABLE_LABELS: dict[str, str] = {
    "spot_price": "Spot price (R$/MWh)",
    "deficit_mw": "Deficit (MW)",
    "net_flow_mw": "Net flow (MW)",
    "earm_final_mwmes": "Stored energy, end of stage (MWmês)",
    "ena_mwmes": "Natural inflow energy (MWmês)",
    "spill_energy_reservoir_mw": "Spilled energy, reservoirs (MW)",
    "spill_energy_rorov_mw": "Spilled energy, run-of-river (MW)",
    "spill_energy_total_mw": "Spilled energy, total (MW)",
    "storage_final_hm3": "Storage (hm³)",
    "generation_mw": "Generation (MW)",
    "turbined_m3s": "Turbined flow (m³/s)",
    "productivity_mw_per_m3s": "Productivity (MW per m³/s)",
    "spillage_m3s": "Spillage (m³/s)",
    "outflow_m3s": "Outflow (m³/s)",
    "inflow_m3s": "Incremental inflow (m³/s)",
    "total_inflow_m3s": "Total inflow (m³/s)",
    "evaporation_m3s": "Evaporation (m³/s)",
    "withdrawal_m3s": "Water withdrawal (m³/s)",
    "water_value_per_hm3": "Water value (R$/hm³)",
    "lower_bound": "Lower bound (R$)",
    "upper_bound_mean": "Upper bound, mean (R$)",
}


def variable_label(variable: str) -> str:
    """The display label of *variable*, or the name itself when it has none."""
    return VARIABLE_LABELS.get(variable, variable)


def is_policy_row(entity_type: str, variable: str) -> bool:
    """Whether a summary row describes the trained policy rather than the
    simulated operation: the convergence bounds and the water values.

    These rows stay in the artifacts and the HTML report but are left out of
    the console table, its headline and the ``--json`` status.
    """
    return entity_type == "convergence" or variable == "water_value_per_hm3"
