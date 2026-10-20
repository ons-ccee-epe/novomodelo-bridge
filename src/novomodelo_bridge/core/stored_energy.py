"""Stored-energy generic-constraint terms, shared by both conversion tracks.

VminOP and RHE bound a REE's useful stored energy in MWh::

    Σ ±@coef * hydro_useful_volume_final(id)  >=  @pct * (Σ ±@emax)

``@coef`` (MWh/hm³) and ``@emax`` (MWh) are novomodelo ``computed`` parameters for
a reservoir whose whole cascade counts in its REE (:func:`cascade_share`). A
share the source model values otherwise names its own ``kind: "expression"``
parameters over the computed ones, so the constraint keeps one term per
reservoir and REE.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import NamedTuple


def rho_eq_mwh_name(hydro_id: int) -> str:
    """``@name`` of a hydro's integrated equivalent productivity, MWh/hm³."""
    return f"rho_eq_mwh_h{hydro_id}"


def rho_acum_mwh_name(hydro_id: int) -> str:
    """``@name`` of a hydro's integrated accumulated productivity, MWh/hm³."""
    return f"rho_acum_mwh_h{hydro_id}"


def earm_max_mwh_name(hydro_id: int) -> str:
    """``@name`` of a hydro's maximum stored energy over its cascade, MWh."""
    return f"earm_max_mwh_h{hydro_id}"


class StoredEnergyShare(NamedTuple):
    """One reservoir's contribution to one REE's stored energy.

    ``coefficient`` (MWh/hm³) weights the useful volume of ``hydro_id``;
    ``max_energy`` (MWh) is the share's stored energy when that reservoir is
    full. ``parameters`` maps each ``kind: "expression"`` parameter the share
    introduces to its body; it is empty for a whole-cascade share.
    """

    hydro_id: int
    coefficient: str
    max_energy: str
    parameters: dict[str, str]


def cascade_share(hydro_id: int) -> StoredEnergyShare:
    """The share of a reservoir whose whole cascade counts in its own REE."""
    return StoredEnergyShare(
        hydro_id, rho_acum_mwh_name(hydro_id), earm_max_mwh_name(hydro_id), {}
    )


def _signed_sum(terms: Sequence[tuple[float, str]]) -> str:
    parts: list[str] = []
    for coefficient, token in terms:
        magnitude = abs(coefficient)
        body = token if magnitude == 1.0 else f"{magnitude!r} * {token}"
        sign = "-" if coefficient < 0.0 else "+"
        parts.append(body if not parts and sign == "+" else f"{sign} {body}")
    return " ".join(parts)


def stored_energy_lhs(shares: Sequence[tuple[float, StoredEnergyShare]]) -> str:
    """``Σ ±@coef * hydro_useful_volume_final(id)`` over signed *shares*."""
    return _signed_sum(
        [
            (sign, f"@{s.coefficient} * hydro_useful_volume_final({s.hydro_id})")
            for sign, s in shares
        ]
    )


def stored_energy_floor(
    shares: Sequence[tuple[float, StoredEnergyShare]], fraction_name: str
) -> str:
    """``lhs >= @fraction * (Σ ±@emax)``: at least a fraction of the maximum."""
    maximum = _signed_sum([(sign, f"@{s.max_energy}") for sign, s in shares])
    if len(shares) > 1:
        maximum = f"({maximum})"
    return f"{stored_energy_lhs(shares)} >= @{fraction_name} * {maximum}"


def share_parameters(shares: Iterable[StoredEnergyShare]) -> dict[str, str]:
    """Every expression parameter *shares* introduce, in first-seen order."""
    out: dict[str, str] = {}
    for share in shares:
        out.update(share.parameters)
    return out
