"""Stored-energy shares of the source model's ``HE`` (RHE) constraints.

The source model values a reservoir's useful volume along its *energy chain*:
from each plant to its ``AC JUSENA`` plant where one is declared, else down
the water route through the operated plants. A ``JUSENA`` target may be a
plant outside the configuration (one that aggregates a complex), which adds
its own productivity to the chain.

A reservoir with an ``EZ`` record has its chain cut where the chain leaves
the reservoir's REE: the plants above the cut count in its own REE, and the
REE below the cut values up to ``EZ`` % of the reservoir's useful volume at
the chain productivity of the first plant there. A reservoir without its own
``EZ`` record counts its whole chain in its own REE, whatever records the
other plants of the cascade carry.

novomodelo's accumulated productivity follows the ``hydros.json`` water route, so a
chain that departs from it is an ``expression`` parameter over the per-plant
integrated productivities, and its maximum stored energy is the useful range
of the computed ``min_storage``/``max_storage`` parameters valued at it. A
plant outside the configuration enters as a ``per_stage`` constant.

The ``EZ`` limit caps a volume, ``min(V, EZ·Vu)``, which an affine constraint
cannot carry: each share counts the whole volume and only its maximum stored
energy is capped. :func:`ez_cap_unmodeled_stages` names the stages where that
differs from the source model.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from novomodelo_bridge.core.stored_energy import (
    StoredEnergyShare,
    cascade_share,
    rho_acum_mwh_name,
    rho_eq_mwh_name,
)
from novomodelo_bridge.core.units import HM3_TO_MWH_PER_RHO
from novomodelo_bridge.decomp.converters.cadastro import is_reservoir, storage_envelope
from novomodelo_bridge.decomp.converters.hydro import (
    _downstream_operated,
    _stored_energy_productivity_mw_per_m3s,
)
from novomodelo_bridge.novomodelo.scalar_parameters import (
    max_storage_name,
    min_storage_name,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from idecomp.decomp.dadger import Dadger

    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
    from novomodelo_bridge.decomp.id_map import DecompIdMap

_LOG = logging.getLogger(__name__)


def read_ez_fractions(dadger: Dadger) -> dict[int, float]:
    """``{plant code: fraction of useful volume}`` from the deck's ``EZ`` records.

    Raises:
        ValueError: a record's percentage lies outside ``[0, 100]``.
    """
    frame = dadger.ez(df=True)
    if frame is None or frame.empty:
        return {}
    fractions: dict[int, float] = {}
    for code, percent in zip(frame["codigo_usina"], frame["volume"], strict=True):
        if not 0.0 <= float(percent) <= 100.0:
            raise ValueError(
                f"EZ record for plant {int(code)} gives {float(percent)}% of the "
                "useful volume; it must lie between 0 and 100"
            )
        fractions[int(code)] = float(percent) / 100.0
    return fractions


def is_stored_energy_reservoir(effective: EffectiveCadastro, code: int) -> bool:
    """True iff the source model counts plant *code*'s storage in a REE's energy.

    A reservoir under :func:`~novomodelo_bridge.decomp.converters.cadastro.
    is_reservoir` (``"M"`` or ``"S"``) with usable storage in the *base*
    cadastro; a run-of-river ``"D"`` plant never stores energy.
    """
    if not is_reservoir(effective, code):
        return False
    vol_min = float(effective.base.loc[code, "volume_minimo"])
    vol_max = float(effective.base.loc[code, "volume_maximo"])
    return vol_max - vol_min > 0.0


@dataclass(frozen=True)
class RheShare:
    """One reservoir's share of a REE's stored energy, with its values.

    ``rho_mwh`` is the share's coefficient per stage (MWh/hm³) and
    ``useful_hm3`` the reservoir's useful range; the share's maximum stored
    energy is their product scaled by ``ez_fraction``. ``constants`` holds the
    ``per_stage`` parameters the share reads, one value per stage.
    """

    share: StoredEnergyShare
    rho_mwh: tuple[float, ...]
    useful_hm3: float
    ez_fraction: float = 1.0
    constants: Mapping[str, tuple[float, ...]] = field(default_factory=dict)

    def max_energy_mwh(self, stage_index: int) -> float:
        """The share's maximum stored energy at *stage_index*, MWh."""
        return self.rho_mwh[stage_index] * self.useful_hm3 * self.ez_fraction


@dataclass(frozen=True)
class RheCoupling:
    """One reservoir whose energy chain an ``EZ`` record cuts at a REE boundary."""

    plant_code: int
    ree_code: int
    cut_code: int
    ez_percent: float
    downstream_ree_code: int
    downstream_code: int


@dataclass(frozen=True)
class RheShares:
    """Every REE's shares, in plant-code order, and the cut reservoirs."""

    by_ree: dict[int, list[RheShare]]
    couplings: list[RheCoupling]


@dataclass(frozen=True)
class _Chain:
    """The plants of one energy chain, from its first plant down to the cut.

    ``crossing`` is the first plant below the cut, in another REE; ``None``
    when the chain runs to the sink.
    """

    members: tuple[int, ...]
    crossing: int | None


@dataclass(frozen=True)
class _Coefficient:
    """A chain's productivity: its ``@name``, body, and per-stage values.

    ``body`` is ``None`` when the name is a novomodelo computed parameter;
    ``water_route`` marks the chain that is its head's ``hydros.json`` water
    route, whose productivity is novomodelo's accumulated one.
    """

    name: str
    body: str | None
    values: tuple[float, ...]
    constants: dict[str, tuple[float, ...]]
    equivalent_ids: frozenset[int]
    water_route: bool = False


def _chain_name(hydro_id: int) -> str:
    return f"rho_chain_mwh_h{hydro_id}"


def _plant_constant_name(code: int) -> str:
    return f"rho_eq_mwh_plant{code}"


class _EnergyCascade:
    """Energy chains and their productivities over one converted case."""

    def __init__(
        self,
        effective: EffectiveCadastro,
        id_map: DecompIdMap,
        hydro_to_ree: Mapping[int, int],
        ez_fractions: Mapping[int, float],
    ) -> None:
        self._effective = effective
        self._id_map = id_map
        self._ree = hydro_to_ree
        self._ez = ez_fractions
        self._operated = set(id_map.hydro_codes)
        self._water: dict[int, int | None] = {}
        self._chains: dict[int, _Chain] = {}

    def water_next(self, code: int) -> int | None:
        """The next operated plant on *code*'s ``hydros.json`` water route."""
        if code not in self._water:
            self._water[code] = _downstream_operated(
                self._effective, code, self._operated
            )
        return self._water[code]

    def _energy_next(self, code: int) -> int | None:
        if self._effective.energy_link_varies(code):
            _LOG.warning(
                "plant %d's energy link (AC JUSENA) varies across stages; the "
                "stage-0 link is used for the whole horizon",
                code,
            )
        link = self._effective.energy_link(code, 0)
        if link is None:
            return self.water_next(code)
        if link == 0:
            return None
        if link not in self._operated and link not in self._effective.base.index:
            raise ValueError(
                f"plant {code}'s AC JUSENA names plant {link}, which is neither "
                "in the configuration nor in hidr.dat"
            )
        return link

    def chain(self, start: int) -> _Chain:
        """*start*'s energy chain, cut at its REE boundary iff it has an ``EZ``."""
        if start in self._chains:
            return self._chains[start]
        ree = self._ree[start]
        cut = start in self._ez
        members = [start]
        crossing: int | None = None
        while (below := self._energy_next(members[-1])) is not None:
            if below in members:
                _LOG.warning(
                    "energy chain from plant %d loops back to plant %d; treating "
                    "plant %d as the end of the chain",
                    start,
                    below,
                    members[-1],
                )
                break
            if cut and below in self._operated and self._ree.get(below) != ree:
                crossing = below
                break
            members.append(below)
        self._chains[start] = _Chain(tuple(members), crossing)
        return self._chains[start]

    def _rho_mwh(self, code: int) -> tuple[float, ...]:
        return tuple(
            _stored_energy_productivity_mw_per_m3s(self._effective, code, stage)
            * HM3_TO_MWH_PER_RHO
            for stage in range(self._effective.n_stages)
        )

    def _accumulated_rho_mwh(self, code: int) -> tuple[float, ...]:
        totals = [0.0] * self._effective.n_stages
        current: int | None = code
        while current is not None:
            for stage, value in enumerate(self._rho_mwh(current)):
                totals[stage] += value
            current = self.water_next(current)
        return tuple(totals)

    def _water_route_from(self, chain: _Chain) -> int:
        """Index of the first member from which *chain* is its water route."""
        members = chain.members
        last = members[-1]
        if (
            chain.crossing is not None
            or last not in self._operated
            or self.water_next(last) is not None
        ):
            return len(members)
        start = len(members) - 1
        while (
            start > 0
            and members[start - 1] in self._operated
            and self.water_next(members[start - 1]) == members[start]
        ):
            start -= 1
        return start

    def coefficient(self, chain: _Chain) -> _Coefficient:
        """*chain*'s productivity, reusing novomodelo's along its water-route tail."""
        members = chain.members
        head_id = self._id_map.hydro_id(members[0])
        route_from = self._water_route_from(chain)
        if route_from == 0:
            return _Coefficient(
                rho_acum_mwh_name(head_id),
                None,
                self._accumulated_rho_mwh(members[0]),
                {},
                frozenset(),
                water_route=True,
            )
        tokens: list[str] = []
        values = [0.0] * self._effective.n_stages
        constants: dict[str, tuple[float, ...]] = {}
        equivalent_ids: set[int] = set()
        for code in members[:route_from]:
            rho = self._rho_mwh(code)
            if code in self._operated:
                hydro_id = self._id_map.hydro_id(code)
                tokens.append(f"@{rho_eq_mwh_name(hydro_id)}")
                equivalent_ids.add(hydro_id)
            else:
                name = _plant_constant_name(code)
                tokens.append(f"@{name}")
                constants[name] = rho
            values = [v + r for v, r in zip(values, rho, strict=True)]
        if route_from < len(members):
            tail = members[route_from]
            tokens.append(f"@{rho_acum_mwh_name(self._id_map.hydro_id(tail))}")
            values = [
                v + r
                for v, r in zip(values, self._accumulated_rho_mwh(tail), strict=True)
            ]
        if len(tokens) == 1 and not constants:
            return _Coefficient(
                tokens[0].removeprefix("@"),
                None,
                tuple(values),
                {},
                frozenset(equivalent_ids),
            )
        return _Coefficient(
            _chain_name(head_id),
            " + ".join(tokens),
            tuple(values),
            constants,
            frozenset(equivalent_ids),
        )


def _useful_range(hydro_id: int) -> str:
    return f"(@{max_storage_name(hydro_id)} - @{min_storage_name(hydro_id)})"


def _chain_share(
    hydro_id: int,
    coefficient: _Coefficient,
    max_name: str,
    useful_hm3: float,
    ez_fraction: float = 1.0,
) -> RheShare:
    scale = "" if ez_fraction == 1.0 else f"{ez_fraction!r} * "
    parameters = (
        {} if coefficient.body is None else {coefficient.name: coefficient.body}
    )
    parameters[max_name] = f"{scale}{_useful_range(hydro_id)} * @{coefficient.name}"
    return RheShare(
        StoredEnergyShare(hydro_id, coefficient.name, max_name, parameters),
        coefficient.values,
        useful_hm3,
        ez_fraction,
        coefficient.constants,
    )


def build_rhe_shares(
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    hydro_to_ree: Mapping[int, int],
    ez_fractions: Mapping[int, float],
) -> RheShares:
    """Every REE's RHE shares over its operated stored-energy reservoirs.

    A reservoir whose chain is its ``hydros.json`` water route takes
    :func:`~novomodelo_bridge.core.stored_energy.cascade_share`, unless another
    share's coefficient names its own equivalent productivity: novomodelo warns
    when one constraint pairs a hydro's maximum stored energy with any
    productivity but its accumulated one, so that reservoir's maximum is its
    useful range valued at the accumulated productivity instead.

    *ez_fractions* maps a plant code to its ``EZ`` fraction
    (:func:`read_ez_fractions`); an ``EZ`` of 0 cuts the chain and gives the
    REE below no share.
    """
    cascade = _EnergyCascade(effective, id_map, hydro_to_ree, ez_fractions)
    operated = set(id_map.hydro_codes)
    reservoirs = [
        code
        for code in sorted(hydro_to_ree)
        if code in operated and is_stored_energy_reservoir(effective, code)
    ]
    own = {code: cascade.coefficient(cascade.chain(code)) for code in reservoirs}
    downstream = {
        code: cascade.coefficient(cascade.chain(crossing))
        for code in reservoirs
        if (crossing := cascade.chain(code).crossing) is not None
        and ez_fractions[code] > 0.0
    }
    paired_ids = frozenset().union(
        *(c.equivalent_ids for c in (*own.values(), *downstream.values()))
    )

    by_ree: dict[int, list[RheShare]] = defaultdict(list)
    couplings: list[RheCoupling] = []
    for code in reservoirs:
        hydro_id = id_map.hydro_id(code)
        min_storage, max_storage = storage_envelope(effective, code)
        useful = max_storage - min_storage
        coefficient = own[code]
        ree_code = hydro_to_ree[code]
        if coefficient.water_route and hydro_id not in paired_ids:
            by_ree[ree_code].append(
                RheShare(cascade_share(hydro_id), coefficient.values, useful)
            )
        else:
            by_ree[ree_code].append(
                _chain_share(
                    hydro_id, coefficient, f"earm_max_chain_mwh_h{hydro_id}", useful
                )
            )

        chain = cascade.chain(code)
        if chain.crossing is None:
            continue
        downstream_ree = hydro_to_ree[chain.crossing]
        if code in downstream:
            by_ree[downstream_ree].append(
                _chain_share(
                    hydro_id,
                    downstream[code],
                    f"earm_max_ez_mwh_h{hydro_id}",
                    useful,
                    ez_fractions[code],
                )
            )
        couplings.append(
            RheCoupling(
                plant_code=code,
                ree_code=ree_code,
                cut_code=chain.members[-1],
                ez_percent=ez_fractions[code] * 100.0,
                downstream_ree_code=downstream_ree,
                downstream_code=chain.crossing,
            )
        )
    return RheShares(dict(by_ree), couplings)


def ez_cap_unmodeled_stages(
    shares: Sequence[tuple[float, RheShare]], floors: Mapping[int, float]
) -> list[int]:
    """Stages where counting a capped share's whole volume may loosen the floor.

    The source model counts a share with ``EZ`` below 100% as
    ``ρ·min(V, EZ·Vu)``. With every share positive the left side is never
    below any one capped share's maximum, so where each meets the stage's
    *floor* (MWh) on its own the cap cannot bind and the uncapped constraint
    is exact. A record with a negative share is never certified.
    """
    capped = [share for _, share in shares if share.ez_fraction < 1.0]
    if not capped:
        return []
    if any(sign < 0.0 for sign, _ in shares):
        return sorted(floors)
    return [
        stage
        for stage, floor in sorted(floors.items())
        if min(share.max_energy_mwh(stage) for share in capped) < floor
    ]
