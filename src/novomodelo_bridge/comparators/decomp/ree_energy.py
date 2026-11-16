"""DECOMP's REE stored and inflow energies, valued over per-plant results.

``dec_oper_ree.csv`` reports each REE's stored energy (EARM) and inflow energy
(ENA) under DECOMP's own definition. novomodelo's per-plant energies value a
plant's water down its whole water route, whatever REE that crosses, so the
comparison values Novomodelo's storage and inflow with DECOMP's definition:

- a plant's water is valued along its energy chain
  (:func:`~novomodelo_bridge.decomp.converters.rhe_shares.energy_chain_next`),
  each chain plant at the head of its current filling
  (:func:`~novomodelo_bridge.decomp.converters.hydro.stored_energy_productivity_at_storage`);
- EARM counts the whole chain in the plant's REE unless the plant has an
  ``EZ`` record; then the chain is cut where it first leaves that REE, and the
  REE below counts at most ``EZ`` % of the plant's useful range;
- ENA moves the chain to the REE of its first plant past a subsystem boundary;
  inside a subsystem it moves to a plant of another REE only until the chain
  has passed a monthly reservoir there. ``EZ`` does not cap it;
- both are in MW-month of the source model's mean month
  (:data:`~novomodelo_bridge.core.units.MEAN_MONTH_HOURS`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, partial
from typing import TYPE_CHECKING

import polars as pl

from novomodelo_bridge.core.units import C_M3S2HM3, MEAN_MONTH_HOURS
from novomodelo_bridge.decomp.converters.cadastro import effective_storage_range
from novomodelo_bridge.decomp.converters.hydro import (
    stored_energy_productivity_at_storage,
)
from novomodelo_bridge.decomp.converters.rhe_shares import energy_chain_next

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro

REE_ENERGY_SCHEMA: dict[str, type[pl.DataType]] = {
    "scenario_id": pl.Int64,
    "stage_id": pl.Int64,
    "codigo_ree": pl.Int64,
    "earm_mwmes": pl.Float64,
    "ena_mwmes": pl.Float64,
}


@dataclass(frozen=True)
class EnergyCredit:
    """One chain plant valuing a plant's water, and the REE that counts it.

    ``capped`` marks a stored-energy credit below an ``EZ`` cut.
    """

    plant: int
    ree: int
    capped: bool = False


def stored_energy_credits(
    chain: Sequence[int], ree: Mapping[int, int], *, cut: bool
) -> tuple[EnergyCredit, ...]:
    """The EARM credits of the water stored at ``chain[0]``.

    With *cut* (the plant has an ``EZ`` record) the credits move, capped, to
    the REE of the first chain plant outside the plant's own REE. A chain plant
    without an REE is counted by the current one.
    """
    owner = ree[chain[0]]
    capped = False
    credits = []
    for plant in chain:
        home = ree.get(plant, owner)
        if cut and not capped and home != owner:
            owner, capped = home, True
        credits.append(EnergyCredit(plant, owner, capped))
    return tuple(credits)


def inflow_energy_credits(
    chain: Sequence[int],
    ree: Mapping[int, int],
    subsystem: Mapping[int, int],
    monthly: Collection[int],
) -> tuple[EnergyCredit, ...]:
    """The ENA credits of the inflow at ``chain[0]``.

    A chain plant without an REE or subsystem is counted by the current ones.
    """
    owner = ree[chain[0]]
    grid = subsystem[chain[0]]
    passed = False
    credits = []
    for plant in chain:
        home = ree.get(plant, owner)
        plant_grid = subsystem.get(plant, grid)
        if plant_grid != grid:
            owner, grid, passed = home, plant_grid, False
        elif home != owner and not passed:
            owner = home
        passed = passed or plant in monthly
        credits.append(EnergyCredit(plant, owner))
    return tuple(credits)


@dataclass(frozen=True)
class ReeEnergyRule:
    """DECOMP's REE energy definition over one deck.

    ``stored``/``inflow`` hold each plant's EARM/ENA credits and
    ``ez_fractions`` each cut plant's ``EZ`` cap. ``storage_range(code,
    stage)`` gives a plant's ``(floor, ceiling)`` in hm³ and
    ``productivity(code, stage)`` its stored-energy ρ as a function of the
    useful volume it holds.
    """

    stored: Mapping[int, tuple[EnergyCredit, ...]]
    inflow: Mapping[int, tuple[EnergyCredit, ...]]
    ez_fractions: Mapping[int, float]
    storage_range: Callable[[int, int], tuple[float, float]]
    productivity: Callable[[int, int], Callable[[float], float]]

    def energies(
        self, plants: pl.DataFrame, stage_hours: Mapping[int, float]
    ) -> pl.DataFrame:
        """Each REE's ``earm_mwmes`` and ``ena_mwmes`` per scenario and stage.

        *plants* holds one row per ``scenario_id``/``stage_id``/``codigo_usina``
        with the plant's ``storage_final_hm3`` and ``incremental_inflow_m3s``;
        a plant without a row holds no useful volume and has no inflow, and a
        row of a plant on no chain is ignored. ENA spans the stage's
        *stage_hours*, one month for a stage without them.
        """
        credits = [*self.stored.values(), *self.inflow.values()]
        rees = sorted({credit.ree for chain in credits for credit in chain})
        chain_plants = {credit.plant for chain in credits for credit in chain}
        rows: list[tuple[int, int, int, float, float]] = []
        for group in plants.partition_by(
            ["scenario_id", "stage_id"], maintain_order=True
        ):
            scenario = int(group["scenario_id"][0])
            stage_id = int(group["stage_id"][0])
            useful = {
                int(code): max(
                    float(volume) - self.storage_range(code, stage_id)[0], 0.0
                )
                for code, volume in zip(
                    group["codigo_usina"], group["storage_final_hm3"], strict=True
                )
                if volume is not None and code in chain_plants
            }
            inflow = {
                int(code): float(flow)
                for code, flow in zip(
                    group["codigo_usina"], group["incremental_inflow_m3s"], strict=True
                )
                if flow is not None
            }
            rho = {
                plant: self.productivity(plant, stage_id)(useful.get(plant, 0.0))
                for plant in chain_plants
            }
            earm = self._stored_energy(stage_id, useful, rho, rees)
            months = stage_hours.get(stage_id, MEAN_MONTH_HOURS) / MEAN_MONTH_HOURS
            ena = dict.fromkeys(rees, 0.0)
            for code, chain in self.inflow.items():
                for credit in chain:
                    ena[credit.ree] += (
                        inflow.get(code, 0.0) * rho[credit.plant] * months
                    )
            rows.extend((scenario, stage_id, ree, earm[ree], ena[ree]) for ree in rees)
        return pl.DataFrame(rows, schema=REE_ENERGY_SCHEMA, orient="row")

    def _stored_energy(
        self,
        stage: int,
        useful: Mapping[int, float],
        rho: Mapping[int, float],
        rees: Sequence[int],
    ) -> dict[int, float]:
        earm = dict.fromkeys(rees, 0.0)
        for code, chain in self.stored.items():
            volume = useful.get(code, 0.0)
            if volume <= 0.0:
                continue
            floor, ceiling = self.storage_range(code, stage)
            capped = min(volume, self.ez_fractions.get(code, 1.0) * (ceiling - floor))
            for credit in chain:
                held = capped if credit.capped else volume
                earm[credit.ree] += held * rho[credit.plant] / C_M3S2HM3
        return earm


def build_ree_energy_rule(
    effective: EffectiveCadastro,
    operated: Collection[int],
    membership: pl.DataFrame,
    ez_fractions: Mapping[int, float],
) -> ReeEnergyRule:
    """The rule over one deck, for each plant of *operated* with a row in
    *membership* (relato's plant, REE and subsystem table).

    Raises:
        ValueError: an ``AC JUSENA`` names a plant outside hidr.dat.
    """
    ree = {
        int(code): int(value)
        for code, value in zip(
            membership["codigo_usina"], membership["codigo_ree"], strict=True
        )
    }
    subsystem = {
        int(code): int(value)
        for code, value in zip(
            membership["codigo_usina"], membership["codigo_submercado"], strict=True
        )
    }
    operated_set = set(operated)
    chains = {
        code: _energy_chain(effective, code, operated_set)
        for code in sorted(operated_set)
        if code in ree
    }
    monthly = {
        plant
        for chain in chains.values()
        for plant in chain
        if str(effective.base.loc[plant, "tipo_regulacao"]).strip() == "M"
    }
    return ReeEnergyRule(
        stored={
            code: stored_energy_credits(chain, ree, cut=code in ez_fractions)
            for code, chain in chains.items()
        },
        inflow={
            code: inflow_energy_credits(chain, ree, subsystem, monthly)
            for code, chain in chains.items()
        },
        ez_fractions=dict(ez_fractions),
        storage_range=cache(partial(effective_storage_range, effective)),
        productivity=cache(partial(stored_energy_productivity_at_storage, effective)),
    )


def _energy_chain(
    effective: EffectiveCadastro, code: int, operated: set[int]
) -> tuple[int, ...]:
    """*code*'s energy chain, stopped before the first plant it revisits."""
    chain = [code]
    while (below := energy_chain_next(effective, chain[-1], operated)) is not None:
        if below in chain:
            break
        chain.append(below)
    return tuple(chain)
