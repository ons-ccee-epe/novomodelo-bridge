"""Run configuration and penalties for DECOMP-like decks.

``config.json`` carries the deck's ``GP`` convergence criterion as a
**relative** ``Gap`` stopping rule (``relative_tolerance``, matching DECOMP's
``Zsup/Zinf - 1 <= GP`` convergence; cobre auto-injects a ``BoundStalling``
companion so an unattainable tolerance degrades to a diagnosed stall), the
deck-faithful ``NI`` iteration backstop, and the external scenario schemes.

No ``state_space.inflow_lag_depth`` is emitted. Under a deferred boundary FCF
the external white-noise inflow model contributes no inflow-lag state, so cobre
resolves a zero depth and reserving lag slots would be dead state (and would
raise cobre's lag-blind-stage advisory for nothing). The inflow-lag depth is a
property of the *boundary policy*: the boundary-FCF importer
(``fcf/importer.py``) reserves exactly the depth the loaded cuts reference — and
only when a boundary policy is actually imported. cobre's own inflow-lag-depth
inference (sized from PAR(p) plus the boundary policy) is slated to derive that
depth from the checkpoint itself, retiring even the importer's patch.

``penalties.json`` reuses the shared ρ-scaled hydro penalty construction
with the deck's deficit cost and the converted productivities — the same
formulas the other converter family applies, so the two case families
share one penalty convention.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cobre_bridge.cobre import schemas as cobre_schemas
from cobre_bridge.core.penalties import PCORTEOL, PEXC, PINT, hydro_penalty_costs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cobre_bridge.decomp.case import DecompCase

_LOG = logging.getLogger(__name__)


def convert_config(case: DecompCase) -> dict:
    """Build ``config.json``: Gap + NI stopping rules, external scenario
    sources, simulation on.

    The ``gap`` stopping rule (relative ``GP``) is emitted unconditionally —
    the faithful analogue of DECOMP's own ``Zsup/Zinf − 1 ≤ GP`` convergence,
    which is itself risk-adjusted. It is admissible under a CVaR risk measure
    too: under enumerated forwards cobre computes the exact *risk-adjusted*
    upper bound (``setup/mod.rs``, cobre commit landing the enumerated CVaR
    UB), which brackets the risk-adjusted lower bound, provided the risk
    measure is **uniform across all stages** — which
    :func:`cobre_bridge.decomp.temporal.stage_records` guarantees by emitting
    the CVaR measure on every stage (``reject_gap_under_nonuniform_risk``).
    Requires that cobre build; an older cobre without the enumerated
    risk-adjusted UB rejects a gap rule under CVaR and would need a
    ``bound_stalling`` fallback instead.

    Both training and simulation use ``selection = {"method": "enumerated"}``:
    the explicit trunk-plus-fan node graph enumerates every root-to-leaf path,
    so training runs the full forward/backward census and simulation runs the
    exact per-node-probability weighted census (cobre 0.14+ wires the
    branching-graph census simulation, retiring the earlier ``sampled`` fallback
    tracked as C9). The simulation omits its own ``scenario_source`` and inherits
    training's external one.

    ``scenario_source.seed`` is a fixed ``0``. A seed controls random sampling;
    with every class ``external`` (a deterministic replay of the explicit tree)
    and ``enumerated`` selection nothing samples, so the value is inert — but
    cobre's schema requires the field whenever any class is external, so a
    constant placeholder is emitted rather than a study-varying value that would
    misleadingly imply a meaningful random draw.

    No ``state_space`` block is emitted: the inflow-lag depth is a property of
    the boundary policy, not the case inputs. With the boundary FCF deferred the
    external inflow model needs no lag state, so cobre resolves a zero depth; the
    boundary-FCF importer reserves the cut-derived depth when a boundary is
    actually imported (see the module docstring).
    """
    dadger = case.dadger
    ni = int(dadger.ni.iteracoes or 500)
    gp = float(dadger.gp.data[0])
    _LOG.info(
        "emitting the deck's GP=%g as a relative Gap stopping rule "
        "(relative_tolerance, DECOMP's Zsup/Zinf-1 <= GP convergence); cobre "
        "auto-injects a BoundStalling companion, with the NI=%d iteration backstop",
        gp,
        ni,
    )
    stopping_rules = [
        {"type": "gap", "relative_tolerance": gp},
        {"type": "iteration_limit", "limit": ni},
    ]

    return {
        "$schema": cobre_schemas.schema_url_for("config.json"),
        "training": {
            "selection": {"method": "enumerated"},
            "stopping_rules": stopping_rules,
            # Under the node-native explicit tree every stochastic class is
            # external: inflow (the tree), NCS (renewables), and load. cobre's
            # scheme-aware load membership admits an external load class
            # regardless of σ (a deterministic std = 0 load standardizes to
            # eta = 0), so load is external here rather than the former
            # in-sample-with-null-std workaround. seed is a schema-required
            # inert placeholder (0) — external + enumerated never samples.
            "scenario_source": {
                "seed": 0,
                "inflow": {"scheme": "external"},
                "load": {"scheme": "external"},
                "ncs": {"scheme": "external"},
            },
        },
        "simulation": {
            "enabled": True,
            "selection": {"method": "enumerated"},
        },
    }


def convert_penalties(
    deficit_cost: float,
    productivities: Sequence[float],
) -> dict:
    """Build ``penalties.json`` from the deck's deficit cost.

    ``ρ_avg`` is the mean converted productivity over every operated plant
    (zeros included) and ``ρ_max`` the maximum — the same convention the
    shared hydro-penalty construction expects; there is no per-deck
    penalty file, so every hydro slot takes its deficit-derived default.
    """
    values = list(productivities)
    rho_avg = sum(values) / len(values) if values else 1.0
    rho_max = max(values) if values else rho_avg
    hydro_costs = hydro_penalty_costs(
        rho_avg=rho_avg,
        rho_max_acum=rho_max,
        penalid_costs={},
        max_deficit_cost=deficit_cost,
    )
    return {
        "$schema": cobre_schemas.schema_url_for("penalties.json"),
        "bus": {
            "deficit_segments": [{"depth_mw": None, "cost": deficit_cost}],
            "excess_cost": PEXC,
        },
        "hydro": hydro_costs,
        "line": {"exchange_cost": PINT},
        "non_controllable_source": {"curtailment_cost": PCORTEOL},
    }
