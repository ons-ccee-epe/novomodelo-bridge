"""Tests for the DECOMP config emitter (``convert_config``)."""

from __future__ import annotations

from pathlib import Path

from novomodelo_bridge.decomp.converters.config import convert_config
from tests.conftest import make_decomp_case


class _Ni:
    iteracoes = 250


class _Gp:
    data = [0.5]


class _Dt:
    dia, mes, ano = 18, 7, 2026


class _Dadger:
    ni = _Ni()
    gp = _Gp()
    dt = _Dt()


class TestConvertConfigStateSpace:
    def test_convert_config_omits_state_space(self) -> None:
        # The inflow-lag depth is a property of the boundary policy, not the
        # case inputs. With the boundary FCF deferred novomodelo resolves a zero
        # depth, so no ``state_space`` block is emitted (reserving lag slots
        # would be dead state); the boundary-FCF importer patches the
        # cut-derived depth in only when a boundary is actually imported.
        result = convert_config(make_decomp_case(Path("unused"), dadger=_Dadger()))

        assert "state_space" not in result

    def test_convert_config_training_and_simulation_enumerated(self) -> None:
        result = convert_config(make_decomp_case(Path("unused"), dadger=_Dadger()))

        # Training enumerates the explicit trunk-plus-fan node graph; every
        # stochastic class is external — inflow (the tree), load, and NCS.
        # novomodelo's scheme-aware load membership admits a deterministic (std = 0)
        # external load class (it standardizes to eta = 0). The seed is a
        # schema-required inert placeholder fixed at 0 (external + enumerated
        # never samples).
        expected_training = {
            "selection": {"method": "enumerated"},
            "stopping_rules": [
                {"type": "gap", "relative_tolerance": 0.5},
                {"type": "iteration_limit", "limit": 250},
            ],
            "scenario_source": {
                "seed": 0,
                "inflow": {"scheme": "external"},
                "load": {"scheme": "external"},
                "ncs": {"scheme": "external"},
            },
        }
        # Simulation is the exact weighted census over the branching graph
        # (novomodelo 0.14+ wires it; the old C9 sampled fallback is retired). With
        # no simulation.scenario_source, novomodelo inherits training's external one.
        expected_simulation = {
            "enabled": True,
            "selection": {"method": "enumerated"},
        }

        assert result["training"] == expected_training
        assert result["simulation"] == expected_simulation
