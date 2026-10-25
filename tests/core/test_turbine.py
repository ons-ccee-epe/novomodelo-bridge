"""Tests for the shared turbine discharge limit (``novomodelo_bridge.core.turbine``)."""

from __future__ import annotations

import pytest

from novomodelo_bridge.core.turbine import (
    MachineSet,
    head_limited_flow,
    power_limited_flow,
    settled_flow,
    tailrace_limited_flow,
)


def test_head_limited_flow_follows_the_affinity_law() -> None:
    flow = head_limited_flow(
        20.0, nominal_head_m=80.0, nominal_flow_m3s=100.0, exponent=0.5
    )
    assert flow == pytest.approx(50.0)


def test_head_limited_flow_is_zero_at_a_negative_head() -> None:
    flow = head_limited_flow(
        -5.0, nominal_head_m=80.0, nominal_flow_m3s=100.0, exponent=0.5
    )
    assert flow == 0.0


def test_head_limited_flow_without_a_rated_head_is_the_rated_flow() -> None:
    flow = head_limited_flow(
        20.0, nominal_head_m=0.0, nominal_flow_m3s=100.0, exponent=0.5
    )
    assert flow == 100.0


def test_power_limited_flow_caps_each_set_at_its_rated_power() -> None:
    # At 20 m every turbine swallows 50 m³/s; the second set's 6 MW generators
    # cap theirs at 6 / (0.009 × 20) ≈ 33.3 m³/s.
    sets = [
        MachineSet(units=2, flow_m3s=100.0, head_m=80.0, power_mw=50.0),
        MachineSet(units=2, flow_m3s=100.0, head_m=80.0, power_mw=6.0),
    ]
    flow = power_limited_flow(
        20.0, sets, generator_head_m=20.0, specific_productivity=0.009, exponent=0.5
    )
    assert flow == pytest.approx(2 * 50.0 + 2 * 6.0 / (0.009 * 20.0))


def test_power_limited_flow_rates_generators_at_their_own_head() -> None:
    sets = [MachineSet(units=1, flow_m3s=100.0, head_m=80.0, power_mw=6.0)]
    flow = power_limited_flow(
        20.0, sets, generator_head_m=30.0, specific_productivity=0.009, exponent=0.5
    )
    assert flow == pytest.approx(6.0 / (0.009 * 30.0))


def test_power_limited_flow_is_zero_at_a_non_positive_head() -> None:
    sets = [MachineSet(units=1, flow_m3s=100.0, head_m=80.0, power_mw=50.0)]
    flow = power_limited_flow(
        0.0, sets, generator_head_m=20.0, specific_productivity=0.009, exponent=0.5
    )
    assert flow == 0.0


def test_settled_flow_stops_once_successive_flows_agree() -> None:
    flow, converged = settled_flow(
        lambda q: 0.5 * q + 10.0, 0.0, tolerance_m3s=1e-9, max_iterations=200
    )
    assert converged
    assert flow == pytest.approx(20.0)


def test_tailrace_limited_flow_settles_where_head_and_flow_agree() -> None:
    def net_head_at(flow_m3s: float) -> float:
        return 80.0 - 0.1 * flow_m3s

    flow, converged = tailrace_limited_flow(
        net_head_at,
        nominal_head_m=80.0,
        nominal_flow_m3s=100.0,
        exponent=0.5,
        tolerance_m3s=1e-9,
        max_iterations=100,
    )
    assert converged
    assert flow == pytest.approx(100.0 * (net_head_at(flow) / 80.0) ** 0.5)
    assert flow < 100.0


def test_tailrace_limited_flow_stops_at_the_tolerance() -> None:
    """The flow after the first step under the tolerance is returned, not the
    exact fixed point."""
    flow, converged = tailrace_limited_flow(
        lambda q: 80.0 - 0.1 * q,
        nominal_head_m=80.0,
        nominal_flow_m3s=100.0,
        exponent=0.5,
        tolerance_m3s=1.0,
        max_iterations=100,
    )
    first = 100.0 * (70.0 / 80.0) ** 0.5
    second = 100.0 * ((80.0 - 0.1 * first) / 80.0) ** 0.5
    assert converged
    assert flow == pytest.approx(second)


def test_tailrace_limited_flow_reports_an_oscillation() -> None:
    """A tailrace steep enough to drown the head at the rated flow swings the
    flow between the rated flow and zero."""
    flow, converged = tailrace_limited_flow(
        lambda q: 80.0 - 2.0 * q,
        nominal_head_m=80.0,
        nominal_flow_m3s=100.0,
        exponent=0.5,
        tolerance_m3s=1.0,
        max_iterations=10,
    )
    assert not converged
    assert flow == 100.0
