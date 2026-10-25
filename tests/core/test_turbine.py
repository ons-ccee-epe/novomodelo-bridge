"""Tests for the shared turbine discharge limit (``novomodelo_bridge.core.turbine``)."""

from __future__ import annotations

import pytest

from novomodelo_bridge.core.turbine import head_limited_flow, tailrace_limited_flow


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
