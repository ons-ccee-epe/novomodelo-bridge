"""Turbine discharge limit at a net head, shared by both conversion tracks.

A turbine set rated for flow ``q_ef`` at net head ``h_ef`` swallows
``q_ef · (h / h_ef)^α`` at net head ``h`` (the affinity law; ``α`` is
:data:`~novomodelo_bridge.core.productivity.KTURB_BY_TIPO_TURBINA`'s exponent).
When the tailrace level rises with the flow, the head depends on the flow it
limits, and :func:`tailrace_limited_flow` solves the two together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def head_limited_flow(
    net_head_m: float,
    *,
    nominal_head_m: float,
    nominal_flow_m3s: float,
    exponent: float,
) -> float:
    """``q_ef · (h / h_ef)^α`` (m³/s), zero at a non-positive net head.

    A non-positive *nominal_head_m* (no rated head declared) returns the rated
    *nominal_flow_m3s* unchanged.
    """
    if nominal_head_m <= 0.0:
        return nominal_flow_m3s
    return nominal_flow_m3s * (max(net_head_m, 0.0) / nominal_head_m) ** exponent


def tailrace_limited_flow(
    net_head_at: Callable[[float], float],
    *,
    nominal_head_m: float,
    nominal_flow_m3s: float,
    exponent: float,
    tolerance_m3s: float,
    max_iterations: int,
) -> tuple[float, bool]:
    """The flow ``q = head_limited_flow(net_head_at(q))`` and whether it converged.

    Iterates from the rated flow until two successive flows differ by less than
    *tolerance_m3s*, the source model's fixed point; *net_head_at* maps a
    turbined flow (m³/s) to the net head (m) at that flow's tailrace level.
    After *max_iterations* without converging it returns the last flow and
    ``False``.
    """
    flow = nominal_flow_m3s
    for _ in range(max_iterations):
        following = head_limited_flow(
            net_head_at(flow),
            nominal_head_m=nominal_head_m,
            nominal_flow_m3s=nominal_flow_m3s,
            exponent=exponent,
        )
        if abs(following - flow) < tolerance_m3s:
            return following, True
        flow = following
    return flow, False
