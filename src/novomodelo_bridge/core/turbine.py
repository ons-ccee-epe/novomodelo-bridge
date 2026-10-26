"""Turbine discharge limit at a net head, shared by both conversion tracks.

A turbine set rated for flow ``q_ef`` at net head ``h_ef`` swallows
``q_ef · (h / h_ef)^α`` at net head ``h`` (the affinity law; ``α`` is
:data:`~novomodelo_bridge.core.productivity.KTURB_BY_TIPO_TURBINA`'s exponent).
When the tailrace level rises with the flow, the head depends on the flow it
limits, and :func:`settled_flow` solves the two together.
:func:`power_limited_flow` also caps the plant where its generators together
reach rated power. :func:`turbine_limit_gap_diagnostic` reports the plants
either track cannot rate this way for lack of data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from novomodelo_bridge.core.diagnostics import Diagnostic, DiagnosticTable, Severity

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


class MachineSet(NamedTuple):
    """*units* identical machines rated at *flow_m3s*, *head_m* and *power_mw*."""

    units: int
    flow_m3s: float
    head_m: float
    power_mw: float


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


def power_limited_flow(
    turbine_head_m: float,
    sets: Iterable[MachineSet],
    *,
    generator_head_m: float,
    specific_productivity: float,
    exponent: float,
) -> float:
    """``min(Σ n · q_ef (h_t / h_ef)^α, Σ n · P_ef / (ρ_esp · h_g))`` (m³/s).

    The sets swallow their head-limited flow at the turbine head *h_t* until
    the plant's generators together reach rated power at the generator head
    *h_g*; a set whose own generators would bind first is not capped on its
    own. Zero when either head is not positive; *specific_productivity* is
    ``ρ_esp`` in MW/(m³/s)/m and must be positive.
    """
    if turbine_head_m <= 0.0 or generator_head_m <= 0.0:
        return 0.0
    machines = tuple(sets)
    swallowed = sum(
        machine.units
        * head_limited_flow(
            turbine_head_m,
            nominal_head_m=machine.head_m,
            nominal_flow_m3s=machine.flow_m3s,
            exponent=exponent,
        )
        for machine in machines
    )
    rated_power = sum(machine.units * machine.power_mw for machine in machines)
    return min(swallowed, rated_power / (specific_productivity * generator_head_m))


def settled_flow(
    flow_at: Callable[[float], float],
    start_m3s: float,
    *,
    tolerance_m3s: float,
    max_iterations: int,
) -> tuple[float, bool]:
    """The flow ``q = flow_at(q)`` and whether it converged.

    Iterates from *start_m3s* until two successive flows differ by less than
    *tolerance_m3s*; after *max_iterations* without converging it returns the
    last flow and ``False``.
    """
    flow = start_m3s
    for _ in range(max_iterations):
        following = flow_at(flow)
        if abs(following - flow) < tolerance_m3s:
            return following, True
        flow = following
    return flow, False


def tailrace_limited_flow(
    net_head_at: Callable[[float], float],
    *,
    nominal_head_m: float,
    nominal_flow_m3s: float,
    exponent: float,
    tolerance_m3s: float,
    max_iterations: int,
) -> tuple[float, bool]:
    """The flow ``q = head_limited_flow(net_head_at(q))``, the source model's
    fixed point from the rated flow (:func:`settled_flow`), and whether it
    converged; *net_head_at* maps a turbined flow (m³/s) to the net head (m) at
    that flow's tailrace level.
    """
    return settled_flow(
        lambda flow: head_limited_flow(
            net_head_at(flow),
            nominal_head_m=nominal_head_m,
            nominal_flow_m3s=nominal_flow_m3s,
            exponent=exponent,
        ),
        nominal_flow_m3s,
        tolerance_m3s=tolerance_m3s,
        max_iterations=max_iterations,
    )


def turbine_limit_gap_diagnostic(rows: Sequence[Sequence[object]]) -> Diagnostic:
    """WARNING for plants whose turbined-flow limit leaves out part of the
    source model's rule because a datum it needs is missing.

    Each row is ``[code, name, missing datum, limit used, stages]``. The limit
    used is above the source model's whenever the missing datum would have
    lowered it.
    """
    return Diagnostic(
        code="turbine-limit-data-missing",
        severity=Severity.WARNING,
        category="Hydro capacity",
        title=f"Turbined-flow limit without its head rule ({len(rows)})",
        summary=(
            f"{len(rows)} plant(s) lack a datum the source model's "
            "turbined-flow limit is computed from; their limit leaves out the "
            "part of the rule that datum feeds and can sit above the source "
            "model's."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Missing", "Limit used", "Stages"],
            rows=[list(row) for row in rows],
            caption="Stage ids are 0-based.",
        ),
    )
