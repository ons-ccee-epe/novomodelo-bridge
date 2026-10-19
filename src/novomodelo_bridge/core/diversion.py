"""Diversion-channel flow limits, shared by both conversion tracks.

The source models write :data:`UNLIMITED_FLOW_M3S` m³/s as a diversion
channel's flow limit when the channel has none. A novomodelo channel without a
limit carries ``max_flow_m3s: null``, so that value, and any larger one, is
converted to ``None`` rather than emitted as a finite bound.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from novomodelo_bridge.core.diagnostics import Diagnostic, DiagnosticTable, Severity

UNLIMITED_FLOW_M3S: Final = 9_999_999.0
"""The source models' "no limit" value for a diversion channel's flow."""


def channel_flow_limit(flow_m3s: float | None) -> float | None:
    """The novomodelo ``max_flow_m3s`` for a channel whose source limit is ``flow_m3s``.

    ``None`` (no limit) when the source gives none or gives at least
    :data:`UNLIMITED_FLOW_M3S`.
    """
    if flow_m3s is None or flow_m3s >= UNLIMITED_FLOW_M3S:
        return None
    return float(flow_m3s)


def is_unlimited_flow(flow_m3s: float | None) -> bool:
    """Whether a source limit of ``flow_m3s`` is the "no limit" value."""
    return flow_m3s is not None and flow_m3s >= UNLIMITED_FLOW_M3S


def unlimited_channels_diagnostic(
    channels: Iterable[tuple[int, str, int, str, float]],
    *,
    source: str,
) -> Diagnostic:
    """INFO listing the ``(code, name, target code, target name, given flow)``
    channels whose ``source`` limit means no limit."""
    rows = [
        [code, name, target, target_name, f"{flow:.0f}"]
        for code, name, target, target_name, flow in channels
    ]
    return Diagnostic(
        code="diversion-channel-unlimited",
        severity=Severity.INFO,
        category="Diversion channels",
        title=f"Diversion channels without a flow limit ({len(rows)})",
        summary=(
            f"{len(rows)} diversion channel(s) give a flow limit of "
            f"{UNLIMITED_FLOW_M3S:.0f} m³/s or more in {source}, the source "
            "model's value for no limit, so they are converted without one."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Target code", "Target name", "Given (m³/s)"],
            rows=rows,
            justify=["right", "left", "right", "left", "right"],
        ),
    )
