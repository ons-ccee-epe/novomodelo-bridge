"""Anticipated thermal commitments, shared by both conversion tracks.

Novomodelo requires every past anticipated commitment to lie within the
plant's generation limits, so a commitment outside them is moved to the
nearest limit and reported.
"""

from __future__ import annotations

from collections.abc import Sequence

from novomodelo_bridge.core.diagnostics import Diagnostic, DiagnosticTable, Severity


def commitment_clamped_diagnostic(
    rows: Sequence[Sequence[object]], *, source: str
) -> Diagnostic:
    """WARNING listing the ``(code, name, delivery, committed, minimum, maximum,
    used)`` commitments of *source* moved into the plant's generation limits."""
    return Diagnostic(
        code="anticipated-commitment-clamped",
        severity=Severity.WARNING,
        category="GNL",
        title=f"Anticipated commitments outside the generation limits ({len(rows)})",
        summary=(
            f"The listed commitments in {source} fall outside the plant's "
            "generation limits, which the case requires them to meet, so the "
            "conversion moves each to the nearest limit, and the committed "
            "generation differs from the source model's."
        ),
        table=DiagnosticTable(
            columns=[
                "Code",
                "Name",
                "Delivery",
                "Committed (MW)",
                "Minimum (MW)",
                "Maximum (MW)",
                "Used (MW)",
            ],
            rows=rows,
            justify=["right", "left", "left", "right", "right", "right", "right"],
            caption="Stage ids are 0-based.",
        ),
        remediation=(
            f"Bring each listed commitment in {source} within the plant's "
            "generation limits."
        ),
    )
