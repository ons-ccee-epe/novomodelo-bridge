"""Natural-to-incremental inflow attribution, shared by both conversion tracks.

A plant's incremental inflow is its gauging station's natural inflow minus
the natural inflow of the stations directly upstream of it. Each plant keeps
its own upstream set: plants that share a station each subtract only their
own upstream stations, and an upstream station shared by two upstream plants
(a fictitious twin and its real plant) is subtracted once.

The artificial gauge (:data:`ARTIFICIAL_GAUGE`) marks a plant with no gauge
at all. Its water arrives only from upstream releases, diversions or
pumping, so its incremental inflow is zero whatever the deck stores for that
station, and nothing is subtracted from it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Final

import numpy as np

from novomodelo_bridge.core.diagnostics import Diagnostic, DiagnosticTable, Severity

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

ARTIFICIAL_GAUGE: Final = 300
"""The source models' "posto artificial": a plant on it has no natural inflow."""


def is_artificial_gauge(posto: int) -> bool:
    """Whether ``posto`` is the artificial gauge."""
    return posto == ARTIFICIAL_GAUGE


def incremental_by_plant(
    natural_by_posto: Mapping[int, np.ndarray],
    posto_of: Mapping[int, int],
    upstream_of: Mapping[int, Iterable[int]],
    codes: Iterable[int],
    *,
    zero_gauges: AbstractSet[int] = frozenset(),
) -> dict[int, np.ndarray]:
    """Return ``{plant code: incremental series}`` for each of ``codes``.

    Args:
        natural_by_posto: Natural series per station, all of one length.
        posto_of: Station of every plant in the cascade, upstream plants
            included.
        upstream_of: Directly upstream cascade plants of each plant.
        codes: Plants to attribute; a plant whose station has no series is
            left out, unless its station is artificial.
        zero_gauges: Further stations whose plants get a zero series, on top
            of the artificial gauge.
    """
    lengths = {len(series) for series in natural_by_posto.values()}
    if len(lengths) > 1:
        raise ValueError(f"natural series of different lengths: {sorted(lengths)}")
    length = lengths.pop() if lengths else None

    incremental: dict[int, np.ndarray] = {}
    for code in codes:
        posto = posto_of[code]
        if is_artificial_gauge(posto) or posto in zero_gauges:
            if length is not None:
                incremental[code] = np.zeros(length)
            continue
        natural = natural_by_posto.get(posto)
        if natural is None:
            continue
        upstream_postos = {posto_of[up] for up in upstream_of.get(code, ())}
        upstream_postos.discard(posto)
        series = np.array(natural, dtype=float)
        for up_posto in sorted(upstream_postos):
            if is_artificial_gauge(up_posto):
                continue
            up_natural = natural_by_posto.get(up_posto)
            if up_natural is not None:
                series = series - up_natural
        incremental[code] = series
    return incremental


def artificial_pass_through(
    posto_of: Mapping[int, int],
    upstream_of: Mapping[int, Iterable[int]],
    codes: Iterable[int],
) -> list[tuple[int, int]]:
    """``(gauged plant, artificial upstream plant)`` pairs that may double count.

    The gauged plant subtracts nothing for the artificial plant directly
    above it, yet that plant has a gauged plant somewhere upstream whose
    water also reaches the gauged plant's station. That upstream inflow is
    then counted in both incremental series.
    """
    pairs: list[tuple[int, int]] = []
    for code in sorted(codes):
        if is_artificial_gauge(posto_of[code]):
            continue
        for up in sorted(upstream_of.get(code, ())):
            if is_artificial_gauge(posto_of[up]) and _has_gauged_ancestor(
                up, posto_of, upstream_of
            ):
                pairs.append((code, up))
    return pairs


def _has_gauged_ancestor(
    code: int,
    posto_of: Mapping[int, int],
    upstream_of: Mapping[int, Iterable[int]],
) -> bool:
    seen: set[int] = set()
    pending = list(upstream_of.get(code, ()))
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if not is_artificial_gauge(posto_of[current]):
            return True
        pending.extend(upstream_of.get(current, ()))
    return False


_CATEGORY = "Historical inflows"


def artificial_gauge_plants_diagnostic(
    plants: Iterable[tuple[int, str, int]],
) -> Diagnostic:
    """INFO listing the ``(code, name, hydro_id)`` plants on the artificial gauge."""
    rows = [[code, name, hydro_id] for code, name, hydro_id in plants]
    return Diagnostic(
        code="artificial-gauge-plants",
        severity=Severity.INFO,
        category=_CATEGORY,
        title=f"Plants without a gauge ({len(rows)})",
        summary=(
            f"{len(rows)} plant(s) sit on the artificial gauge (posto "
            f"{ARTIFICIAL_GAUGE}) and get zero natural inflow. Their water "
            "comes only from upstream releases, diversions or pumping."
        ),
        table=DiagnosticTable(columns=["Code", "Name", "Hydro ID"], rows=rows),
    )


def artificial_gauge_data_ignored_diagnostic(source: str) -> Diagnostic:
    """WARNING: ``source`` stores nonzero values for the artificial gauge."""
    return Diagnostic(
        code="artificial-gauge-data-ignored",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title="Inflow values on the artificial gauge ignored",
        summary=(
            f"{source} stores nonzero inflows for posto {ARTIFICIAL_GAUGE}, "
            "the artificial gauge. Plants on it are converted with zero "
            "natural inflow, so those values are not used."
        ),
        remediation=(
            f"Check whether a gauged plant was meant to use posto {ARTIFICIAL_GAUGE}."
        ),
    )


def artificial_pass_through_diagnostic(
    pairs: Iterable[tuple[int, str, int, str]],
) -> Diagnostic:
    """WARNING listing ``(code, name, upstream code, upstream name)`` pairs
    from :func:`artificial_pass_through`."""
    rows = [list(pair) for pair in pairs]
    return Diagnostic(
        code="artificial-gauge-upstream-double-count",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title=f"Upstream inflow possibly counted twice ({len(rows)})",
        summary=(
            "Each listed plant sits directly below a plant on the artificial "
            "gauge that has gauged plants further upstream. Only the plants "
            "directly upstream are subtracted from a plant's natural inflow, "
            "and the artificial plant counts as zero, so the inflow of those "
            "gauged plants may be counted again in the listed plant's "
            "incremental inflow."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Upstream code", "Upstream name"], rows=rows
        ),
        remediation=(
            "Compare the listed plants' incremental inflow with the source "
            "model's reports."
        ),
    )
