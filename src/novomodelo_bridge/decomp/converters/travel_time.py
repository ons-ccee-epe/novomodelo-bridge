"""Water travel-time conversion for DECOMP-like decks (``VI`` register).

The source model's ``VI`` register declares, per plant, the travel time (in
hours) of that plant's defluent flow down the cascade arc to its downstream
plant, plus that plant's defluent flow in the (up to five) weeks of the month
before the study start — ordered most-recent first (manual §3.4.6.6). A plant
without a ``VI`` register has an instantaneous arc (default travel time nil).

novomodelo models the delayed arc with ``Hydro.travel_time_hours`` and seeds the
water already in transit at the study start with
``hydros[].past_defluences`` of ``boundary_conditions.json`` — one
``[start_date, end_date)`` window per pre-study release period, under the
upstream (releasing) plant. novomodelo's config-time validator
(``travel_time.rs`` rule 5) requires those windows to **cover** the arc's
in-transit span ``[start_0 − travel_time, start_0)`` with no gap and none future-dated
(rule 5b), so this module tiles exactly that span with the ``VI`` weekly flows,
most-recent window first.

Only converted plants with a downstream cascade arc get a travel-time arc; a
``VI`` row for a plant the case leaves out or that has no downstream is dropped
and reported (there is no arc to delay).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Protocol

import pandas as pd

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.core.pandas_utils import as_frame
from novomodelo_bridge.decomp.converters.hydro import _downstream_in

if TYPE_CHECKING:
    from datetime import date

    from novomodelo_bridge.decomp.converters.hydro.entity import CascadeLinks
    from novomodelo_bridge.decomp.id_map import DecompIdMap

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TravelTime:
    """One plant's ``VI`` record: arc travel time plus its pre-study releases.

    ``prior_flows_m3s`` is ordered most-recent first (``vazao_1`` is the week
    just before the study start), matching the ``VI`` register's ordering.
    """

    code: int
    travel_time_hours: float
    prior_flows_m3s: tuple[float, ...]


class _ViRegisters(Protocol):
    def vi(self, *, df: bool) -> object: ...


def read_travel_times(dadger: _ViRegisters) -> dict[int, TravelTime]:
    """Read the ``VI`` registers into ``{plant code: TravelTime}``.

    Returns an empty map when the deck declares no ``VI`` register.
    """
    vi = as_frame(dadger.vi(df=True))
    if vi is None or vi.empty:
        return {}
    flow_columns = sorted(
        (c for c in vi.columns if c.startswith("vazao_")),
        key=lambda c: int(c.split("_")[1]),
    )
    travel_times: dict[int, TravelTime] = {}
    for _, row in vi.iterrows():
        flows = tuple(float(row[c]) for c in flow_columns if not pd.isna(row[c]))
        code = int(row["codigo_usina"])
        travel_times[code] = TravelTime(
            code=code,
            travel_time_hours=float(row["duracao"]),
            prior_flows_m3s=flows,
        )
    return travel_times


def _defluence_windows(
    travel_time: TravelTime,
    hydro_id: int,
    start_date: date,
    week_hours: float,
) -> list[dict]:
    """The ``past_defluences`` windows tiling ``[start_0 − t_v, start_0)``.

    Walks backward from *start_date* in *week_hours*-wide steps (the study's
    first-stage duration, i.e. one operative week), truncating the final step
    at the travel-time span so the union covers the in-transit span exactly —
    no gap (rule 5) and no window ending after the study start (rule 5b). Each
    window takes the corresponding ``VI`` weekly flow, most-recent first; if the
    span needs more windows than the register supplies, the oldest declared flow
    is reused to keep the coverage gap-free.
    """
    t_v_days = max(1, round(travel_time.travel_time_hours / 24.0))
    week_days = max(1, round(week_hours / 24.0))
    flows = travel_time.prior_flows_m3s

    windows: list[dict] = []
    cursor = start_date
    remaining = t_v_days
    index = 0
    while remaining > 0:
        span = min(week_days, remaining)
        window_start = cursor - timedelta(days=span)
        flow = flows[index] if index < len(flows) else (flows[-1] if flows else 0.0)
        windows.append(
            {
                "hydro_id": hydro_id,
                "start_date": window_start.isoformat(),
                "end_date": cursor.isoformat(),
                "value_m3s": flow,
            }
        )
        cursor = window_start
        remaining -= span
        index += 1
    return windows


def convert_travel_time(
    dadger: _ViRegisters,
    id_map: DecompIdMap,
    effective: CascadeLinks,
    start_date: date,
    week_hours: float,
) -> tuple[dict[int, float], list[dict]]:
    """Resolve the ``VI`` registers into novomodelo travel-time inputs.

    Returns ``(travel_time_hours_by_code, past_defluences)`` where the first is
    the ``{plant code: travel_time_hours}`` map :func:`~novomodelo_bridge.decomp.
    hydro.convert_hydros` stamps onto each arc plant's ``hydros.json`` entry,
    and the second is the ``hydros[].past_defluences`` list seeding
    the in-transit water. Both cover the same set of plants — those converted
    **and** carrying a downstream cascade arc — so novomodelo's coverage rule sees a
    seed for every declared arc.
    """
    travel_times = read_travel_times(dadger)
    if not travel_times:
        return {}, []

    converted = set(id_map.hydro_codes)
    hours_by_code: dict[int, float] = {}
    past_defluences: list[dict] = []
    unused: list[list[object]] = []
    for code in sorted(travel_times):
        travel_time = travel_times[code]
        if code not in converted:
            unused.append([code, _plant_name(effective, code), "not in the case"])
            continue
        if _downstream_in(effective, code, converted) is None:
            unused.append([code, _plant_name(effective, code), "no plant downstream"])
            continue
        hours_by_code[code] = travel_time.travel_time_hours
        past_defluences.extend(
            _defluence_windows(
                travel_time, id_map.hydro_id(code), start_date, week_hours
            )
        )
    if unused:
        emit(_travel_time_unused_diagnostic(unused), logger=_LOG)
    return hours_by_code, past_defluences


def _plant_name(effective: CascadeLinks, code: int) -> str:
    base = effective.base
    return str(base.loc[code, "nome_usina"]).strip() if code in base.index else ""


def _travel_time_unused_diagnostic(rows: list[list[object]]) -> Diagnostic:
    return Diagnostic(
        code="decomp-vi-travel-time-unused",
        severity=Severity.INFO,
        category="Travel time",
        title=f"Travel times with no outflow to delay ({len(rows)})",
        summary=(
            "The listed plants carry a VI register but are left out of the case "
            "or have no plant downstream, so there is no outflow for the travel "
            "time to delay and the record has no effect, as in DECOMP."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Reason"],
            rows=rows,
            justify=["right", "left", "left"],
        ),
    )
