"""Water travel-time conversion for DECOMP-like decks (``VI`` register).

The source model's ``VI`` register declares, per plant, the travel time (in
hours) of that plant's defluent flow down the cascade arc to its downstream
plant, plus that plant's defluent flow in the (up to five) weeks of the month
before the study start — ordered most-recent first (manual §3.4.6.6). A plant
without a ``VI`` register has an instantaneous arc (default travel time nil).

cobre models the delayed arc with ``Hydro.travel_time_hours`` and seeds the
water already in transit at the study start with
``initial_conditions.past_defluences`` — one ``[start_date, end_date)`` window
per pre-study release period, keyed by the upstream (releasing) plant. cobre's
config-time validator (``travel_time.rs`` rule 5) requires those windows to
**cover** the arc's in-transit span ``[start_0 − travel_time, start_0)`` with no
gap and none future-dated (rule 5b), so this module tiles exactly that span
with the ``VI`` weekly flows, most-recent window first.

Only operated plants with a downstream cascade arc get a travel-time arc; a
``VI`` row for a plant that is unoperated or has no downstream is dropped with a
warning (there is no arc to delay).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import pandas as pd

from cobre_bridge.decomp.converters.hydro import _downstream_operated

if TYPE_CHECKING:
    from datetime import date

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.converters.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.id_map import DecompIdMap

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


def read_travel_times(dadger: Dadger) -> dict[int, TravelTime]:
    """Read the ``VI`` registers into ``{plant code: TravelTime}``.

    Returns an empty map when the deck declares no ``VI`` register.
    """
    vi = dadger.vi(df=True)
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
    dadger: Dadger,
    id_map: DecompIdMap,
    effective: EffectiveCadastro,
    start_date: date,
    week_hours: float,
) -> tuple[dict[int, float], list[dict]]:
    """Resolve the ``VI`` registers into cobre travel-time inputs.

    Returns ``(travel_time_hours_by_code, past_defluences)`` where the first is
    the ``{plant code: travel_time_hours}`` map :func:`~cobre_bridge.decomp.
    hydro.convert_hydros` stamps onto each arc plant's ``hydros.json`` entry,
    and the second is the ``initial_conditions.past_defluences`` list seeding
    the in-transit water. Both cover the same set of plants — those operated
    **and** carrying a downstream cascade arc — so cobre's coverage rule sees a
    seed for every declared arc.
    """
    travel_times = read_travel_times(dadger)
    if not travel_times:
        return {}, []

    operated = set(id_map.hydro_codes)
    hours_by_code: dict[int, float] = {}
    past_defluences: list[dict] = []
    for code in sorted(travel_times):
        travel_time = travel_times[code]
        if code not in operated:
            _LOG.warning(
                "VI travel-time plant %d is not in the operated set; skipped",
                code,
            )
            continue
        if _downstream_operated(effective, code, operated) is None:
            _LOG.warning(
                "VI travel-time plant %d has no downstream cascade arc; "
                "travel time and in-transit water skipped",
                code,
            )
            continue
        hours_by_code[code] = travel_time.travel_time_hours
        past_defluences.extend(
            _defluence_windows(
                travel_time, id_map.hydro_id(code), start_date, week_hours
            )
        )
    return hours_by_code, past_defluences
