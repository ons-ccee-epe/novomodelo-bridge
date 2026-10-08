"""Tests for the ``VI`` water travel-time converter (``decomp/travel_time.py``).

Tier-1 only: synthetic ``Dadger``/``EffectiveCadastro`` doubles, no real deck.
Covers the ``VI`` reader, the ``past_defluences`` window tiling (novomodelo coverage
rule 5 / no-future-dating rule 5b), and the operated/downstream filtering.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from novomodelo_bridge.decomp.converters.travel_time import (
    TravelTime,
    _defluence_windows,
    convert_travel_time,
    read_travel_times,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap

_START = date(2026, 3, 14)  # a Saturday; 168 h (one operative week) per prior window


class _FakeDadger:
    """A ``Dadger`` double exposing only ``.vi(df=True)``."""

    def __init__(self, vi: pd.DataFrame | None) -> None:
        self._vi = vi

    def vi(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._vi


class _FakeEffective:
    """An ``EffectiveCadastro`` double: a static downstream map + index."""

    def __init__(self, downstream: dict[int, int], codes: list[int]) -> None:
        self._downstream = downstream
        self.base = pd.DataFrame(index=pd.Index(codes, name="codigo_usina"))

    def downstream_plant_varies(self, code: int) -> bool:  # noqa: ARG002
        return False

    def downstream_plant(self, code: int, stage_index: int) -> int:  # noqa: ARG002
        return self._downstream.get(code, 0)


def _vi_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _id_map() -> DecompIdMap:
    # hydro codes 156/162 (VI plants) drain to 111 (SOBRADINHO-like sink arc).
    return DecompIdMap(
        bus_codes=(1,),
        bus_names=("SE",),
        hydro_codes=(111, 156, 162),
    )


# ---------------------------------------------------------------------------
# read_travel_times
# ---------------------------------------------------------------------------


def test_read_travel_times_parses_duration_and_flows() -> None:
    vi = _vi_frame(
        [
            {
                "codigo_usina": 156,
                "duracao": 360,
                "vazao_1": 490.0,
                "vazao_2": 444.0,
                "vazao_3": 423.0,
                "vazao_4": 346.0,
                "vazao_5": 427.0,
            }
        ]
    )
    got = read_travel_times(_FakeDadger(vi))
    assert got[156] == TravelTime(
        code=156,
        travel_time_hours=360.0,
        prior_flows_m3s=(490.0, 444.0, 423.0, 346.0, 427.0),
    )


def test_read_travel_times_empty_when_no_vi() -> None:
    assert read_travel_times(_FakeDadger(None)) == {}
    assert read_travel_times(_FakeDadger(pd.DataFrame())) == {}


def test_read_travel_times_drops_nan_trailing_flows() -> None:
    vi = _vi_frame(
        [
            {
                "codigo_usina": 162,
                "duracao": 168,
                "vazao_1": 27.0,
                "vazao_2": 30.0,
                "vazao_3": float("nan"),
                "vazao_4": float("nan"),
                "vazao_5": float("nan"),
            }
        ]
    )
    assert read_travel_times(_FakeDadger(vi))[162].prior_flows_m3s == (27.0, 30.0)


# ---------------------------------------------------------------------------
# _defluence_windows
# ---------------------------------------------------------------------------


def test_defluence_windows_tile_transit_span_exactly() -> None:
    """360 h over 168 h weeks -> 7 d + 7 d + 1 d, tiling [start-15d, start)."""
    tt = TravelTime(156, 360.0, (490.0, 444.0, 423.0, 346.0, 427.0))
    windows = _defluence_windows(tt, hydro_id=9, start_date=_START, week_hours=168.0)

    assert windows == [
        {
            "hydro_id": 9,
            "start_date": "2026-03-07",
            "end_date": "2026-03-14",
            "value_m3s": 490.0,
        },
        {
            "hydro_id": 9,
            "start_date": "2026-02-28",
            "end_date": "2026-03-07",
            "value_m3s": 444.0,
        },
        {
            "hydro_id": 9,
            "start_date": "2026-02-27",
            "end_date": "2026-02-28",
            "value_m3s": 423.0,
        },
    ]
    # Contiguous (no gap), most-recent first, and the newest window ends at the
    # study start (rule 5b: nothing future-dated); union covers 15 d == 360 h.
    assert windows[0]["end_date"] == _START.isoformat()
    for earlier, later in zip(windows[1:], windows):
        assert earlier["end_date"] == later["start_date"]
    assert windows[-1]["start_date"] == "2026-02-27"  # start - 15 days


def test_defluence_windows_reuse_last_flow_when_span_exceeds_history() -> None:
    """A travel time needing more windows than declared flows reuses the oldest
    flow rather than leaving a coverage gap."""
    tt = TravelTime(1, 504.0, (10.0, 20.0))  # 21 d, only 2 weekly flows
    windows = _defluence_windows(tt, hydro_id=1, start_date=_START, week_hours=168.0)
    assert [w["value_m3s"] for w in windows] == [10.0, 20.0, 20.0]
    assert windows[-1]["start_date"] == "2026-02-21"  # start - 21 days, gap-free


# ---------------------------------------------------------------------------
# convert_travel_time
# ---------------------------------------------------------------------------


def test_convert_travel_time_emits_hours_and_past_defluences() -> None:
    vi = _vi_frame(
        [
            {
                "codigo_usina": 156,
                "duracao": 360,
                "vazao_1": 490.0,
                "vazao_2": 444.0,
                "vazao_3": 423.0,
                "vazao_4": 346.0,
                "vazao_5": 427.0,
            },
            {
                "codigo_usina": 162,
                "duracao": 360,
                "vazao_1": 27.0,
                "vazao_2": 30.0,
                "vazao_3": 31.0,
                "vazao_4": 18.0,
                "vazao_5": 16.0,
            },
        ]
    )
    effective = _FakeEffective({156: 111, 162: 111}, [111, 156, 162])
    id_map = _id_map()

    hours, defluences = convert_travel_time(
        _FakeDadger(vi), id_map, effective, _START, week_hours=168.0
    )

    assert hours == {156: 360.0, 162: 360.0}
    # Three windows per VI plant, keyed by the plant's novomodelo id.
    id156, id162 = id_map.hydro_id(156), id_map.hydro_id(162)
    assert [d["hydro_id"] for d in defluences] == [
        id156,
        id156,
        id156,
        id162,
        id162,
        id162,
    ]
    assert [d["value_m3s"] for d in defluences[:3]] == [490.0, 444.0, 423.0]


def test_convert_travel_time_skips_plant_without_downstream() -> None:
    vi = _vi_frame(
        [
            {
                "codigo_usina": 156,
                "duracao": 360,
                "vazao_1": 490.0,
                "vazao_2": 444.0,
                "vazao_3": 423.0,
                "vazao_4": 346.0,
                "vazao_5": 427.0,
            }
        ]
    )
    effective = _FakeEffective({156: 0}, [111, 156, 162])  # 0 == sink, no arc
    hours, defluences = convert_travel_time(
        _FakeDadger(vi), _id_map(), effective, _START, week_hours=168.0
    )
    assert hours == {}
    assert defluences == []


def test_convert_travel_time_skips_unoperated_plant() -> None:
    vi = _vi_frame(
        [
            {
                "codigo_usina": 999,
                "duracao": 360,
                "vazao_1": 490.0,
                "vazao_2": 444.0,
                "vazao_3": 423.0,
                "vazao_4": 346.0,
                "vazao_5": 427.0,
            }
        ]
    )
    effective = _FakeEffective({999: 111}, [111, 156, 162, 999])
    hours, defluences = convert_travel_time(
        _FakeDadger(vi), _id_map(), effective, _START, week_hours=168.0
    )
    assert hours == {}  # 999 is not in id_map.hydro_codes
    assert defluences == []
