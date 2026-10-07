"""Unit tests for the DECOMP pre-study inflow-observation seed
(``convert_recent_observation_windows``) — the ``recent_observations`` windows
that seed novomodelo's PAR inflow-lag accumulator.

Tier 1 — pure Python. ``_incremental_context`` is patched so the test pins the
window/date construction (the new logic) rather than the posto→plant topology,
which its own module already covers.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.scenarios import convert_recent_observation_windows
from novomodelo_bridge.decomp.temporal import build_operative_calendar

# Study starts 2026-03-14 (Saturday); final stage ends at the 2026-05-01 month
# boundary — the mar-26 reduced-case shape.
_HOURS = [
    [30.0, 74.0, 64.0],
    [30.0, 74.0, 64.0],
    [24.0, 67.0, 77.0],
    [108.0, 261.0, 279.0],
]


def _calendar():
    return build_operative_calendar(date(2026, 3, 14), _HOURS)


def _id_map() -> DecompIdMap:
    # one operated hydro, code 10, gauged at posto column "1"
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(10,))


def _vazoes(monthly: pd.DataFrame | None, weekly: pd.DataFrame | None):
    return SimpleNamespace(observacoes_mensais=monthly, observacoes_semanais=weekly)


def _run(monthly, weekly):
    # station_by_code maps hydro code 10 -> posto column "1"; no parents.
    with patch(
        "novomodelo_bridge.decomp.scenarios._incremental_context",
        return_value=({10: "1"}, {}),
    ):
        return convert_recent_observation_windows(
            _vazoes(monthly, weekly), object(), _id_map(), _calendar()
        )


class TestRecentObservationWindows:
    def test_monthly_and_weekly_windows_non_overlapping(self) -> None:
        monthly = pd.DataFrame({"mes": [1, 2, 3], "1": [50.0, 60.0, 70.0]})
        weekly = pd.DataFrame({"semana": [1, 2], "1": [80.0, 90.0]})
        out = _run(monthly, weekly)
        # 3 monthly (Dec/Jan/Feb 2026, Feb clipped to Feb-28) + 2 weekly.
        windows = [(o["start_date"], o["end_date"], o["value_m3s"]) for o in out]
        assert windows == [
            ("2025-12-01", "2026-01-01", 50.0),  # mes=1 -> Dec 2025 (oldest)
            ("2026-01-01", "2026-02-01", 60.0),  # mes=2 -> Jan 2026
            ("2026-02-01", "2026-02-28", 70.0),  # mes=3 -> Feb 2026, clipped
            ("2026-02-28", "2026-03-07", 80.0),  # semana=1 (oldest)
            ("2026-03-07", "2026-03-14", 90.0),  # semana=2, ends at study start
        ]
        # non-overlapping, adjacent (each start == previous end)
        for prev, nxt in zip(windows, windows[1:]):
            assert prev[1] == nxt[0]

    def test_weekly_only_ends_at_study_start(self) -> None:
        weekly = pd.DataFrame({"semana": [1, 2], "1": [80.0, 90.0]})
        out = _run(None, weekly)
        assert [(o["start_date"], o["end_date"]) for o in out] == [
            ("2026-02-28", "2026-03-07"),
            ("2026-03-07", "2026-03-14"),
        ]

    def test_monthly_only_full_windows(self) -> None:
        monthly = pd.DataFrame({"mes": [1, 2], "1": [50.0, 60.0]})
        out = _run(monthly, None)
        # no weekly floor -> months are full calendar windows up to the study month
        assert [(o["start_date"], o["end_date"]) for o in out] == [
            ("2026-01-01", "2026-02-01"),
            ("2026-02-01", "2026-03-01"),
        ]

    def test_no_observations_returns_empty(self) -> None:
        assert _run(None, None) == []
        assert _run(pd.DataFrame(), pd.DataFrame()) == []
