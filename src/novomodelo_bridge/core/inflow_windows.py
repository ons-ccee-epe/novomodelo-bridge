"""Calendar-window helpers shared by the windowed inflow emitters."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date


def month_window(year: int, month: int) -> tuple[date, date]:
    """Return the ``[start, end)`` calendar window of ``year``-``month``."""
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return start, end


def previous_months(year: int, month: int, count: int) -> list[tuple[int, int]]:
    """Return the ``count`` calendar months before ``year``-``month``.

    Walks backward starting at the month immediately preceding the given
    one; the result is ordered oldest first.
    """
    months: list[tuple[int, int]] = []
    y, m = year, month
    for _ in range(count):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        months.append((y, m))
    months.reverse()
    return months


def format_observation_windows(
    rows: Iterable[tuple[int, date, date, float]],
) -> list[dict]:
    """Format ``(hydro_id, start, end, value_m3s)`` rows as JSON entries.

    Produces the ``recent_observations`` entry shape (ISO dates), preserving
    the caller's ordering.
    """
    return [
        {
            "hydro_id": hydro_id,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "value_m3s": float(value),
        }
        for hydro_id, start, end, value in rows
    ]
