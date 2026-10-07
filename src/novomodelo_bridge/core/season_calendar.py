"""Canonical block-name and season-label conventions shared across the
conversion tracks."""

from __future__ import annotations

# Default block names by number of blocks.
_SINGLE_BLOCK_NAMES = ["SINGLE"]
_TWO_BLOCK_NAMES = ["HEAVY", "LIGHT"]
_THREE_BLOCK_NAMES = ["HEAVY", "MEDIUM", "LIGHT"]


def block_names(n: int) -> list[str]:
    """Return a canonical list of block names for *n* blocks.

    Falls back to ``"BLOCK_0"``, ``"BLOCK_1"``, … for uncommon counts.
    """
    if n == 1:
        return _SINGLE_BLOCK_NAMES
    if n == 2:
        return _TWO_BLOCK_NAMES
    if n == 3:
        return _THREE_BLOCK_NAMES
    return [f"BLOCK_{i}" for i in range(n)]


MONTH_LABELS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def monthly_season_definitions() -> dict:
    """Return the canonical calendar-monthly ``season_definitions`` block.

    Twelve seasons with 0-based ids (Jan=0 … Dec=11), shared by every
    converter family so the season convention has a single source.
    """
    return {
        "cycle_type": "monthly",
        "seasons": [
            {"id": i, "month_start": i + 1, "label": MONTH_LABELS[i]} for i in range(12)
        ],
    }
