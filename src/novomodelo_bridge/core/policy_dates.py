"""Policy-checkpoint dates: the `YYYYMMDD` integer codec and signed spans in hours.

novomodelo writes every per-slot date of a policy checkpoint as an integer
`YYYYMMDD`, or as `UNDATED` (`i32::MIN`, novomodelo's `ENTITY_SLOT_DATE_SENTINEL`
and `STAGE_CUTS_PRICED_STATE_DATE_SENTINEL`) when the slot carries no date: on
the families a date does not apply to, and on a transit bucket whose arrival
would fall past the study end (novomodelo's structural pad).
"""

from __future__ import annotations

from datetime import date

UNDATED: int = -(2**31)


def date_from_yyyymmdd(value: int) -> date:
    """Decode a `YYYYMMDD` integer.

    Raises:
        ValueError: *value* is `UNDATED` or not a real calendar date.
    """
    if value == UNDATED:
        raise ValueError(
            f"{value} is the policy's no-date sentinel, not a YYYYMMDD date"
        )
    year, rest = divmod(value, 10_000)
    month, day = divmod(rest, 100)
    return date(year, month, day)


def yyyymmdd_from_date(day: date) -> int:
    """Encode *day* as a `YYYYMMDD` integer, the inverse of `date_from_yyyymmdd`."""
    return day.year * 10_000 + day.month * 100 + day.day


def hours_between(start: date, end: date) -> float:
    """Hours from *start* to *end*; signed, negative when *end* precedes *start*."""
    return float((end - start).days * 24)
