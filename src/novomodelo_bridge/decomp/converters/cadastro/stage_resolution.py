"""Effective-stage calendar resolution for cadastro override registers.

Cadastro override registers in the source model can carry an optional
``(mes, semana, ano)`` triple that makes an override effective from a given
stage forward; a blank triple means the override is effective from the
initial stage. This module provides the single pure function that resolves
that triple to a 0-based operative-calendar stage index, the same way the
source model's own temporal overrides resolve to per-stage effective values
(see ``_TEMPORAL_OVERRIDE_TYPES`` in the newave hydro converters).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cobre_bridge.decomp.temporal import OperativeStage


_MONTH_ABBR_TO_INT: dict[str, int] = {
    "JAN": 1,
    "FEV": 2,
    "MAR": 3,
    "ABR": 4,
    "MAI": 5,
    "JUN": 6,
    "JUL": 7,
    "AGO": 8,
    "SET": 9,
    "OUT": 10,
    "NOV": 11,
    "DEZ": 12,
}


def _parse_month(mes: str | int | float | None) -> int | None:
    """Normalize an ``AC`` ``mes`` field to a 1..12 month, or ``None`` when blank.

    Blank (``None``, NaN, or an empty/whitespace-only string after
    ``strip()``) resolves to ``None``. An integer/float is coerced with
    ``int(...)``. A non-blank string is matched case-insensitively against
    the source model's 3-letter month abbreviations. A non-blank value that
    is neither a valid month int (1..12) nor a known abbreviation raises
    ``ValueError`` naming the value (a malformed deck is a hard error, never
    a silent default).
    """
    if mes is None:
        return None
    if isinstance(mes, str):
        stripped = mes.strip()
        if not stripped:
            return None
        month = _MONTH_ABBR_TO_INT.get(stripped.upper())
        if month is None:
            raise ValueError(f"unrecognized AC month value: {mes!r}")
        return month
    if pd.isna(mes):
        return None
    month = int(mes)
    if not 1 <= month <= 12:
        raise ValueError(f"unrecognized AC month value: {mes!r}")
    return month


def resolve_effective_stage(
    mes: str | int | float | None,
    semana: int | float | None,
    ano: int | float | None,
    calendar: Sequence[OperativeStage],
) -> int | None:
    """Resolve a ``(mes, semana, ano)`` triple to a 0-based stage index.

    A blank ``mes`` (per :func:`_parse_month`: ``None``, NaN, or an
    empty/whitespace-only string) means the override is effective from the
    initial stage (index ``0``). Otherwise ``(ano, mes)`` is resolved to an
    operative month against *calendar*: a blank ``ano`` defaults to the
    calendar's first stage's year, a month strictly before the calendar's
    horizon also resolves to the initial stage, and a month strictly after
    the horizon resolves to ``None`` for the caller to report as
    out-of-horizon. ``semana`` selects a 1-based operative week within that
    resolved month; a blank ``semana`` (``None``, NaN, or ``0``) resolves to
    the month's first stage, and any other value clamps to the month's last
    available stage (a month represented only by an aggregated stage maps
    every ``semana`` to that one stage).

    Raises
    ------
    ValueError
        If ``mes`` is a non-blank value that is neither a 1..12 int nor a
        known month abbreviation (see :func:`_parse_month`), or if
        ``semana``/``ano`` cannot be coerced to ``int`` (a malformed deck is
        a hard error, never a silent default).
    """
    month = _parse_month(mes)
    if month is None:
        return 0

    resolved_ano = (
        calendar[0].start_date.year if ano is None or pd.isna(ano) else int(ano)
    )

    target = resolved_ano * 12 + (month - 1)
    month_stages = [
        stage
        for stage in calendar
        if stage.season_id == month - 1 and stage.start_date.year == resolved_ano
    ]

    if not month_stages:
        first_ordinal = calendar[0].start_date.year * 12 + calendar[0].season_id
        last_ordinal = calendar[-1].start_date.year * 12 + calendar[-1].season_id
        if target < first_ordinal:
            return 0
        if target > last_ordinal:
            return None

    if semana is None or pd.isna(semana) or semana == 0:
        return month_stages[0].index

    week_index = int(semana) - 1
    return month_stages[min(week_index, len(month_stages) - 1)].index
