"""Tests for the cadastro-override calendar resolver.

Pins the ``(mes, semana, ano)`` -> stage-index resolution rules of
``novomodelo_bridge.decomp.converters.cadastro.stage_resolution.resolve_effective_stage``
against a synthetic two-month operative calendar (two July weekly stages +
one August monthly stage), mirroring the calendar shape the source model's
operative weeks and months produce. ``mes`` is exercised in its real
representation — a 3-letter Portuguese month-abbreviation string, with an
empty string for a blank month, and ``semana``/``ano`` as floats or NaN —
plus a synthetic int/float ``mes`` back-compat case.
"""

from __future__ import annotations

from datetime import date

import pytest

from novomodelo_bridge.decomp.converters.cadastro import resolve_effective_stage
from novomodelo_bridge.decomp.temporal import OperativeStage, build_operative_calendar


@pytest.fixture
def calendar() -> list[OperativeStage]:
    """Stages 0, 1 = July weekly stages; stage 2 = the August monthly stage."""
    stage_block_hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), stage_block_hours)


def test_blank_month_is_stage_zero(calendar: list[OperativeStage]) -> None:
    assert resolve_effective_stage(None, None, None, calendar) == 0


def test_empty_string_month_is_stage_zero(calendar: list[OperativeStage]) -> None:
    assert resolve_effective_stage("", float("nan"), float("nan"), calendar) == 0


def test_string_month_semana_one_is_first_july_stage(
    calendar: list[OperativeStage],
) -> None:
    assert resolve_effective_stage("JUL", 1.0, 2026.0, calendar) == 0


def test_string_month_semana_two_is_second_july_stage(
    calendar: list[OperativeStage],
) -> None:
    assert resolve_effective_stage("JUL", 2.0, 2026.0, calendar) == 1


def test_string_month_blank_semana_is_monthly_stage(
    calendar: list[OperativeStage],
) -> None:
    assert resolve_effective_stage("AGO", None, 2026.0, calendar) == 2


def test_int_month_matches_string_month_resolution(
    calendar: list[OperativeStage],
) -> None:
    """Synthetic int/float ``mes`` callers stay supported (back-compat)."""
    assert resolve_effective_stage(7, 1, 2026, calendar) == resolve_effective_stage(
        "JUL", 1.0, 2026.0, calendar
    )
    assert resolve_effective_stage(
        7.0, 2.0, 2026.0, calendar
    ) == resolve_effective_stage("JUL", 2.0, 2026.0, calendar)
    assert resolve_effective_stage(8, None, 2026, calendar) == resolve_effective_stage(
        "AGO", None, 2026.0, calendar
    )


def test_semana_clamps_to_monthly_stage(calendar: list[OperativeStage]) -> None:
    assert resolve_effective_stage("AGO", 3.0, 2026.0, calendar) == 2


def test_month_before_horizon_is_stage_zero(calendar: list[OperativeStage]) -> None:
    assert resolve_effective_stage("JUN", None, 2026.0, calendar) == 0


def test_month_after_horizon_is_none(calendar: list[OperativeStage]) -> None:
    assert resolve_effective_stage("OUT", None, 2026.0, calendar) is None


def test_unrecognized_month_raises_value_error(
    calendar: list[OperativeStage],
) -> None:
    with pytest.raises(ValueError, match="XYZ"):
        resolve_effective_stage("XYZ", 1.0, 2026.0, calendar)
