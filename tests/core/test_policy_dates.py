"""Tier-1 tests for the policy-date codec and the signed hours span."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from novomodelo_bridge.core.policy_dates import (
    UNDATED,
    date_from_yyyymmdd,
    hours_between,
    yyyymmdd_from_date,
)


class TestDateFromYyyymmdd:
    def test_decodes_a_regular_date(self) -> None:
        assert date_from_yyyymmdd(20260822) == date(2026, 8, 22)

    def test_decodes_a_leap_day(self) -> None:
        assert date_from_yyyymmdd(20240229) == date(2024, 2, 29)

    def test_raises_on_the_undated_sentinel(self) -> None:
        with pytest.raises(ValueError, match="no-date sentinel"):
            date_from_yyyymmdd(UNDATED)

    def test_raises_on_an_impossible_date(self) -> None:
        with pytest.raises(ValueError):
            date_from_yyyymmdd(20260230)

    def test_undated_is_novomodelos_i32_min(self) -> None:
        assert UNDATED == -2147483648


class TestYyyymmddFromDate:
    @pytest.mark.parametrize(
        ("day", "expected"),
        [(date(2026, 8, 22), 20260822), (date(2024, 2, 29), 20240229)],
    )
    def test_encodes_year_month_day(self, day: date, expected: int) -> None:
        assert yyyymmdd_from_date(day) == expected

    def test_round_trips_every_day_over_four_years(self) -> None:
        day = date(2024, 1, 1)
        while day <= date(2027, 12, 31):
            assert date_from_yyyymmdd(yyyymmdd_from_date(day)) == day
            day += timedelta(days=1)

    @pytest.mark.parametrize("value", [20240229, 20261231, 20270101])
    def test_integer_round_trips(self, value: int) -> None:
        assert yyyymmdd_from_date(date_from_yyyymmdd(value)) == value


class TestHoursBetween:
    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            pytest.param(date(2026, 8, 22), date(2026, 8, 29), 168.0, id="week"),
            pytest.param(
                date(2026, 9, 5), date(2026, 10, 1), 624.0, id="month-crossing"
            ),
            pytest.param(date(2024, 2, 1), date(2024, 3, 1), 696.0, id="leap-february"),
            pytest.param(
                date(2026, 12, 26), date(2027, 1, 2), 168.0, id="year-boundary"
            ),
            pytest.param(date(2026, 8, 29), date(2026, 8, 22), -168.0, id="reversed"),
            pytest.param(date(2026, 8, 22), date(2026, 8, 22), 0.0, id="same-day"),
        ],
    )
    def test_spans_whole_days_in_hours(
        self, start: date, end: date, expected: float
    ) -> None:
        assert hours_between(start, end) == expected
