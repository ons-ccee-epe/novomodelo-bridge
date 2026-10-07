"""Tests for the canonical study-horizon module."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from novomodelo_bridge.core.tolerances import BIG_M
from novomodelo_bridge.newave.horizon import (
    POST_STUDY_YEAR,
    build_stage_dates,
    historical_start_date,
    seasonal_step_function,
    stage_dates_for,
    study_horizon,
)


def test_historical_start_date_is_jan_1_of_ano_inicial_historico() -> None:
    dger = SimpleNamespace(ano_inicial_historico=1931)
    assert historical_start_date(dger) == "1931-01-01"


def test_historical_start_date_falls_back_when_absent() -> None:
    dger = SimpleNamespace(ano_inicial_historico=None)
    assert historical_start_date(dger) == "1931-01-01"


def _dger(start_year=2024, start_month=9, num_anos=3, num_anos_pos=3):
    return SimpleNamespace(
        ano_inicio_estudo=start_year,
        mes_inicio_estudo=start_month,
        num_anos_estudo=num_anos,
        num_anos_pos_estudo=num_anos_pos,
    )


def test_sentinels():
    assert POST_STUDY_YEAR == 9999
    assert BIG_M == 99990.0


def test_study_horizon_example_case():
    # Sept-start, 3 study years, 3 post-study years (the example case shape).
    h = study_horizon(_dger())
    assert h.start_year == 2024
    assert h.start_month == 9
    assert h.study_months == (13 - 9) + (3 - 1) * 12  # 28
    assert h.total_stages == 28 + 3 * 12  # 64
    assert h.last_study_stage == 27
    assert h.first_year_stages == 4
    assert h.pos_months == 36


def test_january_start_has_full_first_year():
    h = study_horizon(_dger(start_month=1, num_anos=5, num_anos_pos=0))
    assert h.study_months == 12 + (5 - 1) * 12  # 60
    assert h.total_stages == 60
    assert h.first_year_stages == 12


@pytest.mark.parametrize(
    "num_anos,num_anos_pos",
    [(None, None), (0, 0)],
)
def test_falsy_year_counts_default(num_anos, num_anos_pos):
    h = study_horizon(_dger(num_anos=num_anos, num_anos_pos=num_anos_pos))
    assert h.num_anos == 1  # falls back to one study year
    assert h.num_anos_pos == 0
    assert h.study_months == 13 - 9
    assert h.total_stages == h.study_months


def test_is_post_study_boundary():
    h = study_horizon(_dger())  # study_months == 28
    assert not h.is_post_study(27)  # last study stage
    assert h.is_post_study(28)  # first post-study stage
    assert h.is_post_study(63)


def test_build_stage_dates_wraps_year():
    dates = build_stage_dates(2024, 11, 4)
    assert dates == [
        date(2024, 11, 1),
        date(2024, 12, 1),
        date(2025, 1, 1),
        date(2025, 2, 1),
    ]


def test_stage_dates_for_matches_horizon_length():
    h = study_horizon(_dger())
    dates = stage_dates_for(h)
    assert len(dates) == h.total_stages
    assert dates[0] == date(2024, 9, 1)


def test_seasonal_step_function_forward_fills_and_clears_on_big_m():
    h = study_horizon(_dger(num_anos_pos=0))  # study-only, 28 stages
    # value 100 from Sept-2024 (stage 0); big-M at Jan-2025 (stage 4) clears it.
    recs = [(2024, 9, 100.0), (2025, 1, BIG_M)]
    out = seasonal_step_function(recs, lambda v: v, seasonalize=False, horizon=h)
    assert out[0] == 100.0
    assert out[3] == 100.0  # forward-filled through Dec-2024
    assert 4 not in out  # big-M cleared the fill from Jan-2025 on


def test_seasonal_step_function_freeze_vs_seasonal_post_study():
    h = study_horizon(_dger())  # study_months 28, total 64
    recs = [(2024, 9, 10.0), (2025, 1, 50.0)]  # 10 then 50 from Jan onward
    frozen = seasonal_step_function(recs, lambda v: v, seasonalize=False, horizon=h)
    seasonal = seasonal_step_function(recs, lambda v: v, seasonalize=True, horizon=h)
    # freeze: every post-study stage holds the last study stage's value
    assert frozen[h.study_months - 1] == 50.0
    assert all(frozen[s] == 50.0 for s in range(h.study_months, h.total_stages))
    # seasonal: post-study repeats the last study year's monthly pattern (all 50)
    assert all(seasonal[s] == 50.0 for s in range(h.study_months, h.total_stages))


def test_seasonal_step_function_empty_recs():
    h = study_horizon(_dger())
    assert seasonal_step_function([], lambda v: v, seasonalize=False, horizon=h) == {}


def test_seasonal_step_function_pre_holds_until_first_study_record():
    # Jan-start, 1 study year, no post-study. PRE steps Mar=100, Nov=200 over a
    # repeating year: December (the month before the start) is in force at 200.
    h = study_horizon(_dger(start_month=1, num_anos=1, num_anos_pos=0))
    out = seasonal_step_function(
        [(2024, 6, 300.0)],
        lambda v: v,
        seasonalize=False,
        horizon=h,
        pre_recs=[(3, 100.0), (11, 200.0)],
    )
    assert [out[s] for s in range(5)] == [200.0] * 5
    assert all(out[s] == 300.0 for s in range(5, 12))


def test_seasonal_step_function_pos_steps_wrap_and_replace_freeze():
    # Jan-start, 1 study year, 2 post-study years. POS Jun=500, Dec=100: every
    # post-study June-November is 500 and December-May wraps to 100, instead of
    # freezing the last study value (300).
    h = study_horizon(_dger(start_month=1, num_anos=1, num_anos_pos=2))
    out = seasonal_step_function(
        [(2024, 1, 300.0)],
        lambda v: v,
        seasonalize=False,
        horizon=h,
        pos_recs=[(6, 500.0), (12, 100.0)],
    )
    expected = [100.0] * 5 + [500.0] * 6 + [100.0]
    assert [out[s] for s in range(12, 36)] == expected * 2


def test_seasonal_step_function_big_m_steps_clear_their_months():
    h = study_horizon(_dger(start_month=1, num_anos=1, num_anos_pos=1))
    out = seasonal_step_function(
        [],
        lambda v: v,
        seasonalize=False,
        horizon=h,
        pre_recs=[(1, BIG_M)],
        pos_recs=[(1, 40.0), (7, BIG_M)],
    )
    assert all(s not in out for s in range(12))
    assert [s for s in range(12, 24) if s in out] == list(range(12, 18))
