"""Tests for the effective-cadastro forward-fill, container, and AC ingestion.

Pins ``novomodelo_bridge.decomp.converters.cadastro.overrides._forward_fill_series``
(the pure per-``(plant, param)`` densification), ``EffectiveCadastro`` (the
per-stage-effective view of the cadastro) against hand-built record lists,
and ``_read_scalar_overrides`` (the scalar single-value ``AC`` ingestion)
against a synthetic ``Dadger`` double — no calendar resolution beyond what
``resolve_effective_stage`` already covers, and no real deck.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from idecomp.decomp.modelos.dadger import ACVAZMIN, ACVOLMAX, ACVOLMIN

from novomodelo_bridge.decomp.converters.cadastro import (
    CadastroResolutionReport,
    EffectiveCadastro,
    OutOfHorizon,
    _forward_fill_series,
    _read_scalar_overrides,
    build_effective_cadastro,
)
from novomodelo_bridge.decomp.temporal import OperativeStage, build_operative_calendar


def _hidr_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            1: {"desvio": 0, "volume_minimo": 20.0, "volume_maximo": 100.0},
            2: {"desvio": 0, "volume_minimo": 30.0, "volume_maximo": 150.0},
        }
    ).T
    df.index.name = "codigo_usina"
    return df


@pytest.fixture
def cadastro() -> EffectiveCadastro:
    return EffectiveCadastro(
        base=_hidr_frame(),
        n_stages=3,
        stage_varying={(1, "volume_maximo"): (100.0, 100.0, 250.0)},
    )


def test_forward_fill_series_empty_records_returns_base() -> None:
    assert _forward_fill_series(10.0, [], 3) == [10.0, 10.0, 10.0]


def test_forward_fill_series_permanent_record_overwrites_all_stages() -> None:
    assert _forward_fill_series(10.0, [(0, 5.0)], 3) == [5.0, 5.0, 5.0]


def test_forward_fill_series_unordered_records_forward_fills() -> None:
    result = _forward_fill_series(10.0, [(2, 7.0), (1, 6.0)], 4)
    assert result == [10.0, 6.0, 7.0, 7.0]


def test_forward_fill_series_does_not_mutate_records() -> None:
    records = [(2, 7.0), (1, 6.0)]
    _forward_fill_series(10.0, records, 4)
    assert records == [(2, 7.0), (1, 6.0)]


def test_effective_cadastro_value_stage_varying_returns_override(
    cadastro: EffectiveCadastro,
) -> None:
    assert cadastro.value(1, "volume_maximo", 2) == 250.0


def test_effective_cadastro_value_not_stage_varying_falls_through_to_base(
    cadastro: EffectiveCadastro,
) -> None:
    assert cadastro.value(1, "volume_minimo", 2) == 20.0


def test_effective_cadastro_is_stage_varying_true_for_overridden_param(
    cadastro: EffectiveCadastro,
) -> None:
    assert cadastro.is_stage_varying(1, "volume_maximo") is True


def test_effective_cadastro_is_stage_varying_false_for_untouched_param(
    cadastro: EffectiveCadastro,
) -> None:
    assert cadastro.is_stage_varying(1, "volume_minimo") is False


def test_effective_cadastro_series_stage_varying_returns_stored_tuple(
    cadastro: EffectiveCadastro,
) -> None:
    assert cadastro.series(1, "volume_maximo") == [100.0, 100.0, 250.0]


def test_effective_cadastro_series_not_stage_varying_returns_dense_base(
    cadastro: EffectiveCadastro,
) -> None:
    assert cadastro.series(1, "volume_minimo") == [20.0, 20.0, 20.0]


class _FakeDadger:
    """Return a preset ``AC`` frame (or ``None``) dispatched by ``modificacao`` class."""

    def __init__(self, frames: dict[type, pd.DataFrame | None]) -> None:
        self._frames = frames

    def ac(
        self,
        codigo_usina: int | None = None,
        modificacao: type | None = None,
        df: bool = True,
    ) -> pd.DataFrame | None:
        if modificacao is None:
            return None
        return self._frames.get(modificacao)


@pytest.fixture
def scalar_calendar() -> list[OperativeStage]:
    """Stages 0, 1 = July weekly stages; stage 2 = the August monthly stage."""
    stage_block_hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), stage_block_hours)


def _ac_frame(**row: object) -> pd.DataFrame:
    return pd.DataFrame([row])


def test_read_scalar_overrides_in_horizon_row_resolves_to_stage(
    scalar_calendar: list[OperativeStage],
) -> None:
    dadger = _FakeDadger(
        {
            ACVOLMAX: _ac_frame(
                codigo_usina=1, volume=250.0, mes="AGO", semana=None, ano=2026.0
            ),
            ACVOLMIN: None,
        }
    )
    records, out_of_horizon = _read_scalar_overrides(dadger, scalar_calendar)
    assert records == {(1, "volume_maximo"): [(2, 250.0)]}
    assert out_of_horizon == []


def test_read_scalar_overrides_blank_date_is_permanent_stage_zero(
    scalar_calendar: list[OperativeStage],
) -> None:
    dadger = _FakeDadger(
        {
            ACVOLMIN: _ac_frame(
                codigo_usina=1,
                volume=30.0,
                mes="",
                semana=float("nan"),
                ano=float("nan"),
            ),
            ACVOLMAX: None,
        }
    )
    records, out_of_horizon = _read_scalar_overrides(dadger, scalar_calendar)
    assert records == {(1, "volume_minimo"): [(0, 30.0)]}
    assert out_of_horizon == []


def test_read_scalar_overrides_real_representation_resolves_both_specs(
    scalar_calendar: list[OperativeStage],
) -> None:
    """The real deck shape: a string-month row and an empty-string blank row."""
    dadger = _FakeDadger(
        {
            ACVOLMAX: _ac_frame(
                codigo_usina=1, volume=250.0, mes="AGO", semana=1.0, ano=2026.0
            ),
            ACVOLMIN: _ac_frame(
                codigo_usina=1,
                volume=30.0,
                mes="",
                semana=float("nan"),
                ano=float("nan"),
            ),
        }
    )
    records, out_of_horizon = _read_scalar_overrides(dadger, scalar_calendar)
    assert records == {
        (1, "volume_maximo"): [(2, 250.0)],
        (1, "volume_minimo"): [(0, 30.0)],
    }
    assert out_of_horizon == []


def test_read_scalar_overrides_past_horizon_is_reported_not_dropped(
    scalar_calendar: list[OperativeStage],
) -> None:
    dadger = _FakeDadger(
        {
            ACVOLMAX: _ac_frame(
                codigo_usina=1, volume=999.0, mes="OUT", semana=None, ano=2026.0
            ),
            ACVOLMIN: None,
        }
    )
    records, out_of_horizon = _read_scalar_overrides(dadger, scalar_calendar)
    assert (1, "volume_maximo") not in records
    assert out_of_horizon == [OutOfHorizon(1, "volume_maximo", 10, 2026)]


def test_read_scalar_overrides_empty_frames_returns_empty(
    scalar_calendar: list[OperativeStage],
) -> None:
    empty = pd.DataFrame(columns=["codigo_usina", "volume", "mes", "semana", "ano"])
    dadger = _FakeDadger({ACVOLMIN: empty, ACVOLMAX: empty})
    assert _read_scalar_overrides(dadger, scalar_calendar) == ({}, [])


@pytest.fixture
def resolved_case(
    scalar_calendar: list[OperativeStage],
) -> tuple[EffectiveCadastro, CadastroResolutionReport]:
    """A temporal ``ACVOLMAX`` row plus a permanent ``ACVOLMIN`` row, both for plant 1."""
    dadger = _FakeDadger(
        {
            ACVOLMAX: _ac_frame(
                codigo_usina=1, volume=250.0, mes="AGO", semana=None, ano=2026.0
            ),
            ACVOLMIN: _ac_frame(
                codigo_usina=1,
                volume=30.0,
                mes="",
                semana=float("nan"),
                ano=float("nan"),
            ),
        }
    )
    return build_effective_cadastro(dadger, _hidr_frame(), scalar_calendar)


def test_build_effective_cadastro_temporal_and_permanent_overrides_resolve_series(
    resolved_case: tuple[EffectiveCadastro, CadastroResolutionReport],
) -> None:
    effective, _ = resolved_case
    assert effective.series(1, "volume_maximo") == [100.0, 100.0, 250.0]
    assert effective.series(1, "volume_minimo") == [30.0, 30.0, 30.0]


def test_build_effective_cadastro_reports_applied_counts_and_no_out_of_horizon(
    resolved_case: tuple[EffectiveCadastro, CadastroResolutionReport],
) -> None:
    _, report = resolved_case
    assert report.applied == {"volume_maximo": 1, "volume_minimo": 1}
    assert report.out_of_horizon == ()


def test_build_effective_cadastro_no_overrides_falls_through_to_base(
    scalar_calendar: list[OperativeStage],
) -> None:
    empty = pd.DataFrame(columns=["codigo_usina", "volume", "mes", "semana", "ano"])
    dadger = _FakeDadger({ACVOLMIN: empty, ACVOLMAX: empty})
    effective, report = build_effective_cadastro(dadger, _hidr_frame(), scalar_calendar)
    assert effective.stage_varying == {}
    assert effective.value(1, "volume_maximo", 2) == 100.0
    assert report.applied == {}


def test_build_effective_cadastro_unknown_plant_code_raises_value_error(
    scalar_calendar: list[OperativeStage],
) -> None:
    dadger = _FakeDadger(
        {
            ACVOLMAX: _ac_frame(
                codigo_usina=999, volume=250.0, mes="AGO", semana=None, ano=2026.0
            ),
            ACVOLMIN: None,
        }
    )
    with pytest.raises(ValueError, match="999"):
        build_effective_cadastro(dadger, _hidr_frame(), scalar_calendar)


def test_build_effective_cadastro_permanent_vazmin_resolves_and_is_applied(
    scalar_calendar: list[OperativeStage],
) -> None:
    """A permanent ``AC VAZMIN`` row (the real deck shape — blank
    ``mes``/``semana``/``ano``) resolves to a constant per-stage
    ``vazao_minima_historica`` series and is counted in
    ``report.applied``."""
    hidr = pd.DataFrame({1: {"desvio": 0, "vazao_minima_historica": 73.0}}).T
    hidr.index.name = "codigo_usina"
    dadger = _FakeDadger(
        {
            ACVAZMIN: _ac_frame(
                codigo_usina=1,
                vazao=0.0,
                mes="",
                semana=float("nan"),
                ano=float("nan"),
            ),
        }
    )
    effective, report = build_effective_cadastro(dadger, hidr, scalar_calendar)
    assert [effective.value(1, "vazao_minima_historica", s) for s in range(3)] == [
        0.0,
        0.0,
        0.0,
    ]
    assert report.applied["vazao_minima_historica"] == 1
