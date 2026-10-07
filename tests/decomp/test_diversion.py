"""Tests for the diversion-channel family of cadastro overrides.

Pins the ``VMDESV``/``VSVERT`` scalar specs (which flow through the same
scalar-override machinery as ``VOLMIN``/``VOLMAX``/``VAZMIN``) and the
non-scalar ``AC DESVIO`` diversion-channel reader plus the
``has_diversion``/``diversion`` query methods on ``EffectiveCadastro`` —
against a synthetic ``Dadger`` double, mirroring the doubles established in
``tests/decomp/test_cadastro.py``. No real deck, no calendar resolution
beyond what ``resolve_effective_stage`` already covers.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from idecomp.decomp.modelos.dadger import ACDESVIO, ACVMDESV, ACVSVERT

from novomodelo_bridge.decomp.converters.cadastro import (
    DiversionChannel,
    EffectiveCadastro,
    OutOfHorizon,
    build_effective_cadastro,
)
from novomodelo_bridge.decomp.temporal import OperativeStage, build_operative_calendar


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
def calendar() -> list[OperativeStage]:
    """Stages 0, 1 = July weekly stages; stage 2 = the August monthly stage."""
    stage_block_hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), stage_block_hours)


def _ac_frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _hidr_frame(plants: dict[int, dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(plants).T
    df.index.name = "codigo_usina"
    return df


def test_build_effective_cadastro_vmdesv_vsvert_temporal_resolve_scalar_series(
    calendar: list[OperativeStage],
) -> None:
    """``AC VMDESV``/``AC VSVERT`` flow through the scalar-override
    machinery exactly like ``AC VOLMIN``/``AC VOLMAX``, producing per-stage
    ``volume_desvio``/``volume_vertedouro`` series."""
    dadger = _FakeDadger(
        {
            ACVMDESV: _ac_frame(
                {
                    "codigo_usina": 288,
                    "volume": 2269.10,
                    "mes": "JUL",
                    "semana": None,
                    "ano": 2026.0,
                },
                {
                    "codigo_usina": 288,
                    "volume": 2276.29,
                    "mes": "AGO",
                    "semana": None,
                    "ano": 2026.0,
                },
            ),
            ACVSVERT: _ac_frame(
                {
                    "codigo_usina": 288,
                    "volume": 500.0,
                    "mes": "JUL",
                    "semana": None,
                    "ano": 2026.0,
                },
                {
                    "codigo_usina": 288,
                    "volume": 510.0,
                    "mes": "AGO",
                    "semana": None,
                    "ano": 2026.0,
                },
            ),
        }
    )
    hidr = _hidr_frame(
        {288: {"desvio": 0.0, "volume_desvio": 0.0, "volume_vertedouro": 0.0}}
    )
    effective, _ = build_effective_cadastro(dadger, hidr, calendar)
    assert effective.value(288, "volume_desvio", 0) == 2269.10
    assert effective.value(288, "volume_desvio", 2) == 2276.29
    assert effective.value(288, "volume_vertedouro", 0) == 500.0
    assert effective.value(288, "volume_vertedouro", 2) == 510.0


def test_build_effective_cadastro_base_desvio_sets_diversion_without_limit(
    calendar: list[OperativeStage],
) -> None:
    """A plant with a non-zero base ``desvio`` and no ``AC DESVIO`` override
    has a diversion channel at every stage, carrying no explicit limit."""
    dadger = _FakeDadger({})
    hidr = _hidr_frame({118: {"desvio": 119.0}})
    effective, report = build_effective_cadastro(dadger, hidr, calendar)
    assert effective.has_diversion(118) is True
    for stage in range(len(calendar)):
        assert effective.diversion(118, stage) == DiversionChannel(
            downstream=119, limit=None
        )
    assert "diversion" not in report.applied


def test_build_effective_cadastro_ac_desvio_permanent_overrides_all_stages(
    calendar: list[OperativeStage],
) -> None:
    """A plant with base ``desvio == 0`` gains a diversion solely from a
    permanent ``AC DESVIO`` override, applied at every stage."""
    dadger = _FakeDadger(
        {
            ACDESVIO: _ac_frame(
                {
                    "codigo_usina": 314,
                    "codigo_usina_jusante": 288,
                    "limite_vazao": 9999999.0,
                    "mes": "",
                    "semana": float("nan"),
                    "ano": float("nan"),
                }
            ),
        }
    )
    hidr = _hidr_frame({314: {"desvio": 0.0}})
    effective, report = build_effective_cadastro(dadger, hidr, calendar)
    assert effective.has_diversion(314) is True
    for stage in range(len(calendar)):
        assert effective.diversion(314, stage) == DiversionChannel(288, 9999999.0)
    assert report.applied["diversion"] == 1


def test_build_effective_cadastro_ac_desvio_out_of_horizon_reported_not_applied(
    calendar: list[OperativeStage],
) -> None:
    """An ``AC DESVIO`` whose date resolves past the horizon is reported,
    not applied; a plant with neither a base nor an ``AC`` diversion has no
    diversion channel at all."""
    dadger = _FakeDadger(
        {
            ACDESVIO: _ac_frame(
                {
                    "codigo_usina": 400,
                    "codigo_usina_jusante": 288,
                    "limite_vazao": 100.0,
                    "mes": "OUT",
                    "semana": None,
                    "ano": 2026.0,
                }
            ),
        }
    )
    hidr = _hidr_frame({400: {"desvio": 0.0}, 999: {"desvio": 0.0}})
    effective, report = build_effective_cadastro(dadger, hidr, calendar)
    assert 400 not in effective.diversions
    assert effective.has_diversion(400) is False
    assert effective.diversion(400, 0) is None
    assert OutOfHorizon(400, "diversion", 10, 2026) in report.out_of_horizon
    assert effective.has_diversion(999) is False
    assert effective.diversion(999, 0) is None


def test_build_effective_cadastro_ac_desvio_unknown_source_plant_raises_value_error(
    calendar: list[OperativeStage],
) -> None:
    """An ``AC DESVIO`` naming a source plant absent from the registry is a
    hard error, mirroring the scalar overrides' own invariant; the
    downstream plant it names is not validated at this layer."""
    dadger = _FakeDadger(
        {
            ACDESVIO: _ac_frame(
                {
                    "codigo_usina": 999,
                    "codigo_usina_jusante": 288,
                    "limite_vazao": 100.0,
                    "mes": "",
                    "semana": float("nan"),
                    "ano": float("nan"),
                }
            ),
        }
    )
    hidr = _hidr_frame({118: {"desvio": 0.0}})
    with pytest.raises(ValueError, match="999"):
        build_effective_cadastro(dadger, hidr, calendar)


def test_effective_cadastro_diversions_defaults_to_empty_mapping() -> None:
    """The ``diversions`` field defaults to an empty mapping so every
    existing ``EffectiveCadastro(base=…, n_stages=…, stage_varying=…)``
    construction keeps working untouched."""
    hidr = _hidr_frame({1: {"volume_minimo": 20.0}})
    effective = EffectiveCadastro(base=hidr, n_stages=1, stage_varying={})
    assert effective.diversions == {}
    assert effective.has_diversion(1) is False
    assert effective.diversion(1, 0) is None
