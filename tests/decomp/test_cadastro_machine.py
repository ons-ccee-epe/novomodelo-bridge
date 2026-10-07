"""Tests for the third cadastro-override shape: compound-key machine-set ingestion.

Pins ``_read_machine_set_overrides`` (the ``NUMCON``/``NUMMAQ``/``POTEFE``/
``VAZEFE`` machine-configuration overrides — plant-keyed for ``NUMCON``,
compound-keyed by ``(codigo_usina, indice_conjunto)`` for the other three)
and the ``machine_conjunto_count``/``machine_set`` query methods on
``EffectiveCadastro`` — against a synthetic ``Dadger`` double, mirroring the
doubles established in ``tests/decomp/test_cadastro.py``/
``tests/decomp/test_diversion.py``. No real deck, no calendar resolution
beyond what ``resolve_effective_stage`` already covers.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from idecomp.decomp.modelos.dadger import ACNUMCON, ACNUMMAQ, ACPOTEFE, ACVAZEFE

from novomodelo_bridge.decomp.converters.cadastro import (
    EffectiveCadastro,
    MachineSet,
    OutOfHorizon,
    _read_machine_set_overrides,
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


# --- _read_machine_set_overrides -------------------------------------------


def test_read_machine_set_overrides_numcon_blank_triple_is_plant_keyed_permanent(
    calendar: list[OperativeStage],
) -> None:
    dadger = _FakeDadger(
        {
            ACNUMCON: _ac_frame(
                {
                    "codigo_usina": 9,
                    "numero_conjuntos": 3,
                    "mes": "",
                    "semana": float("nan"),
                    "ano": float("nan"),
                }
            ),
        }
    )
    numero_conjuntos, numero_maquinas, potencia, vazao, out_of_horizon = (
        _read_machine_set_overrides(dadger, calendar)
    )
    assert numero_conjuntos == {9: [(0, 3)]}
    assert numero_maquinas == {}
    assert potencia == {}
    assert vazao == {}
    assert out_of_horizon == []


def test_read_machine_set_overrides_compound_key_rows_grouped_by_code_and_conjunto(
    calendar: list[OperativeStage],
) -> None:
    """A dated row and a blank row for two different conjuntos of the same
    plant resolve to their own stages and are grouped by ``(code,
    conjunto)``, not merged."""
    dadger = _FakeDadger(
        {
            ACNUMMAQ: _ac_frame(
                {
                    "codigo_usina": 275,
                    "indice_conjunto": 1,
                    "numero_maquinas": 4,
                    "mes": "AGO",
                    "semana": 1.0,
                    "ano": 2026.0,
                },
                {
                    "codigo_usina": 275,
                    "indice_conjunto": 2,
                    "numero_maquinas": 2,
                    "mes": "",
                    "semana": float("nan"),
                    "ano": float("nan"),
                },
            ),
        }
    )
    _, numero_maquinas, _, _, out_of_horizon = _read_machine_set_overrides(
        dadger, calendar
    )
    assert numero_maquinas == {(275, 1): [(2, 4)], (275, 2): [(0, 2)]}
    assert out_of_horizon == []


def test_read_machine_set_overrides_past_horizon_row_reported_with_param_label(
    calendar: list[OperativeStage],
) -> None:
    dadger = _FakeDadger(
        {
            ACVAZEFE: _ac_frame(
                {
                    "codigo_usina": 11,
                    "indice_conjunto": 1,
                    "vazao": 999.0,
                    "mes": "OUT",
                    "semana": None,
                    "ano": 2026.0,
                }
            ),
        }
    )
    _, _, _, vazao, out_of_horizon = _read_machine_set_overrides(dadger, calendar)
    assert vazao == {}
    assert out_of_horizon == [OutOfHorizon(11, "vazao", 10, 2026)]


def test_read_machine_set_overrides_numcon_past_horizon_reported_as_numero_conjuntos(
    calendar: list[OperativeStage],
) -> None:
    dadger = _FakeDadger(
        {
            ACNUMCON: _ac_frame(
                {
                    "codigo_usina": 33,
                    "numero_conjuntos": 2,
                    "mes": "OUT",
                    "semana": None,
                    "ano": 2026.0,
                }
            ),
        }
    )
    numero_conjuntos, _, _, _, out_of_horizon = _read_machine_set_overrides(
        dadger, calendar
    )
    assert numero_conjuntos == {}
    assert out_of_horizon == [OutOfHorizon(33, "numero_conjuntos", 10, 2026)]


def test_read_machine_set_overrides_empty_frames_returns_empty(
    calendar: list[OperativeStage],
) -> None:
    empty_plant = pd.DataFrame(
        columns=["codigo_usina", "numero_conjuntos", "mes", "semana", "ano"]
    )
    empty_compound = pd.DataFrame(
        columns=[
            "codigo_usina",
            "indice_conjunto",
            "numero_maquinas",
            "mes",
            "semana",
            "ano",
        ]
    )
    dadger = _FakeDadger(
        {
            ACNUMCON: empty_plant,
            ACNUMMAQ: empty_compound,
            ACPOTEFE: None,
            ACVAZEFE: None,
        }
    )
    assert _read_machine_set_overrides(dadger, calendar) == ({}, {}, {}, {}, [])


# --- build_effective_cadastro -----------------------------------------------


def test_build_effective_cadastro_independent_per_field_densification_mixes_base_and_override(
    calendar: list[OperativeStage],
) -> None:
    """A permanent ``NUMMAQ`` override and a dated ``POTEFE`` override for the
    same ``(code, conjunto)``, with no ``VAZEFE`` override at all, densify
    each field independently: ``numero_maquinas`` follows the override at
    every stage, ``potencia`` is the ``hidr`` base before the dated stage and
    the override from it forward, and ``vazao`` stays at the ``hidr`` base at
    every stage."""
    dadger = _FakeDadger(
        {
            ACNUMMAQ: _ac_frame(
                {
                    "codigo_usina": 174,
                    "indice_conjunto": 1,
                    "numero_maquinas": 5,
                    "mes": "",
                    "semana": float("nan"),
                    "ano": float("nan"),
                }
            ),
            ACPOTEFE: _ac_frame(
                {
                    "codigo_usina": 174,
                    "indice_conjunto": 1,
                    "potencia": 200.0,
                    "mes": "AGO",
                    "semana": 1.0,
                    "ano": 2026.0,
                }
            ),
        }
    )
    hidr = _hidr_frame(
        {
            174: {
                "desvio": 0.0,
                "maquinas_conjunto_1": 4.0,
                "potencia_nominal_conjunto_1": 150.0,
                "vazao_nominal_conjunto_1": 90.0,
            }
        }
    )
    effective, _ = build_effective_cadastro(dadger, hidr, calendar)
    assert [effective.machine_set(174, 1, s) for s in range(3)] == [
        MachineSet(5, 150.0, 90.0),
        MachineSet(5, 150.0, 90.0),
        MachineSet(5, 200.0, 90.0),
    ]


def test_build_effective_cadastro_numcon_permanent_override_and_none_fallthrough(
    calendar: list[OperativeStage],
) -> None:
    dadger = _FakeDadger(
        {
            ACNUMCON: _ac_frame(
                {
                    "codigo_usina": 9,
                    "numero_conjuntos": 3,
                    "mes": "",
                    "semana": float("nan"),
                    "ano": float("nan"),
                }
            ),
        }
    )
    hidr = _hidr_frame(
        {
            9: {"desvio": 0.0, "numero_conjuntos_maquinas": 2.0},
            76: {"desvio": 0.0, "numero_conjuntos_maquinas": 1.0},
        }
    )
    effective, _ = build_effective_cadastro(dadger, hidr, calendar)
    assert [effective.machine_conjunto_count(9, s) for s in range(3)] == [3, 3, 3]
    assert effective.machine_conjunto_count(76, 0) is None
    assert effective.machine_set(76, 1, 0) is None
    # plant 9 has a NUMCON override but no NUMMAQ/POTEFE/VAZEFE override for
    # any of its conjuntos, so machine_set still falls through to None.
    assert effective.machine_set(9, 1, 0) is None


def test_build_effective_cadastro_out_of_horizon_row_absent_and_reported(
    calendar: list[OperativeStage],
) -> None:
    dadger = _FakeDadger(
        {
            ACVAZEFE: _ac_frame(
                {
                    "codigo_usina": 11,
                    "indice_conjunto": 1,
                    "vazao": 999.0,
                    "mes": "OUT",
                    "semana": None,
                    "ano": 2026.0,
                }
            ),
        }
    )
    hidr = _hidr_frame({11: {"desvio": 0.0}})
    effective, report = build_effective_cadastro(dadger, hidr, calendar)
    assert (11, 1) not in effective.machine_sets
    assert effective.machine_set(11, 1, 0) is None
    assert OutOfHorizon(11, "vazao", 10, 2026) in report.out_of_horizon
    assert "vazao" not in report.applied


def test_build_effective_cadastro_applied_counts_only_in_horizon_overrides(
    calendar: list[OperativeStage],
) -> None:
    dadger = _FakeDadger(
        {
            ACPOTEFE: _ac_frame(
                {
                    "codigo_usina": 275,
                    "indice_conjunto": 1,
                    "potencia": 700.0,
                    "mes": "",
                    "semana": float("nan"),
                    "ano": float("nan"),
                },
                {
                    "codigo_usina": 288,
                    "indice_conjunto": 1,
                    "potencia": 800.0,
                    "mes": "OUT",
                    "semana": None,
                    "ano": 2026.0,
                },
            ),
        }
    )
    hidr = _hidr_frame(
        {
            275: {
                "desvio": 0.0,
                "maquinas_conjunto_1": 4.0,
                "potencia_nominal_conjunto_1": 600.0,
                "vazao_nominal_conjunto_1": 60.0,
            },
            288: {"desvio": 0.0},
        }
    )
    _, report = build_effective_cadastro(dadger, hidr, calendar)
    assert report.applied["potencia"] == 1
    assert OutOfHorizon(288, "potencia", 10, 2026) in report.out_of_horizon


def test_build_effective_cadastro_applied_counts_distinct_pairs_and_plants(
    calendar: list[OperativeStage],
) -> None:
    """``applied`` counts distinct plants for ``numero_conjuntos`` and
    distinct ``(code, conjunto)`` pairs for the compound-key mnemonics."""
    dadger = _FakeDadger(
        {
            ACNUMCON: _ac_frame(
                {
                    "codigo_usina": 9,
                    "numero_conjuntos": 3,
                    "mes": "",
                    "semana": float("nan"),
                    "ano": float("nan"),
                },
                {
                    "codigo_usina": 33,
                    "numero_conjuntos": 2,
                    "mes": "",
                    "semana": float("nan"),
                    "ano": float("nan"),
                },
            ),
            ACNUMMAQ: _ac_frame(
                {
                    "codigo_usina": 9,
                    "indice_conjunto": 1,
                    "numero_maquinas": 4,
                    "mes": "",
                    "semana": float("nan"),
                    "ano": float("nan"),
                },
                {
                    "codigo_usina": 9,
                    "indice_conjunto": 2,
                    "numero_maquinas": 3,
                    "mes": "",
                    "semana": float("nan"),
                    "ano": float("nan"),
                },
            ),
        }
    )
    hidr = _hidr_frame(
        {
            9: {
                "desvio": 0.0,
                "numero_conjuntos_maquinas": 2.0,
                "maquinas_conjunto_1": 4.0,
                "maquinas_conjunto_2": 4.0,
                "potencia_nominal_conjunto_1": 100.0,
                "potencia_nominal_conjunto_2": 100.0,
                "vazao_nominal_conjunto_1": 60.0,
                "vazao_nominal_conjunto_2": 60.0,
            },
            33: {"desvio": 0.0, "numero_conjuntos_maquinas": 1.0},
        }
    )
    _, report = build_effective_cadastro(dadger, hidr, calendar)
    assert report.applied["numero_conjuntos"] == 2
    assert report.applied["numero_maquinas"] == 2


def test_build_effective_cadastro_unknown_code_compound_key_raises_value_error(
    calendar: list[OperativeStage],
) -> None:
    dadger = _FakeDadger(
        {
            ACNUMMAQ: _ac_frame(
                {
                    "codigo_usina": 999,
                    "indice_conjunto": 1,
                    "numero_maquinas": 2,
                    "mes": "",
                    "semana": float("nan"),
                    "ano": float("nan"),
                }
            ),
        }
    )
    hidr = _hidr_frame({1: {"desvio": 0.0}})
    with pytest.raises(ValueError, match="999"):
        build_effective_cadastro(dadger, hidr, calendar)


def test_build_effective_cadastro_unknown_code_numcon_raises_value_error(
    calendar: list[OperativeStage],
) -> None:
    dadger = _FakeDadger(
        {
            ACNUMCON: _ac_frame(
                {
                    "codigo_usina": 999,
                    "numero_conjuntos": 2,
                    "mes": "",
                    "semana": float("nan"),
                    "ano": float("nan"),
                }
            ),
        }
    )
    hidr = _hidr_frame({1: {"desvio": 0.0}})
    with pytest.raises(ValueError, match="999"):
        build_effective_cadastro(dadger, hidr, calendar)


def test_effective_cadastro_machine_fields_default_to_empty_mapping() -> None:
    """The two machine-set fields default to empty mappings so every
    existing ``EffectiveCadastro(base=…, n_stages=…, stage_varying=…)``
    construction keeps working untouched."""
    hidr = _hidr_frame({1: {"desvio": 0.0, "volume_minimo": 20.0}})
    effective = EffectiveCadastro(base=hidr, n_stages=1, stage_varying={})
    assert effective.machine_sets == {}
    assert effective.machine_conjunto_counts == {}
    assert effective.machine_conjunto_count(1, 0) is None
    assert effective.machine_set(1, 1, 0) is None
