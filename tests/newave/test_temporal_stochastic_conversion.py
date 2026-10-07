"""Unit tests for temporal and stochastic conversion functions.

All inewave I/O is mocked via ``unittest.mock.patch`` so no real the source model files
are required.  Synthetic DataFrames exercise the core logic of each converter.
``test_six_hundred_posto_matrix_is_laid_out_at_full_width`` is the one
exception: it has to prove the reader lays a 600-posto matrix out as declared,
which a mock cannot show.
"""

from __future__ import annotations

import calendar
import datetime
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.errors import FieldParseError
from novomodelo_bridge.newave.converters.stochastic import _posto_count, _read_vazoes
from novomodelo_bridge.newave.id_map import NewaveIdMap
from tests.conftest import make_case, make_nw_files

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_id_map_hydros(hydro_codes: list[int]) -> NewaveIdMap:
    return NewaveIdMap(subsystem_ids=[], hydro_codes=hydro_codes, thermal_codes=[])


def _make_id_map_buses(subsystem_ids: list[int]) -> NewaveIdMap:
    return NewaveIdMap(subsystem_ids=subsystem_ids, hydro_codes=[], thermal_codes=[])


def _make_dger_mock(
    *,
    mes_inicio: int = 1,
    ano_inicio: int = 2020,
    num_anos: int = 5,
    num_anos_pre: int = 0,
    num_anos_pos: int = 0,
    num_forwards: int = 20,
    num_max_iteracoes: int = 200,
    num_series: int = 500,
    taxa_de_desconto: float = 12.0,
    num_aberturas: int = 10,
    cvar: int = 0,
    tipo_execucao: int = 1,
    tipo_simulacao_final: int = 1,
    considera_reamostragem_cenarios: int = 0,
    ano_inicial_historico: int = 1931,
    consideracao_media_anual_afluencias: int | None = None,
    selecao_de_cortes_forward: int = 1,
    selecao_de_cortes_backward: int = 1,
    ordem_maxima_parp: int = 6,
    impressao_estados_geracao_cortes: int | None = None,
) -> MagicMock:
    dger = MagicMock()
    dger.mes_inicio_estudo = mes_inicio
    dger.ano_inicio_estudo = ano_inicio
    dger.num_anos_estudo = num_anos
    dger.num_anos_pre_estudo = num_anos_pre
    dger.num_anos_pos_estudo = num_anos_pos
    dger.num_forwards = num_forwards
    dger.num_max_iteracoes = num_max_iteracoes
    dger.num_series_sinteticas = num_series
    dger.taxa_de_desconto = taxa_de_desconto
    dger.num_aberturas = num_aberturas
    dger.cvar = cvar
    dger.tipo_execucao = tipo_execucao
    dger.tipo_simulacao_final = tipo_simulacao_final
    dger.considera_reamostragem_cenarios = considera_reamostragem_cenarios
    dger.ano_inicial_historico = ano_inicial_historico
    dger.consideracao_media_anual_afluencias = consideracao_media_anual_afluencias
    dger.selecao_de_cortes_forward = selecao_de_cortes_forward
    dger.selecao_de_cortes_backward = selecao_de_cortes_backward
    dger.ordem_maxima_parp = ordem_maxima_parp
    dger.impressao_estados_geracao_cortes = impressao_estados_geracao_cortes
    return dger


def _make_patamar_mock_single() -> MagicMock:
    """Single block: fraction=1.0 for every calendar month."""
    rows = []
    for month in range(1, 13):
        rows.append(
            {
                "data": datetime.datetime(2020, month, 1),
                "patamar": 1,
                "valor": 1.0,
            }
        )
    df = pd.DataFrame(rows)
    pat = MagicMock()
    pat.duracao_mensal_patamares = df
    pat.numero_patamares = 1
    return pat


def _make_patamar_mock_three_blocks() -> MagicMock:
    """Three blocks: fractions 0.3, 0.4, 0.3 for every calendar month."""
    fractions = [0.3, 0.4, 0.3]
    rows = []
    for month in range(1, 13):
        for pat_idx, frac in enumerate(fractions, start=1):
            rows.append(
                {
                    "data": datetime.datetime(2020, month, 1),
                    "patamar": pat_idx,
                    "valor": frac,
                }
            )
    df = pd.DataFrame(rows)
    pat = MagicMock()
    pat.duracao_mensal_patamares = df
    pat.numero_patamares = 3
    return pat


# ---------------------------------------------------------------------------
# Tests: convert_stages
# ---------------------------------------------------------------------------


class TestConvertStagesSingleBlock:
    def test_stage_count_five_years(self, tmp_path) -> None:
        dger = _make_dger_mock(num_anos=5)
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        assert len(result["stages"]) == 60

    def test_stage_ids_are_sequential_zero_based(self, tmp_path) -> None:
        dger = _make_dger_mock(num_anos=5)
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        ids = [s["id"] for s in result["stages"]]
        assert ids == list(range(60))

    def test_first_stage_dates_jan_2020(self, tmp_path) -> None:
        dger = _make_dger_mock(mes_inicio=1, ano_inicio=2020, num_anos=5)
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        stage0 = result["stages"][0]
        assert stage0["start_date"] == "2020-01-01"
        assert stage0["end_date"] == "2020-02-01"

    def test_last_stage_end_date(self, tmp_path) -> None:
        # 5 years starting Jan 2020 -> last stage is Dec 2024, end = 2025-01-01.
        dger = _make_dger_mock(mes_inicio=1, ano_inicio=2020, num_anos=5)
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        last = result["stages"][-1]
        assert last["end_date"] == "2025-01-01"

    def test_block_hours_january(self, tmp_path) -> None:
        dger = _make_dger_mock(mes_inicio=1, ano_inicio=2020, num_anos=1)
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        stage0 = result["stages"][0]  # January 2020
        assert len(stage0["blocks"]) == 1
        block = stage0["blocks"][0]
        # January 2020 has 31 days * 24 h = 744 h.
        assert block["hours"] == pytest.approx(744.0)
        assert block["name"] == "SINGLE"
        assert block["id"] == 0

    def test_num_openings_from_dger(self, tmp_path) -> None:
        dger = _make_dger_mock(num_aberturas=50)
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        for stage in result["stages"]:
            assert stage["num_openings"] == 50

    def test_discount_rate_percent_to_decimal(self, tmp_path) -> None:
        dger = _make_dger_mock(taxa_de_desconto=12.0)
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        assert result["policy_graph"]["annual_discount_rate"] == pytest.approx(0.12)

    def test_policy_graph_is_finite_horizon(self, tmp_path) -> None:
        dger = _make_dger_mock()
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        assert result["policy_graph"]["type"] == "finite_horizon"

    def test_transitions_are_linear(self, tmp_path) -> None:
        dger = _make_dger_mock(num_anos=2)
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        transitions = result["policy_graph"]["transitions"]
        # 24 stages -> 23 transitions.
        assert len(transitions) == 23
        # Each transition goes from i to i+1 with probability 1.0.
        for i, t in enumerate(transitions):
            assert t["source_id"] == i
            assert t["target_id"] == i + 1
            assert t["probability"] == pytest.approx(1.0)

    def test_risk_measure_is_expectation(self, tmp_path) -> None:
        dger = _make_dger_mock(cvar=0)
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        for stage in result["stages"]:
            assert stage["risk_measure"] == "expectation"

    def test_risk_measure_constant_cvar(self, tmp_path) -> None:
        """dger.cvar == 1 produces constant CVaR risk_measure for all stages."""
        dger = _make_dger_mock(num_anos=1, cvar=1)
        patamar = _make_patamar_mock_single()
        cvar_mock = MagicMock()
        cvar_mock.valores_constantes = [15.0, 40.0]
        case = make_case(tmp_path, dger=dger, patamar=patamar, cvar=cvar_mock)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        for stage in result["stages"]:
            rm = stage["risk_measure"]
            assert isinstance(rm, dict), f"Expected dict, got {rm!r}"
            assert "cvar" in rm
            assert rm["cvar"]["alpha"] == pytest.approx(0.15)
            assert rm["cvar"]["lambda"] == pytest.approx(0.40)

    def test_risk_measure_temporal_cvar_uses_override(self, tmp_path) -> None:
        """dger.cvar==2: temporal alpha/lambda; zero rows fall back to constant."""
        dger = _make_dger_mock(mes_inicio=1, ano_inicio=2020, num_anos=1, cvar=2)
        patamar = _make_patamar_mock_single()

        # Month 1 (Jan 2020): alpha=0 => use constant 0.15; lambda=20% override
        # Month 2 (Feb 2020): alpha=10% override; lambda=0 => use constant 0.40
        alpha_rows = []
        lambda_rows = []
        for m in range(1, 13):
            a_val = 0.0 if m == 1 else (10.0 if m == 2 else 0.0)
            l_val = 20.0 if m == 1 else (0.0 if m == 2 else 0.0)
            alpha_rows.append({"data": datetime.datetime(2020, m, 1), "valor": a_val})
            lambda_rows.append({"data": datetime.datetime(2020, m, 1), "valor": l_val})

        cvar_mock = MagicMock()
        cvar_mock.valores_constantes = [15.0, 40.0]
        cvar_mock.alfa_variavel = pd.DataFrame(alpha_rows)
        cvar_mock.lambda_variavel = pd.DataFrame(lambda_rows)
        case = make_case(tmp_path, dger=dger, patamar=patamar, cvar=cvar_mock)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        stages = result["stages"]
        # Stage 0 (Jan 2020): alpha=0.0 => fallback 0.15; lambda=20.0 => 0.20
        rm0 = stages[0]["risk_measure"]
        assert isinstance(rm0, dict)
        assert rm0["cvar"]["alpha"] == pytest.approx(0.15)
        assert rm0["cvar"]["lambda"] == pytest.approx(0.20)

        # Stage 1 (Feb 2020): alpha=10.0 => 0.10; lambda=0.0 => fallback 0.40
        rm1 = stages[1]["risk_measure"]
        assert isinstance(rm1, dict)
        assert rm1["cvar"]["alpha"] == pytest.approx(0.10)
        assert rm1["cvar"]["lambda"] == pytest.approx(0.40)

    def test_risk_measure_cvar0_without_file_uses_expectation(self, tmp_path) -> None:
        """When cvar.dat is absent, risk_measure defaults to 'expectation'."""
        dger = _make_dger_mock(num_anos=1, cvar=1)
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        # cvar absent => fallback expectation
        result = convert_stages(case, _make_id_map_hydros([]))
        for stage in result["stages"]:
            assert stage["risk_measure"] == "expectation"

    def test_zero_study_years_raises_value_error(self, tmp_path) -> None:
        dger = _make_dger_mock(num_anos=0)
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        with pytest.raises(ValueError, match="zero study years"):
            convert_stages(case, _make_id_map_hydros([]))


class TestConvertStagesThreeBlocks:
    def test_three_blocks_present(self, tmp_path) -> None:
        dger = _make_dger_mock(num_anos=1)
        patamar = _make_patamar_mock_three_blocks()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        for stage in result["stages"]:
            assert len(stage["blocks"]) == 3

    def test_three_blocks_names(self, tmp_path) -> None:
        dger = _make_dger_mock(num_anos=1)
        patamar = _make_patamar_mock_three_blocks()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        stage0 = result["stages"][0]
        names = [b["name"] for b in stage0["blocks"]]
        assert names == ["HEAVY", "MEDIUM", "LIGHT"]

    def test_block_hours_sum_to_month_hours(self, tmp_path) -> None:
        dger = _make_dger_mock(mes_inicio=1, ano_inicio=2020, num_anos=1)
        patamar = _make_patamar_mock_three_blocks()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        for i, stage in enumerate(result["stages"]):
            year = 2020
            month = i + 1
            expected_total = float(calendar.monthrange(year, month)[1] * 24)
            total_hours = sum(b["hours"] for b in stage["blocks"])
            assert total_hours == pytest.approx(expected_total)


class TestConvertStagesPreStudy:
    def test_pre_study_stages_generated(self, tmp_path) -> None:
        dger = _make_dger_mock(
            mes_inicio=1, ano_inicio=2020, num_anos=5, num_anos_pre=1
        )
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        pre = result.get("pre_study_stages", [])
        assert len(pre) == 12

    def test_pre_study_ids_are_negative(self, tmp_path) -> None:
        dger = _make_dger_mock(
            mes_inicio=1, ano_inicio=2020, num_anos=5, num_anos_pre=1
        )
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        pre = result["pre_study_stages"]
        assert all(s["id"] < 0 for s in pre)
        ids = [s["id"] for s in pre]
        assert ids == list(range(-12, 0))

    def test_no_pre_study_key_when_zero(self, tmp_path) -> None:
        dger = _make_dger_mock(num_anos_pre=0)
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages

        result = convert_stages(case, _make_id_map_hydros([]))
        assert "pre_study_stages" not in result


# ---------------------------------------------------------------------------
# Tests: convert_config
# ---------------------------------------------------------------------------


class TestConvertConfig:
    def test_forward_passes(self, tmp_path) -> None:
        dger = _make_dger_mock(num_forwards=20)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["training"]["selection"] == {
            "method": "sampled",
            "forward_passes": 20,
        }

    def test_iteration_limit(self, tmp_path) -> None:
        dger = _make_dger_mock(num_max_iteracoes=200)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        rules = result["training"]["stopping_rules"]
        assert len(rules) == 1
        assert rules[0]["type"] == "iteration_limit"
        assert rules[0]["limit"] == 200

    def test_backward_scheduler_by_node_half_openings(self, tmp_path) -> None:
        # block_size is ceil(num_aberturas / 2): even count halves exactly.
        dger = _make_dger_mock(num_aberturas=20)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        scheduler = result["training"]["parallelism"]["backward_scheduler"]
        assert scheduler == {"method": "by_node", "block_size": 10}

    def test_backward_scheduler_block_size_rounds_up(self, tmp_path) -> None:
        # An odd opening count rounds up: ceil(21 / 2) == 11.
        dger = _make_dger_mock(num_aberturas=21)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        scheduler = result["training"]["parallelism"]["backward_scheduler"]
        assert scheduler["block_size"] == 11

    def test_backward_scheduler_single_opening(self, tmp_path) -> None:
        # A single backward opening yields block_size 1 (the schema minimum).
        dger = _make_dger_mock(num_aberturas=1)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        scheduler = result["training"]["parallelism"]["backward_scheduler"]
        assert scheduler["block_size"] == 1

    def test_simulation_enabled_default(self, tmp_path) -> None:
        dger = _make_dger_mock(tipo_execucao=1, tipo_simulacao_final=1)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["simulation"]["enabled"] is True

    def test_num_scenarios_from_num_series(self, tmp_path) -> None:
        dger = _make_dger_mock(num_series=500)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["simulation"]["selection"] == {
            "method": "sampled",
            "num_scenarios": 500,
        }

    # -- impressao_estados_geracao_cortes / exports.states --

    def test_export_states_true_when_flag_zero(self, tmp_path) -> None:
        dger = _make_dger_mock(impressao_estados_geracao_cortes=0)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["exports"]["states"] is True

    def test_export_states_false_when_flag_nonzero(self, tmp_path) -> None:
        dger = _make_dger_mock(impressao_estados_geracao_cortes=1)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["exports"]["states"] is False

    def test_export_states_false_when_flag_absent(self, tmp_path) -> None:
        dger = _make_dger_mock(impressao_estados_geracao_cortes=None)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["exports"]["states"] is False

    # -- tipo_execucao / training.enabled --

    def test_tipo_execucao_1_enables_training(self, tmp_path) -> None:
        dger = _make_dger_mock(tipo_execucao=1)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert (
            "enabled" not in result["training"]
            or result["training"].get("enabled") is not False
        )

    def test_tipo_execucao_0_disables_training(self, tmp_path) -> None:
        dger = _make_dger_mock(tipo_execucao=0, tipo_simulacao_final=1)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["training"]["enabled"] is False
        assert result["simulation"]["enabled"] is True

    # -- tipo_simulacao_final / simulation --

    def test_tipo_simulacao_final_0_disables_simulation(self, tmp_path) -> None:
        dger = _make_dger_mock(tipo_execucao=1, tipo_simulacao_final=0)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["simulation"]["enabled"] is False

    def test_simulation_reamostragem_1_out_of_sample(self, tmp_path) -> None:
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=1,
            considera_reamostragem_cenarios=1,
        )
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["simulation"]["enabled"] is True
        src = result["simulation"]["scenario_source"]
        assert src["seed"] == 42
        assert src["inflow"]["scheme"] == "out_of_sample"
        assert "historical_years" not in src

    def test_simulation_reamostragem_0_in_sample(self, tmp_path) -> None:
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=1,
            considera_reamostragem_cenarios=0,
        )
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["simulation"]["enabled"] is True
        src = result["simulation"]["scenario_source"]
        assert src["seed"] == 42
        assert src["inflow"]["scheme"] == "in_sample"

    def test_tipo_simulacao_final_2_overrides_reamostragem(self, tmp_path) -> None:
        """tipo_simulacao_final=2 forces historical even when reamostragem=1."""
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=2,
            considera_reamostragem_cenarios=1,
            ano_inicial_historico=1931,
            ano_inicio=2026,
        )
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["simulation"]["enabled"] is True
        src = result["simulation"]["scenario_source"]
        assert src["seed"] == 42
        assert src["inflow"]["scheme"] == "historical"
        assert src["historical_years"] == {"from": 1932, "to": 2025}

    def test_tipo_simulacao_final_2_without_reamostragem(self, tmp_path) -> None:
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=2,
            considera_reamostragem_cenarios=0,
            ano_inicial_historico=1931,
            ano_inicio=2026,
        )
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        src = result["simulation"]["scenario_source"]
        assert src["seed"] == 42
        assert src["inflow"]["scheme"] == "historical"
        assert src["historical_years"] == {"from": 1932, "to": 2025}

    # -- Deterministic mode (1 fwd × 1 abertura × historical × varredura=0 × 1 year) --

    def test_deterministic_mode_mirrors_training_to_simulation(self, tmp_path) -> None:
        """When all five deterministic conditions hold, training reuses the
        simulation's single-year historical scenario_source."""
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=2,
            num_forwards=1,
            num_aberturas=1,
        )

        mock_shist = MagicMock()
        mock_shist.varredura = 0
        mock_shist.anos_inicio_simulacoes = [1983]
        mock_shist.ano_inicio_varredura = 1932
        case = make_case(tmp_path, dger=dger, shist=mock_shist)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        src = result["training"]["scenario_source"]
        assert src["inflow"]["scheme"] == "historical"
        assert src["historical_years"] == [1983]
        # Simulation side stays consistent.
        assert result["simulation"]["scenario_source"]["historical_years"] == [1983]
        assert result["simulation"]["selection"]["num_scenarios"] == 1
        # Deterministic mode also forces estimation.max_order = 0 (workaround
        # for novomodelo's SDDP negative-gap regression when lag-state is present)
        # and pins order_selection to "pacf" to avoid the residual annual
        # coupling that survives even with max_order = 0.
        assert result["estimation"]["max_order"] == 0
        assert result["estimation"]["order_selection"] == "pacf"

    def test_non_deterministic_mode_preserves_estimation_max_order(
        self, tmp_path
    ) -> None:
        """Non-deterministic conversions keep the configured PAR order — the
        max_order override is gated by deterministic mode only.
        """
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=2,
            num_forwards=2,  # breaks deterministic mode
            num_aberturas=1,
            ordem_maxima_parp=4,
            consideracao_media_anual_afluencias=0,  # → "pacf"
        )
        mock_shist = MagicMock()
        mock_shist.varredura = 0
        mock_shist.anos_inicio_simulacoes = [1983]
        mock_shist.ano_inicio_varredura = 1932
        case = make_case(tmp_path, dger=dger, shist=mock_shist)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["estimation"]["max_order"] == 4
        assert result["estimation"]["order_selection"] == "pacf"

    def test_deterministic_mode_disabled_when_num_forwards_gt_1(self, tmp_path) -> None:
        """Any one condition false → deterministic mode off; training keeps
        its standard (or absent) scenario_source."""
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=2,
            num_forwards=2,  # not 1
            num_aberturas=1,
            considera_reamostragem_cenarios=0,
        )
        mock_shist = MagicMock()
        mock_shist.varredura = 0
        mock_shist.anos_inicio_simulacoes = [1983]
        mock_shist.ano_inicio_varredura = 1932
        case = make_case(tmp_path, dger=dger, shist=mock_shist)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        # With no reamostragem and not deterministic, training has no
        # scenario_source — novomodelo defaults apply.
        assert "scenario_source" not in result["training"]

    def test_deterministic_mode_disabled_when_multiple_historical_years(
        self, tmp_path
    ) -> None:
        """Two-year historical list violates the single-year requirement →
        deterministic mode off."""
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=2,
            num_forwards=1,
            num_aberturas=1,
        )
        mock_shist = MagicMock()
        mock_shist.varredura = 0
        mock_shist.anos_inicio_simulacoes = [1983, 1990]
        mock_shist.ano_inicio_varredura = 1932
        case = make_case(tmp_path, dger=dger, shist=mock_shist)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert "scenario_source" not in result["training"]

    def test_deterministic_mode_stages_get_historical_residuals(self, tmp_path) -> None:
        """Every stage in the deterministic case is tagged with
        sampling_method=historical_residuals."""
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=2,
            num_forwards=1,
            num_aberturas=1,
            mes_inicio=1,
            ano_inicio=2024,
            num_anos=2,
            num_anos_pos=0,
        )
        mock_shist = MagicMock()
        mock_shist.varredura = 0
        mock_shist.anos_inicio_simulacoes = [1983]
        mock_shist.ano_inicio_varredura = 1932
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, shist=mock_shist, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages
        from novomodelo_bridge.newave.id_map import NewaveIdMap

        result = convert_stages(
            case,
            NewaveIdMap(subsystem_ids=[1], hydro_codes=[], thermal_codes=[]),
        )
        for stage in result["stages"]:
            assert stage["sampling_method"] == "historical_residuals"

    def test_non_deterministic_mode_stages_omit_sampling_method(self, tmp_path) -> None:
        """Without deterministic mode, sampling_method is omitted so novomodelo
        applies its default (saa)."""
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=1,
            num_forwards=20,  # not deterministic
            mes_inicio=1,
            ano_inicio=2024,
            num_anos=2,
            num_anos_pos=0,
        )
        patamar = _make_patamar_mock_single()
        case = make_case(tmp_path, dger=dger, patamar=patamar)

        from novomodelo_bridge.newave.converters.temporal import convert_stages
        from novomodelo_bridge.newave.id_map import NewaveIdMap

        result = convert_stages(
            case,
            NewaveIdMap(subsystem_ids=[1], hydro_codes=[], thermal_codes=[]),
        )
        for stage in result["stages"]:
            assert "sampling_method" not in stage

    # -- Cut selection (selecao_de_cortes_forward / _backward) --

    def test_cut_selection_both_flags_one_enables(self, tmp_path) -> None:
        dger = _make_dger_mock(
            selecao_de_cortes_forward=1, selecao_de_cortes_backward=1
        )
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        cs = result["training"]["cut_selection"]
        assert cs["selection"]["method"] == "lml1"
        assert cs["selection"]["check_frequency"] == 1
        assert cs["row_activity_tolerance"] == 1e-6
        assert "enabled" not in cs

    def test_cut_selection_only_forward_enables(self, tmp_path) -> None:
        dger = _make_dger_mock(
            selecao_de_cortes_forward=1, selecao_de_cortes_backward=0
        )
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        cs = result["training"]["cut_selection"]
        assert cs["selection"]["method"] == "lml1"
        assert cs["selection"]["check_frequency"] == 1
        assert cs["row_activity_tolerance"] == 1e-6
        assert "enabled" not in cs

    def test_cut_selection_only_backward_enables(self, tmp_path) -> None:
        dger = _make_dger_mock(
            selecao_de_cortes_forward=0, selecao_de_cortes_backward=1
        )
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        cs = result["training"]["cut_selection"]
        assert cs["selection"]["method"] == "lml1"
        assert cs["selection"]["check_frequency"] == 1
        assert cs["row_activity_tolerance"] == 1e-6
        assert "enabled" not in cs

    def test_cut_selection_both_zero_disables(self, tmp_path) -> None:
        dger = _make_dger_mock(
            selecao_de_cortes_forward=0, selecao_de_cortes_backward=0
        )
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        cs = result["training"]["cut_selection"]
        assert "selection" not in cs
        assert cs["row_activity_tolerance"] == 1e-6

    def test_cut_selection_none_treated_as_zero(self, tmp_path) -> None:
        """When the dger.dat field is absent (None), treat as 0 so the
        union rule still applies: both None → disabled."""
        dger = _make_dger_mock()
        dger.selecao_de_cortes_forward = None
        dger.selecao_de_cortes_backward = None
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        cs = result["training"]["cut_selection"]
        assert "selection" not in cs
        assert cs["row_activity_tolerance"] == 1e-6

    # -- Shist-driven historical_years (tipo_simulacao_final == 2) --

    def test_shist_varredura_0_emits_explicit_list(self, tmp_path) -> None:
        """shist.varredura=0 → historical_years is the explicit list from
        anos_inicio_simulacoes."""
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=2,
            ano_inicio=2024,
            num_anos=3,
            num_anos_pos=3,
        )

        mock_shist = MagicMock()
        mock_shist.varredura = 0
        mock_shist.anos_inicio_simulacoes = [1983, 1985, 1990]
        mock_shist.ano_inicio_varredura = 1932
        case = make_case(tmp_path, dger=dger, shist=mock_shist)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        src = result["simulation"]["scenario_source"]
        assert src["inflow"]["scheme"] == "historical"
        assert src["historical_years"] == [1983, 1985, 1990]

    def test_shist_varredura_1_emits_range_with_horizon_aware_end(
        self, tmp_path
    ) -> None:
        """shist.varredura=1 → historical_years is a range from
        ano_inicio_varredura to ano_inicio_estudo - (num_anos + num_anos_pos),
        the most recent year for which the scenario still fits in history."""
        # Horizon = 3 study + 3 post-study = 6 years.  Latest valid start year
        # = 2024 - 6 = 2018.
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=2,
            ano_inicio=2024,
            num_anos=3,
            num_anos_pos=3,
        )

        mock_shist = MagicMock()
        mock_shist.varredura = 1
        mock_shist.ano_inicio_varredura = 1932
        mock_shist.anos_inicio_simulacoes = []
        case = make_case(tmp_path, dger=dger, shist=mock_shist)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        src = result["simulation"]["scenario_source"]
        assert src["inflow"]["scheme"] == "historical"
        assert src["historical_years"] == {"from": 1932, "to": 2018}

    def test_shist_varredura_1_range_collapse_clamps(self, tmp_path) -> None:
        """When the horizon is wider than the gap between ano_inicio_varredura
        and ano_inicio_estudo, the range collapses to a single year — clamp
        ``to=from`` so novomodelo still accepts the config."""
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=2,
            ano_inicio=2024,
            num_anos=100,
            num_anos_pos=0,
        )

        mock_shist = MagicMock()
        mock_shist.varredura = 1
        mock_shist.ano_inicio_varredura = 1932
        mock_shist.anos_inicio_simulacoes = []
        case = make_case(tmp_path, dger=dger, shist=mock_shist)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        src = result["simulation"]["scenario_source"]
        assert src["historical_years"] == {"from": 1932, "to": 1932}

    def test_shist_absent_falls_back_to_legacy_range(self, tmp_path) -> None:
        """When shist.dat is not in NewaveFiles, fall back to the pre-Shist
        default (ano_inicial_historico+1 .. ano_inicio_estudo-1)."""
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=2,
            considera_reamostragem_cenarios=0,
            ano_inicial_historico=1931,
            ano_inicio=2026,
        )
        case = make_case(tmp_path, dger=dger)  # shist absent

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        src = result["simulation"]["scenario_source"]
        assert src["historical_years"] == {"from": 1932, "to": 2025}

    def test_historical_num_scenarios_matches_explicit_list(self, tmp_path) -> None:
        """In historical mode, num_scenarios is overridden to the number of
        distinct start-years — not the synthetic num_series_sinteticas."""
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=2,
            num_series=2000,
            ano_inicio=2024,
            num_anos=3,
            num_anos_pos=3,
        )

        mock_shist = MagicMock()
        mock_shist.varredura = 0
        mock_shist.anos_inicio_simulacoes = [1983, 1985, 1990]
        mock_shist.ano_inicio_varredura = 1932
        case = make_case(tmp_path, dger=dger, shist=mock_shist)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["simulation"]["selection"]["num_scenarios"] == 3

    def test_historical_num_scenarios_matches_range_length(self, tmp_path) -> None:
        """For a range historical_years, num_scenarios = to - from + 1."""
        # Horizon 6 → range [1932, 2018] → 87 years.
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=2,
            num_series=2000,
            ano_inicio=2024,
            num_anos=3,
            num_anos_pos=3,
        )

        mock_shist = MagicMock()
        mock_shist.varredura = 1
        mock_shist.ano_inicio_varredura = 1932
        mock_shist.anos_inicio_simulacoes = []
        case = make_case(tmp_path, dger=dger, shist=mock_shist)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["simulation"]["selection"]["num_scenarios"] == 2018 - 1932 + 1

    def test_non_historical_num_scenarios_uses_num_series_sinteticas(
        self, tmp_path
    ) -> None:
        """When simulation is not historical, num_scenarios stays
        dger.num_series_sinteticas — the override only applies in historical
        mode where the pool size is fixed."""
        dger = _make_dger_mock(
            tipo_execucao=1,
            tipo_simulacao_final=1,  # out_of_sample
            num_series=2000,
        )
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["simulation"]["selection"]["num_scenarios"] == 2000

    # -- considera_reamostragem_cenarios / training.scenario_source --

    def test_reamostragem_adds_training_scenario_source(self, tmp_path) -> None:
        dger = _make_dger_mock(tipo_execucao=1, considera_reamostragem_cenarios=1)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        src = result["training"]["scenario_source"]
        assert src["seed"] == 42
        assert src["inflow"]["scheme"] == "out_of_sample"

    def test_no_reamostragem_no_training_scenario_source(self, tmp_path) -> None:
        dger = _make_dger_mock(tipo_execucao=1, considera_reamostragem_cenarios=0)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert "scenario_source" not in result["training"]

    def test_reamostragem_ignored_when_training_disabled(self, tmp_path) -> None:
        dger = _make_dger_mock(
            tipo_execucao=0,
            considera_reamostragem_cenarios=1,
            tipo_simulacao_final=1,
        )
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["training"]["enabled"] is False
        assert "scenario_source" not in result["training"]

    # -- consideracao_media_anual_afluencias / estimation.order_selection --

    def test_order_selection_omitted_when_field_absent(self, tmp_path) -> None:
        """Old the source model files lacking the field → omit order_selection (novomodelo
        default)."""
        dger = _make_dger_mock(consideracao_media_anual_afluencias=None)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert "order_selection" not in result["estimation"]

    def test_order_selection_pacf_when_zero(self, tmp_path) -> None:
        """consideracao_media_anual_afluencias=0 → classical PAR(p) → 'pacf'."""
        dger = _make_dger_mock(consideracao_media_anual_afluencias=0)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["estimation"]["order_selection"] == "pacf"

    def test_order_selection_pacf_annual_when_three(self, tmp_path) -> None:
        """consideracao_media_anual_afluencias=3 (exact PAR(p)-A) → 'pacf_annual'."""
        dger = _make_dger_mock(consideracao_media_anual_afluencias=3)
        case = make_case(tmp_path, dger=dger)

        from novomodelo_bridge.newave.converters.temporal import convert_config

        result = convert_config(case)
        assert result["estimation"]["order_selection"] == "pacf_annual"

    def test_order_selection_pacf_annual_when_one_or_two(self, tmp_path) -> None:
        """Approximate PAR(p)-A variants (1, 2) also map to 'pacf_annual'."""
        from novomodelo_bridge.newave.converters.temporal import convert_config

        for value in (1, 2):
            dger = _make_dger_mock(consideracao_media_anual_afluencias=value)
            case = make_case(tmp_path, dger=dger)
            result = convert_config(case)
            assert result["estimation"]["order_selection"] == "pacf_annual"


# ---------------------------------------------------------------------------
# Tests: convert_inflow_stats
# ---------------------------------------------------------------------------


def _make_vazoes_mock(
    num_years: int = 10,
    postos: list[int] | None = None,
    start_year: int = 1931,
) -> MagicMock:
    """Build a synthetic Vazoes mock with ``num_years`` of monthly data.

    Parameters
    ----------
    num_years:
        Number of years in the historical record.
    postos:
        List of gauging station codes (columns in the DataFrame).
    start_year:
        First year of the historical record.
    """
    if postos is None:
        postos = [1, 2]

    rows = []
    rng = np.random.default_rng(42)
    for year in range(start_year, start_year + num_years):
        for month in range(1, 13):
            row = {"data": datetime.datetime(year, month, 1)}
            for posto in postos:
                row[posto] = float(rng.uniform(50.0, 500.0))
            rows.append(row)

    df = pd.DataFrame(rows)
    mock = MagicMock()
    mock.vazoes = df
    return mock


def _make_confhd_mock(hydro_to_posto: dict[int, int]) -> MagicMock:
    """Build a synthetic Confhd mock mapping hydro codes to postos."""
    rows = [
        {
            "codigo_usina": code,
            "posto": posto,
            "nome_usina": f"PLANT_{code}",
            "usina_existente": "EX",
            "codigo_usina_jusante": 0,
        }
        for code, posto in hydro_to_posto.items()
    ]
    df = pd.DataFrame(rows)
    mock = MagicMock()
    mock.usinas = df
    return mock


def _make_dger_inflow_mock(
    mes_inicio_estudo: int = 1,
    num_anos_estudo: int = 10,
    ano_inicio_estudo: int = 1941,
    ano_inicial_historico: int = 1931,
) -> MagicMock:
    """Build a minimal Dger mock for inflow stats tests."""
    mock = MagicMock()
    mock.mes_inicio_estudo = mes_inicio_estudo
    mock.num_anos_estudo = num_anos_estudo
    mock.num_anos_pos_estudo = 0
    mock.ano_inicio_estudo = ano_inicio_estudo
    mock.ano_inicial_historico = ano_inicial_historico
    return mock


class TestConvertInflowStats:
    @patch("novomodelo_bridge.newave.converters.stochastic.Vazoes")
    def test_returns_pyarrow_table(self, mock_vazoes_cls, tmp_path) -> None:
        (tmp_path / "vazoes.dat").touch()
        mock_vazoes_cls.read.return_value = _make_vazoes_mock(
            num_years=10, postos=[1, 2]
        )
        case = make_case(
            tmp_path,
            confhd=_make_confhd_mock({1: 1, 2: 2}),
            dger=_make_dger_inflow_mock(),
        )
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1, 2], thermal_codes=[])

        from novomodelo_bridge.newave.converters.stochastic import convert_inflow_stats

        result = convert_inflow_stats(case, id_map)
        assert isinstance(result, pa.Table)

    @patch("novomodelo_bridge.newave.converters.stochastic.Vazoes")
    def test_schema_columns(self, mock_vazoes_cls, tmp_path) -> None:
        (tmp_path / "vazoes.dat").touch()
        mock_vazoes_cls.read.return_value = _make_vazoes_mock(
            num_years=10, postos=[1, 2]
        )
        case = make_case(
            tmp_path,
            confhd=_make_confhd_mock({1: 1, 2: 2}),
            dger=_make_dger_inflow_mock(),
        )
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1, 2], thermal_codes=[])

        from novomodelo_bridge.newave.converters.stochastic import convert_inflow_stats

        result = convert_inflow_stats(case, id_map)
        assert result.column_names == ["hydro_id", "stage_id", "mean_m3s", "std_m3s"]

    @patch("novomodelo_bridge.newave.converters.stochastic.Vazoes")
    def test_column_types(self, mock_vazoes_cls, tmp_path) -> None:
        (tmp_path / "vazoes.dat").touch()
        mock_vazoes_cls.read.return_value = _make_vazoes_mock(
            num_years=10, postos=[1, 2]
        )
        case = make_case(
            tmp_path,
            confhd=_make_confhd_mock({1: 1, 2: 2}),
            dger=_make_dger_inflow_mock(),
        )
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1, 2], thermal_codes=[])

        from novomodelo_bridge.newave.converters.stochastic import convert_inflow_stats

        result = convert_inflow_stats(case, id_map)
        assert result.schema.field("hydro_id").type == pa.int32()
        assert result.schema.field("stage_id").type == pa.int32()
        assert result.schema.field("mean_m3s").type == pa.float64()
        assert result.schema.field("std_m3s").type == pa.float64()

    @patch("novomodelo_bridge.newave.converters.stochastic.Vazoes")
    def test_january_mean_correct(self, mock_vazoes_cls, tmp_path) -> None:
        """mean_m3s for January stages must equal the mean of all January values."""
        (tmp_path / "vazoes.dat").touch()

        num_years = 10
        rows = []
        jan_vals = []
        rng = np.random.default_rng(0)
        for year in range(1931, 1931 + num_years):
            for month in range(1, 13):
                v = float(rng.uniform(100.0, 400.0))
                rows.append({"data": datetime.datetime(year, month, 1), 1: v})
                if month == 1:
                    jan_vals.append(v)
        df = pd.DataFrame(rows)
        mock_vazoes = MagicMock()
        mock_vazoes.vazoes = df
        mock_vazoes_cls.read.return_value = mock_vazoes
        case = make_case(
            tmp_path,
            confhd=_make_confhd_mock({1: 1}),
            dger=_make_dger_inflow_mock(mes_inicio_estudo=1, num_anos_estudo=num_years),
        )

        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1], thermal_codes=[])

        from novomodelo_bridge.newave.converters.stochastic import convert_inflow_stats

        result = convert_inflow_stats(case, id_map)
        df_result = result.to_pydict()

        expected_jan_mean = float(np.mean(jan_vals))
        jan_stage_ids = [
            sid
            for sid, mean in zip(df_result["stage_id"], df_result["mean_m3s"])
            if abs(mean - expected_jan_mean) < 1e-9
        ]
        assert len(jan_stage_ids) > 0, "No January stage found with expected mean"

    @patch("novomodelo_bridge.newave.converters.stochastic.Vazoes")
    def test_empty_vazoes_raises_file_not_found(
        self, mock_vazoes_cls, tmp_path
    ) -> None:
        (tmp_path / "vazoes.dat").touch()
        mock_obj = MagicMock()
        mock_obj.vazoes = pd.DataFrame()
        mock_vazoes_cls.read.return_value = mock_obj
        case = make_case(tmp_path, confhd=_make_confhd_mock({}), dger=_make_dger_mock())

        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[], thermal_codes=[])

        from novomodelo_bridge.newave.converters.stochastic import convert_inflow_stats

        with pytest.raises(FileNotFoundError, match="vazoes.dat not found or empty"):
            convert_inflow_stats(case, id_map)

    @patch("novomodelo_bridge.newave.converters.stochastic.Vazoes")
    def test_missing_posto_column_produces_zero(
        self, mock_vazoes_cls, tmp_path
    ) -> None:
        """Hydro whose posto column is absent in vazoes -> mean/std = 0.0."""
        (tmp_path / "vazoes.dat").touch()
        mock_vazoes_cls.read.return_value = _make_vazoes_mock(num_years=5, postos=[1])
        case = make_case(
            tmp_path,
            confhd=_make_confhd_mock({1: 1, 2: 99}),
            dger=_make_dger_inflow_mock(num_anos_estudo=5),
        )
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1, 2], thermal_codes=[])

        from novomodelo_bridge.newave.converters.stochastic import convert_inflow_stats

        result = convert_inflow_stats(case, id_map)
        df = result.to_pydict()
        hydro2_means = [m for hid, m in zip(df["hydro_id"], df["mean_m3s"]) if hid == 1]
        assert all(m == 0.0 for m in hydro2_means)


# ---------------------------------------------------------------------------
# Tests: convert_load_stats
# ---------------------------------------------------------------------------


def _make_sistema_mock(
    subsystem_codes: list[int],
    num_months: int = 12,
    start_year: int = 2020,
    start_month: int = 1,
) -> MagicMock:
    """Build a synthetic Sistema mock with load data."""
    rows = []
    rng = np.random.default_rng(7)
    year, month = start_year, start_month
    for _ in range(num_months):
        for code in subsystem_codes:
            rows.append(
                {
                    "codigo_submercado": code,
                    "data": datetime.datetime(year, month, 1),
                    "valor": float(rng.uniform(1000.0, 5000.0)),
                }
            )
        month += 1
        if month > 12:
            month = 1
            year += 1
    df = pd.DataFrame(rows)
    mock = MagicMock()
    mock.mercado_energia = df
    return mock


def _make_load_stats_dger_mock(
    *,
    mes_inicio: int = 1,
    ano_inicio: int = 2020,
    num_anos: int = 5,
    num_anos_pos: int = 0,
) -> MagicMock:
    """Minimal Dger mock for convert_load_stats tests."""
    dger = MagicMock()
    dger.mes_inicio_estudo = mes_inicio
    dger.ano_inicio_estudo = ano_inicio
    dger.num_anos_estudo = num_anos
    dger.num_anos_pos_estudo = num_anos_pos
    return dger


class TestConvertLoadStats:
    def test_returns_pyarrow_table(self, tmp_path) -> None:
        case = make_case(
            tmp_path,
            sistema=_make_sistema_mock(subsystem_codes=[1, 2, 3, 4], num_months=60),
            dger=_make_load_stats_dger_mock(num_anos=5),
        )
        id_map = _make_id_map_buses([1, 2, 3, 4])

        from novomodelo_bridge.newave.converters.stochastic import convert_load_stats

        result = convert_load_stats(case, id_map)
        assert isinstance(result, pa.Table)

    def test_schema_columns(self, tmp_path) -> None:
        case = make_case(
            tmp_path,
            sistema=_make_sistema_mock(subsystem_codes=[1, 2], num_months=12),
            dger=_make_load_stats_dger_mock(num_anos=1),
        )
        id_map = _make_id_map_buses([1, 2])

        from novomodelo_bridge.newave.converters.stochastic import convert_load_stats

        result = convert_load_stats(case, id_map)
        assert result.column_names == ["bus_id", "stage_id", "mean_mw", "std_mw"]

    def test_column_types(self, tmp_path) -> None:
        case = make_case(
            tmp_path,
            sistema=_make_sistema_mock(subsystem_codes=[1, 2], num_months=12),
            dger=_make_load_stats_dger_mock(num_anos=1),
        )
        id_map = _make_id_map_buses([1, 2])

        from novomodelo_bridge.newave.converters.stochastic import convert_load_stats

        result = convert_load_stats(case, id_map)
        assert result.schema.field("bus_id").type == pa.int32()
        assert result.schema.field("stage_id").type == pa.int32()
        assert result.schema.field("mean_mw").type == pa.float64()
        assert result.schema.field("std_mw").type == pa.float64()

    def test_row_count_four_subsystems_60_months(self, tmp_path) -> None:
        case = make_case(
            tmp_path,
            sistema=_make_sistema_mock(subsystem_codes=[1, 2, 3, 4], num_months=60),
            dger=_make_load_stats_dger_mock(num_anos=5),
        )
        id_map = _make_id_map_buses([1, 2, 3, 4])

        from novomodelo_bridge.newave.converters.stochastic import convert_load_stats

        result = convert_load_stats(case, id_map)
        assert result.num_rows == 4 * 60

    def test_std_mw_all_zeros(self, tmp_path) -> None:
        case = make_case(
            tmp_path,
            sistema=_make_sistema_mock(subsystem_codes=[1, 2, 3, 4], num_months=60),
            dger=_make_load_stats_dger_mock(num_anos=5),
        )
        id_map = _make_id_map_buses([1, 2, 3, 4])

        from novomodelo_bridge.newave.converters.stochastic import convert_load_stats

        result = convert_load_stats(case, id_map)
        std_vals = result.column("std_mw").to_pylist()
        assert all(v == 0.0 for v in std_vals)

    def test_stage_ids_per_bus_sequential(self, tmp_path) -> None:
        case = make_case(
            tmp_path,
            sistema=_make_sistema_mock(subsystem_codes=[1, 2], num_months=12),
            dger=_make_load_stats_dger_mock(num_anos=1),
        )
        id_map = _make_id_map_buses([1, 2])

        from novomodelo_bridge.newave.converters.stochastic import convert_load_stats

        result = convert_load_stats(case, id_map)
        df = result.to_pydict()
        for bus_id in [0, 1]:
            stages = [
                sid for bid, sid in zip(df["bus_id"], df["stage_id"]) if bid == bus_id
            ]
            assert stages == list(range(12))

    def test_mean_mw_values_match_source(self, tmp_path) -> None:
        """mean_mw values must equal the raw 'valor' from mercado_energia."""
        # Deterministic load: 2 subsystems, 2 months.
        rows = [
            {
                "codigo_submercado": 1,
                "data": datetime.datetime(2020, 1, 1),
                "valor": 3000.0,
            },
            {
                "codigo_submercado": 1,
                "data": datetime.datetime(2020, 2, 1),
                "valor": 2800.0,
            },
            {
                "codigo_submercado": 2,
                "data": datetime.datetime(2020, 1, 1),
                "valor": 1500.0,
            },
            {
                "codigo_submercado": 2,
                "data": datetime.datetime(2020, 2, 1),
                "valor": 1600.0,
            },
        ]
        mock = MagicMock()
        mock.mercado_energia = pd.DataFrame(rows)
        # 1 year study so stage 0=Jan, stage 1=Feb.
        case = make_case(
            tmp_path,
            sistema=mock,
            dger=_make_load_stats_dger_mock(num_anos=1),
        )
        id_map = _make_id_map_buses([1, 2])

        from novomodelo_bridge.newave.converters.stochastic import convert_load_stats

        result = convert_load_stats(case, id_map)
        df = result.to_pydict()

        # Bus 0 (subsystem 1), stage 0 -> 3000.0; stage 1 -> 2800.0.
        bus0_means = [m for bid, m in zip(df["bus_id"], df["mean_mw"]) if bid == 0]
        assert bus0_means[0] == pytest.approx(3000.0)
        assert bus0_means[1] == pytest.approx(2800.0)


def _make_cadic_mock(rows: list[dict]) -> MagicMock:
    """Build a Cadic mock whose ``cargas`` matches inewave's schema."""
    mock = MagicMock()
    mock.cargas = pd.DataFrame(rows) if rows else None
    return mock


class TestParseCadical:
    """``parse_cadical`` aggregates inewave's ``Cadic.cargas`` into a lookup."""

    @patch("novomodelo_bridge.newave.converters.stochastic.Cadic")
    def test_sums_razoes_per_subsystem_year_month(
        self, mock_cadic_cls, tmp_path
    ) -> None:
        from novomodelo_bridge.newave.converters.stochastic import parse_cadical

        # Two razões for (sub 1, 2024-01) sum; sub 2 and POS (year 9999) distinct.
        mock_cadic_cls.read.return_value = _make_cadic_mock(
            [
                {
                    "codigo_submercado": 1,
                    "nome_submercado": "SE",
                    "razao": "SMALL PLANTS",
                    "data": datetime.datetime(2024, 1, 1),
                    "valor": 10.0,
                },
                {
                    "codigo_submercado": 1,
                    "nome_submercado": "SE",
                    "razao": "OTHER",
                    "data": datetime.datetime(2024, 1, 1),
                    "valor": 2.5,
                },
                {
                    "codigo_submercado": 2,
                    "nome_submercado": "S",
                    "razao": "SMALL PLANTS",
                    "data": datetime.datetime(2024, 1, 1),
                    "valor": 7.0,
                },
                {
                    "codigo_submercado": 1,
                    "nome_submercado": "SE",
                    "razao": "SMALL PLANTS",
                    "data": datetime.datetime(9999, 6, 1),  # POS sentinel
                    "valor": 4.0,
                },
            ]
        )
        result = parse_cadical(tmp_path / "c_adic.dat")
        assert result[(1, 2024, 1)] == pytest.approx(12.5)  # 10.0 + 2.5
        assert result[(2, 2024, 1)] == pytest.approx(7.0)
        assert result[(1, 9999, 6)] == pytest.approx(4.0)

    @patch("novomodelo_bridge.newave.converters.stochastic.Cadic")
    def test_skips_nan_values(self, mock_cadic_cls, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.stochastic import parse_cadical

        mock_cadic_cls.read.return_value = _make_cadic_mock(
            [
                {
                    "codigo_submercado": 1,
                    "nome_submercado": "SE",
                    "razao": "X",
                    "data": datetime.datetime(2024, 3, 1),
                    "valor": float("nan"),
                },
            ]
        )
        assert parse_cadical(tmp_path / "c_adic.dat") == {}

    @patch("novomodelo_bridge.newave.converters.stochastic.Cadic")
    def test_empty_cargas_returns_empty(self, mock_cadic_cls, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.stochastic import parse_cadical

        mock_cadic_cls.read.return_value = _make_cadic_mock([])  # cargas is None
        assert parse_cadical(tmp_path / "c_adic.dat") == {}

    @patch("novomodelo_bridge.newave.converters.stochastic.Cadic")
    def test_cadic_additions_reach_load(self, mock_cadic_cls, tmp_path) -> None:
        """C_ADIC must-take energy is added to the per-(subsystem, stage) load."""
        from novomodelo_bridge.newave.converters.stochastic import convert_load_stats

        rows = [
            {
                "codigo_submercado": 1,
                "data": datetime.datetime(2024, month, 1),
                "valor": 1000.0,
            }
            for month in (1, 2)
        ]
        mock = MagicMock()
        mock.mercado_energia = pd.DataFrame(rows)
        mock_cadic_cls.read.return_value = _make_cadic_mock(
            [
                {
                    "codigo_submercado": 1,
                    "nome_submercado": "SE",
                    "razao": "SMALL",
                    "data": datetime.datetime(2024, 1, 1),
                    "valor": 50.0,
                },
            ]
        )
        id_map = _make_id_map_buses([1])
        case = make_case(
            make_nw_files(tmp_path, c_adic=tmp_path / "c_adic.dat"),
            sistema=mock,
            dger=_make_load_stats_dger_mock(ano_inicio=2024, num_anos=1),
        )

        df = convert_load_stats(case, id_map).to_pydict()
        bus0 = [m for bid, m in zip(df["bus_id"], df["mean_mw"]) if bid == 0]
        assert bus0[0] == pytest.approx(1050.0)  # Jan load + C_ADIC
        assert bus0[1] == pytest.approx(1000.0)  # Feb load, no C_ADIC

    @patch("novomodelo_bridge.newave.converters.stochastic.Cadic")
    def test_switched_off_cadic_is_never_read(self, mock_cadic_cls, tmp_path) -> None:
        """``CONS. CARGA ADICIONAL = 0`` leaves a present c_adic.dat unread and
        the load unchanged, with one INFO diagnostic naming the switch."""
        from novomodelo_bridge.core import diagnostics as dx
        from novomodelo_bridge.newave.converters.stochastic import convert_load_stats

        mock = MagicMock()
        mock.mercado_energia = pd.DataFrame(
            [
                {
                    "codigo_submercado": 1,
                    "data": datetime.datetime(2024, month, 1),
                    "valor": 1000.0,
                }
                for month in (1, 2)
            ]
        )
        dger = _make_load_stats_dger_mock(ano_inicio=2024, num_anos=1)
        dger.considera_carga_adicional = 0
        case = make_case(
            make_nw_files(tmp_path, c_adic=tmp_path / "c_adic.dat"),
            sistema=mock,
            dger=dger,
        )

        with dx.collect() as collected:
            df = convert_load_stats(case, _make_id_map_buses([1])).to_pydict()

        mock_cadic_cls.read.assert_not_called()
        assert df["mean_mw"][:2] == [pytest.approx(1000.0), pytest.approx(1000.0)]
        assert [d.code for d in collected] == ["dger-switch-off"]
        assert "c_adic.dat" in collected[0].title


# ---------------------------------------------------------------------------
# Helpers for convert_past_inflows tests
# ---------------------------------------------------------------------------


def _make_vazpast_mock(
    postos: list[int],
    num_months: int = 12,
    end_year: int = 2019,
    end_month: int = 12,
) -> MagicMock:
    """Build a synthetic Vazpast mock with monthly tendency data.

    The ``tendencia`` DataFrame has columns ``codigo_usina``, ``nome_usina``,
    ``mes`` (1-12), and ``valor``, with one row per (posto, calendar month).
    This matches the format expected by ``convert_past_inflows`` after the
    production code was updated to read ``vazpast_obj.tendencia`` instead of
    ``vazpast_obj.vazoes``.

    ``end_year`` and ``end_month`` are accepted for API compatibility but
    the tendencia format is calendar-month based, not date based.
    ``num_months`` is capped at 12 (the tendencia has at most one row per
    calendar month per plant).
    """
    rng = np.random.default_rng(99)
    rows = []
    n_months = min(num_months, 12)
    for posto in postos:
        for mes in range(1, n_months + 1):
            rows.append(
                {
                    "codigo_usina": posto,
                    "nome_usina": f"PLANT_{posto}",
                    "mes": mes,
                    "valor": float(rng.uniform(50.0, 500.0)),
                }
            )
    df = pd.DataFrame(rows)
    mock = MagicMock()
    mock.tendencia = df
    return mock


def _make_confhd_posto_mock(posto_to_code: dict[int, int]) -> MagicMock:
    """Build a Confhd mock mapping postos to hydro codes.

    Includes all columns needed by _build_upstream_postos (cascade logic).
    All plants are headwater (no upstream) unless overridden.
    """
    rows = [
        {
            "posto": p,
            "codigo_usina": c,
            "nome_usina": f"PLANT_{c}",
            "usina_existente": "EX",
            "codigo_usina_jusante": 0,
        }
        for p, c in posto_to_code.items()
    ]
    df = pd.DataFrame(rows)
    mock = MagicMock()
    mock.usinas = df
    return mock


# ---------------------------------------------------------------------------
# Tests: _build_upstream_postos — NE/NC cascade bypass
# ---------------------------------------------------------------------------


def _confhd_row(
    code: int,
    posto: int,
    downstream: int,
    status: str = "EX",
) -> dict[str, object]:
    return {
        "codigo_usina": code,
        "nome_usina": f"PLANT_{code}",
        "posto": posto,
        "codigo_usina_jusante": downstream,
        "usina_existente": status,
    }


class TestBuildUpstreamPostosExpansionPlant:
    """An ``EE`` plant is in service, so its posto is a real inflow node.
    Walking through it instead credits its whole natural inflow to the plant
    below, inflating that plant's incremental series."""

    def test_ee_plant_between_two_ex_plants_is_its_own_node(self) -> None:
        from novomodelo_bridge.newave.converters.stochastic import (
            _build_upstream_postos,
        )

        # A (EX, posto 100) -> B (EE, posto 200) -> C (EX, posto 300)
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 2, "EX"),
                _confhd_row(2, 200, 3, "EE"),
                _confhd_row(3, 300, 0, "EX"),
            ]
        )
        upstream = _build_upstream_postos(confhd)
        assert upstream.get(200) == [100]
        assert upstream.get(300) == [200]


class TestBuildUpstreamPostosNonExistingBypass:
    """``_build_upstream_postos`` must walk through NE/NC plants so the
    posto-level cascade stays connected.  Without this, the downstream
    EX plant's incremental inflow fails to subtract the upstream EX
    plant's natural inflow."""

    def test_nc_plant_between_two_ex_plants_keeps_posto_edge(self) -> None:
        from novomodelo_bridge.newave.converters.stochastic import (
            _build_upstream_postos,
        )

        # A (EX, posto 100) -> B (NC, posto 200) -> C (EX, posto 300)
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 2, "EX"),
                _confhd_row(2, 200, 3, "NC"),
                _confhd_row(3, 300, 0, "EX"),
            ]
        )
        upstream = _build_upstream_postos(confhd)
        assert upstream.get(300) == [100]

    def test_ne_plant_between_two_ex_plants_keeps_posto_edge(self) -> None:
        from novomodelo_bridge.newave.converters.stochastic import (
            _build_upstream_postos,
        )

        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 2, "EX"),
                _confhd_row(2, 200, 3, "NE"),
                _confhd_row(3, 300, 0, "EX"),
            ]
        )
        upstream = _build_upstream_postos(confhd)
        assert upstream.get(300) == [100]

    def test_consecutive_absent_plants_collapse_to_single_edge(self) -> None:
        from novomodelo_bridge.newave.converters.stochastic import (
            _build_upstream_postos,
        )

        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 2, "EX"),
                _confhd_row(2, 200, 3, "NC"),
                _confhd_row(3, 300, 4, "NE"),
                _confhd_row(4, 400, 0, "EX"),
            ]
        )
        upstream = _build_upstream_postos(confhd)
        assert upstream.get(400) == [100]
        # No edge to the bypassed postos 200/300.
        assert 200 not in upstream
        assert 300 not in upstream

    def test_absent_at_chain_end_yields_no_edge(self) -> None:
        from novomodelo_bridge.newave.converters.stochastic import (
            _build_upstream_postos,
        )

        # A (EX) -> B (NC) -> 0 (terminal); A has no downstream edge.
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 2, "EX"),
                _confhd_row(2, 200, 0, "NC"),
            ]
        )
        upstream = _build_upstream_postos(confhd)
        assert upstream == {}

    def test_direct_ex_to_ex_edge_preserved(self) -> None:
        from novomodelo_bridge.newave.converters.stochastic import (
            _build_upstream_postos,
        )

        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 2, "EX"),
                _confhd_row(2, 200, 0, "EX"),
            ]
        )
        upstream = _build_upstream_postos(confhd)
        assert upstream.get(200) == [100]


class TestBuildUpstreamPostosFillingAdmission:
    """An admitted ``NE``-with-filling plant *receives* inflow, so its posto
    must enter the map as a real node — an upstream plant forms a posto edge
    **to** the filling plant instead of walking through it.  The JURUENA case
    (code 309, posto 226, ``NE``) is the live exemplar."""

    def test_posto_map_includes_filling_node(self) -> None:
        from novomodelo_bridge.newave.converters.stochastic import (
            _build_upstream_postos,
        )

        # Upstream EX (posto 100) -> JURUENA code 309 (NE, posto 226) -> 0.
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 309, "EX"),
                _confhd_row(309, 226, 0, "NE"),
            ]
        )
        upstream = _build_upstream_postos(confhd, filling_codes={309})
        # 226 is now a real node: it appears as an edge endpoint.
        assert 226 in upstream

    def test_posto_map_unchanged_without_filling_codes(self) -> None:
        from novomodelo_bridge.newave.converters.stochastic import (
            _build_upstream_postos,
        )

        # Same cascade: with the NE plant walked through, the EX upstream has
        # no downstream EX node, so no edge survives.
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 309, "EX"),
                _confhd_row(309, 226, 0, "NE"),
            ]
        )
        baseline = _build_upstream_postos(confhd)
        # filling_codes=None must be byte-identical to the EX-only result, and
        # the empty set must match too (None is normalised to set()).
        assert _build_upstream_postos(confhd, filling_codes=None) == baseline
        assert _build_upstream_postos(confhd, filling_codes=set()) == baseline
        # The NE posto 226 is NOT a node in the EX-only map.
        assert 226 not in baseline

    def test_posto_edge_to_filling_plant(self) -> None:
        from novomodelo_bridge.newave.converters.stochastic import (
            _build_upstream_postos,
        )

        # Upstream EX (posto 100) -> JURUENA code 309 (NE-filling, posto 226).
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 309, "EX"),
                _confhd_row(309, 226, 0, "NE"),
            ]
        )
        upstream = _build_upstream_postos(confhd, filling_codes={309})
        # The upstream EX (posto 100) forms an edge TO JURUENA's posto 226.
        assert upstream.get(226) == [100]

    def test_filling_plant_with_nan_posto_is_skipped(self) -> None:
        from novomodelo_bridge.newave.converters.stochastic import (
            _build_upstream_postos,
        )

        # JURUENA admitted but its posto is NaN — same pd.isna guard as the
        # EX path: it is skipped, so the upstream edge does not resolve to it.
        confhd = pd.DataFrame(
            [
                _confhd_row(1, 100, 309, "EX"),
                {
                    "codigo_usina": 309,
                    "nome_usina": "JURUENA",
                    "posto": float("nan"),
                    "codigo_usina_jusante": 0,
                    "usina_existente": "NE",
                },
            ]
        )
        upstream = _build_upstream_postos(confhd, filling_codes={309})
        assert 226 not in upstream
        # Walk-through finds no downstream EX, so no edge survives.
        assert upstream == {}


# ---------------------------------------------------------------------------
# Tests: vazoes.dat width
# ---------------------------------------------------------------------------


def _write_vazoes(tmp_path: Path, num_bytes: int) -> Path:
    """A vazoes.dat of the given byte size; only its size matters."""
    path = tmp_path / "vazoes.dat"
    path.write_bytes(b"\x00" * num_bytes)
    return path


def _count(path: Path) -> int:
    return _posto_count(path, hist_start_year=1931, study_start_year=2026)


class TestPostoCount:
    @pytest.mark.parametrize(
        ("num_values", "expected"),
        [
            # One year at 320; not even a whole number of months at 600.
            (320 * 12, 320),
            (600 * 12, 600),
            # 1931-2026 at 600 is also a whole number of years at 320, but
            # one that would run to 2110.
            (600 * 12 * 96, 600),
        ],
    )
    def test_the_only_fitting_width_is_used_silently(
        self, tmp_path, num_values, expected
    ) -> None:
        with dx.collect() as collected:
            assert _count(_write_vazoes(tmp_path, 4 * num_values)) == expected
        assert collected == []

    def test_ambiguous_size_assumes_320_and_warns(self, tmp_path) -> None:
        # 15 years at 320 postos is also 8 years at 600, both ending by 2026.
        with dx.collect() as collected:
            count = _count(_write_vazoes(tmp_path, 4 * 320 * 12 * 15))

        assert count == 320
        assert len(collected) == 1
        diag = collected[0]
        assert diag.code == "vazoes-posto-count-ambiguous"
        assert diag.severity is Severity.WARNING
        assert diag.category == "Historical inflows"
        assert "320" in diag.summary
        assert "600" in diag.summary

    @pytest.mark.parametrize(
        ("num_bytes", "message"),
        [
            (4 * 320 * 12 + 2, "4-byte values"),
            (4 * 1000, "320 or 600"),
            # 13 months at 320 postos: the historical record ends in December.
            (4 * 320 * 13, "320 or 600"),
            # 100 years at 320 postos from 1931 runs to 2030.
            (4 * 320 * 12 * 100, "to 2030 at 320 postos"),
        ],
    )
    def test_size_that_fits_no_width_is_rejected(
        self, tmp_path, num_bytes, message
    ) -> None:
        with pytest.raises(FieldParseError, match=message):
            _count(_write_vazoes(tmp_path, num_bytes))

    def test_empty_file_falls_back_to_the_default(self, tmp_path) -> None:
        # Emptiness is the callers' finding, not the width derivation's.
        with dx.collect() as collected:
            assert _count(_write_vazoes(tmp_path, 0)) == 320
        assert collected == []


def test_six_hundred_posto_matrix_is_laid_out_at_full_width(tmp_path) -> None:
    """Through the real reader: a width it ignores, rejects or lays out
    transposed would mis-assign every posto's series. The 320-posto default is
    covered end to end by the mini deck.
    """
    path = tmp_path / "vazoes.dat"
    # Sequential values, unlike ``_write_vazoes``' zeros: the row-major
    # assertions below are what pin the width.
    path.write_bytes(b"".join(struct.pack("<i", v) for v in range(600 * 12)))

    df = _read_vazoes(path, _make_dger_mock(ano_inicio=2026)).vazoes

    assert df.shape == (12, 600)
    assert df.iloc[0, 0] == 0
    assert df.iloc[0, -1] == 599
    assert df.iloc[1, 0] == 600


class TestLoadFactorsWithoutBlockFactors:
    """A deck whose patamar file carries no block load factors."""

    _ID_MAP = NewaveIdMap(subsystem_ids=[1], hydro_codes=[], thermal_codes=[])

    def _convert(self, tmp_path, num_blocks: int) -> dict:
        from novomodelo_bridge.newave.converters.stochastic import convert_load_factors

        patamar = MagicMock(numero_patamares=num_blocks, carga_patamares=None)
        case = make_case(
            make_nw_files(tmp_path, patamar=tmp_path / "patamar.eas"),
            patamar=patamar,
            dger=_make_dger_mock(),
        )
        return convert_load_factors(case, self._ID_MAP)

    def test_single_block_needs_no_factors_and_is_not_reported(
        self, tmp_path, caplog
    ) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger="novomodelo_bridge"):
            result = self._convert(tmp_path, 1)

        assert result["load_factors"] == []
        assert caplog.text == ""

    def test_multi_block_without_factors_names_the_deck_file(
        self, tmp_path, caplog
    ) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger="novomodelo_bridge"):
            result = self._convert(tmp_path, 3)

        assert result["load_factors"] == []
        assert "patamar.eas declares 3 load blocks" in caplog.text
