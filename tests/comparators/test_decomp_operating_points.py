"""Tests for ``comparators.decomp.operating_points``.

The case is the committed DECOMP mini-deck. The source model's planes, fit
windows and deviation table are the real-format excerpts under
``tests/fixtures/decomp_results/`` (trimmed from a March 2026 weekly run whose
plant data the mini-deck carries). Both models' per-block operation is
synthetic, the source model's placed at the excerpt's printed operating points.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.decomp.operating_points import (
    compare_operating_points,
)
from novomodelo_bridge.comparators.decomp.readers import (
    read_avl_cortesfpha,
    read_dec_desvfpha,
)
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.cadastro import (
    build_effective_cadastro,
    effective_storage_range,
)

_EXCERPT = Path(__file__).resolve().parents[1] / "fixtures" / "decomp_results"
_MODULE = "novomodelo_bridge.comparators.decomp.operating_points"
_HOURS = (30.0, 74.0, 64.0)
_CAMARGOS, _ITUTINGA = 1, 2
_PIMENTAL, _BELO_MONTE = 314, 288


@dataclasses.dataclass
class _Runs:
    usih: pl.DataFrame
    sist: pl.DataFrame
    novomodelo: pl.DataFrame
    prices: pl.DataFrame
    novomodelo_planes: pl.DataFrame | None
    hydro_ids: tuple[int, ...]


def _source_block(
    code: int,
    *,
    estagio: int = 1,
    cenario: int = 1,
    patamar: int,
    v_useful: float,
    q: float,
    gh: float,
    s: float = 0.0,
    submarket: str = "SE",
) -> dict[str, object]:
    return {
        "codigo_usina": code,
        "estagio": estagio,
        "cenario": cenario,
        "patamar": patamar,
        "duracao": _HOURS[patamar - 1],
        "nome_submercado": submarket,
        "volume_util_inicial_hm3": v_useful,
        "volume_util_final_hm3": v_useful,
        "vazao_turbinada_m3s": q,
        "vazao_vertida_m3s": s,
        "vazao_defluente_m3s": q + s,
        "geracao_MW": gh,
        "vazao_incremental_m3s": None,
    }


def _source_stage(
    code: int, *, estagio: int = 1, cenario: int = 1, incremental: float = 0.0
) -> dict[str, object]:
    """The per-stage aggregate row the source model prints next to its blocks."""
    return {
        "codigo_usina": code,
        "estagio": estagio,
        "cenario": cenario,
        "patamar": None,
        "duracao": sum(_HOURS),
        "nome_submercado": None,
        "volume_util_inicial_hm3": None,
        "volume_util_final_hm3": None,
        "vazao_turbinada_m3s": None,
        "vazao_vertida_m3s": None,
        "vazao_defluente_m3s": None,
        "geracao_MW": None,
        "vazao_incremental_m3s": incremental,
    }


def _sist(
    *, submarket: str = "SE", stages: dict[int, tuple[int, ...]] | None = None
) -> pl.DataFrame:
    stages = {1: (1,)} if stages is None else stages
    return pl.DataFrame(
        [
            {
                "estagio": estagio,
                "cenario": cenario,
                "patamar": patamar,
                "nome_submercado": submarket,
                "cmo": 100.0,
            }
            for estagio, cenarios in stages.items()
            for cenario in cenarios
            for patamar in (1, 2, 3)
        ]
    )


def _novomodelo_block(
    hydro_id: int,
    *,
    stage: int = 0,
    scenario: int = 0,
    block: int,
    storage: tuple[float, float],
    q: float,
    gh: float,
    s: float = 0.0,
) -> dict[str, object]:
    return {
        "scenario_id": scenario,
        "stage_id": stage,
        "block_id": block,
        "hydro_id": hydro_id,
        "hours": _HOURS[block],
        "turbined_m3s": q,
        "spillage_m3s": s,
        "outflow_m3s": q + s,
        "incremental_inflow_m3s": 0.0,
        "storage_initial_hm3": storage[0],
        "storage_final_hm3": storage[1],
        "generation_mw": gh,
        "water_value_per_hm3": 50.0,
        "accumulated_productivity_mw_per_m3s": 0.5,
    }


def _prices(
    *, stages: dict[int, tuple[int, ...]] | None = None, bus: int = 0
) -> pl.DataFrame:
    stages = {0: (0,)} if stages is None else stages
    return pl.DataFrame(
        [
            {
                "scenario_id": scenario,
                "stage_id": stage,
                "block_id": block,
                "bus_id": bus,
                "spot_price": 100.0,
            }
            for stage, scenarios in stages.items()
            for scenario in scenarios
            for block in (0, 1, 2)
        ]
    )


def _novomodelo_planes(hydro_id: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "hydro_id": [hydro_id, hydro_id],
            "stage_id": [0, 0],
            "gamma_0": [-10.0, -8.0],
            "gamma_v": [0.02, 0.022],
            "gamma_q": [0.215, 0.197],
            "gamma_s": [-0.004, -0.005],
            "kappa": [1.0, 1.0],
        }
    )


def _camargos_runs(case: DecompCase) -> _Runs:
    """Camargos and Itutinga, its downstream plant, at the printed points of
    stage 1, the source model generating its printed FPHA."""
    eff, _ = build_effective_cadastro(case.dadger, case.hidr, case.calendar)
    printed = read_dec_desvfpha(_EXCERPT)
    blocks = [
        _source_block(
            code,
            patamar=patamar,
            v_useful=v - effective_storage_range(eff, code, 0)[0],
            q=q,
            gh=gh,
        )
        for code, patamar, v, q, gh in printed.select(
            "codigo_usina",
            "patamar",
            "volume_total_hm3",
            "vazao_turbinada_m3s",
            "geracao_hidraulica_fpha",
        ).iter_rows()
    ]
    stages = [_source_stage(code) for code in (_CAMARGOS, _ITUTINGA)]
    camargos, itutinga = (case.id_map.hydro_id(c) for c in (_CAMARGOS, _ITUTINGA))
    novomodelo = [
        _novomodelo_block(camargos, block=b, storage=(700.0, 690.0), q=150.0, gh=34.0)
        for b in range(3)
    ] + [
        _novomodelo_block(itutinga, block=b, storage=(10.64, 10.64), q=200.0, gh=47.0)
        for b in range(3)
    ]
    return _Runs(
        usih=pl.DataFrame(blocks + stages),
        sist=_sist(),
        novomodelo=pl.DataFrame(novomodelo),
        prices=_prices(),
        novomodelo_planes=_novomodelo_planes(camargos),
        hydro_ids=(camargos, itutinga),
    )


def _compare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: DecompCase,
    runs: _Runs,
    names: dict[int, str] | None = None,
    fit_config: dict[str, list[float]] | None = None,
):
    decomp_dir = tmp_path / "decomp"
    decomp_dir.mkdir(exist_ok=True)
    for name in ("avl_cortesfpha_dec.rv2", "eco_fpha_.rv2", "dec_desvfpha.rv2"):
        shutil.copy(_EXCERPT / name, decomp_dir / name)
    monkeypatch.setattr(f"{_MODULE}.read_dec_oper_usih", lambda _d: runs.usih)
    monkeypatch.setattr(f"{_MODULE}.read_dec_oper_sist", lambda _d: runs.sist)
    readers = f"{_MODULE}.novomodelo_readers"
    patches = {
        "read_novomodelo_hydro_block_operation": runs.novomodelo,
        "read_novomodelo_bus_block_prices": runs.prices,
        "read_novomodelo_hydro_bus_labels": {h: frozenset({0}) for h in runs.hydro_ids},
        "read_novomodelo_fpha_planes": runs.novomodelo_planes,
        "read_novomodelo_fpha_fit_config": pl.DataFrame(
            {
                "hydro_id": [runs.hydro_ids[0]],
                "stage_id": [0],
                "volume_min_hm3": [600.0],
                "volume_max_hm3": [792.0],
                "spillage_fit_max_m3s": [50.0],
                "turbined_fit_max_m3s": [None],
            }
            | (fit_config or {}),
            schema_overrides={"turbined_fit_max_m3s": pl.Float64},
        ),
        "read_novomodelo_hydro_metadata": {
            h: {"max_turbined_m3s": 220.0} for h in runs.hydro_ids
        },
        "read_novomodelo_hydro_per_stage_bounds": pl.DataFrame(
            {
                "entity_id": [runs.hydro_ids[0]],
                "stage_id": [0],
                "max_turbined_m3s": [180.0],
            }
        ),
    }
    for name, value in patches.items():
        monkeypatch.setattr(f"{readers}.{name}", lambda _d, value=value: value)
    names = {h: f"P{h}" for h in runs.hydro_ids} if names is None else names
    return compare_operating_points(
        decomp_dir, case, tmp_path / "novomodelo", pl.DataFrame(), names
    )


def _with_cmo(sist: pl.DataFrame, **match: int) -> pl.DataFrame:
    """*sist* with a zero CMO on the rows matching every ``column=value``."""
    hit = pl.all_horizontal(pl.col(k) == v for k, v in match.items())
    return sist.with_columns(
        pl.when(hit).then(0.0).otherwise(pl.col("cmo")).alias("cmo")
    )


class TestComparisonInputs:
    def test_no_source_block_operation_is_no_comparison(
        self, decomp_mini_case: DecompCase, tmp_path: Path
    ) -> None:
        assert (
            compare_operating_points(
                tmp_path, decomp_mini_case, tmp_path, pl.DataFrame(), {}
            )
            is None
        )

    def test_without_novomodelo_planes_only_the_fpha_frames_are_empty(
        self,
        decomp_mini_case: DecompCase,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runs = dataclasses.replace(
            _camargos_runs(decomp_mini_case), novomodelo_planes=None
        )

        result = _compare(monkeypatch, tmp_path, decomp_mini_case, runs)

        assert result is not None
        for frame in (result.split, result.fit_inputs, result.slices, result.metrics):
            assert frame.is_empty()
        assert not result.tie_share.is_empty()
        assert not result.masked_hydro.is_empty()


class TestSourceModelConventions:
    """Both surfaces and the exact function at the printed operating points."""

    def test_exact_function_reproduces_the_printed_fph(
        self,
        decomp_mini_case: DecompCase,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        result = _compare(
            monkeypatch, tmp_path, decomp_mini_case, _camargos_runs(decomp_mini_case)
        )

        assert result is not None
        (check,) = result.fit_inputs["exact_check_max_mw"].to_list()
        # The printed FPH carries two decimals.
        assert check < 0.02
        printed = read_dec_desvfpha(_EXCERPT).filter(
            pl.col("codigo_usina") == _CAMARGOS
        )
        hours = pl.Series([_HOURS[p - 1] for p in printed["patamar"]])
        (row,) = result.split.iter_rows(named=True)
        assert row["rho_exact_at_source"] == pytest.approx(
            (hours * printed["geracao_hidraulica_fph"]).sum()
            / (hours * printed["vazao_turbinada_m3s"]).sum(),
            rel=1e-3,
        )

    def test_source_planes_reproduce_the_printed_fpha(
        self,
        decomp_mini_case: DecompCase,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        result = _compare(
            monkeypatch, tmp_path, decomp_mini_case, _camargos_runs(decomp_mini_case)
        )

        assert result is not None
        (row,) = result.split.iter_rows(named=True)
        assert row["rho_source_planes_at_source"] == pytest.approx(
            row["rho_source"], rel=1e-3
        )

    def test_head_is_taken_at_the_downstream_plant_forebay(
        self,
        decomp_mini_case: DecompCase,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        result = _compare(
            monkeypatch, tmp_path, decomp_mini_case, _camargos_runs(decomp_mini_case)
        )

        assert result is not None
        (row,) = result.fit_inputs.iter_rows(named=True)
        printed = read_dec_desvfpha(_EXCERPT)
        assert row["source_downstream_level_m"] == pytest.approx(
            printed.filter(pl.col("codigo_usina") == _ITUTINGA)["altura_montante"][0],
            abs=0.01,
        )
        assert row["source_forebay_m"] == pytest.approx(
            printed.filter(pl.col("codigo_usina") == _CAMARGOS)["altura_montante"][0],
            abs=0.01,
        )

    def test_lateral_flow_and_gauge_share_reach_the_tailrace(
        self,
        decomp_mini_case: DecompCase,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Belo Monte's ``VU`` routes Pimental's turbined and spilled flow into
        its tailrace and its ``VA`` adds a share of Pimental's gauge; the
        planes see only the ``VU`` part."""
        case = decomp_mini_case
        pimental, belo_monte = (
            case.id_map.hydro_id(c) for c in (_PIMENTAL, _BELO_MONTE)
        )
        q_pimental, s_pimental, q_belo_monte, incremental = (
            200.0,
            3000.0,
            10000.0,
            500.0,
        )
        blocks = [
            _source_block(
                code,
                patamar=p,
                v_useful=0.0,
                q=q,
                s=s,
                gh=gh,
                submarket="N",
            )
            for p in (1, 2, 3)
            for code, q, s, gh in (
                (_PIMENTAL, q_pimental, s_pimental, 10.0),
                (_BELO_MONTE, q_belo_monte, 0.0, 9000.0),
            )
        ]
        stages = [
            _source_stage(_PIMENTAL, incremental=incremental),
            _source_stage(_BELO_MONTE),
        ]
        novomodelo = [
            _novomodelo_block(h, block=b, storage=(2276.29, 2276.29), q=q, gh=gh)
            for b in range(3)
            for h, q, gh in ((pimental, 200.0, 10.0), (belo_monte, 9000.0, 8000.0))
        ]
        runs = _Runs(
            usih=pl.DataFrame(blocks + stages),
            sist=_sist(submarket="N"),
            novomodelo=pl.DataFrame(novomodelo),
            prices=_prices(),
            novomodelo_planes=_novomodelo_planes(belo_monte),
            hydro_ids=(belo_monte, pimental),
        )
        lateral_coefficient = 0.01
        monkeypatch.setattr(
            f"{_MODULE}.read_avl_cortesfpha",
            lambda _d: pl.DataFrame(
                {
                    "codigo_usina": [_BELO_MONTE],
                    "estagio": [1],
                    "fator_correcao": [1.0],
                    "rhs": [0.0],
                    "coeficiente_volume_util": [0.0],
                    "coeficiente_vazao_turbinada": [0.9],
                    "coeficiente_vazao_vertida": [0.0],
                    "coeficiente_vazao_lateral": [lateral_coefficient],
                }
            ),
        )

        result = _compare(monkeypatch, tmp_path, case, runs)

        assert result is not None
        va = case.dadger.va(df=True)
        (factor,) = va.loc[
            va["codigo_usina_influenciada"] == _BELO_MONTE, "fator_impacto_incremental"
        ]
        (row,) = result.fit_inputs.iter_rows(named=True)
        assert row["source_lateral_m3s"] == pytest.approx(
            q_pimental + s_pimental + factor * incremental
        )
        (split,) = result.split.iter_rows(named=True)
        assert split["rho_source_planes_at_source"] == pytest.approx(
            0.9 + lateral_coefficient * (q_pimental + s_pimental) / q_belo_monte
        )


class TestFitInputsAndSlices:
    def test_fit_inputs_report_both_windows_and_limits(
        self,
        decomp_mini_case: DecompCase,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        result = _compare(
            monkeypatch, tmp_path, decomp_mini_case, _camargos_runs(decomp_mini_case)
        )

        assert result is not None
        (row,) = result.fit_inputs.iter_rows(named=True)
        assert (row["source_v_min_hm3"], row["source_v_max_hm3"]) == (631.6, 792.0)
        assert row["source_q_max_m3s"] == 217.2
        assert row["source_grid"] == "5 × 5"
        assert (row["novomodelo_v_min_hm3"], row["novomodelo_v_max_hm3"]) == (
            600.0,
            792.0,
        )
        assert row["novomodelo_spill_fit_max_m3s"] == 50.0
        assert row["novomodelo_q_max_m3s"] == 220.0
        assert row["novomodelo_q_stage_max_m3s"] == 180.0
        assert row["n_planes_source"] == read_avl_cortesfpha(_EXCERPT).height
        assert row["n_planes_novomodelo"] == 2
        assert row["novomodelo_v_hm3"] == pytest.approx(695.0)
        assert row["novomodelo_q_m3s"] == pytest.approx(150.0)

    def test_fit_inputs_report_the_configured_turbined_fit_range(
        self,
        decomp_mini_case: DecompCase,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        result = _compare(
            monkeypatch,
            tmp_path,
            decomp_mini_case,
            _camargos_runs(decomp_mini_case),
            fit_config={"turbined_fit_max_m3s": [205.0]},
        )

        assert result is not None
        (row,) = result.fit_inputs.iter_rows(named=True)
        assert row["novomodelo_q_max_m3s"] == 205.0
        assert row["novomodelo_q_stage_max_m3s"] == 180.0

    def test_slices_sweep_turbined_flow_up_to_the_widest_limit(
        self,
        decomp_mini_case: DecompCase,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        result = _compare(
            monkeypatch, tmp_path, decomp_mini_case, _camargos_runs(decomp_mini_case)
        )

        assert result is not None
        curves = result.slices.group_by("curve").agg(
            pl.len().alias("n"), pl.col("q_m3s").max().alias("q_top")
        )
        assert set(curves["curve"]) == {
            "source_planes",
            "novomodelo_planes",
            "exact",
            "novomodelo_planes_own",
        }
        assert curves["n"].n_unique() == 1
        assert curves["q_top"].to_list() == [220.0] * 4
        (mean_v,) = result.fit_inputs["source_v_hm3"].to_list()
        top = result.slices.filter(
            (pl.col("curve") == "novomodelo_planes") & (pl.col("q_m3s") == 220.0)
        )["gh_mw"][0]
        assert top == pytest.approx(
            min(
                -10.0 + 0.02 * mean_v + 0.215 * 220.0,
                -8.0 + 0.022 * mean_v + 0.197 * 220.0,
            )
        )


class TestTies:
    def test_a_zero_cmo_block_is_a_tie_on_both_models(
        self,
        decomp_mini_case: DecompCase,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runs = _camargos_runs(decomp_mini_case)
        runs = dataclasses.replace(runs, sist=_with_cmo(runs.sist, patamar=3))

        result = _compare(monkeypatch, tmp_path, decomp_mini_case, runs)

        assert result is not None
        (share,) = result.tie_share.iter_rows(named=True)
        assert share["source_tie_share"] == pytest.approx(_HOURS[2] / sum(_HOURS))
        assert share["novomodelo_tie_share"] == pytest.approx(_HOURS[2] / sum(_HOURS))
        (split,) = result.split.iter_rows(named=True)
        assert split["tie_share"] == pytest.approx(_HOURS[2] / sum(_HOURS))
        # The two untied blocks turbine the same printed flow.
        assert split["q_source_m3s"] == pytest.approx(162.83)
        camargos = decomp_mini_case.id_map.hydro_id(_CAMARGOS)
        masked = result.masked_hydro.filter(
            (pl.col("novomodelo_id") == camargos)
            & (pl.col("variable") == "turbined_m3s")
        )
        assert masked["newave_value"].to_list() == [pytest.approx(162.83)]

    def test_zero_water_value_or_spot_price_is_a_novomodelo_tie(
        self,
        decomp_mini_case: DecompCase,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Camargos's water value is zero in block 0 and every plant's bus
        price is zero in block 1."""
        runs = _camargos_runs(decomp_mini_case)
        camargos = runs.hydro_ids[0]
        novomodelo = runs.novomodelo.with_columns(
            pl.when((pl.col("hydro_id") == camargos) & (pl.col("block_id") == 0))
            .then(0.0)
            .otherwise(pl.col("water_value_per_hm3"))
            .alias("water_value_per_hm3")
        )
        prices = runs.prices.with_columns(
            pl.when(pl.col("block_id") == 1)
            .then(0.0)
            .otherwise(pl.col("spot_price"))
            .alias("spot_price")
        )
        runs = dataclasses.replace(runs, novomodelo=novomodelo, prices=prices)

        result = _compare(monkeypatch, tmp_path, decomp_mini_case, runs)

        assert result is not None
        tied = _HOURS[0] + 2 * _HOURS[1]
        (share,) = result.tie_share.iter_rows(named=True)
        assert share["novomodelo_tie_share"] == pytest.approx(tied / (2 * sum(_HOURS)))
        assert share["source_tie_share"] == pytest.approx(tied / (2 * sum(_HOURS)))

    def test_a_single_source_node_ties_every_novomodelo_scenario(
        self,
        decomp_mini_case: DecompCase,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runs = _camargos_runs(decomp_mini_case)
        second = runs.novomodelo.with_columns(
            pl.lit(1, dtype=pl.Int64).alias("scenario_id")
        )
        runs = dataclasses.replace(
            runs,
            sist=_with_cmo(runs.sist, patamar=1),
            novomodelo=pl.concat([runs.novomodelo, second]),
            prices=_prices(stages={0: (0, 1)}),
        )

        result = _compare(monkeypatch, tmp_path, decomp_mini_case, runs)

        assert result is not None
        (share,) = result.tie_share.iter_rows(named=True)
        assert share["novomodelo_tie_share"] == pytest.approx(_HOURS[0] / sum(_HOURS))

    def test_equal_scenario_counts_pair_in_order(
        self,
        decomp_mini_case: DecompCase,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stage 2 has source scenarios 2 and 3 and Novomodelo scenarios 0 and 1;
        a zero CMO in scenario 3 ties Novomodelo's scenario 1 only."""
        runs = _camargos_runs(decomp_mini_case)
        stage_two = [
            {**r, "estagio": 2, "cenario": cenario}
            for cenario in (2, 3)
            for r in runs.usih.to_dicts()
        ]
        novomodelo_two = [
            {**r, "stage_id": 1, "scenario_id": scenario}
            for scenario in (0, 1)
            for r in runs.novomodelo.to_dicts()
        ]
        runs = dataclasses.replace(
            runs,
            usih=pl.concat([runs.usih, pl.DataFrame(stage_two)]),
            sist=_with_cmo(_sist(stages={1: (1,), 2: (2, 3)}), estagio=2, cenario=3),
            novomodelo=pl.concat([runs.novomodelo, pl.DataFrame(novomodelo_two)]),
            prices=_prices(stages={0: (0,), 1: (0, 1)}),
        )

        result = _compare(monkeypatch, tmp_path, decomp_mini_case, runs)

        assert result is not None
        shares = dict(
            result.tie_share.select("stage", "novomodelo_tie_share").iter_rows()
        )
        assert shares == {0: 0.0, 1: pytest.approx(0.5)}

    def test_a_scenario_count_mismatch_flags_each_model_on_its_own_prices(
        self,
        decomp_mini_case: DecompCase,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        runs = _camargos_runs(decomp_mini_case)
        stage_two = [
            {**r, "estagio": 2, "cenario": cenario}
            for cenario in (2, 3)
            for r in runs.usih.to_dicts()
        ]
        novomodelo_two = [
            {**r, "stage_id": 1, "scenario_id": scenario}
            for scenario in (0, 1, 2)
            for r in runs.novomodelo.to_dicts()
        ]
        runs = dataclasses.replace(
            runs,
            usih=pl.concat([runs.usih, pl.DataFrame(stage_two)]),
            sist=_with_cmo(_sist(stages={1: (1,), 2: (2, 3)}), estagio=2),
            novomodelo=pl.concat([runs.novomodelo, pl.DataFrame(novomodelo_two)]),
            prices=_prices(stages={0: (0,), 1: (0, 1, 2)}),
        )

        with caplog.at_level(logging.INFO, logger=_MODULE):
            result = _compare(monkeypatch, tmp_path, decomp_mini_case, runs)

        assert result is not None
        assert (
            "Stage 2: 2 source-model scenarios against 3 Novomodelo scenarios"
            in caplog.text
        )
        shares = {
            stage: (source, novomodelo)
            for stage, source, novomodelo in result.tie_share.iter_rows()
        }
        assert shares[1] == (pytest.approx(1.0), pytest.approx(0.0))
