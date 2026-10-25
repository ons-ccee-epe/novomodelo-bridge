"""Tests for ``comparators.newave.operating_points``.

The case is the committed NEWAVE mini-deck. Both models' per-block operation,
the source model's planes and fit windows, and Novomodelo's planes are synthetic.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.newave.alignment import EntityAlignment, HydroEntity
from novomodelo_bridge.comparators.newave.operating_points import (
    compare_operating_points,
)
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.converters.hydro import read_cadastro
from novomodelo_bridge.newave.converters.hydro.reference_volumes import (
    initial_storage_hm3,
)
from novomodelo_bridge.newave.converters.temporal import block_fractions
from novomodelo_bridge.newave.filling import month_hours

_DECK = Path(__file__).resolve().parents[1] / "decks" / "newave_mini"
_MODULE = "novomodelo_bridge.comparators.newave.operating_points"
_FUNIL_GRANDE, _FURNAS, _M_DE_MORAES = 4, 6, 7
_NOVOMODELO_ID = {_FUNIL_GRANDE: 10, _FURNAS: 11, _M_DE_MORAES: 12}
_FINAL_USEFUL = {_FURNAS: (12000.0, 11000.0)}
_STAGES = (0, 1)


@pytest.fixture(scope="module")
def case() -> NewaveCase:
    return NewaveCase.from_directory(_DECK)


@dataclasses.dataclass
class _Runs:
    blocks: pl.DataFrame
    storage: pl.DataFrame
    cmo: pl.DataFrame
    novomodelo_planes: pl.DataFrame | None


def _blocks(
    case: NewaveCase,
    codes: tuple[int, ...],
    *,
    q: float = 500.0,
    gh_mw: float = 300.0,
    exact_mw: dict[tuple[int, int, int], float] | None = None,
) -> pl.DataFrame:
    """Each plant's per-block table, *gh_mw* printed as MWmes."""
    fractions = block_fractions(case)
    return pl.DataFrame(
        [
            {
                "newave_code": code,
                "stage": stage,
                "scenario": 0,
                "block": block,
                "generation_mwmes": gh_mw * fractions[stage][block],
                "turbined_m3s": q,
                "spillage_m3s": 0.0,
                "exact_generation_mwmes": (
                    None
                    if exact_mw is None
                    else exact_mw[(code, stage, block)] * fractions[stage][block]
                ),
            }
            for code in codes
            for stage in _STAGES
            for block in range(3)
        ],
        schema_overrides={"exact_generation_mwmes": pl.Float64},
    )


def _storage() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "newave_code": code,
                "stage": stage,
                "scenario": 0,
                "useful_storage_final_hm3": final[stage],
            }
            for code, final in _FINAL_USEFUL.items()
            for stage in _STAGES
        ]
    )


def _cmo(zero: tuple[int, int] | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "submarket_code": 1,
                "stage": stage,
                "scenario": 0,
                "block": block,
                "cmo": 0.0 if (stage, block) == zero else 150.0,
            }
            for stage in _STAGES
            for block in range(3)
        ]
    )


def _source_planes(case: NewaveCase, codes: tuple[int, ...]) -> pl.DataFrame:
    first = case.horizon.start_month
    return pl.DataFrame(
        [
            {
                "newave_code": code,
                "periodo": first + stage,
                "plane_id": plane,
                "fator_correcao": 1.0,
                "gamma_0": gamma_0,
                "gamma_v": 0.01,
                "gamma_q": gamma_q,
                "gamma_s": -0.01,
                "gamma_lat": 0.0,
            }
            for code in codes
            for stage in _STAGES
            for plane, (gamma_0, gamma_q) in enumerate(((0.0, 0.9), (200.0, 0.4)))
        ]
    )


def _grid(case: NewaveCase, codes: tuple[int, ...]) -> pl.DataFrame:
    first = case.horizon.start_month
    cadastro = read_cadastro(case)
    return pl.DataFrame(
        [
            {
                "newave_code": code,
                "periodo": first + stage,
                "v_min_hm3": float(cadastro.loc[code, "volume_minimo"]),
                "v_max_hm3": float(cadastro.loc[code, "volume_maximo"]),
                "n_v": 5,
                "q_min_m3s": 0.0,
                "q_max_m3s": 1200.0,
                "n_q": 5,
                "gh_min_mw": 0.0,
                "gh_max_mw": 1300.0,
                "tipo": 1,
            }
            for code in codes
            for stage in _STAGES
        ]
    )


def _novomodelo_planes(codes: tuple[int, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "hydro_id": _NOVOMODELO_ID[code],
                "stage_id": stage,
                "gamma_0": gamma_0,
                "gamma_v": 0.01,
                "gamma_q": gamma_q,
                "gamma_s": -0.01,
                "kappa": 1.0,
            }
            for code in codes
            for stage in _STAGES
            for gamma_0, gamma_q in ((0.0, 0.88), (190.0, 0.41))
        ]
    )


def _novomodelo_blocks(case: NewaveCase, codes: tuple[int, ...]) -> pl.DataFrame:
    fractions = block_fractions(case)
    return pl.DataFrame(
        [
            {
                "scenario_id": 0,
                "stage_id": stage,
                "block_id": block,
                "hydro_id": _NOVOMODELO_ID[code],
                "hours": fractions[stage][block]
                * month_hours(
                    case.horizon.start_year, case.horizon.start_month + stage
                ),
                "turbined_m3s": 480.0,
                "spillage_m3s": 0.0,
                "outflow_m3s": 480.0,
                "incremental_inflow_m3s": 0.0,
                "storage_initial_hm3": 15000.0,
                "storage_final_hm3": 14000.0,
                "generation_mw": 290.0,
                "water_value_per_hm3": 50.0,
                "accumulated_productivity_mw_per_m3s": 0.5,
            }
            for code in codes
            for stage in _STAGES
            for block in range(3)
        ]
    )


def _runs(
    case: NewaveCase,
    codes: tuple[int, ...] = (_FURNAS,),
    **blocks: object,
) -> _Runs:
    return _Runs(
        blocks=_blocks(case, codes, **blocks),  # type: ignore[arg-type]
        storage=_storage(),
        cmo=_cmo(),
        novomodelo_planes=_novomodelo_planes(codes),
    )


def _compare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: NewaveCase,
    runs: _Runs,
    codes: tuple[int, ...] = (_FURNAS,),
):
    source = {
        "read_nwlistop_hydro_blocks": runs.blocks,
        "read_nwlistop_hydro_storage": runs.storage,
        "read_nwlistop_block_cmo": runs.cmo,
    }
    for name, value in source.items():
        monkeypatch.setattr(f"{_MODULE}.{name}", lambda *_a, value=value: value)
    monkeypatch.setattr(
        f"{_MODULE}.read_fpha_planes", lambda _d: _source_planes(case, codes)
    )
    monkeypatch.setattr(f"{_MODULE}.read_fpha_grid", lambda _d: _grid(case, codes))
    hydro_ids = tuple(_NOVOMODELO_ID[c] for c in codes)
    novomodelo = {
        "read_novomodelo_fpha_planes": runs.novomodelo_planes,
        "read_novomodelo_hydro_block_operation": _novomodelo_blocks(case, codes),
        "read_novomodelo_bus_block_prices": pl.DataFrame(
            [
                {
                    "scenario_id": 0,
                    "stage_id": stage,
                    "block_id": block,
                    "bus_id": 0,
                    "spot_price": 150.0,
                }
                for stage in _STAGES
                for block in range(3)
            ]
        ),
        "read_novomodelo_hydro_bus_labels": {h: frozenset({0}) for h in hydro_ids},
        "read_novomodelo_fpha_fit_config": pl.DataFrame(),
        "read_novomodelo_hydro_metadata": {
            h: {"max_turbined_m3s": 1100.0} for h in hydro_ids
        },
        "read_novomodelo_hydro_per_stage_bounds": pl.DataFrame(),
    }
    for name, value in novomodelo.items():
        monkeypatch.setattr(
            f"{_MODULE}.novomodelo_readers.{name}", lambda _d, value=value: value
        )
    alignment = EntityAlignment(
        hydros=[HydroEntity(c, _NOVOMODELO_ID[c], f"P{c}") for c in codes]
    )
    return compare_operating_points(case, alignment, tmp_path / "novomodelo")


def _fit_row(fit_inputs: pl.DataFrame, code: int, stage: int) -> dict[str, object]:
    return fit_inputs.filter(
        (pl.col("novomodelo_id") == _NOVOMODELO_ID[code]) & (pl.col("stage") == stage)
    ).row(0, named=True)


class TestComparisonInputs:
    def test_without_novomodelo_planes_nothing_is_compared(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: NewaveCase
    ) -> None:
        runs = dataclasses.replace(_runs(case), novomodelo_planes=None)

        result = _compare(monkeypatch, tmp_path, case, runs)

        assert result.metrics.is_empty()
        assert result.split.is_empty()
        assert result.tie_share.is_empty()

    def test_without_plant_tables_only_the_surfaces_are_compared(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: NewaveCase
    ) -> None:
        runs = dataclasses.replace(_runs(case), blocks=_runs(case).blocks.clear())

        with dx.collect() as collected:
            result = _compare(monkeypatch, tmp_path, case, runs)

        assert not result.metrics.is_empty()
        assert result.split.is_empty()
        assert result.masked_hydro.is_empty()
        [note] = [d for d in collected if d.code == "nwlistop-plant-tables-missing"]
        assert note.severity is dx.Severity.INFO
        assert "nwlistop.dat" in note.summary


class TestSourceModelConventions:
    def test_storage_is_the_stage_mean_from_the_initial_storage(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: NewaveCase
    ) -> None:
        result = _compare(monkeypatch, tmp_path, case, _runs(case))

        hreg = read_cadastro(case).loc[_FURNAS]
        v_min = float(hreg["volume_minimo"])
        initial = initial_storage_hm3(hreg, 66.68) - v_min
        first, second = _FINAL_USEFUL[_FURNAS]
        assert _fit_row(result.fit_inputs, _FURNAS, 0)["source_v_hm3"] == (
            pytest.approx(v_min + (initial + first) / 2.0)
        )
        assert _fit_row(result.fit_inputs, _FURNAS, 1)["source_v_hm3"] == (
            pytest.approx(v_min + (first + second) / 2.0)
        )

    def test_a_daily_plant_is_held_at_its_reference_volume(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: NewaveCase
    ) -> None:
        codes = (_FURNAS, _FUNIL_GRANDE)

        result = _compare(monkeypatch, tmp_path, case, _runs(case, codes), codes)

        reference = float(read_cadastro(case).loc[_FUNIL_GRANDE, "volume_referencia"])
        assert _fit_row(result.fit_inputs, _FUNIL_GRANDE, 0)["source_v_hm3"] == (
            pytest.approx(reference)
        )

    def test_a_reservoir_without_storage_is_left_out(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        case: NewaveCase,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        codes = (_FURNAS, _M_DE_MORAES)

        with caplog.at_level(logging.INFO, logger=_MODULE):
            result = _compare(monkeypatch, tmp_path, case, _runs(case, codes), codes)

        assert set(result.split["novomodelo_id"]) == {_NOVOMODELO_ID[_FURNAS]}
        assert f"[{_M_DE_MORAES}]" in caplog.text

    def test_generation_is_read_in_mw_from_each_block_share_of_the_month(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: NewaveCase
    ) -> None:
        result = _compare(monkeypatch, tmp_path, case, _runs(case, gh_mw=420.0))

        assert _fit_row(result.fit_inputs, _FURNAS, 0)["source_gh_mw"] == (
            pytest.approx(420.0)
        )

    def test_exact_check_is_the_largest_gap_to_the_printed_exact_production(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: NewaveCase
    ) -> None:
        unprinted = _compare(monkeypatch, tmp_path, case, _runs(case))
        exact = {
            stage: _fit_row(unprinted.fit_inputs, _FURNAS, stage)["source_rho_exact"]
            * 500.0
            for stage in _STAGES
        }
        offset = {0: 0.1, 1: 0.5, 2: 0.2}
        printed = {
            (_FURNAS, stage, block): exact[stage] + offset[block]
            for stage in _STAGES
            for block in range(3)
        }

        result = _compare(monkeypatch, tmp_path, case, _runs(case, exact_mw=printed))

        assert _fit_row(unprinted.fit_inputs, _FURNAS, 0)["exact_check_max_mw"] is None
        assert _fit_row(result.fit_inputs, _FURNAS, 0)["exact_check_max_mw"] == (
            pytest.approx(0.5)
        )


class TestTies:
    def test_a_zero_cmo_block_is_a_tie_on_both_models(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: NewaveCase
    ) -> None:
        runs = dataclasses.replace(_runs(case), cmo=_cmo(zero=(0, 0)))

        result = _compare(monkeypatch, tmp_path, case, runs)

        share = result.tie_share.filter(pl.col("stage") == 0).row(0, named=True)
        assert share["source_tie_share"] == pytest.approx(block_fractions(case)[0][0])
        assert share["novomodelo_tie_share"] == pytest.approx(
            block_fractions(case)[0][0]
        )
        assert result.tie_share.filter(pl.col("stage") == 1)[
            "source_tie_share"
        ].item() == pytest.approx(0.0)
