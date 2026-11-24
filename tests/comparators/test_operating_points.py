"""Tests for ``comparators.operating_points``, the track-neutral core both
tracks' operating-point builders compose; each track's conventions are
tested with its builder."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from novomodelo_bridge.comparators.fpha import dense_grid, novomodelo_plane_arrays
from novomodelo_bridge.comparators.operating_points import (
    INFLUENCE_SCHEMA,
    LINK_SCHEMA,
    PLANE_SCHEMA,
    CaseTailraceFlows,
    FitWindow,
    Hydraulics,
    OperatingPointComparison,
    SourceProductionModel,
    StagePhysics,
    case_tailrace_flows,
    evaluate_points,
    head,
    novomodelo_points,
    operating_point_fields,
    source_envelope,
    window_gh_max,
    with_case_lateral,
    with_envelopes,
)
from novomodelo_bridge.core.tailrace import TailraceFamily, TailraceSegment
from novomodelo_bridge.novomodelo import readers as novomodelo_readers

_LINEAR_TAILRACE = (
    TailraceFamily(
        downstream_reference_level_m=math.nan,
        segments=(
            TailraceSegment(
                outflow_max_m3s=math.inf, coefficients=(100.0, 0.01, 0.0, 0.0, 0.0)
            ),
        ),
    ),
)
_PHYSICS = StagePhysics(
    cota=(200.0, 0.0, 0.0, 0.0, 0.0),
    rho_esp=0.009,
    losses=0.0,
    max_generation_mw=math.inf,
)


def _hydraulics(*, spill_in_tailrace: bool) -> Hydraulics:
    return Hydraulics(
        families=_LINEAR_TAILRACE,
        spill_in_tailrace=spill_in_tailrace,
        tipo_perda=0,
        volume_floor_hm3=50.0,
    )


class TestHead:
    def test_spill_raises_the_tailrace_only_where_it_reaches_it(self) -> None:
        def _tailrace(spill_in_tailrace: bool) -> float:
            return head(
                _hydraulics(spill_in_tailrace=spill_in_tailrace),
                _PHYSICS,
                v=60.0,
                q=100.0,
                s=400.0,
                lateral_m3s=0.0,
                gauge_m3s=0.0,
                downstream_level_m=None,
            ).tailrace_m

        assert _tailrace(False) == pytest.approx(101.0)
        assert _tailrace(True) == pytest.approx(105.0)

    def test_generation_is_capped_at_the_maximum(self) -> None:
        capped = StagePhysics(
            cota=_PHYSICS.cota, rho_esp=0.009, losses=0.0, max_generation_mw=50.0
        )

        point = head(
            _hydraulics(spill_in_tailrace=False),
            capped,
            v=60.0,
            q=100.0,
            s=0.0,
            lateral_m3s=0.0,
            gauge_m3s=0.0,
            downstream_level_m=None,
        )

        assert point.net_head_m == pytest.approx(99.0)
        assert point.generation_mw == 50.0


class TestGaugeFlow:
    def test_gauge_flow_follows_each_stage_and_scenario(self) -> None:
        target, plant, factor = 1, 2, 0.07
        incremental = {(0, 0): 500.0, (0, 1): 520.0, (1, 0): 800.0, (1, 1): 760.0}
        points = pl.DataFrame(
            [
                {
                    "code": code,
                    "stage": stage,
                    "scenario": scenario,
                    "block": 0,
                    "v_hm3": 60.0,
                    "q_m3s": 100.0,
                    "s_m3s": 0.0,
                    "incremental_m3s": incremental[(stage, scenario)]
                    if code == plant
                    else 0.0,
                }
                for code in (target, plant)
                for stage, scenario in incremental
            ]
        )
        stages = (0, 1)
        model = SourceProductionModel(
            planes={},
            windows={},
            hydraulics={target: _hydraulics(spill_in_tailrace=False)},
            physics={(target, stage): _PHYSICS for stage in stages},
            downstream={(target, stage): None for stage in stages},
            cota=lambda _code, _stage: _PHYSICS.cota,
            gauge_links=pl.DataFrame(
                [(target, plant, factor)], schema=LINK_SCHEMA, orient="row"
            ),
        )

        evaluated = evaluate_points(points, model)

        assert evaluated.height == len(incremental)
        for row in evaluated.iter_rows(named=True):
            gauge = factor * incremental[(row["stage"], row["scenario"])]
            assert row["gauge_m3s"] == gauge
            assert (
                row["exact_mw"]
                == head(
                    _hydraulics(spill_in_tailrace=False),
                    _PHYSICS,
                    v=60.0,
                    q=100.0,
                    s=0.0,
                    lateral_m3s=0.0,
                    gauge_m3s=gauge,
                    downstream_level_m=None,
                ).generation_mw
            )


class TestNovomodeloIncremental:
    @staticmethod
    def _points(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        method: str | None,
        slack: float | None,
    ) -> pl.DataFrame:
        block = {
            "scenario_id": 0,
            "stage_id": 0,
            "block_id": 0,
            "hydro_id": 1,
            "hours": 24.0,
            "turbined_m3s": 100.0,
            "spillage_m3s": 0.0,
            "outflow_m3s": 100.0,
            "incremental_inflow_m3s": -5.0,
            "inflow_nonnegativity_slack_m3s": slack,
            "storage_initial_hm3": 60.0,
            "storage_final_hm3": 60.0,
            "generation_mw": 80.0,
            "water_value_per_hm3": 1.0,
            "accumulated_productivity_mw_per_m3s": 0.9,
        }
        patches = {
            "read_novomodelo_hydro_block_operation": pl.DataFrame(
                [block],
                schema_overrides={"inflow_nonnegativity_slack_m3s": pl.Float64},
            ),
            "read_novomodelo_inflow_non_negativity_method": method,
            "read_novomodelo_bus_block_prices": pl.DataFrame(
                schema={
                    "scenario_id": pl.Int64,
                    "stage_id": pl.Int64,
                    "block_id": pl.Int64,
                    "bus_id": pl.Int64,
                    "spot_price": pl.Float64,
                }
            ),
            "read_novomodelo_hydro_bus_labels": {1: frozenset({0})},
        }
        for name, value in patches.items():
            monkeypatch.setattr(
                novomodelo_readers, name, lambda _dir, value=value: value
            )
        return novomodelo_points(tmp_path, {1: 10})

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("truncation", 0.0),
            ("truncation_with_penalty", 0.0),
            ("penalty", -5.0),
            ("none", -5.0),
            (None, -5.0),
        ],
    )
    def test_incremental_is_the_one_the_lp_saw(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        method: str | None,
        expected: float,
    ) -> None:
        points = self._points(monkeypatch, tmp_path, method, slack=5.0)

        assert points["incremental_m3s"].to_list() == [expected]

    def test_a_missing_slack_adds_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        points = self._points(monkeypatch, tmp_path, "truncation", slack=None)

        assert points["incremental_m3s"].to_list() == [-5.0]


class TestSourceEnvelope:
    def test_is_the_lowest_plane_with_volume_above_the_floor(self) -> None:
        planes = pl.DataFrame(
            {
                "gamma_0": [0.0, 30.0],
                "gamma_v": [1.0, 0.0],
                "gamma_q": [0.5, 0.2],
                "gamma_s": [0.0, 0.0],
                "gamma_lat": [0.0, 0.0],
                "fator_correcao": [0.9, 0.9],
            },
            schema=PLANE_SCHEMA,
        )

        envelope = source_envelope(
            planes,
            v_useful=np.array([10.0]),
            q=np.array([100.0]),
            s=np.array([0.0]),
            lateral=np.array([0.0]),
        )

        assert envelope.tolist() == pytest.approx([0.9 * 50.0])


class TestWindowGhMax:
    def test_a_missing_window_or_cap_is_unbounded(self) -> None:
        windows = {
            (1, 0): FitWindow(100.0, 200.0, 50.0, "5 × 5", 80.0),
            (1, 1): FitWindow(100.0, 200.0, 50.0, "5 × 5", None),
        }

        assert window_gh_max(windows, 1, 0) == 80.0
        assert window_gh_max(windows, 1, 1) == math.inf
        assert window_gh_max(windows, 2, 0) == math.inf


class TestOperatingPointFields:
    def test_empty_surfaces_are_none_and_point_frames_pass_through(self) -> None:
        fields = operating_point_fields(OperatingPointComparison())

        assert fields["fpha_metrics"] is None
        assert fields["fpha_surface"] is None
        assert fields["fpha_spill"] is None
        assert isinstance(fields["fpha_split"], pl.DataFrame)
        assert isinstance(fields["hydro_masked"], pl.DataFrame)

    def test_no_comparison_fills_nothing(self) -> None:
        assert operating_point_fields(None) == {}


_CASE_FLOWS = {
    (1, 0): (100.0, 20.0, 5.0),
    (1, 1): (110.0, 30.0, 6.0),
    (2, 0): (200.0, 40.0, 7.0),
    (2, 1): (210.0, 50.0, 8.0),
    (3, 0): (300.0, 60.0, 9.0),
    (3, 1): (310.0, 70.0, 10.0),
}


def _case_points(*, incremental: bool = True) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "code": plant,
                "novomodelo_id": plant,
                "stage": 0,
                "scenario": 0,
                "block": block,
                "v_hm3": 60.0,
                "q_m3s": q,
                "s_m3s": s,
                **({"incremental_m3s": z} if incremental else {}),
            }
            for (plant, block), (q, s, z) in _CASE_FLOWS.items()
        ]
    )


def _influences(*rows: tuple[int, int, float]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=INFLUENCE_SCHEMA, orient="row")


def _covering(*plants: int) -> SourceProductionModel:
    return SourceProductionModel(
        planes={},
        windows={},
        hydraulics={p: _hydraulics(spill_in_tailrace=False) for p in plants},
        physics={(p, 0): _PHYSICS for p in plants},
        downstream={(p, 0): None for p in plants},
        cota=lambda _code, _stage: _PHYSICS.cota,
    )


class TestCaseLateral:
    def test_effective_spill_matches_the_declaration(self) -> None:
        target, influencer = 1, 2
        points = _case_points()
        flows = CaseTailraceFlows(
            own_factor={target: 1.2},
            outflow=_influences((target, influencer, 0.5)),
            incremental=_influences((target, target, 0.3)),
        )

        result = with_case_lateral(points, flows)

        assert result.select(points.columns).equals(points)
        for row in result.iter_rows(named=True):
            if row["novomodelo_id"] != target:
                assert row["case_s_eff_m3s"] == row["s_m3s"]
                continue
            q_j, s_j, _ = _CASE_FLOWS[(influencer, row["block"])]
            z_t = _CASE_FLOWS[(target, row["block"])][2]
            assert row["case_s_eff_m3s"] == pytest.approx(
                1.2 * row["s_m3s"] + 0.5 * (q_j + s_j) + 0.3 * z_t, rel=1e-12
            )

    def test_without_declarations_the_column_is_the_own_spill(self) -> None:
        points = _case_points()

        result = with_case_lateral(points, CaseTailraceFlows())

        assert np.array_equal(
            result.get_column("case_s_eff_m3s").to_numpy(),
            points.get_column("s_m3s").to_numpy(),
        )

    def test_an_influencer_the_model_does_not_cover_still_contributes(self) -> None:
        flows = CaseTailraceFlows(
            outflow=_influences((1, 3, 0.4)), incremental=_influences((1, 3, 0.1))
        )

        evaluated = evaluate_points(
            with_case_lateral(_case_points(), flows), _covering(1)
        )

        assert set(evaluated.get_column("code")) == {1}
        for row in evaluated.iter_rows(named=True):
            q_u, s_u, z_u = _CASE_FLOWS[(3, row["block"])]
            assert row["case_s_eff_m3s"] == pytest.approx(
                row["s_m3s"] + 0.4 * (q_u + s_u) + 0.1 * z_u, rel=1e-12
            )

    def test_a_declared_incremental_influence_needs_the_incremental_column(
        self,
    ) -> None:
        points = _case_points(incremental=False)
        flows = CaseTailraceFlows(incremental=_influences((2, 1, 0.07)))

        with pytest.raises(ValueError, match=r"\[2\]"):
            with_case_lateral(points, flows)

    def test_an_incremental_influence_on_an_absent_plant_is_not_refused(self) -> None:
        points = _case_points(incremental=False)
        flows = CaseTailraceFlows(
            outflow=_influences((1, 2, 0.5)), incremental=_influences((9, 1, 0.07))
        )

        result = with_case_lateral(points, flows)

        for row in result.filter(pl.col("novomodelo_id") == 1).iter_rows(named=True):
            q_j, s_j, _ = _CASE_FLOWS[(2, row["block"])]
            assert row["case_s_eff_m3s"] == pytest.approx(
                row["s_m3s"] + 0.5 * (q_j + s_j), rel=1e-12
            )


class TestCaseTailraceFlows:
    @staticmethod
    def _read(monkeypatch: pytest.MonkeyPatch, hydros: list[dict]) -> CaseTailraceFlows:
        monkeypatch.setattr(
            novomodelo_readers, "read_novomodelo_hydros", lambda _dir: hydros
        )
        return case_tailrace_flows(Path("unused"))

    def test_an_omitted_own_factor_is_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flows = self._read(
            monkeypatch,
            [
                {
                    "id": 1,
                    "tailrace_flow": {
                        "own_outflow_factor": 1.2,
                        "outflow_influences": [{"hydro_id": 2, "factor": 0.5}],
                        "incremental_inflow_influences": [
                            {"hydro_id": 1, "factor": 0.3}
                        ],
                    },
                },
                {
                    "id": 2,
                    "tailrace_flow": {
                        "outflow_influences": [{"hydro_id": 1, "factor": 0.1}]
                    },
                },
                {"id": 3},
            ],
        )

        assert flows.own_factor == {1: 1.2, 2: 1.0}
        assert flows.outflow.rows() == [(1, 2, 0.5), (2, 1, 0.1)]
        assert flows.incremental.rows() == [(1, 1, 0.3)]

    def test_a_case_without_declarations_has_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flows = self._read(monkeypatch, [{"id": 1}, {"id": 2}])

        assert flows.own_factor == {}
        assert flows.outflow.schema == INFLUENCE_SCHEMA
        assert flows.outflow.is_empty()
        assert flows.incremental.is_empty()


class TestEnvelopeSpill:
    _SOURCE_PLANES = pl.DataFrame(
        {
            "gamma_0": [0.0],
            "gamma_v": [1.0],
            "gamma_q": [0.5],
            "gamma_s": [0.0],
            "gamma_lat": [0.0],
            "fator_correcao": [0.9],
        },
        schema=PLANE_SCHEMA,
    )
    _NOVOMODELO_PLANES = pl.DataFrame(
        {
            "gamma_0": [10.0, 30.0],
            "gamma_v": [0.1, 0.0],
            "gamma_q": [0.5, 0.3],
            "gamma_s": [-0.02, -0.01],
            "kappa": [1.0, 0.9],
        }
    )

    def _envelopes(self, flows: CaseTailraceFlows) -> pl.DataFrame:
        model = SourceProductionModel(
            planes={(1, 0): self._SOURCE_PLANES},
            windows={},
            hydraulics={1: _hydraulics(spill_in_tailrace=False)},
            physics={(1, 0): _PHYSICS},
            downstream={(1, 0): None},
            cota=lambda _code, _stage: _PHYSICS.cota,
        )
        points = _case_points().filter(pl.col("novomodelo_id") == 1)
        return with_envelopes(
            evaluate_points(with_case_lateral(points, flows), model),
            model,
            {(1, 0): self._NOVOMODELO_PLANES},
        )

    def _planes_at(self, result: pl.DataFrame, spill: str) -> np.ndarray:
        return dense_grid(
            *novomodelo_plane_arrays(self._NOVOMODELO_PLANES),
            result.get_column("v_hm3").to_numpy(),
            result.get_column("q_m3s").to_numpy(),
            result.get_column(spill).to_numpy(),
        )

    def test_empty_flows_evaluate_at_the_own_spill_bit_for_bit(self) -> None:
        result = self._envelopes(CaseTailraceFlows())

        assert np.array_equal(
            result.get_column("novomodelo_planes_mw").to_numpy(),
            self._planes_at(result, "s_m3s"),
        )

    def test_declared_flows_evaluate_at_the_effective_spill(self) -> None:
        result = self._envelopes(CaseTailraceFlows(own_factor={1: 1.5}))

        planes = result.get_column("novomodelo_planes_mw").to_numpy()
        assert np.array_equal(planes, self._planes_at(result, "case_s_eff_m3s"))
        assert not np.array_equal(planes, self._planes_at(result, "s_m3s"))
