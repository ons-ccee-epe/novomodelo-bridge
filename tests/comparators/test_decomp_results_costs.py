"""Cost-frame tests for ``comparators.decomp.results``.

Second carve out of the legacy ``test_decomp_results_compare.py`` mega file:
the DECOMP-side NPV/cost frames, scenario probabilities and
probability-weighting, the union of source-model and Novomodelo cost rows, and the
Overview tab's cost sections in ``build_decomp_dataset``. The remaining
concern bands (network, energy balance, performance, hydro/thermal detail,
productivity, FPHA, REE, evaporation, constraints, CLI) stay in the mega file
pending their own carve.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.charts import _COST_MAP
from novomodelo_bridge.comparators.decomp.results import (
    _DEVIATION_VIOLATION_LABEL,
    _NW_COST_LABELS,
    _bus_side,
    _cost_frames,
    _probability_weighted_stage_cost,
    _scenario_probabilities,
    _union_cost_rows,
    _union_relato_reports,
    _weighted_group_mean,
    build_decomp_dataset,
)
from novomodelo_bridge.comparators.report_builder import build_comparison_report
from novomodelo_bridge.decomp.id_map import DecompIdMap
from tests.comparators.conftest import _balance_fixture, _patch_aligned_frames


def _relato_costs_frame() -> pl.DataFrame:
    """Two stages, two scenarios each -- known k$ values that pin the exact
    NPV dict and per-stage ``nw_sin`` magnitudes ``_cost_frames`` must produce.

    Per-stage means (across ``cenario``): stage 1 -> geracao_termica=120,
    custo_presente=600, custo_futuro=2200; stage 2 -> geracao_termica=100,
    custo_presente=400, custo_futuro=1200. NPV (summed across stage, x1e3):
    GERACAO TERMICA=220_000, INTERCAMBIO=8_000, VERTIMENTO=12_000,
    VIOL. TURB. MINIMO=2_000, VIOL. TURB. MAXIMO=4_000,
    VIOLACAO DESVIO=16_000.
    """
    return pl.DataFrame(
        {
            "estagio": [1, 1, 2, 2],
            "cenario": [1, 2, 1, 2],
            "probabilidade": [0.5, 0.5, 0.5, 0.5],
            "custo_presente": [500.0, 700.0, 300.0, 500.0],
            "custo_futuro": [2000.0, 2400.0, 1000.0, 1400.0],
            "geracao_termica": [100.0, 140.0, 80.0, 120.0],
            "violacao_desvio": [10.0, 10.0, 6.0, 6.0],
            "penalidade_vertimento_reservatorio": [5.0, 5.0, 3.0, 3.0],
            "penalidade_vertimento_fio": [3.0, 3.0, 1.0, 1.0],
            "violacao_turbinamento_reservatorio": [1.0, 1.0, 1.0, 1.0],
            "violacao_turbinamento_fio": [2.0, 2.0, 2.0, 2.0],
            "penalidade_intercambio": [4.0, 4.0, 4.0, 4.0],
        }
    )


def _relato2_costs_frame() -> pl.DataFrame:
    """A scenario-fan (monthly) stage 3 with two **unequal-probability**
    openings, so its expected cost is genuinely probability-weighted rather
    than a 50/50 mean. stage 3 geracao_termica expectation:
    0.6*1000 + 0.4*500 = 800 k$ (weighted) vs 750 k$ (unweighted) -- the tests
    pin the weighted value."""
    return pl.DataFrame(
        {
            "estagio": [3, 3],
            "cenario": [1, 2],
            "probabilidade": [0.6, 0.4],
            "custo_presente": [900.0, 400.0],
            "custo_futuro": [0.0, 0.0],
            "geracao_termica": [1000.0, 500.0],
            "violacao_desvio": [0.0, 0.0],
            "penalidade_vertimento_reservatorio": [0.0, 0.0],
            "penalidade_vertimento_fio": [0.0, 0.0],
            "violacao_turbinamento_reservatorio": [0.0, 0.0],
            "violacao_turbinamento_fio": [0.0, 0.0],
            "penalidade_intercambio": [0.0, 0.0],
        }
    )


class TestCostFrames:
    """``_cost_frames`` -- the DECOMP-side NPV dict (R$) + the
    per-stage ``nw_sin`` cost rows (10^6 R$), reconciled from native k$."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_costs",
            lambda *_args, **_kwargs: _relato_costs_frame(),
        )
        # No scenario-fan stage by default -- relato2 is optional.
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato2_costs",
            lambda *_args, **_kwargs: pl.DataFrame(),
        )

    def test_kdollars_to_reais_reconciliation_on_both_unit_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The TOP RISK guard, pinned in one place: a known k$ input
        must land at the correct magnitude on BOTH the ``nw_costs`` dict (R$,
        x1e3) and the ``nw_sin`` CTERM per-stage rows (10^6 R$, /1e3).
        ``geracao_termica`` aggregates to 220.0 k$ NPV (120 + 100 stage
        means), so the dict must read 220_000.0 R$; each stage's CTERM row
        must read that stage's mean k$ /1e3."""
        self._patch(monkeypatch)

        nw_costs, nw_sin = _cost_frames(tmp_path)

        assert nw_costs["GERACAO TERMICA"] == pytest.approx(220_000.0)
        cterm = {
            row["stage"]: row["value"]
            for row in nw_sin.filter(pl.col("variable") == "CTERM").iter_rows(
                named=True
            )
        }
        assert cterm == pytest.approx({1: 0.12, 2: 0.1})

    def test_relato2_scenario_fan_stage_is_unioned_in_and_probability_weighted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """relato2's monthly scenario-fan stage is unioned onto relato's weekly
        stages, and its expected cost uses the real (unequal) tree
        probabilities -- not a 50/50 mean."""
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_costs",
            lambda *_args, **_kwargs: _relato_costs_frame(),  # weekly stages 1, 2
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato2_costs",
            lambda *_args, **_kwargs: _relato2_costs_frame(),  # fan stage 3
        )

        nw_costs, nw_sin = _cost_frames(tmp_path)

        cterm = {
            row["stage"]: row["value"]
            for row in nw_sin.filter(pl.col("variable") == "CTERM").iter_rows(
                named=True
            )
        }
        # The fan stage (3) is now present -- it was absent when only relato
        # (weeks 1-2) was read ...
        assert set(cterm) == {1, 2, 3}
        # ... and probability-weighted: 0.6*1000 + 0.4*500 = 800 k$ -> 0.8
        # (10^6 R$), NOT the unweighted 750 -> 0.75.
        assert cterm[3] == pytest.approx(0.8)
        # NPV thermal now includes the fan stage: (120 + 100 + 800) k$ x1e3.
        assert nw_costs["GERACAO TERMICA"] == pytest.approx(1_020_000.0)

    def test_absent_relato2_leaves_only_relato_stages(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A deck with no relato2 (empty frame) contributes no extra stage --
        the cost frames cover exactly relato's weeks."""
        self._patch(monkeypatch)  # read_relato2_costs stubbed to empty

        _nw_costs, nw_sin = _cost_frames(tmp_path)

        stages = set(nw_sin.filter(pl.col("variable") == "CTERM")["stage"].to_list())
        assert stages == {1, 2}

    def test_all_cost_map_categories_are_populated_with_known_magnitudes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        nw_costs, _nw_sin = _cost_frames(tmp_path)

        assert nw_costs["INTERCAMBIO"] == pytest.approx(8_000.0)
        assert nw_costs["VERTIMENTO"] == pytest.approx(12_000.0)
        assert nw_costs["VIOL. TURB. MINIMO"] == pytest.approx(2_000.0)
        assert nw_costs["VIOL. TURB. MAXIMO"] == pytest.approx(4_000.0)

    def test_violacao_desvio_surfaces_as_a_descriptive_residual_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``violacao_desvio`` has no ``charts._COST_MAP`` slot -- it must
        still surface (not be silently dropped) under its own key."""
        self._patch(monkeypatch)

        nw_costs, _nw_sin = _cost_frames(tmp_path)

        assert nw_costs[_DEVIATION_VIOLATION_LABEL] == pytest.approx(16_000.0)

    def test_coper_and_custo_futuro_rows_match_the_stage_means(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        _nw_costs, nw_sin = _cost_frames(tmp_path)

        coper = {
            row["stage"]: row["value"]
            for row in nw_sin.filter(pl.col("variable") == "COPER").iter_rows(
                named=True
            )
        }
        custo_futuro = {
            row["stage"]: row["value"]
            for row in nw_sin.filter(pl.col("variable") == "CUSTO_FUTURO").iter_rows(
                named=True
            )
        }
        assert coper == pytest.approx({1: 0.6, 2: 0.4})
        assert custo_futuro == pytest.approx({1: 2.2, 2: 1.2})

    def test_every_nw_costs_key_maps_to_a_known_cost_map_category(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No dead rows: every emitted key is either a ``charts._COST_MAP``
        the source model label or the one intentional residual
        (``violacao_desvio``)."""
        self._patch(monkeypatch)

        nw_costs, _nw_sin = _cost_frames(tmp_path)

        known_labels = {label for label, *_ in _NW_COST_LABELS}
        mapped_cost_map_keys = {k for _, nw_keys, _, _ in _COST_MAP for k in nw_keys}
        assert nw_costs  # the fixture must exercise real categories
        for key in nw_costs:
            assert key in known_labels
            assert key == _DEVIATION_VIOLATION_LABEL or key in mapped_cost_map_keys

    def test_every_nw_sin_variable_is_a_chart_consumed_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        _nw_costs, nw_sin = _cost_frames(tmp_path)

        assert set(nw_sin["variable"].unique().to_list()) <= {
            "COPER",
            "CUSTO_FUTURO",
            "CTERM",
        }

    def test_stage_column_stays_one_based(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        _nw_costs, nw_sin = _cost_frames(tmp_path)

        assert set(nw_sin["stage"].unique().to_list()) == {1, 2}

    def test_propagates_the_readers_raise_on_missing_relato(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No new swallowing: a missing/empty relato surfaces exactly the
        error ``read_relato_costs`` raises."""

        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("no relato.rvN found")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_costs", _boom
        )

        with pytest.raises(FileNotFoundError):
            _cost_frames(tmp_path)


def _relato_costs_with_overlapping_fan_stage() -> pl.DataFrame:
    """`_relato_costs_frame`'s weekly stages 1-2, plus a deliberately WRONG
    row for the fan stage (3) that `_relato2_costs_frame` also covers --
    pins that relato2 is authoritative for any stage it covers, exactly like
    `_cost_frames`'s own union."""
    overlapping = pl.DataFrame(
        {
            "estagio": [3, 3],
            "cenario": [1, 2],
            "probabilidade": [0.9, 0.1],  # wrong -- must be superseded
            "custo_presente": [1.0, 1.0],
            "custo_futuro": [1.0, 1.0],
            "geracao_termica": [1.0, 1.0],
            "violacao_desvio": [1.0, 1.0],
            "penalidade_vertimento_reservatorio": [1.0, 1.0],
            "penalidade_vertimento_fio": [1.0, 1.0],
            "violacao_turbinamento_reservatorio": [1.0, 1.0],
            "violacao_turbinamento_fio": [1.0, 1.0],
            "penalidade_intercambio": [1.0, 1.0],
        }
    )
    return pl.concat([_relato_costs_frame(), overlapping], how="vertical")


class TestScenarioProbabilities:
    """`_scenario_probabilities` -- the ``(estagio, cenario, probabilidade)``
    lookup `_scenario_mean` consumes, built by unioning `read_relato_costs`
    (deterministic weeks) with `read_relato2_costs` (the real, unequal fan-
    stage opening probabilities), relato2 authoritative for any stage it
    covers."""

    def test_unions_relato_and_relato2(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_costs",
            lambda *_a, **_k: _relato_costs_frame(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato2_costs",
            lambda *_a, **_k: _relato2_costs_frame(),
        )

        result = _scenario_probabilities(tmp_path)

        rows = {
            (int(r["estagio"]), int(r["cenario"])): r["probabilidade"]
            for r in result.iter_rows(named=True)
        }
        assert rows == {
            (1, 1): pytest.approx(0.5),
            (1, 2): pytest.approx(0.5),
            (2, 1): pytest.approx(0.5),
            (2, 2): pytest.approx(0.5),
            (3, 1): pytest.approx(0.6),
            (3, 2): pytest.approx(0.4),
        }

    def test_relato2_is_authoritative_for_stages_it_covers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_costs",
            lambda *_a, **_k: _relato_costs_with_overlapping_fan_stage(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato2_costs",
            lambda *_a, **_k: _relato2_costs_frame(),
        )

        result = _scenario_probabilities(tmp_path)

        stage_3 = {
            int(r["cenario"]): r["probabilidade"]
            for r in result.filter(pl.col("estagio") == 3).iter_rows(named=True)
        }
        # relato2's 0.6/0.4 wins -- NOT relato's (wrong) 0.9/0.1.
        assert stage_3 == {1: pytest.approx(0.6), 2: pytest.approx(0.4)}

    def test_empty_when_neither_report_is_available(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("no relato.rvN found")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_costs", _boom
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato2_costs",
            lambda *_a, **_k: pl.DataFrame(),
        )

        result = _scenario_probabilities(tmp_path)

        assert result.is_empty()

    def test_never_raises_when_relato_is_missing_but_relato2_is_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A deck with only a fan-stage report and no aggregate relato (an
        edge case, but `_scenario_probabilities` must not propagate
        `read_relato_costs`'s raise when relato2 alone already has data)."""

        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("no relato.rvN found")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_costs", _boom
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato2_costs",
            lambda *_a, **_k: _relato2_costs_frame(),
        )

        result = _scenario_probabilities(tmp_path)

        assert set(result["estagio"].unique().to_list()) == {3}


class TestWeightedGroupMeanCostEquivalence:
    """`_probability_weighted_stage_cost`'s weighted branch is
    now `_weighted_group_mean(frame, ["estagio"], present)` -- pin that the
    two calls agree on the exact frame `_cost_frames` builds, and that the
    kernel's zero-weight guard still governs a malformed report."""

    def test_matches_the_probability_weighted_stage_cost_wrapper(self) -> None:
        frame = pl.DataFrame(
            {
                "estagio": [3, 3],
                "probabilidade": [0.6, 0.4],
                "geracao_termica": [1000.0, 500.0],
            }
        )
        via_kernel = _weighted_group_mean(frame, ["estagio"], ["geracao_termica"])
        via_wrapper = _probability_weighted_stage_cost(frame, ["geracao_termica"])

        assert via_kernel.to_dicts() == via_wrapper.to_dicts()
        # Known value from the pre-refactor formula: 0.6*1000 + 0.4*500 = 800.
        assert via_kernel["geracao_termica"].to_list() == [pytest.approx(800.0)]

    def test_zero_probability_sum_falls_back_to_the_plain_mean(self) -> None:
        frame = pl.DataFrame(
            {
                "estagio": [3, 3],
                "probabilidade": [0.0, 0.0],
                "geracao_termica": [1000.0, 500.0],
            }
        )
        out = _probability_weighted_stage_cost(frame, ["geracao_termica"])
        assert out["geracao_termica"].to_list() == [pytest.approx(750.0)]


class TestUnionRelatoReports:
    """`_union_relato_reports` -- the shared union `_cost_frames`
    and `_scenario_probabilities` both delegate to."""

    def test_monthly_empty_returns_raw_unchanged(self) -> None:
        raw = _relato_costs_frame()
        assert _union_relato_reports(raw, pl.DataFrame()) is raw

    def test_raw_empty_returns_monthly_unchanged(self) -> None:
        monthly = _relato2_costs_frame()
        assert _union_relato_reports(pl.DataFrame(), monthly) is monthly

    def test_relato2_drops_the_shared_stage_from_raw(self) -> None:
        combined = _union_relato_reports(
            _relato_costs_with_overlapping_fan_stage(), _relato2_costs_frame()
        )

        stage_3 = combined.filter(pl.col("estagio") == 3)
        # relato2's 0.6/0.4 wins -- NOT relato's (wrong) 0.9/0.1 duplicate rows.
        assert len(stage_3) == 2
        assert sorted(stage_3["probabilidade"].to_list()) == [
            pytest.approx(0.4),
            pytest.approx(0.6),
        ]


def _bus_fan_stage_frame() -> pl.DataFrame:
    """One bus: a deterministic single-scenario stage (1) and a 2-scenario
    fan stage (3) -- already ``_stage_rows``-shaped (patamar-null aggregate
    rows only), matching what `read_dec_oper_sist` returns after that fold."""
    return pl.DataFrame(
        {
            "estagio": [1, 3, 3],
            "no": [1, 1, 2],
            "cenario": [1, 1, 2],
            "patamar": [None, None, None],
            "codigo_submercado": [1, 1, 1],
            "deficit_MW": [0.0, 0.0, 0.0],
            "cmo": [40.0, 100.0, 20.0],
        }
    )


class TestScenarioWeightingIntegration:
    """Integration-style: proves the real pipeline --
    `_scenario_probabilities` built from (mocked) relato/relato2 reports,
    threaded through a real ``_*_side`` fold (`_bus_side`, one of this
    module's physical-variable call sites) -- probability-weights the fan
    stage, and degrades to the pre-existing unweighted mean when no
    probability source is available at all."""

    def _bus_codes(self) -> dict[int, int]:
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
        return {code: id_map.bus_id(code) for code in id_map.bus_codes}

    def test_fan_stage_is_probability_weighted_when_a_source_is_available(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_sist",
            lambda *_a, **_k: _bus_fan_stage_frame(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_costs",
            lambda *_a, **_k: _relato_costs_frame(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato2_costs",
            lambda *_a, **_k: _relato2_costs_frame(),
        )

        probabilities = _scenario_probabilities(tmp_path)
        mapped, _unmapped = _bus_side(
            tmp_path, self._bus_codes(), probabilities=probabilities
        )

        cmo_by_stage = dict(zip(mapped["stage_id"].to_list(), mapped["cmo"].to_list()))
        # Fan stage (0-based stage_id 2): relato2's 0.6/0.4 weighted mean =
        # 0.6*100 + 0.4*20 = 68.0, NOT the unweighted mean(100, 20) = 60.0.
        assert cmo_by_stage[2] == pytest.approx(68.0)
        # Deterministic stage (0-based stage_id 0): unaffected.
        assert cmo_by_stage[0] == pytest.approx(40.0)

    def test_fan_stage_stays_unweighted_when_no_probability_source_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A bare deck (no relato/relato2 at all) -- `_scenario_probabilities`
        degrades to empty, and `_bus_side` must reproduce the exact
        pre-existing unweighted-mean value on the fan stage."""
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_sist",
            lambda *_a, **_k: _bus_fan_stage_frame(),
        )

        probabilities = _scenario_probabilities(tmp_path)
        assert probabilities.is_empty()

        mapped, _unmapped = _bus_side(
            tmp_path, self._bus_codes(), probabilities=probabilities
        )

        cmo_by_stage = dict(zip(mapped["stage_id"].to_list(), mapped["cmo"].to_list()))
        assert cmo_by_stage[2] == pytest.approx(60.0)  # plain mean(100, 20)
        assert cmo_by_stage[0] == pytest.approx(40.0)


class TestUnionCostRows:
    """``_union_cost_rows`` -- additive union onto ``nw_sin``,
    defensive against the dataclass-default columnless ``pl.DataFrame()``."""

    def _typed(self, variable: str, value: float) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "newave_code": [0],
                "stage": [1],
                "variable": [variable],
                "value": [value],
            }
        )

    def test_unions_cost_rows_onto_existing_rows(self) -> None:
        earm = self._typed("EARMF", 7000.0)
        cost = self._typed("CTERM", 0.12)

        combined = _union_cost_rows(earm, cost)

        assert set(combined["variable"].to_list()) == {"EARMF", "CTERM"}

    def test_columnless_nw_sin_default_returns_cost_rows_unchanged(self) -> None:
        cost = self._typed("CTERM", 0.12)

        combined = _union_cost_rows(pl.DataFrame(), cost)

        assert combined is cost

    def test_columnless_cost_rows_returns_nw_sin_unchanged(self) -> None:
        earm = self._typed("EARMF", 7000.0)

        combined = _union_cost_rows(earm, pl.DataFrame())

        assert combined is earm


def _novomodelo_cost_breakdown_fixture() -> dict[str, float]:
    return {"thermal_cost": 200_000.0, "deficit_cost": 1_000.0}


def _novomodelo_stage_costs_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "stage_id": [0, 1],
            "immediate_cost": [600_000.0, 400_000.0],
            "future_cost": [2_200_000.0, 1_200_000.0],
            "thermal_cost": [110_000.0, 95_000.0],
            "anticipated_thermal_cost": [10_000.0, 5_000.0],
            "thermal_cost_total": [120_000.0, 100_000.0],
        }
    )


class TestBuildDecompDatasetCosts:
    """Overview cost metadata (nw_costs/novomodelo_costs/nw_sin cost
    rows/novomodelo_stage_costs) filled by ``build_decomp_dataset``."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_aligned_frames(monkeypatch, _balance_fixture())
        # ``_patch_aligned_frames`` stubs ``_cost_frames`` itself to an
        # empty default (see its own docstring) -- re-point it back to the
        # real function (the module-level name this test file imported,
        # unaffected by that stub) so patching ``read_relato_costs`` below
        # actually takes effect through it.
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results._cost_frames", _cost_frames
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_costs",
            lambda *_args, **_kwargs: _relato_costs_frame(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_cost_breakdown",
            lambda *_args, **_kwargs: _novomodelo_cost_breakdown_fixture(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_stage_costs",
            lambda *_args, **_kwargs: _novomodelo_stage_costs_fixture(),
        )

    def test_nw_sin_retains_earm_ena_rows_alongside_the_new_cost_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Additive union, not overwrite: the EARM/ENA rows must
        survive the cost-row union."""
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        variables = set(dataset.render.nw_sin["variable"].unique().to_list())
        assert {"EARMF", "ENA"} <= variables
        assert {"COPER", "CUSTO_FUTURO", "CTERM"} <= variables

    def test_nw_costs_novomodelo_costs_stage_costs_and_offset_are_populated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.render.nw_costs["GERACAO TERMICA"] == pytest.approx(220_000.0)
        assert dataset.render.novomodelo_costs["thermal_cost"] == pytest.approx(
            200_000.0
        )
        stage_costs = dataset.render.novomodelo_stage_costs
        assert isinstance(stage_costs, pl.DataFrame)
        assert not stage_costs.is_empty()
        assert dataset.render.nw_offset == 1

    def test_overview_cost_sections_render_non_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        assert "Cost Breakdown" in html
        assert "Per-Stage Cost" in html
        assert "No cost data available." not in html
        assert "NEWAVE Thermal Cost" in html
        assert "Novomodelo Thermal Cost" in html
        assert "Plotly.newPlot" in html
