"""Performance tab tests for ``comparators.decomp.results``.

Second carve out of the legacy ``test_decomp_results_compare.py`` mega file:
the convergence frame, TIM stages/iterations, the max-stage guard,
the forward/backward split chart's column-robustness guard, and the
Performance tab's ``build_decomp_dataset`` rows. The remaining concern bands
(network, energy balance, costs, hydro/thermal detail, productivity, FPHA,
REE, evaporation, constraints, CLI) stay in the mega file pending their own
carve.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.charts import performance_fwd_bwd_split_chart
from novomodelo_bridge.comparators.decomp.results import (
    _decomp_convergence_frame,
    _decomp_max_stage,
    _decomp_tim_iterations,
    _decomp_tim_stages,
    build_decomp_dataset,
)
from novomodelo_bridge.comparators.report_builder import build_comparison_report
from tests.comparators.conftest import (
    _aligned_fixture,
    _extract_tab_content,
    _patch_aligned_frames,
)


def _relato_convergence_frame() -> pl.DataFrame:
    """Three iterations of the source model's own convergence table.

    Carries ``gap_percentual``/``tempo`` alongside ``iteracao``/``zinf``/
    ``zsup`` so a test can pin that ``_decomp_convergence_frame`` selects
    only the three chart columns and drops the rest (out of scope)."""
    return pl.DataFrame(
        {
            "iteracao": [1, 2, 3],
            "zinf": [100.0, 150.0, 180.0],
            "zsup": [500.0, 300.0, 190.0],
            "gap_percentual": [400.0, 100.0, 5.0],
            "tempo": [12.0, 15.0, 9.0],
        }
    )


def _novomodelo_convergence_fixture() -> pl.DataFrame:
    """Novomodelo's own canonical convergence schema, straight from
    ``read_novomodelo_convergence``'s contract."""
    return pl.DataFrame(
        {
            "iteration": [1, 2, 3],
            "lower_bound": [95.0, 145.0, 178.0],
            "upper_bound_mean": [520.0, 310.0, 192.0],
        },
        schema={
            "iteration": pl.Int64,
            "lower_bound": pl.Float64,
            "upper_bound_mean": pl.Float64,
        },
    )


class TestDecompConvergenceFrame:
    """``_decomp_convergence_frame`` -- renames the source
    model's ``relato.convergencia`` onto the canonical ``iteration``/
    ``lower_bound``/``upper_bound_mean`` schema ``read_novomodelo_convergence``
    emits, so ``convergence_chart`` can read both sides without a
    source-specific branch."""

    _CANONICAL_SCHEMA = {
        "iteration": pl.Int64,
        "lower_bound": pl.Float64,
        "upper_bound_mean": pl.Float64,
    }

    def _patch(self, monkeypatch: pytest.MonkeyPatch, frame: pl.DataFrame) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_convergence",
            lambda *_args, **_kwargs: frame,
        )

    def test_renames_and_casts_onto_the_canonical_schema(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch, _relato_convergence_frame())

        frame = _decomp_convergence_frame(tmp_path)

        assert frame.columns == ["iteration", "lower_bound", "upper_bound_mean"]
        assert frame.schema == self._CANONICAL_SCHEMA
        assert frame["iteration"].to_list() == [1, 2, 3]
        # zinf/zsup (native k$) reconciled to R$ (x1e3) to match novomodelo's bounds.
        assert frame["lower_bound"].to_list() == pytest.approx(
            [100_000.0, 150_000.0, 180_000.0]
        )
        assert frame["upper_bound_mean"].to_list() == pytest.approx(
            [500_000.0, 300_000.0, 190_000.0]
        )

    def test_gap_and_timing_columns_are_not_carried_over(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Out of scope: ``gap_percentual``/``tempo`` are not
        consumed by ``convergence_chart``, so they must not leak through."""
        self._patch(monkeypatch, _relato_convergence_frame())

        frame = _decomp_convergence_frame(tmp_path)

        assert "gap_percentual" not in frame.columns
        assert "tempo" not in frame.columns

    def test_missing_relato_degrades_to_an_empty_canonical_frame(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("no relato.rvN found")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_convergence",
            _boom,
        )

        frame = _decomp_convergence_frame(tmp_path)

        assert frame.is_empty()
        assert frame.schema == self._CANONICAL_SCHEMA

    def test_empty_relato_table_degrades_to_an_empty_canonical_frame(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise ValueError("relato.rv0 has no convergencia table")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_convergence",
            _boom,
        )

        frame = _decomp_convergence_frame(tmp_path)

        assert frame.is_empty()
        assert frame.schema == self._CANONICAL_SCHEMA


class TestBuildDecompDatasetConvergence:
    """The Overview tab's Convergence overlay
    (``nw_convergence``/``novomodelo_convergence``) filled by
    ``build_decomp_dataset``."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_convergence",
            lambda *_args, **_kwargs: _relato_convergence_frame(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_convergence",
            lambda *_args, **_kwargs: _novomodelo_convergence_fixture(),
        )

    def test_nw_convergence_matches_the_source_table(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        nw_conv = dataset.render.nw_convergence
        assert nw_conv.columns == ["iteration", "lower_bound", "upper_bound_mean"]
        assert nw_conv["iteration"].to_list() == [1, 2, 3]
        # native k$ zinf/zsup reconciled to R$ (x1e3).
        assert nw_conv["lower_bound"].to_list() == pytest.approx(
            [100_000.0, 150_000.0, 180_000.0]
        )
        assert nw_conv["upper_bound_mean"].to_list() == pytest.approx(
            [500_000.0, 300_000.0, 190_000.0]
        )

    def test_novomodelo_convergence_matches_the_reader_verbatim(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        novomodelo_conv = dataset.render.novomodelo_convergence
        assert novomodelo_conv.columns == [
            "iteration",
            "lower_bound",
            "upper_bound_mean",
        ]
        assert novomodelo_conv["lower_bound"].to_list() == pytest.approx(
            [95.0, 145.0, 178.0]
        )
        assert novomodelo_conv["upper_bound_mean"].to_list() == pytest.approx(
            [520.0, 310.0, 192.0]
        )

    def test_report_renders_the_convergence_section_and_both_traces(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        assert "Convergence" in html
        assert "NEWAVE ZINF" in html
        assert "Novomodelo Lower" in html
        assert "Plotly.newPlot" in html

    def test_absent_decomp_convergence_yields_an_empty_canonical_frame_no_raise(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A missing relato must not abort the dataset build; the chart
        degrades to a Novomodelo-only overlay."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("no relato.rvN found")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_convergence",
            _boom,
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_convergence",
            lambda *_args, **_kwargs: _novomodelo_convergence_fixture(),
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        nw_conv = dataset.render.nw_convergence
        assert nw_conv.is_empty()
        assert nw_conv.columns == ["iteration", "lower_bound", "upper_bound_mean"]

        html = build_comparison_report(dataset)
        assert "Convergence" in html
        assert "Novomodelo Lower" in html


# ---------------------------------------------------------------------------
# Performance tab timing metadata (Caveat #2 -- no fabricated
# DECOMP forward/backward split).
# ---------------------------------------------------------------------------


def _decomp_tim_frame() -> pl.DataFrame:
    """``read_decomp_tim``'s shape: ``Etapa``/``Tempo`` (timedelta), matching
    the real reduced deck's ``decomp.tim`` (``Leitura de Dados`` 8s,
    ``Convergencia`` 2m10s, ``Impressao`` 3s, ``Tempo Total`` 2m21s)."""
    return pl.DataFrame(
        {
            "Etapa": ["Leitura de Dados", "Convergencia", "Impressao", "Tempo Total"],
            "Tempo": [
                timedelta(seconds=8),
                timedelta(minutes=2, seconds=10),
                timedelta(seconds=3),
                timedelta(minutes=2, seconds=21),
            ],
        }
    )


class TestDecompTimStages:
    """``_decomp_tim_stages``: ``decomp.tim`` ``Etapa`` phase totals mapped
    onto the two keys ``performance_metric_cards`` reads."""

    def test_maps_the_two_etapa_rows_onto_the_shared_card_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_decomp_tim",
            lambda *_args, **_kwargs: _decomp_tim_frame(),
        )

        stages = _decomp_tim_stages(tmp_path)

        assert stages == {"Tempo Total": 141.0, "Calculo da Politica": 130.0}

    def test_missing_convergencia_phase_falls_back_to_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A truncated ``decomp.tim`` missing the ``Convergencia`` row still
        yields both keys -- ``Calculo da Politica`` defaults to ``0.0``
        rather than being omitted (never KeyError downstream)."""
        table = pl.DataFrame(
            {
                "Etapa": ["Leitura de Dados", "Tempo Total"],
                "Tempo": [timedelta(seconds=8), timedelta(minutes=2, seconds=21)],
            }
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_decomp_tim",
            lambda *_args, **_kwargs: table,
        )

        stages = _decomp_tim_stages(tmp_path)

        assert stages == {"Tempo Total": 141.0, "Calculo da Politica": 0.0}

    def test_missing_decomp_tim_degrades_to_an_empty_dict(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("decomp.tim not found")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_decomp_tim", _boom
        )

        assert _decomp_tim_stages(tmp_path) == {}

    def test_empty_decomp_tim_value_error_degrades_to_an_empty_dict(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise ValueError("decomp.tim parsed empty")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_decomp_tim", _boom
        )

        assert _decomp_tim_stages(tmp_path) == {}


class TestDecompTimIterations:
    """``_decomp_tim_iterations``: ``relato.convergencia``'s per-iteration
    ``tempo`` renamed onto the canonical ``iteration``/``total_seconds``
    schema -- Caveat #2: never a forward/backward split."""

    _CANONICAL_SCHEMA = {"iteration": pl.Int64, "total_seconds": pl.Float64}

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_convergence",
            lambda *_args, **_kwargs: _relato_convergence_frame(),
        )

    def test_renames_iteracao_and_tempo_onto_the_canonical_schema(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        frame = _decomp_tim_iterations(tmp_path)

        assert frame.columns == ["iteration", "total_seconds"]
        assert frame.schema == self._CANONICAL_SCHEMA
        assert frame["iteration"].to_list() == [1, 2, 3]
        assert frame["total_seconds"].to_list() == pytest.approx([12.0, 15.0, 9.0])

    def test_no_forward_or_backward_columns_are_ever_added(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Caveat #2: DECOMP has no forward/backward pass structure to
        split -- the frame must carry EXACTLY the two canonical columns."""
        self._patch(monkeypatch)

        frame = _decomp_tim_iterations(tmp_path)

        assert "forward_seconds" not in frame.columns
        assert "backward_seconds" not in frame.columns
        assert set(frame.columns) == {"iteration", "total_seconds"}

    def test_missing_relato_degrades_to_an_empty_canonical_frame(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("no relato.rvN found")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_convergence",
            _boom,
        )

        frame = _decomp_tim_iterations(tmp_path)

        assert frame.is_empty()
        assert frame.schema == self._CANONICAL_SCHEMA

    def test_empty_relato_value_error_degrades_to_an_empty_canonical_frame(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise ValueError("relato.rv0 has no convergencia table")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_convergence",
            _boom,
        )

        frame = _decomp_tim_iterations(tmp_path)

        assert frame.is_empty()
        assert frame.schema == self._CANONICAL_SCHEMA


class TestDecompMaxStage:
    """``_decomp_max_stage``: the max already-aligned (0-based) ``stage_id``
    across every level, derived without an extra read."""

    def test_uses_the_max_stage_id_across_every_level(self) -> None:
        aligned = dataclasses.replace(
            _aligned_fixture(),
            source_hydro=pl.DataFrame(
                {"entity_id": [0, 0], "stage_id": [0, 2], "geracao_MW": [1.0, 1.0]}
            ),
            source_thermal=pl.DataFrame(
                {"entity_id": [0, 0], "stage_id": [3, 4], "geracao_MW": [1.0, 1.0]}
            ),
            source_bus=pl.DataFrame({"entity_id": [0], "stage_id": [1], "cmo": [1.0]}),
        )

        assert _decomp_max_stage(aligned) == 4

    def test_returns_none_when_every_level_is_empty(self) -> None:
        aligned = dataclasses.replace(
            _aligned_fixture(),
            source_hydro=pl.DataFrame(),
            source_thermal=pl.DataFrame(),
            source_bus=pl.DataFrame(),
        )

        assert _decomp_max_stage(aligned) is None


class TestPerformanceFwdBwdSplitChartColumnGuard:
    """charts.py fix (approved scope expansion): ``has_nw`` must require the
    ``forward_seconds``/``backward_seconds`` columns, not just a non-empty
    frame -- a DECOMP-shaped ``iteration``/``total_seconds``-only frame must
    render Novomodelo-only instead of raising ``ColumnNotFoundError``."""

    _NOVOMODELO_TIMING = pl.DataFrame(
        {
            "iteration": [1, 2],
            "time_forward_ms": [1000.0, 900.0],
            "time_backward_ms": [2000.0, 1800.0],
            "time_total_ms": [3000.0, 2700.0],
        }
    )

    def test_decomp_shaped_frame_renders_novomodelo_only_without_raising(self) -> None:
        decomp_shaped = pl.DataFrame(
            {"iteration": [1, 2], "total_seconds": [12.0, 15.0]}
        )

        html = performance_fwd_bwd_split_chart(decomp_shaped, self._NOVOMODELO_TIMING)

        assert "NEWAVE (s)" not in html
        assert "Novomodelo (s)" in html

    def test_newave_shaped_frame_still_renders_its_own_panel(self) -> None:
        """Parity guard: the source model's own ``newave.tim`` frame (all
        four columns) is unaffected by the DECOMP-motivated guard change."""
        newave_shaped = pl.DataFrame(
            {
                "iteration": [1, 2],
                "backward_seconds": [20.0, 18.0],
                "forward_seconds": [10.0, 9.0],
                "total_seconds": [30.0, 27.0],
            }
        )

        html = performance_fwd_bwd_split_chart(newave_shaped, self._NOVOMODELO_TIMING)

        assert "NEWAVE (s)" in html
        assert "Novomodelo (s)" in html

    def test_columnless_empty_frame_still_renders_novomodelo_only(self) -> None:
        """The prior default (an entirely empty frame) must keep
        working exactly as before -- the guard's ``is_empty()`` half."""
        html = performance_fwd_bwd_split_chart(pl.DataFrame(), self._NOVOMODELO_TIMING)

        assert "NEWAVE (s)" not in html
        assert "Novomodelo (s)" in html


class TestBuildDecompDatasetPerformance:
    """The Performance tab's timing metadata
    (``nw_tim_stages``/``nw_tim_iterations``/``nw_max_stage``/
    ``novomodelo_training_seconds``/``novomodelo_iteration_timing``) filled by
    ``build_decomp_dataset``."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_decomp_tim",
            lambda *_args, **_kwargs: _decomp_tim_frame(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_convergence",
            lambda *_args, **_kwargs: _relato_convergence_frame(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_training_duration",
            lambda *_args, **_kwargs: 26.0,
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_iteration_timing",
            lambda *_args, **_kwargs: pl.DataFrame(
                {
                    "iteration": [1, 2, 3],
                    "time_forward_ms": [4000.0, 3500.0, 3000.0],
                    "time_backward_ms": [6000.0, 5500.0, 5000.0],
                    "time_total_ms": [10000.0, 9000.0, 8000.0],
                }
            ),
        )

    def test_nw_tim_stages_carries_the_two_mapped_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.render.nw_tim_stages == {
            "Tempo Total": 141.0,
            "Calculo da Politica": 130.0,
        }

    def test_nw_tim_iterations_has_exactly_iteration_and_total_seconds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        nw_tim_iterations = dataset.render.nw_tim_iterations
        assert nw_tim_iterations.columns == ["iteration", "total_seconds"]
        assert nw_tim_iterations["total_seconds"].to_list() == pytest.approx(
            [12.0, 15.0, 9.0]
        )

    def test_novomodelo_training_seconds_and_iteration_timing_are_verbatim(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.render.novomodelo_training_seconds == 26.0
        novomodelo_iter_timing = dataset.render.novomodelo_iteration_timing
        assert novomodelo_iter_timing["time_total_ms"].to_list() == pytest.approx(
            [10000.0, 9000.0, 8000.0]
        )

    def test_nw_max_stage_is_derived_from_the_aligned_frames(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        # ``_aligned_fixture``'s every level is single-stage, stage_id 0.
        assert dataset.render.nw_max_stage == 0

    def test_report_renders_metric_cards_and_time_per_iteration_section(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        assert "Time per Iteration" in html
        assert "NEWAVE Total Wall-Clock" in html
        assert "Plotly.newPlot" in html

    def test_forward_backward_split_is_novomodelo_only_no_decomp_trace_no_crash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The crash-guard proof: a NON-EMPTY DECOMP ``nw_tim_iterations``
        (``iteration``/``total_seconds`` only, per Caveat #2) must not raise
        ``ColumnNotFoundError`` inside ``build_comparison_report`` -- the
        "Forward / Backward Split" section renders with a Novomodelo trace and no
        DECOMP forward/backward trace."""
        self._patch(monkeypatch)
        dataset = build_decomp_dataset(tmp_path, tmp_path)
        assert not dataset.render.nw_tim_iterations.is_empty()

        html = build_comparison_report(dataset)  # must not raise

        content = _extract_tab_content(html, "tab-performance")
        assert "Forward / Backward Split" in content
        assert "Novomodelo (s)" in content
        assert "NEWAVE (s)" not in content

    def test_missing_decomp_tim_and_relato_degrades_to_empty_zero_no_raise(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        def _boom_tim(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("decomp.tim not found")

        def _boom_conv(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("no relato.rvN found")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_decomp_tim", _boom_tim
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_relato_convergence",
            _boom_conv,
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)  # must not raise

        assert dataset.render.nw_tim_stages == {}
        assert dataset.render.nw_tim_iterations.is_empty()

        html = build_comparison_report(dataset)  # must not raise either
        assert "Plotly.newPlot" in html
