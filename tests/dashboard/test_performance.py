"""Unit tests for novomodelo_bridge.dashboard.tabs.performance (the Performance tab)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from novomodelo_bridge.dashboard.tabs.performance import (
    TAB_ID,
    TAB_LABEL,
    TAB_ORDER,
    _build_metrics_row,
    _chart_retry_histogram,
    _format_time,
    can_render,
    render,
)

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_solver_df(
    n_rows: int = 5,
    *,
    phase: str = "backward",
    lp_solves: int = 10,
    solve_time_ms: float = 50.0,
    simplex_iterations: int = 200,
) -> pd.DataFrame:
    """Synthetic solver DataFrame (solver_train or solver_sim).

    Includes all columns required by the Performance tab chart functions so that
    tests using the default ``_make_mock_data()`` do not fail on missing columns.
    """
    return pd.DataFrame(
        {
            "iteration": list(range(1, n_rows + 1)),
            "stage": list(range(0, n_rows)),
            "phase": [phase] * n_rows,
            "opening": [
                float("nan") if phase != "backward" else 0 for _ in range(n_rows)
            ],
            "rank": [0] * n_rows,
            "worker_id": [
                float("nan") if phase != "backward" else 0 for _ in range(n_rows)
            ],
            "lp_solves": [lp_solves] * n_rows,
            "lp_successes": [lp_solves] * n_rows,
            "lp_failures": [0] * n_rows,
            "retry_attempts": [0] * n_rows,
            "solve_time_ms": [solve_time_ms] * n_rows,
            "simplex_iterations": [simplex_iterations] * n_rows,
            "basis_offered": [8] * n_rows,
            "basis_consistency_failures": [2] * n_rows,
            "basis_reconstructions": [0] * n_rows,
            "set_bounds_time_ms": [5.0] * n_rows,
            "basis_set_time_ms": [1.0] * n_rows,
            "load_model_time_ms": [2.0] * n_rows,
        }
    )


def _make_timing(n: int = 5) -> pd.DataFrame:
    """Synthetic timing DataFrame mirroring the novomodelo iteration_timing schema."""
    return pd.DataFrame(
        {
            "iteration": list(range(1, n + 1)),
            "rank": [0] * n,
            "worker_id": [0] * n,
            "forward_wall_ms": [100.0 + i for i in range(n)],
            "backward_wall_ms": [150.0 + i for i in range(n)],
            "cut_selection_ms": [0] * n,
            "mpi_allreduce_ms": [0] * n,
            "cut_sync_ms": [0] * n,
            "lower_bound_ms": [0] * n,
            "state_exchange_ms": [0] * n,
            "cut_batch_build_ms": [0] * n,
            "bwd_setup_ms": [10] * n,
            "bwd_load_imbalance_ms": [0] * n,
            "bwd_scheduling_overhead_ms": [0] * n,
            "fwd_setup_ms": [5] * n,
            "fwd_load_imbalance_ms": [0] * n,
            "fwd_scheduling_overhead_ms": [0] * n,
            "overhead_ms": [10.0 + i for i in range(n)],
        }
    )


def _make_solver_train(n_stages: int = 4, n_iters: int = 3) -> pd.DataFrame:
    """Synthetic solver_train DataFrame matching the current novomodelo schema."""
    rows = []
    for it in range(1, n_iters + 1):
        for stage in range(0, n_stages):
            for phase in ("forward", "backward"):
                rows.append(
                    {
                        "iteration": it,
                        "stage": stage,
                        "phase": phase,
                        "opening": 0 if phase == "backward" else None,
                        "rank": 0,
                        "worker_id": 0 if phase == "backward" else None,
                        "lp_solves": 10,
                        "lp_successes": 10,
                        "lp_failures": 0,
                        "retry_attempts": 0,
                        "solve_time_ms": 50.0 + stage * 2.0,
                        "simplex_iterations": 200 + stage * 5,
                        "basis_offered": 8,
                        "basis_consistency_failures": 2,
                        "basis_reconstructions": 0,
                        "set_bounds_time_ms": 5.0 + stage * 0.5,
                        "basis_set_time_ms": 1.0,
                        "load_model_time_ms": 2.0,
                    }
                )
    return pd.DataFrame(rows)


def _make_retry_histogram(
    retry_levels: list[int] | None = None,
    counts: list[int] | None = None,
) -> pd.DataFrame:
    """Synthetic retry_histogram DataFrame matching actual parquet schema.

    Actual schema: iteration, phase, stage, retry_level, count.
    The chart function aggregates by retry_level, summing count.
    """
    if retry_levels is None:
        retry_levels = [0, 0, 1, 1, 3]
    if counts is None:
        counts = [50, 50, 10, 10, 5]
    n = len(retry_levels)
    return pd.DataFrame(
        {
            "iteration": list(range(n)),
            "phase": ["forward"] * n,
            "stage": [0] * n,
            "retry_level": retry_levels,
            "count": counts,
        }
    )


def _make_mock_data(
    solver_train: pd.DataFrame | None = None,
    solver_sim: pd.DataFrame | None = None,
    timing: pd.DataFrame | None = None,
    metadata: dict | None = None,
    simulation_manifest: dict | None = None,
    scaling_report: dict | None = None,
    retry_histogram: pd.DataFrame | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics DashboardData with real DataFrames.

    Maps legacy parameter names to the attribute names used by the current
    implementation:
      - ``metadata`` (with nested ``run_info``) -> ``training_metadata``
        (flat dict with ``duration_seconds``).
      - ``simulation_manifest`` -> ``simulation_metadata``.
    """
    data = MagicMock()
    data.solver_train = solver_train if solver_train is not None else _make_solver_df()
    data.solver_sim = solver_sim if solver_sim is not None else _make_solver_df()
    data.timing = timing if timing is not None else _make_timing()
    # timing_raw exists in DashboardData for the per-worker tiles. Tests treat
    # the aggregated timing frame as both the summed view and the raw view.
    data.timing_raw = data.timing

    # The implementation reads data.training_metadata.get("duration_seconds").
    # The legacy test helper accepted metadata={"run_info": {"duration_seconds": X}};
    # flatten that to the format the implementation expects.
    if metadata is not None:
        run_info = metadata.get("run_info", {})
        data.training_metadata = run_info
    else:
        data.training_metadata = {}

    # The implementation reads data.simulation_metadata.get("duration_seconds").
    data.simulation_metadata = (
        simulation_manifest if simulation_manifest is not None else {}
    )

    data.scaling_report = scaling_report if scaling_report is not None else {}
    data.retry_histogram = (
        retry_histogram if retry_histogram is not None else pd.DataFrame()
    )
    return data


# ---------------------------------------------------------------------------
# test_tab_constants
# ---------------------------------------------------------------------------


def test_tab_constants() -> None:
    """TAB_ID, TAB_LABEL, and TAB_ORDER must match the spec."""
    assert TAB_ID == "tab-performance"
    assert TAB_LABEL == "Performance"
    assert TAB_ORDER == 90


# ---------------------------------------------------------------------------
# test_can_render_returns_true
# ---------------------------------------------------------------------------


def test_can_render_returns_true() -> None:
    """can_render() must always return True regardless of data."""
    data = _make_mock_data()
    assert can_render(data) is True


# ---------------------------------------------------------------------------
# test_format_time_*
# ---------------------------------------------------------------------------


def test_format_time_seconds() -> None:
    """_format_time(45.0) must return '45.0 s'."""
    assert _format_time(45.0) == "45.0 s"


def test_format_time_minutes() -> None:
    """_format_time(125.0) must return '2.1 min'."""
    assert _format_time(125.0) == "2.1 min"


def test_format_time_hours() -> None:
    """_format_time(7200.0) must return '2.00 h'."""
    assert _format_time(7200.0) == "2.00 h"


def test_format_time_boundary_60s() -> None:
    """_format_time(60.0) must switch to minutes at exactly 60 s."""
    result = _format_time(60.0)
    assert "min" in result


def test_format_time_boundary_3600s() -> None:
    """_format_time(3600.0) must switch to hours at exactly 3600 s."""
    result = _format_time(3600.0)
    assert "h" in result


# ---------------------------------------------------------------------------
# test_build_metrics_row_with_data
# ---------------------------------------------------------------------------


def test_build_metrics_row_with_data() -> None:
    """_build_metrics_row() with synthetic solver data contains expected labels."""
    data = _make_mock_data(
        solver_train=_make_solver_df(n_rows=5, lp_solves=10, solve_time_ms=50.0),
        solver_sim=_make_solver_df(n_rows=5, lp_solves=10, solve_time_ms=50.0),
    )
    html = _build_metrics_row(data)
    assert "Avg LP Solve" in html
    assert "Total LP Solves" in html
    assert "Total Simplex Iterations" in html
    assert "metrics-grid" in html


def test_build_metrics_row_lp_solve_avg_value() -> None:
    """Avg LP Solve (training): solve_time / lp_solves across all rows."""
    # 5 rows * 50 ms total / (5 rows * 10 lp_solves) = 1.00 ms avg
    data = _make_mock_data(
        solver_train=_make_solver_df(n_rows=5, lp_solves=10, solve_time_ms=50.0),
        solver_sim=pd.DataFrame(),
    )
    html = _build_metrics_row(data)
    # total_lp_time=250, total_lp_solves=50, avg=5.00 ms
    assert "5.00 ms" in html


def test_build_metrics_row_total_lp_solves_value() -> None:
    """Total LP Solves must be the sum from both solver_train and solver_sim."""
    train = _make_solver_df(n_rows=3, lp_solves=10)  # 30 total
    sim = _make_solver_df(n_rows=2, lp_solves=5)  # 10 total → 40 combined
    data = _make_mock_data(solver_train=train, solver_sim=sim)
    html = _build_metrics_row(data)
    assert "40" in html


# ---------------------------------------------------------------------------
# test_build_metrics_row_empty_solver
# ---------------------------------------------------------------------------


def test_build_metrics_row_empty_solver() -> None:
    """Empty solver DataFrames must not crash and must show zeros."""
    data = _make_mock_data(
        solver_train=pd.DataFrame(),
        solver_sim=pd.DataFrame(),
    )
    html = _build_metrics_row(data)
    # Both avg LP solve values should be 0.00 ms
    assert "0.00 ms" in html
    # Total LP Solves = 0
    assert ">0<" in html or "0,</div>" in html or ">0</div>" in html


def test_build_metrics_row_empty_solver_no_crash() -> None:
    """Empty solver DataFrames must produce a valid metrics-grid HTML string."""
    data = _make_mock_data(
        solver_train=pd.DataFrame(),
        solver_sim=pd.DataFrame(),
    )
    html = _build_metrics_row(data)
    assert "metrics-grid" in html
    assert len(html) > 0


# ---------------------------------------------------------------------------
# test_build_metrics_row_metadata_training_time
# ---------------------------------------------------------------------------


def test_build_metrics_row_uses_metadata_duration() -> None:
    """metadata duration_seconds takes priority over timing fallback."""
    data = _make_mock_data(
        metadata={"run_info": {"duration_seconds": 3661.0}},
        timing=pd.DataFrame(),  # fallback would give 0 s
    )
    html = _build_metrics_row(data)
    # 3661 s → "1.02 h"
    assert "1.02 h" in html


def test_build_metrics_row_uses_simulation_manifest_duration() -> None:
    """simulation_manifest duration_seconds causes 'Total Simulation Time' label."""
    data = _make_mock_data(
        simulation_manifest={"duration_seconds": 120.0},
        solver_sim=pd.DataFrame(),  # fallback would give 0 s
    )
    html = _build_metrics_row(data)
    assert "Total Simulation Time" in html
    assert "2.0 min" in html


def test_build_metrics_row_fallback_simulation_label() -> None:
    """When simulation_manifest has no duration, label says 'Simulation CPU Time'."""
    data = _make_mock_data(
        simulation_manifest={},
        solver_sim=_make_solver_df(n_rows=2, solve_time_ms=30000.0),
    )
    html = _build_metrics_row(data)
    assert "Simulation CPU Time" in html


# ---------------------------------------------------------------------------
# test_render_empty_timing
# ---------------------------------------------------------------------------


def test_render_empty_timing() -> None:
    """render() with empty timing DataFrame must include 'No timing data'."""
    data = _make_mock_data(timing=pd.DataFrame())
    html = render(data)
    assert "No timing data" in html


# ---------------------------------------------------------------------------
# test_render_with_data
# ---------------------------------------------------------------------------


def test_render_with_data() -> None:
    """render() with all synthetic data must contain required section strings."""
    data = _make_mock_data(
        solver_train=_make_solver_df(n_rows=5),
        solver_sim=_make_solver_df(n_rows=5),
        timing=_make_timing(n=5),
    )
    html = render(data)
    assert "Top-Level Iteration Timing" in html
    assert "Forward / Backward Wall-Time Breakdown" in html
    assert "metrics-grid" in html


def test_render_contains_run_summary_section() -> None:
    """render() must produce a 'Run Summary' section title."""
    data = _make_mock_data()
    html = render(data)
    assert "Run Summary" in html


def test_render_no_exception_full_data() -> None:
    """render() with all fields populated must not raise."""
    data = _make_mock_data(
        solver_train=_make_solver_df(n_rows=10),
        solver_sim=_make_solver_df(n_rows=10),
        timing=_make_timing(n=10),
        metadata={"run_info": {"duration_seconds": 300.0}},
        simulation_manifest={"duration_seconds": 60.0},
    )
    html = render(data)
    assert len(html) > 0


# ---------------------------------------------------------------------------
# test_chart_retry_histogram_*
# ---------------------------------------------------------------------------


def test_chart_retry_histogram_with_data() -> None:
    """_chart_retry_histogram() with non-empty DataFrame returns HTML with 'plotly'."""
    df = _make_retry_histogram()
    html = _chart_retry_histogram(df)
    assert "plotly" in html.lower()


def test_chart_retry_histogram_empty() -> None:
    """_chart_retry_histogram() with empty DataFrame returns fallback message."""
    html = _chart_retry_histogram(pd.DataFrame())
    assert "No retry data" in html


def test_chart_retry_histogram_color_coding() -> None:
    """_chart_retry_histogram() with mixed retry counts (0, 1, 2, 5) does not raise."""
    df = _make_retry_histogram(
        retry_levels=[0, 1, 2, 5],
        counts=[80, 15, 10, 3],
    )
    html = _chart_retry_histogram(df)
    assert "plotly" in html.lower()


def test_render_wraps_retry_histogram_in_chart_card() -> None:
    """The retry histogram must sit inside the standard chart-card wrapper,
    matching its sibling heatmap (both wrap_chart(...)-ed at the call site)."""
    data = _make_mock_data(retry_histogram=_make_retry_histogram())
    html = render(data)

    retry_start = html.index("Solver Retries")
    retry_section = html[retry_start : html.index("Simulation", retry_start)]
    # Before the fix, only the heatmap sibling was wrap_chart(...)-ed, so this
    # section held exactly one "chart-card" div; now both charts are wrapped.
    assert retry_section.count('<div class="chart-card">') == 2


# ---------------------------------------------------------------------------
# test_render_full_sections
# ---------------------------------------------------------------------------


def test_render_full_sections() -> None:
    """render() with all non-empty data must contain all section titles."""
    scaling_report = {
        "stages": [
            {
                "stage_id": 0,
                "dimensions": {"num_cols": 100, "num_rows": 80, "num_nz": 500},
                "pre_scaling": {"matrix_coeff_ratio": 1000.0},
                "post_scaling": {"matrix_coeff_ratio": 10.0},
            }
        ]
    }
    data = _make_mock_data(
        solver_train=_make_solver_train(n_stages=3, n_iters=2),
        solver_sim=_make_solver_df(n_rows=3),
        timing=_make_timing(n=3),
        scaling_report=scaling_report,
        retry_histogram=_make_retry_histogram(),
    )
    html = render(data)
    assert "Solver Time Breakdown by Phase" in html
    assert "Per-Stage LP Solve Heatmaps" in html
    assert "Per-Stage Solver Overhead" in html
    assert "LP Dimensions" in html
    assert "Per-Stage LP Detail" in html
    assert "Solver Retries" in html
    assert "Simulation" in html
    # Additional top-level iteration-timing sections.
    assert "Top-Level Iteration Timing" in html
    assert "Forward / Backward Wall-Time Breakdown" in html
    assert "Parallel Overhead Decomposition" in html
    assert "Solver Progression" in html
    assert "Solver CPU Components per Iteration" in html
    assert "Basis Warm-start" in html
    assert "Run Summary" in html
    # Opening 0 (cold) vs Openings 1+ (warm-start) section
    assert "Opening 0 (cold) vs Openings 1+ (warm-start)" in html


# ---------------------------------------------------------------------------
# Opening 0 (cold) vs Openings 1+ (warm-start) charts
# ---------------------------------------------------------------------------


def _make_solver_train_with_openings(
    n_stages: int = 3,
    n_iters: int = 2,
    n_openings: int = 4,
    cold_multiplier: float = 2.0,
) -> pd.DataFrame:
    """Synthetic solver_train with per-opening backward rows.

    Opening 0 is tagged with ``cold_multiplier × base`` solve_time to emulate
    the cold-start cost; openings 1+ use the base value.
    """
    rows = []
    for it in range(1, n_iters + 1):
        for stage in range(n_stages):
            # Forward row (single, opening=None)
            rows.append(
                {
                    "iteration": it,
                    "stage": stage,
                    "phase": "forward",
                    "opening": None,
                    "rank": 0,
                    "worker_id": None,
                    "lp_solves": 1,
                    "lp_successes": 1,
                    "lp_failures": 0,
                    "retry_attempts": 0,
                    "solve_time_ms": 40.0,
                    "simplex_iterations": 150,
                    "basis_offered": 1,
                    "basis_consistency_failures": 0,
                    "basis_reconstructions": 0,
                    "set_bounds_time_ms": 4.0,
                    "basis_set_time_ms": 1.0,
                    "load_model_time_ms": 2.0,
                }
            )
            # Backward rows, one per opening
            base_solve = 50.0 + stage * 2.0
            base_simplex = 200 + stage * 5
            for opening in range(n_openings):
                is_cold = opening == 0
                rows.append(
                    {
                        "iteration": it,
                        "stage": stage,
                        "phase": "backward",
                        "opening": opening,
                        "rank": 0,
                        "worker_id": 0,
                        "lp_solves": 1,
                        "lp_successes": 1,
                        "lp_failures": 0,
                        "retry_attempts": 0,
                        "solve_time_ms": base_solve
                        * (cold_multiplier if is_cold else 1.0),
                        "simplex_iterations": int(
                            base_simplex * (cold_multiplier if is_cold else 1.0)
                        ),
                        "basis_offered": 1,
                        "basis_consistency_failures": 0,
                        "basis_reconstructions": 0,
                        "set_bounds_time_ms": 5.0,
                        "basis_set_time_ms": 1.0,
                        "load_model_time_ms": 2.0,
                    }
                )
    return pd.DataFrame(rows)


def test_chart_backward_opening_0_solve_time_contains_both_series() -> None:
    """The per-stage opening-0 solve chart includes both cold and warm bars."""
    from novomodelo_bridge.dashboard.tabs.performance_charts import (
        chart_backward_opening_0_solve_time,
    )

    solver_train = _make_solver_train_with_openings(
        n_stages=3, n_iters=2, n_openings=5, cold_multiplier=3.0
    )
    html = chart_backward_opening_0_solve_time(solver_train)
    assert "Opening 0 (cold)" in html
    assert "Openings 1+ (warm, mean)" in html


def test_chart_backward_opening_0_simplex_contains_both_series() -> None:
    """The per-stage opening-0 simplex chart includes both cold and warm bars."""
    from novomodelo_bridge.dashboard.tabs.performance_charts import (
        chart_backward_opening_0_simplex,
    )

    solver_train = _make_solver_train_with_openings(
        n_stages=3, n_iters=2, n_openings=5, cold_multiplier=3.0
    )
    html = chart_backward_opening_0_simplex(solver_train)
    assert "Opening 0 (cold)" in html
    assert "Openings 1+ (warm, mean)" in html


def test_chart_backward_opening_0_share_reports_all_metrics() -> None:
    """The share chart reports LP solves, simplex iters, and solve time shares."""
    from novomodelo_bridge.dashboard.tabs.performance_charts import (
        chart_backward_opening_0_share,
    )

    solver_train = _make_solver_train_with_openings(
        n_stages=2, n_iters=2, n_openings=4, cold_multiplier=2.0
    )
    html = chart_backward_opening_0_share(solver_train)
    # The three rows we plot:
    assert "LP solves" in html
    assert "Simplex iters" in html
    assert "Solve time (s)" in html


def test_chart_backward_opening_0_empty_solver_returns_fallback() -> None:
    """All three opening-0 charts return a fallback <p> when no data is present."""
    from novomodelo_bridge.dashboard.tabs.performance_charts import (
        chart_backward_opening_0_share,
        chart_backward_opening_0_simplex,
        chart_backward_opening_0_solve_time,
    )

    empty = pd.DataFrame()
    for fn in (
        chart_backward_opening_0_solve_time,
        chart_backward_opening_0_simplex,
        chart_backward_opening_0_share,
    ):
        html = fn(empty)
        assert "<p>" in html
        assert "per-opening data" in html or "No opening" in html


def test_opening_0_is_slower_than_rest_in_synthetic_data() -> None:
    """Smoke check: cold multiplier >1 leaves Opening 0 with higher mean than rest."""
    from novomodelo_bridge.dashboard.tabs.performance_charts import (
        _backward_opening_0_split,
    )

    solver_train = _make_solver_train_with_openings(
        n_stages=2, n_iters=3, n_openings=4, cold_multiplier=2.5
    )
    split = _backward_opening_0_split(solver_train)
    mean_by_class = split.groupby("opening_class")["solve_time_ms"].mean()
    assert mean_by_class["opening_0"] > mean_by_class["opening_rest"]
    # With cold_multiplier=2.5, opening_0 should be ≈2.5× opening_rest mean.
    ratio = mean_by_class["opening_0"] / mean_by_class["opening_rest"]
    assert 2.4 < ratio < 2.6


# ---------------------------------------------------------------------------
# test_render_empty_solver
# ---------------------------------------------------------------------------


def test_render_empty_solver() -> None:
    """render() with empty solver data must not raise and show degradation messages."""
    data = _make_mock_data(
        solver_train=pd.DataFrame(),
        solver_sim=pd.DataFrame(),
        timing=pd.DataFrame(),
        scaling_report={},
        retry_histogram=pd.DataFrame(),
    )
    html = render(data)
    assert len(html) > 0
    # Graceful degradation: at least one of the expected fallback strings appears
    assert (
        "No solver data" in html
        or "No scaling report" in html
        or "No retry data" in html
    )


# ---------------------------------------------------------------------------
# Lightweight payload tests — boxplots and scatters must not emit per-sample
# raw arrays. The size-of-dashboard scaling work depends on this invariant.
# ---------------------------------------------------------------------------


class TestLightweightPayloads:
    """Long-run dashboards used to grow to GB-scale because per-iteration box
    plots and scatter charts dumped every raw sample into the HTML. These
    tests pin the lighter payloads in place so future edits don't regress.
    """

    def _backward_synthetic_frame(
        self, n_iter: int = 20, n_stages: int = 30, n_openings: int = 10
    ) -> pd.DataFrame:
        """Reasonably-large synthetic backward solver frame."""
        rows: list[dict] = []
        for i in range(n_iter):
            for s in range(n_stages):
                for o in range(n_openings):
                    rows.append(
                        {
                            "iteration": i,
                            "phase": "backward",
                            "stage": s,
                            "opening": o,
                            "rank": 0,
                            "worker_id": 0,
                            "lp_solves": 1,
                            "solve_time_ms": float((i + s + o) % 50 + 1),
                            "simplex_iterations": (i + s * 2 + o) % 100 + 1,
                        }
                    )
        return pd.DataFrame(rows)

    def test_per_opening_solve_time_box_carries_no_raw_points(self) -> None:
        from novomodelo_bridge.dashboard.tabs.performance_charts import (
            chart_backward_per_opening_solve_time,
        )

        df = self._backward_synthetic_frame()
        html = chart_backward_per_opening_solve_time(df)
        # Box trace must use pre-aggregated stats, never a `"y":[`
        # array longer than the number of stages.
        assert '"type":"box"' in html
        # The aggregated fields are present.
        assert '"q1":[' in html
        assert '"median":[' in html
        assert '"q3":[' in html
        assert '"lowerfence":[' in html
        assert '"upperfence":[' in html
        # No raw y array embedding per-sample solve times.
        # (The pre-aggregated API never emits a top-level `"y":[...]`.)
        assert '"y":[' not in html

    def test_per_opening_simplex_box_carries_no_raw_points(self) -> None:
        from novomodelo_bridge.dashboard.tabs.performance_charts import (
            chart_backward_per_opening_simplex,
        )

        df = self._backward_synthetic_frame()
        html = chart_backward_per_opening_simplex(df)
        assert '"type":"box"' in html
        assert '"q1":[' in html
        assert '"median":[' in html
        assert '"y":[' not in html

    def test_cuts_vs_solve_time_uses_customdata_not_text(self) -> None:
        from novomodelo_bridge.dashboard.tabs.performance_charts import (
            chart_cuts_vs_solve_time_scatter,
        )

        solver = self._backward_synthetic_frame(n_iter=10, n_stages=20, n_openings=5)
        cuts = pd.DataFrame(
            [
                {"iteration": i, "stage": s, "cuts_active_after": (i + s) % 50 + 1}
                for i in range(10)
                for s in range(20)
            ]
        )
        html = chart_cuts_vs_solve_time_scatter(solver, cuts)
        assert '"type":"scatter"' in html
        # Hover-template references customdata + marker.color, not a `"text":[`
        # per-point string array. The prior implementation emitted one
        # ~50-char string per sample which dominated long-run file size.
        assert "customdata" in html
        assert '"text":[' not in html

    def test_cuts_vs_solve_time_aggregates_per_iter_stage(self) -> None:
        """The scatter collapses raw backward samples to one point per
        (iteration, stage) group with a p25-p75 error bar — not one marker per
        (iteration, stage, opening, worker) sample, which reached ~12M points
        (191 MB) on the production case and could not render in a browser.
        """
        import re

        from novomodelo_bridge.dashboard.tabs.performance_charts import (
            chart_cuts_vs_solve_time_scatter,
        )

        n_iter, n_stages, n_openings = 10, 20, 5
        solver = self._backward_synthetic_frame(
            n_iter=n_iter, n_stages=n_stages, n_openings=n_openings
        )
        cuts = pd.DataFrame(
            [
                {"iteration": i, "stage": s, "cuts_active_after": (i + s) % 50 + 1}
                for i in range(n_iter)
                for s in range(n_stages)
            ]
        )
        html = chart_cuts_vs_solve_time_scatter(solver, cuts)
        # Spread is preserved as an asymmetric error bar, not raw points.
        assert '"error_y":' in html
        assert '"arrayminus":' in html
        # Exactly one point per (iteration, stage) group — not per raw sample.
        x_match = re.search(r'"x":\[([^\]]*)\]', html)
        assert x_match is not None
        n_points = x_match.group(1).count(",") + 1
        assert n_points == n_iter * n_stages
        assert n_points < n_iter * n_stages * n_openings
