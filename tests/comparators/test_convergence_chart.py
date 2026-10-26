"""Tests for :func:`novomodelo_bridge.comparators.charts.convergence_chart`."""

from __future__ import annotations

import json

import polars as pl

from novomodelo_bridge.comparators.charts import convergence_chart


def _traces(html: str) -> dict[str, dict[int, float]]:
    start = html.index("Plotly.newPlot(")
    payload = html[html.index(", ", start) + 2 :]
    traces = json.JSONDecoder().raw_decode(payload)[0]
    return {t["name"]: dict(zip(t["x"], t["y"], strict=True)) for t in traces}


def _novomodelo() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "iteration": [1, 2, 3],
            "lower_bound": [10.0, 20.0, 25.0],
            "upper_bound_mean": [40.0, 45.0, 30.0],
        }
    )


def test_source_best_bound_pairs_with_the_novomodelo_running_minimum() -> None:
    source = pl.DataFrame(
        {
            "iteration": [1, 2, 3],
            "lower_bound": [12.0, 21.0, 26.0],
            "upper_bound_mean": [41.0, 44.0, 31.0],
            "upper_bound_best": [41.0, 41.0, 31.0],
        }
    )

    traces = _traces(convergence_chart(source, _novomodelo()))

    assert traces["NEWAVE ZSUP ITER"] == {1: 41.0, 2: 44.0, 3: 31.0}
    assert traces["NEWAVE ZSUP"] == {1: 41.0, 2: 41.0, 3: 31.0}
    assert traces["Novomodelo Upper"] == {1: 40.0, 2: 45.0, 3: 30.0}
    assert traces["Novomodelo Upper, running min"] == {1: 40.0, 2: 40.0, 3: 30.0}


def test_source_without_a_per_iteration_upper_bound_draws_only_its_best() -> None:
    source = pl.DataFrame(
        {
            "iteration": [1, 2],
            "lower_bound": [12.0, 21.0],
            "upper_bound_mean": [None, None],
            "upper_bound_best": [41.0, 33.0],
        },
        schema_overrides={"upper_bound_mean": pl.Float64},
    )

    traces = _traces(convergence_chart(source, _novomodelo(), reference_label="DECOMP"))

    assert "DECOMP ZSUP ITER" not in traces
    assert traces["DECOMP ZSUP"] == {1: 41.0, 2: 33.0}
    assert "Novomodelo Upper, running min" in traces


def test_novomodelo_running_minimum_needs_a_source_best_bound() -> None:
    traces = _traces(convergence_chart(pl.DataFrame(), _novomodelo()))

    assert set(traces) == {"Novomodelo Lower", "Novomodelo Upper"}
