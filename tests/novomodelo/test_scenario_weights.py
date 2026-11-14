"""Tests for :class:`ScenarioWeights`, the statistics taken across scenarios.

Uniform weights must reproduce polars' own mean, sample std and linear
quantile exactly, so a sampled run reads as it always did; probabilities give
the expectation, the population std and the midpoint-rule quantile.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import polars as pl
import pytest

from novomodelo_bridge.novomodelo.scenario_weights import (
    UNIFORM,
    WEIGHT,
    ScenarioWeights,
)


def _frame(
    values: Sequence[float | None], groups: list[int] | None = None
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "scenario_id": list(range(len(values))),
            "group": groups if groups is not None else [0] * len(values),
            "value": values,
        },
        schema={"scenario_id": pl.Int64, "group": pl.Int64, "value": pl.Float64},
    )


def _census(probabilities: list[float]) -> ScenarioWeights:
    return ScenarioWeights.from_probabilities(
        list(range(len(probabilities))), probabilities
    )


def _stat(frame: pl.DataFrame, weights: ScenarioWeights, expr: pl.Expr) -> float | None:
    return weights.attach(frame).select(expr).item()


class TestUniform:
    """Every statistic is polars' own when the scenarios weigh the same."""

    @pytest.fixture
    def frame(self) -> pl.DataFrame:
        return _frame(
            [3.0, None, 1.0, 7.0, 4.0, 10.0, 2.0],
            groups=[0, 0, 0, 1, 1, 1, 1],
        )

    def test_uniform_is_the_default(self) -> None:
        assert UNIFORM.uniform
        assert ScenarioWeights().uniform

    def test_attach_leaves_the_frame_unchanged(self, frame: pl.DataFrame) -> None:
        assert UNIFORM.attach(frame).equals(frame)

    @pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_statistics_match_polars(self, frame: pl.DataFrame, q: float) -> None:
        got = (
            frame.group_by("group")
            .agg(
                UNIFORM.mean("value").alias("mean"),
                UNIFORM.std("value").alias("std"),
                UNIFORM.quantile("value", q).alias("q"),
            )
            .sort("group")
        )
        expected = (
            frame.group_by("group")
            .agg(
                pl.col("value").mean().alias("mean"),
                pl.col("value").std().alias("std"),
                pl.col("value").quantile(q, interpolation="linear").alias("q"),
            )
            .sort("group")
        )
        assert got.equals(expected)


class TestAttach:
    def test_casts_the_probabilities_to_the_frame_scenario_dtype(self) -> None:
        frame = _frame([1.0, 2.0]).cast({"scenario_id": pl.Int32})

        attached = _census([0.25, 0.75]).attach(frame)

        assert attached[WEIGHT].to_list() == [0.25, 0.75]
        assert attached.schema["scenario_id"] == pl.Int32

    def test_attaches_to_a_lazy_frame(self) -> None:
        weights = _census([0.25, 0.75])

        result = (
            weights.attach(_frame([100.0, 200.0]).lazy())
            .select(weights.mean("value"))
            .collect(engine="streaming")
        )

        assert result.item() == pytest.approx(175.0)

    def test_a_frame_without_scenarios_is_left_unchanged(self) -> None:
        frame = pl.DataFrame({"stage_id": [0, 1]})

        assert _census([0.5, 0.5]).attach(frame).equals(frame)


class TestWeightedMean:
    def test_is_the_expectation(self) -> None:
        weights = _census([0.25, 0.75])

        assert _stat(
            _frame([100.0, 200.0]), weights, weights.mean("value")
        ) == pytest.approx(175.0)

    def test_a_null_value_drops_with_its_probability(self) -> None:
        weights = _census([0.2, 0.3, 0.5])

        assert _stat(
            _frame([100.0, None, 300.0]), weights, weights.mean("value")
        ) == pytest.approx((0.2 * 100.0 + 0.5 * 300.0) / 0.7)

    def test_an_all_null_group_has_no_mean(self) -> None:
        weights = _census([0.5, 0.5])

        assert _stat(_frame([None, None]), weights, weights.mean("value")) is None

    def test_is_taken_per_group(self) -> None:
        weights = _census([0.1, 0.9, 0.6, 0.4])
        frame = _frame([10.0, 20.0, 30.0, 40.0], groups=[0, 0, 1, 1])

        result = (
            weights.attach(frame)
            .group_by("group")
            .agg(weights.mean("value"))
            .sort("group")
        )

        assert result["value"].to_list() == pytest.approx(
            [(1.0 + 18.0) / 1.0, (18.0 + 16.0) / 1.0]
        )


class TestWeightedStd:
    def test_is_the_population_std_of_the_distribution(self) -> None:
        weights = _census([0.25, 0.75])

        # Mean 175: 0.25 * 75**2 + 0.75 * 25**2 = 1875.
        assert _stat(
            _frame([100.0, 200.0]), weights, weights.std("value")
        ) == pytest.approx(math.sqrt(1875.0))

    def test_equal_probabilities_give_the_population_std(self) -> None:
        weights = _census([0.25] * 4)
        values = [1.0, 2.0, 4.0, 9.0]

        expected = pl.Series(values).std(ddof=0)

        assert _stat(_frame(values), weights, weights.std("value")) == pytest.approx(
            expected
        )

    def test_an_all_null_group_has_no_std(self) -> None:
        weights = _census([0.5, 0.5])

        assert _stat(_frame([None, None]), weights, weights.std("value")) is None


class TestWeightedQuantile:
    @pytest.mark.parametrize(
        ("q", "expected"),
        [
            # Midpoints of the cumulative mass: 100 at 0.125, 200 at 0.625.
            (0.1, 100.0),
            (0.125, 100.0),
            (0.5, 100.0 + (0.5 - 0.125) / 0.5 * 100.0),
            (0.625, 200.0),
            (0.9, 200.0),
        ],
    )
    def test_interpolates_between_probability_midpoints(
        self, q: float, expected: float
    ) -> None:
        weights = _census([0.25, 0.75])

        assert _stat(
            _frame([100.0, 200.0]), weights, weights.quantile("value", q)
        ) == pytest.approx(expected)

    def test_does_not_depend_on_row_order(self) -> None:
        weights = _census([0.75, 0.25])

        assert _stat(
            _frame([200.0, 100.0]), weights, weights.quantile("value", 0.5)
        ) == pytest.approx(175.0)

    def test_equal_probabilities_put_the_median_on_the_middle_value(self) -> None:
        weights = _census([1 / 3] * 3)

        assert _stat(
            _frame([3.0, 1.0, 2.0]), weights, weights.quantile("value", 0.5)
        ) == pytest.approx(2.0)

    def test_a_null_value_drops_with_its_probability(self) -> None:
        weights = _census([0.25, 0.5, 0.25])

        # The two remaining values weigh the same: midpoints 0.25 and 0.75.
        assert _stat(
            _frame([100.0, None, 200.0]), weights, weights.quantile("value", 0.5)
        ) == pytest.approx(150.0)

    @pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
    def test_a_single_value_is_every_quantile(self, q: float) -> None:
        weights = _census([1.0])

        assert _stat(
            _frame([42.0]), weights, weights.quantile("value", q)
        ) == pytest.approx(42.0)

    def test_an_all_null_group_has_no_quantile(self) -> None:
        weights = _census([0.5, 0.5])

        assert (
            _stat(_frame([None, None]), weights, weights.quantile("value", 0.5)) is None
        )

    def test_is_taken_per_group_in_a_streaming_lazy_query(self) -> None:
        weights = _census([0.25, 0.75, 0.5, 0.5])
        frame = _frame([100.0, 200.0, 10.0, 30.0], groups=[0, 0, 1, 1])

        result = (
            weights.attach(frame.lazy())
            .group_by("group")
            .agg(weights.quantile("value", 0.5))
            .sort("group")
            .collect(engine="streaming")
        )

        assert result["value"].to_list() == pytest.approx([175.0, 20.0])
