"""How much each simulation scenario weighs in a statistic across scenarios.

novomodelo samples its simulation scenarios, which then weigh the same, unless
the case enumerates the leaf paths of its policy graph, as a converted DECOMP
case does with its scenario fan. Each scenario is then one leaf path,
simulated once, and ``simulation/scenario_summary.parquet`` carries its
probability; a uniform mean over those scenarios is not the expectation
novomodelo itself reports. Every mean, standard deviation and quantile the
dashboard and the compare report take across scenarios goes through
:class:`ScenarioWeights`, so it uses the case's own probabilities.
"""

from __future__ import annotations

import dataclasses
from typing import overload

import polars as pl

#: Column :meth:`ScenarioWeights.attach` adds, read by the weighted statistics.
WEIGHT = "_scenario_weight"


@dataclasses.dataclass(frozen=True)
class ScenarioWeights:
    """The probability of each simulation scenario, or ``None`` when they
    weigh the same.

    The statistics are aggregation expressions for a ``group_by`` over
    scenario rows. With probabilities, a frame must first go through
    :meth:`attach`. A row whose value is null is left out with its
    probability, as the uniform statistics leave out nulls.
    """

    #: ``scenario_id`` (Int64) and :data:`WEIGHT`, one row per scenario.
    probabilities: pl.DataFrame | None = None

    @classmethod
    def from_probabilities(
        cls, scenario_ids: list[int], probabilities: list[float]
    ) -> ScenarioWeights:
        """Weights from each scenario's probability, in any order."""
        return cls(
            pl.DataFrame(
                {"scenario_id": scenario_ids, WEIGHT: probabilities},
                schema={"scenario_id": pl.Int64, WEIGHT: pl.Float64},
            )
        )

    @property
    def uniform(self) -> bool:
        """Whether every scenario weighs the same."""
        return self.probabilities is None

    @overload
    def attach(self, frame: pl.DataFrame) -> pl.DataFrame: ...
    @overload
    def attach(self, frame: pl.LazyFrame) -> pl.LazyFrame: ...
    def attach(self, frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """*frame* with each row's scenario probability; unchanged when uniform
        or when *frame* has no ``scenario_id`` (an entity the run did not
        write), so a weighted statistic over it fails as a missing column."""
        schema = frame.collect_schema()
        if self.probabilities is None or "scenario_id" not in schema:
            return frame
        dtype = schema["scenario_id"]
        weights = self.probabilities.cast({"scenario_id": dtype})
        if isinstance(frame, pl.LazyFrame):
            return frame.join(weights.lazy(), on="scenario_id", how="left")
        return frame.join(weights, on="scenario_id", how="left")

    def mean(self, column: str) -> pl.Expr:
        """The expected value of *column*."""
        if self.uniform:
            return pl.col(column).mean()
        total = _valid_weight(column).sum()
        return (
            pl.when(total > 0)
            .then((pl.col(column) * pl.col(WEIGHT)).sum() / total)
            .otherwise(None)
            .alias(column)
        )

    def std(self, column: str) -> pl.Expr:
        """The standard deviation of *column*.

        Uniform, the sample standard deviation; with probabilities, the
        population standard deviation of the enumerated distribution, as
        novomodelo reports its own cost spread.
        """
        if self.uniform:
            return pl.col(column).std()
        total = _valid_weight(column).sum()
        deviation = pl.col(column) - self.mean(column)
        variance = (deviation.pow(2) * pl.col(WEIGHT)).sum() / total
        return pl.when(total > 0).then(variance.sqrt()).otherwise(None).alias(column)

    def quantile(self, column: str, q: float) -> pl.Expr:
        """The *q* quantile of *column*.

        Uniform, the linearly interpolated sample quantile. With
        probabilities, each value sits at the midpoint of its probability
        mass on the cumulative scale and the quantile interpolates linearly
        between neighbouring values, clamped to the extreme values.
        """
        if self.uniform:
            return pl.col(column).quantile(q, interpolation="linear")
        valid = pl.col(column).is_not_null()
        present = pl.col(column).filter(valid)
        values = present.sort()
        weights = _valid_weight(column).sort_by(present)
        position = (weights.cum_sum() - weights / 2) / weights.sum()
        next_value, next_position = values.shift(-1), position.shift(-1)
        between = (
            pl.when((position <= q) & (next_position > q))
            .then(
                values
                + (q - position) / (next_position - position) * (next_value - values)
            )
            .otherwise(None)
            .sum()
        )
        return (
            pl.when(values.len() == 0)
            .then(None)
            .when(q <= position.first())
            .then(values.first())
            .when(q >= position.last())
            .then(values.last())
            .otherwise(between)
            .alias(column)
        )


def _valid_weight(column: str) -> pl.Expr:
    """The probabilities of the rows where *column* has a value."""
    return pl.col(WEIGHT).filter(pl.col(column).is_not_null())


#: Every scenario weighing the same: a sampled simulation, or no simulation.
UNIFORM = ScenarioWeights()
