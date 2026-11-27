"""The convergence comparison of a version comparison.

Bounds are in R$; the upper-to-lower ratio is dimensionless.

"Final" is each run's own last iteration with a finite lower bound, so runs of
different lengths still compare. The judged upper-bound row carries the ratio
``r = upper / lower``, never the gap: under :func:`novomodelo_bridge.core.agreement.gap`
its value is ``|r_c - r_b| / r_b = |g_c - g_b| / (1 + g_b)`` for ``g = r - 1``, so the
tolerance acts on the difference of the two optimality gaps, is 0 for identical runs
and stays defined at a gap of 0. The percentage gaps of :class:`BoundsComparison`
are informational and never judged.

A stage risk weight is a CVaR ``lambda`` in [0, 1], 0 being risk-neutral. A map that
holds a weight outside that range, NaN included, is no risk measure: it counts as
``None``, risk unknown, never as risk-neutral or risk-averse.

The functions emit no Diagnostic and read no file.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl

from novomodelo_bridge.comparators.model import ResultComparison
from novomodelo_bridge.comparators.version.rows import result_row

#: The values of ``upper_bound_comparability``, persisted: a one-way contract.
UB_COMPARABILITY: tuple[str, ...] = (
    "comparable",
    "bounds-missing",
    "kind-missing",
    "kind-differs",
    "risk-averse",
    "risk-unknown",
    "bound-nonpositive",
)

RiskAversion = Mapping[int, float] | None

_PAIRED_VARIABLES = (
    ("lower_bound", "lower_bound"),
    ("upper_bound", "upper_bound"),
)

_BOUNDS_COLUMNS = (
    "iteration",
    "lower_bound",
    "upper_bound",
    "upper_bound_std",
    "upper_bound_kind",
)


@dataclass(frozen=True)
class FinalBounds:
    """The last iteration of one run that has a finite lower bound.

    Attributes:
        iteration: The iteration number.
        lower: The lower bound, R$; finite.
        upper: The upper bound, R$; ``None`` when null, otherwise as read, so NaN
            and infinity stay visible to :func:`classify_upper_bound`.
        kind: The upper-bound kind as read; ``None`` when the output does not say.
    """

    iteration: int
    lower: float
    upper: float | None
    kind: str | None


@dataclass(frozen=True)
class BoundsComparison:
    """Two runs' final bounds and whether their upper bounds compare.

    Attributes:
        baseline: ``None`` when the baseline has no final bounds.
        candidate: As *baseline*.
        comparability: A member of :data:`UB_COMPARABILITY`.
        baseline_gap_percent: ``100 * (upper / lower - 1)`` of the baseline, in
            percent; ``None`` unless its lower bound is above 0 and its upper bound
            is finite.
        candidate_gap_percent: As *baseline_gap_percent*, for the candidate.
        gap_delta_pp: Candidate minus baseline, in percentage points; ``None``
            unless both gaps exist. The three gap fields are a finite float or
            ``None``, never NaN or infinity.
        risk_measures_differ: Whether both risk measures are known and their
            positive weights differ, whatever *comparability* says. Held in memory
            only.
    """

    baseline: FinalBounds | None
    candidate: FinalBounds | None
    comparability: str
    baseline_gap_percent: float | None
    candidate_gap_percent: float | None
    gap_delta_pp: float | None
    risk_measures_differ: bool

    @property
    def upper_bound_comparable(self) -> bool:
        return self.comparability == "comparable"


def _require_columns(bounds: pl.DataFrame) -> None:
    for column in _BOUNDS_COLUMNS:
        if column not in bounds.columns:
            raise ValueError(f"convergence bounds frame lacks column {column!r}")


def _is_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _rows_by_iteration(bounds: pl.DataFrame) -> dict[int, dict[str, Any]]:
    _require_columns(bounds)
    return {
        row["iteration"]: row
        for row in bounds.iter_rows(named=True)
        if row["iteration"] is not None
    }


def final_bounds(bounds: pl.DataFrame) -> FinalBounds | None:
    """The row of the highest iteration whose lower bound is non-null and finite.

    ``None`` when there is none, for an empty or all-null frame alike.

    Raises:
        ValueError: *bounds* lacks a column of the bounds reader's frame; the
            message names it. The reader always writes all five, so this is an
            internal invariant.
    """
    rows = _rows_by_iteration(bounds)
    usable = [it for it, row in rows.items() if _is_finite(row["lower_bound"])]
    if not usable:
        return None
    last = max(usable)
    row = rows[last]
    return FinalBounds(
        iteration=last,
        lower=row["lower_bound"],
        upper=row["upper_bound"],
        kind=row["upper_bound_kind"],
    )


def _known_risk(risk: RiskAversion) -> Mapping[int, float] | None:
    if risk is None or not all(0.0 <= weight <= 1.0 for weight in risk.values()):
        return None
    return risk


def _has_usable_bounds(side: FinalBounds) -> bool:
    return (
        side.lower > 0
        and side.upper is not None
        and side.upper > 0
        and _gap_percent(side) is not None
    )


def classify_upper_bound(
    baseline: FinalBounds | None,
    candidate: FinalBounds | None,
    baseline_risk: RiskAversion,
    candidate_risk: RiskAversion,
) -> str:
    """The first member of :data:`UB_COMPARABILITY` that applies, in this order.

    ``bounds-missing``: either side has no final bounds. ``kind-missing``: either
    kind is null. ``kind-differs``. ``risk-averse``: a known risk map of either
    side holds a weight above 0. ``risk-unknown``: either risk map is ``None`` or
    holds a weight outside [0, 1]. ``bound-nonpositive``: a lower bound at or below
    0, or an upper bound null, non-finite or at or below 0, or a ratio or gap that
    is not finite, on either side. Else ``comparable``, so a comparable pair always
    has its ratio row.

    ``risk-averse`` precedes ``risk-unknown``: a known positive weight on one side
    is a fact whatever the other side's measure is. An empty map is risk-neutral.
    """
    if baseline is None or candidate is None:
        return "bounds-missing"
    if baseline.kind is None or candidate.kind is None:
        return "kind-missing"
    if baseline.kind != candidate.kind:
        return "kind-differs"
    risks = (_known_risk(baseline_risk), _known_risk(candidate_risk))
    if any(
        weight > 0 for risk in risks if risk is not None for weight in risk.values()
    ):
        return "risk-averse"
    if any(risk is None for risk in risks):
        return "risk-unknown"
    if not (_has_usable_bounds(baseline) and _has_usable_bounds(candidate)):
        return "bound-nonpositive"
    return "comparable"


def risk_measures_differ(
    baseline_risk: RiskAversion, candidate_risk: RiskAversion
) -> bool:
    """Whether both risk measures are known and their positive weights differ.

    An empty map and a map of zero weights are the same risk-neutral measure. An
    unknown measure, ``None`` or out of range, is not known to differ, so it gives
    ``False``.
    """
    baseline, candidate = _known_risk(baseline_risk), _known_risk(candidate_risk)
    if baseline is None or candidate is None:
        return False
    return {stage: weight for stage, weight in baseline.items() if weight > 0} != {
        stage: weight for stage, weight in candidate.items() if weight > 0
    }


def _ratio(side: FinalBounds | None) -> float | None:
    if side is None or side.upper is None or side.lower <= 0:
        return None
    if not math.isfinite(side.upper):
        return None
    return _finite_or_none(side.upper / side.lower)


def _gap_percent(side: FinalBounds | None) -> float | None:
    ratio = _ratio(side)
    return None if ratio is None else _finite_or_none(100 * (ratio - 1))


def compare_bounds(
    baseline_bounds: pl.DataFrame,
    candidate_bounds: pl.DataFrame,
    baseline_risk: RiskAversion,
    candidate_risk: RiskAversion,
) -> BoundsComparison:
    """Compare the final bounds of two runs.

    Each frame comes from ``read_novomodelo_convergence_bounds`` and may be empty;
    each risk map from ``read_novomodelo_stage_risk_aversion``, where ``{}`` is
    risk-neutral and ``None`` unknown. The gap of a side without final bounds is
    ``None`` while the other side's is still computed.

    Raises:
        ValueError: a frame lacks a column of the bounds reader's frame.
    """
    baseline = final_bounds(baseline_bounds)
    candidate = final_bounds(candidate_bounds)
    baseline_gap = _gap_percent(baseline)
    candidate_gap = _gap_percent(candidate)
    delta = (
        None
        if baseline_gap is None or candidate_gap is None
        else _finite_or_none(candidate_gap - baseline_gap)
    )
    return BoundsComparison(
        baseline=baseline,
        candidate=candidate,
        comparability=classify_upper_bound(
            baseline, candidate, baseline_risk, candidate_risk
        ),
        baseline_gap_percent=baseline_gap,
        candidate_gap_percent=candidate_gap,
        gap_delta_pp=delta,
        risk_measures_differ=risk_measures_differ(baseline_risk, candidate_risk),
    )


def final_bound_rows(comparison: BoundsComparison) -> list[ResultComparison]:
    """The judged rows of *comparison*: entity type ``final_bounds``, name ``Training``.

    ``final_lower_bound`` (R$) when both sides have final bounds, and
    ``final_bound_ratio`` (dimensionless, upper over lower) only when
    ``comparability`` is ``comparable``. ``[]`` when either side has none.
    """
    baseline, candidate = comparison.baseline, comparison.candidate
    if baseline is None or candidate is None:
        return []
    rows = [
        result_row(
            "final_bounds",
            "Training",
            0,
            0,
            0,
            "final_lower_bound",
            baseline.lower,
            candidate.lower,
        )
    ]
    baseline_ratio, candidate_ratio = _ratio(baseline), _ratio(candidate)
    if (
        comparison.upper_bound_comparable
        and baseline_ratio is not None
        and candidate_ratio is not None
    ):
        rows.append(
            result_row(
                "final_bounds",
                "Training",
                0,
                0,
                0,
                "final_bound_ratio",
                baseline_ratio,
                candidate_ratio,
            )
        )
    return rows


def paired_convergence_rows(
    baseline_bounds: pl.DataFrame, candidate_bounds: pl.DataFrame
) -> list[ResultComparison]:
    """Per-iteration rows over the iterations both runs hold, in ascending order.

    An iteration enters when all four of its bounds, R$, are finite. It gives
    ``lower_bound`` then ``upper_bound``, under
    entity type ``convergence`` and name ``iteration_<n>``, as the existing
    comparators write them; the iteration number is both ids and the stage.

    Raises:
        ValueError: a frame lacks a column of the bounds reader's frame.
    """
    baseline = _rows_by_iteration(baseline_bounds)
    candidate = _rows_by_iteration(candidate_bounds)
    rows: list[ResultComparison] = []
    for iteration in sorted(set(baseline) & set(candidate)):
        b, c = baseline[iteration], candidate[iteration]
        if not all(
            _is_finite(side[column])
            for side in (b, c)
            for _, column in _PAIRED_VARIABLES
        ):
            continue
        name = f"iteration_{iteration}"
        rows += [
            result_row(
                "convergence",
                name,
                iteration,
                iteration,
                iteration,
                variable,
                b[column],
                c[column],
            )
            for variable, column in _PAIRED_VARIABLES
        ]
    return rows
