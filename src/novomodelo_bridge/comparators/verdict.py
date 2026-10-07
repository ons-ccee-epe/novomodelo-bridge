"""Leaf ANALYZE-layer module deriving the headline compare verdict.

This module turns a canonical :class:`ComparisonDataset` into a one-line
:class:`CompareVerdict` — the count of variables within tolerance, the total
variable count, and the single worst variable plus its sMAPE. It is the
single-source-of-truth headline consumed by the compare console output: every
number is read straight from ``dataset.summary`` rows, and NO statistic is
recomputed here (no sMAPE, no tolerance re-application). The tolerance was
already applied when the source-model-vs-Cobre summary was built upstream.

It is a leaf: it imports nothing from the other comparators at runtime (the
:class:`ComparisonDataset` type is referenced under ``TYPE_CHECKING`` only),
so the verdict can be wired into the console without creating import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cobre_bridge.comparators.dataset import ComparisonDataset


@dataclass(frozen=True)
class CompareVerdict:
    """One-line headline verdict derived from a :class:`ComparisonDataset`.

    Attributes:
        within_tol: Number of variables fully within tolerance (every
            observation for the variable matched).
        total: Total number of variables compared.
        worst_variable: The variable with the largest ``max_smape``, or ``None``
            when there are no variables.
        worst_smape: That variable's ``max_smape`` — the raw sMAPE ratio (NOT a
            percentage); ``0.0`` when there are no variables.
        all_within_tol: ``True`` iff ``total > 0`` and ``within_tol == total``.
    """

    within_tol: int
    total: int
    worst_variable: str | None
    worst_smape: float
    all_within_tol: bool


def build_compare_verdict(dataset: ComparisonDataset) -> CompareVerdict:
    """Derive the headline verdict from a dataset's per-variable summary.

    Reads only ``dataset.summary`` rows (via ``.to_dicts()``); it does not touch
    ``dataset.tidy`` or ``dataset.metadata`` and recomputes no statistic. A
    variable counts toward ``within_tol`` iff its ``within_tol_rate`` is exactly
    ``1.0`` (every observation within tolerance). The worst variable is the one
    with the largest ``max_smape``, with ties broken by ascending variable name
    so the result is deterministic.

    Args:
        dataset: A validated dataset whose ``summary`` frame conforms to
            ``SUMMARY_SCHEMA``. Assumed already validated by its builder; this
            function does not re-validate.

    Returns:
        The :class:`CompareVerdict` headline. For an empty dataset this is
        ``CompareVerdict(within_tol=0, total=0, worst_variable=None,
        worst_smape=0.0, all_within_tol=False)``.
    """
    rows = dataset.summary.to_dicts()
    total = len(rows)
    within_tol = sum(1 for r in rows if float(r["within_tol_rate"]) == 1.0)
    worst_variable: str | None = None
    worst_smape = 0.0
    if rows:
        worst = min(rows, key=lambda r: (-float(r["max_smape"]), str(r["variable"])))
        worst_variable = str(worst["variable"])
        worst_smape = float(worst["max_smape"])
    return CompareVerdict(
        within_tol=within_tol,
        total=total,
        worst_variable=worst_variable,
        worst_smape=worst_smape,
        all_within_tol=total > 0 and within_tol == total,
    )


def compare_status(dataset: ComparisonDataset) -> str:
    """The converged compare ``--json`` status: one vocabulary for both tracks.

    Returns ``"no-comparable-rows"`` for an empty dataset (``dataset.summary``
    has no rows — zero paired comparisons), else ``"ok"`` when
    :func:`build_compare_verdict` reports every variable within tolerance,
    else ``"mismatch"``. Checked on ``summary``, not ``tidy`` — ``tidy`` also
    carries single-source percentile rows (p10/p50/p90), so a zero-paired-
    comparisons run can leave ``tidy`` non-empty while ``summary`` (the frame
    :func:`build_compare_verdict` counts for ``total``) is empty.
    """
    if dataset.summary.is_empty():
        return "no-comparable-rows"
    return "ok" if build_compare_verdict(dataset).all_within_tol else "mismatch"
