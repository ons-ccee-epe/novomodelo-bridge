"""The findings of a version comparison, one Diagnostic per kind.

Each builder is pure: it returns a Diagnostic, or ``None`` when there is nothing
to report, and emits nothing. The caller that sees every item hands the results
to :func:`emit_findings`, so a finding never splits into one Diagnostic per
entity. A table's side cell is ``baseline``, ``candidate`` or, for an item
missing on both, ``both``. The codes and severities are persisted in the
``--json`` output: a one-way contract.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from novomodelo_bridge.comparators.version.alignment import (
    NameAlignment,
    StageAlignment,
)
from novomodelo_bridge.comparators.version.availability import Unavailable
from novomodelo_bridge.comparators.version.constraints import DifferingConstraint
from novomodelo_bridge.comparators.version.convergence import (
    BoundsComparison,
    FinalBounds,
)
from novomodelo_bridge.comparators.version.labels import SIDES
from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)

_LOG = logging.getLogger(__name__)

_CATEGORY = "Compare data"

_BASELINE, _CANDIDATE = SIDES

_WHOLE_SECTION = "—"

_LOWER_BOUND_ONLY = (
    "The comparison judges the final lower bound only and shows the gap without "
    "judging it."
)

_RISK_MEASURES_SENTENCE = (
    "When the two cases declare different risk measures, their lower bounds bound "
    "different objectives, so a mismatch in the final lower bound can be genuine."
)

_RISK_REASONS = ("risk-averse", "risk-unknown")

_CONSTRAINT_DIFFERS_IN: Mapping[tuple[bool, bool], str] = {
    (True, False): "terms",
    (False, True): "bounds",
    (True, True): "terms and bounds",
}

_UPPER_BOUND_SUMMARIES: Mapping[str, str] = {
    "bounds-missing": (
        "The baseline output or the candidate output has no training convergence "
        "data, so it has no final bounds. The comparison judges neither the final "
        "lower bound nor the gap."
    ),
    "kind-missing": (
        "The baseline output or the candidate output was written before novomodelo "
        "0.14 and does not record how its upper bound was computed, so the two "
        f"upper bounds cannot be told to measure the same thing. {_LOWER_BOUND_ONLY}"
    ),
    "kind-differs": (
        "The baseline output and the candidate output computed their upper bounds "
        "in different ways: one run estimated its upper bound by simulation and "
        "the other computed it exactly, so the two upper bounds are not "
        f"comparable. {_LOWER_BOUND_ONLY}"
    ),
    "risk-averse": (
        "The baseline output or the candidate output comes from a risk-averse run, "
        "so the simulated mean cost it reports as its upper bound does not bound "
        f"the objective of that run. {_LOWER_BOUND_ONLY}"
    ),
    "risk-unknown": (
        "The risk measure of the baseline output or the candidate output could not "
        "be read, so the comparison cannot tell whether the simulated mean cost it "
        "reports as its upper bound bounds the objective of that run. "
        f"{_LOWER_BOUND_ONLY}"
    ),
    "bound-nonpositive": (
        "A final bound of the baseline output or the candidate output is missing, "
        "zero, negative or not finite, so the gap between its upper and lower "
        f"bounds has no meaning. {_LOWER_BOUND_ONLY}"
    ),
}


def one_sided_entities(alignments: Sequence[NameAlignment]) -> Diagnostic | None:
    """The entities one output holds and the other does not, one table row each."""
    rows = sorted(
        [alignment.entity_type, name, side]
        for alignment in alignments
        for side, names in (
            (_BASELINE, alignment.only_in_baseline),
            (_CANDIDATE, alignment.only_in_candidate),
        )
        for name in names
    )
    if not rows:
        return None
    return Diagnostic(
        code="entity-one-sided",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title=f"Entities found in one output only ({len(rows)})",
        summary=(
            "The listed entities are found by name in the baseline output or in "
            "the candidate output but not in both, so the comparison has nothing "
            "to compare them with and leaves them out."
        ),
        table=DiagnosticTable(columns=["Entity type", "Name", "Only in"], rows=rows),
    )


def duplicated_names(alignments: Sequence[NameAlignment]) -> Diagnostic | None:
    """The names an output uses for several entities, one table row per output."""
    rows = sorted(
        [alignment.entity_type, ", ".join(names), side]
        for alignment in alignments
        for duplicate in alignment.duplicates
        for side, names in (
            (_BASELINE, duplicate.baseline_names),
            (_CANDIDATE, duplicate.candidate_names),
        )
        if len(names) > 1
    )
    if not rows:
        return None
    return Diagnostic(
        code="entity-name-duplicated",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title=f"Entity names used more than once in one output ({len(rows)})",
        summary=(
            "The listed names belong to more than one entity in the baseline "
            "output or in the candidate output, so the comparison cannot tell "
            "which entity of the other output each one matches and leaves those "
            "entities out. Names that differ only in letter case or surrounding "
            "blanks count as one name."
        ),
        table=DiagnosticTable(columns=["Entity type", "Name", "Output"], rows=rows),
    )


def differing_stage_sets(stages: StageAlignment) -> Diagnostic | None:
    """The stages one output holds and the other does not, one table row each."""
    rows = sorted(
        [
            *([stage, _BASELINE] for stage in stages.only_in_baseline),
            *([stage, _CANDIDATE] for stage in stages.only_in_candidate),
        ]
    )
    if not rows:
        return None
    return Diagnostic(
        code="stage-sets-differ",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title=f"Stages found in one output only ({len(rows)})",
        summary=(
            "The listed stages are in the baseline output or in the candidate "
            "output but not in both, so the comparison leaves them out and "
            "compares the stages both outputs hold."
        ),
        table=DiagnosticTable(
            columns=["Stage", "Only in"],
            rows=rows,
            justify=["right", "left"],
            caption="Stage ids are 0-based.",
        ),
    )


def differing_constraints(
    differing: Sequence[DifferingConstraint],
) -> Diagnostic | None:
    """The paired generic constraints defined differently, one table row each."""
    rows = sorted(
        [constraint.name, _CONSTRAINT_DIFFERS_IN[constraint.terms, constraint.bounds]]
        for constraint in differing
    )
    if not rows:
        return None
    return Diagnostic(
        code="generic-constraint-differs",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title=f"Generic constraints defined differently in the two outputs ({len(rows)})",
        summary=(
            "The listed generic constraints have the same name in the baseline "
            "output and in the candidate output and different terms or bounds "
            "over the stages both outputs hold. The report draws each one with "
            "the terms and bounds of the candidate output and evaluates each "
            "output with its own definition, so the two values of a listed "
            "constraint are not values of the same constraint."
        ),
        table=DiagnosticTable(columns=["Constraint", "Differs in"], rows=rows),
    )


def unavailable_sections(rows: Sequence[Unavailable]) -> Diagnostic | None:
    """The skipped sections and columns, one table row per ``Unavailable`` row.

    Rows are ordered as ``availability.unavailable_rows`` orders them, a whole
    section before its columns. A row's side is shown as given.
    """
    table_rows = [
        [
            row.section,
            _WHOLE_SECTION if row.column is None else row.column,
            row.side,
        ]
        for row in sorted(
            rows,
            key=lambda row: (
                row.section,
                row.column is not None,
                row.column or "",
                row.side,
            ),
        )
    ]
    if not table_rows:
        return None
    return Diagnostic(
        code="section-unavailable",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title=f"Results unavailable ({len(table_rows)})",
        summary=(
            "Some results are missing in the baseline output, the candidate "
            "output or both, so the comparison leaves the listed sections and "
            "columns out for both outputs and compares everything else."
        ),
        table=DiagnosticTable(
            columns=["Section", "Column", "Missing in"], rows=table_rows
        ),
    )


def _kind_note(output: str, side: FinalBounds | None) -> str:
    if side is None:
        return f"{output}: no training convergence data."
    if side.kind is None:
        return f"{output}: the kind of its upper bound is not recorded."
    return f"{output}: upper bound kind {side.kind}."


def upper_bound_not_comparable(bounds: BoundsComparison) -> Diagnostic | None:
    """Why the two upper bounds are not judged; ``None`` when they are comparable.

    Raises:
        ValueError: ``bounds.comparability`` is not a member of
            ``convergence.UB_COMPARABILITY``.
    """
    if bounds.upper_bound_comparable:
        return None
    summary = _UPPER_BOUND_SUMMARIES.get(bounds.comparability)
    if summary is None:
        raise ValueError(f"unknown upper-bound comparability: {bounds.comparability!r}")
    if bounds.comparability in _RISK_REASONS or bounds.risk_measures_differ:
        summary = f"{summary} {_RISK_MEASURES_SENTENCE}"
    return Diagnostic(
        code="upper-bound-not-comparable",
        severity=Severity.INFO,
        category=_CATEGORY,
        title="Upper bounds of the two runs not comparable",
        summary=summary,
        notes=[
            _kind_note("Baseline output", bounds.baseline),
            _kind_note("Candidate output", bounds.candidate),
        ],
    )


def differing_iteration_counts(bounds: BoundsComparison) -> Diagnostic | None:
    """The two final iterations when they differ; ``None`` when either side has
    no final bounds or the iterations are equal."""
    baseline, candidate = bounds.baseline, bounds.candidate
    if (
        baseline is None
        or candidate is None
        or baseline.iteration == candidate.iteration
    ):
        return None
    return Diagnostic(
        code="iteration-counts-differ",
        severity=Severity.INFO,
        category=_CATEGORY,
        title="Training runs stopped at different iterations",
        summary=(
            f"The final iteration is {baseline.iteration} in the baseline output "
            f"and {candidate.iteration} in the candidate output. The comparison "
            "judges the final bounds of each output at its own last iteration, so "
            "a difference between them can come from the different number of "
            "iterations."
        ),
    )


def same_output_compared(baseline_dir: Path, candidate_dir: Path) -> Diagnostic | None:
    """A note when both directories resolve to one path; ``None`` otherwise."""
    if baseline_dir.resolve() != candidate_dir.resolve():
        return None
    return Diagnostic(
        code="same-output-compared",
        severity=Severity.INFO,
        category=_CATEGORY,
        title="Baseline and candidate are the same output",
        summary=(
            "The baseline output and the candidate output are the same directory, "
            "so the comparison sets a run against itself and every difference it "
            "finds is zero."
        ),
        remediation=(
            "To compare two runs, pass two different output directories as "
            "BASELINE_OUTPUT and CANDIDATE_OUTPUT."
        ),
    )


def emit_findings(findings: Iterable[Diagnostic | None]) -> None:
    """Emit every finding that is not ``None``, in order, on this module's logger."""
    for finding in findings:
        if finding is not None:
            emit(finding, logger=_LOG)
