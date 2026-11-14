"""Structured conversion diagnostics — the data model behind the CLI's warnings.

A converter that hits a degraded input (a fictitious plant excluded, a GTMIN that
exceeds capacity, a clamped initial storage, a file it could not parse) describes
the situation as a :class:`Diagnostic` and hands it to :func:`emit`. The pipeline
installs a context-local sink (:func:`collect`) for the duration of a run and
renders the collected diagnostics through the CLI's Rich rendering layer.

This decouples *what happened* (the :class:`Diagnostic` data model — plain data,
no Rich, no I/O) from *how it is shown* (the Rich rendering layer). It replaces
the previous lossy path where every warning was flattened to a ``logging``
string and structured detail — plant names, the stages involved, the values —
was discarded at the call site.

When no sink is active (a converter called directly, e.g. from a unit test),
:func:`emit` degrades to a single ``logging`` record on the caller's logger, so
standalone use and existing log-based tests keep working unchanged.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

_MODULE_LOGGER = logging.getLogger(__name__)
_COUNT_SUFFIX = re.compile(r" \(\d+\)$")

ColumnJustify = Literal["left", "right", "center"]


class Severity(Enum):
    """Severity of a :class:`Diagnostic`, ordered ``INFO < WARNING < ERROR``."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def rank(self) -> int:
        """Sort key so groups render ``INFO`` first and ``ERROR`` last."""
        return _SEVERITY_RANK[self]

    @property
    def log_level(self) -> int:
        """The :mod:`logging` level used by the no-sink fallback in :func:`emit`."""
        return _SEVERITY_LOG_LEVEL[self]


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.ERROR: 2,
}

_SEVERITY_LOG_LEVEL: dict[Severity, int] = {
    Severity.INFO: logging.INFO,
    Severity.WARNING: logging.WARNING,
    Severity.ERROR: logging.ERROR,
}


@dataclass
class DiagnosticTable:
    """A small per-entity / per-stage detail table attached to a diagnostic.

    Kept deliberately generic (just columns + rows) so the renderer never needs to
    know about any particular diagnostic. ``justify`` optionally gives a per-column
    alignment (``"left"`` / ``"right"`` / ``"center"``); numeric columns read best
    right-justified.
    """

    columns: list[str]
    rows: Sequence[Sequence[object]]
    justify: list[ColumnJustify] | None = None
    caption: str | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly form for the ``--diagnostics-json`` sidecar."""
        return {
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
            "justify": list(self.justify) if self.justify is not None else None,
            "caption": self.caption,
        }


@dataclass
class Diagnostic:
    """One structured, user-facing finding from a conversion.

    Attributes
    ----------
    code:
        Stable kebab-case slug identifying the *kind* of finding (e.g.
        ``"thermal-gtmin-above-capacity"``). Stable across runs so automation and
        the JSON sidecar can key on it.
    severity:
        :class:`Severity` — drives colour and grouping order.
    category:
        Human group header (e.g. ``"Thermal bounds"``); diagnostics sharing a
        category render together.
    title:
        Short headline for the diagnostic's panel.
    summary:
        One-line description; this is also what the no-sink fallback logs.
    table:
        Optional :class:`DiagnosticTable` with the per-entity / per-stage detail.
    notes:
        Extra free-form lines shown under the table.
    remediation:
        Optional hint telling the user what to check or change, as a plain
        sentence; the renderer adds the ``→`` marker.
    """

    code: str
    severity: Severity
    category: str
    title: str
    summary: str
    table: DiagnosticTable | None = None
    notes: list[str] = field(default_factory=list)
    remediation: str | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly form for the ``--diagnostics-json`` sidecar."""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "category": self.category,
            "title": self.title,
            "summary": self.summary,
            "table": self.table.to_dict() if self.table is not None else None,
            "notes": list(self.notes),
            "remediation": self.remediation,
        }


# Context-local sink. ``None`` means "no run in progress" — emit then degrades to a
# log record. A ContextVar (not a module global) keeps concurrent conversions in the
# same process from cross-contaminating each other's diagnostics.
_ACTIVE: ContextVar[list[Diagnostic] | None] = ContextVar(
    "novomodelo_bridge_diagnostics", default=None
)


@contextmanager
def collect() -> Generator[list[Diagnostic]]:
    """Install a fresh diagnostic sink for the duration of the ``with`` block.

    Yields the list that :func:`emit` appends to; the caller keeps the reference, so
    the collected diagnostics remain available after the block exits. Nesting is
    supported (the previous sink is restored on exit) but not normally needed.
    """
    sink: list[Diagnostic] = []
    token = _ACTIVE.set(sink)
    try:
        yield sink
    finally:
        _ACTIVE.reset(token)


def emit(diagnostic: Diagnostic, *, logger: logging.Logger | None = None) -> None:
    """Record *diagnostic* on the active sink, or log it when none is active.

    Inside a :func:`collect` block the diagnostic is appended to the sink for rich
    rendering and is **not** logged. Outside a block it degrades to a single ``logging``
    record at the matching level on *logger* (default: this module's logger) so a
    converter called on its own — or an existing log-assertion test — still surfaces
    the finding. Pass the caller's module logger to keep the record's logger name.
    """
    sink = _ACTIVE.get()
    if sink is not None:
        sink.append(diagnostic)
        return
    (logger or _MODULE_LOGGER).log(
        diagnostic.severity.log_level, "%s", diagnostic.summary
    )


def finalize_diagnostics(collected: Iterable[Diagnostic]) -> list[Diagnostic]:
    """*collected* with each ``(code, summary)`` kept once, in first-met order.

    A converter run twice, or a file that several report tabs read, reports
    its finding once.
    """
    first: dict[tuple[str, str], Diagnostic] = {}
    for diagnostic in collected:
        first.setdefault((diagnostic.code, diagnostic.summary), diagnostic)
    return list(first.values())


def merge_tables(diagnostics: Iterable[Diagnostic]) -> list[Diagnostic]:
    """One diagnostic per code, for an emitter that reports each entity as it
    meets it: the distinct rows of every same-code diagnostic's table in one
    table, in the order met, with their count after the first one's title,
    in place of any count that title already ends with.

    A diagnostic without a table passes through unchanged.
    """
    merged: dict[str, tuple[Diagnostic, DiagnosticTable, list[list[object]]]] = {}
    order: list[Diagnostic | str] = []
    for diagnostic in diagnostics:
        table = diagnostic.table
        if table is None:
            order.append(diagnostic)
            continue
        if diagnostic.code not in merged:
            merged[diagnostic.code] = (diagnostic, table, [])
            order.append(diagnostic.code)
        rows = merged[diagnostic.code][2]
        for row in table.rows:
            if list(row) not in rows:
                rows.append(list(row))
    result: list[Diagnostic] = []
    for item in order:
        if isinstance(item, Diagnostic):
            result.append(item)
            continue
        first, table, rows = merged[item]
        result.append(
            replace(
                first,
                title=f"{_COUNT_SUFFIX.sub('', first.title)} ({len(rows)})",
                table=replace(table, rows=rows),
            )
        )
    return result


def counted(count: int, noun: str, plural: str | None = None) -> str:
    """*count* and *noun* agreeing in number: ``"1 plant"``, ``"3 plants"``."""
    return f"{count} {noun if count == 1 else plural or noun + 's'}"


def format_stage_ranges(stages: Iterable[int]) -> str:
    """Collapse stage ids into a compact range string (``[4,5,6,7,19] -> "4-7, 19"``).

    Ids are sorted and de-duplicated first. Returns ``""`` for an empty input. Stage
    ids are rendered as-is (0-based, matching the ``stage_id`` columns in the Parquet
    outputs).
    """
    ordered = sorted({int(s) for s in stages})
    if not ordered:
        return ""
    groups: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in groups)


def exception_reason(exc: BaseException) -> str:
    """*exc* as the reason a diagnostic gives: the bridge's own messages as
    written, any other exception led by its type, without which a ``KeyError``
    reads as a bare quoted name."""
    text = str(exc)
    if type(exc).__module__.startswith("novomodelo_bridge."):
        return text
    name = type(exc).__name__
    return f"{name}: {text}" if text else name


def unreadable_file_diagnostic(
    *, code: str, file_name: str, category: str, effect: str, exc: Exception
) -> Diagnostic:
    """WARNING for an optional deck file the conversion could not parse.

    *effect* completes ``"<file> could not be read, so ..."``: what the case
    lacks and how the results differ. The exception goes to the notes.
    """
    return Diagnostic(
        code=code,
        severity=Severity.WARNING,
        category=category,
        title=f"{file_name} could not be read",
        summary=f"{file_name} could not be read, so {effect}",
        notes=[f"Reason: {exception_reason(exc)}"],
        remediation=f"Check {file_name} for a malformed record.",
    )


#: The category of every finding about a file that compare, dashboard or
#: export reads: the converted case, its run output, or the source model's
#: own results.
REPORT_FILES_CATEGORY = "Case and result files"


def report_file_missing_diagnostic(
    file_name: str, path: Path, effect: str
) -> Diagnostic:
    """INFO for a file a report reads that is not there.

    *file_name* is the file's path in the case (``system/hydros.json``,
    ``output/training/convergence.parquet``) or a source-model file's name
    (``pmo.dat``); *effect* completes ``"<file>
    was not found, so ..."``: what the report leaves out. A reader whose file
    is absent by design, as for a run without that phase, emits nothing.
    """
    return Diagnostic(
        code="report-file-missing",
        severity=Severity.INFO,
        category=REPORT_FILES_CATEGORY,
        title=f"{file_name} not found",
        summary=f"{file_name} was not found, so {effect}",
        notes=[f"Path: {path}"],
    )


def report_file_unreadable_diagnostic(
    file_name: str, path: Path, effect: str, reason: Exception | str
) -> Diagnostic:
    """WARNING for a file a report reads but cannot use.

    *effect* completes ``"<file> could not be read, so ..."``: what the
    report leaves out of what the run computed. *reason* is the exception,
    or what the file lacks.
    """
    notes = [f"Path: {path}"]
    if isinstance(reason, Exception):
        notes.append(f"Reason: {exception_reason(reason)}")
    elif reason:
        notes.append(f"Reason: {reason}")
    return Diagnostic(
        code="report-file-unreadable",
        severity=Severity.WARNING,
        category=REPORT_FILES_CATEGORY,
        title=f"{file_name} could not be read",
        summary=f"{file_name} could not be read, so {effect}",
        notes=notes,
    )


def report_files_unreadable_diagnostic(
    kind: str, failures: Sequence[tuple[str, Exception]], effect: str
) -> Diagnostic:
    """WARNING for the files of one *kind* a report reads but cannot parse.

    *kind* starts the title (``"Interchange tables"``); *failures* holds each
    file's name and exception, one table row each; *effect* completes
    ``"The listed files could not be read, so ..."``.
    """
    return Diagnostic(
        code="report-file-unreadable",
        severity=Severity.WARNING,
        category=REPORT_FILES_CATEGORY,
        title=f"{kind} that could not be read ({len(failures)})",
        summary=f"The listed files could not be read, so {effect}",
        table=DiagnosticTable(
            columns=["File", "Reason"],
            rows=[[name, exception_reason(exc)] for name, exc in failures],
        ),
    )
