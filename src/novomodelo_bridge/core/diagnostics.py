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
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum

_MODULE_LOGGER = logging.getLogger(__name__)


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
    rows: list[list[object]]
    justify: list[str] | None = None
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
    "cobre_bridge_diagnostics", default=None
)


@contextmanager
def collect() -> Iterator[list[Diagnostic]]:
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
    rendering and is **not** logged (so it is not also captured by the pipeline's
    legacy-warning bridge). Outside a block it degrades to a single ``logging``
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


class WarningCollector(logging.Handler):
    """Capture ``WARNING``+ records emitted under ``cobre_bridge`` during a run.

    Converters log every degraded-input substitution (vazpast → empty, c_adic →
    no load, EXPT/RE skipped, REE.DAT cutoff fallback, …) at ``WARNING`` level.
    Attaching this to the package logger for the duration of a conversion lets
    the pipeline report those degradations through :func:`finalize_diagnostics`
    instead of letting them pass silently.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            self.messages.append(record.getMessage())


def finalize_diagnostics(
    collected: list[Diagnostic], legacy_messages: list[str]
) -> list[Diagnostic]:
    """Combine structured diagnostics with bridged legacy ``logger.warning`` strings.

    Structured diagnostics are de-duplicated by ``(code, summary)`` (a converter run
    more than once should surface a finding only once). Each remaining captured
    warning string — emitted by a site not yet migrated to :func:`emit` — is
    wrapped in a generic WARNING diagnostic so it still renders through the same
    pipeline, with first-occurrence order preserved.
    """
    result: list[Diagnostic] = []
    seen: set[tuple[str, str]] = set()
    for diag in collected:
        key = (diag.code, diag.summary)
        if key in seen:
            continue
        seen.add(key)
        result.append(diag)
    for message in dict.fromkeys(legacy_messages):
        result.append(
            Diagnostic(
                code="legacy-warning",
                severity=Severity.WARNING,
                category="Other warnings",
                title="Conversion warning",
                summary=message,
            )
        )
    return result


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
