"""Typed conversion/compare failures and their mapping to :class:`Diagnostic`.

A hard failure — a required source-model input file missing, a field that could
not be parsed, an existing Cobre output that was unreadable — is raised as a
:class:`BridgeError` subclass carrying the location detail (path, field, row)
that a flat ``str(exc)`` would lose. :func:`diagnostic_from_exception` then maps a
caught ``(exception, context)`` pair into a single ``ERROR``-severity
:class:`Diagnostic` (cause → location → remediation), so the same rich rendering
path used for warnings also carries terminal failures.

This module is pure data + mapping: it performs no I/O and no CLI wiring. The
exception types stay catchable by the existing discovery handlers —
:class:`SourceFileError` deliberately also subclasses :class:`FileNotFoundError`
so an ``except FileNotFoundError`` site keeps catching it unchanged.
"""

from __future__ import annotations

from cobre_bridge.core.diagnostics import Diagnostic, Severity


class BridgeError(Exception):
    """Base class for typed conversion/compare failures raised by the bridge."""


class SourceFileError(BridgeError, FileNotFoundError):
    """A required source-model input file was missing, or its discovery failed.

    Raised when a required input of the source model is absent, or when the
    indirect file-discovery chain (``caso.dat`` → ``arquivos.dat`` → data files)
    could not be resolved. Also subclasses :class:`FileNotFoundError` so existing
    ``except FileNotFoundError`` discovery handlers keep catching it.
    """

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        field: str | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.field = field


class FieldParseError(BridgeError):
    """A field in a source-model file could not be parsed."""

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        field: str | None = None,
        row: int | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.field = field
        self.row = row


class CobreOutputError(BridgeError):
    """An existing Cobre output file was unreadable or malformed."""

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path


class CobrePartitionMissingError(BridgeError):
    """A required Cobre output simulation partition does not exist.

    Raised by readers that must fail loudly instead of silently returning an
    empty frame when their source partition is absent — e.g. because the
    Cobre output directory predates the cobre version that introduced the
    partition. The message names the missing directory and the minimum
    required cobre version, so the caller learns their *output* is stale
    rather than concluding there is nothing to report.
    """

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path


# Per-type mapping: exception type → (code, remediation). Looked up by walking
# ``type(exc).__mro__`` so a subclass still resolves to its base's entry. An
# exception not present here falls back to ``_FALLBACK`` ("unexpected-error").
_TYPE_MAP: dict[type[BaseException], tuple[str, str | None]] = {
    SourceFileError: (
        "source-file-missing",
        "Check the named file exists in the case directory.",
    ),
    FieldParseError: (
        "source-field-parse",
        "Check the named field in the source file for a malformed value.",
    ),
    CobreOutputError: (
        "cobre-output-unreadable",
        "Re-run cobre with --output to regenerate the output directory.",
    ),
    CobrePartitionMissingError: (
        "cobre-partition-missing",
        "Re-run cobre at the required version to produce this partition.",
    ),
    # Base-class fallback (last, for readability — a concrete subclass above
    # always wins the MRO walk regardless of dict order). No remediation: an
    # unmapped BridgeError has no knowable specific fix.
    BridgeError: ("conversion-error", None),
}

_FALLBACK: tuple[str, str | None] = ("unexpected-error", None)

# Location attributes mapped to their note label, in note-rendering order.
_LOCATION_LABELS: tuple[tuple[str, str], ...] = (
    ("path", "file"),
    ("field", "field"),
    ("row", "row"),
)


def _resolve(exc: Exception) -> tuple[str, str | None]:
    """Return ``(code, remediation)`` for *exc*, walking its MRO."""
    for ancestor in type(exc).__mro__:
        entry = _TYPE_MAP.get(ancestor)
        if entry is not None:
            return entry
    return _FALLBACK


def _location_notes(exc: Exception) -> list[str]:
    """Render the present location attributes of *exc* as ``"<label>: <value>"``."""
    notes: list[str] = []
    for attr, label in _LOCATION_LABELS:
        value = getattr(exc, attr, None)
        if value is not None:
            notes.append(f"{label}: {value}")
    return notes


def diagnostic_from_exception(exc: Exception, *, context: str) -> Diagnostic:
    """Map a caught exception to an ``ERROR``-severity :class:`Diagnostic`.

    The ``code`` and ``remediation`` are chosen per exception type (a subclass
    resolves to its base's entry); an unrecognised exception falls back to
    ``"unexpected-error"`` with no remediation. *context* names the operation
    that failed (``"Conversion"``, ``"Dashboard"``, …) and drives both the
    ``category`` and the ``title``, so the same exception reads correctly under
    every command. ``summary`` is ``str(exc)``, and ``notes`` lists any present
    location attributes (``path``, ``field``, ``row``) as ``"<label>: <value>"``.
    """
    code, remediation = _resolve(exc)
    return Diagnostic(
        code=code,
        severity=Severity.ERROR,
        category=f"{context} failure",
        title=f"{context} failed",
        summary=str(exc),
        notes=_location_notes(exc),
        remediation=remediation,
    )
