"""The shared preflight report model."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

from cobre_bridge.core.diagnostics import Diagnostic, Severity

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_OPTIONAL_ABSENT_DETAIL = "absent (optional; conversion proceeds)"

_FilesT = TypeVar("_FilesT", bound="DataclassInstance")


class PreflightVerdict(Enum):
    """Overall outcome of a preflight run, in increasing order of severity."""

    OK = "ok"
    WARNINGS = "warnings"
    WILL_NOT_CONVERT = "will-not-convert"


@dataclass(frozen=True)
class CheckItem:
    """One ``✓/✗`` line in the preflight report.

    Attributes
    ----------
    label:
        Short human description of what was checked.
    passed:
        ``True`` for a ``✓`` line, ``False`` for a ``✗`` (blocking) line.
    detail:
        Optional context shown after the label (e.g. why it failed, or the
        absent-optional note).
    """

    label: str
    passed: bool
    detail: str | None = None


@dataclass(frozen=True)
class PreflightResult:
    """Structured outcome of :func:`run_preflight`.

    Attributes
    ----------
    verdict:
        The overall :class:`PreflightVerdict`.
    diagnostics:
        The :class:`Diagnostic` list — ``ERROR`` for a blocking discovery
        failure, ``INFO`` for each absent optional input.
    checks:
        The ``✓/✗`` :class:`CheckItem` list the renderer consumes.
    """

    verdict: PreflightVerdict
    diagnostics: list[Diagnostic] = field(default_factory=list)
    checks: list[CheckItem] = field(default_factory=list)


def _optional_file_fields(files: DataclassInstance) -> list[str]:
    """Optional (``Path | None``) field names of the passed files dataclass.

    Derived from the dataclass annotations (the ``| None`` ones) rather than a
    duplicated literal list, so the advisory stays in sync if the discovery
    dataclass grows new optional inputs. ``directory`` is excluded as it is not a
    discovered input file.
    """
    return [
        f.name for f in fields(files) if f.name != "directory" and "None" in str(f.type)
    ]


# ruff's PEP 695 rewrite of this signature is unsafe/incomplete (--unsafe-fixes
# renames the type parameter but leaves the body's `_FilesT` annotation as-is).
def optional_input_advisory(  # noqa: UP047
    files: _FilesT,
) -> tuple[list[CheckItem], list[Diagnostic]]:
    """Passing check + INFO advisory per absent optional input of *files*.

    Reflects over the ``Path | None`` fields of *files*'s dataclass (any
    files dataclass — the NEWAVE and DECOMP engines share this helper). Each
    that resolved to ``None`` yields a passing :class:`CheckItem` plus a
    ``Severity.INFO`` :class:`Diagnostic` (``code="optional-file-absent"``):
    an absent optional never blocks conversion and, being INFO not WARNING,
    never drives the ``WARNINGS`` verdict. Present optionals produce nothing.
    """
    checks: list[CheckItem] = []
    diagnostics: list[Diagnostic] = []
    for name in _optional_file_fields(files):
        if getattr(files, name) is not None:
            continue
        checks.append(
            CheckItem(
                label=f"Optional: {name}",
                passed=True,
                detail=_OPTIONAL_ABSENT_DETAIL,
            )
        )
        diagnostics.append(
            Diagnostic(
                code="optional-file-absent",
                severity=Severity.INFO,
                category="Preflight",
                title="Optional input absent",
                summary=(
                    f"Optional input '{name}' was not found; "
                    "the conversion proceeds without it."
                ),
                notes=[f"field: {name}"],
                remediation=(
                    "Provide the file if this case relies on it; "
                    "otherwise this is informational."
                ),
            )
        )
    return checks, diagnostics
