"""Pure pre-conversion validation of a source-model case directory.

The ``check`` command must answer "would this case convert?" *without* running
the conversion or writing any output. This module is that validation engine: it
resolves files via :meth:`NewaveFiles.from_directory`, confirms the required
inputs are present, raises a per-file advisory for each absent optional input,
runs a couple of structural sanity checks, and returns a structured
:class:`PreflightResult` — a ``✓/✗`` :class:`CheckItem` list plus the
:class:`Diagnostic` list and an overall :class:`PreflightVerdict`.

It contains no Typer code and writes no file, so it is unit-testable with mocks
and reusable by both the CLI command and, later, the ``--dry-run`` path. It does
no deep field parsing: binary inputs (``hidr.dat``, ``vazoes.dat``) are checked
only at the discovery/structural level — their contents are never read here.

The verdict is ``WILL_NOT_CONVERT`` if any check failed, else ``WARNINGS`` if any
``WARNING`` diagnostic was emitted, else ``OK``.
"""

from __future__ import annotations

from pathlib import Path

from cobre_bridge.core.diagnostics import Diagnostic, Severity
from cobre_bridge.core.errors import diagnostic_from_exception
from cobre_bridge.core.preflight import (
    CheckItem,
    PreflightResult,
    PreflightVerdict,
    optional_input_advisory,
)
from cobre_bridge.newave.files import NewaveFiles
from cobre_bridge.newave.switches import (
    DgerSwitches,
    switch_off_diagnostic,
    switched_off_inputs,
)

_DISCOVERY_LABEL = "File discovery (caso.dat → arquivos.dat)"
_PREFLIGHT_CONTEXT = "Preflight"


def _structural_checks(src: Path, files: NewaveFiles) -> list[CheckItem]:
    """Discovery-level structural sanity checks that read no file contents.

    Confirms *src* is a directory and that ``caso.dat`` resolved (``files`` only
    exists once discovery succeeded, so its directory entry point is present).
    Deep field parsing is out of scope.
    """
    return [
        CheckItem(
            label="Source path is a directory",
            passed=src.is_dir(),
            detail=str(src),
        ),
        CheckItem(
            label="caso.dat resolved",
            passed=True,
            detail=str(files.directory),
        ),
    ]


def _verdict_from(
    checks: list[CheckItem], diagnostics: list[Diagnostic]
) -> PreflightVerdict:
    """Compute the verdict: a failed check blocks, else a warning warns, else OK."""
    if any(not check.passed for check in checks):
        return PreflightVerdict.WILL_NOT_CONVERT
    if any(diag.severity is Severity.WARNING for diag in diagnostics):
        return PreflightVerdict.WARNINGS
    return PreflightVerdict.OK


def run_preflight(src: Path) -> PreflightResult:
    """Validate the source-model case at *src* without converting or writing.

    Resolves the case files via :meth:`NewaveFiles.from_directory`; on a
    discovery failure (a missing required input, or any unexpected parse error)
    the failure is captured as a failed :class:`CheckItem` plus an ``ERROR``
    :class:`Diagnostic` and the result is returned immediately with verdict
    ``WILL_NOT_CONVERT`` — nothing propagates. On successful discovery it adds the
    required-files-present check, a per-absent-optional advisory (``INFO``),
    and structural sanity checks, then computes the verdict.

    This function writes no file and raises nothing for a missing input.
    """
    try:
        files = NewaveFiles.from_directory(src)
    except Exception as exc:  # noqa: BLE001
        # Any discovery failure — a missing required input (SourceFileError, a
        # FileNotFoundError subclass) or an unexpected parse error — is non-
        # recoverable for preflight; surface it as a blocking verdict.
        return _discovery_failure(exc)

    checks: list[CheckItem] = [
        CheckItem(
            label="Required files present",
            passed=True,
            detail="none missing",
        )
    ]
    diagnostics: list[Diagnostic] = []

    optional_checks, optional_diags = optional_input_advisory(files)
    checks.extend(optional_checks)
    diagnostics.extend(optional_diags)

    switch_checks, switch_diags = _switch_advisory(files)
    checks.extend(switch_checks)
    diagnostics.extend(switch_diags)

    checks.extend(_structural_checks(src, files))

    return PreflightResult(
        verdict=_verdict_from(checks, diagnostics),
        diagnostics=diagnostics,
        checks=checks,
    )


def _read_switches(files: NewaveFiles) -> DgerSwitches:
    """Parse ``dger.dat`` for its switches; the one content read preflight does."""
    from cobre_bridge.newave.case import NewaveCase

    return NewaveCase(files=files).switches


def _switch_advisory(
    files: NewaveFiles,
) -> tuple[list[CheckItem], list[Diagnostic]]:
    """INFO advisory per optional input that is present but switched off in
    ``dger.dat``; a ``dger.dat`` that does not parse is a failed check."""
    from cobre_bridge.newave.converters.constraints import _find_restricao_eletrica

    try:
        switches = _read_switches(files)
    except Exception as exc:  # noqa: BLE001
        # Any parse failure means the conversion would fail on the same read.
        return (
            [CheckItem(label="dger.dat readable", passed=False, detail=str(exc))],
            [diagnostic_from_exception(exc, context=_PREFLIGHT_CONTEXT)],
        )
    off = switched_off_inputs(
        files,
        switches,
        restricao_eletrica_present=_find_restricao_eletrica(files.directory)
        is not None,
    )
    detail = (
        f"{len(off)} present input(s) switched off"
        if off
        else "every present optional input is switched on"
    )
    return (
        [CheckItem(label="dger.dat switches", passed=True, detail=detail)],
        [switch_off_diagnostic(switch) for switch in off],
    )


def _discovery_failure(exc: Exception) -> PreflightResult:
    """Build the blocking result for a failed file-discovery step."""
    return PreflightResult(
        verdict=PreflightVerdict.WILL_NOT_CONVERT,
        diagnostics=[diagnostic_from_exception(exc, context=_PREFLIGHT_CONTEXT)],
        checks=[
            CheckItem(label=_DISCOVERY_LABEL, passed=False, detail=str(exc)),
        ],
    )
