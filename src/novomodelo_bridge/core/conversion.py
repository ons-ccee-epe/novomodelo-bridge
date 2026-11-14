"""Shared conversion-report and output-rollback model for both conversion tracks."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.errors import OutputExistsError


@dataclass
class ConversionReport:
    """Summary of a completed conversion to a Novomodelo case."""

    hydro_count: int = 0
    thermal_count: int = 0
    bus_count: int = 0
    line_count: int = 0
    stage_count: int = 0
    #: Structured findings (rich tables, severities, remediation) for the CLI to
    #: render. Populated by :func:`run_conversion`.
    diagnostics: list[dx.Diagnostic] = field(default_factory=list)
    #: Flat WARNING-severity summary strings, kept for backward-compatible consumers
    #: (derived from :attr:`diagnostics`).
    warnings: list[str] = field(default_factory=list)
    #: Absolute output paths the conversion produced (real run) or would produce
    #: (dry run), in write order. Optional Parquet tables that were skipped are
    #: absent. Populated by each track's conversion.
    would_write_paths: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Converted: {self.hydro_count} hydros, "
            f"{self.thermal_count} thermals, "
            f"{self.bus_count} buses, "
            f"{self.line_count} lines, "
            f"{self.stage_count} stages"
        )


@dataclass(frozen=True)
class ClearedArtifacts:
    """A conversion track's on-disk output set, for --force pre-clear and
    failure rollback: subdirectories removed as a tree, files unlinked."""

    subdirs: tuple[str, ...]
    files: tuple[str, ...]


def clear_dst_contents(dst: Path, artifacts: ClearedArtifacts) -> None:
    """Remove *artifacts*' known output subdirectories and top-level files from dst.

    Only the specific files/subdirectories named by *artifacts* are removed.
    This avoids accidentally deleting unrelated files in the destination
    directory.
    """
    for subdir in artifacts.subdirs:
        target = dst / subdir
        if target.exists():
            shutil.rmtree(target)

    for filename in artifacts.files:
        path = dst / filename
        if path.exists():
            path.unlink()


def _remove_empty_dirs(dirs: list[Path]) -> None:
    """Remove each of *dirs*, given deepest first, until one is not empty."""
    for path in dirs:
        try:
            path.rmdir()
        except OSError:
            return


def run_conversion(
    convert: Callable[[], ConversionReport],
    dst: Path,
    artifacts: ClearedArtifacts,
    *,
    force: bool,
    dry_run: bool,
) -> ConversionReport:
    """Run one track's *convert* into *dst* and return its report with every finding.

    A *dst* that already holds files raises :class:`OutputExistsError` unless
    *force*, which first clears *artifacts* from it. On a failure, *artifacts*
    and the directories this run created are removed, the findings collected
    so far are emitted on the caller's sink (an early warning such as a tab in
    a fixed-column file often explains the failure, and a failed self-check
    details it), and the exception propagates. A dry run never changes *dst*.
    """
    # Refused before the rollback below can delete the earlier case.
    if dst.exists() and any(dst.iterdir()):
        if not force:
            raise OutputExistsError(f"'{dst}' already holds files.")
        if not dry_run:
            # A conditional artifact this run does not write again must not
            # survive into the new case.
            clear_dst_contents(dst, artifacts)
    created = [] if dry_run else [p for p in (dst, *dst.parents) if not p.exists()]
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)
    try:
        with dx.collect() as collected:
            report = convert()
    except BaseException:
        # A half-written case must never be mistaken for a complete one, nor
        # refuse a re-run.
        if not dry_run:
            clear_dst_contents(dst, artifacts)
            _remove_empty_dirs(created)
        for diagnostic in dx.finalize_diagnostics(collected):
            dx.emit(diagnostic)
        raise
    report.diagnostics = dx.finalize_diagnostics(collected)
    report.warnings = [
        d.summary for d in report.diagnostics if d.severity is dx.Severity.WARNING
    ]
    return report
