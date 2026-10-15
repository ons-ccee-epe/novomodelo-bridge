"""Shared conversion-report and output-rollback model for both conversion tracks."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from novomodelo_bridge.core import diagnostics as dx


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


def run_conversion(
    convert: Callable[[], ConversionReport],
    dst: Path,
    artifacts: ClearedArtifacts,
    *,
    dry_run: bool,
) -> ConversionReport:
    """Run one track's *convert* and return its report with every finding in it.

    On a failure, *artifacts* are removed from *dst* unless *dry_run*, the
    below-``ERROR`` findings collected so far are emitted on the caller's sink
    (an early warning such as a tab in a fixed-column file often explains the
    failure), and the exception propagates.
    """
    collector = dx.WarningCollector()
    pkg_logger = logging.getLogger("novomodelo_bridge")
    pkg_logger.addHandler(collector)
    try:
        with dx.collect() as collected:
            report = convert()
    except BaseException:
        # Clear the partial outputs so a half-written case is never mistaken for
        # a complete one and a re-run is not refused as "not empty". Never on a
        # dry run: it wrote nothing, and dst may hold a case the user never
        # asked to clear.
        if not dry_run:
            clear_dst_contents(dst, artifacts)
        # ERROR findings are left out: the exception they raised repeats them.
        for diagnostic in dx.finalize_diagnostics(collected, collector.messages):
            if diagnostic.severity is not dx.Severity.ERROR:
                dx.emit(diagnostic)
        raise
    finally:
        pkg_logger.removeHandler(collector)
    report.diagnostics = dx.finalize_diagnostics(collected, collector.messages)
    report.warnings = [
        d.summary for d in report.diagnostics if d.severity is dx.Severity.WARNING
    ]
    return report
