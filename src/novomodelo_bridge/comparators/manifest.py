"""Provenance manifest for comparison artifacts.

Defines :class:`ComparisonManifest`, a provenance record emitted alongside
comparison artifacts so that a divergence-investigation agent can know exactly
which cases, tolerance, tool versions, and git state produced a given set of
artifacts, plus which artifacts were emitted and the headline divergences. It
mirrors :mod:`cobre_bridge.cli.conversion_manifest`; both subclass
:class:`cobre_bridge.core.provenance.ProvenanceManifest` for their shared
``to_json``/``from_json`` behaviour.

:func:`cobre_bridge.comparators.export.write_artifacts` is the sole caller;
this module never runs the git subprocess at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import cobre_bridge
from cobre_bridge.core.git import git_sha
from cobre_bridge.core.provenance import ProvenanceManifest


@dataclass
class ComparisonManifest(ProvenanceManifest):
    """Provenance record for a comparison run.

    Carries the command, case paths, tolerance, tool/git versions, a UTC
    timestamp, the emitted artifact filenames, the headline divergences, the
    hashed source-deck input files, a per-severity diagnostic summary, and the
    full diagnostic list.
    """

    command: str
    source_dir: str
    cobre_output_dir: str
    tolerance: float
    bridge_version: str
    git_sha: str | None
    timestamp: str
    cobre_version: str | None = None
    newave_version: str | None = None
    artifacts: list[str] = field(default_factory=list)
    top_divergences: list[dict[str, object]] = field(default_factory=list)
    input_files: list[dict[str, object]] = field(default_factory=list)
    diagnostics_summary: dict[str, int] = field(default_factory=dict)
    diagnostics: list[dict[str, object]] = field(default_factory=list)

    _NOT_FOUND_LABEL: ClassVar[str] = "Comparison manifest"

    @classmethod
    def create(
        cls,
        command: str,
        source_dir: Path,
        cobre_output_dir: Path,
        tolerance: float,
        *,
        cobre_version: str | None = None,
        newave_version: str | None = None,
        input_files: list[dict[str, object]] | None = None,
        diagnostics_summary: dict[str, int] | None = None,
        diagnostics: list[dict[str, object]] | None = None,
    ) -> ComparisonManifest:
        """Build a manifest, capturing bridge version, git SHA, and UTC time.

        ``bridge_version`` is taken from :data:`cobre_bridge.__version__`,
        ``timestamp`` from :func:`datetime.now` in UTC (ISO 8601), and
        ``git_sha`` from :func:`cobre_bridge.core.git.git_sha`. ``cobre_version`` /
        ``newave_version`` remain caller-supplied (default ``None``); deriving
        them is out of scope for this module. ``input_files`` /
        ``diagnostics_summary`` / ``diagnostics`` default to ``None``, which
        this method normalizes to an empty list/dict, mirroring
        :meth:`cobre_bridge.cli.conversion_manifest.ConversionManifest.create`.
        """
        return cls(
            command=command,
            source_dir=str(source_dir),
            cobre_output_dir=str(cobre_output_dir),
            tolerance=tolerance,
            bridge_version=cobre_bridge.__version__,
            git_sha=git_sha(),
            timestamp=datetime.now(tz=UTC).isoformat(),
            cobre_version=cobre_version,
            newave_version=newave_version,
            input_files=input_files if input_files is not None else [],
            diagnostics_summary=(
                diagnostics_summary if diagnostics_summary is not None else {}
            ),
            diagnostics=diagnostics if diagnostics is not None else [],
        )
