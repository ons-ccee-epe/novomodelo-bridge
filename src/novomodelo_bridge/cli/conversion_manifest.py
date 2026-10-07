"""Provenance manifest for a conversion run.

Defines :class:`ConversionManifest`, a provenance record emitted alongside a
converted Cobre case directory so that a downstream agent can know exactly which
bridge version, git state, source-model case directory, and input files produced
a given conversion, plus the entity counts and the diagnostics raised during the
run. It mirrors :mod:`cobre_bridge.comparators.manifest`; both subclass
:class:`cobre_bridge.core.provenance.ProvenanceManifest` for their shared
``to_json``/``from_json`` behaviour.

The shared :func:`cobre_bridge.core.git.git_sha` runs the git subprocess only
inside :meth:`ConversionManifest.create`, never at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import cobre_bridge
from cobre_bridge.cobre.compat import MIN_COBRE_VERSION
from cobre_bridge.core.git import git_sha
from cobre_bridge.core.provenance import ProvenanceManifest
from cobre_bridge.ui.console import print_status

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import Console

    from cobre_bridge.core.conversion import ConversionReport
    from cobre_bridge.decomp.files import DecompFiles
    from cobre_bridge.newave.files import NewaveFiles


@dataclass
class ConversionManifest(ProvenanceManifest):
    """Provenance record for a conversion run.

    Carries the originating command, the source-model and output paths, the
    bridge version and git SHA, a UTC timestamp, the converted entity counts,
    the hashed input files, a per-severity diagnostic summary, and the full
    diagnostic list.
    """

    command: str
    source_dir: str
    output_dir: str
    bridge_version: str
    git_sha: str | None
    timestamp: str
    entity_counts: dict[str, int] = field(default_factory=dict)
    input_files: list[dict[str, object]] = field(default_factory=list)
    diagnostics_summary: dict[str, int] = field(default_factory=dict)
    diagnostics: list[dict[str, object]] = field(default_factory=list)
    # Minimum cobre version the converted output requires. ``convert`` sets it on
    # every case (all system entities now carry ``operational_start_date``, added
    # in cobre 0.10.0). Defaults to ``None`` so an older manifest that predates the
    # field round-trips through ``from_json`` unchanged. Appended last so existing
    # positional construction stays unchanged.
    min_cobre_version: str | None = None

    _NOT_FOUND_LABEL: ClassVar[str] = "Conversion manifest"

    @classmethod
    def create(
        cls,
        command: str,
        source_dir: Path,
        output_dir: Path,
        *,
        entity_counts: dict[str, int],
        input_files: list[dict[str, object]],
        diagnostics_summary: dict[str, int],
        diagnostics: list[dict[str, object]],
        min_cobre_version: str | None = None,
    ) -> ConversionManifest:
        """Build a manifest, capturing bridge version, git SHA, and UTC time.

        ``bridge_version`` is taken from :data:`cobre_bridge.__version__`,
        ``timestamp`` from :func:`datetime.now` in UTC (ISO 8601) — the only
        non-deterministic field — and ``git_sha`` from
        :func:`cobre_bridge.core.git.git_sha`. The
        ``source_dir`` / ``output_dir`` paths are stringified via ``str(...)``.
        ``min_cobre_version`` records the minimum cobre version the output
        requires (``None`` only when omitted, e.g. by an older caller). The
        remaining data fields are caller-supplied.
        """
        return cls(
            command=command,
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            bridge_version=cobre_bridge.__version__,
            git_sha=git_sha(),
            timestamp=datetime.now(tz=UTC).isoformat(),
            entity_counts=entity_counts,
            input_files=input_files,
            diagnostics_summary=diagnostics_summary,
            diagnostics=diagnostics,
            min_cobre_version=min_cobre_version,
        )


def _write_conversion_manifest(
    report: ConversionReport,
    src: Path,
    dst: Path,
    *,
    command: str,
    discover: Callable[[Path], NewaveFiles | DecompFiles],
    console: Console,
) -> None:
    """Write the conversion provenance manifest into ``dst`` as JSON.

    Rediscovers the source-model input files via *discover* (each command
    passes its own files-dataclass constructor) to hash, builds a
    :class:`ConversionManifest` labelled with *command* from the bridge
    version/git SHA, the entity counts in *report*, and its diagnostics, then
    writes it to ``dst / "conversion_manifest.json"``.

    Both a discovery failure and a write failure are reported as warnings and
    swallowed — the conversion itself already succeeded, so neither changes the
    exit code.
    """
    from cobre_bridge.core.provenance import (
        hash_input_files,
        summarize_diagnostics,
    )

    try:
        files = discover(src)
    except OSError as exc:
        print_status(
            f"Warning: failed to discover source files for conversion manifest: {exc}",
            console=console,
            style="#F5A623",
        )
        return

    entity_counts = {
        "hydros": report.hydro_count,
        "thermals": report.thermal_count,
        "buses": report.bus_count,
        "lines": report.line_count,
        "stages": report.stage_count,
    }
    # Record the minimum cobre version the output requires. Every converted case
    # now emits a ``training.parallelism.backward_scheduler`` block (cobre 0.12.0+),
    # ``operational_start_date`` on all system entities (cobre 0.10.0+), and a
    # mandatory hydro ``unit_groups`` array with the top-level ``bus_id`` removed
    # (cobre 0.13.0+), so the output is only loadable by cobre >= MIN_COBRE_VERSION.
    manifest = ConversionManifest.create(
        command,
        src,
        dst,
        entity_counts=entity_counts,
        input_files=hash_input_files(files),
        diagnostics_summary=summarize_diagnostics(report.diagnostics),
        diagnostics=[d.to_dict() for d in report.diagnostics],
        min_cobre_version=MIN_COBRE_VERSION,
    )

    path = dst / "conversion_manifest.json"
    try:
        manifest.to_json(path)
    except OSError as exc:
        print_status(
            f"Warning: failed to write conversion manifest: {exc}",
            console=console,
            style="#F5A623",
        )
    else:
        print_status(f"Conversion manifest written to {path}", console=console)
