"""Export manifest: the record of which premise set and inputs produced outputs.

`PREMISES` is the single source of truth for the v1 premises. Documentation
quotes it rather than restating it, so code and prose cannot drift apart.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from novomodelo_bridge.dessem.exporters.fcf.core.layout import (
    sync_directory,
    write_durably,
)
from novomodelo_bridge.dessem.exporters.fcf.core.premises import PREMISES

_LOG = logging.getLogger(__name__)

_TRACKED_LIBRARIES = ("numpy", "novomodelo-python")


@dataclass(frozen=True)
class ExportManifest:
    """Provenance of a single export run.

    `status` and `failed_step` make the manifest a real audit record for a
    failed run, not only a successful one.
    """

    tool_version: str
    created_at: str
    case_path: str
    revision: str
    settings_snapshot: dict[str, Any]
    premises: tuple[str, ...]
    library_versions: dict[str, str]
    outputs: dict[str, str]
    status: str
    failed_step: str | None


def _library_versions() -> dict[str, str]:
    """Collect versions of tracked libraries."""
    resolved: dict[str, str] = {}
    for name in _TRACKED_LIBRARIES:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = "unknown"
    return resolved


def _get_tool_version() -> str:
    """Get novomodelo-bridge version."""
    try:
        from novomodelo_bridge import __version__

        return __version__
    except (ImportError, AttributeError):
        return "unknown"


def build_export_manifest(
    case_path: Path,
    revision: str,
    settings_snapshot: dict[str, Any],
    outputs: Mapping[str, Path],
    status: str = "ok",
    failed_step: str | None = None,
) -> ExportManifest:
    """Assemble the manifest for a run over `case_path` at `revision`.

    `status`/`failed_step` default to the success case.
    """
    return ExportManifest(
        tool_version=_get_tool_version(),
        created_at=datetime.now(UTC).isoformat(),
        case_path=str(case_path.resolve()),
        revision=revision,
        settings_snapshot=settings_snapshot,
        premises=PREMISES,
        library_versions=_library_versions(),
        outputs={name: str(path) for name, path in outputs.items()},
        status=status,
        failed_step=failed_step,
    )


def write_export_manifest(manifest: ExportManifest, path: Path) -> None:
    """Write the manifest as deterministic, key-sorted JSON, atomically.

    `.partial`-then-rename, the same idiom the binary writers use.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    temporary = path.with_name(path.name + ".partial")
    try:
        write_durably(temporary, payload)
        os.replace(temporary, path)
        sync_directory(path.parent)
        _LOG.info("wrote export manifest %s", path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
