"""The provenance record of a version comparison, written as ``comparison.json``.

A version comparison has two Novomodelo outputs and no source case, so it has
its own record instead of :class:`~novomodelo_bridge.comparators.manifest.ComparisonManifest`;
``command`` tells the two apart in the shared file name. The facts block is
copied from :class:`~novomodelo_bridge.comparators.version.facts.VersionFacts`,
which stays the single source of every fact the record prints.

The key names, their order and :data:`MANIFEST_VERSION` are persisted: a one-way
contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

import novomodelo_bridge
from novomodelo_bridge.comparators.export import MANIFEST_FILE, write_dataset_files
from novomodelo_bridge.comparators.version.facts import facts_of
from novomodelo_bridge.core.diagnostics import Diagnostic
from novomodelo_bridge.core.git import git_sha
from novomodelo_bridge.core.provenance import ProvenanceManifest, summarize_diagnostics

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from novomodelo_bridge.comparators.dataset import ComparisonDataset

#: The schema version of ``comparison.json`` as this module writes it.
MANIFEST_VERSION: int = 1


@dataclass
class VersionComparisonManifest(ProvenanceManifest):
    """Provenance record of one version comparison, one field per JSON key.

    The sixteen fields from ``baseline_output_dir`` to ``gap_delta_pp`` are those
    of ``VersionFacts``, in its order. None has a default, so
    :meth:`~novomodelo_bridge.core.provenance.ProvenanceManifest.from_json`
    raises :class:`TypeError` on a file that lacks a key instead of filling it.
    """

    manifest_version: int
    command: str
    baseline_output_dir: str
    candidate_output_dir: str
    baseline_version: str | None
    candidate_version: str | None
    baseline_label: str
    candidate_label: str
    only_in_baseline: dict[str, list[str]]
    only_in_candidate: dict[str, list[str]]
    unavailable: list[dict[str, str | None]]
    upper_bound_comparable: bool
    upper_bound_comparability: str
    baseline_upper_bound_kind: str | None
    candidate_upper_bound_kind: str | None
    baseline_gap_percent: float | None
    candidate_gap_percent: float | None
    gap_delta_pp: float | None
    tolerance: float
    bridge_version: str
    git_sha: str | None
    timestamp: str
    artifacts: list[str]
    top_divergences: list[dict[str, object]]
    diagnostics_summary: dict[str, int]
    diagnostics: list[dict[str, object]]

    _NOT_FOUND_LABEL: ClassVar[str] = "Version comparison manifest"

    def to_json(self, path: Path) -> None:
        """Write the manifest to ``path`` as indented JSON, creating parent directories.

        Raises:
            ValueError: a field holds NaN or an infinity; no file is written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(asdict(self), indent=2, allow_nan=False)
        path.write_text(text, encoding="utf-8")


def write_version_artifacts(
    dataset: ComparisonDataset,
    *,
    command: str,
    tolerance: float,
    out_dir: Path,
    formats: Sequence[str],
    diagnostics: Sequence[Diagnostic] = (),
) -> VersionComparisonManifest:
    """Write the per-format artifacts of ``dataset`` and ``comparison.json``.

    The manifest is written for every ``formats`` value, the empty list included.
    Its ``artifacts`` is the sorted basenames, ``comparison.json`` included and
    ``metadata.json`` left out. ``facts_of`` is read first, so a dataset without
    facts leaves ``out_dir`` untouched.

    Args:
        dataset: The dataset ``build_version_dataset`` returned.
        command: Stored verbatim.
        tolerance: Stored verbatim.
        out_dir: Destination directory; created with ``parents=True``.
        formats: Each in ``{"parquet", "json", "csv"}``.
        diagnostics: The run's finalized Diagnostics, in report order.

    Returns:
        The manifest written to ``out_dir / "comparison.json"``.

    Raises:
        TypeError: the dataset holds no facts record.
        ValueError: ``formats`` holds an unknown value, or the record holds a
            non-finite number.
        SchemaError: the dataset violates its column contract.
        OSError: a write failed.
    """
    facts = facts_of(dataset)
    written = write_dataset_files(dataset, out_dir, formats)
    top_divergences = dataset.metadata.get("top_divergences")
    manifest = VersionComparisonManifest(
        manifest_version=MANIFEST_VERSION,
        command=command,
        **facts.to_dict(),
        tolerance=tolerance,
        bridge_version=novomodelo_bridge.__version__,
        git_sha=git_sha(),
        timestamp=datetime.now(tz=UTC).isoformat(),
        artifacts=sorted([*written, MANIFEST_FILE]),
        top_divergences=top_divergences if isinstance(top_divergences, list) else [],
        diagnostics_summary=summarize_diagnostics(diagnostics),
        diagnostics=[diagnostic.to_dict() for diagnostic in diagnostics],
    )
    manifest.to_json(out_dir / MANIFEST_FILE)
    return manifest
