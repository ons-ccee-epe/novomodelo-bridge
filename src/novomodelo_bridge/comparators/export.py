"""Machine-readable artifact export for the ANALYZE layer.

This module wires the canonical :class:`ComparisonDataset` and the
:class:`ComparisonManifest` into a single export entry point,
:func:`write_artifacts`, that emits the artifacts a divergence-investigation
workflow consumes: the tidy ``comparison.parquet`` frame, the per-variable
``summary.parquet`` / ``summary.json`` / ``summary.csv`` projections, the
``top_divergences.json`` side-table, the ``comparison.csv`` tidy export, and the
``comparison.json`` provenance manifest (always written).

It composes existing primitives only: it never recomputes diffs and adds no
renderer dependencies (no import of the ``charts`` package / ``report``), so the HTML
report and console paths are unaffected. It is wired into ``cli/app.py`` via the
shared ``_export_compare_artifacts`` helper used by both ``compare`` subcommands.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

from cobre_bridge.cobre.readers import (
    cobre_software_version,
    read_cobre_training_metadata,
)
from cobre_bridge.comparators.manifest import ComparisonManifest

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from cobre_bridge.comparators.dataset import ComparisonDataset

#: The artifact formats :func:`write_artifacts` knows how to emit.
_VALID_FORMATS: frozenset[str] = frozenset({"parquet", "json", "csv"})

#: Always-written manifest filename (basename).
_MANIFEST_FILE: str = "comparison.json"

#: Per-format artifact basenames.
_TIDY_PARQUET: str = "comparison.parquet"
_SUMMARY_PARQUET: str = "summary.parquet"
_SUMMARY_JSON: str = "summary.json"
_TOP_DIVERGENCES_JSON: str = "top_divergences.json"
_TIDY_CSV: str = "comparison.csv"
_SUMMARY_CSV: str = "summary.csv"


def write_artifacts(
    dataset: ComparisonDataset,
    *,
    command: str,
    source_dir: Path,
    cobre_output_dir: Path,
    tolerance: float,
    out_dir: Path,
    formats: Sequence[str],
    input_files: list[dict[str, object]] | None = None,
    diagnostics: list[dict[str, object]] | None = None,
) -> ComparisonManifest:
    """Emit comparison artifacts for ``dataset`` and return the written manifest.

    Validates ``formats`` against ``{"parquet", "json", "csv"}`` (fail fast),
    creates ``out_dir``, emits the requested per-format artifacts, and always
    writes the provenance manifest to ``out_dir / "comparison.json"``.

    Per-format outputs:

    - ``"parquet"`` → ``comparison.parquet`` + ``summary.parquet`` (and, as a
      side effect of :meth:`ComparisonDataset.to_dir`, ``metadata.json`` so the
      directory round-trips via :meth:`ComparisonDataset.from_dir`).
    - ``"json"`` → ``summary.json`` (the summary frame as records) +
      ``top_divergences.json`` (from ``dataset.metadata["top_divergences"]``).
    - ``"csv"`` → ``comparison.csv`` (tidy) + ``summary.csv``.

    The returned manifest's :attr:`~ComparisonManifest.artifacts` is the sorted
    list of every primary artifact basename, *including* ``comparison.json``
    itself. ``metadata.json`` is a round-trip side effect of
    :meth:`ComparisonDataset.to_dir` (needed by ``from_dir``) and is
    intentionally excluded from the list. The manifest's
    :attr:`~ComparisonManifest.top_divergences` mirrors the dataset metadata.

    Args:
        dataset: The validated canonical dataset to export.
        command: The originating command label (e.g. ``"compare results"``).
        source_dir: The source-model case directory (recorded in the manifest).
        cobre_output_dir: The Cobre output directory (recorded in the manifest
            and probed for the Cobre version).
        tolerance: The comparison tolerance (recorded in the manifest).
        out_dir: Destination directory; created with ``parents=True``.
        formats: The artifact formats to emit; each must be in
            ``{"parquet", "json", "csv"}``.
        input_files: The hashed source-deck input files (mirrors the
            conversion manifest's ``input_files``); omitted (``None``) records
            an empty list.
        diagnostics: The compare run's diagnostics, each as a
            :meth:`~cobre_bridge.core.diagnostics.Diagnostic.to_dict` dict; omitted
            (``None``) records an empty ``diagnostics`` list and an empty
            ``diagnostics_summary``.

    Returns:
        The :class:`ComparisonManifest` written to ``out_dir / "comparison.json"``.

    Raises:
        ValueError: If ``formats`` contains a value outside the allowed set; the
            message names the offending value(s).
        SchemaError: Propagated from :meth:`ComparisonDataset.validate` (invoked
            by :meth:`ComparisonDataset.to_dir`) when the dataset violates its
            column contract.
    """
    unknown = sorted(set(formats) - _VALID_FORMATS)
    if unknown:
        allowed = sorted(_VALID_FORMATS)
        msg = f"unknown export format(s) {unknown}; allowed formats are {allowed}"
        raise ValueError(msg)

    out_dir.mkdir(parents=True, exist_ok=True)
    requested = set(formats)
    written: list[str] = []

    if "parquet" in requested:
        # to_dir validates (fail fast) and writes comparison.parquet +
        # summary.parquet + metadata.json, so the directory round-trips via
        # ComparisonDataset.from_dir.
        dataset.to_dir(out_dir)
        written.append(_TIDY_PARQUET)
        written.append(_SUMMARY_PARQUET)

    if "json" in requested:
        summary_records = dataset.summary.to_dicts()
        (out_dir / _SUMMARY_JSON).write_text(
            json.dumps(summary_records, indent=2), encoding="utf-8"
        )
        written.append(_SUMMARY_JSON)

        top_divergences = dataset.metadata.get("top_divergences", [])
        (out_dir / _TOP_DIVERGENCES_JSON).write_text(
            json.dumps(top_divergences, indent=2), encoding="utf-8"
        )
        written.append(_TOP_DIVERGENCES_JSON)

    if "csv" in requested:
        dataset.tidy.write_csv(out_dir / _TIDY_CSV)
        written.append(_TIDY_CSV)
        dataset.summary.write_csv(out_dir / _SUMMARY_CSV)
        written.append(_SUMMARY_CSV)

    # The manifest always lists itself.
    written.append(_MANIFEST_FILE)

    top_divergences = _coerce_top_divergences(dataset.metadata.get("top_divergences"))
    diagnostics_summary = dict(
        Counter(str(entry.get("severity")) for entry in diagnostics or [])
    )
    manifest = ComparisonManifest.create(
        command,
        source_dir,
        cobre_output_dir,
        tolerance,
        cobre_version=_read_cobre_version(cobre_output_dir),
        input_files=input_files,
        diagnostics_summary=diagnostics_summary,
        diagnostics=diagnostics,
    )
    manifest.artifacts = sorted(written)
    manifest.top_divergences = top_divergences
    manifest.to_json(out_dir / _MANIFEST_FILE)

    return manifest


def _coerce_top_divergences(value: object) -> list[dict[str, object]]:
    """Return the metadata ``top_divergences`` as a list of dicts.

    ``build_results_dataset`` / ``build_bounds_dataset`` always populate
    ``metadata["top_divergences"]`` with a ``list[dict[str, object]]``; this
    guard keeps the manifest field well-typed even when the key is absent or
    holds a non-list (in which case an empty list is used).

    Args:
        value: The raw ``metadata.get("top_divergences")`` value.

    Returns:
        The value when it is a list, else an empty list.
    """
    if isinstance(value, list):
        return value
    return []


def _read_cobre_version(cobre_output_dir: Path) -> str | None:
    """Return the Cobre version recorded in the training metadata, or ``None``.

    Delegates to :func:`cobre_readers.read_cobre_training_metadata` for the
    unified candidate-directory search; keeps its own
    ``(cobre_output_dir) -> str | None`` signature and ``str``-type guard so
    existing callers/tests stay green.

    Args:
        cobre_output_dir: The Cobre output directory handed to the comparator.

    Returns:
        The recorded Cobre version when present and readable, else ``None``.
    """
    return cobre_software_version(read_cobre_training_metadata(cobre_output_dir))
