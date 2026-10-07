"""Shared provenance-manifest base class and helpers.

Defines :class:`ProvenanceManifest`, the plain base class supplying
``to_json``/``from_json`` for the two provenance-record dataclasses
(:class:`cobre_bridge.cli.conversion_manifest.ConversionManifest` and
:class:`cobre_bridge.comparators.manifest.ComparisonManifest`), plus the
input-hashing and diagnostics-summarizing helpers both records use
(:func:`hash_input_files`, :func:`summarize_diagnostics`). Track-neutral: this
module imports neither manifest module, so either can depend on it without a
cycle.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

if TYPE_CHECKING:
    from cobre_bridge.core.diagnostics import Diagnostic
    from cobre_bridge.decomp.files import DecompFiles
    from cobre_bridge.newave.files import NewaveFiles

_HASH_CHUNK_BYTES = 8192


class ProvenanceManifest:
    """Shared JSON serialization for a provenance-record dataclass.

    Subclasses are ``@dataclass``es that add their own fields; this base
    supplies only ``to_json``/``from_json``, keyed off each subclass's
    ``_NOT_FOUND_LABEL`` for its ``FileNotFoundError`` message.
    """

    _NOT_FOUND_LABEL: ClassVar[str]

    def to_json(self, path: Path) -> None:
        """Write the manifest to ``path`` as indented JSON.

        Parent directories are created if missing.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> Self:
        """Reconstruct a manifest from a JSON file written by :meth:`to_json`.

        Unknown keys (e.g. from a manifest written by a newer bridge) are
        dropped, and absent optional keys fall back to their dataclass defaults,
        so reading a manifest across bridge versions does not raise. Raises
        :class:`FileNotFoundError` (naming ``path``) when the file is absent.
        """
        if not path.exists():
            raise FileNotFoundError(f"{cls._NOT_FOUND_LABEL} not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def hash_input_files(files: NewaveFiles | DecompFiles) -> list[dict[str, object]]:
    """Hash every ``Path``-valued input file of *files* into a sorted list.

    Introspects :func:`dataclasses.fields` of *files* (the instance, so any
    files dataclass works), skipping the ``directory`` field and any field
    whose value is not a :class:`~pathlib.Path` (which also skips an absent
    optional, stored as ``None``, and a non-path field such as
    :class:`~cobre_bridge.decomp.files.DecompFiles`'s ``revision``). For
    each remaining ``(name, path)`` it produces an entry
    ``{"field", "path", "sha256", "size_bytes"}`` where ``sha256`` is the
    SHA-256 hex digest of the file's raw bytes and ``size_bytes`` is the file
    size.

    A file that cannot be read (:class:`OSError`) is recorded with
    ``sha256=None`` and ``size_bytes=None`` rather than aborting the manifest.
    The returned list is sorted by ``"field"`` for deterministic ordering.
    """
    entries: list[dict[str, object]] = []
    for spec in fields(files):
        if spec.name == "directory":
            continue
        value = getattr(files, spec.name)
        if not isinstance(value, Path):
            continue
        path = value
        sha256, size_bytes = _hash_file(path)
        entries.append(
            {
                "field": spec.name,
                "path": str(path),
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    entries.sort(key=lambda entry: entry["field"])
    return entries


def _hash_file(path: Path) -> tuple[str | None, int | None]:
    """Return the SHA-256 hex digest and byte size of *path*.

    The size is read from the open descriptor (``os.fstat``) so it is consistent
    with the bytes just hashed even if the file is replaced afterwards. On
    :class:`OSError` (unreadable / missing file) returns ``(None, None)`` so the
    caller records the failure instead of raising.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
            size_bytes = os.fstat(handle.fileno()).st_size
    except OSError:
        return None, None
    return digest.hexdigest(), size_bytes


def summarize_diagnostics(diagnostics: list[Diagnostic]) -> dict[str, int]:
    """Count *diagnostics* by severity value, omitting absent severities.

    Returns a plain ``dict[str, int]`` keyed by ``severity.value`` (``"info"`` /
    ``"warning"`` / ``"error"``); a severity with no diagnostics is absent from
    the result rather than mapped to zero.
    """
    counts = Counter(diagnostic.severity.value for diagnostic in diagnostics)
    return dict(counts)
